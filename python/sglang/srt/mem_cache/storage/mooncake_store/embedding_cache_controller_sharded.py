"""
Sharded loading extension for EmbeddingCacheController.

This module provides optimized distributed loading that reduces Mooncake
network traffic by having each TP rank load only a shard of the data,
then exchange via all_gather.
"""

import hashlib
import logging
import struct
import time
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _deterministic_hash(s: str) -> int:
    """
    Compute a deterministic hash for a string.

    WARNING: Do NOT use Python's built-in hash() function for sharding
    because it uses random seeding (PYTHONHASHSEED) which produces
    different results across processes. This causes all_gather to hang
    because different ranks assign the same hash to different ranks.

    We use MD5 which is deterministic and consistent across all ranks.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class ShardedLoadHelper:
    """
    Helper class for sharded loading across TP ranks.

    Each rank loads only 1/TP_SIZE of the data from Mooncake,
    then exchanges via all_gather to get full data.
    """

    def __init__(self, tp_rank: int, tp_size: int, tp_group):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group

    def shard_hashes(self, hashes: List[str]) -> List[str]:
        """
        Deterministically assign hashes to ranks using hash-seeded round-robin.

        Sort hashes for deterministic ordering, then use a hash-derived offset
        to shift the round-robin start. This guarantees:
        - Even distribution within a request (round-robin)
        - Different requests spread across different ranks (hash-seeded offset)
        - All ranks agree on the assignment (same sorted order + same offset)
        """
        sorted_hashes = sorted(hashes)
        offset = (
            _deterministic_hash(sorted_hashes[0]) % self.tp_size if sorted_hashes else 0
        )
        my_set = set()
        for i, h in enumerate(sorted_hashes):
            if (i + offset) % self.tp_size == self.tp_rank:
                my_set.add(h)
        return [h for h in hashes if h in my_set]

    def get_shard_assignment(self, all_hashes: List[str]) -> Dict[int, List[str]]:
        """Get the assignment of hashes to each rank."""
        sorted_hashes = sorted(all_hashes)
        offset = (
            _deterministic_hash(sorted_hashes[0]) % self.tp_size if sorted_hashes else 0
        )
        assignment = {i: [] for i in range(self.tp_size)}
        for i, h in enumerate(sorted_hashes):
            assignment[(i + offset) % self.tp_size].append(h)
        return assignment

    def serialize_embeddings(
        self,
        embeddings: Dict[str, torch.Tensor],
        expected_hashes: List[str],
    ) -> torch.Tensor:
        """
        Serialize embeddings to a flat byte tensor for all_gather.

        Format for each embedding:
        - hash_len (4 bytes, int32)
        - hash_bytes (variable)
        - num_tokens (4 bytes, int32)
        - dim (4 bytes, int32)
        - data (num_tokens * dim * 4 bytes, float32)

        Missing embeddings are represented with num_tokens=0.
        """
        buffers = []

        for h in expected_hashes:
            if h in embeddings and embeddings[h] is not None:
                emb = embeddings[h]
                num_tokens, dim = emb.shape
                # Convert to float32 bytes
                emb_bytes = emb.cpu().numpy().astype(np.float32).tobytes()

                # Pack header
                hash_bytes = h.encode("utf-8")
                header = struct.pack(
                    "<iii",  # little-endian: hash_len, num_tokens, dim
                    len(hash_bytes),
                    num_tokens,
                    dim,
                )
                buffers.append(header + hash_bytes + emb_bytes)
            else:
                # Missing embedding: mark with num_tokens=-1
                hash_bytes = h.encode("utf-8")
                header = struct.pack("<iii", len(hash_bytes), -1, 0)
                buffers.append(header + hash_bytes)

        if not buffers:
            return torch.empty(0, dtype=torch.uint8)

        # Concatenate all buffers
        data = b"".join(buffers)
        return torch.frombuffer(data, dtype=torch.uint8).clone()

    def deserialize_embeddings(
        self,
        data: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Deserialize embeddings from flat byte tensor.

        Args:
            data: byte tensor (CPU)
            device: target device, if None stays on CPU
        """
        if data.numel() == 0:
            return {}

        embeddings = {}
        offset = 0
        data_bytes = data.numpy().tobytes()
        total_len = len(data_bytes)

        while offset < total_len:
            # Check if we have enough bytes for header
            if offset + 12 > total_len:
                logger.warning(f"Not enough bytes for header at offset {offset}")
                break

            # Read header
            hash_len, num_tokens, dim = struct.unpack_from("<iii", data_bytes, offset)
            offset += 12

            # Validate header values
            if hash_len <= 0 or hash_len > 1024:  # Reasonable limit for hash string
                logger.warning(f"Invalid hash_len: {hash_len}")
                break

            # Check if we have enough bytes for hash
            if offset + hash_len > total_len:
                logger.warning(f"Not enough bytes for hash at offset {offset}")
                break

            # Read hash
            hash_bytes = data_bytes[offset : offset + hash_len]
            try:
                h = hash_bytes.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode hash bytes at offset {offset}")
                break
            offset += hash_len

            if num_tokens < 0:
                # Missing embedding (marked as -1)
                continue

            # Validate dimensions with reasonable limits
            MAX_TOKENS = 100000  # ~100K tokens max per image
            MAX_DIM = 100000  # ~100K dim max
            if num_tokens <= 0 or dim <= 0 or num_tokens > MAX_TOKENS or dim > MAX_DIM:
                logger.warning(
                    f"Invalid dimensions: num_tokens={num_tokens}, dim={dim}"
                )
                continue

            # Read embedding data
            data_size = num_tokens * dim * 4  # float32 = 4 bytes
            if offset + data_size > total_len:
                logger.warning(
                    f"Not enough bytes for embedding data at offset {offset}"
                )
                break

            emb_bytes = data_bytes[offset : offset + data_size]
            offset += data_size

            # Convert back to tensor
            try:
                emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to create numpy array: {e}")
                continue

            if emb_array.size != num_tokens * dim:
                logger.warning(
                    f"Embedding array size mismatch: expected {num_tokens * dim}, got {emb_array.size}"
                )
                continue

            try:
                emb_tensor = torch.from_numpy(emb_array).view(num_tokens, dim)
                if device is not None:
                    emb_tensor = emb_tensor.to(device)
            except Exception as e:
                logger.warning(f"Failed to create tensor view: {e}")
                continue

            embeddings[h] = emb_tensor

        return embeddings

    def all_gather_embeddings(
        self,
        local_embeddings: Dict[str, torch.Tensor],
        all_hashes: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        All-gather embeddings using tensor gather.

        Packs embeddings into flat tensor with metadata, all_gathers,
        then unpacks. Keeps tensors on GPU throughout.
        """
        if self.tp_group is None or self.tp_size <= 1:
            return local_embeddings

        device = torch.cuda.current_device()

        # Build metadata: [(hash, num_tokens, dim), ...]
        local_hashes = list(local_embeddings.keys())
        local_tensors = []
        local_metadata = []  # (hash, num_tokens, dim)
        offset = 0

        for h in local_hashes:
            emb = local_embeddings[h]
            num_tokens, dim = emb.shape
            local_tensors.append(emb.flatten().to(device))
            local_metadata.append((h, num_tokens, dim))
            offset += num_tokens * dim

        # Pack into flat tensor
        if local_tensors:
            local_flat = torch.cat(local_tensors)
        else:
            local_flat = torch.empty(0, dtype=torch.float32, device=device)

        # All-gather flat tensor sizes
        local_size = torch.tensor(
            [local_flat.numel()], dtype=torch.int64, device=device
        )
        sizes = [
            torch.empty(1, dtype=torch.int64, device=device)
            for _ in range(self.tp_size)
        ]
        torch.distributed.all_gather(sizes, local_size, group=self.tp_group)
        max_size = max(s.item() for s in sizes)

        # Pad to max_size
        if local_flat.numel() < max_size:
            padding = torch.zeros(
                max_size - local_flat.numel(), dtype=torch.float32, device=device
            )
            local_flat = torch.cat([local_flat, padding])

        # All-gather flat tensors
        gathered = [
            torch.empty(max_size, dtype=torch.float32, device=device)
            for _ in range(self.tp_size)
        ]
        torch.distributed.all_gather(gathered, local_flat, group=self.tp_group)

        # All-gather metadata (hash strings + shapes)
        # Serialize metadata: hash bytes + num_tokens + dim
        def serialize_meta(metadata_list):
            buffers = []
            for h, num_tokens, dim in metadata_list:
                h_bytes = h.encode("utf-8")
                header = struct.pack("<iii", len(h_bytes), num_tokens, dim)
                buffers.append(header + h_bytes)
            if buffers:
                return b"".join(buffers)
            return b""

        local_meta_bytes = serialize_meta(local_metadata)
        if local_meta_bytes:
            local_meta_tensor = torch.from_numpy(
                np.frombuffer(local_meta_bytes, dtype=np.uint8)
            ).to(device)
        else:
            local_meta_tensor = torch.empty(0, dtype=torch.uint8, device=device)

        local_meta_size = torch.tensor(
            [local_meta_tensor.numel()], dtype=torch.int64, device=device
        )
        meta_sizes = [
            torch.empty(1, dtype=torch.int64, device=device)
            for _ in range(self.tp_size)
        ]
        torch.distributed.all_gather(meta_sizes, local_meta_size, group=self.tp_group)
        max_meta_size = max(s.item() for s in meta_sizes)

        if local_meta_tensor.numel() < max_meta_size:
            padding = torch.zeros(
                max_meta_size - local_meta_tensor.numel(),
                dtype=torch.uint8,
                device=device,
            )
            local_meta_tensor = torch.cat([local_meta_tensor, padding])

        meta_gathered = [
            torch.empty(max_meta_size, dtype=torch.uint8, device=device)
            for _ in range(self.tp_size)
        ]
        torch.distributed.all_gather(
            meta_gathered, local_meta_tensor, group=self.tp_group
        )

        # Unpack embeddings
        all_embeddings = {}
        for rank_idx, rank_data in enumerate(gathered):
            actual_size = sizes[rank_idx].item()
            if actual_size == 0:
                continue

            # Parse metadata
            meta_bytes = meta_gathered[rank_idx].cpu().numpy().tobytes()
            meta_len = meta_sizes[rank_idx].item()
            meta_parsed = []
            offset = 0
            while offset + 12 <= meta_len:
                if offset + 12 > meta_len:
                    break
                hash_len, num_tokens, dim = struct.unpack_from(
                    "<iii", meta_bytes, offset
                )
                offset += 12
                if offset + hash_len > meta_len:
                    break
                h = meta_bytes[offset : offset + hash_len].decode("utf-8")
                offset += hash_len
                meta_parsed.append((h, num_tokens, dim))

            # Unpack flat tensor
            flat = rank_data[:actual_size]
            pos = 0
            for h, num_tokens, dim in meta_parsed:
                emb = flat[pos : pos + num_tokens * dim].view(num_tokens, dim)
                all_embeddings[h] = emb
                pos += num_tokens * dim

        # Fill missing
        for h in all_hashes:
            if h not in all_embeddings:
                all_embeddings[h] = None

        return all_embeddings


def _check_local_prefetch_done(controller, req_id: str) -> bool:
    """Check if local prefetch is done (no all_reduce sync)."""
    with controller.lock:
        if req_id not in controller.ongoing_prefetch:
            return True
        op = controller.ongoing_prefetch[req_id]
        return op.is_finished and op.success


def get_embeddings_distributed_sharded(
    controller,
    image_hashes: List[str],
    expected_tokens: List[int],
    modality: Optional[str] = None,
) -> List[Optional[torch.Tensor]]:
    """
    Distributed loading with sharding optimization.

    Each TP rank loads only 1/TP_SIZE of the data from Mooncake,
    then exchanges via all_gather. This reduces Mooncake network
    traffic by TP_SIZE times.

    CRITICAL: This function ensures all ranks always reach the same
    synchronization points regardless of timeout, preventing hang.

    Args:
        controller: EmbeddingCacheController instance
        image_hashes: List of image hashes to load
        expected_tokens: Expected token count for each image
        modality: Modality type for dimension lookup

    Returns:
        List of embeddings (same order as image_hashes), None for failed loads
    """
    if not image_hashes:
        return []

    tp_size = controller.tp_world_size
    tp_rank = controller.tp_rank
    tp_group = controller.prefetch_tp_group

    helper = ShardedLoadHelper(tp_rank, tp_size, tp_group)

    # 1. Shard: each rank gets assigned hashes
    shard_hashes = helper.shard_hashes(image_hashes)
    shard_hash_set = set(shard_hashes)
    shard_tokens = [
        expected_tokens[i] for i, h in enumerate(image_hashes) if h in shard_hash_set
    ]

    # 2. Each rank only prefetches its assigned shard
    req_id = None
    if shard_hashes:
        req_id = f"sharded_{tp_rank}_{shard_hashes[0][:16]}"
        controller.prefetch(req_id, shard_hashes, shard_tokens, modality)

        # Wait for local prefetch to complete (NO all_reduce here!)
        # Each rank independently waits for its own shard
        max_wait = 30.0  # 30 second timeout
        start_time = time.time()
        local_timeout = False
        while not _check_local_prefetch_done(controller, req_id):
            if time.time() - start_time > max_wait:
                logger.warning(f"[Rank {tp_rank}] Timeout waiting for shard prefetch")
                local_timeout = True
                break
            time.sleep(0.002)

    # 3. Read local embeddings from cpu_pool
    local_embeddings = {}
    for h in shard_hashes:
        emb = controller._get_embedding_from_local(h)
        if emb is not None:
            local_embeddings[h] = emb

    total_emb_bytes = 0
    for h, emb in local_embeddings.items():
        emb_bytes = emb.numel() * emb.element_size()
        total_emb_bytes += emb_bytes

    # 4. Synchronization barrier: ALL ranks must reach here before all_gather
    # This ensures no rank is left behind in previous communication
    if tp_group is not None and tp_size > 1:
        # Use barrier to ensure all ranks are ready for all_gather
        # This prevents hang when some ranks timeout earlier than others
        try:
            torch.distributed.barrier(group=tp_group)
        except Exception as e:
            logger.warning(f"[Rank {tp_rank}] Barrier failed: {e}")

    # 5. All-gather embeddings from all ranks
    # This is the ONLY collective communication in sharded loading
    all_embeddings = helper.all_gather_embeddings(local_embeddings, image_hashes)

    # 6. Insert missing embeddings into local cpu_pool for future use
    for h, emb in all_embeddings.items():
        if emb is not None and not controller._has_hash(h):
            controller._insert_embedding(h, emb)

    # 7. Cleanup: mark this rank's prefetch as done (if exists)
    if req_id and req_id in controller.ongoing_prefetch:
        with controller.lock:
            controller.ongoing_prefetch.pop(req_id, None)

    # 8. Return in original order
    return [all_embeddings.get(h) for h in image_hashes]
