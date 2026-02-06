from __future__ import annotations

import hashlib
import struct


def sha256_token_ids_le_u32(token_ids: list[int]) -> str:
    """Stable hash for prompt identity (little-endian uint32 stream)."""
    h = hashlib.sha256()
    for tid in token_ids:
        h.update(struct.pack("<I", int(tid)))
    return h.hexdigest()

