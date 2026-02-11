from __future__ import annotations

"""Context plumbing for Heretic packed-MoE adapters.

SGLang's standard LoRA mechanism uses `ForwardBatch.lora_ids` to apply adapters to supported
Linear modules. Packed/fused MoE expert weights (e.g. `FusedMoE.w2_weight`) are not part of
that LoRA stack, so we implement a separate packed-adapter mechanism that still uses the same
per-sequence `lora_ids` selection.

MoE runner cores do not receive `ForwardBatch` directly. This module provides a lightweight,
thread/task-local context (via contextvars) that makes per-token sequence indices and per-sequence
adapter ids available to MoE runner injection code.
"""

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class HereticPackedMoEContext:
    # For a flattened token batch (M tokens), map token index -> sequence index [0..bs-1].
    # Shape: [M] (int32 or int64), device matches model device.
    token_to_seq: torch.Tensor
    # Per-sequence lora id string list as provided by ForwardBatch.lora_ids (len == bs).
    # Entries may be None.
    seq_lora_ids: list[Optional[str]]


_CTX: contextvars.ContextVar[Optional[HereticPackedMoEContext]] = contextvars.ContextVar(
    "heretic_packed_moe_ctx", default=None
)


def get_ctx() -> Optional[HereticPackedMoEContext]:
    return _CTX.get()


@contextlib.contextmanager
def set_ctx(ctx: Optional[HereticPackedMoEContext]):
    token = _CTX.set(ctx)
    try:
        yield
    finally:
        _CTX.reset(token)

