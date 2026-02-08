"""Small utilities for Heretic integrations.

Keep this module dependency-light: no torch, numpy, or model imports.
"""

from __future__ import annotations

import re

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def heretic_parse_layer_expert(weight_name: str) -> tuple[int | None, int | None]:
    """Parse (layer, expert_id) from a named parameter path.

    Examples:
    - "...layers.12.self_attn.o_proj.weight" -> (12, None)
    - "...layers.12.mlp.experts.7.down_proj.weight" -> (12, 7)
    """
    layer = None
    expert_id = None
    m = _LAYER_RE.search(weight_name)
    if m:
        layer = int(m.group(1))
    m = _EXPERT_RE.search(weight_name)
    if m:
        expert_id = int(m.group(1))
    return layer, expert_id

