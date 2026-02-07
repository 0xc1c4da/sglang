import logging
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.distributed import divide
from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.lora.eviction_policy import get_eviction_policy
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.utils import (
    EMBEDDING_NAMES,
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_normalized_target_modules,
    get_stacked_multiply,
    get_target_module_name,
)
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


class LoRAAdapterNotLoadedError(RuntimeError):
    """Raised when a request references a LoRA id not present in CPU adapter cache.

    This is a user-input / protocol error and should be converted into a request abort
    rather than crashing the scheduler.
    """


class EmptySlot:
    """
    Singleton class to represent an empty slot in the memory pool.
    This is used to improve readability by not using special str as a placeholder.
    """

    __slots__ = ()

    def __repr__(self):
        return "|EMPTY|"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance


EMPTY_SLOT = EmptySlot()


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        target_modules: Set[str],
        base_model: torch.nn.Module,
        eviction_policy: str,
        lora_added_tokens_size: int,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_added_tokens_size: int = lora_added_tokens_size
        self.max_lora_rank: int = max_lora_rank
        self.target_modules: Set[str] = target_modules

        # Initialize eviction policy
        self.eviction_policy = get_eviction_policy(eviction_policy)

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # A_buffer contains num_layer number of row-major tensors with shape
        #   (max_loras_per_batch, stacked_num * max_lora_dim, input_dim)
        # B_buffer contains num_layer number of column-major tensors with shape
        #   (stacked_num, max_loras_per_batch, output_dim, max_lora_dim)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        self.lm_head_A_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_B_buffer: Dict[str, torch.Tensor] = {}
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        self.embedding_dim: int = self.base_hf_config.hidden_size

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as `EmptySlot` for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
            EMPTY_SLOT
        ] * self.max_loras_per_batch

        # Cache a representative LoRA-wrapped module per (layer_id, suffix) so we
        # can infer *local* shard dimensions from the actual model modules.
        #
        # This is important for models that override TP for specific submodules
        # (e.g. context-parallel attention duplicating weights), where global
        # `server_args.tp_size` does not match the module's effective tp_size.
        self._lora_module_cache: Dict[Tuple[int, str], BaseLayerWithLoRA] = {}
        # Track whether we already warned about falling back to config-based dims.
        self._warned_shape_fallback: Set[str] = set()

        def _score_candidate(full_name: str, mod: BaseLayerWithLoRA) -> tuple[int, int, str]:
            # Higher is better.
            score = 0
            # Prefer non-expert modules by default (experts can create many duplicates).
            if ".experts." not in full_name:
                score += 10
            # Prefer modules that expose local shard dims on their base layer.
            base = getattr(mod, "base_layer", None)
            suffix = full_name.split(".")[-1]
            if base is not None:
                if suffix in ROW_PARALLELISM_LINEAR_LORA_NAMES:
                    # Row-parallel: A is sliced along input dim; local input size is critical.
                    if getattr(base, "input_size_per_partition", None) is not None:
                        score += 5
                    elif getattr(base, "input_size", None) is not None:
                        score += 1
                else:
                    # Column-parallel: B is sliced along output dim; local output shard is critical.
                    out_part = getattr(base, "output_partition_sizes", None)
                    if out_part is not None and len(out_part) > 0:
                        score += 5
                    elif getattr(base, "output_size", None) is not None:
                        score += 1
            # Tie-breakers: prefer shorter paths (less likely expert) then lexical stability.
            return (score, -len(full_name), full_name)

        best: Dict[
            Tuple[int, str],
            tuple[tuple[int, int, str], BaseLayerWithLoRA],
        ] = {}
        for full_name, mod in base_model.named_modules():
            if not isinstance(mod, BaseLayerWithLoRA):
                continue
            lid = get_layer_id(full_name)
            if lid is None:
                continue
            suffix = full_name.split(".")[-1]
            key = (lid, suffix)
            score = _score_candidate(full_name, mod)
            prev = best.get(key)
            if prev is None or score > prev[0]:
                best[key] = (score, mod)

        self._lora_module_cache = {k: v for k, (_, v) in best.items()}

        self.init_buffers(base_model)

    def _get_layer_module(self, layer_idx: int, module_name: str) -> Optional[BaseLayerWithLoRA]:
        return self._lora_module_cache.get((layer_idx, module_name))

    def can_support(self, config: Union[LoRAConfig, Iterable[LoRAConfig]]) -> bool:
        """
        Check if the memory pool can support the given LoRA adapters.
        """

        def _can_support(config: LoRAConfig) -> bool:
            """
            Check if the memory pool can support a single LoRA adapter.
            """
            if config.r > self.max_lora_rank:
                return False
            if config.lora_added_tokens_size > self.lora_added_tokens_size:
                return False
            target_module_names = get_normalized_target_modules(config.target_modules)
            return target_module_names.issubset(self.target_modules)

        if isinstance(config, LoRAConfig):
            return _can_support(config)
        else:
            return all(_can_support(x) for x in config)

    def get_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        # Prefer inferring the *local* input dim from the actual LoRA-wrapped module.
        mod = self._get_layer_module(layer_idx, module_name)
        if mod is not None and hasattr(mod, "base_layer"):
            base = mod.base_layer
            # Row-parallel linears expose `input_size_per_partition`, which already
            # reflects the module's effective tp_size (may differ from global tp_size).
            input_dim = getattr(base, "input_size_per_partition", None)
            if input_dim is None:
                input_dim = getattr(base, "input_size", None)
        else:
            input_dim = None

        if input_dim is None:
            input_dim, _ = get_hidden_dim(
                module_name, self.base_hf_config, base_model, layer_idx
            )
            # If we couldn't infer local shard dims from the wrapped module, apply
            # a safe TP fallback for known row-parallel modules (A is sliced along input dim).
            if (
                self.tp_size > 1
                and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES
                and isinstance(input_dim, int)
            ):
                if module_name not in self._warned_shape_fallback:
                    logger.warning(
                        "LoRA A buffer shape falling back to TP-divided config dims for %s "
                        "(tp_size=%s). Consider implementing get_hidden_dim() on the model or "
                        "ensuring LoRA module cache contains shard-dimension metadata.",
                        module_name,
                        self.tp_size,
                    )
                    self._warned_shape_fallback.add(module_name)
                input_dim = int(divide(int(input_dim), int(self.tp_size)))
        c = get_stacked_multiply(module_name)
        return (
            self.max_loras_per_batch,
            max_lora_dim * c,
            input_dim,
        )

    def get_embedding_lora_A_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        input_dim, _ = get_hidden_dim(
            module_name, self.base_hf_config, base_model, 0, self.lora_added_tokens_size
        )
        # Have not imp self.tp_size > 1 yet.
        return (
            self.max_loras_per_batch,
            max_lora_dim,
            input_dim,
        )

    def get_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        # Prefer inferring the *local* output dim from the actual LoRA-wrapped module.
        mod = self._get_layer_module(layer_idx, module_name)
        if mod is not None and hasattr(mod, "base_layer"):
            base = mod.base_layer
            # Column-parallel linears expose `output_partition_sizes` for the local shard.
            output_part = getattr(base, "output_partition_sizes", None)
            if output_part is not None and len(output_part) > 0:
                output_dim = int(output_part[0])
            else:
                output_dim = getattr(base, "output_size", None)
        else:
            output_dim = None

        if output_dim is None:
            _, output_dim = get_hidden_dim(
                module_name, self.base_hf_config, base_model, layer_idx
            )
            # If we couldn't infer local shard dims from the wrapped module, apply
            # a safe TP fallback for non-row-parallel modules (B is sliced along output dim).
            if (
                self.tp_size > 1
                and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES
                and isinstance(output_dim, int)
            ):
                key = f"{module_name}:B"
                if key not in self._warned_shape_fallback:
                    logger.warning(
                        "LoRA B buffer shape falling back to TP-divided config dims for %s "
                        "(tp_size=%s). Consider implementing get_hidden_dim() on the model or "
                        "ensuring LoRA module cache contains shard-dimension metadata.",
                        module_name,
                        self.tp_size,
                    )
                    self._warned_shape_fallback.add(key)
                output_dim = int(divide(int(output_dim), int(self.tp_size)))
        return (
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def get_embedding_lora_B_shape(
        self,
        module_name: str,
        base_model: torch.nn.Module,
        max_lora_dim: int,
        layer_idx: int,
    ) -> Tuple[int]:
        _, output_dim = get_hidden_dim(
            module_name, self.base_hf_config, base_model, 0, self.lora_added_tokens_size
        )
        # Have not imp self.tp_size > 1 yet.
        return (
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def init_buffers(self, base_model: torch.nn.Module):
        device = next(base_model.parameters()).device

        def init_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int, int], Tuple[int]],
        ):
            target_modules = target_modules - set(EMBEDDING_NAMES)
            for module_name in target_modules:
                buffer[module_name] = [
                    torch.empty(
                        get_lora_shape_fn(
                            module_name,
                            base_model,
                            self.max_lora_rank,
                            idx,
                        ),
                        dtype=self.dtype,
                        device=device,
                    )
                    for idx in range(self.num_layer)
                ]

        def init_embedding_buffer(
            buffer: Dict[str, torch.Tensor],
            target_modules: Set[str],
            get_lora_shape_fn: Callable[[int], Tuple[int]],
        ):
            target_modules = target_modules & set(EMBEDDING_NAMES)
            for module_name in target_modules:
                buffer[module_name] = torch.empty(
                    get_lora_shape_fn(
                        module_name,
                        base_model,
                        self.max_lora_rank,
                        0,
                    ),
                    dtype=self.dtype,
                    device=device,
                )

        if self.lora_added_tokens_size > 0:
            self.new_embeddings_buffer["input_embeddings"] = torch.empty(
                (
                    self.max_loras_per_batch,
                    self.lora_added_tokens_size,
                    self.embedding_dim,
                ),
                dtype=self.dtype,
                device=device,
            )

        if "embed_tokens" in self.target_modules:
            init_embedding_buffer(
                self.embedding_A_buffer,
                self.target_modules,
                self.get_embedding_lora_A_shape,
            )

            init_embedding_buffer(
                self.embedding_B_buffer,
                self.target_modules,
                self.get_embedding_lora_B_shape,
            )

        if "lm_head" in self.target_modules:
            init_embedding_buffer(
                self.lm_head_A_buffer,
                self.target_modules,
                self.get_embedding_lora_A_shape,
            )

            init_embedding_buffer(
                self.lm_head_B_buffer,
                self.target_modules,
                self.get_embedding_lora_B_shape,
            )

        init_buffer(
            self.A_buffer,
            self.target_modules,
            self.get_lora_A_shape,
        )

        init_buffer(
            self.B_buffer,
            self.target_modules,
            self.get_lora_B_shape,
        )

    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_refs: Dict[str, LoRARef],
        lora_embed_tokens_module: Dict[str, BaseLayerWithLoRA],
        lora_lm_head_module: Dict[str, BaseLayerWithLoRA],
    ):
        def get_available_buffer_slot():
            # 1. Prioritize empty slots
            for buffer_id in range(self.max_loras_per_batch):
                if self.buffer_id_to_uid[buffer_id] == EMPTY_SLOT:
                    return buffer_id

            # 2. Memory pool is full, need to evict using policy
            candidates = set()

            for buffer_id in range(self.max_loras_per_batch):
                uid = self.buffer_id_to_uid[buffer_id]

                # Skip if this adapter is needed by current batch
                if uid in cur_uids:
                    continue

                # Skip if this adapter is pinned
                if uid is not None:
                    lora_ref = lora_refs.get(uid)
                    if lora_ref and lora_ref.pinned:
                        continue

                candidates.add(uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots found. Please ensure the number of active (pinned) loras is less than max_loras_per_batch."
                )

            # Prefer evicting LoRA adapters over the base model (None).
            # Only evict None when the batch consists entirely of LoRA requests
            # and no other adapters can be evicted.
            non_none_candidates = candidates - {None}
            if non_none_candidates:
                # Prioritize evicting actual LoRA adapters
                candidates_to_use = non_none_candidates
            else:
                # Only None is available for eviction (batch is all LoRA requests)
                candidates_to_use = candidates

            # Select victim using eviction policy
            victim_uid = self.eviction_policy.select_victim(candidates_to_use)

            # Evict the selected victim
            victim_buffer_id = self.uid_to_buffer_id[victim_uid]
            self.uid_to_buffer_id.pop(victim_uid)
            self.eviction_policy.remove(victim_uid)
            self.buffer_id_to_uid[victim_buffer_id] = EMPTY_SLOT
            logger.debug(
                f"Evicting LoRA {victim_uid} from buffer slot {victim_buffer_id}."
            )
            return victim_buffer_id

        # Mark all adapters in current batch as used (for LRU tracking)
        for uid in cur_uids:
            self.eviction_policy.mark_used(uid)

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid, None)
                self.load_lora_weight_to_buffer(
                    uid,
                    buffer_id,
                    lora_adapter,
                    lora_modules,
                    lora_embed_tokens_module,
                    lora_lm_head_module,
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: List[Dict[str, BaseLayerWithLoRA]],
        lora_embed_tokens_module: Dict[str, BaseLayerWithLoRA],
        lora_lm_head_module: Dict[str, BaseLayerWithLoRA],
    ):
        def load_lora_weight_tensor(
            buffer_view: torch.Tensor,
            weight: Optional[torch.Tensor],
            *,
            kind: str,
            target_module: str,
            layer_id: int | None,
            source_key: str | None = None,
            pre_slice_shape: tuple[int, ...] | None = None,
            post_slice_shape: tuple[int, ...] | None = None,
        ) -> None:
            if weight is None:
                # If the particular weight is not present in the adapter, we initialize the buffer to zero
                # to avoid contamination from the residual weight of the evicted adapters.
                buffer_view.zero_()
            else:
                if buffer_view.shape != weight.shape:
                    msg = (
                        "LoRA buffer shape mismatch while loading adapter.\n"
                        f"- lora_id={uid}\n"
                        f"- tp_size={self.tp_size} tp_rank={self.tp_rank}\n"
                        f"- layer_id={layer_id}\n"
                        f"- kind={kind}\n"
                        f"- target_module={target_module}\n"
                        f"- source_key={source_key}\n"
                        f"- pre_slice_shape={pre_slice_shape}\n"
                        f"- post_slice_shape={post_slice_shape}\n"
                        f"- buffer_view.shape={tuple(buffer_view.shape)}\n"
                        f"- weight.shape={tuple(weight.shape)}\n"
                        "This usually indicates an inconsistency between TP slicing rules and "
                        "the memory pool buffer sizing (local vs global hidden dimensions)."
                    )
                    raise AssertionError(msg)
                buffer_view.copy_(weight, non_blocking=True)

        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0

            for k in self.embedding_A_buffer.keys():
                self.embedding_A_buffer[k][buffer_id] = 0

            for k in self.lm_head_A_buffer.keys():
                self.lm_head_A_buffer[k][buffer_id] = 0
            return

        if lora_adapter is None:
            raise LoRAAdapterNotLoadedError(
                f"LoRA adapter not loaded for lora_id={uid} (missing in lora_adapters dict)"
            )
        lora_rank = lora_adapter.config.r
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            # Track shapes pre/post TP slicing for better diagnostics.
            pre_shapes_A: Dict[str, tuple[int, ...]] = {}
            pre_shapes_B: Dict[str, tuple[int, ...]] = {}
            post_shapes_A: Dict[str, tuple[int, ...]] = {}
            post_shapes_B: Dict[str, tuple[int, ...]] = {}
            src_key_A: Dict[str, str] = {}
            src_key_B: Dict[str, str] = {}
            # Store weights by exact module base name (full module path), not just suffix.
            # This avoids ambiguous selection when multiple modules share the same suffix
            # (e.g. `...mlp.down_proj` vs `...mlp.shared_experts.down_proj`).
            a_by_module: Dict[str, torch.Tensor] = {}
            b_by_module: Dict[str, torch.Tensor] = {}
            a_key_by_module: Dict[str, str] = {}
            b_key_by_module: Dict[str, str] = {}
            temp_A_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.A_buffer
            }
            temp_B_buffer: Dict[str, Optional[torch.Tensor]] = {
                target_module: None for target_module in self.B_buffer
            }
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    module_base = str(name).split(".lora_A", 1)[0]
                    a_by_module[module_base] = weights
                    a_key_by_module[module_base] = str(name)
                else:
                    module_base = str(name).split(".lora_B", 1)[0]
                    b_by_module[module_base] = weights
                    b_key_by_module[module_base] = str(name)

            cur_layer_modules = lora_modules[layer_id]
            # Choose at most one concrete module per target suffix per layer.
            module_name_by_target: Dict[str, str] = {}
            for module_name in cur_layer_modules.keys():
                target_module = get_target_module_name(module_name, self.target_modules)
                prev = module_name_by_target.get(target_module)
                if prev is not None and prev != module_name:
                    raise AssertionError(
                        "Multiple modules map to the same LoRA target module suffix within a single layer, "
                        "which is unsupported by the current memory pool design.\n"
                        f"- layer_id={layer_id}\n"
                        f"- target_module={target_module}\n"
                        f"- module_1={prev}\n"
                        f"- module_2={module_name}\n"
                        "Consider disabling LoRA for expert-like modules or extending the buffer keying scheme."
                    )
                module_name_by_target[target_module] = module_name

            # Assign per-target weights based on exact module name matches.
            for target_module, module_name in module_name_by_target.items():
                a = a_by_module.get(module_name)
                b = b_by_module.get(module_name)
                if a is not None:
                    temp_A_buffer[target_module] = a
                    pre_shapes_A[target_module] = tuple(int(x) for x in a.shape)
                    src_key_A[target_module] = a_key_by_module.get(module_name, module_name)
                if b is not None:
                    temp_B_buffer[target_module] = b
                    pre_shapes_B[target_module] = tuple(int(x) for x in b.shape)
                    src_key_B[target_module] = b_key_by_module.get(module_name, module_name)

            if self.tp_size > 1:
                for target_module, module_name in module_name_by_target.items():
                    module = cur_layer_modules[module_name]

                    if temp_A_buffer[target_module] is None:
                        # Skip weight slicing if the weight is not present in the adapter
                        continue

                    a_pre = temp_A_buffer[target_module]
                    b_pre = temp_B_buffer[target_module]
                    temp_A_buffer[target_module] = module.slice_lora_a_weights(
                        temp_A_buffer[target_module], self.tp_rank
                    )
                    temp_B_buffer[target_module] = module.slice_lora_b_weights(
                        temp_B_buffer[target_module], self.tp_rank
                    )
                    if a_pre is not None:
                        post_shapes_A[target_module] = tuple(
                            int(x) for x in temp_A_buffer[target_module].shape  # type: ignore[union-attr]
                        )
                    if b_pre is not None and temp_B_buffer[target_module] is not None:
                        post_shapes_B[target_module] = tuple(
                            int(x) for x in temp_B_buffer[target_module].shape
                        )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                target_buffer = self.A_buffer[name][layer_id]
                buffer_view = target_buffer[buffer_id, : lora_rank * c, :]
                load_lora_weight_tensor(
                    buffer_view,
                    weights,
                    kind="A",
                    target_module=name,
                    layer_id=layer_id,
                    source_key=src_key_A.get(name),
                    pre_slice_shape=pre_shapes_A.get(name),
                    post_slice_shape=post_shapes_A.get(name),
                )

            for name, weights in temp_B_buffer.items():
                target_buffer = self.B_buffer[name][layer_id]
                buffer_view = target_buffer[buffer_id, :, :lora_rank]
                load_lora_weight_tensor(
                    buffer_view,
                    weights,
                    kind="B",
                    target_module=name,
                    layer_id=layer_id,
                    source_key=src_key_B.get(name),
                    pre_slice_shape=pre_shapes_B.get(name),
                    post_slice_shape=post_shapes_B.get(name),
                )

        if lora_adapter.embedding_layers:

            org_vocab_size = self.base_hf_config.vocab_size
            lora_added_tokens_size = lora_adapter.config.lora_added_tokens_size
            # Only when LoRA is applied to the embedding layer will it have the extra-token issue that needs to be resolved.
            # Load embeddings weights for extra tokens to buffer
            if lora_adapter.added_tokens_embeddings:
                for name, weights in lora_adapter.added_tokens_embeddings.items():
                    if "input_embeddings" in name:
                        buffer_view = self.new_embeddings_buffer["input_embeddings"][
                            buffer_id, :lora_added_tokens_size
                        ]
                    load_lora_weight_tensor(
                        buffer_view,
                        weights,
                        kind="added_tokens",
                        target_module="added_tokens",
                        layer_id=None,
                        source_key=str(name),
                    )

            # load vocab_emb and lm_head
            for name, weights in lora_adapter.embedding_layers.items():
                target_module = get_target_module_name(name, self.target_modules)
                if (
                    target_module == "embed_tokens"
                    and "embed_tokens" in name
                    and ("lora_embedding_A" in name or "lora_A" in name)
                ):
                    buffer_view = self.embedding_A_buffer[target_module][
                        buffer_id,
                        :lora_rank,
                        : (org_vocab_size + lora_added_tokens_size),
                    ]
                    load_lora_weight_tensor(
                        buffer_view,
                        weights,
                        kind="embed_tokens_A",
                        target_module=target_module,
                        layer_id=None,
                        source_key=str(name),
                    )
                elif (
                    target_module == "embed_tokens"
                    and "embed_tokens" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    lora_b_weights = weights
                    # [to-do] support TP
                    # if self.tp_size > 1:
                    #     cur_module = lora_embeddings_modules[target_module]
                    #     for module_name, module in cur_module:
                    #         lora_b_weights = module.slice_lora_b_weights(
                    #             lora_b_weights, self.tp_rank
                    #         )

                    buffer_view = self.embedding_B_buffer[target_module][
                        buffer_id, :, :lora_rank
                    ]
                    load_lora_weight_tensor(
                        buffer_view,
                        lora_b_weights,
                        kind="embed_tokens_B",
                        target_module=target_module,
                        layer_id=None,
                        source_key=str(name),
                    )

                elif (
                    target_module == "lm_head"
                    and "lm_head" in name
                    and ("lora_embedding_A" in name or "lora_A" in name)
                ):
                    buffer_view = self.lm_head_A_buffer[target_module][
                        # buffer_id, :, :lora_rank
                        buffer_id,
                        :lora_rank,
                        :,
                    ]
                    load_lora_weight_tensor(
                        buffer_view,
                        weights,
                        kind="lm_head_A",
                        target_module=target_module,
                        layer_id=None,
                        source_key=str(name),
                    )
                elif (
                    target_module == "lm_head"
                    and "lm_head" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    lora_b_weights = weights
                    # [to-do] support TP
                    # if self.tp_size > 1:
                    #     cur_module = lora_embeddings_modules[target_module]
                    #     for module_name, module in cur_module:
                    #         lora_b_weights = module.slice_lora_b_weights(
                    #             lora_b_weights, self.tp_rank
                    #         )

                    buffer_view = self.lm_head_B_buffer[target_module][
                        # buffer_id, :lora_rank, : org_vocab_size + extra_vocab_size
                        buffer_id,
                        : (org_vocab_size + self.lora_added_tokens_size),
                        :lora_rank,
                    ]
                    load_lora_weight_tensor(buffer_view, lora_b_weights)

    def get_embedding_tensor(
        self, target_module: str, lora_type: LoRAType
    ) -> Optional[torch.Tensor]:
        """
        Get LoRA tensor for non-layer modules (embed_tokens, lm_head).

        Args:
            target_module: Module name, either "embed_tokens" or "lm_head"
            lora_type: Either LoRAType.LORA_A or LoRAType.LORA_B

        Returns:
            The corresponding buffer tensor, or None if not available
        """

        if target_module == "added_tokens":
            if (
                self.lora_added_tokens_size is not None
                and self.lora_added_tokens_size > 0
            ):
                return self.new_embeddings_buffer["input_embeddings"]
            return None
        elif target_module == "embed_tokens":
            if lora_type == LoRAType.LORA_A:
                return self.embedding_A_buffer[target_module]
            return self.embedding_B_buffer[target_module]
        elif target_module == "lm_head":
            if lora_type == LoRAType.LORA_A:
                return self.lm_head_A_buffer[target_module]
            return self.lm_head_B_buffer[target_module]

        raise ValueError(
            f"Invalid target_module '{target_module}'. "
            f"Expected 'embed_tokens' or 'lm_head'."
        )

    def get_tensor(
        self, target_module: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:

        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[target_module][layer_id]

        return self.B_buffer[target_module][layer_id]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
