# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

import datetime
import gc
import inspect
import json
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.heretic_utils import heretic_parse_layer_expert
from sglang.srt.configs import (
    FalconH1Config,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import AttentionArch, ModelConfig, ModelImpl
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.input_buffers import GraphInputBuffers
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    MultiprocessingSerializer,
    cpu_has_amx_support,
    dynamic_import,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_cpu_ids_by_node,
    get_local_ip_auto,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
    "nsa",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutlass_mla",
    "trtllm_mla",
]

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)


def resolve_language_model(model: nn.Module) -> nn.Module:
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    return model.model


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_size = server_args.dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None
        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            self.eagle_use_aux_hidden_state = True

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                self.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                self.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                self.eagle_aux_hidden_state_layer_ids = None

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # Initialize the model runner
        self.initialize(min_per_gpu_memory)
        self.check_quantized_moe_compatibility()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        (
            ElasticEPStateManager.init(self.server_args)
            if self.server_args.elastic_ep_backend
            else None
        )
        # Load the model
        self.sampler = create_sampler()
        self.load_model()

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Consider PP, so use start_layer and end_layer.
        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        # Update back to model_config.
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()

        # Init Double Sparsity
        if server_args.enable_double_sparsity:
            if server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(server_args.ds_heavy_channel_type)

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        self.configure_kv_cache_dtype()

        # Init memory pool and attention backends
        self.init_memory_pool(min_per_gpu_memory)

        # Init max running requests
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.req_to_token_pool.size,
        )

        # Init routed experts capturer
        self.init_routed_experts_capturer()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            self.init_attention_backend()
            self.kernel_warmup()
            self.init_device_graphs()
        elif self.device in ["npu", "cpu"]:
            self.init_attention_backend()
            self.init_device_graphs()
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if server_args.forward_hooks:
            register_forward_hooks(self.model, server_args.forward_hooks)

        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )

        # Initialize piecewise CUDA graph
        self.init_piecewise_cuda_graphs()

        self.prealloc_symmetric_memory_pool()

    def init_routed_experts_capturer(self):
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip, "P2PHANDSHAKE", "rdma", envs.MOONCAKE_DEVICE.get()
        )
        self.remote_instance_transfer_engine_session_id = (
            f"{local_ip}:{self.remote_instance_transfer_engine.get_rpc_port()}"
        )

    def model_specific_adjustment(self):
        server_args = self.server_args

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0:
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        if self.device == "cuda":
            if self.server_args.elastic_ep_backend == "mooncake":
                backend = "mooncake"
                if self.server_args.mooncake_ib_device:
                    mooncake_ib_device = self.server_args.mooncake_ib_device.split(",")
                    try:
                        from mooncake import ep as mooncake_ep

                        mooncake_ep.set_device_filter(mooncake_ib_device)
                    except:
                        pass  # A warning will be raised in `init_distributed_environment`
            else:
                backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"
        elif self.device == "npu":
            backend = "hccl"
        elif self.device == "musa":
            backend = "mccl"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    def _(data, dim):
                        return torch.cat([data] * self.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = socket.gethostbyname(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        monkey_patch_vllm_parallel_state(reverse=True)

        get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            self.pyt_hooks = PytHooks()
            self.pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )
        if self.server_args.debug_tensor_dump_output_folder is not None:
            register_forward_hook_for_model(
                self.model,
                self.server_args.debug_tensor_dump_output_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        if ElasticEPStateManager.instance() is not None:
            # TODO: refactor the weights update when elastic ep
            old_expert_location_metadata = get_global_expert_location_metadata()
            assert old_expert_location_metadata is not None
            old_expert_location_metadata.update(
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )
            self.update_weights_from_disk(
                self.server_args.model_path,
                self.server_args.load_format,
                lambda name: "mlp.experts" in name and "mlp.shared_experts" not in name,
            )
        else:
            self.expert_location_updater.update(
                self.model.routed_experts_weights_of_layer,
                new_expert_location_metadata,
                update_layer_ids=update_layer_ids,
                nnodes=self.server_args.nnodes,
                rank=self.tp_rank,
            )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (self.device == "cuda" or self.device == "musa"):
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{group_port}",
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = (
                f"Succeeded to init group through {master_address}:{group_port} group."
            )
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        torch.cuda.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        torch.cuda.empty_cache()
        success = False
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {master_address}:{group_port} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        torch.cuda.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        self.device_module = torch.get_device_module(self.device)
        infered_device = self.device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def heretic_module_map(
        self,
        include_projs: Optional[list[str]] = None,
        include_layers: Optional[list[int]] = None,
        include_experts: Optional[list[int]] = None,
        max_experts_per_layer: Optional[int] = None,
        expert_strategy: str = "first",
    ) -> list[dict]:
        """List canonical parameter paths for ablation/LoRA targeting (Heretic extension).

        The returned `module_path` values match `model.named_parameters()` keys (e.g.
        `model.layers.0.self_attn.o_proj.weight`).
        """
        # Default to common LLM projection names.
        #
        # Note: SGLang normalizes q/k/v and gate/up into stacked module names
        # (`qkv_proj`, `gate_up_proj`). Some models only expose the stacked form,
        # so include both to keep this endpoint backend-agnostic.
        default_projs = {
            "q_proj",
            "k_proj",
            "v_proj",
            "qkv_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "gate_up_proj",
            "down_proj",
        }
        allowed = set(include_projs) if include_projs else default_projs
        allowed_layers = set(include_layers) if include_layers is not None else None
        allowed_experts = set(include_experts) if include_experts is not None else None

        modules: list[dict] = []

        # Prefer module-derived logical shapes over parameter tensor shapes.
        # Some quantization/packing methods expose `.weight` as a placeholder tensor
        # (e.g. shape (out, 0)), while the logical (out_features, in_features) is
        # defined by the module's input/output sizes. LoRA weight construction and
        # probe vector sizing should follow the logical dimensions.
        try:
            params = dict(self.model.named_parameters())
        except Exception:
            params = {}

        for module_name, module in self.model.named_modules():
            weight_name = f"{module_name}.weight"
            if weight_name not in params:
                continue

            proj = module_name.split(".")[-1]
            # Normalize projection leaf names across architectures.
            #
            # Heretic's directional ablation currently targets projections whose output lives in
            # residual space (hidden_size). For many MoE implementations, the expert down-projection
            # is not literally named "down_proj"; common variants include:
            # - Phi-3.5-MoE: expert.w2
            # - Granite MoE Hybrid: expert.output_linear
            #
            # Treat these as logical "down_proj" so clients can request include_projs=["down_proj"]
            # and still receive the expert weights.
            if proj in ("w2", "output_linear"):
                proj = "down_proj"
            if proj not in allowed:
                continue

            layer, expert_id = heretic_parse_layer_expert(weight_name)

            if allowed_layers is not None:
                if layer is None or layer not in allowed_layers:
                    continue

            if expert_id is not None and allowed_experts is not None:
                if expert_id not in allowed_experts:
                    continue

            # Unwrap LoRA wrapper modules (e.g. RowParallelLinearWithLoRA) to recover
            # the *logical/global* input/output sizes from the underlying base layer.
            # Without this, TP-sharded placeholder weight shapes (e.g. (out, 0) or
            # (out, in_local)) can leak into the module_map and cause clients to
            # construct already-sharded LoRA tensors, which SGLang will then slice
            # again per TP rank -> empty tensors on ranks > 0.
            logical_module = module
            for _ in range(4):
                base = getattr(logical_module, "base_layer", None)
                if base is None:
                    break
                logical_module = base

            out_f = getattr(logical_module, "output_size", None)
            in_f = getattr(logical_module, "input_size", None)
            shape = None
            if isinstance(out_f, int) and isinstance(in_f, int):
                out_f = int(out_f)
                in_f = int(in_f)
                if out_f > 0 and in_f > 0:
                    shape = [out_f, in_f]
                else:
                    out_f = None
                    in_f = None

            # Fallback: infer from parameter tensor shape (best-effort).
            if shape is None:
                try:
                    p = params[weight_name]
                    shp = tuple(int(x) for x in p.shape)
                    if len(shp) >= 2 and shp[0] > 0 and shp[1] > 0:
                        out_f = int(shp[0])
                        in_f = int(shp[1]) if len(shp) == 2 else int(
                            shp[1] * int(__import__("math").prod(shp[2:]))
                        )
                        shape = [out_f, in_f]
                except Exception:
                    pass

            if shape is None:
                continue

            modules.append(
                {
                    "module_path": weight_name,
                    "kind": "module_weight",
                    "layer": layer,
                    "expert_id": expert_id,
                    "proj": proj,
                    "shape": shape,
                    "out_features": out_f,
                    "in_features": in_f,
                }
            )

        # Packed/Fused MoE expert weights (e.g. `FusedMoE.w2_weight`) are not exposed as
        # `<module>.weight` parameters and therefore are invisible to the loop above.
        #
        # For Kimi/DeepSeek-style fused MoE, the routed experts' down-projection weights live in a
        # single 3D tensor `w2_weight` with shape roughly:
        #   [num_local_experts, out_features(hidden_size), in_features(intermediate_per_partition)]
        #
        # Expose these as truthful packed targets so Heretic can build/apply directional updates.
        if "down_proj" in allowed:
            # If include_experts is explicitly an empty list, the caller wants to exclude expert weights.
            exclude_experts = allowed_experts is not None and len(allowed_experts) == 0
        else:
            exclude_experts = True

        if not exclude_experts:
            for module_name, module in self.model.named_modules():
                # Some quantization methods register alternative packed weight names.
                # Prefer the floating-point `w2_weight` when present (FULL builder requires float),
                # but still expose non-float variants so callers can surface actionable errors
                # (e.g., RAWINT4 / INT4 weights cannot be used by the fp32 FULL builder).
                w2_name = None
                for cand in (
                    "w2_weight",
                    "w2_weight_packed",
                    "w2_qweight",
                    "w2_qweight_packed",
                ):
                    n = f"{module_name}.{cand}"
                    if n in params:
                        w2_name = n
                        break
                if w2_name is None:
                    continue

                # Duck-type: ensure this looks like a fused MoE container.
                w2 = params[w2_name]
                if not (hasattr(module, "num_local_experts") or hasattr(module, "num_experts")):
                    # Some implementations expose the expert count only via the weight tensor.
                    pass
                try:
                    layer, _ = heretic_parse_layer_expert(w2_name)
                except Exception:
                    layer = None
                if allowed_layers is not None:
                    if layer is None or layer not in allowed_layers:
                        continue

                # Expect (E, out, in) or transposed variants; we report the logical expert slice shape.
                shp = tuple(int(x) for x in w2.shape)
                if len(shp) != 3:
                    continue
                num_local_experts = int(shp[0])

                # Logical dims for packed/quantized variants:
                # - Float `w2_weight` is typically [E, out(hidden), in(intermediate)].
                # - Int4-packed variants often store [E, in_packed, out] or [E, out, in_packed].
                # Use module metadata when available to report truthful logical (out,in).
                hidden_size = getattr(module, "hidden_size", None)
                intermediate = getattr(module, "intermediate_size_per_partition", None)
                out_features = None
                in_features = None
                packed_factor = None
                packed_dim = None
                if isinstance(hidden_size, int) and isinstance(intermediate, int):
                    hidden_size = int(hidden_size)
                    intermediate = int(intermediate)
                    out_features = hidden_size
                    in_features = intermediate
                    # Infer packing factor/dimension for common int4 layouts.
                    if str(w2_name).endswith("_packed") or "qweight" in str(w2_name):
                        # If one dim matches hidden, the other is packed intermediate.
                        if shp[2] == hidden_size and shp[1] != hidden_size:
                            # [E, in_packed, out]
                            packed_dim = 1
                            packed_factor = int(round(intermediate / max(1, shp[1])))
                        elif shp[1] == hidden_size and shp[2] != hidden_size:
                            # [E, out, in_packed]
                            packed_dim = 2
                            packed_factor = int(round(intermediate / max(1, shp[2])))
                # Fallback to shape-based inference.
                if out_features is None or in_features is None:
                    out_features = int(shp[1])
                    in_features = int(shp[2])
                modules.append(
                    {
                        "module_path": w2_name,
                        "kind": "moe_packed_w2",
                        "layer": layer,
                        "expert_id": None,
                        "proj": "down_proj",
                        "expert_id_space": "local",
                        "num_local_experts": num_local_experts,
                        "shape": [int(out_features), int(in_features)],
                        "out_features": int(out_features),
                        "in_features": int(in_features),
                        "storage_shape": list(shp),
                        "storage_layout": "E_out_in",
                        "storage_dtype": str(getattr(w2, "dtype", None)),
                        "packed_factor": int(packed_factor) if packed_factor is not None else None,
                        "packed_dim": int(packed_dim) if packed_dim is not None else None,
                    }
                )

        # Optionally cap number of experts per (layer, proj) to avoid MoE explosions.
        if max_experts_per_layer is not None:
            cap = int(max_experts_per_layer)
            if cap <= 0:
                # Drop all expert weights.
                modules = [m for m in modules if m.get("expert_id") is None]
            else:
                keep: list[dict] = []
                non_expert = [m for m in modules if m.get("expert_id") is None]
                experts = [m for m in modules if m.get("expert_id") is not None]
                experts_by_key: dict[tuple[int, str], list[dict]] = {}
                for m in experts:
                    layer = m.get("layer")
                    proj = m.get("proj")
                    if not isinstance(layer, int) or not isinstance(proj, str):
                        continue
                    experts_by_key.setdefault((layer, proj), []).append(m)

                for key, group in experts_by_key.items():
                    if expert_strategy not in ("first", "all"):
                        # Keep behavior deterministic and simple; fallback to 'first'.
                        strat = "first"
                    else:
                        strat = expert_strategy
                    if strat == "all":
                        keep.extend(group)
                        continue

                    group.sort(
                        key=lambda d: (
                            int(d.get("expert_id") or 0),
                            str(d.get("module_path") or ""),
                        )
                    )
                    keep.extend(group[:cap])

                modules = non_expert + keep

        def _sort_key(d: dict):
            return (
                str(d.get("proj") or ""),
                d.get("layer") if d.get("layer") is not None else 1_000_000,
                d.get("expert_id") if d.get("expert_id") is not None else -1,
                str(d.get("module_path") or ""),
            )

        modules.sort(key=_sort_key)
        return modules

    def heretic_build_full_rownorm_lora(
        self,
        *,
        name: str,
        v: list[float],
        weight: float,
        rank: int,
        svd_q: Optional[int] = None,
        svd_niter: int = 6,
        out_dtype: str = "float16",
    ) -> dict:
        """Build FULL row-norm preserving LoRA factors for a named weight.

        Mirrors Heretic's local implementation in `src/heretic/model.py` (RowNormalization.FULL).
        Returns base64-encoded factor tensors.
        """
        import base64

        import numpy as np
        import torch.distributed as dist

        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_group,
        )

        try:
            params = dict(self.model.named_parameters())
            has_param = name in params

            # Determine ownership distribution (TP vs EP) to handle MoE experts correctly.
            tp = get_tensor_model_parallel_group()
            tp_size = tp.world_size
            tp_rank = tp.rank_in_group
            group = tp.device_group
            
            # Default device for communication
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if has_param:
                device = params[name].device
            
            # Check how many ranks have this parameter
            is_owner = torch.tensor([1.0 if has_param else 0.0], device=device)
            if tp_size > 1:
                dist.all_reduce(is_owner, op=dist.ReduceOp.SUM, group=group)
            num_owners = int(is_owner.item())

            if num_owners == 0:
                raise KeyError(f"Parameter {name} not found on any rank (tp_size={tp_size}).")

            lora_A = None
            lora_B = None
            out_dtype_norm = out_dtype
            
            if out_dtype in ("bfloat16", "bf16"):
                out_torch_dtype = torch.bfloat16
                out_dtype_norm = "bfloat16"
            else:
                out_torch_dtype = torch.float16
                out_dtype_norm = "float16"

            # Case 1: Expert Parallelism (Single Owner)
            if num_owners == 1:
                # Find the owner rank
                if tp_size > 1:
                    gathered_owners = [torch.zeros(1, device=device) for _ in range(tp_size)]
                    my_ownership = torch.tensor([1.0 if has_param else 0.0], device=device)
                    dist.all_gather(gathered_owners, my_ownership, group=group)
                    owner_rank = [i for i, t in enumerate(gathered_owners) if t.item() > 0.5][0]
                else:
                    owner_rank = 0

                # Owner computes the factors locally (W is full).
                if tp_rank == owner_rank:
                    W_local = params[name]
                    if W_local.ndim != 2:
                        W_local = W_local.view(W_local.shape[0], -1)
                        
                    # Move heavy compute to CPU to avoid OOM
                    W_full_cpu = W_local.detach().to(dtype=torch.float16).cpu()
                    
                    with torch.no_grad():
                        W_org = W_full_cpu.to(torch.float32)
                        W_row_norms = torch.linalg.vector_norm(W_org, dim=1, keepdim=True)
                        Wn = torch.nn.functional.normalize(W_org, p=2, dim=1)

                        v_t = torch.tensor(v, device="cpu", dtype=torch.float32)
                        # lora_A = v^T W, lora_B = -weight * v
                        lora_A_rank1 = (v_t @ Wn).view(1, -1)
                        lora_B_rank1 = (-float(weight) * v_t).view(-1, 1)

                        W2 = Wn + lora_B_rank1 @ lora_A_rank1
                        W2 = torch.nn.functional.normalize(W2, p=2, dim=1)
                        W2 = W2 * W_row_norms
                        delta = W2 - W_org

                        q = int(svd_q) if svd_q is not None else int(2 * rank + 4)
                        U, S, V = torch.svd_lowrank(delta, q=q, niter=int(svd_niter))
                        
                        U = U[:, :rank]
                        S = S[:rank]
                        Vh = V[:, :rank].T
                        
                        sqrt_S = torch.sqrt(S)
                        lora_B_cpu = U @ torch.diag(sqrt_S)
                        lora_A_cpu = torch.diag(sqrt_S) @ Vh

                        lora_A = lora_A_cpu.to(device=device, dtype=out_torch_dtype)
                        lora_B = lora_B_cpu.to(device=device, dtype=out_torch_dtype)

                # Broadcast result to all ranks
                if tp_size > 1:
                    # Broadcast shapes first
                    shape_info = torch.zeros(4, device=device, dtype=torch.int64)
                    if tp_rank == owner_rank:
                        shape_info[0] = lora_A.shape[0]
                        shape_info[1] = lora_A.shape[1]
                        shape_info[2] = lora_B.shape[0]
                        shape_info[3] = lora_B.shape[1]
                        
                    dist.broadcast(shape_info, src=owner_rank, group=group)
                    
                    if tp_rank != owner_rank:
                        lora_A = torch.empty((shape_info[0], shape_info[1]), device=device, dtype=out_torch_dtype)
                        lora_B = torch.empty((shape_info[2], shape_info[3]), device=device, dtype=out_torch_dtype)
                    
                    dist.broadcast(lora_A, src=owner_rank, group=group)
                    dist.broadcast(lora_B, src=owner_rank, group=group)
                
                # If tp_size=1, lora_A/B are set by the owner logic above (owner_rank=0=tp_rank).
                r = int(lora_A.shape[0])
                in_features = int(lora_A.shape[1])
                out_features = int(lora_B.shape[0])

            # Case 2: Tensor Parallelism (All have it) or Legacy Fallback
            else:
                if not has_param:
                     # Should not happen given num_owners check, but safety fallback
                     raise KeyError(f"Parameter {name} missing on rank {tp_rank} but expected (num_owners={num_owners}).")
                
                W_local = params[name]
                if W_local.ndim != 2:
                    W_local = W_local.view(W_local.shape[0], -1)
            
                v_len = len(v)
                # Infer TP sharding from v length (same convention as compute_vtw).
                if v_len == W_local.shape[0]:
                    shard_mode = "col"  # columns sharded
                elif v_len == W_local.shape[0] * tp_size:
                    shard_mode = "row"  # rows sharded
                else:
                    raise ValueError(
                        f"Shape mismatch: len(v)={v_len} vs W_local.shape={tuple(W_local.shape)} (tp_size={tp_size})"
                    )
    
                # Gather full W on TP rank 0 to avoid duplicated SVD compute.
                W_full = None
                if tp_size == 1:
                    W_full = W_local
                elif shard_mode == "col":
                    if tp_rank == 0:
                        gather_list = [torch.empty_like(W_local) for _ in range(tp_size)]
                        dist.gather(W_local, gather_list=gather_list, dst=0, group=group)
                        W_full = torch.cat(gather_list, dim=1)
                    else:
                        dist.gather(W_local, dst=0, group=group)
                else:  # shard_mode == "row"
                    if tp_rank == 0:
                        gather_list = [torch.empty_like(W_local) for _ in range(tp_size)]
                        dist.gather(W_local, gather_list=gather_list, dst=0, group=group)
                        W_full = torch.cat(gather_list, dim=0)
                    else:
                        dist.gather(W_local, dst=0, group=group)
    
                if tp_rank == 0:
                    assert W_full is not None
                    out_features = int(W_full.shape[0])
                    in_features = int(W_full.shape[1])
                    r = rank
                    
                    # Snapshot W to CPU early and free GPU temp ASAP.
                    W_full_cpu = W_full.detach().to(dtype=torch.float16).cpu()
                    del W_full
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
    
                    with torch.no_grad():
                        W_org = W_full_cpu.to(torch.float32).view(out_features, -1)
                        W_row_norms = torch.linalg.vector_norm(W_org, dim=1, keepdim=True)
                        Wn = torch.nn.functional.normalize(W_org, p=2, dim=1)
    
                        v_t = torch.tensor(v, device="cpu", dtype=torch.float32)
                        # lora_A = v^T W, lora_B = -weight * v
                        lora_A_rank1 = (v_t @ Wn).view(1, -1)
                        lora_B_rank1 = (-float(weight) * v_t).view(-1, 1)
    
                        W2 = Wn + lora_B_rank1 @ lora_A_rank1
                        W2 = torch.nn.functional.normalize(W2, p=2, dim=1)
                        W2 = W2 * W_row_norms
                        delta = W2 - W_org
    
                        q = int(svd_q) if svd_q is not None else int(2 * r + 4)
                        U, S, V = torch.svd_lowrank(delta, q=q, niter=int(svd_niter))
                        
                        U = U[:, :r]
                        S = S[:r]
                        Vh = V[:, :rank].T
                        
                        sqrt_S = torch.sqrt(S)
                        lora_B_cpu = U @ torch.diag(sqrt_S)
                        lora_A_cpu = torch.diag(sqrt_S) @ Vh
                        
                        lora_A = lora_A_cpu.to(device=device, dtype=out_torch_dtype)
                        lora_B = lora_B_cpu.to(device=device, dtype=out_torch_dtype)

                # Broadcast result to other ranks (so they can encode/return consistent values if needed)
                if tp_size > 1:
                     # Broadcast shapes first
                    shape_info = torch.zeros(4, device=device, dtype=torch.int64)
                    if tp_rank == 0:
                        shape_info[0] = lora_A.shape[0]
                        shape_info[1] = lora_A.shape[1]
                        shape_info[2] = lora_B.shape[0]
                        shape_info[3] = lora_B.shape[1]
                        
                    dist.broadcast(shape_info, src=0, group=group)
                    
                    if tp_rank != 0:
                        lora_A = torch.empty((shape_info[0], shape_info[1]), device=device, dtype=out_torch_dtype)
                        lora_B = torch.empty((shape_info[2], shape_info[3]), device=device, dtype=out_torch_dtype)
                    
                    dist.broadcast(lora_A, src=0, group=group)
                    dist.broadcast(lora_B, src=0, group=group)
                
                # Helper variables for return
                r = int(lora_A.shape[0])
                in_features = int(lora_A.shape[1])
                out_features = int(lora_B.shape[0])

            # Return result (all ranks have valid lora_A/B now)
            a_raw = lora_A.detach().cpu().contiguous().numpy().tobytes()
            b_raw = lora_B.detach().cpu().contiguous().numpy().tobytes()
            return {
                "name": name,
                "dtype": out_dtype_norm,
                "lora_A_shape": [r, in_features],
                "lora_B_shape": [out_features, r],
                "lora_A_b64": base64.b64encode(a_raw).decode("ascii"),
                "lora_B_b64": base64.b64encode(b_raw).decode("ascii"),
            }
        except Exception as e:
            logger.error(f"Error when building FULL rownorm LoRA for {name}: {e}")
            return {
                "name": name,
                "dtype": "error",
                "lora_A_shape": [],
                "lora_B_shape": [],
                "lora_A_b64": "",
                "lora_B_b64": "",
                "error": str(e),
            }

    def heretic_build_packed_w2_full_rownorm(
        self,
        *,
        lora_id: str,
        name: str,
        v: list[float],
        weight: float,
        rank: int,
        svd_q: Optional[int] = None,
        svd_niter: int = 6,
        build_device: str = "auto",
        expert_chunk_size: int = 8,
        max_experts: Optional[int] = None,
        max_identity_k: int = 2048,
        out_dtype: str = "float16",
        clear_existing: bool = True,
    ) -> dict:
        """Build and register FULL row-norm preserving factors for packed MoE `w2_weight`.

        This targets a packed 3D tensor parameter (typically `[E_local, out, in]`) and registers
        per-local-expert low-rank factors on the owning `FusedMoE` module under `lora_id`.

        Returns lightweight metadata (no large tensor payloads) to keep RPC overhead low and to
        support EP naturally (each rank builds/registers its own local experts).
        """
        import torch
        import torch.nn.functional as F

        try:
            params = dict(self.model.named_parameters())
            if name not in params:
                raise KeyError(f"Packed w2 parameter not found: {name}")

            w2 = params[name]
            if w2.ndim != 3:
                raise ValueError(
                    f"Expected packed w2 to be 3D [E,...], got shape={tuple(int(x) for x in w2.shape)}"
                )

            # Resolve output dtype for stored factors.
            out_dtype_norm = str(out_dtype)
            if out_dtype_norm in ("bfloat16", "bf16"):
                out_torch_dtype = torch.bfloat16
                out_dtype_norm = "bfloat16"
            elif out_dtype_norm in ("float16", "fp16"):
                out_torch_dtype = torch.float16
                out_dtype_norm = "float16"
            else:
                raise ValueError(f"Unsupported out_dtype for packed w2: {out_dtype!r}")

            # Find the owning module so we can attach factors (avoid global registries).
            module_prefix = str(name).rsplit(".", 1)[0]
            module_obj = dict(self.model.named_modules()).get(module_prefix)
            module_name = module_prefix
            if module_obj is None:
                raise RuntimeError(f"Could not find module for packed parameter: {name}")

            import os

            # Decide where to do the heavy work (GPU-first when possible).
            build_device_norm = str(build_device or "auto").lower()
            if build_device_norm not in ("auto", "cuda", "cpu"):
                raise ValueError(
                    f"Invalid build_device for packed w2 FULL builder: {build_device!r} "
                    f"(expected 'auto'|'cuda'|'cpu')"
                )
            if build_device_norm == "cpu":
                compute_device = torch.device("cpu")
            else:
                use_cuda = bool(w2.is_cuda and torch.cuda.is_available())
                if build_device_norm == "cuda" and not use_cuda:
                    raise RuntimeError(
                        f"build_device='cuda' requested but packed w2 is not on CUDA: name={name} device={w2.device}"
                    )
                compute_device = w2.device if use_cuda else torch.device("cpu")

            debug = str(os.getenv("HERETIC_PACKED_W2_DEBUG", "")).lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
            if debug:
                logger.info(
                    f"[heretic packed-w2] name={name} build_device={build_device_norm} compute_device={compute_device} "
                    f"E_local={int(w2.shape[0])} rank={int(rank)} svd_q={svd_q} svd_niter={int(svd_niter)} "
                    f"expert_chunk_size={int(expert_chunk_size)} max_experts={max_experts} max_identity_k={int(max_identity_k)} "
                    f"out_dtype={out_dtype}"
                )

            # FULL math in fp32 on compute_device.
            v_t = torch.tensor(v, device=compute_device, dtype=torch.float32)

            # Logical dims are determined from the MoE module (not from packed storage).
            hidden_size = int(getattr(module_obj, "hidden_size", 0) or 0)
            intermediate = int(getattr(module_obj, "intermediate_size_per_partition", 0) or 0)
            if hidden_size <= 0 or intermediate <= 0:
                raise RuntimeError(
                    f"Packed w2 owner module missing hidden/intermediate sizes: {module_name} "
                    f"(hidden_size={hidden_size}, intermediate_size_per_partition={intermediate})"
                )
            E_local = int(w2.shape[0])
            out_features = hidden_size
            in_features = intermediate
            if int(v_t.numel()) != int(out_features):
                raise ValueError(
                    f"Refusal vector length mismatch: len(v)={int(v_t.numel())} out_features={int(out_features)}"
                )

            kt_weight_path = getattr(self.server_args, "kt_weight_path", None)
            kt_method = getattr(self.server_args, "kt_method", None)
            _kt_loader = None

            def _unpack_int4_from_int32_rows(
                packed_kn: torch.Tensor, *, pack_factor: int
            ) -> torch.Tensor:
                """Unpack int32-packed 4-bit values into int8 in [-8, 7].

                Input packed_kn: [K_packed, N], where each int32 packs `pack_factor` values
                along the K dimension (LSB-first).
                Output: [K_packed * pack_factor, N] int8.
                """
                if packed_kn.dtype != torch.int32:
                    raise TypeError(f"expected int32 packed, got {packed_kn.dtype}")
                k_packed, n = packed_kn.shape
                mask = (1 << 4) - 1
                out = torch.empty(
                    (int(k_packed) * int(pack_factor), int(n)),
                    device=packed_kn.device,
                    dtype=torch.int8,
                )
                for i in range(int(pack_factor)):
                    vals = (packed_kn >> (4 * i)) & mask
                    out[i:: int(pack_factor), :].copy_((vals.to(torch.int16) - 8).to(torch.int8))
                return out

            def _unpack_uint4_from_int32_rows(
                packed_kn: torch.Tensor, *, pack_factor: int
            ) -> torch.Tensor:
                """Unpack int32-packed 4-bit values into uint8 in [0, 15].

                Input packed_kn: [K_packed, N], where each int32 packs `pack_factor` values
                along the K dimension (LSB-first).
                Output: [K_packed * pack_factor, N] uint8.
                """
                if packed_kn.dtype != torch.int32:
                    raise TypeError(f"expected int32 packed, got {packed_kn.dtype}")
                k_packed, n = packed_kn.shape
                mask = (1 << 4) - 1
                out = torch.empty(
                    (int(k_packed) * int(pack_factor), int(n)),
                    device=packed_kn.device,
                    dtype=torch.uint8,
                )
                for i in range(int(pack_factor)):
                    vals = (packed_kn >> (4 * i)) & mask
                    out[i:: int(pack_factor), :].copy_(vals.to(torch.uint8))
                return out

            def _unpack_int4_from_uint8_cols(
                packed_nk: torch.Tensor, *, pack_factor: int
            ) -> torch.Tensor:
                """Unpack uint8-packed 4-bit values into int8 in [-8, 7].

                Input packed_nk: [N, K_packed], each uint8 packs 2 int4 along K (low/high nibble).
                Output: [N, K_packed * pack_factor] int8.
                """
                if packed_nk.dtype != torch.uint8:
                    raise TypeError(f"expected uint8 packed, got {packed_nk.dtype}")
                n, k_packed = packed_nk.shape
                if int(pack_factor) != 2:
                    raise ValueError(f"uint8 int4 unpack expects pack_factor=2, got {pack_factor}")
                lo = (packed_nk & 0x0F).to(torch.int16) - 8
                hi = ((packed_nk >> 4) & 0x0F).to(torch.int16) - 8
                out = torch.empty((int(n), int(k_packed) * 2), device=packed_nk.device, dtype=torch.int8)
                out[:, 0::2].copy_(lo.to(torch.int8))
                out[:, 1::2].copy_(hi.to(torch.int8))
                return out

            def _unpack_uint4_from_uint8_cols(
                packed_nk: torch.Tensor, *, pack_factor: int
            ) -> torch.Tensor:
                """Unpack uint8-packed 4-bit values into uint8 in [0, 15].

                Input packed_nk: [N, K_packed], each uint8 packs 2 uint4 along K (low/high nibble).
                Output: [N, K_packed * pack_factor] uint8.
                """
                if packed_nk.dtype != torch.uint8:
                    raise TypeError(f"expected uint8 packed, got {packed_nk.dtype}")
                n, k_packed = packed_nk.shape
                if int(pack_factor) != 2:
                    raise ValueError(f"uint8 uint4 unpack expects pack_factor=2, got {pack_factor}")
                lo = (packed_nk & 0x0F).to(torch.uint8)
                hi = ((packed_nk >> 4) & 0x0F).to(torch.uint8)
                out = torch.empty((int(n), int(k_packed) * 2), device=packed_nk.device, dtype=torch.uint8)
                out[:, 0::2].copy_(lo)
                out[:, 1::2].copy_(hi)
                return out

            def _unpack_uint4_from_int32_cols(
                packed_gn: torch.Tensor, *, pack_factor: int
            ) -> torch.Tensor:
                """Unpack int32-packed 4-bit values into uint8 in [0, 15], packing along last dim.

                Input packed_gn: [G, N_packed], each int32 packs `pack_factor` uint4 values along N.
                Output: [G, N_packed * pack_factor] uint8.
                """
                if packed_gn.dtype != torch.int32:
                    raise TypeError(f"expected int32 packed, got {packed_gn.dtype}")
                g, n_packed = packed_gn.shape
                mask = (1 << 4) - 1
                out = torch.empty(
                    (int(g), int(n_packed) * int(pack_factor)),
                    device=packed_gn.device,
                    dtype=torch.uint8,
                )
                for i in range(int(pack_factor)):
                    vals = (packed_gn >> (4 * i)) & mask
                    out[:, i:: int(pack_factor)].copy_(vals.to(torch.uint8))
                return out

            def _get_w2_expert_fp32(e: int, *, out_device: torch.device) -> torch.Tensor:
                """Return logical W_e in fp32 on `out_device` as [out(hidden), in(intermediate)]."""
                W_store = w2[e]

                # Case A: float storage already holds logical (or transposed) expert matrix.
                if torch.is_floating_point(W_store):
                    W2d = W_store.detach()
                    if W2d.ndim != 2:
                        W2d = W2d.view(W2d.shape[0], -1)
                    if int(W2d.shape[0]) == hidden_size and int(W2d.shape[1]) == intermediate:
                        return W2d.to(dtype=torch.float32, device=out_device, non_blocking=True)
                    if int(W2d.shape[0]) == intermediate and int(W2d.shape[1]) == hidden_size:
                        return W2d.T.to(dtype=torch.float32, device=out_device, non_blocking=True)
                    raise RuntimeError(
                        f"Unexpected float w2 expert shape for {name}: got {tuple(int(x) for x in W2d.shape)} "
                        f"expected ({hidden_size},{intermediate}) or ({intermediate},{hidden_size})"
                    )

                # Case B: int4 packed int32 + per-group scales (CompressedTensors pack_quantized / KT RAWINT4 GPU).
                if (
                    W_store.dtype == torch.int32
                    and hasattr(module_obj, "w2_weight_scale")
                    and str(name).endswith(("w2_weight_packed", "w2_qweight", "w2_qweight_packed"))
                ):
                    # Storage can be:
                    # - pack_quantized INT4: [K_packed, N] or [N, K_packed] with N == hidden_size
                    # - Marlin-converted INT4: [K//16, N*2] or [N*2, K//16] with N == hidden_size (uint4b8).
                    Wp = W_store.detach()
                    if Wp.ndim != 2:
                        Wp = Wp.view(Wp.shape[0], -1)

                    # Marlin-converted weights: second dim is hidden*2 (for 4-bit).
                    if int(Wp.shape[1]) == int(hidden_size) * 2 or int(Wp.shape[0]) == int(hidden_size) * 2:
                        # Normalize qweight to Marlin layout [K//16, N*2].
                        if int(Wp.shape[1]) == int(hidden_size) * 2:
                            marlin_qw = Wp.contiguous()
                        else:
                            marlin_qw = Wp.T.contiguous()

                        # Scales are expected in Marlin layout [groups, N] after permutation.
                        s = getattr(module_obj, "w2_weight_scale")[e].detach()
                        if s.ndim != 2:
                            s = s.view(s.shape[0], -1)
                        if int(s.shape[1]) == hidden_size:
                            marlin_scales = s
                        elif int(s.shape[0]) == hidden_size:
                            marlin_scales = s.T.contiguous()
                        else:
                            raise RuntimeError(
                                f"Unexpected w2_weight_scale shape for Marlin snapshot {module_name}: {tuple(int(x) for x in s.shape)}"
                            )
                        # Marlin kernels expect fp16 scales.
                        marlin_scales = marlin_scales.to(dtype=torch.float16).contiguous()

                        # Use Marlin GEMM to materialize W (fast, avoids reverse-engineering packing).
                        from sglang.srt.layers.quantization.marlin_utils import (
                            get_scalar_types,
                            marlin_make_empty_g_idx,
                            marlin_make_workspace,
                            apply_gptq_marlin_linear,
                        )

                        _, scalar_types = get_scalar_types()
                        wtype = scalar_types.uint4b8  # symmetric 4-bit (compressed-tensors MoE uses symmetric)

                        dev = marlin_qw.device
                        workspace = marlin_make_workspace(dev)
                        empty = marlin_make_empty_g_idx(dev)

                        # g_idx/sort_indices may be empty depending on actorder.
                        g_idx_all = getattr(module_obj, "w2_weight_g_idx", None)
                        sort_all = getattr(module_obj, "w2_g_idx_sort_indices", None)
                        g_idx = empty
                        g_idx_sort_indices = empty
                        if isinstance(g_idx_all, torch.Tensor) and g_idx_all.ndim >= 2 and int(g_idx_all.shape[0]) > int(e):
                            g_idx = g_idx_all[int(e)]
                        if isinstance(sort_all, torch.Tensor) and sort_all.ndim >= 2 and int(sort_all.shape[0]) > int(e):
                            g_idx_sort_indices = sort_all[int(e)]

                        if int(intermediate) > int(max_identity_k):
                            raise RuntimeError(
                                f"Marlin snapshot requires identity GEMM of size K={int(intermediate)}, "
                                f"which exceeds max_identity_k={int(max_identity_k)} for {name}"
                            )

                        # Marlin kernels expect fp16 inputs; disable autocast to avoid bf16.
                        x = torch.eye(int(intermediate), device=dev, dtype=torch.float16)
                        # torch.cuda.amp.autocast is deprecated; use torch.amp.autocast.
                        with torch.amp.autocast(device_type="cuda", enabled=False):
                            out = apply_gptq_marlin_linear(
                                input=x,
                                weight=marlin_qw,
                                weight_scale=marlin_scales,
                                weight_zp=empty,
                                g_idx=g_idx,
                                g_idx_sort_indices=g_idx_sort_indices,
                                workspace=workspace,
                                wtype=wtype,
                                output_size_per_partition=int(hidden_size),
                                input_size_per_partition=int(intermediate),
                                is_k_full=True,
                                bias=None,
                            )
                        # out is [intermediate, hidden] => W is [hidden, intermediate]
                        return out.T.to(dtype=torch.float32).to(
                            device=out_device, non_blocking=True
                        )

                    # pack_quantized INT4: normalize to [K_packed, N] where N == hidden.
                    if int(Wp.shape[1]) == hidden_size:
                        packed_kn = Wp
                    elif int(Wp.shape[0]) == hidden_size:
                        packed_kn = Wp.T.contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected int32 packed w2 expert shape for {name}: {tuple(int(x) for x in Wp.shape)} "
                            f"(hidden_size={hidden_size})"
                        )
                    k_packed = int(packed_kn.shape[0])
                    pack_factor = (
                        int(intermediate // k_packed) if (k_packed > 0 and intermediate % k_packed == 0) else int(round(intermediate / max(1, k_packed)))
                    )
                    if pack_factor <= 0:
                        raise RuntimeError(
                            f"Failed to infer pack_factor for {name}: intermediate={intermediate} k_packed={k_packed}"
                        )
                    if int(k_packed) * int(pack_factor) < int(intermediate):
                        raise RuntimeError(
                            f"Packed K too small for {name}: k_packed={k_packed} pack_factor={pack_factor} "
                            f"intermediate={intermediate}"
                        )
                    q_kn = _unpack_int4_from_int32_rows(packed_kn, pack_factor=pack_factor).to(torch.float32)
                    q_kn = q_kn[:intermediate, :hidden_size]

                    # Optional shape metadata (some pack_quantized checkpoints store original shape).
                    w2_shape = getattr(module_obj, "w2_weight_shape", None)
                    if w2_shape is not None:
                        try:
                            shp2 = w2_shape[e].detach().view(-1).to(device="cpu")
                            if int(shp2.numel()) >= 2:
                                k0 = int(shp2[0].item())
                                n0 = int(shp2[1].item())
                                # Accept either (K,N) or (N,K) depending on checkpoint convention.
                                if not (
                                    (k0 == intermediate and n0 == hidden_size)
                                    or (k0 == hidden_size and n0 == intermediate)
                                ):
                                    raise RuntimeError(
                                        f"w2_weight_shape metadata mismatch for {module_name}: "
                                        f"shape=({k0},{n0}) expected ({intermediate},{hidden_size})"
                                    )
                        except Exception:
                            # Best-effort only; don't fail builds purely on metadata inconsistencies.
                            pass

                    s = getattr(module_obj, "w2_weight_scale")[e].detach()
                    if s.ndim != 2:
                        s = s.view(s.shape[0], -1)
                    if int(s.shape[1]) == hidden_size:
                        scales_gn = s
                    elif int(s.shape[0]) == hidden_size:
                        scales_gn = s.T.contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected w2_weight_scale shape for {module_name}: {tuple(int(x) for x in s.shape)}"
                        )
                    num_groups = int(scales_gn.shape[0])
                    if num_groups <= 0:
                        raise RuntimeError(f"Invalid num_groups in scales for {module_name}: {num_groups}")
                    group_size = int(round(intermediate / max(1, num_groups))) if num_groups > 1 else intermediate
                    scales_kn = scales_gn.to(torch.float32).repeat_interleave(group_size, dim=0)
                    scales_kn = scales_kn[:intermediate, :hidden_size]

                    W_kn = q_kn * scales_kn
                    return W_kn.T.contiguous().to(device=out_device, non_blocking=True)

                # Case C: uint8-packed int4 + scales (moe_wna16 symmetric).
                if W_store.dtype == torch.uint8 and hasattr(module_obj, "w2_scales") and str(name).endswith(
                    "w2_qweight"
                ):
                    Wp = W_store.detach()
                    if Wp.ndim != 2:
                        Wp = Wp.view(Wp.shape[0], -1)
                    if int(Wp.shape[0]) != hidden_size:
                        raise RuntimeError(
                            f"Unexpected uint8 w2_qweight shape for {name}: {tuple(int(x) for x in Wp.shape)}"
                        )
                    k_packed = int(Wp.shape[1])
                    pack_factor = (
                        int(intermediate // k_packed) if (k_packed > 0 and intermediate % k_packed == 0) else int(round(intermediate / max(1, k_packed)))
                    )
                    if pack_factor not in (1, 2):
                        raise RuntimeError(
                            f"Unexpected uint8 pack_factor for {name}: inferred={pack_factor} (expected 1 or 2)"
                        )
                    # For symmetric, q values are interpreted as signed int4; for asymmetric, as uint4 with zp.
                    has_zp = getattr(module_obj, "w2_qzeros", None) is not None
                    if has_zp:
                        if pack_factor == 2:
                            q_nk_u = _unpack_uint4_from_uint8_cols(Wp, pack_factor=2).to(torch.float32)
                        else:
                            q_nk_u = Wp.to(torch.float32)
                        q_nk_u = q_nk_u[:, :intermediate]
                    else:
                        if pack_factor == 2:
                            q_nk = _unpack_int4_from_uint8_cols(Wp, pack_factor=2).to(torch.float32)
                        else:
                            # Best-effort: interpret uint8 as signed int8 stored with +128 offset.
                            q_nk = (Wp.to(torch.int16) - 128).to(torch.float32)
                        q_nk = q_nk[:, :intermediate]

                    scales = getattr(module_obj, "w2_scales")[e].detach().to(torch.float32)
                    if scales.ndim != 2:
                        scales = scales.view(scales.shape[0], -1)
                    if int(scales.shape[0]) != hidden_size:
                        raise RuntimeError(
                            f"Unexpected w2_scales shape for {module_name}: {tuple(int(x) for x in scales.shape)}"
                        )
                    group_size = int(getattr(module_obj, "group_size", 0) or 0)
                    if group_size <= 0:
                        g = int(scales.shape[1])
                        group_size = int(round(intermediate / max(1, g))) if g > 1 else intermediate
                    scales_full = scales.repeat_interleave(group_size, dim=1)[:, :intermediate]
                    if not has_zp:
                        W = q_nk * scales_full
                        return W.to(torch.float32).to(device=out_device, non_blocking=True)

                    # Unpack and broadcast zero-points: stored as [hidden//pack_factor, intermediate//group_size]
                    z = getattr(module_obj, "w2_qzeros")[e].detach()
                    if z.ndim != 2:
                        z = z.view(z.shape[0], -1)
                    if int(z.shape[1]) != int(scales.shape[1]):
                        raise RuntimeError(
                            f"Unexpected w2_qzeros group dim for {module_name}: {tuple(int(x) for x in z.shape)} "
                            f"(expected second dim={int(scales.shape[1])})"
                        )
                    if int(z.shape[0]) * int(pack_factor) != hidden_size:
                        raise RuntimeError(
                            f"Unexpected w2_qzeros packed hidden dim for {module_name}: {tuple(int(x) for x in z.shape)} "
                            f"(hidden_size={hidden_size}, pack_factor={pack_factor})"
                        )
                    if pack_factor == 2:
                        zp_ng = _unpack_uint4_from_uint8_cols(z.T.contiguous(), pack_factor=2).T  # [hidden, groups]
                    else:
                        zp_ng = z.to(torch.uint8)  # [hidden, groups]
                    zp_full = zp_ng.to(torch.float32).repeat_interleave(group_size, dim=1)[:, :intermediate]
                    W = (q_nk_u - zp_full) * scales_full
                    return W.to(torch.float32).to(device=out_device, non_blocking=True)

                # Case D: GPTQ-style int32 packed qweight + scales + (optional) packed qzeros (MoE GPTQ/Marlin pre-repack).
                if (
                    W_store.dtype == torch.int32
                    and hasattr(module_obj, "w2_scales")
                    and getattr(module_obj, "w2_qzeros", None) is not None
                    and str(name).endswith("w2_qweight")
                ):
                    Wp = W_store.detach()
                    if Wp.ndim != 2:
                        Wp = Wp.view(Wp.shape[0], -1)
                    # Expect GPTQ pre-repack layout: [K_packed, N] or [N, K_packed]
                    if int(Wp.shape[1]) == hidden_size:
                        packed_kn = Wp
                    elif int(Wp.shape[0]) == hidden_size:
                        packed_kn = Wp.T.contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected GPTQ packed w2_qweight shape for {name}: {tuple(int(x) for x in Wp.shape)} "
                            f"(hidden_size={hidden_size})"
                        )
                    k_packed = int(packed_kn.shape[0])
                    pack_factor = (
                        int(intermediate // k_packed) if (k_packed > 0 and intermediate % k_packed == 0) else int(round(intermediate / max(1, k_packed)))
                    )
                    if pack_factor <= 0:
                        raise RuntimeError(
                            f"Failed to infer pack_factor for {name}: intermediate={intermediate} k_packed={k_packed}"
                        )
                    if int(k_packed) * int(pack_factor) < int(intermediate):
                        raise RuntimeError(
                            f"Packed K too small for {name}: k_packed={k_packed} pack_factor={pack_factor} "
                            f"intermediate={intermediate}"
                        )
                    q_kn_u = _unpack_uint4_from_int32_rows(packed_kn, pack_factor=pack_factor).to(torch.float32)
                    q_kn_u = q_kn_u[:intermediate, :hidden_size]

                    s = getattr(module_obj, "w2_scales")[e].detach()
                    if s.ndim != 2:
                        s = s.view(s.shape[0], -1)
                    if int(s.shape[1]) == hidden_size:
                        scales_gn = s
                    elif int(s.shape[0]) == hidden_size:
                        scales_gn = s.T.contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected w2_scales shape for {module_name}: {tuple(int(x) for x in s.shape)}"
                        )
                    num_groups = int(scales_gn.shape[0])
                    group_size = int(round(intermediate / max(1, num_groups))) if num_groups > 1 else intermediate
                    scales_kn = scales_gn.to(torch.float32).repeat_interleave(group_size, dim=0)
                    scales_kn = scales_kn[:intermediate, :hidden_size]

                    z = getattr(module_obj, "w2_qzeros")[e].detach()
                    if z.ndim != 2:
                        z = z.view(z.shape[0], -1)
                    # Expect packed along hidden dim: [G, N_packed] or [N_packed, G]
                    if int(z.shape[0]) == num_groups:
                        zp_gn_packed = z
                    elif int(z.shape[1]) == num_groups:
                        zp_gn_packed = z.T.contiguous()
                    else:
                        raise RuntimeError(
                            f"Unexpected w2_qzeros shape for {module_name}: {tuple(int(x) for x in z.shape)} "
                            f"(num_groups={num_groups})"
                        )
                    # Unpack to [G, N] uint4.
                    if zp_gn_packed.dtype == torch.int32:
                        zp_gn = _unpack_uint4_from_int32_cols(zp_gn_packed.to(torch.int32), pack_factor=pack_factor)
                    elif zp_gn_packed.dtype == torch.uint8:
                        # uint8 packing uses 2 per byte along hidden
                        if pack_factor != 2:
                            raise RuntimeError(
                                f"Unsupported uint8 qzeros pack_factor for {name}: {pack_factor} (expected 2)"
                            )
                        zp_gn = _unpack_uint4_from_uint8_cols(zp_gn_packed, pack_factor=2)
                    else:
                        raise RuntimeError(
                            f"Unsupported w2_qzeros dtype for {module_name}: {zp_gn_packed.dtype}"
                        )
                    if int(zp_gn.shape[1]) != hidden_size:
                        zp_gn = zp_gn[:, :hidden_size]
                    zp_kn = zp_gn.to(torch.float32).repeat_interleave(group_size, dim=0)
                    zp_kn = zp_kn[:intermediate, :hidden_size]

                    W_kn = (q_kn_u - zp_kn) * scales_kn
                    return W_kn.T.contiguous().to(device=out_device, non_blocking=True)

                # Case E: KT RAWINT4 on-disk fallback (CPU experts / missing in-memory params).
                # Best-effort: load {down_proj.weight_packed, down_proj.weight_scale, down_proj.weight_shape}
                # from kt_weight_path using ktransformers loader, then dequantize to fp32 CPU.
                if (
                    kt_weight_path
                    and kt_method
                    and str(kt_method).upper() == "RAWINT4"
                    and str(name).endswith(("w2_weight_packed", "w2_qweight", "w2_qweight_packed"))
                ):
                    nonlocal _kt_loader
                    try:
                        if _kt_loader is None:
                            from ktransformers.kt_kernel.python.utils.loader import (
                                CompressedSafeTensorLoader,
                            )

                            _kt_loader = CompressedSafeTensorLoader(str(kt_weight_path))

                        # Map local expert idx to global expert id if available.
                        global_e = int(e)
                        if hasattr(module_obj, "dispatcher") and hasattr(
                            module_obj.dispatcher, "local_expert_mapping"
                        ):
                            m = module_obj.dispatcher.local_expert_mapping
                            if m is not None and int(m.numel()) > global_e:
                                global_e = int(m[global_e].item())

                        base = module_name
                        if base.endswith(".mlp.experts"):
                            base = base[: -len(".mlp.experts")]

                        w_key = f"{base}.mlp.experts.{global_e}.down_proj.weight_packed"
                        s_key = f"{base}.mlp.experts.{global_e}.down_proj.weight_scale"
                        sh_key = f"{base}.mlp.experts.{global_e}.down_proj.weight_shape"

                        w_packed = _kt_loader.load_tensor(w_key, device="cpu").to(torch.int32)
                        w_scale = _kt_loader.load_tensor(s_key, device="cpu").to(torch.float32)
                        w_shape = None
                        try:
                            w_shape = _kt_loader.load_tensor(sh_key, device="cpu")
                        except Exception:
                            w_shape = None

                        if w_shape is not None:
                            shp = tuple(int(v) for v in w_shape.view(-1).tolist()[:2])
                        else:
                            shp = (hidden_size, intermediate)

                        # Use the same primitive as KT conversion scripts when available.
                        from compressed_tensors.compressors import unpack_from_int32 as _ct_unpack_from_int32

                        W = _ct_unpack_from_int32(w_packed, 4, shp).to(torch.float32)
                        if W.ndim != 2:
                            W = W.view(int(shp[0]), int(shp[1]))

                        # Expand scales along the input dimension (groupwise).
                        if w_scale.ndim == 1:
                            w_scale = w_scale.unsqueeze(1)
                        if int(w_scale.shape[0]) == int(W.shape[0]):
                            # [out, groups] or [out, 1]
                            groups = int(w_scale.shape[1])
                            gsz = int(round(int(W.shape[1]) / max(1, groups))) if groups > 1 else int(W.shape[1])
                            S = w_scale.repeat_interleave(gsz, dim=1)[:, : int(W.shape[1])]
                        elif int(w_scale.shape[1]) == int(W.shape[1]):
                            # [groups, in] style (unexpected); fallback to reshape/broadcast if possible
                            S = w_scale
                            if S.numel() == W.numel():
                                S = S.reshape_as(W)
                        else:
                            raise RuntimeError(
                                f"KT RAWINT4 scale shape mismatch for {s_key}: "
                                f"scale={tuple(int(x) for x in w_scale.shape)} weight={tuple(int(x) for x in W.shape)}"
                            )

                        W = (W * S).to(torch.float32)
                        # Normalize to logical [hidden, intermediate]
                        if int(W.shape[0]) == hidden_size and int(W.shape[1]) == intermediate:
                            return W.contiguous()
                        if int(W.shape[0]) == intermediate and int(W.shape[1]) == hidden_size:
                            return W.T.contiguous()
                        raise RuntimeError(
                            f"KT RAWINT4 dequant shape mismatch for {w_key}: got {tuple(int(x) for x in W.shape)} "
                            f"expected ({hidden_size},{intermediate})"
                        )
                    except Exception:
                        # Fall through to generic error below.
                        pass

                raise NotImplementedError(
                    f"Unsupported packed w2 storage for Option B snapshot: name={name} dtype={W_store.dtype} "
                    f"module={module_name}"
                )

            # Preallocate stacked factors on the chosen compute device.
            r = int(rank)
            A_stack = torch.zeros(
                (E_local, r, in_features), dtype=out_torch_dtype, device=compute_device
            )
            B_stack = torch.zeros(
                (E_local, out_features, r), dtype=out_torch_dtype, device=compute_device
            )

            q = int(svd_q) if svd_q is not None else int(2 * r + 4)
            niter = int(svd_niter)
            lam = float(weight)
            storage_dtype = str(w2.dtype)
            if torch.is_floating_point(w2):
                quant_scheme = "float"
            elif w2.dtype == torch.int32 and hasattr(module_obj, "w2_weight_scale"):
                quant_scheme = "compressed_tensors_pack_quantized_int4"
            elif w2.dtype == torch.uint8 and hasattr(module_obj, "w2_scales"):
                quant_scheme = (
                    "moe_wna16_uint8_packed"
                    + ("_with_zp" if getattr(module_obj, "w2_qzeros", None) is not None else "_symmetric")
                )
            elif w2.dtype == torch.int32 and hasattr(module_obj, "w2_scales"):
                quant_scheme = "gptq_like_int4"
            else:
                quant_scheme = "unknown"

            E_eff = int(E_local)
            if max_experts is not None:
                E_eff = min(E_eff, int(max_experts))
            chunk = max(1, int(expert_chunk_size))

            with torch.no_grad():
                for e0 in range(0, E_eff, chunk):
                    e1 = min(E_eff, e0 + chunk)
                    for e in range(e0, e1):
                        W_e = _get_w2_expert_fp32(int(e), out_device=compute_device)
                        if (
                            W_e.ndim != 2
                            or int(W_e.shape[0]) != int(out_features)
                            or int(W_e.shape[1]) != int(in_features)
                        ):
                            raise RuntimeError(
                                f"Dequant snapshot shape mismatch for {name} expert {e}: got {tuple(int(x) for x in W_e.shape)} "
                                f"expected ({int(out_features)},{int(in_features)})"
                            )
                        if W_e.dtype != torch.float32:
                            W_e = W_e.to(dtype=torch.float32)

                        W_row_norms = torch.linalg.vector_norm(W_e, dim=1, keepdim=True)
                        Wn = F.normalize(W_e, p=2, dim=1)

                        # Rank-1 directional update in normalized space.
                        A_rank1 = (v_t @ Wn).view(1, -1)  # [1, in]
                        B_rank1 = (-lam * v_t).view(-1, 1)  # [out, 1]

                        W2 = Wn + B_rank1 @ A_rank1
                        W2 = F.normalize(W2, p=2, dim=1)
                        W2 = W2 * W_row_norms
                        delta = (W2 - W_e).to(dtype=torch.float32)

                        U, S, V = torch.svd_lowrank(delta, q=q, niter=niter)
                        U = U[:, :r]
                        S = S[:r]
                        Vh = V[:, :r].T

                        sqrt_S = torch.sqrt(S)
                        B = U @ torch.diag(sqrt_S)  # [out, r]
                        A = torch.diag(sqrt_S) @ Vh  # [r, in]

                        A_stack[e].copy_(A.to(dtype=out_torch_dtype), non_blocking=True)
                        B_stack[e].copy_(B.to(dtype=out_torch_dtype), non_blocking=True)

            # Register factors on the module (per lora_id).
            if clear_existing:
                store0 = getattr(module_obj, "_heretic_packed_w2_by_lora_id", None)
                if isinstance(store0, dict):
                    store0.pop(str(lora_id), None)
            store = getattr(module_obj, "_heretic_packed_w2_by_lora_id", None)
            if not isinstance(store, dict):
                store = {}
                setattr(module_obj, "_heretic_packed_w2_by_lora_id", store)

            dev = w2.device
            store[str(lora_id)] = {
                "A": A_stack.to(device=dev, non_blocking=True),
                "B": B_stack.to(device=dev, non_blocking=True),
                "dtype": out_dtype_norm,
                "module_name": str(module_name or ""),
                "param_name": str(name),
                "expert_id_space": "local",
                "num_local_experts": E_local,
                "out_features": out_features,
                "in_features": in_features,
                "rank": r,
            }

            return {
                "success": True,
                "message": "ok",
                "lora_id": str(lora_id),
                "name": str(name),
                "dtype": out_dtype_norm,
                "num_local_experts": E_local,
                "storage_dtype": storage_dtype,
                "quant_scheme": quant_scheme,
                "lora_A_shape": [E_local, r, in_features],
                "lora_B_shape": [E_local, out_features, r],
            }
        except Exception as e:
            logger.error(f"Error when building packed w2 FULL factors for {name}: {e}")
            return {
                "success": False,
                "message": str(e),
                "lora_id": str(lora_id),
                "name": str(name),
                "dtype": "error",
                "num_local_experts": -1,
                "storage_dtype": "error",
                "quant_scheme": "error",
                "lora_A_shape": [],
                "lora_B_shape": [],
            }

    def heretic_unload_packed_moe_adapter(self, *, lora_id: str) -> dict:
        """Remove any registered packed-MoE payload for this `lora_id` from all modules."""
        removed = 0
        try:
            for _, m in self.model.named_modules():
                store = getattr(m, "_heretic_packed_w2_by_lora_id", None)
                if isinstance(store, dict) and str(lora_id) in store:
                    store.pop(str(lora_id), None)
                    removed += 1
            return {"success": True, "message": f"removed_from={removed}", "lora_id": str(lora_id)}
        except Exception as e:
            return {"success": False, "message": str(e), "lora_id": str(lora_id)}

    def compute_vtw(self, name: str, v: list[float], dtype: str = "float32"):
        """Compute v^T W for a named parameter without exporting full weights.

        Supports simple TP layouts by inferring whether W is row- or column-sharded.
        Returns (vtw_list, implementation_string).
        """
        import torch.distributed as dist

        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_group,
        )

        try:
            params = dict(self.model.named_parameters())
            if name not in params:
                raise KeyError(f"Unknown parameter name: {name}")

            W = params[name]
            # Flatten non-2D weights (e.g. conv) into (out, in) if possible.
            if W.ndim != 2:
                W = W.view(W.shape[0], -1)

            # Parse dtype.
            if dtype in ("float16", "fp16"):
                v_dtype = torch.float16
            elif dtype in ("bfloat16", "bf16"):
                v_dtype = torch.bfloat16
            else:
                v_dtype = torch.float32

            v_t = torch.tensor(v, device=W.device, dtype=v_dtype)

            tp = get_tensor_model_parallel_group()
            tp_size = tp.world_size
            tp_rank = tp.rank_in_group
            group = tp.device_group

            # Ensure matmul dtype compatibility (e.g., bf16 weights).
            if W.dtype in (torch.float16, torch.bfloat16, torch.float32) and v_t.dtype != W.dtype:
                v_t = v_t.to(dtype=W.dtype)

            # Case A: v matches local out dim -> W is column-sharded (or not sharded).
            if v_t.numel() == W.shape[0]:
                vtw_local = v_t @ W  # (in_local or in_global)
                if tp_size == 1:
                    vtw = vtw_local
                else:
                    out_list = [torch.empty_like(vtw_local) for _ in range(tp_size)]
                    dist.all_gather(out_list, vtw_local, group=group)
                    vtw = torch.cat(out_list, dim=-1)
                implementation = "matmul_col_gather"

            # Case B: v matches global out dim -> W is row-sharded.
            elif v_t.numel() == W.shape[0] * tp_size:
                local_out = W.shape[0]
                start = tp_rank * local_out
                end = start + local_out
                v_local = v_t[start:end]
                vtw = v_local @ W  # (in_global)
                if tp_size > 1:
                    dist.all_reduce(vtw, op=dist.ReduceOp.SUM, group=group)
                implementation = "matmul_row_reduce"

            else:
                raise ValueError(
                    f"Shape mismatch: len(v)={v_t.numel()} vs W.shape={tuple(W.shape)} "
                    f"(tp_size={tp_size})"
                )

            return vtw.detach().to(torch.float32).cpu().tolist(), implementation

        except Exception as e:
            logger.error(f"Error when computing v^T W for {name}: {e}")
            return [], "error"

    def compute_vtw_batch(self, items) -> list[dict]:
        """Compute v^T W for many named parameters without exporting full weights."""
        import torch.distributed as dist

        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_group,
        )

        try:
            params = dict(self.model.named_parameters())
            tp = get_tensor_model_parallel_group()
            tp_size = tp.world_size
            tp_rank = tp.rank_in_group
            group = tp.device_group
        except Exception as e:
            logger.error(f"Error initializing compute_vtw_batch: {e}")
            return []

        results: list[dict] = []
        for it in items:
            name = getattr(it, "name", None)
            v = getattr(it, "v", None)
            dtype = getattr(it, "dtype", "float32")

            if not isinstance(name, str) or not isinstance(v, list):
                results.append({"name": name, "vtw": [], "implementation": "error"})
                continue

            try:
                if name not in params:
                    raise KeyError(f"Unknown parameter name: {name}")

                W = params[name]
                if W.ndim != 2:
                    W = W.view(W.shape[0], -1)

                if dtype in ("float16", "fp16"):
                    v_dtype = torch.float16
                elif dtype in ("bfloat16", "bf16"):
                    v_dtype = torch.bfloat16
                else:
                    v_dtype = torch.float32

                v_t = torch.tensor(v, device=W.device, dtype=v_dtype)
                if W.dtype in (torch.float16, torch.bfloat16, torch.float32) and v_t.dtype != W.dtype:
                    v_t = v_t.to(dtype=W.dtype)

                # Case A: v matches local out dim -> W is column-sharded (or not sharded).
                if v_t.numel() == W.shape[0]:
                    vtw_local = v_t @ W
                    if tp_size == 1:
                        vtw = vtw_local
                    else:
                        out_list = [torch.empty_like(vtw_local) for _ in range(tp_size)]
                        dist.all_gather(out_list, vtw_local, group=group)
                        vtw = torch.cat(out_list, dim=-1)
                    implementation = "matmul_col_gather"

                # Case B: v matches global out dim -> W is row-sharded.
                elif v_t.numel() == W.shape[0] * tp_size:
                    local_out = W.shape[0]
                    start = tp_rank * local_out
                    end = start + local_out
                    v_local = v_t[start:end]
                    vtw = v_local @ W
                    if tp_size > 1:
                        dist.all_reduce(vtw, op=dist.ReduceOp.SUM, group=group)
                    implementation = "matmul_row_reduce"

                else:
                    raise ValueError(
                        f"Shape mismatch: len(v)={v_t.numel()} vs W.shape={tuple(W.shape)} "
                        f"(tp_size={tp_size})"
                    )

                results.append(
                    {
                        "name": name,
                        "vtw": vtw.detach().to(torch.float32).cpu().tolist(),
                        "implementation": implementation,
                    }
                )
            except Exception as e:
                logger.error(f"Error when computing v^T W for {name}: {e}")
                results.append({"name": name, "vtw": [], "implementation": "error"})

        return results

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config, FalconH1Config | NemotronHConfig | Lfm2Config | Lfm2MoeConfig
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config
        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return min(self.swa_max_total_num_tokens, self.max_total_num_tokens)
        else:
            return self.max_total_num_tokens

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    @property
    def mambaish_config(self):
        return self.mamba2_config or self.hybrid_gdn_config or self.kimi_linear_config

    def can_run_piecewise_cuda_graph(self):
        if self.is_draft_worker:
            return False

        if self.server_args.enable_torch_compile:
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph has conflict with torch compile",
            )
            return False
        if self.pp_size > 1:
            # TODO(yuwei): support PP
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph does not support PP",
            )
            return False
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            # TODO(yuwei): fix the compilation errors for MOE A2A backend
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph due to existing compilation errors",
            )
            return False
        return True

    def configure_kv_cache_dtype(self):
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        log_info_on_rank0(logger, f"Using KV cache dtype: {self.kv_cache_dtype}")

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        self.prefill_attention_backend_str, self.decode_attention_backend_str = (
            self.server_args.get_attention_backends()
        )

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def init_double_sparsity_channel_config(self, selected_channel):
        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.start_layer, self.end_layer):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def kernel_warmup(self):
        """
        Warmup and tune kernels before cuda graph capture.
        Currently only doing FlashInfer autotune.
        """
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        if self.server_args.disable_flashinfer_autotune:
            return False

        backend_str = self.server_args.moe_runner_backend
        if backend_str not in [
            "flashinfer_trtllm",
            "flashinfer_mxfp4",
            # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
            # "flashinfer_cutlass",
        ]:
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if (
            self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            or self.spec_algorithm.is_ngram()
        ):
            return not self.is_draft_worker

        return True

    def _flashinfer_autotune(self):
        """Run flashinfer autotune."""
        from flashinfer.autotuner import autotune

        logger.info("Running FlashInfer autotune...")

        with torch.inference_mode(), autotune():
            self._dummy_run(batch_size=self.req_to_token_pool.size)

        logger.info("FlashInfer autotune completed.")

    def _dummy_run(self, batch_size: int):
        """Run a dummy forward pass for warmup/profiling."""
        if self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if (
            self.spec_algorithm.is_eagle()
            or self.spec_algorithm.is_standalone()
            or self.spec_algorithm.is_ngram()
        ):
            if self.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                capture_forward_mode = ForwardMode.TARGET_VERIFY
                num_tokens_per_bs = self.server_args.speculative_num_draft_tokens

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()

        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture()

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        buffers: GraphInputBuffers = GraphInputBuffers.create(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=0,
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        if not self.is_generation:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        def get_spec_info():
            spec_info = None
            if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
                from sglang.srt.speculative.eagle_info import EagleVerifyInput

                if self.is_draft_worker:
                    raise RuntimeError("This should not happen.")
                else:
                    spec_info = EagleVerifyInput(
                        draft_token=None,
                        custom_mask=buffers.custom_mask,
                        positions=None,
                        retrive_index=None,
                        retrive_next_token=None,
                        retrive_next_sibling=None,
                        retrive_cum_len=None,
                        spec_steps=self.server_args.speculative_num_steps,
                        topk=self.server_args.speculative_eagle_topk,
                        draft_token_num=self.server_args.speculative_num_draft_tokens,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )

            elif self.spec_algorithm.is_ngram():
                from sglang.srt.speculative.ngram_info import NgramVerifyInput

                spec_info = NgramVerifyInput(
                    draft_token=None,
                    tree_mask=buffers.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    draft_token_num=num_tokens_per_bs,
                )
                spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

            return spec_info

        spec_info = get_spec_info()
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        self.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        run_once()

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        graph_runners = defaultdict(
            lambda: CudaGraphRunner,
            {
                "cpu": CPUGraphRunner,
                "npu": NPUGraphRunner,
            },
        )
        self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_piecewise_cuda_graphs(self):
        """Initialize piecewise CUDA graph runner."""
        self.piecewise_cuda_graph_runner = None

        if (
            not self.server_args.enable_piecewise_cuda_graph
            or not self.can_run_piecewise_cuda_graph()
        ):
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)
        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        for layer in language_model.model.layers:
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    self.attention_layers.append(layer.self_attn.attn)
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    self.attention_layers.append(layer.self_attn.attn_mqa)
            # For hybrid model
            elif hasattr(layer, "attn"):
                self.attention_layers.append(layer.attn)
            elif hasattr(layer, "linear_attn"):
                self.attention_layers.append(layer.linear_attn)
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    self.attention_layers.append(layer.attention.attn)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                forward_batch.attn_backend = self.decode_attn_backend
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if not self.is_generation:
            kwargs["get_embedding"] = True

        can_run_graph = (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            return (
                self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs),
                can_run_graph,
            )

        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        return (
            self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            ),
            can_run_graph,
        )

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches are padded (batch_size > 0) for MLP sync.
        # in this case, we need to reinit the forward metadata, otherwise the stale
        # metadata causes batch_size mismatch in attention kernel(e.g. NSA Indexer).
        if forward_batch.batch_size > 0:
            self.attn_backend.init_forward_metadata(forward_batch)

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ret = self.model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (forward_batch.split_index, next_split_index),
        )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ) as recorder_outputs:
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            elastic_ep_state = ElasticEPStateManager.instance()
            if (
                elastic_ep_state is not None
                and not elastic_ep_state.is_active_equal_last()
            ):
                elastic_ep_state.snapshot_active_to_last()
                elastic_ep_state.sync_active_to_cpu()
                logging.info("EPLB due to rank faults")
                gen = self.eplb_manager.rebalance()
                while True:
                    try:
                        next(gen)
                    except StopIteration:
                        break
                output = self._forward_raw(
                    forward_batch,
                    skip_attn_backend_init,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        # Copy cached routing experts' buffers back to CPU cache
        get_global_experts_capturer().on_forward_end(
            forward_batch=forward_batch,
            can_run_graph=output.can_run_graph,
            cuda_graph_batch=getattr(self.graph_runner, "bs", None),
        )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Per-request layer capture: set aux-hidden-state capture selection on the model.
        # This reuses the existing EAGLE3-style mechanism (`set_eagle3_layers_to_capture`) when available.
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and hasattr(self.model, "set_eagle3_layers_to_capture")
        ):
            try:
                self.model.set_eagle3_layers_to_capture(
                    forward_batch.capture_layers or []
                )
            except TypeError:
                # Some models only support the default capture behavior.
                self.model.set_eagle3_layers_to_capture()

        mode_check = (
            forward_batch.forward_mode.is_cpu_graph
            if self.device == "cpu"
            else forward_batch.forward_mode.is_cuda_graph
        )
        can_run_graph = bool(
            mode_check()
            and self.graph_runner
            and self.graph_runner.can_run(forward_batch)
        )

        if can_run_graph:
            ret = self.graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

        # --- Heretic packed-MoE adapter context ---
        # Packed MoE injection needs to know which sequence each token belongs to and which
        # per-sequence `lora_id` is active. Provide this via a contextvar so MoE runner cores can
        # read it without invasive signature changes.
        _heretic_ctx_mgr = None
        try:
            if forward_batch.lora_ids is not None and forward_batch.batch_size > 0:
                from sglang.srt.layers.moe.heretic_packed_context import (
                    HereticPackedMoEContext,
                    set_ctx as _heretic_set_ctx,
                )

                # Build token_to_seq mapping for the flattened token batch.
                # For decode, seq_lens are 1 so this is cheap.
                if forward_batch.seq_lens_cpu is not None:
                    seq_lens_cpu = [int(x) for x in forward_batch.seq_lens_cpu]
                else:
                    seq_lens_cpu = [int(x) for x in forward_batch.seq_lens.detach().cpu().tolist()]

                # total tokens should match flattened batch dimension.
                # Use int32 for indexing; token_to_seq lives on the model device.
                token_to_seq_cpu = torch.repeat_interleave(
                    torch.arange(
                        int(forward_batch.batch_size), dtype=torch.int32, device="cpu"
                    ),
                    torch.tensor(seq_lens_cpu, dtype=torch.int64, device="cpu"),
                )
                token_to_seq = token_to_seq_cpu.to(self.device, non_blocking=True)

                ctx = HereticPackedMoEContext(
                    token_to_seq=token_to_seq,
                    seq_lora_ids=[(str(x) if x is not None else None) for x in forward_batch.lora_ids],
                )
                _heretic_ctx_mgr = _heretic_set_ctx(ctx)
        except Exception:
            _heretic_ctx_mgr = None

        # Run forward inside packed-MoE context (if any).
        import contextlib as _contextlib
        with (_heretic_ctx_mgr if _heretic_ctx_mgr is not None else _contextlib.nullcontext()):
            # For MLP sync
            if forward_batch.global_num_tokens_cpu is not None:
                forward_batch.prepare_mlp_sync_batch(self)
            else:
                forward_batch.prepare_attn_tp_scatter_input(self)

            # Normalize num_token_non_padded to be local to this attention TP rank if needed.
            if (
                forward_batch.num_token_non_padded is not None
                and forward_batch.global_num_tokens_gpu is not None
                and require_gathered_buffer(self.server_args)
                and not is_nsa_enable_prefill_cp()
            ):
                forward_batch.adjust_num_token_non_padded_for_attn_tp(
                    server_args=self.server_args,
                )

            if forward_batch.forward_mode.is_decode():
                ret = self.forward_decode(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_split_prefill():
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                ret, can_run_graph = self.forward_extend(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        if (
            forward_batch.global_num_tokens_cpu is not None
            and self.pp_group.is_last_rank
        ):
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        self._preprocess_logits(logits_output, forward_batch.sampling_info)
        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(
            self.model_config.hf_text_config, "rope_parameters", None
        ) or getattr(self.model_config.hf_text_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        is_mrope_enabled = "mrope_section" in rope_scaling
        return is_mrope_enabled

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
