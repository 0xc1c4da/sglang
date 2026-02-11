from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ComputeVTWBatchReqInput,
    ComputeVTWBatchReqOutput,
    ComputeVTWReqInput,
    ComputeVTWReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HereticBuildFullRownormLoraReqInput,
    HereticBuildFullRownormLoraReqOutput,
    HereticBuildPackedW2FullRownormReqInput,
    HereticBuildPackedW2FullRownormReqOutput,
    HereticModuleMapReqInput,
    HereticModuleMapReqOutput,
    HereticUnloadPackedMoEAdapterReqInput,
    HereticUnloadPackedMoEAdapterReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def heretic_module_map(self: Scheduler, recv_req: HereticModuleMapReqInput):
        modules = self.tp_worker.heretic_module_map(recv_req)
        return HereticModuleMapReqOutput(modules=modules)

    def heretic_build_full_rownorm_lora(
        self: Scheduler, recv_req: HereticBuildFullRownormLoraReqInput
    ):
        payload = self.tp_worker.heretic_build_full_rownorm_lora(recv_req)
        return HereticBuildFullRownormLoraReqOutput(
            name=payload.get("name", recv_req.name),
            lora_A_b64=payload.get("lora_A_b64", ""),
            lora_B_b64=payload.get("lora_B_b64", ""),
            lora_A_shape=payload.get("lora_A_shape", []),
            lora_B_shape=payload.get("lora_B_shape", []),
            dtype=payload.get("dtype", "error"),
        )

    def heretic_build_packed_w2_full_rownorm(
        self: Scheduler, recv_req: HereticBuildPackedW2FullRownormReqInput
    ):
        payload = self.tp_worker.heretic_build_packed_w2_full_rownorm(recv_req)
        return HereticBuildPackedW2FullRownormReqOutput(
            success=bool(payload.get("success", False)),
            message=str(payload.get("message", "")),
            lora_id=str(payload.get("lora_id", recv_req.lora_id)),
            name=str(payload.get("name", recv_req.name)),
            dtype=str(payload.get("dtype", "error")),
            num_local_experts=int(payload.get("num_local_experts", -1)),
            lora_A_shape=list(payload.get("lora_A_shape", [])),
            lora_B_shape=list(payload.get("lora_B_shape", [])),
        )

    def heretic_unload_packed_moe_adapter(
        self: Scheduler, recv_req: HereticUnloadPackedMoEAdapterReqInput
    ):
        payload = self.tp_worker.heretic_unload_packed_moe_adapter(recv_req)
        return HereticUnloadPackedMoEAdapterReqOutput(
            success=bool(payload.get("success", False)),
            message=str(payload.get("message", "")),
            lora_id=str(payload.get("lora_id", recv_req.lora_id)),
        )

    def compute_vtw(self: Scheduler, recv_req: ComputeVTWReqInput):
        vtw, implementation = self.tp_worker.compute_vtw(recv_req)
        return ComputeVTWReqOutput(
            name=recv_req.name, vtw=vtw, implementation=implementation
        )

    def compute_vtw_batch(self: Scheduler, recv_req: ComputeVTWBatchReqInput):
        results = self.tp_worker.compute_vtw_batch(recv_req)
        return ComputeVTWBatchReqOutput(results=results)
    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self._is_no_request()
        ), "release_memory_occupation should be called only when no ongoing request."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
