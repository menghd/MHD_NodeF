# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Utils (MHD-Utils) - V4
Author: Souray Meng (孟号丁)
Utility Tools: Dataset, Training, Monitoring for MHD Framework V4
License: MIT
"""

import torch
import torch.distributed as dist
import random
import numpy as np
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Tuple, Callable, Any, Union, Optional, Sequence, Set, Mapping
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime
from tqdm import tqdm
import json
import warnings
from contextlib import nullcontext

from .MHD_Framework_V5 import (
    MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph,
)


# ===================== 单卡 / DDP 执行适配 =====================

@dataclass(frozen=True)
class MHD_DistributedContext:
    """One-process-per-device execution metadata for MHD graphs."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    device_mesh: Any = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


@dataclass(frozen=True)
class MHD_ParallelConfig:
    """Optional execution choices; every feature is disabled by default.

    The hypergraph API never depends on this class. It is consumed only when a
    user asks the existing training utilities to prepare distributed execution.
    """

    data_parallel: str = "none"  # none | ddp | fsdp2
    tensor_parallel_plan: Optional[Mapping[str, Any]] = None
    tensor_parallel_size: int = 1
    pipeline_stages: Optional[Mapping[str, int]] = None
    pipeline_loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None
    pipeline_size: int = 1
    pipeline_microbatches: int = 1
    pipeline_schedule: str = "gpipe"  # gpipe | 1f1b
    compile: bool = False
    compile_dynamic: Optional[bool] = None
    compile_backend: Optional[Union[str, Callable]] = None
    fsdp_reshard_after_forward: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.data_parallel not in {"none", "ddp", "fsdp2"}:
            raise ValueError("data_parallel 必须是 none、ddp 或 fsdp2")
        if self.tensor_parallel_size < 1 or self.pipeline_size < 1:
            raise ValueError("TP/PP 并行度必须大于等于 1")
        if self.pipeline_microbatches < 1:
            raise ValueError("pipeline_microbatches 必须大于等于 1")
        if self.pipeline_schedule not in {"gpipe", "1f1b"}:
            raise ValueError("pipeline_schedule 必须是 gpipe 或 1f1b")
        active_families = sum(
            (
                self.data_parallel != "none",
                self.tensor_parallel_size > 1,
                self.pipeline_size > 1,
            )
        )
        if active_families > 1:
            raise ValueError("V4 每次运行只允许 DDP/FSDP2、TP、PP 中一个并行族")
        if self.tensor_parallel_size > 1 and not self.tensor_parallel_plan:
            raise ValueError("启用 Tensor Parallel 时必须显式提供 tensor_parallel_plan")
        if self.pipeline_size > 1 and not self.pipeline_stages:
            raise ValueError("启用 Pipeline Parallel 时必须显式提供 pipeline_stages")


def initialize_mhd_distributed(backend: Optional[str] = None) -> MHD_DistributedContext:
    """Initialize torch.distributed from torchrun environment variables when needed."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        kwargs = {"backend": selected_backend, "init_method": "env://"}
        if device.type == "cuda" and selected_backend == "nccl":
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return MHD_DistributedContext(rank, local_rank, world_size, device, selected_backend)


def destroy_mhd_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def mhd_barrier(context: MHD_DistributedContext) -> None:
    if context.distributed:
        if context.device.type == "cuda" and context.backend == "nccl":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


def mhd_all_reduce_mean(value: torch.Tensor, context: MHD_DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= context.world_size
    return result


class _MHD_GraphAdapter(nn.Module):
    """Expose an MHD Graph through the standard ``nn.Module`` call interface."""

    def __init__(
        self,
        graph: MHD_Graph,
        input_nodes: Sequence[str],
        output_nodes: Sequence[str],
        levels: Sequence[int],
    ) -> None:
        super().__init__()
        self.graph = graph
        self.input_nodes = tuple(input_nodes)
        self.output_nodes = tuple(output_nodes)
        self.levels = tuple(
            graph._validate_levels(levels, graph.num_levels, "Graph adapter")
        )
        missing = [name for name in (*self.input_nodes, *self.output_nodes) if graph.get_node_by_name(name) is None]
        if missing:
            raise ValueError(f"Unknown MHD adapter nodes: {missing}")
        self._input_node_objects = tuple(graph.get_node_by_name(name) for name in self.input_nodes)
        self._output_node_objects = tuple(graph.get_node_by_name(name) for name in self.output_nodes)
        self._all_node_objects = tuple(graph._nodes_in_id_order)
    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        absent = set(self.input_nodes) - set(input_dict)
        if absent:
            raise ValueError(f"Missing MHD inputs: {sorted(absent)}")
        for node in self._all_node_objects:
            node.feature_message.reset()
        for name, node in zip(self.input_nodes, self._input_node_objects):
            node.feature_message.current_state = input_dict[name]
        self.graph.forward(levels=self.levels)
        return {
            name: node.feature_message.current_state
            for name, node in zip(self.output_nodes, self._output_node_objects)
        }


def get_mhd_module_fqns(graph: MHD_Graph, adapter_prefix: str = "graph.") -> Dict[str, str]:
    """Return stable ``edge_name:operation_index -> module FQN`` mappings."""
    name_by_identity = {id(module): name for name, module in graph.edge_module_map.items()}
    result: Dict[str, str] = {}
    for edge in sorted(graph.edges, key=lambda item: item.id):
        for index, operation in enumerate(edge.edge_operations):
            module_name = name_by_identity.get(id(operation.function))
            if module_name is not None:
                result[f"{edge.name}:{index}"] = f"{adapter_prefix}edge_module_map.{module_name}"
    return result


def _normalize_tp_plan(graph: MHD_Graph, plan: Mapping[str, Any]) -> Dict[str, Any]:
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    aliases = get_mhd_module_fqns(graph)
    simple_styles = {
        "colwise": lambda: ColwiseParallel(output_layouts=Replicate()),
        "rowwise": lambda: RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
        ),
    }
    normalized: Dict[str, Any] = {}
    for name, style in plan.items():
        if isinstance(style, str):
            try:
                style = simple_styles[style.lower()]()
            except KeyError as exc:
                raise ValueError("TP 方案字符串只能是 colwise 或 rowwise") from exc
        resolved_name = name
        for alias, module_name in aliases.items():
            if name == alias or name.startswith(alias + "."):
                resolved_name = module_name + name[len(alias):]
                break
        normalized[resolved_name] = style
    return normalized


def _precision_dtype(precision: str, device: torch.device) -> Optional[torch.dtype]:
    normalized = precision.lower()
    if normalized in {"fp32", "float32", "none"}:
        return None
    if normalized in {"bf16", "bfloat16"}:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16")
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError("precision 必须是 fp32、bf16 或 fp16")


def _build_device_mesh(
    context: MHD_DistributedContext,
    config: MHD_ParallelConfig,
) -> Tuple[Any, int]:
    if context.world_size == 1:
        if config.tensor_parallel_size > 1 or config.pipeline_size > 1:
            raise ValueError("单进程不能启用 TP 或 PP")
        return None, 1
    if config.data_parallel == "ddp":
        return None, context.world_size
    if config.data_parallel != "none":
        dimension_name, requested_size = "dp", context.world_size
        data_parallel_size = context.world_size
    elif config.tensor_parallel_size > 1:
        dimension_name, requested_size = "tp", config.tensor_parallel_size
        data_parallel_size = 1
    elif config.pipeline_size > 1:
        dimension_name, requested_size = "pp", config.pipeline_size
        data_parallel_size = 1
    else:
        raise ValueError("WORLD_SIZE>1 时必须显式启用一个独立并行族")
    if requested_size != context.world_size:
        raise ValueError(
            f"独立 {dimension_name.upper()} 模式要求并行度 {requested_size} "
            f"等于 WORLD_SIZE={context.world_size}"
        )
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh(
        context.device.type,
        (requested_size,),
        mesh_dim_names=(dimension_name,),
    )
    return mesh, data_parallel_size


def prepare_mhd_model(
    graph: MHD_Graph,
    input_nodes: Sequence[str],
    output_nodes: Sequence[str],
    *,
    levels: Sequence[int],
    backward_levels: Optional[Sequence[int]] = None,
    parallel: Optional[MHD_ParallelConfig] = None,
    context: Optional[MHD_DistributedContext] = None,
    precision: str = "fp32",
    example_inputs: Optional[Mapping[str, torch.Tensor]] = None,
) -> nn.Module:
    """Prepare the same MHD graph for optional native PyTorch execution choices.

    With no optional arguments this returns a private standard-Module bridge. Advanced
    choices are applied in place before optimizer construction.
    """
    config = parallel or MHD_ParallelConfig()
    context = context or initialize_mhd_distributed()
    graph.to(context.device)
    adapter: nn.Module = _MHD_GraphAdapter(graph, input_nodes, output_nodes, levels)
    mesh, data_parallel_size = _build_device_mesh(context, config)

    if config.pipeline_size > 1:
        return _prepare_mhd_pipeline(
            adapter,
            graph,
            config,
            context,
            mesh,
            input_nodes,
            output_nodes,
            backward_levels,
            example_inputs,
            precision,
        )

    if config.tensor_parallel_size > 1:
        from torch.distributed.tensor.parallel import parallelize_module

        adapter = parallelize_module(
            adapter,
            device_mesh=mesh["tp"],
            parallelize_plan=_normalize_tp_plan(graph, config.tensor_parallel_plan or {}),
        )

    if config.data_parallel == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        dtype = _precision_dtype(precision, context.device)
        policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype, output_dtype=None)
        fsdp_mesh = mesh["dp"] if mesh is not None else None
        fully_shard(
            adapter,
            mesh=fsdp_mesh,
            reshard_after_forward=config.fsdp_reshard_after_forward,
            mp_policy=policy,
        )

    if config.compile:
        compile_kwargs = {"dynamic": config.compile_dynamic}
        if config.compile_backend is not None:
            compile_kwargs["backend"] = config.compile_backend
        adapter = torch.compile(adapter, **compile_kwargs)

    if config.data_parallel == "ddp" and data_parallel_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        adapter = DistributedDataParallel(
            adapter,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            process_group=mesh["dp"].get_group() if mesh is not None else None,
        )
    return adapter


class _MHD_PipelineModel(nn.Module):
    """Thin holder for a native pipeline schedule and its rank-local stage module."""

    def __init__(
        self,
        train_schedule: Any,
        inference_schedule: Any,
        stage_module: nn.Module,
        stage_index: int,
        num_stages: int,
        pipeline_group: Any,
        last_stage_rank: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.stage_module = stage_module
        self.train_schedule = train_schedule
        self.inference_schedule = inference_schedule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.pipeline_group = pipeline_group
        self.last_stage_rank = last_stage_rank
        self.device = device

    def forward(self, *args, target=None, losses=None, **kwargs):
        schedule = self.train_schedule if self.training else self.inference_schedule
        stage = getattr(self.stage_module, "_orig_mod", self.stage_module)
        if hasattr(stage, "begin_execution"):
            stage.begin_execution()
        result = schedule.step(*args, target=target, losses=losses, **kwargs)
        if hasattr(stage, "finish_execution"):
            stage.finish_execution()
        return result

    def synchronize_last_stage_scalar(self, value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Broadcast one detached scalar from the last stage along this PP replica."""
        if value is None:
            scalar = torch.zeros((), device=self.device)
        else:
            scalar = value.detach().float().mean()
        dist.broadcast(scalar, src=self.last_stage_rank, group=self.pipeline_group)
        return scalar


def _embedded_pipeline_loss(output: Any, _target: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError("Pipeline 最终 stage 必须输出 Tensor 或以 Tensor 为首项的序列")
    return value.mean()


class _MHD_AutoPipelineStage(nn.Module):
    """Rank-local stage generated from MHD topology and edge assignments."""

    def __init__(
        self,
        graph: MHD_Graph,
        step_records: Sequence[Tuple[Any, bool]],
        input_names: Sequence[str],
        output_names: Sequence[str],
        selected_node_ids: Set[int],
    ) -> None:
        super().__init__()
        self.step_records = tuple(step_records)
        self.steps = tuple(step for step, _ in self.step_records)
        self.input_names = tuple(input_names)
        self.output_names = tuple(output_names)
        self.nodes_by_id = tuple(graph._nodes_in_id_order)
        self.node_ids_by_name = {node.name: node.id for node in self.nodes_by_id}
        self.selected_node_ids = set(selected_node_ids)
        self._gradient_captures: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._boundary_captures: Dict[int, List[torch.Tensor]] = defaultdict(list)
        relevant_modules: Dict[str, nn.Module] = {}
        seen: Set[int] = set()
        for step in self.steps:
            for operation_index, operation in enumerate(step.edge.edge_operations):
                function = operation.function
                if isinstance(function, nn.Module) and id(function) not in seen:
                    seen.add(id(function))
                    relevant_modules[f"e{step.edge_id}_o{operation_index}"] = function
        self.edge_modules = nn.ModuleDict(relevant_modules)
        self.selected_parameter_ids = {
            id(parameter)
            for step, selected in self.step_records
            if selected
            for _, parameter in step.edge.named_edge_parameters()
        }

    def begin_execution(self) -> None:
        self._gradient_captures.clear()
        self._boundary_captures.clear()
        for node_id in self.selected_node_ids:
            self.nodes_by_id[node_id].gradient_message.reset()

    @staticmethod
    def _combine_gradients(values: Sequence[torch.Tensor]) -> torch.Tensor:
        detached = [value.detach() for value in values]
        if len(detached) == 1:
            return detached[0].clone()
        if all(
            value.ndim > 0 and value.shape[1:] == detached[0].shape[1:]
            for value in detached
        ):
            return torch.cat(detached, dim=0)
        return torch.stack(detached).sum(dim=0)

    def finish_execution(self) -> None:
        for node_id in self.selected_node_ids:
            values = (
                self._boundary_captures.get(node_id)
                or self._gradient_captures.get(node_id)
            )
            if values:
                self.nodes_by_id[node_id].gradient_message.current_state = (
                    self._combine_gradients(values)
                )
        for parameter in self.parameters():
            if id(parameter) not in self.selected_parameter_ids:
                parameter.grad = None

    def forward(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"Pipeline stage 需要 {len(self.input_names)} 个输入，实际 {len(inputs)}"
            )
        state: Dict[int, torch.Tensor] = {
            self.node_ids_by_name[name]: tensor
            for name, tensor in zip(self.input_names, inputs)
        }
        pending: Dict[int, List[torch.Tensor]] = defaultdict(list)

        def value(node_id: int) -> torch.Tensor:
            if node_id not in state:
                state[node_id] = self.nodes_by_id[
                    node_id
                ].feature_message.initial_state
            return state[node_id]

        def flush(node_id: int) -> None:
            incomings = pending.pop(node_id, None)
            if incomings:
                state[node_id] = self.nodes_by_id[node_id].aggregate_messages(
                    value(node_id), incomings
                )

        for step, selected in self.step_records:
            for node_id in step.head_ids:
                flush(node_id)
            head_tensors = [value(node_id) for node_id in step.head_ids]
            outputs = step.edge.execute_edge_operations(head_tensors)
            outputs = [
                output.view_as(output) if output.requires_grad else output
                for output in outputs
            ]
            if len(outputs) != len(step.tail_ids):
                raise ValueError(f"Pipeline 边 '{step.edge.name}' 输出数量不匹配")
            for node_id, output in zip(step.tail_ids, outputs):
                if output.requires_grad and not selected:
                    def block(gradient, target=node_id):
                        self._boundary_captures[target].append(gradient.detach())
                        return torch.zeros_like(gradient)
                    output.register_hook(block)
                pending[node_id].append(output)
        for node_id in sorted(pending):
            flush(node_id)
        for node_id in self.selected_node_ids:
            tensor = state.get(node_id)
            if tensor is not None and tensor.requires_grad:
                tensor.retain_grad()
                tensor.register_hook(
                    lambda gradient, target=node_id: self._gradient_captures[
                        target
                    ].append(gradient.detach())
                )
        outputs = tuple(value(self.node_ids_by_name[name]) for name in self.output_names)
        return outputs[0] if len(outputs) == 1 else outputs


def _build_auto_pipeline_stages(
    graph: MHD_Graph,
    config: MHD_ParallelConfig,
    input_nodes: Sequence[str],
    output_nodes: Sequence[str],
    levels: Sequence[int],
    backward_levels: Sequence[int],
) -> List[nn.Module]:
    normalized_levels = graph._validate_levels(levels, graph.num_levels, "Pipeline Forward")
    flattened_steps = [
        step
        for level in normalized_levels
        for step in graph._execution_plan_per_level[level]
    ]
    normalized_backward = graph._validate_levels(
        backward_levels, graph.num_levels, "Pipeline Backward"
    )
    overlap = sorted(set(normalized_levels).intersection(normalized_backward))
    if overlap:
        raise ValueError(f"Pipeline 前后向 levels 不得重叠: {overlap}")
    reachable = {graph.get_node_by_name(name).id for name in output_nodes}
    unmatched = list(range(len(flattened_steps)))
    selected_positions: List[int] = []
    selected_node_ids: Set[int] = set(reachable)
    for level in normalized_backward:
        for reverse_step in graph._execution_plan_per_level[level]:
            match = None
            for position in reversed(unmatched):
                forward_step = flattened_steps[position]
                if (
                    forward_step.edge_id == reverse_step.edge_id
                    and forward_step.tail_ids == reverse_step.head_ids
                    and forward_step.head_ids == reverse_step.tail_ids
                ):
                    match = position
                    break
            if match is None:
                raise ValueError(
                    f"Pipeline Backward level {level} 的边 '{reverse_step.edge.name}' "
                    "没有兼容的 Forward occurrence"
                )
            if not any(node_id in reachable for node_id in reverse_step.head_ids):
                raise ValueError(
                    f"Pipeline Backward 边 '{reverse_step.edge.name}' 在给定顺序中不可达"
                )
            unmatched.remove(match)
            selected_positions.append(match)
            selected_node_ids.update(reverse_step.head_ids)
            selected_node_ids.update(reverse_step.tail_ids)
            reachable.update(reverse_step.tail_ids)
    if not selected_positions:
        raise ValueError("Pipeline backward_levels 没有选择任何 Forward occurrence")
    selected_position_set = set(selected_positions)
    stage_map = dict(config.pipeline_stages or {})
    edge_names = {step.edge.name for step in flattened_steps}
    unknown = set(stage_map) - edge_names
    missing = edge_names - set(stage_map)
    if unknown or missing:
        raise ValueError(
            f"pipeline_stages 未知边={sorted(unknown)} 缺少边={sorted(missing)}"
        )
    invalid = {
        name: stage for name, stage in stage_map.items()
        if not isinstance(stage, int) or stage < 0 or stage >= config.pipeline_size
    }
    if invalid:
        raise ValueError(f"非法 Pipeline stage 编号: {invalid}")
    stage_sequence = [stage_map[step.edge.name] for step in flattened_steps]
    if stage_sequence != sorted(stage_sequence):
        raise ValueError("pipeline_stages 必须与拓扑执行顺序单调一致")
    steps_by_stage = [
        [
            (step, position in selected_position_set)
            for position, step in enumerate(flattened_steps)
            if stage_map[step.edge.name] == stage
        ]
        for stage in range(config.pipeline_size)
    ]
    if any(not steps for steps in steps_by_stage):
        raise ValueError("每个 Pipeline stage 至少需要一条超边")

    node_name = {node.id: node.name for node in graph.nodes}
    introduced: Dict[str, int] = {name: -1 for name in input_nodes}
    consumers: Dict[str, Set[int]] = defaultdict(set)
    for stage, records in enumerate(steps_by_stage):
        for step, _ in records:
            for node_id in step.head_ids:
                consumers[node_name[node_id]].add(stage)
            for node_id in step.tail_ids:
                introduced.setdefault(node_name[node_id], stage)
    for name in output_nodes:
        consumers[name].add(config.pipeline_size)
    boundaries: List[Tuple[str, ...]] = []
    for boundary in range(config.pipeline_size - 1):
        names = sorted(
            name
            for name, source_stage in introduced.items()
            if source_stage <= boundary
            and any(consumer_stage > boundary for consumer_stage in consumers.get(name, set()))
        )
        if not names:
            raise ValueError(f"Pipeline boundary {boundary} 没有可传递的 Message")
        boundaries.append(tuple(names))
    stage_inputs = [tuple(input_nodes), *boundaries]
    stage_outputs = [*boundaries, tuple(output_nodes)]
    return [
        _MHD_AutoPipelineStage(
            graph,
            steps_by_stage[stage],
            stage_inputs[stage],
            stage_outputs[stage],
            selected_node_ids,
        )
        for stage in range(config.pipeline_size)
    ]


def _prepare_mhd_pipeline(
    adapter: nn.Module,
    graph: MHD_Graph,
    config: MHD_ParallelConfig,
    context: MHD_DistributedContext,
    mesh: Any,
    input_nodes: Sequence[str],
    output_nodes: Sequence[str],
    backward_levels: Optional[Sequence[int]],
    example_inputs: Optional[Mapping[str, torch.Tensor]],
    precision: str,
) -> nn.Module:
    """Build native PP stages directly from MHD Edge/Topo assignments."""
    if backward_levels is None:
        raise ValueError("Pipeline Parallel 必须显式提供 backward_levels")
    stage_modules = _build_auto_pipeline_stages(
        graph,
        config,
        input_nodes,
        output_nodes,
        adapter.levels,
        backward_levels,
    )
    supplied = dict(example_inputs or {})
    stage_values: Dict[str, torch.Tensor] = {
        name: supplied.get(
            name,
            graph.get_node_by_name(name).feature_message.current_state,
        ).to(context.device)
        for name in input_nodes
    }
    stage_example_args: List[Tuple[torch.Tensor, ...]] = []
    with torch.no_grad():
        for module in stage_modules:
            args = tuple(stage_values[name] for name in module.input_names)
            stage_example_args.append(args)
            output = module.to(context.device)(*args)
            values = output if isinstance(output, tuple) else (output,)
            stage_values = dict(zip(module.output_names, values))
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("当前 rank 不属于 MHD DeviceMesh")
    stage_index = int(coordinate[0])
    stage_module: nn.Module = stage_modules[stage_index].to(context.device)
    if config.compile:
        compile_kwargs = {"dynamic": config.compile_dynamic}
        if config.compile_backend is not None:
            compile_kwargs["backend"] = config.compile_backend
        stage_module = torch.compile(stage_module, **compile_kwargs)
    full_example_args = stage_example_args[stage_index]
    for tensor in full_example_args:
        if tensor.ndim > 0 and tensor.shape[0] % config.pipeline_microbatches != 0:
            raise ValueError("Pipeline 样例 batch 必须能被 pipeline_microbatches 整除")
    example_args = tuple(
        tensor.chunk(config.pipeline_microbatches, dim=0)[0]
        if tensor.ndim > 0 and config.pipeline_microbatches > 1
        else tensor
        for tensor in full_example_args
    )
    # PipelineStage 需要用样例张量建立前向与反向通信元数据。流水线中间
    # 激活在运行时是可微的，即便用户给出的纯形状样例默认不带梯度；因此这里
    # 只为元数据推导创建可微副本，既不改变调用方张量，也不保留初始化图。
    metadata_example_args = tuple(
        tensor.detach().requires_grad_(tensor.is_floating_point() or tensor.is_complex())
        for tensor in example_args
    )
    precision_dtype = _precision_dtype(precision, context.device)
    metadata_autocast = (
        torch.autocast(context.device.type, dtype=precision_dtype)
        if precision_dtype is not None
        else nullcontext()
    )
    with metadata_autocast:
        example_output = stage_module(*metadata_example_args)
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe
    from torch.distributed.pipelining.microbatch import sum_reducer

    pipeline_group = mesh["pp"].get_group()

    def build_stage() -> Any:
        return PipelineStage(
            stage_module,
            stage_index,
            config.pipeline_size,
            context.device,
            input_args=metadata_example_args,
            output_args=example_output,
            group=pipeline_group,
        )

    schedule_type = ScheduleGPipe if config.pipeline_schedule == "gpipe" else Schedule1F1B
    merge_spec = (
        sum_reducer
        if isinstance(example_output, torch.Tensor) and example_output.ndim == 0
        else None
    )
    train_schedule = schedule_type(
        build_stage(),
        config.pipeline_microbatches,
        loss_fn=config.pipeline_loss_fn or _embedded_pipeline_loss,
        output_merge_spec=merge_spec,
    )
    inference_schedule = ScheduleGPipe(
        build_stage(),
        config.pipeline_microbatches,
        output_merge_spec=merge_spec,
    )
    pipeline_ranks = dist.get_process_group_ranks(pipeline_group)
    return _MHD_PipelineModel(
        train_schedule,
        inference_schedule,
        stage_module,
        stage_index,
        config.pipeline_size,
        pipeline_group,
        pipeline_ranks[-1],
        context.device,
    )


def unwrap_mhd_graph(module: nn.Module) -> MHD_Graph:
    candidate: Any = module
    visited: Set[int] = set()
    while isinstance(candidate, nn.Module) and id(candidate) not in visited:
        visited.add(id(candidate))
        if isinstance(candidate, _MHD_GraphAdapter):
            return candidate.graph
        for attribute in ("module", "_orig_mod", "stage_module"):
            child = getattr(candidate, attribute, None)
            if isinstance(child, nn.Module):
                candidate = child
                break
        else:
            break
    raise TypeError(f"无法从 {type(module)!r} 解包 MHD_Graph")

# ===================== 状态保存/加载工具函数 =====================

def updown_node(nodes: Set[MHD_Node], path: str, mode: str, target_device: torch.device = None) -> None:
    """
    节点特征图保存/加载函数

    特性：
    1. 支持节点初始状态和当前状态的保存/加载
    2. 自动处理设备迁移
    3. 保存元信息用于验证

    Args:
        nodes: 节点集合
        path: 文件路径
        mode: 模式，'up'为加载，'down'为保存
        target_device: 目标设备（仅加载时有效）

    Raises:
        ValueError: 当模式参数错误时
    """
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        save_dict = {
            "node_messages": {
                n.name: {
                    "feature_message": {
                        "initial_state": n.feature_message.initial_state,
                        "current_state": n.feature_message.current_state,
                    },
                    "gradient_message": {
                        "initial_state": n.gradient_message.initial_state,
                        "current_state": n.gradient_message.current_state,
                    },
                }
                for n in sorted(nodes, key=lambda x: x.id)
            },
            "node_info": {
                n.name: {
                    "id": n.id,
                    "shape": n.feature_message.current_state.shape,
                    "dtype": str(n.feature_message.current_state.dtype),
                    "device": str(n.feature_message.current_state.device)
                }
                for n in sorted(nodes, key=lambda x: x.id)
            }
        }
        torch.save(save_dict, path)

    else:
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        messages = load_dict.get("node_messages")
        if messages is not None:
            feature_initial = {
                name: value["feature_message"]["initial_state"]
                for name, value in messages.items()
            }
            feature_current = {
                name: value["feature_message"]["current_state"]
                for name, value in messages.items()
            }
            gradient_initial = {
                name: value["gradient_message"]["initial_state"]
                for name, value in messages.items()
            }
            gradient_current = {
                name: value["gradient_message"]["current_state"]
                for name, value in messages.items()
            }
        else:
            feature_initial = load_dict.get(
                "feature_message_initial_states",
                load_dict.get("node_initial_states", {}),
            )
            feature_current = load_dict.get(
                "feature_message_current_states",
                load_dict.get("node_current_states", {}),
            )
            gradient_initial = load_dict.get("gradient_message_initial_states", {})
            gradient_current = load_dict.get("gradient_message_current_states", {})

        def move(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            destination = target_device or reference.device
            return tensor.to(destination, dtype=reference.dtype, non_blocking=True)

        for node in sorted(nodes, key=lambda x: x.id):
            if node.name in feature_initial:
                node.feature_message.initial_state = move(
                    feature_initial[node.name], node.feature_message.initial_state
                )
            if node.name in feature_current:
                node.feature_message.current_state = move(
                    feature_current[node.name], node.feature_message.current_state
                )
            else:
                node.feature_message.current_state = (
                    node.feature_message.initial_state.clone()
                )
            if node.name in gradient_initial:
                node.gradient_message.initial_state = move(
                    gradient_initial[node.name], node.gradient_message.initial_state
                )
            if node.name in gradient_current:
                node.gradient_message.current_state = move(
                    gradient_current[node.name], node.gradient_message.current_state
                )
            else:
                node.gradient_message.current_state = (
                    node.gradient_message.initial_state.clone()
                )
            node.gradient_message.validate()
            node.feature_message.validate()

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = (
        sum(1 for n in nodes if n.name in feature_initial)
        if mode == 'up'
        else len(nodes)
    )

    print(f"📊 节点特征图{mode_cn}完成 (Node {mode_en} completed) 处理: {processed}/{len(nodes)}")
    if target_device and mode == 'up':
        print(f" ├─ 目标设备: {target_device}")
    print(f" 📁 路径: {path}")


def updown_edge(edges: Set[MHD_Edge], path: str, mode: str, target_device: torch.device = None) -> None:
    """
    超边可学习模块保存/加载函数（适配 edge_operations）

    特性：
    1. 支持边中可学习参数的保存/加载
    2. 自动处理nn.Module状态字典
    3. 保存操作序列信息

    Args:
        edges: 边集合
        path: 文件路径
        mode: 模式，'up'为加载，'down'为保存
        target_device: 目标设备（仅加载时有效）

    Raises:
        ValueError: 当模式参数错误时
    """
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        # 保存模式
        save_dict = {
            "edge_params": {},
            "edge_info": {}
        }
        for edge in sorted(edges, key=lambda x: x.id):
            save_dict["edge_params"][edge.name] = [
                elem.function.state_dict()
                if isinstance(elem.function, nn.Module)
                else None
                for elem in edge.edge_operations
            ]
            save_dict["edge_info"][edge.name] = {
                "id": edge.id,
                "operations": [str(type(op.function)) for op in edge.edge_operations]
            }
        torch.save(save_dict, path)

    else:
        # 加载模式
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for edge in sorted(edges, key=lambda x: x.id):
            if edge.name in load_dict["edge_params"]:
                saved_params = load_dict["edge_params"][edge.name]
                for idx, elem in enumerate(edge.edge_operations):
                    function = elem.function
                    if idx < len(saved_params) and isinstance(function, nn.Module) and saved_params[idx] is not None:
                        function.load_state_dict(saved_params[idx])
                        # 迁移到目标设备
                        if target_device is not None:
                            elem.to_device(target_device)

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for e in edges if e.name in (load_dict["edge_params"] if mode == 'up' else [e.name for e in edges]))

    print(f"📊 超边可学习参数{mode_cn}完成 (Edge {mode_en} completed) 处理: {processed}/{len(edges)}")
    if target_device and mode == 'up':
        print(f" ├─ 目标设备: {target_device}")
    print(f" 📁 路径: {path}")


# ===================== 统一数据集类 =====================

class MHD_Dataset(Dataset):
    """
    高扩展MHD数据集（支持 shared_loader）

    特性：
    1. 支持 shared_loader 一次性加载所有节点数据，避免重复 I/O
    2. 增强在 loader 内部完成，框架不再提供增强器
    3. 统一设备管理
    """

    def __init__(
        self,
        sample_info_list: List[Any],
        node_configs: Dict[str, Dict],
        base_seed: int = 42,
        target_device: torch.device = None,
        shared_loader: Optional[Callable[[Any], Dict[str, Any]]] = None
    ):
        """
        初始化MHD数据集

        Args:
            sample_info_list: 样本信息列表
            node_configs: 节点配置字典，格式为{节点名: {loader: 加载函数}}
            base_seed: 基础随机种子
            target_device: 目标计算设备
            shared_loader: 可选，一次调用返回包含所有节点所需数据的字典
        """
        self.sample_info_list = sample_info_list
        self.node_configs = node_configs
        self.base_seed = base_seed
        self.target_device = target_device or torch.device("cpu")
        self.node_names = list(node_configs.keys())
        self.shared_loader = shared_loader

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_info_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        单样本加载流程

        Args:
            idx: 样本索引

        Returns:
            节点名到张量的映射字典
        """
        sample_info = self.sample_info_list[idx]

        # 获取worker信息（用于多进程随机种子）
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            sample_seed = self.base_seed + worker_id + idx
        else:
            sample_seed = self.base_seed + idx

        # 如果有共享加载器，则调用一次获取所有数据
        if self.shared_loader is not None:
            shared_data = self.shared_loader(sample_info)
        else:
            shared_data = None

        sample_data = {}
        for node_name, config in self.node_configs.items():
            load_fn = config["loader"]
            try:
                if shared_data is not None:
                    # loader 接收共享数据字典
                    tensor = load_fn(shared_data)
                else:
                    tensor = load_fn(sample_info)
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"节点 {node_name} 加载失败（idx={idx}）: {str(e)}")

            # 数据增强完全由 loader 内部负责，此处不做额外处理
            if tensor.device != self.target_device:
                tensor = tensor.to(self.target_device, non_blocking=True)
            sample_data[node_name] = tensor

        return sample_data

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批次拼接函数

        Args:
            batch: 批次数据列表

        Returns:
            拼接后的批次数据
        """
        if not batch:
            raise ValueError("空批次数据")
        ref_node_names = batch[0].keys()
        batch_data = {}
        for node_name in ref_node_names:
            node_samples = [sample[node_name] for sample in batch]
            batch_data[node_name] = torch.stack(node_samples, dim=0)
        return batch_data


# ===================== 数据加载器工具函数 =====================

def create_mhd_dataloader(
    dataset: MHD_Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = False,
    context: Optional[MHD_DistributedContext] = None,
    sampler: Optional[torch.utils.data.Sampler] = None,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """
    创建MHD数据加载器

    特性：
    1. 支持多进程数据加载及可复现性
    2. 统一设备管理
    """
    def worker_init_fn(worker_id: int) -> None:
        process_seed = dataset.base_seed + worker_id
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(process_seed)
            torch.cuda.manual_seed_all(process_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    worker_init_fn_to_use = worker_init_fn if num_workers > 0 else None
    if dataset.target_device.type != "cpu" and num_workers > 0:
        warnings.warn("多进程 DataLoader 应在 CPU 加载数据，由 Trainer 异步传输到设备")
    if sampler is None and context is not None and context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=MHD_Dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn_to_use,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


# ===================== 监控器类 =====================

class MHD_Monitor:
    """
    MHD训练监控器

    特性：
    1. 监控节点和边的统计信息
    2. 支持训练和验证模式的梯度监控
    3. 安全的张量统计计算
    4. 格式化输出
    """

    def __init__(
        self,
        monitor_nodes: list,
        monitor_edges: list = None,
        *,
        node_states: Sequence[str] = ("current_state",),
        statistics: Sequence[str] = ("mean", "sum", "min", "max"),
    ):
        self.monitor_nodes = monitor_nodes
        self.monitor_edges = monitor_edges
        aliases = {
            "current_state": "feature_message.current_state",
            "initial_state": "feature_message.initial_state",
        }
        self.node_states = tuple(aliases.get(state, state) for state in node_states)
        valid_states = {
            "feature_message.initial_state",
            "feature_message.current_state",
            "gradient_message.initial_state",
            "gradient_message.current_state",
        }
        unknown_states = set(self.node_states) - valid_states
        if unknown_states:
            raise ValueError(f"未知 Node Message State: {sorted(unknown_states)}")
        self.statistics = tuple(statistics)
        valid_statistics = {"mean", "sum", "min", "max"}
        unknown_statistics = set(self.statistics) - valid_statistics
        if unknown_statistics:
            raise ValueError(f"未知统计量: {sorted(unknown_statistics)}")
        self.records = defaultdict(list)
        self.step_counter = 0

    def reset(self):
        self.records = defaultdict(list)
        self.step_counter = 0

    def _safe_tensor_stats(self, tensor: torch.Tensor) -> dict:
        tensor = torch.nan_to_num(
            tensor.detach().float(), nan=0.0, posinf=1e6, neginf=-1e6
        )
        all_stats = {
            "mean": float(tensor.mean().item()),
            "sum": float(tensor.sum().item()),
            "max": float(tensor.max().item()),
            "min": float(tensor.min().item())
        }
        return {name: all_stats[name] for name in self.statistics}

    def monitor_node(self, mhd_graph, prefix: str = "") -> dict:
        node_metrics = {}
        for node_name in self.monitor_nodes:
            node = mhd_graph.get_node_by_name(node_name)
            if node is None:
                warnings.warn(f"监控节点 {node_name} 不存在，跳过")
                continue
            for state_name in self.node_states:
                message_name, attribute = state_name.split(".")
                value = getattr(getattr(node, message_name), attribute)
                stats = self._safe_tensor_stats(value)
                legacy_key = (
                    len(self.node_states) == 1
                    and state_name == "feature_message.current_state"
                )
                state_key = state_name.replace(".", "_")
                for stat_name, stat_value in stats.items():
                    if legacy_key:
                        key = f"{prefix}{node_name}_{stat_name}"
                    else:
                        key = f"{prefix}{node_name}_{state_key}_{stat_name}"
                    node_metrics[key] = stat_value
        for k, v in node_metrics.items():
            self.records[k].append(v)
        self.step_counter += 1
        return node_metrics

    def monitor_edge(self, mhd_graph, prefix: str = "", train_mode: bool = True) -> dict:
        edge_metrics = {}
        target_edges = self.monitor_edges or [e.name for e in mhd_graph.edges]
        for edge_name in target_edges:
            edge = mhd_graph.get_edge_by_name(edge_name)
            if edge is None:
                warnings.warn(f"监控边 {edge_name} 不存在，跳过")
                continue
            for idx, op in enumerate(edge.edge_operations):
                function = op.function
                if not isinstance(function, nn.Module):
                    continue
                weight_key = f"{prefix}{edge_name}_op{idx}"
                if hasattr(function, 'weight') and function.weight is not None:
                    weight = function.weight.detach()
                    weight_stats = self._safe_tensor_stats(weight)
                    edge_metrics[f"{weight_key}_weight_mean"] = weight_stats["mean"]
                    edge_metrics[f"{weight_key}_weight_l2"] = float(torch.norm(weight).item())
                if train_mode and hasattr(function, 'weight') and function.weight.grad is not None:
                    grad = function.weight.grad.detach()
                    grad_stats = self._safe_tensor_stats(grad)
                    edge_metrics[f"{weight_key}_grad_mean"] = grad_stats["mean"]
                    edge_metrics[f"{weight_key}_grad_l2"] = float(torch.norm(grad).item())
        for k, v in edge_metrics.items():
            self.records[k].append(v)
        return edge_metrics

    def get_mean_metrics(self, step_window: int = None) -> dict:
        mean_metrics = {}
        window = slice(-step_window, None) if step_window else slice(None)
        for metric_name, values in self.records.items():
            if len(values) == 0:
                mean_metrics[metric_name] = 0.0
            else:
                clean_values = [v for v in values[window] if not np.isnan(v) and not np.isinf(v)]
                mean_metrics[metric_name] = float(np.mean(clean_values)) if clean_values else 0.0
        return mean_metrics

    def format_metrics(self, metrics: dict, decimal: int = 6) -> str:
        formatted = []
        for k, v in sorted(metrics.items()):
            formatted.append(f"  {k}: {v:.{decimal}f}")
        return "\n".join(formatted)


def display_graph(mhd_graph: MHD_Graph, levels: Sequence[int]) -> str:
    """Display selected global levels as Mermaid source in the supplied order."""
    levels = mhd_graph._validate_levels(levels, mhd_graph.num_levels, "Display")
    mermaid = [
        "flowchart TD",
        "",
        " classDef MHD_Node_Style fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
        " classDef MHD_Edge_Style fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
        "",
    ]

    def escape_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    for node in sorted(mhd_graph.nodes, key=lambda item: item.id):
        mermaid.append(
            f' N{node.id}["{escape_label(node.name)}"]:::MHD_Node_Style'
        )
    for edge in sorted(mhd_graph.edges, key=lambda item: item.id):
        connections: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = defaultdict(list)
        for execution_index, level in enumerate(levels):
            role = mhd_graph.topo.role_matrices[level]
            sort = mhd_graph.topo.sort_matrices[level]
            for node_id, role_value in enumerate(role[edge.id].tolist()):
                if role_value == 0:
                    continue
                if role_value < 0:
                    source_id, target_id = node_id, -(edge.id + 1)
                else:
                    source_id, target_id = -(edge.id + 1), node_id
                connections[(source_id, target_id)].append(
                    (execution_index, level, int(sort[edge.id, node_id].item()))
                )
        if not connections:
            continue
        mermaid.append(
            f' E{edge.id}["{escape_label(edge.name)}"]:::MHD_Edge_Style'
        )
        for (source_id, target_id), order_values in sorted(
            connections.items(),
            key=lambda item: (
                0 if item[0][0] >= 0 else 1,
                item[0][0],
                item[0][1],
            ),
        ):
            source = f"E{-source_id - 1}" if source_id < 0 else f"N{source_id}"
            target = f"E{-target_id - 1}" if target_id < 0 else f"N{target_id}"
            order_text = ", ".join(
                f"#{execution_index}:L{level}:S{sort_value}"
                for execution_index, level, sort_value in order_values
            )
            mermaid.append(f" {source} -->|{order_text}| {target}")
        mermaid.append("")
    mermaid_code = "\n".join(mermaid)
    print("=== MHD Graph ===")
    print(mermaid_code)
    return mermaid_code


# ===================== 训练器类 =====================

def _infer_scalar_terminal_name(graph: MHD_Graph, levels: Sequence[int]) -> str:
    """Infer the unique scalar terminal produced by an explicit Forward path."""
    normalized = graph._validate_levels(levels, graph.num_levels, "Trainer Forward")
    last_produced: Dict[int, int] = {}
    last_consumed: Dict[int, int] = {}
    position = 0
    for level in normalized:
        for step in graph._execution_plan_per_level[level]:
            for node_id in step.head_ids:
                last_consumed[node_id] = position
            for node_id in step.tail_ids:
                last_produced[node_id] = position
            position += 1
    candidates = [
        graph.get_node_by_id(node_id)
        for node_id, produced_at in last_produced.items()
        if produced_at > last_consumed.get(node_id, -1)
        and graph.get_node_by_id(node_id).feature_message.initial_state.numel() == 1
    ]
    if len(candidates) != 1:
        raise ValueError(
            "forward_levels 必须静态产生唯一标量终点，"
            f"实际候选={[node.name for node in candidates]}"
        )
    return candidates[0].name


class MHD_Trainer:
    """Unified trainer for ordinary and optionally parallel MHD graphs.

    Existing V3 arguments retain their meaning. Every V4 execution choice is an
    optional keyword and defaults to the ordinary eager, single-device behavior.
    """

    def __init__(
        self,
        mhd_graph: MHD_Graph,
        optimizer: Union[torch.optim.Optimizer, Callable[[Any], torch.optim.Optimizer]],
        monitor: MHD_Monitor,
        forward_levels: Sequence[int],
        backward_levels: Sequence[int],
        criteria: Callable[[MHD_Graph], Union[torch.Tensor, float]],
        criteria_mode: str = 'min',
        save_dir: str = "./mhd_ckpts",
        grad_clip_norm: float = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        save_interval: int = 0,
        *,
        input_nodes: Optional[Sequence[str]] = None,
        input_mapping: Optional[Mapping[str, str]] = None,
        output_nodes: Optional[Sequence[str]] = None,
        parallel: Optional[MHD_ParallelConfig] = None,
        distributed_context: Optional[MHD_DistributedContext] = None,
        precision: str = "fp32",
        grad_accum_steps: int = 1,
        train_mode_setter: Optional[Callable[[MHD_Graph], None]] = None,
        monitor_interval_steps: int = 1,
    ):
        """
        初始化训练器

        Args:
            mhd_graph: MHD图实例
            optimizer: 优化器
            monitor: 监控器
            forward_levels: 默认 Feature Message 执行序列
            backward_levels: 默认 Gradient Message 执行序列
            criteria: 使用完整验证状态判断最佳模型的 PyTorch callable
            criteria_mode: 'min' 或 'max'，默认 'min'
            save_dir: 保存目录
            grad_clip_norm: 梯度裁剪阈值
            lr_scheduler: 学习率调度器
            save_interval: 常规检查点保存间隔（epoch 数），0 表示不保存常规检查点
        """
        if criteria_mode not in {"min", "max"}:
            raise ValueError("criteria_mode 必须是 min 或 max")
        if not callable(criteria):
            raise TypeError("criteria 必须是 callable")
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps 必须大于等于 1")
        if monitor_interval_steps < 1:
            raise ValueError("monitor_interval_steps 必须大于等于 1")
        self.mhd_graph = mhd_graph
        self.monitor = monitor
        self.forward_levels = tuple(
            mhd_graph._validate_levels(forward_levels, mhd_graph.num_levels, "Trainer Forward")
        )
        self.backward_levels = tuple(
            mhd_graph._validate_levels(backward_levels, mhd_graph.num_levels, "Trainer Backward")
        )
        overlap = sorted(set(self.forward_levels).intersection(self.backward_levels))
        if overlap:
            raise ValueError(f"Trainer 前后向 levels 不得重叠: {overlap}")
        self.loss_node_name = _infer_scalar_terminal_name(mhd_graph, self.forward_levels)
        self.criteria = criteria
        self.criteria_name = getattr(criteria, "__name__", criteria.__class__.__name__)
        self.criteria_mode = criteria_mode
        self.save_dir = save_dir
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler = lr_scheduler
        self.context = distributed_context or initialize_mhd_distributed()
        self.device = self.context.device
        self.save_interval = save_interval
        self.precision = precision.lower()
        self.grad_accum_steps = grad_accum_steps
        self.train_mode_setter = train_mode_setter
        self.monitor_interval_steps = monitor_interval_steps
        self.last_monitor_metrics: Dict[str, float] = {}
        self._micro_step = 0
        self._optimizer_steps = 0
        self._accumulation_paths: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
        self.last_eval_tensors: Dict[str, torch.Tensor] = {}
        self.input_nodes = tuple(input_nodes or ())
        self.input_mapping = {name: name for name in self.input_nodes}
        for node_name, batch_key in dict(input_mapping or {}).items():
            if node_name not in self.input_mapping:
                raise ValueError(f"input_mapping 包含未声明的输入节点 '{node_name}'")
            self.input_mapping[node_name] = batch_key
        requested_outputs = list(output_nodes or ())
        metric_outputs = [self.loss_node_name, *monitor.monitor_nodes]
        for name in metric_outputs:
            if name not in requested_outputs:
                requested_outputs.append(name)
        self.output_nodes = tuple(requested_outputs)
        self.parallel = parallel or MHD_ParallelConfig()
        if self.parallel.pipeline_size > 1 and self.grad_accum_steps != 1:
            raise ValueError("Pipeline schedule 已负责 microbatch，外层 grad_accum_steps 必须为 1")

        advanced = (
            self.parallel.data_parallel != "none"
            or self.parallel.tensor_parallel_size > 1
            or self.parallel.pipeline_size > 1
            or self.parallel.compile
        )
        if advanced:
            if not self.input_nodes:
                raise ValueError("启用高级执行选项时必须提供 input_nodes")
            self.model = prepare_mhd_model(
                mhd_graph,
                self.input_nodes,
                self.output_nodes,
                levels=self.forward_levels,
                backward_levels=self.backward_levels,
                parallel=self.parallel,
                context=self.context,
                precision=self.precision,
            )
        else:
            mhd_graph.to(self.device)
            self.model = mhd_graph

        if isinstance(optimizer, torch.optim.Optimizer):
            if advanced and (
                self.parallel.data_parallel == "fsdp2"
                or self.parallel.tensor_parallel_size > 1
                or self.parallel.pipeline_size > 1
            ):
                raise ValueError("FSDP/TP/PP 必须在模型准备后创建优化器；请传入 optimizer factory")
            self.optimizer = optimizer
        elif callable(optimizer):
            self.optimizer = optimizer(self.model.parameters())
        else:
            raise TypeError("optimizer 必须是 Optimizer 或接收 parameters 的 factory")

        scaler_enabled = self.precision in {"fp16", "float16"} and self.device.type == "cuda"
        try:
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        _precision_dtype(self.precision, self.device)

        if self.context.is_main:
            os.makedirs(save_dir, exist_ok=True)
        mhd_barrier(self.context)
        self.logger = self._setup_logger(save_dir)

        # 历史记录
        self.history = {
            "train": {"metrics": []},
            "eval": {"metrics": []},
            "best_eval_value": float("inf") if criteria_mode == 'min' else -float("inf"),
            "best_epoch": -1,
            "best_loss_value": None
        }

        self._validate_nodes()

        if self.context.is_main:
            self.logger.info("=" * 80)
            self.logger.info("🚀 MHD V4 Trainer 初始化完成")
            self.logger.info(f"设备={self.device} precision={self.precision} accum={self.grad_accum_steps}")
            self.logger.info(
                f"Forward levels={list(self.forward_levels)} "
                f"Backward levels={list(self.backward_levels)} "
                f"标量终点={self.loss_node_name} Criteria={self.criteria_name}"
            )
            self.logger.info("=" * 80)

    def _setup_logger(self, save_dir: str) -> logging.Logger:
        logger = logging.getLogger(f"mhd_train.rank{self.context.rank}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        if not self.context.is_main:
            logger.addHandler(logging.NullHandler())
            return logger
        log_file = os.path.join(save_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _validate_nodes(self):
        if not self.mhd_graph.get_node_by_name(self.loss_node_name):
            raise ValueError(f"标量终点 '{self.loss_node_name}' 不存在")
        for node_name in self.monitor.monitor_nodes:
            if not self.mhd_graph.get_node_by_name(node_name):
                warnings.warn(f"监控节点 '{node_name}' 不存在")

    def _pipeline_batch_size(self, input_dict: dict) -> int:
        names = self.input_nodes or tuple(
            name for name, value in input_dict.items() if isinstance(value, torch.Tensor)
        )
        batch_sizes = []
        for name in names:
            batch_key = self.input_mapping.get(name, name)
            if batch_key not in input_dict:
                raise KeyError(f"输入节点 '{name}' 缺少批次字段 '{batch_key}'")
            tensor = input_dict[batch_key]
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 1:
                raise ValueError(
                    f"输入节点 '{name}' 对应字段 '{batch_key}' 必须是至少一维 Tensor"
                )
            batch_sizes.append(tensor.shape[0])
        if not batch_sizes:
            raise ValueError("空输入")
        if len(set(batch_sizes)) != 1:
            raise ValueError("批次大小不一致")
        return batch_sizes[0]

    def _autocast_context(self):
        dtype = _precision_dtype(self.precision, self.device)
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _forward_graph(
        self,
        input_dict: Dict[str, torch.Tensor],
        forward_levels: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        names = self.input_nodes or tuple(
            name for name, value in input_dict.items() if isinstance(value, torch.Tensor)
        )
        moved = {}
        for name in names:
            batch_key = self.input_mapping.get(name, name)
            if batch_key not in input_dict:
                raise KeyError(f"输入节点 '{name}' 缺少批次字段 '{batch_key}'")
            tensor = input_dict[batch_key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"输入节点 '{name}' 对应字段 '{batch_key}' 必须是 Tensor"
                )
            moved[name] = tensor.to(self.device, non_blocking=True)
        if self.model is self.mhd_graph:
            for node in self.mhd_graph.nodes:
                node.reset()
            for node_name, tensor in moved.items():
                node = self.mhd_graph.get_node_by_name(node_name)
                if node is not None:
                    node.feature_message.current_state = tensor
            self.mhd_graph.forward(levels=list(forward_levels))
            return {
                name: self.mhd_graph.get_node_by_name(name).feature_message.current_state
                for name in self.output_nodes
                if self.mhd_graph.get_node_by_name(name) is not None
            }
        if isinstance(self.model, _MHD_PipelineModel):
            if self.model.stage_index == 0:
                args = tuple(moved[name] for name in self.input_nodes)
                result = self.model(*args)
            else:
                result = self.model()
            if self.model.stage_index == self.model.num_stages - 1 and result is not None:
                value = result[0] if isinstance(result, (tuple, list)) else result
                return {self.loss_node_name: value}
            return {}
        adapters = [
            module for module in self.model.modules()
            if isinstance(module, _MHD_GraphAdapter)
        ]
        if not adapters:
            raise RuntimeError("并行模型中未找到内部 MHD Graph adapter")
        for adapter in adapters:
            adapter.levels = tuple(forward_levels)
        adapter_inputs = {name: moved[name] for name in self.input_nodes}
        return self.model(adapter_inputs)

    def _collect_metrics(
        self,
        outputs: Mapping[str, torch.Tensor],
        excluded_nodes: Sequence[str] = (),
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        excluded = set(excluded_nodes)
        metric_nodes = tuple(
            dict.fromkeys(
                (
                    self.loss_node_name,
                    *self.monitor.monitor_nodes,
                )
            )
        )
        for node_name in metric_nodes:
            if node_name in excluded:
                continue
            value = outputs.get(node_name)
            if value is None:
                node = self.mhd_graph.get_node_by_name(node_name)
                value = node.feature_message.current_state if node is not None else None
            if value is not None:
                scalar = float(value.detach().float().mean().item())
                metrics[node_name] = scalar if np.isfinite(scalar) else 0.0
        return metrics

    def _gather_eval_tensors(
        self,
        local_tensors: Mapping[str, Sequence[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        local = {}
        for node_name, values in local_tensors.items():
            if not values:
                continue
            local[node_name] = torch.cat(tuple(values), dim=0).cpu()
        if self.context.distributed:
            gathered = [None] * self.context.world_size
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        complete = {
            node_name: torch.cat(
                tuple(part[node_name] for part in gathered), dim=0
            )
            for node_name in local
        }
        self.last_eval_tensors = complete
        return complete

    @torch.no_grad()
    def _run_criteria(
        self,
        complete_tensors: Mapping[str, torch.Tensor],
    ) -> float:
        try:
            for node_name, value in complete_tensors.items():
                node = self.mhd_graph.get_node_by_name(node_name)
                if node is not None:
                    node.feature_message.current_state = value.to(
                        self.device, non_blocking=True
                    )
            value = self.criteria(self.mhd_graph)
        finally:
            for node in self.mhd_graph.nodes:
                node.reset()
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RuntimeError(f"Criteria '{self.criteria_name}' 必须返回标量")
            scalar = float(value.detach().float().item())
        else:
            scalar = float(value)
        if not np.isfinite(scalar):
            raise RuntimeError(f"Criteria '{self.criteria_name}' 产生了非有限值")
        return scalar

    def train_step(
        self,
        input_dict: dict,
        *,
        forward_levels: Optional[Sequence[int]] = None,
        backward_levels: Optional[Sequence[int]] = None,
        _force_step: bool = False,
        _loss_divisor: Optional[int] = None,
    ) -> dict:
        active_forward = tuple(
            self.mhd_graph._validate_levels(
                self.forward_levels if forward_levels is None else forward_levels,
                self.mhd_graph.num_levels,
                "Train Forward",
            )
        )
        active_backward = tuple(
            self.mhd_graph._validate_levels(
                self.backward_levels if backward_levels is None else backward_levels,
                self.mhd_graph.num_levels,
                "Train Backward",
            )
        )
        overlap = sorted(set(active_forward).intersection(active_backward))
        if overlap:
            raise ValueError(f"Train 前后向 levels 不得重叠: {overlap}")
        if isinstance(self.model, _MHD_PipelineModel):
            if active_forward != self.forward_levels or active_backward != self.backward_levels:
                raise ValueError("Pipeline stage 已固定，train_step 不支持临时覆盖 level 路径")
            return self._pipeline_train_step(input_dict)
        self.model.train()
        if self.train_mode_setter is not None:
            self.train_mode_setter(self.mhd_graph)
        if self._micro_step % self.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulation_paths = (active_forward, active_backward)
        elif self._accumulation_paths != (active_forward, active_backward):
            raise ValueError("同一梯度累积窗口内 Forward/Backward 路径必须一致")
        should_step = ((self._micro_step + 1) % self.grad_accum_steps == 0) or _force_step
        sync_context = (
            self.model.no_sync()
            if not should_step and hasattr(self.model, "no_sync")
            else nullcontext()
        )
        with sync_context:
            with self._autocast_context():
                outputs = self._forward_graph(input_dict, active_forward)
                loss_value = outputs.get(self.loss_node_name)
                if loss_value is None:
                    loss_node = self.mhd_graph.get_node_by_name(self.loss_node_name)
                    loss_value = (
                        loss_node.feature_message.current_state
                        if loss_node is not None
                        else None
                    )
                if loss_value is None:
                    raise RuntimeError(f"未生成标量终点 {self.loss_node_name}")
                if loss_value.numel() != 1:
                    raise RuntimeError("训练终点必须是标量 Tensor")
            if not loss_value.requires_grad:
                raise RuntimeError("标量终点无梯度")
            if self.grad_scaler.is_enabled():
                # Initialize GradScaler's public optimizer state; MHD backward
                # applies the same scale through the terminal gradient seed.
                self.grad_scaler.scale(loss_value)
            scale = (
                float(self.grad_scaler.get_scale())
                if self.grad_scaler.is_enabled()
                else 1.0
            )
            loss_scale = scale / float(
                _loss_divisor or self.grad_accum_steps
            )
            self.mhd_graph._backward(
                levels=list(active_backward),
                retain_graph=False,
                loss_scale=loss_scale,
            )
            if scale != 1.0:
                for node in self.mhd_graph.nodes:
                    current = node.gradient_message.current_state
                    node.gradient_message.current_state = current / scale

        should_monitor = (
            (self._micro_step + 1) % self.monitor_interval_steps == 0
            or _force_step
        )
        if should_monitor:
            self.monitor.monitor_node(self.mhd_graph, prefix="node/")
        if should_step:
            self.grad_scaler.unscale_(self.optimizer)
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            if should_monitor:
                self.monitor.monitor_edge(
                    self.mhd_graph,
                    prefix="edge/",
                    train_mode=True,
                )
            scale_before_update = float(self.grad_scaler.get_scale())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            if (
                not self.grad_scaler.is_enabled()
                or float(self.grad_scaler.get_scale()) >= scale_before_update
            ):
                self._optimizer_steps += 1
            self._accumulation_paths = None
        self._micro_step += 1
        return self._collect_metrics(outputs)

    def _pipeline_train_step(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch_size = self._pipeline_batch_size(input_dict)
        moved = {name: tensor.to(self.device, non_blocking=True) for name, tensor in input_dict.items()}
        losses: List[torch.Tensor] = []
        with self._autocast_context():
            if self.model.stage_index == 0:
                args = tuple(moved[name] for name in self.input_nodes)
                self.model(*args)
            elif self.model.stage_index == self.model.num_stages - 1:
                dummy_target = torch.zeros(batch_size, device=self.device)
                self.model(target=dummy_target, losses=losses)
            else:
                self.model()
        if self.grad_clip_norm and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self._micro_step += 1
        local_loss = torch.stack([value.detach().float() for value in losses]).mean() if losses else None
        loss = self.model.synchronize_last_stage_scalar(local_loss)
        return {self.loss_node_name: float(loss.item())}

    @torch.no_grad()
    def _eval_outputs(self, input_dict: dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with self._autocast_context():
            return self._forward_graph(input_dict, self.forward_levels)

    @torch.no_grad()
    def eval_step(self, input_dict: dict) -> dict:
        outputs = self._eval_outputs(input_dict)
        if isinstance(self.model, _MHD_PipelineModel):
            local_value = outputs.get(self.loss_node_name)
            value = self.model.synchronize_last_stage_scalar(local_value)
            return {self.loss_node_name: float(value.item())}
        return self._collect_metrics(outputs)

    def _reduce_epoch_metrics(self, metric_sums: Mapping[str, float], steps: int) -> Dict[str, float]:
        names = sorted(metric_sums)
        values = torch.tensor(
            [metric_sums[name] for name in names] + [float(steps)],
            dtype=torch.float64,
            device=self.device,
        )
        if self.context.distributed:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        count = max(float(values[-1].item()), 1.0)
        return {name: float(values[index].item() / count) for index, name in enumerate(names)}

    def train_epoch(self, train_data, epoch: int):
        self.monitor.reset()
        optimizer_steps_before = self._optimizer_steps
        epoch_metrics_sum = defaultdict(float)
        if hasattr(getattr(train_data, "sampler", None), "set_epoch"):
            train_data.sampler.set_epoch(epoch)
        pbar = tqdm(train_data, desc=f"Train Epoch {epoch+1}", leave=False, disable=not self.context.is_main)
        sample_count = 0
        total_steps = len(train_data)
        for step, input_dict in enumerate(pbar):
            force_step = step + 1 == total_steps
            window_start = (step // self.grad_accum_steps) * self.grad_accum_steps
            loss_divisor = min(self.grad_accum_steps, total_steps - window_start)
            step_metrics = self.train_step(
                input_dict,
                _force_step=force_step,
                _loss_divisor=loss_divisor,
            )
            batch_samples = self._pipeline_batch_size(input_dict)
            sample_count += batch_samples if step_metrics else 0
            for k, v in step_metrics.items():
                epoch_metrics_sum[k] += v * batch_samples
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
        avg_metrics = self._reduce_epoch_metrics(epoch_metrics_sum, sample_count)
        local_monitor = self.monitor.get_mean_metrics()
        if self.context.distributed:
            gathered_monitor = [None] * self.context.world_size
            dist.all_gather_object(gathered_monitor, local_monitor)
        else:
            gathered_monitor = [local_monitor]
        monitor_names = sorted({name for item in gathered_monitor for name in item})
        self.last_monitor_metrics = {
            name: float(np.mean([item[name] for item in gathered_monitor if name in item]))
            for name in monitor_names
        }
        self.history["train"]["metrics"].append(avg_metrics)
        if self.lr_scheduler is not None and self._optimizer_steps > optimizer_steps_before:
            self.lr_scheduler.step()
        self.logger.info(f"\n📈 训练轮次 {epoch+1} 指标: {avg_metrics}")
        return avg_metrics

    def eval_epoch(self, eval_data, epoch: int):
        self.monitor.reset()
        epoch_metrics_sum = defaultdict(float)
        eval_tensors = defaultdict(list)
        pbar = tqdm(eval_data, desc=f"Eval  Epoch {epoch+1}", leave=False, disable=not self.context.is_main)
        sample_count = 0
        for step, input_dict in enumerate(pbar):
            batch_samples = self._pipeline_batch_size(input_dict)
            outputs = self._eval_outputs(input_dict)
            step_metrics = self._collect_metrics(outputs)
            for node_name in self.output_nodes:
                value = outputs.get(node_name)
                if value is None:
                    node = self.mhd_graph.get_node_by_name(node_name)
                    value = node.feature_message.current_state if node is not None else None
                if not isinstance(value, torch.Tensor):
                    continue
                detached = value.detach()
                if detached.ndim == 0:
                    detached = detached.reshape(1).expand(batch_samples)
                elif detached.shape[0] != batch_samples:
                    continue
                eval_tensors[node_name].append(detached.cpu())
            sample_count += batch_samples if step_metrics else 0
            for k, v in step_metrics.items():
                epoch_metrics_sum[k] += v * batch_samples
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
        avg_metrics = self._reduce_epoch_metrics(epoch_metrics_sum, sample_count)
        complete_tensors = self._gather_eval_tensors(eval_tensors)
        avg_metrics[self.criteria_name] = self._run_criteria(complete_tensors)
        self.history["eval"]["metrics"].append(avg_metrics)

        cur_loss = avg_metrics.get(self.loss_node_name)
        cur_criteria = avg_metrics[self.criteria_name]

        is_better = False
        if self.criteria_mode == 'min':
            if cur_criteria < self.history["best_eval_value"]:
                is_better = True
        else:
            if cur_criteria > self.history["best_eval_value"]:
                is_better = True

        if is_better:
            self.history["best_eval_value"] = cur_criteria
            self.history["best_epoch"] = epoch + 1
            self.history["best_loss_value"] = cur_loss
            self._save_best_checkpoint()
            loss_str = f"{cur_loss:.6f}" if cur_loss is not None else "N/A"
            self.logger.info(f"🏆 当下最佳模型 | Epoch {epoch+1} | {self.loss_node_name}: {loss_str} | {self.criteria_name}: {cur_criteria:.6f}")
        else:
            best_loss = self.history.get("best_loss_value")
            loss_str = f"{best_loss:.6f}" if best_loss is not None else "N/A"
            self.logger.info(f"🏆 过往最佳模型 | Epoch {self.history['best_epoch']} | {self.loss_node_name}: {loss_str} | {self.criteria_name}: {self.history['best_eval_value']:.6f}")

        self.logger.info(f"📊 验证轮次 {epoch+1} 指标: {avg_metrics}")
        return avg_metrics

    def _checkpoint_state(
        self,
        epoch: int,
        *,
        legacy_node_format: bool = False,
    ) -> Dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "trainer": {
                "history": self.history,
                "epoch": epoch,
                "micro_step": self._micro_step,
                "forward_levels": list(self.forward_levels),
                "backward_levels": list(self.backward_levels),
                "criteria_name": self.criteria_name,
                "criteria_mode": self.criteria_mode,
            },
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else {},
            "scaler": self.grad_scaler.state_dict(),
        }
        ordered_nodes = sorted(self.mhd_graph.nodes, key=lambda item: item.id)
        if legacy_node_format:
            state.update(
                {
                    "nodes": {
                        node.name: node.feature_message.initial_state.detach()
                        for node in ordered_nodes
                    },
                    "feature_message_current": {
                        node.name: node.feature_message.current_state.detach()
                        for node in ordered_nodes
                    },
                    "gradient_message_initial": {
                        node.name: node.gradient_message.initial_state.detach()
                        for node in ordered_nodes
                    },
                    "gradient_message_current": {
                        node.name: node.gradient_message.current_state.detach()
                        for node in ordered_nodes
                    },
                }
            )
        else:
            state["node_messages"] = {
                node.name: {
                    "feature_message": {
                        "initial_state": node.feature_message.initial_state.detach(),
                        "current_state": node.feature_message.current_state.detach(),
                    },
                    "gradient_message": {
                        "initial_state": node.gradient_message.initial_state.detach(),
                        "current_state": node.gradient_message.current_state.detach(),
                    },
                }
                for node in ordered_nodes
            }
        return state

    def _save_best_checkpoint(self):
        self._save_distributed_checkpoint("best", self.history["best_epoch"])

    def save_checkpoint(self, epoch: int):
        self._save_distributed_checkpoint(f"epoch_{epoch}", epoch)

    def save_last_checkpoint(self, epoch: int):
        self._save_distributed_checkpoint("last", epoch)

    def _save_distributed_checkpoint(self, name: str, epoch: int) -> None:
        try:
            from torch.distributed import checkpoint as dcp

            checkpoint_path = os.path.join(self.save_dir, name)
            dcp.save(
                self._checkpoint_state(epoch),
                checkpoint_id=checkpoint_path,
                no_dist=not self.context.distributed,
            )
            if self.context.is_main:
                self.logger.info(f"✅ 检查点保存完成: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"❌ 保存检查点失败: {str(e)}")
            raise

    def load_checkpoint(
        self,
        load_best: bool = False,
        load_last: bool = False,
        epoch: int = None,
    ) -> int:
        try:
            selected = sum((bool(load_best), bool(load_last), epoch is not None))
            if selected != 1:
                raise ValueError("必须且只能指定 best、last 或一个 epoch")
            name = (
                "best"
                if load_best
                else "last"
                if load_last
                else f"epoch_{epoch}"
            )
            checkpoint_path = os.path.join(self.save_dir, name)
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
            from torch.distributed import checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import set_state_dict

            metadata = dcp.FileSystemReader(checkpoint_path).read_metadata()
            checkpoint_keys = tuple(str(key) for key in metadata.state_dict_metadata)
            canonical_messages = any(
                key.startswith("node_messages.") for key in checkpoint_keys
            )
            for split in ("train", "eval"):
                prefix = f"trainer.history.{split}.metrics."
                records: Dict[int, Dict[str, float]] = {}
                for key in checkpoint_keys:
                    if not key.startswith(prefix):
                        continue
                    index_text, separator, metric_name = key[len(prefix):].partition(".")
                    if not separator or not index_text.isdigit() or not metric_name:
                        continue
                    records.setdefault(int(index_text), {})[metric_name] = 0.0
                if records:
                    expected = set(range(max(records) + 1))
                    if set(records) != expected:
                        raise ValueError(
                            f"Checkpoint {split} history indices are not contiguous"
                        )
                    self.history[split]["metrics"] = [
                        records[index] for index in range(len(records))
                    ]
            state = self._checkpoint_state(
                epoch or 0,
                legacy_node_format=not canonical_messages,
            )
            dcp.load(state, checkpoint_id=checkpoint_path, no_dist=not self.context.distributed)
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
            )
            for node in self.mhd_graph.nodes:
                if canonical_messages:
                    message_state = state["node_messages"].get(node.name)
                    if message_state is None:
                        continue
                    feature_initial = message_state["feature_message"]["initial_state"].to(self.device)
                    feature_current = message_state["feature_message"]["current_state"].to(self.device)
                    gradient_initial = message_state["gradient_message"]["initial_state"].to(self.device)
                    gradient_current = message_state["gradient_message"]["current_state"].to(self.device)
                else:
                    if node.name not in state["nodes"]:
                        continue
                    feature_initial = state["nodes"][node.name].to(self.device)
                    feature_current = state.get("feature_message_current", {}).get(
                        node.name, feature_initial
                    ).to(self.device)
                    gradient_initial = state.get("gradient_message_initial", {}).get(
                        node.name, node.gradient_message.initial_state
                    ).to(self.device)
                    gradient_current = state.get("gradient_message_current", {}).get(
                        node.name, gradient_initial
                    ).to(self.device)
                node.feature_message = MHD_Node.Message(feature_initial, feature_current)
                node.gradient_message = MHD_Node.Message(gradient_initial, gradient_current)
            self.history = state["trainer"]["history"]
            saved_criteria_name = state["trainer"].get("criteria_name")
            saved_criteria_mode = state["trainer"].get("criteria_mode")
            if saved_criteria_name not in {None, self.criteria_name}:
                raise ValueError(
                    f"Checkpoint Criteria 为 {saved_criteria_name}，当前为 {self.criteria_name}"
                )
            if saved_criteria_mode not in {None, self.criteria_mode}:
                raise ValueError(
                    f"Checkpoint Criteria mode 为 {saved_criteria_mode}，当前为 {self.criteria_mode}"
                )
            self._micro_step = int(state["trainer"].get("micro_step", 0))
            saved_forward = state["trainer"].get("forward_levels")
            saved_backward = state["trainer"].get("backward_levels")
            if saved_forward is not None or saved_backward is not None:
                if saved_forward is None or saved_backward is None:
                    raise ValueError("Checkpoint 的 Forward/Backward level 配置不完整")
                restored_forward = tuple(self.mhd_graph._validate_levels(
                    saved_forward, self.mhd_graph.num_levels, "Checkpoint Forward"
                ))
                restored_backward = tuple(self.mhd_graph._validate_levels(
                    saved_backward, self.mhd_graph.num_levels, "Checkpoint Backward"
                ))
                overlap = sorted(set(restored_forward).intersection(restored_backward))
                if overlap:
                    raise ValueError(f"Checkpoint 前后向 levels 不得重叠: {overlap}")
                self.forward_levels = restored_forward
                self.backward_levels = restored_backward
            if self.lr_scheduler and state["scheduler"]:
                self.lr_scheduler.load_state_dict(state["scheduler"])
            if state["scaler"]:
                self.grad_scaler.load_state_dict(state["scaler"])
            self.logger.info("✅ 权重加载完成")
            return int(state["trainer"].get("epoch", epoch or 0))
        except Exception as e:
            self.logger.error(f"❌ 加载权重失败: {str(e)}")
            raise

    def save_training_history(self):
        if not self.context.is_main:
            return
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)
        self.logger.info(f"📝 训练历史已保存至: {history_path}")

    def train(self, train_data, eval_data, epochs: int = 10):
        self.logger.info("="*80)
        self.logger.info(f"🚀 开始训练，共 {epochs} 轮")
        self.logger.info("="*80)
        try:
            for epoch in range(epochs):
                self.logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
                self.train_epoch(train_data, epoch)
                self.eval_epoch(eval_data, epoch)   # 内部只更新最佳模型
                self.save_last_checkpoint(epoch + 1)

                cur_ep = epoch + 1
                if self.save_interval > 0 and cur_ep % self.save_interval == 0:
                    self.save_checkpoint(cur_ep)

            self.save_checkpoint(epochs)
            self.save_training_history()
            self.logger.info("\n" + "="*80)
            self.logger.info("🎉 训练完成！")
            best_loss = self.history.get("best_loss_value")
            loss_str = f"{best_loss:.6f}" if best_loss is not None else "N/A"
            self.logger.info(f"🏆 最佳 {self.loss_node_name}: {loss_str} (Epoch {self.history['best_epoch']})")
            self.logger.info(f"🏆 最佳 {self.criteria_name}: {self.history['best_eval_value']:.6f} (Epoch {self.history['best_epoch']})")
            self.logger.info(f"📝 所有结果已保存至: {self.save_dir}")
            self.logger.info("="*80)
        except Exception as e:
            self.logger.error(f"❌ 训练过程出错: {str(e)}", exc_info=True)
            raise

# ===================== 优化器创建函数 =====================

def create_mhd_optimizer(
    mhd_graph,
    edge_optim_config: dict = None,
    default_optimizer_type: str = "adam",
    default_lr: float = 0.001,
    default_weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建MHD图优化器

    特性：
    1. 支持边级别的优化器配置
    2. 自动收集图参数
    """
    edge_optim_config = edge_optim_config or {}
    param_groups = []
    graph = mhd_graph if isinstance(mhd_graph, MHD_Graph) else None
    if graph is None:
        try:
            graph = unwrap_mhd_graph(mhd_graph)
        except TypeError:
            graph = None

    for edge_name, config in edge_optim_config.items():
        edge = graph.get_edge_by_name(edge_name) if graph is not None else None
        if edge is None:
            warnings.warn(f"边 {edge_name} 不存在，跳过自定义配置")
            continue
        edge_params = []
        for op in edge.edge_operations:
            function = op.function
            if isinstance(function, nn.Module):
                edge_params.extend([p for p in function.parameters() if p.requires_grad])
        if edge_params:
            param_groups.append({
                "params": edge_params,
                "lr": config.get("lr", default_lr),
                "weight_decay": config.get("weight_decay", default_weight_decay),
                "name": edge_name
            })

    processed_ids = {id(p) for group in param_groups for p in group["params"]}
    remaining = [p for p in mhd_graph.parameters() if p.requires_grad and id(p) not in processed_ids]
    if remaining:
        param_groups.append({
            "params": remaining,
            "lr": default_lr,
            "weight_decay": default_weight_decay,
            "name": "default"
        })

    opt_map = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    if default_optimizer_type.lower() not in opt_map:
        raise ValueError(f"不支持的优化器: {default_optimizer_type}")

    if default_optimizer_type.lower() == "adam":
        kwargs.setdefault("betas", (0.9, 0.999))
        kwargs.setdefault("eps", 1e-8)
    elif default_optimizer_type.lower() == "sgd":
        kwargs.setdefault("momentum", 0.9)
        kwargs.setdefault("nesterov", True)

    optimizer_parameters = [
        parameter
        for group in param_groups
        for parameter in group["params"]
    ]
    distributed_flags = {
        hasattr(parameter, "placements") and hasattr(parameter, "device_mesh")
        for parameter in optimizer_parameters
    }
    if len(distributed_flags) > 1:
        kwargs.setdefault("foreach", False)

    optimizer = opt_map[default_optimizer_type.lower()](param_groups, **kwargs)

    logger = logging.getLogger("mhd_train")
    logger.info(f"✅ 优化器创建完成: {default_optimizer_type.upper()}")
    for i, group in enumerate(param_groups):
        logger.info(f"  组{i+1}: {group.get('name','unknown')} lr={group['lr']}, wd={group['weight_decay']}")
    return optimizer


# ===================== 推理器 =====================

class MHD_Inferencer:
    """
    MHD 推理器
    自动处理模型加载、batch 调整与前向推理，
    支持批量预测，以及按指定层级顺序前向传播以捕获中间节点状态。
    """
    def __init__(
        self,
        build_graph_fn: Callable[..., MHD_Graph],
        checkpoint_dir: str,
        device: torch.device = None,
        build_kwargs: dict = None,
        input_nodes: Sequence[str] = ("input_img",),
        output_nodes: Sequence[str] = ("logits",),
        levels: Sequence[int] = (),
    ):
        """
        Args:
            build_graph_fn: 图构建函数，签名需为 (batch_size, ...) -> MHD_Graph
            checkpoint_dir: 保存有权重文件的目录（需包含 node_best.pth 和 edge_best.pth）
            device: 计算设备
            build_kwargs: 传递给 build_graph_fn 的其他参数（除 batch_size 和 device 外）
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.build_kwargs = build_kwargs or {}
        self.input_nodes = tuple(input_nodes)
        self.output_nodes = tuple(output_nodes)
        if not levels:
            raise ValueError("MHD_Inferencer 必须显式提供非空 levels")

        train_graph = build_graph_fn(batch_size=1, device=self.device, **self.build_kwargs)
        node_path = os.path.join(checkpoint_dir, "node_best.pth")
        edge_path = os.path.join(checkpoint_dir, "edge_best.pth")
        dcp_path = os.path.join(checkpoint_dir, "best")
        if os.path.isdir(dcp_path):
            from torch.distributed import checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

            state = {
                "model": get_model_state_dict(train_graph),
                "nodes": {node.name: node.feature_message.initial_state for node in train_graph.nodes},
            }
            dcp.load(state, checkpoint_id=dcp_path, no_dist=True)
            set_model_state_dict(train_graph, state["model"])
            for node in train_graph.nodes:
                if node.name in state["nodes"]:
                    node.feature_message.update_initial(
                        state["nodes"][node.name].to(self.device),
                        update_current=True,
                    )
        elif os.path.exists(node_path) and os.path.exists(edge_path):
            updown_node(train_graph.nodes, node_path, mode="up", target_device=self.device)
            updown_edge(train_graph.edges, edge_path, mode="up", target_device=self.device)
        else:
            raise FileNotFoundError(f"未找到 V4 DCP 或 V3 best 检查点: {checkpoint_dir}")

        self.graph = train_graph
        self.model = _MHD_GraphAdapter(train_graph, self.input_nodes, self.output_nodes, levels)
        print("✅ MHD Inferencer 初始化完成")

    @torch.inference_mode()
    def run(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved = {name: tensor.to(self.device, non_blocking=True) for name, tensor in inputs.items()}
        return self.model(moved)

    @torch.inference_mode()
    def predict(self, image_batch: Union[torch.Tensor, Mapping[str, torch.Tensor]]) -> Union[List[int], Dict[str, torch.Tensor]]:
        """
        批量推理
        Args:
            image_batch: shape (B, C, H, W)
        Returns:
            list of predicted class indices
        """
        if isinstance(image_batch, Mapping):
            return self.run(image_batch)
        if len(self.input_nodes) != 1 or len(self.output_nodes) != 1:
            raise ValueError("多输入或多输出推理请使用 run({node_name: tensor})")
        output = self.run({self.input_nodes[0]: image_batch})[self.output_nodes[0]]
        return output.argmax(dim=1).tolist()

# ===================== 孤立节点 & 孤立边自动修剪工具 =====================

def prune_isolated_graph(graph: MHD_Graph, verbose: bool = True) -> MHD_Graph:
    """
    自动检测并删除图中所有孤立节点和孤立边。

    该操作会重排 Node/Edge ID、裁剪 Topo 并重建 Module 注册，因此必须在
    第一次 ``graph.forward(levels=...)``、optimizer 创建和并行包装之前调用。它属于
    建图收尾步骤，不是训练过程中的动态图修改接口。

    孤立节点：在所有层级的 role 矩阵中，该列全为 0（无入度也无出度）。
    孤立边：在所有层级的 role 矩阵中，该行全为 0（未连接任何节点）。

    设计哲学：
        - Framework 允许孤立元素存在（纯数据载体或占位）。
        - Utils 提供主动清理机制，在训练前释放冗余显存。
        - 删除前会打印详细警告，列出所有受影响的节点和边，便于用户复核。

    Args:
        graph: 尚未执行前向、尚未创建 optimizer/并行包装的 MHD_Graph
        verbose: 是否打印详细日志

    Returns:
        修剪后的原图实例（原地修改）
    """
    if not isinstance(graph, MHD_Graph):
        raise TypeError("graph 必须是 MHD_Graph")
    if graph._forward_trace:
        raise RuntimeError(
            "prune_isolated_graph 必须在第一次 graph.forward(levels=...) 之前调用"
        )
    if not graph.topo or graph.num_levels == 0:
        if verbose:
            print("⚠️ 图无拓扑信息，跳过修剪。")
        return graph

    num_edges_orig = len(graph.edges)
    num_nodes_orig = len(graph.nodes)
    device = graph.device

    # ----- 1. 计算孤立节点掩码 -----
    # 聚合所有层级，按列（节点）检查是否出现过非零值
    node_active = torch.zeros(num_nodes_orig, dtype=torch.bool, device=device)
    for role in graph.topo.role_matrices:
        node_active = node_active | (role != 0).any(dim=0)   # any(dim=0) 按列

    isolated_node_ids = [i for i in range(num_nodes_orig) if not node_active[i]]
    isolated_nodes = [graph.get_node_by_id(i) for i in isolated_node_ids if graph.get_node_by_id(i) is not None]

    # ----- 2. 计算孤立边掩码 -----
    # 聚合所有层级，按行（边）检查是否出现过非零值
    edge_active = torch.zeros(num_edges_orig, dtype=torch.bool, device=device)
    for role in graph.topo.role_matrices:
        edge_active = edge_active | (role != 0).any(dim=1)   # any(dim=1) 按行

    isolated_edge_ids = [i for i in range(num_edges_orig) if not edge_active[i]]
    isolated_edges = [graph.get_edge_by_id(i) for i in isolated_edge_ids if graph.get_edge_by_id(i) is not None]

    # ----- 3. 若无孤立元素，直接返回 -----
    if not isolated_nodes and not isolated_edges:
        if verbose:
            print("✅ 未检测到孤立节点或孤立边，无需清理。")
        return graph

    # ----- 4. 打印警告（醒目输出） -----
    if verbose:
        print("=" * 80)
        if isolated_nodes:
            names = ", ".join([n.name for n in isolated_nodes])
            print(f"⚠️ 检测到 {len(isolated_nodes)} 个孤立节点（出入度均为0）: {names}")
        if isolated_edges:
            names = ", ".join([e.name for e in isolated_edges])
            print(f"⚠️ 检测到 {len(isolated_edges)} 条孤立边（未连接任何节点）: {names}")
        print("📌 提示：这些元素将被删除。若某些元素不应被删，请检查拓扑配置。")
        print("=" * 80)

    # ----- 5. 执行删除操作 -----
    # 5.1 计算保留的节点 ID 和边 ID
    keep_node_ids = [i for i in range(num_nodes_orig) if i not in isolated_node_ids]
    keep_edge_ids = [i for i in range(num_edges_orig) if i not in isolated_edge_ids]

    if not keep_node_ids or not keep_edge_ids:
        raise RuntimeError("❌ 删除后图为空（无节点或无边），操作被阻止。请检查拓扑配置。")

    # 5.2 裁剪拓扑矩阵（同时删除孤立行和孤立列）
    new_role_matrices = []
    new_sort_matrices = []
    keep_node_tensor = torch.tensor(keep_node_ids, device=device)
    keep_edge_tensor = torch.tensor(keep_edge_ids, device=device)

    for role, sort_mat in zip(
        graph.topo.role_matrices,
        graph.topo.sort_matrices,
    ):
        # 先按行删除孤立边，再按列删除孤立节点（顺序无关）
        new_role = role.index_select(dim=0, index=keep_edge_tensor)   # 行（边）
        new_role = new_role.index_select(dim=1, index=keep_node_tensor) # 列（节点）
        new_sort = sort_mat.index_select(dim=0, index=keep_edge_tensor)
        new_sort = new_sort.index_select(dim=1, index=keep_node_tensor)
        new_role_matrices.append(new_role)
        new_sort_matrices.append(new_sort)

    graph.topo.role_matrices = new_role_matrices
    graph.topo.sort_matrices = new_sort_matrices

    # 5.3 先从集合中取出对象，再修改基于 ID 的哈希值。
    # MHD_Node/MHD_Edge 的 __hash__ 依赖 id；对象留在 set 中时原地修改 id
    # 会破坏集合的哈希桶，导致后续查找或删除行为不确定。
    kept_nodes = sorted(
        (node for node in graph.nodes if node.id in keep_node_ids),
        key=lambda x: x.id,
    )
    kept_edges = sorted(
        (edge for edge in graph.edges if edge.id in keep_edge_ids),
        key=lambda x: x.id,
    )

    for new_id, node in enumerate(kept_nodes):
        node.id = new_id
    for new_id, edge in enumerate(kept_edges):
        edge.id = new_id

    # ID 稳定后重新创建集合，确保 set 的内部哈希与对象当前 ID 一致。
    graph.nodes = set(kept_nodes)
    graph.edges = set(kept_edges)

    # 5.5 重建内部索引（节点和边映射）
    graph._build_indices()

    # MHD_Graph 在初始化时已把所有边模块注册到 ModuleDict。仅更新
    # graph.edges 不会自动注销已裁剪模块，因此必须重建注册容器，保证
    # parameters/state_dict/optimizer/显存中只包含当前拓扑的活跃模块。
    graph.edge_module_map = nn.ModuleDict()
    graph._register_all_params()

    # 5.6 验证维度一致性
    graph.topo.validate_topo(len(graph.edges), len(graph.nodes))

    # 5.7 重新计算拓扑排序（因为边变了）
    graph.compact_topological_sort()
    graph._forward_trace = []
    graph._last_forward_levels = tuple()

    if verbose:
        print(f"🧹 修剪完成！移除 {len(isolated_nodes)} 个孤立节点，{len(isolated_edges)} 条孤立边。")
        print(f"   当前节点数: {len(graph.nodes)}，边数: {len(graph.edges)}")

    return graph
