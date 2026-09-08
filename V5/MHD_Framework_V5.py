# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Framework (MHD) - V5
Author: Souray Meng (孟号丁)
Core Framework: Hypergraph-based computational graph with multi-level topology
License: MIT
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Set, Any, Sequence, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
import ast
import logging

logger = logging.getLogger(__name__)


def _evaluate_tensor_ast(node: ast.AST, x: torch.Tensor) -> Any:
    """Evaluate the deliberately small expression grammar used by string operations."""
    if isinstance(node, ast.Name) and node.id == "x":
        return x
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_tensor_ast(item, x) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate_tensor_ast(item, x) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_tensor_ast(node.operand, x)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Attribute):
        owner = _evaluate_tensor_ast(node.value, x)
        if node.attr.startswith("_"):
            raise ValueError("字符串操作不能访问私有属性")
        return getattr(owner, node.attr)
    if isinstance(node, ast.Call):
        function = _evaluate_tensor_ast(node.func, x)
        args = [_evaluate_tensor_ast(arg, x) for arg in node.args]
        kwargs = {kw.arg: _evaluate_tensor_ast(kw.value, x) for kw in node.keywords}
        if None in kwargs:
            raise ValueError("字符串操作不支持 **kwargs")
        return function(*args, **kwargs)
    raise ValueError(f"字符串操作包含不受支持的语法: {type(node).__name__}")


def parse_string_operation(op_str: str, x: torch.Tensor) -> torch.Tensor:
    """
    解析并执行字符串形式的张量操作（新规范：直接拼接 x + op_str）

    用户需提供以 '.' 开头的操作字符串，如 ".relu()" 或 ".mean(dim=1)"，
    函数将直接执行 x.op_str，无需额外解析。

    Args:
        op_str: 操作字符串，必须以 '.' 开头，如 ".relu()" 或 ".mean(dim=1)"
        x: 输入张量

    Returns:
        操作后的张量
    """
    if not isinstance(op_str, str) or not op_str.startswith("."):
        raise ValueError("字符串操作必须以 '.' 开头")
    expression = "x" + op_str
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"字符串操作语法错误: {op_str}") from exc
    result = _evaluate_tensor_ast(tree.body, x)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"字符串操作必须返回 Tensor，实际为 {type(result)!r}")
    return result


@dataclass(frozen=True)
class _MHD_ExecutionStep:
    edge_id: int
    head_ids: Tuple[int, ...]
    tail_ids: Tuple[int, ...]
    edge: Any = field(compare=False, repr=False)
    tail_nodes: Tuple[Any, ...] = field(compare=False, repr=False)


# ===================== 核心超图框架类 =====================

@dataclass(eq=False)
class MHD_Node:
    """Hypergraph node carrying symmetric Feature and Gradient messages."""

    @dataclass
    class Message:
        """Two-state message owned by an :class:`MHD_Node`."""

        initial_state: torch.Tensor
        current_state: Optional[torch.Tensor] = None

        def __post_init__(self) -> None:
            if not isinstance(self.initial_state, torch.Tensor):
                raise TypeError("Message initial_state 必须是 Tensor")
            if self.current_state is None:
                self.current_state = self.initial_state.clone(
                    memory_format=torch.contiguous_format
                )
            if not isinstance(self.current_state, torch.Tensor):
                raise TypeError("Message current_state 必须是 Tensor")
            self.validate()

        def validate(self) -> None:
            if self.initial_state.device != self.current_state.device:
                raise ValueError(
                    "Message Initial State 与 Current State 设备不一致: "
                    f"{self.initial_state.device} vs {self.current_state.device}"
                )
            if self.initial_state.shape != self.current_state.shape:
                raise ValueError(
                    "Message Initial State 与 Current State 形状不一致: "
                    f"{self.initial_state.shape} vs {self.current_state.shape}"
                )
            if self.initial_state.dtype != self.current_state.dtype:
                raise ValueError(
                    "Message Initial State 与 Current State dtype 不一致: "
                    f"{self.initial_state.dtype} vs {self.current_state.dtype}"
                )

        def reset(self) -> 'MHD_Node.Message':
            self.current_state = self.initial_state.clone(
                memory_format=torch.contiguous_format
            )
            return self

        def update_initial(
            self,
            new_tensor: torch.Tensor,
            update_current: bool = True,
        ) -> 'MHD_Node.Message':
            if not isinstance(new_tensor, torch.Tensor):
                raise TypeError("new_tensor 必须是 Tensor")
            if not update_current:
                if (
                    new_tensor.shape != self.current_state.shape
                    or new_tensor.device != self.current_state.device
                    or new_tensor.dtype != self.current_state.dtype
                ):
                    raise ValueError("仅更新 Initial State 时必须与 Current State 完全兼容")
            self.initial_state = new_tensor
            if update_current:
                self.current_state = new_tensor.clone(
                    memory_format=torch.contiguous_format
                )
            self.validate()
            return self

        def to_device(self, device: torch.device) -> 'MHD_Node.Message':
            device = torch.device(device)
            if self.initial_state.device != device:
                self.initial_state = self.initial_state.to(device, non_blocking=True)
            if self.current_state.device != device:
                self.current_state = self.current_state.to(device, non_blocking=True)
            return self

    id: int
    name: str
    feature_message: Message
    gradient_message: Optional[Message] = None
    aggregation: Union[str, Callable[[Optional[torch.Tensor], Sequence[torch.Tensor]], torch.Tensor]] = "sum"
    memory: bool = field(default=False, kw_only=True)
    _default_gradient_initial_identity: Optional[int] = field(
        init=False, default=None, repr=False
    )
    _default_gradient_initial_version: Optional[int] = field(
        init=False, default=None, repr=False
    )

    _BUILTIN_AGGREGATIONS = frozenset({"sum", "avg", "max", "min", "mul"})

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("节点 id 必须是非负整数")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("节点 name 必须是非空字符串")
        if not isinstance(self.feature_message, MHD_Node.Message):
            raise TypeError("feature_message 必须是 MHD_Node.Message")
        self._validate_aggregation(self.aggregation, "aggregation")
        if not isinstance(self.memory, bool):
            raise TypeError("memory 必须是 bool")
        gradient_message_was_omitted = self.gradient_message is None
        if gradient_message_was_omitted:
            feature = self.feature_message.initial_state
            gradient_dtype = (
                feature.dtype
                if feature.is_floating_point() or feature.is_complex()
                else torch.float32
            )
            zeros = torch.zeros(
                feature.shape,
                dtype=gradient_dtype,
                device=feature.device,
            )
            self.gradient_message = MHD_Node.Message(zeros)
        if not isinstance(self.gradient_message, MHD_Node.Message):
            raise TypeError("gradient_message 必须是 MHD_Node.Message 或 None")
        gradient = self.gradient_message.initial_state
        if not (gradient.is_floating_point() or gradient.is_complex()):
            raise TypeError("Gradient Message 必须使用浮点或复数 dtype")
        if gradient.shape != self.feature_message.initial_state.shape:
            raise ValueError("Feature Message 与 Gradient Message 的形状必须一致")
        if gradient.device != self.feature_message.initial_state.device:
            raise ValueError("Feature Message 与 Gradient Message 的设备必须一致")
        if gradient_message_was_omitted:
            self._remember_default_zero_gradient()

    def _remember_default_zero_gradient(self) -> None:
        initial = self.gradient_message.initial_state
        self._default_gradient_initial_identity = id(initial)
        self._default_gradient_initial_version = initial._version

    def _gradient_initial_is_zero(self) -> bool:
        """Check the Gradient Initial State without syncing the common case."""
        initial = self.gradient_message.initial_state
        if (
            id(initial) == self._default_gradient_initial_identity
            and initial._version == self._default_gradient_initial_version
        ):
            return True
        return torch.count_nonzero(initial).item() == 0

    @staticmethod
    def _validate_aggregation(aggregation: Any, name: str) -> None:
        if isinstance(aggregation, str):
            if aggregation not in MHD_Node._BUILTIN_AGGREGATIONS:
                raise ValueError(f"不支持的 {name}: {aggregation}")
            return
        if not callable(aggregation):
            raise TypeError(f"{name} 必须是内置字符串或 callable")
        if isinstance(aggregation, nn.Module) and any(
            True for _ in aggregation.parameters()
        ):
            raise ValueError(f"{name} callable 不得持有可学习参数")

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MHD_Node) and self.id == other.id

    def reset(self) -> 'MHD_Node':
        self.feature_message.reset()
        self.gradient_message.reset()
        return self

    def to_device(self, device: torch.device) -> 'MHD_Node':
        default_zero_unchanged = (
            id(self.gradient_message.initial_state)
            == self._default_gradient_initial_identity
            and self.gradient_message.initial_state._version
            == self._default_gradient_initial_version
        )
        self.feature_message.to_device(device)
        self.gradient_message.to_device(device)
        if default_zero_unchanged:
            self._remember_default_zero_gradient()
        return self

    @staticmethod
    def _aggregate_messages(
        current: torch.Tensor,
        incomings: Sequence[torch.Tensor],
        aggregation: Union[str, Callable[[Optional[torch.Tensor], Sequence[torch.Tensor]], torch.Tensor]],
        memory: bool = False,
    ) -> torch.Tensor:
        incoming_list = list(incomings)
        if not incoming_list:
            return current
        for incoming in incoming_list:
            if current.device != incoming.device:
                raise ValueError(
                    f"Message 聚合设备不一致: {current.device} vs {incoming.device}"
                )
        if callable(aggregation) and not isinstance(aggregation, str):
            result = aggregation(current if memory else None, tuple(incoming_list))
            if not isinstance(result, torch.Tensor):
                raise TypeError("自定义 Message aggregation 必须返回 Tensor")
            return result
        count = len(incoming_list) + int(memory)
        result = current if memory else incoming_list.pop(0)
        if aggregation in {"sum", "avg"}:
            for incoming in incoming_list:
                result = result + incoming
            return result / count if aggregation == "avg" else result
        if aggregation == "max":
            for incoming in incoming_list:
                result = torch.maximum(result, incoming)
            return result
        if aggregation == "min":
            for incoming in incoming_list:
                result = torch.minimum(result, incoming)
            return result
        if aggregation == "mul":
            for incoming in incoming_list:
                result = result * incoming
            return result
        raise ValueError(f"不支持的 Message aggregation: {aggregation}")

    def aggregate_messages(
        self,
        current: torch.Tensor,
        incomings: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """Evaluate this Node's one differentiable Message aggregation."""
        values = [incomings] if isinstance(incomings, torch.Tensor) else list(incomings)
        return self._aggregate_messages(current, values, self.aggregation, self.memory)


@dataclass
class MHD_Edge:
    """Hyperedge carrying one ordered sequence of differentiable Operations."""

    @dataclass(eq=False)
    class Operation:
        """One transform owned by an :class:`MHD_Edge`."""

        function: Union[str, nn.Module, partial, Callable]

        def __post_init__(self) -> None:
            if isinstance(self.function, str):
                if not self.function.startswith("."):
                    raise ValueError("字符串 Operation 必须以 '.' 开头")
            elif not callable(self.function):
                raise TypeError("Operation 必须包装字符串、nn.Module、partial 或 callable")

        def to_device(self, device: torch.device) -> 'MHD_Edge.Operation':
            if isinstance(self.function, nn.Module):
                self.function = self.function.to(device, non_blocking=True)
            return self

    id: int
    name: str
    edge_operations: List[Operation]

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("边 id 必须是非负整数")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("边 name 必须是非空字符串")
        if not isinstance(self.edge_operations, list):
            raise TypeError("edge_operations 必须是 list")
        if not self.edge_operations:
            raise ValueError("edge_operations 不能为空")
        if not all(isinstance(operation, MHD_Edge.Operation) for operation in self.edge_operations):
            raise TypeError("edge_operations 中每一项都必须是 MHD_Edge.Operation")

    def __hash__(self):
        """基于ID的哈希函数"""
        return hash(self.id)

    def __eq__(self, other):
        """基于ID的相等判断"""
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id

    def to_device(self, device: torch.device) -> 'MHD_Edge':
        """
        将边中的可学习模块迁移到指定设备

        Args:
            device: 目标计算设备

        Returns:
            设备迁移后的边自身
        """
        for operation in self.edge_operations:
            operation.to_device(device)
        return self

    def named_edge_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Return unique learnable parameters using stable edge-local names."""
        result: List[Tuple[str, nn.Parameter]] = []
        seen: Set[int] = set()
        for operation_index, operation in enumerate(self.edge_operations):
            function = operation.function
            if not isinstance(function, nn.Module):
                continue
            for parameter_name, parameter in function.named_parameters():
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                result.append((f"op{operation_index}.{parameter_name}", parameter))
        return result

    @staticmethod
    def _apply_callable_to_list(op, tensor_list: List[torch.Tensor]):
        """Call one Operation with Sort-Matrix-ordered positional Messages.

        V3 guessed between positional, list-valued and elementwise calls by
        swallowing exceptions.  V4 deliberately has one deterministic rule:
        one Message is one argument and multiple Messages are positional
        arguments.  An Operation that wants elementwise application can return
        ``[fn(message) for message in messages]`` explicitly; real computation
        errors are therefore never mistaken for a different calling convention.
        """
        if len(tensor_list) == 1:
            return op(tensor_list[0])
        return op(*tensor_list)

    def execute_edge_operations(self, input_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        执行 edge_operations 操作序列

        多个 Message 按 Sort Matrix 顺序作为位置参数传入；字符串操作逐
        Message 应用。最终输出统一转换为 Message 列表。

        Args:
            input_list: 输入张量列表（头节点状态，按 sort_matrix 排序）

        Returns:
            输出张量列表
        """
        data = input_list
        for operation in self.edge_operations:
            op = operation.function
            if isinstance(data, tuple):
                data = list(data)
            if isinstance(data, list):
                if isinstance(op, str):
                    data = [parse_string_operation(op, t) for t in data]
                elif isinstance(op, (nn.Module, partial)) or callable(op):
                    data = self._apply_callable_to_list(op, data)
                else:
                    raise TypeError(f"不支持的操作类型: {type(op)}")
            else:
                if isinstance(op, str):
                    data = parse_string_operation(op, data)
                elif isinstance(op, (nn.Module, partial)) or callable(op):
                    data = op(data)
                else:
                    raise TypeError(f"不支持的操作类型: {type(op)}")
        if isinstance(data, torch.Tensor):
            data = [data]
        elif isinstance(data, tuple):
            data = list(data)
        elif not isinstance(data, list):
            raise TypeError(f"edge_operations 最终输出类型错误: {type(data)}")
        if not all(isinstance(tensor, torch.Tensor) for tensor in data):
            bad_types = [type(value).__name__ for value in data if not isinstance(value, torch.Tensor)]
            raise TypeError(f"edge_operations 输出必须全部为 Tensor，发现: {bad_types}")
        return data

@dataclass(eq=False)
class _MHD_StateVersion:
    node_id: int
    value: torch.Tensor
    producers: Tuple[Tuple[int, int], ...] = ()
    previous: Optional['_MHD_StateVersion'] = None


@dataclass(frozen=True)
class _MHD_ForwardTrace:
    level: int
    edge_id: int
    head_ids: Tuple[int, ...]
    tail_ids: Tuple[int, ...]
    inputs: Tuple[torch.Tensor, ...]
    outputs: Tuple[torch.Tensor, ...]
    input_states: Tuple[_MHD_StateVersion, ...]
    output_states: List[Optional[_MHD_StateVersion]]


@dataclass(eq=False)
class MHD_Topo:
    """Global levels with canonical two-dimensional sparse COO Role/Sort tensors."""

    role_matrices: List[torch.Tensor] = field(default_factory=list)
    sort_matrices: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        self.role_matrices = [self._canonical(m) for m in self.role_matrices]
        self.sort_matrices = [self._canonical(m) for m in self.sort_matrices]
        self._validate_matrix_pair(self.role_matrices, self.sort_matrices, "全局")

    @staticmethod
    def _canonical(matrix: torch.Tensor) -> torch.Tensor:
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError("拓扑矩阵必须是二维 Tensor")
        if matrix.layout not in (torch.strided, torch.sparse_coo):
            raise TypeError("拓扑只支持稠密或 sparse COO Tensor")
        if matrix.dtype == torch.bool or matrix.is_floating_point() or matrix.is_complex():
            raise TypeError("Role/Sort 矩阵必须使用整数 dtype")
        matrix = matrix.to_sparse_coo()
        matrix = torch.sparse_coo_tensor(
            matrix._indices(), matrix._values(), matrix.shape,
            device=matrix.device, dtype=matrix.dtype, check_invariants=True,
        ).coalesce()
        keep = matrix.values() != 0
        return torch.sparse_coo_tensor(
            matrix.indices()[:, keep], matrix.values()[keep], matrix.shape,
            device=matrix.device, dtype=matrix.dtype, is_coalesced=True,
        )

    @staticmethod
    def _entries(matrix: torch.Tensor) -> Dict[Tuple[int, int], int]:
        """Read only stored entries; shared by topology operations and utilities."""
        indices = matrix.indices().detach().cpu().tolist()
        values = matrix.values().detach().cpu().tolist()
        return {(e, n): value for e, n, value in zip(*indices, values)}

    @staticmethod
    def _from_entries(entries, shape, device, dtype=torch.int64):
        cells = sorted(cell for cell, value in entries.items() if value != 0)
        indices = torch.tensor(cells, dtype=torch.int64, device=device).reshape(-1, 2).t()
        values = torch.tensor([entries[cell] for cell in cells], dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            indices, values, shape, device=device, dtype=dtype, is_coalesced=True,
        )

    @staticmethod
    def _validate_matrix_pair(roles, sorts, phase):
        if not roles or not sorts or len(roles) != len(sorts):
            raise ValueError(f"{phase} role_matrices 与 sort_matrices 必须非空且等长")
        shape, device = roles[0].shape, roles[0].device
        for i, (role, sort) in enumerate(zip(roles, sorts)):
            if role.shape != shape or sort.shape != shape:
                raise ValueError(f"{phase}第{i}层 Role/Sort 矩阵形状不一致")
            if role.device != device or sort.device != device:
                raise ValueError(f"{phase}第{i}层 Role/Sort 矩阵设备不一致")
            if not bool(((role.values() == -1) | (role.values() == 1)).all().item()):
                raise ValueError(f"{phase}第{i}层 role 矩阵只允许 -1、0、1")

    def _level_entries(self, level):
        return self._entries(self.role_matrices[level]), self._entries(self.sort_matrices[level])

    def to_device(self, device):
        self.role_matrices = [m.to(device, non_blocking=True) for m in self.role_matrices]
        self.sort_matrices = [m.to(device, non_blocking=True) for m in self.sort_matrices]
        return self

    def get_topo(self, level, edge_id, node_id, matrix_type="role"):
        if matrix_type not in ("role", "sort"):
            raise ValueError("matrix_type 必须是 role 或 sort")
        matrices = self.role_matrices if matrix_type == "role" else self.sort_matrices
        if 0 <= level < len(matrices):
            return self._entries(matrices[level]).get((edge_id, node_id), 0)
        return 0

    def to_list(self, matrix_type="role"):
        """Explicit dense export; ordinary topology operations never densify."""
        if matrix_type not in ("role", "sort"):
            raise ValueError("matrix_type 必须是 role 或 sort")
        matrices = self.role_matrices if matrix_type == "role" else self.sort_matrices
        return [m.to_dense().cpu().tolist() for m in matrices]

    def _content(self):
        return tuple(
            (tuple(m.shape), tuple(self._entries(m).items()))
            for matrices in (self.role_matrices, self.sort_matrices) for m in matrices
        )

    def __hash__(self):
        return hash(self._content())

    def __eq__(self, other):
        return isinstance(other, MHD_Topo) and self.num_levels == other.num_levels and self._content() == other._content()

    def validate_topo(self, num_edges, num_nodes):
        self._validate_matrix_pair(self.role_matrices, self.sort_matrices, "全局")
        for matrix in self.role_matrices:
            if tuple(matrix.shape) != (num_edges, num_nodes):
                raise ValueError(f"拓扑维度必须为边{num_edges}×节点{num_nodes}，实际{matrix.shape}")

    @property
    def num_levels(self):
        return len(self.role_matrices)


class MHD_Graph(nn.Module):
    """
    多超图动态框架核心类 - V5（双向 Message + 全局 Level）

    特性：
    1. Node 同时承载 Feature Message 与 Gradient Message
    2. 两类 Message 均包含 Initial State 与 Current State
    3. 前向与反向从同一套全局 Role/Sort level 定义中显式选路
    4. Operation 支持原生 nn.Module、函数、partial 和受限字符串操作
    5. 同一条 Edge 可跨层复用，表达循环网络的静态展开
    6. 反向使用一次原生 autograd，并按显式 level 屏蔽未选路径
    7. 图合并与可视化保留稳定的 Node/Edge/Topo/Graph 概念

    Author: Souray Meng (孟号丁)
    """

    def __init__(self, nodes: Set[MHD_Node], edges: Set[MHD_Edge], topos: Set[MHD_Topo],
                 device: torch.device = None):
        """
        初始化MHD图

        Args:
            nodes: 节点集合，每个节点包含 Feature/Gradient Message
            edges: 超边集合，每条边包含 Operation 序列
            topos: 拓扑集合（应只包含一个 MHD_Topo 对象）
            device: 计算设备，默认为CUDA(可用)或CPU
        """
        super().__init__()

        # 统一设备配置
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 核心超图对象
        self.nodes = nodes
        self.edges = edges

        # 拓扑验证：确保只有一个拓扑对象
        if len(topos) != 1:
            raise ValueError(f"topos 必须且只能包含一个 MHD_Topo，实际为 {len(topos)}")
        self.topo = next(iter(topos))

        # 统一设备
        self._unify_device()

        # 建立索引系统
        self._node_by_id: Dict[int, MHD_Node] = {}
        self._node_by_name: Dict[str, MHD_Node] = {}
        self._edge_by_id: Dict[int, MHD_Edge] = {}
        self._edge_by_name: Dict[str, MHD_Edge] = {}
        self._build_indices()

        # 参数注册容器。名称只由稳定的数值 ID 组成，用户的边名不会污染 FQN。
        self.edge_module_map = nn.ModuleDict()

        # 获取层级数
        self.num_levels = self.topo.num_levels if self.topo else 0

        # 验证拓扑维度
        if self.topo:
            self.topo.validate_topo(len(self.edges), len(self.nodes))

        # 拓扑排序和参数注册
        self._register_all_params()
        self.compact_topological_sort()
        self._forward_trace: List[_MHD_ForwardTrace] = []
        self._last_forward_levels: Tuple[int, ...] = tuple()

        logger.info(
            "MHD图初始化完成 | 设备=%s 节点=%d 边=%d 层级=%d",
            self.device,
            len(nodes),
            len(edges),
            self.num_levels,
        )

    def _unify_device(self) -> None:
        """
        统一所有组件的计算设备
        """
        if self.topo:
            self.topo = self.topo.to_device(self.device)
        for node in self.nodes:
            node.to_device(self.device)
        for edge in self.edges:
            edge.to_device(self.device)

    def _build_indices(self) -> None:
        """
        构建节点和边的索引字典
        """
        self._node_by_id.clear()
        self._node_by_name.clear()
        self._edge_by_id.clear()
        self._edge_by_name.clear()

        for node in self.nodes:
            if node.id in self._node_by_id:
                raise ValueError(f"节点ID重复: {node.id}")
            if node.name in self._node_by_name:
                raise ValueError(f"节点名称重复: {node.name}")
            self._node_by_id[node.id] = node
            self._node_by_name[node.name] = node

        for edge in self.edges:
            if edge.id in self._edge_by_id:
                raise ValueError(f"边ID重复: {edge.id}")
            if edge.name in self._edge_by_name:
                raise ValueError(f"边名称重复: {edge.name}")
            self._edge_by_id[edge.id] = edge
            self._edge_by_name[edge.name] = edge

        expected_node_ids = list(range(len(self.nodes)))
        expected_edge_ids = list(range(len(self.edges)))
        if sorted(self._node_by_id) != expected_node_ids:
            raise ValueError(f"节点 ID 必须连续且从 0 开始，期望 {expected_node_ids}")
        if sorted(self._edge_by_id) != expected_edge_ids:
            raise ValueError(f"边 ID 必须连续且从 0 开始，期望 {expected_edge_ids}")
        self._nodes_in_id_order = tuple(self._node_by_id[index] for index in expected_node_ids)
        self._edges_in_id_order = tuple(self._edge_by_id[index] for index in expected_edge_ids)

    def update_indices(self, new_nodes: Set[MHD_Node] = None, new_edges: Set[MHD_Edge] = None) -> None:
        """
        更新索引（当节点或边集合发生变化时调用）
        """
        if new_nodes:
            for node in new_nodes:
                self._node_by_id[node.id] = node
                self._node_by_name[node.name] = node
        if new_edges:
            for edge in new_edges:
                self._edge_by_id[edge.id] = edge
                self._edge_by_name[edge.name] = edge

    def to(self, *args, **kwargs) -> 'MHD_Graph':
        """Move registered modules, node states and the two topology matrices together."""
        super().to(*args, **kwargs)
        try:
            target_device = torch._C._nn._parse_to(*args, **kwargs)[0]
        except (AttributeError, TypeError):
            target_device = kwargs.get("device", args[0] if args else None)
        if target_device is not None:
            self.device = torch.device(target_device)
            self._unify_device()
        return self

    def get_node_by_id(self, node_id: int) -> Optional[MHD_Node]:
        """通过ID获取节点"""
        return self._node_by_id.get(node_id)

    def get_node_by_name(self, node_name: str) -> Optional[MHD_Node]:
        """通过名称获取节点"""
        return self._node_by_name.get(node_name)

    def get_edge_by_id(self, edge_id: int) -> Optional[MHD_Edge]:
        """通过ID获取边"""
        return self._edge_by_id.get(edge_id)

    def get_edge_by_name(self, edge_name: str) -> Optional[MHD_Edge]:
        """通过名称获取边"""
        return self._edge_by_name.get(edge_name)

    def compact_topological_sort(self) -> 'MHD_Graph':
        """Compile deterministic execution plans for every global level."""
        if self.topo is None or self.num_levels == 0:
            self._edge_sequence_per_level = []
            self._execution_plan_per_level = []
            return self
        (
            self._edge_sequence_per_level,
            self._execution_plan_per_level,
        ) = self._compile_topology_phase(
            self.topo.role_matrices,
            self.topo.sort_matrices,
            "全局",
        )
        return self

    def _compile_topology_phase(
        self,
        role_matrices: Sequence[torch.Tensor],
        sort_matrices: Sequence[torch.Tensor],
        phase: str,
    ) -> Tuple[List[List[int]], List[Tuple[_MHD_ExecutionStep, ...]]]:
        sequences: List[List[int]] = []
        plans: List[Tuple[_MHD_ExecutionStep, ...]] = []
        for level in range(len(role_matrices)):
            role = MHD_Topo._entries(role_matrices[level])
            sort_values = MHD_Topo._entries(sort_matrices[level])
            heads, tails = defaultdict(list), defaultdict(list)
            for (eid, nid), value in role.items():
                (heads if value < 0 else tails)[eid].append(nid)
            active_edge_ids = sorted(set(heads) | set(tails))
            if not active_edge_ids:
                sequences.append([])
                plans.append(tuple())
                continue
            node_to_out_edges = defaultdict(set)
            for eid, node_ids in tails.items():
                for nid in node_ids:
                    node_to_out_edges[nid].add(eid)
            edge_deps = defaultdict(set)
            for eid, node_ids in heads.items():
                for nid in node_ids:
                    edge_deps[eid].update(node_to_out_edges.get(nid, set()))
                edge_deps[eid].discard(eid)
            edge_in_degree = {eid: len(edge_deps.get(eid, set())) for eid in active_edge_ids}
            reverse_deps = defaultdict(set)
            for eid, deps in edge_deps.items():
                for dep in deps:
                    reverse_deps[dep].add(eid)
            remaining = set(active_edge_ids)
            level_sequence: List[int] = []
            while remaining:
                current = sorted([e for e in remaining if edge_in_degree[e] == 0])
                if not current:
                    edge_names = [self.get_edge_by_id(eid).name for eid in remaining]
                    raise ValueError(f"{phase}第{level}层边拓扑存在环: {edge_names}")
                level_sequence.extend(current)
                for eid in current:
                    remaining.remove(eid)
                    for next_eid in reverse_deps.get(eid, set()):
                        edge_in_degree[next_eid] -= 1
            sequences.append(level_sequence)
            steps: List[_MHD_ExecutionStep] = []
            for edge_id in level_sequence:
                head_ids = heads[edge_id]
                tail_ids = tails[edge_id]
                if not head_ids or not tail_ids:
                    continue
                head_ids.sort(key=lambda nid: (sort_values.get((edge_id, nid), 0), nid))
                tail_ids.sort(key=lambda nid: (sort_values.get((edge_id, nid), 0), nid))
                steps.append(
                    _MHD_ExecutionStep(
                        edge_id,
                        tuple(head_ids),
                        tuple(tail_ids),
                        self._edges_in_id_order[edge_id],
                        tuple(self._nodes_in_id_order[nid] for nid in tail_ids),
                    )
                )
            plans.append(tuple(steps))
        return sequences, plans

    def _register_all_params(self) -> 'MHD_Graph':
        """
        注册边中的可学习模块到 ModuleDict
        """
        registered_by_identity: Dict[int, str] = {}
        for edge in sorted(self.edges, key=lambda x: x.id):
            for idx, operation in enumerate(edge.edge_operations):
                op = operation.function
                if isinstance(op, nn.Module):
                    identity = id(op)
                    if identity not in registered_by_identity:
                        module_name = f"e{edge.id}_o{idx}"
                        self.edge_module_map[module_name] = op.to(self.device)
                        registered_by_identity[identity] = module_name
                    operation.function = self.edge_module_map[registered_by_identity[identity]]
        return self

    def sort_nodes_by_topo(self, level: int = 0, edge_id: int = 0) -> List[Tuple[int, int]]:
        """
        按指定层级和边的 sort_matrix 排序节点

        Args:
            level: 层级索引，默认为0
            edge_id: 边索引

        Returns:
            排序后的 (节点索引, 排序值) 列表
        """
        if self.topo is None or level >= self.num_levels:
            return []
        if edge_id >= self.topo.sort_matrices[level].shape[0]:
            return []
        entries = MHD_Topo._entries(self.topo.sort_matrices[level])
        indexed = [(nid, entries.get((edge_id, nid), 0)) for nid in range(len(self.nodes))]
        return sorted(indexed, key=lambda p: p[1])

    @staticmethod
    def _validate_levels(levels: Sequence[int], num_levels: int, phase: str) -> List[int]:
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            raise TypeError(f"{phase} levels 必须是非空整数序列")
        normalized = list(levels)
        if not normalized:
            raise ValueError(f"{phase} levels 不能为空")
        for level in normalized:
            if not isinstance(level, int):
                raise TypeError(f"{phase} level 必须是整数，实际为 {type(level).__name__}")
            if level < 0 or level >= num_levels:
                raise IndexError(f"层级索引 {level} 超出范围 [0, {num_levels - 1}]")
        return normalized

    def forward(self, levels: Sequence[int]) -> 'MHD_Graph':
        """Route Feature Messages in the exact user-supplied level order."""
        if not self.nodes or not self.edges or self.topo is None:
            return self
        levels = self._validate_levels(levels, self.num_levels, "Forward")
        self._forward_trace = []
        self._last_forward_levels = tuple(levels)
        record_trace = torch.is_grad_enabled()
        states = [
            _MHD_StateVersion(node.id, node.feature_message.current_state)
            for node in self._nodes_in_id_order
        ]
        for level in levels:
            pending = defaultdict(list)

            def flush(node_id):
                incomings = pending.pop(node_id, None)
                if not incomings:
                    return
                node = self._nodes_in_id_order[node_id]
                old = states[node_id]
                value = node.aggregate_messages(old.value, [item[2] for item in incomings])
                # Distinguish every post-aggregation state, including identity
                # aggregations, without copying activation storage.
                if value.requires_grad:
                    value = value.view_as(value)
                state = _MHD_StateVersion(
                    node_id, value,
                    tuple((ti, oi) for ti, oi, _ in incomings) if record_trace else (),
                    old if record_trace and node.memory else None,
                )
                states[node_id] = state
                if record_trace:
                    for ti, oi, _ in incomings:
                        self._forward_trace[ti].output_states[oi] = state

            for step in self._execution_plan_per_level[level]:
                for node_id in step.head_ids:
                    flush(node_id)
                input_states = tuple(states[nid] for nid in step.head_ids)
                head_tensors = [state.value for state in input_states]
                outputs = step.edge.execute_edge_operations(head_tensors)
                outputs = [output.view_as(output) if output.requires_grad else output for output in outputs]
                if len(outputs) != len(step.tail_ids):
                    raise ValueError(
                        f"边 '{step.edge.name}' 输出数量 ({len(outputs)}) 与尾节点数 ({len(step.tail_ids)}) 不匹配"
                    )
                trace_index = len(self._forward_trace)
                if record_trace:
                    self._forward_trace.append(_MHD_ForwardTrace(
                        level, step.edge_id, step.head_ids, step.tail_ids,
                        tuple(head_tensors), tuple(outputs), input_states,
                        [None] * len(outputs),
                    ))
                for output_index, (nid, output) in enumerate(zip(step.tail_ids, outputs)):
                    pending[nid].append((trace_index, output_index, output))
            for node_id in sorted(pending):
                flush(node_id)
            for node, state in zip(self._nodes_in_id_order, states):
                node.feature_message.current_state = state.value
        return self

    def backward(
        self,
        levels: Sequence[int],
        *,
        retain_graph: bool = False,
    ) -> 'MHD_Graph':
        """Route Gradient Messages through the exact selected global levels."""
        return self._backward(
            levels,
            retain_graph=retain_graph,
            loss_scale=1.0,
        )

    def _resolve_backward(self, levels):
        """Validate the selected real dependency and return its scalar root."""
        if not self._forward_trace:
            raise RuntimeError("graph.backward(levels=...) 前必须先执行 graph.forward(levels=...)")
        levels = self._validate_levels(levels, self.num_levels, "Backward")
        overlap = sorted(set(levels).intersection(self._last_forward_levels))
        if overlap:
            raise ValueError(f"同一轮 Forward/Backward levels 不得重叠: {overlap}")

        # Match each requested reverse Edge occurrence to the most recent
        # compatible forward occurrence.  The list order is never normalized.
        unmatched = list(range(len(self._forward_trace)))
        selected_trace_indices: List[int] = []
        selected_node_ids: Set[int] = set()
        backward_steps: List[_MHD_ExecutionStep] = []
        for level in levels:
            for step in self._execution_plan_per_level[level]:
                match_index = None
                for trace_index in reversed(unmatched):
                    trace = self._forward_trace[trace_index]
                    if (
                        trace.edge_id == step.edge_id
                        and trace.tail_ids == step.head_ids
                        and trace.head_ids == step.tail_ids
                    ):
                        match_index = trace_index
                        break
                if match_index is None:
                    raise ValueError(
                        f"Backward level {level} 的边 '{step.edge.name}' 没有兼容的 "
                        "Forward trace；反向 Role/Sort 必须对应真实前向输入输出"
                    )
                unmatched.remove(match_index)
                selected_trace_indices.append(match_index)
                selected_node_ids.update(step.head_ids)
                selected_node_ids.update(step.tail_ids)
                backward_steps.append(step)
        if not selected_trace_indices:
            raise ValueError("Backward levels 没有选择任何与本次 Forward 对应的 Edge")

        # Validate everything before touching messages, hooks, or .grad.
        for node in self._nodes_in_id_order:
            if not node._gradient_initial_is_zero():
                raise ValueError(
                    f"节点 '{node.name}' 的非零 Gradient Initial State 不再支持；"
                    "反向只能从所选范围的唯一可微标量终点开始"
                )
        selected_set = set(selected_trace_indices)

        def dependencies(state):
            if not state.value.requires_grad:
                return
            if state.previous is not None:
                yield state.previous
            for ti, oi in state.producers:
                trace = self._forward_trace[ti]
                if ti in selected_set and trace.outputs[oi].requires_grad:
                    yield from trace.input_states

        produced = {
            state for ti in selected_set
            for state in self._forward_trace[ti].output_states
            if state.value.requires_grad
        }
        # State identity, not Node identity, distinguishes overwrites. Node
        # aggregation (including memory) remains part of the real dependency.
        consumed = set()
        stack = [parent for state in produced for parent in dependencies(state)]
        while stack:
            state = stack.pop()
            if state in consumed:
                continue
            consumed.add(state)
            stack.extend(dependencies(state))
        terminals = produced - consumed
        if len(terminals) != 1:
            names = sorted(self.get_node_by_id(state.node_id).name for state in terminals)
            raise RuntimeError(
                f"所选 Backward 范围必须有唯一可微标量终点，实际终点 {len(terminals)} 个: {names}"
            )
        root = next(iter(terminals))
        if root.value.numel() != 1:
            raise RuntimeError(
                f"所选反向终点 '{self.get_node_by_id(root.node_id).name}' 必须是标量，"
                f"实际形状 {tuple(root.value.shape)}"
            )

        reachable = set()
        def expose(states):
            stack = list(states)
            while stack:
                state = stack.pop()
                if state in reachable or not state.value.requires_grad:
                    continue
                reachable.add(state)
                if state.previous is not None:
                    stack.append(state.previous)

        expose([root])
        for ti in selected_trace_indices:
            trace = self._forward_trace[ti]
            differentiable = [
                state for state, output in zip(trace.output_states, trace.outputs)
                if output.requires_grad
            ]
            if not differentiable:
                continue
            if not any(state in reachable for state in differentiable):
                raise ValueError(
                    f"Backward 边 '{self.get_edge_by_id(trace.edge_id).name}' "
                    "在给定 level 顺序中不可达"
                )
            expose(trace.input_states)

        return root, selected_set, selected_node_ids

    def _backward(
        self,
        levels: Sequence[int],
        *,
        retain_graph: bool,
        loss_scale: float,
    ) -> 'MHD_Graph':
        """Route Gradient Messages through selected global levels.

        The supplied sequence is interpreted exactly as written.  Every
        backward Edge occurrence must reverse one compatible occurrence from
        the latest forward trace.  PyTorch still performs one native backward;
        hooks at the recorded Edge outputs block all unselected paths.
        """
        root, selected_set, selected_node_ids = self._resolve_backward(levels)
        hook_handles = []
        captured_node_gradients: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # Gradient Message needs the gradient value, but non-leaf ``retain_grad``
        # would also allocate ``Tensor.grad`` for every intermediate. Capture it
        # once through a hook instead, keeping large-model activation memory near
        # native autograd behavior.
        for node in self._nodes_in_id_order:
            node.gradient_message.reset()
            feature = node.feature_message.current_state
            if feature.is_leaf and feature.grad is not None:
                feature.grad = None
            if node.id in selected_node_ids and feature.requires_grad:
                def capture(gradient: torch.Tensor, target=node.id) -> torch.Tensor:
                    captured_node_gradients[target].append(gradient.detach())
                    return gradient
                hook_handles.append(feature.register_hook(capture))

        def block_output(trace_index: int, output_index: int):
            def hook(gradient: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(gradient)
            return hook

        for trace_index, trace in enumerate(self._forward_trace):
            if trace_index in selected_set:
                continue
            for output_index, output in enumerate(trace.outputs):
                if output.requires_grad:
                    hook_handles.append(
                        output.register_hook(block_output(trace_index, output_index))
                    )

        try:
            torch.autograd.backward(
                root.value,
                torch.ones_like(root.value) * float(loss_scale),
                retain_graph=retain_graph,
            )
        finally:
            for handle in hook_handles:
                handle.remove()

        selected_parameter_ids = {
            id(parameter)
            for trace_index in selected_set
            for _, parameter in self._edges_in_id_order[
                self._forward_trace[trace_index].edge_id
            ].named_edge_parameters()
        }
        for parameter in self.parameters():
            if id(parameter) not in selected_parameter_ids:
                parameter.grad = None

        for node in self._nodes_in_id_order:
            if node.id not in selected_node_ids:
                node.gradient_message.reset()
                continue
            gradients = captured_node_gradients.get(node.id)
            if not gradients:
                node.gradient_message.reset()
            else:
                gradient = gradients[0].clone()
                for contribution in gradients[1:]:
                    gradient = gradient + contribution
                node.gradient_message.current_state = gradient
        if not retain_graph:
            self._forward_trace = []
            self._last_forward_levels = tuple()
        return self

    # ---------- 图合并辅助 ----------
    @staticmethod
    def _merge_tensors(tensors: List[torch.Tensor], label: str) -> torch.Tensor:
        """Copy equal numerical states into a fresh graph, without old history."""
        reference = tensors[0]
        for tensor in tensors[1:]:
            if (
                tensor.shape != reference.shape or tensor.dtype != reference.dtype
                or tensor.device != reference.device
                or tensor.requires_grad != reference.requires_grad
                or not torch.equal(tensor, reference)
            ):
                raise ValueError(f"合并节点状态冲突: {label}")
        return reference.detach().clone().requires_grad_(reference.requires_grad)

    @classmethod
    def merge_graph(cls, graphs: Set['MHD_Graph'], device: torch.device = None) -> 'MHD_Graph':
        """Deterministically merge compatible graphs by node and edge name."""
        if not graphs:
            raise ValueError("图集合不能为空")
        target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ordered_graphs = sorted(
            graphs,
            key=lambda graph: (
                tuple(sorted(node.name for node in graph.nodes)),
                tuple(sorted(edge.name for edge in graph.edges)),
            ),
        )
        node_groups: Dict[str, List[MHD_Node]] = defaultdict(list)
        edge_groups: Dict[str, List[MHD_Edge]] = defaultdict(list)
        for graph in ordered_graphs:
            for node in graph.nodes:
                node_groups[node.name].append(node)
            for edge in graph.edges:
                edge_groups[edge.name].append(edge)

        def compatible_value(left: Any, right: Any) -> bool:
            if isinstance(left, str) or isinstance(right, str):
                return left == right
            return left is right

        merged_nodes: Set[MHD_Node] = set()
        name_to_global_nid: Dict[str, int] = {}
        for global_nid, name in enumerate(sorted(node_groups)):
            grouped = node_groups[name]
            reference = grouped[0].aggregation
            if not all(
                compatible_value(reference, node.aggregation)
                for node in grouped[1:]
            ):
                raise ValueError(f"节点 '{name}' 的 aggregation 不兼容")
            if any(node.memory != grouped[0].memory for node in grouped[1:]):
                raise ValueError(f"节点 '{name}' 的 memory 不兼容")
            feature = MHD_Node.Message(
                cls._merge_tensors(
                    [node.feature_message.initial_state for node in grouped], f"节点 {name!r} 的 feature_message.initial_state"
                ),
                cls._merge_tensors(
                    [node.feature_message.current_state for node in grouped], f"节点 {name!r} 的 feature_message.current_state"
                ),
            )
            gradient = MHD_Node.Message(
                cls._merge_tensors(
                    [node.gradient_message.initial_state for node in grouped], f"节点 {name!r} 的 gradient_message.initial_state"
                ),
                cls._merge_tensors(
                    [node.gradient_message.current_state for node in grouped], f"节点 {name!r} 的 gradient_message.current_state"
                ),
            )
            merged_nodes.add(
                MHD_Node(
                    global_nid,
                    name,
                    feature,
                    gradient,
                    grouped[0].aggregation,
                    memory=grouped[0].memory,
                )
            )
            name_to_global_nid[name] = global_nid

        graph_id_to_global_node = {
            id(graph): {
                node.id: name_to_global_nid[node.name] for node in graph.nodes
            }
            for graph in ordered_graphs
        }

        def validate_operations(
            edge_name: str,
            operation_groups: Sequence[Optional[List[Any]]],
            label: str,
        ) -> None:
            reference = operation_groups[0]
            for operations in operation_groups[1:]:
                if reference is None or operations is None:
                    if reference is not operations:
                        raise ValueError(f"同名边 '{edge_name}' 的 {label} 不兼容")
                    continue
                if len(reference) != len(operations):
                    raise ValueError(f"同名边 '{edge_name}' 的 {label} 长度不兼容")
                for left, right in zip(reference, operations):
                    left_function = left.function
                    right_function = right.function
                    if isinstance(left_function, str) and isinstance(right_function, str):
                        valid = left_function == right_function
                    elif isinstance(left_function, nn.Module) and isinstance(right_function, nn.Module):
                        stateful = bool(left_function.state_dict()) or bool(right_function.state_dict())
                        valid = left_function is right_function if stateful else type(left_function) is type(right_function)
                    else:
                        valid = left_function is right_function
                    if not valid:
                        raise ValueError(
                            f"同名边 '{edge_name}' 的 {label} 操作不兼容"
                        )

        merged_edges: Set[MHD_Edge] = set()
        name_to_global_eid: Dict[str, int] = {}
        for global_eid, name in enumerate(sorted(edge_groups)):
            grouped = edge_groups[name]
            validate_operations(
                name,
                [edge.edge_operations for edge in grouped],
                "edge_operations",
            )
            merged_edges.add(
                MHD_Edge(
                    global_eid,
                    name,
                    grouped[0].edge_operations.copy(),
                )
            )
            name_to_global_eid[name] = global_eid

        graph_id_to_global_edge = {
            id(graph): {
                edge.id: name_to_global_eid[edge.name] for edge in graph.edges
            }
            for graph in ordered_graphs
        }
        max_levels = max(graph.num_levels for graph in ordered_graphs)

        def merge_topology() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            merged_roles: List[torch.Tensor] = []
            merged_sorts: List[torch.Tensor] = []
            for level in range(max_levels):
                cells = {}
                for graph in ordered_graphs:
                    if level >= graph.num_levels:
                        continue
                    source_role, source_sort = graph.topo._level_entries(level)
                    edge_map = graph_id_to_global_edge[id(graph)]
                    node_map = graph_id_to_global_node[id(graph)]
                    for local_edge, local_node in source_role.keys() | source_sort.keys():
                        local_cell = (local_edge, local_node)
                        value = (source_role.get(local_cell, 0), source_sort.get(local_cell, 0))
                        cell = (edge_map[local_edge], node_map[local_node])
                        if cell in cells and cells[cell] != value:
                            raise ValueError(
                                f"合并拓扑在 level={level}, edge={cell[0]}, node={cell[1]} 发生冲突"
                            )
                        cells[cell] = value
                shape = (len(merged_edges), len(merged_nodes))
                role = MHD_Topo._from_entries(
                    {cell: value[0] for cell, value in cells.items()}, shape, target_device,
                )
                sort = MHD_Topo._from_entries(
                    {cell: value[1] for cell, value in cells.items()}, shape, target_device,
                )
                merged_roles.append(role)
                merged_sorts.append(sort)
            return merged_roles, merged_sorts

        merged_roles, merged_sorts = merge_topology()
        merged_topo = MHD_Topo(merged_roles, merged_sorts)
        return cls(
            nodes=merged_nodes,
            edges=merged_edges,
            topos={merged_topo},
            device=target_device,
        )
