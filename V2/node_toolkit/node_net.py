# -*- coding: utf-8 -*-
"""
Multi Hypergraph Dynamic Node Framework (MHD NodeF)
多超图动态节点框架 (MHD NodeF)

核心设计目标 (Core Design Goals):
1. 构建多子图超图网络（子图合并为全局超图）
   Build multi-subgraph hypergraph network (subgraphs merged into global hypergraph)
2. 拓扑驱动的张量流（无冗余张量转换）
   Topology-driven tensor flow (no redundant tensor conversion)
3. 哈希冲突解决（安全的张量字段处理）
   Hash conflict resolution (safe tensor field processing)
4. 易导出架构（可序列化数据结构）
   Exportable architecture (serializable data structures)
5. 结构化拓扑矩阵（支持扩展属性）
   Structured topology matrix (support extended attributes)

核心特性 (Core Features):
- 超图节点/边抽象（直接张量赋值）
  Hypergraph node/edge abstraction (direct tensor assignment)
- 基于拓扑排序的有向无环图前向传播
  Topological sort-based DAG forward propagation
- 动态边操作网络（支持字符串/Module类型操作）
  Dynamic edge operation network (support string/Module type operations)
- 多子图到全局图的映射（自动拓扑合并）
  Multi-subgraph to global graph mapping (automatic topology merging)

Author: Souray孟号丁
Version: 2.0
License: MIT
Release Date: 2026
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, TypeVar, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings
import re

# ===================== 全局配置 (Global Configuration) =====================
# 忽略非关键警告 (Ignore non-critical warnings)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 全局设备/数据类型配置 (Global device/dtype configuration)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ===================== 类型定义 (Type Definitions) =====================
# 张量类型别名 (Tensor type alias)
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# ===================== 全局函数注册表 (Global Function Registry) =====================
# 节点头函数：边输入前的张量预处理 (Node head functions: tensor preprocessing before edge input)
MHD_NODE_HEAD_FUNCS: Dict[str, Callable[..., Any]] = {
    "share": lambda tensor: tensor.clone(memory_format=torch.contiguous_format),
}

# 节点尾函数：边输出后的张量聚合 (Node tail functions: tensor aggregation after edge output)
MHD_NODE_TAIL_FUNCS: Dict[str, Callable[..., Any]] = {
    "sum": lambda tensors: sum(tensors),
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],
    "mul": lambda tensors: torch.prod(torch.stack(tensors), dim=0)
}

# ===================== 工具函数 (Utility Functions) =====================
def MHD_sort_nodes_by_topo_value(topos: Set['MHD_Topo'], edge_id: int = 0) -> List[Tuple[int, int]]:
    """
    按sort类型的拓扑值排序节点 (Sort nodes by sort type topology values)
    
    参数 (Parameters):
        topos (Set[MHD_Topo]): 拓扑集合
        edge_id (int): 边ID，默认0
    
    返回 (Returns):
        List[Tuple[int, int]]: 排序后的(节点ID, 拓扑值)列表
    """
    # 获取sort类型的拓扑矩阵 (Get sort type topology matrix)
    sort_topo = None
    for topo in topos:
        if topo.type == "sort":
            sort_topo = topo
            break
    
    if sort_topo is None:
        # 无sort拓扑时创建默认值 (Create default values if no sort topology)
        role_topo = next(iter(topos)) if topos else None
        if role_topo:
            sort_values = torch.zeros_like(role_topo.value)
        else:
            sort_values = torch.tensor([])
    else:
        sort_values = sort_topo.value
    
    # 提取指定边的拓扑值并排序 (Extract topology values of specified edge and sort)
    if edge_id < sort_values.shape[0]:
        indexed_nodes = list(enumerate(sort_values[edge_id].tolist()))
    else:
        indexed_nodes = []
    
    return sorted(indexed_nodes, key=lambda p: p[1] if p[1] is not None else 0)

def MHD_flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    展平张量（保留批次维度） (Flatten tensor (keep batch dimension))
    
    参数 (Parameters):
        x (torch.Tensor): 输入张量
    
    返回 (Returns):
        torch.Tensor: 展平后的张量
    """
    if x.dim() > 2:
        x_flat = x.reshape(x.shape[0], -1)
    else:
        x_flat = x
    return x_flat

def extract_operation_name(op: Union[str, nn.Module]) -> str:
    """
    提取操作名称（移除路径/参数） (Extract operation name (remove path/parameters))
    
    参数 (Parameters):
        op (Union[str, nn.Module]): 操作（字符串/Module）
    
    返回 (Returns):
        str: 简化后的操作名称
    """
    if isinstance(op, nn.Module):
        op_name = op.__class__.__name__
    elif isinstance(op, str):
        op_name = re.sub(r'\(.*\)', '', op).strip()
    else:
        op_name = str(op)

    op_name = op_name.replace("torch.nn.modules.", "").replace("torch.nn.", "")
    return op_name

def validate_node_group_consistency(node_group: Set[str], sub_node_map: Dict[str, Tuple[str, int, 'MHD_Node']]) -> None:
    """
    验证节点组的维度/函数一致性 (Validate dimension/function consistency of node group)
    
    参数 (Parameters):
        node_group (Set[str]): 待合并的节点名称集合
        sub_node_map (Dict): 子节点映射表
    
    异常 (Raises):
        ValueError: 维度/函数不一致时抛出异常
    """
    if len(node_group) <= 1:
        return  # 单节点无需校验 (No validation for single node)
    
    # 提取组内所有节点 (Extract all nodes in group)
    group_nodes = []
    for node_name in node_group:
        if node_name in sub_node_map:
            _, _, node = sub_node_map[node_name]
            group_nodes.append(node)
    
    # 校验维度一致性 (Validate dimension consistency)
    ref_shape = group_nodes[0].value.shape
    for node in group_nodes[1:]:
        if node.value.shape != ref_shape:
            raise ValueError(
                f"节点维度不一致 (Node dimension mismatch)！"
                f"节点 {node.name} 形状 {node.value.shape} 与参考节点 {group_nodes[0].name} 形状 {ref_shape} 不匹配"
            )
    
    # 校验函数一致性 (Validate function consistency)
    ref_func = group_nodes[0].func
    for node in group_nodes[1:]:
        if node.func != ref_func:
            raise ValueError(
                f"节点函数配置不一致 (Node function config mismatch)！"
                f"节点 {node.name} func {node.func} 与参考节点 {group_nodes[0].name} func {ref_func} 不匹配"
            )

def get_merged_node_value(node_group: Set[str], sub_node_map: Dict[str, Tuple[str, int, 'MHD_Node']]) -> torch.Tensor:
    """
    计算合并节点的初始值（组内节点均值） (Calculate initial value of merged node (mean of group nodes))
    
    参数 (Parameters):
        node_group (Set[str]): 节点组
        sub_node_map (Dict): 子节点映射表
    
    返回 (Returns):
        torch.Tensor: 均值初始化的张量
    """
    group_nodes = []
    for node_name in node_group:
        if node_name in sub_node_map:
            _, _, node = sub_node_map[node_name]
            group_nodes.append(node)
    
    if len(group_nodes) == 1:
        return group_nodes[0].value.clone()
    
    # 计算均值 (Calculate mean)
    merged_tensor = torch.stack([n.value for n in group_nodes]).mean(dim=0)
    return merged_tensor

# ===================== 核心操作函数 (Core Operation Functions) =====================
def MHD_concat(tensors: List[torch.Tensor], sorted_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    按预缓存的拓扑排序结果拼接张量 (Concatenate tensors after sorting by cached topology values)
    
    参数 (Parameters):
        tensors (List[torch.Tensor]): 输入张量列表
        sorted_pairs (List[Tuple[int, int]]): 预排序的(节点ID, 拓扑值)列表
    
    返回 (Returns):
        torch.Tensor: 拼接后的张量
    """
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs if i < len(tensors)]
    return torch.cat(sorted_tensors, dim=1)

def MHD_matmul(tensors: List[torch.Tensor], sorted_pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """
    按预缓存的拓扑排序结果执行矩阵乘法 (Perform matrix multiplication after sorting by cached topology values)
    
    参数 (Parameters):
        tensors (List[torch.Tensor]): 输入张量列表
        sorted_pairs (List[Tuple[int, int]]): 预排序的(节点ID, 拓扑值)列表
    
    返回 (Returns):
        torch.Tensor: 矩阵乘法结果
    
    异常 (Raises):
        ValueError: 输入张量数量不为2时抛出
    """
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs if i < len(tensors)]
    if len(sorted_tensors) != 2:
        raise ValueError(f"Matmul需要2个输入张量 (Matmul requires 2 input tensors)，实际输入 (Got) {len(sorted_tensors)}")
    return torch.matmul(*sorted_tensors)

# 边输入函数注册表 (Edge input function registry)
MHD_EDGE_IN_FUNCS: Dict[str, Callable[..., Any]] = {
    "concat": MHD_concat,
    "matmul": MHD_matmul,
}

def MHD_split(x: torch.Tensor, sorted_pairs: List[Tuple[int, int]], node_channels: List[int]) -> List[torch.Tensor]:
    """
    按预缓存的拓扑排序结果分割张量 (Split tensor by cached topology values)
    
    参数 (Parameters):
        x (torch.Tensor): 输入张量
        sorted_pairs (List[Tuple[int, int]]): 预排序的(节点ID, 拓扑值)列表
        node_channels (List[int]): 节点通道数列表
    
    返回 (Returns):
        List[torch.Tensor]: 分割后的张量列表
    """
    sorted_nodes = sorted_pairs
    sorted_original_indices = [p[0] for p in sorted_nodes if p[0] < len(node_channels)]
    sorted_channel_sizes = [node_channels[i] for i in sorted_original_indices]

    split_tensors = torch.split(x, sorted_channel_sizes, dim=1)
    tensor_map = {idx: t for idx, t in zip(sorted_original_indices, split_tensors)}
    
    result = []
    for i in range(len(node_channels)):
        result.append(tensor_map.get(i, torch.zeros(x.shape[0], node_channels[i], device=x.device, dtype=x.dtype)))
    
    return result

def MHD_svd(x: torch.Tensor, sorted_pairs: List[Tuple[int, int]], node_channels: List[int]) -> List[torch.Tensor]:
    """
    按预缓存的拓扑排序结果执行SVD分解 (Perform SVD decomposition by cached topology values)
    
    参数 (Parameters):
        x (torch.Tensor): 输入张量
        sorted_pairs (List[Tuple[int, int]]): 预排序的(节点ID, 拓扑值)列表
        node_channels (List[int]): 节点通道数列表
    
    返回 (Returns):
        List[torch.Tensor]: SVD分解后分配给各节点的张量列表
    """
    x_flat = MHD_flatten_tensor(x)
    U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)

    sorted_nodes = sorted_pairs
    svd_components = [U, S, Vh]
    sorted_tensors = []

    for i, (orig_idx, _) in enumerate(sorted_nodes):
        if orig_idx < len(node_channels):
            comp_idx = i % len(svd_components)
            tensor = svd_components[comp_idx]
            # 调整张量形状匹配节点通道 (Adjust tensor shape to match node channels)
            if tensor.shape[1] != node_channels[orig_idx]:
                tensor = torch.nn.functional.adaptive_avg_pool1d(tensor.unsqueeze(1), node_channels[orig_idx]).squeeze(1)
            sorted_tensors.append((orig_idx, tensor))

    tensor_map = {idx: t for idx, t in sorted_tensors}
    result = []
    for i in range(len(node_channels)):
        result.append(tensor_map.get(i, torch.zeros(x.shape[0], node_channels[i], device=x.device, dtype=x.dtype)))
    
    return result

def MHD_lu(x: torch.Tensor, sorted_pairs: List[Tuple[int, int]], node_channels: List[int]) -> List[torch.Tensor]:
    """
    按预缓存的拓扑排序结果执行LU分解 (Perform LU decomposition by cached topology values)
    
    参数 (Parameters):
        x (torch.Tensor): 输入张量
        sorted_pairs (List[Tuple[int, int]]): 预排序的(节点ID, 拓扑值)列表
        node_channels (List[int]): 节点通道数列表
    
    返回 (Returns):
        List[torch.Tensor]: LU分解后分配给各节点的张量列表
    """
    x_flat = MHD_flatten_tensor(x)
    P, L, U = torch.linalg.lu(x_flat)

    sorted_nodes = sorted_pairs
    lu_components = [L, U, P]
    sorted_tensors = []

    for i, (orig_idx, _) in enumerate(sorted_nodes):
        if orig_idx < len(node_channels):
            comp_idx = i % len(lu_components)
            tensor = lu_components[comp_idx]
            # 调整张量形状匹配节点通道 (Adjust tensor shape to match node channels)
            if tensor.shape[1] != node_channels[orig_idx]:
                tensor = torch.nn.functional.adaptive_avg_pool1d(tensor.unsqueeze(1), node_channels[orig_idx]).squeeze(1)
            sorted_tensors.append((orig_idx, tensor))

    tensor_map = {idx: t for idx, t in sorted_tensors}
    result = []
    for i in range(len(node_channels)):
        result.append(tensor_map.get(i, torch.zeros(x.shape[0], node_channels[i], device=x.device, dtype=x.dtype)))
    
    return result

# 边输出函数注册表 (Edge output function registry)
MHD_EDGE_OUT_FUNCS: Dict[str, Callable[..., Any]] = {
    "split": MHD_split,
    "svd": MHD_svd,
    "lu": MHD_lu,
}

# ===================== 字符串操作包装类 (String Operation Wrapper Class) =====================
class StringOperation(nn.Module):
    """
    字符串定义的张量操作包装类 (Wrapper class for tensor operations defined by string)
    
    支持格式 (Supported format):
        - 无参数: "relu", "sigmoid"
        - 有参数: "__mul__(0.5)", "__add__(0.1)"
    """
    def __init__(self, op_str: str):
        super().__init__()
        self.op_str = op_str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 (Forward propagation)"""
        if '(' in self.op_str and ')' in self.op_str:
            method_name, args_str = self.op_str.split('(', 1)
            args_str = args_str.rstrip(')')
            args = []
            kwargs = {}

            if args_str:
                for arg in args_str.split(','):
                    arg = arg.strip()
                    if not arg:
                        continue
                    if '=' in arg:
                        k, v = arg.split('=', 1)
                        kwargs[k.strip()] = eval(v.strip())
                    else:
                        args.append(eval(arg.strip()))

            return getattr(x, method_name)(*args, **kwargs)
        else:
            return getattr(x, self.op_str)()

# ===================== 核心数据结构 (Core Data Structures) =====================
@dataclass
class MHD_Node:
    """
    超图节点类 (Hypergraph Node Class)
    
    属性 (Attributes):
        id (int): 节点唯一ID
        name (str): 节点名称
        value (torch.Tensor): 节点张量值
        func (Dict[str, str]): 节点函数配置 {"head": 头函数, "tail": 尾函数}
    """
    id: int
    name: str
    value: torch.Tensor
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"})

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        if not isinstance(other, MHD_Node):
            return False
        return self.id == other.id and self.name == other.name

@dataclass
class MHD_Edge:
    """
    超图边类 (Hypergraph Edge Class)
    
    属性 (Attributes):
        id (int): 边唯一ID
        name (str): 边名称
        value (List[Union[str, nn.Module]]): 边操作列表
        func (Dict[str, str]): 边函数配置 {"in": 输入函数, "out": 输出函数}
    """
    id: int
    name: str
    value: List[Union[str, nn.Module]]
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"})

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id and self.name == other.name

@dataclass
class MHD_Topo:
    """
    超图拓扑类 (Hypergraph Topology Class)
    
    属性 (Attributes):
        type (str): 拓扑类型（role/sort/ext等）
        value (torch.Tensor): 拓扑矩阵（int64类型）
    """
    type: str  # 拓扑类型：role/sort/ext等 (Topology type: role/sort/ext, etc.)
    value: torch.Tensor  # 拓扑矩阵值 (Topology matrix values)
    
    def get_topo_value(self, edge_id: int, node_id: int) -> int:
        """
        获取指定边和节点的拓扑值 (Get topology value of specified edge and node)
        
        参数 (Parameters):
            edge_id (int): 边ID
            node_id (int): 节点ID
        
        返回 (Returns):
            int: 拓扑值（无匹配时返回0）
        """
        if edge_id < self.value.shape[0] and node_id < self.value.shape[1]:
            return int(self.value[edge_id, node_id].item())
        return 0
    
    def to_list(self) -> List[List[int]]:
        """
        转换为列表形式（兼容旧逻辑） (Convert to list format (compatible with old logic))
        
        返回 (Returns):
            List[List[int]]: 拓扑矩阵列表
        """
        return self.value.tolist()
    
    def __hash__(self):
        """实现hash方法，支持放入set集合 (Implement hash method for set storage)"""
        return hash((self.type, tuple(self.value.flatten().tolist())))
    
    def __eq__(self, other):
        """实现eq方法，支持set去重 (Implement eq method for set deduplication)"""
        if not isinstance(other, MHD_Topo):
            return False
        return self.type == other.type and torch.equal(self.value, other.value)

# ===================== 拓扑排序 (Topological Sorting) =====================
def MHD_topological_sort(nodes: set[MHD_Node], edges: set[MHD_Edge], topos: Set[MHD_Topo]) -> List[int]:
    """
    超图节点拓扑排序 (Hypergraph node topological sorting)
    
    参数 (Parameters):
        nodes (Set[MHD_Node]): 节点集合
        edges (Set[MHD_Edge]): 边集合
        topos (Set[MHD_Topo]): 拓扑集合
    
    返回 (Returns):
        List[int]: 排序后的节点ID列表
    
    异常 (Raises):
        ValueError: 检测到环时抛出
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_node_ids = {node.id for node in nodes}
    
    # 使用role类型的拓扑进行排序 (Use role type topology for sorting)
    role_topo = None
    for topo in topos:
        if topo.type == "role":
            role_topo = topo
            break
    
    if role_topo is not None and role_topo.value.numel() > 0:
        for edge_id in range(role_topo.value.shape[0]):
            edge_row = role_topo.value[edge_id]
            for node_id in range(role_topo.value.shape[1]):
                role_value = edge_row[node_id].item()
                if role_value == 0:
                    continue
                if role_value < 0:
                    head_id = node_id
                    tail_ids = [
                        nid for nid in range(role_topo.value.shape[1]) 
                        if role_topo.value[edge_id, nid].item() > 0
                    ]
                    for tail_id in tail_ids:
                        graph[head_id].append(tail_id)
                        in_degree[tail_id] += 1

    # 初始化入度为0的节点 (Initialize nodes with in-degree 0)
    for node_id in all_node_ids:
        if node_id not in in_degree:
            in_degree[node_id] = 0

    queue = deque([node_id for node_id in all_node_ids if in_degree[node_id] == 0])
    sorted_node_ids = []

    while queue:
        current_node = queue.popleft()
        sorted_node_ids.append(current_node)
        for neighbor in graph[current_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 检测环 (Detect cycles)
    if len(sorted_node_ids) != len(all_node_ids):
        raise ValueError(
            f"超图检测到环 (Hypergraph detected cycle)！环中节点 (Nodes in cycle): "
            f"{all_node_ids - set(sorted_node_ids)}"
        )
    return sorted_node_ids

# ===================== 动态网络 (Dynamic Network) =====================
class DNet(nn.Module):
    """
    超边操作动态网络 (Dynamic network for hyperedge operations)
    
    核心功能 (Core Functions):
        - 支持nn.Module/字符串类型的操作组合
        - 自动处理设备一致性
    """
    def __init__(self, operations: List[Union[str, nn.Module]], device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        seq_ops = []
        self.op_names = []

        for op in operations:
            self.op_names.append(extract_operation_name(op))

            if isinstance(op, nn.Module):
                # 确保Module在指定设备上 (Ensure Module is on specified device)
                op = op.to(self.device)
                seq_ops.append(op)
            elif isinstance(op, str):
                seq_ops.append(StringOperation(op))
            else:
                raise ValueError(
                    f"不支持的操作类型 (Unsupported operation type): {type(op)}，"
                    f"仅支持nn.Module/string (only nn.Module/string supported)"
                )

        self.filter = nn.Sequential(*seq_ops)
        self.original_operations = operations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (Forward propagation)
        
        参数 (Parameters):
            x (torch.Tensor): 输入张量
        
        返回 (Returns):
            torch.Tensor: 操作后的张量
        """
        # 确保输入张量在指定设备上 (Ensure input tensor is on specified device)
        x = x.to(self.device)
        return self.filter(x)

# ===================== 超图动态网络 (Hypergraph Dynamic Network) =====================
class HDNet(nn.Module):
    """
    超图动态网络（单子网） (Hypergraph dynamic network (single subnet))
    
    核心功能 (Core Functions):
        - 拓扑驱动的前向传播
        - 节点值梯度追踪
        - 预缓存拓扑排序结果
    """
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topos: Set[MHD_Topo], device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.node_id2obj = {node.id: node for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        
        # 转换拓扑为tensor类型 (Convert topology to tensor type)
        self.topos = set()
        for topo in topos:
            if isinstance(topo.value, list):
                tensor_value = torch.tensor(topo.value, dtype=torch.int64, device=self.device)
            else:
                tensor_value = topo.value.to(self.device)
            self.topos.add(MHD_Topo(type=topo.type, value=tensor_value))

        # 验证拓扑维度 (Validate topology dimensions)
        self._validate_topo()
        # 拓扑排序 (Topological sorting)
        self.sorted_node_ids = MHD_topological_sort(nodes, edges, self.topos)
        print(f"✅ 拓扑排序完成 (Topological sort completed): "
              f"{[self.node_id2obj[nid].name for nid in self.sorted_node_ids]}")

        # 初始化边操作网络 (Initialize edge operation networks)
        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.value, device=self.device)

        # 节点值初始化（注册为可训练参数） (Node value initialization as trainable parameters)
        self.node_values = nn.ParameterDict()
        for node in nodes:
            # 确保节点张量在指定设备上 (Ensure node tensor is on specified device)
            node_value = node.value.to(self.device)
            param = nn.Parameter(node_value, requires_grad=True)
            self.node_values[str(node.id)] = param

        # 预缓存所有边的拓扑排序结果 (Pre-cache topology sorting results for all edges)
        self.edge_sorted_pairs = {}
        for edge_id in self.edge_id2obj.keys():
            self.edge_sorted_pairs[edge_id] = MHD_sort_nodes_by_topo_value(self.topos, edge_id)

    @property
    def node_name2id(self):
        """节点名称到ID的映射 (Node name to ID mapping)"""
        return {v.name: k for k, v in self.node_id2obj.items()}

    def _validate_topo(self) -> None:
        """
        验证拓扑矩阵维度 (Validate topology matrix dimensions)
        
        异常 (Raises):
            ValueError: 维度不匹配时抛出
        """
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)

        # 验证所有拓扑的维度一致性 (Validate dimension consistency of all topologies)
        for topo in self.topos:
            if topo.value.shape[0] != num_edges:
                raise ValueError(
                    f"{topo.type}类型拓扑矩阵边维度不匹配 (Topology matrix edge dimension mismatch): "
                    f"预期 (expected){num_edges}，实际 (actual){topo.value.shape[0]}"
                )
            if topo.value.shape[1] != num_nodes:
                raise ValueError(
                    f"{topo.type}类型拓扑矩阵节点维度不匹配 (Topology matrix node dimension mismatch): "
                    f"预期 (expected){num_nodes}，实际 (actual){topo.value.shape[1]}"
                )

    def get_topo_by_type(self, type_name: str) -> Optional[MHD_Topo]:
        """
        按类型获取拓扑对象 (Get topology object by type)
        
        参数 (Parameters):
            type_name (str): 拓扑类型
        
        返回 (Returns):
            Optional[MHD_Topo]: 拓扑对象（无匹配时返回None）
        """
        for topo in self.topos:
            if topo.type == type_name:
                return topo
        return None

    def get_node_by_name(self, name: str) -> MHD_Node:
        """
        按名称获取节点 (Get node by name)
        
        参数 (Parameters):
            name (str): 节点名称
        
        返回 (Returns):
            MHD_Node: 节点对象
        
        异常 (Raises):
            ValueError: 节点不存在时抛出
        """
        try:
            node_id = self.node_name2id[name]
            return self.node_id2obj[node_id]
        except KeyError:
            raise ValueError(f"节点名称不存在 (Node name does not exist): {name}")

    def forward(self) -> Dict[str, Tensor]:
        """
        拓扑驱动前向传播 (Topology-driven forward propagation)
        
        返回 (Returns):
            Dict[str, Tensor]: 节点名称到张量的映射
        """
        if not self.node_values:
            return {}
            
        edge_affects_nodes = defaultdict(list)
        # 获取role类型拓扑 (Get role type topology)
        role_topo = self.get_topo_by_type("role")
        
        if role_topo is not None and role_topo.value.numel() > 0:
            for edge_id in self.edge_id2obj.keys():
                if edge_id < role_topo.value.shape[0]:
                    edge_row = role_topo.value[edge_id]
                    tail_node_ids = [
                        nid for nid in range(role_topo.value.shape[1]) 
                        if edge_row[nid].item() > 0
                    ]
                    edge_affects_nodes[edge_id] = tail_node_ids

        # 按拓扑排序处理节点 (Process nodes by topological order)
        for target_node_id in self.sorted_node_ids:
            relevant_edges = [eid for eid, node_ids in edge_affects_nodes.items() if target_node_id in node_ids]

            for edge_id in relevant_edges:
                edge = self.edge_id2obj[edge_id]
                edge_net = self.edge_nets[edge.name]
                
                # 获取头节点 (Get head nodes)
                edge_row = role_topo.value[edge_id] if (role_topo and edge_id < role_topo.value.shape[0]) else []
                head_mask = [val.item() < 0 for val in edge_row] if edge_row.numel() > 0 else []
                head_node_ids = [i for i, val in enumerate(head_mask) if val]

                # 处理头节点张量 (Process head node tensors)
                head_tensors = []
                for node_id in head_node_ids:
                    if str(node_id) in self.node_values:
                        node = self.node_id2obj[node_id]
                        head_func_name = node.func.get("head", "share")
                        head_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](self.node_values[str(node_id)])
                        head_tensors.append(head_tensor)

                if not head_tensors:
                    continue

                # 边输入处理 (Edge input processing)
                sorted_pairs = self.edge_sorted_pairs.get(edge_id, [])
                edge_in_func_name = edge.func.get("in", "concat")
                edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](head_tensors, sorted_pairs)

                # 边操作前向传播 (Edge operation forward propagation)
                edge_output = edge_net(edge_input)

                # 获取尾节点 (Get tail nodes)
                tail_mask = [val.item() > 0 for val in edge_row] if edge_row.numel() > 0 else []
                tail_node_ids = [i for i, val in enumerate(tail_mask) if val]
                tail_node_channels = []
                for node_id in tail_node_ids:
                    if node_id in self.node_id2obj and str(node_id) in self.node_values:
                        tail_node_channels.append(self.node_values[str(node_id)].shape[1])
                    else:
                        tail_node_channels.append(0)

                if not tail_node_channels:
                    continue

                # 边输出处理 (Edge output processing)
                edge_out_func_name = edge.func.get("out", "split")
                tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                    edge_output, sorted_pairs, tail_node_channels
                )

                # 节点值更新 (Node value update)
                for idx, node_id in enumerate(tail_node_ids):
                    if idx < len(tail_tensors) and node_id in self.node_id2obj and str(node_id) in self.node_values:
                        node = self.node_id2obj[node_id]
                        tail_func_name = node.func.get("tail", "sum")
                        tensor = tail_tensors[idx]
                        
                        # 确保张量形状匹配 (Ensure tensor shape matches)
                        if tensor.shape[1] != self.node_values[str(node_id)].shape[1]:
                            tensor = torch.nn.functional.adaptive_avg_pool1d(
                                tensor.unsqueeze(1), 
                                self.node_values[str(node_id)].shape[1]
                            ).squeeze(1)
                        
                        # 聚合更新节点值 (Aggregate and update node value)
                        agg_tensor = MHD_NODE_TAIL_FUNCS[tail_func_name](
                            [self.node_values[str(node_id)], tensor]
                        )
                        self.node_values[str(node_id)] = nn.Parameter(agg_tensor, requires_grad=True)

        # 返回节点名称到张量的映射 (Return node name to tensor mapping)
        return {
            self.node_id2obj[int(node_id)].name: tensor
            for node_id, tensor in self.node_values.items()
        }

# ===================== 多超图动态网络 (Multi-Hypergraph Dynamic Network) =====================
class MHDNet(HDNet):
    """
    多超图动态网络（全局超图） (Multi-hypergraph dynamic network (global hypergraph))
    
    核心功能 (Core Functions):
        - 多子图自动合并为全局超图
        - 节点组一致性校验
        - 显式ID映射避免拓扑错位
    """
    def __init__(
        self,
        hdnet_list: List[Tuple[str, HDNet]],
        node_group: Tuple[Set[str], ...],
        device: torch.device = DEVICE
    ):
        # 显式ID映射表 (Explicit ID mapping tables)
        self.sub2global_node_id = {}  # (suffix, sub_node_id) → global_node_id
        self.sub2global_edge_id = {}  # (suffix, sub_edge_id) → global_edge_id
        self.global2sub_node_id = {}  # global_node_id → (suffix, sub_node_id)
        self.global2sub_edge_id = {}  # global_edge_id → (suffix, sub_edge_id)
        
        # 构建全局超图 (Build global hypergraph)
        global_nodes, global_edges, global_topos = self._build_global_hypergraph(hdnet_list, node_group, device)
        
        # 初始化父类 (Initialize parent class)
        super().__init__(nodes=global_nodes, edges=global_edges, topos=global_topos, device=device)
        
        # 保存原始映射 (Save original mappings)
        self.hdnet_list = hdnet_list
        self.node_group = node_group

    def _build_global_hypergraph(self, hdnet_list: List[Tuple[str, HDNet]], node_group: Tuple[Set[str], ...], device: torch.device) -> Tuple[Set[MHD_Node], Set[MHD_Edge], Set[MHD_Topo]]:
        """
        构建完整的全局超图 (Build complete global hypergraph)
        
        参数 (Parameters):
            hdnet_list (List[Tuple[str, HDNet]]): 子图列表 (名称, 子图)
            node_group (Tuple[Set[str]]): 节点合并组
            device (torch.device): 设备
        
        返回 (Returns):
            Tuple[Set[MHD_Node], Set[MHD_Edge], Set[MHD_Topo]]: 全局节点/边/拓扑
        """
        # ===================== 步骤1：预处理所有子图节点/边 =====================
        sub_node_map = {}  # key=suffix::name, value=(suffix, sub_node_id, sub_node)
        sub_edge_map = {}  # key=suffix::name, value=(suffix, sub_edge_id, sub_edge)
        all_sub_node_names = set()
        
        for suffix, hdnet in hdnet_list:
            # 处理节点 (Process nodes)
            for sub_node_id, sub_node in hdnet.node_id2obj.items():
                global_node_name = f"{suffix}::{sub_node.name}"
                sub_node_map[global_node_name] = (suffix, sub_node_id, sub_node)
                all_sub_node_names.add(global_node_name)
            
            # 处理边 (Process edges)
            for sub_edge_id, sub_edge in hdnet.edge_id2obj.items():
                global_edge_name = f"{suffix}::{sub_edge.name}"
                sub_edge_map[global_edge_name] = (suffix, sub_edge_id, sub_edge)

        # ===================== 步骤2：处理节点合并 =====================
        node_id_counter = 0
        merged_node_map = {}  # key=全局节点名, value=MHD_Node
        sub2global_node = {}  # key=子节点名(suffix::name), value=全局节点名
        
        # 处理需要合并的节点组 (Process node groups to be merged)
        merged_node_names = set()
        for group in node_group:
            # 校验节点组一致性 (Validate node group consistency)
            validate_node_group_consistency(group, sub_node_map)
            
            # 按子图顺序排序节点组 (Sort node group by subgraph order)
            sorted_node_names = sorted(
                group,
                key=lambda x: next((i for i, (suffix, _) in enumerate(hdnet_list) if x.startswith(f"{suffix}::")), 999)
            )
            merged_name = "-".join(sorted_node_names)
            merged_node_names.update(sorted_node_names)
            
            # 计算合并节点的初始值（均值） (Calculate initial value of merged node (mean))
            merged_value = get_merged_node_value(group, sub_node_map).to(device)
            
            # 获取参考节点的函数配置 (Get function config of reference node)
            first_node_name = sorted_node_names[0]
            _, _, base_node = sub_node_map[first_node_name]
            
            # 创建合并节点 (Create merged node)
            merged_node = MHD_Node(
                id=node_id_counter,
                name=merged_name,
                value=merged_value,
                func=base_node.func
            )
            merged_node_map[merged_name] = merged_node
            
            # 记录ID映射 (Record ID mapping)
            for node_name in sorted_node_names:
                s, sn_id, _ = sub_node_map[node_name]
                self.sub2global_node_id[(s, sn_id)] = node_id_counter
                self.global2sub_node_id[node_id_counter] = (s, sn_id)
                sub2global_node[node_name] = merged_name
            
            node_id_counter += 1

        # 处理未被合并的独立节点 (Process unmerged independent nodes)
        unmerged_node_names = all_sub_node_names - merged_node_names
        for node_name in sorted(unmerged_node_names):
            suffix, sub_node_id, sub_node = sub_node_map[node_name]
            
            # 创建独立全局节点 (Create independent global node)
            unmerged_node = MHD_Node(
                id=node_id_counter,
                name=node_name,
                value=sub_node.value.to(device),
                func=sub_node.func
            )
            merged_node_map[node_name] = unmerged_node
            
            # 记录ID映射 (Record ID mapping)
            self.sub2global_node_id[(suffix, sub_node_id)] = node_id_counter
            self.global2sub_node_id[node_id_counter] = (suffix, sub_node_id)
            sub2global_node[node_name] = node_name
            
            node_id_counter += 1

        # ===================== 步骤3：构建全局边 =====================
        edge_id_counter = 0
        merged_edge_map = {}
        
        for global_edge_name, (suffix, sub_edge_id, sub_edge) in sub_edge_map.items():
            # 创建独立全局边 (Create independent global edge)
            merged_edge = MHD_Edge(
                id=edge_id_counter,
                name=global_edge_name,
                value=sub_edge.value,
                func=sub_edge.func
            )
            merged_edge_map[global_edge_name] = merged_edge
            
            # 记录ID映射 (Record ID mapping)
            self.sub2global_edge_id[(suffix, sub_edge_id)] = edge_id_counter
            self.global2sub_edge_id[edge_id_counter] = (suffix, sub_edge_id)
            
            edge_id_counter += 1

        # ===================== 步骤4：构建全局拓扑 =====================
        # 获取所有拓扑类型 (Get all topology types)
        all_topo_types = set()
        for _, hdnet in hdnet_list:
            for topo in hdnet.topos:
                all_topo_types.add(topo.type)
        
        # 构建全局拓扑 (Build global topology)
        global_topos = set()
        num_global_edges = len(merged_edge_map)
        num_global_nodes = len(merged_node_map)
        
        for topo_type in all_topo_types:
            # 初始化全局拓扑矩阵 (Initialize global topology matrix)
            global_topo_value = torch.zeros((num_global_edges, num_global_nodes), dtype=torch.int64, device=device)
            
            # 按全局边ID构建拓扑行 (Build topology rows by global edge ID)
            for global_edge_id in range(num_global_edges):
                if global_edge_id not in self.global2sub_edge_id:
                    continue
                
                # 通过映射表获取子图和子边ID (Get subgraph and sub-edge ID via mapping table)
                suffix, sub_edge_id = self.global2sub_edge_id[global_edge_id]
                hdnet = next(h for s, h in hdnet_list if s == suffix)
                
                # 获取子拓扑 (Get sub topology)
                sub_topo = None
                for t in hdnet.topos:
                    if t.type == topo_type:
                        sub_topo = t
                        break
                
                if sub_topo is None or sub_edge_id >= sub_topo.value.shape[0]:
                    continue
                
                # 获取子拓扑行 (Get sub topology row)
                sub_topo_row = sub_topo.value[sub_edge_id]
                
                # 映射子拓扑到全局拓扑 (Map sub topology to global topology)
                for sub_node_id in range(sub_topo_row.shape[0]):
                    topo_value = sub_topo_row[sub_node_id].item()
                    if topo_value == 0:
                        continue
                    
                    # 通过ID映射找到全局节点ID (Find global node ID via ID mapping)
                    if (suffix, sub_node_id) in self.sub2global_node_id:
                        global_node_id = self.sub2global_node_id[(suffix, sub_node_id)]
                        if 0 <= global_node_id < num_global_nodes:
                            global_topo_value[global_edge_id, global_node_id] = topo_value
            
            # 添加全局拓扑 (Add global topology)
            global_topo = MHD_Topo(type=topo_type, value=global_topo_value)
            global_topos.add(global_topo)

        # ===================== 返回全局超图组件 =====================
        global_nodes = set(merged_node_map.values())
        global_edges = set(merged_edge_map.values())
        
        return global_nodes, global_edges, global_topos

# ===================== 可视化工具 (Visualization Tool) =====================
def generate_mermaid(hdnet: HDNet) -> str:
    """
    生成Mermaid拓扑可视化代码 (Generate Mermaid topology visualization code)
    
    参数 (Parameters):
        hdnet (HDNet): 超图网络
    
    返回 (Returns):
        str: Mermaid代码
    """
    mermaid = [
        "graph TD",
        "",
        "    %% 样式定义 (Style definition)",
        "    classDef nodeStyle fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
        "    classDef edgeStyle fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
        "",
    ]

    # 获取role类型拓扑 (Get role type topology)
    role_topo = None
    for topo in hdnet.topos:
        if topo.type == "role":
            role_topo = topo
            break

    # 添加节点样式 (Add node styles)
    for node_id, node in hdnet.node_id2obj.items():
        mermaid.append(f"    {node.name}:::nodeStyle")
    
    # 添加边样式和连接关系 (Add edge styles and connections)
    if role_topo:
        for edge_id, edge in hdnet.edge_id2obj.items():
            edge_name = edge.name
            if edge_id < role_topo.value.shape[0]:
                edge_row = role_topo.value[edge_id]
            
                # 添加边样式 (Add edge style)
                mermaid.append(f"    {edge_name}:::edgeStyle")
                
                # 获取头/尾节点名称 (Get head/tail node names)
                head_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() < 0]
                tail_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() > 0]
                head_node_names = [hdnet.node_id2obj[nid].name for nid in head_node_ids if nid in hdnet.node_id2obj]
                tail_node_names = [hdnet.node_id2obj[nid].name for nid in tail_node_ids if nid in hdnet.node_id2obj]
                
                # 生成连接关系 (Generate connections)
                for head_node in head_node_names:
                    mermaid.append(f"    {head_node} --> {edge_name}")
                for tail_node in tail_node_names:
                    mermaid.append(f"    {edge_name} --> {tail_node}")
                
                mermaid.append("")

    mermaid_code = "\n".join(mermaid)
    print(mermaid_code)
    return mermaid_code

# ===================== 示例用法 (Example Usage) =====================
def example_mhdnet2():
    """
    MHDNet示例 (MHDNet Example)
    - 3个子图拓扑
    - 节点合并（维度/func一致性校验）
    - MUL/AVG聚合函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"✅ 使用设备 (Using device): {device}")

    # ===================== 子HDNet1 =====================
    nodes_net1 = {
        MHD_Node(
            id=0, 
            name="A1", 
            value=torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=1, 
            name="B1", 
            value=torch.randn(1, 2, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=2, 
            name="D1", 
            value=torch.randn(1, 4, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "mul"}
        ),
    }
    edge1_net1 = [
        nn.Conv3d(3, 2, kernel_size=3, padding=1, bias=False).to(device),
        nn.BatchNorm3d(2).to(device),
        nn.ReLU(inplace=True)
    ]
    edge2_net1 = [
        nn.Conv3d(3, 4, kernel_size=1, padding=0, bias=True).to(device),
        nn.Sigmoid()
    ]
    edges_net1 = {
        MHD_Edge(
            id=0, 
            name="e1_A1_to_B1", 
            value=edge1_net1,
            func={"in": "concat", "out": "split"}
        ),
        MHD_Edge(
            id=1, 
            name="e2_A1_to_D1", 
            value=edge2_net1,
            func={"in": "concat", "out": "split"}
        )
    }
    topos_net1 = {
        MHD_Topo(
            type="role", 
            value=torch.tensor([[-1, 1, 0], [-1, 0, 1]], dtype=torch.int64, device=device)
        ),
        MHD_Topo(
            type="sort",
            value=torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.int64, device=device)
        )
    }
    hdnet1 = HDNet(nodes=nodes_net1, edges=edges_net1, topos=topos_net1, device=device)

    # ===================== 子HDNet2 =====================
    nodes_net2 = {
        MHD_Node(
            id=0, 
            name="A2", 
            value=torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype),  # 与A1维度一致 (Same dimension as A1)
            func={"head": "share", "tail": "sum"}  # 与A1 func一致 (Same func as A1)
        ),
        MHD_Node(
            id=1, 
            name="B2", 
            value=torch.randn(1, 2, 8, 8, 8, device=device, dtype=dtype),  # 与B1维度一致 (Same dimension as B1)
            func={"head": "share", "tail": "sum"}  # 与B1 func一致 (Same func as B1)
        ),
        MHD_Node(
            id=2, 
            name="C2", 
            value=torch.randn(1, 5, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
    }
    edge1_net2 = [
        nn.Conv3d(5, 5, kernel_size=3, padding=1, groups=5, bias=False).to(device),
        nn.GELU(),
        nn.Conv3d(5, 5, kernel_size=1, padding=0, bias=True).to(device)
    ]
    edges_net2 = {
        MHD_Edge(
            id=0, 
            name="e1_A2B2_to_C2", 
            value=edge1_net2,
            func={"in": "concat", "out": "split"}
        )
    }
    topos_net2 = {
        MHD_Topo(
            type="role", 
            value=torch.tensor([[-1, -1, 1]], dtype=torch.int64, device=device)
        ),
        MHD_Topo(
            type="sort",
            value=torch.tensor([[1, 2, 1]], dtype=torch.int64, device=device)
        )
    }
    hdnet2 = HDNet(nodes=nodes_net2, edges=edges_net2, topos=topos_net2, device=device)

    # ===================== 子HDNet3 =====================
    nodes_net3 = {
        MHD_Node(
            id=0, 
            name="C3", 
            value=torch.randn(1, 5, 8, 8, 8, device=device, dtype=dtype),  # 与C2维度一致 (Same dimension as C2)
            func={"head": "share", "tail": "sum"}  # 与C2 func一致 (Same func as C2)
        ),
        MHD_Node(
            id=1, 
            name="D3", 
            value=torch.randn(1, 4, 8, 8, 8, device=device, dtype=dtype),  # 与D1维度一致 (Same dimension as D1)
            func={"head": "share", "tail": "mul"}  # 与D1 func一致 (Same func as D1)
        ),
    }
    edge1_net3 = [
        nn.Conv3d(5, 4, kernel_size=3, padding=1, bias=False).to(device),
        nn.Softplus(),
        '__mul__(0.5)'
    ]
    edges_net3 = {
        MHD_Edge(
            id=0, 
            name="e1_C3_to_D3", 
            value=edge1_net3,
            func={"in": "concat", "out": "split"}
        )
    }
    topos_net3 = {
        MHD_Topo(
            type="role", 
            value=torch.tensor([[-1, 1]], dtype=torch.int64, device=device)
        ),
        MHD_Topo(
            type="sort",
            value=torch.tensor([[1, 1]], dtype=torch.int64, device=device)
        )
    }
    hdnet3 = HDNet(nodes=nodes_net3, edges=edges_net3, topos=topos_net3, device=device)

    # ===================== 扩展HDNet3 =====================
    new_node_E3 = MHD_Node(
        id=2,
        name="E3",
        value=torch.randn(1, 6, 8, 8, 8, device=device, dtype=dtype),
        func={"head": "share", "tail": "avg"}
    )
    
    new_edge_e2_C3_to_E3 = [
        nn.BatchNorm3d(5).to(device),
        nn.ReLU(inplace=True),
        nn.Conv3d(5, 6, kernel_size=1, padding=0, bias=True).to(device),
        '__add__(0.1)'
    ]
    new_edge_obj = MHD_Edge(
        id=1,
        name="e2_C3_to_E3",
        value=new_edge_e2_C3_to_E3,
        func={"in": "concat", "out": "split"}
    )
    
    # 更新节点/边 (Update nodes/edges)
    updated_nodes_net3 = set(hdnet3.node_id2obj.values())
    updated_nodes_net3.add(new_node_E3)
    updated_edges_net3 = set(hdnet3.edge_id2obj.values())
    updated_edges_net3.add(new_edge_obj)
    
    # 更新拓扑 (Update topology)
    role_topo_net3 = hdnet3.get_topo_by_type("role")
    sort_topo_net3 = hdnet3.get_topo_by_type("sort")
    
    # 扩展role拓扑 (Extend role topology)
    updated_role_value = torch.cat([
        torch.cat([role_topo_net3.value, torch.zeros(role_topo_net3.value.shape[0], 1, device=device, dtype=torch.int64)], dim=1),
        torch.tensor([[-1, 0, 1]], device=device, dtype=torch.int64)
    ], dim=0)
    
    # 扩展sort拓扑 (Extend sort topology)
    updated_sort_value = torch.cat([
        torch.cat([sort_topo_net3.value, torch.zeros(sort_topo_net3.value.shape[0], 1, device=device, dtype=torch.int64)], dim=1),
        torch.tensor([[1, 0, 2]], device=device, dtype=torch.int64)
    ], dim=0)
    
    updated_topos_net3 = {
        MHD_Topo(type="role", value=updated_role_value),
        MHD_Topo(type="sort", value=updated_sort_value)
    }
    
    # 重新创建HDNet3 (Recreate HDNet3)
    hdnet3 = HDNet(nodes=updated_nodes_net3, edges=updated_edges_net3, topos=updated_topos_net3, device=device)

    # ===================== 构建全局MHDNet =====================
    hdnet_list = [
        ("net1", hdnet1),
        ("net2", hdnet2),
        ("net3test", hdnet3)
    ]

    # 节点合并组 (Node merge groups)
    node_group = (
        {"net1::A1", "net2::A2"},          # 合并A1/A2 (Merge A1/A2)
        {"net1::B1", "net2::B2"},          # 合并B1/B2 (Merge B1/B2)
        {"net2::C2", "net3test::C3"},      # 合并C2/C3 (Merge C2/C3)
        {"net1::D1", "net3test::D3"},      # 合并D1/D3 (Merge D1/D3)
        {"net3test::E3"},                  # 独立节点E3 (Independent node E3)
    )

    # 创建全局超图 (Create global hypergraph)
    mhdnet = MHDNet(
        hdnet_list=hdnet_list,
        node_group=node_group,
        device=device
    )

    # ===================== 加载输入数据 =====================
    input_tensor = torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype)
    print(f"\n✅ 输入张量形状 (Input Tensor Shape): {input_tensor.shape}")

    # 更新输入节点 (Update input node)
    target_node_name = "net1::A1-net2::A2"
    input_node = mhdnet.get_node_by_name(target_node_name)
    input_node.name = "input"
    mhdnet.node_values[str(input_node.id)] = nn.Parameter(input_tensor, requires_grad=True)

    # ===================== 生成可视化代码 =====================
    print("\n📊 MHD NodeF 拓扑可视化 (Topology Visualization):")
    print("="*80)
    generate_mermaid(mhdnet)
    print("="*80 + "\n")

    # ===================== 运行前向传播 =====================
    print("🚀 执行MHDNet前向传播 (Running MHDNet Forward Pass)...")
    all_features = mhdnet.forward()

    # 打印结果 (Print results)
    print("\n✅ 前向传播完成 (Forward propagation completed)！")
    print(f"\n📈 全局节点总数 (Total global nodes): {len(all_features)}")
    total_params = sum(p.numel() for p in mhdnet.parameters())
    print(f"📈 模型总参数 (Total model parameters): {total_params:,}")
    
    return mhdnet

# ===================== 梯度验证函数 (Gradient Verification) =====================
def verify_gradient(model):
    """
    验证梯度反传 (Verify gradient backpropagation)
    
    参数 (Parameters):
        model (MHDNet): MHDNet模型
    """
    all_features = model.forward()
    if "net1::D1-net3test::D3" in all_features:
        output_tensor = all_features["net1::D1-net3test::D3"]
        loss = output_tensor.sum()
        
        # 梯度清零 (Zero gradients)
        model.zero_grad()
        # 反向传播 (Backward propagation)
        loss.backward()
        
        has_gradient = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                has_gradient = True
                print(f"✅ 参数 (Parameter) {name} 梯度正常 (gradient normal): {param.grad.sum().item():.4f}")
        
        if has_gradient:
            print("\n✅ 梯度反传验证通过 (Gradient backpropagation verification passed)！")
        else:
            print("\n❌ 梯度反传验证失败 (Gradient backpropagation verification failed)！")
    else:
        print("\n❌ 无法验证梯度：输出节点不存在 (Cannot verify gradient: output node not found)")

# ===================== 主执行入口 (Main Entry) =====================
if __name__ == "__main__":
    # 运行示例 (Run example)
    model = example_mhdnet2()
    # 验证梯度 (Verify gradient)
    verify_gradient(model)
