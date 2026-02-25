# -*- coding: utf-8 -*-
"""
Hypergraph Neural Network (HDNet/MHDNet) - Final Release Version
超图神经网络（HDNet/MHDNet）- 最终发布版本

Core Design Goals / 核心设计目标:
1. Build multi-subgraph hypergraph network (merge subgraphs into global graph)
   构建多子图超图网络（将多个子图合并为全局超图）
2. Topology-driven tensor flow (no redundant tensor conversion)
   拓扑驱动的张量流（无冗余张量转换）
3. Hash conflict resolution (safe tensor field handling)
   哈希冲突解决（安全的张量字段处理）
4. Export-friendly architecture (serializable data structures)
   易导出架构（可序列化数据结构）
5. Gradient flow compatibility (support automatic differentiation)
   梯度流兼容（支持自动微分）
6. Structured topology matrix (extendable key-value format)
   结构化拓扑矩阵（可扩展键值对格式）

Key Features / 核心特性:
- Hypergraph node/edge abstraction with direct tensor assignment
  支持直接张量赋值的超图节点/边抽象
- Topological sort for DAG-based forward propagation
  基于拓扑排序的有向无环图前向传播
- Dynamic edge operation network (support string/Module operations)
  动态边操作网络（支持字符串/Module类型操作）
- Multi-subgraph to global graph mapping (automatic topology merging)
  多子图到全局图的映射（自动拓扑合并）
- Gradient-preserving tensor operations (support backpropagation)
  保留梯度的张量操作（支持反向传播）
- Structured topology matrix (role/sort/extendable attributes)
  结构化拓扑矩阵（角色/排序/可扩展属性）

Author: Your Name
Date: 2026
Version: 1.1  # 版本更新：梯度兼容+拓扑矩阵结构化
License: MIT
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings
import re
from collections import namedtuple

# ===================== Global Configuration / 全局配置 =====================
# Suppress unnecessary warnings / 抑制不必要的警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Global device/dtype (reserved for extension) / 全局设备/数据类型（预留扩展使用）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ===================== Type Definition / 类型定义 =====================
# Tensor type alias (reserved for type hinting) / 张量类型别名（预留类型提示使用）
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# 结构化拓扑属性命名元组（兼顾字典可读性和元组性能）
# Structured topology attribute named tuple (balance readability and performance)
# role: 节点角色 (-1=头节点/1=尾节点/0=无连接)
# sort: 排序优先级 (1,2,3...)
# ext: 扩展属性字典（预留时序/分支/注意力等扩展空间）
TopoAttr = namedtuple("TopoAttr", ["role", "sort", "ext"], defaults=[None])

# ===================== Global Function Registry / 全局函数注册表 =====================
# Node head functions (preprocess tensor before edge input)
# 节点头函数（边输入前的张量预处理）
MHD_NODE_HEAD_FUNCS: Dict[str, Callable[..., Any]] = {
    # MODIFIED: 克隆张量时显式保留梯度 / Keep gradient when cloning tensor
    "share": lambda tensor: tensor.clone(memory_format=torch.contiguous_format).requires_grad_(tensor.requires_grad),
}

# Node tail functions (aggregate tensor after edge output)
# 节点尾函数（边输出后的张量聚合）
MHD_NODE_TAIL_FUNCS: Dict[str, Callable[..., Any]] = {
    "sum": lambda tensors: sum(tensors),          # Sum aggregation / 求和聚合
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),  # Average aggregation / 平均聚合
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],  # Max aggregation / 最大值聚合
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],  # Min aggregation / 最小值聚合
}

# ===================== Utility Functions / 工具函数 =====================
def MHD_sort_nodes_by_topo_attr(attrs: List[TopoAttr]) -> List[Tuple[int, int]]:
    """
    MODIFIED: 按拓扑属性的sort字段排序（替代原绝对值排序）
    Sort nodes by sort field of topology attribute (replace original abs index sort)
    
    Args:
        attrs: 拓扑属性列表 / List of topology attributes
    Returns:
        List of (original_index, sorted_index) tuples / (原始索引, 排序索引)元组列表
    """
    indexed_nodes = list(enumerate(attrs))
    # 按sort字段排序 / Sort by sort field
    return sorted(indexed_nodes, key=lambda p: p[1].sort if p[1] is not None else 0)

def MHD_flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten tensor for matrix operations (preserve batch dimension)
    展平张量用于矩阵运算（保留批次维度）
    
    Args:
        x: Input tensor (batch_dim first) / 输入张量（批次维度在前）
    Returns:
        Flattened tensor / 展平后的张量
    """
    if x.dim() > 2:
        # Reshape to (batch_size, feature_dim) / 重塑为(批次大小, 特征维度)
        x_flat = x.reshape(x.shape[0], -1)
    else:
        x_flat = x
    return x_flat

def extract_operation_name(op: Union[str, nn.Module]) -> str:
    """
    Extract clean operation name (remove parameters/module paths)
    提取干净的操作名称（移除参数/模块路径）
    
    Args:
        op: Operation (nn.Module or string) / 操作（nn.Module或字符串）
    Returns:
        Clean operation name / 清理后的操作名称
    
    Examples / 示例:
    - nn.Conv3d -> "Conv3d"
    - "relu()" -> "relu"
    - "reshape(1,2,3)" -> "reshape"
    - nn.InstanceNorm3d -> "InstanceNorm3d"
    """
    if isinstance(op, nn.Module):
        # Get class name for nn.Module instance / 获取nn.Module实例的类名
        op_name = op.__class__.__name__
    elif isinstance(op, str):
        # Remove parentheses and parameters from string operation / 移除字符串操作中的括号和参数
        op_name = re.sub(r'\(.*\)', '', op).strip()
    else:
        # Convert other types to string directly / 其他类型直接转为字符串
        op_name = str(op)

    # Remove redundant module path prefixes / 移除冗余的模块路径前缀
    op_name = op_name.replace("torch.nn.modules.", "").replace("torch.nn.", "")
    return op_name

def parse_topo_value(value: Union[int, dict, TopoAttr]) -> TopoAttr:
    """
    MODIFIED: 统一解析拓扑值（兼容旧单值/新字典/命名元组）
    Unified topology value parser (compatible with old int/new dict/named tuple)
    
    Args:
        value: 拓扑值（int/dict/TopoAttr）
    Returns:
        标准化的TopoAttr对象 / Standardized TopoAttr object
    """
    if isinstance(value, int):
        # 兼容旧单值格式 / Compatible with old int format
        role = value if value == 0 else -1 if value < 0 else 1
        sort = abs(value) if value != 0 else 0
        return TopoAttr(role=role, sort=sort, ext={})
    elif isinstance(value, dict):
        # 解析新字典格式 / Parse new dict format
        return TopoAttr(
            role=value.get("role", 0),
            sort=value.get("sort", 0),
            ext=value.get("ext", {})
        )
    elif isinstance(value, TopoAttr):
        # 直接返回命名元组 / Directly return named tuple
        return value
    else:
        raise ValueError(f"Unsupported topology value type: {type(value)}")

# ===================== Core Operation Functions / 核心操作函数 =====================
def MHD_concat(tensors: List[torch.Tensor], attrs: List[TopoAttr]) -> torch.Tensor:
    """
    MODIFIED: 按拓扑属性排序后拼接张量（替代原索引排序）
    Concatenate tensors sorted by topology attribute (replace original index sort)
    
    Args:
        tensors: List of tensors to concatenate / 待拼接的张量列表
        attrs: Corresponding topology attributes / 对应的拓扑属性列表
    Returns:
        Concatenated tensor / 拼接后的张量
    """
    sorted_pairs = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs]
    return torch.cat(sorted_tensors, dim=1)

def MHD_matmul(tensors: List[torch.Tensor], attrs: List[TopoAttr]) -> torch.Tensor:
    """
    MODIFIED: 按拓扑属性排序后执行矩阵乘法（替代原索引排序）
    Matrix multiplication with sorted tensors by topology attribute
    
    Args:
        tensors: List of tensors (must be 2 elements) / 张量列表（必须包含2个元素）
        attrs: Corresponding topology attributes / 对应的拓扑属性列表
    Returns:
        Matrix multiplication result / 矩阵乘法结果
    Raises:
        ValueError: If input tensor count is not 2 / 输入张量数量不为2时抛出异常
    """
    sorted_pairs = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs]
    if len(sorted_tensors) != 2:
        raise ValueError(f"Matmul requires exactly 2 input tensors, got {len(sorted_tensors)}")
    return torch.matmul(*sorted_tensors)

# Edge input functions (process head node tensors)
# 边输入函数（处理头节点张量）
MHD_EDGE_IN_FUNCS: Dict[str, Callable[..., Any]] = {
    # MODIFIED: 参数改为拓扑属性列表 / Change parameter to topology attribute list
    "concat": MHD_concat,  # Concatenate head tensors / 拼接头节点张量
    "matmul": MHD_matmul,  # Matrix multiply head tensors / 头节点张量矩阵乘法
}

def MHD_split(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """
    MODIFIED: 按拓扑属性分割张量（替代原索引分割）
    Split tensor and map back to original nodes by topology attribute
    
    Args:
        x: Input tensor to split / 待分割的输入张量
        attrs: Topology attributes for mapping / 用于映射的拓扑属性列表
        node_channels: Channel counts for each target node / 每个目标节点的通道数
    Returns:
        List of tensors mapped to original nodes / 映射到原始节点的张量列表
    """
    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_original_indices = [p[0] for p in sorted_nodes]
    sorted_channel_sizes = [node_channels[i] for i in sorted_original_indices]

    # Split tensor by channel sizes / 按通道数分割张量
    split_tensors = torch.split(x, sorted_channel_sizes, dim=1)
    # Map back to original node order / 映射回原始节点顺序
    tensor_map = {idx: t for idx, t in zip(sorted_original_indices, split_tensors)}
    return [tensor_map[i] for i in range(len(attrs))]

def MHD_svd(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """
    MODIFIED: 按拓扑属性执行SVD分解（替代原索引分解）
    SVD decomposition and map components to nodes by topology attribute
    
    Args:
        x: Input tensor for SVD / 用于SVD的输入张量
        attrs: Topology attributes for mapping / 用于映射的拓扑属性列表
        node_channels: Channel counts for each target node (unused in SVD) / 每个目标节点的通道数（SVD中未使用）
    Returns:
        List of SVD components mapped to original nodes / 映射到原始节点的SVD分量列表
    """
    x_flat = MHD_flatten_tensor(x)
    # SVD decomposition (full_matrices=False for efficiency) / SVD分解（full_matrices=False提升效率）
    U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)

    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    svd_components = [U, S, Vh]
    sorted_tensors = []

    # Map SVD components to nodes (cycle through components if node count > 3)
    # 将SVD分量映射到节点（节点数>3时循环使用分量）
    for i, (orig_idx, _) in enumerate(sorted_nodes):
        comp_idx = i % len(svd_components)
        tensor = svd_components[comp_idx]
        sorted_tensors.append((orig_idx, tensor))

    tensor_map = {idx: t for idx, t in sorted_tensors}
    return [tensor_map[i] for i in range(len(attrs))]

def MHD_lu(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """
    MODIFIED: 按拓扑属性执行LU分解（替代原索引分解）
    LU decomposition and map components to nodes by topology attribute
    
    Args:
        x: Input tensor for LU / 用于LU的输入张量
        attrs: Topology attributes for mapping / 用于映射的拓扑属性列表
        node_channels: Channel counts for each target node (unused in LU) / 每个目标节点的通道数（LU中未使用）
    Returns:
        List of LU components mapped to original nodes / 映射到原始节点的LU分量列表
    """
    x_flat = MHD_flatten_tensor(x)
    # LU decomposition with pivot matrix / 带枢轴矩阵的LU分解
    P, L, U = torch.linalg.lu(x_flat)

    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    lu_components = [L, U, P]
    sorted_tensors = []

    # Map LU components to nodes (cycle through components if node count > 3)
    # 将LU分量映射到节点（节点数>3时循环使用分量）
    for i, (orig_idx, _) in enumerate(sorted_nodes):
        comp_idx = i % len(lu_components)
        tensor = lu_components[comp_idx]
        sorted_tensors.append((orig_idx, tensor))

    tensor_map = {idx: t for idx, t in sorted_tensors}
    return [tensor_map[i] for i in range(len(attrs))]

# Edge output functions (process edge tensor to tail nodes)
# 边输出函数（处理边张量到尾节点）
MHD_EDGE_OUT_FUNCS: Dict[str, Callable[..., Any]] = {
    # MODIFIED: 参数改为拓扑属性列表 / Change parameter to topology attribute list
    "split": MHD_split,  # Split tensor to tail nodes / 分割张量到尾节点
    "svd": MHD_svd,      # SVD decomposition for tail nodes / 尾节点的SVD分解
    "lu": MHD_lu,        # LU decomposition for tail nodes / 尾节点的LU分解
}

# ===================== String Operation Wrapper / 字符串操作包装类 =====================
class StringOperation(nn.Module):
    """
    Wrapper class for tensor operations defined by string
    字符串定义的张量操作包装类
    
    Support format / 支持格式:
    - Simple operation: "relu", "sigmoid"
    - Operation with params: "reshape(1,2,3)", "transpose(0,1)"
    """
    def __init__(self, op_str: str):
        super().__init__()
        self.op_str = op_str  # Original operation string / 原始操作字符串

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute string-defined operation on tensor
        对张量执行字符串定义的操作
        
        Args:
            x: Input tensor / 输入张量
        Returns:
            Tensor after operation / 操作后的张量
        """
        if '(' in self.op_str and ')' in self.op_str:
            # Split method name and parameters / 分割方法名和参数
            method_name, args_str = self.op_str.split('(', 1)
            args_str = args_str.rstrip(')')
            args = []
            kwargs = {}

            # Parse positional and keyword arguments / 解析位置参数和关键字参数
            if args_str:
                for arg in args_str.split(','):
                    arg = arg.strip()
                    if not arg:
                        continue
                    if '=' in arg:
                        # Keyword argument / 关键字参数
                        k, v = arg.split('=', 1)
                        kwargs[k.strip()] = eval(v.strip())
                    else:
                        # Positional argument / 位置参数
                        args.append(eval(arg.strip()))

            # Execute method with parsed args / 执行解析后的方法
            return getattr(x, method_name)(*args, **kwargs)
        else:
            # Simple method call / 简单方法调用
            return getattr(x, self.op_str)()

# ===================== Core Data Structures / 核心数据结构 =====================
@dataclass
class MHD_Node:
    """
    Hypergraph Node Class (no hash conflict for tensor fields)
    超图节点类（张量字段无哈希冲突）
    
    Attributes:
        id: Unique node ID (hashable) / 节点唯一ID（可哈希）
        name: Global unique node name (hashable) / 全局唯一节点名称（可哈希）
        value: Node feature tensor (direct assignment) / 节点特征张量（直接赋值）
        func: Head/tail function config / 头/尾函数配置
            - head: Preprocess function (default: "share") / 预处理函数（默认："share"）
            - tail: Aggregation function (default: "sum") / 聚合函数（默认："sum"）
        requires_grad: Whether to track gradient for node value / 是否追踪节点张量梯度（默认：True）
    """
    id: int
    name: str
    value: torch.Tensor
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"})
    # MODIFIED: 新增梯度配置字段 / Add gradient config field
    requires_grad: bool = field(default=True)

    def __post_init__(self):
        """
        MODIFIED: 初始化时设置张量梯度追踪 / Set gradient tracking on initialization
        """
        self.value.requires_grad_(self.requires_grad)

    def __hash__(self):
        """Hash only on immutable fields (avoid tensor hash conflict)"""
        return hash((self.id, self.name))

    def __eq__(self, other):
        """Compare only immutable fields"""
        if not isinstance(other, MHD_Node):
            return False
        return self.id == other.id and self.name == other.name

@dataclass
class MHD_Edge:
    """
    Hypergraph Edge Class (no hash conflict)
    超图边类（无哈希冲突）
    
    Attributes:
        id: Unique edge ID (hashable) / 边唯一ID（可哈希）
        name: Unique edge name in subnetwork (hashable) / 子网内唯一边名称（可哈希）
        operations: List of operations (nn.Module/string) / 操作列表（nn.Module/字符串）
        func: Input/output function config / 输入/输出函数配置
            - in: Head tensor process function (default: "concat") / 头张量处理函数（默认："concat"）
            - out: Tail tensor process function (default: "split") / 尾张量处理函数（默认："split"）
    """
    id: int
    name: str
    operations: List[Union[str, nn.Module]]
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"})

    def __hash__(self):
        """Hash only on immutable fields"""
        return hash((self.id, self.name))

    def __eq__(self, other):
        """Compare only immutable fields"""
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id and self.name == other.name

@dataclass
class MHD_Topo:
    """
    MODIFIED: 结构化拓扑类（替代原单值矩阵）
    Hypergraph Topology Class (structured version)
    
    Attributes:
        value: Topology matrix (edge x node) with TopoAttr values
               拓扑矩阵（边 x 节点），元素为TopoAttr命名元组
            - role: -1 = Head node, 1 = Tail node, 0 = No connection
                    角色：-1=头节点，1=尾节点，0=无连接
            - sort: Sort priority (1,2,3...) / 排序优先级
            - ext: Extended attributes (reserved for time/branch/attention)
                   扩展属性（预留时序/分支/注意力等）
    """
    value: Union[torch.Tensor, List[List[Union[TopoAttr, dict, int]]]]

    def __post_init__(self):
        """
        MODIFIED: 初始化时统一解析拓扑矩阵为TopoAttr格式
        Parse topology matrix to TopoAttr format on initialization
        """
        if isinstance(self.value, torch.Tensor):
            # 从张量转换 / Convert from tensor
            self.value = [
                [parse_topo_value(int(val)) for val in row]
                for row in self.value.cpu().tolist()
            ]
        else:
            # 从列表/字典转换 / Convert from list/dict
            self.value = [
                [parse_topo_value(val) for val in row]
                for row in self.value
            ]

    def get_topo_attr(self, edge_id: int, node_id: int) -> TopoAttr:
        """获取指定边-节点的拓扑属性 / Get topology attribute for specific edge-node"""
        return self.value[edge_id][node_id]

    def to_tensor(self) -> torch.Tensor:
        """
        转换为兼容旧逻辑的张量（仅保留role字段）
        Convert to compatible tensor (only keep role field)
        """
        tensor_data = []
        for row in self.value:
            tensor_row = [attr.role for attr in row]
            tensor_data.append(tensor_row)
        return torch.tensor(tensor_data, dtype=torch.int64)

# ===================== Topological Sort / 拓扑排序 =====================
def MHD_topological_sort(nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo) -> List[int]:
    """
    MODIFIED: 适配结构化拓扑矩阵的拓扑排序
    Topological sort for hypergraph nodes with structured topology matrix
    
    Args:
        nodes: Set of hypergraph nodes / 超图节点集合
        edges: Set of hypergraph edges / 超图边集合
        topo: Hypergraph topology matrix / 超图拓扑矩阵
    Returns:
        List of node IDs in topological order / 拓扑顺序的节点ID列表
    Raises:
        ValueError: If cycle detected in hypergraph / 超图中检测到环时抛出异常
    """
    # Build dependency graph / 构建依赖图
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_node_ids = {node.id for node in nodes}

    if len(topo.value) > 0 and len(topo.value[0]) > 0:
        # 遍历所有边-节点连接 / Iterate all edge-node connections
        for edge_id, edge_row in enumerate(topo.value):
            for node_id, attr in enumerate(edge_row):
                if attr.role == 0:
                    continue  # 无连接 / No connection
                # 分类头/尾节点 / Classify head/tail nodes
                if attr.role < 0:
                    head_id = node_id
                    # 找到该边的所有尾节点 / Find all tail nodes of this edge
                    tail_ids = [
                        nid for nid, a in enumerate(edge_row) 
                        if a.role > 0
                    ]
                    # 构建依赖关系（头→尾） / Build dependency (head → tail)
                    for tail_id in tail_ids:
                        graph[head_id].append(tail_id)
                        in_degree[tail_id] += 1

    # Initialize in_degree for isolated nodes / 初始化孤立节点的入度
    for node_id in all_node_ids:
        if node_id not in in_degree:
            in_degree[node_id] = 0

    # Kahn's algorithm for topological sort / Kahn算法实现拓扑排序
    queue = deque([node_id for node_id in all_node_ids if in_degree[node_id] == 0])
    sorted_node_ids = []

    while queue:
        current_node = queue.popleft()
        sorted_node_ids.append(current_node)
        for neighbor in graph[current_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles / 检测环
    if len(sorted_node_ids) != len(all_node_ids):
        raise ValueError(f"Cycle detected in hypergraph! Nodes in cycle: {all_node_ids - set(sorted_node_ids)}")
    return sorted_node_ids

# ===================== DNet (Dynamic Network) / 动态网络 =====================
class DNet(nn.Module):
    """
    Dynamic Network for Hyperedge Operations
    超边操作动态网络（支持动态操作列表）
    
    Attributes:
        filter: Sequential operations / 顺序操作序列
        op_names: Clean operation names (for visualization) / 清理后的操作名称（用于可视化）
        original_operations: Original operation list (reference only) / 原始操作列表（仅参考）
    """
    def __init__(self, operations: List[Union[str, nn.Module]]):
        super().__init__()
        # Build sequential operation list / 构建顺序操作列表
        seq_ops = []
        self.op_names = []  # For topology visualization / 用于拓扑可视化

        for op in operations:
            # Extract clean operation name / 提取干净的操作名称
            self.op_names.append(extract_operation_name(op))

            if isinstance(op, nn.Module):
                # Directly add nn.Module / 直接添加nn.Module
                seq_ops.append(op)
            elif isinstance(op, str):
                # Wrap string operation / 包装字符串操作
                seq_ops.append(StringOperation(op))
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}. Only nn.Module/string allowed.")

        # Build sequential module / 构建顺序模块
        self.filter = nn.Sequential(*seq_ops)
        self.original_operations = operations  # Keep original for reference / 保留原始操作列表供参考

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dynamic operations
        动态操作前向传播
        
        Args:
            x: Input tensor / 输入张量
        Returns:
            Tensor after all operations / 所有操作后的张量
        """
        return self.filter(x)

# ===================== HDNet (Hypergraph Dynamic Network) / 超图动态网络 =====================
class HDNet(nn.Module):
    """
    Hypergraph Dynamic Network (HDNet)
    超图动态网络（单超图子网）
    
    Attributes:
        node_id2obj: Node ID to MHD_Node mapping / 节点ID到MHD_Node的映射
        edge_id2obj: Edge ID to MHD_Edge mapping / 边ID到MHD_Edge的映射
        topo: Hypergraph topology / 超图拓扑
        name: Subnetwork name / 子网名称
        sorted_node_ids: Topologically sorted node IDs / 拓扑排序后的节点ID
        edge_nets: Edge operation networks (ModuleDict) / 边操作网络（ModuleDict）
        node_values: Node feature tensors / 节点特征张量
    """
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo, name: str = "hdnet"):
        super().__init__()
        # Build node/edge mapping for fast access / 构建节点/边映射以快速访问
        self.node_id2obj = {node.id: node for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        self.topo = topo
        self.name = name

        # Validate topology matrix / 验证拓扑矩阵
        self._validate_topo()
        # Topological sort for forward pass / 前向传播的拓扑排序
        self.sorted_node_ids = MHD_topological_sort(nodes, edges, topo)
        print(f"✅ {self.name} DAG sorted: {[self.node_id2obj[nid].name for nid in self.sorted_node_ids]}")

        # Build edge operation networks / 构建边操作网络
        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.operations)

        # Initialize node values (direct assignment) / 初始化节点值（直接赋值）
        self.node_values = {node.id: node.value for node in nodes}

    @property
    def node_name2id(self):
        """
        Node name to ID mapping (dynamic property)
        节点名称到ID的映射（动态属性）
        """
        return {v.name: k for k, v in self.node_id2obj.items()}

    def _validate_topo(self) -> None:
        """
        MODIFIED: 验证结构化拓扑矩阵
        Validate structured topology matrix
        
        Raises:
            ValueError: If dimension/type mismatch / 维度/类型不匹配时抛出异常
        """
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)

        # Check dimension / 检查维度
        if len(self.topo.value) != num_edges:
            raise ValueError(
                f"Topology matrix edge dimension mismatch: expected {num_edges}, "
                f"got {len(self.topo.value)}"
            )
        for edge_row in self.topo.value:
            if len(edge_row) != num_nodes:
                raise ValueError(
                    f"Topology matrix node dimension mismatch: expected {num_nodes}, "
                    f"got {len(edge_row)}"
                )

        # Check TopoAttr type / 检查TopoAttr类型
        for edge_id, edge_row in enumerate(self.topo.value):
            for node_id, attr in enumerate(edge_row):
                if not isinstance(attr, TopoAttr):
                    raise ValueError(
                        f"Topology matrix element at ({edge_id}, {node_id}) must be TopoAttr, "
                        f"got {type(attr)}"
                    )

    def forward(self) -> Dict[str, Tensor]:
        """
        MODIFIED: 适配结构化拓扑矩阵的前向传播
        Topology-driven forward pass with structured topology matrix
        
        Returns:
            Dict of {node_name: feature_tensor} / {节点名称: 特征张量}字典
        """
        # Get device from node values / 从节点值获取设备
        device = next(iter(self.node_values.values())).device

        # Map edges to affected tail nodes / 映射边到受影响的尾节点
        edge_affects_nodes = defaultdict(list)
        if len(self.topo.value) > 0:
            for edge_id in self.edge_id2obj.keys():
                edge_row = self.topo.value[edge_id]
                tail_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role > 0]
                edge_affects_nodes[edge_id] = tail_node_ids

        # Process nodes in topological order / 按拓扑顺序处理节点
        for target_node_id in self.sorted_node_ids:
            # Get relevant edges (affect current node) / 获取相关边（影响当前节点的边）
            relevant_edges = [eid for eid, node_ids in edge_affects_nodes.items() if target_node_id in node_ids]

            for edge_id in relevant_edges:
                edge = self.edge_id2obj[edge_id]
                edge_net = self.edge_nets[edge.name]
                edge_row = self.topo.value[edge_id]

                # Get head nodes (role < 0 in topology) / 获取头节点（拓扑中role<0）
                head_mask = [attr.role < 0 for attr in edge_row]
                head_node_ids = [i for i, val in enumerate(head_mask) if val]
                # 获取头节点的拓扑属性 / Get topology attributes of head nodes
                head_topo_attrs = [edge_row[nid] for nid in head_node_ids]

                # Process head node tensors / 处理头节点张量
                head_tensors = []
                for node_id in head_node_ids:
                    node = self.node_id2obj[node_id]
                    head_func_name = node.func.get("head", "share")
                    # Apply head function / 应用头函数
                    head_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](self.node_values[node_id])
                    head_tensors.append(head_tensor)

                # Edge input processing / 边输入处理
                edge_in_func_name = edge.func.get("in", "concat")
                edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](head_tensors, head_topo_attrs)

                # Edge operation forward / 边操作前向传播
                edge_output = edge_net(edge_input)

                # Get tail nodes (role > 0 in topology) / 获取尾节点（拓扑中role>0）
                tail_mask = [attr.role > 0 for attr in edge_row]
                tail_node_ids = [i for i, val in enumerate(tail_mask) if val]
                # 获取尾节点的拓扑属性 / Get topology attributes of tail nodes
                tail_topo_attrs = [edge_row[nid] for nid in tail_node_ids]
                # Get tail node channel sizes / 获取尾节点通道数
                tail_node_channels = [self.node_id2obj[node_id].value.shape[1] for node_id in tail_node_ids]

                # Edge output processing / 边输出处理
                edge_out_func_name = edge.func.get("out", "split")
                tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                    edge_output, tail_topo_attrs, tail_node_channels
                )

                # MODIFIED: 用in-place操作更新节点值（保留梯度）
                # Update tail node values with in-place operation (preserve gradient)
                for node_id, tensor in zip(tail_node_ids, tail_tensors):
                    node = self.node_id2obj[node_id]
                    tail_func_name = node.func.get("tail", "sum")
                    if node_id in self.node_values:
                        # Aggregate with existing value / 与现有值聚合
                        agg_tensor = MHD_NODE_TAIL_FUNCS[tail_func_name](
                            [self.node_values[node_id], tensor]
                        )
                        # 使用copy_()进行in-place赋值，保留梯度 / In-place assignment with copy_() to preserve gradient
                        self.node_values[node_id].copy_(agg_tensor)
                    else:
                        # Initialize node value / 初始化节点值
                        self.node_values[node_id] = tensor.requires_grad_(node.requires_grad)

        # Return node features (name -> tensor) / 返回节点特征（名称->张量）
        return {
            self.node_id2obj[node_id].name: tensor
            for node_id, tensor in self.node_values.items()
        }

# ===================== MHDNet (Multi-Hypergraph Dynamic Network) / 多超图动态网络 =====================
class MHDNet(nn.Module):
    """
    Multi-Hypergraph Dynamic Network (MHDNet)
    多超图动态网络（合并多个HDNet子网为全局超图）
    
    Attributes:
        sub_hdnets: Dictionary of sub-HDNet / 子HDNet字典
        node_mapping: Global-local node mapping / 全局-局部节点映射
        global_hdnet: Merged global hypergraph network / 合并后的全局超图网络
        mermaid_code: Mermaid topology graph code / Mermaid拓扑图代码
    """
    def __init__(
        self,
        sub_hdnets: Dict[str, HDNet],
        node_mapping: List[Tuple[str, str, str]],  # (global_name, sub_name, sub_node_name)
    ):
        super().__init__()
        self.sub_hdnets = sub_hdnets
        self.node_mapping = node_mapping

        # Build global hypergraph from subgraphs / 从子图构建全局超图
        self.global_hdnet = self._build_global_hypergraph()

        # Generate topology visualization code / 生成拓扑可视化代码
        self.mermaid_code = self.generate_expanded_topological_mermaid()
        print("\n" + "="*80)
        print("✅ EXPANDED TOPOLOGICAL MERMAID CODE (WITH OPERATION NAMES):")
        print("="*80)
        print(self.mermaid_code)
        print("="*80 + "\n")

    def _build_global_hypergraph(self) -> HDNet:
        """
        MODIFIED: 适配结构化拓扑矩阵的全局超图构建
        Build global hypergraph by merging sub-HDNets with structured topology
        
        Returns:
            Global HDNet instance / 全局HDNet实例
        """
        # ID counters for global nodes/edges / 全局节点/边的ID计数器
        node_id_counter = 0
        edge_id_counter = 0

        # Pre-cache subgraph nodes (avoid redundant lookup) / 预缓存子图节点（避免冗余查找）
        sub_node_cache = {}
        for sub_name, sub_hdnet in self.sub_hdnets.items():
            for sub_node_id, sub_node in sub_hdnet.node_id2obj.items():
                sub_node_cache[(sub_name, sub_node.name)] = (sub_node_id, sub_node)

        # Mapping from subgraph to global nodes/edges / 子图到全局节点/边的映射
        sub2global_node = {}
        sub2global_edge = {}

        # Build global nodes / 构建全局节点
        global_nodes = set()
        for global_node_name, sub_name, sub_node_name in self.node_mapping:
            key = (sub_name, sub_node_name)
            if key not in sub2global_node:
                # Get cached subgraph node / 获取缓存的子图节点
                sub_node_id, sub_node = sub_node_cache[key]

                # Create global node (direct tensor assignment) / 创建全局节点（直接张量赋值）
                global_node = MHD_Node(
                    id=node_id_counter,
                    name=global_node_name,
                    value=sub_node.value,
                    func=sub_node.func,
                    requires_grad=sub_node.requires_grad  # 保留梯度配置 / Keep gradient config
                )
                global_nodes.add(global_node)
                sub2global_node[key] = (node_id_counter, global_node_name)
                node_id_counter += 1

        # Build global edges / 构建全局边
        global_edges = set()
        global_topo_data = []

        for sub_name, sub_hdnet in self.sub_hdnets.items():
            # Process each edge in subgraph / 处理子图中的每条边
            for sub_edge_id in sorted(sub_hdnet.edge_id2obj.keys()):
                sub_edge = sub_hdnet.edge_id2obj[sub_edge_id]
                sub_edge_name = sub_edge.name
                key = (sub_name, sub_edge_name)

                if key not in sub2global_edge:
                    # Map sub edge to global edge ID / 子图边映射到全局边ID
                    sub2global_edge[key] = edge_id_counter

                    # Create global edge / 创建全局边
                    global_edge_name = f"{sub_name}_{sub_edge_name}"
                    global_edge = MHD_Edge(
                        id=edge_id_counter,
                        name=global_edge_name,
                        operations=sub_edge.operations,
                        func=sub_edge.func
                    )
                    global_edges.add(global_edge)

                    # Build global topology row / 构建全局拓扑行
                    sub_topo_row = sub_hdnet.topo.value[sub_edge_id]
                    global_topo_row = [TopoAttr(role=0, sort=0, ext={}) for _ in range(node_id_counter)]

                    # Map sub topology to global topology / 子图拓扑映射到全局拓扑
                    for sub_node_id, sub_attr in enumerate(sub_topo_row):
                        if sub_attr.role == 0:
                            continue
                        sub_node_name = sub_hdnet.node_id2obj[sub_node_id].name
                        map_key = (sub_name, sub_node_name)
                        if map_key in sub2global_node:
                            global_node_id, _ = sub2global_node[map_key]
                            # 保留原拓扑属性 / Keep original topology attribute
                            global_topo_row[global_node_id] = sub_attr

                    global_topo_data.append(global_topo_row)
                    edge_id_counter += 1

        # Create global topology matrix / 创建全局拓扑矩阵
        global_topo = MHD_Topo(value=global_topo_data)

        # Create global HDNet / 创建全局HDNet
        global_hdnet = HDNet(nodes=global_nodes, edges=global_edges, topo=global_topo, name="global_hdnet")

        return global_hdnet

    def generate_expanded_topological_mermaid(self) -> str:
        """
        Generate Mermaid code for expanded topology visualization (with operation names)
        生成展开的拓扑可视化Mermaid代码（包含操作名称）
        
        Returns:
            Mermaid graph code / Mermaid图代码
        """
        # Build subgraph to global node mapping / 构建子图到全局节点的映射
        sub2global = {}
        for global_name, sub_name, sub_node_name in self.node_mapping:
            sub2global[(sub_name, sub_node_name)] = global_name

        # Initialize Mermaid code / 初始化Mermaid代码
        mermaid = [
            "graph TD",
            "",
            "    %% ===== GLOBAL TOPOLOGY WITH EXPANDED OPERATIONS ===== / 展开操作的全局拓扑结构",
            "",
        ]

        # Generate topology for each subgraph / 为每个子图生成拓扑
        for sub_name, sub_hdnet in self.sub_hdnets.items():
            mermaid.append(f"    %% {sub_name} topology (expanded operations) / {sub_name} 拓扑结构（展开操作）")

            # Process each edge in subgraph / 处理子图中的每条边
            for edge_id, edge in sub_hdnet.edge_id2obj.items():
                edge_name = edge.name
                base_edge_name = f"{sub_name}_{edge_name}"

                # Get head/tail node IDs from topology / 从拓扑获取头/尾节点ID
                edge_row = sub_hdnet.topo.value[edge_id]
                head_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role < 0]
                tail_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role > 0]

                # Convert to global node names / 转换为全局节点名称
                head_global_nodes = [sub2global[(sub_name, sub_hdnet.node_id2obj[nid].name)] for nid in head_node_ids]
                tail_global_nodes = [sub2global[(sub_name, sub_hdnet.node_id2obj[nid].name)] for nid in tail_node_ids]

                # Get operation names from edge network / 从边网络获取操作名称
                edge_net = sub_hdnet.edge_nets[edge_name]
                op_names = edge_net.op_names

                # Generate operation nodes / 生成操作节点
                if op_names:
                    op_node_names = [f"{base_edge_name}_{idx}_{op_name}" for idx, op_name in enumerate(op_names)]

                    # Connect head nodes to first operation / 头节点连接到第一个操作
                    for head_node in head_global_nodes:
                        mermaid.append(f"    {head_node} --> {op_node_names[0]}")

                    # Connect sequential operations / 连接顺序操作
                    for i in range(len(op_node_names)-1):
                        mermaid.append(f"    {op_node_names[i]} --> {op_node_names[i+1]}")

                    # Connect last operation to tail nodes / 最后一个操作连接到尾节点
                    last_op_node = op_node_names[-1]
                    for tail_node in tail_global_nodes:
                        mermaid.append(f"    {last_op_node} --> {tail_node}")
                else:
                    # No operations - direct connection / 无操作 - 直接连接
                    hyper_edge_node = f"{base_edge_name}_empty"
                    mermaid.append(f"    {hyper_edge_node}")
                    for head_node in head_global_nodes:
                        mermaid.append(f"    {head_node} --> {hyper_edge_node}")
                    for tail_node in tail_global_nodes:
                        mermaid.append(f"    {hyper_edge_node} --> {tail_node}")

            mermaid.append("")

        return "\n".join(mermaid)

    def forward(self) -> Dict[str, Tensor]:
        """
        Forward pass of global hypergraph network
        全局超图网络的前向传播
        
        Returns:
            Global node features (name -> tensor) / 全局节点特征（名称->张量）
        """
        all_node_features = self.global_hdnet.forward()
        return all_node_features

# ===================== Example Usage / 示例用法 =====================
def example_mhdnet2():
    """
    MODIFIED: 适配结构化拓扑矩阵的示例
    Example of MHDNet usage with structured topology matrix
    
    Key features / 核心特性:
    - Direct tensor assignment (no redundant conversion) / 直接张量赋值（无冗余转换）
    - No hash conflicts for tensor fields / 张量字段无哈希冲突
    - Topology-driven forward propagation / 拓扑驱动的前向传播
    - Gradient-preserving tensor operations / 保留梯度的张量操作
    - Structured topology matrix (extendable) / 结构化拓扑矩阵（可扩展）
    """
    # Device configuration / 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"✅ Using device: {device} / 使用设备: {device}")

    # --------------------------
    # 1. Create global source nodes / 创建全局源节点
    # --------------------------
    # Input node (200) - 3D tensor (batch, channel, depth, height, width)
    # 输入节点(200) - 3D张量（批次，通道，深度，高度，宽度）
    g_200 = MHD_Node(
        id=0, 
        name="200", 
        value=torch.randn(1, 2, 16, 16, 16, device=device, dtype=dtype),
        func={"head": "share", "tail": "sum"},
        requires_grad=True  # 显式开启梯度 / Explicitly enable gradient
    )

    # Other global source nodes / 其他全局源节点
    g_201 = MHD_Node(id=1, name="201", value=torch.randn(1, 1, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_202 = MHD_Node(id=2, name="202", value=torch.randn(1, 2, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_203 = MHD_Node(id=3, name="203", value=torch.randn(1, 1, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_204 = MHD_Node(id=4, name="204", value=torch.randn(1, 2, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_205 = MHD_Node(id=5, name="205", value=torch.randn(1, 1, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)

    # --------------------------
    # 2. Create sub-HDNets / 创建子HDNet
    # --------------------------
    # ========== NET1 ==========
    nodes_net1 = {
        MHD_Node(id=0, name="A1", value=g_200.value, requires_grad=True),  # Map to global 200 / 映射到全局200
        MHD_Node(id=1, name="B1", value=g_201.value, requires_grad=True),  # Map to global 201 / 映射到全局201
    }
    # Edge operations (Conv3d -> InstanceNorm3d -> ReLU -> 1x1 Conv)
    # 边操作（3D卷积 -> 3D实例归一化 -> ReLU -> 1x1卷积）
    conv_net1 = nn.Conv3d(2, 8, kernel_size=3, padding=1, bias=False).to(device)
    conv_net1.weight.requires_grad = True
    norm_net1 = nn.InstanceNorm3d(8, affine=False).to(device)
    act_net1 = nn.ReLU(inplace=True)
    adjust_net1 = nn.Conv3d(8, 1, kernel_size=1, bias=True).to(device)

    edges_net1 = {
        MHD_Edge(
            id=0, 
            name="e1", 
            operations=[conv_net1, norm_net1, act_net1, adjust_net1],
            func={"in": "concat", "out": "split"}
        )
    }
    # MODIFIED: 结构化拓扑矩阵（替代原单值矩阵）
    # Topology matrix: [TopoAttr(role=-1, sort=1), TopoAttr(role=1, sort=1)]
    topo_net1 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]  # A1=head, B1=tail
    ])
    hdnet1 = HDNet(nodes=nodes_net1, edges=edges_net1, topo=topo_net1, name="net1")

    # ========== NET2 ==========
    nodes_net2 = {
        MHD_Node(id=0, name="A2", value=g_202.value, requires_grad=True),  # Map to global 202 / 映射到全局202
        MHD_Node(id=1, name="B2", value=g_203.value, requires_grad=True),  # Map to global 203 / 映射到全局203
    }
    conv_net2 = nn.Conv3d(2, 8, kernel_size=3, padding=1, bias=False).to(device)
    conv_net2.weight.requires_grad = True
    norm_net2 = nn.InstanceNorm3d(8, affine=False).to(device)
    act_net2 = nn.ReLU(inplace=True)
    adjust_net2 = nn.Conv3d(8, 1, kernel_size=1, bias=True).to(device)

    edges_net2 = {
        MHD_Edge(
            id=0, 
            name="e1", 
            operations=[conv_net2, norm_net2, act_net2, adjust_net2],
            func={"in": "concat", "out": "split"}
        )
    }
    # MODIFIED: 结构化拓扑矩阵
    topo_net2 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]
    ])
    hdnet2 = HDNet(nodes=nodes_net2, edges=edges_net2, topo=topo_net2, name="net2")

    # ========== NET3 ==========
    nodes_net3 = {
        MHD_Node(id=0, name="A3", value=g_204.value, requires_grad=True),  # Map to global 204 / 映射到全局204
        MHD_Node(id=1, name="B3", value=g_205.value, requires_grad=True),  # Map to global 205 / 映射到全局205
    }
    conv_net3 = nn.Conv3d(2, 8, kernel_size=3, padding=1, bias=False).to(device)
    conv_net3.weight.requires_grad = True
    norm_net3 = nn.InstanceNorm3d(8, affine=False).to(device)
    act_net3 = nn.ReLU(inplace=True)
    adjust_net3 = nn.Conv3d(8, 1, kernel_size=1, bias=True).to(device)

    edges_net3 = {
        MHD_Edge(
            id=0, 
            name="e1", 
            operations=[conv_net3, norm_net3, act_net3, adjust_net3],
            func={"in": "concat", "out": "split"}
        )
    }
    # MODIFIED: 结构化拓扑矩阵
    topo_net3 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]
    ])
    hdnet3 = HDNet(nodes=nodes_net3, edges=edges_net3, topo=topo_net3, name="net3")

    # ========== NET4 ==========
    nodes_net4 = {
        MHD_Node(id=0, name="A4", value=g_200.value, requires_grad=True),  # 200
        MHD_Node(id=1, name="B4", value=g_201.value, requires_grad=True),  # 201
        MHD_Node(id=2, name="C4", value=g_202.value, requires_grad=True),  # 202
        MHD_Node(id=3, name="D4", value=g_203.value, requires_grad=True),  # 203
        MHD_Node(id=4, name="E4", value=g_204.value, requires_grad=True),  # 204
        MHD_Node(id=5, name="F4", value=g_205.value, requires_grad=True),  # 205
    }
    conv_net4 = nn.Conv3d(3, 6, kernel_size=1, padding=0, bias=False).to(device)
    conv_net4.weight.requires_grad = True
    norm_net4 = nn.InstanceNorm3d(6, affine=False).to(device)
    act_net4 = nn.ReLU(inplace=True)

    edges_net4 = {
        MHD_Edge(
            id=0, 
            name="e1", 
            operations=[conv_net4, norm_net4, act_net4],
            func={"in": "concat", "out": "split"}
        )
    }
    # MODIFIED: 结构化拓扑矩阵
    # Topology: A4/B4=head (role=-1), C4/D4/E4/F4=tail (role=1)
    topo_net4 = MHD_Topo(value=[
        [
            {"role": -1, "sort": 1}, {"role": -1, "sort": 2},
            {"role": 1, "sort": 1}, {"role": 1, "sort": 2},
            {"role": 1, "sort": 3}, {"role": 1, "sort": 4}
        ]
    ])
    hdnet4 = HDNet(nodes=nodes_net4, edges=edges_net4, topo=topo_net4, name="net4")

    # ========== NET5 ==========
    nodes_net5 = {
        MHD_Node(id=0, name="A5", value=g_200.value, requires_grad=True),  # 200
        MHD_Node(id=1, name="B5", value=g_201.value, requires_grad=True),  # 201
        MHD_Node(id=2, name="C5", value=g_202.value, requires_grad=True),  # 202
        MHD_Node(id=3, name="D5", value=g_203.value, requires_grad=True),  # 203
        MHD_Node(id=4, name="E5", value=g_204.value, requires_grad=True),  # 204
        MHD_Node(id=5, name="F5", value=g_205.value, requires_grad=True),  # 205
    }
    conv_net5 = nn.Conv3d(6, 3, kernel_size=1, padding=0, bias=False).to(device)
    conv_net5.weight.requires_grad = True
    norm_net5 = nn.InstanceNorm3d(3, affine=False).to(device)
    act_net5 = nn.ReLU(inplace=True)

    edges_net5 = {
        MHD_Edge(
            id=0, 
            name="e1", 
            operations=[conv_net5, norm_net5, act_net5, '__mul__(-1)'],  # Multiply by -1 / 乘以-1
            func={"in": "concat", "out": "split"}
        )
    }
    # MODIFIED: 结构化拓扑矩阵
    # Topology: A5/B5/C5/D5=head, E5/F5=tail
    topo_net5 = MHD_Topo(value=[
        [
            {"role": -1, "sort": 1}, {"role": -1, "sort": 2},
            {"role": -1, "sort": 3}, {"role": -1, "sort": 4},
            {"role": 1, "sort": 1}, {"role": 1, "sort": 2}
        ]
    ])
    hdnet5 = HDNet(nodes=nodes_net5, edges=edges_net5, topo=topo_net5, name="net5")

    # --------------------------
    # 3. Global-local node mapping / 全局-局部节点映射
    #    Format: (global_name, sub_name, sub_node_name)
    # --------------------------
    node_mapping = [
        ("200", "net1", "A1"), ("200", "net4", "A4"), ("200", "net5", "A5"),
        ("201", "net1", "B1"), ("201", "net4", "B4"), ("201", "net5", "B5"),
        ("202", "net2", "A2"), ("202", "net4", "C4"), ("202", "net5", "C5"),
        ("203", "net2", "B2"), ("203", "net4", "D4"), ("203", "net5", "D5"),
        ("204", "net3", "A3"), ("204", "net4", "E4"), ("204", "net5", "E5"),
        ("205", "net3", "B3"), ("205", "net4", "F4"), ("205", "net5", "F5"),
    ]

    # --------------------------
    # 4. Build MHDNet / 构建MHDNet
    # --------------------------
    model = MHDNet(
        sub_hdnets={"net1": hdnet1, "net2": hdnet2, "net3": hdnet3, "net4": hdnet4, "net5": hdnet5},
        node_mapping=node_mapping
    )

    # --------------------------
    # 5. Forward pass / 前向传播
    # --------------------------
    all_features = model.forward()

    # --------------------------
    # 6. Print results / 打印结果
    # --------------------------
    print("\n✅ Pure topology-driven forward completed! / 纯拓扑驱动前向传播完成！")
    print(f"Input node (200) direct shape / 输入节点(200)直接形状: {g_200.value.shape}")
    print("\nAll global node feature maps (direct tensor access) / 所有全局节点特征图（直接张量访问）:")
    for node_name, tensor in sorted(all_features.items()):
        print(f"  - Node {node_name}: shape={tensor.shape}, device={tensor.device}, requires_grad={tensor.requires_grad}")

    # Statistics / 统计信息
    print(f"\nTotal global nodes: {len(all_features)} / 全局节点总数: {len(all_features)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Model Params / 模型总参数: {total_params}")
    
    return model

# ===================== Main Execution / 主执行函数 =====================
if __name__ == "__main__":
    # Run MHDNet example / 运行MHDNet示例
    model = example_mhdnet2()
