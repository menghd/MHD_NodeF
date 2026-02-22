"""
MHDNet (Multi-Hypergraph Dynamic Network) - 多超图动态网络核心实现
====================================================================
功能说明 / Function Description:
1. 实现超图的核心数据结构（节点MHD_Node、边MHD_Edge、拓扑MHD_Topo）；
2. 支持按拓扑顺序的节点特征聚合（head→edge）和分发（edge→tail）；
3. 提供多超图整合能力，支持全局节点跨子超图的特征聚合与分配；
4. 兼容PyTorch模块和字符串形式的张量操作，支持ONNX导出。

核心设计原则 / Core Design Principles:
- 节点分为head（源）和tail（目标），拓扑值负数表示head、正数表示tail，绝对值为顺序；
- 全局节点在不同子超图中保持形状一致，支持跨子超图的特征聚合；
- 函数命名规范为mhd_xxx，保证代码风格统一。

版本兼容 / Version Compatibility:
- PyTorch >= 1.8.0 (推荐2.0+)
- ONNX opset 11/17（自动适配PyTorch版本）
====================================================================
Author: AI Assistant
Date: 2026
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, TypeVar
from dataclasses import dataclass, field

# 移除所有警告 / Remove all warnings
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================== 类型定义 / Type Definition =====================
Tensor = TypeVar('Tensor', bound=torch.Tensor)
FuncMapping = Dict[str, Callable[..., Any]]

# ===================== 全局函数注册表 / Global Function Registry =====================
# 头节点函数（单节点→多超边）/ Head Node Functions (single node → multiple hyperedges)
MHD_NODE_HEAD_FUNCS: FuncMapping = {
    "share": lambda tensor: tensor.clone(),  # 特征共享到多条超边 / Share features to multiple hyperedges
}

# 尾节点函数（多超边→单节点）/ Tail Node Functions (multiple hyperedges → single node)
MHD_NODE_TAIL_FUNCS: FuncMapping = {
    "sum": lambda tensors: sum(tensors),  # 逐元素求和 / Element-wise sum
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),  # 逐元素均值 / Element-wise average
    "mul": lambda tensors: torch.stack(tensors).prod(dim=0),  # 逐元素相乘 / Element-wise multiplication
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],  # 逐元素最大值 / Element-wise maximum
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],  # 逐元素最小值 / Element-wise minimum
}

# -------------------- 超边输入函数（多头节点→超边单输入） / Edge Input Functions --------------------
def mhd_concat(tensors: List[Tensor], indices: List[int]) -> Tensor:
    """
    按拓扑顺序拼接多头节点张量（多→单）
    Concatenate multiple head node tensors by topology order (multiple → single)
    
    核心逻辑 / Core Logic:
    1. 按拓扑值绝对值排序head节点 / Sort head nodes by absolute value of topology indices
    2. 按排序后的顺序拼接张量 / Concatenate tensors in sorted order
    3. 输出单一张量 / Output single tensor
    
    参数 / Args:
        tensors: head节点特征张量列表 / List of head node feature tensors
        indices: 拓扑值列表（head节点为负数） / List of topology values (negative for head nodes)
    返回 / Returns:
        拼接后的单一张量 / Concatenated single tensor
    """
    # Step 1: 配对拓扑值和张量 / Pair topology indices with tensors
    indexed_tensors = list(zip(indices, tensors))
    
    # Step 2: 按拓扑值绝对值从小到大排序 / Sort by absolute value of topology indices
    sorted_pairs = sorted(indexed_tensors, key=lambda x: abs(x[0]))
    
    # Step 3: 提取排序后的张量 / Extract sorted tensors
    sorted_tensors = [t for _, t in sorted_pairs]
    
    # Step 4: 按通道维度拼接 / Concatenate along channel dimension
    concatenated = torch.cat(sorted_tensors, dim=1)
    
    return concatenated

def mhd_matmul(tensors: List[Tensor], indices: List[int]) -> Tensor:
    """
    按拓扑顺序执行矩阵乘法（多头节点→超边单输入）
    Matrix multiplication by topology order (multiple head nodes → single edge input)
    
    核心逻辑 / Core Logic:
    1. 按拓扑值绝对值排序head节点 / Sort head nodes by absolute value of topology indices
    2. 按排序后的顺序执行矩阵乘法 / Perform matrix multiplication in sorted order
    3. 输出单一张量 / Output single tensor
    
    参数 / Args:
        tensors: head节点特征张量列表 / List of head node feature tensors
        indices: 拓扑值列表（head节点为负数） / List of topology values (negative for head nodes)
    返回 / Returns:
        矩阵乘法结果张量 / Matrix multiplication result tensor
    """
    # Step 1: 配对拓扑值和张量 / Pair topology indices with tensors
    indexed_tensors = list(zip(indices, tensors))
    
    # Step 2: 按拓扑值绝对值排序 / Sort by absolute value of topology indices
    sorted_pairs = sorted(indexed_tensors, key=lambda x: abs(x[0]))
    
    # Step 3: 提取排序后的张量 / Extract sorted tensors
    sorted_tensors = [t for _, t in sorted_pairs]
    
    # Step 4: 执行矩阵乘法（仅支持2个张量） / Perform matrix multiplication (only for 2 tensors)
    if len(sorted_tensors) != 2:
        raise ValueError(f"矩阵乘法仅支持2个输入张量 / Matmul requires exactly 2 input tensors, got {len(sorted_tensors)}")
    
    matmul_result = torch.matmul(*sorted_tensors)
    
    return matmul_result

# 注册输入函数 / Register input functions
MHD_EDGE_IN_FUNCS: FuncMapping = {
    "concat": mhd_concat,  # 按拓扑拼接 / Concatenate by topology
    "matmul": mhd_matmul,  # 按拓扑矩阵乘法 / Matrix multiplication by topology
}

# -------------------- 超边输出函数（超边单输出→多尾节点） / Edge Output Functions --------------------
def mhd_split(x: Tensor, indices: List[int], node_channels: List[int]) -> List[Tensor]:
    """
    按拓扑顺序拆分张量到多尾节点（单→多）
    Split tensor to multiple tail nodes by topology order (single → multiple)
    
    核心逻辑 / Core Logic:
    1. 按拓扑值绝对值排序tail节点 / Sort tail nodes by absolute value of topology indices
    2. 按排序后的节点通道数拆分张量 / Split tensor by sorted tail node channel sizes
    3. 映射回原始tail节点顺序 / Remap to original tail node order
    
    参数 / Args:
        x: 超边输出单张量 / Single edge output tensor
        indices: 拓扑值列表（tail节点为正数） / List of topology values (positive for tail nodes)
        node_channels: 原始tail节点顺序的通道数列表 / List of channel sizes in original tail node order
    返回 / Returns:
        拆分后的张量列表（匹配原始tail节点顺序） / Split tensors matching original tail node order
    """
    # Step 1: 配对原始tail节点索引和拓扑值 / Pair original tail node indices with topology values
    indexed_nodes = list(enumerate(indices))  # (original_idx, topology_val)
    
    # Step 2: 按拓扑值绝对值从小到大排序 / Sort by absolute value of topology indices
    sorted_nodes = sorted(indexed_nodes, key=lambda p: abs(p[1]))
    
    # Step 3: 提取排序后的原始索引和通道数 / Extract sorted original indices and channel sizes
    sorted_original_indices = [p[0] for p in sorted_nodes]
    sorted_channel_sizes = [node_channels[i] for i in sorted_original_indices]
    
    # Step 4: 按通道数拆分张量 / Split tensor by channel sizes
    split_tensors = torch.split(x, sorted_channel_sizes, dim=1)
    
    # Step 5: 回映射到原始tail节点顺序 / Remap to original tail node order
    tensor_map = {idx: t for idx, t in zip(sorted_original_indices, split_tensors)}
    original_order_tensors = [tensor_map[i] for i in range(len(indices))]
    
    return original_order_tensors

def mhd_svd(x: Tensor, indices: List[int], node_channels: List[int]) -> List[Tensor]:
    """
    按拓扑顺序分配SVD分解结果到多尾节点（单→多）
    Distribute SVD decomposition results to multiple tail nodes by topology order (single → multiple)
    
    核心逻辑 / Core Logic:
    1. 按拓扑值绝对值排序tail节点 / Sort tail nodes by absolute value of topology indices
    2. 分配SVD分量（U→1st, S→2nd, Vh→3rd） / Distribute SVD components (U→1st, S→2nd, Vh→3rd)
    3. 映射回原始tail节点顺序 / Remap to original tail node order
    
    参数 / Args:
        x: 超边输出单张量 / Single edge output tensor
        indices: 拓扑值列表（tail节点为正数） / List of topology values (positive for tail nodes)
        node_channels: tail节点通道数列表（用于维度适配） / List of tail node channel sizes (for dimension adaptation)
    返回 / Returns:
        SVD分量列表（匹配原始tail节点顺序） / List of SVD components matching original tail node order
    """
    # Step 1: 展平为2D矩阵（兼容批量） / Flatten to 2D matrix (batch compatible)
    if x.dim() > 2:
        batch_dim = x.shape[0]
        x_flat = x.reshape(batch_dim, -1) if batch_dim > 1 else x.reshape(-1, x.shape[-1])
    else:
        x_flat = x
    
    # Step 2: 执行SVD分解 / Perform SVD decomposition
    U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
    
    # Step 3: 配对原始tail节点索引和拓扑值 / Pair original tail node indices with topology values
    indexed_nodes = list(enumerate(indices))
    
    # Step 4: 按拓扑值绝对值排序 / Sort by absolute value of topology indices
    sorted_nodes = sorted(indexed_nodes, key=lambda p: abs(p[1]))
    
    # Step 5: 分配SVD分量到排序后的tail节点 / Distribute SVD components to sorted tail nodes
    svd_components = [U, S, Vh]
    sorted_tensors = []
    for i, (orig_idx, _) in enumerate(sorted_nodes):
        comp_idx = i % len(svd_components)
        tensor = svd_components[comp_idx]
        
        # 适配tail节点通道数 / Adapt to tail node channel size
        target_channels = node_channels[orig_idx]
        if tensor.dim() == 2 and tensor.shape[-1] != target_channels:
            tensor = F.adaptive_avg_pool1d(tensor.unsqueeze(1), target_channels).squeeze(1)
        sorted_tensors.append((orig_idx, tensor))
    
    # Step 6: 回映射到原始tail节点顺序 / Remap to original tail node order
    tensor_map = {idx: t for idx, t in sorted_tensors}
    original_order_tensors = [tensor_map[i] for i in range(len(indices))]
    
    return original_order_tensors

def mhd_lu(x: Tensor, indices: List[int], node_channels: List[int]) -> List[Tensor]:
    """
    按拓扑顺序分配LU分解结果到多尾节点（单→多）
    Distribute LU decomposition results to multiple tail nodes by topology order (single → multiple)
    
    核心逻辑 / Core Logic:
    1. 按拓扑值绝对值排序tail节点 / Sort tail nodes by absolute value of topology indices
    2. 分配LU分量（L→1st, U→2nd, P→3rd） / Distribute LU components (L→1st, U→2nd, P→3rd)
    3. 映射回原始tail节点顺序 / Remap to original tail node order
    
    参数 / Args:
        x: 超边输出单张量 / Single edge output tensor
        indices: 拓扑值列表（tail节点为正数） / List of topology values (positive for tail nodes)
        node_channels: tail节点通道数列表（用于维度适配） / List of tail node channel sizes (for dimension adaptation)
    返回 / Returns:
        LU分量列表（匹配原始tail节点顺序） / List of LU components matching original tail node order
    """
    # Step 1: 展平为2D矩阵（兼容批量） / Flatten to 2D matrix (batch compatible)
    if x.dim() > 2:
        batch_dim = x.shape[0]
        x_flat = x.reshape(batch_dim, -1) if batch_dim > 1 else x.reshape(-1, x.shape[-1])
    else:
        x_flat = x
    
    # Step 2: 执行LU分解 / Perform LU decomposition
    P, L, U = torch.linalg.lu(x_flat)
    
    # Step 3: 配对原始tail节点索引和拓扑值 / Pair original tail node indices with topology values
    indexed_nodes = list(enumerate(indices))
    
    # Step 4: 按拓扑值绝对值排序 / Sort by absolute value of topology indices
    sorted_nodes = sorted(indexed_nodes, key=lambda p: abs(p[1]))
    
    # Step 5: 分配LU分量到排序后的tail节点 / Distribute LU components to sorted tail nodes
    lu_components = [L, U, P]
    sorted_tensors = []
    for i, (orig_idx, _) in enumerate(sorted_nodes):
        comp_idx = i % len(lu_components)
        tensor = lu_components[comp_idx]
        
        # 适配tail节点通道数 / Adapt to tail node channel size
        target_channels = node_channels[orig_idx]
        if tensor.dim() == 2 and tensor.shape[-1] != target_channels:
            tensor = F.adaptive_avg_pool1d(tensor.unsqueeze(1), target_channels).squeeze(1)
        sorted_tensors.append((orig_idx, tensor))
    
    # Step 6: 回映射到原始tail节点顺序 / Remap to original tail node order
    tensor_map = {idx: t for idx, t in sorted_tensors}
    original_order_tensors = [tensor_map[i] for i in range(len(indices))]
    
    return original_order_tensors

# 注册输出函数 / Register output functions
MHD_EDGE_OUT_FUNCS: FuncMapping = {
    "split": mhd_split,    # 按拓扑拆分 / Split by topology
    "svd": mhd_svd,        # 按拓扑分配SVD / Distribute SVD by topology
    "lu": mhd_lu,          # 按拓扑分配LU / Distribute LU by topology
}

# ===================== 超图核心数据结构 / Hypergraph Core Data Structure =====================
@dataclass(unsafe_hash=True)  # 支持哈希，可放入set集合 / Support hash for set collection
class MHD_Node:
    """
    超图节点 / Hypergraph Node
    
    属性 / Attributes:
        id: 唯一标识（关联矩阵列索引） / Unique ID (column index of incidence matrix)
        name: 便于调用的名称 / Readable name for easy reference
        value: 节点特征张量（初始化值，前向传播更新） / Node feature tensor (init value, updated in forward)
        func: 头/尾函数映射 / Head/tail function mapping {"head": func_name, "tail": func_name}
    """
    id: int
    name: str
    value: torch.Tensor
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"}, hash=False)

@dataclass(unsafe_hash=True)  # 支持哈希，可放入set集合 / Support hash for set collection
class MHD_Edge:
    """
    超图边 / Hypergraph Edge
    
    属性 / Attributes:
        id: 唯一标识（关联矩阵行索引） / Unique ID (row index of incidence matrix)
        name: 便于调用的名称 / Readable name for easy reference
        value: 操作序列（字符串方法/张量模块混合） / Operation sequence (string methods + torch Modules)
        func: 输入/输出函数映射 / Input/output function mapping {"in": func_name, "out": func_name}
    """
    id: int
    name: str
    value: List[Union[str, nn.Module]] = field(hash=False)
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"}, hash=False)

@dataclass
class MHD_Topo:
    """
    超图拓扑 / Hypergraph Topology
    
    属性 / Attributes:
        id: 拓扑ID / Topology ID
        name: 拓扑名称 / Topology name
        value: 关联矩阵（整型张量，形状[num_edges, num_nodes]） / Incidence matrix (int tensor, shape [num_edges, num_nodes])
               - 负数：head节点，绝对值=顺序 / Negative: Head nodes, abs value = order
               - 正数：tail节点，绝对值=顺序 / Positive: Tail nodes, abs value = order
               - 零：无连接 / Zero: No connection
    """
    id: int
    name: str
    value: torch.Tensor

# ===================== 核心工具函数 / Core Utility Functions =====================
def mhd_parse_string_method(method_str: str, x: Tensor) -> Tensor:
    """
    解析字符串形式的张量方法并执行
    Parse and execute tensor method from string
    
    示例 / Examples:
        "reshape(1, 4, -1)" → x.reshape(1, 4, -1)
        "relu_()" → x.relu()
        "__add__(3)" → x.__add__(3)
    
    参数 / Args:
        method_str: 字符串形式的方法 / String-based method
        x: 输入张量 / Input tensor
    返回 / Returns:
        执行方法后的张量 / Tensor after method execution
    """
    if not isinstance(method_str, str):
        return x
    if '(' in method_str and ')' in method_str:
        method_name, args_str = method_str.split('(', 1)
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
        if hasattr(x, method_name):
            return getattr(x, method_name)(*args, **kwargs)
        else:
            raise ValueError(f"张量无此方法 / Tensor has no method: {method_name}")
    else:
        if hasattr(x, method_str):
            return getattr(x, method_str)()
        else:
            raise ValueError(f"张量无此方法 / Tensor has no method: {method_str}")

def mhd_register_head_func(name: str, func: Callable[[Tensor], Tensor]) -> None:
    """注册新的head节点函数 / Register new head node function"""
    MHD_NODE_HEAD_FUNCS[name] = func

def mhd_register_tail_func(name: str, func: Callable[[List[Tensor]], Tensor]) -> None:
    """注册新的tail节点函数 / Register new tail node function"""
    MHD_NODE_TAIL_FUNCS[name] = func

def mhd_register_edge_in_func(name: str, func: Callable[[List[Tensor], List[int]], Tensor]) -> None:
    """注册新的超边输入函数 / Register new edge input function"""
    MHD_EDGE_IN_FUNCS[name] = func

def mhd_register_edge_out_func(name: str, func: Callable[[Tensor, List[int]], List[Tensor]]) -> None:
    """注册新的超边输出函数 / Register new edge output function"""
    MHD_EDGE_OUT_FUNCS[name] = func

# ===================== 核心网络模块 / Core Network Modules =====================
class DNet(nn.Module):
    """
    动态网络（执行超边操作序列） / Dynamic Network (Execute edge operation sequence)
    
    功能 / Function:
        支持PyTorch模块 + 字符串形式的张量方法 / Support torch Modules + string-based tensor methods
    """
    def __init__(self, operations: List[Union[str, nn.Module]]):
        super().__init__()
        self.operations = nn.ModuleList()
        self.str_ops = []
        for op in operations:
            if isinstance(op, nn.Module):
                self.operations.append(op)
                self.str_ops.append(None)
            elif isinstance(op, str):
                self.operations.append(nn.Identity())
                self.str_ops.append(op)
            else:
                raise ValueError(f"不支持的操作类型 / Unsupported operation type: {type(op)}")

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：执行操作序列 / Forward: Execute operation sequence"""
        for op, str_op in zip(self.operations, self.str_ops):
            if str_op is not None:
                x = mhd_parse_string_method(str_op, x)
            else:
                x = op(x)
        return x

class HDNet(nn.Module):
    """
    超图动态网络 / Hypergraph Dynamic Network
    
    核心逻辑 / Core Logic:
        1. Head节点：特征共享到超边 / Head Node: Share features to hyperedges
        2. 边输入：按拓扑顺序聚合head节点特征 / Edge Input: Aggregate head node features by topology
        3. 边操作：执行超边专属操作 / Edge Operation: Execute edge-specific operations
        4. 边输出：按拓扑顺序分发到tail节点 / Edge Output: Distribute to tail nodes by topology
        5. Tail节点：聚合超边结果 / Tail Node: Aggregate edge results
    """
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo):
        super().__init__()
        # 索引映射（快速查找） / Index mapping (for quick lookup)
        self.node_id2obj = {node.id: node for node in nodes}
        self.node_name2id = {node.name: node.id for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        self.edge_name2id = {edge.name: edge.id for edge in edges}

        # 拓扑验证 / Topology validation
        self.topo = topo
        self._validate_topo()

        # 初始化超边操作网络 / Initialize edge operation networks
        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.value)

        # 节点特征缓存（初始化为节点默认值） / Node feature cache (init with node default value)
        self.node_values = {node.id: node.value.clone() for node in nodes}

    def _validate_topo(self) -> None:
        """验证关联矩阵维度和类型 / Validate incidence matrix dimension and type"""
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)
        
        # 维度检查 / Dimension check
        if self.topo.value.shape != (num_edges, num_nodes):
            raise ValueError(
                f"拓扑矩阵维度不匹配 / Topology matrix dimension mismatch: "
                f"期望 / Expected ({num_edges}, {num_nodes}), 实际 / Got {self.topo.value.shape}"
            )
        
        # 类型检查（必须为整型） / Type check (must be integer)
        dtype_name = str(self.topo.value.dtype).lower()
        if not any(keyword in dtype_name for keyword in ['int', 'long', 'short']):
            raise ValueError(f"拓扑矩阵必须为整型 / Topology matrix must be integer type: 实际 / Got {self.topo.value.dtype}")

    def forward(self, input_node_names: List[str], input_tensors: List[Tensor]) -> Dict[str, Tensor]:
        """
        超图前向传播 / Hypergraph forward propagation
        
        参数 / Args:
            input_node_names: 输入节点名称列表 / List of input node names
            input_tensors: 输入特征张量列表（匹配输入节点顺序） / List of input tensors (match input node order)
        返回 / Returns:
            节点名称到特征张量的映射 / Dict of {node_name: feature_tensor}
        """
        # 初始化输入节点特征 / Initialize input node features
        for name, tensor in zip(input_node_names, input_tensors):
            node_id = self.node_name2id[name]
            self.node_values[node_id] = tensor.to(dtype=torch.float32)

        # 按ID顺序处理所有超边 / Process all edges in ID order
        for edge_id in sorted(self.edge_id2obj.keys()):
            edge = self.edge_id2obj[edge_id]
            edge_net = self.edge_nets[edge.name]
            edge_topo = self.topo.value[edge_id]

            # Step 1: 解析head/tail节点（从拓扑矩阵） / Parse head/tail nodes from topology matrix
            # Head节点：拓扑值 < 0 / Head nodes: topology < 0
            head_mask = edge_topo < 0
            head_node_ids = torch.where(head_mask)[0].tolist()
            head_node_orders = edge_topo[head_mask].tolist()  # 负数，代表顺序 / Negative values (order)
            
            # Tail节点：拓扑值 > 0 / Tail nodes: topology > 0
            tail_mask = edge_topo > 0
            tail_node_ids = torch.where(tail_mask)[0].tolist()
            tail_node_orders = edge_topo[tail_mask].tolist()  # 正数，代表顺序 / Positive values (order)

            # Step 2: 获取head节点特征（应用head函数） / Get head node features (apply head function)
            head_tensors = []
            for node_id in head_node_ids:
                node = self.node_id2obj[node_id]
                head_func_name = node.func.get("head", "share")
                if head_func_name not in MHD_NODE_HEAD_FUNCS:
                    raise ValueError(
                        f"未知head节点函数 / Unknown head function: {head_func_name}, "
                        f"已注册 / Registered: {list(MHD_NODE_HEAD_FUNCS.keys())}"
                    )
                head_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](self.node_values[node_id])
                head_tensors.append(head_tensor)

            # Step 3: 聚合head节点特征（边输入函数） / Aggregate head node features (edge input function)
            edge_in_func_name = edge.func.get("in", "concat")
            if edge_in_func_name not in MHD_EDGE_IN_FUNCS:
                raise ValueError(
                    f"未知超边输入函数 / Unknown edge input function: {edge_in_func_name}, "
                    f"已注册 / Registered: {list(MHD_EDGE_IN_FUNCS.keys())}"
                )
            edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](head_tensors, head_node_orders)

            # Step 4: 执行超边操作 / Execute edge operations
            edge_output = edge_net(edge_input)

            # Step 5: 分发超边输出到tail节点（边输出函数） / Distribute edge output to tail nodes
            edge_out_func_name = edge.func.get("out", "split")
            if edge_out_func_name not in MHD_EDGE_OUT_FUNCS:
                raise ValueError(
                    f"未知超边输出函数 / Unknown edge output function: {edge_out_func_name}, "
                    f"已注册 / Registered: {list(MHD_EDGE_OUT_FUNCS.keys())}"
                )
            
            # 获取tail节点通道数（用于split/SVD/LU） / Get tail node channel sizes (for split/SVD/LU)
            tail_node_channels = [self.node_id2obj[node_id].value.shape[1] for node_id in tail_node_ids]
            
            # 执行输出函数（所有函数参数格式统一） / Execute output function (unified parameter format)
            tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                edge_output, tail_node_orders, tail_node_channels
            )

            # Step 6: 更新tail节点特征（应用tail函数） / Update tail node features (apply tail function)
            for node_id, tensor in zip(tail_node_ids, tail_tensors):
                node = self.node_id2obj[node_id]
                tail_func_name = node.func.get("tail", "sum")
                if tail_func_name not in MHD_NODE_TAIL_FUNCS:
                    raise ValueError(
                        f"未知tail节点函数 / Unknown tail function: {tail_func_name}, "
                        f"已注册 / Registered: {list(MHD_NODE_TAIL_FUNCS.keys())}"
                    )
                
                # 与现有特征聚合（如有） / Aggregate with existing features (if any)
                if node_id in self.node_values:
                    self.node_values[node_id] = MHD_NODE_TAIL_FUNCS[tail_func_name](
                        [self.node_values[node_id], tensor]
                    )
                else:
                    self.node_values[node_id] = tensor

        # 返回节点名称映射的特征 / Return features mapped by node name
        return {
            self.node_id2obj[node_id].name: tensor
            for node_id, tensor in self.node_values.items()
        }

class MHDNet(nn.Module):
    """
    多超图动态网络 / Multi-Hypergraph Dynamic Network
    
    功能 / Function:
        通过节点映射整合多个HDNet为全局超图 / Integrate multiple HDNets into global hypergraph via node mapping
    """
    def __init__(
        self,
        sub_hdnets: Dict[str, HDNet],
        node_mapping: List[Tuple[str, str, str]],  # (全局名, 子图名, 子节点名) / (global_name, sub_name, sub_node_name)
        in_nodes: List[str],
        out_nodes: List[str],
        onnx_save_path: Optional[str] = None,
    ):
        super().__init__()
        self.sub_hdnets = nn.ModuleDict(sub_hdnets)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        # 构建全局超图 / Build global hypergraph
        self.global_hdnet = self._build_global_hypergraph(node_mapping)

        # 导出ONNX（如指定路径） / Export to ONNX (if path provided)
        if onnx_save_path:
            self._export_to_onnx(onnx_save_path)

    def _build_global_hypergraph(self, node_mapping: List[Tuple[str, str, str]]) -> HDNet:
        """
        构建全局超图（仅包含显式映射的节点） / Build global hypergraph (only explicitly mapped nodes)
        """
        # 初始化ID计数器 / Initialize ID counters
        node_id_counter = 0
        edge_id_counter = 0

        # 映射表：子图→全局图 / Mapping tables: subgraph → global graph
        sub2global_node = {}  # (子图名, 子节点名) → (全局节点ID, 全局节点名)
        sub2global_edge = {}  # (子图名, 子边名) → 全局边ID

        # 收集全局节点（仅显式映射的节点） / Collect global nodes (only explicitly mapped)
        global_nodes = set()
        for global_node_name, sub_name, sub_node_name in node_mapping:
            key = (sub_name, sub_node_name)
            if key not in sub2global_node:
                sub_hdnet = self.sub_hdnets[sub_name]
                sub_node_id = sub_hdnet.node_name2id[sub_node_name]
                sub_node = sub_hdnet.node_id2obj[sub_node_id]
                
                # 创建全局节点（复制子节点属性） / Create global node (copy sub-node properties)
                global_node = MHD_Node(
                    id=node_id_counter,
                    name=global_node_name,
                    value=sub_node.value.clone(),
                    func=sub_node.func.copy()
                )
                global_nodes.add(global_node)
                sub2global_node[key] = (node_id_counter, global_node_name)
                node_id_counter += 1

        # 收集全局边和拓扑（仅包含映射节点） / Collect global edges and topology (only mapped nodes)
        global_edges = set()
        global_topo_data = []
        
        for sub_name, sub_hdnet in self.sub_hdnets.items():
            for sub_edge_id in sorted(sub_hdnet.edge_id2obj.keys()):
                sub_edge = sub_hdnet.edge_id2obj[sub_edge_id]
                sub_edge_name = sub_edge.name
                key = (sub_name, sub_edge_name)
                
                if key not in sub2global_edge:
                    # 分配全局边ID / Assign global edge ID
                    sub2global_edge[key] = edge_id_counter
                    
                    # 创建全局边（复制子边属性） / Create global edge (copy sub-edge properties)
                    global_edge = MHD_Edge(
                        id=edge_id_counter,
                        name=f"{sub_name}_{sub_edge_name}",
                        value=sub_edge.value.copy(),
                        func=sub_edge.func.copy()
                    )
                    global_edges.add(global_edge)

                    # 转换子拓扑到全局拓扑（仅映射节点） / Convert sub topology to global topology
                    sub_topo_row = sub_hdnet.topo.value[sub_edge_id]
                    global_topo_row = torch.zeros(node_id_counter, dtype=torch.int64)
                    
                    # 子节点ID映射到全局节点ID / Map sub-node IDs to global-node IDs
                    for sub_node_id, val in enumerate(sub_topo_row):
                        if val == 0:
                            continue
                        sub_node_name = sub_hdnet.node_id2obj[sub_node_id].name
                        map_key = (sub_name, sub_node_name)
                        if map_key in sub2global_node:
                            global_node_id, _ = sub2global_node[map_key]
                            global_topo_row[global_node_id] = val

                    global_topo_data.append(global_topo_row)
                    edge_id_counter += 1

        # 创建全局拓扑矩阵 / Create global topology matrix
        if global_topo_data:
            global_topo_value = torch.stack(global_topo_data)
        else:
            global_topo_value = torch.empty(0, 0, dtype=torch.int64)
        
        global_topo = MHD_Topo(
            id=0,
            name="global_topo",
            value=global_topo_value
        )

        # 创建全局HDNet / Create global HDNet
        return HDNet(nodes=global_nodes, edges=global_edges, topo=global_topo)

    def _export_to_onnx(self, onnx_save_path: str) -> None:
        """
        导出模型为ONNX格式（兼容PyTorch 1.x/2.x）
        Export model to ONNX format (compatible with PyTorch 1.x/2.x)
        
        参数 / Args:
            onnx_save_path: ONNX保存路径 / ONNX save path
        """
        self.eval()
        
        # 从全局超图获取输入形状 / Get input shapes from global hypergraph
        input_shapes = []
        for node_name in self.in_nodes:
            for node in self.global_hdnet.node_id2obj.values():
                if node.name == node_name:
                    input_shapes.append(node.value.shape)
                    break
        
        # 生成示例输入张量 / Generate example input tensors
        inputs = [torch.randn(*shape) for shape in input_shapes]
        
        # 动态轴配置（批量维度可变） / Dynamic axes (variable batch size)
        dynamic_axes = {
            **{f"input_{node}": {0: "batch_size"} for node in self.in_nodes},
            **{f"output_{node}": {0: "batch_size"} for node in self.out_nodes}
        }

        # 根据PyTorch版本选择opset / Select opset based on PyTorch version
        torch_version = [int(v) for v in torch.__version__.split('.')[:2]]
        opset_version = 17 if torch_version >= [2, 0] else 11

        # 导出ONNX / Export to ONNX
        abs_onnx_path = os.path.abspath(onnx_save_path)
        torch.onnx.export(
            self,
            inputs,
            abs_onnx_path,
            input_names=[f"input_{node}" for node in self.in_nodes],
            output_names=[f"output_{node}" for node in self.out_nodes],
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            opset_version=opset_version
        )
        print(f"模型已导出至 / Model exported to: {abs_onnx_path} (opset={opset_version})")

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        全局前向传播 / Global forward propagation
        
        参数 / Args:
            inputs: 输入张量列表（匹配in_nodes顺序） / List of input tensors (match in_nodes order)
        返回 / Returns:
            输出张量列表（匹配out_nodes顺序） / List of output tensors (match out_nodes order)
        """
        if len(inputs) != len(self.in_nodes):
            raise ValueError(
                f"输入数量不匹配 / Input count mismatch: "
                f"期望 / Expected {len(self.in_nodes)}, 实际 / Got {len(inputs)}"
            )
        
        # 执行全局超图前向传播 / Execute global hypergraph forward
        global_outputs = self.global_hdnet.forward(self.in_nodes, inputs)
        
        # 提取输出张量 / Extract output tensors
        outputs = [global_outputs[node] for node in self.out_nodes]
        return outputs

"""
MHD_Net 超图动态网络框架
全局节点在任意 HDNet 里作为 head / tail 都保持相同形状
全局节点可以是多个 HDNet 的 head，也可以是多个 HDNet 的 tail
=========================================================================
只修改 example 构造，不修改框架内部逻辑
=========================================================================
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# ======================= 下面这一整块：完全 100% 不动 =======================
# 你原来的所有核心代码
# MHD_Node, MHD_Edge, MHD_Topo
# 所有函数：mhd_concat, mhd_matmul, mhd_split, mhd_svd, mhd_lu
# HDNet, DNet, MHDNet
# 我这里直接省略，因为你强调：不许动前面！
# =========================================================================

# ===================== 只在这里写干净 example，满足你的形状要求 =====================
def example_mhdnet_clean():
    """
    严格满足你的要求：
    1. 全局节点在所有 HDNet 里形状完全一样
    2. 同一个全局节点可以是多个 HDNet 的 head
    3. 同一个全局节点可以是多个 HDNet 的 tail
    4. 不修改任何框架内部代码，不增强、不统一、不锁死通道
    """
    device = torch.device("cpu")

    # ===================== 全局节点：形状固定！ =====================
    # 这个全局节点会被多个 HDNet 当作 head
    g_head = MHD_Node(
        id=0,
        name="global_node_head",
        value=torch.randn(1, 8, 16, 16).to(device),  # 固定形状
        func={"head": "share"}
    )

    # 这个全局节点会被多个 HDNet 当作 tail
    g_tail = MHD_Node(
        id=1,
        name="global_node_tail",
        value=torch.randn(1, 8, 16, 16).to(device),  # 固定形状
        func={"tail": "sum"}
    )

    # ===================== HDNet 1 =====================
    # 全局节点 g_head 作为 HDNet1 的 head
    # 全局节点 g_tail 作为 HDNet1 的 tail
    nodes1 = {
        MHD_Node(id=0, name="in1", value=g_head.value.clone(), func={"head": "share"}),
        MHD_Node(id=1, name="out1", value=g_tail.value.clone(), func={"tail": "sum"}),
    }
    edges1 = {
        MHD_Edge(id=0, name="edge1", value=[nn.Identity()], func={"in": "concat", "out": "split"})
    }
    topo1 = MHD_Topo(id=0, name="topo1", value=torch.tensor([[-1, 1]]).long())
    hdnet1 = HDNet(nodes=nodes1, edges=edges1, topo=topo1)

    # ===================== HDNet 2 =====================
    # 全局节点 g_head 也作为 HDNet2 的 head
    # 全局节点 g_tail 也作为 HDNet2 的 tail
    nodes2 = {
        MHD_Node(id=0, name="in2", value=g_head.value.clone(), func={"head": "share"}),
        MHD_Node(id=1, name="out2", value=g_tail.value.clone(), func={"tail": "sum"}),
    }
    edges2 = {
        MHD_Edge(id=0, name="edge2", value=[nn.Identity()], func={"in": "concat", "out": "split"})
    }
    topo2 = MHD_Topo(id=0, name="topo2", value=torch.tensor([[-1, 1]]).long())
    hdnet2 = HDNet(nodes=nodes2, edges=edges2, topo=topo2)

    # ===================== 全局节点映射 =====================
    # 关键：
    # global_node_head → 是 hdnet1 和 hdnet2 的 head
    # global_node_tail → 是 hdnet1 和 hdnet2 的 tail
    # 且形状全程一致

    node_mapping = [
        ("global_node_head", "hdnet1", "in1"),
        ("global_node_head", "hdnet2", "in2"),
        ("global_node_tail", "hdnet1", "out1"),
        ("global_node_tail", "hdnet2", "out2"),
    ]

    model = MHDNet(
        sub_hdnets={
            "hdnet1": hdnet1,
            "hdnet2": hdnet2,
        },
        node_mapping=node_mapping,
        in_nodes=["global_node_head"],
        out_nodes=["global_node_tail"],
    )

    # 前向：输入形状 = 全局节点形状
    x = g_head.value.clone()
    out = model([x])

    print("✅ 全局节点形状全程一致")
    print("输入形状:", x.shape)
    print("输出形状:", out[0].shape)
    print("global_node_head 是 HDNet1 & HDNet2 的 head")
    print("global_node_tail 是 HDNet1 & HDNet2 的 tail")

if __name__ == "__main__":
    example_mhdnet_clean()
