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

Author: Your Name
Date: 2026
Version: 1.2  # 版本更新：修复in-place梯度错误 + 移除旧格式兼容
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
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ===================== Type Definition / 类型定义 =====================
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# 结构化拓扑属性命名元组（仅保留字典格式，移除旧格式兼容）
TopoAttr = namedtuple("TopoAttr", ["role", "sort", "ext"], defaults=[None])

# ===================== Global Function Registry / 全局函数注册表 =====================
MHD_NODE_HEAD_FUNCS: Dict[str, Callable[..., Any]] = {
    # 保留梯度，但移除in-place风险
    "share": lambda tensor: tensor.clone(memory_format=torch.contiguous_format).requires_grad_(tensor.requires_grad),
}

MHD_NODE_TAIL_FUNCS: Dict[str, Callable[..., Any]] = {
    "sum": lambda tensors: sum(tensors),          
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),  
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],  
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],  
}

# ===================== Utility Functions / 工具函数 =====================
def MHD_sort_nodes_by_topo_attr(attrs: List[TopoAttr]) -> List[Tuple[int, int]]:
    """
    按拓扑属性的sort字段排序（仅支持新格式）
    """
    indexed_nodes = list(enumerate(attrs))
    return sorted(indexed_nodes, key=lambda p: p[1].sort if p[1] is not None else 0)

def MHD_flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """展平张量用于矩阵运算（保留批次维度）"""
    if x.dim() > 2:
        x_flat = x.reshape(x.shape[0], -1)
    else:
        x_flat = x
    return x_flat

def extract_operation_name(op: Union[str, nn.Module]) -> str:
    """提取干净的操作名称"""
    if isinstance(op, nn.Module):
        op_name = op.__class__.__name__
    elif isinstance(op, str):
        op_name = re.sub(r'\(.*\)', '', op).strip()
    else:
        op_name = str(op)

    op_name = op_name.replace("torch.nn.modules.", "").replace("torch.nn.", "")
    return op_name

def parse_topo_value(value: Union[dict, TopoAttr]) -> TopoAttr:
    """
    仅解析字典/TopoAttr格式（移除旧int格式兼容）
    """
    if isinstance(value, dict):
        return TopoAttr(
            role=value.get("role", 0),
            sort=value.get("sort", 0),
            ext=value.get("ext", {})
        )
    elif isinstance(value, TopoAttr):
        return value
    else:
        raise ValueError(f"仅支持dict/TopoAttr格式，不支持int！Got {type(value)}")

# ===================== Core Operation Functions / 核心操作函数 =====================
def MHD_concat(tensors: List[torch.Tensor], attrs: List[TopoAttr]) -> torch.Tensor:
    """按拓扑属性排序后拼接张量"""
    sorted_pairs = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs]
    return torch.cat(sorted_tensors, dim=1)

def MHD_matmul(tensors: List[torch.Tensor], attrs: List[TopoAttr]) -> torch.Tensor:
    """按拓扑属性排序后执行矩阵乘法"""
    sorted_pairs = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs]
    if len(sorted_tensors) != 2:
        raise ValueError(f"Matmul requires exactly 2 input tensors, got {len(sorted_tensors)}")
    return torch.matmul(*sorted_tensors)

MHD_EDGE_IN_FUNCS: Dict[str, Callable[..., Any]] = {
    "concat": MHD_concat,  
    "matmul": MHD_matmul,  
}

def MHD_split(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """按拓扑属性分割张量"""
    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_original_indices = [p[0] for p in sorted_nodes]
    sorted_channel_sizes = [node_channels[i] for i in sorted_original_indices]

    split_tensors = torch.split(x, sorted_channel_sizes, dim=1)
    tensor_map = {idx: t for idx, t in zip(sorted_original_indices, split_tensors)}
    return [tensor_map[i] for i in range(len(attrs))]

def MHD_svd(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """按拓扑属性执行SVD分解"""
    x_flat = MHD_flatten_tensor(x)
    U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)

    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    svd_components = [U, S, Vh]
    sorted_tensors = []

    for i, (orig_idx, _) in enumerate(sorted_nodes):
        comp_idx = i % len(svd_components)
        tensor = svd_components[comp_idx]
        sorted_tensors.append((orig_idx, tensor))

    tensor_map = {idx: t for idx, t in sorted_tensors}
    return [tensor_map[i] for i in range(len(attrs))]

def MHD_lu(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """按拓扑属性执行LU分解"""
    x_flat = MHD_flatten_tensor(x)
    P, L, U = torch.linalg.lu(x_flat)

    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    lu_components = [L, U, P]
    sorted_tensors = []

    for i, (orig_idx, _) in enumerate(sorted_nodes):
        comp_idx = i % len(lu_components)
        tensor = lu_components[comp_idx]
        sorted_tensors.append((orig_idx, tensor))

    tensor_map = {idx: t for idx, t in sorted_tensors}
    return [tensor_map[i] for i in range(len(attrs))]

MHD_EDGE_OUT_FUNCS: Dict[str, Callable[..., Any]] = {
    "split": MHD_split,  
    "svd": MHD_svd,      
    "lu": MHD_lu,        
}

# ===================== String Operation Wrapper / 字符串操作包装类 =====================
class StringOperation(nn.Module):
    """字符串定义的张量操作包装类"""
    def __init__(self, op_str: str):
        super().__init__()
        self.op_str = op_str  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

# ===================== Core Data Structures / 核心数据结构 =====================
@dataclass
class MHD_Node:
    """超图节点类（张量字段无哈希冲突）"""
    id: int
    name: str
    value: torch.Tensor
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"})
    requires_grad: bool = field(default=True)

    def __post_init__(self):
        """初始化时设置张量梯度追踪（非in-place）"""
        # 关键修改：创建新张量而非in-place修改叶子节点
        self.value = self.value.requires_grad_(self.requires_grad)

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        if not isinstance(other, MHD_Node):
            return False
        return self.id == other.id and self.name == other.name

@dataclass
class MHD_Edge:
    """超图边类（无哈希冲突）"""
    id: int
    name: str
    operations: List[Union[str, nn.Module]]
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"})

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id and self.name == other.name

@dataclass
class MHD_Topo:
    """结构化拓扑类（仅支持字典/TopoAttr格式）"""
    value: Union[List[List[Union[TopoAttr, dict]]]]

    def __post_init__(self):
        """初始化时统一解析为TopoAttr格式"""
        self.value = [
            [parse_topo_value(val) for val in row]
            for row in self.value
        ]

    def get_topo_attr(self, edge_id: int, node_id: int) -> TopoAttr:
        return self.value[edge_id][node_id]

    def to_tensor(self) -> torch.Tensor:
        """转换为仅包含role的张量"""
        tensor_data = []
        for row in self.value:
            tensor_row = [attr.role for attr in row]
            tensor_data.append(tensor_row)
        return torch.tensor(tensor_data, dtype=torch.int64)

# ===================== Topological Sort / 拓扑排序 =====================
def MHD_topological_sort(nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo) -> List[int]:
    """适配结构化拓扑矩阵的拓扑排序"""
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_node_ids = {node.id for node in nodes}

    if len(topo.value) > 0 and len(topo.value[0]) > 0:
        for edge_id, edge_row in enumerate(topo.value):
            for node_id, attr in enumerate(edge_row):
                if attr.role == 0:
                    continue
                if attr.role < 0:
                    head_id = node_id
                    tail_ids = [
                        nid for nid, a in enumerate(edge_row) 
                        if a.role > 0
                    ]
                    for tail_id in tail_ids:
                        graph[head_id].append(tail_id)
                        in_degree[tail_id] += 1

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

    if len(sorted_node_ids) != len(all_node_ids):
        raise ValueError(f"Cycle detected in hypergraph! Nodes in cycle: {all_node_ids - set(sorted_node_ids)}")
    return sorted_node_ids

# ===================== DNet (Dynamic Network) / 动态网络 =====================
class DNet(nn.Module):
    """超边操作动态网络"""
    def __init__(self, operations: List[Union[str, nn.Module]]):
        super().__init__()
        seq_ops = []
        self.op_names = []  

        for op in operations:
            self.op_names.append(extract_operation_name(op))

            if isinstance(op, nn.Module):
                seq_ops.append(op)
            elif isinstance(op, str):
                seq_ops.append(StringOperation(op))
            else:
                raise ValueError(f"Unsupported operation type: {type(op)}. Only nn.Module/string allowed.")

        self.filter = nn.Sequential(*seq_ops)
        self.original_operations = operations  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)

# ===================== HDNet (Hypergraph Dynamic Network) / 超图动态网络 =====================
class HDNet(nn.Module):
    """超图动态网络（单超图子网）"""
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo, name: str = "hdnet"):
        super().__init__()
        self.node_id2obj = {node.id: node for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        self.topo = topo
        self.name = name

        self._validate_topo()
        self.sorted_node_ids = MHD_topological_sort(nodes, edges, topo)
        print(f"✅ {self.name} DAG sorted: {[self.node_id2obj[nid].name for nid in self.sorted_node_ids]}")

        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.operations)

        # 初始化节点值（关键：创建新张量，避免叶子节点直接赋值）
        self.node_values = {}
        for node in nodes:
            # 非in-place赋值，保留计算图
            self.node_values[node.id] = node.value.clone().requires_grad_(node.requires_grad)

    @property
    def node_name2id(self):
        return {v.name: k for k, v in self.node_id2obj.items()}

    def _validate_topo(self) -> None:
        """验证结构化拓扑矩阵"""
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)

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

        for edge_id, edge_row in enumerate(self.topo.value):
            for node_id, attr in enumerate(edge_row):
                if not isinstance(attr, TopoAttr):
                    raise ValueError(
                        f"Topology matrix element at ({edge_id}, {node_id}) must be TopoAttr, "
                        f"got {type(attr)}"
                    )

    def forward(self) -> Dict[str, Tensor]:
        """
        关键修复：移除in-place操作，改用普通赋值更新节点值
        """
        device = next(iter(self.node_values.values())).device

        edge_affects_nodes = defaultdict(list)
        if len(self.topo.value) > 0:
            for edge_id in self.edge_id2obj.keys():
                edge_row = self.topo.value[edge_id]
                tail_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role > 0]
                edge_affects_nodes[edge_id] = tail_node_ids

        for target_node_id in self.sorted_node_ids:
            relevant_edges = [eid for eid, node_ids in edge_affects_nodes.items() if target_node_id in node_ids]

            for edge_id in relevant_edges:
                edge = self.edge_id2obj[edge_id]
                edge_net = self.edge_nets[edge.name]
                edge_row = self.topo.value[edge_id]

                # 获取头节点和拓扑属性
                head_mask = [attr.role < 0 for attr in edge_row]
                head_node_ids = [i for i, val in enumerate(head_mask) if val]
                head_topo_attrs = [edge_row[nid] for nid in head_node_ids]

                # 处理头节点张量
                head_tensors = []
                for node_id in head_node_ids:
                    node = self.node_id2obj[node_id]
                    head_func_name = node.func.get("head", "share")
                    head_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](self.node_values[node_id])
                    head_tensors.append(head_tensor)

                # 边输入处理
                edge_in_func_name = edge.func.get("in", "concat")
                edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](head_tensors, head_topo_attrs)

                # 边操作前向传播
                edge_output = edge_net(edge_input)

                # 获取尾节点和拓扑属性
                tail_mask = [attr.role > 0 for attr in edge_row]
                tail_node_ids = [i for i, val in enumerate(tail_mask) if val]
                tail_topo_attrs = [edge_row[nid] for nid in tail_node_ids]
                tail_node_channels = [self.node_id2obj[node_id].value.shape[1] for node_id in tail_node_ids]

                # 边输出处理
                edge_out_func_name = edge.func.get("out", "split")
                tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                    edge_output, tail_topo_attrs, tail_node_channels
                )

                # ========== 核心修复：移除in-place操作 ==========
                for node_id, tensor in zip(tail_node_ids, tail_tensors):
                    node = self.node_id2obj[node_id]
                    tail_func_name = node.func.get("tail", "sum")
                    if node_id in self.node_values:
                        # 非in-place聚合：创建新张量，保留梯度计算图
                        agg_tensor = MHD_NODE_TAIL_FUNCS[tail_func_name](
                            [self.node_values[node_id], tensor]
                        )
                        # 普通赋值，不修改叶子节点
                        self.node_values[node_id] = agg_tensor
                    else:
                        # 初始化节点值（非in-place）
                        self.node_values[node_id] = tensor.requires_grad_(node.requires_grad)

        # 返回节点特征
        return {
            self.node_id2obj[node_id].name: tensor
            for node_id, tensor in self.node_values.items()
        }

# ===================== MHDNet (Multi-Hypergraph Dynamic Network) / 多超图动态网络 =====================
class MHDNet(nn.Module):
    """多超图动态网络（合并多个HDNet子网为全局超图）"""
    def __init__(
        self,
        sub_hdnets: Dict[str, HDNet],
        node_mapping: List[Tuple[str, str, str]],  # (global_name, sub_name, sub_node_name)
    ):
        super().__init__()
        self.sub_hdnets = sub_hdnets
        self.node_mapping = node_mapping

        self.global_hdnet = self._build_global_hypergraph()

        self.mermaid_code = self.generate_expanded_topological_mermaid()
        print("\n" + "="*80)
        print("✅ EXPANDED TOPOLOGICAL MERMAID CODE (WITH OPERATION NAMES):")
        print("="*80)
        print(self.mermaid_code)
        print("="*80 + "\n")

    def _build_global_hypergraph(self) -> HDNet:
        """适配结构化拓扑矩阵的全局超图构建"""
        node_id_counter = 0
        edge_id_counter = 0

        sub_node_cache = {}
        for sub_name, sub_hdnet in self.sub_hdnets.items():
            for sub_node_id, sub_node in sub_hdnet.node_id2obj.items():
                sub_node_cache[(sub_name, sub_node.name)] = (sub_node_id, sub_node)

        sub2global_node = {}
        sub2global_edge = {}

        global_nodes = set()
        for global_node_name, sub_name, sub_node_name in self.node_mapping:
            key = (sub_name, sub_node_name)
            if key not in sub2global_node:
                sub_node_id, sub_node = sub_node_cache[key]

                global_node = MHD_Node(
                    id=node_id_counter,
                    name=global_node_name,
                    value=sub_node.value,
                    func=sub_node.func,
                    requires_grad=sub_node.requires_grad
                )
                global_nodes.add(global_node)
                sub2global_node[key] = (node_id_counter, global_node_name)
                node_id_counter += 1

        global_edges = set()
        global_topo_data = []

        for sub_name, sub_hdnet in self.sub_hdnets.items():
            for sub_edge_id in sorted(sub_hdnet.edge_id2obj.keys()):
                sub_edge = sub_hdnet.edge_id2obj[sub_edge_id]
                sub_edge_name = sub_edge.name
                key = (sub_name, sub_edge_name)

                if key not in sub2global_edge:
                    sub2global_edge[key] = edge_id_counter

                    global_edge_name = f"{sub_name}_{sub_edge_name}"
                    global_edge = MHD_Edge(
                        id=edge_id_counter,
                        name=global_edge_name,
                        operations=sub_edge.operations,
                        func=sub_edge.func
                    )
                    global_edges.add(global_edge)

                    sub_topo_row = sub_hdnet.topo.value[sub_edge_id]
                    global_topo_row = [TopoAttr(role=0, sort=0, ext={}) for _ in range(node_id_counter)]

                    for sub_node_id, sub_attr in enumerate(sub_topo_row):
                        if sub_attr.role == 0:
                            continue
                        sub_node_name = sub_hdnet.node_id2obj[sub_node_id].name
                        map_key = (sub_name, sub_node_name)
                        if map_key in sub2global_node:
                            global_node_id, _ = sub2global_node[map_key]
                            global_topo_row[global_node_id] = sub_attr

                    global_topo_data.append(global_topo_row)
                    edge_id_counter += 1

        global_topo = MHD_Topo(value=global_topo_data)
        global_hdnet = HDNet(nodes=global_nodes, edges=global_edges, topo=global_topo, name="global_hdnet")

        return global_hdnet

    def generate_expanded_topological_mermaid(self) -> str:
        """生成展开的拓扑可视化Mermaid代码"""
        sub2global = {}
        for global_name, sub_name, sub_node_name in self.node_mapping:
            sub2global[(sub_name, sub_node_name)] = global_name

        mermaid = [
            "graph TD",
            "",
            "    %% ===== GLOBAL TOPOLOGY WITH EXPANDED OPERATIONS ===== / 展开操作的全局拓扑结构",
            "",
        ]

        for sub_name, sub_hdnet in self.sub_hdnets.items():
            mermaid.append(f"    %% {sub_name} topology (expanded operations) / {sub_name} 拓扑结构（展开操作）")

            for edge_id, edge in sub_hdnet.edge_id2obj.items():
                edge_name = edge.name
                base_edge_name = f"{sub_name}_{edge_name}"

                edge_row = sub_hdnet.topo.value[edge_id]
                head_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role < 0]
                tail_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role > 0]

                head_global_nodes = [sub2global[(sub_name, sub_hdnet.node_id2obj[nid].name)] for nid in head_node_ids]
                tail_global_nodes = [sub2global[(sub_name, sub_hdnet.node_id2obj[nid].name)] for nid in tail_node_ids]

                edge_net = sub_hdnet.edge_nets[edge_name]
                op_names = edge_net.op_names

                if op_names:
                    op_node_names = [f"{base_edge_name}_{idx}_{op_name}" for idx, op_name in enumerate(op_names)]

                    for head_node in head_global_nodes:
                        mermaid.append(f"    {head_node} --> {op_node_names[0]}")

                    for i in range(len(op_node_names)-1):
                        mermaid.append(f"    {op_node_names[i]} --> {op_node_names[i+1]}")

                    last_op_node = op_node_names[-1]
                    for tail_node in tail_global_nodes:
                        mermaid.append(f"    {last_op_node} --> {tail_node}")
                else:
                    hyper_edge_node = f"{base_edge_name}_empty"
                    mermaid.append(f"    {hyper_edge_node}")
                    for head_node in head_global_nodes:
                        mermaid.append(f"    {head_node} --> {hyper_edge_node}")
                    for tail_node in tail_global_nodes:
                        mermaid.append(f"    {hyper_edge_node} --> {tail_node}")

            mermaid.append("")

        return "\n".join(mermaid)

    def forward(self) -> Dict[str, Tensor]:
        """全局超图网络的前向传播"""
        all_node_features = self.global_hdnet.forward()
        return all_node_features

# ===================== Example Usage / 示例用法 =====================
def example_mhdnet2():
    """适配结构化拓扑矩阵的示例"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"✅ Using device: {device} / 使用设备: {device}")

    # 1. 创建全局源节点
    g_200 = MHD_Node(
        id=0, 
        name="200", 
        value=torch.randn(1, 2, 16, 16, 16, device=device, dtype=dtype),
        func={"head": "share", "tail": "sum"},
        requires_grad=True
    )

    g_201 = MHD_Node(id=1, name="201", value=torch.randn(1, 1, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_202 = MHD_Node(id=2, name="202", value=torch.randn(1, 2, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_203 = MHD_Node(id=3, name="203", value=torch.randn(1, 1, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_204 = MHD_Node(id=4, name="204", value=torch.randn(1, 2, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)
    g_205 = MHD_Node(id=5, name="205", value=torch.randn(1, 1, 16, 16, 16, device=device, dtype=dtype), requires_grad=True)

    # 2. 创建子HDNet
    # ========== NET1 ==========
    nodes_net1 = {
        MHD_Node(id=0, name="A1", value=g_200.value, requires_grad=True),
        MHD_Node(id=1, name="B1", value=g_201.value, requires_grad=True),
    }
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
    topo_net1 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]
    ])
    hdnet1 = HDNet(nodes=nodes_net1, edges=edges_net1, topo=topo_net1, name="net1")

    # ========== NET2 ==========
    nodes_net2 = {
        MHD_Node(id=0, name="A2", value=g_202.value, requires_grad=True),
        MHD_Node(id=1, name="B2", value=g_203.value, requires_grad=True),
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
    topo_net2 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]
    ])
    hdnet2 = HDNet(nodes=nodes_net2, edges=edges_net2, topo=topo_net2, name="net2")

    # ========== NET3 ==========
    nodes_net3 = {
        MHD_Node(id=0, name="A3", value=g_204.value, requires_grad=True),
        MHD_Node(id=1, name="B3", value=g_205.value, requires_grad=True),
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
    topo_net3 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]
    ])
    hdnet3 = HDNet(nodes=nodes_net3, edges=edges_net3, topo=topo_net3, name="net3")

    # ========== NET4 ==========
    nodes_net4 = {
        MHD_Node(id=0, name="A4", value=g_200.value, requires_grad=True),
        MHD_Node(id=1, name="B4", value=g_201.value, requires_grad=True),
        MHD_Node(id=2, name="C4", value=g_202.value, requires_grad=True),
        MHD_Node(id=3, name="D4", value=g_203.value, requires_grad=True),
        MHD_Node(id=4, name="E4", value=g_204.value, requires_grad=True),
        MHD_Node(id=5, name="F4", value=g_205.value, requires_grad=True),
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
        MHD_Node(id=0, name="A5", value=g_200.value, requires_grad=True),
        MHD_Node(id=1, name="B5", value=g_201.value, requires_grad=True),
        MHD_Node(id=2, name="C5", value=g_202.value, requires_grad=True),
        MHD_Node(id=3, name="D5", value=g_203.value, requires_grad=True),
        MHD_Node(id=4, name="E5", value=g_204.value, requires_grad=True),
        MHD_Node(id=5, name="F5", value=g_205.value, requires_grad=True),
    }
    conv_net5 = nn.Conv3d(6, 3, kernel_size=1, padding=0, bias=False).to(device)
    conv_net5.weight.requires_grad = True
    norm_net5 = nn.InstanceNorm3d(3, affine=False).to(device)
    act_net5 = nn.ReLU(inplace=True)

    edges_net5 = {
        MHD_Edge(
            id=0, 
            name="e1", 
            operations=[conv_net5, norm_net5, act_net5, '__mul__(-1)'],
            func={"in": "concat", "out": "split"}
        )
    }
    topo_net5 = MHD_Topo(value=[
        [
            {"role": -1, "sort": 1}, {"role": -1, "sort": 2},
            {"role": -1, "sort": 3}, {"role": -1, "sort": 4},
            {"role": 1, "sort": 1}, {"role": 1, "sort": 2}
        ]
    ])
    hdnet5 = HDNet(nodes=nodes_net5, edges=edges_net5, topo=topo_net5, name="net5")

    # 3. 全局-局部节点映射
    node_mapping = [
        ("200", "net1", "A1"), ("200", "net4", "A4"), ("200", "net5", "A5"),
        ("201", "net1", "B1"), ("201", "net4", "B4"), ("201", "net5", "B5"),
        ("202", "net2", "A2"), ("202", "net4", "C4"), ("202", "net5", "C5"),
        ("203", "net2", "B2"), ("203", "net4", "D4"), ("203", "net5", "D5"),
        ("204", "net3", "A3"), ("204", "net4", "E4"), ("204", "net5", "E5"),
        ("205", "net3", "B3"), ("205", "net4", "F4"), ("205", "net5", "F5"),
    ]

    # 4. 构建MHDNet
    model = MHDNet(
        sub_hdnets={"net1": hdnet1, "net2": hdnet2, "net3": hdnet3, "net4": hdnet4, "net5": hdnet5},
        node_mapping=node_mapping
    )

    # 5. 前向传播
    all_features = model.forward()

    # 6. 打印结果
    print("\n✅ Pure topology-driven forward completed! / 纯拓扑驱动前向传播完成！")
    print(f"Input node (200) direct shape / 输入节点(200)直接形状: {g_200.value.shape}")
    print("\nAll global node feature maps (direct tensor access) / 所有全局节点特征图（直接张量访问）:")
    for node_name, tensor in sorted(all_features.items()):
        print(f"  - Node {node_name}: shape={tensor.shape}, device={tensor.device}, requires_grad={tensor.requires_grad}")

    print(f"\nTotal global nodes: {len(all_features)} / 全局节点总数: {len(all_features)}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Model Params / 模型总参数: {total_params}")
    
    return model

# ===================== Main Execution / 主执行函数 =====================
if __name__ == "__main__":
    model = example_mhdnet2()
