# -*- coding: utf-8 -*-
"""
Hypergraph Neural Network (HDNet/MHDNet)
超图神经网络核心框架 (Hypergraph Neural Network Core Framework)

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

Author: Your Name
Date: 2026
Version: 2.0
License: MIT
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, TypeVar, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings
import re
from collections import namedtuple

# 全局配置 (Global Configuration)
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# 类型定义 (Type Definitions)
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# 结构化拓扑属性 (Structured Topology Attributes)
TopoAttr = namedtuple("TopoAttr", ["role", "sort", "ext"], defaults=[None])

# 全局函数注册表 (Global Function Registry)
# 节点头函数（边输入前的张量预处理）
# Node head functions (tensor preprocessing before edge input)
MHD_NODE_HEAD_FUNCS: Dict[str, Callable[..., Any]] = {
    "share": lambda tensor: tensor.clone(memory_format=torch.contiguous_format),
}

# 节点尾函数（边输出后的张量聚合）
# Node tail functions (tensor aggregation after edge output)
MHD_NODE_TAIL_FUNCS: Dict[str, Callable[..., Any]] = {
    "sum": lambda tensors: sum(tensors),
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],
    "mul": lambda tensors: torch.prod(torch.stack(tensors), dim=0)
}

# 工具函数 (Utility Functions)
def MHD_sort_nodes_by_topo_attr(attrs: List[TopoAttr]) -> List[Tuple[int, int]]:
    """按拓扑属性的sort字段排序节点 (Sort nodes by sort field of topology attributes)"""
    indexed_nodes = list(enumerate(attrs))
    return sorted(indexed_nodes, key=lambda p: p[1].sort if p[1] is not None else 0)

def MHD_flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """展平张量（保留批次维度） (Flatten tensor (keep batch dimension))"""
    if x.dim() > 2:
        x_flat = x.reshape(x.shape[0], -1)
    else:
        x_flat = x
    return x_flat

def extract_operation_name(op: Union[str, nn.Module]) -> str:
    """提取操作名称（移除路径/参数） (Extract operation name (remove path/parameters))"""
    if isinstance(op, nn.Module):
        op_name = op.__class__.__name__
    elif isinstance(op, str):
        op_name = re.sub(r'\(.*\)', '', op).strip()
    else:
        op_name = str(op)

    op_name = op_name.replace("torch.nn.modules.", "").replace("torch.nn.", "")
    return op_name

def parse_topo_value(value: Union[dict, TopoAttr]) -> TopoAttr:
    """解析拓扑值为结构化属性 (Parse topology value to structured attributes)"""
    if isinstance(value, dict):
        return TopoAttr(
            role=value.get("role", 0),
            sort=value.get("sort", 0),
            ext=value.get("ext", {})
        )
    elif isinstance(value, TopoAttr):
        return value
    else:
        raise ValueError(f"仅支持dict/TopoAttr格式 (Only dict/TopoAttr format supported)，Got {type(value)}")

# 核心操作函数 (Core Operation Functions)
def MHD_concat(tensors: List[torch.Tensor], attrs: List[TopoAttr]) -> torch.Tensor:
    """按拓扑属性排序后拼接张量 (Concatenate tensors after sorting by topology attributes)"""
    sorted_pairs = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs]
    return torch.cat(sorted_tensors, dim=1)

def MHD_matmul(tensors: List[torch.Tensor], attrs: List[TopoAttr]) -> torch.Tensor:
    """按拓扑属性排序后执行矩阵乘法 (Perform matrix multiplication after sorting by topology attributes)"""
    sorted_pairs = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_tensors = [tensors[i] for i, _ in sorted_pairs]
    if len(sorted_tensors) != 2:
        raise ValueError(f"Matmul需要2个输入张量 (Matmul requires 2 input tensors)，Got {len(sorted_tensors)}")
    return torch.matmul(*sorted_tensors)

MHD_EDGE_IN_FUNCS: Dict[str, Callable[..., Any]] = {
    "concat": MHD_concat,
    "matmul": MHD_matmul,
}

def MHD_split(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """按拓扑属性分割张量 (Split tensor by topology attributes)"""
    sorted_nodes = MHD_sort_nodes_by_topo_attr(attrs)
    sorted_original_indices = [p[0] for p in sorted_nodes]
    sorted_channel_sizes = [node_channels[i] for i in sorted_original_indices]

    split_tensors = torch.split(x, sorted_channel_sizes, dim=1)
    tensor_map = {idx: t for idx, t in zip(sorted_original_indices, split_tensors)}
    return [tensor_map[i] for i in range(len(attrs))]

def MHD_svd(x: torch.Tensor, attrs: List[TopoAttr], node_channels: List[int]) -> List[torch.Tensor]:
    """按拓扑属性执行SVD分解 (Perform SVD decomposition by topology attributes)"""
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
    """按拓扑属性执行LU分解 (Perform LU decomposition by topology attributes)"""
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

# 字符串操作包装类 (String Operation Wrapper Class)
class StringOperation(nn.Module):
    """字符串定义的张量操作包装类 (Wrapper class for tensor operations defined by string)"""
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

# 核心数据结构 (Core Data Structures)
@dataclass
class MHD_Node:
    """超图节点类 (Hypergraph Node Class)"""
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
    """超图边类 (Hypergraph Edge Class)"""
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
    """超图拓扑类 (Hypergraph Topology Class)"""
    value: List[List[Union[TopoAttr, dict]]]

    def __post_init__(self):
        self.value = [
            [parse_topo_value(val) for val in row]
            for row in self.value
        ]

    def get_topo_attr(self, edge_id: int, node_id: int) -> TopoAttr:
        """获取指定边和节点的拓扑属性 (Get topology attribute of specified edge and node)"""
        return self.value[edge_id][node_id]

    def to_tensor(self) -> torch.Tensor:
        """转换为张量形式 (Convert to tensor form)"""
        tensor_data = []
        for row in self.value:
            tensor_row = [attr.role for attr in row]
            tensor_data.append(tensor_row)
        return torch.tensor(tensor_data, dtype=torch.int64)

# 拓扑排序 (Topological Sorting)
def MHD_topological_sort(nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo) -> List[int]:
    """超图节点拓扑排序 (Hypergraph node topological sorting)"""
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
        raise ValueError(f"超图检测到环 (Hypergraph detected cycle)！环中节点 (Nodes in cycle): {all_node_ids - set(sorted_node_ids)}")
    return sorted_node_ids

# 动态网络 (Dynamic Network)
class DNet(nn.Module):
    """超边操作动态网络 (Dynamic network for hyperedge operations)"""
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
                raise ValueError(f"不支持的操作类型 (Unsupported operation type): {type(op)}，仅支持nn.Module/string (only nn.Module/string supported)")

        self.filter = nn.Sequential(*seq_ops)
        self.original_operations = operations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)

# 超图动态网络 (Hypergraph Dynamic Network)
class HDNet(nn.Module):
    """超图动态网络（单子网） (Hypergraph dynamic network (single subnet))"""
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo):
        super().__init__()
        self.node_id2obj = {node.id: node for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        self.topo = topo

        self._validate_topo()
        self.sorted_node_ids = MHD_topological_sort(nodes, edges, topo)
        print(f"✅ 拓扑排序完成 (Topological sort completed): {[self.node_id2obj[nid].name for nid in self.sorted_node_ids]}")

        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.value)

        # 节点值初始化 (Node value initialization)
        self.node_values = {node.id: node.value for node in nodes}

    @property
    def node_name2id(self):
        """节点名称到ID的映射 (Node name to ID mapping)"""
        return {v.name: k for k, v in self.node_id2obj.items()}

    def _validate_topo(self) -> None:
        """验证拓扑矩阵维度 (Validate topology matrix dimensions)"""
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)

        if len(self.topo.value) != num_edges:
            raise ValueError(
                f"拓扑矩阵边维度不匹配 (Topology matrix edge dimension mismatch): 预期 (expected){num_edges}，实际 (actual){len(self.topo.value)}"
            )
        for edge_row in self.topo.value:
            if len(edge_row) != num_nodes:
                raise ValueError(
                    f"拓扑矩阵节点维度不匹配 (Topology matrix node dimension mismatch): 预期 (expected){num_nodes}，实际 (actual){len(edge_row)}"
                )

        for edge_id, edge_row in enumerate(self.topo.value):
            for node_id, attr in enumerate(edge_row):
                if not isinstance(attr, TopoAttr):
                    raise ValueError(
                        f"拓扑矩阵元素 (Topology matrix element)({edge_id}, {node_id})必须为TopoAttr (must be TopoAttr)，Got {type(attr)}"
                    )

    def get_node_by_name(self, name: str) -> MHD_Node:
        """按名称获取节点 (Get node by name)"""
        try:
            node_id = self.node_name2id[name]
            return self.node_id2obj[node_id]
        except KeyError:
            raise ValueError(f"节点名称不存在 (Node name does not exist): {name}")

    def forward(self) -> Dict[str, Tensor]:
        """拓扑驱动前向传播 (Topology-driven forward propagation)"""
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

                # 获取头节点 (Get head nodes)
                head_mask = [attr.role < 0 for attr in edge_row]
                head_node_ids = [i for i, val in enumerate(head_mask) if val]
                head_topo_attrs = [edge_row[nid] for nid in head_node_ids]

                # 处理头节点张量 (Process head node tensors)
                head_tensors = []
                for node_id in head_node_ids:
                    node = self.node_id2obj[node_id]
                    head_func_name = node.func.get("head", "share")
                    head_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](self.node_values[node_id])
                    head_tensors.append(head_tensor)

                # 边输入处理 (Edge input processing)
                edge_in_func_name = edge.func.get("in", "concat")
                edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](head_tensors, head_topo_attrs)

                # 边操作前向传播 (Edge operation forward propagation)
                edge_output = edge_net(edge_input)

                # 获取尾节点 (Get tail nodes)
                tail_mask = [attr.role > 0 for attr in edge_row]
                tail_node_ids = [i for i, val in enumerate(tail_mask) if val]
                tail_topo_attrs = [edge_row[nid] for nid in tail_node_ids]
                tail_node_channels = [self.node_id2obj[node_id].value.shape[1] for node_id in tail_node_ids]

                # 边输出处理 (Edge output processing)
                edge_out_func_name = edge.func.get("out", "split")
                tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                    edge_output, tail_topo_attrs, tail_node_channels
                )

                # 节点值更新 (Node value update)
                for node_id, tensor in zip(tail_node_ids, tail_tensors):
                    node = self.node_id2obj[node_id]
                    tail_func_name = node.func.get("tail", "sum")
                    if node_id in self.node_values:
                        agg_tensor = MHD_NODE_TAIL_FUNCS[tail_func_name](
                            [self.node_values[node_id], tensor]
                        )
                        self.node_values[node_id] = agg_tensor
                    else:
                        self.node_values[node_id] = tensor

        return {
            self.node_id2obj[node_id].name: tensor
            for node_id, tensor in self.node_values.items()
        }

# 多超图动态网络 (Multi-Hypergraph Dynamic Network)
class MHDNet(HDNet):
    """多超图动态网络（全局超图） (Multi-hypergraph dynamic network (global hypergraph))"""
    def __init__(
        self,
        hdnet_list: List[Tuple[str, HDNet]],
        node_group: Tuple[Set[str], ...],
    ):
        # 1. 构建全局节点/边/拓扑 (Build global nodes/edges/topology)
        global_nodes, global_edges, global_topo_data = self._build_global_hypergraph(hdnet_list, node_group)
        
        # 2. 初始化父类HDNet (Initialize parent class HDNet)
        global_topo = MHD_Topo(value=global_topo_data)
        super().__init__(nodes=global_nodes, edges=global_edges, topo=global_topo)
        
        # 3. 保存原始映射 (Save original mappings)
        self.hdnet_list = hdnet_list
        self.node_group = node_group

    def _build_global_hypergraph(self, hdnet_list: List[Tuple[str, HDNet]], node_group: Tuple[Set[str], ...]) -> Tuple[Set[MHD_Node], Set[MHD_Edge], List[List[TopoAttr]]]:
        """构建完整的全局超图（包含所有节点和边） (Build complete global hypergraph with all nodes and edges)"""
        # ===================== 步骤1：预处理所有子图节点/边 =====================
        # 子节点映射：key=suffix::name, value=(suffix, sub_node_id, sub_node)
        sub_node_map = {}
        # 子边映射：key=suffix::name, value=(suffix, sub_edge_id, sub_edge)
        sub_edge_map = {}
        # 所有子节点名称集合（用于后续去重）
        all_sub_node_names = set()
        
        for suffix, hdnet in hdnet_list:
            # 处理节点
            for sub_node_id, sub_node in hdnet.node_id2obj.items():
                global_node_name = f"{suffix}::{sub_node.name}"
                sub_node_map[global_node_name] = (suffix, sub_node_id, sub_node)
                all_sub_node_names.add(global_node_name)
            
            # 处理边（所有子图边都作为独立全局边）
            for sub_edge_id, sub_edge in hdnet.edge_id2obj.items():
                global_edge_name = f"{suffix}::{sub_edge.name}"
                sub_edge_map[global_edge_name] = (suffix, sub_edge_id, sub_edge)

        # ===================== 步骤2：处理节点合并 =====================
        node_id_counter = 0
        merged_node_map = {}  # key=全局节点名, value=MHD_Node
        sub2global_node = {}  # key=子节点名(suffix::name), value=全局节点名
        
        # 第一步：先处理需要合并的节点组
        merged_node_names = set()
        for node_group in node_group:
            # 按hdnet_list顺序排序节点组
            sorted_node_names = sorted(
                node_group,
                key=lambda x: next((i for i, (suffix, _) in enumerate(hdnet_list) if x.startswith(f"{suffix}::")), 999)
            )
            merged_name = "-".join(sorted_node_names)
            merged_node_names.update(sorted_node_names)  # 记录被合并的子节点
            
            # 获取第一个节点作为基础值
            first_node_name = sorted_node_names[0]
            _, _, base_node = sub_node_map[first_node_name]
            
            # 创建合并节点
            merged_node = MHD_Node(
                id=node_id_counter,
                name=merged_name,
                value=base_node.value,
                func=base_node.func
            )
            merged_node_map[merged_name] = merged_node
            node_id_counter += 1
            
            # 建立子节点到合并节点的映射
            for node_name in sorted_node_names:
                sub2global_node[node_name] = merged_name

        # 第二步：处理未被合并的独立节点
        unmerged_node_names = all_sub_node_names - merged_node_names
        for node_name in sorted(unmerged_node_names):
            _, _, sub_node = sub_node_map[node_name]
            
            # 创建独立全局节点（名称保持suffix::name）
            unmerged_node = MHD_Node(
                id=node_id_counter,
                name=node_name,
                value=sub_node.value,
                func=sub_node.func
            )
            merged_node_map[node_name] = unmerged_node
            sub2global_node[node_name] = node_name  # 映射到自身
            node_id_counter += 1

        # ===================== 步骤3：构建全局边（所有子图边独立存在） =====================
        edge_id_counter = 0
        merged_edge_map = {}
        
        for global_edge_name, (suffix, sub_edge_id, sub_edge) in sub_edge_map.items():
            # 创建独立全局边（名称保持suffix::edge_name）
            merged_edge = MHD_Edge(
                id=edge_id_counter,
                name=global_edge_name,
                value=sub_edge.value,
                func=sub_edge.func
            )
            merged_edge_map[global_edge_name] = merged_edge
            edge_id_counter += 1

        # ===================== 步骤4：构建全局拓扑矩阵 =====================
        global_topo_data = []
        global_node_name2id = {name: node.id for name, node in merged_node_map.items()}
        
        for global_edge_name, (suffix, sub_edge_id, sub_edge) in sub_edge_map.items():
            # 获取对应子图的拓扑行
            hdnet = next(h for s, h in hdnet_list if s == suffix)
            sub_topo_row = hdnet.topo.value[sub_edge_id]
            
            # 初始化全局拓扑行（长度=全局节点数）
            global_topo_row = [TopoAttr(role=0, sort=0, ext={}) for _ in range(len(merged_node_map))]
            
            # 映射子拓扑到全局拓扑
            for sub_node_id, sub_attr in enumerate(sub_topo_row):
                if sub_attr.role == 0:
                    continue  # 无拓扑关系的节点跳过
                
                # 获取子节点的全局名称
                sub_node_name = hdnet.node_id2obj[sub_node_id].name
                full_sub_node_name = f"{suffix}::{sub_node_name}"
                
                # 找到对应的全局节点名称和ID
                if full_sub_node_name in sub2global_node:
                    global_node_name = sub2global_node[full_sub_node_name]
                    global_node_id = global_node_name2id[global_node_name]
                    
                    # 转移拓扑属性到全局节点
                    global_topo_row[global_node_id] = sub_attr
            
            global_topo_data.append(global_topo_row)

        # ===================== 转换为集合并返回 =====================
        global_nodes = set(merged_node_map.values())
        global_edges = set(merged_edge_map.values())
        
        return global_nodes, global_edges, global_topo_data


def generate_mermaid(hdnet: HDNet) -> str:
    """生成带颜色区分的极简拓扑可视化Mermaid代码"""
    mermaid = [
        "graph TD",
        "",
        "    %% 样式定义：节点和边区分颜色",
        "    classDef nodeStyle fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
        "    classDef edgeStyle fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
        "",
    ]

    # 1. 先给所有节点添加样式
    for node_id, node in hdnet.node_id2obj.items():
        mermaid.append(f"    {node.name}:::nodeStyle")
    
    # 2. 遍历边，添加边样式和连接关系
    for edge_id, edge in hdnet.edge_id2obj.items():
        edge_name = edge.name
        edge_row = hdnet.topo.value[edge_id]
        
        # 添加边样式
        mermaid.append(f"    {edge_name}:::edgeStyle")
        
        # 获取头/尾节点名称
        head_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role < 0]
        tail_node_ids = [nid for nid, attr in enumerate(edge_row) if attr.role > 0]
        head_node_names = [hdnet.node_id2obj[nid].name for nid in head_node_ids]
        tail_node_names = [hdnet.node_id2obj[nid].name for nid in tail_node_ids]
        
        # 生成连接关系
        for head_node in head_node_names:
            mermaid.append(f"    {head_node} --> {edge_name}")
        for tail_node in tail_node_names:
            mermaid.append(f"    {edge_name} --> {tail_node}")
        
        mermaid.append("")

    mermaid_code = "\n".join(mermaid)
    print(mermaid_code)
    return mermaid_code

# 示例用法 (Example Usage)
def example_mhdnet2():
    """MHDNet示例（自定义3子图拓扑+MUL聚合） (MHDNet example with custom 3-subgraph topology + MUL aggregation)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"✅ 使用设备 (Using device): {device}")

    # ===================== 子HDNet1 (A1→B1、A1→D1，D1用MUL聚合) =====================
    # Sub HDNet1 (A1→B1, A1→D1, D1 uses MUL aggregation)
    nodes_net1 = {
        MHD_Node(
            id=0, 
            name="A1", 
            value=torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype),  # 3通道 (3 channels)
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=1, 
            name="B1", 
            value=torch.randn(1, 2, 8, 8, 8, device=device, dtype=dtype),  # 2通道 (2 channels)
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=2, 
            name="D1", 
            value=torch.randn(1, 4, 8, 8, 8, device=device, dtype=dtype),  # 4通道 (4 channels)
            func={"head": "share", "tail": "mul"}
        ),
    }
    # 超边1：A1→B1（纯Module列表） (Hyperedge 1: A1→B1 (pure Module list))
    edge1_net1 = [
        nn.Conv3d(3, 2, kernel_size=3, padding=1, bias=False).to(device),
        nn.BatchNorm3d(2).to(device),
        nn.ReLU(inplace=True)
    ]
    # 超边2：A1→D1（纯Module列表） (Hyperedge 2: A1→D1 (pure Module list))
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
    # 拓扑矩阵：2条超边 × 3个节点 (Topology matrix: 2 hyperedges × 3 nodes)
    topo_net1 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}, {"role": 0, "sort": 0}],  # 超边1：A1(头)→B1(尾)
        [{"role": -1, "sort": 1}, {"role": 0, "sort": 0}, {"role": 1, "sort": 1}]   # 超边2：A1(头)→D1(尾)
    ])
    hdnet1 = HDNet(nodes=nodes_net1, edges=edges_net1, topo=topo_net1)

    # ===================== 子HDNet2 (A2+B2拼接→C2) =====================
    # Sub HDNet2 (A2+B2 concat→C2)
    nodes_net2 = {
        MHD_Node(
            id=0, 
            name="A2", 
            value=torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype),  # 与A1同维度 (Same dimension as A1)
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=1, 
            name="B2", 
            value=torch.randn(1, 2, 8, 8, 8, device=device, dtype=dtype),  # 与B1同维度 (Same dimension as B1)
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=2, 
            name="C2", 
            value=torch.randn(1, 5, 8, 8, 8, device=device, dtype=dtype),  # 5通道 (5 channels)
            func={"head": "share", "tail": "sum"}
        ),
    }
    # 超边：A2+B2拼接→C2 (Hyperedge: A2+B2 concat→C2)
    edge1_net2 = [
        nn.Conv3d(5, 5, kernel_size=3, padding=1, groups=5, bias=False).to(device),  # 分组卷积 (Group convolution)
        nn.GELU(),
        nn.Conv3d(5, 5, kernel_size=1, padding=0, bias=True).to(device)             # 1x1调整 (1x1 adjustment)
    ]
    edges_net2 = {
        MHD_Edge(
            id=0, 
            name="e1_A2B2_to_C2", 
            value=edge1_net2,
            func={"in": "concat", "out": "split"}
        )
    }
    # 拓扑矩阵：1条超边 × 3个节点 (Topology matrix: 1 hyperedge × 3 nodes)
    topo_net2 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": -1, "sort": 2}, {"role": 1, "sort": 1}]
    ])
    hdnet2 = HDNet(nodes=nodes_net2, edges=edges_net2, topo=topo_net2)

    # ===================== 子HDNet3 (C3→D3，D3用MUL聚合) =====================
    # Sub HDNet3 (C3→D3, D3 uses MUL aggregation)
    nodes_net3 = {
        MHD_Node(
            id=0, 
            name="C3", 
            value=torch.randn(1, 5, 8, 8, 8, device=device, dtype=dtype),  # 与C2同维度 (Same dimension as C2)
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=1, 
            name="D3", 
            value=torch.randn(1, 4, 8, 8, 8, device=device, dtype=dtype),  # 与D1同维度 (Same dimension as D1)
            func={"head": "share", "tail": "mul"}
        ),
    }
    # 超边：C3→D3（Module+字符串操作混合） (Hyperedge: C3→D3 (mixed Module+string operations))
    edge1_net3 = [
        nn.Conv3d(5, 4, kernel_size=3, padding=1, bias=False).to(device),
        nn.Softplus(),
        '__mul__(0.5)'  # 字符串操作 (String operation)
    ]
    edges_net3 = {
        MHD_Edge(
            id=0, 
            name="e1_C3_to_D3", 
            value=edge1_net3,
            func={"in": "concat", "out": "split"}
        )
    }
    # 拓扑矩阵：1条超边 × 2个节点 (Topology matrix: 1 hyperedge × 2 nodes)
    topo_net3 = MHD_Topo(value=[
        [{"role": -1, "sort": 1}, {"role": 1, "sort": 1}]
    ])
    hdnet3 = HDNet(nodes=nodes_net3, edges=edges_net3, topo=topo_net3)

    # ===================== 构建全局HDNet =====================
    # Build global HDNet
    # 1. 定义HDNet列表 (Define HDNet list)
    hdnet_list = [
        ("net1", hdnet1),
        ("net2", hdnet2),
        ("net3test", hdnet3)
    ]

    # 2. 定义NodeF（节点定义组） (Define node_group (node definition groups))
    node_group = (
        {"net1::A1", "net2::A2"},          # 合并为 "net1::A1-net2::A2" (Merge to "net1::A1-net2::A2")
        {"net1::B1", "net2::B2"},          # 合并为 "net1::B1-net2::B2" (Merge to "net1::B1-net2::B2")
        {"net2::C2", "net3test::C3"},      # 合并为 "net2::C2-net3test::C3" (Merge to "net2::C2-net3test::C3")
        {"net1::D1", "net3test::D3"},      # 合并为 "net1::D1-net3test::D3" (Merge to "net1::D1-net3test::D3")
    )

    # 3. 创建MHDNet（全局超图） (Create MHDNet (global hypergraph))
    mhdnet = MHDNet(
        hdnet_list=hdnet_list,
        node_group=node_group
    )

    # ===================== 原生方式加载数据 =====================
    # Load data with native operations
    # 1. 准备输入数据 (Prepare input data)
    input_tensor = torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype)
    print(f"\n✅ 输入张量形状 (Input Tensor Shape): {input_tensor.shape}")

    # 2. 原生方式访问并修改节点 (Directly access and modify node with native operations)
    # 按名称查找目标节点 (Find target node by name)
    target_node_name = "net1::A1-net2::A2"
    input_node = mhdnet.get_node_by_name(target_node_name)
    
    # 修改节点名称和值 (Modify node name and value)
    input_node.name = "input"  # 重命名为input (Rename to input)
    input_node.value = input_tensor  # 更新值 (Update value)
    print(f"\n✅ 更新后的节点 (Updated Node) '{input_node.name}':")
    print(f"   - 新值形状 (New Value Shape): {input_node.value.shape}")
    print(f"   - 新值均值 (New Value Mean): {input_node.value.mean().item():.4f}")

    # ===================== 生成最新可视化 =====================
    # Generate latest visualization
    print("MHD node_group - 最新拓扑可视化 (Latest Topology Visualization):")
    print("="*80)
    generate_mermaid(mhdnet)
    print("="*80 + "\n")

    # ===================== 运行网络 =====================
    # Run network
    print("🚀 执行MHDNet前向传播 (Running MHDNet Forward Pass)...")
    all_features = mhdnet.forward()

    # 打印关键结果 (Print key results)
    print("\n✅ 自定义拓扑前向传播完成 (Custom topology forward propagation completed)！")
    print("\n=== 全局节点特征详情 (Global Node Feature Details) ===")
    for node_name in ["input", "net1::B1-net2::B2", "net2::C2-net3test::C3", "net1::D1-net3test::D3"]:
        if node_name in all_features:
            tensor = all_features[node_name]
            print(f"  - 全局节点 (Global Node) {node_name}: 形状 (shape)={tensor.shape}, 设备 (device)={tensor.device}, 均值 (mean)={tensor.mean().item():.4f}")
    
    print(f"\n全局节点总数 (Total global nodes): {len(all_features)}")
    total_params = sum(p.numel() for p in mhdnet.parameters())
    print(f"模型总参数 (Total model parameters): {total_params:,}")
    
    return mhdnet

# 梯度验证函数 (Gradient Verification Function)
def verify_gradient(model):
    """验证梯度反传 (Verify gradient backpropagation)"""
    all_features = model.forward()
    output_tensor = all_features["net1::D1-net3test::D3"]
    loss = output_tensor.sum()
    
    model.zero_grad()
    loss.backward()
    
    has_gradient = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.sum() != 0:
            has_gradient = True
            print(f"✅ 参数 (Parameter) {name} 梯度正常 (gradient normal): {param.grad.sum().item():.4f}")
    
    if has_gradient:
        print("\n✅ 梯度反传验证通过 (Gradient backpropagation verification passed)！")
    else:
        print("\n❌ 梯度反传验证失败 (Gradient backpropagation verification failed)！")

# 主执行 (Main Execution)
if __name__ == "__main__":
    model = example_mhdnet2()
    verify_gradient(model)
