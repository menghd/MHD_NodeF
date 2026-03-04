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

# 全局配置
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# 类型定义
Tensor = TypeVar('Tensor', bound=torch.Tensor)

# ===================== 核心数据结构 =====================
@dataclass
class MHD_Node:
    """超图节点类"""
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

    # 节点对象获取方法（统一命名）
    @staticmethod
    def id2obj(node_id: int, node_list: Set['MHD_Node']) -> Optional['MHD_Node']:
        """通过ID获取节点对象"""
        for node in node_list:
            if node.id == node_id:
                return node
        return None

    @staticmethod
    def name2obj(name: str, node_list: Set['MHD_Node']) -> Optional['MHD_Node']:
        """通过名称获取节点对象"""
        for node in node_list:
            if node.name == name:
                return node
        return None

    # 节点函数应用方法
    def apply_head_func(self, tensor: torch.Tensor) -> torch.Tensor:
        """应用节点头函数"""
        head_funcs = {
            "share": lambda t: t.clone(memory_format=torch.contiguous_format),
        }
        func_name = self.func.get("head", "share")
        return head_funcs[func_name](tensor)

    def apply_tail_func(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """应用节点尾函数"""
        tail_funcs = {
            "sum": lambda ts: sum(ts),
            "avg": lambda ts: torch.stack(ts).mean(dim=0),
            "max": lambda ts: torch.stack(ts).max(dim=0)[0],
            "min": lambda ts: torch.stack(ts).min(dim=0)[0],
            "mul": lambda ts: torch.prod(torch.stack(ts), dim=0)
        }
        func_name = self.func.get("tail", "sum")
        return tail_funcs[func_name](tensors)

@dataclass
class MHD_Edge:
    """超图边类"""
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

    # 边对象获取方法（统一命名）
    @staticmethod
    def id2obj(edge_id: int, edge_list: Set['MHD_Edge']) -> Optional['MHD_Edge']:
        """通过ID获取边对象"""
        for edge in edge_list:
            if edge.id == edge_id:
                return edge
        return None

    @staticmethod
    def name2obj(name: str, edge_list: Set['MHD_Edge']) -> Optional['MHD_Edge']:
        """通过名称获取边对象"""
        for edge in edge_list:
            if edge.name == name:
                return edge
        return None

    # 边函数应用方法
    def apply_in_func(self, tensors: List[torch.Tensor], sorted_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """应用边输入函数"""
        def concat(ts, sp):
            sorted_ts = [ts[i] for i, _ in sp if i < len(ts)]
            return torch.cat(sorted_ts, dim=1)

        def matmul(ts, sp):
            sorted_ts = [ts[i] for i, _ in sp if i < len(ts)]
            if len(sorted_ts) != 2:
                raise ValueError(f"Matmul需要2个输入张量，实际输入 {len(sorted_ts)}")
            return torch.matmul(*sorted_ts)

        in_funcs = {
            "concat": concat,
            "matmul": matmul,
        }
        func_name = self.func.get("in", "concat")
        return in_funcs[func_name](tensors, sorted_pairs)

    def apply_out_func(self, x: torch.Tensor, sorted_pairs: List[Tuple[int, int]], node_channels: List[int]) -> List[torch.Tensor]:
        """应用边输出函数"""
        def split(x, sp, nc):
            sorted_nodes = sp
            sorted_indices = [p[0] for p in sorted_nodes if p[0] < len(nc)]
            sorted_sizes = [nc[i] for i in sorted_indices]
            split_ts = torch.split(x, sorted_sizes, dim=1)
            tensor_map = {idx: t for idx, t in zip(sorted_indices, split_ts)}
            
            result = []
            for i in range(len(nc)):
                result.append(tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype)))
            return result

        def svd(x, sp, nc):
            x_flat = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
            U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
            sorted_nodes = sp
            components = [U, S, Vh]
            sorted_ts = []

            for i, (orig_idx, _) in enumerate(sorted_nodes):
                if orig_idx < len(nc):
                    comp_idx = i % len(components)
                    tensor = components[comp_idx]
                    if tensor.shape[1] != nc[orig_idx]:
                        tensor = nn.functional.adaptive_avg_pool1d(tensor.unsqueeze(1), nc[orig_idx]).squeeze(1)
                    sorted_ts.append((orig_idx, tensor))

            tensor_map = {idx: t for idx, t in sorted_ts}
            result = []
            for i in range(len(nc)):
                result.append(tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype)))
            return result

        def lu(x, sp, nc):
            x_flat = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
            P, L, U = torch.linalg.lu(x_flat)
            sorted_nodes = sp
            components = [L, U, P]
            sorted_ts = []

            for i, (orig_idx, _) in enumerate(sorted_nodes):
                if orig_idx < len(nc):
                    comp_idx = i % len(components)
                    tensor = components[comp_idx]
                    if tensor.shape[1] != nc[orig_idx]:
                        tensor = nn.functional.adaptive_avg_pool1d(tensor.unsqueeze(1), nc[orig_idx]).squeeze(1)
                    sorted_ts.append((orig_idx, tensor))

            tensor_map = {idx: t for idx, t in sorted_ts}
            result = []
            for i in range(len(nc)):
                result.append(tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype)))
            return result

        out_funcs = {
            "split": split,
            "svd": svd,
            "lu": lu,
        }
        func_name = self.func.get("out", "split")
        return out_funcs[func_name](x, sorted_pairs, node_channels)

@dataclass
class MHD_Topo:
    """超图拓扑类"""
    type: str
    value: torch.Tensor
    
    # 拓扑对象获取方法（统一命名）
    @staticmethod
    def type2obj(type_name: str, topo_list: Set['MHD_Topo']) -> Optional['MHD_Topo']:
        """通过类型获取拓扑对象"""
        for topo in topo_list:
            if topo.type == type_name:
                return topo
        return None

    def get_topo_value(self, edge_id: int, node_id: int) -> int:
        """获取指定边和节点的拓扑值"""
        if edge_id < self.value.shape[0] and node_id < self.value.shape[1]:
            return int(self.value[edge_id, node_id].item())
        return 0
    
    def to_list(self) -> List[List[int]]:
        """转换为列表形式"""
        return self.value.tolist()
    
    def __hash__(self):
        return hash((self.type, tuple(self.value.flatten().tolist())))
    
    def __eq__(self, other):
        if not isinstance(other, MHD_Topo):
            return False
        return self.type == other.type and torch.equal(self.value, other.value)

    def validate_topo(self, num_edges: int, num_nodes: int) -> None:
        """验证拓扑矩阵维度"""
        if self.value.shape[0] != num_edges:
            raise ValueError(
                f"{self.type}拓扑边维度不匹配: 预期{num_edges}，实际{self.value.shape[0]}"
            )
        if self.value.shape[1] != num_nodes:
            raise ValueError(
                f"{self.type}拓扑节点维度不匹配: 预期{num_nodes}，实际{self.value.shape[1]}"
            )

# ===================== 动态网络 =====================
class DNet(nn.Module):
    """超边操作动态网络"""
    def __init__(self, operations: List[Union[str, nn.Module]], device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        seq_ops = []

        for op in operations:
            if isinstance(op, nn.Module):
                op = op.to(self.device)
                seq_ops.append(op)
            elif isinstance(op, str):
                seq_ops.append(self.StringOperation(op))
            else:
                raise ValueError(f"不支持的操作类型: {type(op)}，仅支持nn.Module/string")

        self.filter = nn.Sequential(*seq_ops)
        self.original_operations = operations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = x.to(self.device)
        return self.filter(x)

    class StringOperation(nn.Module):
        """字符串操作包装类"""
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

# ===================== 超图动态网络 =====================
class HDNet(nn.Module):
    """超图动态网络（单子网）"""
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topos: Set[MHD_Topo], device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        
        # 仅保存原始集合，不再维护ID/名称映射（直接调用实体类方法）
        self.nodes = nodes
        self.edges = edges
        
        # 转换拓扑为tensor类型并保存
        self.topos = set()
        for topo in topos:
            if isinstance(topo.value, list):
                tensor_value = torch.tensor(topo.value, dtype=torch.int64, device=self.device)
            else:
                tensor_value = topo.value.to(self.device)
            self.topos.add(MHD_Topo(type=topo.type, value=tensor_value))

        # 验证拓扑维度
        self.validate_all_topo()
        # 拓扑排序
        self.sorted_node_ids = self.topological_sort()
        print(f"✅ 拓扑排序完成: {[MHD_Node.id2obj(nid, self.nodes).name for nid in self.sorted_node_ids]}")

        # 初始化边操作网络
        self.edge_values = nn.ModuleDict()
        for edge in edges:
            self.edge_values[edge.name] = DNet(edge.value, device=self.device)

        # 节点值初始化（注册为可训练参数）
        self.node_values = nn.ParameterDict()
        for node in nodes:
            node_value = node.value.to(self.device)
            param = nn.Parameter(node_value, requires_grad=True)
            self.node_values[str(node.id)] = param

        # 预缓存所有边的拓扑排序结果
        self.edge_sorted_pairs = {}
        for edge in edges:
            self.edge_sorted_pairs[edge.id] = self.sort_nodes_by_topo_value(edge.id)

    def validate_all_topo(self) -> None:
        """验证所有拓扑维度"""
        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        for topo in self.topos:
            topo.validate_topo(num_edges, num_nodes)

    def sort_nodes_by_topo_value(self, edge_id: int = 0) -> List[Tuple[int, int]]:
        """按sort类型的拓扑值排序节点"""
        sort_topo = MHD_Topo.type2obj("sort", self.topos)
        
        if sort_topo is None:
            role_topo = next(iter(self.topos)) if self.topos else None
            if role_topo:
                sort_values = torch.zeros_like(role_topo.value)
            else:
                sort_values = torch.tensor([])
        else:
            sort_values = sort_topo.value
        
        if edge_id < sort_values.shape[0]:
            indexed_nodes = list(enumerate(sort_values[edge_id].tolist()))
        else:
            indexed_nodes = []
        
        return sorted(indexed_nodes, key=lambda p: p[1] if p[1] is not None else 0)

    def topological_sort(self) -> List[int]:
        """超图节点拓扑排序"""
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_node_ids = {node.id for node in self.nodes}
        
        # 使用role类型的拓扑进行排序
        role_topo = MHD_Topo.type2obj("role", self.topos)
        
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

        # 初始化入度为0的节点
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

        # 检测环
        if len(sorted_node_ids) != len(all_node_ids):
            raise ValueError(
                f"超图检测到环！环中节点: {all_node_ids - set(sorted_node_ids)}"
            )
        return sorted_node_ids

    def validate_node_group_consistency(self, node_group: Set[str], sub_node_map: Dict[str, Tuple[str, int, MHD_Node]]) -> None:
        """验证节点组的维度/函数一致性"""
        if len(node_group) <= 1:
            return
        
        group_nodes = []
        for node_name in node_group:
            if node_name in sub_node_map:
                _, _, node = sub_node_map[node_name]
                group_nodes.append(node)
        
        # 校验维度一致性
        ref_shape = group_nodes[0].value.shape
        for node in group_nodes[1:]:
            if node.value.shape != ref_shape:
                raise ValueError(
                    f"节点维度不一致！节点 {node.name} 形状 {node.value.shape} 与参考节点 {group_nodes[0].name} 形状 {ref_shape} 不匹配"
                )
        
        # 校验函数一致性
        ref_func = group_nodes[0].func
        for node in group_nodes[1:]:
            if node.func != ref_func:
                raise ValueError(
                    f"节点函数配置不一致！节点 {node.name} func {node.func} 与参考节点 {group_nodes[0].name} func {ref_func} 不匹配"
                )

    def get_merged_node_value(self, node_group: Set[str], sub_node_map: Dict[str, Tuple[str, int, MHD_Node]]) -> torch.Tensor:
        """计算合并节点的初始值（组内节点均值）"""
        group_nodes = []
        for node_name in node_group:
            if node_name in sub_node_map:
                _, _, node = sub_node_map[node_name]
                group_nodes.append(node)
        
        if len(group_nodes) == 1:
            return group_nodes[0].value.clone()
        
        merged_tensor = torch.stack([n.value for n in group_nodes]).mean(dim=0)
        return merged_tensor

    def generate_mermaid(self) -> str:
        """生成Mermaid拓扑可视化代码"""
        mermaid = [
            "graph TD",
            "",
            "    classDef MHD_Node_Style fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
            "    classDef MHD_Edge_Style fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
            "",
        ]

        # 获取role类型拓扑
        role_topo = MHD_Topo.type2obj("role", self.topos)

        # 添加节点样式
        for node in self.nodes:
            mermaid.append(f"    {node.name}:::MHD_Node_Style")
        
        # 添加边样式和连接关系
        if role_topo:
            for edge in self.edges:
                edge_name = edge.name
                edge_id = edge.id
                if edge_id < role_topo.value.shape[0]:
                    edge_row = role_topo.value[edge_id]
                
                    mermaid.append(f"    {edge_name}:::MHD_Edge_Style")
                    
                    head_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() < 0]
                    tail_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() > 0]
                    
                    head_node_names = []
                    for nid in head_node_ids:
                        node = MHD_Node.id2obj(nid, self.nodes)
                        if node:
                            head_node_names.append(node.name)
                    
                    tail_node_names = []
                    for nid in tail_node_ids:
                        node = MHD_Node.id2obj(nid, self.nodes)
                        if node:
                            tail_node_names.append(node.name)
                    
                    for head_node in head_node_names:
                        mermaid.append(f"    {head_node} --> {edge_name}")
                    for tail_node in tail_node_names:
                        mermaid.append(f"    {edge_name} --> {tail_node}")
                    
                    mermaid.append("")

        mermaid_code = "\n".join(mermaid)
        print(mermaid_code)
        return mermaid_code

    def forward(self) -> Dict[str, Tensor]:
        """拓扑驱动前向传播"""
        if not self.node_values:
            return {}
            
        edge_affects_nodes = defaultdict(list)
        # 获取role类型拓扑
        role_topo = MHD_Topo.type2obj("role", self.topos)
        
        if role_topo is not None and role_topo.value.numel() > 0:
            for edge in self.edges:
                edge_id = edge.id
                if edge_id < role_topo.value.shape[0]:
                    edge_row = role_topo.value[edge_id]
                    tail_node_ids = [
                        nid for nid in range(role_topo.value.shape[1]) 
                        if edge_row[nid].item() > 0
                    ]
                    edge_affects_nodes[edge_id] = tail_node_ids

        # 按拓扑排序处理节点
        for target_node_id in self.sorted_node_ids:
            relevant_edges = [eid for eid, node_ids in edge_affects_nodes.items() if target_node_id in node_ids]

            for edge_id in relevant_edges:
                edge = MHD_Edge.id2obj(edge_id, self.edges)
                if not edge:
                    continue
                edge_value = self.edge_values[edge.name]
                
                # 获取头节点
                edge_row = role_topo.value[edge_id] if (role_topo and edge_id < role_topo.value.shape[0]) else []
                head_mask = [val.item() < 0 for val in edge_row] if edge_row.numel() > 0 else []
                head_node_ids = [i for i, val in enumerate(head_mask) if val]

                # 处理头节点张量
                head_tensors = []
                for node_id in head_node_ids:
                    node_key = str(node_id)
                    if node_key in self.node_values:
                        node = MHD_Node.id2obj(node_id, self.nodes)
                        if node:
                            head_tensor = node.apply_head_func(self.node_values[node_key])
                            head_tensors.append(head_tensor)

                if not head_tensors:
                    continue

                # 边输入处理
                sorted_pairs = self.edge_sorted_pairs.get(edge_id, [])
                edge_input = edge.apply_in_func(head_tensors, sorted_pairs)

                # 边操作前向传播
                edge_output = edge_value(edge_input)

                # 获取尾节点
                tail_mask = [val.item() > 0 for val in edge_row] if edge_row.numel() > 0 else []
                tail_node_ids = [i for i, val in enumerate(tail_mask) if val]
                tail_node_channels = []
                for node_id in tail_node_ids:
                    node = MHD_Node.id2obj(node_id, self.nodes)
                    node_key = str(node_id)
                    if node and node_key in self.node_values:
                        tail_node_channels.append(self.node_values[node_key].shape[1])
                    else:
                        tail_node_channels.append(0)

                if not tail_node_channels:
                    continue

                # 边输出处理
                tail_tensors = edge.apply_out_func(edge_output, sorted_pairs, tail_node_channels)

                # 节点值更新
                for idx, node_id in enumerate(tail_node_ids):
                    node_key = str(node_id)
                    if idx < len(tail_tensors) and node_key in self.node_values:
                        node = MHD_Node.id2obj(node_id, self.nodes)
                        if not node:
                            continue
                        tensor = tail_tensors[idx]
                        
                        # 确保张量形状匹配
                        if tensor.shape[1] != self.node_values[node_key].shape[1]:
                            tensor = nn.functional.adaptive_avg_pool1d(
                                tensor.unsqueeze(1), 
                                self.node_values[node_key].shape[1]
                            ).squeeze(1)
                        
                        # 聚合更新节点值
                        agg_tensor = node.apply_tail_func(
                            [self.node_values[node_key], tensor]
                        )
                        self.node_values[node_key] = nn.Parameter(agg_tensor, requires_grad=True)

        # 返回节点名称到张量的映射
        return {
            MHD_Node.id2obj(int(node_id), self.nodes).name: tensor
            for node_id, tensor in self.node_values.items()
            if MHD_Node.id2obj(int(node_id), self.nodes)
        }

# ===================== 多超图动态网络 =====================
class MHDNet(HDNet):
    """多超图动态网络（全局超图）"""
    def __init__(
        self,
        hdnet_list: List[Tuple[str, HDNet]],
        node_group: Tuple[Set[str], ...],
        device: torch.device = DEVICE
    ):
        # 显式ID映射表
        self.sub2global_node_id = {}
        self.sub2global_edge_id = {}
        self.global2sub_node_id = {}
        self.global2sub_edge_id = {}
        
        # 构建全局超图
        global_nodes, global_edges, global_topos = self._build_global_hypergraph(hdnet_list, node_group, device)
        
        # 初始化父类
        super().__init__(nodes=global_nodes, edges=global_edges, topos=global_topos, device=device)
        
        # 保存原始映射
        self.hdnet_list = hdnet_list
        self.node_group = node_group

    def _build_global_hypergraph(self, hdnet_list: List[Tuple[str, HDNet]], node_group: Tuple[Set[str], ...], device: torch.device) -> Tuple[Set[MHD_Node], Set[MHD_Edge], Set[MHD_Topo]]:
        """构建完整的全局超图"""
        # 步骤1：预处理所有子图节点/边
        sub_node_map = {}
        sub_edge_map = {}
        all_sub_node_names = set()
        
        for suffix, hdnet in hdnet_list:
            for node in hdnet.nodes:
                global_node_name = f"{suffix}{node.name}"
                sub_node_map[global_node_name] = (suffix, node.id, node)
                all_sub_node_names.add(global_node_name)
            
            for edge in hdnet.edges:
                global_edge_name = f"{suffix}{edge.name}"
                sub_edge_map[global_edge_name] = (suffix, edge.id, edge)

        # 步骤2：处理节点合并
        node_id_counter = 0
        merged_node_map = {}
        sub2global_node = {}
        
        merged_node_names = set()
        for group in node_group:
            self.validate_node_group_consistency(group, sub_node_map)
            
            sorted_node_names = sorted(
                group,
                key=lambda x: next((i for i, (suffix, _) in enumerate(hdnet_list) if x.startswith(f"{suffix}")), 999)
            )
            merged_name = "-".join(sorted_node_names)
            merged_node_names.update(sorted_node_names)
            
            merged_value = self.get_merged_node_value(group, sub_node_map).to(device)
            
            first_node_name = sorted_node_names[0]
            _, _, base_node = sub_node_map[first_node_name]
            
            merged_node = MHD_Node(
                id=node_id_counter,
                name=merged_name,
                value=merged_value,
                func=base_node.func
            )
            merged_node_map[merged_name] = merged_node
            
            for node_name in sorted_node_names:
                s, sn_id, _ = sub_node_map[node_name]
                self.sub2global_node_id[(s, sn_id)] = node_id_counter
                self.global2sub_node_id[node_id_counter] = (s, sn_id)
                sub2global_node[node_name] = merged_name
            
            node_id_counter += 1

        # 处理未被合并的独立节点
        unmerged_node_names = all_sub_node_names - merged_node_names
        for node_name in sorted(unmerged_node_names):
            suffix, sub_node_id, sub_node = sub_node_map[node_name]
            
            unmerged_node = MHD_Node(
                id=node_id_counter,
                name=node_name,
                value=sub_node.value.to(device),
                func=sub_node.func
            )
            merged_node_map[node_name] = unmerged_node
            
            self.sub2global_node_id[(suffix, sub_node_id)] = node_id_counter
            self.global2sub_node_id[node_id_counter] = (suffix, sub_node_id)
            sub2global_node[node_name] = node_name
            
            node_id_counter += 1

        # 步骤3：构建全局边
        edge_id_counter = 0
        merged_edge_map = {}
        
        for global_edge_name, (suffix, sub_edge_id, sub_edge) in sub_edge_map.items():
            merged_edge = MHD_Edge(
                id=edge_id_counter,
                name=global_edge_name,
                value=sub_edge.value,
                func=sub_edge.func
            )
            merged_edge_map[global_edge_name] = merged_edge
            
            self.sub2global_edge_id[(suffix, sub_edge_id)] = edge_id_counter
            self.global2sub_edge_id[edge_id_counter] = (suffix, sub_edge_id)
            
            edge_id_counter += 1

        # 步骤4：构建全局拓扑
        all_topo_types = set()
        for _, hdnet in hdnet_list:
            for topo in hdnet.topos:
                all_topo_types.add(topo.type)
        
        global_topos = set()
        num_global_edges = len(merged_edge_map)
        num_global_nodes = len(merged_node_map)
        
        for topo_type in all_topo_types:
            global_topo_value = torch.zeros((num_global_edges, num_global_nodes), dtype=torch.int64, device=device)
            
            for global_edge_id in range(num_global_edges):
                if global_edge_id not in self.global2sub_edge_id:
                    continue
                
                suffix, sub_edge_id = self.global2sub_edge_id[global_edge_id]
                hdnet = next(h for s, h in hdnet_list if s == suffix)
                
                sub_topo = MHD_Topo.type2obj(topo_type, hdnet.topos)
                if sub_topo is None or sub_edge_id >= sub_topo.value.shape[0]:
                    continue
                
                sub_topo_row = sub_topo.value[sub_edge_id]
                
                for sub_node_id in range(sub_topo_row.shape[0]):
                    topo_value = sub_topo_row[sub_node_id].item()
                    if topo_value == 0:
                        continue
                    
                    if (suffix, sub_node_id) in self.sub2global_node_id:
                        global_node_id = self.sub2global_node_id[(suffix, sub_node_id)]
                        if 0 <= global_node_id < num_global_nodes:
                            global_topo_value[global_edge_id, global_node_id] = topo_value
            
            global_topo = MHD_Topo(type=topo_type, value=global_topo_value)
            global_topos.add(global_topo)

        global_nodes = set(merged_node_map.values())
        global_edges = set(merged_edge_map.values())
        
        return global_nodes, global_edges, global_topos

def updown_node_value(
    nodes: Set['MHD_Node'],  
    path: str,
    mode: str
) -> None:
    """
    节点值保存/加载函数 (Node value save/load function)
    按节点名唯一匹配，仅保存value张量，非张量类型统一存空字典
    Match uniquely by node name, only save value tensor, empty dict for non-tensor types
    
    Args:
        nodes: MHD_Node对象集合 (Set of MHD_Node objects)
        path: 保存/加载路径 (Save/load path)
        mode: 操作模式 - 'up'加载/load | 'down'保存/save
    """
    # 初始化统计变量 (Initialize statistical variables)
    total_nodes = len(nodes)
    processed_nodes = 0
    
    if mode == 'down':
        save_dict = {"node_values": {}, "node_info": {}}
        
        for node_obj in nodes:
            node_name = node_obj.name
            # 仅保存value：张量直接存，非张量存空字典（统一规范）
            if isinstance(node_obj.value, torch.Tensor):
                node_val = node_obj.value
            else:
                node_val = {}  # 非张量统一存空字典
            save_dict["node_values"][node_name] = node_val
            save_dict["node_info"][node_name] = {
                "id": node_obj.id,
                "is_tensor": isinstance(node_obj.value, torch.Tensor)
            }
            processed_nodes += 1
        
        torch.save(save_dict, path)
        
    elif mode == 'up':
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        for node_obj in nodes:
            node_name = node_obj.name
            
            if node_name not in load_dict["node_values"]:
                continue
            
            saved_val = load_dict["node_values"][node_name]
            # 仅对张量类型执行赋值，非张量无操作
            if isinstance(saved_val, torch.Tensor) and isinstance(node_obj.value, torch.Tensor):
                # 保持设备和数据类型一致
                node_obj.value = saved_val.to(device=node_obj.value.device, dtype=node_obj.value.dtype)
            processed_nodes += 1
        
    else:
        raise ValueError(f"模式错误 (Invalid mode): {mode}，仅支持 'up'/'down' (only support 'up'/'down')")

    # 统一打印处理结果 (Unified print processing results)
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    print(f"📊 节点值{mode_cn}完成 (Node value {mode_en} completed)")
    print(f"   ├─ 总节点数 (Total nodes): {total_nodes}")
    print(f"   ├─ 处理节点数 (Processed nodes): {processed_nodes}")
    print(f"   📁 路径 (Path): {path}")

def updown_edge_value(
    edges: Set['MHD_Edge'],  
    path: str,
    mode: str
) -> None:
    """
    超边值保存/加载函数 (Edge value save/load function)
    按边名唯一匹配，仅保存params字典，非Module统一存空字典
    Match uniquely by edge name, only save params dict, empty dict for non-Module types
    
    Args:
        edges: MHD_Edge对象集合 (Set of MHD_Edge objects)
        path: 保存/加载路径 (Save/load path)
        mode: 操作模式 - 'up'加载/load | 'down'保存/save
    """
    total_edges = len(edges)
    processed_edges = 0
    
    if mode == 'down':
        save_dict = {"edge_params": {}, "edge_len": {}}
        
        for edge_obj in edges:
            edge_name = edge_obj.name
            params_list = []
            for elem in edge_obj.value:
                if isinstance(elem, nn.Module):
                    params = {p: v for p, v in elem.named_parameters()}
                else:
                    params = {}
                params_list.append(params)
            
            save_dict["edge_params"][edge_name] = params_list
            save_dict["edge_len"][edge_name] = len(edge_obj.value)
            processed_edges += 1
        
        torch.save(save_dict, path)
        
    elif mode == 'up':
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        
        for edge_obj in edges:
            edge_name = edge_obj.name
            
            if edge_name not in load_dict["edge_params"]:
                continue
            
            saved_params_list = load_dict["edge_params"][edge_name]
            for curr_idx, elem in enumerate(edge_obj.value):
                if curr_idx >= len(saved_params_list):
                    continue
                
                saved_params = saved_params_list[curr_idx]
                if isinstance(elem, nn.Module) and saved_params:
                    for p_name, p_val in saved_params.items():
                        elem.state_dict()[p_name].copy_(p_val)
            
            processed_edges += 1
        
    else:
        raise ValueError(f"模式错误 (Invalid mode): {mode}，仅支持 'up'/'down' (only support 'up'/'down')")

    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    print(f"📊 超边值{mode_cn}完成 (Edge value {mode_en} completed)")
    print(f"   ├─ 总边数 (Total edges): {total_edges}")
    print(f"   ├─ 处理边数 (Processed edges): {processed_edges}")
    print(f"   📁 路径 (Path): {path}")

def example_mhdnet2():
    """
    MHDNet示例 (MHDNet Example)
    流程：创建模型 → 保存边值/节点值 → 加载边值/节点值 → 修改值 → 实例化最终模型
    Process: Create model → Save edge/node values → Load edge/node values → Modify values → Instantiate final model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"✅ 使用设备 (Using device): {device}")

    # ========== 1. 创建子HDNet (Create sub HDNet) ==========
    # 子HDNet1
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
    edge1_value1 = [
        nn.Conv3d(3, 2, kernel_size=3, padding=1, bias=False).to(device),  
        nn.BatchNorm3d(2).to(device),                                    
        nn.ReLU(inplace=True)                                           
    ]
    edge2_value1 = [
        nn.Conv3d(3, 4, kernel_size=1, padding=0, bias=True).to(device),   
        nn.Sigmoid()                                                  
    ]
    edges_net1 = {
        MHD_Edge(
            id=0, 
            name="e1_A1_to_B1", 
            value=edge1_value1,
            func={"in": "concat", "out": "split"}
        ),
        MHD_Edge(
            id=1, 
            name="e2_A1_to_D1", 
            value=edge2_value1,
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

    # 子HDNet2
    nodes_net2 = {
        MHD_Node(
            id=0, 
            name="A2", 
            value=torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=1, 
            name="B2", 
            value=torch.randn(1, 2, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=2, 
            name="C2", 
            value=torch.randn(1, 5, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
    }
    edge1_value2 = [
        nn.Conv3d(5, 5, kernel_size=3, padding=1, groups=5, bias=False).to(device),
        nn.GELU(),
        nn.Conv3d(5, 5, kernel_size=1, padding=0, bias=True).to(device)
    ]
    edges_net2 = {
        MHD_Edge(
            id=0, 
            name="e1_A2B2_to_C2", 
            value=edge1_value2,
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

    # 子HDNet3
    nodes_net3 = {
        MHD_Node(
            id=0, 
            name="C3", 
            value=torch.randn(1, 5, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "sum"}
        ),
        MHD_Node(
            id=1, 
            name="D3", 
            value=torch.randn(1, 4, 8, 8, 8, device=device, dtype=dtype),
            func={"head": "share", "tail": "mul"}
        ),
    }
    edge1_value3 = [
        nn.Conv3d(5, 4, kernel_size=3, padding=1, bias=False).to(device),
        nn.Softplus(),
        '__mul__(0.5)'  # 字符串操作 (String operation)
    ]
    edges_net3 = {
        MHD_Edge(
            id=0, 
            name="e1_C3_to_D3", 
            value=edge1_value3,
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

    # 扩展HDNet3 (Extend HDNet3)
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
        '__add__(0.1)'  # 字符串操作 (String operation)
    ]
    new_edge_obj = MHD_Edge(
        id=1,
        name="e2_C3_to_E3",
        value=new_edge_e2_C3_to_E3,
        func={"in": "concat", "out": "split"}
    )
    
    updated_nodes_net3 = set(hdnet3.nodes)
    updated_nodes_net3.add(new_node_E3)
    updated_edges_net3 = set(hdnet3.edges)
    updated_edges_net3.add(new_edge_obj)
    
    role_topo_net3 = MHD_Topo.type2obj("role", hdnet3.topos)
    sort_topo_net3 = MHD_Topo.type2obj("sort", hdnet3.topos)
    
    updated_role_value = torch.cat([
        torch.cat([role_topo_net3.value, torch.zeros(role_topo_net3.value.shape[0], 1, device=device, dtype=torch.int64)], dim=1),
        torch.tensor([[-1, 0, 1]], device=device, dtype=torch.int64)
    ], dim=0)
    
    updated_sort_value = torch.cat([
        torch.cat([sort_topo_net3.value, torch.zeros(sort_topo_net3.value.shape[0], 1, device=device, dtype=torch.int64)], dim=1),
        torch.tensor([[1, 0, 2]], device=device, dtype=torch.int64)
    ], dim=0)
    
    updated_topos_net3 = {
        MHD_Topo(type="role", value=updated_role_value),
        MHD_Topo(type="sort", value=updated_sort_value)
    }
    hdnet3 = HDNet(nodes=updated_nodes_net3, edges=updated_edges_net3, topos=updated_topos_net3, device=device)

    # ========== 2. MHDNet合并得到全局对象 (MHDNet merge to get global objects) ==========
    hdnet_list = [("net1", hdnet1), ("net2", hdnet2), ("net3test", hdnet3)]
    node_group = ({"net1A1", "net2A2"}, {"net1B1", "net2B2"}, {"net2C2", "net3testC3"}, {"net1D1", "net3testD3"}, {"net3testE3"})
    
    mhdnet_bridge = MHDNet(hdnet_list=hdnet_list, node_group=node_group, device=device)
    global_nodes = mhdnet_bridge.nodes    
    global_edges = mhdnet_bridge.edges    
    global_topos = mhdnet_bridge.topos    

    # ========== 3. 核心流程 (Core Process) ==========
    target_edge = MHD_Edge.name2obj("net1e1_A1_to_B1", global_edges)
    edge_path = "./edge_values.pth"
    node_path = "./node_values.pth"

    # 3.1 修改指定节点名为input (Rename target node to 'input')
    target_node = MHD_Node.name2obj("net1A1-net2A2", global_nodes)
    if target_node:
        old_node_name = target_node.name
        target_node.name = "input"  # 重命名为input
        print(f"\n✅ 节点重命名 (Node renamed): {old_node_name} → input")

    # 3.2 保存超边值和节点值 (Save edge and node values)
    print("\n=== 1. 保存超边值 (Save edge values) ===")
    updown_edge_value(edges=global_edges, path=edge_path, mode="down")
    
    print("\n=== 2. 保存节点值 (Save node values) ===")
    updown_node_value(nodes=global_nodes, path=node_path, mode="down")

    # 3.3 加载超边值和节点值 (Load edge and node values)
    if target_edge:
        print("\n=== 3. 加载超边值 (Load edge values) ===")
        updown_edge_value(edges={target_edge}, path=edge_path, mode="up")
        
        print("\n=== 4. 加载节点值 (Load node values) ===")
        updown_node_value(nodes={target_node}, path=node_path, mode="up")
        
        # 3.4 加载后修改边值 (Modify edge values after loading)
        print("\n=== 5. 加载后修改超边值 (Modify edge values after loading) ===")
        target_edge.value = [
            nn.Conv3d(3, 2, kernel_size=3, padding=1, bias=False).to(device),  
            "relu()"                                                        
        ]
        print(f"✅ 超边 (Edge) [{target_edge.id}:{target_edge.name}] 修改完成 (Modified)")
        print(f"   原始长度 (Original len): 3 → 新长度 (New len): {len(target_edge.value)}")

    # 3.5 验证input节点值是否加载成功 (Verify input node value loaded)
    target_node = MHD_Node.name2obj("input", global_nodes)
    if target_node:
        print(f"\n✅ Input节点值验证 (Input node value verification):")
        print(f"   形状 (Shape): {target_node.value.shape}")
        print(f"   均值 (Mean): {target_node.value.mean().item():.6f}")

    # ========== 4. 实例化最终模型 (Instantiate final model) ==========
    final_model = HDNet(
        nodes=global_nodes,
        edges=global_edges,
        topos=global_topos,
        device=device
    )

    # 验证 (Verification)
    print(f"\n=== 6. 验证结果 (Verification Result) ===")
    print(f"✅ 最终模型实例化完成 (Final model instantiated)，节点数 (Node count): {len(final_model.nodes)}")
    if target_edge:
        conv_module = target_edge.value[0]
        print(f"✅ 超边 (Edge) [{target_edge.id}:{target_edge.name}] idx=0 权重均值 (Weight mean): {conv_module.weight.mean().item():.6f}")
    
    outputs = final_model.forward()
    print(f"✅ 前向传播完成 (Forward completed)，输出节点数 (Output node count): {len(outputs)}")

    return final_model

def verify_gradient(model):
    """验证梯度反传"""
    all_features = model.forward()
    target_node_name = "net1D1-net3testD3"
    if target_node_name in all_features:
        output_tensor = all_features[target_node_name]
        loss = output_tensor.sum()
        
        # 梯度清零
        model.zero_grad()
        # 反向传播
        loss.backward()
        
        has_gradient = False
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
                has_gradient = True
                print(f"✅ 参数 {name} 梯度正常: {param.grad.sum().item():.4f}")
        
        if has_gradient:
            print("\n✅ 梯度反传验证通过！")
        else:
            print("\n❌ 梯度反传验证失败！")
    else:
        print(f"\n❌ 无法验证梯度：输出节点 {target_node_name} 不存在")

# ===================== 主执行入口 =====================
if __name__ == "__main__":
    model = example_mhdnet2()
    verify_gradient(model)
    model.generate_mermaid()
