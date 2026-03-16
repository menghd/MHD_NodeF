# -*- coding: utf-8 -*-
"""
Multi Hypergraph Dynamic Framework (MHD)
Version: 2.0 (核心重构：单类统一超图对象管理，聚焦状态更新)
Core: Use objects to describe hypergraph + state update of hypergraph objects (retain object state)
"""
import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import re

# 全局配置
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===================== 通用工具函数 =====================
def get_obj_by_id(obj_id: int, obj_set: Set[Any], id_attr: str = "id") -> Optional[Any]:
    """通用ID查找函数"""
    for obj in obj_set:
        if getattr(obj, id_attr) == obj_id:
            return obj
    return None

def get_obj_by_name(obj_name: str, obj_set: Set[Any], name_attr: str = "name") -> Optional[Any]:
    """通用名称查找函数"""
    for obj in obj_set:
        if getattr(obj, name_attr) == obj_name:
            return obj
    return None

def parse_string_operation(op_str: str, x: torch.Tensor) -> torch.Tensor:
    """解析并执行字符串形式的张量操作（简化版）"""
    if '(' in op_str and ')' in op_str:
        method_name, args_str = op_str.split('(', 1)
        args_str = args_str.rstrip(')').strip()
        
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
        return getattr(x, op_str)()

# ===================== 核心超图框架类（MHD） =====================
@dataclass
class MHD_Node:
    """超图节点类 - 纯数据载体（特征图Tensor）"""
    id: int
    name: str
    value: torch.Tensor  # 特征图，普通Tensor而非nn.Parameter
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"})

    def __hash__(self):
        return hash(self.id)  # ID已唯一，无需name参与哈希

    def __eq__(self, other):
        if not isinstance(other, MHD_Node):
            return False
        return self.id == other.id  # ID唯一标识

    def get_head_transformed_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """纯数据转换：应用节点头函数"""
        head_funcs = {
            "share": lambda t: t.clone(memory_format=torch.contiguous_format),
        }
        return head_funcs[self.func.get("head", "share")](tensor)

    def get_tail_transformed_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """纯数据转换：应用节点尾函数"""
        tail_funcs = {
            "sum": lambda ts: torch.stack(ts).sum(dim=0),  # 修复：用torch.sum替代Python sum
            "avg": lambda ts: torch.stack(ts).mean(dim=0),
            "max": lambda ts: torch.stack(ts).max(dim=0)[0],
            "min": lambda ts: torch.stack(ts).min(dim=0)[0],
            "mul": lambda ts: torch.prod(torch.stack(ts), dim=0)
        }
        return tail_funcs[self.func.get("tail", "sum")](tensors)

@dataclass
class MHD_Edge:
    """超图边类 - 包含可学习模块的载体"""
    id: int
    name: str
    value: List[Union[str, nn.Module]]  # 包含可学习的Module
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"})

    def __hash__(self):
        return hash(self.id)  # 简化哈希逻辑

    def __eq__(self, other):
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id  # 简化相等判断

    def get_in_transformed_tensor(self, tensors: List[torch.Tensor], sorted_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """纯数据转换：应用边输入函数"""
        def concat(ts, sp):
            sorted_ts = [ts[i] for i, _ in sp if i < len(ts)]
            return torch.cat(sorted_ts, dim=1)

        def matmul(ts, sp):
            sorted_ts = [ts[i] for i, _ in sp if i < len(ts)]
            if len(sorted_ts) != 2:
                raise ValueError(f"Matmul需要2个输入张量，实际输入 {len(sorted_ts)}")
            return torch.matmul(*sorted_ts)

        in_funcs = {"concat": concat, "matmul": matmul}
        return in_funcs[self.func.get("in", "concat")](tensors, sorted_pairs)

    def get_out_transformed_tensors(self, x: torch.Tensor, sorted_pairs: List[Tuple[int, int]], node_channels: List[int]) -> List[torch.Tensor]:
        """纯数据转换：应用边输出函数"""
        def split(x, sp, nc):
            sorted_indices = [p[0] for p in sp if p[0] < len(nc)]
            sorted_sizes = [nc[i] for i in sorted_indices]
            split_ts = torch.split(x, sorted_sizes, dim=1)
            tensor_map = {idx: t for idx, t in zip(sorted_indices, split_ts)}
            return [tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)) 
                    for i in range(len(nc))]

        def svd(x, sp, nc):
            x_flat = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
            U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
            components = [U, S, Vh]
            sorted_ts = []
            
            for i, (orig_idx, _) in enumerate(sp):
                if orig_idx < len(nc):
                    comp_idx = i % len(components)
                    tensor = components[comp_idx]
                    if tensor.shape[1] != nc[orig_idx]:
                        tensor = nn.functional.adaptive_avg_pool1d(tensor.unsqueeze(1), nc[orig_idx]).squeeze(1)
                    sorted_ts.append((orig_idx, tensor))
            
            tensor_map = {idx: t for idx, t in sorted_ts}
            return [tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)) 
                    for i in range(len(nc))]

        def lu(x, sp, nc):
            x_flat = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
            P, L, U = torch.linalg.lu(x_flat)
            components = [L, U, P]
            sorted_ts = []
            
            for i, (orig_idx, _) in enumerate(sp):
                if orig_idx < len(nc):
                    comp_idx = i % len(components)
                    tensor = components[comp_idx]
                    if tensor.shape[1] != nc[orig_idx]:
                        tensor = nn.functional.adaptive_avg_pool1d(tensor.unsqueeze(1), nc[orig_idx]).squeeze(1)
                    sorted_ts.append((orig_idx, tensor))
            
            tensor_map = {idx: t for idx, t in sorted_ts}
            return [tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)) 
                    for i in range(len(nc))]

        out_funcs = {"split": split, "svd": svd, "lu": lu}
        return out_funcs[self.func.get("out", "split")](x, sorted_pairs, node_channels)

@dataclass
class MHD_Topo:
    """超图拓扑类"""
    type: str
    value: torch.Tensor

    def get_topo_value(self, edge_id: int, node_id: int) -> int:
        """获取指定边和节点的拓扑值"""
        if 0 <= edge_id < self.value.shape[0] and 0 <= node_id < self.value.shape[1]:
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
            raise ValueError(f"{self.type}拓扑边维度不匹配: 预期{num_edges}，实际{self.value.shape[0]}")
        if self.value.shape[1] != num_nodes:
            raise ValueError(f"{self.type}拓扑节点维度不匹配: 预期{num_nodes}，实际{self.value.shape[1]}")

class MHD_Graph(nn.Module):
    """多超图动态框架核心类"""
    def __init__(self, nodes: Set[MHD_Node], edges: Set[MHD_Edge], topos: Set[MHD_Topo], 
                 device: torch.device = None):
        super().__init__()
        # 统一设备配置
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心超图对象
        self.nodes = nodes
        self.edges = edges
        self.topos = {MHD_Topo(t.type, t.value.to(self.device)) for t in topos}
        
        # 参数注册容器 - 仅注册边中的可学习模块
        self.edge_module_map = nn.ModuleDict()
        
        # 初始化流程
        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        for topo in self.topos:
            topo.validate_topo(num_edges, num_nodes)
        
        self.compact_topological_sort()
        self._register_all_params()

    def compact_topological_sort(self) -> 'MHD_Graph':
        """压缩版拓扑排序"""
        role_topo = get_obj_by_name("role", self.topos, "type")
        if role_topo is None:
            self._edge_sequence = sorted([e.id for e in self.edges])
            self._node_sequence = sorted([n.id for n in self.nodes])
            print("✅ 拓扑分析完成（无role拓扑，默认按ID排序）")
            return self

        # 1. 边拓扑排序
        num_edges, num_nodes = role_topo.value.shape
        edge_deps = defaultdict(set)
        node_to_out_edges = defaultdict(set)
        
        for edge_id in range(num_edges):
            edge_row = role_topo.value[edge_id]
            in_nodes = [nid for nid in range(num_nodes) if edge_row[nid] < 0]
            out_nodes = [nid for nid in range(num_nodes) if edge_row[nid] > 0]
            
            for nid in out_nodes:
                node_to_out_edges[nid].add(edge_id)
            for nid in in_nodes:
                edge_deps[edge_id].update(node_to_out_edges.get(nid, set()))
            edge_deps[edge_id].discard(edge_id)

        edge_in_degree = {e.id: len(edge_deps.get(e.id, set())) for e in sorted(self.edges, key=lambda x: x.id)}
        reverse_edge_deps = defaultdict(set)
        for eid, deps in edge_deps.items():
            for dep in deps:
                reverse_edge_deps[dep].add(eid)

        remaining_edges = set(edge_in_degree.keys())
        edge_sequence = []
        while remaining_edges:
            current = sorted([e for e in remaining_edges if edge_in_degree[e] == 0])
            if not current:
                raise ValueError(f"超图边拓扑检测到环！剩余边: {remaining_edges}")
            edge_sequence.extend(current)
            for eid in current:
                remaining_edges.remove(eid)
                for next_eid in reverse_edge_deps.get(eid, set()):
                    edge_in_degree[next_eid] -= 1

        # 2. 节点拓扑排序
        node_deps = defaultdict(set)
        for edge_id in range(num_edges):
            edge_row = role_topo.value[edge_id]
            head_ids = [nid for nid in range(num_nodes) if edge_row[nid] < 0]
            tail_ids = [nid for nid in range(num_nodes) if edge_row[nid] > 0]
            for tail_id in tail_ids:
                node_deps[tail_id].update(head_ids)

        node_in_degree = {n.id: len(node_deps.get(n.id, set())) for n in sorted(self.nodes, key=lambda x: x.id)}
        reverse_node_deps = defaultdict(set)
        for nid, deps in node_deps.items():
            for dep in deps:
                reverse_node_deps[dep].add(nid)

        remaining_nodes = set(node_in_degree.keys())
        node_sequence = []
        while remaining_nodes:
            current = sorted([n for n in remaining_nodes if node_in_degree[n] == 0])
            if not current:
                raise ValueError(f"超图节点拓扑检测到环！剩余节点: {remaining_nodes}")
            node_sequence.extend(current)
            for nid in current:
                remaining_nodes.remove(nid)
                for next_nid in reverse_node_deps.get(nid, set()):
                    node_in_degree[next_nid] -= 1

        # 保存拓扑排序结果
        self._edge_sequence = edge_sequence
        self._node_sequence = node_sequence
        
        # 打印拓扑日志
        edge_names = [get_obj_by_id(eid, self.edges).name for eid in edge_sequence]
        node_names = [get_obj_by_id(nid, self.nodes).name for nid in node_sequence]
        print(f"✅ 拓扑分析完成（基于role矩阵）:")
        print(f" ├─ 边执行序列: {edge_names}")
        print(f" └─ 节点拓扑序列: {node_names}")
        
        return self

    def _register_all_params(self) -> 'MHD_Graph':
        """仅注册边中的可学习模块（节点Tensor无需注册）"""
        # 边模块注册
        for edge in sorted(self.edges, key=lambda x: x.id):
            for idx, op in enumerate(edge.value):
                if isinstance(op, nn.Module):
                    module_name = f"edge_{edge.name}_op_{idx}"
                    self.edge_module_map[module_name] = op.to(self.device)
                    edge.value[idx] = self.edge_module_map[module_name]
        
        return self

    def sort_nodes_by_topo_value(self, edge_id: int = 0) -> List[Tuple[int, int]]:
        """按sort拓扑值排序节点"""
        sort_topo = get_obj_by_name("sort", self.topos, "type")
        sort_values = sort_topo.value if sort_topo else torch.zeros(len(self.edges), len(self.nodes), device=self.device)
        
        if edge_id >= sort_values.shape[0]:
            return []
        
        indexed_nodes = list(enumerate(sort_values[edge_id].tolist()))
        return sorted(indexed_nodes, key=lambda p: p[1])

    def _execute_edge_operations(self, edge: MHD_Edge, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行边的操作序列（简化版）- 保留计算图"""
        x = input_tensor
        for op in edge.value:
            if isinstance(op, nn.Module):
                x = op(x)  # 通过注册的Module，保留计算图
            elif isinstance(op, str):
                x = parse_string_operation(op, x)  # 自定义操作也保留计算图
        return x

    def forward(self) -> 'MHD_Graph':
        """拓扑驱动前向传播 - 关键：保留计算图"""
        if not self.nodes or not self.edges:
            return self
        
        role_topo = get_obj_by_name("role", self.topos, "type")
        if role_topo is None:
            return self
        
        num_edges, num_nodes = role_topo.value.shape

        # 按拓扑序列执行边操作
        for edge_id in self._edge_sequence:
            edge = get_obj_by_id(edge_id, self.edges)
            if not edge or edge_id >= num_edges:
                continue

            # 获取输入/输出节点
            edge_row = role_topo.value[edge_id]
            head_node_ids = [nid for nid in range(num_nodes) if edge_row[nid].item() < 0]
            tail_node_ids = [nid for nid in range(num_nodes) if edge_row[nid].item() > 0]
            
            if not head_node_ids or not tail_node_ids:
                continue

            # 处理输入张量（特征图，保留计算图）
            head_tensors = []
            for nid in head_node_ids:
                node = get_obj_by_id(nid, self.nodes)
                if node:
                    # 保留计算图的clone
                    head_tensors.append(node.get_head_transformed_tensor(node.value))
            
            if not head_tensors:
                continue

            # 边输入转换（保留计算图）
            sorted_pairs = self.sort_nodes_by_topo_value(edge_id)
            edge_input = edge.get_in_transformed_tensor(head_tensors, sorted_pairs)

            # 执行边操作（包含可学习模块，保留计算图）
            edge_output = self._execute_edge_operations(edge, edge_input)

            # 处理输出通道（保留计算图）
            tail_node_channels = [
                get_obj_by_id(nid, self.nodes).value.shape[1] if get_obj_by_id(nid, self.nodes) else 0
                for nid in tail_node_ids
            ]

            # 边输出拆分（保留计算图）
            tail_tensors = edge.get_out_transformed_tensors(edge_output, sorted_pairs, tail_node_channels)

            # ========== 节点特征图更新逻辑 - 关键：保留计算图 ==========
            for idx, node_id in enumerate(tail_node_ids):
                if idx >= len(tail_tensors):
                    continue
                node = get_obj_by_id(node_id, self.nodes)
                if not node:
                    continue
                
                tensor = tail_tensors[idx]
                
                # 关键：不切断计算图，直接更新（保留grad_fn）
                new_value = node.get_tail_transformed_tensor([node.value, tensor])
                # 不使用to(device)（会创建新Tensor，切断计算图），改用in-place操作
                if new_value.device != self.device:
                    new_value = new_value.to(self.device, non_blocking=True)
                node.value = new_value  # 保留计算图的赋值

        return self

    def generate_mermaid(self) -> str:
        """统一可视化"""
        mermaid = [
            "graph TD",
            "",
            " classDef MHD_Node_Style fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
            " classDef MHD_Edge_Style fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
            "",
        ]

        role_topo = get_obj_by_name("role", self.topos, "type")
        # 添加节点
        for node in sorted(self.nodes, key=lambda x: x.id):
            mermaid.append(f" {node.name}:::MHD_Node_Style")
        
        # 添加边和连接关系
        if role_topo:
            for edge in sorted(self.edges, key=lambda x: x.id):
                edge_name = edge.name
                edge_id = edge.id
                if edge_id < role_topo.value.shape[0]:
                    edge_row = role_topo.value[edge_id]
                    mermaid.append(f" {edge_name}:::MHD_Edge_Style")
                    
                    head_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() < 0]
                    tail_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() > 0]
                    
                    head_node_names = [get_obj_by_id(nid, self.nodes).name for nid in head_node_ids if get_obj_by_id(nid, self.nodes)]
                    tail_node_names = [get_obj_by_id(nid, self.nodes).name for nid in tail_node_ids if get_obj_by_id(nid, self.nodes)]
                    
                    for head_node in head_node_names:
                        mermaid.append(f" {head_node} --> {edge_name}")
                    for tail_node in tail_node_names:
                        mermaid.append(f" {edge_name} --> {tail_node}")
                mermaid.append("")

        mermaid_code = "\n".join(mermaid)
        print("=== 超图可视化 ===")
        print(mermaid_code)
        return mermaid_code

    @classmethod
    def merge_graph(cls, graph_list: List[Tuple[str, 'MHD_Graph']], 
                                node_group: Tuple[Set[str], ...], 
                                device: torch.device = None) -> 'MHD_Graph':
        """最终版：外壳独立 + 可变数据浅拷贝 → 完全隔离子图"""
        sub2global_node_id = {}
        sub2global_edge_id = {}
        
        sub_node_map = {}
        sub_edge_map = {}
        
        for suffix, graph in graph_list:
            for node in sorted(graph.nodes, key=lambda x: x.id):
                global_node_name = f"{suffix}{node.name}"
                sub_node_map[global_node_name] = (suffix, node.id, node)
            for edge in sorted(graph.edges, key=lambda x: x.id):
                global_edge_name = f"{suffix}{edge.name}"
                sub_edge_map[global_edge_name] = (suffix, edge.id, edge)

        # 节点处理（特征图Tensor，浅拷贝隔离）
        node_id_counter = 0
        merged_nodes = set()
        merged_node_names = set()

        for group in node_group:
            sorted_node_names = sorted(
                group, 
                key=lambda x: next((i for i, (suffix, _) in enumerate(graph_list) if x.startswith(f"{suffix}")), 999)
            )
            merged_name = "_MERGE_".join(sorted_node_names)
            merged_node_names.update(sorted_node_names)
            
            group_node_objs = [sub_node_map[name][2] for name in sorted_node_names if name in sub_node_map]
            if not group_node_objs:
                continue
                
            if len(group_node_objs) == 1:
                # 特征图clone保证独立 - 保留requires_grad
                merged_value = group_node_objs[0].value.clone(memory_format=torch.contiguous_format).to(device)
            else:
                merged_value = torch.stack([n.value for n in group_node_objs]).mean(dim=0).to(device)
            
            base_node = group_node_objs[0]
            merged_node = MHD_Node(
                id=node_id_counter,
                name=merged_name,
                value=merged_value,  # 普通Tensor
                func=base_node.func.copy()  # 字典浅拷贝 → 独立字典
            )
            merged_nodes.add(merged_node)
            
            for node_name in sorted_node_names:
                s, sn_id, _ = sub_node_map[node_name]
                sub2global_node_id[(s, sn_id)] = node_id_counter
            node_id_counter += 1

        # 独立节点（特征图隔离）
        unmerged_node_names = set(sub_node_map.keys()) - merged_node_names
        for node_name in sorted(unmerged_node_names):
            suffix, sub_node_id, sub_node = sub_node_map[node_name]
            
            unmerged_node = MHD_Node(
                id=node_id_counter,
                name=node_name,
                value=sub_node.value.clone(memory_format=torch.contiguous_format).to(device),  # tensor clone → 独立特征图
                func=sub_node.func.copy()                  # 字典copy → 独立字典
            )
            merged_nodes.add(unmerged_node)
            
            sub2global_node_id[(suffix, sub_node_id)] = node_id_counter
            node_id_counter += 1

        # 边处理（保留可学习模块）
        edge_id_counter = 0
        merged_edges = set()
        for global_edge_name, (suffix, sub_edge_id, sub_edge) in sorted(sub_edge_map.items(), key=lambda x: x[1][1]):
            merged_edge = MHD_Edge(
                id=edge_id_counter,
                name=global_edge_name,
                value=sub_edge.value.copy(),  # 列表浅拷贝 → 独立列表（内部Module仍引用，但权重可通过clone隔离）
                func=sub_edge.func.copy()     # 字典copy → 独立字典
            )
            merged_edges.add(merged_edge)
            
            sub2global_edge_id[(suffix, sub_edge_id)] = edge_id_counter
            edge_id_counter += 1

        # 拓扑处理（完全独立，无共享）
        all_topo_types = set(topo.type for _, graph in graph_list for topo in graph.topos)
        merged_topos = set()
        num_global_edges = len(merged_edges)
        num_global_nodes = len(merged_nodes)

        for topo_type in all_topo_types:
            global_topo_value = torch.zeros((num_global_edges, num_global_nodes), dtype=torch.int64, device=device)
            
            for global_edge_id in range(num_global_edges):
                sub_info = next(((s, se) for (s, se), ge in sub2global_edge_id.items() if ge == global_edge_id), None)
                if not sub_info:
                    continue
                    
                suffix, sub_edge_id = sub_info
                graph = next(h for s, h in graph_list if s == suffix)
                sub_topo = get_obj_by_name(topo_type, graph.topos, "type")
                
                if sub_topo is None or sub_edge_id >= sub_topo.value.shape[0]:
                    continue
                
                sub_topo_row = sub_topo.value[sub_edge_id]
                for sub_node_id in range(sub_topo_row.shape[0]):
                    topo_value = sub_topo_row[sub_node_id].item()
                    if topo_value == 0:
                        continue
                        
                    if (suffix, sub_node_id) in sub2global_node_id:
                        global_node_id = sub2global_node_id[(suffix, sub_node_id)]
                        if 0 <= global_node_id < num_global_nodes:
                            global_topo_value[global_edge_id, global_node_id] = topo_value
            
            merged_topo = MHD_Topo(type=topo_type, value=global_topo_value)
            merged_topos.add(merged_topo)

        # 最终实例化（完全独立）
        merged_graph = cls(
            nodes=merged_nodes,
            edges=merged_edges,
            topos=merged_topos,
            device=device
        )
        
        merged_graph._register_all_params()
        merged_graph.compact_topological_sort()
        
        return merged_graph

# ===================== 状态保存/加载工具函数 =====================
def updown_node_value(nodes: Set[MHD_Node], path: str, mode: str) -> None:
    """节点特征图保存/加载函数（普通Tensor）"""
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        save_dict = {
            "node_values": {n.name: n.value for n in sorted(nodes, key=lambda x: x.id)},
            "node_info": {
                n.name: {"id": n.id, "shape": n.value.shape, "dtype": str(n.value.dtype), "device": str(n.value.device)}
                for n in sorted(nodes, key=lambda x: x.id)
            }
        }
        torch.save(save_dict, path)
        
    else:
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for node in sorted(nodes, key=lambda x: x.id):
            if node.name in load_dict["node_values"]:
                node.value = load_dict["node_values"][node.name].to(device=node.value.device, dtype=node.value.dtype)

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for n in nodes if n.name in (load_dict["node_values"] if mode == 'up' else [n.name for n in nodes]))
    
    print(f"📊 节点特征图{mode_cn}完成 (Node value {mode_en} completed)")
    print(f" ├─ 总节点数: {len(nodes)}")
    print(f" ├─ 处理节点数: {processed}")
    print(f" 📁 路径: {path}")

def updown_edge_value(edges: Set[MHD_Edge], path: str, mode: str) -> None:
    """超边可学习模块保存/加载函数"""
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        save_dict = {
            "edge_params": {},
            "edge_info": {}
        }
        for edge in sorted(edges, key=lambda x: x.id):
            save_dict["edge_params"][edge.name] = [
                elem.state_dict() if isinstance(elem, nn.Module) else None
                for elem in edge.value
            ]
            save_dict["edge_info"][edge.name] = {
                "id": edge.id,
                "operations": [str(type(op)) for op in edge.value]
            }
        torch.save(save_dict, path)
        
    else:
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for edge in sorted(edges, key=lambda x: x.id):
            if edge.name in load_dict["edge_params"]:
                saved_params = load_dict["edge_params"][edge.name]
                for idx, elem in enumerate(edge.value):
                    if idx < len(saved_params) and isinstance(elem, nn.Module) and saved_params[idx] is not None:
                        elem.load_state_dict(saved_params[idx])

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for e in edges if e.name in (load_dict["edge_params"] if mode == 'up' else [e.name for e in edges]))
    
    print(f"📊 超边可学习参数{mode_cn}完成 (Edge value {mode_en} completed)")
    print(f" ├─ 总边数: {len(edges)}")
    print(f" ├─ 处理边数: {processed}")
    print(f" 📁 路径: {path}")

import torch
import random
import numpy as np
import nibabel as nib
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Callable, Any, Union, Iterable, Optional, Sequence
from pathlib import Path
from abc import ABC, abstractmethod

# ===================== 增强基类：支持多增强器+共享种子 =====================
class MHD_Augmentor(ABC):
    """增强器基类：基于种子保证同一样本的增强一致性"""
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else random.randint(0, 1000000)
    
    def set_seed(self, seed: int):
        self.seed = seed
    
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """执行增强，子类实现具体逻辑"""
        pass

# 示例1：随机旋转增强器
class RandomRotate(MHD_Augmentor):
    def __init__(self, max_angle: float = 10.0, seed: Optional[int] = None):
        super().__init__(seed)
        self.max_angle = max_angle
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # 基于种子生成固定角度，保证同一样本所有节点旋转角度一致
        random.seed(self.seed)
        angle = random.uniform(-self.max_angle, self.max_angle)
        
        # 2D张量旋转（可扩展到3D）
        if tensor.dim() == 3:  # (C, H, W)
            C, H, W = tensor.shape
            tensor_np = tensor.cpu().numpy()
            rotated = []
            for c in range(C):
                img = tensor_np[c]
                center = (W//2, H//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, M, (W, H))
                rotated.append(rotated_img)
            return torch.tensor(np.stack(rotated), dtype=tensor.dtype)
        return tensor

# 示例2：随机翻转增强器
class RandomFlip(MHD_Augmentor):
    def __init__(self, axis: int = 1, seed: Optional[int] = None):
        super().__init__(seed)
        self.axis = axis  # 0:上下翻转，1:左右翻转
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        random.seed(self.seed)
        flip = random.choice([True, False])
        if flip:
            return torch.flip(tensor, dims=[self.axis])
        return tensor

# 示例3：归一化增强器（无随机）
class Normalize(MHD_Augmentor):
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

# ===================== 增强组合器：支持多增强器列表 =====================
class MHD_AugmentComposer:
    """增强组合器：执行增强列表，共享同一种子"""
    def __init__(self, augmentors: Sequence[MHD_Augmentor]):
        self.augmentors = augmentors
    
    def set_seed(self, seed: int):
        """为所有增强器设置同一种子"""
        for aug in self.augmentors:
            aug.set_seed(seed)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """按顺序执行所有增强"""
        for aug in self.augmentors:
            tensor = aug(tensor)
        return tensor

# ===================== 统一数据集：最终完整版 =====================
class MHD_ExtendDataset(Dataset):
    """
    高扩展MHD数据集（最终版）：
    1. 支持多参数传递（元组/字典）
    2. 支持任意文件类型（nii/npy/png等）
    3. 支持增强器列表（多步增强）
    4. 共享增强器实例，保证同一样本增强随机一致性
    5. 增强在CPU完成，GPU只做数据搬运
    6. 基于索引+种子的随机增强，可复现
    """
    def __init__(
        self,
        sample_info_list: List[Any],  # 样本信息列表（索引/路径/元组/字典）
        node_configs: Dict[str, Dict],  # 节点配置：{节点名: {"loader": 加载函数, "augmentor": 增强器/增强器列表}}
        base_seed: int = 42,  # 基础种子，保证可复现
        device: torch.device = None
    ):
        """
        参数说明：
        - sample_info_list: 样本信息列表，示例：[(0, "./data1.nii"), (1, "./data2.npy"), ...]
        - node_configs: 节点配置，示例：
          {
              "input": {
                  "loader": lambda info: 加载函数(info),
                  "augmentor": MHD_AugmentComposer([RandomRotate(10), RandomFlip(1), Normalize(0.5, 0.5)])
              },
              "gt": {
                  "loader": lambda info: 加载函数(info),
                  "augmentor": same_aug_composer  # 共享增强器实例，保证变换一致
              }
          }
        """
        self.sample_info_list = sample_info_list
        self.node_configs = node_configs
        self.base_seed = base_seed
        self.device = device or torch.device("cpu")
        self.node_names = list(node_configs.keys())

        # 预处理增强器：将单个增强器/列表转为组合器
        for node_name, config in self.node_configs.items():
            aug = config.get("augmentor")
            if aug is not None:
                if isinstance(aug, Sequence) and not isinstance(aug, MHD_AugmentComposer):
                    # 列表形式 → 转为组合器
                    config["augmentor"] = MHD_AugmentComposer(aug)
                elif isinstance(aug, MHD_Augmentor) and not isinstance(aug, MHD_AugmentComposer):
                    # 单个增强器 → 转为组合器（统一接口）
                    config["augmentor"] = MHD_AugmentComposer([aug])

    def __len__(self) -> int:
        return len(self.sample_info_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        单样本加载流程：
        1. 基于索引生成唯一种子（保证同一样本所有节点增强一致）
        2. 加载所有节点的原始数据（CPU）
        3. 对每个节点执行增强列表（CPU，共用种子）
        4. 搬运到指定设备（GPU）
        """
        # 1. 生成同一样本的唯一增强种子（全局一致）
        sample_seed = self.base_seed + idx
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)

        # 2. 获取当前样本信息
        sample_info = self.sample_info_list[idx]

        # 3. 加载+增强每个节点的数据（CPU完成）
        sample_data = {}
        for node_name, config in self.node_configs.items():
            # 3.1 调用节点专属加载函数（支持任意文件类型、多参数）
            load_fn = config["loader"]
            try:
                tensor = load_fn(sample_info)
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"节点 {node_name} 加载失败（idx={idx}）: {str(e)}")

            # 3.2 执行增强列表（共用种子）
            augmentor = config.get("augmentor")
            if augmentor is not None:
                augmentor.set_seed(sample_seed)  # 为组合器内所有增强器设置同一种子
                tensor = augmentor(tensor)

            # 3.3 搬运到指定设备（GPU）
            sample_data[node_name] = tensor.to(self.device, non_blocking=True)

        return sample_data

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """批次拼接：保证B维度一致"""
        if not batch:
            raise ValueError("空批次数据")

        ref_node_names = batch[0].keys()
        batch_data = {}

        for node_name in ref_node_names:
            node_samples = [sample[node_name] for sample in batch]
            batch_data[node_name] = torch.stack(node_samples, dim=0)

        return batch_data

# ===================== 工具函数：创建DataLoader =====================
def create_mhd_extend_dataloader(
    dataset: MHD_ExtendDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MHD_ExtendDataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

import warnings
from collections import defaultdict
import logging
from datetime import datetime
from tqdm import tqdm
import os
import json
import torch
import torch.nn as nn
import numpy as np

# 确保依赖的工具函数可访问
try:
    from MHD_Framework import (
        get_obj_by_name, updown_node_value, updown_edge_value
    )
except ImportError:
    # 模拟工具函数（保证代码可独立运行）
    def get_obj_by_name(name, objs):
        for obj in objs:
            if obj.name == name:
                return obj
        return None
    
    def updown_node_value(nodes, path, mode): pass
    def updown_edge_value(edges, path, mode): pass

class MHD_Monitor:
    """保持原有监控器逻辑不变"""
    def __init__(self, monitor_nodes: list, monitor_edges: list = None):
        self.monitor_nodes = monitor_nodes
        self.monitor_edges = monitor_edges
        self.records = defaultdict(list)
        self.step_counter = 0
    
    def reset(self):
        self.records = defaultdict(list)
        self.step_counter = 0
    
    def _safe_tensor_stats(self, tensor: torch.Tensor) -> dict:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        return {
            "mean": float(tensor.mean().item()),
            "sum": float(tensor.sum().item()),
            "max": float(tensor.max().item()),
            "min": float(tensor.min().item())
        }
    
    def monitor_node(self, mhd_graph, prefix: str = "") -> dict:
        node_metrics = {}
        for node_name in self.monitor_nodes:
            node = get_obj_by_name(node_name, mhd_graph.nodes)
            if node is None:
                warnings.warn(f"监控节点 {node_name} 不存在，跳过")
                continue
            
            value = node.value.detach()
            stats = self._safe_tensor_stats(value)
            
            for stat_name, stat_value in stats.items():
                node_metrics[f"{prefix}node_{node_name}_{stat_name}"] = stat_value
        
        for k, v in node_metrics.items():
            self.records[k].append(v)
        self.step_counter += 1
        
        return node_metrics
    
    def monitor_edge(self, mhd_graph, prefix: str = "", train_mode: bool = True) -> dict:
        edge_metrics = {}
        target_edges = self.monitor_edges or [e.name for e in mhd_graph.edges]
        
        for edge_name in target_edges:
            edge = get_obj_by_name(edge_name, mhd_graph.edges)
            if edge is None:
                warnings.warn(f"监控边 {edge_name} 不存在，跳过")
                continue
            
            for idx, op in enumerate(edge.value):
                if not isinstance(op, nn.Module):
                    continue
                
                weight_key = f"{prefix}edge_{edge_name}_op{idx}"
                if hasattr(op, 'weight') and op.weight is not None:
                    weight = op.weight.detach()
                    weight_stats = self._safe_tensor_stats(weight)
                    edge_metrics[f"{weight_key}_weight_mean"] = weight_stats["mean"]
                    edge_metrics[f"{weight_key}_weight_l2"] = float(torch.norm(weight).item())
                
                if train_mode and hasattr(op, 'weight') and op.weight.grad is not None:
                    grad = op.weight.grad.detach()
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
        node_metrics = {k: v for k, v in metrics.items() if "node_" in k}
        edge_metrics = {k: v for k, v in metrics.items() if "edge_" in k}
        
        if node_metrics:
            formatted.append("📌 节点指标:")
            for k, v in sorted(node_metrics.items()):
                formatted.append(f"  {k}: {v:.{decimal}f}")
        
        if edge_metrics:
            formatted.append("🔗 边指标:")
            for k, v in sorted(edge_metrics.items()):
                formatted.append(f"  {k}: {v:.{decimal}f}")
        
        return "\n".join(formatted)

# ===================== 最终修正版 MHD_Trainer =====================
class MHD_Trainer:
    def __init__(
        self,
        mhd_graph,
        optimizer: torch.optim.Optimizer,
        monitor: MHD_Monitor,
        save_dir: str = "./mhd_ckpts",
        target_loss_node: str = "final_fl",
        target_metric_node: str = None,
        device: torch.device = None,
        grad_clip_norm: float = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        self.mhd_graph = mhd_graph
        self.optimizer = optimizer
        self.monitor = monitor
        self.save_dir = save_dir
        self.target_loss_node = target_loss_node
        self.target_metric_node = target_metric_node
        self.device = device or mhd_graph.device
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler = lr_scheduler
        
        os.makedirs(save_dir, exist_ok=True)
        self.logger = self._setup_logger(save_dir)
        
        self.history = {
            "train": {"loss": [], "metrics": []},
            "eval": {"loss": [], "metrics": []},
            "best_eval_loss": float("inf"),
            "best_eval_metric": -float("inf"),
            "best_epoch_loss": -1,
            "best_epoch_metric": -1
        }
        
        self._validate_loss_node()
        
        self.logger.info("="*80)
        self.logger.info("🚀 MHD训练器初始化完成（修复版）")
        self.logger.info(f"📌 损失节点: {self.target_loss_node} (min模式)")
        self.logger.info(f"📈 指标节点: {self.target_metric_node} (max模式)")
        self.logger.info(f"📊 监控节点: {self.monitor.monitor_nodes}")
        self.logger.info(f"🔗 监控边: {self.monitor.monitor_edges or '所有边'}")
        self.logger.info(f"💾 保存目录: {self.save_dir}")
        self.logger.info(f"💻 设备: {self.device}")
        self.logger.info("="*80)
    
    def _setup_logger(self, save_dir: str) -> logging.Logger:
        os.makedirs(save_dir, exist_ok=True)
        
        logger = logging.getLogger("mhd_train")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_file = os.path.join(save_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _validate_loss_node(self):
        loss_node = get_obj_by_name(self.target_loss_node, self.mhd_graph.nodes)
        if loss_node is None:
            raise ValueError(f"损失节点 '{self.target_loss_node}' 不存在！")
        self.logger.info(f"✅ 损失节点验证通过: {self.target_loss_node}")
        
        if self.target_metric_node:
            metric_node = get_obj_by_name(self.target_metric_node, self.mhd_graph.nodes)
            if metric_node is None:
                raise ValueError(f"指标节点 '{self.target_metric_node}' 不存在！")
            self.logger.info(f"✅ 指标节点验证通过: {self.target_metric_node}")
    
    def train_step(self, input_dict: dict) -> tuple[float, float]:
        """单步训练（核心修复：切断计算图复用）"""
        self.mhd_graph.train()
        
        # ========== 关键修复1：彻底清空梯度 ==========
        self.optimizer.zero_grad(set_to_none=True)
        
        # 批次一致性校验
        try:
            batch_size = self._check_batch_consistency(input_dict)
        except ValueError as e:
            self.logger.error(f"批次校验失败: {str(e)}")
            raise
        
        # ========== 关键修复2：注入节点时创建全新Tensor ==========
        for node_name, tensor in input_dict.items():
            node = get_obj_by_name(node_name, self.mhd_graph.nodes)
            if node is None:
                warnings.warn(f"输入数据中的节点 {node_name} 不存在于图中，跳过")
                continue
            
            # 核心：创建全新Tensor，切断历史计算图
            new_tensor = tensor.to(self.device, non_blocking=True).clone().detach()
            new_tensor.requires_grad_(True)
            node.value = new_tensor
        
        # 前向传播
        self.mhd_graph.forward()
        
        # 监控指标
        node_metrics = self.monitor.monitor_node(self.mhd_graph, prefix="train_")
        edge_metrics = self.monitor.monitor_edge(self.mhd_graph, prefix="train_", train_mode=True)
        
        # 获取损失
        loss_node = get_obj_by_name(self.target_loss_node, self.mhd_graph.nodes)
        if loss_node is None:
            raise ValueError(f"损失节点 {self.target_loss_node} 不存在")
        
        loss_tensor = loss_node.value.mean()
        
        # 检查梯度可用性
        if not loss_tensor.requires_grad:
            self.logger.error(f"损失张量无梯度！损失节点 grad_fn: {loss_tensor.grad_fn}")
            key_nodes = list(input_dict.keys()) + [self.target_loss_node]
            for node_name in key_nodes:
                node = get_obj_by_name(node_name, self.mhd_graph.nodes)
                if node:
                    self.logger.error(f"  {node_name}: requires_grad={node.value.requires_grad}, grad_fn={node.value.grad_fn if hasattr(node.value, 'grad_fn') else None}")
            raise RuntimeError("损失张量无梯度，请检查前向计算逻辑")
        
        # 反向传播
        loss_tensor.backward(retain_graph=False)
        
        # 梯度裁剪
        if self.grad_clip_norm and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.mhd_graph.parameters(), self.grad_clip_norm)
        
        # 优化器更新
        self.optimizer.step()
        
        # ========== 关键修复3：正确清理缓存（仅GPU） ==========
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 计算指标
        metric_value = 0.0
        if self.target_metric_node:
            metric_node = get_obj_by_name(self.target_metric_node, self.mhd_graph.nodes)
            if metric_node:
                metric_value = float(metric_node.value.detach().mean().item())
        
        # 数值稳定处理
        loss_value = float(loss_tensor.detach().item())
        loss_value = 0.0 if np.isnan(loss_value) or np.isinf(loss_value) else loss_value
        metric_value = 0.0 if np.isnan(metric_value) or np.isinf(metric_value) else metric_value
        
        return loss_value, metric_value

    @torch.no_grad()  # 关键：验证阶段禁用梯度
    def eval_step(self, input_dict: dict) -> tuple[float, float]:
        """单步验证"""
        self.mhd_graph.eval()
        
        # 批次一致性校验
        try:
            batch_size = self._check_batch_consistency(input_dict)
        except ValueError as e:
            self.logger.error(f"批次校验失败: {str(e)}")
            raise
        
        # 注入节点值
        for node_name, tensor in input_dict.items():
            node = get_obj_by_name(node_name, self.mhd_graph.nodes)
            if node is None:
                warnings.warn(f"输入数据中的节点 {node_name} 不存在于图中，跳过")
                continue
            node.value = tensor.to(self.device, non_blocking=True).detach()
            node.value.requires_grad_(False)
        
        # 前向传播
        self.mhd_graph.forward()
        
        # 监控指标
        node_metrics = self.monitor.monitor_node(self.mhd_graph, prefix="eval_")
        edge_metrics = self.monitor.monitor_edge(self.mhd_graph, prefix="eval_", train_mode=False)
        
        # 获取损失
        loss_node = get_obj_by_name(self.target_loss_node, self.mhd_graph.nodes)
        loss_tensor = loss_node.value.mean()
        
        loss_tensor = torch.nan_to_num(loss_tensor, nan=1e3, posinf=1e3, neginf=-1e3).cpu()
        loss_value = float(loss_tensor.item())
        
        # 计算指标
        metric_value = 0.0
        if self.target_metric_node:
            metric_key = f"eval_node_{self.target_metric_node}_mean"
            metric_value = float(node_metrics.get(metric_key, 0.0))
            metric_value = 0.0 if np.isnan(metric_value) or np.isinf(metric_value) else metric_value
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss_value, metric_value
    
    def _check_batch_consistency(self, input_dict: dict) -> int:
        """校验所有节点数据的批次大小一致性"""
        batch_sizes = []
        invalid_nodes = []
        
        for node_name, tensor in input_dict.items():
            if tensor.dim() < 2:
                invalid_nodes.append(f"{node_name} (维度不足: {tensor.dim()})")
                continue
            
            batch_size = tensor.shape[0]
            batch_sizes.append((node_name, batch_size))
        
        if invalid_nodes:
            raise ValueError(f"节点数据维度不合法（需至少2维: B×C×...）: {', '.join(invalid_nodes)}")
        
        if not batch_sizes:
            raise ValueError("输入字典为空，无节点数据可校验")
        
        ref_batch_size = batch_sizes[0][1]
        inconsistent_nodes = [
            f"{name} (B={bs})" for name, bs in batch_sizes if bs != ref_batch_size
        ]
        
        if inconsistent_nodes:
            raise ValueError(
                f"批次大小不一致！基准批次: {ref_batch_size}, "
                f"不一致节点: {', '.join(inconsistent_nodes)}"
            )
        
        return ref_batch_size
    
    def train_epoch(self, train_data: list, epoch: int):
        self.monitor.reset()
        total_loss = 0.0
        total_metric = 0.0
        pbar = tqdm(train_data, desc=f"Train Epoch {epoch+1}", leave=False)
        
        for step, input_dict in enumerate(pbar):
            # 每批次前清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            loss, metric_value = self.train_step(input_dict)
            total_loss += loss
            total_metric += metric_value
            
            pbar.set_postfix({
                "loss": f"{loss:.6f}",
                "avg_loss": f"{total_loss/(step+1):.6f}",
                "metric": f"{metric_value:.6f}"
            })
        
        avg_loss = total_loss / len(train_data)
        avg_metric = total_metric / len(train_data)
        avg_metrics = self.monitor.get_mean_metrics()
        avg_metrics["train_loss"] = avg_loss
        avg_metrics["train_metric"] = avg_metric
        
        self.history["train"]["loss"].append(avg_loss)
        self.history["train"]["metrics"].append(avg_metrics)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            avg_metrics["lr"] = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(f"\n📈 训练轮次 {epoch+1} 平均指标:")
        self.logger.info(self.monitor.format_metrics(avg_metrics))
        self.logger.info(f"📉 训练轮次 {epoch+1} 平均损失: {avg_loss:.6f}")
        if self.target_metric_node:
            self.logger.info(f"📊 训练轮次 {epoch+1} 平均指标: {avg_metric:.6f}")

    def eval_epoch(self, eval_data: list, epoch: int):
        self.monitor.reset()
        total_loss = 0.0
        total_metric = 0.0
        pbar = tqdm(eval_data, desc=f"Eval Epoch {epoch+1}", leave=False)
        
        for step, input_dict in enumerate(pbar):
            loss, metric_value = self.eval_step(input_dict)
            total_loss += loss
            total_metric += metric_value
            
            pbar.set_postfix({
                "loss": f"{loss:.6f}",
                "avg_loss": f"{total_loss/(step+1):.6f}",
                "metric": f"{metric_value:.6f}"
            })
        
        avg_loss = total_loss / len(eval_data)
        avg_metric = total_metric / len(eval_data)
        avg_metrics = self.monitor.get_mean_metrics()
        avg_metrics["eval_loss"] = avg_loss
        avg_metrics["eval_metric"] = avg_metric
        
        self.history["eval"]["loss"].append(avg_loss)
        self.history["eval"]["metrics"].append(avg_metrics)
        
        if avg_loss < self.history["best_eval_loss"]:
            self.history["best_eval_loss"] = avg_loss
            self.history["best_epoch_loss"] = epoch + 1
            self.save_checkpoint(epoch + 1, is_best_loss=True)
            self.logger.info(f"🏆 找到更佳损失模型！验证损失: {avg_loss:.6f} (Epoch {epoch+1})")
        
        if self.target_metric_node and avg_metric > self.history["best_eval_metric"]:
            self.history["best_eval_metric"] = avg_metric
            self.history["best_epoch_metric"] = epoch + 1
            self.save_checkpoint(epoch + 1, is_best_metric=True)
            self.logger.info(f"🏆 找到更佳指标模型！验证指标: {avg_metric:.6f} (Epoch {epoch+1})")
        
        self.logger.info(f"\n📊 验证轮次 {epoch+1} 平均指标:")
        self.logger.info(self.monitor.format_metrics(avg_metrics))
        self.logger.info(f"📉 验证轮次 {epoch+1} 平均损失: {avg_loss:.6f}")
        if self.target_metric_node:
            self.logger.info(f"📊 验证轮次 {epoch+1} 平均指标: {avg_metric:.6f}")
        self.logger.info(f"🏆 当前最佳验证损失: {self.history['best_eval_loss']:.6f} (Epoch {self.history['best_epoch_loss']})")
        if self.target_metric_node:
            self.logger.info(f"🏆 当前最佳验证指标: {self.history['best_eval_metric']:.6f} (Epoch {self.history['best_epoch_metric']})")
    
    def save_checkpoint(self, epoch: int, is_best_loss: bool = False, is_best_metric: bool = False):
        try:
            node_path = os.path.join(self.save_dir, f"node_epoch_{epoch}.pth")
            edge_path = os.path.join(self.save_dir, f"edge_epoch_{epoch}.pth")
            state_path = os.path.join(self.save_dir, f"train_state_epoch_{epoch}.pth")
            
            updown_node_value(self.mhd_graph.nodes, node_path, mode="down")
            updown_edge_value(self.mhd_graph.edges, edge_path, mode="down")
            
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
                "epoch": epoch,
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }, state_path)
            
            import shutil
            if is_best_loss:
                shutil.copy2(node_path, os.path.join(self.save_dir, "node_best_loss.pth"))
                shutil.copy2(edge_path, os.path.join(self.save_dir, "edge_best_loss.pth"))
                shutil.copy2(state_path, os.path.join(self.save_dir, "train_state_best_loss.pth"))
            
            if is_best_metric:
                shutil.copy2(node_path, os.path.join(self.save_dir, "node_best_metric.pth"))
                shutil.copy2(edge_path, os.path.join(self.save_dir, "edge_best_metric.pth"))
                shutil.copy2(state_path, os.path.join(self.save_dir, "train_state_best_metric.pth"))
            
            self.logger.info(f"✅ 检查点保存完成 (Epoch {epoch})：{self.save_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存检查点失败: {str(e)}")
            raise
    
    def load_checkpoint(self, load_best_loss: bool = True, load_best_metric: bool = False, epoch: int = None):
        try:
            if load_best_loss:
                node_path = os.path.join(self.save_dir, "node_best_loss.pth")
                edge_path = os.path.join(self.save_dir, "edge_best_loss.pth")
                state_path = os.path.join(self.save_dir, "train_state_best_loss.pth")
                self.logger.info("📥 加载最佳损失模型权重")
            elif load_best_metric:
                node_path = os.path.join(self.save_dir, "node_best_metric.pth")
                edge_path = os.path.join(self.save_dir, "edge_best_metric.pth")
                state_path = os.path.join(self.save_dir, "train_state_best_metric.pth")
                self.logger.info("📥 加载最佳指标模型权重")
            else:
                if epoch is None:
                    raise ValueError("未指定加载模式时必须指定epoch")
                node_path = os.path.join(self.save_dir, f"node_epoch_{epoch}.pth")
                edge_path = os.path.join(self.save_dir, f"edge_epoch_{epoch}.pth")
                state_path = os.path.join(self.save_dir, f"train_state_epoch_{epoch}.pth")
                self.logger.info(f"📥 加载Epoch {epoch} 权重")
            
            for path in [node_path, edge_path, state_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"权重文件不存在: {path}")
            
            updown_node_value(self.mhd_graph.nodes, node_path, mode="up")
            updown_edge_value(self.mhd_graph.edges, edge_path, mode="up")
            
            state = torch.load(state_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])
            
            if "history" in state:
                self.history["best_eval_loss"] = state["history"]["best_eval_loss"]
                self.history["best_epoch_loss"] = state["history"]["best_epoch_loss"]
                self.history["best_eval_metric"] = state["history"]["best_eval_metric"]
                self.history["best_epoch_metric"] = state["history"]["best_epoch_metric"]
            
            if self.lr_scheduler and "lr_scheduler" in state and state["lr_scheduler"] is not None:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            
            self.logger.info("✅ 权重加载完成")
            
        except Exception as e:
            self.logger.error(f"❌ 加载权重失败: {str(e)}")
            raise
    
    def save_training_history(self):
        history_path = os.path.join(self.save_dir, "training_history.json")
        serializable_history = {}
        for phase in ["train", "eval"]:
            serializable_history[phase] = {
                "loss": [float(l) for l in self.history[phase]["loss"]],
                "metrics": []
            }
            for metrics in self.history[phase]["metrics"]:
                serializable_metrics = {k: float(v) for k, v in metrics.items()}
                serializable_history[phase]["metrics"].append(serializable_metrics)
        
        serializable_history["best_eval_loss"] = float(self.history["best_eval_loss"])
        serializable_history["best_epoch_loss"] = self.history["best_epoch_loss"]
        serializable_history["best_eval_metric"] = float(self.history["best_eval_metric"])
        serializable_history["best_epoch_metric"] = self.history["best_epoch_metric"]
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, ensure_ascii=False, indent=4)
        
        self.logger.info(f"📝 训练历史已保存至: {history_path}")
    
    def train(self, train_data: list, eval_data: list, epochs: int = 10):
        self.logger.info("="*80)
        self.logger.info("🚀 开始MHD网络训练（修复版）")
        self.logger.info(f"🔢 训练轮次: {epochs}")
        self.logger.info(f"📊 训练数据量: {len(train_data)} 批次")
        self.logger.info(f"📊 验证数据量: {len(eval_data)} 批次")
        self.logger.info("="*80)
        
        try:
            for epoch in range(epochs):
                self.logger.info("\n" + "-"*80)
                self.logger.info(f"📅 开始训练轮次 {epoch+1}/{epochs}")
                
                self.train_epoch(train_data, epoch)
                self.eval_epoch(eval_data, epoch)
            
            self.save_checkpoint(epochs)
            self.save_training_history()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("🎉 训练完成！")
            self.logger.info(f"🏆 最佳验证损失: {self.history['best_eval_loss']:.6f} (Epoch {self.history['best_epoch_loss']})")
            if self.target_metric_node:
                self.logger.info(f"🏆 最佳验证指标: {self.history['best_eval_metric']:.6f} (Epoch {self.history['best_epoch_metric']})")
            self.logger.info(f"📝 所有结果已保存至: {self.save_dir}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ 训练过程出错: {str(e)}", exc_info=True)
            raise

# ===================== 优化器创建函数（保持不变） =====================
def create_optimizer(
    mhd_graph,
    edge_optim_config: dict = None,
    default_optimizer_type: str = "adam",
    default_lr: float = 0.001,
    default_weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    edge_optim_config = edge_optim_config or {}
    param_groups = []
    
    # 为指定边创建自定义参数组
    for edge_name, config in edge_optim_config.items():
        edge = get_obj_by_name(edge_name, mhd_graph.edges)
        if edge is None:
            warnings.warn(f"边 {edge_name} 不存在，跳过自定义配置")
            continue
        
        edge_params = []
        for op in edge.value:
            if isinstance(op, nn.Module):
                edge_params.extend([p for p in op.parameters() if p.requires_grad])
        
        if edge_params:
            param_groups.append({
                "params": edge_params,
                "lr": config.get("lr", default_lr),
                "weight_decay": config.get("weight_decay", default_weight_decay),
                "name": edge_name
            })
    
    # 收集剩余参数
    processed_params = set()
    for group in param_groups:
        for p in group["params"]:
            processed_params.add(id(p))
    
    remaining_params = []
    for p in mhd_graph.parameters():
        if p.requires_grad and id(p) not in processed_params:
            remaining_params.append(p)
    
    if remaining_params:
        param_groups.append({
            "params": remaining_params,
            "lr": default_lr,
            "weight_decay": default_weight_decay,
            "name": "default"
        })
    
    # 创建优化器
    optimizer_map = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW
    }
    
    if default_optimizer_type.lower() not in optimizer_map:
        raise ValueError(f"不支持的优化器: {default_optimizer_type}")
    
    # 添加默认参数
    if default_optimizer_type.lower() == "adam":
        kwargs.setdefault("betas", (0.9, 0.999))
        kwargs.setdefault("eps", 1e-8)
    elif default_optimizer_type.lower() == "sgd":
        kwargs.setdefault("momentum", 0.9)
        kwargs.setdefault("nesterov", True)
    
    optimizer = optimizer_map[default_optimizer_type.lower()](param_groups,** kwargs)
    
    logger = logging.getLogger("mhd_train")
    logger.info(f"✅ 优化器创建完成: {default_optimizer_type.upper()}")
    logger.info(f"📋 参数组配置:")
    for i, group in enumerate(param_groups):
        logger.info(f"  组{i+1}: {group.get('name', 'unknown')} - lr={group['lr']}, weight_decay={group['weight_decay']}")
    
    return optimizer
