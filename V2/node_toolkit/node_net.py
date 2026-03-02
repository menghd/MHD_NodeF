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

    # 节点相关工具函数
    def mhd_node2obj(self, node_id: int, node_list: Set['MHD_Node']) -> Optional['MHD_Node']:
        """通过ID获取节点对象"""
        for node in node_list:
            if node.id == node_id:
                return node
        return None

    def mhd_name2obj(self, name: str, node_list: Set['MHD_Node']) -> Optional['MHD_Node']:
        """通过名称获取节点对象"""
        for node in node_list:
            if node.name == name:
                return node
        return None

    def mhd_apply_head_func(self, tensor: torch.Tensor) -> torch.Tensor:
        """应用节点头函数"""
        head_funcs = {
            "share": lambda t: t.clone(memory_format=torch.contiguous_format),
        }
        func_name = self.func.get("head", "share")
        return head_funcs[func_name](tensor)

    def mhd_apply_tail_func(self, tensors: List[torch.Tensor]) -> torch.Tensor:
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

    # 边相关工具函数
    def mhd_edge2obj(self, edge_id: int, edge_list: Set['MHD_Edge']) -> Optional['MHD_Edge']:
        """通过ID获取边对象"""
        for edge in edge_list:
            if edge.id == edge_id:
                return edge
        return None

    def mhd_name2obj(self, name: str, edge_list: Set['MHD_Edge']) -> Optional['MHD_Edge']:
        """通过名称获取边对象"""
        for edge in edge_list:
            if edge.name == name:
                return edge
        return None

    def mhd_apply_in_func(self, tensors: List[torch.Tensor], sorted_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """应用边输入函数"""
        def mhd_concat(ts, sp):
            sorted_ts = [ts[i] for i, _ in sp if i < len(ts)]
            return torch.cat(sorted_ts, dim=1)

        def mhd_matmul(ts, sp):
            sorted_ts = [ts[i] for i, _ in sp if i < len(ts)]
            if len(sorted_ts) != 2:
                raise ValueError(f"Matmul需要2个输入张量，实际输入 {len(sorted_ts)}")
            return torch.matmul(*sorted_ts)

        in_funcs = {
            "concat": mhd_concat,
            "matmul": mhd_matmul,
        }
        func_name = self.func.get("in", "concat")
        return in_funcs[func_name](tensors, sorted_pairs)

    def mhd_apply_out_func(self, x: torch.Tensor, sorted_pairs: List[Tuple[int, int]], node_channels: List[int]) -> List[torch.Tensor]:
        """应用边输出函数"""
        def mhd_split(x, sp, nc):
            sorted_nodes = sp
            sorted_indices = [p[0] for p in sorted_nodes if p[0] < len(nc)]
            sorted_sizes = [nc[i] for i in sorted_indices]
            split_ts = torch.split(x, sorted_sizes, dim=1)
            tensor_map = {idx: t for idx, t in zip(sorted_indices, split_ts)}
            
            result = []
            for i in range(len(nc)):
                result.append(tensor_map.get(i, torch.zeros(x.shape[0], nc[i], device=x.device, dtype=x.dtype)))
            return result

        def mhd_svd(x, sp, nc):
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

        def mhd_lu(x, sp, nc):
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
            "split": mhd_split,
            "svd": mhd_svd,
            "lu": mhd_lu,
        }
        func_name = self.func.get("out", "split")
        return out_funcs[func_name](x, sorted_pairs, node_channels)

@dataclass
class MHD_Topo:
    """超图拓扑类"""
    type: str
    value: torch.Tensor
    
    def mhd_topo2obj(self, type_name: str, topo_list: Set['MHD_Topo']) -> Optional['MHD_Topo']:
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

    def mhd_validate_topo(self, num_edges: int, num_nodes: int) -> None:
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
        self.op_names = []

        for op in operations:
            self.op_names.append(self.mhd_extract_operation_name(op))

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

    def mhd_extract_operation_name(self, op: Union[str, nn.Module]) -> str:
        """提取操作名称"""
        if isinstance(op, nn.Module):
            op_name = op.__class__.__name__
        elif isinstance(op, str):
            op_name = re.sub(r'\(.*\)', '', op).strip()
        else:
            op_name = str(op)

        op_name = op_name.replace("torch.nn.modules.", "").replace("torch.nn.", "")
        return op_name

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
        self.node_id2obj = {node.id: node for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        
        # 转换拓扑为tensor类型
        self.topos = set()
        for topo in topos:
            if isinstance(topo.value, list):
                tensor_value = torch.tensor(topo.value, dtype=torch.int64, device=self.device)
            else:
                tensor_value = topo.value.to(self.device)
            self.topos.add(MHD_Topo(type=topo.type, value=tensor_value))

        # 验证拓扑维度
        self.mhd_validate_all_topo()
        # 拓扑排序
        self.sorted_node_ids = self.mhd_topological_sort()
        print(f"✅ 拓扑排序完成: {[self.node_id2obj[nid].name for nid in self.sorted_node_ids]}")

        # 初始化边操作网络
        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.value, device=self.device)

        # 节点值初始化
        self.node_values = nn.ParameterDict()
        for node in nodes:
            node_value = node.value.to(self.device)
            param = nn.Parameter(node_value, requires_grad=True)
            self.node_values[str(node.id)] = param

        # 预缓存所有边的拓扑排序结果
        self.edge_sorted_pairs = {}
        for edge_id in self.edge_id2obj.keys():
            self.edge_sorted_pairs[edge_id] = self.mhd_sort_nodes_by_topo_value(edge_id)

    @property
    def node_name2id(self):
        """节点名称到ID的映射"""
        return {v.name: k for k, v in self.node_id2obj.items()}

    def mhd_validate_all_topo(self) -> None:
        """验证所有拓扑维度"""
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)
        for topo in self.topos:
            topo.mhd_validate_topo(num_edges, num_nodes)

    def mhd_sort_nodes_by_topo_value(self, edge_id: int = 0) -> List[Tuple[int, int]]:
        """按sort类型的拓扑值排序节点"""
        sort_topo = None
        for topo in self.topos:
            if topo.type == "sort":
                sort_topo = topo
                break
        
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

    def mhd_topological_sort(self) -> List[int]:
        """超图节点拓扑排序"""
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_node_ids = {node.id for node in self.node_id2obj.values()}
        
        # 使用role类型的拓扑进行排序
        role_topo = None
        for topo in self.topos:
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

    def mhd_validate_node_group_consistency(self, node_group: Set[str], sub_node_map: Dict[str, Tuple[str, int, MHD_Node]]) -> None:
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

    def mhd_get_merged_node_value(self, node_group: Set[str], sub_node_map: Dict[str, Tuple[str, int, MHD_Node]]) -> torch.Tensor:
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

    def mhd_generate_mermaid(self) -> str:
        """生成Mermaid拓扑可视化代码"""
        mermaid = [
            "graph TD",
            "",
            "    classDef nodeStyle fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
            "    classDef edgeStyle fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
            "",
        ]

        # 获取role类型拓扑
        role_topo = None
        for topo in self.topos:
            if topo.type == "role":
                role_topo = topo
                break

        # 添加节点样式
        for node_id, node in self.node_id2obj.items():
            mermaid.append(f"    {node.name}:::nodeStyle")
        
        # 添加边样式和连接关系
        if role_topo:
            for edge_id, edge in self.edge_id2obj.items():
                edge_name = edge.name
                if edge_id < role_topo.value.shape[0]:
                    edge_row = role_topo.value[edge_id]
                
                    mermaid.append(f"    {edge_name}:::edgeStyle")
                    
                    head_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() < 0]
                    tail_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() > 0]
                    head_node_names = [self.node_id2obj[nid].name for nid in head_node_ids if nid in self.node_id2obj]
                    tail_node_names = [self.node_id2obj[nid].name for nid in tail_node_ids if nid in self.node_id2obj]
                    
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
        role_topo = None
        for topo in self.topos:
            if topo.type == "role":
                role_topo = topo
                break
        
        if role_topo is not None and role_topo.value.numel() > 0:
            for edge_id in self.edge_id2obj.keys():
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
                edge = self.edge_id2obj[edge_id]
                edge_net = self.edge_nets[edge.name]
                
                # 获取头节点
                edge_row = role_topo.value[edge_id] if (role_topo and edge_id < role_topo.value.shape[0]) else []
                head_mask = [val.item() < 0 for val in edge_row] if edge_row.numel() > 0 else []
                head_node_ids = [i for i, val in enumerate(head_mask) if val]

                # 处理头节点张量
                head_tensors = []
                for node_id in head_node_ids:
                    if str(node_id) in self.node_values:
                        node = self.node_id2obj[node_id]
                        head_tensor = node.mhd_apply_head_func(self.node_values[str(node_id)])
                        head_tensors.append(head_tensor)

                if not head_tensors:
                    continue

                # 边输入处理
                sorted_pairs = self.edge_sorted_pairs.get(edge_id, [])
                edge_input = edge.mhd_apply_in_func(head_tensors, sorted_pairs)

                # 边操作前向传播
                edge_output = edge_net(edge_input)

                # 获取尾节点
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

                # 边输出处理
                tail_tensors = edge.mhd_apply_out_func(edge_output, sorted_pairs, tail_node_channels)

                # 节点值更新
                for idx, node_id in enumerate(tail_node_ids):
                    if idx < len(tail_tensors) and node_id in self.node_id2obj and str(node_id) in self.node_values:
                        node = self.node_id2obj[node_id]
                        tensor = tail_tensors[idx]
                        
                        # 确保张量形状匹配
                        if tensor.shape[1] != self.node_values[str(node_id)].shape[1]:
                            tensor = nn.functional.adaptive_avg_pool1d(
                                tensor.unsqueeze(1), 
                                self.node_values[str(node_id)].shape[1]
                            ).squeeze(1)
                        
                        # 聚合更新节点值
                        agg_tensor = node.mhd_apply_tail_func(
                            [self.node_values[str(node_id)], tensor]
                        )
                        self.node_values[str(node_id)] = nn.Parameter(agg_tensor, requires_grad=True)

        # 返回节点名称到张量的映射
        return {
            self.node_id2obj[int(node_id)].name: tensor
            for node_id, tensor in self.node_values.items()
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
            for sub_node_id, sub_node in hdnet.node_id2obj.items():
                global_node_name = f"{suffix}::{sub_node.name}"
                sub_node_map[global_node_name] = (suffix, sub_node_id, sub_node)
                all_sub_node_names.add(global_node_name)
            
            for sub_edge_id, sub_edge in hdnet.edge_id2obj.items():
                global_edge_name = f"{suffix}::{sub_edge.name}"
                sub_edge_map[global_edge_name] = (suffix, sub_edge_id, sub_edge)

        # 步骤2：处理节点合并
        node_id_counter = 0
        merged_node_map = {}
        sub2global_node = {}
        
        merged_node_names = set()
        for group in node_group:
            self.mhd_validate_node_group_consistency(group, sub_node_map)
            
            sorted_node_names = sorted(
                group,
                key=lambda x: next((i for i, (suffix, _) in enumerate(hdnet_list) if x.startswith(f"{suffix}::")), 999)
            )
            merged_name = "-".join(sorted_node_names)
            merged_node_names.update(sorted_node_names)
            
            merged_value = self.mhd_get_merged_node_value(group, sub_node_map).to(device)
            
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
                
                sub_topo = None
                for t in hdnet.topos:
                    if t.type == topo_type:
                        sub_topo = t
                        break
                
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

# ===================== 示例和验证函数 =====================
def example_mhdnet2():
    """MHDNet示例"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"✅ 使用设备: {device}")

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

    # 扩展HDNet3
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
    
    # 更新节点/边
    updated_nodes_net3 = set(hdnet3.node_id2obj.values())
    updated_nodes_net3.add(new_node_E3)
    updated_edges_net3 = set(hdnet3.edge_id2obj.values())
    updated_edges_net3.add(new_edge_obj)
    
    # 更新拓扑
    role_topo_net3 = None
    for t in hdnet3.topos:
        if t.type == "role":
            role_topo_net3 = t
            break
    sort_topo_net3 = None
    for t in hdnet3.topos:
        if t.type == "sort":
            sort_topo_net3 = t
            break
    
    # 扩展role拓扑
    updated_role_value = torch.cat([
        torch.cat([role_topo_net3.value, torch.zeros(role_topo_net3.value.shape[0], 1, device=device, dtype=torch.int64)], dim=1),
        torch.tensor([[-1, 0, 1]], device=device, dtype=torch.int64)
    ], dim=0)
    
    # 扩展sort拓扑
    updated_sort_value = torch.cat([
        torch.cat([sort_topo_net3.value, torch.zeros(sort_topo_net3.value.shape[0], 1, device=device, dtype=torch.int64)], dim=1),
        torch.tensor([[1, 0, 2]], device=device, dtype=torch.int64)
    ], dim=0)
    
    updated_topos_net3 = {
        MHD_Topo(type="role", value=updated_role_value),
        MHD_Topo(type="sort", value=updated_sort_value)
    }
    
    # 重新创建HDNet3
    hdnet3 = HDNet(nodes=updated_nodes_net3, edges=updated_edges_net3, topos=updated_topos_net3, device=device)

    # 构建全局MHDNet
    hdnet_list = [
        ("net1", hdnet1),
        ("net2", hdnet2),
        ("net3test", hdnet3)
    ]

    # 节点合并组
    node_group = (
        {"net1::A1", "net2::A2"},
        {"net1::B1", "net2::B2"},
        {"net2::C2", "net3test::C3"},
        {"net1::D1", "net3test::D3"},
        {"net3test::E3"},
    )

    # 创建全局超图
    mhdnet = MHDNet(
        hdnet_list=hdnet_list,
        node_group=node_group,
        device=device
    )

    # 加载输入数据
    input_tensor = torch.randn(1, 3, 8, 8, 8, device=device, dtype=dtype)
    print(f"\n✅ 输入张量形状: {input_tensor.shape}")

    # 更新输入节点
    target_node_name = "net1::A1-net2::A2"
    input_node = None
    for node in mhdnet.node_id2obj.values():
        if node.name == target_node_name:
            input_node = node
            break
    input_node.name = "input"
    mhdnet.node_values[str(input_node.id)] = nn.Parameter(input_tensor, requires_grad=True)

    # 生成可视化代码
    print("\n📊 MHD NodeF 拓扑可视化:")
    print("="*80)
    mhdnet.mhd_generate_mermaid()
    print("="*80 + "\n")

    # 运行前向传播
    print("🚀 执行MHDNet前向传播...")
    all_features = mhdnet.forward()

    # 打印结果
    print("\n✅ 前向传播完成！")
    print(f"\n📈 全局节点总数: {len(all_features)}")
    total_params = sum(p.numel() for p in mhdnet.parameters())
    print(f"📈 模型总参数: {total_params:,}")
    
    return mhdnet

def verify_gradient(model):
    """验证梯度反传"""
    all_features = model.forward()
    if "net1::D1-net3test::D3" in all_features:
        output_tensor = all_features["net1::D1-net3test::D3"]
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
        print("\n❌ 无法验证梯度：输出节点不存在")

# ===================== 主执行入口 =====================
if __name__ == "__main__":
    model = example_mhdnet2()
    verify_gradient(model)
