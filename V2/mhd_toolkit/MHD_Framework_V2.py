# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Framework (MHD) - Version 2.0
Author: Souray Meng (孟号丁)
Core Framework: Hypergraph-based computational graph with state separation
License: MIT
"""

import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import re

# 全局配置：抑制无关警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_string_operation(op_str: str, x: torch.Tensor) -> torch.Tensor:
    """
    解析并执行字符串形式的张量操作
    
    Args:
        op_str: 操作字符串，如 "relu()" 或 "mean(dim=1)"
        x: 输入张量
    
    Returns:
        操作后的张量
    """
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


# ===================== 核心超图框架类 =====================

@dataclass
class MHD_Node:
    """
    超图节点类 - 状态机设计
    
    特性：
    1. 分离初始状态(initial_state)和当前状态(current_state)
    2. 支持状态重置和更新
    3. 包含头尾变换函数用于数据流控制
    
    Attributes:
        id: 节点唯一标识符
        name: 节点名称
        initial_state: 初始状态张量（用于重置）
        current_state: 当前状态张量（前向传播中更新）
        func: 变换函数配置，包含head和tail函数类型
    """
    id: int
    name: str
    initial_state: torch.Tensor
    current_state: torch.Tensor
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"})

    def __post_init__(self):
        """初始化验证：确保设备一致性"""
        if self.initial_state.device != self.current_state.device:
            raise ValueError(
                f"节点 {self.name} 初始状态和当前状态设备不一致: "
                f"{self.initial_state.device} vs {self.current_state.device}"
            )
        
        if self.initial_state.shape != self.current_state.shape:
            raise ValueError(
                f"节点 {self.name} 初始状态和当前状态形状不匹配: "
                f"{self.initial_state.shape} vs {self.current_state.shape}"
            )

    def __hash__(self):
        """基于ID的哈希函数"""
        return hash(self.id)

    def __eq__(self, other):
        """基于ID的相等判断"""
        if not isinstance(other, MHD_Node):
            return False
        return self.id == other.id

    def reset(self) -> 'MHD_Node':
        """
        重置当前状态为初始状态
        
        注意：创建新张量以切断计算图依赖
        
        Returns:
            重置后的节点自身
        """
        self.current_state = self.initial_state.clone(memory_format=torch.contiguous_format)
        return self

    def update_initial(self, new_tensor: torch.Tensor) -> 'MHD_Node':
        """
        更新初始状态（用于数据加载或外部注入）
        
        Args:
            new_tensor: 新的初始状态张量
        
        Returns:
            更新后的节点自身
        """
        self.initial_state = new_tensor
        self.current_state = new_tensor.clone(memory_format=torch.contiguous_format)
        return self

    def to_device(self, device: torch.device) -> 'MHD_Node':
        """
        将节点状态迁移到指定设备
        
        Args:
            device: 目标计算设备
        
        Returns:
            设备迁移后的节点自身
        """
        if self.initial_state.device != device:
            self.initial_state = self.initial_state.to(device, non_blocking=True)
        if self.current_state.device != device:
            self.current_state = self.current_state.to(device, non_blocking=True)
        return self

    def get_head_transformed_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        应用节点头函数进行数据变换
        
        Args:
            tensor: 输入张量
        
        Returns:
            变换后的张量
        """
        head_funcs = {
            "share": lambda t: t.clone(memory_format=torch.contiguous_format),
        }
        return head_funcs[self.func.get("head", "share")](tensor)

    def get_tail_transformed_tensor(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        应用节点尾函数进行数据聚合
        
        Args:
            tensors: 输入张量列表
        
        Returns:
            聚合后的张量
        """
        tail_funcs = {
            "sum": lambda ts: torch.stack(ts).sum(dim=0),
            "avg": lambda ts: torch.stack(ts).mean(dim=0),
            "max": lambda ts: torch.stack(ts).max(dim=0)[0],
            "min": lambda ts: torch.stack(ts).min(dim=0)[0],
            "mul": lambda ts: torch.stack(ts).prod(dim=0)
        }
        return tail_funcs[self.func.get("tail", "sum")](tensors)


@dataclass
class MHD_Edge:
    """
    超图边类 - 可学习模块的载体
    
    特性：
    1. 包含顺序操作序列，支持nn.Module和字符串操作
    2. 支持多种输入输出变换函数
    3. 可学习参数通过nn.Module封装
    
    Attributes:
        id: 边唯一标识符
        name: 边名称
        sequential_operation: 顺序操作序列
        func: 变换函数配置，包含输入(in)和输出(out)函数类型
    """
    id: int
    name: str
    sequential_operation: List[Union[str, nn.Module]]
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"})

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
        for idx, op in enumerate(self.sequential_operation):
            if isinstance(op, nn.Module):
                self.sequential_operation[idx] = op.to(device, non_blocking=True)
        return self

    def get_in_transformed_tensor(self, tensors: List[torch.Tensor], 
                                  sorted_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        应用边输入函数进行数据变换
        
        Args:
            tensors: 输入张量列表
            sorted_pairs: 排序对列表，格式为[(原始索引, 排序值), ...]
        
        Returns:
            变换后的张量
        """
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

    def get_out_transformed_tensors(self, x: torch.Tensor, 
                                    sorted_pairs: List[Tuple[int, int]], 
                                    node_channels: List[int]) -> List[torch.Tensor]:
        """
        应用边输出函数进行数据分发
        
        Args:
            x: 输入张量
            sorted_pairs: 排序对列表
            node_channels: 目标节点通道数列表
        
        Returns:
            分发后的张量列表
        """
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
    """
    超图拓扑类 - 矩阵属性分离
    
    特性：
    1. 角色矩阵(role_matrix)定义连接关系(-1/0/1)
    2. 排序矩阵(sort_matrix)定义参数传递顺序
    3. 双矩阵设计支持复杂拓扑结构
    
    Attributes:
        role_matrix: 角色矩阵，形状为(边数, 节点数)
        sort_matrix: 排序矩阵，形状为(边数, 节点数)
    """
    role_matrix: torch.Tensor
    sort_matrix: torch.Tensor

    def __post_init__(self):
        """初始化验证：确保矩阵维度和设备一致性"""
        if self.role_matrix.shape != self.sort_matrix.shape:
            raise ValueError(
                f"拓扑矩阵形状不匹配: "
                f"role_matrix {self.role_matrix.shape} vs sort_matrix {self.sort_matrix.shape}"
            )
        
        if self.role_matrix.device != self.sort_matrix.device:
            raise ValueError(
                f"拓扑矩阵设备不一致: "
                f"role_matrix {self.role_matrix.device} vs sort_matrix {self.sort_matrix.device}"
            )

    def to_device(self, device: torch.device) -> 'MHD_Topo':
        """
        将拓扑矩阵迁移到指定设备
        
        Args:
            device: 目标计算设备
        
        Returns:
            设备迁移后的拓扑自身
        """
        if self.role_matrix.device != device:
            self.role_matrix = self.role_matrix.to(device, non_blocking=True)
        if self.sort_matrix.device != device:
            self.sort_matrix = self.sort_matrix.to(device, non_blocking=True)
        return self

    def get_topo_value(self, edge_id: int, node_id: int, matrix_type: str = "role") -> int:
        """
        获取指定边和节点的拓扑值
        
        Args:
            edge_id: 边索引
            node_id: 节点索引
            matrix_type: 矩阵类型，'role'或'sort'
        
        Returns:
            拓扑值，如果索引越界返回0
        """
        matrix = self.role_matrix if matrix_type == "role" else self.sort_matrix
        if 0 <= edge_id < matrix.shape[0] and 0 <= node_id < matrix.shape[1]:
            return int(matrix[edge_id, node_id].item())
        return 0

    def to_list(self, matrix_type: str = "role") -> List[List[int]]:
        """
        转换为列表形式
        
        Args:
            matrix_type: 矩阵类型，'role'或'sort'
        
        Returns:
            矩阵的Python列表表示
        """
        matrix = self.role_matrix if matrix_type == "role" else self.sort_matrix
        return matrix.tolist()

    def __hash__(self):
        """基于矩阵内容的哈希函数"""
        return hash((tuple(self.role_matrix.flatten().tolist()), 
                     tuple(self.sort_matrix.flatten().tolist())))

    def __eq__(self, other):
        """基于矩阵内容的相等判断"""
        if not isinstance(other, MHD_Topo):
            return False
        return (torch.equal(self.role_matrix, other.role_matrix) and 
                torch.equal(self.sort_matrix, other.sort_matrix))

    def validate_topo(self, num_edges: int, num_nodes: int) -> None:
        """
        验证拓扑矩阵维度
        
        Args:
            num_edges: 预期的边数
            num_nodes: 预期的节点数
        
        Raises:
            ValueError: 当维度不匹配时
        """
        for matrix, name in [(self.role_matrix, "role"), (self.sort_matrix, "sort")]:
            if matrix.shape[0] != num_edges:
                raise ValueError(f"{name}拓扑边维度不匹配: 预期{num_edges}，实际{matrix.shape[0]}")
            if matrix.shape[1] != num_nodes:
                raise ValueError(f"{name}拓扑节点维度不匹配: 预期{num_nodes}，实际{matrix.shape[1]}")


class MHD_Graph(nn.Module):
    """
    多超图动态框架核心类 - Version 2.0
    
    特性：
    1. 基于状态机的超图计算框架
    2. 分离初始状态和当前状态，消除状态污染
    3. 支持拓扑驱动的自动微分
    4. 统一设备管理，确保计算一致性
    5. 内置索引系统，支持高效查找
    
    Author: Souray Meng (孟号丁)
    
    Attributes:
        device: 统一计算设备
        nodes: 节点集合
        edges: 边集合
        topo: 拓扑对象
    """
    
    def __init__(self, nodes: Set[MHD_Node], edges: Set[MHD_Edge], topos: Set[MHD_Topo], 
                 device: torch.device = None):
        """
        初始化MHD图
        
        Args:
            nodes: 节点集合
            edges: 边集合
            topos: 拓扑集合（应只包含一个元素）
            device: 计算设备，默认为CUDA(可用)或CPU
        """
        super().__init__()
        
        # 统一设备配置：确保所有组件在相同设备上
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心超图对象
        self.nodes = nodes
        self.edges = edges
        
        # 拓扑验证：确保只有一个拓扑对象
        if len(topos) != 1:
            warnings.warn(f"拓扑集合包含 {len(topos)} 个元素，预期为1，将使用第一个")
        self.topo = next(iter(topos)) if topos else None
        
        # 统一设备：确保所有组件都在指定设备上
        self._unify_device()
        
        # 建立索引系统
        self._node_by_id: Dict[int, MHD_Node] = {}
        self._node_by_name: Dict[str, MHD_Node] = {}
        self._edge_by_id: Dict[int, MHD_Edge] = {}
        self._edge_by_name: Dict[str, MHD_Edge] = {}
        self._build_indices()
        
        # 参数注册容器 - 仅注册边中的可学习模块
        self.edge_module_map = nn.ModuleDict()
        
        # 验证拓扑维度
        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        if self.topo:
            self.topo.validate_topo(num_edges, num_nodes)
        
        # 拓扑排序和参数注册
        self.compact_topological_sort()
        self._register_all_params()
        
        print(f"✅ MHD图初始化完成 | 设备: {self.device} | 节点: {num_nodes} | 边: {num_edges}")
    
    def _unify_device(self) -> None:
        """
        统一所有组件的计算设备
        
        确保节点、边、拓扑都位于相同的计算设备上
        """
        # 迁移拓扑到统一设备
        if self.topo:
            self.topo = self.topo.to_device(self.device)
        
        # 迁移所有节点到统一设备
        for node in self.nodes:
            node.to_device(self.device)
        
        # 迁移所有边到统一设备
        for edge in self.edges:
            edge.to_device(self.device)
    
    def _build_indices(self) -> None:
        """
        构建节点和边的索引字典
        
        建立ID和名称到对象的映射，支持O(1)查找
        """
        # 清空索引
        self._node_by_id.clear()
        self._node_by_name.clear()
        self._edge_by_id.clear()
        self._edge_by_name.clear()
        
        # 构建节点索引
        for node in self.nodes:
            if node.id in self._node_by_id:
                warnings.warn(f"节点ID重复: {node.id}，后出现的将覆盖前者")
            if node.name in self._node_by_name:
                warnings.warn(f"节点名称重复: {node.name}，后出现的将覆盖前者")
            
            self._node_by_id[node.id] = node
            self._node_by_name[node.name] = node
        
        # 构建边索引
        for edge in self.edges:
            if edge.id in self._edge_by_id:
                warnings.warn(f"边ID重复: {edge.id}，后出现的将覆盖前者")
            if edge.name in self._edge_by_name:
                warnings.warn(f"边名称重复: {edge.name}，后出现的将覆盖前者")
            
            self._edge_by_id[edge.id] = edge
            self._edge_by_name[edge.name] = edge
    
    def update_indices(self, new_nodes: Set[MHD_Node] = None, new_edges: Set[MHD_Edge] = None) -> None:
        """
        更新索引（当节点或边集合发生变化时调用）
        
        Args:
            new_nodes: 新增节点集合
            new_edges: 新增边集合
        """
        if new_nodes:
            for node in new_nodes:
                self._node_by_id[node.id] = node
                self._node_by_name[node.name] = node
        
        if new_edges:
            for edge in new_edges:
                self._edge_by_id[edge.id] = edge
                self._edge_by_name[edge.name] = edge
    
    def get_node_by_id(self, node_id: int) -> Optional[MHD_Node]:
        """
        通过ID获取节点 (O(1)查找)
        
        Args:
            node_id: 节点ID
        
        Returns:
            节点对象，如果不存在返回None
        """
        return self._node_by_id.get(node_id)
    
    def get_node_by_name(self, node_name: str) -> Optional[MHD_Node]:
        """
        通过名称获取节点 (O(1)查找)
        
        Args:
            node_name: 节点名称
        
        Returns:
            节点对象，如果不存在返回None
        """
        return self._node_by_name.get(node_name)
    
    def get_edge_by_id(self, edge_id: int) -> Optional[MHD_Edge]:
        """
        通过ID获取边 (O(1)查找)
        
        Args:
            edge_id: 边ID
        
        Returns:
            边对象，如果不存在返回None
        """
        return self._edge_by_id.get(edge_id)
    
    def get_edge_by_name(self, edge_name: str) -> Optional[MHD_Edge]:
        """
        通过名称获取边 (O(1)查找)
        
        Args:
            edge_name: 边名称
        
        Returns:
            边对象，如果不存在返回None
        """
        return self._edge_by_name.get(edge_name)
    
    def compact_topological_sort(self) -> 'MHD_Graph':
        """
        压缩版拓扑排序（基于role_matrix）
        
        基于角色矩阵进行边和节点的拓扑排序，确保计算顺序无环
        
        Returns:
            排序后的图自身
        
        Raises:
            ValueError: 当检测到环时
        """
        if self.topo is None:
            self._edge_sequence = sorted([e.id for e in self.edges])
            self._node_sequence = sorted([n.id for n in self.nodes])
            print("✅ 拓扑分析完成（无role矩阵，默认按ID排序）")
            return self

        # 1. 边拓扑排序
        num_edges, num_nodes = self.topo.role_matrix.shape
        edge_deps = defaultdict(set)
        node_to_out_edges = defaultdict(set)
        
        # 第一遍：收集所有节点的出边（输出边）
        node_to_out_edges = defaultdict(set)
        for edge_id in range(num_edges):
            edge_row = self.topo.role_matrix[edge_id]
            for nid in range(num_nodes):
                if edge_row[nid] > 0:
                    node_to_out_edges[nid].add(edge_id)

        # 第二遍：计算每条边的依赖
        edge_deps = defaultdict(set)
        for edge_id in range(num_edges):
            edge_row = self.topo.role_matrix[edge_id]
            for nid in range(num_nodes):
                if edge_row[nid] < 0:
                    edge_deps[edge_id].update(node_to_out_edges.get(nid, set()))
            edge_deps[edge_id].discard(edge_id)

        # 计算入度
        edge_in_degree = {e.id: len(edge_deps.get(e.id, set())) for e in self.edges}
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
            edge_row = self.topo.role_matrix[edge_id]
            head_ids = [nid for nid in range(num_nodes) if edge_row[nid] < 0]
            tail_ids = [nid for nid in range(num_nodes) if edge_row[nid] > 0]
            for tail_id in tail_ids:
                node_deps[tail_id].update(head_ids)

        # 计算节点入度
        node_in_degree = {n.id: len(node_deps.get(n.id, set())) for n in self.nodes}
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
        edge_names = [self.get_edge_by_id(eid).name for eid in edge_sequence]
        node_names = [self.get_node_by_id(nid).name for nid in node_sequence]
        print(f"✅ 拓扑分析完成（基于role矩阵）:")
        print(f" ├─ 边执行序列: {edge_names}")
        print(f" └─ 节点拓扑序列: {node_names}")
        
        return self
    
    def _register_all_params(self) -> 'MHD_Graph':
        """
        仅注册边中的可学习模块
        
        将边中的nn.Module注册到ModuleDict，确保参数能被优化器识别
        
        Returns:
            注册后的图自身
        """
        # 边模块注册
        for edge in sorted(self.edges, key=lambda x: x.id):
            for idx, op in enumerate(edge.sequential_operation):
                if isinstance(op, nn.Module):
                    module_name = f"edge_{edge.name}_op_{idx}"
                    self.edge_module_map[module_name] = op.to(self.device)
                    edge.sequential_operation[idx] = self.edge_module_map[module_name]

        return self

    def sort_nodes_by_topo_value(self, edge_id: int = 0) -> List[Tuple[int, int]]:
        """
        按sort_matrix排序节点
        
        Args:
            edge_id: 边索引
        
        Returns:
            排序后的节点索引和排序值列表
        """
        if self.topo is None:
            return []

        if edge_id >= self.topo.sort_matrix.shape[0]:
            return []

        indexed_nodes = list(enumerate(self.topo.sort_matrix[edge_id].tolist()))
        return sorted(indexed_nodes, key=lambda p: p[1])

    def _execute_edge_operations(self, edge: MHD_Edge, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        执行边的操作序列
        
        Args:
            edge: 边对象
            input_tensor: 输入张量
        
        Returns:
            操作后的张量
        """
        x = input_tensor
        for op in edge.sequential_operation:
            if isinstance(op, nn.Module):
                x = op(x)  # 通过注册的Module，保留计算图
            elif isinstance(op, str):
                x = parse_string_operation(op, x)  # 自定义操作也保留计算图
        return x

    def forward(self) -> 'MHD_Graph':
        """
        拓扑驱动前向传播
        
        按照拓扑排序依次执行边操作，更新节点状态
        
        Returns:
            前向传播后的图自身
        """
        if not self.nodes or not self.edges or self.topo is None:
            return self
        
        num_edges, num_nodes = self.topo.role_matrix.shape

        # 按拓扑序列执行边操作
        for edge_id in self._edge_sequence:
            edge = self.get_edge_by_id(edge_id)
            if not edge or edge_id >= num_edges:
                continue

            # 获取输入/输出节点
            edge_row = self.topo.role_matrix[edge_id]
            head_node_ids = [nid for nid in range(num_nodes) if edge_row[nid].item() < 0]
            tail_node_ids = [nid for nid in range(num_nodes) if edge_row[nid].item() > 0]
            
            if not head_node_ids or not tail_node_ids:
                continue

            # 对入节点和出节点分别按sort_matrix排序
            head_sorted = sorted(head_node_ids, key=lambda nid: self.topo.sort_matrix[edge_id, nid].item())
            tail_sorted = sorted(tail_node_ids, key=lambda nid: self.topo.sort_matrix[edge_id, nid].item())

            # 处理输入张量
            head_tensors = []
            for nid in head_sorted:
                node = self.get_node_by_id(nid)
                if node:
                    head_tensors.append(node.get_head_transformed_tensor(node.current_state))
            
            if not head_tensors:
                continue

            # 构造排序对
            sorted_pairs_head = [(idx, self.topo.sort_matrix[edge_id, nid].item()) for idx, nid in enumerate(head_sorted)]
            edge_input = edge.get_in_transformed_tensor(head_tensors, sorted_pairs_head)

            # 执行边操作
            edge_output = self._execute_edge_operations(edge, edge_input)

            # 处理输出通道
            tail_node_channels = []
            for nid in tail_sorted:
                node = self.get_node_by_id(nid)
                if node:
                    tail_node_channels.append(node.current_state.shape[1])
                else:
                    tail_node_channels.append(0)

            # 构造排序对
            sorted_pairs_tail = [(idx, self.topo.sort_matrix[edge_id, nid].item()) for idx, nid in enumerate(tail_sorted)]
            tail_tensors = edge.get_out_transformed_tensors(edge_output, sorted_pairs_tail, tail_node_channels)

            # 节点特征图更新
            for idx, node_id in enumerate(tail_sorted):
                if idx >= len(tail_tensors):
                    continue
                node = self.get_node_by_id(node_id)
                if not node:
                    continue
                
                tensor = tail_tensors[idx]
                new_value = node.get_tail_transformed_tensor([node.current_state, tensor])
                if new_value.device != self.device:
                    new_value = new_value.to(self.device, non_blocking=True)
                node.current_state = new_value

        return self
    
    def generate_mermaid(self) -> str:
        """
        生成Mermaid图描述
        
        创建图的Mermaid格式描述，用于可视化
        
        Returns:
            Mermaid图描述字符串
        """
        mermaid = [
            "graph TD",
            "",
            " classDef MHD_Node_Style fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
            " classDef MHD_Edge_Style fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
            "",
        ]

        # 添加节点
        for node in sorted(self.nodes, key=lambda x: x.id):
            mermaid.append(f" {node.name}:::MHD_Node_Style")
        
        # 添加边和连接关系
        if self.topo:
            for edge in sorted(self.edges, key=lambda x: x.id):
                edge_name = edge.name
                edge_id = edge.id
                if edge_id < self.topo.role_matrix.shape[0]:
                    edge_row = self.topo.role_matrix[edge_id]
                    mermaid.append(f" {edge_name}:::MHD_Edge_Style")
                    
                    head_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() < 0]
                    tail_node_ids = [nid for nid in range(edge_row.shape[0]) if edge_row[nid].item() > 0]
                    
                    head_node_names = [self.get_node_by_id(nid).name for nid in head_node_ids if self.get_node_by_id(nid)]
                    tail_node_names = [self.get_node_by_id(nid).name for nid in tail_node_ids if self.get_node_by_id(nid)]
                    
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
        """
        合并多个子图为一个超图
        
        支持节点分组合并，创建独立的新图实例
        
        Args:
            graph_list: 子图列表，格式为[(后缀, 子图), ...]
            node_group: 节点分组，每组内的节点将被合并
            device: 目标计算设备
        
        Returns:
            合并后的新图实例
        """
        sub2global_node_id = {}
        sub2global_edge_id = {}
        
        sub_node_map = {}
        sub_edge_map = {}
        
        # 收集所有子图的节点和边
        for suffix, graph in graph_list:
            for node in sorted(graph.nodes, key=lambda x: x.id):
                global_node_name = f"{suffix}{node.name}"
                sub_node_map[global_node_name] = (suffix, node.id, node)
            for edge in sorted(graph.edges, key=lambda x: x.id):
                global_edge_name = f"{suffix}{edge.name}"
                sub_edge_map[global_edge_name] = (suffix, edge.id, edge)

        # 节点处理
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
                merged_initial = group_node_objs[0].initial_state.clone(memory_format=torch.contiguous_format)
                merged_current = group_node_objs[0].current_state.clone(memory_format=torch.contiguous_format)
            else:
                merged_initial = torch.stack([n.initial_state for n in group_node_objs]).mean(dim=0)
                merged_current = torch.stack([n.current_state for n in group_node_objs]).mean(dim=0)
            
            base_node = group_node_objs[0]
            merged_node = MHD_Node(
                id=node_id_counter,
                name=merged_name,
                initial_state=merged_initial,
                current_state=merged_current,
                func=base_node.func.copy()
            )
            merged_nodes.add(merged_node)
            
            for node_name in sorted_node_names:
                s, sn_id, _ = sub_node_map[node_name]
                sub2global_node_id[(s, sn_id)] = node_id_counter
            node_id_counter += 1

        # 独立节点
        unmerged_node_names = set(sub_node_map.keys()) - merged_node_names
        for node_name in sorted(unmerged_node_names):
            suffix, sub_node_id, sub_node = sub_node_map[node_name]
            
            unmerged_node = MHD_Node(
                id=node_id_counter,
                name=node_name,
                initial_state=sub_node.initial_state.clone(memory_format=torch.contiguous_format),
                current_state=sub_node.current_state.clone(memory_format=torch.contiguous_format),
                func=sub_node.func.copy()
            )
            merged_nodes.add(unmerged_node)
            
            sub2global_node_id[(suffix, sub_node_id)] = node_id_counter
            node_id_counter += 1

        # 边处理
        edge_id_counter = 0
        merged_edges = set()
        for global_edge_name, (suffix, sub_edge_id, sub_edge) in sorted(sub_edge_map.items(), key=lambda x: x[1][1]):
            merged_edge = MHD_Edge(
                id=edge_id_counter,
                name=global_edge_name,
                sequential_operation=sub_edge.sequential_operation.copy(),
                func=sub_edge.func.copy()
            )
            merged_edges.add(merged_edge)
            
            sub2global_edge_id[(suffix, sub_edge_id)] = edge_id_counter
            edge_id_counter += 1

        # 拓扑处理
        merged_topos = set()
        num_global_edges = len(merged_edges)
        num_global_nodes = len(merged_nodes)

        # 确定目标设备
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 合并role_matrix和sort_matrix
        global_role_matrix = torch.zeros((num_global_edges, num_global_nodes), dtype=torch.int64, device=target_device)
        global_sort_matrix = torch.zeros((num_global_edges, num_global_nodes), dtype=torch.int64, device=target_device)
        
        for global_edge_id in range(num_global_edges):
            sub_info = next(((s, se) for (s, se), ge in sub2global_edge_id.items() if ge == global_edge_id), None)
            if not sub_info:
                continue
                
            suffix, sub_edge_id = sub_info
            graph = next(h for s, h in graph_list if s == suffix)
            
            if graph.topo is None:
                continue
            
            if sub_edge_id < graph.topo.role_matrix.shape[0]:
                sub_role_row = graph.topo.role_matrix[sub_edge_id].to(target_device)
                sub_sort_row = graph.topo.sort_matrix[sub_edge_id].to(target_device)
                
                for sub_node_id in range(sub_role_row.shape[0]):
                    role_value = sub_role_row[sub_node_id].item()
                    sort_value = sub_sort_row[sub_node_id].item()
                    
                    if role_value == 0 and sort_value == 0:
                        continue
                        
                    if (suffix, sub_node_id) in sub2global_node_id:
                        global_node_id = sub2global_node_id[(suffix, sub_node_id)]
                        if 0 <= global_node_id < num_global_nodes:
                            global_role_matrix[global_edge_id, global_node_id] = role_value
                            global_sort_matrix[global_edge_id, global_node_id] = sort_value

        merged_topo = MHD_Topo(role_matrix=global_role_matrix, sort_matrix=global_sort_matrix)
        merged_topos.add(merged_topo)

        # 最终实例化
        merged_graph = cls(
            nodes=merged_nodes,
            edges=merged_edges,
            topos=merged_topos,
            device=target_device
        )
        
        merged_graph._register_all_params()
        merged_graph.compact_topological_sort()
        
        print(f"✅ 图合并完成 | 目标设备: {target_device} | 总节点: {len(merged_nodes)} | 总边: {len(merged_edges)}")
        return merged_graph
