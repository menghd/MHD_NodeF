# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Utils (MHD-Utils) - Version 2.0
Author: Souray Meng (孟号丁)
Utility Tools: Dataset, Augmentation, Training, Monitoring for MHD Framework
License: MIT
"""

import torch
import random
import numpy as np
import nibabel as nib
import torch.nn as nn
import cv2
import sys
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Callable, Any, Union, Iterable, Optional, Sequence, Set
from pathlib import Path
from abc import ABC, abstractmethod

# 添加当前目录到Python路径，确保能导入MHD核心框架
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入MHD核心框架
from MHD_Framework_V2 import (
    MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph,
)

# ===================== 状态保存/加载工具函数 =====================

def updown_node_value(nodes: Set[MHD_Node], path: str, mode: str, target_device: torch.device = None) -> None:
    """
    节点特征图保存/加载函数
    
    特性：
    1. 支持节点初始状态和当前状态的保存/加载
    2. 自动处理设备迁移
    3. 保存元信息用于验证
    
    Args:
        nodes: 节点集合
        path: 文件路径
        mode: 模式，'up'为加载，'down'为保存
        target_device: 目标设备（仅加载时有效）
    
    Raises:
        ValueError: 当模式参数错误时
    """
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        # 保存模式
        save_dict = {
            "node_initial_states": {n.name: n.initial_state for n in sorted(nodes, key=lambda x: x.id)},
            "node_current_states": {n.name: n.current_state for n in sorted(nodes, key=lambda x: x.id)},
            "node_info": {
                n.name: {
                    "id": n.id, 
                    "shape": n.current_state.shape, 
                    "dtype": str(n.current_state.dtype), 
                    "device": str(n.current_state.device)
                }
                for n in sorted(nodes, key=lambda x: x.id)
            }
        }
        torch.save(save_dict, path)
        
    else:
        # 加载模式
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for node in sorted(nodes, key=lambda x: x.id):
            if node.name in load_dict["node_initial_states"]:
                loaded_tensor = load_dict["node_initial_states"][node.name]
                # 自动迁移到目标设备
                if target_device is not None and loaded_tensor.device != target_device:
                    loaded_tensor = loaded_tensor.to(target_device, non_blocking=True)
                node.initial_state = loaded_tensor.to(
                    dtype=node.current_state.dtype
                )
            
            if node.name in load_dict["node_current_states"]:
                loaded_tensor = load_dict["node_current_states"][node.name]
                # 自动迁移到目标设备
                if target_device is not None and loaded_tensor.device != target_device:
                    loaded_tensor = loaded_tensor.to(target_device, non_blocking=True)
                node.current_state = loaded_tensor.to(
                    dtype=node.current_state.dtype
                )

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for n in nodes if n.name in (load_dict["node_initial_states"] if mode == 'up' else [n.name for n in nodes]))
    
    print(f"📊 节点特征图{mode_cn}完成 (Node value {mode_en} completed)")
    print(f" ├─ 总节点数: {len(nodes)}")
    print(f" ├─ 处理节点数: {processed}")
    print(f" ├─ 目标设备: {target_device}" if target_device and mode == 'up' else "")
    print(f" 📁 路径: {path}")


def updown_edge_value(edges: Set[MHD_Edge], path: str, mode: str, target_device: torch.device = None) -> None:
    """
    超边可学习模块保存/加载函数
    
    特性：
    1. 支持边中可学习参数的保存/加载
    2. 自动处理nn.Module状态字典
    3. 保存操作序列信息
    
    Args:
        edges: 边集合
        path: 文件路径
        mode: 模式，'up'为加载，'down'为保存
        target_device: 目标设备（仅加载时有效）
    
    Raises:
        ValueError: 当模式参数错误时
    """
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        # 保存模式
        save_dict = {
            "edge_params": {},
            "edge_info": {}
        }
        for edge in sorted(edges, key=lambda x: x.id):
            save_dict["edge_params"][edge.name] = [
                elem.state_dict() if isinstance(elem, nn.Module) else None
                for elem in edge.sequential_operation
            ]
            save_dict["edge_info"][edge.name] = {
                "id": edge.id,
                "operations": [str(type(op)) for op in edge.sequential_operation]
            }
        torch.save(save_dict, path)
        
    else:
        # 加载模式
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for edge in sorted(edges, key=lambda x: x.id):
            if edge.name in load_dict["edge_params"]:
                saved_params = load_dict["edge_params"][edge.name]
                for idx, elem in enumerate(edge.sequential_operation):
                    if idx < len(saved_params) and isinstance(elem, nn.Module) and saved_params[idx] is not None:
                        elem.load_state_dict(saved_params[idx])
                        # 迁移到目标设备
                        if target_device is not None:
                            edge.sequential_operation[idx] = elem.to(target_device, non_blocking=True)

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for e in edges if e.name in (load_dict["edge_params"] if mode == 'up' else [e.name for e in edges]))
    
    print(f"📊 超边可学习参数{mode_cn}完成 (Edge value {mode_en} completed)")
    print(f" ├─ 总边数: {len(edges)}")
    print(f" ├─ 处理边数: {processed}")
    print(f" ├─ 目标设备: {target_device}" if target_device and mode == 'up' else "")
    print(f" 📁 路径: {path}")


# ===================== 数据增强器系统 =====================

class MHD_Augmentor(ABC):
    """
    增强器基类
    
    特性：
    1. 基于种子保证同一样本的增强一致性
    2. 线程安全的随机状态管理
    3. 支持随机状态保存和恢复
    
    Author: Souray Meng (孟号丁)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化增强器
        
        Args:
            seed: 随机种子，如果为None则自动生成
        """
        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self._random_state = None
        self._np_random_state = None
        
    def set_seed(self, seed: int):
        """
        设置增强器种子（线程安全）
        
        Args:
            seed: 新的随机种子
        """
        self.seed = seed
        
        # 保存当前随机状态
        self._random_state = random.getstate()
        self._np_random_state = np.random.get_state()
        
        # 设置新种子
        random.seed(seed)
        np.random.seed(seed)
        
    def restore_random_state(self):
        """恢复随机状态（避免影响其他增强器）"""
        if self._random_state is not None:
            random.setstate(self._random_state)
        if self._np_random_state is not None:
            np.random.set_state(self._np_random_state)
    
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        执行增强操作
        
        Args:
            tensor: 输入张量
        
        Returns:
            增强后的张量
        """
        pass


class RandomRotate(MHD_Augmentor):
    """
    随机旋转增强器
    
    特性：
    1. 支持2D/3D张量旋转
    2. 可配置最大旋转角度
    3. 保持张量形状不变
    """
    
    def __init__(self, max_angle: float = 10.0, seed: Optional[int] = None):
        """
        初始化旋转增强器
        
        Args:
            max_angle: 最大旋转角度（度）
            seed: 随机种子
        """
        super().__init__(seed)
        self.max_angle = max_angle
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        执行随机旋转
        
        Args:
            tensor: 输入张量，形状为(C, H, W)或类似
        
        Returns:
            旋转后的张量
        """
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
            return torch.tensor(np.stack(rotated), dtype=tensor.dtype, device=tensor.device)
        return tensor


class RandomFlip(MHD_Augmentor):
    """
    随机翻转增强器
    
    特性：
    1. 支持水平和垂直翻转
    2. 可配置翻转轴
    3. 概率性翻转
    """
    
    def __init__(self, axis: int = 1, seed: Optional[int] = None):
        """
        初始化翻转增强器
        
        Args:
            axis: 翻转轴，0为垂直，1为水平
            seed: 随机种子
        """
        super().__init__(seed)
        self.axis = axis
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        执行随机翻转
        
        Args:
            tensor: 输入张量
        
        Returns:
            翻转后的张量（或原始张量）
        """
        random.seed(self.seed)
        flip = random.choice([True, False])
        if flip:
            return torch.flip(tensor, dims=[self.axis])
        return tensor


class Normalize(MHD_Augmentor):
    """
    归一化增强器
    
    特性：
    1. 无随机性的确定性归一化
    2. 可配置均值和标准差
    3. 支持批量处理
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None):
        """
        初始化归一化增强器
        
        Args:
            mean: 目标均值
            std: 目标标准差
            seed: 随机种子（兼容性参数）
        """
        super().__init__(seed)
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        执行归一化
        
        Args:
            tensor: 输入张量
        
        Returns:
            归一化后的张量
        """
        return (tensor - self.mean) / self.std


class MHD_AugmentComposer:
    """
    增强组合器
    
    特性：
    1. 顺序执行多个增强器
    2. 共享同一种子保证一致性
    3. 自动管理随机状态
    
    Author: Souray Meng (孟号丁)
    """
    
    def __init__(self, augmentors: Sequence[MHD_Augmentor]):
        """
        初始化增强组合器
        
        Args:
            augmentors: 增强器序列
        """
        self.augmentors = augmentors
    
    def set_seed(self, seed: int):
        """
        为所有增强器设置同一种子
        
        Args:
            seed: 随机种子
        """
        for aug in self.augmentors:
            aug.set_seed(seed)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        按顺序执行所有增强
        
        Args:
            tensor: 输入张量
        
        Returns:
            所有增强后的张量
        """
        for aug in self.augmentors:
            tensor = aug(tensor)
            aug.restore_random_state()  # 恢复随机状态
        return tensor


# ===================== 统一数据集类 =====================

class MHD_Dataset(Dataset):
    """
    高扩展MHD数据集（修复版）
    
    特性：
    1. 修复多进程数据加载时的随机种子问题
    2. 支持节点级别的数据加载和增强
    3. 统一设备管理
    4. 支持确定性增强
    
    Author: Souray Meng (孟号丁)
    """
    
    def __init__(
        self,
        sample_info_list: List[Any],
        node_configs: Dict[str, Dict],
        base_seed: int = 42,
        target_device: torch.device = None
    ):
        """
        初始化MHD数据集
        
        Args:
            sample_info_list: 样本信息列表
            node_configs: 节点配置字典，格式为{节点名: {loader: 加载函数, augmentor: 增强器}}
            base_seed: 基础随机种子
            target_device: 目标计算设备
        """
        self.sample_info_list = sample_info_list
        self.node_configs = node_configs
        self.base_seed = base_seed
        self.target_device = target_device or torch.device("cpu")
        self.node_names = list(node_configs.keys())

        # 预处理增强器
        for node_name, config in self.node_configs.items():
            aug = config.get("augmentor")
            if aug is not None:
                if isinstance(aug, Sequence) and not isinstance(aug, MHD_AugmentComposer):
                    config["augmentor"] = MHD_AugmentComposer(aug)
                elif isinstance(aug, MHD_Augmentor) and not isinstance(aug, MHD_AugmentComposer):
                    config["augmentor"] = MHD_AugmentComposer([aug])

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_info_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        单样本加载流程（修复版）
        
        修复逻辑：不再在__getitem__中设置全局随机种子
        增强器使用确定的随机种子
        
        Args:
            idx: 样本索引
        
        Returns:
            节点名到张量的映射字典
        
        Raises:
            RuntimeError: 当数据加载失败时
        """
        # 1. 获取当前样本信息
        sample_info = self.sample_info_list[idx]
        
        # 2. 获取worker信息（用于多进程随机种子）
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 多进程：使用worker_id计算样本特定种子
            worker_id = worker_info.id
            sample_seed = self.base_seed + worker_id + idx
        else:
            # 单进程
            sample_seed = self.base_seed + idx

        # 3. 加载+增强每个节点的数据
        sample_data = {}
        for node_name, config in self.node_configs.items():
            # 3.1 调用节点专属加载函数
            load_fn = config["loader"]
            try:
                tensor = load_fn(sample_info)
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"节点 {node_name} 加载失败（idx={idx}）: {str(e)}")

            # 3.2 执行增强（使用确定的种子）
            augmentor = config.get("augmentor")
            if augmentor is not None:
                # 为增强器设置种子
                augmentor.set_seed(sample_seed)
                tensor = augmentor(tensor)

            # 3.3 搬运到目标设备
            if tensor.device != self.target_device:
                tensor = tensor.to(self.target_device, non_blocking=True)
            sample_data[node_name] = tensor

        return sample_data

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批次拼接函数
        
        保证所有节点数据的批次维度一致
        
        Args:
            batch: 批次数据列表
        
        Returns:
            拼接后的批次数据
        
        Raises:
            ValueError: 当批次为空时
        """
        if not batch:
            raise ValueError("空批次数据")

        ref_node_names = batch[0].keys()
        batch_data = {}

        for node_name in ref_node_names:
            node_samples = [sample[node_name] for sample in batch]
            batch_data[node_name] = torch.stack(node_samples, dim=0)

        return batch_data


# ===================== 数据加载器工具函数 =====================

def create_mhd_extend_dataloader(
    dataset: MHD_Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = False
) -> DataLoader:
    """
    创建MHD数据加载器（修复版）
    
    特性：
    1. 支持可复现的多进程数据加载
    2. 为每个工作进程设置独立的随机种子
    3. 统一设备管理
    
    Args:
        dataset: MHD数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃不完整的批次
        persistent_workers: 是否保持工作进程
    
    Returns:
        配置好的DataLoader实例
    """
    def worker_init_fn(worker_id: int) -> None:
        """
        工作进程初始化函数（闭包，可访问dataset）
        
        为每个工作进程设置不同的随机种子，保证可复现性
        
        Args:
            worker_id: 工作进程ID
        """
        # 使用闭包捕获的dataset.base_seed
        process_seed = dataset.base_seed + worker_id
        
        # 设置Python随机种子
        random.seed(process_seed)
        
        # 设置NumPy随机种子
        np.random.seed(process_seed)
        
        # 设置PyTorch随机种子
        torch.manual_seed(process_seed)
        
        # 设置CUDA随机种子（如果可用）
        if torch.cuda.is_available():
            torch.cuda.manual_seed(process_seed)
            torch.cuda.manual_seed_all(process_seed)
            
        # 确保可复现性设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 记录日志
        if worker_id == 0:  # 只在第一个worker记录
            import logging
            logger = logging.getLogger("mhd_data")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            logger.info(f"工作进程 {worker_id} 随机种子已初始化: {process_seed}")

    # 设置工作进程初始化函数
    worker_init_fn_to_use = None
    if num_workers > 0:
        worker_init_fn_to_use = worker_init_fn
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MHD_Dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn_to_use,
        persistent_workers=persistent_workers
    )


# ===================== 监控器类 =====================

import warnings
from collections import defaultdict
import logging
from datetime import datetime
from tqdm import tqdm
import json


class MHD_Monitor:
    """
    MHD训练监控器
    
    特性：
    1. 监控节点和边的统计信息
    2. 支持训练和验证模式的梯度监控
    3. 安全的张量统计计算
    4. 格式化输出
    
    Author: Souray Meng (孟号丁)
    """
    
    def __init__(self, monitor_nodes: list, monitor_edges: list = None):
        """
        初始化监控器
        
        Args:
            monitor_nodes: 要监控的节点名称列表
            monitor_edges: 要监控的边名称列表，如果为None则监控所有边
        """
        self.monitor_nodes = monitor_nodes
        self.monitor_edges = monitor_edges
        self.records = defaultdict(list)
        self.step_counter = 0
    
    def reset(self):
        """重置监控记录"""
        self.records = defaultdict(list)
        self.step_counter = 0
    
    def _safe_tensor_stats(self, tensor: torch.Tensor) -> dict:
        """
        安全计算张量统计信息
        
        处理NaN和Inf值，返回基本统计量
        
        Args:
            tensor: 输入张量
        
        Returns:
            包含均值、和、最大值、最小值的字典
        """
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        return {
            "mean": float(tensor.mean().item()),
            "sum": float(tensor.sum().item()),
            "max": float(tensor.max().item()),
            "min": float(tensor.min().item())
        }
    
    def monitor_node(self, mhd_graph, prefix: str = "") -> dict:
        """
        监控节点状态
        
        Args:
            mhd_graph: MHD图实例
            prefix: 指标前缀
        
        Returns:
            节点指标字典
        """
        node_metrics = {}
        for node_name in self.monitor_nodes:
            node = mhd_graph.get_node_by_name(node_name)
            if node is None:
                warnings.warn(f"监控节点 {node_name} 不存在，跳过")
                continue
            
            value = node.current_state.detach()  # 改为监控current_state
            stats = self._safe_tensor_stats(value)
            
            for stat_name, stat_value in stats.items():
                node_metrics[f"{prefix}node_{node_name}_{stat_name}"] = stat_value
        
        for k, v in node_metrics.items():
            self.records[k].append(v)
        self.step_counter += 1
        
        return node_metrics
    
    def monitor_edge(self, mhd_graph, prefix: str = "", train_mode: bool = True) -> dict:
        """
        监控边参数和梯度
        
        Args:
            mhd_graph: MHD图实例
            prefix: 指标前缀
            train_mode: 是否为训练模式（影响梯度监控）
        
        Returns:
            边指标字典
        """
        edge_metrics = {}
        target_edges = self.monitor_edges or [e.name for e in mhd_graph.edges]
        
        for edge_name in target_edges:
            edge = mhd_graph.get_edge_by_name(edge_name)
            if edge is None:
                warnings.warn(f"监控边 {edge_name} 不存在，跳过")
                continue
            
            for idx, op in enumerate(edge.sequential_operation):
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
        """
        计算平均指标
        
        Args:
            step_window: 时间窗口大小，如果为None则使用全部历史
        
        Returns:
            平均指标字典
        """
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
        """
        格式化指标输出
        
        Args:
            metrics: 指标字典
            decimal: 小数位数
        
        Returns:
            格式化后的字符串
        """
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


# ===================== 训练器类 =====================

class MHD_Trainer:
    """
    MHD训练器（最终修正版）
    
    特性：
    1. 修复计算图重复使用问题
    2. 统一的训练/验证流程
    3. 自动检查点管理
    4. 详细的训练历史记录
    5. 统一设备管理
    
    Author: Souray Meng (孟号丁)
    """
    
    def __init__(
        self,
        mhd_graph,
        optimizer: torch.optim.Optimizer,
        monitor: MHD_Monitor,
        save_dir: str = "./mhd_ckpts",
        target_loss_node: str = "final_fl",
        target_metric_node: str = None,
        grad_clip_norm: float = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ):
        """
        初始化MHD训练器
        
        Args:
            mhd_graph: MHD图实例
            optimizer: 优化器
            monitor: 监控器
            save_dir: 检查点保存目录
            target_loss_node: 目标损失节点名称
            target_metric_node: 目标指标节点名称
            grad_clip_norm: 梯度裁剪范数阈值
            lr_scheduler: 学习率调度器
        """
        self.mhd_graph = mhd_graph
        self.optimizer = optimizer
        self.monitor = monitor
        self.save_dir = save_dir
        self.target_loss_node = target_loss_node
        self.target_metric_node = target_metric_node
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler = lr_scheduler
        
        # 设备统一：使用图的设备
        self.device = mhd_graph.device
        
        os.makedirs(save_dir, exist_ok=True)
        self.logger = self._setup_logger(save_dir)
        
        # 训练历史记录
        self.history = {
            "train": {"loss": [], "metrics": []},
            "eval": {"loss": [], "metrics": []},
            "best_eval_loss": float("inf"),
            "best_eval_metric": -float("inf"),
            "best_epoch_loss": -1,
            "best_epoch_metric": -1
        }
        
        # 验证损失节点
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
        """
        设置训练日志记录器
        
        Args:
            save_dir: 日志保存目录
        
        Returns:
            配置好的日志记录器
        """
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
        """验证损失节点和指标节点是否存在"""
        loss_node = self.mhd_graph.get_node_by_name(self.target_loss_node)
        if loss_node is None:
            raise ValueError(f"损失节点 '{self.target_loss_node}' 不存在！")
        self.logger.info(f"✅ 损失节点验证通过: {self.target_loss_node}")
        
        if self.target_metric_node:
            metric_node = self.mhd_graph.get_node_by_name(self.target_metric_node)
            if metric_node is None:
                raise ValueError(f"指标节点 '{self.target_metric_node}' 不存在！")
            self.logger.info(f"✅ 指标节点验证通过: {self.target_metric_node}")
    
    def train_step(self, input_dict: dict) -> tuple[float, float]:
        """
        单步训练（修复计算图重复使用问题）
        
        Args:
            input_dict: 输入数据字典，格式为{节点名: 张量}
        
        Returns:
            (损失值, 指标值)元组
        
        Raises:
            RuntimeError: 当损失张量无梯度时
        """
        self.mhd_graph.train()
        
        # 清理梯度
        self.optimizer.zero_grad(set_to_none=True)
        
        # 批次一致性校验
        try:
            batch_size = self._check_batch_consistency(input_dict)
        except ValueError as e:
            self.logger.error(f"批次校验失败: {str(e)}")
            raise
        
        # 第一步：重置所有节点，current回到initial状态
        for node in self.mhd_graph.nodes:
            node.reset()  # 调用reset()将current_state重置为initial_state的clone
        
        # 第二步：读取数据给current（注入新数据）
        for node_name, tensor in input_dict.items():
            node = self.mhd_graph.get_node_by_name(node_name)
            if node is None:
                warnings.warn(f"输入数据中的节点 {node_name} 不存在于图中，跳过")
                continue
            
            # 创建新的可分离张量，重新建立计算图
            new_tensor = tensor.to(self.device, non_blocking=True).detach().requires_grad_(True)
            
            # 只更新current_state，不更新initial_state
            node.current_state = new_tensor
            # 注意：不更新node.initial_state，保持initial为初始状态
        
        # 第三步：前后向传播更新current
        self.mhd_graph.forward()
        
        # 获取损失
        loss_node = self.mhd_graph.get_node_by_name(self.target_loss_node)
        if loss_node is None:
            raise ValueError(f"损失节点 {self.target_loss_node} 不存在")
        
        loss_tensor = loss_node.current_state.mean()
        
        # 检查梯度可用性
        if not loss_tensor.requires_grad:
            self.logger.error(f"损失张量无梯度！请检查前向计算逻辑")
            raise RuntimeError("损失张量无梯度")
        
        # 反向传播
        loss_tensor.backward(retain_graph=False)
        
        # 梯度裁剪
        if self.grad_clip_norm and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.mhd_graph.parameters(), self.grad_clip_norm)
        
        # 优化器更新
        self.optimizer.step()
        
        # 获取损失值和指标值
        loss_value = float(loss_tensor.detach().item())
        loss_value = 0.0 if np.isnan(loss_value) or np.isinf(loss_value) else loss_value
        
        metric_value = 0.0
        if self.target_metric_node:
            metric_node = self.mhd_graph.get_node_by_name(self.target_metric_node)
            if metric_node:
                metric_value = float(metric_node.current_state.detach().mean().item())
                metric_value = 0.0 if np.isnan(metric_value) or np.isinf(metric_value) else metric_value
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss_value, metric_value

    @torch.no_grad()  # 关键：eval阶段禁用所有梯度计算
    def eval_step(self, input_dict: dict) -> tuple[float, float]:
        """
        单步验证（完全无计算图）
        
        Args:
            input_dict: 输入数据字典，格式为{节点名: 张量}
        
        Returns:
            (损失值, 指标值)元组
        """
        self.mhd_graph.eval()
        
        # 批次一致性校验
        try:
            batch_size = self._check_batch_consistency(input_dict)
        except ValueError as e:
            self.logger.error(f"批次校验失败: {str(e)}")
            raise
        
        # 第一步：重置所有节点，current回到initial状态
        for node in self.mhd_graph.nodes:
            node.reset()  # 调用reset()将current_state重置为initial_state的clone
        
        # 第二步：读取数据给current（eval不需要梯度）
        for node_name, tensor in input_dict.items():
            node = self.mhd_graph.get_node_by_name(node_name)
            if node is None:
                warnings.warn(f"输入数据中的节点 {node_name} 不存在于图中，跳过")
                continue
            
            # eval阶段完全detach，不建立计算图
            detached_tensor = tensor.to(self.device, non_blocking=True).detach()
            node.current_state = detached_tensor
        
        # 第三步：前向传播（eval只有前向，无反向）
        self.mhd_graph.forward()
        
        # 获取损失
        loss_node = self.mhd_graph.get_node_by_name(self.target_loss_node)
        if loss_node is None:
            raise ValueError(f"损失节点 {self.target_loss_node} 不存在")
        
        loss_tensor = loss_node.current_state.mean()
        loss_tensor = torch.nan_to_num(loss_tensor, nan=0.0, posinf=1e3, neginf=-1e3)
        loss_value = float(loss_tensor.item())
        loss_value = 0.0 if np.isnan(loss_value) or np.isinf(loss_value) else loss_value
        
        # 获取指标
        metric_value = 0.0
        if self.target_metric_node:
            metric_node = self.mhd_graph.get_node_by_name(self.target_metric_node)
            if metric_node:
                metric_value = float(metric_node.current_state.mean().item())
                metric_value = 0.0 if np.isnan(metric_value) or np.isinf(metric_value) else metric_value
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss_value, metric_value
    
    def _check_batch_consistency(self, input_dict: dict) -> int:
        """
        校验所有节点数据的批次大小一致性
        
        Args:
            input_dict: 输入数据字典
        
        Returns:
            批次大小
        
        Raises:
            ValueError: 当批次大小不一致或数据维度不合法时
        """
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
        """
        训练一个epoch（修复版）
        
        Args:
            train_data: 训练数据列表
            epoch: 当前轮次
        """
        self.monitor.reset()
        total_loss = 0.0
        total_metric = 0.0
        pbar = tqdm(train_data, desc=f"Train Epoch {epoch+1}", leave=False)
        
        for step, input_dict in enumerate(pbar):
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 显式调用垃圾回收
            import gc
            gc.collect()
            
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
        """
        验证一个epoch（保持与train一致的处理）
        
        Args:
            eval_data: 验证数据列表
            epoch: 当前轮次
        """
        self.monitor.reset()
        total_loss = 0.0
        total_metric = 0.0
        pbar = tqdm(eval_data, desc=f"Eval Epoch {epoch+1}", leave=False)
        
        for step, input_dict in enumerate(pbar):
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 显式调用垃圾回收
            import gc
            gc.collect()
            
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
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            is_best_loss: 是否为最佳损失模型
            is_best_metric: 是否为最佳指标模型
        """
        try:
            node_path = os.path.join(self.save_dir, f"node_epoch_{epoch}.pth")
            edge_path = os.path.join(self.save_dir, f"edge_epoch_{epoch}.pth")
            state_path = os.path.join(self.save_dir, f"train_state_epoch_{epoch}.pth")
            
            updown_node_value(self.mhd_graph.nodes, node_path, mode="down", target_device=self.device)
            updown_edge_value(self.mhd_graph.edges, edge_path, mode="down", target_device=self.device)
            
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
        """
        加载检查点
        
        Args:
            load_best_loss: 是否加载最佳损失模型
            load_best_metric: 是否加载最佳指标模型
            epoch: 指定轮次加载
        
        Raises:
            FileNotFoundError: 当权重文件不存在时
        """
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
            
            updown_node_value(self.mhd_graph.nodes, node_path, mode="up", target_device=self.device)
            updown_edge_value(self.mhd_graph.edges, edge_path, mode="up", target_device=self.device)
            
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
        """保存训练历史到JSON文件"""
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
        """
        完整训练流程
        
        Args:
            train_data: 训练数据列表
            eval_data: 验证数据列表
            epochs: 训练轮次
        
        Raises:
            Exception: 训练过程中发生错误时
        """
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


# ===================== 优化器创建函数 =====================

def create_optimizer(
    mhd_graph,
    edge_optim_config: dict = None,
    default_optimizer_type: str = "adam",
    default_lr: float = 0.001,
    default_weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建MHD图优化器
    
    特性：
    1. 支持边级别的优化器配置
    2. 自动收集图参数
    3. 灵活的优化器配置
    
    Args:
        mhd_graph: MHD图实例
        edge_optim_config: 边优化配置字典
        default_optimizer_type: 默认优化器类型
        default_lr: 默认学习率
        default_weight_decay: 默认权重衰减
        **kwargs: 优化器额外参数
    
    Returns:
        配置好的优化器实例
    
    Raises:
        ValueError: 当优化器类型不支持时
    """
    edge_optim_config = edge_optim_config or {}
    param_groups = []
    
    # 为指定边创建自定义参数组
    for edge_name, config in edge_optim_config.items():
        edge = mhd_graph.get_edge_by_name(edge_name)
        if edge is None:
            warnings.warn(f"边 {edge_name} 不存在，跳过自定义配置")
            continue
        
        edge_params = []
        for op in edge.sequential_operation:
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
    
    optimizer = optimizer_map[default_optimizer_type.lower()](param_groups, **kwargs)
    
    logger = logging.getLogger("mhd_train")
    logger.info(f"✅ 优化器创建完成: {default_optimizer_type.upper()}")
    logger.info(f"📋 参数组配置:")
    for i, group in enumerate(param_groups):
        logger.info(f"  组{i+1}: {group.get('name', 'unknown')} - lr={group['lr']}, weight_decay={group['weight_decay']}")
    
    return optimizer
