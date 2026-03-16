# -*- coding: utf-8 -*-
"""
MHD_Utils 使用示例 - 适配最新framework+utils版本（MHD_Graph版）
核心适配：
1. 全量替换MHD_Net为MHD_Graph
2. 适配新的node_dict管理逻辑
3. 节点值改为nn.Parameter保证可训练+梯度追踪
4. 适配eval_step返回格式（loss, metric_value）
5. 修复合并网络的节点/边名称匹配
6. 补充参数注册和梯度流修复逻辑
7. 数据生成改为动态序列化方式（适配统一数据集接口）
8. 修复pin_memory错误：数据先在CPU处理，最后再移到GPU
9. 优化数据生成：不再全随机，生成有业务意义的分类数据
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from typing import Set, Dict, Any, List, Callable

# 导入MHD核心框架 - 核心修改：替换MHD_Net为MHD_Graph
from MHD_Framework import (
    MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph, 
    get_obj_by_name, updown_node_value, updown_edge_value,
    MHD_Trainer, MHD_Monitor, create_optimizer,
    MHD_ExtendDataset, create_mhd_extend_dataloader,
    MHD_Augmentor, MHD_AugmentComposer,
    RandomRotate, RandomFlip, Normalize
)


# ===================== Focal Loss图定义 =====================
def example1_focal_loss_graph(alpha: float = 0.25, gamma: float = 2.0, batch_size: int = 8, num_classes: int = 10):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    EPS = 1e-4
    
    B, C = batch_size, num_classes
    alpha_val = alpha
    gamma_val = gamma
    
    torch.manual_seed(42)
    
    # 定义节点（改为nn.Parameter保证可训练）
    nodes: Set[MHD_Node] = set()
    
    node_logits = MHD_Node(
        id=0,
        name="logits_pred",
        value=nn.Parameter(torch.zeros(B, C, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_onehot_gt = MHD_Node(
        id=1,
        name="onehot_gt",
        value=nn.Parameter(torch.zeros(B, C, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_p = MHD_Node(
        id=2,
        name="p",
        value=nn.Parameter(torch.zeros(B, C, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_p_t_before = MHD_Node(
        id=3,
        name="p_t_before",
        value=nn.Parameter(torch.ones(B, C, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "mul"}
    )
    node_alpha_t = MHD_Node(
        id=4,
        name="alpha_t",
        value=nn.Parameter(torch.zeros(B, 1, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_p_t = MHD_Node(
        id=5,
        name="p_t",
        value=nn.Parameter(torch.zeros(B, 1, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_ce_loss = MHD_Node(
        id=6,
        name="ce_loss",
        value=nn.Parameter(torch.zeros(B, 1, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_focal_factor = MHD_Node(
        id=7,
        name="focal_factor",
        value=nn.Parameter(torch.zeros(B, 1, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    node_focal_loss = MHD_Node(
        id=8,
        name="focal_loss",
        value=nn.Parameter(torch.ones(B, 1, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "mul"}
    )
    node_final_fl = MHD_Node(
        id=9,
        name="final_fl",
        value=nn.Parameter(torch.zeros(1, 1, device=DEVICE, dtype=DTYPE), requires_grad=True),
        func={"head": "share", "tail": "sum"}
    )
    
    nodes.update([
        node_logits, node_onehot_gt, node_p, node_p_t_before,
        node_alpha_t, node_p_t, node_ce_loss, node_focal_factor,
        node_focal_loss, node_final_fl
    ])
    
    # 定义边
    edges: Set[MHD_Edge] = set()
    edge_logits_to_p = MHD_Edge(
        id=0,
        name="edge_logits_to_p",
        value=["softmax(dim=1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_p_to_ptb = MHD_Edge(
        id=1,
        name="edge_p_to_ptb",
        value=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_onehot_to_ptb = MHD_Edge(
        id=2,
        name="edge_onehot_to_ptb",
        value=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_onehot_to_alphat = MHD_Edge(
        id=3,
        name="edge_onehot_to_alphat",
        value=[f"mul({alpha_val})", "sum(dim=1,keepdim=True)"],
        func={"in": "concat", "out": "split"}
    )
    edge_ptb_to_pt = MHD_Edge(
        id=4,
        name="edge_ptb_to_pt",
        value=["sum(dim=1,keepdim=True)"],
        func={"in": "concat", "out": "split"}
    )
    edge_pt_to_ce = MHD_Edge(
        id=5,
        name="edge_pt_to_ce",
        value=[f"add({EPS})", "log()", "mul(-1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_pt_to_ff = MHD_Edge(
        id=6,
        name="edge_pt_to_ff",
        value=["mul(-1)", "add(1)", f"pow({gamma_val})"],
        func={"in": "concat", "out": "split"}
    )
    edge_alphat_to_fl = MHD_Edge(
        id=7,
        name="edge_alphat_to_fl",
        value=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_ff_to_fl = MHD_Edge(
        id=8,
        name="edge_ff_to_fl",
        value=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_ce_to_fl = MHD_Edge(
        id=9,
        name="edge_ce_to_fl",
        value=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_fl_to_final = MHD_Edge(
        id=10,
        name="edge_fl_to_final",
        value=["sum(dim=0,keepdim=True)"],
        func={"in": "concat", "out": "split"}
    )
    
    edges.update([
        edge_logits_to_p, edge_p_to_ptb, edge_onehot_to_ptb,
        edge_onehot_to_alphat, edge_ptb_to_pt, edge_pt_to_ce,
        edge_pt_to_ff, edge_alphat_to_fl, edge_ff_to_fl,
        edge_ce_to_fl, edge_fl_to_final
    ])
    
    # 拓扑矩阵
    role_topo_value = torch.tensor([
        [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 1]
    ], dtype=torch.int64, device=DEVICE)
    
    sort_topo_value = torch.tensor([
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ], dtype=torch.int64, device=DEVICE)
    
    topos = {
        MHD_Topo(type="role", value=role_topo_value),
        MHD_Topo(type="sort", value=sort_topo_value)
    }
    
    fl_graph = MHD_Graph(  # 核心修改：改为MHD_Graph
        nodes=nodes,
        edges=edges,
        topos=topos,
        device=DEVICE
    )
    
    # 注册所有参数（保证梯度追踪）
    fl_graph._register_all_params()
    
    return fl_graph

# ===================== 1. 构建标准MHD图组件 =====================
def build_main_graph(batch_size: int = 9, num_classes: int = 10, device: torch.device = None) -> MHD_Graph:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # 1. 创建节点集合（改为nn.Parameter保证可训练）
    nodes_graph: Set[MHD_Node] = {
        MHD_Node(
            id=0, name="input", 
            value=nn.Parameter(torch.zeros(batch_size, 512, device=device, dtype=dtype), requires_grad=True),
            func={"head": "share", "tail": "sum"},
        ),
        MHD_Node(
            id=1, name="hidden",
            value=nn.Parameter(torch.zeros(batch_size, 256, device=device, dtype=dtype), requires_grad=True),
            func={"head": "share", "tail": "sum"},
        ),
        MHD_Node(
            id=2, name="logits_pred",
            value=nn.Parameter(torch.zeros(batch_size, num_classes, device=device, dtype=dtype), requires_grad=True),
            func={"head": "share", "tail": "sum"},
        )
    }
    
    # 2. 创建边集合
    edge1_value = [
        nn.Linear(512, 256, device=device, dtype=dtype),
        nn.BatchNorm1d(256, device=device),
        nn.ReLU(inplace=True)
    ]
    edge2_value = [
        nn.Linear(256, num_classes, device=device, dtype=dtype)
    ]
    
    edges_graph: Set[MHD_Edge] = {
        MHD_Edge(
            id=0, name="e_input_to_hidden",
            value=edge1_value,
            func={"in": "concat", "out": "split"},
        ),
        MHD_Edge(
            id=1, name="e_hidden_to_logits_pred",
            value=edge2_value,
            func={"in": "concat", "out": "split"},
        )
    }
    
    # 3. 创建拓扑集合
    topos_graph: Set[MHD_Topo] = {
        MHD_Topo(
            type="role",
            value=torch.tensor([
                [-1, 1, 0],
                [0, -1, 1]
            ], dtype=torch.int64, device=device)
        ),
        MHD_Topo(
            type="sort",
            value=torch.tensor([
                [1, 1, 0],
                [0, 1, 1]
            ], dtype=torch.int64, device=device)
        )
    }
    
    main_graph = MHD_Graph(  # 核心修改：改为MHD_Graph
        nodes=nodes_graph,
        edges=edges_graph,
        topos=topos_graph,
        device=device
    )
    
    # 注册参数
    main_graph._register_all_params()
    
    return main_graph

def build_metric_graph(batch_size: int = 9, device: torch.device = None) -> MHD_Graph:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # 1. 创建节点集合（改为nn.Parameter保证可训练）
    nodes_graph: Set[MHD_Node] = {
        MHD_Node(
            id=0, name="ce_loss",  # 关键：改为和FL图相同的节点名
            value=nn.Parameter(torch.zeros(batch_size, 1, device=device, dtype=dtype), requires_grad=True),
            func={"head": "share", "tail": "sum"},
        ),
        MHD_Node(
            id=1, name="node_metric_ce",
            value=nn.Parameter(torch.zeros(batch_size, 1, device=device, dtype=dtype), requires_grad=True),
            func={"head": "share", "tail": "sum"},
        )
    }
    
    # 2. 创建边集合
    edge_value = [
        'sum(dim=1, keepdim=True)',
        'mul(-1)'
    ]
    
    edges_graph: Set[MHD_Edge] = {
        MHD_Edge(
            id=0, name="e_metric_ce_calc",
            value=edge_value,
            func={"in": "concat", "out": "split"},
        )
    }
    
    # 3. 创建拓扑集合
    topos_graph: Set[MHD_Topo] = {
        MHD_Topo(
            type="role",
            value=torch.tensor([[-1, 1]], dtype=torch.int64, device=device)
        ),
        MHD_Topo(
            type="sort",
            value=torch.tensor([[1, 1]], dtype=torch.int64, device=device)
        )
    }
    
    metric_graph = MHD_Graph(  # 核心修改：改为MHD_Graph
        nodes=nodes_graph,
        edges=edges_graph,
        topos=topos_graph,
        device=device
    )
    
    # 注册参数
    metric_graph._register_all_params()
    
    return metric_graph

# ===================== 2. 动态数据生成（优化版：有业务意义的分类数据） =====================
class DynamicDataAugmentor(MHD_Augmentor):
    """自定义数据增强器：为输入特征添加噪声"""
    def __init__(self, noise_level: float = 0.1, seed: int = None):
        super().__init__(seed)
        self.noise_level = noise_level
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        random.seed(self.seed)
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

def generate_class_centers(num_classes: int = 10, feature_dim: int = 512, seed: int = 42) -> torch.Tensor:
    """
    生成有区分度的类别中心（非全随机）
    每个类别的中心在特征空间中相互分离，保证分类任务可学习
    """
    torch.manual_seed(seed)
    
    # 生成正交化的类别中心（保证类别间区分度）
    centers = torch.randn(num_classes, feature_dim)
    
    # 正交化处理：减少类别间重叠
    centers = torch.nn.functional.normalize(centers, p=2, dim=1)
    
    # 缩放：让类别中心间距更大
    centers = centers * 3.0
    
    return centers

def create_dynamic_dataset(batch_size: int = 9, num_batches: int = 100, 
                           num_classes: int = 10, target_device: torch.device = None,
                           is_train: bool = True) -> MHD_ExtendDataset:
    """
    创建动态生成的MHD数据集（适配统一数据集接口）
    核心优化：
    1. 生成有区分度的分类数据（非全随机）
    2. 数据先在CPU处理，最后再移到目标设备
    3. 标签与特征强关联，保证任务可学习
    """
    # 强制所有数据处理在CPU完成
    cpu_device = torch.device("cpu")
    target_device = target_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 样本信息列表：使用索引作为样本ID
    sample_info_list = list(range(num_batches * batch_size))
    
    # 2. 生成有区分度的类别中心（CPU上生成）
    feature_dim = 512
    class_centers = generate_class_centers(
        num_classes=num_classes,
        feature_dim=feature_dim,
        seed=42
    ).to(cpu_device)
    
    # 3. 定义节点加载函数（全部在CPU处理，有业务意义）
    def load_main_input(sample_id: int) -> torch.Tensor:
        """
        加载主图输入节点数据（CPU）
        生成逻辑：类别中心 + 小噪声，保证特征与标签强关联
        """
        # 按样本ID分配标签（固定映射，保证可复现）
        label = sample_id % num_classes
        
        # 基础特征：使用对应类别的中心
        feature = class_centers[label].clone()
        
        # 添加小噪声（模拟真实数据的变异）
        noise_level = 0.1 if is_train else 0.01
        noise = torch.randn_like(feature) * noise_level
        
        # 特征归一化
        feature = feature + noise
        feature = torch.nn.functional.normalize(feature, p=2, dim=0)
        
        return feature
    
    def load_fl_onehot_gt(sample_id: int) -> torch.Tensor:
        """
        加载FL图onehot标签数据（CPU）
        与输入特征严格对应
        """
        # 按样本ID分配标签（与输入特征完全一致）
        label = sample_id % num_classes
        
        # 生成one-hot编码（CPU）
        onehot = F.one_hot(torch.tensor(label, device=cpu_device), num_classes).float()
        
        return onehot
    
    # 4. 创建共享增强器（训练集增强，验证集不增强）
    if is_train:
        shared_aug = MHD_AugmentComposer([
            DynamicDataAugmentor(noise_level=0.05),  # 降低噪声水平，保证任务可学习
            Normalize(mean=0.0, std=3.0)  # 归一化到标准分布
        ])
    else:
        shared_aug = MHD_AugmentComposer([
            Normalize(mean=0.0, std=3.0)  # 仅归一化
        ])
    
    # 5. 节点配置
    node_configs = {
        "main_input": {
            "loader": load_main_input,
            "augmentor": shared_aug
        },
        "fl_onehot_gt": {
            "loader": load_fl_onehot_gt,
            "augmentor": None  # 标签不增强
        }
    }
    
    # 6. 创建数据集（CPU上处理，最后再移到目标设备）
    dataset = MHD_ExtendDataset(
        sample_info_list=sample_info_list,
        node_configs=node_configs,
        base_seed=42,
        device=target_device  # 最后搬运到目标设备
    )
    
    return dataset

def generate_training_data_dynamic(batch_size: int = 9, num_batches: int = 100, 
                                   num_classes: int = 10, device: torch.device = None,
                                   is_train: bool = True) -> List[Dict[str, torch.Tensor]]:
    """
    生成动态训练数据（适配原有训练器接口）
    核心修复：
    1. 关闭pin_memory（GPU时）或保持开启（CPU时）
    2. 数据先在CPU处理，再移到目标设备
    3. 数据有业务意义，非全随机
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建动态数据集
    dataset = create_dynamic_dataset(
        batch_size=batch_size,
        num_batches=num_batches,
        num_classes=num_classes,
        target_device=device,
        is_train=is_train
    )
    
    # 2. 创建DataLoader（核心修复：GPU时关闭pin_memory）
    use_pin_memory = device.type == "cpu"  # 只有CPU才使用pin_memory
    dataloader = create_mhd_extend_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=use_pin_memory,  # 修复：GPU时关闭pin_memory
        drop_last=True
    )
    
    # 3. 转换为原有格式（List[Dict]）
    data_list = []
    for batch in dataloader:
        # 确保批次大小正确
        if batch["main_input"].shape[0] == batch_size:
            # 确保数据在正确设备上
            data_batch = {}
            for k, v in batch.items():
                data_batch[k] = v.to(device, non_blocking=True)
            data_list.append(data_batch)
        
        # 控制生成批次数量
        if len(data_list) >= num_batches:
            break
    
    return data_list

# ===================== 3. 主训练流程 =====================
if __name__ == "__main__":
    # 基础配置
    BATCH_SIZE = 11
    NUM_CLASSES = 10
    EPOCHS = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置全局随机种子，保证可复现
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # ===================== 3.1 构建所有子图 =====================
    print("=== 1. 构建子图 ===")
    # 1. 主图（节点名已与FL对齐）
    main_graph = build_main_graph(
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        device=device
    )
    print(f"✅ 主图构建完成：节点数={len(main_graph.nodes)}, 边数={len(main_graph.edges)}")
    
    # 2. Focal Loss图
    fl_graph = example1_focal_loss_graph(
        alpha=0.25,
        gamma=2.0,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
    )
    for topo in fl_graph.topos:
        topo.value = topo.value.to(device)
    print(f"✅ Focal Loss图构建完成：节点数={len(fl_graph.nodes)}, 边数={len(fl_graph.edges)}")
    
    # 3. 指标图（节点名已与FL对齐）
    metric_graph = build_metric_graph(
        batch_size=BATCH_SIZE,
        device=device
    )
    print(f"✅ 指标图构建完成：节点数={len(metric_graph.nodes)}, 边数={len(metric_graph.edges)}")
    
    # ===================== 3.2 合并为全局超图（核心修改） =====================
    print("\n=== 2. 合并全局超图 ===")
    # 1. 图列表
    graph_list = [
        ("main_", main_graph),
        ("fl_", fl_graph),
        ("metric_", metric_graph)
    ]
    
    # 2. 节点组配置（关键：使用完全相同的节点名，无需映射）
    # 因为主图/指标图的节点名已和FL图完全对齐
    node_group = (
        {"main_logits_pred","fl_logits_pred"},  # 主图+FL图共用该节点
        {"metric_ce_loss","fl_ce_loss"}       # FL图+指标图共用该节点
    )
    
    # 3. 合并图（此时group_nodes必然非空）
    global_graph = MHD_Graph.merge_graph(  # 核心修改：改为merge_graph
        graph_list=graph_list,
        node_group=node_group,
        device=device
    )
    # 注册合并后图的所有参数
    global_graph._register_all_params()
    print(f"✅ 全局图合并完成：总节点数={len(global_graph.nodes)}, 总边数={len(global_graph.edges)}")
    
    # 打印合并后的节点名和边名（关键：用于后续配置）
    print("\n🔍 合并后的节点名列表：")
    node_name_list = []
    for node in sorted(global_graph.nodes, key=lambda x: x.id):
        node_name_list.append(node.name)
        print(f"  - {node.name} (ID: {node.id})")
    
    print("\n🔍 合并后的边名列表：")
    edge_name_list = []
    for edge in sorted(global_graph.edges, key=lambda x: x.id):
        edge_name_list.append(edge.name)
        print(f"  - {edge.name} (ID: {edge.id})")
    
    # ===================== 3.3 配置指标监控器 =====================
    print("\n=== 3. 配置监控器 ===")
    monitor = MHD_Monitor(
        monitor_nodes=[
            "fl_final_fl",                  # FL损失（min）
            "fl_ce_loss",                   # CE损失
            "metric_node_metric_ce"         # 反向CE指标（max）
        ],
        monitor_edges=[
            "main_e_input_to_hidden",       # 修正：匹配合并后的边名
            "main_e_hidden_to_logits_pred", # 修正：匹配合并后的边名
            "fl_edge_pt_to_ce"              # 修正：匹配合并后的边名
        ],
    )
    print("✅ 监控器配置完成")
    
    # ===================== 3.4 配置优化器 =====================
    print("\n=== 4. 配置优化器 ===")
    edge_optim_config = {
        "main_e_input_to_hidden": {        # 修正：匹配合并后的边名
            "lr": 0.0005,
            "weight_decay": 1e-4
        },
        "main_e_hidden_to_logits_pred": {  # 修正：匹配合并后的边名
            "lr": 0.001,
            "weight_decay": 1e-5
        },
        "fl_edge_pt_to_ce": {              # 修正：匹配合并后的边名
            "lr": 0.0008,
            "weight_decay": 1e-4
        }
    }
    
    optimizer = create_optimizer(
        mhd_graph=global_graph,  # 核心修改：改为MHD_Graph
        edge_optim_config=edge_optim_config,
        default_optimizer_type="adam",
        default_lr=0.001,
        default_weight_decay=1e-4
    )
    
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=2,
        gamma=0.9
    )
    print("✅ 优化器配置完成")
    
    # ===================== 3.5 生成动态数据（优化版） =====================
    print("\n=== 5. 生成动态数据 ===")
    train_data = generate_training_data_dynamic(
        batch_size=BATCH_SIZE,
        num_batches=100,
        num_classes=NUM_CLASSES,
        device=device,
        is_train=True
    )
    
    eval_data = generate_training_data_dynamic(
        batch_size=BATCH_SIZE,
        num_batches=20,
        num_classes=NUM_CLASSES,
        device=device,
        is_train=False
    )
    print(f"✅ 动态数据生成完成：训练集={len(train_data)}批次, 验证集={len(eval_data)}批次")
    
    # 验证数据质量：打印第一批数据的基本信息
    if train_data:
        first_batch = train_data[0]
        print(f"\n📊 第一批训练数据信息：")
        print(f"   main_input: 形状={first_batch['main_input'].shape}, 均值={first_batch['main_input'].mean():.4f}, 标准差={first_batch['main_input'].std():.4f}")
        print(f"   fl_onehot_gt: 形状={first_batch['fl_onehot_gt'].shape}, 非零值数量={first_batch['fl_onehot_gt'].sum().item()}")
    
    # ===================== 3.6 初始化训练器 =====================
    print("\n=== 6. 初始化训练器 ===")
    trainer = MHD_Trainer(
        mhd_graph=global_graph,  # 核心修改：改为MHD_Graph
        optimizer=optimizer,
        monitor=monitor,
        save_dir="./focal_loss_ckpts",
        target_loss_node="fl_final_fl",
        target_metric_node="metric_node_metric_ce",
        device=device,
        grad_clip_norm=1.0,
        lr_scheduler=scheduler
    )
    print("✅ 训练器初始化完成")
    
    # ===================== 3.7 开始训练 =====================
    print("\n=== 7. 开始训练 ===")
    trainer.train(
        train_data=train_data,
        eval_data=eval_data,
        epochs=EPOCHS
    )
    
    # ===================== 3.8 测试 =====================
    print("\n=== 8. 加载最佳模型测试 ===")
    try:
        trainer.load_checkpoint(load_best_loss=True)
        test_batch = eval_data[0]
        
        # 适配新版eval_step返回格式：loss_value, metric_value
        loss_value, metric_value = trainer.eval_step(test_batch)
        
        trainer.logger.info("\n" + "="*80)
        trainer.logger.info("📝 最佳模型测试结果")
        trainer.logger.info("="*80)
        trainer.logger.info(f"📊 测试批次损失: {loss_value:.6f}")
        trainer.logger.info(f"📊 测试批次指标: {metric_value:.6f}")
        
        # 获取监控指标
        node_metrics = monitor.monitor_node(global_graph, prefix="test_")
        edge_metrics = monitor.monitor_edge(global_graph, prefix="test_", train_mode=False)
        all_metrics = {**node_metrics, **edge_metrics}
        trainer.logger.info(monitor.format_metrics(all_metrics))
        
        key_nodes = ["fl_final_fl", "fl_ce_loss", "metric_node_metric_ce"]
        trainer.logger.info("\n🔍 关键节点详细值:")
        for node_name in key_nodes:
            node = get_obj_by_name(node_name, global_graph.nodes)
            if node:
                trainer.logger.info(
                    f"  {node_name}: 均值={node.value.mean():.6f}, "
                    f"总和={node.value.sum():.6f}, 最大值={node.value.max():.6f}"
                )
        
        # 打印节点值信息（验证动态数据加载）
        trainer.logger.info("\n🔍 动态数据加载验证:")
        input_node = get_obj_by_name("main_input", global_graph.nodes)
        if input_node:
            trainer.logger.info(f"  main_input: 形状={input_node.value.shape}, 均值={input_node.value.mean():.6f}")
        
        gt_node = get_obj_by_name("fl_onehot_gt", global_graph.nodes)
        if gt_node:
            trainer.logger.info(f"  fl_onehot_gt: 形状={gt_node.value.shape}, 均值={gt_node.value.mean():.6f}")
        
    except Exception as e:
        trainer.logger.error(f"❌ 测试阶段出错: {str(e)}", exc_info=True)
    
    print("\n🎉 训练流程全部完成！")
