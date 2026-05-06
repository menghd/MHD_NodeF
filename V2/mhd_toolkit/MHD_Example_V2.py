# -*- coding: utf-8 -*-
"""
MHD_Example_V2.py - 适配重构后MHD框架版本
更新内容：
1. 适配新的MHD_Framework重构（initial_state/current_state设计）
2. 移除FixedMHD_Trainer，使用标准MHD_Trainer
3. 更新节点、边、拓扑创建方式
4. 修复训练和测试流程
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from typing import Set, Dict, Any, List, Callable
import os
import sys
import warnings

# 添加当前目录到Python路径，确保能导入MHD_Framework
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入MHD核心框架
from MHD_Framework_V2 import (
    MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph,
)

from MHD_Utils_V2 import (
    MHD_Trainer, MHD_Monitor, create_optimizer,
    MHD_Dataset, create_mhd_extend_dataloader,
    MHD_Augmentor, MHD_AugmentComposer,
    RandomRotate, RandomFlip, Normalize
)

# ===================== Focal Loss图定义 =====================
def example1_focal_loss_graph(alpha: float = 0.25, gamma: float = 2.0, batch_size: int = 8, num_classes: int = 10, device: torch.device = None):
    """
    构建Focal Loss计算图
    【修改】：增加device参数，确保图的设备与外部统一
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32
    EPS = 1e-4

    B, C = batch_size, num_classes
    alpha_val = alpha
    gamma_val = gamma

    torch.manual_seed(42)

    # 定义节点（使用分离的initial_state和current_state）
    nodes: Set[MHD_Node] = set()

    # 创建初始张量
    logits_initial = torch.ones(B, C, device=device, dtype=DTYPE)
    logits_current = logits_initial.clone()
    
    node_logits = MHD_Node(
        id=0,
        name="logits_pred",
        initial_state=logits_initial,
        current_state=logits_current,
        func={"head": "share", "tail": "sum"}
    )
    
    onehot_initial = torch.ones(B, C, device=device, dtype=DTYPE)
    onehot_current = onehot_initial.clone()
    node_onehot_gt = MHD_Node(
        id=1,
        name="onehot_gt",
        initial_state=onehot_initial,
        current_state=onehot_current,
        func={"head": "share", "tail": "sum"}
    )
    
    p_initial = torch.ones(B, C, device=device, dtype=DTYPE)
    p_current = p_initial.clone()
    node_p = MHD_Node(
        id=2,
        name="p",
        initial_state=p_initial,
        current_state=p_current,
        func={"head": "share", "tail": "sum"}
    )
    
    ptb_initial = torch.ones(B, C, device=device, dtype=DTYPE)
    ptb_current = ptb_initial.clone()
    node_p_t_before = MHD_Node(
        id=3,
        name="p_t_before",
        initial_state=ptb_initial,
        current_state=ptb_current,
        func={"head": "share", "tail": "mul"}
    )
    
    alphat_initial = torch.ones(B, 1, device=device, dtype=DTYPE)
    alphat_current = alphat_initial.clone()
    node_alpha_t = MHD_Node(
        id=4,
        name="alpha_t",
        initial_state=alphat_initial,
        current_state=alphat_current,
        func={"head": "share", "tail": "sum"}
    )
    
    pt_initial = torch.ones(B, 1, device=device, dtype=DTYPE)
    pt_current = pt_initial.clone()
    node_p_t = MHD_Node(
        id=5,
        name="p_t",
        initial_state=pt_initial,
        current_state=pt_current,
        func={"head": "share", "tail": "sum"}
    )
    
    ce_initial = torch.ones(B, 1, device=device, dtype=DTYPE)
    ce_current = ce_initial.clone()
    node_ce_loss = MHD_Node(
        id=6,
        name="ce_loss",
        initial_state=ce_initial,
        current_state=ce_current,
        func={"head": "share", "tail": "sum"}
    )
    
    ff_initial = torch.ones(B, 1, device=device, dtype=DTYPE)
    ff_current = ff_initial.clone()
    node_focal_factor = MHD_Node(
        id=7,
        name="focal_factor",
        initial_state=ff_initial,
        current_state=ff_current,
        func={"head": "share", "tail": "sum"}
    )
    
    fl_initial = torch.ones(B, 1, device=device, dtype=DTYPE)
    fl_current = fl_initial.clone()
    node_focal_loss = MHD_Node(
        id=8,
        name="focal_loss",
        initial_state=fl_initial,
        current_state=fl_current,
        func={"head": "share", "tail": "mul"}
    )
    
    final_initial = torch.ones(1, 1, device=device, dtype=DTYPE)
    final_current = final_initial.clone()
    node_final_fl = MHD_Node(
        id=9,
        name="final_fl",
        initial_state=final_initial,
        current_state=final_current,
        func={"head": "share", "tail": "sum"}
    )

    nodes.update([
        node_logits, node_onehot_gt, node_p, node_p_t_before,
        node_alpha_t, node_p_t, node_ce_loss, node_focal_factor,
        node_focal_loss, node_final_fl
    ])

    # 定义边（使用sequential_operation而非value）
    edges: Set[MHD_Edge] = set()
    edge_logits_to_p = MHD_Edge(
        id=0,
        name="edge_logits_to_p",
        sequential_operation=["softmax(dim=1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_p_to_ptb = MHD_Edge(
        id=1,
        name="edge_p_to_ptb",
        sequential_operation=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_onehot_to_ptb = MHD_Edge(
        id=2,
        name="edge_onehot_to_ptb",
        sequential_operation=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_onehot_to_alphat = MHD_Edge(
        id=3,
        name="edge_onehot_to_alphat",
        sequential_operation=[f"mul({alpha_val})", "sum(dim=1,keepdim=True)"],
        func={"in": "concat", "out": "split"}
    )
    edge_ptb_to_pt = MHD_Edge(
        id=4,
        name="edge_ptb_to_pt",
        sequential_operation=["sum(dim=1,keepdim=True)"],
        func={"in": "concat", "out": "split"}
    )
    edge_pt_to_ce = MHD_Edge(
        id=5,
        name="edge_pt_to_ce",
        sequential_operation=[f"add({EPS})", "log()", "mul(-1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_pt_to_ff = MHD_Edge(
        id=6,
        name="edge_pt_to_ff",
        sequential_operation=["mul(-1)", "add(1)", f"pow({gamma_val})"],
        func={"in": "concat", "out": "split"}
    )
    edge_alphat_to_fl = MHD_Edge(
        id=7,
        name="edge_alphat_to_fl",
        sequential_operation=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_ff_to_fl = MHD_Edge(
        id=8,
        name="edge_ff_to_fl",
        sequential_operation=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_ce_to_fl = MHD_Edge(
        id=9,
        name="edge_ce_to_fl",
        sequential_operation=["mul(1)"],
        func={"in": "concat", "out": "split"}
    )
    edge_fl_to_final = MHD_Edge(
        id=10,
        name="edge_fl_to_final",
        sequential_operation=["sum(dim=0,keepdim=True)"],
        func={"in": "concat", "out": "split"}
    )

    edges.update([
        edge_logits_to_p, edge_p_to_ptb, edge_onehot_to_ptb,
        edge_onehot_to_alphat, edge_ptb_to_pt, edge_pt_to_ce,
        edge_pt_to_ff, edge_alphat_to_fl, edge_ff_to_fl,
        edge_ce_to_fl, edge_fl_to_final
    ])

    # 拓扑矩阵（使用role_matrix和sort_matrix）
    role_matrix = torch.tensor([
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
    ], dtype=torch.int64, device=device)

    sort_matrix = torch.tensor([
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
    ], dtype=torch.int64, device=device)

    topos = {
        MHD_Topo(role_matrix=role_matrix, sort_matrix=sort_matrix)
    }

    fl_graph = MHD_Graph(
        nodes=nodes,
        edges=edges,
        topos=topos,
        device=device
    )

    fl_graph._register_all_params()
    return fl_graph

# ===================== 1. 构建标准MHD图组件 =====================
def build_main_graph(batch_size: int = 9, num_classes: int = 10, device: torch.device = None) -> MHD_Graph:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 创建节点集合
    input_initial = torch.ones(batch_size, 512, device=device, dtype=dtype)
    input_current = input_initial.clone()
    
    hidden_initial = torch.ones(batch_size, 256, device=device, dtype=dtype)
    hidden_current = hidden_initial.clone()
    
    logits_initial = torch.ones(batch_size, num_classes, device=device, dtype=dtype)
    logits_current = logits_initial.clone()

    nodes_graph: Set[MHD_Node] = {
        MHD_Node(
            id=0, name="input",
            initial_state=input_initial,
            current_state=input_current,
            func={"head": "share", "tail": "sum"},
        ),
        MHD_Node(
            id=1, name="hidden",
            initial_state=hidden_initial,
            current_state=hidden_current,
            func={"head": "share", "tail": "sum"},
        ),
        MHD_Node(
            id=2, name="logits_pred",
            initial_state=logits_initial,
            current_state=logits_current,
            func={"head": "share", "tail": "sum"},
        )
    }

    # 创建边集合
    sequential_op1 = [
        nn.Linear(512, 256, device=device, dtype=dtype),
        nn.BatchNorm1d(256, device=device),
        nn.ReLU(inplace=True)
    ]
    
    sequential_op2 = [
        nn.Linear(256, num_classes, device=device, dtype=dtype)
    ]

    edges_graph: Set[MHD_Edge] = {
        MHD_Edge(
            id=0, name="e_input_to_hidden",
            sequential_operation=sequential_op1,
            func={"in": "concat", "out": "split"},
        ),
        MHD_Edge(
            id=1, name="e_hidden_to_logits_pred",
            sequential_operation=sequential_op2,
            func={"in": "concat", "out": "split"},
        )
    }

    # 创建拓扑集合
    role_matrix = torch.tensor([
        [-1, 1, 0],
        [0, -1, 1]
    ], dtype=torch.int64, device=device)
    
    sort_matrix = torch.tensor([
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=torch.int64, device=device)

    topos_graph: Set[MHD_Topo] = {
        MHD_Topo(role_matrix=role_matrix, sort_matrix=sort_matrix)
    }

    main_graph = MHD_Graph(
        nodes=nodes_graph,
        edges=edges_graph,
        topos=topos_graph,
        device=device
    )

    main_graph._register_all_params()
    return main_graph

def build_metric_graph(batch_size: int = 9, device: torch.device = None) -> MHD_Graph:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ce_initial = torch.ones(batch_size, 1, device=device, dtype=dtype)
    ce_current = ce_initial.clone()
    
    metric_initial = torch.ones(batch_size, 1, device=device, dtype=dtype)
    metric_current = metric_initial.clone()

    nodes_graph: Set[MHD_Node] = {
        MHD_Node(
            id=0, name="ce_loss",
            initial_state=ce_initial,
            current_state=ce_current,
            func={"head": "share", "tail": "sum"},
        ),
        MHD_Node(
            id=1, name="node_metric_ce",
            initial_state=metric_initial,
            current_state=metric_current,
            func={"head": "share", "tail": "sum"},
        )
    }

    sequential_op = [
        'sum(dim=1, keepdim=True)',
        'mul(-1)'
    ]

    edges_graph: Set[MHD_Edge] = {
        MHD_Edge(
            id=0, name="e_metric_ce_calc",
            sequential_operation=sequential_op,
            func={"in": "concat", "out": "split"},
        )
    }

    # 拓扑矩阵
    role_matrix = torch.tensor([[-1, 1]], dtype=torch.int64, device=device)
    sort_matrix = torch.tensor([[1, 1]], dtype=torch.int64, device=device)

    topos_graph: Set[MHD_Topo] = {
        MHD_Topo(role_matrix=role_matrix, sort_matrix=sort_matrix)
    }

    metric_graph = MHD_Graph(
        nodes=nodes_graph,
        edges=edges_graph,
        topos=topos_graph,
        device=device
    )

    metric_graph._register_all_params()
    return metric_graph

# ===================== 2. 动态数据生成 =====================
class DynamicDataAugmentor(MHD_Augmentor):
    def __init__(self, noise_level: float = 0.1, seed: int = None):
        super().__init__(seed)
        self.noise_level = noise_level

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

def generate_class_centers(num_classes: int = 10, feature_dim: int = 512, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    centers = torch.randn(num_classes, feature_dim)
    centers = torch.nn.functional.normalize(centers, p=2, dim=1)
    centers = centers * 3.0
    return centers

def create_dynamic_dataset(batch_size: int = 9, num_batches: int = 100,
                           num_classes: int = 10, target_device: torch.device = None,
                           is_train: bool = True) -> MHD_Dataset:
    cpu_device = torch.device("cpu")
    if target_device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_info_list = list(range(num_batches * batch_size))

    feature_dim = 512
    class_centers = generate_class_centers(
        num_classes=num_classes,
        feature_dim=feature_dim,
        seed=42
    ).to(cpu_device)

    # 预生成随机标签
    torch.manual_seed(42)
    all_labels = torch.randint(0, num_classes, (len(sample_info_list),)).tolist()

    def load_main_input(sample_id: int) -> torch.Tensor:
        label = all_labels[sample_id]
        feature = class_centers[label].clone()
        noise_level = 0.1 if is_train else 0.01
        noise = torch.randn_like(feature) * noise_level
        feature = feature + noise
        feature = torch.nn.functional.normalize(feature, p=2, dim=0)
        return feature

    def load_fl_onehot_gt(sample_id: int) -> torch.Tensor:
        label = all_labels[sample_id]
        onehot = F.one_hot(torch.tensor(label, device=cpu_device), num_classes).float()
        return onehot

    if is_train:
        shared_aug = MHD_AugmentComposer([
            DynamicDataAugmentor(noise_level=0.05),
            Normalize(mean=0.0, std=3.0)
        ])
    else:
        shared_aug = MHD_AugmentComposer([
            Normalize(mean=0.0, std=3.0)
        ])

    node_configs = {
        "main_input": {
            "loader": load_main_input,
            "augmentor": shared_aug
        },
        "fl_onehot_gt": {
            "loader": load_fl_onehot_gt,
            "augmentor": None
        }
    }

    dataset = MHD_Dataset(
        sample_info_list=sample_info_list,
        node_configs=node_configs,
        base_seed=42,
        target_device=target_device
    )

    return dataset

def generate_training_data_dynamic(batch_size: int = 9, num_batches: int = 100,
                                   num_classes: int = 10, device: torch.device = None,
                                   is_train: bool = True) -> List[Dict[str, torch.Tensor]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = create_dynamic_dataset(
        batch_size=batch_size,
        num_batches=num_batches,
        num_classes=num_classes,
        target_device=device,
        is_train=is_train
    )

    use_pin_memory = device.type == "cpu"
    dataloader = create_mhd_extend_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=True
    )

    data_list = []
    for batch in dataloader:
        if batch["main_input"].shape[0] == batch_size:
            data_batch = {}
            for k, v in batch.items():
                data_batch[k] = v.to(device, non_blocking=True)
            data_list.append(data_batch)

        if len(data_list) >= num_batches:
            break

    return data_list

# ===================== 3. 主训练流程 =====================
if __name__ == "__main__":
    # 基础配置
    BATCH_SIZE = 11
    NUM_CLASSES = 10
    EPOCHS = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置全局随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # ===================== 3.1 构建所有子图 =====================
    print("=== 1. 构建子图 ===")
    main_graph = build_main_graph(
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        device=device
    )
    print(f"✅ 主图构建完成：节点数={len(main_graph.nodes)}, 边数={len(main_graph.edges)}")

    fl_graph = example1_focal_loss_graph(
        alpha=0.25,
        gamma=2.0,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        device=device
    )
    print(f"✅ Focal Loss图构建完成：节点数={len(fl_graph.nodes)}, 边数={len(fl_graph.edges)}")

    metric_graph = build_metric_graph(
        batch_size=BATCH_SIZE,
        device=device
    )
    print(f"✅ 指标图构建完成：节点数={len(metric_graph.nodes)}, 边数={len(metric_graph.edges)}")

    # ===================== 3.2 合并为全局超图 =====================
    print("\n=== 2. 合并全局超图 ===")
    graph_list = [
        ("main_", main_graph),
        ("fl_", fl_graph),
        ("metric_", metric_graph)
    ]

    node_group = (
        {"main_logits_pred", "fl_logits_pred"},
        {"metric_ce_loss", "fl_ce_loss"}
    )

    global_graph = MHD_Graph.merge_graph(
        graph_list=graph_list,
        node_group=node_group,
        device=device
    )
    global_graph._register_all_params()
    print(f"✅ 全局图合并完成：总节点数={len(global_graph.nodes)}, 总边数={len(global_graph.edges)}")

    # 打印合并后的节点名和边名
    print("\n🔍 合并后的节点名列表：")
    global_node_names = []
    for node in sorted(global_graph.nodes, key=lambda x: x.id):
        global_node_names.append(node.name)
        print(f"  - {node.name} (ID: {node.id})")

    print("\n🔍 合并后的边名列表：")
    global_edge_names = []
    for edge in sorted(global_graph.edges, key=lambda x: x.id):
        global_edge_names.append(edge.name)
        print(f"  - {edge.name} (ID: {edge.id})")

    # ===================== 3.3 配置指标监控器 =====================
    print("\n=== 3. 配置监控器 ===")
    # 根据实际合并后的节点名配置监控器
    target_monitor_nodes = [
        "fl_final_fl",  # 损失节点
        "fl_ce_loss_MERGE_metric_ce_loss",  # 合并后的CE损失节点
        "metric_node_metric_ce"  # 指标节点
    ]
    
    target_monitor_edges = [
        "main_e_input_to_hidden",
        "main_e_hidden_to_logits_pred", 
        "fl_edge_pt_to_ce"
    ]

    # 验证节点存在性
    for node_name in target_monitor_nodes:
        if node_name not in global_node_names:
            print(f"⚠️  警告：监控节点 '{node_name}' 不在全局图中，请检查！")
        else:
            print(f"✅ 监控节点 '{node_name}' 存在")

    # 验证边存在性
    for edge_name in target_monitor_edges:
        if edge_name not in global_edge_names:
            print(f"⚠️  警告：监控边 '{edge_name}' 不在全局图中，请检查！")
        else:
            print(f"✅ 监控边 '{edge_name}' 存在")

    monitor = MHD_Monitor(
        monitor_nodes=target_monitor_nodes,
        monitor_edges=target_monitor_edges,
    )
    print("✅ 监控器配置完成")

    # ===================== 3.4 配置优化器 =====================
    print("\n=== 4. 配置优化器 ===")
    edge_optim_config = {
        "main_e_input_to_hidden": {
            "lr": 0.0005,
            "weight_decay": 1e-4
        },
        "main_e_hidden_to_logits_pred": {
            "lr": 0.001,
            "weight_decay": 1e-5
        },
        "fl_edge_pt_to_ce": {
            "lr": 0.0008,
            "weight_decay": 1e-4
        }
    }

    # 验证优化器配置的边存在性
    for edge_name in edge_optim_config.keys():
        if edge_name not in global_edge_names:
            print(f"⚠️  警告：优化器配置边 '{edge_name}' 不在全局图中，配置将无效！")
        else:
            print(f"✅ 优化器配置边 '{edge_name}' 存在")

    optimizer = create_optimizer(
        mhd_graph=global_graph,
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

    # ===================== 3.5 生成动态数据 =====================
    print("\n=== 5. 生成动态数据 ===")
    train_data = generate_training_data_dynamic(
        batch_size=BATCH_SIZE,
        num_batches=50,
        num_classes=NUM_CLASSES,
        device=device,
        is_train=True
    )

    eval_data = generate_training_data_dynamic(
        batch_size=BATCH_SIZE,
        num_batches=10,
        num_classes=NUM_CLASSES,
        device=device,
        is_train=False
    )
    print(f"✅ 动态数据生成完成：训练集={len(train_data)}批次, 验证集={len(eval_data)}批次")

    if train_data:
        first_batch = train_data[0]
        print(f"\n📊 第一批训练数据信息：")
        print(f"   main_input: 形状={first_batch['main_input'].shape}, 均值={first_batch['main_input'].mean():.4f}")
        print(f"   fl_onehot_gt: 形状={first_batch['fl_onehot_gt'].shape}, 非零值数量={first_batch['fl_onehot_gt'].sum().item()}")

    # ===================== 3.6 初始化训练器 =====================
    print("\n=== 6. 初始化训练器 ===")
    # 使用标准的MHD_Trainer（不再需要FixedMHD_Trainer）
    # 【修改】：移除device参数，trainer会自动从global_graph获取设备
    trainer = MHD_Trainer(
        mhd_graph=global_graph,
        optimizer=optimizer,
        monitor=monitor,
        save_dir="./focal_loss_ckpts",
        target_loss_node="fl_final_fl",
        target_metric_node="metric_node_metric_ce",
        # device=device,  # 移除此参数
        grad_clip_norm=1.0,
        lr_scheduler=scheduler
    )

    global_graph.generate_mermaid()

    print("✅ 训练器初始化完成")

    # ===================== 3.7 开始训练 =====================
    print("\n=== 7. 开始训练 ===")
    try:
        trainer.train(
            train_data=train_data,
            eval_data=eval_data,
            epochs=EPOCHS
        )
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

    # ===================== 3.8 测试 =====================
    print("\n=== 8. 加载最佳模型测试 ===")
    try:
        if os.path.exists("./focal_loss_ckpts"):
            trainer.load_checkpoint(load_best_loss=True)
            test_batch = eval_data[0]

            loss_value, metric_value = trainer.eval_step(test_batch)

            trainer.logger.info("\n" + "="*80)
            trainer.logger.info("📝 最佳模型测试结果")
            trainer.logger.info("="*80)
            trainer.logger.info(f"📊 测试批次损失: {loss_value:.6f}")
            trainer.logger.info(f"📊 测试批次指标: {metric_value:.6f}")

            print(f"\n✅ 测试完成: 损失={loss_value:.6f}, 指标={metric_value:.6f}")
        else:
            print("⚠️  检查点目录不存在，跳过测试")
    except Exception as e:
        print(f"❌ 测试阶段出错: {str(e)}")

    print("\n🎉 程序执行完毕！")
    
