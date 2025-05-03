"""
MHD_Nodet Project - Training Module
===================================
This module implements the training pipeline for the MHD_Nodet project,
including data preparation, model training, and cross-validation.

项目：MHD_Nodet - 训练模块
本模块实现了 MHD_Nodet 项目的训练流水线，包括数据准备、模型训练和交叉验证。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import numpy as np
import json
from sklearn.model_selection import KFold
from copy import deepcopy
import sys
sys.path.append(r"C:\Users\souray\Desktop\Codes")
from node_toolkit.node_net import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset, MinMaxNormalize, ZScoreNormalize, RandomRotate, RandomFlip, RandomShift, RandomZoom
from node_toolkit.node_utils import train, validate
from node_toolkit.node_results import (
    node_lp_loss, node_focal_loss, node_dice_loss, node_iou_loss,
    node_recall_metric, node_precision_metric, node_f1_metric, node_dice_metric, node_iou_metric, node_mse_metric
)

class WarmupCosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    """
    Learning rate scheduler with warmup and cosine annealing.
    带预热和余弦退火的学习率调度器。

    Args:
        optimizer: Optimizer instance.
        warmup_epochs: Number of warmup epochs.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate.
        last_epoch: Last epoch index.
    """
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        return super().get_lr()

class OrderedSampler(Sampler):
    """
    Custom sampler to enforce a specific order of indices.
    自定义采样器以强制执行特定的索引顺序。

    Args:
        indices: List of indices in desired order.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def worker_init_fn(worker_id):
    """
    Initialize worker with a unique seed for reproducibility.
    使用唯一种子初始化工作进程以确保可重现性。

    Args:
        worker_id: ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)

def main():
    """
    Main function to run the training pipeline.
    运行训练流水线的主函数。
    """
    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and save paths
    # 数据和保存路径
    data_dir = r"C:\Users\souray\Desktop\Tr"
    save_dir = r"C:\Users\souray\Desktop\MHDNet0422"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    # 超参数
    batch_size = 4
    num_dimensions = 3
    num_epochs = 200
    learning_rate = 1e-3
    k_folds = 5
    validation_interval = 1
    patience = 200
    warmup_epochs = 10
    num_workers = 1

    # Subnetwork 12 (Segmentation task: Plaque, binary segmentation)
    # 子网络 12（分割任务：斑块，二值分割）
    node_configs_segmentation = {
        0: (1, 64, 64, 64), 1: (1, 64, 64, 64), 2: (1, 64, 64, 64), 3: (1, 64, 64, 64), 4: (2, 64, 64, 64),
        5: (64, 64, 64, 64), 6: (128, 32, 32, 32), 7: (64, 64, 64, 64), 8: (2, 64, 64, 64)
    }
    node_dtype_segmentation = {4: "long"}
    hyperedge_configs_segmentation = {
        "e1": {"src_nodes": [0, 1, 2, 3, 4], "dst_nodes": [5], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e2": {"src_nodes": [5], "dst_nodes": [6], "params": {
            "convs": [(128, 3, 3, 3), (128, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32), "out_p": 1}},
        "e3": {"src_nodes": [5, 6], "dst_nodes": [7], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64), "out_p": 1}},
        "e4": {"src_nodes": [7], "dst_nodes": [8], "params": {
            "convs": [(2, 3, 3, 3)], "norms": ["batch"], "acts": ["relu"], "feature_size": (64, 64, 64)}},
        "e5": {"src_nodes": [4], "dst_nodes": [8], "params": {
            "convs": [None], "norms": [None], "acts": [None], "feature_size": (64, 64, 64)}},
    }
    in_nodes_segmentation = [0, 1, 2, 3, 4]
    out_nodes_segmentation = [0, 1, 2, 3, 4, 8]

    # Subnetwork 13 (Target node for reshaped features)
    # 子网络 13（存储调整形状特征的目标节点）
    node_configs_target = {
        0: (2, 64, 64, 64)
    }
    node_dtype_target = {
        0: "long"
    }
    hyperedge_configs_target = {}
    in_nodes_target = [0]
    out_nodes_target = [0]

    # Global node mapping
    # 全局节点映射
    node_mapping = [
        (100, "segmentation", 0), (101, "segmentation", 1),
        (102, "segmentation", 2), (103, "segmentation", 3), (104, "segmentation", 4), (508, "segmentation", 8),
        (600, "target", 0)
    ]

    # Instantiate subnetworks
    # 实例化子网络
    sub_networks_configs = {
        "segmentation": (node_configs_segmentation, hyperedge_configs_segmentation, in_nodes_segmentation, out_nodes_segmentation, node_dtype_segmentation),
        "target": (node_configs_target, hyperedge_configs_target, in_nodes_target, out_nodes_target, node_dtype_target),
    }
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions, node_dtype)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes, node_dtype) in sub_networks_configs.items()
    }

    # Global input and output nodes
    # 全局输入和输出节点
    in_nodes = [100, 101, 102, 103, 104, 600]
    out_nodes = [100, 101, 102, 103, 104, 508, 600]

    # Node suffix mapping
    # 节点后缀映射
    node_suffix = [
        (100, "0000"), (101, "0001"), (102, "0002"), (103, "0003"), (104, "0004"),
        (600, "0004")
    ]

    # Instantiate transformations
    # 实例化变换
    random_rotate1 = RandomRotate(max_angle=5)
    random_rotate2 = RandomRotate(max_angle=5)
    random_flip = RandomFlip()
    random_shift = RandomShift(max_shift=5)
    random_zoom1 = RandomZoom(zoom_range=(0.9, 1.1))
    random_zoom2 = RandomZoom(zoom_range=(0.9, 1.1))
    random_zoom3 = RandomZoom(zoom_range=(0.9, 1.1))
    min_max_normalize = MinMaxNormalize()
    z_score_normalize = ZScoreNormalize()

    # Node transformation configuration
    # 节点变换配置
    node_transforms = {
        # 100: [random_rotate1, random_flip, random_shift, random_zoom1, min_max_normalize, z_score_normalize],
        # 101: [random_rotate1, random_flip, random_shift, random_zoom2, min_max_normalize, z_score_normalize],
        # 102: [random_rotate1, random_flip, random_shift, random_zoom3, min_max_normalize, z_score_normalize],
        # 103: [random_rotate1, random_flip, random_shift, random_zoom1, min_max_normalize, z_score_normalize],
        # 104: [random_rotate2, random_flip, random_shift, random_zoom2],
        # 600: [random_rotate2, random_flip, random_shift, random_zoom2],
        # 601: [], 602: [], 603: [], 604: [], 605: [], 606: [], 607: [], 608: [], 609: [],
    }

    # Task configuration
    # 任务配置
    task_configs = {
        "segmentation_plaque": {
            "loss": [
                {"fn": node_dice_loss, "src_node": 508, "target_node": 600, "weight": 1.0, "params": {}},
                {"fn": node_dice_loss, "src_node": 104, "target_node": 600, "weight": 1.0, "params": {}},
                {"fn": node_iou_loss, "src_node": 508, "target_node": 600, "weight": 0.5, "params": {}},
                {"fn": node_iou_loss, "src_node": 104, "target_node": 600, "weight": 0.5, "params": {}},
                {"fn": node_lp_loss, "src_node": 104, "target_node": 600, "weight": 0.5, "params": {}},
                {"fn": node_lp_loss, "src_node": 103, "target_node": 100, "weight": 0.5, "params": {}},
                {"fn": node_lp_loss, "src_node": 103, "target_node": 101, "weight": 0.5, "params": {}},
                {"fn": node_lp_loss, "src_node": 103, "target_node": 102, "weight": 0.5, "params": {}},
                {"fn": node_lp_loss, "src_node": 508, "target_node": 600, "weight": 0.5, "params": {}},
            ],
            "metric": [
                {"fn": node_dice_metric, "src_node": 508, "target_node": 600, "params": {}},
                {"fn": node_iou_metric, "src_node": 508, "target_node": 600, "params": {}},
                {"fn": node_recall_metric, "src_node": 508, "target_node": 600, "params": {}},
                {"fn": node_precision_metric, "src_node": 508, "target_node": 600, "params": {}},
                {"fn": node_f1_metric, "src_node": 508, "target_node": 600, "params": {}},
            ],
        },
    }

    # Collect common case IDs
    # 收集共同的 case IDs
    all_files = sorted(os.listdir(data_dir))
    suffix_to_nodes = {}
    for node, suffix in node_suffix:
        if suffix not in suffix_to_nodes:
            suffix_to_nodes[suffix] = []
        suffix_to_nodes[suffix].append(node)

    suffix_case_ids = {}
    for suffix in suffix_to_nodes:
        case_ids = set()
        for file in all_files:
            if file.startswith('case_') and (file.endswith(f'_{suffix}.nii.gz') or file.endswith(f'_{suffix}.csv')):
                case_id = file.split('_')[1]
                case_ids.add(case_id)
        suffix_case_ids[suffix] = sorted(list(case_ids))

    common_case_ids = set.intersection(*(set(case_ids) for case_ids in suffix_case_ids.values()))
    if not common_case_ids:
        raise ValueError("No common case_ids found across all suffixes!")
    all_case_ids = sorted(list(common_case_ids))

    # Log incomplete cases
    # 记录不完整的 case
    for suffix, case_ids in suffix_case_ids.items():
        missing = set(case_ids) - common_case_ids
        if missing:
            print(f"Warning: Incomplete cases for suffix {suffix}: {sorted(list(missing))}")

    # K-fold cross-validation
    # K 折交叉验证
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_case_ids)):
        print(f"Fold {fold + 1}")

        # Get train and validation case IDs
        # 获取训练和验证的 case IDs
        train_case_ids = [all_case_ids[idx] for idx in train_ids]
        val_case_ids = [all_case_ids[idx] for idx in val_ids]

        # Generate global random order for training
        # 为训练集生成全局随机顺序
        train_case_id_order = np.random.permutation(train_case_ids).tolist()
        val_case_id_order = val_case_ids

        split_info = {
            "fold": fold + 1,
            "train_case_ids": train_case_ids,
            "val_case_ids": val_case_ids,
            "train_case_id_order": train_case_id_order,
            "val_case_id_order": val_case_id_order,
            "train_count": len(train_case_ids),
            "val_count": len(val_case_ids),
        }
        split_save_path = os.path.join(save_dir, f"fold_{fold + 1}_split.json")
        with open(split_save_path, "w") as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split saved to {split_save_path}")

        # Create datasets
        # 创建数据集
        datasets_train = {}
        datasets_val = {}
        for node, suffix in node_suffix:
            target_shape = None
            for g_node, sub_net_name, sub_node_id in node_mapping:
                if g_node == node:
                    target_shape = sub_networks[sub_net_name].node_configs[sub_node_id]
                    break
            if target_shape is None:
                raise ValueError(f"Node {node} not found in node_mapping")
            datasets_train[node] = NodeDataset(
                data_dir, node, suffix, target_shape, node_transforms.get(node, []),
                node_mapping=node_mapping, sub_networks=sub_networks,
                case_ids=train_case_ids, case_id_order=train_case_id_order
            )
            datasets_val[node] = NodeDataset(
                data_dir, node, suffix, target_shape, node_transforms.get(node, []),
                node_mapping=node_mapping, sub_networks=sub_networks,
                case_ids=val_case_ids, case_id_order=val_case_id_order
            )

        # Create DataLoaders with custom sampler and worker initialization
        # 使用自定义采样器和工作进程初始化创建 DataLoader
        dataloaders_train = {}
        dataloaders_val = {}
        for node in datasets_train:
            train_indices = list(range(len(datasets_train[node])))
            val_indices = list(range(len(datasets_val[node])))
            dataloaders_train[node] = DataLoader(
                datasets_train[node],
                batch_size=batch_size,
                sampler=OrderedSampler(train_indices),
                num_workers=num_workers,
                drop_last=True,
                worker_init_fn=worker_init_fn
            )
            dataloaders_val[node] = DataLoader(
                datasets_val[node],
                batch_size=batch_size,
                sampler=OrderedSampler(val_indices),
                num_workers=num_workers,
                drop_last=True,
                worker_init_fn=worker_init_fn
            )

        # Model, optimizer, and scheduler
        # 模型、优化器和调度器
        model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, T_max=num_epochs, eta_min=1e-6)

        # Early stopping
        # 早停
        best_val_loss = float("inf")
        epochs_no_improve = 0
        log = {"fold": fold + 1, "epochs": []}

        for epoch in range(num_epochs):
            # Generate a batch seed for each epoch
            # 为每个 epoch 生成批次种子
            epoch_seed = seed + epoch
            np.random.seed(epoch_seed)
            batch_seeds = np.random.randint(0, 1000000, size=len(dataloaders_train[node]))

            for batch_idx in range(len(dataloaders_train[node])):
                batch_seed = int(batch_seeds[batch_idx])
                for node in datasets_train:
                    datasets_train[node].set_batch_seed(batch_seed)
                for node in datasets_val:
                    datasets_val[node].set_batch_seed(batch_seed)

            train_loss, train_task_losses, train_metrics = train(
                model, dataloaders_train, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, node_transforms
            )

            epoch_log = {"epoch": epoch + 1, "train_loss": train_loss, "train_task_losses": train_task_losses, "train_metrics": train_metrics}

            if (epoch + 1) % validation_interval == 0:
                val_loss, val_task_losses, val_metrics = validate(
                    model, dataloaders_val, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping
                )

                epoch_log.update({"val_loss": val_loss, "val_task_losses": val_task_losses, "metrics": val_metrics})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(save_dir, f"model_fold{fold + 1}_best.pth")
                    torch.save(model.state_dict(), save_path)
                    config = {
                        "sub_networks": {name: {
                            "node_configs": {k: list(v) for k, v in cfg[0].items()},
                            "hyperedge_configs": deepcopy(cfg[1]),
                            "in_nodes": cfg[2],
                            "out_nodes": cfg[3],
                            "node_dtype": cfg[4],
                        } for name, cfg in sub_networks_configs.items()},
                        "node_mapping": node_mapping,
                        "in_nodes": in_nodes,
                        "out_nodes": out_nodes,
                        "num_dimensions": num_dimensions,
                        "node_suffix": node_suffix,
                        "node_transforms": {str(k): [t.__class__.__name__ for t in v] for k, v in node_transforms.items()},
                        "task_configs": {
                            task: {
                                "loss": [{"fn": cfg["fn"].__name__, "src_node": cfg["src_node"], "target_node": cfg["target_node"], "weight": cfg["weight"], "params": cfg["params"]} for cfg in config["loss"]],
                                "metric": [{"fn": cfg["fn"].__name__, "src_node": cfg["src_node"], "target_node": cfg["target_node"], "params": cfg["params"]} for cfg in config["metric"]],
                            } for task, config in task_configs.items()
                        },
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "k_folds": k_folds,
                        "validation_interval": validation_interval,
                        "patience": patience,
                        "warmup_epochs": warmup_epochs,
                    }
                    config_save_path = os.path.join(save_dir, f"model_config_fold{fold + 1}.json")
                    with open(config_save_path, "w") as f:
                        json.dump(config, f, indent=4)
                    print(f"Model saved to {save_path}, Config saved to {config_save_path}")
                else:
                    epochs_no_improve += validation_interval
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            scheduler.step()
            log["epochs"].append(epoch_log)

        log_save_path = os.path.join(save_dir, f"training_log_fold{fold + 1}.json")
        with open(log_save_path, "w") as f:
            json.dump(log, f, indent=4)
        print(f"Training log saved to {log_save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
