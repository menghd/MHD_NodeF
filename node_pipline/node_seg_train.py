import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import numpy as np
import json
from sklearn.model_selection import KFold
from copy import deepcopy
import sys
import random
sys.path.append(r"C:\Users\souray\Desktop\Codes")
from node_toolkit.node_net import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset, MinMaxNormalize, ZScoreNormalize, RandomRotate, RandomFlip, RandomShift, RandomZoom
from node_toolkit.node_utils import train, validate
from node_toolkit.node_results import (
    node_lp_loss, node_focal_loss, node_dice_loss, node_iou_loss,
    node_recall_metric, node_precision_metric, node_f1_metric, node_dice_metric, node_iou_metric, node_mse_metric
)

# Global seed constant
GLOBAL_SEED = 42

def set_global_seed(seed, deterministic=True):
    """
    Set global random seed for reproducibility across random, numpy, torch, and torch.cuda.
    设置全局随机种子，确保 random、numpy、torch 和 torch.cuda 的可重现性。

    Args:
        seed: Seed value.
        deterministic: If True, enable deterministic CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class WarmupCosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    """
    Learning rate scheduler with warmup and cosine annealing.
    带预热和余弦退火的学习率调度器。
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
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def worker_init_fn(worker_id):
    """
    Initialize worker with a deterministic seed for reproducibility.
    使用确定性种子初始化工作进程以确保可重现性。

    Args:
        worker_id: ID of the worker process.
    """
    worker_seed = GLOBAL_SEED + worker_id
    set_global_seed(worker_seed, deterministic=True)

def main():
    """
    Main function to run the training pipeline.
    运行训练流水线的主函数。
    """
    set_global_seed(GLOBAL_SEED, deterministic=True)

    # Data and save paths
    data_dir = r"C:\Users\souray\Desktop\Tr"
    save_dir = r"C:\Users\souray\Desktop\MHDNet0422"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 4
    num_dimensions = 3
    num_epochs = 200
    learning_rate = 1e-3
    k_folds = 5
    validation_interval = 1
    patience = 200
    warmup_epochs = 10
    num_workers = 0

    # Subnetwork 12 (Segmentation task: Plaque, binary segmentation)
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
            "convs": [(2, 3, 3, 3)], "norms": ["batch"], "acts": ["leakyrelu"], "feature_size": (64, 64, 64)}},
        "e5": {"src_nodes": [4], "dst_nodes": [8], "params": {
            "convs": [None], "norms": [None], "acts": [None], "feature_size": (64, 64, 64)}},
    }
    in_nodes_segmentation = [0, 1, 2, 3, 4]
    out_nodes_segmentation = [0, 1, 2, 3, 4, 8]

    # Subnetwork 13 (Target node for reshaped features)
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
    node_mapping = [
        (100, "segmentation", 0), (101, "segmentation", 1),
        (102, "segmentation", 2), (103, "segmentation", 3), (104, "segmentation", 4), (508, "segmentation", 8),
        (600, "target", 0)
    ]

    # Instantiate subnetworks
    sub_networks_configs = {
        "segmentation": (node_configs_segmentation, hyperedge_configs_segmentation, in_nodes_segmentation, out_nodes_segmentation, node_dtype_segmentation),
        "target": (node_configs_target, hyperedge_configs_target, in_nodes_target, out_nodes_target, node_dtype_target),
    }
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions, node_dtype)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes, node_dtype) in sub_networks_configs.items()
    }

    # Global input and output nodes
    in_nodes = [100, 101, 102, 103, 104, 600]
    out_nodes = [104, 508, 600]

    # Node suffix mapping
    node_suffix = [
        (100, "0000"), (101, "0001"), (102, "0002"), (103, "0003"), (104, "0004"),
        (600, "0004")
    ]

    # Instantiate transformations
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
    node_transforms = {
        100: [random_rotate1, random_flip, random_shift, random_zoom1, min_max_normalize, z_score_normalize],
        101: [random_rotate1, random_flip, random_shift, random_zoom2, min_max_normalize, z_score_normalize],
        102: [random_rotate1, random_flip, random_shift, random_zoom3, min_max_normalize, z_score_normalize],
        103: [random_rotate1, random_flip, random_shift, random_zoom1, min_max_normalize, z_score_normalize],
        104: [random_rotate2, random_flip, random_shift, random_zoom2],
        600: [random_rotate2, random_flip, random_shift, random_zoom2],
        601: [], 602: [], 603: [], 604: [], 605: [], 606: [], 607: [], 608: [], 609: [],
    }

    # Task configuration
    task_configs = {
        "segmentation_plaque": {
            "loss": [
                {"fn": node_dice_loss, "src_node": 508, "target_node": 600, "weight": 1.0, "params": {}},
                {"fn": node_iou_loss, "src_node": 508, "target_node": 600, "weight": 0.5, "params": {}},
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
    for suffix, case_ids in suffix_case_ids.items():
        missing = set(case_ids) - common_case_ids
        if missing:
            print(f"Warning: Incomplete cases for suffix {suffix}: {sorted(list(missing))}")

    # K-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_case_ids)):
        print(f"Fold {fold + 1}")

        # Get train and validation case IDs
        train_case_ids = [all_case_ids[idx] for idx in train_ids]
        val_case_ids = [all_case_ids[idx] for idx in val_ids]

        # Generate global random order for training
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
                worker_init_fn=worker_init_fn,
                shuffle=False,  # Disable shuffle to respect OrderedSampler
                persistent_workers=(num_workers > 0)
            )
            dataloaders_val[node] = DataLoader(
                datasets_val[node],
                batch_size=batch_size,
                sampler=OrderedSampler(val_indices),
                num_workers=num_workers,
                drop_last=True,
                worker_init_fn=worker_init_fn,
                shuffle=False,  # Disable shuffle to respect OrderedSampler
                persistent_workers=(num_workers > 0)
            )

        # Model, optimizer, and scheduler
        model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, T_max=num_epochs, eta_min=1e-6)

        # Early stopping
        best_val_loss = float("inf")
        epochs_no_improve = 0
        log = {"fold": fold + 1, "epochs": []}

        for epoch in range(num_epochs):
            # Generate a batch seed for each epoch
            epoch_seed = GLOBAL_SEED + epoch
            np.random.seed(epoch_seed)
            batch_seeds = np.random.randint(0, 1000000, size=len(dataloaders_train[node]))

            for batch_idx in range(len(dataloaders_train[node])):
                batch_seed = int(batch_seeds[batch_idx])
                for node in datasets_train:
                    datasets_train[node].set_batch_seed(batch_seed, batch_idx)
                for node in datasets_val:
                    datasets_val[node].set_batch_seed(batch_seed, batch_idx)

            train_loss, train_task_losses, train_metrics = train(
                model, dataloaders_train, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, 
            )

            epoch_log = {"epoch": epoch + 1, "train_loss": train_loss, "train_task_losses": train_task_losses, "train_metrics": train_metrics}

            if (epoch + 1) % validation_interval == 0:
                val_loss, val_task_losses, val_metrics = validate(
                    model, dataloaders_val, task_configs, out_nodes, epoch, num_epochs,
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
