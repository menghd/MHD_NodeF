"""
MHD_Nodet Project - Training Module
===================================
This module implements the training pipeline for the MHD_Nodet project, integrating network, dataset, and evaluation components.
- Supports custom data loading from separate train and val directories, and batch-consistent augmentations.
- Includes learning rate scheduling (warmup + cosine annealing) and early stopping for robust training.
- Supports flexible HDNet weight saving and loading via save_hdnet and load_hdnet configurations.

项目：MHD_Nodet - 训练模块
本模块实现 MHD_Nodet 项目的训练流水线，集成网络、数据集和评估组件。
- 支持从单独的 train 和 val 目录加载自定义数据，以及批次一致的数据增强。
- 包含学习率调度（预热 + 余弦退火）和早停机制以确保稳健训练。
- 支持通过 save_hdnet 和 load_hdnet 配置灵活保存和加载 HDNet 权重。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from node_toolkit.node_net import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset, MinMaxNormalize, RandomRotate, RandomShift, RandomZoom, OneHot, OrderedSampler, worker_init_fn
from node_toolkit.node_utils import train, validate, WarmupCosineAnnealingLR
from node_toolkit.node_results import (
    node_focal_loss, node_lp_loss, node_recall_metric, node_precision_metric, 
    node_f1_metric, node_accuracy_metric, node_specificity_metric
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the training pipeline.
    运行训练流水线的主函数。
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and save paths
    base_data_dir = r"C:\Users\souray\Desktop\Tr_dependent"
    train_data_dir = os.path.join(base_data_dir, "train")
    val_data_dir = os.path.join(base_data_dir, "val")
    save_dir = r"C:\Users\souray\Desktop\MHDNet0620"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 16
    num_dimensions = 3
    num_epochs = 500
    learning_rate = 1e-3
    weight_decay = 1e-4
    validation_interval = 1
    patience = 50
    warmup_epochs = 0
    num_workers = 0

    # Save and load HDNet configurations
    save_hdnet = {
        "preprocess": os.path.join(save_dir, "preprocess.pth"),
        "resnet18": os.path.join(save_dir, "resnet18.pth"),
        "gt": os.path.join(save_dir, "gt.pth"),
    }
    load_hdnet = {}
    # load_hdnet = {
    #     "preprocess": r"C:\Users\souray\Desktop\MHDNet0619\preprocess.pth",
    #     "resnet18": r"C:\Users\souray\Desktop\MHDNet0619\resnet18.pth",
    #     "gt": r"C:\Users\souray\Desktop\MHDNet0619\gt.pth"
    # }

    # Preprocess HDNet configuration
    node_configs_preprocess = {
        "n0": (1, 64, 64, 64),
        "n1": (1, 64, 64, 64),
        "n2": (1, 64, 64, 64),
        "n3": (1, 64, 64, 64),
        "n4": (1, 64, 64, 64),
        "n5": (32, 64, 64, 64)
    }
    hyperedge_configs_preprocess = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [
                    torch.Size([32, 5, 3, 3, 3]),
                    torch.Size([32, 32, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear"
            }
        }
    }
    in_nodes_preprocess = ["n0", "n1", "n2", "n3", "n4"]
    out_nodes_preprocess = ["n5"]

    # ResNet18 HDNet configuration (exact replication of ResNet18)
    node_configs_resnet18 = {
        "n0": (32, 64, 64, 64),   # Input from preprocess
        "n1": (64, 32, 32, 32),   # After conv1 + maxpool
        "n2": (64, 32, 32, 32),   # Layer1 block1
        "n3": (64, 32, 32, 32),   # Layer1 block2
        "n4": (128, 16, 16, 16),  # Layer2 block1
        "n5": (128, 16, 16, 16),  # Layer2 block2
        "n6": (256, 8, 8, 8),     # Layer3 block1
        "n7": (256, 8, 8, 8),     # Layer3 block2
        "n8": (512, 4, 4, 4),     # Layer4 block1
        "n9": (512, 4, 4, 4),     # Layer4 block2
        "n10": (8, 1, 1, 1)       # Final output
    }
    hyperedge_configs_resnet18 = {
        "e1": {  # Conv1 + MaxPool
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([64, 32, 7, 7, 7])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["relu"],
                "feature_size": (32, 32, 32),
                "intp": "max"
            }
        },
        "e2": {  # Layer1 block1
            "src_nodes": ["n1"],
            "dst_nodes": ["n2"],
            "params": {
                "convs": [
                    torch.Size([64, 64, 3, 3, 3]),
                    torch.Size([64, 64, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "linear"
            }
        },
        "e3": {  # Layer1 block2 main path
            "src_nodes": ["n2"],
            "dst_nodes": ["n3"],
            "params": {
                "convs": [
                    torch.Size([64, 64, 3, 3, 3]),
                    torch.Size([64, 64, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "linear"
            }
        },
        "e4": {  # Layer1 block2 residual path
            "src_nodes": ["n1"],
            "dst_nodes": ["n3"],
            "params": {
                "convs": [torch.eye(64).reshape(64, 64, 1, 1, 1)],
                "reqs": [False],
                "norms": [None],
                "acts": [None],
                "feature_size": (32, 32, 32),
                "intp": "linear"
            }
        },
        "e5": {  # Layer2 block1
            "src_nodes": ["n3"],
            "dst_nodes": ["n4"],
            "params": {
                "convs": [
                    torch.Size([128, 64, 3, 3, 3]),
                    torch.Size([128, 128, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "max"
            }
        },
        "e6": {  # Layer2 block2 main path
            "src_nodes": ["n4"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [
                    torch.Size([128, 128, 3, 3, 3]),
                    torch.Size([128, 128, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "linear"
            }
        },
        "e7": {  # Layer2 block2 residual path
            "src_nodes": ["n3"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [torch.Size([128, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (16, 16, 16),
                "intp": "max"
            }
        },
        "e8": {  # Layer3 block1
            "src_nodes": ["n5"],
            "dst_nodes": ["n6"],
            "params": {
                "convs": [
                    torch.Size([256, 128, 3, 3, 3]),
                    torch.Size([256, 256, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "intp": "max"
            }
        },
        "e9": {  # Layer3 block2 main path
            "src_nodes": ["n6"],
            "dst_nodes": ["n7"],
            "params": {
                "convs": [
                    torch.Size([256, 256, 3, 3, 3]),
                    torch.Size([256, 256, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "intp": "linear"
            }
        },
        "e10": {  # Layer3 block2 residual path
            "src_nodes": ["n5"],
            "dst_nodes": ["n7"],
            "params": {
                "convs": [torch.Size([256, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (8, 8, 8),
                "intp": "max"
            }
        },
        "e11": {  # Layer4 block1
            "src_nodes": ["n7"],
            "dst_nodes": ["n8"],
            "params": {
                "convs": [
                    torch.Size([512, 256, 3, 3, 3]),
                    torch.Size([512, 512, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (4, 4, 4),
                "intp": "max"
            }
        },
        "e12": {  # Layer4 block2 main path
            "src_nodes": ["n8"],
            "dst_nodes": ["n9"],
            "params": {
                "convs": [
                    torch.Size([512, 512, 3, 3, 3]),
                    torch.Size([512, 512, 3, 3, 3])
                ],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (4, 4, 4),
                "intp": "linear"
            }
        },
        "e13": {  # Layer4 block2 residual path
            "src_nodes": ["n7"],
            "dst_nodes": ["n9"],
            "params": {
                "convs": [torch.Size([512, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (4, 4, 4),
                "intp": "max"
            }
        },
        "e14": {  # Final FC layer
            "src_nodes": ["n9"],
            "dst_nodes": ["n10"],
            "params": {
                "convs": [torch.Size([8, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        }
    }
    in_nodes_resnet18 = ["n0"]
    out_nodes_resnet18 = ["n10"]

    # GT HDNet configuration
    node_configs_gt = {
        "n0": (8, 1, 1, 1)
    }
    hyperedge_configs_gt = {}
    in_nodes_gt = ["n0"]
    out_nodes_gt = ["n0"]

    # Global node mapping
    node_mapping = [
        ("n100", "preprocess", "n0"),
        ("n101", "preprocess", "n1"),
        ("n102", "preprocess", "n2"),
        ("n103", "preprocess", "n3"),
        ("n104", "preprocess", "n4"),
        ("n105", "preprocess", "n5"),
        ("n105", "resnet18", "n0"),
        ("n115", "resnet18", "n10"),
        ("n116", "gt", "n0")
    ]

    # Instantiate subnetworks
    sub_networks_configs = {
        "preprocess": (node_configs_preprocess, hyperedge_configs_preprocess, in_nodes_preprocess, out_nodes_preprocess),
        "resnet18": (node_configs_resnet18, hyperedge_configs_resnet18, in_nodes_resnet18, out_nodes_resnet18),
        "gt": (node_configs_gt, hyperedge_configs_gt, in_nodes_gt, out_nodes_gt)
    }
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # Load pretrained weights for specified HDNets
    for net_name, weight_path in load_hdnet.items():
        if net_name in sub_networks and os.path.exists(weight_path):
            sub_networks[net_name].load_state_dict(torch.load(weight_path))
            logger.info(f"Loaded pretrained weights for {net_name} from {weight_path}")
        else:
            logger.warning(f"Could not load weights for {net_name}: {weight_path} does not exist")

    # Global input and output nodes
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n116"]
    out_nodes = ["n115", "n116"]

    # Node file mapping
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n116", "0005.csv")
    ]

    # Instantiate transformations
    random_rotate = RandomRotate(max_angle=5)
    random_shift = RandomShift(max_shift=5)
    random_zoom = RandomZoom(zoom_range=(0.9, 1.1))
    min_max_normalize = MinMaxNormalize()
    one_hot8 = OneHot(num_classes=8)

    # Node transformation configuration for train and validate
    node_transforms = {
        "train": {
            "n100": [random_rotate, random_shift, random_zoom],
            "n101": [random_rotate, random_shift, random_zoom],
            "n102": [random_rotate, random_shift, random_zoom],
            "n103": [random_rotate, random_shift, random_zoom],
            "n104": [random_rotate, random_shift, random_zoom],
            "n116": [one_hot8]
        },
        "validate": {
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
            "n116": [one_hot8]
        }
    }

    invs = [1-1/(164+48), 1-1/(432+128), 1-1/(684+148), 1-1/(424+52), 1-1/(80+24), 1-1/(580+200), 1-1/(296+48), 1-1/(76+40)]
    invs_sum = sum(invs)

    # Task configuration
    task_configs = {
        "type_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n115", "target_node": "n116", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 2}},
                {"fn": node_lp_loss, "src_node": "n115", "target_node": "n116", "weight": 0.0, "params": {"p": 2.0}}
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n115", "target_node": "n116", "params": {}},
                {"fn": node_precision_metric, "src_node": "n115", "target_node": "n116", "params": {}},
                {"fn": node_f1_metric, "src_node": "n115", "target_node": "n116", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n115", "target_node": "n116", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n115", "target_node": "n116", "params": {}}
            ]
        }
    }

    # Collect case IDs for train and val
    def get_case_ids(data_dir, filename):
        all_files = sorted(os.listdir(data_dir))
        case_ids = []
        for file in all_files:
            if file.startswith('case_') and file.endswith(filename):
                case_id = file.split('_')[1]
                case_ids.append(case_id)
        return sorted(case_ids)

    # Initialize filename to nodes mapping
    filename_to_nodes = {}
    for node, filename in load_node:
        if filename not in filename_to_nodes:
            filename_to_nodes[filename] = []
        filename_to_nodes[filename].append(str(node))

    # Get case IDs for train and val directories
    train_filename_case_ids = {}
    val_filename_case_ids = {}
    for filename in filename_to_nodes:
        train_filename_case_ids[filename] = get_case_ids(train_data_dir, filename)
        val_filename_case_ids[filename] = get_case_ids(val_data_dir, filename)

    # Take union of case IDs
    train_case_ids = sorted(list(set().union(*[set(case_ids) for case_ids in train_filename_case_ids.values()])))
    val_case_ids = sorted(list(set().union(*[set(case_ids) for case_ids in val_filename_case_ids.values()])))

    if not train_case_ids:
        raise ValueError("No case_ids found in train directory!")
    if not val_case_ids:
        raise ValueError("No case_ids found in val directory!")

    # Log missing files for each filename
    for filename, case_ids in train_filename_case_ids.items():
        missing = [cid for cid in train_case_ids if cid not in case_ids]
        if missing:
            logger.warning(f"Missing train files for filename {filename}: {sorted(list(missing))}")
    for filename, case_ids in val_filename_case_ids.items():
        missing = [cid for cid in val_case_ids if cid not in case_ids]
        if missing:
            logger.warning(f"Missing val files for filename {filename}: {sorted(list(missing))}")

    # Generate global random order for training
    train_case_id_order = np.random.permutation(train_case_ids).tolist()
    val_case_id_order = val_case_ids

    # Save data split information
    split_info = {
        "train_case_ids": train_case_ids,
        "val_case_ids": val_case_ids,
        "train_case_id_order": train_case_id_order,
        "val_case_id_order": val_case_id_order,
        "train_count": len(train_case_ids),
        "val_count": len(val_case_ids)
    }
    split_save_path = os.path.join(save_dir, "splits_final.json")
    with open(split_save_path, "w") as f:
        json.dump(split_info, f, indent=4)
    logger.info(f"Data split saved to {split_save_path}")

    # Create datasets
    datasets_train = {}
    datasets_val = {}
    for node, filename in load_node:
        target_shape = None
        for global_node, sub_net_name, sub_node_id in node_mapping:
            if global_node == node:
                target_shape = sub_networks[sub_net_name].node_configs[sub_node_id]
                break
        if target_shape is None:
            raise ValueError(f"Node {node} not found in node_mapping")
        datasets_train[node] = NodeDataset(
            train_data_dir, node, filename, target_shape, node_transforms["train"].get(node, []),
            node_mapping=node_mapping, sub_networks=sub_networks,
            case_ids=train_case_ids, case_id_order=train_case_id_order,
            num_dimensions=num_dimensions
        )
        datasets_val[node] = NodeDataset(
            val_data_dir, node, filename, target_shape, node_transforms["validate"].get(node, []),
            node_mapping=node_mapping, sub_networks=sub_networks,
            case_ids=val_case_ids, case_id_order=val_case_id_order,
            num_dimensions=num_dimensions
        )

    # Validate case_id_order consistency across nodes
    for node in datasets_train:
        if datasets_train[node].case_ids != datasets_train[list(datasets_train.keys())[0]].case_ids:
            raise ValueError(f"Case ID order inconsistent for node {node}")
        if datasets_val[node].case_ids != datasets_val[list(datasets_val.keys())[0]].case_ids:
            raise ValueError(f"Case ID order inconsistent for node {node} in validation")

    # Create DataLoaders
    dataloaders_train = {}
    dataloaders_val = {}
    for node in datasets_train:
        train_indices = list(range(len(datasets_train[node])))
        val_indices = list(range(len(datasets_val[node])))
        dataloaders_train[node] = DataLoader(
            datasets_train[node],
            batch_size=batch_size,
            sampler=OrderedSampler(train_indices, num_workers),
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
        dataloaders_val[node] = DataLoader(
            datasets_val[node],
            batch_size=batch_size,
            sampler=OrderedSampler(val_indices, num_workers),
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

    # Model, optimizer, and scheduler
    onnx_save_path = os.path.join(save_dir, "model_config_initial.onnx")
    model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions, onnx_save_path=onnx_save_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, T_max=num_epochs, eta_min=1e-6)

    # Early stopping and logging
    best_val_loss = float("inf")
    epochs_no_improve = 0
    log = {"epochs": []}

    for epoch in range(num_epochs):
        epoch_seed = seed + epoch
        np.random.seed(epoch_seed)
        batch_seeds = np.random.randint(0, 1000000, size=len(dataloaders_train[node]))
        logger.info(f"Epoch {epoch + 1}: Generated {len(batch_seeds)} batch seeds")

        for batch_idx in range(len(dataloaders_train[node])):
            batch_seed = int(batch_seeds[batch_idx])
            logger.debug(f"Batch {batch_idx}, Seed {batch_seed}")
            for node in datasets_train:
                datasets_train[node].set_batch_seed(batch_seed)
            for node in datasets_val:
                datasets_val[node].set_batch_seed(batch_seed)

        train_loss, train_task_losses, train_metrics = train(
            model, dataloaders_train, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, node_transforms["train"], debug=True
        )

        epoch_log = {"epoch": epoch + 1, "train_loss": train_loss, "train_task_losses": train_task_losses, "train_metrics": train_metrics}

        if (epoch + 1) % validation_interval == 0:
            val_loss, val_task_losses, val_metrics = validate(
                model, dataloaders_val, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, debug=True
            )

            epoch_log.update({"val_loss": val_loss, "val_task_losses": val_task_losses, "metrics": val_metrics})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save HDNet weights
                for net_name, save_path in save_hdnet.items():
                    if net_name in sub_networks:
                        torch.save(sub_networks[net_name].state_dict(), save_path)
                        logger.info(f"Saved {net_name} weights to {save_path}")
            else:
                epochs_no_improve += validation_interval
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            log["epochs"].append(epoch_log)
            log_save_path = os.path.join(save_dir, "training_log.json")
            with open(log_save_path, "w") as f:
                json.dump(log, f, indent=4)
            logger.info(f"Training log updated at {log_save_path}")

        scheduler.step()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
