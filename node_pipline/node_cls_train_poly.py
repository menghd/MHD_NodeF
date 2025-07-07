"""
MHD_Nodet Project - Training Module with Auxiliary Classification Heads
===================================
This module extends the training pipeline for the MHD_Nodet project by adding auxiliary classification heads on encoder outputs for deep supervision, integrating network, dataset, and evaluation components.
- Supports custom data loading from separate train and val directories, and batch-consistent augmentations.
- Includes learning rate scheduling (cosine annealing, polynomial decay, or reduce on plateau) and early stopping for robust training.
- Supports flexible HDNet weight saving and loading via save_hdnet and load_hdnet configurations.
- Implements deep supervision with classification heads at multiple encoder and decoder scales.
- Uses single-node label HDNets to handle unified labels for all classification heads, including auxiliary tasks (calcification, bleeding, ulcer, fibrous cap rupture, lipid core, thickness).

项目：MHD_Nodet - 训练模块（带辅助分类头）
本模块通过在编码器输出上添加辅助分类头扩展了 MHD_Nodet 项目的训练流水线，实现深度监督，集成网络、数据集和评估组件。
- 支持从单独的 train 和 val 目录加载自定义数据，以及批次一致的数据增强。
- 包含学习率调度（余弦退火、多项式衰减或基于平台的学习率衰减）和早停机制以确保稳健训练。
- 支持通过 save_hdnet 和 load_hdnet 配置灵活保存和加载 HDNet 权重。
- 在编码器和解码器的多个尺度添加分类头实现深度监督。
- 使用单节点标签 HDNet 处理所有分类头的统一标签，包括辅助任务（钙化、出血、溃疡、纤维帽破裂、脂质核心、厚度）。

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
from node_toolkit.node_utils import train, validate, CosineAnnealingLR, PolynomialLR, ReduceLROnPlateau
from node_toolkit.node_results import (
    node_focal_loss, node_recall_metric, node_precision_metric, 
    node_f1_metric, node_accuracy_metric, node_specificity_metric
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the training pipeline with deep supervision, including auxiliary classification heads on encoder outputs.
    运行带深度监督和编码器输出辅助分类头的训练流水线的主函数。
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and save paths
    base_data_dir = r"/data/menghaoding/thu_xwh/Tr"
    train_data_dir = os.path.join(base_data_dir, "train")
    val_data_dir = os.path.join(base_data_dir, "val")
    save_dir = r"/data/menghaoding/thu_xwh/MHDNet_poly_multi_1e-4"
    load_dir = r"/data/menghaoding/thu_xwh/MHDNet_poly"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_dimensions = 3
    num_epochs = 400
    learning_rate = 1e-3
    weight_decay = 1e-4
    validation_interval = 1
    patience = 100
    num_workers = 16
    scheduler_type = "poly"  # Options: "cosine", "poly", or "reduce_plateau"

    # Save and load HDNet configurations
    save_hdnet = {
        "encoder": os.path.join(save_dir, "encoder.pth"),
        "decoder": os.path.join(save_dir, "decoder.pth"),
        "classifier_n5": os.path.join(save_dir, "classifier_n5.pth"),
        "classifier_n6": os.path.join(save_dir, "classifier_n6.pth"),
        "classifier_n7": os.path.join(save_dir, "classifier_n7.pth"),
        "classifier_n8": os.path.join(save_dir, "classifier_n8.pth"),
        "classifier_n9": os.path.join(save_dir, "classifier_n9.pth"),
        "classifier_n5_calcification": os.path.join(save_dir, "classifier_n5_calcification.pth"),
        "classifier_n6_calcification": os.path.join(save_dir, "classifier_n6_calcification.pth"),
        "classifier_n7_calcification": os.path.join(save_dir, "classifier_n7_calcification.pth"),
        "classifier_n8_calcification": os.path.join(save_dir, "classifier_n8_calcification.pth"),
        "classifier_n9_calcification": os.path.join(save_dir, "classifier_n9_calcification.pth"),
        "classifier_n5_bleeding": os.path.join(save_dir, "classifier_n5_bleeding.pth"),
        "classifier_n6_bleeding": os.path.join(save_dir, "classifier_n6_bleeding.pth"),
        "classifier_n7_bleeding": os.path.join(save_dir, "classifier_n7_bleeding.pth"),
        "classifier_n8_bleeding": os.path.join(save_dir, "classifier_n8_bleeding.pth"),
        "classifier_n9_bleeding": os.path.join(save_dir, "classifier_n9_bleeding.pth"),
        "classifier_n5_ulcer": os.path.join(save_dir, "classifier_n5_ulcer.pth"),
        "classifier_n6_ulcer": os.path.join(save_dir, "classifier_n6_ulcer.pth"),
        "classifier_n7_ulcer": os.path.join(save_dir, "classifier_n7_ulcer.pth"),
        "classifier_n8_ulcer": os.path.join(save_dir, "classifier_n8_ulcer.pth"),
        "classifier_n9_ulcer": os.path.join(save_dir, "classifier_n9_ulcer.pth"),
        "classifier_n5_fibrous_cap_rupture": os.path.join(save_dir, "classifier_n5_fibrous_cap_rupture.pth"),
        "classifier_n6_fibrous_cap_rupture": os.path.join(save_dir, "classifier_n6_fibrous_cap_rupture.pth"),
        "classifier_n7_fibrous_cap_rupture": os.path.join(save_dir, "classifier_n7_fibrous_cap_rupture.pth"),
        "classifier_n8_fibrous_cap_rupture": os.path.join(save_dir, "classifier_n8_fibrous_cap_rupture.pth"),
        "classifier_n9_fibrous_cap_rupture": os.path.join(save_dir, "classifier_n9_fibrous_cap_rupture.pth"),
        "classifier_n5_lipid_core": os.path.join(save_dir, "classifier_n5_lipid_core.pth"),
        "classifier_n6_lipid_core": os.path.join(save_dir, "classifier_n6_lipid_core.pth"),
        "classifier_n7_lipid_core": os.path.join(save_dir, "classifier_n7_lipid_core.pth"),
        "classifier_n8_lipid_core": os.path.join(save_dir, "classifier_n8_lipid_core.pth"),
        "classifier_n9_lipid_core": os.path.join(save_dir, "classifier_n9_lipid_core.pth"),
        "classifier_n5_thickness": os.path.join(save_dir, "classifier_n5_thickness.pth"),
        "classifier_n6_thickness": os.path.join(save_dir, "classifier_n6_thickness.pth"),
        "classifier_n7_thickness": os.path.join(save_dir, "classifier_n7_thickness.pth"),
        "classifier_n8_thickness": os.path.join(save_dir, "classifier_n8_thickness.pth"),
        "classifier_n9_thickness": os.path.join(save_dir, "classifier_n9_thickness.pth"),
        "label_net": os.path.join(save_dir, "label_net.pth"),
        "label_net_calcification": os.path.join(save_dir, "label_net_calcification.pth"),
        "label_net_bleeding": os.path.join(save_dir, "label_net_bleeding.pth"),
        "label_net_ulcer": os.path.join(save_dir, "label_net_ulcer.pth"),
        "label_net_fibrous_cap_rupture": os.path.join(save_dir, "label_net_fibrous_cap_rupture.pth"),
        "label_net_lipid_core": os.path.join(save_dir, "label_net_lipid_core.pth"),
        "label_net_thickness": os.path.join(save_dir, "label_net_thickness.pth"),
    }
    load_hdnet = {
        "encoder": os.path.join(load_dir, "encoder.pth"),
        "decoder": os.path.join(load_dir, "decoder.pth"),
        "classifier_n5": os.path.join(load_dir, "classifier_n5.pth"),
        "classifier_n6": os.path.join(load_dir, "classifier_n6.pth"),
        "classifier_n7": os.path.join(load_dir, "classifier_n7.pth"),
        "classifier_n8": os.path.join(load_dir, "classifier_n8.pth"),
        "classifier_n9": os.path.join(load_dir, "classifier_n9.pth"),
        "label_net": os.path.join(load_dir, "label_net.pth"),
    }

    # Encoder configuration (downsampling path)
    node_configs_encoder = {
        "n0": (1, 64, 64, 64),  # Input
        "n1": (1, 64, 64, 64),
        "n2": (1, 64, 64, 64),
        "n3": (1, 64, 64, 64),
        "n4": (1, 64, 64, 64),
        "n5": (32, 64, 64, 64),
        "n6": (64, 32, 32, 32),
        "n7": (128, 16, 16, 16),
        "n8": (256, 8, 8, 8),
        "n9": (512, 4, 4, 4),   # Bottleneck
    }
    hyperedge_configs_encoder = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [torch.Size([32, 5, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.1,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e2": {
            "src_nodes": ["n5"],
            "dst_nodes": ["n6"],
            "params": {
                "convs": [torch.Size([64, 32, 3, 3, 3]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "max",
                "dropout": 0.2,
                "reshape": (32, 32, 32),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e3": {
            "src_nodes": ["n6"],
            "dst_nodes": ["n7"],
            "params": {
                "convs": [torch.Size([128, 64, 3, 3, 3]), torch.Size([128, 128, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "max",
                "dropout": 0.3,
                "reshape": (16, 16, 16),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e4": {
            "src_nodes": ["n7"],
            "dst_nodes": ["n8"],
            "params": {
                "convs": [torch.Size([256, 128, 3, 3, 3]), torch.Size([256, 256, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "intp": "max",
                "dropout": 0.4,
                "reshape": (8, 8, 8),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e5": {
            "src_nodes": ["n8"],
            "dst_nodes": ["n9"],
            "params": {
                "convs": [torch.Size([512, 256, 3, 3, 3]), torch.Size([512, 512, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (4, 4, 4),
                "intp": "max",
                "dropout": 0.5,
                "reshape": (4, 4, 4),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }
    in_nodes_encoder = ["n0", "n1", "n2", "n3", "n4"]
    out_nodes_encoder = ["n9", "n8", "n7", "n6", "n5"]  # Bottleneck and skip connections

    # Decoder configuration (upsampling path with skip connections)
    node_configs_decoder = {
        "n0": (512, 4, 4, 4),   # Input: bottleneck from Encoder n9
        "n1": (256, 8, 8, 8),   # Input: skip from Encoder n8
        "n2": (128, 16, 16, 16),  # Input: skip from Encoder n7
        "n3": (64, 32, 32, 32),   # Input: skip from Encoder n6
        "n4": (32, 64, 64, 64),   # Input: skip from Encoder n5
        "n5": (256, 8, 8, 8),   # Decoder features
        "n6": (128, 16, 16, 16),
        "n7": (64, 32, 32, 32),
        "n8": (32, 64, 64, 64),
        "n9": (32, 64, 64, 64),  # Output
    }
    hyperedge_configs_decoder = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [torch.Size([256, 512, 1, 1, 1]), torch.Size([256, 256, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.5,
                "reshape": (8, 8, 8),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e2": {
            "src_nodes": ["n1", "n5"],
            "dst_nodes": ["n6"],
            "params": {
                "convs": [torch.Size([128, 512, 1, 1, 1]), torch.Size([128, 128, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "dropout": 0.4,
                "reshape": (16, 16, 16),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e3": {
            "src_nodes": ["n2", "n6"],
            "dst_nodes": ["n7"],
            "params": {
                "convs": [torch.Size([64, 256, 1, 1, 1]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "linear",
                "dropout": 0.3,
                "reshape": (32, 32, 32),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e4": {
            "src_nodes": ["n3", "n7"],
            "dst_nodes": ["n8"],
            "params": {
                "convs": [torch.Size([32, 128, 1, 1, 1]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.2,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e5": {
            "src_nodes": ["n4", "n8"],
            "dst_nodes": ["n9"],
            "params": {
                "convs": [torch.Size([32, 64, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.1,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }
    in_nodes_decoder = ["n0", "n1", "n2", "n3", "n4"]
    out_nodes_decoder = ["n5", "n6", "n7", "n8", "n9"]  # Output all decoder nodes for deep supervision

    # Classifier configurations for decoder outputs (main task, 4 classes)
    # Classifier for n5 (256 channels, 8x8x8)
    node_configs_classifier_n5 = {
        "n0": (256, 8, 8, 8),  # Input from Decoder n5
        "n1": (4, 1, 1, 1),    # Classification output
    }
    hyperedge_configs_classifier_n5 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5 = ["n0"]
    out_nodes_classifier_n5 = ["n1"]

    # Classifier for n6 (128 channels, 16x16x16)
    node_configs_classifier_n6 = {
        "n0": (128, 16, 16, 16),  # Input from Decoder n6
        "n1": (4, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6 = ["n0"]
    out_nodes_classifier_n6 = ["n1"]

    # Classifier for n7 (64 channels, 32x32x32)
    node_configs_classifier_n7 = {
        "n0": (64, 32, 32, 32),  # Input from Decoder n7
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_n7 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7 = ["n0"]
    out_nodes_classifier_n7 = ["n1"]

    # Classifier for n8 (32 channels, 64x64x64)
    node_configs_classifier_n8 = {
        "n0": (32, 64, 64, 64),  # Input from Decoder n8
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_n8 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8 = ["n0"]
    out_nodes_classifier_n8 = ["n1"]

    # Classifier for n9 (32 channels, 64x64x64)
    node_configs_classifier_n9 = {
        "n0": (32, 64, 64, 64),  # Input from Decoder n9
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_n9 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9 = ["n0"]
    out_nodes_classifier_n9 = ["n1"]

    # Auxiliary classifiers for encoder outputs
    # Calcification (2 classes) for n5 (32 channels, 64x64x64)
    node_configs_classifier_n5_calcification = {
        "n0": (32, 64, 64, 64),  # Input from Encoder n5
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n5_calcification = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5_calcification = ["n0"]
    out_nodes_classifier_n5_calcification = ["n1"]

    # Calcification for n6 (64 channels, 32x32x32)
    node_configs_classifier_n6_calcification = {
        "n0": (64, 32, 32, 32),  # Input from Encoder n6
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6_calcification = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6_calcification = ["n0"]
    out_nodes_classifier_n6_calcification = ["n1"]

    # Calcification for n7 (128 channels, 16x16x16)
    node_configs_classifier_n7_calcification = {
        "n0": (128, 16, 16, 16),  # Input from Encoder n7
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n7_calcification = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7_calcification = ["n0"]
    out_nodes_classifier_n7_calcification = ["n1"]

    # Calcification for n8 (256 channels, 8x8x8)
    node_configs_classifier_n8_calcification = {
        "n0": (256, 8, 8, 8),  # Input from Encoder n8
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n8_calcification = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8_calcification = ["n0"]
    out_nodes_classifier_n8_calcification = ["n1"]

    # Calcification for n9 (512 channels, 4x4x4)
    node_configs_classifier_n9_calcification = {
        "n0": (512, 4, 4, 4),  # Input from Encoder n9
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n9_calcification = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9_calcification = ["n0"]
    out_nodes_classifier_n9_calcification = ["n1"]

    # Bleeding (2 classes) for n5 (32 channels, 64x64x64)
    node_configs_classifier_n5_bleeding = {
        "n0": (32, 64, 64, 64),  # Input from Encoder n5
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n5_bleeding = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5_bleeding = ["n0"]
    out_nodes_classifier_n5_bleeding = ["n1"]

    # Bleeding for n6 (64 channels, 32x32x32)
    node_configs_classifier_n6_bleeding = {
        "n0": (64, 32, 32, 32),  # Input from Encoder n6
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6_bleeding = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6_bleeding = ["n0"]
    out_nodes_classifier_n6_bleeding = ["n1"]

    # Bleeding for n7 (128 channels, 16x16x16)
    node_configs_classifier_n7_bleeding = {
        "n0": (128, 16, 16, 16),  # Input from Encoder n7
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n7_bleeding = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7_bleeding = ["n0"]
    out_nodes_classifier_n7_bleeding = ["n1"]

    # Bleeding for n8 (256 channels, 8x8x8)
    node_configs_classifier_n8_bleeding = {
        "n0": (256, 8, 8, 8),  # Input from Encoder n8
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n8_bleeding = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8_bleeding = ["n0"]
    out_nodes_classifier_n8_bleeding = ["n1"]

    # Bleeding for n9 (512 channels, 4x4x4)
    node_configs_classifier_n9_bleeding = {
        "n0": (512, 4, 4, 4),  # Input from Encoder n9
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n9_bleeding = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9_bleeding = ["n0"]
    out_nodes_classifier_n9_bleeding = ["n1"]

    # Ulcer (2 classes) for n5 (32 channels, 64x64x64)
    node_configs_classifier_n5_ulcer = {
        "n0": (32, 64, 64, 64),  # Input from Encoder n5
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n5_ulcer = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5_ulcer = ["n0"]
    out_nodes_classifier_n5_ulcer = ["n1"]

    # Ulcer for n6 (64 channels, 32x32x32)
    node_configs_classifier_n6_ulcer = {
        "n0": (64, 32, 32, 32),  # Input from Encoder n6
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6_ulcer = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6_ulcer = ["n0"]
    out_nodes_classifier_n6_ulcer = ["n1"]

    # Ulcer for n7 (128 channels, 16x16x16)
    node_configs_classifier_n7_ulcer = {
        "n0": (128, 16, 16, 16),  # Input from Encoder n7
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n7_ulcer = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7_ulcer = ["n0"]
    out_nodes_classifier_n7_ulcer = ["n1"]

    # Ulcer for n8 (256 channels, 8x8x8)
    node_configs_classifier_n8_ulcer = {
        "n0": (256, 8, 8, 8),  # Input from Encoder n8
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n8_ulcer = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8_ulcer = ["n0"]
    out_nodes_classifier_n8_ulcer = ["n1"]

    # Ulcer for n9 (512 channels, 4x4x4)
    node_configs_classifier_n9_ulcer = {
        "n0": (512, 4, 4, 4),  # Input from Encoder n9
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n9_ulcer = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9_ulcer = ["n0"]
    out_nodes_classifier_n9_ulcer = ["n1"]

    # Fibrous Cap Rupture (2 classes) for n5 (32 channels, 64x64x64)
    node_configs_classifier_n5_fibrous_cap_rupture = {
        "n0": (32, 64, 64, 64),  # Input from Encoder n5
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n5_fibrous_cap_rupture = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5_fibrous_cap_rupture = ["n0"]
    out_nodes_classifier_n5_fibrous_cap_rupture = ["n1"]

    # Fibrous Cap Rupture for n6 (64 channels, 32x32x32)
    node_configs_classifier_n6_fibrous_cap_rupture = {
        "n0": (64, 32, 32, 32),  # Input from Encoder n6
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6_fibrous_cap_rupture = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6_fibrous_cap_rupture = ["n0"]
    out_nodes_classifier_n6_fibrous_cap_rupture = ["n1"]

    # Fibrous Cap Rupture for n7 (128 channels, 16x16x16)
    node_configs_classifier_n7_fibrous_cap_rupture = {
        "n0": (128, 16, 16, 16),  # Input from Encoder n7
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n7_fibrous_cap_rupture = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7_fibrous_cap_rupture = ["n0"]
    out_nodes_classifier_n7_fibrous_cap_rupture = ["n1"]

    # Fibrous Cap Rupture for n8 (256 channels, 8x8x8)
    node_configs_classifier_n8_fibrous_cap_rupture = {
        "n0": (256, 8, 8, 8),  # Input from Encoder n8
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n8_fibrous_cap_rupture = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8_fibrous_cap_rupture = ["n0"]
    out_nodes_classifier_n8_fibrous_cap_rupture = ["n1"]

    # Fibrous Cap Rupture for n9 (512 channels, 4x4x4)
    node_configs_classifier_n9_fibrous_cap_rupture = {
        "n0": (512, 4, 4, 4),  # Input from Encoder n9
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n9_fibrous_cap_rupture = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9_fibrous_cap_rupture = ["n0"]
    out_nodes_classifier_n9_fibrous_cap_rupture = ["n1"]

    # Lipid Core (3 classes) for n5 (32 channels, 64x64x64)
    node_configs_classifier_n5_lipid_core = {
        "n0": (32, 64, 64, 64),  # Input from Encoder n5
        "n1": (3, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n5_lipid_core = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([3, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5_lipid_core = ["n0"]
    out_nodes_classifier_n5_lipid_core = ["n1"]

    # Lipid Core for n6 (64 channels, 32x32x32)
    node_configs_classifier_n6_lipid_core = {
        "n0": (64, 32, 32, 32),  # Input from Encoder n6
        "n1": (3, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6_lipid_core = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([3, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6_lipid_core = ["n0"]
    out_nodes_classifier_n6_lipid_core = ["n1"]

    # Lipid Core for n7 (128 channels, 16x16x16)
    node_configs_classifier_n7_lipid_core = {
        "n0": (128, 16, 16, 16),  # Input from Encoder n7
        "n1": (3, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n7_lipid_core = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([3, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7_lipid_core = ["n0"]
    out_nodes_classifier_n7_lipid_core = ["n1"]

    # Lipid Core for n8 (256 channels, 8x8x8)
    node_configs_classifier_n8_lipid_core = {
        "n0": (256, 8, 8, 8),  # Input from Encoder n8
        "n1": (3, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n8_lipid_core = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([3, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8_lipid_core = ["n0"]
    out_nodes_classifier_n8_lipid_core = ["n1"]

    # Lipid Core for n9 (512 channels, 4x4x4)
    node_configs_classifier_n9_lipid_core = {
        "n0": (512, 4, 4, 4),  # Input from Encoder n9
        "n1": (3, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n9_lipid_core = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([3, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9_lipid_core = ["n0"]
    out_nodes_classifier_n9_lipid_core = ["n1"]

    # Thickness (2 classes) for n5 (32 channels, 64x64x64)
    node_configs_classifier_n5_thickness = {
        "n0": (32, 64, 64, 64),  # Input from Encoder n5
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n5_thickness = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n5_thickness = ["n0"]
    out_nodes_classifier_n5_thickness = ["n1"]

    # Thickness for n6 (64 channels, 32x32x32)
    node_configs_classifier_n6_thickness = {
        "n0": (64, 32, 32, 32),  # Input from Encoder n6
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n6_thickness = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n6_thickness = ["n0"]
    out_nodes_classifier_n6_thickness = ["n1"]

    # Thickness for n7 (128 channels, 16x16x16)
    node_configs_classifier_n7_thickness = {
        "n0": (128, 16, 16, 16),  # Input from Encoder n7
        "n1": (2, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_n7_thickness = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n7_thickness = ["n0"]
    out_nodes_classifier_n7_thickness = ["n1"]

    # Thickness for n8 (256 channels, 8x8x8)
    node_configs_classifier_n8_thickness = {
        "n0": (256, 8, 8, 8),  # Input from Encoder n8
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n8_thickness = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n8_thickness = ["n0"]
    out_nodes_classifier_n8_thickness = ["n1"]

    # Thickness for n9 (512 channels, 4x4x4)
    node_configs_classifier_n9_thickness = {
        "n0": (512, 4, 4, 4),  # Input from Encoder n9
        "n1": (2, 1, 1, 1),   # Classification output
    }
    hyperedge_configs_classifier_n9_thickness = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([2, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_classifier_n9_thickness = ["n0"]
    out_nodes_classifier_n9_thickness = ["n1"]

    # Label network configurations for each task
    # Main task (4 classes)
    node_configs_label = {
        "n0": (4, 1, 1, 1),  # Unified one-hot encoded label
    }
    hyperedge_configs_label = {}  # No hyperedges needed
    in_nodes_label = ["n0"]
    out_nodes_label = ["n0"]

    # Calcification (2 classes)
    node_configs_label_calcification = {
        "n0": (2, 1, 1, 1),
    }
    hyperedge_configs_label_calcification = {}
    in_nodes_label_calcification = ["n0"]
    out_nodes_label_calcification = ["n0"]

    # Bleeding (2 classes)
    node_configs_label_bleeding = {
        "n0": (2, 1, 1, 1),
    }
    hyperedge_configs_label_bleeding = {}
    in_nodes_label_bleeding = ["n0"]
    out_nodes_label_bleeding = ["n0"]

    # Ulcer (2 classes)
    node_configs_label_ulcer = {
        "n0": (2, 1, 1, 1),
    }
    hyperedge_configs_label_ulcer = {}
    in_nodes_label_ulcer = ["n0"]
    out_nodes_label_ulcer = ["n0"]

    # Fibrous Cap Rupture (2 classes)
    node_configs_label_fibrous_cap_rupture = {
        "n0": (2, 1, 1, 1),
    }
    hyperedge_configs_label_fibrous_cap_rupture = {}
    in_nodes_label_fibrous_cap_rupture = ["n0"]
    out_nodes_label_fibrous_cap_rupture = ["n0"]

    # Lipid Core (3 classes)
    node_configs_label_lipid_core = {
        "n0": (3, 1, 1, 1),
    }
    hyperedge_configs_label_lipid_core = {}
    in_nodes_label_lipid_core = ["n0"]
    out_nodes_label_lipid_core = ["n0"]

    # Thickness (2 classes)
    node_configs_label_thickness = {
        "n0": (2, 1, 1, 1),
    }
    hyperedge_configs_label_thickness = {}
    in_nodes_label_thickness = ["n0"]
    out_nodes_label_thickness = ["n0"]

    # Global node mapping
    node_mapping = [
        ("n100", "encoder", "n0"),
        ("n101", "encoder", "n1"),
        ("n102", "encoder", "n2"),
        ("n103", "encoder", "n3"),
        ("n104", "encoder", "n4"),
        ("n105", "encoder", "n9"),   # Bottleneck
        ("n105", "decoder", "n0"),   # Connect to Decoder input
        ("n106", "encoder", "n8"),   # Skip connection
        ("n106", "decoder", "n1"),
        ("n107", "encoder", "n7"),   # Skip connection
        ("n107", "decoder", "n2"),
        ("n108", "encoder", "n6"),   # Skip connection
        ("n108", "decoder", "n3"),
        ("n109", "encoder", "n5"),   # Skip connection
        ("n109", "decoder", "n4"),
        ("n110", "decoder", "n5"),   # Decoder output n5
        ("n110", "classifier_n5", "n0"),  # Connect to Classifier_n5 input
        ("n111", "decoder", "n6"),   # Decoder output n6
        ("n111", "classifier_n6", "n0"),  # Connect to Classifier_n6 input
        ("n112", "decoder", "n7"),   # Decoder output n7
        ("n112", "classifier_n7", "n0"),  # Connect to Classifier_n7 input
        ("n113", "decoder", "n8"),   # Decoder output n8
        ("n113", "classifier_n8", "n0"),  # Connect to Classifier_n8 input
        ("n114", "decoder", "n9"),   # Decoder output n9
        ("n114", "classifier_n9", "n0"),  # Connect to Classifier_n9 input
        ("n115", "classifier_n5", "n1"),  # Classification output n5 (main task)
        ("n116", "classifier_n6", "n1"),  # Classification output n6 (main task)
        ("n117", "classifier_n7", "n1"),  # Classification output n7 (main task)
        ("n118", "classifier_n8", "n1"),  # Classification output n8 (main task)
        ("n119", "classifier_n9", "n1"),  # Classification output n9 (main task)
        ("n120", "label_net", "n0"),      # Unified label node (main task)
        # Encoder auxiliary classifiers
        ("n109", "classifier_n5_calcification", "n0"),
        ("n108", "classifier_n6_calcification", "n0"),
        ("n107", "classifier_n7_calcification", "n0"),
        ("n106", "classifier_n8_calcification", "n0"),
        ("n105", "classifier_n9_calcification", "n0"),
        ("n121", "classifier_n5_calcification", "n1"),
        ("n122", "classifier_n6_calcification", "n1"),
        ("n123", "classifier_n7_calcification", "n1"),
        ("n124", "classifier_n8_calcification", "n1"),
        ("n125", "classifier_n9_calcification", "n1"),
        ("n126", "label_net_calcification", "n0"),
        ("n109", "classifier_n5_bleeding", "n0"),
        ("n108", "classifier_n6_bleeding", "n0"),
        ("n107", "classifier_n7_bleeding", "n0"),
        ("n106", "classifier_n8_bleeding", "n0"),
        ("n105", "classifier_n9_bleeding", "n0"),
        ("n127", "classifier_n5_bleeding", "n1"),
        ("n128", "classifier_n6_bleeding", "n1"),
        ("n129", "classifier_n7_bleeding", "n1"),
        ("n130", "classifier_n8_bleeding", "n1"),
        ("n131", "classifier_n9_bleeding", "n1"),
        ("n132", "label_net_bleeding", "n0"),
        ("n109", "classifier_n5_ulcer", "n0"),
        ("n108", "classifier_n6_ulcer", "n0"),
        ("n107", "classifier_n7_ulcer", "n0"),
        ("n106", "classifier_n8_ulcer", "n0"),
        ("n105", "classifier_n9_ulcer", "n0"),
        ("n133", "classifier_n5_ulcer", "n1"),
        ("n134", "classifier_n6_ulcer", "n1"),
        ("n135", "classifier_n7_ulcer", "n1"),
        ("n136", "classifier_n8_ulcer", "n1"),
        ("n137", "classifier_n9_ulcer", "n1"),
        ("n138", "label_net_ulcer", "n0"),
        ("n109", "classifier_n5_fibrous_cap_rupture", "n0"),
        ("n108", "classifier_n6_fibrous_cap_rupture", "n0"),
        ("n107", "classifier_n7_fibrous_cap_rupture", "n0"),
        ("n106", "classifier_n8_fibrous_cap_rupture", "n0"),
        ("n105", "classifier_n9_fibrous_cap_rupture", "n0"),
        ("n139", "classifier_n5_fibrous_cap_rupture", "n1"),
        ("n140", "classifier_n6_fibrous_cap_rupture", "n1"),
        ("n141", "classifier_n7_fibrous_cap_rupture", "n1"),
        ("n142", "classifier_n8_fibrous_cap_rupture", "n1"),
        ("n143", "classifier_n9_fibrous_cap_rupture", "n1"),
        ("n144", "label_net_fibrous_cap_rupture", "n0"),
        ("n109", "classifier_n5_lipid_core", "n0"),
        ("n108", "classifier_n6_lipid_core", "n0"),
        ("n107", "classifier_n7_lipid_core", "n0"),
        ("n106", "classifier_n8_lipid_core", "n0"),
        ("n105", "classifier_n9_lipid_core", "n0"),
        ("n145", "classifier_n5_lipid_core", "n1"),
        ("n146", "classifier_n6_lipid_core", "n1"),
        ("n147", "classifier_n7_lipid_core", "n1"),
        ("n148", "classifier_n8_lipid_core", "n1"),
        ("n149", "classifier_n9_lipid_core", "n1"),
        ("n150", "label_net_lipid_core", "n0"),
        ("n109", "classifier_n5_thickness", "n0"),
        ("n108", "classifier_n6_thickness", "n0"),
        ("n107", "classifier_n7_thickness", "n0"),
        ("n106", "classifier_n8_thickness", "n0"),
        ("n105", "classifier_n9_thickness", "n0"),
        ("n151", "classifier_n5_thickness", "n1"),
        ("n152", "classifier_n6_thickness", "n1"),
        ("n153", "classifier_n7_thickness", "n1"),
        ("n154", "classifier_n8_thickness", "n1"),
        ("n155", "classifier_n9_thickness", "n1"),
        ("n156", "label_net_thickness", "n0"),
    ]

    # Instantiate subnetworks
    sub_networks_configs = {
        "encoder": (node_configs_encoder, hyperedge_configs_encoder, in_nodes_encoder, out_nodes_encoder),
        "decoder": (node_configs_decoder, hyperedge_configs_decoder, in_nodes_decoder, out_nodes_decoder),
        "classifier_n5": (node_configs_classifier_n5, hyperedge_configs_classifier_n5, in_nodes_classifier_n5, out_nodes_classifier_n5),
        "classifier_n6": (node_configs_classifier_n6, hyperedge_configs_classifier_n6, in_nodes_classifier_n6, out_nodes_classifier_n6),
        "classifier_n7": (node_configs_classifier_n7, hyperedge_configs_classifier_n7, in_nodes_classifier_n7, out_nodes_classifier_n7),
        "classifier_n8": (node_configs_classifier_n8, hyperedge_configs_classifier_n8, in_nodes_classifier_n8, out_nodes_classifier_n8),
        "classifier_n9": (node_configs_classifier_n9, hyperedge_configs_classifier_n9, in_nodes_classifier_n9, out_nodes_classifier_n9),
        "classifier_n5_calcification": (node_configs_classifier_n5_calcification, hyperedge_configs_classifier_n5_calcification, in_nodes_classifier_n5_calcification, out_nodes_classifier_n5_calcification),
        "classifier_n6_calcification": (node_configs_classifier_n6_calcification, hyperedge_configs_classifier_n6_calcification, in_nodes_classifier_n6_calcification, out_nodes_classifier_n6_calcification),
        "classifier_n7_calcification": (node_configs_classifier_n7_calcification, hyperedge_configs_classifier_n7_calcification, in_nodes_classifier_n7_calcification, out_nodes_classifier_n7_calcification),
        "classifier_n8_calcification": (node_configs_classifier_n8_calcification, hyperedge_configs_classifier_n8_calcification, in_nodes_classifier_n8_calcification, out_nodes_classifier_n8_calcification),
        "classifier_n9_calcification": (node_configs_classifier_n9_calcification, hyperedge_configs_classifier_n9_calcification, in_nodes_classifier_n9_calcification, out_nodes_classifier_n9_calcification),
        "classifier_n5_bleeding": (node_configs_classifier_n5_bleeding, hyperedge_configs_classifier_n5_bleeding, in_nodes_classifier_n5_bleeding, out_nodes_classifier_n5_bleeding),
        "classifier_n6_bleeding": (node_configs_classifier_n6_bleeding, hyperedge_configs_classifier_n6_bleeding, in_nodes_classifier_n6_bleeding, out_nodes_classifier_n6_bleeding),
        "classifier_n7_bleeding": (node_configs_classifier_n7_bleeding, hyperedge_configs_classifier_n7_bleeding, in_nodes_classifier_n7_bleeding, out_nodes_classifier_n7_bleeding),
        "classifier_n8_bleeding": (node_configs_classifier_n8_bleeding, hyperedge_configs_classifier_n8_bleeding, in_nodes_classifier_n8_bleeding, out_nodes_classifier_n8_bleeding),
        "classifier_n9_bleeding": (node_configs_classifier_n9_bleeding, hyperedge_configs_classifier_n9_bleeding, in_nodes_classifier_n9_bleeding, out_nodes_classifier_n9_bleeding),
        "classifier_n5_ulcer": (node_configs_classifier_n5_ulcer, hyperedge_configs_classifier_n5_ulcer, in_nodes_classifier_n5_ulcer, out_nodes_classifier_n5_ulcer),
        "classifier_n6_ulcer": (node_configs_classifier_n6_ulcer, hyperedge_configs_classifier_n6_ulcer, in_nodes_classifier_n6_ulcer, out_nodes_classifier_n6_ulcer),
        "classifier_n7_ulcer": (node_configs_classifier_n7_ulcer, hyperedge_configs_classifier_n7_ulcer, in_nodes_classifier_n7_ulcer, out_nodes_classifier_n7_ulcer),
        "classifier_n8_ulcer": (node_configs_classifier_n8_ulcer, hyperedge_configs_classifier_n8_ulcer, in_nodes_classifier_n8_ulcer, out_nodes_classifier_n8_ulcer),
        "classifier_n9_ulcer": (node_configs_classifier_n9_ulcer, hyperedge_configs_classifier_n9_ulcer, in_nodes_classifier_n9_ulcer, out_nodes_classifier_n9_ulcer),
        "classifier_n5_fibrous_cap_rupture": (node_configs_classifier_n5_fibrous_cap_rupture, hyperedge_configs_classifier_n5_fibrous_cap_rupture, in_nodes_classifier_n5_fibrous_cap_rupture, out_nodes_classifier_n5_fibrous_cap_rupture),
        "classifier_n6_fibrous_cap_rupture": (node_configs_classifier_n6_fibrous_cap_rupture, hyperedge_configs_classifier_n6_fibrous_cap_rupture, in_nodes_classifier_n6_fibrous_cap_rupture, out_nodes_classifier_n6_fibrous_cap_rupture),
        "classifier_n7_fibrous_cap_rupture": (node_configs_classifier_n7_fibrous_cap_rupture, hyperedge_configs_classifier_n7_fibrous_cap_rupture, in_nodes_classifier_n7_fibrous_cap_rupture, out_nodes_classifier_n7_fibrous_cap_rupture),
        "classifier_n8_fibrous_cap_rupture": (node_configs_classifier_n8_fibrous_cap_rupture, hyperedge_configs_classifier_n8_fibrous_cap_rupture, in_nodes_classifier_n8_fibrous_cap_rupture, out_nodes_classifier_n8_fibrous_cap_rupture),
        "classifier_n9_fibrous_cap_rupture": (node_configs_classifier_n9_fibrous_cap_rupture, hyperedge_configs_classifier_n9_fibrous_cap_rupture, in_nodes_classifier_n9_fibrous_cap_rupture, out_nodes_classifier_n9_fibrous_cap_rupture),
        "classifier_n5_lipid_core": (node_configs_classifier_n5_lipid_core, hyperedge_configs_classifier_n5_lipid_core, in_nodes_classifier_n5_lipid_core, out_nodes_classifier_n5_lipid_core),
        "classifier_n6_lipid_core": (node_configs_classifier_n6_lipid_core, hyperedge_configs_classifier_n6_lipid_core, in_nodes_classifier_n6_lipid_core, out_nodes_classifier_n6_lipid_core),
        "classifier_n7_lipid_core": (node_configs_classifier_n7_lipid_core, hyperedge_configs_classifier_n7_lipid_core, in_nodes_classifier_n7_lipid_core, out_nodes_classifier_n7_lipid_core),
        "classifier_n8_lipid_core": (node_configs_classifier_n8_lipid_core, hyperedge_configs_classifier_n8_lipid_core, in_nodes_classifier_n8_lipid_core, out_nodes_classifier_n8_lipid_core),
        "classifier_n9_lipid_core": (node_configs_classifier_n9_lipid_core, hyperedge_configs_classifier_n9_lipid_core, in_nodes_classifier_n9_lipid_core, out_nodes_classifier_n9_lipid_core),
        "classifier_n5_thickness": (node_configs_classifier_n5_thickness, hyperedge_configs_classifier_n5_thickness, in_nodes_classifier_n5_thickness, out_nodes_classifier_n5_thickness),
        "classifier_n6_thickness": (node_configs_classifier_n6_thickness, hyperedge_configs_classifier_n6_thickness, in_nodes_classifier_n6_thickness, out_nodes_classifier_n6_thickness),
        "classifier_n7_thickness": (node_configs_classifier_n7_thickness, hyperedge_configs_classifier_n7_thickness, in_nodes_classifier_n7_thickness, out_nodes_classifier_n7_thickness),
        "classifier_n8_thickness": (node_configs_classifier_n8_thickness, hyperedge_configs_classifier_n8_thickness, in_nodes_classifier_n8_thickness, out_nodes_classifier_n8_thickness),
        "classifier_n9_thickness": (node_configs_classifier_n9_thickness, hyperedge_configs_classifier_n9_thickness, in_nodes_classifier_n9_thickness, out_nodes_classifier_n9_thickness),
        "label_net": (node_configs_label, hyperedge_configs_label, in_nodes_label, out_nodes_label),
        "label_net_calcification": (node_configs_label_calcification, hyperedge_configs_label_calcification, in_nodes_label_calcification, out_nodes_label_calcification),
        "label_net_bleeding": (node_configs_label_bleeding, hyperedge_configs_label_bleeding, in_nodes_label_bleeding, out_nodes_label_bleeding),
        "label_net_ulcer": (node_configs_label_ulcer, hyperedge_configs_label_ulcer, in_nodes_label_ulcer, out_nodes_label_ulcer),
        "label_net_fibrous_cap_rupture": (node_configs_label_fibrous_cap_rupture, hyperedge_configs_label_fibrous_cap_rupture, in_nodes_label_fibrous_cap_rupture, out_nodes_label_fibrous_cap_rupture),
        "label_net_lipid_core": (node_configs_label_lipid_core, hyperedge_configs_label_lipid_core, in_nodes_label_lipid_core, out_nodes_label_lipid_core),
        "label_net_thickness": (node_configs_label_thickness, hyperedge_configs_label_thickness, in_nodes_label_thickness, out_nodes_label_thickness),
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
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n120", "n126", "n132", "n138", "n144", "n150", "n156"]
    out_nodes = ["n115", "n116", "n117", "n118", "n119",  # Main task outputs
                 "n121", "n122", "n123", "n124", "n125",  # Calcification outputs
                 "n127", "n128", "n129", "n130", "n131",  # Bleeding outputs
                 "n133", "n134", "n135", "n136", "n137",  # Ulcer outputs
                 "n139", "n140", "n141", "n142", "n143",  # Fibrous Cap Rupture outputs
                 "n145", "n146", "n147", "n148", "n149",  # Lipid Core outputs
                 "n151", "n152", "n153", "n154", "n155",  # Thickness outputs
                 "n120", "n126", "n132", "n138", "n144", "n150", "n156"]  # Label nodes

    # Node file mapping
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n120", "0006.csv"),  # Main task (4 classes)
        ("n126", "0008.csv"),  # Calcification (2 classes)
        ("n132", "0009.csv"),  # Bleeding (2 classes)
        ("n138", "0010.csv"),  # Ulcer (2 classes)
        ("n144", "0011.csv"),  # Fibrous Cap Rupture (2 classes)
        ("n150", "0012.csv"),  # Lipid Core (3 classes)
        ("n156", "0013.csv"),  # Thickness (2 classes)
    ]

    # Instantiate transformations
    random_rotate = RandomRotate(max_angle=15)
    random_shift = RandomShift(max_shift=10)
    random_zoom = RandomZoom(zoom_range=(0.9, 1.1))
    min_max_normalize = MinMaxNormalize()
    one_hot4 = OneHot(num_classes=4)
    one_hot2 = OneHot(num_classes=2)
    one_hot3 = OneHot(num_classes=3)

    # Node transformation configuration for train and validate
    node_transforms = {
        "train": {
            "n100": [random_rotate, random_shift, random_zoom],
            "n101": [random_rotate, random_shift, random_zoom],
            "n102": [random_rotate, random_shift, random_zoom],
            "n103": [random_rotate, random_shift, random_zoom],
            "n104": [random_rotate, random_shift, random_zoom],
            "n120": [one_hot4],
            "n126": [one_hot2],
            "n132": [one_hot2],
            "n138": [one_hot2],
            "n144": [one_hot2],
            "n150": [one_hot3],
            "n156": [one_hot2],
        },
        "validate": {
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
            "n120": [one_hot4],
            "n126": [one_hot2],
            "n132": [one_hot2],
            "n138": [one_hot2],
            "n144": [one_hot2],
            "n150": [one_hot3],
            "n156": [one_hot2],
        }
    }

    # Class weights
    invs = [1/(260+100), 1/(703+140), 1/(1596+388), 1/(1521+388)]
    invs_sum = sum(invs)
    class_weights = {
        "main": [x / invs_sum for x in invs],
        "calcification": [x / sum([1/(1660+396), 1/(2420+620)]) for x in [1/(1660+396), 1/(2420+620)]],
        "bleeding": [x / sum([1/(2892+692), 1/(1188+324)]) for x in [1/(2892+692), 1/(1188+324)]],
        "ulcer": [x / sum([1/(3765+944), 1/(315+72)]) for x in [1/(3765+944), 1/(315+72)]],
        "fibrous_cap_rupture": [x / sum([1/(3699+912), 1/(381+104)]) for x in [1/(3699+912), 1/(381+104)]],
        "lipid_core": [x / sum([1/(968+256), 1/(1664+368), 1/(1448+392)]) for x in [1/(968+256), 1/(1664+368), 1/(1448+392)]],
        "thickness": [x / sum([1/(1050+240), 1/(3030+776)]) for x in [1/(1050+240), 1/(3030+776)]],
    }

    # Task configuration with deep supervision
    task_configs = {
        "type_cls_n5": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n115", "target_node": "n120", "weight": 1.0, "params": {"alpha": class_weights["main"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n115", "target_node": "n120", "params": {}},
                {"fn": node_precision_metric, "src_node": "n115", "target_node": "n120", "params": {}},
                {"fn": node_f1_metric, "src_node": "n115", "target_node": "n120", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n115", "target_node": "n120", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n115", "target_node": "n120", "params": {}}
            ]
        },
        "type_cls_n6": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n116", "target_node": "n120", "weight": 0.5, "params": {"alpha": class_weights["main"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n116", "target_node": "n120", "params": {}},
                {"fn": node_precision_metric, "src_node": "n116", "target_node": "n120", "params": {}},
                {"fn": node_f1_metric, "src_node": "n116", "target_node": "n120", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n116", "target_node": "n120", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n116", "target_node": "n120", "params": {}}
            ]
        },
        "type_cls_n7": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n117", "target_node": "n120", "weight": 0.25, "params": {"alpha": class_weights["main"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n117", "target_node": "n120", "params": {}},
                {"fn": node_precision_metric, "src_node": "n117", "target_node": "n120", "params": {}},
                {"fn": node_f1_metric, "src_node": "n117", "target_node": "n120", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n117", "target_node": "n120", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n117", "target_node": "n120", "params": {}}
            ]
        },
        "type_cls_n8": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n118", "target_node": "n120", "weight": 0.125, "params": {"alpha": class_weights["main"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n118", "target_node": "n120", "params": {}},
                {"fn": node_precision_metric, "src_node": "n118", "target_node": "n120", "params": {}},
                {"fn": node_f1_metric, "src_node": "n118", "target_node": "n120", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n118", "target_node": "n120", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n118", "target_node": "n120", "params": {}}
            ]
        },
        "type_cls_n9": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n119", "target_node": "n120", "weight": 0.0625, "params": {"alpha": class_weights["main"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n119", "target_node": "n120", "params": {}},
                {"fn": node_precision_metric, "src_node": "n119", "target_node": "n120", "params": {}},
                {"fn": node_f1_metric, "src_node": "n119", "target_node": "n120", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n119", "target_node": "n120", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n119", "target_node": "n120", "params": {}}
            ]
        },
        # Auxiliary tasks for encoder outputs
        "type_cls_n5_calcification": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n121", "target_node": "n126", "weight": 0.5, "params": {"alpha": class_weights["calcification"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n121", "target_node": "n126", "params": {}},
                {"fn": node_precision_metric, "src_node": "n121", "target_node": "n126", "params": {}},
                {"fn": node_f1_metric, "src_node": "n121", "target_node": "n126", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n121", "target_node": "n126", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n121", "target_node": "n126", "params": {}}
            ]
        },
        "type_cls_n6_calcification": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n122", "target_node": "n126", "weight": 0.25, "params": {"alpha": class_weights["calcification"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n122", "target_node": "n126", "params": {}},
                {"fn": node_precision_metric, "src_node": "n122", "target_node": "n126", "params": {}},
                {"fn": node_f1_metric, "src_node": "n122", "target_node": "n126", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n122", "target_node": "n126", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n122", "target_node": "n126", "params": {}}
            ]
        },
        "type_cls_n7_calcification": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n123", "target_node": "n126", "weight": 0.125, "params": {"alpha": class_weights["calcification"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n123", "target_node": "n126", "params": {}},
                {"fn": node_precision_metric, "src_node": "n123", "target_node": "n126", "params": {}},
                {"fn": node_f1_metric, "src_node": "n123", "target_node": "n126", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n123", "target_node": "n126", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n123", "target_node": "n126", "params": {}}
            ]
        },
        "type_cls_n8_calcification": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n124", "target_node": "n126", "weight": 0.0625, "params": {"alpha": class_weights["calcification"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n124", "target_node": "n126", "params": {}},
                {"fn": node_precision_metric, "src_node": "n124", "target_node": "n126", "params": {}},
                {"fn": node_f1_metric, "src_node": "n124", "target_node": "n126", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n124", "target_node": "n126", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n124", "target_node": "n126", "params": {}}
            ]
        },
        "type_cls_n9_calcification": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n125", "target_node": "n126", "weight": 0.03125, "params": {"alpha": class_weights["calcification"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n125", "target_node": "n126", "params": {}},
                {"fn": node_precision_metric, "src_node": "n125", "target_node": "n126", "params": {}},
                {"fn": node_f1_metric, "src_node": "n125", "target_node": "n126", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n125", "target_node": "n126", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n125", "target_node": "n126", "params": {}}
            ]
        },
        "type_cls_n5_bleeding": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n127", "target_node": "n132", "weight": 0.5, "params": {"alpha": class_weights["bleeding"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n127", "target_node": "n132", "params": {}},
                {"fn": node_precision_metric, "src_node": "n127", "target_node": "n132", "params": {}},
                {"fn": node_f1_metric, "src_node": "n127", "target_node": "n132", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n127", "target_node": "n132", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n127", "target_node": "n132", "params": {}}
            ]
        },
        "type_cls_n6_bleeding": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n128", "target_node": "n132", "weight": 0.25, "params": {"alpha": class_weights["bleeding"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n128", "target_node": "n132", "params": {}},
                {"fn": node_precision_metric, "src_node": "n128", "target_node": "n132", "params": {}},
                {"fn": node_f1_metric, "src_node": "n128", "target_node": "n132", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n128", "target_node": "n132", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n128", "target_node": "n132", "params": {}}
            ]
        },
        "type_cls_n7_bleeding": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n129", "target_node": "n132", "weight": 0.125, "params": {"alpha": class_weights["bleeding"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n129", "target_node": "n132", "params": {}},
                {"fn": node_precision_metric, "src_node": "n129", "target_node": "n132", "params": {}},
                {"fn": node_f1_metric, "src_node": "n129", "target_node": "n132", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n129", "target_node": "n132", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n129", "target_node": "n132", "params": {}}
            ]
        },
        "type_cls_n8_bleeding": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n130", "target_node": "n132", "weight": 0.0625, "params": {"alpha": class_weights["bleeding"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n130", "target_node": "n132", "params": {}},
                {"fn": node_precision_metric, "src_node": "n130", "target_node": "n132", "params": {}},
                {"fn": node_f1_metric, "src_node": "n130", "target_node": "n132", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n130", "target_node": "n132", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n130", "target_node": "n132", "params": {}}
            ]
        },
        "type_cls_n9_bleeding": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n131", "target_node": "n132", "weight": 0.03125, "params": {"alpha": class_weights["bleeding"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n131", "target_node": "n132", "params": {}},
                {"fn": node_precision_metric, "src_node": "n131", "target_node": "n132", "params": {}},
                {"fn": node_f1_metric, "src_node": "n131", "target_node": "n132", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n131", "target_node": "n132", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n131", "target_node": "n132", "params": {}}
            ]
        },
        "type_cls_n5_ulcer": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n133", "target_node": "n138", "weight": 0.5, "params": {"alpha": class_weights["ulcer"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n133", "target_node": "n138", "params": {}},
                {"fn": node_precision_metric, "src_node": "n133", "target_node": "n138", "params": {}},
                {"fn": node_f1_metric, "src_node": "n133", "target_node": "n138", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n133", "target_node": "n138", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n133", "target_node": "n138", "params": {}}
            ]
        },
        "type_cls_n6_ulcer": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n134", "target_node": "n138", "weight": 0.25, "params": {"alpha": class_weights["ulcer"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n134", "target_node": "n138", "params": {}},
                {"fn": node_precision_metric, "src_node": "n134", "target_node": "n138", "params": {}},
                {"fn": node_f1_metric, "src_node": "n134", "target_node": "n138", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n134", "target_node": "n138", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n134", "target_node": "n138", "params": {}}
            ]
        },
        "type_cls_n7_ulcer": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n135", "target_node": "n138", "weight": 0.125, "params": {"alpha": class_weights["ulcer"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n135", "target_node": "n138", "params": {}},
                {"fn": node_precision_metric, "src_node": "n135", "target_node": "n138", "params": {}},
                {"fn": node_f1_metric, "src_node": "n135", "target_node": "n138", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n135", "target_node": "n138", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n135", "target_node": "n138", "params": {}}
            ]
        },
        "type_cls_n8_ulcer": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n136", "target_node": "n138", "weight": 0.0625, "params": {"alpha": class_weights["ulcer"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n136", "target_node": "n138", "params": {}},
                {"fn": node_precision_metric, "src_node": "n136", "target_node": "n138", "params": {}},
                {"fn": node_f1_metric, "src_node": "n136", "target_node": "n138", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n136", "target_node": "n138", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n136", "target_node": "n138", "params": {}}
            ]
        },
        "type_cls_n9_ulcer": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n137", "target_node": "n138", "weight": 0.03125, "params": {"alpha": class_weights["ulcer"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n137", "target_node": "n138", "params": {}},
                {"fn": node_precision_metric, "src_node": "n137", "target_node": "n138", "params": {}},
                {"fn": node_f1_metric, "src_node": "n137", "target_node": "n138", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n137", "target_node": "n138", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n137", "target_node": "n138", "params": {}}
            ]
        },
        "type_cls_n5_fibrous_cap_rupture": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n139", "target_node": "n144", "weight": 0.5, "params": {"alpha": class_weights["fibrous_cap_rupture"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n139", "target_node": "n144", "params": {}},
                {"fn": node_precision_metric, "src_node": "n139", "target_node": "n144", "params": {}},
                {"fn": node_f1_metric, "src_node": "n139", "target_node": "n144", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n139", "target_node": "n144", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n139", "target_node": "n144", "params": {}}
            ]
        },
        "type_cls_n6_fibrous_cap_rupture": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n140", "target_node": "n144", "weight": 0.25, "params": {"alpha": class_weights["fibrous_cap_rupture"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n140", "target_node": "n144", "params": {}},
                {"fn": node_precision_metric, "src_node": "n140", "target_node": "n144", "params": {}},
                {"fn": node_f1_metric, "src_node": "n140", "target_node": "n144", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n140", "target_node": "n144", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n140", "target_node": "n144", "params": {}}
            ]
        },
        "type_cls_n7_fibrous_cap_rupture": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n141", "target_node": "n144", "weight": 0.125, "params": {"alpha": class_weights["fibrous_cap_rupture"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n141", "target_node": "n144", "params": {}},
                {"fn": node_precision_metric, "src_node": "n141", "target_node": "n144", "params": {}},
                {"fn": node_f1_metric, "src_node": "n141", "target_node": "n144", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n141", "target_node": "n144", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n141", "target_node": "n144", "params": {}}
            ]
        },
        "type_cls_n8_fibrous_cap_rupture": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n142", "target_node": "n144", "weight": 0.0625, "params": {"alpha": class_weights["fibrous_cap_rupture"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n142", "target_node": "n144", "params": {}},
                {"fn": node_precision_metric, "src_node": "n142", "target_node": "n144", "params": {}},
                {"fn": node_f1_metric, "src_node": "n142", "target_node": "n144", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n142", "target_node": "n144", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n142", "target_node": "n144", "params": {}}
            ]
        },
        "type_cls_n9_fibrous_cap_rupture": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n143", "target_node": "n144", "weight": 0.03125, "params": {"alpha": class_weights["fibrous_cap_rupture"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n143", "target_node": "n144", "params": {}},
                {"fn": node_precision_metric, "src_node": "n143", "target_node": "n144", "params": {}},
                {"fn": node_f1_metric, "src_node": "n143", "target_node": "n144", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n143", "target_node": "n144", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n143", "target_node": "n144", "params": {}}
            ]
        },
        "type_cls_n5_lipid_core": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n145", "target_node": "n150", "weight": 0.5, "params": {"alpha": class_weights["lipid_core"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n145", "target_node": "n150", "params": {}},
                {"fn": node_precision_metric, "src_node": "n145", "target_node": "n150", "params": {}},
                {"fn": node_f1_metric, "src_node": "n145", "target_node": "n150", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n145", "target_node": "n150", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n145", "target_node": "n150", "params": {}}
            ]
        },
        "type_cls_n6_lipid_core": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n146", "target_node": "n150", "weight": 0.25, "params": {"alpha": class_weights["lipid_core"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n146", "target_node": "n150", "params": {}},
                {"fn": node_precision_metric, "src_node": "n146", "target_node": "n150", "params": {}},
                {"fn": node_f1_metric, "src_node": "n146", "target_node": "n150", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n146", "target_node": "n150", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n146", "target_node": "n150", "params": {}}
            ]
        },
        "type_cls_n7_lipid_core": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n147", "target_node": "n150", "weight": 0.125, "params": {"alpha": class_weights["lipid_core"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n147", "target_node": "n150", "params": {}},
                {"fn": node_precision_metric, "src_node": "n147", "target_node": "n150", "params": {}},
                {"fn": node_f1_metric, "src_node": "n147", "target_node": "n150", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n147", "target_node": "n150", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n147", "target_node": "n150", "params": {}}
            ]
        },
        "type_cls_n8_lipid_core": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n148", "target_node": "n150", "weight": 0.0625, "params": {"alpha": class_weights["lipid_core"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n148", "target_node": "n150", "params": {}},
                {"fn": node_precision_metric, "src_node": "n148", "target_node": "n150", "params": {}},
                {"fn": node_f1_metric, "src_node": "n148", "target_node": "n150", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n148", "target_node": "n150", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n148", "target_node": "n150", "params": {}}
            ]
        },
        "type_cls_n9_lipid_core": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n149", "target_node": "n150", "weight": 0.03125, "params": {"alpha": class_weights["lipid_core"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n149", "target_node": "n150", "params": {}},
                {"fn": node_precision_metric, "src_node": "n149", "target_node": "n150", "params": {}},
                {"fn": node_f1_metric, "src_node": "n149", "target_node": "n150", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n149", "target_node": "n150", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n149", "target_node": "n150", "params": {}}
            ]
        },
        "type_cls_n5_thickness": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n151", "target_node": "n156", "weight": 0.5, "params": {"alpha": class_weights["thickness"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n151", "target_node": "n156", "params": {}},
                {"fn": node_precision_metric, "src_node": "n151", "target_node": "n156", "params": {}},
                {"fn": node_f1_metric, "src_node": "n151", "target_node": "n156", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n151", "target_node": "n156", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n151", "target_node": "n156", "params": {}}
            ]
        },
        "type_cls_n6_thickness": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n152", "target_node": "n156", "weight": 0.25, "params": {"alpha": class_weights["thickness"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n152", "target_node": "n156", "params": {}},
                {"fn": node_precision_metric, "src_node": "n152", "target_node": "n156", "params": {}},
                {"fn": node_f1_metric, "src_node": "n152", "target_node": "n156", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n152", "target_node": "n156", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n152", "target_node": "n156", "params": {}}
            ]
        },
        "type_cls_n7_thickness": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n153", "target_node": "n156", "weight": 0.125, "params": {"alpha": class_weights["thickness"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n153", "target_node": "n156", "params": {}},
                {"fn": node_precision_metric, "src_node": "n153", "target_node": "n156", "params": {}},
                {"fn": node_f1_metric, "src_node": "n153", "target_node": "n156", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n153", "target_node": "n156", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n153", "target_node": "n156", "params": {}}
            ]
        },
        "type_cls_n8_thickness": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n154", "target_node": "n156", "weight": 0.0625, "params": {"alpha": class_weights["thickness"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n154", "target_node": "n156", "params": {}},
                {"fn": node_precision_metric, "src_node": "n154", "target_node": "n156", "params": {}},
                {"fn": node_f1_metric, "src_node": "n154", "target_node": "n156", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n154", "target_node": "n156", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n154", "target_node": "n156", "params": {}}
            ]
        },
        "type_cls_n9_thickness": {
            "loss": [
                {"fn": node_focal_loss, "src_node": "n155", "target_node": "n156", "weight": 0.03125, "params": {"alpha": class_weights["thickness"], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": "n155", "target_node": "n156", "params": {}},
                {"fn": node_precision_metric, "src_node": "n155", "target_node": "n156", "params": {}},
                {"fn": node_f1_metric, "src_node": "n155", "target_node": "n156", "params": {}},
                {"fn": node_accuracy_metric, "src_node": "n155", "target_node": "n156", "params": {}},
                {"fn": node_specificity_metric, "src_node": "n155", "target_node": "n156", "params": {}}
            ]
        },
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
    split_save_path = os.path.join(save_dir, "splits_info.json")
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
            case_ids=val_case_ids,
            case_id_order=val_case_id_order,
            num_dimensions=num_dimensions
        )

    # Validate case_id_order consistency across nodes
    for node in datasets_train:
        if datasets_train[node].case_ids != datasets_train[list(datasets_train.keys())[0]].case_ids:
            logger.error(f"Case ID order inconsistent for node {node}")
            raise ValueError(f"Case ID {node}")
        if datasets_val[node].case_ids != datasets_val[list(datasets_val.keys())[0]].case_ids:
            logger.error(f"Case ID {node} inconsistent for node in validation")
            continue

    # Create DataLoaders
    dataloaders_train = {}
    dataloaders_val = {}
    for node in datasets_train:
        train_indices = list(range(len(datasets_train[node])))
        val_indices = list(range(len(datasets_val[node])))
        dataloaders_train[node] = DataLoader(
            datasets_train[node],
            batch_size=batch_size,
            sampler=OrderedSampler(train_indices, num_workers=num_workers),
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )
        dataloaders_val[node] = DataLoader(
            datasets_val[node],
            batch_size=batch_size,
            sampler=OrderedSampler(val_indices, num_workers=num_workers),
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

    # Model, optimizer, and scheduler
    onnx_save_path = os.path.join(save_dir, "model_config_initial.onnx")
    model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions, onnx_save_path=onnx_save_path).to(device)
    optimizer = optim.Adam([
    # 预训练部分：较低的学习率和权重衰减
    {"params": sub_networks["encoder"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": sub_networks["decoder"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": sub_networks["classifier_n5"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": sub_networks["classifier_n6"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": sub_networks["classifier_n7"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": sub_networks["classifier_n8"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": sub_networks["classifier_n9"].parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    # 辅助分类头：较高的学习率和权重衰减
    {"params": sub_networks["classifier_n5_calcification"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n6_calcification"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n7_calcification"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n8_calcification"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n9_calcification"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n5_bleeding"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n6_bleeding"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n7_bleeding"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n8_bleeding"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n9_bleeding"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n5_ulcer"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n6_ulcer"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n7_ulcer"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n8_ulcer"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n9_ulcer"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n5_fibrous_cap_rupture"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n6_fibrous_cap_rupture"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n7_fibrous_cap_rupture"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n8_fibrous_cap_rupture"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n9_fibrous_cap_rupture"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n5_lipid_core"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n6_lipid_core"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n7_lipid_core"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n8_lipid_core"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n9_lipid_core"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n5_thickness"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n6_thickness"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n7_thickness"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n8_thickness"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    {"params": sub_networks["classifier_n9_thickness"].parameters(), "lr": 1e-3, "weight_decay": 5e-4},
    ])
    
    # Select scheduler
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        logger.info("Using CosineAnnealingLR scheduler")
    elif scheduler_type == "poly":
        scheduler = PolynomialLR(optimizer, max_epochs=num_epochs, power=0.9, eta_min=0)
        logger.info("Using PolynomialLR scheduler")
    elif scheduler_type == "reduce_plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, eta_min=0, verbose=True)
        logger.info("Using ReduceLROnPlateau scheduler")
    else:
        raise ValueError(f"Invalid scheduler_type: {scheduler_type}. Choose 'cosine', 'poly', or 'reduce_plateau'.")

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

        # Get current learning rate
        current_lr = [group['lr'] for group in optimizer.param_groups][0]
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_task_losses": train_task_losses,
            "train_metrics": train_metrics,
            "learning_rate": current_lr
        }

        if (epoch + 1) % validation_interval == 0:
            val_loss, val_task_losses, val_metrics = validate(
                model, dataloaders_val, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, debug=True
            )

            epoch_log.update({
                "val_loss": val_loss,
                "val_task_losses": val_task_losses,
                "metrics": val_metrics
            })

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

            # Update learning rate for ReduceLROnPlateau
            if scheduler_type == "reduce_plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            log["epochs"].append(epoch_log)
            log_save_path = os.path.join(save_dir, "training_log.json")
            with open(log_save_path, "w") as f:
                json.dump(log, f, indent=4)
            logger.info(f"Training log updated at {log_save_path}")

if __name__ == "__main__":
    # 指定使用第三张显卡
    device_id = 1  # 第三张显卡的索引（从0开始）
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")
    main()
