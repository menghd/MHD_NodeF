"""
MHD_Nodet Project - Training Module
===================================
This module implements the training pipeline for the MHD_Nodet project, integrating network, dataset, and evaluation components.
- Supports custom data loading from separate train and val directories, and batch-consistent augmentations.
- Includes learning rate scheduling (warmup + cosine annealing) and early stopping for robust training.

项目：MHD_Nodet - 训练模块
本模块实现 MHD_Nodet 项目的训练流水线，集成网络、数据集和评估组件。
- 支持从单独的 train 和 val 目录加载自定义数据，以及批次一致的数据增强。
- 包含学习率调度（预热 + 余弦退火）和早停机制以确保稳健训练。

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
sys.path.append(r"C:\Users\souray\Desktop\Codes")
from node_toolkit.new_node_net import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset, MinMaxNormalize, ZScoreNormalize, RandomRotate, RandomFlip, RandomShift, RandomZoom, OneHot, OrderedSampler, worker_init_fn
from node_toolkit.node_utils import train, validate, WarmupCosineAnnealingLR
from node_toolkit.node_results import (
    node_lp_loss, node_focal_loss, node_dice_loss, node_iou_loss,
    node_recall_metric, node_precision_metric, node_f1_metric, node_dice_metric, node_iou_metric, node_mse_metric, node_accuracy_metric, node_specificity_metric
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    base_data_dir = r"C:\Users\souray\Desktop\new_Tr"
    train_data_dir = os.path.join(base_data_dir, "train")
    val_data_dir = os.path.join(base_data_dir, "val")
    save_dir = r"C:\Users\souray\Desktop\MHDNet0523"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 12
    num_dimensions = 3
    num_epochs = 400
    learning_rate = 1e-3
    validation_interval = 1
    patience = 20
    warmup_epochs = 10
    num_workers = 0

    # Subnetwork 12 (Segmentation task: Plaque, binary segmentation)
    node_configs_merge = {
        0: (1, 64, 64, 64), 1: (1, 64, 64, 64), 2: (1, 64, 64, 64), 3: (1, 64, 64, 64), 4: (2, 64, 64, 64),
        5: (8, 64, 64, 64), 6: (8, 32, 32, 32), 7: (8, 64, 64, 64), 8: (8, 64, 64, 64), 9: (8, 64, 64, 64), 10: (8, 64, 64, 64),
        11: (48, 64, 64, 64)
    }
    hyperedge_configs_merge = {
        "e1": {"src_nodes": [0, 1, 2, 3, 4], "dst_nodes": [10], "params": {
            "convs": [torch.Size([8, 6, 3, 3, 3]), torch.Size([8, 8, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},
        "e2": {"src_nodes": [0], "dst_nodes": [5], "params": {
            "convs": [torch.Size([8, 1, 3, 3, 3]), torch.Size([8, 8, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},
        "e3": {"src_nodes": [1], "dst_nodes": [6], "params": {
            "convs": [torch.Size([8, 1, 3, 3, 3]), torch.Size([8, 8, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},
        "e4": {"src_nodes": [2], "dst_nodes": [7], "params": {
            "convs": [torch.Size([8, 1, 3, 3, 3]), torch.Size([8, 8, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},
        "e5": {"src_nodes": [3], "dst_nodes": [8], "params": {
            "convs": [torch.Size([8, 1, 3, 3, 3]), torch.Size([8, 8, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},      
        "e6": {"src_nodes": [4], "dst_nodes": [9], "params": {
            "convs": [torch.Size([8, 2, 3, 3, 3]), torch.Size([8, 8, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},    
        "e7": {"src_nodes": [5, 6, 7, 8, 9], "dst_nodes": [11], "params": {
            "convs": [torch.Size([48, 40, 3, 3, 3]), torch.Size([48, 48, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},    
        "e8": {"src_nodes": [10], "dst_nodes": [11], "params": {
            "convs": [torch.Size([48, 8, 3, 3, 3]), torch.Size([48, 48, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (64, 64, 64)}},    
    }
    in_nodes_merge = [0, 1, 2, 3, 4]
    out_nodes_merge = [11]

    # Subnetwork 13 (Target node for reshaped features)
    node_configs_split = {
        0: (8, 1, 1, 1), 1: (4, 1, 1, 1), 2: (4, 1, 1, 1), 3: (2, 1, 1, 1), 4: (2, 1, 1, 1), 5: (2, 1, 1, 1), 6: (2, 1, 1, 1), 7: (3, 1, 1, 1), 8: (1, 1, 1, 1),
        9: (32, 1, 1, 1), 10: (32, 1, 1, 1), 11: (32, 1, 1, 1), 12: (32, 1, 1, 1), 13: (32, 1, 1, 1), 14: (32, 1, 1, 1), 15: (32, 1, 1, 1), 16: (32, 1, 1, 1), 17: (32, 1, 1, 1), 18: (32, 1, 1, 1),
        19: (320, 1, 1, 1),
        20: (8, 1, 1, 1), 21: (4, 1, 1, 1), 22: (4, 1, 1, 1), 23: (2, 1, 1, 1), 24: (2, 1, 1, 1), 25: (2, 1, 1, 1), 26: (2, 1, 1, 1), 27: (3, 1, 1, 1), 28: (1, 1, 1, 1),
    }
    hyperedge_configs_split = {
        "e1": {"src_nodes": [19], "dst_nodes": [18], "params": {
            "convs": [torch.Size([32, 320, 1, 1, 1]), torch.Size([32, 32, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},
        "e2": {"src_nodes": [19], "dst_nodes": [9, 10, 11, 12, 13, 14, 15, 16, 17], "params": {
            "convs": [torch.Size([288, 320, 1, 1, 1]), torch.Size([288, 288, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},
        "e3": {"src_nodes": [18], "dst_nodes": [0, 1, 2, 3, 4, 5, 6, 7, 8], "params": {
            "convs": [torch.Size([28, 32, 1, 1, 1]), torch.Size([28, 28, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},
        "e4": {"src_nodes": [9], "dst_nodes": [0], "params": {
            "convs": [torch.Size([8, 32, 1, 1, 1]), torch.Size([8, 8, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},
        "e5": {"src_nodes": [10], "dst_nodes": [1], "params": {
            "convs": [torch.Size([4, 32, 1, 1, 1]), torch.Size([4, 4, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},   
        "e6": {"src_nodes": [11], "dst_nodes": [2], "params": {
            "convs": [torch.Size([4, 32, 1, 1, 1]), torch.Size([4, 4, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}}, 
        "e7": {"src_nodes": [12], "dst_nodes": [3], "params": {
            "convs": [torch.Size([2, 32, 1, 1, 1]), torch.Size([2, 2, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},  
        "e8": {"src_nodes": [13], "dst_nodes": [4], "params": {
            "convs": [torch.Size([2, 32, 1, 1, 1]), torch.Size([2, 2, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},    
        "e9": {"src_nodes": [14], "dst_nodes": [5], "params": {
            "convs": [torch.Size([2, 32, 1, 1, 1]), torch.Size([2, 2, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},
        "e10": {"src_nodes": [15], "dst_nodes": [6], "params": {
            "convs": [torch.Size([2, 32, 1, 1, 1]), torch.Size([2, 2, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},     
        "e11": {"src_nodes": [16], "dst_nodes": [7], "params": {
            "convs": [torch.Size([3, 32, 1, 1, 1]), torch.Size([3, 3, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},  
        "e12": {"src_nodes": [17], "dst_nodes": [8], "params": {
            "convs": [torch.Size([1, 32, 1, 1, 1]), torch.Size([1, 1, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}}, 
    }
    in_nodes_split = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    out_nodes_split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 26, 27, 28]

    node_configs_process = {
        0:(48, 64, 64, 64), 1: (32, 64, 64, 64), 2: (128, 16, 16, 16), 3: (512, 4, 4, 4), 4: (2048, 1, 1, 1), 5: (320, 1, 1, 1)
    }
    hyperedge_configs_process = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "convs": [torch.Size([32, 48, 3, 3, 3])],
            "reqs": [True],
            "norms": [None],
            "acts": [None],
            "feature_size": (64, 64, 64)},},
        "e2": {"src_nodes": [1], "dst_nodes": [2], "params": {
            "convs": [torch.Size([128, 32, 3, 3, 3]), torch.Size([128, 128, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (32, 32, 32)},
            "intp":"max"},
        "e3": {"src_nodes": [2], "dst_nodes": [3], "params": {
            "convs": [torch.Size([512, 128, 3, 3, 3]), torch.Size([512, 512, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (8, 8, 8)},
            "intp":"max"},
        "e4": {"src_nodes": [3], "dst_nodes": [4], "params": {
            "convs": [torch.Size([2048, 512, 3, 3, 3]), torch.Size([2048, 2048, 3, 3, 3])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (2, 2, 2)},
            "intp":"max"},
        "e5": {"src_nodes": [1], "dst_nodes": [2,3,4], "params": {
            "convs": [torch.Size([128+512+2048, 32, 3, 3, 3]), torch.Size([128+512+2048, 128+512+2048, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},
        "e6": {"src_nodes": [1,2], "dst_nodes": [3,4], "params": {
            "convs": [torch.Size([512+2048, 32+128, 3, 3, 3]), torch.Size([512+2048, 512+2048, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},      
        "e7": {"src_nodes": [1,2,3], "dst_nodes": [4], "params": {
            "convs": [torch.Size([2048, 32+128+512, 3, 3, 3]), torch.Size([2048, 2048, 1, 1, 1])],
            "reqs": [True, True],
            "norms": ["batch", "batch"],
            "acts": ["leakyrelu", "leakyrelu"],
            "feature_size": (1, 1, 1)}},    
        "e8": {"src_nodes": [4], "dst_nodes": [5], "params": {
            "convs": [torch.Size([320, 2048, 3, 3, 3])],
            "reqs": [True],
            "norms": [None],
            "acts": [None],
            "feature_size": (1, 1, 1)}},  
    }
    in_nodes_process = [0]
    out_nodes_process = [5]

    # Global node mapping
    node_mapping = [
        (100, "merge", 0), (101, "merge", 1), (102, "merge", 2), (103, "merge", 3), (104, "merge", 4), (200, "merge", 11), 
        (200, "process", 0), (319, "process", 5), 
        (319, "split", 19), (300, "split", 0), (301, "split", 1), (302, "split", 2), (303, "split", 3), (304, "split", 4), (305, "split", 5), (306, "split", 6), (307, "split", 7), (308, "split", 8), 
        (320, "split", 20), (321, "split", 21), (322, "split", 22), (323, "split", 23), (324, "split", 24), (325, "split", 25), (326, "split", 26), (327, "split", 27), (328, "split", 28) 
    ]

    # Instantiate subnetworks
    sub_networks_configs = {
        "merge": (node_configs_merge, hyperedge_configs_merge, in_nodes_merge, out_nodes_merge),
        "process": (node_configs_process, hyperedge_configs_process, in_nodes_process, out_nodes_process),
        "split": (node_configs_split, hyperedge_configs_split, in_nodes_split, out_nodes_split),
    }
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # Global input and output nodes
    in_nodes = [100, 101, 102, 103, 104, 320, 321, 322, 323, 324, 325, 326, 327, 328]
    out_nodes = [300, 301, 302, 303, 304, 305, 306, 307, 308, 320, 321, 322, 323, 324, 325, 326, 327, 328]

    # Node suffix mapping
    node_suffix = [
        (100, "0000"), (101, "0001"), (102, "0002"), (103, "0003"), (104, "0004"),
        (320, "0005"), (321, "0006"), (322, "0007"), (323, "0008"), (324, "0009"), (325, "0010"), (326, "0011"), (327, "0012"), (328, "0013"), 
    ]

    # Instantiate transformations
    random_rotate = RandomRotate(max_angle=5)
    random_shift = RandomShift(max_shift=5)
    random_zoom = RandomZoom(zoom_range=(0.9, 1.1))
    one_hot2 = OneHot(num_classes=2)
    one_hot3 = OneHot(num_classes=3)
    one_hot4 = OneHot(num_classes=4)
    one_hot8 = OneHot(num_classes=8)


    # Node transformation configuration for train and validate
    node_transforms = {
        "train": {
            100: [random_rotate, random_shift, random_zoom],
            101: [random_rotate, random_shift, random_zoom],
            102: [random_rotate, random_shift, random_zoom],
            103: [random_rotate, random_shift, random_zoom],
            104: [random_rotate, random_shift, random_zoom, one_hot2],
            320: [one_hot8],
            321: [one_hot4],
            322: [one_hot4],
            323: [one_hot2],
            324: [one_hot2],
            325: [one_hot2],
            326: [one_hot2],
            327: [one_hot3],
            328: [],
        },
        "validate": {
            104: [one_hot2],
            320: [one_hot8],
            321: [one_hot4],
            322: [one_hot4],
            323: [one_hot2],
            324: [one_hot2],
            325: [one_hot2],
            326: [one_hot2],
            327: [one_hot3],
            328: [],
        }
    }

    # Task configuration
    task_configs = {
        "type_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 300, "target_node": 320, "weight": 0.9, "params": {}},
                {"fn": node_lp_loss, "src_node": 300, "target_node": 320, "weight": 0.1, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 300, "target_node": 320, "params": {}},
                {"fn": node_precision_metric, "src_node": 300, "target_node": 320, "params": {}},
                {"fn": node_f1_metric, "src_node": 300, "target_node": 320, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 300, "target_node": 320, "params": {}},
                {"fn": node_specificity_metric, "src_node": 300, "target_node": 320, "params": {}},
            ],
        },
        "main_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 301, "target_node": 321, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 301, "target_node": 321, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 301, "target_node": 321, "params": {}},
                {"fn": node_precision_metric, "src_node": 301, "target_node": 321, "params": {}},
                {"fn": node_f1_metric, "src_node": 301, "target_node": 321, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 301, "target_node": 321, "params": {}},
                {"fn": node_specificity_metric, "src_node": 301, "target_node": 321, "params": {}},
            ],
        },
        "vice_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 302, "target_node": 322, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 302, "target_node": 322, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 302, "target_node": 322, "params": {}},
                {"fn": node_precision_metric, "src_node": 302, "target_node": 322, "params": {}},
                {"fn": node_f1_metric, "src_node": 302, "target_node": 322, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 302, "target_node": 322, "params": {}},
                {"fn": node_specificity_metric, "src_node": 302, "target_node": 322, "params": {}},
            ],
        },
        "calc_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 303, "target_node": 323, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 303, "target_node": 323, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 303, "target_node": 323, "params": {}},
                {"fn": node_precision_metric, "src_node": 303, "target_node": 323, "params": {}},
                {"fn": node_f1_metric, "src_node": 303, "target_node": 323, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 303, "target_node": 323, "params": {}},
                {"fn": node_specificity_metric, "src_node": 303, "target_node": 323, "params": {}},
            ],
        },
        "bleed_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 304, "target_node": 324, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 304, "target_node": 324, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 304, "target_node": 324, "params": {}},
                {"fn": node_precision_metric, "src_node": 304, "target_node": 324, "params": {}},
                {"fn": node_f1_metric, "src_node": 304, "target_node": 324, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 304, "target_node": 324, "params": {}},
                {"fn": node_specificity_metric, "src_node": 304, "target_node": 324, "params": {}},
            ],
        },
        "ulcer_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 305, "target_node": 325, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 305, "target_node": 325, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 305, "target_node": 325, "params": {}},
                {"fn": node_precision_metric, "src_node": 305, "target_node": 325, "params": {}},
                {"fn": node_f1_metric, "src_node": 305, "target_node": 325, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 305, "target_node": 325, "params": {}},
                {"fn": node_specificity_metric, "src_node": 305, "target_node": 325, "params": {}},
            ],
        },
        "cap_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 306, "target_node": 326, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 306, "target_node": 326, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 306, "target_node": 326, "params": {}},
                {"fn": node_precision_metric, "src_node": 306, "target_node": 326, "params": {}},
                {"fn": node_f1_metric, "src_node": 306, "target_node": 326, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 306, "target_node": 326, "params": {}},
                {"fn": node_specificity_metric, "src_node": 306, "target_node": 326, "params": {}},
            ],
        },
        "lipid_cls": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 307, "target_node": 327, "weight": 0.09, "params": {}},
                {"fn": node_lp_loss, "src_node": 307, "target_node": 327, "weight": 0.01, "params": {}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 307, "target_node": 327, "params": {}},
                {"fn": node_precision_metric, "src_node": 307, "target_node": 327, "params": {}},
                {"fn": node_f1_metric, "src_node": 307, "target_node": 327, "params": {}},
                {"fn": node_accuracy_metric, "src_node": 307, "target_node": 327, "params": {}},
                {"fn": node_specificity_metric, "src_node": 307, "target_node": 327, "params": {}},
            ],
        },
        "mwt_reg": {
            "loss": [
                {"fn": node_lp_loss, "src_node": 308, "target_node": 328, "weight": 0.1, "params": {}},
            ],
            "metric": [
                {"fn": node_mse_metric, "src_node": 308, "target_node": 328, "params": {}},
            ],
        },
    }

    # Collect case IDs for train and val
    def get_case_ids(data_dir, suffix, file_ext):
        all_files = sorted(os.listdir(data_dir))
        case_ids = set()
        for file in all_files:
            if file.startswith('case_') and file.endswith(f'_{suffix}{file_ext}'):
                case_id = file.split('_')[1]
                case_ids.add(case_id)
        return sorted(list(case_ids))

    # Initialize suffix to nodes mapping
    suffix_to_nodes = {}
    for node, suffix in node_suffix:
        if suffix not in suffix_to_nodes:
            suffix_to_nodes[suffix] = []
        suffix_to_nodes[suffix].append(node)

    # Get case IDs for train and val directories
    train_suffix_case_ids = {}
    val_suffix_case_ids = {}
    for suffix in suffix_to_nodes:
        train_suffix_case_ids[suffix] = get_case_ids(train_data_dir, suffix, '.nii.gz') or get_case_ids(train_data_dir, suffix, '.csv')
        val_suffix_case_ids[suffix] = get_case_ids(val_data_dir, suffix, '.nii.gz') or get_case_ids(val_data_dir, suffix, '.csv')

    # Find common case IDs
    train_common_case_ids = set.intersection(*(set(case_ids) for case_ids in train_suffix_case_ids.values()))
    val_common_case_ids = set.intersection(*(set(case_ids) for case_ids in val_suffix_case_ids.values()))
    if not train_common_case_ids:
        raise ValueError("No common case_ids found in train directory!")
    if not val_common_case_ids:
        raise ValueError("No common case_ids found in val directory!")
    train_case_ids = sorted(list(train_common_case_ids))
    val_case_ids = sorted(list(val_common_case_ids))

    # Log incomplete cases
    for suffix, case_ids in train_suffix_case_ids.items():
        missing = set(case_ids) - train_common_case_ids
        if missing:
            logger.warning(f"Incomplete train cases for suffix {suffix}: {sorted(list(missing))}")
    for suffix, case_ids in val_suffix_case_ids.items():
        missing = set(case_ids) - val_common_case_ids
        if missing:
            logger.warning(f"Incomplete val cases for suffix {suffix}: {sorted(list(missing))}")

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
        "val_count": len(val_case_ids),
    }
    split_save_path = os.path.join(save_dir, "data_split.json")
    with open(split_save_path, "w") as f:
        json.dump(split_info, f, indent=4)
    logger.info(f"Data split saved to {split_save_path}")

    # Create datasets
    datasets_train = {}
    datasets_val = {}
    for node, suffix in node_suffix:
        target_shape = None
        for global_node, sub_net_name, sub_node_id in node_mapping:
            if global_node == node:
                target_shape = sub_networks[sub_net_name].node_configs[sub_node_id]
                break
        if target_shape is None:
            raise ValueError(f"Node {node} not found in node_mapping")
        datasets_train[node] = NodeDataset(
            train_data_dir, node, suffix, target_shape, node_transforms["train"].get(node, []),
            node_mapping=node_mapping, sub_networks=sub_networks,
            case_ids=train_case_ids, case_id_order=train_case_id_order,
            num_dimensions=num_dimensions
        )
        datasets_val[node] = NodeDataset(
            val_data_dir, node, suffix, target_shape, node_transforms["validate"].get(node, []),
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

    # Create DataLoaders with custom sampler and worker initialization
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
    model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, T_max=num_epochs, eta_min=1e-6)

    # Save initial ONNX model before training starts
    model.eval()
    input_shapes = [(batch_size, *sub_networks[sub_net_name].node_configs[sub_node_id])
                    for global_node in in_nodes
                    for g_node, sub_net_name, sub_node_id in node_mapping
                    if g_node == global_node]
    inputs = [torch.randn(*shape).to(device) for shape in input_shapes]
    dynamic_axes = {
        **{f"input_{node}": {0: "batch_size"} for node in in_nodes},
        **{f"output_{node}": {0: "batch_size"} for node in out_nodes},
    }
    onnx_save_path = os.path.join(save_dir, "model_config_initial.onnx")
    torch.onnx.export(
        model,
        inputs,
        onnx_save_path,
        input_names=[f"input_{node}" for node in in_nodes],
        output_names=[f"output_{node}" for node in out_nodes],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    logger.info(f"Initial ONNX model saved to {onnx_save_path}")

    # Early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    log = {"epochs": []}

    for epoch in range(num_epochs):
        # Generate unique batch seeds for each epoch and worker
        epoch_seed = seed + epoch
        np.random.seed(epoch_seed)
        batch_seeds = np.random.randint(0, 1000000, size=len(dataloaders_train[node]))
        logger.info(f"Epoch {epoch + 1}: Generated {len(batch_seeds)} batch seeds")

        for batch_idx in range(len(dataloaders_train[node])):
            # Assign unique seed for each batch
            batch_seed = int(batch_seeds[batch_idx])
            logger.debug(f"Batch {batch_idx}, Seed {batch_seed}")
            for node in datasets_train:
                datasets_train[node].set_batch_seed(batch_seed)
            for node in datasets_val:
                datasets_val[node].set_batch_seed(batch_seed)

        train_loss, train_task_losses, train_metrics = train(
            model, dataloaders_train, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, node_transforms["train"]
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
                save_path = os.path.join(save_dir, "model_best.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")
            else:
                epochs_no_improve += validation_interval
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        scheduler.step()
        log["epochs"].append(epoch_log)

    log_save_path = os.path.join(save_dir, "training_log.json")
    with open(log_save_path, "w") as f:
        json.dump(log, f, indent=4)
    logger.info(f"Training log saved to {log_save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
