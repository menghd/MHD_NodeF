import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
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

class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        return super().get_lr()

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 数据路径
    data_dir = r"C:\Users\souray\Desktop\Tr"

    # 保存路径
    save_dir = r"C:\Users\souray\Desktop\MHDNet0419"
    os.makedirs(save_dir, exist_ok=True)

    # 超参数
    batch_size = 2
    num_dimensions = 3
    num_epochs = 200
    learning_rate = 1e-1
    k_folds = 5
    validation_interval = 1
    patience = 200
    warmup_epochs = 10

    # 子网络1（预处理）
    node_configs_pre = {
        0: (1, 64, 64, 64), 1: (1, 64, 64, 64), 2: (1, 64, 64, 64), 3: (1, 64, 64, 64), 4: (2, 64, 64, 64),
        5: (32, 64, 64, 64), 6: (32, 64, 64, 64), 7: (32, 64, 64, 64), 8: (32, 64, 64, 64), 9: (32, 64, 64, 64),
        10: (32, 64, 64, 64), 11: (64, 32, 32, 32),
    }
    hyperedge_configs_pre = {
        "e1": {"src_nodes": [0, 1, 2, 3, 4], "dst_nodes": [5], "params": {
            "convs": [(32, 5, 5, 5), (32, 5, 5, 5)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e2": {"src_nodes": [0], "dst_nodes": [6], "params": {
            "convs": [(32, 5, 5, 5), (32, 5, 5, 5)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e3": {"src_nodes": [1], "dst_nodes": [7], "params": {
            "convs": [(32, 5, 5, 5), (32, 5, 5, 5)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e4": {"src_nodes": [2], "dst_nodes": [8], "params": {
            "convs": [(32, 5, 5, 5), (32, 5, 5, 5)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e5": {"src_nodes": [3], "dst_nodes": [9], "params": {
            "convs": [(32, 5, 5, 5), (32, 5, 5, 5)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e6": {"src_nodes": [4], "dst_nodes": [10], "params": {
            "convs": [(32, 5, 5, 5), (32, 5, 5, 5)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e7": {"src_nodes": [6, 7, 8, 9, 10], "dst_nodes": [11], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e8": {"src_nodes": [5], "dst_nodes": [11], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64), "out_p": "max"}},
    }
    in_nodes_pre = [0, 1, 2, 3, 4]
    out_nodes_pre = [11]
    node_dtype_pre = {4: "long"}

    # 子网络2（主网络，模仿 ResNet-18）
    node_configs_main = {
        0: (64, 32, 32, 32), 1: (64, 32, 32, 32), 2: (64, 32, 32, 32), 3: (64, 32, 32, 32), 4: (64, 32, 32, 32),
        5: (128, 16, 16, 16), 6: (128, 16, 16, 16), 7: (128, 16, 16, 16), 8: (128, 16, 16, 16),
        9: (256, 8, 8, 8), 10: (256, 8, 8, 8), 11: (256, 8, 8, 8), 12: (256, 8, 8, 8),
        13: (512, 4, 4, 4), 14: (512, 4, 4, 4), 15: (512, 4, 4, 4), 16: (512, 4, 4, 4), 17: (512, 4, 4, 4),
    }
    node_dtype_main = {k: "float" for k in node_configs_main}
    hyperedge_configs_main = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32)}},
        "e2": {"src_nodes": [1], "dst_nodes": [2], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32)}},
        "e3": {"src_nodes": [0], "dst_nodes": [2], "params": {"convs": [], "feature_size": (32, 32, 32)}},
        "e4": {"src_nodes": [2], "dst_nodes": [3], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32)}},
        "e5": {"src_nodes": [3], "dst_nodes": [4], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32)}},
        "e6": {"src_nodes": [2], "dst_nodes": [4], "params": {"convs": [], "feature_size": (32, 32, 32)}},
        "e7": {"src_nodes": [4], "dst_nodes": [5], "params": {
            "convs": [(128, 3, 3, 3), (128, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32), "out_p": "avg"}},
        "e8": {"src_nodes": [5], "dst_nodes": [6], "params": {
            "convs": [(128, 3, 3, 3), (128, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (16, 16, 16)}},
        "e9": {"src_nodes": [4], "dst_nodes": [6], "params": {
            "convs": [(128, 1, 1, 1)], "norms": ["batch"], "feature_size": (32, 32, 32), "out_p": "avg"}},
        "e10": {"src_nodes": [6], "dst_nodes": [7], "params": {
            "convs": [(128, 3, 3, 3), (128, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (16, 16, 16)}},
        "e11": {"src_nodes": [7], "dst_nodes": [8], "params": {
            "convs": [(128, 3, 3, 3), (128, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (16, 16, 16)}},
        "e12": {"src_nodes": [6], "dst_nodes": [8], "params": {"convs": [], "feature_size": (16, 16, 16)}},
        "e13": {"src_nodes": [8], "dst_nodes": [9], "params": {
            "convs": [(256, 3, 3, 3), (256, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (16, 16, 16), "out_p": "avg"}},
        "e14": {"src_nodes": [9], "dst_nodes": [10], "params": {
            "convs": [(256, 3, 3, 3), (256, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (8, 8, 8)}},
        "e15": {"src_nodes": [8], "dst_nodes": [10], "params": {
            "convs": [(256, 1, 1, 1)], "norms": ["batch"], "feature_size": (16, 16, 16), "out_p": "avg"}},
        "e16": {"src_nodes": [10], "dst_nodes": [11], "params": {
            "convs": [(256, 3, 3, 3), (256, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (8, 8, 8)}},
        "e17": {"src_nodes": [11], "dst_nodes": [12], "params": {
            "convs": [(256, 3, 3, 3), (256, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (8, 8, 8)}},
        "e18": {"src_nodes": [10], "dst_nodes": [12], "params": {"convs": [], "feature_size": (8, 8, 8)}},
        "e19": {"src_nodes": [12], "dst_nodes": [13], "params": {
            "convs": [(512, 3, 3, 3), (512, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (8, 8, 8), "out_p": "avg"}},
        "e20": {"src_nodes": [13], "dst_nodes": [14], "params": {
            "convs": [(512, 3, 3, 3), (512, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (4, 4, 4)}},
        "e21": {"src_nodes": [12], "dst_nodes": [14], "params": {
            "convs": [(512, 1, 1, 1)], "norms": ["batch"], "feature_size": (8, 8, 8), "out_p": "avg"}},
        "e22": {"src_nodes": [14], "dst_nodes": [15], "params": {
            "convs": [(512, 3, 3, 3), (512, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (4, 4, 4)}},
        "e23": {"src_nodes": [15], "dst_nodes": [16], "params": {
            "convs": [(512, 3, 3, 3), (512, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (4, 4, 4)}},
        "e24": {"src_nodes": [14], "dst_nodes": [16], "params": {"convs": [], "feature_size": (4, 4, 4)}},
        "e25": {"src_nodes": [16], "dst_nodes": [17], "params": {
            "convs": [(512, 3, 3, 3)], "norms": ["batch"], "acts": ["leakyrelu"], "feature_size": (4, 4, 4)}},
    }
    in_nodes_main = [0]
    out_nodes_main = [17]

    # 子网络3（回归任务：管壁厚度）
    node_configs_regression = {0: (512, 4, 4, 4), 1: (1, 1, 1, 1)}
    node_dtype_regression = {k: "float" for k in node_configs_regression}
    hyperedge_configs_regression = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(1, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_regression = [0]
    out_nodes_regression = [1]

    # 子网络4（8分类任务：type）
    node_configs_cls_8 = {0: (512, 4, 4, 4), 1: (8, 1, 1, 1)}
    node_dtype_cls_8 = {k: "float" for k in node_configs_cls_8}
    hyperedge_configs_cls_8 = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(8, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_8 = [0]
    out_nodes_cls_8 = [1]

    # 子网络5（4分类任务：main）
    node_configs_cls_4_main = {0: (512, 4, 4, 4), 1: (4, 1, 1, 1)}
    node_dtype_cls_4_main = {k: "float" for k in node_configs_cls_4_main}
    hyperedge_configs_cls_4_main = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(4, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_4_main = [0]
    out_nodes_cls_4_main = [1]

    # 子网络6（4分类任务：vice）
    node_configs_cls_4_vice = {0: (512, 4, 4, 4), 1: (4, 1, 1, 1)}
    node_dtype_cls_4_vice = {k: "float" for k in node_configs_cls_4_vice}
    hyperedge_configs_cls_4_vice = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(4, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_4_vice = [0]
    out_nodes_cls_4_vice = [1]

    # 子网络7（2分类任务：钙化）
    node_configs_cls_2_calc = {0: (512, 4, 4, 4), 1: (2, 1, 1, 1)}
    node_dtype_cls_2_calc = {k: "float" for k in node_configs_cls_2_calc}
    hyperedge_configs_cls_2_calc = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(2, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_2_calc = [0]
    out_nodes_cls_2_calc = [1]

    # 子网络8（2分类任务：出血）
    node_configs_cls_2_bleed = {0: (512, 4, 4, 4), 1: (2, 1, 1, 1)}
    node_dtype_cls_2_bleed = {k: "float" for k in node_configs_cls_2_bleed}
    hyperedge_configs_cls_2_bleed = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(2, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_2_bleed = [0]
    out_nodes_cls_2_bleed = [1]

    # 子网络9（2分类任务：溃疡）
    node_configs_cls_2_ulcer = {0: (512, 4, 4, 4), 1: (2, 1, 1, 1)}
    node_dtype_cls_2_ulcer = {k: "float" for k in node_configs_cls_2_ulcer}
    hyperedge_configs_cls_2_ulcer = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(2, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_2_ulcer = [0]
    out_nodes_cls_2_ulcer = [1]

    # 子网络10（2分类任务：纤维帽）
    node_configs_cls_2_cap = {0: (512, 4, 4, 4), 1: (2, 1, 1, 1)}
    node_dtype_cls_2_cap = {k: "float" for k in node_configs_cls_2_cap}
    hyperedge_configs_cls_2_cap = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(2, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_2_cap = [0]
    out_nodes_cls_2_cap = [1]

    # 子网络11（3分类任务：脂质）
    node_configs_cls_3_lipid = {0: (512, 4, 4, 4), 1: (3, 1, 1, 1)}
    node_dtype_cls_3_lipid = {k: "float" for k in node_configs_cls_3_lipid}
    hyperedge_configs_cls_3_lipid = {
        "e1": {"src_nodes": [0], "dst_nodes": [1], "params": {
            "feature_size": (1, 1, 1), "convs": [(3, 1, 1, 1)], "norms": [None], "acts": ["leakyrelu"]}}
    }
    in_nodes_cls_3_lipid = [0]
    out_nodes_cls_3_lipid = [1]

    # 子网络12（分割任务：斑块，二值分割）
    node_configs_segmentation = {
        0: (1, 64, 64, 64), 1: (1, 64, 64, 64), 2: (1, 64, 64, 64), 3: (1, 64, 64, 64), 4: (2, 64, 64, 64),
        5: (64, 64, 64, 64), 6: (128, 32, 32, 32), 7: (64, 64, 64, 64), 8: (2, 64, 64, 64),
    }
    node_dtype_segmentation = {4: "long"}
    hyperedge_configs_segmentation = {
        "e1": {"src_nodes": [0, 1, 2, 3, 4], "dst_nodes": [5], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64)}},
        "e2": {"src_nodes": [5], "dst_nodes": [6], "params": {
            "convs": [(128, 3, 3, 3), (128, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (32, 32, 32), "out_p": "avg"}},
        "e3": {"src_nodes": [5, 6], "dst_nodes": [7], "params": {
            "convs": [(64, 3, 3, 3), (64, 3, 3, 3)], "norms": ["batch", "batch"], "acts": ["leakyrelu", "leakyrelu"], "feature_size": (64, 64, 64), "out_p": "max"}},
        "e4": {"src_nodes": [7], "dst_nodes": [8], "params": {
            "convs": [(2, 3, 3, 3)], "norms": ["batch"], "acts": ["leakyrelu"], "feature_size": (64, 64, 64)}},
    }
    in_nodes_segmentation = [0, 1, 2, 3, 4]
    out_nodes_segmentation = [8]

    # 子网络13（目标节点，存储调整形状的特征）
    node_configs_target = {
        0: (2, 64, 64, 64), 1: (8, 1, 1, 1), 2: (4, 1, 1, 1), 3: (4, 1, 1, 1), 4: (2, 1, 1, 1),
        5: (2, 1, 1, 1), 6: (2, 1, 1, 1), 7: (2, 1, 1, 1), 8: (3, 1, 1, 1), 9: (1, 1, 1, 1),
    }
    node_dtype_target = {
        0: "long", 1: "long", 2: "long", 3: "long", 4: "long",
        5: "long", 6: "long", 7: "long", 8: "long", 9: "float"
    }
    hyperedge_configs_target = {}
    in_nodes_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    out_nodes_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 全局节点映射
    node_mapping = [
        (100, "pre", 0), (101, "pre", 1), (102, "pre", 2), (103, "pre", 3), (104, "pre", 4),
        (200, "pre", 11), (200, "main", 0), (300, "main", 17),
        (300, "regression", 0), (300, "cls_8", 0), (300, "cls_4_main", 0), (300, "cls_4_vice", 0),
        (300, "cls_2_calc", 0), (300, "cls_2_bleed", 0), (300, "cls_2_ulcer", 0), (300, "cls_2_cap", 0),
        (300, "cls_3_lipid", 0), (400, "regression", 1), (401, "cls_8", 1), (402, "cls_4_main", 1),
        (403, "cls_4_vice", 1), (404, "cls_2_calc", 1), (405, "cls_2_bleed", 1), (406, "cls_2_ulcer", 1),
        (407, "cls_2_cap", 1), (408, "cls_3_lipid", 1), (100, "segmentation", 0), (101, "segmentation", 1),
        (102, "segmentation", 2), (103, "segmentation", 3), (104, "segmentation", 4), (500, "segmentation", 8),
        (600, "target", 0), (601, "target", 1), (602, "target", 2), (603, "target", 3), (604, "target", 4),
        (605, "target", 5), (606, "target", 6), (607, "target", 7), (608, "target", 8), (609, "target", 9),
    ]

    # 子网络实例化
    sub_networks_configs = {
        "pre": (node_configs_pre, hyperedge_configs_pre, in_nodes_pre, out_nodes_pre, node_dtype_pre),
        "main": (node_configs_main, hyperedge_configs_main, in_nodes_main, out_nodes_main, node_dtype_main),
        "regression": (node_configs_regression, hyperedge_configs_regression, in_nodes_regression, out_nodes_regression, node_dtype_regression),
        "cls_8": (node_configs_cls_8, hyperedge_configs_cls_8, in_nodes_cls_8, out_nodes_cls_8, node_dtype_cls_8),
        "cls_4_main": (node_configs_cls_4_main, hyperedge_configs_cls_4_main, in_nodes_cls_4_main, out_nodes_cls_4_main, node_dtype_cls_4_main),
        "cls_4_vice": (node_configs_cls_4_vice, hyperedge_configs_cls_4_vice, in_nodes_cls_4_vice, out_nodes_cls_4_vice, node_dtype_cls_4_vice),
        "cls_2_calc": (node_configs_cls_2_calc, hyperedge_configs_cls_2_calc, in_nodes_cls_2_calc, out_nodes_cls_2_calc, node_dtype_cls_2_calc),
        "cls_2_bleed": (node_configs_cls_2_bleed, hyperedge_configs_cls_2_bleed, in_nodes_cls_2_bleed, out_nodes_cls_2_bleed, node_dtype_cls_2_bleed),
        "cls_2_ulcer": (node_configs_cls_2_ulcer, hyperedge_configs_cls_2_ulcer, in_nodes_cls_2_ulcer, out_nodes_cls_2_ulcer, node_dtype_cls_2_ulcer),
        "cls_2_cap": (node_configs_cls_2_cap, hyperedge_configs_cls_2_cap, in_nodes_cls_2_cap, out_nodes_cls_2_cap, node_dtype_cls_2_cap),
        "cls_3_lipid": (node_configs_cls_3_lipid, hyperedge_configs_cls_3_lipid, in_nodes_cls_3_lipid, out_nodes_cls_3_lipid, node_dtype_cls_3_lipid),
        "segmentation": (node_configs_segmentation, hyperedge_configs_segmentation, in_nodes_segmentation, out_nodes_segmentation, node_dtype_segmentation),
        "target": (node_configs_target, hyperedge_configs_target, in_nodes_target, out_nodes_target, node_dtype_target),
    }

    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions, node_dtype)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes, node_dtype) in sub_networks_configs.items()
    }

    # 全局输入输出节点
    in_nodes = [100, 101, 102, 103, 104, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609]
    out_nodes = [400, 401, 402, 403, 404, 405, 406, 407, 408, 500, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609]

    # 节点后缀映射
    node_suffix = [
        (100, "0000"), (101, "0001"), (102, "0002"), (103, "0003"), (104, "0004"),
        (600, "0004"), (601, "0005"), (602, "0006"), (603, "0007"), (604, "0008"),
        (605, "0009"), (606, "0010"), (607, "0011"), (608, "0012"), (609, "0013"),
    ]

    # 预先实例化变换
    random_rotate = RandomRotate(max_angle=5)
    random_flip = RandomFlip()
    random_shift = RandomShift(max_shift=5)
    random_zoom = RandomZoom(zoom_range=(0.9, 1.1))
    min_max_normalize = MinMaxNormalize()
    z_score_normalize = ZScoreNormalize()

    # 节点变换配置
    node_transforms = {
        100: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        101: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        102: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        103: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        104: [random_rotate, random_flip, random_shift, random_zoom],
        600: [random_rotate, random_flip, random_shift, random_zoom],
        601: [], 602: [], 603: [], 604: [], 605: [], 606: [], 607: [], 608: [], 609: [],
    }

    # 任务配置
    task_configs = {
        "regression_thickness": {
            "loss": [
                {"fn": node_lp_loss, "src_node": 400, "target_node": 609, "weight": 1.0, "params": {"p": 2.0}},
            ],
            "metric": [
                {"fn": node_mse_metric, "src_node": 400, "target_node": 609, "params": {}},
            ],
        },
        "cls_8_type": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 401, "target_node": 601, "weight": 1.0, "params": {"alpha": [1.0] * 8, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 401, "target_node": 601, "params": {}},
                {"fn": node_precision_metric, "src_node": 401, "target_node": 601, "params": {}},
                {"fn": node_f1_metric, "src_node": 401, "target_node": 601, "params": {}},
            ],
        },
        "cls_4_main": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 402, "target_node": 602, "weight": 1.0, "params": {"alpha": [1.0] * 4, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 402, "target_node": 602, "params": {}},
                {"fn": node_precision_metric, "src_node": 402, "target_node": 602, "params": {}},
                {"fn": node_f1_metric, "src_node": 402, "target_node": 602, "params": {}},
            ],
        },
        "cls_4_vice": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 403, "target_node": 603, "weight": 1.0, "params": {"alpha": [1.0] * 4, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 403, "target_node": 603, "params": {}},
                {"fn": node_precision_metric, "src_node": 403, "target_node": 603, "params": {}},
                {"fn": node_f1_metric, "src_node": 403, "target_node": 603, "params": {}},
            ],
        },
        "cls_2_calc": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 404, "target_node": 604, "weight": 1.0, "params": {"alpha": [1.0] * 2, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 404, "target_node": 604, "params": {}},
                {"fn": node_precision_metric, "src_node": 404, "target_node": 604, "params": {}},
                {"fn": node_f1_metric, "src_node": 404, "target_node": 604, "params": {}},
            ],
        },
        "cls_2_bleed": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 405, "target_node": 605, "weight": 1.0, "params": {"alpha": [1.0] * 2, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 405, "target_node": 605, "params": {}},
                {"fn": node_precision_metric, "src_node": 405, "target_node": 605, "params": {}},
                {"fn": node_f1_metric, "src_node": 405, "target_node": 605, "params": {}},
            ],
        },
        "cls_2_ulcer": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 406, "target_node": 606, "weight": 1.0, "params": {"alpha": [1.0] * 2, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 406, "target_node": 606, "params": {}},
                {"fn": node_precision_metric, "src_node": 406, "target_node": 606, "params": {}},
                {"fn": node_f1_metric, "src_node": 406, "target_node": 606, "params": {}},
            ],
        },
        "cls_2_cap": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 407, "target_node": 607, "weight": 1.0, "params": {"alpha": [1.0] * 2, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 407, "target_node": 607, "params": {}},
                {"fn": node_precision_metric, "src_node": 407, "target_node": 607, "params": {}},
                {"fn": node_f1_metric, "src_node": 407, "target_node": 607, "params": {}},
            ],
        },
        "cls_3_lipid": {
            "loss": [
                {"fn": node_focal_loss, "src_node": 408, "target_node": 608, "weight": 1.0, "params": {"alpha": [1.0] * 3, "gamma": 2.0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "src_node": 408, "target_node": 608, "params": {}},
                {"fn": node_precision_metric, "src_node": 408, "target_node": 608, "params": {}},
                {"fn": node_f1_metric, "src_node": 408, "target_node": 608, "params": {}},
            ],
        },
        "segmentation_plaque": {
            "loss": [
                {"fn": node_dice_loss, "src_node": 500, "target_node": 600, "weight": 1.0, "params": {}},
                {"fn": node_iou_loss, "src_node": 500, "target_node": 600, "weight": 0.5, "params": {}},
            ],
            "metric": [
                {"fn": node_dice_metric, "src_node": 500, "target_node": 600, "params": {}},
                {"fn": node_iou_metric, "src_node": 500, "target_node": 600, "params": {}},
                {"fn": node_recall_metric, "src_node": 500, "target_node": 600, "params": {}},
                {"fn": node_precision_metric, "src_node": 500, "target_node": 600, "params": {}},
                {"fn": node_f1_metric, "src_node": 500, "target_node": 600, "params": {}},
            ],
        },
    }

    # 创建数据集
    datasets = {}
    for node, suffix in node_suffix:
        target_shape = None
        for g_node, sub_net_name, sub_node_id in node_mapping:
            if g_node == node:
                target_shape = sub_networks[sub_net_name].node_configs[sub_node_id]
                break
        if target_shape is None:
            raise ValueError(f"Node {node} not found in node_mapping")
        datasets[node] = NodeDataset(
            data_dir, node, suffix, target_shape, node_transforms.get(node, []),
            node_mapping=node_mapping, sub_networks=sub_networks
        )

    # 获取所有节点的共同 case_ids
    all_case_ids = set.intersection(*(set(datasets[node].case_ids) for node in datasets))
    if not all_case_ids:
        raise ValueError("No common case_ids found across all input nodes!")
    all_case_ids = sorted(list(all_case_ids))

    # K折交叉验证
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_case_ids)):
        print(f"Fold {fold + 1}")

        train_case_ids = [all_case_ids[idx] for idx in train_ids]
        val_case_ids = [all_case_ids[idx] for idx in val_ids]
        split_info = {
            "fold": fold + 1, "train_case_ids": train_case_ids, "val_case_ids": val_case_ids,
            "train_count": len(train_case_ids), "val_count": len(val_case_ids),
        }
        split_save_path = os.path.join(save_dir, f"fold_{fold + 1}_split.json")
        with open(split_save_path, "w") as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split saved to {split_save_path}")

        dataloaders_train = {}
        dataloaders_val = {}
        for node in datasets:
            node_case_ids = datasets[node].case_ids
            train_indices = [node_case_ids.index(case_id) for case_id in train_case_ids if case_id in node_case_ids]
            val_indices = [node_case_ids.index(case_id) for case_id in val_case_ids if case_id in node_case_ids]

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)
            dataloaders_train[node] = DataLoader(datasets[node], batch_size=batch_size, sampler=train_subsampler, num_workers=0)
            dataloaders_val[node] = DataLoader(datasets[node], batch_size=batch_size, sampler=val_subsampler, num_workers=0)

        # 模型、优化器、调度器
        model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, T_max=num_epochs, eta_min=1e-6)

        # 早停
        best_val_loss = float("inf")
        epochs_no_improve = 0
        log = {"fold": fold + 1, "epochs": []}

        for epoch in range(num_epochs):
            train_loss, train_task_losses, train_metrics = train(
                model, dataloaders_train, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping
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
