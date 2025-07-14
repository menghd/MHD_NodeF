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
    Main function to run the training pipeline with deep supervision and unified label handling.
    运行带深度监督和统一标签处理的训练流水线的主函数。
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and save paths
    base_data_dir = r"C:\Users\souray\Desktop\Tr_independent"
    train_data_dir = os.path.join(base_data_dir, "train")
    val_data_dir = os.path.join(base_data_dir, "val")
    save_dir = r"C:\Users\souray\Desktop\MHDNet_poly_unicom2"
    load_dir = r"/data/menghaoding/thu_xwh/MHDNet_poly_unicom1"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 2
    num_dimensions = 3
    num_epochs = 400
    learning_rate = 1e-3
    weight_decay = 1e-4
    validation_interval = 1
    patience = 100
    num_workers = 0
    scheduler_type = "poly"  # Options: "cosine", "poly", or "reduce_plateau"

    # Save and load HDNet configurations
    save_hdnet = {
        "encoder1": os.path.join(save_dir, "encoder1.pth"),
        "decoder1": os.path.join(save_dir, "decoder1.pth"),
        "encoder2": os.path.join(save_dir, "encoder2.pth"),
        "decoder2": os.path.join(save_dir, "decoder2.pth"),
        "encoder3": os.path.join(save_dir, "encoder3.pth"),
        "decoder3": os.path.join(save_dir, "decoder3.pth"),
        "classifier_n5": os.path.join(save_dir, "classifier_n5.pth"),
        "classifier_n6": os.path.join(save_dir, "classifier_n6.pth"),
        "classifier_n7": os.path.join(save_dir, "classifier_n7.pth"),
        "classifier_n8": os.path.join(save_dir, "classifier_n8.pth"),
        "classifier_n9": os.path.join(save_dir, "classifier_n9.pth"),
        "label_net": os.path.join(save_dir, "label_net.pth"),
        "unicom_n5_0": os.path.join(save_dir, "unicom_n5_0.pth"),
        "unicom_n5_1": os.path.join(save_dir, "unicom_n5_1.pth"),
        "unicom_n6_0": os.path.join(save_dir, "unicom_n6_0.pth"),
        "unicom_n6_1": os.path.join(save_dir, "unicom_n6_1.pth"),
        "unicom_n7_0": os.path.join(save_dir, "unicom_n7_0.pth"),
        "unicom_n7_1": os.path.join(save_dir, "unicom_n7_1.pth"),
        "unicom_n8_0": os.path.join(save_dir, "unicom_n8_0.pth"),
        "unicom_n8_1": os.path.join(save_dir, "unicom_n8_1.pth"),
        "unicom_n9_0": os.path.join(save_dir, "unicom_n9_0.pth"),
        "unicom_n9_1": os.path.join(save_dir, "unicom_n9_1.pth"),
    }

    load_hdnet = {
        # "encoder1": os.path.join(load_dir, "encoder1.pth"),
        # "decoder1": os.path.join(load_dir, "decoder1.pth"),
        # "encoder2": os.path.join(load_dir, "encoder2.pth"),
        # "decoder2": os.path.join(load_dir, "decoder2.pth"),
    }

    # Encoder1 configuration (downsampling path, same as original encoder)
    node_configs_encoder1 = {
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
    hyperedge_configs_encoder1 = {
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
                "reshape": (4, 4, 4),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Decoder1 configuration (upsampling path with skip connections)
    node_configs_decoder1 = {
        "n0": (512, 4, 4, 4),   # Input: bottleneck from Encoder1 n9
        "n1": (256, 8, 8, 8),   # Input: skip from Encoder1 n8
        "n2": (128, 16, 16, 16),  # Input: skip from Encoder1 n7
        "n3": (64, 32, 32, 32),   # Input: skip from Encoder1 n6
        "n4": (32, 64, 64, 64),   # Input: skip from Encoder1 n5
        "n5": (256, 8, 8, 8),   # Decoder features
        "n6": (128, 16, 16, 16),
        "n7": (64, 32, 32, 32),
        "n8": (32, 64, 64, 64),
        "n9": (32, 64, 64, 64),  # Intermediate output
        "n10": (32, 64, 64, 64), # Output to Encoder2, Unicom_n9_0, Unicom_n9_1
    }
    hyperedge_configs_decoder1 = {
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e6": {
            "src_nodes": ["n9"],
            "dst_nodes": ["n10"],
            "params": {
                "convs": [torch.Size([32, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Encoder2 configuration (downsampling path, input from Decoder1 n10)
    node_configs_encoder2 = {
        "n0": (32, 64, 64, 64),  # Input from Decoder1 n10
        "n5": (32, 64, 64, 64),
        "n6": (64, 32, 32, 32),
        "n7": (128, 16, 16, 16),
        "n8": (256, 8, 8, 8),
        "n9": (512, 4, 4, 4),   # Bottleneck
    }
    hyperedge_configs_encoder2 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [torch.Size([32, 32, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
                "reshape": (4, 4, 4),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Decoder2 configuration (upsampling path with skip connections)
    node_configs_decoder2 = {
        "n0": (512, 4, 4, 4),   # Input: bottleneck from Encoder2 n9
        "n1": (256, 8, 8, 8),   # Input: skip from Encoder2 n8
        "n2": (128, 16, 16, 16),  # Input: skip from Encoder2 n7
        "n3": (64, 32, 32, 32),   # Input: skip from Encoder2 n6
        "n4": (32, 64, 64, 64),   # Input: skip from Encoder2 n5
        "n5": (256, 8, 8, 8),   # Decoder features
        "n6": (128, 16, 16, 16),
        "n7": (64, 32, 32, 32),
        "n8": (32, 64, 64, 64),
        "n9": (32, 64, 64, 64),  # Intermediate output
        "n10": (32, 64, 64, 64), # Output to Encoder3, Unicom_n9_0, Unicom_n9_1
    }
    hyperedge_configs_decoder2 = {
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
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
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
        "e6": {
            "src_nodes": ["n9"],
            "dst_nodes": ["n10"],
            "params": {
                "convs": [torch.Size([32, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Encoder3 configuration (downsampling path, input from Decoder2 n10)
    node_configs_encoder3 = {
        "n0": (32, 64, 64, 64),  # Input from Decoder2 n10
        "n5": (32, 64, 64, 64),
        "n6": (64, 32, 32, 32),
        "n7": (128, 16, 16, 16),
        "n8": (256, 8, 8, 8),
        "n9": (512, 4, 4, 4),   # Bottleneck
    }
    hyperedge_configs_encoder3 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [torch.Size([32, 32, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
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

    # Decoder3 configuration (upsampling path with skip connections)
    node_configs_decoder3 = {
        "n0": (512, 4, 4, 4),   # Input: bottleneck from Encoder3 n9
        "n1": (256, 8, 8, 8),   # Input: skip from Encoder3 n8
        "n2": (128, 16, 16, 16),  # Input: skip from Encoder3 n7
        "n3": (64, 32, 32, 32),   # Input: skip from Encoder3 n6
        "n4": (32, 64, 64, 64),   # Input: skip from Encoder3 n5
        "n5": (256, 8, 8, 8),   # Decoder features
        "n6": (128, 16, 16, 16),
        "n7": (64, 32, 32, 32),
        "n8": (32, 64, 64, 64),
        "n9": (32, 64, 64, 64),  # Intermediate output
    }
    hyperedge_configs_decoder3 = {
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

    # Unicom_n5_0 (8x8x8, 256 channels)
    node_configs_unicom_n5_0 = {
        "n0": (256, 8, 8, 8),  # From encoder1 n8
        "n1": (256, 8, 8, 8),  # From decoder1 n5
        "n2": (256, 8, 8, 8),  # To encoder2 n8
        "n3": (256, 8, 8, 8),  # To decoder2 n5
        "n4": (256, 8, 8, 8),  # To encoder3 n8
        "n5": (256, 8, 8, 8),  # To decoder3 n5
    }
    hyperedge_configs_unicom_n5_0 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3", "n4", "n5"],
            "params": {
                "convs": [torch.Size([1024, 512, 1, 1, 1])],  # Concatenate n0, n1
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (8, 8, 8),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n5_1 (8x8x8, 256 channels)
    node_configs_unicom_n5_1 = {
        "n0": (256, 8, 8, 8),  # From encoder1 n8
        "n1": (256, 8, 8, 8),  # From decoder1 n5
        "n2": (256, 8, 8, 8),  # From encoder2 n8
        "n3": (256, 8, 8, 8),  # From decoder2 n5
        "n4": (256, 8, 8, 8),  # To encoder3 n8
        "n5": (256, 8, 8, 8),  # To decoder3 n5
    }
    hyperedge_configs_unicom_n5_1 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([512, 1024, 1, 1, 1])],  # Concatenate n0, n1, n2, n3
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (8, 8, 8),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n6_0 (16x16x16, 128 channels)
    node_configs_unicom_n6_0 = {
        "n0": (128, 16, 16, 16),  # From encoder1 n7
        "n1": (128, 16, 16, 16),  # From decoder1 n6
        "n2": (128, 16, 16, 16),  # To encoder2 n7
        "n3": (128, 16, 16, 16),  # To decoder2 n6
        "n4": (128, 16, 16, 16),  # To encoder3 n7
        "n5": (128, 16, 16, 16),  # To decoder3 n6
    }
    hyperedge_configs_unicom_n6_0 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3", "n4", "n5"],
            "params": {
                "convs": [torch.Size([512, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (16, 16, 16),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n6_1 (16x16x16, 128 channels)
    node_configs_unicom_n6_1 = {
        "n0": (128, 16, 16, 16),  # From encoder1 n7
        "n1": (128, 16, 16, 16),  # From decoder1 n6
        "n2": (128, 16, 16, 16),  # From encoder2 n7
        "n3": (128, 16, 16, 16),  # From decoder2 n6
        "n4": (128, 16, 16, 16),  # To encoder3 n7
        "n5": (128, 16, 16, 16),  # To decoder3 n6
    }
    hyperedge_configs_unicom_n6_1 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([256, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (16, 16, 16),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n7_0 (32x32x32, 64 channels)
    node_configs_unicom_n7_0 = {
        "n0": (64, 32, 32, 32),  # From encoder1 n6
        "n1": (64, 32, 32, 32),  # From decoder1 n7
        "n2": (64, 32, 32, 32),  # To encoder2 n6
        "n3": (64, 32, 32, 32),  # To decoder2 n7
        "n4": (64, 32, 32, 32),  # To encoder3 n6
        "n5": (64, 32, 32, 32),  # To decoder3 n7
    }
    hyperedge_configs_unicom_n7_0 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3", "n4", "n5"],
            "params": {
                "convs": [torch.Size([256, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (32, 32, 32),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (32, 32, 32),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n7_1 (32x32x32, 64 channels)
    node_configs_unicom_n7_1 = {
        "n0": (64, 32, 32, 32),  # From encoder1 n6
        "n1": (64, 32, 32, 32),  # From decoder1 n7
        "n2": (64, 32, 32, 32),  # From encoder2 n6
        "n3": (64, 32, 32, 32),  # From decoder2 n7
        "n4": (64, 32, 32, 32),  # To encoder3 n6
        "n5": (64, 32, 32, 32),  # To decoder3 n7
    }
    hyperedge_configs_unicom_n7_1 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([128, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (32, 32, 32),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (32, 32, 32),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n8_0 (64x64x64, 32 channels)
    node_configs_unicom_n8_0 = {
        "n0": (32, 64, 64, 64),  # From encoder1 n5
        "n1": (32, 64, 64, 64),  # From decoder1 n8
        "n2": (32, 64, 64, 64),  # To encoder2 n5
        "n3": (32, 64, 64, 64),  # To decoder2 n8
        "n4": (32, 64, 64, 64),  # To encoder3 n5
        "n5": (32, 64, 64, 64),  # To decoder3 n8
    }
    hyperedge_configs_unicom_n8_0 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3", "n4", "n5"],
            "params": {
                "convs": [torch.Size([128, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n8_1 (64x64x64, 32 channels)
    node_configs_unicom_n8_1 = {
        "n0": (32, 64, 64, 64),  # From encoder1 n5
        "n1": (32, 64, 64, 64),  # From decoder1 n8
        "n2": (32, 64, 64, 64),  # From encoder2 n5
        "n3": (32, 64, 64, 64),  # From decoder2 n8
        "n4": (32, 64, 64, 64),  # To encoder3 n5
        "n5": (32, 64, 64, 64),  # To decoder3 n8
    }
    hyperedge_configs_unicom_n8_1 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([64, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n9_0 (64x64x64, 32 channels)
    node_configs_unicom_n9_0 = {
        "n0": (1, 64, 64, 64),   # From encoder1 n0
        "n1": (1, 64, 64, 64),   # From encoder1 n1
        "n2": (1, 64, 64, 64),   # From encoder1 n2
        "n3": (1, 64, 64, 64),   # From encoder1 n3
        "n4": (1, 64, 64, 64),   # From encoder1 n4
        "n5": (32, 64, 64, 64),  # From decoder1 n9
        "n6": (32, 64, 64, 64),  # To encoder2 n0
        "n7": (32, 64, 64, 64),  # To decoder2 n9
        "n8": (32, 64, 64, 64),  # To encoder3 n0
        "n9": (32, 64, 64, 64),  # To decoder3 n9
    }
    hyperedge_configs_unicom_n9_0 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4", "n5"],
            "dst_nodes": ["n6", "n7", "n8", "n9"],
            "params": {
                "convs": [torch.Size([128, 37, 1, 1, 1])],  # 1+1+1+1+1+32=37
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Unicom_n9_1 (64x64x64, 32 channels)
    node_configs_unicom_n9_1 = {
        "n0": (1, 64, 64, 64),   # From encoder1 n0
        "n1": (1, 64, 64, 64),   # From encoder1 n1
        "n2": (1, 64, 64, 64),   # From encoder1 n2
        "n3": (1, 64, 64, 64),   # From encoder1 n3
        "n4": (1, 64, 64, 64),   # From encoder1 n4
        "n5": (32, 64, 64, 64),  # From decoder1 n9
        "n6": (32, 64, 64, 64),  # From encoder2 n0
        "n7": (32, 64, 64, 64),  # From decoder2 n9
        "n8": (32, 64, 64, 64),  # To encoder3 n0
        "n9": (32, 64, 64, 64),  # To decoder3 n9
    }
    hyperedge_configs_unicom_n9_1 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"],
            "dst_nodes": ["n8", "n9"],
            "params": {
                "convs": [torch.Size([64, 101, 1, 1, 1])], 
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            }
        },
    }

    # Classifier for n5 (256 channels, 8x8x8)
    node_configs_classifier_n5 = {
        "n0": (256, 8, 8, 8),  # Input from Decoder3 n5
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

    # Classifier for n6 (128 channels, 16x16x16)
    node_configs_classifier_n6 = {
        "n0": (128, 16, 16, 16),  # Input from Decoder3 n6
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

    # Classifier for n7 (64 channels, 32x32x32)
    node_configs_classifier_n7 = {
        "n0": (64, 32, 32, 32),  # Input from Decoder3 n7
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

    # Classifier for n8 (32 channels, 64x64x64)
    node_configs_classifier_n8 = {
        "n0": (32, 64, 64, 64),  # Input from Decoder3 n8
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

    # Classifier for n9 (32 channels, 64x64x64)
    node_configs_classifier_n9 = {
        "n0": (32, 64, 64, 64),  # Input from Decoder3 n10
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

    # Label network configuration (single node)
    node_configs_label = {
        "n0": (4, 1, 1, 1),  # Unified one-hot encoded label
    }
    hyperedge_configs_label = {}  # No hyperedges needed

    # Global node mapping
    node_mapping = [
        ("n100", "encoder1", "n0"),
        ("n101", "encoder1", "n1"),
        ("n102", "encoder1", "n2"),
        ("n103", "encoder1", "n3"),
        ("n104", "encoder1", "n4"),
        ("n100", "unicom_n9_0", "n0"),
        ("n101", "unicom_n9_0", "n1"),
        ("n102", "unicom_n9_0", "n2"),
        ("n103", "unicom_n9_0", "n3"),
        ("n104", "unicom_n9_0", "n4"),
        ("n100", "unicom_n9_1", "n0"),
        ("n101", "unicom_n9_1", "n1"),
        ("n102", "unicom_n9_1", "n2"),
        ("n103", "unicom_n9_1", "n3"),
        ("n104", "unicom_n9_1", "n4"),
        ("n105", "encoder1", "n5"),
        ("n105", "decoder1", "n4"),       
        ("n105", "unicom_n8_0", "n0"),   
        ("n105", "unicom_n8_1", "n0"),  
        ("n106", "encoder1", "n6"),
        ("n106", "decoder1", "n3"),       
        ("n106", "unicom_n7_0", "n0"),   
        ("n106", "unicom_n7_1", "n0"),    
        ("n107", "encoder1", "n7"),
        ("n107", "decoder1", "n2"),       
        ("n107", "unicom_n6_0", "n0"),   
        ("n107", "unicom_n6_1", "n0"),  
        ("n108", "encoder1", "n8"),
        ("n108", "decoder1", "n1"),       
        ("n108", "unicom_n5_0", "n0"),   
        ("n108", "unicom_n5_1", "n0"),  
        ("n109", "encoder1", "n9"),
        ("n109", "decoder1", "n0"),       
        ("n110", "decoder1", "n5"),       
        ("n110", "unicom_n5_0", "n1"),   
        ("n110", "unicom_n5_1", "n1"), 
        ("n111", "decoder1", "n6"),       
        ("n111", "unicom_n6_0", "n1"),   
        ("n111", "unicom_n6_1", "n1"), 
        ("n112", "decoder1", "n7"),       
        ("n112", "unicom_n7_0", "n1"),   
        ("n112", "unicom_n7_1", "n1"), 
        ("n113", "decoder1", "n8"),       
        ("n113", "unicom_n8_0", "n1"),   
        ("n113", "unicom_n8_1", "n1"), 
        ("n114", "decoder1", "n9"),       
        ("n114", "unicom_n9_0", "n5"),   
        ("n114", "unicom_n9_1", "n5"), 
        ("n115", "encoder2", "n0"),    
        ("n115", "decoder1", "n10"),       
        ("n115", "unicom_n9_0", "n6"),   
        ("n115", "unicom_n9_1", "n6"), 
        ("n116", "encoder2", "n5"),       
        ("n116", "decoder2", "n4"),     
        ("n116", "unicom_n8_0", "n2"),   
        ("n116", "unicom_n8_1", "n2"), 
        ("n117", "encoder2", "n6"),       
        ("n117", "decoder2", "n3"),     
        ("n117", "unicom_n7_0", "n2"),   
        ("n117", "unicom_n7_1", "n2"), 
        ("n118", "encoder2", "n7"),       
        ("n118", "decoder2", "n2"),     
        ("n118", "unicom_n6_0", "n2"),   
        ("n118", "unicom_n6_1", "n2"), 
        ("n119", "encoder2", "n8"),       
        ("n119", "decoder2", "n1"),     
        ("n119", "unicom_n5_0", "n2"),   
        ("n119", "unicom_n5_1", "n2"), 
        ("n120", "encoder2", "n9"),       
        ("n120", "decoder2", "n0"),  
        ("n121", "decoder2", "n5"),       
        ("n121", "unicom_n5_0", "n3"),   
        ("n121", "unicom_n5_1", "n3"), 
        ("n122", "decoder2", "n6"),       
        ("n122", "unicom_n6_0", "n3"),   
        ("n122", "unicom_n6_1", "n3"), 
        ("n123", "decoder2", "n7"),       
        ("n123", "unicom_n7_0", "n3"),   
        ("n123", "unicom_n7_1", "n3"), 
        ("n124", "decoder2", "n8"),       
        ("n124", "unicom_n8_0", "n3"),   
        ("n124", "unicom_n8_1", "n3"), 
        ("n125", "decoder2", "n9"),       
        ("n125", "unicom_n9_0", "n7"),   
        ("n125", "unicom_n9_1", "n7"), 
        ("n126", "encoder3", "n0"),    
        ("n126", "decoder2", "n10"),       
        ("n126", "unicom_n9_0", "n8"),   
        ("n126", "unicom_n9_1", "n8"), 
        ("n127", "encoder3", "n5"),       
        ("n127", "decoder3", "n4"),     
        ("n127", "unicom_n8_0", "n4"),   
        ("n127", "unicom_n8_1", "n4"), 
        ("n128", "encoder3", "n6"),       
        ("n128", "decoder3", "n3"),     
        ("n128", "unicom_n7_0", "n4"),   
        ("n128", "unicom_n7_1", "n4"), 
        ("n129", "encoder3", "n7"),       
        ("n129", "decoder3", "n2"),     
        ("n129", "unicom_n6_0", "n4"),   
        ("n129", "unicom_n6_1", "n4"), 
        ("n130", "encoder3", "n8"),       
        ("n130", "decoder3", "n1"),     
        ("n130", "unicom_n5_0", "n4"),   
        ("n130", "unicom_n5_1", "n4"), 
        ("n131", "encoder3", "n9"),       
        ("n131", "decoder3", "n0"),  
        ("n132", "decoder3", "n5"),       
        ("n132", "unicom_n5_0", "n5"),   
        ("n132", "unicom_n5_1", "n5"), 
        ("n132", "classifier_n5", "n0"), 
        ("n133", "decoder3", "n6"),       
        ("n133", "unicom_n6_0", "n5"),   
        ("n133", "unicom_n6_1", "n5"), 
        ("n133", "classifier_n6", "n0"), 
        ("n134", "decoder3", "n7"),       
        ("n134", "unicom_n7_0", "n5"),   
        ("n134", "unicom_n7_1", "n5"), 
        ("n134", "classifier_n7", "n0"), 
        ("n135", "decoder3", "n8"),       
        ("n135", "unicom_n8_0", "n5"),   
        ("n135", "unicom_n8_1", "n5"), 
        ("n135", "classifier_n8", "n0"), 
        ("n136", "decoder3", "n9"),       
        ("n136", "unicom_n9_0", "n9"),   
        ("n136", "unicom_n9_1", "n9"), 
        ("n136", "classifier_n9", "n0"), 
        ("n137", "classifier_n5", "n1"), 
        ("n138", "classifier_n6", "n1"),
        ("n139", "classifier_n7", "n1"),
        ("n140", "classifier_n8", "n1"),
        ("n141", "classifier_n9", "n1"),
        ("n142", "label_net", "n0"),    
    ]

    # Sub-network configurations
    sub_networks_configs = {
        "encoder1": (node_configs_encoder1, hyperedge_configs_encoder1),
        "decoder1": (node_configs_decoder1, hyperedge_configs_decoder1),
        "encoder2": (node_configs_encoder2, hyperedge_configs_encoder2),
        "decoder2": (node_configs_decoder2, hyperedge_configs_decoder2),
        "encoder3": (node_configs_encoder3, hyperedge_configs_encoder3),
        "decoder3": (node_configs_decoder3, hyperedge_configs_decoder3),
        "classifier_n5": (node_configs_classifier_n5, hyperedge_configs_classifier_n5),
        "classifier_n6": (node_configs_classifier_n6, hyperedge_configs_classifier_n6),
        "classifier_n7": (node_configs_classifier_n7, hyperedge_configs_classifier_n7),
        "classifier_n8": (node_configs_classifier_n8, hyperedge_configs_classifier_n8),
        "classifier_n9": (node_configs_classifier_n9, hyperedge_configs_classifier_n9),
        "label_net": (node_configs_label, hyperedge_configs_label),
        "unicom_n5_0": (node_configs_unicom_n5_0, hyperedge_configs_unicom_n5_0),
        "unicom_n5_1": (node_configs_unicom_n5_1, hyperedge_configs_unicom_n5_1),
        "unicom_n6_0": (node_configs_unicom_n6_0, hyperedge_configs_unicom_n6_0),
        "unicom_n6_1": (node_configs_unicom_n6_1, hyperedge_configs_unicom_n6_1),
        "unicom_n7_0": (node_configs_unicom_n7_0, hyperedge_configs_unicom_n7_0),
        "unicom_n7_1": (node_configs_unicom_n7_1, hyperedge_configs_unicom_n7_1),
        "unicom_n8_0": (node_configs_unicom_n8_0, hyperedge_configs_unicom_n8_0),
        "unicom_n8_1": (node_configs_unicom_n8_1, hyperedge_configs_unicom_n8_1),
        "unicom_n9_0": (node_configs_unicom_n9_0, hyperedge_configs_unicom_n9_0),
        "unicom_n9_1": (node_configs_unicom_n9_1, hyperedge_configs_unicom_n9_1),
    }

    # Instantiate sub-networks
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, num_dimensions)
        for name, (node_configs, hyperedge_configs) in sub_networks_configs.items()
    }

    # Load pretrained weights for specified HDNets
    for net_name, weight_path in load_hdnet.items():
        if net_name in sub_networks and os.path.exists(weight_path):
            sub_networks[net_name].load_state_dict(torch.load(weight_path))
            logger.info(f"Loaded pretrained weights for {net_name} from {weight_path}")
        else:
            logger.warning(f"Could not load weights for {net_name}: {weight_path} does not exist")

    # Global input and output nodes
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n142"]
    out_nodes = ["n137", "n138", "n139", "n140", "n141", "n142"]

    # Node file mapping
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n142", "0006.csv"),
    ]

    # Instantiate transformations
    random_rotate = RandomRotate(max_angle=15)
    random_shift = RandomShift(max_shift=10)
    random_zoom = RandomZoom(zoom_range=(0.9, 1.1))
    min_max_normalize = MinMaxNormalize()
    one_hot4 = OneHot(num_classes=4)

    # Node transformation configuration for train and validate
    node_transforms = {
        "train": {
            "n100": [random_rotate, random_shift, random_zoom],
            "n101": [random_rotate, random_shift, random_zoom],
            "n102": [random_rotate, random_shift, random_zoom],
            "n103": [random_rotate, random_shift, random_zoom],
            "n104": [random_rotate, random_shift, random_zoom],
            "n142": [one_hot4],
        },
        "validate": {
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
            "n142": [one_hot4],
        }
    }

    invs = [1/(260+100), 1/(703+140), 1/(1596+388), 1/(1521+388)]
    invs_sum = sum(invs)

    # Task configuration with deep supervision
    task_configs = {
        "type_cls_n5": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n137", "target_node": "n142", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n137", "target_node": "n142", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n137", "target_node": "n142", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n137", "target_node": "n142", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n137", "target_node": "n142", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n137", "target_node": "n142", "params": {}}
            ]
        },
        "type_cls_n6": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n138", "target_node": "n142", "weight": 0.5, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n138", "target_node": "n142", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n138", "target_node": "n142", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n138", "target_node": "n142", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n138", "target_node": "n142", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n138", "target_node": "n142", "params": {}}
            ]
        },
        "type_cls_n7": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n139", "target_node": "n142", "weight": 0.25, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n139", "target_node": "n142", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n139", "target_node": "n142", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n139", "target_node": "n142", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n139", "target_node": "n142", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n139", "target_node": "n142", "params": {}}
            ]
        },
        "type_cls_n8": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n140", "target_node": "n142", "weight": 0.125, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n140", "target_node": "n142", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n140", "target_node": "n142", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n140", "target_node": "n142", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n140", "target_node": "n142", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n140", "target_node": "n142", "params": {}}
            ]
        },
        "type_cls_n9": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n141", "target_node": "n142", "weight": 0.0625, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n141", "target_node": "n142", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n141", "target_node": "n142", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n141", "target_node": "n142", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n141", "target_node": "n142", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n141", "target_node": "n142", "params": {}}
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
            raise ValueError(f"Case ID order inconsistent for node {node}")
        if datasets_val[node].case_ids != datasets_val[list(datasets_val.keys())[0]].case_ids:
            logger.error(f"Case ID order inconsistent for node {node} in validation")
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

    # Model
    onnx_save_path = os.path.join(save_dir, "model_config_initial.onnx")
    model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions, onnx_save_path=onnx_save_path).to(device)

    # Optimizer with different learning rates for pretrained and new parts
    unet1_params = []
    unet2_params = []
    new_params = []
    for name, param in model.named_parameters():
        if name.startswith('sub_networks.encoder1') or name.startswith('sub_networks.decoder1'):
            unet1_params.append(param)
        elif name.startswith('sub_networks.encoder2') or name.startswith('sub_networks.decoder2'):
            unet2_params.append(param)
        else:
            new_params.append(param)

    optimizer = optim.Adam([
        {'params': unet1_params, 'lr': learning_rate / 4, 'weight_decay': weight_decay / 4},  # Lower LR and weight decay for unet1 (encoder1, decoder1)
        {'params': unet2_params, 'lr': learning_rate / 2, 'weight_decay': weight_decay / 2},  # Lower LR and weight decay for unet2 (encoder2, decoder2)
        {'params': new_params, 'lr': learning_rate, 'weight_decay': weight_decay}  # Normal LR and weight decay for new parts
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
    epochs_no_improve = 0
    save_mode = "min"
    if save_mode == "min":
        best_save_criterion = float("inf")
    elif save_mode == "max":
        best_save_criterion = -float("inf")
    else:
        raise ValueError("save_mode must be 'min' or 'max'")
    log = {"epochs": []}

    for epoch in range(num_epochs):
        epoch_seed = seed + epoch
        np.random.seed(epoch_seed)
        batch_seeds = np.random.randint(0, 1000000, size=len(dataloaders_train[list(dataloaders_train.keys())[0]]))
        logger.info(f"Epoch {epoch + 1}: Generated {len(batch_seeds)} batch seeds")

        for batch_idx in range(len(dataloaders_train[list(dataloaders_train.keys())[0]])):
            batch_seed = int(batch_seeds[batch_idx])
            logger.debug(f"Batch {batch_idx}, Seed {batch_seed}")
            for node in datasets_train:
                datasets_train[node].set_batch_seed(batch_seed)
            for node in datasets_val:
                datasets_val[node].set_batch_seed(batch_seed)

        train_loss, train_task_losses, train_task_metrics = train(
            model, dataloaders_train, optimizer, task_configs, out_nodes, epoch, num_epochs, debug=True
        )

        # Get current learning rate
        current_lr = [group['lr'] for group in optimizer.param_groups]
        epoch_log = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_task_losses": train_task_losses,
            "train_task_metrics": train_task_metrics,
        }

        if (epoch + 1) % validation_interval == 0:
            val_loss, val_task_losses, val_task_metrics = validate(
                model, dataloaders_val, task_configs, out_nodes, epoch, num_epochs, debug=True
            )

            # Calculate save criterion
            save_criterion = val_loss

            epoch_log.update({
                "val_loss": val_loss,
                "val_task_losses": val_task_losses,
                "val_task_metrics": val_task_metrics,
                "save_criterion": save_criterion
            })

            # Save model weights based on save_criterion and save_mode
            should_save = (save_mode == "min" and save_criterion < best_save_criterion) or \
                          (save_mode == "max" and save_criterion > best_save_criterion)
            if should_save:
                best_save_criterion = save_criterion
                epochs_no_improve = 0
                # Save HDNet weights
                for net_name, save_path in save_hdnet.items():
                    if net_name in sub_networks:
                        torch.save(sub_networks[net_name].state_dict(), save_path)
                        logger.info(f"Saved {net_name} weights to {save_path}")
                logger.info(f"New best save criterion: {save_criterion:.4f}")
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
    # Specify using the third GPU
    device_id = 3  # Index of the third GPU (starting from 0)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")
    main()
