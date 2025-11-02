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
from node_toolkit.node_dataset import NodeDataset, MinMaxNormalize, RandomRotate, RandomShift, RandomFlip, RandomZoom, OneHot, OrderedSampler, worker_init_fn
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
    base_data_dir = r"/data/menghaoding/thu_xwh/TrainNiigzCsvData/Tr_fold1/"
    train_data_dir = os.path.join(base_data_dir, "train")
    val_data_dir = os.path.join(base_data_dir, "val")
    save_dir = r"/data/menghaoding/thu_xwh/uniconnnet1_fold1"
    load_dir = r"/data/menghaoding/thu_xwh/uniconnnet0_fold1"
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    batch_size = 8
    num_dimensions = 3
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-5
    validation_interval = 1
    patience = 100
    num_workers = 16
    scheduler_type = "poly"  # Options: "cosine", "poly", or "reduce_plateau"

    # Save HDNet configurations
    save_hdnet = {
        "unet1": os.path.join(save_dir, "unet1.pth"),
        "unet2": os.path.join(save_dir, "unet2.pth"),
        "classifier_n5": os.path.join(save_dir, "classifier_n5.pth"),
        "classifier_n6": os.path.join(save_dir, "classifier_n6.pth"),
        "classifier_n7": os.path.join(save_dir, "classifier_n7.pth"),
        "classifier_n8": os.path.join(save_dir, "classifier_n8.pth"),
        "classifier_n9": os.path.join(save_dir, "classifier_n9.pth"),
        "label_net": os.path.join(save_dir, "label_net.pth"),
        "uniconnnet_n5": os.path.join(save_dir, "uniconnnet_n5.pth"),
        "uniconnnet_n6": os.path.join(save_dir, "uniconnnet_n6.pth"),
        "uniconnnet_n7": os.path.join(save_dir, "uniconnnet_n7.pth"),
        "uniconnnet_n8": os.path.join(save_dir, "uniconnnet_n8.pth"),
        "uniconnnet_n9": os.path.join(save_dir, "uniconnnet_n9.pth"),
    }

    load_hdnet = {
        "unet1": os.path.join(load_dir, "unet1.pth"),
    }

    # UNet2 configuration (5-channel input, output to 5 1-channel nodes, with dropout 0.1 to 0.5 to 0.1)
    node_configs_unet2 = {
        "n0": (1, 64, 64, 64),    # Input
        "n1": (1, 64, 64, 64),
        "n2": (1, 64, 64, 64),
        "n3": (1, 64, 64, 64),
        "n4": (1, 64, 64, 64),
        "n5": (32, 64, 64, 64),   # Encoder features
        "n6": (64, 32, 32, 32),
        "n7": (128, 16, 16, 16),
        "n8": (256, 8, 8, 8),
        "n9": (512, 4, 4, 4),     # Bottleneck
        "n10": (256, 8, 8, 8),    # Decoder features
        "n11": (128, 16, 16, 16),
        "n12": (64, 32, 32, 32),
        "n13": (32, 64, 64, 64),
        "n14": (32, 64, 64, 64),  # Intermediate output
    }
    hyperedge_configs_unet2 = {
        # Encoder path
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
                "dropout": 0.1
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
                "dropout": 0.2
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
                "dropout": 0.3
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
                "dropout": 0.4
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
                "dropout": 0.5
            }
        },
        # Decoder path
        "e6": {
            "src_nodes": ["n9"],
            "dst_nodes": ["n10"],
            "params": {
                "convs": [torch.Size([256, 512, 1, 1, 1]), torch.Size([256, 256, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.5
            }
        },
        "e7": {
            "src_nodes": ["n8", "n10"],
            "dst_nodes": ["n11"],
            "params": {
                "convs": [torch.Size([128, 512, 1, 1, 1]), torch.Size([128, 128, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "dropout": 0.4
            }
        },
        "e8": {
            "src_nodes": ["n7", "n11"],
            "dst_nodes": ["n12"],
            "params": {
                "convs": [torch.Size([64, 256, 1, 1, 1]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "linear",
                "dropout": 0.3
            }
        },
        "e9": {
            "src_nodes": ["n6", "n12"],
            "dst_nodes": ["n13"],
            "params": {
                "convs": [torch.Size([32, 128, 1, 1, 1]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.2
            }
        },
        "e10": {
            "src_nodes": ["n5", "n13"],
            "dst_nodes": ["n14"],
            "params": {
                "convs": [torch.Size([32, 64, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.1
            }
        },
    }

    # UNet1 configuration (5 1-channel inputs, no dropout, pretrained)
    node_configs_unet1 = {
        "n0": (1, 64, 64, 64),    # Input from unet2 n15
        "n1": (1, 64, 64, 64),    # Input from unet2 n16
        "n2": (1, 64, 64, 64),    # Input from unet2 n17
        "n3": (1, 64, 64, 64),    # Input from unet2 n18
        "n4": (1, 64, 64, 64),    # Input from unet2 n19
        "n5": (32, 64, 64, 64),
        "n6": (64, 32, 32, 32),
        "n7": (128, 16, 16, 16),
        "n8": (256, 8, 8, 8),
        "n9": (512, 4, 4, 4),     # Bottleneck
        "n10": (256, 8, 8, 8),    # Decoder features
        "n11": (128, 16, 16, 16),
        "n12": (64, 32, 32, 32),
        "n13": (32, 64, 64, 64),
        "n14": (32, 64, 64, 64),  # Output for classifiers
    }
    hyperedge_configs_unet1 = {
        # Encoder path
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
                "dropout": 0.1
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
                "dropout": 0.2
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
                "dropout": 0.3
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
                "dropout": 0.4
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
                "dropout": 0.5
            }
        },
        # Decoder path
        "e6": {
            "src_nodes": ["n9"],
            "dst_nodes": ["n10"],
            "params": {
                "convs": [torch.Size([256, 512, 1, 1, 1]), torch.Size([256, 256, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.5
            }
        },
        "e7": {
            "src_nodes": ["n8", "n10"],
            "dst_nodes": ["n11"],
            "params": {
                "convs": [torch.Size([128, 512, 1, 1, 1]), torch.Size([128, 128, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "dropout": 0.4
            }
        },
        "e8": {
            "src_nodes": ["n7", "n11"],
            "dst_nodes": ["n12"],
            "params": {
                "convs": [torch.Size([64, 256, 1, 1, 1]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "linear",
                "dropout": 0.3
            }
        },
        "e9": {
            "src_nodes": ["n6", "n12"],
            "dst_nodes": ["n13"],
            "params": {
                "convs": [torch.Size([32, 128, 1, 1, 1]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.2
            }
        },
        "e10": {
            "src_nodes": ["n5", "n13"],
            "dst_nodes": ["n14"],
            "params": {
                "convs": [torch.Size([32, 64, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.1
            }
        },
    }

    # uniconnnet_n5 (8x8x8, 256 channels, single hyperedge from unet2 to unet1)
    node_configs_uniconnnet_n5 = {
        "n0": (256, 8, 8, 8),  # From unet2 n8
        "n1": (256, 8, 8, 8),  # From unet2 n10
        "n2": (256, 8, 8, 8),  # To unet1 n8
        "n3": (256, 8, 8, 8),  # To unet1 n10
    }
    hyperedge_configs_uniconnnet_n5 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3"],
            "params": {
                "convs": [torch.Size([512, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # uniconnnet_n6 (16x16x16, 128 channels)
    node_configs_uniconnnet_n6 = {
        "n0": (128, 16, 16, 16),  # From unet2 n7
        "n1": (128, 16, 16, 16),  # From unet2 n11
        "n2": (128, 16, 16, 16),  # To unet1 n7
        "n3": (128, 16, 16, 16),  # To unet1 n11
    }
    hyperedge_configs_uniconnnet_n6 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3"],
            "params": {
                "convs": [torch.Size([256, 256, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # uniconnnet_n7 (32x32x32, 64 channels)
    node_configs_uniconnnet_n7 = {
        "n0": (64, 32, 32, 32),  # From unet2 n6
        "n1": (64, 32, 32, 32),  # From unet2 n12
        "n2": (64, 32, 32, 32),  # To unet1 n6
        "n3": (64, 32, 32, 32),  # To unet1 n12
    }
    hyperedge_configs_uniconnnet_n7 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3"],
            "params": {
                "convs": [torch.Size([128, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (32, 32, 32),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # uniconnnet_n8 (64x64x64, 32 channels)
    node_configs_uniconnnet_n8 = {
        "n0": (32, 64, 64, 64),  # From unet2 n5
        "n1": (32, 64, 64, 64),  # From unet2 n13
        "n2": (32, 64, 64, 64),  # To unet1 n5
        "n3": (32, 64, 64, 64),  # To unet1 n13
    }
    hyperedge_configs_uniconnnet_n8 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3"],
            "params": {
                "convs": [torch.Size([64, 64, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # uniconnnet_n9 (64x64x64, connecting unet2 n0-n4, n14 to unet1 n0-n4, n14)
    node_configs_uniconnnet_n9 = {
        "n0": (1, 64, 64, 64),   # From unet2 n0
        "n1": (1, 64, 64, 64),   # From unet2 n1
        "n2": (1, 64, 64, 64),   # From unet2 n2
        "n3": (1, 64, 64, 64),   # From unet2 n3
        "n4": (1, 64, 64, 64),   # From unet2 n4
        "n5": (32, 64, 64, 64),  # From unet2 n14
        "n6": (1, 64, 64, 64),   # To unet1 n0
        "n7": (1, 64, 64, 64),   # To unet1 n1
        "n8": (1, 64, 64, 64),   # To unet1 n2
        "n9": (1, 64, 64, 64),   # To unet1 n3
        "n10": (1, 64, 64, 64),  # To unet1 n4
        "n11": (32, 64, 64, 64), # To unet1 n14
    }
    hyperedge_configs_uniconnnet_n9 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4", "n5"],
            "dst_nodes": ["n6", "n7", "n8", "n9", "n10", "n11"],
            "params": {
                "convs": [torch.Size([37, 37, 1, 1, 1])],  # 1+1+1+1+1+32=37
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # Classifier for n5 (256 channels, 8x8x8)
    node_configs_classifier_n5 = {
        "n0": (256, 8, 8, 8),  # Input from unet2 n10
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
        "n0": (128, 16, 16, 16),  # Input from unet2 n11
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
        "n0": (64, 32, 32, 32),  # Input from unet2 n12
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
        "n0": (32, 64, 64, 64),  # Input from unet2 n13
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
        "n0": (32, 64, 64, 64),  # Input from unet2 n14
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
        ("n100", "unet1", "n0"),
        ("n101", "unet1", "n1"),
        ("n102", "unet1", "n2"),
        ("n103", "unet1", "n3"),
        ("n104", "unet1", "n4"),
        ("n100", "uniconnnet_n9", "n0"),
        ("n101", "uniconnnet_n9", "n1"),
        ("n102", "uniconnnet_n9", "n2"),
        ("n103", "uniconnnet_n9", "n3"),
        ("n104", "uniconnnet_n9", "n4"),
        ("n105", "unet1", "n5"),
        ("n105", "uniconnnet_n8", "n0"),
        ("n106", "unet1", "n6"),
        ("n106", "uniconnnet_n7", "n0"),
        ("n107", "unet1", "n7"),
        ("n107", "uniconnnet_n6", "n0"),
        ("n108", "unet1", "n8"),
        ("n108", "uniconnnet_n5", "n0"),
        ("n109", "unet1", "n9"),
        ("n110", "unet1", "n10"),
        ("n110", "uniconnnet_n5", "n1"),
        ("n111", "unet1", "n11"),
        ("n111", "uniconnnet_n6", "n1"),
        ("n112", "unet1", "n12"),
        ("n112", "uniconnnet_n7", "n1"),
        ("n113", "unet1", "n13"),
        ("n113", "uniconnnet_n8", "n1"),
        ("n114", "unet1", "n14"),
        ("n114", "uniconnnet_n9", "n5"),
        ("n115", "unet2", "n0"),
        ("n115", "uniconnnet_n9", "n6"),
        ("n116", "unet2", "n1"),
        ("n116", "uniconnnet_n9", "n7"),
        ("n117", "unet2", "n2"),
        ("n117", "uniconnnet_n9", "n8"),
        ("n118", "unet2", "n3"),
        ("n118", "uniconnnet_n9", "n9"),
        ("n119", "unet2", "n4"),
        ("n119", "uniconnnet_n9", "n10"),
        ("n120", "unet2", "n5"),
        ("n120", "uniconnnet_n8", "n2"),
        ("n121", "unet2", "n6"),
        ("n121", "uniconnnet_n7", "n2"),
        ("n122", "unet2", "n7"),
        ("n122", "uniconnnet_n6", "n2"),
        ("n123", "unet2", "n8"),
        ("n123", "uniconnnet_n5", "n2"),
        ("n124", "unet2", "n9"),
        ("n125", "unet2", "n10"),
        ("n125", "uniconnnet_n5", "n3"),
        ("n125", "classifier_n5", "n0"),
        ("n126", "unet2", "n11"),
        ("n126", "uniconnnet_n6", "n3"),
        ("n126", "classifier_n6", "n0"),
        ("n127", "unet2", "n12"),
        ("n127", "uniconnnet_n7", "n3"),
        ("n127", "classifier_n7", "n0"),
        ("n128", "unet2", "n13"),
        ("n128", "uniconnnet_n8", "n3"),
        ("n128", "classifier_n8", "n0"),
        ("n129", "unet2", "n14"),
        ("n129", "uniconnnet_n9", "n11"),
        ("n129", "classifier_n9", "n0"),
        ("n130", "classifier_n5", "n1"),
        ("n131", "classifier_n6", "n1"),
        ("n132", "classifier_n7", "n1"),
        ("n133", "classifier_n8", "n1"),
        ("n134", "classifier_n9", "n1"),
        ("n135", "label_net", "n0"),
    ]

    # Sub-network configurations
    sub_networks_configs = {
        "unet1": (node_configs_unet1, hyperedge_configs_unet1),
        "unet2": (node_configs_unet2, hyperedge_configs_unet2),
        "classifier_n5": (node_configs_classifier_n5, hyperedge_configs_classifier_n5),
        "classifier_n6": (node_configs_classifier_n6, hyperedge_configs_classifier_n6),
        "classifier_n7": (node_configs_classifier_n7, hyperedge_configs_classifier_n7),
        "classifier_n8": (node_configs_classifier_n8, hyperedge_configs_classifier_n8),
        "classifier_n9": (node_configs_classifier_n9, hyperedge_configs_classifier_n9),
        "label_net": (node_configs_label, hyperedge_configs_label),
        "uniconnnet_n5": (node_configs_uniconnnet_n5, hyperedge_configs_uniconnnet_n5),
        "uniconnnet_n6": (node_configs_uniconnnet_n6, hyperedge_configs_uniconnnet_n6),
        "uniconnnet_n7": (node_configs_uniconnnet_n7, hyperedge_configs_uniconnnet_n7),
        "uniconnnet_n8": (node_configs_uniconnnet_n8, hyperedge_configs_uniconnnet_n8),
        "uniconnnet_n9": (node_configs_uniconnnet_n9, hyperedge_configs_uniconnnet_n9),
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
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n135"]
    out_nodes = ["n130", "n131", "n132", "n133", "n134", "n135"]

    # Node file mapping
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n135", "0006.csv"),
    ]

    # Instantiate transformations
    random_rotate = RandomRotate(max_angle=15)
    random_shift = RandomShift(max_shift=10)
    random_zoom = RandomZoom(zoom_range=(0.9, 1.1))
    random_flip = RandomFlip(axes=[0,1,2])
    min_max_normalize = MinMaxNormalize()
    one_hot4 = OneHot(num_classes=4)

    # Node transformation configuration for train and validate
    node_transforms = {
        "train": {
            "n100": [random_rotate, random_shift, random_zoom, random_flip],
            "n101": [random_rotate, random_shift, random_zoom, random_flip],
            "n102": [random_rotate, random_shift, random_zoom, random_flip],
            "n103": [random_rotate, random_shift, random_zoom, random_flip],
            "n104": [random_rotate, random_shift, random_zoom, random_flip],
            "n135": [one_hot4],
        },
        "validate": {
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
            "n135": [one_hot4],
        }
    }

    invs = [1/(492+112), 1/(912+226), 1/(1770+530), 1/(2066+460)]
    invs_sum = sum(invs)

    # Task configuration with deep supervision
    task_configs = {
        "type_cls_n5": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n130", "target_node": "n135", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n130", "target_node": "n135", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n130", "target_node": "n135", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n130", "target_node": "n135", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n130", "target_node": "n135", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n130", "target_node": "n135", "params": {}}
            ]
        },
        "type_cls_n6": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n131", "target_node": "n135", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n131", "target_node": "n135", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n131", "target_node": "n135", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n131", "target_node": "n135", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n131", "target_node": "n135", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n131", "target_node": "n135", "params": {}}
            ]
        },
        "type_cls_n7": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n132", "target_node": "n135", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n132", "target_node": "n135", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n132", "target_node": "n135", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n132", "target_node": "n135", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n132", "target_node": "n135", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n132", "target_node": "n135", "params": {}}
            ]
        },
        "type_cls_n8": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n133", "target_node": "n135", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n133", "target_node": "n135", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n133", "target_node": "n135", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n133", "target_node": "n135", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n133", "target_node": "n135", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n133", "target_node": "n135", "params": {}}
            ]
        },
        "type_cls_n9": {
            "loss": [
                {"fn": node_focal_loss, "origin_node": "n134", "target_node": "n135", "weight": 1.0, "params": {"alpha": [x / invs_sum for x in invs], "gamma": 0}},
            ],
            "metric": [
                {"fn": node_recall_metric, "origin_node": "n134", "target_node": "n135", "params": {}},
                {"fn": node_precision_metric, "origin_node": "n134", "target_node": "n135", "params": {}},
                {"fn": node_f1_metric, "origin_node": "n134", "target_node": "n135", "params": {}},
                {"fn": node_accuracy_metric, "origin_node": "n134", "target_node": "n135", "params": {}},
                {"fn": node_specificity_metric, "origin_node": "n134", "target_node": "n135", "params": {}}
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
    newtrained_params_classifier = []
    pretrained_params_unet1 = []
    newtrained_params_unet2 = []
    newtrained_params_uniconnnet = []
    for name, param in model.named_parameters():
        if name.startswith('sub_networks.unet1'):
            pretrained_params_unet1.append(param)
        elif name.startswith('sub_networks.classifier_n'):
            newtrained_params_classifier.append(param)
        elif name.startswith('sub_networks.unet2'):
            newtrained_params_unet2.append(param)
        else:
            newtrained_params_uniconnnet.append(param)

    optimizer = optim.Adam([
        {'params': pretrained_params_unet1, 'lr': learning_rate, 'weight_decay': weight_decay / 2},
        {'params': newtrained_params_classifier, 'lr': learning_rate, 'weight_decay': weight_decay},
        {'params': newtrained_params_unet2, 'lr': learning_rate, 'weight_decay': weight_decay / 2},
        {'params': newtrained_params_uniconnnet, 'lr': learning_rate, 'weight_decay': weight_decay}
    ])

    # Select scheduler
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        logger.info("Using CosineAnnealingLR scheduler")
    elif scheduler_type == "poly":
        scheduler = PolynomialLR(optimizer, max_epochs=num_epochs, power=0.9, eta_min=0)
        logger.info("Using PolynomialLR scheduler")
    elif scheduler_type == "reduce_plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=20, eta_min=1e-6, verbose=True)
        logger.info("Using ReduceLROnPlateau scheduler")
    else:
        raise ValueError(f"Invalid scheduler_type: {scheduler_type}. Choose 'cosine', 'poly', or 'reduce_plateau'.")

    # Early stopping and logging
    epochs_no_improve = 0
    save_mode = "max"
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
            f1_scores = []
            for task in task_configs:
                if "node_f1_metric" in val_task_metrics[task]:
                    f1_score = val_task_metrics[task]["node_f1_metric"]["value"]["avg"]
                    if not np.isnan(f1_score):
                        f1_scores.append(f1_score)
            save_criterion = np.max(f1_scores) if f1_scores else 0.0

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
    device_id = 2  # Index of the third GPU (starting from 0)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    main()

