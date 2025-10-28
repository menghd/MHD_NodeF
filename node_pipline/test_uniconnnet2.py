import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from node_toolkit.node_net import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset, OneHot, OrderedSampler, worker_init_fn
from node_toolkit.node_utils import test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the testing pipeline for the updated MHD_Nodet project.
    运行更新后的 MHD_Nodet 项目测试流水线的主函数。
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and model paths
    base_data_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data"
    test_data_dir = os.path.join(base_data_dir, "scratch_imagesTs")
    save_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\uniconnnet20251018\20251019uniconnnet2_fold1"

    # Load HDNet weights
    load_hdnet = {
        "unet1": os.path.join(save_dir, "unet1.pth"),
        "unet2": os.path.join(save_dir, "unet2.pth"),
        "unet3": os.path.join(save_dir, "unet3.pth"),
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

    # Hyperparameters
    batch_size = 16
    num_dimensions = 3
    num_workers = 0

    # UNet1 configuration (5-channel input, no dropout)
    node_configs_unet1 = {
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
        "n14": (32, 64, 64, 64),  # Output for uniconnnet_n9
    }
    hyperedge_configs_unet1 = {
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
            }
        },
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
            }
        },
    }

    # UNet2 configuration (5-channel input, no dropout)
    node_configs_unet2 = {
        "n0": (1, 64, 64, 64),    # Input from uniconnnet_n9
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
        "n14": (32, 64, 64, 64),  # Output for uniconnnet_n9
    }
    hyperedge_configs_unet2 = {
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
            }
        },
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
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
                "dropout": 0.0
            }
        },
    }

    # UNet3 configuration (5-channel input, with dropout 0.1 to 0.5)
    node_configs_unet3 = {
        "n0": (1, 64, 64, 64),    # Input from uniconnnet_n9
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
        "n14": (32, 64, 64, 64),  # Output for classifiers
    }
    hyperedge_configs_unet3 = {
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

    # uniconnnet_n5 (8x8x8, 256 channels)
    node_configs_uniconnnet_n5 = {
        "n0": (256, 8, 8, 8),  # From unet1 n8
        "n1": (256, 8, 8, 8),  # From unet1 n10
        "n2": (256, 8, 8, 8),  # To unet2 n8
        "n3": (256, 8, 8, 8),  # To unet2 n10
        "n4": (256, 8, 8, 8),  # To unet3 n8
        "n5": (256, 8, 8, 8),  # To unet3 n10
    }
    hyperedge_configs_uniconnnet_n5 = {
        "e1": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n2", "n3", "n4", "n5"],
            "params": {
                "convs": [torch.Size([1024, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "dropout": 0.0
            }
        },
        "e2": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([512, 1024, 1, 1, 1])],
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
        "n0": (128, 16, 16, 16),  # From unet1 n7
        "n1": (128, 16, 16, 16),  # From unet1 n11
        "n2": (128, 16, 16, 16),  # To unet2 n7
        "n3": (128, 16, 16, 16),  # To unet2 n11
        "n4": (128, 16, 16, 16),  # To unet3 n7
        "n5": (128, 16, 16, 16),  # To unet3 n11
    }
    hyperedge_configs_uniconnnet_n6 = {
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
                "dropout": 0.0
            }
        },
        "e2": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([256, 512, 1, 1, 1])],
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
        "n0": (64, 32, 32, 32),  # From unet1 n6
        "n1": (64, 32, 32, 32),  # From unet1 n12
        "n2": (64, 32, 32, 32),  # To unet2 n6
        "n3": (64, 32, 32, 32),  # To unet2 n12
        "n4": (64, 32, 32, 32),  # To unet3 n6
        "n5": (64, 32, 32, 32),  # To unet3 n12
    }
    hyperedge_configs_uniconnnet_n7 = {
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
                "dropout": 0.0
            }
        },
        "e2": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([128, 256, 1, 1, 1])],
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
        "n0": (32, 64, 64, 64),  # From unet1 n5
        "n1": (32, 64, 64, 64),  # From unet1 n13
        "n2": (32, 64, 64, 64),  # To unet2 n5
        "n3": (32, 64, 64, 64),  # To unet2 n13
        "n4": (32, 64, 64, 64),  # To unet3 n5
        "n5": (32, 64, 64, 64),  # To unet3 n13
    }
    hyperedge_configs_uniconnnet_n8 = {
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
                "dropout": 0.0
            }
        },
        "e2": {
            "src_nodes": ["n0", "n1", "n2", "n3"],
            "dst_nodes": ["n4", "n5"],
            "params": {
                "convs": [torch.Size([64, 128, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # uniconnnet_n9 (64x64x64, connecting unet1 to unet2 and unet3)
    node_configs_uniconnnet_n9 = {
        "n0": (1, 64, 64, 64),   # From unet1 n0
        "n1": (1, 64, 64, 64),   # From unet1 n1
        "n2": (1, 64, 64, 64),   # From unet1 n2
        "n3": (1, 64, 64, 64),   # From unet1 n3
        "n4": (1, 64, 64, 64),   # From unet1 n4
        "n5": (32, 64, 64, 64),  # From unet1 n14
        "n6": (1, 64, 64, 64),   # To unet2 n0
        "n7": (1, 64, 64, 64),   # To unet2 n1
        "n8": (1, 64, 64, 64),   # To unet2 n2
        "n9": (1, 64, 64, 64),   # To unet2 n3
        "n10": (1, 64, 64, 64),  # To unet2 n4
        "n11": (32, 64, 64, 64), # To unet2 n14
        "n12": (1, 64, 64, 64),  # To unet3 n0
        "n13": (1, 64, 64, 64),  # To unet3 n1
        "n14": (1, 64, 64, 64),  # To unet3 n2
        "n15": (1, 64, 64, 64),  # To unet3 n3
        "n16": (1, 64, 64, 64),  # To unet3 n4
        "n17": (32, 64, 64, 64), # To unet3 n14
    }
    hyperedge_configs_uniconnnet_n9 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4", "n5"],
            "dst_nodes": ["n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16", "n17"],
            "params": {
                "convs": [torch.Size([74, 37, 1, 1, 1])],  # 1+1+1+1+1+32=37 to 5*1+32+5*1+32=74
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0
            }
        },
        "e2": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11"],
            "dst_nodes": ["n12", "n13", "n14", "n15", "n16", "n17"],
            "params": {
                "convs": [torch.Size([37, 74, 1, 1, 1])],  # 5*1+32+5*1+32=74 to 5*1+32=37
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
        "n0": (256, 8, 8, 8),  # Input from unet3 n10
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
        "n0": (128, 16, 16, 16),  # Input from unet3 n11
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
        "n0": (64, 32, 32, 32),  # Input from unet3 n12
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
        "n0": (32, 64, 64, 64),  # Input from unet3 n13
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
        "n0": (32, 64, 64, 64),  # Input from unet3 n14
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
        ("n126", "unet2", "n11"),
        ("n126", "uniconnnet_n6", "n3"),
        ("n127", "unet2", "n12"),
        ("n127", "uniconnnet_n7", "n3"),
        ("n128", "unet2", "n13"),
        ("n128", "uniconnnet_n8", "n3"),
        ("n129", "unet2", "n14"),
        ("n129", "uniconnnet_n9", "n11"),
        ("n130", "unet3", "n0"),
        ("n130", "uniconnnet_n9", "n12"),
        ("n131", "unet3", "n1"),
        ("n131", "uniconnnet_n9", "n13"),
        ("n132", "unet3", "n2"),
        ("n132", "uniconnnet_n9", "n14"),
        ("n133", "unet3", "n3"),
        ("n133", "uniconnnet_n9", "n15"),
        ("n134", "unet3", "n4"),
        ("n134", "uniconnnet_n9", "n16"),
        ("n135", "unet3", "n5"),
        ("n135", "uniconnnet_n8", "n4"),
        ("n136", "unet3", "n6"),
        ("n136", "uniconnnet_n7", "n4"),
        ("n137", "unet3", "n7"),
        ("n137", "uniconnnet_n6", "n4"),
        ("n138", "unet3", "n8"),
        ("n138", "uniconnnet_n5", "n4"),
        ("n139", "unet3", "n9"),
        ("n140", "unet3", "n10"),
        ("n140", "uniconnnet_n5", "n5"),
        ("n140", "classifier_n5", "n0"),
        ("n141", "unet3", "n11"),
        ("n141", "uniconnnet_n6", "n5"),
        ("n141", "classifier_n6", "n0"),
        ("n142", "unet3", "n12"),
        ("n142", "uniconnnet_n7", "n5"),
        ("n142", "classifier_n7", "n0"),
        ("n143", "unet3", "n13"),
        ("n143", "uniconnnet_n8", "n5"),
        ("n143", "classifier_n8", "n0"),
        ("n144", "unet3", "n14"),
        ("n144", "uniconnnet_n9", "n17"),
        ("n144", "classifier_n9", "n0"),
        ("n145", "classifier_n5", "n1"),
        ("n146", "classifier_n6", "n1"),
        ("n147", "classifier_n7", "n1"),
        ("n148", "classifier_n8", "n1"),
        ("n149", "classifier_n9", "n1"),
        ("n150", "label_net", "n0"),
    ]

    # Sub-network configurations
    sub_networks_configs = {
        "unet1": (node_configs_unet1, hyperedge_configs_unet1),
        "unet2": (node_configs_unet2, hyperedge_configs_unet2),
        "unet3": (node_configs_unet3, hyperedge_configs_unet3),
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
            state_dict = torch.load(weight_path, map_location=device)
            try:
                sub_networks[net_name].load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded pretrained weights for {net_name} from {weight_path} with full matching")
            except RuntimeError as e:
                logger.warning(f"Full matching failed for {net_name}: {e}. Attempting partial matching.")
                sub_networks[net_name].load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded pretrained weights for {net_name} from {weight_path} with partial matching")
        else:
            logger.warning(f"Could not load weights for {net_name}: {weight_path} does not exist")

    # Global input and output nodes
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n150"]
    out_nodes = ["n145", "n146", "n147", "n148", "n149", "n150"]

    # Node file mapping
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "1004.nii.gz"),
        ("n150", "0006.csv"),
    ]

    # Node suffix mapping for saving
    save_node = [
        ("n145", "uniconnnet2_fold1_gt_0015.npy"),
        ("n146", "uniconnnet2_fold1_gt_1015.npy"),
        ("n147", "uniconnnet2_fold1_gt_2015.npy"),
        ("n148", "uniconnnet2_fold1_gt_3015.npy"),
        ("n149", "uniconnnet2_fold1_gt_4015.npy")
    ]

    # Instantiate transformations
    one_hot4 = OneHot(num_classes=4)

    # Node transformation configuration for test
    node_transforms = {
        "test": {
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
            "n150": [one_hot4]
        }
    }

    # Collect case IDs for test
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

    # Get case IDs for test directory
    test_filename_case_ids = {}
    for filename in filename_to_nodes:
        test_filename_case_ids[filename] = get_case_ids(test_data_dir, filename)

    # Take union of case IDs
    test_case_ids = sorted(list(set().union(*[set(case_ids) for case_ids in test_filename_case_ids.values()])))

    if not test_case_ids:
        raise ValueError("No case_ids found in test directory!")

    # Log missing files for each filename
    for filename, case_ids in test_filename_case_ids.items():
        missing = [cid for cid in test_case_ids if cid not in case_ids]
        if missing:
            logger.warning(f"Missing test files for filename {filename}: {sorted(list(missing))}")

    # Use original order for testing
    test_case_id_order = test_case_ids

    # Create datasets
    datasets_test = {}
    for node, filename in load_node:
        target_shape = None
        for global_node, sub_net_name, sub_node_id in node_mapping:
            if global_node == node:
                target_shape = sub_networks[sub_net_name].node_configs[sub_node_id]
                break
        if target_shape is None:
            raise ValueError(f"Node {node} not found in node_mapping")
        datasets_test[node] = NodeDataset(
            test_data_dir, node, filename, target_shape, node_transforms["test"].get(node, []),
            node_mapping=node_mapping, sub_networks=sub_networks,
            case_ids=test_case_ids, case_id_order=test_case_id_order,
            num_dimensions=num_dimensions
        )

    # Validate case_id_order consistency across nodes
    for node in datasets_test:
        if datasets_test[node].case_ids != datasets_test[list(datasets_test.keys())[0]].case_ids:
            raise ValueError(f"Case ID order inconsistent for node {node} in test")

    # Create DataLoaders
    dataloaders_test = {}
    for node in datasets_test:
        test_indices = list(range(len(datasets_test[node])))
        dataloaders_test[node] = DataLoader(
            datasets_test[node],
            batch_size=batch_size,
            sampler=OrderedSampler(test_indices, num_workers),
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )

    # Load model
    onnx_save_path = os.path.join(save_dir, "model_config_initial.onnx")
    model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions, onnx_save_path=onnx_save_path).to(device)

    # Run test
    test(
        model, dataloaders_test, out_nodes, save_node, test_data_dir, debug=True
    )
    logger.info("Testing completed. Predictions saved to test directory.")

if __name__ == "__main__":
    # Specify the third GPU
    device_id = 0  # Index of the third GPU (starting from 0)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting testing on device: {device}")

    main()
