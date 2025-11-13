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
    Main function to run the inference pipeline for the updated MHD_Nodet project.
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and model paths
    base_data_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Val_Data"
    test_data_dir = os.path.join(base_data_dir, "scratch_imagesTs")
    save_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\UniConnNetII_fold1"

    # Load HDNet weights
    load_hdnet = {
        "label_net": os.path.join(save_dir, "label_net.pth"),
        "unet1": os.path.join(save_dir, "unet1.pth"),
        "unet1_classifier_n9": os.path.join(save_dir, "unet1_classifier_n9.pth"),
        "unet1_classifier_n10": os.path.join(save_dir, "unet1_classifier_n10.pth"),
        "unet1_classifier_n11": os.path.join(save_dir, "unet1_classifier_n11.pth"),
        "unet1_classifier_n12": os.path.join(save_dir, "unet1_classifier_n12.pth"),
        "unet1_classifier_n13": os.path.join(save_dir, "unet1_classifier_n13.pth"),
        "unet1_classifier_n14": os.path.join(save_dir, "unet1_classifier_n14.pth"),
        "unet2": os.path.join(save_dir, "unet2.pth"),
        "unet2_classifier_n9": os.path.join(save_dir, "unet2_classifier_n9.pth"),
        "unet2_classifier_n10": os.path.join(save_dir, "unet2_classifier_n10.pth"),
        "unet2_classifier_n11": os.path.join(save_dir, "unet2_classifier_n11.pth"),
        "unet2_classifier_n12": os.path.join(save_dir, "unet2_classifier_n12.pth"),
        "unet2_classifier_n13": os.path.join(save_dir, "unet2_classifier_n13.pth"),
        "unet2_classifier_n14": os.path.join(save_dir, "unet2_classifier_n14.pth"),
        "UniConnNet_n9": os.path.join(save_dir, "UniConnNet_n9.pth"),
        "UniConnNet_n10": os.path.join(save_dir, "UniConnNet_n10.pth"),
        "UniConnNet_n11": os.path.join(save_dir, "UniConnNet_n11.pth"),
        "UniConnNet_n12": os.path.join(save_dir, "UniConnNet_n12.pth"),
        "UniConnNet_n13": os.path.join(save_dir, "UniConnNet_n13.pth"),
        "UniConnNet_n14": os.path.join(save_dir, "UniConnNet_n14.pth"),
    }

    # Hyperparameters
    batch_size = 16
    num_dimensions = 3
    num_workers = 0

    # UNet1 configuration (5-channel input)
    node_configs_unet1 = {
        "n0": (1, 64, 64, 64),    # Input From unet1 n15
        "n1": (1, 64, 64, 64),    # Input From unet1 n16
        "n2": (1, 64, 64, 64),    # Input From unet1 n17
        "n3": (1, 64, 64, 64),    # Input From unet1 n18
        "n4": (1, 64, 64, 64),    # Input From unet1 n19
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
                "reqs":[True, True],
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

    # UNet2 configuration (5-channel input)
    node_configs_unet2 = {
        "n0": (1, 64, 64, 64),    # Input from unet3 n15
        "n1": (1, 64, 64, 64),    # Input from unet3 n16
        "n2": (1, 64, 64, 64),    # Input from unet3 n17
        "n3": (1, 64, 64, 64),    # Input from unet3 n18
        "n4": (1, 64, 64, 64),    # Input from unet3 n19
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

    # UniConnNet_n9 (4x4x4)
    node_configs_UniConnNet_n9 = {
        "n0": (512, 4, 4, 4),  # unet1 n9
        "n1": (512, 4, 4, 4),  # unet2 n9
    }
    hyperedge_configs_UniConnNet_n9 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([512, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (4, 4, 4),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

    # UniConnNet_n10 (8x8x8)
    node_configs_UniConnNet_n10 = {
        "n0": (256, 8, 8, 8),  # unet1 n8
        "n1": (256, 8, 8, 8),  # unet1 n10
        "n2": (256, 8, 8, 8),  # unet2 n8
        "n3": (256, 8, 8, 8),  # unet2 n10
    }
    hyperedge_configs_UniConnNet_n10 = {
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

    # UniConnNet_n11 (16x16x16)
    node_configs_UniConnNet_n11 = {
        "n0": (128, 16, 16, 16),  # unet1 n7
        "n1": (128, 16, 16, 16),  # unet1 n11
        "n2": (128, 16, 16, 16),  # unet2 n7
        "n3": (128, 16, 16, 16),  # unet2 n11
    }
    hyperedge_configs_UniConnNet_n11 = {
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

    # UniConnNet_n12 (32x32x32)
    node_configs_UniConnNet_n12 = {
        "n0": (64, 32, 32, 32),  # unet1 n6
        "n1": (64, 32, 32, 32),  # unet1 n12
        "n2": (64, 32, 32, 32),  # unet2 n6
        "n3": (64, 32, 32, 32),  # unet2 n12
    }
    hyperedge_configs_UniConnNet_n12 = {
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

    # UniConnNet_n13 (64x64x64)
    node_configs_UniConnNet_n13 = {
        "n0": (32, 64, 64, 64),  # unet1 n5
        "n1": (32, 64, 64, 64),  # unet1 n13
        "n2": (32, 64, 64, 64),  # unet2 n5
        "n3": (32, 64, 64, 64),  # unet2 n13
    }
    hyperedge_configs_UniConnNet_n13 = {
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

    # UniConnNet_n14 (64x64x64)
    node_configs_UniConnNet_n14 = {
        "n0": (1, 64, 64, 64),   # unet1 n0
        "n1": (1, 64, 64, 64),   # unet1 n1
        "n2": (1, 64, 64, 64),   # unet1 n2
        "n3": (1, 64, 64, 64),   # unet1 n3
        "n4": (1, 64, 64, 64),   # unet1 n4
        "n5": (32, 64, 64, 64),  # unet1 n14
        "n6": (1, 64, 64, 64),   # unet2 n0
        "n7": (1, 64, 64, 64),   # unet2 n1
        "n8": (1, 64, 64, 64),   # unet2 n2
        "n9": (1, 64, 64, 64),   # unet2 n3
        "n10": (1, 64, 64, 64),  # unet2 n4
        "n11": (32, 64, 64, 64), # unet2 n14
    }
    hyperedge_configs_UniConnNet_n14 = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4", "n5"],
            "dst_nodes": ["n6", "n7", "n8", "n9", "n10", "n11"],
            "params": {
                "convs": [torch.Size([37, 37, 1, 1, 1])],  
                "reqs": [True],
                "norms": ["batch"],
                "acts": [None],
                "feature_size": (64, 64, 64),
                "intp": "linear",
                "dropout": 0.0
            }
        },
    }

# Classifier for unet1 n9 (512 channels, 4x4x4) - bottleneck
    node_configs_classifier_unet1_n9 = {
        "n0": (512, 4, 4, 4),  # Input from unet1 n9
        "n1": (4, 1, 1, 1),    # Classification output
    }
    hyperedge_configs_classifier_unet1_n9 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }

    # Classifier for unet1 n10 (256 channels, 8x8x8)
    node_configs_classifier_unet1_n10 = {
        "n0": (256, 8, 8, 8),  # Input from unet1 n10
        "n1": (4, 1, 1, 1),    # Classification output
    }
    hyperedge_configs_classifier_unet1_n10 = {
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

    # Classifier for unet1 n11 (128 channels, 16x16x16)
    node_configs_classifier_unet1_n11 = {
        "n0": (128, 16, 16, 16),  # Input from unet1 n11
        "n1": (4, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_unet1_n11 = {
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

    # Classifier for unet1 n12 (64 channels, 32x32x32)
    node_configs_classifier_unet1_n12 = {
        "n0": (64, 32, 32, 32),  # Input from unet1 n12
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_unet1_n12 = {
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

    # Classifier for unet1 n13 (32 channels, 64x64x64)
    node_configs_classifier_unet1_n13 = {
        "n0": (32, 64, 64, 64),  # Input from unet1 n13
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_unet1_n13 = {
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

    # Classifier for unet1 n14 (32 channels, 64x64x64)
    node_configs_classifier_unet1_n14 = {
        "n0": (32, 64, 64, 64),  # Input from unet1 n14
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_unet1_n14 = {
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

# Classifier for unet2 n9 (512 channels, 4x4x4) - bottleneck
    node_configs_classifier_unet2_n9 = {
        "n0": (512, 4, 4, 4),  # Input from unet2 n9
        "n1": (4, 1, 1, 1),    # Classification output
    }
    hyperedge_configs_classifier_unet2_n9 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "convs": [torch.Size([4, 512, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }

    # Classifier for unet2 n10 (256 channels, 8x8x8)
    node_configs_classifier_unet2_n10 = {
        "n0": (256, 8, 8, 8),  # Input from unet2 n10
        "n1": (4, 1, 1, 1),    # Classification output
    }
    hyperedge_configs_classifier_unet2_n10 = {
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

    # Classifier for unet2 n11 (128 channels, 16x16x16)
    node_configs_classifier_unet2_n11 = {
        "n0": (128, 16, 16, 16),  # Input from unet2 n11
        "n1": (4, 1, 1, 1),      # Classification output
    }
    hyperedge_configs_classifier_unet2_n11 = {
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

    # Classifier for unet2 n12 (64 channels, 32x32x32)
    node_configs_classifier_unet2_n12 = {
        "n0": (64, 32, 32, 32),  # Input from unet2 n12
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_unet2_n12 = {
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

    # Classifier for unet2 n13 (32 channels, 64x64x64)
    node_configs_classifier_unet2_n13 = {
        "n0": (32, 64, 64, 64),  # Input from unet2 n13
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_unet2_n13 = {
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

    # Classifier for unet2 n14 (32 channels, 64x64x64)
    node_configs_classifier_unet2_n14 = {
        "n0": (32, 64, 64, 64),  # Input from unet2 n14
        "n1": (4, 1, 1, 1),     # Classification output
    }
    hyperedge_configs_classifier_unet2_n14 = {
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
    # label
    ("n99", "label_net", "n0"),
    # unet1
    ("n100", "unet1", "n0"),
    ("n101", "unet1", "n1"),
    ("n102", "unet1", "n2"),
    ("n103", "unet1", "n3"),
    ("n104", "unet1", "n4"),
    ("n105", "unet1", "n5"),
    ("n106", "unet1", "n6"),
    ("n107", "unet1", "n7"),
    ("n108", "unet1", "n8"),
    ("n109", "unet1", "n9"),
    ("n110", "unet1", "n10"),
    ("n111", "unet1", "n11"),
    ("n112", "unet1", "n12"),
    ("n113", "unet1", "n13"),
    ("n114", "unet1", "n14"),
    # unet1 classifier
    ("n109", "unet1_classifier_n9", "n0"),
    ("n110", "unet1_classifier_n10", "n0"),
    ("n111", "unet1_classifier_n11", "n0"),
    ("n112", "unet1_classifier_n12", "n0"),
    ("n113", "unet1_classifier_n13", "n0"),
    ("n114", "unet1_classifier_n14", "n0"),
    ("n115", "unet1_classifier_n9", "n1"),
    ("n116", "unet1_classifier_n10", "n1"),
    ("n117", "unet1_classifier_n11", "n1"),
    ("n118", "unet1_classifier_n12", "n1"),
    ("n119", "unet1_classifier_n13", "n1"),
    ("n120", "unet1_classifier_n14", "n1"),
    # unet2
    ("n121", "unet2", "n0"),
    ("n122", "unet2", "n1"),
    ("n123", "unet2", "n2"),
    ("n124", "unet2", "n3"),
    ("n125", "unet2", "n4"),
    ("n126", "unet2", "n5"),
    ("n127", "unet2", "n6"),
    ("n128", "unet2", "n7"),
    ("n129", "unet2", "n8"),
    ("n130", "unet2", "n9"),
    ("n131", "unet2", "n10"),
    ("n132", "unet2", "n11"),
    ("n133", "unet2", "n12"),
    ("n134", "unet2", "n13"),
    ("n135", "unet2", "n14"),
    # unet2 classifier
    ("n130", "unet2_classifier_n9", "n0"),
    ("n131", "unet2_classifier_n10", "n0"),
    ("n132", "unet2_classifier_n11", "n0"),
    ("n133", "unet2_classifier_n12", "n0"),
    ("n134", "unet2_classifier_n13", "n0"),
    ("n135", "unet2_classifier_n14", "n0"),
    ("n136", "unet2_classifier_n9", "n1"),
    ("n137", "unet2_classifier_n10", "n1"),
    ("n138", "unet2_classifier_n11", "n1"),
    ("n139", "unet2_classifier_n12", "n1"),
    ("n140", "unet2_classifier_n13", "n1"),
    ("n141", "unet2_classifier_n14", "n1"),
    # UniConnNet
    ("n100", "UniConnNet_n14", "n0"),
    ("n101", "UniConnNet_n14", "n1"),
    ("n102", "UniConnNet_n14", "n2"),
    ("n103", "UniConnNet_n14", "n3"),
    ("n104", "UniConnNet_n14", "n4"),
    ("n105", "UniConnNet_n13", "n0"),
    ("n106", "UniConnNet_n12", "n0"),
    ("n107", "UniConnNet_n11", "n0"),
    ("n108", "UniConnNet_n10", "n0"),
    ("n109", "UniConnNet_n9", "n0"),
    ("n110", "UniConnNet_n10", "n1"),
    ("n111", "UniConnNet_n11", "n1"),
    ("n112", "UniConnNet_n12", "n1"),
    ("n113", "UniConnNet_n13", "n1"),
    ("n114", "UniConnNet_n14", "n5"),
    ("n121", "UniConnNet_n14", "n6"),
    ("n122", "UniConnNet_n14", "n7"),
    ("n123", "UniConnNet_n14", "n8"),
    ("n124", "UniConnNet_n14", "n9"),
    ("n125", "UniConnNet_n14", "n10"),
    ("n126", "UniConnNet_n13", "n2"),
    ("n127", "UniConnNet_n12", "n2"),
    ("n128", "UniConnNet_n11", "n2"),
    ("n129", "UniConnNet_n10", "n2"),
    ("n130", "UniConnNet_n9", "n1"),
    ("n131", "UniConnNet_n10", "n3"),
    ("n132", "UniConnNet_n11", "n3"),
    ("n133", "UniConnNet_n12", "n3"),
    ("n134", "UniConnNet_n13", "n3"),
    ("n135", "UniConnNet_n14", "n11"),
    ]

    # Sub-network configurations
    sub_networks_configs = {
        "unet1": (node_configs_unet1, hyperedge_configs_unet1),
        "unet2": (node_configs_unet2, hyperedge_configs_unet2),
        "unet1_classifier_n9": (node_configs_classifier_unet1_n9, hyperedge_configs_classifier_unet1_n9),
        "unet1_classifier_n10": (node_configs_classifier_unet1_n10, hyperedge_configs_classifier_unet1_n10),
        "unet1_classifier_n11": (node_configs_classifier_unet1_n11, hyperedge_configs_classifier_unet1_n11),
        "unet1_classifier_n12": (node_configs_classifier_unet1_n12, hyperedge_configs_classifier_unet1_n12),
        "unet1_classifier_n13": (node_configs_classifier_unet1_n13, hyperedge_configs_classifier_unet1_n13),
        "unet1_classifier_n14": (node_configs_classifier_unet1_n14, hyperedge_configs_classifier_unet1_n14),
        "unet2_classifier_n9": (node_configs_classifier_unet2_n9, hyperedge_configs_classifier_unet2_n9),
        "unet2_classifier_n10": (node_configs_classifier_unet2_n10, hyperedge_configs_classifier_unet2_n10),
        "unet2_classifier_n11": (node_configs_classifier_unet2_n11, hyperedge_configs_classifier_unet2_n11),
        "unet2_classifier_n12": (node_configs_classifier_unet2_n12, hyperedge_configs_classifier_unet2_n12),
        "unet2_classifier_n13": (node_configs_classifier_unet2_n13, hyperedge_configs_classifier_unet2_n13),
        "unet2_classifier_n14": (node_configs_classifier_unet2_n14, hyperedge_configs_classifier_unet2_n14),
        "label_net": (node_configs_label, hyperedge_configs_label),
        "UniConnNet_n9": (node_configs_UniConnNet_n9, hyperedge_configs_UniConnNet_n9),
        "UniConnNet_n10": (node_configs_UniConnNet_n10, hyperedge_configs_UniConnNet_n10),
        "UniConnNet_n11": (node_configs_UniConnNet_n11, hyperedge_configs_UniConnNet_n11),
        "UniConnNet_n12": (node_configs_UniConnNet_n12, hyperedge_configs_UniConnNet_n12),
        "UniConnNet_n13": (node_configs_UniConnNet_n13, hyperedge_configs_UniConnNet_n13),
        "UniConnNet_n14": (node_configs_UniConnNet_n14, hyperedge_configs_UniConnNet_n14),
    }

    # Instantiate sub-networks
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, num_dimensions)
        for name, (node_configs, hyperedge_configs) in sub_networks_configs.items()
    }

    # Load pretrained weights
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
    in_nodes = ["n99", "n100", "n101", "n102", "n103", "n104",]
    out_nodes = ["n99", "n115", "n116", "n117", "n118", "n119", "n120", "n136", "n137", "n138", "n139", "n140", "n141",]

    # Node file mapping
    load_node = [
        ("n99", "0006.csv"),
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
    ]

    # Node suffix mapping for saving
    save_node = [
        ("n130", "UniConnNetII_fold1_gt_0015.npy"),
        ("n131", "UniConnNetII_fold1_gt_1015.npy"),
        ("n132", "UniConnNetII_fold1_gt_2015.npy"),
        ("n133", "UniConnNetII_fold1_gt_3015.npy"),
        ("n134", "UniConnNetII_fold1_gt_4015.npy"),
        ("n135", "UniConnNetII_fold1_gt_5015.npy"),
        ("n136", "UniConnNetII_fold1_gt_6015.npy"),
        ("n137", "UniConnNetII_fold1_gt_7015.npy"),
        ("n138", "UniConnNetII_fold1_gt_8015.npy"),
        ("n139", "UniConnNetII_fold1_gt_9015.npy"),
        ("n140", "UniConnNetII_fold1_gt_10015.npy"),
        ("n141", "UniConnNetII_fold1_gt_11015.npy"),
    ]

    # Instantiate transformations
    one_hot4 = OneHot(num_classes=4)

    # Node transformation configuration for test
    node_transforms = {
        "test": {
            "n99": [one_hot4],
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
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

    # Log missing files
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

    # Validate case_id_order consistency
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
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting testing on device: {device}")

    main()
