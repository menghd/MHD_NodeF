"""
MHD_Nodet Project - Testing Module
==================================
This module implements the testing pipeline for the MHD_Nodet project, integrating network, dataset, and evaluation components.
- Supports custom data loading from test directory with batch-consistent processing.
- Saves predictions for specified nodes to the test directory.
- Supports flexible HDNet weight loading via load_hdnet configuration.

项目：MHD_Nodet - 测试模块
本模块实现 MHD_Nodet 项目的测试流水线，集成网络、数据集和评估组件。
- 支持从测试目录加载自定义数据，采用批次一致的处理方式。
- 将指定节点的预测结果保存到测试目录。
- 支持通过 load_hdnet 配置灵活加载 HDNet 权重。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

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
    Main function to run the testing pipeline.
    运行测试流水线的主函数。
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Data and model paths
    base_data_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Scratch_Data"
    test_data_dir = os.path.join(base_data_dir, "scratch_imagesTs")
    save_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\MHDNet_poly"

    # Load HDNet weights
    load_hdnet = {
        "encoder": os.path.join(save_dir, "encoder.pth"),
        "decoder": os.path.join(save_dir, "decoder.pth"),
        "classifier_n5": os.path.join(save_dir, "classifier_n5.pth"),
        "classifier_n6": os.path.join(save_dir, "classifier_n6.pth"),
        "classifier_n7": os.path.join(save_dir, "classifier_n7.pth"),
        "classifier_n8": os.path.join(save_dir, "classifier_n8.pth"),
        "classifier_n9": os.path.join(save_dir, "classifier_n9.pth"),
        "label_net": os.path.join(save_dir, "label_net.pth"),
    }

    # Hyperparameters
    batch_size = 8
    num_dimensions = 3
    num_workers = 0

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

    # Classifier configurations for each decoder output
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

    # Label network configuration (single node)
    node_configs_label = {
        "n0": (4, 1, 1, 1),  # Unified one-hot encoded label
    }
    hyperedge_configs_label = {}  # No hyperedges needed
    in_nodes_label = ["n0"]
    out_nodes_label = ["n0"]

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
        ("n115", "classifier_n5", "n1"),  # Classification output n5
        ("n116", "classifier_n6", "n1"),  # Classification output n6
        ("n117", "classifier_n7", "n1"),  # Classification output n7
        ("n118", "classifier_n8", "n1"),  # Classification output n8
        ("n119", "classifier_n9", "n1"),  # Classification output n9
        ("n120", "label_net", "n0"),      # Unified label node
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
        "label_net": (node_configs_label, hyperedge_configs_label, in_nodes_label, out_nodes_label),
    }
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # Load pretrained weights for specified HDNets
    for net_name, weight_path in load_hdnet.items():
        if net_name in sub_networks and os.path.exists(weight_path):
            # 关键修改：添加map_location
            state_dict = torch.load(weight_path, map_location=device)
            sub_networks[net_name].load_state_dict(state_dict)
            logger.info(f"Loaded pretrained weights for {net_name} from {weight_path}")
        else:
            logger.warning(f"Could not load weights for {net_name}: {weight_path} does not exist")

    # Global input and output nodes
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n120"]
    out_nodes = ["n115", "n116", "n117", "n118", "n119", "n120"]

    # Node suffix mapping for loading
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n120", "0015.csv")
    ]

    # Node suffix mapping for saving
    save_node = [
        ("n115", "0015.npy"),
        ("n116", "1015.npy"),
        ("n117", "2015.npy"),
        ("n118", "3015.npy"),
        ("n119", "4015.npy")
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
            "n120": [one_hot4]
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
        model, dataloaders_test, out_nodes, save_node, test_data_dir, sub_networks, node_mapping, debug=True
    )
    logger.info("Testing completed. Predictions saved to test directory.")

if __name__ == "__main__":
    # Specify the third GPU
    device_id = 0  # Index of the third GPU (starting from 0)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting testing on device: {device}")
    main()
