"""
MHD_Nodet Project - Testing Module
==================================
This module implements the testing pipeline for the MHD_Nodet project, integrating network and dataset components.
- Loads test data from a specified directory and applies minimal transformations.
- Uses a pre-trained model to generate predictions and saves outputs in .npy format.
- Handles missing input files by relying on NodeDataset to provide placeholder feature maps.

项目：MHD_Nodet - 测试模块
本模块实现 MHD_Nodet 项目的测试流水线，集成网络和数据集组件。
- 从指定目录加载测试数据，应用最小的变换。
- 使用预训练模型生成预测并以 .npy 格式保存输出。
- 通过 NodeDataset 处理缺失输入文件，提供占位特征图。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import sys
sys.path.append(r"C:\Users\souray\Desktop\Codes")
from node_toolkit.new_node_net import MHDNet, HDNet
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
    base_data_dir = r"C:\Users\souray\Desktop\Tr"
    test_data_dir = os.path.join(base_data_dir, "test")
    weights_path = r"C:\Users\souray\Desktop\MHDNet0601\model_best_epoch_178.pth"

    # Hyperparameters
    batch_size = 8
    num_dimensions = 3
    num_workers = 0

    # Subnetwork configuration (3D U-Net for merge)
    node_configs_merge = {
        "n0": (1, 64, 64, 64),
        "n1": (1, 64, 64, 64),
        "n2": (1, 64, 64, 64),
        "n3": (1, 64, 64, 64),
        "n4": (1, 64, 64, 64),
        "n5": (32, 64, 64, 64),
        "n6": (64, 32, 32, 32),
        "n7": (128, 16, 16, 16),
        "n8": (256, 8, 8, 8),
        "n9": (512, 4, 4, 4),
        "n10": (256, 8, 8, 8),
        "n11": (128, 16, 16, 16),
        "n12": (64, 32, 32, 32),
        "n13": (32, 64, 64, 64),
        "n14": (32, 64, 64, 64),
        "n15": (8, 1, 1, 1),
        "n16": (8, 1, 1, 1),
    }
    hyperedge_configs_merge = {
        "e1": {
            "src_nodes": ["n0", "n1", "n2", "n3", "n4"],
            "dst_nodes": ["n5"],
            "params": {
                "convs": [torch.Size([32, 5, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear"
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
                "intp": "max"
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
                "intp": "max"
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
                "intp": "max"
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
                "intp": "max"
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
                "intp": "linear"
            }
        },
        "e7": {
            "src_nodes": ["n8", "n10"],
            "dst_nodes": ["n11"],
            "params": {
                "convs": [torch.Size([256, 512, 1, 1, 1]), torch.Size([256, 256, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "intp": "linear"
            }
        },
        "e8": {
            "src_nodes": ["n7", "n11"],
            "dst_nodes": ["n12"],
            "params": {
                "convs": [torch.Size([128, 256, 1, 1, 1]), torch.Size([128, 128, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "intp": "linear"
            }
        },
        "e9": {
            "src_nodes": ["n6", "n12"],
            "dst_nodes": ["n13"],
            "params": {
                "convs": [torch.Size([64, 128, 1, 1, 1]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "intp": "linear"
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
                "intp": "linear"
            }
        },
        "e11": {
            "src_nodes": ["n14"],
            "dst_nodes": ["n15"],
            "params": {
                "convs": [torch.Size([8, 32, 1, 1, 1])],
                "reqs": [True],
                "norms": ["batch"],
                "acts": ["softmax"],
                "feature_size": (1, 1, 1),
                "intp": "avg"
            }
        },
    }
    in_nodes_merge = ["n0", "n1", "n2", "n3", "n4", "n16"]
    out_nodes_merge = ["n15", "n16"]

    # Global node mapping
    node_mapping = [
        ("n100", "merge", "n0"),
        ("n101", "merge", "n1"),
        ("n102", "merge", "n2"),
        ("n103", "merge", "n3"),
        ("n104", "merge", "n4"),
        ("n115", "merge", "n15"),
        ("n116", "merge", "n16"),
    ]

    # Instantiate subnetworks
    sub_networks_configs = {
        "merge": (node_configs_merge, hyperedge_configs_merge, in_nodes_merge, out_nodes_merge),
    }
    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # Global input and output nodes
    in_nodes = ["n100", "n101", "n102", "n103", "n104", "n116"]
    out_nodes = ["n115", "n116"]

    # Node suffix mapping for loading
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n116", "0005.csv"),
    ]

    # Node suffix mapping for saving
    save_node = [
        ("n115", "0005.npy"),
        ("n116", "0006.npy"),
    ]

    # Instantiate transformations
    one_hot8 = OneHot(num_classes=8)

    # Node transformation configuration for test
    node_transforms = {
        "test": {
            "n100": [],
            "n101": [],
            "n102": [],
            "n103": [],
            "n104": [],
            "n116": [one_hot8],
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

    # Collect all unique case IDs (union) while preserving order
    all_test_case_ids = []
    seen_test = set()
    for filename in filename_to_nodes:
        for case_id in test_filename_case_ids[filename]:
            if case_id not in seen_test:
                all_test_case_ids.append(case_id)
                seen_test.add(case_id)
    test_case_ids = sorted(all_test_case_ids)

    if not test_case_ids:
        raise ValueError("No case_ids found in test directory!")

    # Log missing files for each case
    for filename, nodes in filename_to_nodes.items():
        missing_test = [cid for cid in test_case_ids if cid not in test_filename_case_ids[filename]]
        if missing_test:
            logger.warning(f"Missing test files for filename {filename} (nodes {nodes}): cases {sorted(missing_test)}")

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

    # Initialize model and load weights
    onnx_save_path = os.path.join(os.path.dirname(weights_path), "model_config_initial.onnx")
    model = MHDNet(sub_networks, node_mapping, in_nodes, out_nodes, num_dimensions, onnx_save_path=onnx_save_path).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info(f"Model loaded from {weights_path}")

    # Run test
    test(
        model, dataloaders_test, out_nodes, save_node, test_data_dir, sub_networks, node_mapping, debug=True
    )
    logger.info("Testing completed")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
