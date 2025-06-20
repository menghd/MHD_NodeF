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
    base_data_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Scratch_Data\scratch_imagesTs"
    test_data_dir = base_data_dir
    save_dir = r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\MHDNet0618"

    # Load HDNet weights
    load_hdnet = {
        "preprocess": r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\MHDNet0618\preprocess.pth",
        "resnet18": r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\MHDNet0618\resnet18.pth",
        "gt": r"C:\Users\PC\PycharmProjects\thu_xwh\Codes\Model\MHDNet0618\gt.pth"
    }

    # Hyperparameters
    batch_size = 8
    num_dimensions = 3
    num_workers = 0

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

    # Node suffix mapping for loading
    load_node = [
        ("n100", "0000.nii.gz"),
        ("n101", "0001.nii.gz"),
        ("n102", "0002.nii.gz"),
        ("n103", "0003.nii.gz"),
        ("n104", "0004.nii.gz"),
        ("n116", "0005.csv")
    ]

    # Node suffix mapping for saving
    save_node = [
        ("n115", "0005.npy")
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
            "n116": [one_hot8]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
