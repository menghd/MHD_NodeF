"""
MHD_Nodet Project - Utilities Module
====================================
This module provides utility functions for training and validation in the MHD_Nodet project,
including data type mapping, training, and validation routines.

项目：MHD_Nodet - 工具模块
本模块为 MHD_Nodet 项目提供训练和验证的工具函数，包括数据类型映射、训练和验证例程。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import torch
import numpy as np
from tabulate import tabulate
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_node_dtype_mapping(node_mapping, sub_networks):
    dtype_map = {
        "float": torch.float32,
        "long": torch.int64
    }
    node_dtype_mapping = {}
    
    for global_node, sub_net_name, sub_node_id in node_mapping:
        if sub_net_name in sub_networks:
            sub_net = sub_networks[sub_net_name]
            dtype_str = sub_net.node_dtype.get(sub_node_id, "float")
            if dtype_str not in dtype_map:
                raise ValueError(f"Unsupported dtype {dtype_str} for node {global_node} in subnetwork {sub_net_name}")
            node_dtype_mapping[global_node] = dtype_map[dtype_str]
    
    return node_dtype_mapping

def train(model, dataloaders, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, node_transforms, debug=False):
    model.train()
    running_loss = 0.0
    task_losses = {task: {loss_cfg["fn"].__name__: [] for loss_cfg in task_configs[task]["loss"]} for task in task_configs}
    task_metrics = {task: [] for task in task_configs}
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    class_distributions = {task: [] for task in task_configs}
    case_ids_per_batch = []

    node_dtype_mapping = get_node_dtype_mapping(node_mapping, sub_networks)
    
    data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        inputs_list = []
        batch_case_ids = []
        
        for node in dataloaders:
            dataset = dataloaders[node].dataset
            batch_data = next(data_iterators[node])
            data = batch_data.to(device)
            start_idx = batch_idx * dataloaders[node].batch_size
            end_idx = min((batch_idx + 1) * dataloaders[node].batch_size, len(dataset))
            current_case_ids = dataset.case_ids[start_idx:end_idx]
            batch_case_ids.append(current_case_ids)
            
            expected_dtype = node_dtype_mapping.get(node, torch.float32)
            if data.dtype != expected_dtype:
                if debug:
                    logger.info(f"Converting node {node} data from {data.dtype} to {expected_dtype}")
                data = data.to(dtype=expected_dtype)
            inputs_list.append(data)
        
        batch_case_ids_set = set(batch_case_ids[0])
        if not all(set(cids) == batch_case_ids_set for cids in batch_case_ids):
            logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes: {batch_case_ids}")
        case_ids_per_batch.append(list(batch_case_ids_set))
        
        outputs = model(inputs_list)
        total_loss = torch.tensor(0.0, device=device)

        for task, config in task_configs.items():
            task_loss = torch.tensor(0.0, device=device)
            src_node = config["metric"][0]["src_node"] if config.get("metric") else None
            target_node = config["metric"][0]["target_node"] if config.get("metric") else None
            if src_node and target_node:
                src_idx = out_nodes.index(src_node)
                target_idx = out_nodes.index(target_node)
                all_preds[task].append(outputs[src_idx].detach())
                all_targets[task].append(outputs[target_idx].detach())
                
            target_idx = out_nodes.index(config["loss"][0]["target_node"])
            target_tensor = outputs[target_idx]
            class_indices = torch.argmax(target_tensor, dim=1).flatten().cpu().numpy()
            class_counts = Counter(class_indices)
            class_distributions[task].append(class_counts)
            
            for loss_cfg in config["loss"]:
                fn = loss_cfg["fn"]
                src_node = loss_cfg["src_node"]
                target_node = loss_cfg["target_node"]
                weight = loss_cfg["weight"]
                params = loss_cfg["params"]
                src_idx = out_nodes.index(src_node)
                target_idx = out_nodes.index(target_node)
                loss = fn(outputs[src_idx], outputs[target_idx], **params)
                task_loss += weight * loss
                task_losses[task][fn.__name__].append(loss.item())
            total_loss += task_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += total_loss.item()

    avg_loss = running_loss / num_batches
    task_losses_avg = {
        task: sum(
            np.mean(task_losses[task][loss_cfg["fn"].__name__]) * loss_cfg["weight"]
            for loss_cfg in task_configs[task]["loss"]
        ) for task in task_configs
    }

    for task, config in task_configs.items():
        metrics = []
        if config.get("metric"):
            src_tensor = torch.cat(all_preds[task], dim=0)
            target_tensor = torch.cat(all_targets[task], dim=0)
            for metric_cfg in config["metric"]:
                fn = metric_cfg["fn"]
                src_node = metric_cfg["src_node"]
                target_node = metric_cfg["target_node"]
                params = metric_cfg["params"]
                result = fn(src_tensor, target_tensor, **params)
                metrics.append({"fn": fn.__name__, "src_node": src_node, "target_node": target_node, "result": result})
        task_metrics[task] = metrics

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Total Loss: {avg_loss:.4f}")
    for task, avg_task_loss in task_losses_avg.items():
        print(f"Task: {task}, Avg Loss: {avg_task_loss:.4f}")
        for loss_cfg in task_configs[task]["loss"]:
            fn_name = loss_cfg["fn"].__name__
            src_node = loss_cfg["src_node"]
            target_node = loss_cfg["target_node"]
            weight = loss_cfg["weight"]
            params_str = ", ".join(f"{k}={v}" for k, v in loss_cfg["params"].items())
            avg_loss_value = np.mean(task_losses[task][fn_name])
            print(f"  Loss: {fn_name}({src_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss_value:.4f}")

        for metric in task_metrics[task]:
            fn_name = metric["fn"]
            src_node = metric["src_node"]
            target_node = metric["target_node"]
            result = metric["result"]
            headers = ["Class", metric["fn"].split("_")[1].capitalize()]
            table = [[f"Class {i}", f"{v:.4f}" if not np.isnan(v) else "N/A"] for i, v in enumerate(result["per_class"])] + [["Avg", f"{result['avg']:.4f}" if not np.isnan(result['avg']) else "N/A"]]
            print(f"  Metric: {fn_name}({src_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))

        # 打印类分布
        print(f"  Class Distribution for Task: {task}")
        # 汇总所有批次的类分布
        total_counts = Counter()
        for batch_counts in class_distributions[task]:
            total_counts.update(batch_counts)
        # 准备表格数据
        dist_table = [[f"Class {cls}", count] for cls, count in sorted(total_counts.items())]
        dist_headers = ["Class", "Count"]
        print(tabulate(table, headers=dist_headers, tablefmt="grid"))

    return avg_loss, task_losses_avg, task_metrics

def validate(model, dataloaders, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, debug=False):
    model.eval()
    running_loss = 0.0
    task_losses = {task: {loss_cfg["fn"].__name__: [] for loss_cfg in task_configs[task]["loss"]} for task in task_configs}
    task_metrics = {task: [] for task in task_configs}
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    case_ids_per_batch = []
    class_distributions = {task: [] for task in task_configs}

    node_dtype_mapping = get_node_dtype_mapping(node_mapping, sub_networks)

    with torch.no_grad():
        data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
        num_batches = len(next(iter(data_iterators.values())))

        for batch_idx in range(num_batches):
            inputs_list = []
            batch_case_ids = []
            
            for node in dataloaders:
                dataset = dataloaders[node].dataset
                batch_data = next(data_iterators[node])
                data = batch_data.to(device)
                start_idx = batch_idx * dataloaders[node].batch_size
                end_idx = min((batch_idx + 1) * dataloaders[node].batch_size, len(dataset))
                current_case_ids = dataset.case_ids[start_idx:end_idx]
                batch_case_ids.append(current_case_ids)
                
                expected_dtype = node_dtype_mapping.get(node, torch.float32)
                if data.dtype != expected_dtype:
                    if debug:
                        logger.info(f"Converting node {node} data from {data.dtype} to {expected_dtype}")
                    data = data.to(dtype=expected_dtype)
                inputs_list.append(data)
            
            batch_case_ids_set = set(batch_case_ids[0])
            if not all(set(cids) == batch_case_ids_set for cids in batch_case_ids):
                logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes: {batch_case_ids}")
            case_ids_per_batch.append(list(batch_case_ids_set))
            
            outputs = model(inputs_list)
            total_loss = torch.tensor(0.0, device=device)

            for task, config in task_configs.items():
                task_loss = torch.tensor(0.0, device=device)
                src_node = config["metric"][0]["src_node"] if config.get("metric") else None
                target_node = config["metric"][0]["target_node"] if config.get("metric") else None
                if src_node and target_node:
                    src_idx = out_nodes.index(src_node)
                    target_idx = out_nodes.index(target_node)
                    all_preds[task].append(outputs[src_idx].detach())
                    all_targets[task].append(outputs[target_idx].detach())
                
                target_idx = out_nodes.index(config["loss"][0]["target_node"])
                target_tensor = outputs[target_idx]
                class_indices = torch.argmax(target_tensor, dim=1).flatten().cpu().numpy()
                class_counts = Counter(class_indices)
                class_distributions[task].append(class_counts)
                
                for loss_cfg in config["loss"]:
                    fn = loss_cfg["fn"]
                    src_node = loss_cfg["src_node"]
                    target_node = loss_cfg["target_node"]
                    weight = loss_cfg["weight"]
                    params = loss_cfg["params"]
                    src_idx = out_nodes.index(src_node)
                    target_idx = out_nodes.index(target_node)
                    loss = fn(outputs[src_idx], outputs[target_idx], **params)
                    task_loss += weight * loss
                    task_losses[task][fn.__name__].append(loss.item())
                total_loss += task_loss

            running_loss += total_loss.item()

    avg_loss = running_loss / num_batches
    task_losses_avg = {
        task: sum(
            np.mean(task_losses[task][loss_cfg["fn"].__name__]) * loss_cfg["weight"]
            for loss_cfg in task_configs[task]["loss"]
        ) for task in task_configs
    }

    for task, config in task_configs.items():
        metrics = []
        if config.get("metric"):
            src_tensor = torch.cat(all_preds[task], dim=0)
            target_tensor = torch.cat(all_targets[task], dim=0)
            for metric_cfg in config["metric"]:
                fn = metric_cfg["fn"]
                src_node = metric_cfg["src_node"]
                target_node = metric_cfg["target_node"]
                params = metric_cfg["params"]
                result = fn(src_tensor, target_tensor, **params)
                metrics.append({"fn": fn.__name__, "src_node": src_node, "target_node": target_node, "result": result})
        task_metrics[task] = metrics

    print(f"Epoch [{epoch+1}/{num_epochs}], Val Total Loss: {avg_loss:.4f}")
    for task, avg_task_loss in task_losses_avg.items():
        print(f"Task: {task}, Avg Loss: {avg_task_loss:.4f}")
        for loss_cfg in task_configs[task]["loss"]:
            fn_name = loss_cfg["fn"].__name__
            src_node = loss_cfg["src_node"]
            target_node = loss_cfg["target_node"]
            weight = loss_cfg["weight"]
            params_str = ", ".join(f"{k}={v}" for k, v in loss_cfg["params"].items())
            avg_loss_value = np.mean(task_losses[task][fn_name])
            print(f"  Loss: {fn_name}({src_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss_value:.4f}")

        for metric in task_metrics[task]:
            fn_name = metric["fn"]
            src_node = metric["src_node"]
            target_node = metric["target_node"]
            result = metric["result"]
            headers = ["Class", metric["fn"].split("_")[1].capitalize()]
            table = [[f"Class {i}", f"{v:.4f}" if not np.isnan(v) else "N/A"] for i, v in enumerate(result["per_class"])] + [["Avg", f"{result['avg']:.4f}" if not np.isnan(result['avg']) else "N/A"]]
            print(f"  Metric: {fn_name}({src_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))

        # 打印类分布
        print(f"  Class Distribution for Task: {task}")
        # 汇总所有批次的类分布
        total_counts = Counter()
        for batch_counts in class_distributions[task]:
            total_counts.update(batch_counts)
        # 准备表格数据
        dist_table = [[f"Class {cls}", count] for cls, count in sorted(total_counts.items())]
        dist_headers = ["Class", "Count"]
        print(tabulate(dist_table, headers=dist_headers, tablefmt="grid"))

    return avg_loss, task_losses_avg, task_metrics

