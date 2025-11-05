"""
MHD_Nodet Project - Utilities Module
====================================
This module provides utility functions for the MHD_Nodet project, including training, validation, testing, and data processing helpers.
- Includes functions for training loop, validation loop, testing loop, logging, and learning rate scheduling.
- Supports consistent data handling across multi-node and multi-task setups.

项目：MHD_Nodet - 工具模块
本模块为 MHD_Nodet 项目提供实用工具函数，包括训练、验证、测试和数据处理辅助功能。
- 包含训练循环、验证循环、测试循环、日志记录和学习率调度功能。
- 支持多节点和多任务设置下的一致性数据处理。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import torch
import torch.optim as optim
import numpy as np
from tabulate import tabulate
import logging
from collections import Counter
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    """
    Cosine annealing learning rate scheduler.
    余弦退火学习率调度器。
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)

class PolynomialLR(optim.lr_scheduler._LRScheduler):
    """
    Polynomial learning rate scheduler as used in nnU-Net.
    nnU-Net 使用的多项式学习率调度器。
    """
    def __init__(self, optimizer, max_epochs, power=0.9, eta_min=0, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.max_epochs:
            return [self.eta_min for _ in self.base_lrs]
        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [base_lr * factor + self.eta_min * (1 - factor) for base_lr in self.base_lrs]

class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """
    Reduce learning rate when a metric has stopped improving.
    当指标停止改善时减少学习率。
    """
    def __init__(self, optimizer, factor=0.5, patience=10, eta_min=0, verbose=True, mode='min'):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, min_lr=eta_min, verbose=verbose)

def train(model, dataloaders, optimizer, task_configs, out_nodes, epoch, num_epochs, debug=False):
    model.train()
    running_loss = 0.0
    # Temporary storage for per-batch losses to compute average at epoch end
    temp_task_losses = {
        task: {
            (loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"])): []
            for loss_cfg in task_configs[task]["loss"]
        }
        for task in task_configs
    }
    task_metrics = {task: {} for task in task_configs}
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    class_distributions = {task: [] for task in task_configs}
    case_ids_per_batch = []

    data_iterators = {str(node): iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        inputs_list = []
        batch_case_ids = []
        
        for node in dataloaders:
            dataset = dataloaders[node].dataset
            batch_data = next(data_iterators[str(node)])
            data = batch_data.to(device)
            start_idx = batch_idx * dataloaders[node].batch_size
            end_idx = min((batch_idx + 1) * dataloaders[node].batch_size, len(dataset))
            current_case_ids = dataset.case_ids[start_idx:end_idx]
            batch_case_ids.append(current_case_ids)
            
            if data.dtype != torch.float32:
                if debug:
                    logger.info(f"Converting node {node} data from {data.dtype} to torch.float32")
                data = data.to(dtype=torch.float32)
            inputs_list.append(data)
        
        batch_case_ids_ref = batch_case_ids[0]
        if not all(cids == batch_case_ids_ref for cids in batch_case_ids):
            logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes: {batch_case_ids}")
        case_ids_per_batch.append(batch_case_ids_ref)
        
        outputs = model(inputs_list)
        total_loss = torch.tensor(0.0, device=device)

        for task, config in task_configs.items():
            task_loss = torch.tensor(0.0, device=device)
            origin_node = str(config["metric"][0]["origin_node"]) if config.get("metric") else None
            target_node = str(config["metric"][0]["target_node"]) if config.get("metric") else None
            if origin_node and target_node:
                origin_idx = out_nodes.index(origin_node)
                target_idx = out_nodes.index(target_node)
                all_preds[task].append(outputs[origin_idx].detach())
                all_targets[task].append(outputs[target_idx].detach())
                
            target_idx = out_nodes.index(str(config["loss"][0]["target_node"]))
            target_tensor = outputs[target_idx]
            class_indices = torch.argmax(target_tensor, dim=1).flatten().cpu().numpy()
            class_counts = Counter(class_indices)
            class_distributions[task].append(class_counts)
            
            for loss_cfg in config["loss"]:
                fn = loss_cfg["fn"]
                origin_node = str(loss_cfg["origin_node"])
                target_node = str(loss_cfg["target_node"])
                weight = loss_cfg["weight"]
                params = loss_cfg["params"]
                origin_idx = out_nodes.index(origin_node)
                target_idx = out_nodes.index(target_node)
                loss = fn(outputs[origin_idx], outputs[target_idx], **params)
                task_loss += weight * loss
                temp_task_losses[task][(fn.__name__, origin_node, target_node)].append(loss.item())
            total_loss += task_loss

        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += total_loss.item()

        del inputs_list, outputs, total_loss, task_loss
        torch.cuda.empty_cache()

    avg_loss = running_loss / num_batches
    # Compute final task_losses with averaged values
    task_losses = {
        task: {
            loss_cfg["fn"].__name__: {
                "origin_node": str(loss_cfg["origin_node"]),
                "target_node": str(loss_cfg["target_node"]),
                "weight": loss_cfg["weight"],
                "params": loss_cfg["params"],
                "value": np.mean(temp_task_losses[task][(loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"]))])
            }
            for loss_cfg in task_configs[task]["loss"]
        }
        for task in task_configs
    }
    task_losses_avg = {
        task: sum(loss["value"] * loss["weight"] for loss in task_losses[task].values())
        for task in task_configs
    }

    for task, config in task_configs.items():
        if config.get("metric"):
            origin_tensor = torch.cat(all_preds[task], dim=0)
            target_tensor = torch.cat(all_targets[task], dim=0)
            for metric_cfg in config["metric"]:
                fn = metric_cfg["fn"]
                origin_node = str(metric_cfg["origin_node"])
                target_node = str(metric_cfg["target_node"])
                params = metric_cfg["params"]
                metric_value = fn(origin_tensor, target_tensor, **params)
                task_metrics[task][fn.__name__] = {"origin_node": origin_node, "target_node": target_node, "value": metric_value}
            del origin_tensor, target_tensor
            torch.cuda.empty_cache()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Total Loss: {avg_loss:.4f}")
    for task, avg_task_loss in task_losses_avg.items():
        print(f"Task: {task}, Avg Loss: {avg_task_loss:.4f}")
        print(f"  Class Distribution for Task: {task}")
        total_counts = Counter()
        for batch_counts in class_distributions[task]:
            total_counts.update(batch_counts)
        dist_table = [[f"Class {cls}", count] for cls, count in sorted(total_counts.items())]
        dist_headers = ["Class", "Count"]
        print(tabulate(dist_table, headers=dist_headers, tablefmt="grid"))

        for fn_name, loss in task_losses[task].items():
            origin_node = loss["origin_node"]
            target_node = loss["target_node"]
            weight = loss["weight"]
            params_str = ", ".join(f"{k}={v}" for k, v in loss["params"].items())
            avg_loss_value = loss["value"]
            print(f"  Loss: {fn_name}({origin_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss_value:.4f}")

        for fn_name, metric in task_metrics[task].items():
            origin_node = str(metric["origin_node"])
            target_node = str(metric["target_node"])
            metric_value = metric["value"]
            valid_classes = sorted(total_counts.keys())
            headers = ["Class", fn_name.split("_")[1].capitalize()]
            table = [[f"Class {valid_classes[i]}", f"{v:.4f}" if not np.isnan(v) else "N/A"] for i, v in enumerate(metric_value["per_class"])] + [["Avg", f"{metric_value['avg']:.4f}" if not np.isnan(metric_value['avg']) else "N/A"]]
            print(f"  Metric: {fn_name}({origin_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))

    return avg_loss, task_losses, task_metrics

def validate(model, dataloaders, task_configs, out_nodes, epoch, num_epochs, debug=False):
    model.eval()
    running_loss = 0.0
    # Temporary storage for per-batch losses to compute average at epoch end
    temp_task_losses = {
        task: {
            (loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"])): []
            for loss_cfg in task_configs[task]["loss"]
        }
        for task in task_configs
    }
    task_metrics = {task: {} for task in task_configs}
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    class_distributions = {task: [] for task in task_configs}
    case_ids_per_batch = []

    with torch.no_grad():
        data_iterators = {str(node): iter(dataloader) for node, dataloader in dataloaders.items()}
        num_batches = len(next(iter(data_iterators.values())))

        for batch_idx in range(num_batches):
            inputs_list = []
            batch_case_ids = []
            
            for node in dataloaders:
                dataset = dataloaders[node].dataset
                batch_data = next(data_iterators[str(node)])
                data = batch_data.to(device)
                start_idx = batch_idx * dataloaders[node].batch_size
                end_idx = min((batch_idx + 1) * dataloaders[node].batch_size, len(dataset))
                current_case_ids = dataset.case_ids[start_idx:end_idx]
                batch_case_ids.append(current_case_ids)
                
                if data.dtype != torch.float32:
                    if debug:
                        logger.info(f"Converting node {node} data from {data.dtype} to torch.float32")
                    data = data.to(dtype=torch.float32)
                inputs_list.append(data)
            
            batch_case_ids_ref = batch_case_ids[0]
            if not all(cids == batch_case_ids_ref for cids in batch_case_ids):
                logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes: {batch_case_ids}")
            case_ids_per_batch.append(batch_case_ids_ref)
            
            outputs = model(inputs_list)
            total_loss = torch.tensor(0.0, device=device)

            for task, config in task_configs.items():
                task_loss = torch.tensor(0.0, device=device)
                origin_node = str(config["metric"][0]["origin_node"]) if config.get("metric") else None
                target_node = str(config["metric"][0]["target_node"]) if config.get("metric") else None
                if origin_node and target_node:
                    origin_idx = out_nodes.index(origin_node)
                    target_idx = out_nodes.index(target_node)
                    all_preds[task].append(outputs[origin_idx].detach())
                    all_targets[task].append(outputs[target_idx].detach())
                
                target_idx = out_nodes.index(str(config["loss"][0]["target_node"]))
                target_tensor = outputs[target_idx]
                class_indices = torch.argmax(target_tensor, dim=1).flatten().cpu().numpy()
                class_counts = Counter(class_indices)
                class_distributions[task].append(class_counts)
                
                for loss_cfg in config["loss"]:
                    fn = loss_cfg["fn"]
                    origin_node = str(loss_cfg["origin_node"])
                    target_node = str(loss_cfg["target_node"])
                    weight = loss_cfg["weight"]
                    params = loss_cfg["params"]
                    origin_idx = out_nodes.index(origin_node)
                    target_idx = out_nodes.index(target_node)
                    loss = fn(outputs[origin_idx], outputs[target_idx], **params)
                    task_loss += weight * loss
                    temp_task_losses[task][(fn.__name__, origin_node, target_node)].append(loss.item())
                total_loss += task_loss

            running_loss += total_loss.item()

            del inputs_list, outputs, total_loss, task_loss
            torch.cuda.empty_cache()

    avg_loss = running_loss / num_batches
    # Compute final task_losses with averaged values
    task_losses = {
        task: {
            loss_cfg["fn"].__name__: {
                "origin_node": str(loss_cfg["origin_node"]),
                "target_node": str(loss_cfg["target_node"]),
                "weight": loss_cfg["weight"],
                "params": loss_cfg["params"],
                "value": np.mean(temp_task_losses[task][(loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"]))])
            }
            for loss_cfg in task_configs[task]["loss"]
        }
        for task in task_configs
    }
    task_losses_avg = {
        task: sum(loss["value"] * loss["weight"] for loss in task_losses[task].values())
        for task in task_configs
    }

    for task, config in task_configs.items():
        if config.get("metric"):
            origin_tensor = torch.cat(all_preds[task], dim=0)
            target_tensor = torch.cat(all_targets[task], dim=0)
            for metric_cfg in config["metric"]:
                fn = metric_cfg["fn"]
                origin_node = str(metric_cfg["origin_node"])
                target_node = str(metric_cfg["target_node"])
                params = metric_cfg["params"]
                metric_value = fn(origin_tensor, target_tensor, **params)
                task_metrics[task][fn.__name__] = {"origin_node": origin_node, "target_node": target_node, "value": metric_value}
            del origin_tensor, target_tensor
            torch.cuda.empty_cache()

    print(f"Epoch [{epoch+1}/{num_epochs}], Val Total Loss: {avg_loss:.4f}")
    for task, avg_task_loss in task_losses_avg.items():
        print(f"Task: {task}, Avg Loss: {avg_task_loss:.4f}")
        print(f"  Class Distribution for Task: {task}")
        total_counts = Counter()
        for batch_counts in class_distributions[task]:
            total_counts.update(batch_counts)
        dist_table = [[f"Class {cls}", count] for cls, count in sorted(total_counts.items())]
        dist_headers = ["Class", "Count"]
        print(tabulate(dist_table, headers=dist_headers, tablefmt="grid"))
        
        for fn_name, loss in task_losses[task].items():
            origin_node = loss["origin_node"]
            target_node = loss["target_node"]
            weight = loss["weight"]
            params_str = ", ".join(f"{k}={v}" for k, v in loss["params"].items())
            avg_loss_value = loss["value"]
            print(f"  Loss: {fn_name}({origin_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss_value:.4f}")

        for fn_name, metric in task_metrics[task].items():
            origin_node = str(metric["origin_node"])
            target_node = str(metric["target_node"])
            metric_value = metric["value"]
            valid_classes = sorted(total_counts.keys())
            headers = ["Class", fn_name.split("_")[1].capitalize()]
            table = [[f"Class {valid_classes[i]}", f"{v:.4f}" if not np.isnan(v) else "N/A"] for i, v in enumerate(metric_value["per_class"])] + [["Avg", f"{metric_value['avg']:.4f}" if not np.isnan(metric_value['avg']) else "N/A"]]
            print(f"  Metric: {fn_name}({origin_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))

    return avg_loss, task_losses, task_metrics

def test(model, dataloaders, out_nodes, save_node, save_dir, debug=False):
    """
    Testing function to generate and save predictions for specified nodes.
    - Processes data in batches, consistent with train and validate functions.
    - Saves predictions to specified files in save_dir.
    - Does not compute losses or metrics.

    测试函数，用于生成并保存指定节点的预测结果。
    - 按批次处理数据，与 train 和 validate 函数保持一致。
    - 将预测结果保存到 save_dir 中的指定文件。
    - 不计算损失或指标。
    """
    model.eval()
    case_ids_per_batch = []

    data_iterators = {str(node): iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs_list = []
            batch_case_ids = []

            for node in dataloaders:
                dataset = dataloaders[node].dataset
                batch_data = next(data_iterators[str(node)])
                data = batch_data.to(device)
                start_idx = batch_idx * dataloaders[node].batch_size
                end_idx = min((batch_idx + 1) * dataloaders[node].batch_size, len(dataset))
                current_case_ids = dataset.case_ids[start_idx:end_idx]
                batch_case_ids.append(current_case_ids)

                if data.dtype != torch.float32:
                    if debug:
                        logger.info(f"Converting node {node} data from {data.dtype} to torch.float32")
                    data = data.to(dtype=torch.float32)
                inputs_list.append(data)

            batch_case_ids_ref = batch_case_ids[0]
            if not all(cids == batch_case_ids_ref for cids in batch_case_ids):
                logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes: {batch_case_ids}")
            case_ids_per_batch.append(batch_case_ids_ref)

            outputs = model(inputs_list)

            # Save predictions for specified nodes
            for node, filename in save_node:
                node_idx = out_nodes.index(str(node))
                predictions = outputs[node_idx].detach().cpu().numpy()
                for idx, case_id in enumerate(batch_case_ids_ref):
                    save_path = os.path.join(save_dir, f"case_{case_id}_{filename}")
                    np.save(save_path, predictions[idx])
                    logger.debug(f"Saved prediction for node {node}, case {case_id} to {save_path}")

            # Clear intermediate tensors
            del inputs_list, outputs
            torch.cuda.empty_cache()

    logger.info(f"Processed {num_batches} batches, saved predictions to {save_dir}")
