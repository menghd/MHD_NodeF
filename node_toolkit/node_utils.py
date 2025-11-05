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
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)


class PolynomialLR(optim.lr_scheduler._LRScheduler):
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
    def __init__(self, optimizer, factor=0.5, patience=10, eta_min=0, verbose=True, mode='min'):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, min_lr=eta_min, verbose=verbose)


def _get_batch_inputs(model, dataloaders, data_iterators, batch_idx):
    """按 model.in_nodes 顺序同步获取 batch"""
    inputs_dict = {}
    batch_case_ids_list = []
    for node in model.in_nodes:
        if node not in dataloaders:
            raise ValueError(f"节点 {node} 没有 DataLoader")
        dataset = dataloaders[node].dataset
        batch_data = next(data_iterators[str(node)])
        data = batch_data.to(device)
        start_idx = batch_idx * dataloaders[node].batch_size
        end_idx = min((batch_idx + 1) * dataloaders[node].batch_size, len(dataset))
        current_case_ids = dataset.case_ids[start_idx:end_idx]
        batch_case_ids_list.append(current_case_ids)
        if data.dtype != torch.float32:
            data = data.to(dtype=torch.float32)
        inputs_dict[node] = data
    if batch_case_ids_list:
        ref = batch_case_ids_list[0]
        if not all(cids == ref for cids in batch_case_ids_list):
            logger.warning(f"Batch {batch_idx} case IDs 不一致")
    return inputs_dict, ref if batch_case_ids_list else []


def train(model, dataloaders, optimizer, task_configs, out_nodes, epoch, num_epochs, debug=False):
    model.train()
    running_loss = 0.0
    temp_task_losses = {task: {} for task in task_configs}
    for task in task_configs:
        for loss_cfg in task_configs[task]["loss"]:
            key = (loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"]))
            temp_task_losses[task][key] = []

    task_metrics = {task: {} for task in task_configs}
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    class_distributions = {task: [] for task in task_configs}
    case_ids_per_batch = []

    data_iterators = {str(node): iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        inputs_dict, batch_case_ids = _get_batch_inputs(model, dataloaders, data_iterators, batch_idx)
        case_ids_per_batch.append(batch_case_ids)

        outputs = model(inputs_dict)
        output_dict = {node: out for node, out in zip(model.out_nodes, outputs)}

        total_loss = torch.tensor(0.0, device=device)
        for task, config in task_configs.items():
            task_loss = torch.tensor(0.0, device=device)
            if config.get("metric"):
                origin_node = str(config["metric"][0]["origin_node"])
                target_node = str(config["metric"][0]["target_node"])
                all_preds[task].append(output_dict[origin_node].detach())
                all_targets[task].append(inputs_dict[target_node].detach())

            target_node = str(config["loss"][0]["target_node"])
            target_tensor = inputs_dict[target_node]
            class_indices = torch.argmax(target_tensor, dim=1).flatten().cpu().numpy()
            class_distributions[task].append(Counter(class_indices))

            for loss_cfg in config["loss"]:
                origin = output_dict[loss_cfg["origin_node"]]
                target = inputs_dict[loss_cfg["target_node"]]
                loss = loss_cfg["fn"](origin, target, **loss_cfg["params"])
                task_loss += loss_cfg["weight"] * loss
                key = (loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"]))
                temp_task_losses[task][key].append(loss.item())
            total_loss += task_loss

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        del inputs_dict, outputs, total_loss
        torch.cuda.empty_cache()

    avg_loss = running_loss / num_batches
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
        print(f" Class Distribution for Task: {task}")
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
            print(f" Loss: {fn_name}({origin_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss_value:.4f}")
        for fn_name, metric in task_metrics[task].items():
            origin_node = str(metric["origin_node"])
            target_node = str(metric["target_node"])
            metric_value = metric["value"]
            valid_classes = sorted(total_counts.keys())
            headers = ["Class", fn_name.split("_")[1].capitalize()]
            table = [[f"Class {valid_classes[i]}", f"{v:.4f}" if not np.isnan(v) else "N/A"] for i, v in enumerate(metric_value["per_class"])] + [["Avg", f"{metric_value['avg']:.4f}" if not np.isnan(metric_value['avg']) else "N/A"]]
            print(f" Metric: {fn_name}({origin_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))
    return avg_loss, task_losses, task_metrics


def validate(model, dataloaders, task_configs, out_nodes, epoch, num_epochs, debug=False):
    model.eval()
    running_loss = 0.0
    temp_task_losses = {task: {} for task in task_configs}
    for task in task_configs:
        for loss_cfg in task_configs[task]["loss"]:
            key = (loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"]))
            temp_task_losses[task][key] = []

    task_metrics = {task: {} for task in task_configs}
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    class_distributions = {task: [] for task in task_configs}
    case_ids_per_batch = []

    data_iterators = {str(node): iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs_dict, batch_case_ids = _get_batch_inputs(model, dataloaders, data_iterators, batch_idx)
            case_ids_per_batch.append(batch_case_ids)

            outputs = model(inputs_dict)
            output_dict = {node: out for node, out in zip(model.out_nodes, outputs)}

            total_loss = torch.tensor(0.0, device=device)
            for task, config in task_configs.items():
                task_loss = torch.tensor(0.0, device=device)
                if config.get("metric"):
                    origin_node = str(config["metric"][0]["origin_node"])
                    target_node = str(config["metric"][0]["target_node"])
                    all_preds[task].append(output_dict[origin_node].detach())
                    all_targets[task].append(inputs_dict[target_node].detach())

                target_node = str(config["loss"][0]["target_node"])
                target_tensor = inputs_dict[target_node]
                class_indices = torch.argmax(target_tensor, dim=1).flatten().cpu().numpy()
                class_distributions[task].append(Counter(class_indices))

                for loss_cfg in config["loss"]:
                    origin = output_dict[loss_cfg["origin_node"]]
                    target = inputs_dict[loss_cfg["target_node"]]
                    loss = loss_cfg["fn"](origin, target, **loss_cfg["params"])
                    task_loss += loss_cfg["weight"] * loss
                    key = (loss_cfg["fn"].__name__, str(loss_cfg["origin_node"]), str(loss_cfg["target_node"]))
                    temp_task_losses[task][key].append(loss.item())
                total_loss += task_loss

            running_loss += total_loss.item()
            del inputs_dict, outputs, total_loss
            torch.cuda.empty_cache()

    avg_loss = running_loss / num_batches
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
        print(f" Class Distribution for Task: {task}")
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
            print(f" Loss: {fn_name}({origin_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss_value:.4f}")
        for fn_name, metric in task_metrics[task].items():
            origin_node = str(metric["origin_node"])
            target_node = str(metric["target_node"])
            metric_value = metric["value"]
            valid_classes = sorted(total_counts.keys())
            headers = ["Class", fn_name.split("_")[1].capitalize()]
            table = [[f"Class {valid_classes[i]}", f"{v:.4f}" if not np.isnan(v) else "N/A"] for i, v in enumerate(metric_value["per_class"])] + [["Avg", f"{metric_value['avg']:.4f}" if not np.isnan(metric_value['avg']) else "N/A"]]
            print(f" Metric: {fn_name}({origin_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))
    return avg_loss, task_losses, task_metrics


def test(model, dataloaders, out_nodes, save_node, save_dir, debug=False):
    model.eval()
    data_iterators = {str(node): iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    with torch.no_grad():
        for batch_idx in range(num_batches):
            inputs_dict, batch_case_ids = _get_batch_inputs(model, dataloaders, data_iterators, batch_idx)
            outputs = model(inputs_dict)
            output_dict = {node: out for node, out in zip(model.out_nodes, outputs)}

            for node, filename in save_node:
                pred = output_dict[node].detach().cpu().numpy()
                for idx, case_id in enumerate(batch_case_ids):
                    save_path = os.path.join(save_dir, f"case_{case_id}_{filename}")
                    np.save(save_path, pred[idx])
                    logger.debug(f"Saved prediction for node {node}, case {case_id} to {save_path}")

    logger.info(f"Processed {num_batches} batches, saved predictions to {save_dir}")
