import torch
import numpy as np
from tabulate import tabulate
import logging
from collections import Counter
from torch.utils.data import DataLoader

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_node_dtype_mapping(node_mapping, sub_networks):
    """
    预计算全局节点的数据类型映射。
    """
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

def train(model, dataloaders, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, debug=False):
    """
    训练模型一个 epoch。
    """
    model.train()
    running_loss = 0.0
    task_losses = {task: [] for task in task_configs}
    # 存储每个任务的case IDs和类别分布
    case_ids_per_batch = []
    class_distributions = {task: [] for task in task_configs}

    # 预计算节点数据类型
    node_dtype_mapping = get_node_dtype_mapping(node_mapping, sub_networks)
    
    data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        inputs_list = []
        batch_case_ids = []
        
        # 收集输入数据和case IDs
        for node in dataloaders:
            dataset = dataloaders[node].dataset
            batch_data = next(data_iterators[node])
            data = batch_data.to(device)
            # 获取当前batch的case IDs
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
        
        # 确保case IDs一致
        batch_case_ids_set = set(batch_case_ids[0])
        if not all(set(cids) == batch_case_ids_set for cids in batch_case_ids):
            logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes")
        case_ids_per_batch.append(list(batch_case_ids_set))
        
        outputs = model(inputs_list)
        total_loss = torch.tensor(0.0, device=device)

        # 收集类别分布
        for task, config in task_configs.items():
            task_loss = torch.tensor(0.0, device=device)
            src_node = config["loss"][0]["src_node"]
            target_node = config["loss"][0]["target_node"]
            target_idx = out_nodes.index(target_node)
            target_tensor = outputs[target_idx]  # [batch_size, C, *S]
            # 转换为类别索引
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
                task_losses[task].append(loss.item())
            total_loss += task_loss

        total_loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1000 or grad_norm < 1e-6:
                    print(f"Warning: {name} grad norm = {grad_norm}")
        optimizer.step()
        running_loss += total_loss.item()

    avg_loss = running_loss / num_batches
    task_losses_avg = {task: np.mean(losses) for task, losses in task_losses.items()}

    # 打印训练信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Total Loss: {avg_loss:.4f}")
    for task, losses in task_losses_avg.items():
        print(f"Task: {task}, Avg Loss: {losses:.4f}")
        for loss_cfg in task_configs[task]["loss"]:
            fn_name = loss_cfg["fn"].__name__
            src_node = loss_cfg["src_node"]
            target_node = loss_cfg["target_node"]
            weight = loss_cfg["weight"]
            params_str = ", ".join(f"{k}={v}" for k, v in loss_cfg["params"].items())
            # 查找特定损失的值
            loss_values = [l for l, cfg in zip(task_losses[task], task_configs[task]["loss"]) if cfg["fn"] == loss_cfg["fn"] and cfg["src_node"] == src_node and cfg["target_node"] == target_node]
            avg_loss = np.mean(loss_values) if loss_values else losses
            print(f"  Loss: {fn_name}({src_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss:.4f}")
    return avg_loss, task_losses_avg, {}

def validate(model, dataloaders, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, debug=False):
    """
    验证模型一个 epoch。
    """
    model.eval()
    running_loss = 0.0
    task_losses = {task: [] for task in task_configs}
    task_metrics = {task: [] for task in task_configs}
    # 存储所有batch的预测和目标
    all_preds = {task: [] for task in task_configs}
    all_targets = {task: [] for task in task_configs}
    # 存储每个任务的case IDs和类别分布
    case_ids_per_batch = []
    class_distributions = {task: [] for task in task_configs}

    # 预计算节点数据类型
    node_dtype_mapping = get_node_dtype_mapping(node_mapping, sub_networks)

    with torch.no_grad():
        data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
        num_batches = len(next(iter(data_iterators.values())))

        for batch_idx in range(num_batches):
            inputs_list = []
            batch_case_ids = []
            
            # 收集输入数据和case IDs
            for node in dataloaders:
                dataset = dataloaders[node].dataset
                batch_data = next(data_iterators[node])
                data = batch_data.to(device)
                # 获取当前batch的case IDs
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
            
            # 确保case IDs一致
            batch_case_ids_set = set(batch_case_ids[0])
            if not all(set(cids) == batch_case_ids_set for cids in batch_case_ids):
                logger.warning(f"Batch {batch_idx} case IDs inconsistent across nodes")
            case_ids_per_batch.append(list(batch_case_ids_set))
            
            outputs = model(inputs_list)
            total_loss = torch.tensor(0.0, device=device)

            # 收集类别分布
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
                target_tensor = outputs[target_idx]  # [batch_size, C, *S]
                # 转换为类别索引
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
                    task_losses[task].append(loss.item())
                total_loss += task_loss

            running_loss += total_loss.item()

    avg_loss = running_loss / num_batches
    task_losses_avg = {task: np.mean(losses) for task, losses in task_losses.items()}

    # 计算指标（基于所有batch）
    for task, config in task_configs.items():
        metrics = []
        if config.get("metric"):
            # 拼接所有batch的预测和目标
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

    # 打印验证信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Total Loss: {avg_loss:.4f}")
    for task, losses in task_losses_avg.items():
        print(f"Task: {task}, Avg Loss: {losses:.4f}")
        for loss_cfg in task_configs[task]["loss"]:
            fn_name = loss_cfg["fn"].__name__
            src_node = loss_cfg["src_node"]
            target_node = loss_cfg["target_node"]
            weight = loss_cfg["weight"]
            params_str = ", ".join(f"{k}={v}" for k, v in loss_cfg["params"].items())
            # 查找特定损失的值
            loss_values = [l for l, cfg in zip(task_losses[task], task_configs[task]["loss"]) if cfg["fn"] == loss_cfg["fn"] and cfg["src_node"] == src_node and cfg["target_node"] == target_node]
            avg_loss = np.mean(loss_values) if loss_values else losses
            print(f"  Loss: {fn_name}({src_node}, {target_node}), Weight: {weight:.2f}, Params: {params_str}, Value: {avg_loss:.4f}")

        for metric in task_metrics[task]:
            fn_name = metric["fn"]
            src_node = metric["src_node"]
            target_node = metric["target_node"]
            result = metric["result"]
            headers = ["Class", metric["fn"].split("_")[1].capitalize()]
            table = [[f"Class {i}", f"{v:.4f}"] for i, v in enumerate(result["per_class"])] + [["Avg", f"{result['avg']:.4f}"]]
            print(f"  Metric: {fn_name}({src_node}, {target_node})")
            print(tabulate(table, headers=headers, tablefmt="grid"))

    return avg_loss, task_losses_avg, task_metrics
