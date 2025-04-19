import torch
import numpy as np
from tabulate import tabulate
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_node_dtype_mapping(node_mapping, sub_networks):
    """
    预计算全局节点的数据类型映射。

    Args:
        node_mapping: 节点映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        sub_networks: 子网络字典，键为网络名称，值为 HDNet 实例。

    Returns:
        Dict[int, torch.dtype]: 全局节点 ID 到目标数据类型的映射。
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

    Args:
        model: 待训练的模型（MHDNet 实例）。
        dataloaders: 数据加载器字典，键为节点 ID，值为 DataLoader。
        optimizer: 优化器。
        task_configs: 任务配置字典，包含损失和指标配置。
        out_nodes: 全局输出节点 ID 列表。
        epoch: 当前 epoch。
        num_epochs: 总 epoch 数。
        sub_networks: 子网络字典。
        node_mapping: 节点映射。
        debug: 是否启用调试日志（默认 False）。

    Returns:
        tuple: (平均总损失, 任务平均损失字典, 任务指标字典)。
    """
    model.train()
    running_loss = 0.0
    task_losses = {task: [] for task in task_configs}
    task_metrics = {task: [] for task in task_configs}

    # 预计算节点数据类型
    node_dtype_mapping = get_node_dtype_mapping(node_mapping, sub_networks)
    
    data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    for _ in range(num_batches):
        optimizer.zero_grad()
        inputs_list = []
        for node in dataloaders:
            data = next(data_iterators[node]).to(device)
            # 获取目标数据类型
            expected_dtype = node_dtype_mapping.get(node, torch.float32)
            if data.dtype != expected_dtype:
                if debug:
                    logger.info(f"Converting node {node} data from {data.dtype} to {expected_dtype}")
                data = data.to(dtype=expected_dtype)
            inputs_list.append(data)
        
        outputs = model(inputs_list)
        total_loss = torch.tensor(0.0, device=device)

        for task, config in task_configs.items():
            task_loss = torch.tensor(0.0, device=device)
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

    # 计算指标
    for task, config in task_configs.items():
        metrics = []
        for metric_cfg in config.get("metric", []):
            fn = metric_cfg["fn"]
            src_node = metric_cfg["src_node"]
            target_node = metric_cfg["target_node"]
            params = metric_cfg["params"]
            src_idx = out_nodes.index(src_node)
            target_idx = out_nodes.index(target_node)
            result = fn(outputs[src_idx], outputs[target_idx], **params)
            metrics.append({"fn": fn.__name__, "src_node": src_node, "target_node": target_node, "result": result})
        task_metrics[task] = metrics

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
            avg_loss = np.mean([l for l in task_losses[task]])
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

def validate(model, dataloaders, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping, debug=False):
    """
    验证模型一个 epoch。

    Args:
        model: 待验证的模型（MHDNet 实例）。
        dataloaders: 数据加载器字典，键为节点 ID，值为 DataLoader。
        task_configs: 任务配置字典，包含损失和指标配置。
        out_nodes: 全局输出节点 ID 列表。
        epoch: 当前 epoch。
        num_epochs: 总 epoch 数。
        sub_networks: 子网络字典。
        node_mapping: 节点映射。
        debug: 是否启用调试日志（默认 False）。

    Returns:
        tuple: (平均总损失, 任务平均损失字典, 任务指标字典)。
    """
    model.eval()
    running_loss = 0.0
    task_losses = {task: [] for task in task_configs}
    task_metrics = {task: [] for task in task_configs}

    # 预计算节点数据类型
    node_dtype_mapping = get_node_dtype_mapping(node_mapping, sub_networks)

    with torch.no_grad():
        data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
        num_batches = len(next(iter(data_iterators.values())))

        for _ in range(num_batches):
            inputs_list = []
            for node in dataloaders:
                data = next(data_iterators[node]).to(device)
                # 获取目标数据类型
                expected_dtype = node_dtype_mapping.get(node, torch.float32)
                if data.dtype != expected_dtype:
                    if debug:
                        logger.info(f"Converting node {node} data from {data.dtype} to {expected_dtype}")
                    data = data.to(dtype=expected_dtype)
                inputs_list.append(data)
            
            outputs = model(inputs_list)
            total_loss = torch.tensor(0.0, device=device)

            for task, config in task_configs.items():
                task_loss = torch.tensor(0.0, device=device)
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

    # 计算指标
    for task, config in task_configs.items():
        metrics = []
        for metric_cfg in config.get("metric", []):
            fn = metric_cfg["fn"]
            src_node = metric_cfg["src_node"]
            target_node = metric_cfg["target_node"]
            params = metric_cfg["params"]
            src_idx = out_nodes.index(src_node)
            target_idx = out_nodes.index(target_node)
            result = fn(outputs[src_idx], outputs[target_idx], **params)
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
            avg_loss = np.mean([l for l in task_losses[task]])
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
