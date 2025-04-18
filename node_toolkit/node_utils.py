import torch
import numpy as np
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloaders, optimizer, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping):
    model.train()
    running_loss = 0.0
    task_losses = {task: [] for task in task_configs}
    task_metrics = {task: [] for task in task_configs}

    data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
    num_batches = len(next(iter(data_iterators.values())))

    for _ in range(num_batches):
        optimizer.zero_grad()
        inputs_list = [next(data_iterators[node]).to(device) for node in dataloaders]
        # inputs_list 形状为 [N, C, *S]
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

def validate(model, dataloaders, task_configs, out_nodes, epoch, num_epochs, sub_networks, node_mapping):
    model.eval()
    running_loss = 0.0
    task_losses = {task: [] for task in task_configs}
    task_metrics = {task: [] for task in task_configs}

    with torch.no_grad():
        data_iterators = {node: iter(dataloader) for node, dataloader in dataloaders.items()}
        num_batches = len(next(iter(data_iterators.values())))

        for _ in range(num_batches):
            inputs_list = [next(data_iterators[node]).to(device) for node in dataloaders]
            # inputs_list 形状为 [N, C, *S]
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
