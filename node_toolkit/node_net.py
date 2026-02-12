"""
MHD_Nodet Project - Neural Network Module
=======================================
This module defines the core neural network architectures for the MHD_Nodet project, including DNet, HDNet, and MHDNet.
- DNet: A dynamic convolutional network supporting multi-layer convolution, normalization, and activation with flexible dimensionality.
- HDNet: A hyperedge-based network for dynamic node connections and feature processing.
- MHDNet: A multi-subgraph network integrating multiple HDNet instances for global node processing.

项目：MHD_Nodet - 神经网络模块
本模块定义了 MHD_Nodet 项目的核心神经网络架构，包括 DNet、HDNet 和 MHDNet。
- DNet：动态卷积网络，支持多层卷积、归一化和激活，适应不同维度。
- HDNet：基于超边的网络，支持动态节点连接和特征处理。
- MHDNet：多子图网络，集成多个 HDNet 实例以处理全局节点。

Author: Souray Meng (孟号丁)
Email: souray@qq.com
Institution: Tsinghua University (清华大学)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union, Callable
import warnings

def validate_lengths(*args, names: List[str]):
    """验证参数长度是否一致"""
    lengths = [len(arg) for arg in args]
    if len(set(lengths)) > 1:
        raise ValueError(f"参数长度不一致: {dict(zip(names, lengths))}")

class DNet(nn.Module):
    """动态网络，支持多层卷积、归一化和激活，保持空间形状不变。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        num_dimensions (int): 维度（1D、2D 或 3D）。
        convs (List[Union[torch.Size, torch.Tensor]]): 每层卷积配置，格式为 torch.Size([out_channels, in_channels, kernel_size...]) 或 torch.Tensor。
        reqs (Optional[List[bool]]): 每层卷积是否可学习，True 表示可学习，False 表示不可学习。
        norms (Optional[List[Optional[str]]]): 每层归一化类型，None 表示无归一化。
        acts (Optional[List[Optional[str]]]): 每层激活函数类型，None 表示无激活。
    """
    NORM_TYPES = {
        "instance": lambda dim, ch: getattr(nn, f"InstanceNorm{dim}d")(ch),
        "batch": lambda dim, ch: getattr(nn, f"BatchNorm{dim}d")(ch),
    }
    ACT_TYPES = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leakyrelu": lambda: nn.LeakyReLU(0.1, inplace=True),
        "sigmoid": lambda: nn.Sigmoid(),
        "softmax": lambda: nn.Softmax(dim=1),
        "gelu": lambda: nn.GELU(),
        "swish": lambda: nn.SiLU(),
        "tanh": lambda: nn.Tanh(),
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dimensions: int,
        convs: List[Union[torch.Size, torch.Tensor]],
        reqs: Optional[List[bool]] = None,
        norms: Optional[List[Optional[str]]] = None,
        acts: Optional[List[Optional[str]]] = None,
    ):
        super().__init__()
        self.num_dimensions = num_dimensions
        conv_layer = getattr(nn, f"Conv{num_dimensions}d")
        layers = []

        if not convs:
            raise ValueError("convs 不能为空，必须提供至少一个卷积配置")

        norms = norms if norms is not None else [None] * len(convs)
        acts = acts if acts is not None else [None] * len(convs)
        reqs = reqs if reqs is not None else [True] * len(convs)

        validate_lengths(convs, norms, acts, reqs, names=["convs", "norms", "acts", "reqs"])

        current_channels = in_channels
        for i, (conv_config, req, norm_type, act_type) in enumerate(zip(convs, reqs, norms, acts)):
            if isinstance(conv_config, torch.Size):
                conv_out_channels, conv_in_channels, *kernel_size = conv_config
                weight = None
            elif isinstance(conv_config, torch.Tensor):
                conv_out_channels, conv_in_channels, *kernel_size = conv_config.shape
                weight = conv_config
                if weight.shape != (conv_out_channels, conv_in_channels, *kernel_size):
                    raise ValueError(f"第 {i} 层卷积核形状不正确: {weight.shape}")
            else:
                raise ValueError(f"第 {i} 层卷积配置类型不支持: {type(conv_config)}")

            if conv_in_channels != current_channels:
                warnings.warn(
                    f"第 {i} 层输入通道数不匹配：实际 {current_channels}，预期 {conv_in_channels}，"
                    f"将插入线性卷积调整通道数至 {conv_in_channels}"
                )
                layers.append(
                    self.add_channel_adjustment_conv(
                        conv_layer, current_channels, conv_in_channels
                    )
                )
                current_channels = conv_in_channels

            padding = tuple(k // 2 for k in kernel_size)
            conv = conv_layer(
                conv_in_channels,
                conv_out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
            if weight is not None:
                conv.weight = nn.Parameter(weight, requires_grad=req)
            else:
                conv.weight.requires_grad = req
            layers.append(conv)
            current_channels = conv_out_channels

            if norm_type:
                norm_type = norm_type.lower()
                if norm_type not in self.NORM_TYPES:
                    raise ValueError(f"第 {i} 层不支持的归一化类型: {norm_type}")
                norm_layer = self.NORM_TYPES[norm_type](self.num_dimensions, current_channels)
                # Freeze norm layer parameters if req is False
                for param in norm_layer.parameters():
                    param.requires_grad = req
                layers.append(norm_layer)

            if act_type:
                act_type = act_type.lower()
                if act_type not in self.ACT_TYPES:
                    raise ValueError(f"第 {i} 层不支持的激活函数: {act_type}")
                layers.append(self.ACT_TYPES[act_type]())

        if current_channels != out_channels:
            layers.append(
                self.add_channel_adjustment_conv(
                    conv_layer, current_channels, out_channels
                )
            )

        self.filter = nn.Sequential(*layers)

    def add_channel_adjustment_conv(self, conv_layer, in_channels: int, out_channels: int) -> nn.Module:
        return conv_layer(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)

class HDNet(nn.Module):
    """基于超边的网络，支持动态维度和灵活的节点连接。

    Args:
        node_configs (Dict[str, Tuple[int, ...]]): 节点配置，格式为 {node_id: (channels, size...)}。
        hyperedge_configs (Dict[str, Dict]): 超边配置，包含 src_nodes, dst_nodes 和 params。
        num_dimensions (int): 维度（1D、2D 或 3D）。
        agg_mode (str): 聚合方式，支持add/mul/mean/max，默认add
    """
    def __init__(
        self,
        node_configs: Dict[str, Tuple[int, ...]],
        hyperedge_configs: Dict[str, Dict],
        num_dimensions: int = 2,
        agg_mode: str = "add",  # 新增：聚合方式，支持add/mul/mean/max
    ):
        super().__init__()
        self.node_configs = node_configs
        self.hyperedge_configs = hyperedge_configs
        self.num_dims = num_dimensions
        self.edges = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)
        self.node_order = []
        # 新增：定义聚合函数
        self.agg_func = self._get_agg_func(agg_mode)  # 绑定聚合函数

        self._build_hyperedges()
        self._topological_sort()

    def _compute_edge_channels(self, src_nodes: List[str], dst_nodes: List[str]) -> Tuple[int, int]:
        in_channels = sum(self.node_configs[src][0] for src in src_nodes)
        out_channels = sum(self.node_configs[dst][0] for dst in dst_nodes)
        return in_channels, out_channels

    def _dropout_layer(self):
        return getattr(nn, f"Dropout{self.num_dims}d")

    def _build_hyperedges(self):
        dropout_layer = self._dropout_layer()
        for edge_id, edge_config in self.hyperedge_configs.items():
            src_nodes = edge_config.get("src_nodes", [])
            dst_nodes = edge_config.get("dst_nodes", [])
            params = edge_config.get("params", {})
            dnet = edge_config.get("dnet")  # 检查是否提供 DNet 实例
            in_channels, out_channels = self._compute_edge_channels(src_nodes, dst_nodes)
            dropout_rate = params.get("dropout", 0.0)

            if dnet is None:
                convs = params.get("convs")
                if not convs:
                    raise ValueError(f"超边 {edge_id} 的 convs 不能为空，必须提供卷积配置")
                reqs = params.get("reqs", [True] * len(convs))
                norms = params.get("norms")
                acts = params.get("acts")
                self.edges[edge_id] = DNet(
                    in_channels, out_channels, self.num_dims, convs, reqs, norms, acts
                )
            else:
                self.edges[edge_id] = dnet  # 直接使用传入的 DNet 实例

            if dropout_rate > 0.0:
                self.dropouts[edge_id] = dropout_layer(dropout_rate)
            for src in src_nodes:
                self.out_edges[src].append(edge_id)
            for dst in dst_nodes:
                self.in_edges[dst].append(edge_id)

    def _get_interpolate_mode(self, mode: str) -> str:
        mode_map = {
            1: {"linear": "linear", "nearest": "nearest"},
            2: {"linear": "bilinear", "nearest": "nearest"},
            3: {"linear": "trilinear", "nearest": "nearest"},
        }
        return mode_map[self.num_dims][mode]

    def _interpolate(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, ...]],
        mode: str,
    ) -> torch.Tensor:
        if not target_size or x.shape[2:] == tuple(target_size):
            return x.to(dtype=torch.float32)

        mode = mode.lower()
        if mode in ("max", "avg"):
            pool_layer = getattr(nn, f"Adaptive{mode.title()}Pool{self.num_dims}d")
            return pool_layer(target_size)(x)
        if mode in ("nearest", "linear"):
            return F.interpolate(
                x,
                size=target_size,
                mode=self._get_interpolate_mode(mode),
                align_corners=mode == "linear",
            )
        raise ValueError(f"不支持的插值类型: {mode}")

    def _topological_sort(self):
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set(self.node_configs.keys())

        for edge_id, edge_config in self.hyperedge_configs.items():
            src_nodes = edge_config.get("src_nodes", [])
            dst_nodes = edge_config.get("dst_nodes", [])
            for src in src_nodes:
                for dst in dst_nodes:
                    graph[src].append(dst)
                    in_degree[dst] += 1

        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        sorted_nodes = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(all_nodes):
            raise ValueError("超图中存在环，无法进行拓扑排序")
        self.node_order = sorted_nodes

    def _get_agg_func(self, mode: str) -> Callable:
        mode = mode.lower()
        if mode == "add":
            # 等价于原始代码的 sum(node_inputs)
            # 逻辑：将列表中的tensor堆叠后，在第0维求和（合并多超边的输入）
            def add_agg(x: List[torch.Tensor]) -> torch.Tensor:
                stacked = torch.stack(x, dim=0)  # [num_inputs, batch, channel, H, W, D]
                agg = torch.sum(stacked, dim=0)  # [batch, channel, H, W, D]
                return agg
            return add_agg
    
        elif mode == "mul":
            # 逻辑：将列表中的tensor堆叠后，在第0维逐元素相乘
            # 注意：相乘会放大/缩小数值，建议先归一化（如LayerNorm）
            def mul_agg(x: List[torch.Tensor]) -> torch.Tensor:
                stacked = torch.stack(x, dim=0)  # [num_inputs, batch, channel, H, W, D]
                agg = torch.prod(stacked, dim=0)  # [batch, channel, H, W, D]
                return agg
            return mul_agg
    
        elif mode == "mean":
            # 逻辑：将列表中的tensor堆叠后，在第0维求均值（等价于add后除以输入数量）
            def mean_agg(x: List[torch.Tensor]) -> torch.Tensor:
                stacked = torch.stack(x, dim=0)  # [num_inputs, batch, channel, H, W, D]
                agg = torch.mean(stacked, dim=0)  # [batch, channel, H, W, D]
                return agg
            return mean_agg
    
        elif mode == "max":
            # 逻辑：将列表中的tensor堆叠后，在第0维取最大值（保留多超边中最大的特征值）
            # 注意：torch.max返回 (values, indices)，取第0个元素是值
            def max_agg(x: List[torch.Tensor]) -> torch.Tensor:
                stacked = torch.stack(x, dim=0)  # [num_inputs, batch, channel, H, W, D]
                agg, _ = torch.max(stacked, dim=0)  # [batch, channel, H, W, D]
                return agg
            return max_agg
    
        else:
            raise ValueError(
                f"不支持的聚合方式：{mode}，可选值：add/mul/mean/max\n"
                f"add=求和（默认，和原始代码一致）\n"
                f"mul=逐元素相乘\n"
                f"mean=求均值\n"
                f"max=取最大值"
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = {node: tensor.to(dtype=torch.float32) for node, tensor in inputs.items()}
        for node in self.node_order:
            if node in features:
                continue

            in_edge_ids = self.in_edges[node]
            if not in_edge_ids:
                raise ValueError(f"节点 {node} 没有输入超边")

            node_inputs = []
            for edge_id in in_edge_ids:
                edge_config = self.hyperedge_configs[edge_id]
                src_nodes = edge_config.get("src_nodes", [])
                dst_nodes = edge_config.get("dst_nodes", [])
                params = edge_config.get("params", {})
                feature_size = params.get("feature_size")
                intp = params.get("intp", "linear")
                reshape_size = params.get("reshape", feature_size)
                permute_order = params.get("permute", tuple(range(self.num_dims + 2)))

                if not all(src in features for src in src_nodes):
                    raise ValueError(f"超边 {edge_id} 的源节点 {src_nodes} 中某些节点未准备好")

                src_features = [
                    self._interpolate(features[src], feature_size, intp)
                    for src in src_nodes
                ]
                input_feat = torch.cat(src_features, dim=1)
                output = self.edges[edge_id](input_feat)

                if edge_id in self.dropouts:
                    output = self.dropouts[edge_id](output)

                if reshape_size and output.shape[2:] != tuple(reshape_size):
                    output = output.reshape(-1, output.shape[1], *reshape_size)

                if permute_order != tuple(range(self.num_dims + 2)):
                    output = output.permute(*permute_order)

                channel_sizes = [self.node_configs[dst][0] for dst in dst_nodes]
                split_outputs = torch.split(output, channel_sizes, dim=1)
                dst_features = {
                    dst: self._interpolate(feat, self.node_configs[dst][1:], intp)
                    for dst, feat in zip(dst_nodes, split_outputs)
                }

                if node in dst_features:
                    node_inputs.append(dst_features[node])

            if not node_inputs:
                raise ValueError(f"节点 {node} 没有有效输入")
            # 使用聚合函数替代原来的 sum
            features[node] = self.agg_func(node_inputs).to(dtype=torch.float32)

        return features

class MHDNet(nn.Module):
    """多子超图网络，将所有子网络整合为一个全局超图。

    Args:
        sub_networks: 子网络，键为网络名称，值为 HDNet 实例。
        node_mapping: 节点映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        in_nodes: 全局输入节点 ID 列表。
        out_nodes: 全局输出节点 ID 列表。
        num_dimensions: 维度（1D、2D 或 3D）。
        onnx_save_path: 可选的 ONNX 模型保存路径。
    """
    def __init__(
        self,
        sub_networks: Dict[str, nn.Module],
        node_mapping: List[Tuple[str, str, str]],
        in_nodes: List[str],
        out_nodes: List[str],
        num_dimensions: int = 2,
        onnx_save_path: Optional[str] = None,
    ):
        super().__init__()
        self.sub_networks = nn.ModuleDict(sub_networks)
        self.node_mapping = node_mapping
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes if isinstance(out_nodes, list) else [out_nodes]
        self.num_dims = num_dimensions
        self.global_net = None

        self._build_global_graph()
        if onnx_save_path:
            self._export_to_onnx(onnx_save_path)

    def _build_global_graph(self):
        global_node_configs = {}
        global_hyperedge_configs = {}
        sub_to_global = {}
        global_to_sub = defaultdict(list)

        for g_node, sub_net, sub_node in self.node_mapping:
            global_to_sub[g_node].append((sub_net, sub_node))
            global_node_configs[g_node] = self.sub_networks[sub_net].node_configs[sub_node]

        for g_node, mappings in global_to_sub.items():
            shapes = [self.sub_networks[sub_net].node_configs[sub_node] for sub_net, sub_node in mappings]
            if len(set(shapes)) > 1:
                raise ValueError(f"全局节点 {g_node} 映射到不同形状: {shapes}")

        edge_id_counter = 0
        for sub_net_name, sub_net in self.sub_networks.items():
            for sub_node in sub_net.node_configs:
                if not any(sub_net_name == sn and sub_node == snode for _, sn, snode in self.node_mapping):
                    global_node = f"{sub_net_name}_{sub_node}"
                    sub_to_global[(sub_net_name, sub_node)] = global_node
                    global_node_configs[global_node] = sub_net.node_configs[sub_node]

            for edge_id, edge_config in sub_net.hyperedge_configs.items():
                global_edge_id = f"e{edge_id_counter}_{sub_net_name}"
                edge_id_counter += 1
                src_nodes = edge_config.get("src_nodes", [])
                dst_nodes = edge_config.get("dst_nodes", [])
                params = edge_config.get("params", {})

                global_src_nodes = []
                for src in src_nodes:
                    found = False
                    for g_node, sn, snode in self.node_mapping:
                        if sn == sub_net_name and snode == src:
                            global_src_nodes.append(g_node)
                            found = True
                            break
                    if not found:
                        global_src_nodes.append(sub_to_global[(sub_net_name, src)])

                global_dst_nodes = []
                for dst in dst_nodes:
                    found = False
                    for g_node, sn, snode in self.node_mapping:
                        if sn == sub_net_name and snode == dst:
                            global_dst_nodes.append(g_node)
                            found = True
                            break
                    if not found:
                        global_dst_nodes.append(sub_to_global[(sub_net_name, dst)])

                # 直接引用子网络的 DNet 实例
                global_hyperedge_configs[global_edge_id] = {
                    "src_nodes": global_src_nodes,
                    "dst_nodes": global_dst_nodes,
                    "params": params,
                    "dnet": sub_net.edges[edge_id],  # 使用子网络的 DNet 实例
                }

        self.global_net = HDNet(
            node_configs=global_node_configs,
            hyperedge_configs=global_hyperedge_configs,
            num_dimensions=self.num_dims,
        )

    def _export_to_onnx(self, onnx_save_path: str):
        self.eval()
        input_shapes = []
        seen_global_nodes = set()
        for global_node in self.in_nodes:
            if global_node in self.global_net.node_configs and global_node not in seen_global_nodes:
                input_shapes.append((1, *self.global_net.node_configs[global_node]))
                seen_global_nodes.add(global_node)

        inputs = [torch.randn(*shape) for shape in input_shapes]
        dynamic_axes = {
            **{f"input_{node}": {0: "batch_size"} for node in self.in_nodes},
            **{f"output_{node}": {0: "batch_size"} for node in self.out_nodes},
        }
        input_names = [f"input_{node}" for node in self.in_nodes]
        output_names = [f"output_{node}" for node in self.out_nodes]
        abs_onnx_path = os.path.abspath(onnx_save_path)
        
        torch.onnx.export(
            self,
            inputs,
            abs_onnx_path,  # 使用绝对路径保存
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            opset_version=17,
        )
        # 修改打印语句，显示完整的绝对路径
        print(f"模型已导出到: {abs_onnx_path}")

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(inputs) != len(self.in_nodes):
            raise ValueError(f"预期输入数量为 {len(self.in_nodes)}，实际得到 {len(inputs)}")

        global_inputs = {node: tensor.to(dtype=torch.float32) for node, tensor in zip(self.in_nodes, inputs)}
        global_features = self.global_net(global_inputs)
        outputs = [global_features[node] for node in self.out_nodes if node in global_features]

        if len(outputs) != len(self.out_nodes):
            raise ValueError(f"某些全局输出节点未生成输出: {self.out_nodes}")
        return outputs

def run_example(
    sub_network_configs: Dict[str, Tuple[Dict[str, Tuple[int, ...]], Dict[str, Dict], str]],  # 增加 agg_mode 参数
    node_mapping: List[Tuple[str, str, str]],
    in_nodes: List[str],
    out_nodes: List[str],
    num_dimensions: int,
    input_shapes: List[Tuple[int, ...]],
    onnx_filename: str,
):
    """
    修改后的 run_example 函数，使用显式命名的子网络配置字典
    
    Args:
        sub_network_configs: 子网络配置字典，键为子网络名称（如 "net1", "unet1"），值为 (node_configs, hyperedge_configs, agg_mode)
        node_mapping: 节点映射列表
        in_nodes: 全局输入节点列表
        out_nodes: 全局输出节点列表
        num_dimensions: 维度（1D/2D/3D）
        input_shapes: 输入张量形状列表
        onnx_filename: ONNX 保存路径
    """
    # 直接遍历字典，使用用户定义的子网络名称
    sub_networks = {}
    for net_name, (node_configs, hyperedge_configs, agg_mode) in sub_network_configs.items():  # 增加 agg_mode 解析
        sub_net = HDNet(
            node_configs=node_configs,
            hyperedge_configs=hyperedge_configs,
            num_dimensions=num_dimensions,
            agg_mode=agg_mode  # 传递聚合模式参数
        )
        sub_networks[net_name] = sub_net

    model = MHDNet(
        sub_networks=sub_networks,
        node_mapping=node_mapping,
        in_nodes=in_nodes,
        out_nodes=out_nodes,
        num_dimensions=num_dimensions,
        onnx_save_path=onnx_filename,
    )
    model.eval()

    inputs = [torch.randn(*shape) for shape in input_shapes]
    outputs = model(inputs)
    for node, out in zip(out_nodes, outputs):
        print(f"全局节点 {node} 的输出: {out.shape}, 数据类型: {out.dtype}")


def example_mhdnet():
    # 子网络 1 配置
    node_configs1 = {
        "A1": (2, 16, 16, 16),  # 映射到全局节点 100
        "B1": (1, 16, 16, 16),  # 映射到全局节点 100
        "C1": (8, 4, 4, 4),     # 映射到全局节点 100
    }
    hyperedge_configs1 = {
        "e1": {
            "src_nodes": ["A1"],
            "dst_nodes": ["C1"],
            "params": {
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "convs": [torch.Size([8, 2, 3, 3, 3])],
                "reqs": [True],
                "norms": ["instance"],
                "acts": ["relu"]
            },
        },
        "e2": {
            "src_nodes": ["B1"],
            "dst_nodes": ["C1"],
            "params": {
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "convs": [torch.Size([8, 1, 3, 3, 3])],  # 修正输入通道数为1（匹配B1的1个通道）
                "reqs": [True],
                "norms": ["instance"],
                "acts": ["relu"]
            },
        },
    }

    node_configs2 = {
        "A2": (2, 16, 16, 16),  # 映射到全局节点 100
        "B2": (1, 16, 16, 16),  # 映射到全局节点 100
        "C2": (8, 4, 4, 4),     # 映射到全局节点 100
    }
    hyperedge_configs2 = {
        "e1": {
            "src_nodes": ["A2"],
            "dst_nodes": ["C2"],
            "params": {
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "convs": [torch.Size([8, 2, 3, 3, 3])],
                "reqs": [True],
                "norms": ["instance"],
                "acts": ["relu"]
            },
        },
        "e2": {
            "src_nodes": ["B2"],
            "dst_nodes": ["C2"],
            "params": {
                "feature_size": (16, 16, 16),
                "intp": "linear",
                "convs": [torch.Size([8, 1, 3, 3, 3])],  # 修正输入通道数为1
                "reqs": [True],
                "norms": ["instance"],
                "acts": ["relu"]
            },
        },
    }

    # 子网络配置字典（增加 agg_mode 参数）
    sub_network_configs = {
        "net1": (node_configs1, hyperedge_configs1, 'add'),
        "net2": (node_configs2, hyperedge_configs2, 'add'),
    }

    # 节点映射
    node_mapping = [
        ("100", "net1", "A1"),
        ("100", "net2", "A2"),
        ("101", "net1", "B1"),
        ("101", "net2", "B2"),
        ("102", "net1", "C1"),
        ("102", "net2", "C2"),
    ]

    # 运行示例
    run_example(
        sub_network_configs=sub_network_configs,
        node_mapping=node_mapping,
        in_nodes=["100", "101"],
        out_nodes=["102"],
        num_dimensions=3,
        input_shapes=[(2, 2, 16, 16, 16), (2, 1, 16, 16, 16)],  # 匹配输入节点的通道数
        onnx_filename="MHDNet_custom_example.onnx",
    )

if __name__ == "__main__":
    example_mhdnet()
