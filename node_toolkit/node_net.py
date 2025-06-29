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

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import warnings

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
    
    Note:
        DNet 仅改变输入张量的通道数，不改变空间尺寸。空间尺寸的调整由 HDNet 的 feature_size 参数控制。
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
        "geld": lambda: nn.GELU(),
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

        # 验证 convs 非空
        if not convs:
            raise ValueError("convs 不能为空，必须提供至少一个卷积配置")

        # 默认值处理
        norms = norms if norms is not None else [None] * len(convs)
        acts = acts if acts is not None else [None] * len(convs)
        reqs = reqs if reqs is not None else [True] * len(convs)

        # 验证配置长度
        if not (len(convs) == len(norms) == len(acts) == len(reqs)):
            raise ValueError(
                f"配置长度不一致：convs={len(convs)}, norms={len(norms)}, acts={len(acts)}, reqs={len(reqs)}"
            )

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

            # 输入通道调整
            if conv_in_channels != current_channels:
                warnings.warn(
                    f"第 {i} 层输入通道数不匹配：预期 {current_channels}，实际 {conv_in_channels}，"
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
                layers.append(self.NORM_TYPES[norm_type](self.num_dimensions, current_channels))

            if act_type:
                act_type = act_type.lower()
                if act_type not in self.ACT_TYPES:
                    raise ValueError(f"第 {i} 层不支持的激活函数: {act_type}")
                layers.append(self.ACT_TYPES[act_type]())

        # 输出通道调整
        if current_channels != out_channels:
            layers.append(
                self.add_channel_adjustment_conv(
                    conv_layer, current_channels, out_channels
                )
            )

        self.filter = nn.Sequential(*layers)

    def add_channel_adjustment_conv(self, conv_layer, in_channels: int, out_channels: int) -> nn.Module:
        """添加通道调整卷积层（1x1 卷积）。

        Args:
            conv_layer: 卷积层类型（Conv1d、Conv2d 或 Conv3d）。
            in_channels: 输入通道数。
            out_channels: 输出通道数。

        Returns:
            1x1 卷积层，带偏置。
        """
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
        node_configs (Dict[str, Tuple[int, ...]]): 节点配置，格式为 {node_id: (channels, size...)}，node_id 为字符串。
        hyperedge_configs (Dict[str, Dict]): 超边配置，包含 src_nodes, dst_nodes 和 params。
        in_nodes (List[str]): 输入节点 ID 列表。
        out_nodes (List[str]): 输出节点 ID 列表。
        num_dimensions (int): 维度（1D、2D 或 3D）。
    
    Note:
        - 所有空间尺寸的改变通过 hyperedge_configs 中的 feature_size 参数调整，DNet 本身不改变空间尺寸。
        - 通道拼接/分离通过多个节点与单条超边实现，即超边的 src_nodes 或 dst_nodes 包含多个节点，特征在通道维度上拼接或分离。
        - 特征图相加/共用通过单个节点与多条超边实现，即一个节点作为多条超边的 src_nodes 或 dst_nodes，特征图在通道维度上相加或共享。
        - 每条超边支持 dropout、reshape 和 permute，分别在 DNet 处理后应用。
    """
    def __init__(
        self,
        node_configs: Dict[str, Tuple[int, ...]],
        hyperedge_configs: Dict[str, Dict],
        in_nodes: List[str],
        out_nodes: List[str],
        num_dimensions: int = 2,
    ):
        super().__init__()
        self.node_configs = node_configs
        self.hyperedge_configs = hyperedge_configs
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.num_dims = num_dimensions
        self.edges = nn.ModuleDict()
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)
        self.dropouts = nn.ModuleDict()

        self._validate_nodes()
        self._build_hyperedges()

    def _validate_nodes(self):
        """验证节点配置并移除未使用的节点"""
        all_nodes = set(self.node_configs.keys())
        used_nodes = set(self.in_nodes + self.out_nodes)
        for edge_config in self.hyperedge_configs.values():
            used_nodes.update(edge_config.get("src_nodes", []))
            used_nodes.update(edge_config.get("dst_nodes", []))
        unused_nodes = all_nodes - used_nodes
        if unused_nodes:
            warnings.warn(f"未使用的节点将被忽略: {unused_nodes}")
            self.node_configs = {k: v for k, v in self.node_configs.items() if k in used_nodes}

    def _compute_edge_channels(self, src_nodes: List[str], dst_nodes: List[str]) -> Tuple[int, int]:
        """计算超边的输入和输出通道数"""
        in_channels = sum(self.node_configs[src][0] for src in src_nodes)
        out_channels = sum(self.node_configs[dst][0] for dst in dst_nodes)
        return in_channels, out_channels

    def _build_hyperedges(self):
        """构建超边并初始化 DNet 和 Dropout"""
        dropout_layer = getattr(nn, f"Dropout{self.num_dims}d")
        for edge_id, edge_config in self.hyperedge_configs.items():
            src_nodes = edge_config.get("src_nodes", [])
            dst_nodes = edge_config.get("dst_nodes", [])
            params = edge_config.get("params", {})
            in_channels, out_channels = self._compute_edge_channels(src_nodes, dst_nodes)
            convs = params.get("convs")
            if not convs:
                raise ValueError(f"超边 {edge_id} 的 convs 不能为空，必须提供卷积配置")
            reqs = params.get("reqs", [True] * len(convs))
            norms = params.get("norms")
            acts = params.get("acts")
            dropout_rate = params.get("dropout", 0.0)

            self.edges[edge_id] = DNet(
                in_channels, out_channels, self.num_dims, convs, reqs, norms, acts
            )
            if dropout_rate > 0:
                self.dropouts[edge_id] = dropout_layer(dropout_rate)
            for src in src_nodes:
                self.out_edges[src].append(edge_id)
            for dst in dst_nodes:
                self.in_edges[dst].append(edge_id)

    def _get_interpolate_mode(self, p: str) -> str:
        """根据维度和插值类型返回 F.interpolate 的 mode 参数"""
        mode_map = {
            1: {"linear": "linear", "nearest": "nearest"},
            2: {"linear": "bilinear", "nearest": "nearest"},
            3: {"linear": "trilinear", "nearest": "nearest"},
        }
        return mode_map[self.num_dims][p]

    def _power_interpolate(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, ...]],
        p: Union[float, str],
    ) -> torch.Tensor:
        """使用 p 次幂插值调整张量大小，支持线性插值、最近邻、最大/平均池化。

        Args:
            x: 输入张量，形状为 (batch, channels, *spatial_dims)。
            target_size: 目标空间大小。
            p: 幂次参数，可以是浮点数（p 次幂插值）、'max'（最大池化）、'avg'（平均池化）、'nearest'（最近邻）、'linear'(线性)。

        Returns:
            调整大小后的张量，始终为 float32。
        """
        if not target_size or x.shape[2:] == tuple(target_size):
            return x.to(dtype=torch.float32)

        x = x.to(dtype=torch.float32)
        p = p.lower() if isinstance(p, str) else p
        if p in ("max", "avg"):
            pool_layer = getattr(nn, f"Adaptive{p.title()}Pool{self.num_dims}d")
            return pool_layer(target_size)(x)
        if p in ("nearest", "linear"):
            return F.interpolate(
                x,
                size=target_size,
                mode=self._get_interpolate_mode(p),
                align_corners=p == "linear",
            )

        spatial_dims = tuple(range(2, 2 + self.num_dims))
        min_vals = torch.amin(x, dim=spatial_dims, keepdim=True)
        x_norm = torch.clamp((x - min_vals), min=0.0)
        x_pow = torch.pow(x_norm + 1e-8, p)
        x_resized = F.interpolate(
            x_pow,
            size=target_size,
            mode=self._get_interpolate_mode("linear"),
            align_corners=False,
        )
        x_root = torch.pow(x_resized, 1.0 / p)
        return x_root + min_vals

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """前向传播。

        Args:
            inputs: 输入张量列表，对应 in_nodes。

        Returns:
            输出张量列表，对应 out_nodes。
        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("输入必须是张量列表或元组")
        if len(inputs) != len(self.in_nodes):
            raise ValueError(f"预期输入数量为 {len(self.in_nodes)}，实际得到 {len(inputs)}")

        features = {}
        for node, input_tensor in zip(self.in_nodes, inputs):
            features[node] = input_tensor.to(dtype=torch.float32)

        processed_nodes = set(self.in_nodes)
        all_nodes = set(self.node_configs.keys())

        while processed_nodes != all_nodes:
            progress = False
            for node in all_nodes - processed_nodes:
                in_edge_ids = self.in_edges[node]
                if not in_edge_ids:
                    continue

                if not all(
                    all(
                        src in processed_nodes
                        for src in self.hyperedge_configs[edge_id].get("src_nodes", [])
                    )
                    for edge_id in in_edge_ids
                ):
                    continue

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

                    src_features = [
                        self._power_interpolate(
                            features[src],
                            feature_size,
                            intp
                        )
                        for src in src_nodes
                    ]
                    input_feat = torch.cat(src_features, dim=1)
                    output = self.edges[edge_id](input_feat)

                    # 应用 Dropout
                    if edge_id in self.dropouts:
                        output = self.dropouts[edge_id](output)

                    # 应用 Reshape
                    if reshape_size and output.shape[2:] != tuple(reshape_size):
                        output = output.reshape(-1, output.shape[1], *reshape_size)

                    # 应用 Permute
                    if permute_order != tuple(range(self.num_dims + 2)):
                        output = output.permute(*permute_order)

                    channel_sizes = [self.node_configs[dst][0] for dst in dst_nodes]
                    split_outputs = torch.split(output, channel_sizes, dim=1)
                    dst_features = {
                        dst: self._power_interpolate(
                            feat,
                            self.node_configs[dst][1:],
                            intp
                        )
                        for dst, feat in zip(dst_nodes, split_outputs)
                    }

                    if node in dst_features:
                        node_inputs.append(dst_features[node])

                if node_inputs:
                    features[node] = sum(node_inputs).to(dtype=torch.float32)
                    processed_nodes.add(node)
                    progress = True
                else:
                    raise ValueError(f"节点 {node} 没有有效输入")

            if not progress:
                raise RuntimeError("图中存在无法解析的依赖，可能包含环")

        outputs = [features[node] for node in self.out_nodes]
        if len(outputs) != len(self.out_nodes):
            missing = [node for node in self.out_nodes if node not in features]
            raise ValueError(f"输出节点 {missing} 未被计算")
        return outputs


class MHDNet(nn.Module):
    """多子超图网络（MHDNet），将子 HDNet 作为模块处理全局节点输入和输出。

    Args:
        sub_networks: 子 HDNet 网络，键为网络名称，值为 HDNet 实例。
        node_mapping: 节点映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]，global_node_id 和 sub_node_id 为字符串。
        in_nodes: 全局输入节点 ID 列表。
        out_nodes: 全局输出节点 ID 列表。
        num_dimensions: 维度（1D、2D 或 3D）。
        onnx_save_path: 可选的 ONNX 模型保存路径，若提供则在初始化时保存模型。
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

        self._validate_mapping()
        self._check_node_shapes()

        if onnx_save_path:
            self._export_to_onnx(onnx_save_path)

    def _validate_mapping(self):
        """验证节点映射的有效性"""
        for mapping in self.node_mapping:
            if not isinstance(mapping, tuple) or len(mapping) != 3:
                raise ValueError(f"node_mapping 条目格式错误: {mapping}")
            global_node, sub_net_name, sub_node_id = mapping
            if sub_net_name not in self.sub_networks:
                raise ValueError(f"子网络 {sub_net_name} 未在 sub_networks 中定义")
            sub_net = self.sub_networks[sub_net_name]
            if sub_node_id not in sub_net.node_configs:
                raise ValueError(f"子网络 {sub_net_name} 中不存在节点 {sub_node_id}")

        mapped_global_nodes = {mapping[0] for mapping in self.node_mapping}
        for node in self.in_nodes:
            if node not in mapped_global_nodes:
                raise ValueError(f"全局输入节点 {node} 未在 node_mapping 中定义")
            found = any(
                g_node == node and sub_node_id in self.sub_networks[s_name].in_nodes
                for g_node, s_name, sub_node_id in self.node_mapping
            )
            if not found:
                raise ValueError(f"全局输入节点 {node} 未映射到任何子网络的输入节点")

        for node in self.out_nodes:
            if node not in mapped_global_nodes:
                raise ValueError(f"全局输出节点 {node} 未在 node_mapping 中定义")
            found = any(
                g_node == node and sub_node_id in self.sub_networks[s_name].out_nodes
                for g_node, s_name, sub_node_id in self.node_mapping
            )
            if not found:
                raise ValueError(f"全局输出节点 {node} 未映射到任何子网络的输出节点")

    def _check_node_shapes(self):
        """检查全局节点与子网络节点的形状有效性"""
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            sub_net = self.sub_networks[sub_net_name]
            shape = sub_net.node_configs[sub_node_id]
            if not shape or not all(isinstance(s, int) and s > 0 for s in shape):
                raise ValueError(f"节点 {global_node} 的映射节点 {sub_node_id} 形状无效: {shape}")

        shape_map = {}
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            sub_net = self.sub_networks[sub_net_name]
            shape = sub_net.node_configs[sub_node_id]
            if global_node in shape_map:
                if shape_map[global_node] != shape:
                    raise ValueError(
                        f"全局节点 {global_node} 映射到不同形状: "
                        f"{shape_map[global_node]} vs {shape}"
                    )
            else:
                shape_map[global_node] = shape

    def _export_to_onnx(self, onnx_save_path: str):
        """将模型导出为 ONNX 格式。

        Args:
            onnx_save_path: ONNX 模型保存路径。
        """
        self.eval()
        input_shapes = [
            (1, *self.sub_networks[sub_net_name].node_configs[sub_node_id])
            for global_node in self.in_nodes
            for g_node, sub_net_name, sub_node_id in self.node_mapping
            if g_node == global_node
        ]
        inputs = [torch.randn(*shape) for shape in input_shapes]
        dynamic_axes = {
            **{f"input_{node}": {0: "batch_size"} for node in self.in_nodes},
            **{f"output_{node}": {0: "batch_size"} for node in self.out_nodes},
        }
        torch.onnx.export(
            self,
            inputs,
            onnx_save_path,
            input_names=[f"input_{node}" for node in self.in_nodes],
            output_names=[f"output_{node}" for node in self.out_nodes],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
        print(f"模型已导出为 {onnx_save_path}")

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播，通过子网络处理全局节点输入并收集输出。

        Args:
            inputs: 输入张量列表，对应 in_nodes。

        Returns:
            输出张量列表，对应 out_nodes。
        """
        if len(inputs) != len(self.in_nodes):
            raise ValueError(
                f"预期输入数量为 {len(self.in_nodes)}，实际得到 {len(inputs)}"
            )

        global_features = {}
        for global_node, input_tensor in zip(self.in_nodes, inputs):
            global_features[global_node] = input_tensor.to(dtype=torch.float32)

        sub_net_inputs = {
            name: [None] * len(net.in_nodes)
            for name, net in self.sub_networks.items()
        }

        for global_node in self.in_nodes:
            for g_node, sub_net_name, sub_node_id in self.node_mapping:
                if g_node == global_node:
                    sub_net = self.sub_networks[sub_net_name]
                    idx = sub_net.in_nodes.index(sub_node_id)
                    sub_net_inputs[sub_net_name][idx] = global_features[global_node]

        processed_nets = set()
        sub_net_outputs = {}
        while len(processed_nets) < len(self.sub_networks):
            progress = False
            for name, net in self.sub_networks.items():
                if name in processed_nets:
                    continue

                inputs_ready = True
                for sub_node_id in net.in_nodes:
                    global_node = None
                    for g_node, s_name, s_node_id in self.node_mapping:
                        if s_name == name and s_node_id == sub_node_id:
                            global_node = g_node
                            break
                    if global_node is None:
                        raise ValueError(f"子网络 {name} 的输入节点 {sub_node_id} 未在 node_mapping 中定义")

                    if global_node not in global_features:
                        src_net_name, src_node_id = None, None
                        for g_node, s_name, s_node_id in self.node_mapping:
                            if g_node == global_node and s_name != name:
                                src_net_name, src_node_id = s_name, s_node_id
                                break
                        if src_net_name is None or src_node_id is None:
                            raise ValueError(f"全局节点 {global_node} 未找到输出来源")
                        if src_net_name not in sub_net_outputs or src_node_id not in sub_net_outputs[src_net_name]:
                            inputs_ready = False
                            break

                if inputs_ready:
                    for sub_node_id in net.in_nodes:
                        global_node = None
                        for g_node, s_name, s_node_id in self.node_mapping:
                            if s_name == name and s_node_id == sub_node_id:
                                global_node = g_node
                                break
                        idx = net.in_nodes.index(sub_node_id)
                        if global_node in global_features:
                            if sub_net_inputs[name][idx] is None:
                                sub_net_inputs[name][idx] = global_features[global_node]
                        else:
                            src_net_name, src_node_id = None, None
                            for g_node, s_name, s_node_id in self.node_mapping:
                                if g_node == global_node and s_name != name:
                                    src_net_name, src_node_id = s_name, s_node_id
                                    break
                            sub_net_inputs[name][idx] = sub_net_outputs[src_net_name][src_node_id]

                    outputs = net(sub_net_inputs[name])
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    sub_net_outputs[name] = dict(zip(net.out_nodes, outputs))

                    for sub_node_id, output_tensor in zip(net.out_nodes, outputs):
                        global_node = None
                        for g_node, s_name, s_node_id in self.node_mapping:
                            if s_name == name and s_node_id == sub_node_id:
                                global_node = g_node
                                break
                        if global_node is not None:
                            global_features[global_node] = output_tensor

                    processed_nets.add(name)
                    progress = True

            if not progress:
                raise ValueError("检测到循环依赖或无法满足的子网络输入")

        outputs = []
        for global_node in self.out_nodes:
            if global_node not in global_features:
                raise ValueError(f"全局输出节点 {global_node} 无输出")
            outputs.append(global_features[global_node])

        return outputs


def run_example(
    sub_network_configs: List[Tuple[Dict[str, Tuple[int, ...]], Dict[str, Dict], List[str], List[str]]],
    node_mapping: List[Tuple[str, str, str]],
    in_nodes: List[str],
    out_nodes: List[str],
    num_dimensions: int,
    input_shapes: List[Tuple[int, ...]],
    onnx_filename: str,
):
    """运行并导出 MHDNet 模型为 ONNX。

    Args:
        sub_network_configs: 子网络配置列表，每个元素为 (node_configs, hyperedge_configs, in_nodes, out_nodes)。
        node_mapping: 全局节点到子网络节点的映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        in_nodes: 全局输入节点 ID。
        out_nodes: 全局输出节点 ID。
        num_dimensions: 维度。
        input_shapes: 输入张量形状。
        onnx_filename: 导出的 ONNX 文件名。
    """
    sub_networks = {}
    for i, (node_configs, hyperedge_configs, sub_in_nodes, sub_out_nodes) in enumerate(sub_network_configs):
        sub_net = HDNet(
            node_configs,
            hyperedge_configs,
            sub_in_nodes,
            sub_out_nodes,
            num_dimensions,
        )
        sub_networks[f"net{i+1}"] = sub_net

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
        "n0": (4, 64, 64, 64),
        "n1": (1, 64, 64, 64),
        "n2": (32, 8, 8, 8),
        "n3": (32, 8, 8, 8),
        "n4": (32, 8, 8, 8),
        "n5": (64, 1, 1, 1),
    }
    hyperedge_configs1 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n2"],
            "params": {
                "feature_size": (64, 64, 64),
                "intp": 2,
                "convs": [torch.Size([32, 4, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
                "dropout": 0.1,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            },
        },
        "e2": {
            "src_nodes": ["n1"],
            "dst_nodes": ["n3"],
            "params": {
                "feature_size": (64, 64, 64),
                "intp": 2,
                "convs": [torch.Size([32, 1, 3, 3, 3]), torch.Size([32, 32, 3, 3, 3])],
                "reqs": [True, True],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
                "dropout": 0.2,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            },
        },
        "e3": {
            "src_nodes": ["n0", "n1"],
            "dst_nodes": ["n4"],
            "params": {
                "feature_size": (64, 64, 64),
                "intp": 2,
                "convs": [torch.Size([32, 5, 3, 3, 3]), torch.eye(32).reshape(32, 32, 1, 1, 1)],
                "reqs": [True, False],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
                "dropout": 0.0,
                "reshape": (64, 64, 64),
                "permute": (0, 1, 2, 3, 4),
            },
        },
        "e4": {
            "src_nodes": ["n4"],
            "dst_nodes": ["n5"],
            "params": {
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "convs": [torch.Size([64, 32, 3, 3, 3]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "dropout": 0.1,
                "reshape": (8, 8, 8),
                "permute": (0, 1, 2, 3, 4),
            },
        },
        "e5": {
            "src_nodes": ["n2", "n3"],
            "dst_nodes": ["n5"],
            "params": {
                "feature_size": (8, 8, 8),
                "intp": "linear",
                "convs": [torch.Size([64, 64, 3, 3, 3]), torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True, True],
                "dropout": 0.1,
                "reshape": (8, 8, 8),
                "permute": (0, 1, 2, 3, 4),
            },
        },
    }

    # 子网络 2 配置
    node_configs2 = {
        "n0": (64, 1, 1, 1),
        "n1": (128, 1, 1, 1),
        "n2": (128, 1, 1, 1),
    }
    hyperedge_configs2 = {
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "feature_size": (1, 1, 1),
                "intp": "linear",
                "convs": [torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True],
                "dropout": 0.0,
                "reshape": (1, 1, 1),
                "permute": (0, 1, 2, 3, 4),
            },
        },
        "e2": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n2"],
            "params": {
                "feature_size": (1, 1, 1),
                "intp": "linear",
                "convs": [torch.Size([64, 64, 3, 3, 3])],
                "reqs": [True],
                "dropout": 0.0,
                "reshape": (1, 1, 1),
                "permute": (0, 1, 2, 3, 4),
            },
        },
    }

    # 子网络 3 配置
    node_configs3 = node_configs2
    hyperedge_configs3 = hyperedge_configs2.copy()
    hyperedge_configs3.update({
        "e1": {
            "src_nodes": ["n0"],
            "dst_nodes": ["n1"],
            "params": {
                "feature_size": (1, 1, 1),
                "intp": "linear",
                "convs": [torch.Size([128, 64, 1, 1, 1])],
                "reqs": [True],
                "acts": ["sigmoid"],
                "dropout": 0.1,
                "reshape": (1, 1, 1),
                "permute": (0, 1, 2, 3, 4),
            },
        },
    })

    # 节点映射
    node_mapping = [
        ("n100", "net1", "n0"),
        ("n101", "net1", "n1"),
        ("n102", "net1", "n5"),
        ("n102", "net2", "n0"),
        ("n102", "net3", "n0"),
        ("n103", "net2", "n1"),
        ("n104", "net2", "n2"),
        ("n105", "net3", "n1"),
        ("n106", "net3", "n2"),
    ]

    # 运行示例
    run_example(
        sub_network_configs=[
            (node_configs1, hyperedge_configs1, ["n0", "n1"], ["n5"]),
            (node_configs2, hyperedge_configs2, ["n0"], ["n1", "n2"]),
            (node_configs3, hyperedge_configs3, ["n0"], ["n1", "n2"]),
        ],
        node_mapping=node_mapping,
        in_nodes=["n100", "n101"],
        out_nodes=["n103", "n104", "n105", "n106"],
        num_dimensions=3,
        input_shapes=[(2, 4, 64, 64, 64), (2, 1, 64, 64, 64)],
        onnx_filename="MHDNet_example.onnx",
    )


if __name__ == "__main__":
    example_mhdnet()
