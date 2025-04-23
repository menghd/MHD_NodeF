"""
MHD_Nodet Project - Multi Hypergraph Dynamic Node Network
========================================================
This module defines the neural network architecture for the MHD_Nodet project,
including DNet, HDNet, and MHDNet classes for dynamic hypergraph-based processing.

项目：MHD_Nodet - 多超图动态节点网络
本模块定义了 MHD_Nodet 项目的神经网络架构，包括 DNet、HDNet 和 MHDNet 类，
用于基于动态超图的处理。

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
        convs (List[Optional[Tuple[int, ...]]]): 每层卷积配置，格式为 (out_channels, kernel_size...)，None 表示无卷积。
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
        "swish": lambda: nn.SiLU(),  # Swish 等价于 SiLU
        "tanh": lambda: nn.Tanh(),
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_dimensions: int,
        convs: List[Optional[Tuple[int, ...]]],
        norms: Optional[List[Optional[str]]] = None,
        acts: Optional[List[Optional[str]]] = None,
    ):
        super().__init__()
        self.num_dimensions = num_dimensions
        conv_layer = getattr(nn, f"Conv{num_dimensions}d")
        layers = []

        # 默认值处理
        norms = norms if norms is not None else [None] * len(convs)
        acts = acts if acts is not None else [None] * len(convs)

        # 验证配置长度
        if not (len(convs) == len(norms) == len(acts)):
            raise ValueError(
                f"配置长度不一致：convs={len(convs)}, norms={len(norms)}, acts={len(acts)}"
            )

        current_channels = in_channels
        for i, (conv_config, norm_type, act_type) in enumerate(zip(convs, norms, acts)):
            if conv_config is not None:
                conv_out_channels, *kernel_size = conv_config
                padding = tuple(k // 2 for k in kernel_size)
                layers.append(
                    conv_layer(
                        current_channels,
                        conv_out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    )
                )
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

        # 确保输出通道匹配
        if current_channels != out_channels:
            layers.append(
                conv_layer(current_channels, out_channels, kernel_size=1, bias=False)
            )

        self.filter = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)


class HDNet(nn.Module):
    """基于超边的网络，支持动态维度和灵活的节点连接。

    Args:
        node_configs (Dict[int, Tuple[int, ...]]): 节点配置，格式为 {node_id: (channels, size...)}。
        hyperedge_configs (Dict[str, Dict]): 超边配置，包含 src_nodes, dst_nodes 和 params。
        in_nodes (List[int]): 输入节点 ID 列表。
        out_nodes (List[int]): 输出节点 ID 列表。
        num_dimensions (int): 维度（1D、2D 或 3D）。
        node_dtype (Dict[int, str]): 节点数据类型，格式为 {node_id: dtype}，dtype 为 "float" 或 "long"，默认 "float"。
    """
    DTYPE_MAP = {
        "float": torch.float32,
        "long": torch.int64
    }

    def __init__(
        self,
        node_configs: Dict[int, Tuple[int, ...]],
        hyperedge_configs: Dict[str, Dict],
        in_nodes: List[int],
        out_nodes: List[int],
        num_dimensions: int = 2,
        node_dtype: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.node_configs = node_configs
        self.hyperedge_configs = hyperedge_configs
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.num_dimensions = num_dimensions
        self.node_dtype = node_dtype if node_dtype is not None else {k: "float" for k in node_configs}
        self.edges = nn.ModuleDict()
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)

        # 验证数据类型
        for node_id, dtype in self.node_dtype.items():
            if dtype not in self.DTYPE_MAP:
                raise ValueError(f"不支持的数据类型 {dtype} 在节点 {node_id}")
            if node_id not in self.node_configs:
                raise ValueError(f"节点 {node_id} 在 node_configs 中未定义")

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
            self.node_dtype = {k: v for k, v in self.node_dtype.items() if k in used_nodes}

    def _compute_edge_channels(self, src_nodes: List[int], dst_nodes: List[int]) -> Tuple[int, int]:
        """计算超边的输入和输出通道数"""
        in_channels = sum(self.node_configs[src][0] for src in src_nodes)
        out_channels = sum(self.node_configs[dst][0] for dst in dst_nodes)
        return in_channels, out_channels

    def _build_hyperedges(self):
        """构建超边并初始化 DNet"""
        for edge_id, edge_config in self.hyperedge_configs.items():
            src_nodes = edge_config.get("src_nodes", [])
            dst_nodes = edge_config.get("dst_nodes", [])
            params = edge_config.get("params", {})
            in_channels, out_channels = self._compute_edge_channels(src_nodes, dst_nodes)
            convs = params.get("convs", [(in_channels, 3, 3)])
            norms = params.get("norms")
            acts = params.get("acts")

            self.edges[edge_id] = DNet(
                in_channels, out_channels, self.num_dimensions, convs, norms, acts
            )
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
        return mode_map[self.num_dimensions][p]

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

        x = x.to(dtype=torch.float32)  # 确保输入为 float32
        p = p.lower() if isinstance(p, str) else p
        if p in ("max", "avg"):
            pool_layer = getattr(nn, f"Adaptive{p.title()}Pool{self.num_dimensions}d")
            return pool_layer(target_size)(x)
        if p in ("nearest", "linear"):
            return F.interpolate(
                x,
                size=target_size,
                mode=self._get_interpolate_mode(p),
                align_corners=p == "linear",
            )

        # p 次幂插值
        spatial_dims = tuple(range(2, 2 + self.num_dimensions))
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
        return x_root+min_vals

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

        # 初始化特征字典
        features = {}
        for node, input_tensor in zip(self.in_nodes, inputs):
            features[node] = input_tensor.to(dtype=torch.float32)  # 统一转换为 float32

        processed_nodes = set(self.in_nodes)
        all_nodes = set(self.node_configs.keys())

        # 逐步处理节点
        while processed_nodes != all_nodes:
            progress = False
            for node in all_nodes - processed_nodes:
                in_edge_ids = self.in_edges[node]
                if not in_edge_ids:
                    continue

                # 检查所有输入边是否就绪
                if not all(
                    all(
                        src in processed_nodes
                        for src in self.hyperedge_configs[edge_id].get("src_nodes", [])
                    )
                    for edge_id in in_edge_ids
                ):
                    continue

                # 收集节点输入
                node_inputs = []
                for edge_id in in_edge_ids:
                    edge_config = self.hyperedge_configs[edge_id]
                    src_nodes = edge_config.get("src_nodes", [])
                    dst_nodes = edge_config.get("dst_nodes", [])
                    params = edge_config.get("params", {})
                    feature_size = params.get("feature_size")
                    in_p = params.get("in_p", "linear")
                    out_p = params.get("out_p", "linear")

                    # 调整输入特征大小并拼接，输入统一为 float32
                    src_features = [
                        self._power_interpolate(
                            features[src],  # 已为 float32
                            feature_size,
                            in_p
                        )
                        for src in src_nodes
                    ]
                    input_feat = torch.cat(src_features, dim=1)

                    # 执行 DNet 操作，输出为 float32
                    output = self.edges[edge_id](input_feat)

                    # 分割输出并调整到目标节点形状
                    channel_sizes = [self.node_configs[dst][0] for dst in dst_nodes]
                    split_outputs = torch.split(output, channel_sizes, dim=1)
                    dst_features = {
                        dst: self._power_interpolate(
                            feat,
                            self.node_configs[dst][1:],
                            out_p
                        )  # 输出为 float32
                        for dst, feat in zip(dst_nodes, split_outputs)
                    }

                    if node in dst_features:
                        node_inputs.append(dst_features[node])  # 保持 float32

                if node_inputs:
                    # 确保输出数据类型，仅在最终输出时转换
                    dtype = self.DTYPE_MAP[self.node_dtype.get(node, "float")]
                    features[node] = sum(node_inputs).to(dtype=dtype)  # 仅在此处转换为目标类型
                    processed_nodes.add(node)
                    progress = True
                else:
                    raise ValueError(f"节点 {node} 没有有效输入")

            if not progress:
                raise RuntimeError("图中存在无法解析的依赖，可能包含环或孤立节点")

        # 收集输出
        outputs = [features[node] for node in self.out_nodes]
        if len(outputs) != len(self.out_nodes):
            missing = [node for node in self.out_nodes if node not in features]
            raise ValueError(f"输出节点 {missing} 未被计算")
        return outputs


class MHDNet(nn.Module):
    """多子超图网络（MHDNet），将子 HDNet 作为模块处理全局节点输入和输出。

    Args:
        sub_networks: 子 HDNet 网络，键为网络名称，值为 HDNet 实例。
        node_mapping: 节点映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        in_nodes: 全局输入节点 ID 列表。
        out_nodes: 全局输出节点 ID 列表。
        num_dimensions: 维度（1D、2D 或 3D）。
    """
    def __init__(
        self,
        sub_networks: Dict[str, nn.Module],
        node_mapping: List[Tuple[int, str, int]],
        in_nodes: List[int],
        out_nodes: List[int],
        num_dimensions: int = 2,
    ):
        super().__init__()
        self.sub_networks = nn.ModuleDict(sub_networks)
        self.node_mapping = node_mapping
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes if isinstance(out_nodes, list) else [out_nodes]
        self.num_dimensions = num_dimensions

        self._validate_mapping()
        self._check_node_shapes()

    def _validate_mapping(self):
        """验证节点映射的有效性并检查数据类型一致性"""
        # 检查 node_mapping 格式
        for mapping in self.node_mapping:
            if not isinstance(mapping, tuple) or len(mapping) != 3:
                raise ValueError(f"node_mapping 条目格式错误: {mapping}")
            global_node, sub_net_name, sub_node_id = mapping
            if sub_net_name not in self.sub_networks:
                raise ValueError(f"子网络 {sub_net_name} 未在 sub_networks 中定义")
            sub_net = self.sub_networks[sub_net_name]
            if sub_node_id not in sub_net.node_configs:
                raise ValueError(f"子网络 {sub_net_name} 中不存在节点 {sub_node_id}")

        # 检查全局输入和输出节点的映射
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

        # 检查全局节点数据类型一致性
        dtype_conflicts = defaultdict(list)
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            sub_net = self.sub_networks[sub_net_name]
            dtype = sub_net.node_dtype.get(sub_node_id, "float")
            dtype_conflicts[global_node].append((sub_net_name, sub_node_id, dtype))

        for global_node, mappings in dtype_conflicts.items():
            dtypes = {dtype for _, _, dtype in mappings}
            if len(dtypes) > 1:
                conflict_info = "\n".join(
                    f"  - 子网络 {net_name} 节点 {node_id}: 数据类型 {dtype}"
                    for net_name, node_id, dtype in mappings
                )
                raise ValueError(
                    f"全局节点 {global_node} 映射到不一致的数据类型:\n"
                    f"{conflict_info}\n"
                    f"请确保同一全局节点的所有映射节点具有相同的数据类型（'float' 或 'long'）。"
                )

    def _check_node_shapes(self):
        """检查全局节点与子网络节点的形状有效性"""
        for global_node, sub_net_name, sub_node_id in self.node_mapping:
            sub_net = self.sub_networks[sub_net_name]
            shape = sub_net.node_configs[sub_node_id]
            if not shape or not all(isinstance(s, int) and s > 0 for s in shape):
                raise ValueError(f"节点 {global_node} 的映射节点 {sub_node_id} 形状无效: {shape}")

        # 检查同一全局节点映射的子节点形状一致
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

        # 初始化全局节点特征
        global_features = {}
        for global_node, input_tensor in zip(self.in_nodes, inputs):
            global_features[global_node] = input_tensor

        # 初始化子网络输入
        sub_net_inputs = {
            name: [None] * len(net.in_nodes)
            for name, net in self.sub_networks.items()
        }

        # 填充全局输入到子网络
        for global_node in self.in_nodes:
            for g_node, sub_net_name, sub_node_id in self.node_mapping:
                if g_node == global_node:
                    sub_net = self.sub_networks[sub_net_name]
                    idx = sub_net.in_nodes.index(sub_node_id)
                    sub_net_inputs[sub_net_name][idx] = global_features[global_node]

        # 动态处理子网络依赖
        processed_nets = set()
        sub_net_outputs = {}
        while len(processed_nets) < len(self.sub_networks):
            progress = False
            for name, net in self.sub_networks.items():
                if name in processed_nets:
                    continue

                # 检查该子网络的所有输入是否都已准备好
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
                    # 填充依赖的输入
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

                    # 运行子网络
                    outputs = net(sub_net_inputs[name])
                    if not isinstance(outputs, (list, tuple)):
                        outputs = [outputs]
                    sub_net_outputs[name] = dict(zip(net.out_nodes, outputs))

                    # 更新全局节点特征
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

        # 收集全局输出
        outputs = []
        for global_node in self.out_nodes:
            if global_node not in global_features:
                raise ValueError(f"全局输出节点 {global_node} 无输出")
            outputs.append(global_features[global_node])

        return outputs


def run_example(
    sub_network_configs: List[Tuple[Dict[int, Tuple[int, ...]], Dict[str, Dict], List[int], List[int], Dict[int, str]]],
    node_mapping: List[Tuple[int, str, int]],
    in_nodes: List[int],
    out_nodes: List[int],
    num_dimensions: int,
    input_shapes: List[Tuple[int, ...]],
    onnx_filename: str,
):
    """运行并导出 MHDNet 模型为 ONNX。

    Args:
        sub_network_configs: 子网络配置列表，每个元素为 (node_configs, hyperedge_configs, in_nodes, out_nodes, node_dtype)。
        node_mapping: 全局节点到子网络节点的映射，格式为 [(global_node_id, sub_net_name, sub_node_id), ...]。
        in_nodes: 全局输入节点 ID。
        out_nodes: 全局输出节点 ID。
        num_dimensions: 维度。
        input_shapes: 输入张量形状。
        onnx_filename: 导出的 ONNX 文件名。
    """
    sub_networks = {}
    for i, (node_configs, hyperedge_configs, sub_in_nodes, sub_out_nodes, node_dtype) in enumerate(sub_network_configs):
        sub_net = HDNet(
            node_configs,
            hyperedge_configs,
            sub_in_nodes,
            sub_out_nodes,
            num_dimensions,
            node_dtype=node_dtype
        )
        sub_networks[f"net{i+1}"] = sub_net

    model = MHDNet(
        sub_networks=sub_networks,
        node_mapping=node_mapping,
        in_nodes=in_nodes,
        out_nodes=out_nodes,
        num_dimensions=num_dimensions,
    )
    model.eval()

    inputs = [torch.randn(*shape) for shape in input_shapes]
    outputs = model(inputs)
    for node, out in zip(out_nodes, outputs):
        print(f"全局节点 {node} 的输出: {out.shape}, 数据类型: {out.dtype}")

    dynamic_axes = {
        **{f"input_{node}": {0: "batch_size"} for node in in_nodes},
        **{f"output_{node}": {0: "batch_size"} for node in out_nodes},
    }
    torch.onnx.export(
        model,
        inputs,
        onnx_filename,
        input_names=[f"input_{node}" for node in in_nodes],
        output_names=[f"output_{node}" for node in out_nodes],
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
    print(f"模型已导出为 {onnx_filename}")


def example_mhdnet():
    # 子网络 1 配置
    node_configs1 = {
        0: (4, 64, 64, 64),
        1: (1, 64, 64, 64),
        2: (32, 8, 8, 8),
        3: (32, 8, 8, 8),
        4: (32, 8, 8, 8),
        5: (64, 1, 1, 1),
    }
    hyperedge_configs1 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "feature_size": (64, 64, 64),
                "out_p": 2,
                "convs": [(32, 3, 3, 3), (32, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
            },
        },
        "e2": {
            "src_nodes": [1],
            "dst_nodes": [3],
            "params": {
                "feature_size": (64, 64, 64),
                "out_p": 2,
                "convs": [(32, 3, 3, 3), (32, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
            },
        },
        "e3": {
            "src_nodes": [0, 1],
            "dst_nodes": [4],
            "params": {
                "feature_size": (64, 64, 64),
                "out_p": 2,
                "convs": [(32, 3, 3, 3), (32, 3, 3, 3)],
                "norms": ["instance", "instance"],
                "acts": ["leakyrelu", "leakyrelu"],
            },
        },
        "e4": {
            "src_nodes": [4],
            "dst_nodes": [5],
            "params": {
                "feature_size": (8, 8, 8),
                "in_p": "linear",
                "out_p": "linear",
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
            },
        },
        "e5": {
            "src_nodes": [2, 3],
            "dst_nodes": [5],
            "params": {
                "feature_size": (8, 8, 8),
                "in_p": "linear",
                "out_p": "linear",
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
            },
        },
    }
    # 子网络 1 的数据类型配置
    node_dtype1 = {
        0: "float",  # 输入节点使用 float
        1: "float",
        2: "float",
        3: "float",
        4: "float",
        5: "long",   # 输出节点使用 long
    }

    # 子网络 2 配置
    node_configs2 = {
        0: (64, 1, 1, 1),  # 输入
        1: (128, 1, 1, 1),  # 输出
        2: (128, 1, 1, 1),  # 输出
    }
    hyperedge_configs2 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [],
            },
        },
        "e2": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [],
            },
        },
    }
    # 子网络 2 的数据类型配置
    node_dtype2 = {
        0: "long",  # 输入节点使用 long
        1: "long",  # 输出节点使用 long
        2: "long",
    }

    # 子网络 3 配置
    node_configs3 = node_configs2
    hyperedge_configs3 = hyperedge_configs2.copy()
    hyperedge_configs3.update({
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [(128, 1, 1, 1)],
                "acts": ["sigmoid"],
            },
        },
    })
    # 子网络 3 的数据类型配置（与子网络 2 相同）
    node_dtype3 = node_dtype2

    # 节点映射
    node_mapping = [
        (100, "net1", 0),  # 全局输入 → 子网络 1 节点 0
        (101, "net1", 1),  # 全局输入 → 子网络 1 节点 1
        (102, "net1", 5),  # 子网络 1 节点 5 输出
        (102, "net2", 0),  # 子网络 2 节点 0 输入
        (102, "net3", 0),  # 子网络 3 节点 0 输入
        (103, "net2", 1),  # 子网络 2 节点 1 → 全局输出
        (104, "net2", 2),  # 子网络 2 节点 2 → 全局输出
        (105, "net3", 1),  # 子网络 3 节点 1 → 全局输出
        (106, "net3", 2),  # 子网络 3 节点 2 → 全局输出
    ]

    # 运行示例
    run_example(
        sub_network_configs=[
            (node_configs1, hyperedge_configs1, [0, 1], [5], node_dtype1),  # 子网络 1
            (node_configs2, hyperedge_configs2, [0], [1, 2], node_dtype2),  # 子网络 2
            (node_configs3, hyperedge_configs3, [0], [1, 2], node_dtype3),  # 子网络 3
        ],
        node_mapping=node_mapping,
        in_nodes=[100, 101],  # 全局输入
        out_nodes=[103, 104, 105, 106],  # 全局输出
        num_dimensions=3,
        input_shapes=[(2, 4, 64, 64, 64), (2, 1, 64, 64, 64)],  # 对应 100 和 101
        onnx_filename="MHDNet_example.onnx",
    )


if __name__ == "__main__":
    example_mhdnet()
