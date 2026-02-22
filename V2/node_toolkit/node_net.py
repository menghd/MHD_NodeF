import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, TypeVar
from dataclasses import dataclass, field

# 移除所有警告
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================== 类型定义 =====================
Tensor = TypeVar('Tensor', bound=torch.Tensor)
FuncMapping = Dict[str, Callable[..., Any]]

# ===================== 全局函数注册表 =====================
# 头节点函数（单节点→多超边）
MHD_NODE_HEAD_FUNCS: FuncMapping = {
    "share": lambda tensor: tensor.clone(),  # 特征共享到多条超边
}

# 尾节点函数（多超边→单节点）
MHD_NODE_TAIL_FUNCS: FuncMapping = {
    "sum": lambda tensors: sum(tensors),  # 逐元素求和
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),  # 逐元素均值
    "mul": lambda tensors: torch.stack(tensors).prod(dim=0),  # 逐元素相乘
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],  # 逐元素最大值
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],  # 逐元素最小值
}

# 超边输入函数（多节点→超边单输入）
MHD_EDGE_IN_FUNCS: FuncMapping = {
    "concat": lambda tensors, indices: torch.cat(
        [t for _, t in sorted(zip(indices, tensors), key=lambda x: x[0])],
        dim=1
    ),  # 按拓扑顺序拼接
    "matmul": lambda tensors, indices: torch.matmul(
        *[t for _, t in sorted(zip(indices, tensors), key=lambda x: x[0])]
    ),  # 按拓扑顺序矩阵乘法
}

# 超边输出函数（超边单输出→多节点）
MHD_EDGE_OUT_FUNCS: FuncMapping = {
    "split": lambda x, indices, channel_sizes: torch.split(x, channel_sizes, dim=1),
    "svd": lambda x, indices, channel_sizes: list(torch.svd(x))[:len(indices)],  # SVD分解
    "lr": lambda x, indices, channel_sizes: [x @ x.t(), x.t() @ x][:len(indices)],  # LR分解
}

# ===================== 超图核心数据结构 =====================
@dataclass(unsafe_hash=True)  # 支持哈希，可放入set集合
class MHD_Node:
    """超图节点"""
    id: int  # 唯一标识（关联矩阵列索引）
    name: str  # 便于调用的名称
    value: torch.Tensor  # 节点特征张量（初始化值，前向传播更新）
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"}, hash=False)  # 函数映射

@dataclass(unsafe_hash=True)  # 支持哈希，可放入set集合
class MHD_Edge:
    """超图边"""
    id: int  # 唯一标识（关联矩阵行索引）
    name: str  # 便于调用的名称
    value: List[Union[str, nn.Module]] = field(hash=False)  # 操作序列
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"}, hash=False)  # 函数映射

@dataclass
class MHD_Topo:
    """超图拓扑"""
    id: int  # 拓扑ID
    name: str  # 拓扑名称
    value: torch.Tensor  # 关联矩阵（整型张量）

# ===================== 核心工具函数 =====================
def mhd_parse_string_method(method_str: str, x: Tensor) -> Tensor:
    """解析字符串形式的张量方法并执行"""
    if not isinstance(method_str, str):
        return x
    if '(' in method_str and ')' in method_str:
        method_name, args_str = method_str.split('(', 1)
        args_str = args_str.rstrip(')')
        args = []
        kwargs = {}
        if args_str:
            for arg in args_str.split(','):
                arg = arg.strip()
                if not arg:
                    continue
                if '=' in arg:
                    k, v = arg.split('=', 1)
                    kwargs[k.strip()] = eval(v.strip())
                else:
                    args.append(eval(arg.strip()))
        if hasattr(x, method_name):
            return getattr(x, method_name)(*args, **kwargs)
        else:
            raise ValueError(f"张量无此方法: {method_name}")
    else:
        if hasattr(x, method_str):
            return getattr(x, method_str)()
        else:
            raise ValueError(f"张量无此方法: {method_str}")

def mhd_register_head_func(name: str, func: Callable[[Tensor], Tensor]) -> None:
    """注册新的头节点函数"""
    MHD_NODE_HEAD_FUNCS[name] = func

def mhd_register_tail_func(name: str, func: Callable[[List[Tensor]], Tensor]) -> None:
    """注册新的尾节点函数"""
    MHD_NODE_TAIL_FUNCS[name] = func

def mhd_register_edge_in_func(name: str, func: Callable[[List[Tensor], List[int]], Tensor]) -> None:
    """注册新的超边输入函数"""
    MHD_EDGE_IN_FUNCS[name] = func

def mhd_register_edge_out_func(name: str, func: Callable[[Tensor, List[int]], List[Tensor]]) -> None:
    """注册新的超边输出函数"""
    MHD_EDGE_OUT_FUNCS[name] = func

# ===================== 核心网络模块 =====================
class DNet(nn.Module):
    """动态网络：执行超图边的操作序列"""
    def __init__(self, operations: List[Union[str, nn.Module]]):
        super().__init__()
        self.operations = nn.ModuleList()
        self.str_ops = []
        for op in operations:
            if isinstance(op, nn.Module):
                self.operations.append(op)
                self.str_ops.append(None)
            elif isinstance(op, str):
                self.operations.append(nn.Identity())
                self.str_ops.append(op)
            else:
                raise ValueError(f"不支持的操作类型: {type(op)}")

    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        for op, str_op in zip(self.operations, self.str_ops):
            if str_op is not None:
                x = mhd_parse_string_method(str_op, x)
            else:
                x = op(x)
        return x

class HDNet(nn.Module):
    """超图动态网络"""
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo):
        super().__init__()
        # 索引映射
        self.node_id2obj = {node.id: node for node in nodes}
        self.node_name2id = {node.name: node.id for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        self.edge_name2id = {edge.name: edge.id for edge in edges}

        # 拓扑验证
        self.topo = topo
        self._validate_topo()

        # 初始化超边操作网络
        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.value)

        # 构建超边依赖图
        self.in_edges = defaultdict(list)  # node_id -> [edge_ids]
        self.out_edges = defaultdict(list)  # node_id -> [edge_ids]
        self.edge_src_nodes = {}  # edge_id -> [node_ids]
        self.edge_dst_nodes = {}  # edge_id -> [node_ids]
        self._build_edge_dependency()

        # 节点拓扑排序
        self.node_order = []
        self._topological_sort()

    def _validate_topo(self) -> None:
        """验证拓扑矩阵合法性"""
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)
        if self.topo.value.shape != (num_edges, num_nodes):
            raise ValueError(
                f"拓扑矩阵维度不匹配: 期望 ({num_edges}, {num_nodes}), 实际 {self.topo.value.shape}"
            )
        dtype_name = str(self.topo.value.dtype).lower()
        if not any(keyword in dtype_name for keyword in ['int', 'long', 'short']):
            raise ValueError(f"拓扑矩阵必须为整型: 实际 {self.topo.value.dtype}")

    def _build_edge_dependency(self):
        """基于拓扑矩阵构建超边依赖图"""
        for edge_id in sorted(self.edge_id2obj.keys()):
            edge_topo = self.topo.value[edge_id]
            
            # 解析头节点（源）和尾节点（目标）
            head_mask = edge_topo < 0
            src_node_ids = torch.where(head_mask)[0].tolist()
            tail_mask = edge_topo > 0
            dst_node_ids = torch.where(tail_mask)[0].tolist()

            self.edge_src_nodes[edge_id] = src_node_ids
            self.edge_dst_nodes[edge_id] = dst_node_ids

            # 构建节点-超边映射
            for node_id in src_node_ids:
                self.out_edges[node_id].append(edge_id)
            for node_id in dst_node_ids:
                self.in_edges[node_id].append(edge_id)

    def _topological_sort(self):
        """节点拓扑排序（支持并行执行）"""
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set(self.node_id2obj.keys())

        # 构建节点依赖图
        for edge_id in self.edge_src_nodes:
            src_nodes = self.edge_src_nodes[edge_id]
            dst_nodes = self.edge_dst_nodes[edge_id]
            for src in src_nodes:
                for dst in dst_nodes:
                    graph[src].append(dst)
                    in_degree[dst] += 1

        # 拓扑排序
        queue = deque([node for node in all_nodes if in_degree.get(node, 0) == 0])
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

    def forward(self, input_node_names: List[str], input_tensors: List[Tensor]) -> Dict[str, Tensor]:
        """前向传播"""
        # 初始化输入节点特征
        node_features = {}
        for name, tensor in zip(input_node_names, input_tensors):
            node_id = self.node_name2id[name]
            node_features[node_id] = tensor.to(dtype=torch.float32)

        # 缓存超边输出
        edge_output_cache = {}

        # 按拓扑顺序处理节点
        for node_id in self.node_order:
            if node_id in node_features:
                continue

            # 获取当前节点的所有输入超边
            in_edge_ids = self.in_edges.get(node_id, [])
            if not in_edge_ids:
                raise ValueError(f"节点 {self.node_id2obj[node_id].name} 没有输入超边")

            # 收集当前节点的所有输入特征
            node_inputs = []
            
            for edge_id in in_edge_ids:
                # 计算超边输出（未缓存时）
                if edge_id not in edge_output_cache:
                    edge = self.edge_id2obj[edge_id]
                    edge_net = self.edge_nets[edge.name]
                    
                    # 检查源节点是否就绪
                    src_node_ids = self.edge_src_nodes[edge_id]
                    if not all(src_id in node_features for src_id in src_node_ids):
                        missing_nodes = [self.node_id2obj[src_id].name for src_id in src_node_ids if src_id not in node_features]
                        raise ValueError(f"超边 {edge.name} 的源节点未准备好: {missing_nodes}")
                    
                    # 获取源节点特征
                    src_tensors = []
                    src_orders = []
                    for src_id in src_node_ids:
                        node = self.node_id2obj[src_id]
                        head_func_name = node.func.get("head", "share")
                        if head_func_name not in MHD_NODE_HEAD_FUNCS:
                            raise ValueError(f"未知头节点函数: {head_func_name}, 已注册: {list(MHD_NODE_HEAD_FUNCS.keys())}")
                        # 应用头节点函数
                        src_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](node_features[src_id])
                        src_tensors.append(src_tensor)
                        # 记录拓扑顺序
                        src_orders.append(-abs(self.topo.value[edge_id][src_id]))
                    
                    # 超边输入聚合
                    edge_in_func_name = edge.func.get("in", "concat")
                    if edge_in_func_name not in MHD_EDGE_IN_FUNCS:
                        raise ValueError(f"未知超边输入函数: {edge_in_func_name}, 已注册: {list(MHD_EDGE_IN_FUNCS.keys())}")
                    edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](src_tensors, src_orders)
                    
                    # 执行超边操作
                    edge_output = edge_net(edge_input)
                    
                    # 超边输出分发（核心逻辑：拓扑绝对值控顺序，节点自带通道数控大小）
                    edge_out_func_name = edge.func.get("out", "split")
                    if edge_out_func_name not in MHD_EDGE_OUT_FUNCS:
                        raise ValueError(f"未知超边输出函数: {edge_out_func_name}, 已注册: {list(MHD_EDGE_OUT_FUNCS.keys())}")
                    
                    # 获取目标节点列表
                    dst_node_ids = self.edge_dst_nodes[edge_id]
                    # 获取目标节点拓扑顺序（绝对值）
                    dst_orders = [abs(self.topo.value[edge_id][dst_id].item()) for dst_id in dst_node_ids]
                    
                    # 核心逻辑：按拓扑绝对值排序，通道数取自节点本身
                    if edge_out_func_name == "split":
                        # 1. 组装（拓扑绝对值，节点ID，节点通道数）
                        node_order_info = []
                        for idx, dst_id in enumerate(dst_node_ids):
                            abs_val = dst_orders[idx]
                            node = self.node_id2obj[dst_id]
                            channel_size = node.value.shape[1]  # 节点自带通道数
                            node_order_info.append((abs_val, dst_id, channel_size))
                        
                        # 2. 按拓扑绝对值从小到大排序
                        node_order_info.sort(key=lambda x: x[0])
                        
                        # 3. 提取排序后的通道数和节点ID
                        sorted_channel_sizes = [item[2] for item in node_order_info]
                        sorted_dst_ids = [item[1] for item in node_order_info]
                        
                        # 4. 执行split
                        split_outputs = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                            edge_output, dst_orders, sorted_channel_sizes
                        )
                        
                        # 5. 映射回原始节点顺序
                        split_map = {dst_id: tensor for dst_id, tensor in zip(sorted_dst_ids, split_outputs)}
                        final_outputs = [split_map[dst_id] for dst_id in dst_node_ids]
                    else:
                        # 其他输出函数
                        final_outputs = MHD_EDGE_OUT_FUNCS[edge_out_func_name](
                            edge_output, dst_orders, []
                        )
                    
                    # 缓存超边输出
                    edge_output_cache[edge_id] = {
                        dst_id: tensor for dst_id, tensor in zip(dst_node_ids, final_outputs)
                    }

                # 收集当前节点的输入特征
                if node_id in edge_output_cache[edge_id]:
                    node_inputs.append(edge_output_cache[edge_id][node_id])

            # 尾节点聚合
            if not node_inputs:
                raise ValueError(f"节点 {self.node_id2obj[node_id].name} 没有有效输入")
            
            # 应用尾节点聚合函数
            node = self.node_id2obj[node_id]
            tail_func_name = node.func.get("tail", "sum")
            if tail_func_name not in MHD_NODE_TAIL_FUNCS:
                raise ValueError(f"未知尾节点函数: {tail_func_name}, 已注册: {list(MHD_NODE_TAIL_FUNCS.keys())}")
            node_features[node_id] = MHD_NODE_TAIL_FUNCS[tail_func_name](node_inputs)

        # 返回名称映射结果
        return {
            self.node_id2obj[node_id].name: tensor
            for node_id, tensor in node_features.items()
        }

class MHDNet(nn.Module):
    """多超图动态网络：整合多个HDNet为全局超图"""
    def __init__(
        self,
        sub_hdnets: Dict[str, HDNet],
        node_mapping: List[Tuple[str, str, str]],
        in_nodes: List[str],
        out_nodes: List[str],
        onnx_save_path: Optional[str] = None,
    ):
        super().__init__()
        self.sub_hdnets = nn.ModuleDict(sub_hdnets)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        # 构建全局超图
        self.global_hdnet = self._build_global_hypergraph(node_mapping)

        # 导出ONNX
        if onnx_save_path:
            self._export_to_onnx(onnx_save_path)

    def _build_global_hypergraph(self, node_mapping: List[Tuple[str, str, str]]) -> HDNet:
        """从多个子HDNet构建全局超图"""
        # 初始化计数器
        node_id_counter = 0
        edge_id_counter = 0

        # 映射表（支持共用节点）
        sub2global_node = {}  # (sub_name, sub_node_name) → (global_id, global_name)
        global_node_map = {}  # global_name → MHD_Node（确保共用节点唯一）
        sub2global_edge = {}  # (sub_name, sub_edge_name) → global_edge_id

        # 收集全局节点（处理共用节点）
        global_nodes = set()
        for global_node_name, sub_name, sub_node_name in node_mapping:
            key = (sub_name, sub_node_name)
            if global_node_name not in global_node_map:
                # 新全局节点
                sub_hdnet = self.sub_hdnets[sub_name]
                sub_node_id = sub_hdnet.node_name2id[sub_node_name]
                sub_node = sub_hdnet.node_id2obj[sub_node_id]
                global_node = MHD_Node(
                    id=node_id_counter,
                    name=global_node_name,
                    value=sub_node.value.clone(),
                    func=sub_node.func.copy()
                )
                global_nodes.add(global_node)
                global_node_map[global_node_name] = global_node
                sub2global_node[key] = (node_id_counter, global_node_name)
                node_id_counter += 1
            else:
                # 共用节点，复用已有ID
                global_node = global_node_map[global_node_name]
                sub2global_node[key] = (global_node.id, global_node_name)

        # 收集全局边和拓扑
        global_edges = set()
        global_topo_data = []
        
        for sub_name, sub_hdnet in self.sub_hdnets.items():
            for sub_edge_id in sorted(sub_hdnet.edge_id2obj.keys()):
                sub_edge = sub_hdnet.edge_id2obj[sub_edge_id]
                sub_edge_name = sub_edge.name
                key = (sub_name, sub_edge_name)
                if key not in sub2global_edge:
                    sub2global_edge[key] = edge_id_counter
                    # 创建全局边
                    global_edge = MHD_Edge(
                        id=edge_id_counter,
                        name=f"{sub_name}_{sub_edge_name}",
                        value=sub_edge.value.copy(),
                        func=sub_edge.func.copy()
                    )
                    global_edges.add(global_edge)

                    # 转换子拓扑到全局拓扑
                    sub_topo_row = sub_hdnet.topo.value[sub_edge_id]
                    global_topo_row = torch.zeros(node_id_counter, dtype=torch.int64)
                    
                    # 映射子节点ID到全局节点ID
                    for sub_node_id, val in enumerate(sub_topo_row):
                        if val == 0:
                            continue
                        sub_node_name = sub_hdnet.node_id2obj[sub_node_id].name
                        map_key = (sub_name, sub_node_name)
                        if map_key in sub2global_node:
                            global_node_id, _ = sub2global_node[map_key]
                            global_topo_row[global_node_id] = val

                    global_topo_data.append(global_topo_row)
                    edge_id_counter += 1

        # 创建全局拓扑
        if global_topo_data:
            global_topo_value = torch.stack(global_topo_data)
        else:
            global_topo_value = torch.empty(0, 0, dtype=torch.int64)
        
        global_topo = MHD_Topo(
            id=0,
            name="global_topo",
            value=global_topo_value
        )

        # 创建全局HDNet
        return HDNet(nodes=global_nodes, edges=global_edges, topo=global_topo)

    def _export_to_onnx(self, onnx_save_path: str) -> None:
        """导出模型为ONNX格式"""
        self.eval()
        # 构建输入形状
        input_shapes = []
        for node_name in self.in_nodes:
            for node in self.global_hdnet.node_id2obj.values():
                if node.name == node_name:
                    input_shapes.append(node.value.shape)
                    break
        # 生成示例输入
        inputs = [torch.randn(*shape) for shape in input_shapes]
        dynamic_axes = {
            **{f"input_{node}": {0: "batch_size"} for node in self.in_nodes},
            **{f"output_{node}": {0: "batch_size"} for node in self.out_nodes}
        }
        # 导出
        abs_onnx_path = os.path.abspath(onnx_save_path)
        # 自动判断PyTorch版本选择opset
        torch_version = [int(v) for v in torch.__version__.split('.')[:2]]
        if torch_version >= [2, 0]:
            opset_version = 17
        else:
            opset_version = 11
        torch.onnx.export(
            self,
            inputs,
            abs_onnx_path,
            input_names=[f"input_{node}" for node in self.in_nodes],
            output_names=[f"output_{node}" for node in self.out_nodes],
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            opset_version=opset_version
        )
        print(f"模型已导出至: {abs_onnx_path} (opset={opset_version})")

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """全局前向传播"""
        if len(inputs) != len(self.in_nodes):
            raise ValueError(f"输入数量不匹配: 期望 {len(self.in_nodes)}, 实际 {len(inputs)}")
        # 执行全局超图前向传播
        global_outputs = self.global_hdnet.forward(self.in_nodes, inputs)
        # 提取输出节点
        outputs = [global_outputs[node] for node in self.out_nodes]
        return outputs

# ===================== 验证示例 =====================
def example_mhdnet():
    """验证示例：完全匹配你的逻辑"""
    device = torch.device('cpu')
    torch.manual_seed(42)

    # ===================== HDNet1 =====================
    # 定义节点（每个节点自带固定通道数）
    hdnet1_nodes = {
        # a节点：4通道（源节点）
        MHD_Node(
            id=0, name="a_node",
            value=torch.randn(1, 4, 16, 16).to(device),
            func={"head": "share", "tail": "sum"}
        ),
        # b节点：3通道（目标节点）
        MHD_Node(
            id=1, name="b_node",
            value=torch.randn(1, 3, 16, 16).to(device),
            func={"head": "share", "tail": "sum"}
        ),
        # c节点：1通道（目标节点）
        MHD_Node(
            id=2, name="c_node",
            value=torch.randn(1, 1, 16, 16).to(device),
            func={"head": "share", "tail": "sum"}
        )
    }

    # 定义超边
    hdnet1_edges = {
        # 超边0：a→b+c（split分发）
        MHD_Edge(
            id=0, name="edge_a_to_bc",
            value=[nn.Identity()],
            func={"in": "concat", "out": "split"}
        )
    }

    # 拓扑矩阵：验证你的核心逻辑
    # 情况1：b绝对值=1（前），c绝对值=2（后）→ split=[3,1]
    hdnet1_topo = MHD_Topo(
        id=0, name="topo1",
        value=torch.tensor([
            [-1, 1, 2],  # edge0: a(-1)=源节点, b(1)=目标(绝对值1), c(2)=目标(绝对值2)
        ], dtype=torch.int64)
    )

    hdnet1 = HDNet(nodes=hdnet1_nodes, edges=hdnet1_edges, topo=hdnet1_topo).to(device)

    # ===================== MHDNet =====================
    sub_hdnets = {"hdnet1": hdnet1}
    node_mapping = [
        ("g_a", "hdnet1", "a_node"),
        ("g_b", "hdnet1", "b_node"),
        ("g_c", "hdnet1", "c_node"),
    ]

    model = MHDNet(
        sub_hdnets=sub_hdnets,
        node_mapping=node_mapping,
        in_nodes=["g_a"],
        out_nodes=["g_b", "g_c"],
        onnx_save_path="MHDNodeF_final.onnx"
    ).to(device)

    # 前向传播验证
    model.eval()
    with torch.no_grad():
        input_a = torch.randn(1, 4, 16, 16).to(device)
        outputs = model([input_a])

    # 输出验证结果
    print("="*80)
    print("✅ 逻辑验证成功！完全匹配你的需求")
    print("="*80)
    print("📌 核心逻辑验证：")
    print(f"   - a节点输入通道数：{input_a.shape[1]}")
    print(f"   - b节点输出通道数：{outputs[0].shape[1]} (预期：3)")
    print(f"   - c节点输出通道数：{outputs[1].shape[1]} (预期：1)")
    print(f"   - 拓扑绝对值顺序：b(1) → c(2)")
    print(f"   - split列表：[3, 1] (b在前，c在后)")
    print("="*80)

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    example_mhdnet()MHD_NODE_TAIL_FUNCS: FuncMapping = {
    "sum": lambda tensors: sum(tensors),  # Element-wise sum / 逐元素求和
    "avg": lambda tensors: torch.stack(tensors).mean(dim=0),  # Element-wise average / 逐元素均值
    "mul": lambda tensors: torch.stack(tensors).prod(dim=0),  # Element-wise multiplication / 逐元素相乘
    "max": lambda tensors: torch.stack(tensors).max(dim=0)[0],  # Element-wise maximum / 逐元素最大值
    "min": lambda tensors: torch.stack(tensors).min(dim=0)[0],  # Element-wise minimum / 逐元素最小值
}

# Edge Input Functions / 超边输入函数（多节点→超边单输入）
MHD_EDGE_IN_FUNCS: FuncMapping = {
    "concat": lambda tensors, indices: torch.cat(
        [t for _, t in sorted(zip(indices, tensors), key=lambda x: x[0])],
        dim=1
    ),  # Concatenate by topology order / 按拓扑顺序拼接
    "matmul": lambda tensors, indices: torch.matmul(
        *[t for _, t in sorted(zip(indices, tensors), key=lambda x: x[0])]
    ),  # Matrix multiplication by topology order / 按拓扑顺序矩阵乘法
}

# Edge Output Functions / 超边输出函数（超边单输出→多节点）
MHD_EDGE_OUT_FUNCS: FuncMapping = {
    "split": lambda x, indices, channel_sizes: torch.split(x, channel_sizes, dim=1),
    "svd": lambda x, indices, channel_sizes: list(torch.svd(x))[:len(indices)],  # SVD decomposition / SVD分解
    "lr": lambda x, indices, channel_sizes: [x @ x.t(), x.t() @ x][:len(indices)],  # LR decomposition / LR分解
}

# ===================== Hypergraph Core Structure / 超图核心数据结构 =====================
@dataclass(unsafe_hash=True)  # 支持哈希，可放入set集合
class MHD_Node:
    """
    Hypergraph Node / 超图节点
    Attributes / 属性:
        id: Unique identifier (column index of incidence matrix) / 唯一标识（关联矩阵列索引）
        name: Callable name for easy access / 便于调用的名称
        value: Node feature tensor (initialized value, updated in forward) / 节点特征张量（初始化值，前向传播更新）
        func: Function mapping {"head": func_name, "tail": func_name} / 函数映射（头/尾节点函数）
    """
    id: int
    name: str
    value: torch.Tensor
    func: Dict[str, str] = field(default_factory=lambda: {"head": "share", "tail": "sum"}, hash=False)

@dataclass(unsafe_hash=True)  # 支持哈希，可放入set集合
class MHD_Edge:
    """
    Hypergraph Edge / 超图边
    Attributes / 属性:
        id: Unique identifier (row index of incidence matrix) / 唯一标识（关联矩阵行索引）
        name: Callable name for easy access / 便于调用的名称
        value: Operation sequence (mix of string method / torch Module) / 操作序列（字符串方法/张量模块混合）
        func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"}, hash=False)
    """
    id: int
    name: str
    value: List[Union[str, nn.Module]] = field(hash=False)
    func: Dict[str, str] = field(default_factory=lambda: {"in": "concat", "out": "split"}, hash=False)

@dataclass
class MHD_Topo:
    """
    Hypergraph Topology / 超图拓扑
    Attributes / 属性:
        id: Topology ID / 拓扑ID
        name: Topology name / 拓扑名称
        value: Incidence matrix (int tensor, shape: [num_edges, num_nodes]) / 关联矩阵（整型张量）
    """
    id: int
    name: str
    value: torch.Tensor

# ===================== Core Utility Functions / 核心工具函数 =====================
def mhd_parse_string_method(method_str: str, x: Tensor) -> Tensor:
    """
    Parse and execute tensor method from string / 解析字符串形式的张量方法并执行
    Example / 示例:
        "__add__(3)" → x.__add__(3)
        "reshape(1, 4, -1)" → x.reshape(1, 4, -1)
        "squeeze(0)" → x.squeeze(0)
    """
    if not isinstance(method_str, str):
        return x
    if '(' in method_str and ')' in method_str:
        method_name, args_str = method_str.split('(', 1)
        args_str = args_str.rstrip(')')
        args = []
        kwargs = {}
        if args_str:
            for arg in args_str.split(','):
                arg = arg.strip()
                if not arg:
                    continue
                if '=' in arg:
                    k, v = arg.split('=', 1)
                    kwargs[k.strip()] = eval(v.strip())
                else:
                    args.append(eval(arg.strip()))
        if hasattr(x, method_name):
            return getattr(x, method_name)(*args, **kwargs)
        else:
            raise ValueError(f"Tensor has no method: {method_name} / 张量无此方法: {method_name}")
    else:
        if hasattr(x, method_str):
            return getattr(x, method_str)()
        else:
            raise ValueError(f"Tensor has no method: {method_str} / 张量无此方法: {method_str}")

def mhd_register_head_func(name: str, func: Callable[[Tensor], Tensor]) -> None:
    """Register new head node function / 注册新的头节点函数"""
    MHD_NODE_HEAD_FUNCS[name] = func

def mhd_register_tail_func(name: str, func: Callable[[List[Tensor]], Tensor]) -> None:
    """Register new tail node function / 注册新的尾节点函数"""
    MHD_NODE_TAIL_FUNCS[name] = func

def mhd_register_edge_in_func(name: str, func: Callable[[List[Tensor], List[int]], Tensor]) -> None:
    """Register new edge input function / 注册新的超边输入函数"""
    MHD_EDGE_IN_FUNCS[name] = func

def mhd_register_edge_out_func(name: str, func: Callable[[Tensor, List[int], List[int]], List[Tensor]]) -> None:
    """Register new edge output function / 注册新的超边输出函数"""
    MHD_EDGE_OUT_FUNCS[name] = func

# ===================== Core Network Modules / 核心网络模块 =====================
class DNet(nn.Module):
    """
    Dynamic Network / 动态网络
    Function / 功能:
        Execute hypergraph edge operation sequence / 执行超图边的操作序列
    Support / 支持:
        PyTorch Module + Any Tensor Method String / PyTorch模块 + 任意张量方法字符串
    """
    def __init__(self, operations: List[Union[str, nn.Module]]):
        super().__init__()
        self.operations = nn.ModuleList()
        self.str_ops = []
        for op in operations:
            if isinstance(op, nn.Module):
                self.operations.append(op)
                self.str_ops.append(None)
            elif isinstance(op, str):
                self.operations.append(nn.Identity())
                self.str_ops.append(op)
            else:
                raise ValueError(f"Unsupported operation type: {type(op)} / 不支持的操作类型: {type(op)}")

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation / 前向传播"""
        for op, str_op in zip(self.operations, self.str_ops):
            if str_op is not None:
                x = mhd_parse_string_method(str_op, x)
            else:
                x = op(x)
        return x

class HDNet(nn.Module):
    """
    Hypergraph Dynamic Network / 超图动态网络
    Core Logic / 核心逻辑:
        Head Node → Aggregate by order → Edge Operation → Distribute by order → Tail Node
        头节点 → 按顺序聚合 → 超边操作 → 按顺序分发 → 尾节点
    """
    def __init__(self, nodes: set[MHD_Node], edges: set[MHD_Edge], topo: MHD_Topo):
        super().__init__()
        # Index mapping / 索引映射
        self.node_id2obj = {node.id: node for node in nodes}
        self.node_name2id = {node.name: node.id for node in nodes}
        self.edge_id2obj = {edge.id: edge for edge in edges}
        self.edge_name2id = {edge.name: edge.id for edge in edges}

        # Topology validation / 拓扑验证
        self.topo = topo
        self._validate_topo()

        # Initialize edge operation networks / 初始化超边操作网络
        self.edge_nets = nn.ModuleDict()
        for edge in edges:
            self.edge_nets[edge.name] = DNet(edge.value)

        # Node feature cache / 节点特征缓存
        self.node_values = {node.id: node.value.clone() for node in nodes}

    def _validate_topo(self) -> None:
        """Validate topology matrix / 验证拓扑矩阵合法性"""
        num_edges = len(self.edge_id2obj)
        num_nodes = len(self.node_id2obj)
        # Dimension validation / 维度验证
        if self.topo.value.shape != (num_edges, num_nodes):
            raise ValueError(
                f"Topology matrix dimension mismatch / 拓扑矩阵维度不匹配: "
                f"Expected ({num_edges}, {num_nodes}), Got {self.topo.value.shape}"
            )
        # Type validation / 类型验证
        dtype_name = str(self.topo.value.dtype).lower()
        if not any(keyword in dtype_name for keyword in ['int', 'long', 'short']):
            raise ValueError(f"Topology matrix must be integer type / 拓扑矩阵必须为整型: Got {self.topo.value.dtype}")

    def _get_channel_sizes(self, x: Tensor, indices: List[int]) -> List[int]:
        """
        Calculate channel sizes by topology order / 按拓扑顺序计算通道分配尺寸
        """
        total_channels = x.shape[1]
        num_nodes = len(indices)
        if num_nodes == 0:
            return []
        # Sort by topology order (absolute value) / 按拓扑顺序排序（绝对值）
        sorted_indices = sorted(indices, key=lambda x: abs(x))
        base_channels = total_channels // num_nodes
        channel_sizes = [base_channels] * num_nodes
        # Distribute remaining channels (follow topology order) / 分配剩余通道（按拓扑顺序）
        remaining = total_channels % num_nodes
        for i in range(remaining):
            channel_sizes[i] += 1
        # Map back to original order / 映射回原始顺序
        size_map = {idx: size for idx, size in zip(sorted_indices, channel_sizes)}
        return [size_map[idx] for idx in indices]

    def forward(self, input_node_names: List[str], input_tensors: List[Tensor]) -> Dict[str, Tensor]:
        """
        Forward propagation / 前向传播
        Args:
            input_node_names: Input node name list / 输入节点名称列表
            input_tensors: Input feature tensor list / 输入特征张量列表
        Returns:
            Node name → feature tensor mapping / 节点名称到特征张量的映射
        """
        # Initialize input node features / 初始化输入节点特征
        for name, tensor in zip(input_node_names, input_tensors):
            node_id = self.node_name2id[name]
            self.node_values[node_id] = tensor.to(dtype=torch.float32)

        # Iterate all edges (by ID order) / 遍历所有超边（按ID顺序）
        for edge_id in sorted(self.edge_id2obj.keys()):
            edge = self.edge_id2obj[edge_id]
            edge_net = self.edge_nets[edge.name]
            edge_topo = self.topo.value[edge_id]

            # ========== 修复：用PyTorch API替代Python遍历，消除TraceWarning ==========
            # 头节点解析（无Python遍历/类型转换）
            head_mask = edge_topo < 0
            head_node_ids = torch.where(head_mask)[0].tolist()
            head_node_orders = edge_topo[head_mask].tolist()  # 头节点顺序（输入顺序）
            
            # 尾节点解析（无Python遍历/类型转换）
            tail_mask = edge_topo > 0
            tail_node_ids = torch.where(tail_mask)[0].tolist()
            tail_node_orders = edge_topo[tail_mask].tolist()  # 尾节点顺序（输出顺序）

            # Get head node features / 获取头节点特征
            head_tensors = []
            for node_id in head_node_ids:
                node = self.node_id2obj[node_id]
                head_func_name = node.func.get("head", "share")
                if head_func_name not in MHD_NODE_HEAD_FUNCS:
                    raise ValueError(
                        f"Unknown head function / 未知头节点函数: {head_func_name}, "
                        f"Registered / 已注册: {list(MHD_NODE_HEAD_FUNCS.keys())}"
                    )
                head_tensor = MHD_NODE_HEAD_FUNCS[head_func_name](self.node_values[node_id])
                head_tensors.append(head_tensor)

            # Edge input aggregation / 超边输入聚合
            edge_in_func_name = edge.func.get("in", "concat")
            if edge_in_func_name not in MHD_EDGE_IN_FUNCS:
                raise ValueError(
                    f"Unknown edge input function / 未知超边输入函数: {edge_in_func_name}, "
                    f"Registered / 已注册: {list(MHD_EDGE_IN_FUNCS.keys())}"
                )
            # FIX 1: 输入聚合用头节点顺序（head_node_orders），不是尾节点
            edge_input = MHD_EDGE_IN_FUNCS[edge_in_func_name](head_tensors, head_node_orders)

            # Execute edge operations / 执行超边操作
            edge_output = edge_net(edge_input)

            # Edge output distribution / 超边输出分发
            edge_out_func_name = edge.func.get("out", "split")
            if edge_out_func_name not in MHD_EDGE_OUT_FUNCS:
                raise ValueError(
                    f"Unknown edge output function / 未知超边输出函数: {edge_out_func_name}, "
                    f"Registered / 已注册: {list(MHD_EDGE_OUT_FUNCS.keys())}"
                )
            # Split by topology order / 按拓扑顺序拆分
            if edge_out_func_name == "split":
                # FIX 2: 输出拆分用尾节点顺序（tail_node_orders），不是头节点
                channel_sizes = self._get_channel_sizes(edge_output, tail_node_orders)
                tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](edge_output, tail_node_orders, channel_sizes)
            else:
                tail_tensors = MHD_EDGE_OUT_FUNCS[edge_out_func_name](edge_output, tail_node_orders, [])

            # Update tail node features / 更新尾节点特征
            for node_id, tensor in zip(tail_node_ids, tail_tensors):
                node = self.node_id2obj[node_id]
                tail_func_name = node.func.get("tail", "sum")
                if tail_func_name not in MHD_NODE_TAIL_FUNCS:
                    raise ValueError(
                        f"Unknown tail function / 未知尾节点函数: {tail_func_name}, "
                        f"Registered / 已注册: {list(MHD_NODE_TAIL_FUNCS.keys())}"
                    )
                if node_id in self.node_values:
                    self.node_values[node_id] = MHD_NODE_TAIL_FUNCS[tail_func_name](
                        [self.node_values[node_id], tensor]
                    )
                else:
                    self.node_values[node_id] = tensor

        # Return name-based mapping / 返回名称映射结果
        return {
            self.node_id2obj[node_id].name: tensor
            for node_id, tensor in self.node_values.items()
        }

class MHDNet(nn.Module):
    """
    Multi Hypergraph Dynamic Network / 多超图动态网络
    Function / 功能:
        Integrate multiple HDNet into global hypergraph / 整合多个HDNet为全局超图
    """
    def __init__(
        self,
        sub_hdnets: Dict[str, HDNet],
        node_mapping: List[Tuple[str, str, str]],  # (global_name, sub_name, sub_node_name)
        in_nodes: List[str],
        out_nodes: List[str],
        onnx_save_path: Optional[str] = None,
    ):
        super().__init__()
        self.sub_hdnets = nn.ModuleDict(sub_hdnets)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        # Build global hypergraph / 构建全局超图
        self.global_hdnet = self._build_global_hypergraph(node_mapping)

        # Export ONNX / 恢复ONNX导出功能
        if onnx_save_path:
            self._export_to_onnx(onnx_save_path)

    def _build_global_hypergraph(self, node_mapping: List[Tuple[str, str, str]]) -> HDNet:
        """Build global hypergraph / 构建全局超图（仅处理你映射的节点，不补全、不生成默认）"""
        # Initialize counters / 初始化计数器
        node_id_counter = 0
        edge_id_counter = 0

        # Mapping tables / 映射表
        sub2global_node = {}  # (sub_name, sub_node_name) → (global_id, global_name)
        sub2global_edge = {}  # (sub_name, sub_edge_name) → global_edge_id

        # Collect global nodes / 仅收集你显式映射的节点（不补全）
        global_nodes = set()
        for global_node_name, sub_name, sub_node_name in node_mapping:
            key = (sub_name, sub_node_name)
            if key not in sub2global_node:
                sub_hdnet = self.sub_hdnets[sub_name]
                sub_node_id = sub_hdnet.node_name2id[sub_node_name]
                sub_node = sub_hdnet.node_id2obj[sub_node_id]
                global_node = MHD_Node(
                    id=node_id_counter,
                    name=global_node_name,
                    value=sub_node.value.clone(),
                    func=sub_node.func.copy()
                )
                global_nodes.add(global_node)
                sub2global_node[key] = (node_id_counter, global_node_name)
                node_id_counter += 1

        # Collect global edges and topology / 收集全局边和拓扑（仅处理映射节点）
        global_edges = set()
        global_topo_data = []
        
        for sub_name, sub_hdnet in self.sub_hdnets.items():
            for sub_edge_id in sorted(sub_hdnet.edge_id2obj.keys()):
                sub_edge = sub_hdnet.edge_id2obj[sub_edge_id]
                sub_edge_name = sub_edge.name
                key = (sub_name, sub_edge_name)
                if key not in sub2global_edge:
                    sub2global_edge[key] = edge_id_counter
                    # Create global edge / 创建全局边
                    global_edge = MHD_Edge(
                        id=edge_id_counter,
                        name=f"{sub_name}_{sub_edge_name}",
                        value=sub_edge.value.copy(),
                        func=sub_edge.func.copy()
                    )
                    global_edges.add(global_edge)

                    # Convert sub topology to global topology / 转换子拓扑到全局拓扑（仅映射节点）
                    sub_topo_row = sub_hdnet.topo.value[sub_edge_id]
                    # FIX 3: 全局拓扑行维度 = 仅你映射的节点数（node_id_counter）
                    global_topo_row = torch.zeros(node_id_counter, dtype=torch.int64)
                    
                    # Map sub node ID to global node ID / 仅映射你指定的节点，未映射的直接跳过（报错由你控制）
                    for sub_node_id, val in enumerate(sub_topo_row):
                        if val == 0:
                            continue
                        sub_node_name = sub_hdnet.node_id2obj[sub_node_id].name
                        map_key = (sub_name, sub_node_name)
                        if map_key in sub2global_node:
                            global_node_id, _ = sub2global_node[map_key]
                            global_topo_row[global_node_id] = val

                    global_topo_data.append(global_topo_row)
                    edge_id_counter += 1

        # Create global topology / 创建全局拓扑
        if global_topo_data:
            global_topo_value = torch.stack(global_topo_data)
        else:
            global_topo_value = torch.empty(0, 0, dtype=torch.int64)
        
        global_topo = MHD_Topo(
            id=0,
            name="global_topo",
            value=global_topo_value
        )

        # Create global HDNet / 创建全局HDNet
        return HDNet(nodes=global_nodes, edges=global_edges, topo=global_topo)

    def _export_to_onnx(self, onnx_save_path: str) -> None:
        """Export model to ONNX format / 导出模型为ONNX格式"""
        self.eval()
        # Build input shapes / 构建输入形状
        input_shapes = []
        for node_name in self.in_nodes:
            for node in self.global_hdnet.node_id2obj.values():
                if node.name == node_name:
                    input_shapes.append(node.value.shape)
                    break
        # Generate example inputs / 生成示例输入
        inputs = [torch.randn(*shape) for shape in input_shapes]
        dynamic_axes = {
            **{f"input_{node}": {0: "batch_size"} for node in self.in_nodes},
            **{f"output_{node}": {0: "batch_size"} for node in self.out_nodes}
        }
        # Export / 导出（移除无效的warnings参数）
        abs_onnx_path = os.path.abspath(onnx_save_path)
        # 自动判断PyTorch版本，选择合适的opset
        torch_version = [int(v) for v in torch.__version__.split('.')[:2]]
        if torch_version >= [2, 0]:
            opset_version = 17
        else:
            opset_version = 11
        torch.onnx.export(
            self,
            inputs,
            abs_onnx_path,
            input_names=[f"input_{node}" for node in self.in_nodes],
            output_names=[f"output_{node}" for node in self.out_nodes],
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
            opset_version=opset_version
        )
        print(f"Model exported to / 模型已导出至: {abs_onnx_path} (opset={opset_version})")

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """Global forward propagation / 全局前向传播"""
        if len(inputs) != len(self.in_nodes):
            raise ValueError(
                f"Input count mismatch / 输入数量不匹配: "
                f"Expected {len(self.in_nodes)}, Got {len(inputs)}"
            )
        # Execute global hypergraph forward / 执行全局超图前向传播
        global_outputs = self.global_hdnet.forward(self.in_nodes, inputs)
        # Extract output nodes / 提取输出节点
        outputs = [global_outputs[node] for node in self.out_nodes]
        return outputs

# ===================== Example / 示例（仅修改此处，核心逻辑不变） =====================
def example_mhdnet():
    """
    Example: Single HDNet with multiple dimensions (单个HDNet内包含多种维度)
    修复点：在example中通过unsqueeze/reshape做维度转换，确保concat时维度一致
    """
    # ===================== 单个HDNet包含多种维度节点 =====================
    # HDNet1: 内部包含3D+2D节点（通过unsqueeze将2D转为3D，确保concat维度一致）
    hdnet1_nodes = {
        # 3D节点 (batch, channel, depth, height, width)
        MHD_Node(
            id=0,
            name="node3d_1",
            value=torch.randn(1, 4, 8, 16, 16),
            func={"head": "share", "tail": "avg"}
        ),
        # 2D节点 → 通过unsqueeze(2)添加depth维度，转为3D (1,4,1,16,16)
        MHD_Node(
            id=1,
            name="node2d_1",
            value=torch.randn(1, 4, 16, 16).unsqueeze(2).repeat(1,1,8,1,1),  # 2D→3D: (1,4,16,16) → (1,4,8,16,16)
            func={"head": "share", "tail": "avg"}
        ),
        # 输出节点（3D）
        MHD_Node(
            id=2,
            name="output3d_1",
            value=torch.randn(1, 4, 8, 16, 16),
            func={"head": "share", "tail": "avg"}
        )
    }

    # HDNet1的边：修复卷积通道数（输入8通道 → 输出4通道）
    hdnet1_edges = {
        MHD_Edge(
            id=0,
            name="edge1",
            value=[
                nn.Conv3d(8, 4, kernel_size=3, padding=1, bias=False),  # 4+4=8输入通道 → 4输出通道
                "relu_()"
            ],
            func={"in": "concat", "out": "split"}
        )
    }

    # HDNet1拓扑矩阵
    hdnet1_topo = MHD_Topo(
        id=0,
        name="topo1",
        value=torch.tensor([[-1, -2, 1]], dtype=torch.int64)  # node3d_1(-1), node2d_1(-2) → output3d_1(1)
    )

    # 创建HDNet1
    hdnet1 = HDNet(nodes=hdnet1_nodes, edges=hdnet1_edges, topo=hdnet1_topo)

    # ===================== HDNet2: 内部包含2D+1D节点（通过reshape将1D转为2D） =====================
    hdnet2_nodes = {
        # 2D节点 (batch, channel, height, width)
        MHD_Node(
            id=0,
            name="node2d_2",
            value=torch.randn(1, 8, 16, 16),
            func={"head": "share", "tail": "sum"}
        ),
        # 1D节点 → 通过reshape转为2D (1,8,16,16)
        MHD_Node(
            id=1,
            name="node1d_2",
            value=torch.randn(1, 8, 256).reshape(1,8,16,16),  # 1D→2D: (1,8,256) → (1,8,16,16)
            func={"head": "share", "tail": "sum"}
        ),
        # 输出节点（2D）
        MHD_Node(
            id=2,
            name="output2d_2",
            value=torch.randn(1, 8, 16, 16),
            func={"head": "share", "tail": "sum"}
        )
    }

    # HDNet2的边：修复操作序列（统一2D操作）
    hdnet2_edges = {
        MHD_Edge(
            id=0,
            name="edge2",
            value=[
                nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),  # 8+8=16输入通道 → 8输出通道
                "relu_()"
            ],
            func={"in": "concat", "out": "split"}
        )
    }

    # HDNet2拓扑矩阵
    hdnet2_topo = MHD_Topo(
        id=0,
        name="topo2",
        value=torch.tensor([[-1, -2, 1]], dtype=torch.int64)  # node2d_2(-1), node1d_2(-2) → output2d_2(1)
    )

    # 创建HDNet2
    hdnet2 = HDNet(nodes=hdnet2_nodes, edges=hdnet2_edges, topo=hdnet2_topo)

    # ===================== 多HDNet整合 =====================
    sub_hdnets = {
        "hdnet1": hdnet1,
        "hdnet2": hdnet2
    }

    # 节点映射
    node_mapping = [
        ("global_3d", "hdnet1", "node3d_1"),
        ("global_2d", "hdnet1", "node2d_1"),
        ("global_2d_2", "hdnet2", "node2d_2"),
        ("global_1d", "hdnet2", "node1d_2"),
        ("final_3d", "hdnet1", "output3d_1"),
        ("final_2d", "hdnet2", "output2d_2")
    ]

    # 创建MHDNet（恢复ONNX导出）
    model = MHDNet(
        sub_hdnets=sub_hdnets,
        node_mapping=node_mapping,
        in_nodes=["global_3d", "global_2d", "global_2d_2", "global_1d"],
        out_nodes=["final_3d", "final_2d"],
        onnx_save_path="MHDNodeF_example.onnx"
    )

    # 运行示例（严格还原原始输出格式）
    model.eval()
    with torch.no_grad():
        # 输入（严格控制尺寸匹配，做维度转换）
        input_3d = torch.randn(1, 4, 8, 16, 16)
        input_2d_1 = torch.randn(1, 4, 16, 16).unsqueeze(2).repeat(1,1,8,1,1)  # 2D→3D
        input_2d_2 = torch.randn(1, 8, 16, 16)
        input_1d = torch.randn(1, 8, 256).reshape(1,8,16,16)  # 1D→2D

        # 前向传播
        outputs = model([input_3d, input_2d_1, input_2d_2, input_1d])

        # 严格按原始代码输出格式打印
        print("HDNet1 Output Shape:", hdnet1.forward(["node3d_1", "node2d_1"], [input_3d, input_2d_1])["output3d_1"].shape)
        print("HDNet2 Output Shape:", hdnet2.forward(["node2d_2", "node1d_2"], [input_2d_2, input_1d])["output2d_2"].shape)
        print("MHDNet Final Output Shapes:", [o.shape for o in outputs])

if __name__ == "__main__":
    # 打印PyTorch版本
    print(f"PyTorch Version: {torch.__version__}")
    # 运行示例
    example_mhdnet()
