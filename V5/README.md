# MHD V5 — Memory、统一反向与稀疏拓扑

V5 保留 Node、Edge、Topo、Graph 四个核心类型，在独立 Node memory 的基础上，统一按所选 Level 推断标量反向起点、使用稀疏拓扑存储，并检查图合并的状态冲突。V4 源码和实验保持原样。

## V4 → V5

| 项目 | V4 | V5 |
|---|---|---|
| 默认聚合 | `replace` | `sum` |
| 是否包含旧态 | `sum/avg/max/min/mul` 总是包含 | 独立 `memory`，默认 `False` |
| 内置聚合 | `replace/sum/avg/max/min/mul` | `sum/avg/max/min/mul`；取消 `replace` |
| 自定义聚合 | `fn(current, incomings)` | 参数数量不变；关闭 memory 时 `current=None` |

`memory` 是 keyword-only bool。它只控制本次 Feature 聚合是否包含更新前的 Current State；不控制 reset、跨 batch 保留、梯度截断或 optimizer 梯度累积。Gradient Message 仍由真实 autograd 更新，不增加独立梯度聚合机制。

```python
import torch
from V5.MHD_Framework_V5 import MHD_Node

node = MHD_Node(
    id=0,
    name="state",
    feature_message=MHD_Node.Message(torch.tensor(0.0)),
    aggregation="avg",
    memory=False,
)
incoming = [torch.tensor(2.0), torch.tensor(4.0)]
result = node.aggregate_messages(node.feature_message.current_state, incoming)
assert result.item() == 3.0

node.memory = True
result = node.aggregate_messages(node.feature_message.current_state, incoming)
assert result.item() == 2.0
```

每次有 incoming 时，`memory=False` 只聚合这些消息；`memory=True` 把旧态作为额外一项。`avg` 的分母是本次参与的项数，不是历史消息累计数量。没有 incoming 时保持旧态。只有一条 incoming 且关闭 memory 时，五个内置聚合都直接得到该消息的值。

自定义函数仍负责实际运算，关闭 memory 时不会获得旧态：

```python
def aggregate(current, incomings):
    result = torch.stack(tuple(incomings)).sum(dim=0)
    return result if current is None else current + result
```

## 按 Level 统一反向

Forward/Backward 的 Level 列表保持用户给定顺序，允许重复和不连续，不自动排序或去重；同一轮前后向 Level 不得重叠。Backward 中的每次边执行必须反向匹配本次真实 Forward trace，重复边从最近的兼容执行开始匹配。

反向起点由**所选依赖范围**决定，最终 loss 与中间输出采用同一规则：必须有唯一可微终点，且 Tensor 的元素数量为 1。非标量先通过普通 Operation 归约为标量；不自动求和，不新增 backward_node、种子参数或局部求导接口。多个终点、错误顺序和未匹配 trace 均明确报错。已 detach 的指标不参与起点推断。

```python
import torch
from V5.MHD_Framework_V5 import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph

nodes = {
    MHD_Node(0, "x", MHD_Node.Message(torch.tensor(3.0, requires_grad=True))),
    MHD_Node(1, "h", MHD_Node.Message(torch.tensor(0.0))),
    MHD_Node(2, "loss", MHD_Node.Message(torch.tensor(0.0))),
}
edges = {
    MHD_Edge(0, "double", [MHD_Edge.Operation(lambda x: 2 * x)]),
    MHD_Edge(1, "square", [MHD_Edge.Operation(lambda h: h.square())]),
}
first = torch.tensor([[-1, 1, 0], [0, 0, 0]])
second = torch.tensor([[0, 0, 0], [0, -1, 1]])
first_sort = torch.tensor([[0, 1, 0], [0, 0, 0]])
second_sort = torch.tensor([[0, 0, 0], [0, 0, 1]])
topo = MHD_Topo(
    [first, second, -second, -first],
    [first_sort, second_sort, second_sort, first_sort],
)
graph = MHD_Graph(nodes, edges, {topo}, device="cpu")

# level 0: x -> h; level 1: h -> loss
# level 2: loss -> h; level 3: h -> x
graph.forward(levels=[0, 1])
graph.backward(levels=[2, 3])
assert graph.get_node_by_name("x").gradient_message.current_state.item() == 24.0

for node in graph.nodes:
    node.reset()
graph.forward(levels=[0, 1])
graph.backward(levels=[3])  # h is the selected scalar objective
assert graph.get_node_by_name("x").gradient_message.current_state.item() == 2.0
```

`backward(levels=[2])` 从 loss 反向经过 square；`backward(levels=[3])` 从 h 反向经过 double。传给原生 autograd 的种子为 1，Trainer 的 AMP/梯度累积使用同一根节点并应用相应缩放。每次仍只调用一次原生 autograd，不重新执行 Operation。未选边的真实前向输出通过梯度 hook 屏蔽。

内部记录聚合后的状态版本和 memory 依赖，避免节点覆盖后误用最新值作为历史起点。Gradient Message **始终对应当前 Feature Message**；旧版本可以参与求导，但旧版本梯度不写入该节点的 current Gradient，也不跨版本相加。模块参数依旧通过标准 `.grad` 访问，保留原有优化器和 `retain_graph` 管理行为。多次 backward 的参数梯度会按原有规则累积，开始独立优化目标前应调用原生 optimizer 的 `zero_grad`；当前输入叶 Tensor 的梯度仍按原有 MHD 行为每次清理。

非零 `gradient_message.initial_state` 不再作为额外种子注入，反向前会报错。初态字段继续保留，默认零状态不变。错误检查发生在 Graph 重置 Gradient Message、注册 hook 或修改 `.grad` 之前。

## 统一稀疏拓扑

字段仍是 `role_matrices`、`sort_matrices`，每个 Level 的矩阵仍是 `(边数, 节点数)`，使用整数 dtype。role 的 -1/0/+1 及 sort 的参数顺序含义不变。

构造 Topo 时统一转为 coalesced COO 并移除显式零项。稠密和 COO 输入都进入同一实现，没有稀疏开关或第二套 Topo。重复坐标按 PyTorch 的求和语义合并，再检查 role 是否合法；它不能用来表示同一节点的多个参数位置。需要 `f(x, x)` 时，通过恒等 Operation 创建另一个节点，保留原生 autograd 依赖。

```python
assert topo.role_matrices[0].layout == torch.sparse_coo
assert topo.role_matrices[0].is_coalesced()
assert topo.get_topo(0, 0, 0, matrix_type="role") == -1
```

编译、查询、哈希/相等判断、迁移、合并、裁剪和可视化只读取存储项，不恢复完整稠密矩阵。`sort_nodes_by_topo` 为保持接口仍返回指定行的全部节点及其排序值；`topo.to_list()` 是显式完整导出，允许分配稠密内存。

从稠密输入转换不能避免输入创建时的内存；超大拓扑可以直接向原字段传入 COO。暂不接受 CSR 等其他布局。已有直接对字段调用稠密专用操作（例如 `pad`、`flatten().tolist()`）的代码需要改用稀疏操作或显式 `to_list()` 导出。

## 图合并用于重新组装

`MHD_Graph.merge_graph(graphs, device=...)` 保留原接口：

- 同名节点 aggregation、memory 必须兼容。
- Feature/Gradient 的 initial/current 四份状态逐项比较 shape、dtype、device、requires_grad 和数值；数值使用 `torch.equal`，不使用容差或隐式类型转换。
- 冲突报错并指出节点与具体字段，不再默认取均值，也不新增融合策略参数。
- 相同状态通过独立的 `detach().clone()` 保留数值和 requires_grad 设置，不携带旧 autograd 依赖。
- 合并图没有旧 Forward trace，必须重新前向后才能反向。模块参数共享、同名边和拓扑冲突规则沿用既有行为。

NaN 状态按 `torch.equal` 判定为不相等。图合并不延续源节点的运行中计算轨迹。

## 使用与迁移

Framework 从 `V5.MHD_Framework_V5` 导入，Utils 从 `V5.MHD_Utils_V5` 导入。已有 V4 代码保留原导入。

- V4 显式 `sum/avg/max/min/mul`：V5 增加 `memory=True` 可保留旧态参与方式。
- V4 单消息 `replace`：使用默认 `sum, memory=False`。
- V4 多消息 `replace`：显式选择聚合，不再隐式丢弃前面的 incoming。
- callable 仍为 `fn(current, incomings)`；关闭 memory 时需要处理 `current=None`。
- 从早期 V5 升级：非零梯度初态不再注入；合并不再平均；Topo 字段统一为 COO；部分反向的起点来自所选依赖而非完整 Forward。
- 内置聚合的广播、dtype、并列极值梯度及空 incoming 行为不变；不开启 incoming 的自动 detach。
- 状态文件仍只保存数值状态，加载时应使用相同 aggregation、memory 重建目标图。不添加权重迁移工具或改变 V4 checkpoint 格式。
- Trainer 的 `criteria` 仍由具体任务定义，用于验证和最佳 checkpoint 选择，与所选训练标量目标分开。

## 验证

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests/test_node_memory_v5.py tests/test_unified_v5.py
CUDA_VISIBLE_DEVICES=0 MHD_TEST_CUDA=1 OMP_NUM_THREADS=2 python -m pytest -q tests/test_node_memory_v5.py tests/test_unified_v5.py
```

GPU 检查显式启用；默认运行 CPU 测试。设备安排仅属于测试命令，不进入框架配置。新测试复用历史 ResNet、Transformer、循环消息传递的原生参考模型，并将其构图绑定到 V5，V4 测试文件不变。

## 本轮验证：统一反向、稀疏存储与严格合并（2026-09-08–09，Asia/Riyadh）

环境：ws02，PyTorch 2.8.0+cu128。

- CPU 与显式 GPU0 专项测试：**91 passed, 2 skipped**。跳过的是 CPU FP16 Trainer 两个用例；GPU0 的 FP16、BF16、FP32 全局/局部目标与梯度累积均通过原生更新对照。
- CPU/GPU0 上的 ResNet、Transformer、循环消息传递模型等价检查通过，README Python 示例通过。
- GPU0/1 独立 DDP、FSDP2、TP、PP（GPipe）smoke 通过；每次启动前检查 GPU1 无计算进程。使用显式 localhost rendezvous 和 NCCL loopback、禁用 IB/P2P，不代表默认互联配置或性能测试。
- V1–V4 与原有实验目录相对 e0954e7 无改动；Node/Operation 语义保持不变。本轮未重跑完整历史 V4 测试套件，也未启动完整训练实验。
- 稀疏验证包含禁止普通操作调用 to_dense，以及仅含一个有效非零项的 10^8 × 10^8 逻辑形状；这验证存储行为，不是吞吐性能基准。

## English

V5 keeps the four core classes and independent Node memory. Backward levels now select a real forward dependency whose unique differentiable terminal must contain one element. Full-loss and intermediate-scalar backward use the same interface and one native autograd call. Nonzero Gradient Initial States are rejected; current Gradient Messages always refer to current Feature tensors.

Topo retains two-dimensional Role/Sort fields, canonicalized to sparse COO without a separate mode. Graph merging copies equal numerical states into fresh, detached state tensors and rejects conflicts instead of averaging. Modules retain the existing sharing rules. V4 stays frozen.

## 历史验证：e0954e7 的 memory 扩展（2026-09-08）

以下仅描述升级前的 e0954e7，不是本轮实现的验证结果。

环境：ws02，PyTorch 2.8.0+cu128。V4 文件与基线提交 8f6651ce9af96f54585bf32bbfd70afe0315867d 完全一致。

- V5 专项测试：38 passed，包括两种 memory 设置的原生 optimizer 更新对照。
- GPU 0：两种 memory 的 graph.forward/backward 检查通过；按上述迁移规则适配的既有 ResNet、Transformer、循环消息传递 smoke 均与原生参考一致。
- GPU 0/1：独立 DDP、FSDP2、TP、PP smoke 均通过。每次启动前检查 GPU 1 空闲，使用显式 127.0.0.1 rendezvous、NCCL_SOCKET_IFNAME=lo、NCCL_IB_DISABLE=1、NCCL_P2P_DISABLE=1；不代表默认互联配置或性能验证。初次 standalone 启动超时，相关测试进程已清理。
- 全仓库 pytest 仍有 5 个来自原 V4 的失败：generate_mermaid 缺失，以及 4 个使用旧 criteria_node 接口的测试。V4 源码与这些测试未改动；此次不修复它们。
- 静态对照：Framework 仅 Node 开关校验、聚合函数及 merge_graph 保留/检查开关发生函数级变化；其余函数 AST 一致。Utils 除版本导入外内容一致。
