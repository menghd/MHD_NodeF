# MHD V5 — Memory、统一反向与稀疏拓扑

V5 保留 Node、Edge、Topo、Graph 四个核心类型，在独立 Node memory 的基础上，统一按所选 Level 推断张量反向起点、使用稀疏拓扑存储，并检查图合并的状态冲突。V4 源码和实验保持原样。

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

反向起点由**所选依赖范围**决定，最终 loss 与中间输出采用同一规则：必须有唯一可微终点，支持标量和张量。多个终点、错误顺序和未匹配 trace 均报错；已 detach 的指标不参与推断。

反向计算的是向量—雅可比积（VJP，实数情况下为 `Jᵀv`），不是完整 Jacobian；复数梯度沿用 PyTorch 的原生约定。`v` 是所选终点的传入梯度，沿用 PyTorch 的默认规则：

| 所选终点 | 未配置反向输入 | 显式配置反向输入 |
|---|---|---|
| 实数单元素 Tensor（包含 `()`、`(1,)` 等） | 自动使用全 1 | 使用给定的 `v`，包括全零 |
| 多元素或复数 Tensor | 报错，不自动求和/平均 | 使用给定的 `v` |

通过 `node.gradient_message.update_initial(v)` 或构造 Node 时传入 Gradient Message 配置输入；不新增 seed、backward_node 或局部求导接口。只有**被选为起点的状态所属节点**提供这次输入，其他节点的 initial Gradient 不作为额外起点注入。显式输入 shape、device 和实数/复数类别必须匹配所选历史状态；浮点精度转换沿用原生 autograd。

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

`backward(levels=[2])` 从 loss 反向经过 square；`backward(levels=[3])` 从 h 反向经过 double。这个实数标量示例省略了输入，因此原生 autograd 收到 1；Trainer 的 AMP/梯度累积对同一终点的传入梯度应用相应缩放。每次仍只调用一次原生 autograd，不重新执行 Operation。未选边的真实前向输出通过梯度 hook 屏蔽。

内部记录聚合后的状态版本和 memory 依赖，避免节点覆盖后误用最新值作为历史起点。Gradient Message **始终对应当前 Feature Message**；旧版本可以参与求导，但旧版本梯度不写入该节点的 current Gradient，也不跨版本相加。模块参数依旧通过标准 `.grad` 访问，保留原有优化器管理行为。多次 backward 的参数梯度会按原有规则累积，开始独立优化目标前应调用原生 optimizer 的 `zero_grad`；当前输入叶 Tensor 的梯度仍按原有 MHD 行为每次清理。

显式配置会持续生效，直到再次更新或重建节点；reset 和设备迁移不把显式零变回默认输入。框架自动创建的零初态表示尚未配置，显式传入的零表示零 VJP，二者不能只按数值区分。反向完成后，没有收到梯度的当前 Feature 对应全零 Gradient Current State，不会把配置中的 `v` 当成计算结果。错误检查发生在 Graph 修改 Gradient Current State、注册 hook 或修改 `.grad` 之前。

下面复用上面的拓扑，从张量 h 反向，后续 loss 节点仍可存在：

```python
x_node = graph.get_node_by_name("x")
h_node = graph.get_node_by_name("h")
x_node.feature_message.current_state = torch.tensor([2., 3.], requires_grad=True)
h_node.gradient_message.update_initial(torch.tensor([1., -1.]))
graph.forward(levels=[0, 1])
graph.backward(levels=[3])
torch.testing.assert_close(x_node.gradient_message.current_state, torch.tensor([2., -2.]))
```

## Graph 的 autograd 缓存生命周期

接口保持 `graph.forward(levels=[...])` 与 `graph.backward(levels=[...])`。`retain_graph` 移到 Graph 的 keyword-only bool 设置，默认 `False`；也可以通过 `graph.retain_graph` 修改。它只控制原生反向后是否释放求导缓存，与 Node memory 无关。

```python
graph.retain_graph = True  # 也可在 MHD_Graph(..., retain_graph=True) 构造时设置
x_node.feature_message.current_state = torch.tensor([2., 3.], requires_grad=True)
graph.forward(levels=[0, 1])
h_node.gradient_message.update_initial(torch.tensor([1., 0.]))
graph.backward(levels=[3])
torch.testing.assert_close(x_node.gradient_message.current_state, torch.tensor([2., 0.]))

h_node.gradient_message.update_initial(torch.tensor([0., 1.]))
graph.retain_graph = False  # 最后一次反向释放本次求导缓存
graph.backward(levels=[3])
torch.testing.assert_close(x_node.gradient_message.current_state, torch.tensor([0., 2.]))
```

仅将属性改为 False 不会立即释放缓存；它在下次 backward 时生效。默认 False 下再次 backward 必须重新 forward。参数更新后应重新 forward，不复用更新前的依赖。此设置不打开高阶求导，也不复制激活或重跑 Operation。

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

- 同名节点 aggregation、memory 和反向输入的显式/默认标记必须兼容；隐式零与显式零存在语义冲突。
- Feature/Gradient 的 initial/current 四份状态逐项比较 shape、dtype、device、requires_grad 和数值；数值使用 `torch.equal`，不使用容差或隐式类型转换。
- 冲突报错并指出节点与具体字段，不再默认取均值，也不新增融合策略参数。
- 相同状态通过独立的 `detach().clone()` 保留数值和 requires_grad 设置，不携带旧 autograd 依赖。
- 合并图没有旧 Forward trace，retain_graph 使用默认 False，必须重新前向后才能反向。模块参数共享、同名边和拓扑冲突规则沿用既有行为。

NaN 状态按 `torch.equal` 判定为不相等。图合并不延续源节点的运行中计算轨迹。

## 使用与迁移

Framework 从 `V5.MHD_Framework_V5` 导入，Utils 从 `V5.MHD_Utils_V5` 导入。已有 V4 代码保留原导入。

- V4 显式 `sum/avg/max/min/mul`：V5 增加 `memory=True` 可保留旧态参与方式。
- V4 单消息 `replace`：使用默认 `sum, memory=False`。
- V4 多消息 `replace`：显式选择聚合，不再隐式丢弃前面的 incoming。
- callable 仍为 `fn(current, incomings)`；关闭 memory 时需要处理 `current=None`。
- 从早期 V5 升级：梯度初态只作为所选终点的反向输入，不再多点注入；合并不再平均；Topo 字段统一为 COO；部分反向的起点来自所选依赖而非完整 Forward。
- 从 631895d 升级：取消张量终点限制和非零初态禁令；原 backward 的 retain_graph 关键字改为 Graph 构造参数或属性。
- 内置聚合的广播、dtype、并列极值梯度及空 incoming 行为不变；不开启 incoming 的自动 detach。
- updown_node 与 Trainer checkpoint 保存数值状态和 Gradient Message 的 initial_state_explicit 标记。旧文件缺少标记时，零初态按未配置读取，非零初态按显式输入读取；旧格式无法恢复显式零与隐式零的区别。四份运行状态分别恢复其 shape/dtype，支持初态占位形状与当前 batch 不同、以及 AMP 当前梯度与初态精度不同；加载时使用相同 aggregation、memory 重建目标图。不修改 V4 checkpoint 格式。
- Trainer 的 `criteria` 仍由任务定义，用于验证和最佳 checkpoint 选择。Tensor 输出的均值日志只是指标，不一定等于本次 VJP 对应的目标。AMP 缩放传入梯度，梯度累积和 optimizer 保持原生管理。
- PP 沿用原生流水线的 loss/backward 调度，本轮不扩展为可复用 VJP：retain_graph=True 或输出节点的显式 Gradient 输入会明确报错，包含准备模型后修改设置的情况。PP 继续由已有 pipeline_loss_fn 定义标量目标；Graph 的张量 VJP 适用于普通执行及共享该反向路径的并行模式。

## 验证

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests/test_node_memory_v5.py tests/test_unified_v5.py tests/test_tensor_vjp_v5.py
CUDA_VISIBLE_DEVICES=0 MHD_TEST_CUDA=1 OMP_NUM_THREADS=2 python -m pytest -q tests/test_node_memory_v5.py tests/test_unified_v5.py tests/test_tensor_vjp_v5.py
```

GPU 检查显式启用；默认运行 CPU 测试。设备安排仅属于测试命令，不进入框架配置。新测试复用历史 ResNet、Transformer、循环消息传递的原生参考模型，并将其构图绑定到 V5，V4 测试文件不变。

## 本轮验证：张量 VJP 与 Graph 缓存设置（2026-09-09，Asia/Riyadh）

环境：ws02，PyTorch 2.8.0+cu128。基于 631895d 实现，并对齐新增共享工作区规范的 main f3f5228。

- CPU 与显式 GPU0 专项测试：**149 passed, 3 skipped**。跳过的是 CPU FP16 Trainer 用例；GPU0 的 FP32/BF16/FP16 均通过原生梯度和参数更新对照。
- 覆盖局部/最终张量终点、部分路径、显式零、复数、历史状态、错误不污染已有梯度、同一次前向的多次 VJP，以及 memory、稀疏拓扑、严格合并和模型回归。
- updown_node 与 Trainer 的新旧 checkpoint 通过；包含初态/当前态 shape 或 dtype 不同，以及 BF16/FP16 状态加载到新图后继续训练的参数更新对照。
- GPU0/1 独立 DDP、FSDP2、TP、PP（GPipe）既有标量目标 smoke 通过；每次启动前确认 GPU1 无计算进程。使用 localhost rendezvous、NCCL loopback，禁用 IB/P2P；这是兼容检查，不是吞吐或压力测试，也不代表 PP 已支持显式张量 VJP。
- V1–V4 与既有实验未修改。Edge、Topo、Node 聚合函数和 Graph.forward 的 AST 与升级前一致。未重跑有已知历史问题的完整 V4 测试集，未启动研究训练。
- README 中的 Python 示例全部实际执行通过。checkpoint 测试的单进程 DCP 提示属于预期警告。

## 历史验证：631895d 的统一反向、稀疏存储与严格合并（2026-09-08–09，Asia/Riyadh）

环境：ws02，PyTorch 2.8.0+cu128。

- CPU 与显式 GPU0 专项测试：**91 passed, 2 skipped**。跳过的是 CPU FP16 Trainer 两个用例；GPU0 的 FP16、BF16、FP32 全局/局部目标与梯度累积均通过原生更新对照。
- CPU/GPU0 上的 ResNet、Transformer、循环消息传递模型等价检查通过，README Python 示例通过。
- GPU0/1 独立 DDP、FSDP2、TP、PP（GPipe）smoke 通过；每次启动前检查 GPU1 无计算进程。使用显式 localhost rendezvous 和 NCCL loopback、禁用 IB/P2P，不代表默认互联配置或性能测试。
- V1–V4 与原有实验目录相对 e0954e7 无改动；Node/Operation 语义保持不变。本轮未重跑完整历史 V4 测试套件，也未启动完整训练实验。
- 稀疏验证包含禁止普通操作调用 to_dense，以及仅含一个有效非零项的 10^8 × 10^8 逻辑形状；这验证存储行为，不是吞吐性能基准。

## English

V5 keeps the four core classes and independent Node memory. Backward levels select a real forward dependency with a unique differentiable terminal. Scalars and tensors use one native VJP: an omitted input defaults to one only for real single-element outputs; other outputs require an explicit Gradient Initial State. Explicit zero remains meaningful across reset, device moves, merging and checkpoints. Only the selected root supplies the cotangent; current Gradient Messages refer to current Feature tensors. retain_graph is now a Graph setting (default False), so forward and backward both take only levels. Native PP retains its existing loss schedule and rejects retained graphs or explicit output cotangents.

Topo retains two-dimensional Role/Sort fields, canonicalized to sparse COO without a separate mode. Graph merging copies equal numerical states into fresh, detached state tensors and rejects conflicts instead of averaging. Modules retain the existing sharing rules. V4 stays frozen.

## 历史验证：e0954e7 的 memory 扩展（2026-09-08）

以下仅描述升级前的 e0954e7，不是本轮实现的验证结果。

环境：ws02，PyTorch 2.8.0+cu128。V4 文件与基线提交 8f6651ce9af96f54585bf32bbfd70afe0315867d 完全一致。

- V5 专项测试：38 passed，包括两种 memory 设置的原生 optimizer 更新对照。
- GPU 0：两种 memory 的 graph.forward/backward 检查通过；按上述迁移规则适配的既有 ResNet、Transformer、循环消息传递 smoke 均与原生参考一致。
- GPU 0/1：独立 DDP、FSDP2、TP、PP smoke 均通过。每次启动前检查 GPU 1 空闲，使用显式 127.0.0.1 rendezvous、NCCL_SOCKET_IFNAME=lo、NCCL_IB_DISABLE=1、NCCL_P2P_DISABLE=1；不代表默认互联配置或性能验证。初次 standalone 启动超时，相关测试进程已清理。
- 全仓库 pytest 仍有 5 个来自原 V4 的失败：generate_mermaid 缺失，以及 4 个使用旧 criteria_node 接口的测试。V4 源码与这些测试未改动；此次不修复它们。
- 静态对照：Framework 仅 Node 开关校验、聚合函数及 merge_graph 保留/检查开关发生函数级变化；其余函数 AST 一致。Utils 除版本导入外内容一致。
