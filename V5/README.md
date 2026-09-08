# MHD V5 — Node Memory

V5 基于 V4，仅将旧 Feature 状态是否参与聚合独立为 Node 的 `memory` 开关。V4 源码和实验保持原样。

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

## 保持 V4 的其余行为

- 四个顶层类型、Message 的四个状态、Operation、Role/Sort levels、真实 trace 和单次原生 backward 不变。
- 内置算子的广播、dtype 和并列极值梯度行为沿用 V4；不引入 `mean/prod/amax/amin` 重命名或严格同形状限制。
- `merge_graph` 仍对四个状态分别取均值，保留 Node 的 memory；同名 Node 的 aggregation 或 memory 冲突时报错。没有新增状态融合策略。
- Utils 仅复制到 V5 并切换版本引用。状态文件仍是数值状态而非完整建图配置；加载时应使用相同的 aggregation、memory 建立目标图，正如 V4 需要重建 aggregation。
- 默认关闭 memory 不会 detach incoming 自带的计算图。

## 使用与迁移

Framework 从 `V5.MHD_Framework_V5` 导入，Utils 从 `V5.MHD_Utils_V5` 导入。已有 V4 代码保持其原有导入。

- V4 显式 `sum/avg/max/min/mul`：在 V5 增加 `memory=True` 可保留旧态参与方式。
- V4 单消息 `replace`：改用任一内置聚合并设置 `memory=False`；默认 sum 即可。
- V4 多消息 `replace`：V5 不再隐式丢弃前面的 incoming，应显式选择所需聚合或在自定义 Operation/callable 中表达所需选择。
- 自定义 `fn(current, incomings)`：`memory=True` 保留原调用语义；关闭时需处理 `current=None`。
- 不添加权重迁移工具，不修改 V4 checkpoint 格式。

## 验证

```bash
python -m pytest -q tests/test_node_memory_v5.py
```

测试覆盖五种聚合的前向值/原生梯度等价、默认行为、空/单消息、真实 graph.backward、callable、图合并及 Utils 状态读写与裁剪。GPU 可见设备由运行命令指定，框架不硬编码设备。

## English

V5 adds one independent, keyword-only Node option: `memory=False`. It aggregates incoming Feature messages without the previous current state; `memory=True` includes that state as one extra operand. The default aggregation is `sum`, and `replace` is removed. The remaining names stay `sum/avg/max/min/mul`.

Custom aggregation keeps the two-argument signature `fn(current, incomings)`, with `current=None` when memory is disabled. Empty incoming messages leave the state unchanged. Native autograd, topology, state-merge averaging, and the other V4 behavior are retained. Merge preserves memory and rejects conflicting memory settings on same-name Nodes. V4 remains unchanged.

## 本次验证记录（2026-09-08）

环境：ws02，PyTorch 2.8.0+cu128。V4 文件与基线提交 8f6651ce9af96f54585bf32bbfd70afe0315867d 完全一致。

- V5 专项测试：38 passed，包括两种 memory 设置的原生 optimizer 更新对照。
- GPU 0：两种 memory 的 graph.forward/backward 检查通过；按上述迁移规则适配的既有 ResNet、Transformer、循环消息传递 smoke 均与原生参考一致。
- GPU 0/1：独立 DDP、FSDP2、TP、PP smoke 均通过。每次启动前检查 GPU 1 空闲，使用显式 127.0.0.1 rendezvous、NCCL_SOCKET_IFNAME=lo、NCCL_IB_DISABLE=1、NCCL_P2P_DISABLE=1；不代表默认互联配置或性能验证。初次 standalone 启动超时，相关测试进程已清理。
- 全仓库 pytest 仍有 5 个来自原 V4 的失败：generate_mermaid 缺失，以及 4 个使用旧 criteria_node 接口的测试。V4 源码与这些测试未改动；此次不修复它们。
- 静态对照：Framework 仅 Node 开关校验、聚合函数及 merge_graph 保留/检查开关发生函数级变化；其余函数 AST 一致。Utils 除版本导入外内容一致。
