# MHD Project

[English](README.md)

MHD Project 是一个基于 PyTorch、从超图视角表示神经计算的研究框架。它把计算、状态和拓扑显式分离，同时不把网络限制在某一种模型类型中。

当前源码版本是 **V5**。V1–V4 完整保留用于复现。V5 只将 Node 的 memory 与 aggregation 解耦，见 [V5/README.md](V5/README.md)。

## 核心思想

MHD 用四个结构概念描述网络：

- **Node**：承载参与计算的状态或消息；
- **Edge**：保存作用于一个或多个 Node 的计算；
- **Topo**：通过 Role Matrix 与 Sort Matrix 表示 Node–Edge 角色、连接和位置顺序；
- **Graph**：组合上述对象，并按照用户选择的 topology levels 执行。

不同版本的实现不断演化，但共同结构始终清楚：

```text
Node 状态/消息 → Edge 计算/操作 → Node 状态/消息
                              │
                              └── 由显式 topology levels 控制
```

MHD 不重新实现 PyTorch 的 Tensor kernel、`nn.Module`、optimizer 或 autograd；它在原生 PyTorch 计算外提供显式的超图表达与执行结构。

## 版本演化

| 版本 | 项目定位 | 主要发展 |
|---|---|---|
| [V1](V1/README.md) | 历史原型 | `DNet`、`HDNet`、`MHDNet`，探索 Node 与 Hyperedge 组织方式 |
| [V2](V2/README.md) | 第一版正式超图框架 | `MHD_Node`、`MHD_Edge`、`MHD_Topo`、`MHD_Graph`，Role/Sort Matrix |
| [V3](V3/README.md) | 多层动态超图框架 | Initial/Current State、多 level 执行、图工具与基础分布式支持 |
| [V4](V4/README.md) | 冻结保留版本 | Feature/Gradient Message、Operation 包装、显式前后向 level 路径、原生 autograd 集成 |
| [V5](V5/README.md) | 当前源码版本 | 独立 Node memory，默认 sum、取消 replace，其余沿用 V4 |

每个版本目录只保留 Framework、Utils 和该版本 README。V4 的兼容脚本是正式的 V3→V4 接入入口，因此与 V4 放在一起。示例、实验、测试与 benchmark 全部放在版本目录之外。

## 仓库结构

```text
MHD_Project/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── CONTRIBUTING.md
├── V1/
│   ├── MHD_Framework_V1.py
│   ├── MHD_Utils_V1.py
│   └── README.md
├── V2/
│   ├── MHD_Framework_V2.py
│   ├── MHD_Utils_V2.py
│   └── README.md
├── V3/
│   ├── MHD_Framework_V3.py
│   ├── MHD_Utils_V3.py
│   └── README.md
├── V4/
│   ├── MHD_Framework_V4.py
│   ├── MHD_Utils_V4.py
│   ├── MHD_Compatibility_V3_to_V4.py
│   └── README.md
├── V5/
│   ├── MHD_Framework_V5.py
│   ├── MHD_Utils_V5.py
│   └── README.md
├── docs/
├── examples/
├── experiments/
├── benchmarks/
└── tests/
```

## 源码方式使用

本仓库当前先作为源码仓库维护，**暂不提供 `pip install`**。克隆后在仓库根目录运行 Python：

```bash
git clone https://github.com/souray0410/MHD_Project.git
cd MHD_Project
```

V4 需要兼容的 PyTorch 环境。代码的目标环境是 PyTorch 2.13；当前文档记录的 CPU/GPU 实测环境为 PyTorch 2.8.0。训练与具体实验需要的额外依赖，以相应版本或实验目录的 README 为准。

## V4 最小示例

```python
import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo

nodes = {
    MHD_Node(0, "input", MHD_Node.Message(torch.zeros(4, 1))),
    MHD_Node(1, "prediction", MHD_Node.Message(torch.zeros(4, 1))),
    MHD_Node(2, "loss", MHD_Node.Message(torch.zeros(()))),
}

edges = {
    MHD_Edge(0, "linear", [MHD_Edge.Operation(nn.Linear(1, 1))]),
    MHD_Edge(
        1,
        "criterion",
        [MHD_Edge.Operation(lambda value: value.square().mean())],
    ),
}

forward_linear = torch.tensor([[-1, 1, 0], [0, 0, 0]])
forward_loss = torch.tensor([[0, 0, 0], [0, -1, 1]])
linear_sort = torch.tensor([[0, 1, 0], [0, 0, 0]])
loss_sort = torch.tensor([[0, 0, 0], [0, 0, 1]])

topo = MHD_Topo(
    role_matrices=[forward_linear, forward_loss, -forward_loss, -forward_linear],
    sort_matrices=[linear_sort, loss_sort, loss_sort, linear_sort],
)
graph = MHD_Graph(nodes, edges, {topo}, device="cpu")

value = torch.randn(4, 1, requires_grad=True)
graph.get_node_by_name("input").feature_message.current_state = value
graph.forward(levels=[0, 1])
graph.backward(levels=[2, 3])

print(graph.get_node_by_name("input").gradient_message.current_state)
```

新项目请先阅读 [V4/README.md](V4/README.md)。V4 的完整设计、V3→V4 兼容说明、验证范围和性能边界保存在[详细 V4 中文指南](docs/V4_GUIDE.zh-CN.md)中。

## 历史代码与实验

- V1 原始 pipeline 和辅助代码完整保存在 [`experiments/V1/legacy`](experiments/V1/legacy)；
- V2 原始示例位于 [`examples/V2`](examples/V2)；
- RETFound 2D 分阶段实验位于 [`experiments/V4/retfound_2d`](experiments/V4/retfound_2d)；
- 每个版本的行为以该版本代码和 README 为准，不把新版本接口静默回填到旧版本。

## 当前状态

V5 是当前维护的源码版本，V4 冻结保留；V1–V4 用于复现和理解框架演化。V4 文档中的真实多卡验证在双卡环境完成；代码没有主动锁死两卡，但三卡及以上的真实性能与稳定性尚未在对应硬件环境验证。

## 参与维护

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。涉及 Framework 语义的修改应同时说明兼容性，并更新相应文档与测试。

## 许可证

MHD Project 使用 [MIT License](LICENSE)。

## 作者

孟号丁（Haoding Souray Meng）— [souray0410](https://github.com/souray0410)
