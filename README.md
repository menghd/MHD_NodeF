# Workspace preparation: 2026_09_09_10_30_34

[三项目统一目录与Ibex准备约定](workspace/README.zh-CN.md)。This adds a shared naming/path registry only; no scientific code, running release, framework API, data split or GPU workload is changed.

# MHD Project

[中文说明](README.zh-CN.md)

MHD Project is a PyTorch-based research framework for representing neural computation from a hypergraph perspective. It keeps computation, state, and topology explicit without restricting the network to a particular model family.

The current source version is **V5**. V1–V4 are preserved for reproducibility. V5 combines independent Node memory, selected-scalar backward, unified sparse topology, and strict state merging; see [V5/README.md](V5/README.md).

## Core idea

MHD describes a network through four structural concepts:

- **Node** carries the state or message being computed.
- **Edge** contains the computation applied to one or more nodes.
- **Topo** records node–edge roles and positional order with Role and Sort matrices.
- **Graph** combines the components and executes selected topology levels.

The implementation evolved while this structure remained recognizable:

```text
Node state/message → Edge computation/operation → Node state/message
                              │
                              └── controlled by explicit topology levels
```

MHD does not replace PyTorch tensor kernels, modules, optimizers, or autograd. It supplies an explicit hypergraph representation around ordinary PyTorch computation.

## Versions

| Version | Position in the project | Main development |
|---|---|---|
| [V1](V1/README.md) | Historical prototype | `DNet`, `HDNet`, and `MHDNet`; node- and hyperedge-oriented experiments |
| [V2](V2/README.md) | First formal hypergraph framework | `MHD_Node`, `MHD_Edge`, `MHD_Topo`, `MHD_Graph`; Role/Sort matrices |
| [V3](V3/README.md) | Multi-level dynamic framework | Initial/current node state, multi-level execution, graph utilities, basic distributed support |
| [V4](V4/README.md) | Preserved source version | Feature/Gradient Messages, wrapped Operations, explicit forward/backward level paths, native PyTorch autograd integration |
| [V5](V5/README.md) | Current source version | Independent memory; selected-scalar backward; sparse topology; strict state merging |

Each version directory contains only its framework, utilities, and version documentation. The V4 compatibility entry is retained beside V4 because it is the official V3-to-V4 migration path. Examples, experiments, tests, and benchmarks live outside the version directories.

## Repository layout

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

## Source-tree usage

This repository is currently maintained as source code and is **not packaged for `pip install`**. Clone it and run Python from the repository root:

```bash
git clone https://github.com/souray0410/MHD_Project.git
cd MHD_Project
```

V4 requires a compatible PyTorch environment. The code targets PyTorch 2.13; the documented CPU and GPU validation currently uses PyTorch 2.8.0. Optional training and experiment utilities may require additional dependencies described in their own README files.

## V4 quick start

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

Start new work with [V5/README.md](V5/README.md). The V4 quick start above remains a historical example. The complete V4 design, compatibility notes, validation scope, and performance boundary are recorded in the [detailed V4 guide](docs/V4_GUIDE.zh-CN.md).

## Historical code and experiments

- Original V1 pipelines and auxiliary files are preserved under [`experiments/V1/legacy`](experiments/V1/legacy).
- The original V2 example is under [`examples/V2`](examples/V2).
- The staged RETFound 2D experiment is under [`experiments/V4/retfound_2d`](experiments/V4/retfound_2d).
- Version-specific behavior should be judged against the code and README of that version; later APIs are not silently backported.

## Project status

V5 is the active source version; V4 is frozen. V1–V4 remain available for reproducibility and for understanding the evolution of the abstraction. Real multi-GPU validation documented for V4 was performed on two GPUs; code paths are not intentionally limited to two devices, but three-or-more-device performance and stability have not yet been verified on real hardware.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md). Changes to framework semantics should include a clear compatibility statement and corresponding documentation and tests.

## License

MHD Project is released under the [MIT License](LICENSE).

## Author

Haoding Souray Meng (孟号丁) — [souray0410](https://github.com/souray0410)
