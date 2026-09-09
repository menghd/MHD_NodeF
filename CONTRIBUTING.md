# Contributing / 参与维护

MHD Project preserves several historical versions while actively developing V5. Contributions should keep the distinction between historical reproducibility and current development explicit.

MHD Project 同时保留多个历史版本并继续维护 V5。任何修改都应明确区分“历史复现”和“当前开发”。

## Repository rules / 仓库规则

- V4 is frozen. Develop changes in V5; do not modify V4 source or experiments.
- Do not silently rewrite V1–V3 behavior. Fixes to a historical version must be documented in that version's README.
- 不要静默改变 V1–V3 的行为；如确需修正，必须在对应版本 README 中说明。
- Keep `MHD_Framework_Vx.py` and `MHD_Utils_Vx.py` in the matching version directory. Put examples, experiments, tests, and benchmarks in their top-level directories.
- Framework 与 Utils 保存在对应版本目录；示例、实验、测试和 benchmark 放在仓库顶层的专门目录。
- Preserve the public names `MHD_Node`, `MHD_Edge`, `MHD_Topo`, and `MHD_Graph` unless a versioned design explicitly states otherwise.
- 除非新版本设计明确说明，否则保持四个核心名称不变。
- A semantic change must update code, tests, the version README, and the root version table together.
- 语义变更必须同时更新代码、测试、版本 README 和根目录版本表。
- Validation claims must identify the tested environment and must not generalize beyond actual results.
- 验证结论必须注明真实环境，不得把未运行的范围写成已经通过。

## Proposed changes / 提交修改

Please describe:

1. the problem and intended behavior;
2. the affected version and public API;
3. backward-compatibility impact;
4. tests or experiments performed;
5. documentation changes.

请说明：问题与目标行为、影响版本和公开 API、兼容性影响、实际测试或实验，以及对应文档修改。

## Style / 风格

Prefer a small number of explicit concepts and ordinary PyTorch data structures. Public configuration should remain minimal; optional acceleration or distributed details belong in Utils unless they change the hypergraph model itself.

优先使用少量、显式的概念和普通 PyTorch 数据结构。公开配置应保持精简；可选加速或分布式细节应放在 Utils，除非它们确实改变超图模型本身。

## Package and release contract

See [packaging](docs/PACKAGING.md). Install V4/V5 from this repository as one distribution; research applications consume a pinned commit. V5 is active development; V4 source remains frozen. Shared repository principles (source isolation, tests, version pins, traceability) apply, but a framework's examples/benchmarks/versioned APIs need not imitate a research experiment tree. A release must document API changes, supported Python/PyTorch versions, CPU/GPU validation actually run, and migration limits. The development package is not automatically published to PyPI.
