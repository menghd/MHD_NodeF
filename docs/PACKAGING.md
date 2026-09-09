# Installing the framework

MHD_Project is the authoritative framework repository. It is independently installable; applications must not embed another copy of its Python modules in their own distributions.

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_tensor_vjp_v5.py tests/test_unified_v5.py tests/test_utils_v4.py
```

The distribution version `0.5.0.dev0` identifies this development packaging baseline, not a published stable or PyPI release. Pin the Git commit for reproducibility. Both `V4` and `V5` are installed; choose the API explicitly in imports. Installation does not migrate a graph from V4 to V5. V1–V3 remain in source history/directories for reproduction and are not part of the current installable distribution.

For a fixed application checkout:

```bash
git submodule update --init --recursive
python -m pip install --no-deps -e third_party/MHD_Project
python -m pip install --no-deps -e .
```

Applications lock the upstream commit and tested source digests. Framework changes require its own tests; application-specific adapters and experiments remain in the application repositories. Do not import application packages from framework code. CPU tests do not establish A100 or multi-GPU acceptance.

The packaging migration also refreshes five stale V4 test calls to the existing
`criteria` callable and `display_graph` APIs. Framework implementation bytes are
unchanged. Do not revive the retired criteria_node or graph.generate_mermaid APIs
solely to satisfy old tests. CPU tests do not establish distributed/GPU acceptance.
