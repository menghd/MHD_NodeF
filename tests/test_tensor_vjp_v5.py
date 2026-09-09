"""Tensor cotangents and Graph-owned autograd lifetime, against native PyTorch."""
import os
import pytest
import torch

from V5.MHD_Framework_V5 import MHD_Node, MHD_Graph
from V5.MHD_Utils_V5 import (
    MHD_Trainer, MHD_Monitor, MHD_DistributedContext,
    MHD_ParallelConfig, prepare_mhd_model, updown_node,
)
from test_unified_v5 import make, set_input, CHAIN, double, square, Scale


@pytest.mark.parametrize("levels", [[2, 3], [2], [3]])
@pytest.mark.parametrize("memory", [False, True])
def test_tensor_vjp_matches_native_selected_path(levels, memory, monkeypatch):
    scale = Scale()
    graph = make([scale, square], CHAIN, memory=memory,
                 initial={i: torch.tensor([0.5, 1.0]) for i in range(3)})
    x = set_input(graph, torch.tensor([2., 3.], requires_grad=True))
    v = torch.tensor([1.5, -2.])
    root_id = 1 if levels == [3] else 2
    graph.get_node_by_id(root_id).gradient_message.update_initial(v)
    # Non-root initial states are configuration, never additional VJP inputs.
    graph.get_node_by_id(0).gradient_message.update_initial(torch.tensor([99., 99.]))
    graph.forward([0, 1])
    original = torch.autograd.backward
    calls = []
    def backward(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "backward", backward)
    graph.backward(levels)
    assert len(calls) == 1
    nx = x.detach().clone().requires_grad_()
    nw = scale.weight.detach().clone().requires_grad_()
    h = nw * nx + (torch.tensor([0.5, 1.]) if memory else 0)
    out = h if levels == [3] else h.square() + (torch.tensor([0.5, 1.]) if memory else 0)
    gx, gw, gh = torch.autograd.grad(out, (nx, nw, h), grad_outputs=v)
    torch.testing.assert_close(x.grad, torch.zeros_like(gx) if levels == [2] else gx)
    if levels == [2]:
        assert scale.weight.grad is None
    else:
        torch.testing.assert_close(scale.weight.grad, gw)
    torch.testing.assert_close(graph.get_node_by_id(1).gradient_message.current_state, gh)
    if levels == [3]:
        torch.testing.assert_close(graph.get_node_by_id(2).gradient_message.current_state, torch.zeros(2))


@pytest.mark.parametrize("method", ["constructor", "update", "same_tensor", "assign", "inplace"])
def test_explicit_zero_is_not_implicit_one(method):
    graph = make([double, square], CHAIN)
    root = graph.get_node_by_id(2)
    if method == "constructor":
        replacement = MHD_Node(2, "n2", MHD_Node.Message(torch.tensor(0.)),
                               MHD_Node.Message(torch.tensor(0.)))
        graph.nodes.remove(root)
        graph.nodes.add(replacement)
        graph = MHD_Graph(graph.nodes, graph.edges, {graph.topo}, device="cpu")
        root = replacement
    elif method == "update":
        root.gradient_message.update_initial(torch.tensor(0.))
    elif method == "same_tensor":
        root.gradient_message.update_initial(root.gradient_message.initial_state)
    elif method == "assign":
        root.gradient_message.initial_state = torch.tensor(0.)
    else:
        root.gradient_message.initial_state.zero_()
    root.reset()
    root.to_device("cpu")
    x = set_input(graph)
    graph.forward([0, 1]).backward([2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(0.))
    assert not root._gradient_initial_is_implicit()


@pytest.mark.parametrize("value", [torch.tensor(2.), torch.tensor([2.]), torch.tensor([[2.]])])
def test_implicit_real_single_element_shapes(value):
    graph = make([double, square], CHAIN)
    x = set_input(graph, value.requires_grad_())
    graph.forward([0, 1]).backward([2, 3])
    torch.testing.assert_close(x.grad, 8 * value.detach())


@pytest.mark.parametrize("value", [torch.tensor([1., 2.]), torch.tensor(1 + 2j)])
def test_missing_tensor_or_complex_input_is_atomic(value):
    graph = make([double, square], CHAIN)
    x = set_input(graph, value.requires_grad_())
    graph.forward([0, 1])
    x.grad = torch.ones_like(x) * 9
    previous = {node.id: node.gradient_message.current_state for node in graph.nodes}
    with pytest.raises(RuntimeError, match="显式 Gradient"):
        graph.backward([2, 3])
    torch.testing.assert_close(x.grad, torch.ones_like(x) * 9)
    assert all(node.gradient_message.current_state is previous[node.id] for node in graph.nodes)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_explicit_dtype_matches_native_vjp(dtype):
    graph = make([double, square], CHAIN)
    value = torch.tensor([1., 2.], dtype=dtype)
    if value.is_complex():
        value += 1j
    x = set_input(graph, value.requires_grad_())
    v = torch.tensor([1., -2.], dtype=dtype)
    if v.is_complex():
        v += 2j
    graph.get_node_by_id(2).gradient_message.update_initial(v)
    graph.forward([0, 1]).backward([2, 3])
    native = value.detach().clone().requires_grad_()
    expected, = torch.autograd.grad((2 * native).square(), native, v)
    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize("bad", [torch.ones(1), torch.tensor(1j), torch.tensor(1), torch.empty((), device="meta")])
def test_invalid_explicit_input_preserves_gradients(bad):
    scale = Scale()
    graph = make([scale, square], CHAIN)
    x = set_input(graph)
    graph.forward([0, 1])
    graph.get_node_by_id(2).gradient_message.update_initial(bad)
    previous = {n.id: n.gradient_message.current_state for n in graph.nodes}
    scale.weight.grad = torch.tensor(7.)
    x.grad = torch.tensor(9.)
    with pytest.raises(ValueError, match="gradient_message.initial_state"):
        graph.backward([2, 3])
    torch.testing.assert_close(scale.weight.grad, torch.tensor(7.))
    torch.testing.assert_close(x.grad, torch.tensor(9.))
    assert all(n.gradient_message.current_state is previous[n.id] for n in graph.nodes)


def test_historical_tensor_root_validates_its_own_shape_and_clears_current_gradient():
    levels = [[(0, [0], [1])], [(1, [0], [1])], [(0, [1], [0])]]
    graph = make([double, lambda x: x.sum()], levels, count=2)
    x = set_input(graph, torch.tensor([2., 3.], requires_grad=True))
    graph.forward([0, 1])
    graph.get_node_by_id(1).gradient_message.update_initial(torch.tensor([1., -1.]))
    graph.backward([2])
    torch.testing.assert_close(x.grad, torch.tensor([2., -2.]))
    assert graph.get_node_by_id(1).feature_message.current_state.shape == ()
    torch.testing.assert_close(graph.get_node_by_id(1).gradient_message.current_state, torch.tensor(0.))


def test_repeated_vjp_uses_graph_setting_and_native_parameter_accumulation():
    module = Scale()
    graph = make([module, square], CHAIN)
    x = set_input(graph, torch.tensor([2., 3.], requires_grad=True))
    graph.retain_graph = True
    graph.forward([0, 1])
    root = graph.get_node_by_id(1)
    root.gradient_message.update_initial(torch.tensor([1., 0.]))
    graph.backward([3])
    torch.testing.assert_close(x.grad, torch.tensor([2., 0.]))
    torch.testing.assert_close(module.weight.grad, torch.tensor(2.))
    root.gradient_message.update_initial(torch.tensor([0., -1.]))
    graph.retain_graph = False
    graph.backward([3])
    torch.testing.assert_close(x.grad, torch.tensor([0., -2.]))
    torch.testing.assert_close(module.weight.grad, torch.tensor(-1.))
    with pytest.raises(RuntimeError, match="先执行"):
        graph.backward([3])
    with pytest.raises(TypeError):
        graph.backward([3], retain_graph=True)


def test_graph_retain_bool_constructor_and_mutation():
    graph = make([double, square], CHAIN)
    new = MHD_Graph(graph.nodes, graph.edges, {graph.topo}, device="cpu", retain_graph=True)
    assert new.retain_graph is True
    with pytest.raises(TypeError, match="bool"):
        MHD_Graph(graph.nodes, graph.edges, {graph.topo}, device="cpu", retain_graph=1)
    x = set_input(graph)
    graph.forward([0, 1])
    graph.retain_graph = 1
    x.grad = torch.tensor(5.)
    with pytest.raises(TypeError, match="bool"):
        graph.backward([2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(5.))


@pytest.mark.parametrize("mode", ["implicit", "zero", "nonzero"])
@pytest.mark.parametrize("legacy", [False, True])
def test_node_save_load_preserves_input_semantics(tmp_path, mode, legacy):
    graph = make([double, square], CHAIN)
    root = graph.get_node_by_id(2)
    if mode != "implicit":
        root.gradient_message.update_initial(torch.tensor(0. if mode == "zero" else 3.))
    path = str(tmp_path / "nodes.pth")
    updown_node(graph.nodes, path, "down")
    if legacy:
        state = torch.load(path, weights_only=True)
        for node_state in state["node_messages"].values():
            node_state["gradient_message"].pop("initial_state_explicit")
        torch.save(state, path)
    root.gradient_message.update_initial(torch.tensor(99.))
    updown_node(graph.nodes, path, "up")
    root.reset()
    x = set_input(graph)
    graph.forward([0, 1]).backward([2, 3])
    multiplier = 3. if mode == "nonzero" else 1. if mode == "implicit" or legacy else 0.
    torch.testing.assert_close(x.grad, torch.tensor(24. * multiplier))


@pytest.mark.parametrize("explicit", [False, True])
def test_merge_preserves_zero_input_provenance_and_defaults_lifetime(explicit):
    a, b = make([double, square], CHAIN), make([double, square], CHAIN)
    a.retain_graph = b.retain_graph = True
    for graph in (a, b):
        if explicit:
            graph.get_node_by_id(2).gradient_message.update_initial(torch.tensor(0.))
    merged = MHD_Graph.merge_graph({a, b}, device="cpu")
    assert merged.retain_graph is False
    x = set_input(merged)
    merged.forward([0, 1]).backward([2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(0. if explicit else 24.))


def test_merge_rejects_implicit_and_explicit_zero_conflict():
    a, b = make([double, square], CHAIN), make([double, square], CHAIN)
    b.get_node_by_id(2).gradient_message.update_initial(torch.tensor(0.))
    with pytest.raises(ValueError, match="n2.*initial_state_explicit"):
        MHD_Graph.merge_graph({a, b}, device="cpu")


def test_pipeline_rejects_retain_before_distributed_initialization(monkeypatch):
    import V5.MHD_Utils_V5 as utils
    graph = make([double, square], CHAIN)
    graph.retain_graph = True
    def forbidden():
        raise AssertionError("must reject before distributed initialization")
    monkeypatch.setattr(utils, "initialize_mhd_distributed", forbidden)
    config = MHD_ParallelConfig(pipeline_size=2, pipeline_stages={"e0": 0, "e1": 1})
    with pytest.raises(ValueError, match="PP.*retain_graph"):
        prepare_mhd_model(graph, ["n0"], ["n2"], levels=[0, 1],
                          backward_levels=[2, 3], parallel=config)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_trainer_amp_accumulation_matches_native(tmp_path, precision, device):
    if device == "cuda" and os.environ.get("MHD_TEST_CUDA") != "1":
        pytest.skip("explicit GPU opt-in required")
    if device == "cpu" and precision == "fp16":
        pytest.skip("CPU FP16 is outside Trainer contract")
    target = torch.device(device)
    module = torch.nn.Linear(2, 2, bias=False).to(target)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.25, 0.5], [-0.5, 0.25]], device=target))
    native = torch.nn.Linear(2, 2, bias=False).to(target)
    native.load_state_dict(module.state_dict())
    graph = make([module, square], CHAIN, initial={i: torch.zeros(2, 2) for i in range(3)}, device=device)
    v = torch.tensor([[1., 0.], [-0.5, 2.]], device=target)
    graph.get_node_by_id(1).gradient_message.update_initial(v)
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.1)
    native_optimizer = torch.optim.SGD(native.parameters(), lr=0.1)
    trainer = MHD_Trainer(
        graph, optimizer, MHD_Monitor(["n1"]), [0, 1], [3],
        criteria=lambda g: g.get_node_by_id(1).feature_message.current_state.mean(),
        save_dir=str(tmp_path), input_nodes=["n0"], output_nodes=["n1"],
        precision=precision, grad_accum_steps=2,
        distributed_context=MHD_DistributedContext(0, 0, 1, target, "gloo"),
    )
    enabled = device == "cuda" and precision == "fp16"
    trainer.grad_scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=enabled)
    scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=enabled)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for step in range(2):
        value = torch.tensor([[1., 0.5 + step], [0.25, -1.]], device=target)
        with torch.autocast(device, dtype=dtype, enabled=precision != "fp32"):
            hidden = native(value)
        if enabled:
            scaler.scale(hidden)
        scale = float(scaler.get_scale())
        torch.autograd.backward(hidden, v * scale / 2)
        trainer.train_step({"n0": value})
        torch.testing.assert_close(graph.get_node_by_id(1).gradient_message.current_state,
                                   (v / 2).to(hidden.dtype))
    scaler.step(native_optimizer)
    scaler.update()
    torch.testing.assert_close(module.weight, native.weight)
    torch.testing.assert_close(module.weight.grad, native.weight.grad)


@pytest.mark.skipif(os.environ.get("MHD_TEST_CUDA") != "1", reason="explicit GPU opt-in required")
@pytest.mark.parametrize("explicit", [False, True])
def test_device_migration_preserves_zero_input_provenance(explicit):
    graph = make([double, square], CHAIN)
    if explicit:
        graph.get_node_by_id(2).gradient_message.update_initial(torch.tensor(0.))
    graph.to("cuda")
    x = set_input(graph)
    graph.forward([0, 1]).backward([2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(0. if explicit else 24., device="cuda"))


@pytest.mark.parametrize("mode", ["implicit", "zero", "nonzero"])
@pytest.mark.parametrize("format", ["new", "old_nested", "old_flat"])
def test_trainer_checkpoint_roundtrip_preserves_backward_input(tmp_path, mode, format):
    from torch.distributed import checkpoint as dcp
    graph = make([Scale(), square], CHAIN)
    root = graph.get_node_by_id(2)
    if mode != "implicit":
        root.gradient_message.update_initial(torch.tensor(0. if mode == "zero" else 3.))
    trainer = MHD_Trainer(
        graph, torch.optim.SGD(graph.parameters(), lr=0.1), MHD_Monitor(["n2"]),
        [0, 1], [2, 3], criteria=lambda g: g.get_node_by_id(2).feature_message.current_state,
        save_dir=str(tmp_path), input_nodes=["n0"], output_nodes=["n2"],
        distributed_context=MHD_DistributedContext(0, 0, 1, torch.device("cpu"), "gloo"),
    )
    state = trainer._checkpoint_state(1, legacy_node_format=format == "old_flat")
    if format == "old_nested":
        for saved in state["node_messages"].values():
            saved["gradient_message"].pop("initial_state_explicit")
    dcp.save(state, checkpoint_id=str(tmp_path / "epoch_1"), no_dist=True)
    root.gradient_message.update_initial(torch.tensor(99.))
    assert trainer.load_checkpoint(epoch=1) == 1
    assert not graph._forward_trace
    x = set_input(graph)
    graph.forward([0, 1]).backward([2, 3])
    multiplier = 3. if mode == "nonzero" else 1. if mode == "implicit" or format != "new" else 0.
    torch.testing.assert_close(x.grad, torch.tensor(24. * multiplier))


def test_pipeline_rejects_explicit_boundary_input_without_silent_fallback(monkeypatch):
    import V5.MHD_Utils_V5 as utils
    graph = make([double, square], CHAIN)
    graph.get_node_by_id(2).gradient_message.update_initial(torch.tensor(0.))
    monkeypatch.setattr(utils, "initialize_mhd_distributed",
                        lambda: pytest.fail("must reject before allocating a process group"))
    config = MHD_ParallelConfig(pipeline_size=2, pipeline_stages={"e0": 0, "e1": 1})
    with pytest.raises(ValueError, match="PP.*显式 Gradient"):
        prepare_mhd_model(graph, ["n0"], ["n2"], levels=[0, 1],
                          backward_levels=[2, 3], parallel=config)


def test_pipeline_rechecks_lifetime_after_preparation():
    from V5.MHD_Utils_V5 import _MHD_PipelineModel
    graph = make([double, square], CHAIN)
    class Schedule:
        def step(self, **kwargs):
            pytest.fail("must reject before starting the native schedule")
    model = _MHD_PipelineModel(Schedule(), Schedule(), torch.nn.Identity(), 0, 2,
                              None, 1, torch.device("cpu"), graph, ["n2"])
    graph.retain_graph = True
    with pytest.raises(ValueError, match="PP.*retain_graph"):
        model()


def test_merge_and_node_roundtrip_preserve_runtime_tensor_shapes_and_dtypes(tmp_path):
    graphs = [make([double, square], CHAIN) for _ in range(2)]
    for graph in graphs:
        set_input(graph, torch.tensor([2., 3.], dtype=torch.float16, requires_grad=True))
        graph.get_node_by_id(2).gradient_message.update_initial(torch.tensor([1., -2.]))
        graph.forward([0, 1]).backward([2, 3])
    merged = MHD_Graph.merge_graph(set(graphs), device="cpu")
    path = str(tmp_path / "runtime.pth")
    updown_node(merged.nodes, path, "down")
    restored = make([double, square], CHAIN)
    updown_node(restored.nodes, path, "up")
    for graph in (merged, restored):
        root = graph.get_node_by_id(2)
        assert root.feature_message.initial_state.shape == ()
        assert root.gradient_message.initial_state.shape == (2,)
        assert root.gradient_message.initial_state.dtype == torch.float32
        assert root.gradient_message.current_state.dtype == torch.float16
        graph.get_node_by_id(0).feature_message.current_state.requires_grad_(True)
        graph.forward([0, 1]).backward([2, 3])
        torch.testing.assert_close(graph.get_node_by_id(0).gradient_message.current_state,
                                   torch.tensor([16., -48.], dtype=torch.float16))


@pytest.mark.parametrize("device,precision", [("cpu", "bf16"), ("cuda", "fp16")])
def test_amp_checkpoint_into_fresh_graph_restores_runtime_dtype_and_continues_training(tmp_path, device, precision):
    if device == "cuda" and os.environ.get("MHD_TEST_CUDA") != "1":
        pytest.skip("explicit GPU opt-in required")
    target = torch.device(device)
    def trainer():
        module = torch.nn.Linear(2, 2, bias=False).to(target)
        with torch.no_grad():
            module.weight.fill_(0.25)
        # Deliberately scalar placeholders: actual batch tensors have shape (2, 2).
        graph = make([module, square], CHAIN, device=device)
        graph.get_node_by_id(1).gradient_message.update_initial(torch.tensor([[1., 0.], [-1., 2.]], device=target))
        result = MHD_Trainer(
            graph, torch.optim.SGD(graph.parameters(), lr=0.01), MHD_Monitor(["n1"]),
            [0, 1], [3], criteria=lambda g: g.get_node_by_id(1).feature_message.current_state.mean(),
            save_dir=str(tmp_path), input_nodes=["n0"], output_nodes=["n1"],
            precision=precision,
            distributed_context=MHD_DistributedContext(0, 0, 1, target, "gloo"),
        )
        result.grad_scaler = torch.amp.GradScaler("cuda", init_scale=128,
                                                enabled=device == "cuda" and precision == "fp16")
        return result, module
    source, source_module = trainer()
    value = torch.tensor([[1., 2.], [3., 4.]], device=target)
    source.train_step({"n0": value})
    source.save_checkpoint(1)
    restored, restored_module = trainer()
    restored.load_checkpoint(epoch=1)
    root = restored.mhd_graph.get_node_by_id(1)
    assert root.gradient_message.initial_state.dtype == torch.float32
    assert root.gradient_message.current_state.dtype == (torch.bfloat16 if precision == "bf16" else torch.float16)
    assert root.feature_message.current_state.shape == (2, 2)
    assert root.feature_message.initial_state.shape == ()
    source.train_step({"n0": value})
    restored.train_step({"n0": value})
    torch.testing.assert_close(source_module.weight, restored_module.weight)
