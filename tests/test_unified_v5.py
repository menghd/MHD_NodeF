"""Public V5 contracts: selected scalar roots, sparse topology, fresh merges."""
import os
import pytest
import torch
from V5.MHD_Framework_V5 import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph
from V5.MHD_Utils_V5 import display_graph, prune_isolated_graph


def double(x):
    return 2 * x


def square(x):
    return x.square()


class Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return self.weight * x


def make(functions, levels, count=3, memory=False, sparse=False, initial=None, device="cpu"):
    initial = initial or {}
    nodes = {
        MHD_Node(i, f"n{i}", MHD_Node.Message(initial.get(i, torch.tensor(0.0))), memory=memory)
        for i in range(count)
    }
    edges = {MHD_Edge(i, f"e{i}", [MHD_Edge.Operation(fn)]) for i, fn in enumerate(functions)}
    roles, sorts = [], []
    for specs in levels:
        role = torch.zeros(len(edges), count, dtype=torch.int64)
        sort = torch.zeros_like(role)
        for eid, heads, tails in specs:
            for pos, nid in enumerate(heads):
                role[eid, nid], sort[eid, nid] = -1, pos
            for pos, nid in enumerate(tails):
                role[eid, nid], sort[eid, nid] = 1, pos
        roles.append(role.to_sparse() if sparse else role)
        sorts.append(sort.to_sparse() if sparse else sort)
    return MHD_Graph(nodes, edges, {MHD_Topo(roles, sorts)}, device=device)


CHAIN = [[(0, [0], [1])], [(1, [1], [2])], [(1, [2], [1])], [(0, [1], [0])]]


def set_input(graph, value=None):
    x = torch.tensor(3.0, requires_grad=True, device=graph.device) if value is None else value
    graph.get_node_by_id(0).feature_message.current_state = x
    return x


@pytest.mark.parametrize("levels", [[2, 3], [2], [3]])
@pytest.mark.parametrize("sparse", [False, True])
def test_global_partial_and_local_match_native(levels, sparse, monkeypatch):
    scale = Scale()
    graph = make([scale, square], CHAIN, sparse=sparse)
    x = set_input(graph)
    calls = []
    original = torch.autograd.backward
    def backward(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "backward", backward)
    graph.forward([0, 1])
    graph.backward(levels)
    assert len(calls) == 1
    native_x = torch.tensor(3.0, requires_grad=True)
    native_w = torch.tensor(2.0, requires_grad=True)
    h = native_w * native_x
    expected = h if levels == [3] else h.square()
    gx, gw = torch.autograd.grad(expected, (native_x, native_w))
    if levels == [2]:
        assert scale.weight.grad is None
        torch.testing.assert_close(x.grad, torch.tensor(0.0))
    else:
        torch.testing.assert_close(x.grad, gx)
        torch.testing.assert_close(scale.weight.grad, gw)
    h_gradient = 1.0 if levels == [3] else 12.0
    torch.testing.assert_close(graph.get_node_by_id(1).gradient_message.current_state,
                               torch.tensor(h_gradient))


@pytest.mark.parametrize("memory", [False, True])
def test_repeated_levels_state_versions_match_native(memory):
    triple = lambda x: 3 * x
    levels = [[(0, [0], [1])], [(1, [1], [0])],
              [(0, [1], [0])], [(1, [0], [1])]]
    graph = make([double, triple], levels, count=2, memory=memory,
                 initial={1: torch.tensor(0.5)})
    x = set_input(graph, torch.tensor(1.0, requires_grad=True))
    graph.forward([0, 1, 0])
    current_x = graph.get_node_by_id(0).feature_message.current_state
    current_h = graph.get_node_by_id(1).feature_message.current_state
    native_x = torch.tensor(1.0, requires_grad=True)
    h0 = 2 * native_x + (0.5 if memory else 0)
    x1 = 3 * h0 + (native_x if memory else 0)
    h1 = 2 * x1 + (h0 if memory else 0)
    gx, gx1 = torch.autograd.grad(h1, (native_x, x1))
    torch.testing.assert_close(current_h, h1)
    graph.backward([2, 3, 2])
    torch.testing.assert_close(x.grad, gx)
    torch.testing.assert_close(graph.get_node_by_id(0).gradient_message.current_state, gx1)
    torch.testing.assert_close(graph.get_node_by_id(1).gradient_message.current_state, torch.tensor(1.0))
    assert graph.get_node_by_id(0).feature_message.current_state is current_x


def test_historical_root_does_not_write_historical_gradient_to_current_feature():
    levels = [[(0, [0], [1])], [(1, [0], [1])], [(0, [1], [0])]]
    graph = make([double, square], levels, count=2)
    x = set_input(graph)
    graph.forward([0, 1])
    latest = graph.get_node_by_id(1).feature_message.current_state
    graph.backward([2])
    torch.testing.assert_close(x.grad, torch.tensor(2.0))
    assert graph.get_node_by_id(1).feature_message.current_state is latest
    torch.testing.assert_close(graph.get_node_by_id(1).gradient_message.current_state, torch.tensor(0.0))


@pytest.mark.parametrize("memory", [False, True])
def test_post_aggregation_local_root_with_unselected_producer(memory):
    levels = [[(0, [0], [1]), (1, [0], [1])], [(0, [1], [0])]]
    graph = make([double, square], levels, count=2, memory=memory)
    graph.get_node_by_id(1).aggregation = "avg"
    x = set_input(graph)
    old = torch.tensor(1.0, requires_grad=True)
    graph.get_node_by_id(1).feature_message.current_state = old
    graph.forward([0])
    graph.backward([1])
    torch.testing.assert_close(x.grad, torch.tensor(2.0 / (3 if memory else 2)))
    if memory:
        torch.testing.assert_close(old.grad, torch.tensor(1.0 / 3))
    else:
        assert old.grad is None


def test_memory_connects_updates_without_explicit_edge_between_versions():
    levels = [[(0, [0], [1])], [(1, [0], [1])],
              [(1, [1], [0])], [(0, [1], [0])]]
    graph = make([double, square], levels, count=2, memory=True)
    x = set_input(graph)
    graph.forward([0, 1])
    graph.backward([2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(8.0))


def test_identity_alias_repeated_argument_has_native_gradient():
    levels = [[(0, [0], [1]), (1, [0, 1], [2])],
              [(1, [2], [0, 1]), (0, [1], [0])]]
    graph = make([lambda x: x, lambda x, alias: x * alias], levels)
    x = set_input(graph)
    graph.forward([0])
    graph.backward([1])
    torch.testing.assert_close(x.grad, torch.tensor(6.0))


def test_detached_metric_does_not_hide_scalar_terminal():
    levels = [[(0, [0], [1])], [(1, [1], [2])],
              [(1, [2], [1])], [(0, [1], [0])]]
    graph = make([square, lambda x: x.detach()], levels)
    x = set_input(graph)
    graph.forward([0, 1])
    graph.backward([2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(6.0))


@pytest.mark.parametrize("case", ["vector", "multiple", "none", "order", "unmatched", "overlap", "input_shape"])
def test_invalid_backward_is_atomic(case):
    scale = Scale()
    if case == "multiple":
        levels = [[(0, [0], [1]), (1, [0], [2])],
                  [(0, [1], [0]), (1, [2], [0])]]
        graph = make([scale, square], levels)
        forward, backward = [0], [1]
    elif case == "none":
        graph = make([lambda x: x.detach()], [[(0, [0], [1])], [(0, [1], [0])]], count=2)
        forward, backward = [0], [1]
    else:
        graph = make([scale, square], CHAIN)
        forward, backward = [0, 1], [2, 3]
    x = set_input(graph, torch.ones(2, requires_grad=True) if case == "vector" else None)
    if case == "vector":
        backward = [3]
    if case == "order":
        backward = [3, 2]
    if case == "unmatched":
        forward = [0]
        backward = [2]
    if case == "overlap":
        backward = [0]
    graph.forward(forward)
    if case == "input_shape":
        graph.get_node_by_id(2).gradient_message.update_initial(torch.ones(2))
    saved = {}
    for node in graph.nodes:
        node.gradient_message.current_state = torch.full_like(node.gradient_message.current_state, 7)
        saved[node.id] = node.gradient_message.current_state
    x.grad = torch.full_like(x, 11)
    for parameter in graph.parameters():
        parameter.grad = torch.full_like(parameter, 13)
    trace = graph._forward_trace
    with pytest.raises((ValueError, RuntimeError)):
        graph.backward(backward)
    assert graph._forward_trace is trace
    for node in graph.nodes:
        assert node.gradient_message.current_state is saved[node.id]
    torch.testing.assert_close(x.grad, torch.full_like(x, 11))
    for parameter in graph.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 13))


def test_retain_graph_allows_local_then_global_native_accumulation():
    graph = make([double, square], CHAIN)
    x = set_input(graph)
    graph.forward([0, 1])
    graph.retain_graph = True
    graph.backward([3])
    torch.testing.assert_close(x.grad, torch.tensor(2.0))
    graph.retain_graph = False
    graph.backward([2, 3])
    # Existing MHD behavior clears current input leaf gradients per call.
    torch.testing.assert_close(x.grad, torch.tensor(24.0))
    assert not graph._forward_trace


def test_dense_and_coo_have_same_public_topology_behavior():
    a = make([double, square], CHAIN)
    b = make([double, square], CHAIN, sparse=True)
    assert a.topo == b.topo and hash(a.topo) == hash(b.topo)
    assert a.topo.to_list() == b.topo.to_list()
    assert a.topo.to_list("sort") == b.topo.to_list("sort")
    assert a.sort_nodes_by_topo(0, 0) == b.sort_nodes_by_topo(0, 0)
    assert display_graph(a, [0, 1, 0]) == display_graph(b, [0, 1, 0])
    for matrix in a.topo.role_matrices + a.topo.sort_matrices:
        assert matrix.layout == torch.sparse_coo and matrix.is_coalesced()
        assert (matrix.values() != 0).all()
    assert a.topo.get_topo(0, 0, 0) == -1
    assert a.topo.get_topo(99, 0, 0) == 0


def test_sparse_operations_never_densify(monkeypatch):
    graph = make([double, square], CHAIN, count=4)
    other = make([double, square], CHAIN, count=4, sparse=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected dense conversion")
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    merged = MHD_Graph.merge_graph({graph, other}, device="cpu")
    assert merged.topo == graph.topo
    hash(merged.topo)
    merged.topo.get_topo(0, 0, 1)
    merged.sort_nodes_by_topo()
    display_graph(merged, [0, 1])
    prune_isolated_graph(merged, verbose=False)
    merged.to("cpu")
    set_input(merged)
    merged.forward([0, 1]).backward([2, 3])
    assert len(merged.nodes) == 3


def test_huge_sparse_shape_does_not_allocate_dense_storage(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected dense conversion")
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    shape = (10**8, 10**8)
    role = torch.sparse_coo_tensor(torch.tensor([[1, 1, 9], [2, 2, 7]]),
                                   torch.tensor([-2, 1, 0]), shape)
    sort = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.int64),
                                   torch.empty(0, dtype=torch.int64), shape)
    topo = MHD_Topo([role], [sort])
    assert topo.role_matrices[0]._nnz() == 1
    assert topo.get_topo(0, 1, 2) == -1
    assert hash(topo) == hash(MHD_Topo([role], [sort]))


def test_duplicate_coordinates_validate_after_coalescing():
    role = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 0]]),
                                   torch.tensor([1, 1]), (1, 1))
    with pytest.raises(ValueError, match="role"):
        MHD_Topo([role], [torch.zeros(1, 1, dtype=torch.int64)])


@pytest.mark.parametrize("kind", ["feature", "gradient"])
@pytest.mark.parametrize("state", ["initial", "current"])
def test_merge_state_conflicts_identify_exact_field(kind, state):
    a, b = make([double, square], CHAIN), make([double, square], CHAIN)
    msg = getattr(b.get_node_by_id(1), f"{kind}_message")
    setattr(msg, f"{state}_state", torch.tensor(9.0))
    with pytest.raises(ValueError, match=f"{kind}_message.{state}_state"):
        MHD_Graph.merge_graph({a, b}, device="cpu")


@pytest.mark.parametrize("difference", ["shape", "dtype", "requires_grad"])
def test_merge_rejects_equal_values_with_different_metadata(difference):
    a, b = make([double, square], CHAIN), make([double, square], CHAIN)
    value = {"shape": torch.zeros(1), "dtype": torch.tensor(0., dtype=torch.float64),
             "requires_grad": torch.tensor(0., requires_grad=True)}[difference]
    b.get_node_by_id(1).feature_message.current_state = value
    with pytest.raises(ValueError, match="feature_message.current_state"):
        MHD_Graph.merge_graph({a, b}, device="cpu")


def test_merged_states_are_independent_without_source_autograd_history():
    a, b = make([double, square], CHAIN), make([double, square], CHAIN)
    xa, xb = set_input(a), set_input(b)
    a.forward([0, 1])
    b.forward([0, 1])
    merged = MHD_Graph.merge_graph({a, b}, device="cpu")
    assert not merged._forward_trace
    with pytest.raises(RuntimeError):
        merged.backward([2, 3])
    for nid in range(3):
        for kind in ("feature", "gradient"):
            for which in ("initial", "current"):
                source = getattr(getattr(a.get_node_by_id(nid), f"{kind}_message"), f"{which}_state")
                result = getattr(getattr(merged.get_node_by_id(nid), f"{kind}_message"), f"{which}_state")
                torch.testing.assert_close(source, result)
                assert source.data_ptr() != result.data_ptr()
                assert result.grad_fn is None
    merged.forward([0, 1]).backward([2, 3])
    assert xa.grad is None and xb.grad is None
    torch.testing.assert_close(merged.get_node_by_id(0).gradient_message.current_state, torch.tensor(24.0))


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trainer_amp_and_accumulation_match_native(tmp_path, precision, local, device):
    from V5.MHD_Utils_V5 import MHD_Trainer, MHD_Monitor, MHD_DistributedContext
    if device == "cuda" and os.environ.get("MHD_TEST_CUDA") != "1":
        pytest.skip("GPU tests are opt-in via MHD_TEST_CUDA=1 and CUDA_VISIBLE_DEVICES")
    if device == "cpu" and precision == "fp16":
        pytest.skip("CPU float16 is outside the Trainer precision contract")
    target = torch.device(device)
    module = torch.nn.Linear(2, 1, bias=False).to(target)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.25, 0.5]], device=target))
    native = torch.nn.Linear(2, 1, bias=False).to(target)
    native.load_state_dict(module.state_dict())
    graph = make([module, lambda h: h.square().sum()], CHAIN, device=device)
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.1)
    native_optimizer = torch.optim.SGD(native.parameters(), lr=0.1)
    trainer = MHD_Trainer(
        graph, optimizer, MHD_Monitor(["n2"]), [0, 1], [2, 3],
        criteria=lambda g: g.get_node_by_id(2).feature_message.current_state,
        save_dir=str(tmp_path), input_nodes=["n0"], output_nodes=["n1", "n2"],
        precision=precision, grad_accum_steps=2,
        distributed_context=MHD_DistributedContext(0, 0, 1, target, "gloo"),
    )
    enabled = device == "cuda" and precision == "fp16"
    trainer.grad_scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=enabled)
    scaler = torch.amp.GradScaler("cuda", init_scale=128, enabled=enabled)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for step in range(2):
        value = torch.tensor([[1.0, 0.5 + step]], device=target)
        with torch.autocast(device, dtype=dtype, enabled=precision != "fp32"):
            hidden = native(value)
            loss = hidden if local else hidden.square().sum()
            objective = loss / 2
        scaler.scale(objective).backward()
        metrics = trainer.train_step({"n0": value}, backward_levels=[3] if local else [2, 3])
        if local:
            assert trainer.loss_node_name == "n1"
            assert metrics["n1"] == float(hidden.detach().item())
    scaler.step(native_optimizer)
    scaler.update()
    assert trainer._optimizer_steps == 1
    torch.testing.assert_close(module.weight, native.weight)
    torch.testing.assert_close(module.weight.grad, native.weight.grad)


def test_trainer_local_root_does_not_require_unique_full_forward_terminal(tmp_path):
    from V5.MHD_Utils_V5 import MHD_Trainer, MHD_Monitor, MHD_DistributedContext
    module = Scale()
    levels = [[(0, [0], [1]), (1, [0], [2])], [(0, [1], [0])]]
    graph = make([module, square], levels)
    trainer = MHD_Trainer(
        graph, torch.optim.SGD(graph.parameters(), lr=0.1), MHD_Monitor(["n1"]),
        [0], [1], criteria=lambda g: g.get_node_by_id(1).feature_message.current_state,
        save_dir=str(tmp_path), input_nodes=["n0"],
        distributed_context=MHD_DistributedContext(0, 0, 1, torch.device("cpu"), "gloo"),
    )
    trainer.train_step({"n0": torch.tensor(3.0)})
    torch.testing.assert_close(module.weight, torch.tensor(1.7))


@pytest.mark.parametrize("model", ["resnet", "transformer", "recurrent_hypergraph"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_existing_model_equivalence_with_v5(model, device):
    import importlib.util
    from pathlib import Path
    if device == "cuda" and os.environ.get("MHD_TEST_CUDA") != "1":
        pytest.skip("GPU model checks require explicit test-device selection")
    path = Path(__file__).with_name("model_smoke_v4.py")
    spec = importlib.util.spec_from_file_location("mhd_v5_model_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Reuse the historical native reference models without changing V4 files.
    reference.MHD_Graph, reference.MHD_Topo = MHD_Graph, MHD_Topo
    reference.MHD_Edge, reference.MHD_Node = MHD_Edge, MHD_Node
    reference.node = lambda nid, name, value, aggregation="replace": MHD_Node(
        nid, name, MHD_Node.Message(value),
        aggregation="sum" if aggregation == "replace" else aggregation,
        memory=aggregation != "replace",
    )
    torch.manual_seed(1234)
    getattr(reference, f"run_{model}")(torch.device(device))


def test_readme_python_examples():
    import re
    from pathlib import Path
    readme = Path(__file__).resolve().parents[1] / "V5" / "README.md"
    namespace = {}
    fence = chr(96) * 3
    for snippet in re.findall(fence + "python\n(.*?)" + fence, readme.read_text(), re.S):
        exec(compile(snippet, str(readme), "exec"), namespace)


def test_sparse_indices_must_be_in_bounds():
    role = torch.sparse_coo_tensor(torch.tensor([[2], [0]]), torch.tensor([1]), (1, 1))
    with pytest.raises(RuntimeError):
        MHD_Topo([role], [torch.zeros(1, 1, dtype=torch.int64)])
