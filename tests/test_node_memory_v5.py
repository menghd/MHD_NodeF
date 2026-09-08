"""V5 node-memory semantics and required integrations."""
import pytest
import torch

from V4.MHD_Framework_V4 import MHD_Node as V4Node
from V5.MHD_Framework_V5 import MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph


def reference(values, aggregation):
    if aggregation == "sum":
        return sum(values[1:], values[0])
    if aggregation == "avg":
        return sum(values[1:], values[0]) / len(values)
    fn = {"max": torch.maximum, "min": torch.minimum, "mul": torch.mul}[aggregation]
    result = values[0]
    for value in values[1:]:
        result = fn(result, value)
    return result


@pytest.mark.parametrize("aggregation", ["sum", "avg", "max", "min", "mul"])
@pytest.mark.parametrize("memory", [False, True])
def test_values_and_gradients_match_explicit_native_expression(aggregation, memory):
    tensors = [torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True),
               torch.tensor([2.0, 1.0], dtype=torch.float64, requires_grad=True),
               torch.tensor([4.0, 2.0], dtype=torch.float64, requires_grad=True)]
    old, *incoming = tensors
    node = MHD_Node(0, "out", MHD_Node.Message(old), aggregation=aggregation, memory=memory)
    result = node.aggregate_messages(old, incoming)
    expected = reference(tensors if memory else incoming, aggregation)
    actual_grads = torch.autograd.grad(result.sum(), tensors, allow_unused=True)
    expected_grads = torch.autograd.grad(expected.sum(), tensors, allow_unused=True)
    torch.testing.assert_close(result, expected)
    for actual, target in zip(actual_grads, expected_grads):
        if target is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, target)


@pytest.mark.parametrize("aggregation", ["sum", "avg", "max", "min", "mul"])
def test_memory_true_preserves_v4_builtin_semantics(aggregation):
    current = torch.tensor([0.0, 2.0])
    incoming = [torch.tensor([2.0, 4.0]), torch.tensor([4.0, 8.0])]
    v4 = V4Node(0, "out", V4Node.Message(current), aggregation=aggregation)
    v5 = MHD_Node(0, "out", MHD_Node.Message(current), aggregation=aggregation, memory=True)
    torch.testing.assert_close(v5.aggregate_messages(current, incoming),
                               v4.aggregate_messages(current, incoming))


@pytest.mark.parametrize("memory", [False, True])
def test_empty_incoming_does_not_update(memory):
    current = torch.tensor(10.0, requires_grad=True)
    node = MHD_Node(0, "out", MHD_Node.Message(current), aggregation="avg", memory=memory)
    assert node.aggregate_messages(current, []) is current


@pytest.mark.parametrize("aggregation", ["sum", "avg", "max", "min", "mul"])
def test_single_incoming_without_memory_is_identity(aggregation):
    current = torch.tensor(10.0)
    incoming = torch.tensor(2.0, requires_grad=True)
    node = MHD_Node(0, "out", MHD_Node.Message(current), aggregation=aggregation, memory=False)
    result = node.aggregate_messages(current, [incoming])
    torch.testing.assert_close(result, incoming)
    result.backward()
    torch.testing.assert_close(incoming.grad, torch.ones_like(incoming))


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_memory_must_be_bool(value):
    with pytest.raises(TypeError, match="memory"):
        MHD_Node(0, "out", MHD_Node.Message(torch.tensor(0.0)), memory=value)


def make_graph(memory, device="cpu"):
    nodes = [
        MHD_Node(0, "x", MHD_Node.Message(torch.tensor(0.0))),
        MHD_Node(1, "out", MHD_Node.Message(torch.tensor(0.0)),
                 aggregation="avg", memory=memory),
        MHD_Node(2, "loss", MHD_Node.Message(torch.tensor(0.0))),
    ]
    edges = [
        MHD_Edge(0, "double", [MHD_Edge.Operation(DOUBLE)]),
        MHD_Edge(1, "quadruple", [MHD_Edge.Operation(QUADRUPLE)]),
        MHD_Edge(2, "loss", [MHD_Edge.Operation(SQUARE)]),
    ]
    role = torch.tensor([[-1, 1, 0], [-1, 1, 0], [0, 0, 0]])
    loss_role = torch.tensor([[0, 0, 0], [0, 0, 0], [0, -1, 1]])
    sort = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    loss_sort = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    topo = MHD_Topo([role, loss_role, -loss_role, -role], [sort, loss_sort, loss_sort, sort])
    return MHD_Graph(set(nodes), set(edges), {topo}, device=device)


def DOUBLE(x):
    return x * 2


def QUADRUPLE(x):
    return x * 4


def SQUARE(x):
    return x.square()


@pytest.mark.parametrize("memory, expected", [(False, 3.0), (True, 2.0)])
def test_graph_forward_backward_uses_selected_memory(memory, expected, device="cpu"):
    graph = make_graph(memory, device=device)
    x = torch.tensor(1.0, requires_grad=True, device=device)
    old = torch.tensor(0.0, requires_grad=True, device=device)
    graph.get_node_by_name("x").feature_message.current_state = x
    graph.get_node_by_name("out").feature_message.current_state = old
    graph.forward(levels=[0, 1])
    torch.testing.assert_close(graph.get_node_by_name("out").feature_message.current_state,
                               torch.tensor(expected, device=device))
    graph.backward(levels=[2, 3])
    torch.testing.assert_close(x.grad, torch.tensor(2 * expected ** 2, device=device))
    if memory:
        torch.testing.assert_close(old.grad, torch.tensor(2 * expected / 3, device=device))
    else:
        assert old.grad is None
    torch.testing.assert_close(graph.get_node_by_name("x").gradient_message.current_state, x.grad)


@pytest.mark.parametrize("memory", [False, True])
def test_merge_preserves_memory_and_equal_state(memory):
    first, second = make_graph(memory), make_graph(memory)
    first.get_node_by_name("out").feature_message.update_initial(torch.tensor(2.0))
    second.get_node_by_name("out").feature_message.update_initial(torch.tensor(2.0))
    merged = MHD_Graph.merge_graph({first, second}, device="cpu")
    assert merged.get_node_by_name("out").memory is memory
    torch.testing.assert_close(merged.get_node_by_name("out").feature_message.current_state,
                               torch.tensor(2.0))


def test_merge_rejects_conflicting_memory():
    with pytest.raises(ValueError, match="memory"):
        MHD_Graph.merge_graph({make_graph(False), make_graph(True)}, device="cpu")


def test_defaults_sum_incoming_and_replace_is_removed():
    node = MHD_Node(0, "out", MHD_Node.Message(torch.tensor(10.0)))
    assert node.memory is False
    assert node.aggregation == "sum"
    torch.testing.assert_close(node.aggregate_messages(torch.tensor(10.0),
                                                      [torch.tensor(2.0), torch.tensor(4.0)]),
                               torch.tensor(6.0))
    with pytest.raises(ValueError, match="replace"):
        MHD_Node(0, "out", MHD_Node.Message(torch.tensor(0.0)), aggregation="replace")


@pytest.mark.parametrize("memory", [False, True])
def test_callable_receives_current_or_none_without_signature_change(memory):
    seen = []
    def aggregate(current, incoming):
        seen.append(current)
        return sum(incoming) if current is None else current + sum(incoming)
    old = torch.tensor(10.0, requires_grad=True)
    node = MHD_Node(0, "out", MHD_Node.Message(old), aggregation=aggregate, memory=memory)
    result = node.aggregate_messages(old, [torch.tensor(2.0, requires_grad=True)])
    assert seen[0] is (old if memory else None)
    torch.testing.assert_close(result, torch.tensor(12.0 if memory else 2.0))
    result.backward()
    if memory:
        torch.testing.assert_close(old.grad, torch.tensor(1.0))
    else:
        assert old.grad is None


@pytest.mark.parametrize("memory", [False, True])
def test_utils_prune_and_node_state_roundtrip_preserve_memory(tmp_path, memory):
    from V5.MHD_Utils_V5 import updown_node, prune_isolated_graph
    graph = make_graph(memory)
    graph.nodes.add(MHD_Node(3, "unused", MHD_Node.Message(torch.tensor(0.0))))
    graph.topo.role_matrices = [torch.sparse_coo_tensor(r.indices(), r.values(), (r.shape[0], r.shape[1] + 1), is_coalesced=True) for r in graph.topo.role_matrices]
    graph.topo.sort_matrices = [torch.sparse_coo_tensor(r.indices(), r.values(), (r.shape[0], r.shape[1] + 1), is_coalesced=True) for r in graph.topo.sort_matrices]
    graph.update_indices()
    graph = prune_isolated_graph(graph, verbose=False)
    assert graph.get_node_by_name("out").memory is memory
    path = str(tmp_path / "nodes.pth")
    updown_node(graph.nodes, path, "down")
    graph.get_node_by_name("out").feature_message.update_initial(torch.tensor(8.0))
    updown_node(graph.nodes, path, "up")
    assert graph.get_node_by_name("out").memory is memory
    torch.testing.assert_close(graph.get_node_by_name("out").feature_message.current_state,
                               torch.tensor(0.0))


@pytest.mark.parametrize("memory", [False, True])
def test_trainer_update_matches_native_optimizer(tmp_path, memory):
    from V5.MHD_Utils_V5 import MHD_Trainer, MHD_Monitor, MHD_DistributedContext

    class Scale(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(value))
        def forward(self, value):
            return self.weight * value

    graph = make_graph(memory)
    for edge_id, value in ((0, 2.0), (1, 4.0)):
        graph.get_edge_by_id(edge_id).edge_operations[0].function = Scale(value)
    graph._register_all_params()
    optimizer = torch.optim.SGD(graph.parameters(), lr=0.1)
    native = [Scale(2.0), Scale(4.0)]
    native_optimizer = torch.optim.SGD(
        [parameter for module in native for parameter in module.parameters()], lr=0.1
    )
    trainer = MHD_Trainer(
        graph, optimizer, MHD_Monitor(["loss"]),
        forward_levels=[0, 1], backward_levels=[2, 3],
        criteria=lambda g: g.get_node_by_name("loss").feature_message.current_state.mean(),
        save_dir=str(tmp_path), input_nodes=["x"], output_nodes=["loss"],
        distributed_context=MHD_DistributedContext(0, 0, 1, torch.device("cpu"), "gloo"),
    )
    x = torch.tensor(1.0)
    native_loss = ((native[0](x) + native[1](x)) / (3 if memory else 2)).square()
    native_loss.backward()
    native_optimizer.step()
    trainer.train_step({"x": x})
    for edge_id, module in enumerate(native):
        torch.testing.assert_close(
            graph.get_edge_by_id(edge_id).edge_operations[0].function.weight, module.weight
        )
    assert graph.get_node_by_name("out").memory is memory
