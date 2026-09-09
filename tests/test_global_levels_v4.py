from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo


def message(value: torch.Tensor) -> MHD_Node.Message:
    return MHD_Node.Message(value)


def operation(function) -> MHD_Edge.Operation:
    return MHD_Edge.Operation(function)


def level(
    num_edges: int,
    num_nodes: int,
    edge_id: int,
    heads: tuple[int, ...],
    tails: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    role = torch.zeros(num_edges, num_nodes, dtype=torch.int64)
    sort = torch.zeros_like(role)
    for index, node_id in enumerate(heads):
        role[edge_id, node_id] = -1
        sort[edge_id, node_id] = index
    for index, node_id in enumerate(tails, start=len(heads)):
        role[edge_id, node_id] = 1
        sort[edge_id, node_id] = index
    return role, sort


def graph_from_levels(nodes, edges, definitions) -> MHD_Graph:
    pairs = [level(len(edges), len(nodes), *definition) for definition in definitions]
    return MHD_Graph(set(nodes), set(edges), {MHD_Topo(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    )}, device=torch.device("cpu"))


def make_chain(seed: int = 7):
    torch.manual_seed(seed)
    first = nn.Linear(3, 4, bias=False)
    second = nn.Linear(4, 1, bias=False)
    nodes = [
        MHD_Node(0, "x", message(torch.zeros(2, 3))),
        MHD_Node(1, "hidden", message(torch.zeros(2, 4))),
        MHD_Node(2, "prediction", message(torch.zeros(2, 1))),
        MHD_Node(3, "loss", message(torch.zeros(()))),
    ]
    edges = [
        MHD_Edge(0, "first", [operation(first)]),
        MHD_Edge(1, "second", [operation(second)]),
        MHD_Edge(2, "criterion", [operation(lambda value: value.square().mean())]),
    ]
    definitions = [
        (0, (0,), (1,)),
        (1, (1,), (2,)),
        (2, (2,), (3,)),
        (2, (3,), (2,)),
        (1, (2,), (1,)),
        (0, (1,), (0,)),
    ]
    return graph_from_levels(nodes, edges, definitions), first, second


def test_message_defaults_validation_and_no_direct_state_fields():
    node = MHD_Node(0, "integer", message(torch.tensor([1, 2], dtype=torch.int64)))
    assert node.gradient_message.initial_state.dtype == torch.float32
    assert not hasattr(node, "initial_state")
    assert not hasattr(node, "current_state")
    node.feature_message.current_state.add_(2)
    node.reset()
    torch.testing.assert_close(node.feature_message.current_state, torch.tensor([1, 2]))
    node.feature_message.update_initial(torch.tensor([3, 4]))
    torch.testing.assert_close(node.feature_message.current_state, torch.tensor([3, 4]))
    with pytest.raises(ValueError, match="dtype"):
        MHD_Node.Message(torch.zeros(2), torch.zeros(2, dtype=torch.float64))


def test_operation_has_one_deterministic_positional_rule_and_propagates_errors():
    add = MHD_Edge(0, "add", [operation(lambda left, right: left + right)])
    result = add.execute_edge_operations([torch.tensor(2.0), torch.tensor(3.0)])
    torch.testing.assert_close(result[0], torch.tensor(5.0))
    scaled = MHD_Edge(1, "partial", [operation(partial(torch.mul, other=4.0))])
    torch.testing.assert_close(
        scaled.execute_edge_operations([torch.tensor(2.0)])[0], torch.tensor(8.0)
    )
    string = MHD_Edge(2, "string", [operation(".relu()")])
    torch.testing.assert_close(
        string.execute_edge_operations([torch.tensor(-2.0)])[0], torch.tensor(0.0)
    )

    def fail(_value):
        raise RuntimeError("operation failure")

    with pytest.raises(RuntimeError, match="operation failure"):
        MHD_Edge(3, "failure", [operation(fail)]).execute_edge_operations(
            [torch.tensor(1.0)]
        )


def test_forward_preserves_arbitrary_level_order_and_repetition():
    nodes = [
        MHD_Node(0, "x", message(torch.tensor(1.0))),
        MHD_Node(1, "y", message(torch.tensor(0.0))),
    ]
    edge = MHD_Edge(0, "increment", [operation(lambda value: value + 1)])
    graph = graph_from_levels(
        nodes,
        [edge],
        [(0, (0,), (1,)), (0, (1,), (0,))],
    )
    graph.forward(levels=[0, 1, 0])
    torch.testing.assert_close(graph.get_node_by_name("y").feature_message.current_state, torch.tensor(4.0))
    assert graph._last_forward_levels == (0, 1, 0)


@pytest.mark.parametrize("aggregation,expected", [("sum", 15.0), ("avg", 5.0), ("max", 10.0)])
def test_nary_message_aggregation(aggregation: str, expected: float):
    nodes = [
        MHD_Node(0, "x", message(torch.tensor(1.0))),
        MHD_Node(1, "out", message(torch.tensor(10.0)), aggregation=aggregation),
    ]
    edges = [
        MHD_Edge(0, "twice", [operation(lambda x: x * 2)]),
        MHD_Edge(1, "thrice", [operation(lambda x: x * 3)]),
    ]
    pairs = [level(2, 2, 0, (0,), (1,)), level(2, 2, 1, (0,), (1,))]
    role = pairs[0][0] + pairs[1][0]
    sort = pairs[0][1] + pairs[1][1]
    graph = MHD_Graph(set(nodes), set(edges), {MHD_Topo([role], [sort])}, device="cpu")
    graph.forward(levels=[0])
    torch.testing.assert_close(
        graph.get_node_by_name("out").feature_message.current_state,
        torch.tensor(expected),
    )


def test_full_backward_matches_native_autograd_and_populates_messages():
    graph, first, second = make_chain()
    x = torch.randn(2, 3, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = x
    graph.forward(levels=[0, 1, 2])
    expected_loss = second(first(x)).square().mean()
    expected = torch.autograd.grad(expected_loss, (x, first.weight, second.weight), retain_graph=True)
    graph.backward(levels=[3, 4, 5])
    torch.testing.assert_close(x.grad, expected[0])
    torch.testing.assert_close(first.weight.grad, expected[1])
    torch.testing.assert_close(second.weight.grad, expected[2])
    torch.testing.assert_close(
        graph.get_node_by_name("x").gradient_message.current_state, expected[0]
    )


def test_partial_backward_masks_unselected_upstream_path():
    graph, first, second = make_chain()
    x = torch.randn(2, 3, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = x
    graph.forward(levels=[0, 1, 2])
    graph.backward(levels=[3, 4])
    assert first.weight.grad is None
    assert second.weight.grad is not None
    assert x.grad is None or torch.count_nonzero(x.grad) == 0
    hidden_gradient = graph.get_node_by_name("hidden").gradient_message.current_state
    assert torch.isfinite(hidden_gradient).all() and torch.count_nonzero(hidden_gradient) > 0


def test_backward_rejects_overlap_wrong_order_and_unrelated_topology():
    graph, _, _ = make_chain()
    graph.get_node_by_name("x").feature_message.current_state = torch.randn(2, 3, requires_grad=True)
    graph.forward(levels=[0, 1, 2])
    with pytest.raises(ValueError, match="不得重叠"):
        graph.backward(levels=[2, 3])

    graph, _, _ = make_chain()
    graph.get_node_by_name("x").feature_message.current_state = torch.randn(2, 3, requires_grad=True)
    graph.forward(levels=[0, 1, 2])
    with pytest.raises(ValueError, match="不可达"):
        graph.backward(levels=[5, 3])


class SquareWithTripleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        return gradient * 3 * value


def test_operation_supports_standard_custom_autograd_function():
    nodes = [
        MHD_Node(0, "x", message(torch.tensor(2.0))),
        MHD_Node(1, "loss", message(torch.tensor(0.0))),
    ]
    edge = MHD_Edge(0, "custom", [operation(SquareWithTripleGradient.apply)])
    graph = graph_from_levels(nodes, [edge], [(0, (0,), (1,)), (0, (1,), (0,))])
    x = torch.tensor(2.0, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = x
    graph.forward(levels=[0]).backward(levels=[1])
    torch.testing.assert_close(x.grad, torch.tensor(6.0))


def test_mermaid_draws_one_level_sequence_without_direction_styles():
    graph, _, _ = make_chain()
    from V4.MHD_Utils_V4 import display_graph
    diagram = display_graph(graph, levels=[0, 1, 1, 2, 3, 4, 5])
    assert "#1:L1" in diagram
    assert "#2:L1" in diagram
    assert "N0 -->|#0:L0" in diagram
    assert "E0 -->|#6:L5" in diagram
    assert "Feature" not in diagram
    assert "Gradient" not in diagram
    assert "-." not in diagram


def test_forward_and_backward_levels_are_mandatory():
    graph, _, _ = make_chain()
    with pytest.raises(TypeError):
        graph.forward()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        graph.backward()  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="必须先执行"):
        graph.backward(levels=[3, 4, 5])


def test_retain_graph_supports_a_second_native_backward_and_then_consumes_trace():
    graph, first, second = make_chain()
    graph.get_node_by_name("x").feature_message.current_state = torch.randn(
        2, 3, requires_grad=True
    )
    graph.forward(levels=[0, 1, 2])
    graph.backward(levels=[3, 4, 5], retain_graph=True)
    first_gradient = first.weight.grad.detach().clone()
    second_gradient = second.weight.grad.detach().clone()
    graph.backward(levels=[3, 4, 5])
    torch.testing.assert_close(first.weight.grad, first_gradient * 2)
    torch.testing.assert_close(second.weight.grad, second_gradient * 2)
    with pytest.raises(RuntimeError, match="必须先执行"):
        graph.backward(levels=[3, 4, 5])


def test_path_validation_error_does_not_consume_a_valid_forward_trace():
    graph, first, _ = make_chain()
    graph.get_node_by_name("x").feature_message.current_state = torch.randn(
        2, 3, requires_grad=True
    )
    graph.forward(levels=[0, 1, 2])
    with pytest.raises(ValueError, match="不可达"):
        graph.backward(levels=[5, 3])
    graph.backward(levels=[3, 4, 5])
    assert first.weight.grad is not None


def test_a_second_forward_replaces_trace_with_the_new_occurrences():
    graph, first, second = make_chain()
    old_input = torch.randn(2, 3, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = old_input
    graph.forward(levels=[0, 1, 2])
    new_input = torch.randn(2, 3, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = new_input
    graph.forward(levels=[0, 1, 2]).backward(levels=[3, 4, 5])
    assert old_input.grad is None
    expected = torch.autograd.grad(
        second(first(new_input)).square().mean(), new_input
    )[0]
    torch.testing.assert_close(new_input.grad, expected)


class FailBackwardOnce(torch.autograd.Function):
    should_fail = True

    @staticmethod
    def forward(ctx, value):
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        if FailBackwardOnce.should_fail:
            FailBackwardOnce.should_fail = False
            raise RuntimeError("intentional backward failure")
        return gradient


def test_autograd_failure_removes_temporary_hooks_and_keeps_trace_for_retry():
    FailBackwardOnce.should_fail = True
    nodes = [
        MHD_Node(0, "x", message(torch.tensor(2.0))),
        MHD_Node(1, "loss", message(torch.tensor(0.0))),
    ]
    edge = MHD_Edge(0, "fail_once", [operation(FailBackwardOnce.apply)])
    graph = graph_from_levels(nodes, [edge], [(0, (0,), (1,)), (0, (1,), (0,))])
    value = torch.tensor(2.0, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = value
    graph.forward(levels=[0])
    with pytest.raises(RuntimeError, match="intentional backward failure"):
        graph.backward(levels=[1], retain_graph=True)
    graph.backward(levels=[1])
    torch.testing.assert_close(value.grad, torch.tensor(1.0))


def make_branch(shared: bool = False):
    left = nn.Linear(1, 1, bias=False)
    right = left if shared else nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        left.weight.fill_(2.0)
        right.weight.fill_(3.0)
    nodes = [
        MHD_Node(0, "x", message(torch.zeros(1))),
        MHD_Node(1, "left", message(torch.zeros(1))),
        MHD_Node(2, "right", message(torch.zeros(1))),
        MHD_Node(3, "loss", message(torch.zeros(()))),
    ]
    edges = [
        MHD_Edge(0, "left_edge", [operation(left)]),
        MHD_Edge(1, "right_edge", [operation(right)]),
        MHD_Edge(2, "join", [operation(lambda a, b: (a + b).sum())]),
    ]
    definitions = [
        (0, (0,), (1,)),
        (1, (0,), (2,)),
        (2, (1, 2), (3,)),
        (2, (3,), (1, 2)),
        (1, (2,), (0,)),
        (0, (1,), (0,)),
    ]
    return graph_from_levels(nodes, edges, definitions), left, right


def test_branch_selection_masks_only_the_unselected_parameter_contribution():
    graph, left, right = make_branch()
    graph.get_node_by_name("x").feature_message.current_state = torch.ones(1, requires_grad=True)
    graph.forward(levels=[0, 1, 2]).backward(levels=[3, 5])
    torch.testing.assert_close(left.weight.grad, torch.ones_like(left.weight))
    assert right.weight.grad is None
    assert torch.count_nonzero(
        graph.get_node_by_name("right").gradient_message.current_state
    ) > 0


def test_shared_parameter_keeps_only_selected_branch_contribution():
    graph, shared, same = make_branch(shared=True)
    assert shared is same
    graph.get_node_by_name("x").feature_message.current_state = torch.ones(1, requires_grad=True)
    graph.forward(levels=[0, 1, 2]).backward(levels=[3, 5])
    torch.testing.assert_close(shared.weight.grad, torch.ones_like(shared.weight))


def test_nonzero_gradient_initial_state_is_an_additional_native_seed():
    graph, first, second = make_chain()
    x = torch.randn(2, 3, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = x
    graph.get_node_by_name("prediction").gradient_message.update_initial(
        torch.full((2, 1), 0.25)
    )
    graph.forward(levels=[0, 1, 2])
    prediction = second(first(x))
    loss = prediction.square().mean()
    expected = torch.autograd.grad(
        (loss, prediction),
        x,
        grad_outputs=(torch.ones_like(loss), torch.full_like(prediction, 0.25)),
    )[0]
    graph.backward(levels=[3, 4, 5])
    torch.testing.assert_close(x.grad, expected)


def test_loss_scale_applies_to_loss_and_additional_gradient_seed_together():
    graph, first, second = make_chain()
    x = torch.randn(2, 3, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = x
    seed = torch.full((2, 1), 0.25)
    graph.get_node_by_name("prediction").gradient_message.update_initial(seed)
    graph.forward(levels=[0, 1, 2])

    prediction = second(first(x))
    loss = prediction.square().mean()
    expected = torch.autograd.grad(
        (loss, prediction),
        x,
        grad_outputs=(torch.ones_like(loss), seed),
    )[0]
    scale = 8.0
    graph._backward(levels=[3, 4, 5], retain_graph=False, loss_scale=scale)

    torch.testing.assert_close(x.grad, expected * scale)
    torch.testing.assert_close(
        graph.get_node_by_name("x").gradient_message.current_state,
        expected * scale,
    )


def test_unique_differentiable_scalar_terminal_is_required():
    nodes = [
        MHD_Node(0, "x", message(torch.tensor(1.0))),
        MHD_Node(1, "a", message(torch.tensor(0.0))),
        MHD_Node(2, "b", message(torch.tensor(0.0))),
    ]
    edges = [
        MHD_Edge(0, "a_edge", [operation(lambda x: x * 2)]),
        MHD_Edge(1, "b_edge", [operation(lambda x: x * 3)]),
    ]
    graph = graph_from_levels(
        nodes,
        edges,
        [
            (0, (0,), (1,)),
            (1, (0,), (2,)),
            (0, (1,), (0,)),
            (1, (2,), (0,)),
        ],
    )
    graph.get_node_by_name("x").feature_message.current_state = torch.tensor(1.0, requires_grad=True)
    graph.forward(levels=[0, 1])
    with pytest.raises(RuntimeError, match="唯一可微标量终点"):
        graph.backward(levels=[2])


def test_merge_averages_all_four_states_and_checks_aggregation():
    graph_a, _, _ = make_chain(seed=1)
    graph_b, _, _ = make_chain(seed=1)
    for value, graph in ((2.0, graph_a), (4.0, graph_b)):
        node = graph.get_node_by_name("x")
        node.feature_message.initial_state.fill_(value)
        node.feature_message.current_state.fill_(value + 1)
        node.gradient_message.initial_state.fill_(value + 2)
        node.gradient_message.current_state.fill_(value + 3)
    # Stateful Operations must share identity, so use the same modules.
    for edge_b in graph_b.edges:
        edge_a = graph_a.get_edge_by_name(edge_b.name)
        for op_b, op_a in zip(edge_b.edge_operations, edge_a.edge_operations):
            op_b.function = op_a.function
    merged = MHD_Graph.merge_graph({graph_a, graph_b}, device="cpu")
    merged_x = merged.get_node_by_name("x")
    torch.testing.assert_close(merged_x.feature_message.initial_state, torch.full((2, 3), 3.0))
    torch.testing.assert_close(merged_x.feature_message.current_state, torch.full((2, 3), 4.0))
    torch.testing.assert_close(merged_x.gradient_message.initial_state, torch.full((2, 3), 5.0))
    torch.testing.assert_close(merged_x.gradient_message.current_state, torch.full((2, 3), 6.0))
