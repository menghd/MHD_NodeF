from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V4.MHD_Utils_V4 import (
    MHD_DistributedContext,
    MHD_Monitor,
    MHD_Trainer,
    create_mhd_optimizer,
    prepare_mhd_model,
    updown_node,
)


def loss_criterion(graph):
    return graph.get_node_by_name("loss").feature_message.current_state


def make_training_graph(batch_size: int = 2):
    linear = nn.Linear(1, 1, bias=False)
    nodes = {
        MHD_Node(0, "input", MHD_Node.Message(torch.zeros(batch_size, 1))),
        MHD_Node(1, "prediction", MHD_Node.Message(torch.zeros(batch_size, 1))),
        MHD_Node(2, "loss", MHD_Node.Message(torch.zeros(()))),
    }
    edges = {
        MHD_Edge(0, "linear", [MHD_Edge.Operation(linear)]),
        MHD_Edge(
            1,
            "criterion",
            [MHD_Edge.Operation(lambda prediction: prediction.square().mean())],
        ),
    }
    forward_0 = torch.tensor([[-1, 1, 0], [0, 0, 0]])
    forward_1 = torch.tensor([[0, 0, 0], [0, -1, 1]])
    sort_0 = torch.tensor([[0, 1, 0], [0, 0, 0]])
    sort_1 = torch.tensor([[0, 0, 0], [0, 0, 1]])
    topo = MHD_Topo(
        [forward_0, forward_1, -forward_1, -forward_0],
        [sort_0, sort_1, sort_1, sort_0],
    )
    return MHD_Graph(nodes, edges, {topo}, device="cpu"), linear


def test_graph_adapter_is_private_translation_not_a_second_executor():
    graph, linear = make_training_graph()
    adapter = prepare_mhd_model(
        graph,
        ["input"],
        ["prediction", "loss"],
        levels=[0, 1],
        context=MHD_DistributedContext(0, 0, 1, torch.device("cpu"), "gloo"),
    )
    inputs = torch.randn(2, 1, requires_grad=True)
    outputs = adapter({"input": inputs})
    torch.testing.assert_close(outputs["prediction"], linear(inputs))
    graph.backward(levels=[2, 3])
    assert linear.weight.grad is not None
    assert graph.get_node_by_name("input").gradient_message.current_state.shape == inputs.shape


def test_monitor_keeps_legacy_metric_names_and_can_show_all_message_states():
    graph, _ = make_training_graph()
    graph.forward(levels=[0, 1])
    legacy = MHD_Monitor(["loss"])
    metrics = legacy.monitor_node(graph)
    assert set(metrics) == {"loss_mean", "loss_sum", "loss_min", "loss_max"}
    detailed = MHD_Monitor(
        ["loss"],
        node_states=(
            "feature_message.initial_state",
            "feature_message.current_state",
            "gradient_message.initial_state",
            "gradient_message.current_state",
        ),
        statistics=("mean",),
    )
    assert len(detailed.monitor_node(graph)) == 4


def test_node_checkpoint_round_trip_contains_four_states(tmp_path):
    graph, _ = make_training_graph()
    node = graph.get_node_by_name("input")
    node.feature_message.initial_state.fill_(1)
    node.feature_message.current_state.fill_(2)
    node.gradient_message.initial_state.fill_(3)
    node.gradient_message.current_state.fill_(4)
    path = tmp_path / "nodes.pt"
    updown_node(graph.nodes, str(path), "down")
    node.feature_message.initial_state.zero_()
    node.feature_message.current_state.zero_()
    node.gradient_message.initial_state.zero_()
    node.gradient_message.current_state.zero_()
    updown_node(graph.nodes, str(path), "up", target_device=torch.device("cpu"))
    torch.testing.assert_close(node.feature_message.initial_state, torch.ones(2, 1))
    torch.testing.assert_close(node.feature_message.current_state, torch.full((2, 1), 2.0))
    torch.testing.assert_close(node.gradient_message.initial_state, torch.full((2, 1), 3.0))
    torch.testing.assert_close(node.gradient_message.current_state, torch.full((2, 1), 4.0))


def test_trainer_uses_explicit_paths_and_updates_parameters(tmp_path):
    graph, linear = make_training_graph()
    optimizer = create_mhd_optimizer(graph, default_lr=0.1)
    trainer = MHD_Trainer(
        graph,
        optimizer,
        MHD_Monitor(["loss"]),
        forward_levels=[0, 1],
        backward_levels=[2, 3],
        criteria=loss_criterion,
        save_dir=str(tmp_path),
        input_nodes=["input"],
        output_nodes=["loss"],
    )
    before = linear.weight.detach().clone()
    metrics = trainer.train_step({"input": torch.ones(2, 1)})
    assert "loss" in metrics
    assert not torch.equal(before, linear.weight)
    state = trainer._checkpoint_state(epoch=1)
    assert set(state["node_messages"]["input"]) == {
        "feature_message", "gradient_message"
    }
    assert state["trainer"]["forward_levels"] == [0, 1]
    assert state["trainer"]["backward_levels"] == [2, 3]


def test_trainer_requires_an_explicit_criteria_callable(tmp_path):
    graph, _ = make_training_graph()
    with pytest.raises(TypeError, match="criteria"):
        MHD_Trainer(
            graph,
            create_mhd_optimizer(graph, default_lr=0.1),
            MHD_Monitor(["loss"]),
            forward_levels=[0, 1],
            backward_levels=[2, 3],
            save_dir=str(tmp_path),
        )


def test_trainer_always_collects_required_criteria_without_monitor_duplication(tmp_path):
    graph, _ = make_training_graph()
    trainer = MHD_Trainer(
        graph,
        create_mhd_optimizer(graph, default_lr=0.1),
        MHD_Monitor([]),
        forward_levels=[0, 1],
        backward_levels=[2, 3],
        criteria=loss_criterion,
        save_dir=str(tmp_path),
        input_nodes=["input"],
        output_nodes=["loss"],
    )
    metrics = trainer.train_step({"input": torch.ones(2, 1)})
    assert "loss" in metrics


def test_gradient_accumulation_window_requires_one_path(tmp_path):
    graph, _ = make_training_graph()
    trainer = MHD_Trainer(
        graph,
        lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        MHD_Monitor(["loss"]),
        forward_levels=[0, 1],
        backward_levels=[2, 3],
        criteria=loss_criterion,
        save_dir=str(tmp_path),
        input_nodes=["input"],
        output_nodes=["loss"],
        grad_accum_steps=2,
    )
    trainer.train_step({"input": torch.ones(2, 1)})
    try:
        trainer.train_step({"input": torch.ones(2, 1)}, backward_levels=[3, 2])
    except ValueError as exc:
        assert "梯度累积窗口" in str(exc)
    else:
        raise AssertionError("path change inside accumulation window was accepted")
