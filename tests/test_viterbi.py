"""Unit tests for the current PathSnap src.core Viterbi implementation."""
import numpy as np
from src.core import transition_cost, viterbi, predicted_geometry


def sample_network():
    # Arc 0: 0->1; arc 1: 1->2; arc 2: 3->4 (disconnected)
    nodes = np.array([[0., 0.], [1., 0.], [2., 0.], [10., 0.], [11., 0.]])
    arcs = np.array([[0, 1], [1, 2], [3, 4]], dtype=int)
    return nodes, arcs


def test_transition_matrix_shape_and_connected_cost():
    xy, arcs = sample_network()
    cost = transition_cost(np.array([0, 2]), np.array([1, 2]), arcs, xy, gap=1)
    assert cost.shape == (2, 2)
    assert cost[0, 0] == 0.0  # arc 0 -> arc 1
    assert cost[1, 1] == 0.0  # stay on arc 2
    assert cost[0, 1] > 0.0  # disconnected


def test_long_gap_reduces_disconnected_penalty():
    xy, arcs = sample_network()
    a = transition_cost(np.array([0]), np.array([2]), arcs, xy, gap=1)
    b = transition_cost(np.array([0]), np.array([2]), arcs, xy, gap=10)
    assert b[0, 0] < a[0, 0]


def test_viterbi_returns_one_valid_arc_per_observation():
    xy, arcs = sample_network()
    candidates = np.array([[0, 2], [1, 2], [1, 2]], dtype=int)
    probs = np.array([[0.95, 0.05], [0.85, 0.15], [0.8, 0.2]])
    result = viterbi(probs, candidates, arcs, xy, np.array([0., 1., 2.]))
    assert result.shape == (3,)
    assert all(int(result[i]) in candidates[i] for i in range(3))
    assert result.tolist() == [0, 1, 1]


def test_viterbi_can_prefer_connected_path_over_local_peak():
    xy, arcs = sample_network()
    candidates = np.array([[0, 2], [1, 2]], dtype=int)
    probs = np.array([[0.95, 0.05], [0.45, 0.55]])
    result = viterbi(probs, candidates, arcs, xy, np.array([0., 1.]))
    assert result.tolist() == [0, 1]


def test_predicted_geometry_uses_lat_lon_order():
    nodes = np.array([[77.0, 12.0], [77.1, 12.1]])
    arcs = np.array([[0, 1]])
    assert predicted_geometry(np.array([0]), nodes, arcs) == [
        [[12.0, 77.0], [12.1, 77.1]]
    ]
