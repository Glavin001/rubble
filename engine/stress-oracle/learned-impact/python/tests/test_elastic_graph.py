import pytest
import torch

from learned_impact.capability_probes import chain
from learned_impact.elastic_graph import ElasticGraph, ElasticGraphTransition, exact_graph_transition
from learned_impact.physical_transition import ChainState
from learned_impact.wave_operator import CoupledWaveOperator


def test_new_graph_path_matches_the_original_chain_operator():
    torch.manual_seed(23)
    n = 19
    graph = ElasticGraph(chain(n), torch.ones(n-1, dtype=torch.float64)*100,
                         torch.ones(n, dtype=torch.float64), torch.zeros(n, dtype=torch.bool))
    fitted = CoupledWaveOperator(torch.randn(24, dtype=torch.float64)*.001, .8)
    state = ChainState(torch.randn(3, n, dtype=torch.float64), torch.randn(3, n, dtype=torch.float64))
    a, b = fitted.step(state), ElasticGraphTransition(fitted, graph).step(state)
    torch.testing.assert_close(a.displacement, b.displacement, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.velocity, b.velocity, atol=1e-10, rtol=1e-12)


def test_mass_normalized_graph_teacher_conserves_energy_with_fixed_supports():
    graph = ElasticGraph(chain(5), torch.tensor([10., 20., 15., 17.], dtype=torch.float64),
                         torch.tensor([2., 1., 3., 1.5, 4.], dtype=torch.float64),
                         torch.tensor([True, False, False, False, True]))
    state = ChainState(torch.tensor([[0., .1, -.2, .3, 0.]], dtype=torch.float64), torch.zeros(1, 5, dtype=torch.float64))
    full = exact_graph_transition(graph, state, .8)
    half = exact_graph_transition(graph, exact_graph_transition(graph, state, .4), .4)
    torch.testing.assert_close(full.displacement, half.displacement, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(graph.energy(full), graph.energy(state), atol=1e-12, rtol=1e-12)
    assert (full.displacement[:, graph.fixed] == 0).all()
    response = graph.response(state)
    assert (response['internal_nodal_force']+response['support_reaction'])[:, graph.fixed].abs().sum() == 0


def test_unsupported_spectrum_and_nonzero_fixed_motion_are_rejected():
    fitted = CoupledWaveOperator(torch.zeros(8, dtype=torch.float64), .8)
    graph = ElasticGraph(chain(3), torch.ones(2, dtype=torch.float64)*1000,
                         torch.ones(3, dtype=torch.float64), torch.tensor([True, False, False]))
    with pytest.raises(ValueError, match='spectrum'):
        ElasticGraphTransition(fitted, graph)
    graph.stiffness[:] = 10
    model = ElasticGraphTransition(fitted, graph)
    with pytest.raises(ValueError, match='fixed supports'):
        model.step(ChainState(torch.ones(1, 3), torch.zeros(1, 3)))
