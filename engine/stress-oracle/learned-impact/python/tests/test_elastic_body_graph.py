import pytest
import torch

from learned_impact.elastic_body_graph import ElasticBodyGraph, ElasticBodyTransition, exact_body_transition
from learned_impact.physical_transition import ChainState
from learned_impact.wave_operator import CoupledWaveOperator


def beam():
    mass = torch.diag_embed(torch.tensor([[2., 2., 2., .2, .3, .4], [3., 3., 3., .4, .5, .6]], dtype=torch.float64))
    return ElasticBodyGraph(torch.tensor([[0, 1]]), torch.eye(3, dtype=torch.float64)[None],
                            torch.tensor([[.5, 0., 0.]], dtype=torch.float64),
                            torch.tensor([[-.5, 0., 0.]], dtype=torch.float64),
                            torch.tensor([[10., 12., 13., 1., 2., 3.]], dtype=torch.float64),
                            mass, torch.zeros(2, 6, dtype=torch.bool))


def test_six_dof_rigid_modes_work_and_force_moment_balance():
    graph = beam()
    centers = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float64)
    theta = torch.tensor([.1, -.2, .3], dtype=torch.float64).expand(2, -1)
    u = torch.linalg.cross(theta, centers)+torch.tensor([.2, .4, -.1])
    rigid = torch.cat((u, theta), -1).flatten()
    torch.testing.assert_close(graph.deformation(rigid), torch.zeros(1, 6, dtype=torch.float64), atol=1e-15, rtol=0)
    rng = torch.Generator().manual_seed(24)
    x = torch.randn(12, generator=rng, dtype=torch.float64, requires_grad=True)
    state = ChainState(x, torch.zeros_like(x))
    gradient, = torch.autograd.grad(graph.energy(state), x)
    response = graph.response(state)
    torch.testing.assert_close(gradient, -response['internal_nodal_force'])
    f = response['internal_nodal_force'].reshape(2, 6)
    torch.testing.assert_close(f[:, :3].sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torque = f[:, 3:]+torch.linalg.cross(centers, f[:, :3])
    torch.testing.assert_close(torque.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(graph.from_weighted(graph.to_weighted(x)), x)
    matrix = graph.apply(torch.eye(12, dtype=torch.float64)).T
    torch.testing.assert_close(matrix, matrix.T)
    assert torch.linalg.eigvalsh(matrix).max() <= graph.spectral_bound()*(1+1e-12)


def test_physical_time_scaling_matches_equivalent_dimensionless_system():
    graph = beam()
    model = CoupledWaveOperator(torch.arange(8, dtype=torch.float64)*.0001, .8)
    state = ChainState(torch.arange(12, dtype=torch.float64)*.001, torch.arange(12, dtype=torch.float64)*.01)
    force = torch.arange(12, dtype=torch.float64)*.1
    scale = .025
    physical = ElasticBodyTransition(model, graph, model.dt*scale).step(state, force)
    rescaled = ElasticBodyGraph(graph.endpoints, torch.eye(3, dtype=torch.float64)[None],
                               torch.tensor([[.5, 0., 0.]], dtype=torch.float64),
                               torch.tensor([[-.5, 0., 0.]], dtype=torch.float64),
                               graph.stiffness*scale**2, graph.mass_matrix, graph.fixed)
    dimensionless = ElasticBodyTransition(model, rescaled, model.dt).step(ChainState(state.displacement, scale*state.velocity), scale**2*force)
    torch.testing.assert_close(physical.displacement, dimensionless.displacement)
    torch.testing.assert_close(physical.velocity, dimensionless.velocity/scale)
    expected = exact_body_transition(graph, state, .02, force)
    half = exact_body_transition(graph, exact_body_transition(graph, state, .01, force), .01, force)
    torch.testing.assert_close(expected.displacement, half.displacement, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(expected.velocity, half.velocity, atol=1e-14, rtol=1e-12)
    work = (force*(expected.displacement-state.displacement)).sum()
    torch.testing.assert_close(graph.energy(expected)-graph.energy(state), work, atol=1e-14, rtol=1e-12)
    with pytest.raises(ValueError, match='spectrum and interval'):
        ElasticBodyTransition(model, graph, 100.)
