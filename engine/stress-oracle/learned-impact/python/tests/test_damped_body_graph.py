import math

import pytest
import torch

from learned_impact.damped_body_graph import DampedBodyGraph, DenseDampedReference
from learned_impact.elastic_body_graph import ElasticBodyGraph, exact_body_transition
from learned_impact.physical_transition import ChainState


def oscillator(damping=4.):
    dtype = torch.float64
    return DampedBodyGraph(ElasticBodyGraph(
        torch.tensor([[0, 1]]), torch.eye(3, dtype=dtype)[None],
        torch.zeros(1, 3, dtype=dtype), torch.zeros(1, 3, dtype=dtype),
        torch.full((1, 6), 100., dtype=dtype), torch.eye(6, dtype=dtype).repeat(2, 1, 1),
        torch.tensor([[True]*6, [False]+[True]*5])),
        torch.full((1, 6), damping, dtype=dtype), torch.zeros(2, 6, 6, dtype=dtype))


def test_damped_oscillator_phase_and_affine_force_match_closed_form():
    graph = oscillator()
    model = DenseDampedReference(graph)
    u = torch.zeros(12, dtype=torch.float64)
    v = u.clone()
    u[6], v[6] = .03, -.2
    state = ChainState(u, v)
    force = torch.zeros_like(u)
    force[6] = .4
    dt, decay = .237, 2.
    frequency = math.sqrt(100-decay**2)
    offset = force[6]/100
    a = u[6]-offset
    b = (v[6]+decay*a)/frequency
    c, s = math.cos(frequency*dt), math.sin(frequency*dt)
    expected_u = offset+math.exp(-decay*dt)*(a*c+b*s)
    expected_v = math.exp(-decay*dt)*(-decay*(a*c+b*s)+frequency*(-a*s+b*c))
    result = model.step(state, dt, force)
    torch.testing.assert_close(result.displacement[6], expected_u, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(result.velocity[6], expected_v, rtol=1e-12, atol=1e-14)
    split = model.step(model.step(state, .1, force), dt-.1, force)
    torch.testing.assert_close(result.displacement, split.displacement, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(result.velocity, split.velocity, rtol=1e-12, atol=1e-14)


def test_physical_power_balances_energy_and_zero_damping_matches_elastic_teacher():
    graph = oscillator()
    u = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    v = torch.arange(12, dtype=torch.float64)*.01*(~graph.body.fixed).flatten()
    state = ChainState(u, v)
    force = graph.physical_apply(v)
    torch.testing.assert_close((force*v).sum(), graph.dissipated_power(v))
    # For unit mass the instantaneous energy rate is dE/dx v + v dot a.
    gradient, = torch.autograd.grad(graph.body.energy(state), u)
    acceleration = -graph.body.stiffness_apply(u)-force
    torch.testing.assert_close((gradient*v+v*acceleration).sum(), -graph.dissipated_power(v))
    undamped = oscillator(0.)
    model = DenseDampedReference(undamped)
    actual, expected = model.step(state, .31), exact_body_transition(undamped.body, state, .31)
    torch.testing.assert_close(actual.displacement, expected.displacement, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(actual.velocity, expected.velocity, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match='nonnegative'):
        DampedBodyGraph(graph.body, -graph.bond_damping, graph.mass_damping)
    with pytest.raises(ValueError, match='256'):
        DenseDampedReference(graph, max_free_dof=1000)
    with pytest.raises(ValueError, match='positive interval'):
        model.step(state, float('nan'))


def test_small_endpoint_response_does_not_erase_earlier_stress_peak():
    graph = oscillator(8.)
    u = torch.zeros(12, dtype=torch.float64)
    v = u.clone()
    v[6] = 1.
    state, model = ChainState(u, v), DenseDampedReference(graph)
    omega = math.sqrt(100-4**2)
    first_peak = math.atan(omega/4)/omega
    peak = model.step(state, first_peak)
    late = model.step(state, 3.)
    peak_force = graph.body.response(peak)['bond_force'][0, 0].abs()
    late_force = graph.body.response(late)['bond_force'][0, 0].abs()
    assert peak_force > 5.
    assert late_force < 1e-4
    assert graph.body.energy(late) < 1e-10*graph.body.energy(state)
