import torch

from learned_impact.physical_transition import ChainState, conservative_projection


def test_bond_force_is_derivative_of_the_same_stored_energy():
    u = torch.randn(3, 17, dtype=torch.float64, requires_grad=True)
    state = ChainState(u, torch.zeros_like(u))
    response = state.bond_response(37.)
    gradient = torch.autograd.grad(response['stored_energy'].sum(), u)[0]
    incidence = torch.zeros_like(u)
    incidence[:, :-1] -= response['force']
    incidence[:, 1:] += response['force']
    torch.testing.assert_close(gradient, incidence, atol=1e-12, rtol=1e-12)


def test_projection_preserves_energy_momentum_and_center_for_each_island():
    torch.manual_seed(12)
    initial = ChainState(torch.randn(4, 17, dtype=torch.float64), torch.randn(4, 17, dtype=torch.float64))
    proposed = ChainState(torch.randn(4, 17, dtype=torch.float64)*11, torch.randn(4, 17, dtype=torch.float64)*7)
    final = conservative_projection(initial, proposed, .8, 37.)
    torch.testing.assert_close(final.energy(37.), initial.energy(37.), atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(final.velocity.sum(-1), initial.velocity.sum(-1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(final.displacement.mean(-1), (initial.displacement+.8*initial.velocity).mean(-1), atol=1e-12, rtol=1e-12)
    solo = conservative_projection(ChainState(initial.displacement[:1], initial.velocity[:1]),
                                  ChainState(proposed.displacement[:1], proposed.velocity[:1]), .8, 37.)
    torch.testing.assert_close(final.displacement[:1], solo.displacement, atol=0, rtol=0)


def test_rigid_and_collapsed_proposals_are_finite_with_finite_gradients():
    u = torch.tensor([[0., 0., 0.], [-1., 0., 1.]], dtype=torch.float64)
    v = torch.ones_like(u)*2
    a = torch.zeros_like(u, requires_grad=True)
    b = torch.zeros_like(u, requires_grad=True)
    final = conservative_projection(ChainState(u, v), ChainState(a, b), .3)
    torch.testing.assert_close(final.displacement, u+.3*v, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(final.velocity, v, atol=1e-12, rtol=1e-12)
    (final.displacement.sum()+final.velocity.sum()).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
