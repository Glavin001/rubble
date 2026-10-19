import torch

from learned_impact.capability_probes import exact_transition, WaveTransition, graph


def test_exact_chain_preserves_energy_momentum_and_composition():
    torch.manual_seed(2)
    u, v = torch.randn(4, 19, dtype=torch.float64), torch.randn(4, 19, dtype=torch.float64)
    k, dt = 37., .19
    a, b = exact_transition(u, v, k, dt)
    half_u, half_v = exact_transition(u, v, k, dt/2)
    twice_u, twice_v = exact_transition(half_u, half_v, k, dt/2)
    def energy(x, y):
        return .5 * y.square().sum(-1) + .5 * k * x.diff().square().sum(-1)
    torch.testing.assert_close(energy(a, b), energy(u, v), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(b.sum(-1), v.sum(-1), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(a.mean(-1), (u+dt*v).mean(-1), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(a, twice_u, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(b, twice_v, atol=1e-10, rtol=1e-10)


def test_learned_pair_decoder_preserves_each_islands_momentum_for_arbitrary_weights():
    torch.manual_seed(3)
    model = WaveTransition(8).double()
    with torch.no_grad():
        model.output.weight.normal_()
        model.output.bias.normal_()
    u, v = torch.randn(3, 11, dtype=torch.float64), torch.randn(3, 11, dtype=torch.float64)
    endpoints, hierarchy = graph(11, 3, 8)
    a, b = model(u, v, .2, endpoints, hierarchy)
    torch.testing.assert_close(b.sum(-1), v.sum(-1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.mean(-1), (u+.2*v).mean(-1), atol=1e-12, rtol=1e-12)
