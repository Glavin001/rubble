import pytest
import torch

from learned_impact.physical_transition import ChainState
from learned_impact.wave_operator import CoupledWaveOperator, stiffness_apply


def test_coupled_operator_preserves_linear_superposition_and_rigid_motion():
    torch.manual_seed(19)
    model = CoupledWaveOperator(torch.randn(16, dtype=torch.float64)*.01, .8)
    a, b = [ChainState(torch.randn(3, 11, dtype=torch.float64), torch.randn(3, 11, dtype=torch.float64)) for _ in range(2)]
    x, y = model.step(a), model.step(b)
    combined = model.step(ChainState(2*a.displacement-3*b.displacement, 2*a.velocity-3*b.velocity))
    torch.testing.assert_close(combined.displacement, 2*x.displacement-3*y.displacement, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(combined.velocity, 2*x.velocity-3*y.velocity, atol=1e-10, rtol=1e-12)
    shift, speed = 7., 2.
    shifted = model.step(ChainState(a.displacement+shift, a.velocity+speed))
    torch.testing.assert_close(shifted.displacement, x.displacement+shift+.8*speed, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(shifted.velocity, x.velocity+speed, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(x.velocity.sum(-1), a.velocity.sum(-1), atol=1e-10, rtol=1e-12)
    with pytest.raises(ValueError, match='interval'):
        model.step(a, dt=.4)


def test_shared_propagators_couple_displacement_and_velocity():
    torch.manual_seed(4)
    model = CoupledWaveOperator(torch.randn(12, dtype=torch.float64)*.01, .8)
    x = torch.randn(2, 13, dtype=torch.float64)
    zero = torch.zeros_like(x)
    from_displacement = model.step(ChainState(x, zero))
    from_velocity = model.step(ChainState(zero, x))
    torch.testing.assert_close(from_displacement.displacement, from_velocity.velocity, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(from_displacement.velocity, -stiffness_apply(from_velocity.displacement), atol=1e-10, rtol=1e-12)


def test_unconstrained_force_response_and_nonfinite_domains_are_rejected():
    with pytest.raises(ValueError, match='constant-force consistency'):
        CoupledWaveOperator(torch.zeros(8, dtype=torch.float64), .8).require_constant_force_consistency()
    for dt, stiffness in [(float('nan'), 100.), (.8, float('inf'))]:
        with pytest.raises(ValueError, match='invalid physical operator'):
            CoupledWaveOperator(torch.zeros(8), dt, stiffness)
