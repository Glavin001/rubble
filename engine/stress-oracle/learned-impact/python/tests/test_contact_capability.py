import math

import pytest
import torch

from learned_impact.capability_probes import wave_data
from learned_impact.contact_capability import ElasticImpact
from learned_impact.wave_operator import fit


def test_independent_elastic_collision_exchanges_velocity_and_then_separates():
    data = wave_data(601)
    scales = ((data[2]-data[0]-.8*data[1]).double().square().mean().sqrt(),
              (data[3]-data[1]).double().square().mean().sqrt())
    learned = fit(data, 16, .8, scales, force_consistent=True)
    system = ElasticImpact(learned, [1., 1.])
    state = system.initial()
    dt = .9*system.event_resolution_dt
    events = []
    for i in range(math.ceil(.7/dt)):
        state, changes = system.step(state, dt)
        events.extend([{**event, 'time': event['time']+i*dt} for event in changes])
    assert len(events) == 1 and not events[0]['touching']
    assert abs(events[0]['time']-math.pi/10.) < 1e-8
    torch.testing.assert_close(state.mechanics.velocity[[0, 6]], torch.tensor([0., 1.], dtype=torch.float64), atol=2e-8, rtol=0)
    assert abs(system.energy(state.mechanics)-.5) < 2e-8
    assert not state.touching
    assert system.penetration(state.mechanics) < 0
    with pytest.raises(ValueError, match='event resolution'):
        system.step(state, system.event_resolution_dt*2)
