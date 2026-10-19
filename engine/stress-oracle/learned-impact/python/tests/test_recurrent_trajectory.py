import copy

import numpy as np
import pytest
import torch

from learned_impact.recurrent_trajectory import RecurrentTrajectoryNet, forecast, sequences, fit_scales
from learned_impact.trajectory_response import CompleteTrajectoryNet, current_batch, forecast as old_forecast
from learned_impact.runtime_snapshot import NativeGraph
from test_runtime_snapshot import fixture


def native():
    graph, snapshot = fixture()
    graph['scene']['impactors'] = []
    snapshot['world']['impactors'] = []
    graph = NativeGraph(graph)
    kinematics = {'format': 'native-boundary-kinematics/1', 'time': 0.,
                  'values': np.zeros((graph.e, 12)).tolist()}
    return graph, snapshot, kinematics


def test_zero_heads_preserve_existing_native_wire_contract_exactly():
    graph, snapshot, kinematics = native()
    old = CompleteTrajectoryNet(h=8, p=4, knots=3)
    new = RecurrentTrajectoryNet(h=8, memory=4, knots=3)
    expected = old_forecast(old, graph, snapshot, kinematics)
    result, memory = forecast(new, graph, snapshot, kinematics)
    assert result == expected
    assert memory.ids.tolist() == [0] and memory.next_time == graph.dt
    assert not any(name in new._modules for name in ('mp', 'attn', 'chunk_delta', 'world_motion'))
    with pytest.raises(RuntimeError, match='explicit'):
        new(current_batch(graph, snapshot, kinematics))


def test_recurrent_inference_needs_only_current_state_and_rejects_repeated_clock():
    graph, snapshot, kinematics = native()
    model = RecurrentTrajectoryNet(h=8, memory=4, knots=3)
    with torch.no_grad():
        model.complete_head.weight.normal_(0, .0001)
    expected, state = forecast(model, graph, snapshot, kinematics)
    snapshot['world']['mechanical_energy'] = float('nan')
    snapshot['world']['dissipated_energy'] = float('nan')
    observed, _ = forecast(model, graph, snapshot, kinematics)
    assert expected == observed
    with pytest.raises(ValueError, match='clock'):
        forecast(model, graph, snapshot, kinematics, state)
    snapshot['world']['time'] = graph.dt
    kinematics['time'] = graph.dt
    result, next_state = forecast(model, graph, snapshot, kinematics, state)
    assert result['start_time'] == graph.dt
    assert next_state.next_time == 2 * graph.dt


def test_sequence_admission_requires_full_prefix_and_one_partition():
    rows = [{'event': 'wall', 'partition': 'train', 'frame': i, 'start_time': .01*i, 'dt': .01}
            for i in range(3)]
    assert len(sequences(list(reversed(rows)))[0]) == 3
    with pytest.raises(ValueError, match='warmup'):
        sequences(rows[1:])
    bad = copy.deepcopy(rows)
    bad[1]['partition'] = 'test'
    with pytest.raises(ValueError, match='leakage'):
        sequences(bad)
    bad = copy.deepcopy(rows)
    bad[1]['start_time'] += .001
    with pytest.raises(ValueError, match='clock gap'):
        sequences(bad)


def test_normalization_rejects_heldout_parents_before_reading_targets():
    with pytest.raises(ValueError, match='training parents'):
        fit_scales(RecurrentTrajectoryNet(h=8), [{'partition': 'test'}])
