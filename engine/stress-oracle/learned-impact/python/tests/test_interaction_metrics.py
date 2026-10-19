from copy import deepcopy
import numpy as np
import pytest
from learned_impact.interaction_metrics import compare_interactions


def frame(t, x, v, name='ram'):
    return {'time': t, 'impactors': [{'name': name, 'center': [x, 0., 0.],
        'velocity': [v, 0., 0.], 'mass': 2., 'driven': False}]}


def test_rebound_is_a_direction_failure_even_when_energy_is_identical():
    oracle = [frame(0, 0, 3), frame(1, 3, 3), frame(2, 6, 3)]
    model = [frame(0, 0, 3), frame(1, -2, -3), frame(2, -5, -3)]
    r = compare_interactions(oracle, model)['actors'][0]
    assert r['final_direction_disagreement']
    assert r['oracle']['translational_ke_j'] == r['model']['translational_ke_j']
    assert r['model']['first_saved_reverse_time_s'] == 1
    assert r['final_position_error_m'] == 11
    assert r['model']['net_forward_momentum_change_kg_mps'] == -12
    assert r['velocity_vector_rmse_mps'] == pytest.approx(np.sqrt(27))


def test_time_weighting_and_identity_matching_ignore_actor_order():
    oracle = [frame(0, 0, 1), frame(.1, .1, 1), frame(1, 1, 1)]
    model = deepcopy(oracle); model[1]['impactors'][0]['center'][0] += 2
    for frames in [oracle, model]:
        for f in frames:
            f['impactors'].append({**deepcopy(f['impactors'][0]), 'name': 'second'})
    for f in model: f['impactors'].reverse()
    r = compare_interactions(oracle, model)['actors'][0]
    # Squared error 0,4,0 integrated over [.0,.1,1.0] has mean 2.
    assert r['position_vector_rmse_m'] == pytest.approx(np.sqrt(2))
    assert not r['final_direction_disagreement']


def test_no_actor_and_stationary_actor_do_not_silently_pass_direction_check():
    empty = [{'time': t, 'impactors': []} for t in [0, 1]]
    assert compare_interactions(empty, empty)['status'] == 'no_external_actors'
    rest = [frame(0, 0, 0), frame(1, 0, 0)]
    r = compare_interactions(rest, rest)['actors'][0]
    assert r['axis'] is None and r['final_direction_disagreement'] is None
    assert 'final_direction' not in r['model']


@pytest.mark.parametrize('fault', ['clock', 'identity', 'initial', 'nonfinite', 'mass', 'duplicate'])
def test_invalid_comparisons_fail_closed(fault):
    a = [frame(0, 0, 1), frame(1, 1, 1)]; b = deepcopy(a)
    if fault == 'clock': b[-1]['time'] = 2
    if fault == 'identity': b[-1]['impactors'][0]['name'] = 'wrong'
    if fault == 'initial': b[0]['impactors'][0]['velocity'][0] = 2
    if fault == 'nonfinite': b[-1]['impactors'][0]['center'][0] = float('nan')
    if fault == 'mass': b[-1]['impactors'][0]['mass'] = 3
    if fault == 'duplicate': b[-1]['impactors'].append(b[-1]['impactors'][0])
    with pytest.raises(ValueError): compare_interactions(a, b)
