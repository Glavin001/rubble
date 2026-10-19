import copy
import json

import pytest

from learned_impact.native_comparison import compare
from test_runtime_snapshot import fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def example(tmp_path):
    graph, first = fixture()
    graph['scene']['impactors'] = []
    first['world']['impactors'] = []
    first['world']['mechanical_energy'] = 0.
    first['world']['dissipated_energy'] = 0.
    first['world']['clusters'] = [0, 0]
    last = copy.deepcopy(first)
    last['world']['time'] = graph['dt']
    oracle, run = tmp_path / 'oracle', tmp_path / 'run'
    write(oracle / 'graph.json', graph)
    write(run / 'native/graph.json', graph)
    for directory in (oracle, run / 'native'):
        write(directory / 'snapshots/000000.json', first)
        write(directory / 'snapshots/000001.json', last)
    write(run / 'plan.json', {'status': 'completed', 'frames': 1,
                             'checkpoint': {'sha256': 'fixture'}})
    write(run / 'native/reports.json', [{'report': {'analytic_bond_evaluations': 0}}])
    return oracle, run


def test_native_comparison_requires_matching_worlds_and_no_oracle_fallback(tmp_path):
    oracle, run = example(tmp_path)
    result = compare(oracle, run)
    assert result['final_position_rmse_m'] == 0
    assert result['breaks'] == {'oracle': 0, 'model': 0, 'extra': 0, 'missed': 0}
    write(run / 'native/reports.json', [{'report': {'analytic_bond_evaluations': 1}}])
    with pytest.raises(ValueError, match='fallback'):
        compare(oracle, run)


def test_failed_run_cannot_be_reported_as_success(tmp_path):
    oracle, run = example(tmp_path)
    write(run / 'plan.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='failed or incomplete'):
        compare(oracle, run)


def test_mismatched_initial_state_is_rejected(tmp_path):
    oracle, run = example(tmp_path)
    path = run / 'native/snapshots/000000.json'
    initial = json.loads(path.read_text())
    initial['world']['centers'][0][0] += 1
    write(path, initial)
    with pytest.raises(ValueError, match='start identically'):
        compare(oracle, run)
