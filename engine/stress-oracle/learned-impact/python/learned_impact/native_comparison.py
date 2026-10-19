"""Post-run comparison with a separately recorded native oracle trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np

from .interaction_metrics import compare_interactions
from .rollout_view import first_breaks, timing_scores, fragments
from .runtime_snapshot import NativeGraph


def compare(oracle, rollout):
    oracle, rollout = Path(oracle), Path(rollout)
    plan = json.loads((rollout / 'plan.json').read_text())
    if plan['status'] != 'completed':
        raise ValueError('failed or incomplete independent rollout')
    actual = json.loads((rollout / 'native/graph.json').read_text())
    reference = json.loads((oracle / 'graph.json').read_text())
    for key in ('format', 'scene', 'context', 'chunk_keys', 'bond_keys', 'statics', 'dt', 'initial_time'):
        if actual[key] != reference[key]:
            raise ValueError('oracle/model graph mismatch: ' + key)
    n = plan['frames']
    source = [json.loads((oracle / f'snapshots/{i:06}.json').read_text()) for i in range(n+1)]
    predicted = [json.loads((rollout / f'native/snapshots/{i:06}.json').read_text()) for i in range(n+1)]
    if source[0] != predicted[0]:
        raise ValueError('independent worlds must start identically')
    runtime = NativeGraph(actual)
    for snapshots in (source, predicted):
        for snapshot in snapshots:
            runtime.batch(snapshot, include_projectiles=True)
            if not np.isfinite([snapshot['world'][k] for k in
                                ('time', 'mechanical_energy', 'dissipated_energy')]).all():
                raise ValueError('nonfinite trajectory diagnostic')
    reports = json.loads((rollout / 'native/reports.json').read_text())
    if len(reports) != n or any(r['report']['analytic_bond_evaluations'] != 0 for r in reports):
        raise ValueError('missing execution evidence or analytical stress fallback')
    truth = [s['world'] for s in source]
    model = [s['world'] for s in predicted]
    interactions = compare_interactions(truth, model)
    a = first_breaks(np.asarray([s['joint_flags'] for s in source])[:, :, 3], actual['dt'])
    b = first_breaks(np.asarray([s['joint_flags'] for s in predicted])[:, :, 3], actual['dt'])
    mass = np.asarray(actual['statics']['chunk_mass'])
    active = np.asarray(truth[-1]['active'], bool) & np.asarray(model[-1]['active'], bool)
    error = np.asarray(model[-1]['centers']) - np.asarray(truth[-1]['centers'])
    return {'status': 'completed', 'frames': n, 'duration_seconds': n*actual['dt'],
            'checkpoint_sha256': plan['checkpoint']['sha256'],
            'scope': 'Independent native worlds; zero oracle resets or stress fallback. Break timing here is frame-resolved.',
            'breaks': {'oracle': int(np.isfinite(a).sum()), 'model': int(np.isfinite(b).sum()),
                       'extra': int((~np.isfinite(a) & np.isfinite(b)).sum()),
                       'missed': int((np.isfinite(a) & ~np.isfinite(b)).sum())},
            'timing': timing_scores(a, b, [0., actual['dt'], 2*actual['dt']]),
            'oracle_fragments': fragments(truth[-1], mass),
            'model_fragments': fragments(model[-1], mass),
            'final_position_rmse_m': float(np.sqrt(np.square(error[active]).mean())) if active.any() else None,
            'oracle_initial_mechanical_energy_j': truth[0]['mechanical_energy'],
            'oracle_final_mechanical_energy_j': truth[-1]['mechanical_energy'],
            'model_final_mechanical_energy_j': model[-1]['mechanical_energy'],
            'model_max_mechanical_energy_j': max(f['mechanical_energy'] for f in model),
            'interactions': interactions,
            'analytic_bond_evaluations': 0, 'promoted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', required=True)
    parser.add_argument('--rollout', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(args.oracle, args.rollout)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
