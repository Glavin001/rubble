"""Matched held-out comparison for oracle-force vs causal-preview checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .visualize import load_checkpoint, counts_metrics
from .train_physics import load_records, evaluate


def compare(dataset, oracle_weights, preview_weights):
    torch.set_num_threads(4)
    results, memberships, group_scores = [], [], []
    for weights in (oracle_weights, preview_weights):
        model, meta = load_checkpoint(weights)
        saved_split = json.loads((weights/'split.json').read_text())
        records, split = load_records(dataset, model, meta['run']['split_seed'], split_from=meta['run'].get('split_from'))
        if split['events'] != saved_split['events']:
            raise ValueError('dataset split differs from saved training membership')
        memberships.append(split)
        grouped = defaultdict(list)
        for row in records['test']:
            grouped[row[0]].append(row)
        events = {name:evaluate(model, rows, window_factor=meta['run'].get('evaluation_window',1)) for name,rows in grouped.items()}
        # Multiple clips of a physical scenario stay together in bootstrap samples.
        scores = defaultdict(lambda:np.zeros(3,dtype=np.int64))
        for name, metrics in events.items():
            scores[split['physical_groups'][name]] += [metrics['tp'],metrics['fp'],metrics['fn']]
        group_scores.append(scores)
        results.append(dict(checkpoint=meta, test=evaluate(model,records['test'],window_factor=meta['run'].get('evaluation_window',1)), events=events))
    if memberships[0] != memberships[1]:
        raise ValueError('experiments do not have identical grouped splits/deduplication')
    keys = sorted(group_scores[0])
    if keys != sorted(group_scores[1]):
        raise ValueError('different test scenarios')
    a, b = (np.array([scores[k] for k in keys]) for scores in group_scores)
    rng = np.random.default_rng(20261019)
    differences = []
    for _ in range(2000):
        ids = rng.integers(len(keys),size=len(keys))
        f1 = lambda values: counts_metrics(*map(int,values[ids].sum(0)))['f1']
        x,y=f1(a),f1(b)
        if x is not None and y is not None: differences.append(y-x)
    return dict(scope='Matched physical-group heldout; validation-selected epoch and threshold; one training seed.',
                models=results, test_physical_groups=len(keys),
                preview_minus_oracle_f1=results[1]['test']['f1']-results[0]['test']['f1'],
                paired_group_bootstrap_95_percentile=np.quantile(differences,[.025,.975]).tolist() if differences else None,
                limits=['Bootstrap describes variation among these physical scenarios, not unseen building layouts or training-seed uncertainty.',
                        'Both models receive oracle start states/topology; only interval force/contact inputs differ.',
                        'An offline score does not prove accurate state integration, runtime feature parity, or real-time speed.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','oracle-weights','preview-weights','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=compare(a.dataset,a.oracle_weights,a.preview_weights)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='models'}))


if __name__ == '__main__':main()
