"""Matched-recipe comparison at a common evaluation window, with runtime costs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .train_physics import load_records,evaluate
from .visualize import load_checkpoint,counts_metrics


def compare(root):
    torch.set_num_threads(4)
    models=[];counts=[];memberships=[]
    for name in ('coarse','fine'):
        weights=root/f'{name}-model';model,meta=load_checkpoint(weights)
        run=meta['run'];records,split=load_records(root/f'{name}-dataset',model,split_from=run['split_from'])
        saved=json.loads((weights/'split.json').read_text())
        if saved['events']!=split['events']:raise ValueError('saved split membership changed')
        parts={split['recipe_groups'][event]:part for event,part in split['events'].items()}
        memberships.append(parts)
        by_event=defaultdict(list)
        for row in records['test']:by_event[row[0]].append(row)
        events={event:evaluate(model,rows,window_factor=run['evaluation_window']) for event,rows in by_event.items()}
        counts.append({split['recipe_groups'][event]:np.array([m['window_metrics'][k] for k in ('tp','fp','fn')]) for event,m in events.items()})
        models.append(dict(checkpoint=meta,test=evaluate(model,records['test'],window_factor=run['evaluation_window']),events=events,
                           optimizer_updates=run['epochs']*split['frames']['train'],
                           selected_update=run['selected_epoch']*split['frames']['train'],
                           validation_checks=run['epochs']//run['eval_every']))
    if memberships[0]!=memberships[1]:raise ValueError('cadences do not use identical recipe membership')
    if models[0]['optimizer_updates']!=models[1]['optimizer_updates']:raise ValueError('unmatched optimizer budget')
    keys=sorted(counts[0])
    if keys!=sorted(counts[1]):raise ValueError('unmatched held-out recipes')
    a,b=(np.array([c[k] for k in keys]) for c in counts)
    rng=np.random.default_rng(20261019);diff=[]
    for _ in range(2000):
        ids=rng.integers(len(keys),size=len(keys))
        x,y=(counts_metrics(*map(int,c[ids].sum(0)))['f1'] for c in (a,b))
        if x is not None and y is not None:diff.append(y-x)
    audit=json.loads((root/'horizon-audit.json').read_text())
    test_names={k.split(':',1)[1] for k in keys}
    oracle_drift=[r for r in audit['rows'] if r['recipe'] in test_names]
    return dict(models=models,test_recipes=len(keys),
                fine_minus_coarse_window_f1=models[1]['test']['window_metrics']['f1']-models[0]['test']['window_metrics']['f1'],
                paired_recipe_bootstrap_95_percentile=np.quantile(diff,[.025,.975]).tolist(),
                test_oracle_drift=oracle_drift,
                scope='Same physical recipes, total simulated duration, optimizer updates and common-window validation selection. Fine cadence receives four oracle-state refreshes per coarse interval.',
                limits=['Cadence changes cause measured oracle-trajectory differences; this is not comparison against identical labels.',
                        'Only six held-out wall recipes and one training seed; no claim about unseen layouts.',
                        'Four finer model calls replace one coarse call; closed-loop accuracy and engine speed are unverified.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=compare(a.root);a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('models','test_oracle_drift')}))


if __name__=='__main__':main()
