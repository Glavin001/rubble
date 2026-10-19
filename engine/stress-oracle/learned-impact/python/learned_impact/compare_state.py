"""Held-out fracture and physical state/history evaluation with matched splits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from .physics import PhysicsImpactNet, JOINT_SCALE, symlog
from .motion_metrics import MotionMetrics
from .train_physics import load_records, evaluate
from .visualize import load_checkpoint,counts_metrics

JOINT_FIELDS = ('damage','crush','kappa','kappa_c','ductility','ductility_c','fatigue',
                'plastic_x','plastic_y','plastic_z','plastic_rx','plastic_ry','plastic_rz',
                'rebar_plastic','rebar_slip_x','rebar_slip_y','rebar_work','rebar_broken',
                'strain_rate','governing_stress','dissipated','utilization')
MONOTONIC = (0,1,2,3,6,16,17,20)
BOUNDED = (0,1,6,17)


@torch.inference_mode()
def history_metrics(model, records):
    squared = torch.zeros(22,dtype=torch.float64); persistence = squared.clone()
    decreasing = torch.zeros(22,dtype=torch.int64); bounds = decreasing.clone()
    motion = MotionMetrics(model.delta_rms.numpy()) if hasattr(model,'delta_rms') else None
    bonds = changed = total_chunks = scored_chunks = 0; damage_abs = torch.zeros(2,dtype=torch.float64)
    changed_error = torch.zeros(2,dtype=torch.float64)
    for _,_,_,b in records:
        total_chunks += len(b['chunk_state'])
        scored_chunks += int(b.get('delta_valid',torch.ones(len(b['chunk_state']),dtype=torch.bool)).sum())
        output = model(b)
        if motion is not None:
            motion.add(b['chunk_state_delta'].numpy(),model.decode_delta(output).numpy(),
                       b.get('delta_valid',torch.ones(len(b['chunk_state']),dtype=torch.bool)).numpy())
        pred = output['bond_joint_state_end']
        if not torch.isfinite(pred).all(): raise ValueError('nonfinite predicted history')
        start = symlog(b['bond_joint_state'],JOINT_SCALE)
        target = symlog(b['bond_joint_state_end'],JOINT_SCALE)
        squared += (pred-target).double().square().sum(0)
        persistence += (start-target).double().square().sum(0)
        decreasing += (pred < start-1e-6).sum(0)
        bounds += ((pred < 0) | (pred > np.log(2)+1e-6)).sum(0)
        # Diagnostic damage/crush MAE in physical [0,1] units; no clipping.
        decoded = pred[:,:2].sign()*torch.expm1(pred[:,:2].abs())
        error = (decoded-b['bond_joint_state_end'][:,:2]).abs()
        if not torch.isfinite(error).all():raise ValueError('nonfinite decoded damage')
        damage_abs += error.double().sum(0)
        active = ((b['bond_joint_state_end'][:,:2]-b['bond_joint_state'][:,:2]).abs()>1e-6).any(-1)
        changed_error += error[active].double().sum(0)
        changed += int(active.sum());bonds += len(target)
    return {'motion_by_target_magnitude':motion.report() if motion is not None else None, 'bond_intervals':bonds,'damage_changed_intervals':changed,
            'motion_scored_chunk_intervals':scored_chunks,
            'motion_fragment_reset_excluded_chunk_intervals':total_chunks-scored_chunks,
            'encoded_rmse':dict(zip(JOINT_FIELDS,(squared/max(1,bonds)).sqrt().tolist())),
            'persistence_encoded_rmse':dict(zip(JOINT_FIELDS,(persistence/max(1,bonds)).sqrt().tolist())),
            'damage_crush_mae':(damage_abs/max(1,bonds)).tolist(),
            'damage_crush_mae_when_either_changes':(changed_error/max(1,changed)).tolist(),
            'decreasing_history_count':{JOINT_FIELDS[i]:int(decreasing[i]) for i in MONOTONIC},
            'outside_unit_interval_count':{JOINT_FIELDS[i]:int(bounds[i]) for i in BOUNDED},
            'units':'History RMSE is in signed-log SI reference units; damage/crush MAE is dimensionless physical damage.'}


def compare(dataset, weights):
    torch.set_num_threads(4)
    models=[];reference=None
    for path in weights:
        model,meta=load_checkpoint(path)
        if not isinstance(model,PhysicsImpactNet):raise ValueError('physical-feature checkpoints required')
        saved=json.loads((path/'split.json').read_text())
        records,split=load_records(dataset,model,meta['run']['split_seed'],split_from=meta['run'].get('split_from'))
        if saved['events']!=split['events']:raise ValueError('saved split differs from loaded dataset')
        membership={k:split[k] for k in ('events','physical_groups','frames','deduplicated_training_frames')}
        if reference is not None and reference!=membership:raise ValueError('checkpoints do not have matched partitions')
        reference=membership
        groups=defaultdict(list)
        for record in records['test']:groups[split['physical_groups'][record[0]]].append(record)
        models.append({'checkpoint':meta,
                       'test':evaluate(model,records['test']),
                       'test_groups':{key:evaluate(model,rows) for key,rows in groups.items()},
                       'test_history':history_metrics(model,records['test']),
                       'validation':evaluate(model,records['validation']),
                       'validation_history':history_metrics(model,records['validation'])})
    paired=[]
    for model in models[1:]:
        keys=sorted(models[0]['test_groups'])
        if keys!=sorted(model['test_groups']):raise ValueError('different physical test groups')
        a,b=(np.array([[m['test_groups'][key][k] for k in ('tp','fp','fn')] for key in keys]) for m in (models[0],model))
        rng=np.random.default_rng(20261019);differences=[]
        for _ in range(2000):
            ids=rng.integers(len(keys),size=len(keys));x=counts_metrics(*map(int,a[ids].sum(0)))['f1'];y=counts_metrics(*map(int,b[ids].sum(0)))['f1']
            if x is not None and y is not None:differences.append(y-x)
        paired.append({'baseline':models[0]['checkpoint']['name'],'candidate':model['checkpoint']['name'],
                       'test_groups':len(keys),'f1_difference':model['test']['f1']-models[0]['test']['f1'],
                       'paired_group_bootstrap_95_percentile':np.quantile(differences,[.025,.975]).tolist() if differences else None})
    return {'models':models,'paired_comparisons':paired,'scope':'Matched physical scenarios. Validation selection policy is recorded per checkpoint; state/history errors reported separately.',
            'limits':['Oracle start states and topology reset every interval; no recurrent rollout.',
                      'Irreversible history constraints do not enforce energy balance or full constitutive consistency.',
                      'The packed joint schema cannot represent all solver state, including unset/infinite ductility.',
                      'Group bootstrap describes these scenarios, not training-seed uncertainty or unseen structural layouts.',
                      'No candidate is a replacement integrator based on one-step scores alone.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--weights',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=compare(a.dataset,a.weights)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for m in result['models']:
        print(json.dumps({'name':m['checkpoint']['name'],'f1':m['test']['f1'],
                          'state_rmse':m['test']['state_rmse'],'zero':m['test']['zero_increment_rmse'],
                          'damage_crush_mae':m['test_history']['damage_crush_mae'],
                          'decreasing_history':m['test_history']['decreasing_history_count']}))


if __name__=='__main__':main()
