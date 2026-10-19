"""Audit fracture decisions against the full decoded JointState connectivity.

Mirrors JointState::connected: damage < 1 OR an unbroken reinforcing tie.
The explicit source rebar presence and resolved feature flag define that tie;
zero/nonzero parameter heuristics are deliberately avoided. This is an audit,
not an engine state application or a correction of model predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import list_events,event_batches
from .schema import load_event
from .visualize import load_checkpoint,counts_metrics
from .history_model import HistoryCarryImpactNet
from .experiment import event_digest


def joint_connected(damage,rebar_broken,has_rebar):
    damage=np.asarray(damage);broken=np.asarray(rebar_broken);present=np.asarray(has_rebar)
    if damage.shape!=broken.shape or damage.shape!=present.shape:
        raise ValueError('joint connectivity shape mismatch')
    if not np.isfinite(damage).all() or ((damage<0)|(damage>1)).any():
        raise ValueError('damage outside admissible interval')
    if broken.dtype!=bool or present.dtype!=bool:raise ValueError('explicit rebar booleans required')
    return (damage<1)|(present&~broken)


def decision_counts(truth,head,state):
    return {'eligible_bonds':len(truth),'truth_breaks':int(truth.sum()),
            'head_breaks':int(head.sum()),'state_breaks':int(state.sum()),
            'head_break_state_connected':int((head&~state).sum()),
            'head_intact_state_disconnected':int((~head&state).sum()),
            **{f'{name}_{key}':int(value) for name,pred in [('head',head),('state',state)]
               for key,value in [('tp',(pred&truth).sum()),('fp',(pred&~truth).sum()),('fn',(~pred&truth).sum())]}}


@torch.inference_mode()
def audit(weights,dataset,partition='test'):
    torch.set_num_threads(1)
    model,meta=load_checkpoint(weights)
    if not isinstance(model,HistoryCarryImpactNet):raise ValueError('full JointState decoder required')
    split=json.loads((weights/'split.json').read_text())['events'];rows=[];sources={}
    for path in list_events(dataset):
        if split.get(path.name)!=partition:continue
        ev=load_event(path,world_inputs=True,world_targets=True,forcing_inputs=True,
                      exact_joint_history=True,material_inputs=True,
                      projectile_inputs=getattr(model,'requires_projectile_inputs',False),bond_response_history=getattr(model,'requires_bond_response_history',False))
        scene=json.loads((path/'scene.json').read_text())
        enabled=scene['sim']['features']['rebar']
        if type(enabled) is not bool:raise ValueError('explicit resolved rebar feature required')
        present=np.array([scene['bodies'][s]['bonds'][b]['rebar'] is not None and enabled
                          for s,b in ev.context['bond_keys']],dtype=bool)
        sources[path.name]=event_digest(path)
        for i,raw in enumerate(event_batches(ev,bond_policy=model.bond_policy)):
            ids=raw['bond_ids'].numpy();eligible=raw['bond_failure_eligible'].numpy()
            start=ev.exact_joint_history.values[i,ids];start_flags=ev.exact_joint_history.flags[i,ids]
            end=ev.exact_joint_history.values[i+1,ids];end_flags=ev.exact_joint_history.flags[i+1,ids]
            start_connected=joint_connected(start[:,0],start_flags[:,0].astype(bool),present[ids])
            end_connected=joint_connected(end[:,0],end_flags[:,0].astype(bool),present[ids])
            if not np.array_equal(start_connected,eligible):raise ValueError('source rebar/start connectivity mismatch')
            if not np.array_equal(end_connected&(end_flags[:,2]==1),end_flags[:,3]==1):
                raise ValueError('source rebar/target connectivity mismatch')
            truth=raw['bond_fail_class'].numpy()[eligible]>0
            if not np.array_equal(truth,~end_connected[eligible]):
                raise ValueError('first-fracture labels disagree with oracle joint connectivity')
            batch=model.prepare_batch(raw);pred=model(batch);decoded=model.decode_joint_state(pred,batch)
            connected=joint_connected(decoded['values'][:,0],decoded['rebar_broken'],present[ids])
            head=model.predict_classes(pred).numpy()[eligible]>0
            rows.append({'event':path.name,'interval':i,**decision_counts(truth,head,~connected[eligible])})
    if not rows:raise ValueError('no matching records')
    totals={key:sum(row[key] for row in rows) for key in rows[0] if key not in ('event','interval')}
    for name in ('head','state'):totals[name+'_metrics']=counts_metrics(*(totals[name+'_'+k] for k in ('tp','fp','fn')))
    return {'checkpoint':meta,'checkpoint_sha256':hashlib.sha256((weights/'state_dict.pt').read_bytes()).hexdigest(),
            'partition':partition,'source_sha256':sources,'totals':totals,'rows':rows,
            'scope':'Independent heads on oracle-start intervals. Exact engine JointState connectivity, with explicit source rebar presence. No thresholds added to damage and no state corrections. Not a closed-loop or RtBond/World adapter.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','dataset','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--partition',choices=['train','validation','test'],default='test')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=audit(a.weights,a.dataset,a.partition)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['totals']))


if __name__=='__main__':main()
