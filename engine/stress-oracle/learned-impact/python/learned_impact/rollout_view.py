"""Compare two independently evolved trajectories after inference has finished."""
import json
from pathlib import Path
import numpy as np
from .visualize import counts_metrics
from .joint_history import load_joint_history
from .runtime_snapshot import NativeGraph
from .interaction_metrics import compare_interactions


def first_breaks(flags,dt):
    connected=np.asarray(flags,dtype=bool)
    if connected.ndim!=2 or len(connected)<2:raise ValueError('need initial and endpoint connectivity')
    if np.any(connected[1:]&~connected[:-1]):raise ValueError('rollout healed a disconnected bond')
    changes=connected[:-1]&~connected[1:]
    return np.where(changes.any(0),(changes.argmax(0)+1)*dt,np.inf)

def timing_scores(oracle,model,tolerances):
    oracle=np.asarray(oracle);model=np.asarray(model);actual=np.isfinite(oracle);predicted=np.isfinite(model)
    result=[]
    for tolerance in tolerances:
        both=actual&predicted
        match=np.zeros_like(both);match[both]=np.abs(oracle[both]-model[both])<=tolerance+1e-12
        tp=int(match.sum());fp=int(predicted.sum())-tp;fn=int(actual.sum())-tp
        result.append({'tolerance_seconds':tolerance,**counts_metrics(tp,fp,fn)})
    return result

def fragments(frame,mass):
    active=np.asarray(frame['active'],dtype=bool);ids=np.asarray(frame['clusters'])[active]
    _,groups=np.unique(ids,return_inverse=True);m=np.bincount(groups,weights=mass[active])
    return {'count':len(m),'largest_mass_fraction':float(m.max()/m.sum()) if len(m) else None}

def build(root,source):
    root=Path(root);source=Path(source);plan=json.loads((root/'plan.json').read_text())
    if plan['status']!='completed':raise ValueError('only completed rollouts may be published')
    native=root/'native';graph=json.loads((native/'graph.json').read_text());n=plan['frames'];dt=graph['dt']
    snapshots=[json.loads((native/f'snapshots/{i:06}.json').read_text()) for i in range(n+1)]
    runtime=NativeGraph(graph)
    for snapshot in snapshots:
        runtime.batch(snapshot,include_projectiles=True)
        if not np.isfinite([snapshot['world'][key] for key in ('time','mechanical_energy','dissipated_energy')]).all():
            raise ValueError('nonfinite independent trajectory diagnostics')
    recorded=json.loads((source/'replay.json').read_text());oracle=recorded['frames'][:n+1]
    if len(oracle)!=n+1 or oracle[0]!=snapshots[0]['world']:raise ValueError('independent worlds did not start identically')
    if not np.allclose([f['time'] for f in oracle],[s['world']['time'] for s in snapshots],atol=1e-9,rtol=0):raise ValueError('trajectory time mismatch')
    context=json.loads((source/'context.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    if graph['bond_keys']!=context['bond_keys'] or graph['chunk_keys']!=context['chunk_keys']:raise ValueError('graph identity mismatch')
    joint=load_joint_history(source,context,manifest)
    truth_flags=joint.flags[:n+1,:,3];pred_flags=np.asarray([s['joint_flags'] for s in snapshots])[:,:,3]
    truth_times=first_breaks(truth_flags,dt);pred_times=first_breaks(pred_flags,dt)
    predicted=[s['world'] for s in snapshots]
    masses=np.asarray(graph['statics']['chunk_mass']);reports=json.loads((native/'reports.json').read_text())
    if len(reports)!=n or any(r['report']['analytic_bond_evaluations'] for r in reports):raise ValueError('invalid independent execution evidence')
    integration=plan.get('integration','experimental-endpoint/1')
    if integration not in ('experimental-endpoint/1','native-stress-phase-trajectory/1'):raise ValueError('unsupported rollout integration')
    step_key='stress_substeps' if integration=='native-stress-phase-trajectory/1' else 'contact_substeps'
    def augment(frames,flags,util):
        return [{**f,'new_breaks':np.flatnonzero(flags[i-1]&~flags[i]).tolist() if i else [],'utilization':np.asarray(util[i]).tolist()} for i,f in enumerate(frames)]
    truth_util=joint.values[:n+1,:,20];pred_util=np.asarray([s['joint_values'] for s in snapshots],dtype=object)[:,:,20].astype(float)
    pos=np.asarray(predicted[-1]['centers'])-np.asarray(oracle[-1]['centers']);active=np.asarray(predicted[-1]['active'],dtype=bool)&np.asarray(oracle[-1]['active'],dtype=bool)
    metrics={'timing':timing_scores(truth_times,pred_times,[0.,dt,2*dt,5*dt]),
       'interactions':compare_interactions(oracle,predicted),
       'eventual_breaks':counts_metrics(int((np.isfinite(truth_times)&np.isfinite(pred_times)).sum()),int((~np.isfinite(truth_times)&np.isfinite(pred_times)).sum()),int((np.isfinite(truth_times)&~np.isfinite(pred_times)).sum())),
       'oracle_fragments':fragments(oracle[-1],masses),'model_fragments':fragments(predicted[-1],masses),
       'final_position_rmse_m':float(np.sqrt(np.square(pos[active]).mean())),
       'oracle_initial_mechanical_energy_j':oracle[0]['mechanical_energy'],
       'oracle_final_mechanical_energy_j':oracle[-1]['mechanical_energy'],
       'model_final_mechanical_energy_j':predicted[-1]['mechanical_energy'],
       'model_max_mechanical_energy_j':max(f['mechanical_energy'] for f in predicted),
       'analytic_bond_evaluations':sum(r['report']['analytic_bond_evaluations'] for r in reports),
       'contact_substeps':sum(r['report'][step_key] for r in reports)}
    return {'format':'independent-endpoint-comparison/1','name':manifest['event_id'],'dt':dt,'count':n,
       'integration':integration,
       'checkpoint':plan['checkpoint'],'scope':plan['scope'],'metrics':metrics,'reports':reports,
       'half_extents':np.asarray(graph['statics']['chunk_half_extents']).reshape(-1,3).tolist(),
       'bonds':np.asarray(graph['statics']['bond_endpoints']).reshape(-1,2).tolist(),
       'oracle':augment(oracle,truth_flags.astype(bool),truth_util),
       'model':augment(predicted,pred_flags.astype(bool),pred_util),
       'ground':recorded.get('ground')}
