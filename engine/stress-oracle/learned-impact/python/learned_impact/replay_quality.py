"""Read-only physical-scale diagnostics for recorded oracle trajectories.

Finite values and exact replay identity do not establish physical validity. This
report exposes energy growth and motion extremes without clipping labels,
removing examples, or silently changing a training partition.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def inspect_replay(replay, manifest):
    n=int(manifest['n_chunks']);count=int(manifest['n_frames']);dt=float(manifest['dt'])
    frames=replay['frames']
    if replay.get('format')!='learned-impact-replay/1' or len(frames)!=count+1 or n<=0 or not np.isfinite(dt) or dt<=0:
        raise ValueError('invalid replay quality contract')
    row={key:0. for key in ('max_speed_mps','max_spin_radps','max_position_residual_m','max_velocity_change_mps','max_spin_change_radps','position_residual_squared')}
    row['valid_chunk_intervals']=0;previous=None;energies=[];dissipation=[]
    t0=float(frames[0]['time'])
    for i,frame in enumerate(frames):
        if not np.isclose(float(frame['time'])-t0,i*dt,rtol=1e-6,atol=1e-8):raise ValueError('quality replay timestamp mismatch')
        values=[np.asarray(frame[key],dtype=np.float64) for key in ('centers','velocities','angular_velocities')]
        active=np.asarray(frame['active'])
        if any(a.shape!=(n,3) or not np.isfinite(a).all() for a in values) or active.shape!=(n,) or not np.isin(active,[0,1]).all():
            raise ValueError('invalid quality replay world arrays')
        energy=float(frame['mechanical_energy']);lost=float(frame['dissipated_energy'])
        if not np.isfinite([energy,lost]).all():raise ValueError('nonfinite replay energy')
        energies.append(energy);dissipation.append(lost)
        x,v,w=values;active=active.astype(bool)
        for key,a in [('max_speed_mps',v),('max_spin_radps',w)]:
            if active.any():row[key]=max(row[key],float(np.linalg.norm(a[active],axis=1).max()))
        if previous is not None:
            px,pv,pw,pa=previous;valid=active&pa
            if valid.any():
                residual=x[valid]-px[valid]-dt*pv[valid]
                row['position_residual_squared']+=float(np.square(residual).sum());row['valid_chunk_intervals']+=int(valid.sum())
                for key,a in [('max_position_residual_m',residual),('max_velocity_change_mps',v[valid]-pv[valid]),('max_spin_change_radps',w[valid]-pw[valid])]:
                    magnitude=np.linalg.norm(a,axis=1);j=int(magnitude.argmax())
                    if magnitude[j]>row[key]:
                        row[key]=float(magnitude[j]);row[key+'_at']={'interval':i-1,'chunk_index':int(np.flatnonzero(valid)[j])}
        previous=x,v,w,active
    row.update(initial_mechanical_energy_j=energies[0],max_mechanical_energy_j=max(energies),min_dissipated_energy_j=min(dissipation),
               max_mechanical_energy_increase_j=max(energies)-energies[0],max_energy_at_snapshot=int(np.argmax(energies)))
    # Defined for a positive initial mechanical-energy reference only. A zero or
    # signed reference cannot support an energy ratio and is not guessed.
    row['max_energy_to_initial_ratio']=max(energies)/energies[0] if energies[0]>0 else None
    count=row['valid_chunk_intervals']
    row['constant_velocity_position_rmse_m']=float(np.sqrt(row['position_residual_squared']/(3*count))) if count else None
    return row


def audit_dataset(dataset, split=None):
    root=Path(dataset);raw=(root/'manifest.json').read_bytes();manifest=json.loads(raw)
    membership=json.loads(Path(split).read_text())['events'] if split else None
    rows=[]
    for entry in manifest['events']:
        path=root/entry['path'];replay=(path/'replay.json').read_bytes();scene=(path/'scene.json').read_bytes()
        event_manifest=json.loads((path/'manifest.json').read_text());source=json.loads(scene)
        event=event_manifest['event_id']
        if event!=entry['event_id']:raise ValueError('quality manifest event mismatch')
        row=inspect_replay(json.loads(replay),event_manifest)
        row.update(event=event,partition=membership[event] if membership is not None else None,
                   scene_sha256=hashlib.sha256(scene).hexdigest(),replay_sha256=hashlib.sha256(replay).hexdigest(),
                   prescribed_loads=len(source['loads']),scheduled_events=len(source['events']))
        rows.append(row)
    if not rows:raise ValueError('quality audit requires at least one recorded event')
    partitions={}
    for part in sorted({r['partition'] or 'unspecified' for r in rows}):
        subset=[r for r in rows if (r['partition'] or 'unspecified')==part]
        total=sum(r['position_residual_squared'] for r in subset);count=sum(r['valid_chunk_intervals'] for r in subset)
        partitions[part]={'events':len(subset),'constant_velocity_position_rmse_m':float(np.sqrt(total/(3*count))) if count else None,
                          'largest_event_squared_error_fraction':max(r['position_residual_squared'] for r in subset)/total if total else None}
    return {'format':'oracle-replay-quality/1','events':len(rows),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'scope':'Diagnostic only: no exclusions or label changes. Energy growth requires interpreting external work, constraints and solver bookkeeping; a small value does not prove physical correctness.',
            'partitions':partitions,'rows':sorted(rows,key=lambda r:r['max_position_residual_m'],reverse=True)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--split',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=audit_dataset(a.dataset,a.split)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'events':report['events'],'partitions':report['partitions'],'largest_motion_event':report['rows'][0]}))

if __name__=='__main__':main()
