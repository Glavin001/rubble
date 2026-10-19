"""Audit cadence experiments without claiming identical oracle trajectories.

A shorter frame may change substep rounding and fracture/contact ordering. Compare
actual final motion and break sets; never assume an unchanged numerical solution.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .dataset import list_events
from .schema import load_event


def first_breaks(ev):
    times=np.full(ev.manifest['n_bonds'],np.nan)
    dt=ev.manifest['dt']
    for i,f in enumerate(ev.frames):
        labels=f['bond_fail_class']
        take=np.isnan(times)&(labels>0)
        times[take]=(i+(labels[take]-.5)/ev.manifest['K'])*dt
    return times


def preview_errors(ev):
    squared=np.zeros((2,2),dtype=np.float64)
    for f in ev.frames:
        actual=f['chunk_load_bins'].astype(np.float64)
        preview=f['chunk_preview_load_bins'].astype(np.float64)
        for g,sl in enumerate((slice(0,3),slice(3,6))):
            squared[g] += [np.square(preview[...,sl]-actual[...,sl]).sum(),np.square(actual[...,sl]).sum()]
    # Duration weighting allows pooling clips with different sample periods.
    return squared * ev.manifest['dt']/ev.manifest['K']


def audit(coarse,fine):
    originals={load_event(p).context['generator']['split_group']:p for p in list_events(coarse)}
    totals=np.zeros((2,2,2),dtype=np.float64);rows=[]
    for path in list_events(fine):
        b=load_event(path);key=b.context['generator']['split_group']
        original=originals[key];a=load_event(original)
        ta=a.manifest['n_frames']*a.manifest['dt'];tb=b.manifest['n_frames']*b.manifest['dt']
        factor=a.manifest['dt']/b.manifest['dt']
        if not np.isclose(ta,tb,rtol=1e-12) or not np.isclose(factor,round(factor)) or factor<1:
            raise ValueError(f'{key}: mismatched durations or noninteger subdivision')
        if b.manifest['n_frames'] != a.manifest['n_frames']*round(factor):
            raise ValueError(f'{key}: incomplete finer recording')
        for name,x in a.static.items():
            if not np.array_equal(x,b.static[name]):raise ValueError(f'{key}: static graph/property changed: {name}')
        scenes=[json.loads((p/'scene.json').read_text()) for p in (original,path)]
        for scene in scenes:
            for k in ('name','oracle'):scene.pop(k,None)
            for k in ('duration','frame_dt'):scene['sim'].pop(k,None)
        if scenes[0]!=scenes[1]:raise ValueError(f'{key}: physics recipe changed beyond cadence')
        at,bt=first_breaks(a),first_breaks(b)
        ab,bb=np.isfinite(at),np.isfinite(bt);both=ab&bb
        errors=np.stack([preview_errors(a),preview_errors(b)]);totals+=errors
        replays=[json.loads((p/'replay.json').read_text())['frames'] for p in (original,path)]
        # Initial physical state must be identical, independent of later bifurcations.
        if replays[0][0]!=replays[1][0]:raise ValueError(f'{key}: initial oracle state changed')
        final_a,final_b=(r[-1] for r in replays)
        row=dict(recipe=key,coarse=original.name,fine=path.name,duration_seconds=ta,factor=round(factor),
                 oracle_breaks=[int(ab.sum()),int(bb.sum())],
                 final_broken_set_jaccard=float(both.sum()/max(1,(ab|bb).sum())),
                 common_break_time_mae_s=float(np.abs(at[both]-bt[both]).mean()) if both.any() else None,
                 final_position_rmse_m=float(np.sqrt(np.square(np.array(final_a['centers'])-np.array(final_b['centers'])).mean())),
                 final_velocity_rmse_mps=float(np.sqrt(np.square(np.array(final_a['velocities'])-np.array(final_b['velocities'])).mean())),
                 preview_force_relative_rms=[float(np.sqrt(e[0,0]/max(e[0,1],1e-30))) for e in errors],
                 preview_torque_relative_rms=[float(np.sqrt(e[1,0]/max(e[1,1],1e-30))) for e in errors])
        rows.append(row)
    return dict(events=len(rows),coarse_fine_force_relative_rms=[float(np.sqrt(e[0,0]/max(e[0,1],1e-30))) for e in totals],
                coarse_fine_torque_relative_rms=[float(np.sqrt(e[1,0]/max(e[1,1],1e-30))) for e in totals],rows=rows,
                limits=['Cadence changes can change the oracle fracture trajectory; final-motion/break differences are measured, not assumed zero.',
                        'Force errors compare each cadence with its own oracle bin averages; temporal smoothing differs.',
                        'More frequent oracle-state refresh is not a closed-loop learned rollout and costs more inference calls.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('coarse','fine','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=audit(a.coarse,a.fine)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'}))


if __name__=='__main__':main()
