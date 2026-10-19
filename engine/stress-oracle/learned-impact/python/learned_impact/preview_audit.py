"""Verify paired oracle recordings and quantify causal-preview error/cost.

Comparisons are component-wise SI force/torque RMS (never mix their units).
Timing comes from actual recording calls and includes contention between jobs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .dataset import list_events
from .schema import load_event, FRAME_REQUIRED


def audit(dataset, oracle):
    original = {}
    for path in list_events(oracle):
        name = json.loads((path/'manifest.json').read_text())['event_id']
        if name in original:
            raise ValueError(f'ambiguous original event {name}')
        original[name] = path
    rows = []
    sums = np.zeros((2, 2), dtype=np.float64)
    for path in list_events(dataset):
        ev = load_event(path)
        name = ev.manifest['event_id']
        if name not in original:
            raise ValueError(f'missing paired oracle event {name}')
        prior = load_event(original[name])
        if len(prior.frames) != len(ev.frames):
            raise ValueError(f'{name}: frame count changed')
        for key, value in prior.static.items():
            if not np.array_equal(value, ev.static[key]):
                raise ValueError(f'{name}: static {key} changed')
        squared = np.zeros((2, 2), dtype=np.float64)
        contact = np.zeros(3, dtype=np.int64)
        components = 0
        for i, (before, after) in enumerate(zip(prior.frames, ev.frames)):
            for key in FRAME_REQUIRED:
                if not np.array_equal(before[key], after[key]):
                    raise ValueError(f'{name} frame {i}: oracle {key} changed')
            truth = after['chunk_load_bins'].astype(np.float64)
            preview = after['chunk_preview_load_bins'].astype(np.float64)
            for g, sl in enumerate((slice(0,3), slice(3,6))):
                squared[g] += [np.square(preview[...,sl]-truth[...,sl]).sum(), np.square(truth[...,sl]).sum()]
            a, b = after['chunk_contact'] > 0, after['chunk_preview_contact'] > 0
            contact += [np.count_nonzero(a & b),np.count_nonzero(a & ~b),np.count_nonzero(~a & b)]
            components += truth.shape[0] * truth.shape[1] * 3
        # Parse JSON to ignore presentation/collection provenance only.
        a = json.loads((original[name]/'replay.json').read_text())
        b = json.loads((path/'replay.json').read_text())
        if a['frames'] != b['frames']:
            raise ValueError(f'{name}: oracle motion changed')
        timing = json.loads((path/'timing.json').read_text())
        oracle_seconds, preview_seconds = sum(timing['oracle_seconds']), sum(timing['preview_seconds'])
        if len(timing['oracle_seconds']) != len(ev.frames) or len(timing['preview_seconds']) != len(ev.frames):
            raise ValueError(f'{name}: incomplete timing')
        sums += squared
        rows.append(dict(event=name, frames=len(ev.frames), family=ev.manifest['structure_type'],
                         parameters=ev.context.get('generator',{}).get('parameters',{}),
                         oracle_seconds=oracle_seconds, preview_seconds=preview_seconds,
                         preview_to_oracle_time=preview_seconds/max(oracle_seconds,1e-12),
                         force_rmse_N=float(np.sqrt(squared[0,0]/components)),
                         torque_rmse_Nm=float(np.sqrt(squared[1,0]/components)),
                         force_relative_rms=float(np.sqrt(squared[0,0]/max(squared[0,1],1e-30))),
                         torque_relative_rms=float(np.sqrt(squared[1,0]/max(squared[1,1],1e-30))),
                         contacted_caught_missed_extra=contact.tolist()))
    t_oracle = sum(r['oracle_seconds'] for r in rows)
    t_preview = sum(r['preview_seconds'] for r in rows)
    return dict(events=len(rows), frames=sum(r['frames'] for r in rows), oracle_arrays_and_motion_unchanged=True,
                force_relative_rms=float(np.sqrt(sums[0,0]/max(sums[0,1],1e-30))),
                torque_relative_rms=float(np.sqrt(sums[1,0]/max(sums[1,1],1e-30))),
                oracle_seconds=t_oracle, preview_seconds=t_preview,
                preview_to_oracle_time=t_preview/max(t_oracle,1e-12), rows=rows,
                limits=['Preview freezes local deformation; its forces are approximate inputs, never oracle labels.',
                        'Oracle start states and topology are supplied every interval; this is not a closed-loop rollout.',
                        'Timing includes clone/contact work and concurrent-job contention, excludes training and IO; not a deployment benchmark.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); result=audit(a.dataset,a.oracle)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__': main()
