"""Audit recording completeness and actual supervision, not just recipe counts."""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .dataset import list_events, event_batches
from .schema import load_event
from .experiment import event_digest
from .visualize import load_replay


def coverage(dataset):
    rows, parameters = [], defaultdict(set)
    for path in list_events(dataset):
        ev = load_event(path)
        for key,array in [*ev.static.items(), *((k,v) for f in ev.frames for k,v in f.items())]:
            if not np.isfinite(array).all():
                raise ValueError(f'{path.name}: nonfinite {key}')
        classes = np.zeros(5,dtype=np.int64)
        first = None; last_breaks = 0; loaded = 0; reset_nodes = 0
        for i,b in enumerate(event_batches(ev)):
            labels = b['bond_fail_class'].numpy()
            classes += np.bincount(labels,minlength=5)
            last_breaks = int(np.count_nonzero(labels))
            if last_breaks and first is None: first=i
            loaded += int(np.count_nonzero(np.linalg.norm(b['chunk_load_bins'].numpy().reshape(len(b['chunk_mass']),-1),axis=1)))
            if 'delta_valid' in b: reset_nodes += int((~b['delta_valid']).sum())
        context = ev.context
        native = path/'replay.json'
        if context:
            if context.get('format') != 'learned-impact-context/2':
                raise ValueError(f'{path}: unknown context format')
            for key,shape in [('centers',(ev.manifest['n_chunks'],3)),('rotations',(ev.manifest['n_chunks'],3,3)),('bond_frames',(ev.manifest['n_bonds'],3,3))]:
                a=np.asarray(context[key])
                if a.shape != shape or not np.isfinite(a).all(): raise ValueError(f'{path}: invalid context {key}')
            if not native.exists() or not (path/'scene.json').exists():
                raise ValueError(f'{path}: missing reproduction or motion sidecar')
            replay=json.loads(native.read_text())
            # Raw recorder output has not been fingerprinted by collection yet.
            if 'source_sha256' in replay:
                replay=load_replay(native, {'sha256':event_digest(path)}, ev.manifest)
            if len(replay['frames']) != ev.manifest['n_frames']+1: raise ValueError('missing replay frames')
            for f in replay['frames']:
                for key in ('velocities','angular_velocities'):
                    a=np.asarray(f[key])
                    if a.shape != (ev.manifest['n_chunks'],3) or not np.isfinite(a).all(): raise ValueError(f'{path}: invalid {key}')
                if not np.isfinite([f['mechanical_energy'],f['dissipated_energy']]).all(): raise ValueError('nonfinite energy')
            for k,v in (context.get('generator') or {}).get('parameters',{}).items():
                parameters[k].add(json.dumps(v,sort_keys=True))
        rows.append({'event':path.name, 'family':ev.manifest['structure_type'], 'frames':ev.manifest['n_frames'],
                     'chunks':ev.manifest['n_chunks'], 'dt':ev.manifest['dt'], 'duration':ev.manifest['dt']*ev.manifest['n_frames'],
                     'seed':ev.manifest['seed'], 'class_counts':classes.tolist(), 'new_breaks':int(classes[1:].sum()),
                     'first_break_interval':first, 'breaks_in_last_interval':last_breaks,
                     'loaded_node_intervals':loaded, 'local_coordinate_resets':reset_nodes,
                     'native_complete_recording':bool(context), 'native_motion':native.exists()})
    return {'events':len(rows),'frames':sum(r['frames'] for r in rows),
            'families':dict(Counter(r['family'] for r in rows)),
            'chunks_min_max':[min(r['chunks'] for r in rows),max(r['chunks'] for r in rows)],
            'class_counts':np.array([r['class_counts'] for r in rows]).sum(0).tolist(),
            'events_without_breaks':sum(r['new_breaks']==0 for r in rows),
            'events_still_breaking_at_end':sum(r['breaks_in_last_interval']>0 for r in rows),
            'complete_v2_recordings':sum(r['native_complete_recording'] for r in rows),
            'parameter_values':{k:[json.loads(v) for v in sorted(vs)] for k,vs in sorted(parameters.items())},
            'rows':rows,
            'limits':['Counts are bond-intervals on oracle-live graphs, not independent samples.',
                      'Zero-break cases are necessary negatives; still-breaking clips need longer continuations.',
                      'Legacy records may lack verified motion or feature conventions; those are marked, not invented.',
                      'Interval contact histories require a compatible causal runtime input plan before solver replacement.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=coverage(a.dataset);a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','parameter_values')}))


if __name__=='__main__': main()
