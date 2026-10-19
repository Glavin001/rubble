"""Verified optional f64 bond deformation/rate observations at frame boundaries."""
import json
from pathlib import Path
import numpy as np
import torch
from .stress_phases import PhaseHistory

FORMAT='native-boundary-kinematics/1'

def values(path,time,bonds):
    record=json.loads(Path(path).read_text())
    if record['format']!=FORMAT or record['time']!=time:raise ValueError('boundary kinematics contract/clock mismatch')
    x=np.asarray(record['values'],dtype='<f8')
    if x.shape!=(bonds,12) or not np.isfinite(x).all():raise ValueError('invalid boundary kinematics')
    return x

def audit(phase_root,boundary_root):
    phase=PhaseHistory(phase_root);root=Path(boundary_root)
    report=json.loads((root/'report.json').read_text())
    if report['status']!='completed' or report['frames']!=phase.frames:raise ValueError('incomplete boundary recording')
    if json.loads((root/'graph.json').read_text())!=phase.graph:raise ValueError('boundary graph changed')
    compared=0
    for frame in range(phase.frames+1):
        name=f'{frame:06}.json';snapshot=phase.snapshot(frame)
        if (root/'snapshots'/name).read_bytes()!=(phase.root/'snapshots'/name).read_bytes():
            raise ValueError('boundary capture changed native trajectory or serialization')
        x=values(root/'kinematics'/name,snapshot['world']['time'],len(phase.meta['bond_keys']))
        if frame<phase.frames:
            sl=phase.phase_rows(phase.frame_phases[frame][0]);ids=phase.slots[sl]
            if not np.array_equal(x[ids].view('u8'),np.asarray(phase.inputs[sl,:12]).copy().view('u8')):
                raise ValueError('first phase kinematics differ from pre-contact frame-boundary observations')
            compared+=len(ids)
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='audit.json')
    return {'status':'completed_verified','format':FORMAT,'frames':phase.frames,
        'snapshot_bytes_identical':True,'first_phase_kinematics_bit_identical_rows':compared,
        'phase_audit_sha256':PhaseHistory._hash(phase.root/'audit.json'),
        'files_sha256':{str(p.relative_to(root)):PhaseHistory._hash(p) for p in files},
        'scope':'Current-state capture before native step, exact original snapshots and first-phase kinematics. Future phase arrays used only to audit, never to construct runtime features.'}

def attach(records,root,phase_sources):
    """Attach audited current-frame observations, preserving existing runtime IDs."""
    root=Path(root);verified={}
    for r in records:
        event=r['event'];case=root/event
        if event not in verified:
            q=json.loads((case/'audit.json').read_text())
            if q['status']!='completed_verified' or q['phase_audit_sha256']!=phase_sources[event]:
                raise ValueError('unverified boundary source association')
            for name,digest in q['files_sha256'].items():
                if PhaseHistory._hash(case/name)!=digest:raise ValueError('boundary input changed after audit')
            verified[event]=PhaseHistory._hash(case/'audit.json')
        snapshot=json.loads((case/'snapshots'/f"{r['frame']:06}.json").read_text())
        x=values(case/'kinematics'/f"{r['frame']:06}.json",snapshot['world']['time'],len(snapshot['joint_flags']))
        ids=r['batch']['bond_ids'].numpy()
        selected=x[ids].astype(np.float32)
        if not np.isfinite(selected).all():raise ValueError('boundary kinematics exceed float32 range')
        r['batch']['bond_boundary_kinematics']=torch.from_numpy(selected)
    return verified
