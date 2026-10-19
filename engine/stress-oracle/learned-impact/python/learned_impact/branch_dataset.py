"""Audit corrective oracle labels from a training scenario's learned states.

This does not admit validation/test branches into training, change the current
training corpus, or claim that an oracle-assisted branch is learned inference.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .stress_phases import PhaseHistory

def audit_branch(root,parent,source_root,event):
    root,parent,source_root=map(Path,(root,parent,source_root))
    registry=json.loads((source_root/'pilot-manifest.json').read_text())
    matches=[x for x in registry['events'] if x['event']==event]
    if registry['status']!='completed_verified' or len(matches)!=1 or matches[0]['partition']!='train':
        raise ValueError('corrective training branch requires an unambiguous parent training partition')
    source=source_root/'dataset'/event;quality=json.loads((source/'audit.json').read_text())
    if quality['status']!='completed_verified' or not quality['original_world_replay_exact']:
        raise ValueError('parent oracle scenario lacks original replay audit')
    for name,digest in quality['files_sha256'].items():
        if PhaseHistory._hash(source/name)!=digest:raise ValueError('original scenario changed after audit')
    plan=json.loads((parent/'plan.json').read_text());native=parent/'native'
    if plan['status']!='completed' or plan['integration']!='native-stress-phase-trajectory/1':
        raise ValueError('parent learned rollout is incomplete or unsupported')
    for name in ['graph.json','snapshots/000000.json']:
        if (native/name).read_bytes()!=(source/name).read_bytes():
            raise ValueError('parent learned run does not start from registered training scenario')
    reports=json.loads((native/'reports.json').read_text())
    if len(reports)!=plan['frames'] or any(r['report']['analytic_bond_evaluations']!=0 for r in reports):
        raise ValueError('parent has incomplete execution or analytical fallback')
    data=PhaseHistory(root);report=data.report;prefix=report['branch_from_frame']
    if (type(prefix)!=int or not 0<prefix<plan['frames'] or report['learned_continuation_exact'] is not True
            or report['replayed_snapshots_exact']!=prefix+1 or data.frames!=1
            or Path(report['source_native']).resolve()!=native.resolve()):
        raise ValueError('invalid learned-state branch provenance')
    for target,original in [('graph.json','graph.json'),('snapshots/000000.json',f'snapshots/{prefix:06}.json'),
                            ('learned-control.json',f'snapshots/{prefix+1:06}.json')]:
        if (root/target).read_bytes()!=(native/original).read_bytes():
            raise ValueError('branch differs from its verified learned parent')
    audit=data.audit()
    start_time=data.snapshot(0)['world']['time']
    if report['branch_time']!=start_time or abs(start_time-(data.graph['initial_time']+prefix*data.graph['dt']))>1e-8*data.graph['dt']:
        raise ValueError('branch frame/absolute clock mismatch')
    kin=json.loads((root/'boundary-kinematics.json').read_text());sl=data.phase_rows(0);ids=data.slots[sl]
    if kin['format']!='native-boundary-kinematics/1' or kin['time']!=data.snapshot(0)['world']['time']:
        raise ValueError('branch boundary kinematics clock mismatch')
    x=np.asarray(kin['values'],dtype='<f8')
    if x.shape!=(len(data.meta['bond_keys']),12) or not np.isfinite(x).all() or not np.array_equal(x[ids].view('u8'),np.asarray(data.inputs[sl,:12]).copy().view('u8')):
        raise ValueError('branch boundary kinematics differ from native first phase')
    audit.update(source_kind='learned_state_oracle_branch',partition='train',parent_event=event,
        parent_checkpoint_sha256=plan['checkpoint']['sha256'],parent_prefix_frames=prefix,
        source_oracle_audit_sha256=PhaseHistory._hash(source/'audit.json'),
        parent_files_sha256={str(p.relative_to(parent)):PhaseHistory._hash(p) for p in
            [parent/'plan.json',native/'graph.json',native/'reports.json']+
            [native/f'snapshots/{i:06}.json' for i in range(prefix+2)]+
            [native/f'predictions/{i:06}.json' for i in range(prefix+1)]},
        boundary_kinematics_bit_exact=True,
        scope='One full native oracle continuation from a verified model-generated training state. Original-world replay is not claimed for the branch. Targets are corrective supervision; future phase values never become boundary input features. Separate corpus, not yet used by the matched V33 training experiment.')
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','parent','source-root','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--event',required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=audit_branch(a.root,a.parent,a.source_root,a.event)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def load_records(manifest_path,source_root,knots):
    """Explicit training-only admission with fresh provenance and label audits."""
    import torch
    from .trajectory_force import phase_record
    from .trajectory_response import complete_record
    manifest_path,source_root=map(Path,(manifest_path,source_root))
    manifest=json.loads(manifest_path.read_text())
    if (manifest['status']!='completed_verified' or manifest['source_kind']!='learned_state_oracle_branch'
            or manifest['source_registry_sha256']!=PhaseHistory._hash(source_root/'pilot-manifest.json')):
        raise ValueError('unverified corrective corpus or changed source registry')
    records=[];seen=set();sources={}
    for entry in manifest['entries']:
        identity=(entry['event'],entry['prefix'])
        if entry['partition']!='train' or identity in seen:raise ValueError('held-out or duplicated corrective branch')
        seen.add(identity);root=Path(entry['branch']);parent=Path(entry['parent'])
        if PhaseHistory._hash(root/'audit.json')!=entry['audit_sha256']:raise ValueError('branch audit changed')
        stored=json.loads((root/'audit.json').read_text())
        fresh=audit_branch(root,parent,source_root,entry['event'])
        if fresh!=stored or fresh['parent_prefix_frames']!=entry['prefix']:
            raise ValueError('corrective branch changed after audit')
        data=PhaseHistory(root);key=f"{entry['event']}@learned-frame-{entry['prefix']}"
        r=phase_record(data,0,key,'train',knots);r['parent_event']=entry['event'];r['source_kind']='learned_state_oracle_branch'
        kin=json.loads((root/'boundary-kinematics.json').read_text());ids=r['batch']['bond_ids'].numpy()
        x=np.asarray(kin['values'],dtype=np.float64)[ids].astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('corrective boundary features exceed float32 range')
        r['batch']['bond_boundary_kinematics']=torch.from_numpy(x)
        complete_record(r,data,knots);records.append(r);sources[key]=entry['audit_sha256']
    if not records:raise ValueError('empty corrective corpus')
    return records,{'manifest_sha256':PhaseHistory._hash(manifest_path),'branches':sources}

if __name__=='__main__':main()
