import json,shutil
import pytest
from learned_impact.branch_dataset import audit_branch,load_records
from learned_impact.stress_phases import PhaseHistory
from test_stress_phases import fixture

def branch_fixture(root):
    fixture(root)
    p=root/'graph.json';g=json.loads(p.read_text());g['scene']['impactors']=[];p.write_text(json.dumps(g))
    for p in (root/'snapshots').glob('*.json'):
        s=json.loads(p.read_text());s['world']['impactors']=[];p.write_text(json.dumps(s))
    return PhaseHistory(root)

def setup(root):
    source=root/'source';event='fixture-train';original=source/'dataset'/event;original.mkdir(parents=True)
    truth=branch_fixture(original);quality=truth.audit();quality['original_world_replay_exact']=True
    (original/'audit.json').write_text(json.dumps(quality))
    registry={'status':'completed_verified','events':[{'event':event,'partition':'train'}]}
    (source/'pilot-manifest.json').write_text(json.dumps(registry))
    branch=root/'branch';branch.mkdir();data=branch_fixture(branch);dt=data.graph['dt']
    for i in [0,1]:
        p=branch/f'snapshots/{i:06}.json';s=json.loads(p.read_text());s['world']['time']=(i+1)*dt;p.write_text(json.dumps(s))
    p=branch/'stress_phases/manifest.json';m=json.loads(p.read_text());m['phases'][0]['time']=dt;p.write_text(json.dumps(m))
    parent=root/'parent';native=parent/'native';(native/'snapshots').mkdir(parents=True);(native/'predictions').mkdir()
    shutil.copy2(original/'graph.json',native/'graph.json')
    shutil.copy2(original/'snapshots/000000.json',native/'snapshots/000000.json')
    for i in [0,1]:shutil.copy2(branch/f'snapshots/{i:06}.json',native/f'snapshots/{i+1:06}.json')
    for i in [0,1]:(native/f'predictions/{i:06}.json').write_text('{}')
    shutil.copy2(native/'snapshots/000002.json',branch/'learned-control.json')
    (native/'reports.json').write_text(json.dumps([{'report':{'analytic_bond_evaluations':0}}]*2))
    (parent/'plan.json').write_text(json.dumps({'status':'completed','integration':'native-stress-phase-trajectory/1','frames':2,'checkpoint':{'sha256':'fixture'}}))
    p=branch/'report.json';r=json.loads(p.read_text());r.update(branch_from_frame=1,branch_time=dt,learned_continuation_exact=True,replayed_snapshots_exact=2,source_native=str(native));p.write_text(json.dumps(r))
    (branch/'boundary-kinematics.json').write_text(json.dumps({'format':'native-boundary-kinematics/1','time':dt,'values':[[0.]*12]*len(data.meta['bond_keys'])}))
    return branch,parent,source,event

def test_corrective_labels_preserve_training_lineage_and_reject_changed_parent(tmp_path):
    args=setup(tmp_path);report=audit_branch(*args)
    assert report['partition']=='train' and report['boundary_kinematics_bit_exact']
    assert not report['original_world_replay_exact']
    p=args[0]/'learned-control.json';s=json.loads(p.read_text());s['world']['time']+=.01;p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='verified learned parent'):audit_branch(*args)

def test_validation_and_test_states_cannot_be_admitted_as_corrective_training(tmp_path):
    args=setup(tmp_path);p=args[2]/'pilot-manifest.json';registry=json.loads(p.read_text())
    for partition in ['validation','test']:
        registry['events'][0]['partition']=partition;p.write_text(json.dumps(registry))
        with pytest.raises(ValueError,match='parent training partition'):audit_branch(*args)

def test_explicit_branch_loader_builds_current_inputs_and_rejects_duplicate_weighting(tmp_path):
    branch,parent,source,event=setup(tmp_path)
    audit=audit_branch(branch,parent,source,event);(branch/'audit.json').write_text(json.dumps(audit))
    entry={'event':event,'partition':'train','prefix':1,'branch':str(branch),'parent':str(parent),'audit_sha256':PhaseHistory._hash(branch/'audit.json')}
    manifest={'status':'completed_verified','source_kind':'learned_state_oracle_branch',
        'source_registry_sha256':PhaseHistory._hash(source/'pilot-manifest.json'),'entries':[entry]}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    rows,provenance=load_records(p,source,3);r=rows[0]
    assert r['partition']=='train' and r['parent_event']==event and r['frame']==0
    assert r['target'].shape==(1,1,42) and r['batch']['trajectory_base'].shape==(1,42)
    assert r['batch']['trajectory_base'][0,0]==0 and r['target'][0,0,0]>0
    assert provenance['branches'][r['event']]==entry['audit_sha256']
    manifest['entries'].append(entry);p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='duplicated'):load_records(p,source,3)
