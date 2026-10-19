import copy,json
import numpy as np
import pytest
import torch
from learned_impact.stress_phases import PhaseHistory,FORMAT
from learned_impact.joint_history import FIELDS as JOINT_FIELDS
from learned_impact.response_history import FIELDS as RESPONSE_FIELDS
from test_runtime_snapshot import fixture as native_fixture

def fixture(root):
    graph,start=native_fixture();end=copy.deepcopy(start);end['world']['time']=graph['dt']
    end['joint_values'][0][0]=1.;end['joint_flags'][0]=[0,1,0,0]
    (root/'snapshots').mkdir();(root/'stress_phases').mkdir()
    (root/'graph.json').write_text(json.dumps(graph))
    for i,s in enumerate([start,end]):(root/'snapshots'/f'{i:06}.json').write_text(json.dumps(s))
    x=np.zeros((1,51));x[0,16:18]=np.inf;y=np.zeros((1,42));y[0,0]=1.;y[0,4:6]=np.inf
    arrays={'inputs.f64':x.astype('<f8'),'targets.f64':y.astype('<f8'),'bond_slots.u32':np.array([0],dtype='<u4'),
        'input_flags.i32':np.array([[0,0,1,1]],dtype='<i4'),'target_flags.i32':np.array([[0,1,0,1]],dtype='<i4'),
        'chunks.f64':np.zeros((1,2,18),dtype='<f8'),'chunk_flags.u64':np.array([[[1,0],[1,0]]],dtype='<u8')}
    for name,a in arrays.items():a.tofile(root/'stress_phases'/name)
    meta={'format':FORMAT,'status':'completed','rows':1,'input_width':51,'target_width':42,'chunk_width':18,
        'joint_fields':JOINT_FIELDS,'response_fields':RESPONSE_FIELDS,'chunk_keys':graph['chunk_keys'],'bond_keys':graph['bond_keys'],
        'phases':[{'frame':0,'substep':0,'time':0.,'dt':graph['dt'],'fracture':True,'row_start':0,'rows':1}]}
    (root/'stress_phases/manifest.json').write_text(json.dumps(meta))
    (root/'report.json').write_text(json.dumps({'status':'completed','frames':1,'phases':1,'oracle_response_rows':1,'engine_fallback_evaluations':0}))
    return PhaseHistory(root)

def test_frame_inputs_are_independent_of_future_phase_data(tmp_path):
    data=fixture(tmp_path);before=data.frame_inputs(0,include_projectiles=False)
    # Neither future contact observations nor future target values are consumed
    # by the frame-start model input path, even if those files are corrupted.
    for name in ['targets.f64','chunks.f64']:
        p=tmp_path/'stress_phases'/name;a=np.memmap(p,dtype='<f8',mode='r+');a[:]=np.nan;a.flush()
    after=data.frame_inputs(0,include_projectiles=False)
    for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='invalid joint-history'):data.audit()

def test_phase_audit_tracks_exact_history_and_rejects_missing_rows(tmp_path):
    data=fixture(tmp_path);report=data.audit();assert report['breaks']==1
    path=tmp_path/'stress_phases/inputs.f64';raw=path.read_bytes();path.write_bytes(raw[:-8])
    with pytest.raises(ValueError,match='byte count'):PhaseHistory(tmp_path)

def test_phase_audit_rejects_false_break_and_changed_history(tmp_path):
    data=fixture(tmp_path)
    flags=np.memmap(tmp_path/'stress_phases/target_flags.i32',dtype='<i4',mode='r+',shape=(1,4));flags[0,3]=0;flags.flush()
    with pytest.raises(ValueError,match='connectivity contradiction'):data.audit()
    flags[0,3]=1;flags.flush()
    values=np.memmap(tmp_path/'stress_phases/inputs.f64',dtype='<f8',mode='r+',shape=(1,51));values[0,14]=1.;values.flush()
    with pytest.raises(ValueError,match='history chain mismatch'):data.audit()
