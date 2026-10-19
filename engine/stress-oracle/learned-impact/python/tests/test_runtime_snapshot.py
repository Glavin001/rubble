import copy
import numpy as np
import pytest
import torch
from learned_impact.runtime_snapshot import NativeGraph, FORMAT, WIDTHS
from learned_impact.schema import STATIC_REQUIRED
from learned_impact.response_model import ResponseHistoryImpactNet
from test_material_inputs import fixture as materials
from test_world_inputs import recording


def fixture():
    scene,context,_,_=materials()
    replay,world_scene,_,_=recording(2)
    scene.update(impactors=world_scene['impactors'],ground=world_scene.get('ground'),loads=[],events=[])
    scene['bodies'][0]['name']='body'
    context.update(centers=replay['frames'][0]['centers'],bond_frames=[np.eye(3).tolist()])
    static={k:np.ones(((1 if k.startswith('bond_') else 2)*WIDTHS.get(k,1))).tolist() for k in STATIC_REQUIRED}
    static['bond_endpoints']=[0,1];static['chunk_support']=[0,0]
    graph={'format':FORMAT,'scene':scene,'context':context,'statics':static,'dt':.01,'initial_time':0.,
           'chunk_keys':context['chunk_keys'],'bond_keys':context['bond_keys']}
    values=[0.]*21;values[4]=values[5]=None
    snapshot={'format':FORMAT,'world':replay['frames'][0],
              'chunk_state':[0.]*24,'chunk_gravity_local':[0.,0.,-9.81]*2,
              'bond_joint_state':[0.]*22,'bond_force_prev':[0.]*6,
              'joint_values':[values],'joint_flags':[[0,0,1,1]],'response_values':[[0.]*18]}
    return graph,snapshot


def test_native_targetless_inference_and_diagnostic_independence():
    graph,snapshot=fixture();bridge=NativeGraph(graph);model=ResponseHistoryImpactNet(h=8,p=4)
    pred,batch=bridge.predict(model,snapshot)
    assert torch.isfinite(pred['response_encoded']).all()
    assert not any(k.endswith('_end') for k in batch)
    assert 'bond_fail_class' not in batch and 'chunk_load_bins' not in batch
    snapshot['world']['mechanical_energy']=float('nan')
    snapshot['world']['dissipated_energy']=float('nan')
    after,_=bridge.predict(model,snapshot)
    for k,v in pred.items():torch.testing.assert_close(v,after[k],rtol=0,atol=0)


def test_native_keeps_alive_cracks_filters_retired_and_validates_nulls():
    graph,snapshot=fixture();bridge=NativeGraph(graph)
    snapshot['joint_values'][0][0]=1.;snapshot['joint_flags'][0][3]=0
    batch=bridge.batch(snapshot)
    assert batch['bond_ids'].tolist()==[0] and not batch['bond_failure_eligible'][0]
    assert batch['bond_exact_history'][0,21:23].tolist()==[1.,1.]
    snapshot['joint_flags'][0][2]=0
    assert len(bridge.batch(snapshot)['bond_ids'])==0
    snapshot['joint_values'][0][6]=None
    with pytest.raises(ValueError,match='null outside'):bridge.batch(snapshot)


def test_native_rejects_stale_force_fractional_time_and_noninteger_topology():
    graph,original=fixture();bridge=NativeGraph(graph)
    for field,change,message in [
        ('response_values',lambda s:s['response_values'][0].__setitem__(0,12.),'force mismatch'),
        ('world',lambda s:s['world'].__setitem__('time',.005),'frame boundary'),
        ('joint_flags',lambda s:s['joint_flags'][0].__setitem__(2,.5),'noninteger'),
    ]:
        snapshot=copy.deepcopy(original);change(snapshot)
        with pytest.raises(ValueError,match=message):bridge.batch(snapshot)
