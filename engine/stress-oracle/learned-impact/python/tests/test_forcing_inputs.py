import copy,json
import numpy as np
import pytest
import torch
from learned_impact.forcing_inputs import observations,curve,selection
from learned_impact.physics import WorldMotionImpactNet,ForcingPreviewImpactNet,ForcingNativeImpactNet
from learned_impact.visualize import load_checkpoint
from test_world_inputs import recording,batch


def forced_record(n=8):
    replay,scene,context,manifest=recording(n)
    scene.update(events=[],bodies=[{'name':'wall','chunks':[{'center':[i*.1,0,0],'groups':['top'] if i<3 else []} for i in range(n)]}],
                 loads=[{'type':'pressure','body':'wall','chunks':{'group':'top'},'face_normal':[0,0,2],
                         'pressure':{'type':'constant','value':30000.}},
                        {'type':'blast','position':[5,0,1],'tnt_mass':2.,'time':.01}])
    context['chunk_keys']=[[0,i] for i in range(n)]
    return replay,scene,context,manifest


def forced_batch():
    b=batch();args=forced_record(len(b['chunk_mass']))
    b.update({k:torch.from_numpy(v) for k,v in observations(*args)[0].items()})
    b.update(world_motion_target=torch.full((len(b['chunk_mass']),12),.01),world_motion_valid=torch.ones(len(b['chunk_mass']),dtype=torch.bool))
    return b


def test_forcing_matches_selected_chunks_and_absolute_known_schedule():
    args=forced_record();a,b=observations(*args)
    np.testing.assert_array_equal(a['forcing_selected'][0],[True]*3+[False]*5)
    np.testing.assert_array_equal(a['forcing_direction'][0],[0,0,1])
    assert a['forcing_curve_parameters'][0,2]==30000
    assert a['forcing_blast'][1,1]==pytest.approx(.01)
    assert b['forcing_blast'][1,1]==pytest.approx(0.)
    poison=copy.deepcopy(args);poison[0]['frames'][1]['centers']=[[999,999,999]]*8
    poison[0]['frames'][1]['impactors']=[]
    for k,v in a.items():np.testing.assert_array_equal(observations(*poison)[0][k],v)
    args[1]['loads'][1]['tnt_mass']=4.
    assert observations(*args)[0]['forcing_blast'][1,0]==4.


def test_point_force_offset_and_selectors_are_explicit():
    args=forced_record();scene=args[1]
    scene['loads']=[{'type':'point_force','body':'wall','chunk':2,'direction':[3,0,0],
                     'point':[.2,.1,0],'magnitude':{'type':'ramp','t0':0.,'t1':.1,'value':100.}}]
    o=observations(*args)[0]
    np.testing.assert_array_equal(o['forcing_selected'][0],np.arange(8)==2)
    np.testing.assert_allclose(o['forcing_position'][0],[0,.1,0])
    body=scene['bodies'][0]
    assert selection('all',body)==set(range(8))
    assert selection({'region':{'min':[.1,-1,-1],'max':[.21,1,1]}},body)=={1,2}
    assert selection({'indices':[1,1,3]},body)=={1,3}
    with pytest.raises(ValueError,match='index'):selection({'indices':[99]},body)


def test_curves_keep_parameters_and_every_table_knot():
    assert curve({'type':'half_sine','start':2,'duration':3,'peak':4},1)[1].tolist()==[1,3,4,0]
    assert curve({'type':'friedlander','arrival':2,'duration':3,'peak':4,'decay':5},1)[1].tolist()==[1,3,4,5]
    args=forced_record();points=[[i*.001,float(i)] for i in range(100)]
    args[1]['loads'][0]['pressure']={'type':'table','points':points}
    o=observations(*args)[1]
    assert len(o['forcing_table_points'])==100
    np.testing.assert_allclose(o['forcing_table_points'][:,0],np.array(points)[:,0]-.01,atol=1e-8)
    assert (o['forcing_table_owner']==0).all()
    with pytest.raises(ValueError,match='increase'):curve({'type':'table','points':[[0,1],[0,2]]},0)
    with pytest.raises(ValueError,match='duration'):curve({'type':'half_sine','start':0,'duration':0,'peak':1},0)


def test_missing_or_unsupported_forcing_is_rejected():
    args=forced_record();args[1]['events']=[{'type':'remove_chunks'}]
    with pytest.raises(ValueError,match='removal'):observations(*args)
    args=forced_record();del args[1]['loads']
    with pytest.raises(ValueError,match='require scene loads'):observations(*args)
    args=forced_record();args[1]['loads'][0]['type']='unknown'
    with pytest.raises(ValueError,match='unsupported prescribed'):observations(*args)
    args=forced_record();args[2]['chunk_keys'][0]=[0,1]
    with pytest.raises(ValueError,match='unique'):observations(*args)


def test_native_predictions_ignore_all_interval_loads_and_qs_inputs(tmp_path):
    b=forced_batch();torch.manual_seed(3);m=ForcingNativeImpactNet(h=8,p=4)
    before=m(b)
    keys=['chunk_load_bins','chunk_contact','chunk_preview_load_bins','chunk_preview_contact','bond_qs_force','qs_mode']
    poison={**b,**{k:torch.full_like(b[k],999) for k in keys if k in b}}
    absent={k:v for k,v in b.items() if k not in keys}
    for k,v in before.items():
        torch.testing.assert_close(m(poison)[k],v,rtol=0,atol=0)
        torch.testing.assert_close(m(absent)[k],v,rtol=0,atol=0)
    with pytest.raises(ValueError,match='requires forcing_kind'):m({k:v for k,v in absent.items() if k!='forcing_kind'})
    torch.save(m.state_dict(),tmp_path/'state_dict.pt');(tmp_path/'run.json').write_text(json.dumps({'architecture':m.architecture,'threshold':.5}))
    loaded,_=load_checkpoint(tmp_path)
    for k,v in before.items():torch.testing.assert_close(loaded(absent)[k],v,rtol=0,atol=0)


def test_forcing_encoder_preserves_initial_v6_and_learns_all_sources():
    b=forced_batch();torch.manual_seed(9);v6=WorldMotionImpactNet(h=8,p=4)
    torch.manual_seed(9);m=ForcingPreviewImpactNet(h=8,p=4)
    for k,v in v6(b).items():torch.testing.assert_close(m(b)[k],v,rtol=0,atol=0)
    m.fit_target_scales([('train','test',0,b)])
    m.state_loss(m(b),b)['loss'].backward()
    assert m.forcing_encoder.net[-1].weight.grad.abs().sum()>0
    with torch.no_grad():torch.nn.init.normal_(m.forcing_encoder.net[-1].weight,std=.1)
    original=m(b)
    changed={**b,'forcing_blast':b['forcing_blast'].clone()};changed['forcing_blast'][1,0]*=3
    assert not torch.equal(original['break_logit'],m(changed)['break_logit'])
    keys=['kind','direction','position','curve_kind','curve_parameters','blast','selected']
    permuted={**b,**{'forcing_'+k:b['forcing_'+k].flip(0) for k in keys}}
    for k,v in original.items():torch.testing.assert_close(m(permuted)[k],v,rtol=1e-5,atol=1e-6)
    empty={**b,**{'forcing_'+k:b['forcing_'+k][:0] for k in keys}}
    assert all(torch.isfinite(v).all() for v in m(empty).values())


def test_table_knots_follow_source_permutation_and_receive_gradients():
    b=forced_batch()
    b.update(forcing_table_points=torch.tensor([[0.,0.],[.01,30000.],[.02,0.]]),
             forcing_table_owner=torch.zeros(3,dtype=torch.int64))
    b['forcing_curve_kind']=b['forcing_curve_kind'].clone();b['forcing_curve_kind'][0]=4
    torch.manual_seed(17);m=ForcingNativeImpactNet(h=8,p=4)
    m(b)['break_logit'].sum().backward()
    assert m.forcing_table_encoder.net[-1].weight.grad.abs().sum()>0
    with torch.no_grad():torch.nn.init.normal_(m.forcing_table_encoder.net[-1].weight,std=.1)
    original=m(b)
    changed={**b,'forcing_table_points':b['forcing_table_points'].clone()}
    changed['forcing_table_points'][1,1]*=2
    assert not torch.equal(original['break_logit'],m(changed)['break_logit'])
    keys=['kind','direction','position','curve_kind','curve_parameters','blast','selected']
    permuted={**b,**{'forcing_'+k:b['forcing_'+k].flip(0) for k in keys},
              'forcing_table_owner':1-b['forcing_table_owner']}
    for k,v in original.items():torch.testing.assert_close(m(permuted)[k],v,rtol=1e-5,atol=1e-6)
