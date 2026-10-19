import copy,json
import numpy as np
import pytest
import torch
from learned_impact.world_targets import rotation_exp,rotation_log,targets,reconstruct
from learned_impact.world_inputs import observations
from learned_impact.physics import WorldStateImpactNet,WorldMotionImpactNet
from learned_impact.visualize import load_checkpoint
from test_world_inputs import recording,batch


def test_rotation_log_roundtrip_zero_pi_and_oblique_axes():
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.],[1.,2.,3.]])
    axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    vectors=np.concatenate([axes*a for a in (0.,1e-12,.3,np.pi-1e-9,np.pi,np.pi+1e-8,2*np.pi+.1)])
    matrices=rotation_exp(vectors)
    logs=rotation_log(matrices)
    np.testing.assert_allclose(rotation_exp(logs),matrices,atol=2e-8)
    assert np.linalg.norm(logs,axis=1).max()<=np.pi+1e-12
    with pytest.raises(ValueError,match='proper orthogonal'):rotation_log(np.diag([1,1,-1]))


def moving_record():
    replay,scene,context,manifest=recording(4);manifest['n_frames']=1;replay['frames']=replay['frames'][:2]
    start,end=replay['frames'];dt=manifest['dt'];rng=np.random.default_rng(5)
    start['centers']=rng.normal(size=(4,3)).tolist();context['centers']=start['centers']
    start['velocities']=rng.normal(size=(4,3)).tolist();start['angular_velocities']=rng.normal(size=(4,3)).tolist()
    start['rotations']=rotation_exp(rng.normal(size=(4,3))*.3).tolist()
    residual=rng.normal(size=(4,12))*.01
    end.update({k:v.tolist() for k,v in reconstruct(start,residual,dt).items()})
    start['clusters']=[0]*4;end['clusters']=[1,2,3,4]
    return (replay,scene,context,manifest),residual


def test_world_targets_reconstruct_after_fragment_resets_and_are_covariant():
    args,residual=moving_record();label=targets(args[0],args[3])[0]
    assert label['world_motion_valid'].all()
    np.testing.assert_allclose(label['world_motion_target'],residual,atol=2e-9)
    decoded=reconstruct(args[0]['frames'][0],label['world_motion_target'],args[3]['dt'])
    for key,value in decoded.items():np.testing.assert_allclose(value,args[0]['frames'][1][key],atol=2e-8)
    q=rotation_exp(np.array([.2,-.3,.4]));rotated=copy.deepcopy(args)
    for frame in rotated[0]['frames']:
        for key in ('centers','velocities','angular_velocities'):frame[key]=(np.asarray(frame[key])@q.T).tolist()
        frame['rotations']=(q@np.asarray(frame['rotations'])).tolist()
    got=targets(rotated[0],rotated[3])[0]['world_motion_target']
    np.testing.assert_allclose(got.reshape(-1,4,3),residual.reshape(-1,4,3)@q.T,atol=2e-9)


def test_ballistic_baseline_activity_and_future_labels_not_start_inputs():
    args,_=moving_record();start,end=args[0]['frames'];zero=np.zeros((4,12))
    end.update({k:v.tolist() for k,v in reconstruct(start,zero,args[3]['dt']).items()})
    np.testing.assert_allclose(targets(args[0],args[3])[0]['world_motion_target'],0,atol=1e-12)
    before=observations(*args)[0]
    end['centers'][0][0]+=1;end['active'][1]=False
    label=targets(args[0],args[3])[0]
    assert label['world_motion_target'][0,0]==1 and not label['world_motion_valid'][1]
    for key,value in observations(*args)[0].items():np.testing.assert_array_equal(value,before[key])
    end['rotations'][0][0][0]=999
    with pytest.raises(ValueError,match='proper orthogonal'):targets(args[0],args[3])


def test_world_motion_head_zero_baseline_gradient_and_label_isolation(tmp_path):
    b=batch();n=len(b['chunk_mass'])
    b.update(world_motion_target=torch.full((n,12),.01),world_motion_valid=torch.ones(n,dtype=torch.bool))
    torch.manual_seed(9);old=WorldStateImpactNet(h=8,p=4)
    torch.manual_seed(9);model=WorldMotionImpactNet(h=8,p=4)
    for key,value in old(b).items():torch.testing.assert_close(model(b)[key],value,rtol=0,atol=0)
    model.fit_target_scales([('train','test',0,b)])
    pred=model(b);assert torch.count_nonzero(model.decode_world_motion(pred))==0
    loss=model.state_loss(pred,b);loss['loss'].backward()
    assert model.world_motion.weight.grad.abs().sum()>0 and torch.isfinite(model.world_motion.weight.grad).all()
    poison=dict(b,world_motion_target=b['world_motion_target']*999,world_motion_valid=~b['world_motion_valid'])
    for key,value in model(b).items():torch.testing.assert_close(model(poison)[key],value,rtol=0,atol=0)
    without={k:v for k,v in b.items() if not k.startswith('world_motion_')}
    for key,value in model(b).items():torch.testing.assert_close(model(without)[key],value,rtol=0,atol=0)
    torch.save(model.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':model.architecture,'threshold':model.threshold}))
    loaded,_=load_checkpoint(tmp_path)
    torch.testing.assert_close(loaded.decode_world_motion(loaded(b)),model.decode_world_motion(model(b)))


def test_world_evaluation_retains_split_chunks_and_scores_ballistic_baseline():
    from learned_impact.train_physics import evaluate
    b=batch();n=len(b['chunk_mass']);b['bond_ids']=torch.arange(len(b['bond_endpoints']))
    b['delta_valid']=torch.zeros(n,dtype=torch.bool)
    b.update(world_motion_target=torch.full((n,12),.01),world_motion_valid=torch.ones(n,dtype=torch.bool))
    m=WorldMotionImpactNet(h=8,p=4)
    result=evaluate(m,[('event','family',0,b)])
    assert result['motion_scored_chunk_intervals']==0
    assert result['world_motion']['scored_chunk_intervals']==n
    assert result['world_motion']['rmse']==result['world_motion']['constant_velocity_baseline_rmse']
    assert result['world_motion']['orientation_geodesic_rmse_rad']==pytest.approx(np.sqrt(3)*.01)


def test_world_motion_empty_targets_do_not_report_perfect_error():
    from learned_impact.train_physics import evaluate
    b=batch();n=len(b['chunk_mass']);b['bond_ids']=torch.arange(len(b['bond_endpoints']))
    b.update(world_motion_target=torch.zeros(n,12),world_motion_valid=torch.zeros(n,dtype=torch.bool))
    m=WorldMotionImpactNet(h=8,p=4)
    result=evaluate(m,[('event','family',0,b)])
    assert result['world_motion']['scored_chunk_intervals']==0
    assert all(v is None for v in result['world_motion']['rmse'].values())
    with pytest.raises(ValueError,match='no valid world-motion'):m.fit_target_scales([('train','test',0,b)])


def test_visualizer_world_endpoint_matches_labels_after_all_fragment_resets():
    from learned_impact.visualize import world_comparison
    args,residual=moving_record();start,end=args[0]['frames'];dt=args[3]['dt']
    exact=world_comparison(start,end,residual,np.ones(4,dtype=bool),dt)
    np.testing.assert_allclose(exact['centers'],end['centers'],atol=1e-12)
    np.testing.assert_allclose(exact['rotations'],end['rotations'],atol=1e-12)
    assert exact['position_rmse_m']<1e-12 and exact['orientation_rmse_rad']<1e-12
    assert exact['baseline_position_rmse_m']>0 and exact['scored_pieces']==4
    zero=world_comparison(start,end,np.zeros_like(residual),np.ones(4,dtype=bool),dt)
    assert zero['position_error']==zero['baseline_position_error']
    empty=world_comparison(start,end,residual,np.zeros(4,dtype=bool),dt)
    assert empty['position_rmse_m'] is None and empty['orientation_rmse_rad'] is None
    assert empty['position_error']==[None]*4
    json.dumps(empty,allow_nan=False)
