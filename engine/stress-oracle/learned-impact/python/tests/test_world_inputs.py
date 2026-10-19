import copy
import json
import numpy as np
import pytest
import torch
from learned_impact.world_inputs import observations
from learned_impact.dataset import synthetic_batch
from learned_impact.physics import StatePreviewImpactNet,WorldStateImpactNet,WorldInputControlNet
from learned_impact.visualize import load_checkpoint


def recording(n=8):
    shape={'type':'box','half_extents':[.2,.3,.4]}
    desc={'name':'ram','shape':shape,'mass':200.,'crush':{'max_force':1000.,'energy':40.}}
    scene={'name':'test','ground':{'height':0.},'impactors':[desc]}
    centers=np.zeros((n,3)).tolist()
    context={'source_scene':'test','centers':centers}
    frames=[]
    for i in range(3):
        frames.append({'time':i*.01,'centers':(np.array(centers)+i*.001).tolist(),
                       'rotations':np.tile(np.eye(3),(n,1,1)).tolist(),'active':[True]*n,
                       'velocities':np.full((n,3),i*.1).tolist(),'angular_velocities':np.zeros((n,3)).tolist(),
                       'impactors':[{'name':'ram','shape':shape,'center':[1.-i*.1,0,0],
                                     'velocity':[-10.,0,0],'rotation':np.eye(3).tolist()}]})
    replay={'format':'learned-impact-replay/1','scene':'test','frames':frames}
    return replay,scene,context,{'n_chunks':n,'n_frames':2,'dt':.01}


def batch():
    b=synthetic_batch()
    world=observations(*recording(len(b['chunk_mass'])))[0]
    b.update({k:torch.from_numpy(v) for k,v in world.items()})
    b.update(chunk_preview_load_bins=b['chunk_load_bins'],chunk_preview_contact=torch.zeros(len(b['chunk_mass'])))
    b['bond_joint_state_end']=b['bond_joint_state'].clone()
    return b


def test_world_inputs_use_start_snapshot_and_never_future_or_energy():
    args=recording();before=observations(*args)
    poisoned=copy.deepcopy(args)
    poisoned[0]['frames'][1]['centers']=np.full((8,3),999.).tolist()
    poisoned[0]['frames'][1]['velocities']=np.full((8,3),999.).tolist()
    poisoned[0]['frames'][1]['impactors'][0]['center']=[999.,0,0]
    poisoned[0]['frames'][0]['mechanical_energy']=float('nan')
    poisoned[0]['frames'][0]['dissipated_energy']=float('nan')
    after=observations(*poisoned)
    for k,v in before[0].items():np.testing.assert_array_equal(v,after[0][k])
    assert not np.array_equal(before[1]['world_center'],after[1]['world_center'])
    assert before[1]['world_velocity'][0,0]==pytest.approx(.1)
    poisoned[0]['frames'][1]['time']+=.005
    with pytest.raises(ValueError,match='snapshot time'):observations(*poisoned)


def test_world_inputs_reject_misaligned_identity_and_invalid_rotation():
    args=recording();args[0]['scene']='different'
    with pytest.raises(ValueError,match='identity'):observations(*args)
    args=recording();args[0]['frames'][0]['impactors'][0]['name']='unrecorded'
    with pytest.raises(ValueError,match='identity'):observations(*args)
    args=recording();args[0]['frames'][0]['rotations'][0][0][0]=2
    with pytest.raises(ValueError,match='rotation'):observations(*args)


def test_world_model_initialization_matches_control_and_state_v4(tmp_path):
    b=batch()
    torch.manual_seed(5);old=StatePreviewImpactNet(h=8,p=4)
    torch.manual_seed(5);new=WorldStateImpactNet(h=8,p=4)
    torch.manual_seed(5);control=WorldInputControlNet(h=8,p=4)
    for k,v in old(b).items():
        torch.testing.assert_close(v,new(b)[k],rtol=0,atol=0)
        torch.testing.assert_close(v,control(b)[k],rtol=0,atol=0)
    # Branches must receive gradients at the zero-output initialization.
    new.state_loss(new(b),b)['loss'].backward()
    assert new.world_encoder.net[-1].weight.grad.abs().sum()>0
    assert new.impactor_encoder.net[-1].weight.grad.abs().sum()>0
    torch.save(new.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':new.architecture,'threshold':.5}))
    loaded,_=load_checkpoint(tmp_path)
    with pytest.raises(ValueError,match='requires world_velocity'):
        loaded({k:v for k,v in b.items() if not k.startswith('world_')})
    for k,v in new(b).items():torch.testing.assert_close(v,loaded(b)[k])


def test_world_model_uses_all_projectiles_and_is_permutation_invariant():
    b=batch();model=WorldStateImpactNet(h=8,p=4)
    with torch.no_grad():
        torch.nn.init.normal_(model.impactor_encoder.net[-1].weight,std=.1)
    for k in ('impactor_center','impactor_velocity','impactor_rotation','impactor_properties'):
        b[k]=b[k].repeat(2,*([1]*(b[k].ndim-1)))
    b['impactor_center'][1]+=torch.tensor([1.,2.,3.])
    original=model(b)
    permuted=dict(b,**{k:v.flip(0) for k,v in b.items() if k.startswith('impactor_')})
    for k,v in original.items():torch.testing.assert_close(v,model(permuted)[k],rtol=1e-5,atol=1e-6)
    changed={**b,'impactor_velocity':b['impactor_velocity'].clone()}
    changed['impactor_velocity'][1]*=3
    assert not torch.equal(original['break_logit'],model(changed)['break_logit'])
    empty=dict(b,**{k:v[:0] for k,v in b.items() if k.startswith('impactor_')})
    assert all(torch.isfinite(v).all() for v in model(empty).values())


def test_tail_evaluation_rejects_recipe_partition_leakage():
    from learned_impact.compare_tails import verify_recipe_partitions
    saved={'events':{'house_f120':'test'},'recipe_groups':{'house_f120':'catalog:house'}}
    short={'events':{'house':'test'},'recipe_groups':{'house':'catalog:house'}}
    verify_recipe_partitions(saved,short)
    short['events']['house']='train'
    with pytest.raises(ValueError,match='partition differs'):verify_recipe_partitions(saved,short)
    short['recipe_groups']['house']='catalog:unknown'
    with pytest.raises(ValueError,match='recipe missing'):verify_recipe_partitions(saved,short)
