import torch
import pytest
from learned_impact.trajectory_force import ForceTrajectoryNet,query_force,force_loss,impulse_residual
from learned_impact.runtime_snapshot import NativeGraph
from test_runtime_snapshot import fixture

def test_temporal_head_starts_at_hold_and_has_native_time_gradients():
    graph,snapshot=fixture();graph['scene']['impactors']=[];snapshot['world']['impactors']=[]
    batch=NativeGraph(graph).batch(snapshot,include_projectiles=True)
    model=ForceTrajectoryNet(h=8,p=4,knots=3);basis=torch.tensor([[1.,0.,0.],[0.,.5,.5]])
    pred=model(batch);values=query_force(pred,batch,basis)
    torch.testing.assert_close(values,batch['bond_force_prev'][None].expand_as(values),rtol=0,atol=0)
    r={'basis':basis,'batch':batch,'target':torch.ones_like(values),'mask':torch.ones(values.shape[:2],dtype=torch.bool),'weights':torch.tensor([.25,.75])}
    loss=force_loss(pred,r,model.trajectory_scale);loss.backward()
    assert torch.isfinite(loss) and model.trajectory_head.weight.grad.abs().sum()>0
    assert 'phase_observations' not in batch and 'targets' not in batch

def test_time_weighting_is_invariant_to_splitting_a_constant_interval():
    batch={'bond_force_prev':torch.zeros(1,6)};pred={'force_residual_knots':torch.ones(1,2,6)}
    one={'basis':torch.tensor([[1.,0.]]),'batch':batch,'target':torch.zeros(1,1,6),'mask':torch.ones(1,1,dtype=torch.bool),'weights':torch.ones(1)}
    split={'basis':torch.tensor([[1.,0.],[.5,.5]]),'batch':batch,'target':torch.zeros(2,1,6),'mask':torch.ones(2,1,dtype=torch.bool),'weights':torch.tensor([.25,.75])}
    assert force_loss(pred,one,torch.ones(6))==force_loss(pred,split,torch.ones(6))
    assert force_loss(pred,one,torch.ones(6),1.)==force_loss(pred,split,torch.ones(6),1.)

def test_impulse_preserves_sign_and_uses_native_participation_duration():
    values=torch.tensor([[[2.]*6], [[-2.]*6]],requires_grad=True)
    r={'target':torch.zeros_like(values),'weights':torch.tensor([.5,.5]),'mask':torch.ones(2,1,dtype=torch.bool)}
    torch.testing.assert_close(impulse_residual(values,r),torch.zeros(1,6))
    # Once the reference bond retires, the label ends. Do not renormalize its
    # shortened lifetime into a full-frame force or invent a runtime schedule.
    r['mask'][1]=False
    result=impulse_residual(values,r)
    torch.testing.assert_close(result,torch.ones(1,6))
    result.square().sum().backward()
    assert values.grad[0].abs().sum()>0 and values.grad[1].abs().sum()==0

def test_impulse_auxiliary_penalizes_bias_without_erasing_waveform_error():
    r={'basis':torch.eye(2),'batch':{'bond_force_prev':torch.zeros(1,6)},
       'target':torch.zeros(2,1,6),'mask':torch.ones(2,1,dtype=torch.bool),'weights':torch.tensor([.5,.5])}
    constant={'force_residual_knots':torch.ones(1,2,6)}
    alternating={'force_residual_knots':torch.tensor([[[1.]*6,[-1.]*6]])}
    assert force_loss(constant,r,torch.ones(6),0.)==force_loss(alternating,r,torch.ones(6),0.)==1.
    assert force_loss(constant,r,torch.ones(6),1.)==2.
    assert force_loss(alternating,r,torch.ones(6),1.)==1.

def test_boundary_features_require_current_inputs_and_ignore_future_labels():
    from learned_impact.trajectory_force import KinematicForceTrajectoryNet
    graph,snapshot=fixture();graph['scene']['impactors']=[];snapshot['world']['impactors']=[]
    b=NativeGraph(graph).batch(snapshot,include_projectiles=True)
    model=KinematicForceTrajectoryNet(h=8,p=4,knots=3)
    with pytest.raises(ValueError,match='boundary'):model(b)
    b['bond_boundary_kinematics']=torch.full((len(b['bond_ids']),12),1e-9)
    model.fit_boundary_scale([{'batch':b}])
    clean=model.prepare_batch(b)['_boundary_features']
    poisoned=model.prepare_batch({**b,'future_phase_loads':torch.tensor(float('nan')),
        'bond_force_end':torch.full((len(b['bond_ids']),6),1e30)})['_boundary_features']
    torch.testing.assert_close(clean,poisoned,rtol=0,atol=0)
    assert clean[:,:12].abs().min()>0.6  # tiny, resolvable motion survives feature scaling
    changed=model.prepare_batch({**b,'bond_boundary_kinematics':-b['bond_boundary_kinematics']})['_boundary_features']
    torch.testing.assert_close(changed,-clean)
    control=KinematicForceTrajectoryNet(h=8,p=4,knots=3,kinematic_mode='zero');control.load_state_dict(model.state_dict())
    assert torch.count_nonzero(control.prepare_batch(b)['_boundary_features'])==0
    # Both heads begin at the identical hold baseline; new inputs cannot silently
    # alter the initialization before the actual matched training experiment.
    torch.testing.assert_close(model(b)['force_residual_knots'],control(b)['force_residual_knots'],rtol=0,atol=0)
