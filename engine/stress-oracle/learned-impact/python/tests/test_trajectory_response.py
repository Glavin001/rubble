import numpy as np
import torch
from learned_impact.trajectory_response import averaged_basis,query,loss

def test_complete_query_uses_start_for_force_end_for_history_and_integral_for_work():
    r={'batch':{'trajectory_base':torch.zeros(1,42),'trajectory_flags':torch.zeros(1,4),'trajectory_mode':torch.zeros(1,dtype=torch.long)},
       'basis':torch.tensor([[1.,0.]]),'end_basis':torch.tensor([[0.,1.]]),
       'work_basis':torch.from_numpy(averaged_basis([0.],[1.],2))}
    values=torch.zeros(1,2,42);values[:,1,:]=2.
    out,flags,mode=query({'values':values,'flags':torch.zeros(1,2,9)},r)
    torch.testing.assert_close(out[0,0,:21],torch.full((21,),2.))
    torch.testing.assert_close(out[0,0,21:39],torch.zeros(18))
    torch.testing.assert_close(out[0,0,39:],torch.ones(3))
    assert flags.max()<0 and mode.argmax(-1).item()==0

def test_integrated_hat_basis_preserves_linear_work_under_irregular_native_steps():
    edges=np.array([0.,.13,.51,.93,1.]);basis=averaged_basis(edges[:-1],edges[1:],7)
    np.testing.assert_allclose(basis.sum(1),1.,atol=1e-7)
    rates=np.linspace(2.,8.,7)
    assert abs(np.sum((basis@rates)*np.diff(edges))-5.)<1e-7
    assert (basis>=0).all()

def test_complete_loss_has_gradients_for_all_output_families_and_skips_unset_ductility():
    pred={'values':torch.zeros(1,2,42,requires_grad=True),'flags':torch.zeros(1,2,9,requires_grad=True)}
    r={'batch':{'trajectory_base':torch.zeros(1,42),'trajectory_flags':torch.zeros(1,4),'trajectory_mode':torch.zeros(1,dtype=torch.long)},
       'basis':torch.tensor([[1.,0.]]),'end_basis':torch.tensor([[0.,1.]]),'work_basis':torch.tensor([[.5,.5]]),
       'mask':torch.ones(1,1,dtype=torch.bool),'weights':torch.ones(1),'target':torch.ones(1,1,42),
       'flags':torch.ones(1,1,4),'modes':torch.ones(1,1,dtype=torch.long)}
    value=loss(pred,r,torch.ones(42));value.backward()
    for span in [slice(0,4),slice(6,21),slice(21,39),slice(39,42)]:assert pred['values'].grad[:,:,span].abs().sum()>0
    assert pred['values'].grad[:,:,4:6].abs().sum()==0
    assert pred['flags'].grad[:,:,:4].abs().sum()>0 and pred['flags'].grad[:,:,4:].abs().sum()>0

def test_zero_head_exports_complete_targetless_native_contract_and_exact_cached_values():
    from learned_impact.trajectory_response import CompleteTrajectoryNet,forecast,crossing
    from learned_impact.runtime_snapshot import NativeGraph
    from test_runtime_snapshot import fixture
    graph,snapshot=fixture();graph['scene']['impactors']=[];snapshot['world']['impactors']=[]
    runtime=NativeGraph(graph);model=CompleteTrajectoryNet(h=8,p=4,knots=3)
    kin={'format':'native-boundary-kinematics/1','time':snapshot['world']['time'],'values':np.zeros((runtime.e,12)).tolist()}
    result=forecast(model,runtime,snapshot,kin)
    assert result['format']=='learned-native-trajectory/2'
    for b in result['bonds']:
        assert b['response']==[snapshot['response_values'][b['slot']]]*3
        assert b['work_rate']==[[0.,0.,0.]]*3
        assert all(row==b['history_origin_encoded'] for row in b['history'])
        assert b['ductility_set_time']==[None if v is None else 0. for v in snapshot['joint_values'][b['slot']][4:6]]
    assert crossing(np.array([-2.,2.,3.]))==.25
    assert crossing(np.array([-1.,-1.,-1.])) is None

def test_force_objective_preserves_other_gradients_and_time_participation_masks():
    pred={'values':torch.zeros(2,2,42,requires_grad=True),'flags':torch.zeros(2,2,9,requires_grad=True)}
    r={'batch':{'trajectory_base':torch.zeros(2,42),'trajectory_flags':torch.zeros(2,4),'trajectory_mode':torch.zeros(2,dtype=torch.long)},
       'basis':torch.eye(2),'end_basis':torch.eye(2),'work_basis':torch.eye(2),
       'mask':torch.tensor([[True,True],[True,False]]),'weights':torch.tensor([.2,.8]),
       'target':torch.ones(2,2,42),'flags':torch.zeros(2,2,4),'modes':torch.zeros(2,2,dtype=torch.long)}
    # Error for a retired bond must not influence force supervision.
    r['target'][1,1,21:27]=10000.
    base=loss(pred,r,torch.ones(42));base.backward()
    old_values=pred['values'].grad.clone();old_flags=pred['flags'].grad.clone()
    pred['values'].grad=None;pred['flags'].grad=None
    candidate=loss(pred,r,torch.ones(42),force_weight=1.);candidate.backward()
    torch.testing.assert_close(candidate-base,torch.tensor(1.))
    torch.testing.assert_close(pred['values'].grad[:,:,21:27],old_values[:,:,21:27]*8)
    torch.testing.assert_close(pred['values'].grad[:,:,:21],old_values[:,:,:21])
    torch.testing.assert_close(pred['values'].grad[:,:,27:],old_values[:,:,27:])
    torch.testing.assert_close(pred['flags'].grad,old_flags)
    assert pred['values'].grad[1,1,21:27].abs().sum()==0
