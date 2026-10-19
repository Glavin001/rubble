import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from learned_impact.history_model import HistoryCarryImpactNet
from learned_impact.constitutive_model import ConstitutiveNativeImpactNet
from learned_impact.joint_history import JointHistory,finite_features
from learned_impact.schema import load_event
from learned_impact.dataset import event_batches
from learned_impact.benchmark_physics import make_graph
from learned_impact.visualize import load_checkpoint
from learned_impact.train_physics import evaluate


def history_batch():
    m=HistoryCarryImpactNet(h=8,p=4)
    b=make_graph(8,12,m)
    b['bond_exact_history_end']=b['bond_exact_history'].clone()
    b['bond_joint_state_end']=b['bond_joint_state'].clone()
    b['bond_force_end']=torch.zeros(len(b['bond_endpoints']),6)
    b['bond_ids']=torch.arange(len(b['bond_endpoints']))
    b.update(world_motion_target=torch.zeros(8,12),world_motion_valid=torch.ones(8,dtype=torch.bool))
    return b


def test_runtime_graph_retains_cracked_interfaces_and_removes_retired_bonds():
    ev=load_event(Path(__file__).resolve().parents[2]/'fixtures/tiny_event')
    n=ev.manifest['n_bonds'];t=ev.manifest['n_frames']+1
    assert n>=2 and t>=3
    values=np.zeros((t,n,21));values[:,:,4:6]=np.inf
    flags=np.zeros((t,n,4),np.int32);flags[:,:,2:]=1
    flags[1:,0,3]=0 # Cracked but still mechanically active.
    flags[1:,1,2:]=0 # Retired by a split.
    values[1:,0,0]=1;values[2:,0,19]=4.5
    ev.frames[0]['bond_fail_class'][:2]=1
    ev.frames[1]['bond_fail_class'][0]=4 # Old duplicated labels must not re-break.
    ev.exact_joint_history=JointHistory(values,flags,np.arange(t)*ev.manifest['dt'])
    legacy=list(event_batches(ev));runtime=list(event_batches(ev,bond_policy='runtime_alive'))
    assert 0 not in legacy[1]['bond_ids'] and 0 in runtime[1]['bond_ids']
    assert 1 not in runtime[1]['bond_ids']
    j=runtime[1]['bond_ids'].tolist().index(0)
    assert not runtime[1]['bond_failure_eligible'][j]
    assert runtime[1]['bond_fail_class'][j]==0
    assert runtime[1]['bond_exact_history_end'][j,19]>runtime[1]['bond_exact_history'][j,19]
    assert ev.frames[1]['bond_fail_class'][0]==4 # Recorded labels remain untouched.
    ev.exact_joint_history=None
    with pytest.raises(ValueError,match='requires exact'):list(event_batches(ev,bond_policy='runtime_alive'))


def test_initial_head_preserves_history_and_round_trips_all_joint_fields(tmp_path):
    b=history_batch();e=len(b['bond_endpoints'])
    values=np.zeros((e,21));values[:,4:6]=np.inf
    values[0]=[.2,.3,2,3,4,5,.4,-.01,.02,-.03,.1,-.2,.3,-.02,.01,-.01,12,2,3000,5,1.2]
    flags=np.zeros((e,4),np.int32);flags[:,2:]=1;flags[0,:2]=[1,3]
    b['bond_exact_history']=torch.from_numpy(finite_features(values,flags))
    b['bond_exact_history_end']=b['bond_exact_history'].clone()
    torch.manual_seed(21);old=ConstitutiveNativeImpactNet(h=8,p=4)
    torch.manual_seed(21);m=HistoryCarryImpactNet(h=8,p=4)
    pred=m(b)
    for key,value in old(b).items():torch.testing.assert_close(pred[key],value,rtol=0,atol=0)
    decoded=m.decode_joint_state(pred,b)
    np.testing.assert_allclose(decoded['values'],values,rtol=1e-6,atol=1e-8)
    np.testing.assert_array_equal(decoded['rebar_broken'],flags[:,0])
    np.testing.assert_array_equal(decoded['mode'],flags[:,1])
    torch.save(m.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':m.architecture,'threshold':m.threshold}))
    reloaded,_=load_checkpoint(tmp_path)
    for key,value in pred.items():torch.testing.assert_close(reloaded(b)[key],value,rtol=0,atol=0)


def test_exact_targets_drive_all_heads_but_never_enter_inference():
    b=history_batch();m=HistoryCarryImpactNet(h=8,p=4)
    end=b['bond_exact_history_end'];end[:,:21]=.1;end[:,21:23]=0;end[:,23]=1;end[:,24:29]=0;end[:,26]=1
    m.fit_target_scales([('train','wall',0,b)])
    pred=m(b);m.state_loss(pred,b)['loss'].backward()
    for head in (m.exact_delta,m.ductility_set,m.rebar_break,m.history_mode):assert head.weight.grad.abs().sum()>0
    poisoned={**b,'bond_exact_history_end':torch.full_like(end,float('nan'))}
    for key,value in pred.items():torch.testing.assert_close(m(poisoned)[key],value,rtol=0,atol=0)
    with pytest.raises(ValueError,match='eligibility'):m({**b,'bond_failure_eligible':torch.zeros(len(end),dtype=torch.bool)})


def test_previously_broken_bonds_do_not_contribute_new_failure_loss_or_metrics():
    b=history_batch();b['bond_failure_eligible'][0]=False;b['bond_exact_history'][0,30]=0
    b['bond_fail_class'].zero_();b['bond_fail_class'][1]=2
    m=HistoryCarryImpactNet(h=8,p=4);pred=m(b)
    altered={**pred,'break_logit':pred['break_logit'].clone(),'time_logits':pred['time_logits'].clone()}
    altered['break_logit'][0]=999;altered['time_logits'][0]=torch.tensor([0.,1.,2.,999.])
    assert m.predict_classes(altered)[0]==0
    a=m.state_loss(pred,b);c=m.state_loss(altered,b)
    for key in ('fail','timing','fracture_cdf'):torch.testing.assert_close(a[key],c[key],rtol=0,atol=0)
    scores=evaluate(m,[('one','wall',0,m.prepare_batch(b))])
    assert scores['bonds']==11 and sum(scores['fracture_cdf_score']['bond_intervals'])==11


def test_unset_and_irreversible_constraints_and_nonfinite_decode_rejection():
    b=history_batch();m=HistoryCarryImpactNet(h=8,p=4)
    b['bond_exact_history'][0,21:23]=0;b['bond_exact_history'][0,4:6]=1
    b['bond_exact_history'][0,23]=1
    with torch.no_grad():m.exact_delta.bias.fill_(-2)
    pred=m(b);decoded=m.decode_joint_state(pred,b)
    assert np.isfinite(decoded['values'][0,4:6]).all()
    assert np.isposinf(decoded['values'][1:,4:6]).all()
    assert decoded['rebar_broken'][0]
    for i in (*m.bounded,*m.monotonic):assert (pred['exact_history_encoded'][:,i]>=b['bond_exact_history'][:,i]).all()
    pred['exact_history_encoded']=pred['exact_history_encoded'].clone();pred['exact_history_encoded'][0,7]=1000
    with pytest.raises(ValueError,match='invalid joint-history'):m.decode_joint_state(pred,b)


def test_full_history_metrics_do_not_hide_unset_errors_or_cracked_interfaces():
    from learned_impact.history_metrics import ExactHistoryMetrics
    b=history_batch();b['bond_failure_eligible'][0]=False;b['bond_exact_history'][0,30]=0
    b['bond_exact_history_end'][0,21]=0;b['bond_exact_history_end'][0,4]=2
    m=HistoryCarryImpactNet(h=8,p=4);pred=m(b)
    metrics=ExactHistoryMetrics();metrics.add(pred,b);report=metrics.report()
    assert report['already_cracked_active_bond_intervals']==1
    assert report['finite_target_counts']['ductility']==1
    assert report['encoded_rmse']['ductility']==2
    assert report['ductility_unset_confusion'][0][0][1]==1
    assert report['mode_confusion'][0][0]==12


def test_all_cracked_interfaces_still_train_history_without_new_breaks():
    b=history_batch();b['bond_failure_eligible'][:]=False;b['bond_exact_history'][:,30]=0
    b['bond_exact_history_end'][:,19]=1
    m=HistoryCarryImpactNet(h=8,p=4);pred=m(b);parts=m.state_loss(pred,b)
    assert parts['fail']==parts['timing']==parts['fracture_cdf']==0
    assert parts['exact_continuous']>0
    parts['loss'].backward()
    assert m.exact_delta.weight.grad.abs().sum()>0


def test_residual_objective_recovers_from_negative_and_saturated_history_heads():
    from learned_impact.history_model import ResidualHistoryImpactNet
    b=history_batch();m=ResidualHistoryImpactNet(h=8,p=4)
    b['bond_exact_history_end'][:,0]=np.log1p(.5)
    b['bond_exact_history_end'][:,19]=.1
    with torch.no_grad():m.exact_delta.bias.fill_(-2)
    pred=m(b)
    assert (pred['exact_history_encoded'][:,0]==0).all()
    assert (pred['exact_history_encoded'][:,19]==0).all()
    m.exact_state_loss(pred,b)['continuous'].backward()
    assert m.exact_delta.bias.grad[0]<0 and m.exact_delta.bias.grad[19]<0
    m.zero_grad()
    with torch.no_grad():m.exact_delta.bias.fill_(2)
    pred=m(b)
    assert (pred['exact_history_encoded'][:,0]==torch.tensor(np.log(2),dtype=torch.float32)).all()
    m.exact_state_loss(pred,b)['continuous'].backward()
    assert m.exact_delta.bias.grad[0]>0


def test_decoding_restores_exact_unit_damage_without_hiding_invalid_encodings():
    b=history_batch();m=HistoryCarryImpactNet(h=8,p=4);pred=m(b)
    pred['exact_history_encoded']=pred['exact_history_encoded'].clone()
    pred['exact_history_encoded'][:,0]=np.log(2)
    assert (m.decode_joint_state(pred,b)['values'][:,0]==1.).all()
    pred['exact_history_encoded'][0,0]=1.
    with pytest.raises(ValueError,match='outside admissible'):m.decode_joint_state(pred,b)
