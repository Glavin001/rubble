import json
import numpy as np
import pytest
import torch
from learned_impact.response_model import ResponseHistoryImpactNet
from learned_impact.consistent_model import ConsistentHistoryImpactNet
from learned_impact.response_metrics import ResponseMetrics
from learned_impact.visualize import load_checkpoint
from test_consistent_model import batch as consistent_batch


def batch():
    b=consistent_batch();e=len(b['bond_endpoints'])
    b.update(bond_response_encoded=torch.zeros(e,18),bond_response_encoded_end=torch.zeros(e,18),
             bond_response_alive=torch.ones(e,dtype=torch.bool),bond_response_alive_end=torch.ones(e,dtype=torch.bool))
    return b


def test_persistence_initialization_signed_stress_and_target_causality(tmp_path):
    b=batch();b['bond_response_encoded'][:,13]=-.2;b['bond_response_encoded'][:,16]=-.3
    torch.manual_seed(12);base=ConsistentHistoryImpactNet(h=8,p=4)
    torch.manual_seed(12);model=ResponseHistoryImpactNet(h=8,p=4)
    pred=model(b)
    torch.testing.assert_close(pred['break_logit'],base(b)['break_logit'],rtol=0,atol=0)
    torch.testing.assert_close(pred['response_encoded'],b['bond_response_encoded'],rtol=0,atol=0)
    torch.testing.assert_close(pred['bond_force_end'],pred['response_encoded'][:,:6],rtol=0,atol=0)
    assert (model.decode_response(pred)[:,[13,16]]<0).all()
    poisoned={**b,'bond_response_encoded_end':torch.full_like(b['bond_response_encoded_end'],float('nan')),
              'bond_response_alive_end':~b['bond_response_alive_end']}
    for k,v in pred.items():torch.testing.assert_close(v,model(poisoned)[k],rtol=0,atol=0)
    torch.save(model.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':model.architecture,'threshold':model.threshold}))
    loaded,_=load_checkpoint(tmp_path)
    for k,v in pred.items():torch.testing.assert_close(v,loaded(b)[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='18 exact'):
        model({k:v for k,v in b.items() if k!='bond_response_encoded'})


def test_balanced_cohort_scales_and_gradients_before_constraint_clamp():
    b=batch();b['bond_response_alive_end'][0]=False
    b['bond_response_encoded_end'].fill_(1.);b['bond_response_encoded_end'][0]=10.
    model=ResponseHistoryImpactNet(h=8,p=4);prepared=model.prepare_batch(b)
    model.fit_target_scales([('event','wall',0,prepared)])
    torch.testing.assert_close(model.response_cohort_rms[0],torch.ones(18))
    torch.testing.assert_close(model.response_cohort_rms[1],torch.full((18,),10.))
    torch.testing.assert_close(model.response_rms,torch.full((18,),np.sqrt(50.5)))
    pred=model(prepared);pred['response_raw_encoded'].retain_grad()
    loss=model.response_loss(pred,prepared)
    torch.testing.assert_close(loss['loss'],torch.tensor(1.))
    loss['loss'].backward()
    assert (pred['response_raw_encoded'].grad[:,12]<0).all()
    # Predicted stored energy cannot be negative, but the raw loss still has
    # a correcting gradient when the model proposes an inadmissible value.
    with torch.no_grad():model.response_extra.bias[6]=-2.
    pred=model(prepared);assert (pred['response_encoded'][:,12]==0).all()
    model.zero_grad();model.response_loss(pred,prepared)['loss'].backward()
    assert model.response_extra.bias.grad[6]<0


def test_response_metrics_keep_retired_cache_separate_and_score_baselines():
    b=batch();e=len(b['bond_endpoints']);b['bond_response_alive_end'][0]=False
    b['bond_response_encoded_end'][:,0]=np.log(2)
    pred={'response_encoded':b['bond_response_encoded_end'].clone()}
    metrics=ResponseMetrics();metrics.add(pred,b);report=metrics.report()
    assert report['active_end']['bond_intervals']==e-1 and report['retired_cache']['bond_intervals']==1
    for name in ('active_end','retired_cache'):
        assert report[name]['model']['si_component_rmse']['force_x']==0
        assert report[name]['zero']['si_component_rmse']['force_x']==pytest.approx(1000.,rel=1e-7)
        assert report[name]['persistence']==report[name]['zero']


def test_missing_cohort_scaling_and_inference_without_endpoint_targets():
    b=batch();b['bond_response_encoded_end'].fill_(.2)
    model=ResponseHistoryImpactNet(h=8,p=4);model.fit_target_scales([('event','wall',0,model.prepare_batch(b))])
    torch.testing.assert_close(model.response_cohort_rms[0],model.response_cohort_rms[1])
    raw={k:v for k,v in b.items() if not k.endswith('_end')}
    assert torch.isfinite(model(raw)['response_encoded']).all()
    bad={**b,'bond_response_alive':torch.zeros_like(b['bond_response_alive'])}
    with pytest.raises(ValueError,match='runtime-alive'):model(bad)


def test_nonzero_rms_retains_small_energy_target_and_handles_constant_fields():
    b=batch();b['bond_response_encoded_end'][:,12]=1e-7
    m=ResponseHistoryImpactNet(h=8,p=4);m.response_scale_mode='nonzero-rms'
    prepared=m.prepare_batch(b);m.fit_target_scales([('event','wall',0,prepared)])
    assert float(m.response_rms[12])==pytest.approx(1e-7)
    torch.testing.assert_close(m.response_rms[:12],torch.ones(12))
    torch.testing.assert_close(m.response_cohort_rms[0],m.response_cohort_rms[1])
    loss=m.response_loss(m(prepared),prepared)['loss']
    assert float(loss.detach())==pytest.approx(1/18)
    loss.backward();assert m.response_extra.bias.grad[6]<0
