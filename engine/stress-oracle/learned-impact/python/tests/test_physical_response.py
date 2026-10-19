import json
import numpy as np
import pytest
import torch
from learned_impact.physical_response_model import PhysicalResponseImpactNet,inverse_response
from learned_impact.projectile_model import ProjectileResponseImpactNet
from learned_impact.response_history import SCALE
from learned_impact.visualize import load_checkpoint
from test_projectile_response import batch


def test_inverse_has_correct_zero_and_signed_derivatives_and_rejects_overflow():
    x=torch.zeros((3,18),dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[0].fill_(-.3);x[2].fill_(.2)
    y=inverse_response(x);y.sum().backward()
    np.testing.assert_allclose(y.detach().numpy(),np.sign(x.detach().numpy())*np.expm1(np.abs(x.detach().numpy()))*SCALE)
    np.testing.assert_allclose(x.grad.numpy(),np.exp(np.abs(x.detach().numpy()))*SCALE)
    assert torch.autograd.gradcheck(inverse_response,(x,))
    with pytest.raises(ValueError,match='finite training'):inverse_response(torch.full((1,18),100.))


def test_physical_scaling_cohorts_units_and_inference_roundtrip(tmp_path):
    b=batch();b['bond_response_alive_end'][0]=False
    b['bond_response_encoded_end'].fill_(np.log(2));b['bond_response_encoded_end'][0]=np.log(11)
    torch.manual_seed(3);old=ProjectileResponseImpactNet(h=8,p=4)
    torch.manual_seed(3);model=PhysicalResponseImpactNet(h=8,p=4)
    old.fit_target_scales([('a','wall',0,old.prepare_batch(b))])
    model.fit_target_scales([('a','wall',0,model.prepare_batch(b))])
    np.testing.assert_allclose(model.physical_response_cohort_rms[0].numpy(),SCALE,rtol=1e-6)
    np.testing.assert_allclose(model.physical_response_cohort_rms[1].numpy(),10*SCALE,rtol=1e-6)
    pred=model(b)
    for k,v in old(b).items():torch.testing.assert_close(pred[k],v,rtol=0,atol=0)
    loss=model.response_loss(pred,b)['loss'];assert loss.item()==pytest.approx(1.)
    loss.backward();assert model.response_extra.bias.grad[6]<0
    with torch.no_grad():model.response_extra.bias[6]=-1.
    model.zero_grad();model.response_loss(model(b),b)['loss'].backward()
    assert model.response_extra.bias.grad[6]<0
    torch.save(model.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':model.architecture,'threshold':model.threshold}))
    restored,_=load_checkpoint(tmp_path)
    torch.testing.assert_close(restored.response_loss(restored(b),b)['loss'],model.response_loss(model(b),b)['loss'],rtol=0,atol=0)


def test_small_energy_constant_fields_missing_cohort_and_target_causality():
    b=batch();b['bond_response_encoded_end'][:,12]=1e-7
    m=PhysicalResponseImpactNet(h=8,p=4);m.fit_target_scales([('a','wall',0,m.prepare_batch(b))])
    assert m.physical_response_cohort_rms[0,12].item()==pytest.approx(1e-4)
    torch.testing.assert_close(m.physical_response_cohort_rms[0,:12],torch.ones(12))
    torch.testing.assert_close(m.physical_response_cohort_rms[0],m.physical_response_cohort_rms[1])
    pred=m(b);poison={**b,'bond_response_encoded_end':torch.full_like(b['bond_response_encoded_end'],float('nan'))}
    for k,v in pred.items():torch.testing.assert_close(m(poison)[k],v,rtol=0,atol=0)
    assert m.response_loss(pred,b)['loss'].item()==pytest.approx(1/18)
