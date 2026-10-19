import pytest
import torch
from learned_impact.losses import fracture_cdf_squared_error
from learned_impact.physics import PhysicsImpactNet,physics_loss
from learned_impact.dataset import synthetic_batch


def confident(classes):
    classes=torch.tensor(classes)
    break_logit=torch.where(classes>0,100.,-100.)
    time_logits=torch.full((len(classes),4),-100.)
    time_logits[torch.arange(len(classes)),(classes-1).clamp_min(0)]=100.
    return break_logit,time_logits


def test_score_respects_order_censoring_and_perfect_predictions():
    actual=torch.arange(5)
    b,t=confident(actual.tolist())
    assert fracture_cdf_squared_error(b,t,actual).sum()==0
    b,t=confident([2,4,0,1])
    errors=fracture_cdf_squared_error(b,t,torch.tensor([1,1,1,0])).mean(-1)
    torch.testing.assert_close(errors,torch.tensor([.25,.75,1.,1.]))
    # A late miss against an intact interval differs from an early false break.
    b,t=confident([1,4]);s=fracture_cdf_squared_error(b,t,torch.tensor([0,0])).mean(-1)
    torch.testing.assert_close(s,torch.tensor([1.,.25]))


def test_distribution_score_is_minimized_at_observed_distribution():
    labels=torch.tensor([0]*40+[1]*10+[2]*20+[3]*10+[4]*20)
    # Same probabilistic forecast for every sample. Expected score has zero
    # gradient at the empirical distribution, including the no-break mass.
    b=torch.full((100,),torch.logit(torch.tensor(.6)).item(),requires_grad=True)
    t=torch.tensor([1/6,2/6,1/6,2/6]).log().repeat(100,1).requires_grad_()
    score=fracture_cdf_squared_error(b,t,labels).mean()
    gb,gt=torch.autograd.grad(score,(b,t))
    assert gb.sum().abs()<1e-7
    assert gt.sum(0).abs().max()<1e-7
    altered=fracture_cdf_squared_error(b.detach()+1,t.detach(),labels).mean()
    assert altered>score


def test_optional_objective_preserves_baseline_and_handles_empty_graph():
    batch=synthetic_batch();model=PhysicsImpactNet(h=8,p=4);pred=model(batch)
    base=physics_loss(pred,batch)
    expected=base['fail']+.15*base['timing']+.3*base['util']+.1*base['delta']+.05*(base['joint']+base['force'])
    torch.testing.assert_close(base['loss'],expected,rtol=0,atol=0)
    augmented=physics_loss(pred,batch,fracture_cdf_weight=.3)
    torch.testing.assert_close(augmented['loss'],base['loss']+.3*base['fracture_cdf'])
    augmented['loss'].backward()
    assert torch.isfinite(model.time_head.weight.grad).all()
    b=torch.empty(0,requires_grad=True);t=torch.empty(0,4,requires_grad=True)
    errors=fracture_cdf_squared_error(b,t,torch.empty(0,dtype=torch.int64))
    assert errors.shape==(0,4)
