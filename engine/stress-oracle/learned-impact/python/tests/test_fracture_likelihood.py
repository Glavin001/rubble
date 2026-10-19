import pytest
import torch
from learned_impact.losses import fracture_event_nll
from learned_impact.physics import PhysicsImpactNet, physics_loss
from learned_impact.dataset import synthetic_batch


def test_likelihood_matches_normalized_categorical_distribution_and_gradients():
    b=torch.tensor([-.8,2.,-1.,.5,0.],dtype=torch.float64,requires_grad=True)
    t=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
    labels=torch.arange(5)
    probability=torch.cat([(1-b.sigmoid())[:,None],b.sigmoid()[:,None]*t.softmax(-1)],-1)
    torch.testing.assert_close(probability.sum(-1),torch.ones_like(b))
    expected=-probability[torch.arange(5),labels].log()
    actual=fracture_event_nll(b,t,labels)
    torch.testing.assert_close(actual,expected)
    for a,e in zip(torch.autograd.grad(actual.sum(),(b,t),retain_graph=True),torch.autograd.grad(expected.sum(),(b,t))):
        torch.testing.assert_close(a,e)


def test_likelihood_stationary_at_empirical_distribution_and_handles_extremes():
    labels=torch.tensor([0]*40+[1]*10+[2]*20+[3]*10+[4]*20)
    b=torch.full((100,),torch.logit(torch.tensor(.6)).item(),requires_grad=True)
    t=torch.tensor([1/6,2/6,1/6,2/6]).log().repeat(100,1).requires_grad_()
    gb,gt=torch.autograd.grad(fracture_event_nll(b,t,labels).mean(),(b,t))
    assert gb.sum().abs()<1e-7 and gt.sum(0).abs().max()<1e-7
    b=torch.tensor([-1000.,1000.],requires_grad=True)
    t=torch.tensor([[1000.,-1000.],[-1000.,1000.]],requires_grad=True)
    loss=fracture_event_nll(b,t,torch.tensor([2,0])).sum()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss,(b,t)))


def test_objective_uses_eligible_population_and_masks_retired_fractures():
    batch=synthetic_batch();model=PhysicsImpactNet(h=8,p=4);pred=model(batch)
    eligible=torch.arange(len(batch['bond_fail_class']))%2==0
    batch['bond_failure_eligible']=eligible
    parts=physics_loss(pred,batch,positive_weight=1,fracture_objective='event-nll')
    expected=fracture_event_nll(pred['break_logit'][eligible],pred['time_logits'][eligible],batch['bond_fail_class'][eligible]).mean()
    torch.testing.assert_close(parts['fracture'],expected.detach())
    # Altering ineligible fracture labels cannot affect any component.
    altered={**batch,'bond_fail_class':batch['bond_fail_class'].clone()}
    altered['bond_fail_class'][~eligible]=4
    torch.testing.assert_close(parts['loss'],physics_loss(pred,altered,positive_weight=1,fracture_objective='event-nll')['loss'])
    batch['bond_failure_eligible']=torch.zeros_like(eligible)
    empty=physics_loss(pred,batch,positive_weight=1,fracture_objective='event-nll')
    assert empty['fracture']==0 and torch.isfinite(empty['loss'])
    empty['loss'].backward()
    with pytest.raises(ValueError,match='positive_weight'):
        physics_loss(pred,batch,fracture_objective='event-nll')


def test_empty_and_invalid_likelihood_inputs():
    assert fracture_event_nll(torch.empty(0),torch.empty(0,4),torch.empty(0,dtype=torch.int64)).shape==(0,)
    with pytest.raises(ValueError,match='range'):
        fracture_event_nll(torch.zeros(1),torch.zeros(1,4),torch.tensor([5]))
