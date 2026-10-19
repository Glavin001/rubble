import numpy as np
import pytest
import torch
from learned_impact.fragment_loss import fragment_cut_targets,fragment_cut_loss


def test_fragment_boundaries_exclude_redundant_cracks_and_old_cracks():
    # Triangle stays connected after edge 0 cracks; edges 3,4 together detach
    # the fourth vertex. Edge 5 was cracked already and is not a new target.
    edges=np.array([[0,1],[1,2],[2,0],[1,3],[2,3],[0,3]])
    eligible=np.array([1,1,1,1,1,0],bool);end=np.array([0,1,1,0,0,0],bool)
    groups=fragment_cut_targets(4,edges,eligible,end,np.array([1,0,0,2,3,0]))
    assert groups.tolist()==[-1,-1,-1,0,0,-1]
    with pytest.raises(ValueError,match='disagrees'):
        fragment_cut_targets(4,edges,eligible,end,np.zeros(6,int))


def test_weakest_bridge_gets_gradient_without_rewarding_unrelated_breaks():
    logits=torch.tensor([4.,-2.,-5.,.5],requires_grad=True)
    groups=torch.tensor([0,0,-1,1])
    loss=fragment_cut_loss(logits,groups)
    expected=(torch.nn.functional.softplus(torch.tensor(2.))+torch.nn.functional.softplus(torch.tensor(-.5)))/2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert logits.grad[0]==0 and logits.grad[2]==0
    assert logits.grad[1]<0 and logits.grad[3]<0


def test_ties_extremes_empty_and_permutation_invariance():
    logits=torch.tensor([-1000.,-1000.,1000.],requires_grad=True)
    groups=torch.tensor([0,0,1])
    loss=fragment_cut_loss(logits,groups);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0]==logits.grad[1]<0
    torch.testing.assert_close(loss,fragment_cut_loss(logits[[2,0,1]],groups[[2,0,1]]))
    for size in (0,3):
        x=torch.zeros(size,requires_grad=True)
        zero=fragment_cut_loss(x,torch.full((size,),-1,dtype=torch.int64));zero.backward()
        assert zero==0 and torch.equal(x.grad,torch.zeros(size))
