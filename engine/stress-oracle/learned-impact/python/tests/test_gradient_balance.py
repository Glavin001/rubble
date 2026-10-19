import pytest
import torch
from learned_impact.gradient_balance import projected_backward


def test_conflicting_auxiliary_component_removed_without_changing_private_head():
    shared=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
    head=torch.nn.Parameter(torch.tensor(3.,dtype=torch.float64))
    primary=2*shared[0]
    auxiliary=-3*shared[0]+4*shared[1]+head.square()
    report=projected_backward(primary+auxiliary,primary,[shared])
    torch.testing.assert_close(shared.grad,torch.tensor([2.,4.],dtype=torch.float64))
    torch.testing.assert_close(head.grad,torch.tensor(6.,dtype=torch.float64))
    assert report['projected'] and report['projection_coefficient']==1.5


def test_aligned_orthogonal_zero_and_unused_primary_gradients_preserve_total():
    for coefficients in ((3.,4.),(0.,4.),(0.,0.)):
        shared=torch.nn.Parameter(torch.tensor([1.,2.]))
        unused=torch.nn.Parameter(torch.tensor(2.))
        primary=2*shared[0];auxiliary=coefficients[0]*shared[0]+coefficients[1]*shared[1]+unused.square()
        report=projected_backward(primary+auxiliary,primary,[shared,unused])
        torch.testing.assert_close(shared.grad,torch.tensor([2+coefficients[0],coefficients[1]]))
        assert unused.grad==4 and not report['projected']
    shared=torch.nn.Parameter(torch.tensor(1.));primary=shared*0
    report=projected_backward(shared.square(),primary,[shared])
    assert shared.grad==2 and not report['projected']
    with pytest.raises(ValueError,match='zero_grad'):
        projected_backward(shared.square(),shared,[shared])
    shared.grad=None;head=torch.nn.Parameter(torch.tensor(1.))
    report=projected_backward(head.square(),shared,[shared])
    assert shared.grad==1 and head.grad==2 and report['projected']
