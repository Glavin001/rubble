import torch
from learned_impact.graph_state import GraphStateNet,external_delta

def fixture():
    torch.manual_seed(91)
    net=GraphStateNet(16,2)
    with torch.no_grad():
        net.head[-1].weight.normal_();net.head[-1].bias.normal_()
    x=torch.randn(5,39,dtype=torch.float64);x[:,34:37]=0;x[:,34]=1;x[:,37]=.001
    e=torch.randn(4,84,dtype=torch.float64)
    ends=torch.tensor([[0,1],[1,2],[1,3],[3,4]])
    return net,x,e,ends

def test_graph_prediction_is_node_permutation_equivariant_and_batch_separable():
    net,x,e,ij=fixture();expected=net(x,e,ij)
    order=torch.tensor([3,0,4,1,2]);inverse=torch.empty_like(order);inverse[order]=torch.arange(5)
    torch.testing.assert_close(net(x[order],e,inverse[ij]),expected[order],rtol=1e-5,atol=1e-5)
    both=net(torch.cat((x,x*1.2)),torch.cat((e,e*.7)),torch.cat((ij,ij+len(x))))
    torch.testing.assert_close(both[:len(x)],expected,rtol=1e-5,atol=1e-5)
    assert expected.abs().max()>0 # random nonzero head, not a vacuous zero-network check

def test_isolated_chunks_get_exact_external_motion_without_invented_internal_impulse():
    net,x,_,_=fixture()
    e=torch.empty((0,84),dtype=torch.float64);ij=torch.empty((0,2),dtype=torch.long)
    torch.testing.assert_close(net(x,e,ij),torch.zeros((len(x),6)),rtol=0,atol=0)
    torch.testing.assert_close(net.velocity_delta(x,e,ij),external_delta(x),rtol=0,atol=0)

def test_external_prior_uses_full_inertia_and_honors_fixed_and_pinned_supports():
    x=torch.zeros((3,39),dtype=torch.float64)
    x[:,12:18]=torch.tensor([2.,4.,6.,1.,2.,3.]);x[:,24]=.5;x[:,37]=.1
    inverse=torch.tensor([[2.,.5,0.],[.5,3.,0.],[0.,0.,4.]],dtype=torch.float64)
    x[:,25:34]=inverse.flatten();x[:,34:37]=torch.eye(3)
    actual=external_delta(x);expected=torch.tensor([[.1,.2,.3,.3,.65,1.2],[0.,0.,0.,.3,.65,1.2],[0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
    torch.testing.assert_close(actual,expected)
