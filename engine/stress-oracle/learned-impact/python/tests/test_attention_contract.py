import torch
from learned_impact.model import PhysicsAttention
from learned_impact.response_model import ResponseHistoryImpactNet
from learned_impact.physics import SizeStableAttention


def test_attention_replication_preserves_global_amplitude_and_permutation():
    torch.manual_seed(37);old=PhysicsAttention(8,4).double();new=SizeStableAttention(8,4).double()
    new.load_state_dict(old.state_dict());nodes=torch.randn(7,8,dtype=torch.float64)
    baseline=new(nodes);legacy=old(nodes)-nodes
    for copies in (2,10,1500):
        repeated=nodes.repeat(copies,1)
        torch.testing.assert_close(new(repeated),baseline.repeat(copies,1),rtol=1e-10,atol=1e-10)
        # Independent regression evidence: the old broadcast shrinks by 1/copies.
        torch.testing.assert_close(old(repeated)-repeated,legacy.repeat(copies,1)/copies,rtol=1e-9,atol=1e-10)
    order=torch.randperm(len(nodes))
    torch.testing.assert_close(new(nodes[order]),baseline[order],rtol=1e-12,atol=1e-12)


def test_attention_empty_slices_and_gradient_are_finite():
    torch.manual_seed(38);model=SizeStableAttention(8,4)
    with torch.no_grad():
        model.assign.weight.zero_();model.assign.bias.copy_(torch.tensor([0.,-1e4,-1e4,-1e4]))
    nodes=torch.randn(11,8,requires_grad=True);result=model(nodes)
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    assert torch.isfinite(nodes.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model(torch.empty(0,8)).shape==(0,8)


def test_current_response_model_uses_size_stable_attention():
    model=ResponseHistoryImpactNet(h=8,p=4)
    assert isinstance(model.attn,SizeStableAttention)
