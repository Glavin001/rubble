"""Integration contracts: new observable dynamics, unchanged response semantics."""
import json
import numpy as np
import pytest
import torch
from learned_impact.projectile_model import ProjectileResponseImpactNet
from learned_impact.response_model import ResponseHistoryImpactNet
from learned_impact.state_consistency import joint_connected
from learned_impact.visualize import load_checkpoint
from test_projectile_inputs import dynamic_batch
from test_response_model import batch as response_batch


def batch():
    return {**response_batch(), **{k:v for k,v in dynamic_batch().items() if k.startswith('impactor_')}}


def test_initial_response_parity_observable_spin_and_checkpoint(tmp_path):
    b=batch();torch.manual_seed(31);base=ResponseHistoryImpactNet(h=8,p=4)
    torch.manual_seed(31);model=ProjectileResponseImpactNet(h=8,p=4)
    for k,v in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    for k,v in base(b).items():
        torch.testing.assert_close(model(b)[k],v,rtol=0,atol=0)
    model(b)['break_logit'].sum().backward()
    assert model.dynamic_projectile_encoder.net[-1].weight.grad.abs().sum()>0
    with torch.no_grad():
        torch.nn.init.normal_(model.dynamic_projectile_encoder.net[-1].weight,std=.1)
    pred=model(b)
    flipped={**b,'impactor_angular_velocity':-b['impactor_angular_velocity']}
    assert not torch.equal(model(flipped)['break_logit'],pred['break_logit'])
    torch.save(model.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':model.architecture,'threshold':model.threshold}))
    restored,_=load_checkpoint(tmp_path)
    for k,v in pred.items():torch.testing.assert_close(restored(b)[k],v,rtol=0,atol=0)
    with pytest.raises(ValueError,match='requires impactor_angular_velocity'):
        model({k:v for k,v in b.items() if k!='impactor_angular_velocity'})


def test_causal_response_and_exact_connectivity_contracts():
    b=batch();model=ProjectileResponseImpactNet(h=8,p=4)
    b['bond_has_rebar'][1:3]=True
    b['bond_damage_complete_start'][2]=True;b['bond_exact_history'][2,0]=np.log(2)
    b['bond_response_encoded'][:,13]=-.2
    for value in (-10.,10.):
        with torch.no_grad():
            model.break_head.weight.zero_();model.break_head.bias.fill_(value)
            model.exact_delta.bias[0]=10.
        pred=model(b);state=model.decode_joint_state(pred,b)
        connected=joint_connected(state['values'][:,0],state['rebar_broken'],b['bond_has_rebar'].numpy())
        np.testing.assert_array_equal(connected,b['bond_failure_eligible'].numpy()&~(model.predict_classes(pred).numpy()>0))
        torch.testing.assert_close(pred['bond_force_end'],pred['response_encoded'][:,:6],rtol=0,atol=0)
        assert (model.decode_response(pred)[:,13]<0).all()
        poisoned={k:(torch.full_like(v,float('nan')) if k.endswith('_end') and v.dtype.is_floating_point else
                     ~v if k.endswith('_end') and v.dtype==torch.bool else v) for k,v in b.items()}
        for k,v in pred.items():torch.testing.assert_close(model(poisoned)[k],v,rtol=0,atol=0)
