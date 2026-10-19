import copy,json
import numpy as np
import pytest
import torch
from learned_impact.consistent_model import ConsistentHistoryImpactNet
from learned_impact.history_model import ResidualHistoryImpactNet
from learned_impact.state_consistency import joint_connected
from learned_impact.visualize import load_checkpoint
from test_history_model import history_batch


def batch():
    b=history_batch();n=len(b['bond_endpoints'])
    b.update(bond_has_rebar=torch.zeros(n,dtype=torch.bool),bond_damage_complete_start=torch.zeros(n,dtype=torch.bool),bond_damage_complete_end=torch.zeros(n,dtype=torch.bool))
    return b


def test_connectivity_consistent_across_rebar_states_and_head_decisions():
    b=batch();m=ConsistentHistoryImpactNet(h=8,p=4)
    b['bond_has_rebar'][1:4]=True
    # Reinforced fully cracked, reinforced partial with ruptured tie, and an
    # already disconnected active interface. All three must stay irreversible.
    b['bond_damage_complete_start'][2]=True;b['bond_exact_history'][2,0]=np.log(2)
    b['bond_exact_history'][3,23]=1
    b['bond_damage_complete_start'][4]=True;b['bond_exact_history'][4,0]=np.log(2)
    b['bond_failure_eligible'][4]=False;b['bond_exact_history'][4,30]=0
    for break_logit in (-10.,10.):
        for crack_logit in (-10.,10.):
            for rebar_logit in (-10.,10.):
                with torch.no_grad():
                    for head,value in [(m.break_head,break_logit),(m.material_crack,crack_logit),(m.rebar_break,rebar_logit)]:
                        head.weight.zero_();head.bias.fill_(value)
                    m.exact_delta.bias[0]=10. # Try to overshoot complete damage.
                pred=m(b);d=m.decode_joint_state(pred,b)
                connected=joint_connected(d['values'][:,0],d['rebar_broken'],b['bond_has_rebar'].numpy())
                expected=b['bond_failure_eligible'].numpy()&~(m.predict_classes(pred).numpy()>0)
                np.testing.assert_array_equal(connected,expected)
                assert d['values'][2,0]==1 and d['values'][4,0]==1
                assert d['rebar_broken'][3] # No healing to retain connectivity.


def test_conditioned_damage_gradient_and_material_crack_head():
    b=batch();m=ConsistentHistoryImpactNet(h=8,p=4)
    end=b['bond_exact_history_end'];end[0,0]=np.log(2);b['bond_damage_complete_end'][0]=True
    end[1,0]=.2
    b['bond_has_rebar'][2]=True;end[2,0]=np.log(2);b['bond_damage_complete_end'][2]=True
    pred=m(b);raw=pred['exact_history_residual'];raw.retain_grad()
    m.state_loss(pred,b)['loss'].backward()
    assert raw.grad[0,0]==0 and raw.grad[2,0]==0
    assert raw.grad[1,0]<0
    assert m.material_crack.weight.grad.abs().sum()>0


def test_same_initial_fracture_predictions_and_causal_checkpoint_roundtrip(tmp_path):
    b=batch();torch.manual_seed(12);base=ResidualHistoryImpactNet(h=8,p=4)
    torch.manual_seed(12);m=ConsistentHistoryImpactNet(h=8,p=4)
    torch.testing.assert_close(base(b)['break_logit'],m(b)['break_logit'],rtol=0,atol=0)
    expected=m(b)
    poison={**b,'bond_damage_complete_end':~b['bond_damage_complete_end'],'bond_exact_history_end':torch.full_like(b['bond_exact_history_end'],float('nan'))}
    for k,v in expected.items():torch.testing.assert_close(m(poison)[k],v,rtol=0,atol=0)
    torch.save(m.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':m.architecture,'threshold':m.threshold}))
    loaded,_=load_checkpoint(tmp_path)
    for k,v in expected.items():torch.testing.assert_close(loaded(b)[k],v,rtol=0,atol=0)
    with pytest.raises(ValueError,match='explicit'):
        m({k:v for k,v in b.items() if k!='bond_has_rebar'})


def test_open_damage_boundary_survives_float32_rounding():
    b=batch();b['bond_exact_history'][0,0]=np.log(2) # rounded incomplete state
    m=ConsistentHistoryImpactNet(h=8,p=4)
    with torch.no_grad():m.break_head.weight.zero_();m.break_head.bias.fill_(-100.)
    d=m.decode_joint_state(m(b),b)
    assert d['values'][0,0]==np.nextafter(1.,0.)
