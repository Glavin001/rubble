import json
import numpy as np
import pytest
import torch
from learned_impact.constitutive_model import ConstitutiveNativeImpactNet
from learned_impact.physics import ForcingNativeImpactNet
from learned_impact.visualize import load_checkpoint
from learned_impact.benchmark_physics import make_graph
from test_forcing_inputs import forced_batch


def constitutive_batch():
    b=forced_batch();n=len(b['chunk_mass']);e=len(b['bond_endpoints'])
    b.update(chunk_material_features=torch.linspace(0,1,n*26).reshape(n,26),
             bond_material_features=torch.linspace(0,1,e*26).reshape(e,26),
             constitutive_flags=torch.linspace(0,1,20),
             bond_exact_history=torch.linspace(0,1,e*31).reshape(e,31))
    return b


def test_initial_native_parity_and_checkpoint_round_trip(tmp_path):
    b=constitutive_batch();torch.manual_seed(14);old=ForcingNativeImpactNet(h=8,p=4)
    torch.manual_seed(14);new=ConstitutiveNativeImpactNet(h=8,p=4)
    expected=old(b)
    for key,value in expected.items():torch.testing.assert_close(new(b)[key],value,rtol=0,atol=0)
    torch.save(new.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':new.architecture,'threshold':.5}))
    restored,_=load_checkpoint(tmp_path)
    assert isinstance(restored,ConstitutiveNativeImpactNet)
    for key,value in expected.items():torch.testing.assert_close(restored(b)[key],value,rtol=0,atol=0)


def test_material_history_and_method_inputs_learn_without_target_leakage():
    b=constitutive_batch();torch.manual_seed(15);m=ConstitutiveNativeImpactNet(h=8,p=4)
    m(b)['break_logit'].sum().backward()
    for encoder in (m.constitutive_node_encoder,m.constitutive_edge_encoder):
        assert encoder.net[-1].weight.grad.abs().sum()>0
        with torch.no_grad():torch.nn.init.normal_(encoder.net[-1].weight,std=.1)
    reference=m(b)
    for key in ('chunk_material_features','bond_material_features','constitutive_flags','bond_exact_history'):
        changed={**b,key:b[key]+1}
        assert not torch.equal(m(changed)['break_logit'],reference['break_logit']),key
    # All targets and future interval inputs are deliberately nonsensical.
    keys=('bond_exact_history_end','bond_joint_state_end','bond_force_end','bond_fail_class',
          'bond_peak_util','chunk_state_delta','world_motion_target','world_motion_valid',
          'chunk_load_bins','chunk_contact','chunk_preview_load_bins','chunk_preview_contact','bond_qs_force','qs_mode')
    poisoned={**b,**{k:torch.full_like(b[k],999) if k in b else torch.tensor(float('nan')) for k in keys}}
    for key,value in reference.items():torch.testing.assert_close(m(poisoned)[key],value,rtol=0,atol=0)


def test_required_inputs_are_strict_and_synthetic_scale_graph_runs():
    b=constitutive_batch();m=ConstitutiveNativeImpactNet(h=8,p=4)
    for key in ('chunk_material_features','bond_material_features','constitutive_flags','bond_exact_history'):
        with pytest.raises(ValueError,match=key):m({k:v for k,v in b.items() if k!=key})
        with pytest.raises(ValueError,match=key):m({**b,key:torch.full_like(b[key],float('nan'))})
        with pytest.raises(ValueError,match=key):m({**b,key:b[key][:-1]})
    assert all(torch.isfinite(v).all() for v in m(make_graph(12,24,m)).values())


def test_material_bonds_follow_live_graph_filtering():
    from pathlib import Path
    from learned_impact.schema import load_event
    from learned_impact.dataset import event_batches
    ev=load_event(Path(__file__).resolve().parents[2]/'fixtures/tiny_event')
    n=ev.manifest['n_chunks'];e=ev.manifest['n_bonds']
    ev.material_inputs={'chunk_material_features':np.zeros((n,26),np.float32),
                        'bond_material_features':np.arange(e*26,dtype=np.float32).reshape(e,26),
                        'constitutive_flags':np.zeros(20,np.float32)}
    for batch in event_batches(ev):
        ids=batch['bond_ids'].numpy()
        np.testing.assert_array_equal(batch['bond_material_features'],ev.material_inputs['bond_material_features'][ids])
        assert batch['chunk_material_features'].shape==(n,26)
