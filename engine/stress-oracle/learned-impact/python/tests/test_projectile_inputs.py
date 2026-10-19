import copy,json
import numpy as np
import pytest
import torch
from learned_impact.projectile_inputs import observations
from learned_impact.projectile_model import ProjectileHistoryImpactNet
from learned_impact.history_model import ResidualHistoryImpactNet
from learned_impact.visualize import load_checkpoint
from test_world_inputs import recording
from test_material_inputs import fixture
from test_history_model import history_batch


def enriched():
    replay,scene,context,manifest=recording()
    replay['world_state_format']='learned-world-state/1'
    material=fixture()[0]['materials']['m'];scene['materials']={'ram_material':material}
    scene['impactors'][0]['material']='ram_material'
    for i,f in enumerate(replay['frames']):
        f['impactors'][0].update(angular_velocity=[0.,float(i),.2],mass=200.,inertia=np.diag([10.,20.,30.]).tolist(),
                                 material=copy.deepcopy(material),crush=[1000.,40.],crush_used=50.*i,crush_depth=.01*i,driven=False)
    return replay,scene,context,manifest


def dynamic_batch():
    b=history_batch();state=observations(*enriched())[0]
    b.update({k:torch.from_numpy(v) for k,v in state.items()})
    return b


def test_projectile_inputs_are_current_state_and_keep_capacity_overshoot():
    args=enriched();before=observations(*args)
    assert before[1]['impactor_runtime_properties'][0,13]==-10
    assert before[1]['impactor_runtime_properties'][0,12]==50
    poison=copy.deepcopy(args);poison[0]['frames'][1]['impactors'][0]['angular_velocity']=[1000.,1000.,1000.]
    poison[0]['frames'][1]['impactors'][0]['crush_used']=1000.
    poison[0]['frames'][-1]['impactors']=[{'unusable':'future endpoint'}]
    after=observations(*poison)
    for k,v in before[0].items():np.testing.assert_array_equal(after[0][k],v)
    assert after[1]['impactor_runtime_properties'][0,13]==-960
    assert not np.array_equal(after[1]['impactor_angular_velocity'],before[1]['impactor_angular_velocity'])


def test_missing_mismatched_or_invalid_dynamic_state_fails_explicitly():
    args=enriched();del args[0]['world_state_format']
    with pytest.raises(ValueError,match='enriched'):observations(*args)
    for key in ('angular_velocity','inertia','crush_used','crush_depth','material','driven','mass','crush'):
        args=enriched();del args[0]['frames'][0]['impactors'][0][key]
        with pytest.raises(ValueError,match='missing dynamic'):observations(*args)
    for key,value,match in [('mass',201.,'mass'),('inertia',np.zeros((3,3)).tolist(),'inertia'),
                            ('crush_used',-1.,'history'),('crush_depth',np.nan,'history'),
                            ('driven',1,'boolean'),('crush',[1000.,30.],'law')]:
        args=enriched();args[0]['frames'][0]['impactors'][0][key]=value
        with pytest.raises(ValueError,match=match):observations(*args)
    args=enriched();args[0]['frames'][0]['impactors'][0]['material']['friction']=.1
    with pytest.raises(ValueError,match='material'):observations(*args)


def test_new_branch_preserves_initial_v12_and_learns_dynamic_inputs(tmp_path):
    b=dynamic_batch();torch.manual_seed(31);old=ResidualHistoryImpactNet(h=8,p=4)
    torch.manual_seed(31);m=ProjectileHistoryImpactNet(h=8,p=4)
    reference=old(b)
    for key,value in reference.items():torch.testing.assert_close(m(b)[key],value,rtol=0,atol=0)
    m(b)['break_logit'].sum().backward()
    assert m.dynamic_projectile_encoder.net[-1].weight.grad.abs().sum()>0
    with torch.no_grad():torch.nn.init.normal_(m.dynamic_projectile_encoder.net[-1].weight,std=.1)
    before=m(b)
    for key in ('impactor_angular_velocity','impactor_runtime_properties','impactor_material_features'):
        assert not torch.equal(m({**b,key:b[key]+.5})['break_logit'],before['break_logit']),key
    with pytest.raises(ValueError,match='requires impactor_angular_velocity'):m({k:v for k,v in b.items() if k!='impactor_angular_velocity'})
    torch.save(m.state_dict(),tmp_path/'state_dict.pt');(tmp_path/'run.json').write_text(json.dumps({'architecture':m.architecture,'threshold':m.threshold}))
    loaded,_=load_checkpoint(tmp_path)
    for key,value in before.items():torch.testing.assert_close(loaded(b)[key],value,rtol=0,atol=0)


def test_all_projectiles_contribute_without_order_or_empty_set_errors():
    b=dynamic_batch();m=ProjectileHistoryImpactNet(h=8,p=4)
    for key in b:
        if key.startswith('impactor_'):b[key]=b[key].repeat(2,*([1]*(b[key].ndim-1)))
    b['impactor_center'][1]+=1;b['impactor_angular_velocity'][1]+=2
    with torch.no_grad():torch.nn.init.normal_(m.dynamic_projectile_encoder.net[-1].weight,std=.1)
    original=m(b);permuted={k:v.flip(0) if k.startswith('impactor_') else v for k,v in b.items()}
    for key,value in original.items():torch.testing.assert_close(m(permuted)[key],value,rtol=1e-5,atol=1e-6)
    empty={k:v[:0] if k.startswith('impactor_') else v for k,v in b.items()}
    assert all(torch.isfinite(v).all() for v in m(empty).values())
    args=enriched();args[1]['impactors']=[]
    for f in args[0]['frames']:f['impactors']=[]
    assert observations(*args)[0]['impactor_runtime_properties'].shape==(0,17)
