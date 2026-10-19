import copy
import numpy as np
import pytest
from learned_impact.material_inputs import material_features,observations,FEATURE_FLAGS,METHOD_FLAGS


def fixture():
    m={'density':2400.,'youngs_modulus':3e10,'poisson_ratio':.2,'damping_ratio':.01,
       'tensile_strength':3e6,'compressive_strength':3e7,'cohesion':4.5e6,'friction':.75,
       'fracture_energy':{'tension':120.,'shear':1200.,'compression':20000.},
       'shear_cap':None,'weibull_modulus':None,'dif':None,'static_fatigue':None,'kind':'brittle'}
    sim={'features':dict.fromkeys(FEATURE_FLAGS,True),'methods':dict.fromkeys(METHOD_FLAGS,True),'stiffness_scale':1.,'fracture':True}
    scene={'name':'fixture','materials':{'m':m},'sim':sim,
           'bodies':[{'chunks':[{'material':'m'},{'material':'m'}],'bonds':[{'a':0,'b':1,'material':'m'}]}]}
    context={'source_scene':'fixture','chunk_keys':[[0,0],[0,1]],'bond_keys':[[0,0]],'solver_methods':sim['methods']}
    return scene,context,{'n_chunks':2,'n_bonds':1},{'bond_endpoints':np.array([[0,1]])}


def test_laws_hidden_by_legacy_strength_vector_remain_distinguishable():
    scene,_,_,_=fixture();m=scene['materials']['m'];baseline=material_features(m)
    for key,value in [('friction',.2),('shear_cap',0.),('kind','ductile'),('damping_ratio',.1),
                      ('dif',{'reference_rate':1e-6,'exponent':.1,'transition_rate':1.,'exponent_high':.3,'max':3.}),
                      ('static_fatigue',{'exponent':10.,'test_time':60.})]:
        changed={**m,key:value};encoded=material_features(changed)
        assert encoded.shape==(26,) and np.isfinite(encoded).all() and not np.array_equal(encoded,baseline)
        # The three recorded strength values are identical in every case.
        assert [changed[k] for k in ('tensile_strength','cohesion','compressive_strength')]==[m[k] for k in ('tensile_strength','cohesion','compressive_strength')]
    assert baseline[12]==0 and material_features({**m,'shear_cap':0.})[12]==1
    with pytest.raises(ValueError,match='invalid'):material_features({**m,'friction':np.nan})


def test_material_mapping_flags_and_source_order_are_explicit():
    s,c,m,a=fixture();r=observations(s,c,m,a)
    assert r['chunk_material_features'].shape==(2,26) and r['bond_material_features'].shape==(1,26)
    assert r['constitutive_flags'].shape==(20,)
    assert r['bond_has_rebar'].dtype==bool and not r['bond_has_rebar'][0]
    reinforced=copy.deepcopy(s);reinforced['bodies'][0]['bonds'][0]['rebar']={'area':.001,'material':'m'}
    assert observations(reinforced,c,m,a)['bond_has_rebar'][0]
    reinforced['sim']['features']['rebar']=False
    assert not observations(reinforced,c,m,a)['bond_has_rebar'][0]
    changed=copy.deepcopy(s);changed['sim']['features']['crushing']=False
    assert not np.array_equal(observations(changed,c,m,a)['constitutive_flags'],r['constitutive_flags'])
    changed['loads']=[{'future_only':'not a constitutive input'}]
    np.testing.assert_array_equal(observations(changed,c,m,a)['bond_material_features'],r['bond_material_features'])
    with pytest.raises(ValueError,match='misaligned'):observations(s,c,m,{'bond_endpoints':np.array([[1,0]])})
    bad=copy.deepcopy(c);bad['chunk_keys'][0]=[-1,0]
    with pytest.raises(ValueError,match='invalid'):observations(s,bad,m,a)
    bad=copy.deepcopy(c);bad['solver_methods']['exact_joint_patch']=False
    with pytest.raises(ValueError,match='methods mismatch'):observations(s,bad,m,a)
