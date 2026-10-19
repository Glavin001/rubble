import numpy as np
import pytest
from learned_impact.causal_state import augment
from learned_impact.causal_state import half_turn_rows, sample_case, HALF_TURN_INPUT, HALF_TURN_OUTPUT

def test_physical_features_are_finite_at_rest_and_track_loading_and_plastic_history():
    x=np.zeros((2,69));x[:,33:39]=1e6
    x[:,49]=2e6;x[:,50]=20e6;x[:,51]=3e6;x[:,52]=.6
    x[:,54:57]=100;x[:,59]=1;x[:,60]=.01
    a=augment(x)
    assert a.shape==(2,79) and np.isfinite(a).all()
    np.testing.assert_array_equal(a[:,69:],0)
    x[1,64]=4e6;x[1,65]=1.5e6;x[1,14]=1.5;x[1,0]=.002;x[1,19]=.0004
    b=augment(x)
    assert b[1,69]==2 and b[1,70]==.5 and b[1,72]==.5
    assert np.isclose(b[1,76],.0016)
    np.testing.assert_array_equal(x,b[:,:69])
    event=augment(x,events=True)
    assert event.shape==(2,81) and np.isfinite(event).all()
    np.testing.assert_allclose(event[:,79],np.maximum(b[:,69],b[:,70])-np.maximum(b[:,75],1))
    np.testing.assert_allclose(event[:,80],event[:,79]/np.maximum(b[:,75],1))

def test_half_turn_is_involution_and_preserves_scalar_physics_features():
    rng=np.random.default_rng(11)
    x=rng.normal(size=(16,69));x[:,33:39]=1e6
    x[:,49:52]=1e7;x[:,53]=0;x[:,54:57]=100;x[:,59]=1;x[:,60]=.01
    base=rng.normal(size=(16,42));target=rng.normal(size=(16,42))
    turned=half_turn_rows(x,base,target)
    for actual,expected in zip(half_turn_rows(*turned),(x,base,target)):
        np.testing.assert_array_equal(actual,expected)

    a,b=augment(x,True),augment(turned[0],True)
    columns=HALF_TURN_INPUT+[76,77]
    expected=a.copy();expected[:,columns]*=-1
    np.testing.assert_allclose(b,expected)
    # Verify the batch operation after nonzero-mean asinh normalization.
    xs=np.maximum(np.sqrt((a*a).mean(0)),1e-12)
    z=np.arcsinh(a/xs);mean=z.mean(0);std=np.maximum(z.std(0),.05)
    normalized=(z-mean)/std
    normalized[:,columns]=-normalized[:,columns]-2*mean[columns]/std[columns]
    np.testing.assert_allclose(normalized,(np.arcsinh(b/xs)-mean)/std,atol=1e-12)
    for actual,original in zip(turned[1:],(base,target)):
        expected=original.copy();expected[:,HALF_TURN_OUTPUT]*=-1
        np.testing.assert_array_equal(actual,expected)

def test_capped_sampling_keeps_damage_transitions_but_not_all_already_broken_rows():
    x=np.arange(100)[:,None];base=np.zeros((100,42));target=base.copy();gates=np.zeros((100,29))
    gates[7,0]=1;gates[19,1]=1;gates[83,21]=1
    gates[90:,21]=1;base[90:,0]=1
    row=(x,base,target,gates)
    a=sample_case(row,10,np.random.default_rng(4),True)
    b=sample_case(row,10,np.random.default_rng(4),True)
    assert len(np.unique(a[0]))==10
    assert {7,19,83}<=set(a[0][:,0])
    for actual,expected in zip(a,b):np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError,match="smaller than 3"):
        sample_case(row,2,np.random.default_rng(4),True)
    assert sample_case(row,100,np.random.default_rng(4),True) is row
