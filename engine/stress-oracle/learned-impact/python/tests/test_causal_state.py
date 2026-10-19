import numpy as np
from learned_impact.causal_state import augment

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
