import numpy as np
import pytest
from learned_impact.rollout_view import first_breaks,timing_scores,fragments

def test_delayed_break_is_matched_only_with_declared_tolerance():
    oracle=first_breaks([[1,1,1],[0,1,1],[0,0,1],[0,0,1]],.01)
    model=first_breaks([[1,1,1],[1,1,0],[0,1,0],[0,1,0]],.01)
    scores=timing_scores(oracle,model,[0,.01])
    assert (scores[0]['tp'],scores[0]['fp'],scores[0]['fn'])==(0,2,2)
    assert (scores[1]['tp'],scores[1]['fp'],scores[1]['fn'])==(1,1,1)
    assert np.isinf(oracle[2]) and np.isinf(model[1])
    with pytest.raises(ValueError,match='healed'):first_breaks([[1],[0],[1]],.01)

def test_fragment_statistics_use_each_world_own_active_mask_and_mass():
    mass=np.array([1.,2.,3.]);a={'active':[1,1,0],'clusters':[8,8,9]};b={'active':[1,1,1],'clusters':[8,9,9]}
    assert fragments(a,mass)=={'count':1,'largest_mass_fraction':1.}
    assert fragments(b,mass)=={'count':2,'largest_mass_fraction':5/6}
