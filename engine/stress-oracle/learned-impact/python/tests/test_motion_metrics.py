import numpy as np
import pytest
from learned_impact.motion_metrics import MotionMetrics


def test_motion_strata_include_zero_baseline_and_exclude_resets():
    target=np.zeros((4,12));target[1,:3]=1;target[2,:3]=2;target[3,:3]=100
    metric=MotionMetrics(np.ones(12));metric.add(target,np.zeros_like(target),[1,1,1,0])
    r=metric.report();u=r['fields']['u_m']
    assert r['fragment_reset_excluded_chunk_intervals']==1
    assert u['target_vector_norm_threshold']==pytest.approx(np.sqrt(3))
    assert u['at_or_below_training_rms']['chunk_intervals']==2
    assert u['above_training_rms']=={'chunk_intervals':1,'rmse':2.,'zero_increment_rmse':2.}
    assert u['at_or_below_training_rms']['rmse']==pytest.approx(np.sqrt(.5))
    perfect=MotionMetrics(np.ones(12));perfect.add(target,target,[1,1,1,0])
    assert perfect.report()['fields']['u_m']['above_training_rms']['rmse']==0


def test_motion_strata_empty_is_unavailable_and_accumulation_matches():
    a=MotionMetrics(np.ones(12));b=MotionMetrics(np.ones(12))
    t=np.full((2,12),2.)
    a.add(t,t*.5,[1,1])
    b.add(t[:1],t[:1]*.5,[1]);b.add(t[1:],t[1:]*.5,[1])
    assert a.report()==b.report()
    assert a.report()['fields']['u_m']['at_or_below_training_rms']['rmse'] is None
    with pytest.raises(ValueError,match='positive finite'):MotionMetrics(np.zeros(12))
    with pytest.raises(ValueError,match='nonfinite'):a.add(t,t*np.nan,[1,1])
