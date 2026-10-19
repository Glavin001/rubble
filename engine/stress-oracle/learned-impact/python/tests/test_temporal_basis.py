import numpy as np
from learned_impact.temporal_basis import hat_basis,fit_prefix

def test_basis_retains_constants_linear_motion_and_full_trace():
    times=np.linspace(0,.001,149);values=np.column_stack([np.ones(149),3*times-2,np.sin(times*80000)])
    fitted,_=fit_prefix(values,times,2)
    np.testing.assert_allclose(fitted[:,:2],values[:,:2],rtol=0,atol=1e-14)
    assert np.linalg.norm(fitted[:,2]-values[:,2])>1
    exact,_=fit_prefix(values,times,149)
    np.testing.assert_allclose(exact,values,rtol=0,atol=1e-14)

def test_exact_retirement_boundary_does_not_extend_targets_past_retirement():
    times=np.arange(20,dtype=float);values=np.column_stack([times[:5],times[:5]**2])
    fitted,count=fit_prefix(values,times,20)
    assert count==5 and fitted.shape==values.shape
    np.testing.assert_allclose(fitted,values,rtol=0,atol=1e-14)
    np.testing.assert_allclose(hat_basis(times,[0.,19.]).sum(1),1.)
