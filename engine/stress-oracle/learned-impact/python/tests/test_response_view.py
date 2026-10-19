import json
import numpy as np
import pytest
from learned_impact.response_view import comparison


def test_vector_error_retired_cache_mask_and_shared_incident_maximum():
    truth=np.zeros((3,18));truth[:,0]=[3,0,1e9];truth[:,1]=[4,2,1e9];truth[:,12]=[.1,.2,1e9]
    pred=truth.copy();pred[0,:2]=[-3,-4];pred[1,12]=.3
    r=comparison(truth,pred,np.zeros_like(truth),[True,True,False],[[0,1],[1,2],[2,3]],4)
    assert r['force']['oracle']==[5,5,2,None]
    assert r['force']['model']==[5,5,2,None]
    assert r['force']['error']==[10,10,0,None] # Equal magnitude, reversed force.
    assert r['energy']['oracle']==[.1,.2,.2,None]
    assert r['energy']['error'][1]==pytest.approx(.1)
    assert r['force']['rmse']==pytest.approx(np.sqrt(100/6))
    assert r['force']['zero_rmse']==r['force']['persistence_rmse']
    assert r['retired_caches_excluded']==1 and r['valid']==[True,True,True,False]
    json.dumps(r,allow_nan=False)
    empty=comparison(truth,pred,truth,[False]*3,[[0,1],[1,2],[2,3]],4)
    assert empty['force']['rmse'] is None and empty['energy']['oracle']==[None]*4
