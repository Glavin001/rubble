"""Display active endpoint responses, keeping retired caches out of force maps."""
import numpy as np


def comparison(truth, predicted, previous, alive, endpoints, chunks):
    truth,predicted,previous=(np.asarray(x,dtype=np.float64) for x in (truth,predicted,previous))
    alive=np.asarray(alive,dtype=bool);endpoints=np.asarray(endpoints,dtype=np.int64)
    if any(x.shape!=(len(alive),18) or not np.isfinite(x).all() for x in (truth,predicted,previous)):
        raise ValueError('response view requires aligned finite 18-field responses')
    if endpoints.shape!=(len(alive),2) or (endpoints<0).any() or (endpoints>=chunks).any():
        raise ValueError('response view endpoints outside chunk range')
    valid=np.zeros(chunks,dtype=bool);valid[endpoints[alive].ravel()]=True
    def spread(values):
        out=np.zeros(chunks)
        np.maximum.at(out,endpoints[alive].ravel(),np.repeat(values[alive],2))
        return [float(v) if ok else None for v,ok in zip(out,valid)]
    def rmse(a,b):
        return float(np.sqrt(np.mean(np.square(a[alive]-b[alive])))) if alive.any() else None
    result={'valid':valid.tolist(),'active_bonds':int(alive.sum()),'retired_caches_excluded':int((~alive).sum())}
    for field,columns in [('force',slice(0,3)),('energy',12)]:
        t,p,s=truth[:,columns],predicted[:,columns],previous[:,columns]
        scalar=lambda a:np.linalg.norm(a,axis=1) if a.ndim==2 else np.abs(a)
        result[field]={'oracle':spread(scalar(t)),'model':spread(scalar(p)),
                       'error':spread(scalar(p-t)),
                       'rmse':rmse(p,t),'zero_rmse':rmse(np.zeros_like(t),t),'persistence_rmse':rmse(s,t)}
    return result
