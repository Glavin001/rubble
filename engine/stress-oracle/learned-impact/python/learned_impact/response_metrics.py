"""Cached-response errors split by continuing interfaces and retired caches."""
import numpy as np
from .response_history import FIELDS,SCALE


class ResponseMetrics:
    def __init__(self):
        self.count=np.zeros(2,dtype=np.int64)
        self.squared=np.zeros((2,3,18));self.physical=np.zeros_like(self.squared)

    def add(self,pred,batch):
        def array(t):return t.detach().cpu().numpy().astype(np.float64)
        end=array(batch['bond_response_encoded_end']);start=array(batch['bond_response_encoded']);actual=array(pred['response_encoded'])
        alive=batch['bond_response_alive_end'].detach().cpu().numpy()
        def decode(a):
            with np.errstate(over='ignore',invalid='ignore'):v=np.sign(a)*np.expm1(np.abs(a))*SCALE
            if not np.isfinite(v).all():raise ValueError('nonfinite cached response metrics')
            return v
        target=decode(end)
        for j,estimate in enumerate((actual,start,np.zeros_like(start))):
            error=np.square(estimate-end);physical=np.square(decode(estimate)-target)
            if not np.isfinite(error).all() or not np.isfinite(physical).all():raise ValueError('response metric accumulation overflow')
            for c,mask in enumerate((alive,~alive)):
                self.squared[c,j]+=error[mask].sum(0);self.physical[c,j]+=physical[mask].sum(0)
        self.count+=np.array([alive.sum(),(~alive).sum()])

    def report(self):
        result={}
        for c,name in enumerate(('active_end','retired_cache')):
            result[name]={'bond_intervals':int(self.count[c])}
            for j,method in enumerate(('model','persistence','zero')):
                n=self.count[c]
                result[name][method]={'encoded_rmse':{field:float(np.sqrt(v/n)) if n else None for field,v in zip(FIELDS,self.squared[c,j])},
                    'si_component_rmse':{field:float(np.sqrt(v/n)) if n else None for field,v in zip(FIELDS,self.physical[c,j])}}
        result['scope']='Oracle-start cached endpoint response. Retired values are last caches, not continuing forces. SI errors decode float32 signed-log targets; exact f64 source responses remain on disk. Independent response fields do not enforce force/stress/energy constitutive consistency or provide a World adapter.'
        return result
