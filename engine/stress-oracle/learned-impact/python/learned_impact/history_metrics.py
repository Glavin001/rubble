"""Diagnostics for the full JointState head, including categorical failures."""
import numpy as np
from .joint_history import FIELDS


class ExactHistoryMetrics:
    def __init__(self):
        self.squared=np.zeros(21);self.baseline=np.zeros(21);self.count=np.zeros(21,dtype=np.int64)
        self.unset=np.zeros((2,2,2),dtype=np.int64)
        self.rebar=np.zeros((2,2),dtype=np.int64);self.mode=np.zeros((5,5),dtype=np.int64)
        self.bonds=0;self.cracked=0

    def add(self,pred,batch):
        def array(t):return t.detach().cpu().numpy()
        start=array(batch['bond_exact_history']);end=array(batch['bond_exact_history_end'])
        actual=array(pred['exact_history_encoded']).astype(np.float64)
        if not np.isfinite(actual).all():raise ValueError('nonfinite full-history prediction')
        valid=np.ones_like(actual,dtype=bool);valid[:,4:6]=end[:,21:23]==0
        target=end[:,:21].astype(np.float64)
        self.squared+=(np.square(actual-target)*valid).sum(0)
        self.baseline+=(np.square(start[:,:21].astype(np.float64)-target)*valid).sum(0)
        self.count+=valid.sum(0)
        unset=(start[:,21:23]==1)&(array(pred['ductility_set_logits'])<0)
        for i in range(2):np.add.at(self.unset[i],(end[:,21+i].astype(int),unset[:,i].astype(int)),1)
        rebar=array(pred['rebar_broken']) if 'rebar_broken' in pred else (start[:,23]==1)|(array(pred['rebar_break_logit'])>=0)
        np.add.at(self.rebar,(end[:,23].astype(int),rebar.astype(int)),1)
        np.add.at(self.mode,(end[:,24:29].argmax(-1),array(pred['history_mode_logits']).argmax(-1)),1)
        self.bonds+=len(end);self.cracked+=int((~array(batch['bond_failure_eligible'])).sum())

    def report(self):
        def rmse(sums):return {k:float(np.sqrt(s/n)) if n else None for k,s,n in zip(FIELDS,sums,self.count)}
        return {'bond_intervals':self.bonds,'already_cracked_active_bond_intervals':self.cracked,
                'encoded_rmse':rmse(self.squared),'persistence_encoded_rmse':rmse(self.baseline),
                'finite_target_counts':dict(zip(FIELDS,self.count.tolist())),
                'ductility_unset_confusion':self.unset.tolist(),'rebar_broken_confusion':self.rebar.tolist(),
                'mode_confusion':self.mode.tolist(),
                'confusion_axes':'truth row, prediction column; unset/rebar flags 0 or 1; mode none/tension/shear/compression/buckling',
                'scope':'All runtime-alive bonds, including previously disconnected interfaces. Finite-head encoded errors are scored where targets are finite; unset classification is reported separately. Oracle start state each interval, no independent rollout.'}
