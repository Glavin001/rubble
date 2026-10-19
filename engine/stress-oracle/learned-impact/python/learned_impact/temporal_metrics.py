"""Separate fracture location mistakes from timing mistakes on recorded clips.

Predictions still reset to oracle state each interval. Taking their union is a
diagnostic on that fixed trajectory, never an autonomous fracture rollout.
"""
from __future__ import annotations
import numpy as np
import torch
from .visualize import counts_metrics


class FractureTimeline:
    def __init__(self):
        self.truth={};self.predicted={};self.dt={};self.events=set()

    def add(self,event,frame,dt,bond_ids,truth_classes,predicted_classes):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        ids=np.asarray(bond_ids);actual=np.asarray(truth_classes);pred=np.asarray(predicted_classes)
        if ids.ndim!=1 or actual.shape!=ids.shape or pred.shape!=ids.shape or len(set(ids.tolist()))!=len(ids):
            raise ValueError('aligned unique bond IDs required')
        if not np.isin(actual,np.arange(5)).all() or not np.isin(pred,np.arange(5)).all():raise ValueError('invalid fracture time bin')
        if event in self.dt and self.dt[event]!=dt:raise ValueError('event dt changed')
        self.dt[event]=dt;self.events.add(event)
        for bond,a,p in zip(ids,actual,pred):
            key=(event,int(bond))
            if a:
                if key in self.truth:raise ValueError('oracle bond failed more than once')
                self.truth[key]=(frame+(int(a)-.5)/4)*dt
            if p:
                time=(frame+(int(p)-.5)/4)*dt
                self.predicted[key]=min(self.predicted.get(key,float('inf')),time)

    def report(self):
        actual=set(self.truth);pred=set(self.predicted);shared=actual&pred
        errors=np.array([self.predicted[k]-self.truth[k] for k in shared])
        fractions=np.array([(self.predicted[k]-self.truth[k])/self.dt[k[0]] for k in shared])
        return {'events':len(self.events),'clip_union':counts_metrics(len(shared),len(pred-actual),len(actual-pred)),
                'shared_bonds':len(shared),'median_signed_first_break_error_ms':float(np.median(errors)*1000) if len(errors) else None,
                'median_absolute_first_break_error_ms':float(np.median(np.abs(errors))*1000) if len(errors) else None,
                'first_break_more_than_one_interval_early':int((fractions < -1-1e-7).sum()),
                'first_break_more_than_one_interval_late':int((fractions > 1+1e-7).sum()),
                'within_one_interval_fraction':float((np.abs(fractions)<=1+1e-7).mean()) if len(errors) else None,
                'scope':'First predicted failure and oracle failure per bond over the recorded clip, using bin midpoint times. Oracle state/topology reset each interval; no rollout.',
                'limits':['Clip-union agreement forgives timing errors; it does not establish stable independent motion or fracture.',
                          'Oracle-failed bonds are absent from subsequent inputs, censoring later predictions.',
                          'A predicted break outside the recorded horizon cannot be assessed.']}


@torch.inference_mode()
def evaluate_timeline(model,records):
    timeline=FractureTimeline();model.eval()
    for name,_,frame,batch in records:
        output=model(batch)
        timeline.add(name,frame,float(batch['dt']),batch['bond_ids'].numpy(),
                     batch['bond_fail_class'].numpy(),model.predict_classes(output).numpy())
    return timeline.report()
