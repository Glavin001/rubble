"""Causal cached-response inputs and complete 18-field response prediction.

This extends the consistent joint-state model, not a complete World adapter.
Retired endpoint values are the last cached response, never continuing forces.
"""
import numpy as np
import torch
from torch import nn
from .consistent_model import ConsistentHistoryImpactNet
from .model import MLP
from .response_history import FORMAT,SCALE


class ResponseHistoryImpactNet(ConsistentHistoryImpactNet):
    architecture='physics-response-history-native-v18'
    requires_bond_response_history=True
    response_input_contract=FORMAT
    response_output_contract='learned-cached-response-head/1'
    response_scale_mode='floor-1e-4'

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.response_encoder=MLP(18,h,h)
        self.response_extra=nn.Linear(h,12)
        # Reuse the six total-force outputs: there is one force prediction,
        # shared with the legacy auxiliary objective and viewer.
        for head in (self.response_encoder.net[-1],self.response_extra,self.force_end):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.register_buffer('response_rms',torch.ones(18))
        self.register_buffer('response_cohort_rms',torch.ones(2,18))
        # Tension and extreme-fibre compression can be signed. Only these four
        # fields are nonnegative by their engine definitions.
        self.register_buffer('_response_nonnegative',torch.tensor([i in (12,14,15,17) for i in range(18)]),persistent=False)

    def prepare_batch(self,raw):
        n=len(raw['bond_endpoints'])
        if 'bond_response_encoded' not in raw or raw['bond_response_encoded'].shape!=(n,18):
            raise ValueError('response model requires 18 exact cached-response start features')
        values=raw['bond_response_encoded']
        alive=raw.get('bond_response_alive')
        if not torch.isfinite(values).all() or (values[:,self._response_nonnegative]<0).any():
            raise ValueError('invalid causal cached response')
        if alive is None or alive.dtype!=torch.bool or alive.shape!=(n,) or not alive.all():
            raise ValueError('response model requires aligned runtime-alive inputs')
        return {**super().prepare_batch(raw),'_response_features':values}

    def encode_edges(self,b):
        return super().encode_edges(b)+self.response_encoder(b['_response_features'])

    @torch.no_grad()
    def fit_target_scales(self,records):
        super().fit_target_scales(records)
        squared=torch.zeros(2,18,dtype=torch.float64);counts=torch.zeros(2,dtype=torch.float64)
        for _,_,_,b in records:
            change=(b['bond_response_encoded_end']-b['bond_response_encoded']).double()
            alive=b['bond_response_alive_end']
            for cohort,mask in enumerate((alive,~alive)):
                squared[cohort]+=change[mask].square().sum(0);counts[cohort]+=int(mask.sum())
        variances=squared/counts[:,None].clamp_min(1)
        # Missing cohorts use the observed cohort's scales, not artificial zero
        # variance. Normalization floors do not clip any physical targets.
        if not (counts>0).any():raise ValueError('response scale fit requires training bond intervals')
        balanced=variances[counts>0].mean(0)
        variances=torch.where((counts>0)[:,None],variances,balanced)
        def scales(variance):
            if self.response_scale_mode=='floor-1e-4':return variance.sqrt().clamp_min(1e-4).float()
            if self.response_scale_mode!='nonzero-rms':raise ValueError('unknown response scale mode')
            # A universal encoded floor can erase physically small energy
            # targets. Preserve every nonzero measured RMS; only an exactly
            # constant target uses unit scale. Reject unrepresentable scales.
            result=torch.where(variance>0,variance.sqrt(),torch.ones_like(variance)).float()
            if not torch.isfinite(result).all() or (result<=0).any():
                raise ValueError('response RMS cannot be represented in float32')
            return result
        self.response_cohort_rms.copy_(scales(variances))
        self.response_rms.copy_(scales(balanced))

    def edge_outputs(self,edge,b):
        pred=super().edge_outputs(edge,b)
        residual=torch.cat([pred['bond_force_end'],self.response_extra(edge)],-1)
        candidate=b['bond_response_encoded']+residual*self.response_rms
        encoded=torch.where(self._response_nonnegative,candidate.clamp_min(0),candidate)
        return {**pred,'bond_force_end':encoded[:,:6],
                'response_encoded':encoded,'response_raw_encoded':candidate}

    def response_loss(self,pred,batch):
        target=batch['bond_response_encoded_end'];alive=batch['bond_response_alive_end']
        if target.shape!=pred['response_encoded'].shape or not torch.isfinite(target).all() or (target[:,self._response_nonnegative]<0).any():
            raise ValueError('invalid cached response target')
        if alive.dtype!=torch.bool or alive.shape!=target.shape[:1]:raise ValueError('invalid response endpoint activity target')
        terms=[];parts={};zero=pred['response_raw_encoded'].sum()*0
        for cohort,(name,mask) in enumerate((('active_end',alive),('retired_cache',~alive))):
            # Supervise before the admissibility clamp so zero initialization
            # retains gradients for nonnegative outputs.
            error=(pred['response_raw_encoded'][mask]-target[mask])/self.response_cohort_rms[cohort]
            term=error.square().mean() if mask.any() else zero
            parts[name]=term
            if mask.any():terms.append(term)
        parts['loss']=torch.stack(terms).mean() if terms else zero
        return parts

    def state_loss(self,pred,batch,**kwargs):
        weight=kwargs.pop('response_weight',.05)
        parts=super().state_loss(pred,batch,**kwargs);response=self.response_loss(pred,batch)
        parts['loss']=parts['loss']+weight*response['loss']
        return {**parts,**{'response_'+k:v.detach() for k,v in response.items()}}

    @staticmethod
    def decode_response(pred):
        encoded=pred['response_encoded'].detach().cpu().numpy().astype(np.float64)
        with np.errstate(over='ignore',invalid='ignore'):
            values=np.sign(encoded)*np.expm1(np.abs(encoded))*SCALE
        if not np.isfinite(values).all():raise ValueError('predicted cached response exceeds finite float64 representation')
        return values
