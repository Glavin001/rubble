"""JointState decoder whose connectivity agrees with the fracture classifier.

Unreinforced bonds disconnect exactly at complete damage. Reinforced bonds
may be fully cracked but remain connected through an unbroken reinforcing tie.
This structures the output distribution; it is not a World/RtBond adapter.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .history_model import ResidualHistoryImpactNet


class ConsistentHistoryImpactNet(ResidualHistoryImpactNet):
    architecture='physics-consistent-history-native-v15'
    joint_history_output_contract='learned-joint-state-head/3'
    connectivity_input_contract='explicit-rebar-complete-damage/1'

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.material_crack=nn.Linear(h,1)
        nn.init.zeros_(self.material_crack.weight);nn.init.constant_(self.material_crack.bias,-4.)

    def prepare_batch(self,raw):
        n=len(raw['bond_endpoints'])
        for key in ('bond_has_rebar','bond_damage_complete_start'):
            if key not in raw or raw[key].dtype!=torch.bool or raw[key].shape!=(n,):
                raise ValueError(f'consistent state model requires explicit {key}')
        connected=~raw['bond_damage_complete_start']|(raw['bond_has_rebar']&(raw['bond_exact_history'][:,23]==0))
        if not torch.equal(connected,raw['bond_failure_eligible']):
            raise ValueError('source connectivity disagrees with history')
        return super().prepare_batch(raw)

    @torch.no_grad()
    def fit_target_scales(self,records):
        super().fit_target_scales(records)
        squared=count=0.
        for _,_,_,b in records:
            valid=~b['bond_damage_complete_end']
            change=(b['bond_exact_history_end'][:,0]-b['bond_exact_history'][:,0]).double()
            squared+=float(change[valid].square().sum());count+=int(valid.sum())
        self.exact_history_rms[0]=max(np.sqrt(squared/max(count,1)),1e-4)

    def edge_outputs(self,edge,b):
        pred=super().edge_outputs(edge,b)
        present=b['bond_has_rebar'];start=b['bond_exact_history']
        disconnected=(self.break_head(edge).squeeze(-1).sigmoid()>=self.threshold)|~b['bond_failure_eligible']
        crack_logit=self.material_crack(edge).squeeze(-1)
        full=disconnected|b['bond_damage_complete_start']|(present&(crack_logit>=0))
        # If rebar was already broken, a connected forecast must retain partial
        # material damage; an unbroken reinforcing tie cannot be manufactured.
        full=torch.where(~disconnected & present & (start[:,23]==1),b['bond_damage_complete_start'],full)
        encoded=pred['exact_history_encoded'].clone()
        endpoint=torch.full_like(encoded[:,0],np.log(2))
        upper=torch.nextafter(endpoint,torch.full_like(endpoint,-float('inf')))
        partial=torch.maximum(start[:,0],torch.minimum(encoded[:,0],upper))
        encoded[:,0]=torch.where(full,endpoint,partial)
        rebar=(start[:,23]==1)|(pred['rebar_break_logit']>=0)
        rebar=torch.where(present&disconnected,torch.ones_like(rebar),rebar)
        rebar=torch.where(present&full&~disconnected,torch.zeros_like(rebar),rebar)
        return {**pred,'exact_history_encoded':encoded,'material_crack_logit':crack_logit,
                'damage_complete':full,'rebar_broken':rebar}

    def exact_state_loss(self,pred,batch):
        parts=super().exact_state_loss(pred,batch)
        start=batch['bond_exact_history'];end=batch['bond_exact_history_end']
        valid=torch.ones_like(end[:,:21],dtype=torch.bool);valid[:,4:6]=end[:,21:23]==0
        # Complete damage is a point mass selected by connectivity/crack heads;
        # regress the continuous damage branch only where material stays partial.
        valid[:,0]=~batch['bond_damage_complete_end']
        target=(end[:,:21]-start[:,:21])/self.exact_history_rms
        error=(pred['exact_history_residual']-target).square()
        zero=pred['exact_history_residual'].sum()*0
        parts['continuous']=error[valid].mean() if valid.any() else zero
        mask=batch['bond_has_rebar']&batch['bond_failure_eligible']&~batch['bond_damage_complete_start']&(end[:,30]==1)&(start[:,23]==0)
        parts['material_crack']=F.binary_cross_entropy_with_logits(pred['material_crack_logit'][mask],batch['bond_damage_complete_end'][mask].to(end.dtype)) if mask.any() else zero
        return parts

    def state_loss(self,pred,batch,**kwargs):
        parts=super().state_loss(pred,batch,**kwargs)
        # The parent includes the continuous/ductility/rebar/mode terms.
        extra=self.exact_state_loss(pred,batch)['material_crack']
        parts['loss']=parts['loss']+.1*kwargs.get('history_weight',.05)*extra
        return parts

    def decode_joint_state(self,pred,batch):
        decoded=super().decode_joint_state(pred,batch)
        decoded['rebar_broken']=pred['rebar_broken'].detach().cpu().numpy()
        # A float32 log encoding can round an incomplete value to log(2).
        # The explicit categorical boundary survives that rounding.
        decoded['values'][:,0]=np.where(pred['damage_complete'].detach().cpu().numpy(),1.,np.minimum(decoded['values'][:,0],np.nextafter(1.,0.)))
        return decoded
