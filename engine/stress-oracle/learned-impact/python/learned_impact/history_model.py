"""Full JointState supervision on the runtime-alive bond graph.

Covers all 21 JointState scalars, unset ductility, rebar rupture and failure mode.
This is NOT a complete RtBond/World state adapter: elastic force, stored energy,
contact handovers, chunk frame transfer and topology application still need one.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .constitutive_model import ConstitutiveNativeImpactNet
from .joint_history import SCALE, validate_values


class HistoryCarryImpactNet(ConstitutiveNativeImpactNet):
    architecture='physics-history-native-v11'
    bond_policy='runtime_alive'
    joint_history_output_contract='learned-joint-state-head/1'
    bounded=(0,1,6)
    monotonic=(2,3,16,19)

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.exact_delta=nn.Linear(h,21)
        self.ductility_set=nn.Linear(h,2)
        self.rebar_break=nn.Linear(h,1)
        self.history_mode=nn.Linear(h,5)
        self.register_buffer('exact_history_rms',torch.ones(21))
        for head in (self.exact_delta,self.ductility_set,self.rebar_break,self.history_mode):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        nn.init.constant_(self.ductility_set.bias,-4.)
        nn.init.constant_(self.rebar_break.bias,-4.)
        self.register_buffer('_exact_bounded',torch.tensor([i in self.bounded for i in range(21)]),persistent=False)
        self.register_buffer('_exact_monotonic',torch.tensor([i in self.monotonic for i in range(21)]),persistent=False)
        self.register_buffer('_exact_nonnegative',torch.tensor([i in (4,5,17,18,20) for i in range(21)]),persistent=False)

    def prepare_batch(self,raw):
        if 'bond_failure_eligible' not in raw:raise ValueError('history model requires runtime_alive bond graph and eligibility')
        eligible=raw['bond_failure_eligible']
        history=raw['bond_exact_history']
        if eligible.dtype!=torch.bool or eligible.shape!=(len(history),):raise ValueError('invalid bond_failure_eligible')
        if not torch.equal(eligible,history[:,30]==1) or not (history[:,29]==1).all():
            raise ValueError('runtime_alive graph/history eligibility mismatch')
        return super().prepare_batch(raw)

    @torch.no_grad()
    def fit_target_scales(self,records):
        super().fit_target_scales(records)
        sums=torch.zeros(21,dtype=torch.float64);counts=torch.zeros_like(sums)
        for _,_,_,b in records:
            start=b['bond_exact_history'];end=b['bond_exact_history_end']
            valid=torch.ones_like(end[:,:21],dtype=torch.bool);valid[:,4:6]=end[:,21:23]==0
            change=(end[:,:21]-start[:,:21]).double()
            sums+=(change.square()*valid).sum(0);counts+=valid.sum(0)
        self.exact_history_rms.copy_((sums/counts.clamp_min(1)).sqrt().clamp_min(1e-4).float())

    def edge_outputs(self,edge,b):
        base=super().edge_outputs(edge,b);start=b['bond_exact_history']
        residual=self.exact_delta(edge)*self.exact_history_rms
        positive=torch.where(residual<0,torch.zeros_like(residual),residual)
        result=start[:,:21]+torch.where(self._exact_monotonic,positive,residual)
        # These constraints follow JointModel::evaluate; ductility itself is NOT
        # monotonic. Unset->finite is separate from the finite scalar regression.
        bounded_start=torch.expm1(torch.where(self._exact_bounded,start[:,:21],torch.zeros_like(result)))
        step=torch.where(positive>1,torch.ones_like(positive),positive)
        bounded=torch.log1p(bounded_start+(1-bounded_start)*step)
        result=torch.where(self._exact_bounded,bounded,result)
        result=torch.where(self._exact_nonnegative & (result<0),torch.zeros_like(result),result)
        return {**base,'exact_history_encoded':result,'ductility_set_logits':self.ductility_set(edge),
                'rebar_break_logit':self.rebar_break(edge).squeeze(-1),
                'history_mode_logits':self.history_mode(edge)+4*start[:,24:29],
                'bond_failure_eligible':b['bond_failure_eligible'].to(edge.dtype)}

    def predict_classes(self,pred):
        return torch.where(pred['bond_failure_eligible']>0,super().predict_classes(pred),0)

    def exact_state_loss(self,pred,batch):
        start=batch['bond_exact_history'];end=batch['bond_exact_history_end']
        zero=pred['exact_history_encoded'].sum()*0
        valid=torch.ones_like(end[:,:21],dtype=torch.bool);valid[:,4:6]=end[:,21:23]==0
        errors=((pred['exact_history_encoded']-end[:,:21])/self.exact_history_rms).square()
        continuous=errors[valid].mean() if valid.any() else zero
        unset=start[:,21:23]==1
        set_loss=F.binary_cross_entropy_with_logits(pred['ductility_set_logits'][unset],1-end[:,21:23][unset]) if unset.any() else zero
        intact=start[:,23]==0
        rebar=F.binary_cross_entropy_with_logits(pred['rebar_break_logit'][intact],end[:,23][intact]) if intact.any() else zero
        mode=F.cross_entropy(pred['history_mode_logits'],end[:,24:29].argmax(-1)) if len(end) else zero
        return {'continuous':continuous,'ductility_set':set_loss,'rebar':rebar,'mode':mode}

    def state_loss(self,pred,batch,**kwargs):
        weight=kwargs.get('history_weight',.05)
        base=super().state_loss(pred,batch,**kwargs)
        exact=self.exact_state_loss(pred,batch)
        base['loss']=base['loss']+weight*(exact['continuous']+.1*(exact['ductility_set']+exact['rebar']+exact['mode']))
        return {**base,**{'exact_'+k:v.detach() for k,v in exact.items()}}

    @staticmethod
    def decode_joint_state(pred,batch):
        """Decode all JointState fields, preserving explicit infinity semantics.

        Does not apply a fracture verdict or alter engine topology. Reject any
        finite-valued head that cannot be represented in float64; never clip it.
        """
        encoded=pred['exact_history_encoded'].detach().cpu().numpy().astype(np.float64)
        start=batch['bond_exact_history'].detach().cpu().numpy()
        set_bits=pred['ductility_set_logits'].detach().cpu().numpy()>=0
        unset=(start[:,21:23]==1)&~set_bits
        with np.errstate(over='ignore',invalid='ignore'):
            values=np.sign(encoded)*np.expm1(np.abs(encoded))*SCALE
        # float32 log(2) rounds slightly above the real value; decoding in f64
        # must restore the exact [0,1] endpoint, not manufacture damage > 1.
        bounded=(0,1,6)
        if ((encoded[:,bounded]<0)|(encoded[:,bounded]>float(np.float32(np.log(2))))).any():
            raise ValueError('bounded joint-history encoding outside admissible interval')
        values[:,bounded]=np.minimum(values[:,bounded],1.)
        values[:,4:6]=np.where(unset,np.inf,values[:,4:6])
        rebar=(start[:,23]==1)|(pred['rebar_break_logit'].detach().cpu().numpy()>=0)
        modes=pred['history_mode_logits'].detach().cpu().numpy().argmax(-1)
        # Alive/connectivity are supplied by the caller only for validation;
        # applying a new fracture and connected-component split is engine work.
        flags=np.column_stack([rebar,modes,start[:,29:31]]).astype(np.int32)
        validate_values(values,flags)
        return {'values':values,'rebar_broken':rebar,'mode':modes}


class ResidualHistoryImpactNet(HistoryCarryImpactNet):
    """Direct residual supervision avoids dead gradients through irreversible clamps.

    Train the unconstrained normalized increment toward the actual increment.
    Decode into the admissible encoded interval afterwards. Unlike a surrogate
    clamp gradient, this loss is the ordinary squared error of the raw residual.
    V11 remains available so old checkpoints retain their original semantics.
    """
    architecture='physics-history-residual-native-v12'
    joint_history_output_contract='learned-joint-state-head/2'

    def edge_outputs(self,edge,b):
        pred=super().edge_outputs(edge,b)
        raw=self.exact_delta(edge);start=b['bond_exact_history'][:,:21]
        candidate=start+raw*self.exact_history_rms
        lower=torch.where(self._exact_bounded|self._exact_monotonic,start,
                          torch.where(self._exact_nonnegative,torch.zeros_like(start),torch.full_like(start,-float('inf'))))
        result=torch.maximum(candidate,lower)
        result=torch.where(self._exact_bounded,torch.minimum(result,torch.full_like(result,np.log(2))),result)
        return {**pred,'exact_history_encoded':result,'exact_history_residual':raw}

    def exact_state_loss(self,pred,batch):
        parts=super().exact_state_loss(pred,batch)
        start=batch['bond_exact_history'];end=batch['bond_exact_history_end']
        valid=torch.ones_like(end[:,:21],dtype=torch.bool);valid[:,4:6]=end[:,21:23]==0
        target=(end[:,:21]-start[:,:21])/self.exact_history_rms
        error=(pred['exact_history_residual']-target).square()
        parts['continuous']=error[valid].mean() if valid.any() else pred['exact_history_residual'].sum()*0
        return parts
