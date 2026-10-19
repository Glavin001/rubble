"""V22 inference with a training-only physical-unit response objective.

Inputs, output parameterization and constraints are unchanged. Each physical
field is scaled by its training residual RMS, separately for active endpoints
and retired caches. This is not a constitutive or energy-balance constraint.
"""
import torch
from .projectile_model import ProjectileResponseImpactNet
from .response_history import SCALE


def inverse_response(encoded):
    # sign(x)*expm1(abs(x)) has zero autograd derivative at x=0. Use a
    # piecewise constant direction so the derivative there is correctly SCALE.
    direction=torch.where(encoded>=0,1.,-1.)
    result=direction*torch.expm1(direction*encoded)*encoded.new_tensor(SCALE)
    if not torch.isfinite(result).all():
        raise ValueError('physical response exceeds finite training representation')
    return result


class PhysicalResponseImpactNet(ProjectileResponseImpactNet):
    architecture='physics-projectile-response-si-v23'
    response_objective_contract='physical-residual-rms/1'

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.register_buffer('physical_response_cohort_rms',torch.ones(2,18))

    @torch.no_grad()
    def fit_target_scales(self,records):
        super().fit_target_scales(records)
        squared=torch.zeros(2,18,dtype=torch.float64);counts=torch.zeros(2,dtype=torch.float64)
        for _,_,_,batch in records:
            change=inverse_response(batch['bond_response_encoded_end'].double())-inverse_response(batch['bond_response_encoded'].double())
            alive=batch['bond_response_alive_end']
            for i,mask in enumerate((alive,~alive)):
                squared[i]+=change[mask].square().sum(0);counts[i]+=int(mask.sum())
        if not (counts>0).any():raise ValueError('physical response scaling requires training bonds')
        variances=squared/counts[:,None].clamp_min(1)
        observed=variances[counts>0].mean(0)
        variances=torch.where((counts>0)[:,None],variances,observed)
        # No universal floor that can erase physically small energy targets.
        # Exactly constant fields use one SI unit; nothing is clipped.
        rms=torch.where(variances>0,variances.sqrt(),torch.ones_like(variances)).float()
        if not torch.isfinite(rms).all() or (rms<=0).any():
            raise ValueError('physical response RMS cannot be represented in float32')
        self.physical_response_cohort_rms.copy_(rms)

    def response_loss(self,pred,batch):
        target=batch['bond_response_encoded_end'];alive=batch['bond_response_alive_end']
        if target.shape!=pred['response_raw_encoded'].shape or not torch.isfinite(target).all() or (target[:,self._response_nonnegative]<0).any():
            raise ValueError('invalid physical response target')
        if alive.dtype!=torch.bool or alive.shape!=target.shape[:1]:raise ValueError('invalid response endpoint activity target')
        # Raw prediction retains a correcting gradient below the nonnegative
        # admissibility boundary, as in V18/V22. Evaluation uses decoded output.
        difference=inverse_response(pred['response_raw_encoded'])-inverse_response(target)
        parts={};terms=[];zero=difference.sum()*0
        for i,(name,mask) in enumerate((('active_end',alive),('retired_cache',~alive))):
            error=difference[mask]/self.physical_response_cohort_rms[i]
            term=error.square().mean() if mask.any() else zero
            if not torch.isfinite(term):raise ValueError('nonfinite normalized physical response loss')
            parts[name]=term
            if mask.any():terms.append(term)
        parts['loss']=torch.stack(terms).mean() if terms else zero
        return parts
