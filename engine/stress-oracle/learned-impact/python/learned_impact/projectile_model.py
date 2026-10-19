"""Current projectile dynamics for V13 history and V22 response models.

All projectiles contribute; there is no nearest-object truncation. This closes
specific recorded-input omissions, not all contact/engine feature parity gaps.
"""
import torch
from torch import nn
from .history_model import ResidualHistoryImpactNet
from .response_model import ResponseHistoryImpactNet
from .model import MLP
from .physics import symlog
from .projectile_inputs import CONTRACT


class DynamicProjectileFeatures:
    requires_projectile_inputs=True
    projectile_input_contract=CONTRACT

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.dynamic_projectile_encoder=MLP(52,h,h)
        nn.init.zeros_(self.dynamic_projectile_encoder.net[-1].weight)
        nn.init.zeros_(self.dynamic_projectile_encoder.net[-1].bias)

    def prepare_batch(self,raw):
        count=len(raw['impactor_center']);n=len(raw['chunk_mass'])
        for key,shape in [('impactor_angular_velocity',(count,3)),('impactor_runtime_properties',(count,17)),('impactor_material_features',(count,26))]:
            if key not in raw:raise ValueError(f'dynamic projectile model requires {key}')
            if raw[key].shape!=shape or not torch.isfinite(raw[key]).all():raise ValueError(f'invalid dynamic projectile feature {key}')
        b=super().prepare_batch(raw);p=raw['impactor_runtime_properties']
        metadata=torch.cat([symlog(raw['impactor_angular_velocity']),
                            symlog(p[:,:15],[1000.]+[100.]*9+[1e6,1e4,1e4,1e4,.1]),p[:,15:17],raw['impactor_material_features']],-1)
        offset=raw['world_center'][None,:,:]-raw['impactor_center'][:,None,:]
        spin=raw['impactor_angular_velocity'][:,None,:].expand(count,n,3)
        # Rigid projectile point velocity evaluated at the chunk center is a
        # kinematic feature, not a claim that this point is in contact.
        relative=raw['impactor_velocity'][:,None,:]+torch.linalg.cross(spin,offset,dim=-1)-raw['world_velocity'][None,:,:]
        pair=torch.cat([metadata[:,None,:].expand(count,n,46),symlog(offset),symlog(relative)],-1)
        return {**b,'_dynamic_projectile_features':pair}

    def encode_nodes(self,b):
        node=super().encode_nodes(b);pair=b['_dynamic_projectile_features']
        if len(pair):node=node+self.dynamic_projectile_encoder(pair).sum(0)
        return node


class ProjectileHistoryImpactNet(DynamicProjectileFeatures, ResidualHistoryImpactNet):
    architecture='physics-projectile-history-native-v13'


class ProjectileResponseImpactNet(DynamicProjectileFeatures, ResponseHistoryImpactNet):
    """V18 response/state contract plus current projectile dynamics (V22)."""
    architecture='physics-projectile-response-native-v22'
