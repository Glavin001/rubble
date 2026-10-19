"""Versioned input ablation: native forcing plus exact history and material laws.

The endpoint heads/loss retain V7 semantics for a matched input comparison.
This does not yet decode full joint history or implement a World restart adapter.
"""
import torch
from torch import nn
from .model import MLP
from .physics import ForcingNativeImpactNet
from .joint_history import FORMAT as JOINT_HISTORY_CONTRACT
from .material_inputs import CONTRACT as MATERIAL_CONTRACT


class ConstitutiveNativeImpactNet(ForcingNativeImpactNet):
    architecture='physics-constitutive-native-v10'
    requires_exact_joint_history=True
    requires_material_inputs=True
    joint_history_input_contract=JOINT_HISTORY_CONTRACT
    material_input_contract=MATERIAL_CONTRACT

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.constitutive_node_encoder=MLP(46,h,h)
        self.constitutive_edge_encoder=MLP(77,h,h)
        for encoder in (self.constitutive_node_encoder,self.constitutive_edge_encoder):
            nn.init.zeros_(encoder.net[-1].weight)
            nn.init.zeros_(encoder.net[-1].bias)

    def prepare_batch(self,raw):
        n=len(raw['chunk_mass']);e=len(raw['bond_endpoints'])
        for key,shape in [('chunk_material_features',(n,26)),('bond_material_features',(e,26)),
                          ('constitutive_flags',(20,)),('bond_exact_history',(e,31))]:
            if key not in raw:raise ValueError(f'constitutive model requires {key}')
            if raw[key].shape!=shape or not torch.isfinite(raw[key]).all():
                raise ValueError(f'invalid constitutive feature {key}')
        b=super().prepare_batch(raw);flags=raw['constitutive_flags']
        return {**b,'_constitutive_node_features':torch.cat([raw['chunk_material_features'],flags.expand(n,20)],-1),
                '_constitutive_edge_features':torch.cat([raw['bond_material_features'],raw['bond_exact_history'],flags.expand(e,20)],-1)}

    def encode_nodes(self,b):
        return super().encode_nodes(b)+self.constitutive_node_encoder(b['_constitutive_node_features'])

    def encode_edges(self,b):
        return super().encode_edges(b)+self.constitutive_edge_encoder(b['_constitutive_edge_features'])
