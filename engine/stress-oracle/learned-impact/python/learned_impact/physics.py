"""Versioned physical-feature model and fracture-focused objective.

Fixed SI reference units preserve absolute load/material/dt information. This is
an offline PyTorch model; it deliberately does not export the legacy GPU ABI.
Recorded interval loads are teacher-forcing inputs, not a causal contact model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP, MessagePassing, PhysicsAttention
from .losses import util_loss, fracture_cdf_squared_error, fracture_event_nll

DELTA_SCALE = [.01] * 6 + [1.] * 6
JOINT_SCALE = [1.] * 7 + [.001] * 9 + [1000., 1., 1., 1e6, 1000., 1.]
FORCE_SCALE = [1000.] * 3 + [100.] * 3


def symlog(x, scale=1.):
    scale = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
    return x.sign() * torch.log1p(x.abs() / scale)


def symexp(x, scale=1.):
    scale = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
    return x.sign() * torch.expm1(x.abs()) * scale


class SizeStableAttention(PhysicsAttention):
    """Node-to-token assignments sum to one per node, independent of node count."""
    def forward(self, node):
        w = F.softmax(self.assign(node), dim=-1)
        tokens = (w.T @ node) / w.sum(0)[:, None].clamp_min(1e-8)
        qkv = self.qkv(tokens).view(self.p, 3, self.heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = F.softmax(torch.einsum('phd,qhd->hpq', q, k) * q.shape[-1] ** -.5, dim=-1)
        mixed = torch.einsum('hpq,qhd->phd', attn, v).reshape(self.p, -1)
        tokens = self.norm1(tokens + self.out(mixed))
        tokens = self.norm2(tokens + self.ffn(tokens))
        return node + w @ tokens


class PhysicsImpactNet(nn.Module):
    architecture = 'physics-v2'

    def __init__(self, h=48, p=16, threshold=.5):
        super().__init__()
        self.h, self.p, self.threshold = h, p, threshold
        # See features(): explicit, stable feature contract; never use labels.
        self.encode_n = MLP(67, h, h)
        self.encode_e = MLP(81, h, h)
        self.mp = nn.ModuleList([MessagePassing(h) for _ in range(4)])
        self.attn = SizeStableAttention(h, p)
        self.break_head = nn.Linear(h, 1)
        self.time_head = nn.Linear(h, 4)
        self.bond_util = nn.Linear(h, 1)
        self.chunk_delta = nn.Linear(h, 12)
        self.joint_end = nn.Linear(h, 22)
        self.force_end = nn.Linear(h, 6)

    @staticmethod
    def features(b):
        n, e = len(b['chunk_mass']), len(b['bond_endpoints'])
        ref = b['chunk_state']

        def get(key, shape, default=0.):
            return b[key].float() if key in b else ref.new_full(shape, default)

        dt = get('dt', ()).expand(n, 1)
        loads = b['chunk_load_bins']
        # F_bin * frame_dt / mass makes the velocity scale and dt visible. This
        # uses the full frame duration, not the integrated impulse of a quarter
        # bin; it supplements the four force bins and preserves their magnitude.
        dv = loads[:, :, :3] * dt[:, None] / b['chunk_mass'][:, None, None].clamp_min(1e-8)
        node = torch.cat([
            symlog(b['chunk_mass'][:, None], 10.), symlog(b['chunk_inertia'], .1),
            symlog(b['chunk_half_extents'], .1), symlog(b['chunk_state'], DELTA_SCALE),
            symlog(loads, FORCE_SCALE).flatten(1), b['chunk_gravity_local'] / 9.81,
            F.one_hot(get('chunk_support', (n,)).long(), num_classes=3).float(),
            get('chunk_contact', (n,))[:, None], symlog(dt, .001),
            get('load_basis_v2', ()).expand(n, 1), symlog(dv).flatten(1),
            symlog(get('chunk_level', (n,))[:, None]),
            (get('chunk_coarse_parent', (n,), -1) >= 0).float()[:, None],
            torch.full((n, 1), float('bond_frames' in b), device=ref.device),
        ], -1)
        strength = get('bond_strength', (e, 3))
        area = get('bond_area', (e,))[:, None]
        # Capacity-normalized prior forces supplement, not replace, SI forces.
        caps = area * torch.stack([strength[:, 1], strength[:, 1], strength[:, 0]], -1)
        prev = get('bond_force_prev', (e, 6))
        qs = b['bond_qs_force']
        edge = torch.cat([
            symlog(b['bond_joint_state'], JOINT_SCALE), symlog(prev, FORCE_SCALE),
            symlog(qs, FORCE_SCALE), symlog(area, .01), symlog(get('bond_width', (e, 2)), .1),
            symlog(get('bond_offset', (e, 3)), .1),
            symlog(get('bond_stiffness', (e, 6)), [1e8]*3+[1e6]*3),
            symlog(strength, 1e6), symlog(get('bond_fracture_energy', (e, 3)), 100.),
            symlog(get('bond_rebar', (e, 5)), [1e8, 1e8, 1e5, 1e5, 1e3]),
            symlog(get('bond_weibull', (e,))[:, None]),
            symlog(get('bond_buckling_length', (e,))[:, None]),
            get('qs_mode', (), -1).expand(e, 1), get('load_basis_v2', ()).expand(e, 1),
            symlog(get('dt', ()).expand(e, 1), .001),
            get('bond_frames', (e, 3, 3)).reshape(e, 9),
            torch.full((e, 1), float('bond_frames' in b), device=ref.device),
            symlog(prev[:, :3] / caps.clamp_min(1e-8)),
            symlog(qs[:, :3] / caps.clamp_min(1e-8)),
            (strength == 0).float(),  # zero compressive strength means unlimited in schema v1
        ], -1)
        return node, edge

    def prepare_batch(self, raw):
        node, edge = self.features(raw)
        return {**raw, '_node_features': node, '_edge_features': edge}

    def encode_nodes(self, b):
        return self.encode_n(b['_node_features'])

    def encode_edges(self, b):
        return self.encode_e(b['_edge_features'])

    def node_outputs(self,node,b):
        return {'chunk_state_delta':self.chunk_delta(node)}

    def edge_outputs(self,edge,b):
        return {'bond_joint_state_end':self.joint_end(edge),'bond_force_end':self.force_end(edge)}

    def forward(self, b):
        if '_node_features' not in b:
            b = self.prepare_batch(b)
        node, edge = self.encode_nodes(b), self.encode_edges(b)
        for i, mp in enumerate(self.mp):
            node, edge = mp(node, edge, b['csr_row_ptr'], b['csr_col'], b['bond_endpoints'])
            if i == 1:
                node = self.attn(node)
        logit = self.break_head(edge).squeeze(-1)
        timing = self.time_head(edge)
        # Joint class probabilities are useful for likelihood, but fracture
        # decisions use P(any break), not the max of four competing time bins.
        joint_logp = torch.cat([F.logsigmoid(-logit)[:, None],
                               F.logsigmoid(logit)[:, None] + F.log_softmax(timing, -1)], -1)
        return {'break_logit': logit, 'time_logits': timing, 'bond_fail_logits': joint_logp,
                'bond_peak_util': F.softplus(self.bond_util(edge).squeeze(-1)),
                **self.node_outputs(node,b),
                **self.edge_outputs(edge,b)}

    def predict_classes(self, pred):
        return torch.where(pred['break_logit'].sigmoid() >= self.threshold,
                           pred['time_logits'].argmax(-1)+1, 0)

    @staticmethod
    def decode_delta(pred):
        return symexp(pred['chunk_state_delta'], DELTA_SCALE)


class PreviewImpactNet(PhysicsImpactNet):
    """Same model capacity/loss, causal contact preview input contract.

    No fallback to future oracle loads, including the interval contact mask.
    Still teacher-forced on oracle state at the START of each interval; this is
    not a closed-loop rollout and needs its own runtime implementation/validation.
    """
    architecture = 'physics-preview-v3'

    def prepare_batch(self, raw):
        return {**super().prepare_batch(raw), '_preview_feature_contract': self.architecture}

    def forward(self, batch):
        if '_node_features' in batch and batch.get('_preview_feature_contract') != self.architecture:
            raise ValueError('cached features do not have the causal preview input contract')
        return super().forward(batch)

    @staticmethod
    def features(b):
        for key in ('chunk_preview_load_bins', 'chunk_preview_contact'):
            if key not in b:
                raise ValueError(f'causal preview model requires {key}; regenerate with --rigid-preview')
        return PhysicsImpactNet.features({**b,
            'chunk_load_bins': b['chunk_preview_load_bins'],
            'chunk_contact': b['chunk_preview_contact']})


class StatePreviewImpactNet(PreviewImpactNet):
    """Experimental residual history / physical-MSE state objective.

    Same causal inputs and parameter count as v3. Continuous rebar-broken values
    remain a surrogate probability, not a valid discrete solver checkpoint. The
    constraints do not enforce energy balance or supply missing constitutive data.
    """
    architecture = 'physics-preview-state-v4'
    bounded_history = (0, 1, 6, 17)
    monotonic_history = (2, 3, 16, 20)

    def __init__(self, h=48, p=16, threshold=.5):
        super().__init__(h, p, threshold)
        self.register_buffer('delta_rms', torch.ones(12))
        self.register_buffer('history_rms', torch.ones(22))
        self.register_buffer('_bounded_mask', torch.tensor([i in self.bounded_history for i in range(22)]), persistent=False)
        self.register_buffer('_monotonic_mask', torch.tensor([i in self.monotonic_history for i in range(22)]), persistent=False)
        # Start with exact zero motion / unchanged history, not arbitrary drift.
        nn.init.zeros_(self.chunk_delta.weight); nn.init.zeros_(self.chunk_delta.bias)
        nn.init.zeros_(self.joint_end.weight); nn.init.zeros_(self.joint_end.bias)

    @torch.no_grad()
    def fit_target_scales(self, records):
        delta_sum = torch.zeros(4, dtype=torch.float64); delta_n = 0
        history_sum = torch.zeros(22, dtype=torch.float64); history_n = 0
        for _, _, _, b in records:
            target = b['chunk_state_delta']
            valid = b.get('delta_valid', torch.ones(len(target), dtype=torch.bool))
            delta_sum += target[valid].double().reshape(-1,4,3).square().sum((0,2))
            delta_n += int(valid.sum()) * 3
            change = symlog(b['bond_joint_state_end'], JOINT_SCALE) - symlog(b['bond_joint_state'], JOINT_SCALE)
            history_sum += change.double().square().sum(0); history_n += len(change)
        # Floors define numerical scaling only, never alter physical labels.
        self.delta_rms.copy_((delta_sum/max(1,delta_n)).sqrt().clamp_min(1e-7).float().repeat_interleave(3))
        self.history_rms.copy_((history_sum/max(1,history_n)).sqrt().clamp_min(1e-4).float())

    def forward(self, b):
        pred = super().forward(b)
        residual = pred['bond_joint_state_end'] * self.history_rms
        start = b['bond_joint_state']
        encoded = symlog(start, JOINT_SCALE)
        # Batch all fields to avoid dozens of small GPU launches. Explicit
        # subgradient 1 at zero permits learning from exact zero initialization.
        positive = torch.where(residual < 0, torch.zeros_like(residual), residual)
        result = encoded + torch.where(self._monotonic_mask, positive, residual)
        step = torch.where(positive > 1, torch.ones_like(positive), positive)
        # Mask before log1p so non-bounded signed fields cannot produce NaNs in
        # an unused branch. All bounded fields have SI reference scale 1.
        bounded_start = torch.where(self._bounded_mask, start, torch.zeros_like(start))
        value = bounded_start + (1-bounded_start) * step
        bounded = torch.log1p(value)  # nonzero derivative at value=0
        pred['bond_joint_state_end'] = torch.where(self._bounded_mask, bounded, result)
        return pred

    def decode_delta(self, pred):
        return pred['chunk_state_delta'] * self.delta_rms

    def state_loss(self, pred, batch, **kwargs):
        # Retain the original break/timing/utilization/force objectives exactly.
        target = batch['chunk_state_delta'] / self.delta_rms
        valid = batch.get('delta_valid', torch.ones(len(target), dtype=torch.bool, device=target.device))
        delta = F.mse_loss(pred['chunk_state_delta'][valid], target[valid]) if valid.any() else pred['chunk_state_delta'].sum()*0
        encoded = symlog(batch['bond_joint_state_end'], JOINT_SCALE)
        joint = ((pred['bond_joint_state_end']-encoded)/self.history_rms).square().mean() if len(encoded) else pred['bond_joint_state_end'].sum()*0
        return physics_loss(pred, batch, delta_objective=delta, joint_objective=joint, **kwargs)


class WorldStateImpactNet(StatePreviewImpactNet):
    """Causal recorded start-world observations plus v4 history objective.

    All recorded projectiles contribute through a shared pair encoder and sum.
    Pair work is O(nodes * projectiles); no nearest-object truncation. Initial
    crush capacity is metadata, never claimed to be current remaining capacity.
    """
    architecture = 'physics-preview-world-v5'
    requires_world_inputs = True

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.world_encoder=MLP(21,h,h)
        self.impactor_encoder=MLP(24,h,h)
        # Same initial prediction as v4; new information is learned additively.
        for encoder in (self.world_encoder,self.impactor_encoder):
            nn.init.zeros_(encoder.net[-1].weight);nn.init.zeros_(encoder.net[-1].bias)

    def prepare_batch(self,raw):
        required=('world_velocity','world_angular_velocity','world_rotation','world_displacement',
                  'world_center','world_ground','world_active','impactor_center','impactor_velocity',
                  'impactor_rotation','impactor_properties')
        for key in required:
            if key not in raw:raise ValueError(f'world-start model requires {key}')
        b=super().prepare_batch(raw);n=len(raw['chunk_mass']);g=raw['world_ground']
        world=torch.cat([symlog(raw['world_velocity']),symlog(raw['world_angular_velocity']),
                         raw['world_rotation'].flatten(1),symlog(raw['world_displacement'],.1),
                         symlog((raw['world_center'][:,2:3]-g[0])*g[1]),
                         g[1].expand(n,1),raw['world_active'][:,None]],-1)
        props=raw['impactor_properties'];count=len(props)
        metadata=torch.cat([symlog(props[:,:4],[.1,.1,.1,1000.]),props[:,4:6],
                            symlog(props[:,6:8],[1e6,1000.]),props[:,8:9]],-1)
        pair=torch.cat([symlog(raw['impactor_center'][:,None,:]-raw['world_center'][None,:,:]),
                        symlog(raw['impactor_velocity'][:,None,:]-raw['world_velocity'][None,:,:]),
                        raw['impactor_rotation'].reshape(count,1,9).expand(count,n,9),
                        metadata[:,None,:].expand(count,n,9)],-1)
        return {**b,'_world_features':world,'_impactor_pair_features':pair}

    def encode_nodes(self,b):
        node=super().encode_nodes(b)+self.world_encoder(b['_world_features'])
        pair=b['_impactor_pair_features']
        if len(pair):node=node+self.impactor_encoder(pair).sum(0)
        return node


class WorldMotionImpactNet(WorldStateImpactNet):
    """V5 plus an auxiliary world-motion residual head with reset-safe labels.

    The existing local head and history outputs retain their semantics. This is
    supervised world prediction, not an engine state/topology application path.
    """
    architecture='physics-preview-world-motion-v6'
    requires_world_targets=True
    world_target_contract='world-ballistic-residual/1'

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.world_motion=nn.Linear(h,12)
        nn.init.zeros_(self.world_motion.weight);nn.init.zeros_(self.world_motion.bias)
        self.register_buffer('world_motion_rms',torch.ones(12))

    def node_outputs(self,node,b):
        return {**super().node_outputs(node,b),'world_motion_residual':self.world_motion(node)}

    @torch.no_grad()
    def fit_target_scales(self,records):
        super().fit_target_scales(records)
        squared=torch.zeros(4,dtype=torch.float64);count=0
        for _,_,_,b in records:
            if 'world_motion_target' not in b or 'world_motion_valid' not in b:raise ValueError('world-motion supervision missing')
            valid=b['world_motion_valid'];values=b['world_motion_target'][valid].double().reshape(-1,4,3)
            squared+=values.square().sum((0,2));count+=int(valid.sum())*3
        if not count:raise ValueError('no valid world-motion training targets')
        self.world_motion_rms.copy_((squared/count).sqrt().clamp_min(1e-7).float().repeat_interleave(3))

    def decode_world_motion(self,pred):
        return pred['world_motion_residual']*self.world_motion_rms

    def state_loss(self,pred,batch,world_weight=.1,**kwargs):
        parts=super().state_loss(pred,batch,**kwargs)
        valid=batch['world_motion_valid']
        error=pred['world_motion_residual']-batch['world_motion_target']/self.world_motion_rms
        world=error[valid].square().mean() if valid.any() else pred['world_motion_residual'].sum()*0
        return {**parts,'loss':parts['loss']+world_weight*world,'world_motion':world.detach()}


class ForcingPreviewImpactNet(WorldMotionImpactNet):
    """V6 with prescribed-force descriptors; all loads and table knots retained."""
    architecture='physics-forcing-preview-v7'
    requires_forcing_inputs=True
    forcing_input_contract='prescribed-forcing-descriptors/1'
    input_contract='causal_preview'

    def __init__(self,h=48,p=16,threshold=.5):
        super().__init__(h,p,threshold)
        self.forcing_encoder=MLP(21,h,h)
        self.forcing_table_encoder=MLP(2,h,h)
        for encoder in (self.forcing_encoder,self.forcing_table_encoder):
            nn.init.zeros_(encoder.net[-1].weight);nn.init.zeros_(encoder.net[-1].bias)

    def prepare_batch(self,raw):
        for key in ('kind','direction','position','curve_kind','curve_parameters','blast','selected','table_points','table_owner'):
            if 'forcing_'+key not in raw:raise ValueError(f'forcing model requires forcing_{key}')
        b=super().prepare_batch(raw);n=len(raw['chunk_mass']);kind=raw['forcing_kind'];count=len(kind)
        position=raw['forcing_position'][:,None,:].expand(count,n,3)
        relative=torch.where((kind==2)[:,None,None],position-raw['world_center'][None,:,:],position)
        metadata=torch.cat([F.one_hot(kind,num_classes=3).float(),raw['forcing_direction'],
                            F.one_hot(raw['forcing_curve_kind'],num_classes=6).float(),
                            symlog(raw['forcing_curve_parameters'],[.001,.001,1e4,1.]),
                            symlog(raw['forcing_blast'],[1.,.001])],-1)
        return {**b,'_forcing_features':torch.cat([metadata[:,None,:].expand(count,n,18),symlog(relative,.1)],-1),
                '_forcing_table_features':symlog(raw['forcing_table_points'],[.001,1e4])}

    def encode_nodes(self,b):
        node=super().encode_nodes(b);features=b['_forcing_features'];count=len(features)
        if count:
            encoded=self.forcing_encoder(features)
            table=node.new_zeros((count,self.h))
            if len(b['_forcing_table_features']):
                table.index_add_(0,b['forcing_table_owner'],self.forcing_table_encoder(b['_forcing_table_features']))
            node=node+((encoded+table[:,None,:])*b['forcing_selected'][:,:,None]).sum(0)
        return node


class ForcingNativeImpactNet(ForcingPreviewImpactNet):
    """Experimental frame-start model, without preview or interval contact inputs.

    Zero slots are an explicit architectural ablation, never an engine fallback.
    Prescribed forcing is encoded separately. Accuracy/rollout parity is unproven.
    """
    architecture='physics-forcing-native-v7'
    input_contract='frame_start_forcing'

    @staticmethod
    def features(b):
        ref=b['chunk_state'];n=len(ref);e=len(b['bond_endpoints'])
        return PhysicsImpactNet.features({**b,'chunk_load_bins':ref.new_zeros((n,4,6)),
            'chunk_contact':ref.new_zeros(n),'bond_qs_force':ref.new_zeros((e,6)),'qs_mode':ref.new_zeros(())})


class WorldInputControlNet(WorldStateImpactNet):
    """Matched constructor/initialization with extra world branches disabled."""
    architecture = 'physics-preview-world-control-v5'

    def encode_nodes(self,b):
        return PhysicsImpactNet.encode_nodes(self,b)


def physics_loss(pred, batch, *, positive_weight=3., state_weight=.1, history_weight=.05,
                 delta_objective=None, joint_objective=None, fracture_cdf_weight=0., fracture_objective='factorized'):
    """Separate fracture probability, conditional timing and physical fields.

    Huber in signed-log SI units prevents large velocities from swamping breaks.
    Positive weight and decoder threshold are empirical training choices, saved
    with the checkpoint. No parameter here changes the oracle physics.
    """
    if fracture_objective not in ('factorized','event-nll'):
        raise ValueError('unknown fracture objective')
    if fracture_objective == 'event-nll' and positive_weight != 1.:
        raise ValueError('event-nll requires positive_weight=1 for a normalized likelihood')
    labels = batch['bond_fail_class']
    eligible=batch.get('bond_failure_eligible',torch.ones_like(labels,dtype=torch.bool))
    positive = (labels > 0) & eligible
    zero = pred['break_logit'].sum() * 0
    fail = F.binary_cross_entropy_with_logits(pred['break_logit'][eligible], positive[eligible].float(),
        pos_weight=pred['break_logit'].new_tensor(positive_weight)) if eligible.any() else zero
    timing = F.cross_entropy(pred['time_logits'][positive], labels[positive]-1) if positive.any() else zero
    util = util_loss(pred['bond_peak_util'], torch.log1p(batch['bond_peak_util'].clamp_min(0)))
    target_delta = symlog(batch['chunk_state_delta'], DELTA_SCALE)
    valid = batch.get('delta_valid', torch.ones(len(target_delta), dtype=torch.bool, device=target_delta.device))
    delta = delta_objective if delta_objective is not None else (F.smooth_l1_loss(pred['chunk_state_delta'][valid], target_delta[valid]) if valid.any() else pred['chunk_state_delta'].sum()*0)
    joint = joint_objective if joint_objective is not None else (F.smooth_l1_loss(pred['bond_joint_state_end'], symlog(batch['bond_joint_state_end'], JOINT_SCALE)) if len(labels) and 'bond_joint_state_end' in batch else zero)
    force = F.smooth_l1_loss(pred['bond_force_end'], symlog(batch['bond_force_end'], FORCE_SCALE)) if len(labels) and 'bond_force_end' in batch else zero
    cdf = fracture_cdf_squared_error(pred['break_logit'][eligible], pred['time_logits'][eligible], labels[eligible]).mean() if eligible.any() else zero
    fracture = fail + .15 * timing
    loss = fracture + .3 * util + state_weight * delta + history_weight * (joint + force)
    if fracture_objective == 'event-nll':
        fracture = fracture_event_nll(pred['break_logit'][eligible],pred['time_logits'][eligible],labels[eligible]).mean() if eligible.any() else zero
        loss = fracture + .3 * util + state_weight * delta + history_weight * (joint + force)
    if fracture_cdf_weight:
        loss = loss + fracture_cdf_weight * cdf
    return {'loss': loss, **{k: v.detach() for k, v in dict(fracture=fracture, fail=fail, timing=timing, fracture_cdf=cdf, util=util, delta=delta, joint=joint, force=force).items()}}
