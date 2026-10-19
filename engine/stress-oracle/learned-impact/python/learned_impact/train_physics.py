"""Grouped, reproducible legacy/v2 training with validation-only calibration.

Examples sharing a physical first frame stay together, including partial clips
and QS variants. No random-frame split. Best epoch/threshold use validation;
the test partition is scored only after selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import shutil
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import list_events, event_batches
from .schema import load_event
from .model import LearnedImpactNet
from .constitutive_model import ConstitutiveNativeImpactNet
from .history_model import HistoryCarryImpactNet, ResidualHistoryImpactNet
from .projectile_model import ProjectileHistoryImpactNet, ProjectileResponseImpactNet
from .physical_response_model import PhysicalResponseImpactNet
from .consistent_model import ConsistentHistoryImpactNet
from .response_model import ResponseHistoryImpactNet
from .fragment_loss import fragment_cut_targets, fragment_cut_loss
from .history_metrics import ExactHistoryMetrics
from .response_metrics import ResponseMetrics
from .normalize import normalize_batch
from .losses import total_loss, fracture_cdf_squared_error
from .physics import PhysicsImpactNet, PreviewImpactNet, StatePreviewImpactNet, WorldStateImpactNet, WorldInputControlNet, WorldMotionImpactNet, ForcingPreviewImpactNet, ForcingNativeImpactNet, physics_loss
from .visualize import counts_metrics, load_checkpoint


def initialize_frozen_state_heads(model, source):
    """Retain exact v3 trunk/break/timing/util/force weights; fit state heads only."""
    baseline, meta = load_checkpoint(source)
    if type(baseline) is not PreviewImpactNet or type(model) is not StatePreviewImpactNet:
        raise ValueError('frozen state fitting requires preview v3 source and state v4 destination')
    current = model.state_dict()
    for key, value in baseline.state_dict().items():
        if not key.startswith(('chunk_delta.', 'joint_end.')):
            if current[key].shape != value.shape:raise ValueError('incompatible frozen model dimensions')
            current[key] = value
    model.load_state_dict(current)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('chunk_delta.', 'joint_end.')))
    model.threshold = baseline.threshold
    return meta

INPUTS = ('chunk_state', 'chunk_load_bins', 'chunk_gravity_local', 'chunk_contact',
          'bond_joint_state', 'bond_force_prev')


def physical_group(ev):
    h = hashlib.sha256(str(ev.manifest['dt']).encode())
    for key, value in sorted({**ev.static, **{k: ev.frames[0][k] for k in INPUTS}}.items()):
        h.update(key.encode()); h.update(value.tobytes())
    return h.hexdigest()


def recipe_group(ev):
    generator = ev.context.get('generator') or {}
    group = generator.get('split_group')
    return f"{generator.get('catalog', 'unknown')}:{group}" if group else None


def layout_group(ev):
    """Optional explicit geometry identity; all load/material cases stay together."""
    generator=ev.context.get('generator') or {}
    group=generator.get('layout_group')
    # Load catalogs may reuse a source geometry from another catalog. Preserve
    # that declared identity so different loading cannot leak across partitions.
    return f"{generator.get('layout_catalog',generator.get('catalog','unknown'))}:{group}" if group else None


def grouped_events(events):
    """Union identical first frames, trajectory identities and explicit layout IDs.

    Cadence changes alter first-frame arrays; prefixes/cadences of a recorded
    physical recipe must nevertheless stay in the same partition. New layout
    catalogs also keep all physical load recipes of one geometry together.
    """
    hashes = {name: physical_group(ev) for name, ev in events.items()}
    parent = {h:h for h in hashes.values()}
    def find(h):
        while h != parent[h]:
            parent[h] = parent[parent[h]]; h = parent[h]
        return h
    recipes = {}
    for name, ev in events.items():
        for kind,identity in (('recipe',recipe_group(ev)),('layout',layout_group(ev))):
            if identity is None:continue
            key=(kind,identity);h=hashes[name]
            if key in recipes:
                a,b=find(h),find(recipes[key]);parent[max(a,b)]=min(a,b)
            else:recipes[key]=h
    return {name:find(h) for name,h in hashes.items()}


def split_events(events, seed=17):
    """Family-stratified physical groups; tiny families train-only."""
    groups = grouped_events(events)
    members = {}
    for name, ev in events.items():
        members.setdefault(groups[name], []).append(name)
    families = {}
    for key, names in members.items():
        family = '|'.join(sorted({events[n].manifest['structure_type'] for n in names}))
        families.setdefault(family, {})[key] = names
    split = {}
    for family, members in sorted(families.items()):
        keys = sorted(members)
        random.Random(f'{seed}:{family}').shuffle(keys)
        nv = max(1, round(.2 * len(keys))) if len(keys) >= 4 else 0
        nt = max(1, round(.2 * len(keys))) if len(keys) >= 6 else 0
        for i, key in enumerate(keys):
            partition = 'validation' if i < nv else 'test' if i < nv+nt else 'train'
            for name in members[key]: split[name] = partition
    return split, groups


def inherited_split(events, reference):
    """Reuse exact scenario membership across cadence experiments; never guess."""
    groups = grouped_events(events)
    parts, reference_groups = {}, {}
    recipe_members = {}
    for name, recipe in reference.get('recipe_groups', {}).items():
        if recipe: recipe_members.setdefault(recipe, []).append(name)
    for name, ev in events.items():
        generator = ev.context.get('generator') or {}
        candidates = {name, ev.manifest['event_id'], generator.get('split_group')}
        candidates.update(recipe_members.get(recipe_group(ev), []))
        found = [n for n in candidates if n in reference['events']]
        values = {reference['events'][n] for n in found}
        if len(values) != 1:
            raise ValueError(f'{name}: missing or conflicting reference split membership')
        part = values.pop()
        if part not in ('train','validation','test'):
            raise ValueError(f'{name}: invalid reference partition {part}')
        parts[name] = part
        ref_groups = {reference['physical_groups'][n] for n in found}
        if len(ref_groups) != 1:
            raise ValueError(f'{name}: conflicting reference physical groups')
        reference_groups[name] = ref_groups.pop()
    # A new prefix/cadence can expose an identity absent from an old splitter.
    for identities in (groups, reference_groups):
        for key in set(identities.values()):
            if len({parts[n] for n,g in identities.items() if g == key}) > 1:
                raise ValueError('reference split leaks a physical trajectory across partitions')
    return parts, reference_groups


def frame_key(batch):
    h = hashlib.sha256()
    for key in sorted(k for k in batch if not k.startswith('_') and k not in ('bond_ids', 'delta_valid')):
        h.update(key.encode()); h.update(batch[key].numpy().tobytes())
    return h.digest()


def load_records(dataset, model, split_seed=17, all_data=False, split_from=None, fragment_cut_weight=0.):
    if fragment_cut_weight and getattr(model,'bond_policy',None)!='runtime_alive':
        raise ValueError('fragment cut loss requires exact runtime-alive history')
    events = {p.name: load_event(p,world_inputs=getattr(model,'requires_world_inputs',False),world_targets=getattr(model,'requires_world_targets',False),forcing_inputs=getattr(model,'requires_forcing_inputs',False),exact_joint_history=getattr(model,'requires_exact_joint_history',False),material_inputs=getattr(model,'requires_material_inputs',False),projectile_inputs=getattr(model,'requires_projectile_inputs',False),bond_response_history=getattr(model,'requires_bond_response_history',False)) for p in list_events(dataset)}
    split, groups = inherited_split(events, json.loads(Path(split_from).read_text())) if split_from else split_events(events, split_seed)
    if all_data:
        split = dict.fromkeys(events, 'train')
    records = dict(train=[], validation=[], test=[])
    seen = set()
    duplicates = 0
    for name, ev in events.items():
        part = split[name]
        for i, raw in enumerate(event_batches(ev,bond_policy=getattr(model,"bond_policy","first_failure"))):
            if part == 'train':
                key = frame_key(raw)
                if key in seen:
                    duplicates += 1; continue
                seen.add(key)
            batch = model.prepare_batch(raw) if isinstance(model, PhysicsImpactNet) else normalize_batch(raw)
            if fragment_cut_weight:
                # A loss target only; construct after feature preparation and
                # duplicate detection, preserving baseline sampling identities.
                batch['fragment_cut_targets']=fragment_cut_targets(
                    len(raw['chunk_state']),raw['bond_endpoints'].numpy(),
                    raw['bond_failure_eligible'].numpy(),
                    raw['bond_exact_history_end'][:,30].numpy()==1,
                    raw['bond_fail_class'].numpy())
            records[part].append((name, ev.manifest['structure_type'], i, batch))
    return records, {'events': split, 'physical_groups': groups, 'recipe_groups': {n:recipe_group(ev) for n,ev in events.items()}, 'deduplicated_training_frames': duplicates,
                     'seed': split_seed, 'method': 'inherited scenario membership with trajectory leakage check' if split_from else 'family-stratified union of identical first frames and explicit trajectory IDs',
                     'reference_sha256': hashlib.sha256(Path(split_from).read_bytes()).hexdigest() if split_from else None,
                     'world_input_sources':{n:ev.world_source_sha256 for n,ev in events.items() if ev.world_source_sha256},
                     'material_input_sources':{n:ev.material_source_sha256 for n,ev in events.items() if ev.material_source_sha256},
                     **({'layout_groups':{n:layout_group(ev) for n,ev in events.items()},'layout_split_contract':'all recipes sharing an explicit layout ID stay together'} if any(layout_group(ev) for ev in events.values()) else {}),
                     'frames': {p: len(r) for p, r in records.items()}}


@torch.no_grad()
def evaluate(model, records, calibrate=False, legacy_any=False, window_factor=1):
    thresholds = np.arange(.1, .91, .05) if calibrate and (isinstance(model, PhysicsImpactNet) or legacy_any) else [getattr(model, 'threshold', .5)]
    totals = np.zeros((len(thresholds), 3), dtype=np.int64)
    family_counts = {}
    windows = {}
    util_abs = util_n = correct_time = timed = brier_sum = 0
    bins_error = 0.
    cdf_sum = np.zeros(2); cdf_count = np.zeros(2,dtype=np.int64)
    delta_squared = np.zeros(4); zero_squared = np.zeros(4); delta_n = 0; total_delta_chunks = 0
    state_objective = history_objective = 0.
    world_squared=np.zeros(4);world_zero=np.zeros(4);world_count=0;world_angle_squared=world_angle_zero=0.
    model.eval()
    exact_history=ExactHistoryMetrics() if isinstance(model,HistoryCarryImpactNet) else None
    response_metrics=ResponseMetrics() if getattr(model,'requires_bond_response_history',False) else None
    for name, family, fi, batch in records:
        pred = model(batch)
        if exact_history is not None:exact_history.add(pred,batch)
        if response_metrics is not None:response_metrics.add(pred,batch)
        if isinstance(model, StatePreviewImpactNet):
            losses = model.state_loss(pred,batch)
            state_objective += float(losses['delta']);history_objective += float(losses['joint'])
        if isinstance(model,WorldMotionImpactNet):
            from .world_targets import rotation_exp,rotation_log
            mask=batch['world_motion_valid'].numpy()
            target=batch['world_motion_target'].numpy()[mask].astype(np.float64)
            actual=model.decode_world_motion(pred).numpy()[mask].astype(np.float64)
            world_squared+=np.square(actual-target).reshape(-1,4,3).sum((0,2))
            world_zero+=np.square(target).reshape(-1,4,3).sum((0,2));world_count+=len(target)
            angle=rotation_log(rotation_exp(actual[:,3:6])@np.swapaxes(rotation_exp(target[:,3:6]),-1,-2))
            world_angle_squared+=np.square(angle).sum();world_angle_zero+=np.square(target[:,3:6]).sum()
        eligible=batch.get('bond_failure_eligible',torch.ones_like(batch['bond_fail_class'],dtype=torch.bool)).numpy()
        truth_class = batch['bond_fail_class'].numpy()[eligible]
        truth = truth_class > 0
        if isinstance(model, PhysicsImpactNet):
            probability = pred['break_logit'].sigmoid().numpy()[eligible]
            cdf = fracture_cdf_squared_error(pred['break_logit'],pred['time_logits'],batch['bond_fail_class']).mean(-1).numpy()[eligible]
            for category,mask in enumerate((~truth,truth)):
                cdf_sum[category] += float(cdf[mask].sum()); cdf_count[category] += int(mask.sum())
            time_class = pred['time_logits'].argmax(-1).numpy()[eligible]+1
            targets_util = np.log1p(batch['bond_peak_util'].numpy()[eligible].clip(0))
        else:
            probability = (1-pred['bond_fail_logits'].softmax(-1)[:,0]).numpy()[eligible]
            if not legacy_any:
                probability_for_decision = (pred['bond_fail_logits'].argmax(-1).numpy() > 0).astype(float)
            time_class = (pred['bond_fail_logits'][:,1:].argmax(-1)+1 if legacy_any else pred['bond_fail_logits'].argmax(-1)).numpy()
            targets_util = batch['bond_peak_util'].numpy()[eligible]
        if window_factor > 1:
            ids = batch['bond_ids'].numpy()[eligible]
            window = windows.setdefault((name,fi//window_factor), [set(), [set() for _ in thresholds]])
            window[0].update(map(int,ids[truth]))
        for j, threshold in enumerate(thresholds):
            positive = (probability_for_decision if not isinstance(model, PhysicsImpactNet) and not legacy_any else probability) >= threshold
            counts = np.array([np.count_nonzero(positive & truth), np.count_nonzero(positive & ~truth), np.count_nonzero(~positive & truth)])
            totals[j] += counts
            if window_factor > 1: window[1][j].update(map(int,ids[positive]))
            family_counts.setdefault(family, np.zeros_like(totals))[j] += counts
        hit = truth & ((probability_for_decision if not isinstance(model, PhysicsImpactNet) and not legacy_any else probability) >= getattr(model, 'threshold', .5))
        correct_time += int(np.count_nonzero(time_class[hit] == truth_class[hit])); timed += int(hit.sum())
        bins_error += float(np.abs(time_class[hit]-truth_class[hit]).sum())
        util_abs += float(np.abs(pred['bond_peak_util'].numpy()[eligible]-targets_util).sum()); util_n += len(truth)
        brier_sum += float(np.square(probability-truth).sum())
        if isinstance(model, PhysicsImpactNet):
            predicted_delta = model.decode_delta(pred).numpy()
            target_delta = batch['chunk_state_delta'].numpy()
        else:
            predicted_delta = pred['chunk_state_delta'].numpy().copy()
            target_delta = batch['chunk_state_delta'].numpy().copy()
            for a in (predicted_delta,target_delta):
                a[:,:3] *= batch['_scales']['he'];a[:,6:9] *= batch['_scales']['he']
        valid = batch.get('delta_valid',torch.ones(len(target_delta),dtype=torch.bool)).numpy()
        for g in range(4):
            delta_squared[g] += np.square(predicted_delta[valid,g*3:g*3+3]-target_delta[valid,g*3:g*3+3]).sum()
            zero_squared[g] += np.square(target_delta[valid,g*3:g*3+3]).sum()
        delta_n += int(valid.sum()) * 3
        total_delta_chunks += len(target_delta)
    candidates = [dict(threshold=float(t), **counts_metrics(*map(int, c))) for t, c in zip(thresholds, totals)]
    window_totals = np.zeros_like(totals)
    for truth, predictions in windows.values():
        for i,positive in enumerate(predictions):
            window_totals[i] += [len(truth & positive),len(positive-truth),len(truth-positive)]
    window_candidates = [counts_metrics(*map(int,c)) for c in window_totals] if window_factor > 1 else candidates
    best = max(range(len(candidates)), key=lambda i: window_candidates[i]['f1'] or 0.)
    world_metrics=None
    if isinstance(model,WorldMotionImpactNet):
        keys=('position_m','rotation_residual_rad','velocity_mps','angular_velocity_radps')
        world_metrics={'contract':model.world_target_contract,'scored_chunk_intervals':world_count,
                       'rmse':dict(zip(keys,np.sqrt(world_squared/(3*world_count)).tolist() if world_count else [None]*4)),
                       'constant_velocity_baseline_rmse':dict(zip(keys,np.sqrt(world_zero/(3*world_count)).tolist() if world_count else [None]*4)),
                       'orientation_geodesic_rmse_rad':float(np.sqrt(world_angle_squared/world_count)) if world_count else None,
                       'baseline_orientation_geodesic_rmse_rad':float(np.sqrt(world_angle_zero/world_count)) if world_count else None,
                       'scope':'Oracle-start world endpoint residuals, valid across fragment coordinate resets; no autonomous rollout.'}
    return {'cached_response':response_metrics.report() if response_metrics else None,'full_joint_history':exact_history.report() if exact_history is not None else None,'world_motion':world_metrics,'normalized_state_mse':state_objective/max(1,len(records)) if isinstance(model,StatePreviewImpactNet) else None,
            'normalized_history_mse':history_objective/max(1,len(records)) if isinstance(model,StatePreviewImpactNet) else None,
            'selection_f1':window_candidates[best]['f1'],
            'window_factor':window_factor, 'window_metrics':{**window_candidates[best], 'windows':len(windows) if window_factor > 1 else len(records)},
            **candidates[best], 'frames': len(records), 'bonds': util_n,
            'util_log_mae': util_abs / max(1, util_n), 'brier': brier_sum/max(1, util_n),
            'fracture_cdf_score':{'all':float(cdf_sum.sum()/cdf_count.sum()) if cdf_count.sum() else None,
                                  'no_break':float(cdf_sum[0]/cdf_count[0]) if cdf_count[0] else None,
                                  'break':float(cdf_sum[1]/cdf_count[1]) if cdf_count[1] else None,
                                  'bond_intervals':cdf_count.tolist(),'order':['no_break','break']} if isinstance(model,PhysicsImpactNet) else None,
            'correct_time_bin_on_detected': correct_time / max(1, timed), 'mean_bin_error_on_detected': bins_error/max(1, timed),
            'timing_metrics_threshold': getattr(model, 'threshold', .5),
            'motion_scored_chunk_intervals':delta_n//3,
            'motion_fragment_reset_excluded_chunk_intervals':total_delta_chunks-delta_n//3,
            'state_rmse':dict(zip(('u_m','theta_rad','v_mps','omega_radps'),np.sqrt(delta_squared/delta_n).tolist() if delta_n else [None]*4)),
            'zero_increment_rmse':dict(zip(('u_m','theta_rad','v_mps','omega_radps'),np.sqrt(zero_squared/delta_n).tolist() if delta_n else [None]*4)),
            'families': {f: counts_metrics(*map(int, c[best])) for f, c in family_counts.items()},
            'threshold_candidates': candidates if calibrate else []}


class TrainingSampler:
    """Deterministic training-only ordering with a fixed optimizer-update budget.

    Frame mode retains the historical shuffled pass. Physical-group mode cycles
    equally through trajectory groups, drawing a shuffled non-repeating queue of
    intervals within each group. Queues persist across epochs, so long clips are
    not repeatedly restarted at their prefixes. Every interval remains eligible.
    """
    def __init__(self, records, groups, rng, policy='frame', oversample_breaks=False, updates_per_epoch=None):
        if policy not in ('frame','physical-group'):raise ValueError('unknown sampling policy')
        if policy!='frame' and oversample_breaks:raise ValueError('break oversampling cannot be combined with physical-group sampling')
        if not records:raise ValueError('cannot sample empty training records')
        if updates_per_epoch is not None and (policy!='physical-group' or not isinstance(updates_per_epoch,int) or isinstance(updates_per_epoch,bool) or updates_per_epoch<1):
            raise ValueError('explicit update budget requires physical-group sampling and a positive integer')
        self.records,self.rng,self.policy=records,rng,policy
        self.oversample_breaks=oversample_breaks
        self.groups={}
        for i,record in enumerate(records):
            self.groups.setdefault(groups[record[0]],[]).append(i)
        self.keys=sorted(self.groups)
        self.group_queue=[];self.frame_queues={k:[] for k in self.keys}
        self.visits=[0]*len(records)
        self.steps=len(records)+(sum(bool((r[3]['bond_fail_class']>0).any()) for r in records) if oversample_breaks else 0)
        if updates_per_epoch is not None:self.steps=updates_per_epoch

    def epoch(self):
        if self.policy=='frame':
            order=list(range(len(self.records)))
            if self.oversample_breaks:
                order += [i for i,r in enumerate(self.records) if (r[3]['bond_fail_class']>0).any()]
        else:
            order=[]
            for _ in range(self.steps):
                if not self.group_queue:
                    self.group_queue=self.keys.copy();self.rng.shuffle(self.group_queue)
                key=self.group_queue.pop()
                if not self.frame_queues[key]:
                    self.frame_queues[key]=self.groups[key].copy();self.rng.shuffle(self.frame_queues[key])
                order.append(self.frame_queues[key].pop())
        self.rng.shuffle(order)
        counts=[0]*len(self.records)
        for i in order:counts[i]+=1;self.visits[i]+=1
        report={'policy':self.policy,'updates':len(order),
                'unique_intervals':sum(c>0 for c in counts),
                'cumulative_unique_intervals':sum(c>0 for c in self.visits),
                'cumulative_min_interval_visits':min(self.visits),
                'cumulative_max_interval_visits':max(self.visits),
                'group_updates':{k:sum(counts[i] for i in ids) for k,ids in self.groups.items()}}
        return order,report


def fit(args):
    out = args.out
    if out.exists():
        raise FileExistsError(f'{out} already exists; choose a new run directory')
    retain_physical=getattr(args,'retain_physical_checkpoints',False)
    if retain_physical and (args.all_data or args.arch not in ('physics-response-history-native-v18','physics-projectile-response-native-v22','physics-projectile-response-si-v23')):
        raise ValueError('physical checkpoint retention requires V18/V22/V23 and a held-out validation partition')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    constructors = {'physics-v2':PhysicsImpactNet, 'physics-preview-v3':PreviewImpactNet,
                    'physics-preview-state-v4':StatePreviewImpactNet,
                    'physics-preview-world-v5':WorldStateImpactNet,'physics-preview-world-control-v5':WorldInputControlNet,'physics-preview-world-motion-v6':WorldMotionImpactNet,'physics-forcing-preview-v7':ForcingPreviewImpactNet,'physics-forcing-native-v7':ForcingNativeImpactNet,'physics-constitutive-native-v10':ConstitutiveNativeImpactNet,'physics-history-native-v11':HistoryCarryImpactNet,'physics-history-residual-native-v12':ResidualHistoryImpactNet,'physics-projectile-history-native-v13':ProjectileHistoryImpactNet,'physics-consistent-history-native-v15':ConsistentHistoryImpactNet,'physics-response-history-native-v18':ResponseHistoryImpactNet,'physics-projectile-response-native-v22':ProjectileResponseImpactNet,'physics-projectile-response-si-v23':PhysicalResponseImpactNet}
    model = constructors[args.arch](args.h, args.p, args.threshold) if args.arch != 'legacy' else LearnedImpactNet(args.h, args.p)
    frozen_source = initialize_frozen_state_heads(model,args.freeze_backbone_from) if args.freeze_backbone_from else None
    if frozen_source:
        args.threshold = model.threshold
        if args.split_from is None:args.split_from=args.freeze_backbone_from/'split.json'
    if isinstance(model,ResponseHistoryImpactNet):model.response_scale_mode=getattr(args,'response_scale_mode','floor-1e-4')
    records, split = load_records(args.dataset, model, args.split_seed, args.all_data, args.split_from,getattr(args,'fragment_cut_weight',0.))
    if isinstance(model, StatePreviewImpactNet): model.fit_target_scales(records['train'])
    if not records['train'] or (not args.all_data and not records['validation']):
        raise ValueError('training and validation partitions must be nonempty; use --all-data only for a deployment refit')
    out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(response_objective_contract=getattr(model,'response_objective_contract','encoded-residual-rms/1' if isinstance(model,ResponseHistoryImpactNet) else None),response_input_contract=getattr(model,'response_input_contract',None),response_output_contract=getattr(model,'response_output_contract',None),connectivity_input_contract=getattr(model,'connectivity_input_contract',None),projectile_input_contract=getattr(model,'projectile_input_contract',None),bond_policy=getattr(model,'bond_policy','first_failure'),joint_history_output_contract=getattr(model,'joint_history_output_contract',None),joint_history_input_contract=getattr(model,'joint_history_input_contract',None),material_input_contract=getattr(model,'material_input_contract',None),forcing_input_contract=getattr(model,'forcing_input_contract',None),input_contract=getattr(model,'input_contract','causal_preview' if isinstance(model,PreviewImpactNet) else 'oracle_interval'),world_target_contract=getattr(model,'world_target_contract',None),architecture=args.arch, parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_source=frozen_source,
                  checkpoint_selection='last epoch; no validation or threshold calibration' if args.all_data else 'validation normalized motion/history objective; source break threshold retained' if frozen_source else 'validation break F1 and threshold',
                  dataset=str(args.dataset.resolve()), torch_version=torch.__version__, numpy_version=np.__version__, python_version=platform.python_version(),
                  evaluation='all-data in-sample refit' if args.all_data else ('physical-group heldout, frame-start state and prescribed forcing only; no contact preview' if getattr(model,'input_contract',None)=='frame_start_forcing' else 'physical-group heldout, causal contact preview; oracle start states' if isinstance(model, PreviewImpactNet) else 'physical-group heldout, teacher-forced interval loads'),
                  gpu_export='unsupported; PyTorch only; runtime feature/rollout parity not established',
                  manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest())
    (out/'split.json').write_text(json.dumps(split, indent=2)+'\n')
    (out/'run.json').write_text(json.dumps(config, indent=2)+'\n')
    source_dir = out/'sources';source_dir.mkdir()
    for name in ('fragment_loss.py','topology_metrics.py','response_model.py','response_metrics.py','checkpoint_candidates.py','physical_response_model.py'):
        shutil.copyfile(Path(__file__).with_name(name),source_dir/name)
    for name in ('train_physics.py','physics.py','model.py','dataset.py','normalize.py','losses.py','schema.py','world_inputs.py','world_targets.py','forcing_inputs.py','joint_history.py','material_inputs.py','constitutive_model.py','history_model.py','history_metrics.py','projectile_inputs.py','projectile_model.py','consistent_model.py','state_consistency.py','response_history.py'):
        shutil.copyfile(Path(__file__).with_name(name),source_dir/name)
    print(json.dumps({**config, 'frames': split['frames'], 'duplicate_frames_removed': split['deduplicated_training_frames']}), flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    sampler = TrainingSampler(records['train'], split['physical_groups'], rng, args.sampling, args.oversample_breaks, getattr(args,'updates_per_epoch',None))
    steps_per_epoch = sampler.steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch if args.schedule_by_step else args.epochs, eta_min=args.lr*.1)
    history, best_score = [], -float('inf')
    from .checkpoint_candidates import PhysicalCandidates
    physical_candidates=PhysicalCandidates(out/'physical-candidates') if retain_physical else None
    start = time.monotonic()
    for epoch in range(args.epochs):
        order, sampling_report = sampler.epoch()
        summed = {}
        model.train()
        for i in order:
            batch = records['train'][i][3]
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            objective = model.state_loss if isinstance(model, StatePreviewImpactNet) else physics_loss
            parts = objective(pred, batch, positive_weight=args.positive_weight, state_weight=args.state_weight,
                                 history_weight=args.history_weight,fracture_cdf_weight=getattr(args,'fracture_cdf_weight',0.),fracture_objective=getattr(args,'fracture_objective','factorized'),**({'world_weight':args.world_motion_weight} if isinstance(model,WorldMotionImpactNet) else {}),**({'response_weight':getattr(args,'response_weight',.05)} if isinstance(model,ResponseHistoryImpactNet) else {})) if isinstance(model, PhysicsImpactNet) else total_loss(pred, batch)
            if getattr(args,'fragment_cut_weight',0.):
                cut_loss=fragment_cut_loss(pred['break_logit'],batch['fragment_cut_targets'])
                parts['loss']=parts['loss']+args.fragment_cut_weight*cut_loss
                parts['fragment_cut']=cut_loss.detach()
            if not torch.isfinite(parts['loss']):
                raise ValueError(f'nonfinite loss epoch {epoch} {records["train"][i][:3]}')
            parts['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if args.schedule_by_step: scheduler.step()
            for k, v in parts.items():
                summed[k] = summed.get(k, 0.) + float(v.detach())
        if not args.schedule_by_step: scheduler.step()
        should_evaluate = (epoch+1) % args.eval_every == 0 or epoch+1 == args.epochs
        validation = evaluate(model, records['validation'], calibrate=not bool(frozen_source), window_factor=args.evaluation_window) if records['validation'] and should_evaluate else {}
        if physical_candidates is not None and validation:
            physical_candidates.consider(model,validation,epoch+1,config,split)
        score = -(args.state_weight*validation['normalized_state_mse'] + args.history_weight*validation['normalized_history_mse']) if frozen_source and validation else (validation.get('selection_f1') or 0.)
        if args.all_data or (validation and score > best_score):
            best_score = score
            config['selected_epoch'] = epoch+1
            config['threshold'] = validation.get('threshold', args.threshold)
            torch.save(model.state_dict(), out/'state_dict.pt')
            (out/'run.json').write_text(json.dumps(config, indent=2)+'\n')
        row = {'epoch': epoch+1, 'seconds': time.monotonic()-start,
               'loss': {k: v/len(order) for k, v in summed.items()}, 'validation': validation, 'sampling': sampling_report}
        history.append(row)
        (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps({'epoch':epoch+1, 'seconds':round(row['seconds'],1), **row['loss'],
                          'val_state_mse':validation.get('normalized_state_mse'), 'val_history_mse':validation.get('normalized_history_mse'),
                          'val_f1':validation.get('f1'), 'val_window_f1':validation.get('selection_f1'), 'val_precision':validation.get('precision'),
                          'val_recall':validation.get('recall'), 'threshold':validation.get('threshold')}), flush=True)
    (out/'sampling.json').write_text(json.dumps({'policy':args.sampling,'optimizer_updates':sum(sampler.visits),
        'intervals':[{'event':r[0],'frame':r[2],'group':split['physical_groups'][r[0]],'visits':n} for r,n in zip(records['train'],sampler.visits)]},indent=2)+'\n')
    model.load_state_dict(torch.load(out/'state_dict.pt', weights_only=True))
    if isinstance(model, PhysicsImpactNet):
        model.threshold = config['threshold']
    metrics = {p: evaluate(model, r, window_factor=args.evaluation_window) for p, r in records.items() if r}
    metrics['scope'] = config['evaluation']; metrics['selected_epoch'] = config['selected_epoch']
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({p: {k: v for k, v in m.items() if k in ('f1','precision','recall','util_log_mae','tp','fp','fn')} for p,m in metrics.items() if isinstance(m, dict)}), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arch', choices=['physics-v2', 'physics-preview-v3', 'physics-preview-state-v4', 'physics-preview-world-v5', 'physics-preview-world-control-v5', 'physics-preview-world-motion-v6', 'physics-forcing-preview-v7', 'physics-forcing-native-v7', 'physics-constitutive-native-v10', 'physics-history-native-v11', 'physics-history-residual-native-v12', 'physics-projectile-history-native-v13', 'physics-consistent-history-native-v15', 'physics-response-history-native-v18', 'physics-projectile-response-native-v22', 'physics-projectile-response-si-v23', 'legacy'], default='physics-v2')
    p.add_argument('--split-from', type=Path, help='Reuse exact scenario membership from an earlier split.json; reject missing/conflicting trajectories')
    p.add_argument('--freeze-backbone-from',type=Path,help='State-v4 only: retain a preview-v3 checkpoint and train only motion/history heads; select using validation state loss')
    p.add_argument('--evaluation-window', type=int, default=1, help='Group this many adjacent intervals per event for validation F1/threshold selection; count each bond once')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--schedule-by-step', action='store_true', help='Cosine schedule per optimizer update, for matched-update experiments')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--updates-per-epoch',type=int,help='Physical-group sampling only: explicit update budget independent of dataset size; queues persist across epochs')
    p.add_argument('--h', type=int, default=48); p.add_argument('--p', type=int, default=16)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--split-seed', type=int, default=17)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--positive-weight', type=float, default=3.)
    p.add_argument('--fracture-objective',choices=['factorized','event-nll'],default='factorized',help='Existing weighted BCE plus conditional timing, or normalized no-break/time-bin likelihood (requires positive weight 1)')
    p.add_argument('--state-weight', type=float, default=.1)
    p.add_argument('--world-motion-weight',type=float,default=.1,help='V6 only: auxiliary world ballistic-residual MSE weight')
    p.add_argument('--history-weight', type=float, default=.05)
    p.add_argument('--response-weight',type=float,default=.05,help='V18/V22/V23 only: normalized cached-response loss, averaging active-end and retired-cache cohorts')
    p.add_argument('--response-scale-mode',choices=['floor-1e-4','nonzero-rms'],default='floor-1e-4',help='V18/V22/V23 only: baseline fixed encoded RMS floor, or preserve every nonzero training RMS with unit scale for exactly constant fields')
    p.add_argument('--retain-physical-checkpoints',action='store_true',help='V18/V22/V23 with validation only: also retain lowest validation active-force, stored-energy, and world-position error checkpoints; main fracture selection stays unchanged')
    p.add_argument('--fracture-cdf-weight',type=float,default=0.,help='Optional cumulative fracture-probability score; zero preserves the baseline objective')
    p.add_argument('--fragment-cut-weight',type=float,default=0.,help='Optional weakest-boundary loss using training endpoint topology; requires runtime-alive history; zero preserves baseline')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--sampling',choices=['frame','physical-group'],default='frame',help='Fixed update budget: shuffled intervals, or equal physical-trajectory visits with persistent shuffled interval queues')
    p.add_argument('--oversample-breaks', action='store_true')
    p.add_argument('--all-data', action='store_true', help='Final refit only: no held-out claims; threshold must come from a separate validation run')
    args = p.parse_args()
    if args.fracture_objective=='event-nll' and (args.positive_weight!=1 or args.arch=='legacy'):
        p.error('event-nll requires a physical-feature model and --positive-weight 1')
    if not np.isfinite(args.fracture_cdf_weight) or args.fracture_cdf_weight<0:p.error('fracture CDF weight must be finite and nonnegative')
    if not np.isfinite(args.fragment_cut_weight) or args.fragment_cut_weight<0:p.error('fragment cut weight must be finite and nonnegative')
    if args.arch=='legacy' and args.fracture_cdf_weight:p.error('fracture CDF objective requires a physical-feature model')
    if not np.isfinite(args.response_weight) or args.response_weight<0:p.error('response weight must be finite and nonnegative')
    if args.response_scale_mode!='floor-1e-4' and args.arch not in ('physics-response-history-native-v18','physics-projectile-response-native-v22','physics-projectile-response-si-v23'):p.error('response normalization modes require V18/V22/V23')
    if args.world_motion_weight<0:p.error('world-motion weight must be nonnegative')
    if args.updates_per_epoch is not None and (args.sampling!='physical-group' or args.updates_per_epoch<1):p.error('--updates-per-epoch requires physical-group sampling and a positive value')
    if args.sampling!='frame' and args.oversample_breaks:p.error('physical-group sampling cannot be combined with --oversample-breaks')
    if args.split_from: args.split_from = args.split_from.resolve()
    if args.freeze_backbone_from:
        args.freeze_backbone_from=args.freeze_backbone_from.resolve()
        if args.arch!='physics-preview-state-v4':p.error('--freeze-backbone-from requires physics-preview-state-v4')
    if args.epochs < 1 or args.eval_every < 1 or args.evaluation_window < 1 or not 0 < args.threshold < 1 or min(args.positive_weight, args.lr) <= 0:
        p.error('epochs, learning rate and positive weight must be positive; threshold must be in (0,1)')
    fit(args)


if __name__ == '__main__':
    main()
