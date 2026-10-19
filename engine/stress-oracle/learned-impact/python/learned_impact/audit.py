"""Read-only checkpoint/data audit. Decoder sweeps are diagnostic, not validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from .dataset import event_batches, list_events
from .schema import load_event
from .normalize import normalize_batch
from .physics import PhysicsImpactNet
from .losses import total_loss, focal_ce, util_loss
from .visualize import load_checkpoint, counts_metrics


def audit(dataset, weights):
    torch.set_num_threads(4)
    model, meta = load_checkpoint(weights)
    if isinstance(model, PhysicsImpactNet):
        raise ValueError('This legacy normalization/gradient audit does not support physics models; use train_physics evaluation and preview_audit for their versioned contracts.')
    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    counts = {str(t): np.zeros(3, dtype=np.int64) for t in thresholds}
    counts['argmax_5way'] = np.zeros(3, dtype=np.int64)
    classes = np.zeros(5, dtype=np.int64)
    families = Counter(); dts = Counter(); sizes = []; durations = []
    signatures = {}; duplicate_frames = conflicting_frames = 0
    quiet = frames = 0; timing_correct = timing_count = timing_error = 0
    gradients = {}; loss_values = []; field_mse = np.zeros(4)
    support = Counter(); strengths = []; stiffness = []
    scales_example = None
    for path in list_events(dataset):
        ev = load_event(path); family = ev.manifest['structure_type']
        families[family] += 1; dts[str(ev.manifest['dt'])] += 1
        sizes.append(ev.manifest['n_chunks']); durations.append(ev.manifest['n_frames'] * ev.manifest['dt'])
        support.update(map(int, ev.static['chunk_support']))
        strengths.extend(ev.static['bond_strength'].reshape(-1).tolist())
        stiffness.extend(ev.static['bond_stiffness'].reshape(-1).tolist())
        for fi, raw in enumerate(event_batches(ev)):
            b = normalize_batch(raw); truth = b['bond_fail_class'].numpy(); yes = truth > 0
            classes += np.bincount(truth, minlength=5); frames += 1; quiet += int(not yes.any())
            keys = ['chunk_mass','chunk_inertia','chunk_half_extents','chunk_state','chunk_load_bins','chunk_gravity_local','bond_joint_state','bond_qs_force','bond_endpoints']
            digest = hashlib.sha256()
            for k in keys:
                v=b[k].numpy(); digest.update(str((k,v.shape)).encode());digest.update(v.tobytes())
            key=digest.hexdigest(); labels=hashlib.sha256(truth.tobytes()).hexdigest()
            if key in signatures:
                duplicate_frames += 1
                conflicting_frames += int(signatures[key] != labels)
            else: signatures[key] = labels
            with torch.no_grad():
                pred = model(b); probs=pred['bond_fail_logits'].softmax(-1).numpy()
                chosen=probs.argmax(-1); p_break=1-probs[:,0]
                for label, positive in [('argmax_5way',chosen>0), *((str(t),p_break>=t) for t in thresholds)]:
                    counts[label] += [np.count_nonzero(positive&yes),np.count_nonzero(positive&~yes),np.count_nonzero(~positive&yes)]
                found=yes & (chosen>0);timing_count+=int(found.sum());timing_correct+=int((chosen[found]==truth[found]).sum());timing_error+=int(np.abs(chosen[found]-truth[found]).sum())
                parts=total_loss(pred,b);loss_values.append([float(parts[k]) for k in ('fail','util','delta')])
                err=(pred['chunk_state_delta']-b['chunk_state_delta']).square()
                field_mse += np.array([float(err[:,i:i+3].mean()) for i in (0,3,6,9)])
            if family not in gradients and yes.any():
                pred=model(b)
                losses={'fail':focal_ce(pred['bond_fail_logits'],b['bond_fail_class']), 'util':util_loss(pred['bond_peak_util'],b['bond_peak_util']), 'delta':torch.nn.functional.mse_loss(pred['chunk_state_delta'],b['chunk_state_delta'])}
                shared=[p for name,p in model.named_parameters() if not name.startswith(('bond_fail.','bond_util.','chunk_delta.'))]
                gradients[family]={'event':path.name,'frame':fi}
                for k,v in losses.items():
                    grad=torch.autograd.grad(v,shared,retain_graph=True,allow_unused=True)
                    gradients[family][k]=float(torch.sqrt(sum(g.square().sum() for g in grad if g is not None)))
            if scales_example is None and b['_scales']['load'] > 2:
                doubled=dict(raw,chunk_load_bins=raw['chunk_load_bins']*2)
                scaled=normalize_batch(doubled)
                scales_example={'event':path.name,'frame':fi,'load_rms':b['_scales']['load'],'doubled_load_rms':scaled['_scales']['load'],'max_encoded_input_change':max(float((b[k]-scaled[k]).abs().max()) for k in keys if b[k].numel())}
    with torch.no_grad():
        x = torch.linspace(-1, 1, model.h)[None, :]
        broadcast = {str(n): float((model.attn(x.repeat(n, 1)) - x.repeat(n, 1))[0].norm()) for n in (8, 80, 800, 10000)}
    return {'scope':'All-data checkpoint evaluated on its training corpus. Threshold sweep is descriptive only; no threshold or weights changed.',
            'checkpoint':meta,'events':sum(families.values()),'frames':frames,'families':dict(families),'dt_event_counts':dict(dts),
            'chunks_min_max':[min(sizes),max(sizes)],'duration_seconds_min_max':[min(durations),max(durations)],
            'class_counts':classes.tolist(),'quiet_frames':quiet,'exact_normalized_input_duplicate_frames':duplicate_frames,'duplicate_inputs_with_conflicting_fail_labels':conflicting_frames,
            'decoder_sweep':{k:counts_metrics(*(int(x) for x in v)) for k,v in counts.items()},
            'timing_on_argmax_true_positives':{'n':timing_count,'correct_bin':timing_correct,'mean_absolute_bin_error':timing_error/max(1,timing_count)},
            'frame_mean_loss_components':dict(zip(['fail','util','delta'],np.mean(loss_values,axis=0).tolist())),
            'frame_mean_normalized_delta_mse_by_field':dict(zip(['u','theta','v','omega'],(field_mse/frames).tolist())),
            'shared_gradient_norm_examples':gradients,'load_scale_invariance_example':scales_example,
            'attention_identical_node_broadcast_norm':broadcast,
            'stored_support_counts':dict(support),'stored_bond_strength_range':[min(strengths),max(strengths)],'stored_bond_stiffness_range':[min(stiffness),max(stiffness)]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=audit(a.dataset,a.weights);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
