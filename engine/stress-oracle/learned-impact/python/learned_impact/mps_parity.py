"""Compare CPU and MPS inference on a checkpoint's actual held-out recordings."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .dataset import list_events,event_batches
from .schema import load_event
from .physics import PhysicsImpactNet
from .visualize import load_checkpoint,counts_metrics


@torch.inference_mode()
def parity(weights,dataset,gpu_features=False):
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this execution environment')
    torch.set_num_threads(4)
    cpu,meta=load_checkpoint(weights);gpu,_=load_checkpoint(weights);gpu.to('mps')
    if not isinstance(cpu,PhysicsImpactNet):raise ValueError('physical-feature checkpoint required')
    splits=json.loads((weights/'split.json').read_text())['events']
    maxima={};frames=bonds=break_disagreements=class_disagreements=0
    delta_max=torch.zeros(4);world_max=torch.zeros(4);prob_max=0.;counts=[[0,0,0],[0,0,0]]
    for path in list_events(dataset):
        if splits.get(path.name)!='test':continue
        for raw in event_batches(load_event(path,world_inputs=getattr(cpu,'requires_world_inputs',False),world_targets=getattr(cpu,'requires_world_targets',False),forcing_inputs=getattr(cpu,'requires_forcing_inputs',False),exact_joint_history=getattr(cpu,'requires_exact_joint_history',False),material_inputs=getattr(cpu,'requires_material_inputs',False),projectile_inputs=getattr(cpu,'requires_projectile_inputs',False),bond_response_history=getattr(cpu,'requires_bond_response_history',False)),bond_policy=getattr(cpu,'bond_policy','first_failure')):
            batch=cpu.prepare_batch(raw)
            transferred={k:v.to('mps') if isinstance(v,torch.Tensor) else v for k,v in (raw if gpu_features else batch).items()}
            a=cpu(batch);b={k:v.cpu() for k,v in gpu(transferred).items()}
            for key in a:
                if not torch.isfinite(b[key]).all():raise ValueError(f'nonfinite GPU {key}')
                if a[key].numel():
                    difference=(a[key]!=b[key]).float() if a[key].dtype==torch.bool else (a[key]-b[key]).abs()
                    maxima[key]=max(maxima.get(key,0.),float(difference.max()))
            ca,cb=cpu.predict_classes(a),cpu.predict_classes(b)
            class_disagreements+=int((ca!=cb).sum());break_disagreements+=int(((ca>0)!=(cb>0)).sum())
            if len(ca):prob_max=max(prob_max,float((a['break_logit'].sigmoid()-b['break_logit'].sigmoid()).abs().max()))
            d=(cpu.decode_delta(a)-cpu.decode_delta(b)).abs().reshape(-1,4,3)
            if len(d):delta_max=torch.maximum(delta_max,d.amax(dim=(0,2)))
            if getattr(cpu,'requires_world_targets',False):
                d=(cpu.decode_world_motion(a)-cpu.decode_world_motion(b)).abs().reshape(-1,4,3)
                if len(d):world_max=torch.maximum(world_max,d.amax(dim=(0,2)))
            truth=batch['bond_fail_class']>0
            for count,c in zip(counts,(ca,cb)):
                positive=c>0
                for i,n in enumerate(((positive&truth).sum(),(positive&~truth).sum(),(~positive&truth).sum())):count[i]+=int(n)
            frames+=1;bonds+=len(ca)
    if not frames:raise ValueError('no held-out records match saved split membership')
    return dict(checkpoint=meta,frames=frames,bond_intervals=bonds,gpu_feature_construction=gpu_features,output_max_abs=maxima,
                break_probability_max_abs=prob_max,break_decision_disagreements=break_disagreements,
                fail_class_disagreements=class_disagreements,
                physical_delta_max_abs=dict(zip(('u_m','theta_rad','v_mps','omega_radps'),delta_max.tolist())),
                physical_world_residual_max_abs=dict(zip(('position_m','rotation_rad','velocity_mps','angular_velocity_radps'),world_max.tolist())) if getattr(cpu,'requires_world_targets',False) else None,
                cpu=counts_metrics(*counts[0]),mps=counts_metrics(*counts[1]),
                scope='Forward numerical parity on oracle-conditioned held-out inputs; not rollout, Slang export, or an engine integration test.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','dataset','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-features',action='store_true',help='Construct features on GPU from raw state, instead of transferring CPU features')
    a=p.parse_args();r=parity(a.weights,a.dataset,a.gpu_features)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='checkpoint'}))


if __name__=='__main__':main()
