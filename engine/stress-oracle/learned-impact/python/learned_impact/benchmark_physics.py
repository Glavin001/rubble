"""Synchronized CPU/MPS timing of physical-feature models at explicit graph sizes.

Synthetic topology is for compute scaling only, not physical accuracy. Resident
inference excludes contacts, graph updates, feature construction and rendering.
"""
from __future__ import annotations
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from .dataset import synthetic_batch,bidirectional_csr
from .physics import PhysicsImpactNet,PreviewImpactNet
from .visualize import load_checkpoint


def make_graph(n,e,model,projectiles=1,forcing_loads=2):
    b=synthetic_batch(n=n,e=e)
    # Distinct sparse edges, unlike repeating the same ring edge E/N times.
    i=np.arange(e,dtype=np.int64)
    endpoints=np.stack([i%n,(i%n+1+(i//n)*(max(2,n//100)))%n],axis=1)
    if np.any(endpoints[:,0]==endpoints[:,1]):raise ValueError('synthetic scale selected self-edges')
    row,col=bidirectional_csr(endpoints,n)
    b.update(bond_endpoints=torch.from_numpy(endpoints),csr_row_ptr=torch.from_numpy(row),csr_col=torch.from_numpy(col))
    if isinstance(model,PreviewImpactNet):
        b['chunk_preview_load_bins']=b['chunk_load_bins'].clone()
        b['chunk_preview_contact']=torch.zeros(n)
    if getattr(model,'requires_world_inputs',False):
        if projectiles<0:raise ValueError('projectiles must be nonnegative')
        i=torch.arange(n,dtype=torch.float32)
        centers=torch.stack([i.remainder(100)*.1,torch.floor(i/100)*.1,torch.ones(n)*.5],-1)
        b.update(world_center=centers,world_displacement=torch.zeros(n,3),
                 world_velocity=torch.zeros(n,3),world_angular_velocity=torch.zeros(n,3),
                 world_rotation=torch.eye(3).expand(n,3,3).clone(),world_active=torch.ones(n),world_ground=torch.tensor([0.,1.]),
                 impactor_center=torch.tensor([[5.,-1.,.5]]).repeat(projectiles,1),
                 impactor_velocity=torch.tensor([[0.,15.,0.]]).repeat(projectiles,1),
                 impactor_rotation=torch.eye(3).expand(projectiles,3,3).clone(),
                 impactor_properties=torch.tensor([[.5,1.,.75,1500.,1.,0.,4e5,2e4,1.]]).repeat(projectiles,1))
    if getattr(model,'requires_forcing_inputs',False):
        if forcing_loads<0:raise ValueError('forcing loads must be nonnegative')
        kind=1+torch.arange(forcing_loads).remainder(2);blast=kind==2
        direction=torch.zeros(forcing_loads,3);direction[~blast,2]=1.
        position=torch.zeros(forcing_loads,3);position[blast]=torch.tensor([5.,0.,1.])
        parameters=torch.zeros(forcing_loads,4);parameters[~blast,2]=3e4
        charge=torch.zeros(forcing_loads,2);charge[blast,0]=2.
        b.update(forcing_kind=kind,forcing_direction=direction,forcing_position=position,
                 forcing_curve_kind=torch.where(blast,5,0),forcing_curve_parameters=parameters,
                 forcing_blast=charge,forcing_selected=torch.ones(forcing_loads,n,dtype=torch.bool),
                 forcing_table_points=torch.zeros(0,2),forcing_table_owner=torch.zeros(0,dtype=torch.int64))
    if getattr(model,'requires_exact_joint_history',False):
        b['bond_exact_history']=torch.zeros(e,31)
        b['bond_exact_history'][:,21:23]=1
        b['bond_exact_history'][:,24]=1
        b['bond_exact_history'][:,29:31]=1
    if getattr(model,'bond_policy',None)=='runtime_alive':
        b['bond_failure_eligible']=torch.ones(e,dtype=torch.bool)
    if getattr(model,'requires_material_inputs',False):
        b.update(chunk_material_features=torch.zeros(n,26),bond_material_features=torch.zeros(e,26),
                 constitutive_flags=torch.zeros(20))
    if getattr(model,'requires_projectile_inputs',False):
        props=torch.zeros(projectiles,17);props[:,0]=1500.;props[:,1:10]=torch.eye(3).reshape(1,9)*100
        b.update(impactor_angular_velocity=torch.zeros(projectiles,3),impactor_runtime_properties=props,
                 impactor_material_features=torch.zeros(projectiles,26))
    if getattr(model,'connectivity_input_contract',None):
        b.update(bond_has_rebar=torch.zeros(e,dtype=torch.bool),bond_damage_complete_start=torch.zeros(e,dtype=torch.bool))
    if getattr(model,'requires_bond_response_history',False):
        b.update(bond_response_encoded=torch.zeros(e,18),bond_response_alive=torch.ones(e,dtype=torch.bool))
    return b


def sync(device):
    if device=='mps':torch.mps.synchronize()


def timed(fn,device,repeats):
    for _ in range(3):fn()
    sync(device);values=[]
    for _ in range(repeats):
        sync(device);start=time.perf_counter();fn();sync(device)
        values.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95)),'samples_ms':values}


@torch.inference_mode()
def benchmark(weights,n=10000,e=30000,repeats=12,threads=4,projectiles=1,forcing_loads=2):
    torch.set_num_threads(threads)
    model,meta=load_checkpoint(weights)
    if not isinstance(model,PhysicsImpactNet):raise ValueError('physical-feature checkpoint required')
    raw=make_graph(n,e,model,projectiles,forcing_loads);cpu=model.prepare_batch(raw)
    reference=model(cpu)
    if not all(torch.isfinite(x).all() for x in reference.values()):raise ValueError('nonfinite CPU output')
    results={'cpu_resident':timed(lambda:model(cpu),'cpu',repeats),
             'cpu_with_features':timed(lambda:model(model.prepare_batch(raw)),'cpu',repeats)}
    if torch.backends.mps.is_available():
        gpu_model,_=load_checkpoint(weights);gpu_model.to('mps')
        def transfer(b):return {k:v.to('mps') if isinstance(v,torch.Tensor) else v for k,v in b.items()}
        # Keep the exact int64 training index contract.
        gpu=transfer(cpu)
        gpu_raw=transfer(raw)
        result=gpu_model(gpu);sync('mps')
        outputs={k:v.cpu() for k,v in result.items()}
        if not all(torch.isfinite(x).all() for x in outputs.values()):raise ValueError('nonfinite MPS output')
        results['mps_resident']=timed(lambda:gpu_model(gpu),'mps',repeats)
        results['mps_with_gpu_features']=timed(lambda:gpu_model(gpu_raw),'mps',repeats)
        results['cpu_features_transfer_mps']=timed(lambda:gpu_model(transfer(model.prepare_batch(raw))),'mps',repeats)
        def difference(a,b):
            return float(((a!=b).float() if a.dtype==torch.bool else (a-b).abs()).max())
        results['cpu_mps_output_max_abs']={k:difference(reference[k],v) for k,v in outputs.items()}
        results['synthetic_break_decision_disagreements']=int((model.predict_classes(reference)!=gpu_model.predict_classes(outputs)).sum())
        gpu_features_outputs={k:v.cpu() for k,v in gpu_model(gpu_raw).items()}
        results['gpu_features_output_max_abs']={k:difference(reference[k],v) for k,v in gpu_features_outputs.items()}
        results['gpu_features_synthetic_class_disagreements']=int((model.predict_classes(reference)!=gpu_model.predict_classes(gpu_features_outputs)).sum())
    return {'runtime':{'platform':platform.platform(),'torch':torch.__version__,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available()},'checkpoint':meta,'nodes':n,'edges':e,'world_projectiles':projectiles if getattr(model,'requires_world_inputs',False) else None,'prescribed_loads':forcing_loads if getattr(model,'requires_forcing_inputs',False) else None,'threads':threads,'results':results,
            'limits':['Synthetic graph timing is not physical validation at 10,000 bodies.',
                      'Resident timings exclude contacts, feature construction/transfers, graph changes, state application and rendering.',
                      'The transfer measurement includes CPU feature encoding and transfer of the full batch; an optimized resident engine path is not implemented.',
                      'MPS numerical differences on synthetic inputs do not establish Slang/export or real-trajectory parity.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nodes',type=int,default=10000);p.add_argument('--edges',type=int,default=30000)
    p.add_argument('--repeats',type=int,default=12);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--projectiles',type=int,default=1,help='Number of synthetic projectiles for a world-input model; all are encoded')
    p.add_argument('--forcing-loads',type=int,default=2,help='Number of synthetic prescribed loads; all are encoded')
    a=p.parse_args();r=benchmark(a.weights,a.nodes,a.edges,a.repeats,a.threads,a.projectiles,a.forcing_loads)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='checkpoint'}))


if __name__=='__main__':main()
