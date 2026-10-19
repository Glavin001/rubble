"""Causal frame-start force-trajectory pilot using the existing graph encoder.

Predicts total generalized force at fixed time knots once per frame. This is a
component experiment, not a complete stress-provider model: joint history,
fracture timing, elastic/contact transfer force and work outputs remain required.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .projectile_model import ProjectileResponseImpactNet
from .stress_phases import PhaseHistory
from .temporal_basis import hat_basis
from .visualize import load_checkpoint
from .model import MLP
from .physics import symlog

class ForceTrajectoryNet(ProjectileResponseImpactNet):
    architecture='physics-native-force-trajectory-v28'
    def __init__(self,h=48,p=16,threshold=.5,knots=16):
        super().__init__(h,p,threshold);self.knots=knots
        self.trajectory_head=nn.Linear(h,knots*6)
        nn.init.zeros_(self.trajectory_head.weight);nn.init.zeros_(self.trajectory_head.bias)
        self.register_buffer('trajectory_scale',torch.ones(6))

    def edge_outputs(self,edge,b):
        residual=self.trajectory_head(edge).reshape(len(edge),self.knots,6)
        return {'force_residual_knots':residual*self.trajectory_scale}

    def forward(self,b):
        # Reuse the trained causal encoder, not the legacy endpoint decoder.
        if '_node_features' not in b:b=self.prepare_batch(b)
        node,edge=self.encode_nodes(b),self.encode_edges(b)
        for i,mp in enumerate(self.mp):
            node,edge=mp(node,edge,b['csr_row_ptr'],b['csr_col'],b['bond_endpoints'])
            if i==1:node=self.attn(node)
        return self.edge_outputs(edge,b)

class KinematicForceTrajectoryNet(ForceTrajectoryNet):
    architecture='physics-boundary-kinematic-force-trajectory-v30'

    def __init__(self,h=48,p=16,threshold=.5,knots=16,kinematic_mode='observed'):
        super().__init__(h,p,threshold,knots)
        if kinematic_mode not in ('observed','zero'):raise ValueError('invalid kinematic ablation mode')
        self.kinematic_mode=kinematic_mode
        self.boundary_encoder=MLP(24,h,h)
        nn.init.zeros_(self.boundary_encoder.net[-1].weight)
        nn.init.zeros_(self.boundary_encoder.net[-1].bias)
        self.register_buffer('boundary_scale',torch.ones(12))

    def prepare_batch(self,raw):
        x=raw.get('bond_boundary_kinematics')
        if x is None or x.shape!=(len(raw['bond_endpoints']),12) or not torch.isfinite(x).all():
            raise ValueError('requires finite current boundary kinematics')
        # Elastic load indicators, not an oracle constitutive query: the same
        # geometry/material/current-state inputs exist during learned inference.
        displacement=raw['bond_stiffness']*x[:,:6]/self.trajectory_scale
        increment=raw['bond_stiffness']*x[:,6:]*raw['dt']/self.trajectory_scale
        feature=torch.cat([symlog(x/self.boundary_scale),symlog(displacement),symlog(increment)],-1)
        if not torch.isfinite(feature).all():raise ValueError('nonfinite boundary features')
        if self.kinematic_mode=='zero':feature=torch.zeros_like(feature)
        return {**super().prepare_batch(raw),'_boundary_features':feature}

    def encode_edges(self,b):
        return super().encode_edges(b)+self.boundary_encoder(b['_boundary_features'])

    @torch.no_grad()
    def fit_boundary_scale(self,records):
        # Training inputs only, one isotropic RMS for each physical unit group.
        values=torch.cat([r['batch']['bond_boundary_kinematics'].double() for r in records])
        for start in [0,3,6,9]:
            value=float(values[:,start:start+3].square().mean().sqrt())
            self.boundary_scale[start:start+3]=value if value>0 else 1.

def query_force(pred,batch,basis):
    # Interpolate residuals, preserving the exact hold baseline for a zero head.
    return batch['bond_force_prev'][None,:,:]+torch.einsum('tk,ekc->tec',basis,pred['force_residual_knots'])

def phase_record(data,frame,event,partition,knots):
    indices=data.frame_phases[frame]
    batch=data.frame_inputs(frame);ids=batch['bond_ids'].numpy();slot_to_edge=np.full(len(data.meta['bond_keys']),-1,dtype=np.int64);slot_to_edge[ids]=np.arange(len(ids))
    target=np.zeros((len(indices),len(ids),6),dtype=np.float32);mask=np.zeros((len(indices),len(ids)),dtype=bool)
    for t,index in enumerate(indices):
        sl=data.phase_rows(index);edges=slot_to_edge[data.slots[sl]]
        if (edges<0).any():raise ValueError('new active edge absent at frame start')
        target[t,edges]=data.targets[sl,21:27];mask[t,edges]=True
    start=data.snapshot(frame)['world']['time'];dt=data.graph['dt']
    times=np.array([(data.phases[i]['time']-start)/dt for i in indices])
    if (times<0).any() or (times>1).any():raise ValueError('phase time outside frame')
    basis=hat_basis(times,np.linspace(0.,1.,knots)).astype(np.float32)
    weights=np.array([data.phases[i]['dt']/dt for i in indices],dtype=np.float32)
    return {'event':event,'partition':partition,'frame':frame,'batch':batch,
        'target':torch.from_numpy(target),'mask':torch.from_numpy(mask),'basis':torch.from_numpy(basis),
        'weights':torch.from_numpy(weights),'dt':dt}


def records(root,knots):
    root=Path(root);manifest=json.loads((root/'pilot-manifest.json').read_text())
    if manifest['status']!='completed_verified':raise ValueError('unverified phase pilot')
    result=[];sources={}
    for case in manifest['events']:
        path=root/'dataset'/case['event'];quality=json.loads((path/'audit.json').read_text())
        if quality['status']!='completed_verified' or not quality['original_world_replay_exact']:raise ValueError('missing phase quality check')
        for name,digest in quality['files_sha256'].items():
            if PhaseHistory._hash(path/name)!=digest:raise ValueError('phase source changed after audit')
        sources[case['event']]=PhaseHistory._hash(path/'audit.json')
        data=PhaseHistory(path)
        for frame in range(data.frames):result.append(phase_record(data,frame,case['event'],case['partition'],knots))
    return result,sources

def impulse_residual(values,r):
    """Signed interval impulse error / frame dt, using oracle participation labels.

    These are generalized bond impulses, not world contact impulses. Multiplying
    by frame dt recovers N s / N m s. This auxiliary target does not supply a
    runtime retirement schedule or integrate predicted forces into the world.
    """
    weight=r['weights'][:,None]*r['mask']
    return ((values-r['target'])*weight[:,:,None]).sum(0)

def force_loss(pred,r,scale,impulse_weight=0.):
    values=query_force(pred,r['batch'],r['basis'])
    weight=r['weights'][:,None]*r['mask']
    point=((((values-r['target'])/scale).square()*weight[:,:,None]).sum()/(6*weight.sum()))
    eligible=r['mask'].any(0)
    impulse=(impulse_residual(values,r)[eligible]/scale).square().mean()
    return point+impulse_weight*impulse

@torch.inference_mode()
def evaluate(model,items,baseline=False):
    model.eval();squared=torch.zeros(6,dtype=torch.float64);truth=torch.zeros_like(squared);mass=0.
    impulse_squared=torch.zeros_like(squared);impulse_truth=torch.zeros_like(squared)
    impulse_normalized=torch.zeros_like(squared);bond_intervals=0
    for r in items:
        if baseline:values=r['batch']['bond_force_prev'][None,:,:].expand_as(r['target'])
        else:values=query_force(model(r['batch']),r['batch'],r['basis'])
        weight=(r['weights'][:,None]*r['mask']).double();mass+=float(weight.sum())
        squared+=((values.double()-r['target'].double()).square()*weight[:,:,None]).sum((0,1))
        truth+=(r['target'].double().square()*weight[:,:,None]).sum((0,1))
        eligible=r['mask'].any(0);bond_intervals+=int(eligible.sum())
        delta=impulse_residual(values.double(),r)[eligible]
        target=(r['target'].double()*weight[:,:,None]).sum(0)[eligible]*r['dt']
        impulse_squared+=((delta*r['dt']).square()).sum(0)
        impulse_truth+=target.square().sum(0)
        impulse_normalized+=(delta/model.trajectory_scale.double()).square().sum(0)
    return {'frames':len(items),'time_weighted_bond_intervals':mass,
        'force_rmse_n':float((squared[:3].sum()/(3*mass)).sqrt()),'torque_rmse_nm':float((squared[3:].sum()/(3*mass)).sqrt()),
        'force_relative_l2':float((squared[:3].sum()/truth[:3].sum()).sqrt()),'torque_relative_l2':float((squared[3:].sum()/truth[3:].sum()).sqrt()),
        'normalized_mse':float((squared/model.trajectory_scale.double().square()).sum()/(6*mass)),
        'bond_force_impulse_rmse_ns':float((impulse_squared[:3].sum()/(3*bond_intervals)).sqrt()),
        'bond_torque_impulse_rmse_nms':float((impulse_squared[3:].sum()/(3*bond_intervals)).sqrt()),
        'bond_force_impulse_relative_l2':float((impulse_squared[:3].sum()/impulse_truth[:3].sum()).sqrt()) if impulse_truth[:3].sum()>0 else None,
        'bond_torque_impulse_relative_l2':float((impulse_squared[3:].sum()/impulse_truth[3:].sum()).sqrt()) if impulse_truth[3:].sum()>0 else None,
        'impulse_normalized_mse':float(impulse_normalized.sum()/(6*bond_intervals))}

def train(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed);np.random.seed(args.seed)
    out=Path(args.out);out.mkdir();started=time.monotonic()
    frozen=out/'sources';frozen.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,frozen/p.name)
    (out/'source-hashes.json').write_text(json.dumps({p.name:PhaseHistory._hash(p) for p in frozen.iterdir()},indent=2)+'\n')
    (out/'invocation.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    items,sources=records(args.dataset,args.knots)
    boundary_sources={}
    if args.kinematic_mode!='none':
        from .boundary_kinematics import attach
        boundary_sources=attach(items,args.boundary_root,sources)
    groups={part:[r for r in items if r['partition']==part] for part in ['train','validation','test']}
    if not all(groups.values()):raise ValueError('all three pilot partitions required')
    base,meta=load_checkpoint(Path(args.weights))
    if base.architecture!='physics-projectile-response-native-v22':raise ValueError('pilot requires the specified V22 base encoder')
    model=(ForceTrajectoryNet(h=meta['h'],p=meta['p'],threshold=base.threshold,knots=args.knots) if args.kinematic_mode=='none' else
        KinematicForceTrajectoryNet(h=meta['h'],p=meta['p'],threshold=base.threshold,knots=args.knots,kinematic_mode=args.kinematic_mode))
    missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
    expected={'trajectory_scale','trajectory_head.weight','trajectory_head.bias'}
    if args.kinematic_mode!='none':expected|={k for k in model.state_dict() if k.startswith('boundary_')}
    if set(missing)!=expected or unexpected:raise ValueError('base checkpoint contract mismatch')
    # Physical SI residual RMS on TRAIN ONLY, shared among xyz in each unit group.
    sums=torch.zeros(6,dtype=torch.float64);mass=0.
    for r in groups['train']:
        weight=(r['weights'][:,None]*r['mask']).double();mass+=float(weight.sum())
        delta=r['target'].double()-r['batch']['bond_force_prev'].double()[None,:,:]
        sums+=(delta.square()*weight[:,:,None]).sum((0,1))
    for sl in [slice(0,3),slice(3,6)]:
        value=float((sums[sl].sum()/(3*mass)).sqrt());model.trajectory_scale[sl]=value if value>0 else 1.
    if args.kinematic_mode!='none':model.fit_boundary_scale(groups['train'])
    initial=evaluate(model,groups['train']);baseline=evaluate(model,groups['train'],True)
    if initial!=baseline:raise ValueError('zero head must exactly reproduce hold-previous baseline')
    run_config={**vars(args),'architecture':model.architecture,'base_checkpoint_sha256':PhaseHistory._hash(Path(args.weights)/'state_dict.pt'),
        'training_residual_scale':model.trajectory_scale.tolist(),'source_quality_sha256':sources,
        'boundary_source_quality_sha256':boundary_sources,
        'loss':'time-weighted native-step SI force/torque squared residual plus impulse_weight times squared signed time-integrated residual / frame dt; train-only isotropic force/torque RMS normalization',
        'impulse_scope':'generalized bond impulses with oracle participation labels, not world contact or projectile impulses; no complete provider',
        'scope':__doc__,'runtime_compatible':False,'h':meta['h'],'p':meta['p']}
    (out/'config.json').write_text(json.dumps(run_config,indent=2)+'\n')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    best=float('inf');history=[];rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train();total=0.
        for i in rng.permutation(len(groups['train'])):
            r=groups['train'][i];optimizer.zero_grad();loss=force_loss(model(r['batch']),r,model.trajectory_scale,args.impulse_weight)
            if not torch.isfinite(loss):raise ValueError('nonfinite trajectory loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())
        validation=evaluate(model,groups['validation']);row={'epoch':epoch,'train_loss':total/len(groups['train']),'validation':validation};history.append(row)
        selection=validation['normalized_mse']+args.impulse_weight*validation['impulse_normalized_mse']
        if selection<best:
            best=selection;torch.save(model.state_dict(),out/'state_dict.pt');selected=epoch
        if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    model.load_state_dict(torch.load(out/'state_dict.pt',map_location='cpu',weights_only=True))
    metrics={part:{'model':evaluate(model,r),'hold_previous':evaluate(model,r,True)} for part,r in groups.items()}
    report={'status':'completed','selected_epoch':selected,'epochs':args.epochs,'updates':args.epochs*len(groups['train']),
        'seconds':time.monotonic()-started,'checkpoint_sha256':PhaseHistory._hash(out/'state_dict.pt'),'metrics':metrics,
        'scope':__doc__,'promoted':False}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset','weights','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--knots',type=int,default=16);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--impulse-weight',type=float,default=0.)
    p.add_argument('--boundary-root');p.add_argument('--kinematic-mode',choices=['none','zero','observed'],default='none')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,default=1e-3);a=p.parse_args()
    if a.epochs<=0 or a.knots<2 or a.threads<=0 or not np.isfinite(a.lr) or a.lr<=0:p.error('invalid training parameters')
    if not np.isfinite(a.impulse_weight) or a.impulse_weight<0:p.error('invalid impulse weight')
    if bool(a.boundary_root)!=(a.kinematic_mode!='none'):p.error('boundary root required exactly for kinematic ablations')
    train(a)

if __name__=='__main__':main()
