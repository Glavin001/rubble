"""Complete interval stress forecasts. Experimental; no deployment accuracy claim.

The graph is evaluated at frame start. Native stress phases query forecasts of
joint history, force, elastic response and work. No world poses are predicted or
pasted into the integrator. Current contact feedback within the interval is not
an input to these frozen forecasts; independent evaluation must expose that limit.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .trajectory_force import KinematicForceTrajectoryNet,records as force_records
from .boundary_kinematics import attach
from .stress_phases import PhaseHistory
from .joint_history import finite_features,SCALE as HISTORY_SCALE
from .temporal_basis import hat_basis
from .visualize import load_checkpoint

class CompleteTrajectoryNet(KinematicForceTrajectoryNet):
    architecture='physics-complete-native-trajectory-v31'
    def __init__(self,h=48,p=16,threshold=.5,knots=16):
        super().__init__(h,p,threshold,knots,kinematic_mode='observed')
        self.complete_head=nn.Linear(h,knots*42);self.flag_head=nn.Linear(h,knots*9)
        for head in [self.complete_head,self.flag_head]:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.register_buffer('full_scale',torch.ones(42))
    def edge_outputs(self,edge,b):
        return {'values':self.complete_head(edge).reshape(len(edge),self.knots,42)*self.full_scale,
                'flags':self.flag_head(edge).reshape(len(edge),self.knots,9)}

def averaged_basis(lo,hi,knots):
    """Exact average of linear hat functions over each native phase interval."""
    grid=np.linspace(0.,1.,knots);result=[]
    for a,b in zip(lo,hi):
        if b<=a:raise ValueError('invalid native query interval')
        points=np.r_[a,grid[(grid>a)&(grid<b)],b]
        values=hat_basis(points,grid)
        result.append(np.sum((values[:-1]+values[1:])*.5*np.diff(points)[:,None],axis=0)/(b-a))
    return np.asarray(result,dtype=np.float32)

def records(root,boundary_root,knots):
    items,sources=force_records(root,knots);boundary_sources=attach(items,boundary_root,sources)
    histories={}
    for r in items:
        event=r['event']
        if event not in histories:histories[event]=PhaseHistory(Path(root)/'dataset'/event)
        data=histories[event]
        complete_record(r,data,knots)
    return items,sources,boundary_sources

def complete_record(r,data,knots):
    ids=r['batch']['bond_ids'].numpy();lookup=np.full(len(data.meta['bond_keys']),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids))
    indices=data.frame_phases[r['frame']];snapshot=data.snapshot(r['frame']);batch=r['batch'];nt=len(indices);ne=len(ids)
    initial=np.array([[np.inf if x is None else x for x in row] for row in snapshot['joint_values']])[ids]
    tags=np.asarray(snapshot['joint_flags'])[ids]
    start_flags=np.column_stack([initial[:,0]>=1,tags[:,0],np.isposinf(initial[:,4:6])]).astype(np.float32)
    batch['trajectory_base']=torch.cat([batch['bond_exact_history'][:,:21],torch.tensor(np.asarray(snapshot['response_values'])[ids],dtype=torch.float32),torch.zeros(ne,3)],-1)
    batch['trajectory_flags']=torch.from_numpy(start_flags);batch['trajectory_mode']=torch.from_numpy(tags[:,1].astype(np.int64))
    target=np.zeros((nt,ne,42),np.float32);flags=np.zeros((nt,ne,4),np.float32);modes=np.zeros((nt,ne),np.int64)
    current_flags=start_flags.copy();current_mode=tags[:,1].copy()
    for t,index in enumerate(indices):
        sl=data.phase_rows(index);edge=lookup[data.slots[sl]];y=np.asarray(data.targets[sl]);f=np.asarray(data.target_flags[sl]);dt=data.phases[index]['dt']
        encoded=finite_features(y[:,:21],np.column_stack([f[:,:2],np.ones(len(f),dtype=np.int32),f[:,2]]))[:,:21]
        target[t,edge]=np.concatenate([encoded,y[:,21:39],y[:,39:42]/dt],-1)
        current_flags[edge]=np.column_stack([y[:,0]>=1,f[:,0],np.isposinf(y[:,4:6])]);current_mode[edge]=f[:,1]
        flags[t]=current_flags;modes[t]=current_mode
    if not np.isfinite(target).all():raise ValueError('nonfinite full response targets')
    lo=np.array([(data.phases[i]['time']-snapshot['world']['time'])/data.graph['dt'] for i in indices]);hi=lo+np.array([data.phases[i]['dt']/data.graph['dt'] for i in indices])
    # Native clocks can accumulate a few ULPs at the frame endpoint.
    if hi.max()>1+1e-8:raise ValueError('phase extends beyond frame')
    hi=np.minimum(hi,1.);lo=np.maximum(lo,0.)
    r.update(target=torch.from_numpy(target),flags=torch.from_numpy(flags),modes=torch.from_numpy(modes),
        end_basis=torch.from_numpy(hat_basis(hi,np.linspace(0,1,knots)).astype(np.float32)),
        work_basis=torch.from_numpy(averaged_basis(lo,hi,knots)))

def query(pred,r):
    values=pred['values'];base=r['batch']['trajectory_base']
    history=torch.einsum('tk,ekc->tec',r['end_basis'],values[:,:,:21])
    response=torch.einsum('tk,ekc->tec',r['basis'],values[:,:,21:39])
    work=torch.einsum('tk,ekc->tec',r['work_basis'],values[:,:,39:])
    output=base[None]+torch.cat([history,response,work],-1)
    logits=torch.einsum('tk,ekc->tec',r['end_basis'],pred['flags'])
    flag_base=(r['batch']['trajectory_flags']*2-1)*4
    mode_base=F.one_hot(r['batch']['trajectory_mode'],5)*4
    return output,logits[:,:,:4]+flag_base[None],logits[:,:,4:]+mode_base[None]

def masks(r):
    weight=(r['weights'][:,None]*r['mask'])[:,:,None].expand_as(r['target']).clone()
    weight[:,:,4:6]*=1-r['flags'][:,:,2:4]
    return weight

def loss(pred,r,scale,force_weight=0.):
    if not np.isfinite(force_weight) or force_weight<0:raise ValueError('invalid force objective weight')
    values,flags,modes=query(pred,r);weight=masks(r)
    fields=((((values-r['target'])/scale).square()*weight).sum((0,1))/weight.sum((0,1)).clamp_min(1e-20))
    reg=fields.mean()
    temporal=r['weights'][:,None]
    binary=F.binary_cross_entropy_with_logits(flags,r['flags'],pos_weight=flags.new_tensor([3,3,1,1]),reduction='none')
    initial=r['batch']['trajectory_flags']
    eligible=torch.cat([1-initial[:,:2],initial[:,2:4]],-1)
    binary_weight=temporal[:,:,None]*eligible[None]
    binary=((binary*binary_weight).sum((0,1))/binary_weight.sum((0,1)).clamp_min(1e-20)).mean()
    categorical=F.cross_entropy(modes.reshape(-1,5),r['modes'].reshape(-1),reduction='none').reshape_as(r['modes'])
    objective=reg+.25*binary+.1*(categorical*temporal).sum()/(temporal.sum()*flags.shape[1])
    # Preserve every complete-response target. This additional term emphasizes
    # total generalized force/torque, which drives native motion. It is NOT a
    # rollout, projectile, acceleration, or conservation loss.
    return objective if force_weight==0 else objective+force_weight*fields[21:27].mean()

@torch.inference_mode()
def evaluate(model,items):
    model.eval();total=0.;force=0.;weight_sum=0.;damage_tp=damage_fp=damage_fn=0;event_tp=event_fp=event_fn=0
    by_frame=[]
    for r in items:
        pred=model(r['batch']);total+=float(loss(pred,r,model.full_scale));values,flags,_=query(pred,r)
        weight=r['weights'][:,None]*r['mask'];force+=float(((values[:,:,21:24]-r['target'][:,:,21:24]).double().square()*weight[:,:,None]).sum());weight_sum+=float(weight.sum())
        frame_error=float(((values[:,:,21:24]-r['target'][:,:,21:24]).double().square()*weight[:,:,None]).sum())
        by_frame.append({'event':r.get('event'),'frame':r.get('frame'),'raw_force_rmse_n':(frame_error/(3*float(weight.sum())))**.5})
        predicted=flags[:,:,0]>=0;actual=r['flags'][:,:,0]>0.5
        eligible=r['batch']['trajectory_flags'][:,0]<.5
        ep=predicted.any(0)&eligible;ea=actual.any(0)&eligible
        event_tp+=int((ep&ea).sum());event_fp+=int((ep&~ea).sum());event_fn+=int((~ep&ea).sum())
        predicted=predicted&eligible[None];actual=actual&eligible[None]
        damage_tp+=int((predicted&actual).sum());damage_fp+=int((predicted&~actual).sum());damage_fn+=int((~predicted&actual).sum())
    return {'frames':len(items),'objective':total/len(items),'raw_force_rmse_n':(force/(3*weight_sum))**.5,
        'by_frame':by_frame,
        'damage_complete_phase_samples':{'tp':damage_tp,'fp':damage_fp,'fn':damage_fn},
        'new_material_failure_forecasts':{'tp':event_tp,'fp':event_fp,'fn':event_fn},
        'scope':'Oracle-state queries; only initially incomplete material damage is eligible. Phase flags carry retirement labels. Material failure is not necessarily reinforced bond disconnection. Not independent fracture or motion accuracy.'}

def train(a):
    import shutil
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);out=Path(a.out);out.mkdir();start=time.monotonic()
    (out/'sources').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'sources'/p.name)
    (out/'invocation.json').write_text(json.dumps(vars(a),indent=2)+'\n')
    (out/'source-hashes.json').write_text(json.dumps({p.name:PhaseHistory._hash(p) for p in (out/'sources').iterdir()},indent=2)+'\n')
    items,sources,boundary_sources=records(a.dataset,a.boundary_root,a.knots)
    groups={part:[r for r in items if r['partition']==part] for part in ['train','validation','test']}
    base,meta=load_checkpoint(Path(a.weights));model=CompleteTrajectoryNet(h=meta['h'],p=meta['p'],knots=a.knots)
    missing,unexpected=model.load_state_dict(base.state_dict(),strict=False)
    expected={k for k in model.state_dict() if k.startswith(('trajectory_','boundary_','complete_','flag_','full_scale'))}
    if set(missing)!=expected or unexpected:raise ValueError('pretrained encoder mismatch')
    sums=torch.zeros(42,dtype=torch.float64);counts=torch.zeros(42,dtype=torch.float64)
    for r in groups['train']:
        weight=masks(r).double();delta=r['target'].double()-r['batch']['trajectory_base'][None].double()
        sums+=(delta.square()*weight).sum((0,1));counts+=weight.sum((0,1))
    scale=(sums/counts.clamp_min(1e-20)).sqrt();scale=torch.where(scale>0,scale,torch.ones_like(scale));model.full_scale.copy_(scale.float())
    for sl in [slice(21,24),slice(24,27),slice(27,30),slice(30,33)]:model.full_scale[sl]=float((sums[sl].sum()/counts[sl].sum()).sqrt()) or 1.
    model.trajectory_scale.copy_(model.full_scale[21:27]);model.fit_boundary_scale(groups['train'])
    corrective=[];corrective_sources=None
    if a.branch_manifest:
        from .branch_dataset import load_records
        corrective,corrective_sources=load_records(a.branch_manifest,a.dataset,a.knots)
        if len(groups['train'])!=4*len(corrective):raise ValueError('fixed-budget corrective experiment requires one branch for four original frames')
    config={**vars(a),'architecture':model.architecture,'h':meta['h'],'p':meta['p'],'base_checkpoint_sha256':meta['sha256'],'phase_sources':sources,'boundary_sources':boundary_sources,'corrective_sources':corrective_sources,
        'corrective_sampling':'If enabled: replace one quarter of original frame updates with each corrective branch once per epoch; offset rotates. Original-only normalization and unchanged total updates. Validation/test remain original.',
        'scope':__doc__,'selection':'minimum validation complete response objective; no test selection','promoted':False}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    baseline={p:evaluate(model,v) for p,v in groups.items()};optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5);rng=np.random.default_rng(a.seed);best=float('inf');history=[]
    branch_rng=np.random.default_rng(a.seed+7341)
    for epoch in range(1,a.epochs+1):
        model.train();total=0.
        schedule=[groups['train'][i] for i in rng.permutation(len(groups['train']))]
        if corrective:
            for position,index in zip(range((epoch-1)%4,len(schedule),4),branch_rng.permutation(len(corrective))):schedule[position]=corrective[index]
        for r in schedule:
            optimizer.zero_grad();value=loss(model(r['batch']),r,model.full_scale,a.force_weight)
            if not torch.isfinite(value):raise ValueError('nonfinite complete response loss')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(value.detach())
        validation=evaluate(model,groups['validation']);row={'epoch':epoch,'train_loss':total/len(groups['train']),'validation':validation};history.append(row)
        if validation['objective']<best:best=validation['objective'];selected=epoch;torch.save(model.state_dict(),out/'state_dict.pt')
        if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    model.load_state_dict(torch.load(out/'state_dict.pt',weights_only=True,map_location='cpu'))
    result={'status':'completed','selected_epoch':selected,'seconds':time.monotonic()-start,'baseline':baseline,'metrics':{p:evaluate(model,v) for p,v in groups.items()},'promoted':False}
    if corrective:result['corrective_train']=evaluate(model,corrective)
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['dataset','boundary-root','weights','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--threads',type=int,default=1);p.add_argument('--seed',type=int,default=0);p.add_argument('--knots',type=int,default=16);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--force-weight',type=float,default=0.,help='Additional normalized total-force/torque MSE. Selection remains the common unweighted validation objective.')
    p.add_argument('--branch-manifest',help='Explicit verified corrective training corpus; fixed 25 percent replacement, original normalization and validation.')
    a=p.parse_args()
    if a.epochs<1 or a.threads<1 or a.knots<2 or not np.isfinite(a.lr) or a.lr<=0 or not np.isfinite(a.force_weight) or a.force_weight<0:p.error('invalid training settings')
    train(a)


def load_model(path):
    path=Path(path);c=json.loads((path/'config.json').read_text())
    if c['architecture']!=CompleteTrajectoryNet.architecture:raise ValueError('complete trajectory checkpoint required')
    model=CompleteTrajectoryNet(h=c['h'],p=c['p'],knots=c['knots'])
    model.load_state_dict(torch.load(path/'state_dict.pt',map_location='cpu',weights_only=True));model.eval()
    return model,{'sha256':PhaseHistory._hash(path/'state_dict.pt'),'architecture':model.architecture,'run':c,'h':c['h'],'p':c['p']}


def current_batch(graph,snapshot,kinematics):
    batch=graph.batch(snapshot,include_projectiles=True);ids=batch['bond_ids'].numpy()
    if kinematics['format']!='native-boundary-kinematics/1' or kinematics['time']!=snapshot['world']['time']:
        raise ValueError('runtime boundary clock/format mismatch')
    k=np.asarray(kinematics['values'],dtype=np.float64)
    if k.shape!=(graph.e,12) or not np.isfinite(k).all():raise ValueError('invalid runtime boundary observations')
    batch['bond_boundary_kinematics']=torch.tensor(k[ids],dtype=torch.float32)
    old=np.asarray([[np.inf if v is None else v for v in row] for row in snapshot['joint_values']])[ids]
    flags=np.asarray(snapshot['joint_flags'])[ids]
    batch['trajectory_base']=torch.cat([batch['bond_exact_history'][:,:21],torch.tensor(np.asarray(snapshot['response_values'])[ids],dtype=torch.float32),torch.zeros(len(ids),3)],-1)
    batch['trajectory_flags']=torch.tensor(np.column_stack([old[:,0]>=1,flags[:,0],np.isposinf(old[:,4:6])]),dtype=torch.float32)
    batch['trajectory_mode']=torch.tensor(flags[:,1],dtype=torch.long)
    return batch


def crossing(values):
    indices=np.flatnonzero(values>=0)
    if not len(indices):return None
    i=int(indices[0])
    if i==0:return 0.
    return float((i-1-values[i-1]/(values[i]-values[i-1]))/(len(values)-1))


@torch.inference_mode()
def forecast(model,graph,snapshot,kinematics):
    batch=current_batch(graph,snapshot,kinematics);pred=model(batch)
    return decode_forecast(graph,snapshot,batch,pred)


def decode_forecast(graph,snapshot,batch,pred):
    """Shared physical wire decoder, independent of stateless/recurrent processing."""
    if any(not torch.isfinite(x).all() for x in pred.values()):raise ValueError('nonfinite complete prediction')
    values=(pred['values']+batch['trajectory_base'][:,None,:]).double().numpy()
    flag_base=torch.cat([(batch['trajectory_flags']*2-1)*4,F.one_hot(batch['trajectory_mode'],5)*4],-1)
    logits=(pred['flags']+flag_base[:,None,:]).double().numpy()
    bonds=[]
    for edge,slot in enumerate(batch['bond_ids'].tolist()):
        original=np.array([np.inf if v is None else v for v in snapshot['joint_values'][slot]])
        # Interpolate in encoded space in Rust, then decode. Decoding knots
        # before interpolation would disagree with the training query.
        history=values[edge,:,:21]
        response=values[edge,:,21:39].copy();response[:,[12,14,15,17]]=np.maximum(response[:,[12,14,15,17]],0)
        # Preserve exactly cached f64 response when this head predicts no change.
        unchanged_response=values[edge,:,21:39]==batch['trajectory_base'][edge,21:39].numpy()[None,:]
        response=np.where(unchanged_response,np.asarray(snapshot['response_values'][slot])[None,:],response)
        rates=values[edge,:,39:].copy();rates[:,[0,2]]=np.maximum(rates[:,[0,2]],0)
        rows=history.tolist()
        bonds.append({'slot':slot,'history':rows,'response':response.tolist(),'work_rate':rates.tolist(),
            'mode':logits[edge,:,4:].argmax(-1).tolist(),
            'history_origin_encoded':batch['trajectory_base'][edge,:21].double().tolist(),
            'ductility_set_time':[0. if np.isfinite(original[j+4]) else crossing(-logits[edge,:,j+2]) for j in range(2)],
            'damage_complete_time':0. if original[0]>=1 else crossing(logits[edge,:,0]),
            'rebar_break_time':0. if snapshot['joint_flags'][slot][0] else crossing(logits[edge,:,1])})
    result={'format':'learned-native-trajectory/2','history_encoding':'signed_log','start_time':snapshot['world']['time'],'dt':graph.dt,'bonds':bonds}
    json.dumps(result,allow_nan=False)
    return result

if __name__=='__main__':main()
