"""Learn the wave-carrying graph velocity transition from current physical state.

The neural graph predicts internal velocity increments. Known current external
and mass-damping loads provide a ballistic prior; the engine applies kinematics,
support constraints, rigid-frame transfers and splits. This model replaces the
internal numerical kick, unlike the bond-local material model. Native cadence is
retained for this first accuracy experiment; no efficiency claim is made.
"""
from __future__ import annotations
import argparse,json,struct,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

class GraphTransitions:
    def __init__(self,root):
        self.root=Path(root);folder=self.root/'graph-transitions'
        self.meta=json.loads((folder/'manifest.json').read_text())
        report=json.loads((self.root/'report.json').read_text());m=self.meta
        if m['format']!='native-graph-transition/1' or m['status']!='completed' or report['status']!='completed':
            raise ValueError('complete native graph-transition recording required')
        if (m['node_width'],m['node_target_width'],m['edge_width'])!=(39,12,84):raise ValueError('graph field contract')
        def read(name,dtype,shape):
            path=folder/name
            if path.stat().st_size!=np.prod(shape)*np.dtype(dtype).itemsize:raise ValueError('wrong byte count: '+name)
            return np.memmap(path,mode='r',dtype=dtype,shape=shape) if np.prod(shape) else np.empty(shape,dtype=dtype)
        self.nodes=read('nodes.f64','<f8',(m['nodes'],39));self.targets=read('node-targets.f64','<f8',(m['nodes'],12))
        self.edges=read('edges.f64','<f8',(m['edges'],84));self.ends=read('endpoints.u32','<u4',(m['edges'],2))
        self.node_keys=read('node-keys.u32','<u4',(m['nodes'],2))
        self.phases=m['phases'];no=eo=0
        for p in self.phases:
            if p['node_start']!=no or p['edge_start']!=eo:raise ValueError('noncontiguous phase rows')
            ep=self.ends[eo:eo+p['edges']]
            if len(ep) and (ep>=p['nodes']).any():raise ValueError('edge outside its phase graph')
            no+=p['nodes'];eo+=p['edges']
        if (no,eo)!=(m['nodes'],m['edges']):raise ValueError('phase count mismatch')
        if not all(np.isfinite(a).all() for a in (self.nodes,self.targets,self.edges)):raise ValueError('nonfinite physical data')
        # Native translation kinematics must agree at this exact boundary. Fixed
        # and pinned chunks retain their previous displacement.
        expected=self.nodes[:,:3]+self.targets[:,6:9]*self.nodes[:,37,None]*self.nodes[:,34,None]
        if not np.allclose(expected,self.targets[:,:3],rtol=1e-12,atol=1e-15):raise ValueError('target is not pre-transfer native state')

    def phase(self,i):
        p=self.phases[i];ns=slice(p['node_start'],p['node_start']+p['nodes']);es=slice(p['edge_start'],p['edge_start']+p['edges'])
        return tuple(torch.from_numpy(np.array(a,dtype=d,copy=True)) for a,d in
                     [(self.nodes[ns],'f8'),(self.edges[es],'f8'),(self.ends[es],'i8'),(self.targets[ns],'f8')])

def free_mask(x):
    return torch.cat((x[:,34:35].expand(-1,3),(x[:,34:35]+x[:,35:36]).expand(-1,3)),1)

def external_delta(x):
    linear=(x[:,12:15]+x[:,18:21])*x[:,24:25]
    angular=torch.einsum('nij,nj->ni',x[:,25:34].reshape(-1,3,3),x[:,15:18]+x[:,21:24])
    return torch.cat((linear,angular),1)*x[:,37:38]*free_mask(x)

def pack(phases):
    nodes=[];edges=[];ends=[];targets=[];offset=0
    for n,e,ij,y in phases:
        nodes.append(n);edges.append(e);ends.append(ij+offset);targets.append(y);offset+=len(n)
    return torch.cat(nodes),torch.cat(edges),torch.cat(ends),torch.cat(targets)

def mlp(n,h,out):return nn.Sequential(nn.Linear(n,h),nn.SiLU(),nn.Linear(h,out))

class Processor(nn.Module):
    def __init__(self,h):
        super().__init__();self.edge=mlp(3*h,h,h);self.node=mlp(3*h,h,h)
        self.en=nn.LayerNorm(h);self.nn=nn.LayerNorm(h)
    def forward(self,n,e,ij):
        a,b=ij.T
        e=self.en(e+self.edge(torch.cat((n[a],n[b],e),1)))
        left=torch.zeros_like(n);right=torch.zeros_like(n)
        left.index_add_(0,a,e);right.index_add_(0,b,e)
        return self.nn(n+self.node(torch.cat((n,left,right),1))),e

class GraphStateNet(nn.Module):
    def __init__(self,width=128,passes=4):
        super().__init__();self.width=width;self.passes=passes
        self.node=mlp(39,width,width);self.edge=mlp(84,width,width)
        self.process=nn.ModuleList(Processor(width) for _ in range(passes));self.head=mlp(width,width,6)
        for prefix,n in [('n',39),('e',84)]:
            self.register_buffer(prefix+'scale',torch.ones(n,dtype=torch.float64))
            self.register_buffer(prefix+'mean',torch.zeros(n,dtype=torch.float64))
            self.register_buffer(prefix+'std',torch.ones(n,dtype=torch.float64))
        self.register_buffer('yscale',torch.ones(6,dtype=torch.float64))
        self.register_buffer('active',torch.ones(6,dtype=torch.bool))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,e,ij):
        n=self.node(((torch.asinh(x/self.nscale)-self.nmean)/self.nstd).float())
        edge=self.edge(((torch.asinh(e/self.escale)-self.emean)/self.estd).float())
        for layer in self.process:n,edge=layer(n,edge,ij)
        degree=torch.zeros(len(n),dtype=torch.long,device=n.device)
        degree.index_add_(0,ij.flatten(),torch.ones(ij.numel(),dtype=torch.long,device=n.device))
        # An isolated chunk has no internal bond impulse. Known external motion
        # remains in external_delta; no physical edge is discarded.
        return self.head(n)*free_mask(x).float()*self.active*(degree>0)[:,None]

    def velocity_delta(self,x,e,ij):
        return external_delta(x)+self(x,e,ij).double()*self.yscale

    @torch.no_grad()
    def fit(self,data):
        for prefix,field,encoder in [('n','nodes',self.node),('e','edges',self.edge)]:
            a=torch.from_numpy(np.concatenate([getattr(d,field) for d in data]))
            scale=a.square().mean(0).sqrt().clamp_min(1e-12);z=torch.asinh(a/scale)
            getattr(self,prefix+'scale').copy_(scale);getattr(self,prefix+'mean').copy_(z.mean(0))
            getattr(self,prefix+'std').copy_(z.std(0,unbiased=False).clamp_min(.05))
            encoder[0].weight[:,(z.max(0).values-z.min(0).values)<1e-12]=0
        residual=[]
        for d in data:
            x=torch.from_numpy(np.array(d.nodes));y=torch.from_numpy(np.array(d.targets))
            residual.append((y[:,6:12]-x[:,6:12]-external_delta(x))*free_mask(x))
        r=torch.cat(residual);self.yscale.copy_(r.square().mean(0).sqrt().clamp_min(1e-12));self.active.copy_(r.abs().max(0).values>1e-14)

@torch.no_grad()
def evaluate(net,dataset):
    errors=[];baselines=[]
    for i in np.linspace(0,len(dataset.phases)-1,min(128,len(dataset.phases)),dtype=int):
        x,e,ij,y=dataset.phase(i)
        if not len(x):continue
        t=(y[:,6:12]-x[:,6:12]-external_delta(x))/net.yscale
        mask=free_mask(x)
        errors.append(float(((net(x,e,ij).double()-t)*mask).square().sum()/mask.sum().clamp_min(1)))
        baselines.append(float((t*mask).square().sum()/mask.sum().clamp_min(1)))
    return {'normalized_internal_increment_mse':sum(errors)/max(len(errors),1),
            'zero_internal_impulse_baseline_mse':sum(baselines)/max(len(baselines),1),
            'scope':'Current oracle phase inputs; local prediction only, not an independent rollout.'}

def train(args):
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    train=[GraphTransitions(p) for p in args.train];validation=[GraphTransitions(p) for p in args.validation]
    net=GraphStateNet(args.width,args.passes);net.fit(train)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-6)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=1e-5)
    available=[(d,i) for d in train for i,p in enumerate(d.phases) if p['nodes']]
    config={'width':args.width,'passes':args.passes};log=[];best=float('inf');started=time.perf_counter()
    for step in range(1,args.steps+1):
        x,e,ij,y=pack([available[k][0].phase(available[k][1]) for k in rng.integers(len(available),size=args.batch)])
        target=(y[:,6:12]-x[:,6:12]-external_delta(x))/net.yscale
        error=(net(x,e,ij).double()-target)*free_mask(x)
        loss=error.square().sum()/free_mask(x).sum().clamp_min(1)
        if not torch.isfinite(loss):raise ValueError('nonfinite graph training loss')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();schedule.step()
        if step%500==0 or step==args.steps:
            metrics=[evaluate(net,d) for d in validation]
            score=np.mean([m['normalized_internal_increment_mse'] for m in metrics]) if metrics else float(loss.detach())
            row={'step':step,'loss':float(loss.detach()),'validation':metrics,'seconds':time.perf_counter()-started}
            log.append(row);print(json.dumps(row),flush=True)
            if score<best:
                best=float(score);torch.save({'format':'learned-graph-state/1','configuration':config,'state_dict':net.state_dict(),'step':step},out/'model.pt')
    report={'status':'trained','configuration':vars(args),'parameters':sum(p.numel() for p in net.parameters()),'log':log,
            'persistent_state':'native physical chunk deformation/velocity and bond history; no temporal latent memory',
            'scope':__doc__,'independent_rollout_proven':False,'numerical_internal_integrator_required_at_inference':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def exact_read(stream,n):
    result=bytearray()
    while len(result)<n:
        block=stream.read(n-len(result))
        if not block:raise EOFError('truncated graph inference request')
        result.extend(block)
    return result

def serve(args):
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['format']!='learned-graph-state/1':raise ValueError('wrong graph checkpoint')
    net=GraphStateNet(**checkpoint['configuration']);net.load_state_dict(checkpoint['state_dict']);net.eval()
    src,dst=sys.stdin.buffer,sys.stdout.buffer
    with torch.inference_mode():
        while True:
            header=src.read(8)
            if not header:return
            if len(header)!=8:raise EOFError('truncated graph header')
            n,e=struct.unpack('<II',header)
            x=torch.from_numpy(np.frombuffer(exact_read(src,n*39*8),dtype='<f8').copy().reshape(n,39))
            features=torch.from_numpy(np.frombuffer(exact_read(src,e*84*8),dtype='<f8').copy().reshape(e,84))
            ij=torch.from_numpy(np.frombuffer(exact_read(src,e*2*4),dtype='<u4').astype('i8').reshape(e,2))
            delta=net.velocity_delta(x,features,ij).numpy() if n else np.empty((0,6),dtype='f8')
            if not np.isfinite(delta).all():raise ValueError('nonfinite graph prediction')
            dst.write(delta.astype('<f8').tobytes());dst.flush()

def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--train',nargs='+',required=True);t.add_argument('--validation',nargs='+',required=True)
    t.add_argument('--out',required=True);t.add_argument('--steps',type=int,default=10000);t.add_argument('--batch',type=int,default=4)
    t.add_argument('--width',type=int,default=128);t.add_argument('--passes',type=int,default=4);t.add_argument('--seed',type=int,default=0)
    s=sub.add_parser('serve');s.add_argument('--checkpoint',required=True)
    a=p.parse_args()
    if a.mode=='train':
        if min(a.steps,a.batch,a.width,a.passes)<=0:p.error('positive training configuration required')
        train(a)
    else:serve(a)
if __name__=='__main__':main()
