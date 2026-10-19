"""Train a tiny causal material-state residual MLP; exported native inference.

This is deliberately an accuracy gate, not a claim to replace native time integration.
Physical graph state is fed back. No latent memory, phase index, bond ID or future loads.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

NY, NG = 42, 29

def read(root):
    root = Path(root)
    meta = json.loads((root / "report.json").read_text())
    assert meta["status"] == "completed" and meta["oracle_queries"] == meta["rows"]
    nx = meta["input_width"]
    rows = np.memmap(root / "rows.f64", dtype="<f8", mode="r").reshape(-1, nx + 2*NY + NG)
    assert len(rows) == meta["rows"] and np.isfinite(rows).all()
    return np.asarray(rows[:, :nx]), np.asarray(rows[:, nx:nx+NY]), np.asarray(rows[:, nx+NY:nx+2*NY]), np.asarray(rows[:, nx+2*NY:])

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows-per-case", type=int, default=300000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    data = [read(root) for root in args.train]
    rng=np.random.default_rng(args.seed)
    data=[tuple(v[ix] for v in row) if len(row[0])>args.max_rows_per_case else row
          for row in data for ix in [rng.choice(len(row[0]),min(len(row[0]),args.max_rows_per_case),replace=False)]]
    x, base, y, gate = [np.concatenate([r[i] for r in data]) for i in range(4)]
    # Fit normalizers ONLY to training cases. Constants remain well-conditioned.
    xs = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-12)
    zx = np.arcsinh(x / xs); xm = zx.mean(0); xd = np.maximum(zx.std(0), .05)
    delta = y-base
    ys = np.maximum(np.sqrt(np.mean(delta*delta, axis=0)), 1e-12)
    # Zero target channels must stay exactly zero at export/runtime.
    active = np.max(np.abs(delta), axis=0) > 1e-14
    xt = torch.tensor((zx-xm)/xd, dtype=torch.float32)
    yt = torch.tensor(delta/ys, dtype=torch.float32)
    gt = torch.tensor(gate, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(x.shape[1],args.width),nn.ReLU(),nn.Linear(args.width,args.width),nn.ReLU(),nn.Linear(args.width,NY+NG))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    weights = torch.ones(NY); weights[:2]=8; weights[7:13]=3; weights[27:34]=4
    weights[[17,18,19,21,22,23,24,25,26,34,35,36,37,38,41]]=0 # exact decoder channels
    weights *= torch.tensor(active)
    gw = torch.ones(NG); gw[[0,1,21,22,23]]=4; gw[[17,18,19]]=0
    changed = torch.where((gt[:,0]>0)|(gt[:,1]>0)|(gt[:,21]>0))[0]
    nonlinear = torch.where((torch.tensor(y[:,0])>0)|(torch.tensor(y[:,1])>0))[0]
    log=[]; start=time.perf_counter()
    for step in range(args.steps):
        n=args.batch
        ix=torch.randint(len(xt),(n,))
        # Equal representation of ordinary propagation, nonlinear state, and transitions.
        if len(changed): ix[:n//3]=changed[torch.randint(len(changed),(n//3,))]
        if len(nonlinear): ix[n//3:2*(n//3)]=nonlinear[torch.randint(len(nonlinear),(n//3,))]
        pred=net(xt[ix]); continuous=F.smooth_l1_loss(pred[:,:NY],yt[ix],reduction="none")
        loss=(continuous*weights).sum(-1).mean()/weights.sum()
        classification=(F.binary_cross_entropy_with_logits(pred[:,NY:],gt[ix],reduction="none")*gw).sum(-1).mean()/gw.sum()
        loss=loss+classification
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
        if step%500==0 or step+1==args.steps:
            entry={"step":step+1,"loss":float(loss.detach()),"seconds":time.perf_counter()-start}
            log.append(entry);print(json.dumps(entry),flush=True)
    layers=[]
    for l in net:
        if isinstance(l,nn.Linear): layers.append({"weight":l.weight.detach().numpy().astype(float).tolist(),"bias":l.bias.detach().numpy().astype(float).tolist()})
    # Remove only channels whose target is identically zero in this training scope.
    for j in range(NY):
        if not active[j]: layers[-1]["weight"][j]=[0.]*args.width;layers[-1]["bias"][j]=0.
    artifact={"format":"causal-material-state/2","trained_steps":args.steps,"input_scale":xs.tolist(),"input_mean":xm.tolist(),"input_std":xd.tolist(),"output_scale":ys.tolist(),"layers":layers}
    weights_file=args.out/"weights.json";weights_file.write_text(json.dumps(artifact,allow_nan=False))
    torch.save({"state_dict":net.state_dict(),"args":vars(args)},args.out/"checkpoint.pt")
    results=[]
    with torch.no_grad():
        for role,roots in [("train",args.train),("validation",args.validation)]:
            for root in roots:
                xx,bb,yy,gg=read(root)
                if len(xx)>args.max_rows_per_case:
                    ix=np.linspace(0,len(xx)-1,args.max_rows_per_case,dtype=int);xx,bb,yy,gg=[a[ix] for a in (xx,bb,yy,gg)]
                features=torch.tensor((np.arcsinh(xx/xs)-xm)/xd,dtype=torch.float32)
                pp=torch.cat([net(v) for v in features.split(8192)]).numpy()
                physical=bb+pp[:,:NY]*ys
                truth=gg[:,21]>0;guess=pp[:,NY+21]>0
                tp=int((truth&guess).sum());fp=int((~truth&guess).sum());fn=int((truth&~guess).sum())
                results.append({"role":role,"root":str(root),"rows":len(xx),"teacher_forced_force_rmse_n":float(np.sqrt(np.mean((physical[:,27:30]-yy[:,27:30])**2))),
                    "complete_damage_tp":tp,"complete_damage_fp":fp,"complete_damage_fn":fn,
                    "complete_damage_f1":2*tp/max(2*tp+fp+fn,1)})
    report={"status":"trained","scope":"Live native-phase material-state model; full native motion/contact substeps retained.","parameters":sum(v.numel() for v in net.parameters()),
        "rows":len(x),"changed_rows":len(changed),"nonlinear_rows":len(nonlinear),"seed":args.seed,"steps":args.steps,"width":args.width,
        "sha256":hashlib.sha256(weights_file.read_bytes()).hexdigest(),"log":log,"cases":results}
    (args.out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report),flush=True)

if __name__=="__main__": main()
