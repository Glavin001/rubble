"""Experimental independent endpoint rollout using a persistent native World.

No oracle dataset is an input. The native process advances contacts and rigid
bodies; this model supplies hidden deformation, joint state, and cached responses.
Breaks commit at interval end. This is not production substep constitutive parity.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .runtime_snapshot import NativeGraph
from .visualize import load_checkpoint

FORMAT='learned-external-endpoint/1'

def endpoint(model,graph,snapshot):
    pred,batch=graph.predict(model,snapshot)
    if any(not torch.isfinite(x).all() for x in pred.values()):raise ValueError('nonfinite model prediction')
    delta=model.decode_delta(pred).detach().cpu().numpy().astype(np.float64)
    if delta.shape!=(graph.n,12) or not np.isfinite(delta).all():raise ValueError('invalid decoded local increment')
    ids=batch['bond_ids'].numpy();state=model.decode_joint_state(pred,batch);values=state['values']
    original=np.array([[np.inf if v is None else v for v in row] for row in snapshot['joint_values']],dtype=np.float64)[ids]
    start=batch['bond_exact_history'].numpy()[:,:21];encoded=pred['exact_history_encoded'].numpy()
    # Preserve exact f64 history when the encoded prediction is literally unchanged.
    # Inverting a float32 log encoding must not slowly erode accumulated history.
    unchanged=(encoded==start)&np.isfinite(original)&np.isfinite(values)
    unchanged[:,0]&=~pred['damage_complete'].numpy()
    values=np.where(unchanged,original,values)
    connected=(values[:,0]<1)|(batch['bond_has_rebar'].numpy()&~state['rebar_broken'])
    expected=batch['bond_failure_eligible'].numpy()&~(model.predict_classes(pred).numpy()>0)
    if not np.array_equal(connected,expected):raise ValueError('decoded fracture/state mismatch')
    rows=[]
    for row in values:
        rows.append([None if j in (4,5) and v==np.inf else float(v) for j,v in enumerate(row)])
    response=model.decode_response(pred)
    result=dict(format=FORMAT,start_time=float(snapshot['world']['time']),dt=graph.dt,
                chunk_delta=delta.tolist(),bond_ids=ids.tolist(),joint_values=rows,
                rebar_broken=state['rebar_broken'].tolist(),mode=state['mode'].tolist(),
                connected=connected.tolist(),response=response.tolist())
    json.dumps(result,allow_nan=False) # reject before sending any command
    return result

def run(args):
    args.out.mkdir(exist_ok=False);torch.set_num_threads(args.threads)
    model,meta=load_checkpoint(args.checkpoint)
    checkpoint_hash=meta['sha256'];start=time.monotonic();inference=[]
    plan={'status':'running','checkpoint':meta,'scene':str(args.scene.resolve()),'scene_sha256':hashlib.sha256(args.scene.read_bytes()).hexdigest(),
          'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),'frames':args.frames,
          'scope':'Independent endpoint experiment. All snapshots come from this live World, no oracle resets. Contact substeps unchanged; hidden increments interpolated, all breaks at interval end. Missing damping/overshoot work; no complete conservation or production speed claim.'}
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    proc=None
    try:
        with (args.out/'native.log').open('w') as log:
            proc=subprocess.Popen([str(args.binary.resolve()),str(args.scene.resolve()),str((args.out/'native').resolve()),str(args.frames)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env={**os.environ,'RAYON_NUM_THREADS':'1'})
            first=json.loads(proc.stdout.readline());assert first['kind']=='graph';graph=NativeGraph(first['graph'])
            for i in range(args.frames):
                line=proc.stdout.readline()
                if not line:raise RuntimeError(f'native world stopped before interval {i}; see native.log')
                msg=json.loads(line);assert msg['kind']=='snapshot' and msg['index']==i
                t=time.monotonic();response=endpoint(model,graph,msg['snapshot']);inference.append(time.monotonic()-t)
                proc.stdin.write(json.dumps(response,allow_nan=False)+'\n');proc.stdin.flush()
            line=proc.stdout.readline()
            if not line:raise RuntimeError('native world stopped before final state; see native.log')
            result=json.loads(line);assert result['kind']=='completed' and result['frames']==args.frames
            graph.batch(result['snapshot'],include_projectiles=getattr(model,'requires_projectile_inputs',False))
            proc.stdin.close();rc=proc.wait();assert rc==0
            assert all(r['report']['analytic_bond_evaluations']==0 for r in result['reports'])
            assert hashlib.sha256((args.checkpoint/'state_dict.pt').read_bytes()).hexdigest()==checkpoint_hash
            (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            plan.update(status='completed',inference_seconds=inference,seconds=time.monotonic()-start)
    except Exception as e:
        plan.update(status='failed',error=repr(e),seconds=time.monotonic()-start,inference_seconds=inference)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate();proc.wait()
        (args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'status':plan['status'],'frames':args.frames,'seconds':plan['seconds']}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('binary','scene','checkpoint','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--frames',type=int,default=8);p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    if a.frames<1 or a.threads<1:p.error('frames and threads must be positive')
    run(a)
if __name__=='__main__':main()
