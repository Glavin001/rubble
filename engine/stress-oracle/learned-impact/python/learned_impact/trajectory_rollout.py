"""Independent complete learned stress forecasts through native phase integration.
No oracle dataset input or endpoint pose injection. Experimental accuracy.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .runtime_snapshot import NativeGraph
from .trajectory_response import load_model as load_checkpoint,forecast

FORMAT='learned-native-trajectory/2'

def run(args):
    args.out.mkdir(exist_ok=False);torch.set_num_threads(args.threads)
    (args.out/'inference-inputs').mkdir()
    config=json.loads((args.checkpoint/'config.json').read_text())
    recurrent=config['architecture']=='physics-recurrent-multiscale-trajectory-v36'
    if recurrent:
        from .recurrent_trajectory import load_model,forecast as recurrent_forecast
        from .multiscale import HierarchyCache
        model,meta=load_model(args.checkpoint)
        hierarchy_cache=HierarchyCache()
    else:model,meta=load_checkpoint(args.checkpoint)
    memory=None
    checkpoint_hash=meta['sha256'];start=time.monotonic();inference=[]
    plan={'status':'running','integration':'native-stress-phase-trajectory/1','checkpoint':meta,'scene':str(args.scene.resolve()),'scene_sha256':hashlib.sha256(args.scene.read_bytes()).hexdigest(),
          'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),'frames':args.frames,
          'scope':'Independent complete-trajectory experiment. Native stress integration, contacts and fracture phases retained, zero oracle resets or stress fallback. Forecasts are fixed from frame start and do not react to new within-frame contacts. Constrained learned histories/forces/work; no production accuracy, conservation or speed claim.'}
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
                # Preserve the causal kinematic input alongside the native start
                # snapshot, allowing exact prediction/memory replay after a run.
                (args.out/'inference-inputs'/f'{i:06}.json').write_text(json.dumps(msg['kinematics'],allow_nan=False)+'\n')
                t=time.monotonic()
                if recurrent:response,memory=recurrent_forecast(model,graph,msg['snapshot'],msg['kinematics'],memory,hierarchy_cache=hierarchy_cache)
                else:response=forecast(model,graph,msg['snapshot'],msg['kinematics'])
                inference.append(time.monotonic()-t)
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
            if recurrent:plan['hierarchy_builds']=hierarchy_cache.builds
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
