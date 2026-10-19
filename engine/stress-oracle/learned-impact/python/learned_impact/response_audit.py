"""Score cached responses against exact f64 recorder targets, in physical units."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .dataset import list_events,event_batches
from .schema import load_event
from .visualize import load_checkpoint
from .experiment import event_digest
from .response_history import FIELDS,SCALE


@torch.inference_mode()
def audit(weights,dataset,partition='test'):
    torch.set_num_threads(1);model,meta=load_checkpoint(weights)
    if getattr(model,'bond_policy',None)!='runtime_alive':raise ValueError('runtime-alive response audit required')
    complete=getattr(model,'requires_bond_response_history',False);width=18 if complete else 6
    split=json.loads((weights/'split.json').read_text())['events'];sources={}
    squared=np.zeros((2,3,width));counts=np.zeros(2,dtype=np.int64);frames=0
    for path in list_events(dataset):
        if split.get(path.name)!=partition:continue
        ev=load_event(path,world_inputs=True,world_targets=True,forcing_inputs=True,exact_joint_history=True,
                      material_inputs=True,projectile_inputs=getattr(model,'requires_projectile_inputs',False),bond_response_history=True)
        sources[path.name]=event_digest(path);response=ev.bond_response_history
        for i,raw in enumerate(event_batches(ev,bond_policy='runtime_alive')):
            ids=raw['bond_ids'].numpy();pred=model(model.prepare_batch(raw))
            if complete:actual=model.decode_response(pred)
            else:
                encoded=pred['bond_force_end'].numpy().astype(np.float64)
                with np.errstate(over='ignore',invalid='ignore'):actual=np.sign(encoded)*np.expm1(np.abs(encoded))*SCALE[:6]
            start=response.values[i,ids,:width];end=response.values[i+1,ids,:width]
            alive=response.flags[i+1,ids,0]==1
            for method,estimate in enumerate((actual,start,np.zeros_like(start))):
                error=np.square(estimate-end)
                if not np.isfinite(error).all():raise ValueError(f'{path.name}:{i}: nonfinite exact response error')
                for cohort,mask in enumerate((alive,~alive)):squared[cohort,method]+=error[mask].sum(0)
            counts+=np.array([alive.sum(),(~alive).sum()]);frames+=1
    if not frames:raise ValueError('no response audit records')
    result={}
    groups={'total_force_N':(0,3),'total_moment_Nm':(3,6),'elastic_force_N':(6,9),'elastic_moment_Nm':(9,12),'stored_energy_J':(12,13)}
    for cohort,name in enumerate(('active_end','retired_cache')):
        n=counts[cohort];result[name]={'bond_intervals':int(n)}
        for method,label in enumerate(('model','persistence','zero')):
            sums=squared[cohort,method]
            result[name][label]={'field_rmse':{field:float(np.sqrt(value/n)) if n else None for field,value in zip(FIELDS,sums)},
                'component_rmse':{key:float(np.sqrt(sums[a:b].sum()/(n*(b-a)))) if n else None for key,(a,b) in groups.items() if b<=width}}
    return {'checkpoint':meta,'checkpoint_sha256':hashlib.sha256((weights/'state_dict.pt').read_bytes()).hexdigest(),
            'source_sha256':sources,'partition':partition,'frames':frames,'predicted_fields':FIELDS[:width],'summary':result,
            'scope':'Exact f64 recorder endpoint targets and persistence baseline. Model inputs retain float32 encoding. Active-end responses and retired last caches are separate; a retired cache is not a continuing force. Oracle reset every interval; no coupled dynamics or constitutive consistency claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','dataset','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--partition',choices=['train','validation','test'],default='test');a=parser.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=audit(a.weights,a.dataset,a.partition)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['summary']))


if __name__=='__main__':main()
