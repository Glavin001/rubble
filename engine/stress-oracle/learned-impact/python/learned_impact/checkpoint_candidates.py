"""Retain physical validation candidates without selecting on test outcomes.

These are diagnostic alternatives, not an automatic replacement for the main
fracture-selected checkpoint or proof of constitutive/rollout consistency.
"""
import hashlib
import json
import math
from pathlib import Path
import torch


def objectives(validation):
    response=validation['cached_response']['active_end']
    if response['bond_intervals']<=0:raise ValueError('physical checkpoint selection needs active response targets')
    fields=response['model']['si_component_rmse']
    def components(names):return math.sqrt(sum(fields[k]**2 for k in names)/len(names))
    result={'active_force_N':components(('force_x','force_y','force_z')),
            'active_energy_J':fields['stored_energy'],
            'world_position_m':validation['world_motion']['rmse']['position_m']}
    if any(v is None or not math.isfinite(v) or v<0 for v in result.values()):
        raise ValueError('physical checkpoint selection requires finite nonnegative validation errors')
    return result


class PhysicalCandidates:
    def __init__(self,root):
        self.root=Path(root);self.best={}
        self.root.mkdir(exist_ok=False)

    def consider(self,model,validation,epoch,config,split):
        scores=objectives(validation)
        for criterion,score in scores.items():
            if criterion in self.best and score>=self.best[criterion]['value']:continue
            dest=self.root/criterion;dest.mkdir(exist_ok=True)
            torch.save(model.state_dict(),dest/'state_dict.pt')
            sha=hashlib.sha256((dest/'state_dict.pt').read_bytes()).hexdigest()
            info={'criterion':criterion,'value':score,'epoch':epoch,'checkpoint_sha256':sha,
                  'threshold':validation['threshold'],'validation_fracture_f1':validation['f1']}
            run={**config,'selected_epoch':epoch,'threshold':validation['threshold'],
                 'checkpoint_selection':f'lowest validation {criterion}; earliest epoch wins ties',
                 'candidate_only':True,'physical_candidate':info}
            for name,data in [('run.json',run),('split.json',split),('metrics.json',{'validation':validation,
                'scope':'Validation-only physical candidate; test has not been evaluated for this candidate.'})]:
                (dest/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
            self.best[criterion]=info
        (self.root/'index.json').write_text(json.dumps({'candidates':self.best,
            'scope':'Each candidate minimizes one declared validation metric; no combined winner or test-based selection. SI metrics decode float32 targets; exact-f64 audit is separate. Main fracture selection is unchanged.'},indent=2,allow_nan=False)+'\n')
