"""Compare pre-step Rust snapshots with the recorder and a trained checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .runtime_snapshot import NativeGraph
from .schema import load_event
from .dataset import event_batches
from .visualize import load_checkpoint


def audit(root,weights):
    torch.set_num_threads(1)
    graph=NativeGraph(json.loads((root/'graph.json').read_text()))
    model,_=load_checkpoint(weights)
    event=load_event(root/'recording',world_inputs=True,forcing_inputs=True,
                     exact_joint_history=True,material_inputs=True,bond_response_history=True,
                     projectile_inputs=getattr(model,'requires_projectile_inputs',False))
    frames=outputs=inputs=cracked=retired=0;sources={}
    for i,expected in enumerate(event_batches(event,bond_policy='runtime_alive')):
        path=root/'snapshots'/f'{i:06}.json';blob=path.read_bytes()
        sources[path.name]=hashlib.sha256(blob).hexdigest();snapshot=json.loads(blob)
        actual,batch=graph.predict(model,snapshot)
        for key,value in batch.items():
            torch.testing.assert_close(value,expected[key],rtol=0,atol=0,msg=f'frame {i}: input {key}')
            inputs+=1
        with torch.inference_mode():reference=model(model.prepare_batch(expected))
        for key,value in actual.items():
            torch.testing.assert_close(value,reference[key],rtol=0,atol=0,msg=f'frame {i}: output {key}')
            outputs+=1
        # JSON uses null ONLY for explicitly unset positive-infinite ductility.
        values=np.asarray([[np.inf if x is None else x for x in row] for row in snapshot['joint_values']],dtype=np.float64).reshape(-1,21)
        np.testing.assert_array_equal(values.view('u8'),event.exact_joint_history.values[i].view('u8'))
        np.testing.assert_array_equal(np.asarray(snapshot['response_values'],dtype=np.float64).reshape(-1,18).view('u8'),event.bond_response_history.values[i].view('u8'))
        flags=np.asarray(snapshot['joint_flags']).reshape(-1,4)
        cracked+=int(((flags[:,2]==1)&(flags[:,3]==0)).sum());retired+=int((flags[:,2]==0).sum())
        assert not any(k.endswith('_end') for k in batch)
        assert not {'bond_fail_class','bond_peak_util','chunk_state_delta','chunk_load_bins','chunk_contact','bond_qs_force'}&batch.keys()
        frames+=1
    if not frames:raise ValueError('no parity frames')
    return {'frames':frames,'exact_input_tensor_comparisons':inputs,'exact_output_tensor_comparisons':outputs,
            'cracked_alive_bond_intervals':cracked,'retired_bond_intervals':retired,
            'exact_joint_response_f64_parity':True,'future_labels_required':False,
            'checkpoint_sha256':hashlib.sha256((weights/'state_dict.pt').read_bytes()).hexdigest(),
            'snapshot_sha256':sources,
            'scope':'Read-only snapshots captured before each oracle interval. Bitwise feature/output parity with recording; not an independent learned rollout or runtime speed benchmark.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','weights','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=audit(a.root,a.weights);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
