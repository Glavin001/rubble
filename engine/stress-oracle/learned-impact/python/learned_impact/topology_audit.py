"""Oracle-start fragment forecasts; never an autonomous dynamics rollout."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .dataset import list_events,event_batches
from .schema import load_event
from .visualize import load_checkpoint
from .experiment import event_digest
from .topology_metrics import components,canonical_partition,partition_metrics


@torch.inference_mode()
def audit(weights,dataset,partition='test'):
    torch.set_num_threads(1);model,meta=load_checkpoint(weights)
    if getattr(model,'bond_policy',None)!='runtime_alive':raise ValueError('runtime-alive graph model required')
    split=json.loads((weights/'split.json').read_text())['events'];rows=[];sources={}
    for path in list_events(dataset):
        if split.get(path.name)!=partition:continue
        ev=load_event(path,world_inputs=True,world_targets=True,forcing_inputs=True,exact_joint_history=True,
                      material_inputs=True,projectile_inputs=getattr(model,'requires_projectile_inputs',False),bond_response_history=getattr(model,'requires_bond_response_history',False))
        replay=json.loads((path/'replay.json').read_text());sources[path.name]=event_digest(path)
        n=int(ev.manifest['n_chunks']);mass=ev.static['chunk_mass'].reshape(n)
        for i,raw in enumerate(event_batches(ev,bond_policy='runtime_alive')):
            start_active=np.asarray(replay['frames'][i]['active']);end_active=np.asarray(replay['frames'][i+1]['active'])
            if not start_active.all() or not np.array_equal(start_active,end_active):
                raise ValueError('topology audit requires fixed active recorded chunks; activity changes need a separate contract')
            ids=raw['bond_ids'].numpy();edges=raw['bond_endpoints'].numpy();eligible=raw['bond_failure_eligible'].numpy()
            start=components(n,edges,eligible)
            end_flags=ev.exact_joint_history.flags[i+1,ids]
            truth=components(n,edges,end_flags[:,3]==1)
            for label,snapshot in ((start,i),(truth,i+1)):
                if not np.array_equal(canonical_partition(label),canonical_partition(np.asarray(replay['frames'][snapshot]['clusters']))):
                    raise ValueError(f'{path.name}:{i}: oracle graph components disagree with recorded clusters')
            oracle_retired=truth[edges[:,0]]!=truth[edges[:,1]]
            if not np.array_equal(oracle_retired,end_flags[:,2]==0):
                raise ValueError(f'{path.name}:{i}: graph cuts disagree with oracle bond retirement')
            pred=model(model.prepare_batch(raw));cut=model.predict_classes(pred).numpy()>0
            predicted=components(n,edges,eligible&~cut)
            retired=predicted[edges[:,0]]!=predicted[edges[:,1]]
            def retirement_counts(r):return {'tp':int((r&oracle_retired).sum()),'fp':int((r&~oracle_retired).sum()),'fn':int((~r&oracle_retired).sum())}
            rows.append({'event':path.name,'family':ev.manifest['structure_type'],'interval':i,
                         'model':partition_metrics(start,truth,predicted,mass),
                         'no_new_break_baseline':partition_metrics(start,truth,start,mass),
                         'retirement':retirement_counts(retired)})
    if not rows:raise ValueError('no matching topology records')
    def summarize(subset):
        result={'frames':len(subset)}
        for name in ('model','no_new_break_baseline'):
            samples=[r[name] for r in subset];out={}
            for key in ('false_split_mass_bits','false_merge_mass_bits','largest_fragment_mass_fraction_mae'):
                out[key]=float(np.mean([r[key] for r in samples])) if samples else None
            out['fragment_count_mae']=float(np.mean([abs(r['oracle_fragments']-r['predicted_fragments']) for r in samples])) if samples else None
            out['exact_partition_frames']=sum(r['exact_partition'] for r in samples)
            for key in ('connected_pairs','separated_pairs'):
                totals={k:sum(r[key][k] for r in samples) for k in ('tp','fp','fn')};a,b,c=[totals[k] for k in ('tp','fp','fn')]
                out[key]={**totals,'f1':2*a/(2*a+b+c) if 2*a+b+c else None}
            result[name]=out
        return result
    summary={'all':summarize(rows),'oracle_splits':summarize([r for r in rows if r['model']['oracle_split_parents']]),
             'oracle_no_split':summarize([r for r in rows if not r['model']['oracle_split_parents']])}
    return {'checkpoint':meta,'checkpoint_sha256':hashlib.sha256((weights/'state_dict.pt').read_bytes()).hexdigest(),
            'source_sha256':sources,'partition':partition,'summary':summary,'rows':rows,
            'scope':'New fragment topology from each oracle frame-start graph. Oracle endpoint cluster partition and retirement are verified against connectivity. No model feedback, predicted motion, contact dynamics or independent rollout. Pair metrics exclude unrelated frame-start clusters; mass metrics average over frames.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','dataset','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--partition',choices=['train','validation','test'],default='test');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=audit(a.weights,a.dataset,a.partition);a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['summary']))


if __name__=='__main__':main()
