"""Evaluate short/extended checkpoints on identical physical recipe partitions.

Extended clips inherit recipe identity, so filenames can differ. Refuse any
partition mismatch instead of assuming that a renamed clip remains held out.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .train_physics import load_records,evaluate
from .visualize import load_checkpoint


def verify_recipe_partitions(saved, evaluated):
    known={}
    for name,part in saved['events'].items():
        recipe=saved.get('recipe_groups',{}).get(name)
        if recipe is None:raise ValueError('source checkpoint lacks explicit recipe identities')
        if recipe in known and known[recipe]!=part:raise ValueError('source recipe partition conflict')
        known[recipe]=part
    for name,part in evaluated['events'].items():
        recipe=evaluated['recipe_groups'].get(name)
        if recipe not in known or known[recipe]!=part:
            raise ValueError(f'{name}: recipe missing or partition differs from checkpoint')


def compare(dataset, reference_dataset, reference_split, weights):
    torch.set_num_threads(4)
    original=json.loads((reference_dataset/'manifest.json').read_text())
    lengths={e['event_id']:e['n_frames'] for e in original['events']}
    result=[]
    for path in weights:
        model,meta=load_checkpoint(path)
        records,split=load_records(dataset,model,split_from=reference_split)
        saved=json.loads((path/'split.json').read_text())
        verify_recipe_partitions(saved,split)
        short=[];tail=[]
        for record in records['test']:
            # Catalog-prefixed group is saved independently of clip filename.
            recipe=split['recipe_groups'][record[0]].split(':',1)[1]
            if recipe not in lengths:raise ValueError('unknown original recipe duration')
            (short if record[2]<lengths[recipe] else tail).append(record)
        result.append({'checkpoint':meta,'original_prefix_test':evaluate(model,short),
                       'later_tail_test':evaluate(model,tail),'combined_test':evaluate(model,records['test'])})
    return {'models':result,'scope':'Same held-out recipes and saved thresholds; original-duration intervals versus added later tails.',
            'limits':['Extended fitting adds training intervals and optimizer updates; this is not an equal-compute data ablation.',
                      'Only the extended held-out recipes contribute later-tail metrics; inspect their sample and failure counts before generalizing.',
                      'Every interval still starts from oracle state and topology; no autonomous rollout.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','reference-dataset','reference-split','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--weights',type=Path,action='append',required=True)
    a=p.parse_args();r=compare(a.dataset,a.reference_dataset,a.reference_split,a.weights)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    for row in r['models']:
        print(json.dumps({'name':row['checkpoint']['name'],**{phase:{k:row[phase][k] for k in ('f1','precision','recall','tp','fp','fn','frames')} for phase in ('original_prefix_test','later_tail_test','combined_test')}}))


if __name__=='__main__':main()
