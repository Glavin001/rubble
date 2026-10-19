"""Fragment metrics without allocating an N-by-N chunk-pair matrix.

Pairs are scored only within each frame-start cluster. Mass-weighted conditional
entropies distinguish false splits from false merges without depending on IDs.
"""
import numpy as np


def canonical_partition(labels):
    labels=np.asarray(labels)
    if labels.ndim!=1 or labels.dtype.kind not in 'iu':raise ValueError('integer partition labels required')
    mapping={};out=np.empty(len(labels),dtype=np.int64)
    for i,label in enumerate(labels):out[i]=mapping.setdefault(int(label),len(mapping))
    return out


def components(n,endpoints,connected):
    edges=np.asarray(endpoints);mask=np.asarray(connected)
    if edges.shape!=(len(mask),2) or edges.dtype.kind not in 'iu' or mask.dtype!=bool:
        raise ValueError('invalid topology graph')
    if n<0 or (edges<0).any() or (edges>=n).any():raise ValueError('topology endpoint outside graph')
    parent=np.arange(n,dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    for a,b in edges[mask]:
        a=find(int(a));b=find(int(b))
        if a!=b:parent[max(a,b)]=min(a,b)
    return np.array([find(i) for i in range(n)],dtype=np.int64)


def partition_metrics(start,truth,prediction,mass):
    start=canonical_partition(start);truth=canonical_partition(truth);prediction=canonical_partition(prediction)
    mass=np.asarray(mass,dtype=np.float64);n=len(start)
    if truth.shape!=start.shape or prediction.shape!=start.shape or mass.shape!=start.shape or not np.isfinite(mass).all() or (mass<=0).any():
        raise ValueError('invalid partition/mass alignment')
    # Neither an oracle split nor a deletion-only prediction can merge parents.
    for labels in (truth,prediction):
        if len(np.unique(np.column_stack([start,labels]),axis=0))!=len(np.unique(labels)):
            raise ValueError('end partition merges frame-start clusters')
    total_pairs=tp=fp=fn=0;false_split_bits=false_merge_bits=largest_error=0.
    changed_parents=predicted_changed=0;max_fraction_error=0.
    # Sort membership once: repeated full-array scans become quadratic when
    # destruction leaves thousands of small rigid bodies.
    order=np.argsort(start,kind='stable')
    boundaries=np.flatnonzero(np.diff(start[order]))+1
    for ids in np.split(order,boundaries) if n else ():
        if len(ids)==1:
            # A singleton parent cannot split or merge under the contract.
            continue
        t=truth[ids];p=prediction[ids];w=mass[ids];total=float(w.sum())
        _,ti,tc=np.unique(t,return_inverse=True,return_counts=True)
        _,pi,pc=np.unique(p,return_inverse=True,return_counts=True)
        _,cell,cc=np.unique(np.column_stack([ti,pi]),axis=0,return_inverse=True,return_counts=True)
        same=lambda counts:int(sum(int(c)*(int(c)-1)//2 for c in counts))
        both=same(cc);true_pairs=same(tc);pred_pairs=same(pc)
        tp+=both;fp+=pred_pairs-both;fn+=true_pairs-both
        total_pairs+=len(ids)*(len(ids)-1)//2
        tm=np.bincount(ti,weights=w);pm=np.bincount(pi,weights=w);cm=np.bincount(cell,weights=w)
        entropy=lambda weights:float(-np.sum((weights/total)*np.log2(weights/total)))
        # H(pred|truth) counts false splits; H(truth|pred) counts false merges.
        false_split_bits+=total*max(0.,entropy(cm)-entropy(tm))
        false_merge_bits+=total*max(0.,entropy(cm)-entropy(pm))
        error=abs(float(tm.max()-pm.max()))/total
        largest_error+=total*error;max_fraction_error=max(max_fraction_error,error)
        changed_parents+=len(tm)>1;predicted_changed+=len(pm)>1
    total_mass=float(mass.sum())
    separated_tp=total_pairs-tp-fp-fn
    def scores(a,b,c):
        return {'tp':a,'fp':b,'fn':c,'f1':2*a/(2*a+b+c) if 2*a+b+c else None}
    return {'chunks':n,'parent_clusters':len(np.unique(start)),
            'oracle_fragments':len(np.unique(truth)),'predicted_fragments':len(np.unique(prediction)),
            'oracle_split_parents':int(changed_parents),'predicted_split_parents':int(predicted_changed),
            'within_parent_pairs':total_pairs,'connected_pairs':scores(tp,fp,fn),
            'separated_pairs':scores(separated_tp,fn,fp),
            'false_split_mass_bits':false_split_bits/total_mass if total_mass else None,
            'false_merge_mass_bits':false_merge_bits/total_mass if total_mass else None,
            'largest_fragment_mass_fraction_mae':largest_error/total_mass if total_mass else None,
            'largest_fragment_mass_fraction_max_error':max_fraction_error if n else None,
            'exact_partition':bool(np.array_equal(canonical_partition(truth),canonical_partition(prediction)))}
