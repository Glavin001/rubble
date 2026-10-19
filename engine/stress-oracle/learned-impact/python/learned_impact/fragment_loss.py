"""Optional training-only weakest-boundary supervision for fragment separation.

This is a surrogate, not a likelihood of an entire fragment partition. Existing
bond BCE remains responsible for redundant cracks and false positive breaks.
No target labels or endpoint topology enter model inputs or inference.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .topology_metrics import components


def fragment_cut_targets(n, endpoints, eligible, connected_end, failure_labels):
    """Group newly severed edges by the unordered pair of oracle end fragments.

Already cracked edges are excluded. A connected path that survives within a
fragment needs no additional supervision here. End connectivity must match the
recorded first-fracture labels exactly, rather than assuming label semantics.
"""
    edges=np.asarray(endpoints);eligible=np.asarray(eligible);end=np.asarray(connected_end)
    labels=np.asarray(failure_labels)
    if eligible.dtype!=bool or end.dtype!=bool or end.shape!=eligible.shape or labels.shape!=eligible.shape:
        raise ValueError('invalid fragment supervision shape/flags')
    if (end&~eligible).any() or not np.array_equal(labels>0,eligible&~end):
        raise ValueError('fragment connectivity disagrees with first-fracture targets')
    partition=components(n,edges,end)
    pairs=np.sort(partition[edges],axis=1)
    boundary=eligible&(pairs[:,0]!=pairs[:,1])
    groups=np.full(len(edges),-1,dtype=np.int64)
    if boundary.any():
        _,groups[boundary]=np.unique(pairs[boundary],axis=0,return_inverse=True)
    return torch.from_numpy(groups)


def fragment_cut_loss(break_logit, groups):
    """Mean worst break NLL per oracle fragment boundary, stable for all logits.

One missed bridge can keep two fragments joined. Taking the maximum rather
than the mean emphasizes that weakest break without rewarding extra unrelated
breaks. Targets are computed once at dataset loading, never from predictions.
For ties, scatter_reduce distributes gradients among equal maxima.
"""
    if groups.shape!=break_logit.shape or groups.dtype!=torch.int64:
        raise ValueError('invalid fragment cut groups')
    mask=groups>=0
    if not mask.any():return break_logit.sum()*0
    ids=groups[mask]
    count=int(ids.max())+1
    worst=torch.zeros(count,device=break_logit.device,dtype=break_logit.dtype)
    worst=worst.scatter_reduce(0,ids,F.softplus(-break_logit[mask]),reduce='amax',include_self=False)
    return worst.mean()
