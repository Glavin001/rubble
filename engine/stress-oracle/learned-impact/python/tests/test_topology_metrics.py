import numpy as np
import pytest
from learned_impact.topology_metrics import components,partition_metrics,canonical_partition


def test_redundant_crack_vs_fragmenting_graph_cut():
    e=np.array([[0,1],[1,2],[2,0]])
    original=components(3,e,np.ones(3,bool))
    redundant=components(3,e,np.array([False,True,True]))
    split=components(3,e,np.array([False,False,True]))
    assert partition_metrics(original,original,redundant,np.ones(3))['exact_partition']
    assert not partition_metrics(original,original,split,np.ones(3))['exact_partition']


def test_pair_counts_match_independent_quadratic_definition_on_small_graphs():
    rng=np.random.default_rng(4)
    for n in range(1,20):
        start=np.zeros(n,dtype=int);truth=rng.integers(0,4,n);pred=rng.integers(0,5,n)
        r=partition_metrics(start,truth,pred,np.ones(n))
        counts=dict(tp=0,fp=0,fn=0)
        for i in range(n):
            for j in range(i):
                t=truth[i]==truth[j];p=pred[i]==pred[j]
                if t and p:counts['tp']+=1
                elif p:counts['fp']+=1
                elif t:counts['fn']+=1
        assert all(r['connected_pairs'][k]==v for k,v in counts.items())
        assert sum([r['connected_pairs']['tp'],r['connected_pairs']['fp'],r['connected_pairs']['fn'],r['separated_pairs']['tp']])==n*(n-1)//2


def test_false_split_merge_mass_weighting_ids_and_parent_exclusion():
    parent=np.zeros(4,dtype=int);split=np.array([0,0,1,1]);w=np.ones(4)
    r=partition_metrics(parent,parent,split,w)
    assert r['connected_pairs']['fn']==4 and r['false_split_mass_bits']==1 and r['false_merge_mass_bits']==0
    r=partition_metrics(parent,split,parent,w)
    assert r['connected_pairs']['fp']==4 and r['false_merge_mass_bits']==1 and r['false_split_mass_bits']==0
    r=partition_metrics(split,split,split,w)
    assert r['within_parent_pairs']==2 and r['exact_partition']
    assert partition_metrics(parent,parent,split,np.array([8.,8.,1.,1.]))['largest_fragment_mass_fraction_mae']==pytest.approx(1/9)
    big=np.array([2**63+1,2**63+1,2**63+2],dtype=np.uint64)
    assert canonical_partition(big).tolist()==[0,0,1]
    with pytest.raises(ValueError,match='merges'):partition_metrics(split,parent,split,w)


def test_large_sparse_graph_and_empty_graph_without_pair_matrix():
    n=10400;e=np.column_stack([np.arange(n-1),np.arange(1,n)])
    c=components(n,e,np.ones(n-1,bool));r=partition_metrics(c,c,c,np.ones(n))
    assert r['connected_pairs']['tp']==n*(n-1)//2
    assert components(0,np.empty((0,2),dtype=int),np.empty(0,bool)).size==0
    with pytest.raises(ValueError,match='endpoint'):components(1,np.array([[0,1]]),np.array([True]))


def test_many_small_parent_bodies_and_empty_partition():
    n=20800;start=np.arange(n)//2;truth=np.arange(n);prediction=start.copy()
    r=partition_metrics(start,truth,prediction,np.ones(n))
    assert r['parent_clusters']==10400
    assert r['within_parent_pairs']==10400
    assert r['false_merge_mass_bits']==1
    assert r['largest_fragment_mass_fraction_mae']==.5
    singleton=partition_metrics(truth,truth,truth,np.ones(n))
    assert singleton['exact_partition'] and singleton['within_parent_pairs']==0
    assert singleton['largest_fragment_mass_fraction_mae']==0
    empty=partition_metrics(np.array([],int),np.array([],int),np.array([],int),np.array([]))
    assert empty['exact_partition'] and empty['false_merge_mass_bits'] is None
