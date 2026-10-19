import random
from collections import Counter
import pytest
import torch
from learned_impact.train_physics import TrainingSampler


def records():
    return [(name,'family',i,{'bond_fail_class':torch.tensor([i%2])})
            for name,n in [('short',2),('long',11),('alias',3)] for i in range(n)]


def test_frame_sampler_preserves_historical_rng_order_and_budget():
    rows=records();groups={'short':'a','long':'b','alias':'a'}
    for oversample in (False,True):
        sampler=TrainingSampler(rows,groups,random.Random(4),oversample_breaks=oversample)
        previous=random.Random(4)
        for _ in range(3):
            expected=list(range(len(rows)))
            if oversample:expected += [i for i,r in enumerate(rows) if (r[3]['bond_fail_class']>0).any()]
            previous.shuffle(expected)
            order,report=sampler.epoch()
            assert order==expected
            assert report['updates']==len(expected)==sampler.steps
            assert report['unique_intervals']==len(rows)


def test_group_sampler_balances_physical_aliases_and_covers_long_tails():
    rows=records();groups={'short':'a','long':'b','alias':'a'}
    sampler=TrainingSampler(rows,groups,random.Random(7),'physical-group')
    replica=TrainingSampler(rows,groups,random.Random(7),'physical-group')
    visits=Counter()
    for epoch in range(5):
        order,report=sampler.epoch()
        assert (order,report)==replica.epoch()
        assert len(order)==len(rows)
        assert report['group_updates']=={'a':8,'b':8}
        visits.update(order)
        if epoch>=1:assert len(visits)==len(rows) # persistent queues reach all tail intervals
    assert min(visits[i] for i in range(2,13))>=3
    assert report['cumulative_unique_intervals']==len(rows)
    with pytest.raises(ValueError,match='cannot be combined'):
        TrainingSampler(rows,groups,random.Random(1),'physical-group',True)


def test_uneven_budget_does_not_starve_groups_across_epochs():
    rows=[(str(i),'family',0,{'bond_fail_class':torch.tensor([0])}) for i in range(5)]
    groups={'0':'a','1':'b','2':'c','3':'a','4':'a'}
    sampler=TrainingSampler(rows,groups,random.Random(3),'physical-group')
    counts=Counter()
    for _ in range(3):
        _,r=sampler.epoch();counts.update(r['group_updates'])
        assert max(counts.values())-min(counts.values())<=1
    assert counts=={'a':5,'b':5,'c':5}


def test_explicit_budget_preserves_balance_and_eventual_tail_coverage():
    rows=records();groups={'short':'a','long':'b','alias':'a'}
    sampler=TrainingSampler(rows,groups,random.Random(7),'physical-group',updates_per_epoch=3)
    replica=TrainingSampler(rows,groups,random.Random(7),'physical-group',updates_per_epoch=3)
    counts=Counter()
    for _ in range(22):
        order,report=sampler.epoch();assert (order,report)==replica.epoch()
        assert len(order)==report['updates']==3
        counts.update(report['group_updates'])
        assert max(counts.values())-min(counts.values())<=1
    assert report['cumulative_unique_intervals']==len(rows)
    assert sum(sampler.visits)==66
    for invalid in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='positive integer'):
            TrainingSampler(rows,groups,random.Random(7),'physical-group',updates_per_epoch=invalid)
    with pytest.raises(ValueError,match='physical-group'):
        TrainingSampler(rows,groups,random.Random(7),updates_per_epoch=3)
