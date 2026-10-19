import pytest
from learned_impact.temporal_metrics import FractureTimeline


def test_timeline_separates_early_breaks_from_wrong_locations_and_keeps_first_prediction():
    t=FractureTimeline()
    t.add('a',0,.01,[1,2,3],[0,0,0],[1,0,4])
    t.add('a',2,.01,[1,2,3],[1,2,0],[1,0,0])
    r=t.report()
    assert r['clip_union']['tp']==1 and r['clip_union']['fp']==1 and r['clip_union']['fn']==1
    assert r['median_signed_first_break_error_ms']==pytest.approx(-20)
    assert r['first_break_more_than_one_interval_early']==1
    assert r['within_one_interval_fraction']==0
    t.add('b',0,.01,[1],[1],[1]) # Bond IDs are scoped to a scene.
    assert t.report()['clip_union']['tp']==2


def test_timeline_handles_empty_clips_and_rejects_duplicate_or_misaligned_labels():
    t=FractureTimeline();t.add('quiet',0,.01,[],[],[])
    assert t.report()['clip_union']['f1'] is None
    assert t.report()['median_absolute_first_break_error_ms'] is None
    with pytest.raises(ValueError,match='unique'):t.add('quiet',1,.01,[1,1],[0,0],[0,0])
    t.add('a',0,.01,[1],[1],[0])
    with pytest.raises(ValueError,match='more than once'):t.add('a',1,.01,[1],[1],[0])
    with pytest.raises(ValueError,match='dt changed'):t.add('a',2,.02,[2],[0],[0])
