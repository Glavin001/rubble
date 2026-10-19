from pathlib import Path
import json
import shutil
import numpy as np
import pytest
import torch
from learned_impact.dataset import event_batches, synthetic_batch
from learned_impact.experiment import collect
from learned_impact.losses import total_loss
from learned_impact.model import LearnedImpactNet, BondMLPBaseline
from learned_impact.normalize import normalize_batch
from learned_impact.schema import load_event
from learned_impact.train import eval_mean_loss, run_eval
from learned_impact.visualize import compare_event, scores

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/tiny_event'


def test_normalization_applied_once_in_evaluation():
    torch.manual_seed(0)
    model = LearnedImpactNet(h=8, p=4)
    raw = synthetic_batch()
    direct = eval_mean_loss(model, [raw])
    already = eval_mean_loss(model, [normalize_batch(raw)], normalized=True)
    assert direct == pytest.approx(already)
    # Regression sentinel: repeated gravity/target scaling changes the result.
    twice = eval_mean_loss(model, [normalize_batch(raw)])
    assert twice['loss'] != pytest.approx(direct['loss'], abs=1e-6)


def test_previous_failures_removed_and_empty_graph_remains_finite():
    ev = load_event(FIXTURE)
    ev.frames[0]['bond_fail_class'][:] = 2
    ev.frames[1]['bond_fail_class'][:] = 4
    first, second = list(event_batches(ev))
    assert len(first['bond_ids']) == ev.manifest['n_bonds']
    assert second['bond_endpoints'].shape == (0, 2)
    assert len(second['bond_fail_class']) == 0
    model = LearnedImpactNet(h=8, p=4)
    b = normalize_batch(second)
    parts = total_loss(model(b), b)
    assert torch.isfinite(parts['loss'])
    assert parts['fail'] == parts['util'] == 0
    parts['loss'].backward()
    assert torch.isfinite(total_loss(BondMLPBaseline(h=8)(b), b)['loss'])


def test_collect_deduplicates_bytes_keeps_provenance(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    shutil.copytree(FIXTURE,a);shutil.copytree(FIXTURE,b)
    m=json.loads((b/'manifest.json').read_text());m['event_id']='different-name'
    (b/'manifest.json').write_text(json.dumps(m))
    out=tmp_path/'collection'
    man=collect([a,b],out)
    assert len(man['events'])==1
    assert len(man['events'][0]['sources'])==2
    assert load_event(out/man['events'][0]['path']).manifest['n_frames']==2
    with pytest.raises(FileExistsError):collect([a],out)


def test_collection_keeps_distinct_world_observations_with_identical_local_arrays(tmp_path):
    from learned_impact.experiment import event_digest,collection_digest
    from test_world_inputs import recording
    a,b,c=(tmp_path/name for name in ('a','b','alias'))
    for p in (a,b,c):shutil.copytree(FIXTURE,p)
    replay,scene,context,_=recording(4)
    for i,frame in enumerate(replay['frames']):
        frame['clusters']=[0]*4;frame['time']=i/60
        frame['impactors'][0]['angular_velocity']=[0.,0.,20.]
    for p in (a,b,c):
        for name,value in [('scene',scene),('context',context),('replay',replay)]:
            (p/(name+'.json')).write_text(json.dumps(value))
    for frame in replay['frames']:frame['impactors'][0]['angular_velocity']=[0.,0.,-20.]
    (b/'replay.json').write_text(json.dumps(replay))
    assert event_digest(a)==event_digest(b) # Legacy local-array identity.
    assert collection_digest(a)!=collection_digest(b)
    alias=json.loads((c/'scene.json').read_text());alias.update(name='alias',description='renamed',oracle={'note':'provenance'})
    (c/'scene.json').write_text(json.dumps(alias))
    assert collection_digest(a)==collection_digest(c)
    man=collect([a,b,c],tmp_path/'combined')
    assert len(man['events'])==2 and len({e['path'] for e in man['events']})==2
    assert sorted(len(e['sources']) for e in man['events'])==[1,2]
    for entry in man['events']:
        saved=json.loads((tmp_path/'combined'/entry['path']/'replay.json').read_text())
        assert saved['source_sha256']==entry['sha256']
        assert collection_digest(tmp_path/'combined'/entry['path'])==entry['collection_sha256']


def test_comparison_uses_checkpoint_outputs_and_full_bond_indices():
    torch.manual_seed(0)
    model=LearnedImpactNet(h=8,p=4).eval()
    ev=load_event(FIXTURE)
    result=compare_event(FIXTURE,model,{'centers':[],'rotations':[]})
    for f,b in zip(result['frames'],event_batches(ev)):
        with torch.no_grad(): direct=model(normalize_batch(b))
        assert f['active']==b['bond_ids'].tolist()
        assert f['predicted']==direct['bond_fail_logits'].argmax(-1).tolist()
    # Report serializes with strict finite JSON and scores only new failures.
    json.dumps(result,allow_nan=False)
    assert result['metrics']['tp']==sum(f['metrics']['tp'] for f in result['frames'])


def test_no_failures_are_not_a_perfect_positive_f1():
    assert scores(np.zeros(3,dtype=bool),np.zeros(3,dtype=bool))['f1'] is None


def test_evaluation_refuses_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError,match='refusing random'):
        run_eval(FIXTURE,tmp_path,8,4,structure_type=None,arch='mgn')


def test_replay_requires_matching_source_and_frame_alignment(tmp_path):
    from learned_impact.visualize import load_replay
    frame = {'time': 0, 'centers': [[0, 0, 0]], 'rotations': [np.eye(3).tolist()],
             'active': [True], 'clusters': [0], 'impactors': []}
    replay = {'format': 'learned-impact-replay/1', 'source_sha256': 'exact-record',
              'frames': [frame, {**frame, 'time': .01}]}
    path = tmp_path / 'replay.json'
    entry = {'sha256': 'exact-record'}
    manifest = {'n_frames': 1, 'n_chunks': 1, 'dt': .01}
    def save(): path.write_text(json.dumps(replay))
    save()
    assert len(load_replay(path, entry, manifest)['frames']) == 2
    with pytest.raises(ValueError, match='fingerprint'):
        load_replay(path, {'sha256': 'different'}, manifest)
    replay['frames'][1]['time'] = .02; save()
    with pytest.raises(ValueError, match='time'): load_replay(path, entry, manifest)
    replay['frames'][1]['time'] = .01
    replay['frames'][1]['centers'] = []; save()
    with pytest.raises(ValueError, match='centers'): load_replay(path, entry, manifest)
    replay['frames'].pop(); save()
    with pytest.raises(ValueError, match='initial pose'): load_replay(path, entry, manifest)


def test_collect_keeps_native_replay_with_event_fingerprint(tmp_path):
    src = tmp_path / 'source'
    shutil.copytree(FIXTURE, src)
    (src / 'replay.json').write_text(json.dumps({'format': 'learned-impact-replay/1', 'frames': []}))
    out = tmp_path / 'collection'
    entry = collect([src], out)['events'][0]
    replay = json.loads((out / entry['path'] / 'replay.json').read_text())
    assert replay['source_sha256'] == entry['sha256']
    assert 'OracleSession' in replay['verification']


def test_visualizer_masks_fragment_coordinate_resets_in_motion_views(monkeypatch):
    from learned_impact import visualize
    from learned_impact.physics import PhysicsImpactNet
    ev=load_event(FIXTURE);n=ev.manifest['n_chunks']
    ev.cluster_history=[np.zeros(n,dtype=int).tolist()]+[np.ones(n,dtype=int).tolist() for _ in ev.frames]
    monkeypatch.setattr(visualize,'load_event',lambda *args,**kwargs:ev)
    result=visualize.compare_event(FIXTURE,PhysicsImpactNet(h=8,p=4),{'centers':[],'rotations':[]})
    f=result['frames'][0]
    assert f['motion_valid']==[False]*n
    assert f['truth_delta']==f['predicted_delta']==f['delta_error']==[None]*n
    assert f['metrics']['displacement_rmse_m'] is None
    assert f['metrics']['motion_fragment_reset_excluded_chunk_intervals']==n
    assert all(result['frames'][1]['motion_valid'])
    assert result['frames'][1]['metrics']['displacement_rmse_m'] is not None
