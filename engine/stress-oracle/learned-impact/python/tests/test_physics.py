import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from learned_impact.dataset import event_batches, synthetic_batch, bidirectional_csr
from learned_impact.physics import PhysicsImpactNet, SizeStableAttention, physics_loss, symlog, symexp, DELTA_SCALE
from learned_impact.schema import load_event
from learned_impact.train_physics import physical_group, split_events
from learned_impact.visualize import load_checkpoint, compare_event

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/tiny_event'


def test_csr_preserves_neighbor_order_and_self_edges():
    row,col = bidirectional_csr(np.array([[2,0],[0,1],[0,0],[3,2]],dtype=np.int32),4)
    assert row.tolist() == [0,4,5,7,8]
    assert col.tolist() == [2,1,0,0,0,0,3,2]
    row,col = bidirectional_csr(np.empty((0,2),dtype=np.int32),4)
    assert row.tolist() == [0,0,0,0,0] and col.size == 0
    with pytest.raises(ValueError,match='endpoint'):
        bidirectional_csr(np.array([[0,4]]),4)


def test_physical_inputs_preserve_force_material_and_time_information():
    b = next(event_batches(load_event(FIXTURE)))
    model = PhysicsImpactNet(h=8, p=4)
    original = model.features(b)
    for key in ('chunk_load_bins', 'chunk_mass', 'dt', 'bond_strength', 'bond_stiffness', 'bond_fracture_energy'):
        changed = dict(b, **{key: b[key] * 2 + .01})
        features = model.features(changed)
        assert any(not torch.equal(a, c) for a,c in zip(original,features)), key


def test_token_broadcast_is_invariant_to_replicating_identical_nodes():
    torch.manual_seed(9)
    # Test the pooling invariant in double precision, separately from float32
    # summation roundoff over 10,000 nodes.
    attn = SizeStableAttention(h=8, p=4).double().eval()
    node = torch.randn(5, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = attn(node)
        actual = attn(node.repeat(2000, 1)).reshape(2000, 5, 8)
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=1e-10, atol=1e-10)


def test_break_probability_is_not_diluted_by_uncertain_timing():
    model = PhysicsImpactNet(h=8, p=4, threshold=.5)
    pred = {'break_logit': torch.logit(torch.tensor([.6, .2])), 'time_logits': torch.zeros(2,4)}
    assert model.predict_classes(pred).tolist() == [1,0]
    # Five-class argmax would incorrectly choose no-break at .6 total break probability.
    assert torch.tensor([.4,.15,.15,.15,.15]).argmax() == 0


def test_empty_graph_finite_loss_and_static_features_follow_live_edges():
    ev = load_event(FIXTURE)
    ev.frames[0]['bond_fail_class'][:] = 1
    b = list(event_batches(ev))[1]
    for k,v in b.items():
        if k.startswith('bond_'):
            assert len(v) == 0, k
    model = PhysicsImpactNet(h=8,p=4)
    parts = physics_loss(model(b), b)
    assert all(torch.isfinite(v) for v in parts.values())
    parts['loss'].backward()


def test_delta_scaling_roundtrip_and_split_coordinate_changes_are_masked():
    x = torch.randn(3, 12) * torch.tensor(DELTA_SCALE)
    torch.testing.assert_close(symexp(symlog(x, DELTA_SCALE), DELTA_SCALE), x)
    b = synthetic_batch()
    b['delta_valid'] = torch.zeros(len(b['chunk_state']), dtype=torch.bool)
    model = PhysicsImpactNet(h=8,p=4)
    pred = model(b)
    assert physics_loss(pred,b)['delta'] == 0
    # Labels cannot enter encoders.
    changed = dict(b, bond_fail_class=torch.ones_like(b['bond_fail_class']),
                   bond_peak_util=b['bond_peak_util']*1000, chunk_state_delta=b['chunk_state_delta']*1e9)
    for a,c in zip(model.features(b),model.features(changed)):
        torch.testing.assert_close(a,c)


def test_prefix_clips_and_qs_variants_stay_in_one_partition():
    ev = load_event(FIXTURE)
    short = copy.deepcopy(ev);short.frames=short.frames[:1];short.manifest['n_frames']=1
    probe = copy.deepcopy(ev);probe.frames[0]['bond_qs_force'] *= 3
    assert physical_group(ev) == physical_group(short) == physical_group(probe)
    events = {'full':ev, 'short':short, 'qs':probe}
    for i in range(10):
        varied = copy.deepcopy(ev);varied.static['chunk_mass'] += i+1
        events[str(i)] = varied
    split,groups = split_events(events)
    assert split['full'] == split['short'] == split['qs']
    assert len(set(split.values())) == 3
    for group in set(groups.values()):
        assert len({split[e] for e,g in groups.items() if g==group}) == 1


def test_versioned_checkpoint_and_viewer_use_identical_decoder(tmp_path):
    model = PhysicsImpactNet(h=8,p=4,threshold=.35).eval()
    torch.save(model.state_dict(), tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':'physics-v2','threshold':.35}))
    loaded,meta = load_checkpoint(tmp_path)
    b = next(event_batches(load_event(FIXTURE)))
    with torch.no_grad():
        torch.testing.assert_close(model(b)['break_logit'],loaded(b)['break_logit'])
        expected = model.predict_classes(model(b)).tolist()
    result = compare_event(FIXTURE,loaded,{'centers':[],'rotations':[]})
    assert result['frames'][0]['predicted'] == expected
    json.dumps(result,allow_nan=False)


def test_preview_model_never_reads_future_oracle_loads_or_contact():
    from learned_impact.physics import PreviewImpactNet
    b = next(event_batches(load_event(FIXTURE)))
    model = PreviewImpactNet(h=8, p=4)
    with pytest.raises(ValueError, match='requires chunk_preview_load_bins'):
        model.features(b)
    b['chunk_preview_load_bins'] = b['chunk_load_bins'].clone() + 1.
    b['chunk_preview_contact'] = torch.zeros(len(b['chunk_mass']))
    original = model.features(b)
    with pytest.raises(ValueError, match='cached features'):
        model(PhysicsImpactNet(h=8,p=4).prepare_batch(b))
    for key,value in model(b).items():
        torch.testing.assert_close(value,model(model.prepare_batch(b))[key])
    poisoned = dict(b, chunk_load_bins=torch.full_like(b['chunk_load_bins'], float('nan')),
                    chunk_contact=torch.full_like(b['chunk_contact'], 999))
    for a, c in zip(original, model.features(poisoned)):
        assert torch.equal(a, c)
    changed = dict(b, chunk_preview_load_bins=b['chunk_preview_load_bins'] * 2)
    assert not torch.equal(original[0], model.features(changed)[0])


def test_preview_schema_requires_complete_versioned_finite_arrays(tmp_path):
    import shutil
    from learned_impact.constants import K, LOCAL6_DIM
    root = tmp_path / 'event'
    shutil.copytree(FIXTURE, root)
    context = {'load_preview': {'format': 'rigid-contact-preview/1'}}
    (root/'context.json').write_text(json.dumps(context))
    with pytest.raises(ValueError, match='preview contract/array mismatch'):
        load_event(root)
    n = load_event(FIXTURE).manifest['n_chunks']
    for d in sorted((root/'frames').iterdir()):
        catalog = json.loads((d/'catalog.json').read_text())
        for name, shape, dtype in [('chunk_preview_load_bins', (n,K,LOCAL6_DIM), 'f32'), ('chunk_preview_contact', (n,), 'i32')]:
            np.zeros(shape, dtype=np.float32 if dtype == 'f32' else np.int32).tofile(d/f'{name}.bin')
            catalog['arrays'][name] = dict(dtype=dtype, shape=shape, path=f'{name}.bin')
        (d/'catalog.json').write_text(json.dumps(catalog))
    ev = load_event(root)
    assert 'chunk_preview_load_bins' in next(event_batches(ev))
    a = np.zeros((n,K,LOCAL6_DIM), dtype=np.float32);a[0,0,0] = np.nan
    a.tofile(root/'frames/000000/chunk_preview_load_bins.bin')
    with pytest.raises(ValueError, match='invalid chunk_preview_load_bins'):
        load_event(root)


def test_preview_checkpoint_retains_its_input_contract(tmp_path):
    from learned_impact.physics import PreviewImpactNet
    model = PreviewImpactNet(h=8,p=4,threshold=.7)
    torch.save(model.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':model.architecture,'threshold':.7}))
    loaded,meta = load_checkpoint(tmp_path)
    assert isinstance(loaded,PreviewImpactNet) and loaded.threshold == .7
    # A same-shaped checkpoint must never silently accept oracle-only inputs.
    with pytest.raises(ValueError,match='requires chunk_preview_load_bins'):
        loaded(next(event_batches(load_event(FIXTURE))))


def test_cadences_share_trajectory_split_and_can_inherit_exact_membership():
    from learned_impact.train_physics import inherited_split
    original = load_event(FIXTURE)
    original.context['generator'] = {'catalog':'test','split_group':'base'}
    fine = copy.deepcopy(original)
    fine.manifest['event_id'] = 'base_sub4'
    fine.manifest['dt'] /= 4
    fine.frames[0]['chunk_load_bins'] *= .2
    assert physical_group(original) != physical_group(fine)
    events = {'base':original,'base_sub4':fine}
    parts,groups = split_events(events)
    assert groups['base'] == groups['base_sub4']
    reference = {'events': {'base':'test'},'physical_groups':{'base':'saved-group'}}
    parts,groups = inherited_split(events,reference)
    assert set(parts.values()) == {'test'} and set(groups.values()) == {'saved-group'}
    reference['events']['base_sub4'] = 'train'
    reference['physical_groups']['base_sub4'] = 'other'
    with pytest.raises(ValueError,match='conflicting'):
        inherited_split(events,reference)
    with pytest.raises(ValueError,match='missing'):
        inherited_split(events, {'events':{}, 'physical_groups':{}})


def test_identical_first_frames_cannot_cross_an_inherited_partition():
    from learned_impact.train_physics import inherited_split
    a=load_event(FIXTURE);b=copy.deepcopy(a)
    a.manifest['event_id']='a';b.manifest['event_id']='b'
    reference={'events':{'a':'train','b':'test'},'physical_groups':{'a':'one','b':'two'}}
    with pytest.raises(ValueError,match='leaks'):
        inherited_split({'a':a,'b':b},reference)


def test_common_window_counts_repeated_predictions_once_per_bond():
    from learned_impact.train_physics import evaluate
    class Constant(PhysicsImpactNet):
        def forward(self,b):
            e=len(b['bond_endpoints']);n=len(b['chunk_mass'])
            logits=torch.full((e,),-10.);logits[:2]=10.
            return {'break_logit':logits,'time_logits':torch.zeros(e,4),
                    'bond_peak_util':torch.zeros(e),'chunk_state_delta':torch.zeros(n,12)}
    b=synthetic_batch();b['bond_ids']=torch.arange(len(b['bond_endpoints']))
    records=[]
    for i in range(4):
        frame=dict(b,bond_fail_class=torch.zeros_like(b['bond_fail_class']))
        if i==3:frame['bond_fail_class'][1]=1
        records.append(('event','test',i,frame))
    result=evaluate(Constant(h=8,p=4),records,window_factor=4)
    assert (result['tp'],result['fp'],result['fn'])==(1,7,0)
    window=result['window_metrics']
    assert (window['tp'],window['fp'],window['fn'],window['windows'])==(1,1,0,1)
    assert window['f1']==pytest.approx(2/3)


def test_state_model_identity_constraints_and_boundary_gradients(tmp_path):
    from learned_impact.physics import StatePreviewImpactNet, JOINT_SCALE
    b = next(event_batches(load_event(FIXTURE)))
    b.update(chunk_preview_load_bins=b['chunk_load_bins'].clone(),
             chunk_preview_contact=b['chunk_contact'].clone())
    model = StatePreviewImpactNet(h=8,p=4)
    model.fit_target_scales([('train','test',0,b)])
    pred = model(b)
    assert torch.count_nonzero(model.decode_delta(pred)) == 0
    torch.testing.assert_close(pred['bond_joint_state_end'], symlog(b['bond_joint_state'],JOINT_SCALE))
    # At the zero-increment boundary, increasing damage must remain learnable.
    pred['bond_joint_state_end'][:,0].sum().backward()
    assert model.joint_end.bias.grad[0] > 0
    with torch.no_grad():
        model.joint_end.bias.fill_(-10.)
        prior = symlog(b['bond_joint_state'],JOINT_SCALE)
        p = model(b)['bond_joint_state_end']
        for i in (*model.bounded_history,*model.monotonic_history):
            torch.testing.assert_close(p[:,i],prior[:,i])
        model.joint_end.bias.fill_(100.)
        p = model(b)['bond_joint_state_end']
        for i in (*model.bounded_history,*model.monotonic_history):
            assert (p[:,i] >= prior[:,i]).all()
        for i in model.bounded_history:
            assert (p[:,i] <= torch.log(torch.tensor(2.))).all()
    torch.save(model.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':model.architecture,'threshold':.5}))
    loaded,_ = load_checkpoint(tmp_path)
    torch.testing.assert_close(loaded.delta_rms,model.delta_rms)
    torch.testing.assert_close(loaded(b)['bond_joint_state_end'],model(b)['bond_joint_state_end'])


def test_state_loss_masks_fragment_coordinate_resets_and_has_finite_gradients():
    from learned_impact.physics import StatePreviewImpactNet
    b=next(event_batches(load_event(FIXTURE)))
    b.update(chunk_preview_load_bins=b['chunk_load_bins'],chunk_preview_contact=b['chunk_contact'])
    model=StatePreviewImpactNet(h=8,p=4)
    model.fit_target_scales([('train','test',0,b)])
    b['delta_valid']=torch.zeros(len(b['chunk_state']),dtype=torch.bool)
    pred=model(b);parts=model.state_loss(pred,b)
    assert parts['delta']==0 and all(torch.isfinite(v) for v in parts.values())
    parts['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Changing labels cannot change a fitted model's predictions.
    altered=dict(b,bond_joint_state_end=b['bond_joint_state_end']*100,
                 chunk_state_delta=b['chunk_state_delta']+100)
    for k,v in model(altered).items():torch.testing.assert_close(v,pred[k])


def test_history_metrics_compare_against_persistence_and_count_invalid_predictions():
    from learned_impact.compare_state import history_metrics
    from learned_impact.physics import JOINT_SCALE
    b=next(event_batches(load_event(FIXTURE)))
    b['bond_joint_state'][:,0]=.1
    b['bond_joint_state_end'][:,0]=.25
    class Persist(PhysicsImpactNet):
        def forward(self,b):return {'bond_joint_state_end':symlog(b['bond_joint_state'],JOINT_SCALE)}
    model=Persist(h=8,p=4)
    report=history_metrics(model,[('event','fixture',0,b)])
    assert report['damage_crush_mae'][0]==pytest.approx(.15)
    assert report['damage_changed_intervals']==len(b['bond_endpoints'])
    assert report['encoded_rmse']==report['persistence_encoded_rmse']
    assert sum(report['decreasing_history_count'].values())==0
    class Invalid(Persist):
        def forward(self,b):
            p=super().forward(b);p['bond_joint_state_end'][:,0]=-1
            return p
    report=history_metrics(Invalid(h=8,p=4),[('event','fixture',0,b)])
    assert report['decreasing_history_count']['damage']==len(b['bond_endpoints'])
    assert report['outside_unit_interval_count']['damage']==len(b['bond_endpoints'])


def test_frozen_state_fitting_preserves_all_fracture_outputs(tmp_path):
    from learned_impact.physics import PreviewImpactNet,StatePreviewImpactNet
    from learned_impact.train_physics import initialize_frozen_state_heads
    baseline=PreviewImpactNet(h=8,p=4,threshold=.65)
    torch.save(baseline.state_dict(),tmp_path/'state_dict.pt')
    (tmp_path/'run.json').write_text(json.dumps({'architecture':baseline.architecture,'threshold':.65}))
    model=StatePreviewImpactNet(h=8,p=4)
    initialize_frozen_state_heads(model,tmp_path)
    b=next(event_batches(load_event(FIXTURE)))
    b.update(chunk_preview_load_bins=b['chunk_load_bins'],chunk_preview_contact=b['chunk_contact'])
    model.fit_target_scales([('event','fixture',0,b)])
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        opt.zero_grad();model.state_loss(model(b),b)['loss'].backward();opt.step()
    for key,value in model.state_dict().items():
        if not key.startswith(('chunk_delta.','joint_end.')):assert torch.equal(value,before[key])
    for key in ('break_logit','time_logits','bond_fail_logits','bond_peak_util','bond_force_end'):
        assert torch.equal(model(b)[key],baseline(b)[key]),key
    assert model.threshold==baseline.threshold


def test_explicit_layout_holdout_unions_different_load_recipes_and_rejects_leakage():
    from learned_impact.train_physics import grouped_events, inherited_split, layout_group
    base=load_event(FIXTURE);events={}
    for layout in range(12):
        for case in range(4):
            ev=copy.deepcopy(base)
            ev.static['chunk_mass'] += 1+layout*4+case
            ev.context['generator']={'catalog':'layout-test','layout_group':f'layout-{layout}',
                                     'split_group':f'layout-{layout}-case-{case}'}
            events[f'{layout}-{case}']=ev
    parts,groups=split_events(events)
    assert len(set(groups.values()))==12 and set(parts.values())=={'train','validation','test'}
    for layout in range(12):
        names=[f'{layout}-{case}' for case in range(4)]
        assert len({parts[n] for n in names})==1
        assert len({groups[n] for n in names})==1
        assert len({layout_group(events[n]) for n in names})==1
    reference={'events':parts.copy(),'physical_groups':groups}
    assert inherited_split(events,reference)[0]==parts
    reference['events']['0-0']='test' if parts['0-1']!='test' else 'train'
    with pytest.raises(ValueError,match='leaks'):inherited_split(events,reference)


def test_motion_evaluation_reports_unavailable_when_all_targets_reset():
    from learned_impact.train_physics import evaluate
    b=synthetic_batch();b['bond_ids']=torch.arange(len(b['bond_endpoints']))
    b['delta_valid']=torch.zeros(len(b['chunk_mass']),dtype=torch.bool)
    m=PhysicsImpactNet(h=8,p=4)
    r=evaluate(m,[('reset','test',0,b)])
    assert r['motion_scored_chunk_intervals']==0
    assert r['motion_fragment_reset_excluded_chunk_intervals']==len(b['chunk_mass'])
    assert all(v is None for v in r['state_rmse'].values())
    assert all(v is None for v in r['zero_increment_rmse'].values())


def test_reused_layout_identity_crosses_load_catalogs_and_rejects_split_leakage():
    from learned_impact.train_physics import grouped_events, inherited_split, layout_group
    impact=load_event(FIXTURE);forcing=copy.deepcopy(impact)
    impact.context['generator']={'catalog':'layouts-v4','layout_group':'02','split_group':'ram'}
    forcing.context['generator']={'catalog':'forcing-v8','layout_catalog':'layouts-v4','layout_group':'02','split_group':'pulse'}
    forcing.static['chunk_mass']+=7 # Different raw frame hash must not defeat geometry grouping.
    events={'impact':impact,'forcing':forcing}
    assert layout_group(impact)==layout_group(forcing)=='layouts-v4:02'
    groups=grouped_events(events);assert groups['impact']==groups['forcing']
    with pytest.raises(ValueError,match='leaks'):
        inherited_split(events,{'events':{'impact':'train','forcing':'test'},'physical_groups':groups})
    assert inherited_split(events,{'events':{'impact':'train','forcing':'train'},'physical_groups':groups})[0]=={'impact':'train','forcing':'train'}
