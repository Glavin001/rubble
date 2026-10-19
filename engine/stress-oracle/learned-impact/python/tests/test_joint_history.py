import json
import numpy as np
import pytest
from pathlib import Path
import shutil
from learned_impact.joint_history import FORMAT,FIELDS,FLAGS,load_joint_history,finite_features


def recording(tmp_path):
    d=tmp_path/'joint_history';d.mkdir()
    values=np.zeros((2,21),dtype='<f8');values[:,4:6]=np.inf
    values[1,4]=1234567.891234567;values[0,7]=-0.;values[1,8]=-1.234567891234567
    flags=np.array([[0,0,1,1],[1,4,1,0]],dtype='<i4')
    meta={'format':FORMAT,'fields':FIELDS,'flags':FLAGS,'bond_keys':[[0,2],[0,4]],'snapshots':[]}
    for i in range(2):
        values.tofile(d/f'{i:06}.f64');flags.tofile(d/f'{i:06}.i32')
        meta['snapshots'].append({'time':i*.01,'values':f'{i:06}.f64','flags':f'{i:06}.i32'})
    (d/'manifest.json').write_text(json.dumps(meta))
    context={'joint_history':{'format':FORMAT,'path':'joint_history/manifest.json'},'bond_keys':meta['bond_keys']}
    manifest={'n_bonds':2,'n_frames':1,'dt':.01}
    return context,manifest,values,flags


def test_lossless_values_explicit_infinity_masks_and_no_future_feature_leak(tmp_path):
    c,m,v,f=recording(tmp_path);history=load_joint_history(tmp_path,c,m)
    np.testing.assert_array_equal(history.values[0].view('u8'),v.view('u8'))
    before=history.frame_start(0);assert before.shape==(2,31) and np.isfinite(before).all()
    assert before[0,21]==1 and before[1,21]==0 and before[1,4]>np.log1p(1000.)
    history.values[1]=123.;history.flags[1]=0
    np.testing.assert_array_equal(history.frame_start(0),before)
    with pytest.raises(IndexError):history.frame_start(1)
    zero=v.copy();zero[0,4]=0.
    assert not np.array_equal(finite_features(zero,f),before)
    huge=v.copy();huge[0,8]=np.finfo(np.float64).max
    assert np.isfinite(finite_features(huge,f)).all()


def test_missing_misaligned_corrupt_or_nonfinite_history_is_rejected(tmp_path):
    c,m,v,f=recording(tmp_path)
    with pytest.raises(ValueError,match='required'):load_joint_history(tmp_path,{},m)
    c['bond_keys'].reverse()
    with pytest.raises(ValueError,match='manifest'):load_joint_history(tmp_path,c,m)
    c['bond_keys'].reverse();v[0,4]=-np.inf;v.tofile(tmp_path/'joint_history/000000.f64')
    with pytest.raises(ValueError,match='positive infinite'):load_joint_history(tmp_path,c,m)
    v[0,4]=np.inf;v.tofile(tmp_path/'joint_history/000000.f64');f[0,2]=0
    f.tofile(tmp_path/'joint_history/000000.i32')
    with pytest.raises(ValueError,match='flags'):load_joint_history(tmp_path,c,m)
    (tmp_path/'joint_history/000000.i32').write_bytes(b'')
    with pytest.raises(ValueError,match='byte count'):load_joint_history(tmp_path,c,m)


def test_event_validation_and_fingerprint_include_exact_history(tmp_path):
    from learned_impact.schema import load_event
    from learned_impact.dataset import event_batches
    from learned_impact.experiment import event_digest
    fixture=Path(__file__).resolve().parents[2]/'fixtures/tiny_event'
    root=tmp_path/'event';shutil.copytree(fixture,root)
    with pytest.raises(ValueError,match='required'):load_event(root,exact_joint_history=True)
    m=json.loads((root/'manifest.json').read_text());n=m['n_bonds'];directory=root/'joint_history';directory.mkdir()
    context=json.loads((root/'context.json').read_text()) if (root/'context.json').exists() else {}
    context.update(bond_keys=[[0,i] for i in range(n)],joint_history={'format':FORMAT,'path':'joint_history/manifest.json'})
    (root/'context.json').write_text(json.dumps(context))
    v=np.zeros((n,21),dtype='<f8');v[:,4:6]=np.inf;f=np.zeros((n,4),dtype='<i4');f[:,2:]=1
    meta={'format':FORMAT,'fields':FIELDS,'flags':FLAGS,'bond_keys':context['bond_keys'],'snapshots':[]}
    for i in range(m['n_frames']+1):
        v.tofile(directory/f'{i:06}.f64');f.tofile(directory/f'{i:06}.i32')
        meta['snapshots'].append({'time':i*m['dt'],'values':f'{i:06}.f64','flags':f'{i:06}.i32'})
    (directory/'manifest.json').write_text(json.dumps(meta))
    assert load_event(root).exact_joint_history is None # Validation does not silently change model inputs.
    ev=load_event(root,exact_joint_history=True)
    assert ev.exact_joint_history.frame_start(0).shape==(n,31)
    old=list(event_batches(load_event(root)));batches=list(event_batches(ev))
    for legacy,batch in zip(old,batches):
        for key in legacy:np.testing.assert_array_equal(legacy[key],batch[key])
        ids=batch['bond_ids'].numpy()
        np.testing.assert_array_equal(batch['bond_exact_history'],finite_features(v,f)[ids])
    original=batches[0]['bond_exact_history'].clone()
    ev.exact_joint_history.values[1,:,0]=.8
    poisoned=next(event_batches(ev))
    np.testing.assert_array_equal(poisoned['bond_exact_history'],original)
    assert (poisoned['bond_exact_history_end'][:,0]>0).all()
    before=event_digest(root);v[0,4]=0.;v.tofile(directory/'000000.f64')
    assert event_digest(root)!=before # Legacy packed arrays are unchanged.
    with (directory/'000000.f64').open('ab') as file:file.write(b'!')
    with pytest.raises(ValueError,match='byte count'):load_event(root)
