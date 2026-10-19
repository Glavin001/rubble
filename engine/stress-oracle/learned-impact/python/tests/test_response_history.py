import json
import numpy as np
import pytest
from learned_impact.response_history import FORMAT,FIELDS,FLAGS,load_response_history,finite_features
from test_joint_history import recording as joint_recording


def recording(root):
    context,manifest,_,joint_flags=joint_recording(root)
    d=root/'bond_response';d.mkdir()
    context['bond_response']={'format':FORMAT,'path':'bond_response/manifest.json'}
    values=np.arange(36,dtype='<f8').reshape(2,18)+.123456789012345;values[0,0]=-0.
    flags=joint_flags[:,2:].copy();meta={'format':FORMAT,'fields':FIELDS,'flags':FLAGS,'bond_keys':context['bond_keys'],'snapshots':[]}
    for i in range(2):
        (values+i).tofile(d/f'{i:06}.f64');flags.tofile(d/f'{i:06}.i32')
        meta['snapshots'].append({'time':i*.01,'values':f'{i:06}.f64','flags':f'{i:06}.i32'})
    values.tofile(d/'000000.f64') # Preserve the signed zero sentinel.
    (d/'manifest.json').write_text(json.dumps(meta))
    frames=[{'bond_force_prev':values[:,:6].astype(np.float32),'bond_force_end':(values+1)[:,:6].astype(np.float32)}]
    return context,manifest,values,frames


def test_response_lossless_alignment_and_causal_inputs(tmp_path):
    c,m,v,frames=recording(tmp_path);h=load_response_history(tmp_path,c,m,frames=frames)
    np.testing.assert_array_equal(h.values[0].view('u8'),v.view('u8'))
    assert not np.array_equal(h.values[0,:,:6],h.values[0,:,6:12]) # Total and elastic remain distinct.
    before=h.frame_start(0);h.values[1]=123.;h.flags[1]=0
    np.testing.assert_array_equal(h.frame_start(0),before)
    with pytest.raises(IndexError):h.frame_start(1)
    extreme=v.copy();extreme[0,7]=np.finfo(np.float64).max
    assert np.isfinite(finite_features(extreme)).all()
    frames[0]['bond_force_end'][0,0]+=1
    with pytest.raises(ValueError,match='legacy force'):load_response_history(tmp_path,c,m,frames=frames)


def test_response_rejects_missing_misaligned_corrupt_and_nonfinite_data(tmp_path):
    c,m,v,frames=recording(tmp_path)
    with pytest.raises(ValueError,match='required'):load_response_history(tmp_path,{},m)
    c['bond_keys'].reverse()
    with pytest.raises(ValueError,match='manifest'):load_response_history(tmp_path,c,m)
    c['bond_keys'].reverse()
    f=np.ones((2,2),dtype='<i4');f.tofile(tmp_path/'bond_response/000001.i32')
    with pytest.raises(ValueError,match='response/joint'):load_response_history(tmp_path,c,m)
    f[1,1]=0;f.tofile(tmp_path/'bond_response/000001.i32')
    v[0,12]=np.inf;v.tofile(tmp_path/'bond_response/000001.f64')
    with pytest.raises(ValueError,match='invalid cached'):load_response_history(tmp_path,c,m)
    (tmp_path/'bond_response/000001.f64').write_bytes(b'')
    with pytest.raises(ValueError,match='byte count'):load_response_history(tmp_path,c,m)
