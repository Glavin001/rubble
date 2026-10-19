"""Exact cached total/elastic force, stored energy and stress measures.

Snapshots retain retired bonds' last cached values. Alive flags distinguish
continuing interfaces; these values alone are not an engine restart checkpoint.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .joint_history import load_joint_history

FORMAT='learned-bond-response/1'
FIELDS=['force_x','force_y','force_z','moment_x','moment_y','moment_z',
        'elastic_force_x','elastic_force_y','elastic_force_z','elastic_moment_x','elastic_moment_y','elastic_moment_z',
        'stored_energy','tension','shear','normal_compression','compression','compressive_force']
FLAGS=['alive','connected']
SCALE=np.array([1000.]*3+[100.]*3+[1000.]*3+[100.]*3+[1000.]+[1e6]*4+[1000.])


def finite_features(values):
    if values.ndim!=2 or values.shape[1]!=18 or not np.isfinite(values).all():
        raise ValueError('invalid cached bond response')
    with np.errstate(divide='ignore'):
        return (np.sign(values)*np.logaddexp(0,np.log(np.abs(values))-np.log(SCALE))).astype(np.float32)


@dataclass
class ResponseHistory:
    values:np.ndarray
    flags:np.ndarray
    times:np.ndarray

    def frame_start(self,index):
        if not 0<=index<len(self.times)-1:raise IndexError('frame-start index outside recorded intervals')
        return finite_features(self.values[index])


def load_response_history(root,context,manifest,joint=None,frames=None):
    contract=context.get('bond_response')
    if not contract or contract.get('format')!=FORMAT or contract.get('path')!='bond_response/manifest.json':
        raise ValueError('lossless bond-response sidecar required; elastic force/energy cannot be inferred from total force')
    directory=Path(root)/'bond_response';meta=json.loads((directory/'manifest.json').read_text())
    n=int(manifest['n_bonds']);count=int(manifest['n_frames'])+1;dt=float(manifest['dt']);keys=context.get('bond_keys')
    if keys is None or len(keys)!=n or len({tuple(k) for k in keys})!=n:raise ValueError('unique aligned response bond keys required')
    if meta.get('format')!=FORMAT or meta.get('fields')!=FIELDS or meta.get('flags')!=FLAGS or meta.get('bond_keys')!=keys or len(meta['snapshots'])!=count:
        raise ValueError('bond-response manifest mismatch')
    times=np.array([s['time'] for s in meta['snapshots']],dtype=np.float64)
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(times).all() or not np.allclose(times-times[0],np.arange(count)*dt,atol=1e-8,rtol=1e-6):
        raise ValueError('bond-response time alignment mismatch')
    values=[];flags=[]
    for i,s in enumerate(meta['snapshots']):
        if s['values']!=f'{i:06}.f64' or s['flags']!=f'{i:06}.i32':raise ValueError('bond-response snapshot path mismatch')
        if (directory/s['values']).stat().st_size!=n*18*8 or (directory/s['flags']).stat().st_size!=n*2*4:
            raise ValueError('bond-response byte count mismatch')
        v=np.fromfile(directory/s['values'],dtype='<f8').reshape(n,18);f=np.fromfile(directory/s['flags'],dtype='<i4').reshape(n,2)
        if not np.isfinite(v).all() or not np.isin(f,[0,1]).all() or (f[:,1]>f[:,0]).any():raise ValueError('invalid cached bond response')
        values.append(v);flags.append(f)
    values=np.stack(values);flags=np.stack(flags)
    if joint is None:joint=load_joint_history(root,context,manifest)
    if not np.array_equal(times,joint.times) or not np.array_equal(flags,joint.flags[:,:,2:4]):raise ValueError('response/joint snapshot mismatch')
    if frames is not None:
        for i,frame in enumerate(frames):
            for snapshot,key in [(i,'bond_force_prev'),(i+1,'bond_force_end')]:
                if not np.array_equal(values[snapshot,:,:6].astype(np.float32).view('u4'),frame[key].view('u4')):
                    raise ValueError('cached response/legacy force mismatch')
    return ResponseHistory(values,flags,times)
