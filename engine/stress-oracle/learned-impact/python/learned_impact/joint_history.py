"""Lossless joint history plus finite, explicit frame-start network features.

This optional sidecar cannot be reconstructed from the legacy clamped 22-vector.
It is joint history only, not a complete World checkpoint or rollout adapter.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

FORMAT='learned-joint-history/1'
FIELDS=['damage','crush','kappa','kappa_c','ductility','ductility_c','fatigue',
        'plastic_x','plastic_y','plastic_z','plastic_rx','plastic_ry','plastic_rz','rebar_plastic',
        'rebar_slip_x','rebar_slip_y','rebar_work','strain_rate','governing_stress','dissipated','utilization']
FLAGS=['rebar_broken','mode','alive','connected']
SCALE=np.array([1.]*7+[.001]*9+[1000.,1.,1e6,1000.,1.])


def validate_values(values,flags):
    if values.ndim!=2 or values.shape[1]!=21 or flags.shape!=(len(values),4):raise ValueError('joint-history shape mismatch')
    valid=np.isfinite(values);valid[:,4:6]|=np.isposinf(values[:,4:6])
    if not valid.all():raise ValueError('invalid joint-history float; only positive infinite ductility is allowed')
    if not np.isin(flags[:,[0,2,3]],[0,1]).all() or not np.isin(flags[:,1],np.arange(5)).all() or (flags[:,3]>flags[:,2]).any():
        raise ValueError('invalid joint-history flags')


def finite_features(values,flags):
    """21 unclamped signed-log values, two unset masks, rebar flag, mode, topology."""
    validate_values(values,flags)
    unset=np.isposinf(values[:,4:6]);finite=np.where(np.isfinite(values),values,0.)
    # Stable even for finite values near float64's limit; no amplitude clipping.
    with np.errstate(divide='ignore'):
        encoded=np.sign(finite)*np.logaddexp(0,np.log(np.abs(finite))-np.log(SCALE))
    result=np.concatenate([encoded,unset,flags[:,0:1],np.eye(5)[flags[:,1]],flags[:,2:4]],axis=1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite joint-history feature')
    return result


@dataclass
class JointHistory:
    values:np.ndarray
    flags:np.ndarray
    times:np.ndarray

    def frame_start(self,index):
        if not 0<=index<len(self.times)-1:raise IndexError('frame-start index outside recorded intervals')
        return finite_features(self.values[index],self.flags[index])


def load_joint_history(root,context,manifest):
    contract=context.get('joint_history')
    if not contract or contract.get('format')!=FORMAT or contract.get('path')!='joint_history/manifest.json':
        raise ValueError('lossless joint-history sidecar required; legacy packed values are insufficient')
    directory=Path(root)/'joint_history';meta=json.loads((directory/'manifest.json').read_text())
    n=int(manifest['n_bonds']);count=int(manifest['n_frames'])+1;dt=float(manifest['dt'])
    keys=context.get('bond_keys')
    if keys is None or len(keys)!=n or len({tuple(k) for k in keys})!=n:raise ValueError('unique aligned joint-history bond keys required')
    if meta.get('format')!=FORMAT or meta.get('fields')!=FIELDS or meta.get('flags')!=FLAGS or meta.get('bond_keys')!=keys or len(meta['snapshots'])!=count:
        raise ValueError('joint-history manifest mismatch')
    times=np.array([s['time'] for s in meta['snapshots']],dtype=np.float64)
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(times).all() or not np.allclose(times-times[0],np.arange(count)*dt,atol=1e-8,rtol=1e-6):
        raise ValueError('joint-history time alignment mismatch')
    values=[];flags=[]
    for i,s in enumerate(meta['snapshots']):
        if s['values']!=f'{i:06}.f64' or s['flags']!=f'{i:06}.i32':raise ValueError('joint-history snapshot path mismatch')
        if (directory/s['values']).stat().st_size!=n*21*8 or (directory/s['flags']).stat().st_size!=n*4*4:raise ValueError('joint-history byte count mismatch')
        v=np.fromfile(directory/s['values'],dtype='<f8');f=np.fromfile(directory/s['flags'],dtype='<i4')
        if v.size!=n*21 or f.size!=n*4:raise ValueError('joint-history byte count mismatch')
        v=v.reshape(n,21);f=f.reshape(n,4);validate_values(v,f);values.append(v);flags.append(f)
    return JointHistory(np.stack(values),np.stack(flags),times)
