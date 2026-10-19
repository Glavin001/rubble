"""Exact prescribed-load descriptors, not oracle forces or a pressure solver.

No future replay states are read. All table knots and all loads are retained.
Point-force offsets and pressure normals use original body coordinates; blast
positions use world coordinates. The encoder receives explicit load kinds.
Structural removal events are unsupported and rejected, never silently omitted.
"""
from __future__ import annotations
import numpy as np

CONTRACT='prescribed-forcing-descriptors/1'
CURVES={'constant':0,'ramp':1,'half_sine':2,'friedlander':3,'table':4}


def finite(value,shape,name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'invalid forcing {name}')
    return a


def normal(value):
    a=finite(value,(3,),'direction');length=np.linalg.norm(a)
    if length<=0:raise ValueError('forcing direction must be nonzero')
    return a/length


def selection(value,body):
    chunks=body['chunks']
    if value=='all':return set(range(len(chunks)))
    if not isinstance(value,dict) or len(value)!=1:raise ValueError('invalid forcing selector')
    kind,arg=next(iter(value.items()))
    if kind=='group':return {i for i,c in enumerate(chunks) if arg in c.get('groups',[])}
    if kind=='indices':
        if any(type(i)!=int or i<0 or i>=len(chunks) for i in arg):raise ValueError('invalid forcing chunk index')
        return set(arg)
    if kind=='region':
        lo=finite(arg['min'],(3,),'region min');hi=finite(arg['max'],(3,),'region max')
        if (lo>hi).any():raise ValueError('invalid forcing region bounds')
        return {i for i,c in enumerate(chunks) if ((np.asarray(c['center'])>=lo)&(np.asarray(c['center'])<=hi)).all()}
    raise ValueError(f'unsupported forcing selector {kind}')


def curve(desc,time):
    kind=desc['type']
    if kind not in CURVES:raise ValueError(f'unsupported forcing time function {kind}')
    points=[];params=[0.,0.,0.,0.]
    if kind=='constant':params[2]=desc['value']
    elif kind=='ramp':
        if desc['t1']<=desc['t0']:raise ValueError('forcing ramp requires t1 > t0')
        params=[desc['t0']-time,desc['t1']-time,desc['value'],0.]
    elif kind in ('half_sine','friedlander'):
        if desc['duration']<=0:raise ValueError('forcing duration must be positive')
        params=[desc['start' if kind=='half_sine' else 'arrival']-time,desc['duration'],desc['peak'],desc.get('decay',0.)]
    else:
        points=np.asarray(desc['points'],dtype=np.float64).reshape(-1,2)
        if not np.isfinite(points).all() or (len(points)>1 and (np.diff(points[:,0])<=0).any()):raise ValueError('forcing table times must increase and values must be finite')
        points=points.copy();points[:,0]-=time
    return CURVES[kind],finite(params,(4,),'curve parameters'),points


def describe(scene,context,n):
    for key in ('bodies','loads','events'):
        if key not in scene:raise ValueError(f'forcing descriptors require scene {key}')
    if scene['events']:raise ValueError('forcing descriptors do not support structural removal events')
    keys=context.get('chunk_keys')
    if keys is None or len(keys)!=n or len({tuple(k) for k in keys})!=n:raise ValueError('forcing requires unique aligned chunk_keys')
    bodies=scene['bodies'];names={b['name']:i for i,b in enumerate(bodies)}
    if len(names)!=len(bodies):raise ValueError('ambiguous forcing body names')
    for key in keys:
        if len(key)!=2 or any(type(i)!=int for i in key) or not 0<=key[0]<len(bodies) or not 0<=key[1]<len(bodies[key[0]]['chunks']):raise ValueError('invalid forcing chunk key')
    descriptions=[]
    for load in scene['loads']:
        kind=load['type'];direction=np.zeros(3);position=np.zeros(3);mass=start=0.;wave=None
        if kind=='blast':
            position=finite(load['position'],(3,),'blast position');mass=float(load['tnt_mass']);start=float(load['time']);selected=np.ones(n,dtype=bool)
            if mass<=0 or not np.isfinite([mass,start]).all():raise ValueError('invalid blast mass/time')
        elif kind in ('point_force','pressure'):
            if load['body'] not in names:raise ValueError('unknown forcing body')
            body=names[load['body']];desc=bodies[body]
            if kind=='point_force':
                indices=selection({'indices':[load['chunk']]},desc);direction=normal(load['direction']);wave=load['magnitude']
                center=finite(desc['chunks'][load['chunk']]['center'],(3,),'chunk center')
                position=finite(load['point'],(3,),'point')-center if load.get('point') is not None else np.zeros(3)
            else:
                indices=selection(load['chunks'],desc);direction=normal(load['face_normal']);wave=load['pressure']
            selected=np.array([s==body and c in indices for s,c in keys])
            # These catalogs do not refine or omit selected chunks. Reject an
            # incomplete record instead of inventing missing forces.
            if len(indices)!=int(selected.sum()):raise ValueError('selected forcing chunks are missing from recording')
        else:raise ValueError(f'unsupported prescribed load {kind}')
        descriptions.append(({'point_force':0,'pressure':1,'blast':2}[kind],direction,position,mass,start,wave,selected))
    return descriptions


def snapshot_observations(time, descriptions, n):
    """Known load schedules at the current time; no replay or endpoint needed."""
    time=float(time)
    if not np.isfinite(time):raise ValueError('invalid forcing frame time')
    kinds=[];dirs=[];positions=[];curve_kinds=[];parameters=[];blast=[];selected=[];knots=[];owners=[]
    for i,(kind,direction,position,mass,start,wave,mask) in enumerate(descriptions):
        ck,cp,points=curve(wave,time) if wave is not None else (5,np.zeros(4),[])
        kinds.append(kind);dirs.append(direction);positions.append(position);curve_kinds.append(ck);parameters.append(cp);blast.append([mass,start-time] if kind==2 else [0.,0.]);selected.append(mask)
        knots.extend(points);owners.extend([i]*len(points))
    l=len(kinds)
    values={'forcing_kind':np.asarray(kinds,dtype=np.int64),'forcing_direction':np.asarray(dirs,dtype=np.float64).reshape(l,3),
            'forcing_position':np.asarray(positions,dtype=np.float64).reshape(l,3),'forcing_curve_kind':np.asarray(curve_kinds,dtype=np.int64),
            'forcing_curve_parameters':np.asarray(parameters,dtype=np.float64).reshape(l,4),'forcing_blast':np.asarray(blast,dtype=np.float64).reshape(l,2),
            'forcing_selected':np.asarray(selected,dtype=bool).reshape(l,n),'forcing_table_points':np.asarray(knots,dtype=np.float64).reshape(-1,2),
            'forcing_table_owner':np.asarray(owners,dtype=np.int64)}
    for key,a in values.items():
        if a.dtype==np.float64:
            with np.errstate(over='ignore'):values[key]=a.astype(np.float32)
            if not np.isfinite(values[key]).all():raise ValueError('forcing descriptor exceeds float32 range')
    return values


def observations(replay,scene,context,manifest):
    n=int(manifest['n_chunks']);count=int(manifest['n_frames'])
    descriptions=describe(scene,context,n)
    if len(replay['frames'])!=count+1:raise ValueError('forcing snapshot count mismatch')
    return [snapshot_observations(frame['time'],descriptions,n) for frame in replay['frames'][:count]]
