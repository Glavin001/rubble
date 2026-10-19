"""Strict frame-start world inputs derived from recorded replay and scene metadata.

Snapshot i is the input to interval i. Snapshot i+1, mechanical/dissipated energy,
and interval-end labels are never feature inputs. Projectile angular velocity and
remaining crush energy are not present in historical replay and are not invented.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

CONTRACT = 'world-start-observations/1'


def array(value, shape, name):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'invalid world-start {name}')
    return a


def rotation(value, n, name):
    a=array(value,(n,3,3),name)
    if not np.allclose(a @ np.swapaxes(a,-1,-2),np.eye(3),atol=2e-5) or not np.allclose(np.linalg.det(a),1,atol=2e-5):
        raise ValueError(f'invalid world-start rotation {name}')
    return a


def observations(replay, scene, context, manifest):
    if replay.get('format')!='learned-impact-replay/1':raise ValueError('world-start inputs require replay format 1')
    if replay.get('scene')!=scene.get('name') or replay.get('scene')!=context.get('source_scene'):
        raise ValueError('world-start scene identity mismatch')
    n=int(manifest['n_chunks']);frames=replay['frames'];count=int(manifest['n_frames'])
    if len(frames)!=count+1:raise ValueError('world-start requires initial plus every end snapshot')
    initial=array(frames[0]['centers'],(n,3),'initial centers')
    if not np.allclose(initial,array(context['centers'],(n,3),'context centers'),atol=1e-6):
        raise ValueError('world-start initial geometry mismatch')
    result=[]
    for i,frame in enumerate(frames):
        if not np.isclose(frame['time']-frames[0]['time'],i*manifest['dt'],atol=1e-8,rtol=1e-6):
            raise ValueError('world-start snapshot time mismatch')
        # Last snapshot is only a time/length check, never an input.
        if i==count:break
        result.append(snapshot_observations(frame,scene,initial,n))
    return result


def snapshot_observations(frame, scene, initial, n):
    """One current snapshot, without requiring or fabricating a future state."""
    initial=array(initial,(n,3),'initial centers')
    descriptions={d['name']:d for d in scene['impactors']}
    if len(descriptions)!=len(scene['impactors']):raise ValueError('ambiguous projectile names')
    ground=scene.get('ground');height=float(ground['height']) if ground else 0.
    if not np.isfinite(height):raise ValueError('invalid ground height')
    center=array(frame['centers'],(n,3),'centers')
    active=array(frame['active'],(n,),'active')
    if not np.isin(active,[0,1]).all():raise ValueError('invalid world-start active flag')
    objects=frame['impactors']
    if len({p['name'] for p in objects})!=len(objects) or {p['name'] for p in objects}!=set(descriptions):
        raise ValueError('world-start projectile identity mismatch')
    centers=[];velocities=[];rotations=[];properties=[]
    for p in objects:
        d=descriptions[p['name']]
        if p['shape']!=d['shape']:raise ValueError('world-start projectile shape mismatch')
        shape=d['shape'];kind=shape['type']
        if kind=='sphere':half=[shape['radius']]*3
        elif kind=='box':half=shape['half_extents']
        else:raise ValueError(f'unsupported recorded projectile shape {kind}')
        crush=d.get('crush')
        props=[*half,d['mass'],float(kind=='box'),float(kind=='sphere'),
               crush['max_force'] if crush else 0.,crush['energy'] if crush else 0.,float(crush is not None)]
        props=array(props,(9,),'projectile properties')
        if np.any(props[:4]<=0):raise ValueError('invalid projectile mass or extents')
        centers.append(array(p['center'],(3,),'projectile center'))
        velocities.append(array(p['velocity'],(3,),'projectile velocity'))
        rotations.append(rotation([p['rotation']],1,'projectile rotation')[0])
        properties.append(props)
    return {'world_center':center,'world_displacement':center-initial,
                   'world_velocity':array(frame['velocities'],(n,3),'velocity'),
                   'world_angular_velocity':array(frame['angular_velocities'],(n,3),'angular velocity'),
                   'world_rotation':rotation(frame['rotations'],n,'chunk rotation'),
                   'world_active':active,'world_ground':np.array([height,float(ground is not None)],dtype=np.float32),
                   'impactor_center':np.asarray(centers,dtype=np.float32).reshape(-1,3),
                   'impactor_velocity':np.asarray(velocities,dtype=np.float32).reshape(-1,3),
                   'impactor_rotation':np.asarray(rotations,dtype=np.float32).reshape(-1,3,3),
                   'impactor_properties':np.asarray(properties,dtype=np.float32).reshape(-1,9)}


def load_world_inputs(root, context, manifest, *, include_targets=False, include_forcing=False, include_projectiles=False):
    root=Path(root)
    paths=[root/'replay.json',root/'scene.json']
    if not all(p.is_file() for p in paths):raise ValueError('world-start model requires replay.json and scene.json')
    blobs=[p.read_bytes() for p in paths]
    digest=hashlib.sha256()
    for p,b in zip(paths,blobs):digest.update(p.name.encode());digest.update(b)
    replay=json.loads(blobs[0])
    labels=[]
    if include_targets:
        from .world_targets import targets
        labels=targets(replay,manifest)
    scene=json.loads(blobs[1]);inputs=observations(replay,scene,context,manifest)
    if include_forcing:
        from .forcing_inputs import observations as forcing_observations
        for a,b in zip(inputs,forcing_observations(replay,scene,context,manifest)):a.update(b)
    if include_projectiles:
        from .projectile_inputs import observations as projectile_observations
        for a,b in zip(inputs,projectile_observations(replay,scene,context,manifest)):a.update(b)
    return inputs,digest.hexdigest(),[f['clusters'] for f in replay['frames']],labels
