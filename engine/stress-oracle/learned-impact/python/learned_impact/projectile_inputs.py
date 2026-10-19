"""Strict dynamic projectile inputs from the recorded frame-start snapshot.

Requires the enriched replay contract. Older missing spin/crush state is rejected,
not inferred from later poses, total energy, or original crush capacity.
"""
import numpy as np
from .material_inputs import material_features
from .world_inputs import array

CONTRACT='dynamic-projectile-observations/1'
PROPERTY_FIELDS=('mass','inertia_xx','inertia_xy','inertia_xz','inertia_yx','inertia_yy','inertia_yz',
                 'inertia_zx','inertia_zy','inertia_zz','crush_max_force','crush_capacity',
                 'crush_used','crush_remaining','crush_depth','has_crush','driven')


def observations(replay,scene,context,manifest):
    if replay.get('world_state_format')!='learned-world-state/1':
        raise ValueError('dynamic projectile inputs require enriched world-state replay')
    if replay.get('scene')!=scene.get('name') or replay.get('scene')!=context.get('source_scene'):
        raise ValueError('dynamic projectile source identity mismatch')
    frames=replay['frames'];count=int(manifest['n_frames'])
    if len(frames)!=count+1:raise ValueError('dynamic projectile snapshot count mismatch')
    result=[]
    # The terminal snapshot never enters observations.
    for i,frame in enumerate(frames[:-1]):
        if not np.isclose(frame['time']-frames[0]['time'],i*manifest['dt'],atol=1e-8,rtol=1e-6):
            raise ValueError('dynamic projectile snapshot time mismatch')
        result.append(snapshot_observations(frame,scene))
    return result


def snapshot_observations(frame,scene):
    """Current projectile state only; no terminal/future replay required."""
    descriptions={d['name']:d for d in scene['impactors']}
    if len(descriptions)!=len(scene['impactors']):raise ValueError('duplicate projectile names')
    objects=frame['impactors']
    if len(objects)!=len(descriptions) or {p['name'] for p in objects}!=set(descriptions):
        raise ValueError('dynamic projectile identity mismatch')
    angular=[];properties=[];materials=[]
    for p in objects:
        required=('angular_velocity','mass','inertia','material','crush','crush_used','crush_depth','driven')
        missing=[k for k in required if k not in p]
        if missing:raise ValueError(f'missing dynamic projectile fields: {missing}')
        desc=descriptions[p['name']]
        if p['shape']!=desc['shape'] or p['mass']!=desc['mass']:
            raise ValueError('dynamic projectile shape/mass differs from source scene')
        if p['material']!=scene['materials'][desc['material']]:
            raise ValueError('dynamic projectile material differs from source scene')
        inertia=np.asarray(p['inertia'],dtype=np.float64)
        if inertia.shape!=(3,3) or not np.isfinite(inertia).all() or not np.allclose(inertia,inertia.T,rtol=1e-10,atol=1e-12) or np.linalg.eigvalsh(inertia).min()<=0:
            raise ValueError('invalid dynamic projectile inertia')
        if type(p['driven']) is not bool:raise ValueError('projectile driven flag must be boolean')
        used,depth=float(p['crush_used']),float(p['crush_depth'])
        if not np.isfinite([p['mass'],used,depth]).all() or p['mass']<=0 or used<0 or depth<0:
            raise ValueError('invalid dynamic projectile mass/crush history')
        expected=desc.get('crush');crush=p['crush']
        if expected is None:
            if crush is not None:raise ValueError('dynamic projectile crush law mismatch')
            force=capacity=remaining=0.
        else:
            if crush!=[expected['max_force'],expected['energy']]:raise ValueError('dynamic projectile crush law mismatch')
            force,capacity=map(float,crush)
            if not np.isfinite([force,capacity]).all() or force<=0 or capacity<=0:raise ValueError('invalid crush capacity')
            # Oracle updates can overshoot the capacity within a substep.
            # Preserve signed remaining capacity; do not clamp the input.
            remaining=capacity-used
        angular.append(array(p['angular_velocity'],(3,),'projectile angular velocity'))
        properties.append(array([p['mass'],*inertia.flat,force,capacity,used,remaining,depth,float(crush is not None),float(p['driven'])],(17,),'dynamic projectile properties'))
        materials.append(material_features(p['material']))
    return {'impactor_angular_velocity':np.asarray(angular,dtype=np.float32).reshape(-1,3),
                   'impactor_runtime_properties':np.asarray(properties,dtype=np.float32).reshape(-1,17),
                   'impactor_material_features':np.asarray(materials,dtype=np.float32).reshape(-1,26)}
