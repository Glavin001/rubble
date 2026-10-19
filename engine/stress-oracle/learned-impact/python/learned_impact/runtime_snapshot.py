"""Read-only native inference from a current Rust World snapshot, without labels.

This bridge supports V18's frame-start contract. It does not advance contacts,
apply predicted damage, split clusters, or provide autonomous rollout accuracy.
"""
import numpy as np
import torch
from .dataset import frame_to_tensors
from .schema import STATIC_REQUIRED
from . import world_inputs, forcing_inputs, material_inputs, joint_history, response_history, projectile_inputs

FORMAT='learned-native-start/1'
WIDTHS={'chunk_inertia':3,'chunk_half_extents':3,'bond_width':2,'bond_offset':3,
        'bond_stiffness':6,'bond_strength':3,'bond_fracture_energy':3,'bond_rebar':5,'bond_endpoints':2}
INTEGER={'chunk_material_id','chunk_support','chunk_level','chunk_coarse_parent','bond_endpoints'}


def shaped(value,shape,name,dtype=np.float32):
    a=np.asarray(value)
    if a.size!=int(np.prod(shape)) or not np.isfinite(a).all():raise ValueError(f'invalid native {name}')
    if np.issubdtype(dtype,np.integer):
        bounds=np.iinfo(dtype)
        if (a!=np.floor(a)).any():raise ValueError(f'noninteger native {name}')
        if ((a<bounds.min)|(a>bounds.max)).any():raise ValueError(f'native {name} exceeds dtype range')
    with np.errstate(over='ignore'):a=a.astype(dtype).reshape(shape)
    if not np.isfinite(a).all():raise ValueError(f'native {name} exceeds dtype range')
    return a


class NativeGraph:
    def __init__(self,graph):
        if graph.get('format')!=FORMAT:raise ValueError('native graph format mismatch')
        self.scene=graph['scene'];self.context=graph['context']
        self.n=len(self.context['chunk_keys']);self.e=len(self.context['bond_keys'])
        if graph['chunk_keys']!=self.context['chunk_keys'] or graph['bond_keys']!=self.context['bond_keys']:
            raise ValueError('native graph key mismatch')
        self.dt=float(graph['dt']);self.initial_time=float(graph['initial_time'])
        if not np.isfinite([self.dt,self.initial_time]).all() or self.dt<=0:raise ValueError('invalid native time')
        self.static={}
        for k in STATIC_REQUIRED:
            size=self.e if k.startswith('bond_') else self.n
            shape=(size,WIDTHS[k]) if k in WIDTHS else (size,)
            self.static[k]=shaped(graph['statics'][k],shape,k,np.int32 if k in INTEGER else np.float32)
        manifest={'n_chunks':self.n,'n_bonds':self.e}
        self.materials=material_inputs.observations(self.scene,self.context,manifest,self.static)
        self.loads=forcing_inputs.describe(self.scene,self.context,self.n)

    def batch(self,snapshot,*,include_projectiles=False):
        if snapshot.get('format')!=FORMAT:raise ValueError('native snapshot format mismatch')
        frame=snapshot['world'];t=float(frame['time']);step=(t-self.initial_time)/self.dt
        if not np.isfinite(step) or step<0 or not np.isclose(step,round(step),atol=1e-6,rtol=0):
            raise ValueError('native snapshot must be at a frame boundary')
        e=self.e;n=self.n
        # Null has one explicitly versioned meaning; never silently turn NaN or
        # arbitrary missing state into an unset ductility value.
        objects=np.asarray(snapshot['joint_values'],dtype=object).reshape(e,21)
        missing=np.equal(objects,None)
        allowed=np.zeros((e,21),dtype=bool);allowed[:,4:6]=True
        if (missing & ~allowed).any():raise ValueError('native null outside ductility fields')
        values=np.asarray(np.where(missing,np.inf,objects),dtype=np.float64)
        flags=shaped(snapshot['joint_flags'],(e,4),'joint flags',np.int64)
        joint_history.validate_values(values,flags)
        response=shaped(snapshot['response_values'],(e,18),'response values',np.float64)
        ids=np.flatnonzero(flags[:,2]==1)
        raw={k:shaped(snapshot[k],shape,k) for k,shape in {
            'chunk_state':(n,12),'chunk_gravity_local':(n,3),
            'bond_joint_state':(e,22),'bond_force_prev':(e,6)}.items()}
        if not np.array_equal(raw['bond_force_prev'].view('u4'),response[:,:6].astype(np.float32).view('u4')):
            raise ValueError('native response and previous force mismatch')
        static={k:v[ids] if k.startswith('bond_') else v for k,v in self.static.items()}
        raw={k:v[ids] if k.startswith('bond_') else v for k,v in raw.items()}
        batch=frame_to_tensors(static,raw)
        world=world_inputs.snapshot_observations(frame,self.scene,self.context['centers'],n)
        world.update(forcing_inputs.snapshot_observations(t,self.loads,n))
        if include_projectiles:world.update(projectile_inputs.snapshot_observations(frame,self.scene))
        batch.update({k:torch.from_numpy(v.copy()) for k,v in world.items()})
        batch.update({k:torch.from_numpy((v[ids] if k.startswith('bond_') else v).copy()) for k,v in self.materials.items()})
        batch.update(bond_ids=torch.from_numpy(ids),dt=torch.tensor(self.dt,dtype=torch.float32),
            load_basis_v2=torch.tensor(1.),
            bond_frames=torch.from_numpy(shaped(self.context['bond_frames'],(e,3,3),'bond frames')[ids]),
            bond_failure_eligible=torch.from_numpy(flags[ids,3]==1),
            bond_damage_complete_start=torch.from_numpy(values[ids,0]>=1.),
            bond_exact_history=torch.from_numpy(joint_history.finite_features(values,flags)[ids]),
            bond_response_encoded=torch.from_numpy(response_history.finite_features(response)[ids]),
            bond_response_alive=torch.ones(len(ids),dtype=torch.bool))
        return batch

    def predict(self,model,snapshot):
        if model.architecture not in ('physics-response-history-native-v18','physics-projectile-response-native-v22','physics-projectile-response-si-v23'):
            raise ValueError('native snapshot bridge supports the explicit V18/V22/V23 contracts only')
        batch=self.batch(snapshot,include_projectiles=getattr(model,'requires_projectile_inputs',False))
        with torch.inference_mode():return model(model.prepare_batch(batch)),batch
