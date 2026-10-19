"""World-space supervision that survives fragment coordinate resets.

Inputs remain snapshot i only. These labels use snapshot i+1 explicitly and must
never enter an encoder. Residuals are relative to constant world linear/angular
velocity: x_base=x0+dt*v0, R_base=Exp(dt*w0)@R0. Rotation residual is the principal
world-axis log of R1@R_base.T; composing it on the LEFT reconstructs R1.
"""
from __future__ import annotations
import numpy as np

CONTRACT='world-ballistic-residual/1'


def rotation_exp(v):
    v=np.asarray(v,dtype=np.float64)
    if v.shape[-1:]!=(3,) or not np.isfinite(v).all():raise ValueError('invalid rotation vector')
    k=np.zeros((*v.shape[:-1],3,3));x,y,z=np.moveaxis(v,-1,0)
    k[...,0,1]=-z;k[...,0,2]=y;k[...,1,0]=z;k[...,1,2]=-x;k[...,2,0]=-y;k[...,2,1]=x
    angle=np.linalg.norm(v,axis=-1)
    return np.eye(3)+np.sinc(angle/np.pi)[...,None,None]*k+(.5*np.sinc(angle/(2*np.pi))**2)[...,None,None]*(k@k)


def rotation_log(matrix):
    r=np.asarray(matrix,dtype=np.float64)
    if r.shape[-2:]!=(3,3) or not np.isfinite(r).all():raise ValueError('invalid rotation matrix')
    if not np.allclose(r@np.swapaxes(r,-1,-2),np.eye(3),atol=2e-5) or not np.allclose(np.linalg.det(r),1,atol=2e-5):
        raise ValueError('rotation matrix must be proper orthogonal')
    shape=r.shape[:-2];m=r.reshape(-1,3,3)
    # Largest quaternion component stays >= .5, including rotations near pi.
    candidates=np.stack((1+np.trace(m,axis1=1,axis2=2),1+m[:,0,0]-m[:,1,1]-m[:,2,2],
                         1-m[:,0,0]+m[:,1,1]-m[:,2,2],1-m[:,0,0]-m[:,1,1]+m[:,2,2]),axis=1)
    which=candidates.argmax(1);q=np.zeros((len(m),4))
    for c in range(4):
        ids=np.flatnonzero(which==c);a=m[ids];major=.5*np.sqrt(candidates[ids,c].clip(0));den=4*major
        if c==0:values=(major,(a[:,2,1]-a[:,1,2])/den,(a[:,0,2]-a[:,2,0])/den,(a[:,1,0]-a[:,0,1])/den)
        elif c==1:values=((a[:,2,1]-a[:,1,2])/den,major,(a[:,0,1]+a[:,1,0])/den,(a[:,0,2]+a[:,2,0])/den)
        elif c==2:values=((a[:,0,2]-a[:,2,0])/den,(a[:,0,1]+a[:,1,0])/den,major,(a[:,1,2]+a[:,2,1])/den)
        else:values=((a[:,1,0]-a[:,0,1])/den,(a[:,0,2]+a[:,2,0])/den,(a[:,1,2]+a[:,2,1])/den,major)
        q[ids]=np.stack(values,axis=1)
    q/=np.linalg.norm(q,axis=1,keepdims=True);q[q[:,0]<0]*=-1
    norm=np.linalg.norm(q[:,1:],axis=1);factor=np.full_like(norm,2.)
    np.divide(2*np.arctan2(norm,q[:,0]),norm,out=factor,where=norm>1e-12)
    return (q[:,1:]*factor[:,None]).reshape(*shape,3)


def _array(frame,key,n,tail):
    a=np.asarray(frame[key],dtype=np.float64)
    if a.shape!=(n,*tail) or not np.isfinite(a).all():raise ValueError(f'invalid world target {key}')
    return a


def targets(replay,manifest):
    if replay.get('format')!='learned-impact-replay/1':raise ValueError('world targets require replay format 1')
    n=int(manifest['n_chunks']);count=int(manifest['n_frames']);dt=float(manifest['dt']);frames=replay['frames']
    if len(frames)!=count+1 or not np.isfinite(dt) or dt<=0:raise ValueError('invalid world target frame count or dt')
    states=[]
    for i,frame in enumerate(frames):
        if not np.isclose(frame['time']-frames[0]['time'],i*dt,atol=1e-8,rtol=1e-6):raise ValueError('world target timestamp mismatch')
        state={key:_array(frame,key,n,tail) for key,tail in (('centers',(3,)),('velocities',(3,)),('rotations',(3,3)),('angular_velocities',(3,)),('active',()))}
        rotation_log(state['rotations']) # Validate final pose as well as start inputs.
        if not np.isin(state['active'],[0,1]).all():raise ValueError('invalid world target active flag')
        states.append(state)
    out=[]
    for start,end in zip(states,states[1:]):
        baseline=rotation_exp(start['angular_velocities']*dt)@start['rotations']
        residual=np.concatenate((end['centers']-start['centers']-dt*start['velocities'],
                                 rotation_log(end['rotations']@np.swapaxes(baseline,-1,-2)),
                                 end['velocities']-start['velocities'],end['angular_velocities']-start['angular_velocities']),axis=1)
        with np.errstate(over='ignore'):
            encoded=residual.astype(np.float32)
        if not np.isfinite(encoded).all():raise ValueError('world motion residual exceeds finite float32 range')
        out.append({'world_motion_target':encoded,
                    'world_motion_valid':((start['active']>0)&(end['active']>0))})
    return out


def reconstruct(start,residual,dt):
    """Reference decoder for inspection/tests, not an engine integrator."""
    r=np.asarray(residual,dtype=np.float64)
    return {'centers':np.asarray(start['centers'])+dt*np.asarray(start['velocities'])+r[:,:3],
            'rotations':rotation_exp(r[:,3:6])@rotation_exp(dt*np.asarray(start['angular_velocities']))@np.asarray(start['rotations']),
            'velocities':np.asarray(start['velocities'])+r[:,6:9],
            'angular_velocities':np.asarray(start['angular_velocities'])+r[:,9:12]}
