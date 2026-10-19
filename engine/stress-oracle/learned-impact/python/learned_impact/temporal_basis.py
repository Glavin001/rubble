"""Oracle-assisted lower bound on piecewise-linear stress-trajectory error.

Fits the complete future force trace by least squares, with exact oracle bond
retirement timing. This is deliberately optimistic representation capacity, not
causal prediction, learned accuracy, or a rollout.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .stress_phases import PhaseHistory

def hat_basis(times,knots):
    times=np.asarray(times,dtype=np.float64);knots=np.asarray(knots,dtype=np.float64)
    if times.ndim!=1 or knots.ndim!=1 or not len(knots) or not np.isfinite(times).all() or not np.isfinite(knots).all():raise ValueError('invalid temporal basis')
    if (np.diff(knots)<=0).any() or (np.diff(times)<0).any() or (times<knots[0]).any() or (times>knots[-1]).any():raise ValueError('temporal basis outside knot domain')
    if len(knots)==1:return np.ones((len(times),1))
    return np.column_stack([np.interp(times,knots,np.eye(len(knots))[i]) for i in range(len(knots))])

def fit_prefix(values,times,knot_count,*,frame_end=None):
    """One cohort sharing its oracle-known valid prefix, shape [steps,columns]."""
    values=np.asarray(values,dtype=np.float64);times=np.asarray(times,dtype=np.float64)
    length=len(values)
    if not 1<=length<=len(times) or values.ndim!=2 or knot_count<2:raise ValueError('invalid prefix fit')
    if frame_end is not None:
        basis=hat_basis(times[:length],np.linspace(times[0],frame_end,knot_count))
        return basis@np.linalg.lstsq(basis,values,rcond=None)[0],knot_count
    # Fixed full-frame knots plus the exact last alive native phase. Giving this
    # oracle boundary makes the measured error an optimistic capacity diagnostic.
    indices=np.unique(np.rint(np.linspace(0,len(times)-1,min(knot_count,len(times)))).astype(int))
    indices=np.unique(np.r_[indices[indices<length],length-1])
    basis=hat_basis(times[:length],times[indices])
    weights=np.linalg.lstsq(basis,values,rcond=None)[0]
    return basis@weights,len(indices)

def audit(root,counts=(2,4,8,16,32,64),*,fixed_grid=False):
    data=PhaseHistory(root);rows=[]
    for frame,indices in enumerate(data.frame_phases):
        e=len(data.meta['bond_keys']);steps=len(indices)
        values=np.zeros((steps,e,6));alive=np.zeros((steps,e),dtype=bool)
        for t,index in enumerate(indices):
            sl=data.phase_rows(index);ids=data.slots[sl];values[t,ids]=data.targets[sl,21:27];alive[t,ids]=True
        times=np.array([data.phases[i]['time'] for i in indices]);lengths=alive.sum(0)
        if not np.array_equal(alive,np.arange(steps)[:,None]<lengths[None,:]):raise ValueError('bond participation is not a frame-start prefix')
        start=np.array(data.snapshot(frame)['response_values'])[:,:6]
        truth_squared=[float(np.square(values[:,:,:3][alive]).sum()),float(np.square(values[:,:,3:][alive]).sum())]
        held=values-start[None,:,:]
        hold_squared=[float(np.square(held[:,:,:3][alive]).sum()),float(np.square(held[:,:,3:][alive]).sum())]
        for count in counts:
            predicted=np.zeros_like(values);max_knots=0
            for length in np.unique(lengths[lengths>0]):
                ids=np.flatnonzero(lengths==length);actual=values[:length,ids].reshape(length,-1)
                fitted,n=fit_prefix(actual,times,count,frame_end=times[0]+data.graph['dt'] if fixed_grid else None)
                predicted[:length,ids]=fitted.reshape(length,len(ids),6);max_knots=max(max_knots,n)
            error=predicted-values
            error_squared=[float(np.square(error[:,:,:3][alive]).sum()),float(np.square(error[:,:,3:][alive]).sum())]
            rows.append({'frame':frame,'native_steps':steps,'requested_knots':count,'maximum_actual_knots':max_knots,
                'bond_phase_rows':int(alive.sum()),'truth_squared':truth_squared,'hold_previous_squared':hold_squared,'fit_squared':error_squared})
    summary=[]
    for count in counts:
        selected=[r for r in rows if r['requested_knots']==count];n=sum(r['bond_phase_rows'] for r in selected)
        truth=np.sum([r['truth_squared'] for r in selected],axis=0);error=np.sum([r['fit_squared'] for r in selected],axis=0);hold=np.sum([r['hold_previous_squared'] for r in selected],axis=0)
        summary.append({'knots':count,'force_rmse_n':float(np.sqrt(error[0]/max(3*n,1))),
            'torque_rmse_nm':float(np.sqrt(error[1]/max(3*n,1))),
            'force_relative_l2':float(np.sqrt(error[0]/truth[0])) if truth[0]>0 else 0.,
            'torque_relative_l2':float(np.sqrt(error[1]/truth[1])) if truth[1]>0 else 0.,
            'hold_force_relative_l2':float(np.sqrt(hold[0]/truth[0])) if truth[0]>0 else 0.,
            'hold_torque_relative_l2':float(np.sqrt(hold[1]/truth[1])) if truth[1]>0 else 0.})
    return {'event':Path(root).name,'scope':__doc__,'fixed_frame_grid_without_retirement_knot':fixed_grid,'frames':data.frames,'summary':summary,'rows':rows}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result=audit(args.root);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['summary']))

if __name__=='__main__':main()
