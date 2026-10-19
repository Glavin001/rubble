"""Physical motion errors separated by training-scale target magnitude.

Masks describe targets only. They never modify training or model predictions.
A vector RMS threshold is sqrt(sum(axis RMS**2)), from saved training buffers.
"""
import numpy as np

FIELDS=('u_m','theta_rad','v_mps','omega_radps')


class MotionMetrics:
    def __init__(self, axis_rms):
        scales=np.asarray(axis_rms,dtype=np.float64)
        if scales.shape!=(12,) or not np.isfinite(scales).all() or np.any(scales<=0):
            raise ValueError('motion metrics require twelve positive finite training RMS scales')
        self.thresholds=np.sqrt(np.square(scales.reshape(4,3)).sum(1))
        self.sums=np.zeros((4,2,3),dtype=np.float64) # count, squared error, squared zero baseline
        self.excluded=0

    def add(self, target, predicted, valid):
        target=np.asarray(target,dtype=np.float64);predicted=np.asarray(predicted,dtype=np.float64)
        valid=np.asarray(valid,dtype=bool)
        if target.ndim!=2 or target.shape[1]!=12 or predicted.shape!=target.shape or valid.shape!=(len(target),):
            raise ValueError('invalid motion metric shapes')
        if not np.isfinite(target).all() or not np.isfinite(predicted).all():raise ValueError('nonfinite motion metric input')
        self.excluded+=int((~valid).sum())
        actual=target[valid].reshape(-1,4,3);error=predicted[valid].reshape(-1,4,3)-actual
        norm=np.linalg.norm(actual,axis=2)
        for g in range(4):
            above=norm[:,g]>self.thresholds[g]
            for bucket,mask in enumerate((~above,above)):
                self.sums[g,bucket]+=[int(mask.sum()),np.square(error[mask,g]).sum(),np.square(actual[mask,g]).sum()]

    def report(self):
        result={}
        for g,name in enumerate(FIELDS):
            result[name]={'target_vector_norm_threshold':float(self.thresholds[g])}
            for bucket,label in enumerate(('at_or_below_training_rms','above_training_rms')):
                count,error,zero=self.sums[g,bucket]
                result[name][label]={'chunk_intervals':int(count),'rmse':float(np.sqrt(error/(3*count))) if count else None,
                                    'zero_increment_rmse':float(np.sqrt(zero/(3*count))) if count else None}
        return {'fields':result,'fragment_reset_excluded_chunk_intervals':self.excluded,
                'scope':'Per-axis physical RMSE; target vector magnitude compared with saved training RMS. Oracle start states; not rollout accuracy.'}
