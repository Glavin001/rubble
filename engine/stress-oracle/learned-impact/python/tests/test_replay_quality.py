import copy
import numpy as np
import pytest
from learned_impact.replay_quality import inspect_replay


def replay():
    return {'format':'learned-impact-replay/1','frames':[
        {'time':0.,'centers':[[0,0,0],[0,0,0]],'velocities':[[2,0,0],[999,0,0]],'angular_velocities':[[0,0,0],[0,999,0]],'active':[True,False],'mechanical_energy':10.,'dissipated_energy':0.},
        {'time':.1,'centers':[[.2,0,0],[999,0,0]],'velocities':[[2,0,0],[999,0,0]],'angular_velocities':[[0,0,0],[0,999,0]],'active':[True,False],'mechanical_energy':10.,'dissipated_energy':0.}]}


def test_quality_separates_finite_integrity_from_physical_scale_without_modifying_data():
    source=replay();original=copy.deepcopy(source);manifest={'n_chunks':2,'n_frames':1,'dt':.1}
    row=inspect_replay(source,manifest)
    assert row['max_speed_mps']==2 and row['valid_chunk_intervals']==1
    assert row['constant_velocity_position_rmse_m']==0 and row['max_energy_to_initial_ratio']==1
    source['frames'][1]['velocities'][0]=[4e7,0,0]
    source['frames'][1]['centers'][0]=[1e4,0,0]
    source['frames'][1]['mechanical_energy']=4e18
    before=copy.deepcopy(source);bad=inspect_replay(source,manifest)
    assert bad['max_speed_mps']==4e7 and bad['max_energy_to_initial_ratio']==4e17
    assert bad['max_position_residual_m_at']=={'interval':0,'chunk_index':0}
    assert source==before and original==replay()
    assert bad['max_energy_at_snapshot']==1


def test_quality_rejects_corrupt_shapes_times_and_nonfinite_energy_and_keeps_unknown_ratio():
    manifest={'n_chunks':2,'n_frames':1,'dt':.1}
    for field,value in [('time',.2),('mechanical_energy',np.inf),('active',[1]),('centers',[[0,0,0]])]:
        source=replay();source['frames'][1][field]=value
        with pytest.raises(ValueError):inspect_replay(source,manifest)
    source=replay();source['frames'][0]['mechanical_energy']=0
    assert inspect_replay(source,manifest)['max_energy_to_initial_ratio'] is None
