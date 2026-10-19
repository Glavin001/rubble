import numpy as np
import pytest
from learned_impact.state_consistency import joint_connected,decision_counts


def test_joint_connectivity_keeps_unbroken_rebar_and_requires_exact_complete_damage():
    damage=np.array([0.,.99,1.,1.,1.,np.nextafter(1.,0.)])
    present=np.array([False,False,False,True,True,False])
    broken=np.array([False,False,False,False,True,False])
    assert joint_connected(damage,broken,present).tolist()==[True,True,False,True,False,True]
    with pytest.raises(ValueError,match='booleans'):joint_connected(damage,broken,present.astype(int))
    with pytest.raises(ValueError,match='admissible'):joint_connected(damage+1,broken,present)


def test_consistency_counts_distinguish_agreement_from_accuracy():
    # Both heads can agree and still be wrong; report both confusion matrices.
    result=decision_counts(np.array([True,True,False,False]),np.array([True,False,True,True]),np.array([False,True,True,False]))
    assert result['head_break_state_connected']==2
    assert result['head_intact_state_disconnected']==1
    assert (result['head_tp'],result['head_fp'],result['head_fn'])==(1,2,1)
    assert (result['state_tp'],result['state_fp'],result['state_fn'])==(1,1,1)
