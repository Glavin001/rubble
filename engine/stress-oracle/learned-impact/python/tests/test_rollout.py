import copy,json
import numpy as np
import torch
from learned_impact.rollout import endpoint,FORMAT
from learned_impact.runtime_snapshot import NativeGraph
from learned_impact.response_model import ResponseHistoryImpactNet
from test_runtime_snapshot import fixture

def test_endpoint_uses_live_state_keeps_exact_unchanged_history_and_no_future_targets():
    raw,snapshot=fixture();graph=NativeGraph(raw);model=ResponseHistoryImpactNet(h=8,p=4)
    snapshot['joint_values'][0][19]=0.123456789012345
    with torch.no_grad():model.exact_delta.weight.zero_();model.exact_delta.bias.zero_()
    msg=endpoint(model,graph,snapshot)
    assert msg['format']==FORMAT and msg['start_time']==0 and msg['dt']==.01
    assert msg['joint_values'][0][19]==snapshot['joint_values'][0][19]
    assert msg['joint_values'][0][4:6]==[None,None]
    assert np.asarray(msg['chunk_delta']).shape==(2,12)
    assert msg['bond_ids']==[0] and len(msg['response'][0])==18
    json.dumps(msg,allow_nan=False)
    altered=copy.deepcopy(snapshot);altered['world']['mechanical_energy']=float('nan')
    altered['world']['dissipated_energy']=float('nan')
    assert endpoint(model,graph,altered)==msg
    # A self-generated crack remains cracked in the next command. It cannot be
    # resurrected just because the classifier would now predict no new break.
    snapshot['joint_values'][0][0]=1.;snapshot['joint_flags'][0][3]=0
    msg=endpoint(model,graph,snapshot);assert msg['connected']==[False]
    assert msg['joint_values'][0][0]==1.
