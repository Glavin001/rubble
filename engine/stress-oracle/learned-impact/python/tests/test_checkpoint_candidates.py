import json
import pytest
import torch
from learned_impact.checkpoint_candidates import PhysicalCandidates


def metrics(force,energy,motion,f1=.5,threshold=.6):
    fields=dict.fromkeys(('force_x','force_y','force_z'),force);fields['stored_energy']=energy
    return {'cached_response':{'active_end':{'bond_intervals':12,'model':{'si_component_rmse':fields}}},
            'world_motion':{'rmse':{'position_m':motion}},'f1':f1,'threshold':threshold}


def test_physical_candidates_preserve_distinct_tradeoffs_and_validation_thresholds(tmp_path):
    model=torch.nn.Linear(1,1,bias=False);keeper=PhysicalCandidates(tmp_path/'candidates')
    with torch.no_grad():model.weight.fill_(1.)
    keeper.consider(model,metrics(10.,4.,.3),1,{'architecture':'test'},{'events':{'a':'validation'}})
    with torch.no_grad():model.weight.fill_(2.)
    keeper.consider(model,metrics(8.,5.,.4,f1=.4,threshold=.7),2,{'architecture':'test'},{'events':{'a':'validation'}})
    # Lower fracture F1 must not discard the improved force candidate; energy
    # and motion retain the earlier weights and their own calibrated threshold.
    for name,weight,epoch,threshold in [('active_force_N',2.,2,.7),('active_energy_J',1.,1,.6),('world_position_m',1.,1,.6)]:
        path=keeper.root/name;sd=torch.load(path/'state_dict.pt',weights_only=True)
        assert sd['weight'].item()==weight
        run=json.loads((path/'run.json').read_text())
        assert run['selected_epoch']==epoch and run['threshold']==threshold and run['candidate_only']
        saved=json.loads((path/'metrics.json').read_text());assert 'test' not in saved
    # A tie must not replace the earliest candidate with later parameters.
    with torch.no_grad():model.weight.fill_(3.)
    keeper.consider(model,metrics(8.,4.,.3),3,{},{} )
    assert torch.load(keeper.root/'active_force_N/state_dict.pt',weights_only=True)['weight'].item()==2.
    assert keeper.best['active_energy_J']['epoch']==1


def test_physical_candidate_rejects_empty_or_nonfinite_validation(tmp_path):
    keeper=PhysicalCandidates(tmp_path/'candidates');model=torch.nn.Linear(1,1)
    bad=metrics(10.,1.,.2);bad['cached_response']['active_end']['bond_intervals']=0
    with pytest.raises(ValueError,match='active response'):keeper.consider(model,bad,1,{}, {})
    with pytest.raises(ValueError,match='finite'):keeper.consider(model,metrics(10.,float('nan'),.2),1,{}, {})
    assert not list(keeper.root.iterdir())
