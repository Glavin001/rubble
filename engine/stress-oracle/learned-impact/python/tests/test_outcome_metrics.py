import numpy as np

from learned_impact.outcome import metrics as M


def _ari_reference(a, b):
    from math import comb

    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ct = {}
    for x, y in zip(a, b):
        ct[(x, y)] = ct.get((x, y), 0) + 1
    s = sum(comb(v, 2) for v in ct.values())
    sa = sum(comb(int(v), 2) for v in np.bincount(np.unique(a, return_inverse=True)[1]))
    sb = sum(comb(int(v), 2) for v in np.bincount(np.unique(b, return_inverse=True)[1]))
    e = sa * sb / comb(n, 2)
    return (s - e) / (0.5 * (sa + sb) - e)


def test_weighted_ari_unit_weights_matches_standard_ari():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.integers(0, 4, 30)
        b = rng.integers(0, 5, 30)
        assert abs(M.weighted_ari(a, b) - _ari_reference(a, b)) < 1e-12
    assert M.weighted_ari([0, 0, 1, 1], [5, 5, 7, 7]) == 1.0


def test_weighted_ari_relabelling_invariant():
    a = np.array([0, 0, 1, 1, 2])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(M.weighted_ari(a, (a + 3) % 7, w) - 1.0) < 1e-12


def test_wasserstein_log_mass():
    assert M.wasserstein_log_mass([1, 2, 3], [3, 1, 2]) < 1e-12
    # all mass at e^0 against all at e^1 → distance 1
    assert abs(M.wasserstein_log_mass([1.0], [np.e]) - 1.0) < 1e-12


def test_skill_of_a_member_is_about_one():
    rng = np.random.default_rng(1)
    members = list(rng.normal(30, 1, 16))
    s = M.skill(members[0], members[1:], lambda a, b: abs(a - b))
    assert 0.5 < s < 2.0
    assert M.skill(100.0, members, lambda a, b: abs(a - b)) > 10


def test_ballistic_limit_fit_recovers_parameters():
    v0 = np.linspace(5, 60, 80)
    vr = M.lambert_jonas(v0, a=0.9, p=2.0, v_bl=22.0)
    fit = M.fit_ballistic_limit(v0, vr)
    assert abs(fit["v_bl"] - 22.0) < 0.5 and fit["p"] == 2.0 and abs(fit["a"] - 0.9) < 0.02


def test_calibration_helpers():
    p = np.array([0.1, 0.9, 0.8, 0.2])
    y = np.array([0, 1, 1, 0])
    assert M.brier(p, y) < 0.05 and M.ece(p, y) < 0.2
    assert M.prf(np.array([1, 0, 1], bool), np.array([1, 1, 0], bool)) == (0.5, 0.5, 0.5)
    t = M.fit_temperature(np.array([4.0, -4.0, 4.0, -4.0]), np.array([0.6, 0.4, 0.6, 0.4]))
    assert t > 4


def test_calibrated_gate_uses_heldout_member_skill():
    rng = np.random.default_rng(3)
    members = list(rng.normal(30, 1, 16))
    d = lambda a, b: abs(a - b)  # noqa: E731
    g = M.calibrated_gate(float(np.median(members)), members, d)
    h = M.heldout_member_skills(members, d)
    assert len(h) == 16 and abs(g["heldout_mean"] - h.mean()) < 1e-12
    assert g["bar"] == max(1.25, h.mean()) and g["pass"]
    # an outlier fails both verdicts
    bad = M.calibrated_gate(40.0, members, d)
    assert not bad["pass"] and not bad["pass_literal"]
    # a candidate between the literal bar and a looser oracle passes only the calibrated rule
    wide = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
    c = M.calibrated_gate(5.0, wide, d)
    assert c["bar"] >= 1.25 and c["pass"] == (c["skill"] <= c["bar"])


def test_evaluate_scores_against_the_ensemble(tmp_path):
    import torch

    from learned_impact.outcome import schema as S
    from learned_impact.outcome.batch import HierarchyCache, collate_events
    from learned_impact.outcome.ensemble import aggregate_dataset
    from learned_impact.outcome.model import ImpactOutcomeModel
    from learned_impact.outcome.synthetic import write_dataset

    write_dataset(tmp_path, 0, ensembles=1, ensemble_size=4)
    aggregate_dataset(tmp_path)
    evs = [S.load_event(p) for p in S.find_events(tmp_path)]
    b = collate_events(evs, HierarchyCache(2))
    torch.manual_seed(0)
    res = M.evaluate(ImpactOutcomeModel(), [b])
    for k in ("skill_ram", "skill_bar_ram", "pass_ram", "skill_partition", "pass_sizes",
              "fragments_in_range"):
        assert k in res.metrics, k
    assert np.isfinite(res.score)
