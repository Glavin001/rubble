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


def test_three_way_report(tmp_path):
    import torch

    from learned_impact.outcome import schema as S
    from learned_impact.outcome.batch import HierarchyCache, collate_events
    from learned_impact.outcome.model import ImpactOutcomeModel
    from learned_impact.outcome.synthetic import write_dataset

    write_dataset(tmp_path, 6)
    evs = [S.load_event(p) for p in S.find_events(tmp_path)]
    calls = [e.calls[0] for e in evs]  # strike calls carry the QS baseline
    b = collate_events(evs, HierarchyCache(2), calls)
    torch.manual_seed(0)
    rep = M.three_way(ImpactOutcomeModel(), [b])
    assert rep["available"] == ["model", "qs"]
    fam = rep["families"]["synthetic_wall"]
    assert set(fam["model"]) == set(M.PRIMARY_SINGLE) and "qs" in fam
    assert isinstance(fam["superiority"], bool)


def test_quiet_calls_are_scored_apart(tmp_path):
    """A model that predicts no breaks must not pass on the strength of quiet frames."""
    import torch

    from learned_impact.outcome import schema as S
    from learned_impact.outcome.batch import HierarchyCache, collate_events
    from learned_impact.outcome.model import ImpactOutcomeModel
    from learned_impact.outcome.synthetic import make_event

    for k in range(4):
        m, cs, st = make_event(3, split="val", ensemble=("e3", k, 4), settle_calls=3)
        S.write_event(tmp_path / m["event_id"], m, cs, st)
    from learned_impact.outcome.ensemble import aggregate_dataset
    aggregate_dataset(tmp_path)
    evs = [S.load_event(p) for p in S.find_events(tmp_path)]
    b = collate_events(evs, HierarchyCache(2))
    torch.manual_seed(0)
    from learned_impact.outcome.model import ModelConfig
    res = M.evaluate(ImpactOutcomeModel(ModelConfig(variant="A")), [b])  # near-zero init: no breaks
    fam = res.per_family["synthetic_wall"]
    assert fam["n_active"] >= 1 and fam["n_quiet"] >= 1
    assert "skill_over_bar_partition" in fam and "skill_over_bar_partition_quiet" in fam
    # active calls carry the breaks: missing them all cannot pass there
    assert fam["skill_over_bar_partition"] > fam["skill_over_bar_partition_quiet"]


def _ens_rows(rng, n_ens, members, model_noise, fam="free_wall"):
    """Synthetic aligned ensembles: 6 chunks, members split them at a random cut and exit at
    v ~ N(20, 1); the model's prediction is a member-like draw (noise 1) or far off."""
    from types import SimpleNamespace

    rows = []
    for e in range(n_ens):
        key = np.arange(6, dtype=np.uint64)
        mass = np.ones(6)
        cuts = rng.integers(2, 5, members)
        vs = rng.normal(20, 1, members)
        cut_hat = int(rng.integers(2, 5)) if model_noise < 5 else 1
        v_hat = float(rng.normal(20, model_noise))
        for k in range(members):
            c = SimpleNamespace(ensemble_id=f"e{e}", call_index=0, family=fam, phase="contact",
                                event_id=f"e{e}_m{k}", arrays={"chunk_key": key})
            rows.append({"call": c, "mass": mass, "pred": (np.arange(6) >= cut_hat).astype(int),
                         "true": (np.arange(6) >= cuts[k]).astype(int),
                         "imp_speed_hat": np.array([v_hat]), "imp_speed": np.array([vs[k]]),
                         "v_ref": 20.0, "imp_v0": np.array([30.0]),
                         "broke_hat": np.array([True]), "broke": np.array([True]),
                         "freq": np.array([1.0])})
    return rows


def _summ(rows):
    z = [np.zeros(1)]
    return M._summarise(rows, z, z, [np.zeros(0)], [np.zeros(0)], 0, 1, acceptance=True)


def test_family_acceptance_member_like_passes_outlier_fails():
    rng = np.random.default_rng(0)
    good = _summ(_ens_rows(rng, 40, 8, 1.0))
    acc = good.acceptance["free_wall"]
    assert acc["n_ensembles"] == 40 and acc["n_calls"] == 40
    for k in ("ram", "partition", "sizes"):
        e = acc[k]
        assert 0 <= e["member_pass_rate"] <= 1
        lo, hi = e["pass_rate_diff_ci"]
        assert lo <= e["model_pass_rate"] - e["member_pass_rate"] <= hi
        # a member-like model is statistically indistinguishable from a held-out member
        assert lo < 0.15 and hi > -0.15
    bad = _summ(_ens_rows(np.random.default_rng(0), 40, 8, 10.0))
    b = bad.acceptance["free_wall"]
    assert not b["accept"] and not b["ram"]["accept"] and not b["partition"]["accept"]
    assert b["ram"]["skill_over_bar_diff_ci"][0] > 0
    # the dense selection score ranks the outlier worse
    assert bad.score > good.score


def test_heldout_member_uses_the_identical_rule():
    rng = np.random.default_rng(1)
    rows = _ens_rows(rng, 1, 8, 1.0)
    r = M._acceptance_row(rows, {
        "ram": M.calibrated_gate(0.0, [0.0, 1.0, 2.0], lambda a, b: abs(a - b)),
        "partition": M.calibrated_gate(0.0, [0.0, 1.0, 2.0], lambda a, b: abs(a - b)),
        "sizes": M.calibrated_gate(0.0, [0.0, 1.0, 2.0], lambda a, b: abs(a - b))},
        rows[0]["pred"], rows[0]["mass"])
    speeds = [float(x["imp_speed"][0]) for x in rows]
    held = [M.calibrated_gate(speeds[i], speeds[:i] + speeds[i + 1:], lambda a, b: abs(a - b),
                              tolerance=M.TOLERANCE["ram"] * 20.0) for i in range(8)]
    assert abs(r["member"]["ram"][0] - np.mean([h["pass"] for h in held])) < 1e-12
    assert abs(r["member"]["ram"][1] - np.mean([h["skill"] / h["bar"] for h in held])) < 1e-12
