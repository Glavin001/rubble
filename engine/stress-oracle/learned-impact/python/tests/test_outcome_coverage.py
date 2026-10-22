"""Validation tiers from recipes: in-distribution, covered-factor, extrapolation."""

from learned_impact.outcome import coverage as C


def rec(eid, split, depth=1, chunk=0.1, supports="free", shape="box", mass=100.0, speed=2.0,
        catalog="p1-walls", key=None, extrap=None):
    key = key or f"10x{depth}x10_c{int(chunk * 1000)}mm_{supports}"
    return {"event_id": eid, "family": "f", "parameters": {
        "split": split, "catalog": catalog, "extrapolation": extrap,
        "layout": {"key": key, "counts": [10, depth, 10], "chunk": chunk, "supports": supports, "opening": None},
        "impactors": [{"shape": shape, "mass": mass, "speed": speed}],
        "material": {"stiffness_scale": 1.0, "strength_scale": 1.0, "toughness_scale": 1.0}}, "scene": {}}


def test_tiers():
    train = [rec("t1", "train", depth=1, speed=0.5), rec("t2", "train", depth=2, speed=50.0, mass=500.0),
             rec("b1", "train", depth=1, catalog="p1-ballistic", speed=3.0)]
    cov = C.Coverage(train)
    assert cov.classify(rec("v1", "val", depth=1, speed=10.0))[0] == 1  # a training layout
    assert cov.classify(rec("v2", "val", depth=2, speed=10.0, key="other"))[0] == 2  # new layout
    t, why = cov.classify(rec("v3", "val", depth=2, catalog="p1-ballistic"))
    assert t == 3 and "transition regime at depth 2" in why[0]  # thick ballistic: extrapolation
    assert cov.classify(rec("v4", "val", mass=5000.0))[0] == 3  # mass outside training
    assert cov.classify(rec("v5", "val", speed=0.49))[0] == 1  # within the 5% edge margin
    assert cov.classify(rec("v6", "val", extrap="speed"))[0] == 3  # designed extrapolation
    assert C.event_tier({"events": {}}, "held", "val") == 1  # held out in memory from training
