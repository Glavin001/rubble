"""Plain-assert checks of the OpenSees exporter/runner/post-processor.

    python -I oracles/opensees/test_export.py [--scenes DIR]

Needs the generated scenes (`stress-ref gen-scenes scenes`). Prints the numbers it checks.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import openseespy.opensees as ops  # noqa: E402

import export  # noqa: E402
import observe  # noqa: E402
import run  # noqa: E402

SCENES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "scenes")


def load(name):
    with open(os.path.join(SCENES, name + ".json")) as f:
        return json.load(f)


def static_only(scene):
    s = copy.deepcopy(scene)
    s["sim"]["duration"] = s["sim"].get("sample_interval") or s["sim"]["frame_dt"]
    return s


def run_quiet(model):
    return run.run_model(model, ops, log=lambda *a: None)


def probe_at0(model, raw, name):
    p = next(p for p in model["probes"] if p["name"] == name)
    return float(observe._probe_series(model, raw, p)[0])


def test_cantilever_static_and_rbsn_cross_check():
    """Static tip deflection = P L^3 / 3EI (1 %); RBSN chain within 0.5 % of the beam elements."""
    for name, n in [("b2_cantilever_n10", 10), ("b2_cantilever", 20), ("b2_cantilever_n40", 40)]:
        scene = load(name)
        model = export.export_model(static_only(scene))
        raw = run_quiet(model)
        m = scene["materials"]["beam"]
        e, nu, rho = m["youngs_modulus"], m["poisson_ratio"], m["density"]
        h = 2.0 / n
        p_load, i_sec, area = 30.0, 0.1 ** 4 / 12.0, 0.01
        arm = n * h
        tip = probe_at0(model, raw, "tip")
        root = probe_at0(model, raw, "root_moment")
        eb = -p_load * arm ** 3 / (3 * e * i_sec)
        # Our RBSN chain: rotation jumps M(x_j) h / EI at the interfaces (midpoint rule) plus
        # shear springs G A / h:  P h^3/EI (n^3/3 - n/12) + n P h / (G A).
        g = e / (2 * (1 + nu))
        rbsn = -(p_load * h ** 3 / (e * i_sec) * (n ** 3 / 3.0 - n / 12.0) + n * p_load * h / (g * area))
        print(f"  {name}: tip opensees {tip:.6e}  PL^3/3EI {eb:.6e}  RBSN closed form {rbsn:.6e} "
              f"({100 * (rbsn - tip) / tip:+.3f} %)  root moment {root:.4f} (expected {-p_load * (arm - 0.5 * h):.4f})")
        assert abs(tip - eb) <= 0.01 * abs(eb), (tip, eb)
        assert abs(rbsn - tip) <= 0.005 * abs(tip), (rbsn, tip)
        assert abs(root + p_load * (arm - 0.5 * h)) <= 1e-6 * abs(root), root
        # Eigenvalue f1 vs Euler-Bernoulli (free end at the last chunk's outer face).
        span = (n + 0.5) * h
        f1 = 1.87510407 ** 2 / (2 * math.pi) * math.sqrt(e * i_sec / (rho * area * span ** 4))
        print(f"    eigen f1 {raw['eigen_hz'][0]:.5f} Hz vs Euler-Bernoulli {f1:.5f} Hz")
        assert abs(raw["eigen_hz"][0] - f1) <= 0.01 * f1


def test_axial_sign_and_failure_index():
    """Pulling the tip: section `normal` = +P (tension positive); tension index = P/A / f_t."""
    scene = static_only(load("b2_cantilever_n10"))
    scene["loads"][0]["direction"] = [1.0, 0.0, 0.0]
    scene["probes"].append({"name": "axial", "type": "section_force", "body": "beam", "point": [0.5, 0, 0],
                            "normal": [1.0, 0.0, 0.0], "region": None, "component": "normal"})
    scene["probes"].append({"name": "shear_z", "type": "section_force", "body": "beam", "point": [0.5, 0, 0],
                            "normal": [1.0, 0.0, 0.0], "region": None, "component": {"force": [0.0, 0.0, 1.0]}})
    model = export.export_model(scene)
    raw = run_quiet(model)
    axial = probe_at0(model, raw, "axial")
    print(f"  axial section force {axial:.6f} N (load 30 N)")
    assert abs(axial - 30.0) < 1e-6
    best, _, mode, _ = observe.failure_history(model, raw)
    expect = 30.0 / 0.01 / scene["materials"]["beam"]["tensile_strength"]
    print(f"  max failure index {best[0]:.6e} ({mode[0]}), expected {expect:.6e}")
    assert abs(best[0] - expect) < 1e-9 and mode[0] == "tension"
    # Downward tip load: the +x side of the cut is pushed down, the section holds it up.
    scene["loads"][0]["direction"] = [0.0, 0.0, -1.0]
    raw = run_quiet(export.export_model(scene))
    assert abs(probe_at0(model, raw, "shear_z") - 30.0) < 1e-6


def test_time_functions():
    """Path series reproduce every TimeFunction at the analysis step times."""
    rng = random.Random(1)
    funcs = [
        {"type": "constant", "value": 2.5},
        {"type": "ramp", "t0": 0.1, "t1": 0.3, "value": -4.0},
        {"type": "table", "points": [[0.0, 30.0], [0.05, 30.0], [0.050000001, 0.0]]},
        {"type": "table", "points": [[0.2, 1.0], [0.4, 3.0]]},
        {"type": "half_sine", "start": 0.01, "duration": 0.02, "peak": 7.0},
        {"type": "friedlander", "arrival": 0.001, "peak": 5.0, "duration": 0.01, "decay": 1.0},
    ]
    dt, duration = 2.5e-4, 0.5
    for tf in funcs:
        path = export.time_function_path(tf, duration, dt)
        for _ in range(200):
            t = round(rng.uniform(0, duration) / dt) * dt
            want = export.eval_time_function(tf, t)
            got = path[1] if path[0] == "constant" else float(np.interp(t, path[0], path[1]))
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (tf, t, got, want)


def test_force_replacement_is_exact():
    """A removal whose replacement loads never decay leaves the frame in its static state."""
    scene = load("b8_frame_gradual")
    scene["events"][0]["duration"] = 1e9
    scene["sim"]["duration"] = 0.06
    model = export.export_model(scene)
    raw = run_quiet(model)
    for p in model["probes"]:
        v = observe._probe_series(model, raw, p)
        before = v[np.searchsorted(raw["t"], 0.05)]
        after = v[-1]
        print(f"  {p['name']}: before {before:.4f}, 10 ms after removal {after:.4f}")
        assert abs(after - before) <= 1e-6 * abs(before) + 1e-6, (p["name"], before, after)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("all tests passed")
