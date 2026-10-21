"""The impact-outcome audit: integrity checks and the dataset report, on synthetic events
written in the recorder's on-disk format (no recorded data needed)."""
import json

import numpy as np
import pytest

from learned_impact import outcome_audit as oa


def write_event(root, name, *, frames=40, family="free_wall", tier=1, breaks=((0, 1, 3),), truncated=False, cost=2.0, sim=None):
    """An event of 3 chunks, 2 bonds, 1 impactor; `breaks` are (frame, bond, bin)."""
    d = root / name
    (d / "arrays").mkdir(parents=True)
    F, N, E, I = frames, 3, 2, 1
    conn = np.ones((F + 1, E), np.int32)
    bins = -np.ones((F, E), np.int32)
    for f, b, k in breaks:
        bins[f, b] = k
        conn[f + 1:, b] = 0
    phases = np.array([1] + [4] * (F - 1), np.int32)
    arrays = {
        "state_chunk_pos": np.zeros((F + 1, N, 3), np.float32), "state_bond_connected": conn,
        "state_impactor": np.zeros((F + 1, I, 13), np.float32), "frame_bond_break_bin": bins,
        "frame_bond_peak_util": np.zeros((F, E), np.float32),
        "frame_bond_qs_prior_util": np.where(conn[:-1] == 1, 0.5, -1.0).astype(np.float32),
        "frame_chunk_routed": np.zeros((F, N), np.int32), "frame_phase": phases,
        "frame_impactor_contact": np.zeros((F, I, 4), np.float32),
        "static_chunk_mass": np.ones((1, N), np.float32), "static_bond_endpoints": np.array([[[0, 1], [1, 2]]], np.int32),
        "qs_bond_broken": np.zeros((F, E), np.int32), "qs_chunk_vel": np.zeros((F, N, 3), np.float32),
        "qs_impactor": np.zeros((F, I, 13), np.float32),
    }
    catalog = {}
    for k, a in arrays.items():
        dt = "f32" if a.dtype == np.float32 else "i32"
        rel = f"arrays/{k}.{dt}"
        a.astype("<f4" if dt == "f32" else "<i4").tofile(d / rel)
        catalog[k] = {"dtype": dt, "shape": list(a.shape), "path": rel}
    (d / "catalog.json").write_text(json.dumps({"arrays": catalog}))
    oracle = {"name": "a12_realtime", "provisional": True, "source": "test"}
    (d / "context.json").write_text(json.dumps({"schema": oa.SCHEMA, "n_chunks": N, "n_bonds": E, "n_impactors": I, "family": family,
                                                "tier": tier, "oracle_config": oracle, "stop_rule": {"quiet_frames": 30}}))
    sim = sim if sim is not None else F / 60
    last = max((f for f, _, _ in breaks), default=None)
    (d / "outcome.json").write_text(json.dumps({
        "broken_bonds": len(breaks), "truncated": truncated, "sim_seconds": sim,
        "breaks_last_half_second": sum(1 for f, _, _ in breaks if f / 60 >= sim - 0.5),
        "cost": {"event": cost, "oracle": cost / 2, "qs_branch": cost / 2}, "impactors": [{"regime": "perforate"}],
        "last_break_time": None if last is None else last / 60,
    }))
    per_frame = [sum(1 for f, _, _ in breaks if f == k) for k in range(F)]
    (d / "frames.json").write_text(json.dumps([{"phase": oa.PHASES[p], "breaks": b} for p, b in zip(phases, per_frame)]))
    return d


def test_a_consistent_event_passes(tmp_path):
    d = write_event(tmp_path, "ok")
    ev = oa.load_event(d)
    assert oa.check_event(ev) == []
    assert ev.array("frame_bond_break_bin")[0, 1] == 3


def test_breaks_must_match_disconnections(tmp_path):
    d = write_event(tmp_path, "bad")
    bins = np.fromfile(d / "arrays/frame_bond_break_bin.i32", "<i4").reshape(40, 2)
    bins[0, 1] = -1  # the bond still disconnects: a missing break label
    bins.tofile(d / "arrays/frame_bond_break_bin.i32")
    problems = oa.check_event(oa.load_event(d))
    assert any("disconnections" in p for p in problems)


def test_late_breaks_in_a_finished_event_are_flagged(tmp_path):
    d = write_event(tmp_path, "late", breaks=((35, 0, 2),))
    problems = oa.check_event(oa.load_event(d))
    assert any("last 30 frames" in p for p in problems)
    r = oa.audit([tmp_path])
    assert r["still_breaking_last_half_second"] == ["late"]


def test_report_counts_families_phases_truncation_and_failures(tmp_path):
    run = tmp_path / "job"
    ev = run / "events"
    ev.mkdir(parents=True)
    write_event(ev, "a", cost=2.0)
    write_event(ev, "b", cost=4.0, breaks=())
    write_event(ev, "c", family="anchored_wall", truncated=True, frames=600, cost=30.0, breaks=((590, 0, 1),))
    (run / "failures.jsonl").write_text(json.dumps({"event_id": "d", "error": "nonfinite"}) + "\n")
    r = oa.audit([run])
    assert r["events"] == 3
    assert r["truncated"] == ["c"] and r["truncated_share"] == pytest.approx(1 / 3)
    # A truncated event may still be breaking; it is excluded, not counted as finished.
    assert r["still_breaking_last_half_second"] == []
    fam = r["families"]["free_wall/t1"]
    assert fam["events"] == 2 and fam["cost_s"]["mean"] == pytest.approx(3.0) and fam["no_break_share"] == 0.5
    assert r["frames_per_phase"]["contact"] == 3
    assert len(r["failures"]) == 1
    text = oa.markdown(r)
    assert "free_wall/t1" in text and "anchored_wall/t1" in text and "nonfinite" in text
    assert sum(b["count"] for b in r["duration_histogram_s"]) == 3
