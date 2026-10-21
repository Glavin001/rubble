import json
from pathlib import Path

import numpy as np
import pytest

from learned_impact.outcome import ensemble
from learned_impact.outcome import schema as S
from learned_impact.outcome.synthetic import make_event, write_dataset

DOC = Path(__file__).resolve().parents[2] / "OUTCOME_CALLS_SCHEMA.md"


def test_roundtrip_validates(tmp_path):
    paths = write_dataset(tmp_path, 3)
    for p in paths:
        ev = S.load_event(p)
        assert ev.manifest["schema_version"] == S.SCHEMA_VERSION
        assert len(ev.calls) == 2 and ev.calls[1].prev_call == 0
        assert set(ev.structures) == {0}


def test_doc_names_the_version_and_every_feature_array():
    doc = DOC.read_text()
    assert S.SCHEMA_VERSION in doc
    for name, cols in S.FEATURE_COLUMNS.items():
        assert f"`{name}`" in doc, name


@pytest.mark.parametrize("breakage", ["version", "columns", "twins", "unknown", "ddamage",
                                      "nonfinite", "ledger"])
def test_validator_fails_loud(tmp_path, breakage):
    m, calls, st = make_event(3)
    c = calls[0]
    if breakage == "version":
        m = dict(m)
    elif breakage == "twins":
        c.arrays["bond_dst"] = c.arrays["bond_dst"][::-1].copy()
    elif breakage == "unknown":
        c.arrays["mystery"] = np.zeros(3, np.float32)
    elif breakage == "ddamage":
        c.arrays["y_bond_ddamage"] = c.arrays["y_bond_ddamage"] - 1.0
    elif breakage == "nonfinite":
        c.arrays["chunk_feat"] = c.arrays["chunk_feat"].copy()
        c.arrays["chunk_feat"][0, 0] = np.nan
    elif breakage == "ledger":
        c.energy["D_bond"] += 1.0
    root = S.write_event(tmp_path / "e", m, calls, st)
    if breakage == "version":
        man = json.loads((root / "manifest.json").read_text())
        man["schema_version"] = "learned-impact-outcome/0"
        (root / "manifest.json").write_text(json.dumps(man))
    if breakage == "columns":
        man = json.loads((root / "manifest.json").read_text())
        man["columns"]["bond_feat"] = man["columns"]["bond_feat"][::-1]
        (root / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(S.SchemaError):
        S.load_event(root)


def test_ensemble_aggregation(tmp_path):
    write_dataset(tmp_path, 0, ensembles=1, ensemble_size=5)
    assert ensemble.aggregate_dataset(tmp_path) == {"ens_10000": 1}
    evs = [S.load_event(p) for p in S.find_events(tmp_path)]
    freqs = [e.calls[0].arrays["y_bond_break_freq"] for e in evs]
    broke = np.mean([e.calls[0].arrays["y_bond_broke"] for e in evs], axis=0)
    for f in freqs:
        np.testing.assert_allclose(f, broke)
    assert evs[0].calls[0].arrays["ens_impactor_v_end"].shape == (5, 1, 3)
    # the separation call differs in live bonds across members → not aligned
    assert all("y_bond_break_freq" not in e.calls[1].arrays for e in evs
               ) or len({len(e.calls[1].arrays["bond_key"]) for e in evs}) == 1


def test_zstd_arrays_load_identically(tmp_path):
    m, calls, st = make_event(4)
    a = S.load_event(S.write_event(tmp_path / "raw", m, calls, st))
    z = S.load_event(S.write_event(tmp_path / "zst", m, calls, st, compress=True))
    assert (tmp_path / "zst" / "arrays.bin.zst").exists()
    assert not (tmp_path / "zst" / "arrays.bin").exists()
    for ca, cz in zip(a.calls, z.calls):
        assert set(ca.arrays) == set(cz.arrays)
        for k in ca.arrays:
            np.testing.assert_array_equal(ca.arrays[k], cz.arrays[k])
