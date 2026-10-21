"""`learned-impact-outcome/1`: the recorder ↔ trainer contract (mirror of OUTCOME_CALLS_SCHEMA.md).

One record is one routed impact island at one engine frame (a *call*). The recorder writes
every model input; this module only names, loads, validates and (for synthetic data and the
ensemble tool) writes arrays. Nothing here derives a feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "learned-impact-outcome/1"
N_TIME_BINS = 8
PHASES = ("pre_contact", "contact", "blast", "separation", "secondary", "settling", "quiet")
REGIMES = {"arrest": 0, "perforate": 1, "rebound": 2, "deflect": 3}
REGIME_UNDECIDED = -1
EDGE_TYPES = ("bond", "cracked", "impactor_contact", "structure_contact")
DTYPES = {"f32": np.float32, "i32": np.int32, "i64": np.int64}
AVAILABLE_FLAGS = (
    "qs_hertz",
    "punch",
    "qs_frame",
    "punch_frame",
    "blast",
    "cracked_edges",
    "structure_contacts",
    "bond_energy_split",
)

_ROT = [f"rot_src_{i}" for i in range(6)] + [f"rot_dst_{i}" for i in range(6)]


def _v(name: str, n: int = 3) -> list[str]:
    return [f"{name}_{i}" for i in range(n)]


CHUNK_FEAT_COLUMNS: tuple[str, ...] = tuple(
    ["log_mass", *_v("inertia"), *_v("half_extent"), "support_fixed", "support_pinned",
     "bond_damage_mean", "bond_damage_max", "n_live_bonds", "n_broken_bonds",
     "hops_to_contact", "wave_arrival", "hops_to_anchor", *_v("gravity_body"),
     *_v("v_rel_body"), *_v("w_body"), "punch_in_plug", *_v("blast", 6), "in_contact_any"]
)
IMPACTOR_FEAT_COLUMNS: tuple[str, ...] = tuple(
    ["log_mass_island", "log_mass_chunk", *_v("inertia"), "log_radius", "log_modulus",
     "crushable", *_v("gravity_body"), *_v("v_rel_body"), "speed_over_c", "in_contact_any"]
)
GLOBAL_FEAT_COLUMNS: tuple[str, ...] = (
    "log_energy_ratio", "log_energy_ratio_plug", "dt_c_over_L", "dt_v_over_L", "n_impactors",
    "log_n_chunks", "gravity_mag", "anchored", "blast_log_tnt", "blast_z", "blast_impulse",
    "v_over_c",
)
BOND_FEAT_COLUMNS: tuple[str, ...] = tuple(
    ["log_area", *_v("offset"), *_v("log_stiffness", 6), *_v("log_strength"), *_v("log_ga"),
     "weibull", "damage", "crush", "utilization", "peak_util_prev", "rebar",
     "log_qs_hertz_util", "punch_plug_bond", "log_punch_capacity_ratio", *_v("v_rel_edge"),
     *_ROT]
)
CRACKED_FEAT_COLUMNS: tuple[str, ...] = tuple(
    ["log_area", *_v("offset"), "gap", "friction", *_v("v_rel_edge"), "frames_since_broken",
     *_ROT]
)
IMPACTOR_CONTACT_FEAT_COLUMNS: tuple[str, ...] = tuple(
    ["v_n", *_v("v_t", 2), "v_n_over_c", "log_mass_ratio", "log_radius", "log_modulus",
     *_v("spin"), "t_contact", "log_energy_ratio", "log_hertz_force", "hertz_duration",
     "is_self_impactor", "overlap", "friction", "restitution", *_ROT]
)
STRUCTURE_CONTACT_FEAT_COLUMNS: tuple[str, ...] = tuple(
    ["v_n", *_v("v_t", 2), "v_n_over_c", "log_mass_ratio", "t_contact", "overlap", "friction",
     "log_area", "same_structure", *_ROT]
)
# Variant B groups (blast_parity QS frame outcome on the same state).
CHUNK_QS_FEAT_COLUMNS: tuple[str, ...] = tuple([*_v("qs_dv_body"), *_v("qs_dw_body")])
IMPACTOR_QS_FEAT_COLUMNS = CHUNK_QS_FEAT_COLUMNS
BOND_QS_FEAT_COLUMNS: tuple[str, ...] = (
    "log_qs_frame_peak_util", "qs_frame_broke", "qs_frame_break_time",
)

FEATURE_COLUMNS: dict[str, tuple[str, ...]] = {
    "chunk_feat": CHUNK_FEAT_COLUMNS,
    "impactor_feat": IMPACTOR_FEAT_COLUMNS,
    "global_feat": GLOBAL_FEAT_COLUMNS,
    "bond_feat": BOND_FEAT_COLUMNS,
    "cracked_feat": CRACKED_FEAT_COLUMNS,
    "impactor_contact_feat": IMPACTOR_CONTACT_FEAT_COLUMNS,
    "structure_contact_feat": STRUCTURE_CONTACT_FEAT_COLUMNS,
    "chunk_qs_feat": CHUNK_QS_FEAT_COLUMNS,
    "impactor_qs_feat": IMPACTOR_QS_FEAT_COLUMNS,
    "bond_qs_feat": BOND_QS_FEAT_COLUMNS,
}

# Columns rewritten from the model's own decoded output in pushforward and noised in
# training (GNS recipe). Names, resolved to indices by `carried_indices`.
CARRIED_COLUMNS: dict[str, tuple[str, ...]] = {
    "chunk_feat": (*_v("v_rel_body"), *_v("w_body")),
    "impactor_feat": (*_v("v_rel_body"), "speed_over_c"),
    "bond_feat": ("damage", "peak_util_prev", *_v("v_rel_edge")),
    "cracked_feat": tuple(_v("v_rel_edge")),
    "impactor_contact_feat": ("v_n", "v_t_0", "v_t_1", "v_n_over_c", *_v("spin")),
    "structure_contact_feat": ("v_n", "v_t_0", "v_t_1", "v_n_over_c"),
}


def col(array: str, name: str) -> int:
    return FEATURE_COLUMNS[array].index(name)


def carried_indices(array: str) -> list[int]:
    return [col(array, n) for n in CARRIED_COLUMNS.get(array, ())]


STRUCTURE_ARRAYS = ("chunk_key", "bond_key", "bond_ends", "chunk_rest_pos")

SCALE_KEYS = ("L", "m", "v", "c", "f_t", "E", "GA", "dt", "E_in")
ENERGY_TARGET_KEYS = ("KE_in", "KE_out", "W_gravity", "W_external", "E_elastic_start",
                      "E_elastic_end", "D_bond")
ENERGY_OPTIONAL_KEYS = ("D_bond_joint", "D_bond_damping", "D_fracture_floor",
                        "D_contact_booked", "D_friction_booked", "ledger_residual")


@dataclass(frozen=True)
class ArraySpec:
    dtype: str
    # Shape in symbols: Nc, Ni, E (undirected bonds), Ec, Ei, Es, 2E…, M (ensemble), or ints.
    shape: tuple[Any, ...]
    required: bool = True
    raw: bool = False  # decode/metric only: never a model input


def _feat(n: str, rows: str) -> ArraySpec:
    return ArraySpec("f32", (rows, len(FEATURE_COLUMNS[n])))


CALL_ARRAYS: dict[str, ArraySpec] = {
    # chunks
    "chunk_key": ArraySpec("i64", ("Nc",)),
    "chunk_feat": _feat("chunk_feat", "Nc"),
    "chunk_qs_feat": _feat("chunk_qs_feat", "Nc"),
    "chunk_mass": ArraySpec("f32", ("Nc",), raw=True),
    "chunk_inertia_body": ArraySpec("f32", ("Nc", 3), raw=True),
    "chunk_q_world": ArraySpec("f32", ("Nc", 4), raw=True),
    "chunk_pos": ArraySpec("f32", ("Nc", 3), raw=True),
    "chunk_v": ArraySpec("f32", ("Nc", 3), raw=True),
    "chunk_w": ArraySpec("f32", ("Nc", 3), raw=True),
    "chunk_anchored": ArraySpec("i32", ("Nc",), raw=True),
    # impactors
    "impactor_key": ArraySpec("i64", ("Ni",)),
    "impactor_feat": _feat("impactor_feat", "Ni"),
    "impactor_qs_feat": _feat("impactor_qs_feat", "Ni"),
    "impactor_mass": ArraySpec("f32", ("Ni",), raw=True),
    "impactor_inertia_body": ArraySpec("f32", ("Ni", 3), raw=True),
    "impactor_q_world": ArraySpec("f32", ("Ni", 4), raw=True),
    "impactor_pos": ArraySpec("f32", ("Ni", 3), raw=True),
    "impactor_v": ArraySpec("f32", ("Ni", 3), raw=True),
    "impactor_w": ArraySpec("f32", ("Ni", 3), raw=True),
    # global
    "global_feat": ArraySpec("f32", (1, len(GLOBAL_FEAT_COLUMNS))),
    "bond_qs_feat": _feat("bond_qs_feat", "2E"),
    # targets: bonds
    "y_bond_peak_util": ArraySpec("f32", ("E",)),
    "y_bond_broke": ArraySpec("i32", ("E",)),
    "y_bond_break_bin": ArraySpec("i32", ("E",)),
    "y_bond_fail_mode": ArraySpec("i32", ("E",)),
    "y_bond_ddamage": ArraySpec("f32", ("E",)),
    "y_bond_dissipated": ArraySpec("f32", ("E",)),
    "y_bond_support_path": ArraySpec("i32", ("E",), required=False),
    "y_bond_break_freq": ArraySpec("f32", ("E",), required=False),
    # targets: chunks
    "y_chunk_v_end": ArraySpec("f32", ("Nc", 3)),
    "y_chunk_w_end": ArraySpec("f32", ("Nc", 3)),
    "y_chunk_dx": ArraySpec("f32", ("Nc", 3)),
    "y_chunk_drot": ArraySpec("f32", ("Nc", 3)),
    "y_chunk_fragment": ArraySpec("i32", ("Nc",)),
    # targets: impactors
    "y_impactor_v_end": ArraySpec("f32", ("Ni", 3)),
    "y_impactor_w_end": ArraySpec("f32", ("Ni", 3)),
    "y_impactor_dx": ArraySpec("f32", ("Ni", 3)),
    "y_impactor_drot": ArraySpec("f32", ("Ni", 3)),
    "y_impactor_regime": ArraySpec("i32", ("Ni",)),
    "y_impactor_perforate_freq": ArraySpec("f32", ("Ni",), required=False),
    "ens_impactor_v_end": ArraySpec("f32", ("M", "Ni", 3), required=False),
}
for _t, _n in (("bond", "E"), ("cracked", "Ec"), ("impactor_contact", "Ei"),
               ("structure_contact", "Es")):
    CALL_ARRAYS[f"{_t}_src"] = ArraySpec("i32", (f"2{_n}",))
    CALL_ARRAYS[f"{_t}_dst"] = ArraySpec("i32", (f"2{_n}",))
    CALL_ARRAYS[f"{_t}_key"] = ArraySpec("i64", (_n,))
    CALL_ARRAYS[f"{_t}_feat"] = _feat(f"{_t}_feat", f"2{_n}")
    CALL_ARRAYS[f"{_t}_frame"] = ArraySpec("f32", (_n, 3, 3), raw=True)
for _b in ("qs", "punch"):
    CALL_ARRAYS[f"base_{_b}_bond_broke"] = ArraySpec("i32", ("E",), required=False, raw=True)
    CALL_ARRAYS[f"base_{_b}_bond_peak_util"] = ArraySpec("f32", ("E",), required=False, raw=True)
    CALL_ARRAYS[f"base_{_b}_chunk_v_end"] = ArraySpec("f32", ("Nc", 3), required=False, raw=True)
    CALL_ARRAYS[f"base_{_b}_chunk_w_end"] = ArraySpec("f32", ("Nc", 3), required=False, raw=True)
    CALL_ARRAYS[f"base_{_b}_impactor_v_end"] = ArraySpec("f32", ("Ni", 3), required=False,
                                                         raw=True)

COUNT_SYMBOL = {"chunks": "Nc", "impactors": "Ni", "bond": "E", "cracked": "Ec",
                "impactor_contact": "Ei", "structure_contact": "Es"}


class SchemaError(ValueError):
    pass


@dataclass
class Call:
    """One routed island at one frame. `arrays` hold numpy views into `arrays.bin`."""

    event_id: str
    call_index: int
    frame: int
    island: int
    prev_call: int | None
    phase: str
    transition: bool
    routed: bool
    anchored: bool
    counts: dict[str, int]
    scales: dict[str, float]
    energy: dict[str, float]
    arrays: dict[str, np.ndarray]
    # event-level context copied in for the sampler and splits
    family: str = ""
    split: str = ""
    split_group: str = ""
    ensemble_id: str | None = None
    ensemble_member: int | None = None
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def has(self, name: str) -> bool:
        return name in self.arrays


@dataclass
class Event:
    root: Path
    manifest: dict[str, Any]
    calls: list[Call]
    structures: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


def _resolve(shape: tuple[Any, ...], sym: dict[str, int]) -> tuple[int, ...]:
    out = []
    for s in shape:
        if isinstance(s, int):
            out.append(s)
        elif s.startswith("2"):
            out.append(2 * sym[s[1:]])
        else:
            out.append(sym[s])
    return tuple(out)


def _read(blob: np.memmap, entry: dict[str, Any], where: str) -> np.ndarray:
    if entry["dtype"] not in DTYPES:
        raise SchemaError(f"{where}: dtype {entry['dtype']!r}")
    dt = np.dtype(DTYPES[entry["dtype"]]).newbyteorder("<")
    shape = tuple(int(x) for x in entry["shape"])
    off = int(entry["offset"])
    if off % 8:
        raise SchemaError(f"{where}: offset {off} not 8-byte aligned")
    n = int(np.prod(shape)) if shape else 1
    end = off + n * dt.itemsize
    if end > blob.shape[0]:
        raise SchemaError(f"{where}: past end of arrays.bin")
    return np.frombuffer(blob, dtype=dt, count=n, offset=off).reshape(shape)


def validate_manifest(m: dict[str, Any]) -> None:
    if m.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"schema_version {m.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    if int(m.get("n_time_bins", -1)) != N_TIME_BINS:
        raise SchemaError(f"n_time_bins {m.get('n_time_bins')} != {N_TIME_BINS}")
    cols = m.get("columns", {})
    for name, expected in FEATURE_COLUMNS.items():
        if tuple(cols.get(name, ())) != expected:
            raise SchemaError(f"columns[{name}] differ from schema.py")
    avail = m.get("available", {})
    missing = [f for f in AVAILABLE_FLAGS if f not in avail]
    if missing:
        raise SchemaError(f"available flags missing: {missing}")
    for k in ("event_id", "family", "split_group", "split", "frame_dt", "n_calls"):
        if k not in m:
            raise SchemaError(f"manifest.{k} missing")


def validate_call(c: Call, require_targets: bool = True) -> None:
    if c.phase not in PHASES:
        raise SchemaError(f"{c.event_id}#{c.call_index}: phase {c.phase!r}")
    sym = {COUNT_SYMBOL[k]: int(v) for k, v in c.counts.items()}
    if "ens_impactor_v_end" in c.arrays:
        sym["M"] = int(c.arrays["ens_impactor_v_end"].shape[0])
    for k in SCALE_KEYS:
        v = c.scales.get(k)
        if v is None or not np.isfinite(v) or v <= 0:
            raise SchemaError(f"{c.event_id}#{c.call_index}: scale {k}={v}")
    for k in ENERGY_TARGET_KEYS if require_targets else ():
        if k not in c.energy or not np.isfinite(c.energy[k]):
            raise SchemaError(f"{c.event_id}#{c.call_index}: energy {k}")
    for name, spec in CALL_ARRAYS.items():
        if name not in c.arrays:
            if spec.required and (require_targets or not _is_target(name)):
                raise SchemaError(f"{c.event_id}#{c.call_index}: {name} missing")
            continue
        a = c.arrays[name]
        if a.dtype != DTYPES[spec.dtype]:
            raise SchemaError(f"{name}: dtype {a.dtype}")
        want = _resolve(spec.shape, sym)
        if a.shape != want:
            raise SchemaError(f"{c.event_id}#{c.call_index}: {name} shape {a.shape} != {want}")
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            raise SchemaError(f"{c.event_id}#{c.call_index}: {name} not finite")
    for name in c.arrays:
        if name not in CALL_ARRAYS:
            raise SchemaError(f"{c.event_id}#{c.call_index}: unknown array {name}")
    n_nodes = sym["Nc"] + sym["Ni"]
    for t in EDGE_TYPES:
        s, d = c.arrays[f"{t}_src"], c.arrays[f"{t}_dst"]
        if s.size and (s.min() < 0 or s.max() >= n_nodes or d.min() < 0 or d.max() >= n_nodes):
            raise SchemaError(f"{t}: endpoint out of range")
        if s.size and not (np.array_equal(s[0::2], d[1::2]) and np.array_equal(s[1::2], d[0::2])):
            raise SchemaError(f"{t}: rows 2k/2k+1 are not reverse twins")
    if not require_targets and "y_bond_broke" not in c.arrays:
        return
    if np.any(c.arrays["y_bond_ddamage"] < 0):
        raise SchemaError("y_bond_ddamage < 0")
    bins = c.arrays["y_bond_break_bin"]
    broke = c.arrays["y_bond_broke"]
    if np.any((bins > 0) != (broke > 0)) or bins.max(initial=0) > N_TIME_BINS:
        raise SchemaError("y_bond_break_bin inconsistent with y_bond_broke")
    if abs(float(c.arrays["y_bond_dissipated"].astype(np.float64).sum()) - c.energy["D_bond"]) > (
        1e-6 * max(1.0, abs(c.energy["D_bond"])) + 1e-3
    ):
        raise SchemaError("energy.D_bond != sum(y_bond_dissipated)")


def _is_target(name: str) -> bool:
    return name.startswith("y_") or name.startswith("ens_") or name.startswith("base_")


def with_dummy_targets(c: Call) -> Call:
    """A closed-loop call (no oracle outcome) with zero targets, so it collates; inference
    never reads them."""
    a = dict(c.arrays)
    sym = {COUNT_SYMBOL[k]: int(v) for k, v in c.counts.items()}
    for name, spec in CALL_ARRAYS.items():
        if spec.required and _is_target(name) and name not in a:
            a[name] = np.zeros(_resolve(spec.shape, sym), dtype=DTYPES[spec.dtype])
    if "y_chunk_fragment" in a and not np.any(a["y_chunk_fragment"]):
        a["y_chunk_fragment"] = np.arange(sym["Nc"], dtype=np.int32)
    energy = {k: 0.0 for k in ENERGY_TARGET_KEYS}
    energy.update(c.energy)
    return Call(**{**c.__dict__, "arrays": a, "energy": energy})


def load_event(path: Path | str, validate: bool = True, require_targets: bool = True) -> Event:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if validate:
        validate_manifest(manifest)
    entries = json.loads((root / "calls.json").read_text())
    blob = np.memmap(root / "arrays.bin", dtype=np.uint8, mode="r")
    structures: dict[int, dict[str, np.ndarray]] = {}
    for s in manifest.get("structures", []):
        sid = int(s["structure_id"])
        cat = s["arrays"]
        structures[sid] = {k: _read(blob, cat[k], f"structure/{sid}/{k}")
                           for k in STRUCTURE_ARRAYS}
    calls = []
    for e in entries:
        arrays = {k: _read(blob, v, f"call {e['call_index']}/{k}") for k, v in e["arrays"].items()}
        c = Call(
            event_id=manifest["event_id"], call_index=int(e["call_index"]),
            frame=int(e["frame"]), island=int(e["island"]), prev_call=e.get("prev_call"),
            phase=e["phase"], transition=bool(e["transition"]), routed=bool(e["routed"]),
            anchored=bool(e["anchored"]), counts={k: int(v) for k, v in e["counts"].items()},
            scales={k: float(v) for k, v in e["scales"].items()},
            energy={k: float(v) for k, v in e["energy"].items()}, arrays=arrays,
            family=manifest["family"], split=manifest["split"],
            split_group=manifest["split_group"], ensemble_id=manifest.get("ensemble_id"),
            ensemble_member=manifest.get("ensemble_member"),
            gravity=tuple(float(x) for x in manifest.get("gravity", (0.0, 0.0, 0.0))),
        )
        if validate:
            validate_call(c, require_targets)
        calls.append(c)
    if validate and len(calls) != int(manifest["n_calls"]):
        raise SchemaError(f"n_calls {manifest['n_calls']} != {len(calls)}")
    return Event(root=root, manifest=manifest, calls=calls, structures=structures)


class _BlobWriter:
    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.offset = 0

    def add(self, a: np.ndarray) -> dict[str, Any]:
        kind = {np.dtype(np.float32): "f32", np.dtype(np.int32): "i32",
                np.dtype(np.int64): "i64"}[a.dtype]
        raw = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()
        entry = {"dtype": kind, "shape": list(a.shape), "offset": self.offset}
        pad = (-len(raw)) % 8
        self.parts.append(raw + b"\0" * pad)
        self.offset += len(raw) + pad
        return entry


def write_event(root: Path | str, manifest: dict[str, Any], calls: list[Call],
                structures: dict[int, dict[str, np.ndarray]]) -> Path:
    """Write an event in the exact on-disk format (synthetic data, tests, ensemble tool)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    w = _BlobWriter()
    m = dict(manifest)
    m["schema_version"] = SCHEMA_VERSION
    m["n_time_bins"] = N_TIME_BINS
    m["columns"] = {k: list(v) for k, v in FEATURE_COLUMNS.items()}
    m["n_calls"] = len(calls)
    m["structures"] = [
        {"structure_id": sid, "n_chunks": int(s["chunk_key"].shape[0]),
         "n_bonds": int(s["bond_key"].shape[0]),
         "arrays": {k: w.add(s[k]) for k in STRUCTURE_ARRAYS}}
        for sid, s in structures.items()
    ]
    entries = []
    for c in calls:
        entries.append({
            "call_index": c.call_index, "frame": c.frame, "island": c.island,
            "prev_call": c.prev_call, "phase": c.phase, "transition": c.transition,
            "routed": c.routed, "anchored": c.anchored, "counts": c.counts,
            "scales": c.scales, "energy": c.energy,
            "arrays": {k: w.add(v) for k, v in c.arrays.items()},
        })
    (root / "manifest.json").write_text(json.dumps(m, indent=1))
    (root / "calls.json").write_text(json.dumps(entries))
    (root / "arrays.bin").write_bytes(b"".join(w.parts))
    return root


def find_events(dataset: Path | str) -> list[Path]:
    return sorted(p.parent for p in Path(dataset).rglob("manifest.json")
                  if (p.parent / "calls.json").exists())
