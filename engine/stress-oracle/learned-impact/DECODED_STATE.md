# Decoded state (`learned-impact-decoded-state/1`)

The model's decoded outcome of one call (one routed island, one frame), keyed by stable keys.
Two consumers apply it at **window end, with the island's contacts filtered for that frame**
(P0's validated `window_reduced` boundary):
- a DAgger restart in P1's recorder (`--restart <file> --parent <event>`);
- the closed-loop rollout harness (P0's coupled ceiling harness).

Writer: `python/learned_impact/outcome/state_io.py` (`decoded_states`, `write`, `read`,
`from_targets` for the perfect model).

One JSON file per call, written atomically (a temp file, then rename). Floats are written
as round-trip f64 decimals.

```json
{
  "schema_version": "learned-impact-decoded-state/1",
  "calls_schema": "learned-impact-outcome/1",
  "event_id": "…", "call_index": 3, "frame": 12, "island": 0,
  "meta": {"checkpoint": "…", "sha256": "…", "decode": {…}},
  "flags": {"free": true, "energy_bound_active": false, "energy_bound_infeasible": false},
  "chunks":    [{"key": 4294967301, "anchored": false, "v": [3], "w": [3], "dx": [3], "drot": [3]}],
  "impactors": [{"key": 7, "v": [3], "w": [3], "dx": [3], "drot": [3], "p_perforate": 0.93}],
  "bonds_broken": [{"key": 4294967310, "bin": 4}],
  "bond_damage":  [{"key": 4294967311, "damage": 0.12}]
}
```

| Field | Meaning |
|---|---|
| `key` | the stable chunk / impactor / bond key of the calls schema |
| `v`, `w` | frame-end linear and angular velocity, world axes (m/s, rad/s) |
| `dx` | the body's centre-of-mass displacement over the frame, world axes (m); new position = start + `dx` |
| `drot` | rotation over the frame as a world-axis rotation vector (rad); new orientation = exp(`drot`) · start orientation |
| `anchored` | anchored chunks carry zero motion |
| `bonds_broken` | bonds that break in this frame, with their time bin 1..8 (orders the splits); together with `bond_damage` they partition the call's live bonds exactly |
| `bond_damage` | frame-end damage of every surviving live bond (≥ its frame-start value: monotone) |
| `flags` | `free`: momentum was projected exactly (no anchored chunk, no external work); `energy_bound_*`: the KE bound bound / could not be met without breaking momentum. A consumer logs these |
| `p_perforate` | the regime head, for the consistency check and the fallback gate |

Every value is already conservation-projected and energy-bounded by the decode. A consumer
applies it as given and must not re-project.
