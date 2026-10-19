# Stress + learned break gate prototype (2026-10-19)

## Goal

Wire learned fail-bin decisions into the **stress-solver** path (cluster rigid bodies → bond break → connected-component split), not D6 soft welds. HUD-only `learned` mode stays; new lab mode **`stress-learned`**.

## Hook map (file:line, approx.)

| Step | Where | Role |
|---|---|---|
| Analytic disconnect | `stress-ref/src/statics.rs` `commit_damage` (~1179) | QS cascade damage; `fracture=false` freezes disconnects |
| External break + split | `stress-ref/src/solver.rs` `break_bonds` (~1865) → `split_cluster` (~2047) | **Primary inject point** — already accepts an external verdict list |
| GPU punch path (same API) | `stress-gpu/src/world.rs` ~517 | `mirror.break_bonds` + `rebuild` |
| Resident QS disconnect | `stress-gpu` `commit_resident` / island flags → `resident_frame.rs` split (~786+) | Analytic GPU cascade; left alone when gate clears `config.fracture` |
| **New** external apply | `stress-gpu/src/world.rs` `GpuWorld::apply_external_breaks` | download → `break_bonds` → rebuild → engine write-back |
| **New** live pack | `stress-learned/src/live.rs` `pack_live_features` | NODE_IN=46 / EDGE_IN=28 from mirror |
| **New** wasm bridge | `stress-web/src/lib.rs` | `stress_set_learned_gate`, `stress_learned_capture`, `stress_break_edges` |
| Lab hybrid | `fracture/main.mjs` `stressTick` + `loadStress` | capture → MGN-T (lab device) → `breakLearnedEdges` |

## What landed

1. **`stress-learned::live`** — pack connected bonds; `fail_edge_indices`; unit tests.
2. **`GpuWorld::apply_external_breaks`** — opt-in; does not change `Profile::BlastParity`.
3. **`stress-web` exports** — learned gate flag (clears `config.fracture` after profile apply), feature snapshot from last `view`, pending edge breaks via JSPI `stress_break_edges`.
4. **Fracture lab** — Bonds menu option **Stress + learned break gate**; uses lab WebGPU for MGN-T, stress-web canvas for physics. CPU gate on mirror readback (documented).
5. **`learned-impact.mjs`** — `graphFromLive`, `normalizeLiveFeatures`, `failEdgeIndices`, `resync`.

## Feature gaps vs SCHEMA (honest)

From `FEATURE_GAPS` / `stress_learned_feature_gaps()`:

- **`chunk_load_bins`**: all zeros (no `World::learned_bins` on the resident web path yet).
- **`bond_qs_force`**: current `bond.force`, not a cascade-off QS probe.
- **Normalize**: JS applies train-time RMS scales; still distribution-shifted vs oracle frames with real load bins.
- Do **not** invent contact bins to make demos look good — fails will stay conservative until bins are plumbed.

## How to run

```bash
# 1) Compile checks / unit tests
cd engine/stress-oracle
cargo test -p stress-learned live::
cargo check -p stress-gpu -p stress-web

# 2) Rebuild stress-web wasm (required for new exports)
CARGO_PROFILE_RELEASE_DEBUG=0 wasm-pack build engine/stress-oracle/crates/stress-web --release --target web \
  --out-dir ../../../crates/engine-webgpu/generated/stress-web --no-typescript
# or: engine/scripts/build-web.sh

# 3) Optional: point weights at diversity punch holdout (do not overwrite assets until smoke OK)
# cp /tmp/li-diversity-v1-train/holdout_punch/weights.{bin,json} \
#    engine/crates/engine-webgpu/fracture/assets/learned-impact/

# 4) Serve fracture lab and pick Bonds → "Stress + learned break gate"
cd engine/crates/engine-webgpu && python3 -m http.server 8788 --bind 127.0.0.1
# open http://127.0.0.1:8788/fracture/  (Chrome 137+ for JSPI)
```

## What’s left (PR-sized)

1. Plumb **contact/scripted load bins** into the resident frame (or a cheap host mirror) so features match training.
2. True **QS probe** for `bond_qs_force` (scratch static under gravity) without stalling the frame budget.
3. **GPU-resident MGN** on the stress-web device (same buffers as cascade) — drop dual-device + readback.
4. Cap / confidence fallback: OOD → re-enable analytic fracture for those bonds only.
5. WebGPU smoke, then optionally refresh `fracture/assets/learned-impact/` from `/tmp/li-diversity-v1-train/full/`.

## Constraints honored

- No D6 coupling; no `blast_parity` profile changes; no commits; no `inspiration/`; assets not overwritten.
