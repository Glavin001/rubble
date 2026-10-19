# Background pipeline status — learned-impact (2026-10-19)

## Done

### Catalog / diversity v1
- Added `builders::cantilever_crackling` + `cantilever_root_break_load` in `stress-ref` (tip load held, `fracture=true`). Analytic `cantilever` benchmark unchanged (`fracture=false`).
- Implemented `diversity_jobs()` — **35 events / ~264 frames** (walls soft/push/ram/punch/ring-out, houses, 9 crackling cantilevers, arch stand+keystone, frame sudden+gradual, shear/tension anchors).
- CLI: `--diversity` / `--diversity-v1` + `--confirm-diversity` (default out `/tmp/li-diversity-v1`).
- `--prove` still **17** events; prove cantilevers now use crackling (fracture on).

### Recording
- Fail labels: **first break wins** within a frame (ring-out / multi-break no longer overwrites bin).
- `validate_event` now requires full SCHEMA frame set (`chunk_contact`, `chunk_gravity_local`, `bond_force_prev/end`, `bond_fail_mode`, `bond_joint_state_end`) + fail_mode range 0–5.

### Parallel / throughput
- Rayon path unchanged in ownership model (per-job `Scene` clone → own `World`); documented in `run_jobs_with`.
- Per-job timing + frame-based ETA logging; resume still skips valid event dirs (`--no-resume` to force).

### Docs
- `DIVERSITY.md` implementation status + soft wall speeds 1–4 m/s.
- `README.md` diversity generate command.

### Generate diversity v1 (this pass)
- Command: `--diversity --confirm-diversity --out /tmp/li-diversity-v1 --jobs 16`
- **35 events · 264 frames · 112.7 s wall** (~1.9 min) · ~115 MB · log `/tmp/li-diversity-v1-generate.log`

### Train + holdouts (this pass)
- MPS, H48/P16 MGN-T, 1000 steps × 4 runs. Finding: [`diversity-v1-train-2026-10-19.md`](diversity-v1-train-2026-10-19.md)
- Punch load holdout F1 **0.756** (pass). House structure F1 **0.230** (fail vs train 0.668). Long cantilever F1 **0.404** (weak / unstable).
- Weights under `/tmp/li-diversity-v1-train/{holdout_punch,holdout_house,holdout_cantilever_long,full}/`
- Trainer: `--device` + `--holdout-event-prefix` + `metrics.json` export in `python/learned_impact/train.py`
- Fracture assets **not** overwritten (Slang pack validated on `/tmp/.../full/` only)

## Files touched
- `engine/stress-oracle/crates/stress-ref/src/builders.rs`
- `engine/stress-oracle/crates/stress-learned/src/generate.rs`
- `engine/stress-oracle/crates/stress-learned/src/oracle.rs`
- `engine/stress-oracle/crates/stress-learned/src/validate.rs`
- `engine/stress-oracle/crates/stress-learned/src/lib.rs`
- `engine/stress-oracle/crates/stress-learned/src/bin/learned_impact_generate.rs`
- `engine/stress-oracle/crates/stress-learned/tests/diversity_catalog.rs` (new)
- `engine/stress-oracle/learned-impact/DIVERSITY.md`
- `engine/stress-oracle/learned-impact/README.md`
- `engine/stress-oracle/learned-impact/python/learned_impact/train.py` (MPS + event-prefix holdout + metrics.json)
- `engine/stress-oracle/learned-impact/findings/bg-pipeline-status.md` (this file)
- `engine/stress-oracle/learned-impact/findings/diversity-v1-train-2026-10-19.md` (new)

## Tests run
| Command | Result |
|---|---|
| `cargo test -p stress-learned --release` | **pass** (prior pass; catalog + crackling) |
| `learned_impact_generate --smoke` | **ok** (prior) |
| `learned_impact_generate --diversity --confirm-diversity --jobs 16` | **ok** · 35 / 264 · 112.7 s |
| `learned_impact.train` punch / house / cantilever_long / full @ 1000 steps MPS | **ok** (see finding) |
| `pytest tests/test_export_slang.py` | **pass** |

## Left / next
1. ~~Generate diversity v1~~ **done**
2. ~~Train + holdouts + finding~~ **done** (house transfer still weak)
3. More house coverage + cantilever early-stop / tip_load calibration
4. Optional rollout FT; optional fracture-asset copy after WebGPU smoke
5. Stretch 90-frame clips; multi-host sharding (resume enough for single host)

## Blockers
None for pipeline. House holdout fails the F1-within-0.1 gate — data/model gap, not tooling.

## Constraints respected
- No `blast_parity` changes; no commit; no `inspiration/`; no lossy physics shortcuts; fracture `assets/learned-impact/` unchanged.
