# Diversity v1 generate + train (2026-10-19)

## Generate

```bash
cd engine/stress-oracle
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --diversity --confirm-diversity --out /tmp/li-diversity-v1 --jobs 16
```

| Stat | Value |
|---|---|
| Events | **35** |
| Frames | **264** |
| Wall time | **112.7 s** (~1.9 min) @ 16 Rayon threads |
| Disk | **~115 MB** uncompressed |
| Resume | not needed (clean `/tmp/li-diversity-v1`) |
| Log | `/tmp/li-diversity-v1-generate.log` |

Parallelism cut the ~25–35 min serial ETA to under 2 min on this host (M-series, 16 logical CPUs). Heavy scenes still ~3–9 s/frame when contended.

## Train setup

- Package: `learned-impact/python` (venv + MPS)
- Arch: MGN-T, H=48, P=16, **103,954** params
- Steps: **1000** each run, Adam lr=1e-3, seed 0
- Device: **mps**
- Trainer extras used this pass: `--device auto`, `--holdout-event-prefix` (long cantilever), metrics.json next to weights

### Commands

```bash
cd engine/stress-oracle/learned-impact/python && . .venv/bin/activate

# 1) Punch wall speeds (load holdout)
python -m learned_impact.train --train --dataset /tmp/li-diversity-v1 --steps 1000 --device auto \
  --holdout-load-type ram_25mps --holdout-load-type ram_35mps --holdout-load-type ram_45mps \
  --out /tmp/li-diversity-v1-train/holdout_punch

# 2) Masonry house structure holdout
python -m learned_impact.train --train --dataset /tmp/li-diversity-v1 --steps 1000 --device auto \
  --holdout-structure-type masonry_house \
  --out /tmp/li-diversity-v1-train/holdout_house

# 3) Long cantilever event-prefix holdout
python -m learned_impact.train --train --dataset /tmp/li-diversity-v1 --steps 1000 --device auto \
  --holdout-event-prefix cantilever_long \
  --out /tmp/li-diversity-v1-train/holdout_cantilever_long

# 4) Full data (no holdout) — Slang export candidate
python -m learned_impact.train --train --dataset /tmp/li-diversity-v1 --steps 1000 --device auto \
  --out /tmp/li-diversity-v1-train/full
```

Wall times: punch ~31 s · house ~19 s · cantilever ~17 s · full ~13 s (~1.3 min total train suite).

## Metrics

Gates from `DIVERSITY.md` / `TRAINING.md`: punch holdout F1 ≳ 0.7; house F1 within ~0.1 of train-family; cantilever only meaningful once breaks exist.

| Run | Train frames | Hold frames | Train last loss | Train fail F1 | Holdout loss | **Holdout fail F1** | Holdout util MAE | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Punch loads | 240 | 24 | 0.058 | 0.597 | 0.230 | **0.756** | 0.165 | **pass** (F1 ≥ 0.7) |
| `masonry_house` | 208 | 56 | 0.044 | 0.668 | 1.238 | **0.230** | 0.862 | **fail** (ΔF1 ≈ 0.44 ≫ 0.1) |
| `cantilever_long` | 244 | 20 | 0.050 | 0.595 | 0.509 | **0.404** | 0.196 | weak (mid-run F1 peaked ~0.80 then dropped) |
| Full (no holdout) | 264 | — | 0.293 | — | — | — | — | loss ↓, finite |

Notes:
- Punch generalization looks healthy: dense soft/push/ram coverage transfers to 25/35/45 m/s.
- House structure transfer is the hard gate (matches plan ~45% confidence); util MAE on house holdout is large.
- Long-cantilever F1 oscillated (0.30–0.80 during training) — likely rare fails + small hold set (20 frames); needs more length coverage or early-stop on holdout.

Per-run JSON: `/tmp/li-diversity-v1-train/*/metrics.json`.

## Checkpoints / weights

| Path | Contents |
|---|---|
| `/tmp/li-diversity-v1/` | Dataset + `manifest.json` |
| `/tmp/li-diversity-v1-train/holdout_punch/` | `state_dict.pt`, `weights.bin` + `weights.json` (Slang), `metrics.json` |
| `/tmp/li-diversity-v1-train/holdout_house/` | same |
| `/tmp/li-diversity-v1-train/holdout_cantilever_long/` | same |
| `/tmp/li-diversity-v1-train/full/` | full-data Slang pack |

Slang pack check on `full/`: `format=learned-impact-slang/1`, **102839** floats = `slang_weight_count(48,16)`. `pytest tests/test_export_slang.py` **pass**.

### Fracture demo assets

**Not updated.** Checked-in `engine/crates/engine-webgpu/fracture/assets/learned-impact/` left untouched. Candidate pack is `/tmp/li-diversity-v1-train/full/weights.{bin,json}` (or punch holdout if preferring best fail F1). Copy only after fracture-lab smoke (load + one impact frame) confirms no regression.

## Next gaps

1. **House transfer** — more house trajectories (speeds / openings) before claiming structure generalization; consider longer rollouts on houses only.
2. **Cantilever early-stop / more arms** — holdout F1 unstable; calibrate tip_load_scale vs break location; add more `n` values.
3. **Rollout fine-tune** — teacher-forced only; 4–8 step unroll + input noise (MeshGraphNets pushforward).
4. **Fracture smoke** — optional copy of `full/` weights into fracture assets after WebGPU load smoke.
5. Stretch 90-frame clips still deferred (cost).

## Blockers

None for generate/train. Structure-transfer gate failed on house — expected weakness, not a pipeline stop.

## Constraints

No commits; no `blast_parity` changes; no `inspiration/`.
