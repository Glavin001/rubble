# Learned impact — data, training, and oracle comparison

Side track: residual MGN-T on top of quasi-static (`engine/docs/learned-stress-solver-plan.md`).
**Does not change `blast_parity`.** The combined-data checkpoint and recorded-frame
visualizer are documented in [COMPARISON.md](COMPARISON.md). From the repository root,
run `engine/stress-oracle/scripts/view_learned.sh` and open the printed localhost URL.
The viewer uses exact PyTorch inference on recorded inputs; it is not a free-running
simulation. Local data and checkpoints are under `runs/` (Git-ignored).

| Piece | Path |
|---|---|
| Schema | [`SCHEMA.md`](SCHEMA.md) |
| Rust IO / recorder / smoke | `../crates/stress-learned` |
| Peak utilization (oracle) | `stress-ref` `SimDesc.record_peak_utilization` + `RtBond.peak_utilization` |
| Contact load bins | `World::learned_bins` → `chunk_load_bins` / `chunk_contact` (gravity separate) |
| PyTorch | [`python/`](python/) |

## Quick checks (no bulk generate / train)

```sh
cd engine/stress-oracle

# Schema + synthetic + oracle-hook tests
cargo test -p stress-learned --release

# Synthetic tiny event
cargo run --release -p stress-learned --bin learned_impact_smoke -- /tmp/learned-impact-smoke

# Real oracle event (2 frames; default: b5 wall @ 20 m/s)
cargo run --release -p stress-learned --bin learned_impact_record -- \
  --scene wall --frames 2 --out /tmp/li-wall

# Smoke event set (~4 events, ~15–30 s): tension, cantilever, wall@10, wall@20
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --smoke --out /tmp/li-smoke-set

# Mini catalog (opt-in; ~1 min): walls + diverse structures
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --mini --confirm-mini --out /tmp/li-mini-set

# Wall-heavy M2 set (~4–6 min @ ≈8 s/wall-frame): speed sweep + 8-frame rollouts
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --walls --confirm-walls --out /tmp/li-wall-set

# Diverse prove set (~1.5 min): push/ram/punch walls, fracture cantilevers, arch, frame, house
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --prove --confirm-prove --out /tmp/li-prove-set

# Diversity v1 (~35 events / ~230 frames; see DIVERSITY.md). Resume skips valid dirs.
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --diversity --confirm-diversity --out /tmp/li-diversity-v1
# See TRAINING.md for paper-aligned likes/dislikes + holdout protocol.

# Python: smoke + overfit / train on a generated set
cd learned-impact/python
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m learned_impact.train --params   # ~104k params at h=48 p=16
python -m learned_impact.train --smoke
python -m learned_impact.train --overfit --dataset /tmp/li-smoke-set --steps 80 \
  --structure-type wall_b5 --out /tmp/li-overfit-wall
# Mini-set train with one held-out structure type
python -m learned_impact.train --train --dataset /tmp/li-mini-set --steps 400 \
  --holdout-structure-type masonry_arch --out /tmp/li-train-mini
# MLP vs MGN-T on the same overfit set
python -m learned_impact.train --compare-baselines --dataset /tmp/li-smoke-set \
  --structure-type wall_b5 --steps 120 --out /tmp/li-compare
```

Fixture: [`fixtures/tiny_event`](fixtures/tiny_event) (synthetic). Generated sets go under `--out` (not committed).

## GPU inference weights (Slang / WebGPU / Metal)

This export section describes the original legacy network only. The newer
physical-input V12/V15 checkpoints run through PyTorch, including Metal/MPS;
they do not have matching Slang export, live World integration or validated
closed-loop behavior. See [COMPARISON.md](COMPARISON.md) for current measured
accuracy and the remaining integration work.

Training exports a **Slang-packed** blob (`learned-impact-slang/1`) as `weights.bin` + `weights.json`
via `learned_impact.export.export_weights` / `export_slang_weights`. That layout matches
`engine/slang/kernels/learned_impact.slang` (H=48, P=16 defaults).

- Fracture lab loads the checked-in fixture at
  `engine/crates/engine-webgpu/fracture/assets/learned-impact/`.
- Native Metal can bind the same `weights.bin` as a storage buffer (plan M0); no second format.
- Raw `state_dict` concatenation is also written as `weights_pytorch.bin` for debugging — not for the kernels.
- Smoke: packed float count must equal `slang_weight_count(h, p)` (see `python/tests/test_export_slang.py`).
