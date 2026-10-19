# Learned impact — data + training scaffolding

Side track: residual MGN-T on top of quasi-static (`engine/docs/learned-stress-solver-plan.md`).
**Does not change `blast_parity`.** Bulk dataset generation and full training are separate steps.

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

# Mini catalog (opt-in; minutes): add --mini --confirm-mini

# Python: smoke + overfit on a generated set
cd learned-impact/python
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m learned_impact.train --smoke
python -m learned_impact.train --overfit --dataset /tmp/li-smoke-set --steps 80 \
  --structure-type wall_b5 --h 32 --p 8 --out /tmp/li-overfit-wall
# MLP vs MGN-T on the same overfit set
python -m learned_impact.train --compare-baselines --dataset /tmp/li-smoke-set \
  --structure-type wall_b5 --steps 120 --h 48 --p 16 --out /tmp/li-compare
```

Fixture: [`fixtures/tiny_event`](fixtures/tiny_event) (synthetic). Generated sets go under `--out` (not committed).
