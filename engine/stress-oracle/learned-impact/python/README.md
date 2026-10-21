# Learned-impact PyTorch (smoke only)

Trains the residual MGN-T impact model later; this package is scaffolding + CPU smoke.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m learned_impact.train --smoke
```

Full dataset generation and long training are **not** in scope here.
Default smoke uses `H=8`, `P=4`. Lean production shape is `H=48`, `P=64`.

## Impact-outcome model (P2, `learned_impact.outcome`)

One call per routed impact island per frame. Plan: `docs/learned-stress-solver/learned-stress-solver-plan.md`.
Data contract: [`../OUTCOME_CALLS_SCHEMA.md`](../OUTCOME_CALLS_SCHEMA.md) (`learned-impact-outcome-calls/1`, mirrored by
`learned_impact/outcome/schema.py`). **Until P1 data exists, everything trains only on the synthetic
toy (`outcome/synthetic.py`): no accuracy claims.**

```sh
nvm use   # not needed for Python; the venv below is all
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' scipy
.venv/bin/python -m pytest -q tests                      # CPU, plus MPS where available

# validate one recorded event (P1 output)
.venv/bin/python -c "from learned_impact.outcome.schema import load_event; load_event('<event_dir>')"
# fill ensemble frequencies / members into aligned calls, in place
.venv/bin/python -m learned_impact.outcome.ensemble <dataset>

# train (default = the P0 cost-feasible point: W32, rounds 2/1/2/1/2, variant A)
.venv/bin/python -m learned_impact.outcome.train --dataset <dataset> --out ../runs/<job> --steps 20000
#   --reference          W64, rounds 3/2/2/2/3 (accuracy reference)
#   --variant B          residual over the blast_parity QS frame outcome
#   --seeds 0,1,2        three seeds, runs/<job>/seed<k>/
#   --noise 0.01 --pushforward 0.5   GNS noise on carried columns, carried-state pushforward
#   --energy-tolerance 0.05          KE bound tolerance (P0: ledger residual up to 4.7% E0)
#   --break-mode sample              hash-seeded sampling ablation (default: threshold τ = 1)
# smoke on synthetic data
.venv/bin/python -m learned_impact.outcome.train --synthetic 64 --out ../runs/syn-smoke --steps 300

# check in on a run without interrupting it
cat ../runs/<job>/status.json; tail -5 ../runs/<job>/progress.log
```

Layout: `mgn/` holds the blocks adapted from NVIDIA PhysicsNeMo @ `c8f68428` (Apache-2.0, see
`mgn/LICENSE-Apache-2.0.txt`; BSMS also from BSMS-GNN): `MeshGraphMLP`, edge/node blocks on
`index_add_`, the BSMS layers (fixed `degree(num_nodes)`) and hierarchy (deque BFS, coarse-edge
masking across cracks, rotation-stable seeds). `tests/test_mgn_parity.py` checks them against
`../fixtures/physicsnemo_parity.npz`, generated once by `../scripts/gen_physicsnemo_fixture.py`
in a throwaway venv (the only code that imports `inspiration/`; not run in tests or CI).
`outcome/`: `schema`, `synthetic`, `batch` (collation + per-call hierarchy masks), `model`,
`decode` (momentum projection, energy bound, breaks), `losses`, `metrics`, `pushforward`, `train`.
