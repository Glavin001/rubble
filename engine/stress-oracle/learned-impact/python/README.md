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
Data contract: [`../OUTCOME_CALLS_SCHEMA.md`](../OUTCOME_CALLS_SCHEMA.md) (`learned-impact-outcome/1`, mirrored by
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
#   --pushforward-steps 3            frames chained per pushforward (model's own breaks)
#   --balance uncertainty            learned loss weighting (ablation; default fixed)
#   --batch-budget 20000             nodes + edges per batch (fused AdamW; ~1000 calls/s on MPS)
# offline report: calibrated skill + three-way vs QS / punch (EVALUATION.md for the rest)
.venv/bin/python -m learned_impact.outcome.report --checkpoint ../runs/<job>/best.pt \
    --dataset <dataset> --split val --out ../runs/<job>/report-val.json
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

### Run tracking (Weights & Biases)

Optional: `pip install -e .[tracking]` (pins `wandb==0.30.0`). Every training run (train,
pilot/scaling, overfit, DAgger fine-tunes) then logs to the project `rubble-learned-impact`:
the run name is the run dir's name, the group its path below `runs/` (`v7`, `v7/scaling`),
the tags carry variant, width, rounds, seed, data revision, git SHA and the run's options,
and the config is the full `TrainConfig`. It logs what `history.jsonl` records: each
normalised loss term, LR, |g|, the gradient shares and balancer multipliers per group,
steps/s, every validation metric (per family, gates, admissibility and energy-bound counts)
and the loss curves on fixed train/val samples; the final best/τ/temperature as the summary.
`outcome.report` adds its report as a `<run>-report-<split>` run (summary + artifact).

`--wandb auto|off|offline|online` (train, pilot, overfit_real, report). `auto`, the default,
is `offline` when wandb is installed: runs are written under `<run dir>/wandb/` with no
network and no login. Tracking never stops a run (a W&B error turns it off with one warning).
Runs that predate tracking: `python -m learned_impact.wandb_backfill <run dir or dir of runs> ...`
(skips runs already tracked or backfilled). To upload, once:

    wandb login                       # W&B's own prompt; nothing here reads the key
    wandb sync ../runs/*/wandb/offline-run-* ../runs/*/*/wandb/offline-run-* ../runs/*/*/*/wandb/offline-run-*

and pass `--wandb online` from then on (or set `LEARNED_IMPACT_WANDB=online` for runs started
by scripts, e.g. `scripts/viability.sh`, which keep the `auto` default).

### Videos: oracle vs learned model (stress-viz)

```sh
cd engine/stress-oracle
CARGO_TARGET_DIR=target-p2 cargo build --release -p stress-viz --features learned
# oracle | learned model (closed loop) | perfect-model ceiling, same scene and camera
target-p2/release/stress-viz compare m5_free_wall \
  --variant "oracle:" --variant "learned:engine learned" --variant "ceiling:engine ceiling" \
  --checkpoint learned-impact/runs/<job>/best.pt --duration 0.4 --views fragments \
  --plot broken --fps 15 -o learned-impact/runs/videos/m5_free_wall.mp4
```

`engine learned` runs rubble's engine with every routed island's 1/60 s frame from the model
(`stress_learned::provider_client` spawns `learned_impact.outcome.provider`); `engine ceiling`
uses the oracle through the same handoff. Scenes: any scene JSON (P1 saves one per event) or
`m5_free_wall` / `m5_wall_row` (set `STRESS_DEBRIS_ROUTING=1` for the wall row). The energy
plot does not book the model's dissipation, so use `--plot broken`.
