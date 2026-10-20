# Routing extended by launched-debris reach (2026-10-20)

Decision 5 after P0 ([`p0-feasibility-2026-10-20.md`](p0-feasibility-2026-10-20.md)): close
P0's routing miss. A plug punched out mid-frame reaches the next structure within the same
frame, while that structure is still the engine's.

## The switch

`World::debris_reach_routing`, set by `STRESS_DEBRIS_ROUTING=1`, is **off by default**. It is
read in `select_islands` (`stress-ref/src/world_coupling.rs`), so the CPU coupled world and
the GPU world's host routing both take it. When an impactor's swept bounds reach a bonded
cluster closing faster than the tiered impact speed, that cluster's swept bounds grow by
`2 (v_impactor − v_cluster) dt` along that velocity. A piece knocked out by a heavier body
leaves at no more than twice the closing speed: that is the elastic bound. Any structure
those pieces can reach within the frame then overlaps the cluster and joins its island.
Under `STRESS_TIERED_JOIN=fast` such a join is kept, even though the two structures close
slowly. Nothing else changes.

Test: `stress-rubble/tests/debris_routing.rs`. On `wall_row_late(2)`'s strike frame the
island is {ram, wall 1} with the switch off and {ram, wall 1, wall 2} with it on. Neither
setting routes anything before the strike.

## Guard

- **Off: 32/32 digests identical** (`scripts/destruction_peak.py guard`).
- **On: 28/32.** Only `wall_row2_late` and `wall_row2_late2` change, in both the realtime
  and the accuracy sets. The house, the towns and the free walls are bit-identical.

## Measured

Unperturbed runs (ram m/s / fragments / broken bonds). For reference, the standalone oracle
gives `wall_row_late` 24.03 / 128 / 397.

| Scene | Path | Off | On |
|---|---|---|---|
| `wall_row_late` | CPU f64 coupled to rubble | 28.33 / 77 / 259 | **23.58 / 121 / 384** |
| `wall_row_late` | GPU, full features, routed (`realtime`) | 28.59 / 73 / 249 | **23.21 / 136 / 406** |
| `wall_row_late` | GPU `blast_parity_routed` | 31.65 / 41 / 199 | 31.14 / 41 / 197 |
| `wall_row_late2` | CPU coupled | 27.48 / 100 / 322 | **23.62 / 121 / 385** |
| `wall_row_late2` | GPU `realtime` | 28.44 / 81 / 267 | **23.44 / 132 / 401** |
| `wall_row_late2` | GPU `blast_parity_routed` | 31.20 / 55 / 221 | 29.34 / 65 / 255 |
| `wall_row` (5 cm), free walls | every path | unchanged | unchanged |

Ensembles: the same 16 ±1 mm / ±0.1% members as P0, against the oracle ensemble.
Skill = ram / partition / sizes. The calibrated rule passes a metric when its skill is at
most max(1.25, a held-out oracle member's own mean); that mean is ram 1.12–1.25 and sizes
1.26–1.36 on these scenes.

| Scene | Path | Off: ram median [min, max], skill | On: ram median [min, max], skill | Verdict on (calibrated) |
|---|---|---|---|---|
| `wall_row_late` | CPU coupled | 23.98 [22.06, 27.26]; 2.10 / 1.01 / 1.29 | 23.56 [22.53, 24.92]; **1.01 / 0.99 / 1.14** | pass (was fail) |
| `wall_row_late2` | CPU coupled | 24.08 [22.45, 28.10]; 2.59 / 1.02 / 1.69 | 23.36 [22.13, 24.70]; **1.20 / 0.99 / 1.28** | pass (was fail) |
| `wall_row_late2` | GPU `realtime` | 23.62 [21.57, 28.42]; 2.51 / 1.03 / 1.55 | 23.59 [21.53, 24.25]; **1.20 / 1.01 / 1.25** | pass (was fail) |
| `wall_row_late2` | GPU `blast_parity_routed` | 23.99, 274 fragments; 2.04 / 1.26 / 2.37 | 24.02, 268 fragments; 2.05 / 1.27 / 2.35 | fail (unchanged) |
| free walls, `wall_row` | all | identical | identical | unchanged |

Oracle ensembles for comparison: `wall_row_late` 23.47 [22.59, 24.86], 190 fragments;
`wall_row_late2` 23.32 [21.92, 24.70], 198 fragments.

## Reading

- **The miss is closed.** With the switch on, the CPU coupled world and the full-feature GPU
  routing reproduce the oracle on the late wall rows. The ram's tail to 27–28 m/s, from
  members where wall 2 took a rigid exchange, is gone.
- **What remains of the 31.65 / 41 gap is entirely `blast_parity_routed`'s material
  switches.** Its islands now hold the right bodies, but with softening, crack contact,
  rate effects and dashpots off it still over-fractures on the ensemble (268 vs 198
  fragments) and keeps the plug clean on the symmetric run (41 fragments). Fixing that means
  turning model features on in islands under that profile. That is a fidelity/cost decision,
  not routing.
- **Cost.** The island gains the downstream structure one frame earlier. On these scenes,
  that structure would have joined the next frame anyway. In a town, a fast impact's island
  now reaches up to `2u dt` further along the impact: about 1.3 m at 40 m/s. That is why the
  switch stays off until it is measured on the town tiers.

Raw: `learned-impact/runs/p0-debris/` (gitignored; specs in `runs/debris_spec.json`), analysis
`python -m learned_impact.p0_ceiling runs/p0-debris/out`.
