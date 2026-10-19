# Steering rubble with learned-impact

**Target architecture:** the stress-solver path — **cluster rigid bodies** that stay rigid until bonds fail, then **split** into child clusters (GPU layout / spawn). The **learned model decides which bonds fracture** (fail-bin / util), not soft D6 AVBD welds.

Today **Bonds → Learned impact** is HUD-only on the unbonded rubble path. **Bonds → Stress solver** already does cluster rigid + QS/cascade + splits. Coupling = wire learned preds into that break decision.

## Do not use

- **D6 / soft welds** for learned steering — too jello-like; wrong mechanical model for masonry clusters.
- Replacing the whole AVBD *contact* solve with the GNN.

## Recommended path

### 1. Feature parity (required)
Feed the same tensors training saw: `chunk_load_bins`, QS/contact forces, joint/bond state, velocities — from the stress-gpu / stress-ref mirror each tick (not sparse pack zeros). Otherwise fail-bins stay near “never fail.”

### 2. Fracture gate on the stress path
Keep: QS (or stress-gpu QS) for gravity / resting load; cluster rigid bodies; connected-component **split** when bonds cut.

Replace (or gate): the analytic bond-break step with learned **fail-bin / peak-util**. On impact islands / when `rigid_assumption_fails`, run residual MGN-T and break bonds the network marks; then run the existing split/layout/spawn pipeline unchanged.

Optional viz: tint bonds by util before breaks land.

### 3. Fallback
Low-confidence or OOD bonds → existing stress cascade / capacity break (never silent wrong). Do not fall back to D6.

### 4. Demo wiring
Fracture lab **Learned** should eventually share the stress-solver world (clusters + rubble adoption), not `withoutJoints`. HUD-only pack graph is a speed/weight smoke only.

### 5. Engine residency
Same `learned-impact-slang/1` weights in the resident GPU frame with the cascade (M0); fixed-order CSR; bit-identical repeats.

## Acceptance
- A punched/rammed wall or house **splits cluster rigid bodies** where learned fail preds fire, matching stress-mode look (hard pieces, not jelly).
- Holdout punch F1 stays meaningful on teacher-forced eval; coupled demo is qualitative + optional oracle fragment/speed compare.

## Prototype status (2026-10-19)

Incremental **Stress + learned break gate** is in-tree (see `findings/stress-learned-gate-2026-10-19.md`):

- Hook: `ReferenceSolver::break_bonds` ← `GpuWorld::apply_external_breaks` ← stress-web `stress_break_edges`.
- Lab mode **`stress-learned`**: clears analytic `fracture` for that session only (BlastParity profile unchanged); JS runs MGN-T on the lab device from a CPU feature capture, then breaks.
- Gaps: `chunk_load_bins` still zero; `bond_qs_force` = live `bond.force`. Next: bins + resident GPU MGN.
