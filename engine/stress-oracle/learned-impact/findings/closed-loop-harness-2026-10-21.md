# Closed-loop provider harness: built and checked (2026-10-21)

This was requested by P2, the model session, as the acceptance test that per-frame metrics
can't replace. `learned_p0 ceiling --variant provider` runs P0's coupled ceiling harness with
the island outcome supplied by a provider:

- an in-process **oracle** provider;
- a **file** provider (an external process, `../PROVIDER_PROTOCOL.md`);
- the host **punch**.

Every routed frame applies the provider's `learned-impact-decoded-state/1` at window end,
reduced state, with the island's contacts filtered and 1/60 frames
(`World::step_frame_with_provider`, `World::apply_decoded` in `stress-ref/src/world_handoff.rs`).
The runs go to the end and write the usual outcome, so the skill tables apply unchanged.

## Checks

1. **In-process oracle provider = the `window_reduced` ceiling.**
   - Unperturbed: free wall 35.034 / 6 / 31 and wall row 23.622 / 133 / 401, both identical
     to `window_reduced`.
   - Test `stress-rubble/tests/closed_loop_provider.rs`: after the wall row's first frame,
     the oracle's exact break set and the ram's exact frame-end velocity and position.
2. **The file path reproduces the in-process provider exactly.**
   - `--provider files --echo-oracle` with `scripts/echo_provider.py`, on the wall row:
     12 calls, the end state bit-identical. The f64 values round-trip through JSON.
3. **Ensembles** (16 members ±1 mm / ±0.1%, against the oracle ensemble). Skill = ram /
   partition / sizes. The calibrated rule passes a metric when its skill is at most
   max(1.25, a held-out oracle member's own mean).

| Scene | Oracle provider: ram, fragments (median) | Skill | Verdict | Punch provider: ram, fragments | Punch skill |
|---|---|---|---|---|---|
| `free_wall` | 31.98, 78 (oracle 31.98, 78) | 1.20 / 0.95 / 0.95 | pass | 29.64, 92 | 5.52 / 1.41 / 3.88: fail |
| `wall_row` | 22.95, 200 (22.95, 200) | 1.08 / 0.94 / 1.04 | pass | 9.34, 268 | 23.1 / 1.34 / 3.14: fail |
| `wall_row_late`, debris routing on | 23.23, 186 (23.47, 190) | 1.05 / 0.97 / 1.12 | pass | 11.51, 366 | 14.8 / 1.50 / 3.35: fail |
| `free_wall_late2`, debris routing on | 32.24, 78 (32.28, 77) | 1.10 / 1.00 / 1.24 | pass | 29.48, 91 | 2.53 / 1.20 / 1.85: fail |
| `wall_row_late2`, debris routing on | 23.23, 189 (23.32, 198) | 1.10 / 0.98 / 1.27 | pass calibrated (sizes 1.27 against the oracle's own 1.36) | 13.17, 365 | 14.5 / 1.49 / 3.82: fail |

The host punch here breaks the plug and lets the engine take the frame with its contacts.
It fails on every scene, as the GPU punch did in P0. It differs in detail: on the wall rows
it lets the ram tear through both walls at a much lower exit speed.

## Limits

- **One call per frame holds every routed body.** The island is not yet split into
  connected components, so `island` is always 0. The M5 scenes have one island.
- **The request carries the frame-start state as plain JSON** (`state`), with
  `call_record: null`, until the Rust call builder (`stress-learned/src/outcome_calls.rs`)
  writes the `learned-impact-outcome/1` record into the request directory.
- **The output's energy ledger misses the provider's dissipation**; its balance residual is
  not meaningful in this mode.
- **Late scenes need `STRESS_DEBRIS_ROUTING=1`**: without it the island misses the second
  wall, so the provider never sees it.

Raw: `learned-impact/runs/p0-provider/` (gitignored), `../p0-data/provider.json`.
