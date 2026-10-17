---
name: diagnosing-solver-energy-injection
description: Diagnose energy injection, bouncing, jitter, or exploding stacks in the physics solver by instrumenting per-frame kinetic energy against warm-start key survival and contact-manifold churn. Use when a demo scene (pyramid, stack) visibly gains energy, bodies pop or bounce at rest, or a settle test fails with rising tail speeds.
---

# Diagnosing Solver Energy Injection

Energy injection in an AVBD-style solver is almost never "the solver is wrong per step" — it's usually **contact identity instability**. When a contact's `(body pair, feature_id)` key changes between frames, warm-start misses, `lambda` resets to 0 and `penalty` to `k_start` while resting penetration persists. Re-arming a stiff penalty spring against existing penetration injects kinetic energy. At hundreds of misses/frame this is a continuous pump.

## Diagnostic workflow

1. **Reproduce natively, not in the browser.** Build the exact demo scene in a Rust test (see `crates/rubble3d/tests/pyramid_energy_diagnostic.rs` — reuse it directly for pyramid issues, or copy its structure for other scenes). Mark it `#[ignore = "diagnostic instrumentation, not a gate"]`.

2. **Instrument per frame:**
   - Kinetic energy (translational + rotational) via `support::compute_metrics`.
   - **Warm-start key survival**: fraction of this frame's contacts whose `(min(a,b), max(a,b), feature_id)` key existed last frame. A resting pile should be ~1.0. Values of 0.5–0.8 mean identity churn.
   - **Churn classification**: for pairs present in both frames whose feature-ID set changed, split by whether only the low byte changed (clip-vertex index drift) or the prefix changed (SAT axis type / ref axis / inc axis flip — see the feature encoding near line 852 of `crates/rubble3d/src/gpu/narrowphase_wgsl.rs`). Prefix churn invalidates the whole manifold at once and is the worse failure.
   - Max `|depth|` and max `|lambda_n|` to spot correction spikes.

3. **Run and read the ramp:**

```bash
cargo test -p rubble3d --test pyramid_energy_diagnostic --release -- --ignored --nocapture
```

A sustained monotonic KE climb over many frames while the scene is nominally at rest (e.g. 11 J → 255 J over 100 frames), followed by a partial collapse and repeat, confirms the churn pump. A one-time KE spike instead points at a bad impulse (restitution, CCD, penetration recovery), not churn.

4. **Correlate.** Bucket settle-window frames by KE rising vs falling and compare mean survival/churn. Note: churn is often a *constant background*, so rising and falling buckets can look identical — that still implicates churn (constant injection fighting friction), not exonerates it.

## Pitfalls

- `CONTACT_FLAG_WARMSTARTED` is useless as post-step telemetry: the dual-update kernel overwrites `flags` wholesale every solve (`avbd_solve_wgsl.rs`, end of `dual_update_contact`). Measure survival by key instead.
- Contact download is opt-in: call `world.fetch_last_contacts()` (or `world.set_events_enabled(true)`) — `last_contacts()` is empty on the default hot path.
- Distinguish the two churn mechanisms; they need different fixes:
  - **Prefix churn** → SAT axis flip-flop on near-degenerate configurations. Fix: hysteresis (prefer previous frame's axis unless a challenger wins by a real margin, edge-win margin).
  - **Low-byte churn** → clip-vertex feature IDs derived from loop order (`final_features[final_count] = i`). Fix: geometry-derived stable IDs (e.g. incident-face corner index), not enumeration order.

## Success metric

After a fix, re-run the diagnostic: settle-window key survival should be ≥ 0.99 and the tail KE ramp gone (KE monotonically decaying after settle).
