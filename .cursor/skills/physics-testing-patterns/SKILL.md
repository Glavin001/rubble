---
name: physics-testing-patterns
description: Write and maintain physics tests in this repo - scene settle tests, known-failure gating, diagnostic instrumentation tests, oracle comparisons, and the support harness. Use when adding tests to crates/rubble3d/tests, reproducing a demo bug natively, gating a flaky or not-yet-fixed scenario, or deciding how to assert on simulation behavior.
---

# Physics Testing Patterns

## The support harness

`crates/rubble3d/tests/support/mod.rs` provides the building blocks — use them instead of reinventing:

- `try_world(config)` → `Option<World>`; return early with `eprintln!("SKIP: No GPU adapter found")` when `None` so tests pass on GPU-less machines.
- `add_tracked_body` / `collect_reports` / `scene_report` → per-step `SceneReport3D` with full `SystemMetrics3D` (KE, PE, momentum, max speed, min height).
- Assertion failures should print the last report (`\n{last}`) so failures are diagnosable from CI logs alone.

## Assert on settle windows, not instants

Simulations are noisy during the drop/settle phase. The repo pattern (see `demo_repro_tests.rs`): run N steps, assert only on a tail window (`&reports[N/3..]`), and assert *bands* — min support height, max gap oscillation band, max tail speed — not exact positions. Choose limits from physical reasoning (what would a stable pile do) with headroom, and record the observed value in the assertion message.

## Known-failure gating

A scenario that reproduces a real bug we haven't fixed yet stays in the suite, gated:

```rust
if should_skip_known_failure(
    "test_name",
    "one-line reason + current metric vs limit; owned by <milestone>",
) { return; }
```

Run gated scenarios with `RUBBLE_RUN_KNOWN_FAILURES=1`. Keep the reason string updated as the metric improves (e.g. "tail ~12 vs limit 10") — it's the changelog for the bug. When a fix lands, delete the gate, don't loosen the limit.

## Diagnostic tests (instrumentation, not gates)

For root-causing, write an `#[ignore = "diagnostic instrumentation, not a gate"]` test that prints per-frame telemetry (CSV-ish lines + a summary correlation) rather than asserting. Example: `pyramid_energy_diagnostic.rs` correlates KE ramps with warm-start key survival and manifold churn. Run with:

```bash
cargo test -p rubble3d --test <name> --release -- --ignored --nocapture
```

Keep these checked in — they become the before/after metric when the fix lands.

## Reproduce demo bugs natively

When the web demo misbehaves, rebuild the *exact* scene in a Rust test first (body layout, masses, friction — copy from `build_demo_pyramid_scene` etc.). Native tests iterate in seconds, are immune to browser throttling artifacts, and can download contacts/solver state that the demo cannot.

## API gotchas when touching tests

- Contacts are not downloaded by default: use `world.fetch_last_contacts()` or `world.set_events_enabled(true)` before stepping; `world.last_contacts()` is empty on the default path.
- Oracle comparisons against parry live in `support::parry_oracle`; use tolerance-based comparison there, and reserve bitwise assertions for `determinism_probe_tests.rs`.
- Prefer `cargo nextest run -p rubble3d` for the full suite; single-file runs via `cargo test -p rubble3d --test <file>`.
