# Baselines

Measurements kept over time so progress can be compared against a fixed
reference: oracle engines (three-avbd, PhysX, ...) and rubble itself, on a
named machine.

- One file per measurement run, named `<what>-<YYYY-MM-DD>.<md|json>`. Add a new
  file for a new run instead of editing an old one.
- Record the hardware and adapter (for example "M3 Max, apple · metal-3") and the
  tool or page that produced the numbers.
- Keep them anonymous: no home-directory paths, user names or host names.
- Speed numbers are only comparable on the same machine and API (DECISIONS.md 7).

| File | Run |
|------|-----|
| `three-avbd-oracle-baseline-2026-10-13.md` | three-avbd as the local oracle: accuracy cases and Chrome WebGPU speed, M3 Max |
| `three-avbd-chrome-small-steady-2026-10-13.json` | three-avbd `bench3d.html` small + steady tiers, Chrome WebGPU, M3 Max |
| `three-avbd-chrome-paper-large-steady-2026-10-13.json` | three-avbd `bench3d.html` paper + large + steady tiers, Chrome WebGPU, M3 Max |
