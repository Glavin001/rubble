# P0 raw data (2026-10-20)

- `*.jsonl`: one line per GPU cost-probe configuration (`stress_gpu_gnn_probe`, and
  `stress-web`'s `gnn_probe_run` for `chrome_webgpu`).
  - `metal_metallib`: the quiet Metal sweep of all three kernels. Its `fused` rows use each
    group's own largest in-degree as the loop bound.
  - `metal_metallib_tiled_v2`: the final tiled kernel (`float4` accumulators).
  - `metal_wgsl_naga`: the same through WGSL and Naga.
  - `metal_wgsl_naga_arrays`: the first tiled kernel through Naga (array accumulators,
    spilled).
  - `chrome_webgpu`: every kernel in the in-app Chromium (Dawn/Tint); `fused` with the
    level's largest in-degree as a uniform bound.
- `chaos.json`, `energy.json`, `ceiling.json`, `baselines.json`, `skill_noise.json`,
  `gbdt.json`: the analyses' outputs (`learned_impact.p0_*`, `gbdt_baseline`). They are
  summarised in `../p0-feasibility-2026-10-20.json`.
