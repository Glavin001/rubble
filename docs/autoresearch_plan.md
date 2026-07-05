$autoresearch
Goal: Reduce WebGPU stress-step latency on the fixed 20k Stress Mixed browser path, optimizing for lower step_ms_median while watching step_ms_p95 for jitter.
Scope: crates/rubble3d/src/**/*, crates/rubble-primitives/src/**/*, crates/rubble-wasm/src/lib.rs, web/src/3d/**/*, crates/rubble-scenes/src/**/*
Metric: step_ms_median
Direction: lower is better
Verify: export RUSTUP_TOOLCHAIN=stable WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json; set -o pipefail; npm --prefix web run bench:stress 2>&1 | awk
'/"step_ms_median"/ {gsub(/,/, "", $2); print $2}' | tail -1
Guard: export RUSTUP_TOOLCHAIN=stable WGPU_BACKEND=vulkan VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json; cargo check -p rubble3d -p rubble-wasm && cargo test -p rubble3d --test
wgsl_validation_tests --test gpu_render_path_tests && npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list && npm --prefix web test -- tests/stability-1step.spec.ts
--reporter=list

Validation summary:

- Scope resolves cleanly to the 3D hot path plus shared broadphase/scene source files. I excluded generated crates/rubble-wasm/src/wasm/* artifacts from the writable scope.
- Verify dry run passed and returned a numeric baseline: 92.45000056922436
- Full benchmark baseline from the same path:
    - step_ms_median = 92.94999808073044
    - step_ms_p95 = 111.50000265240669
    - commit 04628b1196e19a09049a24a8b9d180c8999e36c8
- Guard checks passed:
    - cargo check -p rubble3d -p rubble-wasm
    - cargo test -p rubble3d --test wgsl_validation_tests --test gpu_render_path_tests
    - npm --prefix web test -- tests/ground-contact.spec.ts --reporter=list
    - npm --prefix web test -- tests/stability-1step.spec.ts --reporter=list

I left npm --prefix web run bench:stress as Verify, not duplicated inside Guard, because otherwise every iteration pays for the same benchmark twice.
