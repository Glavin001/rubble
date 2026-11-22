#!/bin/sh
# Every build configuration engine-webgpu's kernels come in, checked (cargo check: no
# code generation; about a minute warm). Run it after a change to how kernels are
# built, embedded or looked up (build.rs, kernel_build.rs, kernel.rs, kernel_store.rs,
# explicit_modules.rs, lib.rs's kernel tables): each configuration compiles different
# code (cfg kernels_at_runtime, feature gpu, target wasm32), and a full build of only
# one of them misses the others. From engine/.
set -u
fail=0
run() {
  label="$1"; shift
  if out=$("$@" 2>&1); then
    echo "ok    $label"
  else
    echo "FAIL  $label: $*"
    echo "$out" | grep -E "^error" -A6 | head -20 | sed 's/^/      /'
    fail=1
  fi
}
run "embedded, every target" cargo check --workspace --all-targets
run "kernels at run time (fast), every target" cargo check --profile fast --workspace --all-targets
run "no gpu feature (wgsl_export), embedded" cargo check -p engine-webgpu --no-default-features --lib --bin wgsl_export
run "no gpu feature (wgsl_export), at run time" cargo check -p engine-webgpu --no-default-features --lib --bin wgsl_export --profile fast
run "world wasm, embedded" cargo check -p engine-world-wasm --target wasm32-unknown-unknown
run "world wasm, kernels at run time" env RUBBLE_KERNELS=runtime cargo check -p engine-world-wasm --target wasm32-unknown-unknown --profile fast
run "build-architecture rules" python3 tools/build-time/check.py
exit $fail
