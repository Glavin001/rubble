#!/bin/sh
# Measurement only: the solo core kernel's WGSL built with extra defines (perf doc §8.43),
# for RUBBLE_SOLO_WGSL=<out> (explicit_beams_gpu.rs solo_core_wgsl) in solo_core_check.
#   tools/solo-car/variant.sh OUT.wgsl [-DNAME=1 ...]     (from engine/)
out="${1:?usage: variant.sh OUT.wgsl [-DNAME=VALUE ...]}"
shift
slangc="${SLANGC:-$(command -v slangc || echo "$HOME/.local/bin/slangc")}"
"$slangc" -target wgsl -stage compute -entry explicit_beams_solo_core -o "$out" \
  -DENGINE_WGSL=1 -DEB_LANES=1024 -DEB_AERO=1 "$@" slang/kernels/explicit_beams_solo_core.slang 2>"$out.log" || { cat "$out.log"; exit 1; }
# (build.rs explicit_post: Slang's barrier triple to WGSL's pair.)
sed -i.bak -e 's/storageBarrier(); textureBarrier(); workgroupBarrier();;/storageBarrier(); workgroupBarrier();/g' \
  -e 's/storageBarrier(); textureBarrier(); workgroupBarrier();/storageBarrier(); workgroupBarrier();/g' "$out" && rm -f "$out.bak"
