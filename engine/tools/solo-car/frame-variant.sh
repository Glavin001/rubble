#!/bin/sh
# Measurement only: today's frame kernel (explicit_beams_frame, aero, no surface) at
# LANES lanes, for solo_core_check --frame-wgsl (perf doc §8.43).
#   tools/solo-car/frame-variant.sh OUT.wgsl LANES [-DNAME=VALUE ...]     (from engine/)
out="${1:?usage: frame-variant.sh OUT.wgsl LANES}"
lanes="${2:?lanes}"
shift 2
slangc="${SLANGC:-$(command -v slangc || echo "$HOME/.local/bin/slangc")}"
"$slangc" -target wgsl -stage compute -entry explicit_beams_frame -o "$out" \
  -DENGINE_WGSL=1 -DEB_LANES="$lanes" -DEB_AERO=1 -DEB_NO_SURFACE=1 "$@" slang/kernels/explicit_beams.slang 2>"$out.log" || { cat "$out.log"; exit 1; }
sed -i.bak -e 's/storageBarrier(); textureBarrier(); workgroupBarrier();;/storageBarrier(); workgroupBarrier();/g' \
  -e 's/storageBarrier(); textureBarrier(); workgroupBarrier();/storageBarrier(); workgroupBarrier();/g' "$out" && rm -f "$out.bak"
