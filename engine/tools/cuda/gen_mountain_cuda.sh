#!/bin/sh
# Slang -> CUDA (and WGSL) for the vehicle lab mountain page's default modules (generated
# sedan, 512 lanes, no surface: the frame and the deferred self-split modules), with the
# flags engine-cuda's / engine-webgpu's build.rs use. Measurement input for
# measure_nvcc.py (--dirs cuda_explicit) and size checks. Usage: gen_mountain_cuda.sh OUTDIR
set -e
OUT=${1:?out dir}
ROOT=$(cd "$(dirname "$0")/../../slang" && pwd)
SRC=$ROOT/kernels/explicit_beams.slang
mkdir -p "$OUT/cuda_explicit" "$OUT/wgsl"
gen() { # name entry defines...
  name=$1; entry=$2; shift 2
  defs=""; for d in "$@"; do defs="$defs -D$d"; done
  slangc -target cuda -stage compute -entry "$entry" $defs -I "$ROOT/physics" -I "$ROOT/kernels" \
    -o "$OUT/cuda_explicit/$name.cu" -reflection-json "$OUT/cuda_explicit/$name.json" "$SRC" &
  slangc -target wgsl -stage compute -entry "$entry" $defs -I "$ROOT/physics" -I "$ROOT/kernels" \
    -o "$OUT/wgsl/$name.wgsl" "$SRC" &
}
gen explicit_beams_wide_nosurf explicit_beams_frame ENGINE_WGSL=1 EB_LANES=512 EB_NO_SURFACE=1
S="ENGINE_WGSL=1 EB_LANES=512 EB_SOLO_STEP=1 EB_SELF_SPLIT=1 EB_NO_SURFACE=1"
gen explicit_beams_selfsplit_step_wide explicit_beams_self_step $S
gen explicit_beams_selfsplit_walk_wide explicit_beams_self_walk $S
gen explicit_beams_selfsplit_pairs_wide explicit_beams_self_pairs $S
wait
ls -l "$OUT/cuda_explicit"/*.cu "$OUT/wgsl"/*.wgsl
