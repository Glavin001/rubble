#!/bin/sh
# Slang -> WGSL (and CUDA with -cuda) of one explicit module, with build.rs's flags.
# gen.sh OUTDIR NAME ENTRY [DEFINE=V ...]   (measurement input; writes OUTDIR/wgsl, OUTDIR/cuda_explicit)
set -e
OUT=$1; NAME=$2; ENTRY=$3; shift 3
ROOT=$(cd "$(dirname "$0")/../../slang" && pwd)
SRC=${EB_SRC:-$ROOT/kernels/explicit_beams.slang}
mkdir -p "$OUT/wgsl" "$OUT/cuda_explicit"
D=""; for d in "$@"; do D="$D -D$d"; done
slangc -target wgsl -stage compute -entry "$ENTRY" $D -I "$ROOT/physics" -I "$ROOT/kernels" -o "$OUT/wgsl/$NAME.wgsl" "$SRC" 2>/dev/null &
if [ -n "$CUDA" ]; then
slangc -target cuda -stage compute -entry "$ENTRY" $D -I "$ROOT/physics" -I "$ROOT/kernels" -o "$OUT/cuda_explicit/$NAME.cu" -reflection-json "$OUT/cuda_explicit/$NAME.json" "$SRC" 2>/dev/null &
fi
wait
