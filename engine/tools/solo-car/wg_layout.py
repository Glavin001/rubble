#!/usr/bin/env python3
"""The solo-car kernel's workgroup layout for the lab's sedan (perf doc §8.43), as WGSL
skeletons that declare every region and use it in its phase, 1,024 lanes. Variants:
the layout; the same with the time-shared pool typed atomic; the layout grown to exactly
32,768 B; one 16-byte unit more. Chrome's verdict on each (pipeline creation, the
adapter's limits as the page requests them):

  python3 tools/solo-car/wg_layout.py out/solo/wg-layout
  EXPLICIT_WGSL_DIR=$PWD/out/solo/wg-layout python3 scripts/with-gpu-lock.py --limit 300 -- \
      node --experimental-websocket crates/engine-webgpu/explicit-wgsl-check.mjs

The counts are the sedan's (`cargo run --profile fast -p engine-world --example solo_census`).
Dawn counts each workgroup variable rounded up to 16 bytes; so does this script.
"""
import json
import os
import sys

NN = 548
NSCAL = 3550
NVEC = 335 + 7 + 12 * 4
HELD = (NN * 2 + 31) // 32
out = sys.argv[1] if len(sys.argv) > 1 else "out/solo/wg-layout"
os.makedirs(out, exist_ok=True)
os.chdir(out)


def wgsl(extra_words=0, pool_atomic=False):
    T = NSCAL + NVEC * 3 + extra_words
    pool = f"array<atomic<u32>, {T}>" if pool_atomic else f"array<u32, {T}>"

    def st(i, v):
        return f"atomicStore(&gsT[{i}], {v})" if pool_atomic else f"gsT[{i}] = {v}"

    def ld(i):
        return f"atomicLoad(&gsT[{i}])" if pool_atomic else f"gsT[{i}]"

    src = f"""
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
var<workgroup> gsPos: array<f32, {NN*3}>;
var<workgroup> gsVel: array<f32, {NN*3}>;
var<workgroup> gsHeld: array<atomic<u32>, {HELD}>;
var<workgroup> gsCnt: array<atomic<u32>, 16>;
var<workgroup> gsUni: array<u32, 8>;
var<workgroup> gsT: {pool};
@compute @workgroup_size(1024)
fn solo(@builtin(local_invocation_index) lid: u32) {{
  if (lid < {NN}u) {{
    for (var c = 0u; c < 3u; c++) {{ gsPos[lid*3u+c] = bitcast<f32>(buf[lid*3u+c]); gsVel[lid*3u+c] = bitcast<f32>(buf[{NN*3}u+lid*3u+c]); }}
  }}
  if (lid < {HELD}u) {{ atomicStore(&gsHeld[lid], 0u); }}
  if (lid < 16u) {{ atomicStore(&gsCnt[lid], 0u); }}
  if (lid < 8u) {{ gsUni[lid] = buf[lid]; }}
  let steps = workgroupUniformLoad(&gsUni[0]) & 63u;
  for (var s = 0u; s < steps; s++) {{
    for (var j = lid; j < {T}u; j += 1024u) {{ let a = j % {NN}u; {st('j', 'bitcast<u32>(gsPos[a*3u] - gsVel[a*3u+1u])')}; }}
    workgroupBarrier();
    if (lid < {NN}u) {{ var f = 0.0; for (var e = 0u; e < 16u; e++) {{ f += bitcast<f32>({ld('(lid*7u+e*131u) % ' + str(T) + 'u')}); }} gsVel[lid*3u] += f; atomicOr(&gsHeld[lid/16u], 1u); atomicAdd(&gsCnt[0], 1u); }}
    workgroupBarrier();
    for (var j = lid; j < {1024+NN}u; j += 1024u) {{ {st('j', 'bitcast<u32>(gsPos[(j % ' + str(NN) + 'u)*3u+2u])')}; }}
    workgroupBarrier();
    if (lid < {NN}u) {{ gsPos[lid*3u+1u] += bitcast<f32>({ld('lid')}) + gsVel[lid*3u] + bitcast<f32>(gsUni[1u + (lid & 3u)]); }}
    workgroupBarrier();
  }}
  if (lid < {NN}u) {{ buf[lid] = bitcast<u32>(gsPos[lid*3u+1u]) ^ atomicLoad(&gsHeld[lid/16u]) ^ atomicLoad(&gsCnt[lid & 15u]) ^ {ld('lid + ' + str(T - NN) + 'u')}; }}
}}
"""
    r16 = lambda w: (4 * w + 15) // 16 * 16
    return src, sum(r16(w) for w in (NN * 3, NN * 3, HELD, 16, 8, T))


man = []


def emit(name, **kw):
    src, b = wgsl(**kw)
    open(name + ".wgsl", "w").write(src)
    man.append({"name": name, "entry": "solo"})
    print(f"{name:28} {b:6} bytes")
    return b


b = emit("solo_layout")
emit("solo_layout_atomic_pool", pool_atomic=True)
spare = 32768 - b
emit("solo_layout_full", extra_words=spare // 4)
emit("solo_layout_over", extra_words=spare // 4 + 1)
json.dump(man, open("manifest.json", "w"))
