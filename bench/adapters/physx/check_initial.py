#!/usr/bin/env python3
"""Compare checkpoint 0 of a pose file with the scene's initial state, component-wise.

usage: python3 -I check_initial.py <scene>.scene.json <run>.poses.bin [tolerance=1e-5]

Pose file (bench-core protocol.rs): per record 'RBPS' u32 magic, step u32, n u32, floats u32
(13), then n x 13 f32: pos3, quat4 (xyzw), linvel3, angvel3.
Scene bin (bench-core scene.rs): 16-byte header (magic, version, bodies, joints; v2 adds hull
and vertex counts: 24 bytes), then 24 words per body: shape, flags, half[3], mass, inertia[3],
friction, pos[3], quat[4], linvel[3], angvel[3], restitution.
Quaternions are compared up to sign (q and -q are the same rotation).
"""
import json
import struct
import sys
from pathlib import Path


def main():
    scene_json, poses = Path(sys.argv[1]), Path(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    meta = json.loads(scene_json.read_text())
    sb = (scene_json.parent / meta["bin"]).read_bytes()
    magic, version, nb, nj = struct.unpack_from("<4I", sb, 0)
    head = 24 if version == 2 else 16
    pb = poses.read_bytes()
    pmagic, step, n, floats = struct.unpack_from("<4I", pb, 0)
    assert pmagic == 0x53504252 and floats == 13, "bad pose file"
    assert step == 0, f"first record is step {step}, not 0"
    assert n == nb, f"pose file has {n} bodies, scene {nb}"
    worst = {"pos": (0.0, -1), "quat": (0.0, -1), "linvel": (0.0, -1), "angvel": (0.0, -1)}
    for i in range(nb):
        w = struct.unpack_from("<2I22f", sb, head + i * 96)
        f = w[2:]  # floats from word 2
        pos, quat, lin, ang = f[8:11], f[11:15], f[15:18], f[18:21]
        static = w[1] & 1
        s = struct.unpack_from("<13f", pb, 16 + i * 52)
        dq = min(max(abs(a - b) for a, b in zip(quat, s[3:7])), max(abs(a + b) for a, b in zip(quat, s[3:7])))
        errs = {
            "pos": max(abs(a - b) for a, b in zip(pos, s[0:3])),
            "quat": dq,
            "linvel": 0.0 if static else max(abs(a - b) for a, b in zip(lin, s[7:10])),
            "angvel": 0.0 if static else max(abs(a - b) for a, b in zip(ang, s[10:13])),
        }
        for k, e in errs.items():
            if e > worst[k][0]:
                worst[k] = (e, i)
    ok = all(v[0] <= tol for v in worst.values())
    print(f"{meta['id']}: {nb} bodies, checkpoint 0 max |error|: " + ", ".join(f"{k} {v[0]:.3g} (body {v[1]})" for k, v in worst.items()) + f" -> {'OK' if ok else 'MISMATCH'} (tol {tol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
