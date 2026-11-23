"""Build on Modal's CPUs (no GPU): the Mac only sends the source that changed and reads
the result. The archive lands in the GPU runner's archive volume (gpu_run.py), so a GPU
stage can run it without the Mac downloading or uploading a binary.

    engine/tools/modal/remote_build.sh [--cpus N] -- <cargo selection>   (from the repo)

Volumes (Modal, v2):
* rubble-rb-src      /<tree key>/...      the source, synced by content (manifest.json)
* rubble-rb-target   /<tree key>/...      cargo's target directory, one per tree
* rubble-rb-cargo    /                    CARGO_HOME (registry, git checkouts)
* rubble-rb-kernels  /fatbins, /slang     nvcc's fatbins and slangc's outputs by content
                                           (ENGINE_CUDA_FATBIN_CACHE, RUBBLE_SLANG_OBJ_CACHE):
                                           every tree and every session shares them
* rubble-gpu-run-archives                 the packed archive, by SHA-256 (gpu_run.py's)

The build image is engine/tools/cuda/Dockerfile: the x86_64 toolchain CI and the old
local route use, so the device code is the reference build's.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("rubble-remote-build")

CTX = "/ctx/cuda"
if modal.is_local():
    # (add_python: Modal's own runtime; the image's python3 3.10 stays for the packer.)
    build_image = modal.Image.from_dockerfile(f"{CTX}/Dockerfile", context_dir=CTX, add_python="3.12")
else:
    build_image = modal.Image.debian_slim()


def _vol(name: str) -> modal.Volume:
    return modal.Volume.from_name(name, create_if_missing=True, version=2)


src_vol = _vol("rubble-rb-src")
target_vol = _vol("rubble-rb-target")
cargo_vol = _vol("rubble-rb-cargo")
kern_vol = _vol("rubble-rb-kernels")
archive_vol = modal.Volume.from_name("rubble-gpu-run-archives", create_if_missing=True)


def _run(cmd: list[str], cwd: str, env: dict) -> tuple[int, float]:
    t = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env)
    return p.returncode, time.time() - t


@app.function(image=build_image, cpu=16, memory=16384, timeout=1800, retries=0,
              volumes={"/srcvol": src_vol, "/target": target_vol, "/cargo": cargo_vol,
                       "/kernels": kern_vol, "/archives": archive_vol})
def build(spec: dict) -> dict:
    t0 = time.time()
    key = spec["key"]
    src_vol.reload()
    if not Path("/repo").exists():
        os.symlink(f"/srcvol/{key}", "/repo")
    tgt = Path("/target") / key
    tgt.mkdir(parents=True, exist_ok=True)
    cores = os.cpu_count() or 8
    env = {
        **os.environ,
        "CARGO_HOME": "/cargo",
        "CARGO_TARGET_DIR": str(tgt),
        "ENGINE_CUDA_FATBIN_CACHE": "/kernels/fatbins",
        "RUBBLE_SLANG_OBJ_CACHE": "/kernels/slang",
        "ENGINE_CUDA_REQUIRE_PTX": "1",
        "CARGO_BUILD_JOBS": str(spec.get("jobs") or cores),
        "RUBBLE_BUILD_JOBS": str(spec.get("kernel_jobs") or cores),
        **spec.get("env", {}),
    }
    if not Path("/cargo/bin").exists():
        # CARGO_HOME on the volume holds the registry; the toolchain's binaries stay in
        # the image's own /opt/cargo (PATH).
        Path("/cargo").mkdir(exist_ok=True)
    out = Path("/tmp/out")
    out.mkdir(exist_ok=True)
    t_setup = time.time() - t0
    cmd = ["python3", "/repo/engine/tools/modal/gpu_run_pack.py", str(out), *spec.get("pack_opts", []),
           "--", *spec["selection"]]
    print(f"[remote_build] {key}: {' '.join(spec['selection'])} on {cores} cores", flush=True)
    code, t_build = _run(cmd, "/repo/engine", env)
    result = {"code": code, "setup_s": t_setup, "build_s": t_build, "cores": cores}
    if code == 0:
        data = (out / "archive.tar.gz").read_bytes()
        sha = hashlib.sha256(data).hexdigest() + ".tar.gz"
        dest = Path("/archives") / sha
        if not dest.exists():
            dest.write_bytes(data)
        result.update(archive=sha, archive_mb=len(data) / 1e6)
    t_c = time.time()
    for v in (target_vol, kern_vol, cargo_vol, archive_vol):
        v.commit()
    result["commit_s"] = time.time() - t_c
    result["total_s"] = time.time() - t0
    return result


def sync(key: str, root: Path, files: list[tuple[str, str]]) -> dict:
    """Upload the files whose content changed since the last sync of this tree, remove
    the ones gone. `files`: (repo-relative path, sha256)."""
    t0 = time.time()
    try:
        old = json.loads(b"".join(src_vol.read_file(f"{key}/.rb_manifest.json")))
    except Exception:  # noqa: BLE001 - first sync of this tree
        old = {}
    new = dict(files)
    changed = [p for p, h in new.items() if old.get(p) != h]
    gone = [p for p in old if p not in new]
    up_bytes = 0
    if changed:
        with src_vol.batch_upload(force=True) as b:
            for p in changed:
                b.put_file(str(root / p), f"/{key}/{p}")
                up_bytes += (root / p).stat().st_size
    for p in gone:
        try:
            src_vol.remove_file(f"/{key}/{p}")
        except Exception:  # noqa: BLE001
            pass
    with src_vol.batch_upload(force=True) as b:
        import io
        b.put_file(io.BytesIO(json.dumps(new).encode()), f"/{key}/.rb_manifest.json")
    return {"changed": len(changed), "removed": len(gone), "bytes": up_bytes, "secs": time.time() - t0}


@app.local_entrypoint()
def main(spec: str):
    s = json.loads(Path(spec).read_text())
    root = Path(s["root"])
    files = [tuple(l.split("\t", 1)[::-1]) for l in Path(s["manifest"]).read_text().splitlines() if l]
    st = sync(s["key"], root, [(p, h) for p, h in files])
    print(f"[remote_build] sync: {st['changed']} files changed ({st['bytes'] / 1e6:.2f} MB), "
          f"{st['removed']} removed, {st['secs']:.1f} s", flush=True)
    t = time.time()
    r = build.remote(s)
    cores = r.get("cores", 16)
    rate = cores * 0.0000131 + 16 * 0.00000222
    print(f"[remote_build] build: exit {r['code']}, container {r['total_s']:.0f} s (setup {r['setup_s']:.0f}, "
          f"cargo+pack {r['build_s']:.0f}, volume commits {r['commit_s']:.0f}); call {time.time() - t:.0f} s; "
          f"about ${r['total_s'] * rate:.4f} at {cores} cores", flush=True)
    if r.get("archive"):
        print(f"[remote_build] archive {r['archive']} ({r['archive_mb']:.0f} MB) in rubble-gpu-run-archives", flush=True)
        Path(s["result"]).write_text(json.dumps(r))
    if r["code"] != 0:
        raise SystemExit(r["code"])
