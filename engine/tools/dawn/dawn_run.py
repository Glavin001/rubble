"""The web World on Dawn (Chrome's WebGPU, in Node) on a Modal GPU: engine/tools/dawn/run.sh.

Headless Chrome gets no WebGPU adapter on Modal (the sandbox has no /dev/dri); Dawn's
Node bindings (the npm `webgpu` package) reach the NVIDIA GPU through Vulkan. The
bundle (the web build's World wasm and its JS) and the car are uploaded by content to a
Volume; the harness (world_ab.mjs) runs in a container with Node 24.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("rubble-dawn-run")
# (libx11-6, libxext6, libegl1: NVIDIA's Vulkan driver library loads libX11.)
image = (
    modal.Image.from_registry(f"node:{os.environ.get('DAWN_NODE', '22')}-bookworm-slim", add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends libvulkan1 libx11-6 libxext6 libegl1 "
        "&& rm -rf /var/lib/apt/lists/*",
        "mkdir -p /dawn && cd /dawn && npm init -y >/dev/null && npm install webgpu@0.3.8 --silent",
    )
)
bundles = modal.Volume.from_name("rubble-dawn-bundles", create_if_missing=True)
# Dawn's and the NVIDIA driver's pipeline caches, kept between runs (a kernel version
# compiles once a driver).
caches = modal.Volume.from_name("rubble-dawn-caches", create_if_missing=True)


@app.function(gpu="T4", image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/bundles": bundles, "/caches": caches})
def run(spec: dict) -> dict:
    t0 = time.time()
    work = Path("/work")
    work.mkdir(exist_ok=True)
    subprocess.run(["tar", "-xzf", f"/bundles/{spec['bundle']}", "-C", str(work)], check=True)
    for f in ("world_ab.mjs", "loader.mjs"):
        (Path("/dawn") / f).write_text(spec["scripts"][f])
    nv = Path("/caches/nvidia")
    nv.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "BUNDLE": str(work / "bundle"), "CAR": str(work / "car"),
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "XDG_RUNTIME_DIR": "/tmp",
           "__GL_SHADER_DISK_CACHE_PATH": str(nv), "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP": "1",
           "__GL_SHADER_DISK_CACHE_SIZE": str(4 << 30)}
    cmd = ["node", *spec.get("node_flags", []), "--import", "./loader.mjs", "world_ab.mjs", *spec["args"]]
    print("[dawn_run] " + " ".join(cmd), flush=True)
    # The driver's shader cache is committed every minute while the harness runs: a
    # preempted container (Modal restarts the call from the beginning) or a run that hits
    # its cap keeps the kernels it finished compiling (one took 766 s cold).
    import threading
    done = threading.Event()

    def keep_committing() -> None:
        while not done.wait(60):
            try:
                caches.commit()
            except Exception as e:  # noqa: BLE001 - the next commit tries again
                print(f"[dawn_run] cache commit failed: {e}", flush=True)

    threading.Thread(target=keep_committing, daemon=True).start()
    # Every line as it comes (the lab's per-kernel progress included), and kept for the result.
    p = subprocess.Popen(cmd, cwd="/dawn", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    cap = spec.get("cap", 900)
    deadline = time.time() + cap
    for line in p.stdout:
        line = line.rstrip("\n")
        lines.append(line)
        print(line, flush=True)
        if time.time() > deadline:
            p.kill()
            print(f"[dawn_run] killed at the {cap} s cap", flush=True)
            break
    p.wait()
    done.set()
    caches.commit()
    result = next((json.loads(l[7:]) for l in lines if l.startswith("RESULT ")), None)
    return {"code": p.returncode, "secs": time.time() - t0, "result": result}


@app.local_entrypoint()
def main(spec: str):
    s = json.loads(Path(spec).read_text())
    name = s["bundle"]
    if s.get("bundle_path"):
        # (A bundle packed here; one the remote build made is on the Volume already.)
        bundle = Path(s["bundle_path"])
        have = {e.path.strip("/") for e in bundles.listdir("/")}
        if name not in have:
            with bundles.batch_upload(force=True) as b:
                b.put_file(str(bundle), f"/{name}")
            print(f"[dawn_run] uploaded {bundle.stat().st_size / 1e6:.0f} MB bundle", flush=True)
    s["scripts"] = {f: Path(f"/app/{f}").read_text() for f in ("world_ab.mjs", "loader.mjs")}
    t = time.time()
    r = run.remote(s)
    print(f"[dawn_run] exit {r['code']}, container {r['secs']:.0f} s, call {time.time() - t:.0f} s, "
          f"about ${r['secs'] * (0.000164 + 4 * 0.0000131 + 16 * 0.00000222):.4f}", flush=True)
    if r["result"]:
        Path(s["result_path"]).write_text(json.dumps(r["result"], indent=1))
    raise SystemExit(r["code"])
