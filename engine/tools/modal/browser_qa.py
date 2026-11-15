"""Browser QA of a deployed page on a Modal NVIDIA GPU: headless Chrome with WebGPU on
the driver's Vulkan, driven by engine/crates/engine-webgpu/vehicle-lab-record.mjs
(`--qa`: the vehicle lab's own probe protocol over CDP). Called by browser_qa.sh.

Nothing is built in the container: Chrome and Node are in the image (built once, then
cached by Modal), the recorder script is uploaded as text, the page is the deployment.
The one compile left on the GPU host is the driver compiling the page's WebGPU
pipelines; Chrome's GPU caches (profile GrShaderCache / GraphiteDawnCache /
ShaderCache / GPUCache / Dawn caches) and NVIDIA's shader disk cache are kept in a Modal
volume per GPU, driver and deployment commit (a new commit starts from the newest
snapshot, so unchanged kernels hit), so a kernel version is compiled once.

Prices (modal.com/pricing, 2026-10): T4 $0.000164/s + CPU $0.0000131/core/s + memory
$0.00000222/GiB/s.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import modal

app = modal.App("rubble-browser-qa")

NODE = "22.22.2"  # the repo's .nvmrc
image = (
    modal.Image.from_registry("debian:bookworm-slim", add_python="3.12")
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends"
        " ca-certificates curl gnupg xz-utils fonts-liberation"
        " libvulkan1 libegl1 libx11-6 libxext6",
        "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub"
        " | gpg --dearmor -o /usr/share/keyrings/google.gpg",
        "echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/google.gpg]"
        " http://dl.google.com/linux/chrome/deb/ stable main' > /etc/apt/sources.list.d/google-chrome.list",
        "apt-get update && apt-get install -y --no-install-recommends google-chrome-stable"
        " && rm -rf /var/lib/apt/lists/*",
        f"curl -fsSL https://nodejs.org/dist/v{NODE}/node-v{NODE}-linux-x64.tar.xz"
        " | tar -xJ -C /opt && ln -s /opt/node-v" + NODE + "-linux-x64/bin/node /usr/local/bin/node",
    )
)

cache_volume = modal.Volume.from_name("rubble-browser-qa-cache", create_if_missing=True)
CACHE = "/cache"
PROFILE, NVCACHE, OUT = "/qa/profile", "/qa/nvcache", "/qa/out"
GPU_PRICE = {"T4": 0.000164, "L4": 0.000222, "L40S": 0.000542}
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222
CPUS, MEM_GIB = 16, 16

# WebGPU on the NVIDIA driver's Vulkan, headless: no display, no sandbox (root in the
# container), the GPU blocklist off (a datacenter GPU), ANGLE on Vulkan.
CHROME_FLAGS = (
    "--no-sandbox --ignore-gpu-blocklist --enable-gpu-rasterization --use-angle=vulkan"
    " --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --disable-vulkan-surface"
    " --disable-dev-shm-usage --enable-unsafe-webgpu"
)


def _say(m: str) -> None:
    print(f"[browser_qa] {m}", flush=True)


def _is_gpu_cache(p: Path) -> bool:
    n = p.name
    return p.is_dir() and any(k in n for k in ("ShaderCache", "GPUCache", "Dawn", "Graphite", "GrShader"))


def _cache_dirs(profile: Path) -> list[Path]:
    out = [p for p in profile.glob("*") if _is_gpu_cache(p)]
    d = profile / "Default"
    if d.is_dir():
        out += [p for p in d.glob("*") if _is_gpu_cache(p)]
    return out


def _snaps(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if (p / "DONE").is_file()) if d.is_dir() else []


def _restore(key: str, version: str) -> str:
    root = Path(CACHE) / key
    own = _snaps(root / version)
    pick = own[-1:] or sorted((s for d in (root.glob("*") if root.is_dir() else []) for s in _snaps(d)),
                              key=lambda p: p.name)[-1:]
    Path(PROFILE).mkdir(parents=True, exist_ok=True)
    Path(NVCACHE).mkdir(parents=True, exist_ok=True)
    if not pick:
        return "none (cold)"
    snap = pick[0]
    for part in ("profile", "nvcache"):
        if (snap / part).is_dir():
            subprocess.run(["cp", "-a", f"{snap / part}/.", PROFILE if part == "profile" else NVCACHE], check=True)
    return f"{'this version' if own else 'seed from ' + snap.parent.name}: {snap.name}"


def _save(key: str, version: str, run_id: str) -> str:
    dest = Path(CACHE) / key / version / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}"
    (dest / "profile").mkdir(parents=True)
    for d in _cache_dirs(Path(PROFILE)):
        rel = d.relative_to(PROFILE)
        (dest / "profile" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(d, dest / "profile" / rel)
    shutil.copytree(NVCACHE, dest / "nvcache")
    (dest / "DONE").write_text("")
    for old in _snaps(dest.parent)[:-2]:
        shutil.rmtree(old, ignore_errors=True)
    cache_volume.commit()
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    return f"{dest.name}, {size / 1e6:.1f} MB"


def _check_files(base: str, share: str, mods: str) -> list[str]:
    """The car's files.json and every file it lists, with the share link's cookie."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.open(share, timeout=60).read()
    lines, bad = [], 0
    listing = json.loads(opener.open(f"{base}/{mods}/files.json", timeout=60).read())
    lines.append(f"{mods}/files.json: {len(listing)} entries")
    for f in listing:
        url = f"{base}/{mods}/{urllib.parse.quote(f)}"
        try:
            r = opener.open(urllib.request.Request(url, method="HEAD"), timeout=60)
            size = r.headers.get("content-length")
            if size is None:  # compressed responses: count the body
                size = str(len(opener.open(url, timeout=120).read()))
            lines.append(f"  {r.status} {int(size):>12,} B  {f}")
        except Exception as e:  # noqa: BLE001 - reported per file
            bad += 1
            lines.append(f"  FAILED {f}: {e}")
    lines.append(f"{len(listing) - bad} of {len(listing)} served")
    return lines


@app.function(gpu="T4", image=image, cpu=CPUS, memory=MEM_GIB * 1024, timeout=1800, retries=0,
              single_use_containers=True, volumes={CACHE: cache_volume})
def qa(spec: dict, script: str) -> dict:
    started = time.time()
    deadline = started + spec["cap"] - 60
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    _say(f"nvidia-smi: {smi}")
    driver = smi.split(",")[-1].strip() if "," in smi else "unknown"
    key = f"{spec['gpu']}-{driver}"
    _say(f"GPU cache: {_restore(key, spec['version'])}")
    Path("/qa").mkdir(exist_ok=True)
    Path("/qa/vehicle-lab-record.mjs").write_text(script)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    report = [f"browser QA of {spec['base']} (commit {spec['version']}) on {smi}",
              "Timing on an NVIDIA T4 through Vulkan in Modal's sandbox is not the Mac's: directional only."]
    if spec.get("mods"):
        lines = _check_files(spec["base"], spec["share"], spec["mods"])
        print("\n".join(lines), flush=True)
        report += lines
    env = {
        **os.environ, "CHROME": "/usr/bin/google-chrome-stable", "CHROME_FLAGS": CHROME_FLAGS,
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "XDG_RUNTIME_DIR": "/tmp",
        "__GL_SHADER_DISK_CACHE_PATH": NVCACHE, "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP": "1",
    }
    code = 0
    for step in spec["steps"]:
        left = deadline - time.time()
        if left < 30:
            _say(f"NOT RUN (time cap): {step}")
            report.append(f"NOT RUN (time cap): {' '.join(step)}")
            code = code or 124
            continue
        cmd = ["node", "/qa/vehicle-lab-record.mjs", "--qa", "--out", OUT, "--base", spec["base"],
               "--auth", spec["share"], "--profile", PROFILE, "--limit", str(max(1, int(left // 60))), *step]
        _say("=== " + " ".join(step))
        t0 = time.time()
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="/qa")
        for line in p.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)
            report.append(line.rstrip("\n").replace(spec["share"], "<share link>"))
        rc = p.wait()
        report.append(f"=== exit {rc}, {time.time() - t0:.1f} s wall")
        _say(f"=== exit {rc}, {time.time() - t0:.1f} s wall")
        code = code or rc
        if rc == 3:  # no hardware adapter: nothing else is worth running
            break
    try:
        _say(f"GPU cache saved: {_save(key, spec['version'], spec['id'])}")
    except Exception as e:  # noqa: BLE001 - never fails the run
        _say(f"GPU cache not saved: {e}")
    secs = time.time() - started
    files = {str(f.relative_to(OUT)): f.read_bytes() for f in Path(OUT).rglob("*") if f.is_file()}
    files["qa-report.txt"] = ("\n".join(report) + "\n").encode()
    return {"code": code, "secs": secs, "files": files}


@app.local_entrypoint()
def main(spec: str, out: str):
    s = json.loads(Path(spec).read_text())
    script = Path(s["script_path"]).read_text()
    t0 = time.time()
    call = qa.with_options(gpu=s["gpu"], timeout=s["cap"]).spawn(s, script)
    try:
        res = call.get(timeout=s["cap"] + 300)
    except TimeoutError:
        call.cancel()
        raise SystemExit("[browser_qa] no result within the cap + 300 s of queueing: cancelled")
    outdir = Path(out)
    for name, data in res["files"].items():
        (outdir / name).parent.mkdir(parents=True, exist_ok=True)
        (outdir / name).write_bytes(data)
    rate = GPU_PRICE[s["gpu"]] + CPUS * CPU_PRICE + MEM_GIB * MEM_PRICE
    print(f"[browser_qa] container {res['secs']:.0f} s (call {time.time() - t0:.0f} s); "
          f"cost about ${res['secs'] * rate:.3f}; {len(res['files'])} files in {out}", flush=True)
    (outdir / "exit_code").write_text(str(res["code"]))
    if res["code"]:
        raise SystemExit(res["code"])
