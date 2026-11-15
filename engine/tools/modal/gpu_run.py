"""Remote side of gpu_run.sh: run a prebuilt archive on one single-use Modal GPU.

Called by engine/tools/modal/gpu_run.sh (which builds the archive in local Docker);
not meant to be run by hand. The container gets the archive only: executables,
their small data files and a manifest (gpu_run_pack.py).

Vulkan on Modal: the GPU containers mount the NVIDIA driver with all capabilities
(the Vulkan ICD /etc/vulkan/icd.d/nvidia_icd.json -> libGLX_nvidia.so.0), but the
loader and the ICD's dependencies are the image's: libvulkan1, and libEGL / libX11 /
libXext, without which the ICD loads but returns no vkCreateInstance. No Mesa, so no
software fallback can be picked by mistake; gpu_adapter checks anyway.

Prices (modal.com/pricing, 2026-10), GPU only: T4 $0.000164/s, L4 $0.000222/s,
L40S $0.000542/s; plus CPU $0.0000131/core/s and memory $0.00000222/GiB/s.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import modal

app = modal.App("rubble-gpu-run")

# --no-install-recommends: no Mesa (no lavapipe/llvmpipe to fall back to).
image = modal.Image.from_registry("debian:bookworm-slim", add_python="3.12").run_commands(
    "apt-get update && apt-get install -y --no-install-recommends"
    " libvulkan1 libegl1 libx11-6 libxext6 && rm -rf /var/lib/apt/lists/*"
)

# NVIDIA's shader disk cache, kept between runs. The driver compiles a pipeline's
# SPIR-V on the CPU when it is created, and the current explicit-beams kernels take
# minutes each (explicit_beams_solo: 508 s on a T4 host, single-threaded; the frozen
# reference kernel 2.5 s); from the cache it is 0.1 s. Entries are keyed by the
# driver on the SPIR-V's hash, so a kernel that changed misses and is compiled, and
# sharing the cache between worktrees cannot give one another's binary.
# Snapshots are kept per kernel version (the hash of the generated explicit-beams WGSL)
# and driver: a run copies its version's newest snapshot in (else the newest of any
# version, as a seed), and if it compiled anything writes a new snapshot directory of
# its own (nothing is written in place). WARMED marks a snapshot a completed warm-up
# saved: the cold-compile guard passes for that version. Two runs of one version that
# both compile: the last to finish wins. Reset: modal volume delete rubble-gpu-run-shader-cache
cache_volume = modal.Volume.from_name("rubble-gpu-run-shader-cache", create_if_missing=True)
CACHE_MOUNT, CACHE_LOCAL = "/nvcache", "/tmp/nvc"
EXE_NAME = "rubble-gpu"
MAX_SNAPSHOT_BYTES = 4 << 30

GPU_PRICE = {"T4": 0.000164, "L4": 0.000222, "L40S": 0.000542}
CPUS, MEM_GIB = 4, 8
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222
MAX_TIMEOUT = 1800

COLD_GUARD = (
    "STOPPED: this kernel version has not been compiled on {driver} yet. Its warm-up "
    "compiles the explicit-beams pipelines in parallel (the slowest current module "
    "alone takes ~510 s on a T4 host), then the tests compile anything else they use "
    "on first use: expect 9-15 minutes at about $0.02/min with {cpus} cores. Rerun with "
    "--allow-cold-compile (and --timeout 1200 or more) to pay it; later runs of this "
    "kernel version on this driver are warm. Nothing else was run (exit 3)."
)

DIRECTIONAL = (
    "TIMING IS DIRECTIONAL ONLY: an NVIDIA {gpu} through Vulkan in a gVisor sandbox is "
    "not the M3 Max (Metal) or Chrome (WebGPU). Use it to rank variants run in this same "
    "container; final timing happens on the Mac."
)


def _say(msg: str) -> None:
    print(f"[gpu_run] {msg}", flush=True)


def _cpu_secs(pid: int) -> float:
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return 0.0


def _gpu_use() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() or "?"


def _tree_bytes(root: str) -> int:
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())


def _snaps(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if (p / "DONE").is_file()) if d.is_dir() else []


def _cache_in(key: str, kernel: str) -> bool:
    """Copy this kernel version's newest snapshot (else the newest of any version, as a
    seed: the rigid kernels and unchanged modules hit) to CACHE_LOCAL. True when it is
    this version's and a completed warm-up saved it (WARMED)."""
    Path(CACHE_LOCAL).mkdir(parents=True, exist_ok=True)
    root = Path(CACHE_MOUNT) / key / "v"
    own = _snaps(root / (kernel or "none"))
    pick = own[-1:] or sorted(
        (p for d in (root.glob("*") if root.is_dir() else []) for p in _snaps(d)),
        key=lambda p: p.name,
    )[-1:]
    if not pick:
        _say(f"shader cache: empty for {key} (cold compiles)")
        return False
    snap = pick[0]
    try:
        subprocess.run(["cp", "-a", f"{snap}/GLCache", CACHE_LOCAL], check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        _say(f"shader cache: could not read {snap.name} ({e}); cold compiles")
        return False
    warmed = bool(own) and (snap / "WARMED").is_file()
    _say(f"shader cache: {'this kernel version' if own else 'seed from ' + snap.parent.name}, "
         f"{snap.name}, {_tree_bytes(CACHE_LOCAL) / 1e6:.1f} MB")
    return warmed


def _cache_out(key: str, kernel: str, since: float, run_id: str, warmed: bool, was_warmed: bool) -> None:
    """A new snapshot for this kernel version when this run compiled anything or first
    completed its warm-up; the 2 newest per version and the 16 newest versions kept."""
    new = [f for f in Path(CACHE_LOCAL).rglob("*") if f.is_file() and f.stat().st_mtime > since]
    if not new and not (warmed and not was_warmed):
        return
    size = _tree_bytes(CACHE_LOCAL)
    if size > MAX_SNAPSHOT_BYTES:
        _say(f"shader cache: {size / 1e9:.1f} GB, over the cap; not saved")
        return
    root = Path(CACHE_MOUNT) / key / "v"
    dest = root / (kernel or "none") / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}"
    dest.mkdir(parents=True)
    subprocess.run(["cp", "-a", f"{CACHE_LOCAL}/GLCache", str(dest)], check=True)
    if warmed or was_warmed:
        (dest / "WARMED").write_text("")
    (dest / "DONE").write_text("")
    for old in _snaps(dest.parent)[:-2]:
        subprocess.run(["rm", "-rf", str(old)])
    versions = sorted((d for d in root.glob("*") if _snaps(d)), key=lambda d: _snaps(d)[-1].name)
    for old in versions[:-16]:
        subprocess.run(["rm", "-rf", str(old)])
    cache_volume.commit()
    _say(f"shader cache: saved {len(new)} new files ({size / 1e6:.1f} MB) for kernel version "
         f"{kernel or 'none'}{' (warm)' if warmed or was_warmed else ''}")


NOT_RUN = -1


def _run(cmd: list[str], cwd: str, env: dict, deadline: float, every: int) -> tuple[int, float]:
    """Run to completion or the deadline, streaming its output; every `every` s a line
    of GPU use and CPU use. At the cap: the test that was running and its last output.
    NOT_RUN when the deadline passed before it could start."""
    import collections
    import threading

    t0 = time.time()
    if deadline - t0 <= 0:
        return NOT_RUN, 0.0
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    tail: collections.deque[str] = collections.deque(maxlen=60)
    running = [""]

    def pump() -> None:
        for line in p.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)
            tail.append(line)
            # libtest: "test NAME ... " starts a test (its result ends the line).
            if line.startswith("test ") and " ... " in line:
                rest = line.split(" ... ", 1)[1].strip()
                running[0] = "" if rest.startswith(("ok", "FAILED", "ignored")) else line[5:].split(" ... ")[0]

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    last_t, last_cpu = t0, 0.0
    next_note = t0 + every if every else float("inf")
    while True:
        try:
            code = p.wait(timeout=1)
            reader.join(timeout=5)
            return code, time.time() - t0
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        if now >= next_note:
            cpu = _cpu_secs(p.pid)
            _say(f"t+{now - t0:.0f}s: gpu {_gpu_use()}; process cpu "
                 f"{100 * (cpu - last_cpu) / (now - last_t):.0f}%")
            last_t, last_cpu, next_note = now, cpu, now + every
        if now >= deadline:
            p.kill()
            p.wait()
            reader.join(timeout=5)
            _say(f"time cap reached: killed {Path(cmd[0]).parent.name}"
                 + (f" while running test {running[0]}" if running[0] else ""))
            _say("its last output (tests run with --nocapture, so nothing was held back):")
            print("".join(tail), end="", flush=True)
            return 124, time.time() - t0


FUNCTION = dict(
    gpu="T4", image=image, cpu=CPUS, memory=MEM_GIB * 1024, timeout=600, retries=0,
    single_use_containers=True, volumes={CACHE_MOUNT: cache_volume},
)


@app.function(**FUNCTION)
def run(archive: bytes, spec: dict) -> dict:
    return _run_impl(archive, spec)


# (Modal preempted two of tonight's long cold runs: the call restarts from the beginning,
# billed again. Non-preemptible containers are not offered for GPU functions.)


def _run_impl(archive: bytes, spec: dict) -> dict:
    started = time.time()
    # Leave time to save the shader cache and print the summary before Modal's kill.
    deadline = started + spec["cap"] - 45
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall("/work", filter="data")
    # Data lands at the absolute paths the binaries read.
    data = Path("/work/data")
    if data.is_dir():
        subprocess.run(["cp", "-a", f"{data}/.", "/"], check=True)
    manifest = json.loads(Path("/work/manifest.json").read_text())
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    _say(f"nvidia-smi: {smi.stdout.strip() or smi.stderr.strip()}")
    driver = (smi.stdout.split(",")[1].strip() if smi.stdout.count(",") >= 2 else "unknown")
    cache_key = f"{spec['gpu']}-{driver}"
    cache_on = spec.get("shader_cache", True)
    was_warmed = False
    cache_since = time.time()
    cache_env = {}
    if cache_on:
        was_warmed = _cache_in(cache_key, manifest.get("kernel_hash", ""))
        cache_since = time.time()
        cache_env = {"__GL_SHADER_DISK_CACHE_PATH": CACHE_LOCAL,
                     "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP": "1"}
    base = {
        **os.environ,
        "RUST_BACKTRACE": "1",
        "RUBBLE_GPU_LOCK": "0",  # one job per container: no machine lock to share
        "WGPU_BACKEND": "vulkan",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "SEDAN": "/scenarios/sedan_new",
        **cache_env,
        **spec["env"],
    }
    exes = manifest["exes"]
    for e in exes:
        os.chmod(f"/work/{e['file']}", 0o755)
        # One executable name for every process: NVIDIA's shader disk cache keeps a file
        # per executable basename (measured: same basename in another directory hits,
        # another basename misses), so the warm-up's pipelines reach the tests only if
        # they share it.
        d = Path("/work/exe") / Path(e["file"]).name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f"/work/{e['file']}", d / EXE_NAME)
        e["exec"] = str(d / EXE_NAME)
        # The working directory (a test's crate dir, as cargo test uses) exists even
        # when no data file of that crate is shipped.
        Path(e["cwd"]).mkdir(parents=True, exist_ok=True)
    probe = next((e for e in exes if e["role"] == "probe"), None)
    if probe:
        code, _ = _run([probe["exec"]], "/work", base, deadline, 0)
        if code != 0:
            _say(f"no hardware adapter (gpu_adapter exit {code}); nothing run")
            return {"code": code or 1, "secs": time.time() - started}
    if manifest.get("route") == "cuda":
        # Device code (SASS for the T4 / Ada, PTX otherwise) is in the binaries: no
        # compile here, no shader cache, no warm-up.
        base["ENGINE_CUDA_DEVICE"] = "1"
        cache_on = False
        spec = {**spec, "allow_cold": True, "warm": False}
        _say("CUDA route: device code built locally; nothing compiles on this host")
    # Has this kernel version been compiled on this driver (by a warm-up) before?
    kernel = manifest.get("kernel_hash", "")
    mark_seen = False
    seen = cache_on and bool(kernel) and was_warmed
    if kernel:
        _say(f"kernel version {kernel}: {'warmed on this driver before' if seen else 'NEW on this driver (cold)'}")
    if not seen and not spec.get("allow_cold"):
        _say(COLD_GUARD.format(driver=cache_key, cpus=spec.get("cpus", CPUS)))
        return {"code": 3, "secs": time.time() - started}
    if probe and spec.get("warm", True):
        _say("warm-up: the selection's pipelines, every core at once")
        code, secs = _run([probe["exec"], "--warm"], "/work", base, deadline, 0)
        _say(f"warm-up: exit {code}, {secs:.1f} s")
        # Marked seen once the cache holding the warm-up is saved (end of the run).
        mark_seen = code == 0 and cache_on and bool(kernel) and not seen
    runs = [e for e in exes if e["role"] == "run"]
    variants = spec["variants"] or [{"name": "", "env": {}}]
    timing = spec["timing"] or len(variants) > 1
    if timing:
        _say(DIRECTIONAL.format(gpu=spec["gpu"]))
    worst, results = 0, []
    for rep in range(spec["repeat"]):
        for v in variants:
            env = {**base, **v["env"]}
            for e in runs:
                args = list(manifest["run_args"])
                if e["kind"] == "test" and "--list" not in args:
                    if "--test-threads" not in " ".join(args):
                        args.append("--test-threads=1")  # one GPU: tests in turn
                    # Output as it happens: a test the cap kills keeps what it printed.
                    if "--nocapture" not in args and "--show-output" not in args:
                        args.append("--nocapture")
                label = f"{e['name']}" + (f" [{v['name']}]" if v["name"] else "")
                if spec["repeat"] > 1:
                    label += f" #{rep + 1}"
                envs = " ".join(f"{k}={val}" for k, val in {**spec["env"], **v["env"]}.items())
                _say(f"=== {label}: {' '.join(args)}" + (f"  (env {envs})" if envs else ""))
                code, secs = _run([e["exec"], *args], e["cwd"], env, deadline, spec["monitor"])
                if code == NOT_RUN:
                    _say(f"=== {label}: NOT RUN (time cap)")
                else:
                    _say(f"=== {label}: exit {code}, {secs:.2f} s wall")
                results.append((label, code, secs))
                worst = worst or (124 if code == NOT_RUN else code)
    if cache_on:
        try:
            _cache_out(cache_key, kernel, cache_since, spec["id"], mark_seen, was_warmed)
        except Exception as e:  # noqa: BLE001 - the cache never fails a run
            _say(f"shader cache: not saved ({e})")
    _say("summary:")
    for label, code, secs in results:
        status = {NOT_RUN: "NOT RUN", 0: "ok", 124: "KILLED"}.get(code, "FAIL")
        _say(f"  {status:7} {secs:8.2f} s  {label}")
    if timing:
        _say(DIRECTIONAL.format(gpu=spec["gpu"]))
    return {"code": worst, "secs": time.time() - started}


@app.local_entrypoint()
def main(archive: str, spec: str):
    s = json.loads(Path(spec).read_text())
    cap = int(s["cap"])
    if not 30 <= cap <= MAX_TIMEOUT:
        raise SystemExit(f"--timeout must be 30..{MAX_TIMEOUT} s")
    gpu = s["gpu"]
    if gpu not in GPU_PRICE:
        raise SystemExit(f"--gpu {gpu}: one of {', '.join(GPU_PRICE)}")
    data = Path(archive).read_bytes()
    print(f"[gpu_run] archive {len(data) / 1e6:.1f} MB -> Modal {gpu}, cap {cap} s", flush=True)
    t0 = time.time()
    # Spawned, so the wait has a bound of its own: a GPU type without capacity leaves
    # the call queued (Modal's timeout counts only once a container runs).
    call = run.with_options(gpu=gpu, timeout=cap, cpu=int(s.get("cpus", CPUS))).spawn(data, s)
    try:
        out = call.get(timeout=cap + int(s["queue"]))
    except TimeoutError:
        call.cancel()
        print(f"[gpu_run] no result within {cap} s + {s['queue']} s: cancelled (queued for "
              f"lack of {gpu} capacity, or preempted and restarted past the cap)", flush=True)
        Path(spec).with_name("exit_code").write_text("125")
        raise SystemExit(125)
    secs = out["secs"]
    cpus = int(s.get("cpus", CPUS))
    rate = GPU_PRICE[gpu] + cpus * CPU_PRICE + MEM_GIB * MEM_PRICE
    print(
        f"[gpu_run] container {secs:.1f} s (call {time.time() - t0:.1f} s); "
        f"cost about ${secs * rate:.4f} at ${rate * 60:.4f}/min ({gpu} + {cpus} cores + {MEM_GIB} GiB)",
        flush=True,
    )
    Path(spec).with_name("exit_code").write_text(str(out["code"]))
    if out["code"] != 0:
        raise SystemExit(out["code"])
