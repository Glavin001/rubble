"""Remote side of gpu_run.sh: run a prebuilt archive on single-use Modal GPUs.

Called by engine/tools/modal/gpu_run.sh (which builds the archive in local Docker);
not meant to be run by hand. The containers get the archive only: executables,
their small data files and a manifest (gpu_run_pack.py).

One run or a fan-out. The archive is uploaded once to the Modal Volume
`rubble-gpu-run-archives` (keyed by its SHA-256, so a repeat uploads nothing) and
every container reads it from there. A fan-out (spec "fanout") spawns one container
per shard, all at once:

* "exe":    one container per executable of the selection (each test binary),
* "tests":  one container per test (names listed at build time; an executable whose
            tests could not be listed runs whole),
* "matrix": one container per scenario-matrix row (engine/tools/scenario-matrix),
            running matrix.py for that row against the shipped vehicle_lab, with the
            saved states from the Volume `rubble-scenario-states` (uploaded by hash).

Each shard's output streams prefixed with its label; at the end one summary, the
summed billed estimate and one exit code (the first failure's).

Vulkan on Modal: the GPU containers mount the NVIDIA driver with all capabilities
(the Vulkan ICD /etc/vulkan/icd.d/nvidia_icd.json -> libGLX_nvidia.so.0), but the
loader and the ICD's dependencies are the image's: libvulkan1, and libEGL / libX11 /
libXext, without which the ICD loads but returns no vkCreateInstance. No Mesa, so no
software fallback can be picked by mistake; gpu_adapter checks anyway.

Prices (modal.com/pricing, 2026-10), GPU only: T4 $0.000164/s, L4 $0.000222/s,
L40S $0.000542/s; plus CPU $0.0000131/core/s and memory $0.00000222/GiB/s.
"""

from __future__ import annotations

import hashlib
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
# Archives (by SHA-256), uploaded once per build and read by every shard; files older
# than ARCHIVE_DAYS are pruned at the start of a run.
archive_volume = modal.Volume.from_name("rubble-gpu-run-archives", create_if_missing=True)
ARCHIVE_MOUNT, ARCHIVE_DAYS = "/archives", 2
# Scenario-matrix saved states (by SHA-256; `made/<run id>/ROW.fx` for --make results,
# downloaded by the local side and removed).
states_volume = modal.Volume.from_name("rubble-scenario-states", create_if_missing=True)
STATES_MOUNT = "/svol"
MATRIX_PY = "/repo/engine/tools/scenario-matrix/matrix.py"
EXE_NAME = "rubble-gpu"
MAX_SNAPSHOT_BYTES = 4 << 30

GPU_PRICE = {"T4": 0.000164, "L4": 0.000222, "L40S": 0.000542}
CPUS, MEM_GIB = 4, 8
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222
MAX_TIMEOUT = 1800
# Fan-out: reruns of a cold shard hoping to land on a warm host (~5 s each); warm-up
# containers' cores (the NVIDIA compile is CPU work; 8 cores did not finish this
# kernel version in 24 minutes, 16 take ~19) and rounds of them to reach the drivers
# the cold shards need.
COLD_RETRIES, WARM_CPUS, WARM_TRIES = 2, 16, 3

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

# The shard's label, prefixed to every line it prints (a fan-out's containers stream
# into one terminal).
PREFIX = [""]


def _say(msg: str) -> None:
    print(f"{PREFIX[0]}[gpu_run] {msg}", flush=True)


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


def _run(cmd: list[str], cwd: str, env: dict, deadline: float, every: int,
         keep: list[str] | None = None) -> tuple[int, float]:
    """Run to completion or the deadline, streaming its output; every `every` s a line
    of GPU use and CPU use. At the cap: the test that was running and its last output.
    NOT_RUN when the deadline passed before it could start. `keep` collects the lines."""
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
            print(PREFIX[0] + line, end="", flush=True)
            tail.append(line)
            if keep is not None:
                keep.append(line)
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
            print("".join(PREFIX[0] + l for l in tail), end="", flush=True)
            return 124, time.time() - t0


FUNCTION = dict(
    gpu="T4", image=image, cpu=CPUS, memory=MEM_GIB * 1024, timeout=600, retries=0,
    single_use_containers=True,
    volumes={CACHE_MOUNT: cache_volume, ARCHIVE_MOUNT: archive_volume, STATES_MOUNT: states_volume},
)


@app.function(**FUNCTION)
def run(archive_key: str, spec: dict) -> dict:
    PREFIX[0] = f"[{spec['shard']['label']}] " if spec.get("shard") else ""
    return _run_impl(archive_key, spec)


# (Modal preempted two long cold runs: the call restarts from the beginning, billed
# again. Non-preemptible containers are not offered for GPU functions.)


def _test_args(manifest: dict, e: dict, shard: dict | None) -> list[str]:
    args = list(manifest["run_args"])
    if shard and shard.get("test"):
        # One test: libtest's exact filter (the run args' own filter does not apply).
        args = [a for a in args if a.startswith("-")] + [shard["test"], "--exact"]
    if e["kind"] == "test" and "--list" not in args:
        if "--test-threads" not in " ".join(args):
            args.append("--test-threads=1")  # one GPU: tests in turn
        # Output as it happens: a test the cap kills keeps what it printed.
        if "--nocapture" not in args and "--show-output" not in args:
            args.append("--nocapture")
    return args


def _run_impl(archive_key: str, spec: dict) -> dict:
    started = time.time()
    # Leave time to save the shader cache and print the summary before Modal's kill.
    deadline = started + spec["cap"] - 45
    archive = Path(ARCHIVE_MOUNT) / archive_key
    with tarfile.open(archive, mode="r:gz") as tar:
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
            return {"code": code or 1, "secs": time.time() - started, "results": [], "lines": []}
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
        return {"code": 3, "secs": time.time() - started, "results": [], "lines": [], "driver": cache_key}
    if spec.get("warm_only") and seen:
        # A fan-out's warm-up landed on a driver that is warm already: nothing to do.
        _say(f"warm-up: {cache_key} is warm for this kernel version already")
        return {"code": 0, "secs": time.time() - started, "results": [], "lines": [],
                "driver": cache_key, "already": True}
    warm_code = 0
    if probe and spec.get("warm", True):
        _say("warm-up: the selection's pipelines, every core at once")
        warm_code, secs = _run([probe["exec"], "--warm"], "/work", base, deadline, 0)
        _say(f"warm-up: exit {warm_code}, {secs:.1f} s")
        # Marked seen once the cache holding the warm-up is saved (end of the run).
        mark_seen = warm_code == 0 and cache_on and bool(kernel) and not seen
    if spec.get("warm_only"):
        # A fan-out's warm-up container: compile once here, save the cache (a partial
        # one too: the next warm-up of this version resumes from it), let the shards
        # start warm. Its exit is the warm-up's: a killed warm-up stops the fan-out.
        if cache_on:
            try:
                _cache_out(cache_key, kernel, cache_since, spec["id"], mark_seen, was_warmed)
            except Exception as e:  # noqa: BLE001
                _say(f"shader cache: not saved ({e})")
        return {"code": warm_code, "secs": time.time() - started, "results": [], "lines": [],
                "driver": cache_key}
    shard = spec.get("shard")
    runs = [e for e in exes if e["role"] == "run"]
    if shard and shard.get("exe"):
        runs = [e for e in runs if e["file"] == shard["exe"]]
    variants = spec["variants"] or [{"name": "", "env": {}}]
    timing = spec["timing"] or len(variants) > 1
    if timing:
        _say(DIRECTIONAL.format(gpu=spec["gpu"]))
    worst, results, lines = 0, [], []
    if shard and shard.get("rows"):
        worst, results, lines = _matrix_row(spec, shard, runs, base, deadline)
    else:
        for rep in range(spec["repeat"]):
            for v in variants:
                env = {**base, **v["env"]}
                for e in runs:
                    args = _test_args(manifest, e, shard)
                    label = (shard["label"] if shard else e["name"]) + (f" [{v['name']}]" if v["name"] else "")
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
    if not shard:
        _say("summary:")
        for label, code, secs in results:
            _say(f"  {_status(code):7} {secs:8.2f} s  {label}")
        if timing:
            _say(DIRECTIONAL.format(gpu=spec["gpu"]))
    return {"code": worst, "secs": time.time() - started, "results": results, "lines": lines,
            "driver": cache_key}


def _matrix_row(spec: dict, shard: dict, runs: list[dict], base: dict, deadline: float):
    """One scenario-matrix row: matrix.py against the shipped vehicle_lab, the row's
    saved state from the states Volume (or, for --make, saved into it)."""
    cuda = base.get("MATRIX_CUDA") == "1"
    exe = "explicit_replay" if cuda else "vehicle_lab"
    lab = next((e for e in runs if e["name"] == exe), None)
    if lab is None:
        _say(f"matrix: the archive has no {exe}")
        return 1, [], []
    states = Path("/tmp/states")
    (states / "portable").mkdir(parents=True, exist_ok=True)
    rows = [r["row"] for r in shard["rows"]]
    make = spec["matrix_args"][:1] == ["--make"]
    if not make:
        for r in shard["rows"]:
            src = Path(STATES_MOUNT) / r["state"]
            if not src.is_file():
                _say(f"matrix: no saved state {r['state']} in the states volume")
                return 1, [], []
            (states / ("portable" if cuda else "") / f"{r['row']}{'.pfx' if cuda else '.fx'}").symlink_to(src)
    env = {**base, "LAB_BIN": lab["exec"], "STATES": str(states),
           "CAR": base.get("CAR", "/scenarios/sedan_new")}
    if cuda:
        env["REPLAY_BIN"] = lab["exec"]
    lines: list[str] = []
    # (The rows named on the command line are spread over the containers: this one's
    # are `rows`, run by one matrix.py in turn.)
    picked = set(spec["matrix_rows"])
    cmd = ["python3", MATRIX_PY, *[a for a in spec["matrix_args"] if a not in picked], *rows]
    _say(f"=== rows {','.join(rows)}: {' '.join(cmd[2:])}")
    code, secs = _run(cmd, "/repo/engine", env, deadline, spec["monitor"], keep=lines)
    row = rows[0]
    if make and code == 0 and (states / f"{row}.fx").is_file():
        dest = Path(STATES_MOUNT) / "made" / spec["id"]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(states / f"{row}.fx", dest / f"{row}.fx")
        states_volume.commit()
    return (124 if code == NOT_RUN else code), [(f"rows {','.join(rows)}", code, secs)], lines


def _status(code: int) -> str:
    return {NOT_RUN: "NOT RUN", 0: "ok", 124: "KILLED", 3: "COLD"}.get(code, "FAIL")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _upload(volume: modal.Volume, files: dict[str, Path], what: str) -> None:
    """Upload the files (volume path -> local path) the volume does not have yet."""
    have = {e.path.lstrip("/") for e in volume.listdir("/")}
    todo = {k: v for k, v in files.items() if k.lstrip("/") not in have}
    if not todo:
        print(f"[gpu_run] {what}: already uploaded", flush=True)
        return
    t0 = time.time()
    with volume.batch_upload(force=True) as batch:
        for key, local in todo.items():
            batch.put_file(str(local), "/" + key.lstrip("/"))
    size = sum(p.stat().st_size for p in todo.values())
    print(f"[gpu_run] {what}: uploaded {len(todo)} file(s), {size / 1e6:.1f} MB in {time.time() - t0:.1f} s",
          flush=True)


def _prune_archives() -> None:
    cutoff = time.time() - ARCHIVE_DAYS * 86400
    for e in archive_volume.listdir("/"):
        if e.mtime and e.mtime < cutoff:
            try:
                archive_volume.remove_file(e.path)
            except Exception:  # noqa: BLE001 - pruning never fails a run
                pass


def _shards(s: dict, manifest: dict, run_dir: Path) -> list[dict]:
    mode = s.get("fanout", "")
    runs = [e for e in manifest["exes"] if e["role"] == "run"]
    if mode == "exe":
        return [{"label": e["name"], "exe": e["file"]} for e in runs]
    if mode == "tests":
        out = []
        for e in runs:
            names = e.get("tests")
            if names:
                out += [{"label": f"{e['name']}::{t}", "exe": e["file"], "test": t} for t in names]
            else:
                out.append({"label": e["name"], "exe": e["file"]})
        return out
    if mode == "matrix":
        make = s["matrix_args"][:1] == ["--make"]
        # (CUDA: each row's portable fixture, matrix.py --make-portable.)
        cuda = s["env"].get("MATRIX_CUDA") == "1"
        rel, ext = ("portable/", ".pfx") if cuda else ("", ".fx")
        rows = []
        for row in s["matrix_rows"]:
            sh = {"row": row}
            if not make:
                local = run_dir / "states" / f"{rel}{row}{ext}"
                if not local.is_file():
                    raise SystemExit(f"[gpu_run] no saved state for row {row} "
                                     f"(engine/out/scenario-matrix/{rel}{row}{ext}): run.sh --make {row}"
                                     + (", then matrix.py --make-portable" if cuda else ""))
                sh["state"] = _sha256(local) + ext
                sh["local"] = str(local)
            rows.append(sh)
        # --pack N: the rows in N containers, in turn within each (a container's start
        # is billed once, not once per row). --make keeps one row a container.
        n = len(rows) if make or not s.get("pack") else max(1, min(int(s["pack"]), len(rows)))
        groups = [rows[i::n] for i in range(n)]
        return [{"label": ",".join(r["row"] for r in g), "rows": g} for g in groups]
    return []


def _watch_heartbeat(hb: Path, stale: float = 60.0) -> None:
    """gpu_run.sh touches `hb` every few seconds while its shell lives. If it stops (the
    wrapper was killed, even by SIGKILL, which no trap sees), exit this client: Modal
    then stops the ephemeral app and every container it spawned, instead of a fan-out
    running on, billed, with nobody waiting for it."""
    import threading

    def loop() -> None:
        while True:
            time.sleep(10)
            try:
                age = time.time() - hb.stat().st_mtime
            except OSError:
                age = stale + 1
            if age > stale:
                print(f"[gpu_run] the local wrapper is gone (no heartbeat for {age:.0f} s): "
                      "stopping this run and its containers", flush=True)
                os._exit(130)

    threading.Thread(target=loop, daemon=True).start()


@app.local_entrypoint()
def main(archive: str, spec: str):
    s = json.loads(Path(spec).read_text())
    run_dir = Path(spec).parent
    if (run_dir / "heartbeat").exists():
        _watch_heartbeat(run_dir / "heartbeat")
    cap = int(s["cap"])
    if not 30 <= cap <= MAX_TIMEOUT:
        raise SystemExit(f"--timeout must be 30..{MAX_TIMEOUT} s")
    gpu = s["gpu"]
    if gpu not in GPU_PRICE:
        raise SystemExit(f"--gpu {gpu}: one of {', '.join(GPU_PRICE)}")
    apath = Path(archive)
    key = _sha256(apath) + ".tar.gz"
    print(f"[gpu_run] archive {apath.stat().st_size / 1e6:.1f} MB -> Modal {gpu}, cap {cap} s", flush=True)
    _prune_archives()
    _upload(archive_volume, {key: apath}, "archive")
    with tarfile.open(apath, mode="r:gz") as tar:
        manifest = json.load(tar.extractfile("manifest.json"))  # type: ignore[arg-type]
    shards = _shards(s, manifest, run_dir)
    if shards and s.get("fanout") == "matrix" and s["matrix_args"][:1] != ["--make"]:
        _upload(states_volume, {r["state"]: Path(r["local"]) for sh in shards for r in sh["rows"]},
                "saved states")
    cpus = float(s.get("cpus", CPUS))
    mem = float(s.get("mem", MEM_GIB))
    rate = GPU_PRICE[gpu] + cpus * CPU_PRICE + mem * MEM_PRICE
    fn = run.with_options(gpu=gpu, timeout=cap, cpu=cpus, memory=int(mem * 1024))
    t0 = time.time()
    if not shards:
        # Spawned, so the wait has a bound of its own: a GPU type without capacity leaves
        # the call queued (Modal's timeout counts only once a container runs).
        call = fn.spawn(key, s)
        try:
            out = call.get(timeout=cap + int(s["queue"]))
        except TimeoutError:
            call.cancel()
            print(f"[gpu_run] no result within {cap} s + {s['queue']} s: cancelled (queued for "
                  f"lack of {gpu} capacity, or preempted and restarted past the cap)", flush=True)
            (run_dir / "exit_code").write_text("125")
            raise SystemExit(125)
        secs = out["secs"]
        print(f"[gpu_run] container {secs:.1f} s (call {time.time() - t0:.1f} s); "
              f"cost about ${secs * rate:.4f} at ${rate * 60:.4f}/min ({gpu} + {cpus:g} cores + {mem:g} GiB)",
              flush=True)
        (run_dir / "exit_code").write_text(str(out["code"]))
        if out["code"] != 0:
            raise SystemExit(out["code"])
        return
    _fanout(fn, key, s, shards, run_dir, rate, t0)


def _fanout(fn, key: str, s: dict, shards: list[dict], run_dir: Path, rate: float, t0: float) -> None:
    """Every shard in a container of its own, all spawned at once; wait for all.

    Cold kernels (a shard's kernel version not compiled on its host's driver): never one
    compile per shard (12 matrix rows would pay the ~18-minute warm-up twelve times).
    With --allow-cold-compile the drivers the cold shards found get one warm-up each
    (WARM_CPUS cores, the full cap, landing retried until it reaches a driver that needs
    it), then the cold shards rerun, retried to land on a warm host. A warm-up that fails
    or is killed stops the fan-out. Without --allow-cold-compile they are reported."""
    budget = int(s["cap"]) + int(s["queue"])
    allow_cold = bool(s.get("allow_cold"))
    s = {**s, "allow_cold": False}
    costs: list[tuple[float, float]] = []  # (container seconds, $/s) not in `done`
    gpu = s["gpu"]
    warm_rate = GPU_PRICE[gpu] + WARM_CPUS * CPU_PRICE + MEM_GIB * MEM_PRICE
    warm_fn = run.with_options(gpu=gpu, timeout=MAX_TIMEOUT, cpu=WARM_CPUS)
    print(f"[gpu_run] fan-out: {len(shards)} containers ({s['fanout']}), all at once", flush=True)

    def spawn(i: int, sh: dict, **extra):
        shard = {k: v for k, v in sh.items() if k not in ("local", "rows")}
        if "rows" in sh:
            shard["rows"] = [{k: v for k, v in r.items() if k != "local"} for r in sh["rows"]]
        spec = {**s, "shard": shard, "id": f"{s['id']}-{i}", **extra}
        return fn.spawn(key, spec)

    def collect(label: str, call, timeout: float) -> dict | None:
        # The calls run in parallel: each wait is bounded by the per-call budget.
        try:
            return call.get(timeout=timeout)
        except TimeoutError:
            call.cancel()
            return None
        except Exception as e:  # noqa: BLE001 - one shard's crash is that shard's failure
            print(f"[{label}] [gpu_run] call failed: {e}", flush=True)
            return {"code": 1, "secs": 0.0, "results": [], "lines": []}

    calls = [(sh, spawn(i, sh)) for i, sh in enumerate(shards)]
    done: list[tuple[dict, dict | None]] = [(sh, collect(sh["label"], c, budget)) for sh, c in calls]

    def cold() -> list[int]:
        return [i for i, (_, o) in enumerate(done) if o and o["code"] == 3]

    def rerun(idx: list[int]) -> None:
        for i in idx:
            costs.append((done[i][1]["secs"], rate))  # type: ignore[index]
        again = [(i, spawn(i, done[i][0])) for i in idx]
        for i, c in again:
            done[i] = (done[i][0], collect(done[i][0]["label"], c, budget))

    stopped = ""
    if allow_cold and cold():
        need = {done[i][1].get("driver", "?") for i in cold()}  # type: ignore[union-attr]
        print(f"[gpu_run] {len(cold())} shard(s) cold on {', '.join(sorted(need))}: one warm-up per "
              f"driver ({WARM_CPUS} cores, cap {MAX_TIMEOUT} s), then those shards again", flush=True)
        for attempt in range(WARM_TRIES):
            if not need:
                break
            # One warm-up per driver still needed, at once; where each lands is Modal's.
            ws = [warm_fn.spawn(key, {**s, "warm_only": True, "allow_cold": True, "cap": MAX_TIMEOUT,
                                      "cpus": WARM_CPUS, "id": f"{s['id']}-warm{attempt}-{n}"})
                  for n in range(len(need))]
            for w in ws:
                r = collect("warm-up", w, MAX_TIMEOUT + int(s["queue"]))
                if r is None or r["code"] != 0:
                    stopped = (f"a warm-up failed ({'no result' if r is None else 'exit ' + str(r['code'])}"
                               f"{' on ' + r['driver'] if r and r.get('driver') else ''})")
                    if r:
                        costs.append((r["secs"], warm_rate))
                    continue
                costs.append((r["secs"], warm_rate))
                d = r.get("driver", "?")
                took = "already warm" if r.get("already") else f"{r['secs']:.0f} s"
                print(f"[gpu_run] warm-up on {d}: {took}", flush=True)
                need.discard(d)
            if stopped:
                break
        if stopped:
            print(f"[gpu_run] {stopped}: stopping; the cold shards are not rerun (nothing compiles "
                  "per shard)", flush=True)
        else:
            for _ in range(COLD_RETRIES):
                if not cold():
                    break
                print(f"[gpu_run] {len(cold())} shard(s) again, for a warm host", flush=True)
                rerun(cold())
    elif cold():
        for _ in range(COLD_RETRIES):
            # Another host may be warm already (Modal mixes driver versions).
            idx = cold()
            if not idx:
                break
            print(f"[gpu_run] {len(idx)} shard(s) cold on their host's driver: again, for a warm host", flush=True)
            rerun(idx)
    wall = time.time() - t0
    billed_cost = sum(o["secs"] for _, o in done if o) * rate + sum(sec * r for sec, r in costs)
    billed = sum(o["secs"] for _, o in done if o) + sum(sec for sec, _ in costs)
    print("\n[gpu_run] fan-out summary:", flush=True)
    worst = 0
    for sh, o in done:
        if o is None:
            code, secs, note = 125, 0.0, "no result (queued past the limit, or preempted)"
        else:
            code, secs = o["code"], o["secs"]
            note = o.get("driver", "")
        worst = worst or code
        print(f"[gpu_run]   {_status(code) if code != 125 else 'NO GPU':7} {secs:7.1f} s  {sh['label']}"
              + (f"  ({note})" if note else ""), flush=True)
    if s.get("fanout") == "matrix":
        _matrix_report(s, done, run_dir)
    cold = [sh["label"] for sh, o in done if o and o["code"] == 3]
    if cold:
        print(f"[gpu_run] {len(cold)} shard(s) found the kernel version cold on their host's driver: rerun "
              "with --allow-cold-compile --timeout 1500 (one warm-up per driver; later runs are warm)",
              flush=True)
    print(f"[gpu_run] {len(shards)} containers, wall {wall:.1f} s; billed {billed:.0f} container-s, "
          f"about ${billed_cost:.4f}", flush=True)
    (run_dir / "exit_code").write_text(str(worst))
    if worst:
        raise SystemExit(worst)


def _matrix_report(s: dict, done: list[tuple[dict, dict | None]], run_dir: Path) -> None:
    """One matrix table from the rows' outputs (header once, rows in order, verdict)."""
    make = s["matrix_args"][:1] == ["--make"]
    if make:
        out_dir = run_dir / "states"
        out_dir.mkdir(exist_ok=True)
        made = 0
        for i, (sh, o) in enumerate(done):
            if o and o["code"] == 0:
                where = f"made/{s['id']}-{i}"
                row = sh["rows"][0]["row"]
                blob = b"".join(states_volume.read_file(f"{where}/{row}.fx"))
                (out_dir / f"{row}.fx").write_bytes(blob)
                states_volume.remove_file(where, recursive=True)
                made += 1
        print(f"[gpu_run] matrix: {made} state(s) written to engine/out/scenario-matrix/", flush=True)
        return
    header, rows, bad = [], [], []
    for sh, o in done:
        lines = [l.rstrip("\n") for l in (o or {}).get("lines", [])]
        if not header:
            header = [l for l in lines if l.startswith(("scenario matrix", "row "))]
        body = [l for l in lines if l and not l.startswith(("scenario matrix", "row ", "verdict:", "baseline:"))]
        names = [r["row"] for r in sh["rows"]]
        got = [l for l in body if l.split()[:1] and l.split()[0] in names]
        rows += body + [f"{n:<8} NO RESULT ({_status(o['code']) if o else 'NO GPU'})"
                        for n in names if not any(l.split()[0] == n for l in got)]
        if not o or o["code"] != 0:
            bad += names
    print("\n" + "\n".join(header + rows), flush=True)
    kind = "baseline" if s["matrix_args"][:1] == ["--baseline"] else "verdict"
    verdict = "PASS" if not bad else "FAIL (" + ", ".join(bad) + ")"
    nrows = sum(len(sh["rows"]) for sh, _ in done)
    where = "one container each" if nrows == len(done) else f"in {len(done)} container(s)"
    print(f"{kind}: {verdict}; {nrows} rows on Modal {s['gpu']}, {where} "
          f"(timing directional: each row's on/off in one container)", flush=True)
