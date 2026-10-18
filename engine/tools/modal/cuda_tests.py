"""Run engine-cuda's GPU tests on Modal: only GPU seconds are spent there.

Nothing compiles on Modal. Test binaries (device code embedded) are built for
free and shipped to a tiny single-use GPU container that only executes them
and exits:

* macOS: `run_cuda_tests.sh` builds in local Docker (x86_64 Ubuntu 22.04) and
  passes the archive here with --archive.
* Linux x86_64 with slangc + nvcc (the Cursor VM): built natively here.
* CI: built on the free GitHub runner (`python3 cuda_tests.py --package OUT`).

Usage from the repo root. `modal.sh` runs the pinned Modal CLI in Docker, so
nothing is installed on the host; `modal.sh setup` logs in once.

    engine/tools/modal/run_cuda_tests.sh                       # local build, T4 run
    engine/tools/modal/run_cuda_tests.sh --perf --gpu L40S     # AD102 timing
    engine/tools/modal/modal.sh run engine/tools/modal/cuda_tests.py --archive cuda-tests.tar.gz

Prices (modal.com/pricing, 2026-10): T4 $0.000164/s, L4 $0.000222/s,
L40S $0.000542/s. A correctness run is ~6 s of a T4 (~$0.001).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

# Only the local side builds; inside Modal containers this module sits alone
# at /root/cuda_tests.py.
ENGINE = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
CUDA_PKG = "12.8"

app = modal.App("rubble-engine-cuda")

# Runs prebuilt test binaries only: Modal mounts the NVIDIA driver (libcuda),
# and cudarc loads it at run time, so no CUDA toolkit and a fast cold start.
# Bookworm's glibc (2.36) runs binaries built on Ubuntu 22.04 (2.35) or here.
# Pinned rather than `debian_slim()`, whose distro follows the workspace's image
# builder version (the legacy builder gives bullseye: glibc 2.31, dead mirrors).
BASE = "debian:bookworm-slim"
gpu_image = modal.Image.from_registry(BASE, add_python="3.12")
# `--sanitize TOOL` only: NVIDIA's compute-sanitizer for locating GPU faults
# and races by kernel source line; the kernels carry -lineinfo.
gpu_debug_image = gpu_image.apt_install("curl", "ca-certificates", "gnupg").run_commands(
    "chmod 1777 /tmp"
    " && curl -fsSL -o /opt/cuda-keyring.deb"
    " https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"
    " && dpkg -i /opt/cuda-keyring.deb && apt-get update"
    f" && apt-get install -y --no-install-recommends cuda-sanitizer-{CUDA_PKG.replace('.', '-')}"
)

def package_tests(engine_dir: Path, env: dict[str, str] | None = None) -> bytes:
    """`cargo test --no-run` for engine-cuda; returns a tar.gz of its test executables."""
    cmd = [
        "cargo", "test", "-p", "engine-cuda", "--release", "--tests", "--no-run",
        "--message-format=json-render-diagnostics",
    ]
    out = subprocess.run(
        cmd, cwd=engine_dir, env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, check=True, text=True,
    ).stdout
    exes = []
    for line in out.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg.get("executable") \
                and msg["target"]["name"] != "engine_cuda":
            exes.append(Path(msg["executable"]))
    if not exes:
        raise RuntimeError("cargo produced no engine-cuda test executables")
    # Byte-reproducible (no timestamps or owners), so an unchanged build has the
    # same hash and run_cuda_tests.sh / CI can skip a GPU run that already passed.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for exe in sorted(exes):
                # Strip cargo's hash suffix: cudasim-0123abcd -> cudasim
                info = tar.gettarinfo(exe, arcname=exe.name.rsplit("-", 1)[0])
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                with open(exe, "rb") as f:
                    tar.addfile(info, f)
    return buf.getvalue()


def filter_env(spec: str) -> list[tuple[str, str]]:
    """`K=V,K2=V2` -> [(K, V), ...]; only ENGINE_*/RUST_* names pass."""
    out = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        k, _, v = part.partition("=")
        if not (k.startswith("ENGINE_") or k.startswith("RUST_")):
            raise ValueError(f"--env: {k} is not an ENGINE_* or RUST_* variable")
        out.append((k, v))
    return out


def _run_tests(
    archive: bytes, perf: bool, filter: str = "", sync: bool = False, sanitize: str = "",
    env_vars: str = "",
) -> dict:
    started = time.time()
    root = Path("/tmp/tests")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    log = [f"gpu: {smi.stdout.strip() or smi.stderr.strip()}"]
    env = {**os.environ, "ENGINE_CUDA_DEVICE": "1", "RUST_BACKTRACE": "1"}
    if perf:
        env["ENGINE_CUDA_PERF"] = "1"
    if sync:
        env["ENGINE_CUDA_SYNC"] = "1"
    for kv in filter_env(env_vars):
        env[kv[0]] = kv[1]
        log.append(f"env: {kv[0]}={kv[1]}")
    wrap = []
    if sanitize:
        wrap = [f"/usr/local/cuda-{CUDA_PKG}/bin/compute-sanitizer", "--tool", sanitize,
                "--print-limit", "8", "--show-backtrace", "device"]
    passed, failed, perf_lines = [], [], []
    for exe in sorted(root.iterdir()):
        exe.chmod(0o755)
        listed = subprocess.run(
            [str(exe), "--list", "--format", "terse", *([filter] if filter else [])],
            env=env, capture_output=True, text=True, check=True,
        ).stdout
        tests = [l[: -len(": test")] for l in listed.splitlines() if l.endswith(": test")]
        if perf:
            tests = [t for t in tests if "perf" in t]
        # One process per test: a CUDA fault poisons its context, so it must
        # not fail the tests after it. Serial: one GPU, no memory contention.
        for t in tests:
            t0 = time.time()
            r = subprocess.run(
                [*wrap, str(exe), t, "--exact", "--test-threads=1", "--nocapture"],
                env=env, capture_output=True, text=True,
            )
            status = "ok" if r.returncode == 0 else "FAILED"
            log.append(f"--- {exe.name}::{t} {status} ({time.time() - t0:.1f}s)")
            # Passing tests print too when picked by name (traces, timings).
            if r.returncode != 0 or perf or filter:
                log.append(r.stdout + r.stderr)
            (passed if r.returncode == 0 else failed).append(t)
            # --nocapture prints libtest's "test name ... " prefix on the same line.
            perf_lines += [l[l.index('{"engine"'):] for l in r.stdout.splitlines() if '{"engine"' in l]
    log.append(f"{len(passed)} passed, {len(failed)} failed: {failed}")
    log.append(f"gpu container: {time.time() - started:.1f}s")
    return {"ok": not failed and bool(passed), "log": "\n".join(log), "perf": perf_lines}


# single_use_containers: the container exits after its one input, so no idle
# GPU time is billed after the tests finish (vs a scaledown window).
GPU_FUNCTION = dict(
    image=gpu_image, cpu=1, memory=4096, timeout=600, retries=0, single_use_containers=True,
)


@app.function(gpu="T4", **GPU_FUNCTION)
def gpu_tests(
    archive: bytes, perf: bool = False, filter: str = "", sync: bool = False, env_vars: str = ""
) -> dict:
    return _run_tests(archive, perf, filter, sync, env_vars=env_vars)


@app.function(gpu="T4", **{**GPU_FUNCTION, "image": gpu_debug_image, "timeout": 1800})
def gpu_tests_sanitized(
    archive: bytes, perf: bool = False, filter: str = "", sync: bool = False, tool: str = "memcheck",
    env_vars: str = "",
) -> dict:
    return _run_tests(archive, perf, filter, sync, sanitize=tool, env_vars=env_vars)


def _linux_builder() -> bool:
    return sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")


@app.local_entrypoint()
def main(
    archive: str = "",
    gpu: str = "T4",
    perf: bool = False,
    filter: str = "",
    sync: bool = False,
    sanitize: str = "",
    env: str = "",
):
    """Run engine-cuda's tests on one GPU from --archive (or a native Linux build).

    --perf runs only the timed test (`ENGINE_CUDA_PERF=1`); use --gpu L40S
    for the RTX 4090-class die, T4 for the cheapest correctness runs.
    --filter NAME runs the tests whose names contain NAME; --sync waits after
    every kernel (`ENGINE_CUDA_SYNC=1`) so a GPU fault names its kernel;
    --sanitize TOOL runs each test under compute-sanitizer (memcheck, racecheck,
    synccheck or initcheck; slow, so use --filter). --env K=V,K2=V2 sets
    extra variables for the tests (e.g. ENGINE_* control values).
    """
    if archive:
        data = Path(archive).read_bytes()
    elif _linux_builder():
        print("building engine-cuda tests natively")
        data = package_tests(ENGINE, {"ENGINE_CUDA_REQUIRE_PTX": "1", **dict(filter_env(env))})
    else:
        raise SystemExit(
            "no --archive: build for free with engine/tools/modal/run_cuda_tests.sh "
            "(local Docker), or pass an archive from engine/tools/cuda/build_tests_in_docker.sh"
        )
    print(f"archive: {len(data) / 1e6:.1f} MB")
    fn = gpu_tests_sanitized if sanitize else gpu_tests
    if gpu != "T4":
        fn = fn.with_options(gpu=gpu)
    if sanitize:
        result = fn.remote(data, perf, filter, sync, sanitize, env)
    else:
        result = fn.remote(data, perf, filter, sync, env)
    print(result["log"])
    for line in result["perf"]:
        print(line)
    if not result["ok"]:
        raise SystemExit("engine-cuda GPU tests failed")


if __name__ == "__main__":
    # CI packaging without a Modal login: python cuda_tests.py --package OUT
    if len(sys.argv) == 3 and sys.argv[1] == "--package":
        Path(sys.argv[2]).write_bytes(package_tests(ENGINE, {"ENGINE_CUDA_REQUIRE_PTX": "1"}))
    else:
        raise SystemExit(__doc__)
