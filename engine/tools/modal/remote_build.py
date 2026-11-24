"""Build on Modal's CPUs (no GPU): the Mac only sends the source that changed and reads
the result. The archive lands in the GPU runner's archive volume (gpu_run.py), so a GPU
stage can run it without the Mac downloading or uploading a binary.

    engine/tools/modal/remote_build.sh -- <cargo selection>   (from the repo)

Where things live, per tree (a worktree's path hashed: its "key"):
* the source: `rubble-rb-src`, one zstd snapshot of the tree (`KEY.src.tar.zst`) and its
  manifest (path -> sha256). The Mac sends only the files whose content changed, inside
  the call; the container applies them and writes the snapshot back.
* cargo's target: `rubble-rb-target`, one zstd snapshot (`KEY.tar.zst`).
* CARGO_HOME (registry): `rubble-rb-cargo`, one snapshot, rewritten when Cargo.lock changes.
* nvcc's fatbins and slangc's outputs: `rubble-rb-kernels`, by content (few, large files;
  every tree and session shares them).
* the packed archive: `rubble-gpu-run-archives`, by SHA-256 (gpu_run.py's).

Cargo itself works on the container's local disk. On a Volume (network storage) every
fingerprint check was a round trip: nothing changed took 59 s, a Rust edit 120 s.
Snapshots are one sequential read and write each, and a warm container (kept 60 s after a
build) still has the tree and target on its disk, so it restores nothing: each snapshot
carries a version, and a container whose disk holds that version skips it.

The build image is engine/tools/cuda/Dockerfile: the x86_64 toolchain CI and the old
local route use, so the device code is the reference build's.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import modal

app = modal.App("rubble-remote-build")

CTX = "/ctx/cuda"
if modal.is_local():
    # (add_python: Modal's own runtime; the image's python3 3.10 stays for the packer.)
    build_image = (modal.Image.from_dockerfile(f"{CTX}/Dockerfile", context_dir=CTX, add_python="3.12")
                   .apt_install("zstd")
                   # (The web build, mode "web": wasm32 for the World wasm.)
                   .run_commands("rustup target add wasm32-unknown-unknown"))
else:
    build_image = modal.Image.debian_slim()


def _vol(name: str) -> modal.Volume:
    return modal.Volume.from_name(name, create_if_missing=True, version=2)


src_vol = _vol("rubble-rb-src")
target_vol = _vol("rubble-rb-target")
cargo_vol = _vol("rubble-rb-cargo")
kern_vol = _vol("rubble-rb-kernels")
archive_vol = modal.Volume.from_name("rubble-gpu-run-archives", create_if_missing=True)
# The web build's bundles for the Dawn runner (engine/tools/dawn), by content.
dawn_vol = modal.Volume.from_name("rubble-dawn-bundles", create_if_missing=True)
# The scenario matrix's saved states by SHA-256 (gpu_run.py's): the module list reads them.
states_vol = modal.Volume.from_name("rubble-scenario-states", create_if_missing=True)

BUILD_CPUS = 16
WARM_SECS = 60
CPU_PRICE, MEM_PRICE, MEM_GIB = 0.0000131, 0.00000222, 16


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


class _Snap:
    """A directory kept on local disk and as one zstd snapshot on a Volume, with a
    version: restore only when the disk's version is not the Volume's."""

    def __init__(self, local: Path, snap: Path, times: dict, name: str):
        self.local, self.snap, self.times, self.name = local, snap, times, name
        self.ver = snap.with_name(snap.name + ".version")
        self.mark = local.parent / f".{local.name}.version"

    def restore(self) -> bool:
        want = _read(self.ver)
        if want and _read(self.mark) == want and self.local.is_dir():
            self.times[self.name + "_in"] = 0.0
            return False
        t = time.time()
        shutil.rmtree(self.local, ignore_errors=True)
        self.local.mkdir(parents=True)
        if self.snap.exists():
            subprocess.run(["bash", "-c", f"zstd -dc -T0 '{self.snap}' | tar -x -C '{self.local}'"], check=True)
        self.mark.write_text(want)
        self.times[self.name + "_in"] = round(time.time() - t, 1)
        return True

    def save(self) -> None:
        t = time.time()
        tmp = self.snap.with_name(self.snap.name + ".tmp")
        subprocess.run(["bash", "-c", f"tar -C '{self.local}' -cf - . | zstd -q -T0 -1 -f -o '{tmp}'"], check=True)
        os.replace(tmp, self.snap)
        v = uuid.uuid4().hex
        self.ver.write_text(v)
        self.mark.write_text(v)
        self.times[self.name + "_out"] = round(time.time() - t, 1)


@app.cls(image=build_image, cpu=BUILD_CPUS, memory=MEM_GIB * 1024, timeout=1800, retries=0,
         scaledown_window=WARM_SECS,
         volumes={"/srcvol": src_vol, "/target": target_vol, "/cargo": cargo_vol,
                  "/kernels": kern_vol, "/archives": archive_vol, "/svol": states_vol,
                  "/dawnbundles": dawn_vol})
class Builder:
    """A build container. Cargo's target snapshot is written when the container stops
    (`flush`), not after each build: the next build on this warm container uses its disk,
    and a snapshot only serves a container that starts cold. A container that dies
    without flushing leaves the older snapshot, which is still correct (cargo rebuilds
    from it what changed since)."""

    @modal.enter()
    def start(self) -> None:
        self.dirty: dict[str, _Snap] = {}

    @modal.method()
    def build(self, spec: dict) -> dict:
        return _build(spec, self.dirty)

    @modal.method()
    def build_stream(self, spec: dict):
        """`build`, yielding each output line as it comes, then `{"result": ...}`: a
        deployed class's logs do not stream to a caller, a generator's values do."""
        import io
        import queue
        import sys
        import threading
        q: queue.Queue = queue.Queue()

        class Lines(io.TextIOBase):
            def write(self, s: str) -> int:
                q.put(("out", s))
                return len(s)

        done: dict = {}

        def work() -> None:
            old = sys.stdout
            sys.stdout = Lines()
            try:
                done["r"] = _build(spec, self.dirty)
            except Exception as e:  # noqa: BLE001 - reported to the caller
                done["r"] = {"code": 1, "error": repr(e), "setup_s": 0, "build_s": 0, "commit_s": 0, "total_s": 0}
            finally:
                sys.stdout = old
                q.put(("done", ""))

        threading.Thread(target=work, daemon=True).start()
        buf = ""
        while True:
            kind, s = q.get()
            if kind == "done":
                break
            buf += s
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield {"line": line}
        if buf:
            yield {"line": buf}
        yield {"result": done["r"]}

    @modal.exit()
    def flush(self) -> None:
        for snap in self.dirty.values():
            try:
                snap.save()
            except Exception as e:  # noqa: BLE001 - the older snapshot stays
                print(f"[remote_build] snapshot of {snap.local} not saved: {e}", flush=True)
        if self.dirty:
            target_vol.commit()


def _dawn_bundle(web: Path, car: Path) -> str:
    """The Dawn runner's bundle (engine/tools/dawn/run.sh lays out the same): the World
    wasm (no Asyncify copy) and the JS it needs, and the car's JBeam and .pc text; tarred
    in name order without times or owners, so the same build gives the same bundle,
    stored once by content."""
    import io
    import tarfile
    files = {f"bundle/{n}": web / n for n in ("rubble-world.mjs", "world-loop.mjs", "fracture/gpu-wait.mjs")}
    for f in sorted((web / "generated/world-wasm").rglob("*")):
        if f.is_file() and not f.name.endswith(".asyncify.wasm"):
            files[f"bundle/generated/world-wasm/{f.relative_to(web / 'generated/world-wasm')}"] = f
    names = json.loads((car / "files.json").read_text())
    for n in ["files.json", *[n for n in names if n.endswith((".jbeam", ".pc"))]]:
        files[f"car/{n}"] = car / n
    buf = io.BytesIO()
    import gzip
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name in sorted(files):
            info = tar.gettarinfo(str(files[name]), arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(files[name], "rb") as fh:
                tar.addfile(info, fh)
    data = buf.getvalue()
    name = hashlib.sha256(data).hexdigest()[:24] + ".tar.gz"
    dest = Path("/dawnbundles") / name
    if not dest.exists():
        dest.write_bytes(data)
    dawn_vol.commit()
    print(f"[remote_build] Dawn bundle {name} ({len(data) / 1e6:.0f} MB)", flush=True)
    return name


def _build(spec: dict, dirty: dict) -> dict:
    t0 = time.time()
    key = spec["key"]
    times: dict = {}
    # (kern_vol too: another container's cached module list and fatbins become visible.)
    for v in (src_vol, target_vol, cargo_vol, kern_vol):
        v.reload()
    work = Path("/work") / key

    # Source: the snapshot (when this disk does not hold it), then the changed files.
    src = _Snap(work / "repo", Path("/srcvol") / f"{key}.src.tar.zst", times, "src")
    src.restore()
    for rel in spec.get("removed", []):
        (src.local / rel).unlink(missing_ok=True)
    for rel, data in spec.get("files", {}).items():
        f = src.local / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)  # (written now: newer than the target's outputs, so cargo rebuilds)
    if spec.get("files") or spec.get("removed") or not src.snap.exists():
        src.save()
        (Path("/srcvol") / f"{key}.manifest.json").write_text(json.dumps(spec["manifest"]))
    repo = Path("/repo")
    if repo.is_symlink():
        repo.unlink()
    elif repo.is_dir() and not any(repo.iterdir()):
        repo.rmdir()  # (the image's WORKDIR)
    os.symlink(src.local, repo)

    target = _Snap(work / "target", Path("/target") / f"{key}.tar.zst", times, "target")
    target.restore()
    home = _Snap(Path("/work/cargo"), Path("/cargo/home.tar.zst"), times, "cargo")
    home.restore()
    lock = src.local / "engine/Cargo.lock"  # (gitignored: sent from the worktree when it has one)
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else ""
    lock_file = Path("/cargo/home.lock")

    cores = int(spec.get("cpus", BUILD_CPUS))
    env = {
        **os.environ,
        "CARGO_HOME": str(home.local),
        "CARGO_TARGET_DIR": str(target.local),
        "ENGINE_CUDA_FATBIN_CACHE": "/kernels/fatbins",
        "RUBBLE_SLANG_OBJ_CACHE": "/kernels/slang",
        "ENGINE_CUDA_REQUIRE_PTX": "1",
        "CARGO_BUILD_JOBS": str(spec.get("jobs") or cores),
        "RUBBLE_BUILD_JOBS": str(spec.get("kernel_jobs") or cores),
        # The archive stays on Modal: fast compression (the packer's default is 6).
        "GPU_RUN_PACK_GZIP": "1",
        **spec.get("env", {}),
    }
    if spec.get("modules_from") and not env.get("ENGINE_CUDA_EXPLICIT_MODULES"):
        # The modules the fixtures' layouts launch (`explicit_replay --modules`, no device
        # code), built in a target directory of its own so the two builds never
        # invalidate each other; it lives inside the target, so it is snapshotted too.
        t = time.time()
        # The list depends on the fixtures (their layouts) and the Rust that maps a layout
        # to modules, never on kernel source; but the probe's build reruns slangc for
        # every module after any kernel edit (33-38 s). So the answer is kept by the
        # fixtures, the Rust sources and manifests, and the ENGINE_* settings: a kernel-only
        # edit reuses it, and any Rust change asks again.
        # (The client looks the answer up and passes it in: a container's view of a Volume
        # did not show another container's entry even after reload().)
        memo = Path("/kernels/module-lists") / (_modules_key(spec) + ".txt")
        if spec.get("modules_memo"):
            env["ENGINE_CUDA_EXPLICIT_MODULES"] = spec["modules_memo"]
            times["modules"] = 0.0
        else:
            states_vol.reload()
            probe_env = {**env, "ENGINE_CUDA_EXPLICIT_MODULES": "none",
                         "CARGO_TARGET_DIR": str(target.local / "modules-probe")}
            p = subprocess.run(["cargo", "run", "-q", "--profile", "fast", "-p", "engine-cuda", "--example",
                                "explicit_replay", "--", "--modules", *[f"/svol/{n}" for n in spec["modules_from"]]],
                               cwd="/repo/engine", env=probe_env, capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stderr[-3000:], flush=True)
                return {"code": p.returncode, "setup_s": time.time() - t0, "build_s": 0.0, "cores": cores,
                        "commit_s": 0.0, "steps": times, "total_s": time.time() - t0,
                        "error": "explicit_replay --modules failed"}
            env["ENGINE_CUDA_EXPLICIT_MODULES"] = p.stdout.strip().splitlines()[-1]
            memo.parent.mkdir(parents=True, exist_ok=True)
            memo.write_text(env["ENGINE_CUDA_EXPLICIT_MODULES"])
            times["modules"] = round(time.time() - t, 1)
    out = Path("/tmp/out")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    # Files from outside the repo the archive ships (gpu_run.sh's OUT/stage: the car's
    # JBeam export), sent inside the call: {container path: bytes}.
    for path, data in spec.get("stage", {}).items():
        f = out / "stage" / path.lstrip("/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    stamp = Path("/tmp/before-build")
    stamp.touch()
    t_setup = time.time() - t0
    web = spec.get("mode") == "web"
    if web:
        # The release web build's kernels and World parts (kernels embedded, as the
        # pages ship) and the generated sedan from this tree's code (AGENTS.md: the car
        # for every benchmark), for the Dawn runner.
        env["RUBBLE_TOOLS"] = "/kernels/rubble-tools"
        env["RUBBLE_WEB_PARTS"] = "kernels world"
        cmd = ["bash", "-c", "set -e; bash scripts/build-web.sh; "
               "cargo run --release -q -p engine-softbody-lab --example vehicle_gen_rig -- sedan --out /tmp/out/car"]
        print(f"[remote_build] {key}: web build (kernels, world) and the generated sedan on {cores} cores", flush=True)
    else:
        cmd = ["python3", "/repo/engine/tools/modal/gpu_run_pack.py", str(out), *spec.get("pack_opts", []),
               "--", *spec["selection"]]
        print(f"[remote_build] {key}: {' '.join(spec['selection'])} on {cores} cores", flush=True)
    t = time.time()
    # What compiled, and for how long: a sampler watches the process table for nvcc and
    # slangc (twice a second) and keeps each process's module and run time. Watching from
    # outside leaves the build and its cache keys untouched (a wrapper at NVCC would
    # change the fatbin cache's key, which covers the compiler's own bytes).
    import threading
    seen: dict[int, list] = {}
    stop = threading.Event()

    def sample() -> None:
        # (/proc directly: the build image has no `ps`.)
        while not stop.is_set():
            now = time.time() - t
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                try:
                    words = Path(f"/proc/{d}/cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                words = [w.decode(errors="replace") for w in words if w]
                if not words:
                    continue
                tool = words[0].rsplit("/", 1)[-1]
                if tool not in ("nvcc", "slangc"):
                    continue
                pid = int(d)
                if pid not in seen:
                    src = next((w for w in words if w.endswith((".cu", ".slang"))), "?")
                    seen[pid] = [tool, src.rsplit("/", 1)[-1], now, now]
                seen[pid][3] = now
            stop.wait(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    # (Each line through print: build_stream's caller sees it as it comes.)
    proc = subprocess.Popen(cmd, cwd="/repo/engine", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    code = proc.wait()
    t_build = time.time() - t
    stop.set()
    sampler.join(timeout=2)
    compiled = sorted(((round(e - s, 1), round(e, 1), m) for tool, m, s, e in seen.values() if tool == "nvcc"),
                      reverse=True)
    slang_new = sum(1 for v in seen.values() if v[0] == "slangc")
    result = {"code": code, "setup_s": t_setup, "build_s": t_build, "cores": cores}
    if code == 0 and web:
        result["bundle"] = _dawn_bundle(Path("/repo/engine/crates/engine-webgpu"), out / "car")
        keyfile = Path("/dawnbundles/web-keys") / (_web_key(spec) + ".txt")
        keyfile.parent.mkdir(parents=True, exist_ok=True)
        keyfile.write_text(result["bundle"])
        dawn_vol.commit()
    elif code == 0:
        data = (out / "archive.tar.gz").read_bytes()
        sha = hashlib.sha256(data).hexdigest() + ".tar.gz"
        dest = Path("/archives") / sha
        if not dest.exists():
            dest.write_bytes(data)
        result.update(archive=sha, archive_mb=len(data) / 1e6)
        # The manifest travels back, so the GPU stage needs no copy of the archive here.
        import tarfile
        with tarfile.open(out / "archive.tar.gz", mode="r:gz") as tar:
            result["manifest"] = json.load(tar.extractfile("manifest.json"))  # type: ignore[arg-type]
        # The target's snapshot: now when there is none yet, else when the container
        # stops, and only if cargo wrote something (cargo touches .cargo-lock every run).
        changed = subprocess.run(["find", str(target.local), "-newer", str(stamp), "-type", "f",
                                  "!", "-name", ".cargo-lock", "-print", "-quit"],
                                 capture_output=True, text=True).stdout.strip()
        if not target.snap.exists():
            target.save()
        elif changed:
            dirty[key] = target
        if _read(lock_file) != lock_hash or not home.snap.exists():
            home.save()
            lock_file.write_text(lock_hash)
    t_c = time.time()
    for v in (src_vol, target_vol, kern_vol, cargo_vol, archive_vol):
        v.commit()
    result["commit_s"] = time.time() - t_c
    result["steps"] = times
    result["nvcc"] = compiled  # [(seconds it ran, finished at s after the build began, module)], longest first
    result["slangc_runs"] = slang_new
    result["total_s"] = time.time() - t0
    return result


def _web_key(spec: dict) -> str:
    """The web build's key (mode "web"): every source file's content except the tools and
    docs no web build reads, and the ENGINE_* settings. The same key, the same bundle:
    the client reuses it and calls no build (a web build reruns wasm-opt and Asyncify
    each time, about 6 minutes, even with nothing changed)."""
    h = hashlib.sha256()
    for path, digest in sorted(spec.get("manifest", {}).items()):
        if path.startswith(("engine/tools/", "engine/docs/")) or path.endswith(".md"):
            continue
        h.update(f"{path} {digest}\n".encode())
    for k, v in sorted(spec.get("env", {}).items()):
        h.update(f"{k}={v}\n".encode())
    return h.hexdigest()[:32]


def _modules_key(spec: dict) -> str:
    """The module list's key: the fixtures, the Rust sources and manifests (by content) and
    the ENGINE_* settings; not the kernel sources, which do not change the list."""
    h = hashlib.sha256()
    for path, digest in sorted(spec.get("manifest", {}).items()):
        if path.endswith((".rs", ".toml", ".lock")):
            h.update(f"{path} {digest}\n".encode())
    for k, v in sorted(spec.get("env", {}).items()):
        h.update(f"{k}={v}\n".encode())
    for n in spec["modules_from"]:
        h.update(f"fixture {n}\n".encode())
    return h.hexdigest()[:32]


def _diff(key: str, root: Path, files: dict[str, str]) -> tuple[dict, list[str], int]:
    """The files whose content changed since the tree's last build (its manifest on the
    Volume), read here to travel inside the call, and the files gone."""
    try:
        old = json.loads(b"".join(src_vol.read_file(f"{key}.manifest.json")))
    except Exception:  # noqa: BLE001 - first build of this tree
        old = {}
    changed = {p: (root / p).read_bytes() for p, h in files.items() if old.get(p) != h}
    gone = [p for p in old if p not in files]
    return changed, gone, sum(len(b) for b in changed.values())


def main(spec: str) -> None:
    """The client (remote_build.sh runs it after `modal deploy`): the deployed function,
    so a warm container outlives this call and serves the next build."""
    remote = modal.Cls.from_name("rubble-remote-build", "Builder")().build_stream
    s = json.loads(Path(spec).read_text())
    root = Path(s["root"])
    manifest = {}
    for line in Path(s["manifest"]).read_text().splitlines():
        if line:
            h, p = line.split("\t", 1)
            manifest[p] = h
    t = time.time()
    changed, gone, nbytes = _diff(s["key"], root, manifest)
    print(f"[remote_build] source: {len(changed)} files changed ({nbytes / 1e6:.2f} MB), {len(gone)} removed, "
          f"{time.time() - t:.1f} s", flush=True)
    s.update(files=changed, removed=gone, manifest=manifest)
    if s.get("modules_from_local"):
        # The fixtures go to the states volume by content (as gpu_run.py names them), only
        # when missing: the GPU stage then finds them there too.
        t = time.time()
        have = {e.path.strip("/") for e in states_vol.listdir("/")}
        names, up = [], []
        for rel in s["modules_from_local"]:
            f = root / rel
            name = hashlib.sha256(f.read_bytes()).hexdigest() + f.suffix
            names.append(name)
            if name not in have:
                up.append((f, name))
        if up:
            with states_vol.batch_upload(force=True) as b:
                for f, name in up:
                    b.put_file(str(f), f"/{name}")
        s["modules_from"] = names
        try:
            s["modules_memo"] = b"".join(kern_vol.read_file(f"module-lists/{_modules_key(s)}.txt")).decode().strip()
        except Exception:  # noqa: BLE001 - not listed yet for these fixtures and this Rust
            pass
        print(f"[remote_build] fixtures for the module list: {len(names)}, {len(up)} uploaded, "
              f"{time.time() - t:.1f} s", flush=True)
    if s.get("stage_dir"):
        stage = Path(s["stage_dir"])
        s["stage"] = {"/" + str(f.relative_to(stage)): f.read_bytes() for f in sorted(stage.rglob("*")) if f.is_file()}
    if s.get("mode") == "web":
        # The same web build's sources and settings as a stored bundle: that bundle, no build.
        try:
            name = b"".join(dawn_vol.read_file(f"web-keys/{_web_key(s)}.txt")).decode().strip()
        except Exception:  # noqa: BLE001 - not built yet
            name = ""
        if name:
            print(f"[remote_build] web build unchanged: Dawn bundle {name} (no build)", flush=True)
            Path(s["result"]).write_text(json.dumps({"code": 0, "bundle": name, "reused": True}))
            return
    t = time.time()
    # The build's output as it comes (cargo, slangc, wasm-pack, ...), then its result.
    r = {"code": 1, "setup_s": 0, "build_s": 0, "commit_s": 0, "total_s": 0, "error": "no result"}
    for item in remote.remote_gen(s):
        if "line" in item:
            print("  | " + item["line"], flush=True)
        else:
            r = item["result"]
    call = time.time() - t
    rate = r.get("cores", BUILD_CPUS) * CPU_PRICE + MEM_GIB * MEM_PRICE
    print(f"[remote_build] build: exit {r['code']}, call {call:.0f} s (container {r['total_s']:.0f} s: setup "
          f"{r['setup_s']:.0f}, cargo+pack {r['build_s']:.0f}, commits {r['commit_s']:.0f}; {r.get('steps')}); "
          f"about ${r['total_s'] * rate:.4f} plus up to {WARM_SECS} s warm (${WARM_SECS * rate:.4f})", flush=True)
    if r.get("bundle"):
        print(f"[remote_build] Dawn bundle {r['bundle']} in rubble-dawn-bundles", flush=True)
        Path(s["result"]).write_text(json.dumps(r))
    if r.get("archive"):
        print(f"[remote_build] archive {r['archive']} ({r['archive_mb']:.0f} MB) in rubble-gpu-run-archives", flush=True)
        Path(s["result"]).write_text(json.dumps(r))
    nv = r.get("nvcc", [])
    print(f"[remote_build] nvcc ran {len(nv)} time(s), slangc {r.get('slangc_runs', 0)}"
          + (": " + ", ".join(f"{m} {d:.0f} s (done at {e:.0f} s)" for d, e, m in nv[:8]) if nv else ""), flush=True)
    if r["code"] != 0:
        raise SystemExit(r["code"])


if __name__ == "__main__":
    import sys

    main(sys.argv[sys.argv.index("--spec") + 1])
