"""The CPU reference solver (`stress-ref`) on Modal's CPUs: one scene a container, all at
once, results kept by content.

The reference is pure CPU and deterministic, so nothing here needs a GPU or this
machine: the destruction scorecard's reference outcomes (engine/docs/world-on-cuda.md,
"the accuracy side") are made here and cached. A result is keyed by the bytes of the
`stress-ref` crate and the run's arguments; it is computed once.

Usage from the repo root (`modal.sh` runs the pinned Modal CLI in Docker):

    # once, and after this file changes: the functions the runs go on in
    engine/tools/modal/modal.sh deploy engine/tools/modal/cpu_run.py
    # start every scene's run and return at once (seconds locally; the runs go on)
    engine/tools/modal/modal.sh run engine/tools/modal/cpu_run.py::submit --scenes b2_cantilever,b4_support_loss_sudden
    engine/tools/modal/modal.sh run engine/tools/modal/cpu_run.py::submit --scenes @standing
    # what is done, what is still running; write the finished observations under
    # engine/stress-oracle/reference/modal/<key>/
    engine/tools/modal/modal.sh run engine/tools/modal/cpu_run.py::collect

`--args "--mode explicit --feature softening=off"` passes options to `stress-ref run`
(part of the result's key). `--sets` lists the named scene sets.

Cost: 2 cores and 2 GiB a scene, about $0.00003 a second; the catalogue's scenes take
0.5 s to 2 minutes each, `house_1story_ram`-sized ones about 11 minutes.
"""

import hashlib
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path

import modal

ENGINE = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
CRATE = "stress-oracle/crates/stress-ref"

app = modal.App("rubble-stress-ref")

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.12")
    .apt_install("curl", "ca-certificates", "build-essential")
    .env({"CARGO_HOME": "/opt/cargo", "RUSTUP_HOME": "/opt/rustup"})
    .run_commands("curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
)
# /store/bin/<key>/stress-ref, /store/results/<key>/<run>.json, /store/runs/<key>/<run>.started
store = modal.Volume.from_name("rubble-stress-ref", create_if_missing=True)

# Scene sets of the scorecard, by what they test (stress-ref's own catalogue:
# `stress-ref gen-scenes`).
SETS = {
    "standing": [
        "b2_cantilever",
        "b4_support_loss_sudden",
        "b4_support_loss_gradual",
        "b8_frame_sudden",
        "b8_frame_gradual",
        "s_overhang_thin",
        "s_overhang_thick",
        "s_supports_one_by_one",
        "s_arch",
        "s_arch_keystone_removed",
        "s_house_settlement",
        "s_floor_drop",
    ],
    "impact": [
        "b5_wall_impact_v02",
        "b5_wall_impact_v10",
        "b5_wall_impact_v40",
        "b6_spall",
        "b9_panel_low",
        "b9_panel_high",
        "s_house_car",
        "s_car_brick",
    ],
    "blast": ["s_blast_two_walls", "s_house_blast"],
}

# The workspace the crate builds in alone (the repository's has members that need the
# engine's crates).
WORKSPACE = """[workspace]
resolver = "2"
members = ["crates/stress-ref"]

[workspace.package]
edition = "2021"
license = "MIT"
publish = false

[profile.release]
debug = 1
"""


def crate_archive() -> tuple[bytes, str]:
    """The crate's source and the workspace's lock file as a tar.gz, and its key. (Read
    from the directory, not from git: the Modal CLI's container has no repository.)"""
    root = ENGINE / CRATE
    names = [f"{CRATE}/{p.relative_to(root).as_posix()}" for p in root.rglob("*") if p.is_file() and "target" not in p.relative_to(root).parts and not p.name.startswith(".")]
    names = sorted(names + ["stress-oracle/Cargo.lock"])
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in names:
            data = (ENGINE / name).read_bytes()
            digest.update(name.encode() + b"\0" + data)
            info = tarfile.TarInfo(name.removeprefix("stress-oracle/"))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue(), digest.hexdigest()[:16]


def run_key(scene: str, args: str) -> str:
    tag = hashlib.sha256(args.encode()).hexdigest()[:8] if args else "default"
    return f"{scene}__{tag}"


@app.function(image=image, volumes={"/store": store}, cpu=8, memory=8192, timeout=1500)
def build(archive: bytes, key: str) -> str:
    """`stress-ref` built from `archive`, kept at /store/bin/<key>."""
    out = Path(f"/store/bin/{key}/stress-ref")
    if out.is_file():
        return "kept"
    work = Path("/tmp/build")
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(work)
    (work / "Cargo.toml").write_text(WORKSPACE)
    t0 = time.time()
    subprocess.run(["/opt/cargo/bin/cargo", "build", "--release", "-p", "stress-ref", "--bin", "stress-ref"], cwd=work, check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((work / "target/release/stress-ref").read_bytes())
    out.chmod(0o755)
    store.commit()
    return f"built in {time.time() - t0:.0f} s"


@app.function(image=image, volumes={"/store": store}, cpu=2, memory=2048, timeout=3600)
def run(key: str, scene: str, args: str) -> str:
    """One scene through `stress-ref run`; the observation and how it was made, kept at
    /store/results/<key>/<run>.json."""
    name = run_key(scene, args)
    result = Path(f"/store/results/{key}/{name}.json")
    store.reload()
    if result.is_file():
        return "kept"
    binary = f"/store/bin/{key}/stress-ref"
    scenes = Path("/tmp/scenes")
    subprocess.run([binary, "gen-scenes", str(scenes)], check=True, capture_output=True)
    # The benchmarks, then the showcases (`scenes/showcases/`).
    found = sorted(scenes.rglob(f"{scene}.json"))
    failed = Path(f"/store/results/{key}/{name}.failed")

    def fail(why: str) -> str:
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(why)
        store.commit()
        return f"failed: {why[:200]}"

    if not found:
        known = sorted(p.stem for p in scenes.rglob("*.json"))
        return fail(f"{scene}: not in the catalogue ({', '.join(known)})")
    source = found[0]
    out = Path("/tmp/observation.json")
    t0 = time.time()
    done = subprocess.run([binary, "run", str(source), "--out", str(out), *args.split()], capture_output=True, text=True)
    seconds = time.time() - t0
    if done.returncode != 0:
        return fail(f"{scene}: stress-ref run exited {done.returncode} after {seconds:.0f} s:\n{done.stderr[-4000:]}")
    if failed.is_file():
        failed.unlink()
    record = {
        "scene": scene,
        "args": args,
        "stress_ref": key,
        "seconds": round(seconds, 2),
        "cores": 2,
        "log": done.stderr[-2000:],
        "observation": json.loads(out.read_text()),
    }
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(record, indent=1))
    store.commit()
    return f"{seconds:.1f} s"


@app.function(image=image, volumes={"/store": store}, timeout=120)
def results(key: str) -> dict[str, str]:
    """Every finished run of `key`: name -> its record's text."""
    store.reload()
    folder = Path(f"/store/results/{key}")
    if not folder.is_dir():
        return {}
    done = {p.stem: p.read_text() for p in sorted(folder.glob("*.json"))}
    done.update({f"{p.stem}.failed": p.read_text() for p in sorted(folder.glob("*.failed"))})
    return done


def deployed(name: str):
    try:
        f = modal.Function.from_name("rubble-stress-ref", name)
        f.hydrate()
        return f
    except modal.exception.NotFoundError:
        raise SystemExit("cpu_run: not deployed: engine/tools/modal/modal.sh deploy engine/tools/modal/cpu_run.py") from None


def expand(scenes: str) -> list[str]:
    out = []
    for item in scenes.split(","):
        item = item.strip()
        if item.startswith("@"):
            if item[1:] not in SETS:
                raise SystemExit(f"cpu_run: no scene set {item} (sets: {', '.join(SETS)})")
            out += SETS[item[1:]]
        elif item:
            out.append(item)
    return out


@app.local_entrypoint()
def submit(scenes: str = "", args: str = "", sets: bool = False):
    """Start each scene's run and return: the runs go on without this command."""
    if sets:
        for name, members in SETS.items():
            print(f"@{name}: {', '.join(members)}")
        return
    names = expand(scenes)
    if not names:
        raise SystemExit("cpu_run: --scenes NAME[,NAME...] or @SET (--sets lists them)")
    archive, key = crate_archive()
    # The deployed functions: a run started from them goes on after this command ends.
    print(f"cpu_run: stress-ref {key} ({len(archive) / 1e6:.2f} MB of source): {deployed('build').remote(archive, key)}")
    for scene in names:
        call = deployed("run").spawn(key, scene, args)
        print(f"cpu_run: {run_key(scene, args)} started ({call.object_id})")
    print(f"cpu_run: {len(names)} runs started; `cpu_run.py::collect` writes the finished ones")


@app.local_entrypoint()
def collect(out: str = ""):
    """Write every finished observation of this checkout's `stress-ref` under
    engine/stress-oracle/reference/modal/<key>/ (or `--out DIR`)."""
    _, key = crate_archive()
    done = deployed("results").remote(key)
    folder = Path(out) if out else ENGINE / "stress-oracle/reference/modal" / key
    folder.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, text in done.items():
        if name.endswith(".failed"):
            failures += 1
            print(f"cpu_run: {name}: {text.splitlines()[0][:200]}")
            continue
        (folder / f"{name}.json").write_text(text)
        record = json.loads(text)
        print(f"cpu_run: {name}: {record['seconds']} s")
    print(f"cpu_run: {len(done) - failures} observations of stress-ref {key} in {folder}; {failures} failed")
