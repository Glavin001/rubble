"""Build side of gpu_run.sh: runs inside the x86_64 build container (no GPU, no login).

    python3 gpu_run_pack.py OUT_DIR [--data PATH]... -- <cargo selection>

Builds the selection with cargo (release), plus engine-webgpu's `gpu_adapter` probe,
strips the executables and writes OUT_DIR/archive.tar.gz holding only:

* bin/<name>        the stripped executables (cargo's hash suffix dropped),
* data/<abs path>   the small run-time files they read, at the absolute paths the
                    binaries have baked in (/repo/..., and their build scripts'
                    generated WGSL under the target dir), plus OUT/stage/** (files
                    from outside the repo that gpu_run.sh staged, e.g. the sedan's
                    JBeam export at /scenarios/sedan_new),
* manifest.json     what to run, in which directory, and with what.

No source tree, no credentials: the data list is explicit (DATA below, the target
table TARGET_DATA, and --data paths, each checked for size).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path("/repo")
ENGINE = REPO / "engine"
# Always shipped (repo-relative; small text scenario files the tests and tools read).
DATA = [
    "engine/docs/explicit-beams-acceptance/vehicles",
    "engine/crates/engine-webgpu/vehicle-lab-scripts",
]
# Shipped when that target is selected: explicit_loop_cap checks the kernel sources'
# loop caps, so it reads them.
TARGET_DATA = {
    "explicit_loop_cap": ["engine/slang/kernels"],
}
# Directories under a build script's OUT_DIR the binaries read at run time
# (engine_webgpu::EXPLICIT_BEAMS_WGSL_DIR).
OUT_DIR_DATA = ["wgsl"]
MAX_DATA_BYTES = 32 << 20
DATA_SUFFIXES = {".truck", ".jbeam", ".pc", ".json", ".slang", ".wgsl", ".txt", ".csv", ".toml"}


def cargo_json(args: list[str]) -> list[dict]:
    env = {**os.environ, "CARGO_TERM_COLOR": "always"}
    p = subprocess.run(
        ["cargo", *args, "--release", "--message-format=json-render-diagnostics"],
        cwd=ENGINE, env=env, stdout=subprocess.PIPE, text=True,
    )
    msgs = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0:
        raise SystemExit(f"cargo {' '.join(args)} failed ({p.returncode})")
    return msgs


def split_selection(sel: list[str]) -> tuple[list[str], list[str]]:
    """cargo args, run args (after an inner `--`)."""
    if "--" in sel:
        i = sel.index("--")
        return sel[:i], sel[i + 1:]
    return sel, []


def main() -> None:
    argv = sys.argv[1:]
    out = Path(argv.pop(0))
    extra: list[str] = []
    while argv and argv[0] != "--":
        flag = argv.pop(0)
        if flag == "--data":
            extra.append(argv.pop(0))
        else:
            raise SystemExit(f"unknown option {flag}")
    if not argv:
        raise SystemExit("usage: gpu_run_pack.py OUT [--data P]... -- <cargo selection>")
    cargo_args, run_args = split_selection(argv[1:])
    if not cargo_args:
        raise SystemExit("empty cargo selection")
    is_test = any(a in ("--test", "--tests", "--lib", "--bench", "--benches", "--doc") for a in cargo_args)
    cmd = (["test", "--no-run"] if is_test else ["build"]) + cargo_args

    probe = cargo_json(["build", "-p", "engine-webgpu", "--bin", "gpu_adapter"])
    msgs = probe + cargo_json(cmd)
    exes: list[dict] = []
    out_dirs: list[Path] = []
    seen = set()
    for m in msgs:
        if m.get("reason") == "build-script-executed" and m.get("out_dir"):
            out_dirs.append(Path(m["out_dir"]))
        if m.get("reason") != "compiler-artifact" or not m.get("executable"):
            continue
        exe = Path(m["executable"])
        if exe in seen:
            continue
        seen.add(exe)
        kind = m["target"]["kind"][0]
        name = m["target"]["name"]
        pkg_dir = Path(m["manifest_path"]).parent
        is_probe = name == "gpu_adapter"
        if is_probe and "gpu_adapter" not in " ".join(cargo_args):
            role = "probe"
        else:
            role = "run"
        if role == "run" and kind == "lib":
            continue
        # `cargo test --no-run` also builds the package's bins (integration tests may
        # run them): only the test harnesses are the selection.
        if role == "run" and is_test and not m["profile"].get("test"):
            continue
        exes.append({
            "path": exe, "name": name, "kind": "test" if m["profile"].get("test") else kind,
            "cwd": str(pkg_dir if m["profile"].get("test") else ENGINE), "role": role,
        })
    runs = [e for e in exes if e["role"] == "run"]
    if not runs:
        raise SystemExit("the selection built no executables (did you pass --test/--example/--bin?)")

    files: dict[str, Path] = {}

    def add_tree(p: Path, suffixes: set[str] | None) -> None:
        if p.is_file():
            files[str(p)] = p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and (suffixes is None or f.suffix in suffixes):
                    files[str(f)] = f
        else:
            raise SystemExit(f"data path {p} does not exist")

    for rel in DATA:
        add_tree(REPO / rel, DATA_SUFFIXES)
    for e in runs:
        for rel in TARGET_DATA.get(e["name"], []):
            add_tree(REPO / rel, DATA_SUFFIXES)
    user_bytes = sum(f.stat().st_size for f in files.values())
    for od in out_dirs:
        for sub in OUT_DIR_DATA:
            if (od / sub).is_dir():
                add_tree(od / sub, None)
    generated = sum(f.stat().st_size for f in files.values()) - user_bytes
    for p in extra:
        add_tree(REPO / p, None)
    # Files from outside the repo (gpu_run.sh copies them to OUT/stage/<container path>).
    stage = out / "stage"
    staged: dict[str, Path] = {}
    if stage.is_dir():
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                staged["/" + str(f.relative_to(stage))] = f
    total = sum(f.stat().st_size for f in [*files.values(), *staged.values()])
    # The cap is on what is not build output (the generated WGSL, ~60 MB of text that
    # compresses about 10x, is read at run time by the kernel variants and A/B paths).
    if total - generated > MAX_DATA_BYTES:
        raise SystemExit(
            f"data is {(total - generated) / 1e6:.1f} MB, over the {MAX_DATA_BYTES >> 20} MB cap"
        )

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"run_args": run_args, "cargo": cargo_args, "exes": [], "data_bytes": total}
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for e in exes:
                stripped = out / f"{e['name']}.{e['kind']}"
                # Debug info off, symbols kept (panics' backtraces name functions).
                subprocess.run(["strip", "--strip-debug", "-o", str(stripped), str(e["path"])], check=True)
                arc = f"bin/{stripped.name}"
                tar.add(stripped, arcname=arc)
                stripped.unlink()
                manifest["exes"].append({k: e[k] for k in ("name", "kind", "cwd", "role")} | {"file": arc})
            for key, f in sorted([*files.items(), *staged.items()]):
                tar.add(f, arcname="data" + key)
            blob = json.dumps(manifest, indent=1).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    (out / "archive.tar.gz").write_bytes(buf.getvalue())
    names = ", ".join(f"{e['name']} ({e['kind']})" for e in runs)
    print(f"packed {names}; {len(files) + len(staged)} data files ({total / 1e6:.1f} MB, {generated / 1e6:.1f} MB of it generated WGSL); "
          f"archive {len(buf.getvalue()) / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
