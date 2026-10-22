"""Builds the cross-machine determinism parity kit (data v2, step 0) from this machine's
recordings: 50 viability events spread over the families, with their recorded end
fingerprints, the full state of 5 of them, and the sha256 of their /5 calls' decoded arrays.

    python parity/make_kit.py --runs runs --calls runs/calls-v5 --out KIT_DIR

The kit holds per event: recipe (replay.json), scene.json, context.json, outcome.json, and
for the 5 full events also full_state/ and the arrays the call builder reads. Run
parity/check.sh with it on the other machine.
"""
import argparse, json, random, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import calls_sha256

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--calls", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
random.seed(20261022)
by_fam = {}
for d in sorted(a.runs.glob("viability-v1-*/events/*")):
    if (d / "outcome.json").exists() and (d / "replay.json").exists():
        fam = json.loads((d / "context.json").read_text())["family"]
        by_fam.setdefault(fam, []).append(d)
fams = sorted(by_fam)
pick = []
while len(pick) < 50:
    for f in fams:
        rest = [d for d in by_fam[f] if d not in pick]
        if rest and len(pick) < 50:
            pick.append(random.choice(rest))
full = [d for d in pick if (a.calls / d.name / "manifest.json").exists()][:5]
a.out.mkdir(parents=True, exist_ok=True)
manifest = {"events": [], "full": []}
for d in pick:
    dst = a.out / "events" / d.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("replay.json", "scene.json", "context.json", "outcome.json"):
        shutil.copy2(d / f, dst / f)
    o = json.loads((d / "outcome.json").read_text())
    manifest["events"].append({"event": d.name, "family": json.loads((d / "context.json").read_text())["family"], "frames": o["frames"], "fingerprint": o["fingerprint"]})
for d in full:
    dst = a.out / "events" / d.name
    for x in ("full_state", "arrays"):
        shutil.copytree(d / x, dst / x, dirs_exist_ok=True)
    for f in ("catalog.json", "frames.json"):
        shutil.copy2(d / f, dst / f)
    n, sha = calls_sha256(a.calls / d.name)
    manifest["full"].append({"event": d.name, "calls": n, "calls_sha256": sha})
(a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(len(manifest["events"]), "events,", len(manifest["full"]), "with full state and calls")
