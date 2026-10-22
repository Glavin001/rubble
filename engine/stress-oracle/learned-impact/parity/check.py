"""The cross-machine determinism parity check (data v2, step 0): run on the new machine
with the kit made on the reference machine (parity/make_kit.py).

    python parity/check.py --kit KIT_DIR --bin target/release [--jobs N]

1. the pinned tests: stress-ref goldens, the M5 scene pins, the full-state completeness gate;
2. every kit event's recipe replayed by the oracle: its end fingerprint must equal the
   recorded one (`learned_impact_outcome_record --replay`);
3. for the 5 full events: every boundary restored and stepped equals the next
   (`learned_impact_full_state_check`), and their /5 calls rebuilt hash the same.
Prints PASS or FAIL per item and exits non-zero on any failure: then this machine is not
bit-identical, and becomes the single reference machine (plan v2, step 0).
"""
import argparse, json, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import calls_sha256

ap = argparse.ArgumentParser()
ap.add_argument("--kit", type=Path, required=True)
ap.add_argument("--bin", type=Path, default=Path("target/release"))
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--skip-tests", action="store_true")
a = ap.parse_args()
m = json.loads((a.kit / "manifest.json").read_text())
fails = 0

def report(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)

if not a.skip_tests:
    for args in (["-p", "stress-ref", "--test", "goldens"], ["-p", "stress-learned", "--test", "outcome", "m5_scenes_are_p0s"], ["-p", "stress-learned", "--test", "full_state"]):
        r = subprocess.run(["cargo", "test", "--release", *args], capture_output=True, text=True)
        report("cargo test " + " ".join(args), r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])

def replay(e):
    r = subprocess.run([str(a.bin / "learned_impact_outcome_record"), "--replay", str(a.kit / "events" / e["event"])], capture_output=True, text=True)
    return e, r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1:]

with ThreadPoolExecutor(a.jobs) as ex:
    for e, ok, tail in ex.map(replay, m["events"]):
        report(f"replay {e['event']} ({e['family']})", ok, tail[0] if tail else "")

full = [a.kit / "events" / f["event"] for f in m["full"]]
r = subprocess.run([str(a.bin / "learned_impact_full_state_check"), "--jobs", str(a.jobs), *map(str, full)], capture_output=True, text=True)
report("full-state resume of the 5 full events", r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
with tempfile.TemporaryDirectory() as tmp:
    r = subprocess.run([str(a.bin / "learned_impact_outcome_calls"), "--out", tmp, "--jobs", str(a.jobs), *map(str, full)], capture_output=True, text=True)
    report("call build of the 5 full events", r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
    for f in m["full"]:
        d = Path(tmp) / f["event"]
        if d.exists():
            n, sha = calls_sha256(d)
            report(f"calls {f['event']}", (n, sha) == (f["calls"], f["calls_sha256"]), f"{n} calls")
print(f"\n{'ALL PASS: this machine reproduces the reference bit for bit' if fails == 0 else f'{fails} FAILED: this machine is not bit-identical'}")
sys.exit(1 if fails else 0)
