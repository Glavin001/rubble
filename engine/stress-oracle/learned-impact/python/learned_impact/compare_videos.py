"""Oracle vs trained-model comparison videos in harness A (P0).

Each scenario is one recorded P1 event (its `scene.json`, the member as recorded) rendered
by `stress-viz compare` with three panels in step at the scene's 1/60 s frame:

  oracle          the standalone CPU stress-ref oracle (the reference)
  learned model   harness A: the same oracle with every routed island's frame from the
                  checkpoint (`engine a_learned`: the model's decoded state applied to the
                  island on a copy of the frame-start world)
  quasi-static    CPU `blast_parity` (P1's QS baseline), for context

and a broken-bonds chart under them. Per scenario: `<name>.mp4`, `<name>_mid.png` (a
mid-impact frame) and a summary line; `index.md` explains each. Rendering is the software
rasterizer and the model runs on the CPU, so nothing here takes the GPU (P2 trains on MPS).

    python -m learned_impact.compare_videos --stress-viz BIN --checkpoint best.pt \\
        --python-dir DIR --model-label "bulk-w32 (calls /3, W32)" --p1-runs P1/runs \\
        --out ../runs/videos-2026-10-22 [--only m5_free_wall,...]

Run it under `scripts/job.py` (progress.log, status.json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# (name, event dir under the P1 runs root, what it shows)
SCENARIOS = [
    ("1_m5_free_wall", "dev/m5_free_wall",
     "M5 free wall: P0's pinned acceptance scene (200 kg steel ram at 40 m/s through a free "
     "concrete wall, no gravity). The oracle punches a clean hole (6 pieces)."),
    ("2_m5_wall_row", "dev/m5_wall_row",
     "M5 wall row: the same ram through a row of free walls (133 pieces in the oracle); the "
     "debris of one wall strikes the next."),
    ("3_anchored_gravity", "evalplus-val-anchored-t1/events/p1_anchored_wall_t1_00004_s3001_e5",
     "Anchored wall under gravity (validation split): a wall standing on fixed supports, "
     "perforated (104 pieces in the oracle)."),
    ("4_corner_hit", "evalplus-val-free-t1-corner/events/p1_free_wall_t1_00000_s3002_e0",
     "Corner hit (validation split, corner-aimed set): the ram strikes near a free wall's "
     "corner (25 pieces in the oracle)."),
    ("5_four_rams", "evalplus-val-free-t2/events/p1_free_wall_t2_00002_s3003_e4",
     "Four impactors on one free wall (validation split, tier 2 multi-load)."),
    ("6a_ballistic_limit_perforate", "evalplus-val-anchored-t1-corner/events/p1_anchored_wall_t1_00001_s3005_e2",
     "Near the ballistic limit, member that perforates: this ensemble splits 4 perforate / 4 "
     "arrest under tiny perturbations; here the oracle's ram just gets through (0.75 m/s)."),
    ("6b_ballistic_limit_arrest", "evalplus-val-anchored-t1-corner/events/p1_anchored_wall_t1_00001_s3005_e0",
     "Near the ballistic limit, member that arrests: the same ensemble, the oracle's ram stops "
     "in the wall (0.02 m/s)."),
    ("7_validation_event", "p1-bulk-v1-anchored-t1/events/p1_anchored_wall_t1_00083_s1002_e7",
     "A validation event from the bulk set the checkpoint was trained beside (never trained "
     "on): an anchored wall perforated at speed (159 pieces in the oracle)."),
]

SUMMARY = re.compile(r"^stress-viz summary: (\{.*\})$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [videos] {msg}", flush=True)


def mid_time(event: Path, frame_dt: float) -> float:
    """A mid-impact simulated time: three frames after the oracle's first break (else the
    middle of the event)."""
    frames = json.loads((event / "frames.json").read_text())
    for fr in frames:
        if fr.get("breaks", 0) > 0:
            return (fr["frame"] + 3) * frame_dt
    return 0.5 * len(frames) * frame_dt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stress-viz", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--python-dir", type=Path, required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--threshold", default="break at margin q >= 1.0")
    p.add_argument("--p1-runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", default="")
    p.add_argument("--max-seconds", type=float, default=2.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--index-only", action="store_true", help="rewrite index.md from results.json")
    p.add_argument("--notes", type=Path, help="markdown appended to index.md (a reading of the results)")
    a = p.parse_args()
    if a.index_only:
        write_index(a, json.loads((a.out / "results.json").read_text())["results"])
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    env = {**os.environ, "STRESS_DEBRIS_ROUTING": "1"}
    results = []
    todo = [s for s in SCENARIOS if not only or s[0] in only]
    for k, (name, rel, what) in enumerate(todo):
        event = a.p1_runs / rel
        scene = json.loads((event / "scene.json").read_text())
        outcome = json.loads((event / "outcome.json").read_text())
        context = json.loads((event / "context.json").read_text())
        dt = scene["sim"]["frame_dt"]
        seconds = min(outcome["frames"] * dt, a.max_seconds)
        labels = {"oracle": "oracle (reference)", "model": f"learned model, {a.model_label}", "qs": "quasi-static (blast_parity)"}
        mp4 = a.out / f"{name}.mp4"
        cmd = [str(a.stress_viz), "compare", str(event / "scene.json"),
               "--variant", f"{labels['oracle']}:engine standalone",
               "--variant", f"{labels['model']}:engine a_learned",
               "--variant", f"{labels['qs']}:engine qs",
               "--checkpoint", str(a.checkpoint), "--model-python", str(a.python_dir),
               "--views", "fragments", "--plot", "broken", "--fps", str(a.fps), "--size", "1920x1080", "--ss", "1",
               "--renderer", "cpu", "--duration", f"{seconds:.6f}",
               "--title", f"{name.split('_', 1)[1].replace('_', ' ')}  |  harness A  |  {a.model_label}  |  {a.threshold}",
               "-o", str(mp4)]
        log(f"{k + 1}/{len(todo)} {name}: {seconds:.3f} s simulated ({rel})")
        t = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (a.out / f"{name}.log").write_text(proc.stdout + proc.stderr)
        summaries = {}
        for line in proc.stderr.splitlines():
            m = SUMMARY.match(line.strip())
            if m:
                s = json.loads(m.group(1))
                key = next((k2 for k2, v in labels.items() if v == s["label"]), s["label"])
                summaries[key] = s
        if proc.returncode != 0:
            log(f"{name}: FAILED (exit {proc.returncode}); see {name}.log")
            results.append({"name": name, "event": rel, "what": what, "failed": proc.stderr[-2000:]})
            continue
        t_mid = mid_time(event, dt)
        png = a.out / f"{name}_mid.png"
        # stress-viz plays the 1/60 s frames at --fps: video time = simulated time * 60 / fps.
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t_mid * (1.0 / dt) / a.fps:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)], check=True)
        results.append({"name": name, "event": rel, "what": what, "split": context.get("split"), "family": context.get("family"),
                        "seconds": seconds, "t_mid": t_mid, "summaries": summaries,
                        "recorded": {"fragments": outcome.get("fragments"), "impactors": [{"regime": i.get("regime"), "speed": sum(x * x for x in i["v_end"]) ** 0.5} for i in outcome["impactors"]]},
                        "wall_s": round(time.time() - t, 1)})
        s = summaries
        log(f"{name}: done in {time.time() - t:.0f} s; broken bonds oracle {s.get('oracle', {}).get('broken_bonds')} model {s.get('model', {}).get('broken_bonds')} qs {s.get('qs', {}).get('broken_bonds')}")
        (a.out / "results.json").write_text(json.dumps({"checkpoint": str(a.checkpoint), "model_label": a.model_label, "threshold": a.threshold, "results": results}, indent=1))
    write_index(a, results)
    log(f"index: {a.out / 'index.md'}")
    return int(any("failed" in r for r in results))


def write_index(a, results) -> None:
    L = [f"# Oracle vs trained model, harness A ({time.strftime('%Y-%m-%d')})", "",
         f"Checkpoint: `{a.checkpoint}` ({a.model_label}); decode: {a.threshold}.", "",
         "Every video has three panels in step at the scene's 1/60 s frame (played 4x slow):",
         "- **oracle (reference)**: the standalone CPU stress-ref oracle, the event exactly as P1 recorded it;",
         "- **learned model**: harness A, the same oracle with every routed island's frame from the "
         "checkpoint (its decoded /3 state applied to the island; the rest of the world the oracle's own step);",
         "- **quasi-static (blast_parity)**: P1's QS baseline on the CPU, for context.",
         "",
         "Colours are fragments (each piece its own colour; grey anchored). The chart is broken bonds over time.",
         "`*_mid.png` is a frame three frames after the oracle's first break.", "",
         "| # | scenario | split | oracle pieces / broken | model pieces / broken | QS pieces / broken | impactor exit speed oracle / model (m/s) |",
         "|---|---|---|---|---|---|---|"]
    for r in results:
        if "failed" in r:
            L.append(f"| {r['name']} | FAILED | | | | | |")
            continue
        s = r["summaries"]
        cell = lambda k: f"{s.get(k, {}).get('pieces', '?')} / {s.get(k, {}).get('broken_bonds', '?')}"
        sp = lambda k: ", ".join(f"{v:.2f}" for v in s.get(k, {}).get("impactor_speeds", []))
        L.append(f"| [{r['name']}]({r['name']}.mp4) | {r['event'].split('/')[-1]} | {r['split'] or 'reference'} | {cell('oracle')} | {cell('model')} | {cell('qs')} | {sp('oracle')} / {sp('model')} |")
    # The oracle panel must be the recorded event (P1's record of the same scene).
    ok = []
    for r in results:
        if "failed" in r or "oracle" not in r["summaries"]:
            continue
        o, rec = r["summaries"]["oracle"], r["recorded"]
        same = o["pieces"] == rec["fragments"] and all(abs(x - i["speed"]) < 1e-3 for x, i in zip(o["impactor_speeds"], rec["impactors"]))
        ok.append((r["name"], same))
    L += ["", f"Check: the oracle panel reproduces P1's recorded outcome (pieces and every impactor's exit speed to 1e-3 m/s) "
          f"in {sum(x for _, x in ok)}/{len(ok)} scenarios" + ("" if all(x for _, x in ok) else f"; differs: {[n for n, x in ok if not x]}") + ".", ""]
    if getattr(a, "notes", None) and a.notes.exists():
        L += [a.notes.read_text().rstrip(), ""]
    for r in results:
        L += [f"## {r['name']}", "", r["what"], "", f"Event: `{r['event']}`."]
        if "failed" not in r:
            L += [f"Simulated {r['seconds']:.3f} s; mid-impact frame at t = {r['t_mid']:.3f} s: ![]({r['name']}_mid.png)", ""]
        else:
            L += ["Render failed; see the log.", ""]
    (a.out / "index.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
