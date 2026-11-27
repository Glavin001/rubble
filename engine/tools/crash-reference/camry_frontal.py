#!/usr/bin/env python3
"""The real 2012-2014 Toyota Camry frontal tests (NHTSA 8545, 7520; scenario S11) beside
Rubble at the same times: real stills read off the films by their annotated time base,
Rubble's stills drawn on the CPU reference, and side-by-side sheets
(engine/docs/crash-reference/camry-frontal-comparison.md).

    camry_frontal.py strip ROOT TEST VIDEO --ms 0,10,20 [--crop W:H:X:Y] [--cols N] [--out FILE]
    camry_frontal.py stills ROOT TEST VIDEO --ms 0,10,20 [--out DIR]
    camry_frontal.py scenario [--ms 0,10,...] [--car local:...] > SCENARIO.json
    camry_frontal.py cameras [--names left,right,pit,overall,overall_r] > CAMERAS.json
    camry_frontal.py sheet --rows MS,... --col "label=DIR/{ms}.png" ... --out FILE.png

ROOT: the local NHTSA folder (`inspiration/ccsa-camry-2012`, gitignored; `nhtsa_media.py`
fills it). A film's time base is its note in the committed index
(`engine/docs/crash-reference/nhtsa/media/<test>.json`, `nhtsa_footage.py`): `t0_frame`
(the file frame of first contact), `fps_film` (the camera's rate, the report's camera table)
and `file_frames_per_film_frame` (7520's files repeat each camera frame 3 times; 1 when
absent). The file frame of `ms` after contact is `t0_frame + round(ms * fps_film / 1000) *
file_frames_per_film_frame`.

- `strip`: one image, the film's frames at those times in a row (or `--cols` per row), each
  stamped with test, camera, ms and file frame; for reading the timeline.
- `stills`: one JPEG per time, `DIR/<ms>.jpg` (default `engine/out/camry-frontal/real/<test>/<video>/`).
- `scenario`: S11 exported as a data scenario (`crash_ref -- --export S11`) with its moments
  replaced by the times asked for, for `crash_outlines --scenario`. `--car` replaces the
  vehicle's design (e.g. `local:etkk_tex/etkk/kc6.pc`, with `RUBBLE_LOCAL_MODS` set).
- `cameras`: named views as `crash_outlines --cameras` takes them (`left`: the car's left,
  as the lab's Left Front Half camera; `right`: the right; `overall`; `pit`: from under
  the car, as the Pit Front camera; `angle`: as the lab's Left Angle).
- `sheet`: a grid, a row per time and a column per source, each cell the source's image for
  that time (`{ms}` in its path; missing: a grey cell saying so), labelled.

Real footage is third-party (public domain US government work, but kept out of git like
every reference image): stills and sheets go to `engine/out/` (gitignored) and
`~/Development/rubble-artifacts/crash-reference/camry-frontal/`. Needs ffmpeg.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, "..", ".."))
DOCS = os.path.join(ENGINE, "docs", "crash-reference", "nhtsa", "media")
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def run(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def film(root, test, video):
    """(path to read, note) of `video` (the original's file name, e.g. v08545C009.avi)."""
    doc = json.load(open(os.path.join(DOCS, f"{test}.json")))
    row = next((it for it in doc["items"] if os.path.basename(it["file"]) == video), None)
    if row is None:
        sys.exit(f"{video}: not in {test}.json")
    note = row.get("note") or {}
    if "t0_frame" not in note or "fps_film" not in note:
        sys.exit(f"{video}: its note has no t0_frame/fps_film yet")
    media = os.path.join(root, f"nhtsa-{test}", "media")
    path = os.path.join(media, row["file"])
    # The original is lossless enough and frame-exact; read it (the mp4 is a transcode).
    return path, note, row["name"]


def frame_of(note, ms):
    k = note.get("file_frames_per_film_frame", 1)
    return note["t0_frame"] + round(ms * note["fps_film"] / 1000.0) * k


def label(test, name, ms, n):
    cam = name.split(" - ", 1)[-1].rsplit(".", 1)[0] if " - " in name else name
    return f"{test} {cam}  {ms:+g} ms  (file frame {n})".replace(":", "\\:").replace("'", "")


def extract(path, n, out, crop=None, text=None, width=None):
    vf = [f"select='eq(n\\,{n})'"]
    if crop:
        vf.append(f"crop={crop}")
    if width:
        vf.append(f"scale={width}:-2")
    if text:
        vf.append(f"drawtext=fontfile='{FONT}':text='{text}':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.65")
    run("ffmpeg", "-v", "error", "-y", "-i", path, "-vf", ",".join(vf), "-vsync", "0", "-frames:v", "1", "-q:v", "2", out)


def ms_list(s):
    return [float(x) if "." in x else int(x) for x in s.split(",") if x]


def cmd_stills(a):
    path, note, name = film(a.root, a.test, a.video)
    out = a.out or os.path.join(ENGINE, "out", "camry-frontal", "real", a.test, os.path.splitext(a.video)[0])
    os.makedirs(out, exist_ok=True)
    for ms in ms_list(a.ms):
        n = frame_of(note, ms)
        extract(path, n, os.path.join(out, f"{ms}.jpg"), a.crop, label(a.test, name, ms, n) if a.stamp else None, a.width)
    print(out)


def cmd_strip(a):
    path, note, name = film(a.root, a.test, a.video)
    tmp = os.path.join(ENGINE, "out", "camry-frontal", "tmp-strip")
    os.makedirs(tmp, exist_ok=True)
    files = []
    for i, ms in enumerate(ms_list(a.ms)):
        n = frame_of(note, ms)
        f = os.path.join(tmp, f"{i:03d}.jpg")
        extract(path, n, f, a.crop, label(a.test, name, ms, n), a.width)
        files.append(f)
    cols = a.cols or len(files)
    out = a.out or os.path.join(ENGINE, "out", "camry-frontal", "strips", f"{a.test}-{os.path.splitext(a.video)[0]}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    grid(files, cols, out)
    print(out)


def grid(files, cols, out, pad_w=None, pad_h=None):
    """Tile `files` (same size, else padded to the first's) `cols` a row into `out`."""
    w, h = pad_w, pad_h
    if w is None:
        j = json.loads(run("ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", files[0]))
        w, h = j["streams"][0]["width"], j["streams"][0]["height"]
    args = ["ffmpeg", "-v", "error", "-y"]
    chain = []
    for i, f in enumerate(files):
        args += ["-i", f]
        chain.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x202020[p{i}]")
    layout = "|".join(f"{(i % cols) * w}_{(i // cols) * h}" for i in range(len(files)))
    if len(files) > 1:
        chain.append("".join(f"[p{i}]" for i in range(len(files))) + f"xstack=inputs={len(files)}:layout={layout}:fill=0x202020[o]")
    else:
        chain.append("[p0]copy[o]")
    run(*args, "-filter_complex", ";".join(chain), "-map", "[o]", "-frames:v", "1", "-q:v", "3", out)


# Views for Rubble's stills, in S11's world frame (`crash_ref -- --export S11`: the car
# drives along +Z into the wall box whose face is near z = 2.53, +Y up; seen from +X the
# car moves to the left of the image, as in the lab's left-side films). Explicit cameras
# for `crash_outlines --cameras FILE` (`outline::FreeCamera`).
VIEWS = {
    # The lab's Left Front Half (camera 3): from the car's left, level with the hood.
    "left": {"eye": [10.0, 1.0, 1.0], "target": [0.0, 0.75, 1.0], "fov_y_deg": 22.0},
    # Camera 9, Right Front Half.
    "right": {"eye": [-10.0, 1.0, 1.0], "target": [0.0, 0.75, 1.0], "fov_y_deg": 22.0},
    # The whole car from the left (cameras 1 and 7).
    "overall": {"eye": [12.0, 1.0, -0.5], "target": [0.0, 0.75, -0.5], "fov_y_deg": 28.0},
    # The whole car from the right (camera 7, Right Overall): pitch and rear lift.
    "overall_r": {"eye": [-12.0, 1.0, -0.5], "target": [0.0, 0.75, -0.5], "fov_y_deg": 28.0},
    # The pit (camera 14, Pit Front): from under the floor looking up at the front
    # underbody, the wall at the left of the image as in the lab's film. Rubble draws its
    # collision skin, so from below it shows the floor and the front's outline, not the
    # engine, subframe or rails the film shows.
    "pit": {"eye": [0.0, -2.2, 1.4], "target": [0.0, 0.3, 1.4], "up": [1.0, 0.0, 0.0], "fov_y_deg": 60.0},
    # The lab's Left Angle (camera 4): from the left and behind, raised.
    "angle": {"eye": [4.0, 2.4, -2.5], "target": [0.0, 0.6, 1.2], "fov_y_deg": 40.0},
}


def cmd_scenario(a):
    exe = os.path.join(ENGINE, "target", "release", "examples", "crash_ref")
    desc = json.loads(run(exe, "--export", "S11", cwd=ENGINE))
    if a.id:
        desc["id"] = a.id
    ms = ms_list(a.ms)
    desc["moments"] = [{"name": f"{m}ms", "after_impact_ms": float(m)} for m in ms]
    desc["secs_after_impact"] = max(desc.get("secs_after_impact", 0.4), max(ms) / 1000.0 + 0.01)
    if a.car:
        for b in desc["bodies"]:
            b["design"] = a.car
    json.dump(desc, sys.stdout, indent=1)


def cmd_cameras(a):
    out = []
    for c in a.names.split(","):
        v = dict(VIEWS[c])
        v.setdefault("up", [0.0, 1.0, 0.0])
        out.append(dict(name=c, width=a.size[0], height=a.size[1], **v))
    json.dump(out, sys.stdout, indent=1)


def cmd_shape(a):
    """Per moment, Rubble's shape numbers from `crash_outlines`' outlines.json (car frame:
    x along the car from the rear axle, y up; the ghost is the rest shape carried with
    the cabin and rear): rear pitch, front hub setback, hood rise and chord, front setback."""
    print("| variant | ms | pitch (deg, + nose down) | front hub setback L/R (mm) | hood max rise (mm) at row | hood chord change (mm) | front end setback (mm) |")
    print("|---|---|---|---|---|---|---|")
    for spec in a.dirs:
        lab, d = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        doc = json.load(open(os.path.join(d, "outlines.json")))
        for m in doc["moments"]:
            b = m["bodies"][0]
            cf = b["car_frame"]
            hubs = {w["name"]: (w["ghost"][0] - w["now"][0]) * 1000 for w in cf["wheels"]}
            h = cf["hood_centreline"]
            rises = [(n[1] - g[1]) * 1000 for n, g in zip(h["now"], h["ghost"])]
            k = max(range(len(rises)), key=lambda i: rises[i])
            chord = lambda p: ((p[0][0] - p[-1][0]) ** 2 + (p[0][1] - p[-1][1]) ** 2) ** 0.5
            front = {}
            for key in ("now", "ghost"):
                xs = [pt[0] for part in cf["parts"].values() for ln in part[key].get("lines", []) for pt in ln]
                front[key] = max(xs) if xs else float("nan")
            print(f"| {lab} | {m['after_impact_ms']:g} | {b['rear_pitch_deg']:.2f} | {hubs.get('wheel_fl', float('nan')):.0f} / {hubs.get('wheel_fr', float('nan')):.0f} | {rises[k]:.0f} at {k + 1} of {len(rises)} | {(chord(h['now']) - chord(h['ghost'])) * 1000:.0f} | {(front['ghost'] - front['now']) * 1000:.0f} |")


# The Reference Studio's shots of the real tests: per test, the films and times whose stills
# are frames (S11; ms after first contact). Points only in the committed workspace; images
# into its gitignored library.
STUDIO = {
    "8545": {
        "title": "S11 real: 2014 Toyota Camry, NHTSA 8545 (NCAP frontal, rigid wall, 56.7 km/h)",
        "films": {
            "v08545C009.avi": ("left-front-half", "side", [0, 20, 30, 40, 55, 70, 100, 121, 290]),
            "v08545C015.avi": ("right-front-half", "side", [30, 55, 121]),
            "v08545C013.avi": ("right-overall", "side", [0, 121]),
            "v08545C006.avi": ("pit-front", "below", [0, 30, 55]),
            "v08545C003.avi": ("windshield-overhead", "top", [0, 40]),
        },
    },
    "7520": {
        "title": "S11 real: 2012 Toyota Camry, NHTSA 7520 (NCAP frontal, rigid wall, 56.3 km/h)",
        "films": {
            "v07520C008.avi": ("left-front-half", "side", [0, 30, 55, 121, 290]),
            "v07520C014.avi": ("right-front-half", "side", [30, 55]),
        },
    },
}
LABELS = {0: "first contact", 20: "hood starts to buckle", 30: "first force peak", 40: "hood tent", 55: "second force peak, front wheel stopped", 70: "maximum crush", 100: "rebound begins", 121: "separation (lab)", 290: "end of film"}


def cmd_studio(a):
    import hashlib
    from datetime import datetime, timezone
    repo = os.path.dirname(ENGINE)
    ws = os.path.join(repo, "apps", "reference-studio", "workspace")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    for test, spec in STUDIO.items():
        shot = f"S11_nhtsa{test}"
        doc = json.load(open(os.path.join(DOCS, f"{test}.json")))
        json.dump({
            "id": shot, "title": spec["title"],
            "source": {"kind": "screenshots", "note": f"Stills read off NHTSA test {test}'s high-speed films (public domain US government work; never committed) by camry_frontal.py studio, at exact frames: ms after first contact from each film's note in engine/docs/crash-reference/nhtsa/media/{test}.json. Film URLs there."},
            "rubble": {"scenario": "S11", "contact": "edge", "run": 0},
            "before_frame": f"{shot}_left-front-half_0ms" if "left-front-half" in str(spec) else None,
            "notes": "Frames' times are exact to the film's frame (1 ms); see engine/docs/crash-reference/camry-frontal-comparison.md.",
            "updated": now,
        }, open(os.path.join(ws, "shots", f"{shot}.json"), "w"), indent=1)
        lib = os.path.join(ws, "library", shot)
        os.makedirs(lib, exist_ok=True)
        for video, (tag, view, times) in spec["films"].items():
            path, note, name = film(a.root, test, video)
            row = next(it for it in doc["items"] if os.path.basename(it["file"]) == video)
            fps = row.get("probe", {}).get("playback_fps") or 25.0
            for ms in times:
                n = frame_of(note, ms)
                fid = f"{shot}_{tag}_{ms}ms"
                img = os.path.join(lib, f"{fid}.jpg")
                extract(path, n, img)
                j = json.loads(run("ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", img))["streams"][0]
                sha = hashlib.sha256(open(img, "rb").read()).hexdigest()
                json.dump({
                    "id": fid, "shot": shot,
                    "image": {"file": f"{shot}/{fid}.jpg", "width": j["width"], "height": j["height"], "sha256": sha},
                    "moment": {"label": LABELS.get(ms, f"{ms} ms"), "after_impact_ms": float(ms)},
                    "time": {"video_s": round(n / fps, 4), "approximate": False},
                    "view": view,
                    "notes": f"{name} ({video}), file frame {n}; {ms:+g} ms after first contact. Camera: {note.get('view', '')}.",
                    "rubble": {"moment": None, "camera": None},
                    "camera": None, "regions": [], "updated": now,
                }, open(os.path.join(ws, "frames", f"{fid}.json"), "w"), indent=1)
                print(fid)


def cmd_sheet(a):
    rows = ms_list(a.rows)
    cols = [c.split("=", 1) for c in a.col]
    w, h = a.cell
    tmp = os.path.join(ENGINE, "out", "camry-frontal", "tmp-sheet")
    os.makedirs(tmp, exist_ok=True)
    files = []
    # Header row: the column labels.
    for j, (lab, _) in enumerate(cols):
        f = os.path.join(tmp, f"h{j}.png")
        t = lab.replace(":", "\\:").replace("'", "")
        run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x101010:s={w}x60", "-vf", f"drawtext=fontfile='{FONT}':text='{t}':x=(w-tw)/2:y=16:fontsize=28:fontcolor=white", "-frames:v", "1", f)
        files.append(f)
    for i, ms in enumerate(rows):
        for j, (lab, pat) in enumerate(cols):
            src = pat.replace("{ms}", str(ms))
            f = os.path.join(tmp, f"c{i}_{j}.png")
            text = f"{ms:+g} ms".replace(":", "\\:")
            if os.path.exists(src):
                run("ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x202020,drawtext=fontfile='{FONT}':text='{text}':x=w-tw-8:y=h-th-8:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6", "-frames:v", "1", f)
            else:
                run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x303030:s={w}x{h}", "-vf", f"drawtext=fontfile='{FONT}':text='{text} (none)':x=10:y=10:fontsize=24:fontcolor=white", "-frames:v", "1", f)
            files.append(f)
    # Header cells are shorter: tile the header and the body apart, then stack.
    head = os.path.join(tmp, "head.png")
    grid(files[:len(cols)], len(cols), head, w, 60)
    body = os.path.join(tmp, "body.png")
    grid(files[len(cols):], len(cols), body, w, h)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    run("ffmpeg", "-v", "error", "-y", "-i", head, "-i", body, "-filter_complex", "[0][1]vstack", "-q:v", "3", a.out)
    print(a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("strip", "stills"):
        p = sub.add_parser(name)
        p.add_argument("root")
        p.add_argument("test")
        p.add_argument("video")
        p.add_argument("--ms", required=True)
        p.add_argument("--crop")
        p.add_argument("--width", type=int)
        p.add_argument("--out")
        if name == "strip":
            p.add_argument("--cols", type=int)
        else:
            p.add_argument("--stamp", action="store_true")
    p = sub.add_parser("scenario")
    p.add_argument("--ms", default="0,10,20,30,40,55,70,100,150,300")
    p.add_argument("--car")
    p.add_argument("--id")
    p = sub.add_parser("cameras")
    p.add_argument("--names", default="left,right,pit,overall,overall_r")
    p.add_argument("--size", nargs=2, type=int, default=[1024, 720])
    p = sub.add_parser("studio")
    p.add_argument("root")
    p = sub.add_parser("shape")
    p.add_argument("dirs", nargs="+", help="LABEL=DIR holding crash_outlines' outlines.json")
    p = sub.add_parser("sheet")
    p.add_argument("--rows", required=True)
    p.add_argument("--col", action="append", required=True, help="LABEL=PATH with {ms}")
    p.add_argument("--cell", nargs=2, type=int, default=[640, 400])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"strip": cmd_strip, "stills": cmd_stills, "scenario": cmd_scenario, "cameras": cmd_cameras, "shape": cmd_shape, "studio": cmd_studio, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
