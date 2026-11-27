#!/usr/bin/env python3
"""The 2012-14 Camry side tests (NHTSA 7517, 8560: MDB; 7523, 8558: pole) beside Rubble's
S14 and S13: real stills at chosen times, side-by-side sheets, lateral curves, and the photo
grouping of the committed media index (engine/docs/crash-reference/camry-side-comparison.md).

    side_compare.py real ROOT TEST VIDEO MS[,MS...] [--out DIR]
    side_compare.py sheet SPEC.json [--out FILE.png] [--copy-to DIR]
    side_compare.py curves --test docs/crash-reference/nhtsa/7517.json ... \
        --run "label=out/x/S14-discrete.json" ... --out FILE.png [--title T]
    side_compare.py photos TEST [TEST ...]
    side_compare.py rubble-cams ID EXPORT.json --out SCENARIO.json [--ms MS,...]

`real`: the stills of VIDEO (its file name in the test's `media/videos/`) at each MS after
first contact, to `DIR/<test>/<video stem>/<ms>.jpg` (default DIR
`engine/out/camry-side/real`). The time is the film's own, read from the note the committed
index (`docs/crash-reference/nhtsa/media/<test>.json`) holds for the video: `fps_film`, the
burned-in frame number of first contact `t0_frame`, and `file_frames_per_film_frame` (the
2012 tests' transcodes repeat each film frame three times; the 2014 ones once). The film frame
of MS is `t0_frame + MS * fps_film / 1000`; its index in the file is
`(film_frame - 1) * file_frames_per_film_frame` (the burned-in "Frame: 1" is the file's first
frame). Every still keeps the lab's burned-in caption (time, rate, frame), so a wrong mapping
shows on the picture.

`sheet`: a grid, one row per time and one column per source, from SPEC: {"title": T,
"times_ms": [...], "columns": [{"label": L, "path": "pattern with {ms}"}], "cell": [W, H]}.
A missing image is drawn as an empty cell with "no frame". Labels are burned on each cell.
Third-party footage: the sheets are not committed (`--copy-to` the artifacts folder).

`curves`: the struck car's CG lateral velocity, displacement and acceleration (+Y right,
the tests' SAE axes; Rubble's `car` body's `series.com` in the same axes: its `vel_y`/`disp_y`
over `t_ms`) and, for MDB tests, the barrier's velocity along its axis. NHTSA curves are the
committed CFC 60 ones (`nhtsa_reference.py`). Needs matplotlib.

`photos`: writes each photo's `note` in the committed index from its name (NHTSA's caption):
`group` (exterior pre/post by side, impact zone, door latches, interior/door panels, dummies,
barrier face, underbody/rollover, labels, other), `phase` (pre/post) and `structure: true` for
the views that show the struck structure's final shape. A note written by hand (`by: "hand"`)
is kept as it is.

`rubble-cams`: a copy of scenario ID's export (`crash_ref -- --export ID`) with its moments
replaced by MS (default the sheets' times) and, with `--car`, the struck car's design
replaced, plus `<out>.cameras.json`: the real films' camera angles (overhead with the film's
orientation, along the side from behind and ahead) centred on `--at X Z` (the impact point in
the scenario's world frame), for `crash_outlines -- --scenario FILE --cameras CAMS` to draw at
the films' times and views. The layout, and so the physics, is the export's own: only the
moments (where the run stops to draw) change.

Standard library and ffmpeg (matplotlib for `curves`).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, "..", ".."))
MEDIA_DOCS = os.path.join(ENGINE, "docs", "crash-reference", "nhtsa", "media")
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
TIMES = [-10, 0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300]


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def index(test):
    return json.load(open(os.path.join(MEDIA_DOCS, f"{test}.json")))


def video_note(test, video):
    for it in index(test)["items"]:
        if it["kind"] == "video" and os.path.basename(it["file"]) == video:
            return it, it.get("note") or {}
    raise SystemExit(f"{test}: no video {video} in the index")


def best_file(root, test, video):
    media = os.path.join(root, f"nhtsa-{test}", "media", "videos")
    stem = os.path.splitext(video)[0]
    mp4 = os.path.join(media, stem + ".mp4")
    return mp4 if os.path.exists(mp4) else os.path.join(media, video)


# ------------------------------------------------------------------ real stills
def cmd_real(a):
    it, note = video_note(a.test, a.video)
    for k in ("fps_film", "t0_frame", "file_frames_per_film_frame"):
        if k not in note:
            raise SystemExit(f"{a.test} {a.video}: the note has no {k}; annotate the index first")
    src = best_file(a.root, a.test, a.video)
    n_file = (it.get("probe") or {}).get("frames", 10 ** 9)
    out = os.path.join(a.out, str(a.test), os.path.splitext(a.video)[0])
    os.makedirs(out, exist_ok=True)
    for ms in [float(x) for x in a.ms.split(",")]:
        film = note["t0_frame"] + ms * note["fps_film"] / 1000.0
        idx = int(round((film - 1) * note["file_frames_per_film_frame"]))
        if idx < 0 or idx >= n_file:
            print(f"{a.video} {ms:g} ms: film frame {film:g} is outside the film")
            continue
        dst = os.path.join(out, f"{ms:g}.jpg")
        run("ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"select='eq(n\\,{idx})'", "-vsync", "0", "-frames:v", "1", "-q:v", "2", dst)
        print(dst)


# ------------------------------------------------------------------ sheets
def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace(",", "\\,").replace("%", "\\%")


def cmd_sheet(a):
    spec = json.load(open(a.spec))
    w, h = spec.get("cell", [480, 320])
    cols = spec["columns"]
    rows = []
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "_cells")
    os.makedirs(tmp, exist_ok=True)
    # Header row: the column labels.
    head = os.path.join(tmp, "head.png")
    parts = []
    for i, c in enumerate(cols):
        parts.append(f"color=c=0x202020:s={w}x40,drawtext=fontfile='{FONT}':text='{esc(c['label'])}':x=8:y=8:fontsize=20:fontcolor=white[h{i}]")
    lay = "".join(f"[h{i}]" for i in range(len(cols)))
    run("ffmpeg", "-v", "error", "-y", "-filter_complex", ";".join(parts) + f";{lay}hstack=inputs={len(cols)}[o]" if len(cols) > 1 else parts[0][:-4], "-map", "[o]" if len(cols) > 1 else "0", "-frames:v", "1", head)
    rows.append(head)
    for ms in spec["times_ms"]:
        args, chain = ["ffmpeg", "-v", "error", "-y"], []
        for i, c in enumerate(cols):
            p = c["path"].format(ms=f"{ms:g}", ms_int=int(round(ms)))
            p = os.path.expanduser(p)
            label = f"{ms:g} ms"
            if os.path.exists(p):
                args += ["-i", p]
                chain.append(f"[{len(args) // 2 - 3}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x101010,drawtext=fontfile='{FONT}':text='{esc(label)}':x=w-tw-8:y=h-th-8:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6[c{i}]")
            else:
                chain.append(f"color=c=0x101010:s={w}x{h},drawtext=fontfile='{FONT}':text='{esc(label)} no frame':x=8:y=8:fontsize=20:fontcolor=gray[c{i}]")
        # Inputs are numbered in order of appearance; recompute the labels.
        k = 0
        fixed = []
        for s in chain:
            if s.startswith("["):
                s = re.sub(r"^\[\d+:v\]", f"[{k}:v]", s)
                k += 1
            fixed.append(s)
        dst = os.path.join(tmp, f"row-{ms:g}.png")
        lay = "".join(f"[c{i}]" for i in range(len(cols)))
        run(*args, "-filter_complex", ";".join(fixed) + f";{lay}hstack=inputs={len(cols)}[o]", "-map", "[o]", "-frames:v", "1", dst)
        rows.append(dst)
    args = ["ffmpeg", "-v", "error", "-y"]
    for r in rows:
        args += ["-i", r]
    title = os.path.join(tmp, "title.png")
    run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={w * len(cols)}x44", "-vf", f"drawtext=fontfile='{FONT}':text='{esc(spec.get('title', ''))}':x=10:y=10:fontsize=24:fontcolor=white", "-frames:v", "1", title)
    args = ["ffmpeg", "-v", "error", "-y", "-i", title] + args[4:]
    run(*args, "-filter_complex", "".join(f"[{i}:v]" for i in range(len(rows) + 1)) + f"vstack=inputs={len(rows) + 1}[o]", "-map", "[o]", "-frames:v", "1", a.out)
    print(a.out)
    if a.copy_to:
        d = os.path.expanduser(a.copy_to)
        os.makedirs(d, exist_ok=True)
        shutil.copy(a.out, d)


# ------------------------------------------------------------------ curves
def rubble_series(path, body="car"):
    text = open(path).read().strip()
    try:
        rep = json.loads(text)
    except json.JSONDecodeError:
        rep = json.loads(text.splitlines()[0])
    s = rep["series"]
    for b in s["bodies"]:
        if b["name"] == body:
            return s["t_ms"], b
    raise SystemExit(f"{path}: no body {body} in the series")


# Struck-car structure channels read per test (the EV5 curves' attachment), +Y right.
STRUCTURE_CH = ["VEHICLE CG", "SILL - LEFT FRONT", "SILL - LEFT REAR", "LEFT MID A-POST", "LEFT A-POST @ SILL", "LEFT LOWER A-POST",
                "LEFT B-POST @ SILL", "LEFT LOWER B-POST", "LEFT MID B-POST", "SILL - RIGHT FRONT", "SILL - RIGHT REAR", "RIGHT SILL", "RIGHT ROOF",
                "FIREWALL", "ENGINE", "TRUNK FLOOR - CENTER", "FLOORPAN - RIGHT REAR"]


def cmd_sensors(a):
    """Every usable struck-car structure channel (Y; X for the CG, engine and trunk floor)
    through CFC 60 and integrated from contact (`nhtsa_reference.py`'s processing), the MDB's
    CG X, and the yaw from the engine's and the trunk floor's lateral velocities over their
    distance along the car (`--lever`, from the report's accelerometer table). Writes
    DIR/<test>.json (1 kHz, -10..300 ms) and prints a table."""
    sys.path.insert(0, HERE)
    import nhtsa_reference as nr
    d = os.path.join(a.root, f"nhtsa-{a.test}")
    ch = nr.channels(d)
    pole = "POLE" in ch["title"]
    car = 1 if pole else 2
    grid = [i / 1000.0 for i in range(-10, 301)]
    out = {"test": int(a.test), "pole": pole, "t_ms": [round(1000 * t) for t in grid], "channels": {}}
    rows = []
    for c in ch["tsv_channels"]:
        att = (c.get("sensor_attachment") or "").strip()
        ax = c["axis"][0]
        if not nr.usable(c):
            continue
        if c.get("vehicle_no") == 1 and not pole:
            if not (att == "VEHICLE CG" and ax == "X"):
                continue
            name = "MDB CG X"
        else:
            if c.get("vehicle_no") != car or att not in STRUCTURE_CH or ax not in "XY":
                continue
            if ax == "X" and att not in ("VEHICLE CG", "ENGINE", "TRUNK FLOOR - CENTER"):
                continue
            name = f"{att} {ax}"
        t, acc = nr.accel(d, c)
        v, s = nr.integrate_from_zero(t, acc, 0.0)
        cur = {"file": c["file"].split("/")[-1], "a_g": [round(nr.at(t, acc, x) / nr.G, 3) for x in grid],
               "dv_mps": [round(nr.at(t, v, x), 4) for x in grid], "ds_m": [round(nr.at(t, s, x), 4) for x in grid]}
        after = [i for i in range(len(t)) if 0 <= t[i] <= 0.3]
        ia = max(after, key=lambda i: abs(acc[i]))
        iv = max(after, key=lambda i: abs(v[i]))
        half = next((t[i] for i in after if abs(v[i]) >= 0.5 * abs(v[iv])), None)
        cur.update(peak_g=round(acc[ia] / nr.G, 2), peak_ms=round(1000 * t[ia], 1), dv_peak=round(v[iv], 3), dv_peak_ms=round(1000 * t[iv], 1),
                   half_dv_ms=round(1000 * half, 1) if half is not None else None)
        out["channels"][name] = cur
        rows.append((name, cur["peak_g"], cur["peak_ms"], *[round(nr.at(t, v, x), 2) for x in (0.01, 0.02, 0.05, 0.1, 0.15)], cur["dv_peak"], cur["dv_peak_ms"], cur["half_dv_ms"]))
    eng, trk = out["channels"].get("ENGINE Y"), out["channels"].get("TRUNK FLOOR - CENTER Y")
    if eng and trk and a.lever:
        # + nose to the right (SAE yaw), from contact.
        w = [(e - r) / a.lever for e, r in zip(eng["dv_mps"], trk["dv_mps"])]
        ang, yaw = 0.0, []
        for k in range(len(grid)):
            if grid[k] > 0:
                ang += 0.001 * 0.5 * (w[k] + w[k - 1])
            yaw.append(round(math.degrees(ang), 3))
        out["yaw_deg"] = yaw
        rows.append(("yaw, engine against trunk floor (deg)", None, None, *[yaw[x + 10] for x in (10, 20, 50, 100, 150)], yaw[-1], 300, None))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"{a.test}.json"), "w"))
    print(f"| {a.test}: channel | peak (g) | at (ms) | dv 10 ms | dv 20 | dv 50 | dv 100 | dv 150 | largest dv (m/s) | at (ms) | half of it by (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join("-" if x is None else str(x) for x in r) + " |")


def rubble_fine(path, body="car"):
    rep = json.loads(open(path).read().strip().splitlines()[0])
    f = rep["series"]["fine"]
    for b in f["bodies"]:
        if b["name"] == body:
            return f["t_ms"], b["com"], rep
    raise SystemExit(f"{path}: no body {body}")


def cmd_curves(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = ax.ravel()
    for p in a.sensors:
        r = json.load(open(p))
        t = r["t_ms"]
        cg = r["channels"]["VEHICLE CG Y"]
        lab = f"NHTSA {r['test']}"
        ax[0].plot(t, cg["dv_mps"], lw=2, label=lab)
        ax[1].plot(t, cg["ds_m"], lw=2, label=lab)
        ax[2].plot(t, cg["a_g"], lw=1.2, label=lab + " (CFC 60)")
        if "MDB CG X" in r["channels"]:
            ax[3].plot(t, r["channels"]["MDB CG X"]["dv_mps"], lw=2, label=lab + " MDB dv")
        if "yaw_deg" in r:
            ax[3].plot(t, r["yaw_deg"], ":", lw=2, label=lab + " yaw (deg)")
    for spec in a.run:
        lab, path = spec.split("=", 1)
        t, com, rep = rubble_fine(path)
        v0 = com["vy"][0]
        dv = [x - v0 for x in com["vy"]]
        ax[0].plot(t, dv, "--", lw=1.6, label=lab)
        s, acc = [0.0], 0.0
        for k in range(1, len(t)):
            acc += 0.001 * (t[k] - t[k - 1]) * 0.5 * (dv[k] + dv[k - 1])
            s.append(acc)
        ax[1].plot(t, s, "--", lw=1.6, label=lab)
        ax[2].plot(t, com["ay_g"], "--", lw=1.0, label=lab + " (CFC 60, 2 kHz trace)")
        if a.barrier:
            tb, bc, _ = rubble_fine(path, a.barrier)
            ax[3].plot(tb, [x - bc["vx"][0] for x in bc["vx"]], "--", lw=1.6, label=lab + " barrier dv")
    for spec in a.yaw:
        # Rubble's yaw at the stills' moments: crash_outlines' `rear_yaw_deg` of the car
        # (its rear's turn, + toward its left), drawn in SAE sign (+ nose right).
        lab, path = spec.split("=", 1)
        o = json.load(open(path))
        pts = [(m["after_impact_ms"], -b["rear_yaw_deg"]) for m in o["moments"] for b in m["bodies"] if b["name"] == "car"]
        y0 = next((y for t, y in pts if t <= 0), 0.0)
        ax[3].plot([t for t, _ in pts], [y - y0 for _, y in pts], "-.o", lw=1.4, ms=3, label=lab + " yaw (deg)")
    ax[0].set_ylabel("CG lateral velocity change (m/s, +right)")
    ax[1].set_ylabel("CG lateral displacement from contact (m)")
    ax[2].set_ylabel("CG lateral acceleration (g)")
    ax[3].set_ylabel("MDB dv along its axis (m/s); yaw (deg)" if a.barrier else "yaw (deg)")
    for x in ax:
        x.set_xlabel("ms after first contact")
        x.set_xlim(-10, a.until)
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(a.out)


# ------------------------------------------------------------------ photos
GROUPS = [
    ("impact zone", r"impact point|impact zone|positioned against|pole barrier|impact event|for web"),
    ("door latch", r"latch"),
    ("dummy", r"dummy|seat back|seat pan|seat track|head restraint|belt anchorage|steering wheel|parking brake|chalking"),
    ("interior door panel", r"inner door panel"),
    ("barrier face", r"impactor face"),
    ("rollover / fuel", r"rollover|fuel filler"),
    ("labels / ballast", r"label|placard|monroney|owner manual|ballast|speed trap"),
    ("exterior", r"view of test vehicle|overhead view of test area|as delivered|view of impact zone"),
]
STRUCTURE = r"post-test .*(left side view of test vehicle|left front 3-4|left rear 3-4|overhead view|frontal view of test vehicle|rear view of test vehicle|impact point target|inner door panel|pole barrier)|post-test (driver|rear passenger)? ?dummy and door clearance|shattered vehicle inner door panel"


def cmd_photos(a):
    for t in a.tests:
        path = os.path.join(MEDIA_DOCS, f"{t}.json")
        doc = json.load(open(path))
        for it in doc["items"]:
            if it["kind"] != "photo":
                continue
            if (it.get("note") or {}).get("by") == "hand":
                continue
            name = it["name"].lower()
            group = next((g for g, rx in GROUPS if re.search(rx, name)), "other")
            phase = "post" if re.search(r"post-test", name) else ("pre" if re.search(r"pre-test|as delivered", name) else None)
            side = next((s for s in ("left front", "left rear", "left side", "right front", "right side", "right rear", "frontal", "rear view", "overhead") if s in name), None)
            note = {"group": group, "by": "name"}
            if phase:
                note["phase"] = phase
            if side and group == "exterior":
                note["view"] = side
            if re.search(STRUCTURE, name):
                note["structure"] = True
            it["note"] = {**note, **{k: v for k, v in (it.get("note") or {}).items() if k not in note and k != "by"}}
        json.dump(doc, open(path, "w"), indent=1)
        print(f"{t}: photo notes written")


# ------------------------------------------------------------------ Rubble cameras
def cmd_rubble_cams(a):
    sc = json.load(open(a.export))
    layout = sc["layout"] if "layout" in sc else None
    cams = []
    ms = [float(x) for x in a.ms.split(",")] if a.ms else TIMES
    if a.id == "S14":
        crab = math.radians(27.0)
        d = (-math.cos(crab), 0.0, -math.sin(crab))  # the barrier's travel
        # Overhead, image up along the barrier's travel (the films' C001/C002 orientation),
        # centred on the impact point (target given by --at).
        x, z = a.at
        cams.append({"name": "film overhead", "eye": [x, 9.0, z], "target": [x, 0.0, z], "up": list(d), "fov_y_deg": 28.0})
        # The report's camera table (data sheet 5: from the impact point, +X the MDB's
        # X read as out from the struck side, as the films show, +Y its right, which is the
        # car's rear): C005 0.06 m out and 4.85 m toward the car's rear, 1.13 m up, looking
        # along the car; C006 4.11 m out and 4.24 m toward the car's front, 1.11 m up. Aimed
        # by eye at the car to frame the films' views.
        cams.append({"name": "film rear", "eye": [x + 0.06, 1.13, z - 4.85], "target": [x - 0.9, 0.7, z - 2.6], "fov_y_deg": 45.0})
        cams.append({"name": "film front oblique", "eye": [x + 4.11, 1.11, z + 4.24], "target": [x - 0.9, 0.7, z], "fov_y_deg": 40.0})
    else:
        a75 = math.radians(75.0)
        travel = (math.sin(a75), 0.0, math.cos(a75))
        x, z = a.at
        cams.append({"name": "film overhead", "eye": [x, 7.0, z], "target": [x, 0.0, z], "up": [-travel[0], 0.0, -travel[2]], "fov_y_deg": 30.0})
        # Ground level along the car from its front and its rear (C002 / C006 in 2012
        # numbering), 1.8 m up, 6.1-6.9 m off along the car's axis, looking at the pole.
        cams.append({"name": "film front", "eye": [x - 0.4, 1.83, z + 6.1], "target": [x - 0.6, 0.7, z], "fov_y_deg": 35.0})
        cams.append({"name": "film rear", "eye": [x - 0.4, 1.8, z - 6.9], "target": [x - 0.6, 0.7, z], "fov_y_deg": 35.0})
    sc["moments"] = [{"name": f"t{m:g}", "after_impact_ms": m} for m in ms]
    if a.car:
        for b in sc["bodies"]:
            if b["name"] == "car":
                b["design"] = a.car
    json.dump(sc, open(a.out, "w"), indent=1)
    cam_file = os.path.splitext(a.out)[0] + ".cameras.json"
    json.dump(cams, open(cam_file, "w"), indent=1)
    print(a.out, cam_file)


# ------------------------------------------------------------------ hand notes
def cmd_notes(a):
    """Merge `camry-side-notes.json` (hand annotations) into the committed media index: each
    video's note (the film timing from its `hs` / `rt` common block, with
    `file_frames_per_film_frame` from `hs`), the test's scenario, summary and sources, and the
    photos' hand remarks (`structure_note`; their grouping is `photos`')."""
    n = json.load(open(a.notes))
    for t, test in n["tests"].items():
        path = os.path.join(MEDIA_DOCS, f"{t}.json")
        doc = json.load(open(path))
        doc["scenario"], doc["summary"], doc["sources"] = test["scenario"], test["summary"], test["sources"]
        ph = n["photos"].get(t, {})
        for it in doc["items"]:
            base = os.path.basename(it["file"])
            if it["kind"] == "video" and base in n["videos"]:
                v = dict(n["videos"][base])
                if "hs" in v:
                    note = dict(n["common"]["hs"], file_frames_per_film_frame=v.pop("hs"))
                else:
                    v.pop("rt", None)
                    note = dict(n["common"]["rt"])
                note.update(v)
                it["note"] = note
            elif it["kind"] == "photo":
                key = os.path.splitext(base)[0][-4:]
                if key in ph:
                    it["note"] = dict(it.get("note") or {}, remark=ph[key])
                    if "placeholder" not in ph[key]:
                        it["note"]["structure"] = True
        json.dump(doc, open(path, "w"), indent=1)
        print(f"{t}: notes merged")


# ------------------------------------------------------------------ Reference Studio
STUDIO = os.path.join(ENGINE, "..", "apps", "reference-studio", "workspace")
# Per test: the scenario, the films made shots and Rubble's camera for each, and the times.
STUDIO_FILMS = {
    7517: ("S14", [("C001", "overhead", "top"), ("C005", "rear of struck side", None), ("C006", "front oblique", "front three-quarter")]),
    8560: ("S14", [("C001", "overhead", "top"), ("C005", "rear of struck side", None), ("C006", "front oblique", "front three-quarter")]),
    7523: ("S13", [("C001", "overhead", "top"), ("C002", "front ground level", None), ("C006", "rear ground level", None)]),
    8558: ("S13", [("C001", "overhead", "top"), ("C002", "front ground level", None), ("C009", "rear ground level", None)]),
}
STUDIO_TIMES = {-10: "before", 40: "peak", 80: "deepest", 300: None}


def cmd_studio(a):
    """Reference Studio shots (one per test) and frames (each film's still at -10, 40, 80 and
    300 ms) from the `real` stills: points-only JSON in workspace/{shots,frames}/ (committed),
    the images as WebP in workspace/library/<shot>/ (gitignored). Names start with the test
    number. Rubble's moment is the scenario's own where one is at that time (before, peak,
    deepest), else none; the camera is the scenario camera nearest the film's view, else none."""
    import hashlib
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    for t, (sid, films) in STUDIO_FILMS.items():
        shot = f"{t}_{sid}_camry"
        doc = index(t)
        json.dump({"id": shot, "title": f"{t}: NHTSA {json.load(open(os.path.join(ENGINE, 'docs', 'crash-reference', 'nhtsa', f'{t}.json')))['title'].title()} ({sid})",
                   "source": {"kind": "video", "url": f"https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/get-multimedia-files/{t}",
                              "note": "High-speed films at 1,000 fps; times are ms after first contact from the burned-in caption (exact to the frame)."},
                   "rubble": {"scenario": sid, "contact": "discrete" if sid == "S14" else "edge", "run": 0},
                   "before_frame": f"{t}_C001_m010", "notes": f"Camry side test {t} beside {sid}; engine/docs/crash-reference/camry-side-comparison.md.", "updated": now},
                  open(os.path.join(STUDIO, "shots", f"{shot}.json"), "w"), indent=1)
        lib = os.path.join(STUDIO, "library", shot)
        os.makedirs(lib, exist_ok=True)
        for c, view, cam in films:
            stem = f"v0{t}{c}"
            for ms, label in STUDIO_TIMES.items():
                src = os.path.join(a.real, str(t), stem, f"{ms:g}.jpg")
                if not os.path.exists(src):
                    print(f"missing {src}")
                    continue
                fid = f"{t}_{c}_{'m' if ms < 0 else 't'}{abs(ms):03d}"
                dst = os.path.join(lib, fid + ".webp")
                run("ffmpeg", "-v", "error", "-y", "-i", src, "-quality", "90", dst)
                p = json.loads(run("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", dst))["streams"][0]
                sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
                json.dump({"id": fid, "shot": shot, "image": {"file": f"{shot}/{fid}.webp", "width": p["width"], "height": p["height"], "sha256": sha},
                           "moment": {"label": label or f"{ms} ms", "after_impact_ms": ms}, "time": None,
                           "view": view, "notes": f"NHTSA {t} film {c} ({view}), {ms} ms after first contact (film frame {51 + ms} at 1,000 fps).",
                           "rubble": {"moment": label, "camera": cam}, "camera": None, "regions": [], "updated": now},
                          open(os.path.join(STUDIO, "frames", fid + ".json"), "w"), indent=1)
        print(shot)


# ------------------------------------------------------------------ crush profile
def cmd_profile(a):
    """The struck side's plan-view crush along the car, from crash_outlines' `outlines.json`:
    each left-side group's boundary in the straight-down `top` camera, now against its ghost
    (the rest shape carried rigidly with the cabin and rear), measured inward across the car,
    binned every 150 mm by distance behind the impact line (`--impact`: metres ahead of the
    rear axle; the report's crush grid is measured from the same line, + rearward). Pixels to
    metres by the ghost's length (`--length`, bumper to bumper). The boundary mixes every
    height of the panel, so this is the deepest point of each 150 mm slice at any height, not
    one level of the report's table."""
    o = json.load(open(a.outlines))
    groups = a.groups.split(",")
    out = {}
    for m in o["moments"]:
        if a.moments and m["name"] not in a.moments.split(","):
            continue
        b = next(x for x in m["bodies"] if x["name"] == "car")
        parts = b["cameras"][a.camera]["parts"]
        ends = {}
        for g in ("bumper_f", "bumper_r"):
            pts = [p for l in parts[g]["ghost"]["lines"] for p in l]
            ends[g] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        ax = (ends["bumper_f"][0] - ends["bumper_r"][0], ends["bumper_f"][1] - ends["bumper_r"][1])
        n = math.hypot(*ax)
        ux, uy = ax[0] / n, ax[1] / n
        allg = [p for g in ("bumper_f", "bumper_r") for l in parts[g]["ghost"]["lines"] for p in l]
        along = [p[0] * ux + p[1] * uy for p in allg]
        px_per_m = (max(along) - min(along)) / a.length
        # The left (struck) side's outward normal: from the car's centre toward the ghost's
        # left groups.
        cx = sum(p[0] for p in allg) / len(allg)
        cy = sum(p[1] for p in allg) / len(allg)
        lp = [p for g in groups for l in parts[g]["ghost"]["lines"] for p in l]
        nx, ny = -uy, ux
        if sum((p[0] - cx) * nx + (p[1] - cy) * ny for p in lp) < 0:
            nx, ny = -nx, -ny
        rear = min(along)
        bins = {}
        for g in groups:
            for ln, lg in zip(parts[g]["now"]["lines"], parts[g]["ghost"]["lines"]):
                for pn, pg in zip(ln, lg):
                    x_m = ((pg[0] * ux + pg[1] * uy) - rear) / px_per_m - a.rear_overhang
                    d_m = -((pn[0] - pg[0]) * nx + (pn[1] - pg[1]) * ny) / px_per_m
                    k = int(round((a.impact - x_m) / 0.15))
                    bins[k] = max(bins.get(k, -9.0), d_m)
        out[m["name"]] = {k * 150: round(v * 1000) for k, v in sorted(bins.items())}
    keys = sorted({k for v in out.values() for k in v})
    print("| behind the impact line (mm) | " + " | ".join(out) + " |")
    print("|---|" + "---|" * len(out))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(out[m].get(k, "")) for m in out) + " |")


# ------------------------------------------------------------------ sheet specs
# The films used per test (video stem suffix, view) and Rubble's matching stills.
VIEWS = {
    7517: [("C001", "overhead wide"), ("C002", "overhead close-up"), ("C004", "side overall (MDB onboard)"), ("C005", "rear of struck side"), ("C006", "front oblique")],
    8560: [("C001", "overhead wide"), ("C002", "overhead close-up"), ("C004", "side overall (MDB onboard)"), ("C005", "rear of struck side"), ("C006", "front oblique")],
    7523: [("C001", "overhead wide (cam 10)"), ("C004", "overhead close-up"), ("C002", "front ground level"), ("C006", "rear ground level (cam 8)"), ("C003", "45 deg front")],
    8558: [("C001", "overhead wide (cam 10)"), ("C004", "overhead close-up"), ("C002", "front ground level"), ("C009", "rear ground level"), ("C003", "45 deg front")],
}
# Per comparison: the real film, Rubble's camera name, the scenario and its contact settings.
PAIRS = {
    "S14": [("C001", "film overhead", "overhead"), ("C005", "film rear", "rear of struck side"), ("C006", "film front oblique", "front oblique")],
    "S13": [("C001", "film overhead", "overhead"), ("C002", "film front", "front ground level"), ("C006", "film rear", "rear ground level")],
}
REAL_OF = {"S14": (7517, 8560), "S13": (7523, 8558)}
RUBBLE = {"S14": [("gen sedan, discrete", "S14-film-discrete"), ("gen sedan, edge (creates energy)", "S14-film-edge"), ("ETK K-series, discrete", "S14-etk-film-discrete")],
          "S13": [("gen sedan, edge", "S13-film-edge"), ("ETK K-series, edge", "S13-etk-film-edge")]}


def cmd_specs(a):
    os.makedirs(a.out, exist_ok=True)
    for t, vs in VIEWS.items():
        spec = {"title": f"NHTSA {t}: its films at the same times (ms after first contact)", "times_ms": TIMES, "cell": [420, 280],
                "columns": [{"label": f"{c} {l}", "path": f"out/camry-side/real/{t}/v0{t}{c}/{{ms}}.jpg"} for c, l in vs]}
        json.dump(spec, open(os.path.join(a.out, f"real-{t}.json"), "w"), indent=1)
    for sid, pairs in PAIRS.items():
        t12, t14 = REAL_OF[sid]
        for c, cam, label in pairs:
            # 2014's camera numbering differs on the pole films: its rear ground level is C009.
            c14 = "C009" if (sid == "S13" and c == "C006") else c
            cols = [{"label": f"{t12} {c} {label}", "path": f"out/camry-side/real/{t12}/v0{t12}{c}/{{ms}}.jpg"},
                    {"label": f"{t14} {c14} {label}", "path": f"out/camry-side/real/{t14}/v0{t14}{c14}/{{ms}}.jpg"}]
            cols += [{"label": f"Rubble {lab}", "path": f"out/camry-side/rubble/{d}/t{{ms}}-{cam}.png"} for lab, d in RUBBLE[sid]]
            name = label.replace(" ", "-")
            spec = {"title": f"{sid} beside NHTSA {t12} and {t14}: {label}, ms after first contact", "times_ms": TIMES, "cell": [400, 267], "columns": cols}
            json.dump(spec, open(os.path.join(a.out, f"{sid}-{name}.json"), "w"), indent=1)
    print(a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("specs")
    p.add_argument("--out", default=os.path.join(HERE, "camry-side-sheets"))
    p = sub.add_parser("real")
    p.add_argument("root")
    p.add_argument("test")
    p.add_argument("video")
    p.add_argument("ms")
    p.add_argument("--out", default=os.path.join(ENGINE, "out", "camry-side", "real"))
    p = sub.add_parser("sheet")
    p.add_argument("spec")
    p.add_argument("--out", required=True)
    p.add_argument("--copy-to")
    p = sub.add_parser("notes")
    p.add_argument("--notes", default=os.path.join(HERE, "camry-side-notes.json"))
    p = sub.add_parser("studio")
    p.add_argument("--real", default=os.path.join(ENGINE, "out", "camry-side", "real"))
    p = sub.add_parser("profile")
    p.add_argument("outlines")
    p.add_argument("--camera", default="top")
    p.add_argument("--groups", default="fender_l,door_fl,door_rl,sill_l")
    p.add_argument("--moments")
    p.add_argument("--impact", type=float, required=True, help="the impact line, metres ahead of the rear axle")
    p.add_argument("--length", type=float, default=4.834, help="bumper to bumper (m)")
    p.add_argument("--rear-overhang", type=float, default=1.079, help="rear bumper to rear axle (m)")
    p = sub.add_parser("sensors")
    p.add_argument("root")
    p.add_argument("test")
    p.add_argument("--lever", type=float, help="engine to trunk floor accelerometer along the car (m), for the yaw")
    p.add_argument("--out", default=os.path.join(ENGINE, "out", "camry-side", "sensors"))
    p = sub.add_parser("curves")
    p.add_argument("--sensors", action="append", default=[])
    p.add_argument("--yaw", action="append", default=[], help="label=crash_outlines outlines.json")
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--barrier")
    p.add_argument("--until", type=float, default=250.0)
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("photos")
    p.add_argument("tests", nargs="+")
    p = sub.add_parser("rubble-cams")
    p.add_argument("id")
    p.add_argument("export")
    p.add_argument("--out", required=True)
    p.add_argument("--ms")
    p.add_argument("--at", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Z"))
    p.add_argument("--car", help="the struck car's design (e.g. local:etkk_tex/etkk/kc6.pc)")
    a = ap.parse_args()
    {"specs": cmd_specs, "notes": cmd_notes, "studio": cmd_studio, "profile": cmd_profile, "sensors": cmd_sensors, "real": cmd_real, "sheet": cmd_sheet, "curves": cmd_curves, "photos": cmd_photos, "rubble-cams": cmd_rubble_cams}[a.cmd](a)


if __name__ == "__main__":
    main()
