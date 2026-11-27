#!/usr/bin/env python3
"""An NHTSA test's footage, measured and indexed: each video's frames and rate, contact
sheets to read it by, and the committed index with its annotations.

    nhtsa_footage.py ROOT TEST [TEST ...] [--sheets] [--frames VIDEO A B STEP]

ROOT: the local folder `nhtsa_media.py` filled (`nhtsa-<test>/media/`). For each test:

- `media/probe.json`: every video's size, frame count, rate and duration (ffprobe; the
  transcoded `.mp4` where there is one, else the original).
- `--sheets`: `media/sheets/<video>.jpg`, 24 frames evenly through the film, each stamped
  with its frame number (to find first contact and the moments worth a still), and
  `media/sheets/photos-<k>.jpg`, the photos 30 a sheet with their file numbers.
- `--frames VIDEO A B STEP`: `media/frames/<video>/<n>.jpg` for frames A..B every STEP
  (stamped), to pin a moment exactly.
- `engine/docs/crash-reference/nhtsa/media/<test>.json` (committed; no media in it): every
  item's file, URL, the lab's name for it, bytes, and for a video its probe; plus what a
  person wrote about it, kept across runs: an item's `note` object (`view`, `fps_film`:
  the camera's rate, `t0_frame`: the frame of first contact, `moments`: [{frame, ms,
  what}], `look`: what to compare) and the test's `scenario`, `summary` and `sources`.

A film's time: `ms = (frame - t0_frame) * 1000 / fps_film`. The camera's rate is the
report's (its camera table), not the file's playback rate. Needs ffmpeg and ffprobe.
"""
import argparse
import json
import os
import subprocess

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "crash-reference", "nhtsa", "media")
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def probe(path):
    j = json.loads(run("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,duration", "-of", "json", path))
    s = j["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "frames": int(s["nb_read_packets"]), "playback_fps": round(float(n) / float(d), 3), "duration_s": round(float(s.get("duration") or 0.0), 3)}


def stamp(size=28):
    return f"drawtext=fontfile='{FONT}':text='%{{n}}':x=8:y=8:fontsize={size}:fontcolor=yellow:box=1:boxcolor=black@0.6"


def best(media, items):
    """Per original video: the file to read (its `.mp4` if here, else itself)."""
    mp4 = {it["of"]: it for it in items if it["kind"] == "video-mp4" and it["here"]}
    out = {}
    for it in items:
        if it["kind"] == "video":
            use = mp4.get(os.path.basename(it["file"]), it)
            if use["here"]:
                out[os.path.basename(it["file"])] = os.path.join(media, use["file"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("tests", nargs="+")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--frames", nargs=4, metavar=("VIDEO", "A", "B", "STEP"))
    a = ap.parse_args()
    os.makedirs(DOCS, exist_ok=True)
    for t in a.tests:
        media = os.path.join(a.root, f"nhtsa-{t}", "media")
        idx = json.load(open(os.path.join(media, "index.json")))
        items = idx["items"]
        files = best(media, items)
        if a.frames:
            name, lo, hi, step = a.frames[0], int(a.frames[1]), int(a.frames[2]), int(a.frames[3])
            out = os.path.join(media, "frames", os.path.splitext(name)[0])
            os.makedirs(out, exist_ok=True)
            sel = f"between(n\\,{lo}\\,{hi})*not(mod(n-{lo}\\,{step}))"
            run("ffmpeg", "-v", "error", "-y", "-i", files[name], "-vf", f"{stamp()},select='{sel}'", "-vsync", "0", "-start_number", "0", "-q:v", "3", os.path.join(out, "%04d.jpg"))
            # (numbered in order: file k is frame lo + k * step)
            for k, f in enumerate(sorted(os.listdir(out))):
                if f[0].isdigit() and len(f) == 8:
                    os.replace(os.path.join(out, f), os.path.join(out, f"f{lo + k * step:05d}.jpg"))
            print(out)
            continue
        probes = {}
        for name, path in sorted(files.items()):
            try:
                probes[name] = probe(path)
            except subprocess.CalledProcessError as e:
                probes[name] = {"error": e.stderr.strip()[-200:]}
        json.dump(probes, open(os.path.join(media, "probe.json"), "w"), indent=1)
        if a.sheets:
            sheets = os.path.join(media, "sheets")
            os.makedirs(sheets, exist_ok=True)
            for name, path in sorted(files.items()):
                n = probes[name].get("frames", 0)
                out = os.path.join(sheets, os.path.splitext(name)[0] + ".jpg")
                if n < 2 or os.path.exists(out):
                    continue
                step = max(1, n // 24)
                try:
                    run("ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"{stamp(40)},select='not(mod(n\\,{step}))',scale=480:-2,tile=6x4", "-frames:v", "1", "-q:v", "4", out)
                except subprocess.CalledProcessError as e:
                    print(f"{t} {name}: sheet failed: {e.stderr.strip()[-160:]}")
            photos = sorted(it["file"] for it in items if it["kind"] == "photo" and it["here"])
            for k in range(0, len(photos), 30):
                out = os.path.join(sheets, f"photos-{k // 30:02d}.jpg")
                if os.path.exists(out):
                    continue
                args = ["ffmpeg", "-v", "error", "-y"]
                chain = []
                group = photos[k:k + 30]
                for i, p in enumerate(group):
                    args += ["-i", os.path.join(media, p)]
                    label = os.path.splitext(os.path.basename(p))[0][-4:]
                    chain.append(f"[{i}:v]scale=400:300:force_original_aspect_ratio=decrease,pad=400:300:(ow-iw)/2:(oh-ih)/2,drawtext=fontfile='{FONT}':text='{label}':x=6:y=6:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6[p{i}]")
                cols = 6
                layout = "|".join(f"{(i % cols) * 400}_{(i // cols) * 300}" for i in range(len(group)))
                chain.append("".join(f"[p{i}]" for i in range(len(group))) + f"xstack=inputs={len(group)}:layout={layout}[o]" if len(group) > 1 else "[p0]copy[o]")
                try:
                    run(*args, "-filter_complex", ";".join(chain), "-map", "[o]", "-frames:v", "1", "-q:v", "4", out)
                except subprocess.CalledProcessError as e:
                    print(f"{t} photos {k}: sheet failed: {e.stderr.strip()[-160:]}")
        # The committed index: what is listed and measured, and the notes kept from before.
        doc_path = os.path.join(DOCS, f"{t}.json")
        old = json.load(open(doc_path)) if os.path.exists(doc_path) else {}
        notes = {it["file"]: it["note"] for it in old.get("items", []) if it.get("note")}
        out_items = []
        for it in items:
            if it["kind"] == "video-mp4":
                continue
            row = {"kind": it["kind"], "file": it["file"], "name": it["name"], "url": it["url"], "bytes": it["bytes"]}
            if it["kind"] == "video":
                mp4 = next((x for x in items if x["kind"] == "video-mp4" and x.get("of") == os.path.basename(it["file"])), None)
                if mp4:
                    row["mp4_url"] = mp4["url"]
                if os.path.basename(it["file"]) in probes:
                    row["probe"] = probes[os.path.basename(it["file"])]
            if it["file"] in notes:
                row["note"] = notes[it["file"]]
            out_items.append(row)
        doc = {"test": int(t), "source": idx["source"], "scenario": old.get("scenario"), "summary": old.get("summary", ""), "sources": old.get("sources", []), "items": out_items}
        json.dump(doc, open(doc_path, "w"), indent=1)
        print(f"{t}: {len(probes)} videos probed, {sum(1 for it in out_items if it.get('note'))} annotated -> {os.path.relpath(doc_path)}")


if __name__ == "__main__":
    main()
