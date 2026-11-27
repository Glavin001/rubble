#!/usr/bin/env python3
"""Fetch an NHTSA crash test's photos and videos (public domain, US government) into its
local folder, and list them.

    nhtsa_media.py ROOT [TEST ...] [--list-only] [--jobs 8]

ROOT holds one folder per test (`nhtsa-<test>/api/multimedia-files.json`, saved from the
Vehicle Crash Test Database's API by the reference fetch). Each test's files go to
`nhtsa-<test>/media/{photos,videos}/` (the originals, and a video's transcoded `.mp4` where
NHTSA has one: the form a browser plays); `media/index.json` lists every item: file, bytes,
URL, the lab's own name for it (the camera or the photo's subject) and whether it is here.

Resumable: a file already here at its listed size is kept; a download is written beside
its name and renamed when whole. Local only: ROOT is outside the repository's tracked files
(`inspiration/`), and nothing in the tests or builds reads it. The committed index of this
footage (URLs, names, annotations; no media) is `engine/docs/crash-reference/nhtsa/media/`.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

AGENT = "crash-reference-fetch/1.0"


def listed(test_dir):
    m = json.load(open(os.path.join(test_dir, "api", "multimedia-files.json")))
    r = m[0] if isinstance(m, list) else m.get("results", m)
    if isinstance(r, list):
        r = r[0] if r else {}
    items = []
    for kind, key in (("photo", "photos"), ("video", "videos")):
        for x in r.get(key) or []:
            size = int(x.get("fileSize") or 0)
            items.append({"kind": kind, "file": f"{key}/{x['fileName']}", "url": x["media"], "bytes": size, "name": x.get("originalFileName") or "", "index": x.get("index")})
            mp4 = x.get("transcoded_video_url")
            if kind == "video" and mp4 and not x["fileName"].lower().endswith(".mp4"):
                items.append({"kind": "video-mp4", "file": f"{key}/{os.path.basename(mp4)}", "url": mp4, "bytes": 0, "name": x.get("originalFileName") or "", "index": x.get("index"), "of": x["fileName"]})
    return items


def fetch(media, it):
    path = os.path.join(media, it["file"])
    if os.path.exists(path) and (it["bytes"] == 0 or os.path.getsize(path) == it["bytes"]):
        return "kept"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    try:
        req = urllib.request.Request(it["url"], headers={"User-Agent": AGENT})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        if it["bytes"] and os.path.getsize(part) != it["bytes"]:
            return f"size {os.path.getsize(part)} != listed {it['bytes']}"
        os.replace(part, path)
        return "fetched"
    except Exception as e:  # noqa: BLE001 (reported per file; the run goes on)
        return f"failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("tests", nargs="*")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    tests = a.tests or sorted(d[6:] for d in os.listdir(a.root) if d.startswith("nhtsa-"))
    bad = 0
    for t in tests:
        td = os.path.join(a.root, f"nhtsa-{t}")
        items = listed(td)
        media = os.path.join(td, "media")
        if not a.list_only:
            with cf.ThreadPoolExecutor(a.jobs) as ex:
                for it, res in zip(items, ex.map(lambda it: fetch(media, it), items)):
                    if res not in ("kept", "fetched"):
                        bad += 1
                        print(f"{t} {it['file']}: {res}", file=sys.stderr)
        for it in items:
            p = os.path.join(media, it["file"])
            it["here"] = os.path.exists(p)
            if it["here"] and not it["bytes"]:
                it["bytes"] = os.path.getsize(p)
        os.makedirs(media, exist_ok=True)
        json.dump({"test": int(t), "source": "NHTSA Vehicle Crash Test Database (public domain, US government)", "items": items}, open(os.path.join(media, "index.json"), "w"), indent=1)
        here = sum(1 for it in items if it["here"])
        print(f"{t}: {here}/{len(items)} here, {sum(it['bytes'] for it in items if it['here']) / 1e6:.0f} MB")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
