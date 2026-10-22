"""Pack a training dataset into one archive for Modal (runs on the host, free).

    python3 pack_data.py --train <calls-train> --val <calls-eval> --name bulk-v1-calls3 \
        [--val-split val] [--max-train N]

Writes `learned-impact/runs/modal/<name>.tar` (members are the calls' own files: arrays.bin.zst
is already compressed) with `train/<event>/…` and `val/<event>/…`, plus `<name>.json` (event
lists, byte counts, sha256). Deterministic: sorted members, zero mtimes and owners, so the
same inputs give the same hash and an unchanged dataset is never uploaded twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "modal"


def events(root: Path, split: str | None) -> list[Path]:
    out = []
    for p in sorted(root.iterdir()):
        m = p / "manifest.json"
        if p.is_dir() and m.exists():
            if split is None or json.loads(m.read_text()).get("split") == split:
                out.append(p)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--out-dir", default=str(RUNS))
    a = ap.parse_args(argv)
    runs = Path(a.out_dir)
    runs.mkdir(parents=True, exist_ok=True)
    tr = events(Path(a.train), "train")
    if a.max_train:
        tr = tr[: a.max_train]
    va = events(Path(a.val), a.val_split)
    out = runs / f"{a.name}.tar"
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
        for prefix, evs in (("train", tr), ("val", va)):
            for ev in evs:
                for f in sorted(ev.iterdir()):
                    info = tar.gettarinfo(f, arcname=f"{prefix}/{ev.name}/{f.name}")
                    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                    with open(f, "rb") as fh:
                        tar.addfile(info, fh)
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    meta = {"name": a.name, "file": out.name, "bytes": out.stat().st_size, "sha256": h.hexdigest(),
            "train_events": len(tr), "val_events": len(va),
            "train_src": str(a.train), "val_src": str(a.val), "val_split": a.val_split}
    (runs / f"{a.name}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    main()
