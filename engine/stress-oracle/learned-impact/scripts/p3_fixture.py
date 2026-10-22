"""Copy a `learned-impact-weights/1` export (P2's `outcome.export`) into the Rust parity
fixture of the P3 port: `weights.bin` / `weights.json` as they are, each parity call's
`call/` and `decoded.json` as they are, and `outputs.npz` as `outputs.bin` (little-endian
f32, row-major, each array 16-byte aligned) with `outputs.json` (name → shape, offset), so
the Rust test reads it without a zip reader.

    .venv/bin/python ../scripts/p3_fixture.py EXPORT_DIR ../../crates/stress-learned/tests/fixtures/p3-model

`--extra` adds parity calls the export's own (the first validation calls) do not cover:
cracked and structure-contact edges, broken bonds, anchored walls, several or no
impactors. It runs P2's checkpoint through P2's own package (`--p2-python`, the
`learned-impact/python` directory of P2's checkout, under P2's venv), exactly as
`outcome.export` writes its parity calls:

    P2/python/.venv/bin/python -I ../scripts/p3_fixture.py --extra --p2-python P2/python \
        --checkpoint P2/runs/<job>/best.pt --out FIXTURE --name x0 CALLS_EVENT_DIR:CALL_INDEX …
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main(src: str, dst: str) -> None:
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    for n in ("weights.bin", "weights.json"):
        shutil.copy2(s / n, d / n)
    for p in sorted((s / "parity").iterdir()):
        q = d / "parity" / p.name
        if q.exists():
            shutil.rmtree(q)
        shutil.copytree(p / "call", q / "call")
        shutil.copy2(p / "decoded.json", q / "decoded.json")
        z = np.load(p / "outputs.npz")
        blob, index, off = [], {}, 0
        for k in z.files:
            raw = np.ascontiguousarray(z[k], dtype="<f4").tobytes()
            pad = (-len(raw)) % 16
            index[k] = {"shape": list(z[k].shape), "offset": off}
            blob.append(raw + b"\0" * pad)
            off += len(raw) + pad
        (q / "outputs.bin").write_bytes(b"".join(blob))
        (q / "outputs.json").write_text(json.dumps(index, indent=1))
    print(d)


def extra(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--p2-python", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="x")
    ap.add_argument("calls", nargs="+", help="EVENT_DIR:CALL_INDEX")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.p2_python)
    import torch
    from learned_impact.outcome import decode as D
    from learned_impact.outcome import schema as S
    from learned_impact.outcome import state_io as IO
    from learned_impact.outcome.batch import HierarchyCache, collate
    from learned_impact.outcome.train import load_model

    model, ck = load_model(a.checkpoint, "cpu")
    model.eval()
    tc = ck["train_config"]
    dcfg = D.DecodeConfig(tau=ck.get("tau", 1.0), temperature=ck.get("temperature", 1.0),
                          energy_tolerance=tc["decode"]["energy_tolerance"])
    cache = HierarchyCache(model.cfg.levels)
    for k, spec in enumerate(a.calls):
        ev_dir, idx = spec.rsplit(":", 1)
        ev = S.load_event(ev_dir)
        c = next(c for c in ev.calls if c.call_index == int(idx))
        d = Path(a.out) / "parity" / f"{a.name}{k}"
        if d.exists():
            shutil.rmtree(d)
        S.write_event(d / "call", {**ev.manifest, "event_id": c.event_id}, [c], ev.structures)
        b = collate([c], {c.event_id: ev.structures}, cache)
        with torch.no_grad():
            o = model(b)
        dec = D.decode(o, b, dcfg)
        IO.write(d / "decoded.json", IO.decoded_states(b, o, dec, {"checkpoint": str(a.checkpoint)})[0])
        blob, index, off = [], {}, 0
        for name, v in o.items():
            arr = np.ascontiguousarray(v.numpy(), dtype="<f4")
            raw = arr.tobytes()
            pad = (-len(raw)) % 16
            index[name] = {"shape": list(arr.shape), "offset": off}
            blob.append(raw + b"\0" * pad)
            off += len(raw) + pad
        (d / "outputs.bin").write_bytes(b"".join(blob))
        (d / "outputs.json").write_text(json.dumps(index, indent=1))
        print(d, c.counts)


if __name__ == "__main__":
    if "--extra" in sys.argv:
        extra(sys.argv[1:])
    else:
        main(*sys.argv[1:3])
