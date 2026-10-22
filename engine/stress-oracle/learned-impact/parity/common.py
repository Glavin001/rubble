"""Shared by the parity kit's builder and checker: one hash of a call event's decoded arrays
and energies (independent of how the arrays are laid out in the file)."""
import hashlib, json
from pathlib import Path

W = {"f32": 4, "i32": 4, "i64": 8, "f64": 8}


def calls_sha256(call_dir: Path) -> tuple[int, str]:
    import zstandard
    raw = zstandard.ZstdDecompressor().stream_reader(open(call_dir / "arrays.bin.zst", "rb")).read()
    calls = json.loads((call_dir / "calls.json").read_text())
    h = hashlib.sha256()
    for e in calls:
        for k in sorted(e["arrays"]):
            m = e["arrays"][k]
            n = 1
            for s in m["shape"]:
                n *= s
            h.update(k.encode())
            h.update(raw[m["offset"]:m["offset"] + n * W[m["dtype"]]])
        h.update(json.dumps(e["energy"], sort_keys=True).encode())
    return len(calls), h.hexdigest()
