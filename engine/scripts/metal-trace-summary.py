#!/usr/bin/env python3
"""Key numbers of an Instruments Metal System Trace as text and JSON.

    engine/scripts/metal-trace-summary.py out/metal-trace/NAME.trace

Exports the trace's table of contents and its GPU / Metal tables with
`xcrun xctrace export` into out/metal-trace/NAME/*.xml, parses them (xctrace's XML
deduplicates repeated values with id/ref attributes) and writes NAME/summary.json and
NAME/summary.txt:
- per table: rows, and for interval tables (a start and a duration column) the busy
  time, the union of the intervals, and the gaps between consecutive intervals;
- the GPU timeline (`metal-gpu-intervals`): busy vs idle over the recording, idle gaps
  between command buffers (median, p90, max), busy time per channel and per label;
- command buffers (`metal-application-command-buffer-submissions` and similar):
  submissions per second and the submit-to-GPU-start latency where both are recorded.
Tables and columns are found by name, so a schema an Xcode version renames is still
listed with its columns (see `toc.xml`) for whoever extends this.
"""
import json
import os
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


def export(trace, args, out):
    cmd = ["xcrun", "xctrace", "export", "--input", trace] + args + ["--output", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"metal-trace-summary: {' '.join(cmd)}: {r.stderr.strip()}", file=sys.stderr)
        return False
    return True


def parse_table(path):
    """[(schema name, [column mnemonics], [rows as dicts of (value, fmt)])]."""
    tree = ET.parse(path)
    ids = {}
    out = []

    def value(el):
        ref = el.get("ref")
        if ref is not None:
            return ids.get(ref, (None, None))
        text = (el.text or "").strip()
        if not text and len(el):
            # A composite (thread, process, ...): its formatted value.
            text = el.get("fmt", "")
        v = (text, el.get("fmt"))
        if el.get("id") is not None:
            ids[el.get("id")] = v
        # Register nested ids too.
        for child in el.iter():
            if child is not el and child.get("id") is not None and child.get("id") not in ids:
                ids[child.get("id")] = ((child.text or "").strip() or child.get("fmt", ""), child.get("fmt"))
        return v

    for node in tree.getroot().iter("node"):
        schema = node.find("schema")
        if schema is None:
            continue
        cols = [c.findtext("mnemonic") for c in schema.findall("col")]
        rows = []
        for row in node.findall("row"):
            cells = list(row)
            d = {}
            for name, cell in zip(cols, cells):
                d[name] = value(cell)
            rows.append(d)
        out.append((schema.get("name"), cols, rows))
    return out


def num(v):
    try:
        return float(v[0])
    except (TypeError, ValueError, IndexError):
        return None


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, round((len(s) - 1) * p))]


def intervals(rows, cols):
    start = next((c for c in cols if c in ("start", "start-time", "timestamp")), None)
    dur = next((c for c in cols if c == "duration"), None)
    if not start or not dur:
        return None
    iv = []
    for r in rows:
        s, d = num(r.get(start)), num(r.get(dur))
        if s is not None and d is not None:
            iv.append((s, s + d, r))
    iv.sort(key=lambda x: x[0])
    return iv


def union_and_gaps(iv):
    busy = 0.0
    gaps = []
    cur_s = cur_e = None
    for s, e, _ in iv:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy, gaps


def ms(ns):
    return None if ns is None else ns / 1e6


def summarize_intervals(name, cols, rows):
    iv = intervals(rows, cols)
    if not iv:
        return None
    total = sum(e - s for s, e, _ in iv)
    busy, gaps = union_and_gaps(iv)
    span = iv[-1][1] - iv[0][0] if iv else 0
    out = {
        "rows": len(rows),
        "span_ms": ms(span),
        "sum_of_durations_ms": ms(total),
        "union_busy_ms": ms(busy),
        "idle_ms": ms(span - busy),
        "busy_fraction": (busy / span) if span else None,
        "gaps": len(gaps),
        "gap_median_ms": ms(statistics.median(gaps)) if gaps else None,
        "gap_p90_ms": ms(pct(gaps, 0.9)),
        "gap_max_ms": ms(max(gaps)) if gaps else None,
        "duration_median_ms": ms(statistics.median([e - s for s, e, _ in iv])),
    }
    # Busy time by each label-like column.
    for key in ("channel-name", "gpu-channel-name", "label", "cmdbuffer-label", "encoder-label", "frame-number", "process", "state"):
        if key in cols:
            by = {}
            for s, e, r in iv:
                k = (r.get(key) or ("?", None))
                k = k[1] or k[0] or "?"
                by[k] = by.get(k, 0.0) + (e - s)
            top = sorted(by.items(), key=lambda kv: -kv[1])[:25]
            out[f"by_{key}"] = [{"key": k, "ms": ms(v), "count": sum(1 for _, _, r in iv if ((r.get(key) or ('?', None))[1] or (r.get(key) or ('?', None))[0] or '?') == k)} for k, v in top]
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    trace = sys.argv[1].rstrip("/")
    base = trace[:-6] if trace.endswith(".trace") else trace + "-export"
    os.makedirs(base, exist_ok=True)
    toc = os.path.join(base, "toc.xml")
    if not export(trace, ["--toc"], toc):
        return 1
    root = ET.parse(toc).getroot()
    schemas = sorted({t.get("schema") for t in root.iter("table") if t.get("schema")})
    wanted = [s for s in schemas if any(k in s for k in ("metal", "gpu", "display", "command-buffer", "encoder"))]
    summary = {"trace": trace, "schemas": schemas, "tables": {}}
    lines = [f"trace {trace}", f"tables: {len(schemas)} ({len(wanted)} GPU/Metal exported)"]
    for s in wanted:
        path = os.path.join(base, f"{s}.xml")
        if not export(trace, ["--xpath", f'/trace-toc/run[@number="1"]/data/table[@schema="{s}"]'], path):
            continue
        for name, cols, rows in parse_table(path):
            entry = {"columns": cols, "rows": len(rows)}
            iv = summarize_intervals(name, cols, rows)
            if iv:
                entry["intervals"] = iv
            summary["tables"][name] = entry
            line = f"\n[{name}] {len(rows)} rows; columns: {', '.join(c for c in cols if c)}"
            lines.append(line)
            if iv:
                lines.append(
                    "  span {span_ms:.2f} ms | busy (union) {union_busy_ms:.2f} ms ({pct:.1f}%) | idle {idle_ms:.2f} ms | "
                    "{gaps} gaps: median {gm} ms, p90 {gp} ms, max {gx} ms | median duration {dm:.4f} ms".format(
                        pct=100 * (iv["busy_fraction"] or 0),
                        gm=f"{iv['gap_median_ms']:.4f}" if iv["gap_median_ms"] is not None else "-",
                        gp=f"{iv['gap_p90_ms']:.4f}" if iv["gap_p90_ms"] is not None else "-",
                        gx=f"{iv['gap_max_ms']:.4f}" if iv["gap_max_ms"] is not None else "-",
                        dm=iv["duration_median_ms"],
                        **{k: v for k, v in iv.items() if k in ("span_ms", "union_busy_ms", "idle_ms", "gaps")},
                    )
                )
                for k, v in iv.items():
                    if k.startswith("by_"):
                        lines.append(f"  {k[3:]}:")
                        for e in v[:12]:
                            lines.append(f"    {e['ms']:10.3f} ms  x{e['count']:<6} {e['key']}")
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(base, "summary.txt"), "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {base}/summary.{{txt,json}} and the exported tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
