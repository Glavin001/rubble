"""The viability report (planning session, 2026-10-22): does the full-contract model learn,
without overfitting, with data that scales towards the bar? From finished runs only (no
training here):

1. loss curves per term: the eval-mode loss on equal, fixed samples of training and of
   validation calls at every evaluation (`history.jsonl`), per run;
2. the data-scaling curve: `pilot.py --fractions 0.25,0.5,1` (its `pilot.json`), each
   validation skill metric against the training events, with the saturating power-law fit
   and its projection to the bar;
3. per-frame validation under gates-v1.1 against QS, the punch and the held-out oracle
   member: `report.py` on each run's `best.pt` (`report-val.json`);
4. the checkpoint to hand to P0's closed loop (the best on the dense validation score):
   harness A first (the oracle engine with the model as its island step, against standalone
   oracle ensembles: the acceptance test), then harness B (rubble + model against rubble +
   the oracle island step: integration).

    python -m learned_impact.outcome.viability --runs W64=../runs/v5/w64 W128=../runs/v5/w128 \
        --scaling ../runs/v5/scaling --out ../findings/p2-viability-<date>.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GATES = ("ram", "partition", "sizes")


def read_history(run: Path) -> list[dict]:
    p = run / "history.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def overfit_summary(hist: list[dict]) -> dict:
    """Per term: the val/train ratio at the end, the step of the lowest val loss, and
    whether val still fell over the last quarter of the evaluations."""
    ev = [r for r in hist if r["kind"] == "eval" and "val" in r]
    if not ev:
        return {}
    out = {}
    terms = sorted(ev[-1]["val"]["terms"])
    for k in ["loss", *terms]:
        def get(r, split):
            return r[split]["loss"] if k == "loss" else r[split]["terms"].get(k)
        v = np.array([get(r, "val") for r in ev], float)
        t = np.array([get(r, "train") for r in ev], float)
        q = max(1, len(ev) // 4)
        out[k] = {"val_end": float(v[-1]), "train_end": float(t[-1]),
                  "gap_ratio": float(v[-1] / max(t[-1], 1e-12)),
                  "val_min_step": int(ev[int(np.argmin(v))]["step"]),
                  "val_rising_at_end": bool(v[-1] > v[-q - 1] * 1.02) if len(v) > q else False}
    return out


def plot_curves(name: str, hist: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = [r for r in hist if r["kind"] == "eval" and "val" in r]
    if not ev:
        return
    terms = ["loss", *sorted(ev[-1]["val"]["terms"])]
    n = len(terms)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    steps = [r["step"] for r in ev]
    for ax, k in zip(axes.flat, terms):
        for split, style in (("train", "-"), ("val", "--")):
            y = [r[split]["loss"] if k == "loss" else r[split]["terms"].get(k) for r in ev]
            ax.plot(steps, y, style, label=f"{split} sample")
        ax.set_title(k, fontsize=9)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{name}: per-term loss (normalised by the zero predictor), eval mode", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scaling(pilot: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for variant, d in pilot["variants"].items():
        for ax, g in zip(axes, GATES):
            m = f"skill_over_bar_{g}"
            xs = [n for n in d["sizes"] if d["table"][m][str(n)]["mean"] is not None]
            ys = [d["table"][m][str(n)]["mean"] for n in xs]
            lo = [d["table"][m][str(n)]["min"] for n in xs]
            hi = [d["table"][m][str(n)]["max"] for n in xs]
            eb = ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], marker="o",
                             capsize=3, label=f"variant {variant}")
            f = d["fits"].get(m, {}).get("fit")
            if f:
                xx = np.geomspace(min(xs), max(xs) * 30, 60)
                ax.plot(xx, f["a"] * xx ** (-f["b"]) + f["c"], ":", lw=1,
                        color=eb.lines[0].get_color())
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"{g}: skill ÷ bar (≤ 1 passes)", fontsize=9)
            ax.set_xlabel("training events", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gate_table(rep: dict) -> list[str]:
    lines = ["| family | gate | model pass | member pass | shortfall ub (≤ .05) | model s/bar | "
             "member s/bar | excess ub (≤ 5%) | QS s/bar | punch s/bar |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for fam, a in sorted(rep.get("acceptance", {}).items()):
        fe = rep["families"].get(fam, {})
        for g in GATES:
            e = a.get(g)
            if not e:
                continue
            q = (fe.get("qs") or {}).get(f"skill_over_bar_{g}")
            p = (fe.get("punch") or {}).get(f"skill_over_bar_{g}")
            ok = lambda b: "✓" if b else "✗"  # noqa: E731
            lines.append(
                f"| {fam} ({a['n_ensembles']} ens) | {g} | {e['model_pass_rate']:.3f} | "
                f"{e['member_pass_rate']:.3f} | {e['pass_shortfall_upper']:+.3f} "
                f"{ok(e['pass_shortfall_upper'] <= .05)} | {e['model_mean_skill_over_bar']:.2f} | "
                f"{e['member_mean_skill_over_bar']:.2f} | {100 * e['skill_ratio_excess_upper']:+.0f}% "
                f"{ok(e['skill_ratio_excess_upper'] <= .05)} | "
                f"{'—' if q is None else f'{q:.2f}'} | {'—' if p is None else f'{p:.2f}'} |")
    return lines


def build(runs: dict[str, Path], scaling: Path | None, out: Path) -> Path:
    figdir = out.with_suffix("")
    figdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Viability report: {out.stem}", "",
          "Generated by `learned_impact.outcome.viability` from finished runs. The CPU "
          "oracle is the only truth.", ""]
    best = None
    md += ["## 1. Loss curves (eval-mode, equal fixed samples of training and validation calls)", ""]
    for name, run in runs.items():
        hist = read_history(run)
        plot_curves(name, hist, figdir / f"curves-{name}.png")
        ov = overfit_summary(hist)
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        mc = cfg.get("model", {})
        md += [f"### {name}: W{mc.get('width')} rounds {mc.get('rounds')} variant "
               f"{mc.get('variant')} drop {mc.get('drop_features') or 'none'}", "",
               f"![{name} curves]({figdir.name}/curves-{name}.png)", "",
               "| term | train end | val end | val ÷ train | lowest val at step | val rising at end |",
               "|---|---|---|---|---|---|"]
        md += [f"| {k} | {v['train_end']:.4g} | {v['val_end']:.4g} | {v['gap_ratio']:.2f} | "
               f"{v['val_min_step']} | {'yes' if v['val_rising_at_end'] else 'no'} |"
               for k, v in ov.items()]
        ev = [r for r in hist if r["kind"] == "eval"]
        if ev:
            sc = min(r["score"] for r in ev)
            md += ["", f"Best dense validation score {sc:.3f}.", ""]
            if best is None or sc < best[1]:
                best = (name, sc)
    if scaling and (scaling / "pilot.json").exists():
        pilot = json.loads((scaling / "pilot.json").read_text())
        plot_scaling(pilot, figdir / "scaling.png")
        md += ["## 2. Data scaling (nested shares of the training layout groups)", "",
               f"![scaling]({figdir.name}/scaling.png)", "",
               (scaling / "pilot.md").read_text() if (scaling / "pilot.md").exists() else "", ""]
    md += ["## 3. Per-frame validation, gates-v1.1 (against QS, the punch and the held-out "
           "oracle member)", ""]
    for name, run in runs.items():
        p = run / "report-val.json"
        if not p.exists():
            md += [f"{name}: no `report-val.json` yet.", ""]
            continue
        rep = json.loads(p.read_text())
        md += [f"### {name}: {'ACCEPTED' if rep.get('accepted') else 'not accepted'} "
               f"(τ {rep.get('tau_frozen')}, {rep.get('n_events')} events, "
               f"{rep.get('n_calls')} calls, {len(rep.get('worse_than_qs', []))} worse than QS)", "",
               *gate_table(rep), ""]
    if best:
        md += ["## 4. Closed loop", "",
               "Acceptance is harness A (the oracle engine with the model as its island step, "
               "against standalone oracle ensembles); harness B (rubble + model against rubble + "
               "the oracle island step) checks integration.", "",
               f"Checkpoint for P0's closed-loop comparison against `ceiling_full`: **{best[0]}** "
               f"(`{runs[best[0]] / 'best.pt'}`, best dense validation score {best[1]:.3f}).", ""]
    out.write_text("\n".join(md))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir ...")
    ap.add_argument("--scaling", default=None, help="pilot.py --fractions output dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = {x.split("=", 1)[0]: Path(x.split("=", 1)[1]) for x in a.runs}
    print(build(runs, Path(a.scaling) if a.scaling else None, Path(a.out)))


if __name__ == "__main__":
    main()
