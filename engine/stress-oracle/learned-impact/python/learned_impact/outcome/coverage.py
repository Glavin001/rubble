"""Validation tiers by training coverage (planning with the user, 2026-10-22): validation and
test are reported in three tiers, never one blended average, and selection uses tiers 1+2.

- tier 1, in distribution: a training layout in the same regime (its layout key occurs in
  training with the same regime, see below) and every factor covered; also the in-memory held-out speeds of training ballistic families
  (`train.holdout_ballistic_speeds`).
- tier 2, covered-factor generalisation: an unseen layout whose factors are each covered.
- tier 3, extrapolation: any factor outside training coverage, or a recipe P1 designed as
  extrapolation (the sealed set's x-* splits, `parameters.extrapolation`).

Classified from P1's recipes only (design parameters, never outcomes; the sealed set's
recipes are read, its recordings are not). The factors, each against the training recipes:

- layout: wall depth in chunks, chunk size, thickness (within the training range),
  supports, an opening (any vs none);
- impactors: their count; each one's shape, mass and speed (within the training range for
  that shape);
- material: stiffness, strength and toughness scales (within the training range), and the
  Weibull modulus (set vs unset);
- loads: the kinds of scene loads and scripted events;
- the joint regime factor: a ballistic-limit sweep (P1's p1-ballistic catalog: speeds
  bracketing the perforation transition) at this wall depth. Penetration behaviour
  depends on the depth (findings/p2-ballistic-penetration-vs-layout-2026-10-22.md):
  the transition regime at a depth is covered only by ballistic training data at that depth.

    python -m learned_impact.outcome.coverage --recipes <P1 runs root> --out tiers.json [--md tiers.md]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

TIER_NAMES = {1: "in-distribution", 2: "covered-factor generalisation", 3: "extrapolation"}
# Range factors (thickness, impactor mass and speed, material scales) are covered within the
# training samples' [min, max] widened by this relative margin: a validation sample at the
# edge of the same design distribution (e.g. a speed of 0.10 m/s where training's lowest
# draw is 0.12, both from P1's 0.1–60 m/s) is not extrapolation. P1's designed
# extrapolation sets (x-*) are flagged as such regardless.
RANGE_MARGIN = 0.05


def _inside(x: float, lo: float, hi: float) -> bool:
    return lo * (1 - RANGE_MARGIN) <= x <= hi * (1 + RANGE_MARGIN)


def _split(r: dict) -> str:
    p = r["parameters"]
    return p.get("split") or p.get("ballistic", {}).get("split") or ""


def factors(r: dict) -> dict:
    p = r["parameters"]
    L = p["layout"]
    sc = r.get("scene", {})
    m = p.get("material", {})
    return {
        "layout_key": L["key"],
        "depth": int(L["counts"][1]),
        "chunk": float(L["chunk"]),
        "thickness": round(float(L["counts"][1] * L["chunk"]), 6),
        "supports": L["supports"],
        "opening": L.get("opening") is not None,
        "n_impactors": len(p["impactors"]),
        "impactors": [(i["shape"], float(i["mass"]), float(i["speed"])) for i in p["impactors"]],
        "material": {k: float(m[k]) for k in ("stiffness_scale", "strength_scale", "toughness_scale") if k in m},
        "weibull": m.get("weibull_modulus") is not None,
        "loads": tuple(sorted({str(x.get("kind", x.get("type", "load"))) for x in sc.get("loads", []) or []}
                              | {str(x.get("kind", x.get("type", "event"))) for x in sc.get("events", []) or []})),
        "regime": ("transition", int(L["counts"][1])) if p.get("catalog") == "p1-ballistic" else None,
        "extrapolation": p.get("extrapolation"),
    }


class Coverage:
    def __init__(self, train: list[dict]):
        f = [factors(r) for r in train]
        # tier 1 needs the layout seen in the same regime (a wall layout trained only at broad
        # speeds is not in-distribution for a ballistic transition sweep on it)
        self.keys = {(x["layout_key"], x["regime"]) for x in f}
        self.sets = {k: {x[k] for x in f} for k in ("depth", "chunk", "supports", "opening", "n_impactors",
                                                     "weibull", "loads")}
        self.thickness = (min(x["thickness"] for x in f), max(x["thickness"] for x in f))
        self.regimes = {x["regime"] for x in f if x["regime"]}
        self.shapes: dict[str, list] = defaultdict(lambda: [float("inf"), float("-inf"), float("inf"), float("-inf")])
        for x in f:
            for s, mass, v in x["impactors"]:
                b = self.shapes[s]
                b[0], b[1], b[2], b[3] = min(b[0], mass), max(b[1], mass), min(b[2], v), max(b[3], v)
        self.material = {k: (min(x["material"][k] for x in f if k in x["material"]),
                             max(x["material"][k] for x in f if k in x["material"]))
                         for k in ("stiffness_scale", "strength_scale", "toughness_scale")}

    def classify(self, r: dict) -> tuple[int, list[str]]:
        x = factors(r)
        why = []
        if x["extrapolation"]:
            why.append(f"designed extrapolation: {x['extrapolation']}")
        for k, s in self.sets.items():
            if x[k] not in s:
                why.append(f"{k} {x[k]!r} not in training {sorted(s, key=str)}")
        lo, hi = self.thickness
        if not _inside(x["thickness"], lo, hi):
            why.append(f"thickness {x['thickness']} outside training [{lo}, {hi}]")
        for s, mass, v in x["impactors"]:
            if s not in self.shapes:
                why.append(f"impactor shape {s} not in training")
                continue
            m0, m1, v0, v1 = self.shapes[s]
            if not _inside(mass, m0, m1):
                why.append(f"{s} mass {mass:.0f} outside training [{m0:.0f}, {m1:.0f}]")
            if not _inside(v, v0, v1):
                why.append(f"{s} speed {v:.2f} outside training [{v0:.2f}, {v1:.2f}]")
        for k, (lo, hi) in self.material.items():
            if k in x["material"] and not _inside(x["material"][k], lo, hi):
                why.append(f"{k} {x['material'][k]:.3f} outside training [{lo:.3f}, {hi:.3f}]")
        if x["regime"] and x["regime"] not in self.regimes:
            why.append(f"ballistic transition regime at depth {x['regime'][1]} not in training "
                       f"(training: depths {sorted(d for _, d in self.regimes)})")
        if why:
            return 3, why
        return (1 if (x["layout_key"], x["regime"]) in self.keys else 2), []


def load_recipes(root: Path, dirs: str = "viability-v1-*,sealed-v1-*") -> list[dict]:
    out = []
    for pat in dirs.split(","):
        for f in sorted(glob.glob(str(root / pat / "recipes.json"))):
            out += json.load(open(f))["recipes"]
    return out


def tiers(root: Path, train_ids: set | None = None, dirs: str = "viability-v1-*,sealed-v1-*") -> dict:
    """`train_ids`: the event ids of the run's actual training set (its single runs and
    resident ensembles); default every recipe with split train under `dirs`."""
    recs = load_recipes(root, dirs)
    train = [r for r in recs if _split(r) == "train" and (train_ids is None or r["event_id"] in train_ids)]
    if train_ids is not None:
        missing = train_ids - {r["event_id"] for r in train}
        if missing:
            raise ValueError(f"{len(missing)} training events without a recipe, e.g. {sorted(missing)[:3]}")
    cov = Coverage(train)
    events, fam = {}, defaultdict(Counter)
    for r in recs:
        sp = _split(r)
        if sp == "train":
            continue
        t, why = cov.classify(r)
        events[r["event_id"]] = {"tier": t, "split": sp, "family": r["family"], "why": why,
                                 "layout_key": r["parameters"]["layout"]["key"]}
        fam[(r["family"], sp)][t] += 1
    return {"recipes_root": str(root), "n_train_recipes": len(train), "events": events,
            "families": {f"{f}|{s}": dict(c) for (f, s), c in sorted(fam.items())},
            "rules": __doc__.split("    python -m")[0]}


def event_tier(tmap: dict, event_id: str, split: str) -> int:
    """An event's tier: from the map, or 1 for a training event held out in memory (the
    held-out ballistic speeds; its recipe says train)."""
    e = tmap.get("events", {}).get(event_id)
    if e is not None:
        return int(e["tier"])
    return 1 if split == "val" else 0


def markdown(t: dict) -> str:
    lines = ["| family | split | tier 1 | tier 2 | tier 3 | why tier 3 (first event) |", "|---|---|---:|---:|---:|---|"]
    first = {}
    for eid, e in t["events"].items():
        if e["tier"] == 3:
            first.setdefault((e["family"], e["split"]), "; ".join(e["why"][:2]))
    for k, c in t["families"].items():
        f, s = k.split("|")
        lines.append(f"| {f} | {s} | {c.get(1, 0)} | {c.get(2, 0)} | {c.get(3, 0)} | {first.get((f, s), '')} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--recipes", required=True, help="P1's runs root (…/runs/*/recipes.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    ap.add_argument("--dirs", default="viability-v1-*,sealed-v1-*", help="recipe dirs under --recipes")
    ap.add_argument("--train-events", default="", help="file of training recording dirs (their names "
                    "are event ids), as train.py's --train-events")
    ap.add_argument("--train-dataset", default="", help="calls dir whose split-train events are "
                    "training too (resident ensembles)")
    ap.add_argument("--exclude-families", default="", help="training families the run leaves out")
    a = ap.parse_args(argv)
    ids = None
    if a.train_events or a.train_dataset:
        ids = {Path(x).name for x in Path(a.train_events).read_text().split()} if a.train_events else set()
        if a.train_dataset:
            for m in Path(a.train_dataset).glob("*/manifest.json"):
                d = json.loads(m.read_text())
                if d.get("split") == "train":
                    ids.add(d["event_id"])
    if ids is not None and a.exclude_families:
        ex = set(a.exclude_families.split(","))
        ids = {i for i in ids if not any(i.startswith(f"p1_{f}_") for f in ex)}
    t = tiers(Path(a.recipes), ids, a.dirs)
    Path(a.out).write_text(json.dumps(t, indent=1))
    md = markdown(t)
    if a.md:
        Path(a.md).write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
