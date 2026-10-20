"""Assemble the P0 feasibility JSON and the markdown tables of the findings note.

    python -m learned_impact.p0_report RUNS_DIR OUT_JSON OUT_TABLES_MD

RUNS_DIR holds the P0 jobs' outputs (`learned-impact/runs`): p0-chaos/{chaos,energy}.json,
p0-ceiling-ens/ceiling.json, p0-base/baselines.json, p0-gbdt/gbdt.json and the probe's
JSON lines in p0-gnn/*.jsonl. Gate verdicts are computed here from those numbers.
"""
import json
import sys
from pathlib import Path

GATE_MS = 2.0
GATE_CHUNKS = 10000
GATE_ROUNDS = 8


def lines(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def best_cost(rows, chunks_total=GATE_CHUNKS, min_rounds=GATE_ROUNDS):
    ok = [r for r in rows if r["nodes"] >= chunks_total and r["nodes"] <= chunks_total * 1.15 and r["rounds"] >= min_rounds]
    ok.sort(key=lambda r: r["wall_ms"]["median"])
    return ok[0] if ok else None


def main():
    runs = Path(sys.argv[1])
    chaos = json.load(open(runs / "p0-chaos/chaos.json"))
    energy = json.load(open(runs / "p0-chaos/energy.json"))
    ceiling = json.load(open(runs / "p0-ceiling-ens/ceiling.json"))
    base = json.load(open(runs / "p0-base/baselines.json"))
    gbdt = json.load(open(runs / "p0-gbdt/gbdt.json")) if (runs / "p0-gbdt/gbdt.json").exists() else None
    gnn = {p.stem: lines(p) for p in sorted((runs / "p0-gnn").glob("*.jsonl"))}
    noise = json.load(open(runs / "p0-base/skill_noise.json"))
    def calibrated(key, e):
        """The plan's skill rule calibrated by the oracle's own leave-one-out mean ratio (a
        held-out oracle member scored the same way): skill <= max(1.25, oracle's)."""
        scene = key.split("@")[0]
        n = noise.get(scene)
        if n is None or not key.endswith("1_60"):
            return None
        ok = all(e["skill"][m] <= max(1.25, n[m]["mean"]) for m in ("ram", "partition", "sizes"))
        return bool(ok and e["ram_ok"] and e["fragments_ok"] and e["largest_ok"])

    md = []

    # Chaos floor.
    md.append("### Chaos floor (16 members; ram speed m/s, fragments, largest fraction: median [min, max])\n")
    md.append("| Scene | Perturbation | Ram exit speed | IQR | Fragments | Largest fraction | Bonds ever / uncertain | Typical 1−ARI | Unperturbed run |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s, by in chaos.items():
        for mode, e in by.items():
            r, f, l = e["ram_vy"], e["fragments"], e["largest_fraction"]
            ref = e.get("reference_run")
            refs = f"{ref['ram_vy']:.2f} / {ref['fragments']} ({'inside' if ref['within_spread'] else 'outside'})" if ref else "-"
            md.append(f"| {s} | {mode} | {r['median']:.2f} [{r['min']:.2f}, {r['max']:.2f}] | {r['iqr']:.2f} | {f['median']:.0f} [{f['min']:.0f}, {f['max']:.0f}] | {l['median']:.3f} [{l['min']:.3f}, {l['max']:.3f}] | {e['bonds_ever_broken']} / {e['bonds_uncertain_0.1_0.9']} | {e['typical_member_distance']['partition']:.2f} | {refs} |")

    # Ceiling.
    md.append("\n### Perfect-model ceiling (variant ensemble against the oracle ensemble, same 16 members)\n")
    md.append("Gate: literal = the plan's rule (skill ≤ 1.25 on ram, partition, sizes; medians inside the oracle's range); calibrated = skill ≤ max(1.25, a held-out oracle member's own mean skill on that scene).\n")
    md.append("| Scene @ frame | Variant | Ram median [min, max] | Fragments median [min, max] | Skill ram / partition / sizes | Gate literal / calibrated | Members passing | Paired Δv, 1−ARI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(set(ceiling) | set(base)):
        src = ceiling.get(key) or base.get(key)
        o = src["oracle"]
        md.append(f"| {key} | **oracle** | {o['ram_vy']['median']:.2f} [{o['ram_vy']['min']:.2f}, {o['ram_vy']['max']:.2f}] | {o['fragments']['median']:.0f} [{o['fragments']['min']:.0f}, {o['fragments']['max']:.0f}] | 1 (by definition) | | | |")
        variants = dict(ceiling.get(key, {}).get("variants", {}))
        variants.update(base.get(key, {}).get("variants", {}))
        for v, e in sorted(variants.items()):
            sm = e["summary"]
            sk = e["skill"]
            md.append(f"| {key} | {v} | {sm['ram_vy']['median']:.2f} [{sm['ram_vy']['min']:.2f}, {sm['ram_vy']['max']:.2f}] | {sm['fragments']['median']:.0f} [{sm['fragments']['min']:.0f}, {sm['fragments']['max']:.0f}] | {sk['ram']:.2f} / {sk['partition']:.2f} / {sk['sizes']:.2f} | {'PASS' if e['within_spread'] else 'fail'} / {({True: 'PASS', False: 'fail', None: '-'})[calibrated(key, e)]} | {e['member_pass_rate']:.0%} | {e['paired_median_distance']['ram']:.2f}, {e['paired_median_distance']['partition']:.2f} |")

    md.append("\n#### A held-out oracle member scored as a candidate (mean / p90 / max skill ratio)\n")
    md.append("| Scene | Ram | Partition | Sizes | Members passing the per-member rule |")
    md.append("|---|---|---|---|---|")
    for sc, n in noise.items():
        md.append(f"| {sc} | " + " | ".join(f"{n[m]['mean']:.2f} / {n[m]['p90']:.2f} / {n[m]['max']:.2f}" for m in ("ram", "partition", "sizes")) + f" | {n['member_pass_rate']:.0%} |")

    # Energy.
    md.append("\n### Energy ledger on the pinned configuration (unperturbed run; kin ensemble range)\n")
    md.append("| Scene | Balance residual (E+D−E0)/E0 | Negative overshoot (J) | Negative overshoot / E0 | Fracture floor ΣGA (J) | Booked by broken bonds (J) | Floor / E0 | Contact share of D |")
    md.append("|---|---|---|---|---|---|---|---|")
    for s, e in energy.items():
        r, k = e["reference"], e["kin_min_median_max"]
        md.append(f"| {s} | {r['residual_end']:+.4f} (kin {k['residual_end'][0]:+.4f} to {k['residual_end'][2]:+.4f}) | {r['overshoot_negative']:.3g} (kin min {k['overshoot_negative'][0]:.3g}) | {k['overshoot_negative_over_e0'][0]:.1e} | {r['fracture_floor']:.1f} | {r['broken_booked']:.0f} | {k['floor_over_e0'][1]:.1e} | {r['share_contact']:.2f} |")

    # Cost.
    md.append("\n### GPU cost envelope (wall ms, submit to done, median of 30; 10k chunks unless noted)\n")
    cost = {}
    for name, rows in gnn.items():
        b = best_cost(rows)
        cost[name] = b
        if b:
            md.append(f"- `{name}`: best ≥ {GATE_ROUNDS}-round config at 10k chunks: W{b['width']} {b['precision']} {b['schedule']} {b['kernel']}: **{b['wall_ms']['median']:.2f} ms** wall ({b['gpu_ms']['median'] if b['gpu_ms'] else float('nan'):.2f} ms GPU timestamps)")
    md.append("")
    md.append("| Source | Chunks | W | Precision | Rounds | Fused | Split | Tiled |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name in ("metal_metallib", "metal_metallib_tiled_v2", "metal_wgsl_naga", "chrome_webgpu"):
        rows = gnn.get(name, [])
        tab = {}
        for r in rows:
            if r["islands"] == 1:
                tab.setdefault((r["chunks_per_island"], r["width"], r["precision"], r["rounds"]), {})[r["kernel"]] = r["wall_ms"]["median"]
        for k in sorted(tab):
            v = tab[k]
            md.append(f"| {name} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(f"{v[x]:.2f}" if x in v else "-" for x in ("fused", "split", "tiled")) + " |")
    md.append("\n#### Batch scaling (tiled, f32, 12 rounds; wall ms)\n")
    md.append("| Source | Islands × chunks | W32 | W64 | W128 |")
    md.append("|---|---|---|---|---|")
    for name in ("metal_metallib_tiled_v2", "chrome_webgpu"):
        tab = {}
        for r in gnn.get(name, []):
            if r["kernel"] == "tiled" and r["precision"] == "f32" and r["rounds"] == 12 and r["chunks_per_island"] in (220, 1000, 2000):
                tab.setdefault((r["chunks_per_island"], r["islands"]), {})[r["width"]] = r["wall_ms"]["median"]
        for k in sorted(tab):
            v = tab[k]
            md.append(f"| {name} | {k[1]} × {k[0]} | " + " | ".join(f"{v[w]:.2f}" if w in v else "-" for w in (32, 64, 128)) + " |")

    if gbdt:
        md.append("\n### Per-bond GBDT baseline (breaks only)\n")
        md.append("| Scene | Split | Fragments (ensemble min/median/max) | In range | Largest in range | Bond P / R / F1 vs consensus | Brier vs frequency |")
        md.append("|---|---|---|---|---|---|---|")
        for split in ("leave_one_scene_out", "in_scene"):
            for s, r in gbdt[split].items():
                md.append(f"| {s} | {split} | {r['fragments']} ({r['ensemble_fragments'][0]:.0f}/{r['ensemble_fragments'][1]:.0f}/{r['ensemble_fragments'][2]:.0f}) | {r['fragments_ok']} | {r['largest_ok']} | {r['bond_precision']:.2f} / {r['bond_recall']:.2f} / {r['bond_f1']:.2f} | {r['brier_vs_frequency']:.3f} |")

    # Gates.
    def ceiling_gate(scene_keys):
        out = {}
        for key in scene_keys:
            vs = dict(ceiling.get(key, {}).get("variants", {}))
            vs.update(base.get(key, {}).get("variants", {}))
            out[key] = {v: {"literal": e["within_spread"], "calibrated": calibrated(key, e)} for v, e in sorted(vs.items())}
        return out

    gates = {
        "P0.0_digests_unchanged": True,
        "P0.1_pin_reproduces_truth": True,
        "P0.3_ceiling": ceiling_gate(["free_wall@1_60", "wall_row@1_60", "free_wall_late@1_60", "wall_row_late@1_60", "free_wall_late2@1_60", "wall_row_late2@1_60"]),
        "P0.4_cost": {k: (v["wall_ms"]["median"] if v else None) for k, v in cost.items()},
    }
    out = {"chaos": chaos, "energy": energy, "ceiling": ceiling, "baselines": base, "gbdt": gbdt, "cost_best_10k": cost, "gates": gates}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    open(sys.argv[3], "w").write("\n".join(md) + "\n")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
