#!/usr/bin/env python3
"""Mass report, part statistics, crash-section grouping and load-path sections of a
parsed LS-DYNA vehicle deck.

    python3 -I engine/tools/fe-import/fe_report.py CACHE_DIR OUT_DIR

Needs CACHE_DIR/deck.pickle (lsdyna_deck.py) and CACHE_DIR/mass.pickle (fe_mass.py).
Writes OUT_DIR/report.json, OUT_DIR/report.md and OUT_DIR/part_sections.json (pid ->
crash section, used by the mesh export). Standard library only.

Coordinates of the CCSA Camry deck: x forward (the front bumper is near x = 0, the car
lies at negative x), y to the left, z up, ground at z ~ 0; units t, mm, s, N.
"""
import json
import math
import os
import pickle
import re
import sys

SECTIONS = ["front_lower", "front_upper", "cabin", "door", "rear", "wheel_suspension", "other", "unclassified"]

# Ordered rules on the part name (number prefix stripped, lower case). First match wins.
# A rule may carry a position test on the part's mass centroid (x in mm, the car at x<0).
REAR_X = -2900.0   # behind the rear seat / rear door opening: rear structure
RULES = [
    ("other", r"^(spot weld|accelerometers|dummysprings|wall$)"),
    ("other", r"(engine|engnemount|transmission|alternator|waterpump|intake|exhaust|battery|radiator$|radiator_null|"
              r"radiatorbottom|radiatortop|radiatorfan|radiatormotor|radiatortrim|coolant|oilpan|pully|belt)"),
    ("other", r"^(doorfrontwindow|doorfrontwindowr|doorrearwindow|doorrearwindowsmall|doorearwindowr)$"),  # door glass
    ("door", r"(door|hinges)"),
    ("other", r"(seat|searrear|headrest|dash|console|airbag|steeringwheel|steeringcol|tray|glove|radio|display|"
              r"speaker|trim|accelerat|accell|brake|breakboos|gastank|fuel|taillight|headlight|uppertailleft|"
              r"window|windshield|motorbrkt|supportbrkt$|knee|bench)"),
    ("wheel_suspension", r"(tire|rim|^hub|strubtube|aarcon|rotor|^disk|spindle|strut(?!housing|tower|enginemt)|stut|susp|aarm|"
                         r"armmount|antisway|steeringrack|shock(?!housing|gun))"),
    ("rear", r"(bumperrear|bumperfoamrear|shelfpenel|shockgun|rear bumper|trunk|shelfpanel|rearcloseout|quarterpanel|"
             r"wheelhousingrear|wheelhouserear|shockhousingrear|wheelhousingextrear)"),
    ("rear", r"(subframe|subframr|subfame|subrame|subfrarme|su brame|shotgun|shotfun|floor|wheelwellouter|wheelhousingext)", lambda c: c[0] < REAR_X),
    ("front_lower", r"(bumperfront|bumper_foam_front|bumpermidrail|bumperextra|midrail|midreail|midraril|"
                    r"midrearil|radiatorsupport|radiatorsuprt|radiatorcrossmember|radiatorframe|radiatormid|"
                    r"upperradiatorsupport|uppertiebar|subframe|subfame|subrame|subfrarme|su brame)"),
    ("front_upper", r"(hood|fender|cowl|upperrail|upperreail|uppertarilbridge|struthousing|struttower|"
                    r"strutenginemt|wheelhousingextfront|wheelwellouter|strutplatetop|stutplatetop)"),
    ("cabin", r"(pillar|pilar|rocker|roof|floor|tunnel|firewall|sidepanel|shotgun|shotfun|ipbeam|"
              r"wheelwellsuprt|wheelhousingrearcover|frontroofheader)"),
]
COMPILED = [(r[0], re.compile(r[1]), r[2] if len(r) > 2 else None) for r in RULES]


def classify(name, c):
    n = re.sub(r"^\d+_", "", name).lower()
    for sec, rx, test in COMPILED:
        if rx.search(n) and (test is None or test(c)):
            return sec
    return "unclassified"


def mass_props(nodes_xyz, masses):
    M = sum(masses)
    cx = sum(m * p[0] for m, p in zip(masses, nodes_xyz)) / M
    cy = sum(m * p[1] for m, p in zip(masses, nodes_xyz)) / M
    cz = sum(m * p[2] for m, p in zip(masses, nodes_xyz)) / M
    I = [[0.0] * 3 for _ in range(3)]
    for m, p in zip(masses, nodes_xyz):
        x, y, z = p[0] - cx, p[1] - cy, p[2] - cz
        I[0][0] += m * (y * y + z * z); I[1][1] += m * (x * x + z * z); I[2][2] += m * (x * x + y * y)
        I[0][1] -= m * x * y; I[0][2] -= m * x * z; I[1][2] -= m * y * z
    I[1][0], I[2][0], I[2][1] = I[0][1], I[0][2], I[1][2]
    return M, (cx, cy, cz), I


def main():
    cache, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d = pickle.load(open(os.path.join(cache, "deck.pickle"), "rb"))
    ms = pickle.load(open(os.path.join(cache, "mass.pickle"), "rb"))
    parts, mats, secs = d["parts"], d["mats"], d["sections"]
    pm = ms["parts"]
    X = d["xyz"]; nid = d["node_id"]
    veh = [p for p in parts if parts[p]["source"] == "vehicle"]
    rep = {}

    # ---------------- counts ----------------
    n_shell = sum(pm[p]["n_shell"] for p in veh if p in pm)
    n_solid = sum(pm[p]["n_solid"] for p in veh if p in pm)
    n_beam = sum(pm[p]["n_beam"] for p in veh if p in pm)
    barrier_nodes = d.get("barrier_nodes", set())
    area = sum(pm[p]["area"] for p in veh if p in pm)
    rep["counts"] = dict(parts_vehicle=len(veh), parts_with_elements=len([p for p in veh if p in pm]),
                         nodes_total=len(nid), nodes_barrier=len(barrier_nodes),
                         shells=n_shell, solids=n_solid, beams=n_beam, discrete=len(d["discrete"]),
                         elements=n_shell + n_solid + n_beam,
                         seatbelt_accelerometers=d.get("other_elements", {}).get("*ELEMENT_SEATBELT_ACCELEROMETER", 0),
                         mean_shell_area_mm2=area / n_shell, rms_shell_edge_mm=math.sqrt(area / n_shell))
    kc = {}
    for (src, kw), n in d["keyword_counts"].items():
        kc.setdefault(src, {})[kw] = n
    rep["keyword_cards"] = kc

    # ---------------- materials ----------------
    mt = {}
    for p in veh:
        m = mats.get(parts[p]["mid"], {})
        t = m.get("type", "?")
        e = mt.setdefault(t, dict(parts=0, mass_kg=0.0))
        e["parts"] += 1; e["mass_kg"] += pm.get(p, {}).get("mass", 0.0) * 1000
    rep["material_models"] = mt
    grades = {}
    for p in veh:
        m = mats.get(parts[p]["mid"], {})
        if m.get("type") in ("PIECEWISE_LINEAR_PLASTICITY",) and (m.get("ro") or 0) > 7e-9:
            g = grades.setdefault(m.get("sigy"), dict(parts=0, mass_kg=0.0, lcss=set()))
            g["parts"] += 1; g["mass_kg"] += pm.get(p, {}).get("mass", 0.0) * 1000; g["lcss"].add(m.get("lcss"))
    rep["steel_grades_by_yield_MPa"] = {str(k): dict(parts=v["parts"], mass_kg=round(v["mass_kg"], 2),
                                                     curves=sorted(x for x in v["lcss"] if x))
                                         for k, v in sorted(grades.items(), key=lambda kv: kv[0] or 0)}
    curves = {}
    for lc, pts in d["curves"].items():
        curves[str(lc)] = pts
    rep["curves"] = curves

    # ---------------- mass layers ----------------
    struct = ms["struct"]; L = ms["layers"]
    N = len(struct)
    is_barrier = [False] * N
    for n in barrier_nodes:
        is_barrier[nid[n]] = True
    def layer_sum(arr):
        return sum(arr[i] for i in range(N) if not is_barrier[i])
    struct_kg = layer_sum(struct) * 1000
    lay = {k: sum(a) * 1000 for k, a in L.items()}
    msc = sum(v[0] for v in ms["mass_scaling"].values()) * 1000
    rep["mass_layers_kg"] = dict(elements=struct_kg, **lay, mass_scaling_estimate=msc)
    rep["added_mass_items"] = {k: [(a, b, c * 1000) for (a, b, c) in v] for k, v in ms["layer_items"].items()}

    pts = [(X[3 * i], X[3 * i + 1], X[3 * i + 2]) for i in range(N)]
    configs = {
        "structure_only": ["struct"],
        "as_delivered_without_fuel": ["struct", "added_parts", "point_main"],
        "as_delivered_with_fuel": ["struct", "added_parts", "point_main", "fuel"],
        "ncap_test_mass": ["struct", "added_parts", "point_main", "fuel", "dummies_cargo"],
    }
    fa = (-945.4, 0.0, 337.0)   # front wheel centre (hub/tire centroids, both sides)
    ra = (-3735.6, 0.0, 339.0)  # rear wheel centre
    props = {}
    for name, keys in configs.items():
        arrs = [struct if k == "struct" else L[k] for k in keys]
        mm = [0.0 if is_barrier[i] else sum(a[i] for a in arrs) for i in range(N)]
        idx = [i for i in range(N) if mm[i] > 0]
        M, c, I = mass_props([pts[i] for i in idx], [mm[i] for i in idx])
        t2 = 1e-3 * 1e-6  # t*mm^2 -> kg*m^2 : 1e3 kg * 1e-6 m^2
        props[name] = dict(mass_kg=M * 1000,
                           cg_mm=dict(x=c[0], y=c[1], z=c[2]),
                           cg_aft_of_front_axle_mm=fa[0] - c[0],
                           cg_ahead_of_rear_axle_mm=c[0] - ra[0],
                           cg_height_above_ground_mm=c[2],
                           inertia_kgm2=dict(roll_Ixx=I[0][0] * 1e-3, pitch_Iyy=I[1][1] * 1e-3, yaw_Izz=I[2][2] * 1e-3,
                                             Ixy=I[0][1] * 1e-3, Ixz=I[0][2] * 1e-3, Iyz=I[1][2] * 1e-3))
    rep["mass_properties"] = props
    rep["axles_mm"] = dict(front_x=fa[0], rear_x=ra[0], wheelbase=fa[0] - ra[0])

    # ---------------- per-part table, classes ----------------
    table = []
    for p in veh:
        s = pm.get(p)
        if not s:
            continue
        m = s["mass"]
        c = (s["mx"] / m, s["my"] / m, s["mz"] / m) if m > 0 else tuple((a + b) / 2 for a, b in zip(s["lo"], s["hi"]))
        mat = mats.get(parts[p]["mid"], {})
        sec = secs.get(parts[p]["secid"], {})
        table.append(dict(pid=p, name=parts[p]["title"], mass_kg=m * 1000, section=classify(parts[p]["title"], c),
                          mat=mat.get("type"), sigy=mat.get("sigy"), rho=mat.get("ro"), t=sec.get("t"),
                          kind=sec.get("kind"), n_shell=s["n_shell"], n_solid=s["n_solid"], n_beam=s["n_beam"],
                          centroid=c, lo=s["lo"], hi=s["hi"]))
    table.sort(key=lambda r: -r["mass_kg"])
    rep["parts_top40"] = table[:40]
    tot = sum(r["mass_kg"] for r in table)
    cum, marks, k = 0.0, {}, 0
    for r in table:
        cum += r["mass_kg"]; k += 1
        for q in (0.5, 0.8, 0.9, 0.95, 0.99):
            if q not in marks and cum >= q * tot:
                marks[q] = k
    rep["cumulative_parts_for_element_mass"] = {str(q): n for q, n in marks.items()}
    rep["cumulative_curve"] = [round(sum(r["mass_kg"] for r in table[:n]) / tot, 4) for n in
                               (1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 700, 900, len(table))]
    by = {s: dict(parts=0, mass_kg=0.0) for s in SECTIONS}
    for r in table:
        by[r["section"]]["parts"] += 1; by[r["section"]]["mass_kg"] += r["mass_kg"]
    rep["sections"] = by
    rep["unclassified"] = [(r["name"], round(r["mass_kg"], 3)) for r in table if r["section"] == "unclassified"]
    json.dump({str(r["pid"]): r["section"] for r in table}, open(os.path.join(out, "part_sections.json"), "w"))
    rep["section_rules"] = [(r[0], r[1], "x<%g" % REAR_X if len(r) > 2 else None) for r in RULES]
    # all parts, compact
    json.dump([dict(pid=r["pid"], name=r["name"], mass_kg=round(r["mass_kg"], 4), section=r["section"],
                    mat=r["mat"], sigy=r["sigy"], t=r["t"], centroid=[round(x, 1) for x in r["centroid"]])
               for r in table], open(os.path.join(out, "parts.json"), "w"), indent=0)

    # ---------------- connections ----------------
    nrb_nodes = [len(d["node_sets"].get(ns, [])) for (_, ns) in d["nrb"]]
    rep["connections"] = dict(
        constrained_spotweld=len(d["spotwelds"]),
        spotweld_beams=sum(pm[p]["n_beam"] for p in veh if p in pm and secs.get(parts[p]["secid"], {}).get("elform") == 9),
        tied_contacts=[k for k, _ in d["contacts"]],
        nodal_rigid_bodies=len(d["nrb"]),
        nodal_rigid_body_nodes=dict(min=min(nrb_nodes), median=sorted(nrb_nodes)[len(nrb_nodes) // 2],
                                    max=max(nrb_nodes), total=sum(nrb_nodes)),
        extra_nodes_sets=len(d["extra_nodes"]), rigid_body_merges=len(d["rigid_bodies"]),
        joints=d["joints"], discrete_elements=len(d["discrete"]),
        airbag_simple_models=kc["vehicle"].get("*AIRBAG_SIMPLE_AIRBAG_MODEL_ID", 0),
    )

    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("counts", "mass_layers_kg", "mass_properties", "sections",
                                          "cumulative_parts_for_element_mass", "connections", "material_models",
                                          "steel_grades_by_yield_MPa")}, indent=1, default=str))
    print("unclassified", rep["unclassified"])


if __name__ == "__main__":
    main()
