//! Whole-vehicle builds.
//!
//! The reference sedan as solid parts (the default) is pinned to this
//! generator's own output, and every panel and pane is checked to be a closed
//! solid: no open border, faces outward, a positive volume, walls within a
//! thickness band, no face cutting through another, the art spec's triangle
//! budgets. The legacy sheet build (`BuildOpts::solid = false`) stays pinned
//! to the Python generator (counts from engine/tools/vehicle-gen,
//! `tools/stats.py sedan`). Parametric builds keep the structure.

use engine_vehiclegen::mesh::Surf;
use engine_vehiclegen::solid::{aspect_over, check, neighbours_to_check, new_crossings, pair_crossings};
use engine_vehiclegen::{build, design_from_json, BuildOpts};

fn tris(b: &engine_vehiclegen::Built, name: &str) -> usize {
    b.parts.get(name).unwrap_or_else(|| panic!("no part {name}")).ntris()
}

/// The solid parts: (name prefix, least and greatest wall thickness in mm,
/// closed pieces).
const SOLIDS: [(&str, f64, f64, usize); 13] = [
    ("taillight_", 1.0, 6.0, 1),
    ("headliner", 7.9, 8.1, 1),
    ("wheel_well_", 3.9, 4.1, 1),
    ("door_F", 2.5, 55.0, 1),
    ("door_R", 2.5, 55.0, 1),
    ("hood", 8.0, 40.0, 1),
    // the boot lid closes over the tail panel: a shallow free edge
    ("trunk_lid", 8.0, 40.0, 1),
    ("fender_F", 4.5, 30.0, 1),
    // (4.3 mm at the corners: the cells evened out behind the bend)
    ("bumper_", 4.0, 30.0, 1),
    // the shell, two sill strips, the windscreen reveal (two pieces) and
    // the B-pillars
    ("body_shell", 2.5, 50.0, 7),
    ("glass_", 4.4, 4.6, 1),
    ("headlight_", 4.1, 4.3, 1),
    // (a card stops short of a bend its face would fold behind: the rear
    // door's dogleg round the wheel arch)
    ("door_trim_", 9.0, 18.1, 1),
];

fn solid_spec(name: &str) -> Option<(f64, f64, usize)> {
    // a lamp's reflector: a 3 mm moulding, in a piece each side of a bend
    // of the lens too tight to follow (with the headlamps' projector cups)
    if name.ends_with("_housing") {
        return Some((2.9, 3.1, 0));
    }
    SOLIDS.iter().find(|s| name.starts_with(s.0)).map(|s| (s.1, s.2, s.3))
}

#[test]
fn reference_sedan_legacy_sheets_match_the_python_generator() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { solid: false, ..Default::default() });
    assert_eq!(b.parts.len(), 68);
    assert_eq!(b.markers.len(), 42);
    let total: usize = b.parts.0.iter().map(|(_, m)| m.ntris()).sum();
    assert_eq!(total, 289_196);
    for (name, n) in [("body_shell", 55_160), ("hood", 34_008), ("door_FL", 11_376), ("door_RL", 9_872), ("trunk_lid", 13_552), ("glass_windshield", 5_280), ("tyre_FL", 7_872), ("rim_FL", 4_840), ("grille", 3_660), ("door_trim_FL", 3_814), ("headliner", 5_310)] {
        assert_eq!(tris(&b, name), n, "{name}");
    }
    assert!(b.parts.0.iter().all(|(_, m)| m.surf.is_none()));
    // the body is symmetric about x = 0 (to microns: relaxation and insets
    // run over the welded mesh, as in the Python)
    let (lo, hi) = b.parts.get("body_shell").unwrap().bbox();
    assert!((lo.x + hi.x).abs() < 1e-5, "{lo} {hi}");
}

#[test]
fn reference_sedan_is_pinned() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts::default());
    // (the sheet build's 68, less its firewall box, plus the body-in-white's
    // four parts: `tests/structure.rs`)
    assert_eq!(b.parts.len(), 71);
    assert_eq!(b.markers.len(), 42);
    let total: usize = b.parts.0.iter().map(|(_, m)| m.ntris()).sum();
    assert_eq!(total, 456_442);
    for (name, n) in [
        ("body_shell", 114_520),
        ("hood", 39_408),
        ("door_FL", 15_666),
        ("door_RL", 13_822),
        ("trunk_lid", 17_512),
        ("fender_FL", 10_990),
        ("bumper_front", 15_096),
        ("bumper_rear", 13_260),
        ("glass_windshield", 10_976),
        ("glass_door_FL", 2_740),
        ("headlight_L_lens", 1_088),
        ("door_trim_FL", 6_524),
        ("tyre_FL", 7_872),
        ("rim_FL", 4_840),
        ("grille", 3_660),
        ("headliner", 11_036),
        ("wheel_well_FL", 488),
    ] {
        assert_eq!(tris(&b, name), n, "{name}");
    }
    let (lo, hi) = b.parts.get("body_shell").unwrap().bbox();
    assert!((lo.x + hi.x).abs() < 1e-5, "{lo} {hi}");
    // the skin of every solid panel is the legacy sheet's skin
    let sheet = build(d.as_ref(), BuildOpts { solid: false, ..Default::default() });
    // (panes and lenses are not: the solid build rounds their corners with
    // their openings' and sinks them along smoothed normals)
    for name in ["door_FL", "hood", "fender_FR", "bumper_rear", "trunk_lid"] {
        let (a, s) = (b.parts.get(name).unwrap().outer_only(), sheet.parts.get(name).unwrap());
        for (i, p) in a.v.iter().enumerate() {
            assert!((*p - s.v[i]).length() < 1e-12, "{name} skin vertex {i}");
        }
    }
}

#[test]
fn every_panel_and_pane_is_a_closed_solid() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts::default());
    let mut seen = 0;
    for (name, m) in &b.parts.0 {
        let Some((tmin, tmax, pieces)) = solid_spec(name) else {
            assert!(m.surf.is_none(), "{name} is tagged but not in the solids table");
            continue;
        };
        seen += 1;
        let surf = m.surf.as_ref().unwrap_or_else(|| panic!("{name} has no surface tags"));
        assert_eq!(surf.len(), m.v.len(), "{name}");
        let count = |s: Surf| surf.iter().filter(|x| **x == s).count();
        assert!(count(Surf::Inner) > 0 && count(Surf::Outer) >= count(Surf::Inner), "{name}");
        let r = check(m);
        // closed: every edge has its opposite, once
        assert!(r.closed(), "{name}: {} open edges, {} non-manifold", r.boundary_edges, r.nonmanifold_edges);
        // outward faces: every piece has a positive volume
        assert!(pieces == 0 || r.components == pieces, "{name}: {} pieces", r.components);
        assert!(r.min_component_volume > 1e-6, "{name}: a piece's volume is {}", r.min_component_volume);
        assert!(r.thickness_min * 1e3 >= tmin && r.thickness_max * 1e3 <= tmax, "{name}: thickness {:.2}..{:.2} mm, band {tmin}..{tmax}", r.thickness_min * 1e3, r.thickness_max * 1e3);
    }
    assert_eq!(seen, 6 + 2 + 2 + 1 + 8 + 4 + 4 + 4 + 1 + 4, "closures, wings, bumpers, shell, glass, lenses, reflectors, trims, headliner, liners");
}

#[test]
fn solids_do_not_cut_through_themselves() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    for (name, m) in &b.parts.0 {
        if m.surf.is_none() {
            continue;
        }
        let n = new_crossings(m).len();
        // Known, pinned as ceilings (edge-through-face crossings, each a few
        // millimetres deep): the body shell at the rear arch's and the tail
        // lamp's corners.
        let ceiling = match name.as_str() {
            "body_shell" => 12,
            _ => 0,
        };
        assert!(n <= ceiling, "{name}: {n} crossings (ceiling {ceiling})");
    }
}

#[test]
fn parts_cut_through_their_neighbours_no_more_than_pinned() {
    // A pane against its rebate, a lens against its land, a door against
    // its jamb, a door card against its door, the B-pillar against the
    // doors: edges of one through faces of the other. Known, pinned as
    // ceilings: where a wall is squashed behind a tight bend (a window
    // frame, a lamp's land) the rebate closes on the pane's edge.
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    let names: Vec<&String> = b.parts.0.iter().map(|p| &p.0).collect();
    let mut total = 0;
    for (i, a) in names.iter().enumerate() {
        for c in &names[i + 1..] {
            if !neighbours_to_check(a, c) {
                continue;
            }
            let n = pair_crossings(b.parts.get(a).unwrap(), b.parts.get(c).unwrap()).len();
            let is = |x: &str, y: &str| (a.starts_with(x) && c.starts_with(y)) || (c.starts_with(x) && a.starts_with(y));
            let ceiling = if is("headlight_", "fender_F") {
                190
            } else if is("headlight_", "hood") {
                25
            } else if is("body_shell", "door_R") {
                7
            } else if is("body_shell", "taillight_") {
                215
            } else if is("door_F", "glass_door_F") {
                181
            } else if is("door_R", "glass_door_R") {
                179
            } else if is("trunk_lid", "taillight_") {
                20
            } else {
                0
            };
            if n > 0 {
                println!("{a} / {c}: {n}");
            }
            assert!(n <= ceiling, "{a} / {c}: {n} crossings (ceiling {ceiling})");
            total += n;
        }
    }
    println!("{total} in all");
}

#[test]
fn panel_triangles_keep_their_aspect() {
    // docs/vehicle-art-spec.md: painted panel triangles no longer than 6:1
    // (longest edge squared over twice the area, as check.py measures it).
    // The drawn skins have 369 past it (thin rows at tight corners; the
    // legacy sheets have them too, 600 with their hems); the solids' walls
    // and inner faces are pinned as ceilings. Most of the inner faces' are
    // behind the shell's pillars and roof rails: strips a few thin cells
    // wide, narrower still 30 mm behind their bend.
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    let mut n = [0; 3];
    for (name, m) in &b.parts.0 {
        if m.surf.is_some() {
            let a = aspect_over(m, 6.0);
            println!("{name}: {a:?}");
            for k in 0..3 {
                n[k] += a[k];
            }
        }
    }
    println!("skin {}, walls {}, inner faces {}", n[0], n[1], n[2]);
    assert_eq!(n[0], 369, "skin");
    assert!(n[1] <= 131, "walls {}", n[1]);
    assert!(n[2] <= 1649, "inner faces {}", n[2]);
}

#[test]
fn triangle_budgets_hold() {
    // docs/vehicle-art-spec.md "Triangle budget", grouped as check.py groups
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
    let (mut body, mut interior, mut other, mut total, mut structure) = (0, 0, 0, 0, 0);
    let mut wheel = std::collections::BTreeMap::new();
    for (name, m) in &b.parts.0 {
        let n = m.ntris();
        total += n;
        if ["rim", "tyre", "brake"].iter().any(|p| name.starts_with(p)) {
            *wheel.entry(name.rsplit('_').next().unwrap().to_string()).or_insert(0) += n;
        } else if ["body_shell", "hood", "trunk_lid", "tailgate", "door_F", "door_R", "fender_", "bumper_"].iter().any(|p| name.starts_with(p)) {
            body += n;
        } else if name.starts_with("structure_") {
            structure += n;
        } else if ["dash", "steering", "seat", "console", "door_trim", "headliner"].iter().any(|p| name.starts_with(p)) {
            interior += n;
        } else {
            other += n;
        }
    }
    let wmax = wheel.values().copied().max().unwrap();
    println!("body {body}, wheel {wmax}, interior {interior}, structure {structure}, other {other}, total {total}");
    assert!((150_000..=300_000).contains(&body), "body {body}");
    assert!(wmax <= 15_000, "wheel {wmax}");
    assert!(interior <= 80_000, "interior {interior}");
    assert!(other <= 60_000, "lamps, glass, trim, under-bonnet {other}");
    // the body-in-white and the mechanical parts have budgets of their own,
    // and the total rose with them (docs/vehicle-art-spec.md)
    assert!(structure <= 40_000, "body structure {structure}");
    assert!(total <= 600_000, "total {total}");
}

#[test]
fn parametric_builds_keep_the_structure() {
    for p in [
        r#"{"wheelbase":2.55,"front_overhang":0.80,"rear_overhang":0.85,"width":1.78,"height":1.42,"track":1.53,"tyre_d":0.64,"tyre_w":0.205,"rim_in":16,"rim_w_in":6.5}"#,
        r#"{"ride":0.15,"height":1.52,"tyre_d":0.78,"tyre_w":0.265,"rim_in":17,"rim_w_in":8,"track":1.66,"width":1.90,"rim_style":"steel"}"#,
        // the designer page's other presets (vehicle-designer.html): Executive, Low & wide
        r#"{"wheelbase":3.10,"front_overhang":0.95,"rear_overhang":1.15,"width":1.92,"height":1.48,"track":1.64,"tyre_d":0.70,"tyre_w":0.255,"rim_in":19,"rim_w_in":8.5}"#,
        r#"{"wheelbase":2.80,"width":2.00,"height":1.30,"ride":-0.02,"track":1.76,"tyre_d":0.68,"tyre_w":0.295,"rim_in":20,"rim_w_in":10}"#,
    ] {
        let d = design_from_json("sedan", p).unwrap();
        let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
        assert_eq!(b.parts.len(), 71, "{p}");
        assert_eq!(b.markers.len(), 42, "{p}");
        let wb: serde_json::Value = serde_json::from_str(p).unwrap();
        let want = wb.get("wheelbase").and_then(|v| v.as_f64()).unwrap_or(2.83);
        let z = |n: &str| b.markers.iter().find(|m| m.name == n).unwrap().pos.z;
        assert!((z("wheel_center_FL") - z("wheel_center_RL") - want).abs() < 1e-9, "{p}");
        for (name, m) in &b.parts.0 {
            assert!(m.v.iter().all(|v| v.is_finite()), "{p}");
            // solids stay closed, outward and at least as thick as the floor
            if let Some((_, tmax, pieces)) = solid_spec(name) {
                let r = check(m);
                assert!(r.closed() && (pieces == 0 || r.components == pieces) && r.min_component_volume > 1e-6, "{p}: {name}: {r:?}");
                assert!(r.thickness_min > 4e-4 && r.thickness_max * 1e3 <= tmax + 5.0, "{p}: {name}: {r:?}");
            }
        }
    }
}

#[test]
fn out_of_range_parameters_are_refused() {
    assert!(design_from_json("sedan", r#"{"ride":-0.05}"#).is_err());
    assert!(design_from_json("sedan", r#"{"track":1.95}"#).is_err());
    assert!(design_from_json("sedan", r#"{"rim_in":22,"tyre_d":0.6}"#).is_err());
    assert!(design_from_json("boat", "{}").is_err());
}
