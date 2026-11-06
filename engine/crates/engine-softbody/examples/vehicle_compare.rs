//! Every car of the comparison manifest through the same scenes on the CPU
//! (`engine_softbody::vehicle_compare`), written as a markdown table, a JSON
//! file and a force-time chart.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_compare -- \
//!     [--cars sedan1,etkk,gen2] [--manifest path.json] [--out engine/docs/measurements] [--quick]
//! ```
//!
//! The manifest (`engine/crates/engine-webgpu/vehicle-compare-cars.json`) is
//! shared with the side-by-side recorder (`crash-lab-compare.mjs`): a new car
//! is one entry. JBeam cars load from the crash lab's gitignored `__mods/`
//! (`RUBBLE_COMPARE_MODS` overrides); an absent one is skipped.
use engine_softbody::vehicle_compare::{available, default_manifest, run_car, Car, Manifest};
use serde_json::Value;
use std::fmt::Write as _;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned());
    let quick = a.iter().any(|x| x == "--quick");
    let manifest = opt("--manifest").map(PathBuf::from).unwrap_or_else(default_manifest);
    let out = opt("--out").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/measurements"));
    let m: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?).map_err(|e| e.to_string())?;
    let only: Option<Vec<String>> = opt("--cars").map(|c| c.split(',').map(str::to_string).collect());
    let t0 = std::time::Instant::now();
    let mut results = Vec::new();
    let mut skipped = Vec::new();
    for spec in m.cars.iter().filter(|c| only.as_ref().is_none_or(|o| o.contains(&c.name))) {
        if !available(spec) {
            eprintln!("- {}: files absent, skipped", spec.name);
            skipped.push(spec.name.clone());
            continue;
        }
        let t = std::time::Instant::now();
        let car = Car::load(spec)?;
        eprintln!("{}: loaded in {:.1} s ({} nodes)", spec.name, t.elapsed().as_secs_f32(), car.template.nodes.len());
        let mut r = run_car(&car, quick);
        engine_softbody::vehicle_compare::name_nodes(&mut r, &car);
        eprintln!("{}: scenes in {:.1} s", spec.name, t.elapsed().as_secs_f32());
        results.push(r);
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let stem = if quick { "vehicle-compare-quick" } else { "vehicle-compare" };
    let doc = serde_json::json!({ "quick": quick, "skipped": skipped, "cars": results });
    std::fs::write(out.join(format!("{stem}.json")), serde_json::to_string_pretty(&doc).unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(out.join(format!("{stem}.md")), markdown(&results, quick)).map_err(|e| e.to_string())?;
    std::fs::write(out.join(format!("{stem}-force.svg")), force_svg(&results)).map_err(|e| e.to_string())?;
    eprintln!("→ {}/{stem}.{{md,json}}, {stem}-force.svg ({:.0} s)", out.display(), t0.elapsed().as_secs_f32());
    Ok(())
}

fn g<'a>(v: &'a Value, path: &str) -> &'a Value {
    path.split('.').fold(v, |v, k| match k.parse::<usize>() {
        Ok(i) => &v[i],
        Err(_) => &v[k],
    })
}

fn fmt(v: &Value) -> String {
    match v {
        Value::Null => "–".into(),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            if f.fract() == 0.0 && f.abs() < 1e7 {
                format!("{f:.0}")
            } else if f.abs() >= 1e5 {
                format!("{f:.3e}")
            } else {
                format!("{}", (f * 1000.0).round() / 1000.0)
            }
        }
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Array(a) => a.iter().map(fmt).collect::<Vec<_>>().join(", "),
        Value::Object(_) => v.to_string(),
    }
}

/// (section, row label, JSON path, reference band or note).
const ROWS: &[(&str, &str, &str, &str)] = &[
    ("Census", "nodes (tyres excluded)", "census.nodes", ""),
    ("Census", "mass (kg)", "census.mass_kg", "1,250–1,550 (the NCAP sedans)"),
    ("Census", "centre of mass height (m)", "census.com_height_m", ""),
    ("Census", "front axle's share of the weight", "census.front_axle_share", ""),
    ("Census", "node mass p50 (kg)", "census.node_mass_p50", ""),
    ("Census", "spring ends per node p50 / mean", "census.ends_per_node_p50|census.ends_per_node_mean", ""),
    ("Census", "beam length p50 (m)", "census.len_p50", ""),
    ("Census", "k·L p10 / p50 / p90 (N)", "census.kl_p10|census.kl_p50|census.kl_p90", "k·L ∝ m·L/n at the bound"),
    ("Census", "ε_y p50 (deform / k·L)", "census.eps_y_p50", ""),
    ("Census", "nodes at the step's bound (share)", "census.at_bound_share", ""),
    ("Census", "collision triangles; render triangles", "census.skin_triangles|census.visual_triangles", ""),
    ("Rest 10 s, raycast wheels", "broken / bent beams", "rest.broken_beams|rest.bent_beams", "0 / 0"),
    ("Rest 10 s, raycast wheels", "first break (s)", "rest.first_break_s", "none"),
    ("Rest 10 s, raycast wheels", "creep over the last 5 s (mm); sag (mm)", "rest.creep_mm|rest.sag_mm", ""),
    ("Rest 10 s, raycast wheels", "KE at 10 s (J); fastest node (m/s)", "rest.ke_end_j|rest.vmax_end", ""),
    ("Rest 10 s, raycast wheels", "page's .truck export: broken / bent", "rest_truck_export.broken_beams|rest_truck_export.bent_beams", "0 / 0"),
    ("Wall 56 km/h", "dynamic crush (mm)", "wall56.dynamic_crush_mm", "620–750"),
    ("Wall 56 km/h", "t_m (ms)", "wall56.t_m_ms", "63–76"),
    ("Wall 56 km/h", "peak (g, 100 Hz, rear cabin)", "wall56.peak_g", "34–47"),
    ("Wall 56 km/h", "residual crush max / avg (mm)", "wall56.residual_crush_max_mm|wall56.residual_crush_avg_mm", "380–580 / 261–512"),
    ("Wall 56 km/h", "Campbell b1 (1/s)", "wall56.campbell_b1", "22.9–39.2"),
    ("Wall 56 km/h", "springback (dynamic − residual max) / dynamic", "wall56.springback_ratio", "about 0.25 (0.16 of 0.65 m)"),
    ("Wall 56 km/h", "restitution", "wall56.restitution", "0.13–0.20"),
    ("Wall 56 km/h", "peak force (kN) at (ms)", "wall56.peak_force_kn|wall56.peak_force_ms", ""),
    ("Wall 56 km/h", "energy: start, beams dissipated, elastic end (kJ)", "wall56.energy.start_kj|wall56.energy.beam_dissipation_kj|wall56.energy.elastic_end_kj", ""),
    ("Wall 56 km/h", "ledger closure (share of start); largest rise (J)", "wall56.energy.closure_share|wall56.energy.max_rise_j", "≤ 1e-3; 0"),
    ("Wall 56 km/h", "ripple 0.5–0.6 s: worst part (mm)", "wall56.ripple_0_5s.worst_part|wall56.ripple_0_5s.worst_mm", "≤ 1 (Choice)"),
    ("Wall 56 km/h", "ripple 0.5–0.6 s: median part (mm)", "wall56.ripple_0_5s.median_part_mm", "≤ 1"),
    ("Wall 56 km/h", "ripple at 1 s: worst part (mm)", "wall56.ripple_end.worst_part|wall56.ripple_end.worst_mm", "≤ 1"),
    ("Wall 56 km/h", "intrusion max: feet / knees / chest (mm)", "wall56.intrusion.feet.max_mm|wall56.intrusion.knees.max_mm|wall56.intrusion.chest.max_mm", "firewall 1–67 mm rearward"),
    ("Wall 56 km/h", "penetration into the wall max: node / beam / triangle (mm)", "wall56.penetration.nodes.max_mm|wall56.penetration.beams.max_mm|wall56.penetration.triangles.max_mm", "0 (rigid)"),
    ("Wall 56 km/h", "penetration p99 in contact: node / beam / triangle (mm)", "wall56.penetration.nodes.p99_in_contact_mm|wall56.penetration.beams.p99_in_contact_mm|wall56.penetration.triangles.p99_in_contact_mm", ""),
    ("Wall 56 km/h", "broken beams; hardware broken; parts detached", "wall56.damage.broken_beams|wall56.damage.hardware_broken|wall56.damage.parts_detached", ""),
    ("Wall 56 km/h", "detached parts", "wall56.damage.parts_detached_names", ""),
    ("Wall 56 km/h", "mesh: stretch p99 / max", "wall56.visual.stretch_p99|wall56.visual.stretch_max", ""),
    ("Wall 56 km/h", "mesh: compression p01 / min", "wall56.visual.compress_p01|wall56.visual.compress_min", ""),
    ("Wall 56 km/h", "mesh: triangles > 1.5× / < 0.5×", "wall56.visual.tris_over_1_5|wall56.visual.tris_under_0_5", ""),
    ("Wall 56 km/h", "mesh: torn vertices (share); new flipped triangles", "wall56.visual.torn_fraction|wall56.visual.flipped_new", ""),
    ("Wall 56 km/h", "mesh: seams opened > 1 cm / > 5 cm (of pairs)", "wall56.visual.seam_open_1cm|wall56.visual.seam_open_5cm|wall56.visual.seam_pairs", ""),
    ("Wall 56 km/h", "mesh: seam gap p99 / max (mm)", "wall56.visual.seam_gap_p99_mm|wall56.visual.seam_gap_max_mm", ""),
    ("Wall 56 km/h", "mesh: dihedral change p99 / max (°); edges > 30° / > 60°", "wall56.visual.dihedral_p99_deg|wall56.visual.dihedral_max_deg|wall56.visual.dihedral_over_30|wall56.visual.dihedral_over_60", ""),
    ("Side pole 32 km/h, 75°", "residual / dynamic crush at the pole (mm)", "pole32.residual_crush_mm|pole32.dynamic_crush_mm", "297–445 residual"),
    ("Side pole 32 km/h, 75°", "intrusion max: hip / chest / head (mm)", "pole32.intrusion.hip.max_mm|pole32.intrusion.chest.max_mm|pole32.intrusion.head.max_mm", ""),
    ("Side pole 32 km/h, 75°", "penetration into the pole max: node / beam / triangle (mm)", "pole32.penetration.nodes.max_mm|pole32.penetration.beams.max_mm|pole32.penetration.triangles.max_mm", "0 (rigid)"),
    ("Side pole 32 km/h, 75°", "penetration p99 in contact: node / beam / triangle (mm)", "pole32.penetration.nodes.p99_in_contact_mm|pole32.penetration.beams.p99_in_contact_mm|pole32.penetration.triangles.p99_in_contact_mm", ""),
    ("Side pole 32 km/h, 75°", "energy rise (J); broken beams; hardware broken; detached", "pole32.energy_rise_j|pole32.damage.broken_beams|pole32.damage.hardware_broken|pole32.damage.parts_detached", "0; ; ; 0"),
    ("Side pole 32 km/h, 75°", "mesh: torn share; seams > 5 cm; dihedral p99 (°)", "pole32.visual.torn_fraction|pole32.visual.seam_open_5cm|pole32.visual.dihedral_p99_deg", ""),
    ("Pole sweep 32 km/h, 90°", "offsets; passed through", "pole_sweep32.offsets|pole_sweep32.passed_through", "0 through"),
    ("Pole sweep 32 km/h, 90°", "worst penetration: node / beam / triangle (mm)", "pole_sweep32.worst_node_mm|pole_sweep32.worst_beam_mm|pole_sweep32.worst_triangle_mm", "0 (rigid)"),
    ("MDB 214 (53.9 km/h, 27° crab)", "residual side crush (mm)", "mdb214.residual_crush_mm", "126–206 (four-door sedans)"),
    ("MDB 214 (53.9 km/h, 27° crab)", "intrusion max: hip / chest (mm)", "mdb214.intrusion.hip.max_mm|mdb214.intrusion.chest.max_mm", ""),
    ("MDB 214 (53.9 km/h, 27° crab)", "energy rise (J); broken; hardware; detached", "mdb214.energy_rise_j|mdb214.damage.broken_beams|mdb214.damage.hardware_broken|mdb214.damage.parts_detached", "0"),
    ("MDB 214 (53.9 km/h, 27° crab)", "mesh: torn share; seams > 5 cm; dihedral p99 (°)", "mdb214.visual.torn_fraction|mdb214.visual.seam_open_5cm|mdb214.visual.dihedral_p99_deg", ""),
    ("Roof FMVSS 216a", "peak within 127 mm / weight", "roof.ratio", "4.48–5.32 (≥ 3.0 rule)"),
    ("Door FMVSS 214 static", "door part", "door.door_part", ""),
    ("Door FMVSS 214 static", "initial / intermediate / peak (kN)", "door.initial_kn|door.intermediate_kn|door.peak_kn", "≥ 10.0 / 15.6 / min(2 CW, 31.1)"),
    ("Cost", "wall-clock s per simulated s: rest / wall / pole / MDB", "rest.wall_s_per_sim_s|wall56.wall_s_per_sim_s|pole32.wall_s_per_sim_s|mdb214.wall_s_per_sim_s", "CPU reference, one thread a scene"),
];

fn markdown(cars: &[Value], quick: bool) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Vehicle comparison{}\n", if quick { " (quick: shortened scenes)" } else { "" });
    let _ = writeln!(s, "Generated by `cargo run --release -p engine-softbody --example vehicle_compare`{}; the method is in `vehicle-compare.md`'s companion doc `engine/docs/etkk-audit.md`. Bands are from `crash-reference-targets.md` and `vehicle-test-strategy.md`.\n", if quick { " -- --quick" } else { "" });
    let mut section = "";
    for &(sec, label, path, band) in ROWS {
        if sec != section {
            section = sec;
            let _ = writeln!(s, "\n## {sec}\n");
            let _ = write!(s, "| | ");
            for c in cars {
                let _ = write!(s, "{} | ", c["name"].as_str().unwrap_or("?"));
            }
            let _ = writeln!(s, "reference |");
            let _ = writeln!(s, "|{}", "---|".repeat(cars.len() + 2));
        }
        let _ = write!(s, "| {label} | ");
        for c in cars {
            let cell: Vec<String> = path.split('|').map(|p| fmt(g(c, p))).collect();
            let _ = write!(s, "{} | ", cell.join(" / "));
        }
        let _ = writeln!(s, "{band} |");
    }
    // Rest: where the breaks are.
    let _ = writeln!(s, "\n## Rest: broken beams by part (top six)\n");
    for c in cars {
        for key in ["rest", "rest_truck_export"] {
            if c[key].is_object() {
                let _ = writeln!(s, "- {} ({}): {}", c["name"].as_str().unwrap_or("?"), c[key]["path"].as_str().unwrap_or(""), fmt(&c[key]["broken_by_part"]));
            }
        }
    }
    let _ = writeln!(s, "\n## Ripple: worst five parts (mm, wall 0.5–0.6 s)\n");
    for c in cars {
        let _ = writeln!(s, "- {}: {}", c["name"].as_str().unwrap_or("?"), fmt(&c["wall56"]["ripple_0_5s"]["top"]));
    }
    let _ = writeln!(s, "\n## Pole sweep: per offset (along the car from the rear, through?, reach mm, triangle penetration mm)\n");
    for c in cars {
        let runs = c["pole_sweep32"]["runs"].as_array().cloned().unwrap_or_default();
        let row: Vec<String> = runs
            .iter()
            .map(|r| format!("{}:{}{}/{}", fmt(&r["along"]), if r["through"].as_bool() == Some(true) { "THROUGH " } else { "" }, fmt(&r["reach_mm"]), fmt(&r["penetration"]["triangles"]["max_mm"])))
            .collect();
        let _ = writeln!(s, "- {}: {}", c["name"].as_str().unwrap_or("?"), row.join("; "));
    }
    let _ = writeln!(s, "\nThe wall's force-time curves (100 Hz, from contact): `vehicle-compare{}-force.svg`; every number, and the curves at 1 ms, in the JSON.", if quick { "-quick" } else { "" });
    s
}

fn force_svg(cars: &[Value]) -> String {
    let (w, h, l, b) = (720.0, 360.0, 60.0, 40.0);
    let colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"];
    let curves: Vec<(String, Vec<f64>)> = cars
        .iter()
        .map(|c| (c["name"].as_str().unwrap_or("?").to_string(), c["wall56"]["force_kn_1ms"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default()))
        .collect();
    let fmax = curves.iter().flat_map(|c| c.1.iter().copied()).fold(100.0, f64::max);
    let fmax = (fmax / 100.0).ceil() * 100.0;
    let x = |ms: f64| l + (w - l - 20.0) * ms / 150.0;
    let y = |kn: f64| h - b - (h - b - 20.0) * kn / fmax;
    let mut s = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" font-family=\"sans-serif\" font-size=\"12\"><rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>");
    for k in 0..=6 {
        let ms = 25.0 * k as f64;
        let _ = write!(s, "<line x1=\"{0}\" x2=\"{0}\" y1=\"20\" y2=\"{1}\" stroke=\"#eee\"/><text x=\"{0}\" y=\"{2}\" text-anchor=\"middle\">{ms:.0}</text>", x(ms), h - b, h - b + 16.0);
    }
    for k in 0..=4 {
        let kn = fmax * k as f64 / 4.0;
        let _ = write!(s, "<line x1=\"{l}\" x2=\"{0}\" y1=\"{1}\" y2=\"{1}\" stroke=\"#eee\"/><text x=\"{2}\" y=\"{3}\" text-anchor=\"end\">{kn:.0}</text>", w - 20.0, y(kn), l - 6.0, y(kn) + 4.0);
    }
    let _ = write!(s, "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\">ms after contact</text><text x=\"14\" y=\"{}\" transform=\"rotate(-90 14 {})\" text-anchor=\"middle\">wall force on the car (kN, 100 Hz)</text>", 0.5 * (l + w), h - 6.0, 0.5 * h, 0.5 * h);
    for (k, (name, c)) in curves.iter().enumerate() {
        let col = colours[k % colours.len()];
        let pts: Vec<String> = c.iter().enumerate().map(|(i, &f)| format!("{:.1},{:.1}", x(i as f64), y(f))).collect();
        let _ = write!(s, "<polyline fill=\"none\" stroke=\"{col}\" stroke-width=\"2\" points=\"{}\"/><text x=\"{}\" y=\"{}\" fill=\"{col}\">{name}</text>", pts.join(" "), w - 120.0, 36.0 + 16.0 * k as f64);
    }
    s.push_str("</svg>\n");
    s
}
