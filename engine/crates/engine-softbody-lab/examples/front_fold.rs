//! How the generated front folds in a crash, bay by bay and mode by mode (`engine/docs/beam-yield-law.md`
//! §5.3, `engine/docs/front-load-path.md` §4.1). CPU reference only.
//!
//! `front_fold -- [SCENARIO] [--params design=FILE] [--contact discrete|edge|continuous] [--run K]
//! [--after S] [--out DIR]` (default S11, edge, run 0, 0.15 s after the nominal impact, out
//! `out/rail-content/fold`). Needs a car whose front bays are crush cells (`front_rules.load_path`
//! with `crush_law`); every cell on the first body is found from its chords.
//!
//! Per 4 ms sample and per bay (a cell: two stations of four corner nodes each), written to
//! `DIR/<SCENARIO>-<contact>-<run>-bays.csv`:
//! - the span along the car (station centre to centre) and its chords' mean, top-pair and
//!   bottom-pair length, against spawn;
//! - each station's height (mean y) and its face's pitch against spawn (rad, + nose up);
//! - each mode's generalised force G_m = Φ_m · k(ℓ − r) (axial, strong bending, weak bending,
//!   warping), its capacity at that moment (the axial curve at its set, or its tension; the others'
//!   caps), the strong-axis moment G·lever, and each mode's plastic work so far (the split of
//!   `metrics::beams::CellPlastic`, exact while a mode flows one way between samples).
//!
//! Per station (`-stations.csv`): the force every other beam puts on its four corners (along the
//! car and vertical, N) and that force's moment about the station's centre in the car's vertical
//! plane (N·m): the load the subframe, the braces and the engine's stops bring into the rail there.
//!
//! The summary (`-summary.txt`, also printed): each bay's work by mode at the time the car stops
//! and at the end, its largest |strong moment| against its collapse moment, and its shortening.

use engine_softbody::crash_reference::{self as cr, metrics::beams, Contact, CpuRunner, Jitter, Runner};
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_crush::Cell;
use glam::Vec3;
use std::fmt::Write as _;

const MODES: [&str; 4] = ["axial", "strong", "weak", "warping"];

/// A bay: its cell, its stations' corner node indices (front, rear) and names.
struct Bay {
    name: String,
    cell: Cell,
    front: [usize; 4],
    rear: [usize; 4],
    rest: [f64; 4],
    set: f64,
    work: [f64; 4],
    at_stop: Option<[f64; 4]>,
    max_moment: (f64, f32),
}

/// A station's prefix (`sh_a4_`) and side from a corner's name (`sh_a4_uil` → (`sh_a4_`, `l`)).
fn station_of(n: &str) -> Option<(String, char)> {
    let rest = n.strip_prefix("sh_")?;
    let (g, tail) = rest.split_once('_')?;
    Some((format!("sh_{g}_"), tail.chars().last()?))
}

/// The bays of the first body: one per crush cell, the stations' corners in the cell's chord order
/// (the load path's: outboard top, outboard bottom, inboard top, inboard bottom).
fn bays(v: &ExplicitBeams, names: &[String]) -> Vec<Bay> {
    let mut out = Vec::new();
    for j in 0..v.beams.len() {
        let Some(c) = beams::CellPlastic::of(v, j) else { continue };
        let cell = c.cell;
        let mut front = [0usize; 4];
        let mut rear = [0usize; 4];
        for (i, &b) in cell.beams.iter().enumerate() {
            let bm = &v.beams[b as usize];
            let (a, bb) = (bm.a as usize, bm.b as usize);
            // the front end is the one further along the car (+z)
            if v.nodes[a].pos.z >= v.nodes[bb].pos.z {
                (front[i], rear[i]) = (a, bb);
            } else {
                (front[i], rear[i]) = (bb, a);
            }
        }
        let (fs, side) = station_of(&names[front[0]]).unwrap_or(("?".into(), '?'));
        let (rs, _) = station_of(&names[rear[0]]).unwrap_or(("?".into(), '?'));
        let name = format!("{}-{}{side}", fs.trim_start_matches("sh_").trim_end_matches('_'), rs.trim_start_matches("sh_").trim_end_matches('_'));
        out.push(Bay { name, cell, front, rear, rest: cell.beams.map(|b| f64::from(v.beams[b as usize].rest)), set: f64::from(cell.set), work: [0.0; 4], at_stop: None, max_moment: (0.0, 0.0) });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn mean(v: &ExplicitBeams, ids: &[usize]) -> Vec3 {
    ids.iter().map(|&i| v.nodes[i].pos).sum::<Vec3>() / ids.len() as f32
}

/// A station face's pitch in the car's vertical plane (rad, + its top ahead of its bottom: nose up
/// for a face that leans back): from the mean of its bottom corners to its top's.
fn pitch(v: &ExplicitBeams, c: &[usize; 4]) -> f32 {
    let d = mean(v, &[c[0], c[2]]) - mean(v, &[c[1], c[3]]);
    d.z.atan2(d.y)
}

/// The bay's modes now: generalised forces (tension positive, as the cell's flow) and capacities.
fn modes(v: &ExplicitBeams, b: &Bay) -> ([f64; 4], [f64; 4]) {
    let f: Vec<f64> = b.cell.beams.iter().map(|&j| {
        let bm = &v.beams[j as usize];
        let l = (v.nodes[bm.a as usize].pos - v.nodes[bm.b as usize].pos).length();
        f64::from(bm.k) * f64::from(l - bm.rest)
    }).collect();
    let g = b.cell.phi.map(|row| (0..4).map(|i| f64::from(row[i]) * f[i]).sum::<f64>());
    let live = beams::crush_of(v, b.cell.beams[0] as usize).and_then(|c| c.cell).map_or(b.set as f32, |c| c.set);
    let axial = if g[0] < 0.0 { f64::from(b.cell.axial.at(live)) } else { f64::from(b.cell.cap[0]) };
    (g, [axial, f64::from(b.cell.cap[1]), f64::from(b.cell.cap[2]), f64::from(b.cell.cap[3])])
}

/// The split of [`beams::CellPlastic::advance`] by mode.
fn advance(v: &ExplicitBeams, b: &mut Bay) {
    let r = b.cell.beams.map(|j| f64::from(v.beams[j as usize].rest));
    for m in 0..4 {
        let dpi: f64 = (0..4).map(|i| f64::from(b.cell.phi[m][i]) * (r[i] - b.rest[i])).sum();
        if m == 0 && dpi < 0.0 {
            b.work[0] += b.cell.axial.work(b.set, b.set - dpi);
            b.set -= dpi;
        } else {
            b.work[m] += f64::from(b.cell.cap[m]) * dpi.abs();
        }
    }
    b.rest = r;
}

/// The force every beam not inside the side's stations puts on the station's corners, and its
/// moment about the station's centre in the vertical plane along the car (N, N, N·m).
fn station_load(v: &ExplicitBeams, corners: &[usize; 4], lane: &[usize]) -> (f64, f64, f64) {
    let c = mean(v, corners);
    let (mut fz, mut fy, mut m) = (0.0f64, 0.0f64, 0.0f64);
    for bm in v.beams.iter().filter(|b| !b.broken) {
        let (a, b) = (bm.a as usize, bm.b as usize);
        let (on, other) = if corners.contains(&a) { (a, b) } else if corners.contains(&b) { (b, a) } else { continue };
        if lane.contains(&other) {
            continue;
        }
        let d = v.nodes[on].pos - v.nodes[other].pos;
        let l = d.length().max(1e-9);
        // (positive force = compression: the beam pushes the corner away from its other end)
        let f = d / l * bm.force;
        let r = v.nodes[on].pos - c;
        fz += f64::from(f.z);
        fy += f64::from(f.y);
        m += f64::from(r.z * f.y - r.y * f.z);
    }
    (fz, fy, m)
}

/// [`station_load`] by where it comes from: the other node's part (its name's first word, or the
/// shell's group, `sh_a7`), each (along the car, across, up) in N.
fn station_sources(v: &ExplicitBeams, corners: &[usize; 4], lane: &[usize], names: &[String]) -> std::collections::BTreeMap<String, [f64; 3]> {
    let mut out = std::collections::BTreeMap::new();
    for bm in v.beams.iter().filter(|b| !b.broken) {
        let (a, b) = (bm.a as usize, bm.b as usize);
        let (on, other) = if corners.contains(&a) { (a, b) } else if corners.contains(&b) { (b, a) } else { continue };
        if lane.contains(&other) {
            continue;
        }
        let n = names.get(other).map_or("wheel", String::as_str);
        let mut w = n.split('_');
        let first = w.next().unwrap_or("");
        let part = if first == "sh" { format!("sh_{}", w.next().unwrap_or("")) } else { first.to_string() };
        let d = v.nodes[on].pos - v.nodes[other].pos;
        let f = d / d.length().max(1e-9) * bm.force;
        let e = out.entry(part).or_insert([0.0; 3]);
        e[0] += f64::from(f.z);
        e[1] += f64::from(f.x);
        e[2] += f64::from(f.y);
    }
    out
}

struct Args {
    scene: String,
    contact: Contact,
    run: usize,
    after: f32,
    out: String,
    params: Vec<(String, String)>,
}

fn args() -> Result<Args, String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    Ok(Args {
        scene: a.iter().take_while(|x| !x.starts_with("--")).next().cloned().unwrap_or_else(|| "S11".into()),
        contact: flag("--contact").map_or(Some(Contact::Edge), |c| Contact::from_name(&c)).ok_or("--contact")?,
        run: flag("--run").map_or(Ok(0), |x| x.parse().map_err(|_| "--run"))?,
        after: flag("--after").map_or(Ok(0.15), |x| x.parse().map_err(|_| "--after"))?,
        out: flag("--out").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/rail-content/fold").into()),
        params: flag("--params").map_or(Ok(Vec::new()), |v| engine_softbody_authoring::crash_templates::params_from_list(&v))?,
    })
}

/// Every station of the side the bays use, as corner sets (front to rear by name).
fn stations(bays: &[Bay], names: &[String]) -> Vec<(String, [usize; 4])> {
    let mut st: Vec<(String, [usize; 4])> = Vec::new();
    for b in bays {
        for c in [b.front, b.rear] {
            let (p, s) = station_of(&names[c[0]]).unwrap_or(("?".into(), '?'));
            let n = format!("{}{s}", p.trim_start_matches("sh_").trim_end_matches('_'));
            if !st.iter().any(|x| x.0 == n) {
                st.push((n, c));
            }
        }
    }
    st.sort_by(|a, b| a.0.cmp(&b.0));
    st
}

fn main() -> Result<(), String> {
    let a = args()?;
    let s = cr::get(&a.scene).ok_or(format!("no scenario {}", a.scene))?;
    let templates = engine_softbody_authoring::crash_templates::templates_for_with(&s, &a.params)?;
    let built = s.build(&templates, Jitter::run(a.run), a.contact)?;
    let design = &built.layout.bodies[0].design;
    let names: Vec<String> = templates[design.as_str()].desc.nodes.iter().map(|n| n.name.clone()).collect();
    let mut r = CpuRunner::new(&built);
    let mut bays = bays(&r.0.vehicles[0], &names);
    if bays.is_empty() {
        return Err("no crush cells on the first body (front_rules load_path + crush_law)".into());
    }
    let st = stations(&bays, &names);
    let lane: Vec<usize> = st.iter().flat_map(|x| x.1).collect();
    let v0 = r.0.vehicles[0].clone();
    let span0: Vec<f32> = bays.iter().map(|b| mean(&v0, &b.front).z - mean(&v0, &b.rear).z).collect();
    let pitch0: Vec<f32> = st.iter().map(|x| pitch(&v0, &x.1)).collect();
    let y0: Vec<f32> = st.iter().map(|x| mean(&v0, &x.1).y).collect();
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let tag = format!("{}/{}-{}-{}", a.out, s.id, a.contact.name(), a.run);
    let mut bay_csv = String::from("t_ms,bay,span_m,span0_m,chord_mean_m,chord0_mean_m,chord_top_m,chord_bottom_m,front_dy_m,rear_dy_m,front_pitch_rad,rear_pitch_rad");
    for m in MODES {
        write!(bay_csv, ",G_{m}_N,cap_{m}_N,W_{m}_J").ok();
    }
    bay_csv.push_str(",M_strong_Nm,M_strong_cap_Nm\n");
    let mut st_csv = String::from("t_ms,station,x_m,y_m,z_m,dy_m,pitch_rad,F_along_N,F_up_N,M_Nm\n");
    let mut src_csv = String::from("t_ms,station,part,F_along_N,F_across_N,F_up_N\n");
    let n = ((built.layout.impact_s + a.after) / (cr::DT * cr::SAMPLE_STEPS as f32)).ceil() as usize;
    let com_vz = |v: &ExplicitBeams| {
        let (p, m) = v.nodes.iter().fold((0.0f64, 0.0f64), |(p, m), n| (p + f64::from(n.mass) * f64::from(n.vel.z), m + f64::from(n.mass)));
        p / m
    };
    let mut stop_ms = None;
    for k in 0..=n {
        let t = k as f32 * cr::DT * cr::SAMPLE_STEPS as f32;
        let ms = (t - built.layout.impact_s) * 1e3;
        let v = &r.0.vehicles[0];
        for (bi, b) in bays.iter_mut().enumerate() {
            advance(v, b);
            let (g, cap) = modes(v, b);
            let chord = |i: usize| (v.nodes[b.front[i]].pos - v.nodes[b.rear[i]].pos).length();
            let rest0 = |i: usize| v0.beams[b.cell.beams[i] as usize].rest0;
            let fi = st.iter().position(|x| x.1 == b.front).unwrap_or(0);
            let ri = st.iter().position(|x| x.1 == b.rear).unwrap_or(0);
            let mo = g[1] * f64::from(b.cell.lever[0]);
            if mo.abs() > b.max_moment.0 {
                b.max_moment = (mo.abs(), ms);
            }
            write!(
                bay_csv,
                "{ms:.1},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                b.name,
                mean(v, &b.front).z - mean(v, &b.rear).z,
                span0[bi],
                (0..4).map(chord).sum::<f32>() / 4.0,
                (0..4).map(rest0).sum::<f32>() / 4.0,
                0.5 * (chord(0) + chord(2)),
                0.5 * (chord(1) + chord(3)),
                mean(v, &b.front).y - y0[fi],
                mean(v, &b.rear).y - y0[ri],
                pitch(v, &b.front) - pitch0[fi],
                pitch(v, &b.rear) - pitch0[ri]
            )
            .ok();
            for m in 0..4 {
                write!(bay_csv, ",{:.1},{:.1},{:.2}", g[m], cap[m], b.work[m]).ok();
            }
            writeln!(bay_csv, ",{mo:.1},{:.1}", f64::from(b.cell.cap[1]) * f64::from(b.cell.lever[0])).ok();
        }
        for (si, (name, c)) in st.iter().enumerate() {
            let (fz, fy, m) = station_load(v, c, &lane);
            let p = mean(v, c);
            writeln!(st_csv, "{ms:.1},{name},{:.4},{:.4},{:.4},{:.4},{:.4},{fz:.1},{fy:.1},{m:.1}", p.x, p.y, p.z, p.y - y0[si], pitch(v, c) - pitch0[si]).ok();
            for (part, f) in station_sources(v, c, &lane, &names) {
                writeln!(src_csv, "{ms:.1},{name},{part},{:.1},{:.1},{:.1}", f[0], f[1], f[2]).ok();
            }
        }
        if stop_ms.is_none() && t > built.layout.impact_s && com_vz(v) <= 0.0 {
            stop_ms = Some(ms);
            for b in bays.iter_mut() {
                b.at_stop = Some(b.work);
            }
        }
        if k < n {
            r.frame(cr::SAMPLE_STEPS)?;
        }
    }
    std::fs::write(format!("{tag}-bays.csv"), bay_csv).map_err(|e| e.to_string())?;
    std::fs::write(format!("{tag}-stations.csv"), st_csv).map_err(|e| e.to_string())?;
    std::fs::write(format!("{tag}-sources.csv"), src_csv).map_err(|e| e.to_string())?;
    let summary = summary(&a, &bays, &span0, &r.0.vehicles[0], stop_ms) + &budget_report(&bays, &v0, &names);
    print!("{summary}");
    std::fs::write(format!("{tag}-summary.txt"), summary).map_err(|e| e.to_string())?;
    Ok(())
}

fn summary(a: &Args, bays: &[Bay], span0: &[f32], v: &ExplicitBeams, stop_ms: Option<f32>) -> String {
    let mut o = format!("{} {} run {}: the car stops at {:?} ms after the nominal impact\n", a.scene, a.contact.name(), a.run, stop_ms);
    o.push_str("| bay | work at the stop: axial / strong / weak / warping (kJ) | at the end (kJ) | bending share | largest strong moment / collapse (kN·m) at (ms) | span shortening / chords' mean shortening at the end (m) |\n|---|---|---|---|---|---|\n");
    for (bi, b) in bays.iter().enumerate() {
        let w = b.at_stop.unwrap_or(b.work);
        let tot: f64 = b.work.iter().sum();
        let cap = f64::from(b.cell.cap[1]) * f64::from(b.cell.lever[0]);
        let span = mean(v, &b.front).z - mean(v, &b.rear).z;
        let chords: f32 = (0..4).map(|i| (v.nodes[b.front[i]].pos - v.nodes[b.rear[i]].pos).length()).sum::<f32>() / 4.0;
        let chords0: f32 = b.cell.beams.iter().map(|&j| v.beams[j as usize].rest0).sum::<f32>() / 4.0;
        writeln!(
            o,
            "| {} | {:.2} / {:.2} / {:.2} / {:.2} | {:.2} / {:.2} / {:.2} / {:.2} | {:.2} | {:.2} / {:.2} at {:.0} | {:.3} / {:.3} |",
            b.name,
            w[0] / 1e3,
            w[1] / 1e3,
            w[2] / 1e3,
            w[3] / 1e3,
            b.work[0] / 1e3,
            b.work[1] / 1e3,
            b.work[2] / 1e3,
            b.work[3] / 1e3,
            if tot > 0.0 { (b.work[1] + b.work[2]) / tot } else { 0.0 },
            b.max_moment.0 / 1e3,
            cap / 1e3,
            b.max_moment.1,
            span0[bi] - span,
            chords0 - chords
        )
        .ok();
    }
    o.push_str(&inner_beams(bays, v));
    o
}

/// Each left station corner's mass and springs (the stiffness budget the cell's chords share):
/// Σk/m and the chords', diagonals', faces' and others' shares of Σk, and the cells' elastic
/// stiffnesses against their strengths: axial 4k, bending k·h² and k·w² (N·m/rad) with the
/// rotation at which the collapse moment is reached elastically.
fn budget_report(bays: &[Bay], v: &ExplicitBeams, names: &[String]) -> String {
    let mut o = String::from("\nstation corners (left): mass (kg), Σk/m (1/s²), share of Σk: chords / diagonals / own faces / other beams\n");
    let mut seen = Vec::new();
    for b in bays.iter().filter(|b| b.name.ends_with('l')) {
        let cells: Vec<u32> = bays.iter().flat_map(|x| x.cell.beams).collect();
        let lane: Vec<usize> = bays.iter().flat_map(|x| x.front.iter().chain(&x.rear).copied()).collect();
        for &c in b.front.iter().chain(&b.rear) {
            if seen.contains(&c) {
                continue;
            }
            seen.push(c);
            let mut k = [0.0f64; 4];
            for (j, bm) in v.beams.iter().enumerate() {
                let (x, y) = (bm.a as usize, bm.b as usize);
                if x != c && y != c {
                    continue;
                }
                let other = if x == c { y } else { x };
                let same_station = station_of(&names[other]).map(|s| s.0) == station_of(&names[c]).map(|s| s.0);
                let slot = if cells.contains(&(j as u32)) { 0 } else if lane.contains(&other) && !same_station { 1 } else if same_station { 2 } else { 3 };
                k[slot] += f64::from(bm.k);
            }
            let tot: f64 = k.iter().sum();
            writeln!(o, "  {:10} {:.2} kg, {:.2e}; {:.2} / {:.2} / {:.2} / {:.2}", names[c], v.nodes[c].mass, tot / f64::from(v.nodes[c].mass), k[0] / tot, k[1] / tot, k[2] / tot, k[3] / tot).ok();
        }
    }
    o.push_str("\ncells: chord k (N/m), axial 4k, K_strong = k·h², K_weak = k·w² (kN·m/rad), collapse moments (kN·m), rotation to reach them elastically (rad), axial peak's elastic shortening (mm)\n");
    for b in bays.iter().filter(|b| b.name.ends_with('l')) {
        let k = f64::from(b.cell.k);
        let lev = b.cell.lever.map(f64::from);
        // (lever = Σφ·y over the chords: 2·(h/2)·½·2 = h for a rectangle, so k·lever² = k·h²)
        let (ks, kw) = (k * lev[0] * lev[0], k * lev[1] * lev[1]);
        let (ms, mw) = (f64::from(b.cell.cap[1]) * lev[0], f64::from(b.cell.cap[2]) * lev[1]);
        writeln!(o, "  {}: k {:.3e}, 4k {:.2e}, K {:.1} / {:.1}, M_c {:.2} / {:.2}, θ {:.2} / {:.2}, peak {:.0} mm", b.name, k, 4.0 * k, ks / 1e3, kw / 1e3, ms / 1e3, mw / 1e3, ms / ks, mw / kw, 1e3 * f64::from(b.cell.axial.peak) / (2.0 * k)).ok();
    }
    o
}

/// Every beam between a bay's corners at the end (chords, face diagonals, the end faces' ties):
/// its length and rest against spawn, thresholds, whether broken. Shows which member of the cell
/// let the bay change shape.
fn inner_beams(bays: &[Bay], v: &ExplicitBeams) -> String {
    let mut o = String::from("\nbeams inside each bay at the end: kind, length / rest / spawn rest (m), Tc / Tt / strength (kN), broken\n");
    for b in bays.iter().filter(|b| b.name.ends_with('l')) {
        let corners: Vec<usize> = b.front.iter().chain(&b.rear).copied().collect();
        writeln!(o, "{}:", b.name).ok();
        for (j, bm) in v.beams.iter().enumerate() {
            let (x, y) = (bm.a as usize, bm.b as usize);
            if !(corners.contains(&x) && corners.contains(&y)) {
                continue;
            }
            let kind = if b.cell.beams.contains(&(j as u32)) {
                "chord"
            } else if b.front.contains(&x) == b.front.contains(&y) {
                if b.front.contains(&x) { "front face" } else { "rear face" }
            } else {
                "diagonal"
            };
            let l = (v.nodes[x].pos - v.nodes[y].pos).length();
            writeln!(o, "  {kind:10} {l:.3} / {:.3} / {:.3}  {:.1} / {:.1} / {:.1}  {}", bm.rest, bm.rest0, bm.tc / 1e3, bm.tt / 1e3, bm.strength / 1e3, bm.broken).ok();
        }
    }
    o
}
