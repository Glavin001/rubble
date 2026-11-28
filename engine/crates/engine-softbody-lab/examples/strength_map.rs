//! The generator's mapping from a body's real steel to what the explicit 2 kHz solver
//! carries, as a table (`engine/docs/generated-vehicles.md`, "Real steel to 2 kHz").
//!
//!     cargo run --release -p engine-softbody-lab --example strength_map -- [DESIGN] [PRESET]
//!
//! DESIGN: `sedan` (default). PRESET: a crash character (`real_crash`, the default, or
//! `beamng_feel`). Prints markdown:
//!
//! * the boxed crush members (crash box, front rail, rear rail): the section's own axial
//!   stiffness E·A, squash load and mean crush force beside the lattice chords' k·L and
//!   yield, and the elastic strain each reaches before it yields;
//! * every zone of the shell lattice: its beams' spring, yield, the stretch at yield and
//!   the elastic energy the zone holds with every beam at its yield (what it can give back).
//!
//! Why the two differ: at a fixed 0.5 ms step a node's springs are bounded (Σk/m ≤ 8.5e6
//! s⁻², ω·h ≤ 1.85: `gen2::shell::NODE_BUDGET`), so a chord is tens of times softer than the
//! steel section it stands for. Its yield force is the section's real mean crush force; it
//! therefore stretches tens of times further before it yields, and stores that much more
//! elastic energy. That is the trade-off of an explicit node-beam solver at 2 kHz (BeamNG's
//! and Rigs of Rods' too). Nothing here changes the car: it reads the generator.

use engine_softbody_authoring::authoring::gen2::shell::{self, Class, Zone, NODE_BUDGET, OMEGA_H};
use engine_vehiclegen::physics::CrashFeel;
use engine_vehiclegen::structure::Section;
use engine_vehiclegen::thin_wall::{Material, ThinSection};
use std::collections::BTreeMap;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let design = args.first().map_or("sedan", String::as_str);
    let preset = args.get(1).map_or("real_crash", String::as_str);
    let feel = CrashFeel::preset(preset).unwrap_or_else(|| panic!("unknown crash character {preset}"));
    let d = engine_vehiclegen::design_from_json(design, "").expect("design");
    let st = d.structure().expect("a design on the body definitions");
    let lat = shell::lattice_with(&st, &feel);
    let pos: Vec<_> = st.nodes.iter().map(|n| n.pos).collect();
    let len = |b: &shell::ShellBeam| (pos[b.a] - pos[b.b]).length();

    println!("# {design}, crash character `{preset}`: real steel against the 2 kHz lattice\n");
    println!("Step 0.5 ms; node budget Σk/m ≤ {NODE_BUDGET:.1e} s⁻²; lattice ω·h ≤ {OMEGA_H}.\n");

    // ---- the boxed members: section against chords
    println!("## Boxed crush members: the section as defined against its lattice chords\n");
    println!("| member | section (mm), gauge, grade | real E·A (MN) | real squash σ_y·A (kN) | real mean crush P_m (kN) | real strain at P_m (%) | chords a section | lattice k·L, chords together (MN) | lattice yield, chords together (kN) | lattice strain at yield (%) | softer by | elastic energy to yield, per metre: real / lattice (J) |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (kind, groups, par) in [("crash_box", &["A1", "A2"][..], 2.0), ("front_rail", &["A2", "A4", "A5"][..], 1.0), ("rear_rail", &["E1"][..], 2.0)] {
        let Some(m) = st.members.iter().find(|m| m.kind == kind) else { continue };
        let (Section::Box { w, h } | Section::Hat { w, h }) = m.section else { continue };
        let mat: Material = m.grade.into();
        let p = ThinSection::rect_tube(w, h, m.gauge).properties().expect("section");
        let ea = mat.e * p.area;
        let squash = p.strengths(&mat, m.length.max(1e-3)).squash_n;
        let pm = m.crush_n();
        // the chords the builder gives this member's crush force to (left side)
        let chords: Vec<&shell::ShellBeam> = lat
            .beams
            .iter()
            .filter(|b| {
                let dir = (pos[b.a] - pos[b.b]).normalize();
                b.class == Class::Edge && dir.z.abs() > 0.707 && pos[b.a].x > 0.0 && pos[b.b].x > 0.0 && groups.contains(&st.nodes[b.a].group) && groups.contains(&st.nodes[b.b].group)
            })
            .collect();
        if chords.is_empty() {
            continue;
        }
        let mut kl: Vec<f64> = chords.iter().map(|b| b.k * len(b)).collect();
        let mut fy: Vec<f64> = chords.iter().map(|b| b.deform).collect();
        let (kl, fy) = (median(&mut kl) * par, median(&mut fy) * par);
        let (e_real, e_lat) = (pm / ea, fy / kl);
        println!(
            "| {} | {:.0} × {:.0}, {:.1}, {} | {:.1} | {:.0} | {:.1} | {:.3} | {:.0} | {:.2} | {:.1} | {:.2} | {:.0} × | {:.0} / {:.0} |",
            kind,
            w * 1e3,
            h * 1e3,
            m.gauge * 1e3,
            m.grade.name(),
            ea / 1e6,
            squash / 1e3,
            pm / 1e3,
            e_real * 100.0,
            par,
            kl / 1e6,
            fy / 1e3,
            e_lat * 100.0,
            ea / kl,
            0.5 * pm * e_real,
            0.5 * fy * e_lat
        );
    }

    // ---- every zone of the shell lattice
    println!("\n## The shell lattice by zone\n");
    println!("| zone | beams | spring k, median (MN/m) | length, median (m) | yield, median (kN) | stretch at yield, median (mm; % of length) | elastic energy with every beam at yield (kJ) |");
    println!("|---|---|---|---|---|---|---|");
    let mut zones: BTreeMap<String, Vec<&shell::ShellBeam>> = BTreeMap::new();
    for b in &lat.beams {
        zones.entry(format!("{:?}", b.zone)).or_default().push(b);
    }
    let order = [Zone::CrashBox, Zone::FrontRail, Zone::Tower, Zone::Cabin, Zone::Primary, Zone::Pillar, Zone::Bow, Zone::RearRail, Zone::Tail];
    let mut total = 0.0;
    for z in order {
        let Some(bs) = zones.get(&format!("{z:?}")) else { continue };
        let fin: Vec<&&shell::ShellBeam> = bs.iter().filter(|b| b.deform.is_finite()).collect();
        let mut k: Vec<f64> = bs.iter().map(|b| b.k).collect();
        let mut l: Vec<f64> = bs.iter().map(|b| len(b)).collect();
        let mut f: Vec<f64> = fin.iter().map(|b| b.deform).collect();
        let mut s: Vec<f64> = fin.iter().map(|b| b.deform / b.k).collect();
        let mut pc: Vec<f64> = fin.iter().map(|b| b.deform / b.k / len(b)).collect();
        let e: f64 = fin.iter().map(|b| 0.5 * b.deform * b.deform / b.k).sum();
        total += e;
        println!("| {z:?} | {} | {:.2} | {:.2} | {:.1} | {:.1}; {:.1} % | {:.2} |", bs.len(), median(&mut k) / 1e6, median(&mut l), median(&mut f) / 1e3, median(&mut s) * 1e3, median(&mut pc) * 100.0, e / 1e3);
    }
    println!("| all | {} | | | | | {:.2} |", lat.beams.len(), total / 1e3);
    println!("\n(The shell only: the mechanical parts, closures and their mounts carry their own literal values. Steel yields at 0.1-0.3 % strain.)");
}
