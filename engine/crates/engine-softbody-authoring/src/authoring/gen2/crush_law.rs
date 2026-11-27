//! The load path's bays on Rubble's crush law ([`engine_vehiclegen::physics::FrontRules::crush_law`],
//! `engine/docs/beam-yield-law.md`; the solver's `engine_softbody::explicit_crush`, CPU reference
//! only). With the switch off nothing here runs. With it on (and [`super::load_path`]'s cells), each
//! bay of the crash box and the rail:
//!
//! - its four corner chords are one crush cell (`rubbleCrushCell`): peak at the section's
//!   effective-width squash load, plateau at what the load path gives them today (the section's mean
//!   crush force less the diagonals' share), the drop over the first fold's half-length, densified
//!   at `deformLimit` 0.73 past the solver's 0.1 m (`rubbleMinRest` at the solver's least), tension
//!   at the section's squash share, and the section's collapse moments (strong, weak) held while it
//!   folds; one spring for the four (the least of theirs: a cell's modes need one);
//! - each long face has both diagonals (the second added by `load_path::lattice` with this
//!   switch): in compression at the flat yield the load path gives them (together a quarter of the
//!   bay's force along it, as before), with the same least rest length; in tension at the wall's
//!   plastic shear over the diagonal's sine, so the cell keeps its walls' shear strength while it
//!   folds (one diagonal a face, at a quarter of the crush force, held a tenth of it).
//!
//! Every value is the section's through the calculator ([`engine_vehiclegen::crush_law`]) times the
//! character's multiple and the crash-rate factor the load path already applies; none is fitted.

use super::load_path::{BAYS, FACES, STATIONS};
use super::rig::{Rig, RigInput};
use engine_softbody::explicit_crush::MIN_REST_FLOOR;
use engine_vehiclegen::crush_law::member_crush;
use serde_json::{json, Map, Value};

fn on(inp: &dyn RigInput) -> bool {
    let fr = &inp.physics().front_rules;
    fr.load_path && fr.crush_law && engine_vehiclegen::physics::load_path_step("cells")
}

/// On the rigged car, after [`super::load_path::apply`].
pub fn apply(r: &mut Rig, inp: &dyn RigInput) {
    if !on(inp) {
        return;
    }
    let ph = inp.physics();
    let Some(d) = inp.definitions() else { return };
    let st = d.structure;
    let car = &mut r.car;
    let mut pairs = std::collections::HashMap::new();
    for j in (0..car.beams.len()).filter(|j| car.fixed.contains_key(j)) {
        let (a, b) = (car.nodes[car.beams[j].a].name.clone(), car.nodes[car.beams[j].b].name.clone());
        pairs.entry((b.clone(), a.clone())).or_insert(j);
        pairs.entry((a, b)).or_insert(j);
    }
    let find = |a: &str, b: &str| pairs.get(&(a.to_string(), b.to_string())).copied();
    let log = std::env::var("FRONT_RULES_LOG").is_ok();
    let mut cells = 0;
    for s in ["l", "r"] {
        for (bi, (fi, ri, kind, _)) in BAYS.into_iter().enumerate() {
            let Some(m) = st.members.iter().find(|m| m.kind == kind) else { continue };
            let Some(mc) = member_crush(m) else { continue };
            let scale = f64::from(ph.crash.front_crush) * super::load_path::rate(m, ph);
            let n = |k: usize, c: usize| format!("sh_{}_{}{s}", STATIONS[k].0, STATIONS[k].1[c]);
            let Some(chords): Option<Vec<usize>> = (0..4).map(|c| find(&n(fi, c), &n(ri, c))).collect() else { continue };
            // each long face's two diagonals (`load_path::lattice` adds the second): the
            // side faces (outboard, inboard) carry the side walls' shear, the top and bottom the others'
            let diags: Vec<(usize, f64)> = FACES.iter().flat_map(|&(a, b)| [(a, b), (b, a)].map(|(x, y)| (find(&n(fi, x), &n(ri, y)), if a == 0 && b == 1 || a == 2 && b == 3 { mc.shear[0] } else { mc.shear[1] }))).filter_map(|(j, v)| Some((j?, v))).collect();
            let along = |car: &super::build::Car, j: usize| {
                let v = car.nodes[car.beams[j].a].p - car.nodes[car.beams[j].b].p;
                f64::from(v.y.abs() / v.length())
            };
            let diag_axial: f64 = diags.iter().map(|&(j, _)| f64::from(car.beams[j].deform) * along(car, j)).sum();
            let plateau = chords.iter().map(|&j| f64::from(car.beams[j].deform)).sum::<f64>() / 4.0;
            let peak = ((mc.peak * scale - diag_axial) / 4.0).max(plateau);
            let k = chords.iter().map(|&j| car.beams[j].k).fold(f32::INFINITY, f32::min);
            // the drop over the first fold's half-length, never steeper than the solver's k/2
            let drop = mc.drop.max(2.05 * (peak - plateau) / f64::from(k));
            let cell = format!("load_path_{s}_{bi}");
            for &j in &chords {
                let b = &mut car.beams[j];
                b.k = k;
                b.deform = peak as f32;
                // (never breaking below its own yield: the tear is past the section's squash share)
                b.strength = b.strength.max((1.1 * (mc.tension * scale / 4.0).max(peak)) as f32);
                let mut o = Map::new();
                o.insert("rubbleCrushPlateau".into(), json!(plateau as f32));
                o.insert("rubbleCrushDrop".into(), json!(drop as f32));
                o.insert("rubbleMinRest".into(), json!(MIN_REST_FLOOR));
                o.insert("rubbleTensionYield".into(), json!((mc.tension * scale / 4.0) as f32));
                o.insert("rubbleCrushCell".into(), Value::String(cell.clone()));
                o.insert("rubbleCellMoment".into(), json!([(mc.moment[0] * scale) as f32, (mc.moment[1] * scale) as f32]));
                car.crush_keys.insert(j, o);
                if let Some(l) = car.fixed_limits.get_mut(&j) {
                    l.1 = peak as f32;
                }
            }
            for &(j, v) in &diags {
                // in tension the wall's plastic shear over the diagonal's sine
                let tt = v * scale / (1.0 - along(car, j).powi(2)).sqrt();
                car.beams[j].strength = car.beams[j].strength.max((1.1 * tt) as f32);
                let mut o = Map::new();
                o.insert("rubbleCrushPlateau".into(), json!(car.beams[j].deform));
                o.insert("rubbleMinRest".into(), json!(MIN_REST_FLOOR));
                o.insert("rubbleTensionYield".into(), json!(tt as f32));
                car.crush_keys.insert(j, o);
            }
            cells += 1;
            if log {
                eprintln!(
                    "crush law {cell}: chords peak {:.1} → plateau {:.1} kN over {:.1} mm, tension {:.1} kN, k {:.3e}; moments {:.2} / {:.2} kN·m; diagonals' axial {:.1} kN",
                    peak / 1e3,
                    plateau / 1e3,
                    drop * 1e3,
                    mc.tension * scale / 4e3,
                    k,
                    mc.moment[0] * scale / 1e3,
                    mc.moment[1] * scale / 1e3,
                    diag_axial / 1e3
                );
            }
        }
    }
    if log {
        eprintln!("crush law: {cells} cells");
    }
}
