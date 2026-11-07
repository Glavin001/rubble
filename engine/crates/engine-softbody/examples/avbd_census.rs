//! Beam-kind, mass and stiffness census of a comparison car (the AVBD vehicle
//! mapping's input; `engine/docs/avbd-vehicles.md`).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example avbd_census -- etk_kc6
//! ```
use engine_softbody::vehicle_compare::{Car, Manifest};
use std::collections::BTreeMap;

fn main() {
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/avbd-vehicles-cars.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let want = std::env::args().nth(1).unwrap_or("etk_kc6".into());
    let spec = m.cars.iter().find(|c| c.name == want).unwrap();
    let car = Car::load(spec).unwrap();
    let s = &car.template;
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for b in &s.beams {
        *kinds.entry(format!("{:?}", b.kind)).or_default() += 1;
    }
    let masses: Vec<f32> = s.nodes.iter().map(|n| n.mass).collect();
    let tot: f32 = masses.iter().sum();
    let mut ms = masses.clone();
    ms.sort_by(f32::total_cmp);
    let mut ks: Vec<f32> = s.beams.iter().filter(|b| b.k > 0.0).map(|b| b.k).collect();
    ks.sort_by(f32::total_cmp);
    let q = |v: &[f32], p: f32| v[((v.len() - 1) as f32 * p) as usize];
    let mut wh: Vec<f32> = s
        .beams
        .iter()
        .filter(|b| b.k > 0.0 && !b.broken)
        .map(|b| {
            let m = s.nodes[b.a as usize]
                .mass
                .min(s.nodes[b.b as usize].mass)
                .max(1e-3);
            (b.k / m).sqrt()
        })
        .collect();
    wh.sort_by(f32::total_cmp);
    println!(
        "{}: {} nodes, {} beams, {} triangles, mass {:.0} kg",
        want,
        s.nodes.len(),
        s.beams.len(),
        s.triangles.len(),
        tot
    );
    println!("kinds: {kinds:?}");
    println!(
        "node mass min/p10/p50/p90/max: {:.3} {:.3} {:.3} {:.3} {:.3}",
        ms[0],
        q(&ms, 0.1),
        q(&ms, 0.5),
        q(&ms, 0.9),
        ms[ms.len() - 1]
    );
    println!(
        "beam k p10/p50/p90/max: {:.3e} {:.3e} {:.3e} {:.3e}",
        q(&ks, 0.1),
        q(&ks, 0.5),
        q(&ks, 0.9),
        ks[ks.len() - 1]
    );
    println!(
        "sqrt(k/m_min) p50/p90/max (rad/s): {:.0} {:.0} {:.0}; k h^2/m at 120 Hz p50: {:.1}",
        q(&wh, 0.5),
        q(&wh, 0.9),
        wh[wh.len() - 1],
        (q(&wh, 0.5) / 120.0).powi(2)
    );
    println!(
        "flags: beamng_profile {} centred {} surface {} anti_clip {} dissipative {} fixed {} ext {} tread {} drive {}",
        s.beamng_profile,
        s.settings.centred_damping,
        s.surface_contact,
        s.anti_clip,
        s.dissipative_contact,
        s.fixed_contact,
        s.beam_ext.len(),
        s.tread.len(),
        s.drive.is_some()
    );
}
