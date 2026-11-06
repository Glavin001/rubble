//! Writes the generated `gen2` sedan as a BeamNG mod folder (JBeam parts, a
//! `.pc`, a Collada render mesh and its materials, `files.json`) and prints its
//! census against the BeamNG rule-book (`engine/docs/beamng-authoring-census.md`)
//! and its mass and centre of mass against the Accord reference.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen2_export -- [out-dir]
//! ```
//! `out-dir` defaults to `engine/crates/engine-webgpu/__mods/gen2` (gitignored),
//! which the page loads as `softbody.html?solver=explicit&mod=/__mods/gen2&pc=gen2/gen2.pc&wheels=raycast`.
use engine_softbody::authoring::gen2::{self, build::*, sedan};
use std::collections::BTreeMap;

fn pct(v: &mut Vec<f32>, q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../engine-webgpu/__mods/gen2").to_string()
    });
    let out = std::path::Path::new(&out);
    for (p, bytes) in gen2::files() {
        let path = out.join(&p);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
    }
    println!("wrote {}", out.display());

    let car = sedan::sedan();
    let hub = sedan::hub_extra(&car);
    let rows = car.rows(&hub);
    let total: f32 = car.nodes.iter().map(|n| n.mass).sum::<f32>() + hub.values().sum::<f32>();
    let com = car
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| n.p * (n.mass + hub.get(&i).copied().unwrap_or(0.0)))
        .sum::<glam::Vec3>()
        / total;
    println!(
        "mass {total:.2} kg (kerb {:.0}); centre of mass {:.0} mm behind the front axle (band 1,001–1,248), {:.0} mm up, {:+.0} mm left",
        sedan::KERB,
        com.y * 1e3,
        com.z * 1e3,
        com.x * 1e3
    );
    println!(
        "nodes {} ({} contact-only), beams {}, triangles {}",
        car.nodes.len(),
        car.nodes.iter().filter(|n| n.contact).count(),
        car.beams.len(),
        car.tris.len()
    );
    println!("part | nodes (contact) | kg | node kg p10/p50/p90 | spring ends p50 | L p50 | k p50 | k·L p50 | deform p50 | ε_y p50 | row p50/p90/max | FLT_MAX");
    let mut by: BTreeMap<Part, ()> = BTreeMap::new();
    for n in &car.nodes {
        by.insert(n.part, ());
    }
    for part in by.keys() {
        let ids: Vec<usize> = (0..car.nodes.len())
            .filter(|&i| car.nodes[i].part == *part)
            .collect();
        let st: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&i| !car.nodes[i].contact)
            .collect();
        let kg: f32 = ids.iter().map(|&i| car.nodes[i].mass).sum();
        let mut m: Vec<f32> = st.iter().map(|&i| car.nodes[i].mass).collect();
        let mut ends: Vec<f32> = st
            .iter()
            .map(|&i| {
                car.beams
                    .iter()
                    .filter(|b| {
                        (b.a == i || b.b == i) && b.part == *part && !b.role.spec(b.part).support
                    })
                    .count() as f32
            })
            .collect();
        let beams: Vec<&Beam> = car
            .beams
            .iter()
            .filter(|b| {
                b.part == *part
                    && !car.nodes[b.a].contact
                    && !car.nodes[b.b].contact
                    && !b.role.spec(b.part).support
            })
            .collect();
        let len = |b: &Beam| car.nodes[b.a].p.distance(car.nodes[b.b].p);
        let mut l: Vec<f32> = beams.iter().map(|b| len(b)).collect();
        let mut k: Vec<f32> = beams.iter().map(|b| b.k).collect();
        let mut kl: Vec<f32> = beams.iter().map(|b| b.k * len(b)).collect();
        let mut d: Vec<f32> = beams
            .iter()
            .filter(|b| b.deform.is_finite())
            .map(|b| b.deform)
            .collect();
        let mut ey: Vec<f32> = beams
            .iter()
            .filter(|b| b.deform.is_finite())
            .map(|b| 100.0 * b.deform / (b.k * len(b)))
            .collect();
        let mut r: Vec<f32> = st.iter().map(|&i| rows[i]).collect();
        let fm = beams.iter().filter(|b| !b.strength.is_finite()).count() as f32
            / beams.len().max(1) as f32;
        let (m10, m50, m90) = (pct(&mut m, 0.1), pct(&mut m, 0.5), pct(&mut m, 0.9));
        let (r50, r90, rmax) = (pct(&mut r, 0.5), pct(&mut r, 0.9), pct(&mut r, 1.0));
        println!(
            "{} | {} ({}) | {kg:.1} | {m10:.2}/{m50:.2}/{m90:.2} | {:.0} | {:.2} | {:.2e} | {:.2e} | {:.1e} | {:.2} % | {r50:.2}/{r90:.2}/{rmax:.2} | {:.0} %",
            part.name(),
            ids.len(),
            ids.len() - st.len(),
            pct(&mut ends, 0.5),
            pct(&mut l, 0.5),
            pct(&mut k, 0.5),
            pct(&mut kl, 0.5),
            pct(&mut d, 0.5),
            pct(&mut ey, 0.5),
            fm * 100.0
        );
    }
    if let Ok(p) = std::env::var("GEN2_ROLES") {
        let mut roles: BTreeMap<Role, Vec<f32>> = BTreeMap::new();
        for b in car.beams.iter().filter(|b| b.part.name().contains(&p)) {
            roles
                .entry(b.role)
                .or_default()
                .push(b.k * car.nodes[b.a].p.distance(car.nodes[b.b].p));
        }
        for (r, mut v) in roles {
            let n = v.len();
            println!(
                "  {r:?}: {n} beams, k·L p10/p50/p90 {:.2e}/{:.2e}/{:.2e}",
                pct(&mut v, 0.1),
                pct(&mut v, 0.5),
                pct(&mut v, 0.9)
            );
        }
    }
    let mut lane: Vec<f32> = (0..car.nodes.len())
        .filter(|&i| car.nodes[i].contact)
        .map(|i| rows[i])
        .collect();
    println!("highest spring mode omega_max*h {:.3}", car.omega_h);
    println!(
        "contact-only nodes: row p50 {:.2} max {:.2}",
        pct(&mut lane, 0.5),
        pct(&mut lane, 1.0)
    );
}
