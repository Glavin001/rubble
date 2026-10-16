//! Energy balance per frame: `energy_trace <scene.json> [courant_safety]`.
//!
//! Prints the balance residual (mechanical + dissipated - initial; growth means energy is
//! being injected), the dissipation by kind, the first impactor's velocity, broken bonds
//! and the fastest chunk. Used to tell a numerical instability (residual grows) from a
//! physical event (residual flat, energy moves between terms).

use stress_ref::math::Vec3;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut s = stress_ref::scene::Scene::load(std::path::Path::new(&args[1])).unwrap_or_else(|e| panic!("{e}"));
    if let Some(c) = args.get(2) {
        s.sim.courant_safety = c.parse().expect("courant safety");
    }
    let mut w = stress_ref::world::World::new(&s);
    let e0 = w.mechanical_energy() + w.dissipated_energy();
    let frames = (s.sim.duration / s.sim.frame_dt).round() as usize;
    for f in 0..frames {
        w.step_frame();
        let sv = &w.solver;
        let mut fastest = (0.0, 0, Vec3::ZERO);
        for (st, chunks) in sv.chunks.iter().enumerate() {
            for c in (0..chunks.len()).filter(|&c| chunks[c].active) {
                let p = sv.chunk_position(st, c);
                let v = sv.point_velocity(st, c, p).norm();
                if v > fastest.0 {
                    fastest = (v, c, p);
                }
            }
        }
        let e = &sv.energy;
        let impactor = w.impactors.first().map(|i| i.velocity).unwrap_or(Vec3::ZERO);
        println!(
            "frame {f:3} residual {:8.2} J | mech {:10.2} dissipated {:9.2} (bond {:.1} damping {:.1} overshoot {:.1} split {:.1} contact {:.1}) | impactor {:.3?} | broken {} clusters {} | fastest {:.2} m/s chunk {} at {:.2?}",
            w.mechanical_energy() + w.dissipated_energy() - e0,
            w.mechanical_energy(),
            w.dissipated_energy(),
            e.bond_dissipation,
            e.damping_dissipation,
            e.softening_overshoot,
            e.split_release,
            w.contact.dissipated,
            impactor,
            sv.broken_bond_count(),
            sv.clusters.len(),
            fastest.0,
            fastest.1,
            fastest.2
        );
    }
}
