//! Debug snapshots report physically meaningful fields.

use stress_ref::builders::*;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::snapshot::BondStatus;
use stress_ref::world::World;

/// A bar fixed at one end and pulled at the other: the Love-Weber chunk stress of an
/// interior chunk is the uniaxial `F / A`, every bond is intact, and utilization is the
/// axial stress over the tensile strength.
#[test]
fn bar_in_tension_reports_uniaxial_stress() {
    let mut s = new_scene("snapshot_bar", "bar in tension");
    s.gravity = [0.0; 3];
    s.materials.insert("steel".into(), elastic_steel());
    let mut chunks = grid(Vec3::ZERO, [6, 1, 1], Vec3::new(0.2, 0.1, 0.1), "steel");
    chunks[0].support = Support::Fixed;
    let bonds = auto_bonds(&chunks, |_, _| "steel".into());
    s.bodies.push(body("bar", chunks, bonds));
    let force = 2.0e5;
    s.loads.push(LoadDesc::PointForce {
        body: "bar".into(),
        chunk: 5,
        direction: [1.0, 0.0, 0.0],
        magnitude: TimeFunction::Constant { value: force },
        point: None,
    });
    s.sim.solve_mode = SolveMode::QuasiStatic;
    s.sim.duration = 0.05;
    let mut w = World::new(&s);
    w.run();
    let snap = w.snapshot();
    let area = 0.1 * 0.1;
    let mid = snap.chunks.iter().find(|c| c.chunk == 3).unwrap();
    let sxx = mid.stress[0][0];
    assert!((sxx - force / area).abs() < 1e-6 * force / area, "sigma_xx {sxx} vs {}", force / area);
    assert!((mid.von_mises - force / area).abs() < 1e-6 * force / area);
    assert!((mid.max_principal - force / area).abs() < 1e-6 * force / area);
    assert!(mid.stress[1][1].abs() < 1e-6 * sxx && mid.stress[0][1].abs() < 1e-6 * sxx);
    assert!(snap.bonds.iter().all(|b| b.status == BondStatus::Intact));
    for b in &snap.bonds {
        assert!((b.axial_stress - force / area).abs() < 1e-6 * force / area);
    }
}
