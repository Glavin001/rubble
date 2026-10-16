//! Bond -> joint conversion (formerly fracture/bonds.test.mjs): capacities and
//! springs follow Shardwright's reference formulas, scaling keeps row ratios,
//! and capping bounds the solver.

mod common;
use common::*;
use engine_fracture::bonds::*;
use engine_fracture::math::*;
use engine_fracture::scene::Joint;
use engine_fracture::build_scene;

#[test]
fn capacities_follow_the_reference_stress_solver() {
    let house = house();
    let n = house.material_count() as i32;
    let mortar = house.physics.bonds.iter().find(|b| b.composition.first().is_some_and(|c| c.interface_material == n) && b.b >= 0).unwrap();
    let [t, c, s, m] = bond_capacities(house, mortar, 1.0);
    let ft = house.interface_material(n).unwrap()["tensile_strength"].as_f64().unwrap();
    let (k, area) = (mortar.strength_scale as f64, mortar.area as f64);
    let fc = [mortar.a, mortar.b as u32]
        .iter()
        .map(|&f| house.material(house.physics.fragments[f as usize].material_ids[0])["compressive_strength"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    close(t, ft * area * k, 1e-6);
    close(c, fc * area * k, 1e-6);
    close(s, 1.5 * ft * area * k, 1e-6);
    let ch = (mortar.extent_half[0] as f64).max(mortar.extent_half[1] as f64);
    close(m, ft * (mortar.i_uu as f64).min(mortar.i_vv as f64) / ch * k, 1e-6);
    // A bearing (ft = 0) resists with 0.05 fc instead of nothing.
    let bearing = house.physics.bonds.iter().find(|b| {
        b.composition.first().and_then(|c| house.interface_material(c.interface_material)).is_some_and(|m| m["tensile_strength"].as_f64() == Some(0.0))
    });
    if let Some(b) = bearing {
        assert!(bond_capacities(house, b, 1.0)[0] > 0.0);
    }
}

#[test]
fn springs_are_series_compliances_of_both_fragments_and_the_interface() {
    let house = house();
    let b = house.physics.bonds.iter().find(|x| x.b >= 0 && x.composition.first().is_some_and(|c| c.interface_material >= 0)).unwrap();
    let [kn, ks, ks2, ku, kv, kt] = bond_springs(house, b);
    assert_eq!(ks, ks2);
    assert!([kn, ks, ku, kv, kt].iter().all(|x| x.is_finite() && *x > 0.0));
    close(ku / kn, b.i_uu as f64 / b.area as f64, 1e-6);
    close(kv / kn, b.i_vv as f64 / b.area as f64, 1e-6);
    close(kt / ks, b.polar_moment as f64 / b.area as f64, 1e-6);
    // Every bond of the house has finite springs (incl. anisotropic timber).
    for x in &house.physics.bonds {
        assert!(bond_springs(house, x).iter().all(|k| k.is_finite()), "bond {}", x.id);
    }
}

#[test]
fn joints_weld_at_the_centroid_with_the_rest_pose_and_capped_springs() {
    let house = house();
    let s = build_scene(house, 2, None).unwrap();
    let dt = BOND_TICK;
    assert_eq!(s.joints.len(), house.physics.bonds.iter().filter(|b| b.level == 2).count());
    for j in s.joints.iter().take(200) {
        let (a, b) = (s.bodies[j.body_a as usize], s.bodies[j.body_b as usize]);
        // Both anchors are the same world point at rest.
        let wa = add(a.p, rotate(a.q, j.anchor_a));
        let wb = add(b.p, rotate(b.q, j.anchor_b));
        assert!(length(sub(wa, wb)) < 1e-5);
        // qB = qA * rest.
        let qb = quat_mul(a.q, j.rest);
        assert!(((0..4).map(|k| qb[k] * b.q[k]).sum::<f64>().abs() - 1.0).abs() < 1e-6);
        // Unit normal and u, orthogonal, in A's frame.
        assert!((length(j.normal) - 1.0).abs() < 1e-5 && dot(j.normal, j.frame_u).abs() < 1e-5);
        let max = |v: [f64; 3]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let m = (1.0 / a.inv_mass).min(if b.inv_mass > 0.0 { 1.0 / b.inv_mass } else { f64::INFINITY });
        let i = (1.0 / max(a.inv_i)).min(if b.inv_mass > 0.0 { 1.0 / max(b.inv_i) } else { f64::INFINITY });
        for (r, k) in j.k.iter().enumerate() {
            assert!(*k > 0.0 && *k <= BOND_PENALTY_CAP * if r < 3 { m } else { i } / (dt * dt) * (1.0 + 1e-6));
        }
    }
    // Uncapped rows keep the physical ratios exactly (one scene-wide factor).
    let raw: Vec<[f64; 6]> = house.physics.bonds.iter().filter(|b| b.level == 2).map(|b| bond_springs(house, b)).collect();
    let joints: Vec<Joint> = s.joints.iter().zip(&raw).map(|(j, k)| Joint { k: *k, ..*j }).collect();
    let scaled = scale_bond_springs_with(&joints, &s.bodies, dt, 0.5, f64::INFINITY);
    let f = scaled[0].k[0] / raw[0][0];
    for (j, k) in scaled.iter().zip(&raw).take(50) {
        for (a, b) in j.k.iter().zip(k) {
            close(*a, b * f, 1e-9);
        }
    }
}
