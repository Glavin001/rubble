//! Oracle witnesses for state information a learned replacement must preserve.
use stress_ref::{
    bond::{BondGeometry, BondStiffness, Local6},
    joint::{JointModel, JointState, JointStrength},
    material::Material,
    math::Vec3,
    scene::Features,
};

#[test]
fn identical_current_geometry_has_different_force_after_opposite_slip_histories() {
    let geometry = BondGeometry::new(
        0,
        1,
        Vec3::ZERO,
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::new(0.05, 0.0, 0.0),
        Vec3::X,
        Vec3::Y,
        0.01,
        [0.1, 0.1],
    );
    let material = Material::analytic_test();
    let stiffness = BondStiffness::new(&geometry, &material, 1.0);
    let strength = JointStrength::new(&material, &geometry, None, 1.0, &Features::default());
    let model = JointModel {
        geometry: &geometry,
        stiffness: &stiffness,
        strength: &strength,
        rebar: None,
        weibull: 1.0,
    };
    let mut initial = JointState::new();
    // A compressed cracked interface still carries frictional contact forces.
    initial.damage = 1.0;
    let deformation = |shear| Local6 {
        lin: Vec3::new(shear, 0.0, -1e-7),
        ang: Vec3::ZERO,
    };
    let positive = model
        .evaluate(&initial, &deformation(0.01), 0.001, false)
        .state;
    let negative = model
        .evaluate(&initial, &deformation(-0.01), 0.001, false)
        .state;
    assert!(positive.plastic.lin[0] > 0.0);
    assert!(negative.plastic.lin[0] < 0.0);
    let a = model.evaluate(&positive, &deformation(0.0), 0.001, false);
    let b = model.evaluate(&negative, &deformation(0.0), 0.001, false);
    assert!(a.force.lin.x < 0.0 && b.force.lin.x > 0.0);
    assert!((a.force.lin.x + b.force.lin.x).abs() < 1e-10);
    // Same geometry/material/dt is ambiguous unless physical or latent memory
    // distinguishes these histories. This does not prove a GRU is mandatory:
    // the existing explicit plastic state already contains the distinction.
    assert_ne!(positive, negative);
}
