use stress_learned::{joint_history, response_history as response};
use stress_ref::{bond::Local6, builders, joint::StressMeasures, math::Vec3, world::World};

#[test]
fn response_sidecar_preserves_elastic_total_energy_and_retired_cache() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("response_history_exact");
    if root.exists() {
        std::fs::remove_dir_all(&root).unwrap();
    }
    std::fs::create_dir_all(&root).unwrap();
    let mut world = World::new(&builders::bond_tension());
    let keys = vec![(0, 0)];
    let b = &mut world.solver.bonds[0][0];
    b.force = Local6 {
        lin: Vec3::new(-0., 2., 3.),
        ang: Vec3::new(4., 5., 6.),
    };
    b.elastic = Local6 {
        lin: Vec3::new(7., 8., 9.),
        ang: Vec3::new(10., 11., 12.),
    };
    b.stored = 13.123456789012345;
    b.measures = StressMeasures {
        tension: -14.,
        shear: 15.,
        normal_compression: 16.,
        compression: -17.,
        compressive_force: 18.,
    };
    let expected = [
        -0.,
        2.,
        3.,
        4.,
        5.,
        6.,
        7.,
        8.,
        9.,
        10.,
        11.,
        12.,
        13.123456789012345,
        -14.,
        15.,
        16.,
        -17.,
        18.,
    ];
    let mut joint = joint_history::Recorder::new(&root, &world, &keys).unwrap();
    let mut rec = response::Recorder::new(&root, &world, &keys).unwrap();
    world.solver.time += 0.01;
    world.solver.bonds[0][0].alive = false;
    // Retirement changes participation; the last response remains exactly cached.
    joint.push(&world).unwrap();
    rec.push(&world).unwrap();
    joint.finish().unwrap();
    rec.finish().unwrap();
    response::validate(&root, &keys, 1, 0.01).unwrap();
    for i in 0..2 {
        let (rows, flags) = response::read_snapshot(&root.join("bond_response"), i, 1).unwrap();
        assert_eq!(rows[0].map(f64::to_bits), expected.map(f64::to_bits));
        assert_eq!(flags[0], if i == 0 { [1, 1] } else { [0, 0] });
    }
    // Valid individual flags still must agree with the joint-history topology.
    let path = root.join("bond_response/000001.i32");
    std::fs::write(&path, [1i32.to_le_bytes(), 1i32.to_le_bytes()].concat()).unwrap();
    assert!(response::validate(&root, &keys, 1, 0.01)
        .unwrap_err()
        .contains("topology"));
    std::fs::write(&path, [0i32.to_le_bytes(), 0i32.to_le_bytes()].concat()).unwrap();
    let path = root.join("bond_response/000001.f64");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0..8].copy_from_slice(&f64::NAN.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();
    assert!(response::validate(&root, &keys, 1, 0.01)
        .unwrap_err()
        .contains("invalid"));
}
