use stress_learned::runtime_snapshot::RuntimeGraph;
use stress_ref::{builders, world::World};

#[test]
fn native_capture_retains_cracked_interfaces_retired_caches_and_unset_history() {
    let mut world = World::new(&builders::bond_tension());
    let graph = RuntimeGraph::new(&world).unwrap();
    let before = serde_json::to_value(graph.capture(&world).unwrap()).unwrap();
    assert!(before["joint_values"][0][4].is_null());
    assert_eq!(before["joint_flags"][0][3], 1);
    world.solver.bonds[0][0].joint.damage = 1.;
    world.solver.bonds[0][0].stored = 12.34567890123456;
    let cracked = graph.capture(&world).unwrap();
    assert_eq!(cracked.joint_flags[0][2..], [1, 0]);
    assert_eq!(cracked.response_values[0][12], 12.34567890123456);
    world.solver.bonds[0][0].alive = false;
    let retired = graph.capture(&world).unwrap();
    assert_eq!(retired.joint_flags[0][2..], [0, 0]);
    assert_eq!(retired.response_values[0], cracked.response_values[0]);
    // Capturing twice has no solver side effects.
    assert_eq!(
        serde_json::to_value(retired).unwrap(),
        serde_json::to_value(graph.capture(&world).unwrap()).unwrap()
    );
    world.solver.time += world.scene.sim.frame_dt / 2.;
    assert!(graph
        .capture(&world)
        .err()
        .unwrap()
        .contains("frame boundary"));
}

#[test]
fn native_graph_rejects_new_slots_and_changed_source() {
    let mut world = World::new(&builders::bond_tension());
    world.solver.bonds[0][0].alive = false;
    let graph = RuntimeGraph::new(&world).unwrap();
    world.solver.bonds[0][0].alive = true;
    assert!(graph.capture(&world).err().unwrap().contains("newly alive"));
    world.solver.bonds[0][0].alive = false;
    world.scene.sim.stiffness_scale *= 2.;
    assert!(graph
        .capture(&world)
        .err()
        .unwrap()
        .contains("source scene changed"));
}

#[test]
fn boundary_kinematics_preserve_small_deformation_and_relative_rate_without_mutation() {
    use stress_ref::math::Vec3;
    for corotational in [false,true] {
        let mut scene=builders::bond_tension();scene.sim.methods.corotational_bonds=corotational;
        let mut world=World::new(&scene);let graph=RuntimeGraph::new(&world).unwrap();
        let g=world.solver.bonds[0][0].geometry.clone();
        world.solver.chunks[0][g.a].u=Vec3::new(0.001,0.002,0.003);
        world.solver.chunks[0][g.b].u=world.solver.chunks[0][g.a].u+g.normal*1e-9;
        world.solver.chunks[0][g.a].v=Vec3::new(1.,2.,3.);
        world.solver.chunks[0][g.b].v=world.solver.chunks[0][g.a].v+g.normal*0.02;
        let before=serde_json::to_value(graph.capture(&world).unwrap()).unwrap();
        let (snapshot,kinematics)=graph.capture_with_kinematics(&world).unwrap();
        assert_eq!(serde_json::to_value(snapshot).unwrap(),before);
        assert_eq!(serde_json::to_value(graph.capture(&world).unwrap()).unwrap(),before);
        let q=kinematics.values[0];
        assert!((q[2]-1e-9).abs()<1e-18);assert!((q[8]-0.02).abs()<1e-14);
        for i in [0,1,3,4,5,6,7,9,10,11]{assert!(q[i].abs()<1e-14);}
        world.solver.bonds[0][0].alive=false;
        assert_eq!(graph.capture_with_kinematics(&world).unwrap().1.values[0],[0.;12]);
    }
}

#[test]
fn physical_history_preserves_raw_small_state_full_inertia_and_stable_identity() {
    use stress_ref::math::Vec3;
    let mut world = World::new(&builders::bond_tension());
    let graph = RuntimeGraph::new(&world).unwrap();
    world.solver.chunks[0][0].u = Vec3::new(0.12345678901234567, 1e-14, -1e-17);
    world.solver.chunks[0][0].w = Vec3::new(1.2345678901234567, 2., 3.);
    let (legacy, physical) = graph.capture_with_physical_history(&world).unwrap();
    assert_eq!(serde_json::to_value(legacy).unwrap(), serde_json::to_value(graph.capture(&world).unwrap()).unwrap());
    assert_eq!(physical.chunks[0].state[0], world.solver.chunks[0][0].u.to_array());
    assert_eq!(physical.chunks[0].state[3], world.solver.chunks[0][0].w.to_array());
    let chunk = &world.solver.chunks[0][0];
    assert_eq!(physical.chunks[0].cluster_id, Some(world.solver.clusters[chunk.cluster].id));
    assert_eq!(physical.chunks[0].inertia, world.solver.structures[0].chunks[0].inertia.m);
    assert_eq!(physical.contacts.time, physical.time);
    assert_eq!(physical.chunks[0].key, graph.chunk_keys()[0]);
    assert_eq!(physical.bonds[0].key, graph.bond_keys()[0]);
    assert_eq!(physical.bonds[0].damping, world.solver.bonds[0][0].damping.to_array());
}

#[test]
fn physical_history_distinguishes_runtime_damping_omitted_from_legacy_features() {
    let mut world = World::new(&builders::bond_tension());
    let graph = RuntimeGraph::new(&world).unwrap();
    let before = graph.capture_with_physical_history(&world).unwrap();
    world.solver.bonds[0][0].damping = world.solver.bonds[0][0].damping.scale(2.);
    world.solver.mass_damping = vec![123.45678901234567];
    let after = graph.capture_with_physical_history(&world).unwrap();
    assert_eq!(serde_json::to_value(before.0).unwrap(), serde_json::to_value(&after.0).unwrap());
    assert_ne!(before.1.bonds[0].damping, after.1.bonds[0].damping);
    assert_eq!(after.1.bonds[0].damping, world.solver.bonds[0][0].damping.to_array());
    assert_eq!(after.1.cluster_damping[0].alpha_per_s, 123.45678901234567);
    assert!(!after.1.cluster_damping[0].inertia_relief);
    assert_eq!(after.1.format, "native-physical-history/2");
    assert_eq!(serde_json::to_value(after.0).unwrap(), serde_json::to_value(graph.capture(&world).unwrap()).unwrap());
    world.solver.mass_damping[0] = f64::NAN;
    assert!(graph.capture_with_physical_history(&world).err().unwrap().contains("cluster damping"));
}
