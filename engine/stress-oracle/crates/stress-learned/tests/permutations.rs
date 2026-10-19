use std::collections::HashSet;
use stress_learned::permutations::permutation_jobs;
use stress_ref::{builders, showcases};

#[test]
fn permutations_cover_physical_axes_without_changing_numerical_accuracy() {
    let jobs = permutation_jobs(101);
    assert_eq!(jobs.len(), 80);
    assert_eq!(jobs.iter().map(|j| j.frames).sum::<u32>(), 1216);
    assert_eq!(
        jobs.iter()
            .map(|j| &j.event_id)
            .collect::<HashSet<_>>()
            .len(),
        80
    );
    for j in &jobs {
        let base = match j.structure_type.as_str() {
            "wall_b5" => builders::wall_impact(15.0, "base"),
            "masonry_wall" => builders::masonry_wall(1.0, "base"),
            "masonry_house" => {
                showcases::masonry_house(showcases::HouseEvent::Impact { speed: 15.0 }, "base")
            }
            _ => panic!("unexpected family"),
        };
        let mut a = serde_json::to_value(&j.scene.sim).unwrap();
        let mut b = serde_json::to_value(&base.sim).unwrap();
        // Duration and seed are event choices. All timestep/tolerance/iteration
        // and solver feature settings must remain the builder defaults.
        for key in ["duration", "seed"] {
            a.as_object_mut().unwrap().remove(key);
            b.as_object_mut().unwrap().remove(key);
        }
        assert_eq!(a, b, "{} changed solver accuracy", j.event_id);
        assert!(j.scene.oracle["learned_impact"]["parameters"].is_object());
        assert_eq!(j.scene.oracle["learned_impact"]["split_group"], j.event_id);
        assert!(!j.scene.bodies.is_empty());
    }
    let walls = &jobs[..32];
    let masses: HashSet<_> = walls
        .iter()
        .map(|j| j.scene.impactors[0].mass.to_bits())
        .collect();
    assert!(masses.len() >= 8);
    let xs: HashSet<_> = walls
        .iter()
        .map(|j| j.scene.impactors[0].position[0].to_bits())
        .collect();
    assert!(xs.len() >= 7);
    assert!(walls.iter().any(|j| j.scene.impactors[0].velocity[0] < 0.0));
    assert!(walls.iter().any(|j| j.scene.impactors[0].velocity[0] > 0.0));
}


#[test]
fn finer_recording_preserves_duration_physics_and_split_identity() {
    use stress_learned::permutations::subdivide_frames;
    use stress_ref::world::World;
    let mut job = permutation_jobs(101).remove(0);
    let base = job.clone();
    let dt = World::new(&base.scene).substep_dt();
    subdivide_frames(&mut job, 4).unwrap();
    assert_eq!(job.frames, base.frames * 4);
    assert_eq!(job.scene.sim.frame_dt * job.frames as f64, base.scene.sim.frame_dt * base.frames as f64);
    assert_eq!(job.scene.oracle["learned_impact"]["split_group"], base.event_id);
    assert!(job.event_id.ends_with("_sub4"));
    assert!(World::new(&job.scene).substep_dt() <= dt);
    let mut a = serde_json::to_value(&job.scene).unwrap();
    let mut b = serde_json::to_value(&base.scene).unwrap();
    for v in [&mut a, &mut b] {
        v.as_object_mut().unwrap().remove("name");
        v.as_object_mut().unwrap().remove("oracle");
        v["sim"].as_object_mut().unwrap().remove("duration");
        v["sim"].as_object_mut().unwrap().remove("frame_dt");
    }
    assert_eq!(a,b,"only recording cadence/duration/provenance may change");
    assert!(subdivide_frames(&mut job,0).is_err());
    job.frames = u32::MAX;
    assert!(subdivide_frames(&mut job,2).is_err());
}

#[test]
fn spin_factorial_preserves_each_source_except_spin_and_metadata() {
    use stress_learned::permutations::projectile_spin_jobs;
    let sources = permutation_jobs(101);
    let jobs = projectile_spin_jobs(101);
    assert_eq!(jobs.len(), 52);
    assert_eq!(jobs.iter().map(|j| j.event_id.clone()).collect::<HashSet<_>>().len(), 52);
    for index in [0usize, 6, 11, 19] {
        let base = &sources[index];
        let group: Vec<_> = jobs.iter().filter(|j| j.scene.oracle["learned_impact"]["split_group"] == base.event_id).collect();
        assert_eq!(group.len(), 13);
        let mut spins = HashSet::new();
        for job in group {
            let spin = job.scene.impactors[0].angular_velocity;
            spins.insert(spin.map(f64::to_bits));
            assert_eq!(job.frames, base.frames);
            assert_eq!(job.qs_probe, base.qs_probe);
            let mut a = serde_json::to_value(&job.scene).unwrap();
            let mut b = serde_json::to_value(&base.scene).unwrap();
            for scene in [&mut a, &mut b] {
                for key in ["name", "description", "oracle"] { scene.as_object_mut().unwrap().remove(key); }
                scene["impactors"][0].as_object_mut().unwrap().remove("angular_velocity");
            }
            assert_eq!(a, b, "spin experiment changed another physical parameter");
        }
        assert!(spins.contains(&[0.0_f64.to_bits(); 3]));
        for axis in 0..3 {
            for rate in [-20.0_f64, -10.0, 10.0, 20.0] {
                let mut spin = [0.0_f64; 3];spin[axis] = rate;
                assert!(spins.contains(&spin.map(f64::to_bits)));
            }
        }
    }
}
