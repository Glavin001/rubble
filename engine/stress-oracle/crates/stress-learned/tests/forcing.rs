use std::collections::{BTreeMap, HashSet};
use stress_learned::{
    forcing::{forcing_curve, forcing_jobs, point_force_pilot_jobs},
    layouts::{layout_job, wall_layouts},
};
use stress_ref::scene::{LoadDesc, Support};

#[test]
fn forcing_catalog_balances_factors_and_preserves_geometry_and_accuracy() {
    let jobs = forcing_jobs(101);
    assert_eq!(jobs.len(), 144);
    let mut ids = HashSet::new();
    let mut counts = BTreeMap::new();
    for job in jobs {
        assert!(ids.insert(job.event_id.clone()));
        job.scene.validate().unwrap();
        let info = &job.scene.oracle["learned_impact"];
        let p = &info["parameters"];
        assert_eq!(info["layout_catalog"], "layouts-v4");
        let index = p["layout_index"].as_u64().unwrap() as usize;
        let base = layout_job(
            wall_layouts()[index],
            info["layout_group"].as_str().unwrap(),
            0,
            101,
            8,
        );
        assert_eq!(
            serde_json::to_value(&job.scene.sim).unwrap(),
            serde_json::to_value(&base.scene.sim).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&job.scene.bodies).unwrap(),
            serde_json::to_value(&base.scene.bodies).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&job.scene.materials).unwrap(),
            serde_json::to_value(&base.scene.materials).unwrap()
        );
        assert!(job.scene.impactors.is_empty());
        assert_eq!(job.scene.loads.len(), 1);
        assert_eq!(job.frames, 8);
        assert_eq!(job.scene.sim.duration, 8. * job.scene.sim.frame_dt);
        let key = (
            p["load_kind"].to_string(),
            p["curve_kind"].to_string(),
            p["peak"].to_string(),
            p["duration"].to_string(),
        );
        *counts.entry(key).or_insert(0) += 1;
        match &job.scene.loads[0] {
            LoadDesc::Pressure {
                chunks,
                face_normal,
                ..
            } => {
                assert_eq!(*face_normal, [0., -1., 0.]);
                let selected = chunks.select(&job.scene.bodies[0]);
                assert!(!selected.is_empty());
                assert!(selected
                    .iter()
                    .all(|&i| job.scene.bodies[0].chunks[i].support == Support::None));
            }
            LoadDesc::PointForce { chunk, point, .. } => {
                let c = &job.scene.bodies[0].chunks[*chunk];
                assert_eq!(c.support, Support::None);
                assert!((point.unwrap()[2] - c.center[2] - 0.04).abs() < 1e-12);
            }
            _ => panic!("unexpected load"),
        }
    }
    assert_eq!(counts.len(), 36);
    assert!(counts.values().all(|&n| n == 4));
}

#[test]
fn pulse_tables_preserve_both_pulses_and_zero_tail() {
    let pulse = forcing_curve("table", 100., 0.06);
    assert_eq!(pulse.eval(0.), 0.);
    assert!((pulse.eval(0.008 + 0.06 * 0.15) - 100.).abs() < 1e-10);
    assert_eq!(pulse.eval(0.008 + 0.06 * 0.45), 0.);
    assert!((pulse.eval(0.008 + 0.06 * 0.75) - 60.).abs() < 1e-10);
    assert_eq!(pulse.eval(0.13), 0.);
    assert!(forcing_curve("half_sine", 100., 0.06).eval(0.008 + 0.03) > 99.999);
}

#[test]
fn pressure_and_point_force_worlds_run_without_removed_projectile_probes() {
    let jobs = forcing_jobs(101);
    for kind in ["prescribed_pressure", "prescribed_point_force"] {
        let job = jobs.iter().find(|j| j.load_type == kind).unwrap();
        assert!(
            !job.scene.probes.is_empty(),
            "retain structural diagnostics"
        );
        let mut world = stress_ref::world::World::new(&job.scene);
        world.step_frame();
        assert!(world
            .solver
            .chunk_position(0, 0)
            .to_array()
            .iter()
            .all(|v| v.is_finite()));
    }
}

#[test]
fn point_force_pilot_varies_only_loads_on_a_training_layout() {
    let jobs=point_force_pilot_jobs(101);
    assert_eq!(jobs.len(),18);
    let base=layout_job(wall_layouts()[2],"02",0,101,8);
    let mut factors=BTreeMap::new();let mut chunks=HashSet::new();let mut ids=HashSet::new();
    for job in jobs {
        job.scene.validate().unwrap();assert!(ids.insert(job.event_id.clone()));
        for (actual,expected) in [
            (serde_json::to_value(&job.scene.bodies).unwrap(),serde_json::to_value(&base.scene.bodies).unwrap()),
            (serde_json::to_value(&job.scene.materials).unwrap(),serde_json::to_value(&base.scene.materials).unwrap()),
            (serde_json::to_value(&job.scene.sim).unwrap(),serde_json::to_value(&base.scene.sim).unwrap())] {assert_eq!(actual,expected);}
        let info=&job.scene.oracle["learned_impact"];
        assert_eq!(info["layout_catalog"],"layouts-v4");assert_eq!(info["layout_group"],"02");
        assert_eq!(info["partition_intent"],"train_only_pilot");
        assert!(job.scene.impactors.is_empty());assert_eq!(job.frames,8);
        let LoadDesc::PointForce{chunk,direction,point,magnitude,..}=&job.scene.loads[0] else {panic!("point force required")};
        assert_eq!(*direction,[0.,1.,0.]);assert_eq!(job.scene.bodies[0].chunks[*chunk].support,Support::None);
        assert!((point.unwrap()[2]-job.scene.bodies[0].chunks[*chunk].center[2]-0.04).abs()<1e-12);
        chunks.insert(*chunk);
        let p=&info["parameters"];let peak=p["peak"].as_f64().unwrap();
        assert!((magnitude.eval(0.038)-peak).abs()<1e-8);
        *factors.entry((p["peak"].to_string(),p["curve_kind"].to_string())).or_insert(0)+=1;
    }
    assert_eq!(chunks.len(),3);assert_eq!(factors.len(),6);assert!(factors.values().all(|&n|n==3));
}

#[test]
fn impulse_pilot_has_finite_balanced_loads_and_preserves_oracle_accuracy() {
    check_impulse_catalog(stress_learned::forcing::point_impulse_pilot_jobs(101), [12.,36.,96.]);
}

#[test]
fn second_impulse_range_preserves_finite_factorial_and_physics() {
    check_impulse_catalog(stress_learned::forcing::point_impulse_range_jobs(101), [192.,384.,768.]);
}

fn check_impulse_catalog(jobs: Vec<stress_learned::generate::Job>, levels: [f64;3]) {
    assert_eq!(jobs.len(), 108);
    let base = layout_job(wall_layouts()[2], "02", 0, 101, 8);
    let mut ids = HashSet::new();
    let mut combinations = HashSet::new();
    for job in jobs {
        job.scene.validate().unwrap();
        assert!(ids.insert(job.event_id.clone()));
        assert_eq!(serde_json::to_value(&job.scene.sim).unwrap(), serde_json::to_value(&base.scene.sim).unwrap());
        assert_eq!(serde_json::to_value(&job.scene.bodies).unwrap(), serde_json::to_value(&base.scene.bodies).unwrap());
        assert_eq!(serde_json::to_value(&job.scene.materials).unwrap(), serde_json::to_value(&base.scene.materials).unwrap());
        assert!(job.scene.impactors.is_empty());
        assert_eq!(job.frames, 8);
        let info = &job.scene.oracle["learned_impact"];
        assert_eq!(info["layout_group"], "02");
        assert_eq!(info["layout_catalog"], "layouts-v4");
        assert_eq!(info["partition_intent"], "train_only_pilot");
        let p = &info["parameters"];
        let duration = p["duration"].as_f64().unwrap();
        let impulse = p["impulse"].as_f64().unwrap();
        assert!(levels.contains(&impulse));
        let LoadDesc::PointForce { chunk, direction, point, magnitude, .. } = &job.scene.loads[0] else { panic!("expected point force") };
        let c = &job.scene.bodies[0].chunks[*chunk];
        assert_eq!(c.support, Support::None);
        assert!((point.unwrap()[2]-c.center[2]-p["application"]["offset_z"].as_f64().unwrap()).abs()<1e-12);
        assert_eq!(direction.iter().map(|x|x*x).sum::<f64>(), 1.);
        assert_eq!(magnitude.eval(0.), 0.);
        assert_eq!(magnitude.eval(0.008+duration+1e-8), 0.);
        assert_eq!(magnitude.eval(job.scene.sim.duration), 0.);
        // Independent numerical integration checks the saved TimeFunction, not metadata only.
        let steps = 10000;
        let dt = duration / steps as f64;
        let integrated: f64 = (0..steps).map(|i| magnitude.eval(0.008+(i as f64+0.5)*dt)*dt).sum();
        assert!((integrated-impulse).abs()<impulse*1e-7);
        assert!(combinations.insert((chunk.to_string(),p["impulse"].to_string(),p["duration"].to_string(),
            p["application"]["direction"].to_string(),p["application"]["offset_z"].to_string())));
    }
    assert_eq!(combinations.len(), 108);
}
