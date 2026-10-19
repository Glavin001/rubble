use std::collections::HashSet;
use stress_learned::layouts::{layout_jobs, scale_layout_job};
use stress_ref::{builders, scene::Support};

#[test]
fn independent_layouts_have_valid_geometry_and_unchanged_solver_settings() {
    let jobs = layout_jobs(101);
    assert_eq!(jobs.len(), 48);
    let mut layouts = HashSet::new();
    let mut geometry = HashSet::new();
    for job in &jobs {
        job.scene.validate().expect("valid source scene");
        let gen = &job.scene.oracle["learned_impact"];
        layouts.insert(gen["layout_group"].as_str().unwrap());
        let counts: Vec<usize> =
            serde_json::from_value(gen["parameters"]["counts"].clone()).unwrap();
        let body = &job.scene.bodies[0];
        let mut expected: usize = counts.iter().product();
        if !gen["parameters"]["opening_grid"].is_null() {
            let h: Vec<usize> =
                serde_json::from_value(gen["parameters"]["opening_grid"].clone()).unwrap();
            expected -= (h[1] - h[0]) * (h[3] - h[2]) * counts[1];
        }
        assert_eq!(body.chunks.len(), expected);
        assert_eq!(
            body.chunks
                .iter()
                .filter(|c| c.support == Support::Fixed)
                .count(),
            counts[0] * counts[1]
        );
        assert!(body.chunks.iter().all(|c| c.half_extents == [0.05; 3]));
        geometry.insert(serde_json::to_string(body).unwrap());
        let base = builders::wall_impact(15., "base");
        let mut a = serde_json::to_value(&job.scene.sim).unwrap();
        let mut b = serde_json::to_value(&base.sim).unwrap();
        for field in ["seed", "duration"] {
            a.as_object_mut().unwrap().remove(field);
            b.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(a, b, "accuracy settings changed");
        assert_eq!(
            job.scene.sim.duration,
            job.frames as f64 * job.scene.sim.frame_dt
        );
    }
    assert_eq!(layouts.len(), 12);
    assert_eq!(geometry.len(), 12);
    for group in jobs.chunks(4) {
        assert_eq!(
            group
                .iter()
                .map(|j| j.scene.oracle["learned_impact"]["parameters"]["case"]
                    .as_u64()
                    .unwrap())
                .collect::<HashSet<_>>()
                .len(),
            4
        );
        assert!(group
            .iter()
            .all(|j| serde_json::to_value(&j.scene.bodies).unwrap()
                == serde_json::to_value(&group[0].scene.bodies).unwrap()));
    }
}

#[test]
fn scale_layout_contains_over_ten_thousand_free_chunks_without_coarsening() {
    let job = scale_layout_job(101);
    job.scene.validate().unwrap();
    let body = &job.scene.bodies[0];
    assert_eq!(body.chunks.len(), 10400);
    assert_eq!(
        body.chunks
            .iter()
            .filter(|c| c.support == Support::None)
            .count(),
        10240
    );
    assert!(body.chunks.iter().all(|c| c.half_extents == [0.05; 3]));
    assert_eq!(job.frames, 1);
}
