//! The lab's ball options: radius (a scene transform at load) and speed (the planner's
//! scale on the next shots). The mass stays as set.

use engine_fracture::cuboid::{build_cuboid_house, CuboidOptions};
use engine_fracture::plan::{default_shot_speed, Limits, Planner};
use engine_fracture::scene::grid_config;

const LIMITS: Limits = Limits { max_binding: 1 << 30, max_buffer: 1 << 30 };

#[test]
fn radius_scales_colliders_mesh_and_grid() {
    let one = build_cuboid_house(&CuboidOptions::default()).unwrap();
    let big = one.clone().scale_projectile_radius(4.0).unwrap();
    let r = one.projectile_radius * 4.0;
    assert_eq!(big.projectile_radius, r);
    let slots = big.projectile_start..big.projectile_start + big.projectile_count;
    let balls: Vec<_> = big.colliders.iter().filter(|c| slots.contains(&c.body)).collect();
    assert_eq!(balls.len(), big.projectile_count as usize);
    assert!(balls.iter().all(|c| c.params[0] == r));
    // Every other collider is untouched.
    for (a, b) in one.colliders.iter().zip(&big.colliders).filter(|(c, _)| !slots.contains(&c.body)) {
        assert_eq!(a.params, b.params);
    }
    // The render mesh: every vertex 4x further from its ball's centre, normals kept.
    for (a, b) in one.projectile_vertices.chunks_exact(10).zip(big.projectile_vertices.chunks_exact(10)) {
        for k in 0..3 {
            assert!((b[k] - 4.0 * a[k]).abs() <= 1e-5 * r as f32);
        }
        assert_eq!(a[3..], b[3..]);
    }
    assert_eq!(big.grid_cell_size, grid_config(&big.colliders).0);
    assert_eq!(big.projectile_mass, one.projectile_mass);
    assert!(one.scale_projectile_radius(0.0).is_err());
}

#[test]
fn speed_scale_multiplies_the_shot() {
    let scene = build_cuboid_house(&CuboidOptions::default()).unwrap();
    let mut plan = Planner::new(&scene, LIMITS).unwrap();
    let base = plan.shot_speed();
    assert_eq!(base, default_shot_speed(scene.extent, scene.projectile_speed));
    let shot = |plan: &mut Planner| {
        let (_, bytes) = plan.launch([0.0, 1.0, -20.0], [0.0, 0.0, 1.0], None, Some(false)).unwrap();
        let state: engine_schema::BodyState = bytemuck::pod_read_unaligned(&bytes);
        state.linear_vel
    };
    let v1 = shot(&mut plan);
    assert!((v1[2] as f64 - base).abs() < 1e-4 * base);
    plan.set_projectile_speed_scale(4.0).unwrap();
    let v4 = shot(&mut plan);
    assert!((v4[2] as f64 - 4.0 * base).abs() < 1e-4 * base);
    assert!(plan.set_projectile_speed_scale(-1.0).is_err());
}

/// No limit on shots: past the projectile slots, each shot reuses the oldest.
#[test]
fn shots_reuse_the_oldest_slot() {
    let scene = build_cuboid_house(&CuboidOptions::default()).unwrap();
    let mut plan = Planner::new(&scene, LIMITS).unwrap();
    let n = scene.projectile_count;
    let ids: Vec<u32> = (0..40).map(|_| plan.launch([0.0, 1.0, -20.0], [0.0, 0.0, 1.0], None, Some(false)).unwrap().0).collect();
    for (k, id) in ids.iter().enumerate() {
        assert_eq!(*id, scene.projectile_start + k as u32 % n, "shot {k}");
    }
}
