//! Asynchronous ray queries delivered with the frame, and debug drawing.

use engine_schema::BodyState;
use rubble::*;

fn world() -> World {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 1024,
            max_colliders: 1024,
            max_contacts: 32_768,
            max_joints: 64,
            max_candidates: 65_536,
            max_colors: 32,
            max_commands: 256,
            max_hull_vertices: 1024,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [40.0, 0.5, 40.0])
        .unwrap();
    w
}

fn pile(w: &mut World) -> Vec<BodyId> {
    (0..20)
        .map(|k| {
            let b = w
                .create_body(&BodyDef::dynamic([
                    (k % 4) as f32 * 1.1,
                    0.5 + (k / 4) as f32 * 1.02,
                    0.0,
                ]))
                .unwrap();
            w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
                .unwrap();
            b
        })
        .collect()
}

fn down(x: f32, z: f32) -> RayInput {
    RayInput {
        origin: [x, 20.0, z],
        translation: [0.0, -40.0, 0.0],
        ignore_body: None,
    }
}

#[test]
fn ray_batches_arrive_with_the_frame_of_their_step() {
    let mut w = world();
    let ids = pile(&mut w);
    w.step_sync().unwrap();
    let a = w.cast_rays(&[down(0.0, 0.0), down(30.0, 30.0)]).unwrap();
    let b = w.cast_rays(&[down(50.0, 0.0)]).unwrap();
    let f = w.step_sync().unwrap();
    assert_eq!(f.ray_results.len(), 2);
    let ra = f.ray_results.iter().find(|r| r.batch == a).unwrap();
    assert_eq!(ra.hits.len(), 2);
    let top = ra.hits[0].expect("hits the column of boxes");
    assert!(ids.contains(&top.body), "the first ray hits a box: {top:?}");
    let ground = ra.hits[1].expect("hits the ground");
    assert!(!ids.contains(&ground.body) && (ground.distance - 20.0).abs() < 1e-3);
    let rb = f.ray_results.iter().find(|r| r.batch == b).unwrap();
    assert_eq!(rb.hits, vec![None], "past the ground's edge: no hit");
    // Nothing queued: nothing reported.
    assert!(w.step_sync().unwrap().ray_results.is_empty());
}

#[test]
fn ray_queries_do_not_change_the_simulation() {
    let mut with = world();
    let mut without = world();
    pile(&mut with);
    pile(&mut without);
    for k in 0..200 {
        with.cast_rays(&[down((k % 5) as f32, 0.0); 8]).unwrap();
        with.step().unwrap();
        without.step().unwrap();
        with.poll().unwrap();
        without.poll().unwrap();
    }
    with.poll_blocking().unwrap();
    without.poll_blocking().unwrap();
    let bytes =
        |w: &World| bytemuck::cast_slice::<BodyState, u8>(&w.sim().download_bodies()).to_vec();
    assert!(bytes(&with) == bytes(&without));
}

#[test]
fn an_asleep_world_still_answers_queries() {
    let mut w = world();
    pile(&mut w);
    for _ in 0..1500 {
        w.step().unwrap();
        w.poll().unwrap();
    }
    w.poll_blocking().unwrap();
    assert!(!w.step_sync().unwrap().ran, "asleep and skipping");
    let id = w.cast_rays(&[down(0.0, 0.0)]).unwrap();
    let f = w.step_sync().unwrap();
    let r = f.ray_results.iter().find(|r| r.batch == id).unwrap();
    assert!(r.hits[0].is_some());
}

#[derive(Default)]
struct Count {
    boxes: usize,
    spheres: usize,
    meshes: usize,
    transforms: usize,
}

impl DebugDraw for Count {
    fn draw_box(&mut self, _: Transform, _: [f32; 3], _: DebugColor) {
        self.boxes += 1;
    }
    fn draw_sphere(&mut self, _: [f32; 3], _: f32, _: DebugColor) {
        self.spheres += 1;
    }
    fn draw_capsule(&mut self, _: [f32; 3], _: [f32; 3], _: f32, _: DebugColor) {}
    fn draw_mesh(&mut self, _: Transform, mesh: &ShapeMesh, _: DebugColor) {
        assert!(!mesh.triangles.is_empty());
        self.meshes += 1;
    }
    fn draw_transform(&mut self, _: Transform) {
        self.transforms += 1;
    }
}

#[test]
fn debug_draw_visits_every_shape() {
    let mut w = world();
    pile(&mut w);
    let ball = w.create_body(&BodyDef::dynamic([5.0, 2.0, 0.0])).unwrap();
    w.create_sphere_shape(ball, &ShapeDef::default(), [0.0; 3], 0.5)
        .unwrap();
    let hull = ConvexHull::cuboid(glam::Vec3::splat(0.3));
    w.create_hull_shape(ball, &ShapeDef::default(), &hull)
        .unwrap();
    w.step_sync().unwrap();
    let mut c = Count::default();
    w.draw(
        &mut c,
        DebugDrawOptions {
            body_transforms: true,
            ..DebugDrawOptions::default()
        },
    );
    assert_eq!(c.boxes, 21, "ground + 20 boxes");
    assert_eq!(c.spheres, 1);
    assert_eq!(c.meshes, 1);
    assert_eq!(c.transforms, 22);
}
