//! CCD on native Metal (`MetalSim`: the shared step plan, Slang → MSL): the
//! wgpu tunnelling suite's thin walls
//! (engine-webgpu tests/ccd_tunnel.rs), control-paired. A ball fired at 60 to
//! 250 m/s into a static slab or a wall of bricks 0.05 to 1 m thick, at 60 and
//! 120 Hz: with `BODY_CCD` it stops at the near face; without it the hardest
//! cases tunnel (the suite is sensitive). Then 40 m/s debris onto a slab.
//!
//! The scenes exceed 512 colliders so the broadphase takes its grid path.

use engine_metal::MetalSim;
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Wall {
    /// One static box (wider than a cell: the large-static list).
    Slab,
    /// Small static bricks, each in the grid.
    Bricks,
}

/// `MetalSim` one body and collider at a time, read back whole.
struct MetalWorld(MetalSim);

impl MetalWorld {
    fn add_body(&mut self, b: BodyState) -> u32 {
        self.0.add_bodies(&[b]).start
    }

    fn add_collider(&mut self, c: Collider) {
        self.0.add_colliders(&[c]);
    }

    fn step(&mut self) -> Result<(), engine_exec::ExecError> {
        self.0.step().map(|_| ())
    }

    fn body(&mut self, id: u32) -> BodyState {
        self.0.download_bodies()[id as usize]
    }
}

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 8192,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 64,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn body(pos: [f32; 3], inv_mass: f32, inv_i: [f32; 3], flags: u32, v: [f32; 3]) -> BodyState {
    BodyState {
        position: pos,
        inv_mass,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: inv_i,
        flags,
        linear_vel: v,
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    }
}

fn collider(body: u32, shape: ShapeType, params: [f32; 4]) -> Collider {
    Collider {
        body,
        shape: shape as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params,
    }
}

fn static_box(w: &mut MetalWorld, pos: [f32; 3], half: [f32; 3]) {
    let b = w.add_body(body(pos, 0.0, [0.0; 3], 0, [0.0; 3]));
    let r = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    w.add_collider(collider(b, ShapeType::Box, [half[0], half[1], half[2], r]));
}

fn dynamic_sphere(w: &mut MetalWorld, pos: [f32; 3], r: f32, mass: f32, v: [f32; 3], flags: u32) -> u32 {
    let i = 2.5 / (mass * r * r);
    let b = w.add_body(body(pos, 1.0 / mass, [i; 3], 9 | flags, v));
    w.add_collider(collider(b, ShapeType::Sphere, [r, 0.0, 0.0, r]));
    b
}

fn dynamic_box(w: &mut MetalWorld, pos: [f32; 3], half: [f32; 3], mass: f32, v: [f32; 3], flags: u32) -> u32 {
    let inv_i = [
        3.0 / (mass * (half[1] * half[1] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[1] * half[1])),
    ];
    let b = w.add_body(body(pos, 1.0 / mass, inv_i, 9 | flags, v));
    let r = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    w.add_collider(collider(b, ShapeType::Box, [half[0], half[1], half[2], r]));
    b
}

/// 600 small static boxes far from the action: past the exact pairwise path.
fn filler(w: &mut MetalWorld) {
    for i in 0..600 {
        let x = (i % 30) as f32 * 0.6 - 9.0;
        let y = (i / 30) as f32 * 0.6;
        static_box(w, [x, y, 80.0], [0.2, 0.2, 0.2]);
    }
}

fn world(rate: f32, ccd: bool, gravity: f32) -> MetalWorld {
    let mut w = MetalWorld(MetalSim::new(capacity()).expect("Metal"));
    let mut p = AvbdParams::default();
    p.dt = 1.0 / rate;
    p.gravity_y = gravity;
    p.iterations = 8;
    if ccd {
        p.flags |= FLAG_SPEC_VELOCITY;
    }
    w.0.set_params(p);
    filler(&mut w);
    w
}

/// Largest centre x a sphere (radius `r`) fired along +x at `speed` reaches
/// against a static wall whose near face is at x = 10.
fn shoot(rate: f32, speed: f32, thickness: f32, r: f32, ccd: bool, wall: Wall) -> f32 {
    let mut w = world(rate, ccd, 0.0);
    let near = 10.0;
    match wall {
        Wall::Slab => static_box(&mut w, [near + 0.5 * thickness, 0.0, 0.0], [0.5 * thickness, 1.5, 1.5]),
        Wall::Bricks => {
            let n = 12;
            for iy in 0..n {
                for iz in 0..n {
                    let y = (iy as f32 - (n as f32 - 1.0) * 0.5) * 0.2;
                    let z = (iz as f32 - (n as f32 - 1.0) * 0.5) * 0.2;
                    static_box(&mut w, [near + 0.5 * thickness, y, z], [0.5 * thickness, 0.1, 0.1]);
                }
            }
        }
    }
    let ball = dynamic_sphere(&mut w, [0.37, 0.03, 0.02], r, 5.0, [speed, 0.0, 0.0], if ccd { BODY_CCD } else { 0 });
    let steps = ((25.0 / speed) * rate).ceil() as u32 + 4;
    let mut max_x = f32::MIN;
    for _ in 0..steps {
        w.step().expect("step");
        max_x = max_x.max(w.body(ball).position[0]);
    }
    max_x
}

fn run_matrix(wall: Wall, thicknesses: &[f32]) {
    let (r, near) = (0.1f32, 10.0f32);
    let mut failures = Vec::new();
    let mut control_tunnels = 0;
    for &rate in &[60.0f32, 120.0] {
        for &speed in &[60.0f32, 140.0, 250.0] {
            for &th in thicknesses {
                let off = shoot(rate, speed, th, r, false, wall);
                let ccd = shoot(rate, speed, th, r, true, wall);
                println!("Metal {wall:?} {rate:>3} Hz {speed:>5} m/s wall {th:.2} m: max x off {off:7.3}, ccd {ccd:7.3}");
                if off - r > near + th {
                    control_tunnels += 1;
                }
                if ccd - r > near + th {
                    failures.push(format!("{rate} Hz {speed} m/s {th} m: tunnelled (max x {ccd})"));
                } else if ccd + r > near + 0.5 * th + 0.05 {
                    failures.push(format!("{rate} Hz {speed} m/s {th} m: embedded past mid-wall (max x {ccd})"));
                }
            }
        }
    }
    assert!(control_tunnels > 0, "Metal {wall:?}: no control case tunnelled; the suite is not sensitive");
    assert!(failures.is_empty(), "Metal {wall:?} CCD failures:\n{}", failures.join("\n"));
}

#[test]
fn metal_ccd_balls_stop_at_static_slabs() {
    run_matrix(Wall::Slab, &[0.05, 0.2, 1.0]);
}

#[test]
fn metal_ccd_balls_stop_at_brick_walls() {
    run_matrix(Wall::Bricks, &[0.05, 0.2]);
}

/// 40 m/s debris (0.1 m chunks) showered onto a 0.2 m slab, with gravity: no
/// CCD chunk falls through.
#[test]
fn metal_ccd_debris_shower_stays_on_the_slab() {
    for rate in [60.0f32, 120.0] {
        let fell = |ccd: bool| {
            let mut w = world(rate, ccd, -10.0);
            static_box(&mut w, [0.0, -0.1, 0.0], [3.0, 0.1, 3.0]);
            let chunks: Vec<u32> = (0..64)
                .map(|i| {
                    let (x, z, y) = ((i % 8) as f32 * 0.5 - 1.75, (i / 8) as f32 * 0.5 - 1.75, 2.0 + 0.13 * (i % 5) as f32);
                    dynamic_box(&mut w, [x, y, z], [0.05, 0.05, 0.05], 0.3, [0.0, -40.0, 0.0], if ccd { BODY_CCD } else { 0 })
                })
                .collect();
            for _ in 0..(rate as u32) {
                w.step().expect("step");
            }
            chunks.iter().filter(|&&c| w.body(c).position[1] < -0.2).count()
        };
        let (off, ccd) = (fell(false), fell(true));
        println!("Metal debris {rate} Hz: fell through off {off}, ccd {ccd}");
        assert_eq!(ccd, 0, "{rate} Hz: {ccd} CCD chunks fell through the slab");
    }
}
