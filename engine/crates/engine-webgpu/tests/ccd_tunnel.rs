//! Tunnelling suite (CCD audit, Phase 0): fast bodies against thin walls,
//! control-paired. Each case runs three ways on the same scene:
//!
//! - `Off`: no look-ahead (engine default before the audit);
//! - `LookAhead`: `FLAG_SPEC_VELOCITY` only (reach capped per collider);
//! - `Ccd`: look-ahead plus `BODY_CCD` on the projectile (uncapped reach, the
//!   fast list, time-of-impact contacts and the TOI start pose).
//!
//! The scenes exceed 512 colliders so the broadphase takes its grid path, and
//! every wall is static: a body that ends past the far face tunnelled. Controls
//! must still tunnel in the hardest cases (the suite is sensitive), and every
//! CCD case must stop at the wall.

use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};
use engine_webgpu::WgpuSim;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Off,
    LookAhead,
    Ccd,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Grid {
    Uniform,
    Hierarchical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Wall {
    /// One static box (wider than a cell: the large-static list).
    Slab,
    /// Small static bricks, each in the grid (the 27-cell stencil).
    Bricks,
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

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3]) -> u32 {
    let b = sim.add_body(body(pos, 0.0, [0.0; 3], 0, [0.0; 3]));
    let r = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    sim.add_collider(collider(b, ShapeType::Box, [half[0], half[1], half[2], r]));
    b
}

fn dynamic_sphere(sim: &mut WgpuSim, pos: [f32; 3], r: f32, mass: f32, v: [f32; 3], flags: u32) -> u32 {
    let i = 2.5 / (mass * r * r);
    let b = sim.add_body(body(pos, 1.0 / mass, [i; 3], 9 | flags, v));
    sim.add_collider(collider(b, ShapeType::Sphere, [r, 0.0, 0.0, r]));
    b
}

fn dynamic_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3], mass: f32, v: [f32; 3], flags: u32) -> u32 {
    let inv_i = [
        3.0 / (mass * (half[1] * half[1] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[1] * half[1])),
    ];
    let b = sim.add_body(body(pos, 1.0 / mass, inv_i, 9 | flags, v));
    let r = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    sim.add_collider(collider(b, ShapeType::Box, [half[0], half[1], half[2], r]));
    b
}

/// 600 small static boxes far from the action: the scene leaves the exact
/// pairwise path (≤ 512 colliders) and runs the grid.
fn filler(sim: &mut WgpuSim) {
    for i in 0..600 {
        let x = (i % 30) as f32 * 0.6 - 9.0;
        let y = (i / 30) as f32 * 0.6;
        static_box(sim, [x, y, 80.0], [0.2, 0.2, 0.2]);
    }
}

fn world(rate: f32, mode: Mode, grid: Grid) -> WgpuSim {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.dt = 1.0 / rate;
    p.gravity_y = 0.0;
    p.iterations = 8;
    if mode != Mode::Off {
        p.flags |= FLAG_SPEC_VELOCITY;
    }
    sim.set_params(p);
    sim.set_grid_levels(Some(grid == Grid::Hierarchical));
    sim
}

#[derive(Debug)]
struct Outcome {
    /// Largest centre x reached (the wall's near face is at x = 10).
    max_x: f32,
    final_x: f32,
    tunnelled: bool,
}

/// A sphere of radius `r` fired along +x at `speed` from x = 0 into a static
/// wall whose near face is at x = 10.
fn shoot(rate: f32, speed: f32, thickness: f32, r: f32, mode: Mode, grid: Grid, wall: Wall) -> Outcome {
    let mut sim = world(rate, mode, grid);
    filler(&mut sim);
    let near = 10.0;
    match wall {
        Wall::Slab => {
            static_box(&mut sim, [near + 0.5 * thickness, 0.0, 0.0], [0.5 * thickness, 1.5, 1.5]);
        }
        Wall::Bricks => {
            // 0.2 m bricks, one deep: each sits in the grid, not the large list.
            let n = 12;
            for iy in 0..n {
                for iz in 0..n {
                    let y = (iy as f32 - (n as f32 - 1.0) * 0.5) * 0.2;
                    let z = (iz as f32 - (n as f32 - 1.0) * 0.5) * 0.2;
                    static_box(&mut sim, [near + 0.5 * thickness, y, z], [0.5 * thickness, 0.1, 0.1]);
                }
            }
        }
    }
    let flags = if mode == Mode::Ccd { BODY_CCD } else { 0 };
    let ball = dynamic_sphere(&mut sim, [0.37, 0.03, 0.02], r, 5.0, [speed, 0.0, 0.0], flags);
    // Long enough to cross 25 m.
    let steps = ((25.0 / speed) * rate).ceil() as u32 + 4;
    let mut max_x = f32::MIN;
    let mut final_x = 0.0;
    for _ in 0..steps {
        sim.step().expect("step");
        let b = sim.read_body(ball);
        max_x = max_x.max(b.position[0]);
        final_x = b.position[0];
    }
    Outcome { max_x, final_x, tunnelled: max_x - r > near + thickness }
}

fn run_matrix(grid: Grid, wall: Wall, thicknesses: &[f32]) {
    let mut failures = Vec::new();
    let mut control_tunnels = 0;
    for &rate in &[60.0f32, 120.0] {
        for &speed in &[60.0f32, 140.0, 250.0] {
            for &th in thicknesses {
                for mode in [Mode::Off, Mode::LookAhead, Mode::Ccd] {
                    let o = shoot(rate, speed, th, 0.1, mode, grid, wall);
                    println!(
                        "{grid:?} {wall:?} {rate:>3} Hz {speed:>5} m/s wall {th:.2} m {mode:<9?} max x {:7.3} final x {:7.3} {}",
                        o.max_x,
                        o.final_x,
                        if o.tunnelled { "TUNNELLED" } else { "held" }
                    );
                    if mode == Mode::Ccd {
                        if o.tunnelled {
                            failures.push(format!("{rate} Hz {speed} m/s {th} m: tunnelled ({o:?})"));
                        } else if o.max_x + 0.1 > 10.0 + 0.5 * th + 0.05 {
                            failures.push(format!("{rate} Hz {speed} m/s {th} m: embedded past mid-wall ({o:?})"));
                        }
                    } else if o.tunnelled {
                        control_tunnels += 1;
                    }
                }
            }
        }
    }
    assert!(control_tunnels > 0, "{grid:?} {wall:?}: no control case tunnelled; the suite is not sensitive");
    assert!(failures.is_empty(), "{grid:?} {wall:?} CCD failures:\n{}", failures.join("\n"));
}

#[test]
fn balls_stop_at_static_slabs_uniform_grid() {
    run_matrix(Grid::Uniform, Wall::Slab, &[0.05, 0.2, 1.0]);
}

#[test]
fn balls_stop_at_static_slabs_hierarchical_grid() {
    run_matrix(Grid::Hierarchical, Wall::Slab, &[0.05, 0.2, 1.0]);
}

#[test]
fn balls_stop_at_brick_walls_uniform_grid() {
    run_matrix(Grid::Uniform, Wall::Bricks, &[0.05, 0.2]);
}

#[test]
fn balls_stop_at_brick_walls_hierarchical_grid() {
    run_matrix(Grid::Hierarchical, Wall::Bricks, &[0.05, 0.2]);
}

/// 40 m/s debris (0.1 m chunks) showered onto a 0.2 m slab, with gravity.
fn debris_shower(rate: f32, mode: Mode) -> usize {
    let mut sim = world(rate, mode, Grid::Uniform);
    let mut p = sim.params();
    p.gravity_y = -10.0;
    sim.set_params(p);
    filler(&mut sim);
    static_box(&mut sim, [0.0, -0.1, 0.0], [3.0, 0.1, 3.0]);
    let flags = if mode == Mode::Ccd { BODY_CCD } else { 0 };
    let mut chunks = Vec::new();
    for i in 0..64 {
        let x = (i % 8) as f32 * 0.5 - 1.75;
        let z = (i / 8) as f32 * 0.5 - 1.75;
        let y = 2.0 + 0.13 * (i % 5) as f32;
        chunks.push(dynamic_box(&mut sim, [x, y, z], [0.05, 0.05, 0.05], 0.3, [0.0, -40.0, 0.0], flags));
    }
    for _ in 0..(rate as u32) {
        sim.step().expect("step");
    }
    chunks.iter().filter(|&&c| sim.read_body(c).position[1] < -0.2).count()
}

#[test]
fn debris_shower_stays_on_the_slab() {
    for rate in [60.0f32, 120.0] {
        let off = debris_shower(rate, Mode::Off);
        let look = debris_shower(rate, Mode::LookAhead);
        let ccd = debris_shower(rate, Mode::Ccd);
        println!("debris {rate} Hz: fell through off {off}, look-ahead {look}, ccd {ccd}");
        assert_eq!(ccd, 0, "{rate} Hz: {ccd} CCD chunks fell through the slab");
    }
}
