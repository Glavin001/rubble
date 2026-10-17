//! Demo and benchmark scenes inspired by `three-avbd` paper/GPU showcases.
//!
//! three-avbd is **Z-up**; this engine is **Y-up**. Coordinates are converted as
//! `(x, y, z)_demo → (x, z, y)_engine`.

#[cfg(feature = "metal")]
use engine_metal::MetalWorld;
use engine_schema::scene::{HostWorld, SceneBuilder, SceneSink};
use engine_schema::{AvbdParams, BodyState, WorldCapacity, FLAG_REUSE_CONTACTS};
use glam::{Quat, Vec3};

pub mod joints;
pub use joints::*;
pub mod legacy;

#[derive(Clone, Debug)]
pub struct SceneInfo {
    pub name: &'static str,
    pub group: &'static str,
    pub iterations: u32,
    pub capacity: WorldCapacity,
}

pub type SceneBuildFn = fn(&mut SceneBuilder<'_>);

pub struct SceneDef {
    pub info: SceneInfo,
    pub build: SceneBuildFn,
}

fn yaw_y(angle: f32) -> Quat {
    // Y/Z exchange reverses handedness; rotation is an axial vector.
    Quat::from_rotation_y(-angle)
}

fn add_brick(b: &mut SceneBuilder<'_>, x: f32, y: f32, z: f32, yaw: f32) -> u32 {
    // Demo brick size [1, 0.5, 0.5] → half (0.5, 0.25, 0.25). three-avbd mass =
    // volume×density = 0.25 (was 1.0, which stopped smash balls inside the wall).
    let id = b.add_box(Vec3::new(x, y, z), Vec3::new(0.5, 0.25, 0.25), 0.25);
    let mut body = b.world_mut().body(id);
    body.orientation = yaw_y(yaw).to_array();
    b.world_mut().set_body(id, body);
    id
}

fn rolling_angular_velocity(speed: f32, radius: f32) -> Vec3 {
    Vec3::new(speed / radius, 0.0, 0.0)
}

fn add_sphere_vel(
    b: &mut SceneBuilder<'_>,
    pos: Vec3,
    radius: f32,
    mass: f32,
    vel: Vec3,
) -> u32 {
    let id = b.add_sphere(pos, radius, mass);
    let mut body = b.world_mut().body(id);
    body.linear_vel = vel.to_array();
    // Y/Z exchange negates the axial angular velocity: +v_z/r rolls along +Z.
    let speed = vel.z;
    if radius > 1e-6 && speed.abs() > 1e-6 {
        body.angular_vel = rolling_angular_velocity(speed, radius).to_array();
    }
    b.world_mut().set_body(id, body);
    id
}

/// Paper Fig. 1 — brick ring smashed by a rolling sphere.
pub fn brick_ring(b: &mut SceneBuilder<'_>, radius: f32, courses: u32, tier: u32, rows: u32) {
    for c in 0..courses {
        let depth = (rows.saturating_sub(c / tier)).max(1);
        for j in 0..depth {
            let r = radius + 0.25 + 0.52 * j as f32;
            let n = ((2.0 * std::f32::consts::PI * (r - 0.25)) / 1.02).floor().max(1.0) as u32;
            for i in 0..n {
                let a = ((i as f32 + if c % 2 == 1 { 0.5 } else { 0.0 }) / n as f32)
                    * 2.0
                    * std::f32::consts::PI;
                // demo (r cos a, r sin a, 0.75+0.5c) → (x, y=height, z)
                add_brick(
                    b,
                    r * a.cos(),
                    0.75 + 0.5 * c as f32,
                    r * a.sin(),
                    a + std::f32::consts::FRAC_PI_2,
                );
            }
        }
    }
    let ball_r = courses as f32 / 5.0;
    add_sphere_vel(
        b,
        Vec3::new(0.0, 0.5 + ball_r, -(radius + 0.52 * rows as f32 + ball_r + 2.0)),
        ball_r,
        10.0 * (4.0 / 3.0) * std::f32::consts::PI * ball_r.powi(3),
        Vec3::new(0.0, 0.0, 30.0),
    );
}

/// Matches three-avbd `bench-cases` “Brick ring 9k” (`brickRing(s, 12, 20, 2, 10)`).
pub fn brick_ring_9k(b: &mut SceneBuilder<'_>) {
    brick_ring(b, 12.0, 20, 2, 10);
}

pub fn brick_ring_28k(b: &mut SceneBuilder<'_>) {
    brick_ring(b, 40.0, 20, 2, 10);
}

pub fn brick_ring_110k(b: &mut SceneBuilder<'_>) {
    brick_ring(b, 80.0, 40, 4, 10);
}

/// Triangular brick walls (paper Fig. 3, scaled).
pub fn brick_walls(b: &mut SceneBuilder<'_>, columns: u32, rows: u32, base: u32) {
    let (pitch_x, pitch_z) = (base as f32 + 2.0, 5.0);
    for cx in 0..columns {
        for cz in 0..rows {
            let x0 = (cx as f32 - (columns as f32 - 1.0) / 2.0) * pitch_x;
            let z = (cz as f32 - (rows as f32 - 1.0) / 2.0) * pitch_z;
            for k in 0..base {
                let m = base - k;
                for i in 0..m {
                    let x = x0 + (i as f32 - (m as f32 - 1.0) / 2.0) * 1.01;
                    add_brick(b, x, 0.75 + 0.5 * k as f32, z, 0.0);
                }
            }
        }
    }
    let r = base as f32 / 7.0;
    let mass = 50.0 * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3);
    for &cx in &[2u32, 5] {
        let x = (cx as f32 - (columns as f32 - 1.0) / 2.0) * pitch_x;
        let z0 = -((rows as f32 - 1.0) / 2.0) * pitch_z - r - 12.0;
        add_sphere_vel(b, Vec3::new(x, 0.5 + r, z0), r, mass, Vec3::new(0.0, 0.0, 30.0));
    }
}

pub fn brick_walls_27k(b: &mut SceneBuilder<'_>) {
    brick_walls(b, 8, 16, 20);
}

/// Wall smash: running-bond wall + heavy ball.
pub fn wall_smash(b: &mut SceneBuilder<'_>, w: u32, h: u32) {
    for z in 0..h {
        let offset = if z % 2 == 0 { 0.0 } else { 0.5 };
        for x in 0..w {
            for y in 0..2 {
                add_brick(
                    b,
                    (x as f32 - w as f32 / 2.0) + offset,
                    0.75 + z as f32 * 0.5,
                    y as f32 * 0.5,
                    0.0,
                );
            }
        }
    }
    let r: f32 = 2.0;
    let mass = 20.0 * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3);
    add_sphere_vel(
        b,
        Vec3::new(0.0, 5.0, -25.0),
        r,
        mass,
        Vec3::new(0.0, 4.0, 30.0),
    );
}

pub fn wall_smash_2k(b: &mut SceneBuilder<'_>) {
    wall_smash(b, 40, 25);
}

/// Breakable wall with fixed welds (fracture strength later).
pub fn breakable_wall(b: &mut SceneBuilder<'_>, w: u32, h: u32) {
    let mut ids = vec![vec![0u32; w as usize]; h as usize];
    for z in 0..h {
        for x in 0..w {
            ids[z as usize][x as usize] =
                add_brick(b, x as f32 - w as f32 / 2.0 + 0.5, 0.75 + z as f32 * 0.5, 0.0, 0.0);
        }
    }
    for z in 0..h {
        for x in 0..w {
            if x > 0 {
                b.add_fixed_joint(
                    ids[z as usize][(x - 1) as usize],
                    ids[z as usize][x as usize],
                    Vec3::new(0.5, 0.0, 0.0),
                    Vec3::new(-0.5, 0.0, 0.0),
                );
            }
            if z > 0 {
                b.add_fixed_joint(
                    ids[(z - 1) as usize][x as usize],
                    ids[z as usize][x as usize],
                    Vec3::new(0.0, 0.25, 0.0),
                    Vec3::new(0.0, -0.25, 0.0),
                );
            }
        }
    }
    let r: f32 = 1.5;
    let mass = 20.0 * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3);
    add_sphere_vel(
        b,
        Vec3::new(0.0, 4.0, -20.0),
        r,
        mass,
        Vec3::new(0.0, 3.0, 25.0),
    );
}

pub fn breakable_wall_600(b: &mut SceneBuilder<'_>) {
    breakable_wall(b, 30, 20);
}

/// Chain mail net hung by corners, catches a ball.
pub fn chain_mail(b: &mut SceneBuilder<'_>, n: u32) {
    let s = 0.5;
    let y = 12.0;
    let mut links = vec![vec![0u32; n as usize]; n as usize];
    for x in 0..n {
        for z in 0..n {
            let corner = (x == 0 || x == n - 1) && (z == 0 || z == n - 1);
            let pos = Vec3::new(
                (x as f32 - (n as f32 - 1.0) / 2.0) * s,
                y,
                (z as f32 - (n as f32 - 1.0) / 2.0) * s,
            );
            let id = if corner {
                let body = b.world_mut().add_body(BodyState {
                    position: pos.to_array(),
                    inv_mass: 0.0,
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    inv_inertia: [0.0; 3],
                    flags: 0,
                    linear_vel: [0.0; 3],
                    sleep_timer: 0.0,
                    angular_vel: [0.0; 3],
                    generation: 0,
                    initial_pos: [0.0; 3],
                    initial_pad: 0.0,
                    initial_rot: [0.0, 0.0, 0.0, 1.0],
                });
                b.world_mut().add_collider(engine_schema::Collider {
                    body,
                    shape: engine_schema::ShapeType::Box as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 0.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: [s * 0.45, 0.05, s * 0.45, s * 0.45],
                });
                body
            } else {
                b.add_box(pos, Vec3::new(s * 0.45, 0.05, s * 0.45), 0.2)
            };
            links[x as usize][z as usize] = id;
        }
    }
    for x in 0..n {
        for z in 0..n {
            if x > 0 {
                b.add_ball_joint(
                    links[(x - 1) as usize][z as usize],
                    links[x as usize][z as usize],
                    Vec3::new(s / 2.0, 0.0, 0.0),
                    Vec3::new(-s / 2.0, 0.0, 0.0),
                );
            }
            if z > 0 {
                b.add_ball_joint(
                    links[x as usize][(z - 1) as usize],
                    links[x as usize][z as usize],
                    Vec3::new(0.0, 0.0, s / 2.0),
                    Vec3::new(0.0, 0.0, -s / 2.0),
                );
            }
        }
    }
    let r = 2.0;
    b.add_sphere(Vec3::new(0.0, y + 6.0, 0.0), r, 10.0 * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3));
}

pub fn chain_mail_1_6k(b: &mut SceneBuilder<'_>) {
    chain_mail(b, 40);
}

pub fn heavy_pendulum(b: &mut SceneBuilder<'_>, links: u32, ratio: f32) {
    let len = 0.5;
    let top = links as f32 * len + 5.0;
    let anchor = {
        let body = b.world_mut().add_body(BodyState {
            position: [0.0, top, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        });
        b.world_mut().add_collider(engine_schema::Collider {
            body,
            shape: engine_schema::ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.25, 0.25, 0.25, 0.25],
        });
        body
    };
    let mut prev = anchor;
    let mut prev_off = 0.25;
    for i in 0..links {
        let id = b.add_capsule(
            Vec3::new(0.25 + len * (i as f32 + 0.5), top, 0.0),
            0.05,
            len * 0.45,
            0.05,
        );
        b.add_ball_joint(
            prev,
            id,
            Vec3::new(prev_off, 0.0, 0.0),
            Vec3::new(-len / 2.0, 0.0, 0.0),
        );
        prev = id;
        prev_off = len / 2.0;
    }
    let block = 1.5;
    let link_mass = len * 0.01;
    let mass = ratio * link_mass;
    let weight = b.add_box(
        Vec3::new(0.25 + len * links as f32 + block / 2.0, top, 0.0),
        Vec3::splat(block * 0.5),
        mass,
    );
    b.add_ball_joint(
        prev,
        weight,
        Vec3::new(len / 2.0, 0.0, 0.0),
        Vec3::new(-block / 2.0, 0.0, 0.0),
    );
}

pub fn heavy_pendulum_default(b: &mut SceneBuilder<'_>) {
    heavy_pendulum(b, 50, 50_000.0);
}

pub fn box_pile(b: &mut SceneBuilder<'_>, n: u32, h: u32) {
    let mut seed = 12345u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as f32) / (u32::MAX as f32)
    };
    // Generate in three-avbd (z,x,y) order so sizes/orients match, then upload in
    // Morton order (their spatialSort default) for cache-friendly body adjacency.
    let mut specs: Vec<(Vec3, Vec3, Quat)> = Vec::with_capacity((n * n * h) as usize);
    for z in 0..h {
        for x in 0..n {
            for y in 0..n {
                let half = Vec3::new(
                    0.25 + rand() * 0.35,
                    0.25 + rand() * 0.35,
                    0.25 + rand() * 0.35,
                );
                let pos = Vec3::new(
                    (x as f32 - n as f32 / 2.0) * 1.6,
                    2.0 + z as f32 * 1.6,
                    (y as f32 - n as f32 / 2.0) * 1.6,
                );
                let q = Quat::from_xyzw(
                    rand() - 0.5,
                    rand() - 0.5,
                    rand() - 0.5,
                    rand() - 0.5,
                )
                .normalize();
                specs.push((pos, half, q));
            }
        }
    }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (pos, _, _) in &specs {
        lo = lo.min(*pos);
        hi = hi.max(*pos);
    }
    let spread = |mut x: u32| -> u32 {
        x &= 0x3ff;
        x = (x | (x << 16)) & 0x30000ff;
        x = (x | (x << 8)) & 0x300f00f;
        x = (x | (x << 4)) & 0x30c30c3;
        x = (x | (x << 2)) & 0x9249249;
        x
    };
    let mut order: Vec<usize> = (0..specs.len()).collect();
    order.sort_by_key(|&i| {
        let p = specs[i].0;
        let q = |k: usize| -> u32 {
            let t = ((p[k] - lo[k]) / (hi[k] - lo[k]).max(1e-9) * 1024.0).floor() as i32;
            t.clamp(0, 1023) as u32
        };
        spread(q(0)) | (spread(q(1)) << 1) | (spread(q(2)) << 2)
    });
    for i in order {
        let (pos, half, q) = specs[i];
        // three-avbd: mass = sx·sy·sz·density with density=1 (full extents).
        let mass = 8.0 * half.x * half.y * half.z;
        let id = b.add_box(pos, half, mass);
        let mut body = b.world_mut().body(id);
        body.orientation = q.to_array();
        b.world_mut().set_body(id, body);
    }
}

pub fn box_pile_4k(b: &mut SceneBuilder<'_>) {
    box_pile(b, 20, 10);
}

pub fn box_pile_32k(b: &mut SceneBuilder<'_>) {
    box_pile(b, 40, 20);
}

pub fn box_columns(b: &mut SceneBuilder<'_>, n: u32, h: u32) {
    // three-avbd boxColumns (Z-up → Y-up): unit boxes at y = 1+layer so the first
    // course sits on the ground slab top (y=0.5). Centre 0.5 penetrates 0.5 m.
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                b.add_box(
                    Vec3::new(
                        (x as f32 - n as f32 / 2.0) * 1.5,
                        1.0 + y as f32,
                        (z as f32 - n as f32 / 2.0) * 1.5,
                    ),
                    Vec3::splat(0.5),
                    1.0,
                );
            }
        }
    }
}

pub fn box_columns_100k(b: &mut SceneBuilder<'_>) {
    box_columns(b, 100, 10);
}

pub fn pyramid(b: &mut SceneBuilder<'_>, levels: u32) {
    // Match three-avbd brickPyramid: centres at 0.75 + level*0.5 (no pre-sink). Collision
    // margin seats them; FINDINGS: pre-sink did not fix wall buckle and skewed first C0.
    for level in 0..levels {
        let n = levels - level;
        for x in 0..n {
            for z in 0..n {
                add_brick(
                    b,
                    (x as f32 - (n as f32 - 1.0) / 2.0) * 1.05,
                    0.75 + level as f32 * 0.5,
                    (z as f32 - (n as f32 - 1.0) / 2.0) * 0.55,
                    0.0,
                );
            }
        }
    }
}

pub fn stack(b: &mut SceneBuilder<'_>, n: u32) {
    for i in 0..n {
        b.add_box(Vec3::new(0.0, 0.5 + i as f32 * 1.05, 0.0), Vec3::splat(0.5), 1.0);
    }
}

fn hanging_sphere_rope(b: &mut SceneBuilder<'_>, links: u32, start_y: f32, heavy_end: bool) {
    let pin = {
        let body = b.world_mut().add_body(BodyState {
            position: [0.0, start_y, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        });
        b.world_mut().add_collider(engine_schema::Collider {
            body,
            shape: engine_schema::ShapeType::Sphere as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.15, 0.0, 0.0, 0.15],
        });
        body
    };
    let mut prev = pin;
    let spacing = 0.45f32;
    for i in 0..links {
        let id = b.add_sphere(
            Vec3::new(0.0, start_y - spacing * (i as f32 + 1.0), 0.0),
            0.18,
            0.35,
        );
        b.add_ball_joint(
            prev,
            id,
            Vec3::new(0.0, -0.15, 0.0),
            Vec3::new(0.0, 0.18, 0.0),
        );
        prev = id;
    }
    if heavy_end {
        let y = b.world_mut().body(prev).position[1];
        let block = b.add_box(Vec3::new(0.0, y - 0.8, 0.0), Vec3::splat(0.4), 6.0);
        b.add_ball_joint(prev, block, Vec3::new(0.0, -0.18, 0.0), Vec3::new(0.0, 0.4, 0.0));
    }
}

pub fn rope_scene(b: &mut SceneBuilder<'_>) {
    hanging_sphere_rope(b, 16, 8.0, false);
}

pub fn heavy_rope(b: &mut SceneBuilder<'_>) {
    hanging_sphere_rope(b, 16, 10.0, true);
}

pub fn jointed_drop_small(b: &mut SceneBuilder<'_>) {
    // Scaled-down jointed plates on a net (full 34k is available as jointed_drop_34k).
    let net = 16u32;
    let s = 0.5;
    let y_net = 4.0;
    let mut links = vec![vec![0u32; net as usize]; net as usize];
    for x in 0..net {
        for z in 0..net {
            let edge = x == 0 || z == 0 || x == net - 1 || z == net - 1;
            let pos = Vec3::new(
                (x as f32 - (net as f32 - 1.0) / 2.0) * s,
                y_net,
                (z as f32 - (net as f32 - 1.0) / 2.0) * s,
            );
            links[x as usize][z as usize] = if edge {
                let body = b.world_mut().add_body(BodyState {
                    position: pos.to_array(),
                    inv_mass: 0.0,
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    inv_inertia: [0.0; 3],
                    flags: 0,
                    linear_vel: [0.0; 3],
                    sleep_timer: 0.0,
                    angular_vel: [0.0; 3],
                    generation: 0,
                    initial_pos: [0.0; 3],
                    initial_pad: 0.0,
                    initial_rot: [0.0, 0.0, 0.0, 1.0],
                });
                b.world_mut().add_collider(engine_schema::Collider {
                    body,
                    shape: engine_schema::ShapeType::Box as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 0.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: [s * 0.45, 0.05, s * 0.45, s * 0.45],
                });
                body
            } else {
                b.add_box(pos, Vec3::new(s * 0.45, 0.05, s * 0.45), 0.15)
            };
        }
    }
    for x in 0..net {
        for z in 0..net {
            if x > 0 {
                b.add_ball_joint(
                    links[(x - 1) as usize][z as usize],
                    links[x as usize][z as usize],
                    Vec3::new(s / 2.0, 0.0, 0.0),
                    Vec3::new(-s / 2.0, 0.0, 0.0),
                );
            }
            if z > 0 {
                b.add_ball_joint(
                    links[x as usize][(z - 1) as usize],
                    links[x as usize][z as usize],
                    Vec3::new(0.0, 0.0, s / 2.0),
                    Vec3::new(0.0, 0.0, -s / 2.0),
                );
            }
        }
    }
    // Drop a few jointed 3×3×2 cubes
    let c = 0.4;
    for gx in 0..3 {
        for gz in 0..3 {
            let origin = Vec3::new((gx as f32 - 1.0) * 3.0, y_net + 5.0, (gz as f32 - 1.0) * 3.0);
            let mut cells = Vec::new();
            for x in 0..3 {
                for y in 0..2 {
                    for z in 0..3 {
                        cells.push(b.add_box(
                            origin + Vec3::new((x as f32 - 1.0) * c, y as f32 * c, (z as f32 - 1.0) * c),
                            Vec3::splat(c * 0.5),
                            0.5,
                        ));
                    }
                }
            }
            let at = |x: usize, y: usize, z: usize| cells[(x * 2 + y) * 3 + z];
            for x in 0..3 {
                for y in 0..2 {
                    for z in 0..3 {
                        if x > 0 {
                            b.add_ball_joint(
                                at(x - 1, y, z),
                                at(x, y, z),
                                Vec3::new(c / 2.0, 0.0, 0.0),
                                Vec3::new(-c / 2.0, 0.0, 0.0),
                            );
                        }
                        if y > 0 {
                            b.add_ball_joint(
                                at(x, y - 1, z),
                                at(x, y, z),
                                Vec3::new(0.0, c / 2.0, 0.0),
                                Vec3::new(0.0, -c / 2.0, 0.0),
                            );
                        }
                        if z > 0 {
                            b.add_ball_joint(
                                at(x, y, z - 1),
                                at(x, y, z),
                                Vec3::new(0.0, 0.0, c / 2.0),
                                Vec3::new(0.0, 0.0, -c / 2.0),
                            );
                        }
                    }
                }
            }
        }
    }
}

// ---- convex hull scenes ---------------------------------------------------------

fn scene_rand(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

/// A small library of rubble-like hulls: random clouds (8–24 points), prisms
/// and a 64-vertex bipyramid. Shared hull records exercise arena dedup.
fn hull_library(seed: &mut u32, size: f32) -> Vec<engine_schema::convex::ConvexHull> {
    use engine_schema::convex::ConvexHull;
    let mut hulls = Vec::new();
    for k in 0..12 {
        let n = 8 + (k % 5) * 4;
        let pts: Vec<Vec3> = (0..n)
            .map(|_| {
                Vec3::new(scene_rand(seed) - 0.5, (scene_rand(seed) - 0.5) * 0.7, scene_rand(seed) - 0.5) * size
            })
            .collect();
        hulls.push(ConvexHull::from_points(&pts).expect("rubble hull"));
    }
    for sides in [3usize, 5, 6, 8] {
        let pts: Vec<Vec3> = (0..2 * sides)
            .map(|i| {
                let t = (i % sides) as f32 / sides as f32 * std::f32::consts::TAU;
                Vec3::new(t.cos() * 0.5, if i < sides { -0.35 } else { 0.35 }, t.sin() * 0.5) * size
            })
            .collect();
        hulls.push(ConvexHull::from_points(&pts).expect("prism hull"));
    }
    let mut ring: Vec<Vec3> = (0..62)
        .map(|i| {
            let t = i as f32 / 62.0 * std::f32::consts::TAU;
            Vec3::new(t.cos() * 0.5, 0.0, t.sin() * 0.5) * size
        })
        .collect();
    ring.extend([Vec3::Y * 0.45 * size, -Vec3::Y * 0.45 * size]);
    hulls.push(ConvexHull::from_points(&ring).expect("bipyramid hull"));
    hulls
}

/// Rubble dropped in a jittered grid onto a ground slab. `children` hulls per
/// body form compounds (offset in the body frame, so openings stay open).
pub fn convex_pile(b: &mut SceneBuilder<'_>, n: u32, h: u32, children: u32, size: f32) {
    let mut seed = 2026u32;
    let lib = hull_library(&mut seed, size);
    b.add_ground_box_extent(0.0, (n as f32 * size * 1.6).max(50.0));
    let spacing = size * if children > 1 { 2.4 } else { 1.5 };
    for y in 0..h {
        for z in 0..n {
            for x in 0..n {
                let pos = Vec3::new(
                    (x as f32 - n as f32 / 2.0) * spacing,
                    1.0 + y as f32 * spacing,
                    (z as f32 - n as f32 / 2.0) * spacing,
                );
                let rot = Quat::from_xyzw(
                    scene_rand(&mut seed) - 0.5,
                    scene_rand(&mut seed) - 0.5,
                    scene_rand(&mut seed) - 0.5,
                    scene_rand(&mut seed) - 0.5,
                )
                .normalize();
                let hulls: Vec<_> = (0..children)
                    .map(|k| {
                        let mut hull = lib[(scene_rand(&mut seed) * lib.len() as f32) as usize % lib.len()].clone();
                        if children > 1 {
                            let off = Vec3::new(
                                if k & 1 == 0 { -0.45 } else { 0.45 },
                                if k & 2 == 0 { 0.0 } else { 0.6 },
                                0.0,
                            ) * size;
                            for v in &mut hull.vertices {
                                *v += off;
                            }
                            for f in &mut hull.faces {
                                f.d += f.normal.dot(off);
                            }
                        }
                        hull
                    })
                    .collect();
                b.add_convex_hull(&hulls, pos, rot, 1.0);
            }
        }
    }
}

pub fn convex_pile_10k(b: &mut SceneBuilder<'_>) {
    convex_pile(b, 25, 16, 1, 0.6);
}

pub fn convex_compounds_10k(b: &mut SceneBuilder<'_>) {
    convex_pile(b, 25, 16, 4, 0.6);
}

pub fn convex_rubble_30k(b: &mut SceneBuilder<'_>) {
    convex_pile(b, 40, 19, 1, 0.35);
}

fn convex_cap(bodies: u32, colliders: u32) -> WorldCapacity {
    WorldCapacity {
        max_bodies: bodies.next_power_of_two().max(4096),
        max_colliders: colliders.next_power_of_two().max(8192),
        max_contacts: (colliders * 24).next_power_of_two().max(32_768),
        max_joints: 8192,
        max_candidates: (colliders * 32).next_power_of_two().max(65_536),
        max_hull_vertices: 65_536,
        ..WorldCapacity::default()
    }
}

fn cap(bodies: u32) -> WorldCapacity {
    WorldCapacity {
        max_bodies: bodies.next_power_of_two().max(4096),
        max_colliders: (bodies * 2).next_power_of_two().max(8192),
        max_contacts: (bodies * 16).next_power_of_two().max(32_768),
        max_joints: (bodies * 4).next_power_of_two().max(8192),
        max_candidates: (bodies * 32).next_power_of_two().max(65_536),
        ..WorldCapacity::default()
    }
}

pub fn all_scenes() -> Vec<SceneDef> {
    vec![
        SceneDef {
            info: SceneInfo {
                name: "Brick Ring (9k)",
                group: "Paper scenes (GPU)",
                iterations: 4,
                capacity: cap(16_384),
            },
            build: brick_ring_9k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Brick Ring (28k)",
                group: "Paper scenes (GPU)",
                iterations: 4,
                capacity: cap(40_000),
            },
            build: brick_ring_28k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Brick Ring (110k)",
                group: "Paper scenes (GPU)",
                iterations: 4,
                capacity: WorldCapacity::scale(),
            },
            build: brick_ring_110k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Brick Walls (27k)",
                group: "Paper scenes (GPU)",
                iterations: 3,
                capacity: cap(40_000),
            },
            build: brick_walls_27k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Wall Smash (2k)",
                group: "Paper scenes (GPU)",
                // 8 iters: 120 FPS on M3 Max; 10 was ~8.5–9.5 ms p50 at smash peak.
                iterations: 8,
                // small() max_contacts=32k overflows mid-smash (~20k contacts + spikes).
                capacity: cap(4_096),
            },
            build: wall_smash_2k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Breakable Wall (600)",
                group: "Paper scenes (GPU)",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: breakable_wall_600,
        },
        SceneDef {
            info: SceneInfo {
                name: "Chain Mail (1.6k)",
                group: "Paper scenes (GPU)",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: chain_mail_1_6k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Heavy Pendulum",
                group: "Paper scenes (GPU)",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: heavy_pendulum_default,
        },
        SceneDef {
            info: SceneInfo {
                name: "Convex Pile (10k)",
                group: "Convex hulls (GPU)",
                iterations: 4,
                capacity: convex_cap(10_001, 10_001),
            },
            build: convex_pile_10k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Convex Compounds (10k)",
                group: "Convex hulls (GPU)",
                iterations: 4,
                capacity: convex_cap(10_001, 40_001),
            },
            build: convex_compounds_10k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Convex Rubble (30k)",
                group: "Convex hulls (GPU)",
                iterations: 4,
                capacity: convex_cap(30_401, 30_401),
            },
            build: convex_rubble_30k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Box Pile (4k)",
                group: "Scale tests (GPU)",
                // 4 matches three-avbd settled-pile bench; 10+ ramps k harder and left more deep C0.
                iterations: 4,
                capacity: cap(8_192),
            },
            build: box_pile_4k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Box Pile (32k)",
                group: "Scale tests (GPU)",
                // three-avbd settled pile bench uses 4.
                iterations: 4,
                capacity: cap(65_536),
            },
            build: box_pile_32k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Box Columns (100k)",
                group: "Scale tests (GPU)",
                // three-avbd steady tier uses 4.
                iterations: 4,
                capacity: WorldCapacity::scale(),
            },
            build: box_columns_100k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Jointed Drop (small)",
                group: "Scale tests (GPU)",
                iterations: 10,
                capacity: WorldCapacity::default(),
            },
            build: jointed_drop_small,
        },
        SceneDef {
            info: SceneInfo {
                name: "Pyramid",
                group: "Contacts & friction",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: |b| pyramid(b, 12),
        },
        SceneDef {
            info: SceneInfo {
                name: "Pyramid 30",
                group: "Contacts & friction",
                iterations: 10,
                capacity: cap(16_384),
            },
            build: |b| pyramid(b, 30),
        },
        SceneDef {
            info: SceneInfo {
                name: "Stack",
                group: "Contacts & friction",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: |b| stack(b, 20),
        },
        SceneDef {
            info: SceneInfo {
                name: "Rope",
                group: "Joints & springs",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: rope_scene,
        },
        SceneDef {
            info: SceneInfo {
                name: "Heavy Rope",
                group: "Joints & springs",
                iterations: 12,
                capacity: WorldCapacity::small(),
            },
            build: heavy_rope,
        },
        SceneDef {
            info: SceneInfo {
                name: "Joints Showcase",
                group: "Joints & springs",
                iterations: 10,
                capacity: cap(4096),
            },
            build: joints_showcase,
        },
        SceneDef {
            info: SceneInfo {
                name: "Ragdolls (40)",
                group: "Joints & springs",
                iterations: 10,
                capacity: cap(4096),
            },
            build: |b| ragdoll_pile(b, 40),
        },
        SceneDef {
            info: SceneInfo {
                name: "Ragdoll Pile (10k)",
                group: "Joints & springs",
                iterations: 8,
                capacity: cap(16_384),
            },
            build: |b| ragdoll_pile(b, 1000),
        },
        SceneDef {
            info: SceneInfo {
                name: "Hinge Chains (10k)",
                group: "Joints & springs",
                iterations: 8,
                capacity: cap(16_384),
            },
            build: |b| hinge_chains(b, 400, 25),
        },
        SceneDef {
            info: SceneInfo {
                name: "Motor Field (10k)",
                group: "Joints & springs",
                iterations: 8,
                capacity: cap(16_384),
            },
            build: |b| motor_field(b, 2500),
        },
        SceneDef {
            info: SceneInfo {
                name: "Sphere Pile (1k)",
                group: "Scale tests (GPU)",
                iterations: 10,
                capacity: WorldCapacity::small(),
            },
            build: |b| b.pile_spheres(1000, 0.35),
        },
        SceneDef {
            info: SceneInfo {
                name: "Sphere Pile (10k)",
                group: "Scale tests (GPU)",
                iterations: 8,
                capacity: WorldCapacity::default(),
            },
            build: |b| b.pile_spheres(10_000, 0.3),
        },
        SceneDef {
            info: SceneInfo {
                name: "Empty",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(64),
            },
            build: legacy::empty,
        },
        SceneDef {
            info: SceneInfo {
                name: "Ground",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(64),
            },
            build: legacy::ground,
        },
        SceneDef {
            info: SceneInfo {
                name: "Dynamic Friction",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(64),
            },
            build: legacy::dynamic_friction,
        },
        SceneDef {
            info: SceneInfo {
                name: "Static Friction",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(64),
            },
            build: legacy::static_friction,
        },
        SceneDef {
            info: SceneInfo {
                name: "Legacy Pyramid",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(256),
            },
            build: legacy::pyramid,
        },
        SceneDef {
            info: SceneInfo {
                name: "Legacy Stack",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(64),
            },
            build: legacy::stack,
        },
        SceneDef {
            info: SceneInfo {
                name: "Stack Ratio",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(64),
            },
            build: legacy::stack_ratio,
        },
        SceneDef {
            info: SceneInfo {
                name: "Scatter",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(3001),
            },
            build: legacy::scatter,
        },
        SceneDef {
            info: SceneInfo {
                name: "Scatter Boxes",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(4001),
            },
            build: legacy::scatter_boxes,
        },
        SceneDef {
            info: SceneInfo {
                name: "Grid Boxes",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(1537),
            },
            build: legacy::grid_boxes,
        },
        SceneDef {
            info: SceneInfo {
                name: "Slanted Grid",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(3169),
            },
            build: legacy::slanted_grid_boxes,
        },
        SceneDef {
            info: SceneInfo {
                name: "10k Grid",
                group: legacy::LEGACY_GROUP,
                iterations: legacy::LEGACY_ITERATIONS,
                capacity: cap(10001),
            },
            build: legacy::grid_10k_boxes,
        },
    ]
}

pub fn find_scene(name: &str) -> Option<&'static SceneDef> {
    ALL.iter()
        .find(|s| s.info.name.eq_ignore_ascii_case(name) || slug(s.info.name) == slug(name))
}

fn slug(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

use std::sync::LazyLock;
static ALL: LazyLock<Vec<SceneDef>> = LazyLock::new(all_scenes);

pub fn list_scenes() -> &'static [SceneDef] {
    &ALL
}

/// Solver settings a scene runs with, besides its bodies: shared by the Metal world
/// ([`build_world`]) and the host export ([`build_host`]), so both see the same scene.
pub struct SceneSetup {
    pub scene: &'static SceneDef,
    pub params: AvbdParams,
    /// Island gate rounds for sleeping (Metal world only; not part of a snapshot).
    pub sleep_island_rounds: Option<u32>,
    /// Friction of the default material, when the scene overrides it.
    pub default_friction: Option<f32>,
}

pub fn scene_setup(name: &str) -> anyhow::Result<SceneSetup> {
    let scene = find_scene(name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown scene {:?}; try --list",
            name
        )
    })?;
    let mut params = AvbdParams::default();
    params.iterations = scene.info.iterations;
    if scene.info.group == legacy::LEGACY_GROUP {
        params.gravity_y = legacy::LEGACY_GRAVITY_Y;
    }
    if let Ok(v) = std::env::var("PILE_ITERS") {
        if let Ok(n) = v.parse::<u32>() {
            params.iterations = n;
        }
    }
    // three-avbd defaults reuseContacts=true for both settling and impact scenes.
    if (scene.info.name.starts_with("Box Pile")
        || scene.info.name.starts_with("Pyramid")
        || scene.info.name.starts_with("Box Columns")
        || scene.info.name.starts_with("Brick Ring")
        || scene.info.name.starts_with("Brick Gables")
        || scene.info.name.starts_with("Convex"))
        && std::env::var("NO_REUSE").is_err()
    {
        params.flags |= FLAG_REUSE_CONTACTS;
    }
    // Soft β/α/sleep are Box Pile only (random drop). Pyramid/Columns keep demo defaults
    // (β=1e4, α=0.99) — three-avbd pyramid 30 holds with those.
    if scene.info.name.starts_with("Box Pile") {
        // Softer β than demo default 1e4: after gridCellSize fix, β=2e3 cuts flyers
        // (~50→~6) and KE (~100→~10) at 1200 without reopening deep C0. Override with PILE_BETA.
        params.beta_lin = 2_000.0;
        // Default α=0.99 only puts 1% of C0 into Ĉ → 32k collapse left deep<-0.05≈700
        // at step 1200. α=0.95 recovers penetration (32k deep≈70, 4k deep≈0–3) while
        // keeping iters=4. Override with PILE_ALPHA.
        params.alpha = 0.95;
        // Slightly looser/faster sleep so brief calm windows can commit an island
        // before a surface rock resets timers across the contact component.
        // 0.5 with impact-wake: near-full sleep @1200, deep≈0; was 0.25.
        params.sleep_lin = std::env::var("PILE_SLEEP_LIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        // 0.5 with impact-wake: ~29k/32k sleep, deep≈27, scene_fps ~11.3 ms (beats
        // oracle settled gpuTotal 11.8). 0.25 left ~11k awake / ~13 ms.
        params.sleep_ang = std::env::var("PILE_SLEEP_ANG")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        params.sleep_time = std::env::var("PILE_SLEEP_TIME")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.25);
    }
    if let Ok(v) = std::env::var("PILE_BETA") {
        if let Ok(b) = v.parse::<f32>() {
            params.beta_lin = b;
        }
    }
    // three-avbd finishBody has no trust clamp. TRUST=off (or PILE_TRUST=off) disables ours.
    // Pyramid 30: oracle holds with no clamp; our trustLin·r ≈ 0.175 m/primal was a suspect.
    if let Ok(v) = std::env::var("TRUST").or_else(|_| std::env::var("PILE_TRUST")) {
        if v == "off" || v == "0" {
            params.trust_lin = 1.0e6;
            params.trust_ang = 1.0e6;
        } else if let Ok(t) = v.parse::<f32>() {
            params.trust_lin = t;
        }
    }
    // Pyramid/Columns keep default trustLin unless TRUST=off. three-avbd has no clamp, but
    // with our residual first-frame seating (REST_SPEED over for ~75% of Pyramid 30 @step1)
    // unclamped corrections detonated; clamp contains while reuse/startAtRest catch up.
    if (scene.info.name.starts_with("Pyramid") || scene.info.name.starts_with("Box Columns")
        || scene.info.name.starts_with("Brick Ring")
        || scene.info.name.starts_with("Brick Gables"))
        && std::env::var("TRUST").ok().as_deref() == Some("off")
    {
        params.trust_lin = 1.0e6;
        params.trust_ang = 1.0e6;
    }
    if scene.info.name.starts_with("Box Pile") {
        // Collapse tunnels ~4k deep contacts by step 180; α=0.99 only puts 1% of C0
        // into Ĉ so recovery is glacial vs three-avbd's 94% reuse@600. PILE_ALPHA A/B.
        if let Ok(v) = std::env::var("PILE_ALPHA") {
            if let Ok(a) = v.parse::<f32>() {
                params.alpha = a;
            }
        }
        if let Ok(v) = std::env::var("PILE_DT") {
            if let Ok(dt) = v.parse::<f32>() {
                params.dt = dt;
            }
        }
        if let Ok(v) = std::env::var("PILE_MARGIN") {
            if let Ok(m) = v.parse::<f32>() {
                params.contact_offset = m;
            }
        }
    }
    let mut sleep_island_rounds = None;
    let mut default_friction = None;
    if scene.info.name.starts_with("Box Pile") {
        // Impact-wake + 1-round gate: ~19k/32k sleep @1200 with deep held. rounds=0
        // sleeps a bit less. Stacks/columns keep default 8.
        let rounds = std::env::var("PILE_ISLAND_ROUNDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        sleep_island_rounds = Some(rounds);
        // μ=0.7 settles deep C0 better than three-avbd's 0.5 on our Metal path; 0.5 left
        // more flyers. PILE_MU overrides for A/B.
        let mu = std::env::var("PILE_MU")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        default_friction = Some(mu);
    }
    Ok(SceneSetup { scene, params, sleep_island_rounds, default_friction })
}

/// Adds the scene's ground and bodies to `world`.
pub fn populate(world: &mut dyn SceneSink, scene: &SceneDef) {
    if scene.info.group == legacy::LEGACY_GROUP {
        // Legacy scenes bring their own ground.
        (scene.build)(&mut SceneBuilder::new(world));
        return;
    }
    {
        let mut b = SceneBuilder::new(world);
        // three-avbd `ground(extent)` is a static box slab (top at 0.5), not a plane.
        // Pyramid 30 collapsed on our infinite plane (contacts churned 117k→92k on step 1);
        // box-box ground matches the oracle manifold path. Ring/Pile/Columns same.
        let ring_extent = match scene.info.name {
            "Brick Ring (9k)" => Some(22.0),
            "Brick Ring (28k)" => Some(50.0),
            "Brick Ring (110k)" => Some(90.0),
            _ => None,
        };
        let extent: f32 = if let Some(extent) = ring_extent { extent }
        else if scene.info.name.starts_with("Pyramid") { 30.0 * 0.6 }
        else if scene.info.name.starts_with("Box Columns") { 100.0 * 1.5 }
        else if scene.info.name.starts_with("Box Pile") { 40.0 * 1.6 }
        else { 200.0 };
        let half_xz = (4.0 * extent).max(200.0) * 0.5;
        if ring_extent.is_some() {
            b.add_box(Vec3::ZERO, Vec3::new(half_xz, 0.5, half_xz), 0.0);
        } else {
            b.add_ground_box_extent(0.5, half_xz);
        }
        (scene.build)(&mut b);
    }
}

/// Builds `name` into a Metal world.
#[cfg(feature = "metal")]
pub fn build_world(name: &str) -> anyhow::Result<(MetalWorld, &'static SceneInfo)> {
    let setup = scene_setup(name)?;
    let mut world = MetalWorld::new(setup.scene.info.capacity.clone())?;
    engine_exec::PhysicsBackend::set_params(&mut world, setup.params);
    if let Some(rounds) = setup.sleep_island_rounds {
        world.set_sleep_island_rounds(rounds);
    }
    if let Some(mu) = setup.default_friction {
        world.set_default_friction(mu);
    }
    populate(&mut world, setup.scene);
    Ok((world, &setup.scene.info))
}

/// Builds `name` on the host, without a GPU, for export as a snapshot.
pub fn build_host(name: &str) -> anyhow::Result<(HostWorld, &'static SceneInfo)> {
    let setup = scene_setup(name)?;
    let mut world = HostWorld::new(setup.params);
    if let Some(mu) = setup.default_friction {
        world.set_default_friction(mu);
    }
    populate(&mut world, setup.scene);
    Ok((world, &setup.scene.info))
}


#[cfg(test)]
mod scene_conversion_tests {
    use super::*;

    #[test]
    fn ring_bricks_are_tangent_after_z_up_to_y_up_conversion() {
        for i in 0..97 {
            let angle = i as f32 * std::f32::consts::TAU / 97.0;
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let long_axis = yaw_y(angle + std::f32::consts::FRAC_PI_2) * Vec3::X;
            assert!(radial.dot(long_axis).abs() < 1e-6,
                "ring brick at angle {angle} points into neighbouring rings; Y/Z exchange must reverse yaw handedness");
        }
    }

    #[test]
    fn rolling_sphere_has_no_slip_at_ground_contact() {
        for radius in [4.0, 8.0, 30.0 / 7.0] {
            let contact_velocity = Vec3::new(0.0, 0.0, 30.0)
                + rolling_angular_velocity(30.0, radius).cross(Vec3::new(0.0, -radius, 0.0));
            assert!(contact_velocity.length() < 1e-6,
                "radius {radius}: sphere spins against its travel, contact velocity={contact_velocity:?}");
        }
    }
}

#[cfg(test)]
mod convex_scene_tests {
    #[test]
    fn hull_library_cooks_closed_hulls_within_limits() {
        let mut seed = 2026u32;
        for size in [0.35f32, 0.6] {
            for h in super::hull_library(&mut seed, size) {
                assert!(h.closed, "library hull adjacency certified");
                assert!(h.vertices.len() <= 64 && h.faces.len() <= 128 && h.edges.len() <= 192);
                assert!(h.mass_properties(1.0).volume > 0.0);
            }
        }
    }
}
