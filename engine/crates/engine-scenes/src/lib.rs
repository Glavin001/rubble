//! Demo and benchmark scenes inspired by `three-avbd` paper/GPU showcases.
//!
//! three-avbd is **Z-up**; this engine is **Y-up**. Coordinates are converted as
//! `(x, y, z)_demo → (x, z, y)_engine`.

use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, BodyState, WorldCapacity, FLAG_REUSE_CONTACTS};
use glam::{Quat, Vec3};

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
    Quat::from_rotation_y(angle)
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
    // three-avbd rollingBall: ω = (−v_z/r, 0, 0) for +Z travel (Y-up).
    let speed = vel.z;
    if radius > 1e-6 && speed.abs() > 1e-6 {
        body.angular_vel = [-speed / radius, 0.0, 0.0];
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
                    generation: 1,
                });
                b.world_mut().add_collider(engine_schema::Collider {
                    body,
                    shape: engine_schema::ShapeType::Box as u32,
                    material: 0,
                    generation: 1,
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
            generation: 1,
        });
        b.world_mut().add_collider(engine_schema::Collider {
            body,
            shape: engine_schema::ShapeType::Box as u32,
            material: 0,
            generation: 1,
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
                // three-avbd: mass = sx·sy·sz·density with density=1 (full extents).
                let mass = 8.0 * half.x * half.y * half.z;
                let id = b.add_box(pos, half, mass);
                let q = Quat::from_xyzw(
                    rand() - 0.5,
                    rand() - 0.5,
                    rand() - 0.5,
                    rand() - 0.5,
                )
                .normalize();
                let mut body = b.world_mut().body(id);
                body.orientation = q.to_array();
                b.world_mut().set_body(id, body);
            }
        }
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
            generation: 1,
        });
        b.world_mut().add_collider(engine_schema::Collider {
            body,
            shape: engine_schema::ShapeType::Sphere as u32,
            material: 0,
            generation: 1,
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
                    generation: 1,
                });
                b.world_mut().add_collider(engine_schema::Collider {
                    body,
                    shape: engine_schema::ShapeType::Box as u32,
                    material: 0,
                    generation: 1,
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
                name: "Box Pile (4k)",
                group: "Scale tests (GPU)",
                iterations: 10,
                capacity: cap(8_192),
            },
            build: box_pile_4k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Box Pile (32k)",
                group: "Scale tests (GPU)",
                iterations: 8,
                capacity: cap(65_536),
            },
            build: box_pile_32k,
        },
        SceneDef {
            info: SceneInfo {
                name: "Box Columns (100k)",
                group: "Scale tests (GPU)",
                iterations: 6,
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

pub fn build_world(name: &str) -> anyhow::Result<(MetalWorld, &'static SceneInfo)> {
    let scene = find_scene(name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown scene {:?}; try --list",
            name
        )
    })?;
    let mut world = MetalWorld::new(scene.info.capacity.clone())?;
    let mut params = AvbdParams::default();
    params.iterations = scene.info.iterations;
    // reuseContacts helps random Box Pile settle; on Metal it slows Brick Ring smash.
    if scene.info.name.starts_with("Box Pile") {
        params.flags |= FLAG_REUSE_CONTACTS;
    }
    world.set_params(params);
    {
        let mut b = SceneBuilder::new(&mut world);
        // Seat content on y=0.5 like three-avbd's ground top. A plane (not a box
        // slab) keeps the large-scene broadphase simple: every dynamic already
        // probes planes, and Box Columns 100k settles/sleeps on it. Brick Ring
        // must not air-drop 0.5 m from a plane at y=0.
        b.add_ground_plane(0.5);
        (scene.build)(&mut b);
    }
    Ok((world, &scene.info))
}
