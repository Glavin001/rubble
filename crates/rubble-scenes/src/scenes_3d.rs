//! 3D demo scenes.

use glam::{Quat, Vec3};
use rand::Rng;
use rubble3d::{RigidBodyDesc, ShapeDesc};

#[allow(clippy::too_many_arguments)]
fn box_desc(
    x: f32,
    y: f32,
    z: f32,
    width: f32,
    height: f32,
    depth: f32,
    mass: f32,
    friction: f32,
) -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::new(x, y, z),
        mass,
        friction,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
        },
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn box_desc_with_velocity(
    x: f32,
    y: f32,
    z: f32,
    width: f32,
    height: f32,
    depth: f32,
    mass: f32,
    friction: f32,
    velocity: Vec3,
) -> RigidBodyDesc {
    RigidBodyDesc {
        linear_velocity: velocity,
        ..box_desc(x, y, z, width, height, depth, mass, friction)
    }
}

fn sphere_desc(x: f32, y: f32, z: f32, radius: f32, mass: f32, friction: f32) -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::new(x, y, z),
        mass,
        friction,
        shape: ShapeDesc::Sphere { radius },
        ..Default::default()
    }
}

fn capsule_desc(
    x: f32,
    y: f32,
    z: f32,
    half_height: f32,
    radius: f32,
    mass: f32,
    friction: f32,
) -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::new(x, y, z),
        mass,
        friction,
        shape: ShapeDesc::Capsule {
            half_height,
            radius,
        },
        ..Default::default()
    }
}

pub fn scene_empty() -> Vec<RigidBodyDesc> {
    Vec::new()
}

pub fn scene_ground() -> Vec<RigidBodyDesc> {
    vec![
        box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5),
        box_desc(0.0, 4.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5),
    ]
}

pub fn scene_dynamic_friction() -> Vec<RigidBodyDesc> {
    let mut descs = vec![box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5)];
    for x in 0..=10 {
        let friction = 5.0 - (x as f32 / 10.0 * 5.0);
        descs.push(box_desc_with_velocity(
            0.0,
            0.75,
            -30.0 + x as f32 * 2.0,
            1.0,
            0.5,
            1.0,
            1.0,
            friction,
            Vec3::new(10.0, 0.0, 0.0),
        ));
    }
    descs
}

pub fn scene_static_friction() -> Vec<RigidBodyDesc> {
    let angle = 30.0_f32.to_radians();
    let ramp_rotation = Quat::from_rotation_z(angle);
    let ramp_tangent = ramp_rotation * Vec3::X;
    let ramp_normal = ramp_rotation * Vec3::Y;
    let ramp_position = Vec3::ZERO;

    let mut descs = vec![
        box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5),
        RigidBodyDesc {
            position: ramp_position,
            rotation: ramp_rotation,
            mass: 0.0,
            friction: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 12.0),
            },
            ..Default::default()
        },
    ];

    for i in 0..=10 {
        let friction = i as f32 / 10.0 * 0.25 + 0.25;
        let z_offset = -10.0 + i as f32 * 2.0;
        let position = ramp_position
            + ramp_tangent * -12.0
            + Vec3::new(0.0, 0.0, z_offset)
            + ramp_normal * 1.05;
        descs.push(box_desc(
            position.x, position.y, position.z, 1.0, 1.0, 1.0, 1.0, friction,
        ));
    }

    descs
}

pub fn scene_pyramid() -> Vec<RigidBodyDesc> {
    const SIZE: usize = 16;

    let mut descs = vec![box_desc(0.0, -0.5, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5)];
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            descs.push(box_desc(
                x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                y as f32 * 0.85 + 0.5,
                0.0,
                1.0,
                0.5,
                0.5,
                1.0,
                0.5,
            ));
        }
    }
    descs
}

pub fn scene_stack() -> Vec<RigidBodyDesc> {
    let mut descs = vec![box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5)];
    for i in 0..10 {
        descs.push(box_desc(
            0.0,
            i as f32 * 1.5 + 1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
        ));
    }
    descs
}

pub fn scene_stack_ratio() -> Vec<RigidBodyDesc> {
    let mut descs = vec![box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5)];

    let mut top_y = 0.5;
    let mut size = 1.0;
    for _ in 0..4 {
        let half = size * 0.5;
        let center_y = top_y + half;
        descs.push(box_desc(0.0, center_y, 0.0, size, size, size, 1.0, 0.5));
        top_y = center_y + half;
        size *= 2.0;
    }

    descs
}

pub fn scene_scatter() -> Vec<RigidBodyDesc> {
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }];

    let mut rng = rand::rng();
    for _ in 0..3000 {
        let x = (rng.random::<f32>() - 0.5) * 12.0;
        let y = 3.0 + rng.random::<f32>() * 15.0;
        let z = (rng.random::<f32>() - 0.5) * 12.0;

        match rng.random_range(0..3_u32) {
            0 => descs.push(sphere_desc(
                x,
                y,
                z,
                0.3 + rng.random::<f32>() * 0.4,
                1.0,
                0.5,
            )),
            1 => {
                let size = 0.4 + rng.random::<f32>() * 0.8;
                descs.push(box_desc(x, y, z, size, size, size, 1.0, 0.5));
            }
            _ => descs.push(capsule_desc(x, y, z, 0.3, 0.2, 1.0, 0.5)),
        }
    }

    descs
}

pub fn scene_scatter_boxes() -> Vec<RigidBodyDesc> {
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }];

    let mut rng = rand::rng();
    for _ in 0..4000 {
        let x = (rng.random::<f32>() - 0.5) * 16.0;
        let y = 2.5 + rng.random::<f32>() * 20.0;
        let z = (rng.random::<f32>() - 0.5) * 16.0;
        let sx = 0.2 + rng.random::<f32>() * 0.5;
        let sy = 0.2 + rng.random::<f32>() * 0.5;
        let sz = 0.2 + rng.random::<f32>() * 0.5;
        descs.push(box_desc(x, y, z, sx, sy, sz, 1.0, 0.5));
    }

    descs
}

pub fn scene_grid_boxes() -> Vec<RigidBodyDesc> {
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }];

    // Favor vertical layers over footprint so the default camera reads this as
    // a real 3D stack instead of a mostly flat sheet.
    const NX: usize = 8;
    const NY: usize = 24;
    const NZ: usize = 8;
    let side = 0.42_f32;
    let gap = 0.08_f32;
    let pitch = side + gap;
    let half = side * 0.5;

    let ox = -((NX - 1) as f32 * pitch) * 0.5;
    let oz = -((NZ - 1) as f32 * pitch) * 0.5;
    let base_y = half + 0.03;

    let mut rng = rand::rng();
    for j in 0..NY {
        for i in 0..NX {
            for k in 0..NZ {
                let jitter = 0.012;
                let x = ox + i as f32 * pitch + (rng.random::<f32>() - 0.5) * jitter;
                let y = base_y + j as f32 * pitch + (rng.random::<f32>() - 0.5) * jitter;
                let z = oz + k as f32 * pitch + (rng.random::<f32>() - 0.5) * jitter;
                descs.push(box_desc(x, y, z, side, side, side, 1.0, 0.5));
            }
        }
    }

    descs
}

pub fn scene_grid_10k_boxes() -> Vec<RigidBodyDesc> {
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }];

    const NX: usize = 20;
    const NY: usize = 25;
    const NZ: usize = 20; // 20 * 25 * 20 = 10_000

    let side = 0.42_f32;
    let gap = 0.08_f32;
    let pitch = side + gap;
    let half = side * 0.5;

    let ox = -((NX - 1) as f32 * pitch) * 0.5;
    let oz = -((NZ - 1) as f32 * pitch) * 0.5;
    let base_y = half + 0.03;

    for j in 0..NY {
        for i in 0..NX {
            for k in 0..NZ {
                let x = ox + i as f32 * pitch;
                let y = base_y + j as f32 * pitch;
                let z = oz + k as f32 * pitch;
                descs.push(box_desc(x, y, z, side, side, side, 1.0, 0.5));
            }
        }
    }

    descs
}

pub fn scene_slanted_grid_boxes() -> Vec<RigidBodyDesc> {
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }];

    const NX: usize = 12;
    const NY: usize = 22;
    const NZ: usize = 12;
    let side = 0.42_f32;
    let gap = 0.08_f32;
    let pitch = side + gap;
    let half = side * 0.5;

    let ox = -((NX - 1) as f32 * pitch) * 0.5;
    let oz = -((NZ - 1) as f32 * pitch) * 0.5;
    let base_y = half + 0.03;

    let layer_shift = Vec3::new(0.038, 0.0, 0.026);

    for j in 0..NY {
        let shift = layer_shift * j as f32;
        for i in 0..NX {
            for k in 0..NZ {
                let x = ox + i as f32 * pitch + shift.x;
                let y = base_y + j as f32 * pitch;
                let z = oz + k as f32 * pitch + shift.z;
                descs.push(box_desc(x, y, z, side, side, side, 1.0, 0.5));
            }
        }
    }

    descs
}
