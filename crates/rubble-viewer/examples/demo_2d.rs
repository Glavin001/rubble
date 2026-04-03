use rand::Rng;
use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D};
use rubble_viewer::Viewer2D;
use std::f32::consts::PI;

fn rect_desc(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    angle: f32,
    mass: f32,
    friction: f32,
) -> RigidBodyDesc2D {
    RigidBodyDesc2D {
        x,
        y,
        angle,
        mass,
        friction,
        shape: ShapeDesc2D::Rect {
            half_extents: Vec2::new(width * 0.5, height * 0.5),
        },
        ..Default::default()
    }
}

fn rect_desc_with_velocity(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    angle: f32,
    mass: f32,
    friction: f32,
    vx: f32,
    vy: f32,
) -> RigidBodyDesc2D {
    RigidBodyDesc2D {
        vx,
        vy,
        ..rect_desc(x, y, width, height, angle, mass, friction)
    }
}

fn circle_desc(x: f32, y: f32, radius: f32, mass: f32, friction: f32) -> RigidBodyDesc2D {
    RigidBodyDesc2D {
        x,
        y,
        mass,
        friction,
        shape: ShapeDesc2D::Circle { radius },
        ..Default::default()
    }
}

fn capsule_desc(
    x: f32,
    y: f32,
    half_height: f32,
    radius: f32,
    mass: f32,
    friction: f32,
) -> RigidBodyDesc2D {
    RigidBodyDesc2D {
        x,
        y,
        mass,
        friction,
        shape: ShapeDesc2D::Capsule {
            half_height,
            radius,
        },
        ..Default::default()
    }
}

fn scene_empty() -> Vec<RigidBodyDesc2D> {
    Vec::new()
}

fn scene_ground() -> Vec<RigidBodyDesc2D> {
    vec![rect_desc(0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5)]
}

fn scene_dynamic_friction() -> Vec<RigidBodyDesc2D> {
    let mut descs = vec![rect_desc(0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5)];
    for x in 0..=10 {
        let friction = 5.0 - (x as f32 / 10.0 * 5.0);
        descs.push(rect_desc_with_velocity(
            -30.0 + x as f32 * 2.0,
            0.75,
            1.0,
            0.5,
            0.0,
            1.0,
            friction,
            10.0,
            0.0,
        ));
    }
    descs
}

fn scene_static_friction() -> Vec<RigidBodyDesc2D> {
    let mut descs = vec![rect_desc(0.0, 0.0, 100.0, 1.0, PI / 6.0, 0.0, 1.0)];
    for y in 0..=10 {
        descs.push(rect_desc(
            0.0,
            y as f32 + 1.0,
            5.0,
            0.5,
            PI / 6.0,
            1.0,
            1.0,
        ));
    }
    descs
}

fn scene_pyramid() -> Vec<RigidBodyDesc2D> {
    const SIZE: usize = 20;

    let mut descs = vec![rect_desc(0.0, -2.0, 100.0, 0.5, 0.0, 0.0, 0.5)];
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            descs.push(rect_desc(
                x as f32 * 1.1 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                y as f32 * 0.85,
                1.0,
                0.5,
                0.0,
                1.0,
                0.5,
            ));
        }
    }
    descs
}

fn scene_cards() -> Vec<RigidBodyDesc2D> {
    let mut descs = vec![rect_desc(0.0, -2.0, 80.0, 4.0, 0.0, 0.0, 0.7)];

    let card_height = 0.2 * 2.0;
    let card_thickness = 0.001 * 2.0;
    let angle0 = 25.0 * PI / 180.0;
    let angle1 = -25.0 * PI / 180.0;
    let angle2 = 0.5 * PI;

    let mut count = 5;
    let mut x0 = 0.0;
    let mut y = card_height * 0.5 - 0.02;
    while count > 0 {
        let mut x = x0;
        for i in 0..count {
            if i != count - 1 {
                descs.push(rect_desc(
                    x + 0.25,
                    y + card_height * 0.5 - 0.02,
                    card_thickness,
                    card_height,
                    angle2,
                    1.0,
                    0.7,
                ));
            }

            descs.push(rect_desc(
                x,
                y,
                card_thickness,
                card_height,
                angle1,
                1.0,
                0.7,
            ));

            x += 0.175;

            descs.push(rect_desc(
                x,
                y,
                card_thickness,
                card_height,
                angle0,
                1.0,
                0.7,
            ));

            x += 0.175;
        }
        y += card_height - 0.04;
        x0 += 0.175;
        count -= 1;
    }

    descs
}

fn scene_stack() -> Vec<RigidBodyDesc2D> {
    let mut descs = vec![rect_desc(0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5)];
    for i in 0..20 {
        descs.push(rect_desc(0.0, i as f32 * 2.0 + 1.0, 1.0, 1.0, 0.0, 1.0, 0.5));
    }
    descs
}

fn scene_stack_ratio() -> Vec<RigidBodyDesc2D> {
    let mut descs = vec![rect_desc(0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.5)];

    let mut y = 1_i32;
    let mut size = 1_i32;
    for _ in 0..6 {
        descs.push(rect_desc(
            0.0,
            y as f32,
            size as f32,
            size as f32,
            0.0,
            1.0,
            0.5,
        ));
        y += size * 3 / 2;
        size *= 2;
    }

    descs
}

fn scene_scatter() -> Vec<RigidBodyDesc2D> {
    let world_w = 30.0_f32;
    let world_h = 20.0_f32;
    let wall = 0.25_f32;
    let columns = 40;
    let rows = 25;
    let x_start = 1.25_f32;
    let y_start = 2.0_f32;
    let x_spacing = 27.5_f32 / (columns - 1) as f32;
    let y_spacing = 16.0_f32 / (rows - 1) as f32;

    let mut descs = vec![
        rect_desc(world_w / 2.0, wall / 2.0, world_w, wall, 0.0, 0.0, 0.5),
        rect_desc(wall / 2.0, world_h / 2.0, wall, world_h, 0.0, 0.0, 0.5),
        rect_desc(
            world_w - wall / 2.0,
            world_h / 2.0,
            wall,
            world_h,
            0.0,
            0.0,
            0.5,
        ),
    ];

    let mut rng = rand::rng();
    for row in 0..rows {
        for col in 0..columns {
            let x = x_start + col as f32 * x_spacing + (rng.random::<f32>() - 0.5) * 0.05;
            let y = y_start + row as f32 * y_spacing + (rng.random::<f32>() - 0.5) * 0.05;

            match rng.random_range(0..3_u32) {
                0 => descs.push(circle_desc(
                    x,
                    y,
                    0.12 + rng.random::<f32>() * 0.08,
                    1.0,
                    0.5,
                )),
                1 => {
                    let width = 0.24 + rng.random::<f32>() * 0.16;
                    let height = 0.24 + rng.random::<f32>() * 0.16;
                    let angle = (rng.random::<f32>() - 0.5) * 0.35;
                    descs.push(rect_desc(x, y, width, height, angle, 1.0, 0.5));
                }
                _ => {
                    let half_height = 0.10 + rng.random::<f32>() * 0.06;
                    let radius = 0.06 + rng.random::<f32>() * 0.04;
                    descs.push(capsule_desc(x, y, half_height, radius, 1.0, 0.5));
                }
            }
        }
    }

    descs
}

fn main() {
    let mut viewer = Viewer2D::new(0.0, -9.81);
    let scenes = [
        ("Empty", scene_empty()),
        ("Ground", scene_ground()),
        ("Dynamic Friction", scene_dynamic_friction()),
        ("Static Friction", scene_static_friction()),
        ("Pyramid", scene_pyramid()),
        ("Cards", scene_cards()),
        ("Stack", scene_stack()),
        ("Stack Ratio", scene_stack_ratio()),
        ("Scatter", scene_scatter()),
    ];

    let mut pyramid_scene = 0;
    for (index, (name, descs)) in scenes.into_iter().enumerate() {
        let scene_idx = viewer.add_scene_descs(name, descs);
        if index == 4 {
            pyramid_scene = scene_idx;
        }
    }
    viewer.set_initial_scene(pyramid_scene);
    viewer.run();
}
