//! Export Rust-built scenes as raw GPU snapshots for the browser host.
//!
//! ```bash
//! cargo run -p engine-scenes --release --bin scene_export -- "Joints Showcase" joints
//! ```
//!
//! Writes `crates/engine-webgpu/generated/scenes/<id>.bin` (WorldSnapshot
//! records) and `<id>.json` (counts, interaction slots, joint field offsets).
//! Scenes are built on the host (`build_host`), so this needs no GPU and runs on any
//! platform; `--no-default-features` leaves out the Metal world (the Linux web build).
//! The page uploads the records as-is; scene logic stays in Rust.

use engine_scenes::build_host;
use engine_schema::*;
use glam::Vec3;
use std::mem::offset_of;
use std::path::PathBuf;

/// three-avbd DRAG_STIFFNESS: a soft world-anchored ball joint.
const DRAG_STIFFNESS: f32 = 5000.0;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pairs: Vec<(String, String)> = if args.is_empty() {
        vec![
            ("Joints Showcase".into(), "joints".into()),
            ("Ragdolls (40)".into(), "ragdolls".into()),
            ("Ragdoll Pile (10k)".into(), "ragdolls10k".into()),
            ("Hinge Chains (10k)".into(), "hingechains10k".into()),
            ("Motor Field (10k)".into(), "motorfield10k".into()),
        ]
    } else {
        args.chunks(2)
            .map(|c| {
                (
                    c[0].clone(),
                    c.get(1).cloned().unwrap_or_else(|| c[0].clone()),
                )
            })
            .collect()
    };
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/generated/scenes");
    std::fs::create_dir_all(&out)?;
    for (name, id) in pairs {
        let (world, info) = build_host(&name)?;
        let mut snap = world.snapshot();
        let render_bodies = snap.colliders.len();
        for (i, c) in snap.colliders.iter().enumerate() {
            anyhow::ensure!(
                c.body as usize == i,
                "{name}: collider {i} belongs to body {}",
                c.body
            );
        }
        anyhow::ensure!(
            render_bodies == snap.bodies.len(),
            "{name}: every body needs one collider"
        );

        // Projectile: a parked static sphere the page launches with `F`.
        let projectile = snap.bodies.len() as u32;
        let radius = 0.35f32;
        let mut ball = BodyState::dynamic(
            Vec3::new(0.0, -40.0, 0.0),
            glam::Quat::IDENTITY,
            8.0,
            Vec3::splat(0.4 * 8.0 * radius * radius),
        );
        ball.inv_mass = 0.0;
        ball.inv_inertia = [0.0; 3];
        ball.flags = 0;
        snap.bodies.push(ball);
        snap.colliders.push(Collider {
            body: projectile,
            shape: ShapeType::Sphere as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 8.0 / (4.0 / 3.0 * std::f32::consts::PI * radius.powi(3)),
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [radius, 0.0, 0.0, radius],
        });
        // Mouse body (static, no collider) and a released drag-joint slot.
        let mouse = snap.bodies.len() as u32;
        let mut m = BodyState::dynamic(Vec3::ZERO, glam::Quat::IDENTITY, 1.0, Vec3::ONE);
        m.inv_mass = 0.0;
        m.inv_inertia = [0.0; 3];
        m.flags = 0;
        snap.bodies.push(m);
        let drag_joint = snap.joints.len() as u32;
        let mut drag = JointDesc::ball(mouse, 0, Vec3::ZERO, Vec3::ZERO)
            .with_stiffness(DRAG_STIFFNESS, 0.0)
            .with_torque_arm(1.0)
            .to_gpu();
        drag.flags |= JOINT_FLAG_BROKEN;
        drag.body_a = 0;
        drag.body_b = 0;
        snap.joints.push(drag);

        let motors: Vec<_> = snap
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.flags & JOINT_FLAG_DRIVE != 0 && j.drive_stiffness == 0.0)
            .map(|(i, j)| serde_json::json!({ "id": i, "velocity": j.drive_velocity }))
            .collect();
        let servos: Vec<_> = snap
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.flags & JOINT_FLAG_DRIVE != 0 && j.drive_stiffness > 0.0)
            .map(|(i, j)| serde_json::json!({ "id": i, "target": j.drive_target }))
            .collect();
        let params = snap.params.unwrap_or_default();
        let meta = serde_json::json!({
            "name": info.name,
            "iterations": info.iterations,
            "bodies": snap.bodies.len(),
            "renderBodies": render_bodies + 1,
            "colliders": snap.colliders.len(),
            "joints": snap.joints.len(),
            "projectile": projectile,
            "mouseBody": mouse,
            "dragJoint": drag_joint,
            "motors": motors,
            "servos": servos,
            "params": { "dt": params.dt, "sleepTime": params.sleep_time, "gravityY": params.gravity_y },
            "jointLayout": {
                "stride": std::mem::size_of::<Joint>(),
                "bodyA": offset_of!(Joint, body_a),
                "bodyB": offset_of!(Joint, body_b),
                "flags": offset_of!(Joint, flags),
                "anchorB": offset_of!(Joint, anchor_b),
                "driveTarget": offset_of!(Joint, drive_target),
                "driveVelocity": offset_of!(Joint, drive_velocity),
                "brokenFlag": JOINT_FLAG_BROKEN,
            },
        });
        std::fs::write(out.join(format!("{id}.bin")), snap.to_bytes())?;
        std::fs::write(
            out.join(format!("{id}.json")),
            serde_json::to_string_pretty(&meta)?,
        )?;
        println!(
            "{id}: {} bodies, {} joints ({} motors, {} servos) -> {}",
            snap.bodies.len(),
            snap.joints.len(),
            motors.len(),
            servos.len(),
            out.join(format!("{id}.bin")).display()
        );
    }
    Ok(())
}
