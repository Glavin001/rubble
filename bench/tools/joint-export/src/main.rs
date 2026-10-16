//! Export rubble's D6 joint scenes as benchmark scenes (format v3).
//!
//! `cargo run --release -p bench-joint-export -- [--out DIR] [id,...]`
//!
//! The scenes are built by `engine-scenes` (the same builders the demos use), so every engine
//! gets exactly rubble's bodies and joint frames. three-avbd has no hinge, cone, limit or motor
//! rows, so these scenes compare rubble with PhysX's GPU D6 joints (and rapier on the CPU).
//!
//! v3 = v2 header and bodies, with 32-word joint records:
//! `[a, b, kind (0 ball, 1 D6), flags (bit0 collide), anchorA xyz, frameA xyzw, anchorB xyz,
//!   frameB xyzw, motion (2 bits per DOF X Y Z twist swing1 swing2: 0 locked 1 limited
//!   2 free), twistLo, twistHi, swingY, swingZ, linLoX, linHiX, driveDof (~0 = none),
//!   driveVelocity, driveForceLimit, driveStiffness, driveDamping, driveTarget, bits
//!   (bit0 = limited swings form one elliptical cone)]`.

use anyhow::{ensure, Context, Result};
use engine_schema::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const PROTOCOL: (u32, u32, u32, u32) = (2, 3, 5, 10); // rounds, warm steps, batches, per batch
const LEAD: u32 = 60;
const STEPS: u32 = 300;

/// (bench id, engine-scenes name, what the scene checks)
const SCENES: [(&str, &str, &str); 3] = [
    (
        "ragdoll_pile_11k",
        "Ragdoll Pile (10k)",
        "1000 ragdolls (cone + twist-limited hinge joints) dropped into a pile",
    ),
    (
        "hinge_chains_10k",
        "Hinge Chains (10k)",
        "400 chains of 25 hinged links released horizontally",
    ),
    (
        "motor_field_10k",
        "Motor Field (10k)",
        "2500 velocity-motor windmills, each rotor carrying two ball-jointed weights",
    ),
];

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let bench = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut out = bench.join("scenes/out");
    let mut only: Option<Vec<String>> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(args.next().context("--out DIR")?),
            _ => only = Some(a.split(',').map(String::from).collect()),
        }
    }
    std::fs::create_dir_all(&out)?;
    for (id, name, what) in SCENES {
        if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == id)) {
            continue;
        }
        let summary = export(id, name, what, &out).with_context(|| format!("exporting {id}"))?;
        eprintln!("{id:20} {summary}");
    }
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn export(id: &str, name: &str, what: &str, out: &Path) -> Result<String> {
    let (world, info) = engine_scenes::build_world(name)?;
    let snap = world.snapshot();
    let params = snap.params.unwrap_or_default();
    ensure!(
        snap.colliders.len() == snap.bodies.len(),
        "every body needs exactly one collider"
    );
    let f = |x: f32| x.to_bits();
    let mut words: Vec<u32> = Vec::new();
    let (mut boxes, mut spheres, mut statics) = (0usize, 0usize, 0usize);
    for (i, (b, c)) in snap.bodies.iter().zip(&snap.colliders).enumerate() {
        ensure!(
            c.body as usize == i && c.local_pos == [0.0; 3],
            "body {i}: collider must sit at the body origin"
        );
        let stat = b.inv_mass == 0.0;
        statics += stat as usize;
        let (shape, w2) = match c.shape {
            s if s == ShapeType::Box as u32 => {
                boxes += 1;
                (0u32, [f(c.params[0]), f(c.params[1]), f(c.params[2])])
            }
            s if s == ShapeType::Sphere as u32 => {
                spheres += 1;
                (1u32, [f(c.params[0]); 3])
            }
            s => anyhow::bail!("body {i}: shape {s} is not a box or sphere"),
        };
        let mass = if stat { 0.0 } else { 1.0 / b.inv_mass };
        let inertia = if stat {
            [0.0; 3]
        } else {
            b.inv_inertia.map(|x| 1.0 / x)
        };
        let friction = snap
            .materials
            .get(c.material as usize)
            .map_or(0.5, |m| m.friction);
        words.extend([shape, stat as u32]);
        words.extend(w2);
        words.push(f(mass));
        words.extend(inertia.map(f));
        words.push(f(friction));
        words.extend(b.position.map(f));
        words.extend(b.orientation.map(f));
        words.extend(b.linear_vel.map(f));
        words.extend(b.angular_vel.map(f));
        words.push(f(0.0));
    }
    let nb = snap.bodies.len();
    let (mut drives, mut d6) = (0usize, 0usize);
    let joints: Vec<&Joint> = snap.joints.iter().filter(|j| !j.is_broken()).collect();
    for j in &joints {
        ensure!(
            j.flags & JOINT_FLAG_RADIAL == 0,
            "distance joints are not in the v3 format yet"
        );
        let drive = j.flags & JOINT_FLAG_DRIVE != 0;
        let ball =
            (j.motion & 0xFFF) == 0xA80 && j.flags & (JOINT_FLAG_CONE | JOINT_FLAG_DRIVE) == 0;
        d6 += !ball as usize;
        drives += drive as usize;
        words.extend([
            j.body_a,
            j.body_b,
            !ball as u32,
            (j.flags & JOINT_FLAG_COLLIDE != 0) as u32,
        ]);
        words.extend(j.anchor_a.map(f));
        words.extend(j.frame_a.map(f));
        words.extend(j.anchor_b.map(f));
        words.extend(j.frame_b.map(f));
        words.extend([
            j.motion,
            f(j.twist_lo),
            f(j.twist_hi),
            f(j.swing_y),
            f(j.swing_z),
            f(j.lin_lo[0]),
            f(j.lin_hi[0]),
        ]);
        words.push(if drive {
            (j.flags >> JOINT_DRIVE_DOF_SHIFT) & 7
        } else {
            u32::MAX
        });
        words.extend(
            [
                j.drive_velocity,
                j.drive_force_limit,
                j.drive_stiffness,
                j.drive_damping,
                j.drive_target,
            ]
            .map(f),
        );
        words.push((j.flags & JOINT_FLAG_CONE != 0) as u32);
    }
    let nj = joints.len();
    let mut bin: Vec<u32> = vec![0x4353_4252, 3, nb as u32, nj as u32, 0, 0];
    bin.extend(&words);
    let bytes: Vec<u8> = bin.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sha = hex(&Sha256::digest(&bytes));

    let span = PROTOCOL.0 * (PROTOCOL.1 + PROTOCOL.2 * PROTOCOL.3);
    let checkpoints = vec![0, LEAD, LEAD + span, STEPS];
    let mut gates = vec![
        json!({ "at": STEPS, "metric": "nonfinite", "max": 0, "why": "NaN/Inf pose or velocity" }),
        json!({ "at": STEPS, "metric": "escaped", "max": 0, "why": "body fell through the ground or left the scene" }),
        json!({ "at": STEPS, "metric": "joint_gap_p99", "max": 0.05, "why": "joints hold their anchors together" }),
    ];
    if id.starts_with("ragdoll") {
        gates.push(json!({ "at": STEPS, "metric": "pen_p99", "max": 0.05, "why": "limbs rest on the ground and each other within 5 cm" }));
        gates.push(json!({ "at": STEPS, "metric": "mean_drift", "group": "dynamic", "min": 1.0, "why": "the ragdolls fell into a pile" }));
    }
    if id.starts_with("hinge") {
        gates.push(json!({ "at": STEPS, "metric": "mean_drift", "group": "dynamic", "min": 2.0, "why": "the chains swung down" }));
    }
    let meta = json!({
        "format": "rubble-bench-scene",
        "version": 3,
        "id": id,
        "tier": "joints",
        "source": { "builder": format!("engine-scenes `{name}` (rubble)"), "note": what },
        "frame": "Y-up, right-handed",
        "units": "SI (m, kg, s, rad); quaternions (x, y, z, w)",
        "dt": params.dt,
        "gravity": [0.0, params.gravity_y, 0.0],
        "iterations": info.iterations,
        "avbd": { "alpha": params.alpha, "gamma": params.gamma, "betaLin": params.beta_lin, "betaAng": params.beta_ang },
        "frictionCombine": "per body (rubble materials)",
        "protocol": { "lead": LEAD, "rounds": PROTOCOL.0, "warmSteps": PROTOCOL.1, "batches": PROTOCOL.2, "perBatch": PROTOCOL.3, "steps": STEPS, "checkpoints": checkpoints },
        "counts": { "bodies": nb, "dynamic": nb - statics, "static": statics, "spheres": spheres, "boxes": boxes, "joints": nj, "d6Joints": d6, "drives": drives, "hulls": 0 },
        "bin": format!("{id}.scene.bin"),
        "binBytes": bytes.len(),
        "binSha256": sha,
        "gates": gates,
    });
    std::fs::write(out.join(format!("{id}.scene.bin")), &bytes)?;
    std::fs::write(
        out.join(format!("{id}.scene.json")),
        serde_json::to_string(&meta)? + "\n",
    )?;
    Ok(format!(
        "{nb} bodies ({statics} static), {nj} joints ({d6} D6, {drives} drives)"
    ))
}
