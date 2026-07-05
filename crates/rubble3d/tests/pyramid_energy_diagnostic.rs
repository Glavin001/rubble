//! Diagnostic (not a gate): instruments the demo-pyramid known-failure to
//! localize WHERE the injected energy comes from. Correlates per-frame kinetic
//! energy changes with warm-start hit rate and contact-manifold churn
//! (feature-id sets changing for persisting body pairs).
//!
//! Run with: cargo nextest run -p rubble3d --test pyramid_energy_diagnostic --run-ignored ignored-only

mod support;

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use std::collections::{HashMap, HashSet};
use support::{add_tracked_body, compute_metrics, snapshot_all, TrackedBody3D};

fn box_desc(x: f32, y: f32, z: f32, w: f32, h: f32, d: f32, mass: f32) -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::new(x, y, z),
        mass,
        friction: 0.5,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(w * 0.5, h * 0.5, d * 0.5),
        },
        ..Default::default()
    }
}

fn build_pyramid() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    const SIZE: usize = 16;
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = World::new(SimConfig {
        gravity,
        max_bodies: 256,
        ..Default::default()
    })
    .ok()?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, -0.5, 0.0, 100.0, 1.0, 100.0, 0.0),
    ));
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            tracked.push(add_tracked_body(
                &mut world,
                "box",
                box_desc(
                    x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                    y as f32 * 0.85 + 0.5,
                    0.0,
                    1.0,
                    0.5,
                    0.5,
                    1.0,
                ),
            ));
        }
    }
    Some((world, tracked, gravity))
}

type PairKey = (u32, u32);
type ContactKey = (u32, u32, u32); // (lo, hi, feature_id)

#[derive(Default, Clone)]
struct FrameStats {
    ke: f32,
    contacts: usize,
    /// Fraction of this frame's contacts whose (pair, feature_id) key existed
    /// last frame — i.e. the warm-start hit rate the GPU pair map achieves.
    key_survival: f32,
    /// Persisting pairs whose feature-id set changed vs the previous frame.
    churned_pairs: usize,
    /// Pairs present in both frames.
    persisted_pairs: usize,
    /// Among churned pairs: how many changed only in the low byte (clip-vertex
    /// index) vs in the prefix (axis type / ref axis / inc axis flip).
    churn_low_byte_only: usize,
    churn_prefix: usize,
    max_depth: f32,
    max_lambda_n: f32,
}

#[test]
#[ignore = "diagnostic instrumentation, not a gate"]
fn pyramid_energy_injection_diagnostic() {
    let Some((mut world, tracked, gravity)) = build_pyramid() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let steps = 360usize;
    let mut prev_features: HashMap<PairKey, HashSet<u32>> = HashMap::new();
    let mut prev_keys: HashSet<ContactKey> = HashSet::new();
    let mut frames: Vec<FrameStats> = Vec::with_capacity(steps);

    for _ in 0..steps {
        world.step();

        let snaps = snapshot_all(&world, &tracked);
        let metrics = compute_metrics(&snaps, gravity);

        let contacts = world.fetch_last_contacts().to_vec();
        let mut survived = 0usize;
        let mut features: HashMap<PairKey, HashSet<u32>> = HashMap::new();
        let mut keys: HashSet<ContactKey> = HashSet::new();
        let mut max_depth: f32 = 0.0;
        let mut max_lambda_n: f32 = 0.0;
        for c in &contacts {
            let pair = (c.body_a.min(c.body_b), c.body_a.max(c.body_b));
            let key = (pair.0, pair.1, c.feature_id);
            if prev_keys.contains(&key) {
                survived += 1;
            }
            keys.insert(key);
            features.entry(pair).or_default().insert(c.feature_id);
            max_depth = max_depth.max(c.depth().abs());
            max_lambda_n = max_lambda_n.max(c.lambda.x.abs());
        }

        let mut churned = 0usize;
        let mut persisted = 0usize;
        let mut low_byte_only = 0usize;
        let mut prefix_churn = 0usize;
        for (key, fids) in &features {
            if let Some(prev) = prev_features.get(key) {
                persisted += 1;
                if prev != fids {
                    churned += 1;
                    let prefixes: HashSet<u32> = fids.iter().map(|f| f & 0xFFFF_FF00).collect();
                    let prev_prefixes: HashSet<u32> =
                        prev.iter().map(|f| f & 0xFFFF_FF00).collect();
                    if prefixes == prev_prefixes {
                        low_byte_only += 1;
                    } else {
                        prefix_churn += 1;
                    }
                }
            }
        }

        frames.push(FrameStats {
            ke: metrics.translational_ke + metrics.rotational_ke,
            contacts: contacts.len(),
            key_survival: if contacts.is_empty() {
                1.0
            } else {
                survived as f32 / contacts.len() as f32
            },
            churned_pairs: churned,
            persisted_pairs: persisted,
            churn_low_byte_only: low_byte_only,
            churn_prefix: prefix_churn,
            max_depth,
            max_lambda_n,
        });
        prev_features = features;
        prev_keys = keys;
    }

    // ---- Report ----
    println!(
        "frame,ke,d_ke,contacts,key_survival,churned,persisted,churn_lowbyte,churn_prefix,max_depth,max_lambda_n"
    );
    for (i, f) in frames.iter().enumerate() {
        let d_ke = if i > 0 { f.ke - frames[i - 1].ke } else { 0.0 };
        if i % 10 == 0 || d_ke > 5.0 {
            println!(
                "{},{:.3},{:+.3},{},{:.3},{},{},{},{},{:.4},{:.1}",
                i,
                f.ke,
                d_ke,
                f.contacts,
                f.key_survival,
                f.churned_pairs,
                f.persisted_pairs,
                f.churn_low_byte_only,
                f.churn_prefix,
                f.max_depth,
                f.max_lambda_n
            );
        }
    }

    // Correlation: split settle-window frames (after initial drop) by whether
    // KE rose or fell, compare mean warm-rate and churn in each bucket.
    let window = 120..frames.len();
    let mut rise = (0f32, 0f32, 0usize); // (sum warm, sum churn frac, n)
    let mut fall = (0f32, 0f32, 0usize);
    for i in window.clone() {
        let d_ke = frames[i].ke - frames[i - 1].ke;
        let churn_frac = if frames[i].persisted_pairs > 0 {
            frames[i].churned_pairs as f32 / frames[i].persisted_pairs as f32
        } else {
            0.0
        };
        let bucket = if d_ke > 0.0 { &mut rise } else { &mut fall };
        bucket.0 += frames[i].key_survival;
        bucket.1 += churn_frac;
        bucket.2 += 1;
    }
    println!("--- settle-window correlation (frames {window:?}) ---");
    if rise.2 > 0 {
        println!(
            "KE-RISING frames  (n={}): mean key_survival={:.3} mean churn_frac={:.3}",
            rise.2,
            rise.0 / rise.2 as f32,
            rise.1 / rise.2 as f32
        );
    }
    if fall.2 > 0 {
        println!(
            "KE-FALLING frames (n={}): mean key_survival={:.3} mean churn_frac={:.3}",
            fall.2,
            fall.0 / fall.2 as f32,
            fall.1 / fall.2 as f32
        );
    }
    let tail_ke_max = frames[window.clone()]
        .iter()
        .map(|f| f.ke)
        .fold(0.0f32, f32::max);
    println!("tail KE max: {tail_ke_max:.2}");
}
