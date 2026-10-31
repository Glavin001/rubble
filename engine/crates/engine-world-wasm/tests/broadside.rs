//! The Broadside take natively: the captain walks the length of the choreography and
//! reaches the main deck, every cue fires and breaks wood, nothing hits him, and the
//! closing volley wrecks the ship around him.

use engine_world_wasm::broadside::*;
use rubble::GpuContext;
use std::time::Instant;

const GLB: &[u8] = include_bytes!("../../engine-webgpu/fracture/assets/galleon-stern.glb");

#[test]
fn the_take() {
    let mut b = Broadside::new(GpuContext::Default, GLB).unwrap();
    let frames = (16.0 / b.dt).round() as u32;
    let mut closest = f32::MAX;
    let mut per_shot: Vec<(u32, usize)> = Vec::new();
    let mut last_broken = 0;
    let t0 = Instant::now();
    let mut finale_broken = None;
    let mut lap = Instant::now();
    for _ in 0..frames {
        b.step().unwrap();
        b.poll_blocking().unwrap();
        // Nearest any ball in flight (a shot, not one spent and rolling) has come to the
        // captain's capsule axis.
        let c = glam::Vec3::from(b.captain_at);
        let flying: Vec<[f32; 3]> = b
            .balls
            .iter()
            .filter_map(|(id, _)| b.world.read_body_state(*id).ok())
            .filter(|s| glam::Vec3::from(s.linear_velocity).length() > 50.0)
            .map(|s| s.transform.position)
            .collect();
        for p in flying {
            let d = glam::Vec3::from(p) - c;
            let axial = d.y.clamp(-CAPTAIN_HALF, CAPTAIN_HALF);
            let r = (d - glam::Vec3::new(0.0, axial, 0.0)).length();
            closest = closest.min(r);
        }
        if b.total.broken != last_broken {
            per_shot.push((b.frame, b.total.broken - last_broken));
            last_broken = b.total.broken;
        }
        if b.finale.is_some_and(|f| b.frame == f + 1) {
            finale_broken = Some(b.total.broken);
        }
        if b.frame % 60 == 0 {
            let ms = lap.elapsed().as_secs_f64() * 1e3 / 60.0;
            lap = Instant::now();
            eprintln!(
                "{ms:.1} ms/frame; t {:.1} s captain {:?} ground {} shots {} broken {} splits {} crushed {}",
                b.seconds(),
                b.captain_at.map(|x| (x * 100.0).round() / 100.0),
                b.captain_ground,
                b.total.shots,
                b.total.broken,
                b.total.splits,
                b.total.crushed
            );
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / frames as f64;
    eprintln!(
        "take: {frames} frames, {ms:.1} ms/frame (blocking); {} shots, {} bonds broken, {} splits, {} crushed; finale at {:?}, closest ball {closest:.2} m",
        b.total.shots, b.total.broken, b.total.splits, b.total.crushed, b.finale
    );
    eprintln!("breaks by frame: {per_shot:?}");
    assert!(
        b.captain_stopped(),
        "the captain reaches the end of his walk"
    );
    assert!(
        b.captain_at[1] < ship::MAIN_DECK + CAPTAIN_HALF + CAPTAIN_RADIUS + 0.3
            && b.captain_at[0] < 8.0,
        "he ends on the main deck: {:?}",
        b.captain_at
    );
    assert_eq!(
        b.total.shots,
        GUNS_PER_CUE * cues().len() + finale_targets().len(),
        "every cue and the closing volley fired"
    );
    assert!(
        closest > CAPTAIN_RADIUS + BALL_RADIUS + 0.3,
        "no shot comes near him: {closest} m"
    );
    assert!(b.finale.is_some(), "the closing volley fired");
    let before = finale_broken.unwrap();
    assert!(b.total.broken > before, "the closing volley breaks more");
}

/// The closing volley alone, early (profiling its frames).
#[test]
#[ignore]
fn volley_only() {
    let mut b = Broadside::new(GpuContext::Default, GLB).unwrap();
    b.scripted = false;
    for f in 0..240u32 {
        if f == 30 {
            b.final_volley().unwrap();
        }
        let t = Instant::now();
        b.step().unwrap();
        b.poll_blocking().unwrap();
        if f >= 28 && (f < 40 || f % 20 == 0) {
            eprintln!(
                "frame {f}: {:.1} ms, broken {} splits {} crushed {}",
                t.elapsed().as_secs_f64() * 1e3,
                b.total.broken,
                b.total.splits,
                b.total.crushed
            );
        }
    }
}

/// Frames of the take for Shardwright's `prefracture preview --sim` (every chunk at its
/// fragment's pose, coloured as its mesh; the newest ball): `BROADSIDE_FRAMES=path`.
#[test]
#[ignore]
fn dump_frames() {
    let out = std::env::var("BROADSIDE_FRAMES").expect("BROADSIDE_FRAMES=path");
    let at: Vec<u32> = std::env::var("BROADSIDE_AT")
        .unwrap_or("0,150,240,280,330,420,560,590,640,900".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let mut b = Broadside::new(GpuContext::Default, GLB).unwrap();
    let mut frames = Vec::new();
    let mut colors = serde_json::Map::new();
    for (c, f) in b.fragment.iter().enumerate() {
        if let Some(f) = f {
            colors.insert(f.to_string(), serde_json::json!(b.color[c]));
        }
    }
    let last = *at.iter().max().unwrap();
    for frame in 0..=last {
        if frame > 0 {
            b.step().unwrap();
            b.poll_blocking().unwrap();
        }
        if std::env::var_os("BROADSIDE_TRACE").is_some()
            && (330..440).contains(&frame)
            && frame % 3 == 0
        {
            let shots: Vec<_> = b
                .log
                .iter()
                .filter(|l| l.0 + 3 >= frame)
                .map(|l| l.1.map(|x| x.round()))
                .collect();
            eprintln!(
                "  {frame}: captain {:?} ground {} new shots {shots:?}",
                b.captain_at.map(|x| (x * 100.0).round() / 100.0),
                b.captain_ground
            );
        }
        if !at.contains(&frame) {
            continue;
        }
        let poses = b.chunk_poses();
        let mut bodies = Vec::new();
        for (c, t) in poses.iter().enumerate() {
            let Some(f) = b.fragment[c] else { continue };
            if b.crushed[c] {
                continue;
            }
            // The fragment's mesh is about its centre of mass, unrotated: the chunk's
            // pose less its principal rotation (its rest rotation).
            let q = glam::Quat::from_array(t.rotation)
                * glam::Quat::from_array(b.rest[c].rotation).inverse();
            let p = t.position;
            bodies.push(serde_json::json!([f, p[0], p[1], p[2], q.x, q.y, q.z, q.w]));
        }
        let mut fr = serde_json::json!({ "bodies": bodies });
        if let Some(p) = b.ball_poses().last() {
            fr["projectile"] =
                serde_json::json!([p.position[0], p.position[1], p.position[2], 0.25]);
        }
        eprintln!(
            "frame {frame} ({:.2} s): captain {:?}, broken {}",
            b.seconds(),
            b.captain_at,
            b.total.broken
        );
        frames.push(fr);
    }
    std::fs::write(
        out,
        serde_json::json!({ "frames": frames, "colors": colors }).to_string(),
    )
    .unwrap();
}
