use engine_exec::PhysicsBackend;
use std::collections::HashMap;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "Brick Ring (9k)".into());
    let steps: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let (mut world, _) = engine_scenes::build_world(&name).expect("build");
    for _ in 0..steps {
        world.step().expect("step");
    }
    let c = world.last_counters();
    let contacts = world.download_contacts();
    let bodies = world.download_bodies();
    // Map collider -> body by assuming 1:1 after ground (collider i -> body i) for this scene
    let mut pair_pen: HashMap<(u32, u32), f32> = HashMap::new();
    for ct in &contacts {
        if ct.alive == 0 { continue; }
        let a = ct.collider_a.min(ct.collider_b);
        let b = ct.collider_a.max(ct.collider_b);
        let e = pair_pen.entry((a, b)).or_insert(0.0);
        *e = e.max(-ct.c0_n);
    }
    let mut plane_pairs = 0u32;
    let mut dd_pairs = 0u32;
    let mut deep_pairs = 0u32;
    let mut mid_pairs = 0u32;
    let mut touch_pairs = 0u32;
    let mut far_center = 0u32;
    let mut sum_dist = 0.0f32;
    for ((a, b), pen) in &pair_pen {
        if *a == 0 { plane_pairs += 1; continue; }
        dd_pairs += 1;
        if *pen > 0.05 { deep_pairs += 1; }
        else if *pen > 0.01 { mid_pairs += 1; }
        else { touch_pairs += 1; }
        let ba = &bodies[*a as usize];
        let bb = &bodies[*b as usize];
        let dx = ba.position[0] - bb.position[0];
        let dy = ba.position[1] - bb.position[1];
        let dz = ba.position[2] - bb.position[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        sum_dist += d;
        if d > 1.2 { far_center += 1; }
    }
    println!(
        "cand={} contacts={} unique_pairs={} plane_pairs={} dd_pairs={}",
        c.candidate_count, c.contact_count, pair_pen.len(), plane_pairs, dd_pairs
    );
    println!(
        "dd deep>5cm={} mid>1cm={} touch={} far_center>1.2m={} avg_center_dist={:.3}",
        deep_pairs, mid_pairs, touch_pairs, far_center,
        if dd_pairs > 0 { sum_dist / dd_pairs as f32 } else { 0.0 }
    );
}
