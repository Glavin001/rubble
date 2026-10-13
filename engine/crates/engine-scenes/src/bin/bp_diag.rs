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
    let mut pair_n: HashMap<(u32, u32), u32> = HashMap::new();
    for ct in &contacts {
        if ct.alive == 0 { continue; }
        let a = ct.collider_a.min(ct.collider_b);
        let b = ct.collider_a.max(ct.collider_b);
        *pair_n.entry((a, b)).or_insert(0) += 1;
    }
    let mut hist = [0u32; 5];
    let mut plane = 0u32;
    for ((a, _), n) in &pair_n {
        let k = (*n).min(4) as usize;
        hist[k] += 1;
        if *a == 0 { plane += 1; }
    }
    println!(
        "cand={} contacts={} pairs={} plane_pairs={} hist[1..4]=[{},{},{},{}]",
        c.candidate_count, c.contact_count, pair_n.len(), plane, hist[1], hist[2], hist[3], hist[4]
    );
}
