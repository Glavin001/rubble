//! Broadphase pair oracle for long sweeps (CCD audit Phase 0, gap 2): fast
//! spheres whose look-ahead sweep spans 0.5–10 grid cells, among static and
//! slow spheres, on the uniform and the hierarchical grid. The GPU's candidate
//! pairs must equal a CPU brute force over the same swept bounds (gAabbs as
//! build_aabbs wrote them) with the same relative-sweep test emitCandidate
//! applies: no missed pair (the 27-cell stencil used to drop them) and no
//! duplicate. Spheres only, so no face-axis prune differs between the two.

use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, FLAG_SPEC_VELOCITY};
use engine_webgpu::WgpuSim;
use std::collections::HashSet;

const DT: f32 = 1.0 / 60.0;

struct Rng(u32);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
    fn r(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

fn sphere(sim: &mut WgpuSim, pos: [f32; 3], r: f32, inv_mass: f32, v: [f32; 3]) -> u32 {
    let i = if inv_mass > 0.0 { 2.5 * inv_mass / (r * r) } else { 0.0 };
    let b = sim.add_body(BodyState {
        position: pos,
        inv_mass,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [i; 3],
        flags: if inv_mass > 0.0 { 9 } else { 0 },
        linear_vel: v,
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body: b,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [r, 0.0, 0.0, r],
    })
}

fn run(hier: bool, fast_list: bool) -> usize {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 8192,
        max_colliders: 8192,
        max_contacts: 262_144,
        max_joints: 64,
        max_candidates: 524_288,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.dt = DT;
    p.gravity_y = 0.0;
    p.flags |= FLAG_SPEC_VELOCITY;
    sim.set_params(p);
    sim.set_grid_levels(Some(hier));
    sim.set_fast_list(fast_list);
    let mut rng = Rng(12345);
    // A 3D lattice of static spheres (grid entries) and slow dynamic ones.
    for i in 0..900u32 {
        let pos = [(i % 15) as f32 * 0.6, ((i / 15) % 6) as f32 * 0.6, (i / 90) as f32 * 0.6];
        if i % 3 == 0 {
            sphere(&mut sim, pos, 0.2, 1.0, [rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)]);
        } else {
            sphere(&mut sim, pos, 0.2, 0.0, [0.0; 3]);
        }
    }
    // Fast spheres through the lattice: sweeps of 0.5..10 cells (cell 0.5 m on
    // the uniform grid: 2 × the largest small dynamic radius, floored at 0.5).
    let mut fast = Vec::new();
    for k in 0..60u32 {
        let cells = 0.5 + 9.5 * (k as f32 / 59.0);
        let speed = cells * 0.5 / DT;
        let d = glam::Vec3::new(rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)).normalize();
        let start = glam::Vec3::new(rng.r(0.0, 8.4), rng.r(0.0, 3.0), rng.r(0.0, 5.4)) - d * (0.5 * cells * 0.5);
        fast.push(sphere(&mut sim, start.to_array(), 0.1, 1.0, (d * speed).to_array()));
    }
    // Settle the bookkeeping, then check the next step's broadphase. Without
    // the fast list, fast movers past the hierarchical grid's top level fill
    // the 10-entry oversized list and the step fails loudly (gap 2's other face).
    if let Err(e) = sim.step() {
        assert!(!fast_list, "step: {e:?}");
        println!("{}: control step failed: {e:?}", if hier { "hierarchical" } else { "uniform" });
        return usize::MAX;
    }
    let bodies = sim.download_bodies();
    let pairs = sim.debug_broadphase_pairs();
    let n = sim.collider_count() as usize;
    let max_bodies = sim.capacity().max_bodies as u64;
    let aabbs = sim.read_words(9, max_bodies * 8, n * 8);
    let bb = |i: usize| -> ([f32; 3], [f32; 3]) {
        let w = &aabbs[i * 8..i * 8 + 8];
        let f = |k: usize| f32::from_bits(w[k]);
        ([f(0), f(1), f(2)], [f(4), f(5), f(6)])
    };
    // emitCandidate's relative-sweep test (lookAheadSeparated): undo each box's
    // own sweep (v·h when its reach is positive) and sweep one box by the
    // relative motion.
    let sweep = |i: usize| -> [f32; 3] {
        let col = sim.collider(i as u32);
        let b = &bodies[col.body as usize];
        let moving = b.is_dynamic() && !b.is_sleeping();
        let v = glam::Vec3::from_array(b.linear_vel);
        let r = col.params[0];
        let motion = v.length() * DT + glam::Vec3::from_array(b.angular_vel).length() * r * DT;
        if moving && motion > p.contact_offset { (v * DT).to_array() } else { [0.0; 3] }
    };
    // The GPU's fast-list rule (colliderIsFast): a sweep longer than the
    // uniform cell (0.5 m floor) or the hierarchical base cell.
    let cell = if hier { sim.params().grid_cell_size } else { sim.params().grid_cell_size.max(0.5) };
    let is_fast = |i: usize| {
        let s = sweep(i);
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt() > cell
    };
    let near_segment = |f: usize, e: usize| {
        let (mn, mx) = bb(f);
        let s = glam::Vec3::from_array(sweep(f));
        let umn = glam::Vec3::from_array(mn) - s.min(glam::Vec3::ZERO);
        let umx = glam::Vec3::from_array(mx) - s.max(glam::Vec3::ZERO);
        let s0 = 0.5 * (umn + umx);
        let r = (0.5 * (umx - umn)).length();
        let (emn, emx) = bb(e);
        let (emn, emx) = (glam::Vec3::from_array(emn), glam::Vec3::from_array(emx));
        let c = 0.5 * (emn + emx);
        let t = ((c - s0).dot(s) / s.length_squared().max(1e-20)).clamp(0.0, 1.0);
        (s0 + s * t - c).length() <= r + (0.5 * (emx - emn)).length()
    };
    let mut expected = HashSet::new();
    let mut overlapping = HashSet::new();
    let dynamic = |i: usize| bodies[sim.collider(i as u32).body as usize].is_dynamic();
    for i in 0..n {
        for j in i + 1..n {
            if !dynamic(i) && !dynamic(j) {
                continue;
            }
            let (amn, amx) = bb(i);
            let (bmn, bmx) = bb(j);
            if (0..3).any(|k| amx[k] < bmn[k] || bmx[k] < amn[k]) {
                continue;
            }
            let (si, sj) = (sweep(i), sweep(j));
            let mut imn = [0.0; 3];
            let mut imx = [0.0; 3];
            let mut jmn = [0.0; 3];
            let mut jmx = [0.0; 3];
            for k in 0..3 {
                imn[k] = amn[k] - si[k].min(0.0);
                imx[k] = amx[k] - si[k].max(0.0);
                jmn[k] = bmn[k] - sj[k].min(0.0);
                jmx[k] = bmx[k] - sj[k].max(0.0);
                let rel = si[k] - sj[k];
                let (a, b) = (imn[k], imn[k] + rel);
                imn[k] = a.min(b);
                let (a, b) = (imx[k], imx[k] + rel);
                imx[k] = a.max(b);
            }
            if (0..3).any(|k| imx[k] < jmn[k] || jmx[k] < imn[k]) {
                continue;
            }
            overlapping.insert((i as u32, j as u32));
            // The grid's bounding-sphere prune (radius _pad0: the swept shape's
            // bound around its box centre) also drops pairs that cannot touch.
            let pad = |k: usize| f32::from_bits(aabbs[k * 8 + 3]);
            let ci = (glam::Vec3::from_array(amn) + glam::Vec3::from_array(amx)) * 0.5;
            let cj = (glam::Vec3::from_array(bmn) + glam::Vec3::from_array(bmx)) * 0.5;
            if (ci - cj).length() > pad(i) + pad(j) {
                continue;
            }
            // A fast collider walks only the cells within reach of its sweep
            // segment (broadphase_fast): a grid entry whose bounding sphere
            // stays clear of the swept shape's bounding capsule is not a pair.
            let (fi, fj) = (is_fast(i), is_fast(j));
            if fi != fj {
                let (f, e) = if fi { (i, j) } else { (j, i) };
                if !near_segment(f, e) {
                    continue;
                }
            }
            expected.insert((i as u32, j as u32));
        }
    }
    let mut got = HashSet::new();
    let mut dup = 0;
    for &pair in &pairs {
        if !got.insert(pair) {
            dup += 1;
        }
    }
    let missed: Vec<_> = expected.difference(&got).copied().collect();
    // Upper bound: every candidate's swept bounds overlap (the tube is
    // conservative at cell granularity, so it may keep more than `expected`).
    let extra: Vec<_> = got.difference(&overlapping).copied().collect();
    let fast_pairs = expected.iter().filter(|(a, b)| fast.contains(a) || fast.contains(b)).count();
    println!(
        "{}: {} expected pairs ({fast_pairs} with a fast sphere), {} candidates, {} missed, {} extra, {dup} duplicates",
        if hier { "hierarchical" } else { "uniform" },
        expected.len(),
        got.len(),
        missed.len(),
        extra.len()
    );
    let kind = |i: u32| if fast.contains(&i) { "fast" } else if dynamic(i as usize) { "slow" } else { "static" };
    for &(a, b) in missed.iter().take(6) {
        println!("  missed ({a},{b}) {} {} A {:?} B {:?} sa {:?} sb {:?}", kind(a), kind(b), bb(a as usize), bb(b as usize), sweep(a as usize), sweep(b as usize));
    }
    for &(a, b) in extra.iter().take(6) {
        println!("  extra ({a},{b}) {} {} A {:?} B {:?} sa {:?} sb {:?}", kind(a), kind(b), bb(a as usize), bb(b as usize), sweep(a as usize), sweep(b as usize));
    }
    assert!(fast_pairs > 100, "the fast spheres should sweep through many neighbours ({fast_pairs})");
    assert!(extra.is_empty(), "pairs whose swept bounds do not overlap: {:?}", &extra[..extra.len().min(10)]);
    assert_eq!(dup, 0, "duplicate candidates");
    if fast_list {
        assert!(missed.is_empty(), "missed pairs: {:?}", &missed[..missed.len().min(10)]);
    }
    missed.len()
}

#[test]
fn long_sweeps_miss_no_pairs_on_the_uniform_grid() {
    // Control: without the fast list the 27-cell stencil drops pairs (gap 2).
    let control = run(false, false);
    assert!(control > 0, "the control should miss pairs: the oracle is not sensitive");
    assert_eq!(run(false, true), 0);
}

#[test]
fn long_sweeps_miss_no_pairs_on_the_hierarchical_grid() {
    let control = run(true, false);
    assert!(control > 0, "the control should miss pairs: the oracle is not sensitive");
    assert_eq!(run(true, true), 0);
}
