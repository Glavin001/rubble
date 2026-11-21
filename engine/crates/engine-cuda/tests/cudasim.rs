//! CudaSim replays the shared rigid step graph (engine_exec::rigid_plan) on
//! CUDA. Same settled-columns gate as engine-webgpu's columns_hold and
//! engine-metal's metalsim tests, so every host is held to one bar.
//!
//! GPU tests run when `ENGINE_CUDA_DEVICE=1` (Modal sets it) and fail if no
//! device loads; without it they return early, so one test binary serves
//! GPU-less CI and the GPU runner. `ENGINE_CUDA_PERF=1` adds the timed run.
use engine_cuda::{CudaSim, CUDA_KERNELS, PTX_ARCH, SASS_ARCHS};
use engine_exec::rigid_plan::PERSISTENT_ENTRIES;
use engine_schema::{
    AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, FLAG_REUSE_CONTACTS,
};

fn gpu_required() -> bool {
    std::env::var("ENGINE_CUDA_DEVICE").as_deref() == Ok("1")
}

/// Whether a GPU test should run; panics when one was asked for but none loads.
fn gpu() -> bool {
    if !gpu_required() {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host");
        return false;
    }
    assert!(
        CudaSim::has_ptx(),
        "ENGINE_CUDA_DEVICE=1 but this build embedded no PTX (no nvcc)"
    );
    assert!(
        CudaSim::available(),
        "ENGINE_CUDA_DEVICE=1 but no NVIDIA driver/device loaded"
    );
    true
}

#[test]
fn cuda_sources_emitted() {
    for entry in PERSISTENT_ENTRIES {
        let k = CUDA_KERNELS
            .iter()
            .find(|k| k.entry == *entry)
            .unwrap_or_else(|| panic!("{entry}: not compiled to CUDA"));
        assert!(
            k.cu_bytes > 100,
            "{entry}: CUDA source is {} bytes",
            k.cu_bytes
        );
        // Plans count work in 64-thread groups; the block shape is the kernel's own.
        assert!(
            k.threads.iter().product::<u32>() > 0,
            "{entry}: empty thread group"
        );
    }
    // CI (and Modal archives) build with ENGINE_CUDA_REQUIRE_PTX=1: prove every
    // kernel's fatbin is embedded even where no GPU runs the tests.
    if gpu_required() || std::env::var("ENGINE_CUDA_REQUIRE_PTX").as_deref() == Ok("1") {
        assert!(CudaSim::has_ptx(), "no device code embedded");
        for k in CUDA_KERNELS {
            assert!(
                k.image.len() > 1024,
                "{}: fatbin is {} bytes",
                k.entry,
                k.image.len()
            );
        }
        eprintln!(
            "fatbin: {} kernels, SASS {SASS_ARCHS:?}, PTX {PTX_ARCH}",
            CUDA_KERNELS.len()
        );
    }
}

fn columns(n: u32, sleep_enabled: bool) -> (CudaSim, Vec<BodyState>) {
    let h = 10u32;
    let bodies = n * n * h + 1;
    let mut sim = CudaSim::new(WorldCapacity {
        max_bodies: bodies.next_power_of_two().max(4096),
        max_colliders: bodies.next_power_of_two().max(4096),
        max_contacts: (bodies * 16).next_power_of_two().min(1 << 21),
        max_joints: 64,
        max_candidates: (bodies * 32).next_power_of_two().min(1 << 21),
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("CUDA device");
    let mut params = AvbdParams::default();
    params.iterations = 4;
    params.flags |= FLAG_REUSE_CONTACTS;
    if !sleep_enabled {
        params.sleep_time = 1.0e6;
    }
    sim.set_params(params);
    let (half_h, hx) = (25.0f32, 200.0f32);
    let mut all = vec![BodyState {
        position: [0.0, 0.5 - half_h, 0.0],
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0; 3],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    }];
    let mut cols = vec![Collider {
        body: 0,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [hx, half_h, hx, hx.max(half_h)],
    }];
    let inertia = 1.0f32 / 6.0;
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                let pos = [
                    (x as f32 - n as f32 / 2.0) * 1.5,
                    1.0 + y as f32,
                    (z as f32 - n as f32 / 2.0) * 1.5,
                ];
                cols.push(Collider {
                    body: all.len() as u32,
                    shape: ShapeType::Box as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 1.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: [0.5, 0.5, 0.5, 0.75f32.sqrt()],
                });
                all.push(BodyState {
                    position: pos,
                    inv_mass: 1.0,
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    inv_inertia: [1.0 / inertia; 3],
                    flags: 1 | 8,
                    linear_vel: [0.0; 3],
                    sleep_timer: 0.0,
                    angular_vel: [0.0; 3],
                    generation: 0,
                    initial_pos: pos,
                    initial_pad: 0.0,
                    initial_rot: [0.0, 0.0, 0.0, 1.0],
                });
            }
        }
    }
    sim.add_bodies(&all);
    sim.add_colliders(&cols);
    (sim, all)
}

fn columns_hold(n: u32, sleep_enabled: bool) {
    let (mut sim, initial) = columns(n, sleep_enabled);
    eprintln!("device: {}", sim.device_name());
    let dynamic = n * n * 10;
    for s in 0..180u32 {
        let c = sim.step().expect("step without overflow");
        if !sleep_enabled {
            assert_eq!(
                c.awake_count, dynamic,
                "step {s}: active-body coverage lost to sleep"
            );
        }
        if s == 0 || s % 30 == 29 {
            let snap = sim.download_bodies();
            let (mut ke, mut max_v) = (0.0f32, 0.0f32);
            for (id, b) in snap.iter().enumerate().skip(1) {
                assert!(
                    b.position
                        .iter()
                        .chain(&b.linear_vel)
                        .all(|x| x.is_finite()),
                    "step {s}: body {id} nonfinite"
                );
                let drift = (b.position[1] - initial[id].position[1]).abs();
                assert!(
                    drift < 0.15,
                    "step {s}, body {id}: column layer drifted {drift:.4} m"
                );
                for axis in [0, 2] {
                    assert!(
                        (b.position[axis] - initial[id].position[axis]).abs() < 0.1,
                        "step {s}, body {id}: column toppled sideways"
                    );
                }
                let v2: f32 = b.linear_vel.iter().map(|v| v * v).sum();
                ke += 0.5 * v2 / b.inv_mass;
                max_v = max_v.max(v2.sqrt());
            }
            if s == 179 {
                assert!(max_v < 0.1, "columns did not settle: maxV={max_v}");
                assert!(
                    ke < 0.001 * dynamic as f32,
                    "columns gained energy: KE={ke}"
                );
            }
        }
    }
}

#[test]
fn cudasim_columns_10k_hold_height() {
    if gpu() {
        columns_hold(32, true);
    }
}

/// Every awake body sits exactly once in its colour's packed list (bindings
/// 12/15/17) after the tile count, prefix and scatter passes. Slang's CUDA
/// output once read a tile count back after overwriting it with its prefix, so
/// 56% of a 10k scene was never solved (no Metal/WGSL host can see this).
#[test]
fn cudasim_color_lists_cover_every_awake_body() {
    if !gpu() {
        return;
    }
    let (mut sim, initial) = columns(32, false);
    let n = initial.len();
    let max_b = n.next_power_of_two().max(4096);
    for step in 0..3 {
        let c = sim.step().expect("step");
        let colors = sim.download_words(12, n);
        let counts = sim.download_words(17, 64);
        let lists = sim.download_words(15, max_b * c.color_count as usize);
        let mut seen = vec![0u32; n];
        for color in 0..c.color_count as usize {
            for k in 0..counts[color] as usize {
                let body = lists[color * max_b + k] as usize;
                assert!(
                    body < n,
                    "step {step}: colour {color} slot {k} holds {body:#x}"
                );
                assert_eq!(
                    colors[body] as usize, color,
                    "step {step}: body {body} listed under the wrong colour"
                );
                seen[body] += 1;
            }
        }
        let missing: Vec<usize> = (1..n).filter(|&b| seen[b] != 1).take(8).collect();
        assert!(
            missing.is_empty(),
            "step {step}: bodies not listed exactly once: {missing:?}"
        );
    }
}

/// Per-step counters, worst drift and speeds on the 10k columns
/// (`ENGINE_CUDA_TRACE=1`), to compare with another host step by step.
#[test]
fn cudasim_trace_columns_10k() {
    if std::env::var("ENGINE_CUDA_TRACE").as_deref() != Ok("1") || !gpu() {
        return;
    }
    let (mut sim, initial) = columns(32, false);
    let norm = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    for s in 0..30u32 {
        let c = sim.step().expect("step");
        let snap = sim.download_bodies();
        let bodies = snap.iter().zip(&initial).skip(1);
        let drift = bodies
            .clone()
            .map(|(b, i)| (b.position[1] - i.position[1]).abs())
            .fold(0.0f32, f32::max);
        let unmoved = bodies
            .clone()
            .filter(|(b, i)| b.position == i.position)
            .count();
        let max_v = snap
            .iter()
            .skip(1)
            .map(|b| norm(b.linear_vel))
            .fold(0.0f32, f32::max);
        let max_w = snap
            .iter()
            .skip(1)
            .map(|b| norm(b.angular_vel))
            .fold(0.0f32, f32::max);
        println!(
            "trace step {s:2} cand {:7} contacts {:7} manifolds {:6} colors {:2} rounds {:2} awake {:5} drift {drift:.4} unmoved {unmoved} v {max_v:.4} w {max_w:.4}",
            c.candidate_count, c.contact_count, c.manifold_count, c.color_count, c.color_rounds, c.awake_count
        );
    }
}

/// Smaller scenes and sleep off: separates size-dependent faults from the
/// sleep path when the 10k/100k gates fail.
#[test]
fn cudasim_columns_640_hold_height() {
    if gpu() {
        columns_hold(8, true);
    }
}

#[test]
fn cudasim_columns_640_stay_stable_with_sleep_disabled() {
    if gpu() {
        columns_hold(8, false);
    }
}

#[test]
fn cudasim_columns_10k_stay_stable_with_sleep_disabled() {
    if gpu() {
        columns_hold(32, false);
    }
}

#[test]
fn cudasim_columns_100k_stay_stable_with_sleep_disabled() {
    if gpu() {
        columns_hold(100, false);
    }
}

/// Pipelined submission forwards last step's counts on the GPU; it must give
/// the same trajectory as waiting for every step.
#[test]
fn cudasim_pipelined_matches_waiting() {
    if !gpu() {
        return;
    }
    let (mut waited, _) = columns(16, false);
    let (mut piped, _) = columns(16, false);
    for _ in 0..40 {
        waited.step().expect("step");
        piped.step_submit().expect("submit");
    }
    piped.drain().expect("drain");
    let (a, b) = (waited.download_bodies(), piped.download_bodies());
    for (id, (x, y)) in a.iter().zip(&b).enumerate() {
        for k in 0..3 {
            assert!(
                (x.position[k] - y.position[k]).abs() < 1e-5,
                "body {id}: waited {:?} vs pipelined {:?}",
                x.position,
                y.position
            );
        }
    }
}

/// Indirect commands with their counts read on the GPU (`{entry}_gs`, the default)
/// against the host reading them back: the same columns stepped both ways. Every word
/// of the bodies, counters, colouring and indirect slots is equal at each checkpoint
/// (lossless, not within a tolerance). The append-ordered stores ([`ORDERED`]) are
/// filled in atomic order, which CUDA does not repeat run to run: they differ between
/// two runs of the readback path as much as between the paths (measured 2026-11-21, T4),
/// so they are compared only as the control does, and printed.
#[test]
fn cudasim_device_indirect_matches_readback() {
    /// Contacts, AABBs, candidates, awake list, constraint ids, manifolds, solver
    /// points and pairs: rows appended by atomic counters.
    const ORDERED: [usize; 8] = [4, 9, 10, 11, 14, 25, 27, 28];
    if !gpu() {
        return;
    }
    // Bindings that differ between two sims: (binding, first word, words differing).
    fn diff(a: &mut CudaSim, b: &mut CudaSim) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for slot in 0..engine_exec::rigid_plan::BIND_COUNT {
            let words = a.binding_words(slot);
            let (x, y) = (a.download_words(slot, words), b.download_words(slot, words));
            let n = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            if n > 0 {
                out.push((slot, x.iter().zip(&y).position(|(p, q)| p != q).unwrap(), n));
            }
        }
        out
    }
    let mut bad = Vec::new();
    for (n, steps) in [(8u32, 120u32), (32, 90), (100, 30)] {
        let (mut dev, _) = columns(n, false);
        let (mut host, _) = columns(n, false);
        // Control: the readback path against itself (what CUDA repeats run to run).
        let (mut host2, _) = columns(n, false);
        assert!(dev.device_indirect(), "device-side indirect is the default");
        host.set_device_indirect(false);
        host2.set_device_indirect(false);
        for step in 1..=steps {
            dev.step().expect("device-indirect step");
            host.step().expect("readback step");
            host2.step().expect("readback step");
            if step % 30 != 0 && step != steps {
                continue;
            }
            let (ab, control) = (diff(&mut dev, &mut host), diff(&mut host2, &mut host));
            eprintln!(
                "{} bodies, step {step}: device vs readback {ab:?}; readback vs readback {control:?}",
                n * n * 10
            );
            let state = |d: &[(usize, usize, usize)]| -> Vec<(usize, usize, usize)> {
                d.iter().copied().filter(|x| !ORDERED.contains(&x.0)).collect()
            };
            if !state(&ab).is_empty() || !state(&control).is_empty() {
                bad.push((n, step, ab, control));
            }
        }
        assert_eq!(dev.indirect_syncs, 0, "device-side indirect read binding 23 back");
        eprintln!(
            "{} bodies, {steps} steps: readback path waited {:.1} times a step",
            n * n * 10,
            host.indirect_syncs as f64 / f64::from(steps)
        );
    }
    assert!(bad.is_empty(), "state bindings differ: {bad:?}");
}

/// Steady-state step time on the 10k-body columns with sleep off (every body
/// awake), as one JSON line (`ENGINE_CUDA_PERF=1`).
#[test]
fn cudasim_perf_columns_10k() {
    if std::env::var("ENGINE_CUDA_PERF").as_deref() != Ok("1") || !gpu() {
        return;
    }
    // Both indirect paths, interleaved, three rounds at 10k and two at 100k.
    for (n, rounds) in [(32, 3), (100, 2)] {
        for _ in 0..rounds {
            for device_indirect in [false, true] {
                perf_columns(n, device_indirect);
            }
        }
    }
}

fn perf_columns(n: u32, device_indirect: bool) {
    let (mut sim, _) = columns(n, false);
    sim.set_device_indirect(device_indirect);
    for _ in 0..30 {
        sim.step().expect("warm-up step");
    }
    let (steps, syncs0) = (120u32, sim.indirect_syncs);
    let t = std::time::Instant::now();
    for _ in 0..steps {
        sim.step_submit().expect("submit");
    }
    let c = sim.drain().expect("drain");
    let wall_ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(steps);
    let phases = sim.step_profiled().expect("profiled step");
    let phase_json: Vec<String> = phases
        .iter()
        .map(|(n, ms)| format!("\"{n}\":{ms:.4}"))
        .collect();
    println!(
        "{{\"engine\":\"rubble-cuda\",\"scene\":\"columns_{}_awake\",\"indirect\":\"{}\",\"device\":{:?},\"bodies\":{},\"steps\":{steps},\"wall_ms_per_step\":{wall_ms:.4},\"indirect_syncs_per_step\":{:.1},\"contacts\":{},\"colors\":{},\"phase_ms\":{{{}}}}}",
        if n == 32 { "10k" } else { "100k" },
        if device_indirect { "device" } else { "readback" },
        sim.device_name(),
        c.body_count,
        (sim.indirect_syncs - syncs0) as f64 / f64::from(steps),
        c.contact_count,
        c.color_count,
        phase_json.join(",")
    );
    assert!(wall_ms.is_finite() && wall_ms > 0.0);
}
