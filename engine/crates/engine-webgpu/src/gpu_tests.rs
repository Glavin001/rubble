//! GPU regressions for adjacency construction and contact persistence.
//! These require a real wgpu adapter; absence fails explicitly instead of skipping.
use super::*;
use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

fn download<T: bytemuck::Pod>(sim: &WgpuSim, binding: usize, offset: u64, count: usize) -> Vec<T> {
    let size = (count * std::mem::size_of::<T>()) as u64;
    let staging = sim.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("regression-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = sim.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&sim.bufs[binding], offset, &staging, 0, size);
    sim.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    let _ = sim.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let result = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    result
}

// Independent test-side encoding of the persistent hot/cold ABI. This bypasses
// GPU packing so migration tests can detect matching bugs in GPU writers/readers.
fn write_contact_fixture(sim: &WgpuSim, first: u32, contacts: &[Contact], owners: &[u32]) {
    assert_eq!(contacts.len(), owners.len());
    if !sim.persistent_contacts {
        sim.queue.write_buffer(
            &sim.bufs[4],
            u64::from(first) * 128,
            bytemuck::cast_slice(contacts),
        );
        return;
    }
    let mut hot = Vec::<u32>::new();
    let mut cold = Vec::<u32>::new();
    for (c, &owner) in contacts.iter().zip(owners) {
        assert!(
            c.alive <= 1,
            "fixture encoder accepts resolved points, not transient reuse tags"
        );
        hot.extend(c.anchor_a.map(f32::to_bits));
        hot.push(c.c0_n.to_bits());
        hot.extend(c.anchor_b.map(f32::to_bits));
        hot.push(c.c0_t1.to_bits());
        hot.extend(
            [
                c.lambda_n,
                c.lambda_t1,
                c.lambda_t2,
                c.c0_t2,
                c.k_n,
                c.k_t,
                c._pad[2],
            ]
            .map(f32::to_bits),
        );
        hot.push(if c.alive == 0 {
            0
        } else {
            (owner + 1) | if c._pad[0] != 0.0 { 0x80000000 } else { 0 }
        });
        cold.extend([
            c.collider_a,
            c.collider_b,
            c.feature_id,
            c._pad[1].to_bits(),
        ]);
        cold.extend(c.normal.map(f32::to_bits));
        cold.push(c.friction.to_bits());
    }
    sim.queue.write_buffer(
        &sim.bufs[4],
        u64::from(first) * 32,
        bytemuck::cast_slice(&cold),
    );
    sim.queue.write_buffer(
        &sim.bufs[27],
        u64::from(first) * 64,
        bytemuck::cast_slice(&hot),
    );
}

fn contact_storage_words(sim: &WgpuSim, first: u32, count: usize) -> Vec<Vec<u32>> {
    if sim.persistent_contacts {
        vec![
            download(sim, 4, u64::from(first) * 32, count * 8),
            download(sim, 27, u64::from(first) * 64, count * 16),
        ]
    } else {
        vec![download(sim, 4, u64::from(first) * 128, count * 32)]
    }
}

fn read_contact_fixture(sim: &WgpuSim, first: u32, count: usize) -> Vec<Contact> {
    if !sim.persistent_contacts {
        return download(sim, 4, u64::from(first) * 128, count);
    }
    let words = contact_storage_words(sim, first, count);
    (0..count)
        .map(|i| {
            let m = &words[0][i * 8..i * 8 + 8];
            let p = &words[1][i * 16..i * 16 + 16];
            let h: Vec<f32> = p.iter().copied().map(f32::from_bits).collect();
            let n = [
                f32::from_bits(m[4]),
                f32::from_bits(m[5]),
                f32::from_bits(m[6]),
            ];
            let cross = |a: [f32; 3], b: [f32; 3]| {
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let t = cross(
                n,
                if n[0].abs() < 0.9 {
                    [1., 0., 0.]
                } else {
                    [0., 1., 0.]
                },
            );
            let len = t.iter().map(|x| x * x).sum::<f32>().sqrt();
            let t1 = t.map(|x| if len > 0. { x / len } else { 0. });
            Contact {
                collider_a: m[0],
                collider_b: m[1],
                feature_id: m[2],
                alive: if p[15] & 0x3fffffff == 0 {
                    0
                } else if p[15] & 0x40000000 != 0 {
                    m[2]
                } else {
                    1
                },
                normal: n,
                friction: f32::from_bits(m[7]),
                tangent1: t1,
                tangent2: cross(n, t1),
                anchor_a: [h[0], h[1], h[2]],
                c0_n: h[3],
                penetration: h[3],
                anchor_b: [h[4], h[5], h[6]],
                c0_t1: h[7],
                lambda_n: h[8],
                lambda_t1: h[9],
                lambda_t2: h[10],
                c0_t2: h[11],
                k_n: h[12],
                k_t: h[13],
                _pad: [
                    if p[15] & 0x80000000 != 0 { 1. } else { 0. },
                    f32::from_bits(m[3]),
                    h[14],
                ],
            }
        })
        .collect()
}

#[test]
fn stable_color_packing_orders_active_bodies_and_preserves_partial_tile_bounds() {
    let mut cap = WorldCapacity::small();
    cap.max_colors = 7;
    let mut sim = WgpuSim::new(cap.clone()).expect("GPU adapter required");
    let sentinel = 0xabcdef01u32;
    for kernel in ["prefix_color_tiles", "prefix_color_tiles_parallel"] {
        for (n, invalid) in [
            (0u32, false),
            (1, false),
            (63, false),
            (64, false),
            (65, false),
            (517, false),
            (4093, false),
            (65, true),
        ] {
            sim.body_count = n;
            sim.write_host_uniforms();
            let bodies: Vec<BodyState> = (0..n)
                .map(|i| BodyState {
                    flags: if i % 11 == 0 {
                        0
                    } else if i % 7 == 0 {
                        5
                    } else {
                        9
                    },
                    inv_mass: if i % 11 == 0 { 0.0 } else { 1.0 },
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    ..BodyState::zeroed()
                })
                .collect();
            let colors: Vec<u32> = (0..n)
                .map(|i| {
                    if invalid && i == 2 {
                        u32::MAX
                    } else {
                        (i * 37 + i / 11) % 5
                    }
                })
                .collect();
            let active: Vec<u32> = (0..n).filter(|&i| bodies[i as usize].flags == 9).collect();
            if n > 0 {
                sim.queue
                    .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
                sim.queue
                    .write_buffer(&sim.bufs[12], 0, bytemuck::cast_slice(&colors));
            }
            // The old awake list's append order must not leak into stable buckets.
            let reversed: Vec<u32> = active.iter().copied().rev().collect();
            if !reversed.is_empty() {
                sim.queue
                    .write_buffer(&sim.bufs[11], 0, bytemuck::cast_slice(&reversed));
            }
            sim.queue.write_buffer(
                &sim.bufs[7],
                0,
                bytes_of(&StepCounters {
                    body_count: n,
                    awake_count: active.len() as u32,
                    ..Default::default()
                }),
            );
            let list_words = (cap.max_bodies * (cap.max_colors + 2)) as usize;
            sim.queue.write_buffer(
                &sim.bufs[15],
                0,
                bytemuck::cast_slice(&vec![sentinel; list_words]),
            );
            let hist_words = (cap.max_colors * cap.max_bodies.div_ceil(64)) as usize;
            sim.queue.write_buffer(
                &sim.bufs[16],
                0,
                bytemuck::cast_slice(&vec![sentinel; hist_words]),
            );
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "count_color_tiles", n as u64, None);
                sim.disp(
                    &mut pass,
                    kernel,
                    if kernel.ends_with("_parallel") {
                        u64::from(cap.max_colors) * 64
                    } else {
                        64
                    },
                    None,
                );
                sim.disp(&mut pass, "scatter_color_tiles", n as u64, None);
                sim.disp(&mut pass, "write_color_args", 64, None);
            }
            sim.queue.submit([enc.finish()]);
            let packed: Vec<u32> = download(&sim, 15, 0, list_words);
            let counts: Vec<u32> = download(&sim, 17, 0, cap.max_colors as usize);
            let histogram: Vec<u32> = download(&sim, 16, 0, hist_words);
            let groups = n.div_ceil(64) as usize;
            for color in 0..cap.max_colors {
                let expected: Vec<u32> = active
                    .iter()
                    .copied()
                    .filter(|&i| colors[i as usize] == color)
                    .collect();
                assert_eq!(
                counts[color as usize] as usize,
                expected.len(),
                "{kernel}: n={n} color={color}: lost/duplicated bodies or stale counts after reset"
            );
                let start = (color * cap.max_bodies) as usize;
                assert_eq!(&packed[start..start + expected.len()], &expected,
                "{kernel}: n={n} color={color}: buckets must preserve body-index order independent of awake-list scheduling");
                assert!(
                packed[start + expected.len()..start + cap.max_bodies as usize]
                    .iter()
                    .all(|&v| v == sentinel),
                "{kernel}: n={n} color={color}: partial tile or histogram prefix wrote beyond the bucket"
            );
                for group in 0..groups {
                    let prefix = expected.iter().filter(|&&i| i < group as u32 * 64).count();
                    assert_eq!(histogram[color as usize * groups + group] as usize, prefix,
                    "{kernel}: n={n} color={color} tile={group}: exclusive prefix is wrong; bucket ordering/coverage would regress");
                }
            }
            assert!(
                histogram[cap.max_colors as usize * groups..]
                    .iter()
                    .all(|&v| v == sentinel),
                "{kernel}: n={n}: count/prefix touched capacity beyond the live tile range"
            );
            let history = (cap.max_bodies * cap.max_colors) as usize;
            for i in 0..cap.max_bodies {
                let expected = if i < n
                    && bodies[i as usize].flags == 9
                    && colors[i as usize] < cap.max_colors
                {
                    colors[i as usize] + 1
                } else {
                    sentinel
                };
                assert_eq!(packed[history + i as usize], expected,
                "{kernel}: n={n} body={i}: stable scatter must publish only valid active history and preserve other rows");
            }
            assert!(
                packed[history + cap.max_bodies as usize..]
                    .iter()
                    .all(|&v| v == sentinel),
                "stable packing overwrote the immutable compaction-history row"
            );
            let counters: Vec<StepCounters> = download(&sim, 7, 0, 1);
            assert_eq!(
            counters[0].overflow_flags,
            if invalid {
                engine_schema::OVERFLOW_COLORS
            } else {
                0
            },
            "{kernel}: n={n}: unfinished colors must report overflow, never silently enter a valid bucket"
        );
        }
    }
}

#[test]
fn broadphase_pairs_are_unique_across_large_body_order_and_hash_aliases() {
    for (ball_id, sleeping_id) in [
        (512u32, None),
        (1, None),
        (u32::MAX, None),
        (u32::MAX, Some(1u32)),
        (u32::MAX, Some(2u32)),
    ] {
        let mut sim = WgpuSim::new_with_compact_contacts(WorldCapacity::small(), false)
            .expect("GPU adapter required");
        sim.params.grid_cell_size = 1.225;
        let brick_id = if ball_id == 1 { 2 } else { 1 };
        // >512 colliders selects the grid path; the contacting brick is several
        // cells from the ball's centre, so its surface requires the large list.
        for id in 0..513 {
            sim.add_body(BodyState {
                position: if ball_id == u32::MAX && id == 1 {
                    [0.0, 0.75, 0.0]
                } else if ball_id == u32::MAX && id == 2 {
                    [0.0, 1.25, -0.1]
                } else if id == ball_id {
                    [0.0, 8.5, 0.0]
                } else if id == brick_id {
                    [0.0, 8.5, 7.9]
                } else {
                    [1000.0 + id as f32 * 3.0, 10.0, 1000.0]
                },
                orientation: [0.0, 0.0, 0.0, 1.0],
                inv_mass: if id == 0 { 0.0 } else { 1.0 },
                inv_inertia: [1.0; 3],
                flags: if id == 0 { 0 } else { 9 },
                ..BodyState::zeroed()
            });
            sim.add_collider(Collider {
                body: id,
                shape: if id == ball_id {
                    ShapeType::Sphere as u32
                } else {
                    ShapeType::Box as u32
                },
                params: if id == ball_id {
                    [8.0, 0.0, 0.0, 8.0]
                } else {
                    [0.5, 0.25, 0.25, 0.61237246]
                },
                local_rot: [0.0, 0.0, 0.0, 1.0],
                ..Collider::zeroed()
            });
        }
        sim.write_host_uniforms();
        if let Some(id) = sleeping_id {
            // Sleepers keep their previously fitted AABB. Establish it while
            // awake, then freeze either endpoint to test both index orderings.
            let mut fit = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = fit.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "build_aabbs", 513, None);
            }
            sim.queue.submit([fit.finish()]);
            sim.queue
                .write_buffer(&sim.bufs[0], id as u64 * 112 + 44, bytes_of(&5u32));
        }
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "build_aabbs", 513, None);
            sim.disp(&mut pass, "broadphase_clear", 131072, None);
            sim.disp(&mut pass, "broadphase_insert", 513, None);
            sim.disp(&mut pass, "broadphase_query", 513, None);
            sim.disp(&mut pass, "write_work_args", 64, None);
            sim.disp_work(&mut pass, "narrowphase", 64, None);
        }
        sim.queue.submit([enc.finish()]);
        let counters: Vec<StepCounters> = download(&sim, 7, 0, 1);
        let c = counters[0];
        assert_eq!(
            c.overflow_flags, 0,
            "ball collider {ball_id}: capacity overflow invalidates pair-uniqueness evidence"
        );
        let pairs: Vec<engine_schema::CandidatePair> = if c.candidate_count == 0 {
            Vec::new()
        } else {
            download(&sim, 10, 0, c.candidate_count as usize)
        };
        let target_id = if ball_id == u32::MAX { 2 } else { ball_id };
        let matches = pairs
            .iter()
            .filter(|p| {
                (p.collider_a == brick_id && p.collider_b == target_id)
                    || (p.collider_a == target_id && p.collider_b == brick_id)
            })
            .count();
        assert_eq!(matches, 1,
            "sleeping={sleeping_id:?}, impactor collider {target_id}, brick {brick_id}: expected one surface pair, got {matches}. Endpoint fallback must not duplicate the large list; lower-index sleepers must survive early ownership filtering; aliased hash buckets need exact-cell filtering.");
        assert_eq!(
            c.manifold_count, 1,
            "ball collider {ball_id}: duplicate/missing manifolds alter impact stiffness"
        );
        assert_eq!(c.contact_count, if ball_id == u32::MAX { 4 } else { 1 },
            "impactor collider {target_id}: sphere-box has one contact; box face has four, regardless of broadphase hashing");
    }
}

#[test]
fn broadphase_linked_buckets_preserve_membership_and_match_bruteforce_pairs() {
    use std::collections::BTreeSet;
    const N: usize = 641; // Grid path and partial final workgroup.
    const BUCKETS: usize = 131072;
    let mut sim = WgpuSim::new_with_compact_contacts(WorldCapacity::small(), false)
        .expect("GPU adapter required for grid pair coverage");
    sim.params.grid_cell_size = 1.25;
    let mut centers = Vec::new();
    for id in 0..N {
        // A regular lattice crossing negative cells, with exact touching faces.
        let p = [
            (id % 17) as f32 * 0.75 - 6.,
            ((id / 17) % 9) as f32 * 0.75 - 3.,
            (id / 153) as f32 * 0.75 - 1.5,
        ];
        centers.push(p);
        sim.add_body(BodyState {
            position: p,
            orientation: [0., 0., 0., 1.],
            inv_mass: 1.,
            inv_inertia: [1.; 3],
            flags: 9,
            ..BodyState::zeroed()
        });
        sim.add_collider(Collider {
            body: id as u32,
            shape: ShapeType::Box as u32,
            // A conservative radius avoids a rounded-down decimal radius
            // rejecting an exact corner touch before the grid coverage check.
            params: [0.375, 0.375, 0.375, 0.65],
            local_rot: [0., 0., 0., 1.],
            ..Collider::zeroed()
        });
    }
    sim.write_host_uniforms();
    const CANARY: u32 = 0xfedc_ba98;
    sim.queue.write_buffer(
        &sim.bufs[15],
        N as u64 * 4,
        bytemuck::cast_slice(&[CANARY; 16]),
    );
    let mut enc = sim.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        sim.disp(&mut pass, "build_aabbs", N as u64, None);
        sim.disp(&mut pass, "broadphase_clear", BUCKETS as u64, None);
        sim.disp(&mut pass, "broadphase_insert", N as u64, None);
        sim.disp(&mut pass, "broadphase_query", N as u64, None);
    }
    sim.queue.submit([enc.finish()]);
    let heads: Vec<u32> = download(&sim, 14, 0, BUCKETS);
    let links: Vec<u32> = download(&sim, 15, 0, N);
    let mut seen = vec![false; N];
    for h in 0..BUCKETS {
        let mut encoded = heads[h];
        while encoded != u32::MAX {
            assert_ne!(
                encoded & 0x8000_0000,
                0,
                "Awake dynamic entry lacks early ownership flag"
            );
            let id = (encoded & 0x7fff_ffff) as usize;
            assert!(id<N && !seen[id],"bucket={h}, collider={id}: linked grid lost bijection; encoded links must be masked before indexing and may not form cycles");
            seen[id] = true;
            let cell = centers[id].map(|x| (x / 1.25).floor() as i32 as u32);
            let expected = ((cell[0].wrapping_mul(73856093)
                ^ cell[1].wrapping_mul(19349663)
                ^ cell[2].wrapping_mul(83492791))
                & 131071) as usize;
            assert_eq!(
                h, expected,
                "collider={id}: insertion assigned the wrong signed-coordinate hash bucket"
            );
            encoded = links[id];
        }
    }
    assert!(
        seen.iter().all(|x| *x),
        "An eligible collider is missing from the linked grid"
    );
    assert_eq!(
        download::<u32>(&sim, 15, N as u64 * 4, 16),
        [CANARY; 16],
        "Partial insertion overwrote the first unused collider slots"
    );
    let counters = download::<StepCounters>(&sim, 7, 0, 1)[0];
    assert_eq!(
        counters.overflow_flags, 0,
        "Overflow invalidates broadphase coverage evidence"
    );
    let pairs: Vec<CandidatePair> = download(&sim, 10, 0, counters.candidate_count as usize);
    let mut actual = BTreeSet::new();
    for pair in pairs {
        let key = (
            pair.collider_a.min(pair.collider_b),
            pair.collider_a.max(pair.collider_b),
        );
        assert!(
            actual.insert(key),
            "Duplicate pair {key:?}: hash buckets still require exact-cell filtering"
        );
    }
    let mut expected = BTreeSet::new();
    for a in 0..N {
        for b in a + 1..N {
            if (0..3).all(|axis| (centers[a][axis] - centers[b][axis]).abs() <= 0.7501) {
                expected.insert((a as u32, b as u32));
            }
        }
    }
    let missing: Vec<_> = expected.difference(&actual).take(8).collect();
    let extra: Vec<_> = actual.difference(&expected).take(8).collect();
    assert!(actual==expected,"Filtered linked grid differs from independent all-pairs touching-box oracle: missing={missing:?}, extra={extra:?}, actual={}, expected={}",actual.len(),expected.len());
}

#[test]
fn sleep_dispatch_skips_empty_work_and_restores_current_island_propagation() {
    let mut sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    let mut body = BodyState::zeroed();
    body.orientation = [0.0, 0.0, 0.0, 1.0];
    body.inv_mass = 1.0;
    body.flags = BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32;
    for id in 0..2 {
        sim.add_body(body);
        sim.add_collider(Collider {
            body: id,
            ..Collider::zeroed()
        });
    }
    sim.write_host_uniforms();
    sim.queue.write_buffer(
        &sim.bufs[25],
        0,
        bytes_of(&Manifold {
            collider_a: 0,
            collider_b: 1,
            contact_count: 1,
            ..Manifold::zeroed()
        }),
    );
    // The host has seen no contacts: scheduling must use this frame's GPU count.
    assert_eq!(sim.last_counters.manifold_count, 0);
    for (name, candidates, should_sleep) in [
        ("no candidates", [0u32, 0], false),
        ("moving neighbour vetoes sleep", [1, 0], false),
        ("whole island may sleep", [1, 1], true),
        ("candidates disappear next frame", [0, 0], false),
    ] {
        sim.queue
            .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&[body, body]));
        sim.queue.write_buffer(
            &sim.bufs[7],
            0,
            bytes_of(&StepCounters {
                body_count: 2,
                collider_count: 2,
                manifold_count: 1,
                ..Default::default()
            }),
        );
        sim.queue
            .write_buffer(&sim.bufs[20], 0, bytemuck::cast_slice(&candidates));
        let any = u32::from(candidates.iter().any(|&n| n != 0));
        sim.queue
            .write_buffer(&sim.bufs[21], 13 * 4, bytes_of(&any));
        let canary = 0x12345678u32;
        sim.queue
            .write_buffer(&sim.bufs[23], 0, bytemuck::cast_slice(&[canary; 70 * 3]));
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "write_sleep_args", 64, None);
            sim.disp_work(&mut pass, "sleep_island_gate", 66, None);
            sim.disp_work(&mut pass, "sleep_island_commit", 67, None);
        }
        sim.queue.submit([enc.finish()]);
        let args: Vec<u32> = download(&sim, 23, 0, 70 * 3);
        for slot in 0..70 {
            let expected = if slot == 66 || slot == 67 {
                [any, 1, 1]
            } else {
                [canary; 3]
            };
            assert_eq!(&args[slot * 3..slot * 3 + 3], &expected,
                "{name}: sleep scheduling corrupted slot {slot} or launched empty work; only the consumed manifold/awake slots may be reused.");
        }
        let actual: Vec<BodyState> = download(&sim, 0, 0, 2);
        for (id, b) in actual.iter().enumerate() {
            assert_eq!(b.flags & BodyFlags::Sleeping as u32 != 0, should_sleep,
                "{name}: body {id} slept incorrectly; skipping sleep dispatch must preserve neighbour vetoes and resume when candidates appear.");
        }
        let counters: Vec<StepCounters> = download(&sim, 7, 0, 1);
        assert_eq!(
            counters[0].sleep_events,
            if should_sleep { 2 } else { 0 },
            "{name}: sleep event count must match actual state transitions."
        );
    }
}

#[test]
fn warmstart_preserves_distinct_corners_with_duplicate_feature_ids() {
    warmstart_duplicate_corner_fixture("persist_contacts");
}

#[test]
fn manifold_warmstart_preserves_distinct_corners_and_consumes_reuse_tags_once() {
    warmstart_duplicate_corner_fixture("persist_manifolds");
}

fn warmstart_duplicate_corner_fixture(entry: &str) {
    let mut sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    let mut ground = BodyState::zeroed();
    ground.position = [0.0, -24.5, 0.0];
    ground.orientation = [0.0, 0.0, 0.0, 1.0];
    sim.add_body(ground);
    let mut body = ground;
    body.position = [20.0, 1.0, 20.0];
    body.inv_mass = 1.0;
    body.flags = BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32;
    sim.add_body(body);
    for id in 0..2 {
        let mut col = Collider::zeroed();
        col.body = id;
        col.shape = ShapeType::Box as u32;
        col.params = [0.5, 0.5, 0.5, 0.866];
        sim.add_collider(col);
    }
    let previous: Vec<Contact> = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        .iter()
        .enumerate()
        .map(|(i, &(x, z))| {
            let mut c = Contact::zeroed();
            c.collider_a = 0;
            c.collider_b = 1;
            // A small box's four corners can occupy one quantized cell on a large slab.
            c.feature_id = 0x01010105;
            c.alive = 1;
            c.anchor_a = [20.0 + x, 25.0, 20.0 + z];
            c.anchor_b = [x, -0.5, z];
            c.normal = [0.0, 1.0, 0.0];
            c.tangent1 = [1.0, 0.0, 0.0];
            c.tangent2 = [0.0, 0.0, 1.0];
            c.lambda_n = -10.0 * (i + 1) as f32;
            c.k_n = 4000.0;
            c.k_t = 100.0;
            c._pad[0] = 1.0; // sticking contacts carry their previous anchors
            c
        })
        .collect();
    let order = [2usize, 0, 3, 1];
    let current: Vec<Contact> = order
        .iter()
        .map(|&i| {
            let mut c = previous[i];
            c.lambda_n = 0.0;
            c._pad[0] = 0.0;
            c
        })
        .collect();
    sim.last_counters.contact_count = 4;
    sim.write_host_uniforms();
    let counters = StepCounters {
        body_count: 2,
        collider_count: 2,
        contact_count: 4,
        manifold_count: 1,
        ..StepCounters::default()
    };
    sim.queue.write_buffer(&sim.bufs[7], 0, bytes_of(&counters));
    sim.queue.write_buffer(
        &sim.bufs[25],
        0,
        bytes_of(&Manifold {
            collider_a: 0,
            collider_b: 1,
            contact_start: 0,
            contact_count: 4,
            normal: [0., 1., 0.],
            friction: 0.5,
        }),
    );
    sim.queue
        .write_buffer(&sim.bufs[4], 0, bytemuck::cast_slice(&current));
    sim.queue.write_buffer(
        &sim.bufs[4],
        sim.capacity.max_contacts as u64 * 128,
        bytemuck::cast_slice(&previous),
    );
    let mut enc = sim.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        sim.disp(&mut pass, "contact_hash_clear", 1024, None);
        sim.disp(&mut pass, "contact_hash_insert", 4, None);
        sim.disp(&mut pass, entry, 4, None);
    }
    sim.queue.submit([enc.finish()]);
    let actual: Vec<Contact> = download(&sim, 4, 0, 4);
    for (slot, &source) in order.iter().enumerate() {
        assert_eq!(actual[slot].anchor_b, previous[source].anchor_b,
            "corner {slot} (previous {source}) collapsed onto another support point because feature IDs alias");
        let expected = previous[source].lambda_n * sim.params.alpha * sim.params.gamma;
        assert!(
            (actual[slot].lambda_n - expected).abs() < 1e-4,
            "corner {slot} inherited the wrong normal force: actual={}, expected={expected}",
            actual[slot].lambda_n
        );
    }

    // Reused manifolds encode the exact previous-contact index. Give current
    // records no geometry/forces: persistence must gather the source, preserve
    // corner identity, and decay each independent penalty exactly once.
    let sources: Vec<Contact> = previous
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut c = *c;
            c.k_t = 100.0 + i as f32;
            c._pad[2] = 300.0 + i as f32;
            c
        })
        .collect();
    let reused: Vec<Contact> = order.iter().map(|&i| sources[i]).collect();
    let tagged: Vec<Contact> = order
        .iter()
        .map(|&i| Contact {
            alive: i as u32 + 2,
            ..Contact::zeroed()
        })
        .collect();
    sim.queue
        .write_buffer(&sim.bufs[4], 0, bytemuck::cast_slice(&tagged));
    sim.queue.write_buffer(
        &sim.bufs[4],
        sim.capacity.max_contacts as u64 * 128,
        bytemuck::cast_slice(&sources),
    );
    let mut enc = sim.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        sim.disp(&mut pass, entry, 4, None);
    }
    sim.queue.submit([enc.finish()]);
    let actual: Vec<Contact> = download(&sim, 4, 0, 4);
    for (i, c) in actual.iter().enumerate() {
        assert_eq!(
            c.anchor_a, reused[i].anchor_a,
            "reused corner {i} lost its A anchor"
        );
        assert_eq!(
            c.anchor_b, reused[i].anchor_b,
            "reused corner {i} lost its B anchor"
        );
        assert!(
            (c.c0_n - sim.params.contact_offset).abs() < 1e-6,
            "reused corner {i} must recompute C0 at the current poses before warm-start decay"
        );
        let expected = reused[i].lambda_n * sim.params.alpha * sim.params.gamma;
        assert!((c.lambda_n - expected).abs() < 1e-4,
            "reused corner {i} lost its copied warm-start state or decayed twice: actual={}, expected={expected}", c.lambda_n);
        assert_eq!(
            c.alive, 1,
            "transient reuse tag must be consumed before the solver"
        );
        assert!((c.k_t - reused[i].k_t * sim.params.gamma).abs() < 1e-4);
        assert!(
            (c._pad[2] - reused[i]._pad[2] * sim.params.gamma).abs() < 1e-4,
            "second tangent penalty must decay independently on reused contacts"
        );
    }
}

#[test]
fn csr_prefix_matches_cpu_across_workgroups() {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 524288,
        max_colliders: 1,
        max_contacts: 1024,
        max_candidates: 1024,
        ..WorldCapacity::small()
    })
    .expect("GPU adapter required for CSR regression");
    // Partial lanes, partial blocks, block-sum scan boundaries and full capacity.
    for n in [
        1u32, 3, 4, 63, 64, 255, 256, 257, 511, 512, 513, 1024, 1025, 16384, 16385, 25001, 65536,
        65537, 100001, 131072, 505923, 524288,
    ] {
        for pattern in 0..3 {
            sim.body_count = n;
            sim.write_host_uniforms();
            let degrees: Vec<u32> = (0..n)
                .map(|i| match pattern {
                    0 => (i * 7 + 3) % 11,
                    1 => 0, // Empty adjacency must overwrite stale nonzero totals/cursors.
                    _ => {
                        if i % 256 == 255 || i % 256 == 0 {
                            12345
                        } else {
                            0
                        }
                    }
                })
                .collect();
            let mut input = degrees.clone();
            input.push(0xdeadbeef); // The total must be published, even for all zeros.
            if n < sim.capacity.max_bodies {
                input.push(0xfeedface);
            }
            let offset = u64::from(constraint_id_words(&sim.capacity)) * 4;
            sim.queue
                .write_buffer(&sim.bufs[14], offset, bytemuck::cast_slice(&input));
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "csr_prefix_blocks", u64::from((n + 3) / 4), None);
                sim.disp(&mut pass, "csr_prefix_block_sums", 64, None);
                sim.disp(&mut pass, "csr_prefix_add", n.into(), None);
            }
            sim.queue.submit([enc.finish()]);
            let actual: Vec<u32> = download(&sim, 14, offset, input.len());
            let cursors: Vec<u32> = download(&sim, 12, 0, n as usize);
            let mut expected = 0;
            for i in 0..=n as usize {
                assert_eq!(actual[i], expected,
                "CSR prefix corrupted at body {i} of {n}, pattern {pattern}: block carry or exclusive publication lost a degree");
                if i < n as usize {
                    assert_eq!(
                        cursors[i], expected,
                        "CSR scatter cursor differs from the exclusive offset at body {i} of {n}"
                    );
                }
                if i < degrees.len() {
                    expected += degrees[i];
                }
            }
            if n < sim.capacity.max_bodies {
                assert_eq!(
                    actual[n as usize + 1],
                    0xfeedface,
                    "CSR scan wrote past its end offset for {n} bodies, pattern {pattern}"
                );
            }
        }
    }
}

/// A collision chain must contain one reachable minimum live index per pair,
/// even with duplicate manifold blocks, dead gaps and concurrent insertions.
/// The poison payload makes clear that hash identity does not depend on physics.
#[test]
fn contact_hash_preserves_colliding_pair_heads_across_dead_gaps_and_history_halves() {
    use std::collections::BTreeMap;
    const HASH_WORDS: u32 = 1024;
    fn key(a: u32, b: u32) -> u32 {
        let h = a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663);
        if h == u32::MAX {
            a ^ b ^ 0xa5a5a5a5
        } else {
            h
        }
    }
    let mut points = Vec::new();
    for block in 0..3 {
        for pair in 0..32u32 {
            // All 32 distinct pairs share a hash bucket, with repeated blocks
            // crossing workgroups. Dead points must not hide a later live head.
            let id = if block == 1 { 31 - pair } else { pair };
            for corner in 0..4 {
                points.push(Contact {
                    collider_a: 1,
                    collider_b: 2 + id * HASH_WORDS,
                    alive: u32::from(corner != 0 && !(block == 0 && id % 3 == 0)),
                    feature_id: 0x8000_0000 | corner,
                    normal: [f32::NAN; 3],
                    anchor_a: [f32::NAN; 3],
                    anchor_b: [f32::NAN; 3],
                    lambda_n: f32::NAN,
                    _pad: [
                        if pair % 2 == 0 { 1.0 } else { 0.0 },
                        f32::from_bits(0xffff_fffe),
                        0.0,
                    ],
                    ..Contact::zeroed()
                });
            }
        }
    }
    let mut expected = BTreeMap::new();
    for (i, c) in points.iter().enumerate().filter(|(_, c)| c.alive != 0) {
        expected
            .entry((c.collider_a, c.collider_b))
            .or_insert(i as u32);
    }
    assert_eq!(expected.len(), 32);
    // Previous-step manifolds for the per-manifold (three-avbd hashInsert) shape:
    // one per contiguous four-point block, then a zero-count overflow record.
    // A poisoned record past the published count would add a 33rd pair.
    let mut manifolds: Vec<Manifold> = (0..points.len() as u32 / 4)
        .map(|run| Manifold {
            collider_a: points[run as usize * 4].collider_a,
            collider_b: points[run as usize * 4].collider_b,
            contact_start: run * 4,
            contact_count: 4,
            ..Manifold::zeroed()
        })
        .collect();
    manifolds.push(Manifold {
        collider_a: 1,
        collider_b: 2,
        ..Manifold::zeroed()
    });
    let published_manifolds = manifolds.len() as u32;
    manifolds.push(Manifold {
        collider_a: 1,
        collider_b: 999,
        contact_start: 1,
        contact_count: 4,
        ..Manifold::zeroed()
    });
    let bucket = key(1, 2) & (HASH_WORDS - 1);
    assert!(expected
        .keys()
        .all(|&(a, b)| key(a, b) & (HASH_WORDS - 1) == bucket));
    for persistent in [false, true] {
        let sim = if persistent {
            WgpuSim::new_with_persistent_contacts(WorldCapacity::small())
        } else {
            WgpuSim::new(WorldCapacity::small())
        }
        .expect("GPU adapter required for warm-start hash regression");
        for (parity, per_manifold) in (0..2).flat_map(|p| [(p, false), (p, true)]) {
            let offset = parity * sim.capacity.max_contacts;
            let owners = vec![0; points.len()];
            write_contact_fixture(&sim, offset, &points, &owners);
            let absolute: Vec<Manifold> = manifolds
                .iter()
                .map(|m| Manifold {
                    contact_start: m.contact_start + if m.contact_count > 0 { offset } else { 0 },
                    ..*m
                })
                .collect();
            sim.queue
                .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&absolute));
            let before = contact_storage_words(&sim, offset, points.len());
            let mut host = HostUniforms {
                params: sim.params,
                s: [[0; 4]; 8],
            };
            host.s[1][3] = points.len() as u32;
            host.s[2][0] = offset;
            host.s[2][1] = HASH_WORDS;
            host.s[2][2] = manifolds.len() as u32 + 8; // manifold capacity
            host.s[3][1] = published_manifolds;
            host.s[3][2] = u32::from(per_manifold);
            sim.queue.write_buffer(&sim.bufs[6], 0, bytes_of(&host));
            for repeat in 0..4 {
                let context = format!("persistent={persistent}, parity={parity}, per_manifold={per_manifold}, repeat={repeat}");
                let mut enc = sim.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    sim.disp(&mut pass, "contact_hash_clear", HASH_WORDS as u64, None);
                    if per_manifold {
                        // Production dispatch: GPU-written slot 68 must cover the
                        // published manifold count, not the contact count.
                        sim.disp(&mut pass, "write_work_args", 64, None);
                        sim.disp_work(&mut pass, "contact_hash_insert_manifolds", 68, None);
                    } else {
                        sim.disp(&mut pass, "contact_hash_insert", points.len() as u64, None);
                    }
                }
                sim.queue.submit([enc.finish()]);
                let table: Vec<u32> = download(&sim, 14, 0, HASH_WORDS as usize);
                assert_eq!(table.iter().filter(|&&i| i != u32::MAX).count(), expected.len(),
                    "{context}: hash must publish exactly one head per live pair; duplicates or missing heads lose or repeat warm starts");
                for (&pair, &first) in &expected {
                    let mut found = None;
                    for probe in 0..64 {
                        let index =
                            table[((key(pair.0, pair.1) + probe) & (HASH_WORDS - 1)) as usize];
                        if index == u32::MAX {
                            break;
                        }
                        assert!(
                            (index as usize) < points.len(),
                            "{context}: out-of-range history index {index}"
                        );
                        let c = &points[index as usize];
                        assert_ne!(
                            c.alive, 0,
                            "{context}: dead history index {index} published"
                        );
                        if (c.collider_a, c.collider_b) == pair {
                            found = Some(index);
                            break;
                        }
                    }
                    assert_eq!(found, Some(first),
                        "{context}, pair={pair:?}: minimum live head {first} must be reachable before EMPTY; a hole from weak CAS or a wrong duplicate head loses manifold warm-start support");
                }
                assert_eq!(
                    contact_storage_words(&sim, offset, points.len()),
                    before,
                    "{context}: building the hash modified contact metadata, tags or force history"
                );
            }
        }
    }
}

#[test]
fn hash_clear_covers_more_than_one_webgpu_dispatch_row() {
    let sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    // Exercise the production hash-clear kernel without allocating millions of bodies.
    // 2^22 needs 65536 workgroups: clamping to 65535 silently leaves 64 stale slots.
    for n in [(1u32 << 22), (1 << 23) + 17] {
        let sentinel = 0x12345678u32;
        let buffer = sim
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("large hash regression"),
                contents: bytemuck::cast_slice(&vec![sentinel; n as usize + 1]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let mut host = HostUniforms {
            params: sim.params,
            s: [[0; 4]; 8],
        };
        host.s[2][1] = n;
        sim.queue.write_buffer(&sim.bufs[6], 0, bytes_of(&host));
        let p = &sim.pipes["contact_hash_clear"];
        let bind = sim.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("large hash regression"),
            layout: &p.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: sim.bufs[6].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let read = sim.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("large hash readback"),
            size: (n as u64 + 1) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let (x, y) = groups(n as u64);
            pass.dispatch_workgroups(x, y, 1);
        }
        enc.copy_buffer_to_buffer(&buffer, 0, &read, 0, (n as u64 + 1) * 4);
        sim.queue.submit([enc.finish()]);
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = sim.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let actual: &[u32] = bytemuck::cast_slice(&data);
        for (i, &slot) in actual[..n as usize].iter().enumerate() {
            assert_eq!(slot, u32::MAX,
                "hash slot {i}/{n} was not cleared: dispatch truncation or missing Y-coordinate flattening leaves stale warm-start entries at 506k scale");
        }
        assert_eq!(
            actual[n as usize], sentinel,
            "rounded dispatch wrote beyond the logical hash length"
        );
    }
}

#[test]
fn work_dispatch_uses_current_counts_including_large_contact_growth() {
    let sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    // per_manifold: HOST_S(14) selects the previous manifold count HOST_S(13)
    // (three-avbd hashInsert shape) for slot 68 instead of previous contacts.
    for (per_manifold, contacts) in [false, true]
        .into_iter()
        .flat_map(|m| [0u32, 1, 64, 65, 1_170_611, 4_102_693, (1 << 23) + 17].map(|c| (m, c)))
    {
        let mut host = HostUniforms {
            params: sim.params,
            s: [[u32::MAX; 4]; 8],
        };
        host.s[1][3] = 987654;
        host.s[3][1] = 246_481;
        host.s[3][2] = u32::from(per_manifold);
        sim.queue.write_buffer(&sim.bufs[6], 0, bytes_of(&host));
        let c = StepCounters {
            candidate_count: 765432,
            contact_count: contacts,
            manifold_count: 345678,
            awake_count: 110441,
            _pad0: 12345, // pending recoloring bodies, independent of awake count
            ..Default::default()
        };
        sim.queue.write_buffer(&sim.bufs[7], 0, bytes_of(&c));
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "write_work_args", 64, None);
        }
        sim.queue.submit([enc.finish()]);
        let args: Vec<u32> = download(&sim, 23, 64 * 12, 18);
        for (slot, n) in [
            c.candidate_count,
            c.contact_count,
            c.manifold_count,
            c.awake_count,
            if per_manifold { 246_481 } else { 987654 },
            12345,
        ]
        .into_iter()
        .enumerate()
        {
            let a = &args[slot * 3..slot * 3 + 3];
            let expected_groups = (n as u64).div_ceil(64);
            assert!(
                a[0] <= 32768 && a[1] <= 65535 && a[2] == 1,
                "work slot {} has invalid WebGPU dimensions: {a:?}",
                slot + 64
            );
            assert!(a[0] as u64 * a[1] as u64 >= expected_groups,
                "work slot {} (per_manifold={per_manifold}) missed current-frame work: count={n}, args={a:?}. Prior-frame estimates truncate first-frame ring/gables contacts; slot 68 must follow the selected warm-hash insert granularity.", slot + 64);
            assert_eq!(
                a,
                &[
                    expected_groups.min(32768) as u32,
                    expected_groups.div_ceil(32768).max(1) as u32,
                    1
                ]
            );
        }
    }
}

#[test]
fn color_dispatch_respects_palette_and_preserves_work_slots() {
    let sim = WgpuSim::new(WorldCapacity {
        max_colors: 32,
        ..WorldCapacity::small()
    })
    .expect("GPU adapter required for color dispatch bounds regression");
    // Explicit expected groups at every lane-selection boundary, including an
    // empty bucket. Large counts need no body storage: only run the args writer.
    let cases = [
        (0u32, 0u32),
        (1, 1),
        (4095, 512),
        (4096, 256),
        (32767, 2048),
        (32768, 1024),
        (65535, 2048),
        (65536, 1024),
    ];
    for palette in [1usize, 7, 32] {
        let mut host = HostUniforms {
            params: sim.params,
            s: [[0; 4]; 8],
        };
        host.s[1][0] = palette as u32;
        sim.queue.write_buffer(&sim.bufs[6], 0, bytes_of(&host));
        let counts: Vec<u32> = (0..32).map(|c| cases[c % cases.len()].0).collect();
        sim.queue
            .write_buffer(&sim.bufs[17], 0, bytemuck::cast_slice(&counts));
        let sentinel = vec![0xdeadbeefu32; 70 * 3];
        sim.queue
            .write_buffer(&sim.bufs[23], 0, bytemuck::cast_slice(&sentinel));
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "write_color_args", 64, None);
        }
        sim.queue.submit([enc.finish()]);
        let args: Vec<u32> = download(&sim, 23, 0, 70 * 3);
        for slot in 0..70 {
            let expected = if slot < palette {
                [cases[slot % cases.len()].1, 1, 1]
            } else {
                [0xdeadbeef; 3]
            };
            assert_eq!(&args[slot * 3..slot * 3 + 3], &expected,
                "palette={palette}, slot={slot}: color args must cover active buckets (including empty ones) and leave all other slots untouched; reading 64 counts from a smaller palette is out of bounds on Metal");
        }
    }
}

fn check_rotated_face(height: f32) {
    let mut sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    for id in 0..2 {
        let angle = std::f32::consts::FRAC_PI_4 * 0.5;
        sim.add_body(BodyState {
            position: [0.0, id as f32 * height, 0.0],
            orientation: if id == 0 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [0.0, angle.sin(), 0.0, angle.cos()]
            },
            inv_mass: id as f32,
            inv_inertia: [6.0 * id as f32; 3],
            flags: if id == 0 { 0 } else { 9 },
            ..BodyState::zeroed()
        });
        sim.add_collider(Collider {
            body: id,
            shape: ShapeType::Box as u32,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.5, 0.5, 0.5, 0.8660254],
            ..Collider::zeroed()
        });
    }
    let c = sim.step().expect("rotated two-box contact step");
    assert_eq!(c.contact_count,8,
        "45-degree face overlap is an octagon. Truncating its first four vertices biases support to one side and destabilizes ring courses.");
    let contacts: Vec<Contact> = download(&sim, 4, 0, 8);
    let centroid = contacts.iter().fold([0.0f32; 3], |mut sum, c| {
        for a in 0..3 {
            sum[a] += c.anchor_a[a] / 8.0;
        }
        sum
    });
    assert!(
        centroid[0].abs() < 1e-5 && centroid[2].abs() < 1e-5,
        "clipped support polygon lost symmetry: centroid={centroid:?}"
    );
    for i in 0..8 {
        for j in 0..i {
            let d2 = (0..3)
                .map(|a| (contacts[i].anchor_a[a] - contacts[j].anchor_a[a]).powi(2))
                .sum::<f32>();
            assert!(
                d2 > 1e-5,
                "support corners {i} and {j} collapsed to the same anchor"
            );
        }
    }
}

#[test]
fn rotated_face_keeps_all_eight_support_points() {
    check_rotated_face(0.99);
}

#[test]
fn rotated_face_at_exact_touch_is_not_lost_to_quaternion_roundoff() {
    check_rotated_face(1.0);
}

#[test]
fn friction_penalties_ramp_independently_only_inside_the_unclamped_cone() {
    let mut sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    for id in 0..2 {
        sim.add_body(BodyState {
            orientation: [0.0, 0.0, 0.0, 1.0],
            initial_rot: [0.0, 0.0, 0.0, 1.0],
            inv_mass: id as f32,
            flags: if id == 0 { 0 } else { 9 },
            ..BodyState::zeroed()
        });
        sim.add_collider(Collider {
            body: id,
            shape: ShapeType::Box as u32,
            ..Collider::zeroed()
        });
    }
    for (inside, c0) in [(false, [10.0, 30.0]), (true, [0.01, 0.03])] {
        sim.write_host_uniforms();
        sim.queue.write_buffer(
            &sim.bufs[7],
            0,
            bytes_of(&StepCounters {
                contact_count: 1,
                manifold_count: 1,
                ..StepCounters::default()
            }),
        );
        let mut c = Contact::zeroed();
        c.alive = 1;
        c.collider_a = 0;
        c.collider_b = 1;
        c.normal = [0.0, 1.0, 0.0];
        c.tangent1 = [1.0, 0.0, 0.0];
        c.tangent2 = [0.0, 0.0, 1.0];
        c.friction = 0.5;
        c.lambda_n = -2.0;
        c.k_n = 100.0;
        c.k_t = 10.0;
        c._pad[2] = 30.0;
        c.c0_t1 = c0[0];
        c.c0_t2 = c0[1];
        sim.queue.write_buffer(&sim.bufs[4], 0, bytes_of(&c));
        sim.queue.write_buffer(
            &sim.bufs[25],
            0,
            bytes_of(&Manifold {
                collider_a: 0,
                collider_b: 1,
                contact_count: 1,
                ..Manifold::zeroed()
            }),
        );
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "solve_dual", 1, None);
        }
        sim.queue.submit([enc.finish()]);
        let actual: Vec<Contact> = download(&sim, 4, 0, 1);
        let keep = 1.0 - sim.params.alpha;
        let expected = [10.0, 30.0].map(|x| x); // tangent rows have distinct initial penalties
        for axis in 0..2 {
            let want = expected[axis]
                + if inside {
                    sim.params.beta_lin * c0[axis] * keep
                } else {
                    0.0
                };
            let got = if axis == 0 {
                actual[0].k_t
            } else {
                actual[0]._pad[2]
            };
            assert!((got-want).abs()<1e-3,
                "tangent {axis}, inside_cone={inside}: penalty={got}, expected={want}. Test the unprojected friction force; cone projection always makes the clamped force look in-bounds. Keep each tangent's penalty separate.");
        }
    }
}

#[test]
fn unfinished_coloring_reports_overflow_instead_of_racing_in_color_zero() {
    let mut sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    sim.add_body(BodyState {
        flags: 9,
        inv_mass: 1.0,
        ..BodyState::zeroed()
    });
    sim.write_host_uniforms();
    for color in [u32::MAX, sim.capacity.max_colors] {
        sim.queue.write_buffer(
            &sim.bufs[7],
            0,
            bytes_of(&StepCounters {
                awake_count: 1,
                ..Default::default()
            }),
        );
        sim.queue.write_buffer(&sim.bufs[11], 0, bytes_of(&0u32));
        sim.queue.write_buffer(&sim.bufs[12], 0, bytes_of(&color));
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "pack_color_lists", 1, None);
        }
        sim.queue.submit([enc.finish()]);
        let counters: Vec<StepCounters> = download(&sim, 7, 0, 1);
        assert_ne!(counters[0].overflow_flags & OVERFLOW_COLORS, 0,
            "invalid colour {color} must fail explicitly, never silently join colour zero and race its neighbours");
        let sizes: Vec<u32> = download(&sim, 17, 0, sim.capacity.max_colors as usize);
        assert!(
            sizes.iter().all(|&n| n == 0),
            "invalid body was packed into a solver bucket: {sizes:?}"
        );
    }
}

#[test]
fn sparse_colors_match_ordered_dispatches_across_dependent_chunks_and_empty_buckets() {
    for layout in ["canonical", "compact", "persistent"] {
        let capacity = WorldCapacity {
            max_bodies: 256,
            max_colliders: 256,
            max_contacts: 1024,
            max_candidates: 256,
            ..WorldCapacity::small()
        };
        let mut sim = match layout {
            "persistent" => WgpuSim::new_with_persistent_contacts(capacity),
            _ => WgpuSim::new_with_compact_contacts(capacity, layout == "compact"),
        }
        .expect("GPU required for ordered sparse-color execution regression");
        let dense = if layout == "canonical" {
            "solve_primal_color"
        } else {
            "solve_primal_color_compact"
        };
        let sparse = if layout == "canonical" {
            "solve_primal_sparse"
        } else {
            "solve_primal_sparse_compact"
        };
        // Each color has independent bodies, but constraints cross colors. Later
        // colors must read the poses written by earlier colors in this dispatch.
        // 67 bodies require eight full 8-body chunks plus a partial ninth chunk.
        let buckets: [(u32, u32, u32); 3] = [(8, 0, 67), (10, 67, 9), (12, 76, 17)];
        let initial: Vec<BodyState> = (0..94)
            .map(|i| BodyState {
                position: [i as f32 * 0.003, 0.05, -0.02],
                orientation: [0., 0., 0., 1.],
                initial_rot: [0., 0., 0., 1.],
                inv_mass: if i == 93 { 0. } else { 1. },
                inv_inertia: [2., 3., 4.],
                flags: if i == 93 { 0 } else { 9 },
                ..BodyState::zeroed()
            })
            .collect();
        sim.add_bodies(&initial);
        let mut joints = Vec::new();
        for i in 0..93u32 {
            let other = if i < 67 {
                67 + i % 9
            } else if i < 76 {
                76 + (i - 67) % 17
            } else {
                93
            };
            joints.push(Joint {
                body_a: i,
                body_b: other,
                local_anchor_a: [0.03, 0.04, -0.02],
                local_anchor_b: [-0.02, 0.01, 0.05],
                k: [1800. + i as f32 * 13.; 6],
                ..Joint::zeroed()
            });
        }
        sim.joint_count = joints.len() as u32;
        sim.write_host_uniforms();
        sim.queue
            .write_buffer(&sim.bufs[5], 0, bytemuck::cast_slice(&joints));
        let mut edges = Vec::new();
        let mut offsets = vec![0u32];
        for body in 0..94u32 {
            for (id, joint) in joints.iter().enumerate() {
                if joint.body_a == body || joint.body_b == body {
                    edges.push(id as u32);
                }
            }
            offsets.push(edges.len() as u32);
        }
        sim.queue
            .write_buffer(&sim.bufs[14], 0, bytemuck::cast_slice(&edges));
        sim.queue.write_buffer(
            &sim.bufs[14],
            u64::from(constraint_id_words(&sim.capacity)) * 4,
            bytemuck::cast_slice(&offsets),
        );
        sim.queue.write_buffer(
            &sim.bufs[7],
            0,
            bytes_of(&StepCounters {
                body_count: sim.body_count,
                joint_count: sim.joint_count,
                color_count: 13,
                ..StepCounters::default()
            }),
        );
        let mut counts = [0u32; 32];
        for (color, first, count) in buckets {
            counts[color as usize] = count;
            sim.queue.write_buffer(
                &sim.bufs[15],
                u64::from(color * 256) * 4,
                bytemuck::cast_slice(&(first..first + count).collect::<Vec<_>>()),
            );
        }
        sim.queue
            .write_buffer(&sim.bufs[17], 0, bytemuck::cast_slice(&counts));
        for start in [0u32, 8, 10, 12, 31] {
            let mut outputs = Vec::new();
            for grouped in [false, true] {
                sim.queue
                    .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&initial));
                let mut enc = sim.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for _ in 0..4 {
                        if grouped {
                            sim.disp(&mut pass, sparse, 64, Some(start));
                        } else {
                            for (color, _, count) in buckets {
                                if color >= start {
                                    sim.disp(&mut pass, dense, u64::from(count * 8), Some(color));
                                }
                            }
                        }
                    }
                }
                sim.queue.submit([enc.finish()]);
                outputs.push(download::<BodyState>(&sim, 0, 0, initial.len()));
            }
            for (body, (actual, expected)) in outputs[1].iter().zip(&outputs[0]).enumerate() {
                for (component, (got, want)) in actual
                    .position
                    .iter()
                    .chain(&actual.orientation)
                    .zip(expected.position.iter().chain(&expected.orientation))
                    .enumerate()
                {
                    assert!((got-want).abs()<3e-6,
                        "{layout} sparse tail from color {start}: body={body}, pose component={component}, got={got}, ordered={want}; check EVERY chunk (including body 66), partial lanes, empty buckets, starting color, scratch reuse and storage visibility between dependent colors");
                }
                assert_eq!(
                    &bytes_of(actual)[32..],
                    &bytes_of(&initial[body])[32..],
                    "{layout} sparse solve modified non-pose state for body {body}"
                );
            }
            if start == 31 {
                assert_eq!(
                    bytemuck::cast_slice::<BodyState, u8>(&outputs[1]),
                    bytemuck::cast_slice::<BodyState, u8>(&initial),
                    "empty sparse tail must leave all bodies untouched"
                );
            } else {
                assert!(
                    outputs[0][92].position != initial[92].position,
                    "fixture must move the final tail body"
                );
            }
        }
    }
}

#[test]
fn parallel_primal_reduction_matches_serial_for_every_lane_width() {
    for layout in ["canonical", "compact", "persistent"] {
        primal_reduction_tail_fixture(layout);
    }
}

fn primal_reduction_tail_fixture(layout: &str) {
    const ACTIVE: u32 = 67;
    let capacity = WorldCapacity {
        max_bodies: 65536,
        max_colliders: 16,
        max_contacts: 1024,
        max_candidates: 1024,
        ..WorldCapacity::small()
    };
    let mut sim = match layout {
        "persistent" => WgpuSim::new_with_persistent_contacts(capacity),
        _ => WgpuSim::new_with_compact_contacts(capacity, layout == "compact"),
    }
    .expect("GPU adapter required for cooperative primal tail regression");
    let one = if layout == "canonical" {
        "solve_primal_one"
    } else {
        "solve_primal_one_compact"
    };
    let parallel = if layout == "canonical" {
        "solve_primal_color"
    } else {
        "solve_primal_color_compact"
    };
    let mut initial = Vec::new();
    for id in 0..=ACTIVE {
        let body = BodyState {
            orientation: [0.0, 0.0, 0.0, 1.0],
            initial_rot: [0.0, 0.0, 0.0, 1.0],
            flags: if id < ACTIVE { 9 } else { 0 },
            inv_mass: if id < ACTIVE {
                1.0 / (id + 1) as f32
            } else {
                0.0
            },
            inv_inertia: [2.0, 3.0, 4.0],
            ..BodyState::zeroed()
        };
        sim.add_body(body);
        initial.push(body);
    }
    // Nine distinct rows per body distribute nonzero terms across all eight
    // lanes, including a remainder. 67 independent bodies span several full
    // workgroups and the final partial one, exercising every shared-memory slot.
    // Put those bodies at the END of each color bucket: tail-lane coverage cannot
    // pass just because all nonzero work happens in the first group.
    let joints: Vec<Joint> = (0..ACTIVE * 9)
        .map(|j| {
            let a = (j % 9) as f32 * 0.1;
            Joint {
                body_a: j / 9,
                body_b: ACTIVE,
                local_anchor_a: [a, 0.2 - a, 0.3],
                local_anchor_b: [a + 0.04, 0.15 - a, 0.32],
                k: [2000.0 + j as f32 * 50.0; 6],
                ..Joint::zeroed()
            }
        })
        .collect();
    sim.body_count = 65536;
    sim.joint_count = joints.len() as u32;
    sim.write_host_uniforms();
    sim.queue
        .write_buffer(&sim.bufs[5], 0, bytemuck::cast_slice(&joints));
    let offsets: Vec<u32> = (0..=65536).map(|i| i.min(ACTIVE) * 9).collect();
    sim.queue.write_buffer(
        &sim.bufs[14],
        constraint_id_words(&sim.capacity) as u64 * 4,
        bytemuck::cast_slice(&offsets),
    );
    sim.queue.write_buffer(
        &sim.bufs[14],
        0,
        bytemuck::cast_slice(&(0u32..ACTIVE * 9).collect::<Vec<_>>()),
    );
    // Production slots >=48 contain scan parameters, not body IDs. This
    // isolated fixture explicitly seeds the serial dispatch table for all 67.
    for id in 0..ACTIVE {
        sim.queue
            .write_buffer(&sim.bufs[22], u64::from(id) * PASS_ALIGN, bytes_of(&id));
    }
    let mut enc = sim.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for id in 0..ACTIVE {
            sim.disp(&mut pass, one, 1, Some(id));
        }
    }
    sim.queue.submit([enc.finish()]);
    let expected: Vec<BodyState> = download(&sim, 0, 0, ACTIVE as usize);
    for (id, body) in expected.iter().enumerate() {
        assert!(
            body.position.iter().any(|x| x.abs() > 1e-5),
            "{layout} fixture body {id} must exercise a nonzero serial solve: {:?}",
            body.position
        );
    }
    for (count, lanes) in [(67u32, 8u32), (4097, 4), (32769, 2), (65536, 1)] {
        let mut order = vec![ACTIVE; count as usize];
        for id in 0..ACTIVE {
            order[(count - ACTIVE + id) as usize] = id;
        }
        sim.queue
            .write_buffer(&sim.bufs[15], 0, bytemuck::cast_slice(&order));
        for repetition in 0..3 {
            sim.queue
                .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&initial));
            sim.queue.write_buffer(&sim.bufs[17], 0, bytes_of(&count));
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, parallel, (count * lanes).into(), Some(0));
            }
            sim.queue.submit([enc.finish()]);
            let actual: Vec<BodyState> = download(&sim, 0, 0, ACTIVE as usize);
            for id in 0..ACTIVE as usize {
                for (field, (got, want)) in actual[id]
                    .position
                    .iter()
                    .chain(actual[id].orientation.iter())
                    .zip(
                        expected[id]
                            .position
                            .iter()
                            .chain(expected[id].orientation.iter()),
                    )
                    .enumerate()
                {
                    assert!((got-want).abs() < 2e-6,
                    "{layout}, {lanes}-lane reduction, repeat {repetition}, bucket count {count}, body {id} at slot {}, pose component {field}: {got} != serial {want}; check stage scratch isolation, lower-triangle transfer, barriers and partial-workgroup coverage", count - ACTIVE + id as u32);
                }
            }
            let fixed: Vec<BodyState> = download(&sim, 0, u64::from(ACTIVE) * 112, 1);
            assert_eq!(
                bytes_of(&fixed[0]),
                bytes_of(&initial[ACTIVE as usize]),
                "{layout}, {lanes}-lane reduction changed the static padding body"
            );
        }
    }
}

/// Independent dense f64 solve of the contact Newton system. This catches a
/// dropped tangent Hessian row, a force projection error, or an A/B sign error
/// even if serial and cooperative GPU implementations share the same mistake.
#[test]
fn contact_primal_matches_dense_newton_system_for_both_endpoints_and_sliding() {
    contact_primal_dense_oracle(false, false, false);
}

#[test]
fn compact_contact_primal_matches_independent_dense_newton_oracle() {
    contact_primal_dense_oracle(true, false, false);
}

#[test]
fn compact_parallel_primal_matches_dense_oracle_with_partial_workgroup() {
    contact_primal_dense_oracle(true, true, false);
}

/// Compact solving must preserve the general scene ABI, not merely the ring's
/// one-collider-per-body case. Compare with the canonical solver after mixing
/// joints, multiple manifolds, dead points, compound collider IDs and spheres.
/// The independent dense Newton tests above separately protect the equations.
#[test]
fn compact_primal_preserves_mixed_compound_contacts_for_all_lane_widths() {
    let mut sim = WgpuSim::new_with_compact_contacts(
        WorldCapacity {
            max_bodies: 65536,
            max_colliders: 32,
            max_contacts: 1024,
            max_candidates: 128,
            ..WorldCapacity::small()
        },
        true,
    )
    .expect("GPU adapter required");
    let initial: Vec<BodyState> = (0..10)
        .map(|id| {
            let angle = 0.03 * (id + 1) as f32;
            BodyState {
                position: [0.01 * id as f32, -0.02, 0.03],
                initial_pos: [-0.01, 0.02, -0.03],
                orientation: [0.0, angle.sin(), 0.0, angle.cos()],
                initial_rot: [0.0, 0.0, 0.0, 1.0],
                inv_mass: if id == 8 { 0.0 } else { 1.0 / (id + 1) as f32 },
                inv_inertia: [2.0, 3.0, 4.0],
                flags: if id == 8 {
                    0
                } else if id == 9 {
                    5
                } else {
                    9
                },
                ..BodyState::zeroed()
            }
        })
        .collect();
    for body in &initial {
        sim.add_body(*body);
    }
    // Two colliders per body, deliberately reversed: collider ID != body ID.
    // One of the two shapes is a sphere for selected bodies, testing the pair
    // flag rather than a cached assumption about the body as a whole.
    for collider in 0..20 {
        sim.add_collider(Collider {
            body: 9 - collider % 10,
            shape: if collider < 10 && collider % 3 == 0 {
                ShapeType::Sphere as u32
            } else {
                ShapeType::Box as u32
            },
            ..Collider::zeroed()
        });
    }
    let mut joints = Vec::new();
    for body in 0..8 {
        for row in 0..4 {
            joints.push(Joint {
                body_a: body,
                body_b: 8,
                local_anchor_a: [0.02 * row as f32, 0.1, -0.15],
                local_anchor_b: [-0.1, 0.04 * row as f32, 0.12],
                k: [2000.0 + row as f32 * 50.0; 6],
                ..Joint::zeroed()
            });
        }
    }
    sim.body_count = 65536; // Inactive padding exercises each adaptive lane width.
    sim.joint_count = joints.len() as u32;
    let mut manifolds = Vec::new();
    let mut contacts = Vec::new();
    let mut edges = Vec::new();
    for body in 0..8u32 {
        edges.extend(body * 4..body * 4 + 4);
        for row in 0..5u32 {
            let own = 9 - body + (row % 2) * 10;
            let other = if row % 2 == 0 { 1 } else { 0 }; // static 8 or sleeping 9
            let (a, b) = if row % 2 == 0 {
                (own, other)
            } else {
                (other, own)
            };
            edges.push(sim.joint_count + manifolds.len() as u32);
            manifolds.push(Manifold {
                collider_a: a,
                collider_b: b,
                contact_start: contacts.len() as u32,
                contact_count: if row == 4 { 0 } else { 3 },
                normal: [0.0, 1.0, 0.0],
                friction: 0.5,
            });
            for point in 0..3 {
                contacts.push(Contact {
                    collider_a: a,
                    collider_b: b,
                    alive: u32::from(point != 1),
                    normal: [0.0, 1.0, 0.0],
                    tangent1: [0.0, 0.0, -1.0],
                    tangent2: [-1.0, 0.0, 0.0],
                    anchor_a: [0.1 * point as f32, -0.3, 0.4],
                    anchor_b: [-0.2, 0.1 * row as f32, -0.1],
                    c0_n: -0.02,
                    c0_t1: 0.03,
                    c0_t2: -0.01,
                    friction: 0.5,
                    lambda_n: -100.0,
                    lambda_t1: if point == 0 { 400.0 } else { 10.0 },
                    lambda_t2: -7.0,
                    k_n: 2000.0,
                    k_t: 3000.0,
                    _pad: [0.0, 0.0, 4000.0],
                    ..Contact::zeroed()
                });
            }
        }
    }
    sim.queue
        .write_buffer(&sim.bufs[5], 0, bytemuck::cast_slice(&joints));
    sim.queue
        .write_buffer(&sim.bufs[14], 0, bytemuck::cast_slice(&edges));
    let offsets: Vec<u32> = (0..=65536).map(|i| i.min(8) * 9).collect();
    sim.queue.write_buffer(
        &sim.bufs[14],
        u64::from(constraint_id_words(&sim.capacity)) * 4,
        bytemuck::cast_slice(&offsets),
    );
    sim.queue.write_buffer(
        &sim.bufs[15],
        0,
        bytemuck::cast_slice(&(0u32..65536).collect::<Vec<_>>()),
    );
    // Give each active body its current inertial pose, so the test emphasizes
    // constraint accumulation while retaining nonzero step-start displacements.
    let inertial: Vec<[f32; 8]> = initial
        .iter()
        .map(|b| {
            [
                b.position[0],
                b.position[1],
                b.position[2],
                b.orientation[3],
                b.orientation[0],
                b.orientation[1],
                b.orientation[2],
                0.0,
            ]
        })
        .collect();
    sim.queue
        .write_buffer(&sim.bufs[9], 0, bytemuck::cast_slice(&inertial));
    for parity in 0..2 {
        sim.contacts_parity = parity;
        sim.write_host_uniforms();
        let cur = parity * sim.capacity.max_contacts;
        let mut current_manifolds = manifolds.clone();
        for mf in &mut current_manifolds {
            mf.contact_start += cur;
        }
        sim.queue.write_buffer(
            &sim.bufs[4],
            u64::from(cur) * 128,
            bytemuck::cast_slice(&contacts),
        );
        sim.queue
            .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&current_manifolds));
        sim.queue.write_buffer(
            &sim.bufs[7],
            0,
            bytes_of(&StepCounters {
                body_count: sim.body_count,
                collider_count: sim.collider_count,
                joint_count: sim.joint_count,
                contact_count: contacts.len() as u32,
                manifold_count: manifolds.len() as u32,
                color_count: 1,
                ..Default::default()
            }),
        );
        sim.queue
            .write_buffer(&sim.bufs[14], 0, bytemuck::cast_slice(&edges));
        sim.queue
            .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&initial));
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for body in 0..8 {
                sim.disp(&mut pass, "solve_primal_one", 1, Some(body));
            }
            sim.disp(
                &mut pass,
                "clear_solver_contacts",
                contacts.len() as u64,
                None,
            );
            sim.disp(
                &mut pass,
                "pack_solver_contacts",
                manifolds.len() as u64,
                None,
            );
        }
        sim.queue.submit([enc.finish()]);
        let expected: Vec<BodyState> = download(&sim, 0, 0, initial.len());
        assert!(
            expected[..8]
                .iter()
                .zip(&initial)
                .all(|(a, b)| a.position != b.position),
            "fixture must exercise a nonzero mixed contact/joint solve for every active body"
        );
        // A lane may cross a manifold->joint->empty-manifold->live-manifold
        // boundary. No prior pair cursor or point may leak across that transition.
        // Compare with the ordered reference, not another solver sharing the
        // same reordered traversal, so a common serial/cooperative bug fails.
        for (order_name, order) in [
            ("ordered", [0usize, 1, 2, 3, 4, 5, 6, 7, 8]),
            ("interleaved", [4, 0, 5, 1, 6, 2, 8, 7, 3]),
            ("reverse", [8, 7, 6, 5, 4, 3, 2, 1, 0]),
        ] {
            let shuffled: Vec<u32> = edges
                .chunks_exact(9)
                .flat_map(|row| order.map(|i| row[i]))
                .collect();
            sim.queue
                .write_buffer(&sim.bufs[14], 0, bytemuck::cast_slice(&shuffled));
            for (count, lanes) in [
                (0u32, 0u32),
                (31, 8),
                (4096, 4),
                (32768, 2),
                (65536, 1),
                (31, 9),
            ] {
                sim.queue
                    .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&initial));
                sim.queue.write_buffer(&sim.bufs[17], 0, bytes_of(&count));
                let mut enc = sim.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    if lanes == 0 {
                        for body in 0..8 {
                            sim.disp(&mut pass, "solve_primal_one_compact", 1, Some(body));
                        }
                    } else if lanes == 9 {
                        // sparse tail: fixed eight lanes, multiple chunks
                        sim.disp(&mut pass, "solve_primal_sparse_compact", 64, Some(0));
                    } else {
                        sim.disp(
                            &mut pass,
                            "solve_primal_color_compact",
                            u64::from(count * lanes),
                            Some(0),
                        );
                    }
                }
                sim.queue.submit([enc.finish()]);
                let actual: Vec<BodyState> = download(&sim, 0, 0, initial.len());
                for (body, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    for (component, (got, want)) in a
                        .position
                        .iter()
                        .chain(a.orientation.iter())
                        .zip(e.position.iter().chain(e.orientation.iter()))
                        .enumerate()
                    {
                        assert!((got-want).abs() < 3e-6,
                        "compact mixed solver: order={order_name}, half={parity}, lanes={lanes} (0=serial), body={body}, pose component={component}, actual={got}, canonical={want}; check collider-to-body mapping, sphere step-start lever arms, dead/empty point ranges, manifold-to-joint cursor reset, endpoint signs and lane reduction");
                    }
                }
                assert_eq!(
                    bytemuck::bytes_of(&actual[8]),
                    bytemuck::bytes_of(&initial[8]),
                    "static endpoint was changed by compact solve"
                );
                assert_eq!(
                    bytemuck::bytes_of(&actual[9]),
                    bytemuck::bytes_of(&initial[9]),
                    "sleeping endpoint was changed by compact solve"
                );
            }
        }
    }
}

#[test]
fn persistent_contact_primal_matches_independent_dense_newton_oracle() {
    contact_primal_dense_oracle(true, false, true);
}

#[test]
fn persistent_parallel_primal_matches_dense_oracle_with_partial_workgroup() {
    contact_primal_dense_oracle(true, true, true);
}

#[test]
fn moving_contact_primal_and_dual_match_independent_dense_oracle() {
    for (compact, persistent) in [(false, false), (true, false), (true, true)] {
        for parallel in [false, true] {
            contact_primal_motion_oracle(compact, parallel, persistent, true);
        }
    }
}

fn contact_primal_dense_oracle(compact: bool, parallel: bool, persistent: bool) {
    contact_primal_motion_oracle(compact, parallel, persistent, false);
}

fn contact_primal_motion_oracle(compact: bool, parallel: bool, persistent: bool, moving: bool) {
    let mut sim = if persistent {
        WgpuSim::new_with_persistent_contacts(WorldCapacity::small())
    } else {
        WgpuSim::new_with_compact_contacts(WorldCapacity::small(), compact)
    }
    .expect("GPU adapter required");
    sim.params.dt = 0.05;
    sim.params.trust_lin = 100.0;
    sim.params.trust_ang = 100.0;
    let initial_a = if moving {
        [0.10, -0.06, 0.08, (1.0f32 - 0.02).sqrt()]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let dynamic = BodyState {
        orientation: [0.0, 0.0, 0.0, 1.0],
        initial_rot: initial_a,
        initial_pos: if moving {
            [-0.012, 0.009, -0.02]
        } else {
            [0.0; 3]
        },
        inv_mass: 0.5,
        inv_inertia: [2.0, 3.0, 4.0],
        flags: 9,
        ..BodyState::zeroed()
    };
    sim.add_body(dynamic);
    sim.add_body(BodyState {
        inv_mass: 0.0,
        flags: 0,
        ..dynamic
    });
    for body in 0..2 {
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            ..Collider::zeroed()
        });
    }
    sim.write_host_uniforms();
    // Identity inertial pose, no inertial residual. The moment arm makes all
    // rotational rows nonzero and couples translation to rotation.
    sim.queue.write_buffer(
        &sim.bufs[9],
        0,
        bytemuck::cast_slice(&[0f32, 0., 0., 1., 0., 0., 0., 0.]),
    );
    sim.queue.write_buffer(&sim.bufs[14], 0, bytes_of(&0u32));
    sim.queue.write_buffer(
        &sim.bufs[14],
        u64::from(constraint_id_words(&sim.capacity)) * 4,
        bytemuck::cast_slice(&[0u32, 1, 1]),
    );
    let cases = if moving {
        vec![
            (0u32, false),
            (5, false),
            (9, false),
            (0, true),
            (5, true),
            (9, true),
        ]
    } else {
        vec![(0, false)]
    };
    for (other_flags, sphere) in cases {
        // Poison inactive reference poses too: static/sleeping displacement must
        // be zero, while sphere lever arms still use a dynamic body's initial rotation.
        // Negative quaternion hemisphere represents the same physical rotation.
        let other = BodyState {
            flags: other_flags,
            inv_mass: if other_flags == 0 { 0.0 } else { 0.25 },
            initial_pos: if moving {
                [0.025, -0.01, 0.03]
            } else {
                [0.0; 3]
            },
            initial_rot: if moving {
                [-0.04, -0.08, 0.06, -(1.0f32 - 0.0116).sqrt()]
            } else {
                [0.0, 0.0, 0.0, 1.0]
            },
            ..dynamic
        };
        sim.queue.write_buffer(&sim.bufs[0], 112, bytes_of(&other));
        let cols: Vec<Collider> = (0..2)
            .map(|body| Collider {
                body,
                shape: if sphere && body == 0 {
                    ShapeType::Sphere as u32
                } else {
                    ShapeType::Box as u32
                },
                ..Collider::zeroed()
            })
            .collect();
        sim.queue
            .write_buffer(&sim.bufs[2], 0, bytemuck::cast_slice(&cols));
        for is_a in [true, false] {
            for basis_case in 0..2 {
                for sliding in [false, true] {
                    // The oblique frame makes every translational off-diagonal
                    // nonzero under unequal row penalties. Axis-plane normals alone
                    // cannot catch mistakes in all packed Hessian indices.
                    let inv_sqrt10 = 1.0f32 / 10.0f32.sqrt();
                    let mut c = Contact {
                        collider_a: if is_a { 0 } else { 1 },
                        collider_b: if is_a { 1 } else { 0 },
                        alive: 1,
                        anchor_a: [0.2, -0.3, 0.4],
                        anchor_b: if moving {
                            [-0.35, 0.25, 0.18]
                        } else {
                            [0.2, -0.3, 0.4]
                        },
                        normal: [0.6, 0.8, 0.0],
                        tangent1: if compact {
                            [0.0, 0.0, -1.0]
                        } else {
                            [-0.8, 0.6, 0.0]
                        },
                        tangent2: if compact {
                            [-0.8, 0.6, 0.0]
                        } else {
                            [0.0, 0.0, 1.0]
                        },
                        friction: 0.5,
                        c0_n: 0.01,
                        c0_t1: -0.02,
                        c0_t2: 0.03,
                        lambda_n: -100.0,
                        lambda_t1: if sliding { 400.0 } else { 10.0 },
                        lambda_t2: -7.0,
                        k_n: 2000.0,
                        k_t: 3000.0,
                        _pad: [0.0, 0.0, 4000.0],
                        ..Contact::zeroed()
                    };
                    if basis_case == 1 {
                        c.normal = [3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0];
                        c.tangent1 = [0.0, 3.0 * inv_sqrt10, -inv_sqrt10];
                        c.tangent2 = [
                            -40.0 / 13.0 * inv_sqrt10,
                            3.0 / 13.0 * inv_sqrt10,
                            9.0 / 13.0 * inv_sqrt10,
                        ];
                    }
                    let mf = Manifold {
                        collider_a: c.collider_a,
                        collider_b: c.collider_b,
                        contact_count: 1,
                        normal: c.normal,
                        friction: c.friction,
                        ..Manifold::zeroed()
                    };
                    sim.queue.write_buffer(&sim.bufs[0], 0, bytes_of(&dynamic));
                    write_contact_fixture(&sim, 0, &[c], &[0]);
                    sim.queue.write_buffer(&sim.bufs[25], 0, bytes_of(&mf));
                    let mut enc = sim.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        if compact {
                            sim.queue.write_buffer(
                                &sim.bufs[7],
                                0,
                                bytes_of(&StepCounters {
                                    body_count: 2,
                                    contact_count: 1,
                                    manifold_count: 1,
                                    ..Default::default()
                                }),
                            );
                            if !persistent {
                                sim.disp(&mut pass, "clear_solver_contacts", 1, None);
                            }
                            sim.disp(&mut pass, "pack_solver_contacts", 1, None);
                        }
                        if parallel {
                            sim.queue.write_buffer(&sim.bufs[15], 0, bytes_of(&0u32));
                            sim.queue.write_buffer(&sim.bufs[17], 0, bytes_of(&1u32));
                            sim.disp(
                                &mut pass,
                                if compact {
                                    "solve_primal_color_compact"
                                } else {
                                    "solve_primal_color"
                                },
                                64,
                                Some(0),
                            );
                        } else {
                            sim.disp(
                                &mut pass,
                                if compact {
                                    "solve_primal_one_compact"
                                } else {
                                    "solve_primal_one"
                                },
                                1,
                                Some(0),
                            );
                        }
                    }
                    sim.queue.submit([enc.finish()]);
                    let actual: BodyState = download(&sim, 0, 0, 1)[0];

                    let k = [c.k_n as f64, c.k_t as f64, c._pad[2] as f64];
                    let c0 = [c.c0_n as f64, c.c0_t1 as f64, c.c0_t2 as f64];
                    let mut f = [c.lambda_n as f64, c.lambda_t1 as f64, c.lambda_t2 as f64];
                    // Independent row-wise J*dq calculation in f64. Keep this separate
                    // from any point-displacement factorization used by the shader.
                    let endpoint_a = if is_a { dynamic } else { other };
                    let endpoint_b = if is_a { other } else { dynamic };
                    let lever = |body: BodyState, anchor: [f32; 3]| {
                        let q = if sphere && body.flags & 1 != 0 {
                            body.initial_rot
                        } else {
                            body.orientation
                        };
                        let [x, y, z, w] = q.map(f64::from);
                        let [a, b, c] = anchor.map(f64::from);
                        [
                            (1.0 - 2.0 * (y * y + z * z)) * a
                                + 2.0 * (x * y - z * w) * b
                                + 2.0 * (x * z + y * w) * c,
                            2.0 * (x * y + z * w) * a
                                + (1.0 - 2.0 * (x * x + z * z)) * b
                                + 2.0 * (y * z - x * w) * c,
                            2.0 * (x * z - y * w) * a
                                + 2.0 * (y * z + x * w) * b
                                + (1.0 - 2.0 * (x * x + y * y)) * c,
                        ]
                    };
                    let displacement = |body: BodyState| -> [f64; 6] {
                        if body.flags & 1 == 0 || body.flags & 4 != 0 {
                            return [0.0; 6];
                        }
                        // Current orientation is identity in this fixture; qsub's
                        // hemisphere-corrected vector is -2*initial.xyz (or its negation).
                        let sign = if body.initial_rot[3] < 0.0 { 2.0 } else { -2.0 };
                        [
                            f64::from(body.position[0]) - f64::from(body.initial_pos[0]),
                            f64::from(body.position[1]) - f64::from(body.initial_pos[1]),
                            f64::from(body.position[2]) - f64::from(body.initial_pos[2]),
                            sign * f64::from(body.initial_rot[0]),
                            sign * f64::from(body.initial_rot[1]),
                            sign * f64::from(body.initial_rot[2]),
                        ]
                    };
                    let ra = lever(endpoint_a, c.anchor_a);
                    let rb = lever(endpoint_b, c.anchor_b);
                    let da = displacement(endpoint_a);
                    let db = displacement(endpoint_b);
                    let mut residual = [0.0f64; 3];
                    for (row, direction) in [c.normal, c.tangent1, c.tangent2].iter().enumerate() {
                        let n = direction.map(f64::from);
                        let ja = [
                            n[0],
                            n[1],
                            n[2],
                            ra[1] * n[2] - ra[2] * n[1],
                            ra[2] * n[0] - ra[0] * n[2],
                            ra[0] * n[1] - ra[1] * n[0],
                        ];
                        let jb = [
                            n[0],
                            n[1],
                            n[2],
                            rb[1] * n[2] - rb[2] * n[1],
                            rb[2] * n[0] - rb[0] * n[2],
                            rb[0] * n[1] - rb[1] * n[0],
                        ];
                        let delta: f64 = (0..6).map(|i| jb[i] * db[i] - ja[i] * da[i]).sum();
                        residual[row] = c0[row] * (1.0 - sim.params.alpha as f64) + delta;
                        f[row] += k[row] * residual[row];
                    }
                    f[0] = f[0].min(0.0);
                    let limit = -f[0] * c.friction as f64;
                    let tangent = f[1].hypot(f[2]);
                    if tangent > limit {
                        f[1] *= limit / tangent;
                        f[2] *= limit / tangent;
                    }
                    if moving && !parallel {
                        // Evaluate dual on the SAME input poses, not the updated
                        // primal result. Its oracle is the independently projected
                        // J*dq force above, not another GPU storage-layout variant.
                        sim.queue.write_buffer(&sim.bufs[0], 0, bytes_of(&dynamic));
                        sim.queue.write_buffer(
                            &sim.bufs[7],
                            0,
                            bytes_of(&StepCounters {
                                body_count: 2,
                                contact_count: 1,
                                manifold_count: 1,
                                ..Default::default()
                            }),
                        );
                        let mut enc = sim.device.create_command_encoder(&Default::default());
                        {
                            let mut pass = enc.begin_compute_pass(&Default::default());
                            sim.disp(
                                &mut pass,
                                if compact {
                                    "solve_dual_compact"
                                } else {
                                    "solve_dual"
                                },
                                1,
                                Some(1),
                            );
                        }
                        sim.queue.submit([enc.finish()]);
                        let forces: [f32; 3] = if compact {
                            let hot: Vec<f32> = download(&sim, 27, 0, 16);
                            [hot[8], hot[9], hot[10]]
                        } else {
                            let contact: Contact = download(&sim, 4, 0, 1)[0];
                            assert!((f64::from(contact.penetration) - residual[0]).abs() < 2e-6,
                            "dual normal residual differs: endpointA={is_a}, neighbor_flags={other_flags}, sphere={sphere}, got={}, expected={}; check angular/linear displacement signs", contact.penetration, residual[0]);
                            [contact.lambda_n, contact.lambda_t1, contact.lambda_t2]
                        };
                        for row in 0..3 {
                            let tolerance = 2e-4 + 2e-6 * f[row].abs();
                            assert!((f64::from(forces[row]) - f[row]).abs() < tolerance,
                            "dual force differs from independent J*dq oracle: basis={basis_case}, compact={compact}, persistent={persistent}, endpointA={is_a}, neighbor_flags={other_flags}, sphere={sphere}, sliding={sliding}, row={row}, got={}, expected={}, residual={:?}; check point displacement, sphere reference rotation and cone projection", forces[row], f[row], residual);
                        }
                    }
                    let mut system = [[0.0f64; 7]; 6];
                    let dt2 = (sim.params.dt as f64).powi(2);
                    for i in 0..6 {
                        system[i][i] = if i < 3 {
                            2.0 / dt2
                        } else {
                            1.0 / dynamic.inv_inertia[i - 3] as f64 / dt2
                        };
                    }
                    let r = if is_a { ra } else { rb };
                    for (row, direction) in [c.normal, c.tangent1, c.tangent2].iter().enumerate() {
                        let n = direction.map(f64::from);
                        let mut j = [
                            n[0],
                            n[1],
                            n[2],
                            r[1] * n[2] - r[2] * n[1],
                            r[2] * n[0] - r[0] * n[2],
                            r[0] * n[1] - r[1] * n[0],
                        ];
                        if is_a {
                            for v in &mut j {
                                *v = -*v;
                            }
                        }
                        for i in 0..6 {
                            system[i][6] -= j[i] * f[row];
                            for col in 0..6 {
                                system[i][col] += k[row] * j[i] * j[col];
                            }
                        }
                    }
                    if basis_case == 1 {
                        for row in 1..3 {
                            for col in 0..row {
                                assert!(system[row][col].abs() > 1.0,
                                "oblique oracle must exercise every translational off-diagonal: row={row}, col={col}, value={}",system[row][col]);
                            }
                        }
                    }
                    // Dense Gaussian elimination, independent of the GPU's equilibrated LDLT.
                    for pivot in 0..6 {
                        let divisor = system[pivot][pivot];
                        for col in pivot..7 {
                            system[pivot][col] /= divisor;
                        }
                        for row in 0..6 {
                            if row == pivot {
                                continue;
                            }
                            let factor = system[row][pivot];
                            for col in pivot..7 {
                                system[row][col] -= factor * system[pivot][col];
                            }
                        }
                    }
                    let q = [
                        system[3][6] * 0.5,
                        system[4][6] * 0.5,
                        system[5][6] * 0.5,
                        1.0,
                    ];
                    let qnorm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let expected = [
                        system[0][6],
                        system[1][6],
                        system[2][6],
                        q[0] / qnorm,
                        q[1] / qnorm,
                        q[2] / qnorm,
                        q[3] / qnorm,
                    ];
                    for (component, (got, want)) in actual
                        .position
                        .iter()
                        .chain(actual.orientation.iter())
                        .zip(expected)
                        .enumerate()
                    {
                        assert!((*got as f64 - want).abs() < 3e-6,
                    "contact Newton solve differs from dense CPU oracle: basis={basis_case}, compact={compact}, persistent={persistent}, parallel={parallel}, moving={moving}, neighbor_flags={other_flags}, sphere={sphere}, endpoint={}, sliding={sliding}, pose component={component}, actual={got}, expected={want}; check angular/linear displacement signs, hemisphere, inactive endpoints, sphere reference arms, all three Hessian rows and cone projection",
                    if is_a { "A" } else { "B" });
                    }
                }
            }
        }
    }
}

#[test]
fn bulk_scene_upload_preserves_append_indices_and_previous_poses() {
    let mut sim = WgpuSim::new(WorldCapacity::small()).expect("GPU adapter required");
    let bodies: Vec<BodyState> = (0..3)
        .map(|id| BodyState {
            position: [id as f32 * 2.0, 0.75, -3.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            initial_pos: [id as f32 * 2.0, 0.75, -3.0],
            initial_rot: [0.0, 0.0, 0.0, 1.0],
            flags: if id == 0 { 0 } else { 9 },
            inv_mass: if id == 0 { 0.0 } else { 4.0 },
            linear_vel: [id as f32, 0.0, 0.0],
            ..BodyState::zeroed()
        })
        .collect();
    let colliders: Vec<Collider> = (0..3)
        .map(|body| Collider {
            body,
            shape: ShapeType::Box as u32,
            params: [0.5, 0.25, 0.25, 0.6123724],
            local_rot: [0.0, 0.0, 0.0, 1.0],
            ..Collider::zeroed()
        })
        .collect();
    assert_eq!(sim.add_body(bodies[0]), 0);
    assert_eq!(sim.add_bodies(&bodies[1..]), 1..3);
    assert_eq!(
        sim.add_bodies(&[]),
        3..3,
        "empty upload must not consume body indices"
    );
    assert_eq!(sim.add_collider(colliders[0]), 0);
    assert_eq!(sim.add_colliders(&colliders[1..]), 1..3);
    assert_eq!(
        sim.add_colliders(&[]),
        3..3,
        "empty upload must not consume collider indices"
    );
    let current: Vec<BodyState> = download(&sim, 0, 0, 3);
    let previous: Vec<BodyState> = download(&sim, 0, u64::from(sim.capacity.max_bodies) * 112, 3);
    let actual_cols: Vec<Collider> = download(&sim, 2, 0, 3);
    let expected_bodies: &[u8] = bytemuck::cast_slice(&bodies);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&current),
        expected_bodies,
        "bulk upload overwrote existing bodies or used the wrong append offset"
    );
    assert_eq!(bytemuck::cast_slice::<_, u8>(&previous), expected_bodies,
        "bulk upload must initialize previous poses too; stale history corrupts the first velocity/warm-start step");
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&actual_cols),
        bytemuck::cast_slice::<_, u8>(&colliders),
        "bulk collider upload changed body associations or append order"
    );
    assert_eq!(sim.body_count, 3);
    assert_eq!(sim.collider_count, 3);
}

#[test]
fn incremental_coloring_reuses_valid_history_and_repairs_new_conflicts() {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 16,
        max_colliders: 16,
        max_contacts: 1024,
        max_candidates: 1024,
        max_joints: 8,
        ..WorldCapacity::small()
    })
    .expect("GPU adapter required for incremental coloring regression");
    let mut bodies: Vec<BodyState> = (0..8)
        .map(|id| BodyState {
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_mass: if id == 0 { 0.0 } else { 1.0 },
            flags: if id == 0 { 0 } else { 9 },
            ..BodyState::zeroed()
        })
        .collect();
    sim.add_bodies(&bodies);
    let colliders: Vec<Collider> = (0..8)
        .map(|id| Collider {
            body: id,
            ..Collider::zeroed()
        })
        .collect();
    sim.add_colliders(&colliders);
    sim.joint_count = 1;
    let joint = Joint {
        body_a: 1,
        body_b: 2,
        ..Joint::zeroed()
    };
    sim.queue.write_buffer(&sim.bufs[5], 0, bytes_of(&joint));
    let history_offset = (sim.capacity.max_bodies * sim.capacity.max_colors) as u64 * 4;
    let csr_offset = u64::from(constraint_id_words(&sim.capacity)) * 4;

    for (case, expected_pending) in [
        ("stable", 0),
        ("new contact", 1),
        ("waking neighbour", 1),
        ("scene reset", 6),
        ("new body", 1),
        ("invalid history", 1),
    ] {
        let waking = case == "waking neighbour";
        bodies[7].flags = if waking { 9 } else { 5 };
        sim.queue
            .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
        // Includes a joint, contacts, a static endpoint and a sleeping endpoint.
        let mut edges = vec![(1u32, 2u32), (2, 3), (3, 4), (4, 5), (1, 0), (1, 7)];
        if case == "new contact" {
            edges.push((1, 3));
        }
        let manifolds: Vec<Manifold> = edges[1..]
            .iter()
            .map(|&(a, b)| Manifold {
                collider_a: a,
                collider_b: b,
                ..Manifold::zeroed()
            })
            .collect();
        sim.queue
            .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&manifolds));
        let mut offsets = vec![0u32];
        let mut adjacency = Vec::new();
        for body in 0..8 {
            for (id, &(a, b)) in edges.iter().enumerate() {
                if body == a || body == b {
                    adjacency.push(id as u32);
                }
            }
            offsets.push(adjacency.len() as u32);
        }
        sim.queue
            .write_buffer(&sim.bufs[14], 0, bytemuck::cast_slice(&adjacency));
        sim.queue
            .write_buffer(&sim.bufs[14], csr_offset, bytemuck::cast_slice(&offsets));
        // Deliberately scramble awake-list order: the compact work list must
        // contain body IDs, not list indices. Body 6 has no edges.
        let mut awake = vec![6u32, 3, 1, 5, 2, 4];
        if waking {
            awake.push(7);
        }
        sim.queue
            .write_buffer(&sim.bufs[11], 0, bytemuck::cast_slice(&awake));
        let mut history = vec![1u32, 1, 2, 1, 2, 32, 8, 1];
        if case == "new body" {
            history[6] = 0;
        }
        if case == "invalid history" {
            history[6] = 33;
        }
        sim.queue.write_buffer(
            &sim.bufs[15],
            history_offset,
            bytemuck::cast_slice(&history),
        );
        sim.steps_done = if case == "scene reset" { 0 } else { 1 };
        sim.write_host_uniforms();
        let counters = StepCounters {
            body_count: 8,
            collider_count: 8,
            joint_count: 1,
            awake_count: awake.len() as u32,
            manifold_count: manifolds.len() as u32,
            ..StepCounters::default()
        };
        sim.queue.write_buffer(&sim.bufs[7], 0, bytes_of(&counters));
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            sim.disp(&mut pass, "csr_sort", awake.len() as u64, None);
            sim.disp(&mut pass, "init_colors", awake.len() as u64, None);
            sim.disp(&mut pass, "write_work_args", 64, None);
        }
        sim.queue.submit([enc.finish()]);
        let before: Vec<u32> = download(&sim, 12, 0, 8);
        if case == "stable" {
            assert_eq!(before[6], 0,
                "isolated body retained a fragmented high color; local maxima must compact toward the smallest available color");
            assert_eq!(
                before[5], 0,
                "local maximum with a color-one neighbour failed to compact its retained color 31"
            );
        }
        let counters: Vec<StepCounters> = download(&sim, 7, 0, 1);
        let pending = counters[0]._pad0;
        assert_eq!(pending, expected_pending,
            "{case}: pending count must contain only invalidated bodies; stable graphs must skip every JP round");
        let args: Vec<u32> = download(&sim, 23, 69 * 12, 3);
        assert_eq!(
            args,
            vec![u32::from(pending != 0), 1, 1],
            "{case}: slot 69 did not publish the current compact recoloring workload"
        );
        let unchanged: Vec<u32> = download(&sim, 15, history_offset, 8);
        assert_eq!(
            unchanged, history,
            "{case}: initialization mutated the shared history while neighbours were reading it"
        );
        let mut expected_ids: Vec<u32> = awake
            .iter()
            .copied()
            .filter(|&id| before[id as usize] == u32::MAX)
            .collect();
        expected_ids.sort_unstable();
        if pending > 0 {
            let mut work: Vec<u32> = download(&sim, 20, 0, pending as usize);
            work.sort_unstable();
            assert_eq!(
                work, expected_ids,
                "{case}: compact recoloring list lost or duplicated a body ID"
            );
        }
        let mut enc = sim.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..32 {
                sim.disp_work(&mut pass, "color_bodies", 69, None);
            }
            sim.disp(&mut pass, "pack_color_lists", awake.len() as u64, None);
            sim.disp(&mut pass, "write_color_args", 64, None);
        }
        sim.queue.submit([enc.finish()]);
        let colors: Vec<u32> = download(&sim, 12, 0, 8);
        let saved: Vec<u32> = download(&sim, 15, history_offset, 8);
        let counters: Vec<StepCounters> = download(&sim, 7, 0, 1);
        assert_eq!(
            counters[0].overflow_flags, 0,
            "{case}: coloring did not finish within its palette/round budget"
        );
        for &body in &awake {
            let id = body as usize;
            assert!(
                colors[id] < 32,
                "{case}: active body {body} still has invalid color {}",
                colors[id]
            );
            assert_eq!(
                saved[id],
                colors[id] + 1,
                "{case}: body {body} history was overwritten by packed color indices"
            );
            if before[id] != u32::MAX {
                assert_eq!(
                    colors[id], before[id],
                    "{case}: valid color for body {body} changed during recoloring"
                );
            }
        }
        assert_eq!(
            counters[0].color_count,
            awake.iter().map(|&b| colors[b as usize] + 1).max().unwrap(),
            "{case}: retained high colors were omitted from the published color count"
        );
        for &(a, b) in &edges {
            if awake.contains(&a) && awake.contains(&b) {
                assert_ne!(colors[a as usize], colors[b as usize],
                    "{case}: active edge ({a}, {b}) shares a color; concurrent primal solves would race");
            }
        }
    }
}

/// Compare every published word, including cold feature/generation metadata,
/// across both ping-pong halves. Empty reserved contact slots must stay dead.
#[test]
fn compact_dual_preserves_metadata_and_publishes_only_on_final_iteration() {
    let mut sim = WgpuSim::new_with_compact_contacts(WorldCapacity::small(), true)
        .expect("GPU adapter required");
    for id in 0..2 {
        sim.add_body(BodyState {
            position: [id as f32 * 0.02, 0.01, -0.03],
            orientation: [0.0, 0.0, 0.0, 1.0],
            initial_rot: [0.0, 0.0, 0.0, 1.0],
            inv_mass: if id == 0 { 0.0 } else { 1.0 },
            flags: if id == 0 { 0 } else { 9 },
            ..BodyState::zeroed()
        });
    }
    // IDs are intentionally unrelated to body IDs; colliders 0 and 2 share a body.
    for body in [1, 0, 1] {
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            ..Collider::zeroed()
        });
    }
    // Exercise state-dependent paths as well as storage publication. In particular,
    // sphere contacts must use step-start lever arms, and two inactive endpoints
    // must preserve forces instead of publishing stale scratch data.
    for (case, dynamic_a, sleeping, sphere, reversed) in [
        ("static-box/awake-box", false, false, false, false),
        ("awake-box/awake-box", true, false, false, false),
        ("rotating-sphere/awake-box", true, false, true, false),
        (
            "reversed-awake-box/rotating-sphere",
            true,
            false,
            true,
            true,
        ),
        ("sleeping-box/sleeping-box", true, true, false, false),
        ("static-box/sleeping-box", false, true, false, false),
    ] {
        let bodies: Vec<BodyState> = (0..2)
            .map(|id| {
                let dynamic = id != 0 || dynamic_a;
                BodyState {
                    position: [id as f32 * 0.02, 0.01, -0.03],
                    initial_pos: [0.01, -0.02, 0.03],
                    orientation: [0.0, (0.2f32).sin(), 0.0, (0.2f32).cos()],
                    initial_rot: [0.0, 0.0, 0.0, 1.0],
                    inv_mass: if dynamic { 1.0 } else { 0.0 },
                    flags: if !dynamic {
                        0
                    } else if sleeping {
                        5
                    } else {
                        9
                    },
                    ..BodyState::zeroed()
                }
            })
            .collect();
        sim.queue
            .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
        let colliders: Vec<Collider> = [1, 0, 1]
            .into_iter()
            .map(|body| Collider {
                body,
                shape: if sphere && body == 0 {
                    ShapeType::Sphere as u32
                } else {
                    ShapeType::Box as u32
                },
                ..Collider::zeroed()
            })
            .collect();
        sim.queue
            .write_buffer(&sim.bufs[2], 0, bytemuck::cast_slice(&colliders));
        let (collider_a, collider_b) = if reversed { (2, 1) } else { (1, 2) };
        for parity in 0..2 {
            sim.contacts_parity = parity;
            sim.write_host_uniforms();
            let cur = parity * sim.capacity.max_contacts;
            let offset = cur as u64 * 128;
            let mut contacts = vec![Contact::zeroed(); 4];
            for i in [0usize, 2] {
                contacts[i] = Contact {
                    collider_a,
                    collider_b,
                    feature_id: 0xfabc0000 + i as u32,
                    alive: 1,
                    normal: [0.0, 1.0, 0.0],
                    tangent1: [0.0, 0.0, -1.0],
                    tangent2: [-1.0, 0.0, 0.0],
                    anchor_a: [0.2, -0.3, 0.4],
                    anchor_b: [-0.1, 0.3, 0.2],
                    c0_n: -0.02,
                    c0_t1: 0.03,
                    c0_t2: -0.01,
                    friction: 0.5,
                    lambda_n: -12.0,
                    lambda_t1: if i == 0 { 0.1 } else { 100.0 },
                    lambda_t2: 0.3,
                    k_n: 100.0,
                    k_t: 10.0,
                    _pad: [1.0, f32::from_bits(0xabcde123), 30.0],
                    ..Contact::zeroed()
                };
            }
            // Slot 1 is a reserved hole outside any live manifold; slot 3 is a dead
            // point inside a manifold. Both must ignore stale compact buffer data.
            let manifolds = [
                Manifold {
                    collider_a,
                    collider_b,
                    contact_start: cur,
                    contact_count: 1,
                    normal: [0., 1., 0.],
                    friction: 0.5,
                },
                Manifold {
                    collider_a,
                    collider_b,
                    contact_start: cur + 2,
                    contact_count: 2,
                    normal: [0., 1., 0.],
                    friction: 0.5,
                },
            ];
            sim.queue.write_buffer(
                &sim.bufs[7],
                0,
                bytes_of(&StepCounters {
                    body_count: 2,
                    contact_count: 4,
                    manifold_count: 2,
                    ..Default::default()
                }),
            );
            sim.queue
                .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&manifolds));
            sim.queue
                .write_buffer(&sim.bufs[4], offset, bytemuck::cast_slice(&contacts));
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..4 {
                    sim.disp(&mut pass, "solve_dual", 4, None);
                }
            }
            sim.queue.submit([enc.finish()]);
            let expected: Vec<u32> = download(&sim, 4, offset, 4 * 32);
            if sleeping {
                assert_eq!(expected, bytemuck::cast_slice::<Contact, u32>(&contacts),
                "{case} half={parity}: inactive endpoints changed canonical force/penalty or geometry");
            }
            sim.queue
                .write_buffer(&sim.bufs[4], offset, bytemuck::cast_slice(&contacts));
            sim.queue.write_buffer(
                &sim.bufs[27],
                0,
                bytemuck::cast_slice(&vec![0x12345678u32; 4 * 16]),
            );
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "clear_solver_contacts", 4, None);
                sim.disp(&mut pass, "pack_solver_contacts", 2, None);
                for _ in 0..3 {
                    sim.disp(&mut pass, "solve_dual_compact", 4, Some(0));
                }
            }
            sim.queue.submit([enc.finish()]);
            let untouched: Vec<u32> = download(&sim, 4, offset, 4 * 32);
            assert_eq!(
            untouched,
            bytemuck::cast_slice::<Contact, u32>(&contacts),
            "{case} half={parity}: intermediate compact iterations must not publish canonical contacts"
        );
            let points: Vec<u32> = download(&sim, 27, 0, 4 * 16);
            assert_eq!(
                points[1 * 16 + 15] & 0x7fffffff,
                0x7fffffff,
                "{case} half={parity}: reserved hole reused a stale pair index"
            );
            assert_eq!(
                points[3 * 16 + 15] & 0x7fffffff,
                0x7fffffff,
                "{case} half={parity}: dead manifold point became live"
            );
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "solve_dual_compact", 4, Some(1));
            }
            sim.queue.submit([enc.finish()]);
            let actual: Vec<u32> = download(&sim, 4, offset, 4 * 32);
            for (word, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                // Every integer/cold word must match exactly. Updated float words
                // allow only floating expression reassociation, not lost penalties.
                let changed = [7, 19, 23, 24, 25, 26, 29, 31].contains(&(word % 32));
                if changed && contacts[word / 32].alive != 0 {
                    let (a, e) = (f32::from_bits(a), f32::from_bits(e));
                    assert!((a-e).abs() <= 1e-5 * e.abs().max(1.0), "{case} half={parity} contact={} word={}: compact final publication {a} differs from canonical {e}", word/32, word%32);
                } else {
                    assert_eq!(a,e,"{case} half={parity} contact={} word={}: cold geometry/feature/generation or dead slot was corrupted", word/32,word%32);
                }
            }
        }
    }
}

#[test]
fn adjacency_order_uses_physical_neighbors_across_append_permutations_and_high_degrees() {
    let mut capacity = WorldCapacity::small();
    capacity.max_colliders = 65538;
    capacity.max_candidates = 65540;
    let mut sim = WgpuSim::new(capacity).expect("GPU adapter required");
    let n_bodies = 2048u32;
    sim.body_count = n_bodies;
    sim.joint_count = 3;
    sim.write_host_uniforms();
    let bodies: Vec<BodyState> = (0..n_bodies)
        .map(|i| BodyState {
            inv_mass: if i == 0 { 1.0 } else { 0.0 },
            flags: if i == 0 { 9 } else { 0 },
            ..BodyState::zeroed()
        })
        .collect();
    sim.queue
        .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
    let colliders: Vec<Collider> = (0..65538u32)
        .map(|i| Collider {
            // Multiple colliders can belong to the same neighbor body. Their IDs
            // must break ties without depending on append IDs or contact addresses.
            body: if i == 0 {
                0
            } else {
                1 + ((i - 1) / 2 * 41) % (n_bodies - 1)
            },
            ..Collider::zeroed()
        })
        .collect();
    sim.queue
        .write_buffer(&sim.bufs[2], 0, bytemuck::cast_slice(&colliders));
    let joints: Vec<Joint> = (0..3)
        .map(|i| Joint {
            body_a: 0,
            body_b: i + 1,
            ..Joint::zeroed()
        })
        .collect();
    sim.queue
        .write_buffer(&sim.bufs[5], 0, bytemuck::cast_slice(&joints));
    sim.queue.write_buffer(&sim.bufs[11], 0, bytes_of(&0u32));
    let sentinel = 0xfeedcafeu32;
    for degree in [0u32, 1, 28, 29, 30, 32, 33, 65, 1025, 4093, 65537] {
        for permutation in 0..3 {
            let mut physical: Vec<u32> = (1..=degree).collect();
            if permutation == 1 {
                physical.reverse();
            }
            if permutation == 2 {
                let mut seed = 0x12345678u32;
                for i in (1..physical.len()).rev() {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    physical.swap(i, seed as usize % (i + 1));
                }
            }
            let manifolds: Vec<Manifold> = physical
                .iter()
                .enumerate()
                .map(|(id, &col)| Manifold {
                    collider_a: if id % 2 == 0 { 0 } else { col },
                    collider_b: if id % 2 == 0 { col } else { 0 },
                    contact_start: id as u32,
                    contact_count: 1,
                    ..Manifold::zeroed()
                })
                .collect();
            if degree > 0 {
                sim.queue
                    .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&manifolds));
            }
            let key = |id: u32| -> [u32; 4] {
                if id < 3 {
                    [0, id, 0, 0]
                } else {
                    let m = manifolds[(id - 3) as usize];
                    let other = m.collider_a.max(m.collider_b);
                    [
                        1,
                        colliders[other as usize].body,
                        m.collider_a.min(m.collider_b),
                        other,
                    ]
                }
            };
            let count = degree + 3;
            let mut ids: Vec<u32> = (0..count).rev().collect();
            ids.rotate_left((permutation as usize * 7) % count as usize);
            let mut upload = vec![sentinel; count as usize + 6];
            upload[3..3 + count as usize].copy_from_slice(&ids);
            sim.queue
                .write_buffer(&sim.bufs[14], 0, bytemuck::cast_slice(&upload));
            sim.queue.write_buffer(
                &sim.bufs[14],
                u64::from(constraint_id_words(&sim.capacity)) * 4,
                bytemuck::cast_slice(&[3u32, 3 + count]),
            );
            sim.queue.write_buffer(
                &sim.bufs[7],
                0,
                bytes_of(&StepCounters {
                    body_count: n_bodies,
                    joint_count: 3,
                    manifold_count: degree,
                    awake_count: 1,
                    ..Default::default()
                }),
            );
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "csr_sort", 1, None);
            }
            sim.queue.submit([enc.finish()]);
            let actual: Vec<u32> = download(&sim, 14, 0, upload.len());
            ids.sort_by_key(|&id| key(id));
            assert_eq!(
                &actual[..3],
                &[sentinel; 3],
                "degree={degree}: sort corrupted the preceding body's adjacency"
            );
            assert_eq!(
                &actual[3 + count as usize..],
                &[sentinel; 3],
                "degree={degree}: sort wrote outside its CSR range"
            );
            let actual_keys: Vec<_> = actual[3..3 + count as usize]
                .iter()
                .map(|&id| key(id))
                .collect();
            let expected_keys: Vec<_> = ids.iter().map(|&id| key(id)).collect();
            for (slot, (got, want)) in actual_keys.iter().zip(&expected_keys).enumerate() {
                assert_eq!(got,want,"degree={degree} permutation={permutation} slot={slot}: adjacency must order joints by identity then contacts by physical neighbor and collider pair. Sorting atomic append IDs changes floating-point summation order between runs.");
            }
        }
    }
}

/// A different dispatch cardinality must still visit each live point exactly once.
/// Compare all storage words, including reserved holes and the unlaunched tail.
#[test]
fn manifold_persistence_matches_point_updates_across_ragged_ranges_and_both_halves() {
    let mut sim = WgpuSim::new_with_compact_contacts(WorldCapacity::small(), true)
        .expect("GPU adapter required");
    for id in 0..4 {
        sim.add_body(BodyState::zeroed());
        sim.add_collider(Collider {
            body: id,
            shape: ShapeType::Box as u32,
            params: [0.5, 0.5, 0.5, 0.866],
            ..Collider::zeroed()
        });
    }
    for sleeping in [false, true] {
        let bodies: Vec<BodyState> = (0..4)
            .map(|id| BodyState {
                position: [
                    0.1 * (id * id) as f32,
                    0.2 * id as f32,
                    -0.1 * (id * id) as f32,
                ],
                orientation: [0.0, 0.0, 0.0, 1.0],
                inv_mass: if id % 2 == 0 { 0.0 } else { 1.0 },
                flags: if id % 2 == 0 {
                    0
                } else if sleeping {
                    5
                } else {
                    9
                },
                ..BodyState::zeroed()
            })
            .collect();
        sim.queue
            .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
        for parity in 0..2 {
            sim.contacts_parity = parity;
            let cur = parity * sim.capacity.max_contacts;
            let prev = (1 - parity) * sim.capacity.max_contacts;
            let dead = Contact {
                feature_id: 0xfedcba98,
                anchor_a: [91., 92., 93.],
                ..Contact::zeroed()
            };
            let mut current = Vec::new();
            let mut previous = Vec::new();
            let mut manifolds = Vec::new();
            for pair in 0..67u32 {
                let count = pair % 9;
                let collider_a = (pair % 2) * 2;
                manifolds.push(Manifold {
                    collider_a,
                    collider_b: collider_a + 1,
                    contact_start: cur + current.len() as u32,
                    contact_count: count,
                    normal: [0., 1., 0.],
                    friction: 0.5,
                });
                for point in 0..count {
                    let index = previous.len() as u32;
                    let old = Contact {
                        collider_a,
                        collider_b: collider_a + 1,
                        feature_id: pair * 8 + point,
                        alive: 1,
                        normal: [0., 1., 0.],
                        tangent1: [1., 0., 0.],
                        tangent2: [0., 0., 1.],
                        anchor_a: [0.01 * point as f32, 0.2, -0.3],
                        anchor_b: [-0.2, 0.3, 0.4],
                        c0_n: -0.01,
                        c0_t1: 0.02,
                        c0_t2: -0.03,
                        lambda_n: -1.0 - index as f32,
                        lambda_t1: 0.2,
                        lambda_t2: -0.3,
                        k_n: 5000.,
                        k_t: 30.,
                        friction: 0.5,
                        _pad: [1., f32::from_bits(0xabcdef12), 50.],
                        ..Contact::zeroed()
                    };
                    previous.push(old);
                    current.push(if point == 3 {
                        dead
                    } else if pair % 2 == 0 {
                        Contact {
                            alive: index + 2,
                            ..Contact::zeroed()
                        }
                    } else {
                        old
                    });
                }
                // Not covered by any manifold; neither scheduler may touch it.
                current.push(dead);
                previous.push(dead);
            }
            let live_range = current.len();
            current.extend([dead; 5]);
            sim.last_counters.contact_count = previous.len() as u32;
            sim.write_host_uniforms();
            sim.queue.write_buffer(
                &sim.bufs[7],
                0,
                bytes_of(&StepCounters {
                    body_count: 4,
                    collider_count: 4,
                    contact_count: live_range as u32,
                    manifold_count: manifolds.len() as u32,
                    ..Default::default()
                }),
            );
            sim.queue
                .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&manifolds));
            sim.queue.write_buffer(
                &sim.bufs[4],
                prev as u64 * 128,
                bytemuck::cast_slice(&previous),
            );
            let mut outputs = Vec::new();
            let mut packed_outputs = Vec::new();
            for (entry, threads) in [
                ("persist_contacts", live_range),
                ("persist_manifolds", manifolds.len()),
                ("persist_manifolds_compact", manifolds.len()),
            ] {
                sim.queue.write_buffer(
                    &sim.bufs[4],
                    cur as u64 * 128,
                    bytemuck::cast_slice(&current),
                );
                sim.queue.write_buffer(
                    &sim.bufs[27],
                    0,
                    bytemuck::cast_slice(&vec![0x12345678u32; current.len() * 16]),
                );
                // Poison the pair cache before each preparation path. Without this,
                // fused persistence could accidentally reuse the previous path's data.
                const PAIR_CANARY: u32 = 0xdeadbeef;
                sim.queue.write_buffer(
                    &sim.bufs[28],
                    0,
                    bytemuck::cast_slice(&vec![PAIR_CANARY; (manifolds.len() + 1) * 8]),
                );
                let mut enc = sim.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    sim.disp(&mut pass, "contact_hash_clear", 4096, None);
                    sim.disp(
                        &mut pass,
                        "contact_hash_insert",
                        previous.len() as u64,
                        None,
                    );
                    sim.disp(&mut pass, "clear_solver_contacts", live_range as u64, None);
                    sim.disp(&mut pass, entry, threads as u64, None);
                    if entry != "persist_manifolds_compact" {
                        sim.disp(
                            &mut pass,
                            "pack_solver_contacts",
                            manifolds.len() as u64,
                            None,
                        );
                    }
                }
                sim.queue.submit([enc.finish()]);
                let pairs = download::<u32>(&sim, 28, 0, (manifolds.len() + 1) * 8);
                for (pair, mf) in manifolds.iter().enumerate() {
                    // This fixture maps the four box colliders directly to bodies;
                    // the mixed primal/dual fixtures test nonidentity/compound maps.
                    let expected = [
                        mf.collider_a,
                        mf.collider_b,
                        mf.contact_start,
                        mf.contact_count,
                        0.0f32.to_bits(),
                        1.0f32.to_bits(),
                        0.0f32.to_bits(),
                        mf.friction.to_bits(),
                    ];
                    assert_eq!(&pairs[pair * 8..pair * 8 + 8], &expected,
                        "{entry}: half={parity}, sleeping={sleeping}, pair={pair}: solver pair cache was stale or packed incorrectly; fused persistence must refresh bodies, point range, shape flags and material each step");
                }
                assert_eq!(
                    &pairs[manifolds.len() * 8..],
                    &[PAIR_CANARY; 8],
                    "{entry}: partial pair dispatch overwrote the cache tail canary"
                );
                packed_outputs.push(download::<u32>(&sim, 27, 0, current.len() * 16));
                outputs.push(download::<u32>(
                    &sim,
                    4,
                    cur as u64 * 128,
                    current.len() * 32,
                ));
            }
            for output in &outputs[1..] {
                for (word, (&actual, &expected)) in output.iter().zip(&outputs[0]).enumerate() {
                    let arithmetic = [7, 11, 19, 23, 24, 25, 26, 27, 28, 31].contains(&(word % 32));
                    if arithmetic && !sleeping && current[word / 32].alive != 0 {
                        // Hoisting pair bodies changes backend FMA association (observed
                        // one-ULP residual difference), but not the update equations.
                        let (a, e) = (f32::from_bits(actual), f32::from_bits(expected));
                        assert!((a-e).abs() <= 1e-6 * e.abs().max(1.0),
                        "half={parity} point={} word={}: manifold warm-start arithmetic {a} differs from point scheduling {e}; check source tags and once-only decay", word/32, word%32);
                    } else {
                        assert_eq!(actual, expected,
                        "sleeping={sleeping} half={parity} point={} word={}: manifold persistence corrupted metadata, inactive point, or hole; check current-offset subtraction and range coverage", word/32, word%32);
                    }
                }
            }
            let input: &[u32] = bytemuck::cast_slice(&current);
            for (word, (&a, &e)) in packed_outputs[2].iter().zip(&packed_outputs[1]).enumerate() {
                let point = word / 16;
                let dead = packed_outputs[1][point * 16 + 15] & 0x7fffffff == 0x7fffffff;
                if word % 16 == 15 || point >= live_range {
                    assert_eq!(a, e, "half={parity} point={point} word={}: fused preparation corrupted pair/stick tag or tail canary", word%16);
                } else if !dead {
                    let (a, e) = (f32::from_bits(a), f32::from_bits(e));
                    assert!((a-e).abs() <= 1e-6 * e.abs().max(1.0),
                        "half={parity} point={point} word={}: fused compact value {a} differs from separate pack {e}; packing must follow warm-start exactly once", word%16);
                }
            }
            for output in &outputs[1..] {
                assert_eq!(&output[live_range * 32..], &input[live_range * 32..],
                    "sleeping={sleeping} half={parity}: partial final workgroup overwrote contact tail canaries");
            }
        }
    }
}

/// Mixed reused, generation-invalidated, cold, and separated pairs cross the
/// 64-thread boundary. Both history halves must produce the same point set as
/// the independent fixture expectations, without losing or duplicating warm starts.
#[test]
fn contact_reuse_preserves_mixed_pairs_history_halves_and_warmstart() {
    mixed_contact_reuse_fixture(false, false);
}

#[test]
fn persistent_contacts_preserve_reuse_features_generation_and_both_history_halves() {
    mixed_contact_reuse_fixture(true, false);
}

#[test]
fn persistent_fused_warmstart_preserves_history_and_refreshes_pair_cache() {
    mixed_contact_reuse_fixture(true, true);
}

#[test]
fn persistent_dual_updates_hot_state_without_corrupting_metadata_or_history() {
    let cap = WorldCapacity::small();
    let mut control =
        WgpuSim::new_with_compact_contacts(cap.clone(), true).expect("GPU adapter required");
    let mut persistent = WgpuSim::new_with_persistent_contacts(cap).expect("GPU adapter required");
    for sim in [&mut control, &mut persistent] {
        for _ in 0..2 {
            sim.add_body(BodyState::zeroed());
        }
        for body in [1, 0, 1] {
            sim.add_collider(Collider {
                body,
                shape: ShapeType::Box as u32,
                ..Collider::zeroed()
            });
        }
    }
    for (case, dynamic_a, sleeping, sphere) in [
        ("static/awake", false, false, false),
        ("both-awake", true, false, false),
        ("rotating-sphere", true, false, true),
        ("both-sleeping", true, true, false),
        ("static/sleeping", false, true, false),
    ] {
        let bodies: Vec<BodyState> = (0..2)
            .map(|id| {
                let dynamic = id != 0 || dynamic_a;
                BodyState {
                    position: [id as f32 * 0.02, 0.01, -0.03],
                    initial_pos: [0.01, -0.02, 0.03],
                    orientation: [0., 0.2f32.sin(), 0., 0.2f32.cos()],
                    initial_rot: [0., 0., 0., 1.],
                    inv_mass: if dynamic { 1. } else { 0. },
                    flags: if !dynamic {
                        0
                    } else if sleeping {
                        5
                    } else {
                        9
                    },
                    ..BodyState::zeroed()
                }
            })
            .collect();
        let colliders: Vec<Collider> = [1, 0, 1]
            .into_iter()
            .map(|body| Collider {
                body,
                shape: if sphere && body == 0 {
                    ShapeType::Sphere as u32
                } else {
                    ShapeType::Box as u32
                },
                ..Collider::zeroed()
            })
            .collect();
        let mut contacts = vec![Contact::zeroed(); 4];
        for i in [0usize, 2] {
            contacts[i] = Contact {
                collider_a: if i == 0 { 1 } else { 2 },
                collider_b: if i == 0 { 2 } else { 1 },
                feature_id: 0xfabc0000 + i as u32,
                alive: 1,
                normal: [0., 1., 0.],
                tangent1: [0., 0., -1.],
                tangent2: [-1., 0., 0.],
                anchor_a: [0.2, -0.3, 0.4],
                anchor_b: [-0.1, 0.3, 0.2],
                c0_n: -0.02,
                c0_t1: 0.03,
                c0_t2: -0.01,
                friction: 0.5,
                lambda_n: -12.,
                lambda_t1: if i == 0 { 0.1 } else { 100. },
                lambda_t2: 0.3,
                k_n: 100.,
                k_t: 10.,
                // A NaN-shaped generation bit pattern must survive as raw u32.
                _pad: [
                    if i == 0 { 1. } else { 0. },
                    f32::from_bits(0xffffffff - i as u32),
                    30.,
                ],
                ..Contact::zeroed()
            };
        }
        for parity in 0..2 {
            let cur = parity * persistent.capacity.max_contacts;
            let prev = (1 - parity) * persistent.capacity.max_contacts;
            let manifolds = [
                Manifold {
                    collider_a: 1,
                    collider_b: 2,
                    contact_start: cur,
                    contact_count: 1,
                    normal: [0., 1., 0.],
                    friction: 0.5,
                },
                Manifold {
                    collider_a: 2,
                    collider_b: 1,
                    contact_start: cur + 2,
                    contact_count: 2,
                    normal: [0., 1., 0.],
                    friction: 0.5,
                },
            ];
            for sim in [&mut control, &mut persistent] {
                sim.contacts_parity = parity;
                sim.write_host_uniforms();
                sim.queue
                    .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
                sim.queue
                    .write_buffer(&sim.bufs[2], 0, bytemuck::cast_slice(&colliders));
                sim.queue
                    .write_buffer(&sim.bufs[25], 0, bytemuck::cast_slice(&manifolds));
                sim.queue.write_buffer(
                    &sim.bufs[7],
                    0,
                    bytes_of(&StepCounters {
                        body_count: 2,
                        collider_count: 3,
                        contact_count: 4,
                        manifold_count: 2,
                        ..Default::default()
                    }),
                );
                write_contact_fixture(sim, cur, &contacts, &[0, 0, 1, 1]);
                write_contact_fixture(sim, prev, &contacts, &[7, 7, 8, 8]);
                let mut enc = sim.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    sim.disp(&mut pass, "contact_hash_clear", 1024, None);
                    sim.disp(&mut pass, "persist_contacts", 4, None);
                    if !sim.persistent_contacts {
                        sim.disp(&mut pass, "clear_solver_contacts", 4, None);
                    }
                    sim.disp(&mut pass, "pack_solver_contacts", 2, None);
                }
                sim.queue.submit([enc.finish()]);
            }
            let history_before = contact_storage_words(&persistent, prev, 4);
            let cold_before: Vec<u32> = download(&persistent, 4, u64::from(cur) * 32, 32);
            let hot_before: Vec<u32> = download(&persistent, 27, u64::from(cur) * 64, 64);
            for point in [0usize, 2] {
                assert_eq!(
                    cold_before[point * 8 + 2],
                    contacts[point].feature_id,
                    "persistence truncated feature bit 31"
                );
                assert_eq!(cold_before[point*8+3],contacts[point]._pad[1].to_bits(),"persistence changed a full-width generation (including NaN-shaped bit patterns)");
            }
            for iteration in 0..4 {
                for sim in [&control, &persistent] {
                    let mut enc = sim.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        sim.disp(
                            &mut pass,
                            "solve_dual_compact",
                            4,
                            Some(u32::from(iteration == 3)),
                        );
                    }
                    sim.queue.submit([enc.finish()]);
                }
                let expected: Vec<u32> = download(&control, 27, 0, 64);
                let actual: Vec<u32> = download(&persistent, 27, u64::from(cur) * 64, 64);
                for point in 0..4 {
                    if contacts[point].alive == 0 {
                        assert_eq!(&actual[point*16..point*16+16],&hot_before[point*16..point*16+16],
                            "{case}, half={parity}, iter={iteration}, point={point}: dead point or reserved hole changed");
                        continue;
                    }
                    for word in 0..15 {
                        let a = f32::from_bits(actual[point * 16 + word]);
                        let e = f32::from_bits(expected[point * 16 + word]);
                        assert!((a-e).abs() <= 1e-5*e.abs().max(1.),
                            "{case}, half={parity}, iter={iteration}, point={point}, hot word={word}: persistent {a} differs from compact control {e}; check current-half indexing, force projection, independent penalties and sphere lever arms");
                    }
                    let tag = actual[point * 16 + 15];
                    assert_eq!(
                        tag & 0x40000000,
                        0,
                        "resolved solver point retained a transient reuse flag"
                    );
                    assert_eq!(
                        (tag & 0x3fffffff) - 1,
                        expected[point * 16 + 15] & 0x7fffffff,
                        "persistent owner changed during dual"
                    );
                    assert_eq!(
                        tag & 0x80000000,
                        expected[point * 16 + 15] & 0x80000000,
                        "persistent stick flag differs from control"
                    );
                }
                assert_eq!(download::<u32>(&persistent,4,u64::from(cur)*32,32),cold_before,
                    "{case}, half={parity}, iter={iteration}: dual overwrote full-width feature/generation or cold geometry metadata");
                assert_eq!(contact_storage_words(&persistent,prev,4),history_before,
                    "{case}, half={parity}, iter={iteration}: current dual corrupted previous history");
            }
        }
    }
}

fn mixed_contact_reuse_fixture(persistent: bool, fused: bool) {
    let mut sim = if persistent {
        WgpuSim::new_with_persistent_contacts(WorldCapacity::small())
    } else {
        WgpuSim::new(WorldCapacity::small())
    }
    .expect("GPU adapter required");
    let count = 67u32;
    for id in 0..=count {
        sim.add_body(BodyState {
            position: if id == 0 {
                [0.0, -1.0, 0.0]
            } else {
                [id as f32 * 2.0, 0.49, 0.0]
            },
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_mass: if id == 0 { 0.0 } else { 1.0 },
            inv_inertia: [1.0; 3],
            flags: if id == 0 { 0 } else { 9 },
            generation: 19,
            ..BodyState::zeroed()
        });
        sim.add_collider(Collider {
            body: id,
            shape: if id % 3 == 1 {
                ShapeType::Sphere as u32
            } else {
                ShapeType::Box as u32
            },
            params: if id == 0 {
                [200.0, 1.0, 5.0, 201.0]
            } else {
                [0.5, 0.5, 0.5, 0.8660254]
            },
            local_rot: [0.0, 0.0, 0.0, 1.0],
            ..Collider::zeroed()
        });
    }
    let pairs: Vec<CandidatePair> = (1..=count)
        .map(|id| CandidatePair {
            collider_a: 0,
            collider_b: id,
            key: id,
            _pad: 0,
        })
        .collect();
    sim.steps_done = 20;
    sim.params.flags |= 1;
    sim.write_host_uniforms();
    let counters = StepCounters {
        body_count: count + 1,
        collider_count: count + 1,
        candidate_count: count,
        ..Default::default()
    };
    sim.queue.write_buffer(&sim.bufs[7], 0, bytes_of(&counters));
    sim.queue
        .write_buffer(&sim.bufs[10], 0, bytemuck::cast_slice(&pairs));
    let mut enc = sim.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        sim.disp(&mut pass, "build_aabbs", (count + 1) as u64, None);
        sim.disp(&mut pass, "contact_hash_clear", 1024, None);
        sim.disp(&mut pass, "narrowphase", count as u64, None);
    }
    sim.queue.submit([enc.finish()]);
    let seed = download::<StepCounters>(&sim, 7, 0, 1)[0];
    assert_eq!(
        seed.manifold_count, count,
        "fixture requires one colliding pair for every candidate"
    );
    assert_eq!(
        seed.overflow_flags, 0,
        "fixture overflow cannot validate collision equivalence"
    );
    let mut previous: Vec<Contact> = read_contact_fixture(&sim, 0, seed.contact_count as usize);
    // No previous history for every fifth pair; these must run fresh SAT.
    previous.retain(|c| c.collider_b % 5 != 0);
    for (i, c) in previous.iter_mut().enumerate() {
        c.feature_id |= 0x8000_0000; // Feature bit 31 must not become the stick bit.
        c._pad[0] = if i % 2 == 0 { 1.0 } else { 0.0 };
        c.lambda_n = -1.0 - i as f32 * 0.1;
        c.k_n = 5000.0;
        c.k_t = 50.0 + i as f32;
        c._pad[2] = 150.0 + i as f32;
    }
    let mut bodies: Vec<BodyState> = download(&sim, 0, 0, (count + 1) as usize);
    for (id, b) in bodies.iter_mut().enumerate().skip(1) {
        if id % 4 == 0 {
            b.generation = 21;
            b.position[1] += 0.005;
        }
        if id == count as usize {
            b.generation = 21;
            b.position[1] = 4.0;
        }
    }
    sim.queue
        .write_buffer(&sim.bufs[0], 0, bytemuck::cast_slice(&bodies));
    sim.steps_done = 22;
    sim.last_counters.contact_count = previous.len() as u32;
    let hash_n = (previous.len().max(256) * 4).next_power_of_two() as u64;
    for parity in 0..2 {
        for reuse in [false, true] {
            let context =
                format!("persistent={persistent}, fused={fused}, parity={parity}, reuse={reuse}");
            sim.contacts_parity = parity;
            sim.params.flags = (sim.params.flags & !1) | u32::from(reuse);
            sim.write_host_uniforms();
            let cur = parity * sim.capacity.max_contacts;
            let prev = (1 - parity) * sim.capacity.max_contacts;
            write_contact_fixture(&sim, prev, &previous, &vec![0; previous.len()]);
            let history_before = contact_storage_words(&sim, prev, previous.len());
            const PAIR_CANARY: u32 = 0xdeadbeef;
            if fused {
                // A stale pair cache can otherwise make a missing fused prepare
                // look correct on the second half/reuse pass of this fixture.
                sim.queue.write_buffer(
                    &sim.bufs[28],
                    0,
                    bytemuck::cast_slice(&vec![PAIR_CANARY; (count as usize + 1) * 8]),
                );
            }
            sim.queue.write_buffer(&sim.bufs[7], 0, bytes_of(&counters));
            sim.queue
                .write_buffer(&sim.bufs[10], 0, bytemuck::cast_slice(&pairs));
            // The first inactive candidate is deliberately invalid: padded
            // invocations must use the GPU count, never consume this record.
            let poison = CandidatePair {
                collider_a: u32::MAX - 1,
                collider_b: u32::MAX - 1,
                key: 123,
                _pad: 456,
            };
            sim.queue
                .write_buffer(&sim.bufs[10], count as u64 * 16, bytes_of(&poison));
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                sim.disp(&mut pass, "build_aabbs", (count + 1) as u64, None);
                sim.disp(&mut pass, "contact_hash_clear", hash_n, None);
                sim.disp(
                    &mut pass,
                    "contact_hash_insert",
                    previous.len() as u64,
                    None,
                );
                sim.disp(&mut pass, "narrowphase", count as u64, None);
                sim.disp(&mut pass, "write_work_args", 64, None);
                sim.disp_work(
                    &mut pass,
                    if fused {
                        "persist_manifolds_compact"
                    } else {
                        "persist_contacts"
                    },
                    if fused { 66 } else { 65 },
                    None,
                );
            }
            sim.queue.submit([enc.finish()]);
            let c = download::<StepCounters>(&sim, 7, 0, 1)[0];
            assert_eq!(
                c.overflow_flags, 0,
                "{context}: fixture must complete without overflow"
            );
            assert_eq!(
                c.manifold_count,
                count - 1,
                "{context}: separated final pair must disappear, every other pair must appear once"
            );
            let actual: Vec<Contact> = read_contact_fixture(&sim, cur, c.contact_count as usize);
            if persistent {
                let manifolds: Vec<Manifold> = download(&sim, 25, 0, c.manifold_count as usize);
                let hot: Vec<u32> =
                    download(&sim, 27, cur as u64 * 64, c.contact_count as usize * 16);
                let cached: Vec<u32> = if fused {
                    download(&sim, 28, 0, (c.manifold_count as usize + 1) * 8)
                } else {
                    Vec::new()
                };
                for (owner, mf) in manifolds.iter().enumerate() {
                    for absolute in mf.contact_start..mf.contact_start + mf.contact_count {
                        let local = (absolute - cur) as usize;
                        let tag = hot[local * 16 + 15];
                        assert_eq!(tag & 0x7fff_ffff, owner as u32 + 1,
                            "{context}, manifold={owner}, point={local}: persistence lost the current owner or left the transient reuse bit set; the next dual would read the wrong pair");
                    }
                    if fused {
                        // Fixture colliders map to like-numbered bodies; every
                        // third dynamic is a sphere, so preserve its solver flag.
                        let sphere = mf.collider_b % 3 == 1;
                        let expected = [
                            mf.collider_a,
                            mf.collider_b,
                            mf.contact_start,
                            mf.contact_count | if sphere { 0x8000_0000 } else { 0 },
                            mf.normal[0].to_bits(),
                            mf.normal[1].to_bits(),
                            mf.normal[2].to_bits(),
                            mf.friction.to_bits(),
                        ];
                        assert_eq!(&cached[owner * 8..owner * 8 + 8], &expected,
                            "{context}, manifold={owner}: fused persistent warm start did not refresh body IDs, current-half range, sphere flag or material");
                    }
                }
                if fused {
                    assert_eq!(
                        &cached[c.manifold_count as usize * 8..],
                        &[PAIR_CANARY; 8],
                        "{context}: padded manifold invocations overwrote the pair-cache tail"
                    );
                }
            }
            assert!(
                actual.iter().all(|c| c.alive == 1),
                "{context}: transient reuse tags escaped persistence"
            );
            for id in 1..count {
                let points: Vec<_> = actual
                    .iter()
                    .filter(|c| c.collider_a == 0 && c.collider_b == id)
                    .collect();
                let expected_count = if id % 3 == 1 { 1 } else { 4 };
                assert_eq!(points.len(), expected_count,
                    "{context}, pair {id}: expected one sphere point or four distinct rectangular support corners");
                let reused = reuse && id % 5 != 0 && id % 4 != 0;
                for point in points {
                    if id % 5 == 0 {
                        // No prior manifold exists for this pair. Geometry alone
                        // is insufficient: cold support needs the mass/dt² floor.
                        let expected_kn = 1.0 / (sim.params.dt * sim.params.dt);
                        assert_eq!([point.lambda_n, point.lambda_t1, point.lambda_t2], [0.0; 3],
                            "{context}, cold pair {id}: forces must start at zero; unrelated history leaked into a new contact");
                        assert!((point.k_n - expected_kn).abs() <= expected_kn * 1e-6,
                            "{context}, cold pair {id}: normal penalty={} expected mass/dt²={expected_kn}; contact initialization was omitted or applied with the wrong body mass", point.k_n);
                        assert_eq!([point.k_t, point._pad[2]], [1.0; 2],
                            "{context}, cold pair {id}: tangent penalties must start independently at one, without inheriting normal stiffness");
                    }
                    assert_eq!(point._pad[1].to_bits(), if reused { 20 } else { 22 },
                        "{context}, pair {id}: stale-generation and cold pairs must regenerate; eligible pairs must retain their generation");
                    assert!(
                        point
                            .normal
                            .iter()
                            .chain(point.anchor_a.iter())
                            .chain(point.anchor_b.iter())
                            .all(|x| x.is_finite()),
                        "{context}, pair {id}: invalid contact geometry"
                    );
                    if reused {
                        let src = previous
                            .iter()
                            .find(|p| {
                                p.collider_b == id
                                    && p.feature_id == point.feature_id
                                    && p.anchor_b == point.anchor_b
                            })
                            .expect(
                                "reused contact must retain a specific previous support corner",
                            );
                        assert_eq!(
                            point.anchor_a, src.anchor_a,
                            "{context}, pair {id}: reused A anchor changed"
                        );
                        let decay = sim.params.alpha * sim.params.gamma;
                        assert!(
                            (point.lambda_n - src.lambda_n * decay).abs() < 1e-4,
                            "{context}, pair {id}: warm-start normal force lost or decayed twice"
                        );
                        assert!((point.k_t - src.k_t * sim.params.gamma).abs() < 1e-4 &&
                            (point._pad[2] - src._pad[2] * sim.params.gamma).abs() < 1e-4,
                            "{context}, pair {id}: tangent penalties must decay independently exactly once");
                    }
                }
            }
            let expected_total: usize = (1..count).map(|id| if id % 3 == 1 { 1 } else { 4 }).sum();
            assert_eq!(
                actual.len(),
                expected_total,
                "{context}: unexpected pair or padded invocation emitted contacts"
            );
            assert_eq!(
                contact_storage_words(&sim, prev, previous.len()),
                history_before,
                "{context}: current writes corrupted the previous contact half"
            );
        }
    }
}

/// Independent scan oracle at ring/gables scale, with ragged lane ranges, empty
/// colors, poisoned outputs, zero-body reset and a protected allocation tail.
#[test]
fn color_tile_prefix_matches_cpu_for_ragged_ranges_at_full_scene_scale() {
    let cap = WorldCapacity {
        max_bodies: 524288,
        max_colors: 9,
        ..WorldCapacity::small()
    };
    let mut sim = WgpuSim::new(cap.clone()).expect("GPU adapter required");
    let words = (cap.max_colors * cap.max_bodies.div_ceil(64)) as usize;
    let sentinel = 0xdeadbeefu32;
    for groups in [1u32, 63, 64, 65, 127, 129, 1726, 7906, 8191, 0] {
        sim.body_count = if groups == 0 { 0 } else { groups * 64 - 3 };
        sim.write_host_uniforms();
        sim.queue.write_buffer(
            &sim.bufs[7],
            0,
            bytes_of(&StepCounters {
                body_count: sim.body_count,
                ..Default::default()
            }),
        );
        let mut input = vec![sentinel; words];
        let mut expected = input.clone();
        let mut totals = vec![0u32; cap.max_colors as usize];
        for color in 0..cap.max_colors {
            for tile in 0..groups {
                let index = (color * groups + tile) as usize;
                let count = if color >= 7 {
                    0
                } else {
                    (tile * 17 + color * 11) % 65
                };
                input[index] = count;
                expected[index] = totals[color as usize];
                totals[color as usize] += count;
            }
        }
        for kernel in ["prefix_color_tiles", "prefix_color_tiles_parallel"] {
            sim.queue
                .write_buffer(&sim.bufs[16], 0, bytemuck::cast_slice(&input));
            sim.queue.write_buffer(
                &sim.bufs[17],
                0,
                bytemuck::cast_slice(&vec![sentinel; cap.max_colors as usize]),
            );
            let mut enc = sim.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                // Extra workgroups must return uniformly without touching outputs.
                let threads = if kernel.ends_with("_parallel") {
                    (u64::from(cap.max_colors) + 2) * 64
                } else {
                    64
                };
                sim.disp(&mut pass, kernel, threads, None);
            }
            sim.queue.submit([enc.finish()]);
            let actual: Vec<u32> = download(&sim, 16, 0, words);
            let counts: Vec<u32> = download(&sim, 17, 0, cap.max_colors as usize);
            for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(got, want,
                    "{kernel}: groups={groups}, word={index}, color={}, tile={}: exclusive prefix or tail canary differs; check ragged lane ranges, block-sum synchronization and in-place read/write ownership",
                    index / groups.max(1) as usize, index % groups.max(1) as usize);
            }
            assert_eq!(counts, totals,
                "{kernel}: groups={groups}: published color totals differ; empty colors and zero-body reset must overwrite stale counts");
        }
    }
}
