//! The GPU-resident fracture's split (`fracture.slang`) against the host reference
//! (`fracture::reference_split`) on a synthetic engine: a grid of unit cubes, one body,
//! bonds broken at random over several rounds (pieces of pieces). The topology and the
//! slots must match exactly; mass, centre of mass and the colliders' local poses to f32.

use stress_gpu::fracture::{reference_split, FractureTables, GpuFracture, INFO_SUPPORT};
use stress_gpu::gpu::Gpu;
use stress_gpu::ingest::{ColliderInfo, EngineBuffers, EngineIngest, KIND_BODY};

const BODY_BYTES: u64 = 112;

struct Grid {
    t: FractureTables,
}

/// `dims` unit cubes (mass 1), bonded face to face (capacity 1 N), one body at slot 0
/// with every other slot free; the bottom layer supports when `anchored`.
fn grid(dims: [u32; 3], anchored: bool) -> Grid {
    let [nx, ny, nz] = dims;
    let n = (nx * ny * nz) as usize;
    let idx = |x: u32, y: u32, z: u32| (x + nx * (y + ny * z)) as u32;
    let centre = [(nx as f32 - 1.0) / 2.0, (ny as f32 - 1.0) / 2.0, (nz as f32 - 1.0) / 2.0];
    let mut t = FractureTables { max_bodies: n as u32 + 8, pool_cap: 64, impact_peak: true, band: 0.1, dt: 1.0 / 60.0, ..Default::default() };
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let c = [x as f32 - centre[0], y as f32 - centre[1], z as f32 - centre[2]];
                let i = 1.0 / 6.0;
                let k = t.chunks.len() as u32;
                t.chunks.push([[c[0], c[1], c[2], 1.0], [i, i, i, 0.0], [0.0, 0.0, f32::from_bits(k), f32::from_bits(1)], [3e10, 0.2, 1.0, 0.5]]);
                t.info.push(if anchored && z == 0 { INFO_SUPPORT } else { 0 });
                t.parts.push([[c[0], c[1], c[2], f32::from_bits(k)], [0.0, 0.0, 0.0, 1.0]]);
                t.chunk_body.push(0);
                for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                    if x + dx < nx && y + dy < ny && z + dz < nz {
                        t.bonds.push([k, idx(x + dx, y + dy, z + dz), 1f32.to_bits(), 0]);
                    }
                }
            }
        }
    }
    // Slot 0: the whole grid, its centre of mass at the structure's origin, principal
    // axes the structure's (a box's inertia is diagonal).
    t.frames = vec![[[0.0; 4], [0.0, 0.0, 0.0, 1.0]]; n];
    t.free = vec![1; n];
    t.free[0] = 0;
    Grid { t }
}

struct Engine {
    bodies: wgpu::Buffer,
    colliders: wgpu::Buffer,
    contacts: wgpu::Buffer,
    counters: wgpu::Buffer,
}

impl Engine {
    fn buffers(&self) -> EngineBuffers<'_> {
        EngineBuffers { bodies: &self.bodies, colliders: &self.colliders, contacts: &self.contacts, counters: &self.counters }
    }
}

/// The structure's body at slot 0 in both pose regions, moving (v, w in rubble's frame);
/// one box collider per chunk on it.
fn engine(gpu: &Gpu, g: &Grid, v: [f32; 3], w: [f32; 3]) -> Engine {
    let n = g.t.chunks.len();
    let max = g.t.max_bodies as usize;
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let mut words = vec![[0f32; 4]; 7 * 2 * max];
    let anchored = g.t.info.iter().any(|&f| f & INFO_SUPPORT != 0);
    let body = [
        [0.0, 0.0, 0.0, if anchored { 0.0 } else { 1.0 / n as f32 }],
        // rot_to_rubble(identity): -90 degrees about x.
        [-h, 0.0, 0.0, h],
        [1.0, 1.0, 1.0, f32::from_bits(if anchored { 0 } else { 9 })],
        [v[0], v[1], v[2], 0.0],
        [w[0], w[1], w[2], 0.0],
        [0.0; 4],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for region in 0..2 {
        for k in 0..7 {
            words[7 * (region * max) + k] = body[k];
        }
    }
    let bodies = gpu.storage("bodies", &words);
    let mut cols = vec![[0f32; 4]; 4 * n];
    for (k, c) in g.t.chunks.iter().enumerate() {
        cols[4 * k] = [f32::from_bits(0), f32::from_bits(2), f32::from_bits(0), f32::from_bits(0)];
        cols[4 * k + 1] = [c[0][0], c[0][1], c[0][2], 1.0];
        cols[4 * k + 2] = [0.0, 0.0, 0.0, 1.0];
        cols[4 * k + 3] = [0.5, 0.5, 0.5, 0.0];
    }
    let colliders = gpu.storage("colliders", &cols);
    let _ = BODY_BYTES;
    Engine { bodies, colliders, contacts: gpu.storage("contacts", &[0u32; 64]), counters: gpu.storage("counters", &[0u32; 64]) }
}

fn ingest(gpu: &Gpu, g: &Grid) -> EngineIngest {
    let table: Vec<ColliderInfo> = (0..g.t.chunks.len() as u32).map(|k| ColliderInfo { a: [k, 0, KIND_BODY, 1], b: [1.0, 3e10, 0.5, 0.0] }).collect();
    EngineIngest::new(gpu, 64, 1, g.t.max_bodies, g.t.chunk_count(), &table)
}

/// A deterministic shuffle (splitmix64).
fn rng(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn check(dims: [u32; 3], anchored: bool, rounds: usize, per_round: f64) {
    let gpu = Gpu::new().expect("GPU");
    let g = grid(dims, anchored);
    // A static body does not move: pieces off it start at rest.
    let (v, w) = if anchored { ([0.0; 3], [0.0; 3]) } else { ([1.0, 0.0, 0.5], [0.0, 0.4, 0.0]) };
    let e = engine(&gpu, &g, v, w);
    let ing = ingest(&gpu, &g);
    let mut f = GpuFracture::new(&gpu, &g.t, &e.buffers(), &ing).expect("fracture");
    let mut intact = vec![true; g.t.bonds.len()];
    let mut chunk_body = g.t.chunk_body.clone();
    let mut free = g.t.free.clone();
    let mut seed = 7u64 + dims[0] as u64;
    for round in 0..rounds {
        let broken: Vec<u32> = (0..g.t.bonds.len() as u32).filter(|&i| intact[i as usize] && (rng(&mut seed) as f64 / u64::MAX as f64) < per_round).collect();
        f.break_bonds(&gpu, &broken, &intact, &chunk_body, &g.t.bonds);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        f.encode_split(&gpu, &mut enc, 100 + round as u32);
        gpu.queue.submit([enc.finish()]);
        let pieces = reference_split(&g.t, &mut intact, &broken, &mut chunk_body, &mut free);
        let got = f.chunk_bodies(&gpu);
        assert_eq!(got, chunk_body, "{dims:?} round {round}: every chunk's body");
        let c = f.counters(&gpu);
        assert_eq!(c.error, 0, "{dims:?} round {round}: errors {:#x}", c.error);
        // Every piece's mass and centre of mass (structure frame).
        let frames = f.body_frames(&gpu);
        let words: Vec<[f32; 4]> = gpu.read(&e.bodies);
        for (slot, m, com, _) in &pieces {
            let fr = frames[*slot as usize][0];
            for a in 0..3 {
                assert!((fr[a] as f64 - com[a]).abs() < 1e-4 * (1.0 + com[a].abs()), "{dims:?} round {round} slot {slot}: com {:?} vs {com:?}", fr);
            }
            let inv = words[7 * *slot as usize][3];
            let anchored_piece = inv == 0.0;
            if !anchored_piece {
                assert!(((1.0 / inv) as f64 - m).abs() < 1e-4 * m, "{dims:?} round {round} slot {slot}: mass {} vs {m}", 1.0 / inv);
                // The parent's rigid motion: v + w x (p - p_parent), the parent at the
                // origin in the rubble frame (rubble's x, z, -y of the structure frame).
                let p = words[7 * *slot as usize];
                let vel = words[7 * *slot as usize + 3];
                let expect = [v[0] + w[1] * p[2] - w[2] * p[1], v[1] + w[2] * p[0] - w[0] * p[2], v[2] + w[0] * p[1] - w[1] * p[0]];
                for a in 0..3 {
                    assert!((vel[a] - expect[a]).abs() < 1e-3, "{dims:?} round {round} slot {slot}: v {:?} vs {expect:?}", vel);
                }
                // Both pose regions written alike.
                let prev = words[7 * (g.t.max_bodies as usize + *slot as usize)];
                assert_eq!(prev, p, "{dims:?} slot {slot}: previous-pose region");
            }
        }
        // Each collider on its chunk's body, at the chunk's centre less the body's centre
        // of mass, in the body's principal frame.
        let cols: Vec<[f32; 4]> = gpu.read(&e.colliders);
        for k in 0..g.t.chunks.len() {
            assert_eq!(cols[4 * k][0].to_bits(), chunk_body[k], "collider {k}'s body");
            let fr = frames[chunk_body[k] as usize];
            let (q, com) = (fr[1], fr[0]);
            let d = [g.t.chunks[k][0][0] - com[0], g.t.chunks[k][0][1] - com[1], g.t.chunks[k][0][2] - com[2]];
            // conj(q) d
            let (qx, qy, qz, qw) = (-q[0], -q[1], -q[2], q[3]);
            let t = [2.0 * (qy * d[2] - qz * d[1]), 2.0 * (qz * d[0] - qx * d[2]), 2.0 * (qx * d[1] - qy * d[0])];
            let r = [d[0] + qw * t[0] + (qy * t[2] - qz * t[1]), d[1] + qw * t[1] + (qz * t[0] - qx * t[2]), d[2] + qw * t[2] + (qx * t[1] - qy * t[0])];
            for a in 0..3 {
                assert!((cols[4 * k + 1][a] - r[a]).abs() < 1e-3, "collider {k}: local {:?} vs {r:?}", cols[4 * k + 1]);
            }
        }
    }
}

#[test]
fn free_grid_splits_like_the_reference() {
    check([6, 5, 4], false, 4, 0.25);
}

#[test]
fn anchored_grid_splits_like_the_reference() {
    check([5, 5, 6], true, 4, 0.3);
}

#[test]
fn large_grid_shatters_like_the_reference() {
    check([40, 30, 20], false, 3, 0.4);
}

/// Wall time of the split on the GPU (submit to done), the median of `reps` runs after a
/// warm-up: `cargo test --release -p stress-gpu --test fracture bench -- --ignored --nocapture`.
fn time_split(dims: [u32; 3], fraction: f64, reps: usize) -> (f64, usize) {
    let gpu = Gpu::new().expect("GPU");
    let g = grid(dims, false);
    let all = vec![true; g.t.bonds.len()];
    let mut ms = Vec::new();
    let mut pieces = 0;
    for _ in 0..reps {
        let e = engine(&gpu, &g, [1.0, 0.0, 0.0], [0.0, 0.3, 0.0]);
        let ing = ingest(&gpu, &g);
        let mut f = GpuFracture::new(&gpu, &g.t, &e.buffers(), &ing).expect("fracture");
        // A quiet frame first: every buffer touched once.
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        f.encode_split(&gpu, &mut enc, 0);
        gpu.queue.submit([enc.finish()]);
        let mut seed = 99u64;
        let broken: Vec<u32> = if fraction <= 0.0 {
            Vec::new()
        } else if fraction < 1e-6 {
            (0..g.t.bonds.len() as u32).filter(|&i| g.t.bonds[i as usize][0] == 0).collect()
        } else {
            (0..g.t.bonds.len() as u32).filter(|_| (rng(&mut seed) as f64 / u64::MAX as f64) < fraction).collect()
        };
        f.break_bonds(&gpu, &broken, &all, &g.t.chunk_body, &g.t.bonds);
        // The uploads done before the clock starts.
        gpu.queue.submit([]);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let t = std::time::Instant::now();
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        f.encode_split(&gpu, &mut enc, 1);
        gpu.queue.submit([enc.finish()]);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        let c = f.counters(&gpu);
        pieces = c.pieces as usize;
        assert_eq!(c.error, 0);
    }
    ms.sort_by(f64::total_cmp);
    (ms[ms.len() / 2], pieces)
}

#[test]
#[ignore]
fn bench_split() {
    for (name, dims, fraction) in [
        ("tower-like 16x16x15 (3840 chunks), one chip", [16, 16, 15], 1e-9),
        ("tower-like 16x16x15, 5% bonds", [16, 16, 15], 0.05),
        ("100k chunks (100x50x20), quiet frame", [100, 50, 20], 0.0),
        ("100k chunks, one chip", [100, 50, 20], 1e-9),
        ("100k chunks, 5% of bonds in one frame", [100, 50, 20], 0.05),
        ("100k chunks, 40% of bonds in one frame", [100, 50, 20], 0.4),
        ("100k chunks, every bond in one frame", [100, 50, 20], 1.0),
    ] {
        let (ms, pieces) = time_split(dims, fraction, 5);
        println!("{name}: split {ms:.3} ms, {pieces} new bodies");
    }
}

#[test]
fn one_chip_off_a_large_body() {
    let gpu = Gpu::new().expect("GPU");
    let g = grid([16, 16, 15], false);
    let e = engine(&gpu, &g, [0.0; 3], [0.0; 3]);
    let ing = ingest(&gpu, &g);
    let mut f = GpuFracture::new(&gpu, &g.t, &e.buffers(), &ing).expect("fracture");
    // The corner chunk's three bonds: it breaks off alone.
    let corner: Vec<u32> = (0..g.t.bonds.len() as u32).filter(|&i| g.t.bonds[i as usize][0] == 0).collect();
    f.break_bonds(&gpu, &corner, &vec![true; g.t.bonds.len()], &g.t.chunk_body, &g.t.bonds);
    let t = std::time::Instant::now();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    f.encode_split(&gpu, &mut enc, 1);
    gpu.queue.submit([enc.finish()]);
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    eprintln!("one chip: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    let bodies = f.chunk_bodies(&gpu);
    assert_eq!(bodies[0], 1, "the chip takes the first free slot");
    assert!(bodies[1..].iter().all(|&b| b == 0), "the rest keeps the body");
    assert_eq!(f.counters(&gpu), stress_gpu::fracture::Counters { broken: 0, pieces: 1, error: 0, split_frames: 1, ..Default::default() });
}
