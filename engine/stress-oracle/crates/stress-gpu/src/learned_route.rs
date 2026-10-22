//! The learned impact path's routing on the GPU (`shaders/learned_route.slang`): stress-ref's
//! `World::select_islands` from the same state, so a learned call holds the bodies the CPU
//! routing would put in islands (P1's recorder takes them from `predict_island_members`).
//! `STRESS_LEARNED_ROUTE` picks whose routing: `world` (the default) the coupled world's own,
//! the deployed boundary the model is trained on (tiered under the quasi-static solve: only
//! an impactor's fast contact seeds; P0's closed-loop ceiling's); `recorder` a standalone
//! world's, as P1's recorder first took it (untiered: any contact whose rigid assumption
//! fails seeds an island, fragments meeting fragments included; the routing check's flag does
//! not gate it). The other switches are the
//! world's (`tiered_fast_joins`, `debris_reach_routing`, the ground), read once. Capacities are
//! explicit: joining pairs past `STRESS_LEARNED_MAX_ROUTE_PAIRS` (default 16 per island
//! slot) or labels not converged in the rounds fail the frame with an error.

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

const ROUNDS: u32 = 48;
const PART_F4: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RouteParams {
    island_cap: u32,
    impactor_count: u32,
    engine_base: u32,
    n_pad: u32,
    dt: f32,
    speed: f32,
    tiered: u32,
    fast_joins: u32,
    debris: u32,
    ground: u32,
    ground_height: f32,
    ground_modulus: f32,
    edge_cap: u32,
    always: u32,
    pad0: u32,
    pad1: u32,
}

const BINDINGS: [Binding; 21] = [
    Binding::Uniform,   // 0 params
    Binding::Storage,   // 1 islands
    Binding::Storage,   // 2 chunk statics
    Binding::Storage,   // 3 chunk state
    Binding::Storage,   // 4 ctab
    Binding::Storage,   // 5 engine bodies
    Binding::Storage,   // 6 bond statics
    Binding::Uniform,   // 7 materials
    Binding::Storage,   // 8 chunk rows and their wave speeds
    Binding::Storage,   // 9 impactors
    Binding::Storage,   // 10 counters
    Binding::Storage,   // 11 routing check flag
    Binding::StorageRw, // 12 parts
    Binding::StorageRw, // 13 sorted
    Binding::StorageRw, // 14 edges
    Binding::StorageRw, // 15 labels (atomic)
    Binding::StorageRw, // 16 control (atomic)
    Binding::StorageRw, // 17 seeds
    Binding::StorageRw, // 18 learned islands (`isl`)
    Binding::Storage,   // 19 this dispatch's pass
    Binding::StorageRw, // 20 learned control words
];

/// The routing switches, as the world has them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteConfig {
    pub tiered: bool,
    pub fast_joins: bool,
    pub debris: bool,
    /// The ground: height and plane-strain modulus (scaled).
    pub ground: Option<(f32, f32)>,
}

impl RouteConfig {
    /// `world`'s switches (`select_islands`' reads of them), tiered as `STRESS_LEARNED_ROUTE`
    /// says (`world`, the default: the world's own; `recorder`: untiered).
    pub fn of(world: &stress_ref::world::World) -> Result<RouteConfig, String> {
        let tiered = match std::env::var("STRESS_LEARNED_ROUTE").ok().as_deref() {
            Some("recorder") => false,
            None | Some("world") => world.qs_islands && world.scene.sim.solve_mode == stress_ref::scene::SolveMode::QuasiStatic,
            Some(o) => return Err(format!("STRESS_LEARNED_ROUTE={o}: world (default) or recorder")),
        };
        let ground = world.scene.ground.as_ref().map(|g| (g.height as f32, crate::contacts::modulus(&world.scene, &g.material) as f32));
        Ok(RouteConfig { tiered, fast_joins: world.tiered_fast_joins, debris: world.debris_reach_routing, ground })
    }
}

struct Kernels {
    islands: Kernel,
    impactors: Kernel,
    debris: Kernel,
    keys: Kernel,
    bitonic: Kernel,
    sweep: Kernel,
    hook: Kernel,
    jump: Kernel,
    roots: Kernel,
    finish: Kernel,
    mark: Kernel,
}

pub struct Route {
    k: Kernels,
    pub cfg: RouteConfig,
    rtab: wgpu::Buffer,
    parts: wgpu::Buffer,
    sorted: wgpu::Buffer,
    edges: wgpu::Buffer,
    lab: wgpu::Buffer,
    actl: wgpu::Buffer,
    seed: wgpu::Buffer,
    /// Per sort step (k, j), then per label round (-, -, round): one small buffer each.
    sort_passes: Vec<wgpu::Buffer>,
    rounds: Vec<wgpu::Buffer>,
    n_pad: u32,
    edge_cap: u32,
    edges_per_slot: u32,
}

/// The tables the routing reads beyond the learned path's own.
pub struct RouteTables<'a> {
    pub islands: &'a wgpu::Buffer,
    pub chunks: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub ctab: &'a wgpu::Buffer,
    pub bodies: &'a wgpu::Buffer,
    pub bond_static: &'a wgpu::Buffer,
    pub materials: &'a wgpu::Buffer,
    pub counters: &'a wgpu::Buffer,
    pub route: &'a wgpu::Buffer,
    pub imps: &'a wgpu::Buffer,
    pub isl: &'a wgpu::Buffer,
    pub ctl: &'a wgpu::Buffer,
}

fn rw(gpu: &Gpu, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

impl Route {
    /// The routing for `world`'s chunks (wave speed per stable chunk row) and switches.
    pub fn new(gpu: &Gpu, world: &stress_ref::world::World) -> Route {
        let k = |e: &str| gpu.compute(&shaders::LEARNED_ROUTE, e, &BINDINGS);
        let kernels = Kernels {
            islands: k("lr_islands"),
            impactors: k("lr_impactors"),
            debris: k("lr_debris"),
            keys: k("lr_keys"),
            bitonic: k("lr_bitonic"),
            sweep: k("lr_sweep"),
            hook: k("lr_hook"),
            jump: k("lr_jump"),
            roots: k("lr_roots"),
            finish: k("lr_finish"),
            mark: k("lr_mark"),
        };
        // rows: chunk0 per structure (learned_feat's `sinc` layout: [structures, rows, chunk0..])
        let scale = world.scene.sim.stiffness_scale;
        let mut rsinc = vec![world.solver.structures.len() as u32, 0];
        let mut rspeed = Vec::new();
        for st in &world.solver.structures {
            rsinc.push(rspeed.len() as u32);
            for ch in &st.chunks {
                let m = world.scene.material(&ch.material);
                rspeed.push((m.youngs_modulus * scale / m.density).sqrt() as f32);
            }
        }
        rsinc[1] = rspeed.len() as u32;
        let mut rtab = rsinc;
        rtab.extend(rspeed.iter().map(|s| s.to_bits()));
        let edges_per_slot = std::env::var("STRESS_LEARNED_MAX_ROUTE_PAIRS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        Route {
            k: kernels,
            cfg: RouteConfig { tiered: false, fast_joins: false, debris: false, ground: None },
            rtab: gpu.storage("learned route chunk rows", &rtab),
            parts: rw(gpu, "learned route parts", 16),
            sorted: rw(gpu, "learned route sorted", 16),
            edges: rw(gpu, "learned route edges", 16),
            lab: rw(gpu, "learned route labels", 16),
            actl: rw(gpu, "learned route control", 4 * (8 + ROUNDS as u64)),
            seed: rw(gpu, "learned route seeds", 16),
            sort_passes: Vec::new(),
            rounds: (0..ROUNDS).map(|r| gpu.storage("learned route round", &[[0u32, 0, r, 0]])).collect(),
            n_pad: 0,
            edge_cap: 0,
            edges_per_slot,
        }
    }

    /// The last routing's parts (diagnostics): per part PART_F4 float4 words, `n` parts.
    pub fn parts(&self, gpu: &Gpu, n: usize) -> Vec<[f32; 4]> {
        gpu.read_range(&self.parts, 0, PART_F4 as usize * n)
    }

    /// The last routing's seeds (bit 1 seeded, bit 2 its component's root seeded), labels and
    /// control words (diagnostics).
    pub fn seeds(&self, gpu: &Gpu, n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        (gpu.read_range(&self.seed, 0, n), gpu.read_range(&self.lab, 0, n), gpu.read_range(&self.actl, 0, 8 + ROUNDS as usize))
    }

    /// Batches run every frame (untiered: a contact the routing check does not see may seed).
    pub fn always(&self) -> bool {
        !self.cfg.tiered
    }

    /// Room for `island_cap` islands and the impactors.
    pub fn ensure(&mut self, gpu: &Gpu, island_cap: u32) {
        let n = (island_cap + 32).next_power_of_two().max(64);
        if n <= self.n_pad {
            return;
        }
        self.n_pad = n;
        self.edge_cap = self.edges_per_slot * n;
        self.parts = rw(gpu, "learned route parts", 16 * PART_F4 * n as u64);
        self.sorted = rw(gpu, "learned route sorted", 8 * n as u64);
        self.edges = rw(gpu, "learned route edges", 8 * self.edge_cap as u64);
        self.lab = rw(gpu, "learned route labels", 4 * n as u64);
        self.seed = rw(gpu, "learned route seeds", 4 * n as u64);
        self.sort_passes.clear();
        let mut k = 2;
        while k <= n {
            let mut j = k / 2;
            while j > 0 {
                self.sort_passes.push(gpu.storage("learned route sort pass", &[[k, j, 0u32, 0]]));
                j /= 2;
            }
            k *= 2;
        }
    }

    /// The routing into `t.isl` (routed islands ROUTED_BIT with the routed impactors' mask;
    /// errors into `t.ctl[20]`).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, t: &RouteTables, island_cap: u32, engine_base: u32, impactors: u32, dt: f64, speed: f64, timer: Option<&crate::learned_timing::StageTimer>) {
        assert!(island_cap + 32 <= self.n_pad, "learned routing sized for fewer islands (Route::ensure)");
        let (ground, ground_height, ground_modulus) = match self.cfg.ground {
            Some((h, m)) => (1, h, m),
            None => (0, 0.0, 1.0),
        };
        let params = gpu.uniform(
            "learned route params",
            &RouteParams {
                island_cap,
                impactor_count: impactors,
                engine_base,
                n_pad: self.n_pad,
                dt: dt as f32,
                speed: speed as f32,
                tiered: self.cfg.tiered as u32,
                fast_joins: self.cfg.fast_joins as u32,
                debris: self.cfg.debris as u32,
                ground,
                ground_height,
                ground_modulus,
                edge_cap: self.edge_cap,
                always: self.always() as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let n = self.n_pad;
        let run = |enc: &mut wgpu::CommandEncoder, k: &Kernel, pass: &wgpu::Buffer, groups: u32| {
            let bufs: [&wgpu::Buffer; 21] = [&params, t.islands, t.chunks, t.state, t.ctab, t.bodies, t.bond_static, t.materials, &self.rtab, t.imps, t.counters, t.route, &self.parts, &self.sorted, &self.edges, &self.lab, &self.actl, &self.seed, t.isl, pass, t.ctl];
            let group = gpu.bind(&k.group, &bufs);
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: timer.and_then(|t| t.writes("route")) });
            p.set_pipeline(&k.pipeline);
            p.set_bind_group(0, &group, &[]);
            p.dispatch_workgroups(groups.max(1), 1, 1);
        };
        let r0 = &self.rounds[0];
        let threads = |m: u32| m.div_ceil(64);
        run(enc, &self.k.islands, r0, island_cap);
        run(enc, &self.k.impactors, r0, 1);
        run(enc, &self.k.debris, r0, threads(island_cap));
        run(enc, &self.k.keys, r0, threads(n));
        for p in &self.sort_passes {
            run(enc, &self.k.bitonic, p, threads(n / 2));
        }
        run(enc, &self.k.sweep, r0, threads(n));
        for r in &self.rounds {
            run(enc, &self.k.hook, r, threads(self.edge_cap));
            run(enc, &self.k.jump, r, threads(n));
        }
        run(enc, &self.k.roots, r0, threads(n));
        run(enc, &self.k.finish, r0, threads(n));
        run(enc, &self.k.mark, r0, threads(island_cap));
    }
}
