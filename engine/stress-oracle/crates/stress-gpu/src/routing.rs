//! The routed resident frame's check (`shaders/routing.slang`, engine/docs/stress-baseline-goal.md
//! M5): at the end of a resident frame, whether the next frame may route an island, read
//! with the frame's status. Under the tiered coupling only an impactor's contact seeds an
//! island; the check is a superset of the host's test (`World::select_islands`), so a frame
//! it lets run resident is one the host's routing would have given no island.

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RouteParams {
    island_count: u32,
    impactor_count: u32,
    engine_base: u32,
    pad0: u32,
    dt: f32,
    speed: f32,
    margin: f32,
    pad1: f32,
}

const BINDINGS: [Binding; 8] = [
    Binding::Uniform,   // params
    Binding::Storage,   // islands
    Binding::Storage,   // chunk statics
    Binding::Storage,   // chunk state
    Binding::Storage,   // ctab
    Binding::Storage,   // engine bodies
    Binding::Storage,   // impactors
    Binding::StorageRw, // route
];

/// The solver's tables the check reads (`GpuSolver::routing_inputs`).
pub struct RouteInputs<'a> {
    pub islands: &'a wgpu::Buffer,
    pub chunks: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    /// The punch's too (`Punch`): bond statics and records, materials, the cascade's status.
    pub bonds: &'a wgpu::Buffer,
    pub bond_dyn: &'a wgpu::Buffer,
    pub materials: &'a wgpu::Buffer,
    pub status: Option<&'a wgpu::Buffer>,
    /// Indirect arguments over the live islands (one group each), when the GPU may add
    /// islands within the frame; otherwise `island_count` groups.
    pub island_args: Option<&'a wgpu::Buffer>,
    pub island_count: u32,
}

pub struct Routing {
    kernel: Kernel,
    pub(crate) route: wgpu::Buffer,
    impactors: wgpu::Buffer,
    /// Per impactor: its engine body and bounding radius, as uploaded.
    list: Vec<(u32, f32)>,
}

impl Routing {
    pub fn new(gpu: &Gpu) -> Routing {
        Routing {
            kernel: gpu.compute(&shaders::ROUTING, "route_islands", &BINDINGS),
            route: gpu.storage("routing flag", &[0u32; 4]),
            impactors: gpu.storage("routing impactors", &[[0f32; 4]]),
            list: Vec::new(),
        }
    }

    /// The impactors: (engine body, bounding radius about the body's centre) each.
    pub fn set_impactors(&mut self, gpu: &Gpu, list: &[(u32, f32)]) {
        if list == self.list.as_slice() {
            return;
        }
        let words: Vec<[f32; 4]> = list.iter().map(|&(b, r)| [f32::from_bits(b), r, 0.0, 0.0]).collect();
        self.impactors = if words.is_empty() { gpu.storage("routing impactors", &[[0f32; 4]]) } else { gpu.storage("routing impactors", &words) };
        self.list = list.to_vec();
    }

    /// The check, after everything else in the frame's submission: the flag cleared, then
    /// set by any island an impactor may strike within `dt`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, inputs: &RouteInputs, ctab: &wgpu::Buffer, engine_base: u32, bodies: &wgpu::Buffer, dt: f64, speed: f64) {
        encoder.clear_buffer(&self.route, 0, None);
        if self.list.is_empty() {
            return;
        }
        let params = gpu.uniform(
            "routing params",
            &RouteParams {
                island_count: inputs.island_count,
                impactor_count: self.list.len() as u32,
                engine_base,
                pad0: 0,
                dt: dt as f32,
                // f32 against the host's f64: a pair within a part in a thousand of the
                // speed or a millimetre of the bounds counts.
                speed: (speed * (1.0 - 1e-3)) as f32,
                margin: 1e-3,
                pad1: 0.0,
            },
        );
        let group = gpu.bind(&self.kernel.group, &[&params, inputs.islands, inputs.chunks, inputs.state, ctab, bodies, &self.impactors, &self.route]);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        match inputs.island_args {
            Some(a) => pass.dispatch_workgroups_indirect(a, 0),
            None => pass.dispatch_workgroups(inputs.island_count.max(1), 1, 1),
        }
    }

    /// The flag's range (for the frame's one read).
    pub fn range(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.route, 0, 16)
    }

    /// Whether the read flag says the next frame may route an island.
    pub fn flagged(bytes: &[u8]) -> bool {
        bytemuck::cast_slice::<u8, u32>(&bytes[..4])[0] != 0
    }
}

/// `STRESS_IMPACT_PUNCH=gpu` (a model under measurement): the predictive punch on the GPU
/// (`shaders/punch.slang`), at the end of a resident frame, with a split pass after it.
pub fn punch_gpu() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("STRESS_IMPACT_PUNCH").is_ok_and(|v| v == "gpu"))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PunchParams {
    island_count: u32,
    impactor_count: u32,
    engine_base: u32,
    pad0: u32,
    dt: f32,
    speed: f32,
    pad1: f32,
    pad2: f32,
}

const PUNCH_BINDINGS: [Binding; 13] = [
    Binding::Uniform,   // params
    Binding::Storage,   // islands
    Binding::Storage,   // chunk statics
    Binding::Storage,   // chunk state
    Binding::Storage,   // ctab
    Binding::Storage,   // engine bodies
    Binding::Storage,   // impactors
    Binding::Storage,   // bond statics
    Binding::StorageRw, // bond records
    Binding::Uniform,   // materials
    Binding::StorageRw, // cascade status
    Binding::StorageRw, // the fast free islands (declared before the flag: Metal's order)
    Binding::Storage,   // the routing check's flag
];

pub struct Punch {
    kernel: Kernel,
    fast_kernel: Kernel,
    /// threshold.slang `threshold_predict` (`Profile::Threshold`): every bond the strike's
    /// force beats, in place of the plug.
    predict_kernel: Kernel,
    fast: wgpu::Buffer,
    impactors: wgpu::Buffer,
    /// Per impactor: engine body, bounding radius, face radius, mass, plane-strain modulus.
    list: Vec<[f32; 8]>,
}

impl Punch {
    pub fn new(gpu: &Gpu) -> Punch {
        Punch {
            kernel: gpu.compute(&shaders::PUNCH, "punch_islands", &PUNCH_BINDINGS),
            fast_kernel: gpu.compute(&shaders::PUNCH, "punch_fast", &PUNCH_BINDINGS),
            predict_kernel: gpu.compute(&shaders::THRESHOLD, "threshold_predict", &crate::solver::THRESHOLD_BINDINGS),
            fast: gpu.storage("punch fast islands", &vec![[0f32; 4]; 1 + 3 * 128]),
            impactors: gpu.storage("punch impactors", &[[0f32; 4]; 2]), list: Vec::new() }
    }

    /// (engine body, bounding radius, face radius, mass, plane-strain modulus) per impactor.
    pub fn set_impactors(&mut self, gpu: &Gpu, list: &[(u32, f32, f32, f32, f32)]) {
        let words: Vec<[f32; 8]> = list.iter().map(|&(b, r, f, m, e)| [f32::from_bits(b), r, f, m, e, 0.0, 0.0, 0.0]).collect();
        if words == self.list {
            return;
        }
        self.impactors = if words.is_empty() { gpu.storage("punch impactors", &[[0f32; 4]; 2]) } else { gpu.storage("punch impactors", &words) };
        self.list = words;
    }

    #[allow(clippy::too_many_arguments)]
    /// `predict`: the threshold profile's predicted breaks (`threshold_predict`) instead of
    /// the punch's plug.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, inputs: &RouteInputs, ctab: &wgpu::Buffer, engine_base: u32, bodies: &wgpu::Buffer, dt: f64, speed: f64, route: &Routing, predict: bool) {
        let Some(status) = inputs.status else { return };
        if self.list.is_empty() {
            return;
        }
        let params = gpu.uniform(
            "punch params",
            &PunchParams { island_count: inputs.island_count, impactor_count: self.list.len() as u32, engine_base, pad0: 0, dt: dt as f32, speed: speed as f32, pad1: 0.0, pad2: 0.0 },
        );
        let buffers: [&wgpu::Buffer; 13] = [&params, inputs.islands, inputs.chunks, inputs.state, ctab, bodies, &self.impactors, inputs.bonds, inputs.bond_dyn, inputs.materials, status, &self.fast, &route.route];
        let (kernel, group) = if predict {
            let tparams = gpu.uniform(
                "threshold predict params",
                &crate::solver::ThresholdParams { island_count: inputs.island_count, impactor_count: self.list.len() as u32, engine_base, dt: dt as f32, speed: speed as f32, ..Default::default() },
            );
            let tbuffers: [&wgpu::Buffer; 12] = [&tparams, inputs.islands, inputs.bonds, inputs.bond_dyn, ctab, inputs.materials, status, inputs.chunks, inputs.state, bodies, &self.impactors, &self.fast];
            (&self.predict_kernel, gpu.bind(&self.predict_kernel.group, &tbuffers))
        } else {
            (&self.kernel, gpu.bind(&self.kernel.group, &buffers))
        };
        let fast_group = gpu.bind(&self.fast_kernel.group, &buffers);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        // The fast free islands first (one group), then the islands.
        pass.set_pipeline(&self.fast_kernel.pipeline);
        pass.set_bind_group(0, &fast_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &group, &[]);
        match inputs.island_args {
            Some(a) => pass.dispatch_workgroups_indirect(a, 0),
            None => pass.dispatch_workgroups(inputs.island_count.max(1), 1, 1),
        }
    }
}
