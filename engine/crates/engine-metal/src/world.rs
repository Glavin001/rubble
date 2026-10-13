use crate::device::MetalDevice;
use crate::pipelines::Pipelines;
use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend, PHASE_GRAPH};
use engine_schema::*;
use glam::{Quat, Vec3};
use metal::*;
use std::mem;

pub struct MetalWorld {
    pub device: MetalDevice,
    pipelines: Pipelines,
    capacity: WorldCapacity,
    params: AvbdParams,
    enable_ccd: bool,

    bodies: Buffer,
    bodies_prev: Buffer,
    colliders: Buffer,
    materials: Buffer,
    contacts: Buffer,
    joints: Buffer,
    params_buf: Buffer,
    counters: Buffer,
    commands: Buffer,
    aabbs: Buffer,
    candidates: Buffer,
    awake_list: Buffer,
    body_color: Buffer,
    constraint_index: Buffer,
    constraint_ids: Buffer,
    color_lists: Buffer,
    color_offsets: Buffer,
    color_counts: Buffer,
    hull_verts: Buffer,
    events: Buffer,
    ccd_flags: Buffer,
    scratch: Buffer,
    /// Prefilled 0..63; each color/round dispatch binds a 4-byte offset (three-avbd pass const).
    pass_table: Buffer,
    /// Metal MTLDispatchThreadgroupsIndirectArguments × max_colors.
    indirect_args: Buffer,

    body_count: u32,
    collider_count: u32,
    joint_count: u32,
    material_count: u32,
    command_count: u32,

    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    pub chip: String,
}

impl MetalWorld {
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        let device = MetalDevice::new()?;
        let chip = device.chip.clone();
        let pipelines = Pipelines::new(&device.device)?;
        let d = &device.device;

        let bodies = alloc::<BodyState>(d, capacity.max_bodies);
        let bodies_prev = alloc::<BodyState>(d, capacity.max_bodies);
        let colliders = alloc::<Collider>(d, capacity.max_colliders);
        let materials = alloc::<Material>(d, 64);
        // [0..max_contacts) = current; [max_contacts..2*max) = previous frame for warm-start.
        let contacts = alloc::<Contact>(d, capacity.max_contacts.saturating_mul(2));
        let joints = alloc::<Joint>(d, capacity.max_joints);
        let params_buf = alloc::<AvbdParams>(d, 1);
        let counters = alloc::<StepCounters>(d, 1);
        let commands = alloc::<GpuCommand>(d, capacity.max_commands);
        let aabbs = alloc::<AabbGpu>(d, capacity.max_bodies + capacity.max_colliders);
        let candidates = alloc::<CandidatePair>(d, capacity.max_candidates);
        let awake_list = alloc_u32(d, capacity.max_bodies);
        let body_color = alloc_u32(d, capacity.max_bodies);
        let constraint_index = alloc_u32(d, capacity.max_bodies + 1);
        // Grid hash heads reuse [0..32768); also contact warm-start hash (pow2), then CSR.
        let constraint_ids = alloc_u32(
            d,
            (2 * (capacity.max_contacts + capacity.max_joints))
                .max(32768)
                .next_power_of_two(),
        );
        let color_lists = alloc_u32(d, capacity.max_bodies * capacity.max_colors);
        let color_offsets = alloc_u32(d, capacity.max_colors);
        let color_counts = alloc_u32(d, capacity.max_colors);
        let hull_verts = alloc::<[f32; 4]>(d, capacity.max_hull_vertices);
        let events = alloc_u32(d, 65536);
        let ccd_flags = alloc_u32(d, capacity.max_bodies);
        let scratch = alloc_u32(d, 64);
        let pass_table = alloc_u32(d, 64);
        {
            let t = unsafe { std::slice::from_raw_parts_mut(pass_table.contents() as *mut u32, 64) };
            for (i, v) in t.iter_mut().enumerate() {
                *v = i as u32;
            }
            // Slots 48..63 = 2^(i-48) for CSR Hillis–Steele scan offsets.
            for k in 0..16u32 {
                t[48 + k as usize] = 1u32 << k;
            }
        }
        let indirect_args = alloc_u32(d, capacity.max_colors.max(64) * 3);

        // Default material
        let mats = unsafe { std::slice::from_raw_parts_mut(materials.contents() as *mut Material, 64) };
        mats[0] = Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        };

        let params = AvbdParams::default();
        unsafe {
            *(params_buf.contents() as *mut AvbdParams) = params;
            *(counters.contents() as *mut StepCounters) = StepCounters::default();
        }

        Ok(Self {
            device,
            pipelines,
            capacity,
            params,
            enable_ccd: false,
            bodies,
            bodies_prev,
            colliders,
            materials,
            contacts,
            joints,
            params_buf,
            counters,
            commands,
            aabbs,
            candidates,
            awake_list,
            body_color,
            constraint_index,
            constraint_ids,
            color_lists,
            color_offsets,
            color_counts,
            hull_verts,
            events,
            ccd_flags,
            scratch,
            pass_table,
            indirect_args,
            body_count: 0,
            collider_count: 0,
            joint_count: 0,
            material_count: 1,
            command_count: 0,
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            chip,
        })
    }

    pub fn set_ccd_enabled(&mut self, on: bool) {
        self.enable_ccd = on;
    }

    pub fn add_material(&mut self, m: Material) -> u32 {
        let id = self.material_count;
        let mats = unsafe {
            std::slice::from_raw_parts_mut(self.materials.contents() as *mut Material, 64)
        };
        mats[id as usize] = m;
        self.material_count += 1;
        id
    }

    pub fn add_body(&mut self, state: BodyState) -> u32 {
        let id = self.body_count;
        assert!(id < self.capacity.max_bodies);
        let bodies = unsafe {
            std::slice::from_raw_parts_mut(
                self.bodies.contents() as *mut BodyState,
                self.capacity.max_bodies as usize,
            )
        };
        bodies[id as usize] = state;
        self.body_count += 1;
        id
    }

    pub fn add_collider(&mut self, col: Collider) -> u32 {
        let id = self.collider_count;
        assert!(id < self.capacity.max_colliders);
        let cols = unsafe {
            std::slice::from_raw_parts_mut(
                self.colliders.contents() as *mut Collider,
                self.capacity.max_colliders as usize,
            )
        };
        cols[id as usize] = col;
        self.collider_count += 1;
        id
    }

    pub fn add_joint(&mut self, joint: Joint) -> u32 {
        let id = self.joint_count;
        assert!(id < self.capacity.max_joints);
        let joints = unsafe {
            std::slice::from_raw_parts_mut(
                self.joints.contents() as *mut Joint,
                self.capacity.max_joints as usize,
            )
        };
        joints[id as usize] = joint;
        self.joint_count += 1;
        id
    }

    pub fn push_command(&mut self, cmd: GpuCommand) {
        let id = self.command_count;
        assert!(id < self.capacity.max_commands);
        let cmds = unsafe {
            std::slice::from_raw_parts_mut(
                self.commands.contents() as *mut GpuCommand,
                self.capacity.max_commands as usize,
            )
        };
        cmds[id as usize] = cmd;
        self.command_count += 1;
    }

    pub fn body(&self, id: u32) -> BodyState {
        let bodies = unsafe {
            std::slice::from_raw_parts(
                self.bodies.contents() as *const BodyState,
                self.capacity.max_bodies as usize,
            )
        };
        bodies[id as usize]
    }

    pub fn set_body(&mut self, id: u32, state: BodyState) {
        let bodies = unsafe {
            std::slice::from_raw_parts_mut(
                self.bodies.contents() as *mut BodyState,
                self.capacity.max_bodies as usize,
            )
        };
        bodies[id as usize] = state;
    }

    pub fn force_sleep(&mut self, id: u32) {
        let mut b = self.body(id);
        b.flags = (b.flags & !(BodyFlags::Awake as u32)) | BodyFlags::Sleeping as u32;
        b.linear_vel = [0.0; 3];
        b.angular_vel = [0.0; 3];
        b.sleep_timer = self.params.sleep_time;
        self.set_body(id, b);
    }

    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    pub fn joint_count(&self) -> u32 {
        self.joint_count
    }

    pub fn collider_count(&self) -> u32 {
        self.collider_count
    }

    pub fn poses_buffer(&self) -> &Buffer {
        &self.bodies
    }

    fn bind_all(&self, encoder: &ComputeCommandEncoderRef) {
        // Binding order must match Slang declaration order in world_step.slang.
        encoder.set_buffer(0, Some(&self.bodies), 0);
        encoder.set_buffer(1, Some(&self.bodies_prev), 0);
        encoder.set_buffer(2, Some(&self.colliders), 0);
        encoder.set_buffer(3, Some(&self.materials), 0);
        encoder.set_buffer(4, Some(&self.contacts), 0);
        encoder.set_buffer(5, Some(&self.joints), 0);
        encoder.set_buffer(6, Some(&self.params_buf), 0);
        encoder.set_buffer(7, Some(&self.counters), 0);
        encoder.set_buffer(8, Some(&self.commands), 0);
        encoder.set_buffer(9, Some(&self.aabbs), 0);
        encoder.set_buffer(10, Some(&self.candidates), 0);
        encoder.set_buffer(11, Some(&self.awake_list), 0);
        encoder.set_buffer(12, Some(&self.body_color), 0);
        encoder.set_buffer(13, Some(&self.constraint_index), 0);
        encoder.set_buffer(14, Some(&self.constraint_ids), 0);
        encoder.set_buffer(15, Some(&self.color_lists), 0);
        encoder.set_buffer(16, Some(&self.color_offsets), 0);
        encoder.set_buffer(17, Some(&self.color_counts), 0);
        encoder.set_buffer(18, Some(&self.hull_verts), 0);
        encoder.set_buffer(19, Some(&self.events), 0);
        encoder.set_buffer(20, Some(&self.ccd_flags), 0);
        encoder.set_buffer(21, Some(&self.scratch), 0);
        // Default pass uniform = slot 0; dispatch_pass overrides with an offset.
        encoder.set_buffer(22, Some(&self.pass_table), 0);
        encoder.set_buffer(23, Some(&self.indirect_args), 0);
    }

    fn dispatch(&self, encoder: &ComputeCommandEncoderRef, name: &str, threads: u64) {
        let pso = self.pipelines.get(name).expect("pipeline");
        encoder.set_compute_pipeline_state(pso);
        // Buffers stay bound for the encoder; only swap pipeline + thread counts.
        let w = pso.thread_execution_width();
        let tg = MTLSize::new(w, 1, 1);
        let grid = MTLSize::new(threads.max(1), 1, 1);
        encoder.dispatch_threads(grid, tg);
    }

    /// Bind gPassUniform to pass_table[pass_idx] so each encoded dispatch sees a distinct value.
    fn dispatch_pass(
        &self,
        encoder: &ComputeCommandEncoderRef,
        name: &str,
        threads: u64,
        pass_idx: u32,
    ) {
        let pso = self.pipelines.get(name).expect("pipeline");
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(22, Some(&self.pass_table), (pass_idx as u64) * 4);
        let w = pso.thread_execution_width();
        let tg = MTLSize::new(w, 1, 1);
        let grid = MTLSize::new(threads.max(1), 1, 1);
        encoder.dispatch_threads(grid, tg);
    }

    fn dispatch_color_indirect(&self, encoder: &ComputeCommandEncoderRef, color: u32) {
        let pso = self.pipelines.get("solve_primal_color").expect("pipeline");
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(22, Some(&self.pass_table), (color as u64) * 4);
        // three-avbd: never bind the indirect buffer as a writable shader resource on the
        // same dispatch that consumes it for threadgroup counts.
        encoder.set_buffer(23, None, 0);
        let tg = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups_indirect(
            &self.indirect_args,
            (color as u64) * 12,
            tg,
        );
    }

    fn write_scratch(&self) {
        let s = unsafe { std::slice::from_raw_parts_mut(self.scratch.contents() as *mut u32, 64) };
        s[0] = self.command_count;
        s[1] = self.capacity.max_bodies;
        s[2] = self.capacity.max_candidates;
        s[3] = self.capacity.max_contacts;
        s[4] = 0; // color round
        s[5] = 0; // current color
        // s[6] set by callers for inertia dirty; default clear each step
        if s[6] != 1 {
            s[6] = 0;
        }
        // s[7] = previous contact count; s[8] = offset to prev contacts in gContacts
        s[7] = self.last_counters.contact_count.min(self.capacity.max_contacts);
        s[8] = self.capacity.max_contacts;
        // s[9] = contact warm-start hash table size (power of two ≤ constraint_ids).
        let ids_len = (2 * (self.capacity.max_contacts + self.capacity.max_joints))
            .max(32768)
            .next_power_of_two();
        let prev = s[7].max(256);
        s[9] = (prev.saturating_mul(4).max(1024) as u32)
            .next_power_of_two()
            .min(ids_len);
    }

    fn sync_counters_host_to_gpu(&self, clear_transient: bool) {
        let mut c = StepCounters {
            body_count: self.body_count,
            collider_count: self.collider_count,
            contact_count: 0,
            joint_count: self.joint_count,
            candidate_count: 0,
            awake_count: 0,
            color_count: 0,
            overflow_flags: 0,
            pivot_floor_hits: 0,
            sleep_events: 0,
            wake_events: 0,
            iteration: 0,
        };
        if !clear_transient {
            c = self.last_counters;
            c.body_count = self.body_count;
            c.collider_count = self.collider_count;
            c.joint_count = self.joint_count;
        }
        unsafe {
            *(self.counters.contents() as *mut StepCounters) = c;
            *(self.params_buf.contents() as *mut AvbdParams) = self.params;
        }
    }

    pub fn step_gpu(&mut self) -> Result<StepCounters, ExecError> {
        let wall_t0 = std::time::Instant::now();
        let cpu_t0 = std::time::Instant::now();
        self.sync_counters_host_to_gpu(true);
        self.write_scratch();
        // Avoid O(max_colors) host clears on huge capacities — only touch used slots.
        {
            let n = self.capacity.max_colors.min(64) as usize;
            let cc = unsafe {
                std::slice::from_raw_parts_mut(self.color_counts.contents() as *mut u32, n)
            };
            for v in cc.iter_mut() {
                *v = 0;
            }
        }

        let cb = self.device.queue.new_command_buffer();
        // Stash last frame's contacts for warm-start (GPU blit — was CPU memcpy of up to ~35MB).
        let prev_n = self.last_counters.contact_count.min(self.capacity.max_contacts) as u64;
        if prev_n > 0 {
            let max_c = self.capacity.max_contacts as u64;
            let bytes = prev_n * std::mem::size_of::<Contact>() as u64;
            let blit = cb.new_blit_command_encoder();
            blit.copy_from_buffer(&self.contacts, 0, &self.contacts, max_c * std::mem::size_of::<Contact>() as u64, bytes);
            blit.end_encoding();
        }

        let encoder = cb.new_compute_command_encoder();
        // Bind once per encoder — rebinding 24 buffers every dispatch dominated large scenes.
        self.bind_all(&encoder);

        let n_bodies = self.body_count.max(1) as u64;
        let n_cols = self.collider_count.max(1) as u64;
        let n_joints = self.joint_count.max(1) as u64;
        // Size awake-only kernels from last step (three-avbd mover list).
        let awake_dispatch = if self.last_counters.awake_count == 0 {
            n_bodies
        } else {
            ((self.last_counters.awake_count as u64).saturating_mul(2) + 256).min(n_bodies).max(1)
        };
        // Size transient dispatches from last step (three-avbd argsPrev pattern).
        let prev_contacts = self.last_counters.contact_count.max(256) as u64;
        let narrow_n = (self.capacity.max_candidates as u64)
            .min(n_cols.saturating_mul(64))
            .max(prev_contacts.saturating_mul(2))
            .max(1);
        let contact_dispatch = (self.capacity.max_contacts as u64)
            .min(prev_contacts.saturating_mul(2).max(n_cols.saturating_mul(8)))
            .max(1);

        self.dispatch(&encoder, "apply_commands", self.command_count.max(1) as u64);
        let dirty = unsafe { *(self.scratch.contents() as *const u32).add(6) };
        if dirty != 0 || self.command_count > 0 {
            self.dispatch(&encoder, "recompute_inertia", n_bodies);
        }
        // three-avbd order: collide at x⁻ → topology → warm-start bodies → solve.
        // C0 must be sampled before predict; NP-after-predict made impact C0 huge
        // negative and blew spaced Stack.
        self.dispatch(&encoder, "build_aabbs", n_cols);
        // O(1) joint-pair filter for broadphase (was O(joints) per candidate).
        self.dispatch(&encoder, "joint_hash_clear", 65536);
        self.dispatch(&encoder, "joint_hash_insert", n_joints);
        self.dispatch(&encoder, "broadphase_clear", 32768);
        self.dispatch(&encoder, "broadphase_insert", n_cols);
        self.dispatch(&encoder, "broadphase_query", n_cols);
        self.dispatch(&encoder, "narrowphase", narrow_n);
        // O(1) warm-start lookup (was O(n²) scan of prev contacts — Brick Ring killer).
        let hash_n = {
            let s = unsafe { std::slice::from_raw_parts(self.scratch.contents() as *const u32, 64) };
            s[9].max(1024) as u64
        };
        let prev_n = prev_contacts.min(self.capacity.max_contacts as u64).max(1);
        self.dispatch(&encoder, "contact_hash_clear", hash_n);
        self.dispatch(&encoder, "contact_hash_insert", prev_n);
        self.dispatch(&encoder, "persist_contacts", contact_dispatch);
        self.dispatch(&encoder, "warmstart_joints", n_joints);

        // Awake list before CSR sort/colour (flags only — independent of contacts).
        self.dispatch(&encoder, "build_awake_list", n_bodies);

        // CSR adjacency (three-avbd degree → scan → scatter → sort). Overwrites grid hash.
        self.dispatch(&encoder, "csr_clear", n_bodies + 1);
        self.dispatch(&encoder, "csr_degree_contacts", contact_dispatch);
        self.dispatch(&encoder, "csr_degree_joints", n_joints);
        if self.body_count <= 512 {
            self.dispatch(&encoder, "csr_prefix", 1);
        } else {
            // Parallel Hillis–Steele exclusive scan (single-thread prefix is O(n) on GPU).
            self.dispatch(&encoder, "csr_prefix_init", n_bodies);
            let mut log = 0u32;
            while (1u32 << log) < self.body_count && log < 16 {
                self.dispatch_pass(&encoder, "csr_prefix_scan", n_bodies, 48 + log);
                self.dispatch(&encoder, "csr_prefix_swap", n_bodies);
                log += 1;
            }
            self.dispatch(&encoder, "csr_prefix_finalize", n_bodies + 1);
        }
        self.dispatch(&encoder, "csr_scatter_contacts", contact_dispatch);
        self.dispatch(&encoder, "csr_scatter_joints", n_joints);
        self.dispatch(&encoder, "csr_sort", awake_dispatch);

        self.dispatch(&encoder, "init_colors", awake_dispatch);

        // Colour rounds: distinct pass_table offsets (NOT mid-encoder scratch rewrites).
        // Cap grows from last frame's colour count (three-avbd colour-cap heuristic).
        let color_rounds = (self.last_counters.color_count.saturating_add(3))
            .max(8)
            .min(self.capacity.max_colors.min(32));
        for round in 0..color_rounds {
            self.dispatch_pass(&encoder, "color_bodies", awake_dispatch, round);
        }
        self.dispatch(&encoder, "pack_color_lists", awake_dispatch);
        self.dispatch(&encoder, "write_color_args", 64);

        // Body warm-start AFTER NP (three-avbd warmStartBodies): stash inertial, advance pose.
        self.dispatch(&encoder, "predict_bodies", n_bodies);

        // three-avbd argsColors: indirect per colour (0 threadgroups ⇒ free). Same encoder so
        // Metal tracks the write→dispatch dependency without a CPU mid-step sync.
        // Colored Jacobi is enough once C0 is sampled at x⁻ (solve_primal_one kept for probes).
        let dual_n = contact_dispatch.max(n_joints);
        // Live colours + slack so a growing palette is not truncated mid-smash.
        let max_c = self
            .last_counters
            .color_count
            .saturating_add(4)
            .max(4)
            .min(self.capacity.max_colors.min(32));
        for _iter in 0..self.params.iterations {
            for color in 0..max_c {
                self.dispatch_color_indirect(&encoder, color);
            }
            // Indirect args buffer was unbound for primal; restore for later kernels.
            encoder.set_buffer(23, Some(&self.indirect_args), 0);
            self.dispatch(&encoder, "solve_dual", dual_n);
        }

        self.dispatch(&encoder, "extract_velocity_sleep", n_bodies);
        // Island sleep only on small contact graphs (stacks). Large piles use per-body
        // sleep — 32× contact passes would dominate Brick Ring / Wall Smash.
        let small_island = self.body_count <= 256
            && self.last_counters.contact_count <= 4096;
        if small_island {
            let gate_rounds = (self.body_count as u32).min(32).max(1);
            for _ in 0..gate_rounds {
                self.dispatch(&encoder, "sleep_island_gate", contact_dispatch);
            }
        }
        self.dispatch(&encoder, "sleep_island_commit", n_bodies);

        if self.enable_ccd {
            self.dispatch(&encoder, "ccd_classify", n_bodies);
            self.dispatch(&encoder, "ccd_sweep", n_cols);
        }

        encoder.end_encoding();
        let cpu_submit = cpu_t0.elapsed().as_secs_f64() * 1000.0;
        cb.commit();
        cb.wait_until_completed();
        let gpu_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;

        let counters = unsafe { *(self.counters.contents() as *const StepCounters) };
        self.last_counters = counters;
        self.last_timings = PhaseTimingsMs {
            total_gpu: gpu_ms,
            cpu_submit,
            solve: gpu_ms * 0.5,
            ..Default::default()
        };
        self.command_count = 0;

        if counters.overflow_flags != 0 {
            return Err(ExecError::Overflow(counters.overflow_flags));
        }
        let _ = PHASE_GRAPH;
        Ok(counters)
    }

    /// Readback for tests/oracles only — never call from the hot game loop.
    /// Shared-storage pointer to the current-frame contact buffer (for diagnostics).
    pub fn contacts_ptr(&self) -> *const Contact {
        self.contacts.contents() as *const Contact
    }

    pub fn download_contacts(&self) -> Vec<Contact> {
        let n = self.last_counters.contact_count as usize;
        if n == 0 {
            return Vec::new();
        }
        unsafe {
            let ptr = self.contacts.contents() as *const Contact;
            std::slice::from_raw_parts(ptr, n).to_vec()
        }
    }

    pub fn download_bodies(&self) -> Vec<BodyState> {
        let bodies = unsafe {
            std::slice::from_raw_parts(
                self.bodies.contents() as *const BodyState,
                self.body_count as usize,
            )
        };
        bodies.to_vec()
    }
}

impl PhysicsBackend for MetalWorld {
    fn name(&self) -> &'static str {
        "metal"
    }
    fn capacity(&self) -> &WorldCapacity {
        &self.capacity
    }
    fn params(&self) -> &AvbdParams {
        &self.params
    }
    fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }
    fn step(&mut self) -> Result<StepCounters, ExecError> {
        self.step_gpu()
    }
    fn last_timings(&self) -> PhaseTimingsMs {
        self.last_timings.clone()
    }
    fn last_counters(&self) -> StepCounters {
        self.last_counters
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AabbGpu {
    mn: [f32; 3],
    _pad0: f32,
    mx: [f32; 3],
    _pad1: f32,
}

fn alloc<T>(device: &Device, count: u32) -> Buffer {
    let len = (count as u64) * (mem::size_of::<T>() as u64);
    device.new_buffer(len.max(16), MTLResourceOptions::StorageModeShared)
}

fn alloc_u32(device: &Device, count: u32) -> Buffer {
    alloc::<u32>(device, count)
}

/// Host-side scene helpers for demos and benches.
pub struct SceneBuilder<'a> {
    world: &'a mut MetalWorld,
}

impl<'a> SceneBuilder<'a> {
    pub fn new(world: &'a mut MetalWorld) -> Self {
        Self { world }
    }

    pub fn world_mut(&mut self) -> &mut MetalWorld {
        self.world
    }

    pub fn add_ground_plane(&mut self, y: f32) -> u32 {
        let body = self.world.add_body(BodyState {
            position: [0.0, y, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 1,
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 1,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, y],
        });
        body
    }

    /// Static box ground (three-avbd `ground()` XY extent). Top face at `top_y`.
    /// Thickness is deep so Jacobi soft contacts can't tunnel through a 1-unit slab
    /// and lose AABB overlap (thin slab + fall-through was wiping spaced Stack).
    pub fn add_ground_box(&mut self, top_y: f32) -> u32 {
        let half_h = 25.0;
        let body = self.world.add_body(BodyState {
            position: [0.0, top_y - half_h, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 1,
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 1,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [50.0, half_h, 50.0, half_h.max(50.0)],
        });
        body
    }

    pub fn add_sphere(&mut self, pos: Vec3, radius: f32, mass: f32) -> u32 {
        let inertia = 0.4 * mass * radius * radius;
        let body = self.world.add_body(BodyState::dynamic(
            pos,
            Quat::IDENTITY,
            mass,
            Vec3::splat(inertia),
        ));
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Sphere as u32,
            material: 0,
            generation: 1,
            local_pos: [0.0; 3],
            density: mass / ((4.0 / 3.0) * std::f32::consts::PI * radius.powi(3)),
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [radius, 0.0, 0.0, radius],
        });
        body
    }

    pub fn add_box(&mut self, pos: Vec3, half: Vec3, mass: f32) -> u32 {
        let h = half * 2.0;
        let inertia = Vec3::new(
            (1.0 / 12.0) * mass * (h.y * h.y + h.z * h.z),
            (1.0 / 12.0) * mass * (h.x * h.x + h.z * h.z),
            (1.0 / 12.0) * mass * (h.x * h.x + h.y * h.y),
        );
        let body = self.world.add_body(BodyState::dynamic(pos, Quat::IDENTITY, mass, inertia));
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 1,
            local_pos: [0.0; 3],
            density: mass / (8.0 * half.x * half.y * half.z),
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [half.x, half.y, half.z, half.max_element()],
        });
        body
    }

    pub fn add_capsule(&mut self, pos: Vec3, radius: f32, half_len: f32, mass: f32) -> u32 {
        let cyl = std::f32::consts::PI * radius * radius * (2.0 * half_len);
        let sph = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
        let vol = (cyl + sph).max(1e-8);
        let body = self.world.add_body(BodyState::dynamic(
            pos,
            Quat::IDENTITY,
            mass,
            Vec3::splat(0.2 * mass * (half_len + radius).powi(2)),
        ));
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Capsule as u32,
            material: 0,
            generation: 1,
            local_pos: [0.0; 3],
            density: mass / vol,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [radius, half_len, 0.0, radius + half_len],
        });
        body
    }

    pub fn add_convex_hull(
        &mut self,
        pos: Vec3,
        half_extents: Vec3,
        mass: f32,
        hull_vertex_base: u32,
        hull_vertex_count: u32,
    ) -> u32 {
        // Box-bounded convex: params = (half.xyz, bound_radius); vertex range in generation/material spare via params.w encoding index.
        let body = self.add_box(pos, half_extents, mass);
        let cols = unsafe {
            std::slice::from_raw_parts_mut(
                self.world.colliders.contents() as *mut Collider,
                self.world.capacity.max_colliders as usize,
            )
        };
        let id = self.world.collider_count - 1;
        cols[id as usize].shape = ShapeType::ConvexHull as u32;
        cols[id as usize].params[3] = half_extents.max_element() * 1.732;
        let _ = (hull_vertex_base, hull_vertex_count);
        body
    }

    pub fn add_compound_box_sphere(
        &mut self,
        pos: Vec3,
        box_half: Vec3,
        sphere_offset: Vec3,
        sphere_radius: f32,
        mass: f32,
    ) -> u32 {
        let body = self.world.add_body(BodyState::dynamic(
            pos,
            Quat::IDENTITY,
            mass,
            Vec3::splat(0.15 * mass),
        ));
        let box_vol = 8.0 * box_half.x * box_half.y * box_half.z;
        let sph_vol = (4.0 / 3.0) * std::f32::consts::PI * sphere_radius.powi(3);
        let dens = mass / (box_vol + sph_vol).max(1e-8);
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 1,
            local_pos: [0.0; 3],
            density: dens,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [box_half.x, box_half.y, box_half.z, box_half.max_element()],
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Sphere as u32,
            material: 0,
            generation: 1,
            local_pos: sphere_offset.to_array(),
            density: dens,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [sphere_radius, 0.0, 0.0, sphere_radius],
        });
        body
    }

    pub fn add_ball_joint(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> u32 {
        self.world.add_joint(Joint {
            body_a: a,
            body_b: b,
            joint_type: JointType::Ball as u32,
            flags: 0,
            local_anchor_a: anchor_a.to_array(),
            stiffness: 1.0e5,
            local_anchor_b: anchor_b.to_array(),
            damping: 0.0,
            axis_a: [0.0, 1.0, 0.0],
            limit_min: 0.0,
            axis_b: [0.0, 1.0, 0.0],
            limit_max: 0.0,
            c0: [0.0; 3],
            _pad_c0: 0.0,
            lambda: [0.0; 6],
            k: [1.0e4; 6],
        })
    }

    pub fn add_fixed_joint(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> u32 {
        self.world.add_joint(Joint {
            body_a: a,
            body_b: b,
            joint_type: JointType::Fixed as u32,
            flags: 0,
            local_anchor_a: anchor_a.to_array(),
            stiffness: 1.0e5,
            local_anchor_b: anchor_b.to_array(),
            damping: 0.0,
            axis_a: [0.0, 1.0, 0.0],
            limit_min: 0.0,
            axis_b: [0.0, 1.0, 0.0],
            limit_max: 0.0,
            c0: [0.0; 3],
            _pad_c0: 0.0,
            lambda: [0.0; 6],
            k: [1.0e5; 6],
        })
    }

    pub fn add_slider(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> u32 {
        self.world.add_joint(Joint {
            body_a: a,
            body_b: b,
            joint_type: JointType::Slider as u32,
            flags: 0,
            local_anchor_a: anchor_a.to_array(),
            stiffness: 1.0e5,
            local_anchor_b: anchor_b.to_array(),
            damping: 0.0,
            axis_a: axis.to_array(),
            limit_min: -1.0,
            axis_b: axis.to_array(),
            limit_max: 1.0,
            c0: [0.0; 3],
            _pad_c0: 0.0,
            lambda: [0.0; 6],
            k: [1.0e4; 6],
        })
    }

    pub fn add_d6(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> u32 {
        self.world.add_joint(Joint {
            body_a: a,
            body_b: b,
            joint_type: JointType::D6 as u32,
            flags: 0,
            local_anchor_a: anchor_a.to_array(),
            stiffness: 1.0e5,
            local_anchor_b: anchor_b.to_array(),
            damping: 0.0,
            axis_a: axis.to_array(),
            limit_min: -std::f32::consts::PI,
            axis_b: axis.to_array(),
            limit_max: std::f32::consts::PI,
            c0: [0.0; 3],
            _pad_c0: 0.0,
            lambda: [0.0; 6],
            k: [1.0e4; 6],
        })
    }

    pub fn add_hinge(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> u32 {
        self.world.add_joint(Joint {
            body_a: a,
            body_b: b,
            joint_type: JointType::Hinge as u32,
            flags: 0,
            local_anchor_a: anchor_a.to_array(),
            stiffness: 5.0e3,
            local_anchor_b: anchor_b.to_array(),
            damping: 0.0,
            axis_a: axis.to_array(),
            limit_min: -std::f32::consts::PI,
            axis_b: axis.to_array(),
            limit_max: std::f32::consts::PI,
            c0: [0.0; 3],
            _pad_c0: 0.0,
            lambda: [0.0; 6],
            k: [1.0e4; 6],
        })
    }

    pub fn pile_spheres(&mut self, n: u32, spacing: f32) {
        let side = (n as f32).cbrt().ceil() as u32;
        let mut placed = 0u32;
        for y in 0..side {
            for z in 0..side {
                for x in 0..side {
                    if placed >= n {
                        return;
                    }
                    let pos = Vec3::new(
                        (x as f32 - side as f32 * 0.5) * spacing,
                        1.0 + y as f32 * spacing,
                        (z as f32 - side as f32 * 0.5) * spacing,
                    );
                    self.add_sphere(pos, spacing * 0.45, 1.0);
                    placed += 1;
                }
            }
        }
    }

    pub fn rope_capsules(&mut self, segments: u32, start: Vec3, seg_len: f32) {
        let half = seg_len * 0.35;
        let radius = (seg_len * 0.12).min(0.05);
        let mut prev = None;
        for i in 0..segments {
            let pos = start + Vec3::new(0.0, -(i as f32) * seg_len, 0.0);
            let id = self.add_capsule(pos, radius, half, 0.2);
            if let Some(p) = prev {
                self.add_ball_joint(
                    p,
                    id,
                    Vec3::new(0.0, -half, 0.0),
                    Vec3::new(0.0, half, 0.0),
                );
            }
            prev = Some(id);
        }
    }

    pub fn sleeping_field(&mut self, n: u32, origin: Vec3, spacing: f32) {
        for i in 0..n {
            let x = (i % 600) as f32 * spacing;
            let z = (i / 600) as f32 * spacing;
            let id = self.add_sphere(origin + Vec3::new(x, 0.5, z), spacing * 0.4, 1.0);
            self.world.force_sleep(id);
        }
    }
}
