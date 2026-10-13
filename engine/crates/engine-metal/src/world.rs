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
    manifolds: Buffer,
    joints: Buffer,
    params_buf: Buffer,
    /// Ping-pong so the CPU can prepare step N+1 while the GPU drains step N
    /// (three-avbd wallMs). Index = `host_parity`.
    counters: [Buffer; 2],
    commands: Buffer,
    aabbs: Buffer,
    candidates: Buffer,
    awake_list: Buffer,
    body_color: Buffer,
    constraint_index: Buffer,
    constraint_ids: Buffer,
    color_lists: Buffer,
    color_offsets: Buffer,
    color_counts: [Buffer; 2],
    hull_verts: Buffer,
    events: Buffer,
    ccd_flags: Buffer,
    world_obbs: Buffer,
    ref_poses: Buffer,
    scratch: [Buffer; 2],
    /// Prefilled 0..63; each color/round dispatch binds a 4-byte offset (three-avbd pass const).
    pass_table: Buffer,
    /// Metal MTLDispatchThreadgroupsIndirectArguments × max_colors.
    indirect_args: Buffer,

    body_count: u32,
    collider_count: u32,
    joint_count: u32,
    /// Island-gate BFS rounds (8 for stacks; 2 for random piles so one rocker cannot pin all).
    sleep_island_rounds: u32,
    material_count: u32,
    command_count: u32,

    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    steps_done: u32,
    /// 0 → current contacts in [0, max); 1 → current in [max, 2max). Prev is the other half.
    contacts_parity: u32,
    /// Which counters/scratch/color_counts slot the next encode uses.
    host_parity: u32,
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
        // Uninitialized prevVel made frame-0 adaptive warm-start explode piles.
        unsafe {
            let n = capacity.max_bodies as usize * std::mem::size_of::<BodyState>();
            std::ptr::write_bytes(bodies_prev.contents() as *mut u8, 0, n);
        }
        let colliders = alloc::<Collider>(d, capacity.max_colliders);
        let materials = alloc::<Material>(d, 64);
        // [0..max_contacts) = current; [max_contacts..2*max) = previous frame for warm-start.
        let contacts = alloc::<Contact>(d, capacity.max_contacts.saturating_mul(2));
        // One manifold per broadphase pair (≤ max_candidates).
        let manifolds = alloc::<Manifold>(d, capacity.max_candidates);
        let joints = alloc::<Joint>(d, capacity.max_joints);
        let params_buf = alloc::<AvbdParams>(d, 1);
        let counters = [alloc::<StepCounters>(d, 1), alloc::<StepCounters>(d, 1)];
        let commands = alloc::<GpuCommand>(d, capacity.max_commands);
        let aabbs = alloc::<AabbGpu>(d, capacity.max_bodies + capacity.max_colliders);
        let candidates = alloc::<CandidatePair>(d, capacity.max_candidates);
        let awake_list = alloc_u32(d, capacity.max_bodies);
        let body_color = alloc_u32(d, capacity.max_bodies);
        let constraint_index = alloc_u32(d, capacity.max_bodies + 1);
        // Grid hash heads reuse [0..GRID_HASH_SIZE); also contact warm-start hash, then CSR.
        let constraint_ids = alloc_u32(
            d,
            (2 * (capacity.max_contacts + capacity.max_joints))
                .max(32768)
                .next_power_of_two(),
        );
        let color_lists = alloc_u32(d, capacity.max_bodies * capacity.max_colors);
        let color_offsets = alloc_u32(d, capacity.max_colors);
        let color_counts = [
            alloc_u32(d, capacity.max_colors),
            alloc_u32(d, capacity.max_colors),
        ];
        let hull_verts = alloc::<[f32; 4]>(d, capacity.max_hull_vertices);
        let events = alloc_u32(d, 65536);
        let ccd_flags = alloc_u32(d, capacity.max_bodies);
        // WorldObbGpu = 5 × float4
        let world_obbs = alloc::<[f32; 20]>(d, capacity.max_colliders);
        // RefPoseGpu = 2 × float4 (three-avbd reuseContacts updateRefs).
        let ref_poses = alloc::<[f32; 8]>(d, capacity.max_bodies);
        unsafe {
            let n = capacity.max_bodies as usize * 8;
            std::ptr::write_bytes(ref_poses.contents() as *mut u8, 0, n * 4);
        }
        let scratch = [alloc_u32(d, 64), alloc_u32(d, 64)];
        // 96 slots: 0..47 identity (colour rounds / misc), 48..67 = 2^0..2^19 for CSR scan
        // (must cover max_bodies=524288; the old 16-slot table broke adjacency above 65k).
        let pass_table = alloc_u32(d, 96);
        {
            let t = unsafe { std::slice::from_raw_parts_mut(pass_table.contents() as *mut u32, 96) };
            for (i, v) in t.iter_mut().enumerate() {
                *v = i as u32;
            }
            for k in 0..20u32 {
                // CSR ping-pong scan: offset in low bits, ping=(k%2) in bit 31.
                t[48 + k as usize] = (1u32 << k) | ((k & 1) << 31);
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
            *(counters[0].contents() as *mut StepCounters) = StepCounters::default();
            *(counters[1].contents() as *mut StepCounters) = StepCounters::default();
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
            manifolds,
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
            world_obbs,
            ref_poses,
            scratch,
            pass_table,
            indirect_args,
            body_count: 0,
            collider_count: 0,
            joint_count: 0,
            sleep_island_rounds: 8,
            material_count: 1,
            command_count: 0,
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            steps_done: 0,
            contacts_parity: 0,
            host_parity: 0,
            chip,
        })
    }

    pub fn set_ccd_enabled(&mut self, on: bool) {
        self.enable_ccd = on;
    }

    pub fn set_sleep_island_rounds(&mut self, rounds: u32) {
        // 0 = skip island gate (independent sleep; piles). Stacks/columns keep ≥1.
        self.sleep_island_rounds = rounds.min(16);
    }

    pub fn set_default_friction(&mut self, friction: f32) {
        let mats = unsafe {
            std::slice::from_raw_parts_mut(self.materials.contents() as *mut Material, 64)
        };
        mats[0].friction = friction;
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
        let prev = unsafe {
            std::slice::from_raw_parts_mut(
                self.bodies_prev.contents() as *mut BodyState,
                self.capacity.max_bodies as usize,
            )
        };
        bodies[id as usize] = state;
        prev[id as usize] = state; // x⁻ = x until the first predictOne
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
        let prev = unsafe {
            std::slice::from_raw_parts_mut(
                self.bodies_prev.contents() as *mut BodyState,
                self.capacity.max_bodies as usize,
            )
        };
        bodies[id as usize] = state;
        prev[id as usize] = state;
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

    fn host_slot(&self) -> usize {
        (self.host_parity & 1) as usize
    }

    fn bind_all(&self, encoder: &ComputeCommandEncoderRef) {
        // Binding order must match Slang declaration order in world_step.slang.
        let slot = self.host_slot();
        encoder.set_buffer(0, Some(&self.bodies), 0);
        encoder.set_buffer(1, Some(&self.bodies_prev), 0);
        encoder.set_buffer(2, Some(&self.colliders), 0);
        encoder.set_buffer(3, Some(&self.materials), 0);
        encoder.set_buffer(4, Some(&self.contacts), 0);
        encoder.set_buffer(5, Some(&self.joints), 0);
        encoder.set_buffer(6, Some(&self.params_buf), 0);
        encoder.set_buffer(7, Some(&self.counters[slot]), 0);
        encoder.set_buffer(8, Some(&self.commands), 0);
        encoder.set_buffer(9, Some(&self.aabbs), 0);
        encoder.set_buffer(10, Some(&self.candidates), 0);
        encoder.set_buffer(11, Some(&self.awake_list), 0);
        encoder.set_buffer(12, Some(&self.body_color), 0);
        encoder.set_buffer(13, Some(&self.constraint_index), 0);
        encoder.set_buffer(14, Some(&self.constraint_ids), 0);
        encoder.set_buffer(15, Some(&self.color_lists), 0);
        encoder.set_buffer(16, Some(&self.color_offsets), 0);
        encoder.set_buffer(17, Some(&self.color_counts[slot]), 0);
        encoder.set_buffer(18, Some(&self.hull_verts), 0);
        encoder.set_buffer(19, Some(&self.events), 0);
        encoder.set_buffer(20, Some(&self.ccd_flags), 0);
        encoder.set_buffer(21, Some(&self.scratch[slot]), 0);
        // Default pass uniform = slot 0; dispatch_pass overrides with an offset.
        encoder.set_buffer(22, Some(&self.pass_table), 0);
        encoder.set_buffer(23, Some(&self.indirect_args), 0);
        encoder.set_buffer(24, Some(&self.world_obbs), 0);
        encoder.set_buffer(25, Some(&self.manifolds), 0);
        encoder.set_buffer(26, Some(&self.ref_poses), 0);
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
        let s = unsafe {
            std::slice::from_raw_parts_mut(self.scratch[self.host_slot()].contents() as *mut u32, 64)
        };
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
        // s[7] = previous contact count; s[8] = offset to prev half; s[11] = current half.
        let max_c = self.capacity.max_contacts;
        s[7] = self.last_counters.contact_count.min(max_c);
        let cur = if self.contacts_parity == 0 { 0 } else { max_c };
        let prev = if self.contacts_parity == 0 { max_c } else { 0 };
        s[8] = prev;
        s[11] = cur;
        // s[9] = contact warm-start hash table size (power of two ≤ constraint_ids).
        let ids_len = (2 * (self.capacity.max_contacts + self.capacity.max_joints))
            .max(32768)
            .next_power_of_two();
        let prev = s[7].max(256);
        s[9] = (prev.saturating_mul(4).max(1024) as u32)
            .next_power_of_two()
            .min(ids_len);
        // s[10] = max manifolds (== max_candidates; one manifold per BP pair).
        s[10] = self.capacity.max_candidates;
        // s[12] = last CSR ping (ceil(log2(n))-1) & 1 for csr_prefix_finalize.
        let mut log = 0u32;
        while (1u32 << log) < self.body_count.max(1) && log < 20 {
            log += 1;
        }
        s[12] = if log == 0 { 0 } else { (log - 1) & 1 };
        s[13] = 0; // any sleep candidate (set by extract)
        s[17] = self.steps_done; // step index for reuseContacts generation stamps
        s[18] = 0; // reuseContacts hit count (narrowphase)
        s[19] = 0; // reuse hash miss
        s[20] = 0; // reuse generation gate reject
        s[21] = 0; // update_refs noRef
        s[22] = 0; // update_refs linMove
        s[23] = 0; // update_refs angMove
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
            manifold_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        if !clear_transient {
            c = self.last_counters;
            c.body_count = self.body_count;
            c.collider_count = self.collider_count;
            c.joint_count = self.joint_count;
        }
        unsafe {
            *(self.counters[self.host_slot()].contents() as *mut StepCounters) = c;
            *(self.params_buf.contents() as *mut AvbdParams) = self.params;
        }
    }

    pub fn step_gpu(&mut self) -> Result<StepCounters, ExecError> {
        self.step_gpu_inner(true)
    }

    /// Encode+commit; when `wait` is false, flip host buffers and return without draining.
    pub fn step_gpu_ex(&mut self, wait: bool) -> Result<StepCounters, ExecError> {
        self.step_gpu_inner(wait)
    }

    /// three-avbd cellSize ≈ 2·maxSmallRadius (exclude planes / oversized smash balls).
    fn refresh_grid_cell_size(&mut self) {
        let cols = unsafe {
            std::slice::from_raw_parts(
                self.colliders.contents() as *const Collider,
                self.collider_count as usize,
            )
        };
        let bodies = unsafe {
            std::slice::from_raw_parts(
                self.bodies.contents() as *const BodyState,
                self.body_count as usize,
            )
        };
        let mut max_r = 0.25f32;
        for c in cols {
            if c.shape == ShapeType::Plane as u32 {
                continue;
            }
            let body = c.body as usize;
            if body >= bodies.len() || bodies[body].inv_mass <= 0.0 {
                continue;
            }
            let r = if c.shape == ShapeType::Sphere as u32 || c.shape == ShapeType::Capsule as u32 {
                c.params[0]
            } else if c.params[3] > 0.0 {
                c.params[3]
            } else {
                (c.params[0] * c.params[0] + c.params[1] * c.params[1] + c.params[2] * c.params[2])
                    .sqrt()
            };
            // Smash balls / large dynamics stay out of maxSmall (three-avbd largestSmall).
            if r > 2.0 {
                continue;
            }
            max_r = max_r.max(r);
        }
        // three-avbd: cellSize = 2·maxSmallRadius so findPairs' centre±1 (27 cells)
        // covers every overlap. A smaller mult (e.g. 1.25 for Ring smash) leaves
        // centres up to ~1.6 cells apart → missed pairs → tunnels. Override only for
        // deliberate A/B via GRID_CELL_MULT.
        let mult = std::env::var("GRID_CELL_MULT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0);
        self.params.grid_cell_size = (mult * max_r * 1.0001).max(0.5);
    }

    fn step_gpu_inner(&mut self, wait: bool) -> Result<StepCounters, ExecError> {
        let wall_t0 = std::time::Instant::now();
        let cpu_t0 = std::time::Instant::now();
        // Fully asleep world: skip the GPU step (sleeper BP was ~180 ms on 100k columns).
        // Require a prior step so the initial awake_count==0 default does not no-op frame 0.
        if self.steps_done > 0
            && self.last_counters.awake_count == 0
            && self.command_count == 0
            && self.body_count > 0
        {
            let gpu_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;
            self.last_timings = PhaseTimingsMs {
                total_gpu: gpu_ms,
                cpu_submit: 0.0,
                ..Default::default()
            };
            self.steps_done = self.steps_done.saturating_add(1);
            return Ok(self.last_counters);
        }
        self.refresh_grid_cell_size();
        self.sync_counters_host_to_gpu(true);
        self.write_scratch();
        // Avoid O(max_colors) host clears on huge capacities — only touch used slots.
        {
            let n = self.capacity.max_colors.min(64) as usize;
            let slot = self.host_slot();
            let cc = unsafe {
                std::slice::from_raw_parts_mut(self.color_counts[slot].contents() as *mut u32, n)
            };
            for v in cc.iter_mut() {
                *v = 0;
            }
        }

        let cb = self.device.queue.new_command_buffer();
        // Contact warm-start: ping-pong halves of gContacts (see contacts_parity). No blit.
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
        // Do NOT use n_cols*32 — that over-dispatched NP ~9× on Brick Ring 9k (292k vs 33k).
        let prev_contacts = self.last_counters.contact_count.max(256) as u64;
        let prev_cand = self.last_counters.candidate_count.max(256) as u64;
        let narrow_n = (self.capacity.max_candidates as u64)
            .min(
                prev_cand
                    .saturating_add(prev_cand / 2)
                    .saturating_add(2048)
                    .max(n_cols.saturating_mul(4))
                    .max(
                        prev_contacts
                            .saturating_add(prev_contacts / 4)
                            .saturating_add(1024),
                    ),
            )
            .max(1);
        // Size to last count + 25% headroom (2× over-dispatch wasted bandwidth on smash peaks).
        let contact_dispatch = (self.capacity.max_contacts as u64)
            .min(
                prev_contacts
                    .saturating_add(prev_contacts / 4)
                    .saturating_add(1024)
                    .max(n_cols.saturating_mul(4)),
            )
            .max(1);
        // Manifolds ≤ candidates; size like narrowphase so the first smash frame is covered.
        let prev_manifolds = self.last_counters.manifold_count.max(256) as u64;
        let manifold_dispatch = (self.capacity.max_candidates as u64)
            .min(narrow_n.max(
                prev_manifolds
                    .saturating_add(prev_manifolds / 4)
                    .saturating_add(1024),
            ))
            .max(1);

        if self.command_count > 0 {
            self.dispatch(&encoder, "apply_commands", self.command_count as u64);
        }
        let dirty = unsafe { *(self.scratch[self.host_slot()].contents() as *const u32).add(6) };
        if dirty != 0 || self.command_count > 0 {
            self.dispatch(&encoder, "recompute_inertia", n_bodies);
        }
        // three-avbd order: collide at x⁻ → topology → warm-start bodies → solve.
        // C0 must be sampled before predict; NP-after-predict made impact C0 huge
        // negative and blew spaced Stack.
        self.dispatch(&encoder, "build_aabbs", n_cols);
        // O(1) joint-pair filter for broadphase (was O(joints) per candidate).
        // Ring/Wall smash have zero joints — skip the 64k clear entirely.
        if self.joint_count > 0 {
            self.dispatch(&encoder, "joint_hash_clear", 65536);
            self.dispatch(&encoder, "joint_hash_insert", n_joints);
        }
        self.dispatch(&encoder, "broadphase_clear", 131072);
        self.dispatch(&encoder, "broadphase_insert", n_cols);
        self.dispatch(&encoder, "broadphase_query", n_cols);
        // Pair hash before NP when reuseContacts is on (three-avbd order); otherwise
        // after NP is enough for persist-only warm-start.
        let hash_n = {
            let s = unsafe {
                std::slice::from_raw_parts(self.scratch[self.host_slot()].contents() as *const u32, 64)
            };
            s[9].max(1024) as u64
        };
        let prev_n = prev_contacts.min(self.capacity.max_contacts as u64).max(1);
        let reuse = (self.params.flags & engine_schema::FLAG_REUSE_CONTACTS) != 0;
        // Settled piles: update_refs over all bodies was part of the ~9 ms floor with
        // ~20 awake / few contacts. Refs only matter for reuse; skip when nearly asleep.
        let mostly_asleep = self.last_counters.awake_count > 0
            && self.last_counters.awake_count <= 64
            && self.last_counters.contact_count <= 256;
        if reuse {
            if !mostly_asleep {
                self.dispatch(&encoder, "update_refs", n_bodies);
            }
            self.dispatch(&encoder, "contact_hash_clear", hash_n);
            self.dispatch(&encoder, "contact_hash_insert", prev_n);
        }
        self.dispatch(&encoder, "narrowphase", narrow_n);
        if !reuse {
            self.dispatch(&encoder, "contact_hash_clear", hash_n);
            self.dispatch(&encoder, "contact_hash_insert", prev_n);
        }
        self.dispatch(&encoder, "persist_contacts", contact_dispatch);
        if self.joint_count > 0 {
            self.dispatch(&encoder, "warmstart_joints", n_joints);
        }

        // Awake list before CSR sort/colour (flags only — independent of contacts).
        self.dispatch(&encoder, "build_awake_list", n_bodies);

        // CSR adjacency (three-avbd degree → scan → scatter → sort). Overwrites grid hash.
        // Degree/scatter walk manifolds (Stage 8b), not individual contact points.
        self.dispatch(&encoder, "csr_clear", n_bodies + 1);
        self.dispatch(&encoder, "csr_degree_contacts", manifold_dispatch);
        if self.joint_count > 0 {
            self.dispatch(&encoder, "csr_degree_joints", n_joints);
        }
        if self.body_count <= 512 {
            self.dispatch(&encoder, "csr_prefix", 1);
        } else {
            // Parallel Hillis–Steele exclusive scan (ping-pong; no swap dispatch).
            self.dispatch(&encoder, "csr_prefix_init", n_bodies);
            let mut log = 0u32;
            // Offsets 2^0 .. 2^19 cover WorldCapacity::scale max_bodies (524288).
            while (1u32 << log) < self.body_count && log < 20 {
                self.dispatch_pass(&encoder, "csr_prefix_scan", n_bodies, 48 + log);
                log += 1;
            }
            self.dispatch(&encoder, "csr_prefix_finalize", n_bodies + 1);
        }
        self.dispatch(&encoder, "csr_scatter_contacts", manifold_dispatch);
        if self.joint_count > 0 {
            self.dispatch(&encoder, "csr_scatter_joints", n_joints);
        }
        self.dispatch(&encoder, "csr_sort", awake_dispatch);

        self.dispatch(&encoder, "init_colors", awake_dispatch);

        // Colour rounds: distinct pass_table offsets (NOT mid-encoder scratch rewrites).
        // Large smash scenes: hard-cap 10 JP rounds (colouring was visible in the non-solve base).
        // JP rounds: one past last colour is enough once the palette is stable.
        let color_cap = if self.body_count > 8192 { 8 } else { 12 };
        let color_rounds = (self.last_counters.color_count.saturating_add(1))
            .max(4)
            .min(color_cap);
        for round in 0..color_rounds {
            self.dispatch_pass(&encoder, "color_bodies", awake_dispatch, round % 16);
        }
        self.dispatch(&encoder, "pack_color_lists", awake_dispatch);
        self.dispatch(&encoder, "write_color_args", 64);

        // Body warm-start AFTER NP (three-avbd warmStartBodies): stash inertial, advance pose.
        self.dispatch(&encoder, "predict_bodies", n_bodies);

        // three-avbd argsColors: indirect per colour (0 threadgroups ⇒ free). Same encoder so
        // Metal tracks the write→dispatch dependency without a CPU mid-step sync.
        // Colored Jacobi is enough once C0 is sampled at x⁻ (solve_primal_one kept for probes).
        // Dual stays per-contact (occupancy); CSR/island use manifolds.
        let dual_n = contact_dispatch.max(n_joints);
        // Live colours + slack so a growing palette is not truncated mid-smash.
        // Live colours + 1 slack (was +4; empty colour indirects still cost encoder time).
        let max_c = self
            .last_counters
            .color_count
            .saturating_add(1)
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
        // Island sleep gate: a body may sleep only if every dynamic contact neighbour is
        // also a candidate (or already asleep). Skipping this lets upper boxes freeze while
        // the stack below settles — columns tore apart and injected energy. Always run a
        // fixed handful of rounds (accuracy); manifolds cut the fan-out vs per-contact.
        if self.last_counters.manifold_count > 0 && self.sleep_island_rounds > 0 {
            let rounds = self.sleep_island_rounds.min(16);
            for _ in 0..rounds {
                self.dispatch(&encoder, "sleep_island_gate", manifold_dispatch);
            }
        }
        self.dispatch(&encoder, "sleep_island_commit", n_bodies);

        if self.enable_ccd {
            self.dispatch(&encoder, "ccd_classify", n_bodies);
            self.dispatch(&encoder, "ccd_sweep", n_cols);
        }

        encoder.end_encoding();
        let cpu_submit = cpu_t0.elapsed().as_secs_f64() * 1000.0;
        let slot = self.host_slot();
        cb.commit();
        self.contacts_parity ^= 1;
        self.host_parity ^= 1;
        self.steps_done = self.steps_done.saturating_add(1);
        self.command_count = 0;
        if !wait {
            self.last_timings = PhaseTimingsMs {
                total_gpu: wall_t0.elapsed().as_secs_f64() * 1000.0,
                cpu_submit,
                ..Default::default()
            };
            let _ = PHASE_GRAPH;
            return Ok(self.last_counters);
        }
        cb.wait_until_completed();
        let gpu_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;
        let counters = unsafe { *(self.counters[slot].contents() as *const StepCounters) };
        self.last_counters = counters;
        self.last_timings = PhaseTimingsMs {
            total_gpu: gpu_ms,
            cpu_submit,
            solve: gpu_ms * 0.5,
            ..Default::default()
        };
        if counters.overflow_flags != 0 {
            return Err(ExecError::Overflow(counters.overflow_flags));
        }
        let _ = PHASE_GRAPH;
        Ok(counters)
    }

    /// Drain the queue and refresh `last_counters` from the slot that just finished
    /// (opposite of the next encode slot after an odd number of `step_gpu_ex(false)`).
    pub fn wait_gpu(&mut self) {
        let cb = self.device.queue.new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
        // Last completed step wrote slot = host_parity ^ 1 (we flipped after commit).
        let done = ((self.host_parity ^ 1) & 1) as usize;
        let counters = unsafe { *(self.counters[done].contents() as *const StepCounters) };
        self.last_counters = counters;
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
        // After step, parity has flipped — last frame's current is now the prev half.
        let base = if self.contacts_parity == 0 {
            self.capacity.max_contacts as usize
        } else {
            0
        };
        unsafe {
            let ptr = (self.contacts.contents() as *const Contact).add(base);
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

    /// Readback for diagnostics only.
    pub fn colliders_ptr(&self) -> *const Collider {
        self.colliders.contents() as *const Collider
    }

    /// Scratch atomics from the last completed step (slot = host_parity ^ 1).
    pub fn scratch_u32(&self, index: usize) -> u32 {
        let done = ((self.host_parity ^ 1) & 1) as usize;
        unsafe {
            *(self.scratch[done].contents() as *const u32).add(index)
        }
    }

    /// Per-body JP colour ids (diagnostics).
    pub fn download_body_colors(&self) -> Vec<u32> {
        unsafe {
            std::slice::from_raw_parts(
                self.body_color.contents() as *const u32,
                self.body_count as usize,
            )
            .to_vec()
        }
    }

    /// Manifold records for the last step (diagnostics).
    pub fn download_manifolds(&self) -> Vec<Manifold> {
        let n = self.last_counters.manifold_count as usize;
        if n == 0 {
            return Vec::new();
        }
        unsafe {
            std::slice::from_raw_parts(self.manifolds.contents() as *const Manifold, n).to_vec()
        }
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
            generation: 0,
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, y],
        });
        body
    }

    /// Static box ground (three-avbd `ground(extent)`). Top face at `top_y`.
    /// `half_xz` is the horizontal half-extent (three-avbd: max(100, 2·extent)).
    /// Thickness is deep so Jacobi soft contacts can't tunnel through a thin slab.
    pub fn add_ground_box(&mut self, top_y: f32) -> u32 {
        self.add_ground_box_extent(top_y, 200.0)
    }

    pub fn add_ground_box_extent(&mut self, top_y: f32, half_xz: f32) -> u32 {
        // Top face matches three-avbd (y=0.5). Thickness is deeper than their size.z=1
        // slab: our Jacobi soft contacts otherwise tunnel through a thin floor at scale.
        let half_h = 25.0;
        let hx = half_xz.max(50.0);
        let body = self.world.add_body(BodyState {
            position: [0.0, top_y - half_h, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [hx, half_h, hx, hx.max(half_h)],
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
            generation: 0,
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
            generation: 0,
            local_pos: [0.0; 3],
            density: mass / (8.0 * half.x * half.y * half.z),
            local_rot: [0.0, 0.0, 0.0, 1.0],
            // params.w = bounding-sphere radius (three-avbd length(half)), used by BP sphere cull.
            params: [half.x, half.y, half.z, half.length()],
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
            generation: 0,
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
            generation: 0,
            local_pos: [0.0; 3],
            density: dens,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [box_half.x, box_half.y, box_half.z, box_half.length()],
        });
        self.world.add_collider(Collider {
            body,
            shape: ShapeType::Sphere as u32,
            material: 0,
            generation: 0,
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
