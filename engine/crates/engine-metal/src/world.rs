use crate::device::MetalDevice;
use crate::pipelines::Pipelines;
use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend, PHASE_GRAPH};
use engine_schema::*;
use engine_vehicle::{
    FunctionalState, Ray, RayHit, VehicleBackend, VehicleCapacity, VehicleDesc, VehicleHandle, VehicleHost,
    VehicleInput, VehicleState, WheelDesc,
};
use glam::{Quat, Vec3};
use metal::*;
use std::mem;

/// Metal `[[buffer(N)]]` of `gVehicleData` in vehicle_step.slang (declaration order
/// after world_step.slang). Shares the slot with the compact solver's points, so
/// vehicle dispatches rebind it; `vehicle_kernels_bind_vehicle_data_at_27` guards it.
pub const VEHICLE_DATA_SLOT: u64 = 27;

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
    solver_points: Buffer,
    compact_contacts: bool,
    scratch: [Buffer; 2],
    /// Prefilled 0..63; each color/round dispatch binds a 4-byte offset (three-avbd pass const).
    pass_table: Buffer,
    /// Metal MTLDispatchThreadgroupsIndirectArguments × max_colors.
    indirect_args: Buffer,
    /// gVehicleData (float4 records, see slang/physics/vehicle_layout.slang).
    vehicle_data: Buffer,
    vehicle_slots: usize,
    vehicles: Option<VehicleHost>,

    body_count: u32,
    collider_count: u32,
    joint_count: u32,
    /// Released joint slots, reused by `add_joint` (three-avbd releaseJoints).
    free_joints: Vec<u32>,
    /// Per-body bounding radius of its largest dynamic collider (torque arm default).
    body_radius: Vec<f32>,
    /// Event ring read cursors (`poll_joint_breaks`), one per scratch buffer:
    /// the step alternates between the two.
    events_seen: [u32; 2],
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
    /// Convex SAT work list (Slang gConvexWork). Metal assigns it buffer 27,
    /// the slot compact solves use for solver points, so convex dispatches
    /// rebind 27 around themselves.
    convex_work: Buffer,
    /// SAT separating-feature cache (Slang gSatCache, Metal buffer 28 in the
    /// SAT entries; nothing else uses 28). Power of two >= max(1024, candidates).
    sat_cache: Buffer,
    /// Host copy of the hull arena uploaded to `hull_verts`.
    pub hull_arena: engine_schema::convex::HullArena,
    /// Any convex hull collider: schedules the convex_* narrowphase entries.
    has_convex: bool,
    /// Hierarchical broadphase size levels (0 = uniform grid).
    grid_levels: u32,
    /// (collider count, base cell, levels): recomputed only when colliders change.
    grid_cache: Option<(u32, f32, u32)>,
    /// Any collider carries a collision filter (scratch s[30]).
    has_filters: bool,
    /// The last step's manifolds still describe the scene (whole-step reuse).
    manifolds_valid: bool,
    /// Next step reuses every previous manifold (cleared after use).
    reuse_next: bool,
}

impl MetalWorld {
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        Self::new_with_vehicles(capacity, VehicleCapacity::default())
    }

    pub fn new_with_vehicles(mut capacity: WorldCapacity, vehicle_capacity: VehicleCapacity) -> anyhow::Result<Self> {
        // One tyre joint per vehicle wheel slot rides on top of the user joints.
        capacity.max_joints += vehicle_capacity.max_wheels;
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
        let aabb_tail = capacity.max_bodies + capacity.max_colliders;
        // Collider AABBs, then a 4-entry wheel-row block per vehicle and a
        // 4-entry ground block per wheel slot (two-body tyre rows).
        let aabbs = alloc::<AabbGpu>(d, aabb_tail + 4 * (vehicle_capacity.max_vehicles + vehicle_capacity.max_wheels));
        let vehicle_data = alloc::<[u32; 4]>(d, vehicle_capacity.layout().total_slots);
        unsafe {
            std::ptr::write_bytes(vehicle_data.contents() as *mut u8, 0, vehicle_capacity.bytes() as usize);
        }
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
        let color_lists = alloc_u32(d, capacity.max_bodies * (capacity.max_colors + 2));
        unsafe {
            let history = (color_lists.contents() as *mut u32)
                .add((capacity.max_bodies * capacity.max_colors.min(32)) as usize);
            std::ptr::write_bytes(history, 0, capacity.max_bodies as usize);
        }
        let color_offsets = alloc_u32(
            d,
            capacity.max_colors * capacity.max_bodies.div_ceil(64).max(1),
        );
        // Colour counts (0..63) and per-colour max CSR degrees (64..127).
        let color_counts = [
            alloc_u32(d, capacity.max_colors.max(128)),
            alloc_u32(d, capacity.max_colors.max(128)),
        ];
        let hull_verts = alloc::<[f32; 4]>(d, capacity.max_hull_vertices);
        // Joint-pair hash: three words per slot (joint index, body pair).
        let events = alloc_u32(d, engine_schema::joint_hash_size(capacity.max_joints) * 3);
        let ccd_flags = alloc_u32(d, capacity.max_bodies);
        // WorldObbGpu = 5 × float4
        let world_obbs = alloc::<[f32; 20]>(d, capacity.max_colliders);
        // RefPoseGpu = 2 × float4 (three-avbd reuseContacts updateRefs).
        let ref_poses = alloc::<[f32; 8]>(d, capacity.max_bodies);
        let compact_contacts = std::env::var("ENGINE_COMPACT_CONTACTS").as_deref() == Ok("1");
        anyhow::ensure!(
            capacity.max_candidates < 0x7fff_ffff,
            "compact pair index exceeds 31 bits"
        );
        let solver_points = alloc::<[f32; 16]>(
            d,
            if compact_contacts {
                capacity.max_contacts
            } else {
                1
            },
        );
        unsafe {
            let n = capacity.max_bodies as usize * 8;
            std::ptr::write_bytes(ref_poses.contents() as *mut u8, 0, n * 4);
        }
        // 0..63 host words and per-step counters; 64..79 the uniform grid's large list.
        // Host words, lists, then the event ring and the oversized-static ids
        // (engine-schema SCRATCH_WORDS); the ring cursor starts at zero.
        let scratch = [alloc_u32(d, engine_schema::SCRATCH_WORDS), alloc_u32(d, engine_schema::SCRATCH_WORDS)];
        for b in &scratch {
            unsafe {
                std::ptr::write_bytes(b.contents() as *mut u32, 0, engine_schema::SCRATCH_WORDS as usize);
            }
        }
        // 96 slots: 0..47 identity (colour rounds / misc), 48..67 = 2^0..2^19 for CSR scan
        // (must cover max_bodies=524288; the old 16-slot table broke adjacency above 65k).
        let pass_table = alloc_u32(d, 96);
        {
            let t =
                unsafe { std::slice::from_raw_parts_mut(pass_table.contents() as *mut u32, 96) };
            for (i, v) in t.iter_mut().enumerate() {
                *v = i as u32;
            }
            for k in 0..20u32 {
                // CSR ping-pong scan: offset in low bits, ping=(k%2) in bit 31.
                t[48 + k as usize] = (1u32 << k) | ((k & 1) << 31);
            }
        }
        // Colour slots 0..63, work slots 64..69, convex slots 70..72.
        // Colour slots 0..63, work 64..69, convex 70..72, previous contacts 73.
        let indirect_args = alloc_u32(d, capacity.max_colors.max(77) * 3);
        let convex_work = alloc::<[u32; 8]>(d, capacity.max_candidates);
        let sat_cache = alloc::<[u32; 2]>(d, capacity.max_candidates.max(1024).next_power_of_two());
        // Every hull arena starts with the unit cube reused by box/hull pairs.
        {
            let arena = engine_schema::convex::HullArena::new();
            let records = arena.records();
            assert!(records.len() <= capacity.max_hull_vertices as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    records.as_ptr(),
                    hull_verts.contents() as *mut [u32; 4],
                    records.len(),
                );
            }
        }

        // Default material
        let mats =
            unsafe { std::slice::from_raw_parts_mut(materials.contents() as *mut Material, 64) };
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
            solver_points,
            compact_contacts,
            scratch,
            pass_table,
            indirect_args,
            vehicle_data,
            vehicle_slots: vehicle_capacity.layout().total_slots as usize,
            vehicles: Some(VehicleHost::new(vehicle_capacity, aabb_tail)),
            body_count: 0,
            collider_count: 0,
            joint_count: 0,
            free_joints: Vec::new(),
            body_radius: Vec::new(),
            events_seen: [0; 2],
            sleep_island_rounds: 8,
            material_count: 1,
            command_count: 0,
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            steps_done: 0,
            contacts_parity: 0,
            host_parity: 0,
            chip,
            convex_work,
            sat_cache,
            hull_arena: engine_schema::convex::HullArena::new(),
            has_convex: false,
            grid_levels: 0,
            grid_cache: None,
            has_filters: false,
            manifolds_valid: false,
            reuse_next: false,
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
        self.manifolds_valid = false;
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
        self.has_convex |= col.shape == ShapeType::ConvexHull as u32;
        self.has_filters |= engine_schema::convex::CollisionFilter::is_set(&col);
        self.manifolds_valid = false;
        if let Some(r) = collider_reach(&col) {
            let b = col.body as usize;
            if self.body_radius.len() <= b {
                self.body_radius.resize(b + 1, 0.0);
            }
            self.body_radius[b] = self.body_radius[b].max(r);
        }
        id
    }

    /// Replace a collider (e.g. swap a fractured chunk's hull or its filter).
    /// Release the old hull with `release_hull` once nothing uses it.
    pub fn set_collider(&mut self, id: u32, col: Collider) {
        assert!(id < self.collider_count, "collider {id} out of range");
        let cols = unsafe {
            std::slice::from_raw_parts_mut(
                self.colliders.contents() as *mut Collider,
                self.capacity.max_colliders as usize,
            )
        };
        cols[id as usize] = col;
        self.has_convex |= col.shape == ShapeType::ConvexHull as u32;
        self.has_filters = cols[..self.collider_count as usize]
            .iter()
            .any(engine_schema::convex::CollisionFilter::is_set);
        self.grid_cache = None;
        self.manifolds_valid = false;
    }

    /// Drop one arena reference to a hull; its records are reused once free.
    pub fn release_hull(&mut self, base: u32) -> bool {
        self.manifolds_valid = false;
        self.hull_arena.release(base)
    }

    /// Insert one hull into the world's arena (shared when identical), growing
    /// the buffer when needed, and upload only the changed records.
    pub fn insert_hull(&mut self, hull: &engine_schema::convex::ConvexHull) -> u32 {
        let base = self.hull_arena.insert(hull).expect("hull arena range");
        self.sync_hull_arena();
        base
    }

    /// Grow `hull_verts` (power of two, contents preserved) and copy dirty ranges.
    fn sync_hull_arena(&mut self) {
        let needed = self.hull_arena.len() as u32;
        if needed > self.capacity.max_hull_vertices {
            let records = needed.next_power_of_two();
            let grown = alloc::<[u32; 4]>(&self.device.device, records);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.hull_verts.contents() as *const u8,
                    grown.contents() as *mut u8,
                    self.capacity.max_hull_vertices as usize * 16,
                );
            }
            self.hull_verts = grown;
            self.capacity.max_hull_vertices = records;
        }
        let dirty = self.hull_arena.take_dirty();
        let records = self.hull_arena.records();
        for r in dirty {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    records[r.start as usize..r.end as usize].as_ptr(),
                    (self.hull_verts.contents() as *mut [u32; 4]).add(r.start as usize),
                    (r.end - r.start) as usize,
                );
            }
        }
        self.manifolds_valid = false;
    }

    /// Replace the hull arena and upload it whole (growing the buffer if needed).
    pub fn set_hulls(&mut self, arena: &engine_schema::convex::HullArena) {
        self.hull_arena = arena.clone();
        self.hull_arena.mark_all_dirty();
        self.sync_hull_arena();
    }


    fn joints_mut(&mut self) -> &mut [Joint] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.joints.contents() as *mut Joint,
                self.capacity.max_joints as usize,
            )
        }
    }

    /// Live and broken joints written so far (`[0, joint_count)`).
    pub fn joints(&self) -> &[Joint] {
        unsafe {
            std::slice::from_raw_parts(
                self.joints.contents() as *const Joint,
                self.joint_count as usize,
            )
        }
    }

    pub fn joint(&self, id: u32) -> Joint {
        self.joints()[id as usize]
    }

    /// Write a raw GPU joint, reusing a released slot when one is free.
    pub fn add_joint(&mut self, joint: Joint) -> u32 {
        let id = match self.free_joints.pop() {
            Some(id) => id,
            None => {
                let id = self.joint_count;
                assert!(id < self.capacity.max_joints, "joint capacity exceeded");
                self.joint_count += 1;
                id
            }
        };
        self.joints_mut()[id as usize] = joint;
        id
    }

    /// Default torque arm (three-avbd `|size_a + size_b|²`) from collider bounds.
    pub fn default_torque_arm(&self, a: u32, b: u32) -> f32 {
        let r = |id: u32| {
            let body = self.body(id);
            if body.inv_mass == 0.0 {
                return 0.0;
            }
            self.body_radius.get(id as usize).copied().unwrap_or(0.0)
        };
        default_torque_arm(r(a), r(b))
    }

    pub fn add_joint_desc(&mut self, desc: &JointDesc) -> u32 {
        let mut j = desc.to_gpu();
        if desc.torque_arm <= 0.0 {
            j.torque_arm = self.default_torque_arm(desc.body_a, desc.body_b);
        }
        self.add_joint(j)
    }

    /// Release a joint: the pair collides again and the slot is reused.
    pub fn remove_joint(&mut self, id: u32) {
        // Released = broken with both ends on body 0 (never a real joint, and a
        // safe index if any kernel reads it).
        let j = &mut self.joints_mut()[id as usize];
        if j.flags & JOINT_FLAG_BROKEN == 0 || j.body_a != j.body_b {
            j.flags |= JOINT_FLAG_BROKEN;
            j.body_a = 0;
            j.body_b = 0;
            self.free_joints.push(id);
        }
    }

    /// Joints the GPU broke since the last call. Their slots are released.
    pub fn drain_broken_joints(&mut self) -> Vec<u32> {
        let broken: Vec<u32> = (0..self.joint_count)
            .filter(|&i| {
                let j = self.joint(i);
                j.is_broken() && j.body_a != j.body_b
            })
            .collect();
        for &id in &broken {
            self.remove_joint(id);
        }
        broken
    }

    /// Update a drive's targets in place (motors, servos).
    pub fn set_joint_drive(&mut self, id: u32, velocity: f32, target: f32) {
        let j = &mut self.joints_mut()[id as usize];
        j.drive_velocity = velocity;
        j.drive_target = target;
    }

    pub fn set_joint(&mut self, id: u32, joint: Joint) {
        self.joints_mut()[id as usize] = joint;
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
        self.manifolds_valid = false;
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
        self.manifolds_valid = false;
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
        // Default pass uniform = slot 0; indirect dispatches override the offset.
        encoder.set_buffer(22, Some(&self.pass_table), 0);
        encoder.set_buffer(23, Some(&self.indirect_args), 0);
        encoder.set_buffer(24, Some(&self.world_obbs), 0);
        encoder.set_buffer(25, Some(&self.manifolds), 0);
        encoder.set_buffer(26, Some(&self.ref_poses), 0);
        encoder.set_buffer(27, Some(&self.solver_points), 0);
    }

    fn dispatch(&self, encoder: &ComputeCommandEncoderRef, name: &str, threads: u64) {
        if name == "write_work_args"
            || name == "write_color_args"
            || name == "write_sleep_args"
            || name == "convex_write_args"
        {
            encoder.set_buffer(23, Some(&self.indirect_args), 0);
        }
        let pso = self.pipelines.get(name).expect("pipeline");
        encoder.set_compute_pipeline_state(pso);
        if name == "csr_prefix_blocks"
            || name == "csr_prefix_block_sums"
            || name == "count_color_tiles"
        {
            // Cooperative scans require all 64 lanes, including padding in the
            // last group. SIMD width (32 on Apple) is not the workgroup width.
            encoder.dispatch_thread_groups(
                MTLSize::new(threads.max(1).div_ceil(64), 1, 1),
                MTLSize::new(64, 1, 1),
            );
            return;
        }
        // Buffers stay bound for the encoder; only swap pipeline + thread counts.
        let w = pso.thread_execution_width();
        let tg = MTLSize::new(w, 1, 1);
        let grid = MTLSize::new(threads.max(1), 1, 1);
        encoder.dispatch_threads(grid, tg);
    }

    fn dispatch_work(
        &self,
        encoder: &ComputeCommandEncoderRef,
        name: &str,
        slot: u32,
        pass_idx: u32,
    ) {
        encoder.set_compute_pipeline_state(self.pipelines.get(name).expect("pipeline"));
        encoder.set_buffer(22, Some(&self.pass_table), pass_idx as u64 * 4);
        encoder.set_buffer(23, None, 0);
        encoder.dispatch_thread_groups_indirect(
            &self.indirect_args,
            slot as u64 * 12,
            MTLSize::new(64, 1, 1),
        );
    }

    /// Vehicle kernels read gVehicleData at buffer 27 (see VEHICLE_DATA_SLOT).
    /// `per_group` dispatches one 64-lane workgroup per item (ray / wheel casts).
    fn dispatch_vehicle(&self, encoder: &ComputeCommandEncoderRef, name: &str, items: u64, per_group: bool) {
        encoder.set_buffer(VEHICLE_DATA_SLOT, Some(&self.vehicle_data), 0);
        if per_group {
            encoder.set_compute_pipeline_state(self.pipelines.get(name).expect("pipeline"));
            encoder.dispatch_thread_groups(MTLSize::new(items.max(1), 1, 1), MTLSize::new(64, 1, 1));
        } else {
            self.dispatch(encoder, name, items);
        }
        encoder.set_buffer(VEHICLE_DATA_SLOT, Some(&self.solver_points), 0);
    }

    fn dispatch_color_indirect(&self, encoder: &ComputeCommandEncoderRef, color: u32) {
        let pso = self
            .pipelines
            .get(if self.compact_contacts {
                "solve_primal_color_compact"
            } else {
                "solve_primal_color"
            })
            .expect("pipeline");
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(22, Some(&self.pass_table), (color as u64) * 4);
        // three-avbd: never bind the indirect buffer as a writable shader resource on the
        // same dispatch that consumes it for threadgroup counts.
        encoder.set_buffer(23, None, 0);
        let tg = MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups_indirect(&self.indirect_args, (color as u64) * 12, tg);
    }

    fn write_scratch(&self) {
        let s = unsafe {
            std::slice::from_raw_parts_mut(
                self.scratch[self.host_slot()].contents() as *mut u32,
                64,
            )
        };
        s[0] = self.command_count;
        s[1] = self.capacity.max_bodies;
        s[2] = self.capacity.max_candidates;
        s[3] = self.capacity.max_contacts;
        s[4] = self.capacity.max_colors.min(32);
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
        // One entry per manifold; bound hash memory/work independently of dispatch size.
        // The clear pass uses exactly this host size.
        const MAX_WARM_HASH: u32 = 1 << 21;
        s[9] = (prev.saturating_mul(4).max(1024) as u32)
            .next_power_of_two()
            .min(ids_len)
            .min(MAX_WARM_HASH);
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
        s[26] = 1; // incremental coloring: history row + compact work slot 69
        s[18] = self.grid_levels; // hierarchical broadphase levels (0 = uniform grid)
        // Per-step broadphase counters (the WebGPU hosts rewrite all 64 words):
        // legacy large list (25), planes (42), level mask (51), oversized statics (53).
        s[25] = 0;
        s[42] = 0;
        s[51] = 0;
        s[53] = 0;
        s[31] = 0; // hierarchical levels holding a moving collider
        s[19] = 0; // reuse hash miss
        s[20] = 0; // split broadphase grids (WebGPU opt-in; off on Metal)
        s[21] = 0; // two-pass narrowphase tail cap (unused on Metal)
        s[21] = 0; // update_refs noRef
        s[22] = 0; // update_refs linMove
        s[23] = 0; // update_refs angMove
        s[29] = u32::from(self.has_convex); // degree-aware primal lanes for convex/compound scenes
        // Collision filters present (bit 0); vehicle sever grace by collider
        // generation when the scene has none (bit 1).
        s[30] = u32::from(self.has_filters)
            | (u32::from(!self.has_filters && self.vehicles.as_ref().is_some_and(|v| v.vehicle_count() > 0)) << 1);
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

            manifold_count: 0,
            _pad0: 0,
            convex_sat_small: 0,
            narrowphase_deferred: 0,
            _pad2: 0,
            convex_sat_large: 0,
            color_rounds: 0,
            color_round_limit: 0,
            _pad3: 0,
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

    /// three-avbd `reuseManifolds`: skip bounds, broadphase and narrowphase and
    /// carry every previous manifold forward (persistence re-evaluates C0). Falls
    /// back to a full step when the scene changed since the last step.
    pub fn step_reuse_manifolds(&mut self) -> Result<StepCounters, ExecError> {
        self.reuse_next = true;
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
        let hier = match std::env::var("ENGINE_GRID_LEVELS").as_deref() {
            Ok("off") => false,
            Ok("on") => true,
            _ => self.has_convex,
        };
        if hier {
            let (base, levels) = match self.grid_cache {
                Some((n, base, levels)) if n == self.collider_count => (base, levels),
                _ => {
                    let (base, levels) = engine_schema::convex::hierarchical_grid(cols, bodies);
                    self.grid_cache = Some((self.collider_count, base, levels));
                    (base, levels)
                }
            };
            self.params.grid_cell_size = base;
            self.grid_levels = levels;
            return;
        }
        self.grid_levels = 0;
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
        let vehicles_dirty = self.vehicles.as_ref().is_some_and(|v| v.dirty);
        if self.steps_done > 0
            && self.last_counters.awake_count == 0
            && self.command_count == 0
            && self.body_count > 0
            && !vehicles_dirty
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
            let n = 128usize;
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

        if self.command_count > 0 {
            self.dispatch(&encoder, "apply_commands", self.command_count as u64);
        }
        let dirty = unsafe { *(self.scratch[self.host_slot()].contents() as *const u32).add(6) };
        if dirty != 0 || self.command_count > 0 {
            self.dispatch(&encoder, "recompute_inertia", n_bodies);
        }
        let (n_veh, n_wheels) = self
            .vehicles
            .as_ref()
            .map_or((0, 0), |v| (u64::from(v.vehicle_count()), u64::from(v.wheel_count())));
        if n_veh > 0 {
            self.dispatch_vehicle(&encoder, "vehicle_input", n_veh, false);
        }
        // three-avbd order: collide at x⁻ → topology → warm-start bodies → solve.
        // C0 must be sampled before predict; NP-after-predict made impact C0 huge
        // negative and blew spaced Stack.
        let reuse_manifolds = std::mem::take(&mut self.reuse_next) && self.manifolds_valid;
        if !reuse_manifolds {
        self.dispatch(&encoder, "build_aabbs", n_cols);
        // O(1) joint-pair filter for broadphase (was O(joints) per candidate).
        // Ring/Wall smash have zero joints — skip the 64k clear entirely.
        if self.joint_count > 0 {
            self.dispatch(&encoder, "joint_hash_clear", u64::from(engine_schema::joint_hash_size(self.joint_count)));
            self.dispatch(&encoder, "joint_hash_insert", n_joints);
        }
        self.dispatch(&encoder, "broadphase_clear", 131072);
        self.dispatch(&encoder, "broadphase_insert", n_cols);
        self.dispatch(&encoder, "broadphase_query", n_cols);
        if n_veh > 0 {
            self.dispatch_vehicle(&encoder, "wheel_query", n_wheels, true);
            // Before the constraint graph: link wheels on dynamic bodies.
            self.dispatch_vehicle(&encoder, "vehicle_link", n_wheels, false);
        }
        }
        self.dispatch(&encoder, "write_work_args", 64);
        // Pair hash before NP when reuseContacts is on (three-avbd order); otherwise
        // after NP is enough for persist-only warm-start.
        let hash_n = {
            let s = unsafe {
                std::slice::from_raw_parts(
                    self.scratch[self.host_slot()].contents() as *const u32,
                    64,
                )
            };
            s[9].max(1024) as u64
        };
        let reuse = (self.params.flags & engine_schema::FLAG_REUSE_CONTACTS) != 0;
        // Settled piles: update_refs over all bodies was part of the ~9 ms floor with
        // ~20 awake / few contacts. Refs only matter for reuse; skip when nearly asleep.
        let mostly_asleep = self.last_counters.awake_count > 0
            && self.last_counters.awake_count <= 64
            && self.last_counters.contact_count <= 256;
        if reuse_manifolds {
            // Every previous run becomes this step's manifold; no hash needed.
            self.dispatch_work(&encoder, "reuse_all_manifolds", 73, 0);
        } else {
        if reuse {
            if !mostly_asleep {
                self.dispatch(&encoder, "update_refs", n_bodies);
            }
            self.dispatch(&encoder, "contact_hash_clear", hash_n);
            self.dispatch_work(&encoder, "contact_hash_insert", 68, 0);
        }
        self.dispatch_work(&encoder, "narrowphase", 64, 0);
        if self.has_convex {
            // Hull pairs deferred by narrowphase. gConvexWork is buffer 27 in the
            // convex entries; restore the solver points binding afterwards.
            encoder.set_buffer(27, Some(&self.convex_work), 0);
            encoder.set_buffer(28, Some(&self.sat_cache), 0);
            self.dispatch_work(&encoder, "convex_classify", 64, 0);
            self.dispatch_work(&encoder, "convex_bucket", 64, 0);
            self.dispatch(&encoder, "convex_write_args", 64);
            self.dispatch_work(&encoder, "convex_sat_small", 74, 0);
            self.dispatch_work(&encoder, "convex_sat_large", 75, 0);
            self.dispatch_work(&encoder, "convex_manifold", 76, 0);
            encoder.set_buffer(27, Some(&self.solver_points), 0);
        }
        }
        self.dispatch(&encoder, "write_work_args", 64);
        if !reuse && !reuse_manifolds {
            self.dispatch(&encoder, "contact_hash_clear", hash_n);
            self.dispatch_work(&encoder, "contact_hash_insert", 68, 0);
        }
        self.dispatch_work(&encoder, "persist_contacts", 65, 0);
        if self.joint_count > 0 {
            self.dispatch(&encoder, "warmstart_joints", n_joints);
        }

        // Awake list before CSR sort/colour (flags only — independent of contacts).
        self.dispatch(&encoder, "build_awake_list", n_bodies);
        self.dispatch(&encoder, "write_work_args", 64);

        // CSR adjacency (three-avbd degree → scan → scatter → sort). Overwrites grid hash.
        // Degree/scatter walk manifolds (Stage 8b), not individual contact points.
        self.dispatch(&encoder, "csr_clear", n_bodies + 1);
        self.dispatch_work(&encoder, "csr_degree_contacts", 66, 0);
        if self.joint_count > 0 {
            self.dispatch(&encoder, "csr_degree_joints", n_joints);
        }
        if self.body_count <= 512 {
            self.dispatch(&encoder, "csr_prefix", 1);
        } else {
            self.dispatch(&encoder, "csr_prefix_blocks", (n_bodies + 3) / 4);
            self.dispatch(&encoder, "csr_prefix_block_sums", 64);
            self.dispatch(&encoder, "csr_prefix_add", n_bodies);
        }
        self.dispatch_work(&encoder, "csr_scatter_contacts", 66, 0);
        if self.joint_count > 0 {
            self.dispatch(&encoder, "csr_scatter_joints", n_joints);
        }
        self.dispatch_work(&encoder, "csr_sort", 67, 0);

        self.dispatch_work(&encoder, "init_colors", 67, 0);
        self.dispatch(&encoder, "write_work_args", 64);

        // A colour count is not the JP dependency depth. The active 110k
        // ring still has uncoloured bodies after 16 rounds at step 60.
        let color_rounds = 32;
        for round in 0..color_rounds {
            self.dispatch_work(&encoder, "color_bodies", 69, round % 16);
        }
        self.dispatch(&encoder, "count_color_tiles", n_bodies);
        self.dispatch(&encoder, "prefix_color_tiles", 64);
        self.dispatch(&encoder, "scatter_color_tiles", n_bodies);
        // Per-colour max CSR degree for degree-aware primal lanes (scratch s[20]).
        self.dispatch_work(&encoder, "record_color_degrees", 67, 0);
        self.dispatch(&encoder, "write_color_args", 64);

        // Body warm-start AFTER NP (three-avbd warmStartBodies): stash inertial, advance pose.
        self.dispatch(&encoder, "predict_bodies", n_bodies);
        if n_veh > 0 {
            self.dispatch_vehicle(&encoder, "vehicle_prepare", n_veh, false);
        }

        // three-avbd argsColors: indirect per colour (0 threadgroups ⇒ free). Same encoder so
        // Metal tracks the write→dispatch dependency without a CPU mid-step sync.
        // Colored Jacobi is enough once C0 is sampled at x⁻ (solve_primal_one kept for probes).
        // Dual stays per-contact (occupancy); CSR/island use manifolds.
        // Dispatch the full palette; empty GPU indirect buckets cost no work.
        let max_c = self.capacity.max_colors.min(32);
        if self.compact_contacts {
            self.dispatch_work(&encoder, "clear_solver_contacts", 65, 0);
            self.dispatch_work(&encoder, "pack_solver_contacts", 66, 0);
        }
        for iter in 0..self.params.iterations {
            for color in 0..max_c {
                self.dispatch_color_indirect(&encoder, color);
            }
            // Indirect args buffer was unbound for primal; restore for later kernels.
            encoder.set_buffer(23, Some(&self.indirect_args), 0);
            self.dispatch_work(
                &encoder,
                if self.compact_contacts {
                    "solve_dual_compact"
                } else {
                    "solve_dual"
                },
                65,
                u32::from(iter + 1 == self.params.iterations),
            );
            if n_veh > 0 {
                self.dispatch_vehicle(&encoder, "vehicle_iterate", n_veh, false);
            }
        }

        self.dispatch(&encoder, "extract_velocity_sleep", n_bodies);
        if n_veh > 0 {
            self.dispatch_vehicle(&encoder, "vehicle_post", n_veh, false);
        }
        self.dispatch(&encoder, "write_sleep_args", 64);
        // Island sleep gate: a body may sleep only if every dynamic contact neighbour is
        // also a candidate (or already asleep). Skipping this lets upper boxes freeze while
        // the stack below settles — columns tore apart and injected energy. Always run a
        // fixed handful of rounds (accuracy); manifolds cut the fan-out vs per-contact.
        if self.sleep_island_rounds > 0 {
            let rounds = self.sleep_island_rounds.min(16);
            for _ in 0..rounds {
                self.dispatch_work(&encoder, "sleep_island_gate", 66, 0);
            }
        }
        self.dispatch_work(&encoder, "sleep_island_commit", 67, 0);

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
        if let Some(v) = self.vehicles.as_mut() {
            v.dirty = false;
        }
        if !wait {
            self.last_timings = PhaseTimingsMs {
                total_gpu: wall_t0.elapsed().as_secs_f64() * 1000.0,
                cpu_submit,
                ..Default::default()
            };
            let _ = PHASE_GRAPH;
            // Unwaited: overflow is unknown until wait_gpu; manifolds were produced.
            self.manifolds_valid = true;
            return Ok(self.last_counters);
        }
        cb.wait_until_completed();
        let gpu_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;
        let counters = unsafe { *(self.counters[slot].contents() as *const StepCounters) };
        self.last_counters = counters;
        self.manifolds_valid = counters.overflow_flags == 0;
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

    /// Everything a scene wrote, for replay on another backend.
    pub fn snapshot(&self) -> WorldSnapshot {
        let colliders = unsafe {
            std::slice::from_raw_parts(
                self.colliders.contents() as *const Collider,
                self.collider_count as usize,
            )
        };
        let materials =
            unsafe { std::slice::from_raw_parts(self.materials.contents() as *const Material, 64) };
        WorldSnapshot {
            bodies: self.download_bodies(),
            colliders: colliders.to_vec(),
            materials: materials.to_vec(),
            joints: self.joints().to_vec(),
            params: Some(self.params),
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
        unsafe { *(self.scratch[done].contents() as *const u32).add(index) }
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

    /// One joint record after the last completed step (multipliers, broken flag).

    /// Joints that broke since the previous call (event rings after the last
    /// completed step). Steps alternate between the two scratch buffers, so
    /// each has its own ring and cursor.
    pub fn poll_joint_breaks(&mut self) -> Vec<engine_schema::JointBreak> {
        let base = engine_schema::EVENT_RING_BASE as usize;
        let len = (4 + 4 * engine_schema::EVENT_RING_CAPACITY) as usize;
        let mut out = Vec::new();
        for (k, buf) in self.scratch.iter().enumerate() {
            let words = unsafe { std::slice::from_raw_parts((buf.contents() as *const u32).add(base), len) };
            let lost = engine_schema::decode_joint_breaks(words, &mut self.events_seen[k], &mut out);
            if lost > 0 {
                eprintln!("event ring overflow: {lost} events lost");
            }
        }
        out
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

impl MetalWorld {
    fn with_vehicles<R>(&mut self, f: impl FnOnce(&mut VehicleHost, &mut Self) -> R) -> R {
        let mut host = self.vehicles.take().expect("vehicle host");
        let r = f(&mut host, self);
        self.vehicles = Some(host);
        r
    }

    pub fn vehicle_host(&self) -> &VehicleHost {
        self.vehicles.as_ref().expect("vehicle host")
    }

    pub fn add_vehicle(
        &mut self,
        desc: &VehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
    ) -> Result<VehicleHandle, String> {
        self.with_vehicles(|h, w| h.add_vehicle(w, desc, position, rotation))
    }

    pub fn set_vehicle_input(&mut self, v: VehicleHandle, input: VehicleInput) {
        self.with_vehicles(|h, w| h.set_input(w, v, input));
    }

    pub fn set_functional_state(&mut self, v: VehicleHandle, state: FunctionalState) {
        self.with_vehicles(|h, w| h.set_functional_state(w, v, state));
    }

    pub fn detach_wheel(&mut self, v: VehicleHandle, wheel: u32) {
        self.with_vehicles(|h, w| h.detach_wheel(w, v, wheel));
    }

    pub fn disable_wheel(&mut self, v: VehicleHandle, wheel: u32) {
        self.with_vehicles(|h, w| h.disable_wheel(w, v, wheel));
    }

    pub fn detach_part(&mut self, v: VehicleHandle, part: u32) {
        self.with_vehicles(|h, w| h.detach_part(w, v, part));
    }

    pub fn disable_part(&mut self, v: VehicleHandle, part: u32) {
        self.with_vehicles(|h, w| h.disable_part(w, v, part));
    }

    pub fn add_wheel(&mut self, v: VehicleHandle, wheel: &WheelDesc) -> Result<u32, String> {
        self.with_vehicles(|h, w| h.add_wheel(w, v, wheel))
    }

    pub fn set_wheel_params(&mut self, v: VehicleHandle, wheel: u32, desc: &WheelDesc) {
        self.with_vehicles(|h, w| h.set_wheel_params(w, v, wheel, desc));
    }

    pub fn wake_vehicle(&mut self, v: VehicleHandle) {
        self.with_vehicles(|h, w| h.wake(w, v));
    }

    /// Reads GPU outputs; call after a waited step (`step_gpu`).
    pub fn vehicle_state(&self, v: VehicleHandle) -> VehicleState {
        self.vehicle_host().read_state(self, v)
    }

    /// Batched ray casts against the current poses (synchronous).
    pub fn raycast_batch(&mut self, rays: &[Ray]) -> Vec<Option<RayHit>> {
        if rays.is_empty() {
            return Vec::new();
        }
        self.with_vehicles(|h, w| h.write_rays(w, rays));
        self.write_scratch();
        self.sync_counters_host_to_gpu(false);
        let cb = self.device.queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        self.bind_all(encoder);
        self.dispatch(encoder, "build_aabbs", u64::from(self.collider_count.max(1)));
        self.dispatch_vehicle(encoder, "raycast_batch", rays.len() as u64, true);
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        self.vehicle_host().read_ray_hits(self)
    }

    #[allow(clippy::mut_from_ref)]
    fn vehicle_slots_mut(&self) -> &mut [[u32; 4]] {
        // Not via vehicle_host(): the host is taken while it drives this backend.
        unsafe { std::slice::from_raw_parts_mut(self.vehicle_data.contents() as *mut [u32; 4], self.vehicle_slots) }
    }
}

impl VehicleBackend for MetalWorld {
    fn body_count(&self) -> u32 {
        self.body_count
    }

    fn joint_count(&self) -> u32 {
        self.joint_count
    }

    fn append_joints(&mut self, joints: &[Joint]) {
        for j in joints {
            self.add_joint(*j);
        }
    }

    fn collider_count(&self) -> u32 {
        self.collider_count
    }

    fn append_bodies(&mut self, bodies: &[BodyState]) {
        for b in bodies {
            self.add_body(*b);
        }
    }

    fn append_colliders(&mut self, colliders: &[Collider]) {
        for c in colliders {
            self.add_collider(*c);
        }
    }

    fn write_body(&mut self, index: u32, body: &BodyState) {
        self.set_body(index, *body);
    }

    fn write_collider(&mut self, index: u32, collider: &Collider) {
        let cols = unsafe {
            std::slice::from_raw_parts_mut(
                self.colliders.contents() as *mut Collider,
                self.capacity.max_colliders as usize,
            )
        };
        cols[index as usize] = *collider;
    }

    fn write_slots(&mut self, slot: u32, data: &[[u32; 4]]) {
        let s = slot as usize;
        self.vehicle_slots_mut()[s..s + data.len()].copy_from_slice(data);
    }

    fn write_curve_points(&mut self, byte_offset: u64, points: &[[f32; 2]]) {
        let bytes: &[u8] = bytemuck::cast_slice(points);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.vehicle_data.contents() as *mut u8).add(byte_offset as usize),
                bytes.len(),
            );
        }
    }

    fn read_slots(&self, slot: u32, count: u32) -> Vec<[u32; 4]> {
        let s = slot as usize;
        self.vehicle_slots_mut()[s..s + count as usize].to_vec()
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
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
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
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
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
        let body = self
            .world
            .add_body(BodyState::dynamic(pos, Quat::IDENTITY, mass, inertia));
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

    /// Dynamic body made of convex hulls (one hull, or a compound). Hulls are in
    /// one input frame; `pos`/`rot` place that frame. Mass and inertia are the
    /// compound's, from uniform `density`; children keep their own colliders.
    pub fn add_convex_hull(
        &mut self,
        hulls: &[engine_schema::convex::ConvexHull],
        pos: Vec3,
        rot: Quat,
        density: f32,
    ) -> u32 {
        let cooked = engine_schema::convex::cook_body(hulls, density);
        let (p, q) = cooked.body_pose(pos, rot);
        let body = self
            .world
            .add_body(BodyState::dynamic(p, q, cooked.mass, cooked.inertia));
        for child in &cooked.children {
            let base = self.world.insert_hull(&child.hull);
            self.world.add_collider(Collider {
                body,
                shape: ShapeType::ConvexHull as u32,
                material: 0,
                generation: 0,
                local_pos: child.local_pos.to_array(),
                density,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: engine_schema::convex::convex_collider_params(
                    base,
                    1.0,
                    child.hull.bounding_radius(),
                ),
            });
        }
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

    /// Add a joint at rest in the bodies' current orientations (frame B is
    /// derived from frame A, three-avbd `rest: 'current'`).
    pub fn add_joint(&mut self, desc: JointDesc) -> u32 {
        let qa = Quat::from_array(self.world.body(desc.body_a).orientation);
        let qb = Quat::from_array(self.world.body(desc.body_b).orientation);
        self.world.add_joint_desc(&desc.rest_from_bodies(qa, qb))
    }

    /// Body-local frames for a joint placed at a world anchor with a world
    /// axis (hinge / slider / twist axis = frame X).
    pub fn world_frames(
        &self,
        a: u32,
        b: u32,
        anchor: Vec3,
        axis: Vec3,
    ) -> (JointFrame, JointFrame) {
        let ba = self.world.body(a);
        let bb = self.world.body(b);
        let qa = Quat::from_array(ba.orientation);
        let qb = Quat::from_array(bb.orientation);
        let la = qa.conjugate() * (anchor - Vec3::from_array(ba.position));
        let lb = qb.conjugate() * (anchor - Vec3::from_array(bb.position));
        let axis_a = qa.conjugate() * axis;
        (JointFrame::with_axis(la, axis_a), JointFrame::at(lb))
    }

    pub fn add_ball_joint(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> u32 {
        self.add_joint(JointDesc::ball(a, b, anchor_a, anchor_b))
    }

    pub fn add_fixed_joint(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3) -> u32 {
        self.add_joint(JointDesc::fixed(
            a,
            b,
            JointFrame::at(anchor_a),
            JointFrame::at(anchor_b),
        ))
    }

    /// Prismatic joint; `axis` is in body A's local frame.
    pub fn add_slider(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        axis: Vec3,
    ) -> u32 {
        self.add_joint(JointDesc::slider(
            a,
            b,
            JointFrame::with_axis(anchor_a, axis),
            JointFrame::at(anchor_b),
        ))
    }

    /// D6 with every DOF locked except twist about `axis` (body A local);
    /// adjust motions with `add_joint(JointDesc::d6(..).with_motion(..))`.
    pub fn add_d6(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, axis: Vec3) -> u32 {
        self.add_joint(
            JointDesc::d6(
                a,
                b,
                JointFrame::with_axis(anchor_a, axis),
                JointFrame::at(anchor_b),
            )
            .with_motion(Dof::Twist, Motion::Free),
        )
    }

    /// Revolute joint about `axis` (body A local).
    pub fn add_hinge(&mut self, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, axis: Vec3) -> u32 {
        self.add_joint(JointDesc::hinge(
            a,
            b,
            JointFrame::with_axis(anchor_a, axis),
            JointFrame::at(anchor_b),
        ))
    }

    pub fn add_distance(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        length: f32,
    ) -> u32 {
        self.add_joint(JointDesc::distance(a, b, anchor_a, anchor_b, length))
    }

    pub fn add_spring(
        &mut self,
        a: u32,
        b: u32,
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest: f32,
        stiffness: f32,
    ) -> u32 {
        self.add_joint(JointDesc::spring(a, b, anchor_a, anchor_b, rest, stiffness))
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
                self.add_ball_joint(p, id, Vec3::new(0.0, -half, 0.0), Vec3::new(0.0, half, 0.0));
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

#[cfg(test)]
mod scan_tests {
    use super::*;

    #[test]
    fn stable_color_tiles_use_full_workgroups_and_keep_body_order() {
        let mut world = MetalWorld::new(WorldCapacity {
            max_colors: 7,
            ..WorldCapacity::small()
        })
        .expect("Metal GPU required for stable color packing regression");
        for n in [1u32, 63, 64, 65, 129] {
            world.body_count = n;
            world.write_scratch();
            unsafe {
                *(world.counters[0].contents() as *mut StepCounters) = StepCounters {
                    body_count: n,
                    awake_count: n - 1,
                    ..Default::default()
                };
                let bodies = std::slice::from_raw_parts_mut(
                    world.bodies.contents() as *mut BodyState,
                    n as usize,
                );
                let colors = std::slice::from_raw_parts_mut(
                    world.body_color.contents() as *mut u32,
                    n as usize,
                );
                for i in 0..n as usize {
                    bodies[i] = BodyState {
                        inv_mass: if i == 0 { 0.0 } else { 1.0 },
                        flags: if i == 0 { 0 } else { 9 },
                        ..bytemuck::Zeroable::zeroed()
                    };
                    colors[i] = (i % 3) as u32;
                }
                std::slice::from_raw_parts_mut(
                    world.color_lists.contents() as *mut u32,
                    (world.capacity.max_bodies * (world.capacity.max_colors + 2)) as usize,
                )
                .fill(u32::MAX);
            }
            let cb = world.device.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            world.bind_all(&encoder);
            world.dispatch(&encoder, "count_color_tiles", n as u64);
            world.dispatch(&encoder, "prefix_color_tiles", 64);
            world.dispatch(&encoder, "scatter_color_tiles", n as u64);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed, "n={n}: stable packing requires complete 64-lane workgroups, including the padded last tile");
            unsafe {
                let counts =
                    std::slice::from_raw_parts(world.color_counts[0].contents() as *const u32, 7);
                let lists = std::slice::from_raw_parts(
                    world.color_lists.contents() as *const u32,
                    (world.capacity.max_bodies * world.capacity.max_colors) as usize,
                );
                for color in 0..7u32 {
                    let expected: Vec<u32> = (1..n).filter(|i| i % 3 == color).collect();
                    assert_eq!(
                        counts[color as usize] as usize,
                        expected.len(),
                        "n={n} color={color}: missing lanes or incorrect tile prefix"
                    );
                    let start = (color * world.capacity.max_bodies) as usize;
                    assert_eq!(
                        &lists[start..start + expected.len()],
                        &expected,
                        "n={n} color={color}: Metal color bucket lost stable body order"
                    );
                    assert_eq!(
                        lists[start + expected.len()],
                        u32::MAX,
                        "n={n} color={color}: partial tile wrote beyond the live bucket"
                    );
                }
            }
        }
    }

    #[test]
    fn color_dispatch_respects_palette_and_preserves_work_slots() {
        for palette in [1u32, 7, 32] {
            let world = MetalWorld::new(WorldCapacity {
                max_colors: palette,
                ..WorldCapacity::small()
            })
            .expect("Metal GPU required for color dispatch bounds regression");
            world.write_scratch();
            // Allocate exactly the configured count buffer. Slots after the
            // palette are canaries; never execute their indirect arguments.
            unsafe {
                std::slice::from_raw_parts_mut(
                    world.color_counts[0].contents() as *mut u32,
                    palette as usize,
                )
                .fill(1);
                *(world.color_counts[0].contents() as *mut u32) = 0;
                std::slice::from_raw_parts_mut(world.indirect_args.contents() as *mut u32, 70 * 3)
                    .fill(0xdeadbeef);
            }
            let cb = world.device.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            world.bind_all(&encoder);
            world.dispatch(&encoder, "write_color_args", 64);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed,
                "Metal color args failed for palette={palette}; count-buffer reads must stay inside the allocation");
            let args = unsafe {
                std::slice::from_raw_parts(world.indirect_args.contents() as *const u32, 70 * 3)
            };
            for slot in 0..70usize {
                let expected = if slot < palette as usize {
                    [u32::from(slot != 0), 1, 1]
                } else {
                    [0xdeadbeef; 3]
                };
                assert_eq!(&args[slot * 3..slot * 3 + 3], &expected,
                    "palette={palette}, slot={slot}: color args touched a slot outside the palette or failed to publish an active bucket; a 64-thread launch must not read 64 color counts unconditionally");
            }
        }
    }

    #[test]
    fn csr_block_scan_uses_complete_workgroups_on_metal() {
        let mut world = MetalWorld::new(WorldCapacity {
            max_bodies: 16386,
            max_contacts: 1024,
            max_candidates: 1024,
            ..WorldCapacity::small()
        })
        .expect("Metal GPU required for cooperative CSR scan regression");
        // 257 and 513 require padded workgroups. 16385 also needs more than
        // one block sum per lane in the second pass. Never run scatter here:
        // corrupt offsets must produce a diagnostic assertion, not GPU OOB work.
        for n in [1u32, 255, 256, 257, 513, 16385] {
            world.body_count = n;
            let degrees: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % 11).collect();
            unsafe {
                let indices = std::slice::from_raw_parts_mut(
                    world.constraint_index.contents() as *mut u32,
                    n as usize + 2,
                );
                indices[..n as usize].copy_from_slice(&degrees);
                indices[n as usize] = 0xdeadbeef;
                indices[n as usize + 1] = 0xfeedface;
                *(world.counters[0].contents() as *mut StepCounters) = StepCounters {
                    body_count: n,
                    ..StepCounters::default()
                };
            }
            let cb = world.device.queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            world.bind_all(&encoder);
            world.dispatch(&encoder, "csr_prefix_blocks", u64::from((n + 3) / 4));
            world.dispatch(&encoder, "csr_prefix_block_sums", 64);
            world.dispatch(&encoder, "csr_prefix_add", n.into());
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(
                cb.status(),
                MTLCommandBufferStatus::Completed,
                "Metal CSR scan failed before readback for {n} bodies"
            );
            let (indices, cursors) = unsafe {
                (
                    std::slice::from_raw_parts(
                        world.constraint_index.contents() as *const u32,
                        n as usize + 2,
                    ),
                    std::slice::from_raw_parts(
                        world.body_color.contents() as *const u32,
                        n as usize,
                    ),
                )
            };
            let mut expected = 0;
            for i in 0..=n as usize {
                assert_eq!(indices[i], expected,
                    "Metal CSR offset {i}/{n} is wrong: cooperative scans require 64 lanes and a padded final workgroup, not the GPU SIMD width");
                if i < n as usize {
                    assert_eq!(
                        cursors[i], expected,
                        "Metal CSR scatter cursor {i}/{n} was not published"
                    );
                    expected += degrees[i];
                }
            }
            assert_eq!(
                indices[n as usize + 1],
                0xfeedface,
                "Metal CSR scan wrote beyond its end offset for {n} bodies"
            );
        }
    }
}
