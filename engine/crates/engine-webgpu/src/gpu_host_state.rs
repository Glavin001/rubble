//! Host access to the simulation state, for hosts that couple the engine to
//! another solver (the stress oracle's `RigidEngine`, `stress-rubble`): write body
//! state back, read the last step's contact forces, snapshot and restore the whole
//! simulation. All additive: a host that never calls these steps exactly as before.

use super::*;
use engine_schema::commands::{CommandType, GpuCommand, CMD_RUN_MEMBER};

const BODY_COMMANDS: [u32; 11] = [
    CommandType::WakeBody as u32,
    CommandType::TeleportBody as u32,
    CommandType::SetVelocity as u32,
    CommandType::AttachForce as u32,
    CommandType::ApplyImpulse as u32,
    CommandType::AddForceAt as u32,
    CommandType::AngularImpulse as u32,
    CommandType::SetMassFrame as u32,
    CommandType::SetKinematicTarget as u32,
    CommandType::Damp as u32,
    CommandType::SetLocks as u32,
];

/// Every GPU buffer and the host mirror of a [`WgpuSim`] at one instant
/// ([`WgpuSim::save_state`]).
pub struct SimSnapshot {
    /// Per binding: the copy, the bytes copied, and whether the live buffer's bytes
    /// past them were zeros (a compact snapshot), cleared again on restore.
    bufs: Vec<(wgpu::Buffer, u64, bool)>,
    body_count: u32,
    collider_count: u32,
    joint_count: u32,
    command_count: u32,
    last_counters: StepCounters,
    steps_done: u32,
    contacts_parity: u32,
    host_bodies: Vec<BodyState>,
    host_cols: Vec<Collider>,
    has_convex: bool,
    has_height_field: bool,
    has_filters: bool,
    grid_levels: u32,
    grid_cache: Option<(u32, f32, u32)>,
    movable_radius: f32,
    manifolds_valid: bool,
    params: AvbdParams,
    last_contact_offset: u32,
    vehicles: Option<engine_vehicle::VehicleHost>,
    events_seen: u32,
    /// The event ring's other readers' cursors: the ring's words are restored
    /// with the buffers, so each reader must go back to where it was.
    mount_events_seen: u32,
    beam_events_seen: u32,
    motion_events_seen: u32,
    soft: Option<engine_softbody::SoftHost>,
    free_joints: Vec<u32>,
}

/// The terrain pool's bindings (MPM and the live surface), which snapshots skip.
/// A compact snapshot's measure in flight ([`WgpuSim::measure_state`]).
pub struct StateMeasure {
    staging: wgpu::Buffer,
    measured: Vec<bool>,
    stamp: [u32; 5],
}

impl StateMeasure {
    /// The extents, one `u32` per binding: map it for reading, then pass its words to
    /// [`WgpuSim::save_state_measured`].
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.staging
    }

    /// The extents' count: the simulation's bindings.
    pub fn bindings(&self) -> usize {
        self.measured.len()
    }
}

fn is_terrain_binding(i: usize) -> bool {
    (engine_terrain::layout::MPM_PARAMS_BINDING..=engine_terrain::surface::SURFACE_BINDING)
        .contains(&i)
}

impl WgpuSim {
    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    pub fn collider_count(&self) -> u32 {
        self.collider_count
    }

    pub fn params(&self) -> AvbdParams {
        self.params
    }

    /// The counters of the last step.
    pub fn last_counters(&self) -> StepCounters {
        self.last_counters
    }

    /// The buffers another pipeline on the same device reads or writes (the stress
    /// solver's GPU-resident coupling): bodies (binding 0, `BodyState`: current rows at
    /// `[0, max_bodies)`, previous at `max_bodies + i`), colliders (binding 2,
    /// `Collider`), contacts (binding 4, `Contact`: the last step's rows from
    /// `last_contact_offset`, `StepCounters::contact_count` of them, dead rows included)
    /// and the step counters (binding 7, `StepCounters`, written by the GPU).
    pub fn body_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[0]
    }

    pub fn collider_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[2]
    }

    pub fn contact_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[4]
    }

    /// The joint records (binding 5, `Joint`, by joint slot): renderers read a bond's
    /// broken flag (word 2, bit 0) in place.
    pub fn joint_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[5]
    }

    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[7]
    }

    /// The step's colour bucket sizes (binding 17: bodies per primal colour in
    /// words `[0, max_colors)`, then per-colour max degrees), for checks that
    /// the colouring covered every awake body.
    pub fn color_count_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[17]
    }

    /// The soft-body arena (binding 32, `vec4<u32>` slots laid out by
    /// `SoftCapacity::layout`: body records, beams, groups, triangles, ...),
    /// for renderers (beam views) and reads of soft state.
    pub fn soft_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[32]
    }

    /// The step's broadphase candidate pairs (binding 10, `CandidatePair`: two colliders),
    /// `StepCounters::candidate_count` of them once the broadphase has run (a hook of
    /// `step_encode_with` at `"broadphase"` sees this step's).
    pub fn candidate_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[10]
    }

    /// First row of the last step's contacts in `contact_buffer` (the ping-pong half).
    pub fn last_contact_offset(&self) -> u32 {
        self.last_contact_offset
    }

    /// First row of the next step's contacts in `contact_buffer`: what
    /// [`Self::last_contact_offset`] is once that step is encoded (a hook within it reads
    /// its rows there).
    pub fn next_contact_offset(&self) -> u32 {
        if self.contacts_parity == 0 {
            0
        } else {
            self.capacity.max_contacts
        }
    }

    pub fn capacity(&self) -> WorldCapacity {
        self.capacity.clone()
    }

    /// Overwrite the state of bodies `first..first + bodies.len()` (pose, velocity,
    /// mass, flags), current and previous rows alike, as `add_bodies` writes them.
    /// The next step runs even if the last one left no body awake. Writing
    /// `inv_mass = 0` (and zero velocity) holds a body still for the steps until it
    /// is written back: the solver treats it as static (contacts against it are
    /// resolved for the others; it is neither predicted nor integrated).
    pub fn write_bodies(&mut self, first: u32, bodies: &[BodyState]) {
        self.mark_rest_stale();
        let count = u32::try_from(bodies.len()).expect("body batch exceeds u32 index space");
        assert!(
            first <= self.body_count && count <= self.body_count - first,
            "write_bodies {first}..{} outside the {} bodies",
            first + count,
            self.body_count
        );
        if count == 0 {
            return;
        }
        let stride = std::mem::size_of::<BodyState>() as u64;
        let bytes = bytemuck::cast_slice(bodies);
        self.queue
            .write_buffer(&self.bufs[0], u64::from(first) * stride, bytes);
        self.queue.write_buffer(
            &self.bufs[0],
            (u64::from(self.capacity.max_bodies) + u64::from(first)) * stride,
            bytes,
        );
        self.host_bodies[first as usize..(first + count) as usize].copy_from_slice(bodies);
        self.manifolds_valid = false;
        self.wake_pending = true;
        self.note_motion_flags(bodies);
    }

    /// Queue body commands for the next step, without reading anything back:
    /// wake, teleport, set velocity, apply an impulse at a point, add a force and
    /// torque for one step (`GpuCommand::{wake, teleport, set_velocity,
    /// apply_impulse, add_force}`). The step applies them before anything reads
    /// body state, in submission order per body, with the result a host
    /// `read_body` → edit → `write_bodies` would give (both rows written; dynamic
    /// bodies woken with their sleep timer reset). More than `max_commands` in one
    /// step is an error, never a silent drop.
    pub fn push_body_commands(&mut self, cmds: &[GpuCommand]) -> anyhow::Result<()> {
        let total = self.pending_commands.len() + cmds.len();
        anyhow::ensure!(
            total <= self.capacity.max_commands as usize,
            "{total} body commands this step exceed WorldCapacity::max_commands ({})",
            self.capacity.max_commands
        );
        for c in cmds {
            anyhow::ensure!(
                c.a < self.body_count,
                "body command for body {} outside the {} bodies",
                c.a,
                self.body_count
            );
            anyhow::ensure!(
                BODY_COMMANDS.contains(&c.op),
                "op {} is not a body command",
                c.op
            );
        }
        for c in cmds {
            if c.op == CommandType::SetMassFrame as u32 {
                // Broadphase cell sizing reads which bodies are dynamic.
                let b = &mut self.host_bodies[c.a as usize];
                b.inv_mass = c.f2[3];
                b.inv_inertia = [c.f2[0], c.f2[1], c.f2[2]];
            }
            if c.op == CommandType::TeleportBody as u32 {
                // The broadphase cell sizing reads host poses.
                let b = &mut self.host_bodies[c.a as usize];
                b.position = [c.f0[0], c.f0[1], c.f0[2]];
                b.orientation = c.f1;
            }
        }
        self.pending_commands.extend_from_slice(cmds);
        self.command_count = self.pending_commands.len() as u32;
        self.manifolds_valid = false;
        Ok(())
    }

    /// Collider `id` as last written (host mirror).
    pub fn collider_record(&self, id: u32) -> Collider {
        self.host_cols[id as usize]
    }

    /// The body collider `id` belongs to (host mirror).
    pub fn collider_body(&self, id: u32) -> Option<u32> {
        self.host_cols.get(id as usize).map(|c| c.body)
    }

    /// Drop the queued commands for `body` (it is being removed or overwritten).
    pub fn cancel_body_commands(&mut self, body: u32) {
        self.pending_commands.retain(|c| c.a != body);
        self.command_count = self.pending_commands.len() as u32;
    }

    /// Body commands queued for the next step.
    pub fn pending_body_commands(&self) -> usize {
        self.pending_commands.len()
    }

    /// Write the queued commands to binding 8, grouped into one run per body
    /// (stable: submission order within a body), so one thread applies each run.
    pub(crate) fn flush_commands(&mut self) {
        if self.pending_commands.is_empty() {
            return;
        }
        let mut cmds = std::mem::take(&mut self.pending_commands);
        cmds.sort_by_key(|c| c.a);
        let mut start = 0;
        while start < cmds.len() {
            let mut end = start + 1;
            while end < cmds.len() && cmds[end].a == cmds[start].a {
                cmds[end].c = CMD_RUN_MEMBER;
                end += 1;
            }
            cmds[start].c = (end - start) as u32;
            start = end;
        }
        self.queue
            .write_buffer(&self.bufs[8], 0, bytemuck::cast_slice(&cmds));
        self.command_count = cmds.len() as u32;
    }

    /// The contact rows the last step solved, with their final forces: live rows
    /// (`alive != 0`) carry the normal and friction forces `lambda_n` (<= 0 pushes
    /// the bodies apart), `lambda_t1`, `lambda_t2` along `normal` (from collider A's
    /// body towards B's) and the tangents. The force on B's body is
    /// `-(lambda_n normal + lambda_t1 tangent1 + lambda_t2 tangent2)` at
    /// `position + rotate(orientation, anchor_b)` (A's: the opposite, at its anchor);
    /// over the step it delivers that force times `dt`. Not available with
    /// persistent contact storage, which keeps no per-row forces in binding 4.
    pub fn last_contacts(&self) -> anyhow::Result<Vec<Contact>> {
        anyhow::ensure!(
            !self.persistent_contacts,
            "contact readback needs the default or compact contact storage"
        );
        if self.steps_done == 0 {
            return Ok(Vec::new());
        }
        let n = self
            .last_counters
            .contact_count
            .min(self.capacity.max_contacts);
        Ok(self.read_buffer_range(
            4,
            u64::from(self.last_contact_offset) * std::mem::size_of::<Contact>() as u64,
            n as usize,
        ))
    }

    /// Diagnostic: the last step's candidate pairs ([a, b, key, pad]) and the split
    /// narrowphase's deferred tail (binding 10; waits).
    pub fn last_candidates(&self) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
        if self.steps_done == 0 {
            return (Vec::new(), Vec::new());
        }
        let cap = u64::from(self.capacity.max_candidates);
        let n = u64::from(self.last_counters.candidate_count).min(cap);
        let d = u64::from(self.last_counters.narrowphase_deferred).min(cap - n);
        let head: Vec<u32> = self.read_buffer_range(10, 0, (n * 4) as usize);
        let tail: Vec<u32> = self.read_buffer_range(10, (cap - d) * 16, (d * 4) as usize);
        let q = |v: &[u32]| v.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<_>>();
        (q(&head), q(&tail))
    }

    /// Diagnostic: the last step's persistent contact rows, as (colliderA, colliderB,
    /// featureId, pair tag, λ normal, penalty normal, c0N) (bindings 4 and 27; waits).
    pub fn last_persistent_rows(&self) -> Vec<(u32, u32, u32, u32, f32, f32, f32)> {
        if !self.persistent_contacts || self.steps_done == 0 {
            return Vec::new();
        }
        let n = self.last_counters.contact_count.min(self.capacity.max_contacts) as usize;
        let at = u64::from(self.last_contact_offset);
        let meta: Vec<u32> = self.read_buffer_range(4, at * 32, n * 8);
        let pts: Vec<u32> = self.read_buffer_range(27, at * 64, n * 16);
        (0..n)
            .map(|k| {
                let m = &meta[k * 8..k * 8 + 8];
                let p = &pts[k * 16..k * 16 + 16];
                (m[0], m[1], m[2], p[15], f32::from_bits(p[8]), f32::from_bits(p[12]), f32::from_bits(p[3]))
            })
            .collect()
    }

    /// Copy every buffer and the host mirror (bodies, colliders, counters, contact
    /// ping-pong, warm-start state): `restore_state` returns the simulation to this
    /// instant exactly. The hull arena (binding 18) is the host's: a host that
    /// inserts hulls between save and restore keeps them.
    pub fn save_state(&self) -> SimSnapshot {
        self.save_into(Vec::new(), None)
    }

    /// `save_state` into an older snapshot's GPU buffers where their sizes still fit
    /// (no allocation per save for a host that saves every frame).
    pub fn save_state_reusing(&self, old: SimSnapshot) -> SimSnapshot {
        self.save_into(old.bufs, None)
    }

    /// First half of a compact snapshot: measure each buffer's extent (one plus its
    /// last non-zero word, `state_extent.slang`) into [`StateMeasure::buffer`]. Map
    /// that buffer, then [`WgpuSim::save_state_measured`] with its words before
    /// anything else touches the simulation. A compact snapshot copies only the
    /// words below each extent and restores the zeros past them, so it restores the
    /// same bytes as [`WgpuSim::save_state`] in a fraction of the memory when the
    /// buffers are mostly unused (a scene before its first step: contacts, solver
    /// rows and archives are empty).
    pub fn measure_state(&self) -> StateMeasure {
        let device = &self.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("state_extent"),
            source: wgpu::ShaderSource::Wgsl(crate::STATE_EXTENT_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("state_extent"),
            layout: None,
            module: &module,
            entry_point: Some("state_extent"),
            compilation_options: Default::default(),
            cache: None,
        });
        let n = self.bufs.len();
        let extents = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state extents"),
            size: (n.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state extents read"),
            size: (n.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        // Measured: storage buffers that bind whole; the rest are copied whole.
        let measured: Vec<bool> = self
            .bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                b.usage().contains(wgpu::BufferUsages::STORAGE)
                    && b.size() >= 4
                    && b.size() % 4 == 0
                    && b.size() <= max_binding
                    && !(self.terrain.is_some() && is_terrain_binding(i))
            })
            .collect();
        let mut groups_of = Vec::new();
        let params: Vec<Option<(wgpu::BindGroup, u32)>> = self
            .bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                if !measured[i] {
                    return None;
                }
                let words = (b.size() / 4) as u32;
                let groups = words.div_ceil(64).clamp(1, 4096);
                let p = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                    label: Some("state extent params"),
                    contents: bytemuck::cast_slice(&[words, i as u32, groups * 64, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("state extent"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: extents.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: p.as_entire_binding() },
                    ],
                });
                groups_of.push(p);
                Some((bg, groups))
            })
            .collect();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("measure-state"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("state_extent"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            for (bg, groups) in params.iter().flatten() {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
        }
        enc.copy_buffer_to_buffer(&extents, 0, &staging, 0, (n.max(1) * 4) as u64);
        self.queue.submit([enc.finish()]);
        StateMeasure {
            staging,
            measured,
            stamp: self.state_stamp(),
        }
    }

    /// Second half of a compact snapshot: `extents` are [`StateMeasure::buffer`]'s
    /// words, read after [`WgpuSim::measure_state`]. Fails if the simulation stepped
    /// or took commands or records since the measure.
    pub fn save_state_measured(&self, m: &StateMeasure, extents: &[u32]) -> anyhow::Result<SimSnapshot> {
        anyhow::ensure!(
            m.stamp == self.state_stamp(),
            "the simulation changed between measure_state and save_state_measured"
        );
        anyhow::ensure!(extents.len() >= self.bufs.len(), "extents of another simulation");
        let sizes: Vec<Option<u64>> = (0..self.bufs.len())
            .map(|i| m.measured[i].then(|| u64::from(extents[i]) * 4))
            .collect();
        Ok(self.save_into(Vec::new(), Some(&sizes)))
    }

    /// A compact snapshot, waiting for the measure (native hosts; the web maps
    /// [`StateMeasure::buffer`] itself).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_state_compact(&self) -> anyhow::Result<SimSnapshot> {
        let m = self.measure_state();
        let slice = m.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()??;
        let extents: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        m.staging.unmap();
        self.save_state_measured(&m, &extents)
    }

    /// What a measure must still match when its snapshot is taken.
    fn state_stamp(&self) -> [u32; 5] {
        [
            self.steps_done,
            self.command_count,
            self.body_count,
            self.collider_count,
            self.joint_count,
        ]
    }

    fn save_into(&self, old: Vec<(wgpu::Buffer, u64, bool)>, compact: Option<&[Option<u64>]>) -> SimSnapshot {
        let mut old = old.into_iter();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("save-state"),
            });
        let bufs = self
            .bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                // The terrain pool is not saved (its buffers are the
                // largest): a restore starts it over (`terrain_reset`). A compact
                // snapshot copies a measured buffer up to its extent.
                let measured = compact.and_then(|c| c[i]);
                let size = if self.terrain.is_some() && is_terrain_binding(i) {
                    0
                } else if let Some(extent) = measured {
                    extent.min(b.size())
                } else {
                    b.size()
                };
                let copy = match old.next() {
                    Some((buf, s, _)) if s == size => buf,
                    _ => self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("snapshot-b{i}")),
                        size: size.max(4),
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                };
                if size > 0 {
                    enc.copy_buffer_to_buffer(b, 0, &copy, 0, size);
                }
                (copy, size, measured.is_some())
            })
            .collect();
        self.queue.submit([enc.finish()]);
        SimSnapshot {
            bufs,
            body_count: self.body_count,
            collider_count: self.collider_count,
            joint_count: self.joint_count,
            command_count: self.command_count,
            last_counters: self.last_counters,
            steps_done: self.steps_done,
            contacts_parity: self.contacts_parity,
            host_bodies: self.host_bodies.clone(),
            host_cols: self.host_cols.clone(),
            has_convex: self.has_convex,
            has_height_field: self.has_height_field,
            has_filters: self.has_filters,
            grid_levels: self.grid_levels,
            grid_cache: self.grid_cache,
            movable_radius: self.movable_radius,
            manifolds_valid: self.manifolds_valid,
            params: self.params,
            last_contact_offset: self.last_contact_offset,
            vehicles: self.vehicles.clone(),
            events_seen: self.events_seen,
            mount_events_seen: self.mount_events_seen,
            beam_events_seen: self.beam_events_seen,
            motion_events_seen: self.motion_events_seen,
            soft: self.soft.clone(),
            free_joints: self.free_joints.clone(),
        }
    }

    /// Return to a snapshot of this simulation (`save_state`).
    pub fn restore_state(&mut self, s: &SimSnapshot) {
        self.mark_rest_stale();
        assert_eq!(
            s.bufs.len(),
            self.bufs.len(),
            "snapshot of another simulation"
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("restore-state"),
            });
        for (i, (copy, size, zero_tail)) in s.bufs.iter().enumerate() {
            if i == 18 {
                continue; // the hull arena stays the host's
            }
            let live = self.bufs[i].size();
            if *size > 0 {
                enc.copy_buffer_to_buffer(copy, 0, &self.bufs[i], 0, (*size).min(live));
            }
            // A compact snapshot's buffer was zeros past what it copied. (Otherwise a
            // size of 0 is the terrain pool, which was not saved.)
            if *zero_tail && *size < live {
                enc.clear_buffer(&self.bufs[i], *size, None);
            }
        }
        self.queue.submit([enc.finish()]);
        self.body_count = s.body_count;
        self.collider_count = s.collider_count;
        self.joint_count = s.joint_count;
        self.command_count = s.command_count;
        self.last_counters = s.last_counters;
        self.steps_done = s.steps_done;
        self.contacts_parity = s.contacts_parity;
        self.host_bodies.clone_from(&s.host_bodies);
        self.host_cols.clone_from(&s.host_cols);
        self.has_convex = s.has_convex;
        self.has_height_field = s.has_height_field;
        self.has_filters = s.has_filters;
        self.grid_levels = s.grid_levels;
        self.grid_cache = s.grid_cache;
        // The grid cache was sized with the snapshot's reservation.
        self.movable_radius = s.movable_radius;
        self.manifolds_valid = s.manifolds_valid;
        self.params = s.params;
        self.last_contact_offset = s.last_contact_offset;
        self.vehicles.clone_from(&s.vehicles);
        if let Some(v) = self.vehicles.as_mut() {
            v.dirty = true;
        }
        self.events_seen = s.events_seen;
        self.mount_events_seen = s.mount_events_seen;
        self.beam_events_seen = s.beam_events_seen;
        self.motion_events_seen = s.motion_events_seen;
        if s.soft.is_some() {
            self.soft.clone_from(&s.soft);
        }
        self.free_joints.clone_from(&s.free_joints);
        // The restored state may have bodies awake that the last step did not.
        self.wake_pending = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box dropped on a plane: `restore_state` repeats the steps after a save bit for
    /// bit, the resting contact forces carry the weight, and a body written static by
    /// `write_bodies` holds still and steps again once written back.
    #[test]
    fn host_state_round_trips_and_contact_forces_carry_the_weight() {
        let Ok(mut sim) = WgpuSim::new_with_compact_contacts(WorldCapacity::small(), false) else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let params = AvbdParams {
            gravity_y: -9.81,
            flags: FLAG_CONTACT_GAP | FLAG_WORLD_INERTIA,
            sleep_lin: -1.0,
            sleep_ang: -1.0,
            ..AvbdParams::default()
        };
        sim.set_params(params);
        let ground = sim.add_body(BodyState::static_plane_support());
        sim.add_collider(Collider {
            body: ground,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, 0.0],
        });
        let mass = 8.0;
        let body = sim.add_body(BodyState::dynamic(
            glam::Vec3::new(0.0, 0.8, 0.0),
            glam::Quat::from_rotation_z(0.3),
            mass,
            glam::Vec3::splat(mass * 0.5 / 3.0),
        ));
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 1000.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.5, 0.5, 0.5, 0.0],
        });
        for _ in 0..10 {
            sim.step().unwrap();
        }
        let saved = sim.save_state();
        let run = |sim: &mut WgpuSim| {
            for _ in 0..30 {
                sim.step().unwrap();
            }
            (sim.download_bodies(), sim.last_contacts().unwrap())
        };
        let (b1, c1) = run(&mut sim);
        sim.restore_state(&saved);
        let (b2, c2) = run(&mut sim);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&b1),
            bytemuck::cast_slice::<_, u8>(&b2)
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&c1),
            bytemuck::cast_slice::<_, u8>(&c2)
        );

        // Resting: the forces on the box (B of its plane pairs) add up to its weight.
        for _ in 0..120 {
            sim.step().unwrap();
        }
        let force: glam::Vec3 = sim
            .last_contacts()
            .unwrap()
            .iter()
            .filter(|c| c.alive != 0)
            .map(|c| {
                let f = glam::Vec3::from_array(c.normal) * c.lambda_n
                    + glam::Vec3::from_array(c.tangent1) * c.lambda_t1
                    + glam::Vec3::from_array(c.tangent2) * c.lambda_t2;
                if sim.host_cols[c.collider_b as usize].body == body {
                    -f
                } else {
                    f
                }
            })
            .sum();
        assert!(
            (force.y - mass * 9.81).abs() < 0.02 * mass * 9.81,
            "{force:?}"
        );

        // Static for a few steps: no motion; written back, it falls again.
        let mut s = sim.read_body(body);
        let resting = s;
        s.position[1] = 3.0;
        s.inv_mass = 0.0;
        s.inv_inertia = [0.0; 3];
        s.linear_vel = [0.0; 3];
        s.angular_vel = [0.0; 3];
        sim.write_bodies(body, &[s]);
        for _ in 0..5 {
            sim.step().unwrap();
        }
        assert_eq!(sim.read_body(body).position[1], 3.0);
        sim.write_bodies(
            body,
            &[BodyState {
                position: s.position,
                ..resting
            }],
        );
        sim.step().unwrap();
        assert!(sim.read_body(body).position[1] < 3.0);
    }

    /// Every byte of a buffer (COPY_SRC).
    fn bytes_of_buffer(sim: &WgpuSim, b: &wgpu::Buffer, size: u64) -> Vec<u8> {
        if size == 0 {
            return Vec::new();
        }
        let staging = sim.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = sim.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, size);
        sim.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = sim.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let out = staging.slice(..).get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        out
    }

    /// A compact snapshot restores the same bytes as a full one: saved together after
    /// a pile has settled into contact, the world stepped on (every buffer written),
    /// then restored from the compact one, every buffer equals the full copy. And it
    /// is smaller: most of a small world's capacity is unused.
    #[test]
    fn compact_snapshot_restores_every_byte() {
        let Ok(mut sim) = WgpuSim::new_with_compact_contacts(WorldCapacity::small(), false) else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        sim.set_params(AvbdParams {
            gravity_y: -9.81,
            flags: FLAG_CONTACT_GAP | FLAG_WORLD_INERTIA,
            ..AvbdParams::default()
        });
        let ground = sim.add_body(BodyState::static_plane_support());
        sim.add_collider(Collider {
            body: ground,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, 0.0],
        });
        for k in 0..12 {
            let mass = 2.0;
            let body = sim.add_body(BodyState::dynamic(
                glam::Vec3::new((k % 4) as f32 * 0.6, 0.3 + (k / 4) as f32 * 0.65, 0.0),
                glam::Quat::from_rotation_y(0.1 * k as f32),
                mass,
                glam::Vec3::splat(mass * 0.18 / 3.0),
            ));
            sim.add_collider(Collider {
                body,
                shape: ShapeType::Box as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: 1000.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [0.25, 0.3, 0.25, 0.0],
            });
        }
        for _ in 0..40 {
            sim.step().unwrap();
        }
        let full = sim.save_state();
        let compact = sim.save_state_compact().unwrap();
        let full_bytes: u64 = full.bufs.iter().map(|b| b.1).sum();
        let compact_bytes: u64 = compact.bufs.iter().map(|b| b.1).sum();
        assert!(
            compact_bytes * 2 < full_bytes,
            "compact {compact_bytes} B against full {full_bytes} B"
        );
        for _ in 0..25 {
            sim.step().unwrap();
        }
        sim.restore_state(&compact);
        for (i, (copy, size, _)) in full.bufs.iter().enumerate() {
            if i == 18 {
                continue; // the hull arena is the host's, not restored
            }
            let live = bytes_of_buffer(&sim, &sim.bufs[i], *size);
            let saved = bytes_of_buffer(&sim, copy, *size);
            assert!(live == saved, "binding {i}: restored bytes differ from the full snapshot");
        }

        // Stepping or a new record after the measure refuses the snapshot.
        let m = sim.measure_state();
        sim.step().unwrap();
        assert!(sim.save_state_measured(&m, &vec![0; sim.bufs.len()]).is_err());
    }
}
