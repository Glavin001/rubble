//! Host access to the simulation state, for hosts that couple the engine to
//! another solver (the stress oracle's `RigidEngine`, `stress-rubble`): write body
//! state back, read the last step's contact forces, snapshot and restore the whole
//! simulation. All additive: a host that never calls these steps exactly as before.

use super::*;

/// Every GPU buffer and the host mirror of a [`WgpuSim`] at one instant
/// ([`WgpuSim::save_state`]).
pub struct SimSnapshot {
    bufs: Vec<(wgpu::Buffer, u64)>,
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
    has_filters: bool,
    grid_levels: u32,
    grid_cache: Option<(u32, f32, u32)>,
    manifolds_valid: bool,
    params: AvbdParams,
    last_contact_offset: u32,
    vehicles: Option<engine_vehicle::VehicleHost>,
    events_seen: u32,
    free_joints: Vec<u32>,
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

    pub fn counters_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[7]
    }

    /// First row of the last step's contacts in `contact_buffer` (the ping-pong half).
    pub fn last_contact_offset(&self) -> u32 {
        self.last_contact_offset
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
        self.queue.write_buffer(&self.bufs[0], u64::from(first) * stride, bytes);
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
        let n = self.last_counters.contact_count.min(self.capacity.max_contacts);
        Ok(self.read_buffer_range(
            4,
            u64::from(self.last_contact_offset) * std::mem::size_of::<Contact>() as u64,
            n as usize,
        ))
    }

    /// Copy every buffer and the host mirror (bodies, colliders, counters, contact
    /// ping-pong, warm-start state): `restore_state` returns the simulation to this
    /// instant exactly. The hull arena (binding 18) is the host's: a host that
    /// inserts hulls between save and restore keeps them.
    pub fn save_state(&self) -> SimSnapshot {
        self.save_into(Vec::new())
    }

    /// `save_state` into an older snapshot's GPU buffers where their sizes still fit
    /// (no allocation per save for a host that saves every frame).
    pub fn save_state_reusing(&self, old: SimSnapshot) -> SimSnapshot {
        self.save_into(old.bufs)
    }

    fn save_into(&self, old: Vec<(wgpu::Buffer, u64)>) -> SimSnapshot {
        let mut old = old.into_iter();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("save-state"),
        });
        let bufs = self
            .bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let size = b.size();
                let copy = match old.next() {
                    Some((buf, s)) if s == size => buf,
                    _ => self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("snapshot-b{i}")),
                        size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                };
                enc.copy_buffer_to_buffer(b, 0, &copy, 0, size);
                (copy, size)
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
            has_filters: self.has_filters,
            grid_levels: self.grid_levels,
            grid_cache: self.grid_cache,
            manifolds_valid: self.manifolds_valid,
            params: self.params,
            last_contact_offset: self.last_contact_offset,
            vehicles: self.vehicles.clone(),
            events_seen: self.events_seen,
            free_joints: self.free_joints.clone(),
        }
    }

    /// Return to a snapshot of this simulation (`save_state`).
    pub fn restore_state(&mut self, s: &SimSnapshot) {
        assert_eq!(s.bufs.len(), self.bufs.len(), "snapshot of another simulation");
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("restore-state"),
        });
        for (i, (copy, size)) in s.bufs.iter().enumerate() {
            if i == 18 {
                continue; // the hull arena stays the host's
            }
            enc.copy_buffer_to_buffer(copy, 0, &self.bufs[i], 0, (*size).min(self.bufs[i].size()));
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
        self.has_filters = s.has_filters;
        self.grid_levels = s.grid_levels;
        self.grid_cache = s.grid_cache;
        self.manifolds_valid = s.manifolds_valid;
        self.params = s.params;
        self.last_contact_offset = s.last_contact_offset;
        self.vehicles.clone_from(&s.vehicles);
        if let Some(v) = self.vehicles.as_mut() {
            v.dirty = true;
        }
        self.events_seen = s.events_seen;
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
        assert_eq!(bytemuck::cast_slice::<_, u8>(&b1), bytemuck::cast_slice::<_, u8>(&b2));
        assert_eq!(bytemuck::cast_slice::<_, u8>(&c1), bytemuck::cast_slice::<_, u8>(&c2));

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
                if sim.host_cols[c.collider_b as usize].body == body { -f } else { f }
            })
            .sum();
        assert!((force.y - mass * 9.81).abs() < 0.02 * mass * 9.81, "{force:?}");

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
        sim.write_bodies(body, &[BodyState { position: s.position, ..resting }]);
        sim.step().unwrap();
        assert!(sim.read_body(body).position[1] < 3.0);
    }
}
