//! Host bookkeeping for soft bodies (design step 5): allocates particles,
//! colliders, beams and break groups, merges every vehicle's spawn colouring into
//! the global colour lists (colour k of every vehicle is dispatched together),
//! uploads the `gSoftData` arena and decodes readbacks.

use crate::desc::SoftVehicleDesc;
use crate::layout::*;
use crate::model::Beam;
use crate::pack::{pack_soft, CorePlan, Placement, SpawnOptions};
use engine_schema::{BodyState, Collider, JointDesc, ShapeType};
use engine_vehicle::{VehicleBackend, VehicleHandle, VehicleHost};
use glam::{Quat, Vec3};

pub type Slot = [u32; 4];

/// What a soft host needs from a GPU world (on top of the vehicle backend).
pub trait SoftBackend: VehicleBackend {
    fn write_soft_slots(&mut self, slot: u32, data: &[Slot]);
    fn read_soft_slots(&self, slot: u32, count: u32) -> Vec<Slot>;
    /// Host mirror of an uploaded record.
    fn host_body(&self, index: u32) -> BodyState;
    fn host_collider(&self, index: u32) -> Collider;
    /// Joint record for a descriptor (hosts fill in the default torque arm).
    fn joint_from_desc(&self, desc: &JointDesc) -> engine_schema::Joint {
        desc.to_gpu()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SoftHandle(pub u32);

#[derive(Clone, Debug)]
pub struct SoftEntry {
    pub first_body: u32,
    pub particles: u32,
    pub first_collider: u32,
    pub first_beam: u32,
    pub beams: u32,
    pub color_count: u32,
    pub group: i16,
    pub core: Option<(VehicleHandle, u32)>,
    pub core_joints: std::ops::Range<u32>,
    /// Per rigid part (`desc.rigid_parts`): its body and its joints.
    pub rigid_parts: Vec<(u32, std::ops::Range<u32>)>,
    pub desc: SoftVehicleDesc,
    pub masses: Vec<f32>,
    pub node_collider: Vec<u32>,
    pub colors: Vec<u32>,
    pub adjacency: Vec<Vec<u32>>,
    /// Triangles (relative) touching each node.
    pub tri_adjacency: Vec<Vec<u32>>,
    pub first_tri: u32,
    pub tris: u32,
    /// Ordinal of the first particle across all soft bodies.
    pub ordinal_base: u32,
    /// Beam endpoints (relative node indices), in beam order.
    pub beam_ends: Vec<(u32, u32)>,
    /// Beams known broken (from break events, [`SoftHost::note_breaks`]).
    pub broken: Vec<bool>,
    /// Nodes already reported as part of a detached piece.
    pub detached: Vec<bool>,
    /// The core's [1/m, 1/I xyz] in soft mode (its own share) and in rigid mode
    /// (the whole vehicle), for `soft_mode` to swap.
    pub core_soft_mass: [f32; 4],
    pub core_rigid_mass: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct SoftHost {
    pub capacity: SoftCapacity,
    pub settings: SoftSettings,
    pub entries: Vec<SoftEntry>,
    beams_used: u32,
    groups_used: u32,
    tris_used: u32,
    particles_used: u32,
    /// Thin static obstacles (collider indices) tested against soft triangles.
    obstacles: Vec<u32>,
    /// Steering input per soft body (hydros).
    steer: Vec<f32>,
    pub dirty: bool,
}

/// Per-step dispatch sizes for `engine_exec::rigid_plan`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SoftCounts {
    pub bodies: u32,
    pub beams: u32,
    pub colors: u32,
    /// Largest colour (threads per particle dispatch).
    pub max_color: u32,
    pub groups: u32,
    /// Solved (non-fixed) particles: colour-list entries.
    pub particles: u32,
    /// Every particle (fixed included): ordinals.
    pub particles_total: u32,
    pub triangles: u32,
    pub obstacles: u32,
    /// Soft contact passes run (contacts on and something to collide).
    pub contacts: bool,
    /// The dual pass has work: penalties start below k*.
    pub dual: bool,
    /// Particle sweeps per solver iteration.
    pub sweeps: u32,
}

impl SoftHost {
    pub fn new(capacity: SoftCapacity) -> Self {
        Self {
            capacity,
            settings: SoftSettings::default(),
            entries: Vec::new(),
            beams_used: 0,
            groups_used: 0,
            tris_used: 0,
            particles_used: 0,
            obstacles: Vec::new(),
            steer: Vec::new(),
            dirty: true,
        }
    }

    pub fn counts(&self) -> SoftCounts {
        let colors = self
            .entries
            .iter()
            .map(|e| e.color_count)
            .max()
            .unwrap_or(0);
        let mut per = vec![0u32; colors as usize];
        for e in &self.entries {
            for (i, &c) in e.colors.iter().enumerate() {
                if !self.entry_particle_fixed(e, i) {
                    per[c as usize] += 1;
                }
            }
        }
        SoftCounts {
            bodies: self.entries.len() as u32,
            beams: self.beams_used,
            colors,
            max_color: per.iter().copied().max().unwrap_or(0),
            groups: self.groups_used,
            particles: self.list_len(),
            particles_total: self.particles_used,
            triangles: self.tris_used,
            obstacles: self.obstacles.len() as u32,
            contacts: self.settings.contacts
                && self.tris_used > 0
                && (self.entries.len() > 1 || !self.obstacles.is_empty()),
            dual: self.settings.ramp_start < 1.0,
            sweeps: self.settings.sweeps.max(1),
        }
    }

    fn entry_particle_fixed(&self, e: &SoftEntry, i: usize) -> bool {
        e.desc.nodes[i].fixed
    }

    /// Spawn a soft vehicle: particles, node colliders, beams, then (stage 1) its
    /// rigid core with raycast wheels and the core–frame ball joints.
    pub fn add<B: SoftBackend>(
        &mut self,
        backend: &mut B,
        vehicles: Option<&mut VehicleHost>,
        desc: &SoftVehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
        opts: &SpawnOptions,
    ) -> Result<SoftHandle, String> {
        let s = self.entries.len() as u32;
        if s >= self.capacity.max_bodies.min(255) {
            return Err("soft body capacity exhausted".into());
        }
        if self.beams_used + desc.beams.len() as u32 > self.capacity.max_beams {
            return Err("beam capacity exhausted".into());
        }
        if self.particles_used + desc.nodes.len() as u32 > self.capacity.max_particles {
            return Err("particle capacity exhausted".into());
        }
        if self.tris_used + desc.triangles.len() as u32 > self.capacity.max_triangles {
            return Err("triangle capacity exhausted".into());
        }
        let place = Placement {
            position: Vec3::from_array(position),
            rotation: Quat::from_array(rotation).normalize(),
        };
        let group = -(1000 + s as i32) as i16;
        let first_body = backend.body_count();
        let first_collider = backend.collider_count();
        let mut packed = pack_soft(desc, place, opts, first_body, group);
        if packed.color_count > self.capacity.max_colors {
            return Err(format!(
                "{} colours exceed the soft colour capacity {}",
                packed.color_count, self.capacity.max_colors
            ));
        }
        // Break groups get global ids.
        let mut group_map = std::collections::HashMap::new();
        for b in &mut packed.beams {
            b.body = s;
            b.penalty = (b.spring * self.settings.ramp_start).min(b.spring);
            if b.group != 0 {
                let next = self.groups_used + group_map.len() as u32 + 1;
                b.group = *group_map.entry(b.group).or_insert(next);
            }
        }
        if self.groups_used + group_map.len() as u32 + 1 > self.capacity.max_groups {
            return Err("break group capacity exhausted".into());
        }
        self.groups_used += group_map.len() as u32;
        let mut cols: Vec<Collider> = packed.colliders.clone();
        for c in &mut cols {
            c.body += first_body;
        }
        backend.append_bodies(&packed.bodies);
        backend.append_colliders(&cols);
        let first_beam = self.beams_used;
        // Beams: append-only region (their plastic state lives on the GPU).
        let l = self.capacity.layout();
        let words: Vec<Slot> = packed
            .beams
            .iter()
            .flat_map(|b| {
                let w = b.to_words();
                [
                    [w[0], w[1], w[2], w[3]],
                    [w[4], w[5], w[6], w[7]],
                    [w[8], w[9], w[10], w[11]],
                    [w[12], w[13], w[14], w[15]],
                ]
            })
            .collect();
        backend.write_soft_slots(l.beam_base + BEAM_STRIDE * first_beam, &words);
        self.beams_used += packed.beams.len() as u32;
        // Triangles (static): global particle indices, owner and collide flag.
        let first_tri = self.tris_used;
        let mut tri_adjacency = vec![Vec::new(); desc.nodes.len()];
        let tri_slots: Vec<Slot> = desc
            .triangles
            .iter()
            .enumerate()
            .map(|(t, tri)| {
                for &n in &tri.nodes {
                    tri_adjacency[n as usize].push(t as u32);
                }
                [
                    first_body + tri.nodes[0],
                    first_body + tri.nodes[1],
                    first_body + tri.nodes[2],
                    s | (u32::from(tri.collision) << 8),
                ]
            })
            .collect();
        if !tri_slots.is_empty() {
            backend.write_soft_slots(l.tri_base + first_tri, &tri_slots);
            let rest: Vec<Slot> = desc
                .triangles
                .iter()
                .map(|tri| {
                    let p = |i: u32| Vec3::from_array(desc.nodes[i as usize].position);
                    let [a, b, c] = tri.nodes.map(p);
                    [
                        (a - b).length().to_bits(),
                        (b - c).length().to_bits(),
                        (c - a).length().to_bits(),
                        0,
                    ]
                })
                .collect();
            backend.write_soft_slots(l.tri_rest_base + first_tri, &rest);
        }
        self.tris_used += tri_slots.len() as u32;
        let ordinal_base = self.particles_used;
        self.particles_used += desc.nodes.len() as u32;
        // Node contact slots start empty (fixed particles are never rewritten).
        let none = [[0u32, 0, 0, SOFT_NONE], [0, 0, 0, 0], [0, 0, 0, 0]];
        let slots: Vec<Slot> = (0..desc.nodes.len()).flat_map(|_| none).collect();
        backend.write_soft_slots(l.node_slot_base + NODE_SLOT_STRIDE * ordinal_base, &slots);

        // Rigid core (stage 1): wheels and drivetrain on a frame proxy.
        let mut core = None;
        let mut core_joints = 0..0;
        let mut rigid_props: Option<([f32; 4], [f32; 4])> = None;
        if let (Some(plan), Some(vh)) = (packed.core.clone(), vehicles) {
            let CorePlan {
                vehicle,
                origin,
                frame_nodes,
            } = plan;
            let world_origin = place.apply(origin);
            let cols_before = backend.collider_count();
            let h = vh
                .add_vehicle(
                    backend,
                    &vehicle,
                    world_origin.to_array(),
                    place.rotation.to_array(),
                )
                .map_err(|e| format!("rigid core: {e:?}"))?;
            let chassis = vh.chassis_body(h);
            // The core's colliders join the vehicle's no-self-collision group.
            for ci in cols_before..backend.collider_count() {
                let mut c = backend.host_collider(ci);
                engine_schema::convex::CollisionFilter {
                    group,
                    ..Default::default()
                }
                .apply(&mut c);
                backend.write_collider(ci, &c);
            }
            if opts.velocity != [0.0; 3] {
                let mut b = backend.host_body(chassis);
                b.linear_vel = opts.velocity;
                backend.write_body(chassis, &b);
            }
            let mut j = Vec::new();
            for &i in &frame_nodes {
                let local = Vec3::from_array(desc.nodes[i as usize].position) - origin;
                let jd = JointDesc::ball(chassis, first_body + i, local, Vec3::ZERO);
                j.push(backend.joint_from_desc(&jd));
            }
            let start = backend.joint_count();
            backend.append_joints(&j);
            core_joints = start..backend.joint_count();
            // Mass properties of the core alone (soft mode) and of the whole
            // vehicle about the core (rigid mode), in the core's frame.
            let cb = backend.host_body(chassis);
            let cpos = Vec3::from_array(cb.position);
            let cq = Quat::from_array(cb.orientation).inverse();
            let core_m = if cb.inv_mass > 0.0 {
                1.0 / cb.inv_mass
            } else {
                0.0
            };
            let mut inertia =
                Vec3::from_array(cb.inv_inertia).map(|x| if x > 0.0 { 1.0 / x } else { 0.0 });
            let mut total = core_m;
            for (k, n) in desc.nodes.iter().enumerate() {
                let m = packed.masses.get(k).copied().unwrap_or(0.0);
                if n.fixed || m <= 0.0 {
                    continue;
                }
                let r = cq * (place.apply(Vec3::from_array(n.position)) - cpos);
                inertia += m * Vec3::new(
                    r.y * r.y + r.z * r.z,
                    r.x * r.x + r.z * r.z,
                    r.x * r.x + r.y * r.y,
                );
                total += m;
            }
            rigid_props = Some((
                [
                    cb.inv_mass,
                    cb.inv_inertia[0],
                    cb.inv_inertia[1],
                    cb.inv_inertia[2],
                ],
                [
                    1.0 / total.max(1e-6),
                    1.0 / inertia.x.max(1e-6),
                    1.0 / inertia.y.max(1e-6),
                    1.0 / inertia.z.max(1e-6),
                ],
            ));
            core = Some((h, chassis));
        }

        // Rigid parts: free bodies held to their nodes by breakable ball joints.
        let mut rigid_parts = Vec::new();
        for part in &desc.rigid_parts {
            let o = part.center();
            let mut b = BodyState::dynamic(place.apply(o), place.rotation, part.mass, part.inertia());
            b.linear_vel = opts.velocity;
            b.initial_pos = b.position;
            b.initial_rot = b.orientation;
            let body = backend.body_count();
            backend.append_bodies(&[b]);
            let cols: Vec<Collider> = part
                .colliders
                .iter()
                .map(|c| {
                    let (shape, params) = match c.shape {
                        engine_vehicle::ShapeDesc::Box { half } => (ShapeType::Box, [half[0], half[1], half[2], Vec3::from_array(half).length()]),
                        engine_vehicle::ShapeDesc::Sphere { radius } => (ShapeType::Sphere, [radius, 0.0, 0.0, radius]),
                        engine_vehicle::ShapeDesc::Capsule { radius, half_length } => {
                            (ShapeType::Capsule, [radius, half_length, 0.0, radius + half_length])
                        }
                    };
                    let mut col = Collider {
                        body,
                        shape: shape as u32,
                        material: c.material,
                        generation: 0,
                        local_pos: (Vec3::from_array(c.position) - o).to_array(),
                        density: 0.0,
                        local_rot: c.rotation,
                        params,
                    };
                    // No contact with its own vehicle's nodes, core or parts: the joints hold it.
                    engine_schema::convex::CollisionFilter { group, ..Default::default() }.apply(&mut col);
                    col
                })
                .collect();
            backend.append_colliders(&cols);
            let joints: Vec<engine_schema::Joint> = part
                .attach
                .iter()
                .map(|a| {
                    let local = Vec3::from_array(desc.nodes[a.node as usize].position) - o;
                    let jd = JointDesc::ball(body, first_body + a.node, local, Vec3::ZERO).with_break(a.break_force, a.break_torque);
                    backend.joint_from_desc(&jd)
                })
                .collect();
            let start = backend.joint_count();
            backend.append_joints(&joints);
            rigid_parts.push((body, start..backend.joint_count()));
        }

        let beam_ends: Vec<(u32, u32)> = packed
            .beams
            .iter()
            .map(|b| (b.a - first_body, b.b - first_body))
            .collect();
        let n_beams = beam_ends.len();
        self.entries.push(SoftEntry {
            beam_ends,
            broken: vec![false; n_beams],
            detached: vec![false; desc.nodes.len()],
            core_soft_mass: rigid_props.map_or([0.0; 4], |r| r.0),
            core_rigid_mass: rigid_props.map_or([0.0; 4], |r| r.1),
            first_body,
            particles: desc.nodes.len() as u32,
            first_collider,
            first_beam,
            beams: packed.beams.len() as u32,
            color_count: packed.color_count,
            group,
            core,
            core_joints,
            rigid_parts,
            desc: desc.clone(),
            masses: packed.masses,
            node_collider: packed.node_collider,
            colors: packed.colors,
            adjacency: packed.adjacency,
            tri_adjacency,
            first_tri,
            tris: desc.triangles.len() as u32,
            ordinal_base,
        });
        self.steer.push(0.0);
        self.upload_topology(backend);
        self.write_body_mode(backend, s);
        // Rest positions (desc frame), second slot of each particle's rigid pair.
        let ord = self.entries[s as usize].ordinal_base;
        for (k, n) in desc.nodes.iter().enumerate() {
            let p = n.position;
            backend.write_soft_slots(
                l.rigid_base + 2 * (ord + k as u32) + 1,
                &[[p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), 0]],
            );
        }
        Ok(SoftHandle(s))
    }

    pub fn set_steer(&mut self, backend: &mut impl SoftBackend, h: SoftHandle, steer: f32) {
        self.steer[h.0 as usize] = steer.clamp(-1.0, 1.0);
        self.write_body_record(backend, h.0);
    }

    pub fn set_settings(&mut self, backend: &mut impl SoftBackend, settings: SoftSettings) {
        self.settings = settings;
        self.write_header(backend);
    }

    fn write_body_record(&self, backend: &mut impl SoftBackend, s: u32) {
        let e = &self.entries[s as usize];
        let l = self.capacity.layout();
        let rec = [
            [e.first_body, e.particles, e.first_beam, e.beams],
            [
                self.steer[s as usize].to_bits(),
                e.group as i32 as u32,
                e.core.map_or(SOFT_NONE, |c| c.1),
                e.ordinal_base,
            ],
            [
                e.first_tri,
                e.tris,
                e.desc.collision_range.max(0.005).to_bits(),
                0,
            ],
        ];
        backend.write_soft_slots(l.body_base + BODY_STRIDE * s, &rec);
    }

    /// Rigid-mode state (spawn only: `soft_mode` owns the mode and quiet count).
    fn write_body_mode(&self, backend: &mut impl SoftBackend, s: u32) {
        let e = &self.entries[s as usize];
        let l = self.capacity.layout();
        let has_core = u32::from(e.core.is_some());
        let joints = e.core_joints.end - e.core_joints.start;
        let rec = [
            [
                0,
                0,
                has_core
                    | (joints.min(255) << 8)
                    | (((self.settings.rigid_quiet * 1000.0) as u32).min(65535) << 16),
                e.core_joints.start,
            ],
            e.core_soft_mass.map(f32::to_bits),
            e.core_rigid_mass.map(f32::to_bits),
        ];
        backend.write_soft_slots(l.body_base + BODY_STRIDE * s + 3, &rec);
    }

    fn write_header(&self, backend: &mut impl SoftBackend) {
        let l = self.capacity.layout();
        let c = self.counts();
        let st = &self.settings;
        let flags = u32::from(st.plasticity) * SOFT_PLASTICITY
            | u32::from(st.breaking) * SOFT_BREAKING
            | u32::from(st.sleep_sync) * SOFT_SLEEP_SYNC
            | u32::from(c.contacts) * SOFT_CONTACTS
            | u32::from(st.rigid_mode) * SOFT_RIGID_MODE;
        let list_count: u32 = self.list_len();
        let header: [Slot; 9] = [
            [c.bodies, c.beams, c.colors, list_count],
            [l.body_base, l.beam_base, l.list_base, l.adj_base],
            [l.color_base, l.group_base, c.groups + 1, flags],
            [
                st.damping_scale.to_bits(),
                st.beta.to_bits(),
                st.min_length.to_bits(),
                l.rigid_base,
            ],
            [
                st.threshold_scale.to_bits(),
                st.ramp_start.to_bits(),
                st.chebyshev_rho.to_bits(),
                st.contact_stiffness.to_bits(),
            ],
            [l.cheb_base, l.tri_base, c.triangles, l.tri_contact_base],
            [l.node_slot_base, l.flag_base, l.obstacle_base, c.obstacles],
            [
                l.aabb_base,
                c.particles_total,
                st.contact_friction.to_bits(),
                l.tri_rest_base,
            ],
            [l.body_color_base, l.body_color_stride, 0, 0],
        ];
        backend.write_soft_slots(0, &header);
    }

    fn list_len(&self) -> u32 {
        self.entries
            .iter()
            .map(|e| e.desc.nodes.iter().filter(|n| !n.fixed).count() as u32)
            .sum()
    }

    /// Rebuild the colour lists and adjacency of every soft body (static topology;
    /// broken beams are skipped by the kernels) and the header.
    fn upload_topology(&mut self, backend: &mut impl SoftBackend) {
        let l = self.capacity.layout();
        let counts = self.counts();
        let mut lists: Vec<Vec<Slot>> = vec![Vec::new(); counts.colors as usize];
        let mut adj: Vec<u32> = Vec::new();
        for (s, e) in self.entries.iter().enumerate() {
            for (i, &c) in e.colors.iter().enumerate() {
                if e.desc.nodes[i].fixed {
                    continue;
                }
                let start = adj.len() as u32;
                for &b in &e.adjacency[i] {
                    adj.push(e.first_beam + b);
                }
                for &t in &e.tri_adjacency[i] {
                    adj.push(e.first_tri + t);
                }
                let counts = e.adjacency[i].len() as u32 | (e.tri_adjacency[i].len() as u32) << 16;
                let flags = if e.desc.nodes[i].contacter && e.desc.nodes[i].collision {
                    ENTRY_CONTACTER
                } else {
                    0
                };
                lists[c as usize].push([e.first_body + i as u32, start, counts, s as u32 | flags]);
            }
        }
        let mut color_slots = Vec::with_capacity(lists.len());
        let mut flat = Vec::new();
        for list in &lists {
            color_slots.push([flat.len() as u32, list.len() as u32, 0, 0]);
            flat.extend_from_slice(list);
        }
        if !color_slots.is_empty() {
            backend.write_soft_slots(l.color_base, &color_slots);
        }
        // Each body's range in every colour list (entries are in body order).
        let stride = l.body_color_stride as usize;
        let mut ranges = vec![[0u32; 4]; self.entries.len() * stride];
        for (c, list) in lists.iter().enumerate() {
            let base = color_slots[c][0];
            for (k, entry) in list.iter().enumerate() {
                let s = (entry[3] & 0xFFFF) as usize;
                let r = &mut ranges[s * stride + c / 2][2 * (c % 2)..2 * (c % 2) + 2];
                if r[1] == 0 {
                    r[0] = base + k as u32;
                }
                r[1] += 1;
            }
        }
        if !ranges.is_empty() {
            backend.write_soft_slots(l.body_color_base, &ranges);
        }
        if !flat.is_empty() {
            backend.write_soft_slots(l.list_base, &flat);
        }
        while adj.len() % 4 != 0 {
            adj.push(0);
        }
        let adj_slots: Vec<Slot> = adj.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        if !adj_slots.is_empty() {
            backend.write_soft_slots(l.adj_base, &adj_slots);
        }
        // Break-group flags start clear.
        let groups = vec![[0u32; 4]; (counts.groups + 1) as usize];
        backend.write_soft_slots(l.group_base, &groups);
        for s in 0..self.entries.len() as u32 {
            self.write_body_record(backend, s);
        }
        self.write_header(backend);
        self.dirty = true;
    }

    /// Register an obstacle (sphere, capsule or box collider) that soft triangles
    /// collide with: poles and bollards narrower than the node spacing, or a rigid
    /// ram whose edges would slip between nodes. One-way: the triangles are pushed,
    /// the obstacle's own reaction comes through the node spheres.
    pub fn add_obstacle(
        &mut self,
        backend: &mut impl SoftBackend,
        collider: u32,
    ) -> Result<(), String> {
        if self.obstacles.len() as u32 >= self.capacity.max_obstacles {
            return Err("obstacle capacity exhausted".into());
        }
        let l = self.capacity.layout();
        backend.write_soft_slots(
            l.obstacle_base + self.obstacles.len() as u32,
            &[[collider, 0, 0, 0]],
        );
        self.obstacles.push(collider);
        self.write_header(backend);
        Ok(())
    }

    /// Current beam records of a soft body (GPU readback).
    pub fn read_beams(&self, backend: &impl SoftBackend, h: SoftHandle) -> Vec<Beam> {
        let e = &self.entries[h.0 as usize];
        let l = self.capacity.layout();
        let slots = backend.read_soft_slots(
            l.beam_base + BEAM_STRIDE * e.first_beam,
            BEAM_STRIDE * e.beams,
        );
        slots
            .chunks(4)
            .map(|c| {
                let mut w = [0u32; 16];
                for (k, s) in c.iter().enumerate() {
                    w[k * 4..k * 4 + 4].copy_from_slice(s);
                }
                Beam::from_words(&w)
            })
            .collect()
    }

    pub fn entry(&self, h: SoftHandle) -> &SoftEntry {
        &self.entries[h.0 as usize]
    }

    /// Bind a render mesh to soft body `h`: each vertex (in the vehicle's own
    /// frame, the desc's node coordinates) gets a frame of three nodes, from the
    /// part `part` when given (so a door's mesh follows only the door's nodes and
    /// leaves with it). See [`SkinBinding`] for the skinning formula.
    pub fn bind_mesh(
        &self,
        h: SoftHandle,
        vertices: &[Vec3],
        part: Option<&str>,
    ) -> Result<Vec<SkinBinding>, String> {
        let e = self.entries.get(h.0 as usize).ok_or("no such soft body")?;
        let nodes: Vec<Vec3> = e
            .desc
            .nodes
            .iter()
            .map(|n| Vec3::from_array(n.position))
            .collect();
        let pool: Option<Vec<u32>> = match part {
            Some(name) => Some(
                e.desc
                    .part(name)
                    .ok_or_else(|| format!("no part `{name}`"))?
                    .nodes
                    .clone(),
            ),
            None => None,
        };
        let opts = crate::flexbody::BindOptions::default();
        let got = crate::flexbody::bind(vertices, &nodes, pool.as_deref(), opts);
        Ok(vertices
            .iter()
            .zip(got)
            .map(|(v, b)| {
                // Fall back to any nodes, then to the nearest node alone.
                let b = b.or_else(|| {
                    crate::flexbody::bind(std::slice::from_ref(v), &nodes, None, opts)[0]
                });
                match b {
                    Some(b) => {
                        let [c, x, y] = b.nodes.map(|i| nodes[i as usize]);
                        SkinBinding {
                            nodes: b.nodes.map(|i| e.first_body + i),
                            coords: b.coords,
                            rest_edge: (x - c).length().max((y - c).length()),
                        }
                    }
                    None => {
                        let c = pool
                            .as_deref()
                            .unwrap_or(&[])
                            .iter()
                            .copied()
                            .chain(0..nodes.len() as u32)
                            .min_by(|&a, &b| {
                                (nodes[a as usize] - *v)
                                    .length_squared()
                                    .total_cmp(&(nodes[b as usize] - *v).length_squared())
                            })
                            .unwrap_or(0);
                        SkinBinding {
                            nodes: [e.first_body + c; 3],
                            coords: [0.0; 3],
                            rest_edge: 0.0,
                        }
                    }
                }
            })
            .collect())
    }

    /// Record beam breaks (from the event ring, e.g. `WgpuSim::poll_beam_breaks`)
    /// and return the pieces they cut loose: groups of nodes no longer joined by
    /// any unbroken beam to the body's main part (the one holding the most mass).
    /// Each piece is reported once.
    pub fn note_breaks(&mut self, breaks: &[BeamBreak]) -> Vec<PieceDetached> {
        let mut touched = Vec::new();
        for b in breaks {
            let Some(e) = self.entries.get_mut(b.soft_body as usize) else {
                continue;
            };
            let local = b.beam.wrapping_sub(e.first_beam) as usize;
            if local < e.broken.len() && !e.broken[local] {
                e.broken[local] = true;
                if !touched.contains(&b.soft_body) {
                    touched.push(b.soft_body);
                }
            }
        }
        let mut out = Vec::new();
        for s in touched {
            out.extend(self.new_pieces(SoftHandle(s)));
        }
        out
    }

    /// Replace a body's known broken set from a beam readback (after the event
    /// ring overflowed, or to resynchronise), returning newly detached pieces.
    pub fn sync_broken(&mut self, h: SoftHandle, beams: &[Beam]) -> Vec<PieceDetached> {
        let Some(e) = self.entries.get_mut(h.0 as usize) else {
            return Vec::new();
        };
        for (k, b) in beams.iter().enumerate().take(e.broken.len()) {
            e.broken[k] = b.broken();
        }
        self.new_pieces(h)
    }

    fn new_pieces(&mut self, h: SoftHandle) -> Vec<PieceDetached> {
        let e = &mut self.entries[h.0 as usize];
        let n = e.desc.nodes.len();
        let mut label = vec![u32::MAX; n];
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (k, &(a, b)) in e.beam_ends.iter().enumerate() {
            if !e.broken[k] && (a as usize) < n && (b as usize) < n {
                adj[a as usize].push(b);
                adj[b as usize].push(a);
            }
        }
        let mut comps: Vec<Vec<u32>> = Vec::new();
        for start in 0..n {
            if label[start] != u32::MAX {
                continue;
            }
            let id = comps.len() as u32;
            let mut stack = vec![start as u32];
            label[start] = id;
            let mut nodes = Vec::new();
            while let Some(v) = stack.pop() {
                nodes.push(v);
                for &w in &adj[v as usize] {
                    if label[w as usize] == u32::MAX {
                        label[w as usize] = id;
                        stack.push(w);
                    }
                }
            }
            comps.push(nodes);
        }
        let mass = |c: &Vec<u32>| {
            c.iter()
                .map(|&i| e.masses.get(i as usize).copied().unwrap_or(0.0))
                .sum::<f32>()
        };
        let main = (0..comps.len())
            .max_by(|&a, &b| mass(&comps[a]).total_cmp(&mass(&comps[b])))
            .unwrap_or(0);
        let mut out = Vec::new();
        for (ci, c) in comps.iter().enumerate() {
            if ci == main || c.iter().all(|&i| e.detached[i as usize]) {
                continue;
            }
            for &i in c {
                e.detached[i as usize] = true;
            }
            let mut nodes = c.clone();
            nodes.sort_unstable();
            // Parts mostly inside the piece.
            let parts = e
                .desc
                .parts
                .iter()
                .filter(|p| {
                    !p.nodes.is_empty()
                        && p.nodes
                            .iter()
                            .filter(|i| nodes.binary_search(i).is_ok())
                            .count()
                            * 2
                            > p.nodes.len()
                })
                .map(|p| p.name.clone())
                .collect();
            out.push(PieceDetached {
                soft_body: h,
                mass: mass(c),
                nodes,
                parts,
            });
        }
        out
    }
}

/// A render-mesh vertex bound to a frame of three particles (global body
/// indices: rows of the body buffer). With c, x, y their current positions,
/// e_x = x − c, e_y = y − c:
///
/// p = c + coords[0]·e_x + coords[1]·e_y + coords[2]·(e_x × e_y)
///
/// The frame's longest edge at rest is `rest_edge`; a frame stretched past
/// 1.8 × `rest_edge` is torn (drop triangles that use the vertex).
/// `rest_edge` = 0 marks a vertex pinned to one node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinBinding {
    pub nodes: [u32; 3],
    pub coords: [f32; 3],
    pub rest_edge: f32,
}

impl SkinBinding {
    /// GPU record (`soft_skin.slang`): c, x, y, flags; coords and rest edge (f32 bits).
    pub fn to_words(&self, flags: u32) -> [u32; 8] {
        [
            self.nodes[0],
            self.nodes[1],
            self.nodes[2],
            flags,
            self.coords[0].to_bits(),
            self.coords[1].to_bits(),
            self.coords[2].to_bits(),
            self.rest_edge.to_bits(),
        ]
    }

    /// CPU twin of the kernel: the vertex for body positions `pos(i)`, and
    /// whether its frame is torn.
    pub fn skin(&self, pos: impl Fn(u32) -> Vec3) -> (Vec3, bool) {
        let c = pos(self.nodes[0]);
        let ex = pos(self.nodes[1]) - c;
        let ey = pos(self.nodes[2]) - c;
        let torn = self.rest_edge > 0.0 && ex.length().max(ey.length()) > 1.8 * self.rest_edge;
        (
            c + ex * self.coords[0] + ey * self.coords[1] + ex.cross(ey) * self.coords[2],
            torn,
        )
    }
}

/// Nodes of a soft body cut loose from its main part by beam breaks.
#[derive(Clone, Debug, PartialEq)]
pub struct PieceDetached {
    pub soft_body: SoftHandle,
    /// Relative node indices (sorted).
    pub nodes: Vec<u32>,
    /// Names of the parts more than half inside the piece (a door, a bumper).
    pub parts: Vec<String>,
    pub mass: f32,
}

/// A beam that broke (from the shared event ring, `EVENT_BEAM_BREAK` records).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeamBreak {
    pub beam: u32,
    pub force: f32,
    pub soft_body: u32,
}

/// Decode beam breaks from the event ring words (cursor, then 4-word records)
/// since `seen`; returns how many records were overwritten before being read.
pub fn decode_beam_breaks(
    words: &[u32],
    seen: &mut u32,
    capacity: u32,
    out: &mut Vec<BeamBreak>,
) -> u32 {
    let cursor = words[0];
    let lost = cursor.wrapping_sub(*seen).saturating_sub(capacity);
    let mut n = seen.wrapping_add(lost);
    while n != cursor {
        let r = 4 + 4 * (n % capacity) as usize;
        if words[r] == EVENT_BEAM_BREAK {
            out.push(BeamBreak {
                beam: words[r + 1],
                force: f32::from_bits(words[r + 2]),
                soft_body: words[r + 3],
            });
        }
        n = n.wrapping_add(1);
    }
    *seen = cursor;
    lost
}
