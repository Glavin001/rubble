//! Backend-agnostic vehicle bookkeeping: pool allocation, the host-owned
//! control slots (V_IN0..V_IN3), and decoding of GPU outputs.

use crate::desc::*;
use crate::pack::{self, PackedVehicle, Slot, VehicleCapacity, VehicleSlots};
use engine_schema::vehicle::*;
use engine_schema::{BodyState, Collider};
use glam::{Quat, Vec3};

/// What a GPU host provides to the vehicle system.
pub trait VehicleBackend {
    fn body_count(&self) -> u32;
    fn collider_count(&self) -> u32;
    fn append_bodies(&mut self, bodies: &[BodyState]);
    fn append_colliders(&mut self, colliders: &[Collider]);
    /// Overwrite records already uploaded (parked bodies / rims of spare wheels).
    fn write_body(&mut self, index: u32, body: &BodyState);
    fn write_collider(&mut self, index: u32, collider: &Collider);
    /// Write float4 slots into gVehicleData starting at `slot`.
    fn write_slots(&mut self, slot: u32, data: &[Slot]);
    /// Write curve points (8 bytes each) at point index `point` of the curve region.
    fn write_curve_points(&mut self, byte_offset: u64, points: &[[f32; 2]]);
    fn read_slots(&self, slot: u32, count: u32) -> Vec<Slot>;
    fn joint_count(&self) -> u32;
    /// Append joints (vehicle tyre joints) after the existing ones.
    fn append_joints(&mut self, joints: &[engine_schema::Joint]);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VehicleHandle(pub u32);

#[derive(Clone, Debug)]
struct Entry {
    slots: VehicleSlots,
    wheel_slots: u32,
    wheels_used: u32,
    desc: VehicleDesc,
    controls: [Slot; 4],
    chassis: u32,
    first_part_body: u32,
    /// Design-space position of the body origin (centre of mass).
    origin: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WheelState {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub suspension_length: f32,
    /// Mount force/torque from the ground on the chassis (chassis frame).
    pub mount_force: [f32; 3],
    pub mount_torque: [f32; 3],
    pub suspension_force: f32,
    pub longitudinal_force: f32,
    pub lateral_force: f32,
    pub contact_point: [f32; 3],
    pub contact_normal: [f32; 3],
    pub ground_body: Option<u32>,
    pub omega: f32,
    pub slip_ratio: f32,
    pub slip_angle_deg: f32,
    pub in_contact: bool,
    pub sliding: bool,
    pub detached: bool,
    pub disabled: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VehicleState {
    pub forward_speed: f32,
    pub lateral_speed: f32,
    pub yaw_rate: f32,
    pub engine_rpm: f32,
    pub gear: i32,
    pub clutch: f32,
    pub contacts: u32,
    pub wheel_mask: u32,
    pub drive_mask: u32,
    pub steer_mask: u32,
    pub engine_connected: bool,
    pub parts_lost: u32,
    /// Increments whenever the functional state changes (severing, masks).
    pub event_seq: u32,
    pub wheels: Vec<WheelState>,
}

/// Per-wheel impulse wrench at its mount over the last step (chassis frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WheelLoad {
    pub vehicle: VehicleHandle,
    pub wheel: u32,
    pub impulse: [f32; 3],
    pub angular_impulse: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
    pub max_dist: f32,
    /// Body to ignore (e.g. the caster's own chassis).
    pub exclude_body: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub distance: f32,
    pub collider: u32,
}

#[derive(Clone, Debug)]
pub struct VehicleHost {
    pub capacity: VehicleCapacity,
    pub aabb_tail: u32,
    entries: Vec<Entry>,
    wheels_used: u32,
    curve_points_used: u32,
    diffs_used: u32,
    anti_rolls_used: u32,
    parts_used: u32,
    rays: u32,
    /// Bumped on every host write; WGSL hosts must not skip an all-asleep step while dirty.
    pub dirty: bool,
    /// Last state seen by `poll_events`, per vehicle.
    seen: Vec<Option<VehicleState>>,
    /// Wheel probes scan every collider instead of the broadphase grid (oracle).
    pub full_scan_queries: bool,
}

/// Changes observed between two `poll_events` calls.
#[derive(Clone, Debug, PartialEq)]
pub enum VehicleEvent {
    /// The wheel became a free rigid body (`body`).
    WheelDetached { vehicle: VehicleHandle, wheel: u32, body: u32 },
    /// The wheel was removed without a body.
    WheelRemoved { vehicle: VehicleHandle, wheel: u32 },
    /// A part left the chassis: severed into `body`, disabled, or reparented elsewhere.
    PartLost { vehicle: VehicleHandle, part: u32, body: u32 },
    /// Wheel / drive / steer masks or engine connection changed.
    FunctionalStateChanged { vehicle: VehicleHandle, wheel_mask: u32, drive_mask: u32, steer_mask: u32, engine_connected: bool },
    WheelContactBegin { vehicle: VehicleHandle, wheel: u32, ground_body: Option<u32> },
    WheelContactEnd { vehicle: VehicleHandle, wheel: u32 },
    /// Tyre force saturated (skid marks, smoke, audio).
    WheelSlideBegin { vehicle: VehicleHandle, wheel: u32 },
    WheelSlideEnd { vehicle: VehicleHandle, wheel: u32 },
}

/// Vehicle half of a world snapshot (the host saves bodies and colliders itself).
#[derive(Clone, Debug)]
pub struct VehicleSnapshot {
    data: Vec<Slot>,
    host: VehicleHost,
}

fn fbits(v: f32) -> u32 {
    v.to_bits()
}

fn f3(s: Slot) -> [f32; 3] {
    [f32::from_bits(s[0]), f32::from_bits(s[1]), f32::from_bits(s[2])]
}

fn fw(s: Slot) -> f32 {
    f32::from_bits(s[3])
}

impl VehicleHost {
    pub fn new(capacity: VehicleCapacity, aabb_tail: u32) -> Self {
        Self {
            capacity,
            aabb_tail,
            entries: Vec::new(),
            wheels_used: 0,
            curve_points_used: 0,
            diffs_used: 0,
            anti_rolls_used: 0,
            parts_used: 0,
            rays: 0,
            dirty: true,
            seen: Vec::new(),
            full_scan_queries: false,
        }
    }

    /// Diff GPU outputs against the previous call (call after a completed step).
    pub fn poll_events(&mut self, backend: &impl VehicleBackend) -> Vec<VehicleEvent> {
        let mut out = Vec::new();
        self.seen.resize(self.entries.len(), None);
        for i in 0..self.entries.len() {
            let h = VehicleHandle(i as u32);
            let now = self.read_state(backend, h);
            if let Some(prev) = &self.seen[i] {
                if (prev.wheel_mask, prev.drive_mask, prev.steer_mask, prev.engine_connected)
                    != (now.wheel_mask, now.drive_mask, now.steer_mask, now.engine_connected)
                {
                    out.push(VehicleEvent::FunctionalStateChanged {
                        vehicle: h,
                        wheel_mask: now.wheel_mask,
                        drive_mask: now.drive_mask,
                        steer_mask: now.steer_mask,
                        engine_connected: now.engine_connected,
                    });
                }
                let lost = now.parts_lost & !prev.parts_lost;
                for p in 0..32 {
                    if lost & (1 << p) != 0 {
                        out.push(VehicleEvent::PartLost { vehicle: h, part: p, body: self.part_body(h, p) });
                    }
                }
                for (w, (a, b)) in prev.wheels.iter().zip(&now.wheels).enumerate() {
                    let w = w as u32;
                    if b.detached && !a.detached {
                        out.push(VehicleEvent::WheelDetached { vehicle: h, wheel: w, body: self.wheel_body(h, w) });
                    }
                    if b.disabled && !a.disabled {
                        out.push(VehicleEvent::WheelRemoved { vehicle: h, wheel: w });
                    }
                    if b.in_contact && !a.in_contact {
                        out.push(VehicleEvent::WheelContactBegin { vehicle: h, wheel: w, ground_body: b.ground_body });
                    }
                    if !b.in_contact && a.in_contact {
                        out.push(VehicleEvent::WheelContactEnd { vehicle: h, wheel: w });
                    }
                    if b.sliding && !a.sliding {
                        out.push(VehicleEvent::WheelSlideBegin { vehicle: h, wheel: w });
                    }
                    if !b.sliding && a.sliding {
                        out.push(VehicleEvent::WheelSlideEnd { vehicle: h, wheel: w });
                    }
                }
            }
            self.seen[i] = Some(now);
        }
        out
    }

    /// Copy every vehicle record (static, dynamic and control state) for rollback.
    pub fn save(&self, backend: &impl VehicleBackend) -> VehicleSnapshot {
        VehicleSnapshot { data: backend.read_slots(0, self.capacity.layout().total_slots), host: self.clone() }
    }

    /// Restore a [`VehicleSnapshot`]; restore the matching bodies/colliders too.
    pub fn restore(&mut self, backend: &mut impl VehicleBackend, snap: &VehicleSnapshot) {
        backend.write_slots(0, &snap.data);
        *self = snap.host.clone();
        self.dirty = true;
    }

    pub fn vehicle_count(&self) -> u32 {
        self.entries.len() as u32
    }

    pub fn wheel_count(&self) -> u32 {
        self.wheels_used
    }

    pub fn header(&self) -> [Slot; 4] {
        let mut h = self.capacity.header(self.aabb_tail, self.vehicle_count(), self.wheels_used, self.rays, self.parts_used);
        h[VH_MISC as usize][3] = u32::from(self.full_scan_queries);
        h
    }

    pub fn chassis_body(&self, h: VehicleHandle) -> u32 {
        self.entries[h.0 as usize].chassis
    }

    pub fn desc(&self, h: VehicleHandle) -> &VehicleDesc {
        &self.entries[h.0 as usize].desc
    }

    /// Body index of a part's spawned rigid body (valid after `detach_part`).
    pub fn part_body(&self, h: VehicleHandle, part: u32) -> u32 {
        self.entries[h.0 as usize].first_part_body + part
    }

    /// Global wheel slot of a vehicle's wheel (index into the wheel records).
    pub fn wheel_slot(&self, h: VehicleHandle, wheel: u32) -> u32 {
        self.entries[h.0 as usize].slots.wheel_base + wheel
    }

    /// Body index of a wheel's free rigid body (valid after `detach_wheel`).
    pub fn wheel_body(&self, h: VehicleHandle, wheel: u32) -> u32 {
        self.entries[h.0 as usize].chassis + 2 + wheel
    }

    pub fn add_vehicle(
        &mut self,
        backend: &mut impl VehicleBackend,
        desc: &VehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
    ) -> anyhow_lite::Result<VehicleHandle> {
        let req = pack::requirements(desc);
        let cap = self.capacity;
        let v = self.vehicle_count();
        anyhow_lite::ensure(v < cap.max_vehicles, "vehicle capacity exhausted")?;
        anyhow_lite::ensure(self.wheels_used + req.wheel_slots <= cap.max_wheels, "wheel capacity exhausted")?;
        anyhow_lite::ensure(
            self.curve_points_used + req.curve_points <= cap.max_curve_points,
            "curve capacity exhausted",
        )?;
        anyhow_lite::ensure(self.diffs_used + req.diffs <= cap.max_diffs, "differential capacity exhausted")?;
        anyhow_lite::ensure(self.anti_rolls_used + req.anti_rolls <= cap.max_anti_rolls, "anti-roll capacity exhausted")?;
        anyhow_lite::ensure(self.parts_used + req.parts <= cap.max_parts, "part capacity exhausted")?;
        let slots = VehicleSlots {
            vehicle: v,
            wheel_base: self.wheels_used,
            curve_point_base: self.curve_points_used,
            diff_base: self.diffs_used,
            anti_roll_base: self.anti_rolls_used,
            part_base: self.parts_used,
            body_base: backend.body_count(),
            collider_base: backend.collider_count(),
            aabb_block: self.aabb_tail + 4 * v,
            joint_base: backend.joint_count(),
        };
        let packed = pack::pack(desc, Vec3::from_array(position), Quat::from_array(rotation), slots);
        self.upload(backend, &packed, &slots);
        self.wheels_used += req.wheel_slots;
        self.curve_points_used += packed.curve_points.len() as u32;
        self.diffs_used += packed.diffs.len() as u32;
        self.anti_rolls_used += packed.anti_rolls.len() as u32;
        self.parts_used += packed.parts.len() as u32;
        let mut controls = [[0u32; 4]; 4];
        controls.copy_from_slice(&packed.vehicle[V_IN0 as usize..V_IN0 as usize + 4]);
        self.entries.push(Entry {
            slots,
            wheel_slots: req.wheel_slots,
            wheels_used: desc.wheels.len() as u32,
            desc: desc.clone(),
            controls,
            chassis: slots.body_base,
            first_part_body: slots.body_base + 2 + req.wheel_slots,
            origin: desc.center_of_mass(),
        });
        backend.write_slots(0, &self.header());
        self.dirty = true;
        Ok(VehicleHandle(v))
    }

    fn upload(&self, backend: &mut impl VehicleBackend, p: &PackedVehicle, s: &VehicleSlots) {
        let l = self.capacity.layout();
        backend.append_bodies(&p.bodies);
        backend.append_colliders(&p.colliders);
        backend.append_joints(&p.joints);
        backend.write_slots(l.vehicle_base + s.vehicle * VEH_STRIDE, &p.vehicle);
        for (i, w) in p.wheels.iter().enumerate() {
            backend.write_slots(l.wheel_base + (s.wheel_base + i as u32) * WHEEL_STRIDE, w);
        }
        if !p.curve_points.is_empty() {
            backend.write_curve_points(
                u64::from(l.curve_base) * 16 + u64::from(s.curve_point_base) * 8,
                &p.curve_points,
            );
        }
        for (i, d) in p.diffs.iter().enumerate() {
            backend.write_slots(l.diff_base + (s.diff_base + i as u32) * DIFF_STRIDE, d);
        }
        for (i, a) in p.anti_rolls.iter().enumerate() {
            backend.write_slots(l.anti_roll_base + (s.anti_roll_base + i as u32) * ANTIROLL_STRIDE, &[*a]);
        }
        for (i, part) in p.parts.iter().enumerate() {
            backend.write_slots(l.part_base + (s.part_base + i as u32) * PART_STRIDE, part);
        }
    }

    fn write_controls(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle) {
        let l = self.capacity.layout();
        let e = &self.entries[h.0 as usize];
        backend.write_slots(l.vehicle_base + e.slots.vehicle * VEH_STRIDE + V_IN0, &e.controls);
        self.dirty = true;
    }

    pub fn set_input(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, input: VehicleInput) {
        let e = &mut self.entries[h.0 as usize];
        let c = &mut e.controls;
        let new0 = [fbits(input.forward), fbits(input.steer), fbits(input.brake), fbits(input.handbrake)];
        let gear = match input.gear {
            GearRequest::Auto => GEAR_AUTO,
            GearRequest::Gear(g) => g as u32,
        };
        if c[0] == new0 && c[1][0] == gear {
            return;
        }
        c[0] = new0;
        c[1][0] = gear;
        self.write_controls(backend, h);
    }

    pub fn set_functional_state(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, s: FunctionalState) {
        let c = &mut self.entries[h.0 as usize].controls;
        c[2] = [s.wheel_mask, s.drive_mask, s.steer_mask, s.brake_mask];
        c[3][2] = (c[3][2] & !1) | u32::from(s.engine_connected);
        self.write_controls(backend, h);
    }

    /// Sever a wheel: it becomes a free rigid body carrying its spin.
    pub fn detach_wheel(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, wheel: u32) {
        self.entries[h.0 as usize].controls[1][2] |= 1 << wheel;
        self.write_controls(backend, h);
    }

    /// Remove a wheel without spawning a body (destroyed / `remove_wheel`).
    pub fn disable_wheel(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, wheel: u32) {
        self.entries[h.0 as usize].controls[1][3] |= 1 << wheel;
        self.write_controls(backend, h);
    }

    pub fn remove_wheel(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, wheel: u32) {
        self.disable_wheel(backend, h, wheel);
    }

    pub fn detach_part(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, part: u32) {
        self.entries[h.0 as usize].controls[3][0] |= 1 << part;
        self.write_controls(backend, h);
    }

    pub fn disable_part(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, part: u32) {
        self.entries[h.0 as usize].controls[3][1] |= 1 << part;
        self.write_controls(backend, h);
    }

    /// Wake a parked vehicle without changing its input.
    pub fn wake(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle) {
        let c = &mut self.entries[h.0 as usize].controls;
        c[3][3] = c[3][3].wrapping_add(1);
        self.write_controls(backend, h);
    }

    /// Attach a new wheel into one of the vehicle's spare slots.
    pub fn add_wheel(
        &mut self,
        backend: &mut impl VehicleBackend,
        h: VehicleHandle,
        wheel: &WheelDesc,
    ) -> anyhow_lite::Result<u32> {
        let l = self.capacity.layout();
        let idx = {
            let e = &self.entries[h.0 as usize];
            anyhow_lite::ensure(e.wheels_used < e.wheel_slots, "no spare wheel slot (VehicleDesc::spare_wheels)")?;
            e.wheels_used
        };
        let e = &self.entries[h.0 as usize];
        let wheel = &WheelDesc { position: (Vec3::from_array(wheel.position) - Vec3::from_array(e.origin)).to_array(), ..wheel.clone() };
        let slot_index = e.slots.wheel_base + idx;
        let rec = l.wheel_base + slot_index * WHEEL_STRIDE;
        // Keep the slot's curves and parked body/rims, update everything static.
        let old = backend.read_slots(rec, WHEEL_STRIDE);
        let mut w = pack::pack_wheel(
            &e.desc,
            wheel,
            e.slots.vehicle,
            idx,
            true,
            old[W_PARK as usize][0],
            old[W_PARK as usize][1],
            (old[W_TIRE as usize][0], old[W_TIRE as usize][1]),
        );
        w[W_C0 as usize][2] = WD_PENDING_ADD;
        let parked = old[W_PARK as usize][0];
        let rims = old[W_PARK as usize][2];
        w[W_PARK as usize][2] = rims;
        let up = Vec3::from_array(e.desc.up).normalize();
        let fwd = Vec3::from_array(e.desc.forward).normalize();
        let axle = up.cross(fwd).normalize();
        let ia = 0.5 * wheel.mass * wheel.radius * wheel.radius;
        let im = wheel.mass * (3.0 * wheel.radius * wheel.radius + wheel.width * wheel.width) / 12.0;
        let inertia = Vec3::splat(im) + (ia - im) * axle * axle;
        let mut body = BodyState::dynamic(Vec3::from_array(pack::parked_position(parked)), Quat::IDENTITY, wheel.mass, inertia);
        body.flags = 0;
        body.initial_pos = body.position;
        backend.write_body(parked, &body);
        for (k, c) in pack::rim_colliders_n(wheel, up, fwd, parked, rims).iter().enumerate() {
            backend.write_collider(old[W_PARK as usize][1] + k as u32, c);
        }
        // Keep the slot's tyre joint and shell; resize the shell for this wheel
        // (vehicle_post moves it onto the chassis once the wheel is attached).
        w[W_TJ as usize] = [old[W_TJ as usize][0], old[W_TJ as usize][1], NO_INDEX, 0];
        write_shell(backend, wheel, up, fwd, old[W_TJ as usize][1], e.slots.body_base + 1);
        backend.write_slots(rec, &w);
        let entry = &mut self.entries[h.0 as usize];
        entry.wheels_used += 1;
        entry.desc.wheels.push(wheel.clone());
        self.dirty = true;
        Ok(idx)
    }

    /// Replace a wheel's tuning (suspension, tyre, brakes, steering) in place.
    pub fn set_wheel_params(&mut self, backend: &mut impl VehicleBackend, h: VehicleHandle, wheel: u32, desc: &WheelDesc) {
        let l = self.capacity.layout();
        let e = &self.entries[h.0 as usize];
        let rec = l.wheel_base + (e.slots.wheel_base + wheel) * WHEEL_STRIDE;
        let old = backend.read_slots(rec, W_C0);
        let shifted = WheelDesc { position: (Vec3::from_array(desc.position) - Vec3::from_array(e.origin)).to_array(), ..desc.clone() };
        let w = pack::pack_wheel(
            &e.desc,
            &shifted,
            e.slots.vehicle,
            wheel,
            true,
            old[W_PARK as usize][0],
            old[W_PARK as usize][1],
            (old[W_TIRE as usize][0], old[W_TIRE as usize][1]),
        );
        // Static slots only (dynamic state stays GPU-owned).
        backend.write_slots(rec + W_HP, &w[W_HP as usize..W_PARK as usize]);
        // Resize the shell to the new tyre (keeps whichever body owns it now).
        let shell = backend.read_slots(rec + W_TJ, 1)[0][1];
        let up = Vec3::from_array(e.desc.up).normalize();
        let fwd = Vec3::from_array(e.desc.forward).normalize();
        write_shell(backend, &shifted, up, fwd, shell, e.chassis);
        backend.write_slots(rec + W_UP, &w[W_UP as usize..W_UP as usize + 1]);
        let mut cfg = old[W_CFG as usize];
        cfg[1] = w[W_CFG as usize][1];
        backend.write_slots(rec + W_CFG, &[cfg]);
        self.entries[h.0 as usize].desc.wheels[wheel as usize] = desc.clone();
        self.dirty = true;
    }

    pub fn read_state(&self, backend: &impl VehicleBackend, h: VehicleHandle) -> VehicleState {
        let l = self.capacity.layout();
        let e = &self.entries[h.0 as usize];
        let v = backend.read_slots(l.vehicle_base + e.slots.vehicle * VEH_STRIDE, VEH_STRIDE);
        let o0 = v[V_O0 as usize];
        let o1 = v[V_O1 as usize];
        let o2 = v[V_O2 as usize];
        let o3 = v[V_O3 as usize];
        let wheels = backend.read_slots(l.wheel_base + e.slots.wheel_base * WHEEL_STRIDE, e.wheels_used * WHEEL_STRIDE);
        let mut out = VehicleState {
            forward_speed: f32::from_bits(o0[0]),
            engine_rpm: f32::from_bits(o0[1]),
            gear: f32::from_bits(o0[2]).round() as i32,
            lateral_speed: f32::from_bits(o1[0]),
            yaw_rate: f32::from_bits(o1[1]),
            clutch: f32::from_bits(o1[2]),
            contacts: o1[3],
            wheel_mask: o2[0],
            drive_mask: o2[1],
            steer_mask: o2[2],
            event_seq: o2[3],
            engine_connected: o3[0] != 0,
            parts_lost: o3[1],
            wheels: Vec::new(),
        };
        for wi in 0..e.wheels_used as usize {
            let w = &wheels[wi * WHEEL_STRIDE as usize..(wi + 1) * WHEEL_STRIDE as usize];
            let st = w[W_O_ST as usize];
            let flags = st[3];
            let dynf = w[W_C0 as usize][2];
            let ground = w[W_O_N as usize][3];
            out.wheels.push(WheelState {
                position: f3(w[W_O_POS as usize]),
                rotation: [
                    f32::from_bits(w[W_O_ROT as usize][0]),
                    f32::from_bits(w[W_O_ROT as usize][1]),
                    f32::from_bits(w[W_O_ROT as usize][2]),
                    f32::from_bits(w[W_O_ROT as usize][3]),
                ],
                suspension_length: fw(w[W_O_POS as usize]),
                mount_force: f3(w[W_O_FM as usize]),
                suspension_force: fw(w[W_O_FM as usize]),
                mount_torque: f3(w[W_O_TM as usize]),
                longitudinal_force: fw(w[W_O_TM as usize]),
                contact_point: f3(w[W_O_CP as usize]),
                lateral_force: fw(w[W_O_CP as usize]),
                contact_normal: f3(w[W_O_N as usize]),
                ground_body: if ground == NO_INDEX { None } else { Some(ground) },
                omega: f32::from_bits(st[0]),
                slip_ratio: f32::from_bits(st[1]),
                slip_angle_deg: f32::from_bits(st[2]),
                in_contact: flags & WD_CONTACT != 0,
                sliding: flags & WD_SLIDE != 0,
                detached: dynf & WD_DETACHED != 0,
                disabled: dynf & WD_DISABLED != 0,
            });
        }
        out
    }

    /// Mount impulse wrenches for the last step (`force · dt`), for a stress solver.
    pub fn step_loads(&self, backend: &impl VehicleBackend, dt: f32) -> Vec<WheelLoad> {
        let mut out = Vec::new();
        for (i, _) in self.entries.iter().enumerate() {
            let h = VehicleHandle(i as u32);
            for (wi, w) in self.read_state(backend, h).wheels.iter().enumerate() {
                out.push(WheelLoad {
                    vehicle: h,
                    wheel: wi as u32,
                    impulse: w.mount_force.map(|x| x * dt),
                    angular_impulse: w.mount_torque.map(|x| x * dt),
                });
            }
        }
        out
    }

    /// Upload a ray batch; the host dispatches `raycast_batch` over `rays.len()` groups.
    pub fn write_rays(&mut self, backend: &mut impl VehicleBackend, rays: &[Ray]) {
        assert!(rays.len() as u32 <= self.capacity.max_rays, "ray capacity exceeded");
        let l = self.capacity.layout();
        let mut data = Vec::with_capacity(rays.len() * RAY_STRIDE as usize);
        for r in rays {
            data.push([fbits(r.origin[0]), fbits(r.origin[1]), fbits(r.origin[2]), fbits(r.max_dist)]);
            data.push([fbits(r.dir[0]), fbits(r.dir[1]), fbits(r.dir[2]), r.exclude_body.unwrap_or(NO_INDEX)]);
            data.push([0, 0, 0, fbits(-1.0)]);
            data.push([0, 0, 0, NO_INDEX]);
        }
        backend.write_slots(l.ray_base, &data);
        self.rays = rays.len() as u32;
        backend.write_slots(0, &self.header());
    }

    pub fn ray_count(&self) -> u32 {
        self.rays
    }

    pub fn read_ray_hits(&self, backend: &impl VehicleBackend) -> Vec<Option<RayHit>> {
        let l = self.capacity.layout();
        let data = backend.read_slots(l.ray_base, self.rays * RAY_STRIDE);
        (0..self.rays as usize)
            .map(|i| {
                let o0 = data[i * 4 + 2];
                let o1 = data[i * 4 + 3];
                let dist = f32::from_bits(o0[3]);
                if dist < 0.0 || o1[3] == NO_INDEX {
                    None
                } else {
                    Some(RayHit { point: f3(o0), normal: f3(o1), distance: dist, collider: o1[3] })
                }
            })
            .collect()
    }
}

/// Minimal error type so the crate stays dependency-light.
pub mod anyhow_lite {
    pub type Result<T> = std::result::Result<T, String>;
    pub fn ensure(cond: bool, msg: &str) -> Result<()> {
        if cond {
            Ok(())
        } else {
            Err(msg.to_string())
        }
    }
}

/// Rewrite a wheel slot's shell spheres for `wheel` (centre at full droop; the
/// kernel re-places them every step).
fn write_shell(backend: &mut impl VehicleBackend, wheel: &WheelDesc, up: Vec3, fwd: Vec3, first: u32, body: u32) {
    if first == NO_INDEX {
        return;
    }
    let down = wheel.suspension_dir.map(Vec3::from_array).unwrap_or(-up).normalize();
    let center = Vec3::from_array(wheel.position) + down * wheel.suspension_max;
    for (k, c) in pack::shell_colliders(wheel, center, down, fwd, body).iter().enumerate() {
        backend.write_collider(first + k as u32, c);
    }
}
