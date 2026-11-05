//! The rigid coupling's records in an [`ExplicitBeamsGpu`] built
//! [`ExplicitBeamsGpu::with_coupling`] (`slang/kernels/explicit_beams_rigid.slang`):
//! a CPU [`ProxySet`] written as a vehicle's proxies for the next frames, and read back.
//! The World fills the same records on the GPU (its coupling plan); these are for tests
//! and hosts that plan on the CPU.

use crate::explicit_beams_gpu::{CouplingLayout, ExplicitBeamsGpu};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet};
use glam::{Quat, Vec3};

fn f4(x: f32, y: f32, z: f32, w: f32) -> [u32; 4] {
    [x.to_bits(), y.to_bits(), z.to_bits(), w.to_bits()]
}

fn q4(q: Quat) -> [u32; 4] {
    f4(q.x, q.y, q.z, q.w)
}

/// Shape kind codes (the kernel's CPL_*).
pub const CPL_BOX: u32 = 0;
pub const CPL_SPHERE: u32 = 1;
pub const CPL_CAPSULE: u32 = 2;
pub const CPL_HULL: u32 = 3;

/// A vehicle's coupling records for `set` (its window start: call
/// [`ProxySet::begin_window`] first, which sets the moments' reference points).
pub fn proxy_records(layout: &CouplingLayout, set: &ProxySet) -> Result<Vec<[u32; 4]>, String> {
    let cap = layout.cap;
    if set.bodies.len() > cap.bodies as usize || set.shapes.len() > cap.shapes as usize {
        return Err(format!(
            "explicit coupling: {} bodies and {} shapes, room for {} and {}",
            set.bodies.len(),
            set.shapes.len(),
            cap.bodies,
            cap.shapes
        ));
    }
    let mut recs = vec![[0u32; 4]; layout.recs() as usize];
    let mut planes = 0u32;
    for (j, b) in set.bodies.iter().enumerate() {
        let at = 1 + j * CouplingLayout::BODY_RECS as usize;
        recs[at] = f4(b.pos.x, b.pos.y, b.pos.z, b.inv_mass);
        recs[at + 1] = q4(b.rot);
        recs[at + 2] = f4(b.vel.x, b.vel.y, b.vel.z, b.share);
        let w = f4(b.ang_vel.x, b.ang_vel.y, b.ang_vel.z, 0.0);
        recs[at + 3] = [w[0], w[1], w[2], b.body];
        recs[at + 4] = f4(b.inv_inertia.x, b.inv_inertia.y, b.inv_inertia.z, 0.0);
    }
    let shape0 = 1 + (cap.bodies * CouplingLayout::BODY_RECS) as usize;
    let plane0 = shape0 + (cap.shapes * CouplingLayout::SHAPE_RECS) as usize;
    for (k, s) in set.shapes.iter().enumerate() {
        let at = shape0 + k * CouplingLayout::SHAPE_RECS as usize;
        let (kind, p, radius, span) = match &s.geom {
            ProxyGeom::Box { half } => (CPL_BOX, [half.x, half.y, half.z], half.length(), 0),
            ProxyGeom::Sphere { radius } => (CPL_SPHERE, [*radius, 0.0, 0.0], *radius, 0),
            ProxyGeom::Capsule { radius, half_len } => {
                (CPL_CAPSULE, [*radius, *half_len, 0.0], radius + half_len, 0)
            }
            ProxyGeom::Hull { planes: list, radius } => {
                if planes as usize + list.len() > cap.planes as usize || list.len() > 0xFFFF {
                    return Err(format!("explicit coupling: more than {} hull planes", cap.planes));
                }
                for (i, &(n, d)) in list.iter().enumerate() {
                    recs[plane0 + planes as usize + i] = f4(n.x, n.y, n.z, d);
                }
                let span = planes | (list.len() as u32) << 16;
                planes += list.len() as u32;
                (CPL_HULL, [0.0; 3], *radius, span)
            }
        };
        recs[at] = [kind, s.body, s.collider, span];
        recs[at + 1] = f4(s.local_pos.x, s.local_pos.y, s.local_pos.z, p[0]);
        recs[at + 2] = q4(s.local_rot);
        recs[at + 3] = f4(p[1], p[2], radius + 0.01, 0.0);
        recs[at + 4] = f4(0.0, 0.0, 0.0, 0.0);
        recs[at + 5] = f4(0.0, 0.0, 0.0, 0.0);
        recs[at + 6] = f4(s.reference.x, s.reference.y, s.reference.z, 0.0);
        let _ = s.chunk;
    }
    recs[0] = [set.bodies.len() as u32, set.shapes.len() as u32, planes, 0];
    Ok(recs)
}

/// Write vehicle `v`'s proxies (its accumulators cleared) for the next frames.
pub fn write_proxies(gpu: &ExplicitBeamsGpu, queue: &wgpu::Queue, v: usize, set: &ProxySet) -> Result<(), String> {
    let layout = gpu.coupling().ok_or("explicit coupling: built without coupling room")?;
    let recs = proxy_records(&layout, set)?;
    queue.write_buffer(
        gpu.misc_buffer(),
        u64::from(layout.vehicle(v as u32)) * 16,
        bytemuck::cast_slice(&recs),
    );
    let zeros = vec![0i32; layout.acc_per_vehicle() as usize];
    queue.write_buffer(gpu.acc_buffer(), u64::from(layout.acc(v as u32)) * 4, bytemuck::cast_slice(&zeros));
    Ok(())
}

/// Read vehicle `v`'s proxies back into `set` (blocking): the bodies' state, each shape's
/// window impulse, moment and peak.
pub fn read_proxies(gpu: &ExplicitBeamsGpu, device: &wgpu::Device, queue: &wgpu::Queue, v: usize, set: &mut ProxySet) {
    let layout = gpu.coupling().expect("coupling room");
    let n = layout.recs() as u64;
    let words: Vec<u32> = read_words(device, queue, gpu.misc_buffer(), u64::from(layout.vehicle(v as u32)) * 16, n * 16);
    let rec = |i: usize| -> [f32; 4] {
        let w = &words[i * 4..i * 4 + 4];
        [f32::from_bits(w[0]), f32::from_bits(w[1]), f32::from_bits(w[2]), f32::from_bits(w[3])]
    };
    for (j, b) in set.bodies.iter_mut().enumerate() {
        let at = 1 + j * CouplingLayout::BODY_RECS as usize;
        let r0 = rec(at);
        let r1 = rec(at + 1);
        let r2 = rec(at + 2);
        let r3 = rec(at + 3);
        b.pos = Vec3::new(r0[0], r0[1], r0[2]);
        b.rot = Quat::from_xyzw(r1[0], r1[1], r1[2], r1[3]);
        b.vel = Vec3::new(r2[0], r2[1], r2[2]);
        b.ang_vel = Vec3::new(r3[0], r3[1], r3[2]);
    }
    let shape0 = 1 + (layout.cap.bodies * CouplingLayout::BODY_RECS) as usize;
    for (k, s) in set.shapes.iter_mut().enumerate() {
        let at = shape0 + k * CouplingLayout::SHAPE_RECS as usize;
        let i = rec(at + 4);
        let m = rec(at + 5);
        s.impulse = Vec3::new(i[0], i[1], i[2]);
        s.peak = i[3];
        s.moment = Vec3::new(m[0], m[1], m[2]);
    }
}

/// `size` bytes of `buffer` from `offset` (blocking).
pub fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("explicit-coupling-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("explicit-coupling-read") });
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map ok");
    let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
    staging.unmap();
    out
}
