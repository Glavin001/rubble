//! The opt-in `contact_events` pass (P4) on raw `WgpuSim` records: shape flags
//! in `Collider::material` bits 12..15 change no body, the pass reads and
//! never writes physics state, and its records describe the touch (pair,
//! deepest point, normal, closing speed from the step-start velocities).

use engine_schema::{
    AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, COLLIDER_CONTACT_EVENTS,
    COLLIDER_HIT_EVENTS, CONTACT_EVENT_CAP, CONTACT_EVENT_WORDS,
};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

/// One record of the contact-event region (`world_step.slang` `contact_events`).
#[derive(Clone, Copy, Debug)]
struct Record {
    collider_a: u32,
    collider_b: u32,
    approach_speed: f32,
    point: [f32; 3],
    normal: [f32; 3],
}

/// The last step's records (`contact_event_range`), and how many the region
/// could not hold.
fn read_events(s: &WgpuSim) -> (Vec<Record>, u32) {
    let (buffer, offset, size) = s.contact_event_range();
    let staging = s.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("contact events readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = s
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    s.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    let _ = s.device().poll(wgpu::PollType::wait_indefinitely());
    let words: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
    let held = words[0].min(CONTACT_EVENT_CAP) as usize;
    let f = f32::from_bits;
    let w = CONTACT_EVENT_WORDS as usize;
    let records = (0..held)
        .map(|k| {
            let r = &words[4 + k * w..4 + (k + 1) * w];
            Record {
                collider_a: r[0],
                collider_b: r[1],
                approach_speed: f(r[2]),
                point: [f(r[4]), f(r[5]), f(r[6])],
                normal: [f(r[8]), f(r[9]), f(r[10])],
            }
        })
        .collect();
    (records, words[0] - held as u32)
}

const G: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;

fn sim() -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu adapter");
    let mut p = AvbdParams::default();
    p.gravity_y = -G;
    p.dt = DT;
    sim.set_params(p);
    sim
}

/// A plane, and boxes dropped from `heights` (gap under the box), every
/// collider's material word or-ed with `flags`.
fn scene(flags: u32, heights: &[f32]) -> (WgpuSim, Vec<u32>) {
    let mut s = sim();
    let ground = s.add_body(BodyState::static_plane_support());
    s.add_collider(Collider {
        body: ground,
        shape: ShapeType::Plane as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [0.0, 1.0, 0.0, 0.0],
    });
    let h = 0.25;
    let i = 2.0 * h * h / 3.0;
    let mut boxes = Vec::new();
    for (k, gap) in heights.iter().enumerate() {
        let pos = Vec3::new(2.0 * k as f32, h + gap, 0.0);
        let body = s.add_body(BodyState::dynamic(pos, Quat::IDENTITY, 1.0, Vec3::splat(i)));
        s.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: flags,
            generation: 0,
            local_pos: [0.0; 3],
            density: 1.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [h, h, h, (3.0f32).sqrt() * h],
        });
        boxes.push(body);
    }
    (s, boxes)
}

#[test]
fn flags_only_add_the_pass_and_change_no_body() {
    let heights = [0.3, 0.8, 1.4];
    let (mut plain, boxes) = scene(0, &heights);
    let (mut flagged, _) = scene(COLLIDER_CONTACT_EVENTS | COLLIDER_HIT_EVENTS, &heights);
    flagged.set_contact_events(true);
    for _ in 0..90 {
        plain.step().unwrap();
        flagged.clear_contact_events();
        flagged.step().unwrap();
        for &b in &boxes {
            let (p, f) = (plain.read_body(b), flagged.read_body(b));
            assert_eq!(
                bytemuck::bytes_of(&p),
                bytemuck::bytes_of(&f),
                "body {b}: the event pass is read-only"
            );
        }
    }
    // The unflagged sim never ran the pass: its region was never written.
    let (records, lost) = read_events(&plain);
    assert!(records.is_empty() && lost == 0);
}

#[test]
fn records_carry_the_touch_at_the_impact_speed() {
    let gap = 0.8;
    let (mut s, boxes) = scene(COLLIDER_CONTACT_EVENTS | COLLIDER_HIT_EVENTS, &[gap]);
    s.set_contact_events(true);
    let v = (2.0 * G * gap).sqrt();
    let mut first = None;
    for step in 0..60 {
        s.clear_contact_events();
        let c = s.step().unwrap();
        let (records, lost) = read_events(&s);
        assert_eq!(lost, 0);
        assert!(
            records.len() as u32 <= c.manifold_count,
            "one record per manifold at most"
        );
        if let Some(r) = records.first() {
            first.get_or_insert((step, *r));
        }
    }
    let (step, r) = first.expect("the box touched the plane");
    let body = s.collider(r.collider_b).body;
    assert!(boxes.contains(&body) || boxes.contains(&s.collider(r.collider_a).body));
    assert!(
        (r.approach_speed - v).abs() <= G * DT + 0.02 * v,
        "step {step}: approach {} m/s, √(2gh) = {v}",
        r.approach_speed
    );
    // Normal from A to B, the approach measured along it.
    let up = if s.collider(r.collider_a).shape == ShapeType::Plane as u32 {
        1.0
    } else {
        -1.0
    };
    assert!((r.normal[1] - up).abs() < 1e-3, "normal {:?}", r.normal);
    assert!(r.point[1].abs() < 0.1, "point {:?} at the plane", r.point);
    assert!(CONTACT_EVENT_CAP > 0);
}
