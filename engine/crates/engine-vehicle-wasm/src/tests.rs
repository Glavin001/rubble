//! The recorded op stream, replayed the way `vehicle.mjs` does, must leave the
//! same records as `VehicleHost` writing straight into a backend.

use super::*;
use engine_vehicle::presets;

#[derive(Default)]
struct Mem {
    bodies: Vec<u32>,
    colliders: Vec<u32>,
    joints: Vec<u32>,
    data: Vec<u32>,
}

const BODY_WORDS: usize = std::mem::size_of::<BodyState>() / 4;
const COLLIDER_WORDS: usize = std::mem::size_of::<Collider>() / 4;
const JOINT_WORDS: usize = std::mem::size_of::<Joint>() / 4;

fn put(v: &mut Vec<u32>, at: usize, words: &[u32]) {
    if v.len() < at + words.len() {
        v.resize(at + words.len(), 0);
    }
    v[at..at + words.len()].copy_from_slice(words);
}

impl Mem {
    fn counts(&self) -> [u32; 3] {
        [
            (self.bodies.len() / BODY_WORDS) as u32,
            (self.colliders.len() / COLLIDER_WORDS) as u32,
            (self.joints.len() / JOINT_WORDS) as u32,
        ]
    }

    fn slots(&self, slot: u32, count: u32) -> Vec<u32> {
        let mut out = vec![0; count as usize * 4];
        for (i, w) in out.iter_mut().enumerate() {
            *w = self.data.get(slot as usize * 4 + i).copied().unwrap_or(0);
        }
        out
    }

    /// Mirror of `applyOps` in vehicle.mjs.
    fn replay(&mut self, ops: &[u32]) {
        let mut i = 0;
        while i < ops.len() {
            let (op, arg, n) = (ops[i], ops[i + 1] as usize, ops[i + 2] as usize);
            let w = &ops[i + 3..i + 3 + n];
            i += 3 + n;
            match op {
                OP_APPEND_BODY => {
                    assert_eq!(arg, self.bodies.len() / BODY_WORDS);
                    self.bodies.extend_from_slice(w);
                }
                OP_APPEND_COLLIDER => {
                    assert_eq!(arg, self.colliders.len() / COLLIDER_WORDS);
                    self.colliders.extend_from_slice(w);
                }
                OP_APPEND_JOINT => {
                    assert_eq!(arg, self.joints.len() / JOINT_WORDS);
                    self.joints.extend_from_slice(w);
                }
                OP_WRITE_BODY => put(&mut self.bodies, arg * BODY_WORDS, w),
                OP_WRITE_COLLIDER => put(&mut self.colliders, arg * COLLIDER_WORDS, w),
                OP_WRITE_SLOTS => put(&mut self.data, arg * 4, w),
                OP_WRITE_BYTES => put(&mut self.data, arg / 4, w),
                _ => panic!("unknown op {op}"),
            }
        }
    }
}

impl VehicleBackend for Mem {
    fn body_count(&self) -> u32 {
        self.counts()[0]
    }
    fn collider_count(&self) -> u32 {
        self.counts()[1]
    }
    fn joint_count(&self) -> u32 {
        self.counts()[2]
    }
    fn append_bodies(&mut self, bodies: &[BodyState]) {
        self.bodies.extend_from_slice(bytemuck::cast_slice(bodies));
    }
    fn append_colliders(&mut self, colliders: &[Collider]) {
        self.colliders
            .extend_from_slice(bytemuck::cast_slice(colliders));
    }
    fn append_joints(&mut self, joints: &[Joint]) {
        self.joints.extend_from_slice(bytemuck::cast_slice(joints));
    }
    fn write_body(&mut self, index: u32, body: &BodyState) {
        put(
            &mut self.bodies,
            index as usize * BODY_WORDS,
            bytemuck::cast_slice(std::slice::from_ref(body)),
        );
    }
    fn write_collider(&mut self, index: u32, collider: &Collider) {
        put(
            &mut self.colliders,
            index as usize * COLLIDER_WORDS,
            bytemuck::cast_slice(std::slice::from_ref(collider)),
        );
    }
    fn write_slots(&mut self, slot: u32, data: &[Slot]) {
        put(&mut self.data, slot as usize * 4, data.as_flattened());
    }
    fn write_curve_points(&mut self, byte_offset: u64, points: &[[f32; 2]]) {
        let words: Vec<u32> = points.as_flattened().iter().map(|x| x.to_bits()).collect();
        put(&mut self.data, byte_offset as usize / 4, &words);
    }
    fn read_slots(&self, slot: u32, count: u32) -> Vec<Slot> {
        self.slots(slot, count)
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }
}

#[test]
fn replayed_ops_match_direct_host_for_every_preset() {
    let cap = VehicleCapacity::default();
    let mut direct = (VehicleHost::new(cap, 4000), Mem::default());
    let mut replayed = (VehicleHost::new(cap, 4000), Mem::default());
    // Unrelated scene objects first so every base is non-zero.
    for m in [&mut direct.1, &mut replayed.1] {
        m.bodies.resize(3 * BODY_WORDS, 0);
        m.colliders.resize(5 * COLLIDER_WORDS, 0);
        m.joints.resize(2 * JOINT_WORDS, 0);
    }
    let all = presets::all();
    assert!(all.len() > 3);
    for (i, d) in all.iter().enumerate() {
        let pos = [i as f32 * 7.0, 1.5, -3.0];
        let rot = glam::Quat::from_rotation_y(0.3 + i as f32).to_array();
        direct.0.add_vehicle(&mut direct.1, d, pos, rot).unwrap();
        let mut r = Recorder::new(replayed.1.counts(), None);
        replayed.0.add_vehicle(&mut r, d, pos, rot).unwrap();
        replayed.1.replay(&r.ops);
    }
    for i in 0..all.len() as u32 {
        let h = VehicleHandle(i);
        let input = VehicleInput {
            forward: 0.5,
            steer: -0.25,
            gear: GearRequest::Gear(2),
            ..Default::default()
        };
        direct.0.set_input(&mut direct.1, h, input);
        direct.0.detach_wheel(&mut direct.1, h, 0);
        let mut r = Recorder::new([0; 3], None);
        replayed.0.set_input(&mut r, h, input);
        replayed.0.detach_wheel(&mut r, h, 0);
        replayed.1.replay(&r.ops);
    }
    // A spare-slot wheel goes through the reader callback.
    let (h, d) = all
        .iter()
        .enumerate()
        .find(|(_, d)| d.spare_wheels > 0)
        .map(|(i, d)| (VehicleHandle(i as u32), d))
        .expect("a preset with spares");
    let wheel = WheelDesc {
        position: [0.3, 0.1, 0.2],
        ..d.wheels[0].clone()
    };
    let a = direct.0.add_wheel(&mut direct.1, h, &wheel).unwrap();
    let mem = &replayed.1;
    let reader = |slot, count| mem.slots(slot, count);
    let mut r = Recorder::new([0; 3], Some(&reader));
    let b = replayed.0.add_wheel(&mut r, h, &wheel).unwrap();
    let ops = r.ops;
    replayed.1.replay(&ops);
    assert_eq!(a, b);
    assert_eq!(
        direct.0.desc(h).wheels.last(),
        Some(&wheel),
        "descriptor keeps the design-space wheel"
    );

    assert_eq!(direct.1.bodies, replayed.1.bodies, "bodies");
    assert_eq!(direct.1.colliders, replayed.1.colliders, "colliders");
    assert_eq!(direct.1.joints, replayed.1.joints, "joints");
    assert_eq!(direct.1.data, replayed.1.data, "vehicle data");
    for i in 0..all.len() as u32 {
        let h = VehicleHandle(i);
        assert_eq!(
            direct.0.read_state(&direct.1, h),
            replayed.0.read_state(&replayed.1, h)
        );
    }
}

#[test]
fn layout_constants_cover_the_records() {
    let v: serde_json::Value = serde_json::from_str(&layout_constants()).unwrap();
    assert_eq!(v["WHEEL_STRIDE"], engine_schema::vehicle::WHEEL_STRIDE);
    assert_eq!(v["NO_INDEX"], engine_schema::vehicle::NO_INDEX);
    assert_eq!(v["JOINT_FLAG_TYRE"], engine_schema::JOINT_FLAG_TYRE);
    assert_eq!(v["JOINT_BYTES"], 288);
}
