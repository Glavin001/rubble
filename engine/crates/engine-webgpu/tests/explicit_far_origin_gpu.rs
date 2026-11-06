//! Vehicle contact's precision does not depend on where the vehicles are: vehicle–vehicle
//! contact runs in the owner's frame (its geometry relative to the owner's origin,
//! `explicit_beams.slang` `vvRel`; the CPU's `ExplicitBeamsWorld::owner_frame`), so two
//! procedural sedans (fixed-point contact sums: deterministic) meeting head on at the
//! origin and the same crash with both origins at (4096, 0, 2048) m step bit for bit
//! alike, every node word, on the GPU and on the CPU. (In world coordinates, as before,
//! every contact position at 4 km rounded to 0.49 mm.) The rigid coupling's counterpart
//! is `explicit_rigid_gpu.rs` `far_from_the_origin_the_coupling_is_the_origins_bit_for_bit`.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

/// The procedural sedan (collision triangles) with fixed-point contact sums (the BeamNG
/// profile's: order-free, so GPU runs are deterministic), its front (+z) at `front_z`
/// heading along `dir` (±1) at 6 m/s.
fn car(front_z: f32, dir: f32) -> ExplicitBeams {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let v = Vec3::new(0.0, 0.0, 6.0 * dir);
    let mut s = ExplicitBeams::from_truck(&text, Vec3::ZERO, v, settings).unwrap();
    s.fixed_contact = true;
    if dir < 0.0 {
        for n in &mut s.nodes {
            n.pos = Vec3::new(-n.pos.x, n.pos.y, -n.pos.z);
        }
    }
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let tip = s
        .nodes
        .iter()
        .map(|n| n.pos.z * dir)
        .fold(f32::MIN, f32::max);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
        n.pos.z += front_z - tip * dir;
    }
    s.roll_tyres(v);
    s
}

/// Two cars 0.6 m apart nose to nose, both origins at `origin` (node positions the same
/// local values whatever the origin).
fn crash(origin: Vec3) -> Vec<ExplicitBeams> {
    let mut f = vec![car(-0.3, 1.0), car(0.3, -1.0)];
    for v in &mut f {
        v.origin = origin;
    }
    f
}

const FAR: Vec3 = Vec3::new(4096.0, 0.0, 2048.0);
const FRAMES: usize = 30;

fn local_bits(v: &ExplicitBeams) -> Vec<[u32; 3]> {
    v.nodes
        .iter()
        .map(|n| n.pos.to_array().map(f32::to_bits))
        .collect()
}

#[test]
fn vehicle_contact_at_4_km_is_the_origins_bit_for_bit_on_the_cpu() {
    let mut a = ExplicitBeamsWorld::new(crash(Vec3::ZERO));
    let mut b = ExplicitBeamsWorld::new(crash(FAR));
    for f in 0..FRAMES {
        a.frame(17);
        b.frame(17);
        for k in 0..2 {
            assert_eq!(
                local_bits(&a.vehicles[k]),
                local_bits(&b.vehicles[k]),
                "CPU frame {f}, vehicle {k}"
            );
        }
    }
    let fronts: Vec<(f32, f32)> = a
        .vehicles
        .iter()
        .map(|v| {
            v.nodes.iter().fold((f32::MAX, f32::MIN), |(lo, hi), n| {
                (lo.min(n.pos.z), hi.max(n.pos.z))
            })
        })
        .collect();
    eprintln!("CPU contact: {:?}; z extents {fronts:?}", a.total);
    assert!(a.total.hits > 0, "the cars met");
    eprintln!(
        "CPU: {FRAMES} frames, {} contact hits, bit for bit at the origin and at 4 km",
        a.total.hits
    );
}

#[test]
fn vehicle_contact_at_4_km_is_the_origins_bit_for_bit_on_the_gpu() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut a = ExplicitBeamsGpu::new(&device, &queue, &crash(Vec3::ZERO));
    let mut b = ExplicitBeamsGpu::new(&device, &queue, &crash(FAR));
    let nodes = |g: &ExplicitBeamsGpu| {
        g.state_words()
            .into_iter()
            .find(|(n, _)| *n == "nodes")
            .unwrap()
            .1
    };
    let mut met = false;
    for f in 0..FRAMES {
        a.frame(17);
        b.frame(17);
        let (x, y) = (nodes(&a), nodes(&b));
        let differ = x.iter().zip(&y).filter(|(p, q)| p != q).count();
        assert_eq!(
            differ, 0,
            "GPU frame {f}: {differ} node words differ at 4 km"
        );
        met |= a.contact_groups() > 0;
    }
    a.contact_status().unwrap();
    b.contact_status().unwrap();
    assert!(met, "the cars never formed a contact group");
    eprintln!(
        "GPU: {FRAMES} frames head on, every node word bit for bit at the origin and at 4 km"
    );
}
