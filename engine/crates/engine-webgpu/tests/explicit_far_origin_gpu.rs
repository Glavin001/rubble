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

/// Spawned as the World spawns vehicles: origin 0 and the nodes far out, so both
/// vehicles re-base on their first frame (each origin onto its node 0) and then meet.
#[test]
fn rebased_vehicles_step_and_meet() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut fleet = crash(Vec3::ZERO);
    for v in &mut fleet {
        for n in &mut v.nodes {
            n.pos += FAR;
        }
    }
    let time = |fleet: &[ExplicitBeams]| {
        let mut g = ExplicitBeamsGpu::new(&device, &queue, fleet);
        let t = std::time::Instant::now();
        let mut met = false;
        for _ in 0..FRAMES {
            g.frame(17);
            met |= g.contact_groups() > 0;
        }
        g.wait();
        (g, met, t.elapsed().as_secs_f64())
    };
    let (_, _, near) = time(&crash(Vec3::ZERO));
    let (g, met, secs) = time(&fleet);
    g.contact_status().unwrap();
    let ks = g.kernel_status();
    let o = g.origins();
    eprintln!("re-based origins {o:?}, met {met}; {secs:.1} s against {near:.1} s near the origin; kernel status {ks:?}");
    ks.unwrap();
    assert!(o.iter().all(|o| o.length() > 4000.0), "both re-based");
    assert!(met, "they met");
}

/// A BeamNG-profile JBeam rig with its origin at `origin` (local nodes the same whatever
/// the origin): the fixture's slide node on a fixed rail (hanging on its spring under
/// gravity), or with `raycast` the fixture car on raycast wheels resting 2 cm above the
/// ground, driven at half throttle.
fn rig(raycast: bool, origin: Vec3) -> ExplicitBeams {
    use engine_softbody::jbeam_fixture::{PART2_CAR, PART2_RAIL};
    let text = if raycast { PART2_CAR } else { PART2_RAIL };
    let d = engine_softbody::jbeam::load_jbeam(&[("t.jbeam", text)], None).unwrap();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let mut s = if raycast {
        ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, settings)
    } else {
        ExplicitBeams::from_desc_jbeam_with(
            &d,
            engine_softbody::explicit_beams::JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings,
        )
    }
    .unwrap()
    .0;
    s.fixed_contact = true;
    if raycast {
        let lift = s.spawn_clearance() - s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        for n in &mut s.nodes {
            n.pos.y += lift;
        }
        s.set_drive_input(engine_softbody::explicit_drive::DriveInput {
            forward: 0.5,
            ..Default::default()
        });
    } else {
        // Clear of the ground.
        for n in &mut s.nodes {
            n.pos.y += 2.0;
        }
    }
    s.origin = origin;
    s
}

/// Slide nodes and raycast wheels work in the vehicle's frame (the slide pass's rail
/// distance, the wheels' hubs, rays and ground height relative to its origin): a slide
/// node on its rail and the fixture car on raycast wheels step bit for bit alike at the
/// origin and at 4 km, on the CPU and on the GPU. (In World coordinates the rail distance
/// at 4 km rounded to 0.24 mm, enough to keep a parked ETK awake for good.)
#[test]
fn slide_nodes_and_raycast_wheels_at_4_km_are_the_origins_bit_for_bit() {
    let gpu = default_device().ok();
    for raycast in [false, true] {
        // The rail rig (clear of the ground, which it never meets) also high up: its
        // node hangs off the rail along y, which World y at 0 would keep exact.
        let far = if raycast {
            FAR
        } else {
            FAR + Vec3::new(0.3, 97.7, 0.1)
        };
        let near = rig(raycast, Vec3::ZERO);
        if raycast {
            assert!(near.drive.is_some(), "raycast wheels");
        } else {
            assert!(!near.slides.is_empty(), "a slide node");
        }
        let mut a = ExplicitBeamsWorld::new(vec![near.clone()]);
        let mut b = ExplicitBeamsWorld::new(vec![rig(raycast, far)]);
        for f in 0..FRAMES {
            a.frame(17);
            b.frame(17);
            assert_eq!(
                local_bits(&a.vehicles[0]),
                local_bits(&b.vehicles[0]),
                "CPU frame {f}, raycast {raycast}"
            );
        }
        let Some((_, device, queue)) = gpu.as_ref() else {
            eprintln!("no GPU adapter: GPU half skipped");
            continue;
        };
        let mut a = ExplicitBeamsGpu::new(device, queue, &[near]);
        let mut b = ExplicitBeamsGpu::new(device, queue, &[rig(raycast, far)]);
        let nodes = |g: &ExplicitBeamsGpu| {
            g.state_words()
                .into_iter()
                .find(|(n, _)| *n == "nodes")
                .unwrap()
                .1
        };
        for f in 0..FRAMES {
            a.frame(17);
            b.frame(17);
            let (x, y) = (nodes(&a), nodes(&b));
            let differ = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            assert_eq!(
                differ, 0,
                "GPU frame {f}, raycast {raycast}: {differ} node words differ at 4 km"
            );
        }
        a.kernel_status().unwrap();
        b.kernel_status().unwrap();
    }
}
