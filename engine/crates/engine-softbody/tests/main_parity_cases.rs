//! Cases of `main`'s (30ab0e1d) tests whose input or reference the branch edited,
//! kept here as main had them (PR #105 regression audit), so the edit does not drop
//! what main tested.

use engine_softbody::explicit_beams::*;
use glam::Vec3;

fn exact() -> ExplicitBeamsSettings {
    let mut s = ExplicitBeamsSettings::default();
    s.fast_inv_sqrt = false;
    s.drag = false;
    s.turbulence = false;
    s
}

/// `explicit_beams.rs::self_collision_pushes_a_contacter_out_of_a_triangle` as on
/// main: the contacter falls at 1 m/s (the branch's copy uses 10 m/s).
#[test]
fn main_self_collision_pushes_a_contacter_out_of_a_triangle_at_1_m_s() {
    let mut s = ExplicitBeams::empty(exact());
    s.ground = -1.0e3;
    for p in [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.015, 0.0),
    ] {
        s.nodes.push(Node { pos: p, vel: Vec3::ZERO, force: Vec3::ZERO, mass: 1.0, friction: 1.0, flags: NODE_NO_GROUND });
    }
    for n in &mut s.nodes[..3] {
        n.flags |= NODE_IMMOVABLE | NODE_ON_TRIANGLE;
    }
    s.nodes[3].flags |= NODE_CONTACTER;
    s.nodes[3].vel = Vec3::new(0.0, -1.0, 0.0);
    s.triangles.push(Triangle { nodes: [0, 1, 2], wait: 0, idle: 0 });
    s.finish();
    for _ in 0..200 {
        s.step();
        // Never through the face; it settles at the edge of the range.
        assert!(s.nodes[3].pos.y > 0.0, "through at {}", s.nodes[3].pos.y);
    }
    let y = s.nodes[3].pos.y;
    assert!(y < s.collision_range + 0.01, "contacter at {y}");
    assert!(s.triangles[0].wait < 0, "a hit resets the throttle");
}

/// The exact path's beam length against main's formula `q * (1 / sqrt(q))`: the
/// branch computes it another way (`exact_length`), which changes last bits. It must
/// stay within two ulp of main's value (measured: 92 of 174 sampled values differ, at
/// most 2 ulp) and no further from the true length.
#[test]
fn the_exact_length_is_within_two_ulp_of_mains() {
    let mut worst = 0u32;
    let mut differing = 0u32;
    let mut n = 0u32;
    let mut x = 1.0e-6f32;
    while x < 1.0e6 {
        for q in [x, x * 1.37, x * 2.9] {
            let main = q * (1.0 / q.sqrt());
            let (len, inv) = exact_length(q);
            let d = (len.to_bits() as i64 - main.to_bits() as i64).unsigned_abs() as u32;
            worst = worst.max(d);
            differing += u32::from(d != 0);
            n += 1;
            let truth = f64::from(q).sqrt();
            assert!(
                (f64::from(len) - truth).abs() <= (f64::from(main) - truth).abs() + f64::EPSILON * truth,
                "q {q}: exact_length {len} is further from sqrt than main's {main}"
            );
            assert!((f64::from(inv) * truth - 1.0).abs() < 4.0e-7, "q {q}: inverse {inv}");
        }
        x *= 1.618;
    }
    eprintln!("exact length: {differing}/{n} values differ from main's formula, at most {worst} ulp");
    assert!(worst <= 2, "exact_length is {worst} ulp from main's q * (1 / sqrt(q))");
}

/// Audit S9: the JBeam loader's collision range is main's 0.03 m everywhere it was
/// used on main (the desc the other soft solver uploads, `ExplicitBeams::new`, the
/// `.truck` export and its spawn); only the explicit solver's JBeam mode (the BeamNG
/// profile, new behaviour of this branch) takes BeamNG's 2.5 cm.
#[test]
fn the_jbeam_loaders_collision_range_is_mains_outside_the_beamng_profile() {
    use engine_softbody::jbeam::load_jbeam;
    use engine_softbody::truck_export::{desc_to_truck, TyreDefaults};
    let text = engine_softbody::jbeam_fixture::latched_box_jbeam();
    let desc = load_jbeam(&[("v.jbeam", text.as_str())], None).unwrap();
    assert_eq!(desc.collision_range, 0.03, "the desc (the soft solver's upload)");
    let plain = ExplicitBeams::new(&desc, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default());
    assert_eq!(plain.collision_range, 0.03, "ExplicitBeams::new");
    let (truck, _) = desc_to_truck(&desc, TyreDefaults::for_vehicle(&desc));
    assert!(truck.lines().any(|l| l.trim() == "set_collision_range 0.03"), "the export's line");
    let spawned = ExplicitBeams::from_truck(&truck, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap();
    assert_eq!(spawned.collision_range, 0.03, "the exported .truck, spawned");
    let (native, _) =
        ExplicitBeams::from_desc_jbeam(&desc, TyreDefaults::for_vehicle(&desc), Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default())
            .unwrap();
    assert_eq!(native.collision_range, BEAMNG_COLLISION_RANGE, "JBeam mode: the BeamNG profile's band");
    // A desc without the profile's band (hand-built, or from an older file) keeps its own.
    let mut own = desc.clone();
    own.explicit_collision_range = 0.0;
    let (native, _) =
        ExplicitBeams::from_desc_jbeam(&own, TyreDefaults::for_vehicle(&own), Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default())
            .unwrap();
    assert_eq!(native.collision_range, 0.03);
}

/// Audit S6 / T1: vehicle contact runs in the owner's frame (`ExplicitBeamsWorld::owner_frame`)
/// instead of World coordinates, so far from the origin the result differs from main. The
/// reference is the same crash at the World origin (every position within metres of 0,
/// f32 resolves 2.4e-7 m there): two repo sedans head on at 10 m/s each, spawned at four
/// offsets about 100 m and four about 360 m out, 25 frames of 33 steps. Per band, the mean
/// over frames 20-25 (after the impact at step 625) of the largest node distance from the
/// origin run, positions rebuilt in f64. Main (30ab0e1d, measured with
/// `examples/far_origin_contact_error.rs`, which builds unchanged on main):
/// 0.0704 m at ~100 m, 0.1157 m at ~360 m. The branch must stay at or below main's error.
/// Branch when written: 0.0688 m and 0.0550 m.
#[test]
fn far_from_the_origin_vehicle_contact_is_no_further_from_the_origin_run_than_mains() {
    use engine_softbody::explicit_contact::ExplicitBeamsWorld;
    const SEDAN: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ));
    const MAIN_100: f64 = 0.0704;
    const MAIN_360: f64 = 0.1157;
    fn world(at: Vec3) -> ExplicitBeamsWorld {
        let mut s = ExplicitBeamsSettings::default();
        s.fast_inv_sqrt = true;
        let a = ExplicitBeams::from_truck(SEDAN, at + Vec3::new(-4.0, 0.3, 0.0), Vec3::new(10.0, 0.0, 0.0), s).unwrap();
        let b = ExplicitBeams::from_truck(SEDAN, at + Vec3::new(4.0, 0.3, 0.4), Vec3::new(-10.0, 0.0, 0.0), s).unwrap();
        ExplicitBeamsWorld::new(vec![a, b])
    }
    fn positions(w: &ExplicitBeamsWorld, at: Vec3) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for v in &w.vehicles {
            for n in &v.nodes {
                let mut p = [0.0f64; 3];
                for k in 0..3 {
                    p[k] = v.origin.to_array()[k] as f64 + n.pos.to_array()[k] as f64 - at.to_array()[k] as f64;
                }
                out.push(p);
            }
        }
        out
    }
    let mut reference = world(Vec3::ZERO);
    let refs: Vec<_> = (0..25)
        .map(|_| {
            reference.frame(33);
            positions(&reference, Vec3::ZERO)
        })
        .collect();
    let band = |offsets: [Vec3; 4]| {
        let (mut sum, mut n) = (0.0f64, 0);
        for at in offsets {
            let mut w = world(at);
            for (f, r) in refs.iter().enumerate() {
                w.frame(33);
                if f + 1 >= 20 {
                    let e = positions(&w, at)
                        .iter()
                        .zip(r)
                        .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt())
                        .fold(0.0f64, f64::max);
                    sum += e;
                    n += 1;
                }
            }
            assert!(w.first_hit_step.is_some(), "the cars met at {at}");
        }
        sum / n as f64
    };
    let near = band([
        Vec3::new(100.0, 0.0, 100.0),
        Vec3::new(71.3, 0.0, 71.9),
        Vec3::new(-64.0, 0.0, 75.5),
        Vec3::new(98.7, 0.0, -29.3),
    ]);
    let far = band([
        Vec3::new(300.0, 0.0, -200.0),
        Vec3::new(255.1, 0.0, 255.3),
        Vec3::new(-180.4, 0.0, 310.2),
        Vec3::new(-350.9, 0.0, -88.6),
    ]);
    eprintln!("mean distance from the origin run: ~100 m {near:.4} m (main {MAIN_100}), ~360 m {far:.4} m (main {MAIN_360})");
    assert!(near <= MAIN_100, "~100 m: {near} m from the origin run, main {MAIN_100} m");
    assert!(far <= MAIN_360, "~360 m: {far} m from the origin run, main {MAIN_360} m");
}
