//! Directed debris: guided bodies in free flight are steered toward authored
//! zones by a launch kick inside a small cone and a constant extra
//! acceleration, and never once they touch anything. The tests check the
//! invariants that make it read as physics: off changes nothing, the extra
//! acceleration stays within its limit and stays constant (no visible
//! homing), spin is untouched, and the arc ends at the first contact.

use rubble::*;

const DT: f32 = 1.0 / 60.0;
const G: f32 = -10.0;

fn world(guidance: Option<GuidanceCapacity>, ground: bool) -> World {
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            guidance,
            ..WorldDef::default()
        },
    )
    .unwrap();
    if ground {
        let g = w
            .create_body(&BodyDef {
                position: [0.0, -0.5, 0.0],
                ..BodyDef::default()
            })
            .unwrap();
        w.create_box_shape(g, &ShapeDef::default(), [80.0, 0.5, 80.0])
            .unwrap();
    }
    w
}

fn guided_world(ground: bool) -> World {
    world(Some(GuidanceCapacity::default()), ground)
}

fn chunk(w: &mut World, position: [f32; 3], velocity: [f32; 3]) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            linear_velocity: velocity,
            ..BodyDef::dynamic(position)
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.15; 3])
        .unwrap();
    b
}

/// Every body's GPU state after each of `steps` steps.
fn record(w: &mut World, steps: u32) -> Vec<Vec<engine_schema::BodyState>> {
    let mut out = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        w.step().unwrap();
        w.poll().unwrap();
        out.push(w.sim().download_bodies());
    }
    w.poll_blocking().unwrap();
    out
}

/// Where body `b` first crosses height `y` going down (linear between steps).
fn crossing(track: &[Vec<engine_schema::BodyState>], b: BodyId, y: f32) -> Option<[f32; 3]> {
    let i = b.index() as usize;
    for k in 1..track.len() {
        let (p0, p1) = (track[k - 1][i].position, track[k][i].position);
        if p0[1] >= y && p1[1] < y {
            let s = (p0[1] - y) / (p0[1] - p1[1]);
            return Some([
                p0[0] + s * (p1[0] - p0[0]),
                y,
                p0[2] + s * (p1[2] - p0[2]),
            ]);
        }
    }
    None
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn zone_at(p: [f32; 3], shape: ZoneShape) -> GuidanceZone {
    GuidanceZone::new(ZoneAnchor::World(p), shape)
}

fn enable(w: &mut World, settings: GuidanceSettings, zones: &[GuidanceZone]) {
    w.set_guidance(&settings).unwrap();
    w.set_guidance_zones(zones).unwrap();
}

/// A ~2.9 s flight from 1 m up.
const LAUNCH_P: [f32; 3] = [0.0, 1.0, 0.0];
const LAUNCH_V: [f32; 3] = [10.0, 14.0, 0.0];

fn natural_landing() -> [f32; 3] {
    let mut w = world(None, false);
    let b = chunk(&mut w, LAUNCH_P, LAUNCH_V);
    let t = record(&mut w, 240);
    crossing(&t, b, 0.0).expect("lands")
}

#[test]
fn guidance_off_changes_nothing() {
    let run = |mode: u32| {
        let mut w = if mode == 0 { world(None, true) } else { guided_world(true) };
        let bodies: Vec<BodyId> = (0..12)
            .map(|k| {
                let x = (k % 4) as f32 * 0.31 - 0.5;
                let y = 0.2 + (k / 4) as f32 * 0.31;
                chunk(&mut w, [x, y, 0.0], [3.0, 6.0 + k as f32 * 0.1, 0.5])
            })
            .collect();
        let target = zone_at([20.0, 0.0, 2.0], ZoneShape::Sphere { radius: 1.0 });
        match mode {
            // Enabled with zones, but no body is guided.
            1 => enable(&mut w, GuidanceSettings::default(), &[target]),
            // Guided bodies and zones, but disabled.
            2 => {
                enable(
                    &mut w,
                    GuidanceSettings {
                        enabled: false,
                        ..GuidanceSettings::default()
                    },
                    &[target],
                );
                for &b in &bodies {
                    w.body_set_guided(b, true).unwrap();
                }
            }
            // Enabled and guided, but no zones.
            3 => {
                enable(&mut w, GuidanceSettings::default(), &[]);
                for &b in &bodies {
                    w.body_set_guided(b, true).unwrap();
                }
            }
            _ => {}
        }
        let t = record(&mut w, 150);
        t.last().unwrap().clone()
    };
    let base = run(0);
    for mode in 1..=3 {
        let other = run(mode);
        assert_eq!(base.len(), other.len());
        for (a, b) in base.iter().zip(&other) {
            assert_eq!(
                bytemuck::bytes_of(a),
                bytemuck::bytes_of(b),
                "mode {mode}: bitwise the same as a world without guidance"
            );
        }
    }
}

#[test]
fn a_long_flight_lands_in_a_reachable_zone_and_skips_an_unreachable_one() {
    let natural = natural_landing();
    // 3 m to the side of where it would land: inside the budget.
    let near = [natural[0] - 1.0, 0.0, natural[2] + 2.8];
    let mut w = guided_world(false);
    let b = chunk(&mut w, LAUNCH_P, LAUNCH_V);
    w.body_set_guided(b, true).unwrap();
    enable(
        &mut w,
        GuidanceSettings::default(),
        &[zone_at(near, ZoneShape::Point)],
    );
    let t = record(&mut w, 240);
    let land = crossing(&t, b, 0.0).unwrap();
    assert!(
        dist(natural, near) > 2.5,
        "the zone is off the natural landing: {natural:?} vs {near:?}"
    );
    assert!(
        dist(land, near) < 0.35,
        "guided landing {land:?} reaches {near:?} (natural {natural:?})"
    );

    // 9 m away: out of budget, so the body keeps its own arc exactly.
    let far = [natural[0], 0.0, natural[2] + 9.0];
    let mut w = guided_world(false);
    let b = chunk(&mut w, LAUNCH_P, LAUNCH_V);
    w.body_set_guided(b, true).unwrap();
    enable(
        &mut w,
        GuidanceSettings::default(),
        &[zone_at(far, ZoneShape::Point)],
    );
    let t = record(&mut w, 240);
    let land = crossing(&t, b, 0.0).unwrap();
    assert!(
        dist(land, natural) < 1e-4,
        "unreachable: landed at {land:?}, natural {natural:?}"
    );
    assert_eq!(w.body_guidance(b).unwrap().state, GuideState::Done);
}

#[test]
fn steering_stays_inside_its_budget_and_holds_still() {
    let natural = natural_landing();
    let target = [natural[0] + 1.5, 0.0, natural[2] - 2.0];
    let settings = GuidanceSettings::default();
    let mut w = guided_world(false);
    let spin = [1.0, 2.0, 3.0];
    let b = w
        .create_body(&BodyDef {
            linear_velocity: LAUNCH_V,
            angular_velocity: spin,
            ..BodyDef::dynamic(LAUNCH_P)
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.15; 3])
        .unwrap();
    w.body_set_guided(b, true).unwrap();
    enable(
        &mut w,
        settings.clone(),
        &[zone_at(target, ZoneShape::Point)],
    );
    let t = record(&mut w, 170);

    // The unguided twin, for spin.
    let mut u = world(None, false);
    let ub = u
        .create_body(&BodyDef {
            linear_velocity: LAUNCH_V,
            angular_velocity: spin,
            ..BodyDef::dynamic(LAUNCH_P)
        })
        .unwrap();
    u.create_box_shape(ub, &ShapeDef::default(), [0.15; 3])
        .unwrap();
    let ut = record(&mut u, 170);

    let i = b.index() as usize;
    let speed = (LAUNCH_V[0].powi(2) + LAUNCH_V[1].powi(2)).sqrt();
    let amax = settings.accel_max_g * G.abs();
    let mut kick = [0.0f32; 3];
    let mut held: Option<[f32; 3]> = None;
    for k in 1..t.len() {
        let (v0, v1) = (t[k - 1][i].linear_vel, t[k][i].linear_vel);
        let a = [
            (v1[0] - v0[0]) / DT,
            (v1[1] - v0[1]) / DT - G,
            (v1[2] - v0[2]) / DT,
        ];
        let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        if mag > amax * 1.05 {
            // A kick step: its velocity change counts toward the launch kick.
            for c in 0..3 {
                kick[c] += (a[c] - held.map_or(0.0, |h| h[c])) * DT;
            }
            assert!(k < 12, "kicks only right after launch (step {k})");
            continue;
        }
        if mag > 1e-3 {
            match held {
                None => held = Some(a),
                Some(h) => {
                    let d = ((a[0] - h[0]).powi(2) + (a[1] - h[1]).powi(2) + (a[2] - h[2]).powi(2))
                        .sqrt();
                    assert!(
                        d < 0.02 * amax + 2e-3,
                        "step {k}: held acceleration {a:?} drifted from {h:?}"
                    );
                }
            }
        }
        let wv = t[k][i].angular_vel;
        let uv = ut[k][ub.index() as usize].angular_vel;
        for c in 0..3 {
            assert!((wv[c] - uv[c]).abs() < 1e-5, "step {k}: spin untouched {wv:?} vs {uv:?}");
        }
    }
    let held = held.expect("an acceleration was held");
    let hm = (held[0].powi(2) + held[1].powi(2) + held[2].powi(2)).sqrt();
    assert!(hm <= amax * 1.02, "held {hm} m/s² within {amax}");
    let km = (kick[0].powi(2) + kick[1].powi(2) + kick[2].powi(2)).sqrt();
    let budget = speed * settings.launch_kick.min(settings.launch_cone.tan());
    assert!(km <= budget * 1.05 + 1e-3, "kick {km} within {budget}");
    let cone = (km / speed).atan();
    assert!(cone <= settings.launch_cone * 1.05, "direction change {cone} rad");
    assert!(km > 0.0, "the kick was used");
}

#[test]
fn a_burst_lands_around_a_ring_not_on_its_centre() {
    // 48 chunks from separate launch points whose natural arcs all come down
    // on one spot (the ring's centre): without guidance they pile up there.
    let centre = [20.0, 0.0, 0.0];
    let mut w = guided_world(false);
    let mut u = world(None, false);
    let mut bodies = Vec::new();
    let mut ubodies = Vec::new();
    for k in 0..48 {
        let ang = k as f32 / 48.0 * std::f32::consts::TAU;
        let p = [6.0 * ang.cos(), 1.0 + (k % 3) as f32, 6.0 * ang.sin()];
        let t = 3.0 + (k % 4) as f32 * 0.05;
        let v = [
            (centre[0] - p[0]) / t,
            (centre[1] - p[1] - 0.5 * G * t * t) / t,
            (centre[2] - p[2]) / t,
        ];
        let b = chunk(&mut w, p, v);
        w.body_set_guided(b, true).unwrap();
        bodies.push(b);
        ubodies.push(chunk(&mut u, p, v));
    }
    let ut = record(&mut u, 240);
    let natural: Vec<[f32; 3]> = ubodies.iter().map(|&b| crossing(&ut, b, 0.0).unwrap()).collect();
    enable(
        &mut w,
        GuidanceSettings::default(),
        &[zone_at(
            centre,
            ZoneShape::Ring {
                inner: 2.0,
                outer: 3.5,
                height: 0.0,
            },
        )],
    );
    let t = record(&mut w, 240);
    let radial = |p: [f32; 3]| ((p[0] - centre[0]).powi(2) + (p[2] - centre[2]).powi(2)).sqrt();
    let mut quadrants = [0u32; 4];
    let mut in_ring = 0;
    for &b in &bodies {
        let p = crossing(&t, b, 0.0).unwrap();
        if (1.8..3.7).contains(&radial(p)) {
            in_ring += 1;
        }
        let q = ((p[2] - centre[2]).atan2(p[0] - centre[0]) + std::f32::consts::PI)
            / std::f32::consts::FRAC_PI_2;
        quadrants[(q as usize).min(3)] += 1;
    }
    let natural_in = natural.iter().filter(|&&p| (1.8..3.7).contains(&radial(p))).count();
    assert!(natural_in == 0, "{natural_in}/48 would land in the ring unguided");
    assert!(in_ring >= 40, "{in_ring}/48 guided land in the ring");
    assert!(
        quadrants.iter().all(|&c| c >= 6),
        "spread all around the ring: {quadrants:?}"
    );
}

fn two_zone_run(quota: Option<u32>, budget: u32) -> (Vec<u32>, Vec<engine_schema::BodyState>) {
    let mut w = world(
        Some(GuidanceCapacity {
            plan_budget: budget,
            ..GuidanceCapacity::default()
        }),
        false,
    );
    let mut bodies = Vec::new();
    for k in 0..60 {
        let p = [(k % 10) as f32 * 0.5, 1.0 + (k / 10) as f32 * 0.5, 0.0];
        let b = chunk(&mut w, p, LAUNCH_V);
        w.body_set_guided(b, true).unwrap();
        bodies.push(b);
    }
    let natural = natural_landing();
    let a = GuidanceZone {
        weight: 2.0,
        quota,
        ..zone_at([natural[0] + 2.0, 0.0, natural[2] + 1.0], ZoneShape::Sphere { radius: 0.5 })
    };
    let b = zone_at([natural[0] + 2.0, 0.0, natural[2] - 1.0], ZoneShape::Sphere { radius: 0.5 });
    enable(&mut w, GuidanceSettings::default(), &[a, b]);
    let t = record(&mut w, 60);
    (w.guidance_claims(), t.last().unwrap().clone())
}

#[test]
fn zones_share_a_burst_by_weight_and_quota_deterministically() {
    let (claims, last) = two_zone_run(None, 256);
    assert_eq!(claims.iter().sum::<u32>(), 60, "every body chose a zone: {claims:?}");
    assert!(
        (30..=50).contains(&claims[0]),
        "weights 2:1 share about 40:20, got {claims:?}"
    );
    let (again, last2) = two_zone_run(None, 256);
    assert_eq!(claims, again, "the same choices every run");
    assert_eq!(bytemuck::cast_slice::<_, u8>(&last), bytemuck::cast_slice::<_, u8>(&last2));

    let (claims, _) = two_zone_run(Some(10), 256);
    assert_eq!(claims[0], 10, "the quota caps zone 0: {claims:?}");
    // Turned away by the quota, a body plans again for zone 1 (unless zone 1
    // is out of its reach).
    assert!(claims[1] >= 45, "the rest go to zone 1: {claims:?}");
}

#[test]
fn a_backed_up_queue_stays_deterministic_and_late_plans_skip_the_kick() {
    // 60 candidates at 8 plans per step: the queue drains over ~8 steps.
    // Bodies planned after the launch window have the held acceleration only,
    // so fewer reach the zones than with an unlimited budget.
    let (claims, last) = two_zone_run(None, 8);
    let (again, last2) = two_zone_run(None, 8);
    assert_eq!(claims, again);
    assert_eq!(bytemuck::cast_slice::<_, u8>(&last), bytemuck::cast_slice::<_, u8>(&last2));
    let planned = claims.iter().sum::<u32>();
    assert!((30..60).contains(&planned), "late plans lose reach: {claims:?}");

    // Planned after the launch window: no kick, only the held acceleration.
    let mut w = world(
        Some(GuidanceCapacity {
            plan_budget: 1,
            ..GuidanceCapacity::default()
        }),
        false,
    );
    let mut bodies = Vec::new();
    for k in 0..12 {
        let b = chunk(&mut w, [k as f32 * 0.5, 1.0, 0.0], LAUNCH_V);
        w.body_set_guided(b, true).unwrap();
        bodies.push(b);
    }
    let natural = natural_landing();
    let settings = GuidanceSettings::default();
    enable(
        &mut w,
        settings.clone(),
        &[zone_at(
            // The last body's natural landing is 5.5 m further along x; its
            // aim is 0.7 m off it, within the acceleration-only reach.
            [natural[0] + 5.5 + 0.5, 0.0, natural[2] + 0.5],
            ZoneShape::Point,
        )],
    );
    let t = record(&mut w, 40);
    let amax = settings.accel_max_g * G.abs();
    let last = *bodies.last().unwrap();
    let i = last.index() as usize;
    for k in 1..t.len() {
        let (v0, v1) = (t[k - 1][i].linear_vel, t[k][i].linear_vel);
        let a = [(v1[0] - v0[0]) / DT, (v1[1] - v0[1]) / DT - G, (v1[2] - v0[2]) / DT];
        let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        assert!(mag <= amax * 1.05 + 1e-3, "step {k}: no kick, {mag} m/s²");
    }
    assert_eq!(w.body_guidance(last).unwrap().state, GuideState::Steering);
}

#[test]
fn a_zone_on_a_moving_body_is_led() {
    let natural = natural_landing();
    let t_flight = 2.85;
    let speed = 3.0;
    // The anchor drives along +Z and reaches 2.5 m beside the natural landing
    // as the debris comes down.
    let mut w = guided_world(false);
    let car = w
        .create_body(&BodyDef {
            body_type: BodyType::Kinematic,
            position: [natural[0] - 1.5, 0.0, natural[2] + 2.0 - speed * t_flight],
            linear_velocity: [0.0, 0.0, speed],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(car, &ShapeDef::default(), [0.9, 0.5, 2.0])
        .unwrap();
    // The aim point is on the ground beside the car, clear of its low box.
    let b = chunk(&mut w, LAUNCH_P, LAUNCH_V);
    w.body_set_guided(b, true).unwrap();
    let zone = GuidanceZone {
        offset: [-1.5, 0.0, 0.0],
        anchor_relative: true,
        ..GuidanceZone::new(ZoneAnchor::Body(car), ZoneShape::Point)
    };
    enable(&mut w, GuidanceSettings::default(), &[zone]);
    let t = record(&mut w, 240);
    let land = crossing(&t, b, 0.0).unwrap();
    // Where the car was at that moment.
    let k = t
        .iter()
        .position(|s| s[b.index() as usize].position[1] < 0.0)
        .unwrap();
    let c = t[k][car.index() as usize].position;
    // Anchor-relative +Z is the car's forward and +X its left: -1.5 m in X
    // is beside it on the right.
    let aim = [c[0] - 1.5, 0.0, c[2]];
    assert!(
        dist(land, aim) < 0.6,
        "landed {land:?} beside the moving car's aim {aim:?} (natural {natural:?})"
    );
}

#[test]
fn a_fast_car_gathers_debris_coming_down_beside_it() {
    // Gauntlet's case: a car at 10 m/s with a ring zone all around it, and
    // blocks blasted across the street from a facade 6 m up. Their arcs pass
    // high over where the car is earlier, so the hit is timed where each arc
    // comes down to the ring's height, not at its closest approach in 3D.
    let speed = 10.0;
    let flight = (2.6 + (2.6f32 * 2.6 + 120.0).sqrt()) / 10.0;
    let mut w = guided_world(false);
    let car = w
        .create_body(&BodyDef {
            body_type: BodyType::Kinematic,
            position: [-2.5, 0.5, -speed * flight],
            linear_velocity: [0.0, 0.0, speed],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(car, &ShapeDef::default(), [0.9, 0.5, 2.0])
        .unwrap();
    // Natural touchdowns from 3 m behind to 7 m ahead of the car.
    let ahead = [-3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0, 7.0];
    let blocks: Vec<BodyId> = ahead
        .iter()
        .map(|&a| {
            let b = chunk(&mut w, [8.0, 6.0, a], [-6.5, 2.6, 0.0]);
            w.body_set_guided(b, true).unwrap();
            b
        })
        .collect();
    let ring = GuidanceZone {
        offset: [0.0, -0.5, 0.0],
        anchor_relative: true,
        ..GuidanceZone::new(
            ZoneAnchor::Body(car),
            ZoneShape::Ring {
                inner: 2.5,
                outer: 6.0,
                height: 0.0,
            },
        )
    };
    let settings = GuidanceSettings {
        launch_kick: 0.12,
        launch_cone: 6.0f32.to_radians(),
        accel_max_g: 0.08,
        min_flight: 0.5,
        ..GuidanceSettings::default()
    };
    enable(&mut w, settings, &[ring]);
    let t = record(&mut w, 150);
    for (&b, &a) in blocks.iter().zip(&ahead) {
        let land = crossing(&t, b, 0.0).unwrap();
        let k = t
            .iter()
            .position(|s| s[b.index() as usize].position[1] < 0.0)
            .unwrap();
        let c = t[k][car.index() as usize].position;
        let d = ((land[0] - c[0]).powi(2) + (land[2] - c[2]).powi(2)).sqrt();
        let g = w.body_guidance(b).unwrap();
        assert_ne!(g.end, GuideEnd::NoZone, "block {a} m ahead found no zone");
        assert!(
            (2.2..=6.3).contains(&d),
            "block {a} m ahead came down {d} m from the car ({land:?}, car {c:?})"
        );
    }
}

#[test]
fn guidance_ends_at_the_first_contact() {
    let natural = natural_landing();
    let mut w = guided_world(true);
    let b = chunk(&mut w, LAUNCH_P, LAUNCH_V);
    w.body_set_guided(b, true).unwrap();
    // Two chunks welded together (a bond still intact) fly as one piece:
    // never steered.
    let pair_a = chunk(&mut w, [0.0, 3.0, -3.0], LAUNCH_V);
    let pair_b = chunk(&mut w, [0.0, 3.3, -3.0], LAUNCH_V);
    let mut base = JointBase::new(pair_a, pair_b);
    base.local_frame_a = Transform::new([0.0, 0.15, 0.0], [0.0, 0.0, 0.0, 1.0]);
    base.local_frame_b = Transform::new([0.0, -0.15, 0.0], [0.0, 0.0, 0.0, 1.0]);
    w.create_joint(&JointDef::Weld(WeldJointDef {
        base,
        linear_stiffness: None,
        angular_stiffness: None,
        plastic: None,
    }))
    .unwrap();
    w.body_set_guided(pair_a, true).unwrap();
    w.body_set_guided(pair_b, true).unwrap();
    enable(
        &mut w,
        GuidanceSettings::default(),
        &[zone_at(
            [natural[0] + 1.0, 0.0, natural[2] + 1.0],
            ZoneShape::Point,
        )],
    );
    record(&mut w, 60);
    assert_eq!(w.body_guidance(b).unwrap().state, GuideState::Steering);
    for p in [pair_a, pair_b] {
        assert_eq!(w.body_guidance(p).unwrap().state, GuideState::Armed, "welded: not free");
    }
    let t = record(&mut w, 200);
    let g = w.body_guidance(b).unwrap();
    assert_eq!(g.state, GuideState::Done, "it landed: {g:?}");
    // Sliding on the ground afterwards, it feels nothing but friction: no
    // sideways (z) acceleration beyond what friction gives.
    let i = b.index() as usize;
    let land = t
        .iter()
        .position(|s| s[i].position[1] < 0.2)
        .expect("landed");
    for k in land + 2..t.len() {
        let dvz = (t[k][i].linear_vel[2] - t[k - 1][i].linear_vel[2]) / DT;
        let vz = t[k - 1][i].linear_vel[2];
        assert!(
            dvz * vz <= 1e-4,
            "step {k}: after contact z velocity only decays ({vz} -> {dvz} m/s²)"
        );
    }
}

#[test]
fn a_raised_box_zone_lifts_the_arc() {
    // A "third floor" 6 m up on a facade the arc would pass under.
    let mut u = world(None, false);
    let ub = chunk(&mut u, LAUNCH_P, LAUNCH_V);
    let ut = record(&mut u, 240);
    // Height of the natural arc at x = 24.
    let x = 24.0;
    let h = |track: &[Vec<engine_schema::BodyState>], i: usize| {
        (1..track.len())
            .find_map(|k| {
                let (a, b) = (track[k - 1][i].position, track[k][i].position);
                (a[0] <= x && b[0] > x).then(|| a[1] + (x - a[0]) / (b[0] - a[0]) * (b[1] - a[1]))
            })
            .unwrap()
    };
    let natural_h = h(&ut, ub.index() as usize);
    let floor = natural_h + 2.0;
    let mut w = guided_world(false);
    let b = chunk(&mut w, LAUNCH_P, LAUNCH_V);
    w.body_set_guided(b, true).unwrap();
    enable(
        &mut w,
        GuidanceSettings::default(),
        &[zone_at(
            [x, floor, 0.0],
            ZoneShape::Box {
                half_extents: [0.2, 0.4, 2.0],
            },
        )],
    );
    let t = record(&mut w, 240);
    let guided_h = h(&t, b.index() as usize);
    assert!(
        (guided_h - floor).abs() < 0.5,
        "at x = {x} the arc passes {guided_h} m up, the floor is {floor} (natural {natural_h})"
    );
}

#[test]
fn guidance_needs_a_capacity_and_validates_zones() {
    let mut w = world(None, false);
    assert!(w.set_guidance(&GuidanceSettings::default()).is_err());
    let too_many_plans = World::new(
        GpuContext::Default,
        &WorldDef {
            guidance: Some(GuidanceCapacity {
                max_zones: 8,
                plan_budget: 257,
            }),
            ..WorldDef::default()
        },
    );
    assert!(too_many_plans.is_err(), "plan_budget is capped at 256 per step");
    let mut w = world(
        Some(GuidanceCapacity {
            max_zones: 2,
            plan_budget: 4,
        }),
        false,
    );
    w.set_guidance(&GuidanceSettings::default()).unwrap();
    let z = zone_at([0.0; 3], ZoneShape::Point);
    let err = w.set_guidance_zones(&[z.clone(), z.clone(), z.clone()]).unwrap_err();
    assert!(err.to_string().contains("max_zones"), "{err}");
    let bad = GuidanceZone {
        weight: f32::NAN,
        ..z.clone()
    };
    assert!(w.set_guidance_zones(&[bad]).is_err());
    let ring = zone_at(
        [0.0; 3],
        ZoneShape::Ring {
            inner: 3.0,
            outer: 1.0,
            height: 0.0,
        },
    );
    assert!(w.set_guidance_zones(&[ring]).is_err());
    assert!(w
        .set_guidance(&GuidanceSettings {
            participation: 1.5,
            ..GuidanceSettings::default()
        })
        .is_err());
}
