//! Headless crash tests of the procedural soft-body car on the native wgpu host:
//! no rendering, one configurable scenario, a time series and a summary.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin soft_crash -- wall speed=100
//! cargo run -p engine-webgpu --release --bin soft_crash -- wall speed=64 overlap=0.4
//! cargo run -p engine-webgpu --release --bin soft_crash -- pole speed=60 offset=0.5
//! cargo run -p engine-webgpu --release --bin soft_crash -- tbone speed=60 at=1.2 mass=1500
//! cargo run -p engine-webgpu --release --bin soft_crash -- headon speed=80 overlap=0.5
//! ```
//! Scenarios (`speed` in km/h; the car drives along +Z):
//! * `wall`: rigid wall. `overlap` = share of the car's width that hits it (1 =
//!   full width; 0.25 = the small-overlap test), `angle` = wall yaw (degrees).
//! * `pole`: 25 cm pole, `offset` = lateral offset from the centre line (m).
//! * `tbone`: a `mass` kg block at `speed` into the parked car's left side,
//!   `at` m forward of the car's centre (+2 ≈ the front wheel, −2 ≈ the rear).
//! * `headon`: two cars at `speed` each, `overlap` = share of the width.
//!
//! Car overrides (`deform=`, `strength=`, `cell_deform=`, `cell_strength=`,
//! `spring=`), `threshold=` (default `SoftSettings::calibrated_for(hz)`),
//! `hz=`, `iters=`, `sweeps=`, `seconds=`.
//!
//! Reports per 50 ms: speed, crush (shortening of the car), pitch and roll of
//! the passenger cell, height; then the summary: crush, beams bent and broken,
//! loose pieces (connected parts of the unbroken beam graph besides the car).
use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::procedural::{car_truck, CarParams, CAR_RING};
use engine_softbody::truck::load_truck;
use engine_softbody::{
    soft_scene_params, SoftCapacity, SoftHandle, SoftSettings, SoftVehicleDesc, SpawnOptions,
};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};
use std::collections::HashMap;

struct Args(HashMap<String, f32>);
impl Args {
    fn get(&self, k: &str, d: f32) -> f32 {
        self.0.get(k).copied().unwrap_or(d)
    }
}

fn body(pos: Vec3, rot: Quat, mass: f32, half: [f32; 3]) -> BodyState {
    if mass <= 0.0 {
        let p = pos.to_array();
        let q = rot.to_array();
        return BodyState {
            position: p,
            inv_mass: 0.0,
            orientation: q,
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: p,
            initial_pad: 0.0,
            initial_rot: q,
        };
    }
    let h = Vec3::from_array(half);
    let inertia = Vec3::new(
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ) * (mass / 3.0);
    BodyState::dynamic(pos, rot, mass, inertia)
}

fn add_box(sim: &mut WgpuSim, b: BodyState, half: [f32; 3]) -> (u32, u32) {
    let id = sim.add_body(b);
    let c = sim.add_collider(Collider {
        body: id,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], Vec3::from_array(half).length()],
    });
    (id, c)
}

struct Car {
    h: SoftHandle,
    rest: Vec<Vec3>,
}

/// Mean of ring `k`'s six shell nodes.
fn ring_centre(x: &[Vec3], k: usize) -> Vec3 {
    (0..6).map(|j| x[k * CAR_RING + j]).sum::<Vec3>() / 6.0
}

fn main() {
    let mut words = std::env::args().skip(1);
    let scenario = words.next().unwrap_or_else(|| "wall".into());
    let a = Args(
        words
            .filter_map(|w| {
                let (k, v) = w.split_once('=')?;
                Some((k.to_string(), v.parse().ok()?))
            })
            .collect(),
    );
    let hz = a.get("hz", 240.0);
    let kmh = a.get("speed", 56.0);
    let v = kmh / 3.6;
    let d = CarParams::default();
    let params = CarParams {
        deform: a.get("deform", d.deform),
        strength: a.get("strength", d.strength),
        cell_deform_scale: a.get("cell_deform", d.cell_deform_scale),
        cell_strength_scale: a.get("cell_strength", d.cell_strength_scale),
        spring: a.get("spring", d.spring),
        pillar_strength: a.get("pillar", d.pillar_strength),
        ..d
    };
    let desc: SoftVehicleDesc = engine_softbody::procedural::car_desc(&params).0;
    let stations = params.stations;
    let cap = WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    let mut p = soft_scene_params(hz, a.get("iters", 10.0) as u32);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    let cal = SoftSettings::calibrated_for(hz);
    sim.set_soft_settings(SoftSettings {
        threshold_scale: a.get("threshold", cal.threshold_scale),
        sweeps: a.get("sweeps", 1.0) as u32,
        rigid_mode: a.get("rigidmode", 0.0) > 0.5,
        ..cal
    });
    for (m, f) in [(0, 1.0), (9, 0.8)] {
        sim.set_material(
            m,
            Material {
                friction: f,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
    }
    add_box(
        &mut sim,
        body(
            Vec3::new(0.0, -5.0, 0.0),
            Quat::IDENTITY,
            0.0,
            [400.0, 5.0, 400.0],
        ),
        [400.0, 5.0, 400.0],
    );
    let width = params.width;
    let front = 0.5 * params.length + 0.2; // bumper face
    let spawn = |sim: &mut WgpuSim, pos: Vec3, yaw: f32, vel: Vec3| -> Car {
        let h = sim
            .add_soft_vehicle(
                &desc,
                pos.to_array(),
                Quat::from_rotation_y(yaw).to_array(),
                &SpawnOptions {
                    velocity: vel.to_array(),
                    ..Default::default()
                },
            )
            .expect("spawn");
        let rest = sim
            .read_soft_particles(h)
            .iter()
            .map(|b| Vec3::from_array(b.position))
            .collect();
        Car { h, rest }
    };
    // `back=` m of extra run-up (time to settle, and to turn rigid in rigid mode).
    let gap = 0.3 + a.get("back", 0.0);
    let mut cars = Vec::new();
    let mut ram = None;
    match scenario.as_str() {
        "wall" => {
            // Wall face at z = front + gap; `overlap` of the width (from the left, +X) hits it.
            let overlap = a.get("overlap", 1.0);
            let angle = a.get("angle", 0.0).to_radians();
            let half = [2.0, 1.5, 0.3];
            let edge = 0.5 * width - overlap * width; // the wall spans x ≥ edge
            let centre = Vec3::new(edge + half[0], 1.5, front + gap + half[2]);
            add_box(
                &mut sim,
                body(centre, Quat::from_rotation_y(angle), 0.0, half),
                half,
            );
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::Z * v));
        }
        "pole" => {
            let r = 0.125;
            let pb = sim.add_body(body(
                Vec3::new(a.get("offset", 0.0), 1.5, front + gap + r),
                Quat::IDENTITY,
                0.0,
                [r; 3],
            ));
            let c = sim.add_collider(Collider {
                body: pb,
                shape: ShapeType::Capsule as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [r, 1.5, 0.0, r + 1.5],
            });
            sim.add_soft_obstacle(c).expect("obstacle");
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::Z * v));
        }
        "tbone" => {
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::ZERO));
            for _ in 0..(0.5 * hz) as u32 {
                sim.step().unwrap();
            }
            let half = [0.6, 0.6, 0.9];
            let mut b = body(
                Vec3::new(0.5 * width + gap + half[0], 0.75, a.get("at", 0.0)),
                Quat::IDENTITY,
                a.get("mass", 1500.0),
                half,
            );
            b.linear_vel = [-v, 0.0, 0.0];
            let (id, c) = add_box(&mut sim, b, half);
            sim.add_soft_obstacle(c).expect("obstacle");
            ram = Some(id);
        }
        "headon" => {
            let overlap = a.get("overlap", 1.0);
            let dx = width * (1.0 - overlap);
            cars.push(spawn(
                &mut sim,
                Vec3::new(0.0, 0.05, -front - gap),
                0.0,
                Vec3::Z * v,
            ));
            cars.push(spawn(
                &mut sim,
                Vec3::new(dx, 0.05, front + gap),
                std::f32::consts::PI,
                -Vec3::Z * v,
            ));
        }
        "sidepole" => {
            let r = 0.125;
            let pb = sim.add_body(body(
                Vec3::new(-(0.5 * width + gap + r), 1.5, a.get("at", 0.3)),
                Quat::IDENTITY,
                0.0,
                [r; 3],
            ));
            let c = sim.add_collider(Collider {
                body: pb,
                shape: ShapeType::Capsule as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [r, 1.5, 0.0, r + 1.5],
            });
            sim.add_soft_obstacle(c).expect("obstacle");
            cars.push(spawn(
                &mut sim,
                Vec3::new(0.0, 0.05, 0.0),
                0.0,
                -Vec3::X * v,
            ));
        }
        "rollover" => {
            // Sliding sideways (right side first) into a curb `curb` m tall.
            let hc = 0.5 * a.get("curb", 0.25);
            let half = [0.3, hc, 4.0];
            let (_, c) = add_box(
                &mut sim,
                body(
                    Vec3::new(-(0.5 * width + 1.2) - 0.3, hc, 0.0),
                    Quat::IDENTITY,
                    0.0,
                    half,
                ),
                half,
            );
            sim.add_soft_obstacle(c).expect("obstacle");
            cars.push(spawn(
                &mut sim,
                Vec3::new(0.0, 0.05, 0.0),
                0.0,
                -Vec3::X * v,
            ));
        }
        "ramp" => {
            // A ramp under the left wheels: `rise` m over 4 m.
            let ang = (a.get("rise", 1.1) / 4.0).atan();
            let half = [0.45, 0.2, 2.0];
            let c0 = Vec3::new(
                0.55,
                2.0 * ang.sin() - 0.2 * ang.cos() + 0.02,
                front + 5.0 + 2.0,
            );
            let (_, c) = add_box(
                &mut sim,
                body(c0, Quat::from_rotation_x(-ang), 0.0, half),
                half,
            );
            sim.add_soft_obstacle(c).expect("obstacle");
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::Z * v));
        }
        "drop" => {
            let q = Quat::from_rotation_x(a.get("pitch", 35.0).to_radians());
            let h = sim
                .add_soft_vehicle(
                    &desc,
                    [0.0, a.get("height", 8.0), 0.0],
                    q.to_array(),
                    &SpawnOptions::default(),
                )
                .expect("spawn");
            let rest = sim
                .read_soft_particles(h)
                .iter()
                .map(|b| Vec3::from_array(b.position))
                .collect();
            cars.push(Car { h, rest });
        }
        "limbo" => {
            // A fixed bar across the road at `bar` m (underside): the bonnet passes
            // under it, the pillars and the roof do not.
            let y = a.get("bar", 1.1) + 0.1;
            let (_, c) = add_box(&mut sim, body(Vec3::new(0.0, y, front + 1.0), Quat::IDENTITY, 0.0, [3.0, 0.1, 0.1]), [3.0, 0.1, 0.1]);
            sim.add_soft_obstacle(c).expect("obstacle");
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::Z * v));
        }
        "sandwich" => {
            // The car stopped behind a parked 15 t truck; an 18 t truck at `speed`
            // hits it from behind.
            let half = [1.25, 1.6, 6.0];
            let mut parked = body(Vec3::new(0.0, 1.75, front + 0.4 + half[2]), Quat::IDENTITY, a.get("parked", 15000.0), half);
            parked.linear_vel = [0.0; 3];
            let (_, c1) = add_box(&mut sim, parked, half);
            sim.add_soft_obstacle(c1).expect("obstacle");
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::ZERO));
            let mut ram = body(Vec3::new(0.0, 1.75, -front - 3.0 - half[2]), Quat::IDENTITY, a.get("mass", 18000.0), half);
            ram.linear_vel = [0.0, 0.0, v];
            let (_, c2) = add_box(&mut sim, ram, half);
            sim.add_soft_obstacle(c2).expect("obstacle");
        }
        "crusher" => {
            // A `mass` kg block dropped from `height` m onto the roof.
            let half = [1.2, 0.5, 1.5];
            let (_, c) = add_box(&mut sim, body(Vec3::new(0.0, a.get("height", 4.0) + 1.5, 0.0), Quat::IDENTITY, a.get("mass", 10000.0), half), half);
            sim.add_soft_obstacle(c).expect("obstacle");
            cars.push(spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::ZERO));
        }
        "drive" => {
            // Full throttle from rest on a flat road: the soft car's speed, rpm and
            // gear against the rigid sedan preset (`rigid=1`).
            let rigid = a.get("rigid", 0.0) > 0.5;
            let v = if rigid {
                sim.add_vehicle(&engine_vehicle::presets::sedan(), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("sedan")
            } else {
                let c = spawn(&mut sim, Vec3::new(0.0, 0.05, 0.0), 0.0, Vec3::ZERO);
                sim.soft_host().entry(c.h).core.expect("core").0
            };
            for _ in 0..(0.5 * hz) as u32 {
                sim.step().unwrap();
            }
            sim.set_vehicle_input(v, engine_vehicle::VehicleInput { forward: a.get("throttle", 1.0), ..Default::default() });
            println!("   t (s) | speed (km/h) | rpm | gear | contacts");
            for k in 0..=(a.get("seconds", 10.0) * 4.0) as u32 {
                if k > 0 {
                    for _ in 0..(hz / 4.0) as u32 {
                        sim.step().unwrap();
                    }
                }
                let st = sim.vehicle_state(v);
                println!("{:8.2} | {:12.1} | {:4.0} | {:4} | {}", k as f32 / 4.0, st.forward_speed * 3.6, st.engine_rpm, st.gear, st.contacts);
            }
            return;
        }
        s => panic!("unknown scenario {s}: wall, pole, sidepole, tbone, headon, rollover, ramp, drop, drive"),
    }
    let _ = ram;
    let len0 = |c: &Car| (ring_centre(&c.rest, stations - 1) - ring_centre(&c.rest, 0)).length();
    let lens0: Vec<f32> = cars.iter().map(len0).collect();
    let seconds = a.get("seconds", 2.0);
    let steps = (seconds * hz) as u32;
    let every = (hz / 20.0) as u32;
    let (mut max_pitch, mut max_roll, mut max_height, mut flipped) =
        (0.0f32, 0.0f32, 0.0f32, false);
    println!(
        "{scenario} at {kmh} km/h · {hz} Hz × {} it",
        a.get("iters", 10.0)
    );
    println!("   t (s) | speed (km/h) | crush (m) | pitch (°) | roll (°) | cell height (m)");
    let mut torn: Vec<String> = Vec::new();
    for s in 0..=steps {
        if s > 0 {
            sim.step().unwrap();
        }
        if s % every != 0 {
            continue;
        }
        // Destruction events (engine API): name the parts that tore off.
        for p in sim.poll_soft_events().pieces {
            let who = cars.iter().position(|c| c.h == p.soft_body).unwrap_or(0);
            torn.push(format!("car {who}: {}", if p.parts.is_empty() { format!("{} nodes", p.nodes.len()) } else { p.parts.join("+") }));
        }
        let c = &cars[0];
        let ps = sim.read_soft_particles(c.h);
        let x: Vec<Vec3> = ps.iter().map(|b| Vec3::from_array(b.position)).collect();
        let vel = ps
            .iter()
            .map(|b| Vec3::from_array(b.linear_vel))
            .sum::<Vec3>()
            / ps.len() as f32;
        let crush = lens0[0] - (ring_centre(&x, stations - 1) - ring_centre(&x, 0)).length();
        // Passenger cell frame: rings 3..7.
        let (r0, r1) = (ring_centre(&x, 3), ring_centre(&x, stations - 4));
        let fwd = (r1 - r0).normalize();
        let left = ((3..stations - 3)
            .map(|k| x[k * CAR_RING] - x[k * CAR_RING + 1])
            .sum::<Vec3>())
        .normalize();
        let up = fwd.cross(left).normalize();
        let pitch = fwd.y.asin().to_degrees();
        let roll = left.y.asin().to_degrees();
        flipped |= up.y < 0.0;
        max_pitch = max_pitch.max(pitch.abs());
        max_roll = max_roll.max(roll.abs());
        let height = 0.5 * (r0.y + r1.y);
        max_height = max_height.max(height);
        println!(
            "{:8.2} | {:12.1} | {:9.3} | {:9.1} | {:8.1} | {:8.2}",
            s as f32 / hz,
            vel.length() * 3.6,
            crush,
            pitch,
            roll,
            height
        );
    }
    println!();
    for (i, c) in cars.iter().enumerate() {
        let first = sim.soft_host().entry(c.h).first_body;
        let beams = sim.read_soft_beams(c.h);
        let x: Vec<Vec3> = sim
            .read_soft_particles(c.h)
            .iter()
            .map(|b| Vec3::from_array(b.position))
            .collect();
        let n = x.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], i: usize) -> usize {
            let mut r = i;
            while p[r] != r {
                r = p[r];
            }
            let mut i = i;
            while p[i] != r {
                let nx = p[i];
                p[i] = r;
                i = nx;
            }
            r
        }
        let local = |g: u32| {
            if g >= first {
                (g - first) as usize
            } else {
                g as usize
            }
        };
        for b in beams.iter().filter(|b| !b.broken()) {
            let (u, w) = (local(b.a), local(b.b));
            if u < n && w < n {
                let (ru, rw) = (find(&mut parent, u), find(&mut parent, w));
                parent[ru] = rw;
            }
        }
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            *sizes.entry(find(&mut parent, i)).or_default() += 1;
        }
        let mut parts: Vec<usize> = sizes.into_values().collect();
        parts.sort_unstable_by(|a, b| b.cmp(a));
        let crush = lens0[i] - (ring_centre(&x, stations - 1) - ring_centre(&x, 0)).length();
        let bent = beams
            .iter()
            .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
            .count();
        let broken = beams.iter().filter(|b| b.broken()).count();
        let loose: Vec<usize> = parts.iter().skip(1).copied().collect();
        println!(
            "car {i}: crush {crush:.2} m · bent {bent} / {} · broken {broken} · loose pieces {} (nodes {:?})",
            beams.len(),
            loose.len(),
            loose
        );
    }
    println!("max pitch {max_pitch:.0}° · max roll {max_roll:.0}° · max cell height {max_height:.2} m · rolled over: {flipped}");
    // Self-penetration: own nodes inside (behind and within the prism of) own
    // skin triangles they are not excluded from, by more than 2 cm.
    {
        let x: Vec<Vec3> = sim.read_soft_particles(cars[0].h).iter().map(|b| Vec3::from_array(b.position)).collect();
        let rest: Vec<Vec3> = desc.nodes.iter().map(|n| Vec3::from_array(n.position)).collect();
        let closest = |p: Vec3, a: Vec3, b: Vec3, c: Vec3| -> Vec3 {
            let (ab, ac, ap) = (b - a, c - a, p - a);
            let n = ab.cross(ac);
            let l = n.length();
            if l < 1e-9 {
                return a;
            }
            let n = n / l;
            let q = p - n * n.dot(p - a);
            // barycentric inside test; fall back to vertices/edges roughly
            let v0 = ab; let v1 = ac; let v2 = q - a;
            let (d00, d01, d11, d20, d21) = (v0.dot(v0), v0.dot(v1), v1.dot(v1), v2.dot(v0), v2.dot(v1));
            let den = d00 * d11 - d01 * d01;
            let v = (d11 * d20 - d01 * d21) / den;
            let w = (d00 * d21 - d01 * d20) / den;
            if v >= 0.0 && w >= 0.0 && v + w <= 1.0 { q } else { Vec3::splat(f32::MAX) }
        };
        let mut bad = 0;
        for (i, &p) in x.iter().enumerate() {
            for t in &desc.triangles {
                if t.nodes.contains(&(i as u32)) {
                    continue;
                }
                let [ra, rb, rc] = t.nodes.map(|k| rest[k as usize]);
                let rq = closest(rest[i], ra, rb, rc);
                let rn = (rb - ra).cross(rc - ra).normalize_or_zero();
                if rq.x != f32::MAX && (rest[i] - ra).dot(rn).abs() < 0.12 {
                    continue; // excluded: it sat over this face at rest
                }
                let [a, b, c] = t.nodes.map(|k| x[k as usize]);
                let n = (b - a).cross(c - a).normalize_or_zero();
                let d = (p - a).dot(n);
                if d < -0.02 && d > -0.3 && closest(p, a, b, c).x != f32::MAX {
                    bad += 1;
                }
            }
        }
        println!("self-penetration: {bad} node-triangle pairs (own body)");
    }
    if let Some(roof) = desc.part("roof") {
        let first = sim.soft_host().entry(cars[0].h).first_body;
        let is_roof = |g: u32| roof.nodes.contains(&(g - first));
        let beams = sim.read_soft_beams(cars[0].h);
        let pillars: Vec<_> = beams.iter().filter(|b| is_roof(b.a) != is_roof(b.b)).collect();
        println!("roof pillars: {} of {} broken", pillars.iter().filter(|b| b.broken()).count(), pillars.len());
    }
    println!("torn off: {}", if torn.is_empty() { "nothing".to_string() } else { torn.join(", ") });
}
