//! Vehicle ruts on a soil heightfield (deformable terrain T5,
//! `engine/docs/deformable-terrain.md` "T5 ruts"). Expectations come from the
//! Bekker–Wong relations (`engine_schema::soil`, itself checked against the
//! rigid-wheel arc integrals), never from another program:
//! - a wheel loaded quasi-statically rests at the rigid-wheel sinkage of its
//!   load, in a rut that deep;
//! - the compaction resistance it reports is b·k·z^(n+1)/(n+1) for that
//!   sinkage, and a coasting car on the soil decelerates more than on the same
//!   field without soil by ΣR_c over its mass;
//! - the rut follows the driven path at the sinkage of the load that passed,
//!   as wide as the tyre (to the grid); a second pass leaves it unchanged
//!   wherever its own sinkage is no deeper (multipass compaction is not
//!   modelled), deepens it only to its own sinkage elsewhere, and pays almost
//!   no compaction resistance;
//! - rolling at a steady load, the sinkage is the static one of that load.
//! - a load past the soil's max sinkage fails the step by name.

use engine_schema::commands::GpuCommand;
use engine_schema::convex::{convex_params_base, HullArena};
use engine_schema::heightfield::HeightField;
use engine_schema::soil::Soil;
use engine_schema::{AvbdParams, Material, WorldCapacity, OVERFLOW_SOIL_SINKAGE};
use engine_vehicle::{presets, DriveDesc, VehicleDesc, VehicleHandle, VehicleState};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const G: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;
/// Field cell (m): the sedan's 0.22 m tyre spans four cells.
const CELL: f32 = 0.05;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32_768,
        max_joints: 64,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn params(gravity: f32) -> AvbdParams {
    let mut p = AvbdParams {
        dt: DT,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = -gravity;
    p.sleep_time = 1.0e9;
    p
}

/// A flat field at y = 0 (`nx` × `nz` samples, `CELL` apart), with or without soil.
fn world(nx: u32, nz: u32, soil: Option<Soil>) -> (WgpuSim, HeightField, u32) {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    sim.set_params(params(G));
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    let mut field = HeightField::new(nx, nz, CELL, CELL, vec![0.0; (nx * nz) as usize]).unwrap();
    if let Some(s) = soil {
        field = field.with_soil(s).unwrap();
    }
    let mut arena = HullArena::new();
    let (_, collider) = sim
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    (sim, field, collider)
}

/// The sedan coasting: no drivetrain (no engine braking), no aero drag.
fn car() -> VehicleDesc {
    let mut d = presets::sedan();
    d.drive = DriveDesc::Direct {
        max_torque: 0.0,
        top_speed: 1.0,
    };
    d.aero = Default::default();
    d.assists = Default::default();
    d
}

fn mass_of(desc: &VehicleDesc) -> f32 {
    desc.parts.iter().map(|p| p.mass).sum::<f32>() + desc.wheels.iter().map(|w| w.mass).sum::<f32>()
}

/// Mass plus the wheels' spin inertia seen at the contact (Σ I/R²): what a
/// longitudinal force on the tyres decelerates.
fn effective_mass(desc: &VehicleDesc) -> f32 {
    mass_of(desc)
        + desc
            .wheels
            .iter()
            .map(|w| {
                w.inertia.unwrap_or(0.5 * w.mass * w.radius * w.radius) / (w.radius * w.radius)
            })
            .sum::<f32>()
}

fn chassis(sim: &WgpuSim, v: VehicleHandle) -> u32 {
    sim.vehicle_host().chassis_body(v)
}

/// Spawn the car above (`x`, `z`), lower it until its wheels just touch, and
/// load it quasi-statically: gravity eases (smoothstep) from 0 to g over `ramp` s, then
/// holds for `hold` s. Returns the largest load each wheel saw.
fn settle(
    sim: &mut WgpuSim,
    desc: &VehicleDesc,
    x: f32,
    z: f32,
    ramp: f32,
    hold: f32,
) -> (VehicleHandle, Vec<f32>) {
    sim.set_params(params(0.0));
    let v = sim
        .add_vehicle(desc, [x, 1.0, z], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle");
    sim.step().expect("step");
    let st = sim.vehicle_state(v);
    let bottom = st
        .wheels
        .iter()
        .map(|w| w.position[1] - w.radius)
        .fold(f32::INFINITY, f32::min);
    let body = chassis(sim, v);
    let b = sim.read_body(body);
    let mut p = b.position;
    p[1] -= bottom - 0.002;
    sim.push_body_commands(&[
        GpuCommand::teleport(body, p, b.orientation),
        GpuCommand::set_velocity(body, [0.0; 3], [0.0; 3]),
    ])
    .unwrap();
    let mut peak = vec![0.0f32; desc.wheels.len()];
    let steps = ((ramp + hold) / DT).round() as u32;
    for k in 0..steps {
        let t = k as f32 * DT;
        let u = (t / ramp).min(1.0);
        sim.set_params(params(G * u * u * (3.0 - 2.0 * u)));
        sim.step().expect("step");
        for (i, w) in sim.vehicle_state(v).wheels.iter().enumerate() {
            peak[i] = peak[i].max(w.soil_load);
        }
    }
    sim.set_params(params(G));
    (v, peak)
}

/// (heights, uncompacted heights) of a soil field as the GPU holds them.
fn planes(sim: &WgpuSim, collider: u32, field: &HeightField) -> (Vec<f32>, Vec<f32>) {
    let base = convex_params_base(sim.collider(collider).params);
    let (nx, nz) = field.samples();
    let n = (nx * nz) as usize;
    let plane = field.plane_records();
    let read = |first: u32| -> Vec<f32> {
        sim.read_words(18, u64::from(first) * 4, n)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    };
    let h = read(base + 2);
    let fresh = if field.soil().is_some() {
        read(base + 4 + plane)
    } else {
        vec![0.0; n]
    };
    (h, fresh)
}

/// Rut depth (uncompacted − height) at every sample within `r` of (x, z).
fn deepest_near(field: &HeightField, depth: &[f32], x: f32, z: f32, r: f32) -> f32 {
    let (nx, _) = field.samples();
    let mut best = 0.0f32;
    for (k, d) in depth.iter().enumerate() {
        let p = field.vertex(k as u32 % nx, k as u32 / nx);
        if (p.x - x).powi(2) + (p.z - z).powi(2) <= r * r {
            best = best.max(*d);
        }
    }
    best
}

fn depths(sim: &WgpuSim, collider: u32, field: &HeightField) -> Vec<f32> {
    let (h, fresh) = planes(sim, collider, field);
    fresh.iter().zip(&h).map(|(f, h)| f - h).collect()
}

#[test]
fn a_loaded_wheel_rests_at_the_rigid_wheel_sinkage_of_its_load() {
    let soil = Soil::sandy_loam();
    let (mut sim, field, collider) = world(101, 101, Some(soil));
    let desc = car();
    let (v, peak) = settle(&mut sim, &desc, 0.0, 0.0, 3.0, 2.0);
    let st = sim.vehicle_state(v);
    let depth = depths(&sim, collider, &field);
    let total: f32 = st.wheels.iter().map(|w| w.soil_load).sum();
    let weight = mass_of(&desc) * G;
    eprintln!("total load {total:.0} N vs weight {weight:.0} N");
    assert!(
        (total - weight).abs() < 0.02 * weight,
        "the soil carries the car: {total} vs {weight}"
    );
    for (i, w) in st.wheels.iter().enumerate() {
        let (b, r) = (w.width, w.radius);
        // Quasi-static: the load at rest is within 2 % of the largest it saw
        // (the suspension settling after the ramp).
        eprintln!(
            "wheel {i}: peak load {:.0} N, at rest {:.0} N",
            peak[i], w.soil_load
        );
        assert!(
            peak[i] < 1.02 * w.soil_load,
            "wheel {i}: peak {} vs rest {}",
            peak[i],
            w.soil_load
        );
        // The soil is plastic: the wheel rests at the sinkage of the peak load,
        // within 1.5 % of the static sinkage of its load at rest.
        let want = soil.wheel_sinkage(peak[i], b, r);
        let at_rest = soil.wheel_sinkage(w.soil_load, b, r);
        assert!(
            w.sinkage >= at_rest - 1e-5 && w.sinkage <= want + 1e-5,
            "wheel {i}: z {} outside [{at_rest}, {want}]",
            w.sinkage
        );
        assert!(
            (want - at_rest) < 0.015 * at_rest,
            "wheel {i}: {want} vs {at_rest}"
        );
        // The deepest sample under the wheel is within half a cell of its
        // bottom: z − (CELL/2)²/(2R) ≤ depth ≤ z.
        let cp = w.contact_point;
        let rut = deepest_near(&field, &depth, cp[0], cp[2], 1.5 * CELL);
        let sag = (0.5 * CELL).powi(2) / (2.0 * r);
        // The wheel sits in its rut: its bottom is at the rut's floor.
        let floor = w.position[1] - r;
        eprintln!(
            "wheel {i}: load {:.0} N, z {:.4} m (Bekker {want:.4}), rut {rut:.4} m, wheel bottom {floor:.4} m, R_c {:.0} N",
            w.soil_load, w.sinkage, w.compaction_resistance
        );
        assert!(
            rut <= want + 1e-4 && rut >= want - sag - 1e-4,
            "wheel {i}: rut {rut} vs z {want}"
        );
        assert!(
            (floor + want).abs() < 0.003,
            "wheel {i}: bottom at {floor}, rut floor at {}",
            -want
        );
        // At rest the soil ahead is fresh: the full first-pass resistance.
        let rc = soil.compaction_resistance(b, w.sinkage, 0.0);
        assert!(
            (w.compaction_resistance - rc).abs() < 1e-3 * rc,
            "wheel {i}: R_c {} vs {rc}",
            w.compaction_resistance
        );
    }
}

struct Pass {
    /// Forward speed after each step.
    speed: Vec<f32>,
    /// Wheel states after each step.
    wheels: Vec<VehicleState>,
}

fn coast(sim: &mut WgpuSim, v: VehicleHandle, speed: f32, steps: u32) -> Pass {
    let body = chassis(sim, v);
    sim.push_body_commands(&[GpuCommand::set_velocity(body, [0.0, 0.0, speed], [0.0; 3])])
        .unwrap();
    let mut out = Pass {
        speed: Vec::new(),
        wheels: Vec::new(),
    };
    for _ in 0..steps {
        sim.step().expect("step");
        let st = sim.vehicle_state(v);
        out.speed.push(st.forward_speed);
        out.wheels.push(st);
    }
    out
}

/// Steps of the measured window (after the launch transient).
const W0: usize = 30;
const W1: usize = 120;

fn decel(p: &Pass) -> f32 {
    (p.speed[W0] - p.speed[W1]) / ((W1 - W0) as f32 * DT)
}

/// Mean of Σ R_c over the steps whose resistance acts in the window (the
/// resistance a step reports is applied by the next).
fn mean_resistance(p: &Pass) -> f32 {
    let sum: f32 = (W0..W1)
        .map(|k| {
            p.wheels[k]
                .wheels
                .iter()
                .map(|w| w.compaction_resistance)
                .sum::<f32>()
        })
        .sum();
    sum / (W1 - W0) as f32
}

#[test]
fn ruts_follow_the_path_cost_compaction_and_a_second_pass_digs_no_deeper() {
    let soil = Soil::sandy_loam();
    let desc = car();
    let (x0, z0, v0, steps) = (0.0, -15.0, 6.0, 150u32);
    let m_eff = effective_mass(&desc);

    // The same field without soil.
    let (mut rigid, _, _) = world(121, 801, None);
    let (vr, _) = settle(&mut rigid, &desc, x0, z0, 2.0, 1.0);
    let base = coast(&mut rigid, vr, v0, steps);
    assert!(base.wheels[W1]
        .wheels
        .iter()
        .all(|w| w.sinkage == 0.0 && w.compaction_resistance == 0.0));

    let (mut sim, field, collider) = world(121, 801, Some(soil));
    let (v, _) = settle(&mut sim, &desc, x0, z0, 2.0, 1.0);
    let body = chassis(&sim, v);
    let start = sim.read_body(body);
    let first = coast(&mut sim, v, v0, steps);
    let after_first = depths(&sim, collider, &field);

    // Motion resistance: the extra deceleration is ΣR_c over the mass.
    let (a_soil, a_rigid) = (decel(&first), decel(&base));
    let rc = mean_resistance(&first);
    let want = rc / m_eff;
    eprintln!(
        "decel rigid {a_rigid:.3} m/s², soil {a_soil:.3} m/s²: extra {:.3} vs ΣR_c/m {want:.3} (ΣR_c {rc:.0} N)",
        a_soil - a_rigid
    );
    assert!(want > 0.5, "the soil resists: {want} m/s²");
    assert!(
        ((a_soil - a_rigid) - want).abs() < 0.1 * want,
        "extra deceleration {} vs {want}",
        a_soil - a_rigid
    );

    // Rolling at a steady load, the sinkage is the static one of that load
    // (Bekker's quasi-static wheel), here on average within 2 %. The front
    // wheels meet fresh soil: their resistance is the first-pass relation's for
    // their sinkage. The rear wheels run in the front wheels' ruts (same track)
    // and pay only for compacting them further.
    let mut off = 0.0f32;
    for k in W0..W1 {
        for (i, w) in first.wheels[k].wheels.iter().enumerate() {
            let zs = soil.wheel_sinkage(w.soil_load, w.width, w.radius);
            off += (w.sinkage - zs).abs() / zs;
            let r = soil.compaction_resistance(w.width, w.sinkage, 0.0);
            if i < 2 {
                assert!(
                    (w.compaction_resistance - r).abs() < 1e-3 * r,
                    "R_c {} vs {r}",
                    w.compaction_resistance
                );
            } else {
                assert!(
                    w.compaction_resistance <= r * (1.0 + 1e-3),
                    "rear R_c {} vs {r}",
                    w.compaction_resistance
                );
            }
        }
    }
    let off = off / (4 * (W1 - W0)) as f32;
    eprintln!(
        "rolling sinkage vs static sinkage of the load: {:.2} % on average",
        100.0 * off
    );
    assert!(
        off < 0.02,
        "rolling sinkage off the static one by {:.1} %",
        100.0 * off
    );

    // The rut follows each wheel's path at the sinkage of the load that passed.
    let mut worst = 0.0f32;
    for k in W0..W1 {
        for w in &first.wheels[k].wheels {
            let cp = w.contact_point;
            let rut = deepest_near(&field, &after_first, cp[0], cp[2], 0.75 * CELL);
            worst = worst.max((rut - w.sinkage).abs() / w.sinkage);
        }
    }
    eprintln!(
        "rut depth along the path: within {:.1} % of the sinkage",
        100.0 * worst
    );
    assert!(
        worst < 0.06,
        "rut depth off the sinkage by {:.1} %",
        100.0 * worst
    );

    // As wide as the tyre, to the grid: full depth inside, untouched beyond the
    // tyre's edge plus half a cell (and the heading's wobble).
    let (nx, _) = field.samples();
    let k = (W0 + W1) / 2;
    let w = &first.wheels[k].wheels[0];
    let (cx, cz) = (w.contact_point[0], w.contact_point[2]);
    for (i, d) in after_first.iter().enumerate() {
        let p = field.vertex(i as u32 % nx, i as u32 / nx);
        if (p.z - cz).abs() > 2.0 * CELL {
            continue;
        }
        let lat = (p.x - cx).abs();
        if lat < 0.5 * w.width - CELL {
            assert!(*d > 0.9 * w.sinkage, "inside the tyre at {lat}: {d}");
        } else if lat > 0.5 * w.width + 1.5 * CELL && lat < 0.6 {
            assert_eq!(*d, 0.0, "outside the tyre at {lat}: {d}");
        }
    }

    // The path again: the same car from the same start at the same speed.
    sim.push_body_commands(&[
        GpuCommand::teleport(body, start.position, start.orientation),
        GpuCommand::set_velocity(body, [0.0; 3], [0.0; 3]),
    ])
    .unwrap();
    let mut held = Vec::new();
    for _ in 0..60 {
        sim.step().expect("step");
        held.push(sim.vehicle_state(v));
    }
    let second = coast(&mut sim, v, v0, steps);
    let after_second = depths(&sim, collider, &field);
    // Over the first pass's ruts (from its launch to half a metre short of
    // where its front wheels were last; the second pass runs further, into
    // fresh soil), a sample is deeper only where a second-pass wheel that
    // crossed it sank further than the first pass had pressed it (its load
    // was higher: it decelerates less, so less load moves to the front), and
    // then exactly to that wheel's sinkage. Where the second pass's sinkage was
    // no deeper than the rut, the rut is unchanged. (The second pass includes
    // the second settle at the start.)
    // On the ruts' floors (deeper than 0.8 z: not the imprints' circular ends).
    let z_end = first.wheels.last().unwrap().wheels[0].contact_point[2] - 0.5;
    let z_begin = start.position[2] - 1.5;
    let z_ref = first.wheels[W0]
        .wheels
        .iter()
        .map(|w| w.sinkage)
        .fold(f32::INFINITY, f32::min);
    let (mut same, mut deepened, mut deepest) = (0usize, 0usize, 0.0f32);
    for (i, (b, a)) in after_second.iter().zip(&after_first).enumerate() {
        let p = field.vertex(i as u32 % nx, i as u32 / nx);
        if *a < 0.8 * z_ref || p.z <= z_begin || p.z >= z_end {
            continue;
        }
        let crossed = held
            .iter()
            .chain(&second.wheels)
            .flat_map(|s| s.wheels.iter())
            .filter(|w| {
                let cp = w.contact_point;
                (cp[0] - p.x).abs() < 0.5 * w.width + 2.0 * CELL && (cp[2] - p.z).abs() < 0.3
            })
            .map(|w| w.sinkage)
            .fold(0.0f32, f32::max);
        assert!(
            *b <= a.max(crossed) + 1e-4,
            "sample at {p}: {b} below the first rut {a} and the second sinkage {crossed}"
        );
        if crossed <= *a {
            assert_eq!(
                b, a,
                "sample at {p}: the second pass sank {crossed} in a rut {a} deep"
            );
            same += 1;
        } else if b > a {
            deepened += 1;
            deepest = deepest.max(b - a);
        }
    }
    let (a2, rc2) = (decel(&second), mean_resistance(&second));
    eprintln!(
        "second pass: {same} rut samples under no deeper sinkage unchanged; {deepened} under a higher load deeper (by up to {:.2} mm); ΣR_c {rc2:.0} N vs {rc:.0} N; decel {a2:.3} m/s²",
        1e3 * deepest
    );
    assert!(same > 100, "too few samples compared: {same}");
    assert!(
        rc2 < 0.15 * rc,
        "a compacted track costs little: {rc2} vs {rc}"
    );
    assert!(
        a2 - a_rigid < 0.25 * (a_soil - a_rigid),
        "the second pass rolls nearly freely: {a2}"
    );
}

#[test]
fn a_load_past_the_soil_limit_fails_the_step() {
    let soil = Soil::sandy_loam().with_max_sinkage(0.01);
    let (mut sim, _, _) = world(101, 101, Some(soil));
    let desc = car();
    sim.add_vehicle(&desc, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle");
    let mut failed = None;
    for _ in 0..240 {
        if let Err(e) = sim.step() {
            failed = Some(e);
            break;
        }
    }
    match failed {
        Some(engine_exec::ExecError::Overflow(flags)) => {
            assert_eq!(
                flags & OVERFLOW_SOIL_SINKAGE,
                OVERFLOW_SOIL_SINKAGE,
                "flags {flags:#x}"
            )
        }
        other => panic!("expected the soil limit to fail the step, got {other:?}"),
    }
}
