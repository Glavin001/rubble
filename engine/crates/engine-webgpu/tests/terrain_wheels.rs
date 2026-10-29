//! Two-way wheel coupling with the terrain MPM (`engine/docs/deformable-terrain.md`,
//! "Wheels on open sand"): a vehicle's wheels ride an open area's live
//! surface (its patch), and the sand carries the force they press there.
//! Checked against what the coupling must do:
//! - a car set on open sand sinks into it and comes to rest: the sinkage stops
//!   growing, the wheels' normal forces carry the car's weight, and the sand
//!   under it is at rest (its momentum a small fraction of the weight's
//!   impulse over a frame), so the sand's stress holds the load;
//! - a car driving through an open area leaves a rut along its tracks, and
//!   the volume the rut displaces is all still there after the bake (the
//!   heightfield's volume is unchanged to the quanta's rounding);
//! - wheels outside every open area press nothing (no load listed), and a
//!   terrain with `max_wheel_loads` 0 lists none.

use engine_schema::commands::GpuCommand;
use engine_schema::convex::{convex_params_base, HullArena};
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, Material, WorldCapacity};
use engine_terrain::layout::MpmParticle;
use engine_terrain::surface::{SurfacePatch, SurfacePlan, DEFAULT_SURFACE_CELLS};
use engine_terrain::{substeps_for, RegionPlan, SandMaterial, TerrainConfig, TerrainProfile};
use engine_vehicle::{presets, DriveDesc, VehicleDesc, VehicleHandle, VehicleInput};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;
const G: f32 = 9.81;
/// The MPM cell and the field's sample spacing (m).
const DX: f32 = 0.15;
/// Field samples along x and z (centred on the origin).
const NX: u32 = 55;
const NZ: u32 = 121;
/// Samples of the surface patch along x and z.
const PN: u32 = 48;

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
        dt: FRAME,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = -gravity;
    p.sleep_time = 1.0e9;
    p
}

/// The sedan coasting: no drivetrain, no aero drag.
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

struct Scene {
    s: WgpuSim,
    cfg: TerrainConfig,
    sand: SandMaterial,
    bed: u32,
    surface: SurfacePlan,
}

impl Scene {
    /// A flat sandy field at y = 0, an area open over x ∈ [-2, 2], z ∈ [-3, 3]
    /// down to 1.2 m, and its surface patch; the sand settled for a few frames.
    fn new(max_wheel_loads: u32) -> Self {
        let mut s = WgpuSim::new(capacity()).expect("wgpu adapter");
        s.set_params(params(G));
        s.set_material(
            0,
            Material {
                friction: 1.0,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
        let cfg = TerrainConfig {
            max_particles: 1 << 18,
            max_blocks: 1 << 12,
            max_wheel_loads,
            ..TerrainConfig::default()
        }
        .with_profile(TerrainProfile::Game);
        s.terrain_enable(&cfg).unwrap();
        let sand = SandMaterial::default();
        s.terrain_set_material(Some(&sand)).unwrap();
        let field = HeightField::new(NX, NZ, DX, DX, vec![0.0; (NX * NZ) as usize]).unwrap();
        let mut arena = HullArena::new();
        let (_, bed) = s
            .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
        let sunk = HeightField::new(PN, PN, DX, DX, vec![-1.0e4; (PN * PN) as usize]).unwrap();
        let (_, patch) = s
            .add_height_field(&mut arena, &sunk, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
        let surface = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS)
            .unwrap()
            .with_patches(vec![SurfacePatch {
                collider: patch,
                nx: PN,
                nz: PN,
            }])
            .unwrap();
        s.terrain_surface_enable(&surface).unwrap();
        let plan = RegionPlan::spanning(
            &cfg,
            Vec3::new(-2.0, -1.2, -3.0),
            Vec3::new(2.0, 0.6, 3.0),
            DX,
            0,
            0,
        )
        .unwrap();
        let opened = s
            .terrain_open_area(
                &plan,
                bed,
                &field,
                Vec3::ZERO,
                sand.density,
                sand.friction_angle.tan(),
            )
            .unwrap();
        assert!(opened.activated);
        let mut sc = Self {
            s,
            cfg,
            sand,
            bed,
            surface,
        };
        sc.frames(10, |_| {});
        sc
    }

    /// `n` coupled frames, each followed by the surface pass (the wheels'
    /// patch); `each` sees the frame's report.
    fn frames(&mut self, n: u32, mut each: impl FnMut(&engine_terrain::TerrainStepReport)) {
        for _ in 0..n {
            let (k, dt) =
                substeps_for(&self.cfg, FRAME, DX, self.sand.p_wave_speed(), 5.0).unwrap();
            let (_, r) = self.s.step_with_terrain(k, dt).unwrap();
            let surface = self.surface.clone();
            self.s.terrain_surface(&surface).unwrap();
            each(&r);
        }
    }

    /// The car with its wheels just touching the ground at (`x`, `z`), at rest.
    fn place(&mut self, desc: &VehicleDesc, x: f32, z: f32) -> VehicleHandle {
        let v = self
            .s
            .add_vehicle(desc, [x, 1.5, z], [0.0, 0.0, 0.0, 1.0])
            .expect("vehicle");
        self.frames(1, |_| {});
        let st = self.s.vehicle_state(v);
        let bottom = st
            .wheels
            .iter()
            .map(|w| w.position[1] - w.radius)
            .fold(f32::INFINITY, f32::min);
        let body = self.s.vehicle_host().chassis_body(v);
        let b = self.s.read_body(body);
        let mut p = b.position;
        p[1] -= bottom - 0.002;
        self.s
            .push_body_commands(&[
                GpuCommand::teleport(body, p, b.orientation),
                GpuCommand::set_velocity(body, [0.0; 3], [0.0; 3]),
            ])
            .unwrap();
        v
    }

    fn heights(&self) -> Vec<f32> {
        let base = convex_params_base(self.s.collider(self.bed).params);
        self.s
            .read_words(18, u64::from(base + 2) * 4, (NX * NZ) as usize)
            .into_iter()
            .map(f32::from_bits)
            .collect()
    }
}

fn sand_momentum(ps: &[MpmParticle]) -> DVec3 {
    ps.iter()
        .map(|p| Vec3::from_slice(&p.vj[..3]).as_dvec3() * f64::from(p.xm[3]))
        .sum()
}

/// The field sample nearest (x, z).
fn sample(x: f32, z: f32) -> usize {
    let i = (x / DX + (NX - 1) as f32 / 2.0).round() as usize;
    let j = (z / DX + (NZ - 1) as f32 / 2.0).round() as usize;
    j * NX as usize + i
}

#[test]
fn a_car_on_open_sand_sinks_to_rest_and_the_sand_carries_it() {
    let mut sc = Scene::new(64);
    let desc = car();
    let v = sc.place(&desc, 0.0, 0.0);
    let weight = mass_of(&desc) * G;
    // Mean contact depth below the sand's surface per frame, the wheels
    // listed, and the sand's momentum each second.
    let mut depth = Vec::new();
    let mut listed = Vec::new();
    let mut momentum = Vec::new();
    for f in 0..600 {
        let mut wheels = 0;
        sc.frames(1, |r| wheels = r.wheels);
        listed.push(wheels);
        let st = sc.s.vehicle_state(v);
        let mean =
            st.wheels.iter().map(|w| w.contact_point[1]).sum::<f32>() / st.wheels.len() as f32;
        depth.push(-mean);
        if f % 60 == 59 {
            momentum.push(sand_momentum(&sc.s.terrain_particles()));
            eprintln!("t {:.0} s: sinkage {:.4} m", (f + 1) as f32 * FRAME, -mean);
        }
    }
    let st = sc.s.vehicle_state(v);
    let carried: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    // Sinkage rates (m/s): over the first second, and over the last.
    let rate = |a: usize, b: usize| (depth[b] - depth[a]) / ((b - a) as f32 * FRAME);
    let (first, last) = (rate(0, 59), rate(539, 599));
    let n = momentum.len();
    let dp = (momentum[n - 1] - momentum[n - 2]).length();
    eprintln!(
        "sinkage {:.4} m at 1 s, {:.4} m at 10 s; rate {first:.4} -> {last:.5} m/s; wheels listed {:?}; \
         wheels carry {carried:.0} N of {weight:.0} N; sand momentum change over the last second \
         {dp:.2} N·s (the weight's impulse {:.0} N·s)",
        depth[59],
        depth[599],
        &listed[listed.len() - 3..],
        weight,
    );
    assert!(
        listed.iter().skip(5).all(|&n| n == 4),
        "all four wheels on the sand: {listed:?}"
    );
    assert!(
        depth[599] > 0.01,
        "the wheels sink into the sand: {} m",
        depth[599]
    );
    assert!(
        last < 0.1 * first,
        "the sinkage comes to rest: {first} m/s, then {last} m/s"
    );
    assert!(
        (carried - weight).abs() < 0.03 * weight,
        "the wheels carry the car: {carried} N vs {weight} N"
    );
    // Were the load not carried by the sand's stress, its momentum would
    // grow by the weight's impulse each second.
    assert!(
        dp < 0.01 * f64::from(weight),
        "the sand carries the load: its momentum moved {dp} N·s in a second"
    );
}
#[test]
fn a_car_driving_through_open_sand_leaves_a_rut_and_the_volume_stays() {
    let mut sc = Scene::new(64);
    // The field the area was opened from: flat at y = 0.
    let before = vec![0.0f32; (NX * NZ) as usize];
    let throttle: f32 = std::env::var("WHEEL_THROTTLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.3);
    let volume = |h: &[f32]| h.iter().map(|&y| f64::from(y)).sum::<f64>() * f64::from(DX * DX);
    let desc = presets::sedan();
    let v = sc.place(&desc, 0.0, -5.5);
    let body = sc.s.vehicle_host().chassis_body(v);
    sc.frames(30, |_| {});
    sc.s.set_vehicle_input(
        v,
        VehicleInput {
            forward: throttle,
            ..VehicleInput::default()
        },
    );
    let mut most = 0;
    let mut tracks: Vec<[f32; 2]> = Vec::new();
    for _ in 0..900 {
        sc.frames(1, |r| most = most.max(r.wheels));
        let st = sc.s.vehicle_state(v);
        for w in &st.wheels {
            if w.contact_point[2].abs() < 2.4 && w.in_contact {
                tracks.push([w.contact_point[0], w.contact_point[2]]);
            }
        }
        if st.wheels.iter().all(|w| w.contact_point[2] > 3.6) {
            break;
        }
    }
    let z = sc.s.read_body(body).position[2];
    assert!(z > 3.6, "the car crossed the area: z {z}");
    assert_eq!(most, 4, "all four wheels pressed the sand at once");
    // Let the sand come to rest, then bake.
    sc.frames(60, |_| {});
    sc.s.terrain_bake().unwrap();
    let after = sc.heights();
    let (v0, v1) = (volume(&before), volume(&after));
    // Under the tracks the ground is lower than it was.
    let rut: Vec<f32> = tracks.iter().map(|t| after[sample(t[0], t[1])]).collect();
    let mean_rut = rut.iter().sum::<f32>() / rut.len() as f32;
    let deepest = rut.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = after.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lowest = after.iter().copied().fold(f32::INFINITY, f32::min);
    // The window's samples well clear of the tracks: the bake's own noise.
    let mut clear = Vec::new();
    for j in 0..NZ {
        for i in 0..NX {
            let (x, z) = (
                (i as f32 - (NX - 1) as f32 / 2.0) * DX,
                (j as f32 - (NZ - 1) as f32 / 2.0) * DX,
            );
            if x.abs() < 2.2 && z.abs() < 2.4 && tracks.iter().all(|t| (t[0] - x).abs() > 0.45) {
                clear.push(after[(j * NX + i) as usize]);
            }
        }
    }
    let mean_clear = clear.iter().sum::<f32>() / clear.len() as f32;
    eprintln!(
        "at throttle {throttle}: rut under the tracks mean {mean_rut:.4} m, deepest {deepest:.4} m; clear of \
         them {mean_clear:.4} m; field {lowest:.4}..{highest:.4} m; volume {v0:.5} -> {v1:.5} m³"
    );
    assert!(
        mean_rut < mean_clear - 0.002,
        "a rut along the tracks: {mean_rut} m (clear of them {mean_clear} m)"
    );
    assert!(
        highest > 0.003,
        "the displaced sand stands beside it: {highest} m"
    );
    assert!(
        (v1 - v0).abs() < 1.0e-4 * (6.0 * 4.0 * 1.2),
        "the rut's volume is all still there: {v0} -> {v1} m³"
    );
}

#[test]
fn wheels_outside_every_open_area_press_nothing() {
    let mut sc = Scene::new(64);
    let desc = car();
    // Parked on the unactivated ground beside the area.
    let v = sc.place(&desc, 0.0, -6.0);
    let mut listed = Vec::new();
    sc.frames(60, |r| listed.push(r.wheels));
    assert!(sc.s.vehicle_state(v).wheels.iter().all(|w| w.in_contact));
    assert!(listed.iter().all(|&n| n == 0), "no load listed: {listed:?}");

    // With no wheel loads configured, a car on the sand lists none either.
    let mut sc = Scene::new(0);
    sc.place(&desc, 0.0, 0.0);
    let mut listed = Vec::new();
    sc.frames(30, |r| listed.push(r.wheels));
    assert!(
        listed.iter().all(|&n| n == 0),
        "max_wheel_loads 0: {listed:?}"
    );
}

/// The wheel loads' GPU cost: the car resting on the open sand, the terrain's
/// GPU time per frame with the wheels pressing (four loads listed) and with
/// `max_wheel_loads` 0 (no wheel kernels). Run alone, release:
///
/// `RUBBLE_TERRAIN_GPU_MS=1 cargo test --release -p engine-webgpu --test
/// terrain_wheels -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run alone, release, with RUBBLE_TERRAIN_GPU_MS=1"]
fn wheel_load_cost() {
    for loads in [64, 0, 64, 0] {
        let mut sc = Scene::new(loads);
        sc.place(&car(), 0.0, 0.0);
        sc.frames(120, |_| {});
        let mut ms = Vec::new();
        for _ in 0..120 {
            sc.frames(1, |_| {});
            ms.extend(sc.s.terrain_last_gpu_ms());
        }
        ms.sort_by(f64::total_cmp);
        let (k, _) = substeps_for(&sc.cfg, FRAME, DX, sc.sand.p_wave_speed(), 5.0).unwrap();
        eprintln!(
            "max_wheel_loads {loads}: terrain GPU per frame median {:.3} ms, p95 {:.3} ms, worst {:.3} ms \
             ({k} substeps of a 1/60 s frame; {} particles)",
            ms[ms.len() / 2],
            ms[ms.len() * 95 / 100],
            ms[ms.len() - 1],
            sc.s.terrain_particles().len(),
        );
    }
}
