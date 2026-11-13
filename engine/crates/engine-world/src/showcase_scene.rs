//! The one-World showcase (`one-world.html`): short staged crashes in which 2 kHz
//! node–beam soft vehicles (the explicit solver) and prefractured, stress-solved rigid
//! destruction (the fracture lab's cuboid house as a threshold destructible) share one
//! World and act on each other both ways. One builder per preset; the page only draws.
//!
//! Presets ([`ShowcasePreset`]):
//! - `house`: a car at 120 km/h into the brick pier between a house's back windows.
//! - `crates`: a car at 50 km/h into a wall of loose crates.
//! - `twoway`: rigid on soft: a parked car under a falling concrete slab, then a
//!   wrecking ball on a rigid rod swung into its side.
//! - `convoy`: 2 to 4 cars side by side, each into the back wall of its own house.
//!
//! The house's frame: the cuboid house (`engine_fracture::cuboid`) is 8 m along x and
//! 6 m along z about its origin, its door in the +z wall, a porch on that side, a chimney
//! outside the −x wall (its picket fence is left out).

use crate::explicit_scene::place_vehicle;
use crate::{
    BodyDef, DestructibleAssetDesc, DestructibleDef, ExplicitVehiclesDef, Geometry, GpuContext,
    JointBase, JointDef, ShapeDef, SphericalJointDef, StrengthModel, Transform, World, WorldDef,
};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use glam::{Quat, Vec3};

/// The back wall's outer face (z, house frame): the convoy's cars come from behind the
/// houses, into the brick pier between the two back windows.
const BACK: f32 = -3.0;

/// The showcase's house: the fracture lab's cuboid house.
fn house_asset() -> anyhow::Result<DestructibleAssetDesc> {
    DestructibleAssetDesc::cuboid_house(&Default::default())
}
/// Lateral pitch of the convoy's houses (m): a house is 8.5 m wide with its chimney.
const CONVOY_PITCH: f32 = 13.0;

/// A staged crash of the showcase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShowcasePreset {
    House,
    Crates,
    TwoWay,
    Convoy,
}

impl ShowcasePreset {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "house" => Self::House,
            "crates" => Self::Crates,
            "twoway" => Self::TwoWay,
            "convoy" => Self::Convoy,
            _ => return None,
        })
    }
}

/// Where a vehicle starts: its front face's middle, its speed (m/s) along its heading,
/// `turns` quarter turns about +y from +z (0: +z, 1: +x, 2: −z, 3: −x).
#[derive(Clone, Copy, Debug)]
pub struct ShowcaseStart {
    pub front: [f32; 3],
    pub speed: f32,
    pub turns: u32,
}

/// A wrecking ball's pivot (static) and the ball's radius: the page draws its rod.
#[derive(Clone, Copy, Debug)]
pub struct ShowcaseRope {
    pub anchor: [f32; 3],
    pub body: u32,
    pub radius: f32,
}

/// What a page needs to frame a preset: the ground, the houses' origins, where the
/// vehicles start, the point of first impact and when (s, World time).
#[derive(Clone, Debug)]
pub struct ShowcaseLayout {
    pub preset: String,
    pub ground: f32,
    pub houses: Vec<[f32; 3]>,
    pub vehicles: Vec<ShowcaseStart>,
    pub impact: [f32; 3],
    pub impact_time: f32,
    pub ropes: Vec<ShowcaseRope>,
    /// Chunks in all the houses (threshold chunks at the start).
    pub chunks: usize,
    pub crates: usize,
}

/// The layout of `preset` with `count` vehicles (convoy: 2 to 4; the others: 1) at
/// `speed` m/s (0: the preset's own), `run_up` m from the first obstacle (0: its own).
pub fn showcase_layout(
    preset: ShowcasePreset,
    count: usize,
    speed: f32,
    run_up: f32,
) -> anyhow::Result<ShowcaseLayout> {
    let asset = house_asset()?;
    let stress = asset
        .stress
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("showcase: no stress source"))?;
    let ground = stress.ground_height().unwrap_or(0.0) - stress.placement()[1] as f32;
    let chunks_per_house = stress
        .threshold_scene(1.0, 1.0 / 60.0)?
        .bodies
        .iter()
        .map(|b| b.chunks.len())
        .sum::<usize>();
    let kmh = |k: f32| if speed > 0.0 { speed } else { k / 3.6 };
    let run = |d: f32| if run_up > 0.0 { run_up } else { d };
    let mut l = ShowcaseLayout {
        preset: format!("{preset:?}").to_lowercase(),
        ground,
        houses: Vec::new(),
        vehicles: Vec::new(),
        impact: [0.0, ground, 0.0],
        impact_time: 0.0,
        ropes: Vec::new(),
        chunks: 0,
        crates: 0,
    };
    match preset {
        ShowcasePreset::House => {
            // Into the back wall's pier between its windows, heading +z.
            l.houses.push([0.0, ground, 0.0]);
            let v = kmh(HOUSE_KMH);
            let d = run(30.0);
            l.vehicles.push(ShowcaseStart {
                front: [0.0, ground, BACK - d],
                speed: v,
                turns: 0,
            });
            l.impact = [0.0, ground + 1.0, BACK];
            l.impact_time = d / v;
            l.chunks = chunks_per_house;
        }
        ShowcasePreset::Convoy => {
            let n = count.clamp(2, 4);
            for k in 0..n {
                let x = (k as f32 - 0.5 * (n - 1) as f32) * CONVOY_PITCH;
                l.houses.push([x, ground, 0.0]);
                // A little faster each, so the strikes do not land on one frame.
                let v = kmh(HOUSE_KMH) * (1.0 + 0.04 * k as f32);
                let d = run(30.0) + 1.5 * k as f32;
                l.vehicles.push(ShowcaseStart {
                    front: [x, ground, BACK - d],
                    speed: v,
                    turns: 0,
                });
            }
            let v = l.vehicles[0];
            l.impact = [v.front[0], ground + 1.0, BACK];
            l.impact_time = (BACK - v.front[2]) / v.speed;
            l.chunks = chunks_per_house * l.houses.len();
        }
        ShowcasePreset::Crates => {
            let v = kmh(50.0);
            let d = run(16.0);
            l.vehicles.push(ShowcaseStart {
                front: [0.0, ground, -d],
                speed: v,
                turns: 0,
            });
            l.impact = [0.0, ground + 1.0, 0.0];
            l.impact_time = d / v;
            l.crates = CRATES[0] * CRATES[1] * CRATES[2];
        }
        ShowcasePreset::TwoWay => {
            // The car parked across the origin, nose to +z.
            l.vehicles.push(ShowcaseStart {
                front: [0.0, ground, 2.3],
                speed: 0.0,
                turns: 0,
            });
            l.impact = [0.0, ground + 1.4, 0.0];
            l.impact_time = SLAB_DROP_TIME;
            // The ball is the fourth body made (after the ground, the slab and the pivot).
            l.ropes.push(ShowcaseRope {
                anchor: [0.0, ground + BALL_LOW + ROD, BALL_Z],
                body: 3,
                radius: BALL_R,
            });
        }
    }
    Ok(l)
}

/// The house presets' speed (km/h) and their houses' settings ([`ShowcaseHouses::showcase`]).
pub const HOUSE_KMH: f32 = 120.0;

/// Crates: across, high, deep; edge (m) and density (kg/m³).
const CRATES: [usize; 3] = [6, 5, 1];
const CRATE: f32 = 0.6;
const CRATE_DENSITY: f32 = 220.0;
/// The two-way preset: the slab's half extents, density and drop (its underside above
/// the car's roof line), the wrecking ball's rod, radius, density, the height of its
/// centre at the bottom of the swing, the swing's start angle and the z of its plane.
const SLAB_HALF: [f32; 3] = [0.85, 0.22, 0.95];
const SLAB_DENSITY: f32 = 2400.0;
const SLAB_DROP: f32 = 5.5;
const ROOF: f32 = 1.45;
const SLAB_DROP_TIME: f32 = 1.06;
const ROD: f32 = 8.0;
const BALL_R: f32 = 0.6;
const BALL_DENSITY: f32 = 5000.0;
const BALL_LOW: f32 = 1.15;
const BALL_START_DEG: f32 = 78.0;
const BALL_Z: f32 = -0.6;

/// The showcase World and its layout.
pub struct Showcase {
    pub world: World,
    pub layout: ShowcaseLayout,
    pub vehicles: Vec<ExplicitBeams>,
}

/// The houses' settings: `strength_scale` multiplies every bond capacity
/// (`DestructibleDef::strength_scale`), `support_loss` lets a structure whose supports
/// broke come down (`DestructibleDef::support_loss`).
#[derive(Clone, Copy, Debug)]
pub struct ShowcaseHouses {
    pub strength_scale: f32,
    pub support_loss: bool,
}

impl Default for ShowcaseHouses {
    fn default() -> Self {
        Self {
            strength_scale: 1.0,
            support_loss: false,
        }
    }
}

impl ShowcaseHouses {
    /// The showcase's houses, stated: an old house's lime mortar (a quarter of the cuboid
    /// house's bond capacities: its brick joints 75 kPa in tension instead of 0.3 MPa;
    /// lime mortars test at 0.05 to 0.2 MPa), and support loss on with the sudden-loss
    /// factor 2 (the undamped step response), so what loses its support comes down. At
    /// full strength a car at 90 km/h stops in the outer wall.
    pub fn showcase() -> Self {
        Self {
            strength_scale: 0.25,
            support_loss: true,
        }
    }
}

/// A showcase vehicle from a BeamNG JBeam vehicle: its `.jbeam` files (name, text) and
/// a `.pc` configuration, spawned in JBeam mode (the BeamNG profile, its pressure
/// wheels) and turned so its forward axis (`engine_softbody::pack::forward_axis`) is
/// +Z, as [`showcase`] places its vehicles. Returns it and the loader's warnings.
pub fn jbeam_vehicle(
    files: &[(&str, &str)],
    pc: Option<&str>,
    settings: ExplicitBeamsSettings,
) -> anyhow::Result<(ExplicitBeams, Vec<String>)> {
    jbeam_vehicle_on(files, pc, settings, JbeamWheels::Pressure)
}

/// [`jbeam_vehicle`] on `wheels`: [`JbeamWheels::Pressure`] (its own pressure wheels) or
/// [`JbeamWheels::Raycast`] (drivable: `engine_softbody::explicit_drive`'s raycast wheels
/// on its hubs, `ExplicitBeams::from_desc_jbeam_raycast`).
pub fn jbeam_vehicle_on(
    files: &[(&str, &str)],
    pc: Option<&str>,
    settings: ExplicitBeamsSettings,
    wheels: JbeamWheels,
) -> anyhow::Result<(ExplicitBeams, Vec<String>)> {
    let desc = engine_softbody::jbeam::load_jbeam(files, pc).map_err(anyhow::Error::msg)?;
    let (mut sim, warnings) = match wheels {
        JbeamWheels::Raycast => {
            ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings)
        }
        w => ExplicitBeams::from_desc_jbeam_with(&desc, w, Vec3::ZERO, Vec3::ZERO, settings),
    }
    .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(!sim.nodes.is_empty(), "the vehicle has no nodes");
    let f = engine_softbody::pack::forward_axis(&desc);
    let angle = -f.x.atan2(f.z);
    let quarter = angle / std::f32::consts::FRAC_PI_2;
    let k = quarter.round();
    // A quarter turn exactly when it is one (no rounding of the rest pose).
    let rot = |p: Vec3| -> Vec3 {
        if (quarter - k).abs() > 1e-6 {
            return Quat::from_rotation_y(angle) * p;
        }
        match (k as i32).rem_euclid(4) {
            0 => p,
            1 => Vec3::new(p.z, p.y, -p.x),
            2 => Vec3::new(-p.x, p.y, -p.z),
            _ => Vec3::new(-p.z, p.y, p.x),
        }
    };
    for n in &mut sim.nodes {
        n.pos = rot(n.pos);
        n.vel = rot(n.vel);
    }
    Ok((sim, warnings))
}

/// Builds `preset` on `gpu` ([`showcase_layout`] for the arguments): vehicle `k` is a
/// copy of `templates[k % len]`, each facing +Z ([`jbeam_vehicle`]; a `.truck`
/// vehicle's longer horizontal axis is its length). The two-way preset's parked car has
/// its parking brake on (pressure wheels with `parkingTorque`). `def`: the vehicles'
/// coupling ([`crate::explicit_scene::town_coupling`]).
#[allow(clippy::too_many_arguments)]
pub fn showcase(
    gpu: GpuContext,
    templates: &[ExplicitBeams],
    preset: ShowcasePreset,
    count: usize,
    speed: f32,
    run_up: f32,
    houses: ShowcaseHouses,
    def: &ExplicitVehiclesDef,
) -> anyhow::Result<Showcase> {
    anyhow::ensure!(!templates.is_empty(), "showcase: no vehicle");
    let mut layout = showcase_layout(preset, count, speed, run_up)?;
    let asset = house_asset()?;
    let stress = asset
        .stress
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("showcase: no stress source"))?;
    let offset = stress.placement().map(|x| -x as f32);
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("showcase: no threshold solver"))?;
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    // Capacity as `crash_town`'s: per house three times its live slots (a split's
    // children take new slots until compaction reclaims the retired ones), plus the
    // loose bodies, at least one house's worth so every preset shares the sizing.
    let h = layout.houses.len().max(1) as u32;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let parts: usize = scene
        .bodies
        .iter()
        .flat_map(|b| &b.chunks)
        .map(|c| c.compound.as_ref().map_or(1, |p| p.len().max(1)))
        .sum();
    let mut cap = stress_rubble::capacity_for(&scene);
    let extra = 3 * (layout.crates as u32 + 64);
    cap.max_bodies = (3 * (chunks as u32 + 2) * h + extra).next_power_of_two();
    cap.max_colliders = (3 * (parts as u32 + 2) * h + extra).next_power_of_two();
    cap.max_hull_vertices *= h;
    cap.max_contacts =
        stress_rubble::threshold::contact_capacity(chunks * h as usize, cap.max_contacts, 1 << 30);
    cap.max_candidates = cap.max_candidates.max(cap.max_contacts);
    cap.max_joints = cap.max_joints.max(8);
    let ground = layout.ground;
    let mut world = World::new(
        gpu,
        &WorldDef {
            capacity: cap,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: crate::ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )?;
    let g = world.create_body(&BodyDef::default())?;
    world.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: ground,
        },
    )?;
    if !layout.houses.is_empty() {
        let a = world.create_destructible_asset(&asset)?;
        for p in &layout.houses {
            world.create_destructible(&DestructibleDef {
                model: StrengthModel::Threshold,
                transform: Transform::new(
                    [offset[0] + p[0], offset[1], offset[2] + p[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ),
                asleep: false,
                strength_scale: houses.strength_scale,
                support_loss: houses.support_loss,
                support_loss_dynamic_factor: if houses.support_loss { 2.0 } else { 1.0 },
                ..DestructibleDef::new(a)
            })?;
        }
    }
    match preset {
        ShowcasePreset::Crates => {
            let [nx, ny, nz] = CRATES;
            let pitch = CRATE + 0.01;
            for k in 0..nz {
                for j in 0..ny {
                    // Running bond: odd courses offset by half a crate.
                    let shift = if j % 2 == 1 { 0.5 * pitch } else { 0.0 };
                    for i in 0..nx {
                        let x = (i as f32 - 0.5 * (nx - 1) as f32) * pitch + shift - 0.25 * pitch;
                        let at = [
                            x,
                            ground + 0.5 * CRATE + 0.002 + j as f32 * (CRATE + 0.002),
                            0.5 * CRATE + k as f32 * pitch,
                        ];
                        let b = world.create_body(&BodyDef::dynamic(at))?;
                        world.create_box_shape(
                            b,
                            &ShapeDef {
                                density: CRATE_DENSITY,
                                ..ShapeDef::default()
                            },
                            [0.5 * CRATE; 3],
                        )?;
                    }
                }
            }
        }
        ShowcasePreset::TwoWay => {
            let roof = ground + ROOF;
            let slab = world.create_body(&BodyDef::dynamic([
                0.0,
                roof + SLAB_DROP + SLAB_HALF[1],
                0.35,
            ]))?;
            world.create_box_shape(
                slab,
                &ShapeDef {
                    density: SLAB_DENSITY,
                    ..ShapeDef::default()
                },
                SLAB_HALF,
            )?;
            layout.impact_time = (2.0 * SLAB_DROP / -solver.gravity_y).sqrt();
            let rope = &mut layout.ropes[0];
            let pivot = world.create_body(&BodyDef {
                position: rope.anchor,
                ..BodyDef::default()
            })?;
            let a = BALL_START_DEG.to_radians();
            let off = [-ROD * a.sin(), -ROD * a.cos(), 0.0];
            let ball = world.create_body(&BodyDef::dynamic([
                rope.anchor[0] + off[0],
                rope.anchor[1] + off[1],
                rope.anchor[2] + off[2],
            ]))?;
            world.create_sphere_shape(
                ball,
                &ShapeDef {
                    density: BALL_DENSITY,
                    ..ShapeDef::default()
                },
                [0.0; 3],
                BALL_R,
            )?;
            world.create_joint(&JointDef::Spherical(SphericalJointDef {
                base: JointBase::at(pivot, [0.0; 3], ball, off.map(|x| -x)),
                cone_angle: None,
                twist_limit: None,
            }))?;
            anyhow::ensure!(
                world.body_pose_index(ball)? == rope.body,
                "showcase: the ball's body slot moved"
            );
        }
        _ => {}
    }
    let vehicles: Vec<ExplicitBeams> = layout
        .vehicles
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let template = &templates[k % templates.len()];
            let mut sim = place_vehicle(template, Vec3::from(v.front), v.turns, v.speed, ground);
            if preset == ShowcasePreset::TwoWay {
                sim.parking_brake = 1.0;
            }
            sim
        })
        .collect();
    world.add_explicit_vehicles(&vehicles, def)?;
    Ok(Showcase {
        world,
        layout,
        vehicles,
    })
}
