//! Ready-to-drive vehicles. Wheel order is front-left, front-right, then rear
//! pairs (left before right). Chassis frame: +Z forward, +Y up, +X left.

use crate::desc::*;

/// Wheel at (x, y, z) with shared tuning.
fn wheel(x: f32, y: f32, z: f32, base: &WheelDesc) -> WheelDesc {
    WheelDesc { position: [x, y, z], ..base.clone() }
}

fn box_part(name: &str, role: PartRole, mass: f32, half: [f32; 3], pos: [f32; 3]) -> PartDesc {
    PartDesc {
        name: name.into(),
        role,
        mass,
        colliders: vec![PartColliderDesc { shape: ShapeDesc::Box { half }, position: pos, ..Default::default() }],
        mount: MountDesc::default(),
    }
}

fn four_wheels(half_track: f32, y: f32, front_z: f32, rear_z: f32, front: &WheelDesc, rear: &WheelDesc) -> Vec<WheelDesc> {
    vec![
        wheel(half_track, y, front_z, front),
        wheel(-half_track, y, front_z, front),
        wheel(half_track, y, rear_z, rear),
        wheel(-half_track, y, rear_z, rear),
    ]
}

/// Jolt `VehicleConstraintTest` defaults (1500 kg, 4 wheels, FWD, auto box).
pub fn jolt_default() -> VehicleDesc {
    let front = WheelDesc { steerable: true, driven: true, radius: 0.3, width: 0.1, ..Default::default() };
    let rear = WheelDesc { handbrake: true, max_steer: 0.0, radius: 0.3, width: 0.1, ..Default::default() };
    VehicleDesc {
        name: "jolt_default".into(),
        parts: vec![box_part("body", PartRole::Body, 1420.0, [0.75, 0.2, 2.0], [0.0, 0.0, 0.0])],
        wheels: four_wheels(0.9, -0.1, 1.5, -1.5, &front, &rear),
        drive: DriveDesc::Drivetrain(DrivetrainDesc {
            differentials: vec![DifferentialDesc { left: Some(0), right: Some(1), ..Default::default() }],
            ..Default::default()
        }),
        anti_roll: vec![
            AntiRollDesc { left: 0, right: 1, stiffness: 1000.0 },
            AntiRollDesc { left: 2, right: 3, stiffness: 1000.0 },
        ],
        ..Default::default()
    }
}

/// Road car: rear-wheel drive, limited-slip rear, Ackermann, light aero.
pub fn sedan() -> VehicleDesc {
    let base = WheelDesc {
        radius: 0.34,
        width: 0.22,
        mass: 18.0,
        inertia: Some(1.0),
        suspension_min: 0.06,
        suspension_max: 0.28,
        preload: 0.04,
        spring: Spring::FrequencyDamping { hz: 1.7, zeta: 0.6 },
        max_steer: 35f32.to_radians(),
        max_brake_torque: 2200.0,
        max_handbrake_torque: 4000.0,
        rolling_resistance: 0.012,
        long_mu_scale: 1.05,
        lat_mu_scale: 1.0,
        rim_material: crate::pack::WHEEL_RIM_MATERIAL,
        ..Default::default()
    };
    let front = WheelDesc { steerable: true, ..base.clone() };
    let rear = WheelDesc { driven: true, handbrake: true, max_steer: 0.0, ..base };
    VehicleDesc {
        name: "sedan".into(),
        // Engine, floor pan and fuel tank sit low: the body boxes overstate it.
        center_of_mass_offset: [0.0, -0.18, 0.0],
        parts: vec![
            box_part("body", PartRole::Body, 950.0, [0.76, 0.22, 2.15], [0.0, 0.05, 0.0]),
            box_part("cabin", PartRole::Body, 300.0, [0.7, 0.24, 1.05], [0.0, 0.5, -0.25]),
        ],
        wheels: four_wheels(0.82, 0.02, 1.35, -1.35, &front, &rear),
        drive: DriveDesc::Drivetrain(DrivetrainDesc {
            engine: EngineDesc { max_torque: 420.0, max_rpm: 6800.0, ..Default::default() },
            transmission: TransmissionDesc {
                gear_ratios: vec![3.4, 2.2, 1.55, 1.18, 0.95, 0.78],
                reverse_ratios: vec![-3.2],
                shift_up_rpm: 5800.0,
                shift_down_rpm: 2400.0,
                switch_time: 0.3,
                clutch_release_time: 0.2,
                ..Default::default()
            },
            differentials: vec![DifferentialDesc { left: Some(2), right: Some(3), ratio: 3.9, ..Default::default() }],
            ..Default::default()
        }),
        anti_roll: vec![
            AntiRollDesc { left: 0, right: 1, stiffness: 9000.0 },
            AntiRollDesc { left: 2, right: 3, stiffness: 6000.0 },
        ],
        aero: AeroDesc { drag: 0.4, downforce: 0.3, center: [0.0, 0.1, 0.0] },
        steering: SteeringDesc { speed_curve: Curve::road_steering(), ..Default::default() },
        assists: AssistDesc { traction_control: true, ..Default::default() },
        ..Default::default()
    }
}

/// The sedan built from separable parts: bodywork, hood, engine, rear driveshaft,
/// steering rack. Sever any of them through the parts API.
pub fn destructible_sedan() -> VehicleDesc {
    let mut d = sedan();
    d.name = "destructible_sedan".into();
    // The separable engine (250 kg at the nose) and bumpers put the part-derived centre
    // of mass ~0.25 m ahead of the sedan's, unloading the driven rear: shift it back so
    // both cars balance alike.
    d.center_of_mass_offset = [0.0, -0.18, -0.25];
    d.parts = vec![
        box_part("frame", PartRole::Body, 560.0, [0.76, 0.15, 2.0], [0.0, 0.0, 0.0]),
        box_part("cabin", PartRole::Body, 300.0, [0.7, 0.24, 1.05], [0.0, 0.42, -0.25]),
        box_part("hood", PartRole::Body, 60.0, [0.72, 0.06, 0.55], [0.0, 0.21, 1.42]),
        box_part("engine", PartRole::Engine, 250.0, [0.4, 0.2, 0.4], [0.0, 0.0, 1.4]),
        box_part("trunk_lid", PartRole::Body, 40.0, [0.72, 0.06, 0.42], [0.0, 0.21, -1.55]),
        box_part("bumper_front", PartRole::Body, 35.0, [0.78, 0.1, 0.1], [0.0, -0.02, 2.12]),
        box_part("bumper_rear", PartRole::Body, 35.0, [0.78, 0.1, 0.1], [0.0, -0.02, -2.12]),
        box_part("door_left", PartRole::Body, 25.0, [0.04, 0.2, 0.55], [0.74, 0.32, -0.15]),
        box_part("door_right", PartRole::Body, 25.0, [0.04, 0.2, 0.55], [-0.74, 0.32, -0.15]),
        box_part("driveshaft", PartRole::DriveShaft { wheels: 0b1100 }, 40.0, [0.07, 0.07, 0.8], [0.0, -0.15, -0.6]),
        box_part("steering_rack", PartRole::Steering { wheels: 0b0011 }, 20.0, [0.6, 0.05, 0.05], [0.0, -0.1, 1.2]),
        box_part("brake_lines", PartRole::Brake { wheels: 0b1111 }, 5.0, [0.05, 0.05, 1.0], [0.0, -0.2, 0.2]),
    ];
    d
}

/// Light rear-drive sports car with downforce and quick steering.
pub fn sports() -> VehicleDesc {
    let mut d = sedan();
    d.name = "sports".into();
    d.parts = vec![
        box_part("body", PartRole::Body, 900.0, [0.8, 0.2, 2.2], [0.0, 0.02, 0.0]),
        box_part("cabin", PartRole::Body, 150.0, [0.62, 0.17, 0.8], [0.0, 0.37, -0.3]),
    ];
    for w in &mut d.wheels {
        w.radius = 0.33;
        w.width = 0.28;
        w.suspension_min = 0.05;
        w.suspension_max = 0.22;
        w.spring = Spring::FrequencyDamping { hz: 2.2, zeta: 0.7 };
        w.long_mu_scale = 1.35;
        w.lat_mu_scale = 1.35;
    }
    if let DriveDesc::Drivetrain(dt) = &mut d.drive {
        dt.engine.max_torque = 650.0;
        dt.engine.max_rpm = 7800.0;
        dt.transmission.gear_ratios = vec![3.1, 2.1, 1.55, 1.2, 1.0, 0.82];
        dt.transmission.shift_up_rpm = 7000.0;
        dt.transmission.switch_time = 0.2;
        dt.transmission.clutch_release_time = 0.15;
    }
    d.aero = AeroDesc { drag: 0.35, downforce: 1.6, center: [0.0, 0.0, -0.2] };
    d.steering.rate = 6.0;
    // Sports car: no traction control, so throttle can still steer the tail.
    d.assists.traction_control = false;
    d
}

/// Four-wheel-drive monster truck: huge travel, soft springs, centre LSD.
pub fn monster_truck() -> VehicleDesc {
    let base = WheelDesc {
        radius: 0.8,
        width: 0.55,
        mass: 90.0,
        inertia: Some(25.0),
        suspension_min: 0.3,
        suspension_max: 1.2,
        preload: 0.0,
        spring: Spring::FrequencyDamping { hz: 1.1, zeta: 0.45 },
        max_steer: 32f32.to_radians(),
        max_brake_torque: 4500.0,
        max_handbrake_torque: 8000.0,
        driven: true,
        long_mu_scale: 1.2,
        lat_mu_scale: 1.0,
        rolling_resistance: 0.02,
        rim_material: crate::pack::WHEEL_RIM_MATERIAL,
        ..Default::default()
    };
    let front = WheelDesc { steerable: true, ..base.clone() };
    let rear = WheelDesc { handbrake: true, max_steer: 0.0, ..base };
    VehicleDesc {
        name: "monster_truck".into(),
        parts: vec![box_part("body", PartRole::Body, 2600.0, [1.1, 0.45, 2.3], [0.0, 0.0, 0.0])],
        wheels: four_wheels(1.45, -0.1, 1.75, -1.75, &front, &rear),
        drive: DriveDesc::Drivetrain(DrivetrainDesc {
            engine: EngineDesc { max_torque: 900.0, max_rpm: 5200.0, inertia: 1.2, ..Default::default() },
            transmission: TransmissionDesc {
                gear_ratios: vec![2.9, 1.9, 1.35, 1.0],
                shift_up_rpm: 4400.0,
                shift_down_rpm: 2000.0,
                clutch_strength: 25.0,
                ..Default::default()
            },
            differentials: vec![
                DifferentialDesc { left: Some(0), right: Some(1), ratio: 4.1, engine_torque_ratio: 0.5, ..Default::default() },
                DifferentialDesc { left: Some(2), right: Some(3), ratio: 4.1, engine_torque_ratio: 0.5, ..Default::default() },
            ],
            inter_diff_limited_slip: Some(1.4),
        }),
        anti_roll: vec![
            AntiRollDesc { left: 0, right: 1, stiffness: 15000.0 },
            AntiRollDesc { left: 2, right: 3, stiffness: 15000.0 },
        ],
        max_pitch_roll_deg: 60.0,
        wheel_query: WheelQuery::Sphere,
        assists: AssistDesc { air_control_pitch: 2.5, air_control_yaw: 1.5, upright_stiffness: 0.0, upright_damping: 0.0, ..Default::default() },
        ..Default::default()
    }
}

/// Six-wheeled truck (three axles, two driven rear axles, front steer).
pub fn six_wheel_truck() -> VehicleDesc {
    let base = WheelDesc {
        radius: 0.5,
        width: 0.35,
        mass: 50.0,
        inertia: Some(6.0),
        suspension_min: 0.2,
        suspension_max: 0.55,
        preload: 0.0,
        spring: Spring::FrequencyDamping { hz: 1.4, zeta: 0.55 },
        max_steer: 30f32.to_radians(),
        max_brake_torque: 6000.0,
        max_handbrake_torque: 9000.0,
        rolling_resistance: 0.015,
        rim_material: crate::pack::WHEEL_RIM_MATERIAL,
        ..Default::default()
    };
    let front = WheelDesc { steerable: true, ..base.clone() };
    let rear = WheelDesc { driven: true, handbrake: true, max_steer: 0.0, ..base };
    let mut wheels = vec![wheel(1.1, -0.2, 2.6, &front), wheel(-1.1, -0.2, 2.6, &front)];
    for z in [-1.0f32, -2.4] {
        wheels.push(wheel(1.1, -0.2, z, &rear));
        wheels.push(wheel(-1.1, -0.2, z, &rear));
    }
    VehicleDesc {
        name: "six_wheel_truck".into(),
        parts: vec![
            box_part("cab", PartRole::Body, 2200.0, [0.95, 0.8, 0.9], [0.0, 0.7, 2.4]),
            box_part("bed", PartRole::Body, 3000.0, [0.95, 0.35, 2.4], [0.0, 0.25, -1.1]),
        ],
        wheels,
        drive: DriveDesc::Drivetrain(DrivetrainDesc {
            engine: EngineDesc { max_torque: 1200.0, max_rpm: 4200.0, inertia: 2.0, ..Default::default() },
            transmission: TransmissionDesc {
                gear_ratios: vec![4.0, 2.8, 2.0, 1.5, 1.15, 0.9],
                reverse_ratios: vec![-5.0],
                shift_up_rpm: 3600.0,
                shift_down_rpm: 1700.0,
                clutch_strength: 40.0,
                ..Default::default()
            },
            differentials: vec![
                DifferentialDesc { left: Some(2), right: Some(3), ratio: 4.5, engine_torque_ratio: 0.5, ..Default::default() },
                DifferentialDesc { left: Some(4), right: Some(5), ratio: 4.5, engine_torque_ratio: 0.5, ..Default::default() },
            ],
            inter_diff_limited_slip: Some(1.4),
        }),
        anti_roll: vec![AntiRollDesc { left: 0, right: 1, stiffness: 25000.0 }],
        spare_wheels: 2,
        ..Default::default()
    }
}

/// Arcade kart: direct drive, grippy, twitchy steering, no gearbox.
pub fn kart() -> VehicleDesc {
    let base = WheelDesc {
        radius: 0.22,
        width: 0.2,
        mass: 6.0,
        inertia: Some(0.15),
        suspension_min: 0.05,
        suspension_max: 0.18,
        spring: Spring::FrequencyDamping { hz: 2.5, zeta: 0.7 },
        max_steer: 28f32.to_radians(),
        max_brake_torque: 600.0,
        max_handbrake_torque: 900.0,
        long_mu_scale: 1.4,
        lat_mu_scale: 1.5,
        rim_material: crate::pack::WHEEL_RIM_MATERIAL,
        ..Default::default()
    };
    let front = WheelDesc { steerable: true, ..base.clone() };
    let rear = WheelDesc { driven: true, handbrake: true, max_steer: 0.0, ..base };
    VehicleDesc {
        name: "kart".into(),
        parts: vec![
            box_part("frame", PartRole::Body, 150.0, [0.42, 0.08, 0.95], [0.0, 0.0, 0.0]),
            box_part("seat", PartRole::Body, 30.0, [0.25, 0.2, 0.25], [0.0, 0.25, -0.25]),
        ],
        wheels: four_wheels(0.6, -0.02, 0.75, -0.75, &front, &rear),
        drive: DriveDesc::Direct { max_torque: 220.0, top_speed: 26.0 },
        assists: AssistDesc { air_control_pitch: 4.0, air_control_yaw: 3.0, upright_stiffness: 0.0, upright_damping: 0.0, ..Default::default() },
        steering: SteeringDesc { rate: 8.0, ..Default::default() },
        aero: AeroDesc { drag: 0.15, downforce: 0.2, center: [0.0; 3] },
        ..Default::default()
    }
}

/// Tank (Jolt `TankTest`): two tracks of nine road wheels, rear sprockets,
/// steering by track speed ratio, pivot turns when stopped.
pub fn tank() -> VehicleDesc {
    // Jolt's TankTest has nine wheels a side; seven keep both tracks within
    // MAX_VEHICLE_WHEELS (16).
    let zs = [2.95f32, 2.0, 1.0, 0.0, -1.0, -2.0, -2.75];
    let mut wheels = Vec::new();
    let mut tracks = TracksDesc::default();
    for (t, x) in [1.7f32, -1.7].into_iter().enumerate() {
        let mut ids = Vec::new();
        for (i, &z) in zs.iter().enumerate() {
            let end = i == 0 || i == zs.len() - 1;
            ids.push(wheels.len() as u32);
            wheels.push(WheelDesc {
                position: [x, if end { 0.0 } else { -0.3 }, z],
                radius: 0.3,
                width: 0.3,
                mass: 40.0,
                inertia: Some(1.0),
                suspension_min: 0.3,
                suspension_max: if end { 0.3 } else { 0.5 },
                spring: Spring::FrequencyDamping { hz: 1.0, zeta: 0.5 },
                max_steer: 0.0,
                // Jolt WheelSettingsTV uses flat friction 4 (long) / 2 (lat); lateral
                // 1 (closer to real track scrub) leaves pivots a clear margin over
                // the scrub torque instead of sitting on the stick/slip threshold.
                long_friction: Curve::default(),
                lat_friction: Curve::default(),
                long_mu_scale: 4.0,
                lat_mu_scale: 1.0,
                friction_model: FrictionModel::Box,
                rim_material: crate::pack::WHEEL_RIM_MATERIAL,
                ..Default::default()
            });
        }
        // A solid belt run over the wheels: nothing wedges between road wheels, and
        // what the track touches is driven at track speed (climbs logs and kerbs).
        // MBT-wide (0.6 m, twice the road wheels) so debris cannot slip past
        // the wheel line under the track's edge.
        let pod = Some(TrackPodDesc { width: 0.6, ..Default::default() });
        let track = TrackDesc { driven_wheel: *ids.last().unwrap(), wheels: ids, pod, ..Default::default() };
        if t == 0 {
            tracks.left = track;
        } else {
            tracks.right = track;
        }
    }
    VehicleDesc {
        name: "tank".into(),
        parts: vec![
            box_part("hull", PartRole::Body, 3200.0, [1.4, 0.5, 3.2], [0.0, 0.5, 0.0]),
            // Belly plate between the road wheels: ~0.5 m ground clearance (an MBT's),
            // so what fits between the tracks meets the hull. It stops short of the
            // track ramps, as a sloped glacis would, so a ledge meets the ramps first.
            // Nearly massless: the drivetrain and steering are tuned at the hull's
            // centre of mass, and 5 cm lower changed how hard turns end.
            box_part("belly", PartRole::Body, 1.0, [1.4, 0.25, 2.0], [0.0, -0.25, 0.0]),
            box_part("turret", PartRole::Body, 800.0, [1.0, 0.35, 1.4], [0.0, 1.35, -0.3]),
        ],
        wheels,
        drive: DriveDesc::Drivetrain(DrivetrainDesc {
            // Jolt TrackedVehicleControllerSettings (scaled-down M1 Abrams).
            engine: EngineDesc { max_torque: 500.0, min_rpm: 500.0, max_rpm: 4000.0, ..Default::default() },
            transmission: TransmissionDesc {
                gear_ratios: vec![4.0, 3.0, 2.0, 1.0],
                reverse_ratios: vec![-4.0, -3.0],
                shift_up_rpm: 3500.0,
                shift_down_rpm: 1000.0,
                ..Default::default()
            },
            differentials: Vec::new(),
            ..Default::default()
        }),
        tracks: Some(tracks),
        max_pitch_roll_deg: 60.0,
        center_of_mass_offset: [0.0, -0.5, 0.0],
        ..Default::default()
    }
}

/// Motorcycle (Jolt `MotorcycleTest`): two in-line wheels, raked fork, lean
/// controller and lean-limited steering, chain-driven rear wheel.
pub fn motorcycle() -> VehicleDesc {
    let caster = 30f32.to_radians();
    let fork = [0.0, -1.0, caster.tan()];
    let fl = (fork[1] * fork[1] + fork[2] * fork[2]).sqrt();
    let fork = [0.0, fork[1] / fl, fork[2] / fl];
    let y = -0.9 * 0.3;
    let front = WheelDesc {
        position: [0.0, y, 0.75],
        suspension_dir: Some(fork),
        steer_axis: Some([0.0, -fork[1], -fork[2]]),
        suspension_force_point: Some([0.0, y + fork[1] * 0.3, 0.75 + fork[2] * 0.3]),
        radius: 0.31,
        width: 0.05,
        mass: 12.0,
        inertia: Some(0.4),
        suspension_min: 0.3,
        suspension_max: 0.5,
        spring: Spring::FrequencyDamping { hz: 1.5, zeta: 0.5 },
        max_steer: 30f32.to_radians(),
        max_brake_torque: 500.0,
        steerable: true,
        rim_material: crate::pack::WHEEL_RIM_MATERIAL,
        ..Default::default()
    };
    let back = WheelDesc {
        position: [0.0, y, -0.75],
        suspension_force_point: Some([0.0, y - 0.3, -0.75]),
        radius: 0.31,
        width: 0.05,
        mass: 12.0,
        inertia: Some(0.4),
        suspension_min: 0.3,
        suspension_max: 0.5,
        spring: Spring::FrequencyDamping { hz: 2.0, zeta: 0.5 },
        max_steer: 0.0,
        max_brake_torque: 250.0,
        max_handbrake_torque: 250.0,
        handbrake: true,
        driven: true,
        rim_material: crate::pack::WHEEL_RIM_MATERIAL,
        ..Default::default()
    };
    VehicleDesc {
        name: "motorcycle".into(),
        parts: vec![box_part("frame", PartRole::Body, 216.0, [0.2, 0.3, 0.4], [0.0, 0.0, 0.0])],
        wheels: vec![front, back],
        drive: DriveDesc::Drivetrain(DrivetrainDesc {
            // A motorcycle crank is ~0.05 kg·m²; Jolt's car default (0.5) times the 11:1 overall
            // ratio squared would add ~600 kg of equivalent mass to braking in gear.
            engine: EngineDesc { max_torque: 150.0, min_rpm: 1000.0, max_rpm: 10000.0, inertia: 0.05, ..Default::default() },
            transmission: TransmissionDesc {
                gear_ratios: vec![2.27, 1.63, 1.3, 1.09, 0.96, 0.88],
                reverse_ratios: vec![-4.0],
                shift_up_rpm: 8000.0,
                shift_down_rpm: 2000.0,
                clutch_strength: 2.0,
                ..Default::default()
            },
            differentials: vec![DifferentialDesc {
                left: None,
                right: Some(1),
                ratio: 1.93 * 40.0 / 16.0,
                left_right_split: 1.0,
                limited_slip_ratio: None,
                engine_torque_ratio: 1.0,
            }],
            inter_diff_limited_slip: None,
        }),
        steering: SteeringDesc { ackermann: 0.0, rate: 1.5, ..Default::default() },
        // A 35° lean cap keeps the lean-limited steer below the tyres' grip.
        motorcycle: Some(MotorcycleDesc { max_lean_deg: 35.0, ..Default::default() }),
        // Rider + bike CdA ~0.6 m² → ½ρ·CdA ≈ 0.35 (Jolt's sample has no drag and tops out
        // only at the rev limiter).
        // Drag at the centre of mass: applied above it, it unloaded the front tyre at
        // speed until any disturbance could swing the heading.
        aero: AeroDesc { drag: 0.35, downforce: 0.0, center: [0.0, 0.0, 0.0] },
        // The engine delivers ~2× the rear tyre's grip in the low gears; without TC full
        // throttle spins it and a spinning tyre has no side grip.
        assists: AssistDesc { traction_control: true, ..Default::default() },
        // Jolt offsets the box's center of mass to its bottom face.
        center_of_mass_offset: [0.0, -0.3, 0.0],
        max_pitch_roll_deg: 60.0,
        ..Default::default()
    }
}

/// vibe-land's dune buggy ("Sand rail"), baked by `engine/scenes/dune-buggy`
/// (`bun run bake:physics`) into `assets/dune_buggy.json`: 17 parts with their
/// recipe masses and primitive colliders, mounts rated from the measured joints,
/// and the rig's hubs and travel. Handling follows vibe-land's garage setup:
/// springs at 30% static sag, critically damped, all-wheel drive.
pub fn dune_buggy() -> VehicleDesc {
    vibe_vehicle("dune_buggy", include_str!("../assets/dune_buggy.json"))
}

/// vibe-land's Baja trophy truck ("Baja / T1"): the buggy chassis under panel
/// bodywork, an open bed and a tailgate.
pub fn vibe_trophy() -> VehicleDesc {
    vibe_vehicle("vibe_trophy", include_str!("../assets/vibe_trophy.json"))
}

/// vibe-land's rally hatchback ("Vector / R3"): enclosed cabin, hatch, rear wing.
pub fn vibe_rally() -> VehicleDesc {
    vibe_vehicle("vibe_rally", include_str!("../assets/vibe_rally.json"))
}

/// vibe-land's monster truck ("Goliath / M4"): lifted, 0.72 m tyres half again as wide.
pub fn vibe_monster() -> VehicleDesc {
    vibe_vehicle("vibe_monster", include_str!("../assets/vibe_monster.json"))
}

/// vibe-land's derby sedan ("Brawler / D8"): reinforced doors, window guards, trunk.
pub fn vibe_derby() -> VehicleDesc {
    vibe_vehicle("vibe_derby", include_str!("../assets/vibe_derby.json"))
}

/// vibe-land's sprint car ("Firefly / S9"): open wheels, top and front wings, nerf bars.
pub fn vibe_sprint() -> VehicleDesc {
    vibe_vehicle("vibe_sprint", include_str!("../assets/vibe_sprint.json"))
}

/// vibe-land's semi tractor ("Hauler / H6"); its flatbed is [`vibe_semi_trailer`],
/// hitched at [`VIBE_SEMI_HITCH`] by a ball joint.
pub fn vibe_semi() -> VehicleDesc {
    vibe_vehicle("vibe_semi", include_str!("../assets/vibe_semi.json"))
}

/// The semi's fifth wheel (design space, both vehicles): the trailer's kingpin.
pub const VIBE_SEMI_HITCH: [f32; 3] = [0.0, 1.12, -2.3];

/// The semi's flatbed trailer: one rigid body on two raycast wheels (its leaf
/// springs), no drive, hitched at [`VIBE_SEMI_HITCH`]. Its front rests on the
/// tractor, so each wheel's spring is sized for a quarter of its weight.
pub fn vibe_semi_trailer() -> VehicleDesc {
    #[derive(serde::Deserialize)]
    struct Wheel {
        hub: [f32; 3],
        radius: f32,
        width: f32,
        mass: f32,
    }
    #[derive(serde::Deserialize)]
    struct Trailer {
        mass: f32,
        colliders: Vec<PartColliderDesc>,
        hitch: [f32; 3],
        wheels: Vec<Wheel>,
    }
    #[derive(serde::Deserialize)]
    struct Asset {
        trailer: Trailer,
    }
    let a: Asset = serde_json::from_str(include_str!("../assets/vibe_semi.json")).expect("assets/vibe_semi.json");
    let t = a.trailer;
    debug_assert_eq!(t.hitch, VIBE_SEMI_HITCH);
    const G: f32 = 9.81;
    let (compression, extension) = (0.10, 0.08); // vibe-land's semi handling
    let corner = 0.25 * (t.mass + t.wheels.iter().map(|w| w.mass).sum::<f32>());
    let k = corner * G / (0.3 * (compression + extension));
    let wheels = t
        .wheels
        .iter()
        .map(|w| {
            let neutral = 0.05 + compression;
            WheelDesc {
                position: [w.hub[0], w.hub[1] + neutral, w.hub[2]],
                radius: w.radius,
                width: w.width,
                mass: w.mass,
                inertia: None,
                suspension_min: 0.05,
                suspension_max: neutral + extension,
                spring: Spring::StiffnessDamping { k, c: 2.0 * (k * corner).sqrt() },
                max_steer: 0.0,
                max_brake_torque: 1500.0,
                rolling_resistance: 0.015,
                long_mu_scale: 1.2,
                lat_mu_scale: 1.2,
                rim_material: crate::pack::WHEEL_RIM_MATERIAL,
                ..Default::default()
            }
        })
        .collect();
    VehicleDesc {
        name: "vibe_semi_trailer".into(),
        parts: vec![PartDesc { name: "trailer".into(), role: PartRole::Body, mass: t.mass, colliders: t.colliders, mount: MountDesc::default() }],
        wheels,
        drive: DriveDesc::Direct { max_torque: 0.0, top_speed: 30.0 },
        wheel_query: WheelQuery::Sphere,
        ..Default::default()
    }
}

/// A vibe-land catalog vehicle from its baked asset (`engine/scenes/dune-buggy`):
/// parts with their recipe masses, primitive colliders and rated mounts (parts
/// bolted to parts), the rig's hubs and travel. Handling is vibe-land's garage
/// setup: springs at 30% static sag, critically damped, bump stops ten times
/// as stiff, all-wheel drive at 7.5 m/s² to 30 m/s.
pub fn vibe_vehicle(name: &str, json: &str) -> VehicleDesc {
    #[derive(serde::Deserialize)]
    struct Wheel {
        hub: [f32; 3],
        radius: f32,
        width: f32,
        mass: f32,
        compression: f32,
        extension: f32,
        steerable: bool,
        mount: MountDesc,
    }
    #[derive(serde::Deserialize)]
    struct Steering {
        max_steer_deg: f32,
    }
    #[derive(serde::Deserialize)]
    struct Asset {
        total_mass: f32,
        parts: Vec<PartDesc>,
        wheels: Vec<Wheel>,
        steering: Steering,
    }
    let a: Asset = serde_json::from_str(json).unwrap_or_else(|e| panic!("{name} asset: {e}"));
    const G: f32 = 9.81;
    const SAG: f32 = 0.3; // vibe-land customization.mjs STATIC_SAG_FRACTION
    const S_MIN: f32 = 0.05;
    let corner_mass = a.total_mass / a.wheels.len() as f32;
    let wheels = a
        .wheels
        .iter()
        .map(|w| {
            // Design pose = the rig's neutral: room to compress and to droop either side.
            let neutral = S_MIN + w.compression;
            let travel = w.compression + w.extension;
            let k = corner_mass * G / (SAG * travel);
            let c = 2.0 * (k * corner_mass).sqrt();
            let hz = (k / corner_mass).sqrt() / std::f32::consts::TAU;
            WheelDesc {
                position: [w.hub[0], w.hub[1] + neutral, w.hub[2]],
                radius: w.radius,
                width: w.width,
                mass: w.mass,
                inertia: None,
                suspension_min: S_MIN,
                suspension_max: neutral + w.extension,
                spring: Spring::StiffnessDamping { k, c },
                // vibe-land's bump stop: ten times the spring rate (√10 × its frequency).
                bump_stop: BumpStop::Spring { hz: hz * 10f32.sqrt(), zeta: 0.7, max_force: 0.0 },
                steerable: w.steerable,
                max_steer: if w.steerable { a.steering.max_steer_deg.to_radians() } else { 0.0 },
                driven: true,
                handbrake: !w.steerable,
                // Heavier models brake harder: the buggy (1.47 t) keeps 2.5 kN·m.
                max_brake_torque: 2500.0 * (a.total_mass / 1500.0).max(1.0),
                max_handbrake_torque: 4000.0,
                long_mu_scale: 1.4,
                lat_mu_scale: 1.4,
                rolling_resistance: 0.015,
                rim_material: crate::pack::WHEEL_RIM_MATERIAL,
                mount: w.mount,
                ..Default::default()
            }
        })
        .collect::<Vec<_>>();
    // vibe-land drivingSetup: 7.5 m/s² (within 80% of the grip budget on four
    // driven wheels), 30 m/s top speed.
    let r = wheels[0].radius;
    let max_torque = a.total_mass * 7.5 * r / wheels.len() as f32;
    VehicleDesc {
        name: name.into(),
        parts: a.parts,
        wheels,
        drive: DriveDesc::Direct { max_torque, top_speed: 30.0 },
        // vibe-land PHYSX_CENTER_OF_MASS_Y: 0.2 m below the computed centre.
        center_of_mass_offset: [0.0, -0.2, 0.0],
        // Full lock to 6 m/s, tapering to 35% at 28 m/s (vibe-land's steering shaping).
        steering: SteeringDesc { speed_curve: Curve(vec![[0.0, 1.0], [6.0, 1.0], [28.0, 0.35]]), ..Default::default() },
        wheel_query: WheelQuery::Sphere,
        ..Default::default()
    }
}

/// Every preset by name (also exported as JSON for the browser demo).
pub fn all() -> Vec<VehicleDesc> {
    vec![
        jolt_default(),
        sedan(),
        destructible_sedan(),
        sports(),
        monster_truck(),
        six_wheel_truck(),
        kart(),
        tank(),
        motorcycle(),
        dune_buggy(),
        vibe_trophy(),
        vibe_rally(),
        vibe_monster(),
        vibe_derby(),
        vibe_sprint(),
        vibe_semi(),
        vibe_semi_trailer(),
    ]
}

pub fn by_name(name: &str) -> Option<VehicleDesc> {
    all().into_iter().find(|d| d.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{pack, requirements, VehicleSlots};
    use engine_schema::vehicle::*;
    use glam::{Quat, Vec3};

    #[test]
    fn dune_buggy_is_complete() {
        let d = dune_buggy();
        assert_eq!(d.parts.len(), 17);
        assert!(d.parts.len() <= 32);
        assert_eq!(d.parts[0].name, "frame");
        assert!(d.parts.iter().all(|p| !p.colliders.is_empty() && p.mass > 0.0), "every part has mass and shape");
        // The root never breaks; every other part and every wheel can.
        assert!(!d.parts[0].mount.breakable());
        assert!(d.parts[1..].iter().all(|p| p.mount.breakable()), "{:?}", d.parts.iter().map(|p| (&p.name, p.mount)).collect::<Vec<_>>());
        assert!(d.wheels.iter().all(|w| w.mount.breakable()));
        assert!(d.breakable());
        // Wheel order FL, FR, RL, RR with +X left, +Z forward.
        let p: Vec<[f32; 3]> = d.wheels.iter().map(|w| w.position).collect();
        assert!(p[0][0] > 0.0 && p[1][0] < 0.0 && p[2][0] > 0.0 && p[3][0] < 0.0);
        assert!(p[0][2] > 0.0 && p[2][2] < 0.0);
        assert!(d.wheels[0].steerable && !d.wheels[2].steerable);
        // The recipe's mass, about 1.47 t with four 128 kg wheels.
        let m: f32 = d.parts.iter().map(|p| p.mass).sum::<f32>() + d.wheels.iter().map(|w| w.mass).sum::<f32>();
        assert!((1400.0..1550.0).contains(&m), "mass {m}");
        let req = requirements(&d);
        let packed = pack(&d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default());
        assert_eq!(req.colliders as usize, packed.colliders.len());
        assert_eq!(packed.parts.len(), 17);
    }

    #[test]
    fn mounts_pack_their_limits() {
        let mut d = destructible_sedan();
        let flags = |d: &VehicleDesc| pack(d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default()).vehicle[V_CFG as usize][3];
        // Every existing preset stays unbreakable.
        assert!(!d.breakable());
        assert_eq!(flags(&d) & VF_BREAKABLE, 0);
        let p = pack(&d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default());
        for part in &p.parts {
            assert_eq!(part.len(), PART_STRIDE as usize);
            assert_eq!(f32::from_bits(part[P_MOUNT as usize][3]), 1e30);
            assert_eq!(f32::from_bits(part[P_LOAD as usize][0]), 1e30);
        }
        // A finite force limit; torque left unbreakable.
        d.parts[1].mount = MountDesc { break_force: 5e4, break_torque: 0.0, anchor: Some([0.0, 0.5, 0.0]), parent: None };
        assert!(d.breakable());
        assert_ne!(flags(&d) & VF_BREAKABLE, 0);
        let c = d.center_of_mass();
        let p = pack(&d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default());
        let m = p.parts[1][P_MOUNT as usize];
        assert_eq!(f32::from_bits(m[3]), 5e4);
        assert_eq!(f32::from_bits(p.parts[1][P_LOAD as usize][0]), 1e30);
        // The anchor is centred with the rest of the design.
        assert!((f32::from_bits(m[1]) - (0.5 - c[1])).abs() < 1e-5);
        // The root's mount is never packed, whatever it says.
        d.parts[0].mount.break_force = 1.0;
        let p = pack(&d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default());
        assert_eq!(f32::from_bits(p.parts[0][P_MOUNT as usize][3]), 1e30);
    }
}

#[cfg(test)]
mod vibe_tests {
    use super::*;
    use crate::pack::{pack, requirements, VehicleSlots};
    use glam::{Quat, Vec3};

    #[test]
    fn every_vibe_land_model_packs() {
        for d in [dune_buggy(), vibe_trophy(), vibe_rally(), vibe_monster(), vibe_derby(), vibe_sprint(), vibe_semi()] {
            assert!(d.parts.len() <= 32 && d.parts[0].name == "frame", "{}", d.name);
            assert!(d.parts.iter().all(|p| !p.colliders.is_empty() && p.mass > 0.0), "{}", d.name);
            assert!(d.parts[1..].iter().all(|p| p.mount.breakable()), "{}: every part has a rated mount", d.name);
            for (i, p) in d.parts.iter().enumerate() {
                if let Some(parent) = p.mount.parent {
                    assert!((parent as usize) < i, "{}: {} on a later part", d.name, p.name);
                }
            }
            assert_eq!(d.wheels.len(), 4);
            let p: Vec<[f32; 3]> = d.wheels.iter().map(|w| w.position).collect();
            assert!(p[0][0] > 0.0 && p[1][0] < 0.0 && p[0][2] > p[2][2], "{}: FL, FR, RL, RR", d.name);
            let req = requirements(&d);
            let packed = pack(&d, Vec3::ZERO, Quat::IDENTITY, VehicleSlots::default());
            assert_eq!(req.colliders as usize, packed.colliders.len(), "{}", d.name);
        }
        let monster = vibe_monster();
        assert!(monster.wheels[0].radius > 0.7 && monster.wheels[0].width > 0.4, "monster tyres");
    }

    #[test]
    fn the_semi_trailer_hitches_behind_the_cab() {
        let t = vibe_semi_trailer();
        assert_eq!(t.wheels.len(), 2);
        assert!(t.wheels.iter().all(|w| !w.driven && !w.steerable));
        // Hitch over the tractor's rear axle, the trailer axle well behind it (+Z forward).
        let tractor_rear = vibe_semi().wheels[2].position[2];
        assert!(VIBE_SEMI_HITCH[2] < 0.0 && VIBE_SEMI_HITCH[2] > tractor_rear - 1.5);
        assert!(t.wheels[0].position[2] < VIBE_SEMI_HITCH[2] - 2.0);
        assert!(t.parts[0].mass > 1000.0);
    }
}
