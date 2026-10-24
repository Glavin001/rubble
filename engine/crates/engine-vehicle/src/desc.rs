//! Serializable vehicle descriptors. Every field has a default, so presets and
//! user JSON only spell out what differs. Units are SI; the chassis body origin
//! is its centre of mass; `forward`/`up` are chassis-local (default +Z / +Y).

use serde::{Deserialize, Serialize};

/// Piecewise-linear curve as `[x, y]` points with clamped ends.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Curve(pub Vec<[f32; 2]>);

impl Curve {
    pub fn eval(&self, x: f32) -> f32 {
        let p = &self.0;
        if p.is_empty() {
            return 1.0;
        }
        if x <= p[0][0] {
            return p[0][1];
        }
        for w in p.windows(2) {
            if x <= w[1][0] {
                let t = (x - w[0][0]) / (w[1][0] - w[0][0]).max(1e-9);
                return w[0][1] + (w[1][1] - w[0][1]) * t;
            }
        }
        p[p.len() - 1][1]
    }

    pub fn peak(&self) -> f32 {
        if self.0.is_empty() {
            return 1.0;
        }
        self.0.iter().fold(0.0f32, |m, p| m.max(p[1]))
    }

    /// Jolt `WheelSettingsWV::mLongitudinalFriction` default (x = slip ratio).
    pub fn jolt_longitudinal() -> Self {
        Self(vec![[0.0, 0.0], [0.06, 1.2], [0.2, 1.0]])
    }

    /// Jolt `WheelSettingsWV::mLateralFriction` default (x = slip angle, degrees).
    pub fn jolt_lateral() -> Self {
        Self(vec![[0.0, 0.0], [3.0, 1.2], [20.0, 1.0]])
    }

    /// Road-car steer-vs-speed scale (x = m/s): full lock while parking, a quarter at
    /// motorway speed, so a full keyboard steer stays near the grip limit instead of
    /// asking for ~2 g and sliding.
    pub fn road_steering() -> Self {
        Self(vec![[0.0, 1.0], [8.0, 1.0], [20.0, 0.5], [35.0, 0.32], [60.0, 0.25]])
    }

    /// Jolt `VehicleEngineSettings::mNormalizedTorque` default (x = RPM / maxRPM).
    pub fn jolt_torque() -> Self {
        Self(vec![[0.0, 0.8], [0.66, 1.0], [1.0, 0.8]])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Spring {
    /// Jolt-style: stiffness/damping derived from the effective sprung mass.
    FrequencyDamping { hz: f32, zeta: f32 },
    /// Absolute spring rate (N/m) and damper (N·s/m).
    StiffnessDamping { k: f32, c: f32 },
}

impl Default for Spring {
    fn default() -> Self {
        Spring::FrequencyDamping { hz: 1.5, zeta: 0.5 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BumpStop {
    /// Compliant stop (recommended: rigid stops spike loads at full compression).
    Spring { hz: f32, zeta: f32, max_force: f32 },
    /// Hard stop (Jolt `mSuspensionMaxUpPart` parity).
    Rigid,
    None,
}

impl Default for BumpStop {
    fn default() -> Self {
        BumpStop::Spring { hz: 6.0, zeta: 0.7, max_force: 0.0 }
    }
}

/// How a wheel's longitudinal and lateral friction limits combine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrictionModel {
    /// (f_L/μ_L·F_n)² + (f_T/μ_T·F_n)² ≤ 1: braking or driving costs cornering grip.
    #[default]
    Ellipse,
    /// |f_L| ≤ μ_L·F_n and |f_T| ≤ μ_T·F_n independently (Jolt); lets tracks
    /// scrub sideways at full drive force so tanks can pivot.
    Box,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WheelDesc {
    /// Suspension attachment (hardpoint) in chassis space.
    pub position: [f32; 3],
    /// Suspension travel direction (defaults to -up).
    pub suspension_dir: Option<[f32; 3]>,
    /// Steering axis (defaults to up; tilt for caster / kingpin).
    pub steer_axis: Option<[f32; 3]>,
    /// Wheel forward (toe) and up (camber) in chassis space.
    pub forward: Option<[f32; 3]>,
    pub up: Option<[f32; 3]>,
    pub radius: f32,
    pub width: f32,
    pub mass: f32,
    /// Spin inertia (kg·m²); default ½·m·r².
    pub inertia: Option<f32>,
    pub angular_damping: f32,
    pub suspension_min: f32,
    pub suspension_max: f32,
    pub preload: f32,
    pub spring: Spring,
    pub bump_stop: BumpStop,
    /// Max steering angle (radians) at full steer input.
    pub max_steer: f32,
    pub max_brake_torque: f32,
    pub max_handbrake_torque: f32,
    pub handbrake: bool,
    pub driven: bool,
    pub steerable: bool,
    /// Driven-torque multiplier in direct-drive mode.
    pub drive_multiplier: f32,
    pub long_friction: Curve,
    pub lat_friction: Curve,
    pub long_mu_scale: f32,
    pub lat_mu_scale: f32,
    /// How the longitudinal and lateral limits combine.
    pub friction_model: FrictionModel,
    pub rolling_resistance: f32,
    /// 1 = physical; < 1 lifts the lateral force toward the COM (less body roll).
    pub roll_influence: f32,
    /// Apply the suspension force at this chassis-space point instead of the
    /// contact (Jolt `mSuspensionForcePoint`; steadies motorcycles).
    pub suspension_force_point: Option<[f32; 3]>,
    /// Rim spheres of the free rigid wheel spawned on detach.
    pub rim_spheres: u32,
    pub rim_material: u32,
    /// Wheel mount (hub, bearing, arms): past its limits the wheel comes off as
    /// a free body, as `detach_wheel` would.
    pub mount: MountDesc,
}

impl Default for WheelDesc {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            suspension_dir: None,
            steer_axis: None,
            forward: None,
            up: None,
            radius: 0.3,
            width: 0.1,
            mass: 20.0,
            inertia: Some(0.9),
            angular_damping: 0.2,
            suspension_min: 0.3,
            suspension_max: 0.5,
            preload: 0.0,
            spring: Spring::default(),
            bump_stop: BumpStop::default(),
            max_steer: 70f32.to_radians(),
            max_brake_torque: 1500.0,
            max_handbrake_torque: 4000.0,
            handbrake: false,
            driven: false,
            steerable: false,
            drive_multiplier: 1.0,
            long_friction: Curve::jolt_longitudinal(),
            lat_friction: Curve::jolt_lateral(),
            long_mu_scale: 1.0,
            lat_mu_scale: 1.0,
            friction_model: FrictionModel::Ellipse,
            rolling_resistance: 0.0,
            roll_influence: 1.0,
            suspension_force_point: None,
            rim_spheres: 12,
            rim_material: 7,
            mount: MountDesc::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShapeDesc {
    Box { half: [f32; 3] },
    Sphere { radius: f32 },
    /// Capsule along local Y.
    Capsule { radius: f32, half_length: f32 },
}

impl ShapeDesc {
    pub fn volume(&self) -> f32 {
        match *self {
            ShapeDesc::Box { half } => 8.0 * half[0] * half[1] * half[2],
            ShapeDesc::Sphere { radius } => 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3),
            ShapeDesc::Capsule { radius, half_length } => {
                std::f32::consts::PI * radius * radius * (2.0 * half_length + 4.0 / 3.0 * radius)
            }
        }
    }

    /// Principal inertia about the shape centre for mass `m` (diagonal, local axes).
    pub fn inertia(&self, m: f32) -> [f32; 3] {
        match *self {
            ShapeDesc::Box { half } => {
                let [x, y, z] = [2.0 * half[0], 2.0 * half[1], 2.0 * half[2]];
                [m * (y * y + z * z) / 12.0, m * (x * x + z * z) / 12.0, m * (x * x + y * y) / 12.0]
            }
            ShapeDesc::Sphere { radius } => {
                let i = 0.4 * m * radius * radius;
                [i, i, i]
            }
            ShapeDesc::Capsule { radius, half_length } => {
                let h = 2.0 * half_length;
                let ia = 0.5 * m * radius * radius;
                let it = m * (3.0 * radius * radius + h * h) / 12.0;
                [it, ia, it]
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PartColliderDesc {
    pub shape: ShapeDesc,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub material: u32,
}

impl Default for PartColliderDesc {
    fn default() -> Self {
        Self {
            shape: ShapeDesc::Box { half: [0.9, 0.3, 2.0] },
            position: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            material: 0,
        }
    }
}

/// What severing a part does to the vehicle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartRole {
    /// Bodywork: losing it only changes mass and shape.
    #[default]
    Body,
    /// Losing it disconnects the engine (no drive torque).
    Engine,
    /// Losing it cuts drive to these wheels (they still roll, brake, steer).
    DriveShaft { wheels: u32 },
    /// Losing it locks these wheels straight.
    Steering { wheels: u32 },
    /// Losing it removes brakes from these wheels.
    Brake { wheels: u32 },
}

/// How a part (or wheel) is fixed to the chassis. Every step `vehicle_loads`
/// measures the wrench the mount carries (the part's own inertia and weight
/// against the contacts on its colliders); past either limit the part is
/// severed before the next step, exactly as `detach_part` would. A limit ≤ 0 is
/// unbreakable (the default), as for joints. Part 0's mount is never checked.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MountDesc {
    /// N; the magnitude of the force through the mount.
    pub break_force: f32,
    /// N·m; the moment about `anchor`.
    pub break_torque: f32,
    /// Design-space point the moment is taken about (default: the part's
    /// centre of mass; a wheel's: its hardpoint).
    pub anchor: Option<[f32; 3]>,
    /// Parts only: the part this one is bolted to (its index, lower than this
    /// part's; default the chassis). While both are on the car this part's load
    /// counts toward the parent's mount; when the parent leaves (severed, or its
    /// mount broke) this part goes with it as one rigid piece.
    pub parent: Option<u32>,
}

impl MountDesc {
    pub fn breakable(&self) -> bool {
        self.break_force > 0.0 || self.break_torque > 0.0
    }

    /// A mount on the chassis with these limits.
    pub fn limits(break_force: f32, break_torque: f32) -> Self {
        Self { break_force, break_torque, ..Default::default() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PartDesc {
    pub name: String,
    pub role: PartRole,
    pub mass: f32,
    pub colliders: Vec<PartColliderDesc>,
    pub mount: MountDesc,
}

impl Default for PartDesc {
    fn default() -> Self {
        Self {
            name: "body".into(),
            role: PartRole::Body,
            mass: 1500.0,
            colliders: vec![PartColliderDesc::default()],
            mount: MountDesc::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineDesc {
    pub max_torque: f32,
    pub min_rpm: f32,
    pub max_rpm: f32,
    pub inertia: f32,
    pub angular_damping: f32,
    /// Normalised torque vs RPM / maxRPM.
    pub torque_curve: Curve,
}

impl Default for EngineDesc {
    fn default() -> Self {
        Self {
            max_torque: 500.0,
            min_rpm: 1000.0,
            max_rpm: 6000.0,
            inertia: 0.5,
            angular_damping: 0.2,
            torque_curve: Curve::jolt_torque(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TransmissionMode {
    #[default]
    Auto,
    Manual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TransmissionDesc {
    pub mode: TransmissionMode,
    pub gear_ratios: Vec<f32>,
    /// Negative ratios.
    pub reverse_ratios: Vec<f32>,
    pub switch_time: f32,
    pub clutch_release_time: f32,
    pub switch_latency: f32,
    pub shift_up_rpm: f32,
    pub shift_down_rpm: f32,
    pub clutch_strength: f32,
}

impl Default for TransmissionDesc {
    fn default() -> Self {
        Self {
            mode: TransmissionMode::Auto,
            gear_ratios: vec![2.66, 1.78, 1.3, 1.0, 0.74],
            reverse_ratios: vec![-2.90],
            switch_time: 0.5,
            clutch_release_time: 0.3,
            switch_latency: 0.5,
            shift_up_rpm: 4000.0,
            shift_down_rpm: 2000.0,
            clutch_strength: 10.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DifferentialDesc {
    pub left: Option<u32>,
    pub right: Option<u32>,
    pub ratio: f32,
    /// 0 = all torque left, 1 = all right.
    pub left_right_split: f32,
    /// Jolt `mLimitedSlipRatio`; `None` = open differential.
    pub limited_slip_ratio: Option<f32>,
    pub engine_torque_ratio: f32,
}

impl Default for DifferentialDesc {
    fn default() -> Self {
        Self {
            left: None,
            right: None,
            ratio: 3.42,
            left_right_split: 0.5,
            limited_slip_ratio: Some(1.4),
            engine_torque_ratio: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DrivetrainDesc {
    pub engine: EngineDesc,
    pub transmission: TransmissionDesc,
    pub differentials: Vec<DifferentialDesc>,
    /// Torque bias between differentials (Jolt `mDifferentialLimitedSlipRatio`); `None` = open.
    pub inter_diff_limited_slip: Option<f32>,
}

impl Default for DrivetrainDesc {
    fn default() -> Self {
        Self {
            engine: EngineDesc::default(),
            transmission: TransmissionDesc::default(),
            differentials: Vec::new(),
            inter_diff_limited_slip: Some(1.4),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DriveDesc {
    /// Engine + clutch + gearbox + differentials (Jolt-faithful).
    Drivetrain(DrivetrainDesc),
    /// Torque straight to driven wheels (arcade / electric). Full torque up to
    /// a third of `top_speed`, then a linear fade to zero.
    Direct { max_torque: f32, top_speed: f32 },
}

impl Default for DriveDesc {
    fn default() -> Self {
        DriveDesc::Drivetrain(DrivetrainDesc::default())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AntiRollDesc {
    pub left: u32,
    pub right: u32,
    pub stiffness: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SteeringDesc {
    /// 0 = parallel steer, 1 = full Ackermann.
    pub ackermann: f32,
    /// Auto-derived from wheel positions when `None`.
    pub wheelbase: Option<f32>,
    pub track: Option<f32>,
    /// Steer input slew rate (input units / s; 0 = instant).
    pub rate: f32,
    /// Steer scale vs forward speed (m/s); empty = 1.
    pub speed_curve: Curve,
}

impl Default for SteeringDesc {
    fn default() -> Self {
        Self { ackermann: 1.0, wheelbase: None, track: None, rate: 4.0, speed_curve: Curve::default() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseDesc {
    pub throttle_rate: f32,
    pub brake_rate: f32,
    pub handbrake_rate: f32,
}

impl Default for ResponseDesc {
    fn default() -> Self {
        Self { throttle_rate: 6.0, brake_rate: 10.0, handbrake_rate: 20.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AeroDesc {
    /// ½·ρ·Cd·A (N per (m/s)²).
    pub drag: f32,
    /// ½·ρ·Cl·A along -up per (forward m/s)².
    pub downforce: f32,
    pub center: [f32; 3],
}

/// Motorcycle lean controller (Jolt `MotorcycleControllerSettings`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MotorcycleDesc {
    pub max_lean_deg: f32,
    /// Lean spring (N·m/rad) toward the direction of the ground force.
    pub lean_spring: f32,
    pub lean_damping: f32,
    pub lean_integration: f32,
    pub lean_integration_decay: f32,
    /// Exponential smoothing of the target lean [0, 1).
    pub lean_smoothing: f32,
    /// Cap the steer angle by the lean it would need at the current speed.
    pub lean_steering_limit: bool,
}

impl Default for MotorcycleDesc {
    fn default() -> Self {
        Self {
            max_lean_deg: 45.0,
            lean_spring: 5000.0,
            lean_damping: 1000.0,
            lean_integration: 0.0,
            lean_integration_decay: 4.0,
            lean_smoothing: 0.8,
            lean_steering_limit: true,
        }
    }
}

/// One track (Jolt `VehicleTrackSettings`): its wheels share one spin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackDesc {
    /// Indices into `VehicleDesc::wheels`.
    pub wheels: Vec<u32>,
    /// The sprocket; carries the track's inertia, brake and drive.
    pub driven_wheel: u32,
    pub inertia: f32,
    pub angular_damping: f32,
    pub max_brake_torque: f32,
    pub differential_ratio: f32,
    /// Solid track run on the chassis: closes the gaps between road wheels and
    /// drives what it touches at the track's speed (a conveyor belt).
    pub pod: Option<TrackPodDesc>,
}

impl Default for TrackDesc {
    fn default() -> Self {
        Self {
            wheels: Vec::new(),
            driven_wheel: 0,
            inertia: 10.0,
            angular_damping: 0.5,
            max_brake_torque: 15000.0,
            differential_ratio: 6.0,
            pod: None,
        }
    }
}

/// Track pod: two belt boxes over a track's wheels (`pack::track_pod_colliders`).
/// One spans every wheel from their highest bottomed-out tread line up to `top`;
/// the other spans the road wheels at the lowest tread line, so a raised sprocket
/// and idler give the run a stepped nose. Both float `clearance` above the
/// bottomed-out wheels, so the ground probes carry the load on flat ground and
/// the pod meets only debris, steps and hard landings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackPodDesc {
    /// Track width (m); 0 = the widest wheel's width.
    pub width: f32,
    /// Gap above the bottomed-out wheels' tread line (m).
    pub clearance: f32,
    /// Height of the top run above the wheel hardpoints (chassis up, m).
    pub top: f32,
    /// Belt-to-world friction material.
    pub material: u32,
}

impl Default for TrackPodDesc {
    fn default() -> Self {
        Self { width: 0.0, clearance: 0.05, top: 0.3, material: 0 }
    }
}

/// Tracked drive (tanks): left/right tracks, steering by drive ratio.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TracksDesc {
    pub left: TrackDesc,
    pub right: TrackDesc,
    /// How much steer feeds each track in the arcade mix (left = throttle + k·steer,
    /// right = throttle − k·steer); 1 pivots at full steer from rest.
    pub steer_strength: f32,
    /// Cap the track ratio so the turn it asks for needs at most this lateral
    /// acceleration (in g) at the current speed; 0 = uncapped, which spins a fast
    /// tank out at full steer.
    pub turn_grip_g: f32,
}

impl Default for TracksDesc {
    fn default() -> Self {
        Self {
            left: TrackDesc::default(),
            right: TrackDesc::default(),
            steer_strength: 1.0,
            turn_grip_g: 0.5,
        }
    }
}

/// Opt-in arcade assists (explicit torques scaled by the chassis inertia).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AssistDesc {
    /// Airborne pitch acceleration per unit forward input (rad/s²).
    pub air_control_pitch: f32,
    /// Airborne yaw acceleration per unit steer input (rad/s²).
    pub air_control_yaw: f32,
    /// Torque toward world up per radian of tilt (1/s²); 0 = off.
    pub upright_stiffness: f32,
    /// Damping of the tilt rate (1/s).
    pub upright_damping: f32,
    /// Traction control: cut engine torque while a driven wheel's slip ratio is past
    /// the grip peak, so full throttle does not spin the tyre (a spinning tyre has
    /// little lateral grip left and swings the tail out).
    pub traction_control: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VehicleDesc {
    pub name: String,
    pub forward: [f32; 3],
    pub up: [f32; 3],
    /// Chassis collider groups (bodywork, engine, …). Mass sums with the wheels.
    pub parts: Vec<PartDesc>,
    /// Override the computed chassis inertia (kg·m², body axes).
    pub inertia: Option<[f32; 3]>,
    /// Move the body origin to the computed centre of mass (parts + wheels);
    /// positions in this descriptor are then design-space coordinates.
    pub auto_center_of_mass: bool,
    /// Extra shift of the centre of mass (design space), e.g. lower it for stability.
    pub center_of_mass_offset: [f32; 3],
    pub wheels: Vec<WheelDesc>,
    pub drive: DriveDesc,
    pub anti_roll: Vec<AntiRollDesc>,
    pub steering: SteeringDesc,
    pub response: ResponseDesc,
    pub aero: AeroDesc,
    pub assists: AssistDesc,
    /// Contacts steeper than this (from the suspension axis) are ignored.
    pub max_slope_deg: f32,
    /// Jolt `mMaxPitchRollAngle`: chassis up stays within this cone of world up
    /// (180 = unlimited).
    pub max_pitch_roll_deg: f32,
    /// 1 = world gravity.
    pub gravity_scale: f32,
    pub can_sleep: bool,
    /// Throttle against the direction of travel brakes first (arcade reverse).
    pub reverse_brake: bool,
    /// Spare wheel slots for `add_wheel` at runtime.
    pub spare_wheels: u32,
    /// Ground probe per wheel (Jolt VehicleCollisionTester).
    pub wheel_query: WheelQuery,
    /// Steps between full ground probes (0/1 = every step); in between, contacts
    /// are extrapolated on the last hit plane. Staggered per wheel.
    pub query_interval: u32,
    /// Motorcycle lean controller (two in-line wheels).
    pub motorcycle: Option<MotorcycleDesc>,
    /// Tracked vehicle (needs a `Drivetrain` drive; differentials are unused).
    pub tracks: Option<TracksDesc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WheelQuery {
    /// Ray along the suspension (cheapest; Jolt `VehicleCollisionTesterRay`).
    #[default]
    Ray,
    /// Sphere of the wheel radius swept along the suspension (kerbs, box edges).
    Sphere,
}

impl VehicleDesc {
    /// True when some part (other than the root) or wheel can break off.
    pub fn breakable(&self) -> bool {
        self.parts.iter().skip(1).any(|p| p.mount.breakable()) || self.wheels.iter().any(|w| w.mount.breakable())
    }

    /// Design-space centre of mass the body origin is placed at.
    pub fn center_of_mass(&self) -> [f32; 3] {
        let off = self.center_of_mass_offset;
        if !self.auto_center_of_mass {
            return off;
        }
        let up = glam::Vec3::from_array(self.up).normalize();
        let mut m = 0.0f32;
        let mut c = glam::Vec3::ZERO;
        for p in &self.parts {
            let vol: f32 = p.colliders.iter().map(|k| k.shape.volume()).sum::<f32>().max(1e-9);
            for k in &p.colliders {
                c += glam::Vec3::from_array(k.position) * (p.mass * k.shape.volume() / vol);
            }
            m += p.mass;
        }
        for w in &self.wheels {
            let dir = w.suspension_dir.map(glam::Vec3::from_array).unwrap_or(-up);
            c += (glam::Vec3::from_array(w.position) + dir * w.suspension_max) * w.mass;
            m += w.mass;
        }
        let com = if m > 0.0 { c / m } else { glam::Vec3::ZERO };
        (com + glam::Vec3::from_array(off)).to_array()
    }

    /// Copy with every position expressed relative to [`Self::center_of_mass`].
    pub fn centered(&self) -> VehicleDesc {
        let o = glam::Vec3::from_array(self.center_of_mass());
        let mut d = self.clone();
        let sh = |p: [f32; 3]| (glam::Vec3::from_array(p) - o).to_array();
        for p in &mut d.parts {
            for k in &mut p.colliders {
                k.position = sh(k.position);
            }
            p.mount.anchor = p.mount.anchor.map(sh);
        }
        for w in &mut d.wheels {
            w.position = sh(w.position);
            w.suspension_force_point = w.suspension_force_point.map(sh);
            w.mount.anchor = w.mount.anchor.map(sh);
        }
        d.aero.center = sh(d.aero.center);
        d.auto_center_of_mass = false;
        d.center_of_mass_offset = [0.0; 3];
        d
    }
}

impl Default for VehicleDesc {
    fn default() -> Self {
        Self {
            name: "vehicle".into(),
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            parts: vec![PartDesc::default()],
            inertia: None,
            auto_center_of_mass: true,
            center_of_mass_offset: [0.0; 3],
            wheels: Vec::new(),
            drive: DriveDesc::default(),
            anti_roll: Vec::new(),
            steering: SteeringDesc::default(),
            response: ResponseDesc::default(),
            aero: AeroDesc::default(),
            assists: AssistDesc::default(),
            max_slope_deg: 80.0,
            max_pitch_roll_deg: 180.0,
            gravity_scale: 1.0,
            can_sleep: true,
            reverse_brake: true,
            spare_wheels: 0,
            wheel_query: WheelQuery::Ray,
            query_interval: 0,
            motorcycle: None,
            tracks: None,
        }
    }
}

/// Per-frame driver input.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VehicleInput {
    /// -1..1 (negative = reverse).
    pub forward: f32,
    /// -1..1 (positive = right).
    pub steer: f32,
    pub brake: f32,
    pub handbrake: f32,
    pub gear: GearRequest,
}

impl Default for VehicleInput {
    fn default() -> Self {
        Self { forward: 0.0, steer: 0.0, brake: 0.0, handbrake: 0.0, gear: GearRequest::Auto }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GearRequest {
    #[default]
    Auto,
    /// Signed gear: 0 neutral, negative reverse (manual transmissions only).
    Gear(i32),
}

/// Functional overrides, ANDed with the state derived from part ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionalState {
    pub wheel_mask: u32,
    pub drive_mask: u32,
    pub steer_mask: u32,
    pub brake_mask: u32,
    pub engine_connected: bool,
}

impl Default for FunctionalState {
    fn default() -> Self {
        Self { wheel_mask: !0, drive_mask: !0, steer_mask: !0, brake_mask: !0, engine_connected: true }
    }
}
