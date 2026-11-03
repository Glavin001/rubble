//! Explicit beams: Rubble's fixed-step node–beam solver (CPU reference).
//!
//! A soft-body vehicle is nodes (point masses) joined by beams (spring–dampers that
//! yield, harden, weaken and break), advanced with small fixed explicit steps of
//! [`STEP`] seconds. Every step runs three passes in order: nodes (ground and box
//! contact, integration, force restart with gravity and drag), beams (forces,
//! plasticity, breaking, detacher groups) and self-collision (contacters against the
//! vehicle's own collision triangles). Forces accumulate across the passes and across
//! the step boundary: the node pass consumes what the previous step's beam and
//! self-collision passes produced.
//!
//! This is the CPU reference, in single precision, written from
//! `engine/docs/explicit-beams.md`; `engine_webgpu::ExplicitBeamsGpu`
//! (`slang/kernels/explicit_beams.slang`) runs the same rules with every vehicle on
//! the GPU. The behaviours that reproduce the established simulator's results (its
//! fast approximations, its yield force, its break exception) form the compatible
//! profile; [`ExplicitBeamsSettings`] switches the optional ones.

use crate::desc::{BeamKind, SoftVehicleDesc};
use crate::explicit_aero::{
    airfoil_from_env, chain_wings, Airfoil, CenteringCommand, FuseDrag, Wing, WingSide, BLD, BLU,
    BRD, BRU, FLD, FLU, FRD, FRU,
};
use crate::explicit_triggers::{segment_point, KeyCommand, RailSegment, SlideNode, Trigger, KEYS};
use crate::truck::{load_truck_config, parse_truck_config};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;

/// The fixed step (s).
pub const STEP: f32 = 0.0005;
/// Gravity along y (m/s²).
pub const GRAVITY: f32 = -9.807;
/// Air drag: F = −coefficient·|v|·v.
pub const DRAG: f32 = 0.05;
/// A beam yielding in compression keeps at least this rest length (m).
pub const MIN_REST: f32 = 0.1;
/// Supports break past this many rest lengths of extension unless the file says.
pub const SUPPORT_BREAK_FACTOR: f32 = 4.0;
/// Node 0 farther than this from the origin moves the origin to it (m).
pub const REBASE_RADIUS: f32 = 100.0;
/// Turbulence scale on the drag force.
pub const TURBULENCE: f32 = 0.005;
/// Collision range when the file gives none (m).
pub const DEFAULT_COLLISION_RANGE: f32 = 0.02;
/// Collision triangles a vehicle may have (a file with more fails to spawn).
pub const MAX_COLLISION_TRIANGLES: usize = 3000;

/// Node flags ([`Node::flags`]).
pub const NODE_NO_GROUND: u32 = 1;
pub const NODE_IMMOVABLE: u32 = 2;
pub const NODE_ON_TRIANGLE: u32 = 4;
pub const NODE_CONTACTER: u32 = 8;
pub const NODE_TYRE: u32 = 16;

/// Optional parts of the compatible profile. All on by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplicitBeamsSettings {
    /// Beam lengths through the fast inverse square root (off: exact).
    pub fast_inv_sqrt: bool,
    /// Air drag on every node.
    pub drag: bool,
    /// Pseudo-random turbulence on top of the drag (needs `drag`).
    pub turbulence: bool,
}

impl Default for ExplicitBeamsSettings {
    fn default() -> Self {
        Self {
            fast_inv_sqrt: true,
            drag: true,
            turbulence: true,
        }
    }
}

/// A contact surface's friction model: adhesion below `va`, static μ `ms`,
/// sliding μ `mc`, hydrodynamic friction `t2`, Stribeck speed `vs` and exponent
/// `alpha`, and a strength multiplier on the reaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactSurface {
    pub va: f32,
    pub ms: f32,
    pub mc: f32,
    pub t2: f32,
    pub vs: f32,
    pub alpha: f32,
    pub strength: f32,
}

impl ContactSurface {
    /// The ground ("gravel").
    pub const GROUND: Self = Self {
        va: 3.0,
        ms: 0.85,
        mc: 0.6,
        t2: 0.006,
        vs: 3.0,
        alpha: 2.0,
        strength: 1.0,
    };
    /// Static boxes and self-collision ("concrete").
    pub const BOX: Self = Self {
        va: 3.0,
        ms: 1.2,
        mc: 0.75,
        t2: 0.01,
        vs: 6.0,
        alpha: 2.0,
        strength: 1.0,
    };
}

/// How a beam's stiffness depends on its extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplicitKind {
    Plain,
    /// Spring–damper that moves toward a stiff limit spring past its bounds (shocks,
    /// tyre spokes).
    LimitedSpring,
    /// No stiffness when stretched; breaks past a long extension.
    Support,
    /// No stiffness when compressed.
    Rope,
    /// A `shocks2` progressive shock: its rates in [`ExplicitBeams::shocks2`] at
    /// [`Beam::shock`] ([`shock2_force`]).
    Shock2,
    /// A trigger: no stiffness or damping (zero force); its keys in
    /// [`ExplicitBeams::triggers`] at [`Beam::shock`], its bounds the short and long
    /// bounds (`explicit_triggers`).
    Trigger,
}

impl ExplicitKind {
    pub fn code(self) -> u32 {
        match self {
            ExplicitKind::Plain => 0,
            ExplicitKind::LimitedSpring => 1,
            ExplicitKind::Support => 2,
            ExplicitKind::Rope => 3,
            ExplicitKind::Shock2 => 4,
            ExplicitKind::Trigger => 5,
        }
    }
}

/// Whether a beam may yield.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplicitType {
    /// May yield (unless it is a limited spring).
    Deformable,
    /// Shocks and hydros: never yield.
    Actuator,
    /// Tyre rigidity beams: never yield.
    Virtual,
}

impl ExplicitType {
    pub fn code(self) -> u32 {
        match self {
            ExplicitType::Deformable => 0,
            ExplicitType::Actuator => 1,
            ExplicitType::Virtual => 2,
        }
    }
}

/// A point mass. `pos` is relative to the vehicle's origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub pos: Vec3,
    pub vel: Vec3,
    /// Accumulated force, consumed by the next node pass.
    pub force: Vec3,
    pub mass: f32,
    pub friction: f32,
    /// `NODE_*` bits.
    pub flags: u32,
}

impl Node {
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

/// A beam between nodes `a` and `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beam {
    pub a: u32,
    pub b: u32,
    /// Stiffness (N/m) and damping (N·s/m).
    pub k: f32,
    pub c: f32,
    /// Rest length (changes as the beam yields) and spawn rest length (m).
    pub rest: f32,
    pub rest0: f32,
    /// Compression and tension yield thresholds (N, both positive).
    pub tc: f32,
    pub tt: f32,
    /// Force above which the yield and break rules run.
    pub gate: f32,
    /// Break strength (N).
    pub strength: f32,
    /// Plastic coefficient 0..1.
    pub plastic: f32,
    pub kind: ExplicitKind,
    pub ty: ExplicitType,
    /// Limited springs: limit stiffness and damping, short and long bounds
    /// (fractions of the rest length). Supports: `long_bound` is the break factor.
    pub limit_k: f32,
    pub limit_c: f32,
    pub short_bound: f32,
    pub long_bound: f32,
    /// Detacher group: 0 none, > 0 a group master, < 0 a follower of group −g.
    pub group: i32,
    pub broken: bool,
    /// Last force (N, positive = compression).
    pub force: f32,
    /// [`ExplicitKind::Shock2`]: its record in [`ExplicitBeams::shocks2`];
    /// [`ExplicitKind::Trigger`]: its record in [`ExplicitBeams::triggers`].
    pub shock: u32,
}

impl Beam {
    /// A beam of `kind` and `ty` with its thresholds set from `deform` and
    /// `strength` as spawning does (deformable: Tc = Tt = G = deform; otherwise
    /// no yield and G = S).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: u32,
        b: u32,
        kind: ExplicitKind,
        ty: ExplicitType,
        k: f32,
        c: f32,
        rest: f32,
        deform: f32,
        strength: f32,
        plastic: f32,
    ) -> Self {
        let deformable = ty == ExplicitType::Deformable;
        let (tc, gate) = if deformable {
            (deform, deform)
        } else {
            (f32::INFINITY, strength)
        };
        Self {
            a,
            b,
            k,
            c,
            rest,
            rest0: rest,
            tc,
            tt: tc,
            gate,
            strength,
            plastic,
            kind,
            ty,
            limit_k: 0.0,
            limit_c: 0.0,
            short_bound: 0.0,
            long_bound: 0.0,
            group: 0,
            broken: false,
            force: 0.0,
            shock: 0,
        }
    }

    /// Yielded: alive with a rest length off its spawn length by more than 0.1 %.
    pub fn bent(&self) -> bool {
        !self.broken && (self.rest - self.rest0).abs() > 1e-3 * self.rest0
    }
}

/// A `shocks2` shock's rates (`engine/docs/explicit-beams-wheels2-shocks2-spec.md`
/// §5–6), fixed at spawn. Its beam holds the rest length and the bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shock2 {
    pub k_in: f32,
    pub c_in: f32,
    pub k_out: f32,
    pub c_out: f32,
    /// Progression factors of the spring and damping, in and out.
    pub pk_in: f32,
    pub pc_in: f32,
    pub pk_out: f32,
    pub pc_out: f32,
    /// Bump spring and damping (past the bounds).
    pub k_bump: f32,
    pub c_bump: f32,
    /// Soft bump (option `s`).
    pub soft: bool,
}

/// `min(x, 1)` as the spec writes it: 1 if 1 < x, else x (NaN passes through).
fn min1(x: f32) -> f32 {
    if 1.0 < x {
        1.0
    } else {
        x
    }
}

/// `max(a, b)`: b if a < b, else a.
fn max_ab(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

impl Shock2 {
    /// Rates with progression: `k0 + (p_k·k0)·λ`, `c0 + (p_c·c0)·λ`, with
    /// λ = 1 for a zero bound, else min((e / (bound·L))², 1).
    fn progressive(k0: f32, c0: f32, pk: f32, pc: f32, e: f32, bound: f32, rest: f32) -> (f32, f32) {
        let lambda = if bound == 0.0 {
            1.0
        } else {
            let r = e / (bound * rest);
            min1(r * r)
        };
        (k0 + (pk * k0) * lambda, c0 + (pc * c0) * lambda)
    }

    fn out_rates(&self, e: f32, lb: f32, rest: f32) -> (f32, f32) {
        Self::progressive(self.k_out, self.c_out, self.pk_out, self.pc_out, e, lb, rest)
    }

    fn in_rates(&self, e: f32, sb: f32, rest: f32) -> (f32, f32) {
        Self::progressive(self.k_in, self.c_in, self.pk_in, self.pc_in, e, sb, rest)
    }

    /// Stiffness and damping at extension `e` (positive extended) and relative
    /// speed `u` (positive separating), with rest length `rest` and bounds `sb`,
    /// `lb` (fractions of it): spec §6.1–6.3.
    pub fn rates(&self, e: f32, u: f32, rest: f32, sb: f32, lb: f32) -> (f32, f32) {
        // §6.1: the velocity picks the side.
        let (mut k, mut c) = if u > 0.0 {
            self.out_rates(e, lb, rest)
        } else {
            self.in_rates(e, sb, rest)
        };
        let past = e > lb * rest || e < (-sb) * rest;
        if !self.soft {
            // §6.2 hard bump: a replacement.
            if past {
                k = self.k_bump;
                c = self.c_bump;
            }
            return (k, c);
        }
        // §6.3 soft bump.
        let soft_rest = rest * 0.8;
        let el = lb * soft_rest;
        let es = (-sb) * soft_rest;
        let bump = |k: f32, c: f32, bound: f32, q_num: f32| {
            let mu = if bound == 0.0 {
                1.0
            } else {
                let q = (q_num * 5.0) / (bound * rest);
                min1(q * q)
            };
            (
                k + ((k + 100.0) * self.pk_out) * mu,
                c + ((c + 100.0) * self.pc_out) * mu,
            )
        };
        if e > el {
            let (k0, c0) = self.out_rates(e, lb, rest);
            (k, c) = bump(k0, c0, lb, e - el);
            if u < 0.0 {
                k = self.k_in;
                c = self.c_in;
            }
        } else if e < es {
            let (k0, c0) = self.in_rates(e, sb, rest);
            // The out factors here too (quirk).
            (k, c) = bump(k0, c0, sb, e - es);
            if u > 0.0 {
                k = self.k_out;
                c = self.c_out;
            }
        }
        if past {
            k = max_ab(k, self.k_bump);
            c = max_ab(c, self.c_bump);
        }
        (k, c)
    }
}

/// A collision triangle (nodes in file order) and its test throttle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Triangle {
    pub nodes: [u32; 3],
    pub wait: i32,
    pub idle: i32,
}

/// An axis-aligned static box (world coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticBox {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl StaticBox {
    /// The box with centre `c` and half extents `half`.
    pub fn centered(c: Vec3, half: Vec3) -> Self {
        Self {
            lo: c - half,
            hi: c + half,
        }
    }
}

/// Steps per 60 fps frame with the remainder carried: 1/60 s is 100 units of
/// 1/6000 s and a step is 3, so frames run 33, 33, 34, … steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameClock {
    carry: u32,
}

impl FrameClock {
    /// Steps of the next frame.
    pub fn next(&mut self) -> u32 {
        let total = self.carry + 100;
        self.carry = total % 3;
        total / 3
    }
}

/// One vehicle.
#[derive(Clone, Debug)]
pub struct ExplicitBeams {
    pub nodes: Vec<Node>,
    pub beams: Vec<Beam>,
    pub triangles: Vec<Triangle>,
    pub boxes: Vec<StaticBox>,
    /// World position the node positions are relative to.
    pub origin: Vec3,
    /// Flat ground height (m).
    pub ground: f32,
    /// Self-collision range r (m).
    pub collision_range: f32,
    /// Soft tyres built from a `.truck` file's `wheels` (for [`Self::roll_tyres`]).
    pub tyres: Vec<Tyre>,
    /// Self-collision contacts in the last step (a statistic; not simulated state).
    pub last_contacts: u32,
    pub settings: ExplicitBeamsSettings,
    /// Turbulence noise state (starts at 1).
    pub noise: i32,
    /// Per node, the beams it belongs to.
    pub incidence: Vec<Vec<u32>>,
    /// Contacters that are not tyre nodes, in node order.
    pub candidates: Vec<u32>,
    /// Self-collision grid cell size (0: test every candidate); see [`Self::finish`].
    pub grid_cell: f32,
    /// The grid's per-step scratch: bucket counts, offsets, sorted candidates, and
    /// one triangle's candidates.
    grid: CollisionGrid,
    /// Steps run so far.
    pub steps: u64,
    pub clock: FrameClock,
    /// Wing panels in file order ([`crate::explicit_aero`]).
    pub wings: Vec<Wing>,
    /// Airfoil tables the wings and the fuselage drag use, one per file name.
    pub airfoils: Vec<Arc<Airfoil>>,
    /// The `fusedrag` line: with one, the node pass adds [`Self::fuse_force`]
    /// instead of the air drag and turbulence.
    pub fusedrag: Option<FuseDrag>,
    /// The fuselage drag on every node, from the previous step (0 at spawn).
    pub fuse_force: Vec3,
    /// Self-centering command beams (the only commands that move with no input).
    pub centering: Vec<CenteringCommand>,
    /// `shocks2` rates, by [`Beam::shock`].
    pub shocks2: Vec<Shock2>,
    /// Triggers, by [`Beam::shock`] of their beams; the keys they set in the last
    /// beam pass (1.0 held), read and cleared by the next command pass.
    pub triggers: Vec<Trigger>,
    pub key_input: [f32; KEYS],
    /// With triggers, every command line (self-centering ones too), moved by its
    /// keys; [`Self::centering`] is then empty.
    pub commands: Vec<KeyCommand>,
    /// Slide nodes in spawn order and their rails' segments.
    pub slides: Vec<SlideNode>,
    pub segments: Vec<RailSegment>,
    /// Spawn notes (airfoil files not found: their tables are zero).
    pub warnings: Vec<String>,
    /// Vehicle contact ([`crate::explicit_contact`]): the nodes other vehicles'
    /// triangles test (contacters, then contactable nodes, merged in node order).
    pub contact_points: Vec<u32>,
    /// Vehicle contact's side vote: node i's spawn-time beam neighbours are
    /// `neighbours[neighbour_start[i]..neighbour_start[i + 1]]`, in beam order.
    pub neighbour_start: Vec<u32>,
    pub neighbours: Vec<u32>,
    /// Per collision triangle its vehicle-contact throttle (cwait, cidle).
    pub contact_throttle: Vec<[i32; 2]>,
}

// --- numerical primitives (compatible profile) ---

/// The fast inverse square root: magic-constant estimate, one Newton iteration.
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let i = 0x5f37_59df_i32.wrapping_sub((x.to_bits() as i32) >> 1);
    let y = f32::from_bits(i as u32);
    y * (1.5 - 0.5 * x * y * y)
}

/// Bit-trick square root: halves the exponent.
pub fn approx_sqrt(x: f32) -> f32 {
    let i = ((x.to_bits() as i32).wrapping_sub(0x3F80_0000) >> 1).wrapping_add(0x3F80_0000);
    f32::from_bits(i as u32)
}

/// Schraudolph's exponential: 0 below −15, 1e38 above 88.
pub fn approx_exp(x: f32) -> f32 {
    if x < -15.0 {
        return 0.0;
    }
    if x > 88.0 {
        return 1e38;
    }
    let i = (12_102_203.0f32 * x + 1_064_652_319.0f32) as i32;
    f32::from_bits(i as u32)
}

/// Bit-trick power xʸ.
pub fn approx_pow(x: f32, y: f32) -> f32 {
    let e = (x.to_bits() as i32).wrapping_sub(1_065_353_216) as f32;
    let i = (y * e + 1_065_353_216.0f32) as i32;
    f32::from_bits(i as u32)
}

/// One turbulence draw: advance the state, return a value in [−1, 1).
pub fn noise_draw(state: &mut i32) -> f32 {
    *state = state.wrapping_mul(16807);
    f32::from_bits(0x4000_0000 | (*state as u32 & 0x007F_FFFF)) - 3.0
}

/// A node offset turned by `yaw_degrees` about +y, in single precision:
/// (c·x + s·z, y, −s·x + c·z) with c, s the cosine and sine of the angle.
pub fn yaw_offset(p: Vec3, yaw_degrees: f32) -> Vec3 {
    let (s, c) = yaw_degrees.to_radians().sin_cos();
    Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z)
}

/// The contact force on a node with force `f` and velocity `v` (relative to the
/// surface), mass `m`, against unit normal `n` with penetration `p`: cancels the
/// normal force, removes 80 % of an approaching velocity (plus a penetration push),
/// and applies static or sliding friction with the node's coefficient `friction`.
#[allow(clippy::too_many_arguments)]
pub fn contact_force(
    f: Vec3,
    v: Vec3,
    m: f32,
    n: Vec3,
    h: f32,
    surface: &ContactSurface,
    p: f32,
    friction: f32,
) -> Vec3 {
    if p < 0.0 {
        return Vec3::ZERO;
    }
    let vn = v.dot(n);
    let fn_ = f.dot(n);
    let mut r = -fn_;
    if vn < 0.0 {
        r -= (0.8 * vn - 0.2 * p / h) * m / h;
    }
    if r <= 0.0 {
        return Vec3::ZERO;
    }
    let sf = f - fn_ * n;
    let mut s = v - vn * n;
    let sigma = s.length();
    if sigma > 0.0 {
        s /= sigma;
    }
    let gr = r * surface.strength * friction;
    let ms = surface.ms * gr;
    if sigma < surface.va && gr > 0.0 && sf.length_squared() <= ms * ms {
        r * n - ms * (1.0 - approx_exp(-sigma / surface.va)) * s - sf
    } else {
        let g = surface.mc
            + (surface.ms - surface.mc)
                * approx_exp(-approx_pow(sigma / surface.vs, surface.alpha));
        r * n - (g + (surface.t2 * sigma).min(5.0)) * gr * s
    }
}

/// Beam type and kind for a desc beam.
fn classify(kind: BeamKind) -> (ExplicitKind, ExplicitType) {
    match kind {
        BeamKind::Normal | BeamKind::Anisotropic => (ExplicitKind::Plain, ExplicitType::Deformable),
        BeamKind::Support => (ExplicitKind::Support, ExplicitType::Deformable),
        BeamKind::Rope => (ExplicitKind::Rope, ExplicitType::Deformable),
        BeamKind::Bounded => (ExplicitKind::LimitedSpring, ExplicitType::Actuator),
        BeamKind::Hydro => (ExplicitKind::Plain, ExplicitType::Actuator),
    }
}

/// Spawn order class: plain/support/rope, then shocks, then hydros.
fn order_class(kind: BeamKind) -> u32 {
    match kind {
        BeamKind::Bounded => 1,
        BeamKind::Hydro => 2,
        _ => 0,
    }
}

impl ExplicitBeams {
    /// An empty vehicle (add nodes and beams, then [`Self::finish`]).
    pub fn empty(settings: ExplicitBeamsSettings) -> Self {
        Self {
            nodes: Vec::new(),
            beams: Vec::new(),
            triangles: Vec::new(),
            boxes: Vec::new(),
            origin: Vec3::ZERO,
            ground: 0.0,
            collision_range: DEFAULT_COLLISION_RANGE,
            last_contacts: 0,
            tyres: Vec::new(),
            settings,
            noise: 1,
            incidence: Vec::new(),
            candidates: Vec::new(),
            grid_cell: 0.0,
            grid: CollisionGrid::default(),
            steps: 0,
            clock: FrameClock::default(),
            wings: Vec::new(),
            airfoils: Vec::new(),
            fusedrag: None,
            fuse_force: Vec3::ZERO,
            centering: Vec::new(),
            shocks2: Vec::new(),
            triggers: Vec::new(),
            key_input: [0.0; KEYS],
            commands: Vec::new(),
            slides: Vec::new(),
            segments: Vec::new(),
            warnings: Vec::new(),
            contact_points: Vec::new(),
            neighbour_start: Vec::new(),
            neighbours: Vec::new(),
            contact_throttle: Vec::new(),
        }
    }

    /// Spawn a `.truck` file with its node offsets at `position`, every node moving
    /// at `velocity`, its tyres built from the `wheels` section (other wheel
    /// sections are an error). Airfoil tables (`wings`, `fusedrag`) are read from
    /// the directory in `RUBBLE_AIRFOIL_DIR` ([`crate::explicit_aero::airfoil_from_env`]).
    pub fn from_truck(
        text: &str,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<Self, String> {
        Self::from_truck_with_airfoils(text, position, velocity, settings, &airfoil_from_env)
    }

    /// [`Self::from_truck`] with a section configuration: `None` the file's default
    /// (the first `section` name in byte order), `Some("")` the root alone
    /// ([`crate::truck::parse_truck_config`]).
    pub fn from_truck_config(
        text: &str,
        config: Option<&str>,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<Self, String> {
        Self::from_truck_config_with_airfoils(
            text,
            config,
            position,
            velocity,
            settings,
            &airfoil_from_env,
        )
    }

    /// [`Self::from_truck`] with the airfoil files' text from `airfoil` (by file
    /// name; `None`: not found, all-zero tables).
    pub fn from_truck_with_airfoils(
        text: &str,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
        airfoil: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        Self::from_truck_config_with_airfoils(text, None, position, velocity, settings, airfoil)
    }

    /// [`Self::from_truck_config`] with the airfoil files' text from `airfoil`.
    pub fn from_truck_config_with_airfoils(
        text: &str,
        config: Option<&str>,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
        airfoil: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        Self::from_truck_yawed(text, config, position, 0.0, velocity, settings, airfoil)
    }

    /// [`Self::from_truck_config`] turned by `yaw_degrees` about +y: every node
    /// offset is rotated ([`yaw_offset`]) before it is added to `position`, so the
    /// tyres, rest lengths and everything else derive from the turned spawn.
    /// `velocity` is in world axes. A yaw of 0 is exactly
    /// [`Self::from_truck_config`]. Airfoils from `RUBBLE_AIRFOIL_DIR`.
    pub fn from_truck_config_yaw(
        text: &str,
        config: Option<&str>,
        position: Vec3,
        yaw_degrees: f32,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<Self, String> {
        Self::from_truck_yawed(
            text,
            config,
            position,
            yaw_degrees,
            velocity,
            settings,
            &airfoil_from_env,
        )
    }

    /// [`Self::from_truck_config_yaw`] with the airfoil files' text from `airfoil`.
    pub fn from_truck_yawed(
        text: &str,
        config: Option<&str>,
        position: Vec3,
        yaw_degrees: f32,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
        airfoil: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        let file = parse_truck_config(text, config);
        if !file.errors.is_empty() {
            return Err(file.errors.join("; "));
        }
        // Collision triangles take the concrete surface; a named one is not
        // supported yet (refused rather than run on concrete).
        if let Some(g) = file.submesh_groundmodel.as_deref().filter(|g| *g != "concrete") {
            return Err(format!(
                "`submesh_groundmodel {g}`: collision triangles on a surface other than \
                 concrete are not supported"
            ));
        }
        let collision_tris = file.triangles.iter().filter(|t| t.collision).count();
        if collision_tris > MAX_COLLISION_TRIANGLES {
            return Err(format!(
                "{collision_tris} collision triangles: at most {MAX_COLLISION_TRIANGLES} per vehicle"
            ));
        }
        if let Some(w) = file
            .wheels
            .iter()
            .find(|w| !matches!(w.kind, "wheels" | "meshwheels" | "meshwheels2"))
        {
            return Err(format!(
                "unsupported section `{}` (the explicit solver builds tyres from `wheels`, \
                 `meshwheels` and `meshwheels2` only)",
                w.kind
            ));
        }
        let desc = load_truck_config(text, config);
        let cine_beams = file.cinecams.len() * 8;
        let mut s = Self::spawn_nodes(&desc, position, yaw_degrees, velocity, settings);
        // Contacters exactly as listed in the file.
        for n in &mut s.nodes {
            n.flags &= !NODE_CONTACTER;
        }
        for &c in &file.contacters {
            s.nodes[c as usize].flags |= NODE_CONTACTER;
        }
        // Beam order: cinecam beams, tyre beams, then the file's beams, shocks, hydros.
        for b in &desc.beams[..cine_beams.min(desc.beams.len())] {
            s.push_desc_beam(b);
        }
        // Wheels section by section (`wheels`, then `meshwheels`, then
        // `meshwheels2`), each in file order; `tyre_of` maps a file wheel to its tyre.
        let mut tyre_of = vec![0u32; file.wheels.len()];
        let mut spawn_order = Vec::with_capacity(file.wheels.len());
        for kind in ["wheels", "meshwheels", "meshwheels2"] {
            for (i, w) in file.wheels.iter().enumerate().filter(|(_, w)| w.kind == kind) {
                tyre_of[i] = spawn_order.len() as u32;
                spawn_order.push(w);
            }
        }
        // A turned spawn builds its tyres in the file's axes, then turns them like
        // every other node.
        let yawed = yaw_degrees != 0.0;
        let local: Vec<Vec3> = if yawed {
            desc.nodes.iter().map(|n| Vec3::from_array(n.position)).collect()
        } else {
            Vec::new()
        };
        for w in spawn_order {
            let at = |s: &Self, i: u32| if yawed { local[i as usize] } else { s.nodes[i as usize].pos };
            let place = |p: Vec3| if yawed { position + yaw_offset(p, yaw_degrees) } else { p };
            // meshwheels2: the axle-to-own-ring spokes harden only past 0.15 L of
            // stretch.
            let spoke_long = if w.kind == "meshwheels2" { 0.15f32 } else { 0.0 };
            // The axle runs toward +z: the node of lower z is A1, whichever the
            // file names first (equal z keeps the file's order).
            let (a1, a2) = if at(&s, w.node1).z > at(&s, w.node2).z {
                (w.node2, w.node1)
            } else {
                (w.node1, w.node2)
            };
            let pa1 = at(&s, a1);
            let pa2 = at(&s, a2);
            let n = w.rays.max(1);
            let axis = (pa2 - pa1).normalize();
            let mut spoke = axis.cross(Vec3::X);
            if spoke.length_squared() < 1e-12 {
                spoke = axis.cross(Vec3::Y);
            }
            let mut spoke = spoke.normalize() * w.radius;
            let theta = -360.0f32 / (2 * n) as f32;
            let half = (theta * 0.5).to_radians();
            let qw = half.cos();
            let qv = half.sin() * axis;
            let rotate = |u: Vec3| u + 2.0 * qw * qv.cross(u) + 2.0 * qv.cross(qv.cross(u));
            let first = s.nodes.len() as u32;
            s.tyres.push(Tyre {
                axle: [a1, a2],
                first,
                count: 2 * n,
                radius: w.radius,
            });
            let tyre = |pos: Vec3| Node {
                pos,
                vel: velocity,
                force: Vec3::ZERO,
                mass: w.mass / (2 * n) as f32,
                friction: w.node_friction,
                flags: NODE_CONTACTER | NODE_TYRE,
            };
            for _ in 0..n {
                s.nodes.push(tyre(place(pa1 + spoke)));
                spoke = rotate(spoke);
                s.nodes.push(tyre(place(pa2 + spoke)));
                spoke = rotate(spoke);
            }
            let outer = |i: u32| first + 2 * (i % n);
            let inner = |i: u32| first + 2 * (i % n) + 1;
            let group = signed_group(&w.beam);
            let deform = w.beam.deform;
            let strength = w.beam.strength;
            let plastic = w.beam.plastic_coef;
            let rigidity = w.rigidity.map(|p| {
                let pp = at(&s, p);
                (p, pp.distance_squared(pa1) < pp.distance_squared(pa2))
            });
            for i in 0..n {
                let add = |s: &mut Self, a: u32, b: u32, kind: ExplicitKind, ty: ExplicitType, rim: bool| {
                    let rest = s.nodes[a as usize].pos.distance(s.nodes[b as usize].pos);
                    let (k, c) = if rim {
                        (w.rim_spring, w.rim_damp)
                    } else {
                        (w.spring, w.damp)
                    };
                    let mut beam = Beam::new(a, b, kind, ty, k, c, rest, deform, strength, plastic);
                    beam.group = group;
                    if kind == ExplicitKind::LimitedSpring {
                        beam.short_bound = 0.66;
                        beam.long_bound = spoke_long;
                        beam.limit_k = 9_000_000.0;
                        beam.limit_c = 12_000.0;
                    }
                    s.beams.push(beam);
                };
                let (o, ii) = (outer(i), inner(i));
                let lim = ExplicitKind::LimitedSpring;
                let plain = ExplicitKind::Plain;
                let def = ExplicitType::Deformable;
                add(&mut s, a1, o, lim, def, false);
                add(&mut s, a2, ii, lim, def, false);
                add(&mut s, a2, o, plain, def, false);
                add(&mut s, a1, ii, plain, def, false);
                add(&mut s, o, ii, plain, def, true);
                add(&mut s, o, outer(i + 1), plain, def, true);
                add(&mut s, ii, inner(i + 1), plain, def, true);
                add(&mut s, ii, outer(i + 1), plain, def, true);
                if let Some((p, near_a1)) = rigidity {
                    let end = if near_a1 { o } else { ii };
                    add(&mut s, p, end, plain, ExplicitType::Virtual, false);
                }
            }
        }
        // Then the file's beams, shocks, shocks2, shocks3, commands and hydros, each
        // section in file order. Commands and hydros gate their break test on the
        // deform threshold at their line.
        let rest: Vec<_> = desc.beams[cine_beams.min(desc.beams.len())..].to_vec();
        let gates: HashMap<usize, f32> = file.actuator_gates.iter().copied().collect();
        let commands: HashMap<usize, &crate::truck::TruckCommand> =
            file.commands.iter().map(|c| (c.beam, c)).collect();
        let shocks2: HashMap<usize, &crate::truck::TruckShock2> =
            file.shocks2.iter().map(|c| (c.beam, c)).collect();
        let shocks3: std::collections::HashSet<usize> = file.shocks3.iter().copied().collect();
        let triggers: HashMap<usize, &crate::truck::TruckTrigger> =
            file.triggers.iter().map(|t| (t.beam, t)).collect();
        let animators: std::collections::HashSet<usize> = file.animators.iter().copied().collect();
        // With triggers, every command line moves by its keys (spec §2.6).
        let keyed = !file.triggers.is_empty();
        if keyed {
            let held: std::collections::HashSet<i32> = file
                .triggers
                .iter()
                .flat_map(|t| [t.short_key as i32, t.long_key as i32])
                .collect();
            for c in &file.commands {
                if (held.contains(&c.contract_key) || held.contains(&c.extend_key))
                    && (c.one_press || !c.no_inertia)
                {
                    return Err(format!(
                        "a command on keys {}/{} that a trigger holds has key inertia or a \
                         one-press option (`o`, `p`): not supported",
                        c.contract_key, c.extend_key
                    ));
                }
            }
        }
        let order = |i: usize| {
            if triggers.contains_key(&i) {
                6
            } else if animators.contains(&i) {
                7
            } else if commands.contains_key(&i) {
                4
            } else if shocks2.contains_key(&i) {
                2
            } else if shocks3.contains(&i) {
                3
            } else {
                match order_class(rest[i].kind) {
                    2 => 5,
                    c => c,
                }
            }
        };
        // `beams` lines on generated tyre nodes, at their place among the file's.
        let tyre_node = |s: &Self, r: crate::truck::TruckNodeRef| match r {
            crate::truck::TruckNodeRef::File(n) => n,
            crate::truck::TruckNodeRef::Tyre(w, k) => s.tyres[tyre_of[w as usize] as usize].first + k,
        };
        let mut tyre_beams = file.tyre_beams.iter().peekable();
        for class in 0..8 {
            for (i, b) in rest.iter().enumerate().filter(|&(i, _)| order(i) == class) {
                while class == 0 && tyre_beams.peek().is_some_and(|t| t.position <= i) {
                    let t = tyre_beams.next().unwrap();
                    let (a, b) = (tyre_node(&s, t.a), tyre_node(&s, t.b));
                    s.push_desc_beam(&crate::desc::SoftBeam { a, b, ..t.beam.clone() });
                }
                let at = s.beams.len() as u32;
                s.push_desc_beam(b);
                if let Some(&g) = gates.get(&i) {
                    s.beams[at as usize].gate = g;
                }
                if let Some(sh) = shocks2.get(&i) {
                    s.make_shock2(at as usize, sh);
                }
                if let Some(t) = triggers.get(&i) {
                    let beam = &mut s.beams[at as usize];
                    beam.kind = ExplicitKind::Trigger;
                    beam.short_bound = t.short_bound;
                    beam.long_bound = t.long_bound;
                    beam.shock = s.triggers.len() as u32;
                    s.triggers.push(Trigger {
                        beam: at,
                        short_key: t.short_key,
                        long_key: t.long_key,
                        enabled: true,
                    });
                }
                if let Some(c) = commands.get(&i) {
                    if c.rope {
                        s.beams[at as usize].kind = ExplicitKind::Rope;
                    }
                    if keyed {
                        s.commands.push(KeyCommand::new(
                            at,
                            c.shorten_rate,
                            c.lengthen_rate,
                            c.max_contraction,
                            c.max_extension,
                            c.contract_key,
                            c.extend_key,
                            c.centering,
                        ));
                    } else if c.centering {
                        s.centering.push(CenteringCommand::new(
                            at,
                            c.shorten_rate,
                            c.lengthen_rate,
                            c.max_contraction,
                            c.max_extension,
                            c.contract_key,
                            c.extend_key,
                        ));
                    }
                }
            }
            // Tyre-node beams after the last plain one.
            while class == 0 && tyre_beams.peek().is_some() {
                let t = tyre_beams.next().unwrap();
                let (a, b) = (tyre_node(&s, t.a), tyre_node(&s, t.b));
                s.push_desc_beam(&crate::desc::SoftBeam { a, b, ..t.beam.clone() });
            }
        }
        s.add_slidenodes(&file);
        s.collision_range = desc.collision_range;
        s.add_triangles(&desc);
        s.add_aero(&file, airfoil);
        s.finish();
        Ok(s)
    }

    /// Slide nodes and their rails, after every beam (spec §3.2): each rail pair is
    /// the lowest-index beam on those two nodes, in that beam's direction; a rail
    /// with a missing beam leaves its slide node inert. The current segment starts
    /// as the nearest.
    fn add_slidenodes(&mut self, file: &crate::truck::TruckFile) {
        if file.slidenodes.is_empty() {
            return;
        }
        let mut first: HashMap<(u32, u32), u32> = HashMap::new();
        for (j, b) in self.beams.iter().enumerate() {
            first
                .entry((b.a.min(b.b), b.a.max(b.b)))
                .or_insert(j as u32);
        }
        for sn in &file.slidenodes {
            let beams: Option<Vec<u32>> = sn
                .rail
                .windows(2)
                .map(|p| first.get(&(p[0].min(p[1]), p[0].max(p[1]))).copied())
                .collect();
            let mut slide = SlideNode {
                node: sn.node,
                spring: sn.spring,
                break_force: sn.break_force,
                tolerance: sn.tolerance,
                segment: None,
                broken_away: false,
            };
            let Some(beams) = beams else {
                self.warnings.push(format!(
                    "slide node {}: a rail pair has no beam; the node is inert",
                    sn.node
                ));
                self.slides.push(slide);
                continue;
            };
            let base = self.segments.len() as u32;
            let n = beams.len() as u32;
            let looped = sn.rail.first() == sn.rail.last();
            for (k, &beam) in beams.iter().enumerate() {
                let k = k as u32;
                let prev = if k > 0 {
                    Some(base + k - 1)
                } else if looped {
                    Some(base + n - 1)
                } else {
                    None
                };
                let next = if k + 1 < n {
                    Some(base + k + 1)
                } else if looped {
                    Some(base)
                } else {
                    None
                };
                self.segments.push(RailSegment { beam, prev, next });
            }
            // The nearest segment (ties: the earliest).
            let x = self.origin + self.nodes[sn.node as usize].pos;
            let mut best = (base, f32::INFINITY);
            for seg in base..base + n {
                let d = self.segment_distance(seg, x).1;
                if d < best.1 {
                    best = (seg, d);
                }
            }
            slide.segment = Some(best.0);
            self.slides.push(slide);
        }
    }

    /// The nearest point of segment `seg` to world point `x`, its distance and its
    /// ratio along the segment.
    fn segment_distance(&self, seg: u32, x: Vec3) -> (Vec3, f32, f32) {
        let b = &self.beams[self.segments[seg as usize].beam as usize];
        let p1 = self.origin + self.nodes[b.a as usize].pos;
        let p2 = self.origin + self.nodes[b.b as usize].pos;
        segment_point(p1, p2, x)
    }

    /// Turn beam `j` (spawned from a `shocks2` line's bounded beam) into a shock2
    /// beam (spec §5): bounds through the `m` / `M` options on the
    /// un-precompressed length, rest length L₀·π, gate the deform threshold.
    fn make_shock2(&mut self, j: usize, sh: &crate::truck::TruckShock2) {
        let (a, b) = (self.beams[j].a as usize, self.beams[j].b as usize);
        // Spawned positions are world positions here (the origin is zero at spawn).
        let lw = self.nodes[a].pos.distance(self.nodes[b].pos);
        let (mut sb, mut lb) = (sh.short_bound, sh.long_bound);
        if sh.metric {
            sb /= lw;
            lb /= lw;
        }
        if sh.abs_metric {
            sb = (lw - sb) / lw;
            lb = (lb - lw) / lw;
            if lb < 0.0 {
                lb = 0.0;
            }
            if sb > 1.0 {
                sb = 1.0;
            }
        }
        let rest = lw * sh.precompression;
        let index = self.shocks2.len() as u32;
        self.shocks2.push(Shock2 {
            k_in: sh.spring_in,
            c_in: sh.damp_in,
            k_out: sh.spring_out,
            c_out: sh.damp_out,
            pk_in: sh.prog_spring_in,
            pc_in: sh.prog_damp_in,
            pk_out: sh.prog_spring_out,
            pc_out: sh.prog_damp_out,
            k_bump: sh.bump_spring,
            c_bump: sh.bump_damp,
            soft: sh.soft,
        });
        let group = self.beams[j].group;
        let mut beam = Beam::new(
            a as u32,
            b as u32,
            ExplicitKind::Shock2,
            ExplicitType::Actuator,
            sh.spring_in,
            sh.damp_in,
            rest,
            sh.gate,
            sh.strength,
            0.0,
        );
        beam.gate = sh.gate;
        beam.limit_k = sh.bump_spring;
        beam.limit_c = sh.bump_damp;
        beam.short_bound = sb;
        beam.long_bound = lb;
        beam.group = group;
        beam.shock = index;
        self.beams[j] = beam;
    }

    /// Wings (break references, chains) and the fuselage drag from spawn
    /// positions, with their airfoil tables.
    fn add_aero(&mut self, file: &crate::truck::TruckFile, airfoil: &dyn Fn(&str) -> Option<String>) {
        let table = |s: &mut Self, name: &str| -> u32 {
            if let Some(i) = s.airfoils.iter().position(|a| a.name == name) {
                return i as u32;
            }
            let a = match airfoil(name) {
                Some(text) => Airfoil::parse(name, &text),
                None => {
                    s.warnings.push(format!("airfoil `{name}` not found: its tables are zero"));
                    Airfoil::zero(name)
                }
            };
            s.airfoils.push(Arc::new(a));
            s.airfoils.len() as u32 - 1
        };
        let x: Vec<Vec3> = self.nodes.iter().map(|n| self.origin + n.pos).collect();
        for w in &file.wings {
            let t = table(self, &w.airfoil);
            self.wings.push(Wing::new(w.nodes, w.efficacy, w.chord_point, w.control, t, &x));
        }
        chain_wings(&mut self.wings, &x);
        if let Some(fd) = &file.fusedrag {
            let t = table(self, &fd.airfoil);
            self.fusedrag = Some(FuseDrag {
                front: fd.front,
                width: fd.width,
                airfoil: t,
            });
        }
        self.check_wings();
    }

    /// The wing break check on the current world positions (once per frame, before
    /// the origin rebase, and at spawn).
    pub fn check_wings(&mut self) {
        let origin = self.origin;
        let nodes = &self.nodes;
        for w in &mut self.wings {
            w.check_break(|i| origin + nodes[i as usize].pos);
        }
    }

    /// Spawn a BeamNG JBeam vehicle (`files`: name and text of each `.jbeam`,
    /// `config` an optional `.pc`) through its `.truck` form
    /// ([`crate::truck_export::desc_to_truck`]), the same text Rigs of Rods runs:
    /// pressure wheels become soft tyres. Also returns that text and the
    /// conversion's warnings.
    pub fn from_jbeam(
        files: &[(&str, &str)],
        config: Option<&str>,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, String, Vec<String>), String> {
        let desc = crate::jbeam::load_jbeam(files, config)?;
        let (text, warnings) =
            crate::truck_export::desc_to_truck(&desc, crate::truck_export::TyreDefaults::for_vehicle(&desc));
        let sim = Self::from_truck(&text, position, velocity, settings)?;
        Ok((sim, text, warnings))
    }

    /// Spawn a desc (no tyres): its nodes at `position` moving at `velocity`, its
    /// beams in the desc's order, contacters as the desc marks them.
    pub fn new(
        desc: &SoftVehicleDesc,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Self {
        let mut s = Self::spawn_nodes(desc, position, 0.0, velocity, settings);
        for b in &desc.beams {
            s.push_desc_beam(b);
        }
        s.collision_range = if desc.collision_range > 0.0 {
            desc.collision_range
        } else {
            DEFAULT_COLLISION_RANGE
        };
        s.add_triangles(desc);
        s.finish();
        s
    }

    fn spawn_nodes(
        desc: &SoftVehicleDesc,
        position: Vec3,
        yaw_degrees: f32,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Self {
        let mut s = Self::empty(settings);
        for n in &desc.nodes {
            let mut flags = 0;
            if !n.collision {
                flags |= NODE_NO_GROUND;
            }
            if n.fixed {
                flags |= NODE_IMMOVABLE;
            }
            if n.contacter {
                flags |= NODE_CONTACTER;
            }
            let offset = Vec3::from_array(n.position);
            let offset = if yaw_degrees == 0.0 {
                offset
            } else {
                yaw_offset(offset, yaw_degrees)
            };
            s.nodes.push(Node {
                pos: position + offset,
                vel: if n.fixed { Vec3::ZERO } else { velocity },
                force: Vec3::ZERO,
                mass: n.mass,
                friction: n.friction,
                flags,
            });
        }
        s
    }

    fn push_desc_beam(&mut self, b: &crate::desc::SoftBeam) {
        let (kind, ty) = classify(b.kind);
        let mut rest = self.nodes[b.a as usize]
            .pos
            .distance(self.nodes[b.b as usize].pos);
        if b.kind == BeamKind::Bounded {
            rest *= b.precompression;
        }
        let mut beam = Beam::new(
            b.a,
            b.b,
            kind,
            ty,
            b.spring,
            b.damp,
            rest,
            b.deform,
            b.strength,
            b.plastic_coef,
        );
        beam.group = signed_group(b);
        match kind {
            ExplicitKind::LimitedSpring => {
                beam.limit_k = b.deform;
                beam.limit_c = if b.limit_damp == 0.0 {
                    12_000.0
                } else {
                    b.limit_damp
                };
                beam.short_bound = b.short_bound;
                beam.long_bound = b.long_bound;
            }
            ExplicitKind::Support => beam.long_bound = b.long_bound,
            _ => {}
        }
        self.beams.push(beam);
    }

    fn add_triangles(&mut self, desc: &SoftVehicleDesc) {
        for t in desc.triangles.iter().filter(|t| t.collision) {
            for &n in &t.nodes {
                self.nodes[n as usize].flags |= NODE_ON_TRIANGLE;
            }
            self.triangles.push(Triangle {
                nodes: t.nodes,
                wait: 0,
                idle: 0,
            });
        }
    }

    /// Build the per-node beam lists and the self-collision candidates (after
    /// adding nodes, beams or contacter flags by hand).
    pub fn finish(&mut self) {
        self.incidence = vec![Vec::new(); self.nodes.len()];
        for (i, b) in self.beams.iter().enumerate() {
            self.incidence[b.a as usize].push(i as u32);
            self.incidence[b.b as usize].push(i as u32);
        }
        self.candidates = (0..self.nodes.len() as u32)
            .filter(|&i| {
                let n = &self.nodes[i as usize];
                n.has(NODE_CONTACTER) && !n.has(NODE_TYRE)
            })
            .collect();
        self.grid_cell = self.default_grid_cell();
        self.finish_contact();
    }

    /// Vehicle contact's spawn data ([`crate::explicit_contact`], spec §2): the
    /// contact points, the neighbour lists and zeroed contact throttles.
    pub fn finish_contact(&mut self) {
        let no_triangles = self.triangles.is_empty();
        let mut on_triangle = vec![false; self.nodes.len()];
        for t in &self.triangles {
            for &n in &t.nodes {
                on_triangle[n as usize] = true;
            }
        }
        self.contact_points = (0..self.nodes.len() as u32)
            .filter(|&i| {
                let n = &self.nodes[i as usize];
                n.has(NODE_CONTACTER)
                    || (!n.has(NODE_NO_GROUND) && (on_triangle[i as usize] || no_triangles))
            })
            .collect();
        self.neighbour_start = Vec::with_capacity(self.nodes.len() + 1);
        self.neighbours.clear();
        self.neighbour_start.push(0);
        for (i, inc) in self.incidence.iter().enumerate() {
            for &j in inc {
                let b = &self.beams[j as usize];
                self.neighbours.push(if b.a == i as u32 { b.b } else { b.a });
            }
            self.neighbour_start.push(self.neighbours.len() as u32);
        }
        self.contact_throttle = vec![[0, 0]; self.triangles.len()];
    }

    /// The self-collision grid's cell size: the median collision triangle's box at
    /// spawn padded by the contact range, for more than [`GRID_MIN_CANDIDATES`]
    /// candidates (0 otherwise). The grid only skips candidates outside a
    /// triangle's padded box, and the rest are tested in candidate order, so the
    /// results are those of testing every candidate.
    pub fn default_grid_cell(&self) -> f32 {
        if self.candidates.len() <= GRID_MIN_CANDIDATES || self.triangles.is_empty() {
            return 0.0;
        }
        let mut extents: Vec<f32> = self
            .triangles
            .iter()
            .map(|t| {
                let p = t.nodes.map(|n| self.nodes[n as usize].pos);
                (p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2])).max_element()
            })
            .collect();
        extents.sort_by(f32::total_cmp);
        (extents[extents.len() / 2] + 2.0 * self.collision_range).max(0.05)
    }

    /// World positions of the nodes.
    pub fn positions(&self) -> Vec<Vec3> {
        self.nodes.iter().map(|n| self.origin + n.pos).collect()
    }

    /// Live plain beams at `node`.
    pub fn live_plain(&self, node: u32) -> u32 {
        self.incidence[node as usize]
            .iter()
            .filter(|&&j| {
                let b = &self.beams[j as usize];
                !b.broken && b.kind == ExplicitKind::Plain
            })
            .count() as u32
    }

    /// Broken and bent ([`Beam::bent`]) beam counts.
    pub fn damage(&self) -> (usize, usize) {
        (
            self.beams.iter().filter(|b| b.broken).count(),
            self.beams.iter().filter(|b| b.bent()).count(),
        )
    }

    /// If node 0 is farther than [`REBASE_RADIUS`] from the origin, move the origin
    /// to it (node positions keep their world positions).
    pub fn rebase_origin(&mut self) {
        let Some(first) = self.nodes.first() else {
            return;
        };
        let offset = first.pos;
        if offset.length() > REBASE_RADIUS {
            self.origin += offset;
            for n in &mut self.nodes {
                n.pos -= offset;
            }
        }
    }

    /// One 60 fps frame: rebase, then its steps ([`FrameClock`]). Returns the steps.
    pub fn frame(&mut self) -> u32 {
        self.check_wings();
        self.rebase_origin();
        let steps = self.clock.next();
        for _ in 0..steps {
            self.step();
        }
        steps
    }

    /// Simulate `seconds` as 60 fps frames.
    pub fn run(&mut self, seconds: f32) {
        let frames = (seconds * 60.0).round() as u32;
        for _ in 0..frames {
            self.frame();
        }
    }

    /// One fixed step: nodes, wings, fuselage drag, commands, beams,
    /// self-collision.
    pub fn step(&mut self) {
        self.node_pass();
        if !self.wings.is_empty() {
            self.wing_pass();
        }
        if self.fusedrag.is_some() {
            self.fuse_pass();
        }
        if !self.triggers.is_empty() {
            self.key_command_pass();
        } else if !self.centering.is_empty() {
            self.command_pass();
        }
        self.beam_pass();
        if !self.triangles.is_empty() && !self.candidates.is_empty() {
            self.collision_pass();
        }
        if !self.slides.is_empty() {
            self.slide_pass();
        }
        self.steps += 1;
    }

    fn node_pass(&mut self) {
        let h = STEP;
        let origin = self.origin;
        let ground = self.ground;
        let settings = self.settings;
        for i in 0..self.nodes.len() {
            let mut n = self.nodes[i];
            if !n.has(NODE_NO_GROUND) {
                let x = origin + n.pos;
                if ground > x.y {
                    n.force += contact_force(
                        n.force,
                        n.vel,
                        n.mass,
                        Vec3::Y,
                        h,
                        &ContactSurface::GROUND,
                        ground - x.y,
                        n.friction,
                    );
                }
                for bx in &self.boxes {
                    if let Some(normal) = box_face(bx, x) {
                        n.force += contact_force(
                            n.force,
                            n.vel,
                            n.mass,
                            normal,
                            h,
                            &ContactSurface::BOX,
                            0.0,
                            n.friction,
                        );
                    }
                }
            }
            if !n.has(NODE_IMMOVABLE) {
                n.vel += n.force / n.mass * h;
                n.pos += n.vel * h;
            }
            n.force = Vec3::new(0.0, n.mass * GRAVITY, 0.0);
            if self.fusedrag.is_some() {
                // The fuselage drag replaces the air drag and turbulence (no noise
                // drawn).
                n.force += self.fuse_force;
            } else if settings.drag {
                let s = approx_sqrt(n.vel.dot(n.vel));
                n.force += -DRAG * s * n.vel;
                if settings.turbulence {
                    let t = DRAG * s * s * TURBULENCE;
                    let x = noise_draw(&mut self.noise);
                    let y = noise_draw(&mut self.noise);
                    let z = noise_draw(&mut self.noise);
                    n.force += t * Vec3::new(x, y, z);
                }
            }
            self.nodes[i] = n;
        }
    }

    /// Wings in file order: forces from this step's velocities and positions.
    fn wing_pass(&mut self) {
        for wi in 0..self.wings.len() {
            let w = &self.wings[wi];
            if w.broken {
                continue;
            }
            let nodes = &self.nodes;
            let y = self.origin.y + nodes[w.nodes[FLD] as usize].pos.y;
            let out = w.forces(
                |i| nodes[i as usize].pos,
                |i| nodes[i as usize].vel,
                y,
                &self.airfoils[w.airfoil as usize],
            );
            let slots = w.nodes;
            let mut add = |slot: usize, f: Vec3| self.nodes[slots[slot] as usize].force += f;
            match out.induced {
                Some((i, WingSide::Left)) => {
                    add(BLU, i);
                    add(BLD, i);
                }
                Some((i, WingSide::Right)) => {
                    add(BRU, i);
                    add(BRD, i);
                }
                None => {}
            }
            for s in [FLD, FLU, FRD, FRU] {
                add(s, out.front);
            }
            for s in [BLD, BLU, BRD, BRU] {
                add(s, out.back);
            }
        }
    }

    /// The fuselage drag D for the next step's force restart.
    fn fuse_pass(&mut self) {
        let Some(fd) = &self.fusedrag else {
            return;
        };
        let p = &self.nodes[fd.front as usize];
        let y = self.origin.y + p.pos.y;
        if let Some(d) = fd.force(p.vel, y, self.nodes.len(), &self.airfoils[fd.airfoil as usize]) {
            self.fuse_force = d;
        }
    }

    /// Commands with no input: only self-centering beams change their rest length.
    fn command_pass(&mut self) {
        for c in &mut self.centering {
            let b = &mut self.beams[c.beam as usize];
            b.rest = c.update(b.rest, b.rest0, STEP);
        }
    }

    /// Commands with key input (spec §2.6): this step's key values are the
    /// triggers' inputs from the last beam pass, which are then cleared.
    fn key_command_pass(&mut self) {
        let mut values = [0.0f32; KEYS];
        for (v, input) in values.iter_mut().zip(self.key_input.iter_mut()).skip(1) {
            // max(player input 0, trigger input).
            *v = if 0.0 < *input { *input } else { 0.0 };
            *input = 0.0;
        }
        self.key_input[0] = 0.0;
        for c in &mut self.commands {
            let b = &mut self.beams[c.beam as usize];
            b.rest = c.update(b.rest, b.rest0, STEP, &values);
        }
    }

    /// Slide nodes in spawn order (spec §3.4), after self-collision.
    fn slide_pass(&mut self) {
        for i in 0..self.slides.len() {
            let Some(cur) = self.slides[i].segment else {
                continue;
            };
            if self.beams[self.segments[cur as usize].beam as usize].broken {
                continue;
            }
            let x = self.origin + self.nodes[self.slides[i].node as usize].pos;
            // At most one segment's hop: the previous one if nearer, then the next
            // one if nearer still.
            let mut best = (cur, self.segment_distance(cur, x));
            let seg = self.segments[cur as usize];
            if let Some(p) = seg.prev {
                let d = self.segment_distance(p, x);
                if d.1 < best.1 .1 {
                    best = (p, d);
                }
            }
            if let Some(n) = seg.next {
                let d = self.segment_distance(n, x);
                if d.1 < best.1 .1 {
                    best = (n, d);
                }
            }
            let (seg, (q, _, ratio)) = best;
            let slide = &mut self.slides[i];
            slide.segment = Some(seg);
            if slide.broken_away {
                continue;
            }
            let f = slide.force(q, x, ratio);
            let node = slide.node;
            let b = self.beams[self.segments[seg as usize].beam as usize];
            self.nodes[node as usize].force += f.node;
            self.nodes[b.a as usize].force += f.a;
            self.nodes[b.b as usize].force += f.b;
        }
    }

    fn beam_pass(&mut self) {
        for j in 0..self.beams.len() {
            if self.beams[j].broken {
                continue;
            }
            let mut beam = self.beams[j];
            let na = self.nodes[beam.a as usize];
            let nb = self.nodes[beam.b as usize];
            let out = if beam.kind == ExplicitKind::Shock2 {
                let sh = self.shocks2[beam.shock as usize];
                shock2_force(&mut beam, &sh, &na, &nb, self.settings.fast_inv_sqrt)
            } else {
                beam_force(&mut beam, &na, &nb, self.settings.fast_inv_sqrt)
            };
            if beam.kind == ExplicitKind::Trigger {
                // The trigger test (spec §2.5), for the next step's command pass.
                let t = self.triggers[beam.shock as usize];
                if let Some(k) = t.held(out.e, beam.rest, beam.short_bound, beam.long_bound) {
                    self.key_input[k as usize] = 1.0;
                }
            }
            let mut f = out.force;
            let mut cascade = false;
            if out.check_break {
                let saved = [beam.a, beam.b].iter().any(|&e| {
                    self.nodes[e as usize].has(NODE_ON_TRIANGLE) && self.live_plain(e) < 3
                });
                if saved {
                    beam.strength = 2.0 * beam.gate;
                } else {
                    beam.broken = true;
                    f = 0.0;
                    cascade = beam.group > 0;
                }
            }
            self.beams[j] = beam;
            if cascade {
                let g = beam.group;
                for other in &mut self.beams {
                    if other.group.abs() == g {
                        other.broken = true;
                    }
                }
            }
            let push = out.d * (f * out.inv);
            self.nodes[beam.a as usize].force += push;
            self.nodes[beam.b as usize].force -= push;
        }
    }

    fn collision_pass(&mut self) {
        self.last_contacts = 0;
        if self.grid_cell > 0.0 {
            let cell = self.grid_cell;
            let nodes = &self.nodes;
            self.grid
                .build(self.candidates.iter().map(|&c| nodes[c as usize].pos), cell);
        }
        let r = self.collision_range;
        let h = STEP;
        for t in 0..self.triangles.len() {
            let tri = &mut self.triangles[t];
            if tri.wait > 0 {
                tri.idle += 1;
                tri.wait -= 1;
                continue;
            } else if tri.idle > 0 {
                tri.wait = tri.idle.min(12);
                tri.idle = 0;
            }
            let [i0, i1, i2] = tri.nodes;
            let world = |i: u32| self.origin + self.nodes[i as usize].pos;
            let (p0, p1, p2) = (world(i0), world(i1), world(i2));
            let frame = TriFrame::new(p0, p1, p2);
            let lo = p0.min(p1).min(p2) - Vec3::splat(r);
            let hi = p0.max(p1).max(p2) + Vec3::splat(r);
            let mut hit = false;
            // The candidates to test, in candidate order: all of them, or those in
            // the grid buckets the padded box covers.
            let mut list = std::mem::take(&mut self.grid.list);
            list.clear();
            if self.grid_cell > 0.0 {
                self.grid.gather(lo - self.origin, hi - self.origin, self.grid_cell, &mut list);
            }
            let count = if self.grid_cell > 0.0 { list.len() } else { self.candidates.len() };
            for idx in 0..count {
                let k = if self.grid_cell > 0.0 { list[idx] as usize } else { idx };
                let hn = self.candidates[k];
                if hn == i0 || hn == i1 || hn == i2 {
                    continue;
                }
                let p = self.origin + self.nodes[hn as usize].pos;
                if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                    continue;
                }
                let Some((alpha, beta, gamma, delta, n)) = frame.hit(p, r) else {
                    continue;
                };
                hit = true;
                self.last_contacts += 1;
                let node = self.nodes[hn as usize];
                let (n0, n1, n2) = (
                    self.nodes[i0 as usize],
                    self.nodes[i1 as usize],
                    self.nodes[i2 as usize],
                );
                let v = node.vel - (alpha * n1.vel + beta * n2.vel + gamma * n0.vel);
                let mt = alpha * n1.mass + beta * n2.mass + gamma * n0.mass;
                let m = node.mass * mt / (node.mass + mt);
                let f = contact_force(
                    node.force,
                    v,
                    m,
                    n,
                    h,
                    &ContactSurface::BOX,
                    r - delta,
                    node.friction,
                );
                self.nodes[hn as usize].force += f;
                self.nodes[i1 as usize].force -= alpha * f;
                self.nodes[i2 as usize].force -= beta * f;
                self.nodes[i0 as usize].force -= gamma * f;
            }
            self.grid.list = list;
            let tri = &mut self.triangles[t];
            if hit {
                tri.wait = -20000;
            } else {
                tri.wait += 1;
            }
        }
    }
}

fn signed_group(b: &crate::desc::SoftBeam) -> i32 {
    let g = b.break_group as i32;
    if b.break_group_passive {
        -g
    } else {
        g
    }
}

/// Candidates above which self-collision uses the grid.
pub const GRID_MIN_CANDIDATES: usize = 64;
const GRID_BUCKETS: usize = 2048;

/// The self-collision grid: candidates counting-sorted into hashed buckets of
/// their cell (positions relative to the origin) each step.
#[derive(Clone, Debug, Default)]
struct CollisionGrid {
    /// Per bucket the first entry, one past the end at GRID_BUCKETS.
    start: Vec<u32>,
    /// Candidate indices, bucket by bucket.
    entries: Vec<u32>,
    /// Scratch: each candidate's bucket, one triangle's buckets and candidates, and
    /// a bit per candidate (to list them in candidate order without sorting).
    bucket_of: Vec<u32>,
    buckets: Vec<u32>,
    list: Vec<u32>,
    bits: Vec<u64>,
}

fn grid_cell_of(p: Vec3, cell: f32) -> [i32; 3] {
    (p / cell).floor().as_ivec3().to_array()
}

fn grid_bucket(c: [i32; 3]) -> u32 {
    ((c[0] as u32).wrapping_mul(73_856_093)
        ^ (c[1] as u32).wrapping_mul(19_349_663)
        ^ (c[2] as u32).wrapping_mul(83_492_791))
        & (GRID_BUCKETS as u32 - 1)
}

impl CollisionGrid {
    fn build(&mut self, positions: impl Iterator<Item = Vec3>, cell: f32) {
        self.start.clear();
        self.start.resize(GRID_BUCKETS + 1, 0);
        self.bucket_of.clear();
        for p in positions {
            let b = grid_bucket(grid_cell_of(p, cell));
            self.bucket_of.push(b);
            self.start[b as usize + 1] += 1;
        }
        for b in 0..GRID_BUCKETS {
            self.start[b + 1] += self.start[b];
        }
        self.entries.clear();
        self.entries.resize(self.bucket_of.len(), 0);
        let mut fill = self.start.clone();
        for (k, &b) in self.bucket_of.iter().enumerate() {
            self.entries[fill[b as usize] as usize] = k as u32;
            fill[b as usize] += 1;
        }
    }

    /// The candidates in the buckets of the cells the box `lo..hi` covers (each
    /// bucket once), in candidate order; every candidate when the box covers too
    /// many cells.
    fn gather(&mut self, lo: Vec3, hi: Vec3, cell: f32, out: &mut Vec<u32>) {
        let (c0, c1) = (grid_cell_of(lo, cell), grid_cell_of(hi, cell));
        let cells: i64 = (0..3).map(|i| i64::from(c1[i] - c0[i] + 1)).product();
        if cells > 64 {
            out.extend(0..self.bucket_of.len() as u32);
            return;
        }
        self.buckets.clear();
        self.bits.clear();
        self.bits.resize(self.bucket_of.len().div_ceil(64), 0);
        for x in c0[0]..=c1[0] {
            for y in c0[1]..=c1[1] {
                for z in c0[2]..=c1[2] {
                    let b = grid_bucket([x, y, z]);
                    if !self.buckets.contains(&b) {
                        self.buckets.push(b);
                        let (a, e) = (self.start[b as usize], self.start[b as usize + 1]);
                        for &k in &self.entries[a as usize..e as usize] {
                            self.bits[k as usize / 64] |= 1 << (k % 64);
                        }
                    }
                }
            }
        }
        for (w, &word) in self.bits.iter().enumerate() {
            let mut m = word;
            while m != 0 {
                out.push(w as u32 * 64 + m.trailing_zeros());
                m &= m - 1;
            }
        }
    }
}

/// A soft tyre: its axle nodes and its ring nodes `first..first + count`.
#[derive(Clone, Copy, Debug)]
pub struct Tyre {
    pub axle: [u32; 2],
    pub first: u32,
    pub count: u32,
    pub radius: f32,
}

impl ExplicitBeams {
    /// Set every tyre's ring nodes to roll without slipping at `velocity` on flat
    /// ground (+Y up): the velocity plus the spin about the axle that makes the
    /// contact point still, |ω| = |v| / radius. A vehicle spawned moving with
    /// still tyres would skid them on the ground until they spun up.
    pub fn roll_tyres(&mut self, velocity: Vec3) {
        for t in &self.tyres {
            let a0 = self.nodes[t.axle[0] as usize].pos;
            let a1 = self.nodes[t.axle[1] as usize].pos;
            let axis = (a1 - a0).normalize_or_zero();
            let centre = 0.5 * (a0 + a1);
            let spin = Vec3::Y.cross(velocity) / t.radius.max(1e-3);
            let omega = spin.dot(axis) * axis;
            for i in t.first..t.first + t.count {
                let n = &mut self.nodes[i as usize];
                n.vel = velocity + omega.cross(n.pos - centre);
            }
        }
    }
}

/// The face of `b` a point strictly inside it is pushed out through (its outward
/// normal), or `None` when the point is not strictly inside.
pub fn box_face(b: &StaticBox, x: Vec3) -> Option<Vec3> {
    let inside = x.x > b.lo.x
        && x.x < b.hi.x
        && x.y > b.lo.y
        && x.y < b.hi.y
        && x.z > b.lo.z
        && x.z < b.hi.z;
    if !inside {
        return None;
    }
    let mut normal = Vec3::NEG_Z;
    let mut dist = x.z - b.lo.z;
    for (d, n) in [
        (b.hi.z - x.z, Vec3::Z),
        (x.x - b.lo.x, Vec3::NEG_X),
        (b.hi.x - x.x, Vec3::X),
        (x.y - b.lo.y, Vec3::NEG_Y),
        (b.hi.y - x.y, Vec3::Y),
    ] {
        if d < dist {
            dist = d;
            normal = n;
        }
    }
    Some(normal)
}

/// What [`beam_force`] decided.
#[derive(Clone, Copy, Debug)]
pub struct BeamOut {
    /// a − b and 1/|a − b|.
    pub d: Vec3,
    pub inv: f32,
    /// Extension ℓ − L.
    pub e: f32,
    /// Force to apply (positive = compression), before the break test.
    pub force: f32,
    /// τ exceeded the strength: the beam breaks unless an end saves it.
    pub check_break: bool,
}

/// One beam's force with the kind rules and the yield rules applied to `beam`
/// (rest length, thresholds, strength, last force; a support stretched too far is
/// marked broken). The break test that needs the node graph is left to the caller
/// (`check_break`).
pub fn beam_force(beam: &mut Beam, na: &Node, nb: &Node, fast: bool) -> BeamOut {
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let inv = if fast {
        fast_inv_sqrt(q)
    } else {
        1.0 / q.sqrt()
    };
    let len = q * inv;
    let e = len - beam.rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let mut k = beam.k;
    let mut c = beam.c;
    let mut support_broke = false;
    match beam.kind {
        ExplicitKind::LimitedSpring => {
            let long = beam.long_bound * beam.rest;
            let short = beam.short_bound * beam.rest;
            let o = if e > long {
                e - long
            } else if e < -short {
                -e - short
            } else {
                0.0
            };
            if o != 0.0 {
                k += (beam.limit_k - k) * o;
                c += (beam.limit_c - c) * o;
            }
        }
        ExplicitKind::Support => {
            if e > 0.0 {
                k = 0.0;
                c *= 0.1;
                let factor = if beam.long_bound > 0.0 {
                    beam.long_bound
                } else {
                    SUPPORT_BREAK_FACTOR
                };
                if e > beam.rest * factor {
                    beam.broken = true;
                    support_broke = true;
                }
            }
        }
        ExplicitKind::Rope => {
            if e < 0.0 {
                k = 0.0;
                c *= 0.1;
            }
        }
        // Shock2 beams take `shock2_force`; here they act as plain ones. Triggers
        // have k = c = 0.
        ExplicitKind::Plain | ExplicitKind::Shock2 | ExplicitKind::Trigger => {}
    }
    let mut f = -k * e - c * u;
    beam.force = f;
    let mut tau = f.abs();
    let mut check_break = false;
    if !support_broke && tau > beam.gate {
        if beam.ty == ExplicitType::Deformable
            && beam.kind != ExplicitKind::LimitedSpring
            && k != 0.0
        {
            if f > beam.tc && e < 0.0 {
                let delta = e + (beam.tc / k) * (1.0 - beam.plastic);
                let old = beam.rest;
                beam.rest = (beam.rest + delta).max(MIN_REST);
                f -= (f - beam.tc) / 2.0;
                tau = f;
                if beam.rest < old {
                    beam.tc = beam.tc * old / beam.rest;
                    beam.gate = beam.tc.min(beam.tt).min(beam.strength);
                }
            } else if f < -beam.tt && e > 0.0 {
                let delta = e - (beam.tt / k) * (1.0 - beam.plastic);
                let old = beam.rest;
                beam.rest += delta;
                f -= (f + beam.tt) / 2.0;
                tau = -f;
                if beam.rest > old {
                    beam.tt = beam.tt * beam.rest / old;
                    beam.gate = beam.tc.min(beam.tt).min(beam.strength);
                }
                beam.strength -= delta * k;
            }
        }
        check_break = tau > beam.strength;
    }
    BeamOut {
        d,
        inv,
        e,
        force: f,
        check_break,
    }
}

/// A shock2 beam's force (spec §6): the rates by [`Shock2::rates`], no yield,
/// and the break test when |F| exceeds both the gate and the strength.
pub fn shock2_force(beam: &mut Beam, sh: &Shock2, na: &Node, nb: &Node, fast: bool) -> BeamOut {
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let inv = if fast {
        fast_inv_sqrt(q)
    } else {
        1.0 / q.sqrt()
    };
    let len = q * inv;
    let e = len - beam.rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let (k, c) = sh.rates(e, u, beam.rest, beam.short_bound, beam.long_bound);
    let f = -k * e - c * u;
    beam.force = f;
    let tau = f.abs();
    BeamOut {
        d,
        inv,
        e,
        force: f,
        check_break: tau > beam.gate && tau > beam.strength,
    }
}

/// A triangle's barycentric frame for the self-collision test.
#[derive(Clone, Copy, Debug)]
pub struct TriFrame {
    pub p0: Vec3,
    pub n: Vec3,
    /// Rows of the inverse of the matrix with columns (u, w, n), or `None` when its
    /// determinant is too small.
    pub inv: Option<[Vec3; 3]>,
}

impl TriFrame {
    pub fn new(p0: Vec3, p1: Vec3, p2: Vec3) -> Self {
        let u = p1 - p0;
        let w = p2 - p0;
        let c = u.cross(w);
        let len = c.length();
        let n = if len > 0.0 { c / len } else { c };
        let det = u.dot(w.cross(n));
        let inv = (det.abs() > 1e-6).then(|| {
            let r = 1.0 / det;
            [w.cross(n) * r, n.cross(u) * r, u.cross(w) * r]
        });
        Self { p0, n, inv }
    }

    /// (α, β, γ, δ, normal) when `p` is over the triangle within `r` of its plane;
    /// the normal points to the side `p` is on (δ ≥ 0).
    pub fn hit(&self, p: Vec3, r: f32) -> Option<(f32, f32, f32, f32, Vec3)> {
        let q = p - self.p0;
        let (alpha, beta, delta) = match self.inv {
            Some([ra, rb, rd]) => (ra.dot(q), rb.dot(q), rd.dot(q)),
            None => (0.0, 0.0, 0.0),
        };
        let gamma = 1.0 - alpha - beta;
        if alpha >= 0.0 && beta >= 0.0 && gamma >= 0.0 && delta.abs() <= r {
            if delta < 0.0 {
                Some((alpha, beta, gamma, -delta, -self.n))
            } else {
                Some((alpha, beta, gamma, delta, self.n))
            }
        } else {
            None
        }
    }
}
