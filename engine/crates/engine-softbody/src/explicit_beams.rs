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
use glam::{Quat, Vec3};
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
/// The BeamNG profile's contact band (spec §36): BeamNG documents a triangle
/// thickness of 2.5 cm ("a node starts feeling a triangle at around 2.5cm").
pub const BEAMNG_COLLISION_RANGE: f32 = 0.025;
/// The GPU's contact-force fixed point: units per newton (the kernel's `FIXED`).
pub const CONTACT_FIXED: f32 = 32.0;
/// The contact position step's fixed point (§37): units per metre, 2²⁴ (the
/// kernel's `SHIFT_FIXED`).
pub const SHIFT_FIXED: f32 = 16_777_216.0;

/// A position-step component in [`SHIFT_FIXED`] units, as the kernel's
/// `toFixedShift`: clamped to ±2·10⁹, rounded to nearest with ties to even.
pub fn to_fixed_shift(x: f32) -> i32 {
    (x * SHIFT_FIXED).clamp(-2.0e9, 2.0e9).round_ties_even() as i32
}

/// A force component in [`CONTACT_FIXED`] units, as the kernel's `toFixed`:
/// clamped to ±2·10⁹, rounded to nearest with ties to even.
pub fn to_fixed(x: f32) -> i32 {
    (x * CONTACT_FIXED).clamp(-2.0e9, 2.0e9).round_ties_even() as i32
}
/// Collision triangles a vehicle may have (a file with more fails to spawn).
pub const MAX_COLLISION_TRIANGLES: usize = 3000;

/// Node flags ([`Node::flags`]).
pub const NODE_NO_GROUND: u32 = 1;
pub const NODE_IMMOVABLE: u32 = 2;
pub const NODE_ON_TRIANGLE: u32 = 4;
pub const NODE_CONTACTER: u32 = 8;
pub const NODE_TYRE: u32 = 16;
/// JBeam pressure-wheel nodes: ground friction never reverses the slip in a step
/// (`engine/docs/jbeam-semantics-spec.md` §28).
pub const NODE_FRICTION_LIMIT: u32 = 32;
/// A pressure wheel's tread node (JBeam mode): its ground contact is the tread
/// rubber's compliant, hysteretic one ([`tread_hysteresis`],
/// `engine/docs/jbeam-semantics-spec.md` §34) in place of the inelastic rule.
pub const NODE_TREAD: u32 = 64;
/// The tread compound's loss factor tan δ at 60 °C and 10 Hz: the mean of the three
/// silica-filled S-SBR tread compounds of Um et al., "The Influence of Styrene
/// Content in Solution Styrene Butadiene Rubber on Silica-Filled Tire Tread
/// Compounds", Polymers 15(21) 4288 (2023), doi:10.3390/polym15214288, Table 9
/// (0.0823, 0.0818, 0.0805; DMA, 10 Hz, 0.2 % strain). Spec §34.
pub const TREAD_TAN_DELTA: f32 = 0.0815;
/// The tread contact's hysteretic slope while its hysteretic deflection grows
/// (μ = r·λ, r = 4/π − 1; spec §34).
pub const TREAD_GROW: f32 = 0.184_065_8;
/// Its slope after a reversal, until its hysteretic deflection passes zero
/// (λ, from [`TREAD_TAN_DELTA`]; spec §34).
pub const TREAD_REVERSE: f32 = 0.673_642_6;
/// 1/[`TREAD_REVERSE`] (a product on both solvers, no division).
pub const TREAD_INV_REVERSE: f32 = 1.484_466_7;
/// The tread contact's elastic stiffness per kilogram of node, 1/(h²·(1 + λ)) (s⁻²):
/// its stiffest tangent, k·(1 + λ), is m/h² (ω·h = 1). **Choice** (spec §34).
pub const TREAD_K: f32 = 2_389_996.5;

/// Optional parts of the compatible profile. All on by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplicitBeamsSettings {
    /// Beam lengths through the fast inverse square root (off: exact).
    pub fast_inv_sqrt: bool,
    /// Air drag on every node.
    pub drag: bool,
    /// Pseudo-random turbulence on top of the drag (needs `drag`).
    pub turbulence: bool,
    /// Rigs of Rods' idle-triangle throttle: a collision triangle with no
    /// candidate is skipped for up to 12 steps, in self-collision and vehicle
    /// contact (`explicit-beams.md` §3). Off in the BeamNG profile
    /// ([`Self::beamng`], jbeam-semantics-spec.md §36): every triangle is tested
    /// every step.
    pub triangle_throttle: bool,
    /// Centred damping (jbeam-semantics-spec.md §39; on in the BeamNG profile,
    /// off in the Rigs of Rods one): each JBeam element's damping reads the mean
    /// of the step's start and end velocities instead of the start's, by a
    /// correction after the node update. Lifts the damped explicit step's limit
    /// from ω·h ≤ 2(√(1+ζ²) − ζ) to 2 and past it; JBeam elements only.
    pub centred_damping: bool,
    /// The centred damping's correction from each beam's change of extension
    /// rate, each rate along the beam's own direction at its time
    /// (jbeam-semantics-spec.md §39.6): Δu = (v′_a − v′_b)·d̂′ − (v_a − v_b)·d̂.
    /// On by default (it acts only with [`Self::centred_damping`]; the CPU
    /// reference and the GPU kernels, `FLAG_CENTRED_OBJECTIVE`). Off: the
    /// change of the relative velocity along the end direction alone, which
    /// reads a beam that only turns as shortening at ω²·L·h and adds energy to
    /// anything that spins (known-bugs S3); kept for comparison.
    pub centred_objective: bool,
}

impl Default for ExplicitBeamsSettings {
    fn default() -> Self {
        Self {
            fast_inv_sqrt: true,
            drag: true,
            turbulence: true,
            triangle_throttle: true,
            centred_damping: false,
            centred_objective: true,
        }
    }
}

impl ExplicitBeamsSettings {
    /// The BeamNG profile (jbeam-semantics-spec.md §36): JBeam content and a
    /// `.truck` with a `rubble_beamng_profile` line. Each switch can be turned
    /// back alone; the default (Rigs of Rods) profile is unchanged.
    pub fn beamng(self) -> Self {
        Self {
            fast_inv_sqrt: false,
            triangle_throttle: false,
            centred_damping: true,
            ..self
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
    /// The friction law's exponentials by [`exact_exp`] / [`exact_pow`] (the BeamNG
    /// profile, jbeam-semantics-spec.md §36); false: RoR's bit tricks.
    pub exact: bool,
    /// The normal force does no positive work (the BeamNG profile's self- and
    /// vehicle contact, jbeam-semantics-spec.md §37): RoR's penetration push is
    /// capped so it never makes the pair separate ([`contact_force`]). False:
    /// RoR's rule.
    pub dissipative: bool,
}

impl ContactSurface {
    /// This surface with the exact exponentials when `on`.
    pub const fn exact_if(self, on: bool) -> Self {
        Self { exact: on, ..self }
    }
    /// This surface with [`Self::dissipative`] when `on`.
    pub const fn dissipative_if(self, on: bool) -> Self {
        Self {
            dissipative: on,
            ..self
        }
    }
    /// The ground ("gravel").
    pub const GROUND: Self = Self {
        va: 3.0,
        ms: 0.85,
        mc: 0.6,
        t2: 0.006,
        vs: 3.0,
        alpha: 2.0,
        strength: 1.0,
        exact: false,
        dissipative: false,
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
        exact: false,
        dissipative: false,
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
    /// JBeam NORMAL (and HYDRO with no input) beams in JBeam mode
    /// (`engine/docs/jbeam-semantics-spec.md` §3): the plain force and yield rule,
    /// the rest length kept within `short_bound ..= long_bound` (m; `deformLimit`,
    /// `deformLimitExpansion`), the yield thresholds hardening up to `limit_k`
    /// (`deformLimitStress`), and no break exception.
    JbeamPlain,
    /// JBeam SUPPORT beams in JBeam mode (spec §4): no force at all while longer
    /// than rest; break longer than `long_bound` (m: `(1 + beamLongBound)` × the
    /// spawn length); in compression a [`ExplicitKind::JbeamPlain`] beam.
    JbeamSupport,
    /// JBeam BOUNDED beams in JBeam mode (spec §5): past the bounds (`short_bound`,
    /// `long_bound` × rest) the spring and damping move to `limit_k`, `limit_c`
    /// over the bound zone (`plastic`, m); damping `c` up to the velocity split
    /// (`tc`, m/s) and `tt` past it. Never yields.
    JbeamBounded,
    /// A JBeam coupler in JBeam mode (spec §7): the two nodes held together by a
    /// zero-length isotropic spring–damper (`k`, `c`, [`coupler_rates`]),
    /// breaking when that force passes the strength.
    JbeamCoupler,
    /// A JBeam torsion bar (and a torsion hydro with no input) in JBeam mode (spec
    /// §9), as two beam records: this one on nodes 1–2 computes the element, the
    /// next ([`ExplicitKind::JbeamTorsionPair`], on nodes 3–4) only carries the
    /// forces on nodes 3 and 4. Torque −k·wrap(φ − φ₀) − c·φ̇ on the dihedral angle φ
    /// about the 2→3 axis ([`torsion_force`]): `k`, `c` (N·m/rad, N·m·s/rad), φ₀ the
    /// rest angle in `rest` (spawn angle in `rest0`), the deform and break torques in
    /// `tc` and `strength`.
    JbeamTorsion,
    /// The second record of a [`ExplicitKind::JbeamTorsion`] element (no force of
    /// its own).
    JbeamTorsionPair,
    /// JBeam ANISOTROPIC beams (spec §20; extension record E1 = (k expansion, c
    /// expansion, transition zone, break length m)).
    JbeamAniso,
    /// JBeam PRESSURED beams (spec §21; E1 = (surface, absolute spawn pressure,
    /// volume coefficient, spawn length), E2 = (gauge limit, gauge break)).
    JbeamPressured,
    /// JBeam LBEAMs (spec §24) between id1 (`a`) and id2 (`b`), the vertex id3 in E1
    /// .x (bits); `rest` holds cos θ₀.
    JbeamLbeam,
    /// A pressure group (spec §26): no force; E1 = (first triangle record, triangle
    /// count (bits), V₀, absolute spawn pressure), E2 .x the gauge ceiling.
    JbeamPressureGroup,
    /// A pressure group's triangle, first record (nodes x₀, x₁; E1 = (x₂, group
    /// record) as bits); the next record ([`ExplicitKind::JbeamPressureTri2`], x₂–x₂)
    /// carries x₂'s force.
    JbeamPressureTri,
    JbeamPressureTri2,
    /// A triangle's aerodynamics (spec §31), first record (nodes x₀, x₁; E1 = (x₂
    /// as bits, drag factor, lift factor, sin of the stall angle)); the next record
    /// ([`ExplicitKind::JbeamAeroTri2`], x₂–x₂) carries x₂'s force.
    JbeamAeroTri,
    JbeamAeroTri2,
    /// An LBEAM's second record (x₃–x₃, after its [`ExplicitKind::JbeamLbeam`]):
    /// carries the vertex's force (the arm terms, spec §24) on the GPU; no force
    /// of its own.
    JbeamLbeam2,
    /// A rope by JBeam's NORMAL rule (the BeamNG profile's `.truck` ropes, spec
    /// §36): in compression no spring and a tenth of its damping (RoR's rope), in
    /// tension a [`ExplicitKind::JbeamPlain`] beam (yield, limits, tear).
    JbeamRope,
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
            ExplicitKind::JbeamPlain => 6,
            ExplicitKind::JbeamSupport => 7,
            ExplicitKind::JbeamBounded => 8,
            ExplicitKind::JbeamCoupler => 9,
            ExplicitKind::JbeamTorsion => 10,
            ExplicitKind::JbeamTorsionPair => 11,
            ExplicitKind::JbeamAniso => 12,
            ExplicitKind::JbeamPressured => 13,
            ExplicitKind::JbeamLbeam => 14,
            ExplicitKind::JbeamPressureGroup => 15,
            ExplicitKind::JbeamPressureTri => 16,
            ExplicitKind::JbeamPressureTri2 => 17,
            ExplicitKind::JbeamAeroTri => 18,
            ExplicitKind::JbeamAeroTri2 => 19,
            ExplicitKind::JbeamLbeam2 => 20,
            ExplicitKind::JbeamRope => 21,
        }
    }

    /// One of the JBeam-mode kinds (no break exception; their own rules).
    pub fn is_jbeam(self) -> bool {
        matches!(
            self,
            ExplicitKind::JbeamPlain
                | ExplicitKind::JbeamRope
                | ExplicitKind::JbeamSupport
                | ExplicitKind::JbeamBounded
                | ExplicitKind::JbeamCoupler
                | ExplicitKind::JbeamTorsion
                | ExplicitKind::JbeamTorsionPair
                | ExplicitKind::JbeamAniso
                | ExplicitKind::JbeamPressured
                | ExplicitKind::JbeamLbeam
                | ExplicitKind::JbeamPressureGroup
                | ExplicitKind::JbeamPressureTri
                | ExplicitKind::JbeamPressureTri2
                | ExplicitKind::JbeamAeroTri
                | ExplicitKind::JbeamAeroTri2
                | ExplicitKind::JbeamLbeam2
        )
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
    /// [`ExplicitKind::Trigger`]: its record in [`ExplicitBeams::triggers`];
    /// JBeam-mode kinds: 1 + its record in [`ExplicitBeams::damp_filters`]
    /// (0 = unfiltered damping).
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
        !self.broken && (self.rest - self.rest0).abs() > 1e-3 * self.rest0.abs()
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
    fn progressive(
        k0: f32,
        c0: f32,
        pk: f32,
        pc: f32,
        e: f32,
        bound: f32,
        rest: f32,
    ) -> (f32, f32) {
        let lambda = if bound == 0.0 {
            1.0
        } else {
            let r = e / (bound * rest);
            min1(r * r)
        };
        (k0 + (pk * k0) * lambda, c0 + (pc * c0) * lambda)
    }

    fn out_rates(&self, e: f32, lb: f32, rest: f32) -> (f32, f32) {
        Self::progressive(
            self.k_out,
            self.c_out,
            self.pk_out,
            self.pc_out,
            e,
            lb,
            rest,
        )
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

/// A static collider other than an axis-aligned [`StaticBox`] (Rubble only: RoR's
/// reference has flat ground and axis-aligned boxes). The contact rule is the box
/// rule: a node strictly inside is pushed out along the normal of the nearest
/// surface, with the box surface ("concrete") and no penetration term. Build it
/// with the constructors (they normalise the rotation and fill `axes`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StaticShape {
    /// A box: centre, half extents, rotation (unit quaternion) and the rotation's
    /// matrix columns (the box's local x, y, z axes in world). An identity
    /// rotation is the axis-aligned box `StaticBox::centered(centre, half)`,
    /// bit for bit.
    OrientedBox {
        centre: Vec3,
        half: Vec3,
        rotation: Quat,
        axes: [Vec3; 3],
    },
    /// A vertical (world Y) cylinder: centre, radius, half height. Pushed out
    /// radially through its side or along ±Y through a cap.
    Cylinder {
        centre: Vec3,
        radius: f32,
        half_height: f32,
    },
    /// A capsule: segment ends a, b and radius (pushed out from the segment).
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    /// A sphere: centre and radius.
    Sphere { centre: Vec3, radius: f32 },
}

impl StaticShape {
    /// A box with centre `c`, half extents `half` and rotation `rotation`.
    pub fn oriented_box(c: Vec3, half: Vec3, rotation: Quat) -> Self {
        let rotation = if rotation == Quat::IDENTITY {
            rotation
        } else {
            rotation.normalize()
        };
        let m = glam::Mat3::from_quat(rotation);
        Self::OrientedBox {
            centre: c,
            half,
            rotation,
            axes: [m.x_axis, m.y_axis, m.z_axis],
        }
    }

    /// A vertical cylinder.
    pub fn cylinder(c: Vec3, radius: f32, half_height: f32) -> Self {
        Self::Cylinder {
            centre: c,
            radius,
            half_height,
        }
    }

    /// A capsule from `a` to `b`.
    pub fn capsule(a: Vec3, b: Vec3, radius: f32) -> Self {
        Self::Capsule { a, b, radius }
    }

    /// A sphere.
    pub fn sphere(c: Vec3, radius: f32) -> Self {
        Self::Sphere { centre: c, radius }
    }

    /// The kind code the GPU records and the page layout use: 1 oriented box,
    /// 2 cylinder, 3 capsule, 4 sphere (0 is an axis-aligned box).
    pub fn kind(&self) -> u32 {
        match self {
            Self::OrientedBox { .. } => 1,
            Self::Cylinder { .. } => 2,
            Self::Capsule { .. } => 3,
            Self::Sphere { .. } => 4,
        }
    }

    /// An oriented box with an identity rotation (it takes the axis-aligned path).
    pub fn as_aabb(&self) -> Option<StaticBox> {
        match *self {
            Self::OrientedBox {
                centre,
                half,
                rotation,
                ..
            } if rotation == Quat::IDENTITY => Some(StaticBox::centered(centre, half)),
            _ => None,
        }
    }

    /// A world box holding the shape with a margin (a cull before the exact
    /// test, the same on the CPU and the GPU): 1 cm plus 1e-5 of the largest
    /// coordinate, far above the rounding of the tests.
    pub fn bounds(&self) -> StaticBox {
        let (lo, hi) = match *self {
            Self::OrientedBox {
                centre, half, axes, ..
            } => {
                let ext = axes[0].abs() * half.x + axes[1].abs() * half.y + axes[2].abs() * half.z;
                (centre - ext, centre + ext)
            }
            Self::Cylinder {
                centre,
                radius,
                half_height,
            } => {
                let e = Vec3::new(radius, half_height, radius);
                (centre - e, centre + e)
            }
            Self::Capsule { a, b, radius } => (
                a.min(b) - Vec3::splat(radius),
                a.max(b) + Vec3::splat(radius),
            ),
            Self::Sphere { centre, radius } => {
                (centre - Vec3::splat(radius), centre + Vec3::splat(radius))
            }
        };
        let pad = 0.01 + 1e-5 * lo.abs().max(hi.abs()).max_element();
        StaticBox {
            lo: lo - Vec3::splat(pad),
            hi: hi + Vec3::splat(pad),
        }
    }

    /// The centre, half extents, rotation and kind the page renders: a cylinder
    /// is (radius, half height, radius), a capsule its midpoint with
    /// (radius, half segment length, radius) and the rotation taking +Y along
    /// a → b, a sphere (radius, radius, radius).
    pub fn render_row(&self) -> [f32; 11] {
        let (c, h, q) = match *self {
            Self::OrientedBox {
                centre,
                half,
                rotation,
                ..
            } => (centre, half, rotation),
            Self::Cylinder {
                centre,
                radius,
                half_height,
            } => (
                centre,
                Vec3::new(radius, half_height, radius),
                Quat::IDENTITY,
            ),
            Self::Capsule { a, b, radius } => {
                let d = b - a;
                let len = d.length();
                let q = if len > 0.0 {
                    Quat::from_rotation_arc(Vec3::Y, d / len)
                } else {
                    Quat::IDENTITY
                };
                (0.5 * (a + b), Vec3::new(radius, 0.5 * len, radius), q)
            }
            Self::Sphere { centre, radius } => (centre, Vec3::splat(radius), Quat::IDENTITY),
        };
        [
            c.x,
            c.y,
            c.z,
            h.x,
            h.y,
            h.z,
            q.x,
            q.y,
            q.z,
            q.w,
            self.kind() as f32,
        ]
    }
}

/// A node grab (Rubble only; BeamNG-style "node grabber"): an external
/// spring–damper pulling one node toward a world point. The node pass, after its
/// force restart, adds `k·(target − x) − c·v` (x the node's world position and v
/// its velocity after integration) to the node's force, which the beam and
/// self-collision passes then add to and the next node pass consumes, as every
/// other force. Set per frame by an interactive host; `None` leaves every step
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grab {
    /// The grabbed node (vehicle-local index).
    pub node: u32,
    /// The point it is pulled toward (world, m).
    pub target: Vec3,
    /// Stiffness (N/m) and damping (N·s/m).
    pub k: f32,
    pub c: f32,
}

impl Grab {
    /// The force on a node at world position `x` moving at `v`.
    pub fn force(&self, x: Vec3, v: Vec3) -> Vec3 {
        (self.target - x) * self.k - v * self.c
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
    /// Oriented boxes, cylinders, capsules and spheres, tested after `boxes` in
    /// order (Rubble only; see [`StaticShape`]).
    pub shapes: Vec<StaticShape>,
    /// World position the node positions are relative to.
    pub origin: Vec3,
    /// Flat ground height (m).
    pub ground: f32,
    /// Self-collision range r (m).
    pub collision_range: f32,
    /// Soft tyres built from a `.truck` file's `wheels` (for [`Self::roll_tyres`]).
    pub tyres: Vec<Tyre>,
    /// Pressure wheels' parking brakes (JBeam `parkingTorque`, spec §25.1), in
    /// file order; empty for content without them (no pass runs).
    pub brakes: Vec<WheelBrake>,
    /// The parking brake input, 0 (released, the default) to 1 (applied).
    pub parking_brake: f32,
    /// Self-collision contacts in the last step (a statistic; not simulated state).
    pub last_contacts: u32,
    /// Self-collision (triangle, contacter) pairs inside the box test in the last step
    /// (those tested for a hit; a statistic, not simulated state).
    pub last_self_pairs: u32,
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
    /// JBeam mode's self-collision groups (`engine/docs/jbeam-semantics-spec.md`
    /// §8): per node its group (0 = none); a node skips the collision triangles
    /// whose three nodes are all in its group. Empty: no exclusion.
    pub self_groups: Vec<u32>,
    /// Per collision triangle the self-collision nodes it never collides with (they
    /// started inside its range at spawn, spec §10.1). Empty: none.
    pub tri_excl: Vec<Vec<u32>>,
    /// Every node (but those with no collision) is a contacter against other
    /// vehicles' triangles (spec §10.2); off: the file's contacters only.
    pub contact_every_node: bool,
    /// The file is its vehicle at rest (`rubble_rest_pose`): the depth of its
    /// lowest node (a tyre's contact patch) under the ground it stands on
    /// ([`Self::spawn_clearance`]).
    pub rest_pose: Option<f32>,
    /// Vehicle contact's two-body rule (jbeam-semantics-spec.md §10.3): a hit where
    /// this vehicle is the owner or the partner splits masses per turn and cancels
    /// the pair's relative acceleration. On in JBeam mode; off (both vehicles): RoR's
    /// rule, the contacter's whole force as against a static surface (bit for bit
    /// as before; the `.truck` default until the T-bone leak is fixed).
    pub two_body_contact: bool,
    /// Anti-clip (BeamNG profile, jbeam-semantics-spec.md §36): a contacter that
    /// crossed a collision triangle's plane over the last step, inside the
    /// triangle, is caught from the side it came from (a swept test, whatever its
    /// speed), in self-collision and vehicle contact. Off: RoR's band test alone.
    pub anti_clip: bool,
    /// Anti-clip's box padding by each contacter's own travel (a Rubble setting, not
    /// in the specification; off by default): a contacter is tested against a triangle
    /// when inside its box padded by r + (the triangle's fastest node's speed + the
    /// contacter's own speed, at most the vehicle's fastest) · h + CLIP_DEPTH, instead
    /// of the vehicle's fastest node's speed (§36). A node's anti-clip contact needs
    /// only its own and the triangle's travel; the fastest node's padding also catches
    /// what only some other fast node's speed reached. Without anti-clip it does nothing.
    pub own_travel_reach: bool,
    /// Retired debris (`crate::explicit_debris::DebrisTtl`; the GPU's
    /// `ExplicitBeamsGpu::set_retired`): per node whether it is retired (empty: none). A
    /// retired node is no longer simulated: the node pass leaves it at rest where it is,
    /// and it takes part in no contact (self-collision as contacter or triangle node,
    /// surface contact as point, segment or triangle node). Its own beams are still
    /// evaluated (they cannot move it). The host sets it; nothing here changes it.
    pub retired: Vec<bool>,
    /// Anti-clip's side memory (§36; `explicit_contact::clip_word`): per node the
    /// sides held this step and those recorded for the next. Sized when anti-clip
    /// first runs; rolled by the node pass.
    pub clip_sides: Vec<[u32; 4]>,
    /// Per collision triangle: anti-clip only (BeamNG `NONCOLLIDABLE`, §36): no
    /// force in front of its plane, a node behind it pushed back to the plane. Such
    /// triangles follow the collision ones in [`Self::triangles`] and take part only
    /// while [`Self::anti_clip`] is on. Empty: none.
    pub anti_only_tris: Vec<bool>,
    /// Contact forces through the GPU's fixed-point sum (BeamNG profile, spec §36):
    /// self-collision and vehicle-contact forces are added per node as integers of
    /// 1/[`CONTACT_FIXED`] N (rounded to nearest, ties to even), and the node pass
    /// adds their sum to the node's force, then the slide forces (the kernel's
    /// order). The CPU then computes the same values as the GPU. Off: added to the
    /// node's force as floats (the RoR profile).
    pub fixed_contact: bool,
    /// Self- and vehicle contact never do positive work (the BeamNG profile,
    /// jbeam-semantics-spec.md §37; [`ContactSurface::dissipative`]): RoR's
    /// penetration push may stop an approach but not make the pair separate.
    /// Vehicle contact takes it when the owner or the partner has it. Off: RoR's
    /// rule.
    pub dissipative_contact: bool,
    /// Surface contact (opt-in, jbeam-semantics-spec.md §38;
    /// [`crate::explicit_surface`]): the collision triangles meet the static boxes and
    /// shapes and the rigid proxies by their surface (edges and faces), beside node
    /// contact, with the two-body dissipative rule in the vehicle's band. Off (the
    /// default): node contact only, every step bit for bit as before.
    pub surface_contact: bool,
    /// The surface pass's hits in the last step (a statistic; not simulated state).
    pub surface_last: u32,
    /// The surface pass's narrow-phase queries in the last step (triangles and segments
    /// against a shape, casts aside): a work counter, not simulated state.
    pub surface_queries: u32,
    /// The surface pass's separation cull ([`engine_schema::feature_query::separated`];
    /// lossless, on by default): off, every element whose box meets a shape's runs the
    /// narrow phase (tests and measurement).
    pub surface_separation: bool,
    /// The surface pass's segments (beam-edge contact, [`crate::explicit_surface::surface_segments`]),
    /// found at its first step.
    pub surface_segs: Option<Vec<u32>>,
    /// The surface pass's total force on the nodes in the last step (N; as their
    /// contact sums take it): a statistic for momentum ledgers, not simulated state.
    pub surface_force: glam::DVec3,
    /// [`Self::dissipative_contact`]'s position steps this step, per node in
    /// [`SHIFT_FIXED`] units (summed as integers, order free); the next node pass
    /// adds them to the positions after x + h·v (empty until used).
    pub contact_shift: Vec<[i32; 3]>,
    /// The position steps the last node pass applied (m; empty: none).
    pub shift_applied: Vec<Vec3>,
    /// The ground position steps the last node pass applied (m up, per node; empty
    /// without [`Self::dissipative_contact`]).
    pub ground_lift: Vec<f32>,
    /// RoR's rule that a beam's strength falls by k × its plastic stretch (true,
    /// the RoR profile and the solver's JBeam rule until §36). The BeamNG profile:
    /// false (BeamNG documents no such rule; its `deformLimitStress` and deform
    /// limits are the documented plastic caps).
    pub plastic_strength_drop: bool,
    /// The friction law's exponentials exact ([`exact_exp`], [`exact_pow`]) in
    /// ground, static-shape, self- and vehicle contact (the BeamNG profile, §36).
    pub exact_friction: bool,
    /// [`Self::fixed_contact`]'s per-node sums this step (empty until used).
    pub contact_acc: Vec<[i32; 3]>,
    /// [`Self::fixed_contact`]: this step's slide-force sums, added by the next
    /// node pass after the contact sums.
    pub slide_hold: Vec<(u32, Vec3)>,
    /// A measurement (`None`: off; `Some(Vec::new())` turns it on): beside
    /// [`Self::contact_acc`], every node's contact forces summed exactly (f64) and
    /// counted, so the fixed point's quantisation is `contact_sum − exact`
    /// ([`Self::contact_quantisation`]), at most `count`·√3/64 N (each add rounds
    /// each component by at most half a unit; the rest of the arithmetic is exact).
    pub contact_exact: Option<Vec<(glam::DVec3, u32)>>,
    /// A measurement (`None`: off): the last step's two-body self-collision hits
    /// as found (contacter, triangle nodes, weights, depth, normal).
    pub self_hit_log: Option<Vec<(u32, [u32; 3], [f32; 3], f32, Vec3)>>,
    /// A measurement (`None`: off): the last step's two-body self-contact forces
    /// per node by phase: the first pass's normal part, its friction, the residual
    /// pass (N; the node's share, as added).
    pub self_contact_parts: Option<Vec<[Vec3; 3]>>,
    /// The contact-force log (jbeam-semantics-spec.md §32): per node the sum of
    /// its contacts' force magnitudes per step, and the contact breaks it drives
    /// (`None`: off, the default; [`Self::enable_contact_force`]).
    pub contact_force: Option<Box<ContactForceLog>>,
    /// Per node its tread contact's state ([`NODE_TREAD`] nodes, spec §34): its
    /// hysteretic deflection z and its depth the step before, both 0 off the
    /// ground. Empty when no node is a tread node (the pressure wheels' builder sizes it).
    pub tread: Vec<[f32; 2]>,
    /// JBeam mode's damping filters (`dampCutoffHz`), by [`Beam::shock`] − 1.
    pub damp_filters: Vec<DampFilter>,
    /// JBeam torsion elements' watched break groups (0 = none), by [`Beam::shock`]
    /// − 1 of their [`ExplicitKind::JbeamTorsion`] record: the groups of the beams
    /// between their consecutive nodes (spec §9). A torsion element breaks the
    /// step after one of them breaks.
    pub torsion_watch: Vec<[u32; 3]>,
    /// Break groups set off in the last beam pass (for [`Self::torsion_watch`]) and
    /// in this one (state).
    pub groups_last: Vec<u32>,
    pub groups_now: Vec<u32>,
    /// The node grab applied each step ([`Grab`]); `None` (the default) for none.
    pub grab: Option<Grab>,
    /// JBeam Part II (`engine/docs/jbeam-semantics-spec.md` §16–§26;
    /// [`crate::explicit_jbeam_ext`]). All empty for vehicles that use none of it.
    /// Extension records, and per beam its record + 1 (0 none; empty: none).
    pub beam_ext: Vec<crate::explicit_jbeam_ext::BeamExt>,
    pub ext_of: Vec<u32>,
    /// Per collision triangle the beams whose break kills it (u32::MAX none; §18)
    /// and whether it is dead (from the start of this step's beam pass).
    pub tri_breaks: Vec<[u32; 4]>,
    pub tri_dead: Vec<bool>,
    /// Per deform group whether it has triggered (§19), and the groups' names.
    pub deform_state: Vec<u32>,
    pub deform_names: Vec<String>,
    /// Rigid bodies this vehicle meets ([`crate::explicit_rigid`]); `None` (the
    /// default) for none, every step unchanged.
    pub rigid: Option<Box<crate::explicit_rigid::ProxySet>>,
    /// The energy ledger's per-node force split of the last step (None: not
    /// traced; [`crate::explicit_contact::EnergyLedger`]).
    pub trace: Option<Box<ForceTrace>>,
    /// Raycast wheels, drive and driver input ([`crate::explicit_drive`]); `None`
    /// (the default) for none, every step unchanged.
    pub drive: Option<Box<crate::explicit_drive::Drive>>,
    /// Spawned on the BeamNG profile (jbeam-semantics-spec.md §36): JBeam
    /// content, or a `.truck` with `rubble_beamng_profile`; else Rigs of
    /// Rods'. What hosts report as the vehicle's profile.
    pub beamng_profile: bool,
    /// Vehicle sleep's state ([`crate::explicit_sleep`]; used only by a fleet with the
    /// setting on): still count, asleep, anchor.
    pub sleep: crate::explicit_sleep::SleepState,
    /// Measurement (jbeam-semantics-spec.md §39's comparison; never on in a
    /// profile): the velocity the beam pass's damping reads, by a dry beam pass
    /// (state restored) that predicts each node's end-of-step velocity. 0: off,
    /// the step's velocity. 1: the predicted velocity (one Jacobi step toward
    /// backward-Euler damping). 2: the mean of the two (what
    /// [`ExplicitBeamsSettings::centred_damping`] gives at about a third of the cost).
    pub damp_scheme: u32,
    /// [`ExplicitBeamsSettings::centred_damping`]: each beam record's damping
    /// slope from the last beam pass ([`BeamOut::slope`]; 0 for a record that
    /// added no damping force). The kernel keeps it in the record's first
    /// incidence entry's `.w`.
    pub damp_slopes: Vec<f32>,
}

/// The wheel model of a JBeam-mode spawn ([`ExplicitBeams::from_desc_jbeam_with`]).
#[derive(Clone, Copy, Debug)]
pub enum JbeamWheels {
    /// The `.truck` path's soft tyres with these defaults (spec §12).
    Soft(crate::truck_export::TyreDefaults),
    /// JBeam pressure wheels (spec §25).
    Pressure,
    /// No wheel nodes: raycast wheels on the axle nodes ([`crate::explicit_drive`]).
    Raycast,
}

/// A JBeam beam's damping low-pass (`dampCutoffHz`, spec §6): the damping uses
/// y = a·u − b·y₋₁ of the extension rate u, the discrete RC low-pass at the cutoff
/// (a = 1 − exp(−2π·f_c·h), b = a − 1). Its phase lag stays small up to the step's
/// Nyquist rate, so it never turns damping into anti-damping on a fast mode (a
/// filter with no gain at the Nyquist rate does, past ω·h = √2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DampFilter {
    pub a: f32,
    pub b: f32,
    /// Last step's rate and output (state).
    pub u: f32,
    pub y: f32,
}

impl DampFilter {
    /// The filter for a cutoff of `hz` at the fixed step.
    pub fn new(hz: f32) -> Self {
        let a = 1.0 - exact_exp(-2.0 * std::f32::consts::PI * hz * STEP);
        Self {
            a,
            b: a - 1.0,
            u: 0.0,
            y: 0.0,
        }
    }

    /// Filter this step's rate.
    pub fn step(&mut self, u: f32) -> f32 {
        let y = self.a * u - self.b * self.y;
        self.u = u;
        self.y = y;
        y
    }
}

// --- numerical primitives (compatible profile) ---

/// The fast inverse square root: magic-constant estimate, one Newton iteration.
pub fn fast_inv_sqrt(x: f32) -> f32 {
    let i = 0x5f37_59df_i32.wrapping_sub((x.to_bits() as i32) >> 1);
    let y = f32::from_bits(i as u32);
    y * (1.5 - 0.5 * x * y * y)
}

/// The exact path's inverse square root (`rubble_exact_lengths`,
/// jbeam-semantics-spec.md §33): the fast inverse square root's estimate refined by
/// three more Newton steps, y ← y·(1.5 − (½x·y)·y), in IEEE single precision in
/// this order (products and differences only, each rounded once). The GPU kernel
/// runs the same sequence (`sExactInvSqrt`), so the two agree bit for bit (a
/// correctly rounded 1/√x needs a correctly rounded square root and division,
/// which the GPU backends do not promise). Within 2 ulp of 1/√x
/// (tests/energy_ledger.rs).
pub fn exact_inv_sqrt(x: f32) -> f32 {
    let i = 0x5f37_59df_i32.wrapping_sub((x.to_bits() as i32) >> 1);
    let mut y = f32::from_bits(i as u32);
    let hx = 0.5 * x;
    y = y * (1.5 - hx * y * y);
    y = y * (1.5 - hx * y * y);
    y = y * (1.5 - hx * y * y);
    y * (1.5 - hx * y * y)
}

/// The exact path's length and inverse length of a beam whose squared length is
/// `q` (`rubble_exact_lengths`, spec §33): ([`exact_inv_sqrt`]'s y, and the length
/// q·y with one Newton correction through its exact residual,
/// ℓ ← fma(fma(−ℓ, ℓ, q), ½y, ℓ)). The kernel runs the same sequence
/// (`sExactLength`); the length is √q correctly rounded (tests/energy_ledger.rs).
pub fn exact_length(q: f32) -> (f32, f32) {
    let y = exact_inv_sqrt(q);
    let l = q * y;
    ((-l).mul_add(l, q).mul_add(0.5 * y, l), y)
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

/// eˣ by one specified sequence the CPU and the kernel (`sExactExp`) run identically
/// (jbeam-semantics-spec.md §36): k = ⌊x·log₂e + ½⌋, r = (x − k·ln2_hi) − k·ln2_lo
/// (|r| ≤ 0.35), eʳ by its Taylor series to r⁷ (truncation below 6·10⁻⁹), Horner,
/// times 2ᵏ by its bits. 0 below −87, 3.4·10³⁸ above 88 (no subnormals: GPUs flush
/// them). Within 2 ulp of the exact value (`exact_exp_is_the_exponential`).
pub fn exact_exp(x: f32) -> f32 {
    if x < -87.0 {
        return 0.0;
    }
    if x > 88.0 {
        return 3.4e38;
    }
    let k = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = EXP_C[7];
    for i in (0..7).rev() {
        p = p * r + EXP_C[i];
    }
    p * f32::from_bits(((k as i32 + 127) as u32) << 23)
}

/// ln 2 split for [`exact_exp`] and [`exact_ln`]: 0.693359375 (exact in 10 bits)
/// plus the rest.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
/// 1/k!, k = 0..7.
const EXP_C: [f32; 8] = [
    1.0,
    1.0,
    0.5,
    0.166_666_67,
    0.041_666_668,
    0.008_333_334,
    0.001_388_888_9,
    0.000_198_412_7,
];

/// ln x by one specified sequence (the kernel's `sExactLn`): x = m·2ᵉ with
/// m ∈ [√½, √2), s = (m − 1)/(m + 1), ln m = 2s·Σ s²ⁱ/(2i + 1) to s¹⁰ (truncation
/// below 2·10⁻¹⁰), then e·ln2_hi + (ln m + e·ln2_lo). x below the smallest normal
/// (zero, negative or not a number): −1000 (no NaN or infinity literals, which WGSL
/// rejects; exp of it is 0).
pub fn exact_ln(x: f32) -> f32 {
    if !(x >= f32::MIN_POSITIVE) {
        return -1000.0;
    }
    let b = x.to_bits();
    let mut e = ((b >> 23) as i32 - 127) as f32;
    let mut m = f32::from_bits((b & 0x007F_FFFF) | 0x3F80_0000);
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1.0;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut q = 1.0 / 11.0f32;
    for c in [1.0 / 9.0f32, 1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0, 1.0] {
        q = q * s2 + c;
    }
    let lnm = (s + s) * q;
    e * LN2_HI + (lnm + e * LN2_LO)
}

/// xʸ (x ≥ 0) as exp(y·ln x) by [`exact_exp`] and [`exact_ln`]: 0 for x = 0 and
/// y > 0, 1 for y = 0.
pub fn exact_pow(x: f32, y: f32) -> f32 {
    if y == 0.0 {
        return 1.0;
    }
    if x == 0.0 && y > 0.0 {
        return 0.0;
    }
    exact_exp(y * exact_ln(x))
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
    contact_response(f, v, m, n, h, surface, p, friction).0
}

/// [`contact_force`] and, on a [`ContactSurface::dissipative`] surface (§37), the
/// energy its normal force takes out of the pair this step (J, ≥ 0): the budget of
/// the penetration's position step ([`shift_depth`]). RoR's rule: (force, 0).
///
/// §37's normal force stops an approach (vn′ = 0) and, separating, at most
/// cancels the inward force with vn′ ≥ −vn: r = −fn − (vn + max(vn, 0))·m/h. Its
/// work r·(vn + vn′)/2·h is −(½·m·vn² + ½·h·fn·vn) approaching and 0 otherwise, so
/// it does none positive and makes no separating speed. Friction is then capped
/// ([`no_positive_work`]).
#[allow(clippy::too_many_arguments)]
pub fn contact_response(
    f: Vec3,
    v: Vec3,
    m: f32,
    n: Vec3,
    h: f32,
    surface: &ContactSurface,
    p: f32,
    friction: f32,
) -> (Vec3, f32) {
    if p < 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    let vn = v.dot(n);
    let fn_ = f.dot(n);
    let mut r = -fn_;
    let mut budget = 0.0;
    if surface.dissipative {
        r = -fn_ - (vn + vn.max(0.0)) * m / h;
        if vn < 0.0 {
            budget = (0.5 * m * (vn * vn) + 0.5 * h * (fn_ * vn)).max(0.0);
        }
    } else if vn < 0.0 {
        r -= (0.8 * vn - 0.2 * p / h) * m / h;
    }
    if r <= 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    let c = contact_friction(r, f, v, n, surface, friction);
    if surface.dissipative {
        (no_positive_work(c, r, n, f, v, m, h), budget)
    } else {
        (c, 0.0)
    }
}

/// Surface contact's response (jbeam-semantics-spec.md §38.8): §37's dissipative normal
/// force r (the approach stopped, no separation made) and friction with a true stick. The
/// tangential force that stops the pair's slip within the step, −(f_t + v_t·m/h), is taken
/// whole while it is within the static limit μs·r (μs = `surface.ms`·friction·strength);
/// past it the pair slides on RoR's kinetic curve ([`contact_friction`]'s sliding branch:
/// from μs at no slip toward `surface.mc`, plus `t2`·σ) against its slip (against the
/// tangential force when it has none). Then [`no_positive_work`]. RoR's law
/// ([`contact_response`]: the stick cancels the force only, and a slip under `va` = 3 m/s
/// meets a fraction 1 − e^(−σ/va) of the limit) let a panel held on a bar by its skin creep
/// down a 25° slope and a car slide along an oblique pole; node contact's interlock (a
/// node inside the shape) hid it. Returns the force and the normal force's budget.
#[allow(clippy::too_many_arguments)]
pub fn surface_response(
    f: Vec3,
    v: Vec3,
    m: f32,
    n: Vec3,
    h: f32,
    surface: &ContactSurface,
    p: f32,
    friction: f32,
) -> (Vec3, f32) {
    if p < 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    let vn = v.dot(n);
    let fn_ = f.dot(n);
    let r = -fn_ - (vn + vn.max(0.0)) * m / h;
    let budget = if vn < 0.0 {
        (0.5 * m * (vn * vn) + 0.5 * h * (fn_ * vn)).max(0.0)
    } else {
        0.0
    };
    if r <= 0.0 {
        return (Vec3::ZERO, 0.0);
    }
    let ft = f - n * fn_;
    let vt = v - n * vn;
    let stick = -(ft + vt * (m / h));
    let gr = r * surface.strength * friction;
    let ms = surface.ms * gr;
    let c = if stick.dot(stick) <= ms * ms {
        n * r + stick
    } else {
        // Sliding: RoR's kinetic curve (contact_friction's), against the slip (against
        // the tangential force when there is none).
        let sl = vt.length();
        let dir = if sl > 0.0 {
            -(vt / sl)
        } else {
            stick / stick.length()
        };
        let e = if surface.exact {
            exact_exp(-exact_pow(sl / surface.vs, surface.alpha))
        } else {
            approx_exp(-approx_pow(sl / surface.vs, surface.alpha))
        };
        let g = surface.mc + (surface.ms - surface.mc) * e;
        n * r + dir * ((g + (surface.t2 * sl).min(5.0)) * gr)
    };
    let capped = no_positive_work(c, r, n, f, v, m, h);
    (capped, budget)
}

/// §37: the penetration's position step, the pair's separation δ along the
/// normal this step (m), from the depth `p` in the band, the `budget` the step's
/// normal force took out of the pair ([`contact_response`]) and the pair's normal
/// force `fn` (from the forces at the pass's start). RoR's push moves the pair
/// 0.2·p apart per step as speed that stays; here δ = 0.2·p is a move without
/// speed, as a position impulse P = δ·m shared like the hit's force (each node by
/// its sub-body's inverse mass). Against an inward force (fn < 0) it stores |fn|·δ
/// (first order), capped to the budget: δ ≤ budget/|fn|; with fn ≥ 0 the move
/// releases energy and is not capped.
pub fn shift_depth(p: f32, budget: f32, fn_: f32) -> f32 {
    let full = 0.2 * p;
    if fn_ >= 0.0 || -fn_ * full <= budget {
        full
    } else {
        budget / -fn_
    }
}

/// [`no_positive_work`] for a pair whose response to a force is not isotropic: a dynamic
/// rigid body that turns (spec §38.4). `k(x)` is the pair's relative velocity change per
/// unit impulse x (the element's Σ w²·k/m plus the body's 1/M and (I⁻¹(a × x)) × a, its
/// mass split), so v′ = v + h·(f/m) + h·k(c). The normal part keeps its scalar rule (m is
/// the reduced mass along n: n·k(n) = 1/m); the friction t = c − r·n (r recomputed from
/// the same inputs as [`contact_response`]'s dissipative rule) is scaled by the largest
/// s ∈ [0, 1] for which (r·n + s·t)·(u + h·k(r·n + s·t)) ≤ 0, u = 2v + h·f/m. With an
/// isotropic k (1/m) it is no_positive_work's cap; with a turning body the friction no
/// longer drives the contact point's slip past zero and spins the body up (a 6 cm steel
/// ball at 100 m/s into a door took 3,500 rad/s from one hit: more energy than it had).
/// Applied after contact_response's own cap, so the friction is the lesser of both.
#[allow(clippy::too_many_arguments)]
pub fn no_positive_work_body(
    c: Vec3,
    f: Vec3,
    v: Vec3,
    m: f32,
    n: Vec3,
    h: f32,
    k: impl Fn(Vec3) -> Vec3,
) -> Vec3 {
    if m <= 0.0 {
        return c;
    }
    let vn = v.dot(n);
    let fn_ = f.dot(n);
    let r = -fn_ - (vn + vn.max(0.0)) * m / h;
    if r <= 0.0 {
        return c;
    }
    let hm = h / m;
    let rn = n * r;
    let t = c - rn;
    let u = v + v + f * hm;
    let krn = k(rn);
    let a0 = rn.dot(u) + h * rn.dot(krn);
    let a1 = t.dot(u) + 2.0 * h * t.dot(krn);
    let b = h * t.dot(k(t));
    if a0 + a1 + b <= 0.0 || b <= 0.0 {
        return c;
    }
    let disc = a1 * a1 - 4.0 * b * a0;
    let s = if disc < 0.0 {
        0.0
    } else {
        ((disc.sqrt() - a1) / (2.0 * b)).clamp(0.0, 1.0)
    };
    rn + t * s
}

/// §37: contact force `c` (normal part r·n, r > 0, n unit) on a pair of relative
/// velocity `v`, reduced mass `m`, under force `f`, with its tangential (friction)
/// part t = c − r·n scaled by the largest s ∈ [0, 1] for which the force's work
/// over the step at the pair's mean velocity, (r·n + s·t)·(v + v′)/2·h with
/// v′ = v + h·(f + r·n + s·t)/m, is ≤ 0. That work is ∝ a0 + a1·s + b·s² with
/// u = 2v + h·f/m, a0 = r·n·u + h·r²/m (≤ 0 by the normal rule), a1 = t·u +
/// 2h·(r·n)·t/m, b = h·|t|²/m; s is its positive root. (Friction's stick term
/// cancels the tangential force like RoR's normal term, and both do positive work
/// while the beams decelerate the pair.) The normal part is left whole: it holds
/// the pair apart.
pub fn no_positive_work(c: Vec3, r: f32, n: Vec3, f: Vec3, v: Vec3, m: f32, h: f32) -> Vec3 {
    if m <= 0.0 {
        return c;
    }
    let hm = h / m;
    let rn = n * r;
    let t = c - rn;
    let u = v + v + f * hm;
    let a0 = rn.dot(u) + rn.dot(rn) * hm;
    let a1 = t.dot(u) + 2.0 * hm * rn.dot(t);
    let b = t.dot(t) * hm;
    let (mut s, mut whole) = (1.0f32, true);
    if a0 + a1 + b > 0.0 && b > 0.0 {
        whole = false;
        let disc = a1 * a1 - 4.0 * b * a0;
        s = if disc < 0.0 {
            0.0
        } else {
            ((disc.sqrt() - a1) / (2.0 * b)).clamp(0.0, 1.0)
        };
    }
    // Nor does friction reverse the slip: the slip vt after the step, vt′ = vt +
    // h·(f + s·t)/m along vt, stays ≥ 0 (at zero work a sliding pair could swap
    // vt for −vt each step for good: a period-two chatter).
    let vt = v - n * v.dot(n);
    let g = vt.dot(vt) + hm * f.dot(vt);
    let d = hm * t.dot(vt);
    if d < 0.0 && g + d * s < 0.0 {
        s = (g / -d).clamp(0.0, s);
        whole = false;
    }
    if whole {
        c
    } else {
        rn + t * s
    }
}

/// The contact force of normal force `r` > 0 along `n` with its friction
/// ([`contact_force`]'s friction law), on a node of force `f` and velocity `v`.
pub fn contact_friction(
    r: f32,
    f: Vec3,
    v: Vec3,
    n: Vec3,
    surface: &ContactSurface,
    friction: f32,
) -> Vec3 {
    let vn = v.dot(n);
    let fn_ = f.dot(n);
    let sf = f - fn_ * n;
    let mut s = v - vn * n;
    let sigma = s.length();
    if sigma > 0.0 {
        s /= sigma;
    }
    let gr = r * surface.strength * friction;
    let ms = surface.ms * gr;
    if sigma < surface.va && gr > 0.0 && sf.length_squared() <= ms * ms {
        let e = if surface.exact {
            exact_exp(-sigma / surface.va)
        } else {
            approx_exp(-sigma / surface.va)
        };
        r * n - ms * (1.0 - e) * s - sf
    } else {
        let e = if surface.exact {
            exact_exp(-exact_pow(sigma / surface.vs, surface.alpha))
        } else {
            approx_exp(-approx_pow(sigma / surface.vs, surface.alpha))
        };
        let g = surface.mc + (surface.ms - surface.mc) * e;
        r * n - (g + (surface.t2 * sigma).min(5.0)) * gr * s
    }
}

/// The tread contact's normal force law (spec §34, rate independent): at depth
/// `p` (> 0), its depth the step before `p_prev` (0 when it was off the ground) and
/// hysteretic deflection `z`, a node of mass `m` is pushed out with
/// k·(p + z′), k = m·[`TREAD_K`], and z′ is the new deflection. z follows the
/// depth with slope μ ([`TREAD_GROW`]) while |z| grows and λ ([`TREAD_REVERSE`])
/// after a reversal until it passes zero (the rest of that step's travel again at
/// μ): a closed loop for every cycle, centred on z = 0, of loss factor
/// [`TREAD_TAN_DELTA`] whatever its amplitude. z′ ≥ −p (no pull). Returns (force,
/// z′). Strict IEEE single arithmetic in this order (the kernel's `treadHysteresis`).
#[inline]
pub fn tread_hysteresis(z: f32, p: f32, p_prev: f32, m: f32) -> (f32, f32) {
    let dp = p - p_prev;
    let mut z1 = if z * dp >= 0.0 {
        z + TREAD_GROW * dp
    } else {
        let t = z + TREAD_REVERSE * dp;
        if (t > 0.0) != (z > 0.0) {
            TREAD_GROW * (dp + z * TREAD_INV_REVERSE)
        } else {
            t
        }
    };
    if z1 < -p {
        z1 = -p;
    }
    ((m * TREAD_K) * (p + z1), z1)
}

/// The contact-force log (jbeam-semantics-spec.md §32). A node's contact force in
/// a step is the sum of the magnitudes of every contact force on it that step:
/// the node pass's (ground, boxes, static shapes, rigid bodies), self-collision's
/// and vehicle contact's, as the contacter and as a triangle's node (its share
/// w·|F|). Sums are fixed point, 1/[`crate::explicit_rigid::FIXED`] N, as the
/// kernel's accumulators. A step's sum is complete at the next node pass, which
/// moves it to `last` (and the running peak) and tests the contact breaks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContactForceLog {
    /// This step's sums so far.
    pub sum: Vec<i32>,
    /// The last complete step's sums.
    pub last: Vec<i32>,
    /// The largest complete step's sum since [`ExplicitBeams::take_contact_force_peak`].
    pub peak: Vec<i32>,
    /// Per node its contact break (break group, force N; group 0: none).
    pub node_break: Vec<(u32, f32)>,
    /// Per break group (by |group|): 0 idle, 1 set off by the last step's contact
    /// force (it breaks in this beam pass, as if by its first beam), 2 done.
    pub pending: Vec<u8>,
}

impl ContactForceLog {
    /// Adds contact force `f` (its magnitude) to node `i`'s sum.
    #[inline]
    pub fn add(&mut self, i: usize, f: Vec3) {
        let m = ((f.x * f.x + f.y * f.y) + f.z * f.z).sqrt();
        self.add_magnitude(i, m);
    }

    /// Adds a magnitude `m` (N) to node `i`'s sum.
    #[inline]
    pub fn add_magnitude(&mut self, i: usize, m: f32) {
        let q = (m * crate::explicit_rigid::FIXED).clamp(-2.0e9, 2.0e9).round() as i32;
        self.sum[i] = self.sum[i].wrapping_add(q);
    }

    /// Node `i` at the node pass: its last step's sum becomes `last` (and the
    /// peak), its contact break is tested, and its sum restarts.
    #[inline]
    fn roll(&mut self, i: usize) {
        let s = self.sum[i];
        self.last[i] = s;
        self.peak[i] = self.peak[i].max(s);
        self.sum[i] = 0;
        let (g, limit) = self.node_break[i];
        if g != 0 && s as f32 * (1.0 / crate::explicit_rigid::FIXED) > limit {
            let p = &mut self.pending[g as usize];
            *p = (*p).max(1);
        }
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
            shapes: Vec::new(),
            origin: Vec3::ZERO,
            ground: 0.0,
            collision_range: DEFAULT_COLLISION_RANGE,
            last_contacts: 0,
            last_self_pairs: 0,
            tyres: Vec::new(),
            brakes: Vec::new(),
            parking_brake: 0.0,
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
            self_groups: Vec::new(),
            tri_excl: Vec::new(),
            contact_every_node: false,
            rest_pose: None,
            two_body_contact: false,
            anti_clip: false,
            own_travel_reach: false,
            retired: Vec::new(),
            clip_sides: Vec::new(),
            anti_only_tris: Vec::new(),
            fixed_contact: false,
            dissipative_contact: false,
            surface_contact: false,
            surface_last: 0,
            surface_queries: 0,
            surface_separation: true,
            surface_segs: None,
            surface_force: glam::DVec3::ZERO,
            contact_shift: Vec::new(),
            shift_applied: Vec::new(),
            ground_lift: Vec::new(),
            plastic_strength_drop: true,
            exact_friction: false,
            contact_acc: Vec::new(),
            slide_hold: Vec::new(),
            contact_exact: None,
            self_hit_log: None,
            self_contact_parts: None,
            contact_force: None,
            tread: Vec::new(),
            damp_filters: Vec::new(),
            torsion_watch: Vec::new(),
            groups_last: Vec::new(),
            groups_now: Vec::new(),
            grab: None,
            beam_ext: Vec::new(),
            ext_of: Vec::new(),
            tri_breaks: Vec::new(),
            tri_dead: Vec::new(),
            deform_state: Vec::new(),
            deform_names: Vec::new(),
            rigid: None,
            trace: None,
            drive: None,
            beamng_profile: false,
            sleep: Default::default(),
            damp_scheme: 0,
            damp_slopes: Vec::new(),
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
        Self::from_truck_yawed_tyres(
            text,
            config,
            position,
            yaw_degrees,
            velocity,
            settings,
            airfoil,
            true,
        )
    }

    /// [`Self::from_truck_yawed`], building the `wheels` sections' soft tyres
    /// (`tyres`) or not: without them each wheel's mass goes to its two axle nodes
    /// (half each), for raycast wheels ([`crate::explicit_drive`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_truck_yawed_tyres(
        text: &str,
        config: Option<&str>,
        position: Vec3,
        yaw_degrees: f32,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
        airfoil: &dyn Fn(&str) -> Option<String>,
        tyres: bool,
    ) -> Result<Self, String> {
        let file = parse_truck_config(text, config);
        if !file.errors.is_empty() {
            return Err(file.errors.join("; "));
        }
        // Collision triangles take the concrete surface; a named one is not
        // supported yet (refused rather than run on concrete).
        if let Some(g) = file
            .submesh_groundmodel
            .as_deref()
            .filter(|g| *g != "concrete")
        {
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
            for (i, w) in file
                .wheels
                .iter()
                .enumerate()
                .filter(|(_, w)| w.kind == kind)
            {
                tyre_of[i] = spawn_order.len() as u32;
                spawn_order.push(w);
            }
        }
        // A turned spawn builds its tyres in the file's axes, then turns them like
        // every other node.
        let yawed = yaw_degrees != 0.0;
        let mut local: Vec<Vec3> = if yawed {
            desc.nodes
                .iter()
                .map(|n| Vec3::from_array(n.position))
                .collect()
        } else {
            Vec::new()
        };
        for (t, w) in spawn_order.into_iter().enumerate() {
            let file_wheel = tyre_of.iter().position(|&x| x as usize == t).unwrap_or(t) as u32;
            let posed = file.tyre_pose.iter().any(|p| p.0 == file_wheel);
            // (Rubble's `rubble_tyre_pose`: a tyre's rest lengths are built
            // round its axle's rest positions.)
            let rest = file.tyre_rest_axle.iter().find(|r| r.0 == file_wheel);
            let saved = [s.nodes[w.node1 as usize].pos, s.nodes[w.node2 as usize].pos];
            if let Some(&(_, r1, r2)) = rest {
                for (node, r) in [(w.node1, r1), (w.node2, r2)] {
                    let r = Vec3::from_array(r);
                    s.nodes[node as usize].pos =
                        position + if yawed { yaw_offset(r, yaw_degrees) } else { r };
                    if yawed {
                        local[node as usize] = r;
                    }
                }
            }
            if tyres {
                s.add_tyre(w, position, yaw_degrees, velocity, &local, posed);
            } else {
                s.nodes[w.node1 as usize].mass += 0.5 * w.mass;
                s.nodes[w.node2 as usize].mass += 0.5 * w.mass;
                s.add_hub_rigidity(w.node1, w.node2, w.rigidity, w.spring, w.damp);
            }
            if rest.is_some() {
                s.nodes[w.node1 as usize].pos = saved[0];
                s.nodes[w.node2 as usize].pos = saved[1];
                if yawed {
                    local[w.node1 as usize] =
                        Vec3::from_array(desc.nodes[w.node1 as usize].position);
                    local[w.node2 as usize] =
                        Vec3::from_array(desc.nodes[w.node2 as usize].position);
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
            crate::truck::TruckNodeRef::Tyre(w, k) => {
                s.tyres[tyre_of[w as usize] as usize].first + k
            }
        };
        let mut tyre_beams = file.tyre_beams.iter().peekable();
        for class in 0..8 {
            for (i, b) in rest.iter().enumerate().filter(|&(i, _)| order(i) == class) {
                while class == 0 && tyre_beams.peek().is_some_and(|t| t.position <= i) {
                    let t = tyre_beams.next().unwrap();
                    let (a, b) = (tyre_node(&s, t.a), tyre_node(&s, t.b));
                    s.push_desc_beam(&crate::desc::SoftBeam {
                        a,
                        b,
                        ..t.beam.clone()
                    });
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
                s.push_desc_beam(&crate::desc::SoftBeam {
                    a,
                    b,
                    ..t.beam.clone()
                });
            }
        }
        // Rubble's `rubble_tyre_pose`: tyre nodes spawn where the file puts
        // them (the tyre at rest, deflected), their beams keeping the
        // undeflected tyre's rest lengths; `rubble_rest_pose <y>`: the file is
        // the vehicle at rest on ground at `y`.
        for &(w, k, p) in &file.tyre_pose {
            let Some(t) = tyre_of
                .get(w as usize)
                .and_then(|&t| s.tyres.get(t as usize))
            else {
                continue;
            };
            if k < t.count {
                let p = Vec3::from_array(p);
                let i = (t.first + k) as usize;
                s.nodes[i].pos = position + if yawed { yaw_offset(p, yaw_degrees) } else { p };
            }
        }
        if let Some(g) = file.rest_ground {
            let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            s.rest_pose = Some((position.y + g - low).max(0.0));
        }
        s.add_slidenodes(&file);
        // The BeamNG profile's band (§36) unless the file sets its own.
        s.collision_range = if file.beamng_profile && !file.collision_range_set {
            BEAMNG_COLLISION_RANGE
        } else {
            desc.collision_range
        };
        s.add_triangles(&desc);
        s.add_aero(&file, airfoil);
        // The BeamNG profile's air drag (§31, §36): the triangles' (every cab
        // triangle at BeamNG's defaults: dragCoef 100, liftCoef 100, stall
        // 0.58 rad), replacing the per-node drag and turbulence.
        if file.beamng_profile && s.settings.drag {
            // (A cab line of both triangle groups is two collision triangles on
            // the same nodes: one aero triangle.)
            let mut aero = desc.clone();
            let mut seen = std::collections::HashSet::new();
            aero.triangles.retain(|t| {
                let mut k = t.nodes;
                k.sort_unstable();
                seen.insert(k)
            });
            if let Some((d, l)) = file.drag_coef {
                for t in &mut aero.triangles {
                    t.aero[0] = d;
                    t.aero[1] = l;
                }
            }
            // `rubble_aero_tri` (Rubble's extension, §36): per triangle; unlisted
            // ones none.
            if !file.aero_tri.is_empty() {
                let key = |mut k: [u32; 3]| {
                    k.sort_unstable();
                    k
                };
                let mut at: HashMap<[u32; 3], usize> = HashMap::new();
                for (i, t) in aero.triangles.iter().enumerate() {
                    at.insert(key(t.nodes), i);
                }
                for t in &mut aero.triangles {
                    t.aero[0] = 0.0;
                    t.aero[1] = 0.0;
                }
                for &(n, d, l) in &file.aero_tri {
                    let Some(&i) = at.get(&key(n)) else {
                        return Err(format!("rubble_aero_tri: {n:?} is not a cab triangle"));
                    };
                    aero.triangles[i].aero[0] = d;
                    aero.triangles[i].aero[1] = l;
                }
            }
            s.add_triangle_aero(&aero);
            if !s.beam_ext.is_empty() {
                s.ext_of.resize(s.beams.len(), 0);
            }
            s.settings.drag = false;
            s.settings.turbulence = false;
        }
        s.finish();
        // Rubble's extensions (none in Rigs of Rods files: unchanged): cab
        // lines that break (spec §18) and contact from every node (§10.2).
        s.add_truck_triangle_breaks(&desc);
        if file.full_contact {
            s.opt_in_full_contact();
        }
        // `rubble_exact_lengths` (Rubble's extension): exact beam lengths.
        if file.exact_lengths {
            s.settings.fast_inv_sqrt = false;
        }
        // `rubble_beamng_profile` (Rubble's extension, spec §36): the BeamNG
        // profile's switches and two-body vehicle contact.
        if file.beamng_profile {
            s.beamng_profile = true;
            s.settings = s.settings.beamng();
            s.two_body_contact = true;
            s.anti_clip = true;
            s.fixed_contact = true;
            s.dissipative_contact = true;
            s.exact_friction = true;
        }
        // `rubble_damper_tear` (Rubble's extension): a damper-only beam tears when it
        // is stretched past the ratio (its `long_bound`, as a support's break factor).
        if let Some(r) = file.damper_tear.filter(|r| *r > 0.0) {
            for b in &mut s.beams {
                if b.kind == ExplicitKind::Plain && b.k == 0.0 && b.c > 0.0 {
                    b.long_bound = r;
                }
            }
        }
        // `rubble_two_body_contact` (Rubble's extension, spec §10.3): the two-body
        // rule in vehicle and self contact, the profile otherwise unchanged (RoR's
        // normal force and ground contact).
        if file.two_body_contact {
            s.two_body_contact = true;
        }
        // `rubble_contact_breaks` (Rubble's extension, spec §32).
        if !file.contact_breaks.is_empty() {
            s.set_contact_breaks(&file.contact_breaks)?;
        }
        // `rubble_jbeam_plastic` (Rubble's extension, spec §3).
        if !file.jbeam_plastic.is_empty() {
            s.set_jbeam_plastic(&file.jbeam_plastic)?;
        }
        // `rubble_tension_yield` (Rubble's extension, spec §18.4).
        if !file.tension_yield.is_empty() {
            s.set_tension_yield(&file.tension_yield)?;
        }
        if !file.spawn_contacts.is_empty() {
            let sorted = |t: [u32; 3]| {
                let mut t = t;
                t.sort_unstable();
                t
            };
            let at: HashMap<[u32; 3], u32> = s
                .triangles
                .iter()
                .enumerate()
                .map(|(i, t)| (sorted(t.nodes), i as u32))
                .collect();
            let keep: std::collections::HashSet<(u32, u32)> = file
                .spawn_contacts
                .iter()
                .filter_map(|&(t, c)| at.get(&sorted(t)).map(|&i| (i, c)))
                .collect();
            s.tri_excl.clear();
            s.spawn_exclusions_except(&keep);
        }
        // The BeamNG profile's plasticity (§36): JBeam's NORMAL rule (§3) on every
        // deformable plain beam, without RoR's strength drop.
        if file.beamng_profile {
            s.use_jbeam_plasticity();
        }
        // `rubble_damp_cutoff` (Rubble's extension, §36): JBeam's dampCutoffHz on
        // the profile's JBeam beams.
        if !file.damp_cutoff.is_empty() {
            s.set_damp_cutoff(&file.damp_cutoff)?;
        }
        Ok(s)
    }

    /// JBeam's `dampCutoffHz` on beams of a `.truck` (`rubble_damp_cutoff`, spec
    /// §36): per line (a, b, hz) the JBeam beam a–b (a BeamNG-profile file's plain
    /// beams are) damps its extension rate filtered at hz.
    pub fn set_damp_cutoff(&mut self, lines: &[(u32, u32, f32)]) -> Result<(), String> {
        let mut at: HashMap<(u32, u32), usize> = HashMap::new();
        for (j, b) in self.beams.iter().enumerate() {
            if matches!(b.kind, ExplicitKind::JbeamPlain | ExplicitKind::JbeamRope) {
                at.entry((b.a.min(b.b), b.a.max(b.b))).or_insert(j);
            }
        }
        for &(a, b, hz) in lines {
            let Some(&j) = at.get(&(a.min(b), a.max(b))) else {
                return Err(format!(
                    "rubble_damp_cutoff: no JBeam beam {a}–{b} (the BeamNG profile's plain beams)"
                ));
            };
            if self.beams[j].shock != 0 {
                return Err(format!("rubble_damp_cutoff: beam {a}–{b} listed twice"));
            }
            self.damp_filters.push(DampFilter::new(hz));
            self.beams[j].shock = self.damp_filters.len() as u32;
        }
        Ok(())
    }

    /// The BeamNG profile's plasticity for `.truck` content (jbeam-semantics-spec.md
    /// §36): every deformable plain (RoR) beam takes JBeam's NORMAL rule (§3,
    /// `JbeamPlain`) with no deform limits and no `deformLimitStress` cap (JBeam's
    /// defaults: none), keeping the file's plastic coefficient; and no beam's
    /// strength falls with its plastic stretch (`plastic_strength_drop` off).
    pub fn use_jbeam_plasticity(&mut self) {
        for b in &mut self.beams {
            if b.ty != ExplicitType::Deformable {
                continue;
            }
            let kind = match b.kind {
                ExplicitKind::Plain => ExplicitKind::JbeamPlain,
                ExplicitKind::Rope => ExplicitKind::JbeamRope,
                _ => continue,
            };
            b.kind = kind;
            b.limit_k = f32::INFINITY;
            b.short_bound = 0.0;
            b.long_bound = f32::INFINITY;
        }
        self.plastic_strength_drop = false;
    }

    /// One `wheels` / `meshwheels` / `meshwheels2` tyre (its nodes after the
    /// existing ones, then its beams), spawned at `position` turned by
    /// `yaw_degrees` (`local`: the file's node positions, for a turned spawn).
    fn add_tyre(
        &mut self,
        w: &crate::truck::TruckWheel,
        position: Vec3,
        yaw_degrees: f32,
        velocity: Vec3,
        local: &[Vec3],
        posed: bool,
    ) {
        let yawed = yaw_degrees != 0.0;
        let at = |s: &Self, i: u32| {
            if yawed {
                local[i as usize]
            } else {
                s.nodes[i as usize].pos
            }
        };
        let place = |p: Vec3| {
            if yawed {
                position + yaw_offset(p, yaw_degrees)
            } else {
                p
            }
        };
        // meshwheels2: the axle-to-own-ring spokes harden only past 0.15 L of
        // stretch.
        let spoke_long = if w.kind == "meshwheels2" {
            0.15f32
        } else {
            0.0
        };
        // The axle runs toward +z: the node of lower z is A1, whichever the
        // file names first (equal z keeps the file's order). A tyre the file
        // poses (Rubble's `rubble_tyre_pose`) keeps the file's order: its pose
        // names its nodes as built from its first axle node.
        let (a1, a2) = if !posed && at(self, w.node1).z > at(self, w.node2).z {
            (w.node2, w.node1)
        } else {
            (w.node1, w.node2)
        };
        let pa1 = at(self, a1);
        let pa2 = at(self, a2);
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
        let first = self.nodes.len() as u32;
        self.tyres.push(Tyre {
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
            self.nodes.push(tyre(place(pa1 + spoke)));
            spoke = rotate(spoke);
            self.nodes.push(tyre(place(pa2 + spoke)));
            spoke = rotate(spoke);
        }
        let outer = |i: u32| first + 2 * (i % n);
        let inner = |i: u32| first + 2 * (i % n) + 1;
        let group = signed_group(&w.beam);
        let deform = w.beam.deform;
        let strength = w.beam.strength;
        let plastic = w.beam.plastic_coef;
        let rigidity = w.rigidity.map(|p| {
            let pp = at(self, p);
            (p, pp.distance_squared(pa1) < pp.distance_squared(pa2))
        });
        for i in 0..n {
            let add =
                |s: &mut Self, a: u32, b: u32, kind: ExplicitKind, ty: ExplicitType, rim: bool| {
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
            add(self, a1, o, lim, def, false);
            add(self, a2, ii, lim, def, false);
            add(self, a2, o, plain, def, false);
            add(self, a1, ii, plain, def, false);
            add(self, o, ii, plain, def, true);
            add(self, o, outer(i + 1), plain, def, true);
            add(self, ii, inner(i + 1), plain, def, true);
            add(self, ii, outer(i + 1), plain, def, true);
            if let Some((p, near_a1)) = rigidity {
                let end = if near_a1 { o } else { ii };
                add(self, p, end, plain, ExplicitType::Virtual, false);
            }
        }
    }

    /// A wheel without its tyre (raycast wheels): the tyre's rigidity beams tied its
    /// rigidity (arm) node to the tyre, which holds the axle; without the tyre the
    /// arm is tied to the two axle nodes directly (tyre-rigidity beams, the tyre's
    /// spring and damping, never yielding), so the wheel keeps its toe and camber
    /// as the tyre kept them. No arm: nothing.
    fn add_hub_rigidity(&mut self, a1: u32, a2: u32, arm: Option<u32>, k: f32, c: f32) {
        let Some(p) = arm else {
            return;
        };
        for a in [a1, a2] {
            if a == p {
                continue;
            }
            let rest = self.nodes[a as usize].pos.distance(self.nodes[p as usize].pos);
            let beam = Beam::new(
                p,
                a,
                ExplicitKind::Plain,
                ExplicitType::Virtual,
                k,
                c,
                rest,
                f32::INFINITY,
                f32::INFINITY,
                0.0,
            );
            self.beams.push(beam);
        }
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
            let x = Vec3::ZERO + self.nodes[sn.node as usize].pos;
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

    /// The nearest point of segment `seg` to point `x` (in the vehicle's frame: relative
    /// to its origin), its distance and its ratio along the segment.
    fn segment_distance(&self, seg: u32, x: Vec3) -> (Vec3, f32, f32) {
        let b = &self.beams[self.segments[seg as usize].beam as usize];
        let p1 = Vec3::ZERO + self.nodes[b.a as usize].pos;
        let p2 = Vec3::ZERO + self.nodes[b.b as usize].pos;
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
    fn add_aero(
        &mut self,
        file: &crate::truck::TruckFile,
        airfoil: &dyn Fn(&str) -> Option<String>,
    ) {
        let table = |s: &mut Self, name: &str| -> u32 {
            if let Some(i) = s.airfoils.iter().position(|a| a.name == name) {
                return i as u32;
            }
            let a = match airfoil(name) {
                Some(text) => Airfoil::parse(name, &text),
                None => {
                    s.warnings
                        .push(format!("airfoil `{name}` not found: its tables are zero"));
                    Airfoil::zero(name)
                }
            };
            s.airfoils.push(Arc::new(a));
            s.airfoils.len() as u32 - 1
        };
        let x: Vec<Vec3> = self.nodes.iter().map(|n| self.origin + n.pos).collect();
        for w in &file.wings {
            let t = table(self, &w.airfoil);
            self.wings.push(Wing::new(
                w.nodes,
                w.efficacy,
                w.chord_point,
                w.control,
                t,
                &x,
            ));
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
    /// conversion's warnings. This is the Rigs of Rods reading of the vehicle (what
    /// RoR itself runs); [`Self::from_jbeam_native`] runs it with BeamNG's rules.
    pub fn from_jbeam(
        files: &[(&str, &str)],
        config: Option<&str>,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, String, Vec<String>), String> {
        let desc = crate::jbeam::load_jbeam(files, config)?;
        let (text, warnings) = crate::truck_export::desc_to_truck(
            &desc,
            crate::truck_export::TyreDefaults::for_vehicle(&desc),
        );
        let sim = Self::from_truck(&text, position, velocity, settings)?;
        Ok((sim, text, warnings))
    }

    /// Spawn a BeamNG JBeam vehicle (`files`: name and text of each `.jbeam`,
    /// `config` an optional `.pc`) in JBeam mode ([`Self::from_desc_jbeam`]), with
    /// [`crate::truck_export::TyreDefaults::for_vehicle`] for its wheels. Also
    /// returns the loader's and the spawn's warnings (what is not simulated).
    pub fn from_jbeam_native(
        files: &[(&str, &str)],
        config: Option<&str>,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, Vec<String>), String> {
        let desc = crate::jbeam::load_jbeam(files, config)?;
        let tyres = crate::truck_export::TyreDefaults::for_vehicle(&desc);
        Self::from_desc_jbeam(&desc, tyres, position, velocity, settings)
    }

    /// Spawn a JBeam-loaded desc in JBeam mode (`engine/docs/jbeam-semantics-spec.md`):
    /// the desc's nodes and beams directly, in desc order, with BeamNG's rules where
    /// they differ from the `.truck` path's ([`ExplicitKind::JbeamPlain`],
    /// [`ExplicitKind::JbeamSupport`], [`ExplicitKind::JbeamBounded`]): passive
    /// break groups, precompression on every beam, supports that push only and
    /// break by length, bounded beams with their bound zone and limit damping,
    /// deform limits, no break exception, and self-collision for the nodes that
    /// ask for it. Wheels become the `.truck` path's soft tyres (`tyres`), after
    /// the desc's nodes and beams. Fails on more than [`MAX_COLLISION_TRIANGLES`]
    /// collision triangles. Returns the desc's warnings and what this spawn does
    /// not simulate.
    pub fn from_desc_jbeam(
        desc: &SoftVehicleDesc,
        tyres: crate::truck_export::TyreDefaults,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, Vec<String>), String> {
        Self::from_desc_jbeam_with(desc, JbeamWheels::Soft(tyres), position, velocity, settings)
    }

    /// [`Self::from_desc_jbeam`] with a choice of wheel model: the `.truck` path's
    /// soft tyres, or JBeam pressure wheels (`engine/docs/jbeam-semantics-spec.md`
    /// §25). Node order is the desc's, then the wheels'; beam order is the desc's
    /// beams, torsion elements, couplers, LBEAMs, then the wheels' beams, rail
    /// segment beams last.
    pub fn from_desc_jbeam_with(
        desc: &SoftVehicleDesc,
        wheels: JbeamWheels,
        position: Vec3,
        velocity: Vec3,
        settings: ExplicitBeamsSettings,
    ) -> Result<(Self, Vec<String>), String> {
        let collision_tris = desc.triangles.iter().filter(|t| t.collision).count();
        if collision_tris > MAX_COLLISION_TRIANGLES {
            return Err(format!(
                "{collision_tris} collision triangles: at most {MAX_COLLISION_TRIANGLES} per vehicle"
            ));
        }
        let mut warnings = desc.warnings.clone();
        // Exact beam lengths: the fast inverse square root is a Rigs of Rods
        // compatibility rule (its 0.2 % error strains stiff beams at rest).
        // JBeam content runs the BeamNG profile (§36; exact beam lengths among
        // its switches: the fast inverse square root is a Rigs of Rods rule whose
        // 0.2 % error strains stiff beams at rest).
        let settings = settings.beamng();
        let mut s = Self::spawn_nodes(desc, position, 0.0, velocity, settings);
        s.beamng_profile = true;
        for b in &desc.beams {
            s.push_jbeam_beam(b);
        }
        // Torsion elements break when a beam between consecutive nodes breaks: each
        // such beam's break group is watched (one of its own, numbered past the
        // file's groups, when it has none).
        let mut pair_beam: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, b) in s.beams.iter().enumerate() {
            pair_beam.entry((b.a.min(b.b), b.a.max(b.b))).or_insert(i);
        }
        let mut next_group = s
            .beams
            .iter()
            .map(|b| b.group.unsigned_abs())
            .chain(desc.elements.iter().map(|e| e.break_group))
            .chain(desc.couplers.iter().map(|c| c.break_group))
            .max()
            .unwrap_or(0);
        let mut unwatched = 0usize;
        for e in desc
            .elements
            .iter()
            .filter(|e| e.kind == crate::desc::ElementKind::Torsion)
        {
            let [n1, n2, n3, n4] = e.nodes;
            let mut watch = [0u32; 3];
            for (k, (p, q)) in [(n1, n2), (n2, n3), (n3, n4)].into_iter().enumerate() {
                let Some(&i) = pair_beam.get(&(p.min(q), p.max(q))) else {
                    continue;
                };
                let g = &mut s.beams[i].group;
                if *g == 0 {
                    next_group += 1;
                    *g = next_group as i32;
                } else if *g < 0 {
                    // A passive member's own break does not set off its group.
                    unwatched += 1;
                }
                watch[k] = g.unsigned_abs();
            }
            let x = [n1, n2, n3, n4].map(|n| s.nodes[n as usize].pos);
            let phi0 = wrap_angle(dihedral(x) + e.precompression_angle);
            let mut a = Beam::new(
                n1,
                n2,
                ExplicitKind::JbeamTorsion,
                ExplicitType::Actuator,
                e.spring,
                e.damp,
                phi0,
                f32::INFINITY,
                e.strength,
                0.0,
            );
            a.tc = e.deform;
            a.group = e.break_group as i32;
            s.torsion_watch.push(watch);
            a.shock = s.torsion_watch.len() as u32;
            let mut b = Beam::new(
                n3,
                n4,
                ExplicitKind::JbeamTorsionPair,
                ExplicitType::Actuator,
                0.0,
                0.0,
                0.0,
                f32::INFINITY,
                f32::INFINITY,
                0.0,
            );
            b.group = a.group;
            s.beams.push(a);
            s.beams.push(b);
        }
        if unwatched > 0 {
            warnings.push(format!(
                "{unwatched} torsion element links are passive break-group members: their \
                 own breaks do not break the torsion element"
            ));
        }
        for cp in &desc.couplers {
            let (k, c) = coupler_rates(s.nodes[cp.a as usize].mass, s.nodes[cp.b as usize].mass);
            let mut beam = Beam::new(
                cp.a,
                cp.b,
                ExplicitKind::JbeamCoupler,
                ExplicitType::Actuator,
                k,
                c,
                0.0,
                f32::INFINITY,
                cp.strength,
                0.0,
            );
            beam.group = if cp.break_group_passive {
                -(cp.break_group as i32)
            } else {
                cp.break_group as i32
            };
            s.beams.push(beam);
            if cp.extra_break_groups != [0, 0] && beam.group != 0 {
                // Several break groups (spec §7.1, §17): an extension record.
                s.attach_coupler_groups(s.beams.len() - 1, cp.extra_break_groups);
            }
        }
        s.add_lbeams(desc);
        let tyres = match wheels {
            JbeamWheels::Soft(t) => Some(t),
            JbeamWheels::Pressure => {
                let mut w = s.add_pressure_wheels(desc, velocity);
                warnings.append(&mut w);
                None
            }
            JbeamWheels::Raycast => {
                // Raycast wheels (crate::explicit_drive): no wheel nodes; each
                // wheel's mass goes to its axle nodes, its arm held as the tyre held it.
                let t = crate::truck_export::TyreDefaults::for_vehicle(desc);
                for w in &desc.wheels {
                    s.nodes[w.node1 as usize].mass += 0.5 * w.mass;
                    s.nodes[w.node2 as usize].mass += 0.5 * w.mass;
                    s.add_hub_rigidity(w.node1, w.node2, w.arm, t.spring, t.damp);
                }
                None
            }
        };
        if let (Some(tyres), false) = (tyres, desc.wheels.is_empty()) {
            // The `.truck` path's tyres, from the same text it writes.
            let (text, _) = crate::truck_export::desc_to_truck(desc, tyres);
            let file = parse_truck_config(&text, None);
            for w in &file.wheels {
                s.add_tyre(w, position, 0.0, velocity, &[], false);
            }
            warnings.push(format!(
                "{} pressure wheels simulated as {}-ray soft tyres ({} kg, {} N/m, {} N·s/m)",
                desc.wheels.len(),
                tyres.rays,
                tyres.mass,
                tyres.spring,
                tyres.damp
            ));
        }
        s.add_jbeam_rails(desc, &mut warnings);
        let stiff = s.stiff_dampers();
        if stiff > 0 {
            warnings.push(format!(
                "{stiff} beams damp harder than the 0.5 ms step integrates (c·h/μ > 1): \
                 backward-Euler damping c/(1 + c·h/μ) (jbeam-semantics-spec.md §28)"
            ));
        }
        for (n, what) in [(desc.rigid_parts.len(), "rigid parts")] {
            if n > 0 {
                warnings.push(format!(
                    "{n} {what} are not simulated by the explicit solver"
                ));
            }
        }
        s.collision_range = if desc.collision_range > 0.0 {
            desc.collision_range
        } else {
            DEFAULT_COLLISION_RANGE
        };
        s.add_triangles(desc);
        // NONCOLLIDABLE triangles: anti-clip only (§36), after the collision ones.
        s.add_anti_only_triangles(desc);
        s.add_triangle_breaks(desc, &mut warnings);
        if s.settings.drag {
            // JBeam mode's air drag is the triangles' (spec §31), not the nodes'.
            s.add_triangle_aero(desc);
            s.settings.drag = false;
            s.settings.turbulence = false;
        }
        s.deform_names = desc.deform_groups.clone();
        s.deform_state = vec![0; desc.deform_groups.len()];
        if !s.beam_ext.is_empty() {
            s.ext_of.resize(s.beams.len(), 0);
        }
        s.finish();
        // Self-collision: the nodes with `selfCollision` (and collision at all).
        s.candidates = (0..desc.nodes.len() as u32)
            .filter(|&i| {
                let n = &desc.nodes[i as usize];
                n.self_collision && n.collision && !n.fixed
            })
            .collect();
        s.grid_cell = s.default_grid_cell();
        if !desc.node_groups.is_empty() {
            s.self_groups = (0..s.nodes.len())
                .map(|i| desc.nodes.get(i).map_or(0, |n| n.self_group))
                .collect();
        }
        s.two_body_contact = true;
        s.anti_clip = true;
        s.fixed_contact = true;
        s.dissipative_contact = true;
        s.plastic_strength_drop = false;
        s.exact_friction = true;
        let pairs = s.spawn_exclusions();
        if pairs > 0 {
            warnings.push(format!(
                "{pairs} self-collision node-triangle pairs start inside the contact range: \
                 they never collide (spec §10.1)"
            ));
        }
        Ok((s, warnings))
    }

    /// A desc beam in JBeam mode (spec §2–5; Part II features through
    /// [`Self::attach_ext`]).
    pub(crate) fn push_jbeam_beam(&mut self, b: &crate::desc::SoftBeam) {
        let jb = b.jbeam.unwrap_or(crate::desc::JbeamBeam {
            deform: b.deform,
            precompression: b.precompression,
            ..Default::default()
        });
        let spawn = self.nodes[b.a as usize]
            .pos
            .distance(self.nodes[b.b as usize].pos);
        let rest = spawn * jb.precompression;
        let (kind, ty) = match b.kind {
            BeamKind::Bounded => (ExplicitKind::JbeamBounded, ExplicitType::Actuator),
            BeamKind::Support => (ExplicitKind::JbeamSupport, ExplicitType::Deformable),
            _ => (ExplicitKind::JbeamPlain, ExplicitType::Deformable),
        };
        let mut beam = Beam::new(
            b.a, b.b, kind, ty, b.spring, b.damp, rest, jb.deform, b.strength, 0.0,
        );
        beam.group = signed_group(b);
        if jb.damp_cutoff_hz > 0.0 {
            self.damp_filters.push(DampFilter::new(jb.damp_cutoff_hz));
            beam.shock = self.damp_filters.len() as u32;
        }
        match kind {
            ExplicitKind::JbeamBounded => {
                beam.limit_k = jb.limit_spring;
                beam.limit_c = jb.limit_damp;
                beam.short_bound = jb.short_bound;
                beam.long_bound = jb.long_bound;
                beam.plastic = jb.bound_zone;
                beam.tc = jb.damp_velocity_split;
                beam.tt = jb.damp_fast;
            }
            _ => {
                // The yield and break tests start at the lower of deform and strength.
                beam.gate = jb.deform.min(b.strength);
                beam.limit_k = b.deform_limit_stress;
                // Rest-length limits and the support's break length, in metres.
                beam.short_bound = if jb.deform_limit.is_finite() {
                    (1.0 - jb.deform_limit) * rest
                } else {
                    0.0
                };
                beam.long_bound = if kind == ExplicitKind::JbeamSupport {
                    (1.0 + jb.long_bound) * spawn
                } else {
                    (1.0 + jb.deform_limit_expansion) * rest
                };
            }
        }
        self.beams.push(beam);
        if jb.needs_ext() || jb.long_extent.is_some() {
            self.attach_ext(self.beams.len() - 1, &jb, spawn);
        }
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
        // (Bounded: RoR shock precompression; any other beam only with
        // Rubble's `.truck` extension, 1 otherwise.)
        if b.kind == BeamKind::Bounded || b.precompression != 1.0 {
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

    /// The desc's NONCOLLIDABLE triangles as anti-clip-only collision triangles
    /// (BeamNG: "NONCOLLIDABLE triangles still have anti-clip"; spec §36), after
    /// the collision ones.
    pub(crate) fn add_anti_only_triangles(&mut self, desc: &SoftVehicleDesc) {
        let before = self.triangles.len();
        for t in desc.triangles.iter().filter(|t| !t.collision) {
            self.triangles.push(Triangle {
                nodes: t.nodes,
                wait: 0,
                idle: 0,
            });
        }
        if self.triangles.len() > before {
            self.anti_only_tris = vec![false; before];
            self.anti_only_tris.resize(self.triangles.len(), true);
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

    /// How far above the ground to spawn the vehicle's lowest node: 2 cm (the
    /// usual drop onto its tyres), or, for a file that is its vehicle at rest
    /// (`rubble_rest_pose`), its lowest node's depth under it (its tyres
    /// already stand on the ground, in contact).
    pub fn spawn_clearance(&self) -> f32 {
        match self.rest_pose {
            Some(depth) => -depth,
            None => 0.02,
        }
    }

    /// Opt in to JBeam's contact semantics (spec §10.1–10.2) for a vehicle of the
    /// `.truck` path (JBeam mode has them): every node with collision is a contacter
    /// against other vehicles' triangles, and self-collision pairs that start inside
    /// the contact range never collide. Call after spawn, before the GPU upload.
    /// Returns the excluded self-collision pairs.
    /// Turns the contact-force log on (spec §32; no contact breaks unless set).
    pub fn enable_contact_force(&mut self) {
        if self.contact_force.is_some() {
            return;
        }
        let n = self.nodes.len();
        let groups = self.beams.iter().map(|b| b.group.unsigned_abs()).max().unwrap_or(0) as usize;
        self.contact_force = Some(Box::new(ContactForceLog {
            sum: vec![0; n],
            last: vec![0; n],
            peak: vec![0; n],
            node_break: vec![(0, 0.0); n],
            pending: vec![0; groups + 1],
        }));
    }

    /// JBeam plastic limits on `.truck` beams (`rubble_jbeam_plastic`, spec
    /// §3): per line (a, b, deformLimitStress N, deformLimit, deformLimitExpansion)
    /// the plain beam between nodes a and b becomes a JBeam NORMAL beam
    /// ([`ExplicitKind::JbeamPlain`]) with its spring, damping, thresholds,
    /// strength and plastic coefficient, its hardening capped at the stress
    /// and its rest length held within (1 − d)·L and (1 + dₑ)·L of its rest
    /// length L now (a negative limit: none), exactly as JBeam mode loads
    /// them. Errors name a pair with no plain beam.
    pub fn set_jbeam_plastic(&mut self, lines: &[(u32, u32, f32, f32, f32, bool)]) -> Result<(), String> {
        let mut at: HashMap<(u32, u32), usize> = HashMap::new();
        for (j, b) in self.beams.iter().enumerate() {
            if matches!(b.kind, ExplicitKind::Plain | ExplicitKind::Rope) && b.ty == ExplicitType::Deformable {
                at.entry((b.a.min(b.b), b.a.max(b.b))).or_insert(j);
            }
        }
        for &(a, b, stress, d, de, tear) in lines {
            let Some(&j) = at.get(&(a.min(b), a.max(b))) else {
                return Err(format!("rubble_jbeam_plastic: no plain beam or rope {a}–{b}"));
            };
            if tear && !(de >= 0.0) {
                return Err(format!(
                    "rubble_jbeam_plastic: {a}–{b} tears at its expansion limit but has none ({de})"
                ));
            }
            let beam = &mut self.beams[j];
            beam.kind = if beam.kind == ExplicitKind::Rope {
                ExplicitKind::JbeamRope
            } else {
                ExplicitKind::JbeamPlain
            };
            beam.limit_k = if stress >= 0.0 { stress } else { f32::INFINITY };
            beam.short_bound = if d >= 0.0 { (1.0 - d) * beam.rest } else { 0.0 };
            beam.long_bound = if de >= 0.0 {
                let l = (1.0 + de) * beam.rest;
                if tear {
                    -l
                } else {
                    l
                }
            } else {
                f32::INFINITY
            };
        }
        Ok(())
    }

    /// A tension-side yield force on `.truck` beams (`rubble_tension_yield`,
    /// spec §18.4): per line (a, b, Tt) the deformable plain beam a–b (`Plain`
    /// or `JbeamPlain`) yields in tension at Tt; its compression threshold Tc
    /// stays. The yield rule already keeps Tc and Tt apart (they start equal at
    /// spawn), so only the thresholds and the gate (min(Tc, Tt, S)) change.
    pub fn set_tension_yield(&mut self, lines: &[(u32, u32, f32)]) -> Result<(), String> {
        let mut at: HashMap<(u32, u32), usize> = HashMap::new();
        for (j, b) in self.beams.iter().enumerate() {
            if matches!(b.kind, ExplicitKind::Plain | ExplicitKind::JbeamPlain)
                && b.ty == ExplicitType::Deformable
            {
                at.entry((b.a.min(b.b), b.a.max(b.b))).or_insert(j);
            }
        }
        for &(a, b, tt) in lines {
            let Some(&j) = at.get(&(a.min(b), a.max(b))) else {
                return Err(format!(
                    "rubble_tension_yield: no deformable plain beam {a}–{b}"
                ));
            };
            if !(tt > 0.0) {
                return Err(format!("rubble_tension_yield: {a}–{b}: Tt {tt} is not > 0"));
            }
            let beam = &mut self.beams[j];
            beam.tt = tt;
            beam.gate = beam.tc.min(beam.tt).min(beam.strength);
        }
        Ok(())
    }

    /// Contact breaks (`rubble_contact_breaks`, spec §32): per line (break group,
    /// force N, nodes), the group goes off when the contact force on any of the
    /// nodes exceeds the force in a step. A group must be some beam's own break
    /// group, and a node in one line at most.
    pub fn set_contact_breaks(&mut self, lines: &[(u32, f32, Vec<u32>)]) -> Result<(), String> {
        self.enable_contact_force();
        let n = self.nodes.len();
        let log = self.contact_force.as_mut().unwrap();
        for (g, force, nodes) in lines {
            if !self.beams.iter().any(|b| b.group.unsigned_abs() == *g) {
                return Err(format!("rubble_contact_breaks: group {g} breaks no beam"));
            }
            for &i in nodes {
                if i as usize >= n {
                    return Err(format!("rubble_contact_breaks: group {g}: no node {i}"));
                }
                let slot = &mut log.node_break[i as usize];
                if slot.0 != 0 {
                    return Err(format!(
                        "rubble_contact_breaks: node {i} is in groups {} and {g} (one line per node)",
                        slot.0
                    ));
                }
                *slot = (*g, *force);
            }
        }
        Ok(())
    }

    /// Per node the last complete step's contact force (N; spec §32), or `None`
    /// when the log is off.
    pub fn contact_forces(&self) -> Option<Vec<f32>> {
        let l = self.contact_force.as_ref()?;
        Some(l.last.iter().map(|&q| q as f32 / crate::explicit_rigid::FIXED).collect())
    }

    /// Per node the largest step's contact force since the last call (N), reset.
    pub fn take_contact_force_peak(&mut self) -> Option<Vec<f32>> {
        let l = self.contact_force.as_mut()?;
        let out = l.peak.iter().map(|&q| q as f32 / crate::explicit_rigid::FIXED).collect();
        l.peak.iter_mut().for_each(|q| *q = 0);
        Some(out)
    }

    /// The contact breaks set off by the last step's contact force (spec §32):
    /// each group breaks once, as if by the beam pass's first beam.
    fn fire_contact_breaks(&mut self) {
        let Some(log) = self.contact_force.as_mut() else {
            return;
        };
        let fire: Vec<i32> = log
            .pending
            .iter_mut()
            .enumerate()
            .filter(|(_, p)| **p == 1)
            .map(|(g, p)| {
                *p = 2;
                g as i32
            })
            .collect();
        for g in fire {
            self.cascade(g);
        }
    }

    pub fn opt_in_full_contact(&mut self) -> usize {
        self.contact_every_node = true;
        self.finish_contact();
        self.spawn_exclusions()
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
                    || (!n.has(NODE_NO_GROUND)
                        && (self.contact_every_node || on_triangle[i as usize] || no_triangles))
            })
            .collect();
        self.neighbour_start = Vec::with_capacity(self.nodes.len() + 1);
        self.neighbours.clear();
        self.neighbour_start.push(0);
        for (i, inc) in self.incidence.iter().enumerate() {
            for &j in inc {
                let b = &self.beams[j as usize];
                self.neighbours
                    .push(if b.a == i as u32 { b.b } else { b.a });
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

    /// Collision triangle `t`'s self-collision group: its nodes' group when all
    /// three share one (and the vehicle has [`Self::self_groups`]), else 0.
    pub fn triangle_self_group(&self, t: usize) -> u32 {
        if self.self_groups.is_empty() {
            return 0;
        }
        let [a, b, c] = self.triangles[t]
            .nodes
            .map(|n| self.self_groups[n as usize]);
        if a == b && b == c {
            a
        } else {
            0
        }
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
        if self.drive.is_some() {
            crate::explicit_drive::drive_pass(self);
        }
        if self.settings.centred_damping {
            let v0: Vec<Vec3> = self.nodes.iter().map(|n| n.vel).collect();
            // The objective form reads each beam's direction before the update too.
            let x0: Option<Vec<Vec3>> = self
                .settings
                .centred_objective
                .then(|| self.nodes.iter().map(|n| n.pos).collect());
            self.node_pass();
            self.centred_correction(&v0, x0.as_deref());
        } else {
            self.node_pass();
        }
        if let Some(speed) = self.sleep.track {
            // Vehicle sleep: the velocities this node pass (and its centred correction)
            // left (the GPU's point).
            if !self.sleep.moving {
                self.sleep.moving = self.nodes.iter().any(|n| crate::explicit_sleep::moving(n.vel, speed));
            }
        }
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
        if !self.brakes.is_empty() && self.parking_brake > 0.0 {
            self.brake_pass();
        }
        let mut trace = self.trace.take();
        if let Some(t) = trace.as_mut() {
            t.capture(&forces_now_of(&self), 0);
        }
        if self.damp_scheme == 0 {
            self.beam_pass();
        } else {
            self.beam_pass_predicted();
        }
        if let Some(t) = trace.as_mut() {
            t.capture(&forces_now_of(&self), 1);
        }
        if !self.triangles.is_empty() && !self.candidates.is_empty() {
            self.collision_pass();
        }
        if self.surface_contact {
            self.surface_pass();
        }
        // The rigid proxies integrate this step's reactions, the node pass's and the
        // surface pass's (nothing between the node pass and here reads them).
        if let Some(r) = self.rigid.as_mut() {
            r.end_step();
        }
        if let Some(t) = trace.as_mut() {
            t.capture(&forces_now_of(&self), 2);
        }
        if !self.slides.is_empty() {
            self.slide_pass();
        }
        if let Some(t) = trace.as_mut() {
            t.capture(&forces_now_of(&self), 3);
        }
        self.trace = trace;
        self.steps += 1;
    }

    fn node_pass(&mut self) {
        let exact = self.exact_friction;
        // Anti-clip's side memory (§36): last step's contacts are held now (a retired
        // node's stays as it was: the GPU's node phase skips it).
        let any_retired = self.retired.iter().any(|&r| r);
        for (i, c) in self.clip_sides.iter_mut().enumerate() {
            if any_retired && self.retired.get(i).copied().unwrap_or(false) {
                continue;
            }
            crate::explicit_contact::clip_roll(c);
        }
        // Fixed-point contact (§36): last step's contact sums, then its slide forces.
        if self.fixed_contact {
            for i in 0..self.nodes.len() {
                let f = self.force_now(i);
                self.nodes[i].force = f;
            }
            self.contact_acc.clear();
            self.slide_hold.clear();
            if let Some(e) = self.contact_exact.as_mut() {
                e.clear();
            }
        }
        // §37: last step's contact position steps.
        self.shift_applied.clear();
        self.ground_lift.clear();
        if self.dissipative_contact {
            self.ground_lift.resize(self.nodes.len(), 0.0);
        }
        let dissipative = self.dissipative_contact;
        if !self.contact_shift.is_empty() {
            self.shift_applied.extend(self.contact_shift.iter().map(|a| {
                Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32) * (1.0 / SHIFT_FIXED)
            }));
            self.contact_shift.clear();
        }
        let h = STEP;
        let origin = self.origin;
        let ground = self.ground;
        let settings = self.settings;
        // The shapes' culling boxes (none allocated without shapes).
        let shape_bounds: Vec<StaticBox> = self.shapes.iter().map(StaticShape::bounds).collect();
        let mut rigid = self.rigid.take();
        if let Some(r) = rigid.as_mut() {
            r.begin_step(origin, &self.nodes);
        }
        // §38.4: with surface contact, a fast proxy pushes the nodes inside it out the side
        // they came from.
        let fast_side = self.surface_contact && !crate::explicit_surface::casts_off();
        let mut cf = self.contact_force.take();
        // (Sized by the pressure wheels' builder: every tread node has a slot.)
        let mut tread = std::mem::take(&mut self.tread);
        for i in 0..self.nodes.len() {
            if any_retired && self.retired.get(i).copied().unwrap_or(false) {
                // Retired debris: at rest where it is, nothing else.
                self.nodes[i].vel = Vec3::ZERO;
                continue;
            }
            let mut n = self.nodes[i];
            if let Some(l) = cf.as_mut() {
                l.roll(i);
            }
            // §37's ground position step (m, up).
            let mut lift = 0.0f32;
            if !n.has(NODE_NO_GROUND) {
                let x = origin + n.pos;
                if n.has(NODE_TREAD) {
                    let s = &mut tread[i];
                    if ground > x.y {
                        let p = ground - x.y;
                        let (r, z) = tread_hysteresis(s[0], p, s[1], n.mass);
                        *s = [z, p];
                        if r > 0.0 {
                            let c = contact_friction(
                                r,
                                n.force,
                                n.vel,
                                Vec3::Y,
                                &ContactSurface::GROUND.exact_if(exact),
                                n.friction,
                            );
                            let c = crate::explicit_jbeam_ext::slip_limit(
                                c,
                                n.force,
                                n.vel,
                                n.mass,
                                Vec3::Y,
                                h,
                            );
                            if let Some(l) = cf.as_mut() {
                                l.add(i, c);
                            }
                            n.force += c;
                        }
                    } else {
                        *s = [0.0; 2];
                    }
                } else if ground > x.y {
                    let (c, budget) = contact_response(
                        n.force,
                        n.vel,
                        n.mass,
                        Vec3::Y,
                        h,
                        &ContactSurface::GROUND
                            .exact_if(exact)
                            .dissipative_if(dissipative),
                        ground - x.y,
                        n.friction,
                    );
                    // §37: the ground's position step (the node's own, up).
                    if dissipative && c != Vec3::ZERO {
                        lift = shift_depth(ground - x.y, budget, n.force.y);
                    }
                    let c = if n.has(NODE_FRICTION_LIMIT) {
                        crate::explicit_jbeam_ext::friction_limit(
                            c,
                            n.force,
                            n.vel,
                            n.mass,
                            Vec3::Y,
                            h,
                            ground - x.y,
                        )
                    } else {
                        c
                    };
                    if let Some(l) = cf.as_mut() {
                        l.add(i, c);
                    }
                    n.force += c;
                }
                for bx in &self.boxes {
                    if let Some(normal) = box_face(bx, x) {
                        let c = contact_force(
                            n.force,
                            n.vel,
                            n.mass,
                            normal,
                            h,
                            &ContactSurface::BOX
                                .exact_if(exact)
                                .dissipative_if(dissipative),
                            0.0,
                            n.friction,
                        );
                        if let Some(l) = cf.as_mut() {
                            l.add(i, c);
                        }
                        n.force += c;
                    }
                }
                for (sh, bd) in self.shapes.iter().zip(&shape_bounds) {
                    if let Some(normal) = shape_face(sh, bd, x) {
                        let c = contact_force(
                            n.force,
                            n.vel,
                            n.mass,
                            normal,
                            h,
                            &ContactSurface::BOX
                                .exact_if(exact)
                                .dissipative_if(dissipative),
                            0.0,
                            n.friction,
                        );
                        if let Some(l) = cf.as_mut() {
                            l.add(i, c);
                        }
                        n.force += c;
                    }
                }
                if let Some(r) = rigid.as_mut() {
                    let c = r.node_contact_with(x, &n, fast_side);
                    if let Some(l) = cf.as_mut() {
                        l.add(i, c);
                    }
                    n.force += c;
                }
            }
            if !n.has(NODE_IMMOVABLE) {
                n.vel += n.force / n.mass * h;
                n.pos += n.vel * h;
                if let Some(&d) = self.shift_applied.get(i) {
                    n.pos += d;
                }
                if lift != 0.0 {
                    n.pos.y += lift;
                    if let Some(x) = self.ground_lift.get_mut(i) {
                        *x = lift;
                    }
                }
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
        self.contact_force = cf;
        self.tread = tread;
        // The proxies integrate their reactions at the step's end (`Self::step`), after
        // the surface pass has added its own.
        self.rigid = rigid;
        if let Some(g) = self.grab {
            self.grab_pass(g);
        }
    }

    /// The node grab ([`Grab`]): its force added to the grabbed node's restarted
    /// force (a node index past the vehicle's nodes is ignored).
    fn grab_pass(&mut self, g: Grab) {
        let origin = self.origin;
        if let Some(n) = self.nodes.get_mut(g.node as usize) {
            n.force += g.force(origin + n.pos, n.vel);
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
        if let Some(d) = fd.force(
            p.vel,
            y,
            self.nodes.len(),
            &self.airfoils[fd.airfoil as usize],
        ) {
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

    /// Slide nodes in spawn order (spec §3.4), after self-collision. Each node's
    /// slide forces are summed (from zero, in slide order: node, rail end a, rail
    /// end b) and that sum is added to its force, as the kernel's slide slots do
    /// (jbeam-semantics-spec.md §35: bit for bit when a rail end takes several).
    fn slide_pass(&mut self) {
        let mut sums: Vec<(u32, Vec3)> = Vec::new();
        let mut add = |n: u32, f: Vec3| match sums.iter_mut().find(|s| s.0 == n) {
            Some(s) => s.1 += f,
            None => sums.push((n, Vec3::ZERO + f)),
        };
        for i in 0..self.slides.len() {
            let Some(cur) = self.slides[i].segment else {
                continue;
            };
            if self.beams[self.segments[cur as usize].beam as usize].broken {
                continue;
            }
            // The vehicle's frame (positions relative to its origin, as the kernel's
            // SLIDE_FRAME): the rail distance keeps its precision far from the World's
            // origin (World positions at 4 km are 0.24 mm apart).
            let x = Vec3::ZERO + self.nodes[self.slides[i].node as usize].pos;
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
            add(node, f.node);
            add(b.a, f.a);
            add(b.b, f.b);
        }
        if self.fixed_contact {
            // Added by the next node pass after the contact sums (the kernel's order).
            self.slide_hold = sums;
            return;
        }
        for (n, f) in sums {
            self.nodes[n as usize].force += f;
        }
    }

    /// [`Self::damp_scheme`] > 0: a dry beam pass (state restored) predicts each
    /// node's end-of-step velocity; the real pass then reads that (or its mean
    /// with the step's) and the step's velocities are put back.
    fn beam_pass_predicted(&mut self) {
        let h = STEP;
        let f0: Vec<Vec3> = self.nodes.iter().map(|n| n.force).collect();
        let v0: Vec<Vec3> = self.nodes.iter().map(|n| n.vel).collect();
        let snap = (
            self.beams.clone(),
            self.damp_filters.clone(),
            self.beam_ext.clone(),
            self.groups_now.clone(),
            self.groups_last.clone(),
            self.tri_breaks.clone(),
            self.tri_dead.clone(),
            self.deform_state.clone(),
            self.key_input,
            self.contact_force.clone(),
        );
        self.beam_pass();
        let va: Vec<Vec3> = self
            .nodes
            .iter()
            .map(|n| if n.has(NODE_IMMOVABLE) { n.vel } else { n.vel + n.force / n.mass * h })
            .collect();
        (
            self.beams,
            self.damp_filters,
            self.beam_ext,
            self.groups_now,
            self.groups_last,
            self.tri_breaks,
            self.tri_dead,
            self.deform_state,
            self.key_input,
            self.contact_force,
        ) = snap;
        let mix = if self.damp_scheme == 2 { 0.5 } else { 1.0 };
        for (i, n) in self.nodes.iter_mut().enumerate() {
            n.force = f0[i];
            n.vel = v0[i] + (va[i] - v0[i]) * mix;
        }
        self.beam_pass();
        for (i, n) in self.nodes.iter_mut().enumerate() {
            n.vel = v0[i];
        }
    }

    /// The centred damping's correction after the node update
    /// ([`ExplicitBeamsSettings::centred_damping`], spec §39). Δvᵢ is the velocity
    /// change the update gave node i. Each beam record with a damping slope s from
    /// the last beam pass gives its ends −½·s·Δu along its current direction (Δu
    /// its extension rate's change under Δv; a torsion bar −½·s·Δφ̇ through its
    /// dihedral gradient, a coupler −½·s·(Δv_a − Δv_b)), equal and opposite, so the
    /// damping it applied reads the mean of the step's start and end velocities.
    /// Each node sums its incidence entries in order (the kernel's slots), then
    /// v += h·c/m and x += h·(h·c/m).
    ///
    /// `x0` (the positions before the node update;
    /// [`ExplicitBeamsSettings::centred_objective`], spec §39.6): a plain beam's Δu
    /// is then the change of its extension rate itself, each rate along the beam's
    /// direction at its own time, (v′_a − v′_b)·d̂′ − (v_a − v_b)·d̂. Without it Δu is
    /// (Δv_a − Δv_b)·d̂′, which is the same while the beam keeps its direction and
    /// reads −ω²·L·h for a beam that only turns at ω (its relative velocity turns
    /// with it): an outward push on every damped beam of a spinning body, which
    /// then gains energy (known-bugs S3).
    fn centred_correction(&mut self, v0: &[Vec3], x0: Option<&[Vec3]>) {
        let h = STEP;
        let nb = self.beams.len();
        self.damp_slopes.resize(nb, 0.0);
        let dv: Vec<Vec3> = self.nodes.iter().zip(v0).map(|(n, v)| n.vel - *v).collect();
        // Per beam record its two entries' corrections (end a, end b).
        let mut slot = vec![[Vec3::ZERO; 2]; nb];
        for j in 0..nb {
            let slope = self.damp_slopes[j];
            if slope == 0.0 {
                continue;
            }
            let b = self.beams[j];
            let (a, e) = (b.a as usize, b.b as usize);
            match b.kind {
                ExplicitKind::JbeamTorsion => {
                    let n = self.beams[j + 1];
                    let ids = [a, e, n.a as usize, n.b as usize];
                    let g = dihedral_jacobian(ids.map(|i| self.nodes[i].pos));
                    let dr = match x0 {
                        // The dihedral rate's change, each rate with its own gradient.
                        Some(x0) => {
                            let g0 = dihedral_jacobian(ids.map(|i| x0[i]));
                            (0..4)
                                .map(|q| g[q].dot(self.nodes[ids[q]].vel) - g0[q].dot(v0[ids[q]]))
                                .sum::<f32>()
                        }
                        None => {
                            g[0].dot(dv[ids[0]])
                                + g[1].dot(dv[ids[1]])
                                + g[2].dot(dv[ids[2]])
                                + g[3].dot(dv[ids[3]])
                        }
                    };
                    let t = -0.5 * slope * dr;
                    slot[j] = [g[0] * t, g[1] * t];
                    slot[j + 1] = [g[2] * t, g[3] * t];
                }
                ExplicitKind::JbeamCoupler => {
                    let f = (dv[a] - dv[e]) * (-0.5 * slope);
                    slot[j] = [f, -f];
                }
                _ => {
                    let d = self.nodes[a].pos - self.nodes[e].pos;
                    let inv = exact_length(d.dot(d)).1;
                    let du = match x0 {
                        Some(x0) => {
                            let d0 = x0[a] - x0[e];
                            let inv0 = exact_length(d0.dot(d0)).1;
                            (self.nodes[a].vel - self.nodes[e].vel).dot(d) * inv
                                - (v0[a] - v0[e]).dot(d0) * inv0
                        }
                        None => (dv[a] - dv[e]).dot(d) * inv,
                    };
                    let push = d * ((-0.5 * slope * du) * inv);
                    slot[j] = [push, -push];
                }
            }
        }
        for i in 0..self.nodes.len() {
            if self.is_retired(i) {
                // Retired debris: left at rest (the node pass's).
                continue;
            }
            let mut c = Vec3::ZERO;
            let mut seen: Option<u32> = None;
            for &j in &self.incidence[i] {
                let b = &self.beams[j as usize];
                // The entry's end as the kernel's layout assigns it (a beam from a
                // node to itself has both entries in this list).
                let other_end = b.b == i as u32 && b.a != i as u32;
                let end = if other_end || (b.a == b.b && seen == Some(j)) {
                    1
                } else {
                    0
                };
                if b.a == b.b {
                    seen = Some(j);
                }
                c += slot[j as usize][end];
            }
            let n = &mut self.nodes[i];
            if !n.has(NODE_IMMOVABLE) {
                let dv = c / n.mass * h;
                n.vel += dv;
                n.pos += dv * h;
            }
        }
    }

    fn beam_pass(&mut self) {
        let slopes = self.settings.centred_damping;
        if slopes {
            self.damp_slopes.clear();
            self.damp_slopes.resize(self.beams.len(), 0.0);
        }
        std::mem::swap(&mut self.groups_last, &mut self.groups_now);
        self.groups_now.clear();
        if !self.tri_breaks.is_empty() {
            self.triangle_breaks();
        }
        let fire = self
            .contact_force
            .as_ref()
            .is_some_and(|l| l.pending.contains(&1));
        for j in 0..self.beams.len() {
            if fire && j == 1 {
                self.fire_contact_breaks();
            }
            if self.beams[j].broken {
                if !self.ext_of.is_empty() && self.ext_of[j] != 0 {
                    self.ext_broken(j);
                }
                continue;
            }
            if !self.ext_of.is_empty() && self.ext_of[j] != 0 {
                self.ext_beam(j);
                continue;
            }
            let mut beam = self.beams[j];
            let na = self.nodes[beam.a as usize];
            let nb = self.nodes[beam.b as usize];
            if beam.kind == ExplicitKind::JbeamTorsionPair {
                continue;
            }
            if beam.kind == ExplicitKind::JbeamTorsion {
                let (c, d) = (self.beams[j + 1].a, self.beams[j + 1].b);
                let nodes = [na, nb, self.nodes[c as usize], self.nodes[d as usize]];
                let watch = self.torsion_watch[beam.shock as usize - 1];
                let f = if watch
                    .iter()
                    .any(|&g| g != 0 && self.groups_last.contains(&g))
                {
                    beam.broken = true;
                    beam.force = 0.0;
                    [Vec3::ZERO; 4]
                } else {
                    torsion_force(&mut beam, &nodes)
                };
                self.beams[j] = beam;
                if beam.broken {
                    self.beams[j + 1].broken = true;
                    if beam.group > 0 {
                        self.cascade(beam.group);
                    }
                }
                if slopes && !beam.broken {
                    self.damp_slopes[j] = beam.c;
                }
                for (n, f) in [beam.a, beam.b, c, d].into_iter().zip(f) {
                    self.nodes[n as usize].force += f;
                }
                continue;
            }
            if beam.kind == ExplicitKind::JbeamCoupler {
                let push = coupler_force(&mut beam, &na, &nb);
                self.beams[j] = beam;
                if slopes && !beam.broken {
                    self.damp_slopes[j] = beam.c;
                }
                if beam.broken && beam.group > 0 {
                    self.cascade(beam.group);
                }
                self.nodes[beam.a as usize].force += push;
                self.nodes[beam.b as usize].force -= push;
                continue;
            }
            let out = if beam.kind == ExplicitKind::Shock2 {
                let sh = self.shocks2[beam.shock as usize];
                shock2_force(&mut beam, &sh, &na, &nb, self.settings.fast_inv_sqrt)
            } else if beam.kind.is_jbeam() {
                let filter = match beam.shock {
                    0 => None,
                    i => Some(&mut self.damp_filters[i as usize - 1]),
                };
                jbeam_force_filtered(
                    &mut beam,
                    &na,
                    &nb,
                    self.settings.fast_inv_sqrt,
                    filter,
                    !self.plastic_strength_drop,
                )
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
            // A JBeam support stretched past its break length breaks its group too.
            let mut cascade = beam.broken && beam.kind.is_jbeam() && beam.group > 0;
            if out.check_break {
                let saved = !beam.kind.is_jbeam()
                    && [beam.a, beam.b].iter().any(|&e| {
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
            if slopes && !beam.broken && beam.kind.is_jbeam() {
                self.damp_slopes[j] = out.slope;
            }
            if cascade {
                self.cascade(beam.group);
            }
            let push = out.d * (f * out.inv);
            self.nodes[beam.a as usize].force += push;
            self.nodes[beam.b as usize].force -= push;
        }
        if fire && self.beams.len() == 1 {
            self.fire_contact_breaks();
        }
    }

    /// Set off break group `g` (> 0): every member is broken (spec §8, §17).
    pub(crate) fn cascade(&mut self, g: i32) {
        self.groups_now.push(g as u32);
        for other in &mut self.beams {
            if other.group.abs() == g {
                other.broken = true;
            }
        }
        if !self.ext_of.is_empty() {
            self.cascade_ext(g);
        }
    }

    fn collision_pass(&mut self) {
        let exact = self.exact_friction;
        self.last_contacts = 0;
        if let Some(p) = self.self_contact_parts.as_mut() {
            p.clear();
            p.resize(self.nodes.len(), [Vec3::ZERO; 3]);
        }
        if self.grid_cell > 0.0 {
            let cell = self.grid_cell;
            let nodes = &self.nodes;
            self.grid
                .build(self.candidates.iter().map(|&c| nodes[c as usize].pos), cell);
        }
        let r = self.collision_range;
        let h = STEP;
        self.last_self_pairs = 0;
        // Without the throttle (§36) a triangle never waits: its count stays ≤ 0.
        let cap = if self.settings.triangle_throttle { i32::MAX } else { 0 };
        // The two-body rule (spec §36, as vehicle contact's §10.3): the step's hits
        // are found first (positions only), then applied in the same order with
        // mass splitting and the pair's relative acceleration from the forces at
        // the pass's start. Without it RoR's rule applies each hit as it is found.
        let two_body = self.two_body_contact;
        let vmax = if self.anti_clip { self.max_speed() } else { 0.0 };
        let mut found: Vec<(u32, [u32; 3], [f32; 3], f32, Vec3)> = Vec::new();
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
            let retired = !self.retired.is_empty() && [i0, i1, i2].iter().any(|&i| self.is_retired(i as usize));
            if retired || (!self.tri_dead.is_empty() && self.tri_dead[t]) || (self.tri_anti_only(t) && !self.anti_clip) {
                // A broken triangle (spec §18), or an anti-clip-only one with
                // anti-clip off: no candidates.
                self.triangles[t].wait = (self.triangles[t].wait + 1).min(cap);
                continue;
            }
            let tri_group = self.triangle_self_group(t);
            let world = |i: u32| self.origin + self.nodes[i as usize].pos;
            let (p0, p1, p2) = (world(i0), world(i1), world(i2));
            let frame = TriFrame::new(p0, p1, p2);
            // Anti-clip (§36): the box also reaches what crossed it over the step
            // or is remembered up to CLIP_DEPTH behind it.
            let anti = self.anti_clip;
            let tv = if anti {
                [i0, i1, i2]
                    .iter()
                    .map(|&i| self.nodes[i as usize].vel.length())
                    .fold(0.0f32, f32::max)
            } else {
                0.0
            };
            let reach = if anti {
                r + (tv + vmax) * STEP + crate::explicit_contact::CLIP_DEPTH
            } else {
                r
            };
            let lo = p0.min(p1).min(p2) - Vec3::splat(reach);
            let hi = p0.max(p1).max(p2) + Vec3::splat(reach);
            // Each contacter padded by its own travel (`own_travel_reach`): a box inside
            // the one above (its speed at most vmax), whose cells the grid gathers.
            let own = anti && self.own_travel_reach;
            let only = self.tri_anti_only(t);
            let mut hit = false;
            // The candidates to test, in candidate order: all of them, or those in
            // the grid buckets the padded box covers.
            let mut list = std::mem::take(&mut self.grid.list);
            list.clear();
            if self.grid_cell > 0.0 {
                self.grid.gather(
                    lo - self.origin,
                    hi - self.origin,
                    self.grid_cell,
                    &mut list,
                );
            }
            let count = if self.grid_cell > 0.0 {
                list.len()
            } else {
                self.candidates.len()
            };
            for idx in 0..count {
                let k = if self.grid_cell > 0.0 {
                    list[idx] as usize
                } else {
                    idx
                };
                let hn = self.candidates[k];
                if hn == i0 || hn == i1 || hn == i2 {
                    continue;
                }
                if !self.retired.is_empty() && self.is_retired(hn as usize) {
                    continue;
                }
                if tri_group != 0 && self.self_groups[hn as usize] == tri_group {
                    continue;
                }
                if !self.tri_excl.is_empty() && self.tri_excl[t].contains(&hn) {
                    continue;
                }
                let p = self.origin + self.nodes[hn as usize].pos;
                if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                    continue;
                }
                if own {
                    let vh = self.nodes[hn as usize].vel.length();
                    let s = if vh < vmax { vh } else { vmax };
                    let reach = r + (tv + s) * STEP + crate::explicit_contact::CLIP_DEPTH;
                    let lo = p0.min(p1).min(p2) - Vec3::splat(reach);
                    let hi = p0.max(p1).max(p2) + Vec3::splat(reach);
                    if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                        continue;
                    }
                }
                self.last_self_pairs += 1;
                // (α, β, γ, depth into the contact, normal).
                let found_hit = if anti {
                    self.clip_hit(t, hn, p, &frame, [i0, i1, i2], r, only)
                } else {
                    frame.hit(p, r).map(|(a, b, g, d, n)| (a, b, g, r - d, n))
                };
                let Some((alpha, beta, gamma, pen, n)) = found_hit else {
                    continue;
                };
                hit = true;
                self.last_contacts += 1;
                if two_body {
                    found.push((hn, [i0, i1, i2], [alpha, beta, gamma], pen, n));
                    continue;
                }
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
                    self.force_now(hn as usize),
                    v,
                    m,
                    n,
                    h,
                    &ContactSurface::BOX
                        .exact_if(exact)
                        .dissipative_if(self.dissipative_contact),
                    pen,
                    node.friction,
                );
                self.add_contact(hn as usize, f);
                self.add_contact(i1 as usize, -(alpha * f));
                self.add_contact(i2 as usize, -(beta * f));
                self.add_contact(i0 as usize, -(gamma * f));
                if let Some(l) = self.contact_force.as_mut() {
                    l.add(hn as usize, f);
                    l.add(i1 as usize, alpha * f);
                    l.add(i2 as usize, beta * f);
                    l.add(i0 as usize, gamma * f);
                }
            }
            self.grid.list = list;
            let tri = &mut self.triangles[t];
            if hit {
                tri.wait = -20000;
            } else {
                tri.wait = (tri.wait + 1).min(cap);
            }
        }
        if let Some(l) = self.self_hit_log.as_mut() {
            l.clone_from(&found);
        }
        if !found.is_empty() {
            self.apply_two_body_self_hits(&found);
        }
    }

    /// Whether node `i` is retired ([`Self::retired`]).
    pub fn is_retired(&self, i: usize) -> bool {
        self.retired.get(i).copied().unwrap_or(false)
    }

    /// The fastest node's speed (m/s).
    pub fn max_speed(&self) -> f32 {
        self.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max)
    }

    /// Adds contact force `f` to node `i`: to its fixed-point sum
    /// ([`Self::fixed_contact`]) or to its force.
    pub fn add_contact(&mut self, i: usize, f: Vec3) {
        if !self.fixed_contact {
            self.nodes[i].force += f;
            return;
        }
        if self.contact_acc.len() != self.nodes.len() {
            self.contact_acc = vec![[0; 3]; self.nodes.len()];
        }
        let a = &mut self.contact_acc[i];
        for (k, x) in f.to_array().into_iter().enumerate() {
            a[k] = a[k].wrapping_add(to_fixed(x));
        }
        if let Some(e) = self.contact_exact.as_mut() {
            if e.len() != self.nodes.len() {
                *e = vec![(glam::DVec3::ZERO, 0); self.nodes.len()];
            }
            e[i].0 += f.as_dvec3();
            e[i].1 += 1;
        }
    }

    /// The measurement [`Self::contact_exact`]: node `i`'s fixed-point contact
    /// sum less the exact sum of its contact forces (N), and how many forces it
    /// adds (zero when off).
    pub fn contact_quantisation(&self, i: usize) -> (glam::DVec3, u32) {
        match self.contact_exact.as_ref().and_then(|e| e.get(i)) {
            Some(&(e, k)) => (self.contact_sum(i).as_dvec3() - e, k),
            None => (glam::DVec3::ZERO, 0),
        }
    }

    /// Adds position step `d` (m) to node `i`'s ([`Self::contact_shift`]).
    pub fn add_shift(&mut self, i: usize, d: Vec3) {
        if self.contact_shift.len() != self.nodes.len() {
            self.contact_shift = vec![[0; 3]; self.nodes.len()];
        }
        let a = &mut self.contact_shift[i];
        for (k, x) in d.to_array().into_iter().enumerate() {
            a[k] = a[k].wrapping_add(to_fixed_shift(x));
        }
    }

    /// Node `i`'s fixed-point contact sum in newtons (zero when none).
    pub fn contact_sum(&self, i: usize) -> Vec3 {
        match self.contact_acc.get(i) {
            Some(a) => Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32) * (1.0 / CONTACT_FIXED),
            None => Vec3::ZERO,
        }
    }

    /// Node `i`'s force as the next node pass will take it: with
    /// [`Self::fixed_contact`] its force, plus its contact sum, plus this step's
    /// slide forces (the kernel's order); else its force.
    pub fn force_now(&self, i: usize) -> Vec3 {
        let n = &self.nodes[i];
        if !self.fixed_contact {
            return n.force;
        }
        let f = n.force + self.contact_sum(i);
        match self.slide_hold.iter().find(|s| s.0 == i as u32) {
            Some(s) => f + s.1,
            None => f,
        }
    }

    /// Every node's [`Self::force_now`].
    pub fn forces_now(&self) -> Vec<Vec3> {
        (0..self.nodes.len()).map(|i| self.force_now(i)).collect()
    }

    /// Whether collision triangle `t` is anti-clip only (BeamNG `NONCOLLIDABLE`).
    pub fn tri_anti_only(&self, t: usize) -> bool {
        self.anti_only_tris.get(t).copied().unwrap_or(false)
    }

    /// Self-collision's anti-clip hit test (§36; vehicle contact's rule in
    /// `explicit_contact.rs`): node `hn` at `p` against triangle `t`
    /// (`explicit_contact::clip_test`), recording its side when in contact; the hit
    /// (α, β, γ, depth, normal) when it takes a force: an anti-clip-only triangle
    /// (`only`) none in front of its plane and the depth behind it to the plane.
    #[allow(clippy::too_many_arguments)]
    fn clip_hit(
        &mut self,
        t: usize,
        hn: u32,
        p: Vec3,
        frame: &TriFrame,
        tri: [u32; 3],
        r: f32,
        only: bool,
    ) -> Option<(f32, f32, f32, f32, Vec3)> {
        use crate::explicit_contact::{
            clip_record, clip_side, clip_test, clip_word, crossing_weights, CLIP_EMPTY, CLIP_SELF,
        };
        if self.clip_sides.len() != self.nodes.len() {
            self.clip_sides = vec![[CLIP_EMPTY; 4]; self.nodes.len()];
        }
        let local = frame.local(p);
        let held = clip_side(&self.clip_sides[hn as usize], CLIP_SELF, t as u32);
        let crossing = || {
            let back = |i: u32| self.origin + self.nodes[i as usize].pos - self.nodes[i as usize].vel * STEP;
            let pf = TriFrame::new(back(tri[0]), back(tri[1]), back(tri[2]));
            let p_prev = p - self.nodes[hn as usize].vel * STEP;
            let (a0, b0, _, d0) = pf.local(p_prev);
            crossing_weights(a0, b0, d0, local.0, local.1, local.3)
        };
        let delta = local.3;
        let c = clip_test(local, held, crossing, r, || if delta < 0.0 { -1.0 } else { 1.0 })?;
        clip_record(&mut self.clip_sides[hn as usize], clip_word(CLIP_SELF, t as u32, c.side));
        if only && c.d >= 0.0 {
            return None;
        }
        let pen = crate::explicit_contact::clip_depth(only, c.d, r);
        Some((c.w[0], c.w[1], c.w[2], pen, frame.n * c.side))
    }

    /// Self-collision's two-body rule (spec §36; vehicle contact's §10.3 within
    /// one vehicle): every node of the step's hits (as the contacter, or a vertex
    /// of the hit's triangle) is k sub-bodies of mass m/k (an immovable node has no
    /// inverse mass); the triangle point's inverse mass Σ w²·k/m; the reduced mass
    /// m = 1/(k_h/m_h + Σ w²·k/m); the cancelled force (F_h/m_h − Σ w·F_k/m_k)·m
    /// from the forces at the pass's start, in place of RoR's whole F_h (which
    /// moves the contacter's own load onto the triangle's nodes). Hits in the
    /// order they were found (triangles in order, then candidates).
    fn apply_two_body_self_hits(&mut self, found: &[(u32, [u32; 3], [f32; 3], f32, Vec3)]) {
        let exact = self.exact_friction;
        let mut split = vec![0u32; self.nodes.len()];
        for (hn, tri, _, _, _) in found {
            split[*hn as usize] += 1;
            for &i in tri {
                split[i as usize] += 1;
            }
        }
        let pass: Vec<Vec3> = self.forces_now();
        let im = |n: &Node, k: u32| {
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                k as f32 / n.mass
            }
        };
        let acc = |f: Vec3, n: &Node| {
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                f / n.mass
            }
        };
        let mut first: Vec<Vec3> = Vec::with_capacity(found.len());
        // §37.1: with the residual and the guard, the first pass's sums are a
        // staging the later passes read; each hit's total is added once at the end
        // (the accumulators as they were, then every hit's total: the first pass's
        // quantisation, whatever the later passes add).
        let staged = self.dissipative_contact;
        let snapshot = staged.then(|| (self.contact_acc.clone(), self.contact_exact.clone()));
        for &(hn, [i0, i1, i2], [alpha, beta, gamma], depth, n) in found {
            let node = self.nodes[hn as usize];
            let (n0, n1, n2) = (
                self.nodes[i0 as usize],
                self.nodes[i1 as usize],
                self.nodes[i2 as usize],
            );
            let v = node.vel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma);
            let imh = im(&node, split[hn as usize]);
            let imt = crate::explicit_contact::two_body_inverse_mass(
                [
                    im(&n1, split[i1 as usize]),
                    im(&n2, split[i2 as usize]),
                    im(&n0, split[i0 as usize]),
                ],
                [alpha, beta, gamma],
            );
            let inv = imh + imt;
            let m = if inv > 0.0 { 1.0 / inv } else { 0.0 };
            let at = acc(pass[i1 as usize], &n1) * alpha
                + acc(pass[i2 as usize], &n2) * beta
                + acc(pass[i0 as usize], &n0) * gamma;
            let fh = (acc(pass[hn as usize], &node) - at) * m;
            let (f, budget) = contact_response(
                fh,
                v,
                m,
                n,
                STEP,
                &ContactSurface::BOX
                    .exact_if(exact)
                    .dissipative_if(self.dissipative_contact),
                depth,
                node.friction,
            );
            if self.dissipative_contact && f != Vec3::ZERO {
                // §37: the position step, a position impulse δ·m shared like the
                // hit's force: each node moved by its share over its own mass (as
                // the force moves it; the sub-bodies' k/m would move a node of k
                // hits k times).
                let tri = [(i1, alpha, n1), (i2, beta, n2), (i0, gamma, n0)];
                let delta = shift_depth(depth, budget, fh.dot(n));
                if delta > 0.0 {
                    let q = delta * m;
                    self.add_shift(hn as usize, n * (q * im(&node, 1)));
                    for (i, w, nd) in tri {
                        if w > 0.0 {
                            self.add_shift(i as usize, n * -(w * (q * im(&nd, 1))));
                        }
                    }
                }
            }
            first.push(f);
            if let Some(p) = self.self_contact_parts.as_mut() {
                let fnm = n * f.dot(n);
                for (i, w) in [(hn, 1.0f32), (i1, -alpha), (i2, -beta), (i0, -gamma)] {
                    p[i as usize][0] += fnm * w;
                    p[i as usize][1] += (f - fnm) * w;
                }
            }
            self.add_contact(hn as usize, f);
            self.add_contact(i1 as usize, -(f * alpha));
            self.add_contact(i2 as usize, -(f * beta));
            self.add_contact(i0 as usize, -(f * gamma));
            if !staged {
                if let Some(l) = self.contact_force.as_mut() {
                    l.add(hn as usize, f);
                    l.add(i1 as usize, f * alpha);
                    l.add(i2 as usize, f * beta);
                    l.add(i0 as usize, f * gamma);
                }
            }
        }
        let Some((acc0, exact0)) = snapshot else {
            return;
        };
        let res = if residual_on() {
            self.residual_self_pass(found, &split)
        } else {
            vec![None; found.len()]
        };
        let with_res: Vec<Vec3> = found
            .iter()
            .zip(&first)
            .zip(&res)
            .map(|((x, &f), r)| match r {
                Some(r) => f + x.4 * *r,
                None => f,
            })
            .collect();
        let guard = self.friction_guard_pass(found, &with_res);
        self.contact_acc = acc0;
        self.contact_exact = exact0;
        for (j, &(hn, [i0, i1, i2], [alpha, beta, gamma], _, n)) in found.iter().enumerate() {
            let mut f = with_res[j];
            if let Some(g) = guard[j] {
                f = f + g;
            }
            if let Some(p) = self.self_contact_parts.as_mut() {
                let extra = [res[j].map_or(Vec3::ZERO, |r| n * r), guard[j].unwrap_or(Vec3::ZERO)];
                for (i, w) in [(hn, 1.0f32), (i1, -alpha), (i2, -beta), (i0, -gamma)] {
                    p[i as usize][2] += extra[0] * w;
                    p[i as usize][1] += extra[1] * w;
                }
            }
            self.add_contact(hn as usize, f);
            self.add_contact(i1 as usize, -(f * alpha));
            self.add_contact(i2 as usize, -(f * beta));
            self.add_contact(i0 as usize, -(f * gamma));
            if let Some(l) = self.contact_force.as_mut() {
                l.add(hn as usize, f);
                l.add(i1 as usize, f * alpha);
                l.add(i2 as usize, f * beta);
                l.add(i0 as usize, f * gamma);
            }
        }
    }

    /// §37's residual pass over the step's two-body self hits (after the first,
    /// [`Self::apply_two_body_self_hits`], with its counts `split`): a second
    /// iteration of the mass-split pass for the approach the first left. Its
    /// sub-bodies are lighter than the nodes (k/m), so a hit whose nodes are in
    /// other hits stops only m_split·inv of its pair's approach alone (the rest a
    /// ghost speed the position steps cancel each step). Each hit's pair, at the
    /// forces now (the first pass's contact sums with them), approaching (vn < 0)
    /// and still approaching over the step, a·h + vn < 0 ([`contact_response`]'s
    /// rule, a the pair's relative normal acceleration), takes r = −(a·m +
    /// vn·m/h) > 0 along its normal with the first pass's reduced mass m:
    /// momentum-exact, and never more than the first pass's split allows (counts of
    /// the active hits only were measured to over-stop: +0.23 J of self-contact
    /// work in a tenth of the 58 km/h wall). No friction and no position step (the
    /// first pass's). All found from the sums before any is added (order free).
    /// §37.1's work guard over the step's two-body self hits, after the residual
    /// pass. A hit's force t (its first-pass force with its residual, `first`) is
    /// judged at the pair's realised mean velocity over the step: the forces now
    /// (the staging sums with the residual's) on the nodes' own masses. Its work is
    /// W(s) = s·A + s²·B, with
    /// * A = t·(2v + h·a′), a′ the pair's relative acceleration without t;
    /// * B = h·|t|²·inv, inv the nodes' own inverse masses.
    ///
    /// Where W(1) > 0, the force is scaled to the root s = −A/B (0 if A ≥ 0), and
    /// (1 − s)·t is taken back from the pair (returned per hit). The first pass's
    /// caps judge each hit at its own predicted velocity with split masses; hits
    /// sharing nodes then did +1–2 J of friction work in a tenth at rest. A force
    /// that does positive work moves the pair the way it pushes, so scaling its
    /// normal part lets nothing in. Every guard is found from the same sums before
    /// any is applied (order free, momentum-exact).
    fn friction_guard_pass(
        &self,
        found: &[(u32, [u32; 3], [f32; 3], f32, Vec3)],
        first: &[Vec3],
    ) -> Vec<Option<Vec3>> {
        let mut out = vec![None; found.len()];
        if !friction_guard() {
            return out;
        }
        let now: Vec<Vec3> = self.forces_now();
        let acc = |i: u32| {
            let n = &self.nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                now[i as usize] / n.mass
            }
        };
        let im = |i: u32| {
            let n = &self.nodes[i as usize];
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                1.0 / n.mass
            }
        };
        for (j, (&(hn, [i0, i1, i2], [alpha, beta, gamma], _, _), &t)) in
            found.iter().zip(first).enumerate()
        {
            if t == Vec3::ZERO {
                continue;
            }
            let node = &self.nodes[hn as usize];
            let (n0, n1, n2) = (
                &self.nodes[i0 as usize],
                &self.nodes[i1 as usize],
                &self.nodes[i2 as usize],
            );
            let v = node.vel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma);
            let a = acc(hn) - (acc(i1) * alpha + acc(i2) * beta + acc(i0) * gamma);
            let inv = im(hn)
                + crate::explicit_contact::two_body_inverse_mass(
                    [im(i1), im(i2), im(i0)],
                    [alpha, beta, gamma],
                );
            let a_rest = a - t * inv;
            let aa = t.dot(v + v + a_rest * STEP);
            let bb = STEP * t.dot(t) * inv;
            if aa + bb > 0.0 && bb > 0.0 {
                let s = if aa < 0.0 { (-aa / bb).min(1.0) } else { 0.0 };
                out[j] = Some(t * -(1.0 - s));
            }
        }
        out
    }

    fn residual_self_pass(
        &mut self,
        found: &[(u32, [u32; 3], [f32; 3], f32, Vec3)],
        split: &[u32],
    ) -> Vec<Option<f32>> {
        let now: Vec<Vec3> = self.forces_now();
        let acc = |i: u32, n: &Node| {
            if n.has(NODE_IMMOVABLE) {
                Vec3::ZERO
            } else {
                now[i as usize] / n.mass
            }
        };
        let mut drive = Vec::with_capacity(found.len());
        for &(hn, [i0, i1, i2], [alpha, beta, gamma], _, n) in found {
            let node = &self.nodes[hn as usize];
            let (n0, n1, n2) = (
                &self.nodes[i0 as usize],
                &self.nodes[i1 as usize],
                &self.nodes[i2 as usize],
            );
            let v = node.vel - (n1.vel * alpha + n2.vel * beta + n0.vel * gamma);
            let at = acc(i1, n1) * alpha + acc(i2, n2) * beta + acc(i0, n0) * gamma;
            let a = (acc(hn, node) - at).dot(n);
            let vn = v.dot(n);
            let x = vn + vn.max(0.0);
            // (Approaching pairs only: a separating one has no ghost speed, and
            // its residual over-stopped where hits share nodes, +0.4 J of
            // self-contact work in a tenth of the 48 km/h wall.)
            drive.push((a, x, vn < 0.0 && a * STEP + x < 0.0));
        }
        let im = |n: &Node, k: u32| {
            if n.has(NODE_IMMOVABLE) {
                0.0
            } else {
                k as f32 / n.mass
            }
        };
        let mut out = vec![None; found.len()];
        for (j, (&(hn, [i0, i1, i2], [alpha, beta, gamma], _, _), &(a, x, on))) in
            found.iter().zip(&drive).enumerate()
        {
            if !on {
                continue;
            }
            let inv = im(&self.nodes[hn as usize], split[hn as usize])
                + crate::explicit_contact::two_body_inverse_mass(
                    [
                        im(&self.nodes[i1 as usize], split[i1 as usize]),
                        im(&self.nodes[i2 as usize], split[i2 as usize]),
                        im(&self.nodes[i0 as usize], split[i0 as usize]),
                    ],
                    [alpha, beta, gamma],
                );
            if inv <= 0.0 {
                continue;
            }
            let m = 1.0 / inv;
            let r = -(a * m) - x * m / STEP;
            if r > 0.0 {
                out[j] = Some(r);
            }
        }
        // (Into the staging sums, read by the guard.)
        for (j, &(hn, [i0, i1, i2], [alpha, beta, gamma], _, n)) in found.iter().enumerate() {
            let Some(r) = out[j] else { continue };
            let f = n * r;
            self.add_contact(hn as usize, f);
            self.add_contact(i1 as usize, -(f * alpha));
            self.add_contact(i2 as usize, -(f * beta));
            self.add_contact(i0 as usize, -(f * gamma));
        }
        out
    }
}

/// §37.1's friction guard (on; `FRICTION_GUARD=0`, a measurement switch: off).
pub fn friction_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("FRICTION_GUARD").is_ok_and(|v| v == "0"))
}

/// §37.1's residual pass (on; `RESIDUAL_PASSES=0`, a measurement switch: off).
/// Measured on sedan1 (51 km/h wall, at rest 8–12 s, self-contact work per
/// second): none 800 J sloshed against the steps, one 25–50 J; two and three no
/// steadier and did positive work in a tenth (Jacobi over-stops where hits share
/// nodes with conflicting normals).
pub fn residual_on() -> bool {
    residual_passes() > 0
}

/// The residual passes vehicle contact's CPU measurement runs (`RESIDUAL_PASSES`,
/// 1).
pub fn residual_passes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RESIDUAL_PASSES")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(1)
            .min(1)
    })
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
        // The box's cells counted in float (as the kernel does): a stretched or
        // non-finite triangle (a flung node) takes every candidate instead of
        // overflowing an integer cell count into an unbounded loop.
        let (f0, f1) = ((lo / cell).floor(), (hi / cell).floor());
        let span = f1 - f0 + Vec3::ONE;
        if !(span.cmple(Vec3::splat(64.0)).all() && span.x * span.y * span.z <= 64.0) {
            out.extend(0..self.bucket_of.len() as u32);
            return;
        }
        let (c0, c1) = (grid_cell_of(lo, cell), grid_cell_of(hi, cell));
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

/// A pressure wheel's parking brake (jbeam-semantics-spec.md §25.1): a Coulomb
/// torque between the hub's ring nodes (the disc) and the wheel's `nodeArm:`
/// (the caliper's mount) about the axle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelBrake {
    pub axle: [u32; 2],
    /// The torque reaction node (`nodeArm:`).
    pub arm: u32,
    /// The hub ring nodes `first..first + count`.
    pub first: u32,
    pub count: u32,
    /// Holding torque at full input (N·m, `parkingTorque`).
    pub torque: f32,
}

impl ExplicitBeams {
    /// The parking brakes' forces from this step's integrated state (spec §25.1).
    fn brake_pass(&mut self) {
        let u = self.parking_brake.min(1.0);
        for k in 0..self.brakes.len() {
            let b = self.brakes[k];
            brake_apply(&mut self.nodes, &b, b.torque * u, STEP);
        }
    }
}

/// One brake's forces (spec §25.1), added to the nodes' forces:
///
/// * the axis a is the axle (node1 → node2), its point c and velocity v_c the
///   axle nodes' mean; r⊥ a node's offset from the axis line;
/// * the disc's spin ω_w = Σ m (r⊥ × (v − v_c))·a / I_w, I_w = Σ m |r⊥|² over the
///   hub ring; the caliper's ω_a = (r⊥a × (v_a − v_c))·a / |r⊥a|²;
/// * the torque on the disc τ = −sgn(ω_w − ω_a) · min(T, |ω_w − ω_a| · I_eff / h),
///   I_eff = I_w I_a / (I_w + I_a), I_a = m_a |r⊥a|²: at most the impulse that
///   stops the two bodies' relative spin in one step (stick), else T (slip);
/// * the disc's nodes take m (a × r⊥) τ / I_w (torque τ about a), the arm
///   (a × r⊥a)(−τ / |r⊥a|²) (torque −τ), the two axle nodes half the negative
///   of their sum each: no net force, no net torque about the axle.
pub fn brake_apply(nodes: &mut [Node], b: &WheelBrake, t_max: f32, h: f32) {
    if !(t_max > 0.0) {
        return;
    }
    let (p1, p2) = (nodes[b.axle[0] as usize].pos, nodes[b.axle[1] as usize].pos);
    let (v1, v2) = (nodes[b.axle[0] as usize].vel, nodes[b.axle[1] as usize].vel);
    let span = p2 - p1;
    let ls = span.dot(span);
    if !(ls > 1e-12) {
        return;
    }
    let a = span * (1.0 / ls.sqrt());
    let c = (p1 + p2) * 0.5;
    let vc = (v1 + v2) * 0.5;
    let perp = |p: Vec3| {
        let r = p - c;
        r - a * a.dot(r)
    };
    let (mut l, mut iw) = (0.0f32, 0.0f32);
    for i in b.first..b.first + b.count {
        let n = &nodes[i as usize];
        let r = perp(n.pos);
        l += n.mass * a.dot(r.cross(n.vel - vc));
        iw += n.mass * r.dot(r);
    }
    let arm = nodes[b.arm as usize];
    let ra = perp(arm.pos);
    let da = ra.dot(ra);
    if !(iw > 0.0) || !(da > 1e-8) {
        return;
    }
    let ia = arm.mass * da;
    let wr = l / iw - a.dot(ra.cross(arm.vel - vc)) / da;
    let ieff = iw * ia / (iw + ia);
    let mag = (wr.abs() * ieff / h).min(t_max);
    let tau = if wr > 0.0 { -mag } else { mag };
    let g = tau / iw;
    let mut s = Vec3::ZERO;
    for i in b.first..b.first + b.count {
        let n = &mut nodes[i as usize];
        let f = a.cross(perp(n.pos)) * (g * n.mass);
        n.force += f;
        s += f;
    }
    let fa = a.cross(ra) * (-tau / da);
    nodes[b.arm as usize].force += fa;
    let half = (s + fa) * -0.5;
    nodes[b.axle[0] as usize].force += half;
    nodes[b.axle[1] as usize].force += half;
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

/// `a · b` in the order the kernel's `sdot` takes ((x + y) + z, no fusing).
fn dot3(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// `d / l` component by component (the kernel's `sdiv3`).
fn div3(d: Vec3, l: f32) -> Vec3 {
    Vec3::new(d.x / l, d.y / l, d.z / l)
}

/// The outward normal of the surface of `s` nearest a point strictly inside it
/// (the box rule of [`box_face`], in the shape's terms), or `None` when the point
/// is not strictly inside. `bounds` is `s.bounds()`: a point outside it is not
/// tested further. An oriented box with an identity rotation is [`box_face`] on
/// `StaticBox::centered(centre, half)`.
pub fn shape_face(s: &StaticShape, bounds: &StaticBox, x: Vec3) -> Option<Vec3> {
    let near = x.x > bounds.lo.x
        && x.x < bounds.hi.x
        && x.y > bounds.lo.y
        && x.y < bounds.hi.y
        && x.z > bounds.lo.z
        && x.z < bounds.hi.z;
    if !near {
        return None;
    }
    match *s {
        StaticShape::OrientedBox {
            centre,
            half,
            rotation,
            axes,
        } => {
            if rotation == Quat::IDENTITY {
                return box_face(&StaticBox::centered(centre, half), x);
            }
            let d = x - centre;
            let l = Vec3::new(dot3(axes[0], d), dot3(axes[1], d), dot3(axes[2], d));
            let inside = l.x > -half.x
                && l.x < half.x
                && l.y > -half.y
                && l.y < half.y
                && l.z > -half.z
                && l.z < half.z;
            if !inside {
                return None;
            }
            // Faces in box_face's order (−z, +z, −x, +x, −y, +y), the first nearest.
            let mut normal = -axes[2];
            let mut dist = l.z + half.z;
            for (dd, n) in [
                (half.z - l.z, axes[2]),
                (l.x + half.x, -axes[0]),
                (half.x - l.x, axes[0]),
                (l.y + half.y, -axes[1]),
                (half.y - l.y, axes[1]),
            ] {
                if dd < dist {
                    dist = dd;
                    normal = n;
                }
            }
            Some(normal)
        }
        StaticShape::Cylinder {
            centre,
            radius,
            half_height,
        } => {
            let d = x - centre;
            let r = (d.x * d.x + d.z * d.z).sqrt();
            if !(r < radius && d.y > -half_height && d.y < half_height) {
                return None;
            }
            // The side (radial; +x on the axis), then the top and bottom caps.
            let mut normal = if r > 0.0 {
                Vec3::new(d.x / r, 0.0, d.z / r)
            } else {
                Vec3::X
            };
            let mut dist = radius - r;
            if half_height - d.y < dist {
                dist = half_height - d.y;
                normal = Vec3::Y;
            }
            if d.y + half_height < dist {
                normal = Vec3::NEG_Y;
            }
            Some(normal)
        }
        StaticShape::Capsule { a, b, radius } => {
            let ab = b - a;
            let l2 = dot3(ab, ab);
            let t = if l2 > 0.0 {
                (dot3(x - a, ab) / l2).max(0.0).min(1.0)
            } else {
                0.0
            };
            let p = Vec3::new(a.x + ab.x * t, a.y + ab.y * t, a.z + ab.z * t);
            round_face(x - p, radius)
        }
        StaticShape::Sphere { centre, radius } => round_face(x - centre, radius),
    }
}

/// A point at `d` from a sphere's centre (or a capsule's nearest axis point):
/// strictly inside radius `radius`, pushed out along `d` (+y at the centre).
fn round_face(d: Vec3, radius: f32) -> Option<Vec3> {
    let l = dot3(d, d).sqrt();
    if !(l < radius) {
        return None;
    }
    Some(if l > 0.0 { div3(d, l) } else { Vec3::Y })
}

/// One step's force accumulation split by source, per node (the energy ledger):
/// the accumulator after the node pass and the air passes (gravity, drag, wings),
/// then what the beam pass, self-collision and slide nodes added. The next step's
/// node pass consumes their sum (plus vehicle contact, added by the world).
#[derive(Clone, Debug, Default)]
pub struct ForceTrace {
    /// [air, beams, self-collision, slides] per node.
    pub parts: [Vec<Vec3>; 4],
    last: Vec<Vec3>,
}

/// The forces a trace records (`ExplicitBeams::forces_now`).
fn forces_now_of(s: &ExplicitBeams) -> Vec<Vec3> {
    s.forces_now()
}

impl ForceTrace {
    /// Record stage `k`: the accumulator's growth since the last stage (stage 0:
    /// the accumulator itself).
    fn capture(&mut self, forces: &[Vec3], k: usize) {
        let n = forces.len();
        self.parts[k].resize(n, Vec3::ZERO);
        self.last.resize(n, Vec3::ZERO);
        for (i, &f) in forces.iter().enumerate() {
            self.parts[k][i] = if k == 0 { f } else { f - self.last[i] };
            self.last[i] = f;
        }
    }
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
    /// The damping law's slope ∂(damping force)/∂u at this step's rate (N·s/m;
    /// the low-pass's gain included; 0 where the law has no damping). Read only by
    /// the centred damping (jbeam-semantics-spec.md §39).
    pub slope: f32,
}

/// One beam's force with the kind rules and the yield rules applied to `beam`
/// (rest length, thresholds, strength, last force; a support stretched too far is
/// marked broken). The break test that needs the node graph is left to the caller
/// (`check_break`).
pub fn beam_force(beam: &mut Beam, na: &Node, nb: &Node, fast: bool) -> BeamOut {
    if beam.kind.is_jbeam() {
        return jbeam_force(beam, na, nb, fast);
    }
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let (len, inv) = if fast {
        let inv = fast_inv_sqrt(q);
        (q * inv, inv)
    } else {
        exact_length(q)
    };
    let e = len - beam.rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let mut k = beam.k;
    let mut c = beam.c;
    let mut support_broke = false;
    let mut torn = false;
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
        // A damper-only beam with a tear ratio (`rubble_damper_tear`): stretched past
        // it, the material it damped has parted (the kernel's damperTorn).
        ExplicitKind::Plain if beam.k == 0.0 && beam.long_bound > 0.0 => {
            if e > beam.rest * beam.long_bound {
                // Torn: no force this step (nothing joins its ends any more).
                beam.broken = true;
                support_broke = true;
                torn = true;
            }
        }
        // Shock2 beams take `shock2_force`; here they act as plain ones. Triggers
        // have k = c = 0.
        // JBeam-mode kinds take `jbeam_force` (never here).
        ExplicitKind::Plain
        | ExplicitKind::Shock2
        | ExplicitKind::Trigger
        | ExplicitKind::JbeamPlain
        | ExplicitKind::JbeamRope
        | ExplicitKind::JbeamSupport
        | ExplicitKind::JbeamBounded
        | ExplicitKind::JbeamCoupler
        | ExplicitKind::JbeamTorsion
        | ExplicitKind::JbeamTorsionPair
        | ExplicitKind::JbeamAniso
        | ExplicitKind::JbeamPressured
        | ExplicitKind::JbeamLbeam
        | ExplicitKind::JbeamPressureGroup
        | ExplicitKind::JbeamPressureTri
        | ExplicitKind::JbeamPressureTri2
        | ExplicitKind::JbeamAeroTri
        | ExplicitKind::JbeamAeroTri2
        | ExplicitKind::JbeamLbeam2 => {}
    }
    let mut f = if torn { 0.0 } else { -k * e - c * u };
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
        slope: if support_broke { 0.0 } else { c },
    }
}

/// A JBeam-mode beam's force ([`ExplicitKind::JbeamPlain`], `JbeamSupport`,
/// `JbeamBounded`; `engine/docs/jbeam-semantics-spec.md` §3–5). A support stretched
/// past its break length is marked broken with no force (the caller breaks its
/// group); the break test past the strength is left to the caller (`check_break`),
/// which has no exception for these kinds.
pub fn jbeam_force(beam: &mut Beam, na: &Node, nb: &Node, fast: bool) -> BeamOut {
    jbeam_force_filtered(beam, na, nb, fast, None, false)
}

/// [`jbeam_force`] with the beam's damping filter (`dampCutoffHz`): the damping
/// takes the filtered extension rate.
pub fn jbeam_force_filtered(
    beam: &mut Beam,
    na: &Node,
    nb: &Node,
    fast: bool,
    filter: Option<&mut DampFilter>,
    keep: bool,
) -> BeamOut {
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    // JBeam mode's exact path: the length and inverse length of `exact_length`
    // (spec §35), the kernel's `sExactLength`, bit for bit.
    let (len, inv) = if fast {
        let inv = fast_inv_sqrt(q);
        (q * inv, inv)
    } else {
        exact_length(q)
    };
    let e = len - beam.rest;
    let u = (na.vel - nb.vel).dot(d) * inv;
    let (u, gain) = match filter {
        Some(f) => (f.step(u), f.a),
        None => (u, 1.0),
    };
    let mut k = beam.k;
    let c = beam.c;
    // The damping force (positive along u), and its slope in u.
    let mut damp = c * u;
    let mut slope = c;
    match beam.kind {
        ExplicitKind::JbeamBounded => {
            // Slow damping up to the split, the fast rate past it (continuous).
            let split = beam.tc;
            let au = u.abs();
            if au > split {
                damp = (c * split + beam.tt * (au - split)).copysign(u);
                slope = beam.tt;
            }
            let long = beam.long_bound * beam.rest;
            let short = beam.short_bound * beam.rest;
            let o = if e > long {
                e - long
            } else if e < -short {
                -e - short
            } else {
                0.0
            };
            if o > 0.0 {
                let zone = beam.plastic;
                let s = if zone > 0.0 { (o / zone).min(1.0) } else { 1.0 };
                k += (beam.limit_k - k) * s;
                damp += (beam.limit_c * u - damp) * s;
                slope += (beam.limit_c - slope) * s;
            }
        }
        ExplicitKind::JbeamRope if e < 0.0 => {
            k = 0.0;
            damp = (c * 0.1) * u;
            slope = c * 0.1;
        }
        ExplicitKind::JbeamSupport => {
            if e > 0.0 {
                if len > beam.long_bound {
                    beam.broken = true;
                    beam.force = 0.0;
                    return BeamOut {
                        d,
                        inv,
                        e,
                        force: 0.0,
                        check_break: false,
                        slope: 0.0,
                    };
                }
                k = 0.0;
                damp = 0.0;
                slope = 0.0;
            }
        }
        _ => {}
    }
    let mut f = -k * e - damp;
    beam.force = f;
    let mut tau = f.abs();
    let mut tear = false;
    if tau > beam.gate && beam.ty == ExplicitType::Deformable && k != 0.0 {
        if f > beam.tc && e < 0.0 && e + (beam.tc / k) * (1.0 - beam.plastic) < 0.0 {
            let delta = e + (beam.tc / k) * (1.0 - beam.plastic);
            let old = beam.rest;
            let floor = beam.short_bound.max(MIN_REST);
            let free = beam.rest + delta;
            if free < floor {
                // At the compression limit: elastic about the limit length.
                beam.rest = floor.min(old);
                f = -k * (len - beam.rest) - damp;
            } else {
                beam.rest = free;
                f -= (f - beam.tc) / 2.0;
            }
            tau = f.abs();
            if beam.rest < old {
                beam.tc = (beam.tc * old / beam.rest).min(beam.limit_k.max(beam.tc));
                beam.gate = beam.tc.min(beam.tt).min(beam.strength);
            }
        } else if f < -beam.tt && e > 0.0 && e - (beam.tt / k) * (1.0 - beam.plastic) > 0.0 {
            let delta = e - (beam.tt / k) * (1.0 - beam.plastic);
            let old = beam.rest;
            // A negative bound tears there (`rubble_jbeam_plastic`'s
            // breakAtExpansion, §36).
            let ceiling = beam.long_bound.abs();
            let free = beam.rest + delta;
            if free > ceiling {
                // At the expansion limit: elastic about the limit length.
                beam.rest = ceiling.max(old);
                f = -k * (len - beam.rest) - damp;
                tear = beam.long_bound < 0.0;
            } else {
                beam.rest = free;
                f -= (f + beam.tt) / 2.0;
            }
            tau = f.abs();
            if beam.rest > old {
                beam.tt = (beam.tt * beam.rest / old).min(beam.limit_k.max(beam.tt));
                beam.gate = beam.tc.min(beam.tt).min(beam.strength);
                if !keep {
                    beam.strength -= (beam.rest - old) * k;
                }
            }
        }
    }
    BeamOut {
        d,
        inv,
        e,
        force: f,
        check_break: tau > beam.strength || tear,
        slope: slope * gain,
    }
}

/// [`exact_atan2`]'s series coefficients (−1)ᵏ/(2k + 1), k = 0..11, as bits (the
/// kernel's `ATAN_C`).
#[rustfmt::skip]
const ATAN_C: [u32; 12] = [
    0x3f80_0000, 0xbeaa_aaab, 0x3e4c_cccd, 0xbe12_4925, 0x3de3_8e39, 0xbdba_2e8c,
    0x3d9d_89d9, 0xbd88_8889, 0x3d70_f0f1, 0xbd57_9436, 0x3d43_0c31, 0xbd32_1643,
];

/// atan2(y, x) by one explicitly specified sequence the CPU and the kernel
/// (`sAtan2`) run identically (spec §35): a = min(|x|, |y|)/max(|x|, |y|); past
/// tan(π/8), atan a = π/4 + atan((a − 1)/(a + 1)); atan t by its series to t²³
/// (|t| ≤ tan(π/8), truncation below 4·10⁻⁹), Horner in t²; then the octant.
/// IEEE single products, sums and quotients in this order; no library atan2 (the
/// backends' are not the same function). 0 when x = y = 0.
pub fn exact_atan2(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    let swap = ay > ax;
    let (lo, hi) = if swap { (ax, ay) } else { (ay, ax) };
    if hi == 0.0 {
        return 0.0;
    }
    let a = lo / hi;
    let (t, base) = if a > f32::from_bits(0x3ed4_13cd) {
        ((a - 1.0) / (a + 1.0), std::f32::consts::FRAC_PI_4)
    } else {
        (a, 0.0)
    };
    let t2 = t * t;
    let mut p = f32::from_bits(ATAN_C[11]);
    for k in (0..11).rev() {
        p = p * t2 + f32::from_bits(ATAN_C[k]);
    }
    let mut r = base + t * p;
    if swap {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = std::f32::consts::PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

/// Dihedral angle φ ∈ (−π, π] of nodes 1–2–3–4 about the 2→3 axis
/// ([`exact_atan2`], bit for bit with the kernel).
pub fn dihedral(x: [Vec3; 4]) -> f32 {
    let (b1, b2, b3) = (x[1] - x[0], x[2] - x[1], x[3] - x[2]);
    let (n1, n2) = (b1.cross(b2), b2.cross(b3));
    exact_atan2(b2.length() * b1.dot(n2), n1.dot(n2))
}

/// An angle difference of two angles in (−π, π] wrapped into (−π, π].
pub fn wrap_angle(a: f32) -> f32 {
    if a > std::f32::consts::PI {
        a - std::f32::consts::TAU
    } else if a <= -std::f32::consts::PI {
        a + std::f32::consts::TAU
    } else {
        a
    }
}

/// ∂φ/∂x of [`dihedral`] for the four nodes (Blondel and Karplus 1996); zero where
/// a plane is degenerate.
pub fn dihedral_jacobian(x: [Vec3; 4]) -> [Vec3; 4] {
    let (b1, b2, b3) = (x[1] - x[0], x[2] - x[1], x[3] - x[2]);
    let (n1, n2) = (b1.cross(b2), b2.cross(b3));
    let (l1, l2) = (n1.dot(n1), n2.dot(n2));
    let ll = b2.dot(b2);
    if l1 < 1e-12 || l2 < 1e-12 || ll < 1e-12 {
        return [Vec3::ZERO; 4];
    }
    let lb = ll.sqrt();
    let g1 = n1 * (-lb / l1);
    let g4 = n2 * (lb / l2);
    let p = -b1.dot(b2) / ll;
    let q = -b3.dot(b2) / ll;
    let g2 = g1 * (p - 1.0) - g4 * q;
    let g3 = g4 * (q - 1.0) - g1 * p;
    [g1, g2, g3, g4]
}

/// A torsion element's forces on its four nodes ([`ExplicitKind::JbeamTorsion`];
/// spec §9): the torque τ = −k·wrap(φ − φ₀) − c·φ̇ as forces τ·∂φ/∂xᵢ. Past the
/// deform torque (`tc`) the rest angle follows φ so the spring torque stays at it;
/// past the strength the element breaks (`broken`, no force). The torque is kept
/// as the last force.
pub fn torsion_force(beam: &mut Beam, n: &[Node; 4]) -> [Vec3; 4] {
    let x = n.map(|n| n.pos);
    let g = dihedral_jacobian(x);
    let phi = dihedral(x);
    let rate = g[0].dot(n[0].vel) + g[1].dot(n[1].vel) + g[2].dot(n[2].vel) + g[3].dot(n[3].vel);
    let mut d = wrap_angle(phi - beam.rest);
    if beam.k * d.abs() > beam.tc {
        // Yield: the rest angle moves so the spring torque is the deform torque.
        d = (beam.tc / beam.k).copysign(d);
        beam.rest = wrap_angle(phi - d);
    }
    let t = -beam.k * d - beam.c * rate;
    beam.force = t;
    if t.abs() > beam.strength {
        beam.broken = true;
        return [Vec3::ZERO; 4];
    }
    g.map(|gi| gi * t)
}

/// A coupler's stiffness and damping for node masses `ma`, `mb`
/// (`engine/docs/jbeam-semantics-spec.md` §7; BeamNG gives none, the nodes are
/// "considered one"): the stiffest pair spring that stays well inside the fixed
/// step's stability limit, ω·[`STEP`] = 0.4 on the pair's reduced mass μ, so
/// k = 0.16·μ/STEP², with damping at a damping ratio of 0.5, c = √(k·μ).
pub fn coupler_rates(ma: f32, mb: f32) -> (f32, f32) {
    let mu = ma * mb / (ma + mb);
    let k = 0.16 * mu / (STEP * STEP);
    (k, (k * mu).sqrt())
}

/// A coupler's force on node a (b takes the opposite): −k·(a − b) − c·(v_a − v_b);
/// when its magnitude passes the strength the coupler breaks (`broken`, no force).
/// The magnitude is kept as the last force.
pub fn coupler_force(beam: &mut Beam, na: &Node, nb: &Node) -> Vec3 {
    let f = -beam.k * (na.pos - nb.pos) - beam.c * (na.vel - nb.vel);
    let m = f.length();
    beam.force = m;
    if m > beam.strength {
        beam.broken = true;
        Vec3::ZERO
    } else {
        f
    }
}

/// A shock2 beam's force (spec §6): the rates by [`Shock2::rates`], no yield,
/// and the break test when |F| exceeds both the gate and the strength.
pub fn shock2_force(beam: &mut Beam, sh: &Shock2, na: &Node, nb: &Node, fast: bool) -> BeamOut {
    let d = na.pos - nb.pos;
    let q = d.dot(d);
    let (len, inv) = if fast {
        let inv = fast_inv_sqrt(q);
        (q * inv, inv)
    } else {
        exact_length(q)
    };
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
        slope: c,
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

    /// (α, β, γ, δ) of `p`, δ signed along the normal.
    pub fn local(&self, p: Vec3) -> (f32, f32, f32, f32) {
        let q = p - self.p0;
        let (alpha, beta, delta) = match self.inv {
            Some([ra, rb, rd]) => (ra.dot(q), rb.dot(q), rd.dot(q)),
            None => (0.0, 0.0, 0.0),
        };
        (alpha, beta, 1.0 - alpha - beta, delta)
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
