//! Explicit beams: wing panels (`wings`), the fuselage drag (`fusedrag`), airfoil
//! tables and self-centering command beams (`commands` with no input), written from
//! `engine/docs/explicit-beams-aero-commands-spec.md`. The solver
//! ([`crate::explicit_beams::ExplicitBeams`]) runs them in its step: wings and the
//! fuselage drag after the node pass, then the command pass, then the beams; the
//! wing break check once per frame before the origin rebase.
//!
//! Every expression is single precision except those the specification evaluates in
//! double precision (`f64` here, rounded once where it says).

use crate::explicit_beams::approx_pow;
use glam::Vec3;
use std::sync::Arc;

/// Entries of an airfoil table: 0.1° from −180° (index 0) to 180° (3600).
pub const AIRFOIL_ENTRIES: usize = 3601;
/// The airfoil a `fusedrag` line uses when it names none.
pub const DEFAULT_AIRFOIL: &str = "NACA0009.afl";
/// Environment variable naming the directory [`airfoil_from_env`] reads `.afl`
/// files from (Rubble ships none: the tables are loaded at run time).
pub const AIRFOIL_DIR_ENV: &str = "RUBBLE_AIRFOIL_DIR";

/// Lift, drag and moment coefficients per 0.1° of angle of attack.
#[derive(Clone, Debug, PartialEq)]
pub struct Airfoil {
    pub name: String,
    pub cl: Vec<f32>,
    pub cd: Vec<f32>,
    pub cm: Vec<f32>,
    /// The file was found (a missing file leaves the tables zero).
    pub loaded: bool,
}

impl Airfoil {
    /// All-zero tables (a missing file).
    pub fn zero(name: &str) -> Self {
        Self {
            name: name.to_string(),
            cl: vec![0.0; AIRFOIL_ENTRIES],
            cd: vec![0.0; AIRFOIL_ENTRIES],
            cm: vec![0.0; AIRFOIL_ENTRIES],
            loaded: false,
        }
    }

    /// The tables from an X-Plane airfoil file: header lines up to the one starting
    /// `alpha` ignored, then rows `angle cl cd cm` at an index from the angle's
    /// integer part and one-digit fraction (negated until the 0.0 row), gaps between
    /// rows filled linearly, reading stopped once index 3600 is stored.
    pub fn parse(name: &str, text: &str) -> Self {
        let mut a = Self::zero(name);
        a.loaded = true;
        let mut started = false;
        let mut seen_zero = false;
        let mut prev: Option<usize> = None;
        for line in text.lines() {
            let l = line.trim();
            if !started {
                started = l.starts_with("alpha");
                continue;
            }
            let mut it = l.split_whitespace();
            let Some(angle) = it.next() else {
                continue;
            };
            let vals: Vec<f32> = it.take(3).filter_map(|v| v.parse().ok()).collect();
            if vals.len() < 3 {
                continue;
            }
            let (int_part, frac_part) = angle.split_once('.').unwrap_or((angle, "0"));
            let (Ok(ip), Ok(fp)) = (int_part.parse::<i64>(), frac_part.parse::<i64>()) else {
                continue;
            };
            if ip == 0 && fp == 0 {
                seen_zero = true;
            }
            let frac = if seen_zero { fp } else { -fp };
            let idx = 10 * ip + frac + 1800;
            if !(0..AIRFOIL_ENTRIES as i64).contains(&idx) {
                continue;
            }
            let i = idx as usize;
            a.cl[i] = vals[0];
            a.cd[i] = vals[1];
            a.cm[i] = vals[2];
            if let Some(j) = prev.filter(|&j| i > j + 1) {
                for table in [&mut a.cl, &mut a.cd, &mut a.cm] {
                    let (vj, vi) = (table[j], table[i]);
                    let span = (i - j) as f32;
                    for k in 0..i - j - 1 {
                        table[j + 1 + k] = vj + ((k + 1) as f32 * (vi - vj)) / span;
                    }
                }
            }
            prev = Some(i);
            if i == AIRFOIL_ENTRIES - 1 {
                break;
            }
        }
        a
    }

    /// (lift, drag, moment) coefficients at angle `a` (degrees) with chord ratio
    /// `cr` and deflection `d` (degrees): the 0.1° bin at or below the wrapped angle,
    /// no interpolation; the drag bin shifted by the deflection.
    pub fn lookup(&self, a: f32, cr: f32, d: f32) -> (f32, f32, f32) {
        let a = wrap_angle(a);
        let i = bin(a);
        let ad = wrap_angle((a as f64 + 1.15 * (1.0 - cr as f64) * d as f64) as f32);
        let id = bin(ad);
        let sg = if d < 0.0 { -1.0f64 } else { 1.0 };
        let root = d.abs().sqrt() as f64;
        let cz = (self.cl[i] as f64 - 0.66 * sg * (1.0 - cr as f64) * root) as f32;
        let cx = (self.cd[id] as f64 + 0.00015 * (1.0 - cr as f64) * d as f64 * d as f64) as f32;
        let cm = (self.cm[i] as f64 + 0.20 * sg * (1.0 - cr as f64) * root) as f32;
        (cz, cx, cm)
    }
}

/// An angle (degrees) wrapped into [−180, 180].
fn wrap_angle(a: f32) -> f32 {
    let k = (a as f64 / 360.0).trunc();
    let mut w = a - (k * 360.0) as f32;
    if w > 180.0 {
        w -= 360.0;
    }
    if w < -180.0 {
        w += 360.0;
    }
    w
}

/// The table index of a wrapped angle (truncated, kept in range).
fn bin(a: f32) -> usize {
    let i = ((a + 180.0) * 10.0) as i32;
    i.clamp(0, AIRFOIL_ENTRIES as i32 - 1) as usize
}

/// Reads `name` from the directory in [`AIRFOIL_DIR_ENV`], if set and present.
pub fn airfoil_from_env(name: &str) -> Option<String> {
    let dir = std::env::var_os(AIRFOIL_DIR_ENV)?;
    std::fs::read_to_string(std::path::Path::new(&dir).join(name)).ok()
}

/// Air density at height `y` (m) for wings: the standard atmosphere through the
/// approximate power, the last product in double precision.
pub fn wing_density(y: f32) -> f32 {
    let t = (1.0 - (0.0065 * y as f64) / 288.15) as f32;
    let p = 101_325.0f32 * approx_pow(t, 5.24947);
    (p as f64 * 0.0000120896) as f32
}

/// Air density at height `y` (m) for the fuselage drag: as [`wing_density`] with
/// 288.1 for 288.15 and a single-precision last product.
pub fn fuse_density(y: f32) -> f32 {
    let t = (1.0 - (0.0065 * y as f64) / 288.1) as f32;
    let p = 101_325.0f32 * approx_pow(t, 5.24947);
    p * 0.0000120896f32
}

/// `v` scaled by the single-precision reciprocal of its length (unchanged at 0).
pub fn normalise(v: Vec3) -> Vec3 {
    let l = v.length();
    if l > 0.0 {
        v * (1.0 / l)
    } else {
        v
    }
}

/// The signed angle of attack (degrees) between the chord `c` and the airflow `w`
/// projected onto the plane of `c` and the normal `n`, its sign from the span `b`:
/// the rotation from `c` to the projected flow, taken as a quaternion and back to
/// an angle, in the specification's order of operations.
pub fn angle_of_attack(c: Vec3, n: Vec3, b: Vec3, w: Vec3) -> f32 {
    let q = normalise(n.cross(c));
    let g = -w;
    let row = |i: usize| -> Vec3 {
        let qi = q[i];
        let mut r = Vec3::new(-(qi * q.x), -(qi * q.y), -(qi * q.z));
        r[i] = 1.0 - qi * qi;
        r
    };
    let p = Vec3::new(
        row(0).x * g.x + row(0).y * g.y + row(0).z * g.z,
        row(1).x * g.x + row(1).y * g.y + row(1).z * g.z,
        row(2).x * g.x + row(2).y * g.y + row(2).z * g.z,
    );
    let u = c;
    let t = -p;
    let a = (u.dot(u) * t.dot(t)).sqrt();
    let beta = a + t.dot(u);
    if (beta - 2.0 * a).abs() <= f32::EPSILON || a == 0.0 {
        return 0.0;
    }
    let (qw, axis) = if beta < 1e-6f32 * a {
        let axis = if u.x.abs() > u.z.abs() {
            Vec3::new(-u.y, u.x, 0.0)
        } else {
            Vec3::new(0.0, -u.z, u.y)
        };
        (0.0f32, axis)
    } else {
        (beta, u.cross(t))
    };
    let nq = (qw * qw + axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt();
    let inv = 1.0 / nq;
    let (qw, axis) = (qw * inv, axis * inv);
    let s2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
    let (theta, axis) = if s2 > 0.0 {
        let acos = if qw >= 1.0 {
            0.0
        } else if qw <= -1.0 {
            std::f32::consts::PI
        } else {
            qw.acos()
        };
        (2.0 * acos, axis * (1.0 / s2.sqrt()))
    } else {
        (0.0, Vec3::X)
    };
    let r2d = 180.0f32 / (4.0f32 * 1.0f32.atan());
    let alpha = theta * r2d;
    if axis.dot(b) > 0.0 {
        -alpha
    } else {
        alpha
    }
}

/// Which back corner nodes a wing's induced drag goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WingSide {
    Left,
    Right,
}

/// Induced drag of a wing at the end of a chain: the chain's span and area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Induced {
    pub span: f32,
    pub area: f32,
    pub side: WingSide,
}

/// Wing slots in [`Wing::nodes`].
pub const FLD: usize = 0;
pub const FRD: usize = 1;
pub const FLU: usize = 2;
pub const FRU: usize = 3;
pub const BLD: usize = 4;
pub const BRD: usize = 5;
pub const BLU: usize = 6;
pub const BRU: usize = 7;

/// One wing panel.
#[derive(Clone, Debug, PartialEq)]
pub struct Wing {
    /// Node per slot ([`FLD`] … [`BRU`]); slots may repeat nodes.
    pub nodes: [u32; 8],
    pub efficacy: f32,
    pub chord_ratio: f32,
    /// Control surface `S`, `T`, `U` or `V`: the deflection rotates the angle.
    pub stabilator: bool,
    /// Control deflection (degrees): 0 with no input.
    pub deflection: f32,
    /// Its table in [`crate::explicit_beams::ExplicitBeams::airfoils`].
    pub airfoil: u32,
    /// Squared break reference: a doubled half-panel area above it breaks the wing.
    pub break_ref: f32,
    pub induced: Option<Induced>,
    pub broken: bool,
}

/// A wing's forces in one step: induced drag (on its side's back corners), the
/// front slots' force and the back slots' force.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WingForces {
    pub induced: Option<(Vec3, WingSide)>,
    pub front: Vec3,
    pub back: Vec3,
}

impl Wing {
    /// From spawn positions `x` (indexed by node): the break reference and the
    /// panel area; chains and induced drag are set by [`chain_wings`].
    pub fn new(
        nodes: [u32; 8],
        efficacy: f32,
        chord_ratio: f32,
        control: char,
        airfoil: u32,
        x: &[Vec3],
    ) -> Self {
        let p = |s: usize| x[nodes[s] as usize];
        let xa = p(FRD) - p(FLD);
        let t1 = (2.0 * xa.cross(p(BLD) - p(FLD)).length() as f64) as f32;
        let t2 = (2.0 * xa.cross(p(BRD) - p(FRD)).length() as f64) as f32;
        let s = if t1 > t2 { t1 } else { t2 };
        Self {
            nodes,
            efficacy,
            chord_ratio,
            stabilator: matches!(control, 'S' | 'T' | 'U' | 'V'),
            deflection: 0.0,
            airfoil,
            break_ref: s * s,
            induced: None,
            broken: false,
        }
    }

    /// Panel area from positions `x`: half the sum of its two triangles' doubled
    /// areas.
    pub fn area(&self, x: &[Vec3]) -> f32 {
        let p = |s: usize| x[self.nodes[s] as usize];
        let a = (p(FRD) - p(FLD)).cross(p(BLD) - p(FLD)).length();
        let b = (p(FRD) - p(BRD)).cross(p(BLD) - p(BRD)).length();
        (a + b) * 0.5
    }

    /// The per-frame break check on world positions `x`; breaking is permanent.
    pub fn check_break(&mut self, x: impl Fn(u32) -> Vec3) {
        let p = |s: usize| x(self.nodes[s]);
        let xa = p(FRD) - p(FLD);
        let l = xa.cross(p(BLD) - p(FLD));
        let r = xa.cross(p(BRD) - p(FRD));
        self.broken = self.broken || l.dot(l) > self.break_ref || r.dot(r) > self.break_ref;
    }

    /// This step's forces from relative positions `x`, velocities `v` and the
    /// world height of the front-left-down node `y`.
    pub fn forces(
        &self,
        x: impl Fn(u32) -> Vec3,
        v: impl Fn(u32) -> Vec3,
        y: f32,
        table: &Airfoil,
    ) -> WingForces {
        let n8 = self.nodes;
        let xs = |s: usize| x(n8[s]);
        let w = -((v(n8[FLD]) + v(n8[FRD])) * 0.5);
        let speed = w.length();
        let c = ((xs(BLD) - xs(FLD)) + (xs(BRD) - xs(FRD))) * 0.5;
        let chord = c.length();
        let b = ((xs(FRD) - xs(FLD)) + (xs(BRD) - xs(BLD))) * 0.5;
        let span = b.length();
        let lift_dir = b.cross(-w);
        let s = span * chord;
        let n = normalise(c.cross(b));
        let alpha = angle_of_attack(c, n, b, w);
        let (cz, cx, cm) = if self.stabilator {
            table.lookup(alpha - self.deflection, self.chord_ratio, 0.0)
        } else {
            table.lookup(alpha, self.chord_ratio, self.deflection)
        };
        let rho = wing_density(y) as f64;
        let vv = speed as f64;
        let kd = (cx as f64 * 0.5 * rho * vv * s as f64) as f32;
        let mut wf = w * kd;
        let induced = self.induced.map(|ind| {
            let num = (cx * cx) as f64 * 0.25 * rho * vv * ind.area as f64 * ind.area as f64;
            let den = 3.14159 * ind.span as f64 * ind.span as f64;
            (w * (num / den) as f32, ind.side)
        });
        let kl = (cz as f64 * 0.5 * rho * vv * chord as f64) as f32;
        wf += lift_dir * kl;
        let mu = ((-cm) as f64 * 0.5 * rho * vv * vv * s as f64) as f32;
        let e = self.efficacy as f64;
        let e1 = (e * 0.75 / 4.0) as f32;
        let e2 = (e * 0.25 / 4.0) as f32;
        let em = self.efficacy * mu;
        WingForces {
            induced,
            front: wf * e1 + n * em,
            back: wf * e2 - n * (em / 3.0),
        }
    }
}

/// Group wings (in file order) into chains, a wing continuing its predecessor's
/// chain when its FRD node is that wing's FLD node, and give each chain's first
/// wing induced drag on its right and its last on its left (a lone wing: left), with
/// the chain's span and summed panel area from spawn positions `x`.
pub fn chain_wings(wings: &mut [Wing], x: &[Vec3]) {
    let mut start = 0;
    for i in 0..=wings.len() {
        let continues = i > 0 && i < wings.len() && wings[i].nodes[FRD] == wings[i - 1].nodes[FLD];
        if i > 0 && !continues {
            let last = i - 1;
            let area = wings[start..=last]
                .iter()
                .fold(0.0f32, |sum, w| sum + w.area(x));
            let span =
                (x[wings[start].nodes[FRD] as usize] - x[wings[last].nodes[FLD] as usize]).length();
            wings[start].induced = Some(Induced {
                span,
                area,
                side: WingSide::Right,
            });
            wings[last].induced = Some(Induced {
                span,
                area,
                side: WingSide::Left,
            });
            start = i;
        }
    }
}

/// The `fusedrag` line as simulated.
#[derive(Clone, Debug, PartialEq)]
pub struct FuseDrag {
    /// Front node P.
    pub front: u32,
    /// Width (an area, used as a length): at or below 0 there is no force.
    pub width: f32,
    pub airfoil: u32,
}

impl FuseDrag {
    /// The per-node force D from P's velocity `v` and world height `y`, for a
    /// vehicle of `nodes` nodes; `None` when the width is not positive (D keeps its
    /// value). The fuselage axis runs from P to P, so its area is 0 and the airfoil
    /// adds nothing: D = (W²/4)·ρ·|v|·(−v) / N.
    pub fn force(&self, v: Vec3, y: f32, nodes: usize, table: &Airfoil) -> Option<Vec3> {
        if self.width <= 0.0 {
            return None;
        }
        let a = -v;
        let speed = a.length();
        let s = 0.0f32 * self.width;
        let cx = table.cd[1800];
        let rho = fuse_density(y);
        let k = ((cx * s) as f64 + (self.width * self.width) as f64 * 0.5)
            * 0.5
            * rho as f64
            * speed as f64
            / nodes as f64;
        Some(a * k as f32)
    }
}

/// A self-centering command beam (option `c`): with no input it drives its rest
/// length toward `centre` × its spawn length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenteringCommand {
    pub beam: u32,
    pub shorten_rate: f32,
    pub lengthen_rate: f32,
    pub max_contraction: f32,
    pub max_extension: f32,
    /// (max_extension − max_contraction)/2 + max_contraction.
    pub centre: f32,
    /// The contraction entry runs before the extension entry (its key is lower, or
    /// the keys are equal).
    pub contract_first: bool,
    /// −1 contracting, +1 extending, 0 holding.
    pub mode: i32,
}

impl CenteringCommand {
    pub fn new(
        beam: u32,
        shorten_rate: f32,
        lengthen_rate: f32,
        max_contraction: f32,
        max_extension: f32,
        contract_key: i32,
        extend_key: i32,
    ) -> Self {
        Self {
            beam,
            shorten_rate,
            lengthen_rate,
            max_contraction,
            max_extension,
            centre: (max_extension - max_contraction) / 2.0 + max_contraction,
            contract_first: contract_key <= extend_key,
            mode: 0,
        }
    }

    /// One step of both entries in key order: the new rest length from the rest
    /// length `rest` and the spawn length `rest0`, with step `h`.
    pub fn update(&mut self, mut rest: f32, rest0: f32, h: f32) -> f32 {
        let order: [i32; 2] = if self.contract_first {
            [-1, 1]
        } else {
            [1, -1]
        };
        for sigma in order {
            if rest0 == 0.0 || rest == 0.0 {
                continue;
            }
            let r = rest / rest0;
            if (((r - self.centre).abs()) as f64) < 1e-4 {
                self.mode = 0;
            } else {
                let old = self.mode;
                self.mode = if r > self.centre { -1 } else { 1 };
                if old != 0 && old != self.mode {
                    rest = self.centre * rest0;
                    self.mode = 0;
                }
            }
            let r = rest / rest0;
            let may =
                (sigma == 1 && r < self.max_extension) || (sigma == -1 && r > self.max_contraction);
            if may {
                let v = if sigma * self.mode > 0 { 1.0f32 } else { 0.0 };
                let rate = if sigma > 0 {
                    self.lengthen_rate
                } else {
                    self.shorten_rate
                };
                let cf = 1.0f32;
                let x = rate * v * cf * h / rest;
                let x = if sigma > 0 { x } else { -x };
                rest = ((1.0 + x as f64) * rest as f64) as f32;
            }
        }
        rest
    }
}

/// Shared tables for a vehicle: one per distinct file name.
pub type AirfoilSet = Vec<Arc<Airfoil>>;
