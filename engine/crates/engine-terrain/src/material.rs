//! The materials of the MPM active region (`engine/docs/deformable-terrain.md`,
//! "T2 kernel design" and "T5 materials"): their physical parameters
//! ([`SandMaterial`] for dry sand and cohesive soil, [`RockMaterial`] for
//! damage-softening rock, either as a [`TerrainMaterial`]) and the CPU mirror of
//! `engine/slang/physics/mpm_material.slang`, step for step in f32, for tests and
//! the host. engine-webgpu `tests/mpm_material.rs` checks the GPU code against it.
//!
//! - [`svd3`] (`mpmSvd3`), the 3×3 SVD of McAdams et al. (2011).
//! - [`sand_project`] (`mpmSandProject`), the Drucker–Prager return map in Hencky
//!   strain (Klár et al. 2016) with the log-volume correction (Tampubolon et al.
//!   2017).
//! - [`sand_project_mohr_coulomb`] (`mpmSandProjectMohrCoulomb`), the
//!   Mohr–Coulomb return map in Hencky strain (the default yield surface).
//! - [`soil_project_mohr_coulomb`] (`mpmSoilProjectMohrCoulomb`), Mohr–Coulomb
//!   with cohesion in stress space and its tension cutoff at the apex.
//! - [`rock_damage`] (`mpmRockDamage`), Rankine damage with crack-band
//!   exponential softening, and [`principal_kirchhoff`] (`mpmPrincipalKirchhoff`).
//! - [`hencky_kirchhoff`] (`mpmHenckyKirchhoff`), the Kirchhoff stress.
//! - [`friction_alpha`] (`mpmFrictionAlpha`).

use crate::config::TerrainError;
use glam::{Mat3, Quat, Vec3};

/// Jacobi sweeps of [`svd3`] (`MPM_SVD_SWEEPS`), each conjugating the pairs (0,1),
/// (0,2), (1,2).
pub const SVD_SWEEPS: u32 = 6;
/// Largest |entry| the SVD and the return map accept (`MPM_INPUT_LIMIT`): keeps FᵀF
/// and the log strains finite in f32.
pub const INPUT_LIMIT: f32 = 1.0e18;
/// Smallest |σ| the return map takes the log of (`MPM_SIGMA_FLOOR`).
pub const SIGMA_FLOOR: f32 = 1.0e-4;
/// Largest elastic log stretch per axis a cohesion may hold at the apex,
/// c·cot φ / (3λ + 2μ): past it the material would hold an e-fold stretch
/// before it separates.
pub const MAX_APEX_STRAIN: f32 = 1.0;

/// (1 + √2)²: below it the approximate half angle is within π/8.
const SVD_GAMMA: f32 = 5.828_427;
const COS_PI_8: f32 = 0.923_879_5;
const SIN_PI_8: f32 = 0.382_683_43;

/// The yield surface bounding sand's elastic stress.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum YieldSurface {
    /// Mohr–Coulomb: φ is the friction angle in every stress state.
    #[default]
    MohrCoulomb,
    /// Klár et al.'s Drucker–Prager cone, matching Mohr–Coulomb only on the
    /// compression meridian (stronger elsewhere: in triaxial extension a 30°
    /// cone acts like about 46°, so spreading flows stop short).
    DruckerPrager,
}

/// Elastoplastic sand or soil: Hencky elasticity bounded by a frictional yield
/// surface, cohesionless (dry sand) or cohesive (soil).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandMaterial {
    /// Bulk density (kg/m³).
    pub density: f32,
    /// Young's modulus E (Pa).
    pub youngs_modulus: f32,
    /// Poisson's ratio ν.
    pub poisson_ratio: f32,
    /// Internal friction angle φ (radians).
    pub friction_angle: f32,
    /// Cohesion c (Pa; 0 for dry sand), in stress space: the Mohr–Coulomb
    /// surface is (τ₁ − τ₃) + (τ₁ + τ₃)·sin φ = 2c·cos φ, cut off in tension at
    /// its apex, the hydrostatic tension c·cot φ. The Drucker–Prager cone takes
    /// the same apex. A cohesive material needs φ > 0.
    pub cohesion: f32,
    /// Shape of the yield surface.
    pub yield_surface: YieldSurface,
}

impl Default for SandMaterial {
    /// Dry sand. Typical values for dry quartz sand in geotechnical references:
    /// bulk density 1500–1700 kg/m³, E 10–100 MPa, ν 0.25–0.35, φ 28–35°, and no
    /// cohesion.
    fn default() -> Self {
        Self {
            density: 1600.0,
            youngs_modulus: 50.0e6,
            poisson_ratio: 0.3,
            friction_angle: 30f32.to_radians(),
            cohesion: 0.0,
            yield_surface: YieldSurface::MohrCoulomb,
        }
    }
}

impl SandMaterial {
    /// Rejects nonphysical parameters: density > 0, E > 0, ν in (−1, 0.5), φ in
    /// [0, 90°), cohesion ≥ 0 (and then φ > 0, with its apex stretch
    /// [`Self::apex_strain`] at most [`MAX_APEX_STRAIN`]), all finite.
    pub fn validate(&self) -> Result<(), TerrainError> {
        let bad = |c: bool, why| {
            if c {
                Err(TerrainError::InvalidConfig(why))
            } else {
                Ok(())
            }
        };
        bad(
            !(self.density.is_finite() && self.density > 0.0),
            "sand density must be finite and positive",
        )?;
        bad(
            !(self.youngs_modulus.is_finite() && self.youngs_modulus > 0.0),
            "sand Young's modulus must be finite and positive",
        )?;
        bad(
            !(self.poisson_ratio > -1.0 && self.poisson_ratio < 0.5),
            "sand Poisson's ratio must lie in (-1, 0.5)",
        )?;
        bad(
            !(self.friction_angle >= 0.0 && self.friction_angle < std::f32::consts::FRAC_PI_2),
            "sand friction angle must lie in [0, 90°)",
        )?;
        bad(
            !(self.cohesion.is_finite() && self.cohesion >= 0.0),
            "sand cohesion must be finite and non-negative (Pa)",
        )?;
        bad(
            self.cohesion > 0.0 && self.friction_angle <= 0.0,
            "cohesive soil needs a friction angle above 0: its tension cutoff is the apex c·cot φ",
        )?;
        bad(
            !(self.apex_strain().is_finite() && self.apex_strain() <= MAX_APEX_STRAIN),
            "sand cohesion's apex stretch c·cot φ / (3λ + 2μ) must be at most 1 (log strain)",
        )?;
        Ok(())
    }

    /// The elastic log stretch per axis at the apex of the yield surface, the
    /// hydrostatic tension c·cot φ: c·cot φ / (3λ + 2μ) (0 without cohesion).
    /// The Drucker–Prager return map takes it as its tip offset.
    pub fn apex_strain(&self) -> f32 {
        if self.cohesion == 0.0 {
            return 0.0;
        }
        self.cohesion / self.friction_angle.tan() / (3.0 * self.lambda() + 2.0 * self.mu())
    }

    /// c·cos φ (Pa): the cohesive soil kernel's word.
    pub fn cohesion_cos_phi(&self) -> f32 {
        self.cohesion * self.friction_angle.cos()
    }

    /// Shear modulus μ = E / 2(1 + ν) (Pa).
    pub fn mu(&self) -> f32 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Lamé's first parameter λ = Eν / (1 + ν)(1 − 2ν) (Pa).
    pub fn lambda(&self) -> f32 {
        let nu = self.poisson_ratio;
        self.youngs_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// The Drucker–Prager α of [`Self::friction_angle`].
    pub fn alpha(&self) -> f32 {
        friction_alpha(self.friction_angle)
    }

    /// P-wave speed c = √((λ + 2μ) / ρ) (m/s), which bounds the explicit substep.
    pub fn p_wave_speed(&self) -> f32 {
        ((self.lambda() + 2.0 * self.mu()) / self.density).sqrt()
    }
}

/// Width of the crack band in cells. The narrowest opening the grid holds is
/// a velocity jump between two neighbouring nodes, and with quadratic B-splines
/// every particle within a cell of the plane between them reads both (a
/// particle's stencil is the three nodes around it), so a crack strains the
/// particles of two cells. engine-webgpu `tests/terrain_materials.rs` measures
/// cracks exactly two cells wide at two cell sizes. The softening is
/// regularised over h = `CRACK_BAND_CELLS`·dx.
pub const CRACK_BAND_CELLS: f32 = 2.0;

/// Damage-softening rock (continuum damage in the spirit of Wolper et al. 2019,
/// CD-MPM): Hencky elasticity whose tensile principal stresses are scaled by
/// (1 − d), with a per-particle damage d ∈ [0, 1] that grows when the largest
/// principal stress of the undamaged material passes the tensile strength
/// (Rankine), and exponential softening regularised by the crack band
/// h = [`CRACK_BAND_CELLS`]·dx (Bažant & Oh 1983): under uniaxial tension the
/// stress past the peak is f_t·exp(−(ε − f_t/E)/ε_s) with ε_s chosen so the
/// energy dissipated per unit volume is G_f/h, hence G_f per crack area whatever
/// the cell size. Compressive principal stresses are never degraded (cracks
/// close).
///
/// The band must stay below `2·E·G_f/f_t²` (a cell below
/// [`Self::max_cell_size`]): past it the softening branch would have to snap
/// back. That is an error ([`TerrainError::CrackBand`]), never a silently
/// weakened rock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RockMaterial {
    /// Bulk density (kg/m³).
    pub density: f32,
    /// Young's modulus E (Pa).
    pub youngs_modulus: f32,
    /// Poisson's ratio ν.
    pub poisson_ratio: f32,
    /// Tensile strength f_t (Pa): the uniaxial tensile stress at which damage
    /// starts.
    pub tensile_strength: f32,
    /// Fracture energy G_f (J/m²): the energy a crack dissipates per unit area
    /// as it opens fully.
    pub fracture_energy: f32,
}

impl Default for RockMaterial {
    /// Sound sedimentary or igneous rock, inside the ranges of rock-mechanics
    /// references (E 10–70 GPa, ν 0.15–0.3, f_t 2–15 MPa, G_f 30–200 J/m²):
    /// ρ = 2600 kg/m³, E = 30 GPa, ν = 0.25, f_t = 5 MPa, G_f = 100 J/m²
    /// (characteristic length E·G_f/f_t² = 0.12 m, so cells up to 0.12 m: finer
    /// than most terrain regions, which then fail with
    /// [`TerrainError::CrackBand`] until given rock parameters for their scale).
    fn default() -> Self {
        Self {
            density: 2600.0,
            youngs_modulus: 30.0e9,
            poisson_ratio: 0.25,
            tensile_strength: 5.0e6,
            fracture_energy: 100.0,
        }
    }
}

impl RockMaterial {
    /// Rejects nonphysical parameters: density, E, f_t and G_f finite and
    /// positive, ν in (−1, 0.5).
    pub fn validate(&self) -> Result<(), TerrainError> {
        let bad = |c: bool, why| {
            if c {
                Err(TerrainError::InvalidConfig(why))
            } else {
                Ok(())
            }
        };
        bad(
            !(self.density.is_finite() && self.density > 0.0),
            "rock density must be finite and positive",
        )?;
        bad(
            !(self.youngs_modulus.is_finite() && self.youngs_modulus > 0.0),
            "rock Young's modulus must be finite and positive",
        )?;
        bad(
            !(self.poisson_ratio > -1.0 && self.poisson_ratio < 0.5),
            "rock Poisson's ratio must lie in (-1, 0.5)",
        )?;
        bad(
            !(self.tensile_strength.is_finite() && self.tensile_strength > 0.0),
            "rock tensile strength must be finite and positive (Pa)",
        )?;
        bad(
            !(self.fracture_energy.is_finite() && self.fracture_energy > 0.0),
            "rock fracture energy must be finite and positive (J/m²)",
        )?;
        bad(
            !self.band_stress().is_finite(),
            "rock E·G_f/f_t must be finite in f32",
        )?;
        Ok(())
    }

    /// Shear modulus μ = E / 2(1 + ν) (Pa).
    pub fn mu(&self) -> f32 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Lamé's first parameter λ = Eν / (1 + ν)(1 − 2ν) (Pa).
    pub fn lambda(&self) -> f32 {
        let nu = self.poisson_ratio;
        self.youngs_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// P-wave speed c = √((λ + 2μ) / ρ) (m/s), which bounds the explicit substep.
    pub fn p_wave_speed(&self) -> f32 {
        ((self.lambda() + 2.0 * self.mu()) / self.density).sqrt()
    }

    /// Hillerborg's characteristic length E·G_f/f_t² (m).
    pub fn characteristic_length(&self) -> f32 {
        self.youngs_modulus * self.fracture_energy / (self.tensile_strength * self.tensile_strength)
    }

    /// The largest cell size the crack band allows, 2·E·G_f/f_t² over
    /// [`CRACK_BAND_CELLS`] (m): there the softening branch is vertical (all of
    /// G_f/h is the elastic energy at the peak).
    pub fn max_cell_size(&self) -> f32 {
        2.0 * self.characteristic_length() / CRACK_BAND_CELLS
    }

    /// E·G_f/(f_t·[`CRACK_BAND_CELLS`]) (Pa·m), the kernel's word: the softening
    /// stress is this over the cell size, less f_t/2.
    pub fn band_stress(&self) -> f32 {
        self.youngs_modulus * self.fracture_energy / (self.tensile_strength * CRACK_BAND_CELLS)
    }

    /// The softening stress σ_s = E·ε_s = E·G_f/(f_t·h) − f_t/2 (Pa) over the
    /// crack band h = [`CRACK_BAND_CELLS`]·`dx`, computed as the kernel does;
    /// fails with [`TerrainError::CrackBand`] unless it is positive (dx below
    /// [`Self::max_cell_size`]).
    pub fn softening_stress(&self, dx: f32) -> Result<f32, TerrainError> {
        let s = self.band_stress() / dx - 0.5 * self.tensile_strength;
        if !(dx > 0.0 && s > 0.0 && s.is_finite()) {
            return Err(TerrainError::CrackBand {
                cell: dx,
                max: self.max_cell_size(),
            });
        }
        Ok(s)
    }
}

/// The constitutive model of an active region's material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TerrainMaterial {
    /// No stress: transfers only (tests).
    Inert,
    /// Dry sand (no cohesion) or cohesive soil.
    Sand(SandMaterial),
    /// Damage-softening rock.
    Rock(RockMaterial),
}

impl TerrainMaterial {
    pub fn validate(&self) -> Result<(), TerrainError> {
        match self {
            Self::Inert => Ok(()),
            Self::Sand(m) => m.validate(),
            Self::Rock(m) => m.validate(),
        }
    }

    /// P-wave speed (m/s; 0 without stress).
    pub fn p_wave_speed(&self) -> f32 {
        match self {
            Self::Inert => 0.0,
            Self::Sand(m) => m.p_wave_speed(),
            Self::Rock(m) => m.p_wave_speed(),
        }
    }
}

impl From<SandMaterial> for TerrainMaterial {
    fn from(m: SandMaterial) -> Self {
        Self::Sand(m)
    }
}

impl From<&SandMaterial> for TerrainMaterial {
    fn from(m: &SandMaterial) -> Self {
        Self::Sand(*m)
    }
}

/// `None` is [`TerrainMaterial::Inert`] (the form `terrain_set_material` has
/// always taken).
impl From<Option<&SandMaterial>> for TerrainMaterial {
    fn from(m: Option<&SandMaterial>) -> Self {
        m.map_or(Self::Inert, |m| Self::Sand(*m))
    }
}

impl From<RockMaterial> for TerrainMaterial {
    fn from(m: RockMaterial) -> Self {
        Self::Rock(m)
    }
}

impl From<&RockMaterial> for TerrainMaterial {
    fn from(m: &RockMaterial) -> Self {
        Self::Rock(*m)
    }
}

/// `F = U·diag(sigma)·Vᵀ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd3 {
    /// A rotation (det +1).
    pub u: Mat3,
    /// Decreasing in magnitude; only `z` is negative (when det F < 0).
    pub sigma: Vec3,
    /// A rotation (det +1).
    pub v: Mat3,
}

/// The result of [`sand_project`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandProjection {
    /// Elastic singular values after the return map (positive).
    pub sigma: Vec3,
    /// Log of the plastic volume ratio carried to the next step.
    pub log_jp: f32,
}

/// F's entries pass [`svd3`] without overflow (false for NaN or ±inf).
pub fn svd_input_valid(f: Mat3) -> bool {
    f.to_cols_array().iter().all(|x| x.abs() <= INPUT_LIMIT)
}

/// The return map's inputs are in range (false for NaN or ±inf).
pub fn sand_input_valid(sigma: Vec3, log_jp: f32) -> bool {
    sigma.to_array().iter().all(|x| x.abs() <= INPUT_LIMIT) && log_jp.abs() <= INPUT_LIMIT
}

/// √(a² + b²) without squaring tiny or huge values (a 1e-23 squared underflows
/// to 0 in f32, and h = p − r is exactly 0 when singular values repeat). Mirrors
/// `mpmHypot`.
fn hypot(a: f32, b: f32) -> f32 {
    let (x, y) = (a.abs(), b.abs());
    let (hi, lo) = (x.max(y), x.min(y));
    if hi == 0.0 {
        return 0.0;
    }
    let r = lo / hi;
    hi * (1.0 + r * r).sqrt()
}

/// One Jacobi conjugation S ← GᵀSG on the pair (p, q), entries permuted to
/// (p, q, k); G turns e_p towards e_q. Returns McAdams' approximate half angle
/// (cos, sin), or π/8 where that approximation would exceed π/8. A pair already
/// diagonal (s_pq = 0) is not turned, so a diagonal F (F = I at rest) stays exact.
fn jacobi_pair(
    spp: &mut f32,
    sqq: &mut f32,
    spq: &mut f32,
    spk: &mut f32,
    sqk: &mut f32,
) -> (f32, f32) {
    let (mut ch, mut sh) = (2.0 * (*spp - *sqq), *spq);
    if sh == 0.0 {
        return (1.0, 0.0);
    }
    // Only the ratio matters: scale to the larger first, so the squares neither
    // underflow nor overflow.
    let m = ch.abs().max(sh.abs());
    ch /= m;
    sh /= m;
    if SVD_GAMMA * sh * sh < ch * ch {
        let w = 1.0 / (ch * ch + sh * sh).sqrt();
        ch *= w;
        sh *= w;
    } else {
        ch = COS_PI_8;
        sh = SIN_PI_8;
    }
    let c = ch * ch - sh * sh;
    let s = 2.0 * ch * sh;
    let (app, aqq, apq, apk, aqk) = (*spp, *sqq, *spq, *spk, *sqk);
    *spp = c * c * app + 2.0 * c * s * apq + s * s * aqq;
    *sqq = s * s * app - 2.0 * c * s * apq + c * c * aqq;
    *spq = (c * c - s * s) * apq - c * s * (app - aqq);
    *spk = c * apk + s * aqk;
    *sqk = c * aqk - s * apk;
    (ch, sh)
}

/// Swap two columns, negating one, so the matrix keeps its determinant.
fn swap_cols(a: &mut Vec3, b: &mut Vec3) {
    let t = *a;
    *a = *b;
    *b = -t;
}

/// (x, y) ← (c·x + s·y, c·y − s·x).
fn rotate_pair(c: f32, s: f32, x: &mut f32, y: &mut f32) {
    let t = *x;
    *x = c * t + s * *y;
    *y = c * *y - s * t;
}

/// The Givens rotation that zeroes `b` against the pivot `a`, leaving the pivot
/// √(a² + b²) ≥ 0. No rotation when both are zero.
fn givens(a: f32, b: f32) -> (f32, f32) {
    let r = hypot(a, b);
    if r > 0.0 {
        (a / r, b / r)
    } else {
        (1.0, 0.0)
    }
}

/// (a, b) ← (c·a + s·b, c·b − s·a): columns a, b times the rotation that turns
/// e_a towards e_b.
fn rotate_cols(c: f32, s: f32, a: &mut Vec3, b: &mut Vec3) {
    let t = *a;
    *a = c * t + s * *b;
    *b = c * *b - s * t;
}

/// The rotation (c, s) with GᵀMG diagonal for the symmetric M = [[p, q], [q, r]]:
/// t = tan θ is the root of q·t² + (p − r)·t − q = 0 with |t| ≤ 1.
fn sym2_jacobi(p: f32, q: f32, r: f32) -> (f32, f32) {
    if q == 0.0 {
        return (1.0, 0.0);
    }
    let h = p - r;
    let root = hypot(h, 2.0 * q);
    let t = 2.0 * q / if h >= 0.0 { h + root } else { h - root };
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, t * c)
}

/// Rotate rows (p, q) of the columns `b` by (c, s).
macro_rules! rotate_rows {
    ($c:expr, $s:expr, [$($b:ident),*], $p:ident, $q:ident) => {
        $(rotate_pair($c, $s, &mut $b.$p, &mut $b.$q);)*
    };
}

/// The SVD by McAdams et al. (2011), "Computing the singular value decomposition
/// of 3×3 matrices with minimal branching and elementary floating point
/// operations": [`SVD_SWEEPS`] Jacobi sweeps on FᵀF with V accumulated as a
/// quaternion, the columns of B = F·V sorted by decreasing norm, then QR of B by
/// Givens rotations (U = G₁G₂G₃; R's diagonal is sigma), and a 2×2 SVD of R's
/// trailing block. Mirrors `mpmSvd3`.
///
/// ‖UΣVᵀ − F‖ is about 1e-6·‖F‖. The trailing 2×2 SVD keeps that when F is nearly
/// rank 1 (σ₂ ≲ √ε·σ₁): the two small singular values are then below FᵀF's
/// rounding, so the Jacobi stage leaves their plane unresolved, but B holds it to
/// ε·‖F‖.
pub fn svd3(f: Mat3) -> Svd3 {
    let (f0, f1, f2) = (f.x_axis, f.y_axis, f.z_axis);
    let (mut s00, mut s11, mut s22) = (f0.dot(f0), f1.dot(f1), f2.dot(f2));
    let (mut s01, mut s02, mut s12) = (f0.dot(f1), f0.dot(f2), f1.dot(f2));
    let mut q = Quat::IDENTITY;
    for _ in 0..SVD_SWEEPS {
        // (0,1) turns about +z, (0,2) about −y, (1,2) about +x.
        let (ch, sh) = jacobi_pair(&mut s00, &mut s11, &mut s01, &mut s02, &mut s12);
        q = quat_mul(q, Quat::from_xyzw(0.0, 0.0, sh, ch));
        let (ch, sh) = jacobi_pair(&mut s00, &mut s22, &mut s02, &mut s01, &mut s12);
        q = quat_mul(q, Quat::from_xyzw(0.0, -sh, 0.0, ch));
        let (ch, sh) = jacobi_pair(&mut s11, &mut s22, &mut s12, &mut s01, &mut s02);
        q = quat_mul(q, Quat::from_xyzw(sh, 0.0, 0.0, ch));
    }
    let (mut v0, mut v1, mut v2) = quat_to_cols(normalize4(q));

    let mut b0 = f0 * v0.x + f1 * v0.y + f2 * v0.z;
    let mut b1 = f0 * v1.x + f1 * v1.y + f2 * v1.z;
    let mut b2 = f0 * v2.x + f1 * v2.y + f2 * v2.z;
    let (mut r0, mut r1, mut r2) = (b0.dot(b0), b1.dot(b1), b2.dot(b2));
    if r0 < r1 {
        swap_cols(&mut b0, &mut b1);
        swap_cols(&mut v0, &mut v1);
        std::mem::swap(&mut r0, &mut r1);
    }
    if r0 < r2 {
        swap_cols(&mut b0, &mut b2);
        swap_cols(&mut v0, &mut v2);
        std::mem::swap(&mut r0, &mut r2);
    }
    if r1 < r2 {
        swap_cols(&mut b1, &mut b2);
        swap_cols(&mut v1, &mut v2);
    }

    let (mut u0, mut u1, mut u2) = (Vec3::X, Vec3::Y, Vec3::Z);
    // Rows (0,1), then (0,2), zero column 0 below the diagonal; rows (1,2) zero b1.z.
    let (c, s) = givens(b0.x, b0.y);
    rotate_rows!(c, s, [b0, b1, b2], x, y);
    rotate_cols(c, s, &mut u0, &mut u1);
    let (c, s) = givens(b0.x, b0.z);
    rotate_rows!(c, s, [b0, b1, b2], x, z);
    rotate_cols(c, s, &mut u0, &mut u2);
    let (c, s) = givens(b1.y, b1.z);
    rotate_rows!(c, s, [b1, b2], y, z);
    rotate_cols(c, s, &mut u1, &mut u2);

    // 2×2 SVD of the trailing block [[b1.y, b2.y], [b1.z, b2.z]]: a left rotation
    // makes it symmetric, then a two-sided Jacobi rotation diagonalises it.
    let (c, s) = givens(b1.y + b2.z, b1.z - b2.y);
    rotate_rows!(c, s, [b1, b2], y, z);
    rotate_cols(c, s, &mut u1, &mut u2);
    let (c, s) = sym2_jacobi(b1.y, 0.5 * (b2.y + b1.z), b2.z);
    rotate_rows!(c, s, [b1, b2], y, z);
    rotate_cols(c, s, &mut b1, &mut b2);
    rotate_cols(c, s, &mut u1, &mut u2);
    rotate_cols(c, s, &mut v1, &mut v2);
    // Restore the sign rule: |σ₂| ≥ |σ₃| and σ₂ ≥ 0, so σ₃ carries det F's sign.
    if b1.y.abs() < b2.z.abs() {
        swap_cols(&mut u1, &mut u2);
        swap_cols(&mut v1, &mut v2);
        std::mem::swap(&mut b1.y, &mut b2.z);
    }
    if b1.y < 0.0 {
        b1.y = -b1.y;
        b2.z = -b2.z;
        u1 = -u1;
        u2 = -u2;
    }

    Svd3 {
        u: Mat3::from_cols(u0, u1, u2),
        sigma: Vec3::new(b0.x, b1.y, b2.z),
        v: Mat3::from_cols(v0, v1, v2),
    }
}

/// Hamilton product in the Slang code's operation order (glam's `Mul` may fuse or
/// reorder differently).
fn quat_mul(a: Quat, b: Quat) -> Quat {
    let (av, bv) = (Vec3::new(a.x, a.y, a.z), Vec3::new(b.x, b.y, b.z));
    let v = bv * a.w + av * b.w + av.cross(bv);
    Quat::from_xyzw(v.x, v.y, v.z, a.w * b.w - av.dot(bv))
}

fn normalize4(q: Quat) -> Quat {
    let n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
    Quat::from_xyzw(q.x / n, q.y / n, q.z / n, q.w / n)
}

fn quat_to_cols(q: Quat) -> (Vec3, Vec3, Vec3) {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    (
        Vec3::new(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        Vec3::new(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        Vec3::new(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )
}

/// The Drucker–Prager α for friction angle φ (radians): √(2/3)·2 sin φ / (3 − sin φ).
pub fn friction_alpha(phi: f32) -> f32 {
    let s = phi.sin();
    (2.0f32 / 3.0).sqrt() * 2.0 * s / (3.0 - s)
}

/// The Drucker–Prager return map in Hencky strain (Klár et al. 2016) with the
/// log-volume correction (Tampubolon et al. 2017, β = 1). With
/// ε = ln max(|σ|, [`SIGMA_FLOOR`]) − cohesion + log Jp / 3 (compression first
/// consumes the volume gained when the particle separated) and tr = Σε:
///
/// - tr ≥ 0 (expansion): the tip, σ = e^cohesion, and log Jp keeps tr;
/// - otherwise log Jp is cleared, ε̂ = ε − tr/3 (traceless) and
///   δγ = ‖ε̂‖ + (3λ + 2μ)/(2μ)·tr·α; when δγ > 0, ε moves by δγ along ε̂.
///
/// Mirrors `mpmSandProject`.
pub fn sand_project(
    sigma: Vec3,
    log_jp: f32,
    alpha: f32,
    cohesion: f32,
    lambda: f32,
    mu: f32,
) -> SandProjection {
    let floor = Vec3::splat(SIGMA_FLOOR);
    let mut eps = Vec3::new(
        sigma.x.abs().max(floor.x).ln(),
        sigma.y.abs().max(floor.y).ln(),
        sigma.z.abs().max(floor.z).ln(),
    ) - cohesion
        + log_jp / 3.0;
    let tr = eps.x + eps.y + eps.z;
    if tr >= 0.0 {
        return SandProjection {
            sigma: Vec3::splat(cohesion.exp()),
            log_jp: tr,
        };
    }
    let eps_hat = eps - tr / 3.0;
    let eps_hat_norm = eps_hat.length();
    let d_gamma = eps_hat_norm + (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * tr * alpha;
    // ‖ε̂‖ = 0 implies δγ ≤ 0 when α ≥ 0; the norm test keeps the division safe
    // for any α.
    if d_gamma > 0.0 && eps_hat_norm > 0.0 {
        eps -= (d_gamma / eps_hat_norm) * eps_hat;
    }
    let e = eps + cohesion;
    SandProjection {
        sigma: Vec3::new(e.x.exp(), e.y.exp(), e.z.exp()),
        log_jp: 0.0,
    }
}

/// Mohr–Coulomb return map in Hencky strain with the same volume correction,
/// tip and cohesion shift as [`sand_project`]; non-dilatant flow (ψ = 0).
/// Principal Kirchhoff stresses τᵢ = 2μεᵢ + λ tr ε (compression negative,
/// ε₁ ≥ ε₂ ≥ ε₃ as [`svd3`] orders σ) must satisfy
/// f = (τ₁ − τ₃) + (τ₁ + τ₃)·sin φ ≤ 0. The return is closed form onto the main
/// plane, and a 2×2 solve onto an edge (τ₁ = τ₂ or τ₂ = τ₃) when the main-plane
/// return would reorder the strains (its multipliers clamped at 0 only against
/// rounding, for a state already on the edge).
/// Mirrors `mpmSandProjectMohrCoulomb`. The engine runs it for dry sand only
/// (`cohesion` 0); cohesive soil takes [`soil_project_mohr_coulomb`].
pub fn sand_project_mohr_coulomb(
    sigma: Vec3,
    log_jp: f32,
    sin_phi: f32,
    cohesion: f32,
    lambda: f32,
    mu: f32,
) -> SandProjection {
    let floor = Vec3::splat(SIGMA_FLOOR);
    let eps = Vec3::new(
        sigma.x.abs().max(floor.x).ln(),
        sigma.y.abs().max(floor.y).ln(),
        sigma.z.abs().max(floor.z).ln(),
    ) - cohesion
        + log_jp / 3.0;
    let tr = eps.x + eps.y + eps.z;
    if tr >= 0.0 {
        return SandProjection {
            sigma: Vec3::splat(cohesion.exp()),
            log_jp: tr,
        };
    }
    let tau = 2.0 * mu * eps + Vec3::splat(lambda * tr);
    let g_a = Vec3::new(1.0 + sin_phi, 0.0, -(1.0 - sin_phi));
    let f_a = g_a.dot(tau);
    let done = |e: Vec3| {
        let e = e + cohesion;
        SandProjection {
            sigma: Vec3::new(e.x.exp(), e.y.exp(), e.z.exp()),
            log_jp: 0.0,
        }
    };
    if f_a <= 0.0 {
        return done(eps);
    }
    let n_a = Vec3::new(1.0, 0.0, -1.0);
    let mut e = eps - (f_a / (4.0 * mu)) * n_a;
    if e.x < e.y || e.y < e.z {
        let right = e.y > e.x;
        let (n_b, g_b) = if right {
            (
                Vec3::new(0.0, 1.0, -1.0),
                Vec3::new(0.0, 1.0 + sin_phi, -(1.0 - sin_phi)),
            )
        } else {
            (
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0 + sin_phi, -(1.0 - sin_phi), 0.0),
            )
        };
        let f_b = g_b.dot(tau);
        let (a11, a12, a21, a22) = (2.0, g_a.dot(n_b), g_b.dot(n_a), 2.0);
        let det = (a11 * a22 - a12 * a21) * 2.0 * mu;
        let da = ((a22 * f_a - a12 * f_b) / det).max(0.0);
        let db = ((a11 * f_b - a21 * f_a) / det).max(0.0);
        e = eps - da * n_a - db * n_b;
    }
    done(e)
}

/// Mohr–Coulomb return map for cohesive soil, cohesion c in stress space, with
/// the volume correction of [`sand_project`] (ε = ln max(|σ|, [`SIGMA_FLOOR`]) +
/// log Jp / 3, tr = Σε) and non-dilatant flow (ψ = 0). `cohesion_cos_phi` is
/// c·cos φ (Pa); `sin_phi` must be positive when it is (validated by
/// [`SandMaterial::validate`]).
///
/// - Tension cutoff at the apex, the hydrostatic tension c·cot φ, whose
///   volumetric strain is tr_a = 3·c·cot φ / (3λ + 2μ): when tr ≥ tr_a no
///   deviatoric return reaches the surface, so σ = exp(tr_a/3) and log Jp keeps
///   the excess tr − tr_a (the volume the material gains as it separates).
/// - Otherwise log Jp is cleared and the principal Kirchhoff stresses
///   τᵢ = 2μεᵢ + λ tr (ε₁ ≥ ε₂ ≥ ε₃) must satisfy
///   f = (τ₁ − τ₃) + (τ₁ + τ₃)·sin φ − 2c·cos φ ≤ 0, returned as in
///   [`sand_project_mohr_coulomb`]: closed form onto the main plane, a 2×2
///   solve onto an edge. Below the apex an edge return always exists.
///
/// It is the cohesionless map translated by the apex (f(τ) is the cohesionless
/// f of τ − c·cot φ, and the flow is deviatoric), so c = 0 gives
/// [`sand_project_mohr_coulomb`]'s result. Mirrors `mpmSoilProjectMohrCoulomb`.
pub fn soil_project_mohr_coulomb(
    sigma: Vec3,
    log_jp: f32,
    sin_phi: f32,
    cohesion_cos_phi: f32,
    lambda: f32,
    mu: f32,
) -> SandProjection {
    let floor = Vec3::splat(SIGMA_FLOOR);
    let eps = Vec3::new(
        sigma.x.abs().max(floor.x).ln(),
        sigma.y.abs().max(floor.y).ln(),
        sigma.z.abs().max(floor.z).ln(),
    ) + log_jp / 3.0;
    let tr = eps.x + eps.y + eps.z;
    let tr_apex = 3.0 * cohesion_cos_phi / (sin_phi * (3.0 * lambda + 2.0 * mu));
    if tr >= tr_apex {
        return SandProjection {
            sigma: Vec3::splat((tr_apex / 3.0).exp()),
            log_jp: tr - tr_apex,
        };
    }
    let tau = 2.0 * mu * eps + Vec3::splat(lambda * tr);
    let k = 2.0 * cohesion_cos_phi;
    let g_a = Vec3::new(1.0 + sin_phi, 0.0, -(1.0 - sin_phi));
    let f_a = g_a.dot(tau) - k;
    let done = |e: Vec3| SandProjection {
        sigma: Vec3::new(e.x.exp(), e.y.exp(), e.z.exp()),
        log_jp: 0.0,
    };
    if f_a <= 0.0 {
        return done(eps);
    }
    let n_a = Vec3::new(1.0, 0.0, -1.0);
    let mut e = eps - (f_a / (4.0 * mu)) * n_a;
    if e.x < e.y || e.y < e.z {
        let right = e.y > e.x;
        let (n_b, g_b) = if right {
            (
                Vec3::new(0.0, 1.0, -1.0),
                Vec3::new(0.0, 1.0 + sin_phi, -(1.0 - sin_phi)),
            )
        } else {
            (
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0 + sin_phi, -(1.0 - sin_phi), 0.0),
            )
        };
        let f_b = g_b.dot(tau) - k;
        let (a11, a12, a21, a22) = (2.0, g_a.dot(n_b), g_b.dot(n_a), 2.0);
        let det = (a11 * a22 - a12 * a21) * 2.0 * mu;
        let da = ((a22 * f_a - a12 * f_b) / det).max(0.0);
        let db = ((a11 * f_b - a21 * f_a) / det).max(0.0);
        e = eps - da * n_a - db * n_b;
    }
    done(e)
}

/// The result of [`rock_damage`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RockState {
    /// Principal Kirchhoff stresses (Pa), in the SVD's order.
    pub tau: Vec3,
    /// Damage after the update, in [0, 1].
    pub damage: f32,
}

/// Rankine damage with crack-band exponential softening ([`RockMaterial`]).
/// With ε = ln max(|σ|, [`SIGMA_FLOOR`]) (ε₁ ≥ ε₂ ≥ ε₃) the undamaged principal
/// Kirchhoff stresses are τ̄ᵢ = 2μεᵢ + λ tr ε. When τ̄₁ passes the tensile
/// strength f_t, the damage it implies is
/// d(τ̄₁) = 1 − (f_t/τ̄₁)·exp(−(τ̄₁ − f_t)/σ_s), with the softening stress
/// σ_s = `band_stress`/dx − f_t/2 (`band_stress` = E·G_f/(f_t·b), b =
/// [`CRACK_BAND_CELLS`], so σ_s = E·G_f/(f_t·h) − f_t/2 over the band h = b·dx); d never
/// decreases (d = max(d, d(τ̄₁)), equivalent to keeping the largest τ̄₁ reached,
/// as d is increasing in it). Under uniaxial tension the stress is then
/// (1 − d)·τ̄₁ = f_t·exp(−(τ̄₁ − f_t)/σ_s), and the energy dissipated per unit
/// volume once broken is f_t²/2E + f_t·σ_s/E = G_f/h. Positive τ̄ᵢ are scaled
/// by (1 − d); compressive ones are kept (a closed crack carries compression).
/// dx must give σ_s > 0 ([`RockMaterial::softening_stress`]). Mirrors
/// `mpmRockDamage`.
pub fn rock_damage(
    sigma: Vec3,
    damage: f32,
    lambda: f32,
    mu: f32,
    tensile_strength: f32,
    band_stress: f32,
    dx: f32,
) -> RockState {
    let floor = Vec3::splat(SIGMA_FLOOR);
    let e = Vec3::new(
        sigma.x.abs().max(floor.x).ln(),
        sigma.y.abs().max(floor.y).ln(),
        sigma.z.abs().max(floor.z).ln(),
    );
    let tr = e.x + e.y + e.z;
    let te = 2.0 * mu * e + Vec3::splat(lambda * tr);
    let mut d = damage;
    if te.x > tensile_strength {
        let soft = band_stress / dx - 0.5 * tensile_strength;
        let fresh = 1.0 - tensile_strength / te.x * (-(te.x - tensile_strength) / soft).exp();
        d = d.max(fresh);
    }
    let keep = 1.0 - d;
    let degrade = |t: f32| if t > 0.0 { keep * t } else { t };
    RockState {
        tau: Vec3::new(degrade(te.x), degrade(te.y), degrade(te.z)),
        damage: d,
    }
}

/// The Kirchhoff stress U·diag(tau)·Uᵀ of principal stresses `tau` in the
/// columns of U. Mirrors `mpmPrincipalKirchhoff`.
pub fn principal_kirchhoff(u: Mat3, tau: Vec3) -> Mat3 {
    let (u0, u1, u2) = (u.x_axis, u.y_axis, u.z_axis);
    let col = |j: usize| u0 * (tau.x * u0[j]) + u1 * (tau.y * u1[j]) + u2 * (tau.z * u2[j]);
    Mat3::from_cols(col(0), col(1), col(2))
}

/// Kirchhoff stress of Hencky elasticity, τ = U·diag(2μ ln σ + λ Σ ln σ)·Uᵀ, for
/// σ > 0 (after [`sand_project`]). Mirrors `mpmHenckyKirchhoff`.
pub fn hencky_kirchhoff(u: Mat3, sigma: Vec3, lambda: f32, mu: f32) -> Mat3 {
    debug_assert!(sigma.min_element() > 0.0, "Hencky stress needs sigma > 0");
    let e = Vec3::new(sigma.x.ln(), sigma.y.ln(), sigma.z.ln());
    let d = 2.0 * mu * e + lambda * (e.x + e.y + e.z);
    let (u0, u1, u2) = (u.x_axis, u.y_axis, u.z_axis);
    let col = |j: usize| u0 * (d.x * u0[j]) + u1 * (d.y * u1[j]) + u2 * (d.z * u2[j]);
    Mat3::from_cols(col(0), col(1), col(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// splitmix64: deterministic cases without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in [lo, hi).
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
        }

        fn rotation(&mut self) -> Mat3 {
            let q = Quat::from_xyzw(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            Mat3::from_quat(q.normalize())
        }

        fn matrix(&mut self, lo: f32, hi: f32) -> Mat3 {
            Mat3::from_cols_array(&std::array::from_fn(|_| self.range(lo, hi)))
        }
    }

    fn frob(m: Mat3) -> f32 {
        m.to_cols_array().iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// The SVD families of the invariant tests: generic, near-singular,
    /// rotations, reflections, repeated singular values, and scaled.
    fn svd_cases() -> Vec<Mat3> {
        let mut rng = Rng(7);
        let mut out = vec![Mat3::IDENTITY, Mat3::ZERO, -Mat3::IDENTITY];
        for _ in 0..2000 {
            out.push(rng.matrix(-2.0, 2.0));
        }
        for _ in 0..500 {
            // Rank 1 or 2 plus a small perturbation (nearly rank 1 with σ₂ between
            // 1e-7 and 0.1 has its own test).
            let (r1, r2) = (rng.rotation(), rng.rotation());
            let s3 = if rng.next() & 1 == 0 { 0.0 } else { 1e-6 };
            let s2 = if rng.next() & 3 == 0 {
                0.0
            } else {
                rng.range(0.1, 1.0)
            };
            let d = Mat3::from_diagonal(Vec3::new(rng.range(0.5, 2.0), s2, s3));
            out.push(r1 * d * r2.transpose() + rng.matrix(-1e-7, 1e-7));
        }
        for _ in 0..300 {
            let r = rng.rotation();
            out.push(r);
            // det < 0: a reflection, and a general matrix with one column negated.
            out.push(-r);
            let mut m = rng.matrix(-2.0, 2.0);
            if m.determinant() > 0.0 {
                m.x_axis = -m.x_axis;
            }
            out.push(m);
        }
        for _ in 0..300 {
            let (r1, r2) = (rng.rotation(), rng.rotation());
            let (a, b) = (rng.range(0.2, 2.0), rng.range(0.2, 2.0));
            let sign = if rng.next() & 1 == 0 { 1.0 } else { -1.0 };
            for d in [
                Vec3::splat(a),
                Vec3::new(a, a, b),
                Vec3::new(b, a, a),
                Vec3::new(a, a, sign * a),
            ] {
                out.push(r1 * Mat3::from_diagonal(d) * r2.transpose());
            }
        }
        for _ in 0..200 {
            out.push(rng.matrix(-2.0, 2.0) * 1e3);
            out.push(rng.matrix(-2.0, 2.0) * 1e-3);
            // MPM-like: near a rotation, slightly strained.
            out.push(rng.rotation() * (Mat3::IDENTITY + rng.matrix(-0.05, 0.05)));
            // Sand at rest: a rotation by microradians, singular values equal to
            // rounding (the trailing 2×2 block has p = r and an off-diagonal whose
            // square underflows).
            let axis = Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            );
            let angle = 10f32.powf(rng.range(-7.0, -4.0));
            out.push(
                Mat3::from_axis_angle(axis.normalize_or(Vec3::X), angle) * rng.range(0.999, 1.001),
            );
        }
        // A bed particle's F (rows, bit for bit) whose SVD once came out NaN on the
        // GPU: its trailing block had p = r exactly.
        let rows: [[u32; 3]; 3] = [
            [0x3f800000, 0x368594c2, 0x36d338d8],
            [0xb68594a0, 0x3f800001, 0xb7cef85d],
            [0xb6d2e165, 0x37cef067, 0x3f7fffff],
        ];
        out.push(Mat3::from_cols_array_2d(&rows.map(|r| r.map(f32::from_bits))).transpose());
        out
    }

    #[test]
    fn the_symmetric_2x2_rotation_is_finite_for_repeated_values() {
        for q in [1e-23f32, -1e-23, 1e-30, 1e-38, 3e-45, 1e-3, 1e18] {
            for p in [1.0f32, 1.0000001, 0.0, 1e-20] {
                let (c, s) = sym2_jacobi(p, q, p);
                assert!(
                    c.is_finite() && s.is_finite(),
                    "p = r = {p}, q = {q}: ({c}, {s})"
                );
                assert!((c * c + s * s - 1.0).abs() < 1e-6, "({c}, {s})");
                // θ = ±π/4 diagonalises [[p, q], [q, p]].
                assert!((c - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
                assert_eq!(s > 0.0, q > 0.0);
            }
        }
        let (c, s) = givens(1e-30, 1e-31);
        assert!((c * c + s * s - 1.0).abs() < 1e-6 && c > 0.99, "({c}, {s})");
    }

    #[test]
    fn svd_reconstructs_with_rotations_and_ordered_sigma() {
        let (mut worst_rec, mut worst_orth, mut worst_det) = (0.0f32, 0.0f32, 0.0f32);
        for f in svd_cases() {
            assert!(svd_input_valid(f));
            let Svd3 { u, sigma, v } = svd3(f);
            let rec = u * Mat3::from_diagonal(sigma) * v.transpose();
            let scale = frob(f).max(f32::MIN_POSITIVE);
            let rel = if frob(f) == 0.0 {
                frob(rec)
            } else {
                frob(rec - f) / scale
            };
            worst_rec = worst_rec.max(rel);
            for m in [u, v] {
                worst_orth = worst_orth.max(frob(m.transpose() * m - Mat3::IDENTITY));
                worst_det = worst_det.max((m.determinant() - 1.0).abs());
            }
            assert!(rel < 4e-6, "‖UΣVᵀ − F‖/‖F‖ = {rel} for {f:?}");
            // Ordering and sign: |σ| decreasing (to rounding), σ.x, σ.y ≥ 0, and
            // σ.z carries det F's sign.
            let tol = 2e-6 * sigma.x.abs().max(1e-30) + 1e-30;
            assert!(sigma.x >= 0.0 && sigma.y >= 0.0, "{sigma:?}");
            assert!(sigma.x + tol >= sigma.y, "{sigma:?}");
            assert!(sigma.y + tol >= sigma.z.abs(), "{sigma:?}");
            let det = f.determinant();
            if det.abs() > 1e-4 * scale.powi(3) {
                assert_eq!(sigma.z < 0.0, det < 0.0, "{sigma:?} det {det}");
            }
        }
        assert!(worst_orth < 2e-6, "orthonormality error {worst_orth}");
        assert!(worst_det < 2e-6, "det error {worst_det}");
        eprintln!(
            "svd3 ({SVD_SWEEPS} sweeps): max relative reconstruction {worst_rec:.2e}, \
             max ‖MᵀM − I‖ {worst_orth:.2e}, max |det − 1| {worst_det:.2e}"
        );
    }

    #[test]
    fn nearly_rank_one_svd_is_resolved() {
        // σ₂, σ₃ ≤ √ε·σ₁: their eigenvalues in FᵀF are below its rounding; the
        // trailing 2×2 SVD must still resolve their plane to f32 accuracy.
        let mut rng = Rng(19);
        let mut worst = 0.0f32;
        for _ in 0..20_000 {
            let s1 = 10f32.powf(rng.range(-7.0, -1.0));
            let d = Mat3::from_diagonal(Vec3::new(1.0, s1, s1 * rng.range(0.0, 1.0)));
            let f = rng.rotation() * d * rng.rotation().transpose();
            let Svd3 { u, sigma, v } = svd3(f);
            let rel = frob(u * Mat3::from_diagonal(sigma) * v.transpose() - f) / frob(f);
            worst = worst.max(rel);
        }
        assert!(worst < 4e-6, "{worst}");
        eprintln!("svd3 nearly rank 1: max relative reconstruction {worst:.2e}");
    }

    #[test]
    fn svd_of_a_diagonal_is_exact_up_to_order() {
        let Svd3 { sigma, .. } = svd3(Mat3::from_diagonal(Vec3::new(0.5, 3.0, -2.0)));
        assert!(
            (sigma - Vec3::new(3.0, 2.0, -0.5)).abs().max_element() < 1e-6,
            "{sigma:?}"
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        for bad in [f32::NAN, f32::INFINITY, -f32::INFINITY, 1e19] {
            let mut f = Mat3::IDENTITY;
            f.y_axis.z = bad;
            assert!(!svd_input_valid(f));
            assert!(!sand_input_valid(Vec3::new(1.0, bad, 1.0), 0.0));
            assert!(!sand_input_valid(Vec3::ONE, bad));
        }
        assert!(sand_input_valid(Vec3::new(1.0, 0.5, -0.2), 0.3));
    }

    fn sand() -> (f32, f32, f32) {
        let m = SandMaterial::default();
        (m.alpha(), m.lambda(), m.mu())
    }

    /// Yield function of an elastic strain with no stored volume:
    /// ‖dev ε‖ + α·(3λ + 2μ)/(2μ)·tr ε.
    fn yield_fn(eps: Vec3, alpha: f32, lambda: f32, mu: f32) -> f32 {
        let tr = eps.x + eps.y + eps.z;
        (eps - tr / 3.0).length() + alpha * (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * tr
    }

    fn strain(sigma: Vec3, cohesion: f32) -> Vec3 {
        Vec3::new(sigma.x.ln(), sigma.y.ln(), sigma.z.ln()) - cohesion
    }

    #[test]
    fn strain_inside_the_cone_is_unchanged() {
        let (alpha, lambda, mu) = sand();
        let mut rng = Rng(11);
        let mut n = 0;
        while n < 500 {
            let sigma = Vec3::new(
                rng.range(0.9, 1.0),
                rng.range(0.9, 1.0),
                rng.range(0.9, 1.0),
            );
            if yield_fn(strain(sigma, 0.0), alpha, lambda, mu) >= 0.0 {
                continue;
            }
            n += 1;
            let p = sand_project(sigma, 0.0, alpha, 0.0, lambda, mu);
            assert!(
                (p.sigma - sigma).abs().max_element() <= 1e-6,
                "{sigma:?} → {p:?}"
            );
            assert_eq!(p.log_jp, 0.0);
        }
    }

    #[test]
    fn projection_is_admissible_and_idempotent() {
        let mut rng = Rng(13);
        for i in 0..20_000 {
            let mat = SandMaterial {
                friction_angle: rng.range(0.0, 45.0).to_radians(),
                ..SandMaterial::default()
            };
            // The cone's tip offset (log strain; SandMaterial::apex_strain).
            let apex = if i % 3 == 0 {
                rng.range(0.0, 0.05)
            } else {
                0.0
            };
            let (alpha, lambda, mu) = (mat.alpha(), mat.lambda(), mat.mu());
            let mut sigma = Vec3::new(
                rng.range(0.5, 1.5),
                rng.range(0.5, 1.5),
                rng.range(0.5, 1.5),
            );
            if i % 7 == 0 {
                sigma.z = -sigma.z;
            }
            let log_jp = if i % 2 == 0 { 0.0 } else { rng.range(0.0, 0.2) };
            let p = sand_project(sigma, log_jp, alpha, apex, lambda, mu);
            assert!(p.sigma.min_element() > 0.0 && p.log_jp >= 0.0, "{p:?}");
            // On or inside the cone, and never in tension.
            let eps = strain(p.sigma, apex);
            let f = yield_fn(eps, alpha, lambda, mu);
            assert!(f <= 1e-5, "yield {f} after projecting {sigma:?}, {log_jp}");
            assert!(eps.x + eps.y + eps.z <= 1e-5, "tension {eps:?}");
            let again = sand_project(p.sigma, p.log_jp, alpha, apex, lambda, mu);
            assert!(
                (again.sigma - p.sigma).abs().max_element() <= 2e-5
                    && (again.log_jp - p.log_jp).abs() <= 2e-5,
                "not idempotent: {p:?} → {again:?}"
            );
        }
    }

    #[test]
    fn expansion_goes_to_the_tip_and_keeps_the_volume() {
        let (alpha, lambda, mu) = sand();
        let sigma = Vec3::new(1.1, 1.05, 0.98);
        let p = sand_project(sigma, 0.02, alpha, 0.0, lambda, mu);
        assert_eq!(p.sigma, Vec3::ONE);
        let volume = sigma.x.ln() + sigma.y.ln() + sigma.z.ln() + 0.02;
        assert!((p.log_jp - volume).abs() < 1e-6, "{} vs {volume}", p.log_jp);
        // With cohesion the tip sits at the cohesive stretch.
        let c = 0.01;
        let p = sand_project(sigma, 0.0, alpha, c, lambda, mu);
        assert!((p.sigma - Vec3::splat(c.exp())).abs().max_element() < 1e-7);
        assert!((p.log_jp - (volume - 0.02 - 3.0 * c)).abs() < 1e-6);
    }

    #[test]
    fn compression_first_consumes_stored_volume() {
        let (alpha, lambda, mu) = sand();
        let stored = 0.06;
        // Compressed by less than the stored volume: still at the tip, with the
        // rest of the volume kept.
        let half = Vec3::splat((-stored / 6.0f32).exp());
        let p = sand_project(half, stored, alpha, 0.0, lambda, mu);
        assert_eq!(p.sigma, Vec3::ONE);
        assert!((p.log_jp - stored / 2.0).abs() < 1e-6, "{p:?}");
        // Compressed by more: tr shifts by the stored volume, so the elastic
        // compression is only the excess (isotropic, hence inside the cone).
        let sigma = Vec3::splat((-2.0 * stored / 3.0f32).exp());
        let p = sand_project(sigma, stored, alpha, 0.0, lambda, mu);
        assert_eq!(p.log_jp, 0.0);
        let tr = p.sigma.x.ln() + p.sigma.y.ln() + p.sigma.z.ln();
        assert!((tr - (-2.0 * stored + stored)).abs() < 1e-6, "{tr}");
        // A deviatoric part outside the cone is projected from the shifted strain.
        let sigma = Vec3::new(0.8, 1.0, 1.1);
        let p = sand_project(sigma, stored, alpha, 0.0, lambda, mu);
        assert_eq!(p.log_jp, 0.0);
        let eps = strain(p.sigma, 0.0);
        assert!((eps - strain(sigma, 0.0)).length() > 0.1, "not projected");
        let want = sigma.x.ln() + sigma.y.ln() + sigma.z.ln() + stored;
        assert!((eps.x + eps.y + eps.z - want).abs() < 1e-6);
        assert!(yield_fn(eps, alpha, lambda, mu).abs() < 1e-5);
    }

    #[test]
    fn isotropic_compression_stays_elastic() {
        let (alpha, lambda, mu) = sand();
        for s in [0.999, 0.95, 0.7, 0.3] {
            let sigma = Vec3::splat(s);
            let p = sand_project(sigma, 0.0, alpha, 0.0, lambda, mu);
            assert!((p.sigma - sigma).abs().max_element() <= 1e-6, "{s}: {p:?}");
            assert_eq!(p.log_jp, 0.0);
        }
        // Frictionless (α = 0): pure compression is still on the cone's axis.
        let p = sand_project(Vec3::splat(0.9), 0.0, 0.0, 0.0, lambda, mu);
        assert!((p.sigma - Vec3::splat(0.9)).abs().max_element() <= 1e-6);
    }

    #[test]
    fn identity_has_zero_stress() {
        let (alpha, lambda, mu) = sand();
        let Svd3 { u, sigma, .. } = svd3(Mat3::IDENTITY);
        let p = sand_project(sigma, 0.0, alpha, 0.0, lambda, mu);
        let tau = hencky_kirchhoff(u, p.sigma, lambda, mu);
        assert_eq!((u, sigma), (Mat3::IDENTITY, Vec3::ONE));
        assert_eq!(p.log_jp, 0.0);
        assert_eq!(tau, Mat3::ZERO);
    }

    #[test]
    fn kirchhoff_is_symmetric_and_matches_a_pure_stretch() {
        let (_, lambda, mu) = sand();
        let mut rng = Rng(17);
        for _ in 0..1000 {
            let u = rng.rotation();
            let sigma = Vec3::new(
                rng.range(0.8, 1.2),
                rng.range(0.8, 1.2),
                rng.range(0.8, 1.2),
            );
            let tau = hencky_kirchhoff(u, sigma, lambda, mu);
            assert!(frob(tau - tau.transpose()) <= 1e-6 * frob(tau), "{tau:?}");
        }
        // U = I: τ is diagonal, 2μ ln σᵢ + λ Σ ln σ.
        let sigma = Vec3::new(1.1, 0.9, 0.95);
        let e = Vec3::new(sigma.x.ln(), sigma.y.ln(), sigma.z.ln());
        let want = 2.0 * mu * e + lambda * (e.x + e.y + e.z);
        let tau = hencky_kirchhoff(Mat3::IDENTITY, sigma, lambda, mu);
        let diag = Vec3::new(tau.x_axis.x, tau.y_axis.y, tau.z_axis.z);
        assert!((diag - want).abs().max_element() <= 1e-6 * want.abs().max_element());
        assert_eq!(tau - Mat3::from_diagonal(diag), Mat3::ZERO);
    }

    /// Mohr–Coulomb yield function of an elastic strain (compression negative).
    fn mc_yield(eps: Vec3, sin_phi: f32, lambda: f32, mu: f32) -> f32 {
        let tau = 2.0 * mu * eps + Vec3::splat(lambda * (eps.x + eps.y + eps.z));
        (tau.x - tau.z) + (tau.x + tau.z) * sin_phi
    }

    /// Sorted (decreasing) singular values near 1 (compressive log strains ~1e-2).
    fn sorted_sigma(rng: &mut Rng, spread: f32) -> Vec3 {
        let mut v = [
            rng.range(1.0 - spread, 1.0),
            rng.range(1.0 - spread, 1.0),
            rng.range(1.0 - spread, 1.0),
        ];
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Vec3::from_array(v)
    }

    #[test]
    fn mohr_coulomb_returns_onto_the_surface_keeping_trace_and_order() {
        let (_, lambda, mu) = sand();
        let mut rng = Rng(23);
        let (mut yielded, mut edges) = (0, 0);
        for _ in 0..20_000 {
            let phi = rng.range(5.0, 45.0).to_radians();
            let s = phi.sin();
            let sigma = sorted_sigma(&mut rng, 0.02);
            let e0 = strain(sigma, 0.0);
            let tr0 = e0.x + e0.y + e0.z;
            let p = sand_project_mohr_coulomb(sigma, 0.0, s, 0.0, lambda, mu);
            if tr0 >= 0.0 {
                continue;
            }
            let e = strain(p.sigma, 0.0);
            let f = mc_yield(e, s, lambda, mu);
            let scale = mu * e0.abs().max_element();
            assert!(
                f <= 2e-4 * scale,
                "f = {f} (scale {scale}) for {sigma:?} → {:?}",
                p.sigma
            );
            assert!(e.x + 2e-6 >= e.y && e.y + 2e-6 >= e.z, "order lost: {e:?}");
            let tr = e.x + e.y + e.z;
            assert!(
                (tr - tr0).abs() <= 2e-6 + 1e-4 * tr0.abs(),
                "trace {tr0} → {tr}"
            );
            if mc_yield(e0, s, lambda, mu) > 0.0 {
                yielded += 1;
                assert!(f >= -2e-4 * scale, "returned inside the surface: f = {f}");
                if (e.x - e.y).abs() < 1e-6 || (e.y - e.z).abs() < 1e-6 {
                    edges += 1;
                }
                let again = sand_project_mohr_coulomb(p.sigma, 0.0, s, 0.0, lambda, mu);
                assert!(
                    (again.sigma - p.sigma).abs().max_element() <= 2e-6,
                    "not idempotent: {sigma:?} → {:?} → {:?} (φ {:.1}°, f {f}, scale {scale})",
                    p.sigma,
                    again.sigma,
                    phi.to_degrees()
                );
            } else {
                assert!(
                    (p.sigma - sigma).abs().max_element() <= 1e-6,
                    "inside, yet moved"
                );
            }
        }
        assert!(
            yielded > 1000 && edges > 50,
            "cases: {yielded} yielded, {edges} on an edge"
        );
    }

    /// Klár's Drucker–Prager cone is fitted to Mohr–Coulomb on the compression
    /// meridian (τ₁ = τ₂ > τ₃): there the two return maps agree. On the
    /// extension meridian (τ₁ > τ₂ = τ₃) Mohr–Coulomb holds less shear.
    #[test]
    fn mohr_coulomb_matches_the_drucker_prager_fit_on_its_meridian_only() {
        let (_, lambda, mu) = sand();
        let mut rng = Rng(29);
        for _ in 0..2_000 {
            let phi = rng.range(10.0, 40.0).to_radians();
            let (s, alpha) = (phi.sin(), friction_alpha(phi));
            let a = rng.range(0.97, 0.995);
            let b = rng.range(0.90, 0.97);
            for (sigma, meridian) in [
                (Vec3::new(a, a, b), "compression"),
                (Vec3::new(a, b, b), "extension"),
            ] {
                let mc = sand_project_mohr_coulomb(sigma, 0.0, s, 0.0, lambda, mu);
                let dp = sand_project(sigma, 0.0, alpha, 0.0, lambda, mu);
                let dev = |p: &SandProjection| {
                    let e = strain(p.sigma, 0.0);
                    (e - (e.x + e.y + e.z) / 3.0).length()
                };
                if meridian == "compression" {
                    assert!(
                        (mc.sigma - dp.sigma).abs().max_element() <= 2e-6,
                        "φ {:.1}°: MC {:?} vs DP {:?}",
                        phi.to_degrees(),
                        mc.sigma,
                        dp.sigma
                    );
                } else if mc_yield(strain(sigma, 0.0), s, lambda, mu) > 0.0 {
                    assert!(dev(&mc) < dev(&dp), "extension: MC should hold less shear");
                }
            }
        }
    }

    #[test]
    fn mohr_coulomb_tip_and_stored_volume_match_the_cone() {
        let (_, lambda, mu) = sand();
        let s = 30f32.to_radians().sin();
        // Expansion: the tip, logJp keeps the volume.
        let p = sand_project_mohr_coulomb(Vec3::new(1.01, 1.0, 0.995), 0.0, s, 0.0, lambda, mu);
        assert_eq!(p.sigma, Vec3::ONE);
        assert!((p.log_jp - (1.01f32.ln() + 0.995f32.ln())).abs() < 1e-6);
        // Stored volume is consumed first (as the cone does).
        let sigma = Vec3::splat(0.99);
        let p = sand_project_mohr_coulomb(sigma, 0.01, s, 0.0, lambda, mu);
        let q = sand_project(sigma, 0.01, 0.3, 0.0, lambda, mu);
        assert_eq!(p, q, "isotropic compression with stored volume");
    }

    #[test]
    fn default_sand_is_valid_with_its_lame_parameters() {
        let m = SandMaterial::default();
        m.validate().unwrap();
        // E = 50 MPa, ν = 0.3: μ = E/2.6, λ = 0.3E/0.52.
        assert!((m.mu() / 19.230_77e6 - 1.0).abs() < 1e-6, "{}", m.mu());
        assert!(
            (m.lambda() / 28.846_15e6 - 1.0).abs() < 1e-6,
            "{}",
            m.lambda()
        );
        // φ = 30°: α = √(2/3)·2·0.5/2.5.
        assert!((m.alpha() - 0.326_598_6).abs() < 1e-6, "{}", m.alpha());
        // c = √((λ + 2μ)/ρ) = √(67.31 MPa / 1600 kg/m³) ≈ 205.1 m/s.
        assert!(
            (m.p_wave_speed() - 205.1).abs() < 0.1,
            "{}",
            m.p_wave_speed()
        );
    }

    #[test]
    fn nonphysical_sand_is_rejected() {
        let d = SandMaterial::default();
        let bad = [
            SandMaterial { density: 0.0, ..d },
            SandMaterial {
                density: f32::NAN,
                ..d
            },
            SandMaterial {
                youngs_modulus: -1.0,
                ..d
            },
            SandMaterial {
                youngs_modulus: f32::INFINITY,
                ..d
            },
            SandMaterial {
                poisson_ratio: 0.5,
                ..d
            },
            SandMaterial {
                poisson_ratio: -1.0,
                ..d
            },
            SandMaterial {
                poisson_ratio: f32::NAN,
                ..d
            },
            SandMaterial {
                friction_angle: -0.1,
                ..d
            },
            SandMaterial {
                friction_angle: std::f32::consts::FRAC_PI_2,
                ..d
            },
            SandMaterial {
                cohesion: -0.01,
                ..d
            },
            SandMaterial {
                cohesion: f32::NAN,
                ..d
            },
            SandMaterial {
                cohesion: f32::INFINITY,
                ..d
            },
            // Cohesive, frictionless: no apex to cut tension off at.
            SandMaterial {
                cohesion: 1.0e3,
                friction_angle: 0.0,
                ..d
            },
            // Apex stretch c·cot φ / (3λ + 2μ) past an e-fold (E = 50 MPa).
            SandMaterial {
                cohesion: 1.0e8,
                ..d
            },
        ];
        for m in bad {
            assert!(
                matches!(m.validate(), Err(TerrainError::InvalidConfig(_))),
                "{m:?} accepted"
            );
        }
    }

    /// The cohesive soil of the soil tests: dry-sand elasticity, friction angle
    /// `phi`, and the cohesion whose apex stretch is `apex` (log strain).
    fn soil(phi: f32, apex: f32) -> SandMaterial {
        let d = SandMaterial::default();
        SandMaterial {
            friction_angle: phi,
            cohesion: apex * (3.0 * d.lambda() + 2.0 * d.mu()) * phi.tan(),
            ..d
        }
    }

    /// Stress-space Mohr–Coulomb yield function of an elastic strain, with
    /// cohesion (compression negative).
    fn soil_yield(eps: Vec3, m: &SandMaterial) -> f32 {
        let (lambda, mu) = (m.lambda(), m.mu());
        let tau = 2.0 * mu * eps + Vec3::splat(lambda * (eps.x + eps.y + eps.z));
        let (s, c) = (m.friction_angle.sin(), m.friction_angle.cos());
        (tau.x - tau.z) + (tau.x + tau.z) * s - 2.0 * m.cohesion * c
    }

    /// Hencky strain of principal Kirchhoff stresses τ:
    /// ε = τ/2μ − λ·tr τ / (2μ(3λ + 2μ)).
    fn strain_of(tau: Vec3, lambda: f32, mu: f32) -> Vec3 {
        let tr = tau.x + tau.y + tau.z;
        tau / (2.0 * mu) - Vec3::splat(lambda * tr / (2.0 * mu * (3.0 * lambda + 2.0 * mu)))
    }

    fn exp3(e: Vec3) -> Vec3 {
        Vec3::new(e.x.exp(), e.y.exp(), e.z.exp())
    }

    #[test]
    fn cohesionless_soil_is_dry_sand_bit_for_bit() {
        let (_, lambda, mu) = sand();
        let mut rng = Rng(31);
        for i in 0..20_000 {
            let s = rng.range(0.0, 45.0).to_radians().sin();
            let sigma = sorted_sigma(&mut rng, 0.05) * if i % 3 == 0 { 1.02 } else { 1.0 };
            let log_jp = if i % 2 == 0 {
                0.0
            } else {
                rng.range(0.0, 0.05)
            };
            assert_eq!(
                soil_project_mohr_coulomb(sigma, log_jp, s, 0.0, lambda, mu),
                sand_project_mohr_coulomb(sigma, log_jp, s, 0.0, lambda, mu),
                "{sigma:?}, {log_jp}"
            );
        }
    }

    /// Cohesion in stress space translates the cohesionless surface along the
    /// hydrostatic axis by its apex c·cot φ, and the flow is deviatoric, so the
    /// return map equals the cohesionless one applied to the strain shifted by
    /// the apex stretch (the log-strain offset the cone takes).
    #[test]
    fn cohesive_soil_is_the_cohesionless_map_shifted_by_its_apex() {
        let (_, lambda, mu) = sand();
        let mut rng = Rng(37);
        let mut worst = 0.0f32;
        for i in 0..20_000 {
            let phi = rng.range(5.0, 45.0).to_radians();
            let m = soil(phi, rng.range(1e-4, 0.02));
            let apex = m.apex_strain();
            let sigma = sorted_sigma(&mut rng, 0.05) * (1.0 + rng.range(0.0, 2.0 * apex));
            let log_jp = if i % 2 == 0 {
                0.0
            } else {
                rng.range(0.0, 0.05)
            };
            let p = soil_project_mohr_coulomb(
                sigma,
                log_jp,
                phi.sin(),
                m.cohesion_cos_phi(),
                lambda,
                mu,
            );
            let q = sand_project_mohr_coulomb(sigma, log_jp, phi.sin(), apex, lambda, mu);
            let d = ((p.sigma - q.sigma).abs() / q.sigma)
                .max_element()
                .max((p.log_jp - q.log_jp).abs());
            worst = worst.max(d);
            assert!(d <= 2e-6, "{sigma:?}, {log_jp}: {p:?} vs {q:?}");
        }
        eprintln!("cohesive MC vs shifted cohesionless MC: max difference {worst:.2e}");
    }

    #[test]
    fn cohesive_soil_returns_onto_its_surface_with_deviatoric_normal_flow() {
        let (_, lambda, mu) = sand();
        let mut rng = Rng(41);
        let (mut yielded, mut edges, mut apexes) = (0, 0, 0);
        for i in 0..20_000 {
            let phi = rng.range(5.0, 45.0).to_radians();
            let m = soil(phi, rng.range(1e-3, 0.02));
            let (s, kc) = (phi.sin(), m.cohesion_cos_phi());
            let apex = m.apex_strain();
            // Strains from compression to past the apex.
            let sigma = sorted_sigma(&mut rng, 0.03) * (1.0 + rng.range(0.0, 2.0 * apex));
            let log_jp = if i % 4 == 0 {
                rng.range(0.0, 0.02)
            } else {
                0.0
            };
            let e0 = strain(sigma, 0.0) + log_jp / 3.0;
            let tr0 = e0.x + e0.y + e0.z;
            let p = soil_project_mohr_coulomb(sigma, log_jp, s, kc, lambda, mu);
            assert!(p.sigma.min_element() > 0.0 && p.log_jp >= 0.0, "{p:?}");
            let e = strain(p.sigma, 0.0);
            let scale = mu * e0.abs().max_element().max(apex);
            let f = soil_yield(e, &m);
            assert!(f <= 2e-4 * scale, "f = {f} (scale {scale}) after {sigma:?}");
            if p.log_jp > 0.0 || tr0 >= 3.0 * apex {
                // The apex: hydrostatic at c·cot φ, the excess volume kept.
                apexes += 1;
                assert!((e - Vec3::splat(apex)).abs().max_element() <= 1e-6, "{e:?}");
                assert!((p.log_jp - (tr0 - 3.0 * apex)).abs() <= 2e-6, "{p:?}");
                continue;
            }
            let tr = e.x + e.y + e.z;
            assert!(
                (tr - tr0).abs() <= 2e-6 + 1e-4 * tr0.abs(),
                "trace {tr0} → {tr}"
            );
            assert!(e.x + 2e-6 >= e.y && e.y + 2e-6 >= e.z, "order lost: {e:?}");
            if soil_yield(e0, &m) <= 0.0 {
                assert!((e - e0).abs().max_element() <= 1e-6, "inside, yet moved");
                continue;
            }
            yielded += 1;
            assert!(f >= -2e-4 * scale, "returned inside the surface: f = {f}");
            // Koiter's rule for the plastic potential τ₁ − τ₃ (ψ = 0): the
            // plastic strain is a non-negative combination of the active planes'
            // flow directions, (1, 0, −1) and, on an edge, (0, 1, −1) (τ₁ = τ₂)
            // or (1, −1, 0) (τ₂ = τ₃).
            let dp = e0 - e;
            let tol = 2e-6 + 1e-4 * dp.abs().max_element();
            assert!(dp.x >= -tol && dp.z <= tol, "flow {dp:?}");
            if dp.y > tol {
                edges += 1;
                assert!((e.x - e.y).abs() <= 2e-6, "flow {dp:?} off the edge {e:?}");
            } else if dp.y < -tol {
                edges += 1;
                assert!((e.y - e.z).abs() <= 2e-6, "flow {dp:?} off the edge {e:?}");
            }
            let again = soil_project_mohr_coulomb(p.sigma, 0.0, s, kc, lambda, mu);
            assert!(
                (again.sigma - p.sigma).abs().max_element() <= 2e-6,
                "not idempotent: {sigma:?} → {:?} → {:?}",
                p.sigma,
                again.sigma
            );
        }
        assert!(
            yielded > 1000 && edges > 50 && apexes > 50,
            "cases: {yielded} yielded, {edges} on an edge, {apexes} at the apex"
        );
    }

    /// Unconfined strengths of a Mohr–Coulomb soil: it holds uniaxial tension up
    /// to 2c·cos φ/(1 + sin φ) and uniaxial compression up to
    /// q_u = 2c·cos φ/(1 − sin φ) = 2c·tan(45° + φ/2) (so H_c = 2q_u/γ).
    #[test]
    fn cohesive_soil_has_its_unconfined_strengths() {
        let phi = 30f32.to_radians();
        let m = soil(phi, 0.005);
        let (lambda, mu) = (m.lambda(), m.mu());
        let (s, c) = (phi.sin(), m.cohesion);
        let kc = m.cohesion_cos_phi();
        let tension = 2.0 * c * phi.cos() / (1.0 + s);
        let qu = 2.0 * c * (std::f32::consts::FRAC_PI_4 + phi / 2.0).tan();
        assert!((qu - 2.0 * c * phi.cos() / (1.0 - s)).abs() <= 1e-6 * qu);
        for (tau, what) in [
            (Vec3::new(tension, 0.0, 0.0), "tension"),
            (Vec3::new(0.0, 0.0, -qu), "compression"),
        ] {
            for (k, inside) in [(0.99, true), (1.01, false)] {
                let sigma = exp3(strain_of(tau * k, lambda, mu));
                let p = soil_project_mohr_coulomb(sigma, 0.0, s, kc, lambda, mu);
                let moved = (p.sigma - sigma).abs().max_element() > 1e-7;
                assert_eq!(!moved, inside, "uniaxial {what} at {k}× strength");
            }
        }
    }

    /// Uniaxial stress along x (the lateral strain −ν·ε₁ keeps τ₂ = τ₃ = 0 at
    /// any damage): the stretches of the Hencky strain ε₁.
    fn uniaxial(m: &RockMaterial, e1: f64) -> Vec3 {
        let lateral = (-f64::from(m.poisson_ratio) * e1).exp() as f32;
        Vec3::new(e1.exp() as f32, lateral, lateral)
    }

    fn rock_step(m: &RockMaterial, sigma: Vec3, d: f32, dx: f32) -> RockState {
        rock_damage(
            sigma,
            d,
            m.lambda(),
            m.mu(),
            m.tensile_strength,
            m.band_stress(),
            dx,
        )
    }

    /// Uniaxial tension: elastic to the tensile strength, then exponential
    /// softening, and the energy dissipated per crack area (per volume times the
    /// cell size) is G_f at any cell size the crack band allows.
    #[test]
    fn rock_in_uniaxial_tension_peaks_at_its_strength_and_dissipates_its_fracture_energy() {
        let m = RockMaterial::default();
        let (ft, e) = (f64::from(m.tensile_strength), f64::from(m.youngs_modulus));
        for share in [0.1f32, 0.3, 0.6, 0.9] {
            let dx = share * m.max_cell_size();
            let ss = f64::from(m.softening_stress(dx).unwrap());
            let (mut d, mut peak, mut work) = (0.0f32, 0.0f64, 0.0f64);
            let (mut last_e, mut last_t) = (0.0f64, 0.0f64);
            let end = (ft + 30.0 * ss) / e;
            let steps = 200_000;
            let (lambda, mu) = (f64::from(m.lambda()), f64::from(m.mu()));
            let mut most = 0.0f64;
            for k in 1..=steps {
                let sigma = uniaxial(&m, end * k as f64 / steps as f64);
                let r = rock_step(&m, sigma, d, dx);
                assert!(r.damage >= d, "damage healed");
                d = r.damage;
                // The lateral stresses stay zero (to the stretch's rounding).
                assert!(
                    f64::from(r.tau.y.abs().max(r.tau.z.abs())) <= 2e-3 * ft,
                    "{r:?}"
                );
                // The law at the strain the stretches hold (f32 rounds them by
                // about 6e-8, a third of a percent of the peak strain).
                let el = sigma.as_dvec3();
                let el = glam::DVec3::new(el.x.ln(), el.y.ln(), el.z.ln());
                let e1 = el.x;
                let bar = 2.0 * mu * e1 + lambda * (el.x + el.y + el.z);
                let t = f64::from(r.tau.x);
                peak = peak.max(t);
                // The damage is that of the largest τ̄₁ reached (the rounding of
                // the stretches can step τ̄₁ back by a few kPa).
                most = most.max(bar);
                let want = if most <= ft {
                    assert_eq!(d, 0.0, "damage before the peak at ε = {e1}");
                    bar
                } else {
                    ft / most * (-(most - ft) / ss).exp() * bar
                };
                assert!(
                    (t - want).abs() <= 1e-4 * ft,
                    "dx {dx}: τ = {t} at ε = {e1}, the law gives {want}"
                );
                work += 0.5 * (t + last_t) * (e1 - last_e);
                (last_e, last_t) = (e1, t);
            }
            let per_area = work * f64::from(CRACK_BAND_CELLS * dx);
            let gf = f64::from(m.fracture_energy);
            eprintln!(
                "rock, dx = {dx:.4} m ({share} of the crack-band limit): peak {peak:.4e} Pa \
                 (f_t {ft:.4e}), dissipated {per_area:.3} J/m² (G_f {gf}), final d {d}"
            );
            assert!((peak / ft - 1.0).abs() <= 1e-3, "peak {peak} vs f_t {ft}");
            assert!(
                (per_area / gf - 1.0).abs() <= 5e-3,
                "dx {dx}: {per_area} J/m² vs G_f {gf}"
            );
        }
    }

    /// Past the peak, unloading follows the secant (the damage is kept), and
    /// reloading meets the softening curve where it left it.
    #[test]
    fn rock_unloads_along_the_secant_and_keeps_its_damage() {
        let m = RockMaterial::default();
        let dx = 0.5 * m.max_cell_size();
        let e = f64::from(m.youngs_modulus);
        let e1 = 1.5 * f64::from(m.tensile_strength) / e;
        let loaded = rock_step(&m, uniaxial(&m, e1), 0.0, dx);
        assert!(loaded.damage > 0.1 && loaded.damage < 0.99, "{loaded:?}");
        for k in [0.75, 0.5, 0.1] {
            let r = rock_step(&m, uniaxial(&m, k * e1), loaded.damage, dx);
            assert_eq!(r.damage, loaded.damage, "damage changed on unloading");
            let secant = f64::from(1.0 - loaded.damage) * e * k * e1;
            assert!(
                (f64::from(r.tau.x) - secant).abs() <= 2e-3 * f64::from(m.tensile_strength),
                "unloading to {k}: {} vs secant {secant}",
                r.tau.x
            );
        }
        let again = rock_step(&m, uniaxial(&m, e1), loaded.damage, dx);
        assert_eq!(again, loaded);
    }

    /// Compression up to ten times the tensile strength (a rock's compressive
    /// strength is about 10–20 f_t) stays elastic and undamaged, uniaxial,
    /// confined or hydrostatic; and cracked rock carries compression in full
    /// (its cracks close).
    #[test]
    fn rock_in_compression_stays_elastic() {
        let m = RockMaterial::default();
        let (lambda, mu) = (m.lambda(), m.mu());
        let dx = 0.5 * m.max_cell_size();
        let limit = 10.0 * m.tensile_strength;
        let hencky = |sigma: Vec3| {
            let e = Vec3::new(sigma.x.ln(), sigma.y.ln(), sigma.z.ln());
            2.0 * mu * e + Vec3::splat(lambda * (e.x + e.y + e.z))
        };
        for k in [0.1f32, 0.5, 1.0] {
            let q = k * limit;
            // Uniaxial stress, the compressed axis last (the SVD's order);
            // uniaxial strain (confined); hydrostatic.
            let uniaxial = exp3(strain_of(Vec3::new(0.0, 0.0, -q), lambda, mu));
            let confined = Vec3::new(1.0, 1.0, (-q / (lambda + 2.0 * mu)).exp());
            let hydro = Vec3::splat((-q / (3.0 * lambda + 2.0 * mu)).exp());
            for sigma in [uniaxial, confined, hydro] {
                let want = hencky(sigma);
                for d in [0.0f32, 0.5, 1.0] {
                    let r = rock_step(&m, sigma, d, dx);
                    assert_eq!(r.damage, d, "compression changed the damage: {sigma:?}");
                    // The uniaxial case's zero lateral stresses come out of the
                    // stretches' rounding (2e-7 of 3λ + 2μ, a few kPa), which
                    // the damage may scale; the rest is exact.
                    let err = (r.tau - want).abs().max_element();
                    let round = 2e-7 * (3.0 * lambda + 2.0 * mu);
                    assert!(
                        err <= round + 1e-5 * q,
                        "{sigma:?}, d {d}: τ {:?} vs {want:?}",
                        r.tau
                    );
                }
            }
        }
    }

    #[test]
    fn rock_damage_only_grows_and_no_stress_passes_the_tensile_strength() {
        let m = RockMaterial::default();
        let mut rng = Rng(43);
        for _ in 0..20_000 {
            let dx = rng.range(0.05, 0.95) * m.max_cell_size();
            let mut v = [
                rng.range(-2e-3, 2e-3),
                rng.range(-2e-3, 2e-3),
                rng.range(-2e-3, 2e-3),
            ];
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let sigma = exp3(Vec3::from_array(v));
            let d0 = if rng.next() & 1 == 0 {
                0.0
            } else {
                rng.range(0.0, 1.0)
            };
            let r = rock_step(&m, sigma, d0, dx);
            assert!(r.damage >= d0 && r.damage <= 1.0, "{d0} → {}", r.damage);
            assert!(r.tau.is_finite(), "{r:?}");
            // (1 − d)·τ̄₁ ≤ (1 − d(τ̄₁))·τ̄₁ = f_t·exp(…) ≤ f_t.
            assert!(
                r.tau.max_element() <= m.tensile_strength * (1.0 + 1e-5),
                "{sigma:?}, d {d0}: {r:?}"
            );
        }
        // Far past the peak the rock is fully broken: no tension at all.
        let r = rock_step(&m, Vec3::splat(1.01), 0.0, 0.5 * m.max_cell_size());
        assert_eq!(r.damage, 1.0);
        assert_eq!(r.tau, Vec3::ZERO);
    }

    #[test]
    fn the_crack_band_rejects_cells_past_its_limit() {
        let m = RockMaterial::default();
        // E·G_f/f_t² = 30e9·100/25e12 = 0.12 m; the largest cell 2·0.12/2 m.
        assert!((m.characteristic_length() - 0.12).abs() < 1e-6);
        assert!((m.max_cell_size() - 0.12).abs() < 1e-6);
        assert!(m.softening_stress(0.99 * m.max_cell_size()).is_ok());
        for dx in [
            m.max_cell_size(),
            1.01 * m.max_cell_size(),
            10.0,
            0.0,
            -1.0,
            f32::NAN,
        ] {
            assert!(
                matches!(m.softening_stress(dx), Err(TerrainError::CrackBand { .. })),
                "dx {dx} accepted"
            );
        }
        // σ_s over the band h = 2·dx: E·G_f/(f_t·h) − f_t/2.
        let s = m.softening_stress(0.05).unwrap();
        assert!((s - (30e9 * 100.0 / (5e6 * 0.1) - 2.5e6)).abs() <= 1e-6 * s);
    }

    #[test]
    fn nonphysical_rock_is_rejected() {
        let d = RockMaterial::default();
        d.validate().unwrap();
        let bad = [
            RockMaterial { density: 0.0, ..d },
            RockMaterial {
                youngs_modulus: f32::NAN,
                ..d
            },
            RockMaterial {
                poisson_ratio: 0.5,
                ..d
            },
            RockMaterial {
                tensile_strength: 0.0,
                ..d
            },
            RockMaterial {
                tensile_strength: f32::INFINITY,
                ..d
            },
            RockMaterial {
                fracture_energy: -1.0,
                ..d
            },
            RockMaterial {
                fracture_energy: f32::NAN,
                ..d
            },
        ];
        for m in bad {
            assert!(
                matches!(m.validate(), Err(TerrainError::InvalidConfig(_))),
                "{m:?} accepted"
            );
        }
        assert_eq!(TerrainMaterial::from(None), TerrainMaterial::Inert);
        assert_eq!(TerrainMaterial::from(&d), TerrainMaterial::Rock(d));
    }
}
