//! Point sources for km/s impacts and explosions (`engine/docs/deformable-terrain.md`,
//! "Event sources"). The contact and compression stage is not simulated: it is
//! replaced by an initial excavation velocity field ([`crate::zmodel::ZModel`]),
//! and the MPM computes excavation, ejection, rim uplift and collapse. Scaling
//! laws enter in two places only, both here: the size of the active region
//! ([`CraterPrediction::region`]) and the magnitude of the initial field
//! ([`ImpactSource::field`], [`ExplosionSource::field`]). The crater's shape is
//! never prescribed.
//!
//! # Crater size: Holsapple's point-source π-scaling
//!
//! Holsapple (1993, "The scaling of impact processes in planetary sciences",
//! Annu. Rev. Earth Planet. Sci. 21, Eq. 18), in the form of Holsapple's
//! "Theory and equations for Craters from Impacts and Explosions" (web page of
//! the crater calculator, Eq. 7; the table Singer et al. 2020, JGR 125, record
//! as downloaded in 2013) and Holsapple (2022, "IMPACTS!", arXiv:2203.07476,
//! Eq. 3):
//!
//! ```text
//! πV = K1 · { π2·(ρ/δ)^((6ν−2−μ)/(3μ)) + [K2·π3·(ρ/δ)^((6ν−2)/(3μ))]^((2+μ)/2) }^(−3μ/(2+μ))
//! πV = ρV/m,   π2 = g·a/U²,   π3 = Y/(ρU²)
//! ```
//!
//! for an impactor of radius a, density δ, mass m and speed U into a target of
//! density ρ and strength Y under gravity g. The π2 term dominates large events
//! (the gravity regime), the π3 term small ones (the strength regime). Simple
//! craters have R = Kr·V^(1/3) and depth d = Kd·V^(1/3) (both at the pre-impact
//! surface). In radius form, πR = R·(ρ/m)^(1/3) = Kr·K1^(1/3)·{…}^(−μ/(2+μ)).
//!
//! Constants ([`CraterScaling`]) are the web page's table (cgs there, SI here):
//! dry sand K1 = 0.132, K2 = 0, μ = 0.41, ν = 0.33, Y = 0, ρ = 1.7 g/cm³, and
//! Kr = 1.4, Kd = 0.35. Holsapple (2022, Table 1) revises dry sand to K1 = 0.15,
//! μ = 0.40; the 2013 table is kept because its μ = 0.41 matches the ejecta
//! fits of Housen & Holsapple (2011) used for the field.
//!
//! Final crater (the web page, §2.1 and Eqs. 11–12): the rim radius of a simple
//! crater is 1.3× the excavation radius; past the simple-to-complex transition
//! (lunar rim diameter 10.6 km, Pike 1977; R* ∝ Y/(ρg), scaled here by gravity
//! alone) the final rim radius is 1.33·Re^1.086·R*^(−0.086). Formation time
//! T = 0.8·√(V^(1/3)/g) (Schmidt & Housen 1987, via the web page Eq. 8).
//!
//! # Obliquity
//!
//! Holsapple (web page §2; 2022 §3.3): "For non-vertical impacts, the vertical
//! component U cos(θ) is used." The scaling here uses the velocity's component
//! into the ground (the target surface is horizontal, +y up). The field stays
//! axisymmetric about the vertical through the impact point: Anderson, Schultz
//! & Heineck (2003, JGR 108) found a single stationary Z-model inadequate for
//! oblique impacts, and no published tilt or offset of the flow centre was
//! found that is reliable enough to adopt.
//!
//! # Explosions
//!
//! Holsapple (web page §3): for explosives "the specific energy of the
//! impactor, which is ½U², is replaced by the specific energy Q of the
//! explosive", and (§2) impacts give about the same crater volume as an
//! explosive of the same specific energy and density "buried about 1.5
//! impactor radii" (Holsapple 1980, "The equivalent depth of burst for impact
//! cratering", Proc. LPSC 11). A charge of energy E, specific energy Q and
//! density δ is therefore the impactor of mass W = E/Q, radius
//! a = (3W/4πδ)^(1/3) and speed √(2Q), provided it is buried near that
//! equivalent depth. Other depths of burial need the scaled-depth curves of
//! Holsapple & Schmidt (1980, JGR 85) or the Nordyke/Chabai charts, which were
//! not available to check, so they are rejected ([`EQUIVALENT_BURIAL_RANGE`])
//! rather than guessed: in particular, surface-tangent and contact charges.
//! TNT equivalence uses the convention 1 kg TNT = 4.184 MJ.

use crate::config::{TerrainConfig, TerrainError};
use crate::material::SandMaterial;
use crate::region::RegionPlan;
use crate::zmodel::{EjectaScaling, ZModel};
use glam::{DVec3, Vec3};

/// Energy of 1 kg of TNT by the TNT-equivalence convention (J/kg).
pub const TNT_SPECIFIC_ENERGY: f64 = 4.184e6;
/// Density of cast TNT (kg/m³).
pub const TNT_DENSITY: f64 = 1650.0;
/// Slowest impact (normal component, m/s) treated as a point source. Holsapple
/// (2022, §3.3): below 1 km/s "the data is sketchy and the point source
/// assumption becomes iffy". Slower bodies couple to the MPM directly (T3).
pub const MIN_POINT_SOURCE_SPEED: f64 = 1000.0;
/// Depth of burial (in charge radii, to the charge centre) at which an
/// explosion makes the crater of the impact of the same specific energy and
/// density (Holsapple 1980, via the web page §2). Also the default depth of an
/// impact's flow centre, in impactor radii.
pub const EQUIVALENT_BURIAL: f64 = 1.5;
/// Depths of burial (charge radii) accepted for explosions: Holsapple's "about
/// 1.5" with ±0.5 for "about". Outside it, the request fails.
pub const EQUIVALENT_BURIAL_RANGE: (f64, f64) = (1.0, 2.0);
/// Rim radius of a simple crater over its excavation radius (Holsapple, web page
/// §2.1; 2022 §3.4).
pub const RIM_TO_EXCAVATION: f64 = 1.3;
/// Lunar surface gravity (m/s²) and the rim radius at which lunar craters turn
/// complex (half Pike's 1977 10.6 km diameter, via Holsapple).
pub const LUNAR_GRAVITY: f64 = 1.62;
pub const LUNAR_COMPLEX_RIM_RADIUS: f64 = 5300.0;

/// (4π/3)^(1/3): the radius of a sphere over the cube root of its volume.
fn sphere_k() -> f64 {
    (4.0 * std::f64::consts::PI / 3.0).cbrt()
}

fn out_of_range(quantity: &'static str, value: f64, min: f64, max: f64) -> TerrainError {
    TerrainError::SourceOutOfRange {
        quantity,
        value,
        min,
        max,
    }
}

fn positive(quantity: &'static str, value: f64) -> Result<(), TerrainError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(out_of_range(quantity, value, f64::MIN_POSITIVE, f64::MAX))
    }
}

/// Holsapple's crater-size constants for one target type (SI units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CraterScaling {
    pub k1: f64,
    pub k2: f64,
    /// Velocity exponent of the coupling parameter a·U^μ·δ^ν.
    pub mu: f64,
    /// Density exponent of the coupling parameter.
    pub nu: f64,
    /// Cratering strength Y (Pa).
    pub strength: f64,
    /// Target bulk density ρ (kg/m³).
    pub density: f64,
    /// R = Kr·V^(1/3).
    pub kr: f64,
    /// Depth = Kd·V^(1/3).
    pub kd: f64,
}

impl CraterScaling {
    /// Dry (Ottawa) sand, gravity regime only. Holsapple's table: K1 0.132,
    /// K2 0, μ 0.41, ν 0.33, Y 0, ρ 1.7 g/cm³; Kr 1.4, Kd 0.35.
    pub const DRY_SAND: Self = Self {
        k1: 0.132,
        k2: 0.0,
        mu: 0.41,
        nu: 0.33,
        strength: 0.0,
        density: 1700.0,
        kr: 1.4,
        kd: 0.35,
    };
    /// Dry soil with some cohesion: K1 0.132, K2 0.26, μ 0.41, ν 0.33,
    /// Y 2·10⁶ dyn/cm², ρ 1.7 g/cm³; Kr 1.1, Kd 0.6.
    pub const DRY_SOIL: Self = Self {
        k1: 0.132,
        k2: 0.26,
        mu: 0.41,
        nu: 0.33,
        strength: 2.0e5,
        density: 1700.0,
        kr: 1.1,
        kd: 0.6,
    };
    /// Soft rock (hard soil): K1 0.095, K2 0.215, μ 0.55, ν 0.33,
    /// Y 10⁷ dyn/cm², ρ 2.1 g/cm³; Kr 1.1, Kd 0.6.
    pub const SOFT_ROCK: Self = Self {
        k1: 0.095,
        k2: 0.215,
        mu: 0.55,
        nu: 0.33,
        strength: 1.0e6,
        density: 2100.0,
        kr: 1.1,
        kd: 0.6,
    };
    /// Hard rock: K1 0.095, K2 0.257, μ 0.55, ν 0.33, Y 10⁸ dyn/cm²,
    /// ρ 3.2 g/cm³; Kr 1.1, Kd 0.6 (Holsapple 2022, Table 2).
    pub const HARD_ROCK: Self = Self {
        k1: 0.095,
        k2: 0.257,
        mu: 0.55,
        nu: 0.33,
        strength: 1.0e7,
        density: 3200.0,
        kr: 1.1,
        kd: 0.6,
    };

    pub fn validate(&self) -> Result<(), TerrainError> {
        positive("k1", self.k1)?;
        positive("density", self.density)?;
        positive("kr", self.kr)?;
        positive("kd", self.kd)?;
        // μ between momentum (1/3) and energy (2/3) scaling, with slack for
        // fits such as perlite/sand (0.35).
        if !(self.mu > 0.0 && self.mu <= 1.0) {
            return Err(out_of_range("mu", self.mu, 0.0, 1.0));
        }
        if !(0.0..=1.0).contains(&self.nu) {
            return Err(out_of_range("nu", self.nu, 0.0, 1.0));
        }
        if !(self.k2.is_finite() && self.k2 >= 0.0) {
            return Err(out_of_range("k2", self.k2, 0.0, f64::MAX));
        }
        if !(self.strength.is_finite() && self.strength >= 0.0) {
            return Err(out_of_range("strength", self.strength, 0.0, f64::MAX));
        }
        Ok(())
    }

    /// The gravity and strength terms inside the braces of πV.
    pub fn terms(&self, pi2: f64, pi3: f64, density_ratio: f64) -> (f64, f64) {
        let (mu, nu) = (self.mu, self.nu);
        let gravity = pi2 * density_ratio.powf((6.0 * nu - 2.0 - mu) / (3.0 * mu));
        let strength = (self.k2 * pi3 * density_ratio.powf((6.0 * nu - 2.0) / (3.0 * mu)))
            .powf((2.0 + mu) / 2.0);
        (gravity, strength)
    }

    /// πV = ρV/m (Holsapple 1993, Eq. 18). `density_ratio` is ρ/δ.
    pub fn pi_v(&self, pi2: f64, pi3: f64, density_ratio: f64) -> f64 {
        let (g, s) = self.terms(pi2, pi3, density_ratio);
        self.k1 * (g + s).powf(-3.0 * self.mu / (2.0 + self.mu))
    }

    /// The transient crater of an impactor of radius `a` (m), density `delta`
    /// (kg/m³) and normal speed `u` (m/s) under gravity `g` (m/s²). Inputs are
    /// not range-checked here; [`ImpactSource::crater`] does that.
    pub fn predict(&self, a: f64, u: f64, delta: f64, g: f64) -> CraterPrediction {
        let rho = self.density;
        let m = 4.0 / 3.0 * std::f64::consts::PI * delta * a.powi(3);
        let pi2 = g * a / (u * u);
        let pi3 = self.strength / (rho * u * u);
        let ratio = rho / delta;
        let (gt, st) = self.terms(pi2, pi3, ratio);
        let pi_v = self.pi_v(pi2, pi3, ratio);
        let volume = pi_v * m / rho;
        let cube = volume.cbrt();
        let radius = self.kr * cube;
        let rim_radius = RIM_TO_EXCAVATION * radius;
        let transition = LUNAR_COMPLEX_RIM_RADIUS * LUNAR_GRAVITY / g;
        let complex = rim_radius > transition;
        let final_rim_radius = if complex {
            1.33 * radius.powf(1.086) * transition.powf(-0.086)
        } else {
            rim_radius
        };
        CraterPrediction {
            pi2,
            pi3,
            pi_v,
            regime: if gt >= st {
                Regime::Gravity
            } else {
                Regime::Strength
            },
            volume,
            radius,
            depth: self.kd * cube,
            rim_radius,
            final_rim_radius,
            complex,
            formation_time: 0.8 * (cube / g).sqrt(),
        }
    }
}

/// Which term of πV dominates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Gravity,
    Strength,
}

/// A predicted transient crater (point-source π-scaling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CraterPrediction {
    pub pi2: f64,
    pub pi3: f64,
    pub pi_v: f64,
    pub regime: Regime,
    /// Transient (excavation) crater volume below the pre-impact surface (m³).
    pub volume: f64,
    /// Transient crater radius at the pre-impact surface, Kr·V^(1/3) (m).
    pub radius: f64,
    /// Transient crater depth, Kd·V^(1/3) (m).
    pub depth: f64,
    /// Rim radius of the simple crater, 1.3·radius (m).
    pub rim_radius: f64,
    /// Final rim radius (m): the simple rim, or Holsapple's complex-crater fit.
    pub final_rim_radius: f64,
    pub complex: bool,
    /// Crater formation time 0.8·√(V^(1/3)/g) (s).
    pub formation_time: f64,
}

impl CraterPrediction {
    /// The active region for this crater: cells of radius/`cells_per_radius`
    /// and a box reaching radius·(1 + `margin`) around `center` (the impact
    /// point on the surface) sideways and below it, and as far above. The
    /// margin must at least reach the final rim; how much more it needs so the
    /// box does not limit the cavity is what the insensitivity test measures.
    pub fn region(
        &self,
        config: &TerrainConfig,
        center: Vec3,
        margin: f64,
        active_regions: u32,
        active_blocks: u32,
    ) -> Result<RegionPlan, TerrainError> {
        let need = self.final_rim_radius / self.radius - 1.0;
        if !(margin.is_finite() && margin >= need) {
            return Err(out_of_range("margin", margin, need, f64::MAX));
        }
        RegionPlan::around(
            config,
            center,
            self.radius as f32,
            margin as f32,
            active_regions,
            active_blocks,
        )
    }
}

/// The scaling constants of one target: crater size (Holsapple) and ejection
/// velocity (Housen & Holsapple 2011), which sets the initial field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CraterTarget {
    pub crater: CraterScaling,
    pub ejecta: EjectaScaling,
}

impl CraterTarget {
    pub const DRY_SAND: Self = Self {
        crater: CraterScaling::DRY_SAND,
        ejecta: EjectaScaling::DRY_SAND,
    };

    /// Dry sand at the MPM material's density. The dry-sand constants are for
    /// cohesionless sand, so a cohesive material is rejected.
    pub fn sand(material: &SandMaterial) -> Result<Self, TerrainError> {
        material.validate()?;
        if material.cohesion != 0.0 {
            return Err(out_of_range(
                "sand cohesion",
                f64::from(material.cohesion),
                0.0,
                0.0,
            ));
        }
        let mut t = Self::DRY_SAND;
        t.crater.density = f64::from(material.density);
        Ok(t)
    }

    pub fn validate(&self) -> Result<(), TerrainError> {
        self.crater.validate()?;
        self.ejecta.validate()
    }
}

fn check_gravity(g: f64) -> Result<(), TerrainError> {
    positive("gravity", g)
}

/// A hypervelocity impact on horizontal ground (+y up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactSource {
    /// Impactor radius a (m).
    pub impactor_radius: f64,
    /// Impactor density δ (kg/m³).
    pub impactor_density: f64,
    /// Impact velocity (m/s); its downward component sets the scaling.
    pub velocity: DVec3,
    pub target: CraterTarget,
}

impl ImpactSource {
    pub fn validate(&self) -> Result<(), TerrainError> {
        positive("impactor_radius", self.impactor_radius)?;
        positive("impactor_density", self.impactor_density)?;
        self.target.validate()?;
        if !self.velocity.is_finite() {
            return Err(out_of_range("velocity", f64::NAN, f64::MIN, f64::MAX));
        }
        let u = self.normal_speed();
        if u < MIN_POINT_SOURCE_SPEED {
            return Err(out_of_range(
                "normal impact speed",
                u,
                MIN_POINT_SOURCE_SPEED,
                f64::MAX,
            ));
        }
        Ok(())
    }

    /// The velocity's component into the ground, U·cos θ (m/s).
    pub fn normal_speed(&self) -> f64 {
        -self.velocity.y
    }

    pub fn mass(&self) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * self.impactor_density * self.impactor_radius.powi(3)
    }

    /// ½·m·|v|² (J).
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass() * self.velocity.length_squared()
    }

    /// The transient crater under gravity `g` (m/s²).
    pub fn crater(&self, g: f64) -> Result<CraterPrediction, TerrainError> {
        self.validate()?;
        check_gravity(g)?;
        Ok(self.target.crater.predict(
            self.impactor_radius,
            self.normal_speed(),
            self.impactor_density,
            g,
        ))
    }

    /// The initial excavation field with exponent `z`, its flow centre
    /// [`EQUIVALENT_BURIAL`] impactor radii below the impact point and its core
    /// twice as far. Fails if the field holds more kinetic energy than the
    /// impactor brought.
    pub fn field(&self, g: f64, z: f64) -> Result<ZModel, TerrainError> {
        let h = EQUIVALENT_BURIAL * self.impactor_radius;
        self.field_with(g, z, h, 2.0 * h)
    }

    /// [`Self::field`] with the flow-centre depth and core radius chosen.
    pub fn field_with(
        &self,
        g: f64,
        z: f64,
        flow_center_depth: f64,
        core_radius: f64,
    ) -> Result<ZModel, TerrainError> {
        let crater = self.crater(g)?;
        let field = ZModel::calibrated(
            z,
            flow_center_depth,
            core_radius,
            &self.target.ejecta,
            self.impactor_radius,
            self.normal_speed(),
            self.impactor_density,
            self.target.crater.density,
            crater.radius,
        )?
        .bounded(crater.radius)?;
        let field_j = field.kinetic_energy(self.target.crater.density);
        let source_j = self.kinetic_energy();
        if field_j > source_j {
            return Err(TerrainError::FieldEnergy { field_j, source_j });
        }
        Ok(field)
    }

    /// The active region around the impact point `center` (see
    /// [`CraterPrediction::region`]).
    pub fn region(
        &self,
        config: &TerrainConfig,
        g: f64,
        center: Vec3,
        margin: f64,
        active_regions: u32,
        active_blocks: u32,
    ) -> Result<RegionPlan, TerrainError> {
        self.crater(g)?
            .region(config, center, margin, active_regions, active_blocks)
    }
}

/// An explosive's specific energy (J/kg) and density (kg/m³).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Explosive {
    pub specific_energy: f64,
    pub density: f64,
}

impl Explosive {
    pub const TNT: Self = Self {
        specific_energy: TNT_SPECIFIC_ENERGY,
        density: TNT_DENSITY,
    };
}

/// A buried charge under horizontal ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplosionSource {
    /// Energy released (J).
    pub energy: f64,
    /// Depth of the charge centre below the surface (m).
    pub depth_of_burial: f64,
    pub explosive: Explosive,
    pub target: CraterTarget,
}

impl ExplosionSource {
    /// `kg` of TNT-equivalent buried at `depth_of_burial` (m, to its centre).
    pub fn tnt(kg: f64, depth_of_burial: f64, target: CraterTarget) -> Self {
        Self {
            energy: kg * TNT_SPECIFIC_ENERGY,
            depth_of_burial,
            explosive: Explosive::TNT,
            target,
        }
    }

    /// Charge mass W = E/Q (kg).
    pub fn charge_mass(&self) -> f64 {
        self.energy / self.explosive.specific_energy
    }

    /// Radius of the spherical charge (m).
    pub fn charge_radius(&self) -> f64 {
        (self.charge_mass() / self.explosive.density).cbrt() / sphere_k()
    }

    /// The impact this charge is equivalent to (Holsapple: ½U² → Q, same mass
    /// and density). Fails unless the charge is buried within
    /// [`EQUIVALENT_BURIAL_RANGE`] charge radii.
    pub fn equivalent_impact(&self) -> Result<ImpactSource, TerrainError> {
        positive("energy", self.energy)?;
        positive("specific_energy", self.explosive.specific_energy)?;
        positive("explosive density", self.explosive.density)?;
        let a = self.charge_radius();
        let scaled = self.depth_of_burial / a;
        let (lo, hi) = EQUIVALENT_BURIAL_RANGE;
        if !(scaled >= lo && scaled <= hi) {
            return Err(out_of_range("scaled depth of burial", scaled, lo, hi));
        }
        Ok(ImpactSource {
            impactor_radius: a,
            impactor_density: self.explosive.density,
            velocity: DVec3::new(0.0, -(2.0 * self.explosive.specific_energy).sqrt(), 0.0),
            target: self.target,
        })
    }

    pub fn crater(&self, g: f64) -> Result<CraterPrediction, TerrainError> {
        self.equivalent_impact()?.crater(g)
    }

    /// The initial excavation field with exponent `z`, flowing from the charge
    /// centre, with its core at twice the depth of burial. Fails if it holds
    /// more kinetic energy than the charge releases.
    pub fn field(&self, g: f64, z: f64) -> Result<ZModel, TerrainError> {
        let impact = self.equivalent_impact()?;
        let h = self.depth_of_burial;
        let crater = impact.crater(g)?;
        let field = ZModel::calibrated(
            z,
            h,
            2.0 * h,
            &impact.target.ejecta,
            impact.impactor_radius,
            impact.normal_speed(),
            impact.impactor_density,
            impact.target.crater.density,
            crater.radius,
        )?
        .bounded(crater.radius)?;
        let field_j = field.kinetic_energy(impact.target.crater.density);
        if field_j > self.energy {
            return Err(TerrainError::FieldEnergy {
                field_j,
                source_j: self.energy,
            });
        }
        Ok(field)
    }

    /// The active region around the surface point above the charge.
    pub fn region(
        &self,
        config: &TerrainConfig,
        g: f64,
        center: Vec3,
        margin: f64,
        active_regions: u32,
        active_blocks: u32,
    ) -> Result<RegionPlan, TerrainError> {
        self.crater(g)?
            .region(config, center, margin, active_regions, active_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::BLOCK_CELLS;

    const G: f64 = 9.81;

    fn sand_impact(a: f64, u: f64) -> ImpactSource {
        ImpactSource {
            impactor_radius: a,
            impactor_density: 2700.0,
            velocity: DVec3::new(0.0, -u, 0.0),
            target: CraterTarget::DRY_SAND,
        }
    }

    /// d ln y / d ln x across [x0, x1] from the end points.
    fn slope(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        (y1 / y0).ln() / (x1 / x0).ln()
    }

    #[test]
    fn gravity_regime_slopes_are_the_point_source_exponents() {
        // πR ∝ π2^(−μ/(2+μ)): R vs impactor radius at fixed U goes as
        // a^(2/(2+μ)), so R vs energy E ∝ a³ as E^(2/(3(2+μ))); at fixed a,
        // R vs U goes as U^(2μ/(2+μ)), i.e. E^(μ/(2+μ)). Two decades of energy.
        let mu = CraterScaling::DRY_SAND.mu;
        let (a0, a1) = (0.01, 0.01 * 100f64.cbrt());
        let r0 = sand_impact(a0, 3000.0).crater(G).unwrap();
        let r1 = sand_impact(a1, 3000.0).crater(G).unwrap();
        assert_eq!(r0.regime, Regime::Gravity);
        let e0 = sand_impact(a0, 3000.0).kinetic_energy();
        let e1 = sand_impact(a1, 3000.0).kinetic_energy();
        assert!((e1 / e0 - 100.0).abs() < 1e-9);
        let s = slope(e0, r0.radius, e1, r1.radius);
        assert!((s - 2.0 / (3.0 * (2.0 + mu))).abs() < 1e-9, "{s}");
        // πR against π2.
        let m = |c: &CraterPrediction, a: f64| {
            c.radius * (CraterScaling::DRY_SAND.density / sand_impact(a, 3000.0).mass()).cbrt()
        };
        let s = slope(r0.pi2, m(&r0, a0), r1.pi2, m(&r1, a1));
        assert!((s + mu / (2.0 + mu)).abs() < 1e-9, "{s}");
        // Velocity at fixed size: E ∝ U², two decades in E.
        let v0 = sand_impact(0.01, 1500.0).crater(G).unwrap();
        let v1 = sand_impact(0.01, 15000.0).crater(G).unwrap();
        let s = slope(1500f64.powi(2), v0.radius, 15000f64.powi(2), v1.radius);
        assert!((s - mu / (2.0 + mu)).abs() < 1e-9, "{s}");
        // Gravity: R ∝ g^(−μ/(2+μ)).
        let lo = sand_impact(0.01, 3000.0).crater(1.62).unwrap();
        let s = slope(1.62, lo.radius, G, r0.radius);
        assert!((s + mu / (2.0 + mu)).abs() < 1e-9, "{s}");
    }

    #[test]
    fn energy_scaling_gives_quarter_root_gravity_scaling_for_explosions() {
        // With μ = 2/3 (energy scaling) and ν = 1/3, gravity-regime volume goes
        // as W^(3/4), R as W^(1/4) (Holsapple & Schmidt 1980).
        let mut target = CraterTarget::DRY_SAND;
        target.crater.mu = 2.0 / 3.0;
        target.crater.nu = 1.0 / 3.0;
        let c = |kg: f64| {
            let e = ExplosionSource::tnt(kg, 0.0, target);
            let e = ExplosionSource {
                depth_of_burial: 1.5 * e.charge_radius(),
                ..e
            };
            e.crater(G).unwrap()
        };
        let (c0, c1) = (c(1.0), c(100.0));
        assert!((slope(1.0, c0.volume, 100.0, c1.volume) - 0.75).abs() < 1e-9);
        assert!((slope(1.0, c0.radius, 100.0, c1.radius) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn rock_exponents_and_size_match_schmidt_housen() {
        // Collins, Melosh & Marcus (2005, MAPS 40, Eq. 21) quote Schmidt &
        // Housen (1987) for rock: D_tc = 1.161 (δ/ρ)^(1/3) L^0.78 v^0.44 g^−0.22
        // (L the impactor diameter). With μ = 0.55 the point-source exponents
        // are 2/(2+μ) = 0.784, 2μ/(2+μ) = 0.431, −μ/(2+μ) = −0.216.
        let rock = CraterScaling::HARD_ROCK;
        let mu = rock.mu;
        assert!((2.0 / (2.0 + mu) - 0.78).abs() < 0.005);
        assert!((2.0 * mu / (2.0 + mu) - 0.44).abs() < 0.01);
        assert!((mu / (2.0 + mu) - 0.22).abs() < 0.005);
        // Absolute size for a 1 km rock impactor at 20 km/s on Earth, equal
        // densities: two independent fits within 30%.
        let (l, v) = (1000.0, 20000.0);
        let ours = 2.0 * rock.predict(l / 2.0, v, rock.density, G).radius;
        let theirs = 1.161 * l.powf(0.78) * v.powf(0.44) * G.powf(-0.22);
        assert_eq!(
            rock.predict(l / 2.0, v, rock.density, G).regime,
            Regime::Gravity
        );
        let r = ours / theirs;
        assert!((0.7..1.3).contains(&r), "ours {ours} m, theirs {theirs} m");
    }

    #[test]
    fn sand_radius_agrees_with_housen_holsapple_2011() {
        // Housen & Holsapple (2011, Icarus 211; sand constants via Hirata &
        // Ikeya 2022): R·(ρ/m)^(1/3) = H1·(ρ/δ)^((2+μ−6ν)/(3(2+μ)))·π2^(−μ/(2+μ))
        // with H1 = 0.59, μ = 0.41, ν = 0.4. Holsapple's table gives
        // Kr·K1^(1/3) = 0.71 for the same quantity: an independent fit.
        let s = sand_impact(0.003, 2000.0);
        let c = s.crater(G).unwrap();
        let (mu, nu, rho, delta) = (0.41, 0.4, CraterScaling::DRY_SAND.density, 2700.0);
        let hh = 0.59
            * (s.mass() / rho).cbrt()
            * (rho / delta).powf((2.0 + mu - 6.0 * nu) / (3.0 * (2.0 + mu)))
            * c.pi2.powf(-mu / (2.0 + mu));
        let r = c.radius / hh;
        assert!(
            (0.8..1.25).contains(&r),
            "Holsapple {} m, H&H {hh} m",
            c.radius
        );
    }

    #[test]
    fn strength_to_gravity_transition_is_monotonic() {
        let soil = CraterScaling::DRY_SOIL;
        let (u, delta) = (2000.0, 2700.0);
        let mut last_efficiency = f64::INFINITY;
        let mut regimes = Vec::new();
        for k in 0..=60 {
            let a = 1e-3 * 10f64.powf(k as f64 / 10.0); // 1 mm to 1 km
            let c = soil.predict(a, u, delta, G);
            // πV never rises with size, and lies under both asymptotes.
            assert!(c.pi_v <= last_efficiency * (1.0 + 1e-12), "a = {a}");
            last_efficiency = c.pi_v;
            let (gt, st) = soil.terms(c.pi2, c.pi3, soil.density / delta);
            let pow = -3.0 * soil.mu / (2.0 + soil.mu);
            assert!(c.pi_v <= soil.k1 * gt.powf(pow) * (1.0 + 1e-12));
            assert!(c.pi_v <= soil.k1 * st.powf(pow) * (1.0 + 1e-12));
            regimes.push(c.regime);
        }
        assert_eq!(regimes[0], Regime::Strength);
        assert_eq!(*regimes.last().unwrap(), Regime::Gravity);
        // One switch only.
        let switches = regimes.windows(2).filter(|w| w[0] != w[1]).count();
        assert_eq!(switches, 1);
        // Deep in the strength regime the crater is geometrically similar
        // (R ∝ a) and independent of gravity.
        let c0 = soil.predict(1e-3, u, delta, G);
        let c1 = soil.predict(1e-2, u, delta, G);
        let s = slope(1e-3, c0.radius, 1e-2, c1.radius);
        assert!((s - 1.0).abs() < 0.01, "{s}");
        let moon = soil.predict(1e-3, u, delta, 1.62);
        assert!((moon.radius / c0.radius - 1.0).abs() < 0.01);
        // Deep in the gravity regime it follows the gravity slope.
        let c0 = soil.predict(100.0, u, delta, G);
        let c1 = soil.predict(1000.0, u, delta, G);
        let s = slope(100.0, c0.radius, 1000.0, c1.radius);
        assert!((s - 2.0 / (2.0 + soil.mu)).abs() < 0.01, "{s}");
    }

    #[test]
    fn crater_shape_constants_and_final_rim() {
        let c = sand_impact(0.01, 3000.0).crater(G).unwrap();
        assert!((c.depth / c.radius - 0.35 / 1.4).abs() < 1e-12);
        assert!((c.rim_radius / c.radius - 1.3).abs() < 1e-12);
        assert!(!c.complex);
        assert_eq!(c.final_rim_radius, c.rim_radius);
        assert!((c.formation_time - 0.8 * (c.volume.cbrt() / G).sqrt()).abs() < 1e-12);
        // A 1 km impactor on the Moon makes a complex crater wider than its
        // simple rim would be.
        let big = ImpactSource {
            impactor_radius: 500.0,
            impactor_density: 3000.0,
            velocity: DVec3::new(0.0, -15000.0, 0.0),
            target: CraterTarget {
                crater: CraterScaling::HARD_ROCK,
                ..CraterTarget::DRY_SAND
            },
        };
        let c = big.crater(LUNAR_GRAVITY).unwrap();
        assert!(c.complex && c.final_rim_radius > c.rim_radius);
    }

    #[test]
    fn oblique_impacts_scale_with_the_normal_component() {
        let vertical = sand_impact(0.01, 3000.0);
        let mut oblique = vertical;
        let t = 45f64.to_radians();
        oblique.velocity = DVec3::new(3000.0 / t.cos() * t.sin(), -3000.0, 0.0);
        let (a, b) = (vertical.crater(G).unwrap(), oblique.crater(G).unwrap());
        assert!((a.radius - b.radius).abs() < 1e-12);
        // Rotating the impact's azimuth changes nothing.
        let mut turned = oblique;
        turned.velocity = DVec3::new(0.0, -3000.0, 3000.0);
        assert_eq!(turned.crater(G).unwrap(), b);
    }

    #[test]
    fn invalid_and_out_of_range_sources_fail() {
        assert!(sand_impact(0.01, 900.0).crater(G).is_err(), "below 1 km/s");
        let mut up = sand_impact(0.01, 3000.0);
        up.velocity.y = 3000.0;
        assert!(up.crater(G).is_err());
        assert!(sand_impact(-1.0, 3000.0).crater(G).is_err());
        assert!(sand_impact(0.01, 3000.0).crater(0.0).is_err());
        assert!(sand_impact(0.01, 3000.0).crater(f64::NAN).is_err());
        // Cohesive sand has no dry-sand constants.
        let wet = SandMaterial {
            cohesion: 0.01,
            ..SandMaterial::default()
        };
        assert!(CraterTarget::sand(&wet).is_err());
        let t = CraterTarget::sand(&SandMaterial::default()).unwrap();
        assert_eq!(t.crater.density, 1600.0);
    }

    #[test]
    fn explosions_map_to_the_equivalent_impact() {
        let e = ExplosionSource::tnt(1.0, 0.0, CraterTarget::DRY_SAND);
        assert_eq!(e.energy, 4.184e6);
        let a = e.charge_radius();
        // Holsapple (web page, Eq. 18): (g/Q)(W/δ)^(1/3) = 3.22·g·a/U² for the
        // equivalent impactor.
        let pi2_explosive =
            G / e.explosive.specific_energy * (e.charge_mass() / TNT_DENSITY).cbrt();
        let buried = ExplosionSource {
            depth_of_burial: 1.5 * a,
            ..e
        };
        let imp = buried.equivalent_impact().unwrap();
        let pi2 = G * a / imp.normal_speed().powi(2);
        assert!((pi2_explosive / pi2 - 3.22).abs() < 0.005);
        assert!((imp.mass() - 1.0).abs() < 1e-12);
        assert!((imp.kinetic_energy() - buried.energy).abs() < 1e-6);
        let c = buried.crater(G).unwrap();
        assert_eq!(c, imp.crater(G).unwrap());
        // About 1 m radius for 1 kg of TNT in dry sand.
        assert!((0.5..2.0).contains(&c.radius), "{}", c.radius);
        // Surface-tangent, contact and deep charges have no published
        // coefficients here: they fail.
        for dob in [0.0, 0.5 * a, 2.5 * a, 10.0 * a] {
            let x = ExplosionSource {
                depth_of_burial: dob,
                ..e
            };
            assert!(
                matches!(
                    x.crater(G),
                    Err(TerrainError::SourceOutOfRange {
                        quantity: "scaled depth of burial",
                        ..
                    })
                ),
                "dob {dob}"
            );
        }
    }

    #[test]
    fn region_scales_with_the_predicted_radius_and_fails_past_limits() {
        let config = TerrainConfig::default();
        let center = Vec3::new(10.0, 0.0, -4.0);
        let small = sand_impact(0.01, 3000.0);
        // Eight times the energy at the same speed: radius × 8^(2/(3(2+μ))).
        let large = sand_impact(0.02, 3000.0);
        let (cs, cl) = (small.crater(G).unwrap(), large.crater(G).unwrap());
        let ps = small.region(&config, G, center, 0.5, 0, 0).unwrap();
        let pl = large.region(&config, G, center, 0.5, 0, 0).unwrap();
        assert!((ps.dx - cs.radius as f32 / config.cells_per_radius).abs() < 1e-6);
        assert!((pl.dx / ps.dx - (cl.radius / cs.radius) as f32).abs() < 1e-5);
        for (p, c) in [(ps, cs), (pl, cl)] {
            let reach = (c.radius * 1.5) as f32;
            for k in 0..3 {
                assert!(p.origin[k] <= center[k] - reach + 1e-4);
                assert!(p.end()[k] >= center[k] + reach - 1e-4);
            }
            // The same number of cells across, to within block rounding.
            let cells = (2.0 * reach / p.dx) as i32;
            let across = p.blocks.x * BLOCK_CELLS as i32;
            assert!(across >= cells && across <= cells + 2 * BLOCK_CELLS as i32);
        }
        // The margin must reach the rim.
        assert!(matches!(
            small.region(&config, G, center, 0.2, 0, 0),
            Err(TerrainError::SourceOutOfRange {
                quantity: "margin",
                ..
            })
        ));
        // Limits fail by name.
        let tight = TerrainConfig {
            max_blocks: 1000,
            ..config
        };
        assert!(matches!(
            small.region(&tight, G, center, 0.5, 0, 0),
            Err(TerrainError::LimitExceeded {
                limit: "max_blocks",
                ..
            })
        ));
        let fine = TerrainConfig {
            min_cell_size: 1.0,
            ..config
        };
        assert!(matches!(
            small.region(&fine, G, center, 0.5, 0, 0),
            Err(TerrainError::CellTooSmall { .. })
        ));
        assert!(matches!(
            small.region(&config, G, center, 0.5, config.max_regions, 0),
            Err(TerrainError::LimitExceeded {
                limit: "max_regions",
                ..
            })
        ));
    }

    /// The source's field moves only the transient crater (zmodel "Extent"),
    /// and the target then takes the impactor's momentum plus the ejecta's
    /// recoil: a momentum-transfer β of at least 1 (momentum is conserved) and
    /// within the few measured for granular targets.
    #[test]
    fn the_bounded_field_carries_the_impactors_momentum_and_a_recoil() {
        let sand = SandMaterial::default();
        let s = sand_impact(0.005, 2000.0);
        let crater = s.crater(G).unwrap();
        let f = s
            .field(G, crate::zmodel::point_source_z(s.target.ejecta.mu))
            .unwrap();
        assert_eq!(f.extent, crater.radius);
        let rho = f64::from(sand.density);
        let (n, r) = (96usize, crater.radius);
        let d = 2.0 * r / n as f64;
        let mut py = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n / 2 {
                    let p = DVec3::new(
                        -r + (i as f64 + 0.5) * d,
                        -(k as f64 + 0.5) * d,
                        -r + (j as f64 + 0.5) * d,
                    );
                    py += rho * d * d * d * f.velocity(p).y;
                }
            }
        }
        let impactor = 2.0 * s.kinetic_energy() / s.normal_speed();
        let beta = -py / impactor;
        assert!((1.0..=3.0).contains(&beta), "momentum transfer β = {beta}");
        // Nothing moves beyond the crater.
        assert_eq!(f.velocity(DVec3::new(1.01 * r, -0.01, 0.0)), DVec3::ZERO);
    }
}
