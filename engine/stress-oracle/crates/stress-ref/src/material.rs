//! Per-material parameters and the closed-form material laws that depend only on
//! them: dynamic increase factor, static fatigue, Weibull sampling.
//!
//! Units are SI throughout (kg, m, s, Pa, J/m^2).

use serde::{Deserialize, Serialize};

/// How a bond softens after its strength is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BondKind {
    /// Concrete, masonry, mortar, glass: linear softening; the area under the
    /// traction-separation curve is the fracture energy.
    #[default]
    Brittle,
    /// Steel: holds its strength (a plateau) until the absorbed energy reaches the
    /// fracture energy, then snaps.
    Ductile,
}

/// Fracture energies per unit bond area (J/m^2) for each failure family.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FractureEnergy {
    /// Mode I (opening).
    pub tension: f64,
    /// Mode II (sliding).
    pub shear: f64,
    /// Crushing.
    pub compression: f64,
}

/// Dynamic increase factor: strength multiplier as a function of strain rate.
///
/// Piecewise power law in the CEB-FIP / fib Model Code 2010 form:
/// `1` below `reference_rate`, `(rate/reference)^exponent` up to `transition_rate`,
/// then continuing with `exponent_high`, clamped to `max`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DifParams {
    pub reference_rate: f64,
    pub exponent: f64,
    pub transition_rate: f64,
    pub exponent_high: f64,
    pub max: f64,
}

impl DifParams {
    pub fn factor(&self, strain_rate: f64) -> f64 {
        let r = strain_rate.abs();
        if r <= self.reference_rate {
            return 1.0;
        }
        let f = if r <= self.transition_rate {
            (r / self.reference_rate).powf(self.exponent)
        } else {
            (self.transition_rate / self.reference_rate).powf(self.exponent)
                * (r / self.transition_rate).powf(self.exponent_high)
        };
        f.clamp(1.0, self.max)
    }
}

/// Static fatigue: delayed failure under sustained load by subcritical crack growth
/// ("creak, crack, give way"). This is a loss of strength with time, not creep: no
/// viscous deformation is modelled.
///
/// Cracks grow at `da/dt = A K^n` below the fracture toughness (stress corrosion;
/// Charles 1958, Wiederhorn 1967). Integrating this law with `K = Y sigma sqrt(a)`
/// (Evans & Wiederhorn 1974) gives, for a strength `S` relative to its initial value,
/// `d(S^(n-2))/dt = -C sigma^n`. Normalised by the strength `f` measured in a standard
/// test lasting `t_test` (constant stress rate), with `s = sigma / f`:
///
/// * consumed life `omega` grows at `d omega/dt = (n + 1) s^n / t_test`;
/// * the residual strength factor is `(1 - omega)^(1/(n-2))`;
/// * under a constant `s` the bond fails (strength factor = `s`) after
///   `t_f = t_test (1 - s^(n-2)) / ((n + 1) s^n)` — the classical static/dynamic fatigue
///   equivalence `t_f ~ t_test / (n + 1) s^-n` (Ritter 1978) for `s` well below 1.
///
/// The same mechanism is what makes quasi-static strength rate dependent: strength
/// grows as `rate^(1/(n+1))`, so `n` follows from the low-rate exponent of the material's
/// dynamic increase factor and `t_test` from its reference strain rate
/// ([`StaticFatigue::from_dif`]). Combined with that DIF the reference-rate strength is
/// counted twice only to first order in `1/n` (about 1% for concrete).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticFatigue {
    /// Crack-growth (stress-corrosion) exponent `n` (> 2).
    pub exponent: f64,
    /// Duration of the standard strength test the material strengths refer to (s).
    pub test_time: f64,
}

impl StaticFatigue {
    /// Consistent with a DIF whose low-rate branch is `(rate/ref)^alpha`:
    /// `n = 1/alpha - 1`, and `t_test` is the time to reach `strength / E` at the
    /// reference strain rate.
    pub fn from_dif(dif: &DifParams, strength: f64, youngs_modulus: f64) -> StaticFatigue {
        StaticFatigue { exponent: 1.0 / dif.exponent - 1.0, test_time: strength / youngs_modulus / dif.reference_rate }
    }

    /// Rate at which life is consumed at stress ratio `s` (stress over test strength), 1/s.
    pub fn life_rate(&self, s: f64) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        (self.exponent + 1.0) * s.powf(self.exponent) / self.test_time
    }

    /// Residual strength factor after consuming the life fraction `omega`.
    pub fn strength_factor(&self, omega: f64) -> f64 {
        (1.0 - omega.clamp(0.0, 1.0)).powf(1.0 / (self.exponent - 2.0))
    }

    /// Time to failure under a constant stress ratio `s` in (0, 1].
    pub fn lifetime(&self, s: f64) -> f64 {
        let n = self.exponent;
        self.test_time * (1.0 - s.powf(n - 2.0)) / ((n + 1.0) * s.powf(n))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Material {
    /// kg/m^3.
    pub density: f64,
    /// Physical Young's modulus (Pa). The solver may scale it (see `SimDesc::stiffness_scale`);
    /// strengths and fracture energies are never scaled.
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    /// Tension cutoff (Pa).
    pub tensile_strength: f64,
    /// Crushing strength (Pa).
    pub compressive_strength: f64,
    /// Mohr-Coulomb cohesion: shear strength at zero normal stress (Pa).
    pub cohesion: f64,
    /// Mohr-Coulomb friction coefficient (shear strength gain per unit compression);
    /// also the Coulomb friction of a cracked joint.
    pub friction: f64,
    /// Upper limit of the Mohr-Coulomb shear strength (Pa); `None` means unlimited.
    #[serde(default)]
    pub shear_cap: Option<f64>,
    pub fracture_energy: FractureEnergy,
    #[serde(default)]
    pub kind: BondKind,
    /// Weibull modulus of the bond strengths; `None` keeps strengths deterministic.
    #[serde(default)]
    pub weibull_modulus: Option<f64>,
    #[serde(default)]
    pub dif: Option<DifParams>,
    /// Delayed failure under sustained load; `None` for materials without data.
    #[serde(default)]
    pub static_fatigue: Option<StaticFatigue>,
    /// Fraction of critical damping of the bond dashpots (light; removes ringing).
    #[serde(default = "default_damping")]
    pub damping_ratio: f64,
    /// Chunk crushing (`crush.rs`): a chunk of this material comminutes when its mean
    /// stress leaves the crush surface. `None`: its chunks never crush. Read only by the
    /// threshold mode's opt-in crush switch (rubble `DestructibleDef::crush`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crush: Option<crate::crush::CrushParams>,
}

fn default_damping() -> f64 {
    0.01
}

impl Material {
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// One-dimensional wave speed `sqrt(E/rho)` with the physical modulus.
    pub fn bar_wave_speed(&self) -> f64 {
        (self.youngs_modulus / self.density).sqrt()
    }

    pub fn validate(&self, name: &str) -> Result<(), String> {
        let positive = [
            ("density", self.density),
            ("youngs_modulus", self.youngs_modulus),
            ("tensile_strength", self.tensile_strength),
            ("compressive_strength", self.compressive_strength),
            ("fracture_energy.tension", self.fracture_energy.tension),
            ("fracture_energy.shear", self.fracture_energy.shear),
            ("fracture_energy.compression", self.fracture_energy.compression),
        ];
        for (field, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!("material '{name}': {field} must be positive, got {v}"));
            }
        }
        if !(0.0..0.5).contains(&self.poisson_ratio) {
            return Err(format!("material '{name}': poisson_ratio must be in [0, 0.5)"));
        }
        if self.cohesion < 0.0 || self.friction < 0.0 || self.damping_ratio < 0.0 {
            return Err(format!("material '{name}': cohesion, friction and damping must be >= 0"));
        }
        if let Some(m) = self.weibull_modulus {
            if !(m > 0.0) {
                return Err(format!("material '{name}': weibull_modulus must be positive"));
            }
        }
        if let Some(c) = &self.crush {
            c.validate(name)?;
        }
        Ok(())
    }

    // ---- Presets. Values are typical handbook numbers; scenes may override any field. ----

    /// Normal-strength concrete (C30).
    pub fn concrete() -> Material {
        Material {
            density: 2400.0,
            youngs_modulus: 30e9,
            poisson_ratio: 0.2,
            tensile_strength: 3.0e6,
            compressive_strength: 30e6,
            cohesion: 4.5e6,
            friction: 0.75,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 120.0, shear: 1200.0, compression: 20_000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            // fib Model Code 2010 (5.1-105/106), tensile strength: (rate / 1e-6)^0.018 up
            // to 10 /s, then 0.0062 (rate / 1e-6)^(1/3) (continuous at 10 /s).
            dif: Some(DifParams {
                reference_rate: 1e-6,
                exponent: 0.018,
                transition_rate: 10.0,
                exponent_high: 1.0 / 3.0,
                max: 6.0,
            }),
            // fib Model Code 2010 low-rate tension exponent 0.018 at 1e-6 /s:
            // n = 54.6, t_test = (3 MPa / 30 GPa) / 1e-6 /s = 100 s.
            static_fatigue: Some(StaticFatigue { exponent: 1.0 / 0.018 - 1.0, test_time: 100.0 }),
            damping_ratio: 0.01,
            crush: None,
        }
    }

    /// Fired-clay brick (bulk).
    pub fn brick() -> Material {
        Material {
            density: 1900.0,
            youngs_modulus: 5e9,
            poisson_ratio: 0.15,
            tensile_strength: 1.0e6,
            compressive_strength: 15e6,
            cohesion: 1.5e6,
            friction: 0.7,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 20.0, shear: 200.0, compression: 10_000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.01,
            crush: None,
        }
    }

    /// Brick-mortar joint (effective joint properties of masonry).
    pub fn mortar() -> Material {
        Material {
            density: 1900.0,
            youngs_modulus: 3e9,
            poisson_ratio: 0.15,
            tensile_strength: 0.3e6,
            compressive_strength: 8e6,
            cohesion: 0.4e6,
            friction: 0.75,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 10.0, shear: 100.0, compression: 5_000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.01,
            crush: None,
        }
    }

    /// Soda-lime glass.
    pub fn glass() -> Material {
        Material {
            density: 2500.0,
            youngs_modulus: 70e9,
            poisson_ratio: 0.22,
            tensile_strength: 40e6,
            compressive_strength: 1000e6,
            cohesion: 40e6,
            friction: 0.2,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 8.0, shear: 8.0, compression: 1000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            dif: None,
            // Soda-lime glass in humid air: n = 16 (Wiederhorn 1967); strength from
            // EN 1288 tests at 2 MPa/s, i.e. 20 s to the 40 MPa strength.
            static_fatigue: Some(StaticFatigue { exponent: 16.0, test_time: 20.0 }),
            damping_ratio: 0.005,
            crush: None,
        }
    }

    /// Structural softwood, strength class C24 (EN 338 mean values, loaded along the
    /// grain). The bond model is isotropic, so splitting across the grain (tension
    /// perpendicular ~0.4 MPa) is not represented; members fail at their along-grain
    /// strength. Duration of load follows the Madison curve (Wood 1951: about 9% strength
    /// per decade of load duration, n ~ 24) referred to the EN 408 short-term test
    /// (failure in about 300 s).
    pub fn timber() -> Material {
        Material {
            density: 420.0,
            youngs_modulus: 11e9,
            poisson_ratio: 0.3,
            tensile_strength: 14.5e6,
            compressive_strength: 21e6,
            cohesion: 4.0e6,
            friction: 0.5,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 300.0, shear: 1000.0, compression: 20_000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            dif: None,
            static_fatigue: Some(StaticFatigue { exponent: 24.0, test_time: 300.0 }),
            damping_ratio: 0.02,
            crush: None,
        }
    }

    /// Building stone (sandstone or limestone ashlar), typical handbook values.
    pub fn stone() -> Material {
        Material {
            density: 2300.0,
            youngs_modulus: 20e9,
            poisson_ratio: 0.2,
            tensile_strength: 3.0e6,
            compressive_strength: 60e6,
            cohesion: 4.0e6,
            friction: 0.7,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 60.0, shear: 600.0, compression: 20_000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.01,
            crush: None,
        }
    }

    /// Structural steel: ductile bond with a high fracture energy.
    pub fn steel() -> Material {
        Material {
            density: 7850.0,
            youngs_modulus: 200e9,
            poisson_ratio: 0.3,
            tensile_strength: 355e6,
            compressive_strength: 355e6,
            cohesion: 205e6,
            friction: 0.0,
            shear_cap: Some(205e6),
            fracture_energy: FractureEnergy { tension: 2.0e5, shear: 2.0e5, compression: 2.0e5 },
            kind: BondKind::Ductile,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.005,
            crush: None,
        }
    }

    // ---- Vehicles (`vehicles.rs`). A vehicle's chunks are solid boxes and prisms standing
    // for thin metal shells, so their metal is smeared over each chunk's gross volume and
    // each joint's gross face: a smeared property is the metal's times its solidity (the
    // metal's share of the volume or of the section). ----

    /// The body shell of a road vehicle, cold-formed high-strength low-alloy sheet steel
    /// (HC340LA to EN 10268: yield 340-420 MPa, tensile 410-510 MPa; E = 210 GPa,
    /// 7850 kg/m^3), smeared over the chunk at a volume solidity of 1.5%: a 1.4 t car in its
    /// 4.5 x 1.8 x 1.45 m envelope (and an 11 t bus in 12 x 2.5 x 3.2 m) is about 1.5% steel
    /// by volume. Density 7850 x 0.015 = 118 kg/m^3 (the parts' masses are the vehicle's
    /// own, `vehicles::Vehicle::masses`), E = 210 GPa x 0.015 = 3.2 GPa: the chunks' contact
    /// stiffness. Its strengths are the spot-welded seams' ([`Material::spot_weld`]).
    pub fn sheet_steel() -> Material {
        Material { density: 7850.0 * 0.015, youngs_modulus: 210e9 * 0.015, poisson_ratio: 0.3, ..Material::spot_weld() }
    }

    /// A spot-welded seam of a vehicle's body shell, smeared over the gross section a
    /// chunk face cuts through the body (steel area / gross area, the section solidity,
    /// 0.3%: about 6-7e-3 m^2 of sills, pillars, rails, floor and roof panels in the
    /// 2.3 m^2 section of a passenger car).
    ///
    /// * Shear (cohesion) 0.375 MPa: a resistance spot weld in 1.0-1.5 mm HSLA sheet
    ///   (nugget 5 sqrt(t), about 5-6 mm) holds about 6 kN in tensile shear (the minimum
    ///   tensile-shear loads of AWS D8.9M and ISO 18278-2 tests are of order 5-7 kN for
    ///   this sheet), at a 40 mm pitch: 150 kN per metre of seam, or 6 kN / (0.04 m x
    ///   1.2 mm) = 125 MPa over the sheet it joins; x 0.003 = 0.375 MPa.
    /// * Tension 0.15 MPa: the cross-tension strength of a spot weld is 0.3-0.5 of its
    ///   tensile-shear strength (ISO 14272 against ISO 14273 tests): 0.4 x 125 MPa x
    ///   0.003.
    /// * Compression 0.37 MPa: the seam does not govern; the shell crushes by progressive
    ///   folding at its mean crushing stress, about a third of its squash load for thin
    ///   steel sections (Wierzbicki & Abramowicz 1983): 355 MPa x 0.35 x 0.003. As a check,
    ///   a 1.4 t car stopping from 56 km/h over 0.6 m of front crush (Euro NCAP's offset
    ///   test) averages 280 kN over about 1 m^2 of front section, 0.28 MPa.
    /// * E = 210 GPa x 0.003 = 0.63 GPa; a ductile bond (steel yields, then tears), its
    ///   fracture energy steel's 200 kJ/m^2 (`Material::steel`) x 0.003 = 600 J/m^2.
    pub fn spot_weld() -> Material {
        Material {
            density: 7850.0 * 0.015,
            youngs_modulus: 210e9 * 0.003,
            poisson_ratio: 0.3,
            tensile_strength: 0.15e6,
            compressive_strength: 0.37e6,
            cohesion: 0.375e6,
            friction: 0.3,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 600.0, shear: 600.0, compression: 600.0 },
            kind: BondKind::Ductile,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.02,
            crush: None,
        }
    }

    /// A bolted or hinged attachment of a vehicle (a door on its hinges and latch, a wheel
    /// on its suspension, a bonnet, a bumper, a bolted wing, an engine on its mounts),
    /// smeared over the face it is bonded on, weaker than the welded shell: doors and
    /// wheels come off first.
    ///
    /// * Tension 0.14 MPa, shear 0.11 MPa: FMVSS 206 (49 CFR 571.206) requires a side
    ///   door's hinge system to hold 11,000 N longitudinally and 8,900 N transversely, and
    ///   its primary latch the same, each over one of the 0.08 m^2 edge faces a door is
    ///   bonded on here (`vehicles::car`): 11,000 / 0.08 = 0.14 MPa, 8,900 / 0.08 =
    ///   0.11 MPa. A wheel on its suspension takes the same per face (a car's 0.17 m^2 hub
    ///   face holds about 24 kN, seven times its static load).
    /// * Compression 0.3 MPa: a bolted face in compression bears on the shell behind it, so
    ///   about the shell's crushing stress (`Material::spot_weld`).
    /// * E 0.63 GPa (the shell's smeared modulus), ductile, 300 J/m^2.
    pub fn bolted() -> Material {
        Material { tensile_strength: 0.14e6, compressive_strength: 0.3e6, cohesion: 0.11e6, fracture_energy: FractureEnergy { tension: 300.0, shear: 300.0, compression: 300.0 }, ..Material::spot_weld() }
    }

    /// An airliner's airframe, aluminium alloy 2024-T3 sheet (MMPDS-01 Table 3.2.3.0(b),
    /// A-basis, L direction: F_tu 483 MPa, F_ty 345 MPa, F_cy 269 MPa, F_su 283 MPa;
    /// E 72.4 GPa, 2780 kg/m^3), smeared over the chunk at a volume solidity of 4%: a
    /// regional jet of about 37 t in the 300-350 m^3 of its fuselage barrel and wings
    /// (Embraer E175 class: 31.7 m long, 3.0 m fuselage, 26 m span, MTOW 37.5-40 t) is
    /// about 4% aluminium by volume, structure, systems and fuel together. Density
    /// 2780 x 0.04 = 111 kg/m^3 (the parts' masses are the aircraft's own), E = 72.4 GPa x
    /// 0.04 = 2.9 GPa. Its strengths are the riveted joints' ([`Material::riveted`]).
    pub fn aluminium() -> Material {
        Material { density: 2780.0 * 0.04, youngs_modulus: 72.4e9 * 0.04, ..Material::riveted() }
    }

    /// A riveted joint of an airframe's skin, stringers and frames (2024-T3,
    /// [`Material::aluminium`]), smeared over the gross section a chunk face cuts through
    /// the airframe at a section solidity of 0.4% (a 3 m barrel's 1.6-2 mm skin with its
    /// stringers and frames is about 4 mm of aluminium round its 9.4 m perimeter:
    /// 0.04 m^2 in its 7 m^2 section), at a riveted lap joint's efficiency of 0.7 (the
    /// joint's strength over the sheet's; Niu, *Airframe Structural Design*, 1988, ch. 7):
    ///
    /// * tension 483 MPa x 0.7 x 0.004 = 1.35 MPa; shear 283 MPa x 0.7 x 0.004 = 0.79 MPa;
    /// * compression: the thin shell crushes, 269 MPa x 0.35 x 0.004 = 0.38 MPa (as
    ///   `Material::spot_weld`);
    /// * E = 72.4 GPa x 0.004 = 0.29 GPa; ductile; 2024-T3's tearing energy of about
    ///   100 kJ/m^2 x 0.004 = 400 J/m^2.
    pub fn riveted() -> Material {
        Material {
            density: 2780.0 * 0.04,
            youngs_modulus: 72.4e9 * 0.004,
            poisson_ratio: 0.33,
            tensile_strength: 1.35e6,
            compressive_strength: 0.38e6,
            cohesion: 0.79e6,
            friction: 0.3,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 400.0, shear: 400.0, compression: 400.0 },
            kind: BondKind::Ductile,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.02,
            crush: None,
        }
    }

    /// A deterministic, rate-independent linear-elastic brittle material for analytic tests.
    pub fn analytic_test() -> Material {
        Material {
            density: 2000.0,
            youngs_modulus: 1e9,
            poisson_ratio: 0.25,
            tensile_strength: 1e6,
            compressive_strength: 10e6,
            cohesion: 1.5e6,
            friction: 0.6,
            shear_cap: None,
            fracture_energy: FractureEnergy { tension: 100.0, shear: 400.0, compression: 5000.0 },
            kind: BondKind::Brittle,
            weibull_modulus: None,
            dif: None,
            static_fatigue: None,
            damping_ratio: 0.0,
            crush: None,
        }
    }
}

/// Look up a preset by name.
pub fn preset(name: &str) -> Option<Material> {
    Some(match name {
        "concrete" => Material::concrete(),
        "brick" => Material::brick(),
        "mortar" => Material::mortar(),
        "glass" => Material::glass(),
        "steel" => Material::steel(),
        "analytic_test" => Material::analytic_test(),
        _ => return None,
    })
}

/// Gamma function (Lanczos, g = 7), accurate to ~1e-15 for positive arguments.
pub fn gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// Strength multiplier of bond `bond` of body `body` for a scene seed: Weibull with unit
/// mean, or 1 for deterministic materials. Exporters receive the same value in the
/// bond's `derived.weibull`.
pub fn bond_strength_factor(modulus: Option<f64>, seed: u64, body: usize, bond: usize) -> f64 {
    match modulus {
        Some(m) => weibull_unit_mean(m, crate::rng::uniform(seed, &[body as u64, bond as u64, 0x5745])),
        None => 1.0,
    }
}

/// Weibull strength multiplier with unit mean, from a uniform sample `u` in (0, 1).
pub fn weibull_unit_mean(modulus: f64, u: f64) -> f64 {
    let u = u.clamp(1e-300, 1.0 - 1e-16);
    (-(1.0 - u).ln()).powf(1.0 / modulus) / gamma(1.0 + 1.0 / modulus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_known_values() {
        assert!((gamma(1.0) - 1.0).abs() < 1e-13);
        assert!((gamma(5.0) - 24.0).abs() < 1e-11);
        assert!((gamma(0.5) - std::f64::consts::PI.sqrt()).abs() < 1e-13);
    }

    #[test]
    fn dif_is_continuous_and_capped() {
        let d = Material::concrete().dif.unwrap();
        assert_eq!(d.factor(1e-7), 1.0);
        let below = d.factor(d.transition_rate * 0.999_999);
        let above = d.factor(d.transition_rate * 1.000_001);
        assert!((below - above).abs() < 1e-4);
        assert!(d.factor(1e9) <= d.max);
    }

    #[test]
    fn static_fatigue_lifetime_matches_the_integrated_law() {
        let f = StaticFatigue { exponent: 20.0, test_time: 10.0 };
        for s in [0.7, 0.9, 0.97] {
            // Integrate the consumed life until the residual strength drops to s.
            let (mut omega, mut t) = (0.0, 0.0);
            let dt = f.lifetime(s) * 1e-5;
            while f.strength_factor(omega) > s {
                omega += f.life_rate(s) * dt;
                t += dt;
            }
            assert!((t / f.lifetime(s) - 1.0).abs() < 1e-3, "s {s}: {t} vs {}", f.lifetime(s));
        }
        // Consistency with the fib tension DIF of concrete.
        let c = Material::concrete();
        let from = StaticFatigue::from_dif(&c.dif.unwrap(), c.tensile_strength, c.youngs_modulus);
        let preset = c.static_fatigue.unwrap();
        assert!((from.exponent - preset.exponent).abs() < 1e-9 && (from.test_time - preset.test_time).abs() < 1e-9);
    }

    #[test]
    fn weibull_mean_is_one() {
        let n = 200_000;
        let mean: f64 = (0..n).map(|i| weibull_unit_mean(8.0, (i as f64 + 0.5) / n as f64)).sum::<f64>() / n as f64;
        assert!((mean - 1.0).abs() < 1e-3, "mean {mean}");
    }
}
