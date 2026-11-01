//! Chunk crushing (comminution): when a chunk's own material is overwhelmed it stops being
//! a rigid chunk (engine/docs/destruction-reactions.md, section 1).
//!
//! The law is the one PhysX Blast's stress extension ships (`extStressCrushStep`,
//! `blast/blast-stress-solver/docs/chunk-crushing.md`), written fresh here:
//!
//! ```text
//! sigma = (1/V) sum sym(branch (x) force)      contact and bond tractions, no body forces
//! p = -tr(sigma)/3 (compression positive)      q = sqrt(1.5 s:s), s = sigma + p I
//! crushing needs p > 0
//! excess = max(q - (cohesion + frictionSlope p), p - capPressure)
//! D += excess^2 dt / (crushViscosity crushEnergy)       crushed at D >= 1
//! ```
//!
//! A Drucker-Prager cone with a pressure cap, tension excluded (a chunk in net tension
//! cracks, which the bonds represent), Perzyna overstress viscoplasticity with the plastic
//! work normalised by the specific comminution energy. Quadratic in overstress, nothing
//! accumulates below yield. Strain-rate hardening (the reference's `strainRateExponent`)
//! is not modelled (v1).
//!
//! [`crush_step_f32`] is the single-precision mirror of `stress-gpu`'s `crush.slang`
//! `crush_step`, operation for operation (the GPU parity test compares them).

use serde::{Deserialize, Serialize};

/// A material's crush parameters (ScenePack v3 `materials[].crush`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrushParams {
    /// Hydrostatic cap (Pa): confined pore collapse. > 0.
    pub cap_pressure: f64,
    /// Drucker-Prager deviatoric intercept at p = 0 (Pa). >= 0.
    pub cohesion: f64,
    /// dq/dp of the cone. >= 0.
    pub friction_slope: f64,
    /// Specific comminution energy (J/m^3). > 0.
    pub crush_energy: f64,
    /// Perzyna flow viscosity (Pa s). > 0.
    pub crush_viscosity: f64,
}

impl CrushParams {
    /// The cone pinned to a material's unconfined compressive strength `fc` (Pa): under
    /// uniaxial compression at `fc` the state is `p = fc / 3, q = fc`, so a cohesion of
    /// `fc (1 - k / 3)` yields there exactly; the cap at `2.5 fc`, where confined pore
    /// collapse begins (vibe-land `materials.mjs` `crushOf`, as the PhysX reference
    /// building's exporter derives it). `k` = 1.2: about 30 degrees of internal friction.
    pub fn from_compressive_strength(fc: f64, crush_energy: f64, crush_viscosity: f64, k: f64) -> CrushParams {
        CrushParams { cap_pressure: 2.5 * fc, cohesion: fc * (1.0 - k / 3.0), friction_slope: k, crush_energy, crush_viscosity }
    }

    pub fn validate(&self, name: &str) -> Result<(), String> {
        for (field, v) in [("cap_pressure", self.cap_pressure), ("crush_energy", self.crush_energy), ("crush_viscosity", self.crush_viscosity)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!("material '{name}': crush.{field} must be positive, got {v}"));
            }
        }
        for (field, v) in [("cohesion", self.cohesion), ("friction_slope", self.friction_slope)] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("material '{name}': crush.{field} must be >= 0, got {v}"));
            }
        }
        Ok(())
    }

    /// `max(q / (cohesion + frictionSlope p), p / capPressure)`: 1 at yield (the crush
    /// analogue of a joint's utilisation); 0 in tension.
    pub fn utilisation(&self, p: f64, q: f64) -> f64 {
        if !(p > 0.0) {
            return 0.0;
        }
        let cone = self.cohesion + self.friction_slope * p;
        let cone_use = if cone > 0.0 { q / cone } else { 0.0 };
        cone_use.max(p / self.cap_pressure)
    }

    /// The overstress (Pa), `None` where nothing accumulates (tension, inside the
    /// surface).
    pub fn excess(&self, p: f64, q: f64) -> Option<f64> {
        if !(p > 0.0) {
            return None;
        }
        let excess = (q - (self.cohesion + self.friction_slope * p)).max(p - self.cap_pressure);
        (excess > 0.0).then_some(excess)
    }
}

/// A chunk's crush state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrushState {
    /// [0, 1].
    pub damage: f64,
    pub crushed: bool,
}

/// The stress invariants of a virial `[xx, yy, zz, xy, xz, yz]` (N m, the symmetric sum
/// of branch (x) force) over a volume (m^3): `p` (Pa, compression positive) and `q` (Pa).
pub fn invariants(virial: [f64; 6], volume: f64) -> (f64, f64) {
    let s = virial.map(|v| v / volume);
    let p = -(s[0] + s[1] + s[2]) / 3.0;
    let (dx, dy, dz) = (s[0] + p, s[1] + p, s[2] + p);
    let ss = dx * dx + dy * dy + dz * dz + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    (p, (1.5 * ss.max(0.0)).sqrt())
}

/// The symmetric part of `branch (x) force`, as `[xx, yy, zz, xy, xz, yz]`.
pub fn sym_outer(r: [f64; 3], f: [f64; 3]) -> [f64; 6] {
    [r[0] * f[0], r[1] * f[1], r[2] * f[2], 0.5 * (r[0] * f[1] + r[1] * f[0]), 0.5 * (r[0] * f[2] + r[2] * f[0]), 0.5 * (r[1] * f[2] + r[2] * f[1])]
}

/// One step of the law over `dt` under the stress of `virial` over `volume`. Returns the
/// new state and the stress invariants.
pub fn crush_step(virial: [f64; 6], volume: f64, dt: f64, m: &CrushParams, state: CrushState) -> (CrushState, f64, f64) {
    let (p, q) = invariants(virial, volume);
    let mut out = state;
    if state.crushed || !(volume > 0.0) || !p.is_finite() || !q.is_finite() {
        return (out, p, q);
    }
    if let Some(excess) = m.excess(p, q) {
        out.damage += excess * excess * dt / (m.crush_viscosity * m.crush_energy);
        if out.damage >= 1.0 {
            out.damage = 1.0;
            out.crushed = true;
        }
    }
    (out, p, q)
}

/// [`crush_step`] in f32, as `crush.slang` `crush_step` computes it (the GPU parity
/// test's reference): `(damage, p, q, crushed)`. `m` is (cap, cohesion, slope, energy),
/// (viscosity).
pub fn crush_step_f32(virial: [f32; 6], volume: f32, dt: f32, m: [f32; 5], damage: f32) -> (f32, f32, f32, bool) {
    let inv = 1.0 / volume;
    let (xx, yy, zz) = (virial[0] * inv, virial[1] * inv, virial[2] * inv);
    let (xy, xz, yz) = (virial[3] * inv, virial[4] * inv, virial[5] * inv);
    let p = -(xx + yy + zz) / 3.0;
    let (dx, dy, dz) = (xx + p, yy + p, zz + p);
    let ss = dx * dx + dy * dy + dz * dz + 2.0 * (xy * xy + xz * xz + yz * yz);
    let q = (1.5 * ss.max(0.0)).sqrt();
    if !(p > 0.0) {
        return (damage, p, q, false);
    }
    let excess = (q - (m[1] + m[2] * p)).max(p - m[0]);
    if !(excess > 0.0) {
        return (damage, p, q, false);
    }
    let d = damage + excess * excess * dt / (m[4] * m[3]);
    if d >= 1.0 {
        (1.0, p, q, true)
    } else {
        (d, p, q, false)
    }
}

/// The crush impulse an impactor pays for `energy` (J) out of its kinetic energy:
/// `J = M (v - sqrt(v^2 - 2 dE / M))`, capped at `M v` (it can stop a striker, never
/// bounce it). `speed` is its closing speed (m/s), `mass` its mass (kg).
pub fn crush_impulse(mass: f64, speed: f64, energy: f64) -> f64 {
    let v2 = speed * speed - 2.0 * energy / mass;
    mass * (speed - v2.max(0.0).sqrt())
}

/// Cited crush presets, derived from each material's compressive strength (vibe-land
/// `structures/town-kit/src/materials.mjs` `CRUSH`, whose citations these copy).
///
/// crushEnergy is the specific comminution energy, Bond's law `W = 10 Wi (1/sqrt(P80) -
/// 1/sqrt(F80))` kWh/t with Bond's (1961) work indices ([`bond_energy`]); crushViscosity
/// the overstress the CEB-FIP Model Code 1990 (2.1.6.4) dynamic increase factor gives at a
/// 30/s strain rate, over that rate: `(DIF - 1) fc / 30` ([`mc90_viscosity`]).
pub mod presets {
    use super::CrushParams;

    /// `k`: the cone's slope (about 30 degrees of internal friction).
    const K: f64 = 1.2;

    /// Bond (1961): specific comminution energy (J/m^3) from F80 `from` to P80 `to` (m) at
    /// work index `wi` (kWh/t) and density `rho` (kg/m^3).
    pub fn bond_energy(wi: f64, from: f64, to: f64, rho: f64) -> f64 {
        10.0 * wi * (1.0 / (to * 1e6).sqrt() - 1.0 / (from * 1e6).sqrt()) * 3.6e6 / 1000.0 * rho
    }

    /// CEB-FIP MC90 (2.1.6.4) compressive DIF at 30/s for strength `fc` (Pa), as the
    /// viscosity `(DIF - 1) fc / 30` (Pa s).
    pub fn mc90_viscosity(fc: f64) -> f64 {
        let a = 1.0 / (5.0 + 9.0 * fc / 10e6);
        ((30.0f64 / 30e-6).powf(1.026 * a) - 1.0) * fc / 30.0
    }

    /// Clay-brick masonry: f_k = 0.55 f_b^0.7 f_m^0.3 (EN 1996-1-1 eq. 3.1, Group 1 clay
    /// units, general-purpose mortar) = 6.8 MPa for common facing brick (f_b 20 MPa, EN
    /// 771-1 / AS/NZS 4455) in M4 mortar. Brick to 20 mm rubble from 100 mm at Wi 13 kWh/t
    /// (fired clay; cement clinker 13.5): 0.51 kWh/t = 1.8 kJ/kg x 1900 kg/m^3 = 3.5 MJ/m^3.
    /// DIF 3.6 -> 5.9e5 Pa s.
    pub fn brick_masonry() -> CrushParams {
        CrushParams::from_compressive_strength(6.8e6, 3.5e6, 5.9e5, K)
    }

    /// Gypsum board: the core crushes at ~3.5 MPa. To 5 mm from its 13 mm at Wi 8.2
    /// (gypsum rock 8.16): 0.44 kWh/t = 1.6 kJ/kg x 700 kg/m^3 = 1.1 MJ/m^3. DIF 5.4 ->
    /// 5.1e5 Pa s.
    pub fn gypsum() -> CrushParams {
        CrushParams::from_compressive_strength(3.5e6, 1.1e6, 5.1e5, K)
    }

    /// Concrete C30/37: f_ck 30 MPa (EN 1992-1-1 Table 3.1). Rubble to 20 mm from 100 mm
    /// at Wi 11.6 kWh/t (limestone aggregate, Bond 1961): 0.45 kWh/t = 1.6 kJ/kg x 2400 =
    /// 3.9 MJ/m^3. MC90 DIF at 30/s 1.56 -> 5.6e5 Pa s. Reinforced concrete crushes its
    /// concrete only; its bonds still have to snap.
    pub fn concrete() -> CrushParams {
        CrushParams::from_compressive_strength(30e6, 3.9e6, 5.6e5, K)
    }

    /// Annealed float glass fails by tensile cracking under contact (Hertzian cones), at
    /// its characteristic bending strength f_g,k = 45 MPa (EN 572-1), pressure-independent
    /// (slope 0). To 1 mm from a 6 mm pane at Wi 3.08 kWh/t (glass, Bond 1961): 0.58 kWh/t
    /// = 2.1 kJ/kg x 2500 = 5.2 MJ/m^3. Taken as rate-insensitive (an assumption, not a
    /// measurement): 3.2e3 Pa s.
    pub fn glass() -> CrushParams {
        CrushParams { cap_pressure: 45e6, cohesion: 45e6, friction_slope: 0.0, crush_energy: 5.2e6, crush_viscosity: 3.2e3 }
    }

    /// Natural-stone masonry: f_k = K f_b^0.7 f_m^0.3 (EN 1996-1-1 eq. 3.1, Table 3.3
    /// natural stone K 0.45) = 10.5 MPa for units of f_b 50 MPa (building limestone and
    /// sandstone, BS EN 1926) in M4 mortar. Rubble to 20 mm from 100 mm at Wi 11.6 kWh/t
    /// (limestone, Bond 1961) at 2600 kg/m^3; MC90 viscosity.
    pub fn stone_masonry() -> CrushParams {
        CrushParams::from_compressive_strength(10.5e6, bond_energy(11.6, 0.1, 0.02, 2600.0), mc90_viscosity(10.5e6), K)
    }

    /// Structural softwood (C24, EN 338): crushing along the grain at f_c,0,k 21 MPa.
    /// Energy: the crush plateau to densification, f_c,0 (1 - rho / rho_cell) = 21 MPa x
    /// (1 - 420 / 1500) = 15 MJ/m^3 (Reid & Peng 1997, Int. J. Impact Eng. 19(5-6); cell
    /// wall ~1500 kg/m^3, Wood Handbook FPL-GTR-282 ch. 4). Rate: ~10% per tenfold loading
    /// rate (Wood Handbook ch. 5), 30/s against a 1e-5/s test: 6.5 decades, DIF 1.86.
    pub fn softwood() -> CrushParams {
        CrushParams::from_compressive_strength(21e6, 21e6 * (1.0 - 420.0 / 1500.0), (1.1f64.powf(6.5) - 1.0) * 21e6 / 30.0, K)
    }

    /// Concrete roof tiles (EN 490/491) as a 50 mm layer of 920 kg/m^3: a tile breaks in
    /// flexure at f_ctm,fl = (1.6 - h/1000) f_ctm = 5.6 MPa (EN 1992-1-1 3.1.8, C40/50),
    /// on the layer's smeared stress 920/2300 of that: 2.2 MPa, pressure-independent.
    /// Pieces to 5 mm from 12 mm at Wi 11.6; MC90 tensile DIF at 30/s 1.53.
    pub fn roof_tile() -> CrushParams {
        CrushParams { cap_pressure: 2.2e6, cohesion: 2.2e6, friction_slope: 0.0, crush_energy: bond_energy(11.6, 0.012, 0.005, 920.0), crush_viscosity: 0.53 * 2.2e6 / 30.0 }
    }

    /// Roofing slate (EN 12326): modulus of rupture >= 35 MPa along the grain on a 2100
    /// kg/m^3 layer of 2800 kg/m^3 slate: 26 MPa. Pieces to 5 mm from 6 mm at Wi 13.8
    /// kWh/t (Bond 1961). Rate-insensitive as glass (an assumption).
    pub fn slate() -> CrushParams {
        CrushParams { cap_pressure: 26e6, cohesion: 26e6, friction_slope: 0.0, crush_energy: bond_energy(13.8, 0.006, 0.005, 2100.0), crush_viscosity: 3.2e3 }
    }

    /// A preset by name.
    pub fn preset(name: &str) -> Option<CrushParams> {
        Some(match name {
            "brick_masonry" => brick_masonry(),
            "gypsum" => gypsum(),
            "concrete" => concrete(),
            "glass" => glass(),
            "stone_masonry" => stone_masonry(),
            "softwood" => softwood(),
            "roof_tile" => roof_tile(),
            "slate" => slate(),
            _ => return None,
        })
    }

    /// The preset for a calibrated stress-ref material by its name (`material.rs`
    /// presets: concrete, brick, stone, glass, timber), `None` for those that do not
    /// comminute (steel is ductile: a struck member yields and bends; mortar is a joint).
    pub fn for_material(name: &str) -> Option<CrushParams> {
        Some(match name {
            "concrete" => concrete(),
            "brick" => brick_masonry(),
            "stone" => stone_masonry(),
            "glass" => glass(),
            "timber" => softwood(),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn concrete() -> CrushParams {
        presets::concrete()
    }

    /// Uniaxial compression `s` (Pa) along z of a unit cube: two opposite face loads.
    fn uniaxial(s: f64) -> [f64; 6] {
        // Faces at z = +-0.5 pushed inwards with force s (area 1).
        let a = sym_outer([0.0, 0.0, 0.5], [0.0, 0.0, -s]);
        let b = sym_outer([0.0, 0.0, -0.5], [0.0, 0.0, s]);
        std::array::from_fn(|i| a[i] + b[i])
    }

    #[test]
    fn uniaxial_compression_yields_at_the_compressive_strength() {
        let m = concrete();
        let fc = 30e6;
        let (p, q) = invariants(uniaxial(fc), 1.0);
        assert!((p - fc / 3.0).abs() < 1e-3 && (q - fc).abs() < 1e-3, "p {p} q {q}");
        assert!((m.utilisation(p, q) - 1.0).abs() < 1e-12);
        let (below, ..) = crush_step(uniaxial(0.99 * fc), 1.0, 1.0, &m, CrushState::default());
        assert_eq!(below.damage, 0.0, "nothing accumulates below yield");
    }

    #[test]
    fn tension_never_crushes() {
        let m = concrete();
        // Pulling the faces apart, far past any strength.
        let v = uniaxial(-1e12);
        let (p, _) = invariants(v, 1.0);
        assert!(p < 0.0);
        let (s, ..) = crush_step(v, 1.0, 1e3, &m, CrushState::default());
        assert_eq!(s, CrushState::default());
    }

    #[test]
    fn damage_is_quadratic_in_overstress() {
        let m = concrete();
        let fc = 30e6;
        // q - cone at uniaxial s: s - (cohesion + k s/3) = (s - fc)(1 - k/3).
        let step = |over: f64| crush_step(uniaxial(fc + over), 1.0, 1e-6, &m, CrushState::default()).0.damage;
        let (d1, d2) = (step(1e6), step(2e6));
        assert!(d1 > 0.0);
        assert!((d2 / d1 - 4.0).abs() < 1e-9, "{d1} {d2}");
        let excess = 1e6 * (1.0 - 1.2 / 3.0);
        let expected = excess * excess * 1e-6 / (m.crush_viscosity * m.crush_energy);
        assert!((d1 / expected - 1.0).abs() < 1e-9);
    }

    #[test]
    fn crushed_at_one_and_then_stays() {
        let m = concrete();
        let (s, ..) = crush_step(uniaxial(1e10), 1.0, 1.0, &m, CrushState::default());
        assert!(s.crushed && s.damage == 1.0);
        let (again, ..) = crush_step(uniaxial(1e10), 1.0, 1.0, &m, s);
        assert_eq!(again, s);
    }

    #[test]
    fn cap_crushes_under_confinement() {
        let m = concrete();
        // Hydrostatic compression: q = 0, only the cap.
        let p = 3.0 * m.cap_pressure;
        let v = [-p, -p, -p, 0.0, 0.0, 0.0];
        let (pp, q) = invariants(v, 1.0);
        assert!((pp - p).abs() < 1e-3 && q.abs() < 1e-3);
        assert_eq!(m.excess(pp, q), Some(pp - m.cap_pressure));
    }

    #[test]
    fn f32_mirror_matches_f64() {
        let m = concrete();
        let mf = [m.cap_pressure as f32, m.cohesion as f32, m.friction_slope as f32, m.crush_energy as f32, m.crush_viscosity as f32];
        for s in [10e6, 40e6, 90e6] {
            let v = uniaxial(s);
            let (st, p, q) = crush_step(v, 0.5, 1e-3, &m, CrushState::default());
            let (d, pf, qf, c) = crush_step_f32(v.map(|x| x as f32), 0.5, 1e-3, mf, 0.0);
            assert!(((pf as f64) - p).abs() <= 1e-5 * p.abs().max(1.0) && ((qf as f64) - q).abs() <= 1e-5 * q.abs().max(1.0));
            assert!(((d as f64) - st.damage).abs() <= 1e-5 * st.damage.max(1e-30) && c == st.crushed);
        }
    }

    #[test]
    fn impulse_takes_the_energy_and_never_bounces() {
        let (m, v) = (100.0, 30.0);
        let e = 1000.0;
        let j = crush_impulse(m, v, e);
        let after = v - j / m;
        assert!((0.5 * m * (v * v - after * after) - e).abs() < 1e-6);
        assert_eq!(crush_impulse(m, v, 1e9), m * v);
    }

    #[test]
    fn presets_are_valid_and_derived_from_fc() {
        for name in ["brick_masonry", "gypsum", "concrete", "glass", "stone_masonry", "softwood", "roof_tile", "slate"] {
            presets::preset(name).unwrap().validate(name).unwrap();
        }
        let c = presets::concrete();
        assert!((c.cohesion - 30e6 * 0.6).abs() < 1.0 && c.cap_pressure == 75e6);
        // Bond's law, vibe-land's stone: 0.45 kWh/t x 2600 kg/m^3.
        let e = presets::bond_energy(11.6, 0.1, 0.02, 2600.0);
        assert!((e / 4.2e6 - 1.0).abs() < 0.02, "{e}");
        // MC90 at 30 MPa: DIF ~1.56 -> ~5.6e5 Pa s.
        assert!((presets::mc90_viscosity(30e6) / 5.6e5 - 1.0).abs() < 0.03, "{}", presets::mc90_viscosity(30e6));
    }
}
