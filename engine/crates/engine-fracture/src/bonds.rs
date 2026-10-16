//! Baked bonds as breakable AVBD welds: break capacities and elastic springs
//! from Shardwright's reference models, mapped to solver penalties.

use crate::asset::{Bond, FractureAsset};
use crate::scene::{Body, Joint};
use serde_json::Value;

/// The tick bond springs are scaled for (the lab runs at 120 Hz).
pub const BOND_TICK: f64 = 1.0 / 120.0;
/// AVBD penalty for bonds: one scene-wide factor maps the physical springs to
/// solver stiffness, so rows keep the bake's relative stiffness (and so their
/// share of load in redundant structures); the multipliers carry the load. The
/// median bond's normal row sits at BOND_PENALTY_SCALE x the median lighter
/// body's m/h^2. At steady state a bond acts like a spring of iterations x k:
/// 0.5 holds statics to ~0.03 mm / 0.1 deg under 125 kg (bond-regression.mjs).
/// Rows stiffer than BOND_PENALTY_CAP x their lighter body's inertial term
/// (m/h^2, smallest I/h^2) are capped: past ~20x a bonded cluster locks in
/// Gauss-Seidel (a released pair hangs in the air) and f32 6x6 solves lose
/// precision; the few capped rows are still the stiffest in the structure.
pub const BOND_PENALTY_SCALE: f64 = 0.5;
pub const BOND_PENALTY_CAP: f64 = 5.0;

fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

/// The fragment's first physical material (an empty record when missing).
fn fragment_material(asset: &FractureAsset, f: u32) -> &Value {
    let id = asset.physics.fragments[f as usize].material_ids.first().copied().unwrap_or(0);
    asset.material(id)
}

fn interface<'a>(asset: &'a FractureAsset, bond: &Bond) -> Option<&'a Value> {
    let id = bond.composition.first().map_or(-1, |c| c.interface_material);
    if id >= 0 {
        asset.interface_material(id)
    } else {
        None
    }
}

/// Break capacities of a baked bond, as Shardwright's reference stress solver
/// computes them (tools/sim): ft is the bond's interface-material tensile
/// strength, else the weaker fragment material's; fc the weaker compressive
/// strength; k the baked strength scale. Bearings (ft = 0) resist with 0.05 fc.
/// Returns [tension N, compression N, shear N, bending N m].
pub fn bond_capacities(asset: &FractureAsset, bond: &Bond, strength_scale: f64) -> [f64; 4] {
    let strengths = |f: u32| {
        let m = fragment_material(asset, f);
        let ft = num(m, "tensile_strength").unwrap_or(1e6);
        (ft, num(m, "compressive_strength").unwrap_or(10.0 * ft))
    };
    let (fta, fca) = strengths(bond.a);
    let (ftb, fcb) = if bond.b >= 0 { strengths(bond.b as u32) } else { (fta, fca) };
    let ft = interface(asset, bond).and_then(|m| num(m, "tensile_strength")).unwrap_or(fta.min(ftb));
    let fc = fca.min(fcb);
    let k = bond.strength_scale as f64 * strength_scale;
    let c = (bond.extent_half[0] as f64).max(bond.extent_half[1] as f64).max(1e-6);
    let fte = if ft > 0.0 { ft } else { 0.05 * fc };
    let area = bond.area as f64;
    [
        (fte * area * k).max(1e-6),
        (fc * area * k).max(1e-6),
        (1.5 * fte * area * k).max(1e-6),
        (fte * (bond.i_uu as f64).min(bond.i_vv as f64) / c * k).max(1e-9),
    ]
}

/// Elastic bond springs, Shardwright's reference model (frac-validate
/// network.rs, StiffnessModel::Spec): series compliances along the normal from
/// each fragment's centroid (distA, distB) plus the interface layer,
///   ce = dA/E_A + dB/E_B + t/E_i,  cg = dA/G_A + dB/G_B + t/G_i,
///   kn = A/ce, ks = A/cg, ktu = Iuu/ce, ktv = Ivv/ce, kt = J/cg.
/// World bonds (b = -1) have no B side. Returns [kn, ks, ks, ktu, ktv, kt] in
/// the bond frame (normal, u, v | bending about u, v, torsion about normal).
pub fn bond_springs(asset: &FractureAsset, bond: &Bond) -> [f64; 6] {
    let elastic = |f: u32| {
        let m = fragment_material(asset, f);
        // Anisotropic (wood): longitudinal E and shear G, as frac-material elastic().
        if let Some(a) = m.get("anisotropy").filter(|a| !a.is_null()) {
            let e = num(a, "E_long").or_else(|| num(a, "e_long")).unwrap_or(f64::NAN);
            let g = num(a, "G").or_else(|| num(a, "g")).unwrap_or(f64::NAN);
            return (e, g);
        }
        if let Some(e) = num(m, "youngs_modulus") {
            return (e, e / (2.0 * (1.0 + num(m, "poisson").unwrap_or(0.2))));
        }
        (1e9, 1e9 / 2.4)
    };
    let (ea, ga) = elastic(bond.a);
    let (eb, gb) = if bond.b >= 0 { elastic(bond.b as u32) } else { (f64::INFINITY, f64::INFINITY) };
    let (em, gm, tm) = match interface(asset, bond) {
        Some(im) => {
            let em = num(im, "youngs_modulus").unwrap_or(ea);
            (em, num(im, "shear_modulus").unwrap_or(em / 2.4), num(im, "thickness").unwrap_or(0.0))
        }
        None => (1.0, 1.0, 0.0),
    };
    let da = (bond.dist_a as f64).max(1e-9);
    let db = if bond.b >= 0 { (bond.dist_b as f64).max(1e-9) } else { 0.0 };
    let side = |d: f64, x: f64| if x.is_finite() { d / x } else { 0.0 };
    let layer = |x: f64| if tm > 0.0 { tm / x } else { 0.0 };
    let ce = da / ea + side(db, eb) + layer(em);
    let cg = da / ga + side(db, gb) + layer(gm);
    let area = bond.area as f64;
    let ks = area / cg;
    [area / ce, ks, ks, bond.i_uu as f64 / ce, bond.i_vv as f64 / ce, bond.polar_moment as f64 / cg]
}

/// Scale physical springs to solver penalties (see `BOND_PENALTY_SCALE`).
pub fn scale_bond_springs(joints: &[Joint], bodies: &[Body], dt: f64) -> Vec<Joint> {
    scale_bond_springs_with(joints, bodies, dt, BOND_PENALTY_SCALE, BOND_PENALTY_CAP)
}

pub fn scale_bond_springs_with(joints: &[Joint], bodies: &[Body], dt: f64, scale: f64, cap: f64) -> Vec<Joint> {
    if joints.is_empty() {
        return vec![];
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() >> 1]
    };
    let inertial = |i: u32| match bodies.get(i as usize) {
        Some(b) if b.inv_mass > 0.0 => (1.0 / b.inv_mass, 1.0 / b.inv_i.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        _ => (f64::INFINITY, f64::INFINITY),
    };
    let lighter = |j: &Joint| {
        let (a, b) = (inertial(j.body_a), inertial(j.body_b));
        (a.0.min(b.0), a.1.min(b.1))
    };
    let s = scale * median(joints.iter().map(|j| lighter(j).0).collect()) / (dt * dt) / median(joints.iter().map(|j| j.k[0]).collect());
    joints
        .iter()
        .map(|j| {
            let (m, i) = lighter(j);
            let (cap_lin, cap_ang) = (cap * m / (dt * dt), cap * i / (dt * dt));
            let mut out = *j;
            for (r, k) in out.k.iter_mut().enumerate() {
                *k = (*k * s).min(if r < 3 { cap_lin } else { cap_ang });
            }
            out
        })
        .collect()
}
