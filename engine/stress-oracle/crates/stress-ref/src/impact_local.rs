//! The local impact model (`capability::Profile::ImpactLocal`; stress-gpu fracture.slang
//! `fr_il_*`) in f64: what a predicted strike does to the struck wall near the hit, in
//! closed form, before the engine steps (forward only). The host's twin for the tests.
//!
//! The strike is a sphere (the shot: mass `m`, radius `r`) closing at `v` on a wall of
//! local thickness `H` (along the path) of the entry chunk's material. Engineering's
//! local/global split for hard missiles: the local response (contact, penetration,
//! plugging, spall) from closed forms; the global one is the engine's (the wall's pieces
//! as rigid bodies) and, afterwards, the fracture world's contact loads.
//!
//! - **Hertz contact** (Johnson, Contact Mechanics, 1985, §11.4): approach
//!   `h0 = (15 m v^2 / (16 E* sqrt r))^(2/5)`, duration `t_c = 2.94 h0 / v`, peak force
//!   `F = 4/3 E* sqrt(r) h0^(3/2)`, contact radius `a = sqrt(r h0)`, peak pressure
//!   `p0 = 3 F / (2 pi a^2)`, and the tensile ring stress at the contact's edge
//!   `(1 - 2 nu) p0 / 3`. Below the (rate-raised) strengths: no damage (the engine's
//!   contact is the whole story).
//! - **Rate**: the struck region's strain rate `v / H` raises the strengths by the
//!   material's dynamic increase factor (`Material::dif`, fib Model Code 2010 for concrete).
//! - **Ballistic limit**: the energy to push a plug through, the least of penetrating the
//!   whole thickness against the dynamic crushing strength
//!   (`E_pen = DIF f_c pi r^2 H`) and shearing out a cone plug of half-angle 45 degrees
//!   (`E_plug = 1/2 pi d_m H^2 DIF tau`: the shear stress `tau = min(cohesion, shear cap)`
//!   over the plug's mean circumference `pi d_m`, `d_m = 2 r + H`, while it slides through
//!   `H`). `V_bl = sqrt(2 E_bl / m_eff)`. At or above it the plug comes out; Recht and
//!   Ipson's residual speed `m / (m + M_plug) sqrt(v^2 - V_bl^2)` is the prediction the
//!   engine's own outcome is compared with.
//! - **Crater**: below the limit, crushed to the depth `x = 1/2 m_eff v^2 / (DIF f_c pi r^2)`
//!   within the contact (Poncelet's form with a constant resisting stress), when it is
//!   deeper than the contact is wide (a hard sphere's peak Hertz pressure beats a brittle
//!   surface's crushing strength at any speed; that alone is surface damage).
//! - **Spall**: the compressive wave from the contact, at most the planar impedance stress
//!   `Z_t v Z_p / (Z_p + Z_t)`, the contact's mean pressure and the (rate-raised) crushing
//!   strength (past it the front crushes, and the shot's whole face, radius `r`, loads it:
//!   the wave's source radius; the contact's `a` otherwise), spreading spherically (source
//!   radius over `H` at the back face), reflects from the free back face as tension. A pulse
//!   longer than the round trip (`c t_c > 2 H`) cancels part of it: the net tension is
//!   the reflected stress times `min(1, 2 H / (c t_c))`. Above the dynamic tensile
//!   strength a scab of depth `min(c t_c, H) / 2` comes off the back face, within the
//!   plug cone's footprint.
//!
//! Nothing here is fitted to the reference: every constant is the formula's or the
//! material's. The chunks resolve what breaks: a bond breaks when it crosses a zone's
//! boundary (one chunk's centre inside, the other's outside), so a zone smaller than a
//! chunk breaks nothing; the entry chunk is always in the plug.

use crate::material::DifParams;

/// The plug cone's half-angle's tangent (45 degrees, as world_coupling.rs `predict_punch`).
pub const CONE_TAN: f64 = 1.0;

/// The struck material, as the entry chunk has it.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub density: f64,
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    pub tensile: f64,
    pub compressive: f64,
    /// `min(cohesion, shear cap)`.
    pub shear: f64,
    pub dif: Option<DifParams>,
}

/// A predicted strike.
#[derive(Clone, Copy, Debug)]
pub struct Impact {
    /// The shot's mass, and the pair's reduced mass (the shot's for an anchored wall).
    pub mass: f64,
    pub reduced_mass: f64,
    pub radius: f64,
    /// The shot's plane-strain modulus `E / (1 - nu^2)`; the pair's Hertz modulus is
    /// `1 / (1/E'_shot + 1/E'_target)`.
    pub shot_modulus: f64,
    /// Closing speed along the path.
    pub speed: f64,
    /// The wall's thickness along the path.
    pub thickness: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Strike {
    pub contact_time: f64,
    pub peak_force: f64,
    pub contact_radius: f64,
    pub peak_pressure: f64,
    pub ring_stress: f64,
    pub strain_rate: f64,
    pub dif: f64,
    pub ballistic_energy: f64,
    pub ballistic_speed: f64,
    pub crater_depth: f64,
    pub plug_mass: f64,
    /// Recht-Ipson's prediction (0 below the limit).
    pub residual_speed: f64,
    /// The net tension the back face reflects.
    pub back_tension: f64,
    pub spall_depth: f64,
    pub damage: bool,
    pub crater: bool,
    pub spall: bool,
    pub perforate: bool,
}

/// The material's dynamic increase factor at `rate` (`DifParams::factor`; 1 without one).
pub fn dif(d: Option<&DifParams>, rate: f64) -> f64 {
    d.map_or(1.0, |d| d.factor(rate))
}

pub fn strike(i: &Impact, t: &Target) -> Strike {
    let pi = std::f64::consts::PI;
    let e_target = t.youngs_modulus / (1.0 - t.poisson_ratio * t.poisson_ratio);
    let e_star = 1.0 / (1.0 / i.shot_modulus + 1.0 / e_target);
    let v = i.speed.max(1e-3);
    let m = i.reduced_mass;
    let r = i.radius;
    let h = i.thickness.max(1e-6);
    // Hertz.
    let h0 = (15.0 * m * v * v / (16.0 * e_star * r.sqrt())).powf(0.4);
    let contact_time = 2.94 * h0 / v;
    let peak_force = 4.0 / 3.0 * e_star * r.sqrt() * h0.powf(1.5);
    let a = (r * h0).sqrt();
    let peak_pressure = 3.0 * peak_force / (2.0 * pi * a * a);
    let ring_stress = (1.0 - 2.0 * t.poisson_ratio) * peak_pressure / 3.0;
    // Rate.
    let strain_rate = v / h;
    let k = dif(t.dif.as_ref(), strain_rate);
    let (ft, fc, tau) = (k * t.tensile, k * t.compressive, k * t.shear);
    let damage = peak_pressure > fc || ring_stress > ft;
    // Ballistic limit.
    let area = pi * r * r;
    let e_pen = fc * area * h;
    let e_plug = 0.5 * pi * (2.0 * r + CONE_TAN * h) * h * h * tau;
    let ballistic_energy = e_pen.min(e_plug);
    let ballistic_speed = (2.0 * ballistic_energy / m).sqrt();
    let perforate = damage && v >= ballistic_speed;
    let (r1, r2) = (r, r + CONE_TAN * h);
    let plug_mass = t.density * pi * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2);
    let residual_speed = if perforate { i.mass / (i.mass + plug_mass) * (v * v - ballistic_speed * ballistic_speed).max(0.0).sqrt() } else { 0.0 };
    // Crater.
    let crater_depth = if damage && !perforate { (0.5 * m * v * v / (fc * area)).min(h) } else { 0.0 };
    // Crushed deeper than the contact is wide: a hard sphere's Hertz pressure crushes a
    // brittle surface at any speed, a crater only past the contact patch's own scale.
    let crater = crater_depth > a;
    // Spall.
    let c = (e_target / t.density).sqrt();
    let z_t = t.density * c;
    let shot_density = i.mass / (4.0 / 3.0 * pi * r * r * r);
    let z_p = shot_density * (i.shot_modulus / shot_density).sqrt();
    let mean_pressure = peak_force / (pi * a * a);
    let crushes = mean_pressure > fc;
    let source = (z_t * v * z_p / (z_p + z_t)).min(mean_pressure).min(fc);
    let source_radius = if crushes { a.max(r) } else { a };
    let pulse = c * contact_time;
    let back_tension = source * (source_radius / h).min(1.0) * (2.0 * h / pulse).min(1.0);
    let spall = damage && !perforate && back_tension > ft;
    let spall_depth = if spall { 0.5 * pulse.min(h) } else { 0.0 };
    Strike {
        contact_time,
        peak_force,
        contact_radius: a,
        peak_pressure,
        ring_stress,
        strain_rate,
        dif: k,
        ballistic_energy,
        ballistic_speed,
        crater_depth,
        plug_mass,
        residual_speed,
        back_tension,
        spall_depth,
        damage,
        crater,
        spall,
        perforate,
    }
}

/// Where a chunk's centre sits relative to the strike: `along` the path from the contact
/// point, `radial` distance from the path, whether it is the entry chunk, and whether the
/// shot overlaps its box on its way through the wall (`crossed`).
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub along: f64,
    pub radial: f64,
    pub entry: bool,
    pub crossed: bool,
}

/// The zones a strike's outcome breaks out: the plug (perforation: the cone, and every
/// chunk the shot crosses, since a perforation goes through what is in its path, in whole
/// chunks), the scab (spall), the crushed region (crater).
pub fn in_plug(s: &Strike, radius: f64, thickness: f64, p: &Place) -> bool {
    s.perforate && (p.entry || p.crossed || (p.along >= 0.0 && p.along <= thickness && p.radial <= radius + CONE_TAN * p.along))
}

pub fn in_scab(s: &Strike, radius: f64, thickness: f64, p: &Place) -> bool {
    s.spall && p.along >= thickness - s.spall_depth && p.along <= thickness && p.radial <= radius + CONE_TAN * thickness
}

pub fn in_crater(s: &Strike, radius: f64, p: &Place) -> bool {
    s.crater && p.along >= 0.0 && p.along <= s.crater_depth && p.radial <= radius.max(s.contact_radius)
}

/// Whether the strike breaks the bond between chunks at `a` and `b`: it crosses the
/// plug's or the scab's boundary, or touches the crushed region.
pub fn breaks(s: &Strike, radius: f64, thickness: f64, a: &Place, b: &Place) -> bool {
    in_plug(s, radius, thickness, a) != in_plug(s, radius, thickness, b)
        || in_scab(s, radius, thickness, a) != in_scab(s, radius, thickness, b)
        || in_crater(s, radius, a)
        || in_crater(s, radius, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    fn concrete() -> Target {
        let m = Material::concrete();
        Target { density: m.density, youngs_modulus: m.youngs_modulus, poisson_ratio: m.poisson_ratio, tensile: m.tensile_strength, compressive: m.compressive_strength, shear: m.cohesion, dif: m.dif }
    }

    fn steel_ball(radius: f64, speed: f64, thickness: f64) -> Impact {
        let mass = 7850.0 * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        Impact { mass, reduced_mass: mass, radius, shot_modulus: 200e9 / (1.0 - 0.09), speed, thickness }
    }

    /// Below the ballistic limit nothing comes out; at and above it the plug does, and the
    /// predicted exit speed rises from zero with the strike's speed.
    #[test]
    fn exit_speed_is_zero_below_the_limit_and_rises_above_it() {
        let t = concrete();
        let base = strike(&steel_ball(0.05, 50.0, 0.2), &t);
        let vbl = base.ballistic_speed;
        println!("ballistic limit {vbl:.1} m/s ({:.0} J)", base.ballistic_energy);
        let mut last = 0.0;
        for k in 1..=30 {
            let v = vbl * (0.5 + 0.05 * k as f64);
            let s = strike(&steel_ball(0.05, v, 0.2), &t);
            if v < 0.999 * s.ballistic_speed {
                assert!(!s.perforate && s.residual_speed == 0.0, "{v} m/s: below the limit");
            } else if v > 1.001 * s.ballistic_speed {
                assert!(s.perforate && s.residual_speed > last, "{v} m/s: out, faster than the last ({} after {last})", s.residual_speed);
                assert!(s.residual_speed < v, "slower than it came in");
                last = s.residual_speed;
            }
        }
        assert!(last > 0.0, "the sweep perforates");
    }

    /// A thinner wall is perforated where a thicker one is not; between them the wall
    /// spalls; the thickest only craters. (While the pulse is longer than the round trip,
    /// the back face's net tension does not depend on the thickness: the spherical spread
    /// and the overlap `2 H / (c t_c)` cancel.)
    #[test]
    fn thickness_sweep_goes_perforate_spall_crater() {
        let t = concrete();
        let v = 120.0;
        let outcomes: Vec<(f64, &str)> = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.6]
            .iter()
            .map(|&h| {
                let s = strike(&steel_ball(0.05, v, h), &t);
                (h, if s.perforate { "perforate" } else if s.spall { "spall" } else if s.crater { "crater" } else { "none" })
            })
            .collect();
        println!("{outcomes:?}");
        let rank = |o: &str| match o {
            "perforate" => 0,
            "spall" => 1,
            "crater" => 2,
            _ => 3,
        };
        for w in outcomes.windows(2) {
            assert!(rank(w[0].1) <= rank(w[1].1), "thicker walls do no worse: {outcomes:?}");
        }
        assert_eq!(outcomes[0].1, "perforate");
        assert!(outcomes.iter().any(|o| o.1 == "spall"), "a spall band: {outcomes:?}");
        assert_eq!(outcomes.last().unwrap().1, "crater");
    }

    /// Equal momentum, different speed: a light fast shot holes the wall, a heavy slow one
    /// (the same momentum) does not.
    #[test]
    fn equal_momentum_light_and_fast_holes_heavy_and_slow_does_not() {
        let t = concrete();
        let light = steel_ball(0.03, 400.0, 0.1);
        let p = light.mass * light.speed;
        let heavy_r = 0.15;
        let mut heavy = steel_ball(heavy_r, 0.0, 0.1);
        heavy.speed = p / heavy.mass;
        let (a, b) = (strike(&light, &t), strike(&heavy, &t));
        println!("light {:.2} kg at {:.0} m/s: {:?}; heavy {:.1} kg at {:.1} m/s: {:?}", light.mass, light.speed, a.perforate, heavy.mass, heavy.speed, b.perforate);
        assert!(a.perforate && !b.perforate);
    }

    /// A slow touch does no damage (a press, not a strike).
    #[test]
    fn a_slow_touch_does_no_damage() {
        let s = strike(&steel_ball(0.05, 0.05, 0.2), &concrete());
        assert!(!s.crater && !s.spall && !s.perforate, "{s:?}");
    }

    #[test]
    fn a_plug_bond_crosses_the_cone() {
        let s = strike(&steel_ball(0.05, 1000.0, 0.2), &concrete());
        assert!(s.perforate);
        let inside = Place { along: 0.05, radial: 0.0, entry: false, crossed: false };
        let outside = Place { along: 0.05, radial: 0.5, entry: false, crossed: false };
        assert!(breaks(&s, 0.05, 0.2, &inside, &outside));
        assert!(!breaks(&s, 0.05, 0.2, &inside, &Place { along: 0.15, radial: 0.0, entry: false, crossed: false }));
    }

    /// A chunk whose centre is outside the cone but whose box the shot crosses goes with the
    /// plug when the shot perforates (it would stand in the hole), and stays when it does not.
    #[test]
    fn a_perforation_takes_every_chunk_in_its_path() {
        let rim = Place { along: 0.05, radial: 0.12, entry: false, crossed: true };
        let held = strike(&steel_ball(0.05, 1000.0, 0.2), &concrete());
        assert!(held.perforate && in_plug(&held, 0.05, 0.2, &rim));
        let stopped = strike(&steel_ball(0.05, 0.05, 0.2), &concrete());
        assert!(!in_plug(&stopped, 0.05, 0.2, &rim));
    }
}
