//! Blast loading of threshold destructibles (`engine/docs/destruction-reactions.md`, row 3):
//! a charge's overpressure on the exposed faces of every blast-loaded destructible's chunks,
//! with the stress oracle's face-pressure model, entering the threshold model's break rule
//! as per-chunk loads.
//!
//! The physics is the oracle's own (`stress_ref::blast::FaceBlast`): Kinney-Graham peak
//! overpressure, duration and impulse at the scaled distance `Z = R / W^(1/3)`, the
//! Friedlander pulse fitted to that impulse, oblique reflection, shadowing (a face whose line
//! of sight to the charge crosses another chunk gets the diffracted fraction) and clearing.
//! Exposure (a face's area less the intact bonds on it) and clearing distances are the
//! oracle's (`stress_gpu::loads::exposure`, `clearing_distances`) on a mirror of the bond
//! states; shadowing culls occluders by reach as stress-ref `World::rebuild_blast` does.
//!
//! Forward-only (AGENTS.md): [`World::detonate`] is a queued command. At the start of each
//! step the pulse reaches, before the engine steps, each chunk's load over that step's window
//! is evaluated from the faces; a bond breaks when the load on either of its chunks beats its
//! tensile capacity (the threshold model's rule, `fracture.slang` `fr_breaks`: the peak of
//! the pulse, as `impact_peak` takes an impact's peak; the window's mean force without it);
//! the bodies split on the GPU (`GpuFracture::encode_split`); then each piece takes its
//! chunks' share of the blast impulse, so the engine's step meets moving pieces. Nothing is
//! re-simulated.
//!
//! Opt-in: only destructibles placed with `DestructibleDef::blast_loading` feel blasts;
//! without one, `detonate` changes nothing but the optional radial impulse.
//!
//! Cost: a step a pulse reaches waits for the GPU (the readbacks in flight, the split) and
//! evaluates the faces on the host; faces turned away from the charge carry no pressure and
//! are skipped.
//!
//! The per-chunk load ([`ChunkBlastLoad`]: force, moment, `Σ sym(r ⊗ F)` and impulse, tagged
//! [`LoadCause::Blast`]) is also written to a GPU buffer per destructible for kernels that
//! need it (chunk crushing).

use crate::{ExplosionDef, World};

/// At most this many blasts act at once (a queued blast acts until its last face's positive
/// phase ends).
pub const BLAST_CAP: usize = 16;

/// The smallest scaled distance the Kinney-Graham fits cover (m/kg^(1/3)): a loaded face
/// nearer the charge is an error, never clamped.
pub const MIN_SCALED_DISTANCE: f64 = 0.05;

/// A charge (`World::detonate`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlastDef {
    /// World position of the charge (m).
    pub position: [f32; 3],
    /// TNT-equivalent mass (kg).
    pub tnt_kg: f32,
    /// Also push bodies with rubble's radial impulse ([`World::explode`]) from the same
    /// point. It reaches every shape its mask meets, destructible pieces included: give
    /// destructibles their own category and leave it out of the mask for them to take only
    /// the face-pressure impulse.
    pub radial: Option<RadialImpulse>,
}

impl Default for BlastDef {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            tnt_kg: 1.0,
            radial: None,
        }
    }
}

/// [`ExplosionDef`]'s radial impulse without its position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialImpulse {
    pub radius: f32,
    pub falloff: f32,
    pub impulse_per_area: f32,
    pub mask: u16,
}

/// What loaded a chunk (the charter's causes; [`LoadCause::code`] tags GPU records).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoadCause {
    Impact,
    Blast,
    Contact,
    Load,
}

impl LoadCause {
    pub fn code(self) -> u32 {
        match self {
            LoadCause::Impact => 0,
            LoadCause::Blast => 1,
            LoadCause::Contact => 2,
            LoadCause::Load => 3,
        }
    }
}

/// One chunk's blast load over a step's window (world frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkBlastLoad {
    pub chunk: u32,
    /// The force the break rule compares (N): at the window's peak (or its mean force
    /// without `impact_peak`).
    pub force: [f32; 3],
    /// Its moment about the chunk centre (N m).
    pub moment: [f32; 3],
    /// `Σ sym(r ⊗ F)` over the loaded faces when the largest face pressure acts (N m; xx,
    /// yy, zz, xy, xz, yz), `r` from the chunk centre: the blast's part of the chunk's mean
    /// stress times its volume.
    pub stress_volume: [f32; 6],
    /// The impulse over the window (N s) and its moment about the chunk centre (N m s).
    pub impulse: [f32; 3],
    pub angular_impulse: [f32; 3],
    /// The largest face pressure in the window (Pa).
    pub peak_pressure: f32,
    pub cause: LoadCause,
}

/// One blast while its pulse lasts.
#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "stress"), allow(dead_code))]
pub(crate) struct ActiveBlast {
    pub id: u64,
    pub position: [f64; 3],
    pub tnt: f64,
    /// The step it starts at (blast time 0 is that step's start).
    pub start: u64,
    /// When its last face's positive phase ends (s, blast time), as last built.
    pub end: f64,
}

#[derive(Default)]
#[cfg_attr(not(feature = "stress"), allow(dead_code))]
pub(crate) struct Blasts {
    pub active: Vec<ActiveBlast>,
    pub next: u64,
    /// Events of the splits blasts made before a step, for that step's frame.
    pub events: Vec<(u64, crate::events::DestructionEvent)>,
    /// The topology (bonds broken across blast-loaded destructibles) the face caches were
    /// built at.
    pub version: Option<u64>,
}

impl World {
    /// Detonate a charge at the start of the next step: its overpressure loads the exposed
    /// chunk faces of every blast-loaded threshold destructible (`DestructibleDef::
    /// blast_loading`) over the steps its pulse lasts, breaking the bonds it overloads and
    /// pushing the pieces; [`BlastDef::radial`] also pushes bodies. A face nearer the charge
    /// than the Kinney-Graham fits reach ([`MIN_SCALED_DISTANCE`]) is an error.
    pub fn detonate(&mut self, def: &BlastDef) -> anyhow::Result<()> {
        anyhow::ensure!(
            def.position.iter().all(|x| x.is_finite()),
            "blast position must be finite"
        );
        anyhow::ensure!(
            def.tnt_kg.is_finite() && def.tnt_kg > 0.0,
            "blast TNT mass must be finite and positive (got {} kg)",
            def.tnt_kg
        );
        anyhow::ensure!(
            self.blasts.active.len() < BLAST_CAP,
            "more than {BLAST_CAP} blasts at once"
        );
        if let Some(r) = def.radial {
            anyhow::ensure!(
                r.radius.is_finite() && r.falloff.is_finite() && r.impulse_per_area.is_finite(),
                "blast radial impulse must be finite"
            );
            anyhow::ensure!(
                r.radius >= 0.0 && r.falloff >= 0.0,
                "negative explosion radius or falloff"
            );
            anyhow::ensure!(
                (self.queries.explosions.len() as u32) < engine_schema::EXPLODE_CAP,
                "more than {} explosions in one step",
                engine_schema::EXPLODE_CAP
            );
        }
        // Without the `stress` feature no destructible feels blasts: only the radial
        // impulse acts.
        #[cfg(feature = "stress")]
        {
            self.blasts.next += 1;
            let id = self.blasts.next;
            self.blasts.active.push(ActiveBlast {
                id,
                position: def.position.map(f64::from),
                tnt: f64::from(def.tnt_kg),
                start: self.step,
                end: 0.0,
            });
            if let Err(e) = self.blast_prepare() {
                self.blasts.active.retain(|b| b.id != id);
                self.blast_forget(id);
                return Err(e);
            }
        }
        if let Some(r) = def.radial {
            self.explode(&ExplosionDef {
                position: def.position,
                radius: r.radius,
                falloff: r.falloff,
                impulse_per_area: r.impulse_per_area,
                mask: r.mask,
            })?;
        }
        self.dirty = true;
        Ok(())
    }

    /// Every ready frame processed and kept for the next poll: the host's view of the
    /// destructibles is the GPU's as of the last step.
    #[cfg(feature = "stress")]
    pub(crate) fn blast_settle(&mut self) -> anyhow::Result<()> {
        let frames = self.poll_blocking()?;
        self.ready_frames = frames;
        Ok(())
    }

    /// The splits blasts made before `step`, for its frame (and any earlier frame that
    /// never ran).
    #[cfg_attr(not(feature = "stress"), allow(dead_code))]
    pub(crate) fn take_blast_events(&mut self, step: u64) -> Vec<crate::events::DestructionEvent> {
        if self.blasts.events.is_empty() {
            return Vec::new();
        }
        let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.blasts.events)
            .into_iter()
            .partition(|e| e.0 <= step);
        self.blasts.events = later;
        now.into_iter().map(|e| e.1).collect()
    }
}

#[cfg(feature = "stress")]
impl crate::DestructibleAssetDesc {
    /// A stress-oracle scene (`stress_ref::builders`' benchmark and showcase scenes) as a
    /// destructible for `StrengthModel::Threshold`: its structures' chunks as boxes, the
    /// asset frame the scene's turned Y up. The scene's loads and impactors are not part of
    /// the asset.
    pub fn from_stress_scene(scene: &stress_ref::scene::Scene) -> anyhow::Result<Self> {
        use glam::{DQuat, DVec3};
        scene
            .validate()
            .map_err(|e| anyhow::anyhow!("stress scene: {e}"))?;
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let r = DQuat::from_xyzw(-h, 0.0, 0.0, h);
        let wxyz = |q: [f64; 4]| DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize();
        let mut chunks = Vec::new();
        for body in &scene.bodies {
            let (bp, bq) = (DVec3::from(body.position), wxyz(body.orientation));
            for c in &body.chunks {
                let half = DVec3::from(c.half_extents);
                let density = scene.material(&c.material).density;
                let mass = density * 8.0 * half.x * half.y * half.z;
                let e = half * half;
                let inertia = DVec3::new(e.y + e.z, e.x + e.z, e.x + e.y) * (mass / 3.0);
                let fixed = c.support != stress_ref::scene::Support::None;
                let p = r * (bp + bq * DVec3::from(c.center));
                let q = (r * bq * wxyz(c.orientation)).normalize();
                chunks.push(crate::ChunkDesc {
                    pose: crate::Transform::new(p.as_vec3().to_array(), q.as_quat().to_array()),
                    mass: if fixed { 0.0 } else { mass as f32 },
                    inertia: inertia.as_vec3().to_array(),
                    shapes: vec![crate::Geometry::Box {
                        half: half.as_vec3().to_array(),
                        transform: crate::Transform::IDENTITY,
                    }],
                    fragment: None,
                    node: None,
                    node_frame: None,
                });
            }
        }
        let desc = Self {
            name: Some(scene.name.clone()),
            chunks,
            bonds: Vec::new(),
            preload: None,
            solver: None,
            stress: Some(crate::StressSource {
                scene: std::sync::Arc::new(Ok(scene.clone())),
                placement: [0.0; 3],
                nodes: Vec::new(),
            }),
        };
        desc.validate()?;
        Ok(desc)
    }
}

/// The face-pressure model as the threshold destructibles take it: per-chunk loads over a
/// step's window from the oracle's `FaceBlast`s.
#[cfg(feature = "stress")]
pub(crate) mod model {
    use glam::DVec3;
    use stress_ref::blast::FaceBlast;

    /// A loaded face as it is now: centre, outward unit normal (world), exposed area.
    #[derive(Clone, Copy, Debug)]
    pub struct Face {
        pub point: DVec3,
        pub normal: DVec3,
        pub area: f64,
        pub blast: FaceBlast,
    }

    /// Simpson intervals per smooth piece of a pulse.
    const SIMPSON: usize = 256;

    /// `∫ p dt` of a face's pulse over `[t0, t1]` (Pa s; blast time): composite Simpson on
    /// each smooth piece (the positive phase, split where clearing ends).
    pub fn pressure_integral(b: &FaceBlast, t0: f64, t1: f64) -> f64 {
        let lo = t0.max(b.arrival);
        let hi = t1.min(b.arrival + b.duration);
        if hi <= lo {
            return 0.0;
        }
        let mut cuts = vec![lo];
        let clear = b.arrival + b.clearing_time;
        if b.clearing_time > 0.0 && clear > lo && clear < hi {
            cuts.push(clear);
        }
        cuts.push(hi);
        let mut sum = 0.0;
        for w in cuts.windows(2) {
            let (a, c) = (w[0], w[1]);
            let h = (c - a) / SIMPSON as f64;
            // Smooth on `[a, c]`; `pressure` is right-continuous at the arrival's jump.
            let f = |t: f64| b.pressure(t);
            let mut s = f(a) + f(c);
            for i in 1..SIMPSON {
                s += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
            }
            sum += s * h / 3.0;
        }
        sum
    }

    /// A chunk's load over a window.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Window {
        pub force: DVec3,
        pub moment: DVec3,
        /// xx, yy, zz, xy, xz, yz.
        pub stress_volume: [f64; 6],
        pub impulse: DVec3,
        pub angular_impulse: DVec3,
        pub peak_pressure: f64,
    }

    impl Window {
        pub fn is_zero(&self) -> bool {
            self.impulse == DVec3::ZERO && self.force == DVec3::ZERO && self.peak_pressure == 0.0
        }
    }

    /// The load of `faces` (one chunk's, centre `centre`) over `[t0, t1]` (blast time).
    /// `peak`: the force is the largest resultant in the window (each face's pressure jumps
    /// to its peak at arrival and only falls after it, so the resultant is sampled at the
    /// window's start and at every arrival in it); else the window's mean force.
    pub fn window(faces: &[Face], centre: DVec3, t0: f64, t1: f64, peak: bool) -> Window {
        let mut out = Window::default();
        if faces.is_empty() || t1 <= t0 {
            return out;
        }
        for f in faces {
            let j = -f.normal * (f.area * pressure_integral(&f.blast, t0, t1));
            out.impulse += j;
            out.angular_impulse += (f.point - centre).cross(j);
        }
        let at = |t: f64| {
            let mut force = DVec3::ZERO;
            let mut moment = DVec3::ZERO;
            let mut sv = [0.0; 6];
            let mut p_max: f64 = 0.0;
            for f in faces {
                let p = f.blast.pressure(t);
                p_max = p_max.max(p);
                let ff = -f.normal * (f.area * p);
                let r = f.point - centre;
                force += ff;
                moment += r.cross(ff);
                sv[0] += r.x * ff.x;
                sv[1] += r.y * ff.y;
                sv[2] += r.z * ff.z;
                sv[3] += 0.5 * (r.x * ff.y + r.y * ff.x);
                sv[4] += 0.5 * (r.x * ff.z + r.z * ff.x);
                sv[5] += 0.5 * (r.y * ff.z + r.z * ff.y);
            }
            (force, moment, sv, p_max)
        };
        let mut times = vec![t0];
        times.extend(
            faces
                .iter()
                .map(|f| f.blast.arrival)
                .filter(|&a| a > t0 && a < t1),
        );
        // The force and its moment at the largest resultant; the stress at the largest
        // face pressure (a chunk loaded on opposite faces feels stress, not force).
        let mut best: Option<(DVec3, DVec3, [f64; 6], f64)> = None;
        let mut sv = [0.0; 6];
        for &t in &times {
            let s = at(t);
            if s.3 > out.peak_pressure {
                out.peak_pressure = s.3;
                sv = s.2;
            }
            if best.is_none_or(|b| s.0.length() > b.0.length()) {
                best = Some(s);
            }
        }
        let (force, moment, _, _) = best.expect("a sample");
        out.stress_volume = sv;
        if peak {
            out.force = force;
            out.moment = moment;
        } else {
            let dt = t1 - t0;
            out.force = out.impulse / dt;
            out.moment = out.angular_impulse / dt;
        }
        out
    }

    /// Gauss-Legendre nodes and weights on [-1, 1], 8 points.
    const GL8: [(f64, f64); 8] = [
        (-0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
        (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (-0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (-0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
    ];

    /// Quadrature nodes `(t, weight)` over `[t0, t1]` for an integrand of the faces'
    /// pressures: split where any face's pressure is not smooth (its arrival, the end of
    /// its clearing, the end of its positive phase), each piece graded geometrically from
    /// its start (the first sub-piece `grade` times the fastest decay time `duration / b`
    /// of the faces acting, each next twice the last), 8-point Gauss-Legendre on each
    /// sub-piece. Open nodes: none with a weight sits on a jump; each piece's start is a
    /// node of weight 0, where the peaks are taken.
    pub fn nodes(faces: &[Face], t0: f64, t1: f64, grade: f64) -> Vec<(f64, f64)> {
        let mut cuts = vec![t0, t1];
        let mut tau = f64::INFINITY;
        for f in faces {
            let b = &f.blast;
            for t in [
                b.arrival,
                b.arrival + b.clearing_time,
                b.arrival + b.duration,
            ] {
                if t > t0 && t < t1 {
                    cuts.push(t);
                }
            }
            if b.arrival < t1 && b.arrival + b.duration > t0 {
                tau = tau.min(b.duration / b.decay.max(1.0));
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let mut out = Vec::new();
        if !tau.is_finite() {
            return out;
        }
        for w in cuts.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            // The piece's start with no weight: the peaks (a pressure jumps there).
            out.push((lo, 0.0));
            let mut a = lo;
            let mut width = (grade * tau).min(hi - lo);
            while a < hi {
                let b = (a + width).min(hi);
                let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
                out.extend(GL8.iter().map(|&(x, w)| (mid + half * x, w * half)));
                a = b;
                width *= 2.0;
            }
        }
        out
    }

    /// The grading [`crush_damage`] integrates with (a quarter of the fastest decay time).
    pub const GRADE: f64 = 0.25;

    /// A blast window's crush damage on one chunk (`engine/docs/destruction-reactions.md`
    /// section 1, "Blast"): the faces' forces `f = -n A p(t)` at their centres, balanced as
    /// `fracture.slang` `balanced` balances contact loads (reactions through the chunk's
    /// intact bonds at their area-weighted centroid `bonds`, relative to `centre`; with no
    /// intact bond, by its inertia at the loads' |f|-weighted centroid), through the law's
    /// invariants over `volume`, `D += integral excess^2 dt / (crushViscosity crushEnergy)`
    /// over `[t0, t1]` (blast time; [`nodes`]). Returns the increment and the peak (p, q).
    #[allow(clippy::too_many_arguments)]
    pub fn crush_damage(
        faces: &[Face],
        centre: DVec3,
        bonds: Option<DVec3>,
        volume: f64,
        m: &stress_ref::crush::CrushParams,
        t0: f64,
        t1: f64,
        grade: f64,
    ) -> (f64, f64, f64) {
        use stress_ref::crush::{invariants, sym_outer};
        let (mut damage, mut p_max, mut q_max) = (0.0, 0.0f64, 0.0f64);
        for (t, w) in nodes(faces, t0, t1, grade) {
            let mut virial = [0.0; 6];
            let mut force = DVec3::ZERO;
            let mut spread = (DVec3::ZERO, 0.0);
            for f in faces {
                let p = f.blast.pressure(t);
                if p == 0.0 {
                    continue;
                }
                let ff = -f.normal * (f.area * p);
                let r = f.point - centre;
                let s = sym_outer(r.to_array(), ff.to_array());
                for (v, x) in virial.iter_mut().zip(s) {
                    *v += x;
                }
                force += ff;
                let m = ff.length();
                spread.0 += m * r;
                spread.1 += m;
            }
            if spread.1 == 0.0 {
                continue;
            }
            let at = bonds.unwrap_or(spread.0 / spread.1);
            let s = sym_outer(at.to_array(), (-force).to_array());
            for (v, x) in virial.iter_mut().zip(s) {
                *v += x;
            }
            let (p, q) = invariants(virial, volume);
            p_max = p_max.max(p);
            q_max = q_max.max(q);
            if let Some(e) = m.excess(p, q) {
                damage += w * e * e;
            }
        }
        (damage / (m.crush_viscosity * m.crush_energy), p_max, q_max)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use stress_ref::blast::{
            friedlander_impulse_fraction, incident_impulse, incident_overpressure,
            oblique_overpressure, reflected_overpressure, P_ATM, SHADOW_FACTOR,
        };

        fn face(blast: FaceBlast, normal: DVec3, area: f64) -> Face {
            Face {
                point: DVec3::ZERO,
                normal,
                area,
                blast,
            }
        }

        /// A face square to the charge at the scaled distance of the Kinney-Graham table's
        /// Z = 1 m/kg^(1/3) (9.96 atm side-on): its peak is the normal reflection of that.
        #[test]
        fn peak_overpressure_is_kinney_graham_reflected() {
            let w: f64 = 8.0;
            let r = 1.0 * w.cbrt();
            let fb = FaceBlast::new(w, 0.0, r, 1.0, false, f64::INFINITY);
            let p_so = incident_overpressure(1.0);
            assert!((p_so / P_ATM - 9.96).abs() < 0.05, "{}", p_so / P_ATM);
            let area = 0.25;
            // The charge is along +x of the face, the face's outward normal too.
            let f = face(fb, DVec3::X, area);
            let win = window(&[f], DVec3::new(-0.1, 0.0, 0.0), 0.0, 1.0, true);
            let want = reflected_overpressure(p_so);
            assert!(
                (win.peak_pressure - want).abs() < 1e-9 * want,
                "{} vs {want}",
                win.peak_pressure
            );
            // Pushed away from the charge, by peak x area.
            assert!(
                (win.force.x + want * area).abs() < 1e-9 * want * area,
                "{:?}",
                win.force
            );
            assert!(win.force.y == 0.0 && win.force.z == 0.0);
            // At incidence theta the oracle's oblique law.
            let c = 0.6;
            let ob = FaceBlast::new(w, 0.0, r, c, false, f64::INFINITY);
            assert!((ob.peak - oblique_overpressure(p_so, c)).abs() < 1e-9 * want);
            // Shadowed: the diffracted fraction of side-on.
            let sh = FaceBlast::new(w, 0.0, r, 1.0, true, f64::INFINITY);
            assert!((sh.peak - SHADOW_FACTOR * p_so).abs() < 1e-9 * p_so);
        }

        /// Without clearing a face's impulse is the Kinney-Graham incident impulse times its
        /// reflection factor (the pulse's decay is fitted to that impulse), times its area;
        /// the impulse toward the charge is that times cos(theta). Split windows add up.
        #[test]
        fn face_impulse_is_the_friedlander_integral() {
            let w: f64 = 20.0;
            let r = 6.0;
            let z = r / w.cbrt();
            let (area, c) = (0.5, 0.8);
            let fb = FaceBlast::new(w, 0.0, r, c, false, f64::INFINITY);
            let i_s = incident_impulse(z, w);
            let want = i_s * fb.peak / fb.incident * area;
            let shape = fb.peak * fb.duration * friedlander_impulse_fraction(fb.decay) * area;
            assert!((shape - want).abs() < 1e-6 * want, "{shape} vs {want}");
            // The face normal at theta from the direction to the charge (+x).
            let n = DVec3::new(c, (1.0 - c * c).sqrt(), 0.0);
            let f = face(fb, n, area);
            let whole = window(&[f], DVec3::ZERO, 0.0, 1.0, true);
            let j = whole.impulse.length();
            assert!((j - want).abs() < 1e-6 * want, "{j} vs {want}");
            let toward = -whole.impulse.dot(DVec3::X);
            assert!((toward - want * c).abs() < 1e-6 * want, "{toward}");
            // Three windows across the arrival and the positive phase sum to the whole.
            let a = fb.arrival;
            let cuts = [0.0, a + 0.3 * fb.duration, a + 0.7 * fb.duration, 1.0];
            let parts: f64 = cuts
                .windows(2)
                .map(|k| window(&[f], DVec3::ZERO, k[0], k[1], true).impulse.length())
                .sum();
            assert!((parts - want).abs() < 1e-6 * want, "{parts} vs {want}");
            // A window before arrival carries nothing.
            assert!(window(&[f], DVec3::ZERO, 0.0, 0.99 * a, true).is_zero());
        }

        /// Clearing relaxes the reflected pressure, so it lowers the impulse, never the peak.
        #[test]
        fn clearing_lowers_impulse_not_peak() {
            let w: f64 = 20.0;
            let open = FaceBlast::new(w, 0.0, 6.0, 1.0, false, f64::INFINITY);
            let cleared = FaceBlast::new(w, 0.0, 6.0, 1.0, false, 0.2);
            let i_open = pressure_integral(&open, 0.0, 1.0);
            let i_clear = pressure_integral(&cleared, 0.0, 1.0);
            assert!(i_clear < i_open && i_clear > 0.0);
            assert_eq!(open.peak, cleared.peak);
        }

        /// Pressure and impulse fall with distance; the peak force mode takes the peak, the
        /// mean mode the window's mean (impulse / window).
        #[test]
        fn falls_with_distance_and_modes() {
            let w = 10.0;
            let near = FaceBlast::new(w, 0.0, 3.0, 1.0, false, f64::INFINITY);
            let far = FaceBlast::new(w, 0.0, 12.0, 1.0, false, f64::INFINITY);
            assert!(near.peak > far.peak && near.arrival < far.arrival);
            let a = window(&[face(near, DVec3::X, 1.0)], DVec3::ZERO, 0.0, 1.0, true);
            let b = window(&[face(far, DVec3::X, 1.0)], DVec3::ZERO, 0.0, 1.0, true);
            assert!(a.force.length() > b.force.length() && a.impulse.length() > b.impulse.length());
            let m = window(&[face(near, DVec3::X, 1.0)], DVec3::ZERO, 0.0, 0.5, false);
            assert!((m.force - m.impulse / 0.5).length() < 1e-9 * m.force.length());
        }

        /// A chunk loaded on two opposite faces equally feels no net force, but a stress.
        #[test]
        fn balanced_faces_load_stress_not_force() {
            let fb = FaceBlast::new(5.0, 0.0, 4.0, 1.0, false, f64::INFINITY);
            let h = 0.1;
            let faces = [
                Face {
                    point: DVec3::new(h, 0.0, 0.0),
                    normal: DVec3::X,
                    area: 0.04,
                    blast: fb,
                },
                Face {
                    point: DVec3::new(-h, 0.0, 0.0),
                    normal: -DVec3::X,
                    area: 0.04,
                    blast: fb,
                },
            ];
            let win = window(&faces, DVec3::ZERO, 0.0, 1.0, true);
            assert!(win.force.length() < 1e-9 && win.impulse.length() < 1e-9);
            // Compression along x: r . F < 0 on both faces.
            assert!(win.stress_volume[0] < 0.0 && win.stress_volume[1] == 0.0);
            assert!((win.stress_volume[0] + 2.0 * h * fb.peak * 0.04).abs() < 1e-6);
        }

        /// The reference building's drywall (its pack's `crush` block).
        fn drywall() -> stress_ref::crush::CrushParams {
            stress_ref::crush::CrushParams {
                cap_pressure: 2e6,
                cohesion: 4.8e5,
                friction_slope: 1.2,
                crush_energy: 2e6,
                crush_viscosity: 2e5,
            }
        }

        /// A slab 2h thick loaded on one face, its reactions through bonds at its centre:
        /// the law's mean stress is the uniaxial `sigma_n = -P A h / V = -P/2`, so
        /// `p = P/6`, `q = P/2`. The damage is that law integrated over the pulse: equal to
        /// a brute-force midpoint sum over the positive phase (1e6 points).
        #[test]
        fn blast_crush_damage_is_the_law_over_the_pulse() {
            let (w, r) = (20.0, 0.2);
            let fb = FaceBlast::new(w, 0.0, r, 1.0, false, f64::INFINITY);
            let (h, area) = (0.06, 1.8 * 2.8);
            let volume = 2.0 * h * area;
            let f = Face {
                point: DVec3::new(0.0, 0.0, h),
                normal: DVec3::Z,
                area,
                blast: fb,
            };
            let m = drywall();
            let (d, p, q) = crush_damage(
                &[f],
                DVec3::ZERO,
                Some(DVec3::ZERO),
                volume,
                &m,
                0.0,
                1.0 / 60.0,
                GRADE,
            );
            assert!(
                (p - fb.peak / 6.0).abs() < 1e-3 * fb.peak,
                "{p} vs {}",
                fb.peak / 6.0
            );
            assert!((q - fb.peak / 2.0).abs() < 1e-3 * fb.peak, "{q}");
            let n = 1_000_000;
            let (a, end) = (fb.arrival, (fb.arrival + fb.duration).min(1.0 / 60.0));
            let dt = (end - a) / n as f64;
            let mut want = 0.0;
            for i in 0..n {
                let pr = fb.pressure(a + (i as f64 + 0.5) * dt);
                if let Some(e) = m.excess(pr / 6.0, pr / 2.0) {
                    want += e * e * dt;
                }
            }
            want /= m.crush_viscosity * m.crush_energy;
            assert!(want > 0.5, "the charge crushes: {want}");
            assert!((d - want).abs() < 2e-3 * want, "{d} vs {want}");
            // Finer grading changes nothing that matters.
            let (fine, ..) = crush_damage(
                &[f],
                DVec3::ZERO,
                Some(DVec3::ZERO),
                volume,
                &m,
                0.0,
                1.0 / 60.0,
                GRADE / 16.0,
            );
            assert!((d - fine).abs() < 1e-3 * fine, "{d} vs {fine}");
            // Windows add up.
            let split: f64 = [
                (0.0, 0.3 * fb.duration + a),
                (0.3 * fb.duration + a, 1.0 / 60.0),
            ]
            .iter()
            .map(|&(t0, t1)| {
                crush_damage(
                    &[f],
                    DVec3::ZERO,
                    Some(DVec3::ZERO),
                    volume,
                    &m,
                    t0,
                    t1,
                    GRADE,
                )
                .0
            })
            .sum();
            assert!((split - d).abs() < 2e-3 * d, "{split} vs {d}");
        }

        /// A lone chunk's reaction is its inertia: one loaded face is balanced at its own
        /// centre and never crushes it; nor does a pulse that has not arrived.
        #[test]
        fn blast_crushes_no_lone_chunk_on_one_face() {
            let fb = FaceBlast::new(20.0, 0.0, 0.2, 1.0, false, f64::INFINITY);
            let f = Face {
                point: DVec3::new(0.0, 0.0, 0.06),
                normal: DVec3::Z,
                area: 5.0,
                blast: fb,
            };
            let (d, p, _) = crush_damage(
                &[f],
                DVec3::ZERO,
                None,
                0.6,
                &drywall(),
                0.0,
                1.0 / 60.0,
                GRADE,
            );
            assert_eq!(d, 0.0);
            assert!(p.abs() < 1.0, "{p}");
            let (d, ..) = crush_damage(
                &[f],
                DVec3::ZERO,
                Some(DVec3::ZERO),
                0.6,
                &drywall(),
                0.0,
                0.5 * fb.arrival,
                GRADE,
            );
            assert_eq!(d, 0.0);
        }
    }
}
