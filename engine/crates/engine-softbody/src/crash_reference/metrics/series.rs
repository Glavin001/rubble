//! Time histories of a crash run, the curves crash tests record: per body, the acceleration,
//! velocity and displacement of its reference location (a rear-seat crossmember's
//! accelerometers: [`reference_nodes`]) and of its centre of mass; for the scene, the force
//! on the static boxes (a wall's or a pole's load cells). Host code on every executor's
//! samples (every [`super::super::SAMPLE_STEPS`] steps, 4 ms), as every metric is.
//!
//! **Axes** (SAE J211, as NHTSA's channels): x along the body's first heading (forward),
//! y toward its right; time in ms after the layout's first contact ([`super::super::Layout::impact_s`]).
//!
//! **Method** (stated so the numbers can be checked):
//! - velocity: the sample's node velocities, averaged (the reference location: unweighted,
//!   a place; the centre of mass: mass-weighted), point samples every 4 ms;
//! - displacement: the same nodes' positions, averaged, less their value at first contact
//!   (linear between samples);
//! - acceleration: the derivative of the velocity samples by central differences
//!   ([`signal::differentiate`]): exactly the mean acceleration over the 8 ms around each
//!   sample (Δv over 2 Δt), a box-car filter. The J211 CFC 60 filter cannot be applied to
//!   4 ms samples: its design frequency (2.0775 × 60 = 124.7 Hz) is the 125 Hz Nyquist
//!   frequency of the sampling ([`signal::cfc`] refuses it). What the 8 ms box-car keeps
//!   against CFC 60, measured on the CPU reference's 2 kHz state and on the NHTSA records
//!   resampled the same way: `engine/docs/crash-reference.md`, "Time histories";
//! - the 2 kHz trace ([`Fine`]): an executor that can ([`super::super::Runner::start_trace`],
//!   the CPU reference) hands the reference locations' velocity after every step, and the
//!   acceleration is then filtered as the tests' is (central differences, CFC 60). It changes
//!   nothing in the run (`engine-softbody-lab/tests/crash_series_rate.rs`: the same metrics to
//!   the bit); executors that cannot trace report the 4 ms series alone;
//! - force on the static boxes: −d(Σ m v)/dt over every node of every body (the same
//!   central difference), along body 0's initial velocity. Exact for vehicles against fixed
//!   boxes up to the other horizontal forces on them: tyre and ground friction and drag.
//!   Those are measured before contact (`all.box_force_pre_n`) and bounded by μ m g during
//!   it.

use super::signal;
use glam::{DVec3, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Standard gravity (m/s², the unit "g" of every acceleration here, as SAE J211's).
pub const G0: f64 = 9.80665;
/// Rebound is read up to this long after first contact (s), as the NHTSA references are.
pub const REBOUND_UNTIL_S: f64 = 0.200;
/// The box force's error floor is read over this long before first contact (s).
const PRE_S: f64 = 0.040;

/// One location's kinematics on the series' time base (SAE axes of the body's first frame).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Kinematics {
    /// Acceleration along x and y (g).
    pub ax_g: Vec<f64>,
    pub ay_g: Vec<f64>,
    /// Velocity along x and y (m/s).
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    /// Displacement from first contact along x and y (m).
    pub dx: Vec<f64>,
    pub dy: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodySeries {
    pub name: String,
    pub mass_kg: f64,
    /// The reference location's node names (empty: it is the centre of mass).
    pub ref_nodes: Vec<String>,
    /// The reference location (the rear-seat crossmember's place).
    pub reference: Kinematics,
    pub com: Kinematics,
}

/// A run's time histories.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Series {
    /// ms after first contact.
    pub t_ms: Vec<f64>,
    pub bodies: Vec<BodySeries>,
    /// Force on the static boxes along body 0's initial velocity (N; empty without boxes).
    pub box_force_n: Vec<f64>,
    /// How each curve was made (the module's method, in one line).
    pub method: String,
    /// The reference locations' 2 kHz trace, from an executor that traces every step
    /// ([`super::super::Runner::start_trace`]: the CPU reference); None from the others.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fine: Option<Fine>,
}

/// The reference locations at the solver's step rate, filtered as a test's accelerometer
/// is: the velocity after every step, its derivative by central differences (the mean over
/// 1 ms) through SAE J211 CFC 60 ([`signal::cfc`], valid at 2 kHz); decimated to 1 kHz.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Fine {
    /// The trace's own rate (Hz) and the curves' (after decimation).
    pub rate_hz: f64,
    pub curve_hz: f64,
    /// ms after first contact.
    pub t_ms: Vec<f64>,
    pub bodies: Vec<FineBody>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FineBody {
    pub name: String,
    /// The reference location and the centre of mass.
    pub reference: FineKinematics,
    pub com: FineKinematics,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FineKinematics {
    /// CFC 60 acceleration along x and y (g), velocity along x and y (m/s).
    pub ax_g: Vec<f64>,
    pub ay_g: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
}

/// The nodes of a design's rear-seat crossmember, chosen by position at rest (the design's
/// own frame: forward +Z, up +Y): its cabin floor behind the B-pillar and ahead of the rear
/// axle. The floor: the nodes of the cabin section within 5 cm of the section's lowest node.
/// The B-pillar: the generator's `cabin_width` gauge (measured across the B-pillar), else
/// the wheelbase's middle. Strictly between the two by 5 cm. Empty when the design has no
/// cabin or no wheels (its centre of mass is then the reference).
pub fn reference_nodes(t: &super::super::Template) -> Vec<usize> {
    let d = &t.desc;
    let section = |i: usize| t.zones.parts.iter().filter(|(p, _)| d.nodes[i].name.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, s)| *s);
    let cabin: Vec<usize> = (0..d.nodes.len()).filter(|&i| section(i) == Some(super::super::Section::Cabin)).collect();
    let e = super::super::Extent::of(d);
    let (Some((front, rear)), false) = (e.axles, cabin.is_empty()) else { return Vec::new() };
    let named = |n: &str| d.nodes.iter().find(|x| x.name == n).map(|x| x.position[2]);
    let pillar = t.zones.gauges.iter().find(|g| g.name == "cabin_width").and_then(|g| named(&g.a)).unwrap_or(0.5 * (front + rear));
    let floor = cabin.iter().map(|&i| d.nodes[i].position[1]).fold(f32::MAX, f32::min);
    cabin.into_iter().filter(|&i| {
        let p = d.nodes[i].position;
        p[1] <= floor + 0.05 && p[2] < pillar - 0.05 && p[2] > rear + 0.05
    })
    .collect()
}

/// One body's samples.
#[derive(Default)]
pub(super) struct BodyTrack {
    pub ref_nodes: Vec<usize>,
    pub ref_names: Vec<String>,
    /// Per sample: reference position and velocity, COM position and velocity (world).
    rp: Vec<DVec3>,
    rv: Vec<DVec3>,
    cp: Vec<DVec3>,
    cv: Vec<DVec3>,
}

fn mean(x: &[Vec3], idx: &[usize]) -> DVec3 {
    idx.iter().filter_map(|&i| x.get(i)).map(|p| p.as_dvec3()).sum::<DVec3>() / idx.len().max(1) as f64
}

impl BodyTrack {
    pub fn new(ref_nodes: Vec<usize>, ref_names: Vec<String>) -> Self {
        Self { ref_nodes, ref_names, ..Default::default() }
    }
    /// The reference location's and the centre of mass's velocities at the first sample
    /// (zero before one).
    pub fn first_velocities(&self) -> [DVec3; 2] {
        [self.rv.first().copied().unwrap_or(DVec3::ZERO), self.cv.first().copied().unwrap_or(DVec3::ZERO)]
    }
    pub fn push(&mut self, pos: &[Vec3], vel: &[Vec3], com_p: DVec3, com_v: DVec3) {
        let (p, v) = if self.ref_nodes.is_empty() { (com_p, com_v) } else { (mean(pos, &self.ref_nodes), mean(vel, &self.ref_nodes)) };
        self.rp.push(p);
        self.rv.push(v);
        self.cp.push(com_p);
        self.cv.push(com_v);
    }
}

/// Samples for the whole scene.
#[derive(Default)]
pub(super) struct SceneTrack {
    pub t: Vec<f64>,
    /// Σ m v over every node of every body.
    pub momentum: Vec<DVec3>,
}

/// The value of `y` (on times `t`) at time `at`, linear between samples (clamped).
fn at_time(t: &[f64], y: &[f64], at: f64) -> f64 {
    let i = t.partition_point(|x| *x <= at);
    match (i.checked_sub(1), y.get(i)) {
        (Some(a), Some(&b)) if t[i] > t[a] => y[a] + (b - y[a]) * (at - t[a]) / (t[i] - t[a]),
        (Some(a), _) => y[a],
        (None, Some(&b)) => b,
        _ => 0.0,
    }
}

/// A displacement `x` (on times `t`, ms) at `at`, cubic Hermite between samples on its
/// velocity `v` (m/s): exact for motion at constant acceleration, where the linear form is
/// off by a² Δt²/8 at a turning point (0.8 mm at 40 g and 4 ms).
fn hermite(t: &[f64], x: &[f64], v: &[f64], at: f64) -> f64 {
    let i = t.partition_point(|s| *s <= at);
    let (Some(a), Some(_)) = (i.checked_sub(1), t.get(i)) else { return at_time(t, x, at) };
    let h = (t[i] - t[a]) / 1000.0;
    if h <= 0.0 {
        return x[a];
    }
    let s = (at - t[a]) / 1000.0 / h;
    let (h00, h10, h01, h11) = (2.0 * s.powi(3) - 3.0 * s * s + 1.0, s.powi(3) - 2.0 * s * s + s, -2.0 * s.powi(3) + 3.0 * s * s, s.powi(3) - s * s);
    h00 * x[a] + h10 * h * v[a] + h01 * x[i] + h11 * h * v[i]
}

fn kinematics(t: &[f64], dt: f64, p: &[DVec3], v: &[DVec3], x: DVec3, y: DVec3, t0: f64) -> Kinematics {
    let along = |s: &[DVec3], a: DVec3| s.iter().map(|q| q.dot(a)).collect::<Vec<f64>>();
    let (vx, vy) = (along(v, x), along(v, y));
    let (px, py) = (along(p, x), along(p, y));
    let (px0, py0) = (at_time(t, &px, t0), at_time(t, &py, t0));
    Kinematics {
        ax_g: signal::differentiate(&vx, dt).iter().map(|a| a / G0).collect(),
        ay_g: signal::differentiate(&vy, dt).iter().map(|a| a / G0).collect(),
        dx: px.iter().map(|q| q - px0).collect(),
        dy: py.iter().map(|q| q - py0).collect(),
        vx,
        vy,
    }
}

/// The pulse's scalars from a location's curves (t in ms after contact), as the NHTSA
/// references read them (`engine/tools/crash-reference/nhtsa_reference.py`): `<prefix>.<q>`.
/// - `peak_decel_g`, `peak_ms`: the largest −ax after contact and when;
/// - `stop_ms`, `dyn_crush`, `mean_decel_g`: the first time vx reaches 0 after contact
///   (linear between samples), dx then, and vx at contact / that time (g); only when it
///   stops;
/// - `rebound`, `restitution`: −min vx from the stop to [`REBOUND_UNTIL_S`] after contact
///   (m/s), over vx at contact; only when it stops;
/// - `vy_peak`, `vy_peak_ms`, `ay_peak_g`, `ay_peak_ms`, `dy_peak`: the largest-magnitude
///   lateral velocity (signed, + right) after contact and when, the largest-magnitude lateral
///   acceleration (signed) and when, the largest-magnitude lateral displacement (signed);
/// - `dvy_peak`, `dvy_peak_ms`, `dvx_150ms`, `dvy_150ms`: the largest-magnitude change of vy
///   from contact (signed) and when; vx and vy [`DV_AT_MS`] after contact less at contact
///   (only when the run lasts that long).
pub(super) fn pulse_scalars(put: &mut dyn FnMut(&str, f64), prefix: &str, t_ms: &[f64], k: &Kinematics) {
    let after: Vec<usize> = (0..t_ms.len()).filter(|&i| t_ms[i] >= 0.0).collect();
    let Some(&first) = after.first() else { return };
    let signed_max = |s: &[f64]| after.iter().map(|&i| (i, s[i])).fold((first, 0.0f64), |b, (i, v)| if v.abs() > b.1.abs() { (i, v) } else { b });
    let (ip, peak) = after.iter().map(|&i| (i, -k.ax_g[i])).fold((first, f64::MIN), |b, x| if x.1 > b.1 { x } else { b });
    put(&format!("{prefix}.peak_decel_g"), peak);
    put(&format!("{prefix}.peak_ms"), t_ms[ip]);
    let v0 = at_time(t_ms, &k.vx, 0.0);
    if v0 > 0.0 {
        if let Some(j) = after.iter().copied().find(|&i| k.vx[i] <= 0.0) {
            let ts = match j.checked_sub(1) {
                Some(a) if k.vx[a] > 0.0 => t_ms[a] + (t_ms[j] - t_ms[a]) * k.vx[a] / (k.vx[a] - k.vx[j]),
                _ => t_ms[j],
            };
            let crush = hermite(t_ms, &k.dx, &k.vx, ts);
            let until = 1000.0 * REBOUND_UNTIL_S;
            let rebound = -(j..t_ms.len()).filter(|&i| t_ms[i] <= until).map(|i| k.vx[i]).fold(0.0f64, f64::min);
            put(&format!("{prefix}.stop_ms"), ts);
            put(&format!("{prefix}.dyn_crush"), crush);
            put(&format!("{prefix}.mean_decel_g"), if ts > 0.0 { v0 / (ts / 1000.0) / G0 } else { 0.0 });
            put(&format!("{prefix}.rebound"), rebound);
            put(&format!("{prefix}.restitution"), rebound / v0);
        }
    }
    let (iv, vy) = signed_max(&k.vy);
    put(&format!("{prefix}.vy_peak"), vy);
    put(&format!("{prefix}.vy_peak_ms"), t_ms[iv]);
    let (ia, ay) = signed_max(&k.ay_g);
    put(&format!("{prefix}.ay_peak_g"), ay);
    put(&format!("{prefix}.ay_peak_ms"), t_ms[ia]);
    put(&format!("{prefix}.dy_peak"), signed_max(&k.dy).1);
    // Velocity changes from contact (a struck body's, as its accelerometer integrates them).
    let vy0 = at_time(t_ms, &k.vy, 0.0);
    let dvy: Vec<f64> = k.vy.iter().map(|v| v - vy0).collect();
    let (id, dv) = signed_max(&dvy);
    put(&format!("{prefix}.dvy_peak"), dv);
    put(&format!("{prefix}.dvy_peak_ms"), t_ms[id]);
    if t_ms.last().is_some_and(|&e| e >= DV_AT_MS) {
        put(&format!("{prefix}.dvx_150ms"), at_time(t_ms, &k.vx, DV_AT_MS) - v0);
        put(&format!("{prefix}.dvy_150ms"), at_time(t_ms, &k.vy, DV_AT_MS) - vy0);
    }
}

/// When a struck body's velocity change is read (ms after contact), as the NHTSA references.
pub const DV_AT_MS: f64 = 150.0;

pub(super) struct SeriesInput<'a> {
    pub names: Vec<&'static str>,
    pub masses: Vec<f64>,
    pub headings: Vec<Vec3>,
    /// Body 0's initial velocity's direction (the box force's axis).
    pub travel0: DVec3,
    pub tracks: &'a [&'a BodyTrack],
    pub scene: &'a SceneTrack,
    pub impact_s: f64,
    pub has_boxes: bool,
    /// The per-step trace (per step, per body; the first entry at the first sample), empty
    /// without one, and the step (s).
    pub fine: &'a [Vec<DVec3>],
    pub step_s: f64,
}

/// The series and its scalars (`m`: `<body>.ref.*`, `<body>.com.*`, `all.box_*`).
pub(super) fn finish(i: SeriesInput, m: &mut BTreeMap<String, f64>) -> Series {
    let t = &i.scene.t;
    let dt = if t.len() > 1 { (t[t.len() - 1] - t[0]) / (t.len() - 1) as f64 } else { 1.0 };
    let t_ms: Vec<f64> = t.iter().map(|x| 1000.0 * (x - i.impact_s)).collect();
    let mut bodies = Vec::new();
    for (k, tr) in i.tracks.iter().enumerate() {
        let h = i.headings[k].as_dvec3();
        let right = h.cross(DVec3::Y);
        let reference = kinematics(t, dt, &tr.rp, &tr.rv, h, right, i.impact_s);
        let com = kinematics(t, dt, &tr.cp, &tr.cv, h, right, i.impact_s);
        let name = i.names[k];
        let mut put = |n: &str, v: f64| {
            m.insert(format!("{name}.{n}"), v);
        };
        put("ref.nodes", tr.ref_nodes.len() as f64);
        pulse_scalars(&mut put, "ref", &t_ms, &reference);
        pulse_scalars(&mut put, "com", &t_ms, &com);
        bodies.push(BodySeries { name: name.into(), mass_kg: i.masses[k], ref_nodes: tr.ref_names.clone(), reference, com });
    }
    let mut box_force_n = Vec::new();
    if i.has_boxes && !t.is_empty() {
        let p: Vec<f64> = i.scene.momentum.iter().map(|q| q.dot(i.travel0)).collect();
        box_force_n = signal::differentiate(&p, dt).iter().map(|f| -f).collect();
        let after: Vec<usize> = (0..t.len()).filter(|&k| t_ms[k] >= 0.0).collect();
        if let Some((ip, peak)) = after.iter().map(|&k| (k, box_force_n[k])).reduce(|a, b| if b.1 > a.1 { b } else { a }) {
            m.insert("all.box_force_peak_n".into(), peak);
            m.insert("all.box_force_peak_ms".into(), t_ms[ip]);
        }
        let pre: Vec<f64> = (0..t.len()).filter(|&k| t_ms[k] < 0.0 && t_ms[k] >= -1000.0 * PRE_S).map(|k| box_force_n[k].abs()).collect();
        m.insert("all.box_force_pre_n".into(), if pre.is_empty() { 0.0 } else { pre.iter().sum::<f64>() / pre.len() as f64 });
        // The impulse from first contact to the end: the momentum lost along the travel.
        m.insert("all.box_impulse_ns".into(), at_time(&t_ms, &p, 0.0) - p[p.len() - 1]);
    }
    let fine = fine(&i, m);
    Series { t_ms, bodies, fine, box_force_n, method: "4 ms samples; velocity: mean of the location's nodes (COM: mass-weighted); displacement: positions from first contact; acceleration: central difference of the velocity samples (mean over 8 ms); box force: -d(sum m v)/dt along body 0's initial velocity".into() }
}

/// [`Fine`] from the per-step trace (per step: each body's reference location, then its
/// centre of mass), and its scalars (`<body>.{ref,com}.cfc60_*`): the largest CFC 60
/// deceleration along x after contact and when, the largest-magnitude lateral one (signed) and
/// when.
fn fine(i: &SeriesInput, m: &mut BTreeMap<String, f64>) -> Option<Fine> {
    let n = i.fine.len();
    if n < 3 || i.fine.iter().any(|x| x.len() != 2 * i.names.len()) {
        return None;
    }
    let dt = i.step_s;
    let t0 = i.scene.t.first().copied().unwrap_or(0.0);
    let t_ms: Vec<f64> = (0..n).map(|k| 1000.0 * (t0 + k as f64 * dt - i.impact_s)).collect();
    let every = ((0.001 / dt).round() as usize).max(1);
    let mut bodies = Vec::new();
    for (b, name) in i.names.iter().enumerate() {
        let h = i.headings[b].as_dvec3();
        let right = h.cross(DVec3::Y);
        let mut loc = |slot: usize, prefix: &str| -> Option<FineKinematics> {
            let vx: Vec<f64> = i.fine.iter().map(|x| x[2 * b + slot].dot(h)).collect();
            let vy: Vec<f64> = i.fine.iter().map(|x| x[2 * b + slot].dot(right)).collect();
            let filt = |v: &[f64]| signal::cfc(&signal::differentiate(v, dt), dt, 60.0).map(|a| a.iter().map(|x| x / G0).collect::<Vec<f64>>());
            let (ax, ay) = (filt(&vx).ok()?, filt(&vy).ok()?);
            let after = (0..n).filter(|&k| t_ms[k] >= 0.0);
            if let Some((k, a)) = after.clone().map(|k| (k, -ax[k])).reduce(|p, q| if q.1 > p.1 { q } else { p }) {
                m.insert(format!("{name}.{prefix}.cfc60_peak_decel_g"), a);
                m.insert(format!("{name}.{prefix}.cfc60_peak_ms"), t_ms[k]);
            }
            if let Some((k, a)) = after.map(|k| (k, ay[k])).reduce(|p, q| if q.1.abs() > p.1.abs() { q } else { p }) {
                m.insert(format!("{name}.{prefix}.cfc60_ay_peak_g"), a);
                m.insert(format!("{name}.{prefix}.cfc60_ay_peak_ms"), t_ms[k]);
            }
            let dec = |x: &[f64]| x.iter().step_by(every).copied().collect::<Vec<f64>>();
            Some(FineKinematics { ax_g: dec(&ax), ay_g: dec(&ay), vx: dec(&vx), vy: dec(&vy) })
        };
        let reference = loc(0, "ref")?;
        let com = loc(1, "com")?;
        bodies.push(FineBody { name: name.to_string(), reference, com });
    }
    Some(Fine { rate_hz: 1.0 / dt, curve_hz: 1.0 / (dt * every as f64), t_ms: t_ms.iter().step_by(every).copied().collect(), bodies })
}
