//! Layer 2 of the vehicle test strategy (`engine/docs/vehicle-test-strategy.md`):
//! physical invariants of full cars in standard scenes, each a reusable checker
//! with its threshold's source or derivation (anything else labelled a Choice).
//! The authored `sedan` and `sedan1` at Medium on the CPU reference. Each scene
//! runs once per car per test binary (cached) and every test asserts one
//! invariant of it.
//! Run: `cargo test --release -p engine-softbody --test crash_invariants -- --nocapture --include-ignored`
//! (the `#[ignore]`d ones are real failures, their measured values in comments).

use engine_softbody::authoring::skin::{
    deform, deform_beams, showcase_visual, unit_states, BeamState, ShowcaseVisual,
};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, STEP,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_skin::closest_on_triangle;
use glam::{DQuat, DVec3, Vec3};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

// --- Thresholds (each with its source) ------------------------------------------

/// Plastic strain counts past the 0.2 % offset: the engineering yield definition
/// (ASTM E8/E8M, the 0.2 % offset proof strain). Below it a beam's rest-length
/// change is not a visible set.
const PLASTIC: f32 = 2e-3;
/// Relative speed above which node-against-triangle contact can be crossed in one
/// step: the band r/h = 0.02 m / 0.5 ms (layer 1: half of the crossings above it
/// pass through).
const BAND_SPEED: f32 = 40.0;
/// Panel ripple visible: 1 mm of displacement amplitude about a part's rigid
/// motion. **Choice**: about half a pixel at 2 m on a 1080p, 60° view
/// (2 m · tan 30° / 540 ≈ 2 mm per pixel).
const VISIBLE: f64 = 1e-3;
/// Measured first (out-of-plane) modes of production hoods on the car (Hz): the
/// lowest and highest published. 17.7 Hz: a front hood's first torsion,
/// experimental modal analysis (MDPI *Machines* 12(12) 862, 2024; its other
/// first-torsion estimate 22.9 Hz); 41.5 Hz: the least first mode over a
/// lightweight hood's design variants (modal analysis with stamping effects).
/// Design targets quoted: 25–30 Hz and "above 40 Hz". A decklid is the same
/// construction (outer, hemmed inner, hinges and a latch): the same range,
/// a Choice.
const LID_MODE_HZ: [f64; 2] = [17.7, 41.5];
/// The boot lid relative to the rear body in a 56 km/h frontal: the
/// quasi-static deflection a/ω² of a lid at the measured range's lowest mode
/// under the measured pulse's top, a = 47 g (`crash-reference-targets.md` §8):
/// 47 · 9.807 / (2π · 17.7)² = 37 mm. (The pulse's 63–76 ms rise is long
/// against a mode above 15 Hz, so the lid follows it quasi-statically.)
const LID_REL: f64 = 47.0 * 9.807
    / ((2.0 * std::f64::consts::PI * LID_MODE_HZ[0])
        * (2.0 * std::f64::consts::PI * LID_MODE_HZ[0]));
/// The rear body's own deformation (RMS of its nodes about their rigid fit) in a
/// 56 km/h frontal. **Choice**: 5 mm, the same reasoning (the rear third carries
/// only its own inertia, ≤ 47 g).
const REAR_RIGID: f64 = 5e-3;
/// The visual skin's offset from the physics collision skin may grow by at most the
/// contact band (2 cm, `DEFAULT_COLLISION_RANGE`): beyond it, what is drawn is not
/// where contact happens.
const VISUAL_BAND: f64 = 0.02;

// --- Cars and scenes -----------------------------------------------------------

fn built(name: &str) -> Arc<Built> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Built>>>> = OnceLock::new();
    let m = CACHE.get_or_init(Default::default);
    if let Some(b) = m.lock().unwrap().get(name) {
        return b.clone();
    }
    let b = Arc::new(
        build(
            &style(name).unwrap(),
            &BuildParams {
                resolution: Resolution::Medium,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    m.lock().unwrap().insert(name.into(), b.clone());
    b
}

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

/// Standing on its tyres (the file is the car at rest), moving at `vel`.
fn spawn_truck(truck: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let probe = ExplicitBeams::from_truck(truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(
        truck,
        at + Vec3::new(0.0, probe.spawn_clearance() - low, 0.0),
        vel,
        settings(),
    )
    .unwrap();
    s.roll_tyres(vel);
    // (CI_ANTI: `0` or `1` sets anti-clip, spec §36; measurements only.)
    if let Ok(a) = std::env::var("CI_ANTI") {
        s.anti_clip = a == "1";
    }
    // (CI_DROP: `1` puts back RoR's strength drop with plastic stretch; CI_PSET:
    // the plastic coefficient of every deformable beam; spec §36, measurements.)
    if std::env::var("CI_DROP").as_deref() == Ok("1") {
        s.plastic_strength_drop = true;
    }
    if let Some(p) = std::env::var("CI_PSET").ok().and_then(|v| v.parse::<f32>().ok()) {
        for b in &mut s.beams {
            if b.ty == engine_softbody::explicit_beams::ExplicitType::Deformable {
                b.plastic = p;
            }
        }
    }
    s
}

/// The car turned 180° about y (its front facing −z).
fn turned_truck(b: &Built) -> String {
    let mut sk = b.skeleton.clone();
    let len = b.def.profile.length;
    let turn = |p: Vec3| {
        let q = p - Vec3::new(0.0, 0.0, 0.5 * len);
        Vec3::new(-q.x, q.y, -q.z) + Vec3::new(0.0, 0.0, 0.5 * len)
    };
    for nd in &mut sk.nodes {
        nd.p = turn(nd.p);
    }
    // (The tyres' rest poses too: they are in the same coordinates.)
    for w in &mut sk.tyre_pose {
        for p in w.iter_mut() {
            *p = turn(*p);
        }
    }
    for a in &mut sk.tyre_rest_axle {
        for p in a.iter_mut() {
            *p = turn(*p);
        }
    }
    sk.to_truck()
}

// --- A car under the scenes: authored, or BeamNG's own (opt-in) -----------------

/// The scenes' view of a car: an authored build, or a BeamNG car's JBeam
/// (`bx`, opt-in through `RUBBLE_JBEAM_BX`) run in JBeam mode on pressure
/// wheels. Node order is the solver's (the desc's, then the wheels').
struct SceneCar {
    built: Option<Arc<Built>>,
    /// The JBeam car settled 2 s at rest, facing +z.
    settled: Option<ExplicitBeams>,
    /// The structure's nodes (wheels after).
    n: usize,
    len: f32,
    /// Each node's part, by the authored car's names (`body`, `bootlid`,
    /// `door_f_l`, `suspension` …; a BeamNG car's parts are mapped onto them,
    /// a Choice per part, else keep their own names).
    part: Vec<String>,
    /// Breakable hardware: (name, solver beam indices).
    hardware: Vec<(String, Vec<usize>)>,
    /// The far (right) doors' and the lid's hardware name prefixes.
    far_doors: Vec<String>,
    lid: String,
}

impl SceneCar {
    fn authored(b: Arc<Built>) -> Self {
        let sk = &b.skeleton;
        SceneCar {
            n: sk.nodes.len(),
            len: b.def.profile.length,
            part: sk
                .nodes
                .iter()
                .map(|q| sk.parts[q.part as usize].clone())
                .collect(),
            hardware: b
                .hardware
                .iter()
                .map(|h| (h.name.clone(), h.beams.clone()))
                .collect(),
            far_doors: vec!["door_f_r".into(), "door_r_r".into()],
            lid: "bootlid".into(),
            settled: None,
            built: Some(b),
        }
    }

    /// The BX coupe on its slot defaults (`showcase_car.rs`'s `beamng_*`), with
    /// the donor tyre when `RUBBLE_BEAMNG_MODS` is set. None without the content.
    fn beamng() -> Option<Self> {
        use engine_softbody::explicit_beams::JbeamWheels;
        use engine_softbody::reference_cars as rc;
        let dir = std::env::var("RUBBLE_JBEAM_BX").ok()?;
        let pc = r#"{"format":2,"model":"bx","parts":{"bx_body":"bx_body_coupe"}}"#;
        let mut desc = rc::load(std::path::Path::new(&dir), "bx", Some(pc)).ok()?;
        if let Some(d) = std::env::var("RUBBLE_BEAMNG_MODS")
            .ok()
            .and_then(|m| rc::donor_tyre_car(std::path::Path::new(&m)))
        {
            rc::fit_donor_tyre(&mut desc, &d);
        }
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
            &desc,
            JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings(),
        )
        .ok()?;
        let f = engine_softbody::pack::forward_axis(&desc);
        let turn = glam::Quat::from_rotation_y(-f.x.atan2(f.z));
        for nd in &mut s.nodes {
            nd.pos = turn * nd.pos;
        }
        let low = s.nodes.iter().map(|q| q.pos.y).fold(f32::MAX, f32::min);
        let mid = s.nodes.iter().map(|q| q.pos).sum::<Vec3>() / s.nodes.len() as f32;
        for nd in &mut s.nodes {
            nd.pos -= Vec3::new(mid.x, low - 0.02, mid.z);
        }
        s.run(2.0);
        let n = desc.nodes.len();
        // Each node's first part (the desc lists a node under every part that
        // names it; the selected body first).
        let mut own: Vec<Option<&str>> = vec![None; n];
        for p in &desc.parts {
            for &i in &p.nodes {
                own[i as usize].get_or_insert(p.name.as_str());
            }
        }
        let map = |p: &str| -> String {
            let chassis = [
                "suspension",
                "hub",
                "subframe",
                "arm",
                "tierod",
                "strut",
                "shock",
                "swaybar",
                "steering",
                "halfshaft",
                "differential",
                "mounts",
            ];
            if p.starts_with("bx_body") || p == "bx_floor" {
                "body".into()
            } else if p.starts_with("bx_trunk") {
                "bootlid".into()
            } else if p.starts_with("bx_hood") {
                "bonnet".into()
            } else if p.starts_with("bx_fender_L") {
                "fender_l".into()
            } else if p == "bx_door_L" || p == "bx_doorpanel_L" {
                "door_f_l".into()
            } else if p == "bx_door_R" || p == "bx_doorpanel_R" {
                "door_f_r".into()
            } else if chassis.iter().any(|c| p.contains(c)) {
                "suspension".into()
            } else {
                p.trim_start_matches("bx_").into()
            }
        };
        let part = (0..n).map(|i| own[i].map_or("other".into(), map)).collect();
        let mut hw: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for j in 0..desc.beams.len() {
            for g in desc.beam_group_names(j).1 {
                hw.entry(g.to_string()).or_default().push(j);
            }
        }
        let len = {
            let (a, c) = s.nodes[..n].iter().fold((f32::MAX, f32::MIN), |(a, c), q| {
                (a.min(q.pos.z), c.max(q.pos.z))
            });
            c - a
        };
        Some(SceneCar {
            built: None,
            settled: Some(s),
            n,
            len,
            part,
            hardware: hw.into_iter().collect(),
            // (Its door latches and trunk latch are couplers, not break groups:
            // the hinges are what is counted.)
            far_doors: vec!["doorhinge_a_R".into(), "doorhinge_b_R".into()],
            lid: "trunkhinge".into(),
        })
    }

    /// Standing at `at` (its centre along and across, as the authored spawn),
    /// moving at `vel`; `turned`: facing −z.
    fn spawn(&self, at: Vec3, vel: Vec3, turned: bool) -> ExplicitBeams {
        if let Some(b) = &self.built {
            return if turned {
                spawn_truck(&turned_truck(b), at, vel)
            } else {
                spawn_truck(&b.truck, at, vel)
            };
        }
        let mut s = self.settled.clone().unwrap();
        if turned {
            let q = glam::Quat::from_rotation_y(std::f32::consts::PI);
            for nd in &mut s.nodes {
                nd.pos = q * nd.pos;
                nd.vel = q * nd.vel;
            }
        }
        for nd in &mut s.nodes {
            nd.pos += at;
            nd.vel += vel;
        }
        s.roll_tyres(vel);
        s
    }

    fn label(&self, a: u32, c: u32) -> String {
        if let Some(b) = &self.built {
            return label(b, a, c);
        }
        let p = |i: u32| self.part.get(i as usize).map_or("tyre", |x| x.as_str());
        let (x, y) = (p(a), p(c));
        if x == y {
            x.into()
        } else {
            format!("{x}~{y}")
        }
    }

    fn hardware_broken(&self, s: &ExplicitBeams, prefix: &str) -> Vec<String> {
        self.hardware
            .iter()
            .filter(|h| h.0.starts_with(prefix))
            .filter(|h| h.1.iter().any(|&j| s.beams[j].broken))
            .map(|h| h.0.clone())
            .collect()
    }
}

fn scene_car(name: &str) -> Arc<SceneCar> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<SceneCar>>>> = OnceLock::new();
    let m = CACHE.get_or_init(Default::default);
    if let Some(c) = m.lock().unwrap().get(name) {
        return c.clone();
    }
    let c = Arc::new(if name == "bx" {
        SceneCar::beamng().expect("RUBBLE_JBEAM_BX: no bx")
    } else {
        SceneCar::authored(built(name))
    });
    m.lock().unwrap().insert(name.into(), c.clone());
    c
}

// --- Reusable checkers ------------------------------------------------------------

/// Best rigid fit (Horn 1987, unit quaternions; Jacobi eigen-solve):
/// the rotation R and centroids (c₀, c) with x ≈ R·(X − c₀) + c over `ids`, mass
/// weighted.
fn rigid_fit(
    rest: &[DVec3],
    now: &[DVec3],
    w: &[f64],
    ids: &[usize],
    q0: DQuat,
) -> (DQuat, DVec3, DVec3) {
    let tw: f64 = ids.iter().map(|&i| w[i]).sum();
    let c0 = ids.iter().map(|&i| rest[i] * w[i]).sum::<DVec3>() / tw;
    let c = ids.iter().map(|&i| now[i] * w[i]).sum::<DVec3>() / tw;
    let mut s = [[0.0f64; 3]; 3];
    for &i in ids {
        let (a, b) = ((rest[i] - c0).to_array(), (now[i] - c).to_array());
        for r in 0..3 {
            for k in 0..3 {
                s[r][k] += w[i] * a[r] * b[k];
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let q = top_eigenvector(n);
    let _ = q0;
    (DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize(), c0, c)
}

/// The eigenvector of the largest eigenvalue of a symmetric 4×4 (cyclic Jacobi).
fn top_eigenvector(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [[0.0f64; 4]; 4];
    for (i, r) in v.iter_mut().enumerate() {
        r[i] = 1.0;
    }
    for _ in 0..50 {
        let off: f64 = (0..4)
            .flat_map(|i| (0..4).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for r in v.iter_mut() {
                    let (vp, vq) = (r[p], r[q]);
                    r[p] = c * vp - s * vq;
                    r[q] = s * vp + c * vq;
                }
            }
        }
    }
    let k = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap();
    [v[0][k], v[1][k], v[2][k], v[3][k]]
}

/// Each node's displacement about a rigid fit: R⁻¹(x − c) − (X − c₀).
fn residuals(
    rest: &[DVec3],
    now: &[DVec3],
    fit: (DQuat, DVec3, DVec3),
    of: &[usize],
) -> Vec<DVec3> {
    let inv = fit.0.inverse();
    of.iter()
        .map(|&i| inv * (now[i] - fit.2) - (rest[i] - fit.1))
        .collect()
}

type Plastic = (usize, usize, f32, BTreeMap<String, usize>);

/// Beams past [`PLASTIC`] or broken among `beams` (solver indices): (count past,
/// broken, worst strain, by member or part pair).
fn plastic(s: &ExplicitBeams, car: &SceneCar, beams: &[usize]) -> Plastic {
    let (mut n, mut br, mut worst) = (0, 0, 0.0f32);
    let mut by = BTreeMap::new();
    for &j in beams {
        let bm = &s.beams[j];
        let strain = (bm.rest - bm.rest0).abs() / bm.rest0.abs().max(1e-6);
        if bm.broken {
            br += 1;
        } else if strain > PLASTIC {
            n += 1;
            worst = worst.max(strain);
        } else {
            continue;
        }
        // Stretched (+, a tension yield) or shortened (−), or broken (x).
        let sign = if bm.broken {
            "x"
        } else if bm.rest > bm.rest0 {
            "+"
        } else {
            "-"
        };
        *by.entry(format!("{}{sign}", car.label(bm.a, bm.b)))
            .or_default() += 1;
    }
    (n, br, worst, by)
}

/// A beam's place: its member group if both ends are in one, else its parts.
fn label(b: &Built, a: u32, c: u32) -> String {
    let mut keys: Vec<&String> = b.groups.keys().collect();
    keys.sort();
    for k in keys {
        let g = &b.groups[k];
        if g.contains(&a) && g.contains(&c) {
            return k.clone();
        }
    }
    let sk = &b.skeleton;
    let p = |i: u32| {
        sk.nodes
            .get(i as usize)
            .map(|n| sk.parts[n.part as usize].as_str())
            .unwrap_or("tyre")
    };
    let (x, y) = (p(a), p(c));
    if x == y {
        x.into()
    } else {
        format!("{x}~{y}")
    }
}

fn finite(s: &ExplicitBeams) -> bool {
    s.nodes
        .iter()
        .all(|n| n.pos.is_finite() && n.vel.is_finite())
}

fn dpos(s: &ExplicitBeams) -> Vec<DVec3> {
    s.positions().iter().map(|p| p.as_dvec3()).collect()
}

/// The visual skin's offset from the collision skin, sampled (every 37th visual
/// vertex on the exterior): (offset at rest, offset now) per live sample.
fn visual_offsets(
    v: &ShowcaseVisual,
    b: &Built,
    s: &ExplicitBeams,
    rest: &[Vec3],
) -> Vec<(f64, f64)> {
    let tris: Vec<[u32; 3]> = s.triangles.iter().map(|t| t.nodes).collect();
    let now = s.positions();
    // As drawn: a shattered pane or lens is gone (the renderer drops it).
    let beams: Vec<BeamState> = s
        .beams
        .iter()
        .map(|b| BeamState {
            broken: b.broken,
            rest: b.rest,
        })
        .collect();
    let units = unit_states(v, &beams);
    let (p_rest, _) = deform(v, rest);
    let (p_now, _) = deform_beams(v, &now, Some(&units), Some(&beams));
    let dist = |p: Vec3, pos: &[Vec3]| {
        tris.iter()
            .map(|t| {
                let (q, _, _) = closest_on_triangle(
                    p,
                    pos[t[0] as usize],
                    pos[t[1] as usize],
                    pos[t[2] as usize],
                );
                p.distance(q)
            })
            .fold(f32::MAX, f32::min) as f64
    };
    // Wheels ride their tyres (which turn and steer), not the body's skin.
    let sk_nodes = b.skeleton.nodes.len() as u32;
    let on_body = |i: usize| v.bind_nodes[i].iter().all(|&k| k < sk_nodes);
    if std::env::var("CI_DEBUG").is_ok() {
        let on_tri: std::collections::HashSet<u32> = tris.iter().flatten().copied().collect();
        let mut w: Vec<(f64, usize)> = (0..p_now.len())
            .step_by(37)
            .filter(|&i| p_now[i].is_finite() && p_rest[i].is_finite() && on_body(i))
            .map(|i| (dist(p_now[i], &now) - dist(p_rest[i], rest), i))
            .filter(|&(_, i)| dist(p_rest[i], rest) <= VISUAL_BAND)
            .collect();
        w.sort_by(|a, c| c.0.total_cmp(&a.0));
        // Shut-line frames (nodes of two parts) stretched now: drawn as a
        // sheet across an opening gap until the renderer's tear (2.5×).
        let stretch = |t: [u32; 3]| {
            let e = |a: u32, c: u32, p: &[Vec3]| p[a as usize].distance(p[c as usize]);
            [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
                .iter()
                .map(|&(a, c)| e(a, c, &now) / e(a, c, rest).max(1e-6))
                .fold(0.0f32, f32::max)
        };
        let parts_of = |t: [u32; 3]| t.map(|k| b.skeleton.nodes.get(k as usize).map(|n| n.part));
        let (mut mixed, mut s15, mut s25) = (0, 0, 0);
        for i in (0..p_now.len())
            .step_by(37)
            .filter(|&i| on_body(i) && p_now[i].is_finite())
        {
            let t = v.bind_nodes[i];
            let ps = parts_of(t);
            if ps[0] != ps[1] || ps[1] != ps[2] {
                mixed += 1;
                let x = stretch(t);
                s15 += usize::from(x > 1.5);
                s25 += usize::from(x > 2.5);
            }
        }
        eprintln!(
            "  shut-line frames: {mixed} samples, stretched > 1.5x {s15}, > 2.5x (torn) {s25}"
        );
        // The fine skin's own offset growth from the collision skin (the
        // floor of any binding to it): its triangles' centroids.
        let cen = |t: &[u32; 3], p: &[Vec3]| {
            (p[t[0] as usize] + p[t[1] as usize] + p[t[2] as usize]) / 3.0
        };
        let mut nd: Vec<(f64, usize)> = b
            .skeleton
            .skin
            .iter()
            .enumerate()
            .filter(|(_, t)| cen(t, &now).is_finite())
            .map(|(k, t)| (dist(cen(t, &now), &now) - dist(cen(t, rest), rest), k))
            .collect();
        nd.sort_by(|a, c| c.0.total_cmp(&a.0));
        for &(d, k) in nd.iter().take(3) {
            let t = b.skeleton.skin[k];
            let k = t[0];
            eprintln!(
                "  skin tri {t:?} centroid +{:.0} mm part {}",
                d * 1e3,
                b.skeleton.parts[b.skeleton.nodes[k as usize].part as usize]
            );
        }
        for &(d, i) in w.iter().take(5) {
            let bn = v.bind_nodes[i];
            eprintln!(
                "  visual vertex {i} +{:.0} mm, material {}, bound to {:?} parts {:?} on collision tris {:?}, h {:.3}",
                d * 1e3,
                v.material[i],
                bn,
                bn.map(|k| b.skeleton.nodes.get(k as usize).map(|q| b.skeleton.parts[q.part as usize].clone()).unwrap_or_default()),
                bn.map(|k| on_tri.contains(&k)),
                v.bind_coords[i][2]
            );
            let mut st = bn;
            st.sort_unstable();
            let is_skin = b.skeleton.skin.iter().any(|t| {
                let mut u = *t;
                u.sort_unstable();
                u == st
            });
            let (q, _, _) = closest_on_triangle(
                p_rest[i],
                rest[bn[0] as usize],
                rest[bn[1] as usize],
                rest[bn[2] as usize],
            );
            eprintln!(
                "    coords {:?} rest distance to frame {:.0} mm, skin triangle {is_skin}, rigid {}, piece {:?}",
                v.bind_coords[i],
                p_rest[i].distance(q) * 1e3,
                v.rigid[i],
                v.pieces.iter().find(|(_, f, c)| (i as u32) >= *f && (i as u32) < f + c).map(|p| p.0.clone())
            );
        }
    }
    // The exterior skin's vertices: those within the band of the collision skin at
    // rest (interior trim, seats and the like are not on it).
    (0..p_now.len())
        .step_by(37)
        .filter(|&i| p_now[i].is_finite() && p_rest[i].is_finite() && on_body(i))
        .map(|i| (dist(p_rest[i], rest), dist(p_now[i], &now)))
        .filter(|&(d0, _)| d0 <= VISUAL_BAND)
        .collect()
}

// --- The measurements of one run ----------------------------------------------------

#[derive(Clone, Debug, Default)]
struct Run {
    finite: bool,
    /// The ledger: largest rise of the mechanical energy over the start (J), the
    /// start (J), the worst closure (J).
    energy_rise: f64,
    energy_start: f64,
    closure: f64,
    /// Rear third: beams past 0.2 %, broken, worst strain, by member.
    rear: Plastic,
    rear_beams: usize,
    /// Rear body nodes' RMS and largest displacement about their rigid fit
    /// (worst over the run, m).
    rear_rms: f64,
    rear_max: f64,
    /// Boot lid nodes' largest displacement relative to the rear body's rigid fit
    /// (worst over the run and at the end, m), and its hardware broken.
    lid_peak: f64,
    lid_end: f64,
    lid_hw: Vec<String>,
    /// Per part, its ripple amplitude (RMS over its nodes of √2·σ of their
    /// displacement about the part's rigid fit, sampled at 1 kHz), per 0.1 s window.
    ripple: BTreeMap<String, Vec<f64>>,
    /// Two-car scenes: car 0's node-samples within 0.3 m of a car-1 node with
    /// relative speed > BAND_SPEED, of all such samples; the fastest.
    fast: (u64, u64, f32),
    /// Far side (side pole): beams past 0.2 % or broken; far doors' hardware broken.
    far: Plastic,
    far_hw: Vec<String>,
    /// Whole car: (broken, past 0.2 %), by member.
    all: (usize, usize),
    all_by: BTreeMap<String, usize>,
    /// At rest: horizontal centre-of-mass drift and sway over the last 4 s (m).
    creep: f64,
    sway: f64,
    /// Visual vs collision skin: worst growth of the offset (m).
    visual: f64,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Scene {
    Rest,
    Drive,
    Wall,
    HeadOn,
    Pole,
    /// Diagnosis: the same with the solver's per-node air drag off.
    DriveNoDrag,
    WallNoDrag,
}

type Slot = Arc<Mutex<Option<Arc<Run>>>>;

fn scene(name: &str, sc: Scene) -> Arc<Run> {
    static CACHE: OnceLock<Mutex<HashMap<(String, Scene), Slot>>> = OnceLock::new();
    let slot = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .entry((name.into(), sc))
        .or_default()
        .clone();
    let mut g = slot.lock().unwrap();
    if let Some(r) = g.as_ref() {
        return r.clone();
    }
    let r = Arc::new(run_scene(name, sc));
    let mut short = (*r).clone();
    short.ripple.clear();
    eprintln!("== {name} {sc:?}: {short:?}");
    *g = Some(r.clone());
    r
}

fn run_scene(name: &str, sc: Scene) -> Run {
    let car = scene_car(name);
    let n = car.n;
    let len = car.len;
    let v56 = 56.0 / 3.6;
    let (vehicles, seconds) = match sc {
        Scene::Rest => (vec![car.spawn(Vec3::ZERO, Vec3::ZERO, false)], 6.0),
        Scene::Drive | Scene::DriveNoDrag => {
            (vec![car.spawn(Vec3::ZERO, Vec3::Z * 15.0, false)], 2.0)
        }
        Scene::Wall | Scene::WallNoDrag => {
            let mut s = car.spawn(Vec3::ZERO, Vec3::Z * v56, false);
            let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
            s.boxes.push(StaticBox::centered(
                Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
                Vec3::new(3.0, 2.0, 1.0),
            ));
            (vec![s], 1.5)
        }
        Scene::HeadOn => {
            // The car at 56 km/h into an identical car standing, front to front,
            // with two-body contact (jbeam-semantics-spec.md §10.3).
            let mut s = car.spawn(Vec3::ZERO, Vec3::Z * v56, false);
            let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
            let probe = car.spawn(Vec3::ZERO, Vec3::ZERO, true);
            let back = probe
                .positions()
                .iter()
                .map(|p| p.z)
                .fold(f32::MAX, f32::min);
            let mut o = car.spawn(Vec3::new(0.0, 0.0, front + 0.05 - back), Vec3::ZERO, true);
            s.two_body_contact = true;
            o.two_body_contact = true;
            (vec![s, o], 1.5)
        }
        Scene::Pole => {
            // 32 km/h sideways into a 254 mm pole (FMVSS 214's) at the front door.
            let v = Vec3::X * (32.0 / 3.6);
            let mut s = car.spawn(Vec3::ZERO, v, false);
            let p = s.positions();
            let door: Vec<usize> = (0..n).filter(|&i| car.part[i] == "door_f_l").collect();
            let zd = door.iter().map(|&i| p[i].z).sum::<f32>() / door.len() as f32;
            let side = p.iter().map(|q| q.x).fold(f32::MIN, f32::max);
            s.shapes.push(StaticShape::cylinder(
                Vec3::new(side + 0.03 + 0.127, 2.0, zd),
                0.127,
                2.0,
            ));
            (vec![s], 1.0)
        }
    };
    let mut vehicles = vehicles;
    if matches!(sc, Scene::DriveNoDrag | Scene::WallNoDrag) {
        for v in &mut vehicles {
            v.settings.drag = false;
        }
    }
    let rest_pos = vehicles[0].positions();
    let rest = dpos(&vehicles[0]);
    let mass: Vec<f64> = vehicles[0].nodes.iter().map(|q| q.mass as f64).collect();
    let part = |i: usize| car.part[i].as_str();
    // Along the car (the front is +z).
    let z = |i: usize| rest[i].z as f32;
    let zmin = (0..n).map(z).fold(f32::MAX, f32::min);
    let rear_cut = zmin + len / 3.0;
    let rear_body: Vec<usize> = (0..n)
        .filter(|&i| part(i) == "body" && z(i) < rear_cut)
        .collect();
    let lid: Vec<usize> = (0..n).filter(|&i| part(i) == "bootlid").collect();
    let in_car = |j: usize, f: &dyn Fn(usize) -> bool| {
        let bm = &vehicles[0].beams[j];
        let (a, c) = (bm.a as usize, bm.b as usize);
        a < n && c < n && f(a) && f(c) && part(a) != "suspension" && part(c) != "suspension"
    };
    let rear_beams: Vec<usize> = (0..vehicles[0].beams.len())
        .filter(|&j| in_car(j, &|i| z(i) < rear_cut))
        .collect();
    let mut parts: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        parts.entry(part(i).into()).or_default().push(i);
    }
    // Far side for the pole: x below the centre line by 0.3 m (the pole is at +x).
    let xmid = (0..n).map(|i| rest[i].x).sum::<f64>() / n as f64;
    let far_beams: Vec<usize> = (0..vehicles[0].beams.len())
        .filter(|&j| in_car(j, &|i| rest[i].x < xmid - 0.3))
        .collect();

    let mut w = ExplicitBeamsWorld::new(vehicles);
    w.enable_ledger();
    let mut r = Run {
        rear_beams: rear_beams.len(),
        ..Default::default()
    };
    let start = w.ledger.as_ref().unwrap().start;
    r.energy_start = start;
    let ke0 = w.ledger.as_ref().unwrap().kinetic;
    let mut q_rear = DQuat::IDENTITY;
    let mut q_part: BTreeMap<String, DQuat> = BTreeMap::new();
    // Per part, per node, the residual samples of the current 0.1 s window.
    let mut win: BTreeMap<String, Vec<Vec<DVec3>>> = BTreeMap::new();
    // 1 kHz: modes up to 500 Hz are seen (not aliased into the window's mean).
    let sample_every = 2;
    let steps = (seconds / STEP).round() as usize;
    let window = (0.1 / STEP).round() as usize;
    let mut com_track: Vec<DVec3> = Vec::new();
    // Visual drift: sampled every 0.1 s through the crash (its worst).
    let visual = car
        .built
        .as_ref()
        .filter(|_| matches!(sc, Scene::Wall | Scene::WallNoDrag | Scene::Pole))
        .map(|b| (showcase_visual(b).unwrap(), b.clone()));
    for step in 1..=steps {
        w.step();
        if let Some((v, b)) = &visual {
            if step % window == 0 || step == steps {
                let off = visual_offsets(v, b, &w.vehicles[0], &rest_pos);
                r.visual = off.iter().map(|(a, c)| c - a).fold(r.visual, f64::max);
            }
        }
        let l = w.ledger.as_ref().unwrap();
        r.energy_rise = r.energy_rise.max(l.mechanical() - start);
        r.closure = r
            .closure
            .max(((l.kinetic - ke0) - l.kinetic_change()).abs());
        if w.vehicles.len() == 2 && step % 4 == 0 {
            let (a, o) = (&w.vehicles[0], &w.vehicles[1]);
            let po = o.positions();
            let (lo, hi) = po
                .iter()
                .fold((Vec3::MAX, Vec3::MIN), |(l, h), p| (l.min(*p), h.max(*p)));
            for (i, p) in a.positions().iter().enumerate() {
                if p.cmplt(lo - 0.3).any() || p.cmpgt(hi + 0.3).any() {
                    continue;
                }
                let (j, d) = po
                    .iter()
                    .enumerate()
                    .map(|(j, q)| (j, q.distance_squared(*p)))
                    .fold((0, f32::MAX), |x, y| if y.1 < x.1 { y } else { x });
                if d > 0.3 * 0.3 {
                    continue;
                }
                let rel = (a.nodes[i].vel - o.nodes[j].vel).length();
                r.fast.1 += 1;
                if rel > BAND_SPEED {
                    r.fast.0 += 1;
                }
                r.fast.2 = r.fast.2.max(rel);
            }
        }
        if step % sample_every != 0 {
            continue;
        }
        let s = &w.vehicles[0];
        let now = dpos(s);
        if sc == Scene::Rest && step * 3 >= steps && step % 200 == 0 {
            let tm: f64 = mass.iter().sum();
            com_track.push((0..now.len()).map(|i| now[i] * mass[i]).sum::<DVec3>() / tm);
        }
        // The rear body about its rigid fit, and the lid in that frame.
        let fit = rigid_fit(&rest, &now, &mass, &rear_body, q_rear);
        q_rear = fit.0;
        let res = residuals(&rest, &now, fit, &rear_body);
        let rms = (res.iter().map(|d| d.length_squared()).sum::<f64>() / res.len() as f64).sqrt();
        r.rear_rms = r.rear_rms.max(rms);
        r.rear_max = r
            .rear_max
            .max(res.iter().map(|d| d.length()).fold(0.0, f64::max));
        let lid_now = residuals(&rest, &now, fit, &lid)
            .iter()
            .map(|d| d.length())
            .fold(0.0, f64::max);
        if let (Ok(_), Some(b)) = (std::env::var("CI_DEBUG"), car.built.as_ref()) {
            if step % 200 == 0 {
                let res = residuals(&rest, &now, fit, &rear_body);
                let (k, d) = res.iter().enumerate().fold((0, 0.0), |a, (k, d)| {
                    if d.length() > a.1 {
                        (k, d.length())
                    } else {
                        a
                    }
                });
                let i = rear_body[k];
                let lr = residuals(&rest, &now, fit, &lid);
                let (kl, dl) = lr.iter().enumerate().fold((0, 0.0), |a, (k, d)| {
                    if d.length() > a.1 {
                        (k, d.length())
                    } else {
                        a
                    }
                });
                eprintln!("{name} {sc:?} t {:.2}: rear worst node {i} at {:?} {:.1} mm {:?}; lid worst {} at {:?} {:.1} mm {:?}; mass {:.2}",
                step as f32 * STEP, rest[i], d * 1e3, b.groups.iter().filter(|(_, g)| g.contains(&(i as u32))).map(|(k, _)| k.clone()).collect::<Vec<_>>(),
                lid[kl], rest[lid[kl]], dl * 1e3, lr[kl], mass[lid[kl]]);
            }
        }
        r.lid_peak = r.lid_peak.max(lid_now);
        r.lid_end = lid_now;
        // Ripple: each part about its own rigid fit.
        for (p, ids) in &parts {
            let q0 = q_part.get(p).copied().unwrap_or(DQuat::IDENTITY);
            let f = rigid_fit(&rest, &now, &mass, ids, q0);
            q_part.insert(p.clone(), f.0);
            let wv = win
                .entry(p.clone())
                .or_insert_with(|| vec![Vec::new(); ids.len()]);
            for (k, d) in residuals(&rest, &now, f, ids).into_iter().enumerate() {
                wv[k].push(d);
            }
        }
        if step % window == 0 {
            for (p, wv) in win.iter_mut() {
                // √2·σ per node (a sinusoid's amplitude), RMS over the part's nodes.
                let a2: f64 = wv
                    .iter()
                    .map(|v| {
                        let m = v.iter().copied().sum::<DVec3>() / v.len() as f64;
                        2.0 * v.iter().map(|d| (*d - m).length_squared()).sum::<f64>()
                            / v.len() as f64
                    })
                    .sum::<f64>()
                    / wv.len() as f64;
                r.ripple.entry(p.clone()).or_default().push(a2.sqrt());
                for v in wv.iter_mut() {
                    v.clear();
                }
            }
        }
    }
    let s = &w.vehicles[0];
    r.finite = w.vehicles.iter().all(finite);
    r.rear = plastic(s, &car, &rear_beams);
    r.far = plastic(s, &car, &far_beams);
    if sc == Scene::Pole {
        r.far_hw = car
            .far_doors
            .iter()
            .flat_map(|p| car.hardware_broken(s, p))
            .collect();
    }
    r.lid_hw = car.hardware_broken(s, &car.lid);
    let all: Vec<usize> = (0..s.beams.len()).collect();
    let a = plastic(s, &car, &all);
    r.all = (a.1, a.0);
    r.all_by = a.3;
    if com_track.len() > 2 {
        // `explicit_jbeam_ext.rs`'s creep rule: a straight-line fit of the centre of
        // mass's horizontal position over the window (the last 4 s, every 0.1 s);
        // drift = its slope × the window, sway = the largest residual.
        let t: Vec<f64> = (0..com_track.len()).map(|k| k as f64 * 0.1).collect();
        let tm = t.iter().sum::<f64>() / t.len() as f64;
        for axis in [0usize, 2] {
            let y: Vec<f64> = com_track.iter().map(|c| c.to_array()[axis]).collect();
            let ym = y.iter().sum::<f64>() / y.len() as f64;
            let slope = t
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - tm) * (b - ym))
                .sum::<f64>()
                / t.iter().map(|a| (a - tm).powi(2)).sum::<f64>();
            let resid = t
                .iter()
                .zip(&y)
                .map(|(a, b)| (b - ym - slope * (a - tm)).abs())
                .fold(0.0, f64::max);
            r.creep = r.creep.max((slope * (t.last().unwrap() - t[0])).abs());
            r.sway = r.sway.max(resid);
        }
    }
    r
}

const ALL_CARS: [&str; 2] = ["sedan", "sedan1"];

/// The authored cars the tests run (`CI_CARS=sedan1` narrows them, a
/// measurement switch).
fn cars() -> Vec<&'static str> {
    match std::env::var("CI_CARS") {
        Ok(v) => ALL_CARS
            .iter()
            .copied()
            .filter(|c| v.split(',').any(|w| w == *c))
            .collect(),
        Err(_) => ALL_CARS.to_vec(),
    }
}

fn row(name: &str, car: &str, measured: String, bound: String, pass: bool) {
    eprintln!(
        "| {name} | {car} | {measured} | {bound} | {} |",
        if pass { "pass" } else { "FAIL" }
    );
}

/// No NaN; the ledger closes (10⁻³ of the start energy, `energy_ledger.rs`) and the
/// mechanical energy never rises past 0.5 % of the start (the symplectic step's
/// bounded oscillation, as `energy_ledger.rs`).
fn energy_ok(name: &str, sc: Scene) -> bool {
    let r = scene(name, sc);
    let pass =
        r.finite && r.closure < 1e-3 * r.energy_start && r.energy_rise <= 5e-3 * r.energy_start;
    row(
        &format!("{sc:?}: finite, ledger closes, energy never rises"),
        name,
        format!(
            "closure {:.1} J, rise {:.1} J of {:.0} kJ",
            r.closure,
            r.energy_rise,
            r.energy_start * 1e-3
        ),
        "10⁻³, 5·10⁻³ of start".into(),
        pass,
    );
    pass
}

// --- At rest and driving ------------------------------------------------------------

/// At rest 6 s: nothing yields past 0.2 % and nothing breaks (nothing loads the car
/// beyond its own weight), and the centre of mass does not creep: its drift over
/// the last 4 s is at most its sway plus 1 mm (no net horizontal force on a car at
/// rest; `explicit_jbeam_ext.rs`'s rule).
#[test]
fn rest_no_plastic_no_breaks_no_creep() {
    let mut ok = true;
    for car in cars() {
        let rr = scene(car, Scene::Rest);
        let pass = rr.all == (0, 0) && rr.creep <= rr.sway + 1e-3;
        row(
            "rest: (broken, plastic); creep",
            car,
            format!(
                "{:?}; drift {:.2} mm, sway {:.2} mm",
                rr.all,
                rr.creep * 1e3,
                rr.sway * 1e3
            ),
            "(0, 0); drift ≤ sway + 1 mm".into(),
            pass,
        );
        ok &= pass && energy_ok(car, Scene::Rest);
    }
    assert!(ok);
}

/// Driving straight at 15 m/s for 2 s: nothing yields past 0.2 % and nothing breaks.
///
/// FAILS (2026-11-05): sedan passes; sedan1 yields 2 beams in tension
/// (bonnet_inner_1, rail_f_r) from the spawn at speed with the per-node air drag
/// (0.05·|v|·v on every node whatever its mass: 112 m/s² on a 0.1 kg node at
/// 15 m/s); with drag off nothing yields (`drag_effect_report`).
#[test]
#[ignore]
fn drive_no_plastic_no_breaks() {
    let mut ok = true;
    for car in cars() {
        let dr = scene(car, Scene::Drive);
        let pass = dr.all == (0, 0);
        row(
            "drive 15 m/s: (broken, plastic)",
            car,
            format!("{:?} {:?}", dr.all, dr.all_by),
            "(0, 0)".into(),
            pass,
        );
        ok &= pass && energy_ok(car, Scene::Drive);
    }
    assert!(ok);
}

// --- Frontal --------------------------------------------------------------------------

/// 56 km/h wall and front-to-car head-on: the rear third (beams with both ends
/// within the rearmost third of the car's length, suspension excluded) has no beam
/// past 0.2 % plastic strain and none broken. NCAP frontals crush the front only
/// (residual crush is measured at the front, `crash-reference-targets.md` §1).
///
/// FAILS (2026-11-05): wall: sedan 21 beams past 0.2 % (worst 15 %), 0 broken; sedan1
/// 131 past (worst 23 %) and 24 broken (body skin both ways, rear bumper mounts,
/// lid inner frame, lid seals). Head-on: sedan passes (0); sedan1 37 past (7.1 %).
/// Mechanism: `lids_first_mode_in_the_measured_range` (rear panels' lowest
/// modes 2.5–7 Hz deflect 0.2–1.6 m quasi-statically at 47 g) and
/// `sheet_beams_yield_in_tension_at_the_sheet_yield` (skin edges yield at 0.6–15 % of
/// the sheet's tension yield). Not drag: the same with drag off (`drag_effect_report`).
#[test]
#[ignore]
fn frontal_rear_third_stays_elastic() {
    let mut ok = true;
    for car in cars() {
        for sc in [Scene::Wall, Scene::HeadOn] {
            let r = scene(car, sc);
            let pass = r.rear.0 == 0 && r.rear.1 == 0;
            row(
                &format!("{sc:?}: rear third plastic / broken"),
                car,
                format!(
                    "{} past 0.2 % (worst {:.2} %), {} broken of {}; {:?}",
                    r.rear.0,
                    r.rear.2 * 100.0,
                    r.rear.1,
                    r.rear_beams,
                    r.rear.3
                ),
                "0, 0".into(),
                pass,
            );
            ok &= pass;
        }
    }
    assert!(ok);
}

/// The boot lid stays latched and hinged and moves relative to the rear body by
/// at most [`LID_REL`]; the rear body stays within [`REAR_RIGID`] RMS of its rigid
/// fit.
///
/// FAILS (2026-11-05), lid vs rear body peak/end; rear body RMS/max: wall sedan
/// 631/348 mm, 40.6/238 mm; sedan1 564/486 mm, 100/518 mm; head-on sedan 172/18 mm,
/// 9.9/70 mm; sedan1 253/126 mm, 29.6/173 mm. No lid hardware breaks. The lid's
/// attachments are stiff enough as a rigid lid (2.5–4.5 mm at 40 g,
/// `boot_lid_attachment_report`); its own panel and the rear deck are not (lowest
/// modes 2.5–5.4 Hz, `panel_modes_report`): the lid's rear corner swings 0.62 m
/// forward at 0.2 s, then its middle sinks 0.33 m into the boot and stays.
#[test]
#[ignore]
fn frontal_boot_lid_and_rear_body_stay_put() {
    let mut ok = true;
    for car in cars() {
        for sc in [Scene::Wall, Scene::HeadOn] {
            let r = scene(car, sc);
            let pass = r.lid_hw.is_empty() && r.lid_peak <= LID_REL && r.rear_rms <= REAR_RIGID;
            row(
                &format!("{sc:?}: lid hardware; lid vs rear body peak/end; rear body RMS/max"),
                car,
                format!(
                    "{:?}; {:.1}/{:.1} mm; {:.1}/{:.1} mm",
                    r.lid_hw,
                    r.lid_peak * 1e3,
                    r.lid_end * 1e3,
                    r.rear_rms * 1e3,
                    r.rear_max * 1e3
                ),
                format!(
                    "none; ≤ {:.0} mm; ≤ {:.0} mm RMS",
                    LID_REL * 1e3,
                    REAR_RIGID * 1e3
                ),
                pass,
            );
            ok &= pass;
        }
    }
    assert!(ok);
}

/// After the wall crash every panel part's ripple (its displacement amplitude
/// about its own rigid motion, [`Run::ripple`]) is below [`VISIBLE`] from 0.5 s on.
///
/// FAILS (2026-11-05), largest amplitude from 0.5 to 1.5 s: sedan bootlid 17.9 mm,
/// bonnet 14.7, bumper_f 14.7, body 7.4, fender_r 6.0, bumper_r 5.0, doors 1.5–2.7;
/// sedan1 bumper_r 15.0, bumper_f 8.2, bonnet 7.4, bootlid 6.4, doors 1.4–4.5. Panel
/// modes are 2.5–7 Hz (lids, bonnets) and 17–37 Hz (doors), decaying to 1 % in
/// 0.5–7 s; the amplitude for a given struck speed is v/ω, so a lid at the measured 17.7–41.5 Hz
/// rings 2–8× smaller and decays ζω-proportionally faster.
#[test]
#[ignore]
fn wall_ripple_falls_below_visible_by_half_a_second() {
    let mut ok = true;
    for car in cars() {
        let r = scene(car, Scene::Wall);
        let mut bad = Vec::new();
        for (p, a) in &r.ripple {
            eprintln!(
                "  ripple {car} {p} (mm per 0.1 s): {}",
                a.iter()
                    .map(|x| format!("{:.2}", x * 1e3))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            if ["powertrain", "suspension", "radiator", "battery"].contains(&p.as_str()) {
                continue;
            }
            let late = a[5..].iter().copied().fold(0.0, f64::max);
            if late > VISIBLE {
                bad.push(format!("{p} {:.1} mm", late * 1e3));
            }
        }
        let pass = bad.is_empty();
        row(
            "wall: ripple amplitude from 0.5 s",
            car,
            format!("{bad:?}"),
            format!("all ≤ {:.0} mm", VISIBLE * 1e3),
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}

/// Head-on, front to front: node samples in contact range that meet the other
/// car faster than the band speed r/h (layer 1: above it half the crossings pass).
///
/// FAILS (2026-11-05): sedan 15 of 300,040 node samples within 0.3 m of the other
/// car meet it faster than 40 m/s (fastest 59.9 m/s); sedan1 15 of 197,234 (59.6 m/s).
/// Rare (5–8 per 100,000) but each can pass through the band (layer 1: half do above
/// r/h). A swept (continuous) crossing test for nodes faster than r/h is the
/// lossless fix (solver).
#[test]
#[ignore]
fn head_on_contact_speeds_within_the_band() {
    let mut ok = true;
    for car in cars() {
        let r = scene(car, Scene::HeadOn);
        let pass = r.fast.0 == 0;
        row(
            "head-on: node samples in range faster than r/h",
            car,
            format!("{} of {} (fastest {:.1} m/s)", r.fast.0, r.fast.1, r.fast.2),
            format!("0 above {BAND_SPEED} m/s"),
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}

/// Visuals follow physics: sampled every 0.1 s through the wall and the pole,
/// no exterior visual vertex's distance to the collision skin grows by more
/// than [`VISUAL_BAND`] (wheels, which ride their tyres, aside; a shattered
/// pane or lens is dropped as the renderer drops it).
///
/// The bind geometry bounds it: an exterior vertex rides the skin triangle
/// under it at |h| ≤ `SKIN_BAND` (2 cm), and the crease envelope adds at most
/// `ENVELOPE_MAX` (1 cm), so 3 cm for a vertex over its triangle; the 2 cm
/// asked is the contact band (a **Choice**: beyond it what is drawn is not
/// where contact happens).
///
/// FAILS (2026-11-05): 151.3 / 102.9 mm (sedan wall / pole), 172.1 / 150.0 mm
/// (sedan1). A vertex on its part's skin rides its part's own triangle
/// (BeamNG's flexbody binding; a shut line's triangles across two parts
/// drew ribbons and a sawtooth along the bonnet's rear edge once torn), but
/// the panels' drawn edges lie up to 0.15 m past their last node row, so
/// those edges ride nodes that far inboard. Content: node rows at the
/// panels' real edges (requested from the showcase car, 2026-11-05).
/// Riding the shut line's triangles measured 10.7 / 19.2 and 15.1 / 10.0 mm,
/// at the cost of those ribbons. It was 200 / 206 and 271 / 153 mm with
/// vertices bound to triangles 0.22–0.37 m away along their normal.
#[test]
#[ignore]
fn visual_skin_follows_the_collision_skin() {
    let mut ok = true;
    for car in cars() {
        for sc in [Scene::Wall, Scene::Pole] {
            let r = scene(car, sc);
            let pass = r.visual <= VISUAL_BAND;
            row(
                &format!("{sc:?}: visual offset growth"),
                car,
                format!("{:.1} mm", r.visual * 1e3),
                format!("≤ {:.0} mm", VISUAL_BAND * 1e3),
                pass,
            );
            ok &= pass;
        }
    }
    assert!(ok);
}

/// A detached door takes its edge with it: from the moment the door comes
/// off (every hinge and latch rope broken), for 0.3 s, no drawn visual
/// triangle joins it to the car (a ribbon of skin). A vertex on a frame
/// with a node across a part border (a one-row part's) switches to its own
/// part's frame once that frame dies (`ShowcaseVisual::alt`).
#[test]
fn detached_door_takes_its_edge() {
    let b = built("sedan1");
    let v = showcase_visual(&b).unwrap();
    let mut s = spawn_truck(&b.truck, Vec3::ZERO, Vec3::ZERO);
    for _ in 0..30 {
        s.frame();
    }
    let rest = s.positions();
    let sk = &b.skeleton;
    let part = sk.parts.iter().position(|p| p == "door_f_l").unwrap() as u16;
    let door: Vec<usize> = (0..sk.nodes.len())
        .filter(|&i| sk.nodes[i].part == part)
        .collect();
    let m: f32 = door.iter().map(|&i| s.nodes[i].mass).sum();
    let out = Vec3::X * s.nodes[door[0]].pos.x.signum();
    let mut at: HashMap<(u32, u32), usize> = HashMap::new();
    for (j, bm) in s.beams.iter().enumerate() {
        at.entry((bm.a.min(bm.b), bm.a.max(bm.b))).or_insert(j);
    }
    let hw: Vec<usize> = b
        .hardware
        .iter()
        .filter(|h| h.part == part)
        .flat_map(|h| h.beams.iter().map(|&j| sk.beams[j].clone()))
        .map(|bm| at[&(bm.a.min(bm.b), bm.a.max(bm.b))])
        .collect();
    // Drawn triangles joining the door to the car: a vertex whose frame (its
    // own, or its fallback once its shut-line triangle died) has nodes of
    // the door and of another part, or vertices of one triangle on frames
    // of different parts. Without the fallback (`alt` cleared) for scale.
    let pof = |k: u32| sk.nodes.get(k as usize).map(|n| n.part);
    let ribbons = |v: &ShowcaseVisual, now: &[Vec3], beams: &[BeamState]| {
        let (p, _) = deform_beams(v, now, None, Some(beams));
        let frame = |i: usize| {
            engine_softbody::authoring::skin::on_alt(v, i, now, Some(beams))
                .map_or(v.bind_nodes[i], |a| v.alt_nodes[a])
                .map(pof)
        };
        v.indices
            .chunks(3)
            .filter(|t| t.iter().all(|&i| p[i as usize].is_finite()))
            .filter(|t| {
                let all: Vec<Option<u16>> = t.iter().flat_map(|&i| frame(i as usize)).collect();
                all.contains(&Some(part)) && all.iter().any(|&q| q != Some(part))
            })
            .count()
    };
    let mut bare = v.clone();
    bare.alt.clear();
    // Pulled until it comes off, then let go; sampled every 10 steps for
    // 0.3 s (the shut line's triangles stretch before they tear).
    let (mut t, mut off) = (0.0f32, None);
    let (mut r0, mut r1, mut k) = (0, 0, 0);
    let mut now = s.positions();
    while t < 16.0 && off.is_none_or(|o: f32| t < o + 0.3) {
        let f = if off.is_some() { 0.0 } else { 10_000.0 * t };
        for &i in &door {
            let mi = s.nodes[i].mass;
            s.nodes[i].force += out * (f * mi / m);
        }
        s.step();
        t += STEP;
        if off.is_none() && hw.iter().all(|&j| s.beams[j].broken) {
            off = Some(t);
        }
        if off.is_some() {
            k += 1;
            if k % 10 == 1 {
                now = s.positions();
                let beams: Vec<BeamState> = s
                    .beams
                    .iter()
                    .map(|b| BeamState {
                        broken: b.broken,
                        rest: b.rest,
                    })
                    .collect();
                r0 = r0.max(ribbons(&bare, &now, &beams));
                let rv = ribbons(&v, &now, &beams);
                let first = rv > 0 && r1 == 0;
                r1 = r1.max(rv);
                if std::env::var("CI_DEBUG").is_ok() && first {
                    eprintln!("  at {:.3} s after it came off:", t - off.unwrap());
                    let (p, _) = deform_beams(&v, &now, None, Some(&beams));
                    let pn = |q: Option<u16>| {
                        q.map_or("tyre".to_string(), |q| sk.parts[q as usize].clone())
                    };
                    for tr in v
                        .indices
                        .chunks(3)
                        .filter(|t| t.iter().all(|&i| p[i as usize].is_finite()))
                    {
                        let fr: Vec<([Option<u16>; 3], Option<usize>, u32)> = tr
                            .iter()
                            .map(|&i| {
                                let a = engine_softbody::authoring::skin::on_alt(
                                    &v,
                                    i as usize,
                                    &now,
                                    Some(&beams),
                                );
                                (
                                    a.map_or(v.bind_nodes[i as usize], |a| v.alt_nodes[a])
                                        .map(pof),
                                    a,
                                    v.alt.get(i as usize).copied().unwrap_or(0),
                                )
                            })
                            .collect();
                        let all: Vec<Option<u16>> = fr.iter().flat_map(|f| f.0).collect();
                        if all.contains(&Some(part)) && all.iter().any(|&q| q != Some(part)) {
                            let i = tr[0];
                            let name = v
                                .pieces
                                .iter()
                                .find(|pc| i >= pc.1 && i < pc.1 + pc.2)
                                .map_or("-".into(), |pc| pc.0.clone());
                            for (f, a, al) in &fr {
                                let dl = if *al > 0 {
                                    v.skin_death[v.alt_tri[*al as usize - 1] as usize]
                                } else {
                                    [0; 4]
                                };
                                eprintln!(
                                    "    {name}: {:?} switched {a:?} alt {al} death {dl:?}",
                                    f.map(pn)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let off = off.expect("the door never came off");
    // The door's centre against the car's (every other node's).
    let mean = |idx: &mut dyn Iterator<Item = usize>| {
        let (mut c, mut n) = (Vec3::ZERO, 0.0);
        for i in idx {
            c += now[i] - rest[i];
            n += 1.0;
        }
        c / n
    };
    let door_c = mean(&mut door.iter().copied())
        - mean(&mut (0..sk.nodes.len()).filter(|&i| sk.nodes[i].part != part));
    row(
        "Detached door: drawn triangles joining it to the car (without / with the own-part fallback)",
        "sedan1",
        format!("{r0} / {r1} at most (off at {off:.2} s, door moved {:.0} mm)", door_c.length() * 1e3),
        "0".into(),
        r1 == 0,
    );
    assert_eq!(r1, 0);
}

// --- Side ------------------------------------------------------------------------------

/// 32 km/h side pole at the front door: the far side (beams with both ends 0.3 m
/// past the centre line away from the pole) stays elastic and its doors' hardware
/// holds: the deformation is local to the impact.
///
/// FAILS (2026-11-05): sedan 40 beams past 0.2 % and 22 broken on the far side (body
/// skin, firewall, right fender and bonnet edges); sedan1 89 past and 10 broken, and
/// the far side's door hardware breaks (door_f_r.latch_secondary,
/// door_r_r.hinge_upper), against FMVSS 214's rule that non-struck doors stay closed.
#[test]
#[ignore]
fn pole_far_side_stays_elastic() {
    let mut ok = true;
    for car in cars() {
        let r = scene(car, Scene::Pole);
        let pass = r.far.0 == 0 && r.far.1 == 0 && r.far_hw.is_empty();
        row(
            "pole: far side plastic / broken; far doors' hardware",
            car,
            format!(
                "{} past 0.2 % (worst {:.2} %), {} broken {:?}; {:?}",
                r.far.0,
                r.far.2 * 100.0,
                r.far.1,
                r.far.3,
                r.far_hw
            ),
            "0, 0; none".into(),
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}

/// BeamNG's own car (the BX coupe, opt-in: `RUBBLE_JBEAM_BX`, and
/// `RUBBLE_BEAMNG_MODS` for the donor tyre) on the same invariants: the rear
/// third in the frontals, the boot lid and rear body, the wall's ripple, and
/// the pole's far side. A calibration of what BeamNG's content does on this
/// solver: reported, never asserted. Skips without the content.
#[test]
#[ignore]
fn beamng_reference_invariants_report() {
    if std::env::var("RUBBLE_JBEAM_BX").is_err() {
        eprintln!("skip: RUBBLE_JBEAM_BX not set");
        return;
    }
    let car = "bx";
    for sc in [Scene::Wall, Scene::HeadOn] {
        let r = scene(car, sc);
        row(
            &format!("{sc:?}: rear third plastic / broken"),
            car,
            format!(
                "{} past 0.2 % (worst {:.2} %), {} broken of {}; {:?}",
                r.rear.0,
                r.rear.2 * 100.0,
                r.rear.1,
                r.rear_beams,
                r.rear.3
            ),
            "0, 0".into(),
            r.rear.0 == 0 && r.rear.1 == 0,
        );
        row(
            &format!("{sc:?}: lid hardware; lid vs rear body peak/end; rear body RMS/max"),
            car,
            format!(
                "{:?}; {:.1}/{:.1} mm; {:.1}/{:.1} mm",
                r.lid_hw,
                r.lid_peak * 1e3,
                r.lid_end * 1e3,
                r.rear_rms * 1e3,
                r.rear_max * 1e3
            ),
            format!(
                "none; ≤ {:.0} mm; ≤ {:.0} mm RMS",
                LID_REL * 1e3,
                REAR_RIGID * 1e3
            ),
            r.lid_hw.is_empty() && r.lid_peak <= LID_REL && r.rear_rms <= REAR_RIGID,
        );
    }
    let r = scene(car, Scene::Wall);
    let mut bad = Vec::new();
    for (p, a) in &r.ripple {
        if p == "suspension" || a.len() < 6 {
            continue;
        }
        let late = a[5..].iter().copied().fold(0.0, f64::max);
        eprintln!(
            "  ripple {car} {p} (mm per 0.1 s): {}",
            a.iter()
                .map(|x| format!("{:.2}", x * 1e3))
                .collect::<Vec<_>>()
                .join(" ")
        );
        if late > VISIBLE {
            bad.push(format!("{p} {:.1} mm", late * 1e3));
        }
    }
    row(
        "wall: ripple amplitude from 0.5 s",
        car,
        format!("{bad:?}"),
        format!("all ≤ {:.0} mm", VISIBLE * 1e3),
        bad.is_empty(),
    );
    let r = scene(car, Scene::Pole);
    row(
        "pole: far side plastic / broken; far doors' hardware",
        car,
        format!(
            "{} past 0.2 % (worst {:.2} %), {} broken {:?}; {:?}",
            r.far.0,
            r.far.2 * 100.0,
            r.far.1,
            r.far.3,
            r.far_hw
        ),
        "0, 0; none".into(),
        r.far.0 == 0 && r.far.1 == 0 && r.far_hw.is_empty(),
    );
    for sc in [Scene::Wall, Scene::HeadOn, Scene::Pole] {
        let r = scene(car, sc);
        eprintln!(
            "  {sc:?}: finite {}, whole car (broken, past 0.2 %) {:?}",
            r.finite, r.all
        );
    }
}

/// The ledgers in the crash scenes.
#[test]
fn crash_ledgers_close_and_energy_never_rises() {
    let mut ok = true;
    for car in cars() {
        for sc in [Scene::Wall, Scene::HeadOn, Scene::Pole] {
            ok &= energy_ok(car, sc);
        }
    }
    assert!(ok);
}

/// The rigid fit recovers a known rotation and translation exactly.
#[test]
fn rigid_fit_recovers_a_rigid_motion() {
    let rest: Vec<DVec3> = (0..20)
        .map(|i| {
            DVec3::new(
                (i as f64 * 0.37).sin(),
                (i as f64 * 0.11).cos(),
                i as f64 * 0.05,
            )
        })
        .collect();
    let q = DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.4);
    let t = DVec3::new(1.0, -2.0, 0.5);
    let now: Vec<DVec3> = rest.iter().map(|p| q * *p + t).collect();
    let ids: Vec<usize> = (0..20).collect();
    let w = vec![1.0; 20];
    let f = rigid_fit(&rest, &now, &w, &ids, DQuat::IDENTITY);
    let worst = residuals(&rest, &now, f, &ids)
        .iter()
        .map(|d| d.length())
        .fold(0.0, f64::max);
    assert!(worst < 1e-9, "{worst}");
}

/// Diagnosis (reported, asserts only finiteness): the same drive and wall with the
/// solver's per-node air drag (0.05·|v|·v on every node whatever its mass) off,
/// beside the default profile.
#[test]
fn drag_effect_report() {
    for car in cars() {
        for (on, off) in [
            (Scene::Drive, Scene::DriveNoDrag),
            (Scene::Wall, Scene::WallNoDrag),
        ] {
            for sc in [on, off] {
                let r = scene(car, sc);
                let late = |p: &str| {
                    r.ripple
                        .get(p)
                        .map(|a| a[a.len().min(5)..].iter().copied().fold(0.0, f64::max))
                        .unwrap_or(0.0)
                };
                eprintln!(
                    "| {sc:?} | {car} | lid vs rear body peak/end {:.1}/{:.1} mm; rear body RMS/max {:.1}/{:.1} mm; rear third plastic {} broken {}; ripple from 0.5 s: body {:.1}, bootlid {:.1}, bonnet {:.1}, door_f_l {:.1} mm; all (broken, plastic) {:?} |",
                    r.lid_peak * 1e3,
                    r.lid_end * 1e3,
                    r.rear_rms * 1e3,
                    r.rear_max * 1e3,
                    r.rear.0,
                    r.rear.1,
                    late("body") * 1e3,
                    late("bootlid") * 1e3,
                    late("bonnet") * 1e3,
                    late("door_f_l") * 1e3,
                    r.all
                );
                assert!(r.finite);
            }
        }
    }
}

/// Panel modes (the mechanism behind ripple): each panel part cut out of the car
/// standing at rest, its neighbours held (immovable), struck with 0.1 m/s along
/// its thinnest axis: its lowest mode's frequency and damping ratio (zero
/// crossings and peak decay of its mass-weighted mean displacement over 2 s).
/// Beside it the lowest mode of a flat steel plate of its gauge and its two
/// spans, simply supported (Kirchhoff: f₁₁ = (π/2)(1/a² + 1/b²)√(D/ρt), D =
/// Et³/12(1 − ν²)): a real panel is curved and double-skinned, which only raises
/// it. For the same struck speed v the amplitude is v/ω, so our amplitude over a
/// real panel's is at least f_real/f_ours. Reported; asserts finiteness.
fn panel_modes() -> Vec<(&'static str, &'static str, f64, f64)> {
    use engine_softbody::authoring::vehicle::PANEL_GAUGE;
    let mut out = Vec::new();
    use engine_softbody::explicit_beams::{Node, NODE_IMMOVABLE, NODE_NO_GROUND};
    // (`CI_REF=1`: BeamNG's BX too, opt-in as `beamng_reference_invariants_report`.)
    let mut list: Vec<&str> = cars();
    if std::env::var("CI_REF").is_ok() && std::env::var("RUBBLE_JBEAM_BX").is_ok() {
        list.push("bx");
    }
    for car in list {
        let sc = scene_car(car);
        let s = sc.spawn(Vec3::ZERO, Vec3::ZERO, false);
        let n = sc.n;
        for part in ["bootlid", "bonnet", "door_f_l", "door_r_l", "fender_l"] {
            let ids: Vec<usize> = (0..n).filter(|&i| sc.part[i] == part).collect();
            if ids.len() < 3 {
                continue;
            }
            let inside: std::collections::HashSet<usize> = ids.iter().copied().collect();
            let mut map: HashMap<usize, u32> = HashMap::new();
            let mut nodes: Vec<Node> = Vec::new();
            let g = engine_softbody::explicit_beams::GRAVITY;
            let mut add = |i: usize, fixed: bool, nodes: &mut Vec<Node>| {
                *map.entry(i).or_insert_with(|| {
                    let mut q = s.nodes[i];
                    q.vel = Vec3::ZERO;
                    q.flags = NODE_NO_GROUND | if fixed { NODE_IMMOVABLE } else { 0 };
                    q.force = Vec3::new(0.0, q.mass * g, 0.0);
                    nodes.push(q);
                    nodes.len() as u32 - 1
                })
            };
            for &i in &ids {
                add(i, false, &mut nodes);
            }
            let mut beams = Vec::new();
            let mut skipped = 0;
            for (j, bm) in s.beams.iter().enumerate() {
                let (a, c) = (bm.a as usize, bm.b as usize);
                if !(inside.contains(&a) || inside.contains(&c)) {
                    continue;
                }
                if !s.ext_of.is_empty() && s.ext_of[j] != 0 {
                    skipped += 1;
                    continue;
                }
                let mut nb = *bm;
                nb.a = add(a, !inside.contains(&a), &mut nodes);
                nb.b = add(c, !inside.contains(&c), &mut nodes);
                nb.shock = 0;
                beams.push(nb);
            }
            {
                // The part's mass and its attachments (beams to other parts) by kind.
                let pm: f32 = ids.iter().map(|&i| s.nodes[i].mass).sum();
                let mut by: BTreeMap<String, (usize, f32, f32)> = BTreeMap::new();
                for bm in &s.beams {
                    let (a, c) = (bm.a as usize, bm.b as usize);
                    if inside.contains(&a) != inside.contains(&c) && a < n && c < n {
                        let other = if inside.contains(&a) { c } else { a };
                        let e = by
                            .entry(format!("{:?}->{}", bm.kind, sc.part[other]))
                            .or_default();
                        e.0 += 1;
                        e.1 += bm.k;
                        e.2 += bm.strength.min(1e9);
                    }
                }
                eprintln!(
                    "  {car} {part}: {pm:.1} kg; attachments (count, Σk N/m, Σstrength N) {by:?}"
                );
            }
            let mut iso = ExplicitBeams::empty(ExplicitBeamsSettings {
                fast_inv_sqrt: s.settings.fast_inv_sqrt,
                drag: false,
                turbulence: false,
                ..s.settings
            });
            iso.ground = -1.0e3;
            iso.nodes = nodes;
            iso.beams = beams;
            iso.finish();
            // Thinnest axis of the part.
            let p: Vec<Vec3> = ids.iter().map(|&i| s.nodes[i].pos).collect();
            let (lo, hi) = p
                .iter()
                .fold((Vec3::MAX, Vec3::MIN), |(l, h), q| (l.min(*q), h.max(*q)));
            let ext = hi - lo;
            let axis = if ext.x <= ext.y && ext.x <= ext.z {
                0
            } else if ext.y <= ext.z {
                1
            } else {
                2
            };
            let dir = Vec3::AXES[axis];
            let mut spans = [ext.x, ext.y, ext.z];
            spans.sort_by(f32::total_cmp);
            let (a, bb) = (spans[1] as f64, spans[2] as f64);
            let free: Vec<usize> = (0..ids.len()).collect();
            for &k in &free {
                iso.nodes[k].vel = dir * 0.1;
            }
            let m: f64 = free.iter().map(|&k| iso.nodes[k].mass as f64).sum();
            let mut y = Vec::new();
            for _ in 0..4000 {
                iso.step();
                y.push(
                    free.iter()
                        .map(|&k| iso.nodes[k].mass as f64 * iso.nodes[k].pos.dot(dir) as f64)
                        .sum::<f64>()
                        / m,
                );
            }
            let mean = y[2000..].iter().sum::<f64>() / 2000.0;
            let y: Vec<f64> = y.iter().map(|v| v - mean).collect();
            let x = crossings_(&y);
            let f = if x.len() > 2 {
                (x.len() - 1) as f64 / ((x[x.len() - 1] - x[0]) * STEP as f64)
            } else {
                f64::NAN
            };
            let pk: Vec<(f64, f64)> = peaks_(&y)
                .into_iter()
                .filter(|p| p.1 > 1e-7)
                .map(|(t, v)| (t * STEP as f64, v.ln()))
                .collect();
            let sigma = -slope_(&pk);
            let zeta = sigma / (2.0 * std::f64::consts::PI * f);
            let (e, nu, rho, t) = (210e9f64, 0.3, 7850.0, PANEL_GAUGE as f64);
            let d = e * t.powi(3) / (12.0 * (1.0 - nu * nu));
            let f11 = std::f64::consts::FRAC_PI_2
                * (1.0 / (a * a) + 1.0 / (bb * bb))
                * (d / (rho * t)).sqrt();
            let amp0 = y.iter().take(400).map(|v| v.abs()).fold(0.0, f64::max);
            eprintln!(
                "| panel mode {part} | {car} | f {f:.1} Hz, ζ {zeta:.4}, decay to 1 % {:.0} ms, first 0.2 s amplitude {:.2} mm at 0.1 m/s ({} nodes, {} ext beams skipped) | flat {:.0}×{:.0} mm plate f₁₁ {f11:.1} Hz: our amplitude ≥ {:.1}× a real panel's | report |",
                1e3 * (100f64).ln() / sigma,
                amp0 * 1e3,
                ids.len(),
                skipped,
                a * 1e3,
                bb * 1e3,
                f11 / f
            );
            assert!(y.iter().all(|v| v.is_finite()));
            out.push((car, part, f, zeta));
        }
    }
    out
}

#[test]
fn panel_modes_report() {
    assert!(!panel_modes().is_empty());
}

/// The lids' (bonnet, boot lid) lowest out-of-plane mode on the car lies within
/// the measured production range [`LID_MODE_HZ`]: a lid's response to the crash
/// pulse is a/ω² (quasi-static), so its first mode sets how far it moves; a lid
/// below the range moves further than any measured hood, one above it is stiffer
/// than any. The doors and fenders are reported (no published modal data here).
///
/// FAILS (2026-11-05): sedan1 bootlid 6.7 Hz, bonnet 8.7 Hz (doors 23.6 and 26.3,
/// fender 117). The lids are single sheets on a rigidifier: no inner panel.
#[test]
#[ignore]
fn lids_first_mode_in_the_measured_range() {
    let a = 47.0 * 9.807;
    let mut ok = true;
    for (car, part, f, zeta) in panel_modes() {
        let lid = part == "bonnet" || part == "bootlid";
        let pass = !lid || (LID_MODE_HZ[0]..=LID_MODE_HZ[1]).contains(&f);
        row(
            &format!("panel lowest mode {part}"),
            car,
            format!(
                "{f:.1} Hz (ζ {zeta:.3}); 47 g deflects {:.0} mm",
                1e3 * a / (2.0 * std::f64::consts::PI * f).powi(2)
            ),
            if lid {
                format!("{:.1}–{:.1} Hz", LID_MODE_HZ[0], LID_MODE_HZ[1])
            } else {
                "report".into()
            },
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}

fn crossings_(y: &[f64]) -> Vec<f64> {
    (1..y.len())
        .filter(|&i| y[i - 1] < 0.0 && y[i] >= 0.0)
        .map(|i| (i - 1) as f64 + y[i - 1] / (y[i - 1] - y[i]))
        .collect()
}

fn peaks_(y: &[f64]) -> Vec<(f64, f64)> {
    (1..y.len() - 1)
        .filter(|&i| y[i] > y[i - 1] && y[i] >= y[i + 1] && y[i] > 0.0)
        .map(|i| (i as f64, y[i]))
        .collect()
}

fn slope_(p: &[(f64, f64)]) -> f64 {
    let n = p.len() as f64;
    let (mx, my) = p
        .iter()
        .fold((0.0, 0.0), |a, q| (a.0 + q.0 / n, a.1 + q.1 / n));
    p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum::<f64>()
        / p.iter().map(|q| (q.0 - mx).powi(2)).sum::<f64>()
}

/// Diagnosis: the boot lid's attachments, each beam's kind and direction (unit,
/// lid → body) and stiffness, and which of them resists the lid moving forward
/// (+z, its inertia in a frontal): a rope (tension only) resists only if the
/// motion stretches it, a support (compression only) only if it shortens it.
#[test]
fn boot_lid_attachment_report() {
    use engine_softbody::explicit_beams::ExplicitKind;
    for car in cars() {
        let b = built(car);
        let sk = &b.skeleton;
        let s = spawn_truck(&b.truck, Vec3::ZERO, Vec3::ZERO);
        let n = sk.nodes.len();
        let lid = |i: usize| i < n && sk.parts[sk.nodes[i].part as usize] == "bootlid";
        let (mut fwd, mut back, mut down, mut up) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for bm in &s.beams {
            let (a, c) = (bm.a as usize, bm.b as usize);
            if lid(a) == lid(c) || a >= n || c >= n {
                continue;
            }
            let (l, o) = if lid(a) { (a, c) } else { (c, a) };
            let d = (s.nodes[o].pos - s.nodes[l].pos).normalize();
            // Moving the lid by u changes the beam's length by −u·d.
            let resists = |u: Vec3| match bm.kind {
                ExplicitKind::Rope => -u.dot(d) > 0.0,
                ExplicitKind::Support | ExplicitKind::JbeamSupport => -u.dot(d) < 0.0,
                _ => true,
            };
            let k_along = |u: Vec3| {
                if resists(u) {
                    bm.k * u.dot(d).powi(2)
                } else {
                    0.0
                }
            };
            fwd += k_along(Vec3::Z);
            back += k_along(-Vec3::Z);
            down += k_along(-Vec3::Y);
            up += k_along(Vec3::Y);
            if bm.kind == ExplicitKind::Rope {
                eprintln!(
                    "  {car} lid hardware rope {}: k {:.0}, dir {:.2?}, strength {:.0}",
                    label(&b, bm.a, bm.b),
                    bm.k,
                    d,
                    bm.strength
                );
            }
        }
        let m: f32 = (0..n).filter(|&i| lid(i)).map(|i| s.nodes[i].mass).sum();
        let a40 = 40.0 * 9.807 * m;
        eprintln!(
            "| boot lid attachment stiffness (small motion) | {car} | forward {fwd:.0}, back {back:.0}, down {down:.0}, up {up:.0} N/m; lid {m:.1} kg: 40 g forward → {:.1} mm if linear | report |",
            if fwd > 0.0 { 1e3 * a40 / fwd } else { f32::INFINITY }
        );
    }
}

// --- Community JBeam cars (opt-in: RUBBLE_BEAMNG_MODS) ----------------------------------

/// The same frontal invariants on community JBeam cars (BeamNG content on Rubble's
/// solver; pressure wheels): if their rear thirds also yield in a 56 km/h wall, the
/// cause is the solver; if not, the authored car's content. Measured: rear third
/// past 0.2 % and broken, the rear third's RMS / largest displacement about its
/// rigid fit, energy, NaN. Reported (the community cars have their own targets);
/// asserts finiteness and that energy never rises.
#[test]
fn community_wall_56_report() {
    use engine_softbody::explicit_beams::JbeamWheels;
    use engine_softbody::jbeam::load_jbeam;
    let Ok(mods) = std::env::var("RUBBLE_BEAMNG_MODS") else {
        eprintln!("skip: RUBBLE_BEAMNG_MODS not set");
        return;
    };
    let m = std::path::Path::new(&mods);
    for (dir, name, pc) in [
        ("monster_supercar/vehicles", "supercar", "config.pc"),
        (
            "qlourie_astral_r/vehicles",
            "qlourie_astral_r",
            "wrc17_tarmac_balanced.pc",
        ),
    ] {
        let dir = m.join(dir);
        let Ok(files) = engine_softbody::jbeam::read_vehicle_dir(&dir, name) else {
            continue;
        };
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
        let desc = load_jbeam(&refs, Some(&pcs)).unwrap();
        let v = Vec3::Z * (56.0 / 3.6);
        let st = ExplicitBeamsSettings {
            turbulence: false,
            ..Default::default()
        };
        let (mut s, _) =
            ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Pressure, Vec3::ZERO, v, st)
                .unwrap();
        let f = engine_softbody::pack::forward_axis(&desc);
        let turn = glam::Quat::from_rotation_y(-f.x.atan2(f.z));
        for nd in &mut s.nodes {
            nd.pos = turn * nd.pos;
        }
        let low = s.nodes.iter().map(|q| q.pos.y).fold(f32::MAX, f32::min);
        for nd in &mut s.nodes {
            nd.pos.y -= low - 0.02;
        }
        s.roll_tyres(v);
        // Settle a moment at speed before the wall (0.3 s of road).
        let mut w = ExplicitBeamsWorld::new(vec![s]);
        for _ in 0..600 {
            w.step();
        }
        let s = &mut w.vehicles[0];
        let p = s.positions();
        let front = p.iter().map(|q| q.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        let rest = dpos(s);
        let mass: Vec<f64> = s.nodes.iter().map(|q| q.mass as f64).collect();
        let (zmin, zmax) = rest
            .iter()
            .fold((f64::MAX, f64::MIN), |a, q| (a.0.min(q.z), a.1.max(q.z)));
        let cut = zmin + (zmax - zmin) / 3.0;
        // Tyre and hub nodes are not the body: leave out the pressure wheels' nodes.
        let wheel: std::collections::HashSet<usize> = s
            .tyres
            .iter()
            .flat_map(|t| (t.first..t.first + t.count).map(|i| i as usize))
            .collect();
        let rear: Vec<usize> = (0..rest.len())
            .filter(|&i| rest[i].z < cut && !wheel.contains(&i))
            .collect();
        let rset: std::collections::HashSet<usize> = rear.iter().copied().collect();
        let rear_beams: Vec<usize> = (0..s.beams.len())
            .filter(|&j| {
                rset.contains(&(s.beams[j].a as usize)) && rset.contains(&(s.beams[j].b as usize))
            })
            .collect();
        let rest0: Vec<f32> = s.beams.iter().map(|b| b.rest).collect();
        w.enable_ledger();
        let start = w.ledger.as_ref().unwrap().start;
        let (mut rise, mut rms_w, mut max_w) = (0.0f64, 0.0f64, 0.0f64);
        for step in 0..3000 {
            w.step();
            rise = rise.max(w.ledger.as_ref().unwrap().mechanical() - start);
            if step % 10 == 0 {
                let now = dpos(&w.vehicles[0]);
                let fit = rigid_fit(&rest, &now, &mass, &rear, DQuat::IDENTITY);
                let res = residuals(&rest, &now, fit, &rear);
                rms_w = rms_w.max(
                    (res.iter().map(|d| d.length_squared()).sum::<f64>() / res.len() as f64).sqrt(),
                );
                max_w = max_w.max(res.iter().map(|d| d.length()).fold(0.0, f64::max));
            }
        }
        let s = &w.vehicles[0];
        let (mut past, mut broken) = (0, 0);
        for &j in &rear_beams {
            let b = &s.beams[j];
            if b.broken {
                broken += 1;
            } else if (b.rest - rest0[j]).abs() > PLASTIC * rest0[j].abs() {
                past += 1;
            }
        }
        let all_broken = s.beams.iter().filter(|b| b.broken).count();
        let tm: f64 = mass.iter().sum();
        let vz = s
            .nodes
            .iter()
            .map(|q| q.mass as f64 * q.vel.z as f64)
            .sum::<f64>()
            / tm;
        let front_now = s.positions().iter().map(|q| q.z).fold(f32::MIN, f32::max);
        eprintln!("  {name}: centre-of-mass speed at 1.5 s {vz:.2} m/s; front at {front_now:.2} (wall face {:.2})", front + 0.05);
        eprintln!(
            "| Wall (community) | {name} | rear third {past} past 0.2 %, {broken} broken of {}; rear RMS/max {:.1}/{:.1} mm; broken overall {all_broken}; energy rise {rise:.1} J of {:.0} kJ | report |",
            rear_beams.len(),
            rms_w * 1e3,
            max_w * 1e3,
            start * 1e-3
        );
        assert!(finite(s));
        assert!(rise <= 5e-3 * start);
    }
}

/// Sheet beams yield in tension at the sheet's yield stress. Winter's effective
/// width (`sheet_ultimate_stress`) is a compression result (a thin plate buckles,
/// then carries load on its effective width); sheet steel pulled in tension does not
/// buckle and yields at σ_y over its full width. So a body sheet edge (a
/// triangular lattice of the 0.8 mm shell, `SHELL_GAUGE`) yields in tension at no
/// less than (√3/4)·σ_y·t·L (σ_y 210 MPa, `SHEET_YIELD`). The `.truck` beam has one
/// yield threshold for both directions.
///
/// FAILS (2026-11-05), measured in the report row: sedan's body skin edges yield at
/// 236–377 N (the two-tier skin's scaled thresholds), less than one 1.5 kg node's own
/// inertia at 47 g (690 N), so its rear skin yields under its own inertia in a
/// frontal; sedan1's at the Winter compression value (≈ 3.4 kN, about 0.13 of the
/// tension yield). Content (`vehicle.rs`, "the same both ways, JBeam's one
/// beamDeform"); a fix needs separate tension and compression yields (content or a
/// `.truck` extension).
#[test]
#[ignore]
fn sheet_beams_yield_in_tension_at_the_sheet_yield() {
    use engine_softbody::authoring::vehicle::{SHEET_YIELD, SHELL_GAUGE};
    let lattice = 3f32.sqrt() / 4.0;
    let mut ok = true;
    for car in cars() {
        let b = built(car);
        let sk = &b.skeleton;
        // The body's sheet edges: structure beams within the body, not ties, dampers
        // or group members, no longer than a bay (0.5 m), yielding below the sheet's
        // tension yield (members' chords yield above it and are left out).
        let mut ratios: Vec<f32> = Vec::new();
        let mut n_struct = 0usize;
        for bm in &sk.beams {
            let (na, nc) = (&sk.nodes[bm.a as usize], &sk.nodes[bm.b as usize]);
            if na.part != 0 || nc.part != 0 || !bm.structure || bm.tie || bm.damper || bm.group != 0
            {
                continue;
            }
            let l = na.p.distance(nc.p);
            if l > 0.5 || bm.mat.deform <= 0.0 {
                continue;
            }
            n_struct += 1;
            let tension = lattice * SHEET_YIELD * SHELL_GAUGE * l;
            if bm.mat.deform < 0.99 * tension {
                ratios.push(bm.mat.deform / tension);
            }
        }
        ratios.sort_by(f32::total_cmp);
        let med = ratios.get(ratios.len() / 2).copied().unwrap_or(f32::NAN);
        let pass = ratios.is_empty();
        row(
            "body sheet edges: tension yield / (√3/4)σ_y·t·L",
            car,
            format!(
                "{} of {n_struct} below 1, median {med:.3}, worst {:.3}",
                ratios.len(),
                ratios.first().copied().unwrap_or(f32::NAN)
            ),
            "none below 1".into(),
            pass,
        );
        ok &= pass;
    }
    assert!(ok);
}

/// Where a wall crash's mechanical energy rises (a measurement, `--ignored`):
/// the world's ledger per term over each 0.1 s, the car `PROBE_CAR`
/// (default sedan1), 56 km/h, drag on (`PROBE_DRAG=0` off), `NO_SELF=1`
/// without self-collision. A tenth whose mechanical energy rises names the
/// source whose work does it: the beams' net (their dissipation negative),
/// self-contact, ground, air.
#[test]
#[ignore]
fn energy_terms_probe() {
    let name = std::env::var("PROBE_CAR").unwrap_or_else(|_| "sedan1".into());
    let car = scene_car(&name);
    let v56 = std::env::var("PROBE_KMH")
        .ok()
        .and_then(|k| k.parse::<f32>().ok())
        .unwrap_or(56.0)
        / 3.6;
    let mut s = car.spawn(Vec3::ZERO, Vec3::Z * v56, false);
    s.settings.drag = std::env::var("PROBE_DRAG").map_or(true, |v| v != "0");
    if std::env::var("NO_SELF").is_ok() {
        s.triangles.clear();
        s.finish();
    }
    let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let snap = |l: &engine_softbody::explicit_contact::EnergyLedger| {
        [
            l.mechanical(),
            l.kinetic,
            l.elastic,
            l.w_beams,
            l.beam_dissipation(),
            l.w_self_contact,
            l.w_ground_vertical + l.w_ground_horizontal,
            l.w_air,
            l.w_contact_normal + l.w_contact_friction,
            l.discretisation,
        ]
    };
    let window = (0.1 / STEP).round() as usize;
    // Tenth means of the mechanical energy (as `showcase_car.rs`'s ledger
    // judges rises), and the terms' changes over each tenth.
    let mut prev = snap(w.ledger.as_ref().unwrap());
    let mut prev_mean = f64::NAN;
    for t in 1..=20 {
        let mut mean = 0.0;
        for _ in 0..window {
            let e0 = w.ledger.as_ref().unwrap().mechanical();
            w.step();
            let e1 = w.ledger.as_ref().unwrap().mechanical();
            mean += e1 / window as f64;
            // A step whose mechanical energy jumps (`PROBE_JUMP` J, 100):
            // the fastest node and its beams.
            let jump: f64 = std::env::var("PROBE_JUMP")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(100.0);
            if e1 - e0 > jump {
                let s = &w.vehicles[0];
                let (sp, i) = (0..s.nodes.len())
                    .map(|i| (s.nodes[i].vel.length(), i))
                    .fold((0.0f32, 0), |a, b| if b.0 > a.0 { b } else { a });
                eprintln!(
                    "  step jump {:+.1} J (kinetic now {:.1} J): fastest node {i} {} {:.3} kg {sp:.1} m/s",
                    e1 - e0,
                    w.ledger.as_ref().unwrap().kinetic,
                    car.part.get(i).map_or("wheel", |x| x.as_str()),
                    s.nodes[i].mass
                );
                for &bm in s.incidence[i].iter().take(12) {
                    let b = &s.beams[bm as usize];
                    let o = if b.a as usize == i { b.b } else { b.a } as usize;
                    let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length();
                    eprintln!(
                        "    beam {bm} to {o} ({}): len {l:.3} rest {:.3} rest0 {:.3} k {:.2e} tc {:.0} tt {:.0} broken {}",
                        car.part.get(o).map_or("wheel", |x| x.as_str()),
                        b.rest, b.rest0, b.k, b.tc, b.tt, b.broken
                    );
                }
            }
        }
        let now = snap(w.ledger.as_ref().unwrap());
        let d: Vec<f64> = now.iter().zip(&prev).map(|(a, b)| a - b).collect();
        eprintln!(
            "{name} tenth {t:2}: mean mech {:+9.2} J; Δ mech {:+9.2} (kinetic {:+9.2}, elastic {:+9.2}); work beams {:+9.2} (dissipation {:+9.2}), self-contact {:+7.2}, ground {:+7.2}, air {:+7.2}, obstacle {:+7.2}; discretisation {:+7.3}",
            mean - prev_mean,
            d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]
        );
        prev = now;
        prev_mean = mean;
    }
    // The fastest nodes at the end (a contact limit cycle shows as a node
    // still moving fast in a car at rest).
    let s = &w.vehicles[0];
    let mut v: Vec<(f32, usize)> = (0..car.n).map(|i| (s.nodes[i].vel.length(), i)).collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(sp, i) in v.iter().take(8) {
        let groups: Vec<String> = car
            .built
            .as_ref()
            .map(|b| {
                b.groups
                    .iter()
                    .filter(|(_, g)| g.contains(&(i as u32)))
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default();
        eprintln!(
            "  node {i} {} {:.3} kg at {:.3?}: {sp:.3} m/s {groups:?}",
            car.part[i], s.nodes[i].mass, s.nodes[i].pos
        );
    }

/// The BeamNG profile's air drag (jbeam-semantics-spec.md §31, §36): sedan1 takes
/// the triangles' drag at BeamNG's default dragCoef (100) in place of the per-node
/// drag. Coasting at 100 km/h clear of the ground (0.05 s): the drag force against
/// the per-node rule's (DRAG·|v|² per node, independent of mass) and a real
/// sedan's (½ρ·Cd·A, Cd·A ≈ 0.65 m²).
#[test]
fn profile_drag_is_the_triangles() {
    let b = built("sedan1");
    let v = 100.0 / 3.6;
    let mut s = spawn_truck(&b.truck, Vec3::new(0.0, 50.0, 0.0), Vec3::Z * v);
    assert!(!s.settings.drag && !s.settings.turbulence);
    let tris = s
        .beams
        .iter()
        .filter(|bm| bm.kind == engine_softbody::explicit_beams::ExplicitKind::JbeamAeroTri)
        .count();
    assert!(tris > 0);
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let p0: f32 = s.nodes.iter().map(|n| n.mass * n.vel.z).sum();
    let steps = 100;
    for _ in 0..steps {
        s.step();
    }
    let p1: f32 = s.nodes.iter().map(|n| n.mass * n.vel.z).sum();
    let f = -(p1 - p0) / (steps as f32 * STEP);
    let per_node = engine_softbody::explicit_beams::DRAG * s.nodes.len() as f32 * v * v;
    let real = 0.5 * 1.225 * 0.65 * v * v;
    eprintln!(
        "sedan1 at 100 km/h, {m:.0} kg: triangle drag {f:.0} N ({tris} triangles); per-node rule {per_node:.0} N; a real sedan {real:.0} N"
    );
    assert!(f > 0.0);
    // `rubble_drag_coef 9` (Rubble's extension): the triangles at 9 % of a flat
    // plate, about a real sedan's drag.
    let text = b.truck.replace("rubble_beamng_profile", "rubble_beamng_profile\nrubble_drag_coef 9");
    let mut s = spawn_truck(&text, Vec3::new(0.0, 50.0, 0.0), Vec3::Z * v);
    let p0: f32 = s.nodes.iter().map(|n| n.mass * n.vel.z).sum();
    for _ in 0..steps {
        s.step();
    }
    let p1: f32 = s.nodes.iter().map(|n| n.mass * n.vel.z).sum();
    let f9 = -(p1 - p0) / (steps as f32 * STEP);
    eprintln!("with rubble_drag_coef 9: {f9:.0} N");
    assert!((f9 / f - 0.09).abs() < 0.02, "{f9} against {f}");
}
