//! Scenarios as data ([`ScenarioDesc`]): what the Reference Studio writes when a crash is
//! recreated (`apps/reference-studio/workspace/scenarios/<id>.json`), executed by the same
//! [`CrashScenario::build`] and [`super::run`] as the scenarios defined in Rust.
//!
//! A data scenario places everything absolutely (the studio moves boxes and bodies by hand);
//! the run's [`Jitter`] is applied as the Rust scenarios apply it: every speed multiplied,
//! the striking body (the first that moves) moved across its path and turned. Its first
//! contact is not stated but computed ([`first_contact`]). Its criteria are the generic
//! law-based ones only ([`law_criteria`]): a recreated crash that earns cited bands is
//! promoted to a Rust file.

use super::*;
use ts_rs::TS;

/// One vehicle of a data scenario.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, TS)]
pub struct BodyDesc {
    /// The name metrics and reports use (`car`, `truck`...).
    pub name: String,
    /// A generated design (`suv_4x4`, `sedan`, `wagon`, `truck_cab`, `box_trailer`, `mdb214`...).
    pub design: String,
    /// Where its own origin stands (m, world; y is the ground's height, normally 0: it is
    /// stood on its wheels from there).
    pub at: [f32; 3],
    /// Its heading from +Z toward +X (degrees).
    #[serde(default)]
    pub yaw_deg: f32,
    /// Its speed along its heading (km/h).
    #[serde(default)]
    pub speed_kmh: f32,
    /// Its velocity when it does not move along its heading (m/s, world); overrides the
    /// speed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub velocity: Option<[f32; 3]>,
    /// It rides a side-impact carrier (its tyres on low-friction pads).
    #[serde(default)]
    pub carrier: bool,
}

/// A static box (fixed, of infinite mass).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, TS)]
pub struct BoxDesc {
    pub name: String,
    /// Centre (m, world).
    pub centre: [f32; 3],
    /// Half extents along its own axes (m).
    pub half: [f32; 3],
    /// Its turn about +Y (degrees; 0: axis-aligned).
    #[serde(default)]
    pub yaw_deg: f32,
}

/// A crash scenario as data: the Reference Studio's editor reads and writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, TS)]
pub struct ScenarioDesc {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub notes: String,
    pub bodies: Vec<BodyDesc>,
    #[serde(default)]
    pub boxes: Vec<BoxDesc>,
    /// Named cameras (world). A reference image's own camera lives on its frame.
    #[serde(default)]
    pub cameras: Vec<Camera>,
    /// Named instants (ms after the computed first contact).
    #[serde(default)]
    pub moments: Vec<Moment>,
    /// Simulated seconds after first contact.
    pub secs_after_impact: f32,
    /// The contact setting it is run in by default.
    #[serde(default = "edge")]
    pub contact: Contact,
}

fn edge() -> Contact {
    Contact::Edge
}

/// The ids a data scenario may have: letters, digits, `-`, `_`, `.`, and not one of the
/// Rust suite's (`S01`...).
pub fn valid_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 64 || !id.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) || id.starts_with('.') {
        return Err(format!("scenario id {id:?}: 1 to 64 letters, digits, - _ ."));
    }
    if all().iter().any(|s| s.id.eq_ignore_ascii_case(id)) {
        return Err(format!("scenario id {id}: a built-in scenario has it"));
    }
    Ok(())
}

impl ScenarioDesc {
    /// Checks a description: an id, at least one body, finite numbers, positive box sizes,
    /// unique names, a positive duration.
    pub fn check(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("a scenario needs an id".into());
        }
        if self.bodies.is_empty() {
            return Err(format!("{}: a scenario needs a body", self.id));
        }
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        let mut names = std::collections::BTreeSet::new();
        for b in &self.bodies {
            if !names.insert(b.name.as_str()) {
                return Err(format!("{}: two bodies are named {}", self.id, b.name));
            }
            if b.design.is_empty() || !finite(&b.at) || !b.yaw_deg.is_finite() || !b.speed_kmh.is_finite() || b.velocity.is_some_and(|v| !finite(&v)) {
                return Err(format!("{}: body {}: a design and finite numbers", self.id, b.name));
            }
        }
        for x in &self.boxes {
            if !finite(&x.centre) || !x.yaw_deg.is_finite() || !x.half.iter().all(|h| h.is_finite() && *h > 0.0) {
                return Err(format!("{}: box {}: a finite centre and positive half sizes", self.id, x.name));
            }
        }
        if !(self.secs_after_impact.is_finite() && self.secs_after_impact > 0.0 && self.secs_after_impact <= 10.0) {
            return Err(format!("{}: secs_after_impact in (0, 10]", self.id));
        }
        Ok(())
    }

    /// The designs it needs, in first-use order.
    pub fn designs(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for b in &self.bodies {
            if !out.contains(&b.design) {
                out.push(b.design.clone());
            }
        }
        out
    }

    /// The scenario this description is: the same [`CrashScenario`] every executor takes.
    pub fn scenario(&self) -> CrashScenario {
        CrashScenario {
            id: self.id.clone(),
            title: self.title.clone(),
            reference: Reference { video: "", at_s: &[], real_test: "" },
            designs: self.designs(),
            layout: LayoutSource::Data(Box::new(self.clone())),
            secs_after_impact: self.secs_after_impact,
            cameras: self.cameras.clone(),
            moments: self.moments.clone(),
            criteria: law_criteria(!self.boxes.is_empty()),
            recreated: true,
        }
    }

    /// The layout for one run: every body at its pose, each speed times the jitter's, the
    /// striking body (the first that moves) moved across its path by the jitter's lateral
    /// and turned by its yaw; boxes as stated; first contact computed ([`first_contact`]).
    pub fn resolve(&self, extent: &dyn Fn(&str) -> Extent, j: Jitter) -> Layout {
        let striking = self.bodies.iter().position(|b| b.speed_kmh != 0.0 || b.velocity.is_some());
        let mut bodies = Vec::new();
        for (i, b) in self.bodies.iter().enumerate() {
            let yaw0 = b.yaw_deg.to_radians();
            let mut at = Vec3::from(b.at);
            let mut yaw = yaw0;
            if striking == Some(i) {
                // Across its path: its right (+X at yaw 0), as the Rust scenarios move it.
                at += Vec3::new(yaw0.cos(), 0.0, -yaw0.sin()) * j.lateral;
                yaw = yaw0 + j.yaw;
            }
            let mut body = Body::new(&b.name, &b.design, at, yaw, b.speed_kmh / 3.6 * j.speed);
            body.vel = b.velocity.map(|v| Vec3::from(v) * j.speed);
            body.carrier = b.carrier;
            bodies.push(body);
        }
        let boxes: Vec<(Vec3, Vec3)> = self.boxes.iter().map(|x| (Vec3::from(x.centre), Vec3::from(x.half))).collect();
        let box_yaw: Vec<f32> = self.boxes.iter().map(|x| x.yaw_deg.to_radians()).collect();
        let mut layout = Layout { bodies, boxes, box_yaw, impact_s: 0.0 };
        layout.impact_s = first_contact(&layout, extent).unwrap_or(0.0);
        layout
    }
}

impl CrashScenario {
    /// This scenario as data, laid out for the nominal run (run 0) on these templates: every
    /// body and box placed absolutely (`None` when it needs a design the templates lack).
    /// The data path reproduces the Rust one bit for bit at run 0
    /// (`tests::s02_as_data_is_s02`); at other runs a Rust layout's relative placements
    /// (a wall a tenth of a second ahead at the jittered speed) are not kept.
    pub fn to_desc(&self, templates: &Templates) -> Result<ScenarioDesc, String> {
        let l = self.layout_for(templates, Jitter::run(0))?;
        // The stated numbers, rounded where the Rust scenarios state round ones (km/h to
        // 1/1000, degrees to 1/10000), so that the data path computes what the Rust did.
        let r = |x: f32, k: f32| (x * k).round() / k;
        Ok(ScenarioDesc {
            id: self.id.clone(),
            title: self.title.clone(),
            notes: format!("exported from the Rust scenario {} (run 0)", self.id),
            bodies: l
                .bodies
                .iter()
                .map(|b| BodyDesc {
                    name: b.name.clone(),
                    design: b.design.clone(),
                    at: b.at.to_array(),
                    yaw_deg: r(b.yaw.to_degrees(), 1e4),
                    speed_kmh: r(b.speed * 3.6, 1e3),
                    velocity: b.vel.map(|v| v.to_array()),
                    carrier: b.carrier,
                })
                .collect(),
            boxes: l
                .boxes
                .iter()
                .enumerate()
                .map(|(i, (c, h))| BoxDesc { name: format!("box{i}"), centre: c.to_array(), half: h.to_array(), yaw_deg: r(l.yaw_of(i).to_degrees(), 1e4) })
                .collect(),
            cameras: self.cameras.clone(),
            moments: self.moments.clone(),
            secs_after_impact: self.secs_after_impact,
            contact: Contact::Edge,
        })
    }
}

/// A body's box at spawn in the world (as [`CrashScenario::build`] places it: turned by its
/// yaw and stood on its wheels): the box around its rest extent's eight corners turned.
pub fn body_box(b: &Body, e: &Extent) -> (Vec3, Vec3) {
    let (s, c) = b.yaw.sin_cos();
    let turn = |p: Vec3| Vec3::new(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for k in 0..8 {
        let p = Vec3::new(if k & 1 == 0 { e.lo.x } else { e.hi.x }, if k & 2 == 0 { e.lo.y } else { e.hi.y }, if k & 4 == 0 { e.lo.z } else { e.hi.z });
        let q = turn(p);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    let lift = 0.02 - e.wheel_low.map_or(e.lo.y, |w| w.min(e.lo.y));
    let at = b.at + Vec3::Y * lift;
    (lo + at, hi + at)
}

/// A static box's world box (an oriented one's axis-aligned bound).
pub fn static_box(c: Vec3, h: Vec3, yaw: f32) -> (Vec3, Vec3) {
    let (s, co) = yaw.sin_cos();
    let hx = (h.x * co).abs() + (h.z * s).abs();
    let hz = (h.x * s).abs() + (h.z * co).abs();
    let e = Vec3::new(hx, h.y, hz);
    (c - e, c + e)
}

/// The earliest time `t ≥ 0` at which box `a` moving at `v` (relative to `b`) touches `b`
/// (slab sweep, each axis's entry and exit), or None.
fn sweep(a: (Vec3, Vec3), v: Vec3, b: (Vec3, Vec3)) -> Option<f32> {
    let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
    for k in 0..3 {
        let (alo, ahi, blo, bhi, vk) = (a.0[k], a.1[k], b.0[k], b.1[k], v[k]);
        if vk == 0.0 {
            if ahi < blo || alo > bhi {
                return None;
            }
            continue;
        }
        let (e, x) = ((blo - ahi) / vk, (bhi - alo) / vk);
        let (e, x) = if e < x { (e, x) } else { (x, e) };
        t0 = t0.max(e);
        t1 = t1.min(x);
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// A data scenario's first contact (s): the earliest time any body's spawn box ([`body_box`]),
/// carried along its velocity, touches a static box's ([`static_box`]) or another body's
/// (carried along its own: their relative velocity). Boxes, not shapes: a contact the boxes
/// predict may come a little before the surfaces touch (a box's corner where the body is
/// rounded), never after. None when nothing meets.
pub fn first_contact(l: &Layout, extent: &dyn Fn(&str) -> Extent) -> Option<f32> {
    let boxes: Vec<(Vec3, Vec3)> = l.bodies.iter().map(|b| body_box(b, &extent(&b.design))).collect();
    let statics: Vec<(Vec3, Vec3)> = l.boxes.iter().enumerate().map(|(i, &(c, h))| static_box(c, h, l.yaw_of(i))).collect();
    let mut best: Option<f32> = None;
    let mut take = |t: Option<f32>| {
        if let Some(t) = t {
            best = Some(best.map_or(t, |b| b.min(t)));
        }
    };
    for (i, b) in l.bodies.iter().enumerate() {
        let v = b.velocity();
        for s in &statics {
            take(sweep(boxes[i], v, *s));
        }
        for (k, o) in l.bodies.iter().enumerate().skip(i + 1) {
            take(sweep(boxes[i], v - o.velocity(), boxes[k]));
        }
    }
    best
}

/// The law-based criteria every recreated scenario carries: finite, no energy created,
/// nothing into a static box (with boxes), and, without boxes, the horizontal momentum
/// changed by no more than the ground's friction and the air's drag can change it.
pub fn law_criteria(boxes: bool) -> fn() -> Vec<Criterion> {
    if boxes {
        || {
            let mut c = laws_common();
            c.push(Criterion::new(Tier::Invariant, "nothing passes into a box", "all.box_depth_m", None, Some(0.05), "no node deeper than 5 cm inside a static box: two steps of a node at 50 m/s").law("a static box is impenetrable"));
            c
        }
    } else {
        || {
            let mut c = laws_common();
            c.push(
                Criterion::new(Tier::Invariant, "momentum within the ground's impulse", "all.momentum_friction_ratio", None, Some(1.0), "the change of the bodies' horizontal momentum over the most the outside can give: μ ≤ 1.2 of the weight for the whole run, plus the drag of a C_d·A ≤ 8 m² body at its starting speed")
                    .law("momentum changes only by the external impulse; with no static box the only horizontal outside forces are the ground's friction (at most μ·N) and air drag"),
            );
            c
        }
    }
}

fn laws_common() -> Vec<Criterion> {
    vec![
        Criterion::new(Tier::Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
        Criterion::new(Tier::Invariant, "no energy created", "all.energy_gain_frac", None, Some(1e-3), "the largest rise of the total energy over the kinetic energy at the start: at most 0.1 %")
            .law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
    ]
}

/// Every data scenario in a folder (`<id>.json`), sorted by id; a file that does not parse
/// or check is an error naming it.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_dir(dir: &std::path::Path) -> Result<Vec<ScenarioDesc>, String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return Ok(out) };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let d: ScenarioDesc = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
            d.check().map_err(|e| format!("{}: {e}", p.display()))?;
            out.push(d);
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}
