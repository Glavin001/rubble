//! The crash reference suite (`engine/docs/crash-reference.md`): real crash tests and their
//! BeamNG counterparts, staged once and run everywhere.
//!
//! A [`CrashScenario`] is defined once here, in host Rust, on the explicit 2 kHz solver:
//! its bodies (generated designs by name, each with its own pose and speed), its static
//! boxes, cameras, named moments and acceptance criteria. [`CrashScenario::build`] turns it
//! into a fleet of [`ExplicitBeams`], which is what every executor takes: the CPU reference
//! ([`CpuRunner`]), CUDA (`engine-cuda`), wgpu and the browser's WebGPU. [`run`] steps any
//! [`Runner`], samples the state and returns a [`CrashReport`]; the tests, the command line,
//! the recorder and the web page all read that report. Nothing about a scenario is described
//! a second time anywhere else.
//!
//! Adding a scenario: one file with a `pub(super) fn scenario() -> CrashScenario`, and one
//! line in [`all`].

pub mod criteria;
pub mod metrics;
mod s02_wall;
mod s03_side_barrier;

use crate::desc::SoftVehicleDesc;
use crate::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use crate::explicit_contact::{Dat, ExplicitBeamsWorld};
pub use criteria::{Basis, Criterion, CriterionResult, Tier};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The solver's step (2 kHz) and how often a run is sampled (every 4 ms).
pub const DT: f32 = 0.0005;
pub const SAMPLE_STEPS: u32 = 8;
/// Runs per scenario: a tier-1 criterion holds on every one.
pub const RUNS: usize = 8;

/// Which part of a vehicle a beam belongs to. The generator names each part's section
/// ([`ZoneMap::parts`]); left and right come from the node's side at rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    /// Bumper, crush rails, subframe: the low load path ahead of the firewall.
    FrontLower,
    /// Hood, fenders, cowl: the high load path ahead of the firewall.
    FrontUpper,
    /// Pillars, sills, roof rails, floor, firewall: the survival cell.
    Cabin,
    Door,
    /// Behind the rear seats: trunk, rear rails, rear bumper.
    Rear,
    Wheel,
    Other,
}

/// A tape measure between two named nodes (door opening, wheelbase, roof height...).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gauge {
    pub name: String,
    pub a: String,
    pub b: String,
}

/// What the generator says about a design for crash measurement (written by
/// `engine-vehiclegen` beside the JBeam as `crash.json`). `parts`: a node-name prefix and
/// its section, longest prefix wins.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ZoneMap {
    pub parts: Vec<(String, Section)>,
    pub gauges: Vec<Gauge>,
}

/// A design ready to spawn.
#[derive(Clone)]
pub struct Template {
    pub desc: SoftVehicleDesc,
    pub settings: ExplicitBeamsSettings,
    pub zones: ZoneMap,
}

pub type Templates = BTreeMap<String, Template>;

/// A design's box at rest in its own frame (forward +Z, up +Y).
#[derive(Clone, Copy, Debug)]
pub struct Extent {
    pub lo: Vec3,
    pub hi: Vec3,
    /// Its front and rear axles along +Z (the wheels' hub midpoints, foremost and
    /// rearmost); None without wheels.
    pub axles: Option<(f32, f32)>,
    /// Its wheels' lowest point (the lowest hub's height less its radius); None without
    /// wheels.
    pub wheel_low: Option<f32>,
}

impl Extent {
    pub fn of(d: &SoftVehicleDesc) -> Self {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for n in &d.nodes {
            lo = lo.min(Vec3::from(n.position));
            hi = hi.max(Vec3::from(n.position));
        }
        let hub = |i: u32| d.nodes[i as usize].position[2];
        let z: Vec<f32> = d.wheels.iter().map(|w| 0.5 * (hub(w.node1) + hub(w.node2))).collect();
        let axles = (!z.is_empty()).then(|| z.iter().fold((f32::MIN, f32::MAX), |(f, r), &x| (f.max(x), r.min(x))));
        let wheel_low = d.wheels.iter().map(|w| 0.5 * (d.nodes[w.node1 as usize].position[1] + d.nodes[w.node2 as usize].position[1]) - w.radius).reduce(f32::min);
        Self { lo, hi, axles, wheel_low }
    }
    pub fn width(&self) -> f32 {
        self.hi.x - self.lo.x
    }
    pub fn length(&self) -> f32 {
        self.hi.z - self.lo.z
    }
}

/// One vehicle in a scenario. `at`: where its own origin stands on the ground; `yaw`: its
/// heading from +Z toward +X (rad); `speed`: along its heading (m/s).
#[derive(Clone, Debug)]
pub struct Body {
    /// The name criteria and reports use (`car`, `truck`, `trailer`...).
    pub name: &'static str,
    pub design: &'static str,
    pub at: Vec3,
    pub yaw: f32,
    pub speed: f32,
    /// Its velocity when it does not move along its heading (a crabbed barrier, a car carried
    /// sideways); None: `heading() * speed`.
    pub vel: Option<Vec3>,
}

impl Body {
    pub fn new(name: &'static str, design: &'static str, at: Vec3, yaw: f32, speed: f32) -> Self {
        Self { name, design, at, yaw, speed, vel: None }
    }
    pub fn heading(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }
    pub fn velocity(&self) -> Vec3 {
        self.vel.unwrap_or(self.heading() * self.speed)
    }
}

/// A scenario laid out for one run: its bodies, its static boxes (centre, half extents)
/// and when the first contact is due (s).
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub bodies: Vec<Body>,
    pub boxes: Vec<(Vec3, Vec3)>,
    pub impact_s: f32,
}

/// A run's small deviation from nominal, so that no criterion is tuned to one lucky run.
/// Run 0 is nominal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Jitter {
    /// Multiplies every speed (within 2 %).
    pub speed: f32,
    /// Moves the striking body across its path (m, within 2 cm).
    pub lateral: f32,
    /// Turns the striking body (rad, within 0.5°).
    pub yaw: f32,
}

impl Jitter {
    pub fn run(k: usize) -> Self {
        // A fixed low-discrepancy table in [-1, 1]: the same eight runs everywhere.
        const H: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [1.0, -0.5, 0.33],
            [-1.0, 0.5, -0.33],
            [0.5, 1.0, -1.0],
            [-0.5, -1.0, 1.0],
            [0.25, 0.75, 0.66],
            [-0.75, -0.25, -0.66],
            [0.75, 0.25, 0.11],
        ];
        let h = H[k % H.len()];
        Self { speed: 1.0 + 0.02 * h[0], lateral: 0.02 * h[1], yaw: 0.5f32.to_radians() * h[2] }
    }
}

/// The three contact settings every scenario is run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Contact {
    Discrete,
    Edge,
    /// Edge contact plus continuous contact.
    Continuous,
}

impl Contact {
    pub const ALL: [Contact; 3] = [Contact::Discrete, Contact::Edge, Contact::Continuous];
    pub fn name(self) -> &'static str {
        match self {
            Contact::Discrete => "discrete",
            Contact::Edge => "edge",
            Contact::Continuous => "continuous",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name() == s)
    }
}

/// A fixed camera for stills and video, relative to the scene's point of impact.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Camera {
    pub name: &'static str,
    pub eye: [f32; 3],
    pub target: [f32; 3],
}

/// A named instant to compare with a reference frame (ms after first contact).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Moment {
    pub name: &'static str,
    pub after_impact_ms: f32,
}

/// Where the reference footage is.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Reference {
    pub video: &'static str,
    /// Seconds into the video.
    pub at_s: &'static [u32],
    pub real_test: &'static str,
}

pub const VIDEO: &str = "https://www.youtube.com/watch?v=Lqx2KKWI8aM";

pub struct CrashScenario {
    pub id: &'static str,
    pub title: &'static str,
    pub reference: Reference,
    /// The designs it needs (the consumer loads these templates).
    pub designs: &'static [&'static str],
    /// The run's layout from each design's extent.
    pub layout: fn(&dyn Fn(&str) -> Extent, Jitter) -> Layout,
    /// Simulated seconds after first contact.
    pub secs_after_impact: f32,
    pub cameras: &'static [Camera],
    pub moments: &'static [Moment],
    pub criteria: fn() -> Vec<Criterion>,
}

/// Every scenario, in order.
pub fn all() -> Vec<CrashScenario> {
    vec![s02_wall::scenario(), s03_side_barrier::scenario()]
}

pub fn get(id: &str) -> Option<CrashScenario> {
    all().into_iter().find(|s| s.id.eq_ignore_ascii_case(id))
}

/// A scenario built for one run.
pub struct Built {
    pub layout: Layout,
    pub fleet: Vec<ExplicitBeams>,
    pub contact: Contact,
}

fn turned(d: &SoftVehicleDesc, yaw: f32) -> SoftVehicleDesc {
    let (s, c) = yaw.sin_cos();
    let mut o = d.clone();
    for n in &mut o.nodes {
        let [x, y, z] = n.position;
        n.position = [x * c + z * s, y, -x * s + z * c];
    }
    o
}

impl CrashScenario {
    pub fn layout_for(&self, templates: &Templates, jitter: Jitter) -> Result<Layout, String> {
        for d in self.designs {
            if !templates.contains_key(*d) {
                return Err(format!("{}: no template for design {d}", self.id));
            }
        }
        Ok((self.layout)(&|d| Extent::of(&templates[d].desc), jitter))
    }

    /// The fleet for one run: each body spawned at its pose and speed with every static box,
    /// in the contact setting asked for.
    pub fn build(&self, templates: &Templates, jitter: Jitter, contact: Contact) -> Result<Built, String> {
        let layout = self.layout_for(templates, jitter)?;
        let mut fleet = Vec::new();
        for b in &layout.bodies {
            let t = templates.get(b.design).ok_or_else(|| format!("{}: no template for design {}", self.id, b.design))?;
            let d = turned(&t.desc, b.yaw);
            // Standing on its wheels (their tread 2 cm up), or on its lowest node when that is lower
            // or it has none.
            let e = Extent::of(&d);
            let lift = 0.02 - e.wheel_low.map_or(e.lo.y, |w| w.min(e.lo.y));
            let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(&d, b.at + Vec3::Y * lift, b.velocity(), t.settings)?;
            s.surface_contact = true;
            s.edge_contact = contact != Contact::Discrete;
            for &(c, h) in &layout.boxes {
                s.boxes.push(StaticBox::centered(c, h));
            }
            s.finish();
            fleet.push(s);
        }
        Ok(Built { layout, fleet, contact })
    }

    /// Frames of [`SAMPLE_STEPS`] steps in a whole run.
    pub fn samples(&self, layout: &Layout) -> usize {
        ((layout.impact_s + self.secs_after_impact) / (DT * SAMPLE_STEPS as f32)).ceil() as usize
    }
}

/// A beam's changing state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeamState {
    pub rest: f32,
    pub broken: bool,
}

/// The state read from an executor: per body, its nodes' world positions and velocities and
/// its beams.
#[derive(Clone, Debug, Default)]
pub struct Sample {
    pub t: f32,
    pub pos: Vec<Vec<Vec3>>,
    pub vel: Vec<Vec<Vec3>>,
    pub beams: Vec<Vec<BeamState>>,
}

/// An executor running a built fleet: the only thing a backend provides.
pub trait Runner {
    fn frame(&mut self, steps: u32) -> Result<(), String>;
    /// Positions, velocities and beams now (`t` is filled in by [`run`]).
    fn sample(&mut self) -> Result<Sample, String>;
    /// Energy the contact response has removed so far (J), where the executor counts it.
    fn contact_removed(&mut self) -> Option<f64> {
        None
    }
}

/// The CPU reference.
pub struct CpuRunner(pub ExplicitBeamsWorld);

impl CpuRunner {
    pub fn new(built: &Built) -> Self {
        let mut w = ExplicitBeamsWorld::new(built.fleet.clone());
        w.continuous = (built.contact == Contact::Continuous).then(|| Dat::new(0.08, 0.85));
        Self(w)
    }
}

impl Runner for CpuRunner {
    fn frame(&mut self, steps: u32) -> Result<(), String> {
        self.0.frame(steps);
        Ok(())
    }
    fn sample(&mut self) -> Result<Sample, String> {
        let v = &self.0.vehicles;
        Ok(Sample {
            t: 0.0,
            pos: v.iter().map(|s| s.positions()).collect(),
            vel: v.iter().map(|s| s.nodes.iter().map(|n| n.vel).collect()).collect(),
            beams: v.iter().map(|s| s.beams.iter().map(|b| BeamState { rest: b.rest, broken: b.broken }).collect()).collect(),
        })
    }
    fn contact_removed(&mut self) -> Option<f64> {
        self.0.continuous.as_ref().map(|d| -d.total.ccd.ke)
    }
}

/// One run's result: every metric by name, and each criterion judged on it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrashReport {
    pub id: String,
    pub contact: Contact,
    pub run: usize,
    pub metrics: BTreeMap<String, f64>,
    pub results: Vec<CriterionResult>,
}

impl CrashReport {
    /// Asserted criteria (tier 0 or 1, with a basis) that did not hold.
    pub fn failures(&self) -> Vec<&CriterionResult> {
        self.results.iter().filter(|r| !r.pass && r.asserted).collect()
    }
    pub fn table(&self) -> String {
        let mut o = format!("{} run {} ({})\n", self.id, self.run, self.contact.name());
        for r in &self.results {
            let mark = match (r.pass, r.asserted) {
                (true, true) => "pass",
                (false, true) => "FAIL",
                (true, false) => "in  ",
                (false, false) => "out ",
            };
            o += &format!("  {mark} [{}] {}: {} against {} ({})\n", if r.asserted { r.tier.name() } else { "reported" }, r.name, r.shown(), r.band(), r.basis.split(':').next().unwrap_or(""));
        }
        o
    }
}

/// Steps `runner` through the scenario, sampling every [`SAMPLE_STEPS`] steps (each sample
/// also handed to `each`: frames for a film, a page's drawing), and judges the result.
pub fn run(
    s: &CrashScenario,
    templates: &Templates,
    built: &Built,
    run: usize,
    runner: &mut dyn Runner,
    each: &mut dyn FnMut(&Sample),
) -> Result<CrashReport, String> {
    let mut rec = metrics::Recorder::new(templates, built);
    let n = s.samples(&built.layout);
    for k in 0..=n {
        let mut x = runner.sample()?;
        x.t = k as f32 * DT * SAMPLE_STEPS as f32;
        rec.push(&x);
        each(&x);
        if k < n {
            runner.frame(SAMPLE_STEPS)?;
        }
    }
    let mut metrics = rec.finish();
    if let Some(j) = runner.contact_removed() {
        metrics.insert("all.contact_removed_j".into(), j);
    }
    let results = (s.criteria)().iter().map(|c| c.judge(&metrics)).collect();
    Ok(CrashReport { id: s.id.into(), contact: built.contact, run, metrics, results })
}
