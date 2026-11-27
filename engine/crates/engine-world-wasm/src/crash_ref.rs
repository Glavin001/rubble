//! The crash reference suite (`engine_softbody::crash_reference`,
//! `engine/docs/crash-reference.md`) in the web crash lab
//! (`softbody.html?solver=explicit&crashRef=S02&run=0&contact=edge`).
//!
//! Nothing about a scenario is described here or in the page. The registry the page lists
//! is [`crash_reference::all`] as JSON ([`registry`]); a scenario's fleet is
//! [`CrashScenario::build`] on the templates
//! [`engine_softbody_authoring::crash_templates::templates_for`] rigs (the function the CPU
//! reference and CUDA use: the generated designs rigged from the generator's code, here in
//! the wasm); the report is [`metrics::Recorder`] and each criterion's
//! [`Criterion::judge`](engine_softbody::crash_reference::Criterion::judge), fed by reads of
//! the GPU state every [`SAMPLE_STEPS`] steps (the sampling [`crash_reference::run`] does).
//! What the web adds is the read: positions, velocities and beams copied from the solver's
//! buffers after each sample's step, decoded into a [`Sample`] in order.

use super::{ExplicitSim, Spawn, Spawned};
use crate::err;
use engine_softbody::crash_reference::{
    self as cr, metrics::Recorder, BeamState, Built, Contact, CrashReport, Jitter, Sample,
    DT, RUNS, SAMPLE_STEPS,
};
use glam::Vec3;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

/// The registry as the page reads it: every scenario of [`cr::all`], in order, with its
/// reference footage (and a link at each of its times), designs, cameras (world
/// coordinates), moments (ms after first contact) and criteria (band, tier, basis, why,
/// whether failing it fails a run), and the contact settings and run count a run takes.
pub fn registry() -> serde_json::Value {
    let scenarios: Vec<serde_json::Value> = cr::all()
        .iter()
        .map(|s| {
            let criteria: Vec<serde_json::Value> = (s.criteria)()
                .iter()
                .map(|c| {
                    let mut v = serde_json::to_value(c).unwrap_or_default();
                    v["asserted"] = c.asserted().into();
                    v
                })
                .collect();
            serde_json::json!({
                "id": s.id,
                "title": s.title,
                "reference": {
                    "video": s.reference.video,
                    "atS": s.reference.at_s,
                    "links": s.reference.at_s.iter().map(|t| format!("{}&t={t}s", s.reference.video)).collect::<Vec<_>>(),
                    "realTest": s.reference.real_test,
                },
                "designs": s.designs,
                // The designs the page loads a mod folder for (the others are built from code).
                "folderDesigns": s.designs.iter().filter(|d| !engine_softbody_authoring::crash_templates::from_code(d)).collect::<Vec<_>>(),
                "secsAfterImpact": s.secs_after_impact,
                "cameras": s.cameras,
                "moments": s.moments,
                "criteria": criteria,
            })
        })
        .collect();
    serde_json::json!({
        "scenarios": scenarios,
        "contacts": Contact::ALL.iter().map(|c| c.name()).collect::<Vec<_>>(),
        "runs": RUNS,
        "sampleSteps": SAMPLE_STEPS,
        "dt": DT,
    })
}

/// [`registry`] as JSON text.
#[wasm_bindgen(js_name = crashReferenceRegistry)]
pub fn crash_reference_registry() -> String {
    registry().to_string()
}

/// One run in progress: the built fleet, the recorder and the samples read so far.
pub(super) struct CrashRun {
    id: String,
    run: usize,
    built: Built,
    /// Samples in a whole run (`CrashScenario::samples` + 1: the first at step 0).
    samples: usize,
    rec: Option<Recorder>,
    /// The next sample to read, and the next to hand the recorder.
    issued: usize,
    pushed: usize,
    /// Read samples waiting for an earlier one.
    waiting: BTreeMap<usize, Sample>,
    report: Option<CrashReport>,
    error: Option<String>,
}

impl CrashRun {
    fn push(&mut self, k: usize, x: Sample) {
        self.waiting.insert(k, x);
        while let Some(x) = self.waiting.remove(&self.pushed) {
            if let Some(r) = self.rec.as_mut() {
                r.push(&x);
            }
            self.pushed += 1;
        }
        if self.pushed == self.samples && self.report.is_none() {
            if let (Some(rec), Some(s)) = (self.rec.take(), cr::get(&self.id)) {
                let (metrics, series) = rec.finish_with_series();
                let results = (s.criteria)().iter().map(|c| c.judge(&metrics)).collect();
                self.report = Some(CrashReport { id: s.id.into(), contact: self.built.contact, run: self.run, metrics, results, series: Some(series) });
            }
        }
    }
}

/// The scenario's templates: a design built from code alone (the barrier) by
/// `crash_templates::template`, every other from its mod folder's files
/// (`designs: { <design>: [[path, text], ...] }`, the folder `__mods/<design>_new` the site
/// build generates) by `crash_templates::template_from_files`, the constructor
/// `crash_templates::template` itself uses on the rigger's output.
fn templates(s: &cr::CrashScenario, designs: Option<&serde_json::Value>) -> Result<cr::Templates, String> {
    use engine_softbody_authoring::crash_templates as ct;
    let mut out = cr::Templates::new();
    for d in s.designs {
        let t = if ct::from_code(d) {
            ct::template(d)?
        } else {
            let list = designs.and_then(|x| x.get(*d)).and_then(|x| x.as_array()).ok_or_else(|| format!("{}: crashRef.designs.{d}: the files of its mod folder ([[path, text], ...])", s.id))?;
            // A folder holds one file per path (files.json may list a path twice: the later
            // is the file).
            let mut files: Vec<(String, Vec<u8>)> = Vec::new();
            for f in list {
                let path = f.get(0).and_then(|x| x.as_str()).unwrap_or("").to_string();
                let text = f.get(1).and_then(|x| x.as_str()).unwrap_or("").as_bytes().to_vec();
                files.retain(|(p, _)| *p != path);
                files.push((path, text));
            }
            ct::template_from_files(d, &files)?
        };
        out.insert(d.to_string(), t);
    }
    Ok(out)
}

/// `spec.crashRef`: `{ id, run (default 0), contact ('discrete' | 'edge' | 'continuous',
/// default 'edge'), designs (see [`templates`]) }`: the scenario's fleet for that run,
/// [`cr::CrashScenario::build`] as every executor takes it.
pub(super) fn spawn(c: &serde_json::Value) -> Result<Spawn, String> {
    let id = c.get("id").and_then(|x| x.as_str()).ok_or("crashRef.id: a scenario id")?;
    let s = cr::get(id).ok_or_else(|| format!("no crash reference scenario {id}: {}", cr::all().iter().map(|s| s.id).collect::<Vec<_>>().join(", ")))?;
    let run = c.get("run").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let contact = match c.get("contact").and_then(|x| x.as_str()) {
        Some(n) => Contact::from_name(n).ok_or_else(|| format!("no contact setting {n}"))?,
        None => Contact::Edge,
    };
    let templates = templates(&s, c.get("designs"))?;
    let built = s.build(&templates, Jitter::run(run), contact)?;
    let mut vehicles = Vec::new();
    for (b, sim) in built.layout.bodies.iter().zip(&built.fleet) {
        vehicles.push(Spawned {
            name: templates[b.design].desc.name.clone(),
            sim: sim.clone(),
            reference: super::reference_node(sim),
            forward: b.heading(),
            warnings: Vec::new(),
        });
    }
    let samples = s.samples(&built.layout) + 1;
    let rec = Recorder::new(&templates, &built);
    Ok(Spawn {
        vehicles,
        boxes: built.layout.boxes.clone(),
        shapes: Vec::new(),
        pull: None,
        scene: format!("crashRef:{}", s.id),
        crash: Some(CrashRun { id: s.id.into(), run, built, samples, rec: Some(rec), issued: 0, pushed: 0, waiting: BTreeMap::new(), report: None, error: None }),
    })
}

/// The GPU solver's settings for a run: as `engine-cuda`'s `CudaRunner` (vehicle contact
/// on, the default; continuous contact for `Contact::Continuous`, on the split group path:
/// group split on, self split and the wide passes off, as the vehicle lab's
/// `continuous_on` sets it).
pub(super) fn configure(gpu: &mut engine_webgpu::ExplicitBeamsGpu, run: &CrashRun) -> Result<(), String> {
    gpu.set_vehicle_contact(true);
    if run.built.contact == Contact::Continuous {
        gpu.set_continuous(true)?;
        gpu.set_group_split(true);
        gpu.set_self_split(false);
        let _ = gpu.set_wide(false);
    }
    Ok(())
}

fn run_of(sim: &ExplicitSim) -> Result<&Rc<RefCell<CrashRun>>, JsValue> {
    sim.crash.as_ref().ok_or_else(|| err("not a crash reference run (spec.crashRef)"))
}

#[wasm_bindgen]
impl ExplicitSim {
    /// The run's layout and schedule: `{ id, run, contact, impactS, impactStep,
    /// sampleSteps, samples, totalSteps, bodies: [{ name, design, at, yaw, speed }], boxes }`
    /// (null when this sim is not a crash reference run).
    #[wasm_bindgen(js_name = crashInfo)]
    pub fn crash_info(&self) -> JsValue {
        let Some(c) = &self.crash else { return JsValue::NULL };
        let c = c.borrow();
        let l = &c.built.layout;
        super::js(serde_json::json!({
            "id": c.id, "run": c.run, "contact": c.built.contact.name(),
            "impactS": l.impact_s, "impactStep": (l.impact_s / DT).round(),
            "sampleSteps": SAMPLE_STEPS, "samples": c.samples,
            "totalSteps": (c.samples - 1) as u32 * SAMPLE_STEPS,
            "bodies": l.bodies.iter().map(|b| serde_json::json!({ "name": b.name, "design": b.design, "at": b.at.to_array(), "yaw": b.yaw, "speed": b.speed })).collect::<Vec<_>>(),
            "boxes": l.boxes.iter().map(|(c, h)| [c.x, c.y, c.z, h.x, h.y, h.z]).collect::<Vec<_>>(),
        }))
    }

    /// Read the state for the run's next sample, which must be the state now: the steps
    /// run so far are that sample's (`sampleSteps` × its index). The copy is submitted now
    /// (frames submitted after it are not in it); once mapped, the state goes to the
    /// recorder in sample order. Resolves to the samples read so far, or null when the
    /// run is complete or the steps are not the next sample's (an error once a sample is
    /// passed: the report would be wrong).
    #[wasm_bindgen(js_name = crashSample)]
    pub fn crash_sample(&self) -> Result<js_sys::Promise, JsValue> {
        let run = run_of(self)?.clone();
        let k = {
            let mut r = run.borrow_mut();
            let at = self.steps / u64::from(SAMPLE_STEPS);
            if r.issued >= r.samples || self.steps % u64::from(SAMPLE_STEPS) != 0 || (at as usize) < r.issued {
                return Ok(js_sys::Promise::resolve(&JsValue::NULL));
            }
            if at as usize > r.issued {
                let e = format!("crashSample: sample {} was not read (the sim is at sample {at})", r.issued);
                r.error = Some(e.clone());
                return Err(err(e));
            }
            r.issued += 1;
            at as usize
        };
        let nodes = self.gpu.nodes_buffer().clone();
        let beams = self.gpu.beams_buffer().clone();
        let misc = self.gpu.misc_buffer().clone();
        let mut ranges = Vec::new();
        let mut counts = Vec::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            let (n, m) = (v.sim.nodes.len() as u64, v.sim.beams.len() as u64);
            ranges.push((misc.clone(), i as u64 * 16, 16));
            ranges.push((nodes.clone(), u64::from(self.gpu.node_offset(i)) * 48, n * 48));
            ranges.push((beams.clone(), u64::from(self.gpu.beam_offset(i)) * 32, m * 32));
            counts.push((n as usize, m as usize));
        }
        Ok(crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let f = |x: u32| f32::from_bits(x);
            let mut x = Sample { t: k as f32 * DT * SAMPLE_STEPS as f32, ..Default::default() };
            let mut at = 0;
            for (n, m) in counts {
                let o = Vec3::new(f(w[at]), f(w[at + 1]), f(w[at + 2]));
                at += 4;
                let (mut pos, mut vel) = (Vec::with_capacity(n), Vec::with_capacity(n));
                for j in 0..n {
                    let r = &w[at + j * 12..at + j * 12 + 12];
                    pos.push(o + Vec3::new(f(r[0]), f(r[1]), f(r[2])));
                    vel.push(Vec3::new(f(r[4]), f(r[5]), f(r[6])));
                }
                at += n * 12;
                let st = (0..m).map(|j| BeamState { rest: f(w[at + j * 8]), broken: f(w[at + j * 8 + 6]) != 0.0 }).collect();
                at += m * 8;
                x.pos.push(pos);
                x.vel.push(vel);
                x.beams.push(st);
            }
            let mut r = run.borrow_mut();
            r.push(k, x);
            Ok(JsValue::from(r.pushed as u32))
        }))
    }

    /// The run's [`CrashReport`] once every sample is in (`{ id, contact, run, metrics,
    /// results: [{ name, metric, value, lo, hi, tier, pass, asserted, why, basis }] }`),
    /// else null.
    #[wasm_bindgen(js_name = crashReport)]
    pub fn crash_report(&self) -> Result<JsValue, JsValue> {
        let r = run_of(self)?.borrow();
        if let Some(e) = &r.error {
            return Err(err(e.clone()));
        }
        Ok(r.report.as_ref().map_or(JsValue::NULL, |x| super::js(serde_json::to_value(x).unwrap_or_default())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The page lists exactly the suite: every scenario of `crash_reference::all()`, in
    /// order, with each one's criteria, cameras and moments.
    #[test]
    fn registry_lists_the_suite() {
        let r = registry();
        let got = r["scenarios"].as_array().unwrap();
        let all = cr::all();
        assert_eq!(got.len(), all.len());
        for (g, s) in got.iter().zip(&all) {
            assert_eq!(g["id"], s.id);
            assert_eq!(g["title"], s.title);
            let crit = (s.criteria)();
            let gc = g["criteria"].as_array().unwrap();
            assert_eq!(gc.len(), crit.len(), "{}", s.id);
            for (a, c) in gc.iter().zip(&crit) {
                assert_eq!(a["metric"], c.metric);
                assert_eq!(a["asserted"], c.asserted());
            }
            assert_eq!(g["cameras"].as_array().unwrap().len(), s.cameras.len());
            assert_eq!(g["moments"].as_array().unwrap().len(), s.moments.len());
            assert_eq!(g["reference"]["links"].as_array().unwrap().len(), s.reference.at_s.len());
        }
        assert_eq!(r["contacts"].as_array().unwrap().len(), Contact::ALL.len());
        assert!(got.iter().all(|g| g["folderDesigns"].as_array().is_some()));
        // The text the page parses is the same value.
        let text: serde_json::Value = serde_json::from_str(&crash_reference_registry()).unwrap();
        assert_eq!(text, r);
    }

    /// The page's templates are the CPU reference's: a design's mod folder as the site build
    /// writes it (the rigger's files, one per path, the later of a repeated path) gives the
    /// template `crash_templates::template` rigs from code. Slow (it rigs the sedan):
    /// `cargo test --release -p engine-world-wasm --lib folder_templates -- --ignored`.
    #[test]
    #[ignore]
    fn folder_templates_are_the_reference_templates() {
        use engine_softbody_authoring::{authoring::rig, crash_templates as ct};
        let s = cr::get("S03").unwrap();
        let built = rig::rig_vehicle("sedan", None, None).unwrap();
        let mut folder: Vec<(String, Vec<u8>)> = Vec::new();
        for (p, b) in rig::mod_files(&built) {
            folder.retain(|(q, _)| *q != p);
            folder.push((p, b));
        }
        let files: Vec<serde_json::Value> = folder.iter().filter(|(p, _)| p.ends_with(".jbeam") || p.ends_with(".pc") || p.ends_with("crash.json")).map(|(p, b)| serde_json::json!([p, String::from_utf8_lossy(b)])).collect();
        let web = templates(&s, Some(&serde_json::json!({ "sedan": files }))).unwrap();
        let native = ct::templates_for(&s).unwrap();
        for d in s.designs {
            let (a, b) = (&web[*d], &native[*d]);
            assert_eq!(a.desc.nodes.len(), b.desc.nodes.len(), "{d}");
            assert_eq!(a.desc.beams.len(), b.desc.beams.len(), "{d}");
            assert_eq!(serde_json::to_string(&a.desc).unwrap(), serde_json::to_string(&b.desc).unwrap(), "{d}");
            assert_eq!(a.zones, b.zones, "{d}");
        }
    }
}
