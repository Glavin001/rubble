//! The crash reference suite (`engine_softbody::crash_reference`,
//! `engine/docs/crash-reference.md`) in the web crash lab
//! (`crash-lab.html?crashRef=S02&run=0&contact=edge`).
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
    DT, SAMPLE_STEPS,
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
    registry_with(&[])
}

/// [`registry`] with data scenarios (`crash_reference::ScenarioDesc`, the Reference Studio's
/// recreated crashes) after the built-in ones, each marked `recreated` and carrying its own
/// description (`desc`) and default contact setting.
pub fn registry_with(data: &[cr::ScenarioDesc]) -> serde_json::Value {
    let r = cr::registry::registry(data, &engine_softbody_authoring::crash_templates::from_code);
    serde_json::to_value(r).unwrap_or_default()
}

/// [`registry`] as JSON text; `data` (optional): the data scenarios to list after the
/// built-in ones (`ScenarioDesc[]` as JSON text: the Reference Studio's workspace).
#[wasm_bindgen(js_name = crashReferenceRegistry)]
pub fn crash_reference_registry(data: Option<String>) -> Result<String, JsValue> {
    let list: Vec<cr::ScenarioDesc> = match data {
        Some(t) if !t.is_empty() => serde_json::from_str(&t).map_err(|e| err(format!("data scenarios: {e}")))?,
        _ => Vec::new(),
    };
    Ok(registry_with(&list).to_string())
}

/// One run in progress: the built fleet, the recorder and the samples read so far.
pub(super) struct CrashRun {
    id: String,
    /// The scenario (built-in or data): its criteria judge the run (`crash_reference::judge`).
    scenario: cr::CrashScenario,
    /// Every sample read, kept for playback (`crashTakeRecording`: the Reference Studio's
    /// recording), when asked for (`crashRef.record`).
    recording: Option<cr::recording::Recording>,
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
            if let Some(r) = self.recording.as_mut() {
                r.push(&x);
            }
            self.pushed += 1;
        }
        if self.pushed == self.samples && self.report.is_none() {
            if let Some(rec) = self.rec.take() {
                let (metrics, series) = rec.finish_with_series();
                let results = cr::judge(&self.scenario, &metrics, self.built.contact);
                self.report = Some(CrashReport { id: self.scenario.id.clone(), contact: self.built.contact, run: self.run, metrics, results, series: Some(series) });
            }
        }
    }
}

/// The scenario's templates: a design built from code alone (the barrier) by
/// `crash_templates::template`, every other from its mod folder's files
/// (`designs: { <design>: [[path, text], ...] }`, the folder `__mods/<design>_new` the site
/// build generates) by `crash_templates::template_from_files`, the constructor
/// `crash_templates::template` itself uses on the rigger's output.
pub(crate) fn templates(s: &cr::CrashScenario, designs: Option<&serde_json::Value>) -> Result<cr::Templates, String> {
    use engine_softbody_authoring::crash_templates as ct;
    let mut out = cr::Templates::new();
    for d in s.designs.iter().map(String::as_str) {
        let t = if ct::from_code(d) {
            ct::template(d)?
        } else {
            let list = designs.and_then(|x| x.get(d)).and_then(|x| x.as_array()).ok_or_else(|| format!("{}: crashRef.designs.{d}: the files of its mod folder ([[path, text], ...])", s.id))?;
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

/// The scenario `crashRef` names: `desc` (a data scenario, `crash_reference::ScenarioDesc`)
/// or `id` (a built-in one), and its default contact setting.
pub(crate) fn scenario_of(c: &serde_json::Value) -> Result<(cr::CrashScenario, Contact), String> {
    if let Some(d) = c.get("desc").filter(|d| !d.is_null()) {
        let d: cr::ScenarioDesc = serde_json::from_value(d.clone()).map_err(|e| format!("crashRef.desc: {e}"))?;
        d.check()?;
        return Ok((d.scenario(), d.contact));
    }
    let id = c.get("id").and_then(|x| x.as_str());
    let family = c.get("family").and_then(|x| x.as_str());
    if id.is_none() && family.is_none() {
        return Err("crashRef.id (a scenario id), crashRef.family (a scenario family) or crashRef.desc (a scenario description)".into());
    }
    let values = params_of(c)?;
    Ok((cr::family::resolve(id, family, &values)?, Contact::Edge))
}

/// `crashRef.params`: family parameter values by key (numbers, flags, names; a number or a
/// flag may be given as text, as a URL holds it).
fn params_of(c: &serde_json::Value) -> Result<cr::family::Values, String> {
    match c.get("params").filter(|p| !p.is_null()) {
        Some(p) => serde_json::from_value(p.clone()).map_err(|e| format!("crashRef.params: {e}")),
        None => Ok(cr::family::Values::default()),
    }
}

/// The scenario `spec` names (`{ id | family, params }`, as `spec.crashRef`) as the page
/// lists it (`crash_reference::registry::ScenarioInfo`: its designs, criteria, family and
/// values): what a page shows for a reference crash with changed values, or a family opened
/// directly, before it spawns it.
#[wasm_bindgen(js_name = crashReferenceScenario)]
pub fn crash_reference_scenario(spec: String) -> Result<String, JsValue> {
    scenario_info(&spec).map_err(err)
}

fn scenario_info(spec: &str) -> Result<String, String> {
    let c: serde_json::Value = serde_json::from_str(spec).map_err(|e| format!("crashReferenceScenario: {e}"))?;
    let (s, _) = scenario_of(&c)?;
    let info = cr::registry::info(&s, None, &engine_softbody_authoring::crash_templates::from_code);
    Ok(serde_json::to_string(&info).unwrap_or_default())
}

/// `spec.crashRef`: `{ id | desc, run (default 0), contact ('discrete' | 'edge' |
/// 'continuous', default 'edge' or the description's), designs (see [`templates`]), record
/// (keep every sample: `crashTakeRecording`) }`: the scenario's fleet for that run,
/// [`cr::CrashScenario::build`] as every executor takes it.
pub(super) fn spawn(c: &serde_json::Value) -> Result<Spawn, String> {
    let (s, default_contact) = scenario_of(c)?;
    let run = c.get("run").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let contact = match c.get("contact").and_then(|x| x.as_str()) {
        Some(n) => Contact::from_name(n).ok_or_else(|| format!("no contact setting {n}"))?,
        None => default_contact,
    };
    let templates = templates(&s, c.get("designs"))?;
    let built = s.build(&templates, Jitter::run(run), contact)?;
    let mut vehicles = Vec::new();
    for (b, sim) in built.layout.bodies.iter().zip(&built.fleet) {
        vehicles.push(Spawned {
            name: templates[&b.design].desc.name.clone(),
            sim: sim.clone(),
            reference: super::reference_node(sim),
            forward: b.heading(),
            warnings: Vec::new(),
        });
    }
    let samples = s.samples(&built.layout) + 1;
    let rec = Recorder::new(&templates, &built);
    let record = c.get("record").and_then(|x| x.as_bool()).unwrap_or(false);
    let recording = record.then(|| cr::recording::Recording::new(&s, &built, "webgpu"));
    // Turned boxes are oriented box shapes (drawn as shapes); the others the layout's boxes.
    let l = &built.layout;
    let boxes = (0..l.boxes.len()).filter(|&i| l.yaw_of(i) == 0.0).map(|i| l.boxes[i]).collect();
    let shapes = (0..l.boxes.len()).filter(|&i| l.yaw_of(i) != 0.0).map(|i| engine_softbody::explicit_beams::StaticShape::oriented_box(l.boxes[i].0, l.boxes[i].1, l.rotation_of(i))).collect();
    Ok(Spawn {
        vehicles,
        boxes,
        shapes,
        pull: None,
        scene: format!("crashRef:{}", s.id),
        crash: Some(CrashRun { id: s.id.clone(), scenario: s.clone(), recording, run, built, samples, rec: Some(rec), issued: 0, pushed: 0, waiting: BTreeMap::new(), report: None, error: None }),
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
        let (family, values) = c.scenario.family().map_or((None, None), |(f, v)| (Some(f.id), Some(v.clone())));
        super::js(serde_json::json!({
            "id": c.id, "run": c.run, "contact": c.built.contact.name(),
            "family": family, "values": values, "modifiedFrom": c.scenario.modified_from,
            "impactS": l.impact_s, "impactStep": (l.impact_s / DT).round(),
            "sampleSteps": SAMPLE_STEPS, "samples": c.samples,
            "totalSteps": (c.samples - 1) as u32 * SAMPLE_STEPS,
            "bodies": l.bodies.iter().map(|b| serde_json::json!({ "name": &b.name, "design": &b.design, "at": b.at.to_array(), "yaw": b.yaw, "speed": b.speed })).collect::<Vec<_>>(),
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
    /// results: [{ name, metric, value, lo, hi, tier, pass, asserted, why, basis, known,
    /// known_occurs?, known_note? }] }`: `known`, a known failure recorded for the run's
    /// contact setting), else null.
    #[wasm_bindgen(js_name = crashReport)]
    pub fn crash_report(&self) -> Result<JsValue, JsValue> {
        let r = run_of(self)?.borrow();
        if let Some(e) = &r.error {
            return Err(err(e.clone()));
        }
        Ok(r.report.as_ref().map_or(JsValue::NULL, |x| super::js(serde_json::to_value(x).unwrap_or_default())))
    }

    /// The run's recording (every sample's node positions and broken beams, the Reference
    /// Studio's playback: `crash_reference::recording::Recording::to_bytes`) once every
    /// sample is in and `crashRef.record` was set, else null. Taken: a second call is null.
    #[wasm_bindgen(js_name = crashTakeRecording)]
    pub fn crash_take_recording(&self) -> Result<JsValue, JsValue> {
        let mut r = run_of(self)?.borrow_mut();
        if r.report.is_none() {
            return Ok(JsValue::NULL);
        }
        let report = r.report.clone();
        Ok(match r.recording.take() {
            Some(mut rec) => {
                rec.report = report;
                js_sys::Uint8Array::from(&rec.to_bytes()[..]).into()
            }
            None => JsValue::NULL,
        })
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
                assert!(a["basis"]["kind"].is_string());
                assert_eq!(a["asserted"], c.asserted());
            }
            assert_eq!(g["cameras"].as_array().unwrap().len(), s.cameras.len());
            assert_eq!(g["moments"].as_array().unwrap().len(), s.moments.len());
            assert_eq!(g["reference"]["links"].as_array().unwrap().len(), s.reference.at_s.len());
        }
        // Every scenario is a family preset: its family listed with its parameters, its values
        // every parameter's.
        let fams = r["families"].as_array().unwrap();
        assert_eq!(fams.len(), cr::families::all().len());
        for (g, s) in got.iter().zip(&all) {
            let f = fams.iter().find(|f| f["id"] == g["family"]).unwrap_or_else(|| panic!("{}: family {}", s.id, g["family"]));
            let keys: Vec<&str> = f["params"].as_array().unwrap().iter().map(|p| p["key"].as_str().unwrap()).collect();
            assert_eq!(g["values"].as_object().unwrap().keys().map(|k| k.as_str()).collect::<std::collections::BTreeSet<_>>(), keys.iter().copied().collect(), "{}", s.id);
            assert!(f["presets"].as_array().unwrap().iter().any(|p| p == &g["id"]), "{}", s.id);
        }
        // A reference crash with a value changed: the law-based criteria, marked modified; a
        // family opened directly.
        let m: serde_json::Value = serde_json::from_str(&scenario_info(r#"{"id":"S11","params":{"speed":"64","car":"wagon"}}"#).unwrap()).unwrap();
        assert_eq!(m["modifiedFrom"], "S11");
        assert_eq!(m["designs"], serde_json::json!(["wagon"]));
        assert!(m["criteria"].as_array().unwrap().iter().all(|c| c["basis"]["kind"] == "law"));
        let same: serde_json::Value = serde_json::from_str(&scenario_info(r#"{"id":"S11","params":{"speed":56.3}}"#).unwrap()).unwrap();
        assert!(same.get("modifiedFrom").is_none(), "the preset's own value");
        let f: serde_json::Value = serde_json::from_str(&scenario_info(r#"{"family":"pole","params":{"offset":0.4}}"#).unwrap()).unwrap();
        assert_eq!(f["family"], "pole");
        assert!(scenario_info(r#"{"id":"S11","params":{"speed":-3}}"#).is_err());
        assert_eq!(r["contacts"].as_array().unwrap().len(), Contact::ALL.len());
        assert!(got.iter().all(|g| g["folderDesigns"].as_array().is_some()));
        // The text the page parses is the same value.
        let text: serde_json::Value = serde_json::from_str(&crash_reference_registry(None).unwrap()).unwrap();
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
        for d in &s.designs {
            let (a, b) = (&web[d], &native[d]);
            assert_eq!(a.desc.nodes.len(), b.desc.nodes.len(), "{d}");
            assert_eq!(a.desc.beams.len(), b.desc.beams.len(), "{d}");
            assert_eq!(serde_json::to_string(&a.desc).unwrap(), serde_json::to_string(&b.desc).unwrap(), "{d}");
            assert_eq!(a.zones, b.zones, "{d}");
        }
    }
}
