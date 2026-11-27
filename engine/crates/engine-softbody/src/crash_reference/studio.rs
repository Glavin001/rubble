//! The Reference Studio's scene, as the page's wasm runs it (`engine-world-wasm`
//! `studio.rs` binds it): a scenario placed from its description (or a built-in one by id),
//! what the page draws of it (each body's collision surface, the static boxes), the
//! recording of its run (made on the page's GPU and handed in, or by the CPU reference for
//! the automated smoke test only), and everything computed from the recording: positions at
//! any time, Rubble's outline of a body at an image's pinned time through its pinned camera,
//! and the comparison table. The page computes none of it.

use super::outline::{BodyParts, FrameCamera, View};
use super::recording::Recording;
use super::registry::CriterionInfo;
use super::shape::{self, rubble, Instance, Schema, Shape, Source};
use super::*;
use serde_json::{json, Value};
use ts_rs::TS;

/// What a scene is made from: a data scenario's description, or a built-in scenario's id.
#[derive(Clone, Debug, Serialize, Deserialize, TS)]
pub struct SceneSource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub desc: Option<ScenarioDesc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub id: Option<String>,
    /// The contact setting (default: the description's, or edge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub contact: Option<Contact>,
}

impl SceneSource {
    /// The scenario, its contact setting and the hash its recording is filed under.
    pub fn resolve(&self) -> Result<(CrashScenario, Contact, String), String> {
        if let Some(d) = &self.desc {
            d.check()?;
            let c = self.contact.unwrap_or(d.contact);
            return Ok((d.scenario(), c, d.digest(c)));
        }
        let id = self.id.as_deref().ok_or("a scene is { desc } or { id }")?;
        let s = get(id).ok_or_else(|| format!("no scenario {id}"))?;
        let c = self.contact.unwrap_or(Contact::Edge);
        let h = format!("builtin-{}-{}", s.id, c.name());
        Ok((s, c, h))
    }
}

/// One body as the page draws it.
#[derive(Clone, Debug, Serialize, TS)]
pub struct SceneBody {
    pub name: String,
    pub design: String,
    /// Its first node in the scene's position arrays (all bodies' nodes in order).
    pub node_offset: u32,
    pub nodes: u32,
    /// Its collision surface: node indices (its own), three per triangle.
    pub triangles: Vec<u32>,
    pub mass_kg: f64,
    pub at: [f32; 3],
    pub yaw_deg: f32,
    pub speed_kmh: f32,
    /// Its box at spawn (world), as the first contact is computed from.
    pub box_lo: [f32; 3],
    pub box_hi: [f32; 3],
}

/// A static box as the page draws it.
#[derive(Clone, Debug, Serialize, TS)]
pub struct SceneBox {
    pub name: String,
    pub centre: [f32; 3],
    pub half: [f32; 3],
    pub yaw_deg: f32,
}

/// A placed scene.
#[derive(Clone, Debug, Serialize, TS)]
pub struct SceneInfo {
    pub id: String,
    pub title: String,
    pub recreated: bool,
    /// What its recording is filed under (the description and contact setting).
    pub hash: String,
    pub contact: Contact,
    /// First contact (s from the start): computed for a data scenario
    /// ([`desc::first_contact`]), stated by a Rust one.
    pub impact_s: f32,
    pub duration_s: f32,
    pub samples: u32,
    pub sample_s: f32,
    pub bodies: Vec<SceneBody>,
    pub boxes: Vec<SceneBox>,
    pub cameras: Vec<Camera>,
    pub moments: Vec<Moment>,
    pub criteria: Vec<CriterionInfo>,
    /// Milliseconds the placement took (the live-edit cost).
    pub place_ms: f64,
}

/// The recording's state against the placed scene.
#[derive(Clone, Debug, Serialize, TS)]
pub struct RecordingStatus {
    pub present: bool,
    /// Made from the scene as it is now (its hash is the scene's).
    pub fresh: bool,
    pub executor: String,
    pub hash: String,
    pub samples: u32,
    pub duration_s: f32,
    pub impact_s: f32,
    pub sample_s: f32,
    /// The first and last sample's time (ms after first contact).
    pub start_ms: f32,
    pub end_ms: f32,
    pub has_report: bool,
}

/// A CPU run in progress (the smoke test's fallback).
#[derive(Clone, Debug, Serialize, TS)]
pub struct CpuProgress {
    pub sample: u32,
    pub samples: u32,
    pub done: bool,
}

/// Rubble's outline of one body at a time through a camera (or in its own side frame).
#[derive(Clone, Debug, Deserialize, TS)]
pub struct RubbleSpec {
    /// Ms after first contact (between samples: interpolated).
    pub at_ms: f32,
    /// The image's camera; absent: the car's own side frame (metres).
    #[serde(default)]
    #[ts(optional)]
    pub camera: Option<FrameCamera>,
    #[serde(default = "one")]
    pub width: f32,
    #[serde(default = "one")]
    pub height: f32,
    #[serde(default)]
    pub body: u32,
}

fn one() -> f32 {
    1.0
}

/// A traced shape (an instance in its region's normalised coordinates and the region's
/// size in pixels).
#[derive(Clone, Debug, Deserialize, TS)]
pub struct TracingSpec {
    pub instance: Instance,
    pub width: f64,
    pub height: f64,
}

/// One shape of a comparison: a tracing or Rubble's.
#[derive(Clone, Debug, Deserialize, TS)]
pub struct ShapeSpec {
    #[serde(default)]
    #[ts(optional)]
    pub tracing: Option<TracingSpec>,
    #[serde(default)]
    #[ts(optional)]
    pub rubble: Option<RubbleSpec>,
}

#[derive(Clone, Debug, Deserialize, TS)]
pub struct CompareSource {
    pub name: String,
    pub before: ShapeSpec,
    pub now: ShapeSpec,
}

struct Scene {
    scenario: CrashScenario,
    built: Built,
    parts: Vec<BodyParts>,
    info: SceneInfo,
}

struct CpuRun {
    runner: CpuRunner,
    rec: metrics::Recorder,
    recording: Recording,
    k: usize,
    n: usize,
}

/// The studio's scene and recording.
#[derive(Default)]
pub struct Studio {
    pub templates: Templates,
    scene: Option<Scene>,
    recording: Option<Recording>,
    cpu: Option<CpuRun>,
}

fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64() * 1000.0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

impl Studio {
    pub fn new() -> Self {
        Self::default()
    }

    /// The designs `src` needs that have no template yet.
    pub fn missing(&self, src: &SceneSource) -> Result<Vec<String>, String> {
        let (s, _, _) = src.resolve()?;
        Ok(s.designs.into_iter().filter(|d| !self.templates.contains_key(d)).collect())
    }

    /// Place the scene: build its fleet (as every executor takes it) and describe what the
    /// page draws. A recording made before stays (its status says whether it is still the
    /// scene's). `clock`: the host's millisecond clock (wasm has no std clock), for
    /// [`SceneInfo::place_ms`].
    pub fn place(&mut self, src: &SceneSource, clock: &dyn Fn() -> f64) -> Result<SceneInfo, String> {
        let t0 = clock();
        let (s, contact, hash) = src.resolve()?;
        let built = s.build(&self.templates, Jitter::run(0), contact)?;
        let mut bodies = Vec::new();
        let mut parts = Vec::new();
        let mut offset = 0u32;
        for (b, f) in built.layout.bodies.iter().zip(&built.fleet) {
            let t = &self.templates[&b.design];
            let (lo, hi) = desc::body_box(b, &Extent::of(&t.desc));
            let triangles: Vec<u32> = f.triangles.iter().enumerate().filter(|(k, _)| !f.tri_anti_only(*k)).flat_map(|(_, x)| x.nodes).collect();
            bodies.push(SceneBody {
                name: b.name.clone(),
                design: b.design.clone(),
                node_offset: offset,
                nodes: f.nodes.len() as u32,
                triangles,
                mass_kg: f.nodes.iter().map(|n| f64::from(n.mass)).sum(),
                at: b.at.to_array(),
                yaw_deg: b.yaw.to_degrees(),
                speed_kmh: b.speed * 3.6,
                box_lo: lo.to_array(),
                box_hi: hi.to_array(),
            });
            parts.push(BodyParts::new(b, f, t));
            offset += f.nodes.len() as u32;
        }
        let l = &built.layout;
        let names: Vec<String> = match &s.layout {
            LayoutSource::Data(d) => d.boxes.iter().map(|x| x.name.clone()).collect(),
            LayoutSource::Code(_) | LayoutSource::Family { .. } => Vec::new(),
        };
        let boxes = l.boxes.iter().enumerate().map(|(i, (c, h))| SceneBox { name: names.get(i).cloned().unwrap_or(format!("box{i}")), centre: c.to_array(), half: h.to_array(), yaw_deg: l.yaw_of(i).to_degrees() }).collect();
        let samples = s.samples(l) as u32 + 1;
        let sample_s = DT * SAMPLE_STEPS as f32;
        let info = SceneInfo {
            id: s.id.clone(),
            title: s.title.clone(),
            recreated: s.recreated,
            hash,
            contact,
            impact_s: l.impact_s,
            duration_s: (samples - 1) as f32 * sample_s,
            samples,
            sample_s,
            bodies,
            boxes,
            cameras: s.cameras.clone(),
            moments: s.moments.clone(),
            criteria: (s.criteria)().iter().map(CriterionInfo::from).collect(),
            place_ms: clock() - t0,
        };
        self.cpu = None;
        self.scene = Some(Scene { scenario: s, built, parts, info: info.clone() });
        Ok(info)
    }

    /// [`Self::place`] with the native clock.
    pub fn place_now(&mut self, src: &SceneSource) -> Result<SceneInfo, String> {
        self.place(src, &now_ms)
    }

    fn scene(&self) -> Result<&Scene, String> {
        self.scene.as_ref().ok_or_else(|| "no scene placed".into())
    }

    /// Every body's node positions at spawn (x, y, z, bodies in order).
    pub fn spawn_positions(&self) -> Result<Vec<f32>, String> {
        Ok(self.scene()?.built.fleet.iter().flat_map(|f| f.positions()).flat_map(|p| p.to_array()).collect())
    }

    /// Start the CPU reference's run of the placed scene: the automated smoke test's
    /// fallback where the browser has no WebGPU. Its recording is labelled `cpu-fallback`.
    pub fn cpu_start(&mut self) -> Result<(), String> {
        let sc = self.scene()?;
        let mut recording = Recording::new(&sc.scenario, &sc.built, "cpu-fallback");
        recording.hash = sc.info.hash.clone();
        let rec = metrics::Recorder::new(&self.templates, &sc.built);
        let n = sc.scenario.samples(&sc.built.layout);
        self.cpu = Some(CpuRun { runner: CpuRunner::new(&sc.built), rec, recording, k: 0, n });
        Ok(())
    }

    /// Up to `samples` more samples of the CPU run; when it is complete its recording (with
    /// the report) becomes the studio's.
    pub fn cpu_step(&mut self, samples: u32) -> Result<CpuProgress, String> {
        let scenario = self.scene()?.scenario.clone();
        let id = self.scene()?.scenario.id.clone();
        let run = self.cpu.as_mut().ok_or("no CPU run started")?;
        for _ in 0..samples {
            if run.k > run.n {
                break;
            }
            let mut x = run.runner.sample()?;
            x.t = run.k as f32 * DT * SAMPLE_STEPS as f32;
            run.rec.push(&x);
            run.recording.push(&x);
            if run.k < run.n {
                run.runner.frame(SAMPLE_STEPS)?;
            }
            run.k += 1;
        }
        let (sample, total) = (run.k as u32, run.n as u32 + 1);
        if run.k > run.n {
            let CpuRun { runner, rec, mut recording, .. } = self.cpu.take().unwrap();
            let (mut metrics, series) = rec.finish_with_series();
            let mut runner = runner;
            if let Some(j) = runner.contact_removed() {
                metrics.insert("all.contact_removed_j".into(), j);
            }
            if let Some(on) = runner.continuous_engaged() {
                metrics.insert("all.continuous_engaged".into(), f64::from(u8::from(on)));
            }
            let contact = recording.contact;
            let results = judge(&scenario, &metrics, contact);
            recording.report = Some(CrashReport { id, contact, run: 0, metrics, results, series: Some(series) });
            self.recording = Some(recording);
            return Ok(CpuProgress { sample, samples: total, done: true });
        }
        Ok(CpuProgress { sample, samples: total, done: false })
    }

    /// A recording made elsewhere (the page's GPU run: `crashTakeRecording`, or a cached
    /// file). It must be of the placed scene's bodies (same node and beam counts); `hash`
    /// (when given) is what it was made from.
    pub fn load_recording(&mut self, bytes: &[u8], hash: Option<&str>) -> Result<RecordingStatus, String> {
        let mut r = Recording::from_bytes(bytes)?;
        let sc = self.scene()?;
        let same = r.bodies.len() == sc.built.fleet.len() && r.bodies.iter().zip(&sc.built.fleet).all(|(b, f)| b.nodes as usize == f.nodes.len() && b.beams as usize == f.beams.len());
        if !same {
            return Err("the recording's bodies are not the placed scene's".into());
        }
        if let Some(h) = hash {
            r.hash = h.to_string();
        }
        self.recording = Some(r);
        self.recording_status()
    }

    pub fn recording_bytes(&self) -> Result<Vec<u8>, String> {
        Ok(self.recording()?.to_bytes())
    }

    fn recording(&self) -> Result<&Recording, String> {
        self.recording.as_ref().ok_or_else(|| "no recording: simulate first".into())
    }

    pub fn recording_status(&self) -> Result<RecordingStatus, String> {
        let hash = self.scene.as_ref().map(|s| s.info.hash.clone()).unwrap_or_default();
        Ok(match &self.recording {
            None => RecordingStatus { present: false, fresh: false, executor: String::new(), hash: String::new(), samples: 0, duration_s: 0.0, impact_s: 0.0, sample_s: DT * SAMPLE_STEPS as f32, start_ms: 0.0, end_ms: 0.0, has_report: false },
            Some(r) => RecordingStatus {
                present: true,
                fresh: !r.hash.is_empty() && r.hash == hash,
                executor: r.executor.clone(),
                hash: r.hash.clone(),
                samples: r.samples() as u32,
                duration_s: r.duration_s(),
                impact_s: r.impact_s,
                sample_s: r.sample_s,
                start_ms: r.ms_of(0),
                end_ms: r.ms_of(r.samples().saturating_sub(1)),
                has_report: r.report.is_some(),
            },
        })
    }

    pub fn report(&self) -> Result<Option<CrashReport>, String> {
        Ok(self.recording()?.report.clone().map(|mut r| {
            r.series = None;
            r
        }))
    }

    /// Every body's node positions at `ms` after first contact (x, y, z, bodies in order).
    pub fn positions_at_ms(&self, ms: f32) -> Result<Vec<f32>, String> {
        Ok(self.recording()?.positions_at_ms(ms).iter().flatten().flat_map(|p| p.to_array()).collect())
    }

    /// Per beam (bodies in order) whether it is broken at `ms` after first contact.
    pub fn broken_at_ms(&self, ms: f32) -> Result<Vec<u8>, String> {
        let r = self.recording()?;
        let k = r.index_at_ms(ms).round() as u32;
        Ok(r.broken_at.iter().map(|&b| u8::from(b <= k)).collect())
    }

    /// Body `spec.body`'s outlines at `spec.at_ms` (from the recording), through its camera
    /// (named `image`) or in its car frame: what `outlines.json` holds for a body.
    fn body_outlines(&self, spec: &RubbleSpec) -> Result<(Value, String), String> {
        let sc = self.scene()?;
        let r = self.recording()?;
        let b = spec.body as usize;
        let parts = sc.parts.get(b).ok_or_else(|| format!("no body {b}"))?;
        let now = r.positions_at_ms(spec.at_ms).into_iter().nth(b).ok_or("the recording has no such body")?;
        let views: Vec<(String, View)> = spec.camera.iter().map(|c| ("image".to_string(), View::look(&c.free("image"), spec.width, spec.height))).collect();
        let view = if spec.camera.is_some() { "camera:image" } else { "car_frame" };
        Ok((parts.outlines(&now, &views), view.into()))
    }

    fn shape(&self, schema: &Schema, s: &ShapeSpec) -> Result<Shape, String> {
        if let Some(t) = &s.tracing {
            return Ok(shape::from_tracing(&t.instance, schema, t.width, t.height));
        }
        let r = s.rubble.as_ref().ok_or("a shape is { tracing } or { rubble }")?;
        let (body, view) = self.body_outlines(r)?;
        Ok(rubble::derive(schema, &rubble::BodyView::new(&body, &view)?))
    }

    /// Rubble's outline in `schema` at `spec.at_ms` through `spec.camera`, as a tracing
    /// instance over a `width` × `height` image (normalised coordinates), with every part's
    /// outline as `loops`: the ghost the studio draws over a reference image.
    pub fn outline(&self, schema: &Schema, spec: &RubbleSpec) -> Result<Value, String> {
        let (body, view) = self.body_outlines(spec)?;
        let v = rubble::BodyView::new(&body, &view)?;
        let s = rubble::derive(schema, &v);
        let mut inst = rubble::to_instance(schema, &s, &v, f64::from(spec.width), f64::from(spec.height));
        inst["width"] = json!(spec.width);
        inst["height"] = json!(spec.height);
        inst["at_ms"] = json!(spec.at_ms);
        Ok(inst)
    }

    /// The comparison: every source normalised by its own before shape, and the measures'
    /// table ([`shape::compare`]); Rubble's shapes from the recording.
    pub fn compare(&self, schema: &Schema, sources: &[CompareSource]) -> Result<Value, String> {
        let mut list = Vec::new();
        for x in sources {
            list.push(Source { name: x.name.clone(), before: self.shape(schema, &x.before).map_err(|e| format!("{} before: {e}", x.name))?, now: self.shape(schema, &x.now).map_err(|e| format!("{} now: {e}", x.name))? });
        }
        Ok(shape::compare(schema, &list))
    }
}

impl Studio {
    /// The generated designs with templates, ranked by proportion against a traced side
    /// view (`instance` over a `width` × `height` region): [`shape::design`].
    pub fn rank_designs(&self, schema: &Schema, instance: &Instance, width: f64, height: f64) -> Vec<shape::design::DesignMatch> {
        let s = shape::from_tracing(instance, schema, width, height);
        let traced = shape::design::traced_ratios(&s);
        let dims: Vec<shape::design::DesignDims> = self.templates.iter().filter_map(|(d, t)| shape::design::DesignDims::of(d, t)).collect();
        shape::design::rank(&traced, &dims)
    }
}

/// The moment and named camera a reference frame (its moment label and view label) is
/// compared with in a scenario ([`shape::link`]).
pub fn pick(moments: &[Moment], cameras: &[Camera], scenario: &str, label: &str, view: &str) -> Value {
    json!({ "moment": shape::link::pick_moment(scenario, label, moments), "camera": shape::link::pick_camera(view, cameras) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_builtin_source_is_filed_by_id_and_contact() {
        let (s, c, h) = SceneSource { desc: None, id: Some("s02".into()), contact: None }.resolve().unwrap();
        assert_eq!((s.id.as_str(), c, h.as_str()), ("S02", Contact::Edge, "builtin-S02-edge"));
        assert!(SceneSource { desc: None, id: None, contact: None }.resolve().is_err());
    }
}
