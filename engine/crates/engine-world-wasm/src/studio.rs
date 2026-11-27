//! The Reference Studio's wasm (`apps/reference-studio/`): `engine_softbody::crash_reference::studio`
//! bound for the page, with the templates built from the generated designs' mod folders as
//! the crash lab builds them (`crash_templates::template_from_files`). The types crossing
//! here are generated from the Rust (`apps/reference-studio/src/generated/crash-reference.ts`).
//!
//! The run itself is the crash lab's: `WebWorld.explicitSimAsync({ crashRef: { desc | id,
//! designs, record: true } })` on the page's WebGPU, its recording handed here with
//! [`CrashStudio::load_recording`]. [`CrashStudio::cpu_start`] runs the CPU reference instead,
//! for the automated smoke test where headless Chrome has no WebGPU (labelled `cpu-fallback`).

use crate::err;
use engine_softbody::crash_reference::{self as cr, shape::Schema, studio as st};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn js<T: Serialize>(v: &T) -> Result<JsValue, JsValue> {
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
}

fn from<T: serde::de::DeserializeOwned>(v: JsValue, what: &str) -> Result<T, JsValue> {
    serde_wasm_bindgen::from_value(v).map_err(|e| err(format!("{what}: {e}")))
}

fn clock() -> f64 {
    js_sys::Date::now()
}

/// The studio's scene, its templates and its recording.
#[wasm_bindgen]
pub struct CrashStudio(st::Studio);

#[wasm_bindgen]
impl CrashStudio {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(st::Studio::new())
    }

    /// The designs `src` (`SceneSource`) needs that have no template yet.
    pub fn missing(&self, src: JsValue) -> Result<JsValue, JsValue> {
        js(&self.0.missing(&from(src, "scene")?).map_err(err)?)
    }

    /// Whether `design` is built from code alone (no mod folder: the moving deformable barrier).
    #[wasm_bindgen(js_name = fromCode)]
    pub fn from_code(design: &str) -> bool {
        engine_softbody_authoring::crash_templates::from_code(design)
    }

    /// Add a design's template: from code (see [`Self::from_code`]), or from its mod folder's
    /// files (`[[path, text], ...]`: its `.jbeam` parts, its `.pc` and `crash.json`).
    #[wasm_bindgen(js_name = addTemplate)]
    pub fn add_template(&mut self, design: &str, files: JsValue) -> Result<(), JsValue> {
        use engine_softbody_authoring::crash_templates as ct;
        let t = if ct::from_code(design) {
            ct::template(design).map_err(err)?
        } else {
            let list: Vec<(String, String)> = from(files, "files")?;
            let mut folder: Vec<(String, Vec<u8>)> = Vec::new();
            for (p, text) in list {
                folder.retain(|(q, _)| *q != p);
                folder.push((p, text.into_bytes()));
            }
            ct::template_from_files(design, &folder).map_err(err)?
        };
        self.0.templates.insert(design.to_string(), t);
        Ok(())
    }

    #[wasm_bindgen(js_name = hasTemplate)]
    pub fn has_template(&self, design: &str) -> bool {
        self.0.templates.contains_key(design)
    }

    /// Place a scene (`SceneSource`): its fleet built as every executor builds it; returns
    /// `SceneInfo` (each body's collision surface, the boxes, the first contact, the schedule).
    pub fn place(&mut self, src: JsValue) -> Result<JsValue, JsValue> {
        js(&self.0.place(&from(src, "scene")?, &clock).map_err(err)?)
    }

    /// Every body's node positions at spawn (x, y, z; bodies in order).
    #[wasm_bindgen(js_name = spawnPositions)]
    pub fn spawn_positions(&self) -> Result<Vec<f32>, JsValue> {
        self.0.spawn_positions().map_err(err)
    }

    /// A built-in scenario as data (`ScenarioDesc`, run 0) on the templates added: a
    /// recreation can start from it.
    #[wasm_bindgen(js_name = exportBuiltin)]
    pub fn export_builtin(&self, id: &str) -> Result<JsValue, JsValue> {
        let s = cr::get(id).ok_or_else(|| err(format!("no scenario {id}")))?;
        js(&s.to_desc(&self.0.templates).map_err(err)?)
    }

    /// Start the CPU reference's run (the smoke test's fallback, `cpu-fallback`).
    #[wasm_bindgen(js_name = cpuStart)]
    pub fn cpu_start(&mut self) -> Result<(), JsValue> {
        self.0.cpu_start().map_err(err)
    }

    /// Up to `samples` more samples of the CPU run (`CpuProgress`).
    #[wasm_bindgen(js_name = cpuStep)]
    pub fn cpu_step(&mut self, samples: u32) -> Result<JsValue, JsValue> {
        js(&self.0.cpu_step(samples).map_err(err)?)
    }

    /// A recording (the GPU run's `crashTakeRecording`, or a cached file) of the placed
    /// scene, made from `hash` when given (`RecordingStatus`).
    #[wasm_bindgen(js_name = loadRecording)]
    pub fn load_recording(&mut self, bytes: &[u8], hash: Option<String>) -> Result<JsValue, JsValue> {
        js(&self.0.load_recording(bytes, hash.as_deref()).map_err(err)?)
    }

    #[wasm_bindgen(js_name = recordingBytes)]
    pub fn recording_bytes(&self) -> Result<Vec<u8>, JsValue> {
        self.0.recording_bytes().map_err(err)
    }

    #[wasm_bindgen(js_name = recordingStatus)]
    pub fn recording_status(&self) -> Result<JsValue, JsValue> {
        js(&self.0.recording_status().map_err(err)?)
    }

    /// The recording's report (`CrashReport` without its time histories), or null.
    pub fn report(&self) -> Result<JsValue, JsValue> {
        js(&self.0.report().map_err(err)?)
    }

    /// Every body's node positions at `ms` after first contact (from the recording).
    #[wasm_bindgen(js_name = positionsAtMs)]
    pub fn positions_at_ms(&self, ms: f32) -> Result<Vec<f32>, JsValue> {
        self.0.positions_at_ms(ms).map_err(err)
    }

    /// Per beam whether it is broken at `ms` after first contact.
    #[wasm_bindgen(js_name = brokenAtMs)]
    pub fn broken_at_ms(&self, ms: f32) -> Result<Vec<u8>, JsValue> {
        self.0.broken_at_ms(ms).map_err(err)
    }

    /// Rubble's outline in `schema` (a structure schema) at `spec` (`RubbleSpec`: a time, a
    /// frame camera, an image size, a body), from the recording.
    pub fn outline(&self, schema: JsValue, spec: JsValue) -> Result<JsValue, JsValue> {
        let schema: Schema = from(schema, "schema")?;
        js(&self.0.outline(&schema, &from(spec, "spec")?).map_err(err)?)
    }

    /// The comparison of `sources` (`CompareSource[]`) in `schema`.
    pub fn compare(&self, schema: JsValue, sources: JsValue) -> Result<JsValue, JsValue> {
        let schema: Schema = from(schema, "schema")?;
        let sources: Vec<st::CompareSource> = from(sources, "sources")?;
        js(&self.0.compare(&schema, &sources).map_err(err)?)
    }
}

impl Default for CrashStudio {
    fn default() -> Self {
        Self::new()
    }
}

/// The registry (`Registry`): the built-in scenarios, then `data` (`ScenarioDesc[]`).
#[wasm_bindgen(js_name = studioRegistry)]
pub fn studio_registry(data: JsValue) -> Result<JsValue, JsValue> {
    let list: Vec<cr::ScenarioDesc> = if data.is_undefined() || data.is_null() { Vec::new() } else { from(data, "data scenarios")? };
    js(&cr::registry::registry(&list, &engine_softbody_authoring::crash_templates::from_code))
}

/// Check a scenario description (an error says what is wrong).
#[wasm_bindgen(js_name = studioCheckDesc)]
pub fn studio_check_desc(desc: JsValue) -> Result<(), JsValue> {
    let d: cr::ScenarioDesc = from(desc, "desc")?;
    d.check().map_err(err)
}

/// Whether `id` may name a data scenario (not a built-in one's).
#[wasm_bindgen(js_name = studioValidId)]
pub fn studio_valid_id(id: &str) -> Result<(), JsValue> {
    cr::desc::valid_id(id).map_err(err)
}

/// The moment and named camera a frame (moment label, view label) is compared with in a
/// scenario of these moments and cameras: `{ moment, camera }`.
#[wasm_bindgen(js_name = studioPick)]
pub fn studio_pick(moments: JsValue, cameras: JsValue, scenario: &str, label: &str, view: &str) -> Result<JsValue, JsValue> {
    let m: Vec<cr::Moment> = from(moments, "moments")?;
    let c: Vec<cr::Camera> = from(cameras, "cameras")?;
    js(&st::pick(&m, &c, scenario, label, view))
}
