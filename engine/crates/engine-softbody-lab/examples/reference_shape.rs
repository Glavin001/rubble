//! The Reference Studio's shape maths as a JSON service (`apps/reference-studio/`,
//! `engine/tools/crash-reference/outlines.py`): one place that normalises tracings and
//! Rubble's outlines and measures them (`engine_softbody::crash_reference::shape`).
//!
//! `reference_shape [--serve]`: reads one JSON request per line on stdin and writes one JSON
//! reply per line on stdout (`{"id", "ok", "result" | "error"}`); without `--serve`, reads a
//! single request (the whole of stdin) and exits. Requests (`op`):
//!
//! - `pick`: `{scenario, moment, view}`: the scenario moment and camera a reference frame
//!   with that moment label and view label is compared with (`shape::link`).
//! - `scenarios`: the crash reference registry (id, title, designs, cameras, moments,
//!   reference video and times).
//! - `compare`: `{schema, sources: [{name, before, now}]}`, each shape either
//!   `{"tracing": {instance, width, height}}` (an instance in its region's normalised
//!   coordinates and the region's size in pixels) or `{"rubble": {outlines, moment, view,
//!   body?}}` (an `outlines.json` from `crash_outlines`, a moment's name, `car_frame` or
//!   `camera:<name>`). Replies with the frame used, the normalised shapes and the table.
//! - `rubble_instance`: `{schema, outlines, moment, camera, body?}`: the schema's elements
//!   on Rubble's body as that camera sees it, as a tracing instance over its still
//!   (normalised coordinates), with every part's outline as `loops`.

use engine_softbody::crash_reference::shape::{self, rubble, Instance, Schema, Shape, Source};
use engine_softbody::crash_reference::{self as cr};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::time::SystemTime;

/// Parsed outlines files, by path, kept while their modification time holds.
#[derive(Default)]
struct Cache {
    files: HashMap<String, (SystemTime, Value)>,
}

impl Cache {
    fn outlines(&mut self, path: &str) -> Result<&Value, String> {
        let t = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .map_err(|e| format!("{path}: {e}"))?;
        let fresh = self.files.get(path).is_some_and(|(m, _)| *m == t);
        if !fresh {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let v: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
            self.files.insert(path.to_string(), (t, v));
        }
        Ok(&self.files[path].1)
    }

    fn body(&mut self, path: &str, moment: &str, body: usize) -> Result<Value, String> {
        let doc = self.outlines(path)?;
        let m = doc["moments"]
            .as_array()
            .and_then(|ms| ms.iter().find(|m| m["name"] == moment))
            .ok_or(format!("{path}: no moment {moment}"))?;
        m["bodies"]
            .get(body)
            .cloned()
            .ok_or(format!("{path}: no body {body}"))
    }
}

fn s(v: &Value, k: &str) -> Result<String, String> {
    v[k].as_str()
        .map(str::to_string)
        .ok_or(format!("missing {k}"))
}

fn shape_of(schema: &Schema, spec: &Value, cache: &mut Cache) -> Result<Shape, String> {
    if let Some(t) = spec.get("tracing") {
        let inst: Instance =
            serde_json::from_value(t["instance"].clone()).map_err(|e| format!("instance: {e}"))?;
        let (w, h) = (
            t["width"].as_f64().ok_or("missing width")?,
            t["height"].as_f64().ok_or("missing height")?,
        );
        return Ok(shape::from_tracing(&inst, schema, w, h));
    }
    if let Some(r) = spec.get("rubble") {
        let body = cache.body(
            &s(r, "outlines")?,
            &s(r, "moment")?,
            r["body"].as_u64().unwrap_or(0) as usize,
        )?;
        let view = rubble::BodyView::new(&body, &s(r, "view")?)?;
        return Ok(rubble::derive(schema, &view));
    }
    Err("a shape is {tracing} or {rubble}".into())
}

/// The built-in scenarios, then the Reference Studio's data scenarios (its workspace's
/// `scenarios/`, or `STUDIO_SCENARIOS`), marked `recreated`.
fn scenarios() -> Value {
    let dir = std::env::var("STUDIO_SCENARIOS").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../../apps/reference-studio/workspace/scenarios").into());
    let data = cr::desc::load_dir(std::path::Path::new(&dir)).unwrap_or_else(|e| {
        eprintln!("reference_shape: {e}");
        Vec::new()
    });
    let list: Vec<cr::CrashScenario> = cr::all().into_iter().chain(data.iter().map(|d| d.scenario())).collect();
    json!(list
        .iter()
        .map(|x| json!({
            "id": x.id, "title": x.title, "designs": x.designs, "cameras": x.cameras, "moments": x.moments,
            "reference": x.reference, "secs_after_impact": x.secs_after_impact, "recreated": x.recreated,
        }))
        .collect::<Vec<_>>())
}

fn handle(req: &Value, cache: &mut Cache) -> Result<Value, String> {
    let schema = || -> Result<Schema, String> {
        serde_json::from_value(req["schema"].clone()).map_err(|e| format!("schema: {e}"))
    };
    match req["op"].as_str().unwrap_or_default() {
        "scenarios" => Ok(scenarios()),
        "pick" => {
            let id = s(req, "scenario")?;
            let sc = cr::get(&id).ok_or(format!("no scenario {id}"))?;
            Ok(json!({
                "moment": shape::link::pick_moment(&sc.id, &s(req, "moment")?, &sc.moments),
                "camera": shape::link::pick_camera(&s(req, "view")?, &sc.cameras),
            }))
        }
        "compare" => {
            let sc = schema()?;
            let mut sources = Vec::new();
            for x in req["sources"].as_array().ok_or("missing sources")? {
                sources.push(Source {
                    name: s(x, "name")?,
                    before: shape_of(&sc, &x["before"], cache)
                        .map_err(|e| format!("{} before: {e}", x["name"]))?,
                    now: shape_of(&sc, &x["now"], cache)
                        .map_err(|e| format!("{} now: {e}", x["name"]))?,
                });
            }
            Ok(shape::compare(&sc, &sources))
        }
        "rubble_instance" => {
            let sc = schema()?;
            let path = s(req, "outlines")?;
            let camera = s(req, "camera")?;
            let body = cache.body(
                &path,
                &s(req, "moment")?,
                req["body"].as_u64().unwrap_or(0) as usize,
            )?;
            let doc = cache.outlines(&path)?;
            let size = doc["explicit_cameras"]
                .as_array()
                .and_then(|l| l.iter().find(|c| c["camera"]["name"] == camera.as_str()))
                .map(|c| (c["width"].as_f64(), c["height"].as_f64()))
                .unwrap_or((
                    doc["projection"]["width"].as_f64(),
                    doc["projection"]["height"].as_f64(),
                ));
            let (w, h) = (size.0.ok_or("no width")?, size.1.ok_or("no height")?);
            let view = rubble::BodyView::new(&body, &format!("camera:{camera}"))?;
            let shape = rubble::derive(&sc, &view);
            let mut inst = rubble::to_instance(&sc, &shape, &view, w, h);
            inst["width"] = json!(w);
            inst["height"] = json!(h);
            Ok(inst)
        }
        op => Err(format!(
            "unknown op {op:?} (scenarios, pick, compare, rubble_instance)"
        )),
    }
}

fn reply(req: &Value, cache: &mut Cache) -> String {
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    match handle(req, cache) {
        Ok(r) => json!({ "id": id, "ok": true, "result": r }),
        Err(e) => json!({ "id": id, "ok": false, "error": e }),
    }
    .to_string()
}

fn main() {
    let serve = std::env::args().any(|a| a == "--serve");
    let mut cache = Cache::default();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    if !serve {
        let mut text = String::new();
        stdin.lock().read_to_string(&mut text).expect("stdin");
        let req: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| json!({ "op": format!("unparsable: {e}") }));
        writeln!(out, "{}", reply(&req, &mut cache)).expect("stdout");
        return;
    }
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| json!({ "op": format!("unparsable: {e}") }));
        writeln!(out, "{}", reply(&req, &mut cache)).expect("stdout");
        out.flush().expect("stdout");
    }
}
