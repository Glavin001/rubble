//! A closed-loop client for the learned impact-outcome model (the files protocol of
//! `learned-impact/PROVIDER_PROTOCOL.md`): the host writes each routed island's call record
//! (`outcome_calls::live_call`, the single feature builder) and reads the model's decoded
//! state (`learned-impact-decoded-state/1`) back from a Python provider process
//! (`python -m learned_impact.outcome.provider`) that this client spawns and owns.
//!
//! Used by `stress-viz` (`engine learned`) to render the model end to end; the protocol is
//! the one P0's harness (`learned_p0 ceiling --variant provider --provider files`) speaks.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;
use stress_ref::world::{DecodedState, IslandKeys, World};

use crate::outcome_calls;

/// How to start the model.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// The trained checkpoint (`runs/<job>/best.pt`).
    pub checkpoint: PathBuf,
    /// The Python package directory (`learned-impact/python`), whose `.venv` is used.
    pub python_dir: PathBuf,
    /// `cpu` or `mps`.
    pub device: String,
    /// Seconds to wait for one response (the first includes loading the model).
    pub timeout: f64,
    /// Variant-B QS-frame features in the call record (`live_call`'s `qs`): `None` asks the
    /// provider (`model.json`'s `needs_qs`, true only for variant B). Building them re-runs
    /// the frame's contact substeps under `blast_parity`, most of a call's host cost.
    pub qs: Option<bool>,
}

impl ModelSpec {
    /// The defaults for this checkout: `engine/stress-oracle/learned-impact/python`.
    pub fn new(checkpoint: impl Into<PathBuf>) -> ModelSpec {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        ModelSpec {
            checkpoint: checkpoint.into(),
            python_dir: here.join("../../learned-impact/python"),
            device: "cpu".into(),
            timeout: 120.0,
            qs: None,
        }
    }
}

/// A running provider process and its request directory.
pub struct ModelProvider {
    spec: ModelSpec,
    dir: PathBuf,
    scene: String,
    child: Child,
    /// Whether call records carry the QS-frame features (resolved at spawn).
    qs: bool,
    /// Wall-clock milliseconds of every call (request written → response read).
    pub call_ms: Vec<f64>,
    /// `<dir>.lock`, holding this process's id while the provider runs.
    lock: PathBuf,
    /// The provider reads `learned-impact-outcome/3` records.
    v3: bool,
}

impl ModelProvider {
    /// Start the provider in a fresh directory `dir` (removed first).
    pub fn spawn(spec: &ModelSpec, dir: &Path, scene: &str) -> Result<ModelProvider, String> {
        // One provider per directory: two harnesses writing the same `--dir` (e.g. two
        // evaluations with the same `--out`) would answer each other's requests and stop each
        // other's provider through `done`. A live owner's lock is refused; a stale one is taken.
        let lock = lock_path(dir);
        if let Ok(text) = std::fs::read_to_string(&lock) {
            if let Ok(pid) = text.trim().parse::<u32>() {
                let alive = pid != std::process::id()
                    && Command::new("kill").args(["-0", &pid.to_string()]).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
                if alive {
                    return Err(format!("{} is in use by process {pid} (another run with the same output directory?)", dir.display()));
                }
            }
        }
        if let Some(parent) = lock.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&lock, std::process::id().to_string()).map_err(|e| format!("{}: {e}", lock.display()))?;
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir.join("resp")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let py = spec.python_dir.join(".venv/bin/python");
        if !py.exists() {
            return Err(format!("{} not found: create the venv (learned-impact/python/README.md)", py.display()));
        }
        // The provider runs in the Python package directory: pass absolute paths.
        let checkpoint = std::fs::canonicalize(&spec.checkpoint).map_err(|e| format!("checkpoint {}: {e}", spec.checkpoint.display()))?;
        let dir = &std::fs::canonicalize(dir).map_err(|e| e.to_string())?;
        let log = std::fs::File::create(dir.join("provider.log")).map_err(|e| e.to_string())?;
        let child = Command::new(&py)
            .current_dir(&spec.python_dir)
            .args(["-m", "learned_impact.outcome.provider", "--dir"])
            .arg(dir)
            .arg("--checkpoint")
            .arg(&checkpoint)
            .args(["--device", &spec.device])
            .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|e| format!("starting {}: {e}", py.display()))?;
        // The calls schema the provider validates against (its `schema.py`): a `/3` package
        // (models trained on `/3`) gets each record downgraded (`downgrade_to_v3`, exact).
        let schema_py = std::fs::read_to_string(spec.python_dir.join("learned_impact/outcome/schema.py")).unwrap_or_default();
        let v3 = schema_py.lines().any(|l| l.trim_start().starts_with("SCHEMA_VERSION") && l.contains("\"learned-impact-outcome/3\""));
        let mut p = ModelProvider { spec: spec.clone(), dir: dir.to_path_buf(), scene: scene.to_string(), child, qs: true, call_ms: Vec::new(), lock, v3 };
        p.qs = match spec.qs {
            Some(q) => q,
            None => p.read_needs_qs()?,
        };
        Ok(p)
    }

    /// Wait for the provider's `model.json` (written once the model is loaded).
    fn read_needs_qs(&mut self) -> Result<bool, String> {
        let path = self.dir.join("model.json");
        let t = Instant::now();
        loop {
            if let Ok(text) = std::fs::read_to_string(&path) {
                let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
                return Ok(v.get("needs_qs").and_then(|x| x.as_bool()).unwrap_or(true));
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                let log = std::fs::read_to_string(self.dir.join("provider.log")).unwrap_or_default();
                return Err(format!("model provider exited ({status}) while loading:\n{log}"));
            }
            if t.elapsed().as_secs_f64() > self.spec.timeout {
                return Err(format!("no {} within {} s", path.display(), self.spec.timeout));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The model's outcome of the routed island `keys` from the frame-start state `view`.
    pub fn outcome(&mut self, view: &World, keys: &IslandKeys, frame: i64) -> Result<DecodedState, String> {
        let t = Instant::now();
        let name = format!("{frame:06}_0");
        let req = self.dir.join("req").join(&name);
        std::fs::create_dir_all(&req).map_err(|e| e.to_string())?;
        let mut standalone = view.standalone_copy();
        let mut ev = outcome_calls::live_call(&mut standalone, &self.scene, &json!({}), &keys.chunks, &keys.impactors, frame as usize, self.qs)?;
        if self.v3 {
            outcome_calls::downgrade_to_v3(&mut ev)?;
        }
        outcome_calls::write_event(&req.join("call"), &ev, true)?;
        let request = json!({"schema_version": "learned-impact-provider-request/0", "event_id": self.scene, "frame": frame, "island": 0,
            "time": view.solver.time, "frame_dt": view.scene.sim.frame_dt, "keys": keys, "state": {}, "call_record": "call"});
        std::fs::write(req.join("request.json"), serde_json::to_string(&request).unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(req.join("ready"), b"").map_err(|e| e.to_string())?;
        let resp = self.dir.join("resp").join(format!("{name}.json"));
        loop {
            if let Ok(text) = std::fs::read_to_string(&resp) {
                self.call_ms.push(t.elapsed().as_secs_f64() * 1e3);
                return serde_json::from_str(&text).map_err(|e| format!("{}: {e}", resp.display()));
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                let log = std::fs::read_to_string(self.dir.join("provider.log")).unwrap_or_default();
                let tail: String = log.lines().rev().take(15).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
                return Err(format!("model provider exited ({status}) before answering frame {frame}:\n{tail}"));
            }
            if t.elapsed().as_secs_f64() > self.spec.timeout {
                return Err(format!("no response {} within {} s", resp.display(), self.spec.timeout));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

fn lock_path(dir: &Path) -> PathBuf {
    let mut name = dir.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".lock");
    dir.with_file_name(name)
}

impl Drop for ModelProvider {
    fn drop(&mut self) {
        let _ = std::fs::write(self.dir.join("done"), b"");
        let _ = std::fs::remove_file(&self.lock);
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(5) {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
