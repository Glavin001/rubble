//! Progress logging for long jobs (plan: "Progress logging for every long-running job"):
//! every line flushed to `runs/<job>/progress.log` (and stderr), a `status.json` rewritten at
//! least every 30 s by a heartbeat thread, and a heartbeat line for each event still running
//! after 30 s. A `status.json` older than 5 minutes means the job is hung.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub const HEARTBEAT: Duration = Duration::from_secs(30);
/// A `status.json` not rewritten for this long marks a hung job.
pub const STALE_AFTER_S: f64 = 300.0;

/// What `status.json` holds.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub state: String,
    pub total: usize,
    pub done: usize,
    pub skipped: usize,
    pub failures: usize,
    pub last_error: Option<String>,
    /// Events running now: id -> (frame, simulated time, phase, breaks, last beat).
    pub running: BTreeMap<String, (usize, f64, String, usize)>,
    pub extra: Value,
}

struct Inner {
    log: File,
    status_path: PathBuf,
    status: Status,
    started: Instant,
    /// Last heartbeat line per running event.
    beaten: BTreeMap<String, Instant>,
}

/// A job's log and status, shared by the worker threads.
#[derive(Clone)]
pub struct Progress {
    inner: Arc<Mutex<Inner>>,
    stop: Arc<AtomicBool>,
}

fn now_s() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

impl Progress {
    /// Open (append to) `dir/progress.log` and start the status heartbeat.
    pub fn open(dir: &Path, total: usize) -> Result<Progress, String> {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let log = OpenOptions::new().create(true).append(true).open(dir.join("progress.log")).map_err(|e| e.to_string())?;
        let inner = Inner { log, status_path: dir.join("status.json"), status: Status { state: "running".into(), total, ..Default::default() }, started: Instant::now(), beaten: BTreeMap::new() };
        let p = Progress { inner: Arc::new(Mutex::new(inner)), stop: Arc::new(AtomicBool::new(false)) };
        p.write_status();
        let q = p.clone();
        std::thread::spawn(move || {
            let step = Duration::from_secs(1);
            let mut since = Duration::ZERO;
            while !q.stop.load(Ordering::Relaxed) {
                std::thread::sleep(step);
                since += step;
                if since >= HEARTBEAT {
                    since = Duration::ZERO;
                    q.write_status();
                    q.heartbeat_lines();
                }
            }
        });
        Ok(p)
    }

    /// A timestamped line to the log and stderr, flushed.
    pub fn line(&self, text: &str) {
        let mut g = self.inner.lock().unwrap();
        let stamp = chrono_like(now_s());
        let l = format!("[{stamp}] {text}\n");
        let _ = g.log.write_all(l.as_bytes());
        let _ = g.log.flush();
        eprint!("{l}");
    }

    pub fn update(&self, f: impl FnOnce(&mut Status)) {
        let mut g = self.inner.lock().unwrap();
        f(&mut g.status);
    }

    /// Record an event's progress (the heartbeat thread prints it every 30 s).
    pub fn beat(&self, id: &str, frame: usize, time: f64, phase: &str, breaks: usize) {
        let mut g = self.inner.lock().unwrap();
        g.status.running.insert(id.into(), (frame, time, phase.into(), breaks));
        g.beaten.entry(id.into()).or_insert_with(Instant::now);
    }

    pub fn finish_event(&self, id: &str) {
        let mut g = self.inner.lock().unwrap();
        g.status.running.remove(id);
        g.beaten.remove(id);
    }

    pub fn elapsed(&self) -> f64 {
        self.inner.lock().unwrap().started.elapsed().as_secs_f64()
    }

    fn heartbeat_lines(&self) {
        let lines: Vec<String> = {
            let mut g = self.inner.lock().unwrap();
            let now = Instant::now();
            let due: Vec<String> = g.beaten.iter().filter(|(_, t)| now.duration_since(**t) >= HEARTBEAT).map(|(k, _)| k.clone()).collect();
            let mut out = Vec::new();
            for id in due {
                if let Some((frame, time, phase, breaks)) = g.status.running.get(&id).cloned() {
                    out.push(format!("[beat] {id}: {time:.3} s simulated, {frame} frames, phase={phase}, {breaks} bonds broken so far"));
                }
                g.beaten.insert(id, now);
            }
            if out.is_empty() {
                let s = &g.status;
                out.push(format!("[beat] {}/{} events, {} running, {} failures", s.done + s.skipped, s.total, s.running.len(), s.failures));
            }
            out
        };
        for l in lines {
            self.line(&l);
        }
    }

    pub fn write_status(&self) {
        let g = self.inner.lock().unwrap();
        let s = &g.status;
        let elapsed = g.started.elapsed().as_secs_f64();
        let finished = s.done;
        let rate = if elapsed > 0.0 { finished as f64 / elapsed } else { 0.0 };
        let left = s.total.saturating_sub(s.done + s.skipped + s.failures);
        let v = json!({
            "state": s.state, "total": s.total, "done": s.done, "skipped": s.skipped, "failures": s.failures,
            "elapsed_s": elapsed, "events_per_min": rate * 60.0,
            "eta_s": if rate > 0.0 { Some(left as f64 / rate) } else { None },
            "updated": now_s(), "stale_after_s": STALE_AFTER_S, "last_error": s.last_error,
            "running": s.running.iter().map(|(k, (f, t, p, b))| (k.clone(), json!({"frame": f, "time": t, "phase": p, "breaks": b}))).collect::<serde_json::Map<_, _>>(),
            "extra": s.extra,
        });
        let tmp = g.status_path.with_extension("json.tmp");
        if fs::write(&tmp, serde_json::to_vec_pretty(&v).unwrap()).is_ok() {
            let _ = fs::rename(&tmp, &g.status_path);
        }
    }

    pub fn close(&self, state: &str) {
        self.update(|s| s.state = state.into());
        self.write_status();
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// `HH:MM:SS` (UTC) without a date crate.
fn chrono_like(t: f64) -> String {
    let s = t as u64 % 86400;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// `h m` / `m s` for an ETA.
pub fn duration_str(s: f64) -> String {
    if !s.is_finite() {
        return "?".into();
    }
    let s = s.max(0.0) as u64;
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s / 60) % 60)
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}
