//! Host-side spans of a frame's work (`world.step`, `sim.encode`, `world.poll`, ...), for the
//! profilers: the browser's profiler panel and `tools/webgpu-profile` harness, and native
//! runs (`RUBBLE_HOST_PROFILE=1`).
//!
//! Off by default and then free: [`span`] returns an inert guard after one relaxed load.
//! On, each span records its start and duration on the host clock (the browser's
//! `performance.now()`, installed by the wasm crate with [`set_clock`]) into a buffer the
//! host drains once a frame ([`drain`]), and a sink ([`set_sink`]) may forward it, for
//! example as a `performance.measure` that Chrome's trace shows under User Timing.
//!
//! A span that a GPU wait suspended (the browser's JSPI yield, `engine_gpu_wait::YIELDS`)
//! includes event-loop time that is not this host's work; it carries the count of yields
//! so a report can keep it apart instead of double counting it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ENABLED: AtomicBool = AtomicBool::new(false);
static SPANS: Mutex<Vec<Span>> = Mutex::new(Vec::new());
static CLOCK: Mutex<Option<fn() -> f64>> = Mutex::new(None);
static SINK: Mutex<Option<fn(&'static str, f64, f64)>> = Mutex::new(None);
/// Spans kept between drains (a host that never drains loses the newest, not memory).
const CAP: usize = 8192;

/// One timed piece of host work.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub name: &'static str,
    /// Host clock at its start (ms).
    pub start_ms: f64,
    pub dur_ms: f64,
    /// GPU-wait suspensions inside it (the browser's; 0 natively).
    pub yields: u64,
}

/// Turn span recording on or off (natively also `RUBBLE_HOST_PROFILE=1`, read once).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    if !on {
        SPANS.lock().expect("spans").clear();
    }
}

pub fn enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static ENV: std::sync::Once = std::sync::Once::new();
        ENV.call_once(|| {
            if std::env::var("RUBBLE_HOST_PROFILE").is_ok_and(|v| v == "1") {
                ENABLED.store(true, Ordering::Relaxed);
            }
        });
    }
    ENABLED.load(Ordering::Relaxed)
}

/// The host clock in ms (the browser: `performance.now`).
pub fn set_clock(clock: fn() -> f64) {
    *CLOCK.lock().expect("clock") = Some(clock);
}

/// Called with every finished span (name, start ms, end ms).
pub fn set_sink(sink: Option<fn(&'static str, f64, f64)>) {
    *SINK.lock().expect("sink") = sink;
}

pub fn now_ms() -> f64 {
    if let Some(c) = *CLOCK.lock().expect("clock") {
        return c();
    }
    static ORIGIN: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
    ORIGIN.get_or_init(web_time::Instant::now).elapsed().as_secs_f64() * 1e3
}

/// Times until dropped (inert when recording is off).
#[must_use = "a span times until it is dropped"]
pub struct Guard {
    name: &'static str,
    start: Option<(f64, u64)>,
}

/// A span named `name` from now until the guard drops.
#[inline]
pub fn span(name: &'static str) -> Guard {
    if !enabled() {
        return Guard { name, start: None };
    }
    Guard {
        name,
        start: Some((now_ms(), engine_gpu_wait::YIELDS.load(Ordering::Relaxed))),
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let Some((start, y0)) = self.start else {
            return;
        };
        let end = now_ms();
        let yields = engine_gpu_wait::YIELDS.load(Ordering::Relaxed).saturating_sub(y0);
        {
            let mut s = SPANS.lock().expect("spans");
            if s.len() < CAP {
                s.push(Span {
                    name: self.name,
                    start_ms: start,
                    dur_ms: end - start,
                    yields,
                });
            }
        }
        if let Some(sink) = *SINK.lock().expect("sink") {
            sink(self.name, start, end);
        }
    }
}

/// The spans recorded since the last drain, in the order they ended.
pub fn drain() -> Vec<Span> {
    std::mem::take(&mut *SPANS.lock().expect("spans"))
}

/// `drain()` as JSON: `[{"n": name, "s": start ms, "d": ms, "y": yields}, ...]`.
pub fn drain_json() -> String {
    let spans = drain();
    let mut out = String::with_capacity(spans.len() * 48 + 2);
    out.push('[');
    for (i, s) in spans.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"n\":\"{}\",\"s\":{:.4},\"d\":{:.4},\"y\":{}}}",
            s.name, s.start_ms, s.dur_ms, s.yields
        ));
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_record_only_when_on() {
        set_enabled(false);
        drop(span("off"));
        assert!(drain().is_empty());
        set_enabled(true);
        {
            let _outer = span("outer");
            let _inner = span("inner");
        }
        let s = drain();
        set_enabled(false);
        assert_eq!(s.iter().map(|s| s.name).collect::<Vec<_>>(), ["inner", "outer"]);
        assert!(s.iter().all(|s| s.dur_ms >= 0.0 && s.yields == 0));
        assert!(drain_json().starts_with('['));
    }
}
