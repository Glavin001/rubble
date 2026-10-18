//! Counts of GPU queue submissions, host reads and host waits, by call site, within a tick.
//!
//! Every `queue.submit`, every `map_async` and every blocking `device.poll` in rubble's host
//! and the stress solver calls [`submit`], [`read`] or [`wait`] first. Nothing is recorded
//! outside [`begin`] and [`end`], so code that never opens a tick pays one relaxed atomic
//! load per call. Plan: `engine/docs/stress-5d-one-command-buffer.md`, W1.

use std::panic::Location;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Io {
    Submit,
    Read,
    Wait,
}

/// One recorded call: what it was and where it was made.
pub type Event = (Io, &'static Location<'static>);

static ON: AtomicBool = AtomicBool::new(false);
static LOG: Mutex<Vec<Event>> = Mutex::new(Vec::new());

fn record(io: Io, at: &'static Location<'static>) {
    if ON.load(Ordering::Relaxed) {
        LOG.lock().unwrap().push((io, at));
    }
}

/// A queue submission at the caller.
#[track_caller]
pub fn submit() {
    record(Io::Submit, Location::caller());
}

/// A host read of device memory (a map and its wait) at the caller.
#[track_caller]
pub fn read() {
    record(Io::Read, Location::caller());
}

/// A blocking wait on the device that is not part of a read, at the caller.
#[track_caller]
pub fn wait() {
    record(Io::Wait, Location::caller());
}

/// Starts recording a tick, dropping anything recorded before.
pub fn begin() {
    LOG.lock().unwrap().clear();
    ON.store(true, Ordering::Relaxed);
}

/// Stops recording and returns the tick's calls in order.
pub fn end() -> Tick {
    ON.store(false, Ordering::Relaxed);
    Tick(std::mem::take(&mut *LOG.lock().unwrap()))
}

/// The calls one tick made.
#[derive(Clone, Debug, Default)]
pub struct Tick(pub Vec<Event>);

impl Tick {
    pub fn count(&self, io: Io) -> usize {
        self.0.iter().filter(|e| e.0 == io).count()
    }
    pub fn submits(&self) -> usize {
        self.count(Io::Submit)
    }
    pub fn reads(&self) -> usize {
        self.count(Io::Read)
    }
    pub fn waits(&self) -> usize {
        self.count(Io::Wait)
    }
    /// The call sites of one kind, `file:line`, in order.
    pub fn sites(&self, io: Io) -> Vec<String> {
        self.0.iter().filter(|e| e.0 == io).map(|e| format!("{}:{}", e.1.file(), e.1.line())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_only_inside_a_tick() {
        submit();
        begin();
        submit();
        let at = line!() + 1;
        read();
        read();
        wait();
        let t = end();
        submit();
        assert_eq!((t.submits(), t.reads(), t.waits()), (1, 2, 1));
        assert!(t.sites(Io::Read)[0].ends_with(&format!(":{at}")));
        assert_eq!(end().0.len(), 0);
    }
}
