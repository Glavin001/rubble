//! Waiting for a GPU buffer map from synchronous code.
//!
//! Natively the caller's `device.poll(wait_indefinitely)` has already blocked until the
//! map callback ran, so [`until`] returns at once and [`recv`] is `Receiver::recv`: native
//! behaviour is unchanged. In the browser a poll cannot block: the map completes on the
//! event loop. There [`until`] suspends the wasm stack to the event loop until the
//! callback has run, through JavaScript Promise Integration: the import
//! `rubble-gpu-wait.yield_to_browser` is a `WebAssembly.Suspending` that resolves after
//! one macrotask, and the export that reached the wait is called through
//! `WebAssembly.promising` (`engine-webgpu/fracture/gpu-wait.mjs`).

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "rubble-gpu-wait")]
extern "C" {
    fn yield_to_browser();
}

/// Returns once `done()` is true (a map callback has run).
#[inline]
pub fn until(mut done: impl FnMut() -> bool) {
    #[cfg(target_arch = "wasm32")]
    while !done() {
        // SAFETY: an import without arguments or results; it only suspends this stack.
        unsafe { yield_to_browser() }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = &mut done;
}

/// The value a map callback sends on `rx`.
pub fn recv<T>(rx: &std::sync::mpsc::Receiver<T>) -> T {
    #[cfg(target_arch = "wasm32")]
    loop {
        if let Ok(v) = rx.try_recv() {
            return v;
        }
        // SAFETY: as in `until`.
        unsafe { yield_to_browser() }
    }
    #[cfg(not(target_arch = "wasm32"))]
    rx.recv().expect("map callback")
}
