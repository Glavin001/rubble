//! The explicit kernels' loop caps on CUDA: the caps live in the Slang source
//! (`slang/kernels/explicit_loop_cap.slang`), so the CUDA modules carry them
//! (build.rs checks every explicit module calls `ebLoopCapHit`). Every build checks
//! the test kernel (`explicit_loop_cap_test.slang`) was emitted with its bindings;
//! with `ENGINE_CUDA_DEVICE=1` (an NVIDIA host with nvcc at build time) it runs a
//! loop whose exit never comes, a counted loop to a huge bound and a stepped loop
//! with a zero step: each stops (no hang) and the flag in gAcc's last word names
//! the stopped loop's source line. As engine-webgpu's and engine-metal's
//! tests/explicit_loop_cap.rs do on wgpu and Metal.

use cudarc::driver::result;
use engine_cuda::{CudaExplicitBeams, CUDA_LOOP_CAP_TEST};
use engine_exec::explicit_plan::{explicit_loop_cap_hit, EXPLICIT_LOOP_CAP, EXPLICIT_LOOP_CAP_TAG};
use std::ffi::{c_void, CString};

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host (built with nvcc)");
        return false;
    }
    assert!(
        !CUDA_LOOP_CAP_TEST.image.is_empty(),
        "ENGINE_CUDA_DEVICE=1 but this build embedded no loop-cap test image (no nvcc)"
    );
    assert!(CudaExplicitBeams::available(), "ENGINE_CUDA_DEVICE=1 but no NVIDIA driver/device loaded");
    true
}

/// The source line of the test kernel's loop marked `loop-cap-test: <tag>`.
fn line_of(tag: &str) -> i32 {
    let src = include_str!("../../../slang/kernels/explicit_loop_cap_test.slang");
    let at = src
        .lines()
        .position(|l| l.contains(&format!("loop-cap-test: {tag}")))
        .unwrap_or_else(|| panic!("no loop marked {tag}"));
    at as i32 + 1
}

#[test]
fn cuda_loop_cap_test_emitted() {
    let k = &CUDA_LOOP_CAP_TEST;
    assert!(k.cu_bytes > 0, "empty CUDA source");
    assert_eq!(k.threads[0], 1);
    let mut slots: Vec<u32> = k.binds.iter().map(|b| b.2).collect();
    slots.sort();
    assert_eq!(slots, [0, 1], "gAcc and gOut mapped");
}

/// One launch with `out` (8 words) from the host: (out, gAcc) afterwards.
fn run(out: [u32; 8]) -> ([u32; 8], [i32; 4]) {
    let k = &CUDA_LOOP_CAP_TEST;
    let map = |e: cudarc::driver::DriverError| format!("CUDA: {e:?}");
    let go = || -> Result<([u32; 8], [i32; 4]), String> {
        result::init().map_err(map)?;
        let dev = result::device::get(0).map_err(map)?;
        let ctx = unsafe { result::primary_ctx::retain(dev) }.map_err(map)?;
        unsafe { result::ctx::set_current(ctx) }.map_err(map)?;
        let module = unsafe { result::module::load_data(k.image.as_ptr() as *const c_void) }.map_err(map)?;
        let entry = CString::new(k.entry).unwrap();
        let function = unsafe { result::module::get_function(module, entry) }.map_err(map)?;
        let name = CString::new("SLANG_globalParams").unwrap();
        let (globals, bytes) = unsafe { result::module::get_global(module, name) }.map_err(map)?;
        let acc = unsafe { result::malloc_sync(16) }.map_err(map)?;
        let o = unsafe { result::malloc_sync(32) }.map_err(map)?;
        unsafe { result::memcpy_htod_sync(acc, &[0i32; 4]) }.map_err(map)?;
        unsafe { result::memcpy_htod_sync(o, &out) }.map_err(map)?;
        // The bank: (pointer, element count) or a bare pointer per global.
        let mut bank = vec![0u8; bytes];
        for &(offset, size, slot) in k.binds {
            let (ptr, n) = if slot == 0 { (acc, 4u64) } else { (o, 8u64) };
            let at = offset as usize;
            bank[at..at + 8].copy_from_slice(&ptr.to_le_bytes());
            if size == 16 {
                bank[at + 8..at + 16].copy_from_slice(&n.to_le_bytes());
            }
        }
        unsafe { result::memcpy_htod_sync(globals, &bank) }.map_err(map)?;
        let stream = std::ptr::null_mut();
        unsafe { result::launch_kernel(function, (1, 1, 1), (1, 1, 1), 0, stream, &mut []) }.map_err(map)?;
        unsafe { result::stream::synchronize(stream) }.map_err(map)?;
        let mut o_back = [0u32; 8];
        let mut a_back = [0i32; 4];
        unsafe { result::memcpy_dtoh_sync(&mut o_back, o) }.map_err(map)?;
        unsafe { result::memcpy_dtoh_sync(&mut a_back, acc) }.map_err(map)?;
        unsafe {
            let _ = result::free_sync(acc);
            let _ = result::free_sync(o);
            let _ = result::module::unload(module);
        }
        Ok((o_back, a_back))
    };
    go().unwrap()
}

#[test]
fn a_runaway_loop_stops_at_the_cap_and_raises_the_flag_on_cuda() {
    if !gpu() {
        return;
    }
    let cap = EXPLICIT_LOOP_CAP;
    let tag = EXPLICIT_LOOP_CAP_TAG;
    let (pass, end, step) = (line_of("pass"), line_of("end"), line_of("step"));
    let (o, a) = run([0, 1, 0, 5, 12, 4, 0, 0]);
    assert_eq!((o[0], o[2], o[3], o[6]), (0, 3, 5, 3));
    assert_eq!(a, [0; 4]);
    let (o, a) = run([0, 0, 0, 5, 12, 4, 0, 0]);
    assert_eq!(o[0], cap, "passes of the runaway loop");
    assert_eq!(a, [0, 0, 0, tag | pass]);
    let (o, a) = run([0, 1, 0, u32::MAX, 12, 4, 0, 0]);
    assert_eq!((o[0], o[3]), (0, cap), "passes of the runaway counted loop");
    assert_eq!(a, [0, 0, 0, tag | end]);
    let (o, a) = run([0, 1, 0, 5, 12, 0, 0, 0]);
    assert_eq!((o[3], o[6]), (5, 0));
    assert_eq!(a, [0, 0, 0, tag | step]);
    let (_, a) = run([0, 0, 0, u32::MAX, 12, 0, 0, 0]);
    assert_eq!(a[3], tag | pass.max(end).max(step));
    let e = explicit_loop_cap_hit(a[3]).unwrap_err();
    assert!(e.contains(&format!("line {}", pass.max(end).max(step))), "{e}");
}
