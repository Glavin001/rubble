//! The explicit kernels' loop caps on native Metal: the caps live in the Slang
//! source (`slang/kernels/explicit_loop_cap.slang`), so the MSL modules carry them
//! (build.rs checks every explicit module calls `ebLoopCapHit`). Here
//! `explicit_loop_cap_test.slang`, compiled to MSL with the same macros, runs a
//! loop whose exit never comes, a counted loop to a huge bound and a stepped loop
//! with a zero step: each stops (no hang) and the flag in gAcc's last word names
//! the stopped loop's source line (`engine_exec::explicit_plan::explicit_loop_cap_hit`).
//! As engine-webgpu's tests/explicit_loop_cap.rs does on wgpu.

use engine_exec::explicit_plan::{explicit_loop_cap_hit, EXPLICIT_LOOP_CAP, EXPLICIT_LOOP_CAP_TAG};
use engine_metal::LOOP_CAP_TEST;
use metal::*;

/// The source line of the test kernel's loop marked `loop-cap-test: <tag>`.
fn line_of(tag: &str) -> i32 {
    let src = include_str!("../../../slang/kernels/explicit_loop_cap_test.slang");
    let at = src
        .lines()
        .position(|l| l.contains(&format!("loop-cap-test: {tag}")))
        .unwrap_or_else(|| panic!("no loop marked {tag}"));
    at as i32 + 1
}

struct Gpu {
    device: Device,
    queue: CommandQueue,
    pso: ComputePipelineState,
}

impl Gpu {
    fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let options = CompileOptions::new();
        // As the explicit host compiles its modules (fast math on unless
        // ENGINE_FAST_MATH=off).
        options.set_fast_math_enabled(std::env::var("ENGINE_FAST_MATH").as_deref() != Ok("off"));
        let lib = device.new_library_with_source(LOOP_CAP_TEST.msl, &options).expect("compile");
        let func = lib.get_function(LOOP_CAP_TEST.entry, None).expect("function");
        let pso = device.new_compute_pipeline_state_with_function(&func).expect("pipeline");
        let queue = device.new_command_queue();
        Some(Self { device, queue, pso })
    }

    /// One dispatch with `out` (8 words) from the host: (out, gAcc) afterwards.
    fn run(&self, out: [u32; 8]) -> ([u32; 8], [i32; 4]) {
        let opt = MTLResourceOptions::StorageModeShared;
        let acc = self.device.new_buffer_with_data([0i32; 4].as_ptr().cast(), 16, opt);
        let o = self.device.new_buffer_with_data(out.as_ptr().cast(), 32, opt);
        let cb = self.queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&self.pso);
        for &(index, slot) in LOOP_CAP_TEST.binds {
            let b = if slot == 0 { &acc } else { &o };
            e.set_buffer(u64::from(index), Some(b), 0);
        }
        e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed, "the dispatch failed");
        let read = |b: &Buffer, n: usize| unsafe { std::slice::from_raw_parts(b.contents() as *const u32, n) }.to_vec();
        let o = read(&o, 8);
        let a = read(&acc, 4);
        (o.try_into().unwrap(), std::array::from_fn(|i| a[i] as i32))
    }
}

#[test]
fn a_runaway_loop_stops_at_the_cap_and_raises_the_flag_on_metal() {
    assert_eq!(LOOP_CAP_TEST.binds.len(), 2, "gAcc and gOut bound");
    let Some(gpu) = Gpu::new() else {
        eprintln!("no Metal device: skipped");
        return;
    };
    let cap = EXPLICIT_LOOP_CAP;
    let tag = EXPLICIT_LOOP_CAP_TAG;
    let (pass, end, step) = (line_of("pass"), line_of("end"), line_of("step"));
    // Ordinary bounds: every loop runs as written and no flag.
    let (o, a) = gpu.run([0, 1, 0, 5, 12, 4, 0, 0]);
    assert_eq!((o[0], o[2], o[3], o[6]), (0, 3, 5, 3));
    assert_eq!(a, [0; 4]);
    // The loop whose exit never comes: stopped after the cap's passes.
    let (o, a) = gpu.run([0, 0, 0, 5, 12, 4, 0, 0]);
    assert_eq!(o[0], cap, "passes of the runaway loop");
    assert_eq!((o[2], o[3], o[6]), (3, 5, 3));
    assert_eq!(a, [0, 0, 0, tag | pass]);
    // The counted loop to a huge bound: its end clamped to the cap.
    let (o, a) = gpu.run([0, 1, 0, u32::MAX, 12, 4, 0, 0]);
    assert_eq!((o[0], o[3]), (0, cap), "passes of the runaway counted loop");
    assert_eq!(a, [0, 0, 0, tag | end]);
    // A zero step: no pass, flagged.
    let (o, a) = gpu.run([0, 1, 0, 5, 12, 0, 0, 0]);
    assert_eq!((o[3], o[6]), (5, 0));
    assert_eq!(a, [0, 0, 0, tag | step]);
    // All three: the flag holds the highest line (atomic max), the error names it.
    let (o, a) = gpu.run([0, 0, 0, u32::MAX, 12, 0, 0, 0]);
    assert_eq!((o[0], o[3], o[6]), (cap, cap, 0));
    assert_eq!(a[3], tag | pass.max(end).max(step));
    let e = explicit_loop_cap_hit(a[3]).unwrap_err();
    assert!(e.contains(&format!("line {}", pass.max(end).max(step))), "{e}");
}
