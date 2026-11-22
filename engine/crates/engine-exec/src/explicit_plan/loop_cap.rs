//! The explicit kernels' loop cap and the error for a flag read back
//! (`explicit_loop_cap.slang`).

/// The explicit kernels' loop cap (`EB_LOOP_CAP` in `explicit_beams.slang`, every
/// target): a loop whose bound is not a constant runs at most this many passes.
/// Far above any legitimate count (a few loops run on one lane over a whole
/// vehicle's beams: tens of thousands for a large vehicle), so a capped loop
/// computes what an uncapped one does.
pub const EXPLICIT_LOOP_CAP: u32 = 1 << 20;

/// The kernel's tag on a loop-cap flag (`EB_LOOP_CAP_TAG`): the flag is the tag
/// | the stopped loop's id (its `EB_END(ID, ..)` / `EB_PASS(ID, ..)`). Untagged, the flag is the loop's number in
/// its WGSL module (counting from 1): the frozen test modules
/// (`explicit_beams_baseline.slang`, `explicit_beams_ref.slang`), capped by
/// engine-webgpu's WGSL post-process.
pub const EXPLICIT_LOOP_CAP_TAG: i32 = 1 << 24;

/// The error for a loop-cap flag read back from gAcc ([`ExplicitLayout::loop_cap_flag`];
/// 0: none).
pub fn explicit_loop_cap_hit(flag: i32) -> Result<(), String> {
    if flag == 0 {
        return Ok(());
    }
    let place = if flag & EXPLICIT_LOOP_CAP_TAG != 0 {
        let id = flag & !EXPLICIT_LOOP_CAP_TAG;
        format!("the loop with id {id} (`rg -n 'EB_[A-Z_]*\\({id},|ebLoopCapHit\\({id}\\)' engine/slang/kernels`)")
    } else {
        format!("a kernel loop (number {flag} in its WGSL module, counting from 1)")
    };
    // The flag names the highest stopped loop: past its own cap (EXPLICIT_LOOP_CAP passes,
    // or a barrier loop's real maximum), or the first to find the frame's pass budget
    // spent (the passes were taken by all the loops before it).
    Err(format!(
        "{place} was stopped: at its cap ({EXPLICIT_LOOP_CAP} passes, or a barrier loop's \
         maximum) or when the frame's pass budget ran out; that frame's results are invalid \
         (a runaway loop: a corrupt count or bound)"
    ))
}
