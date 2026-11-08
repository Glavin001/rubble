//! The demos' scenes (`rubble::play`) through the C ABI: a preset by name (the same scene
//! its page builds, with any of the page's URL switches over it), placed in its own world,
//! stepped, and its counts. Rigid stacks, vehicles, soft bodies, motion and terrain.

use super::*;
use rubble::play::Playground;

/// A placed preset and its world (`rbCreatePlay`).
pub struct RbPlay {
    play: Playground,
}

/// A play's counts as of its last step (`rbPlay_GetStats`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbPlayStats {
    pub steps: u64,
    pub bodies: u32,
    pub awake: u32,
    pub contacts: u32,
    /// The last step's overflow flags (0: none).
    pub overflow: u32,
}

unsafe fn play_mut<'a>(play: *mut RbPlay) -> anyhow::Result<&'a mut RbPlay> {
    play.as_mut().ok_or_else(|| invalid("null play"))
}

unsafe fn c_text<'a>(s: *const c_char, what: &'static str) -> anyhow::Result<&'a str> {
    if s.is_null() {
        return Err(invalid(what));
    }
    Ok(CStr::from_ptr(s).to_str()?)
}

/// Preset `preset`'s scene (`gauntlet`, `soft-tbone`, `ring`, ...) placed in a world on
/// the default GPU; `query` (null or empty: none) is any of its page's URL switches
/// (`speed=80&overlap=0.5`) over the preset's own. A preset placed elsewhere (the lab,
/// the range, the city) fails with its reason (`rbGetLastError`).
#[no_mangle]
pub unsafe extern "C" fn rbCreatePlay(
    preset: *const c_char,
    query: *const c_char,
    out: *mut *mut RbPlay,
) -> RbResult {
    guard_args(|| {
        let name = c_text(preset, "null preset")?;
        let query = if query.is_null() {
            ""
        } else {
            c_text(query, "null query")?
        };
        let play = Playground::preset(GpuContext::Default, name, query)?;
        write(out, Box::into_raw(Box::new(RbPlay { play })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbDestroyPlay(play: *mut RbPlay) {
    if !play.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(play))));
    }
}

/// `steps` steps, waiting for each step's results.
#[no_mangle]
pub unsafe extern "C" fn rbPlay_Step(play: *mut RbPlay, steps: u32) -> RbResult {
    guard_args(|| play_mut(play)?.play.step(steps))
}

/// The play's counts as of its last step.
#[no_mangle]
pub unsafe extern "C" fn rbPlay_GetStats(play: *mut RbPlay, out: *mut RbPlayStats) -> RbResult {
    guard_args(|| {
        let s = play_mut(play)?.play.stats();
        write(
            out,
            RbPlayStats {
                steps: s.steps,
                bodies: s.bodies,
                awake: s.awake,
                contacts: s.contacts,
                overflow: s.overflow,
            },
        )
    })
}
