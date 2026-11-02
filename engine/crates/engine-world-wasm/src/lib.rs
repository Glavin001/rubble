//! Rubble's world API in the browser: Broadside (`engine-webgpu/broadside.html`) and the
//! timber yard (`timber.html`). The scene ([`broadside`]: the Shardwright galleon stern as a
//! threshold destructible, the captain, the broadsides and the closing volley; or
//! [`timber_yard`]: a timber-frame house and two plank walls, free play or a raze; both on a
//! [`range`]) is stepped here on this module's WebGPU device; each frame's poses go to the
//! page, which draws them with the fracture lab's renderer (`fracture/render.mjs`) over the
//! render scene it builds from the same GLB (`fracture/wasm.mjs` `buildScene`, the same
//! `engine_fracture::build_scene`). The `broadside_*` exports drive either scene
//! ([`broadside_set_scene`]).
//!
//! The pose buffer (the draw shader's `BodyPose`, [`POSE_FLOATS`] floats a body): the render
//! scene's bodies (the ground, every chunk in order, its parked projectiles), then the
//! extras the page draws as a dynamic mesh ([`broadside_extra_vertices`]): the captain,
//! [`broadside::MAX_BALLS`] balls, and the crush splinters so far.
//!
//! Exports that wait for the GPU (`broadside_load`, `broadside_frame`) suspend the wasm
//! stack (`engine-gpu-wait`): the page calls them through `WebAssembly.promising` (JSPI), as
//! the lab's stress mode does (`fracture/gpu-wait.mjs`).

pub mod broadside;
pub mod play;
pub mod range;
pub mod timber_yard;

use play::Play;
use rubble::GpuContext;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

thread_local! {
    static GPU: RefCell<Option<stress_gpu::gpu::Gpu>> = const { RefCell::new(None) };
    static ASSET: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LEVEL: RefCell<u8> = const { RefCell::new(broadside::LEVEL) };
    static TAKE: RefCell<u8> = const { RefCell::new(0) };
    static SCENE: RefCell<u8> = const { RefCell::new(0) };
    static LOADED: RefCell<Option<Play>> = const { RefCell::new(None) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}
static PANIC: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
/// Why this module's device was lost, when it was (`broadside_lost`).
static LOST: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

fn fail(e: impl std::fmt::Display) -> bool {
    ERROR.set(e.to_string());
    false
}

/// Open this module's WebGPU device (the browser's adapter, no canvas: the page draws).
#[wasm_bindgen]
pub async fn broadside_init() -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&info.to_string().into());
        if let Ok(mut last) = PANIC.try_lock() {
            *last = info.to_string();
        }
    }));
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let gpu = stress_gpu::gpu::Gpu::with_instance(&instance, None)
        .await
        .map_err(|e| JsValue::from_str(&e))?;
    gpu.device.set_device_lost_callback(|reason, message| {
        if let Ok(mut lost) = LOST.lock() {
            *lost = format!("{reason:?}: {message}");
        }
    });
    GPU.set(Some(gpu));
    Ok(())
}

/// The last panic's message (a panic traps the export: "unreachable").
#[wasm_bindgen]
pub fn broadside_panic() -> String {
    PANIC.lock().map(|p| p.clone()).unwrap_or_default()
}

/// Why this module's WebGPU device was lost ("" while it is not): the browser's reason,
/// which a phone's missing console would otherwise hide.
#[wasm_bindgen]
pub fn broadside_lost() -> String {
    LOST.lock().map(|l| l.clone()).unwrap_or_default()
}

/// Why the last call that returned false failed.
#[wasm_bindgen]
pub fn broadside_error() -> String {
    ERROR.with_borrow(|e| e.clone())
}

/// The ship's baked GLB for the next [`broadside_load`]. (A waiting export takes no
/// arguments: `WebAssembly.promising` calls the raw export, past wasm-bindgen's glue.)
#[wasm_bindgen]
pub fn broadside_set_asset(glb: Vec<u8>) {
    ASSET.set(glb);
}

/// The fracture level the next [`broadside_load`] builds (`broadside::LEVEL`'s list).
#[wasm_bindgen]
pub fn broadside_set_level(level: u8) {
    LEVEL.set(level);
}

/// The script the next [`broadside_load`] runs: Broadside 0 the captain's walk, 1 raze the
/// ship; the timber yard 0 free play, 1 raze the yard.
#[wasm_bindgen]
pub fn broadside_set_take(take: u8) {
    TAKE.set(take);
}

/// The scene the next [`broadside_load`] builds from the asset: 0 Broadside (the galleon
/// stern), 1 the timber yard.
#[wasm_bindgen]
pub fn broadside_set_scene(scene: u8) {
    SCENE.set(scene);
}

/// Build the scene from the asset ([`broadside_set_asset`]) on this module's device.
/// Waits for the GPU. False on an error (`broadside_error`).
#[wasm_bindgen]
pub fn broadside_load() -> bool {
    LOADED.set(None);
    match load() {
        Ok(l) => {
            LOADED.set(Some(l));
            true
        }
        Err(e) => fail(format!("{e:#}")),
    }
}

fn load() -> anyhow::Result<Play> {
    let glb = ASSET.with_borrow(|a| a.clone());
    anyhow::ensure!(!glb.is_empty(), "broadside_set_asset first");
    let context = GPU.with_borrow(|g| {
        g.as_ref().map(|g| GpuContext::Shared {
            info: g.adapter_info.clone(),
            device: g.device.clone(),
            queue: g.queue.clone(),
        })
    });
    Play::load(
        context.ok_or_else(|| anyhow::anyhow!("broadside_init first"))?,
        &glb,
        SCENE.with_borrow(|s| *s),
        LEVEL.with_borrow(|l| *l),
        TAKE.with_borrow(|t| *t) == 1,
    )
}

/// The first extra body in the pose buffer (after the render scene's).
#[wasm_bindgen]
pub fn broadside_body_base() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.render_bodies as u32))
}

/// The extras' render vertices ([`Play::extra_vertices`]). Redraw them when
/// [`broadside_extras_version`] changes.
#[wasm_bindgen]
pub fn broadside_extra_vertices() -> Vec<f32> {
    LOADED.with_borrow(|l| l.as_ref().map_or(Vec::new(), |l| l.extra_vertices()))
}

/// The vertex where debris `k`'s vertices start in [`broadside_extra_vertices`] (the
/// captain's and the balls' come first): the page appends new debris there.
#[wasm_bindgen]
pub fn broadside_splinter_vertex_offset(k: u32) -> u32 {
    Play::splinter_vertex_offset(k)
}

/// Debris so far (crush particles and fragments).
#[wasm_bindgen]
pub fn broadside_splinter_count() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.splinter_count()))
}

/// The render vertices of debris `from` onward (append them at
/// [`broadside_splinter_vertex_offset`]`(from)`).
#[wasm_bindgen]
pub fn broadside_splinter_vertices(from: u32) -> Vec<f32> {
    LOADED.with_borrow(|l| l.as_ref().map_or(Vec::new(), |l| l.splinter_vertices(from)))
}

/// Changes whenever [`broadside_extra_vertices`] would.
#[wasm_bindgen]
pub fn broadside_extras_version() -> u32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(0, |l| l.extras_version()))
}

/// Advance `steps` steps, then wait for their readbacks. Waits for the GPU: the scene is
/// taken out of its slot while this runs (no borrow is held across a suspension). False
/// on an error (`broadside_error`).
#[wasm_bindgen]
pub fn broadside_frame(steps: u32) -> bool {
    let Some(mut l) = LOADED.take() else {
        return fail("no scene loaded");
    };
    let r = l.frame(steps);
    LOADED.set(Some(l));
    match r {
        Ok(()) => true,
        Err(e) => fail(format!("{e:#}")),
    }
}

/// The pose buffer at `alpha` (0..1) between the last two steps ([`Play::poses`]).
#[wasm_bindgen]
pub fn broadside_poses(alpha: f32) -> Vec<f32> {
    LOADED.with_borrow(|l| l.as_ref().map_or(Vec::new(), |l| l.poses(alpha)))
}

/// The take's camera for a frame `wall` seconds after the last: `[eye, target]` (empty:
/// none, the page's own camera).
#[wasm_bindgen]
pub fn broadside_camera(wall: f32) -> Vec<f32> {
    LOADED.with_borrow_mut(|l| {
        l.as_mut()
            .and_then(|l| l.scene.camera(wall))
            .map_or(Vec::new(), |(e, t)| [e, t].concat())
    })
}

/// Simulated seconds per wall second the take plays at now.
#[wasm_bindgen]
pub fn broadside_time_scale() -> f32 {
    LOADED.with_borrow(|l| l.as_ref().map_or(1.0, |l| l.scene.time_scale()))
}

/// Free play: stop the script (barrage, walk, finale) from here on, or start it again.
#[wasm_bindgen]
pub fn broadside_set_scripted(on: bool) {
    LOADED.with_borrow_mut(|l| {
        if let Some(l) = l.as_mut() {
            l.scene.set_scripted(on);
        }
    })
}

/// Fire a ball from `origin` along `direction` (world, the camera's aim) at the take's
/// speed, before the next step.
#[wasm_bindgen]
pub fn broadside_shoot_ray(origin: &[f32], direction: &[f32]) -> bool {
    LOADED.with_borrow_mut(|l| {
        let Some(l) = l.as_mut() else { return false };
        if origin.len() != 3 || direction.len() != 3 {
            return false;
        }
        let v = |s: &[f32]| [s[0], s[1], s[2]];
        l.shoot_ray(v(origin), v(direction));
        true
    })
}

/// The scene's state as JSON ([`Play::stats`]).
#[wasm_bindgen]
pub fn broadside_stats() -> String {
    LOADED.with_borrow(|l| l.as_ref().map_or("{}".into(), |l| l.stats().to_string()))
}

/// The save area of the Asyncify build (`engine_world_wasm_bg.asyncify.wasm`, for browsers
/// without JSPI; `engine/scripts/build-web.sh`, `fracture/gpu-wait.mjs`): a header of the
/// next free address and the end address, then room for the stack a suspended export saves.
/// Returns the area's address; the page calls it once. The JSPI build never calls it.
#[no_mangle]
pub extern "C" fn broadside_asyncify_buffer() -> u32 {
    const BYTES: usize = 1 << 20;
    let area: &'static mut [u8] = Box::leak(vec![0u8; BYTES].into_boxed_slice());
    let base = area.as_mut_ptr() as usize as u32;
    area[0..4].copy_from_slice(&(base + 8).to_le_bytes());
    area[4..8].copy_from_slice(&(base + BYTES as u32).to_le_bytes());
    base
}
