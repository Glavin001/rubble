//! Rubble's [`World`] for the web pages (wasm-bindgen): the host native and Unreal
//! run, on the browser's WebGPU through wgpu. A thin boundary like `engine-ffi`'s C
//! ABI: definitions arrive as plain objects whose missing fields keep the Rust
//! defaults, handles are numbers (`u64` bits as `f64`, exact below 2^53), and frames
//! leave as flat typed arrays.
//!
//! Nothing here waits on the GPU. [`WebWorld::step`] steps only while a readback
//! slot is free (otherwise it returns `false` and the page tries again next frame),
//! so a step never blocks on an earlier one's readback; [`WebWorld::poll`] returns
//! the frames that are ready.
//!
//! The page renders on the world's own device ([`WebWorld::device`]) and can bind
//! the world's buffers in place ([`WebWorld::pose_buffer`]).

use rubble::*;
use wasm_bindgen::prelude::*;

mod defs;
mod frame;

pub use frame::WebFrame;

fn err(e: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&e.to_string()).into()
}

fn res<T>(r: anyhow::Result<T>) -> Result<T, JsValue> {
    r.map_err(|e| err(format!("{e:#}")))
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&info.to_string().into());
        if let Ok(mut last) = PANIC.lock() {
            *last = info.to_string();
        }
    }));
}

static PANIC: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// The last panic's message (a panic traps the export as "unreachable").
#[wasm_bindgen(js_name = lastPanic)]
pub fn last_panic() -> String {
    PANIC.lock().map(|s| s.clone()).unwrap_or_default()
}

/// A high-performance WebGPU adapter's device with its full limits: what
/// `engine_webgpu::default_device` opens natively.
async fn web_device() -> Result<(wgpu::AdapterInfo, wgpu::Device, wgpu::Queue), JsValue> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            ..Default::default()
        })
        .await
        .map_err(|e| err(format!("no WebGPU adapter: {e}")))?;
    let info = adapter.get_info();
    let features = adapter.features()
        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("rubble world"),
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|e| err(format!("WebGPU device: {e}")))?;
    Ok((info, device, queue))
}

/// A world and the device it runs on.
#[wasm_bindgen]
pub struct WebWorld {
    world: World,
}

#[wasm_bindgen]
impl WebWorld {
    /// A world on a new device. `def`: a `WorldDef` (`defs::world_def`).
    pub async fn create(def: JsValue) -> Result<WebWorld, JsValue> {
        let def = defs::world_def(def)?;
        let (info, device, queue) = web_device().await?;
        let world = res(World::new(GpuContext::Shared { info, device, queue }, &def))?;
        Ok(WebWorld { world })
    }

    /// The world's `GPUDevice`: the page renders on it, so its passes read the
    /// world's buffers in place.
    pub fn device(&self) -> JsValue {
        self.world
            .device()
            .as_webgpu()
            .map(|d| JsValue::from(d.clone()))
            .unwrap_or(JsValue::NULL)
    }

    /// The `GPUBuffer` of body poses (`engine_webgpu::POSE_RECORD_BYTES` per
    /// body slot, in slot order), as the step left it.
    #[wasm_bindgen(js_name = poseBuffer)]
    pub fn pose_buffer(&self) -> JsValue {
        self.world
            .sim()
            .pose_buffer()
            .as_webgpu()
            .map(|b| JsValue::from(b.clone()))
            .unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen(js_name = stepCount)]
    pub fn step_count(&self) -> f64 {
        self.world.step_count() as f64
    }

    /// Whether a step can be taken now without waiting on a readback.
    #[wasm_bindgen(js_name = canStep)]
    pub fn can_step(&self) -> bool {
        self.world.can_step_without_waiting()
    }

    /// One fixed step if it can be taken without waiting (see [`Self::can_step`]);
    /// `false`: nothing happened, try again after a poll.
    pub fn step(&mut self) -> Result<bool, JsValue> {
        if !self.world.can_step_without_waiting() {
            let _ = self.world.device().poll(wgpu::PollType::Poll);
            return Ok(false);
        }
        res(self.world.step())?;
        Ok(true)
    }

    /// Every ready frame folded into one (never waits).
    pub fn poll(&mut self) -> Result<WebFrame, JsValue> {
        let frame = res(self.world.poll_coalesced())?;
        Ok(WebFrame::new(&self.world, frame))
    }

    // ---- bodies ----

    /// A body (`BodyDef`); returns its handle.
    #[wasm_bindgen(js_name = createBody)]
    pub fn create_body(&mut self, def: JsValue) -> Result<f64, JsValue> {
        let def = defs::body_def(def)?;
        Ok(handle(res(self.world.create_body(&def))?.to_bits()))
    }

    #[wasm_bindgen(js_name = destroyBody)]
    pub fn destroy_body(&mut self, body: f64) -> Result<(), JsValue> {
        res(self.world.destroy_body(BodyId::from_bits(bits(body))))
    }

    /// A shape on `body`: `def` a `ShapeDef`, `geometry` a tagged `Geometry`
    /// (`{ sphere: { center, radius } }`, `{ box: { half, transform } }`, ...).
    #[wasm_bindgen(js_name = createShape)]
    pub fn create_shape(&mut self, body: f64, def: JsValue, geometry: JsValue) -> Result<f64, JsValue> {
        let def = defs::shape_def(def)?;
        let geometry = defs::geometry(geometry)?;
        let id = res(self.world.create_shape(BodyId::from_bits(bits(body)), &def, geometry))?;
        Ok(handle(id.to_bits()))
    }

    /// A body's slot: its index in [`Self::pose_buffer`] and in frames.
    #[wasm_bindgen(js_name = bodySlot)]
    pub fn body_slot(&self, body: f64) -> u32 {
        BodyId::from_bits(bits(body)).index()
    }
}

fn handle(bits: u64) -> f64 {
    debug_assert!(bits < (1 << 53));
    bits as f64
}

fn bits(handle: f64) -> u64 {
    handle as u64
}
