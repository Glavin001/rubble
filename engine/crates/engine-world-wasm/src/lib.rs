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

pub mod broadside;
/// Broadside (`broadside.html`) and the timber yard (`timber.html`): their scenes stepped
/// in Rust, the page draws their poses.
mod broadside_page;
pub use broadside_page::*;
/// What a range page plays (also drawn natively by engine-viewer's `world_viewer`).
pub mod play;
/// The shooting range under Broadside and the timber yard.
pub mod range;
pub mod timber_yard;

/// The large-maps city (`city.html`): `rubble::city` placed over frames.
mod city;
mod defs;
mod demo;
mod explicit;
mod explicit_world;
mod frame;
/// Real-time pacing of the explicit solver's fixed steps on a display.
mod frame_pacer;
pub mod step_gate;
mod guidance;
/// The fracture lab (`fracture/`): its scenes as destructibles, its shots.
mod lab;
mod motion;
mod read;
mod scene;
mod showcase;
mod showcase_world;
mod ring_drive;
mod vehicle_lab;
mod soft;
mod terrain;
mod vehicle;

pub use city::WebCity;
pub use frame::WebFrame;
pub use frame_pacer::FramePacer;
pub use lab::{check_lab_chunk_order, LabRun, LabScene, LabSettleJob};
/// The lab's scene bindings (`engine-fracture-wasm`), exported from this module too.
pub use engine_fracture_wasm;
pub use scene::WebSnapshot;
pub use explicit::{explicit_sedan_truck, ExplicitSim};

fn err(e: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&e.to_string()).into()
}

/// An error, with the device's first uncaptured error or loss after it: a failed
/// readback says only that the browser refused to map it.
fn res<T>(r: anyhow::Result<T>) -> Result<T, JsValue> {
    r.map_err(|e| {
        let gpu = gpu_error();
        if gpu.is_empty() {
            err(format!("{e:#}"))
        } else {
            err(format!("{e:#} (GPU: {gpu})"))
        }
    })
}

static GPU_ERROR: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// The world device's first uncaptured WebGPU error and its loss, if any ("" while
/// there is none): what a phone's missing console would otherwise hide.
#[wasm_bindgen(js_name = gpuError)]
pub fn gpu_error() -> String {
    GPU_ERROR.lock().map(|s| s.clone()).unwrap_or_default()
}

fn note_gpu_error(what: String) {
    web_sys::console::error_1(&what.clone().into());
    if let Ok(mut e) = GPU_ERROR.lock() {
        if e.len() < 2000 {
            if !e.is_empty() {
                e.push_str("; ");
            }
            e.push_str(&what);
        }
    }
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
    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| note_gpu_error(e.to_string())));
    device.set_device_lost_callback(|reason, message| note_gpu_error(format!("device lost ({reason:?}): {message}")));
    Ok((info, device, queue))
}

/// Compile `specs` with `createComputePipelineAsync`, all at once, and wait for
/// them (`engine_webgpu::ComputeSpec`): the host's synchronous creation of the same
/// descriptors afterwards is a cache hit. Chrome compiles an async pipeline on a
/// worker thread; a synchronous one on the GPU process's main thread, where a
/// kernel compiling for longer than the GPU watchdog (the explicit contact frame:
/// about a minute cold on Metal) kills the GPU process in a windowed browser.
/// A spec this cannot express, or a compile the browser rejects, is left to the
/// synchronous creation, which reports it as before. `done(module, ms)` hears each
/// pipeline as it finishes (in completion order; `ms` since the compiles started).
/// A deferred spec ([`engine_webgpu::ComputeSpec::deferred`]) starts once the others
/// are done and is not waited for: the host creates it at first use.
pub(crate) async fn warm_pipelines(
    device: &wgpu::Device,
    specs: &[engine_webgpu::ComputeSpec],
    done: std::rc::Rc<dyn Fn(&str, f64)>,
) {
    #[cfg(target_arch = "wasm32")]
    {
        use js_sys::{Array, Function, Object, Promise, Reflect};
        use wasm_bindgen::JsCast;
        let Some(gpu) = device.as_webgpu() else {
            return;
        };
        let gpu: &JsValue = gpu.as_ref();
        let call = |obj: &JsValue, name: &str, arg: &JsValue| -> Result<JsValue, JsValue> {
            let f: Function = Reflect::get(obj, &name.into())?.dyn_into()?;
            f.call1(obj, arg)
        };
        let object = |fields: &[(&str, JsValue)]| -> Result<JsValue, JsValue> {
            let o = Object::new();
            for (k, v) in fields {
                Reflect::set(&o, &(*k).into(), v)?;
            }
            Ok(o.into())
        };
        let entry = |e: &wgpu::BindGroupLayoutEntry| -> Option<Result<JsValue, JsValue>> {
            let wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } = e.ty
            else {
                return None;
            };
            let kind = match ty {
                wgpu::BufferBindingType::Uniform => "uniform",
                wgpu::BufferBindingType::Storage { read_only: true } => "read-only-storage",
                wgpu::BufferBindingType::Storage { read_only: false } => "storage",
            };
            Some((|| {
                let buffer = object(&[
                    ("type", kind.into()),
                    ("hasDynamicOffset", has_dynamic_offset.into()),
                    (
                        "minBindingSize",
                        (min_binding_size.map_or(0, |n| n.get()) as f64).into(),
                    ),
                ])?;
                object(&[
                    ("binding", e.binding.into()),
                    ("visibility", e.visibility.bits().into()),
                    ("buffer", buffer),
                ])
            })())
        };
        let start = |s: &engine_webgpu::ComputeSpec| -> Result<Option<Promise>, JsValue> {
            let layout = match &s.layout {
                None => JsValue::from_str("auto"),
                Some(entries) => {
                    let list = Array::new();
                    for e in entries {
                        match entry(e) {
                            Some(v) => list.push(&v?),
                            None => return Ok(None),
                        };
                    }
                    let bgl = call(
                        gpu,
                        "createBindGroupLayout",
                        &object(&[("entries", list.into())])?,
                    )?;
                    let groups = Array::of1(&bgl);
                    call(
                        gpu,
                        "createPipelineLayout",
                        &object(&[("bindGroupLayouts", groups.into())])?,
                    )?
                }
            };
            let module = call(
                gpu,
                "createShaderModule",
                &object(&[("label", s.label.into()), ("code", s.wgsl.as_str().into())])?,
            )?;
            let compute = object(&[("module", module), ("entryPoint", s.entry.into())])?;
            let desc = object(&[
                ("label", s.label.into()),
                ("layout", layout),
                ("compute", compute),
            ])?;
            Ok(Some(
                call(gpu, "createComputePipelineAsync", &desc)?.dyn_into()?,
            ))
        };
        let now = || js_sys::Date::now();
        let t0 = now();
        // The pipelines the first frame needs, all at once; then (not waited for) the
        // deferred ones, so they do not hold up the others' compiles.
        let mut pending = Vec::new();
        let mut later = Vec::new();
        for s in specs.iter().filter(|s| !s.deferred) {
            match start(s) {
                Ok(Some(p)) => {
                    // Each pipeline's completion, as it lands (settled either way).
                    let (done, module) = (done.clone(), s.module);
                    let hear = wasm_bindgen::closure::Closure::once_into_js(move |_: JsValue| {
                        done(module, now() - t0);
                    });
                    // p.then(hear, hear): one of the two runs; the derived promise only
                    // reports.
                    if let Ok(then) = Reflect::get(&p, &"then".into()).and_then(|f| f.dyn_into::<Function>()) {
                        let _ = then.call2(&p, &hear, &hear);
                    }
                    pending.push((s.label, wasm_bindgen_futures::JsFuture::from(p)));
                }
                Ok(None) => {}
                Err(e) => web_sys::console::warn_2(
                    &format!("pipeline warm-up skipped ({})", s.label).into(),
                    &e,
                ),
            }
        }
        for (label, p) in pending {
            if let Err(e) = p.await {
                web_sys::console::warn_2(
                    &format!(
                        "pipeline warm-up failed ({label}); the synchronous creation reports it"
                    )
                    .into(),
                    &e,
                );
            }
        }
        for s in specs.iter().filter(|s| s.deferred) {
            match start(s) {
                Ok(Some(p)) => {
                    let (done, module) = (done.clone(), s.module);
                    let hear = wasm_bindgen::closure::Closure::once_into_js(move |_: JsValue| {
                        done(module, now() - t0);
                    });
                    if let Ok(then) = Reflect::get(&p, &"then".into()).and_then(|f| f.dyn_into::<Function>()) {
                        let _ = then.call2(&p, &hear, &hear);
                    }
                    later.push(p);
                }
                Ok(None) => {}
                Err(e) => web_sys::console::warn_2(
                    &format!("pipeline warm-up skipped ({})", s.label).into(),
                    &e,
                ),
            }
        }
        drop(later);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (device, specs, done);
}

/// The limits a world's buffers and kernels run into, as one line.
pub(crate) fn limits_text(l: &wgpu::Limits) -> String {
    format!(
        "maxBufferSize {}, maxStorageBufferBindingSize {}, maxStorageBuffersPerShaderStage {}, maxBindGroups {}, maxBindingsPerBindGroup {}, maxComputeWorkgroupStorageSize {}, maxComputeInvocationsPerWorkgroup {}, maxComputeWorkgroupsPerDimension {}, maxUniformBufferBindingSize {}",
        l.max_buffer_size,
        l.max_storage_buffer_binding_size,
        l.max_storage_buffers_per_shader_stage,
        l.max_bind_groups,
        l.max_bindings_per_bind_group,
        l.max_compute_workgroup_storage_size,
        l.max_compute_invocations_per_workgroup,
        l.max_compute_workgroups_per_dimension,
        l.max_uniform_buffer_binding_size,
    )
}

/// Fails with every buffer, layout, shader or pipeline creation the browser rejected
/// while the world was built (`engine_webgpu::creation_errors`), named, with the device's
/// limits: the uncaptured error the page would see instead names none of them.
pub(crate) async fn creation_check(world: &World) -> Result<(), JsValue> {
    let errors = engine_webgpu::creation_errors().await;
    if errors.is_empty() {
        return Ok(());
    }
    let (total, largest, at) = world.sim().buffer_bytes();
    Err(err(format!(
        "WebGPU rejected {} object(s) while building the world: {} — device limits: {}; world buffers {} MB, largest b{at} {} MB",
        errors.len(),
        errors.join(" | "),
        limits_text(&world.device().limits()),
        total >> 20,
        largest >> 20
    )))
}

/// A world and the device it runs on.
#[wasm_bindgen]
pub struct WebWorld {
    world: World,
    /// Back-pressure on completed GPU work for a world with explicit vehicles
    /// ([`step_gate::StepGate`]; shared with the queue's work-done callbacks).
    gate: std::sync::Arc<std::sync::Mutex<step_gate::StepGate>>,
}

/// Milliseconds on a monotonic clock (the step gate's).
fn now_ms() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<web_time::Instant> = OnceLock::new();
    START.get_or_init(web_time::Instant::now).elapsed().as_secs_f64() * 1e3
}

impl WebWorld {
    pub(crate) fn of(world: World) -> WebWorld {
        WebWorld { world, gate: Default::default() }
    }

    /// Whether the step gate applies: a world with explicit vehicles (their 2 kHz
    /// frame is the GPU work that cannot be taken back; rigid-only worlds keep their
    /// readback gate alone, unchanged).
    fn gated(&self) -> bool {
        self.world.explicit_vehicles().is_some()
    }
}

#[wasm_bindgen]
impl WebWorld {
    /// A world on a new device. `def`: a `WorldDef` (`defs::world_def`).
    pub async fn create(def: JsValue) -> Result<WebWorld, JsValue> {
        let def = defs::world_def(def)?;
        let (info, device, queue) = web_device().await?;
        let world = res(World::new(GpuContext::Shared { info, device, queue }, &def))?;
        creation_check(&world).await?;
        Ok(WebWorld::of(world))
    }

    /// The device's limits that size a world (requested as the adapter's): for a page's
    /// error line and diagnostics.
    #[wasm_bindgen(js_name = deviceLimits)]
    pub fn device_limits(&self) -> String {
        let (total, largest, at) = self.world.sim().buffer_bytes();
        format!(
            "{}; world buffers {} MB, largest b{at} {} MB",
            limits_text(&self.world.device().limits()),
            total >> 20,
            largest >> 20
        )
    }

    /// The world's `GPUDevice`: the page renders on it, so its passes read the
    /// world's buffers in place.
    pub fn device(&self) -> JsValue {
        gpu_device(self.world.device())
    }

    /// The `GPUBuffer` of body poses (`engine_webgpu::POSE_RECORD_BYTES` per
    /// body slot, in slot order), as the step left it.
    #[wasm_bindgen(js_name = poseBuffer)]
    pub fn pose_buffer(&self) -> JsValue {
        gpu_buffer(self.world.sim().pose_buffer())
    }

    #[wasm_bindgen(js_name = stepCount)]
    pub fn step_count(&self) -> f64 {
        self.world.step_count() as f64
    }

    /// Whether a step can be taken now without waiting on a readback.
    #[wasm_bindgen(js_name = canStep)]
    pub fn can_step(&self) -> bool {
        self.world.can_step_without_waiting() && (!self.gated() || self.gate.lock().unwrap().may_step(now_ms()))
    }

    /// One fixed step if it can be taken without waiting (see [`Self::can_step`]);
    /// `false`: nothing happened, try again after a poll.
    pub fn step(&mut self) -> Result<bool, JsValue> {
        if !self.world.can_step_without_waiting() {
            let _ = self.world.device().poll(wgpu::PollType::Poll);
            return Ok(false);
        }
        let gated = self.gated();
        if gated && !self.gate.lock().unwrap().may_step(now_ms()) {
            let _ = self.world.device().poll(wgpu::PollType::Poll);
            return Ok(false);
        }
        res(self.world.step())?;
        if gated {
            // In flight until the queue reports this submission done.
            self.gate.lock().unwrap().submitted(now_ms());
            let gate = self.gate.clone();
            self.world.queue().on_submitted_work_done(move || gate.lock().unwrap().completed(now_ms()));
        }
        Ok(true)
    }

    /// The step gate's pause message (a world with explicit vehicles: steps over the
    /// GPU budget twice in a row, a step that does not finish, or a lost device), or
    /// `undefined` while it runs. While set, [`Self::can_step`] is false: the page
    /// shows the message and offers [`Self::resume_steps`].
    #[wasm_bindgen(js_name = overload)]
    pub fn overload(&self) -> Option<String> {
        self.gate.lock().unwrap().paused().map(str::to_string)
    }

    /// Step again after [`Self::overload`] (the user's decision).
    #[wasm_bindgen(js_name = resumeSteps)]
    pub fn resume_steps(&self) {
        self.gate.lock().unwrap().resume(now_ms());
    }

    /// The device was lost: nothing is submitted through this world again.
    #[wasm_bindgen(js_name = stopSteps)]
    pub fn stop_steps(&self, reason: &str) {
        self.gate.lock().unwrap().stop(reason);
    }

    /// The step gate's readout: `[steps in flight, last step's GPU service time (ms),
    /// the worst since the last resume (ms), steps refused, budget (ms)]`.
    #[wasm_bindgen(js_name = stepPace)]
    pub fn step_pace(&self) -> Vec<f64> {
        let g = self.gate.lock().unwrap();
        let p = g.pace();
        vec![f64::from(p.in_flight), p.last_ms, p.worst_ms, p.refused as f64, g.budget_ms()]
    }

    /// The step budget (ms; default 50): a step's GPU service time over it twice in a
    /// row pauses stepping ([`Self::overload`]).
    #[wasm_bindgen(js_name = setStepBudget)]
    pub fn set_step_budget(&self, ms: f64) {
        self.gate.lock().unwrap().set_budget_ms(ms);
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

    #[wasm_bindgen(js_name = bodyCount)]
    pub fn body_count(&self) -> u32 {
        self.world.body_count() as u32
    }

    /// Mass, centre (body frame) and principal inertia about it, axes aligned.
    #[wasm_bindgen(js_name = bodySetMassData)]
    pub fn body_set_mass_data(&mut self, body: f64, mass: f32, center: &[f32], inertia: &[f32]) -> Result<(), JsValue> {
        let mass = MassData::principal(mass, vec3(center)?, vec3(inertia)?);
        res(self.world.body_set_mass_data(id(body), mass))
    }

    #[wasm_bindgen(js_name = bodySetTransform)]
    pub fn body_set_transform(&mut self, body: f64, position: &[f32], rotation: &[f32]) -> Result<(), JsValue> {
        let t = Transform::new(vec3(position)?, vec4(rotation)?);
        res(self.world.body_set_transform(id(body), t))
    }

    #[wasm_bindgen(js_name = bodySetVelocity)]
    pub fn body_set_velocity(&mut self, body: f64, linear: &[f32], angular: &[f32]) -> Result<(), JsValue> {
        res(self.world.body_set_velocity(id(body), vec3(linear)?, vec3(angular)?))
    }

    #[wasm_bindgen(js_name = bodyApplyLinearImpulse)]
    pub fn body_apply_linear_impulse(&mut self, body: f64, impulse: &[f32], point: &[f32], wake: bool) -> Result<(), JsValue> {
        res(self.world.body_apply_linear_impulse(id(body), vec3(impulse)?, vec3(point)?, wake))
    }

    #[wasm_bindgen(js_name = bodySetAwake)]
    pub fn body_set_awake(&mut self, body: f64, awake: bool) -> Result<(), JsValue> {
        res(self.world.body_set_awake(id(body), awake))
    }

    // ---- shapes ----

    /// A heightfield of `nx × nz` samples (row-major `heights`, centred on
    /// `transform`) on a static body.
    #[wasm_bindgen(js_name = createHeightField)]
    #[allow(clippy::too_many_arguments)]
    pub fn create_height_field(
        &mut self,
        body: f64,
        def: JsValue,
        nx: u32,
        nz: u32,
        dx: f32,
        dz: f32,
        heights: Vec<f32>,
        transform: JsValue,
    ) -> Result<f64, JsValue> {
        let def = defs::shape_def(def)?;
        let field = HeightField::new(nx, nz, dx, dz, heights).map_err(err)?;
        let mut f = defs::Fields::new("createHeightField", serde_json::Map::new());
        if !transform.is_undefined() && !transform.is_null() {
            let mut m = serde_json::Map::new();
            m.insert("t".into(), serde_wasm_bindgen::from_value(transform).map_err(err)?);
            f = defs::Fields::new("createHeightField", m);
        }
        let t = defs::transform(&mut f, "t")?;
        let s = res(self.world.create_height_field_shape(id(body), &def, &field, t))?;
        Ok(handle(s.to_bits()))
    }

    /// Overwrite samples `[i0, i0 + w) × [j0, j0 + d)` (row-major `values`).
    #[wasm_bindgen(js_name = setHeightFieldRegion)]
    pub fn set_height_field_region(&mut self, shape: f64, i0: u32, j0: u32, w: u32, d: u32, values: &[f32]) -> Result<(), JsValue> {
        res(self.world.set_height_field_region(ShapeId::from_bits(bits(shape)), i0, j0, w, d, values))
    }

    /// The heightfield's surface height under shape-local `(x, z)`, as the GPU
    /// evaluates it (the triangle containing the point); `NaN` off the field.
    #[wasm_bindgen(js_name = heightFieldSurface)]
    pub fn height_field_surface(&self, shape: f64, x: f32, z: f32) -> Result<f32, JsValue> {
        match res(self.world.shape_geometry(ShapeId::from_bits(bits(shape))))? {
            Geometry::HeightField { field, .. } => Ok(field.surface(x, z).map_or(f32::NAN, |(h, _)| h)),
            _ => Err(err("not a heightfield")),
        }
    }

    /// The shape's index in [`Self::collider_buffer`].
    #[wasm_bindgen(js_name = shapeCollider)]
    pub fn shape_collider(&self, shape: f64) -> Result<u32, JsValue> {
        res(self.world.shape_collider(ShapeId::from_bits(bits(shape))))
    }

    /// The arena base of a hull or heightfield shape's records in
    /// [`Self::hull_arena_buffer`].
    #[wasm_bindgen(js_name = shapeArenaBase)]
    pub fn shape_arena_base(&self, shape: f64) -> Result<u32, JsValue> {
        let c = self.shape_collider(shape)?;
        Ok(engine_schema::convex::convex_params_base(self.world.sim().collider_record(c).params))
    }

    /// Collider records in use (the instance count for drawing every collider).
    #[wasm_bindgen(js_name = colliderCount)]
    pub fn collider_count(&self) -> u32 {
        self.world.sim().collider_count()
    }

    // ---- solver ----

    /// The solver constants (`AvbdParams`) as an object.
    pub fn solver(&self) -> JsValue {
        defs::solver_to_js(&self.world.solver())
    }

    /// Change solver constants: the fields given replace the current ones.
    #[wasm_bindgen(js_name = setSolver)]
    pub fn set_solver(&mut self, params: JsValue) -> Result<(), JsValue> {
        let p = defs::solver(params, self.world.solver())?;
        self.world.set_solver(p);
        Ok(())
    }

    // ---- GPU buffers, for renderers on [`Self::device`] ----
    // A buffer can be replaced when it grows: fetch it each frame and rebind
    // when it is a different object.

    /// `BodyState` records (112 B) by body slot: solver-frame (centre of mass) poses.
    #[wasm_bindgen(js_name = bodyBuffer)]
    pub fn body_buffer(&self) -> JsValue {
        gpu_buffer(self.world.sim().body_buffer())
    }

    /// `Collider` records (64 B).
    #[wasm_bindgen(js_name = colliderBuffer)]
    pub fn collider_buffer(&self) -> JsValue {
        gpu_buffer(self.world.sim().collider_buffer())
    }

    /// The hull and heightfield arena (`vec4<u32>` records).
    #[wasm_bindgen(js_name = hullArenaBuffer)]
    pub fn hull_arena_buffer(&self) -> JsValue {
        gpu_buffer(self.world.sim().hull_arena_buffer())
    }
}

/// Solver flag bits (`AvbdParams::flags`).
#[wasm_bindgen(js_name = solverFlags)]
pub fn solver_flags() -> JsValue {
    let o = js_sys::Object::new();
    for (k, v) in [
        ("REUSE_CONTACTS", engine_schema::FLAG_REUSE_CONTACTS),
        ("CONTACT_GAP", engine_schema::FLAG_CONTACT_GAP),
        ("SPEC_VELOCITY", engine_schema::FLAG_SPEC_VELOCITY),
        ("WORLD_INERTIA", engine_schema::FLAG_WORLD_INERTIA),
    ] {
        let _ = js_sys::Reflect::set(&o, &k.into(), &v.into());
    }
    o.into()
}

/// The browser's `GPUBuffer` under `b` (`null` off the WebGPU backend, e.g. a
/// native build of this crate for its tests).
fn gpu_buffer(b: &wgpu::Buffer) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    if let Some(b) = b.as_webgpu() {
        return JsValue::from(b.clone());
    }
    let _ = b;
    JsValue::NULL
}

/// The browser's `GPUDevice` under `d` (`null` off the WebGPU backend).
fn gpu_device(d: &wgpu::Device) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    if let Some(d) = d.as_webgpu() {
        return JsValue::from(d.clone());
    }
    let _ = d;
    JsValue::NULL
}

fn id(body: f64) -> BodyId {
    BodyId::from_bits(bits(body))
}

fn vec3(v: &[f32]) -> Result<[f32; 3], JsValue> {
    v.try_into().map_err(|_| err("expected 3 numbers"))
}

fn vec4(v: &[f32]) -> Result<[f32; 4], JsValue> {
    v.try_into().map_err(|_| err("expected 4 numbers"))
}

fn handle(bits: u64) -> f64 {
    debug_assert!(bits < (1 << 53));
    bits as f64
}

fn bits(handle: f64) -> u64 {
    handle as u64
}
