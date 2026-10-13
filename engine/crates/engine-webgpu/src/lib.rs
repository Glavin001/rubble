//! WebGPU backend: Slang → SPIR-V → naga → WGSL.
//!
//! **Fair speed bar vs three-avbd:** same scenes in **Chrome WebGPU** on this machine
//! (not native Metal `scene_fps` vs their browser `gpuTotal`). Metal remains the M3
//! interactive gate; this crate is the apples-to-apples oracle path.

mod gpu;
pub use gpu::WgpuSim;

use engine_exec::{ExecError, PhaseTimingsMs, PhysicsBackend};
use engine_schema::{AvbdParams, StepCounters, WorldCapacity};
use std::borrow::Cow;

pub struct WebGpuWorld {
    capacity: WorldCapacity,
    params: AvbdParams,
    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    pub wgsl_shaders: Vec<(String, String)>,
}

fn spirv_entry_names() -> Vec<&'static str> {
    // Generated at build time; keep a compile-time fallback for rust-analyzer.
    include_str!(concat!(env!("OUT_DIR"), "/spirv_entries.txt"))
        .lines()
        .filter(|l| !l.is_empty())
        .map(|s| {
            // Leak is fine — process lifetime, one-time at first call via OnceLock below.
            Box::leak(s.to_string().into_boxed_str()) as &'static str
        })
        .collect()
}

fn entry_list() -> &'static [&'static str] {
    use std::sync::OnceLock;
    static LIST: OnceLock<Vec<&'static str>> = OnceLock::new();
    LIST.get_or_init(spirv_entry_names).as_slice()
}

impl WebGpuWorld {
    pub fn spirv_bytes(name: &str) -> anyhow::Result<Vec<u8>> {
        let path = format!("{}/spirv/{name}.spv", env!("OUT_DIR"));
        Ok(std::fs::read(&path).map_err(|e| anyhow::anyhow!("{path}: {e}"))?)
    }

    pub fn entry_names() -> &'static [&'static str] {
        entry_list()
    }

    /// Translate every SPIR-V entry to WGSL. Returns `(name, wgsl)` or `(name, err)`.
    pub fn compile_shaders_report() -> Vec<(String, Result<String, String>)> {
        let mut out = Vec::new();
        for name in entry_list() {
            let result = (|| -> anyhow::Result<String> {
                let bytes = Self::spirv_bytes(name)?;
                let module = naga::front::spv::parse_u8_slice(
                    &bytes,
                    &naga::front::spv::Options::default(),
                )?;
                let mut validator = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::empty(),
                    naga::valid::Capabilities::all(),
                );
                let info = validator.validate(&module)?;
                Ok(naga::back::wgsl::write_string(
                    &module,
                    &info,
                    naga::back::wgsl::WriterFlags::empty(),
                )?)
            })();
            out.push((
                (*name).to_string(),
                result.map_err(|e| format!("{e:#}")),
            ));
        }
        out
    }

    pub fn compile_shaders() -> anyhow::Result<Vec<(String, String)>> {
        let mut ok = Vec::new();
        let mut errs = Vec::new();
        for (name, r) in Self::compile_shaders_report() {
            match r {
                Ok(wgsl) => ok.push((name, wgsl)),
                Err(e) => errs.push(format!("{name}: {e}")),
            }
        }
        if !errs.is_empty() {
            anyhow::bail!(
                "SPIR-V→WGSL failed for {}/{} entries:\n{}",
                errs.len(),
                ok.len() + errs.len(),
                errs.join("\n")
            );
        }
        Ok(ok)
    }

    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        let wgsl_shaders = Self::compile_shaders()?;
        Ok(Self {
            capacity,
            params: AvbdParams::default(),
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            wgsl_shaders,
        })
    }

    /// Create a wgpu device and load every successfully translated WGSL module.
    /// Returns adapter label + counts (ok / fail).
    pub fn smoke_device() -> anyhow::Result<String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("no wgpu adapter"))?;
        let info = adapter.get_info();
        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("engine-webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))?;

        let report = Self::compile_shaders_report();
        let mut loaded = 0u32;
        let mut failed = 0u32;
        let mut fail_names = Vec::new();
        for (name, r) in &report {
            match r {
                Ok(wgsl) => {
                    let _ = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
                    });
                    loaded += 1;
                }
                Err(_) => {
                    failed += 1;
                    fail_names.push(name.clone());
                }
            }
        }
        Ok(format!(
            "{} / {} — WGSL loaded={loaded} failed={failed}{}",
            info.name,
            info.backend,
            if fail_names.is_empty() {
                String::new()
            } else {
                format!(" ({})", fail_names.join(", "))
            }
        ))
    }
}

impl PhysicsBackend for WebGpuWorld {
    fn name(&self) -> &'static str {
        "webgpu"
    }
    fn capacity(&self) -> &WorldCapacity {
        &self.capacity
    }
    fn params(&self) -> &AvbdParams {
        &self.params
    }
    fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }
    fn step(&mut self) -> Result<StepCounters, ExecError> {
        Err(ExecError::Backend(
            "WebGPU full step: Slang→WGSL modules ready; Chrome bench harness next \
             (fair bar vs three-avbd WebGPU — not native Metal scene_fps)"
                .into(),
        ))
    }
    fn last_timings(&self) -> PhaseTimingsMs {
        self.last_timings.clone()
    }
    fn last_counters(&self) -> StepCounters {
        self.last_counters
    }
}
