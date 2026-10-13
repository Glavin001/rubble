//! WebGPU backend: Slang → SPIR-V artifacts, naga → WGSL, wgpu submission.
//! Same phase graph / schema as Metal. M3 proof remains MetalWorld.

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

impl WebGpuWorld {
    pub fn spirv_bytes(name: &str) -> anyhow::Result<Vec<u8>> {
        let path = format!("{}/spirv/{name}.spv", env!("OUT_DIR"));
        Ok(std::fs::read(path)?)
    }

    pub fn compile_shaders() -> anyhow::Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        for name in [
            "abi_roundtrip",
            "predict_bodies",
            "narrowphase",
            "solve_primal_color",
            "solve_dual",
            "extract_velocity_sleep",
        ] {
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
            let wgsl = naga::back::wgsl::write_string(
                &module,
                &info,
                naga::back::wgsl::WriterFlags::empty(),
            )?;
            out.push((name.to_string(), wgsl));
        }
        Ok(out)
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
        let shaders = Self::compile_shaders()?;
        for (name, wgsl) in &shaders {
            let _ = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
            });
        }
        Ok(format!(
            "{} / {} ({} WGSL modules)",
            info.name,
            info.backend,
            shaders.len()
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
            "WebGPU full step shares the Metal phase graph; enable after M3 budgets".into(),
        ))
    }
    fn last_timings(&self) -> PhaseTimingsMs {
        self.last_timings.clone()
    }
    fn last_counters(&self) -> StepCounters {
        self.last_counters
    }
}
