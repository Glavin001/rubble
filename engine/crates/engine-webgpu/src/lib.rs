//! WebGPU backend. Default: Slang → WGSL directly at build time (`-DENGINE_WGSL`, see
//! `slang/physics/atomics.slang`). `WG_NAGA=1` selects the older Slang → SPIR-V → naga path.
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

mod direct {
    include!(concat!(env!("OUT_DIR"), "/direct_wgsl.rs"));
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

    /// WGSL for every entry: `(name, wgsl)` or `(name, err)`.
    pub fn compile_shaders_report() -> Vec<(String, Result<String, String>)> {
        if std::env::var_os("WG_NAGA").is_some() {
            return Self::naga_shaders_report();
        }
        Self::direct_shaders()
            .iter()
            .map(|(n, w)| {
                (
                    (*n).to_string(),
                    Ok(uniformize_workgroup_lanes(&cap_loop_iterations(w))),
                )
            })
            .collect()
    }

    /// Build-time Slang → WGSL modules (embedded; works on wasm32).
    pub fn direct_shaders() -> &'static [(&'static str, &'static str)] {
        direct::DIRECT_WGSL
    }

    /// Translate every SPIR-V entry to WGSL through naga.
    pub fn naga_shaders_report() -> Vec<(String, Result<String, String>)> {
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
                let wgsl = naga::back::wgsl::write_string(
                    &module,
                    &info,
                    naga::back::wgsl::WriterFlags::empty(),
                )?;
                // naga puts the for-loop test `break` inside `switch 0i`, so it
                // exits the switch and the loop increments forever (GPU hang).
                Ok(cap_loop_iterations(&fix_naga_loop_exits(&wgsl)))
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

/// `loop { switch 0i { default { if (i < n) {} else { break; } ... } } i++; continue; }`
/// never terminates: that `break` leaves the switch. Turn the first else-break into a
/// real loop exit.
fn fix_naga_loop_exits(src: &str) -> String {
    let mut sites: Vec<(usize, usize, usize)> = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = src[cursor..].find("switch 0i") {
        let switch_at = cursor + rel;
        let before = src[..switch_at].trim_end();
        if !before.ends_with("loop {") {
            cursor = switch_at + "switch 0i".len();
            continue;
        }
        let Some(open_rel) = src[switch_at..].find('{') else { break };
        let switch_open = switch_at + open_rel;
        let Some(switch_close) = matching_brace(src, switch_open) else { break };
        let default_at = match src[switch_at..switch_close].find("default:") {
            Some(p) => switch_at + p,
            None => {
                cursor = switch_close;
                continue;
            }
        };
        let Some(def_rel) = src[default_at..].find('{') else { break };
        let def_open = default_at + def_rel;
        let Some(def_close) = matching_brace(src, def_open) else { break };
        let body = &src[def_open + 1..def_close];
        let Some(br) = first_else_break(body) else {
            cursor = switch_close;
            continue;
        };
        let loop_open = before.len() - 1; // the '{' of `loop {`
        sites.push((loop_open, def_open + 1 + br, switch_close));
        cursor = def_open + 1;
    }
    let mut edits: Vec<(usize, String)> = Vec::new();
    for (i, &(loop_open, break_at, switch_close)) in sites.iter().enumerate() {
        let flag = format!("nagaExit{i}");
        edits.push((switch_close + 1, format!("\n if ({flag}) {{ break; }}\n")));
        edits.push((break_at, format!("{flag} = true; ")));
        edits.push((loop_open + 1, format!("\n var {flag} = false;\n")));
    }
    edits.sort_by(|a, b| b.0.cmp(&a.0));
    let mut s = src.to_string();
    for (at, text) in edits {
        s.insert_str(at, &text);
    }
    s
}

/// Hard cap on every WGSL loop. A non-terminating kernel on Apple Silicon can stall the
/// window server and take the whole machine down, so no loop may run unbounded.
pub const WGSL_LOOP_CAP: u32 = 1 << 16;

/// Chrome treats a storage/workgroup load as non-uniform, so a later barrier
/// inside `if (lanes == 1)` is rejected. `workgroupUniformLoad` makes it uniform.
fn uniformize_workgroup_lanes(src: &str) -> String {
    const NAME: &str = "gsUniformLanes_0";
    if !src.contains(NAME) {
        return src.to_string();
    }
    let mut out = String::with_capacity(src.len() + 256);
    for line in src.lines() {
        if line.contains("var<workgroup>") && line.contains(NAME) {
            out.push_str(line);
        } else {
            out.push_str(
                &line
                    .replace(NAME, &format!("workgroupUniformLoad(&{NAME})"))
                    .replace(
                        "workgroupUniformLoad(&gsUniformLanes_0) =",
                        "gsUniformLanes_0 =",
                    ),
            );
        }
        out.push('\n');
    }
    out
}

fn cap_loop_iterations(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 8192);
    let mut last = 0usize;
    let mut i = 0usize;
    let mut k = 0usize;
    while let Some(rel) = src[i..].find("loop") {
        let at = i + rel;
        let after = at + 4;
        let word = (at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_'))
            && (after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_'));
        let mut j = after;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if word && j < b.len() && b[j] == b'{' {
            out.push_str(&src[last..at]);
            out.push_str(&format!(
                "var loopGuard{k}: u32 = 0u;\nloop {{\n loopGuard{k} += 1u; if (loopGuard{k} > {WGSL_LOOP_CAP}u) {{ break; }}\n"
            ));
            k += 1;
            last = j + 1;
            i = j + 1;
        } else {
            i = after;
        }
    }
    out.push_str(&src[last..]);
    out
}

fn first_else_break(body: &str) -> Option<usize> {
    let mut i = 0;
    let b = body.as_bytes();
    while i + 4 < b.len() {
        if b[i..].starts_with(b"else") && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            let mut j = i + 4;
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'{' {
                j += 1;
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                if b[j..].starts_with(b"break") {
                    return Some(j);
                }
            }
        }
        i += 1;
    }
    None
}

fn matching_brace(src: &str, open: usize) -> Option<usize> {
    let b = src.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_wgsl_loop_is_capped_and_valid() {
        for (name, res) in WebGpuWorld::naga_shaders_report() {
            let wgsl = res.unwrap_or_else(|e| panic!("{name}: {e}"));
            let loops = wgsl.matches("loop {").count();
            let guards = wgsl.matches("loopGuard").count() / 3;
            assert_eq!(loops, guards, "{name}: uncapped loop");
            let module = naga::front::wgsl::parse_str(&wgsl)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&wgsl)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn direct_wgsl_covers_every_entry_and_validates() {
        let direct = WebGpuWorld::direct_shaders();
        for name in WebGpuWorld::entry_names().iter().filter(|n| **n != "abi_roundtrip") {
            assert!(
                direct.iter().any(|(n, _)| n == name),
                "{name}: missing direct WGSL"
            );
        }
        for (name, res) in WebGpuWorld::compile_shaders_report() {
            let wgsl = &res.unwrap();
            let loops = direct.iter().find(|(n, _)| *n == name).unwrap().1;
            let raw = loops.matches("loop").count();
            assert_eq!(raw, wgsl.matches("loopGuard").count() / 3, "{name}: uncapped loop");
            let module = naga::front::wgsl::parse_str(wgsl)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(wgsl)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
