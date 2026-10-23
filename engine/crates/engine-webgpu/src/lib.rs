//! WebGPU backend. Default: Slang → WGSL directly at build time (`-DENGINE_WGSL`, see
//! `slang/physics/atomics.slang`). `WG_NAGA=1` selects the older Slang → SPIR-V → naga path.
//!
//! **Fair speed bar vs three-avbd:** same scenes in **Chrome WebGPU** on this machine
//! (not native Metal `scene_fps` vs their browser `gpuTotal`). Metal remains the M3
//! interactive gate; this crate is the apples-to-apples oracle path.

mod gpu;
mod soft_skin;
pub use gpu::{default_device, CharacterHandle, MotionEvent, SimSnapshot, SoftEvents, WgpuSim};
pub use soft_skin::{SoftSkinner, SOFT_SKIN_WGSL};

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
                    Ok(uniformize_workgroup_lanes(&cap_loop_iterations(&if n
                        .starts_with("solve_primal_sparse")
                        || n.starts_with("soft_primal_fused")
                        || n.starts_with("soft_primal_bodies")
                        || n.starts_with("color_bodies_local")
                        || n.starts_with("color_bodies_tail")
                    {
                        // These entries access buffers only. Slang over-lowers
                        // DeviceMemoryBarrierWithGroupSync to a texture barrier
                        // unavailable in the supported WGSL/Naga version, and
                        // ends it with an empty statement (`;;`) that Tint
                        // accepts but Safari's WGSL parser rejects.
                        w.replace(
                            "storageBarrier(); textureBarrier(); workgroupBarrier();;",
                            "storageBarrier(); workgroupBarrier();",
                        )
                        .replace(
                            "storageBarrier(); textureBarrier(); workgroupBarrier();",
                            "storageBarrier(); workgroupBarrier();",
                        )
                    } else {
                        w.to_string()
                    }))),
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
            out.push(((*name).to_string(), result.map_err(|e| format!("{e:#}"))));
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
        let (info, device, _queue) = crate::gpu::default_device()?;

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
        let Some(open_rel) = src[switch_at..].find('{') else {
            break;
        };
        let switch_open = switch_at + open_rel;
        let Some(switch_close) = matching_brace(src, switch_open) else {
            break;
        };
        let default_at = match src[switch_at..switch_close].find("default:") {
            Some(p) => switch_at + p,
            None => {
                cursor = switch_close;
                continue;
            }
        };
        let Some(def_rel) = src[default_at..].find('{') else {
            break;
        };
        let def_open = default_at + def_rel;
        let Some(def_close) = matching_brace(src, def_open) else {
            break;
        };
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
                    // Restore stores only: `= ` must not also match a `==` read.
                    .replace(
                        "workgroupUniformLoad(&gsUniformLanes_0) = ",
                        "gsUniformLanes_0 = ",
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
        for name in WebGpuWorld::entry_names()
            .iter()
            .filter(|n| **n != "abi_roundtrip")
        {
            assert!(
                direct.iter().any(|(n, _)| n == name),
                "{name}: missing direct WGSL"
            );
        }
        for (name, res) in WebGpuWorld::compile_shaders_report() {
            let wgsl = &res.unwrap();
            let loops = direct.iter().find(|(n, _)| *n == name).unwrap().1;
            let raw = loops.matches("loop").count();
            assert_eq!(
                raw,
                wgsl.matches("loopGuard").count() / 3,
                "{name}: uncapped loop"
            );
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

    /// Chrome exposes 10 storage buffers per compute stage. Every entry must fit,
    /// which is why convex work is split out of `narrowphase`.
    #[test]
    fn every_entry_fits_the_webgpu_storage_buffer_budget() {
        for (name, res) in WebGpuWorld::compile_shaders_report() {
            let wgsl = res.unwrap();
            let storage = wgsl.matches("var<storage").count();
            assert!(storage <= 10, "{name}: {storage} storage buffers > 10");
        }
    }

    #[test]
    fn compact_primal_reads_64_byte_points_without_canonical_contact_binding() {
        for name in ["solve_primal_color_compact", "solve_primal_one_compact"] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("compact entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let globals: Vec<_> = module.global_variables.iter().map(|(_, g)| g).collect();
            assert!(!globals.iter().any(|g| g.binding.as_ref().is_some_and(|b| b.binding == 4)), "{name}: compact primal still reads canonical 128-byte contacts; hot-path traffic reduction was not compiled");
            let points = globals
                .iter()
                .find(|g| g.binding.as_ref().is_some_and(|b| b.binding == 27))
                .expect("compact point binding missing");
            let naga::TypeInner::Array { base, stride, .. } = module.types[points.ty].inner else {
                panic!("compact points must be an array");
            };
            assert_eq!(stride,64,"{name}: solver point stride changed; host allocations and bandwidth contract require 64 bytes");
            let naga::TypeInner::Struct { ref members, span } = module.types[base].inner else {
                panic!("compact points must be structs");
            };
            assert_eq!(span, 64);
            assert_eq!(
                members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                vec![0, 12, 16, 28, 32, 44, 48, 60],
                "{name}: compact field offsets differ from the packing ABI"
            );
            assert!(
                globals
                    .iter()
                    .filter(|g| matches!(g.space, naga::AddressSpace::Storage { .. }))
                    .count()
                    <= 10,
                "{name}: exceeds browser storage-buffer budget"
            );
        }
    }

    #[test]
    fn compact_solvers_use_cached_body_pairs_without_collider_indirection() {
        for name in [
            "solve_primal_color_compact",
            "solve_primal_one_compact",
            "solve_dual_compact",
        ] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("compact entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let globals: Vec<_> = module.global_variables.iter().map(|(_, g)| g).collect();
            for (binding, resource) in [(2, "colliders"), (25, "canonical manifolds")] {
                assert!(!globals.iter().any(|g| g.binding.as_ref().is_some_and(|b| b.binding == binding)),
                    "{name}: still reads {resource}; cached solver pairs must remove collider-to-body lookups from every solver iteration");
            }
            let pairs = globals
                .iter()
                .find(|g| g.binding.as_ref().is_some_and(|b| b.binding == 28))
                .expect("compact solver pair binding missing");
            let naga::TypeInner::Array { base, stride, .. } = module.types[pairs.ty].inner else {
                panic!("{name}: solver pairs must be an array");
            };
            assert_eq!(
                stride, 32,
                "{name}: solver pair stride differs from host allocation"
            );
            let naga::TypeInner::Struct { ref members, span } = module.types[base].inner else {
                panic!("{name}: solver pairs must be structs");
            };
            assert_eq!(span, 32);
            assert_eq!(members.iter().map(|m| m.offset).collect::<Vec<_>>(), vec![0,4,8,12,16,28],
                "{name}: solver pair body IDs, point range, sphere flag or normal/friction offsets changed");
        }
    }

    #[test]
    fn persistent_contact_kernels_keep_hot_cold_layouts_and_omit_copyback() {
        for name in [
            "narrowphase_persistent",
            "contact_hash_insert_persistent",
            "contact_hash_insert_manifolds_persistent",
            "persist_contacts_persistent",
            "persist_manifolds_persistent",
            "solve_primal_color_persistent",
            "solve_dual_persistent",
        ] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("persistent entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let globals: Vec<_> = module.global_variables.iter().map(|(_, g)| g).collect();
            let solver = name.starts_with("solve_");
            for (binding, expected_stride) in [(4, 32), (27, 64)] {
                let view = globals
                    .iter()
                    .find(|g| g.binding.as_ref().is_some_and(|b| b.binding == binding));
                if solver && binding == 4 {
                    assert!(view.is_none(), "{name}: solver still accesses cold contact metadata/canonical copyback; persistent forces must remain in hot storage");
                    continue;
                }
                let view =
                    view.unwrap_or_else(|| panic!("{name}: missing contact binding {binding}"));
                let naga::TypeInner::Array { base, stride, .. } = module.types[view.ty].inner
                else {
                    panic!("{name}: contact storage must be an array")
                };
                assert_eq!(
                    stride, expected_stride,
                    "{name}: binding {binding} stride disagrees with host allocation"
                );
                if binding == 4 {
                    let naga::TypeInner::Struct { ref members, .. } = module.types[base].inner
                    else {
                        panic!("metadata must be a struct")
                    };
                    assert_eq!(
                        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                        vec![0, 4, 8, 12, 16, 28]
                    );
                    for field in [2, 3] {
                        assert!(matches!(module.types[members[field].ty].inner,naga::TypeInner::Scalar(naga::Scalar {kind:naga::ScalarKind::Uint,width:4})),
                            "{name}: feature/generation field {field} must retain all 32 unsigned bits independently of owner/reuse/stick flags");
                    }
                }
            }
            assert!(
                globals
                    .iter()
                    .filter(|g| matches!(g.space, naga::AddressSpace::Storage { .. }))
                    .count()
                    <= 10,
                "{name}: persistent layout exceeds the browser's ten storage-buffer budget"
            );
        }
    }

    #[test]
    fn collision_clipping_uses_pointer_arguments_and_eight_vertex_scratch() {
        for name in ["narrowphase"] {
            let source = WebGpuWorld::compile_shaders_report()
                .into_iter()
                .find(|(n, _)| n == name)
                .expect("collision entry must be exported")
                .1
                .unwrap();
            let module = naga::front::wgsl::parse_str(&source).expect("valid collision WGSL");
            let clip = module
                .functions
                .iter()
                .find(|(_, f)| {
                    f.name
                        .as_deref()
                        .is_some_and(|n| n.starts_with("boxClipPoly"))
                })
                .expect("clipping function must be present")
                .1;
            for index in [0, 4] {
                let naga::TypeInner::Pointer {
                    base,
                    space: naga::AddressSpace::Function,
                } = module.types[clip.arguments[index].ty].inner
                else {
                    panic!("{name} clip argument {index}: array passed by value; four clipping planes must share local scratch without copying whole polygons");
                };
                let naga::TypeInner::Array {
                    size: naga::ArraySize::Constant(size),
                    ..
                } = module.types[base].inner
                else {
                    panic!("{name}: clipping scratch must be a fixed vertex array");
                };
                assert_eq!(size.get(), 8,
                    "{name}: four half-plane clips of a convex quad need exactly eight slots; smaller drops support points, larger increases shader scratch");
            }
        }
    }

    #[test]
    fn production_contact_reuse_has_no_shared_diagnostic_counter_buffer() {
        for name in ["update_refs", "narrowphase"] {
            let shaders = WebGpuWorld::compile_shaders_report();
            let source = shaders
                .iter()
                .find(|(n, _)| n == name)
                .unwrap()
                .1
                .as_ref()
                .unwrap();
            let module = naga::front::wgsl::parse_str(source).expect("valid reuse shader");
            assert!(!module.global_variables.iter().any(|(_, global)|
                global.binding.as_ref().is_some_and(|b| b.group == 0 && b.binding == 21)),
                "{name}: production contact reuse binds the shared scratch/statistics buffer. The removed hit/miss/generation diagnostics serialized every body/pair on six global atomic counters; keep optional instrumentation out of the default performance path.");
        }
    }

    #[test]
    fn exported_wgsl_has_no_empty_statements() {
        // Safari's WGSL parser rejects `;;` ("Not a valid statement"); Tint accepts it.
        for (name, result) in WebGpuWorld::compile_shaders_report() {
            let source = result.unwrap();
            assert!(
                !source.contains(";;"),
                "{name}: empty statement `;;` breaks Safari"
            );
        }
    }

    #[test]
    fn sparse_color_barrier_lowering_preserves_buffer_visibility_and_has_no_textures() {
        // Slang emits a texture barrier alongside storage/workgroup barriers.
        // Removing it is legal ONLY while these entry points access no textures.
        for (name, result) in WebGpuWorld::compile_shaders_report() {
            if !name.starts_with("solve_primal_sparse") {
                continue;
            }
            let source = result.unwrap();
            let module = naga::front::wgsl::parse_str(&source).expect("valid sparse shader");
            assert!(!module.types.iter().any(|(_,ty)| matches!(ty.inner,
                naga::TypeInner::Image{..} | naga::TypeInner::Sampler{..})),
                "{name}: texture resources invalidate the buffer-only barrier lowering; retain the appropriate texture synchronization before adding textures");
            assert!(source.contains("storageBarrier(); workgroupBarrier();"),
                "{name}: later colors depend on earlier pose writes; preserve storage visibility AND workgroup execution synchronization");
            assert!(
                !source.contains("textureBarrier"),
                "{name}: unsupported Slang texture barrier escaped lowering"
            );
        }
    }

    #[test]
    fn primal_topology_uses_plain_loads_and_writes_only_pose_fields() {
        fn stores(block: &naga::Block, out: &mut Vec<naga::Handle<naga::Expression>>) {
            for statement in block.iter() {
                match statement {
                    naga::Statement::Store { pointer, .. } => out.push(*pointer),
                    naga::Statement::Block(b) => stores(b, out),
                    naga::Statement::If { accept, reject, .. } => {
                        stores(accept, out);
                        stores(reject, out);
                    }
                    naga::Statement::Switch { cases, .. } => {
                        for case in cases {
                            stores(&case.body, out);
                        }
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => {
                        stores(body, out);
                        stores(continuing, out);
                    }
                    _ => {}
                }
            }
        }
        for (name, source) in WebGpuWorld::compile_shaders_report()
            .into_iter()
            .filter(|(name, _)| name.starts_with("solve_primal_"))
        {
            let source = source.unwrap();
            let module = naga::front::wgsl::parse_str(&source).expect("valid primal WGSL");
            for binding in if name.starts_with("solve_primal_color") {
                &[14, 17][..]
            } else {
                &[14][..]
            } {
                let views: Vec<_> = module
                    .global_variables
                    .iter()
                    .filter(|(_, g)| {
                        g.binding
                            .as_ref()
                            .is_some_and(|b| b.group == 0 && b.binding == *binding)
                    })
                    .collect();
                assert_eq!(views.len(), 1, "{name}: binding {binding} must retain exactly one topology view after entry-point elimination");
                let naga::TypeInner::Array { base, .. } = module.types[views[0].1.ty].inner else {
                    panic!("topology must remain a u32 array");
                };
                assert!(matches!(module.types[base].inner, naga::TypeInner::Scalar(_)),
                    "{name}: binding {binding} still uses atomic topology loads inside every solver iteration; construction has finished before primal dispatch");
            }
            let mut pose_stores = 0;
            for function in module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
            {
                let mut pointers = Vec::new();
                stores(&function.body, &mut pointers);
                for mut pointer in pointers {
                    let mut member = None;
                    loop {
                        pointer = match function.expressions[pointer] {
                            naga::Expression::AccessIndex { base, index } => {
                                member = Some(index);
                                base
                            }
                            naga::Expression::Access { base, .. } => base,
                            _ => break,
                        };
                    }
                    let naga::Expression::GlobalVariable(g) = function.expressions[pointer] else {
                        continue;
                    };
                    let global = &module.global_variables[g];
                    if !global
                        .binding
                        .as_ref()
                        .is_some_and(|b| b.group == 0 && b.binding == 0)
                    {
                        continue;
                    }
                    let naga::TypeInner::Array { base, .. } = module.types[global.ty].inner else {
                        panic!("body array ABI");
                    };
                    let naga::TypeInner::Struct { ref members, .. } = module.types[base].inner
                    else {
                        panic!("body struct ABI");
                    };
                    let field = member.map(|i| &members[i as usize]);
                    assert!(field.is_some_and(|f| f.offset == 0 || f.offset == 16),
                        "{name}: primal writes non-pose body field {:?}; full 112-byte BodyState stores waste bandwidth and risk overwriting unrelated state", field.and_then(|f| f.name.as_ref()));
                    pose_stores += 1;
                }
            }
            assert_eq!(
                pose_stores, 2,
                "{name}: publish exactly position and orientation after the Newton solve"
            );
        }
    }

    #[test]
    fn count_only_kernels_do_not_load_unrelated_atomic_counters() {
        // These kernels need immutable body/collider counts; reparent commands
        // additionally read the contact count in their invalidation branch, and the
        // awake-list per-body passes (PASS_U == 1) bound threads by the awake count
        // in place of bodyCount: one read of a counter no thread of that dispatch
        // writes, the same as every other awake-list kernel. Loading a
        // whole StepCounters snapshot emits ten atomic reads for every body on
        // WGSL, including counters being incremented by neighbouring threads.
        // Check parsed IR rather than generated identifiers or text formatting.
        let entries = [
            "apply_commands",
            "build_aabbs",
            "broadphase_insert",
            "broadphase_query",
            "build_awake_list",
            "predict_bodies",
            "extract_velocity_sleep",
            "record_stab_velocity",
            "sleep_island_commit",
            "recompute_inertia",
        ];
        let shaders = WebGpuWorld::compile_shaders_report();
        for name in entries {
            let source = shaders
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing count-access regression kernel {name}"))
                .1
                .as_ref()
                .unwrap();
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
            let mut unexpected = Vec::new();
            let mut atomic_offsets = Vec::new();
            for function in module
                .functions
                .iter()
                .map(|(_, f)| f)
                .chain(module.entry_points.iter().map(|e| &e.function))
            {
                for (_, expr) in function.expressions.iter() {
                    let naga::Expression::Load { pointer } = expr else {
                        continue;
                    };
                    let naga::Expression::AccessIndex { index: member, .. } =
                        function.expressions[*pointer]
                    else {
                        continue;
                    };
                    let mut root = *pointer;
                    loop {
                        root = match function.expressions[root] {
                            naga::Expression::AccessIndex { base, .. }
                            | naga::Expression::Access { base, .. } => base,
                            _ => break,
                        };
                    }
                    let naga::Expression::GlobalVariable(global) = function.expressions[root]
                    else {
                        continue;
                    };
                    let global = &module.global_variables[global];
                    if !global
                        .binding
                        .as_ref()
                        .is_some_and(|b| b.group == 0 && b.binding == 7)
                    {
                        continue;
                    }
                    let naga::TypeInner::Array { base, .. } = module.types[global.ty].inner else {
                        panic!("{name}: counter buffer ABI is no longer an array");
                    };
                    let naga::TypeInner::Struct { ref members, .. } = module.types[base].inner
                    else {
                        panic!("{name}: counter buffer element is no longer a struct");
                    };
                    let field = &members[member as usize];
                    if matches!(module.types[field.ty].inner, naga::TypeInner::Atomic(_)) {
                        atomic_offsets.push(field.offset);
                        unexpected.push(format!(
                            "{} at byte {} in {}",
                            field.name.as_deref().unwrap_or("unnamed counter"),
                            field.offset,
                            function.name.as_deref().unwrap_or("entry point")
                        ));
                    }
                }
            }
            atomic_offsets.sort_unstable();
            let expected = match name {
                "apply_commands" => vec![8],
                "predict_bodies"
                | "extract_velocity_sleep"
                | "record_stab_velocity"
                | "sleep_island_commit" => vec![20],
                _ => vec![],
            };
            assert_eq!(atomic_offsets, expected,
                "{name}: unrelated atomic counter reads per invocation: {unexpected:?}. Read bodyCount/colliderCount directly; a full counter snapshot can cause GPU contention at 100k–506k scale. Atomic increments remain allowed.");
        }
    }
}
