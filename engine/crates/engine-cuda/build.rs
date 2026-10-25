//! The shared rigid plan's kernels (`engine_exec::rigid_plan::PERSISTENT_ENTRIES`),
//! Slang → CUDA C++ in the browser's `ENGINE_WGSL` layout, then → a fatbin with
//! nvcc when it is installed. Emits `cuda_kernels.rs`: per entry the fatbin
//! (empty without nvcc), thread-group size and `SLANG_globalParams` field →
//! BIND slot.
//!
//! `writes_indirect` marks the kernels whose body indexes `gIndirectArgs`: the
//! host reads binding 23 back only after one of them ran.
//!
//! Each fatbin holds SASS for `ENGINE_CUDA_SASS` (default `75,89`: T4, and the
//! Ada L4 / L40S / RTX 4090) so those GPUs load it without JIT, plus PTX for
//! `ENGINE_CUDA_ARCH` (default compute_75) that any newer GPU JIT-compiles.
//! Hosts without nvcc (macOS) still build and check the CUDA sources;
//! `ENGINE_CUDA_REQUIRE_PTX=1` turns a missing nvcc into a build error (CI,
//! Modal archives).

use anyhow::{anyhow, Result};
use engine_exec::rigid_plan::{kernel_source, PERSISTENT_ENTRIES};
use engine_slang_build::cuda_layout::{compile_kernel, deferred_load_hazards, fatbin, CudaKernel};
use engine_slang_build::msl_layout::bind_slots;
use engine_slang_build::{find_nvcc, find_slangc};
use std::env;
use std::fmt::Write;
use std::path::PathBuf;

/// (reflection, CUDA source bytes, fatbin built, writes binding 23)
type Compiled = Result<(CudaKernel, u64, bool, bool)>;

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let root = manifest.join("../../slang");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let dir = out_dir.join("cuda");
    std::fs::create_dir_all(&dir)?;
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=ENGINE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=ENGINE_CUDA_SASS");
    println!("cargo:rerun-if-env-changed=ENGINE_CUDA_LOAD_CACHE");
    println!("cargo:rerun-if-env-changed=ENGINE_CUDA_REQUIRE_PTX");

    let slangc = find_slangc()?;
    let nvcc = match find_nvcc() {
        Ok(p) => Some(p),
        Err(e) if env::var("ENGINE_CUDA_REQUIRE_PTX").as_deref() == Ok("1") => return Err(e),
        Err(_) => {
            println!("cargo:warning=nvcc not found: engine-cuda embeds CUDA sources without PTX (GPU tests skip)");
            None
        }
    };
    let arch = env::var("ENGINE_CUDA_ARCH").unwrap_or_else(|_| "compute_75".into());
    let sass_list = env::var("ENGINE_CUDA_SASS").unwrap_or_else(|_| "75,89".into());
    let sass: Vec<&str> = sass_list
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let kernels = root.join("kernels");
    let slots = bind_slots(&[
        kernels.join("world_step.slang"),
        kernels.join("convex_step.slang"),
        kernels.join("vehicle_step.slang"),
        kernels.join("character_step.slang"),
        root.join("physics/contact_storage.slang"),
    ])?;

    // world_step entries are large and nvcc is slow: compile in parallel.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut compiled: Vec<(usize, Compiled)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&entry) = PERSISTENT_ENTRIES.get(k) else {
                            break;
                        };
                        let cu = dir.join(format!("{entry}.cu"));
                        let r = compile_kernel(
                            &slangc,
                            &root,
                            &kernels.join(kernel_source(entry)),
                            entry,
                            &cu,
                        )
                        .and_then(|kernel| {
                            let source = std::fs::read_to_string(&cu)?;
                            // Slang keeps every global in the bank, so whether a
                            // kernel rewrites binding 23 comes from its body.
                            let writes_indirect = source.contains("gIndirectArgs_0)[");
                            let hazards = deferred_load_hazards(&source);
                            if !hazards.is_empty() {
                                anyhow::bail!(
                                    "{entry}: Slang's CUDA output reads a buffer element after storing to it \
                                     (use the value before the store in the .slang source):\n  {}",
                                    hazards.join("\n  ")
                                );
                            }
                            if let Some(nvcc) = &nvcc {
                                fatbin(nvcc, &cu, &cu.with_extension("fatbin"), &arch, &sass)?;
                            }
                            Ok((kernel, source.len() as u64, nvcc.is_some(), writes_indirect))
                        });
                        done.push((k, r));
                    }
                    done
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    });
    compiled.sort_by_key(|(k, _)| *k);

    let mut table = format!(
        "/// PTX virtual architecture in each fatbin (empty: no nvcc).\n\
         pub const PTX_ARCH: &str = {:?};\n\
         /// SASS architectures in each fatbin (`sm_XX`, no JIT on those GPUs).\n\
         pub const SASS_ARCHS: &[&str] = &{:?};\n\
         /// Rigid plan kernels: entry, CUDA source bytes, fatbin, threads, globals.\n\
         pub static CUDA_KERNELS: &[CudaKernel] = &[\n",
        if nvcc.is_some() { arch.as_str() } else { "" },
        if nvcc.is_some() {
            sass.clone()
        } else {
            Vec::new()
        }
    );
    for (k, r) in compiled {
        let entry = PERSISTENT_ENTRIES[k];
        let (kernel, cu_bytes, has_image, writes_indirect) = r?;
        let mut binds = String::new();
        let mut params_bytes = 0;
        for f in &kernel.fields {
            let slot = slots
                .get(&f.name)
                .ok_or_else(|| anyhow!("{entry}: global {} has no BIND slot", f.name))?;
            write!(binds, "({}, {}, {slot}), ", f.offset, f.size)?;
            params_bytes = params_bytes.max(f.offset + f.size);
        }
        let image = if has_image {
            format!(
                "include_bytes!({:?})",
                dir.join(format!("{entry}.fatbin")).display().to_string()
            )
        } else {
            "&[]".into()
        };
        writeln!(
            table,
            "    CudaKernel {{ entry: {entry:?}, cu_bytes: {cu_bytes}, image: {image}, threads: {:?}, params_bytes: {params_bytes}, writes_indirect: {writes_indirect}, binds: &[{binds}] }},",
            kernel.threads
        )?;
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("cuda_kernels.rs"), table)?;
    terrain_kernels(&slangc, nvcc.as_deref(), &arch, &sass, &root, &out_dir)?;
    Ok(())
}

/// `terrain_mpm_fast.slang`'s entries (the native fast paths, engine-metal
/// `TerrainCaps`).
const TERRAIN_FAST_ENTRIES: &[&str] = &[
    "mpm_p2g_cells",
    "mpm_p2g_nodes",
    "mpm_touch_keys",
    "mpm_count_keys",
    "mpm_scatter_keys",
];

/// Grid-stride entries of `terrain_mpm_cuda.slang`: each stands for the plan's
/// indirect dispatch of the kernel it names (without `_gs`).
const TERRAIN_GS_ENTRIES: &[&str] = &[
    "mpm_touch_gs",
    "mpm_count_gs",
    "mpm_scatter_gs",
    "mpm_payload_gs",
    "mpm_g2p_gs",
    "mpm_geostatic_gs",
    "mpm_source_gs",
    "mpm_bake_bin_gs",
    "mpm_parcels_gs",
    "mpm_cell_sort_gs",
    "mpm_p2g_gs",
    "mpm_grid_gs",
    "mpm_couple_apply_gs",
    "mpm_p2g_nodes_gs",
    "mpm_scan_cells_gs",
    "mpm_touch_keys_gs",
    "mpm_count_keys_gs",
    "mpm_scatter_keys_gs",
    "mpm_p2g_cells_gs",
];

/// The terrain plan's kernels (`engine_terrain::plan::KERNELS`, bindings
/// 37-51), the native fast paths (`terrain_mpm_fast.slang`) and the CUDA
/// grid-stride entries for the plan's indirect dispatches
/// (`terrain_mpm_cuda.slang`), Slang → CUDA with the same hazard check as the
/// rigid kernels, and fatbins when nvcc is present. Emits
/// `cuda_terrain_kernels.rs` (`CUDA_TERRAIN_KERNELS`). No host executor runs
/// them yet (engine/docs/deformable-terrain.md, "Native backends").
fn terrain_kernels(
    slangc: &std::path::Path,
    nvcc: Option<&std::path::Path>,
    arch: &str,
    sass: &[&str],
    root: &std::path::Path,
    out_dir: &std::path::Path,
) -> Result<()> {
    let dir = out_dir.join("cuda_terrain");
    std::fs::create_dir_all(&dir)?;
    let kernels = root.join("kernels");
    let mpm = kernels.join("terrain_mpm.slang");
    let fast = kernels.join("terrain_mpm_fast.slang");
    let gs = kernels.join("terrain_mpm_cuda.slang");
    let slots = bind_slots(&[
        kernels.join("world_step.slang"),
        root.join("physics/contact_storage.slang"),
        mpm.clone(),
        fast.clone(),
    ])?;
    let entries: Vec<(&str, &std::path::Path)> = engine_terrain::plan::KERNELS
        .iter()
        .map(|&e| (e, mpm.as_path()))
        .chain(TERRAIN_FAST_ENTRIES.iter().map(|&e| (e, fast.as_path())))
        .chain(TERRAIN_GS_ENTRIES.iter().map(|&e| (e, gs.as_path())))
        .collect();
    let compiled = engine_slang_build::par_map(&entries, |&(entry, source)| -> Compiled {
        let cu = dir.join(format!("{entry}.cu"));
        let kernel = compile_kernel(slangc, root, source, entry, &cu)?;
        let text = std::fs::read_to_string(&cu)?;
        let hazards = deferred_load_hazards(&text);
        if !hazards.is_empty() {
            anyhow::bail!(
                "{entry}: Slang's CUDA output reads a buffer element after storing to it \
                 (use the value before the store in the .slang source):\n  {}",
                hazards.join("\n  ")
            );
        }
        if let Some(nvcc) = nvcc {
            fatbin(nvcc, &cu, &cu.with_extension("fatbin"), arch, sass)?;
        }
        Ok((kernel, text.len() as u64, nvcc.is_some(), false))
    });
    let mut table = String::from(
        "/// Terrain kernels: plan, fast paths, grid-stride entries (see build.rs).\n\
         pub static CUDA_TERRAIN_KERNELS: &[CudaKernel] = &[\n",
    );
    for (&(entry, _), r) in entries.iter().zip(compiled) {
        let (kernel, cu_bytes, has_image, _) = r?;
        let mut binds = String::new();
        let mut params_bytes = 0;
        for f in &kernel.fields {
            let slot = slots
                .get(&f.name)
                .ok_or_else(|| anyhow!("{entry}: global {} has no BIND slot", f.name))?;
            write!(binds, "({}, {}, {slot}), ", f.offset, f.size)?;
            params_bytes = params_bytes.max(f.offset + f.size);
        }
        let image = if has_image {
            format!(
                "include_bytes!({:?})",
                dir.join(format!("{entry}.fatbin")).display().to_string()
            )
        } else {
            "&[]".into()
        };
        writeln!(
            table,
            "    CudaKernel {{ entry: {entry:?}, cu_bytes: {cu_bytes}, image: {image}, threads: {:?}, params_bytes: {params_bytes}, writes_indirect: false, binds: &[{binds}] }},",
            kernel.threads
        )?;
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("cuda_terrain_kernels.rs"), table)?;
    Ok(())
}
