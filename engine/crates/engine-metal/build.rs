use anyhow::Result;
use engine_slang_build::{compile, find_slangc, SlangTarget};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let msl_dir = out_dir.join("msl");
    std::fs::create_dir_all(&msl_dir)?;

    println!("cargo:rerun-if-changed={}", slang_root.display());
    let _ = find_slangc()?;

    let abi = slang_root.join("kernels/abi_probe.slang");
    compile(
        &abi,
        "abi_roundtrip",
        SlangTarget::Metal,
        &msl_dir.join("abi_roundtrip.metal"),
        &[],
    )?;

    let world = slang_root.join("kernels/world_step.slang");
    let compact = slang_root.join("kernels/compact_solve.slang");
    let vehicle = slang_root.join("kernels/vehicle_step.slang");
    let vehicle_entries = [
        "vehicle_input",
        "wheel_query",
        "vehicle_link",
        "vehicle_prepare",
        "vehicle_iterate",
        "vehicle_loads",
        "vehicle_post",
        "raycast_batch",
        "wheel_soil",
        "wheel_soil_press",
    ];
    let entries = [
        "apply_commands",
        "predict_bodies",
        "build_aabbs",
        "joint_hash_clear",
        "joint_hash_insert",
        "broadphase_clear",
        "broadphase_insert",
        "broadphase_query",
        "broadphase_grid",
        "narrowphase",
        "update_refs",
        "contact_hash_clear",
        "contact_hash_insert",
        "persist_contacts",
        "persist_manifolds",
        "persist_manifolds_compact",
        "warmstart_joints",
        "build_awake_list",
        "csr_clear",
        "csr_degree_contacts",
        "csr_degree_joints",
        "csr_prefix",
        "csr_prefix_blocks",
        "csr_prefix_block_sums",
        "csr_prefix_add",
        "csr_prefix_init",
        "csr_prefix_scan",
        "csr_prefix_swap",
        "csr_prefix_finalize",
        "csr_scatter_contacts",
        "csr_scatter_joints",
        "csr_sort",
        "csr_sort_append",
        "color_bodies",
        "init_colors",
        "pack_color_lists",
        "count_color_tiles",
        "prefix_color_tiles",
        "scatter_color_tiles",
        "write_color_args",
        "write_work_args",
        "write_sleep_args",
        "clear_solver_contacts",
        "pack_solver_contacts",
        "solve_primal_color_compact",
        "solve_primal_one_compact",
        "solve_dual_compact",
        "solve_primal_color",
        "solve_primal_one",
        "solve_dual",
        "sleep_island_gate",
        "sleep_island_commit",
        "extract_velocity_sleep",
        "recompute_inertia",
        "record_color_degrees",
        "reuse_all_manifolds",
        "convex_classify",
        "convex_bucket",
        "convex_write_args",
        "convex_sat_small",
        "convex_sat_large",
        "convex_manifold",
        "vehicle_input",
        "wheel_query",
        "vehicle_link",
        "vehicle_prepare",
        "vehicle_iterate",
        "vehicle_loads",
        "vehicle_post",
        "raycast_batch",
        "wheel_soil",
        "wheel_soil_press",
    ];
    let convex = slang_root.join("kernels/convex_step.slang");
    for entry in entries {
        compile(
            if vehicle_entries.contains(&entry) {
                &vehicle
            } else if entry.starts_with("convex_") {
                &convex
            } else if entry.ends_with("_compact")
                || (entry == "pack_solver_contacts" || entry == "clear_solver_contacts")
            {
                &compact
            } else {
                &world
            },
            entry,
            SlangTarget::Metal,
            &msl_dir.join(format!("{entry}.metal")),
            &[],
        )?;
    }

    // Also emit SPIR-V and CUDA artifacts for the port crates to consume from OUT_DIR copy.
    let spirv_dir = out_dir.join("spirv");
    let cuda_dir = out_dir.join("cuda");
    std::fs::create_dir_all(&spirv_dir)?;
    std::fs::create_dir_all(&cuda_dir)?;
    for entry in entries.iter().chain(["abi_roundtrip"].iter()) {
        let src = if *entry == "abi_roundtrip" {
            &abi
        } else if vehicle_entries.contains(entry) {
            &vehicle
        } else if entry.starts_with("convex_") {
            &convex
        } else if entry.ends_with("_compact")
            || (*entry == "pack_solver_contacts" || *entry == "clear_solver_contacts")
        {
            &compact
        } else {
            &world
        };
        let _ = compile(
            src,
            entry,
            SlangTarget::Spirv,
            &spirv_dir.join(format!("{entry}.spv")),
            &[],
        );
        let _ = compile(
            src,
            entry,
            SlangTarget::Cuda,
            &cuda_dir.join(format!("{entry}.cu")),
            &[],
        );
    }

    println!("cargo:rustc-env=ENGINE_MSL_DIR={}", msl_dir.display());
    rigid_kernels(&slang_root, &out_dir)?;
    terrain_kernels(&slang_root, &out_dir)?;
    Ok(())
}

/// The terrain plan's kernels (`engine_terrain::plan::KERNELS`, bindings
/// 37-51) and the look-ahead's (`terrain_watch.slang`, binding 52), in the
/// same `ENGINE_WGSL` layout as the rigid table, with each kernel's buffers
/// renumbered from 0 (Slang numbers them across the whole module, past
/// Metal's 31). Emits `terrain_kernels.rs`.
fn terrain_kernels(root: &std::path::Path, out_dir: &std::path::Path) -> Result<()> {
    use engine_slang_build::msl_layout::{
        bind_slots, compact_buffers, compile_kernel, msl_buffers,
    };
    use std::fmt::Write;
    let slangc = find_slangc()?;
    let dir = out_dir.join("terrain_msl");
    std::fs::create_dir_all(&dir)?;
    let kernels = root.join("kernels");
    let mpm = kernels.join("terrain_mpm.slang");
    let watch = kernels.join("terrain_watch.slang");
    let fast = kernels.join("terrain_mpm_fast.slang");
    let slots = bind_slots(&[
        kernels.join("world_step.slang"),
        root.join("physics/contact_storage.slang"),
        mpm.clone(),
        watch.clone(),
        fast.clone(),
    ])?;
    // (table name, Slang entry, source): a fast path may be a plan kernel
    // compiled from terrain_mpm_fast.slang under a name of its own.
    let entries: Vec<(&str, &str, &std::path::Path)> = engine_terrain::plan::KERNELS
        .iter()
        .map(|&e| (e, e, mpm.as_path()))
        .chain(
            TERRAIN_WATCH_ENTRIES
                .iter()
                .map(|&e| (e, e, watch.as_path())),
        )
        .chain(
            TERRAIN_FAST_ENTRIES
                .iter()
                .map(|&(n, e)| (n, e, fast.as_path())),
        )
        .collect();
    let compiled = engine_slang_build::par_map(&entries, |&(name, entry, source)| {
        let dest = dir.join(format!("{name}.metal"));
        let r = compile_kernel(&slangc, root, source, entry, &dest).and_then(|(msl, threads)| {
            let mut msl = compact_buffers(&msl, entry, "[[kernel]]")?;
            if name != entry {
                let from = format!("[[kernel]] void {entry}(");
                anyhow::ensure!(msl.contains(&from), "{entry}: no kernel signature");
                msl = msl.replace(&from, &format!("[[kernel]] void {name}("));
            }
            std::fs::write(&dest, &msl)?;
            Ok((msl, threads))
        });
        (dest, r)
    });
    let mut table = String::from(
        "/// Terrain kernels: (entry, MSL, threads, [(metal buffer, host slot)])\n\
         pub static TERRAIN_KERNELS: &[RigidKernel] = &[\n",
    );
    for (&(entry, _, _), (dest, r)) in entries.iter().zip(compiled) {
        let (msl, threads) = r?;
        let mut binds = String::new();
        for (global, index) in msl_buffers(&msl, entry, "[[kernel]]")? {
            let slot = slots
                .get(&global)
                .ok_or_else(|| anyhow::anyhow!("{entry}: global {global} has no BIND slot"))?;
            write!(binds, "({index}, {slot}), ")?;
        }
        writeln!(
            table,
            "    RigidKernel {{ entry: {entry:?}, msl: include_str!({:?}), threads: {threads:?}, binds: &[{binds}] }},",
            dest.display().to_string()
        )?;
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("terrain_kernels.rs"), table)?;
    Ok(())
}

/// `terrain_mpm_fast.slang`'s entries (native fast paths behind `TerrainCaps`).
/// (table name, Slang entry).
const TERRAIN_FAST_ENTRIES: &[(&str, &str)] = &[
    ("mpm_p2g_cells", "mpm_p2g_cells"),
    ("mpm_p2g_nodes", "mpm_p2g_nodes"),
    ("mpm_touch_keys", "mpm_touch_keys"),
    ("mpm_count_keys", "mpm_count_keys"),
    ("mpm_scatter_keys", "mpm_scatter_keys"),
];

/// `terrain_watch.slang`'s entries (the look-ahead after the rigid step).
const TERRAIN_WATCH_ENTRIES: &[&str] = &[
    "terrain_watch_energy",
    "terrain_watch_pick",
    "terrain_watch_write",
];

/// The shared rigid step graph's kernels (`engine_exec::rigid_plan`), compiled
/// with the browser's `ENGINE_WGSL` buffer layout so `MetalSim` replays exactly
/// the commands the WebGPU host does. Emits `rigid_kernels.rs`.
fn rigid_kernels(root: &std::path::Path, out_dir: &std::path::Path) -> Result<()> {
    use engine_exec::rigid_plan::{kernel_source, PERSISTENT_ENTRIES};
    use engine_slang_build::msl_layout::{bind_slots, compile_kernel, msl_buffers};
    use std::fmt::Write;
    let slangc = find_slangc()?;
    let dir = out_dir.join("rigid_msl");
    std::fs::create_dir_all(&dir)?;
    let kernels = root.join("kernels");
    let slots = bind_slots(&[
        kernels.join("world_step.slang"),
        kernels.join("convex_step.slang"),
        kernels.join("vehicle_step.slang"),
        kernels.join("character_step.slang"),
        root.join("physics/contact_storage.slang"),
    ])?;
    // The world_step entries are large: compile them in parallel.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut compiled: Vec<_> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&entry) = PERSISTENT_ENTRIES.get(k) else {
                            break;
                        };
                        let dest = dir.join(format!("{entry}.metal"));
                        let source = kernels.join(kernel_source(entry));
                        done.push((
                            k,
                            dest.clone(),
                            compile_kernel(&slangc, root, &source, entry, &dest),
                        ));
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
    compiled.sort_by_key(|(k, _, _)| *k);
    let mut table = String::from(
        "/// Rigid plan kernels: (entry, MSL, threads, [(metal buffer, host slot)])\n\
         pub static RIGID_KERNELS: &[RigidKernel] = &[\n",
    );
    for (k, dest, r) in compiled {
        let entry = PERSISTENT_ENTRIES[k];
        let (msl, threads) = r?;
        let mut binds = String::new();
        for (global, index) in msl_buffers(&msl, entry, "[[kernel]]")? {
            let slot = slots
                .get(&global)
                .ok_or_else(|| anyhow::anyhow!("{entry}: global {global} has no BIND slot"))?;
            write!(binds, "({index}, {slot}), ")?;
        }
        writeln!(
            table,
            "    RigidKernel {{ entry: {entry:?}, msl: include_str!({:?}), threads: {threads:?}, binds: &[{binds}] }},",
            dest.display().to_string()
        )?;
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("rigid_kernels.rs"), table)?;
    Ok(())
}
