//! Compile the fracture demo's Slang kernels + draw shader to MSL and emit a binding table.
//!
//! The kernels are built with the same `ENGINE_WGSL` layout the browser fracture demo
//! validated (host uniforms, prev bodies in `gBodies[maxBodies..]`, CSR index in
//! `gConstraintIds[csrBase..]`), so one host buffer set mirrors `simulation.mjs`.
//! Slang assigns Metal `[[buffer(n)]]` per entry, so each entry's signature is parsed
//! and mapped back to the shared `BIND(slot)` numbering.

use anyhow::{anyhow, bail, Context, Result};
use engine_slang_build::{compile_stage, compile_with_defines, find_slangc, SlangTarget};
use std::collections::HashMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

/// Dispatch names used by the host, in `fracture/simulation.mjs` order.
const KERNELS: &[&str] = &[
    "build_aabbs",
    "broadphase_clear",
    "broadphase_insert",
    "broadphase_query",
    "update_refs",
    "prepare_candidates",
    "write_sat_args",
    "sat",
    "narrowphase",
    "contact_hash_clear",
    "contact_hash_insert",
    "persist_contacts",
    "build_awake_list",
    "csr_clear",
    "csr_degree_contacts",
    "csr_prefix",
    "csr_prefix_init",
    "csr_prefix_scan",
    "csr_prefix_finalize",
    "csr_prefix_swap",
    "csr_scatter_contacts",
    "csr_sort",
    "init_colors",
    "color_bodies",
    "pack_color_lists",
    "write_color_args",
    "write_work_args",
    "predict_bodies",
    "solve_primal_color",
    "solve_dual",
    "extract_velocity_sleep",
    "sleep_island_gate",
    "sleep_island_commit",
];
/// Names served by the opt-in convex kernels (`fracture_<name>` in fracture_step.slang).
const FRACTURE: &[&str] = &[
    "prepare_candidates",
    "write_sat_args",
    "sat",
    "narrowphase",
    "broadphase_query",
    "broadphase_insert",
    "build_aabbs",
];

/// `BIND(n) <type> gName;` → slot n, from the kernel sources.
fn bind_slots(sources: &[&Path]) -> Result<HashMap<String, u32>> {
    let mut map = HashMap::new();
    for path in sources {
        for line in std::fs::read_to_string(path)?.lines() {
            let Some(rest) = line.trim_start().strip_prefix("BIND(") else {
                continue;
            };
            let Some((n, decl)) = rest.split_once(')') else {
                continue;
            };
            let Some(decl) = decl.split(';').next() else {
                continue;
            };
            let Some(name) = decl.split_whitespace().last() else {
                continue;
            };
            map.insert(name.to_string(), n.trim().parse()?);
        }
    }
    Ok(map)
}

/// `[numthreads(x, y, z)]` directly preceding `void <entry>(`.
fn numthreads(source: &str, entry: &str) -> Result<[u32; 3]> {
    let lines: Vec<&str> = source.lines().collect();
    let at = lines
        .iter()
        .position(|l| {
            let l = l.trim_start();
            l.starts_with(&format!("void {entry}(")) || l.starts_with(&format!("void {entry} ("))
        })
        .ok_or_else(|| anyhow!("entry {entry} not found"))?;
    for l in lines[..at].iter().rev().take(3) {
        if let Some(rest) = l.trim().strip_prefix("[numthreads(") {
            let nums: Vec<u32> = rest
                .trim_end_matches(")]")
                .split(',')
                .map(|s| s.trim().parse())
                .collect::<Result<_, _>>()?;
            return Ok([nums[0], nums[1], nums[2]]);
        }
    }
    bail!("no [numthreads] for {entry}")
}

/// `(global name, Metal buffer index)` pairs from the `[[kernel]]` / stage signature.
fn msl_buffers(msl: &str, entry: &str, stage: &str) -> Result<Vec<(String, u32)>> {
    let sig = msl
        .lines()
        .find(|l| l.starts_with(stage) && l.contains(&format!(" {entry}(")))
        .ok_or_else(|| anyhow!("no {stage} signature for {entry}"))?;
    let mut out = Vec::new();
    for part in sig.split(", ") {
        let Some((decl, rest)) = part.split_once(" [[buffer(") else {
            continue;
        };
        let index: u32 = rest.split(')').next().unwrap().parse()?;
        let var = decl.split_whitespace().last().unwrap();
        // Slang suffixes globals with `_<n>`.
        let name = var.rsplit_once('_').map_or(var, |(base, _)| base);
        out.push((name.to_string(), index));
    }
    Ok(out)
}

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let root = manifest.join("../../slang");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?).join("msl");
    std::fs::create_dir_all(&out_dir)?;
    println!("cargo:rerun-if-changed={}", root.display());
    find_slangc()?;

    let world = root.join("kernels/world_step.slang");
    let fracture = root.join("kernels/fracture_step.slang");
    let includes = [root.join("physics"), root.join("kernels")];
    let includes: Vec<&Path> = includes.iter().map(|p| p.as_path()).collect();
    let slots = bind_slots(&[&world, &fracture])?;
    let world_src = std::fs::read_to_string(&world)?;
    let fracture_src = std::fs::read_to_string(&fracture)?;

    let mut table = String::from(
        "/// (dispatch name, Slang entry, MSL source, numthreads, [(metal buffer, host slot)])\n\
         pub static KERNELS: &[Kernel] = &[\n",
    );
    for name in KERNELS {
        let is_fracture = FRACTURE.contains(name);
        let entry = if is_fracture {
            format!("fracture_{name}")
        } else {
            name.to_string()
        };
        let (src, text) = if is_fracture {
            (&fracture, &fracture_src)
        } else {
            (&world, &world_src)
        };
        let dest = out_dir.join(format!("{entry}.metal"));
        compile_with_defines(
            src,
            &entry,
            SlangTarget::Metal,
            &dest,
            &includes,
            &[("ENGINE_WGSL", "1")],
        )
        .with_context(|| format!("compile {entry}"))?;
        let msl = std::fs::read_to_string(&dest)?;
        let threads = numthreads(text, &entry)?;
        let mut binds = String::new();
        for (global, index) in msl_buffers(&msl, &entry, "[[kernel]]")? {
            let slot = slots
                .get(&global)
                .ok_or_else(|| anyhow!("{entry}: global {global} has no BIND slot"))?;
            write!(binds, "({index}, {slot}), ")?;
        }
        writeln!(
            table,
            "    Kernel {{ name: {name:?}, entry: {entry:?}, msl: include_str!({:?}), threads: {threads:?}, binds: &[{binds}] }},",
            dest.display().to_string()
        )?;
    }
    table.push_str("];\n");

    // Draw shader: one library with vs + three fragment entries.
    let draw = root.join("render/fracture_draw.slang");
    let draw_out = out_dir.join("fracture_draw.metal");
    compile_stage(
        &draw,
        "vs",
        "vertex",
        SlangTarget::Metal,
        &draw_out,
        &[],
        &[],
    )?;
    let vs = std::fs::read_to_string(&draw_out)?;
    let draw_binds = msl_buffers(&vs, "vs", "[[vertex]]")?;
    let expect = |g: &str, i: u32| draw_binds.iter().any(|(n, k)| n == g && *k == i);
    if !(expect("gBodies", 0) && expect("gVerts", 1) && expect("gCamera", 2)) {
        bail!("unexpected draw bindings {draw_binds:?}");
    }
    let mut libs = String::new();
    for (entry, stage) in [
        ("fs", "fragment"),
        ("ground", "fragment"),
        ("wire", "fragment"),
    ] {
        let dest = out_dir.join(format!("fracture_draw_{entry}.metal"));
        compile_stage(&draw, entry, stage, SlangTarget::Metal, &dest, &[], &[])?;
        writeln!(
            libs,
            "    ({entry:?}, include_str!({:?})),",
            dest.display().to_string()
        )?;
    }
    writeln!(
        table,
        "pub static DRAW_VS: &str = include_str!({:?});",
        draw_out.display().to_string()
    )?;
    writeln!(table, "pub static DRAW_FS: &[(&str, &str)] = &[\n{libs}];")?;
    std::fs::write(
        PathBuf::from(std::env::var("OUT_DIR")?).join("fracture_kernels.rs"),
        table,
    )?;
    Ok(())
}
