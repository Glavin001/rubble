//! Compile the fracture lab's kernels and draw shader to MSL and emit their tables.
//!
//! The kernel list is the shared plan's (`engine_kernel_manifest::fracture::KERNELS`, in
//! command-id order), built from the same Slang entries as the browser lab's
//! WGSL with the same `ENGINE_WGSL` buffer layout, so one host buffer set and
//! one command list serve both hosts. Slang assigns Metal `[[buffer(n)]]` per
//! entry, so each entry's signature is mapped back to the shared `BIND(slot)`
//! numbering; thread-group sizes come from Slang's reflection.

use anyhow::{anyhow, bail, Result};
use engine_kernel_manifest::fracture::{kernel_source, KERNELS};
use engine_slang_build::msl_layout::{bind_slots, compile_kernel, msl_buffers};
use engine_slang_build::{compile_stage, find_slangc, par_map, SlangTarget};
use std::fmt::Write;
use std::path::PathBuf;

fn main() -> Result<()> {
    engine_slang_build::report_slangc_for_cargo();
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let root = manifest.join("../../slang");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?).join("msl");
    std::fs::create_dir_all(&out_dir)?;
    println!("cargo:rerun-if-changed={}", root.display());
    let slangc = find_slangc()?;

    let kernels = root.join("kernels");
    let slots = bind_slots(&[
        kernels.join("world_step.slang"),
        kernels.join("convex_step.slang"),
        kernels.join("guide_step.slang"),
        root.join("physics/contact_storage.slang"),
    ])?;

    // The world_step entries are large: compile them in parallel (each one cached by
    // what it depends on, engine_slang_build's per-compile cache).
    let compiled: Vec<_> = par_map(KERNELS, |&(_, entry)| {
        let dest = out_dir.join(format!("{entry}.metal"));
        let source = root.join("kernels").join(kernel_source(entry));
        compile_kernel(&slangc, &root, &source, entry, &dest).map(|(msl, threads)| (dest, msl, threads))
    })
    .into_iter()
    .map(|r| r.map(Some))
    .collect::<Result<_>>()?;
    let mut compiled: Vec<Option<(PathBuf, String, [u32; 3])>> = compiled;

    let mut table = String::from(
        "/// Plan kernels in command-id order: (dispatch name, entry, MSL, threads, [(metal buffer, host slot)])\n\
         pub static KERNELS: &[Kernel] = &[\n",
    );
    for (k, &(name, entry)) in KERNELS.iter().enumerate() {
        let (dest, msl, threads) = compiled[k].take().unwrap();
        let mut binds = String::new();
        for (global, index) in msl_buffers(&msl, entry, "[[kernel]]")? {
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
    if !(expect("gBodies", 0) && expect("gVerts", 1) && expect("gView", 2)) {
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
