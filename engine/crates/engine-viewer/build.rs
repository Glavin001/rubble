//! Compile the fracture lab's kernels and draw shader to MSL and emit their tables.
//!
//! The kernel list is the shared plan's (`engine_fracture::plan::KERNELS`, in
//! command-id order), built from the same Slang entries as the browser lab's
//! WGSL with the same `ENGINE_WGSL` buffer layout, so one host buffer set and
//! one command list serve both hosts. Slang assigns Metal `[[buffer(n)]]` per
//! entry, so each entry's signature is mapped back to the shared `BIND(slot)`
//! numbering; thread-group sizes come from Slang's reflection.

use anyhow::{anyhow, bail, Context, Result};
use engine_fracture::plan::{kernel_source, KERNELS};
use engine_slang_build::{compile_stage, find_slangc, SlangTarget};
use std::collections::HashMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `BIND(n) <type> gName;` → slot n, from the kernel sources.
fn bind_slots(sources: &[PathBuf]) -> Result<HashMap<String, u32>> {
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
            let slot: u32 = n.trim().parse()?;
            if let Some(old) = map.insert(name.to_string(), slot) {
                if old != slot {
                    bail!("{name} is bound to both {old} and {slot}");
                }
            }
        }
    }
    Ok(map)
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

/// Compile one compute entry to MSL with reflection; returns (msl, threads).
fn compile_kernel(
    slangc: &Path,
    root: &Path,
    entry: &str,
    dest: &Path,
) -> Result<(String, [u32; 3])> {
    let source = root.join("kernels").join(kernel_source(entry));
    let reflection = dest.with_extension("json");
    let output = Command::new(slangc)
        .args([
            "-target",
            "metal",
            "-stage",
            "compute",
            "-entry",
            entry,
            "-DENGINE_WGSL=1",
            "-I",
        ])
        .arg(root.join("physics"))
        .arg("-I")
        .arg(root.join("kernels"))
        .arg("-o")
        .arg(dest)
        .arg("-reflection-json")
        .arg(&reflection)
        .arg(&source)
        .output()
        .with_context(|| format!("spawn {}", slangc.display()))?;
    if !output.status.success() {
        bail!(
            "slangc {entry}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&reflection)?)?;
    let size = json["entryPoints"]
        .as_array()
        .and_then(|e| e.iter().find(|e| e["name"] == entry))
        .map(|e| &e["threadGroupSize"])
        .ok_or_else(|| anyhow!("{entry}: no reflected entry point"))?;
    let t = |k: usize| {
        size[k]
            .as_u64()
            .map(|x| x as u32)
            .ok_or_else(|| anyhow!("{entry}: thread group size"))
    };
    Ok((std::fs::read_to_string(dest)?, [t(0)?, t(1)?, t(2)?]))
}

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

    // The world_step entries are large: compile them in parallel.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<_> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(_, entry)) = KERNELS.get(k) else {
                            break;
                        };
                        let dest = out_dir.join(format!("{entry}.metal"));
                        done.push((
                            k,
                            dest.clone(),
                            compile_kernel(&slangc, &root, entry, &dest),
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
    let mut compiled: Vec<Option<(PathBuf, String, [u32; 3])>> = vec![None; KERNELS.len()];
    for (k, dest, r) in results {
        let (msl, threads) = r?;
        compiled[k] = Some((dest, msl, threads));
    }

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
