use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The sources and every file they reach by `#include "..."` in their own
/// directory tree (relative to the including file; one a file split into parts
/// declares its globals in), each once.
fn with_includes(sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = sources.iter().map(|p| p.canonicalize().unwrap_or_else(|_| p.clone())).collect();
    let mut at = 0;
    while at < files.len() {
        let path = files[at].clone();
        at += 1;
        for line in std::fs::read_to_string(&path)?.lines() {
            let Some(inc) = line.trim_start().strip_prefix("#include \"").and_then(|r| r.split('"').next()) else {
                continue;
            };
            let Some(found) = path.parent().map(|d| d.join(inc)).filter(|p| p.is_file()) else {
                continue;
            };
            let found = found.canonicalize().unwrap_or(found);
            if !files.contains(&found) {
                files.push(found);
            }
        }
    }
    Ok(files)
}

/// `BIND(n) <type> gName;` → slot n, from the kernel sources and their includes.
pub fn bind_slots(sources: &[PathBuf]) -> Result<HashMap<String, u32>> {
    let mut map = HashMap::new();
    for path in with_includes(sources)? {
        for line in std::fs::read_to_string(&path)?.lines() {
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
pub fn msl_buffers(msl: &str, entry: &str, stage: &str) -> Result<Vec<(String, u32)>> {
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

/// Metal allows `[[buffer(0..30)]]` per stage, but Slang numbers buffers
/// across every global the module declares, so a kernel in a file that
/// includes `world_step.slang` (the terrain's, bindings 37-52) gets indices
/// past 30. Renumbers the `[[kernel]]` signature's buffers 0, 1, … in
/// order (they are only referenced there); the indices then come from
/// [`msl_buffers`] on the result.
pub fn compact_buffers(msl: &str, entry: &str, stage: &str) -> Result<String> {
    let mut out = String::with_capacity(msl.len());
    let mut found = false;
    for line in msl.split_inclusive('\n') {
        if !found && line.starts_with(stage) && line.contains(&format!(" {entry}(")) {
            found = true;
            let mut rest = line;
            let mut next = 0u32;
            while let Some(at) = rest.find("[[buffer(") {
                let (head, tail) = rest.split_at(at + "[[buffer(".len());
                out.push_str(head);
                let close = tail
                    .find(")]]")
                    .ok_or_else(|| anyhow!("{entry}: unterminated buffer attribute"))?;
                out.push_str(&next.to_string());
                next += 1;
                rest = &tail[close..];
            }
            out.push_str(rest);
            if next > 31 {
                bail!("{entry} binds {next} buffers; Metal allows 31 per stage");
            }
        } else {
            out.push_str(line);
        }
    }
    if !found {
        bail!("no {stage} signature for {entry}");
    }
    Ok(out)
}

/// `[[vk::binding(n, 0)]] <type> gName;` → slot n, from kernel sources that
/// number their globals with Vulkan bindings (the explicit-beams kernels,
/// `engine_exec::explicit_plan`'s `BIND_*`).
/// Each source with the files it reaches by `#include "..."` (relative to the
/// including file): a kernel split into parts declares its globals in any of them.
pub fn vk_binding_slots(sources: &[PathBuf]) -> Result<HashMap<String, u32>> {
    let mut map = HashMap::new();
    for path in with_includes(sources)? {
        let text = std::fs::read_to_string(&path)?;
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("[[vk::binding(") else {
                continue;
            };
            let Some((n, decl)) = rest.split_once(')') else {
                continue;
            };
            let slot: u32 = n.split(',').next().unwrap_or_default().trim().parse()?;
            let Some(decl) = decl.trim_start_matches(']').split(';').next() else {
                continue;
            };
            let Some(name) = decl.split_whitespace().last() else {
                continue;
            };
            if let Some(old) = map.insert(name.to_string(), slot) {
                if old != slot {
                    bail!("{name} is bound to both {old} and {slot}");
                }
            }
        }
    }
    Ok(map)
}

/// Compile one compute entry to MSL (`-DENGINE_WGSL=1`) with reflection;
/// returns (msl, thread-group size).
pub fn compile_kernel(
    slangc: &Path,
    root: &Path,
    source: &Path,
    entry: &str,
    dest: &Path,
) -> Result<(String, [u32; 3])> {
    compile_kernel_with_defines(slangc, root, source, entry, dest, &[("ENGINE_WGSL", "1")])
}

/// [`compile_kernel`] with `-D NAME=VALUE` for each define (`ENGINE_WGSL=1`
/// among them for the browser's layout).
pub fn compile_kernel_with_defines(
    slangc: &Path,
    root: &Path,
    source: &Path,
    entry: &str,
    dest: &Path,
    defines: &[(&str, &str)],
) -> Result<(String, [u32; 3])> {
    let _ = slangc;
    let reflection = dest.with_extension("json");
    let include_dirs = [root.join("physics"), root.join("kernels")];
    super::Slangc {
        source,
        entries: &[(entry, "compute")],
        target: "metal",
        include_dirs: &include_dirs.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
        defines,
        out: dest,
        out_ext: "metal",
        reflection: Some(&reflection),
        base: None,
    }
    .run()
    .with_context(|| format!("slangc {entry}"))?;
    let json = std::fs::read_to_string(&reflection)?;
    Ok((
        std::fs::read_to_string(dest)?,
        reflected_threads(&json, entry)?,
    ))
}

/// `threadGroupSize` of `entry` in Slang's reflection JSON.
pub fn reflected_threads(json: &str, entry: &str) -> Result<[u32; 3]> {
    let missing = || anyhow!("{entry}: no reflected thread group size");
    let at = json
        .find(&format!("\"name\": \"{entry}\""))
        .ok_or_else(missing)?;
    let rest = &json[at..];
    let key = rest.find("\"threadGroupSize\"").ok_or_else(missing)?;
    let open = rest[key..].find('[').ok_or_else(missing)? + key;
    let close = rest[open..].find(']').ok_or_else(missing)? + open;
    let v: Vec<u32> = rest[open + 1..close]
        .split(',')
        .map(|x| x.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    match v[..] {
        [x, y, z] => Ok([x, y, z]),
        _ => bail!("{entry}: thread group size {v:?}"),
    }
}
