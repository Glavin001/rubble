//! A local, browser-sized copy of a community BeamNG vehicle for the crash lab's
//! `?mod=` loader: its JBeam parts, configurations and materials as they are, its
//! Collada meshes as they are (one level of detail: BeamNG's files carry one per
//! mesh), and only the textures its configurations' flexbodies use, decoded (DDS:
//! BC1–BC7; PNG) and downscaled to at most `size` pixels, written as PNG
//! beside the original's name (`<stem>.png`). Writes `files.json` (the folder's
//! file list, the page's index). Nothing here is committed: point `out` at the
//! gitignored `inspiration/` or `engine/crates/engine-webgpu/__mods/`.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example community_cache -- \
//!     <vehicles-dir> <vehicle> <all|a.pc,b.pc> <out-dir> [size=1024] [label=...]
//! ```
//!
//! The mod's own files are data only: no script in them is read or run.
use std::path::{Path, PathBuf};

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn ext(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default()
}

/// RGBA8 of a DDS (DX10 or legacy header) or PNG file.
fn decode(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    if !bytes.starts_with(b"DDS ") {
        let img = engine_softbody::flextex::decode(bytes)?;
        return Ok((img.w, img.h, img.rgba));
    }
    let u32_at = |o: usize| -> u32 {
        bytes
            .get(o..o + 4)
            .map_or(0, |s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    };
    let (h, w) = (u32_at(12), u32_at(16));
    let fourcc = &bytes[84..88];
    // BCn kind: 1, 2, 3, 4, 5, 7 (DXGI formats: Microsoft's documented numbers).
    let (kind, start) = if fourcc == b"DX10" {
        let k = match u32_at(128) {
            70..=72 => 1,
            73..=75 => 2,
            76..=78 => 3,
            79..=81 => 4,
            82..=84 => 5,
            95 => 6,
            96 => 60,
            97..=99 => 7,
            f => return Err(format!("DXGI format {f} not read")),
        };
        (k, 148)
    } else {
        let k = match fourcc {
            b"DXT1" => 1,
            b"DXT2" | b"DXT3" => 2,
            b"DXT4" | b"DXT5" => 3,
            b"ATI1" | b"BC4U" => 4,
            b"ATI2" | b"BC5U" => 5,
            _ => {
                let img = engine_softbody::flextex::decode(bytes)?;
                return Ok((img.w, img.h, img.rgba));
            }
        };
        (k, 128)
    };
    let bsize = if matches!(kind, 1 | 4) { 8 } else { 16 };
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let data = bytes
        .get(start..start + bw * bh * bsize)
        .ok_or("DDS data truncated")?;
    let (pw, ph) = (bw * 4, bh * 4);
    let mut rgba = vec![0u8; pw * ph * 4];
    let mut tmp = [0u8; 64];
    for by in 0..bh {
        for bx in 0..bw {
            let b = &data[(by * bw + bx) * bsize..][..bsize];
            match kind {
                1 => bcdec_rs::bc1(b, &mut tmp, 16),
                2 => bcdec_rs::bc2(b, &mut tmp, 16),
                3 => bcdec_rs::bc3(b, &mut tmp, 16),
                7 => bcdec_rs::bc7(b, &mut tmp, 16),
                6 | 60 => {
                    // BC6H (half-float RGB): clamped to [0, 1] (the data maps' range).
                    let mut f = [0f32; 48];
                    bcdec_rs::bc6h_float(b, &mut f, 12, kind == 60);
                    for i in 0..16 {
                        let c = |k: usize| (f[i * 3 + k].clamp(0.0, 1.0) * 255.0).round() as u8;
                        tmp[i * 4..i * 4 + 4].copy_from_slice(&[c(0), c(1), c(2), 255]);
                    }
                }
                4 => {
                    let mut r = [0u8; 16];
                    bcdec_rs::bc4(b, &mut r, 4, false);
                    for i in 0..16 {
                        tmp[i * 4..i * 4 + 4].copy_from_slice(&[r[i], r[i], r[i], 255]);
                    }
                }
                _ => {
                    // Two channels (a normal map's x, y): z rebuilt as a unit vector.
                    let mut rg = [0u8; 32];
                    bcdec_rs::bc5(b, &mut rg, 8, false);
                    for i in 0..16 {
                        let (x, y) = (rg[2 * i], rg[2 * i + 1]);
                        let (fx, fy) = (x as f32 / 127.5 - 1.0, y as f32 / 127.5 - 1.0);
                        let fz = (1.0 - fx * fx - fy * fy).max(0.0).sqrt();
                        let z = ((fz + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                        tmp[i * 4..i * 4 + 4].copy_from_slice(&[x, y, z, 255]);
                    }
                }
            }
            for row in 0..4 {
                let o = ((by * 4 + row) * pw + bx * 4) * 4;
                rgba[o..o + 16].copy_from_slice(&tmp[row * 16..row * 16 + 16]);
            }
        }
    }
    // Crop the block padding.
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h as usize {
        out.extend_from_slice(&rgba[y * pw * 4..y * pw * 4 + w as usize * 4]);
    }
    Ok((w, h, out))
}

/// Box-filter halving until the larger side is at most `size`.
fn shrink(mut w: u32, mut h: u32, mut px: Vec<u8>, size: u32) -> (u32, u32, Vec<u8>) {
    while w.max(h) > size && w > 1 && h > 1 {
        let (nw, nh) = (w / 2, h / 2);
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                for k in 0..4 {
                    let at = |xx: u32, yy: u32| px[((yy * w + xx) * 4 + k) as usize] as u32;
                    let v = at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1);
                    next[((y * nw + x) * 4 + k) as usize] = ((v + 2) / 4) as u8;
                }
            }
        }
        (w, h, px) = (nw, nh, next);
    }
    (w, h, px)
}

/// The Collada text without the `<geometry>` elements whose `name` (else `id`)
/// is not in `keep` (a scene node naming a removed geometry draws nothing).
fn strip_geometries(text: &str, keep: &std::collections::BTreeSet<&str>) -> String {
    let attr = |tag: &str, k: &str| -> Option<String> {
        let at = tag.find(&format!(" {k}=\""))? + k.len() + 3;
        Some(tag[at..].split('"').next()?.to_string())
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find("<geometry ") {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let open_end = tail.find('>').unwrap_or(tail.len());
        let tag = &tail[..open_end];
        let close = tail.find("</geometry>").map_or(tail.len(), |c| c + "</geometry>".len());
        let id = attr(tag, "id").unwrap_or_default();
        let name = attr(tag, "name").unwrap_or_else(|| id.clone());
        if keep.contains(name.as_str()) || keep.contains(id.as_str()) {
            out.push_str(&tail[..close]);
        }
        rest = &tail[close..];
    }
    out.push_str(rest);
    out
}

fn write_png(path: &Path, w: u32, h: u32, px: &[u8]) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(px).map_err(|e| e.to_string())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (vehicles, name, pc, out) = (Path::new(&a[0]), &a[1], &a[2], Path::new(&a[3]));
    let size: u32 = a
        .iter()
        .find_map(|x| x.strip_prefix("size="))
        .map_or(1024, |s| s.parse().unwrap());
    // The vehicle's folder and the mod's shared `common` folder.
    let mut files = Vec::new();
    walk(&vehicles.join(name), &mut files);
    walk(&vehicles.join("common"), &mut files);
    files.sort();
    let rel = |p: &Path| p.strip_prefix(vehicles).unwrap().to_string_lossy().replace('\\', "/");
    let mut listed: Vec<String> = Vec::new();
    let copy = |p: &Path, listed: &mut Vec<String>| {
        let r = rel(p);
        let dst = out.join(&r);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::copy(p, &dst).unwrap();
        listed.push(r);
    };
    let (mut jbeam, mut mats, mut daes) = (Vec::new(), Vec::new(), Vec::new());
    for p in &files {
        let n = p.file_name().unwrap().to_string_lossy().to_ascii_lowercase();
        match ext(p).as_str() {
            "jbeam" => {
                jbeam.push((rel(p), std::fs::read_to_string(p).unwrap()));
                copy(p, &mut listed);
            }
            "pc" => copy(p, &mut listed),
            "json" if n.ends_with(".materials.json") => {
                mats.push(std::fs::read_to_string(p).unwrap());
                copy(p, &mut listed);
            }
            "dae" => daes.push(p.clone()),
            _ => {}
        }
    }
    // The configurations: `all` (every .pc the vehicle ships) or a comma list.
    let pc_names: Vec<String> = if pc == "all" {
        let mut v: Vec<String> = std::fs::read_dir(vehicles.join(name))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|f| f.to_ascii_lowercase().ends_with(".pc"))
            .collect();
        v.sort();
        v
    } else {
        pc.split(',').map(str::to_string).collect()
    };
    let refs: Vec<(&str, &str)> = jbeam.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    // Every mesh of every .dae (the configurations decide which go).
    let mut all = Vec::new();
    let mut dae_meshes = Vec::new();
    for p in &daes {
        let text = String::from_utf8_lossy(&std::fs::read(p).unwrap()).into_owned();
        let m = engine_softbody::flexmesh::read_collada(&text).unwrap_or_default();
        dae_meshes.push((
            p.clone(),
            m.iter().map(|x| (x.name.clone(), x.geometry.clone())).collect::<Vec<_>>(),
        ));
        all.extend(m);
    }
    let mut info = std::collections::HashMap::new();
    for m in &mats {
        info.extend(engine_softbody::flexmesh::read_materials(m));
    }
    let mut used: std::collections::BTreeSet<String> = Default::default();
    let mut tex: std::collections::BTreeSet<String> = Default::default();
    let mut configs: Vec<serde_json::Value> = Vec::new();
    for pc in &pc_names {
        let path = format!("{name}/{pc}");
        let Ok(pcs) = std::fs::read_to_string(vehicles.join(name).join(pc)) else {
            continue;
        };
        let loaded = engine_softbody::jbeam::load_jbeam(&refs, Some(&pcs)).and_then(|desc| {
            let flex = engine_softbody::jbeam::assembled_rows(&refs, Some(&pcs), "flexbodies")?;
            let props = engine_softbody::jbeam::assembled_rows(&refs, Some(&pcs), "props")?;
            let (sim, _) = engine_softbody::explicit_beams::ExplicitBeams::from_desc_jbeam_with(
                &desc,
                engine_softbody::explicit_beams::JbeamWheels::Raycast,
                glam::Vec3::ZERO,
                glam::Vec3::ZERO,
                Default::default(),
            )?;
            let (vis, rep) =
                engine_softbody::flexmesh::jbeam_visual(&flex, &props, &desc, &sim, &all, &info, [0.5; 3])?;
            Ok((desc, sim, flex, props, vis, rep))
        });
        match loaded {
            Ok((desc, sim, flex, props, vis, rep)) => {
                used.extend(
                    flex.iter()
                        .chain(&props)
                        .filter_map(|(_, f)| f.get("mesh").and_then(|m| m.as_str()).map(str::to_string)),
                );
                tex.extend(vis.textures.iter().chain(&vis.data_textures).cloned());
                println!(
                    "{path}: {} nodes, {} beams, {} flexbodies, {} triangles, unbound {}",
                    sim.nodes.len(),
                    sim.beams.len(),
                    rep.flexbodies,
                    rep.triangles,
                    rep.unbound
                );
                configs.push(serde_json::json!({
                    "pc": path,
                    "name": desc.name,
                    "nodes": sim.nodes.len(),
                    "beams": sim.beams.len(),
                    "triangles": rep.triangles,
                }));
            }
            Err(e) => println!("{path}: {e} (left out of the index)"),
        }
    }
    let used: std::collections::BTreeSet<&str> = used.iter().map(String::as_str).collect();
    // Each .dae a used mesh is in, without the geometries no used mesh draws.
    for (p, meshes) in &dae_meshes {
        let keep: std::collections::BTreeSet<&str> = meshes
            .iter()
            .filter(|(n, g)| used.contains(n.as_str()) || used.contains(g.as_str()))
            .map(|(_, g)| g.as_str())
            .collect();
        if keep.is_empty() {
            continue;
        }
        let text = String::from_utf8_lossy(&std::fs::read(p).unwrap()).into_owned();
        let slim = strip_geometries(&text, &keep);
        let r = rel(p);
        let dst = out.join(&r);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::write(&dst, &slim).unwrap();
        println!("{r}: {:.1} -> {:.1} MB", text.len() as f64 / 1e6, slim.len() as f64 / 1e6);
        listed.push(r);
    }
    let wanted: Vec<&String> = tex.iter().collect();
    let mut bytes_out = 0u64;
    for t in &wanted {
        let src = files.iter().find(|p| {
            p.file_name()
                .is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(t))
        });
        // BeamNG materials may name a .png the folder holds as .dds (or the reverse).
        let stem = |s: &str| s.rsplit_once('.').map_or(s.to_ascii_lowercase(), |x| x.0.to_ascii_lowercase());
        let src = src.or_else(|| {
            files.iter().find(|p| {
                matches!(ext(p).as_str(), "dds" | "png")
                    && p.file_name().is_some_and(|f| stem(&f.to_string_lossy()) == stem(t))
            })
        });
        let Some(src) = src else {
            println!("texture {t}: not in the mod");
            continue;
        };
        match decode(&std::fs::read(src).unwrap()) {
            Ok((w, h, px)) => {
                let (w2, h2, px) = shrink(w, h, px, size);
                let r = rel(src);
                let r = format!("{}.png", r.rsplit_once('.').map_or(r.as_str(), |x| x.0));
                let dst = out.join(&r);
                std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
                write_png(&dst, w2, h2, &px).unwrap();
                bytes_out += std::fs::metadata(&dst).unwrap().len();
                println!("texture {t}: {w}x{h} -> {w2}x{h2} {r}");
                listed.push(r);
            }
            Err(e) => println!("texture {t}: {e} (left out)"),
        }
    }
    listed.push("files.json".into());
    listed.sort();
    listed.dedup();
    std::fs::write(out.join("files.json"), serde_json::to_string(&listed).unwrap()).unwrap();
    let total: u64 = listed
        .iter()
        .filter_map(|r| std::fs::metadata(out.join(r)).ok())
        .map(|m| m.len())
        .sum();
    println!(
        "{name}: {} configurations, {} files, {:.1} MB ({:.1} MB textures)",
        configs.len(),
        listed.len(),
        total as f64 / 1e6,
        bytes_out as f64 / 1e6
    );
    // The page's picker: `<out's parent>/index.json`, one entry per cache folder.
    let label = a
        .iter()
        .find_map(|x| x.strip_prefix("label="))
        .map_or_else(|| name.to_string(), str::to_string);
    let folder = out.file_name().unwrap().to_string_lossy().into_owned();
    let index_path = out.parent().unwrap().join("index.json");
    let mut index: Vec<serde_json::Value> = std::fs::read_to_string(&index_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    index.retain(|e| e.get("folder").and_then(|f| f.as_str()) != Some(folder.as_str()));
    index.push(serde_json::json!({
        "folder": folder,
        "label": label,
        "vehicle": name,
        "megabytes": (total as f64 / 1e5).round() / 10.0,
        "configs": configs,
    }));
    index.sort_by(|a, b| a["label"].as_str().cmp(&b["label"].as_str()));
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
}
