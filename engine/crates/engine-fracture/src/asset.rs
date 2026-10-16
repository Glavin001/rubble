//! SHARDWRIGHT_fracture v1 GLB reader (port of `fracture/asset.mjs`).
//!
//! Reads only what a rigid-body host needs: fragments, hulls + face topology, bonds
//! and the glTF render meshes. Every offset is bounds-checked; malformed input fails.

use anyhow::{anyhow, bail, ensure, Result};
use serde_json::Value;

pub const EXTENSION: &str = "SHARDWRIGHT_fracture";

#[derive(Clone, Debug)]
pub struct Fragment {
    pub id: u32,
    pub level: u8,
    pub mass: f32,
    pub com: [f32; 3],
    /// Row-major 3×3 inertia about the COM.
    pub inertia: [f32; 9],
    pub material_ids: Vec<u16>,
    pub hull_begin: u32,
    pub hull_count: u32,
    pub node: i32,
    pub volume: f32,
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub level: u8,
    pub a: u32,
    pub b: i32,
    pub centroid: [f32; 3],
    pub anchor: bool,
}

#[derive(Clone, Debug)]
pub struct Hull {
    pub fragment: u32,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub sizes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Physics {
    pub name: String,
    pub fragments: Vec<Fragment>,
    pub bonds: Vec<Bond>,
    pub hulls: Vec<Hull>,
}

pub struct FractureAsset {
    pub json: Value,
    pub physics: Physics,
    pub material_count: usize,
    pub bytes: usize,
    bin: Vec<u8>,
}

/// Bounds-checked FlatBuffers reader for Shardwright `schemas/frac.fbs` (schema 1).
struct Frac<'a> {
    b: &'a [u8],
}

impl<'a> Frac<'a> {
    fn check(&self, p: usize, n: usize) -> Result<usize> {
        match p.checked_add(n) {
            Some(end) if end <= self.b.len() => Ok(p),
            _ => bail!("Truncated FRAC payload"),
        }
    }
    fn u8(&self, p: usize) -> Result<u8> {
        Ok(self.b[self.check(p, 1)?])
    }
    fn u16(&self, p: usize) -> Result<u16> {
        let p = self.check(p, 2)?;
        Ok(u16::from_le_bytes([self.b[p], self.b[p + 1]]))
    }
    fn u32(&self, p: usize) -> Result<u32> {
        let p = self.check(p, 4)?;
        Ok(u32::from_le_bytes(self.b[p..p + 4].try_into().unwrap()))
    }
    fn i32(&self, p: usize) -> Result<i32> {
        Ok(self.u32(p)? as i32)
    }
    fn f32(&self, p: usize) -> Result<f32> {
        let x = f32::from_bits(self.u32(p)?);
        ensure!(x.is_finite(), "Non-finite FRAC value");
        Ok(x)
    }
    /// Absolute position of field `i` in table `t`, or 0 when absent.
    fn field(&self, t: usize, i: usize) -> Result<usize> {
        let v = (t as i64) - self.i32(t)? as i64;
        ensure!(v >= 0, "Truncated FRAC payload");
        let v = v as usize;
        let n = self.u16(v)? as usize;
        self.check(v, n)?;
        let o = 4 + 2 * i;
        let offset = if o < n { self.u16(v + o)? as usize } else { 0 };
        if offset == 0 {
            return Ok(0);
        }
        self.check(t + offset, 1)
    }
    fn scalar_u32(&self, t: usize, i: usize, fallback: u32) -> Result<u32> {
        let p = self.field(t, i)?;
        if p == 0 {
            Ok(fallback)
        } else {
            self.u32(p)
        }
    }
    fn scalar_i32(&self, t: usize, i: usize, fallback: i32) -> Result<i32> {
        let p = self.field(t, i)?;
        if p == 0 {
            Ok(fallback)
        } else {
            self.i32(p)
        }
    }
    fn scalar_u16(&self, t: usize, i: usize) -> Result<u16> {
        let p = self.field(t, i)?;
        if p == 0 {
            Ok(0)
        } else {
            self.u16(p)
        }
    }
    fn scalar_u8(&self, t: usize, i: usize) -> Result<u8> {
        let p = self.field(t, i)?;
        if p == 0 {
            Ok(0)
        } else {
            self.u8(p)
        }
    }
    fn scalar_f32(&self, t: usize, i: usize) -> Result<f32> {
        let p = self.field(t, i)?;
        if p == 0 {
            Ok(0.0)
        } else {
            self.f32(p)
        }
    }
    fn structf<const N: usize>(&self, t: usize, i: usize) -> Result<[f32; N]> {
        let p = self.field(t, i)?;
        let mut out = [0.0; N];
        if p != 0 {
            for (k, x) in out.iter_mut().enumerate() {
                *x = self.f32(p + 4 * k)?;
            }
        }
        Ok(out)
    }
    /// (start, count) of a vector field; elements are `stride` bytes.
    fn vector(&self, t: usize, i: usize, stride: usize) -> Result<(usize, usize)> {
        let p = self.field(t, i)?;
        if p == 0 {
            return Ok((0, 0));
        }
        let v = p + self.u32(p)? as usize;
        let n = self.u32(v)? as usize;
        self.check(
            v + 4,
            n.checked_mul(stride)
                .ok_or_else(|| anyhow!("Truncated FRAC payload"))?,
        )?;
        Ok((v + 4, n))
    }
    fn tables(&self, t: usize, i: usize) -> Result<Vec<usize>> {
        let (start, n) = self.vector(t, i, 4)?;
        (0..n)
            .map(|k| {
                let p = start + 4 * k;
                Ok(p + self.u32(p)? as usize)
            })
            .collect()
    }
    fn string(&self, t: usize, i: usize) -> Result<String> {
        let (start, n) = self.vector(t, i, 1)?;
        Ok(std::str::from_utf8(&self.b[start..start + n])?.to_string())
    }
}

pub fn read_physics(bytes: &[u8]) -> Result<(Physics, String, String)> {
    ensure!(
        bytes.len() >= 8 && &bytes[4..8] == b"FRAC",
        "Missing FRAC identifier"
    );
    let r = Frac { b: bytes };
    let root = r.u32(0)? as usize;
    ensure!(r.scalar_u16(root, 0)? == 1, "Unsupported physics schema");

    let mut fragments = Vec::new();
    for t in r.tables(root, 6)? {
        let (ms, mn) = r.vector(t, 8, 2)?;
        let (_, fnn) = r.vector(t, 9, 4)?;
        ensure!(mn == fnn, "Invalid fragment ranges");
        fragments.push((
            Fragment {
                id: r.scalar_u32(t, 0, 0)?,
                level: r.scalar_u8(t, 1)?,
                mass: r.scalar_f32(t, 5)?,
                com: r.structf::<3>(t, 6)?,
                inertia: r.structf::<9>(t, 7)?,
                material_ids: (0..mn).map(|k| r.u16(ms + 2 * k)).collect::<Result<_>>()?,
                hull_begin: r.scalar_u32(t, 10, 0)?,
                hull_count: r.scalar_u16(t, 11)? as u32,
                node: r.scalar_i32(t, 14, -1)?,
                volume: r.scalar_f32(t, 15)?,
            },
            r.scalar_i32(t, 2, -1)?,
            r.scalar_u32(t, 3, 0)?,
            r.scalar_u32(t, 4, 0)?,
        ));
    }
    let (_, loop_count) = r.vector(root, 10, 4)?;
    let (_, spawn_count) = r.vector(root, 12, 24)?;
    let (_, bond_children) = r.vector(root, 17, 4)?;
    let mut bonds = Vec::new();
    for (i, t) in r.tables(root, 7)?.into_iter().enumerate() {
        let bond = Bond {
            level: r.scalar_u8(t, 1)?,
            a: r.scalar_u32(t, 2, 0)?,
            b: r.scalar_i32(t, 3, 0)?,
            centroid: r.structf::<3>(t, 5)?,
            anchor: r.scalar_u8(t, 30)? != 0,
        };
        let nf = fragments.len() as i64;
        let ok = r.scalar_u32(t, 0, 0)? as usize == i
            && (bond.a as i64) < nf
            && bond.b >= -1
            && (bond.b as i64) < nf
            && r.scalar_f32(t, 4)? >= 0.0
            && (r.scalar_u32(t, 24, 0)? + r.scalar_u32(t, 25, 0)?) as usize <= bond_children
            && (r.scalar_u32(t, 26, 0)? + r.scalar_u16(t, 27)? as u32) as usize <= loop_count
            && (r.scalar_u32(t, 28, 0)? + r.scalar_u32(t, 29, 0)?) as usize <= spawn_count;
        ensure!(ok, "Invalid bond references");
        bonds.push(bond);
    }
    let hull_tables = r.tables(root, 8)?;
    let face_tables = r.tables(root, 9)?;
    ensure!(hull_tables.len() == face_tables.len(), "Missing hull faces");
    let mut hulls = Vec::with_capacity(hull_tables.len());
    for (t, ft) in hull_tables.into_iter().zip(face_tables) {
        let (vs, vn) = r.vector(t, 0, 12)?;
        let vertices = (0..vn)
            .map(|k| {
                Ok([
                    r.f32(vs + 12 * k)?,
                    r.f32(vs + 12 * k + 4)?,
                    r.f32(vs + 12 * k + 8)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let (is, inn) = r.vector(ft, 0, 4)?;
        let (ss, sn) = r.vector(ft, 1, 1)?;
        let hull = Hull {
            fragment: r.scalar_u32(t, 1, 0)?,
            vertices,
            indices: (0..inn).map(|k| r.u32(is + 4 * k)).collect::<Result<_>>()?,
            sizes: (0..sn).map(|k| r.u8(ss + k)).collect::<Result<_>>()?,
        };
        let total: usize = hull.sizes.iter().map(|&s| s as usize).sum();
        ensure!(
            (hull.fragment as usize) < fragments.len()
                && (4..=64).contains(&hull.vertices.len())
                && total == hull.indices.len()
                && hull
                    .indices
                    .iter()
                    .all(|&v| (v as usize) < hull.vertices.len()),
            "Invalid convex hull"
        );
        hulls.push(hull);
    }
    let n = fragments.len();
    for (i, (f, parent, child_begin, child_count)) in fragments.iter().enumerate() {
        ensure!(
            f.id as usize == i
                && f.mass >= 0.0
                && *parent >= -1
                && (*parent as i64) < n as i64
                && (f.hull_begin + f.hull_count) as usize <= hulls.len()
                && (child_begin + child_count) as usize <= n,
            "Invalid fragment ranges"
        );
    }
    let physics = Physics {
        name: r.string(root, 13)?,
        fragments: fragments.into_iter().map(|f| f.0).collect(),
        bonds,
        hulls,
    };
    Ok((physics, r.string(root, 4)?, r.string(root, 5)?))
}

pub fn parse_fracture_glb(buffer: &[u8]) -> Result<FractureAsset> {
    let u32_at = |p: usize| u32::from_le_bytes(buffer[p..p + 4].try_into().unwrap());
    ensure!(
        buffer.len() >= 20
            && u32_at(0) == 0x4654_6c67
            && u32_at(4) == 2
            && u32_at(8) as usize == buffer.len(),
        "Invalid GLB header or length"
    );
    let (mut json, mut bin): (Option<Value>, Option<&[u8]>) = (None, None);
    let mut p = 12;
    while p < buffer.len() {
        ensure!(p + 8 <= buffer.len(), "Truncated GLB chunk");
        let (n, kind) = (u32_at(p) as usize, u32_at(p + 4));
        p += 8;
        ensure!(
            n % 4 == 0 && p + n <= buffer.len(),
            "Invalid GLB chunk range"
        );
        if kind == 0x4e4f_534a {
            ensure!(json.is_none(), "Duplicate JSON");
            json = Some(serde_json::from_slice(&buffer[p..p + n])?);
        }
        if kind == 0x004e_4942 {
            ensure!(bin.is_none(), "Duplicate BIN");
            bin = Some(&buffer[p..p + n]);
        }
        p += n;
    }
    let json = json.ok_or_else(|| anyhow!("This GLB needs the SHARDWRIGHT_fracture v1 export"))?;
    let ext = &json["extensions"][EXTENSION];
    let bin = match bin {
        Some(b)
            if ext["version"] == 1
                && ext["encoding"] == "flatbuffers"
                && ext["physicsSchemaVersion"] == 1 =>
        {
            b
        }
        _ => bail!("This GLB needs the SHARDWRIGHT_fracture v1 export"),
    };
    let coords = &ext["coordinates"];
    ensure!(
        coords["units"] == "meters" && coords["up"] == "Y" && coords["handedness"] == "right",
        "Unsupported fracture coordinate convention"
    );
    let required = json["extensionsRequired"].as_array();
    ensure!(
        !required.is_some_and(|r| r.iter().any(|e| e == "EXT_meshopt_compression")),
        "Export with --uncompressed for this demo"
    );
    let buffers = json["buffers"].as_array();
    ensure!(
        buffers.is_some_and(|b| b.len() == 1
            && b[0].get("uri").is_none()
            && b[0]["byteLength"]
                .as_u64()
                .is_some_and(|n| n as usize <= bin.len()))
            && !json["images"]
                .as_array()
                .is_some_and(|i| i.iter().any(|i| i.get("uri").is_some())),
        "Expected a self-contained GLB"
    );
    let asset = FractureAsset {
        bytes: buffer.len(),
        bin: bin.to_vec(),
        json: json.clone(),
        physics: Physics {
            name: String::new(),
            fragments: vec![],
            bonds: vec![],
            hulls: vec![],
        },
        material_count: 0,
    };
    let view = ext["bufferView"]
        .as_u64()
        .ok_or_else(|| anyhow!("Invalid bufferView"))? as usize;
    let (physics, library_id, library_version) = read_physics(asset.slice(view)?)?;
    let library = &ext["materialLibrary"];
    ensure!(
        library["library_id"] == library_id.as_str()
            && library["library_version"] == library_version.as_str(),
        "Mismatched material library"
    );
    let materials = library["material"].as_array().map_or(0, |m| m.len());
    for f in &physics.fragments {
        if f.node >= 0 {
            ensure!(
                json["nodes"][f.node as usize]["extras"]["fragment_id"] == f.id,
                "Mismatched fragment node"
            );
        }
        ensure!(
            f.material_ids.iter().all(|&id| (id as usize) < materials),
            "Missing physical material"
        );
    }
    Ok(FractureAsset {
        physics,
        material_count: materials,
        ..asset
    })
}

impl FractureAsset {
    fn slice(&self, index: usize) -> Result<&[u8]> {
        let v = &self.json["bufferViews"][index];
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = v["byteLength"]
            .as_u64()
            .ok_or_else(|| anyhow!("Invalid bufferView"))? as usize;
        let total = self.json["buffers"][0]["byteLength"].as_u64().unwrap_or(0) as usize;
        ensure!(
            v["buffer"] == 0
                && off
                    .checked_add(len)
                    .is_some_and(|e| e <= total && e <= self.bin.len()),
            "Invalid bufferView"
        );
        Ok(&self.bin[off..off + len])
    }

    /// Decode an accessor into `count` rows of `width` components (as f64, like JS numbers).
    pub fn accessor(&self, index: usize) -> Result<Vec<Vec<f64>>> {
        let a = &self.json["accessors"][index];
        ensure!(
            a.is_object() && a.get("sparse").is_none(),
            "Missing or sparse accessor unsupported in this demo"
        );
        let view_index = a["bufferView"]
            .as_u64()
            .ok_or_else(|| anyhow!("Unsupported accessor"))? as usize;
        let v = &self.json["bufferViews"][view_index];
        ensure!(
            v["extensions"].get("EXT_meshopt_compression").is_none(),
            "Export with --uncompressed for this demo"
        );
        let bytes = self.slice(view_index)?;
        let width = match a["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            _ => bail!("Unsupported accessor"),
        };
        let kind = a["componentType"].as_u64().unwrap_or(0);
        let size = match kind {
            5121 => 1,
            5123 => 2,
            5125 | 5126 => 4,
            _ => bail!("Unsupported accessor"),
        };
        let count = a["count"]
            .as_u64()
            .ok_or_else(|| anyhow!("Unsupported accessor"))? as usize;
        let stride = v["byteStride"]
            .as_u64()
            .map_or(width * size, |s| s as usize);
        let off = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        ensure!(
            stride >= width * size
                && (count == 0 || off + (count - 1) * stride + width * size <= bytes.len()),
            "Accessor outside bufferView"
        );
        (0..count)
            .map(|i| {
                (0..width)
                    .map(|k| {
                        let p = off + i * stride + k * size;
                        let x = match kind {
                            5121 => bytes[p] as f64,
                            5123 => u16::from_le_bytes([bytes[p], bytes[p + 1]]) as f64,
                            5125 => u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as f64,
                            _ => f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as f64,
                        };
                        ensure!(x.is_finite(), "Non-finite mesh attribute");
                        Ok(x)
                    })
                    .collect()
            })
            .collect()
    }

    /// Hierarchy levels present in the asset, ascending.
    pub fn levels(&self) -> Vec<u8> {
        let mut levels: Vec<u8> = self.physics.fragments.iter().map(|f| f.level).collect();
        levels.sort_unstable();
        levels.dedup();
        levels
    }
}
