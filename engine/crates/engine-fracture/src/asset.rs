//! SHARDWRIGHT_fracture v1 GLB reader: a bounds-checked GLB container and
//! FlatBuffers (Shardwright `schemas/frac.fbs`, schema 1) decoder.
//!
//! Exposes full inertia tensors, bond frames/areas/moments, composition,
//! strength scales, reinforcement, boundary loops and child mappings, plus the
//! glTF render meshes and the embedded material library. Interface material
//! ids share the material id space:
//! `library.interface_material[id - library.material.len()]`; -1 is unspecified.
//! Every offset is checked; malformed input fails before any GPU allocation.

use anyhow::{anyhow, bail, ensure, Result};
use serde::Serialize;
use serde_json::Value;

pub const EXTENSION: &str = "SHARDWRIGHT_fracture";

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Fragment {
    pub id: u32,
    pub level: u8,
    pub parent: i32,
    pub child_begin: u32,
    pub child_count: u32,
    pub mass: f32,
    pub com: [f32; 3],
    /// Row-major 3x3 inertia about the COM.
    pub inertia: [f32; 9],
    pub material_ids: Vec<u16>,
    pub material_fractions: Vec<f32>,
    pub hull_begin: u32,
    pub hull_count: u32,
    pub particle_candidate: bool,
    pub role: u8,
    pub node: i32,
    pub volume: f32,
    pub interior_area: f32,
    pub component: u32,
    pub cell_begin: u32,
    pub cell_count: u32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Composition {
    pub kind: u8,
    pub interface_material: i32,
    pub fraction: f32,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bond {
    pub id: u32,
    pub level: u8,
    pub a: u32,
    /// Second fragment, or -1 for a world (anchor) bond.
    pub b: i32,
    pub area: f32,
    pub centroid: [f32; 3],
    pub normal: [f32; 3],
    pub planarity: f32,
    pub frame_u: [f32; 3],
    pub frame_v: [f32; 3],
    #[serde(rename = "iUU")]
    pub i_uu: f32,
    #[serde(rename = "iVV")]
    pub i_vv: f32,
    #[serde(rename = "iUV")]
    pub i_uv: f32,
    pub polar_moment: f32,
    pub extent_center: [f32; 3],
    pub extent_half: [f32; 2],
    pub dist_a: f32,
    pub dist_b: f32,
    pub composition: Vec<Composition>,
    pub rebar_count: u16,
    pub rebar_area: f32,
    pub rebar_direction: [f32; 3],
    pub strength_scale: f32,
    pub parent_bond: i32,
    pub child_begin: u32,
    pub child_count: u32,
    pub loop_begin: u32,
    pub loop_count: u32,
    pub spawn_begin: u32,
    pub spawn_count: u32,
    pub anchor: bool,
    pub interface_count: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Hull {
    pub fragment: u32,
    pub vertices: Vec<[f32; 3]>,
}

/// Baked face topology of one hull: `sizes[k]` indices per face.
#[derive(Clone, Debug, Serialize)]
pub struct HullFaces {
    pub indices: Vec<u32>,
    pub sizes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Spawn {
    pub p: [f32; 3],
    pub n: [f32; 3],
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Physics {
    pub schema_version: u32,
    /// u64 seed as a decimal string (JSON numbers cannot hold it).
    pub seed: String,
    pub variant: u32,
    pub level_fragment_begin: Vec<u32>,
    pub level_fragment_count: Vec<u32>,
    pub name: String,
    pub tool_version: String,
    pub settings_hash: String,
    pub material_library_id: String,
    pub material_library_version: String,
    pub levels: u8,
    pub fragments: Vec<Fragment>,
    pub bonds: Vec<Bond>,
    pub hulls: Vec<Hull>,
    pub hull_faces: Vec<HullFaces>,
    pub loops: Vec<Vec<[f32; 3]>>,
    pub spawn: Vec<Spawn>,
    pub bond_children: Vec<u32>,
}

pub struct FractureAsset {
    pub json: Value,
    pub physics: Physics,
    /// The embedded material library (`extensions.SHARDWRIGHT_fracture.materialLibrary`).
    pub library: Value,
    pub bytes: usize,
    bin: Vec<u8>,
}

/// Bounds-checked FlatBuffers reader.
struct Frac<'a> {
    b: &'a [u8],
}

impl Frac<'_> {
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
    fn u64(&self, p: usize) -> Result<u64> {
        let p = self.check(p, 8)?;
        Ok(u64::from_le_bytes(self.b[p..p + 8].try_into().unwrap()))
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
    fn get<T>(
        &self,
        t: usize,
        i: usize,
        fallback: T,
        read: impl Fn(&Self, usize) -> Result<T>,
    ) -> Result<T> {
        let p = self.field(t, i)?;
        if p == 0 {
            Ok(fallback)
        } else {
            read(self, p)
        }
    }
    fn u32_or(&self, t: usize, i: usize, fallback: u32) -> Result<u32> {
        self.get(t, i, fallback, Self::u32)
    }
    fn i32_or(&self, t: usize, i: usize, fallback: i32) -> Result<i32> {
        self.get(t, i, fallback, Self::i32)
    }
    fn u16_at(&self, t: usize, i: usize) -> Result<u16> {
        self.get(t, i, 0, Self::u16)
    }
    fn u8_at(&self, t: usize, i: usize) -> Result<u8> {
        self.get(t, i, 0, Self::u8)
    }
    fn f32_at(&self, t: usize, i: usize) -> Result<f32> {
        self.get(t, i, 0.0, Self::f32)
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
    /// (start, count) of a vector field whose elements are `stride` bytes.
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
    fn list<T>(
        &self,
        t: usize,
        i: usize,
        stride: usize,
        read: impl Fn(&Self, usize) -> Result<T>,
    ) -> Result<Vec<T>> {
        let (start, n) = self.vector(t, i, stride)?;
        (0..n).map(|k| read(self, start + k * stride)).collect()
    }
    fn vec3s(&self, t: usize, i: usize) -> Result<Vec<[f32; 3]>> {
        self.list(t, i, 12, |r, p| {
            Ok([r.f32(p)?, r.f32(p + 4)?, r.f32(p + 8)?])
        })
    }
    fn tables(&self, t: usize, i: usize) -> Result<Vec<usize>> {
        self.list(t, i, 4, |r, p| Ok(p + r.u32(p)? as usize))
    }
    fn string(&self, t: usize, i: usize) -> Result<String> {
        let (start, n) = self.vector(t, i, 1)?;
        Ok(std::str::from_utf8(&self.b[start..start + n])?.to_string())
    }
}

pub fn read_physics(bytes: &[u8]) -> Result<Physics> {
    ensure!(
        bytes.len() >= 8 && &bytes[4..8] == b"FRAC",
        "Missing FRAC identifier"
    );
    let r = Frac { b: bytes };
    let root = r.u32(0)? as usize;
    ensure!(r.u16_at(root, 0)? == 1, "Unsupported physics schema");

    let fragments = r
        .tables(root, 6)?
        .into_iter()
        .map(|t| {
            Ok(Fragment {
                id: r.u32_or(t, 0, 0)?,
                level: r.u8_at(t, 1)?,
                parent: r.i32_or(t, 2, -1)?,
                child_begin: r.u32_or(t, 3, 0)?,
                child_count: r.u32_or(t, 4, 0)?,
                mass: r.f32_at(t, 5)?,
                com: r.structf::<3>(t, 6)?,
                inertia: r.structf::<9>(t, 7)?,
                material_ids: r.list(t, 8, 2, Frac::u16)?,
                material_fractions: r.list(t, 9, 4, Frac::f32)?,
                hull_begin: r.u32_or(t, 10, 0)?,
                hull_count: r.u16_at(t, 11)? as u32,
                particle_candidate: r.u8_at(t, 12)? != 0,
                role: r.u8_at(t, 13)?,
                node: r.i32_or(t, 14, -1)?,
                volume: r.f32_at(t, 15)?,
                interior_area: r.f32_at(t, 16)?,
                component: r.u32_or(t, 17, 0)?,
                cell_begin: r.u32_or(t, 18, 0)?,
                cell_count: r.u32_or(t, 19, 0)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let bonds = r
        .tables(root, 7)?
        .into_iter()
        .map(|t| {
            let composition = r
                .tables(t, 18)?
                .into_iter()
                .map(|c| {
                    Ok(Composition {
                        kind: r.u8_at(c, 0)?,
                        interface_material: r.i32_or(c, 1, -1)?,
                        fraction: r.f32_at(c, 2)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Bond {
                id: r.u32_or(t, 0, 0)?,
                level: r.u8_at(t, 1)?,
                a: r.u32_or(t, 2, 0)?,
                b: r.i32_or(t, 3, 0)?,
                area: r.f32_at(t, 4)?,
                centroid: r.structf::<3>(t, 5)?,
                normal: r.structf::<3>(t, 6)?,
                planarity: r.f32_at(t, 7)?,
                frame_u: r.structf::<3>(t, 8)?,
                frame_v: r.structf::<3>(t, 9)?,
                i_uu: r.f32_at(t, 10)?,
                i_vv: r.f32_at(t, 11)?,
                i_uv: r.f32_at(t, 12)?,
                polar_moment: r.f32_at(t, 13)?,
                extent_center: r.structf::<3>(t, 14)?,
                extent_half: r.structf::<2>(t, 15)?,
                dist_a: r.f32_at(t, 16)?,
                dist_b: r.f32_at(t, 17)?,
                composition,
                rebar_count: r.u16_at(t, 19)?,
                rebar_area: r.f32_at(t, 20)?,
                rebar_direction: r.structf::<3>(t, 21)?,
                strength_scale: r.f32_at(t, 22)?,
                parent_bond: r.i32_or(t, 23, -1)?,
                child_begin: r.u32_or(t, 24, 0)?,
                child_count: r.u32_or(t, 25, 0)?,
                loop_begin: r.u32_or(t, 26, 0)?,
                loop_count: r.u16_at(t, 27)? as u32,
                spawn_begin: r.u32_or(t, 28, 0)?,
                spawn_count: r.u32_or(t, 29, 0)?,
                anchor: r.u8_at(t, 30)? != 0,
                interface_count: r.u32_or(t, 31, 0)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let hulls = r
        .tables(root, 8)?
        .into_iter()
        .map(|t| {
            Ok(Hull {
                fragment: r.u32_or(t, 1, 0)?,
                vertices: r.vec3s(t, 0)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let hull_faces = r
        .tables(root, 9)?
        .into_iter()
        .map(|t| {
            Ok(HullFaces {
                indices: r.list(t, 0, 4, Frac::u32)?,
                sizes: r.list(t, 1, 1, Frac::u8)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let loops = r
        .tables(root, 10)?
        .into_iter()
        .map(|t| r.vec3s(t, 0))
        .collect::<Result<Vec<_>>>()?;
    let spawn = r.list(root, 12, 24, |r, p| {
        Ok(Spawn {
            p: [r.f32(p)?, r.f32(p + 4)?, r.f32(p + 8)?],
            n: [r.f32(p + 12)?, r.f32(p + 16)?, r.f32(p + 20)?],
        })
    })?;
    let bond_children = r.list(root, 17, 4, Frac::u32)?;

    let n = fragments.len();
    for (i, f) in fragments.iter().enumerate() {
        ensure!(
            f.id as usize == i
                && f.mass >= 0.0
                && f.parent >= -1
                && (f.parent as i64) < n as i64
                && f.material_ids.len() == f.material_fractions.len()
                && (f.hull_begin + f.hull_count) as usize <= hulls.len()
                && (f.child_begin + f.child_count) as usize <= n,
            "Invalid fragment ranges"
        );
    }
    for (i, b) in bonds.iter().enumerate() {
        ensure!(
            b.id as usize == i
                && (b.a as usize) < n
                && b.b >= -1
                && (b.b as i64) < n as i64
                && b.area >= 0.0
                && (b.child_begin + b.child_count) as usize <= bond_children.len()
                && (b.loop_begin + b.loop_count) as usize <= loops.len()
                && (b.spawn_begin + b.spawn_count) as usize <= spawn.len(),
            "Invalid bond references"
        );
    }
    ensure!(hull_faces.len() == hulls.len(), "Missing hull faces");
    for (h, faces) in hulls.iter().zip(&hull_faces) {
        let total: usize = faces.sizes.iter().map(|&s| s as usize).sum();
        ensure!(
            (h.fragment as usize) < n
                && (4..=64).contains(&h.vertices.len())
                && total == faces.indices.len()
                && faces
                    .indices
                    .iter()
                    .all(|&v| (v as usize) < h.vertices.len()),
            "Invalid convex hull"
        );
    }
    Ok(Physics {
        schema_version: 1,
        seed: r.get(root, 3, 0, Frac::u64)?.to_string(),
        variant: r.u32_or(root, 14, 0)?,
        level_fragment_begin: r.list(root, 15, 4, Frac::u32)?,
        level_fragment_count: r.list(root, 16, 4, Frac::u32)?,
        name: r.string(root, 13)?,
        tool_version: r.string(root, 1)?,
        settings_hash: r.string(root, 2)?,
        material_library_id: r.string(root, 4)?,
        material_library_version: r.string(root, 5)?,
        levels: r.u8_at(root, 11)?,
        fragments,
        bonds,
        hulls,
        hull_faces,
        loops,
        spawn,
        bond_children,
    })
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
    let json = json.unwrap_or(Value::Null);
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
    ensure!(
        !json["extensionsRequired"]
            .as_array()
            .is_some_and(|r| r.iter().any(|e| e == "EXT_meshopt_compression")),
        "Export with --uncompressed for this demo"
    );
    ensure!(
        json["buffers"].as_array().is_some_and(|b| b.len() == 1
            && b[0].get("uri").is_none()
            && b[0]["byteLength"]
                .as_u64()
                .is_some_and(|n| n as usize <= bin.len()))
            && !json["images"]
                .as_array()
                .is_some_and(|i| i.iter().any(|i| i.get("uri").is_some())),
        "Expected a self-contained GLB"
    );
    let view = ext["bufferView"]
        .as_u64()
        .ok_or_else(|| anyhow!("Invalid bufferView"))? as usize;
    let library = ext["materialLibrary"].clone();
    let mut asset = FractureAsset {
        bytes: buffer.len(),
        bin: bin.to_vec(),
        json,
        physics: Physics::empty(),
        library,
    };
    asset.physics = read_physics(asset.slice(view)?)?;
    let (physics, library, json) = (&asset.physics, &asset.library, &asset.json);
    ensure!(
        library["library_id"] == physics.material_library_id.as_str()
            && library["library_version"] == physics.material_library_version.as_str(),
        "Mismatched material library"
    );
    let materials = asset.material_count();
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
    for b in &physics.bonds {
        for c in &b.composition {
            ensure!(
                c.interface_material < 0
                    || asset.interface_material(c.interface_material).is_some(),
                "Missing interface material"
            );
        }
    }
    Ok(asset)
}

impl Physics {
    fn empty() -> Self {
        Self {
            schema_version: 1,
            seed: String::new(),
            variant: 0,
            level_fragment_begin: vec![],
            level_fragment_count: vec![],
            name: String::new(),
            tool_version: String::new(),
            settings_hash: String::new(),
            material_library_id: String::new(),
            material_library_version: String::new(),
            levels: 0,
            fragments: vec![],
            bonds: vec![],
            hulls: vec![],
            hull_faces: vec![],
            loops: vec![],
            spawn: vec![],
            bond_children: vec![],
        }
    }
}

impl FractureAsset {
    fn slice(&self, index: usize) -> Result<&[u8]> {
        let v = &self.json["bufferViews"][index];
        let off = v.get("byteOffset").map_or(Some(0), Value::as_u64);
        let len = v["byteLength"].as_u64();
        let total = self.json["buffers"][0]["byteLength"].as_u64().unwrap_or(0) as usize;
        let (Some(off), Some(len)) = (off, len) else {
            bail!("Invalid bufferView")
        };
        let (off, len) = (off as usize, len as usize);
        ensure!(
            v["buffer"] == 0
                && off
                    .checked_add(len)
                    .is_some_and(|e| e <= total && e <= self.bin.len()),
            "Invalid bufferView"
        );
        Ok(&self.bin[off..off + len])
    }

    pub fn material_count(&self) -> usize {
        self.library["material"].as_array().map_or(0, Vec::len)
    }

    /// Physical material record of a fragment material id.
    pub fn material(&self, id: u16) -> &Value {
        &self.library["material"][id as usize]
    }

    /// Interface material record for an id in the shared material id space.
    pub fn interface_material(&self, id: i32) -> Option<&Value> {
        let k = (id as i64).checked_sub(self.material_count() as i64)?;
        self.library["interface_material"]
            .as_array()?
            .get(usize::try_from(k).ok()?)
    }

    /// Decode an accessor into `count` rows of `width` components.
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

    /// Fragments at `level`.
    pub fn level_fragments(&self, level: u8) -> usize {
        self.physics
            .fragments
            .iter()
            .filter(|f| f.level == level)
            .count()
    }

    /// `{physics, materialLibrary}` as JSON (the lab's metadata download).
    pub fn metadata_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "physics": self.physics,
            "materialLibrary": self.library,
        }))?)
    }
}
