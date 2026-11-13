//! A vehicle-gen design as the rigger's input ([`super::rig::RigInput`]):
//! the art `.glb` (part meshes and the art spec's markers) and the design's
//! `rig.json` (`engine/tools/vehicle-gen/vehgen/rigexport.py`): its key lines,
//! the body tube's sections, the nose and tail faces, the wheel arches, the
//! running gear's dimensions and its `PHYSICS` block.
//!
//! The body surface is the design's own: a signed field from the tube's
//! section at z (the left half's polyline, mirrored) cut by the end faces
//! (`front_z`, `rear_z`). Section rays (`side`, `top`, `corner`) march it as
//! gen3 marches the Rust surface. The belt and shoulder are the design's key
//! lines; the arches are its wheel openings.
//!
//! [`build`] rigs the design and binds the art `.glb` to it; [`files`] writes
//! the mod folder (JBeam parts, `.pc`, the `.glb`, the `.rskin`), named after
//! the design. Guide: `engine/docs/generated-vehicles.md`.

use super::build::Part;
use super::gen3;
use super::model::{Marker, Model, Part as MPart};
use super::rig::{self, Dims, Physics, Rig, RigInput};
use crate::flexskin::Skin;
use crate::glb::Glb;
use glam::{Quat, Vec3};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
pub struct RigDims {
    pub wheelbase: f32,
    pub track: f32,
    pub axle_f: f32,
    pub axle_r: f32,
    pub tyre_d: f32,
    pub tyre_w: f32,
    pub rim_d: f32,
    pub wheel_y: f32,
    pub z_front: f32,
    pub z_rear: f32,
    pub doors: Vec<String>,
    pub rear_closure: String,
}

#[derive(Deserialize)]
pub struct Arch {
    pub center: [f32; 2],
    #[serde(rename = "R")]
    pub r: f32,
}

#[derive(Deserialize)]
pub struct Keys {
    pub z0: f32,
    pub dz: f32,
    pub n: usize,
    /// Per key line, per station: (x, y).
    pub lines: BTreeMap<String, Vec<[f32; 2]>>,
}

#[derive(Deserialize)]
pub struct Sections {
    pub z0: f32,
    pub dz: f32,
    pub nz: usize,
    pub dv: f32,
    pub nv: usize,
    /// Per station, per v: (x, y) of the tube's left half.
    pub xy: Vec<Vec<[f32; 2]>>,
}

#[derive(Deserialize)]
pub struct Ends {
    pub x0: f32,
    pub y0: f32,
    pub d: f32,
    pub nx: usize,
    pub ny: usize,
    pub front_z: Vec<Vec<f32>>,
    pub rear_z: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct MarkerJson {
    pub pos: [f32; 3],
    pub axis: [f32; 3],
}

/// `<design>.rig.json`.
#[derive(Deserialize)]
pub struct RigJson {
    pub format: u32,
    pub design: String,
    pub base: Option<String>,
    pub title: String,
    pub dims: RigDims,
    pub arches: Vec<Arch>,
    pub keys: Keys,
    pub zlines: Vec<(String, f32)>,
    pub vlines: Vec<f32>,
    pub sections: Sections,
    pub ends: Ends,
    /// Each part's welded skin boundary loops (glTF points).
    pub skins: BTreeMap<String, Vec<Vec<[f32; 3]>>>,
    pub markers: BTreeMap<String, MarkerJson>,
    #[serde(default)]
    pub physics: serde_json::Value,
}

/// A vehicle-gen design, ready to rig.
pub struct VgInput {
    pub rig: RigJson,
    pub model: Model,
    pub glb: Glb,
    /// The art `.glb` as built (written to the mod unchanged).
    pub glb_bytes: Vec<u8>,
    pub physics: Physics,
    ray_y: f32,
    /// The key line the rig's middle panel row follows (`shoulder_y`).
    crease: String,
    /// The design's body-in-white and its mechanical and interior parts
    /// (`engine_vehiclegen::RigSpec`), when it defines them.
    pub structure: Option<engine_vehiclegen::structure::Structure>,
    pub mechanical: Vec<engine_vehiclegen::mechanical::MechPart>,
}

impl VgInput {
    /// Reads `<dir>/<design>.glb` and `<dir>/<design>.rig.json`.
    pub fn load(dir: &Path, design: &str) -> Result<Self, String> {
        let gp = dir.join(format!("{design}.glb"));
        let rp = dir.join(format!("{design}.rig.json"));
        let glb_bytes = std::fs::read(&gp).map_err(|e| format!("{}: {e}", gp.display()))?;
        if glb_bytes.starts_with(b"version https://git-lfs") {
            return Err(format!(
                "{} is a Git LFS pointer: run `git lfs pull` or rebuild the design",
                gp.display()
            ));
        }
        let text = std::fs::read_to_string(&rp).map_err(|e| format!("{}: {e}", rp.display()))?;
        let rig: RigJson =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", rp.display()))?;
        Self::new(rig, glb_bytes)
    }

    pub fn new(rig: RigJson, glb_bytes: Vec<u8>) -> Result<Self, String> {
        if rig.format != 1 {
            return Err(format!("rig.json format {} (this reads 1)", rig.format));
        }
        let glb = crate::glb::read(&glb_bytes)?;
        let physics: Physics = if rig.physics.is_null() {
            Physics::default()
        } else {
            serde_json::from_value(rig.physics.clone())
                .map_err(|e| format!("{}: PHYSICS: {e}", rig.design))?
        };
        let mut model = model_of(&glb);
        // A tailgate's markers under the boot lid's names (the rigger's).
        if rig.dims.rear_closure != "trunk_lid" {
            let rc = rig.dims.rear_closure.clone();
            let alias: Vec<Marker> = model
                .markers
                .iter()
                .filter_map(|m| {
                    let n = m.name.replace(&format!("_{rc}"), "_trunk");
                    (n != m.name).then(|| Marker {
                        name: n,
                        pos: m.pos,
                        rot: m.rot,
                    })
                })
                .collect();
            model.markers.extend(alias);
        }
        for name in rig.markers.keys() {
            if !model.markers.iter().any(|m| &m.name == name) {
                return Err(format!("rig.json marker {name} is not in the .glb"));
            }
        }
        let mut me = Self {
            rig,
            model,
            glb,
            glb_bytes,
            physics,
            ray_y: 0.0,
            crease: String::new(),
            structure: None,
            mechanical: Vec::new(),
        };
        // The section rays start between the sill and the belt at mid car.
        me.ray_y = 0.5 * (me.key("sill", 0.0)[1] + me.key("belt", 0.0)[1]);
        // The rig's side crease (gen3's shoulder line: the middle row of the
        // doors', fenders' and quarters' grids) is the design's character
        // line nearest the middle of the door at mid car, between its bottom
        // and the belt: the waist crease on the sedan (its shoulder line runs
        // 6 cm under the belt, where the grid's rule drops a row).
        let mid = 0.5 * (me.key("door_bottom", 0.0)[1] + me.key("belt", 0.0)[1]);
        me.crease = ["waist", "shoulder"]
            .into_iter()
            .filter(|k| me.rig.keys.lines.contains_key(*k))
            .min_by(|a, b| {
                (me.key(a, 0.0)[1] - mid)
                    .abs()
                    .total_cmp(&(me.key(b, 0.0)[1] - mid).abs())
            })
            .unwrap_or("shoulder")
            .to_string();
        Ok(me)
    }

    /// A design of the Rust generator (`engine-vehiclegen`), built in memory
    /// from its parameters (JSON, any subset; the rest the design's
    /// defaults): its meshes and markers, and the same data as a Python
    /// design's `rig.json` taken straight from the build (tube, key lines,
    /// skins, the design's [`engine_vehiclegen::RigSpec`] and physics).
    pub fn from_rust(name: &str, params: &str) -> Result<Self, String> {
        use engine_vehiclegen::export::{glb, render_parts};
        let d = engine_vehiclegen::design_from_json(name, params)?;
        // `VG_SHEET=1`: the legacy sheet parts (single-sided skins, hems,
        // loose inner panels), to compare a binding against.
        let opts = engine_vehiclegen::BuildOpts {
            solid: std::env::var("VG_SHEET").map_or(true, |v| v != "1"),
            ..Default::default()
        };
        let (b, cb) = engine_vehiclegen::build_with_body(d.as_ref(), opts);
        // The art: every panel and pane a closed solid (skin, edge walls,
        // inner face; `engine_vehiclegen::solid`). The flexbody binds all of
        // it; `spec.surfaces` tags every vertex outer / edge / inner.
        let rp = render_parts(&b.parts);
        let spec = engine_vehiclegen::assemble::rig_spec_with(d.as_ref(), &rp)
            .ok_or_else(|| format!("design {name} has no rig spec"))?;
        let bytes = glb(&rp, &b.markers, b.paint, b.metallic, b.roughness);
        let rig = rig_json_of(name, d.as_ref(), &cb, &b.markers, &spec);
        let mut me = Self::new(rig, bytes)?;
        me.structure = spec.structure.as_deref().cloned();
        me.mechanical = spec.mechanical.as_ref().clone();
        if !opts.solid {
            return Ok(me);
        }
        // The lattice and the collision skin do not follow the solid's edge
        // walls and inner faces. The rigger's model (same parts, same order)
        // is the design's legacy sheets: the same outer skins with their
        // hems and loose inner panels, which the rigger's edge nodes and
        // areas were built on, so the rig is the one the sheet car had.
        // `VG_RIG_OUTER=1`: each solid's `Outer` faces only (the bare
        // skins); the hood's edge nodes then drift 16 mm parked
        // (`gen3_rig`), so it waits for the rigger to read the tags itself.
        let sheets = if std::env::var("VG_RIG_OUTER").is_ok_and(|v| v == "1") {
            engine_vehiclegen::assemble::outer_parts(&b.parts)
        } else {
            let o = engine_vehiclegen::BuildOpts {
                solid: false,
                ..opts
            };
            engine_vehiclegen::build(d.as_ref(), o).parts
        };
        let outer = glb(
            &render_parts(&sheets),
            &b.markers,
            b.paint,
            b.metallic,
            b.roughness,
        );
        let parts = model_of(&crate::glb::read(&outer)?).parts;
        if !me.defined() {
            // (The rigger's own body: the sheet build's parts alone, as
            // before the design defined its structure.)
            me.model.parts = parts;
            return Ok(me);
        }
        // The sheet build has no body-in-white and no mechanical parts: the
        // solid build's stay in the rigger's model under their own names
        // (same order as the `.glb`).
        let mut sheet: std::collections::HashMap<String, MPart> =
            parts.into_iter().map(|p| (p.name.clone(), p)).collect();
        for p in &mut me.model.parts {
            if let Some(s) = sheet.remove(&p.name) {
                *p = s;
            }
        }
        Ok(me)
    }

    /// A design by name: the Rust generator's when it has it (`params` its
    /// JSON parameters), else a Python build's `.glb` and `rig.json` from
    /// `dir`, or by default `engine/tools/vehicle-gen/out/<design>` (a local
    /// build) then `engine/scenes/vehicles/<design>` (a committed one).
    pub fn find(design: &str, params: Option<&str>, dir: Option<&Path>) -> Result<Self, String> {
        if dir.is_none() && engine_vehiclegen::default_params_json(design).is_ok() {
            return Self::from_rust(design, params.unwrap_or(""));
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = match dir {
            Some(d) => d.to_path_buf(),
            None => {
                let built = root.join("tools/vehicle-gen/out").join(design);
                if built.join(format!("{design}.rig.json")).exists() {
                    built
                } else {
                    root.join("scenes/vehicles").join(design)
                }
            }
        };
        Self::load(&dir, design)
    }

    pub fn design(&self) -> &str {
        &self.rig.design
    }

    /// The rigger builds the body and the mechanical parts from the
    /// design's definitions (`RIG_OLD_BODY=1`: its own members instead).
    pub fn defined(&self) -> bool {
        self.structure.is_some() && std::env::var("RIG_OLD_BODY").map_or(true, |v| v != "1")
    }

    /// A key line's (x, y) at z.
    pub fn key(&self, line: &str, z: f32) -> [f32; 2] {
        let k = &self.rig.keys;
        let v = k
            .lines
            .get(line)
            .unwrap_or_else(|| panic!("rig.json has no key line {line}"));
        let t = ((z - k.z0) / k.dz).clamp(0.0, (k.n - 1) as f32);
        let i = (t.floor() as usize).min(k.n - 2);
        let f = t - i as f32;
        [
            v[i][0] + (v[i + 1][0] - v[i][0]) * f,
            v[i][1] + (v[i + 1][1] - v[i][1]) * f,
        ]
    }

    /// The tube's signed 2-D distance at (x ≥ 0, y) on section z: negative
    /// inside the section (the left half's polyline closed on the centre line).
    fn section_d(&self, x: f32, y: f32, z: f32) -> f32 {
        let s = &self.rig.sections;
        let t = ((z - s.z0) / s.dz).clamp(0.0, (s.nz - 1) as f32);
        let i = (t.floor() as usize).min(s.nz - 2);
        let f = t - i as f32;
        let (ra, rb) = (&s.xy[i], &s.xy[i + 1]);
        let at = |k: usize| -> (f32, f32) {
            (
                ra[k][0] + (rb[k][0] - ra[k][0]) * f,
                ra[k][1] + (rb[k][1] - ra[k][1]) * f,
            )
        };
        let mut best = f32::MAX;
        let mut inside = false;
        let mut p0 = at(0);
        for k in 1..s.nv {
            let p1 = at(k);
            // Distance to the segment.
            let (ex, ey) = (p1.0 - p0.0, p1.1 - p0.1);
            let l2 = ex * ex + ey * ey;
            let u = if l2 > 0.0 {
                (((x - p0.0) * ex + (y - p0.1) * ey) / l2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (dx, dy) = (x - (p0.0 + ex * u), y - (p0.1 + ey * u));
            best = best.min(dx * dx + dy * dy);
            // Crossings of the ray toward +x.
            if (p0.1 > y) != (p1.1 > y) {
                let xc = p0.0 + (y - p0.1) / (p1.1 - p0.1) * ex;
                if xc > x {
                    inside = !inside;
                }
            }
            p0 = p1;
        }
        let d = best.sqrt();
        if inside {
            -d
        } else {
            d
        }
    }

    /// An end face's z at (x ≥ 0, y) (bilinear on its table) and its slopes.
    fn end_z(&self, front: bool, x: f32, y: f32) -> (f32, f32, f32) {
        let e = &self.rig.ends;
        let tab = if front { &e.front_z } else { &e.rear_z };
        let tx = ((x - e.x0) / e.d).clamp(0.0, (e.nx - 1) as f32);
        let ty = ((y - e.y0) / e.d).clamp(0.0, (e.ny - 1) as f32);
        let (i, j) = (
            (tx.floor() as usize).min(e.nx - 2),
            (ty.floor() as usize).min(e.ny - 2),
        );
        let (a, b) = (tx - i as f32, ty - j as f32);
        let (z00, z10, z01, z11) = (tab[i][j], tab[i + 1][j], tab[i][j + 1], tab[i + 1][j + 1]);
        let z =
            z00 * (1.0 - a) * (1.0 - b) + z10 * a * (1.0 - b) + z01 * (1.0 - a) * b + z11 * a * b;
        let zx = ((z10 - z00) * (1.0 - b) + (z11 - z01) * b) / e.d;
        let zy = ((z01 - z00) * (1.0 - a) + (z11 - z10) * a) / e.d;
        (z, zx, zy)
    }

    /// vehicle-gen's rigid pieces: gen3's, and the wheel-well liners
    /// (moulded plastic clipped into the arch: skinned, the liner beside the
    /// rear suspension's tower warped 4.6 mm parked).
    pub fn vg_rigid(mesh: &str) -> bool {
        gen3::rigid(mesh) || mesh.starts_with("wheel_well_")
    }

    /// The physics part a vehicle-gen mesh rides.
    pub fn vg_part_of(mesh: &str) -> Option<Part> {
        if mesh.starts_with("glass_quarter_") {
            // vehicle-gen's quarter light is in the body's C-pillar, not in
            // the rear door's frame (gen3's model).
            return Some(Part::Body);
        }
        if mesh == "tailgate" || mesh.starts_with("glass_tailgate") {
            return Some(Part::Trunk);
        }
        gen3::part_of(mesh)
    }
}

/// A `.glb`'s meshes and markers as a render model (world space).
pub fn model_of(g: &Glb) -> Model {
    let mut parts = Vec::new();
    for m in &g.meshes {
        let mut p = MPart {
            name: m.name.clone(),
            ..Default::default()
        };
        for pr in &m.prims {
            let base = p.pos.len() as u32;
            p.pos.extend(pr.pos.iter().map(|q| Vec3::from_array(*q)));
            p.nrm.extend(pr.nrm.iter().map(|q| Vec3::from_array(*q)));
            for t in pr.idx.chunks_exact(3) {
                p.tris.push([base + t[0], base + t[1], base + t[2]]);
            }
        }
        parts.push(p);
    }
    let markers = g
        .markers
        .iter()
        .map(|k| Marker {
            name: k.name.clone(),
            pos: Vec3::from_array(k.pos),
            rot: Quat::from_array(k.rot),
        })
        .collect();
    Model { parts, markers }
}

impl RigInput for VgInput {
    fn model(&self) -> &Model {
        &self.model
    }
    fn dims(&self) -> Dims {
        let d = &self.rig.dims;
        Dims {
            axle_f: d.axle_f,
            axle_r: d.axle_r,
            tyre_r: d.wheel_y,
            track_half: 0.5 * d.track,
            arch_r: self.rig.arches.first().map_or(0.4, |a| a.r),
        }
    }
    fn physics(&self) -> &Physics {
        &self.physics
    }
    fn part_of(&self, mesh: &str) -> Option<Part> {
        Self::vg_part_of(mesh)
    }
    fn sdf(&self, q: Vec3) -> f32 {
        let (x, y, z) = (q.x.abs(), q.y, q.z);
        let ds = self.section_d(x, y, z);
        let (zf, fx, fy) = self.end_z(true, x, y);
        let df = (z - zf) / (1.0 + fx * fx + fy * fy).sqrt();
        let (zr, rx, ry) = self.end_z(false, x, y);
        let dr = (zr - z) / (1.0 + rx * rx + ry * ry).sqrt();
        ds.max(df).max(dr)
    }
    fn belt(&self, z: f32) -> f32 {
        self.key("belt", z)[1]
    }
    fn shoulder_y(&self, z: f32) -> f32 {
        self.key(&self.crease, z)[1]
    }
    fn arch_y(&self, z: f32) -> Option<f32> {
        for a in &self.rig.arches {
            let d = z - a.center[0];
            if d.abs() <= a.r {
                return Some(a.center[1] + (a.r * a.r - d * d).max(0.0).sqrt());
            }
        }
        None
    }
    fn end_pt(&self, front: bool, x: f32, y: f32) -> Vec3 {
        Vec3::new(x, y, self.end_z(front, x, y).0)
    }
    fn ray_y(&self) -> f32 {
        self.ray_y
    }
    /// The greenhouse's corner is the design's roof-rail key line (the
    /// daylight opening's top, down the A- and C-pillars), not gen3's
    /// "normal 50° from up" (a raked windscreen leans past that at its
    /// centre, so the search found the belt's shoulder).
    fn theta_corner(&self, z: f32) -> f32 {
        let [x, y] = self.key("roof_rail", z);
        x.atan2(y - self.ray_y)
    }
    fn hull_triangles(&self) -> bool {
        true
    }
    fn rear_closure(&self) -> String {
        self.rig.dims.rear_closure.clone()
    }
    fn definitions(&self) -> Option<super::body2::Defs<'_>> {
        self.structure.as_ref().map(|s| super::body2::Defs {
            structure: s,
            parts: &self.mechanical,
        })
    }
    fn seam_points(&self) -> Option<Vec<(Vec3, Vec3, usize, usize)>> {
        // Points two parts' welded skins share (the skins are cut from one
        // surface, so a shut line's points coincide exactly).
        let key = |p: &[f32; 3]| {
            (
                (p[0] * 1e4).round() as i64,
                (p[1] * 1e4).round() as i64,
                (p[2] * 1e4).round() as i64,
            )
        };
        let mut at: std::collections::HashMap<(i64, i64, i64), Vec<usize>> = Default::default();
        let mesh = |name: &str| self.model.parts.iter().position(|p| p.name == name);
        for (name, loops) in &self.rig.skins {
            let Some(k) = mesh(name) else { continue };
            for lp in loops {
                for p in lp {
                    let e = at.entry(key(p)).or_default();
                    if !e.contains(&k) {
                        e.push(k);
                    }
                }
            }
        }
        let mut out = Vec::new();
        let mut keys: Vec<_> = at.keys().copied().collect();
        keys.sort();
        for kk in keys {
            let ks = &at[&kk];
            let q = Vec3::new(kk.0 as f32, kk.1 as f32, kk.2 as f32) * 1e-4;
            for &a in ks {
                for &b in ks {
                    if a != b {
                        out.push((q, q, a, b));
                    }
                }
            }
        }
        Some(out)
    }
}

/// The skin sidecar's and render model's file names for a design.
pub fn skin_file(design: &str) -> String {
    format!("{design}.rskin")
}
pub fn glb_file(design: &str) -> String {
    format!("{design}.glb")
}

/// A design rigged and bound.
pub struct Built {
    pub input: VgInput,
    pub rig: Rig,
    pub skin: Skin,
}

/// Rig a design and bind its art `.glb` to the lattice.
pub fn build(input: VgInput) -> Result<Built, String> {
    let rig = rig::rig_with(&input);
    let axle_f = input.rig.dims.axle_f;
    let skin = gen3::skins_with(
        &input.glb,
        &rig.car,
        axle_f,
        &VgInput::vg_part_of,
        // (a solid whose part's definition names its nodes: rigid as defined)
        &|m| match rig.car.bind.get(m) {
            Some(b) => b.rigid,
            None => VgInput::vg_rigid(m),
        },
    )?;
    skin.packable()?;
    Ok(Built { input, rig, skin })
}

/// The JBeam parts and the `.pc` alone (no render model, no skins): what the
/// solver reads. A crash measurement needs no more, and skips the binding.
pub fn physics_files(input: &VgInput, rig: &Rig) -> Vec<(String, Vec<u8>)> {
    let name = input.design().to_string();
    let axle_f = input.rig.dims.axle_f;
    let mut flex = gen3::flex_rows_with(
        &input.glb,
        axle_f,
        &VgInput::vg_part_of,
        &skin_file(&name),
    );
    // A definition part's solid is listed in that part's own JBeam part.
    for (_, f) in &mut flex {
        if let Some(owner) = rig.car.bind.get(&f.mesh).and_then(|x| x.owner.clone()) {
            f.groups = vec![super::export::sub_name(&owner)];
            f.owner = Some(owner);
        }
    }
    // (A pane with trigger beams of its own keeps those.)
    let own: Vec<&str> = rig
        .car
        .fixed
        .values()
        .filter_map(|f| f.deform_group.as_ref().map(|g| g.0.as_str()))
        .collect();
    let mut groups =
        gen3::deform_groups_with(&input.glb, &rig.car, axle_f, &VgInput::vg_part_of);
    groups.retain(|g| !own.contains(&g.0.as_str()));
    super::export::jbeam_files(&rig.car, &flex, &groups)
        .into_iter()
        .map(|(p, t)| (rename_path(&p, &name), rename(&t, &name).into_bytes()))
        .collect()
}

/// The mod folder's files (paths relative to it): `<design>/` JBeam parts and
/// `.pc` (parts named `<design>_body` …), the art `.glb` and its `.rskin`,
/// then `files.json`.
pub fn files(b: &Built) -> Vec<(String, Vec<u8>)> {
    let name = b.input.design().to_string();
    let mut out = physics_files(&b.input, &b.rig);
    out.push((
        format!("{name}/{}", glb_file(&name)),
        b.input.glb_bytes.clone(),
    ));
    out.push((format!("{name}/{}", skin_file(&name)), b.skin.encode()));
    let list: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
    let index = serde_json::to_string(&list).unwrap();
    out.push(("files.json".into(), index.into_bytes()));
    out
}

/// gen2's export names (vehicle `gen2`, parts `gen2_*`) as the design's.
fn rename(s: &str, name: &str) -> String {
    s.replace("gen2_", &format!("{name}_"))
        .replace("\"gen2\"", &format!("\"{name}\""))
        .replace("Rubble gen2 sedan", &format!("Rubble {name} (vehicle-gen)"))
}

/// `gen2/gen2_body.jbeam` as `<name>/<name>_body.jbeam`.
fn rename_path(p: &str, name: &str) -> String {
    p.replace("gen2", name)
}

/// The rig data of a Rust-generated design (what `rigexport.py` writes for a
/// Python one), in memory.
fn rig_json_of(
    name: &str,
    d: &dyn engine_vehiclegen::Design,
    cb: &engine_vehiclegen::carbody::CarBody,
    markers: &[engine_vehiclegen::export::Marker],
    spec: &engine_vehiclegen::RigSpec,
) -> RigJson {
    const KEYLINES: [&str; 10] = [
        "floor",
        "sill",
        "door_bottom",
        "widest",
        "waist",
        "shoulder",
        "belt",
        "roof_rail",
        "glass_edge",
        "centre",
    ];
    let (dz, dv, dxy) = (0.02f64, 1.0 / 16.0, 0.02f64);
    let tube = &cb.l.tube;
    let z0 = spec.z_rear - 0.06;
    let nz = (((spec.z_front + 0.06) - z0) / dz + 0.5).floor() as usize + 1;
    let zs: Vec<f64> = (0..nz).map(|i| z0 + dz * i as f64).collect();
    let f = |x: f64| x as f32;
    let mut lines = BTreeMap::new();
    for (k, n) in KEYLINES.iter().enumerate().take(tube.nk) {
        lines.insert(
            n.to_string(),
            zs.iter()
                .map(|&z| {
                    let p = tube.key(k, z);
                    [f(p.x), f(p.y)]
                })
                .collect(),
        );
    }
    let nv = ((tube.nk - 1) as f64 / dv).round() as usize + 1;
    let xy: Vec<Vec<[f32; 2]>> = zs
        .iter()
        .map(|&z| {
            (0..nv)
                .map(|j| {
                    let p = tube.eval(z, dv * j as f64);
                    [f(p.x), f(p.y)]
                })
                .collect()
        })
        .collect();
    let (nx, ny) = (61usize, 111usize);
    let tab = |g: &dyn Fn(f64, f64) -> f64| -> Vec<Vec<f32>> {
        (0..nx)
            .map(|i| {
                (0..ny)
                    .map(|j| f(g(dxy * i as f64, dxy * j as f64)))
                    .collect()
            })
            .collect()
    };
    let skins = cb
        .skins
        .iter()
        .map(|(n, m)| {
            let loops = engine_vehiclegen::panels::boundary_loops(m)
                .into_iter()
                .filter(|l| l.len() >= 3)
                .map(|l| {
                    l.iter()
                        .map(|&i| {
                            let p = m.v[i as usize];
                            // 0.1 mm, as rigexport.py rounds (shut lines'
                            // points coincide across parts).
                            let r = |x: f64| ((x * 1e4).round() * 1e-4) as f32;
                            [r(p.x), r(p.y), r(p.z)]
                        })
                        .collect()
                })
                .collect();
            (n.clone(), loops)
        })
        .collect();
    let mk = markers
        .iter()
        .map(|m| {
            (
                m.name.clone(),
                MarkerJson {
                    pos: [f(m.pos.x), f(m.pos.y), f(m.pos.z)],
                    axis: [f(m.xaxis.x), f(m.xaxis.y), f(m.xaxis.z)],
                },
            )
        })
        .collect();
    RigJson {
        format: 1,
        design: name.to_string(),
        base: None,
        title: d.name(),
        dims: RigDims {
            wheelbase: f(spec.wheelbase),
            track: f(spec.track),
            axle_f: f(spec.axle_f),
            axle_r: f(spec.axle_r),
            tyre_d: f(spec.tyre_d),
            tyre_w: f(spec.tyre_w),
            rim_d: f(spec.rim_d),
            wheel_y: f(spec.wheel_y),
            z_front: f(spec.z_front),
            z_rear: f(spec.z_rear),
            doors: spec.doors.clone(),
            rear_closure: spec.rear_closure.clone(),
        },
        arches: spec
            .arches
            .iter()
            .map(|a| Arch {
                center: [f(a.0), f(a.1)],
                r: f(a.2),
            })
            .collect(),
        keys: Keys {
            z0: f(z0),
            dz: f(dz),
            n: nz,
            lines,
        },
        zlines: d
            .zlines()
            .iter()
            .map(|(n, z)| (n.to_string(), f(*z)))
            .collect(),
        vlines: d.vlines().iter().map(|&v| f(v)).collect(),
        sections: Sections {
            z0: f(z0),
            dz: f(dz),
            nz,
            dv: f(dv),
            nv,
            xy,
        },
        ends: Ends {
            x0: 0.0,
            y0: 0.0,
            d: f(dxy),
            nx,
            ny,
            front_z: tab(&|x, y| (spec.front_z)(x, y)),
            rear_z: tab(&|x, y| (spec.rear_z)(x, y)),
        },
        skins,
        markers: mk,
        physics: serde_json::to_value(&spec.physics).unwrap(),
    }
}
