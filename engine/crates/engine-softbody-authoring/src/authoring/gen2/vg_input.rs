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

mod rename_rig;
use rename_rig::*;

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
    /// The rear axle's wheel when it differs from the front's (the fields
    /// above are the front's).
    #[serde(default)]
    pub rear_wheel: Option<RearWheel>,
    /// The car's pose in the world: everything else here is in the body
    /// frame, which the rigger rigs in; the rig is posed afterwards.
    #[serde(default)]
    pub pose: Option<engine_vehiclegen::Pose>,
}

#[derive(Deserialize, Clone, Copy)]
pub struct RearWheel {
    pub wheel_y: f32,
    pub tyre_d: f32,
    pub tyre_w: f32,
    pub rim_d: f32,
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
        // The rigger works in the design's body frame: a posed design's
        // render model (built in the world) goes back through the pose.
        if let Some(ps) = spec.pose {
            unpose(&mut me.model, &ps);
        }
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
            let mut o = b.parts.clone();
            if let Some(ps) = spec.pose {
                // (the body frame, as the sheet build below)
                let r = ps.rotation().transpose();
                for (_, m) in o.0.iter_mut() {
                    let t = ps.invert(glam::DVec3::ZERO);
                    m.transform(Some(r), t);
                }
            }
            engine_vehiclegen::assemble::outer_parts(&o)
        } else {
            let o = engine_vehiclegen::BuildOpts {
                solid: false,
                posed: false,
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

/// A posed design's render model back in its body frame.
fn unpose(m: &mut Model, ps: &engine_vehiclegen::Pose) {
    let inv = |p: Vec3| {
        let q = ps.invert(glam::DVec3::new(p.x as f64, p.y as f64, p.z as f64));
        Vec3::new(q.x as f32, q.y as f32, q.z as f32)
    };
    let back = ps.rotation().transpose();
    for p in &mut m.parts {
        for q in &mut p.pos {
            *q = inv(*q);
        }
        for n in &mut p.nrm {
            let d = back * glam::DVec3::new(n.x as f64, n.y as f64, n.z as f64);
            *n = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
        }
    }
    let rot = Quat::from_rotation_x(ps.pitch as f32);
    for k in &mut m.markers {
        k.pos = inv(k.pos);
        k.rot = rot * k.rot;
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
    fn wheel_sizes(&self) -> Option<[(f32, f32); 2]> {
        let (f, r) = (&self.rig.dims, self.rig.dims.rear_wheel?);
        Some([(0.5 * f.tyre_d, f.tyre_w), (0.5 * r.tyre_d, r.tyre_w)])
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
    fn doors(&self) -> Vec<String> {
        self.rig.dims.doors.clone()
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
    let rig = rig_only(&input);
    let skin = bind_art(&input, &rig, &input.glb)?;
    Ok(Built { input, rig, skin })
}

/// The lattice alone (the physics): `build` less the binding of the art's skins.
pub fn rig_only(input: &VgInput) -> Rig {
    // (A design of defined parts alone, no unibody shell: `parts_rig`.)
    let mut rig = if super::parts_rig::applies(input) { super::parts_rig::rig(input) } else { rig::rig_with(input) };
    // (A posed design is rigged in its body frame and put in the world.)
    if let Some(ps) = input.rig.dims.pose {
        rig.transform(|g| {
            let q = ps.apply(glam::DVec3::new(g.x as f64, g.y as f64, g.z as f64));
            Vec3::new(q.x as f32, q.y as f32, q.z as f32)
        });
    }
    rig
}

/// An art `.glb`'s weighted skins on `rig`'s lattice (the design's own art, or a
/// lighter render of it: [`lite_art`]).
fn bind_art(input: &VgInput, rig: &Rig, glb: &Glb) -> Result<Skin, String> {
    let skin = gen3::skins_with(
        glb,
        &rig.car,
        input.rig.dims.axle_f,
        &VgInput::vg_part_of,
        // (a solid whose part's definition names its nodes: rigid as defined)
        &|m| match rig.car.bind.get(m) {
            Some(b) => b.rigid,
            None => VgInput::vg_rigid(m),
        },
    )?;
    skin.packable()?;
    Ok(skin)
}

/// The body mesh's target edge length of the lighter render ([`lite_art`], metres;
/// the design's own art is meshed at `engine_vehiclegen::BuildOpts::default().h`).
pub const LITE_EDGE: f64 = 0.035;

/// A lighter render of a Rust-generator design rigged as `b`: the same design
/// (`params`, as `VgInput::from_rust` took them) with its body meshed at edge `h`
/// (coarser than the art's), every part under the same name, and its own skins
/// bound to `b`'s lattice by the same binder. The physics (`b.rig`, the JBeam parts)
/// is not touched: only what is drawn is lighter. For devices that cannot hold the
/// full art (a phone's GPU memory); the page picks it only there.
/// Returns the `.glb` bytes and its skins.
pub fn lite_art(b: &Built, params: &str, h: f64) -> Result<(Vec<u8>, Skin), String> {
    lite_art_of(&b.input, &b.rig, params, h)
}

/// [`lite_art`] from the design's input and its lattice (no art skins needed).
pub fn lite_art_of(
    input: &VgInput,
    rig: &Rig,
    params: &str,
    h: f64,
) -> Result<(Vec<u8>, Skin), String> {
    use engine_vehiclegen::export::{glb, render_parts};
    let name = input.design();
    let d = engine_vehiclegen::design_from_json(name, params)?;
    let opts = engine_vehiclegen::BuildOpts {
        h,
        ..Default::default()
    };
    let built = engine_vehiclegen::build(d.as_ref(), opts);
    let bytes = glb(
        &render_parts(&built.parts),
        &built.markers,
        built.paint,
        built.metallic,
        built.roughness,
    );
    let lite = crate::glb::read(&bytes)?;
    // The same meshes, by name and order: the JBeam's flexbody rows name them.
    let names = |g: &Glb| g.meshes.iter().map(|m| m.name.clone()).collect::<Vec<_>>();
    let (lo, art) = (names(&lite), names(&input.glb));
    if lo != art {
        let only = |a: &[String], b: &[String]| {
            a.iter()
                .filter(|n| !b.contains(n))
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(format!(
            "{name}: the lighter render's meshes differ from the art's (only in the lighter: [{}]; only in the art: [{}]; {} and {} meshes)",
            only(&lo, &art),
            only(&art, &lo),
            lo.len(),
            art.len()
        ));
    }
    let skin = bind_art(input, rig, &lite)?;
    Ok((bytes, skin))
}

/// The lighter render's files beside the art's (`<design>.lite.glb`, `.lite.rskin`).
pub fn lite_glb_file(design: &str) -> String {
    format!("{design}.lite.glb")
}
pub fn lite_skin_file(design: &str) -> String {
    format!("{design}.lite.rskin")
}

/// The JBeam parts and the `.pc` alone (no render model, no skins): what the
/// solver reads. A crash measurement needs no more, and skips the binding.
pub fn physics_files(input: &VgInput, rig: &Rig) -> Vec<(String, Vec<u8>)> {
    let name = input.design().to_string();
    let axle_f = input.rig.dims.axle_f;
    let mut flex =
        gen3::flex_rows_with(&input.glb, axle_f, &VgInput::vg_part_of, &skin_file(&name));
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
    let mut groups = gen3::deform_groups_with(&input.glb, &rig.car, axle_f, &VgInput::vg_part_of);
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
    files_with(b, None)
}

/// [`files`], with a lighter render ([`lite_art`]) beside the art when given.
pub fn files_with(b: &Built, lite: Option<&(Vec<u8>, Skin)>) -> Vec<(String, Vec<u8>)> {
    let name = b.input.design().to_string();
    let mut out = physics_files(&b.input, &b.rig);
    out.push((
        format!("{name}/{}", glb_file(&name)),
        b.input.glb_bytes.clone(),
    ));
    out.push((format!("{name}/{}", skin_file(&name)), b.skin.encode()));
    if let Some((glb, skin)) = lite {
        out.push((format!("{name}/{}", lite_glb_file(&name)), glb.clone()));
        out.push((format!("{name}/{}", lite_skin_file(&name)), skin.encode()));
    }
    add_contact_offsets(b, &mut out);
    out.extend(super::crash_zones::file(&b.input, &b.rig));
    let list: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
    let index = serde_json::to_string(&list).unwrap();
    out.push(("files.json".into(), index.into_bytes()));
    out
}


/// The files a browser needs to drive a Rust-generator design from its parameters: the
/// physics (JBeam parts, `.pc`) and the lighter render ([`lite_art_of`], body edge `h`)
/// bound to the lattice, with `files.json` listing them. The same bytes as the CLI's
/// `<design>.lite.*` and physics files (`files_with`), without the full art and its
/// skins (140 MB the page would not draw). `params` is the JSON `VgInput::from_rust` takes.
pub fn web_files(design: &str, params: &str, h: f64) -> Result<Vec<(String, Vec<u8>)>, String> {
    let input = VgInput::from_rust(design, params)?;
    let rig = rig_only(&input);
    let lite = lite_art_of(&input, &rig, params, h)?;
    let name = input.design().to_string();
    let mut out = physics_files(&input, &rig);
    // (`physics_files` lists a part's path twice where two parts share a name: a folder
    // on disk keeps the later file, so the later one is the part. Same here.)
    let mut last: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, (p, _)) in out.iter().enumerate() {
        last.insert(p.clone(), i);
    }
    let mut i = 0;
    out.retain(|(p, _)| {
        i += 1;
        last[p] == i - 1
    });
    out.push((format!("{name}/{}", lite_glb_file(&name)), lite.0));
    out.push((format!("{name}/{}", lite_skin_file(&name)), lite.1.encode()));
    out.extend(super::crash_zones::file(&input, &rig));
    let list: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
    let index = serde_json::to_string(&list).unwrap();
    out.push(("files.json".into(), index.into_bytes()));
    Ok(out)
}

/// The most a node's contact surface sits out from it (m): past the generated sedan's
/// bumpers' and body sides' bulges between nodes (to 0.2 m), short of a part's own depth.
pub const CONTACT_OFFSET_CAP: f32 = 0.12;

/// Each skin node's contact offset (`authoring::contact_offset`: the contact surface at the
/// paint, not the straight triangles between the nodes) into its JBeam node row as Rubble's
/// option `rubbleContactOffset` (`RIG_CONTACT_OFFSET=0`: none, as before). From the car the
/// files describe and its bound render skin.
fn add_contact_offsets(b: &Built, out: &mut [(String, Vec<u8>)]) {
    if std::env::var("RIG_CONTACT_OFFSET").is_ok_and(|v| v == "0") {
        return;
    }
    let refs: Vec<(String, String)> = out
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(p, x)| (p.clone(), String::from_utf8_lossy(x).into_owned()))
        .collect();
    let pc = out.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, x)| String::from_utf8_lossy(x).into_owned());
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, x)| (a.as_str(), x.as_str())).collect();
    let (desc, flex) = match (
        engine_softbody::jbeam::load_jbeam(&r, pc.as_deref()),
        engine_softbody::jbeam::assembled_rows(&r, pc.as_deref(), "flexbodies"),
    ) {
        (Ok(d), Ok(f)) => (d, f),
        (d, f) => {
            eprintln!("contact offsets: the files do not load ({:?} / {:?})", d.err(), f.err());
            return;
        }
    };
    let Ok((sim, _)) = engine_softbody::explicit_beams::ExplicitBeams::from_desc_jbeam_raycast(
        &desc,
        Vec3::ZERO,
        Vec3::ZERO,
        crate::vehicle_compare::settings(),
    ) else {
        return;
    };
    let g = &b.input.glb;
    let meshes: Vec<_> = g.meshes.iter().map(|m| crate::flexskin::glb_mesh(g, m)).collect();
    let paint = crate::glb::paint_of(g).unwrap_or([0.5, 0.05, 0.04]);
    let vis = match crate::flexmesh::jbeam_visual_skinned(&flex, &[], &desc, &sim, &meshes, &Default::default(), paint, Some(&b.skin)) {
        Ok((v, _)) => v,
        Err(e) => {
            eprintln!("contact offsets: no visual ({e})");
            return;
        }
    };
    let offs = crate::authoring::contact_offset::contact_offsets(&desc, &vis, CONTACT_OFFSET_CAP);
    let by: std::collections::HashMap<&str, f32> =
        desc.nodes.iter().zip(&offs).filter(|(_, &o)| o > 5e-4).map(|(n, &o)| (n.name.as_str(), o)).collect();
    for (p, bytes) in out.iter_mut() {
        if !p.ends_with(".jbeam") {
            continue;
        }
        let text = String::from_utf8_lossy(bytes).into_owned();
        let mut res = String::with_capacity(text.len() + 1024);
        for line in text.split_inclusive('\n') {
            let t = line.trim_start();
            // A node row: ["name", x, ... {options}].
            let name = t.strip_prefix("[\"").and_then(|x| x.split('"').next());
            let after = name.map(|n| &t[n.len() + 4..]);
            let is_node = after.is_some_and(|a| a.starts_with(|c: char| c == '-' || c.is_ascii_digit()));
            match (name.and_then(|n| by.get(n)), is_node, line.find(",{")) {
                (Some(o), true, Some(k)) => {
                    res.push_str(&line[..k + 2]);
                    res.push_str(&format!("\"rubbleContactOffset\":{o:.4},"));
                    res.push_str(&line[k + 2..]);
                }
                _ => res.push_str(line),
            }
        }
        *bytes = res.into_bytes();
    }
}

