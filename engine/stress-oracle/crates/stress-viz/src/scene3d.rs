//! Draws one recorded frame into a [`Target`] (software or GPU raster) for one view
//! mode: flat-shaded chunk boxes with outlines, impactors, the ground plane, bond
//! overlays (crack lines and damage decals) and fragment velocity arrows.

use stress_ref::math::Vec3;
use stress_ref::scene::ImpactorShape;

use crate::camera::{Aabb, Camera};
use crate::colormap::{rgb8, scale, Colormap, Rgb};
use crate::raster::{object_id, Depth, OutlineStyle, Target};
use crate::record::{ChunkRec, Frame, HullMesh};
use crate::views::{fmt_number, nice_ceil, Scales, ViewMode, BROKEN};

/// Background gradient of the 3-D views.
pub const SKY_TOP: Rgb = [1.0, 1.0, 1.0];
pub const SKY_BOTTOM: Rgb = rgb8(0xe4, 0xe8, 0xee);
const METAL: Rgb = rgb8(0x9a, 0x9e, 0xa6);
const GROUND: Rgb = rgb8(0xe0, 0xe2, 0xe6);
const GRID: Rgb = rgb8(0xc4, 0xc8, 0xd0);
const ARROW: Rgb = rgb8(0x20, 0x22, 0x28);
pub const CRACK_DARK: Rgb = rgb8(0x14, 0x16, 0x1c);
/// Object number of the ground plane (chunks and impactors count up from 1).
const GROUND_OBJECT: u32 = (u32::MAX >> 3) - 1;

const OUTLINES: OutlineStyle = OutlineStyle { silhouette: 0.55, object: 0.38, crease: 0.22 };
const AMBIENT: f32 = 0.58;
const DIFFUSE: f32 = 0.47;

/// Hides chunks on one side of an axis-aligned plane (cut-away / section view).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clip {
    pub axis: usize,
    /// Hide points with `p[axis] > value` (else `< value`).
    pub hide_above: bool,
    pub value: f64,
}

impl Clip {
    /// Parses `x>0.1`, `y<0`, ... (hide the side the comparison selects).
    pub fn parse(s: &str) -> Result<Clip, String> {
        let axis = match s.chars().next() {
            Some('x') => 0,
            Some('y') => 1,
            Some('z') => 2,
            _ => return Err(format!("--clip '{s}': expected x|y|z followed by > or < and a value")),
        };
        let hide_above = match s.chars().nth(1) {
            Some('>') => true,
            Some('<') => false,
            _ => return Err(format!("--clip '{s}': expected > or < after the axis")),
        };
        let value = s[2..].parse().map_err(|e| format!("--clip '{s}': {e}"))?;
        Ok(Clip { axis, hide_above, value })
    }

    pub fn hides(&self, p: [f32; 3]) -> bool {
        let v = p[self.axis] as f64;
        if self.hide_above {
            v > self.value
        } else {
            v < self.value
        }
    }
}

/// Per-rendering drawing options.
#[derive(Clone, Debug)]
pub struct DrawOptions {
    /// Overlay broken bonds as crack lines in the heat-map views.
    pub cracks: bool,
    /// Crack line colour; default white, dark on the light-centred stress map.
    pub crack_color: Option<Rgb>,
    /// Velocity arrows per fragment in the fragment view.
    pub arrows: bool,
    pub clip: Option<Clip>,
    /// Output pixels per raster sample row (supersampling factor).
    pub ss: usize,
}

/// Static context of a panel: the camera, scales and scene-dependent sizes.
pub struct PanelScene<'a> {
    pub camera: &'a Camera,
    pub mode: ViewMode,
    pub scales: &'a Scales,
    pub opts: &'a DrawOptions,
    pub ground: Option<f64>,
    /// Initial scene bounds (ground extent, arrow lengths).
    pub bounds: Aabb,
    /// Overlay depth tolerance (m).
    pub bias: f32,
}

fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}

fn col(r: &[[f32; 3]; 3], k: usize) -> Vec3 {
    Vec3::new(r[0][k] as f64, r[1][k] as f64, r[2][k] as f64)
}

impl PanelScene<'_> {
    fn view(&self, p: Vec3) -> [f32; 3] {
        let v = self.camera.world_to_view(p);
        [v.x as f32, v.y as f32, v.z as f32]
    }

    /// Key light: from the upper left, slightly behind the viewer.
    fn light(&self) -> Vec3 {
        let c = self.camera;
        (c.forward * -0.6 + Vec3::new(0.0, 0.0, 0.65) + c.right * -0.3).normalized()
    }

    fn lambert(&self, base: Rgb, n: Vec3) -> Rgb {
        scale(base, AMBIENT + DIFFUSE * n.dot(self.light()).max(0.0) as f32)
    }

    /// Draws `frame` into `r` (clears it first).
    pub fn draw(&self, r: &mut dyn Target, frame: &Frame) {
        r.clear(SKY_TOP, SKY_BOTTOM);
        if let Some(h) = self.ground {
            self.draw_ground(r, h);
        }
        for (i, c) in frame.chunks.iter().enumerate() {
            if self.opts.clip.is_some_and(|k| k.hides(c.center)) {
                continue;
            }
            let d = self.mode.displacement(c, self.scales);
            let center = v3(c.center) + v3(d);
            let color = self.mode.chunk_color(c, self.scales);
            match &c.hull {
                Some(mesh) => self.draw_hull(r, center, &c.rotation, mesh, color, i as u32 + 1),
                None => self.draw_box(r, center, &c.rotation, c.half_extents, color, i as u32 + 1),
            }
        }
        let first_impactor = frame.chunks.len() as u32 + 1;
        for (j, imp) in frame.impactors.iter().enumerate() {
            let obj = first_impactor + j as u32;
            match imp.shape {
                ImpactorShape::Sphere { radius } => self.draw_sphere(r, v3(imp.center), radius, obj),
                ImpactorShape::Box { half_extents } => {
                    let h = [half_extents[0] as f32, half_extents[1] as f32, half_extents[2] as f32];
                    self.draw_box(r, v3(imp.center), &imp.rotation, h, METAL, obj)
                }
            }
        }
        r.outline(OUTLINES);
        self.draw_bonds(r, frame);
        if self.mode == ViewMode::Fragments && self.opts.arrows {
            self.draw_arrows(r, &frame.chunks);
        }
    }

    fn draw_box(&self, r: &mut dyn Target, center: Vec3, rot: &[[f32; 3]; 3], h: [f32; 3], base: Rgb, obj: u32) {
        for k in 0..3 {
            let (u, v) = (col(rot, (k + 1) % 3), col(rot, (k + 2) % 3));
            let (hu, hv) = (h[(k + 1) % 3] as f64, h[(k + 2) % 3] as f64);
            for (s, sign) in [(0u32, -1.0), (1, 1.0)] {
                let n = col(rot, k) * sign;
                let fc = center + n * h[k] as f64;
                if n.dot(self.camera.eye - fc) <= 0.0 {
                    continue;
                }
                let pts = [
                    self.view(fc - u * hu - v * hv),
                    self.view(fc + u * hu - v * hv),
                    self.view(fc + u * hu + v * hv),
                    self.view(fc - u * hu + v * hv),
                ];
                r.polygon(&pts, self.lambert(base, n), Depth::Write { id: object_id(obj, 2 * k as u32 + s) });
            }
        }
    }

    /// A convex hull chunk: each face polygon facing the camera, flat shaded.
    fn draw_hull(&self, r: &mut dyn Target, center: Vec3, rot: &[[f32; 3]; 3], mesh: &HullMesh, base: Rgb, obj: u32) {
        let world = |p: [f32; 3]| center + col(rot, 0) * p[0] as f64 + col(rot, 1) * p[1] as f64 + col(rot, 2) * p[2] as f64;
        for (k, (normal, poly)) in mesh.iter().enumerate() {
            let n = col(rot, 0) * normal[0] as f64 + col(rot, 1) * normal[1] as f64 + col(rot, 2) * normal[2] as f64;
            let fc = world(poly[0]);
            if n.dot(self.camera.eye - fc) <= 0.0 {
                continue;
            }
            let pts: Vec<[f32; 3]> = poly.iter().map(|p| self.view(world(*p))).collect();
            r.polygon(&pts, self.lambert(base, n), Depth::Write { id: object_id(obj, k as u32) });
        }
    }

    /// Faceted sphere with a soft specular highlight (metallic look).
    fn draw_sphere(&self, r: &mut dyn Target, c: Vec3, radius: f64, obj: u32) {
        const STACKS: usize = 18;
        const SLICES: usize = 32;
        let point = |i: usize, j: usize| {
            let th = std::f64::consts::PI * i as f64 / STACKS as f64;
            let ph = 2.0 * std::f64::consts::PI * j as f64 / SLICES as f64;
            Vec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos())
        };
        let light = self.light();
        for i in 0..STACKS {
            for j in 0..SLICES {
                let q = [point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)];
                let n = (q[0] + q[1] + q[2] + q[3]).normalized();
                let fc = c + n * radius;
                let to_eye = (self.camera.eye - fc).normalized();
                if n.dot(to_eye) <= 0.0 {
                    continue;
                }
                let half = (light + to_eye).normalized();
                let spec = 0.45 * n.dot(half).max(0.0).powi(28) as f32;
                let base = self.lambert(METAL, n);
                let shade = [base[0] + spec, base[1] + spec, base[2] + spec];
                let pts: Vec<[f32; 3]> = q.iter().map(|&p| self.view(c + p * radius)).collect();
                r.polygon(&pts, shade, Depth::Write { id: object_id(obj, 0) });
            }
        }
    }

    fn draw_ground(&self, r: &mut dyn Target, h: f64) {
        let s = self.bounds.size();
        let half = 1.5 * s.x.max(s.y).max(0.5 * s.z).max(0.1);
        let c = self.bounds.center();
        let corner = |dx: f64, dy: f64| self.view(Vec3::new(c.x + dx * half, c.y + dy * half, h));
        let pts = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
        r.polygon(&pts, GROUND, Depth::Write { id: object_id(GROUND_OBJECT, 0) });
        let step = nice_ceil(half / 8.0);
        let n = (half / step).ceil() as i64;
        let (x0, y0) = ((c.x / step).round() * step, (c.y / step).round() * step);
        let w = self.opts.ss as f32;
        for i in -n..=n {
            let x = x0 + i as f64 * step;
            let y = y0 + i as f64 * step;
            let bias = Depth::Test { bias: self.bias };
            r.line(self.view(Vec3::new(x, c.y - half, h)), self.view(Vec3::new(x, c.y + half, h)), w, GRID, bias);
            r.line(self.view(Vec3::new(c.x - half, y, h)), self.view(Vec3::new(c.x + half, y, h)), w, GRID, bias);
        }
    }

    /// Crack lines (broken bonds) in the heat-map views; damage decals in the damage
    /// view. A patch outline shows where the joint meets a visible surface; a filled
    /// patch shows where a fracture face is exposed.
    fn draw_bonds(&self, r: &mut dyn Target, frame: &Frame) {
        let damage_view = self.mode == ViewMode::Damage;
        if !damage_view && (!self.opts.cracks || self.mode == ViewMode::Deformation) {
            return;
        }
        let ss = self.opts.ss as f32;
        let depth = Depth::Test { bias: self.bias };
        let crack_color = self.opts.crack_color.unwrap_or(if self.mode == ViewMode::Stress { CRACK_DARK } else { [1.0; 3] });
        for b in &frame.bonds {
            if self.opts.clip.is_some_and(|k| k.hides(b.centroid)) {
                continue;
            }
            let (color, width, fill) = if damage_view {
                let c = if b.broken { BROKEN } else { Colormap::Heat.sample(b.damage as f64) };
                (c, 2.0 * ss, true)
            } else if b.broken {
                (crack_color, 1.25 * ss, false)
            } else {
                continue;
            };
            let (n, t) = (v3(b.normal), v3(b.tangent));
            let (bt, c) = (n.cross(t), v3(b.centroid));
            let (hw, hh) = (0.5 * b.width[0] as f64, 0.5 * b.width[1] as f64);
            let q = [
                self.view(c - t * hw - bt * hh),
                self.view(c + t * hw - bt * hh),
                self.view(c + t * hw + bt * hh),
                self.view(c - t * hw + bt * hh),
            ];
            if fill {
                r.polygon(&q, color, depth);
            }
            for k in 0..4 {
                r.line(q[k], q[(k + 1) % 4], width, color, depth);
            }
        }
    }

    /// One arrow per moving fragment, from its centroid along its mean velocity.
    fn draw_arrows(&self, r: &mut dyn Target, chunks: &[ChunkRec]) {
        let mut acc: std::collections::BTreeMap<u64, (Vec3, Vec3, f64)> = Default::default();
        for c in chunks.iter().filter(|c| !c.anchored) {
            if self.opts.clip.is_some_and(|k| k.hides(c.center)) {
                continue;
            }
            let e = acc.entry(c.cluster).or_insert((Vec3::ZERO, Vec3::ZERO, 0.0));
            e.0 += v3(c.center);
            e.1 += v3(c.velocity);
            e.2 += 1.0;
        }
        let length = 0.15 * self.bounds.size().norm() / self.scales.speed;
        let ss = self.opts.ss as f32;
        for (p, v, n) in acc.values() {
            let (p, v) = (*p / *n, *v / *n);
            if v.norm() < 0.05 * self.scales.speed {
                continue;
            }
            let tip = p + v * length;
            let dir = v.normalized();
            let side = dir.cross(self.camera.forward).normalized();
            let head = 0.25 * (v.norm() * length).min(0.3 * self.bounds.size().norm());
            r.line(self.view(p), self.view(tip), 1.5 * ss, ARROW, Depth::Always);
            for s in [-1.0, 1.0] {
                let wing = tip - dir * head + side * (0.45 * head * s);
                r.line(self.view(tip), self.view(wing), 1.5 * ss, ARROW, Depth::Always);
            }
        }
    }
}

/// Panel label suffix for a clip (e.g. `section x>0`).
pub fn clip_label(c: &Clip) -> String {
    format!("section {}{}{}", ["x", "y", "z"][c.axis], if c.hide_above { '>' } else { '<' }, fmt_number(c.value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraSpec;
    use crate::raster::Raster;
    use crate::record::BondRec;

    fn unit_chunk(x: f32) -> ChunkRec {
        ChunkRec {
            center: [x, 0.0, 0.0],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            half_extents: [0.5; 3],
            cluster: 7,
            anchored: false,
            velocity: [0.0; 3],
            deformation: [0.0; 3],
            von_mises: 0.0,
            principal: 0.0,
            utilization: 0.5,
            hull: None,
        }
    }

    #[test]
    fn clip_parsing() {
        assert_eq!(Clip::parse("y>0.25").unwrap(), Clip { axis: 1, hide_above: true, value: 0.25 });
        assert!(Clip::parse("x<-1").unwrap().hides([-2.0, 0.0, 0.0]));
        assert!(Clip::parse("w>1").is_err());
        assert!(Clip::parse("x=1").is_err());
    }

    #[test]
    fn two_chunks_render_with_a_dark_joint_and_a_crack() {
        let bounds = Aabb { lo: Vec3::new(-1.0, -0.5, -0.5), hi: Vec3::new(1.0, 0.5, 0.5) };
        let spec = CameraSpec { yaw_deg: 0.0, pitch_deg: 0.0, ..Default::default() };
        let cam = spec.build(&bounds, 2.0);
        let scales = Scales { stress: 1.0, von_mises: 1.0, speed: 1.0, deformation: 1.0, exaggeration: 1.0 };
        let opts = DrawOptions { cracks: true, crack_color: None, arrows: false, clip: None, ss: 1 };
        let panel = PanelScene { camera: &cam, mode: ViewMode::Utilization, scales: &scales, opts: &opts, ground: None, bounds, bias: 0.1 };
        let mut frame = Frame {
            time: 0.0,
            chunks: vec![unit_chunk(-0.5), unit_chunk(0.5)],
            bonds: vec![],
            impactors: vec![],
            broken_bonds: 0,
            mechanical: 0.0,
            dissipated: 0.0,
        };
        let (w, h) = (200, 100);
        let mut r = Raster::new(w, h, cam.projection(w, h));
        panel.draw(&mut r, &frame);
        let face = r.color_at(50, 50);
        let joint = r.color_at(100, 50);
        assert_ne!(r.id_at(50, 50), r.id_at(150, 50), "two chunks, two ids");
        assert!(joint.iter().sum::<f32>() < face.iter().sum::<f32>(), "joint darker: {joint:?} vs {face:?}");
        assert!(r.color_at(2, 0).iter().zip(SKY_TOP).all(|(a, b)| (a - b).abs() < 1e-2), "background corner");
        // A broken bond at the joint draws a white crack line there.
        frame.bonds.push(BondRec {
            centroid: [0.0; 3],
            normal: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            width: [1.0, 1.0],
            broken: true,
            damage: 1.0,
        });
        panel.draw(&mut r, &frame);
        assert_eq!(r.color_at(100, 50), [1.0; 3]);
    }
}
