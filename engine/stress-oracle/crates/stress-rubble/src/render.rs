//! Offscreen video renderer for the GPU world coupled to rubble (`stress_gpu::world::GpuWorld`
//! with [`crate::RubbleEngine::with_device`] on the solver's device), or any `ReferenceSolver`
//! state: every chunk drawn as its own convex hull (or box) at the pose the solver gives it
//! (cluster rotation, hidden rotation, chunk rotation; `World::chunk_box`), the impactors,
//! and a ground plane.
//! Lambert shading under a key light and a sky/ground ambient, 4x MSAA, written in WGSL
//! directly (no Slang). Frames are read back as RGBA (sRGB) for an overlay drawn on the
//! host (`Overlay`) and piped into ffmpeg (`FfmpegSink`).
//!
//! Colours: a chunk takes its material's colour while its cluster still stands on a
//! support; once its cluster is free (debris, a toppled piece) the colour is mixed with a
//! palette colour hashed from the cluster id, so each fragment reads as one piece.
//!
//! Views (the fracture lab's): `colors` picks what a chunk is painted by ([`ColorMode`]):
//! the material (above), its body's state as the lab's own renderer does (moving orange, at
//! rest blue, static grey; the coupled engine keeps every body awake, so "at rest" is a body
//! slower than `REST_SPEED`), or its body, one colour each, kept across splits so a
//! structure is seen breaking up. `bond_graph` draws each intact bond as a line between its
//! chunks' centres, over the geometry, gone once the bond breaks.
//!
//! Nothing here touches the simulation: it reads the host mirror after a frame.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::process::{Child, Command, Stdio};

use stress_gpu::gpu::Gpu;
use stress_ref::math::{Mat3, Quat, Vec3};
use stress_ref::scene::ImpactorShape;
use stress_ref::solver::ReferenceSolver;
use stress_ref::world::Impactor;

use wgpu::util::DeviceExt as _;

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
};
struct Object {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> objects: array<Object>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) object: u32) -> VsOut {
    let o = objects[object];
    let p = vec4<f32>(pos, 1.0);
    let world = vec3<f32>(dot(o.r0, p), dot(o.r1, p), dot(o.r2, p));
    let n = vec3<f32>(dot(o.r0.xyz, normal), dot(o.r1.xyz, normal), dot(o.r2.xyz, normal));
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = n;
    out.color = o.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    let to_eye = normalize(camera.eye.xyz - in.world);
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }
    var base = in.color.rgb;
    let dist = length(camera.eye.xyz - in.world);
    // The ground grid's screen-space widths, outside the branch (WGSL takes derivatives
    // only in uniform control flow).
    let g = in.world.xy;
    let w1 = fwidth(g);
    let w5 = fwidth(g / 5.0);
    if (in.color.a > 1.5) {
        // Ground: a 1 m grid with 5 m major lines, faded with distance.
        let f1 = abs(fract(g + 0.5) - 0.5) / w1;
        let l1 = 1.0 - clamp(min(f1.x, f1.y), 0.0, 1.0);
        let f5 = abs(fract(g / 5.0 + 0.5) - 0.5) / w5;
        let l5 = 1.0 - clamp(min(f5.x, f5.y), 0.0, 1.0);
        let fade = clamp(1.0 - dist / camera.light.w, 0.0, 1.0);
        base = mix(base, base * 0.78, max(0.5 * l1, l5) * fade);
    }
    let l = normalize(camera.light.xyz);
    let diffuse = max(dot(n, l), 0.0);
    let sky = mix(vec3<f32>(0.33, 0.31, 0.29), vec3<f32>(0.52, 0.58, 0.68), 0.5 + 0.5 * n.z);
    var c = base * (sky + vec3<f32>(1.0, 0.96, 0.88) * 0.85 * diffuse);
    // Aerial perspective toward the sky colour.
    let haze = clamp(dist / (2.5 * camera.light.w), 0.0, 0.6);
    c = mix(c, vec3<f32>(0.70, 0.76, 0.84), haze * haze);
    return vec4<f32>(c, 1.0);
}
"#;

/// The bond graph's lines (`Renderer::bond_graph`): drawn over the geometry.
const LINE_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn line_vs(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> LineOut {
    var out: LineOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn line_fs(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// What [`Renderer::colors`] paints a chunk by (the fracture lab's View menu; the same
/// numbering as `fracture_draw.slang`'s colour mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorMode {
    /// The material's colour; a free fragment's mixed with a colour of its cluster.
    #[default]
    Material,
    /// The body's state: moving, at rest, static.
    Sleep,
    /// One colour per body (cluster), kept as it breaks ([`BodyColors`]).
    Bodies,
}

impl ColorMode {
    /// 0 material, 1 sleep state, 2 bodies; anything else is material.
    pub fn from_index(i: u32) -> Self {
        match i {
            1 => Self::Sleep,
            2 => Self::Bodies,
            _ => Self::Material,
        }
    }
}

/// [`ColorMode::Bodies`]: a colour slot per cluster id. A split retires its parent and gives
/// every child a new id; the child with most of the parent's chunks keeps the parent's slot
/// and the others take new ones, so a body keeps its colour while pieces break off it.
#[derive(Default)]
pub struct BodyColors {
    slot: HashMap<u64, u32>,
    next: u32,
}

impl BodyColors {
    /// Follow the solver's clusters since the last call. O(clusters) when nothing split.
    pub fn update(&mut self, solver: &ReferenceSolver) {
        self.follow(solver.clusters.iter().map(|c| (c.id, c.parent, c.chunks.len())));
    }

    /// [`Self::update`] from each cluster's (id, parent id, chunk count).
    pub fn follow(&mut self, clusters: impl Iterator<Item = (u64, Option<u64>, usize)> + Clone) {
        let mut fresh: Vec<(u64, Option<u64>, usize)> = clusters.clone().filter(|(id, ..)| !self.slot.contains_key(id)).collect();
        if fresh.is_empty() {
            return;
        }
        let present: HashSet<u64> = clusters.map(|(id, ..)| id).collect();
        // Heirs: per retired parent, its largest child (ties: the lower id).
        fresh.sort_unstable_by_key(|&(id, ..)| id);
        let mut heir: HashMap<u64, (usize, u64)> = HashMap::new();
        for &(id, parent, n) in &fresh {
            if let Some(p) = parent.filter(|p| self.slot.contains_key(p) && !present.contains(p)) {
                let e = heir.entry(p).or_insert((n, id));
                if n > e.0 {
                    *e = (n, id);
                }
            }
        }
        let inherited: HashMap<u64, u32> = heir.iter().map(|(p, &(_, id))| (id, self.slot[p])).collect();
        for &(id, ..) in &fresh {
            let slot = inherited.get(&id).copied().unwrap_or_else(|| {
                self.next += 1;
                self.next - 1
            });
            self.slot.insert(id, slot);
        }
        self.slot.retain(|id, _| present.contains(id));
    }

    /// The colour of cluster `id` (after [`Self::update`]).
    pub fn color(&self, id: u64) -> [f32; 3] {
        slot_color(self.slot.get(&id).copied().unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A body keeps its colour as pieces break off it; each piece takes a new one, and a
    /// piece's own pieces follow the same rule.
    #[test]
    fn body_colors_survive_splits() {
        let mut b = BodyColors::default();
        b.follow([(0, None, 100), (1, None, 40)].into_iter());
        let (c0, c1) = (b.color(0), b.color(1));
        assert_ne!(c0, c1);
        // Body 0 splits into 2 (90 chunks), 3 (6) and 4 (4); body 1 is untouched.
        b.follow([(1, None, 40), (2, Some(0), 90), (3, Some(0), 6), (4, Some(0), 4)].into_iter());
        assert_eq!(b.color(2), c0, "the largest piece keeps the body's colour");
        assert_eq!(b.color(1), c1);
        let (c3, c4) = (b.color(3), b.color(4));
        for c in [c3, c4] {
            assert!(c != c0 && c != c1);
        }
        assert_ne!(c3, c4);
        // Nothing new: nothing changes.
        b.follow([(1, None, 40), (2, Some(0), 90), (3, Some(0), 6), (4, Some(0), 4)].into_iter());
        assert_eq!([b.color(1), b.color(2), b.color(3), b.color(4)], [c1, c0, c3, c4]);
        // Body 2 splits evenly: the lower id keeps it.
        b.follow([(1, None, 40), (3, Some(0), 6), (4, Some(0), 4), (5, Some(2), 45), (6, Some(2), 45)].into_iter());
        assert_eq!(b.color(5), c0);
        assert!(![c0, c1, c3, c4].contains(&b.color(6)));
    }

    #[test]
    fn slot_colors_are_distinct() {
        let colors: Vec<[f32; 3]> = (0..64).map(slot_color).collect();
        for (i, a) in colors.iter().enumerate() {
            for b in &colors[..i] {
                assert!(a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() > 1e-3);
            }
        }
    }
}

/// A distinct colour per slot: hues a golden-ratio step apart.
pub fn slot_color(slot: u32) -> [f32; 3] {
    let h = (slot as f64 * 0.618_033_988_749_895).fract() * 6.0;
    let (s, v) = (0.62, 0.92);
    let f = h.fract();
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let rgb = match h as u32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    };
    rgb.map(|x| x as f32)
}

/// The colours of `ColorMode::Sleep` (the fracture lab's: `fracture_draw.slang`).
const MOVING: [f32; 3] = [1.0, 0.45, 0.12];
const AT_REST: [f32; 3] = [0.25, 0.45, 0.85];
const STATIC: [f32; 3] = [0.45, 0.45, 0.48];
/// A body slower than this (m/s, and rad/s) shows as at rest in `ColorMode::Sleep`.
pub const REST_SPEED: f64 = 0.05;
/// A bond's line in `bond_graph`.
const BOND_LINE: [f32; 4] = [1.0, 0.86, 0.25, 1.0];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    object: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Object {
    rows: [[f32; 4]; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    /// Key light direction (towards the light) and, in `w`, the fade distance (m).
    light: [f32; 4],
}

/// A camera orbiting `target` (Z up): `yaw` (rad, from +x), `pitch` above the horizon,
/// `distance` (m), vertical field of view `fov` (rad).
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub target: Vec3,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub fov: f64,
}

impl CameraPose {
    pub fn eye(&self) -> Vec3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        self.target + Vec3::new(self.yaw.cos() * cp, self.yaw.sin() * cp, sp) * self.distance
    }

    fn view_proj(&self, aspect: f64) -> [[f32; 4]; 4] {
        let eye = self.eye();
        let f = (self.target - eye).normalized();
        let s = f.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
        let u = s.cross(f);
        // Column-major view (right-handed, looking down -z) and projection (depth 0..1).
        let view = [
            [s.x, u.x, -f.x, 0.0],
            [s.y, u.y, -f.y, 0.0],
            [s.z, u.z, -f.z, 0.0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        ];
        let (near, far) = (0.05 * self.distance.min(20.0), 50.0 * self.distance);
        let t = 1.0 / (0.5 * self.fov).tan();
        let proj = [[t / aspect, 0.0, 0.0, 0.0], [0.0, t, 0.0, 0.0], [0.0, 0.0, far / (near - far), -1.0], [0.0, 0.0, near * far / (near - far), 0.0]];
        std::array::from_fn(|c| std::array::from_fn(|r| (0..4).map(|k| proj[k][r] * view[c][k]).sum::<f64>() as f32))
    }
}

/// The colour of a calibrated material (stress-ref material names).
pub fn material_color(name: &str) -> [f32; 3] {
    // A pack's chunk at its own mass is `<material>#mass<node>` (`scene_pack`).
    let name = name.split('#').next().unwrap_or(name);
    match name {
        // Vehicles (`stress_ref::vehicles`): a red body shell, black wheels, a white airframe.
        "sheet-steel" => [0.72, 0.09, 0.08],
        "wheel" => [0.10, 0.10, 0.11],
        "aluminium" => [0.90, 0.91, 0.93],
        "brick" => [0.66, 0.33, 0.24],
        "stone" => [0.74, 0.70, 0.62],
        "glass" => [0.58, 0.74, 0.84],
        "steel" => [0.50, 0.53, 0.58],
        "timber" => [0.66, 0.50, 0.32],
        "mortar" => [0.80, 0.78, 0.74],
        _ => [0.80, 0.79, 0.76],
    }
}

/// A palette colour for a fragment (cluster id).
fn fragment_color(id: u64) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 10] = [
        [0.90, 0.45, 0.20],
        [0.25, 0.55, 0.85],
        [0.35, 0.72, 0.38],
        [0.85, 0.30, 0.45],
        [0.95, 0.75, 0.25],
        [0.55, 0.40, 0.80],
        [0.20, 0.70, 0.70],
        [0.80, 0.55, 0.35],
        [0.60, 0.80, 0.30],
        [0.90, 0.40, 0.70],
    ];
    let mut h = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    PALETTE[(h % PALETTE.len() as u64) as usize]
}

/// Flat-shaded triangles of a convex hull (faces as vertex index loops in any order).
fn hull_triangles(vertices: &[Vec3], faces: &[(Vec<usize>, Vec3)], object: u32, out: &mut Vec<Vertex>) {
    for (idx, normal) in faces {
        if idx.len() < 3 {
            continue;
        }
        let centroid = idx.iter().fold(Vec3::ZERO, |a, &i| a + vertices[i]) * (1.0 / idx.len() as f64);
        let e1 = (vertices[idx[0]] - centroid).normalized();
        let e2 = normal.cross(e1);
        let mut ring: Vec<(f64, usize)> = idx
            .iter()
            .map(|&i| {
                let d = vertices[i] - centroid;
                (d.dot(e2).atan2(d.dot(e1)), i)
            })
            .collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n = [normal.x as f32, normal.y as f32, normal.z as f32];
        let v = |i: usize| {
            let p = vertices[i];
            Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], normal: n, object }
        };
        for k in 1..ring.len() - 1 {
            out.push(v(ring[0].1));
            out.push(v(ring[k].1));
            out.push(v(ring[k + 1].1));
        }
    }
}

/// One chunk's flat-shaded triangles in its own frame (centred, chunk axes): each hull of
/// a compound chunk at its offset, its convex hull, or its box when it has none.
/// The centre of a mesh's bounding box (its object's frame) and the radius about it.
fn mesh_bounds(vertices: &[Vertex]) -> [f32; 4] {
    if vertices.is_empty() {
        return [0.0; 4];
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(v.pos[k]);
            hi[k] = hi[k].max(v.pos[k]);
        }
    }
    let c: [f32; 3] = std::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
    let r = (0..3).map(|k| (hi[k] - c[k]).powi(2)).sum::<f32>().sqrt();
    [c[0], c[1], c[2], r]
}

fn chunk_mesh(ch: &stress_ref::structure::ChunkData, object: u32, out: &mut Vec<Vertex>) {
    if let Some(parts) = &ch.parts {
        for p in parts.iter() {
            let vertices: Vec<Vec3> = p.hull.vertices.iter().map(|v| *v + p.offset).collect();
            let faces: Vec<(Vec<usize>, Vec3)> = p.hull.faces.iter().map(|f| (f.vertices.clone(), f.normal)).collect();
            hull_triangles(&vertices, &faces, object, out);
        }
        return;
    }
    match &ch.hull {
        Some(h) => {
            let faces: Vec<(Vec<usize>, Vec3)> = h.faces.iter().map(|f| (f.vertices.clone(), f.normal)).collect();
            hull_triangles(&h.vertices, &faces, object, out);
        }
        None => {
            let (v, f) = box_faces(ch.half_extents);
            hull_triangles(&v, &f, object, out);
        }
    }
}

/// [`chunk_mesh`] as (position, face normal) per vertex, three per triangle, for other
/// renderers (`destruction-view` draws these at the GPU solver's chunk poses).
pub fn chunk_triangles(ch: &stress_ref::structure::ChunkData) -> Vec<([f32; 3], [f32; 3])> {
    let mut verts = Vec::new();
    chunk_mesh(ch, 0, &mut verts);
    verts.into_iter().map(|v| (v.pos, v.normal)).collect()
}

fn box_faces(h: Vec3) -> (Vec<Vec3>, Vec<(Vec<usize>, Vec3)>) {
    let verts: Vec<Vec3> = (0..8).map(|k| Vec3::new(if k & 1 == 0 { -h.x } else { h.x }, if k & 2 == 0 { -h.y } else { h.y }, if k & 4 == 0 { -h.z } else { h.z })).collect();
    let mut faces = Vec::new();
    for axis in 0..3 {
        for sign in [-1.0f64, 1.0] {
            let bit = 1usize << axis;
            let idx: Vec<usize> = (0..8).filter(|&k| ((k & bit) != 0) == (sign > 0.0)).collect();
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            faces.push((idx, n));
        }
    }
    (verts, faces)
}

fn sphere_triangles(radius: f64, object: u32, out: &mut Vec<Vertex>) {
    let (rings, segs) = (12usize, 20usize);
    let point = |i: usize, j: usize| {
        let th = std::f64::consts::PI * i as f64 / rings as f64;
        let ph = 2.0 * std::f64::consts::PI * j as f64 / segs as f64;
        Vec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos())
    };
    let v = |n: Vec3| Vertex { pos: [(n.x * radius) as f32, (n.y * radius) as f32, (n.z * radius) as f32], normal: [n.x as f32, n.y as f32, n.z as f32], object };
    for i in 0..rings {
        for j in 0..segs {
            let (a, b, c, d) = (point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1));
            out.extend([v(a), v(b), v(c), v(a), v(c), v(d)]);
        }
    }
}

fn rows(r: Mat3, t: Vec3) -> [[f32; 4]; 3] {
    std::array::from_fn(|i| [r.m[i][0] as f32, r.m[i][1] as f32, r.m[i][2] as f32, t[i] as f32])
}

/// Renders the mirror of a solver (and its impactors) offscreen at a fixed size, or into a
/// target of the caller's (`for_target`, `draw`: a canvas in the browser).
pub struct Renderer {
    pub width: u32,
    pub height: u32,
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    camera: wgpu::Buffer,
    objects: wgpu::Buffer,
    vertices: wgpu::Buffer,
    vertex_count: u32,
    msaa: wgpu::TextureView,
    /// Offscreen (`new`): the resolved image and its readback, for `render`.
    resolve: Option<wgpu::Texture>,
    depth: wgpu::TextureView,
    readback: Option<wgpu::Buffer>,
    row: u32,
    /// Object index of each chunk, by structure.
    chunk_object: Vec<Vec<u32>>,
    impactor_base: u32,
    object_count: usize,
    /// Distance (m) over which the ground grid fades.
    pub fade: f64,
    pub background: [f64; 3],
    /// Draw the ground plane (off for a scene without ground).
    pub ground: bool,
    /// What a chunk is painted by.
    pub colors: ColorMode,
    body_colors: RefCell<BodyColors>,
    /// Draw the intact bonds as lines between their chunks' centres.
    pub bond_graph: bool,
    line_pipeline: wgpu::RenderPipeline,
    line_bind: wgpu::BindGroup,
    lines: wgpu::Buffer,
    /// Line vertices `lines` holds (two per bond).
    line_capacity: usize,
    /// Each object's mesh centre (object frame) and radius; 0 for the ground and the
    /// impactors (they shed no fragment effects).
    bounds: Vec<[f32; 4]>,
    format: wgpu::TextureFormat,
    /// Dust and chips ([`Self::set_fx`]), and the simulated time the next draw advances them.
    fx: Option<crate::fx::Fx>,
    fx_dt: std::cell::Cell<f32>,
}

const SAMPLES: u32 = 4;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

impl Renderer {
    /// Meshes for every chunk of `solver` (their hulls, or boxes) and every impactor.
    pub fn new(gpu: &Gpu, solver: &ReferenceSolver, impactors: &[Impactor], width: u32, height: u32) -> Renderer {
        Self::with_format(gpu, solver, impactors, width, height, FORMAT, true)
    }

    /// A renderer that draws into the caller's `format` target (`draw`), without the
    /// offscreen image.
    pub fn for_target(gpu: &Gpu, solver: &ReferenceSolver, impactors: &[Impactor], width: u32, height: u32, format: wgpu::TextureFormat) -> Renderer {
        Self::with_format(gpu, solver, impactors, width, height, format, false)
    }

    fn with_format(gpu: &Gpu, solver: &ReferenceSolver, impactors: &[Impactor], width: u32, height: u32, format: wgpu::TextureFormat, offscreen: bool) -> Renderer {
        let device = &gpu.device;
        let mut verts: Vec<Vertex> = Vec::new();
        let mut chunk_object = Vec::new();
        let mut object = 1u32; // 0 is the ground
        let mut bounds = vec![[0.0f32; 4]];
        // Ground: a large quad at z = 0 (the scene packs' grade).
        let g = 2000.0f32;
        let up = [0.0, 0.0, 1.0];
        for p in [[-g, -g], [g, -g], [g, g], [-g, -g], [g, g], [-g, g]] {
            verts.push(Vertex { pos: [p[0], p[1], 0.0], normal: up, object: 0 });
        }
        for st in &solver.structures {
            let mut ids = Vec::with_capacity(st.chunks.len());
            for ch in &st.chunks {
                let first = verts.len();
                chunk_mesh(ch, object, &mut verts);
                bounds.push(mesh_bounds(&verts[first..]));
                ids.push(object);
                object += 1;
            }
            chunk_object.push(ids);
        }
        let impactor_base = object;
        bounds.resize(bounds.len() + impactors.len(), [0.0; 4]);
        for imp in impactors {
            match imp.shape {
                ImpactorShape::Sphere { radius } => sphere_triangles(radius, object, &mut verts),
                ImpactorShape::Box { half_extents } => {
                    let (v, f) = box_faces(Vec3::from_array(half_extents));
                    hull_triangles(&v, &f, object, &mut verts);
                }
            }
            object += 1;
        }
        let object_count = object as usize;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let objects = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render objects"),
            size: (object_count * std::mem::size_of::<Object>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("render"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: objects.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("render"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &attributes })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
            multiview_mask: None,
            cache: None,
        });
        // The bond graph's lines: the camera only, over the geometry (no depth test).
        let line_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("render lines"), source: wgpu::ShaderSource::Wgsl(LINE_SHADER.into()) });
        let line_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render lines"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let line_bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("render lines"), layout: &line_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }] });
        let line_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("render lines"), bind_group_layouts: &[Some(&line_layout)], immediate_size: 0 });
        let line_attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render lines"),
            layout: Some(&line_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_module,
                entry_point: Some("line_vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<LineVertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &line_attributes })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState { module: &line_module, entry_point: Some("line_fs"), compilation_options: Default::default(), targets: &[Some(format.into())] }),
            multiview_mask: None,
            cache: None,
        });
        let line_capacity = (2 * solver.bonds.iter().map(|b| b.len()).sum::<usize>()).max(2);
        let lines = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render lines"),
            size: (line_capacity * std::mem::size_of::<LineVertex>()) as u64,
            // STORAGE: the GPU-resident threshold world writes them (`draw_resident`).
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let texture = |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = texture("render msaa", format, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let resolve = offscreen.then(|| texture("render resolve", format, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC));
        let depth = texture("render depth", wgpu::TextureFormat::Depth32Float, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let row = (width * 4).div_ceil(256) * 256;
        let readback = offscreen.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render readback"),
                size: (row * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        Renderer {
            width,
            height,
            pipeline,
            bind,
            camera,
            objects,
            vertices,
            vertex_count: verts.len() as u32,
            msaa,
            resolve,
            depth,
            readback,
            row,
            chunk_object,
            impactor_base,
            object_count,
            fade: 120.0,
            background: [0.70, 0.76, 0.84],
            ground: true,
            colors: ColorMode::Material,
            body_colors: RefCell::default(),
            bond_graph: false,
            line_pipeline,
            line_bind,
            lines,
            line_capacity,
            bounds,
            format,
            fx: None,
            fx_dt: std::cell::Cell::new(0.0),
        }
    }

    /// Draw dust and chips where chunks change speed sharply (an impact, a break, a
    /// landing), or not: render only, from the poses drawn (`fx.rs`). Off by default.
    pub fn set_fx(&mut self, gpu: &Gpu, on: bool) {
        if !on {
            self.fx = None;
        } else if self.fx.is_none() {
            self.fx = Some(crate::fx::Fx::new(gpu, &self.camera, &self.objects, &self.bounds, self.format, SAMPLES));
        }
    }

    /// The simulated time (s) the next draw advances the effects by: what the frame stepped
    /// (0 when paused).
    pub fn set_fx_dt(&self, dt: f64) {
        self.fx_dt.set(dt as f32);
    }

    /// Triangles drawn per frame.
    pub fn triangles(&self) -> usize {
        self.vertex_count as usize / 3
    }

    /// Render the solver's current state from `cam`; returns tightly packed RGBA (sRGB).
    pub fn render(&self, gpu: &Gpu, solver: &ReferenceSolver, impactors: &[Impactor], cam: &CameraPose) -> Vec<u8> {
        let (resolve, readback) = (self.resolve.as_ref().expect("an offscreen renderer (`new`)"), self.readback.as_ref().expect("an offscreen renderer"));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let view = resolve.create_view(&Default::default());
        self.draw(gpu, &mut encoder, &view, solver, impactors, cam.view_proj(self.width as f64 / self.height as f64), cam.eye());
        encoder.copy_texture_to_buffer(
            resolve.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.row), rows_per_image: Some(self.height) } },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map the frame"));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mut out = vec![0u8; (self.width * self.height * 4) as usize];
        {
            let data = readback.get_mapped_range(..).expect("mapped frame");
            let w = (self.width * 4) as usize;
            for y in 0..self.height as usize {
                out[y * w..(y + 1) * w].copy_from_slice(&data[y * self.row as usize..][..w]);
            }
        }
        readback.unmap();
        out
    }

    /// The objects buffer (per object 3 rows and a colour), the bond lines' vertices
    /// (7 floats each), each structure's chunks' objects, and the first impactor's object:
    /// what a GPU writer fills for [`Self::draw_resident`].
    pub fn resident_targets(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &[Vec<u32>], u32) {
        (&self.objects, &self.lines, &self.chunk_object, self.impactor_base)
    }

    /// Room for bond-line vertices.
    pub fn line_capacity(&self) -> usize {
        self.line_capacity
    }

    /// Draw what a GPU writer left in the objects and lines buffers ([`Self::resident_targets`],
    /// recorded into `encoder` before this): nothing read from the host but the ground's
    /// object and the camera. `line_vertices`: the bond lines' vertices to draw (0: none).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_resident(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, view_proj: [[f32; 4]; 4], eye: Vec3, line_vertices: u32) {
        let ground = if self.ground { Object { rows: rows(Mat3::IDENTITY, Vec3::ZERO), color: [0.56, 0.58, 0.55, 2.0] } } else { Object { rows: [[0.0; 4]; 3], color: [0.0; 4] } };
        gpu.queue.write_buffer(&self.objects, 0, bytemuck::bytes_of(&ground));
        self.pass(gpu, encoder, target, view_proj, eye, line_vertices.min(self.line_capacity as u32));
    }

    /// The camera, then the geometry and `line_vertices` of the bond lines into `target`.
    fn pass(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, view_proj: [[f32; 4]; 4], eye: Vec3, line_vertices: u32) {
        let light = Vec3::new(-0.45, -0.55, 0.70).normalized();
        let uniform = CameraUniform {
            view_proj,
            eye: [eye.x as f32, eye.y as f32, eye.z as f32, 1.0],
            light: [light.x as f32, light.y as f32, light.z as f32, self.fade as f32],
        };
        gpu.queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(&uniform));
        if let Some(fx) = &self.fx {
            fx.encode(gpu, encoder, self.fx_dt.replace(0.0));
        }
        let view = target;
        let bg = self.background;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa,
                depth_slice: None,
                resolve_target: Some(view),
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }), store: wgpu::StoreOp::Discard },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
        if let Some(fx) = &self.fx {
            fx.draw(&mut pass);
        }
        if line_vertices > 0 {
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.line_bind, &[]);
            pass.set_vertex_buffer(0, self.lines.slice(..));
            pass.draw(0..line_vertices, 0..1);
        }
    }

    /// Record the solver's current state, seen through `view_proj` from `eye` (Z up),
    /// into `target` (resolved from the multisampled image; the renderer's size and format).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, solver: &ReferenceSolver, impactors: &[Impactor], view_proj: [[f32; 4]; 4], eye: Vec3) {
        let mut objs = vec![Object { rows: rows(Mat3::IDENTITY, Vec3::ZERO), color: [0.56, 0.58, 0.55, 2.0] }; self.object_count];
        if !self.ground {
            objs[0] = Object { rows: [[0.0; 4]; 3], color: [0.0; 4] };
        }
        let mut body_colors = self.body_colors.borrow_mut();
        if self.colors == ColorMode::Bodies {
            body_colors.update(solver);
        }
        for (s, st) in solver.structures.iter().enumerate() {
            for (c, ch) in st.chunks.iter().enumerate() {
                let state = &solver.chunks[s][c];
                let k = self.chunk_object[s][c] as usize;
                if !state.active || state.removed {
                    objs[k] = Object { rows: [[0.0; 4]; 3], color: [0.0; 4] };
                    continue;
                }
                let cl = &solver.clusters[state.cluster];
                let hidden = Quat::from_axis_angle(state.th, state.th.norm()).to_mat3();
                let r = cl.rotation() * hidden * ch.rotation;
                let base = material_color(&ch.material);
                let color = match self.colors {
                    ColorMode::Sleep if cl.anchored => STATIC,
                    ColorMode::Sleep if cl.velocity.norm() > REST_SPEED || cl.angular_velocity.norm() > REST_SPEED => MOVING,
                    ColorMode::Sleep => AT_REST,
                    ColorMode::Bodies => body_colors.color(cl.id),
                    ColorMode::Material if cl.anchored => base,
                    ColorMode::Material => {
                        let f = fragment_color(cl.id);
                        std::array::from_fn(|i| 0.45 * base[i] + 0.55 * f[i])
                    }
                };
                objs[k] = Object { rows: rows(r, solver.chunk_position(s, c)), color: [color[0], color[1], color[2], 1.0] };
            }
        }
        for (i, imp) in impactors.iter().enumerate() {
            let k = self.impactor_base as usize + i;
            if k < self.object_count {
                objs[k] = Object { rows: rows(imp.pose.rotation.to_mat3(), imp.pose.position), color: [0.18, 0.19, 0.22, 1.0] };
            }
        }
        gpu.queue.write_buffer(&self.objects, 0, bytemuck::cast_slice(&objs));
        // The intact bonds, chunk centre to chunk centre (as the chunks are drawn).
        let mut line_verts: Vec<LineVertex> = Vec::new();
        if self.bond_graph {
            for (s, bonds) in solver.bonds.iter().enumerate() {
                for b in bonds.iter().filter(|b| b.connected()) {
                    let (a, b2) = (b.geometry.a, b.geometry.b);
                    let (ca, cb) = (&solver.chunks[s][a], &solver.chunks[s][b2]);
                    if !ca.active || ca.removed || !cb.active || cb.removed {
                        continue;
                    }
                    for c in [a, b2] {
                        let p = solver.chunk_position(s, c);
                        line_verts.push(LineVertex { pos: [p.x as f32, p.y as f32, p.z as f32], color: BOND_LINE });
                    }
                }
            }
            line_verts.truncate(self.line_capacity);
            gpu.queue.write_buffer(&self.lines, 0, bytemuck::cast_slice(&line_verts));
        }
        self.pass(gpu, encoder, target, view_proj, eye, line_verts.len() as u32);
    }
}

/// Text and bars drawn into an RGBA frame (a 5x7 bitmap font, upper case).
pub struct Overlay<'a> {
    pub rgba: &'a mut [u8],
    pub width: u32,
    pub height: u32,
}

fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        _ => [0; 7],
    }
}

impl Overlay<'_> {
    /// Blend `color` (RGB, sRGB bytes) with opacity `alpha` over the rectangle.
    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: [u8; 3], alpha: f32) {
        let (x0, y0) = (x.max(0), y.max(0));
        let (x1, y1) = ((x + w).min(self.width as i32), (y + h).min(self.height as i32));
        for yy in y0..y1 {
            for xx in x0..x1 {
                let p = &mut self.rgba[((yy as u32 * self.width + xx as u32) * 4) as usize..][..3];
                for i in 0..3 {
                    p[i] = (p[i] as f32 * (1.0 - alpha) + color[i] as f32 * alpha).round() as u8;
                }
            }
        }
    }

    /// Width in pixels of `text` at `scale`.
    pub fn text_width(text: &str, scale: i32) -> i32 {
        text.chars().count() as i32 * 6 * scale - scale
    }

    /// Draw `text` with its top-left corner at (x, y), each font pixel `scale` pixels.
    pub fn text(&mut self, x: i32, y: i32, scale: i32, color: [u8; 3], text: &str) {
        for (i, ch) in text.chars().enumerate() {
            let g = glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits & (0x10 >> col) != 0 {
                        self.rect(x + (i as i32 * 6 + col) * scale, y + row as i32 * scale, scale, scale, color, 1.0);
                    }
                }
            }
        }
    }
}

/// ffmpeg reading raw RGBA frames on stdin, writing H.264 (yuv420p).
pub struct FfmpegSink {
    child: Child,
}

impl FfmpegSink {
    pub fn new(path: &std::path::Path, width: u32, height: u32, fps: u32) -> std::io::Result<FfmpegSink> {
        let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| if std::path::Path::new("/opt/homebrew/bin/ffmpeg").exists() { "/opt/homebrew/bin/ffmpeg".into() } else { "ffmpeg".into() });
        let child = Command::new(ffmpeg)
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i", "-"])
            // `STRESS_VIDEO_CODEC=videotoolbox`: the GPU's hardware H.264 encoder (macOS).
            .args(if std::env::var("STRESS_VIDEO_CODEC").is_ok_and(|v| v == "videotoolbox") {
                vec!["-c:v", "h264_videotoolbox", "-b:v", "30M", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            } else {
                vec!["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            })
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()?;
        Ok(FfmpegSink { child })
    }

    pub fn write(&mut self, rgba: &[u8]) -> std::io::Result<()> {
        self.child.stdin.as_mut().expect("ffmpeg stdin").write_all(rgba)
    }

    /// Close the input and wait for the encoder.
    pub fn finish(mut self) -> std::io::Result<()> {
        drop(self.child.stdin.take());
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("ffmpeg exited with {status}")))
        }
    }
}
