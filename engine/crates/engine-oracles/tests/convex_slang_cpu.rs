//! CPU conformance for the Slang convex narrowphase. The production kernels
//! (`convex_classify`, `convex_sat_*`, `convex_manifold`) are compiled with
//! `slangc -target cpp -DENGINE_CPU_TEST=1` (sequential atomics, one SAT lane per
//! pair) and driven by `convex_cpu/harness.cpp`. Results are compared against the
//! exhaustive SAT and distance oracles in `engine_schema::convex`.
//! Needs `slangc` and a C++17 compiler; no GPU.

use engine_schema::convex::*;
use engine_schema::*;
use glam::{Quat, Vec3};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const MARGIN: f32 = 0.01;
/// Separations this close to zero may legitimately disagree (f32 vs exhaustive).
const AMBIGUOUS: f32 = 2e-4;
const FEATURE_TYPE_EDGE: u32 = 3;

fn harness() -> &'static Path {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let slang = root.join("../../slang/kernels/convex_step.slang");
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("convex_cpu");
        std::fs::create_dir_all(&dir).unwrap();
        let generated = dir.join("convex_generated.cpp");
        let slangc = engine_slang_build::find_slangc().expect("slangc is required for this test");
        let mut cmd = Command::new(slangc);
        cmd.arg(&slang).args(["-target", "cpp", "-DENGINE_CPU_TEST=1"]);
        for e in ["convex_classify", "convex_sat_small", "convex_sat_large", "convex_manifold", "build_aabbs", "broadphase_clear", "broadphase_insert", "broadphase_query", "narrowphase", "reuse_all_manifolds"] {
            cmd.args(["-entry", e, "-stage", "compute"]);
        }
        let out = cmd.arg("-o").arg(&generated).output().unwrap();
        assert!(out.status.success(), "slangc: {}", String::from_utf8_lossy(&out.stderr));
        let bin = dir.join("harness");
        let out = Command::new(std::env::var("CXX").unwrap_or_else(|_| "c++".into()))
            .args(["-std=c++17", "-O2", "-w"])
            .arg(format!("-DCONVEX_GENERATED=\"{}\"", generated.display()))
            .arg(root.join("convex_cpu/harness.cpp"))
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        assert!(out.status.success(), "c++: {}", String::from_utf8_lossy(&out.stderr));
        bin
    })
}

#[derive(Clone)]
enum Shape {
    Hull(ConvexHull, f32),
    Box(Vec3),
    Sphere(f32),
    Capsule(f32, f32),
    Plane(Vec3, f32),
}

#[derive(Clone, Default)]
struct Scene {
    bodies: Vec<BodyState>,
    colliders: Vec<Collider>,
    shapes: Vec<Shape>,
    arena: Option<HullArena>,
}

impl Scene {
    fn new() -> Self {
        Self { arena: Some(HullArena::new()), ..Default::default() }
    }
    fn body(&mut self, p: Vec3, q: Quat) -> u32 {
        self.bodies.push(BodyState::dynamic(p, q, 1.0, Vec3::ONE));
        self.bodies.len() as u32 - 1
    }
    fn static_body(&mut self) -> u32 {
        self.bodies.push(BodyState::static_plane_support());
        self.bodies.len() as u32 - 1
    }
    fn add(&mut self, body: u32, shape: Shape, local_pos: Vec3, local_rot: Quat) -> u32 {
        let (kind, params) = match &shape {
            Shape::Hull(h, s) => {
                let base = self.arena.as_mut().unwrap().insert(h).unwrap();
                (ShapeType::ConvexHull, convex_collider_params(base, *s, h.bounding_radius()))
            }
            Shape::Box(h) => (ShapeType::Box, [h.x, h.y, h.z, h.length()]),
            Shape::Sphere(r) => (ShapeType::Sphere, [*r, 0.0, 0.0, *r]),
            Shape::Capsule(r, l) => (ShapeType::Capsule, [*r, *l, 0.0, r + l]),
            Shape::Plane(n, w) => (ShapeType::Plane, [n.x, n.y, n.z, *w]),
        };
        self.colliders.push(Collider {
            body,
            shape: kind as u32,
            material: 0,
            generation: 0,
            local_pos: local_pos.to_array(),
            density: 1.0,
            local_rot: local_rot.to_array(),
            params,
        });
        self.shapes.push(shape);
        self.colliders.len() as u32 - 1
    }
    fn pose(&self, c: usize) -> Pose {
        let col = &self.colliders[c];
        let b = &self.bodies[col.body as usize];
        let (bp, bq) = (Vec3::from_array(b.position), Quat::from_array(b.orientation));
        Pose {
            position: bp + bq * Vec3::from_array(col.local_pos),
            rotation: bq * Quat::from_array(col.local_rot),
        }
    }
    fn polytope(&self, c: usize) -> Option<WorldPolytope> {
        match &self.shapes[c] {
            Shape::Hull(h, s) => Some(WorldPolytope::new(h, Vec3::splat(*s), self.pose(c))),
            Shape::Box(half) => Some(WorldPolytope::new(&ConvexHull::cuboid(Vec3::ONE), *half, self.pose(c))),
            _ => None,
        }
    }
    fn hull_of(&self, c: usize) -> Option<(ConvexHull, Vec3)> {
        match &self.shapes[c] {
            Shape::Hull(h, s) => Some((h.clone(), Vec3::splat(*s))),
            Shape::Box(half) => Some((ConvexHull::cuboid(Vec3::ONE), *half)),
            _ => None,
        }
    }
    /// Every cross-body pair involving a hull whose bounding spheres overlap.
    fn candidates(&self, hints: &std::collections::HashMap<(u32, u32), u32>) -> Vec<CandidatePair> {
        let mut out = Vec::new();
        for i in 0..self.colliders.len() {
            for j in i + 1..self.colliders.len() {
                let (a, b) = (&self.colliders[i], &self.colliders[j]);
                if a.body == b.body || (a.shape != ShapeType::ConvexHull as u32 && b.shape != ShapeType::ConvexHull as u32) {
                    continue;
                }
                let plane = a.shape == ShapeType::Plane as u32 || b.shape == ShapeType::Plane as u32;
                let d = (self.pose(i).position - self.pose(j).position).length();
                if !plane && d > a.params[3] + b.params[3] + 0.05 {
                    continue;
                }
                let hint = hints.get(&(i as u32, j as u32)).copied().unwrap_or(0);
                out.push(CandidatePair { collider_a: i as u32, collider_b: j as u32, key: 0, _pad: 0x8000_0000 | hint });
            }
        }
        out
    }
}

struct Output {
    counters: StepCounters,
    contacts: Vec<Contact>,
    manifolds: Vec<Manifold>,
    /// Pipeline mode: broadphase candidates and collider AABBs (mn, r, mx, body).
    candidates: Vec<CandidatePair>,
    aabbs: Vec<[f32; 8]>,
    /// Mode 2: the collision step's contacts/manifold count before reuse.
    prev_contacts: Vec<Contact>,
    prev_manifolds: u32,
}

fn run(scene: &Scene, candidates: &[CandidatePair]) -> Output {
    run_mode(scene, candidates, 0, 0)
}

/// Whole collision front end on the CPU: bounds, hierarchical broadphase,
/// narrowphase (hull pairs deferred) and the convex stages.
fn run_pipeline(scene: &Scene, capacity: usize) -> Output {
    let empty = vec![CandidatePair { collider_a: 0, collider_b: 0, key: 0, _pad: 0 }; capacity];
    run_mode(scene, &empty, 1, 1 << 17)
}

/// Pipeline step, then a whole-step manifold reuse into the other contact half.
fn run_pipeline_then_reuse(scene: &Scene, capacity: usize) -> Output {
    let empty = vec![CandidatePair { collider_a: 0, collider_b: 0, key: 0, _pad: 0 }; capacity];
    run_mode(scene, &empty, 2, 1 << 17)
}

fn run_mode(scene: &Scene, candidates: &[CandidatePair], mode: u32, grid_words: u32) -> Output {
    let bin = harness();
    let arena = scene.arena.as_ref().unwrap();
    let max_contacts = (candidates.len() * 8).max(64) as u32;
    let mut bytes: Vec<u8> = Vec::new();
    let head = [0x3158_5643u32, scene.bodies.len() as u32, scene.colliders.len() as u32, candidates.len() as u32, arena.len() as u32, max_contacts, mode, grid_words];
    bytes.extend_from_slice(bytemuck::cast_slice(&head));
    let (base, levels) = hierarchical_grid(&scene.colliders, &scene.bodies);
    let params = AvbdParams { contact_offset: MARGIN, grid_cell_size: base, ..AvbdParams::default() };
    bytes.extend_from_slice(bytemuck::bytes_of(&params));
    let mut scratch = [0u32; 64];
    scratch[1] = scene.bodies.len() as u32; // collider AABBs start after body slots
    scratch[2] = candidates.len() as u32; // max candidates (and the work list size)
    scratch[3] = max_contacts;
    scratch[10] = candidates.len() as u32; // max manifolds
    scratch[16] = grid_words;
    scratch[17] = if mode >= 1 { 0 } else { 1 }; // first step builds sleeping bounds
    scratch[18] = if mode >= 1 { levels } else { 0 };
    scratch[23] = u32::from(scene.colliders.iter().any(CollisionFilter::is_set));
    bytes.extend_from_slice(bytemuck::cast_slice(&scratch));
    bytes.extend_from_slice(bytemuck::cast_slice(&scene.bodies));
    bytes.extend_from_slice(bytemuck::cast_slice(&scene.colliders));
    if mode == 0 {
        bytes.extend_from_slice(bytemuck::cast_slice(candidates));
    }
    bytes.extend_from_slice(bytemuck::cast_slice(arena.records()));
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("convex_cpu");
    let id = std::thread::current().id();
    let input = dir.join(format!("in-{id:?}.bin"));
    let output = dir.join(format!("out-{id:?}.bin"));
    std::fs::write(&input, &bytes).unwrap();
    let status = Command::new(bin).arg(&input).arg(&output).output().unwrap();
    assert!(status.status.success(), "harness failed: {}", String::from_utf8_lossy(&status.stderr));
    let out = std::fs::read(&output).unwrap();
    let counters: StepCounters = *bytemuck::from_bytes(&out[..64]);
    let nc = counters.contact_count as usize;
    let nm = counters.manifold_count as usize;
    let mut at = 64;
    let contacts = bytemuck::cast_slice::<u8, Contact>(&out[at..at + nc * 128]).to_vec();
    at += nc * 128;
    let manifolds = bytemuck::cast_slice::<u8, Manifold>(&out[at..at + nm * 32]).to_vec();
    at += nm * 32 + candidates.len() * 32;
    let found = (counters.candidate_count as usize).min(candidates.len());
    let cands = bytemuck::cast_slice::<u8, CandidatePair>(&out[at..at + found * 16]).to_vec();
    at += found * 16;
    let aabbs = bytemuck::cast_slice::<u8, [f32; 8]>(&out[at..at + scene.colliders.len() * 32]).to_vec();
    at += scene.colliders.len() * 32;
    let tail: [u32; 2] = *bytemuck::from_bytes(&out[at..at + 8]);
    at += 8;
    let prev_contacts = if mode == 2 {
        bytemuck::cast_slice::<u8, Contact>(&out[at..at + tail[0] as usize * 128]).to_vec()
    } else {
        Vec::new()
    };
    Output { counters, contacts, manifolds, candidates: cands, aabbs, prev_contacts, prev_manifolds: tail[1] }
}

/// Distance from a world point to a convex polytope (0 inside).
fn point_distance(h: &ConvexHull, scale: Vec3, pose: Pose, p: Vec3) -> f32 {
    let local = pose.rotation.inverse() * (p - pose.position);
    let verts: Vec<Vec3> = h.vertices.iter().map(|v| *v * scale).collect();
    let planes: Vec<(Vec3, f32)> = h
        .faces
        .iter()
        .map(|f| {
            let n = (f.normal / scale).normalize();
            (n, n.dot(verts[f.indices[0] as usize]))
        })
        .collect();
    if planes.iter().all(|(n, d)| n.dot(local) <= *d) {
        return 0.0;
    }
    let mut best = f32::INFINITY;
    for (f, (n, d)) in h.faces.iter().zip(&planes) {
        let s = n.dot(local) - d;
        if s <= 0.0 {
            continue;
        }
        let q = local - *n * s;
        let mut inside = true;
        let mut edge = f32::INFINITY;
        for k in 0..f.indices.len() {
            let a = verts[f.indices[k] as usize];
            let b = verts[f.indices[(k + 1) % f.indices.len()] as usize];
            let e = b - a;
            if e.cross(*n).dot(q - a) > 0.0 {
                inside = false;
                let t = ((local - a).dot(e) / e.dot(e)).clamp(0.0, 1.0);
                edge = edge.min((local - (a + e * t)).length());
            }
        }
        best = best.min(if inside { s } else { edge });
    }
    best
}

/// Signed distance of a world point to the hull surface (negative inside).
fn surface_distance(h: &ConvexHull, scale: Vec3, pose: Pose, p: Vec3) -> f32 {
    let d = point_distance(h, scale, pose, p);
    if d > 0.0 {
        return d;
    }
    let local = pose.rotation.inverse() * (p - pose.position);
    h.faces
        .iter()
        .map(|f| {
            let n = (f.normal / scale).normalize();
            n.dot(local) - n.dot(h.vertices[f.indices[0] as usize] * scale)
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Oracle separation for one candidate (positive = apart).
fn oracle_separation(scene: &Scene, a: usize, b: usize) -> f32 {
    let shape_a = &scene.shapes[a];
    let shape_b = &scene.shapes[b];
    match (shape_a, shape_b) {
        (Shape::Plane(n, w), _) | (_, Shape::Plane(n, w)) => {
            let h = if matches!(shape_a, Shape::Plane(..)) { b } else { a };
            let poly = scene.polytope(h).unwrap();
            poly.vertices.iter().map(|v| n.dot(*v) - w).fold(f32::INFINITY, f32::min)
        }
        (Shape::Sphere(r), _) | (_, Shape::Sphere(r)) => {
            let (s, h) = if matches!(shape_a, Shape::Sphere(..)) { (a, b) } else { (b, a) };
            let (hull, scale) = scene.hull_of(h).unwrap();
            point_distance(&hull, scale, scene.pose(h), scene.pose(s).position) - r
        }
        (Shape::Capsule(r, l), _) | (_, Shape::Capsule(r, l)) => {
            let (c, h) = if matches!(shape_a, Shape::Capsule(..)) { (a, b) } else { (b, a) };
            let (hull, scale) = scene.hull_of(h).unwrap();
            let pose = scene.pose(c);
            let axis = pose.rotation * Vec3::Y * *l;
            (0..=256)
                .map(|k| {
                    let p = pose.position - axis + axis * (2.0 * k as f32 / 256.0);
                    point_distance(&hull, scale, scene.pose(h), p)
                })
                .fold(f32::INFINITY, f32::min)
                - r
        }
        _ => sat_separation(&scene.polytope(a).unwrap(), &scene.polytope(b).unwrap()).1,
    }
}

/// Best face-axis and best edge-axis separations (exhaustive projections).
fn face_edge_separation(a: &WorldPolytope, b: &WorldPolytope) -> (f32, f32) {
    let ca = a.vertices.iter().copied().sum::<Vec3>() / a.vertices.len() as f32;
    let cb = b.vertices.iter().copied().sum::<Vec3>() / b.vertices.len() as f32;
    let axis = |v: Vec3| {
        let l = v.length();
        if l < 1e-6 {
            return None;
        }
        let n = v / l;
        Some(if n.dot(cb - ca) < 0.0 { -n } else { n })
    };
    // Face axes are one-sided (the Minkowski faces): B against A's outward
    // normals, A against B's. Reversed face normals are not candidates.
    let span = |vs: &[Vec3], n: Vec3| vs.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(n.dot(*v)), hi.max(n.dot(*v))));
    let face_a = a.normals.iter().map(|n| span(&b.vertices, *n).0 - span(&a.vertices, *n).1);
    let face_b = b.normals.iter().map(|n| span(&a.vertices, *n).0 - span(&b.vertices, *n).1);
    let face = face_a.chain(face_b).fold(f32::NEG_INFINITY, f32::max);
    let mut edge = f32::NEG_INFINITY;
    for da in &a.edges {
        for db in &b.edges {
            if let Some(n) = axis(da.cross(*db)) {
                edge = edge.max(a.separation_along(b, n));
            }
        }
    }
    (face, edge)
}

struct Report {
    expected: usize,
    shallow: usize,
    /// Largest depth shortfall vs the true penetration for face-bias patches.
    starved_deficit: f32,
    checked_depth: usize,
    worst_axis: f32,
    worst_surface: f32,
}

/// Contact existence, manifold normal/depth quality and point placement.
fn check(scene: &Scene, candidates: &[CandidatePair], out: &Output, label: &str) -> Report {
    assert_eq!(out.counters.overflow_flags, 0, "{label}: overflow");
    let mut by_pair: std::collections::HashMap<(u32, u32), Vec<&Contact>> = Default::default();
    for c in &out.contacts {
        by_pair.entry((c.collider_a, c.collider_b)).or_default().push(c);
    }
    for m in &out.manifolds {
        assert!(m.contact_count as usize <= 8, "{label}: manifold size");
    }
    let mut report = Report { expected: 0, shallow: 0, starved_deficit: 0.0, checked_depth: 0, worst_axis: 0.0, worst_surface: 0.0 };
    for cand in candidates {
        let (a, b) = (cand.collider_a as usize, cand.collider_b as usize);
        let sep = oracle_separation(scene, a, b);
        let got = by_pair.get(&(cand.collider_a, cand.collider_b));
        if sep.abs() < AMBIGUOUS {
            continue;
        }
        if sep > 0.0 {
            assert!(got.is_none(), "{label}: false contact {a},{b} (separation {sep})");
            continue;
        }
        report.expected += 1;
        let got = got.unwrap_or_else(|| panic!("{label}: missed contact {a},{b} (separation {sep})"));
        let n = Vec3::from_array(got[0].normal);
        assert!((n.length() - 1.0).abs() < 1e-3, "{label}: non-unit normal {a},{b}");
        for c in got.iter() {
            assert!(c.c0_n.is_finite() && c.anchor_a.iter().chain(&c.anchor_b).all(|x| x.is_finite()));
        }
        if let (Some(pa), Some(pb)) = (scene.polytope(a), scene.polytope(b)) {
            assert!(got.len() <= 4, "{label}: {} points for {a},{b} (MAX_CONVEX_CONTACTS 4)", got.len());
            let (best_face, best_edge) = face_edge_separation(&pa, &pb);
            let best = best_face.max(best_edge);
            let along = pa.separation_along(&pb, n);
            report.worst_axis = report.worst_axis.min(along - best);
            // The winning feature type is in feature bits 30..31. A face must be
            // the best face; an edge must be the best edge and clear the face
            // bias (edge > 0.95·face + 0.01, CONVEX_EDGE_*).
            let ty = got[0].feature_id >> 30;
            if ty == FEATURE_TYPE_EDGE {
                // Either the edge cleared the face bias, or the biased face's
                // patch clipped empty and fell back to the best edge pair.
                assert!(along >= best_edge - 1e-3, "{label}: edge {a},{b} gives {along} vs best edge {best_edge}");
            } else {
                assert!(along >= best_face - 1e-3, "{label}: face {a},{b} gives {along} vs best face {best_face}");
                assert!(best_edge <= 0.95 * along + 0.01 + 1e-3, "{label}: face {a},{b} kept over a clearly better edge {best_edge}");
            }
            // Shallow (resting) overlap: the deepest point equals the SAT depth.
            // Deep overlap: clipped incident points may lie deeper than the
            // minimum translation, never shallower.
            // When face bias keeps a face over a slightly better edge axis, the
            // patch cannot reach the full face depth; it is still at least as
            // deep as the true minimum penetration `best`.
            let deepest = got.iter().map(|c| c.c0_n).fold(f32::INFINITY, f32::min) - MARGIN;
            if along >= best - 1e-4 {
                if best > -0.1 {
                    assert!(deepest <= best + 2e-3, "{label}: depth {a},{b}: {deepest} shallower than penetration {best}");
                }
            } else {
                // Face bias kept a face over a slightly better edge; on sliver
                // faces the deepest support can fall outside the reference face.
                report.starved_deficit = report.starved_deficit.max(deepest - best);
            }
            if along > -0.05 && along >= best - 1e-4 {
                report.shallow += 1;
                assert!((deepest - along).abs() < 2e-3 + 0.05 * along.abs(), "{label}: depth {a},{b}: {deepest} vs {along}");
            }
            report.checked_depth += 1;
            // Points lie on their bodies' surfaces (reference points are
            // projected onto the reference face; incident points are hull points).
            let (ha, sa) = scene.hull_of(a).unwrap();
            let (hb, sb) = scene.hull_of(b).unwrap();
            let ba = &scene.bodies[scene.colliders[a].body as usize];
            let bb = &scene.bodies[scene.colliders[b].body as usize];
            for c in got.iter() {
                let wa = Vec3::from_array(ba.position) + Quat::from_array(ba.orientation) * Vec3::from_array(c.anchor_a);
                let wb = Vec3::from_array(bb.position) + Quat::from_array(bb.orientation) * Vec3::from_array(c.anchor_b);
                let da = surface_distance(&ha, sa, scene.pose(a), wa);
                let db = surface_distance(&hb, sb, scene.pose(b), wb);
                report.worst_surface = report.worst_surface.max(da.abs()).max(db.abs());
                let tol = 2e-3 + along.abs();
                assert!(da.abs() < tol && db.abs() < tol, "{label}: contact {a},{b} off surface ({da}, {db})");
                assert!(n.dot(wb - wa) <= 2e-3, "{label}: contact {a},{b} is not penetrating along n");
            }
        }
    }
    report
}

fn rng(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

fn random_rot(seed: &mut u32) -> Quat {
    Quat::from_xyzw(rng(seed) - 0.5, rng(seed) - 0.5, rng(seed) - 0.5, rng(seed) - 0.5).normalize()
}

fn prism(n: usize, r: f32, half_h: f32) -> ConvexHull {
    let mut v = Vec::new();
    for y in [-half_h, half_h] {
        for i in 0..n {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            v.push(Vec3::new(r * t.cos(), y, r * t.sin()));
        }
    }
    ConvexHull::from_points(&v).unwrap()
}

fn hull_library(seed: &mut u32) -> Vec<ConvexHull> {
    let mut hulls = vec![
        ConvexHull::cuboid(Vec3::new(0.5, 0.3, 0.4)),
        prism(5, 0.5, 0.4),
        prism(32, 0.6, 0.3),
        prism(12, 0.4, 0.6),
    ];
    for k in 0..6 {
        let n = 8 + 6 * k;
        let pts: Vec<Vec3> = (0..n).map(|_| Vec3::new(rng(seed) - 0.5, rng(seed) - 0.5, rng(seed) - 0.5) * 1.1).collect();
        hulls.push(ConvexHull::from_points(&pts).unwrap());
    }
    // 64-vertex bipyramid (124 faces, 186 edges).
    let mut v: Vec<Vec3> = (0..62)
        .map(|i| {
            let t = i as f32 / 62.0 * std::f32::consts::TAU;
            Vec3::new(0.55 * t.cos(), 0.0, 0.55 * t.sin())
        })
        .collect();
    v.extend([Vec3::Y * 0.6, -Vec3::Y * 0.6]);
    hulls.push(ConvexHull::from_points(&v).unwrap());
    // Uncertified adjacency: the exhaustive (no Gauss map, no hill climb) path.
    let mut open = prism(6, 0.45, 0.35);
    open.closed = false;
    hulls.push(open);
    hulls
}

fn random_scene(seed: u32, count: usize) -> Scene {
    let mut seed = seed;
    let lib = hull_library(&mut seed);
    let mut s = Scene::new();
    let ground = s.static_body();
    s.add(ground, Shape::Plane(Vec3::Y, 0.0), Vec3::ZERO, Quat::IDENTITY);
    for i in 0..count {
        let cluster = (i / 6) as f32 * 4.0;
        let p = Vec3::new(cluster + rng(&mut seed) * 1.6, 0.25 + rng(&mut seed) * 1.4, rng(&mut seed) * 1.6);
        let b = s.body(p, random_rot(&mut seed));
        let pick = (rng(&mut seed) * 10.0) as usize;
        let shape = match pick {
            0 => Shape::Box(Vec3::new(0.3, 0.25, 0.4)),
            1 => Shape::Sphere(0.35),
            2 => Shape::Capsule(0.2, 0.35),
            _ => Shape::Hull(lib[(rng(&mut seed) * lib.len() as f32) as usize % lib.len()].clone(), if i % 9 == 0 { 1.4 } else { 1.0 }),
        };
        s.add(b, shape, Vec3::ZERO, Quat::IDENTITY);
        // Compound child, offset and rotated in the body frame.
        if i % 5 == 0 {
            s.add(b, Shape::Hull(lib[i % lib.len()].clone(), 0.6), Vec3::new(0.6, 0.0, 0.2), random_rot(&mut seed));
        }
    }
    s
}

/// Pairs of random hulls/boxes slid along their SAT axis to a 1–30 mm overlap:
/// resting-contact geometry, where depth and feature choice must be exact.
fn shallow_scene(seed: u32, pairs: usize) -> Scene {
    let mut seed = seed;
    let lib = hull_library(&mut seed);
    let mut s = Scene::new();
    for i in 0..pairs {
        let origin = Vec3::new((i % 10) as f32 * 4.0, 0.0, (i / 10) as f32 * 4.0);
        let a = s.body(origin, random_rot(&mut seed));
        let pick = |seed: &mut u32| {
            if rng(seed) < 0.2 {
                Shape::Box(Vec3::new(0.2 + rng(seed) * 0.3, 0.2 + rng(seed) * 0.3, 0.2 + rng(seed) * 0.3))
            } else {
                Shape::Hull(lib[(rng(seed) * lib.len() as f32) as usize % lib.len()].clone(), 0.8 + rng(seed) * 0.6)
            }
        };
        let ca = s.add(a, pick(&mut seed), Vec3::ZERO, Quat::IDENTITY);
        let dir = Vec3::new(rng(&mut seed) - 0.5, rng(&mut seed) - 0.5, rng(&mut seed) - 0.5).normalize();
        let b = s.body(origin + dir * 1.5, random_rot(&mut seed));
        let cb = s.add(b, pick(&mut seed), Vec3::ZERO, Quat::IDENTITY);
        let target = -(0.001 + rng(&mut seed) * 0.029);
        for _ in 0..6 {
            let (n, sep) = sat_separation(&s.polytope(ca as usize).unwrap(), &s.polytope(cb as usize).unwrap());
            let p = Vec3::from_array(s.bodies[b as usize].position) + n * (target - sep);
            s.bodies[b as usize].position = p.to_array();
        }
    }
    s
}

#[test]
fn resting_contacts_have_exact_depth_and_feature() {
    for seed in [3u32, 11, 77] {
        let scene = shallow_scene(seed, 120);
        let cands = scene.candidates(&Default::default());
        let out = run(&scene, &cands);
        let r = check(&scene, &cands, &out, &format!("shallow seed {seed}"));
        assert!(r.shallow > 60, "seed {seed}: only {} exact-depth checks", r.shallow);
        eprintln!("shallow seed {seed}: {} contacts, {} exact depth, worst surface {:.5}, face-bias depth deficit {:.4}", r.expected, r.shallow, r.worst_surface, r.starved_deficit);
        assert!(r.starved_deficit < 0.01, "face-bias depth deficit {}", r.starved_deficit);
    }
}

fn hint_from_feature(feature: u32) -> u32 {
    let ty = feature >> 30;
    if ty == 0 {
        0
    } else if ty == FEATURE_TYPE_EDGE {
        (ty << 16) | (((feature >> 8) & 255) << 8) | (feature & 255)
    } else {
        (ty << 16) | (((feature >> 23) & 127) << 8)
    }
}

#[test]
fn slang_convex_kernels_match_exhaustive_oracles_on_cpu() {
    let mut total = 0;
    for seed in [1u32, 7, 42, 1234] {
        let scene = random_scene(seed, 72);
        let cands = scene.candidates(&Default::default());
        let out = run(&scene, &cands);
        let r = check(&scene, &cands, &out, &format!("seed {seed}"));
        assert!(r.expected > 20, "seed {seed}: fixture produced only {} contacts", r.expected);
        assert!(r.checked_depth > 10);
        total += r.expected;
        eprintln!(
            "seed {seed}: {} candidates, {} contacts ({} shallow), {} SAT small / {} large, worst axis {:.4}, worst surface {:.5}",
            cands.len(),
            r.expected,
            r.shallow,
            out.counters.convex_sat_small,
            out.counters.convex_sat_large,
            r.worst_axis,
            r.worst_surface
        );
    }
    assert!(total > 100);
}

#[test]
fn kernels_are_deterministic_and_hints_keep_features() {
    let scene = random_scene(99, 60);
    let cands = scene.candidates(&Default::default());
    let a = run(&scene, &cands);
    let b = run(&scene, &cands);
    assert_eq!(bytemuck::cast_slice::<Contact, u8>(&a.contacts), bytemuck::cast_slice::<Contact, u8>(&b.contacts));
    // Seed every SAT pair with last frame's feature: the cached-face fast path
    // and edge hysteresis must reproduce the same manifolds on an unchanged pose.
    let mut hints = std::collections::HashMap::new();
    for c in &a.contacts {
        hints.entry((c.collider_a, c.collider_b)).or_insert(hint_from_feature(c.feature_id));
    }
    let hinted = scene.candidates(&hints);
    let h = run(&scene, &hinted);
    check(&scene, &hinted, &h, "hinted");
    let key = |o: &Output| {
        let mut v: Vec<(u32, u32, u32)> = o.contacts.iter().map(|c| (c.collider_a, c.collider_b, c.feature_id)).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(key(&a), key(&h), "hints changed contact features on an unchanged pose");
}

#[test]
fn resting_box_on_hull_gives_four_point_face_manifold() {
    let mut s = Scene::new();
    let base = s.body(Vec3::ZERO, Quat::IDENTITY);
    s.add(base, Shape::Hull(ConvexHull::cuboid(Vec3::new(2.0, 0.5, 2.0)), 1.0), Vec3::ZERO, Quat::IDENTITY);
    let top = s.body(Vec3::new(0.1, 0.5 + 0.25 - 0.004, -0.05), Quat::from_rotation_y(0.3));
    s.add(top, Shape::Box(Vec3::new(0.4, 0.25, 0.3)), Vec3::ZERO, Quat::IDENTITY);
    let cands = s.candidates(&Default::default());
    let out = run(&s, &cands);
    check(&s, &cands, &out, "resting box");
    assert_eq!(out.contacts.len(), 4, "box face on hull face keeps four corners");
    for c in &out.contacts {
        assert!((Vec3::from_array(c.normal) - Vec3::Y).length() < 1e-4);
        assert!((c.c0_n - MARGIN + 0.004).abs() < 1e-4, "depth {}", c.c0_n - MARGIN);
    }
}

#[test]
fn compound_children_offsets_and_rotations_are_respected() {
    // An L-shaped compound resting across a wall: only the child that reaches
    // the wall may touch it; the body origin is far from both children.
    let mut s = Scene::new();
    let wall = s.body(Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY);
    s.add(wall, Shape::Hull(ConvexHull::cuboid(Vec3::new(0.2, 2.0, 2.0)), 1.0), Vec3::ZERO, Quat::IDENTITY);
    let l = s.body(Vec3::new(0.0, 0.0, 0.0), Quat::from_rotation_z(0.2));
    let near = s.add(l, Shape::Hull(ConvexHull::cuboid(Vec3::new(1.0, 0.2, 0.2)), 1.0), Vec3::new(2.1, 0.4, 0.0), Quat::from_rotation_y(0.1));
    let far = s.add(l, Shape::Hull(ConvexHull::cuboid(Vec3::new(0.2, 1.0, 0.2)), 1.0), Vec3::new(-1.0, 0.0, 0.0), Quat::IDENTITY);
    let cands = s.candidates(&Default::default());
    let out = run(&s, &cands);
    check(&s, &cands, &out, "compound");
    assert!(out.contacts.iter().any(|c| c.collider_b == near || c.collider_a == near), "near child touches the wall");
    assert!(out.contacts.iter().all(|c| c.collider_a != far && c.collider_b != far));
}

#[test]
#[ignore]
fn debug_pair() {
    let seed: u32 = std::env::var("SEED").unwrap().parse().unwrap();
    let count: usize = std::env::var("COUNT").unwrap().parse().unwrap();
    let pa: u32 = std::env::var("PA").unwrap().parse().unwrap();
    let pb: u32 = std::env::var("PB").unwrap().parse().unwrap();
    let scene = if std::env::var_os("SHALLOW").is_some() { shallow_scene(seed, count) } else { random_scene(seed, count) };
    let mut cands: Vec<CandidatePair> = scene.candidates(&Default::default()).into_iter().filter(|c| c.collider_a == pa && c.collider_b == pb).collect();
    if std::env::var_os("HINTED").is_some() {
        let first = run(&scene, &cands);
        let hint = first.contacts.first().map_or(0, |c| hint_from_feature(c.feature_id));
        eprintln!("first run: {} contacts, hint {hint:x}", first.contacts.len());
        cands[0]._pad |= hint;
    }
    let out = run(&scene, &cands);
    let desc = |c: usize| match &scene.shapes[c] {
        Shape::Hull(h, s) => format!("hull v{} f{} e{} closed {} scale {s}", h.vertices.len(), h.faces.len(), h.edges.len(), h.closed),
        Shape::Box(h) => format!("box {h}"),
        _ => "other".into(),
    };
    eprintln!("A {} | B {}", desc(pa as usize), desc(pb as usize));
    let (a, b) = (scene.polytope(pa as usize).unwrap(), scene.polytope(pb as usize).unwrap());
    eprintln!("oracle face/edge {:?} sat {:?}", face_edge_separation(&a, &b), sat_separation(&a, &b));
    for c in &out.contacts {
        let n = Vec3::from_array(c.normal);
        let ba = &scene.bodies[scene.colliders[pa as usize].body as usize];
        let bb = &scene.bodies[scene.colliders[pb as usize].body as usize];
        let wa = Vec3::from_array(ba.position) + Quat::from_array(ba.orientation) * Vec3::from_array(c.anchor_a);
        let wb = Vec3::from_array(bb.position) + Quat::from_array(bb.orientation) * Vec3::from_array(c.anchor_b);
        let (ha, sa) = scene.hull_of(pa as usize).unwrap();
        let (hb, sb) = scene.hull_of(pb as usize).unwrap();
        eprintln!(
            "contact feat {:08x} type {} kind {} n {n} along {} c0-m {} surfA {} surfB {}",
            c.feature_id,
            c.feature_id >> 30,
            (c.feature_id >> 14) & 3,
            a.separation_along(&b, n),
            c.c0_n - MARGIN,
            surface_distance(&ha, sa, scene.pose(pa as usize), wa),
            surface_distance(&hb, sb, scene.pose(pb as usize), wb)
        );
    }
}


fn moving(scene: &Scene, c: usize) -> bool {
    let b = &scene.bodies[scene.colliders[c].body as usize];
    b.is_dynamic() && !b.is_sleeping()
}

/// Exact world AABB of a collider (planes: None).
fn oracle_aabb(scene: &Scene, c: usize) -> Option<(Vec3, Vec3)> {
    let pose = scene.pose(c);
    match &scene.shapes[c] {
        Shape::Plane(..) => None,
        Shape::Sphere(r) => Some((pose.position - Vec3::splat(*r), pose.position + Vec3::splat(*r))),
        Shape::Capsule(r, l) => {
            let e = (pose.rotation * Vec3::Y * *l).abs() + Vec3::splat(*r);
            Some((pose.position - e, pose.position + e))
        }
        _ => {
            let p = scene.polytope(c).unwrap();
            let lo = p.vertices.iter().copied().fold(Vec3::INFINITY, Vec3::min);
            let hi = p.vertices.iter().copied().fold(Vec3::NEG_INFINITY, Vec3::max);
            Some((lo, hi))
        }
    }
}

fn pipeline_scene(seed: u32) -> Scene {
    let mut seed = seed;
    let lib = hull_library(&mut seed);
    let mut s = Scene::new();
    let ground = s.static_body();
    s.add(ground, Shape::Plane(Vec3::Y, 0.0), Vec3::ZERO, Quat::IDENTITY);
    // Oversized static slab: beyond the top grid level, tested from the large list.
    let slab = s.static_body();
    s.bodies[slab as usize].position = [0.0, -0.5, 0.0];
    s.add(slab, Shape::Hull(ConvexHull::cuboid(Vec3::new(40.0, 0.5, 40.0)), 1.0), Vec3::ZERO, Quat::IDENTITY);
    for i in 0..220 {
        let size = [0.15f32, 0.5, 1.0, 2.2][i % 4];
        let p = Vec3::new(rng(&mut seed) * 14.0, 0.2 + rng(&mut seed) * 4.0, rng(&mut seed) * 14.0);
        let b = s.body(p, random_rot(&mut seed));
        match i % 7 {
            0 => s.bodies[b as usize].flags = BodyFlags::Dynamic as u32 | BodyFlags::Sleeping as u32,
            1 => {
                s.bodies[b as usize].inv_mass = 0.0;
                s.bodies[b as usize].flags = 0;
            }
            _ => {}
        }
        let shape = match i % 11 {
            3 => Shape::Sphere(0.3 * size),
            5 => Shape::Capsule(0.15 * size, 0.3 * size),
            8 => Shape::Box(Vec3::new(0.4, 0.3, 0.2) * size),
            _ => Shape::Hull(lib[i % lib.len()].clone(), size),
        };
        s.add(b, shape, Vec3::ZERO, Quat::IDENTITY);
        if i % 4 == 1 {
            // Compound child far from the body origin.
            s.add(b, Shape::Hull(lib[(i + 3) % lib.len()].clone(), 0.5 * size), Vec3::new(1.2 * size, 0.0, 0.0), random_rot(&mut seed));
        }
    }
    s
}

#[test]
fn hierarchical_broadphase_finds_every_overlap_once() {
    for seed in [5u32, 17] {
        let scene = pipeline_scene(seed);
        let out = run_pipeline(&scene, 1 << 15);
        assert_eq!(out.counters.overflow_flags, 0, "overflow 0x{:x}", out.counters.overflow_flags);
        // Exact hull/box bounds (padded 1e-4).
        for c in 0..scene.colliders.len() {
            if let (Some((lo, hi)), true) = (oracle_aabb(&scene, c), matches!(scene.shapes[c], Shape::Hull(..))) {
                let g = out.aabbs[c];
                let (glo, ghi) = (Vec3::new(g[0], g[1], g[2]), Vec3::new(g[4], g[5], g[6]));
                assert!((glo - (lo - Vec3::splat(1e-4))).abs().max_element() < 2e-4, "collider {c} min {glo} vs {lo}");
                assert!((ghi - (hi + Vec3::splat(1e-4))).abs().max_element() < 2e-4, "collider {c} max {ghi} vs {hi}");
            }
        }
        let mut seen = std::collections::HashSet::new();
        for p in &out.candidates {
            let (a, b) = (p.collider_a as usize, p.collider_b as usize);
            assert!(a < b, "canonical order");
            assert!(seen.insert((a, b)), "duplicate candidate {a},{b}");
            assert_ne!(scene.colliders[a].body, scene.colliders[b].body, "same-body pair {a},{b}");
            assert!(moving(&scene, a) || moving(&scene, b), "pair {a},{b} has no moving body");
        }
        let mut required = 0;
        for a in 0..scene.colliders.len() {
            for b in a + 1..scene.colliders.len() {
                if scene.colliders[a].body == scene.colliders[b].body || !(moving(&scene, a) || moving(&scene, b)) {
                    continue;
                }
                let (Some(ba), Some(bb)) = (oracle_aabb(&scene, a), oracle_aabb(&scene, b)) else {
                    // Plane pairs: required when the collider reaches the plane.
                    let other = if oracle_aabb(&scene, a).is_none() { b } else { a };
                    if oracle_aabb(&scene, other).is_some_and(|(lo, _)| lo.y < -1e-3) && moving(&scene, other) {
                        required += 1;
                        assert!(seen.contains(&(a, b)), "missed plane pair {a},{b}");
                    }
                    continue;
                };
                if !(ba.0.cmple(bb.1).all() && bb.0.cmple(ba.1).all()) {
                    continue;
                }
                // Hull pairs and box/box pairs use exact separation (the broadphase
                // OBB test may reject AABB overlaps that a face axis separates).
                let involves_hull = [a, b].iter().any(|&c| matches!(scene.shapes[c], Shape::Hull(..)))
                    || [a, b].iter().all(|&c| matches!(scene.shapes[c], Shape::Box(..)));
                let sep = if involves_hull { oracle_separation(&scene, a, b) } else { -1.0 };
                if involves_hull && sep > -AMBIGUOUS {
                    continue;
                }
                required += 1;
                let kind = |c: usize| match scene.shapes[c] { Shape::Hull(..) => "hull", Shape::Box(..) => "box", Shape::Sphere(..) => "sphere", Shape::Capsule(..) => "capsule", Shape::Plane(..) => "plane" };
                assert!(seen.contains(&(a, b)), "seed {seed}: missed pair {a},{b} {}/{} moving {}/{} (separation {sep})", kind(a), kind(b), moving(&scene, a), moving(&scene, b));
            }
        }
        assert!(required > 50, "fixture produced only {required} required pairs");
        eprintln!("seed {seed}: {} candidates, {} required overlaps, {} manifolds", out.candidates.len(), required, out.counters.manifold_count);
        // Hull pairs found by the broadphase reach the convex stages and agree
        // with the oracles.
        let hull_pairs: Vec<CandidatePair> = out
            .candidates
            .iter()
            .filter(|p| [p.collider_a, p.collider_b].iter().any(|&c| matches!(scene.shapes[c as usize], Shape::Hull(..))))
            .copied()
            .collect();
        check(&scene, &hull_pairs, &out, &format!("pipeline seed {seed}"));
    }
}


#[test]
fn whole_step_manifold_reuse_carries_every_run_once() {
    let capacity = 1usize << 15;
    let scene = pipeline_scene(5);
    let out = run_pipeline_then_reuse(&scene, capacity);
    assert_eq!(out.counters.overflow_flags, 0);
    let prev = &out.prev_contacts;
    assert!(prev.len() > 100, "fixture produced {} contacts", prev.len());
    assert_eq!(out.contacts.len(), prev.len(), "every previous contact is carried");
    assert_eq!(out.counters.manifold_count, out.prev_manifolds, "one manifold per previous run");
    // Canonical storage tags reused points with alive = source index + 2; each
    // manifold's points are consecutive sources from the head of one run.
    let cur_off = capacity * 8; // the reuse wrote into the second contact half
    let mut seen = vec![false; prev.len()];
    for m in &out.manifolds {
        assert!(m.contact_count > 0 && m.contact_count <= 8);
        let local = m.contact_start as usize - cur_off;
        let head = (out.contacts[local].alive - 2) as usize;
        assert!(head == 0 || prev[head - 1].collider_a != m.collider_a || prev[head - 1].collider_b != m.collider_b, "run head");
        for k in 0..m.contact_count as usize {
            let src = (out.contacts[local + k].alive - 2) as usize;
            assert_eq!(src, head + k, "consecutive sources");
            assert!(!seen[src], "source {src} carried twice");
            seen[src] = true;
            assert_eq!((prev[src].collider_a, prev[src].collider_b), (m.collider_a, m.collider_b));
        }
        assert_eq!(m.normal, prev[head].normal);
    }
    assert!(seen.iter().all(|&s| s), "a previous contact was dropped");
}

#[test]
fn collision_filters_remove_pairs_and_keep_the_rest() {
    let mut scene = pipeline_scene(17);
    // Alternate categories, one negative group, one positive group that
    // overrides a mask which would otherwise exclude it.
    for (k, c) in scene.colliders.iter_mut().enumerate() {
        if matches!(scene.shapes[k], Shape::Plane(..)) {
            continue;
        }
        let f = match k % 5 {
            0 => CollisionFilter { categories: 0b01, mask: 0b01, group: 0 },
            1 => CollisionFilter { categories: 0b10, mask: 0b11, group: 0 },
            2 => CollisionFilter { categories: 0b01, mask: 0b10, group: -3 },
            3 => CollisionFilter { categories: 0b10, mask: 0b10, group: 4 },
            _ => CollisionFilter::default(),
        };
        f.apply(c);
    }
    let out = run_pipeline(&scene, 1 << 15);
    assert_eq!(out.counters.overflow_flags, 0);
    let mut seen = std::collections::HashSet::new();
    for p in &out.candidates {
        let (a, b) = (p.collider_a as usize, p.collider_b as usize);
        assert!(CollisionFilter::allows(&scene.colliders[a], &scene.colliders[b]), "filtered pair {a},{b} emitted");
        seen.insert((a, b));
    }
    let mut required = 0;
    for a in 0..scene.colliders.len() {
        for b in a + 1..scene.colliders.len() {
            if scene.colliders[a].body == scene.colliders[b].body || !(moving(&scene, a) || moving(&scene, b)) {
                continue;
            }
            if !CollisionFilter::allows(&scene.colliders[a], &scene.colliders[b]) {
                continue;
            }
            let (Some(ba), Some(bb)) = (oracle_aabb(&scene, a), oracle_aabb(&scene, b)) else { continue };
            if !(ba.0.cmple(bb.1).all() && bb.0.cmple(ba.1).all()) {
                continue;
            }
            let polytopes = [a, b].iter().all(|&c| matches!(scene.shapes[c], Shape::Hull(..) | Shape::Box(..)));
            let involves_hull = [a, b].iter().any(|&c| matches!(scene.shapes[c], Shape::Hull(..)));
            if (polytopes || involves_hull) && oracle_separation(&scene, a, b) > -AMBIGUOUS {
                continue;
            }
            required += 1;
            assert!(seen.contains(&(a, b)), "allowed pair {a},{b} missed");
        }
    }
    assert!(required > 20, "fixture kept only {required} allowed overlaps");
}
