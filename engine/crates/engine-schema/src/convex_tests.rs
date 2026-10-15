use super::*;
use bytemuck::Zeroable;
use glam::Quat;

/// Mirror of the Slang `hull*` accessors: decode one packed hull record.
struct View<'a> {
    w: Vec<u32>,
    _r: &'a [[u32; 4]],
}

impl<'a> View<'a> {
    fn new(records: &'a [[u32; 4]], base: u32) -> Self {
        let w = records[base as usize..].iter().flatten().copied().collect();
        Self { w, _r: records }
    }
    fn counts(&self) -> (usize, usize, usize, u32) {
        let h = self.w[0];
        ((h & 255) as usize, (h >> 8 & 255) as usize, (h >> 16 & 255) as usize, h >> 24)
    }
    fn f(&self, word: usize) -> f32 {
        f32::from_bits(self.w[word])
    }
    fn vertex(&self, i: usize) -> Vec3 {
        let o = (4 + i) * 4;
        Vec3::new(self.f(o), self.f(o + 1), self.f(o + 2))
    }
    fn vertex_adj(&self, i: usize) -> (usize, usize) {
        let x = self.w[(4 + i) * 4 + 3];
        ((x & 0xffff) as usize, (x >> 16) as usize)
    }
    fn plane(&self, k: usize) -> (Vec3, f32) {
        let (v, ..) = self.counts();
        let o = (4 + v + k) * 4;
        (Vec3::new(self.f(o), self.f(o + 1), self.f(o + 2)), self.f(o + 3))
    }
    fn byte(&self, word: usize, b: usize) -> u8 {
        (self.w[word + b / 4] >> ((b % 4) * 8) & 255) as u8
    }
    fn face_loop(&self, k: usize) -> Vec<usize> {
        let (v, f, ..) = self.counts();
        let entry = self.w[(4 + v + f) * 4 + k];
        let (start, count) = ((entry & 0xffff) as usize, (entry >> 16) as usize);
        (0..count).map(|i| self.byte(self.w[1] as usize, start + i) as usize).collect()
    }
    fn edge(&self, k: usize) -> [usize; 4] {
        let x = self.w[self.w[2] as usize + k];
        [(x & 255) as usize, (x >> 8 & 255) as usize, (x >> 16 & 255) as usize, (x >> 24) as usize]
    }
    fn adj(&self, b: usize) -> usize {
        self.byte(self.w[3] as usize, b) as usize
    }
}

fn bipyramid(n: usize) -> (Vec<Vec3>, Vec<u32>, Vec<u32>) {
    let mut v: Vec<Vec3> = (0..n)
        .map(|i| {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            Vec3::new(t.cos(), 0.0, t.sin())
        })
        .collect();
    v.push(Vec3::Y * 1.5);
    v.push(-Vec3::Y * 1.5);
    let (top, bottom) = (n as u32, n as u32 + 1);
    let mut idx = Vec::new();
    for i in 0..n as u32 {
        let j = (i + 1) % n as u32;
        idx.extend_from_slice(&[i, top, j]);
        idx.extend_from_slice(&[j, bottom, i]);
    }
    (v, vec![3; 2 * n], idx)
}

fn check_hull(h: &ConvexHull) {
    let interior = h.interior();
    for f in &h.faces {
        assert!((f.normal.length() - 1.0).abs() < 1e-5);
        assert!(f.normal.dot(interior) < f.d, "interior behind every face");
        for p in &h.vertices {
            assert!(f.normal.dot(*p) <= f.d + 1e-4, "vertex outside face plane");
        }
        // Counter-clockwise about the outward normal.
        let mut sum = Vec3::ZERO;
        let p0 = h.vertices[f.indices[0] as usize];
        for k in 1..f.indices.len() - 1 {
            sum += (h.vertices[f.indices[k] as usize] - p0)
                .cross(h.vertices[f.indices[k + 1] as usize] - p0);
        }
        assert!(sum.dot(f.normal) > 0.0, "loop winding");
    }
    if h.closed {
        // Euler characteristic over referenced vertices.
        let mut used: Vec<u8> = h.faces.iter().flat_map(|f| f.indices.clone()).collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len() + h.faces.len(), h.edges.len() + 2);
        for e in &h.edges {
            let left = &h.faces[e[2] as usize].indices;
            let right = &h.faces[e[3] as usize].indices;
            let walks = |l: &Vec<u8>, a: u8, b: u8| {
                (0..l.len()).any(|k| l[k] == a && l[(k + 1) % l.len()] == b)
            };
            assert!(walks(left, e[0], e[1]) && walks(right, e[1], e[0]));
        }
    }
}

#[test]
fn cuboid_topology_and_mass() {
    let h = ConvexHull::cuboid(Vec3::new(1.0, 0.5, 0.25));
    assert_eq!((h.vertices.len(), h.faces.len(), h.edges.len()), (8, 6, 12));
    assert!(h.closed);
    check_hull(&h);
    let m = h.mass_properties(2.0);
    let (x, y, z) = (2.0f32, 1.0f32, 0.5f32);
    assert!((m.volume - x * y * z).abs() < 1e-5);
    assert!(m.center_of_mass.length() < 1e-5);
    let mass = 2.0 * x * y * z;
    let expect = Vec3::new(y * y + z * z, x * x + z * z, x * x + y * y) * (mass / 12.0);
    let diag = Vec3::new(m.inertia.x_axis.x, m.inertia.y_axis.y, m.inertia.z_axis.z);
    assert!((diag - expect).abs().max_element() < 1e-4, "{diag} vs {expect}");
    assert!(m.inertia.x_axis.y.abs() < 1e-5);
}

#[test]
fn point_cloud_hull_is_closed_and_convex() {
    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    let pts: Vec<Vec3> = (0..40).map(|_| Vec3::new(rnd(), rnd() * 0.7, rnd() * 0.4)).collect();
    let h = ConvexHull::from_points(&pts).unwrap();
    assert!(h.closed);
    check_hull(&h);
    for p in &pts {
        for f in &h.faces {
            assert!(f.normal.dot(*p) <= f.d + 1e-4);
        }
    }
}

#[test]
fn large_clouds_are_reduced_to_64_vertices() {
    let pts: Vec<Vec3> = (0..500)
        .map(|i| {
            let phi = (i as f32 + 0.5) / 500.0;
            let theta = i as f32 * 2.399_963;
            let r = (1.0 - (2.0 * phi - 1.0).powi(2)).sqrt();
            Vec3::new(r * theta.cos(), 2.0 * phi - 1.0, r * theta.sin())
        })
        .collect();
    let h = ConvexHull::from_points(&pts).unwrap();
    assert!(h.vertices.len() <= HULL_MAX_VERTICES);
    assert!(h.faces.len() <= HULL_MAX_FACES && h.edges.len() <= HULL_MAX_EDGES);
    assert!(h.closed);
    check_hull(&h);
}

#[test]
fn bipyramid_64_keeps_124_faces_186_edges() {
    let (v, s, i) = bipyramid(62);
    let h = ConvexHull::from_faces(&v, &s, &i).unwrap();
    assert_eq!((h.vertices.len(), h.faces.len(), h.edges.len()), (64, 124, 186));
    assert!(h.closed);
    check_hull(&h);
}

#[test]
fn coplanar_triangles_merge_and_duplicates_weld() {
    // Cube faces split into triangles, with a duplicated corner position.
    let mut v: Vec<Vec3> = ConvexHull::cuboid(Vec3::ONE).vertices;
    v.push(v[0]);
    let h = ConvexHull::from_points(&v).unwrap();
    assert_eq!(h.faces.len(), 6);
    assert!(h.faces.iter().all(|f| f.indices.len() == 4));
    // Face soup referencing the duplicate still welds into a closed hull.
    let cube = ConvexHull::cuboid(Vec3::ONE);
    let mut sizes = Vec::new();
    let mut idx = Vec::new();
    for f in &cube.faces {
        sizes.push(f.indices.len() as u32);
        idx.extend(f.indices.iter().map(|&k| if k == 0 { 8 } else { k as u32 }));
    }
    let h = ConvexHull::from_faces(&v, &sizes, &idx).unwrap();
    assert!(h.closed);
    assert_eq!(h.vertices.len(), 9, "original vertex records are kept");
}

#[test]
fn flat_and_tiny_inputs_are_rejected() {
    let flat: Vec<Vec3> = (0..6).map(|i| Vec3::new(i as f32, (i * i) as f32, 0.0)).collect();
    assert_eq!(ConvexHull::from_points(&flat), Err(HullError::ZeroVolume));
    assert_eq!(
        ConvexHull::from_points(&[Vec3::ZERO, Vec3::X, Vec3::Y]),
        Err(HullError::VertexCount(3))
    );
}

#[test]
fn packed_records_decode_like_the_gpu() {
    let (v, s, i) = bipyramid(30);
    let hulls = [ConvexHull::cuboid(Vec3::new(0.3, 0.2, 0.1)), ConvexHull::from_faces(&v, &s, &i).unwrap()];
    let mut arena = HullArena::new();
    for h in &hulls {
        let base = arena.insert(h).unwrap();
        let view = View::new(arena.records(), base);
        let (nv, nf, ne, flags) = view.counts();
        assert_eq!((nv, nf, ne), (h.vertices.len(), h.faces.len(), h.edges.len()));
        assert_eq!(flags & HULL_FLAG_CLOSED != 0, h.closed);
        for k in 0..nv {
            assert_eq!(view.vertex(k), h.vertices[k]);
        }
        for (k, f) in h.faces.iter().enumerate() {
            assert_eq!(view.plane(k), (f.normal, f.d));
            assert_eq!(view.face_loop(k), f.indices.iter().map(|&x| x as usize).collect::<Vec<_>>());
        }
        for (k, e) in h.edges.iter().enumerate() {
            assert_eq!(view.edge(k), e.map(|x| x as usize));
        }
        // Every vertex lists exactly its incident edges.
        for k in 0..nv {
            let (start, deg) = view.vertex_adj(k);
            let listed: Vec<usize> = (0..deg).map(|j| view.adj(start + j)).collect();
            let expect: Vec<usize> = (0..ne).filter(|&e| h.edges[e][0] as usize == k || h.edges[e][1] as usize == k).collect();
            assert_eq!(listed, expect);
        }
        assert_eq!(arena.records().len() - base as usize, h.packed_records());
    }
}

#[test]
fn arena_starts_with_unit_box_and_dedups() {
    let mut arena = HullArena::new();
    let view = View::new(arena.records(), UNIT_BOX_HULL_BASE);
    assert_eq!(view.counts().0, 8);
    let before = arena.len();
    let h = ConvexHull::cuboid(Vec3::splat(2.0));
    let a = arena.insert(&h).unwrap();
    let b = arena.insert(&h).unwrap();
    assert_eq!(a, b);
    assert_eq!(arena.len(), before + h.packed_records());
    let p = convex_collider_params(a, 1.5, h.bounding_radius());
    assert!(p[0].is_normal());
    assert_eq!(convex_params_base(p), a);
    assert!((p[3] - 1.5 * 12f32.sqrt()).abs() < 1e-5);
}

#[test]
fn recentering_moves_offset_into_local_pos() {
    let pts: Vec<Vec3> = ConvexHull::cuboid(Vec3::ONE)
        .vertices
        .iter()
        .map(|v| *v + Vec3::new(50.0, 0.0, -3.0))
        .collect();
    let h = ConvexHull::from_points(&pts).unwrap();
    let (c, offset) = h.recentered();
    assert!((offset - Vec3::new(50.0, 0.0, -3.0)).length() < 1e-4);
    assert!((c.bounding_radius() - 3f32.sqrt()).abs() < 1e-4);
    check_hull(&c);
}

#[test]
fn oracle_detects_overlap_and_separation() {
    let a = ConvexHull::cuboid(Vec3::ONE);
    let pose = |p: Vec3, q: Quat| Pose { position: p, rotation: q };
    let rot = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3);
    let hit = sat_oracle(&a, Vec3::ONE, pose(Vec3::ZERO, Quat::IDENTITY), &a, Vec3::ONE, pose(Vec3::new(1.9, 0.0, 0.0), rot));
    assert!(hit.is_some());
    let (n, depth) = hit.unwrap();
    assert!(depth < 0.0 && n.x > 0.5, "{n} {depth}");
    let miss = sat_oracle(&a, Vec3::ONE, pose(Vec3::ZERO, Quat::IDENTITY), &a, Vec3::ONE, pose(Vec3::new(3.8, 0.0, 0.0), rot));
    assert!(miss.is_none());
}

#[test]
fn compound_body_has_principal_frame_and_offset_children() {
    // Two unit cubes side by side along a rotated axis: an L-free bar.
    let rot = Quat::from_rotation_z(0.6);
    let a: Vec<Vec3> = ConvexHull::cuboid(Vec3::ONE).vertices.iter().map(|v| rot * (*v + Vec3::X * 1.0)).collect();
    let b: Vec<Vec3> = ConvexHull::cuboid(Vec3::ONE).vertices.iter().map(|v| rot * (*v - Vec3::X * 1.0) + Vec3::Y * 5.0).collect();
    let ha = ConvexHull::from_points(&a).unwrap();
    let hb = ConvexHull::from_points(&b).unwrap();
    let body = cook_body(&[ha.clone(), hb.clone()], 1.0);
    assert!((body.mass - 16.0).abs() < 1e-3);
    assert!((body.center_of_mass - Vec3::Y * 2.5).length() < 1e-4);
    // Children are recentred and their offsets reconstruct the input geometry.
    for (child, src) in body.children.iter().zip([&ha, &hb]) {
        assert!(child.hull.bounding_radius() < 1.8, "child radius stays local");
        check_hull(&child.hull);
        for (v, s) in child.hull.vertices.iter().zip(&src.vertices) {
            let input = body.center_of_mass + body.principal * (*v + child.local_pos);
            assert!((input - *s).length() < 1e-4, "{input} vs {s}");
        }
    }
    // Principal moments match the tensor rotated into the body frame.
    let parts = [ha.mass_properties(1.0), hb.mass_properties(1.0)];
    let mut t = Mat3::ZERO;
    for m in &parts {
        let d = m.center_of_mass - body.center_of_mass;
        t += m.inertia + (Mat3::from_diagonal(Vec3::splat(d.length_squared())) - Mat3::from_cols(d * d.x, d * d.y, d * d.z)) * m.mass;
    }
    let r = Mat3::from_quat(body.principal);
    let local = r.transpose() * t * r;
    assert!(local.x_axis.y.abs() < 1e-2 && local.x_axis.z.abs() < 1e-2 && local.y_axis.z.abs() < 1e-2, "{local:?}");
    let diag = Vec3::new(local.x_axis.x, local.y_axis.y, local.z_axis.z);
    assert!((diag - body.inertia).abs().max_element() < 1e-2, "{diag} vs {}", body.inertia);
}

/// Face-soup fixtures shared with fracture/convex.test.mjs. Integer words must
/// match the JS packer exactly; float words to f32 round-off (f64 vs f32 sums).
type FaceSoup = (&'static str, Vec<Vec3>, Vec<u32>, Vec<u32>);

fn golden_inputs() -> Vec<FaceSoup> {
    let prism = |n: usize, dup: bool| {
        let mut v: Vec<Vec3> = Vec::new();
        for y in [-1.0f32, 1.0] {
            for i in 0..n {
                let t = i as f32 / n as f32 * std::f32::consts::TAU;
                v.push(Vec3::new(t.cos(), y, t.sin()));
            }
        }
        let mut faces: Vec<Vec<u32>> = vec![(0..n as u32).collect(), (n as u32..2 * n as u32).collect()];
        for i in 0..n as u32 {
            let j = (i + 1) % n as u32;
            faces.push(vec![i, j, j + n as u32, i + n as u32]);
        }
        if dup {
            let base = v.len() as u32;
            v.extend(v.clone());
            let more: Vec<Vec<u32>> = faces.iter().map(|f| f.iter().map(|k| k + base).collect()).collect();
            faces.extend(more);
        }
        let sizes = faces.iter().map(|f| f.len() as u32).collect();
        (v, sizes, faces.concat())
    };
    let (bv, bs, bi) = bipyramid(10);
    let cube = ConvexHull::cuboid(Vec3::ONE);
    let mut tri = Vec::new();
    for f in &cube.faces {
        for k in 1..f.indices.len() - 1 {
            tri.extend([f.indices[0] as u32, f.indices[k] as u32, f.indices[k + 1] as u32]);
        }
    }
    let (p5, s5, i5) = prism(5, false);
    let (pd, sd, id) = prism(4, true);
    vec![
        ("prism5", p5, s5, i5),
        ("prism4_duplicated", pd, sd, id),
        ("bipyramid12", bv, bs, bi),
        ("cube_triangles", cube.vertices.clone(), vec![3; tri.len() / 3], tri),
    ]
}

#[test]
fn golden_packing_fixture_is_current() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/hull-golden.json");
    let mut entries = Vec::new();
    for (name, v, s, i) in golden_inputs() {
        let mut out = Vec::new();
        ConvexHull::from_faces(&v, &s, &i).unwrap().pack(&mut out).unwrap();
        let words: Vec<String> = out.iter().flatten().map(|w| w.to_string()).collect();
        let verts: Vec<String> = v.iter().map(|p| format!("[{},{},{}]", p.x, p.y, p.z)).collect();
        entries.push(format!(
            "  {{\"name\":\"{name}\",\"vertices\":[{}],\"sizes\":{s:?},\"indices\":{i:?},\"words\":[{}]}}",
            verts.join(","),
            words.join(",")
        ));
    }
    let text = format!("[\n{}\n]\n", entries.join(",\n"));
    if std::env::var_os("UPDATE_HULL_GOLDEN").is_some() {
        std::fs::write(&path, &text).unwrap();
    }
    let current = std::fs::read_to_string(&path)
        .expect("missing fracture/hull-golden.json; run with UPDATE_HULL_GOLDEN=1");
    assert_eq!(current, text, "hull ABI changed; rerun with UPDATE_HULL_GOLDEN=1 and the Node golden test");
}

#[test]
fn arena_refcounts_reuses_freed_blocks_and_tracks_dirty_ranges() {
    let mut arena = HullArena::new();
    arena.take_dirty();
    let a = ConvexHull::cuboid(Vec3::new(0.5, 0.5, 0.5));
    let b = ConvexHull::cuboid(Vec3::new(0.7, 0.2, 0.3));
    let c = ConvexHull::cuboid(Vec3::new(0.1, 0.2, 0.3));
    let ba = arena.insert(&a).unwrap();
    let ba2 = arena.insert(&a).unwrap();
    assert_eq!(ba, ba2);
    assert_eq!(arena.refs(ba), 2);
    let bb = arena.insert(&b).unwrap();
    let bc = arena.insert(&c).unwrap();
    let dirty = arena.take_dirty();
    assert_eq!(dirty, vec![ba..bc + c.packed_records() as u32], "contiguous inserts merge");
    // One reference left: not freed.
    assert!(!arena.release(ba));
    assert!(arena.release(ba));
    assert!(!arena.release(UNIT_BOX_HULL_BASE), "unit cube is permanent");
    // b freed next: merges with a's hole.
    assert!(arena.release(bb));
    assert_eq!(arena.free_records(), (a.packed_records() + b.packed_records()) as u32);
    // A new hull reuses the freed space (first fit) and only it is dirty.
    let d = ConvexHull::cuboid(Vec3::new(0.3, 0.3, 0.9));
    let bd = arena.insert(&d).unwrap();
    assert_eq!(bd, ba);
    assert_eq!(arena.take_dirty(), vec![bd..bd + d.packed_records() as u32]);
    let view = View::new(arena.records(), bd);
    assert_eq!(view.counts().0, 8);
    // Freeing the tail shrinks the arena.
    let len = arena.len();
    assert!(arena.release(bc));
    assert!(arena.len() < len);
    // Dedup still works after frees.
    assert_eq!(arena.insert(&d).unwrap(), bd);
}

#[test]
fn collision_filter_round_trips_and_matches_box2d_rules() {
    let mut a = crate::Collider::zeroed();
    let mut b = crate::Collider::zeroed();
    a.material = 3;
    assert!(CollisionFilter::allows(&a, &b), "default collides");
    assert!(!CollisionFilter::is_set(&a));
    CollisionFilter { categories: 0b01, mask: 0b10, group: 0 }.apply(&mut a);
    CollisionFilter { categories: 0b10, mask: 0b01, group: 0 }.apply(&mut b);
    assert_eq!(a.material & 0xFFFF, 3, "material index kept");
    assert!(CollisionFilter::allows(&a, &b));
    CollisionFilter { categories: 0b10, mask: 0b10, group: 0 }.apply(&mut b);
    assert!(!CollisionFilter::allows(&a, &b), "a's category not in b's mask");
    CollisionFilter { categories: 0b10, mask: 0b10, group: 7 }.apply(&mut b);
    CollisionFilter { categories: 0b01, mask: 0b10, group: 7 }.apply(&mut a);
    assert!(CollisionFilter::allows(&a, &b), "shared positive group overrides masks");
    CollisionFilter { categories: 0xFFFF, mask: 0xFFFF, group: -2 }.apply(&mut a);
    CollisionFilter { categories: 0xFFFF, mask: 0xFFFF, group: -2 }.apply(&mut b);
    assert!(!CollisionFilter::allows(&a, &b), "shared negative group never collides");
    assert_eq!(CollisionFilter::of(&a).group, -2);
}
