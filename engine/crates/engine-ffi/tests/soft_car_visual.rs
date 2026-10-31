//! The procedural car's render mesh through the C ABI (`rbCreateSoftCarVisual`),
//! skinned part by part (`rbSoft_BindSkinParts`), and the damage a wall crash
//! reports with it: per-vertex strain and tears, per-face strain, and break
//! events with points. Run in Rubble's units and in Unreal's.

use rubble_ffi::coords::RbCoordinates;
use rubble_ffi::*;
use std::ffi::{c_char, CStr, CString};
use std::ptr;

fn ok(r: RbResult) {
    if r != RbResult::RB_OK {
        let msg = unsafe { CStr::from_ptr(rbGetLastError()) };
        panic!("{r:?}: {}", msg.to_string_lossy());
    }
}

/// Rubble (Y-up metres) to the host's axes and units.
#[derive(Clone, Copy)]
struct Host(RbCoordinates);

impl Host {
    fn unreal(self) -> bool {
        self.0 == RbCoordinates::RB_COORDINATES_UNREAL
    }
    fn p(self, x: f32, y: f32, z: f32) -> RbVec3 {
        if self.unreal() {
            RbVec3 {
                x: -z * 100.0,
                y: x * 100.0,
                z: y * 100.0,
            }
        } else {
            RbVec3 { x, y, z }
        }
    }
    fn len(self, m: f32) -> f32 {
        if self.unreal() {
            m * 100.0
        } else {
            m
        }
    }
    /// Half extents (sizes along Rubble's x, y, z) in the host's axes.
    fn half(self, x: f32, y: f32, z: f32) -> RbVec3 {
        if self.unreal() {
            RbVec3 {
                x: z * 100.0,
                y: x * 100.0,
                z: y * 100.0,
            }
        } else {
            RbVec3 { x, y, z }
        }
    }
}

fn at(p: RbVec3) -> RbTransform {
    RbTransform {
        p,
        ..RbTransform::default()
    }
}

unsafe fn json(f: impl Fn(*mut c_char, usize, *mut usize) -> RbResult) -> String {
    let mut n = 0;
    ok(f(ptr::null_mut(), 0, &mut n));
    let mut buf = vec![0u8; n + 1];
    ok(f(buf.as_mut_ptr().cast(), buf.len(), &mut n));
    CStr::from_bytes_with_nul(&buf)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

unsafe fn static_box(w: *mut RbWorld, h: Host, centre: RbVec3, half: RbVec3) -> RbShapeId {
    let mut b = 0;
    ok(rbCreateBody(
        w,
        &RbBodyDef {
            transform: at(centre),
            ..rbDefaultBodyDef()
        },
        &mut b,
    ));
    let mut s = 0;
    ok(rbCreateBoxShape(
        w,
        b,
        &rbDefaultShapeDef(),
        half,
        RbTransform::default(),
        &mut s,
    ));
    let _ = h;
    s
}

struct Visual {
    vertices: Vec<RbVec3>,
    triangles: Vec<u32>,
    materials: Vec<u8>,
    parts: Vec<u8>,
    names: Vec<String>,
}

unsafe fn visual(h: Host) -> Visual {
    let mut v = ptr::null();
    ok(rbCreateSoftCarVisual(
        c"procedural_car".as_ptr(),
        h.0,
        &mut v,
    ));
    let r = &*v;
    let nv = r.vertex_count as usize;
    let nt = r.triangle_count as usize;
    let out = Visual {
        vertices: std::slice::from_raw_parts(r.vertices, nv).to_vec(),
        triangles: std::slice::from_raw_parts(r.triangles, nt * 3).to_vec(),
        materials: std::slice::from_raw_parts(r.triangle_materials, nt).to_vec(),
        parts: std::slice::from_raw_parts(r.vertex_parts, nv).to_vec(),
        names: std::slice::from_raw_parts(r.part_names, r.part_count as usize)
            .iter()
            .map(|&n| CStr::from_ptr(n).to_string_lossy().into_owned())
            .collect(),
    };
    rbSoftVisual_Destroy(v);
    out
}

#[test]
fn the_car_visual_is_the_detailed_mesh() {
    unsafe {
        let v = visual(Host(RbCoordinates::RB_COORDINATES_Y_UP_METRES));
        assert!(v.vertices.len() > 5000, "{} vertices", v.vertices.len());
        assert_eq!(v.materials.len() * 3, v.triangles.len());
        assert!(v.triangles.iter().all(|&i| (i as usize) < v.vertices.len()));
        for m in 0..10u8 {
            assert!(v.materials.contains(&m), "material {m} is used");
        }
        assert_eq!(v.names.len(), 12);
        assert!(v.parts.iter().all(|&p| (p as usize) < v.names.len()));
        // The same mesh in Unreal's axes and centimetres, wound the other way.
        let u = visual(Host(RbCoordinates::RB_COORDINATES_UNREAL));
        let h = Host(RbCoordinates::RB_COORDINATES_UNREAL);
        for (a, b) in v.vertices.iter().zip(&u.vertices) {
            let e = h.p(a.x, a.y, a.z);
            assert!(
                (e.x - b.x).abs() < 1e-3 && (e.y - b.y).abs() < 1e-3 && (e.z - b.z).abs() < 1e-3
            );
        }
        // The paint faces out in each host's winding: Rubble's (b - a) x (c - a), Unreal's
        // (c - a) x (b - a).
        let out = |m: &Visual, unreal: bool| {
            let c = m.vertices.iter().fold([0.0f32; 3], |s, p| [s[0] + p.x, s[1] + p.y, s[2] + p.z]);
            let n = m.vertices.len() as f32;
            let c = [c[0] / n, c[1] / n, c[2] / n];
            let (mut outward, mut paint) = (0, 0);
            for (t, &mat) in m.triangles.chunks_exact(3).zip(&m.materials) {
                if mat != 0 {
                    continue;
                }
                let [a, b, d] = [t[0], t[1], t[2]].map(|i| m.vertices[i as usize]);
                let e1 = [b.x - a.x, b.y - a.y, b.z - a.z];
                let e2 = [d.x - a.x, d.y - a.y, d.z - a.z];
                let mut x = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                if unreal {
                    x = x.map(|k| -k);
                }
                let r = [a.x - c[0], a.y - c[1], a.z - c[2]];
                paint += 1;
                outward += usize::from(x[0] * r[0] + x[1] * r[1] + x[2] * r[2] > 0.0);
            }
            outward as f32 / paint as f32
        };
        let (rubble, unreal) = (out(&v, false), out(&u, true));
        assert!(rubble > 0.8 && unreal > 0.8, "paint facing out: Rubble {rubble}, Unreal {unreal}");
    }
}

struct Crash {
    vertices: usize,
    max_face_strain: f32,
    max_vertex_strain: f32,
    torn_faces: usize,
    torn_vertices: usize,
    shattered_glass: usize,
    beams_broken: usize,
    pieces: Vec<(String, RbVec3)>,
    points_inside: bool,
    /// The bumper mesh's centre from the torn-off bumper's nodes' centre: at the end,
    /// and at rest.
    bumper_miss: (f32, f32),
}

fn centre(vertices: &[RbVec3], parts: &[u8], part: u8) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    let mut n = 0.0;
    for (p, &k) in vertices.iter().zip(parts) {
        if k == part {
            c[0] += p.x;
            c[1] += p.y;
            c[2] += p.z;
            n += 1.0;
        }
    }
    c.map(|x| x / n)
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A procedural car driven into a wall at `speed` m/s (the softbody page's
/// crash test), skinned with its render mesh part by part.
unsafe fn crash(h: Host, speed: f32) -> Crash {
    let mut def = rbDefaultWorldDef();
    def.coordinates = h.0;
    def.gravity = h.p(0.0, -9.81, 0.0);
    def.time_step = 1.0 / 240.0;
    def.enable_soft_bodies = true;
    def.contacts = RbContacts::RB_CONTACTS_CANONICAL;
    def.max_bodies = 8192;
    def.max_colliders = 8192;
    def.max_contacts = 131_072;
    def.max_candidates = 262_144;
    def.max_joints = 4096;
    let mut w = ptr::null_mut();
    ok(rbCreateWorld(&def, &mut w));
    let settings = json(|b, c, n| rbGetCalibratedSoftSettingsJson(240.0, b, c, n));
    let settings = CString::new(settings).unwrap();
    ok(rbWorld_SetSoftSettingsJson(w, settings.as_ptr()));
    let ground = static_box(w, h, h.p(0.0, -0.5, 0.0), h.half(40.0, 0.5, 40.0));
    ok(rbSoft_AddObstacle(w, ground));
    let wall = static_box(w, h, h.p(0.0, 1.5, 6.0), h.half(4.0, 1.5, 0.5));
    ok(rbSoft_AddObstacle(w, wall));
    let desc = json(|b, c, n| rbGetSoftVehiclePresetJson(c"procedural_car".as_ptr(), 0.0, b, c, n));
    let desc = CString::new(desc).unwrap();
    let mut s = 0;
    ok(rbCreateSoftVehicle(
        w,
        desc.as_ptr(),
        ptr::null(),
        at(h.p(0.0, 0.05, 0.0)),
        h.p(0.0, 0.0, speed),
        &mut s,
    ));
    ok(rbSoft_EnableOutput(w, s, true));
    let v = visual(h);
    let names: Vec<CString> = v
        .names
        .iter()
        .map(|n| CString::new(n.as_str()).unwrap())
        .collect();
    let name_ptrs: Vec<*const c_char> = names.iter().map(|n| n.as_ptr()).collect();
    let mut skin = 0;
    ok(rbSoft_BindSkinParts(
        w,
        s,
        v.vertices.as_ptr(),
        v.vertices.len() as u32,
        v.triangles.as_ptr(),
        v.materials.len() as u32,
        v.parts.as_ptr(),
        name_ptrs.as_ptr(),
        name_ptrs.len() as u32,
        &mut skin,
    ));
    let mut out = Crash {
        vertices: 0,
        max_face_strain: 0.0,
        max_vertex_strain: 0.0,
        torn_faces: 0,
        torn_vertices: 0,
        shattered_glass: 0,
        beams_broken: 0,
        pieces: Vec::new(),
        points_inside: true,
        bumper_miss: (0.0, 0.0),
    };
    // The same mesh bound to the whole cage, to compare.
    let mut whole = 0;
    ok(rbSoft_BindSkin(
        w,
        s,
        v.vertices.as_ptr(),
        v.vertices.len() as u32,
        v.triangles.as_ptr(),
        v.materials.len() as u32,
        &mut whole,
    ));
    let bumper = v.names.iter().position(|n| n == "bumper_front").unwrap() as u8;
    let mut piece_at: Option<Vec<u32>> = None;
    let mut nodes: Vec<RbVec3> = Vec::new();
    let mut last: [Vec<RbVec3>; 2] = [Vec::new(), Vec::new()];
    let mut rest: Option<(Vec<RbVec3>, Vec<RbVec3>)> = None;
    for _ in 0..720 {
        ok(rbWorld_Step(w));
        let mut f: RbFrame = std::mem::zeroed();
        ok(rbWorld_Poll(w, true, true, &mut f));
        if f.skin_count == 0 {
            continue;
        }
        let all = std::slice::from_raw_parts(f.skins, f.skin_count as usize);
        for k in all {
            let i = usize::from(k.skin == whole);
            last[i] = std::slice::from_raw_parts(k.positions, k.vertex_count as usize).to_vec();
        }
        let k = all.iter().find(|k| k.skin == skin).unwrap();
        if f.soft_count > 0 {
            let o = &*f.soft;
            nodes = std::slice::from_raw_parts(o.nodes, o.node_count as usize).to_vec();
            if rest.is_none() {
                rest = Some((nodes.clone(), last[0].clone()));
            }
        }
        let nv = k.vertex_count as usize;
        let nt = k.triangle_count as usize;
        assert_eq!(nt, v.materials.len(), "a strain per triangle");
        let pos = std::slice::from_raw_parts(k.positions, nv);
        let torn = std::slice::from_raw_parts(k.torn, nv);
        let strain = std::slice::from_raw_parts(k.strain, nv);
        let faces = std::slice::from_raw_parts(k.face_strain, nt);
        out.vertices = nv;
        out.max_vertex_strain = strain.iter().copied().fold(out.max_vertex_strain, f32::max);
        out.max_face_strain = faces.iter().copied().fold(out.max_face_strain, f32::max);
        out.torn_faces = out
            .torn_faces
            .max(faces.iter().filter(|&&x| x < 0.0).count());
        out.torn_vertices = out
            .torn_vertices
            .max(torn.iter().filter(|&&t| t != 0).count());
        out.shattered_glass = out.shattered_glass.max(
            faces
                .iter()
                .zip(&v.materials)
                .filter(|(&x, &m)| m == 1 && (x < 0.0 || x > 0.06))
                .count(),
        );
        // The skin's bounds, a metre wider, hold every break point.
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in pos {
            for (i, c) in [p.x, p.y, p.z].into_iter().enumerate() {
                lo[i] = lo[i].min(c);
                hi[i] = hi[i].max(c);
            }
        }
        let margin = h.len(1.0);
        for e in std::slice::from_raw_parts(f.destruction, f.destruction_count as usize) {
            if e.soft != s {
                continue;
            }
            match e.type_ {
                RbDestructionType::RB_BEAM_BROKE => out.beams_broken += 1,
                RbDestructionType::RB_PIECE_DETACHED => {
                    let name = CStr::from_ptr(e.name).to_string_lossy().into_owned();
                    if name.split(',').any(|p| p == "bumper_front") {
                        piece_at = Some(
                            std::slice::from_raw_parts(e.nodes, e.node_count as usize).to_vec(),
                        );
                    }
                    out.pieces.push((name, e.point));
                }
                _ => continue,
            }
            assert!(e.has_point, "a soft break has a point");
            let p = [e.point.x, e.point.y, e.point.z];
            if (0..3).any(|i| p[i] < lo[i] - margin || p[i] > hi[i] + margin) {
                out.points_inside = false;
            }
        }
    }
    if let (Some(piece), Some((rest_nodes, rest_skin))) = (piece_at, rest) {
        let at = |nodes: &[RbVec3]| {
            let mut c = [0.0f32; 3];
            for &i in &piece {
                let p = nodes[i as usize];
                c[0] += p.x / piece.len() as f32;
                c[1] += p.y / piece.len() as f32;
                c[2] += p.z / piece.len() as f32;
            }
            c
        };
        out.bumper_miss = (
            dist(centre(&last[0], &v.parts, bumper), at(&nodes)),
            dist(centre(&rest_skin, &v.parts, bumper), at(&rest_nodes)),
        );
    }
    rbDestroyWorld(w);
    out
}

#[test]
fn a_wall_crash_reports_strain_tears_and_break_points() {
    for c in [
        RbCoordinates::RB_COORDINATES_Y_UP_METRES,
        RbCoordinates::RB_COORDINATES_UNREAL,
    ] {
        let h = Host(c);
        let r = unsafe { crash(h, 15.6) };
        eprintln!(
            "{c:?}: bumper miss {:?}, vertices {} max strain face {:.3} vertex {:.3}, torn faces {} vertices {}, glass shattered {}, beams broken {}, pieces {:?}",
            r.bumper_miss, r.vertices, r.max_face_strain, r.max_vertex_strain, r.torn_faces, r.torn_vertices, r.shattered_glass, r.beams_broken, r.pieces
        );
        assert!(r.vertices > 5000);
        assert!(
            r.max_face_strain > 0.06,
            "the front crumples: {}",
            r.max_face_strain
        );
        assert!(r.max_vertex_strain > 0.06);
        assert!(r.beams_broken > 0, "beams break");
        assert!(r.points_inside, "break points lie on the car");
        assert!(
            r.pieces
                .iter()
                .any(|(n, _)| n.split(',').any(|p| p == "bumper_front")),
            "the front bumper tears off: {:?}",
            r.pieces
        );
        // Bound to its own nodes, the bumper's mesh leaves with the torn-off bumper: it
        // ends about where it sat on the bumper at rest.
        assert!(
            (r.bumper_miss.0 - r.bumper_miss.1).abs() < h.len(0.1),
            "the bumper mesh rides the bumper: {:?}",
            r.bumper_miss
        );
    }
}
