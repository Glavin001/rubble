use super::*;
use glam::Quat;

fn lattice() -> (Vec<u32>, Vec<Vec3>) {
    let mut p = Vec::new();
    for i in 0..5 {
        for j in 0..4 {
            for k in 0..2 {
                p.push(Vec3::new(
                    i as f32 * 0.5,
                    j as f32 * 0.45,
                    k as f32 * 0.08 + 0.02 * (i as f32).sin(),
                ));
            }
        }
    }
    ((0..p.len() as u32).collect(), p)
}

fn verts() -> (Vec<Vec3>, Vec<Vec3>) {
    let v: Vec<Vec3> = (0..200)
        .map(|k| {
            let f = k as f32;
            Vec3::new(
                0.2 + 1.6 * (f * 0.37).fract(),
                0.1 + 1.1 * (f * 0.61).fract(),
                0.09 + 0.01 * (f * 0.13).fract(),
            )
        })
        .collect();
    let n: Vec<Vec3> = (0..200)
        .map(|k| Vec3::new(0.1 * (k as f32 * 0.7).sin(), 0.05, 1.0).normalize())
        .collect();
    (v, n)
}

#[test]
fn rest_and_rigid_motion_are_exact() {
    let (ids, p) = lattice();
    let (v, n) = verts();
    let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
    let q = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3);
    let t = Vec3::new(3.0, -1.0, 2.0);
    let moved: Vec<Vec3> = p.iter().map(|x| q * *x + t).collect();
    for ((s, x), nn) in sk.iter().zip(&v).zip(&n) {
        let sum: f32 = s.w.iter().map(|e| e.1).sum();
        assert!((sum - 1.0).abs() < 1e-5, "partition of unity {sum}");
        let (a, an) = eval(&s.w, s.off, s.nrm, &p);
        assert!(
            a.distance(*x) < 2e-5 && an.angle_between(*nn) < 1e-3,
            "rest"
        );
        let (b, bn) = eval(&s.w, s.off, s.nrm, &moved);
        assert!(
            b.distance(q * *x + t) < 2e-5,
            "rigid {}",
            b.distance(q * *x + t)
        );
        assert!(bn.angle_between(q * *nn) < 1e-3);
    }
}

#[test]
fn a_smooth_bend_moves_the_skin_smoothly() {
    let (ids, p) = lattice();
    let (v, n) = verts();
    let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
    // Bend about the y axis: x → arc.
    let bend = |x: Vec3| {
        let r = 2.0;
        let a = x.x / r;
        Vec3::new((r - x.z) * a.sin(), x.y, r - (r - x.z) * a.cos())
    };
    let moved: Vec<Vec3> = p.iter().map(|x| bend(*x)).collect();
    for (s, x) in sk.iter().zip(&v) {
        let (b, _) = eval(&s.w, s.off, s.nrm, &moved);
        assert!(b.distance(bend(*x)) < 0.02, "{}", b.distance(bend(*x)));
    }
}

#[test]
fn the_sidecar_round_trips() {
    let (ids, p) = lattice();
    let (v, n) = verts();
    let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
    let s = Skin {
        nodes: ids.iter().map(|i| format!("n{i}")).collect(),
        meshes: vec![MeshSkin {
            name: "panel".into(),
            verts: sk,
        }],
    };
    assert_eq!(Skin::decode(&s.encode()).unwrap(), s);
}

/// Two meshes on the lattice's nodes.
fn two_meshes() -> Skin {
    let (ids, p) = lattice();
    let (v, n) = verts();
    let (sk, _) = bind(&ids, &p, &v, &n).unwrap();
    Skin {
        nodes: ids.iter().map(|i| format!("n{i}")).collect(),
        meshes: vec![
            MeshSkin {
                name: "panel".into(),
                verts: sk[..120].to_vec(),
            },
            MeshSkin {
                name: "door".into(),
                verts: sk[120..].to_vec(),
            },
        ],
    }
}

/// The version 1 sidecar (f32 entries, per vertex its first entry), as the
/// first exports wrote it.
fn v1_bytes(s: &Skin) -> Vec<u8> {
    let mut o = Vec::new();
    o.extend_from_slice(MAGIC);
    o.extend_from_slice(&1u32.to_le_bytes());
    let st = |o: &mut Vec<u8>, t: &str| {
        o.extend_from_slice(&(t.len() as u16).to_le_bytes());
        o.extend_from_slice(t.as_bytes());
    };
    o.extend_from_slice(&(s.nodes.len() as u32).to_le_bytes());
    for n in &s.nodes {
        st(&mut o, n);
    }
    o.extend_from_slice(&(s.meshes.len() as u32).to_le_bytes());
    for m in &s.meshes {
        st(&mut o, &m.name);
        let ne: usize = m.verts.iter().map(|v| v.w.len()).sum();
        o.extend_from_slice(&(m.verts.len() as u32).to_le_bytes());
        o.extend_from_slice(&(ne as u32).to_le_bytes());
        let mut at = 0u32;
        for v in &m.verts {
            o.extend_from_slice(&at.to_le_bytes());
            for f in v.off.iter().chain(&v.nrm) {
                o.extend_from_slice(&f.to_le_bytes());
            }
            at += v.w.len() as u32;
        }
        o.extend_from_slice(&at.to_le_bytes());
        for v in &m.verts {
            for &(n, a, b, c) in &v.w {
                o.extend_from_slice(&n.to_le_bytes());
                for f in [a, b, c] {
                    o.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
    }
    o
}

/// An entry as the sidecar's steps: exact (back to the same f32 bits) or none.
#[test]
fn steps_are_exact_or_none() {
    let s = two_meshes();
    let d = Skin::decode(&s.encode()).unwrap();
    for v in d.meshes.iter().flat_map(|m| &m.verts) {
        for &(n, a, b, c) in &v.w {
            let (n16, q) = steps(n, [a, b, c]).expect("a decoded entry is on its grids");
            assert_eq!(n16 as u32, n);
            let back = [
                (q[0] as f64 * Q_A) as f32,
                (q[1] as f64 * Q_D) as f32,
                (q[2] as f64 * Q_D) as f32,
            ];
            assert_eq!(back.map(f32::to_bits), [a, b, c].map(f32::to_bits));
        }
    }
    assert!(steps(3, [0.1, 0.0, 0.0]).is_none(), "off the A grid");
    assert!(
        steps(3, [0.0, 0.001, 0.0]).is_none(),
        "off the gradient grid"
    );
    assert!(steps(3, [4.0, 0.0, 0.0]).is_none(), "past the i16 range");
    assert!(steps(65_536, [0.5, 0.0, 0.0]).is_none(), "a node past u16");
    assert!(steps(3, [f32::NAN, 0.0, 0.0]).is_none());
    assert_eq!(
        steps(65_535, [-4.0, -64.0, 0.0]),
        Some((65_535, [-32_768, -32_768, 0]))
    );
    // -0.0 is not 0 steps' bits.
    assert!(steps(1, [-0.0, 0.0, 0.0]).is_none());
}

/// Read in place, a sidecar gives the same bits as decoded, in both versions,
/// and fails where decode fails with decode's message.
#[test]
fn read_in_place_is_decode() {
    let s = two_meshes();
    for b in [s.encode(), v1_bytes(&s)] {
        let d = Skin::decode(&b).unwrap();
        let r = SkinBytes::parse(&b).unwrap();
        assert_eq!(r.nodes, d.nodes);
        assert_eq!(r.meshes.len(), d.meshes.len());
        for (m, dm) in r.meshes.iter().zip(&d.meshes) {
            assert_eq!(m.name, dm.name);
            assert_eq!(m.len(), dm.verts.len());
            assert_eq!(
                m.entries(),
                dm.verts.iter().map(|v| v.w.len()).sum::<usize>()
            );
            let got: Vec<Vert> = m.verts().collect();
            assert_eq!(format!("{got:?}"), format!("{:?}", dm.verts));
        }
        for cut in (0..b.len()).step_by(7) {
            assert_eq!(
                SkinBytes::parse(&b[..cut]).err(),
                Skin::decode(&b[..cut]).err(),
                "cut at {cut}"
            );
        }
    }
    // A node index past the names: the same error.
    let mut bad = v1_bytes(&s);
    let n = bad.len();
    bad[n - 16..n - 12].copy_from_slice(&999u32.to_le_bytes());
    assert_eq!(SkinBytes::parse(&bad).err(), Skin::decode(&bad).err());
    assert!(SkinBytes::parse(&bad).is_err());
    // Either reading, through SkinRef, the same meshes.
    let b = s.encode();
    let (d, r) = (Skin::decode(&b).unwrap(), SkinBytes::parse(&b).unwrap());
    for name in ["panel", "door"] {
        let a: Vec<Vert> = SkinRef::Decoded(&d).mesh(name).unwrap().verts().collect();
        let c: Vec<Vert> = SkinRef::Bytes(&r).mesh(name).unwrap().verts().collect();
        assert_eq!(format!("{a:?}"), format!("{c:?}"));
    }
    assert!(SkinRef::Bytes(&r).mesh("hood").is_none());
    assert_eq!(SkinRef::Bytes(&r).nodes(), &d.nodes[..]);
}

/// A windscreen's frame as the generated 4x4 has it (JBeam frame, m): a sparse roof
/// (a header of three nodes, the rows behind, the A-pillars' middles) and a dense
/// cowl and bonnet rear 0.6 m below the header; no node in the opening.
fn windscreen_frame() -> (Vec<u32>, Vec<Vec3>) {
    // The roof: sparse (its kernels reach far).
    let mut p = vec![Vec3::new(0.0, 1.087, 1.83)];
    for y in [1.4f32, 1.8, 2.18] {
        p.push(Vec3::new(0.0, y, 1.846));
    }
    for s in [-1.0f32, 1.0] {
        p.push(Vec3::new(0.763 * s, 1.05, 1.79));
        p.push(Vec3::new(0.767 * s, 0.84, 1.495));
        for y in [1.4f32, 1.8, 2.18] {
            p.push(Vec3::new(0.763 * s, y, 1.81));
        }
    }
    // The cowl and the bonnet's rear: dense (their kernels are short).
    for k in 0..=10 {
        let x = -0.75 + 0.15 * k as f32;
        p.push(Vec3::new(x, 0.57, 1.18));
        p.push(Vec3::new(x, 0.45, 1.10));
        p.push(Vec3::new(x, 0.30, 1.05));
    }
    ((0..p.len() as u32).collect(), p)
}

/// The pane in the opening: from the header line down to the cowl line.
fn windscreen() -> (Vec<Vec3>, Vec<Vec3>) {
    let (top, low) = (Vec3::new(0.0, 0.95, 1.714), Vec3::new(0.0, 0.62, 1.22));
    let n = (low - top).cross(Vec3::X).normalize();
    let mut v = Vec::new();
    for i in 0..=12 {
        for j in 0..=8 {
            let x = -0.6 + 1.2 * i as f32 / 12.0;
            v.push(top.lerp(low, j as f32 / 8.0) + Vec3::X * x);
        }
    }
    let nn = vec![n; v.len()];
    (v, nn)
}

#[test]
fn a_pane_across_an_opening_interpolates_its_frame() {
    let (ids, p) = windscreen_frame();
    let (v, n) = windscreen();
    let amp = |sk: &[Vert]| sk.iter().map(amplification).fold(0.0, f64::max);
    let (free, _) = bind(&ids, &p, &v, &n).unwrap();
    let (held, _) = bind_with(&ids, &p, &v, &n, Some(3.0)).unwrap();
    println!(
        "amplification: free {:.2}, held {:.2}",
        amp(&free),
        amp(&held)
    );
    assert!(amp(&held) <= 3.0);
    // The header's middle pushed 0.3 m down (a roof crush). Unbounded, the
    // pane's top extrapolates from the roof alone (a weight of about −2 on the
    // header) and rises the more the header falls: the generated 4x4's spike
    // (0.89 m here). Bounded, the cowl is in reach: it rises 3 cm at most.
    let mut q = p.clone();
    q[0].z -= 0.3;
    let rise = |sk: &[Vert]| {
        sk.iter()
            .zip(&v)
            .map(|(s, x)| eval(&s.w, s.off, s.nrm, &q).0.z - x.z)
            .fold(f32::MIN, f32::max)
    };
    println!(
        "highest rise: free {:.3} m, held {:.3} m",
        rise(&free),
        rise(&held)
    );
    assert!(
        rise(&free) > 0.3,
        "the case no longer extrapolates: {}",
        rise(&free)
    );
    assert!(
        rise(&held) < 0.05,
        "a pane vertex rose {} m as its header went down",
        rise(&held)
    );
}

/// A panel's collision skin: one layer of nodes, 4 mm out of plane at most (a
/// gently curved door skin), and paint up to 4 cm above it (the paint's bulge
/// between the skin's nodes).
fn sheet() -> (Vec<u32>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
    let mut p = Vec::new();
    for i in 0..5 {
        for j in 0..3 {
            p.push(Vec3::new(i as f32 * 0.25, j as f32 * 0.3, 0.004 * (i as f32 * 0.9).sin()));
        }
    }
    let v: Vec<Vec3> = (0..200)
        .map(|k| {
            let f = k as f32;
            Vec3::new(0.05 + 0.9 * (f * 0.37).fract(), 0.05 + 0.5 * (f * 0.61).fract(), 0.04 * (f * 0.13).fract())
        })
        .collect();
    let n = vec![Vec3::Z; v.len()];
    ((0..p.len() as u32).collect(), p, v, n)
}

#[test]
fn a_sheet_fit_follows_its_surface_and_keeps_the_height_in_the_offset() {
    let (ids, p, v, n) = sheet();
    let amp = |sk: &[Vert]| sk.iter().map(amplification).fold(0.0, f64::max);
    // The volume fit cannot use one layer: it grows its kernels until the 4 mm of
    // curvature gives it a third direction, then reaches the paint's height with
    // weights of both signs.
    let free = bind(&ids, &p, &v, &n);
    let (sk, _) = bind_sheet(&ids, &p, &v, &n, None).unwrap();
    println!(
        "amplification: volume fit {}, sheet fit {:.2}",
        free.as_ref().map_or("no fit".to_string(), |f| format!("{:.2}", amp(&f.0))),
        amp(&sk)
    );
    assert!(amp(&sk) < 1.6, "the sheet fit extrapolates: {}", amp(&sk));
    if let Ok((f, _)) = &free {
        assert!(amp(f) > amp(&sk), "the case no longer needs the sheet fit");
    }
    // Rest and any rigid motion exact.
    let q = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3);
    let t = Vec3::new(3.0, -1.0, 2.0);
    let moved: Vec<Vec3> = p.iter().map(|x| q * *x + t).collect();
    for ((s, x), nn) in sk.iter().zip(&v).zip(&n) {
        let (a, an) = eval(&s.w, s.off, s.nrm, &p);
        assert!(a.distance(*x) < 2e-5 && an.angle_between(*nn) < 1e-3, "rest");
        let (b, bn) = eval(&s.w, s.off, s.nrm, &moved);
        assert!(b.distance(q * *x + t) < 2e-5, "rigid {}", b.distance(q * *x + t));
        assert!(bn.angle_between(q * *nn) < 1e-3);
    }
    // One node pushed 5 cm into the panel (a dent): no vertex moves more than the
    // node, and none the other way.
    let mut d = p.clone();
    d[7].z -= 0.05;
    for (s, x) in sk.iter().zip(&v) {
        let dz = eval(&s.w, s.off, s.nrm, &d).0.z - x.z;
        assert!(dz <= 1e-3 && dz >= -0.05 * 1.05, "a vertex moved {dz} m for the node's -0.05");
    }
}
