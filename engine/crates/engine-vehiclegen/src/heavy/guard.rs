//! A semi-trailer's rear impact (underride) guard: a horizontal bar of square tube on
//! two supports (a post and a diagonal brace each) bolted to the frame's rails.
//!
//! **The regulation it is built to** (US FMVSS No. 223 "Rear impact guards", 49 CFR
//! 571.223, as amended in 2022 to the Canadian CMVSS No. 223's levels, and FMVSS No. 224
//! for where it sits; `generated-vehicles-lessons.md` §24):
//!
//! * placement (224 S5.1): the bar's lower edge at most 560 mm over the ground across its
//!   whole width, its outer ends within 100 mm of the trailer's sides (and not past them),
//!   its rear face at most 305 mm ahead of the trailer's rear extremity;
//! * section (223 S5.1): a horizontal member at least 100 mm high everywhere;
//! * strength (223 S5.2, S6.4–S6.8, a forward quasi-static push at 2–9 cm/min): at P1
//!   (3/8 of the guard's width from the centre plane, either side, 50 mm above its bottom)
//!   and at P2 (on the centre plane, 50 mm above its bottom) it resists 50,000 N with at
//!   most 125 mm of deflection; a uniform distributed load (a rigid block 203 mm high,
//!   wider than the outermost supports, centred, 50 mm above the bottom) of at least
//!   350,000 N, absorbing at least 20,000 J by plastic deformation within the first 125
//!   mm (or more than 700,000 N without the energy requirement); in every test no
//!   pre-existing load path is eliminated (nothing breaks off).
//!
//! **How the strengths follow from it.** Nothing here is a free number: the guard is the
//! *minimally compliant* one, every member yielding at the regulation's loads.
//!
//! * The bar is the lightest standard 100 mm square tube that carries P1: P1's 50 kN at
//!   its lever arm outboard of the support (`p1_x − support_x`) is a bending moment the
//!   tube's plastic moment `M_p = σ_y·Z_pl` must reach ([`RearGuard::wall`]). The model's
//!   bar is four corner chords `d` apart (the section less the nodes' inset), so a chord
//!   yields at `M_p / (2·d)` (two chords on each side of the bending axis): the bar yields
//!   in bending at the tube's plastic moment. Each face's diagonal yields at the face's
//!   plastic shear `t·b·σ_y/√3` over its angle.
//! * The supports carry the uniform load (350 kN) at the bar's centre height, forward.
//!   Each is a post (the bar's top nodes to the rails' bottom corners at their last
//!   station) and a brace (its front nodes to the corners a station ahead), four beams
//!   each. Their forces at that load come from one statically admissible solution, the
//!   bar a rigid body on beams of equal EA ([`support_forces`]): a member's beams all
//!   yield at its largest share (a break group goes with its weakest beam), so by the
//!   lower-bound theorem the guard collapses at no less than 350 kN. (P1 and P2 load a
//!   support with less: 50 kN × (`p1_x` + `support_x`) / (2·`support_x`) on the nearer.)
//! * The attachments (a support's bolts and its member are one beam: it yields as the
//!   member and breaks as the bolts) are capacity-designed: they hold what the member
//!   carries at its steel's ultimate strength, `σ_u / σ_y` × its yield (engineering
//!   practice for connections, an estimate of a real guard's margin: none is regulated
//!   past "nothing breaks off" in the tests). The solver's yield hardens with the plastic
//!   strain (a beam's threshold grows with its length ratio), so a brace breaks once
//!   shortened by about `1 − σ_y/σ_u` of its length (19 %: measured 17-19 %); in
//!   tension a beam's strength falls as it stretches plastically (the solver's rule), and
//!   the posts broke at 33-37 % stretch.
//!
//! Every value is checked by `engine-softbody-authoring/tests/box_trailer.rs`: a rigid
//! block pushed slowly into P1, P2 and the uniform-load block, force against deflection.

use super::*;

/// A rear impact guard (FMVSS 223/224), in the vehicle's frame (+z forward).
#[derive(Clone, Debug)]
pub struct RearGuard {
    /// The bar's rear face (m).
    pub z_rear: f64,
    /// Its lower edge over the ground (m; at most 0.56).
    pub lower_y: f64,
    /// Half its width (m; its ends within 0.1 m of the vehicle's sides).
    pub half_w: f64,
    /// The tube's outer side and wall (m).
    pub section: f64,
    pub wall: f64,
    /// The steel's yield and ultimate strength (Pa).
    pub yield_pa: f64,
    pub ultimate_pa: f64,
    /// The two supports' x (±, m) and P1's (3/8 of the width from the centre).
    pub support_x: f64,
    /// The rail stations the posts and the braces bolt to (index into
    /// [`LadderFrame::stations`]).
    pub post_station: usize,
    pub brace_station: usize,
    /// The regulation's uniform distributed load (N): each support collapses at half.
    pub udl_n: f64,
    pub kg: f64,
}

/// The nodes' inset from the tube's faces (m).
const INSET: f64 = 0.005;

impl RearGuard {
    /// P1's x: 3/8 of the guard's width from the centre plane.
    pub fn p1_x(&self) -> f64 {
        0.375 * 2.0 * self.half_w
    }
    /// The test points' height: 50 mm above the guard's bottom.
    pub fn test_y(&self) -> f64 {
        self.lower_y + 0.05
    }
    /// A square tube's plastic section modulus (m³) about an axis parallel to a side.
    pub fn z_pl(&self) -> f64 {
        let (b, i) = (self.section, self.section - 2.0 * self.wall);
        (b * b * b - i * i * i) / 4.0
    }
    /// Its plastic moment (N m).
    pub fn m_p(&self) -> f64 {
        self.yield_pa * self.z_pl()
    }
    /// The chords' spacing (m).
    pub fn d(&self) -> f64 {
        self.section - 2.0 * INSET
    }
    /// The bar's stations along x: its ends, P1, the supports, the centre.
    pub fn xs(&self) -> [f64; 7] {
        let (w, p, s) = (self.half_w - INSET, self.p1_x(), self.support_x);
        [-w, -p, -s, 0.0, s, p, w]
    }
    /// σ_u / σ_y: the attachments' break over their yield.
    pub fn capacity(&self) -> f64 {
        self.ultimate_pa / self.yield_pa
    }
}

/// A guard bar node's name: station `i`, `j` 0 bottom / 1 top, `k` 0 rear / 1 front.
pub fn guard_node(i: usize, j: usize, k: usize) -> String {
    format!("gb{i}{j}{k}")
}

/// `(guard_bar, its meshes)`: the bar (class `guard`, zone `Rear`) with its supports as
/// its mounts (`guard_post_{L,R}`, `guard_brace_{L,R}`) to `f`'s rails.
pub fn rear_guard(g: &RearGuard, f: &LadderFrame) -> (MechPart, Vec<(String, Mesh)>) {
    let mut p = part("guard_bar", "guard", Zone::Rear);
    let xs = g.xs();
    let n = xs.len() * 4;
    for (i, &x) in xs.iter().enumerate() {
        for j in 0..2 {
            for k in 0..2 {
                let y = g.lower_y + INSET + j as f64 * g.d();
                let z = g.z_rear + INSET + k as f64 * g.d();
                add_node(&mut p, guard_node(i, j, k), DVec3::new(x, y, z), g.kg / n as f64, true);
            }
        }
    }
    let id = |i: usize, j: usize, k: usize| 4 * i + 2 * j + k;
    let cap = g.capacity();
    let chord_y = g.m_p() / (2.0 * g.d());
    let (mut chords, mut ring) = (Vec::new(), Vec::new());
    for i in 0..xs.len() {
        // the section's own ring (90 mm: under the solver's 0.1 m, it keeps its shape)
        ring.extend([
            (id(i, 0, 0), id(i, 0, 1)),
            (id(i, 1, 0), id(i, 1, 1)),
            (id(i, 0, 0), id(i, 1, 0)),
            (id(i, 0, 1), id(i, 1, 1)),
            (id(i, 0, 0), id(i, 1, 1)),
            (id(i, 0, 1), id(i, 1, 0)),
        ]);
        if i + 1 < xs.len() {
            for (j, k) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                chords.push((id(i, j, k), id(i + 1, j, k)));
            }
        }
    }
    p.internal.push(Internal {
        pairs: chords,
        v: vals(0.0, 0.0, chord_y, cap * chord_y),
        deform_group: None,
    });
    p.internal.push(Internal {
        pairs: ring,
        v: vals(0.0, 0.0, INF, cap * chord_y),
        deform_group: None,
    });
    // each face's diagonal, zig-zag along the bar, at the face's plastic shear
    let v_face = g.wall * g.section * g.yield_pa / 3f64.sqrt();
    for i in 0..xs.len() - 1 {
        let s = xs[i + 1] - xs[i];
        let sin = g.d() / (g.d() * g.d() + s * s).sqrt();
        let dy = v_face / sin;
        let faces = [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
        ];
        let pairs = faces
            .iter()
            .map(|&((ja, ka), (jb, kb))| {
                if i % 2 == 0 {
                    (id(i, ja, ka), id(i + 1, jb, kb))
                } else {
                    (id(i, jb, kb), id(i + 1, ja, ka))
                }
            })
            .collect();
        p.internal.push(Internal {
            pairs,
            v: vals(0.0, 0.0, dy, cap * dy),
            deform_group: None,
        });
    }
    supports(&mut p, g, f);
    p.mesh = box_(
        DVec3::new(0.0, g.lower_y + 0.5 * g.section, g.z_rear + 0.5 * g.section),
        DVec3::new(g.half_w, 0.5 * g.section, 0.5 * g.section),
        Mat::TrimBlack,
        None,
    );
    let extra = guard_meshes(g, f);
    p.meshes = extra.iter().map(|e| e.0.clone()).collect();
    (p, extra)
}

/// One support beam: (break group, bar node, rail node, the bar node's and the rail
/// node's positions).
pub type SupportBeam = (String, String, String, DVec3, DVec3);

/// A support's beams on side `s`: the post (the bar's two top nodes at the support, each
/// to the rail's inner and outer bottom corners at the post station) and the brace (its
/// two front nodes to the same corners at the brace station).
pub fn support_beams(g: &RearGuard, f: &LadderFrame, s: f64) -> Vec<SupportBeam> {
    let st = f.stations();
    let i = if s > 0.0 { 4 } else { 2 };
    let x = s * g.support_x;
    let bar = |j: usize, k: usize| DVec3::new(x, g.lower_y + INSET + j as f64 * g.d(), g.z_rear + INSET + k as f64 * g.d());
    let rail = |k: usize, c: &str| {
        let (x, y) = f.corner(s, c);
        DVec3::new(x, y, st[k])
    };
    let sd = side_tag(s);
    let mut out = Vec::new();
    for (member, k, ends) in [("post", g.post_station, [(1, 0), (1, 1)]), ("brace", g.brace_station, [(0, 1), (1, 1)])] {
        for (j, kk) in ends {
            for c in ["ib", "ob"] {
                out.push((format!("guard_{member}_{sd}"), guard_node(i, j, kk), rail_node(s, k, c), bar(j, kk), rail(k, c)));
            }
        }
    }
    out
}

/// Each support beam's axial force (N, + tension), both supports' beams (the right's,
/// then the left's), when the guard carries the uniform load, `udl_n` forward on the
/// centre plane at the bar's centre height, with the bar a rigid body on its supports'
/// beams of equal axial stiffness EA (k = EA/L): one statically admissible distribution
/// of the load, so a guard whose every support beam yields at its share collapses at no
/// less than that load (the lower-bound theorem of plastic collapse).
pub fn support_forces(g: &RearGuard, f: &LadderFrame) -> Vec<f64> {
    let beams: Vec<SupportBeam> = [-1.0, 1.0].iter().flat_map(|&s| support_beams(g, f, s)).collect();
    let c = DVec3::new(0.0, g.lower_y + 0.5 * g.section, g.z_rear + 0.5 * g.section);
    let rows: Vec<([f64; 6], f64)> = beams
        .iter()
        .map(|b| {
            let (p, q) = (b.3, b.4);
            let n = (p - q).normalize();
            let r = (p - c).cross(n);
            ([n.x, n.y, n.z, r.x, r.y, r.z], 1.0 / (p - q).length())
        })
        .collect();
    let mut k = [[0.0f64; 6]; 6];
    for (a, w) in &rows {
        for i in 0..6 {
            for j in 0..6 {
                k[i][j] += w * a[i] * a[j];
            }
        }
    }
    let load = [0.0, 0.0, g.udl_n, 0.0, 0.0, 0.0];
    let x = solve6(k, load);
    // (a beam's extension n·δp, n from the rail node to the bar's, times its stiffness:
    // tension when the bar moves away from the rail node)
    rows.iter().map(|(a, w)| w * (0..6).map(|i| a[i] * x[i]).sum::<f64>()).collect()
}

/// `k·x = b` for a symmetric positive (semi)definite 6×6 `k` (Gaussian elimination with
/// partial pivoting; a direction no beam restrains, a near-zero pivot, takes no
/// displacement).
fn solve6(mut k: [[f64; 6]; 6], mut b: [f64; 6]) -> [f64; 6] {
    let scale = (0..6).map(|i| k[i][i].abs()).fold(0.0, f64::max).max(1e-30);
    let mut free = [false; 6];
    for col in 0..6 {
        let piv = (col..6).max_by(|&a, &c| k[a][col].abs().total_cmp(&k[c][col].abs())).unwrap();
        k.swap(col, piv);
        b.swap(col, piv);
        if k[col][col].abs() < 1e-9 * scale {
            free[col] = true;
            continue;
        }
        for r in col + 1..6 {
            let m = k[r][col] / k[col][col];
            for c in col..6 {
                k[r][c] -= m * k[col][c];
            }
            b[r] -= m * b[col];
        }
    }
    let mut x = [0.0; 6];
    for r in (0..6).rev() {
        if free[r] {
            continue;
        }
        let s: f64 = (r + 1..6).map(|c| k[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / k[r][r];
    }
    x
}

/// The supports as the bar's mounts, one beam each ([`support_beams`]), each in its
/// member's break group (the post's, the brace's). A break group breaks when any of its
/// beams does (lessons §16), so a member's beams all take its largest share of the
/// support's half of the uniform load ([`support_forces`]): each yields at no less than
/// its own share, and breaks at σ_u/σ_y of that.
fn supports(p: &mut MechPart, g: &RearGuard, f: &LadderFrame) {
    let cap = g.capacity();
    let forces = support_forces(g, f);
    let beams: Vec<SupportBeam> = [-1.0, 1.0].iter().flat_map(|&s| support_beams(g, f, s)).collect();
    for b in &beams {
        let y = beams.iter().zip(&forces).filter(|(o, _)| o.0 == b.0).map(|(_, x)| x.abs()).fold(0.0, f64::max);
        mount(p, &b.0, vec![b.1.clone()], vec![b.2.clone()], 1, vals(0.0, 0.0, y, cap * y));
    }
}

/// The posts and braces (riding the bar), and the conspicuity tape on its face.
fn guard_meshes(g: &RearGuard, f: &LadderFrame) -> Vec<(String, Mesh)> {
    let st = f.stations();
    let mut sup = Mesh::new();
    let yc = g.lower_y + 0.5 * g.section;
    for s in [1.0f64, -1.0] {
        let x = s * g.support_x;
        // the post: a 100 mm channel from the bar's top to the rail's bottom
        let (y0, y1) = (g.lower_y + g.section, f.bottom_y());
        let zp = g.z_rear + 0.5 * g.section;
        sup.add(&box_(
            DVec3::new(x, 0.5 * (y0 + y1), zp),
            DVec3::new(0.04, 0.5 * (y1 - y0), 0.04),
            Mat::TrimBlack,
            None,
        ));
        // the brace: from the bar's front face up to the rail at the brace station
        let a = DVec3::new(x, yc, g.z_rear + g.section);
        let b = DVec3::new(x, f.bottom_y() - 0.02, st[g.brace_station]);
        let dir = (b - a).normalize();
        // (a right-handed frame: x across, z along the brace, y = z × x)
        let r = glam::DMat3::from_cols(DVec3::X, dir.cross(DVec3::X), dir);
        sup.add(&box_(0.5 * (a + b), DVec3::new(0.03, 0.03, 0.5 * (b - a).length()), Mat::TrimBlack, Some(r)));
    }
    // alternating red and white retroreflective tape (FMVSS 108): two lens meshes, the
    // design's `lens_tint` colours them by name
    let (mut red, mut white) = (Mesh::new(), Mesh::new());
    let n = 8;
    for k in 0..n {
        let w = 2.0 * (g.half_w - 0.02) / n as f64;
        let x = -g.half_w + 0.02 + (k as f64 + 0.5) * w;
        let m = box_(
            DVec3::new(x, g.lower_y + 0.5 * g.section, g.z_rear - 0.002),
            DVec3::new(0.5 * w - 0.003, 0.025, 0.002),
            Mat::LampLens,
            None,
        );
        if k % 2 == 0 { red.add(&m) } else { white.add(&m) };
    }
    vec![("guard_supports".into(), sup), ("guard_tape_red".into(), red), ("guard_tape_white".into(), white)]
}
