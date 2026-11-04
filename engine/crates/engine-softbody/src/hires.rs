//! An original high-resolution test car at BeamNG's resolution (about 1,000
//! nodes and 5,500 beams), generated as JBeam text so it goes through the real
//! importer with BeamNG's node groups, collision flags and break groups.
//! Licensed with Rubble: no third-party content.
//!
//! The body is a closed shell of 16-node rings, 12 cm apart, every quad
//! cross-braced, with bend beams over two rings and two perimeter steps, and
//! bulkheads at both ends and at the firewall. Inside the front bay run two
//! box-section crash rails, graded to fold front first, carrying a braced engine
//! block, a gearbox behind it and a radiator ahead of it. The bonnet, boot lid
//! and four doors are their own node sheets a few centimetres outside the
//! shell, on mounts in one break group per panel, resting on compression-only
//! supports. The passenger cell yields at several times the crumple zones'
//! threshold, so a crash folds the ends and keeps the cabin.
//!
//! Every node carries a group (`hr_body`, `hr_bonnet`, `hr_door_fl`, …,
//! `hr_engine`): BeamNG's self-collision skips a triangle whose nodes share the
//! node's group, which is how a panel slides over the shell under it while a
//! loose door still meets the fender beside it.

use crate::desc::SoftVehicleDesc;
use glam::Vec3;
use std::fmt::Write;

/// Tunables of the high-resolution car.
#[derive(Clone, Debug)]
pub struct HiresParams {
    pub name: String,
    pub length: f32,
    pub width: f32,
    /// Rings along the length.
    pub stations: usize,
    /// Stiffness of a 0.15 m beam (N/m); a beam of length L gets k·0.15/L, so
    /// shorter beams are stiffer as in a continuum.
    pub spring: f32,
    pub damp: f32,
    /// Crumple-zone yield threshold and break strength (N).
    pub deform: f32,
    pub strength: f32,
    /// Passenger cell over the crumple zones.
    pub cell_deform_scale: f32,
    pub cell_strength_scale: f32,
    /// Crash rails' yield threshold, graded 0.5× at the bumper to 1.5× at the
    /// firewall.
    pub rail_deform: f32,
    /// Shell node weight and the cabin's interior load spread over its floor (kg).
    pub node_weight: f32,
    pub interior_mass: f32,
    pub engine_mass: f32,
    pub gearbox_mass: f32,
    /// Panel mounts' break strength over the body's.
    pub mount_strength: f32,
    pub wheel_radius: f32,
    /// Beams shorter than this get this length's stiffness (`spring`·0.15/L is
    /// capped here). 0.03 m by default; a fixed explicit step needs longer (about
    /// 0.12 m with 2 kg nodes at 0.5 ms), or the short panel mounts outrun it.
    pub stiffness_length: f32,
}

impl Default for HiresParams {
    fn default() -> Self {
        Self {
            name: "Rubble Saloon HR (procedural high-resolution test car)".into(),
            length: 4.4,
            width: 1.76,
            stations: 37,
            spring: 3.0e6,
            damp: 250.0,
            deform: 6_000.0,
            strength: 150_000.0,
            cell_deform_scale: 5.0,
            cell_strength_scale: 3.0,
            rail_deform: 40_000.0,
            node_weight: 1.0,
            interior_mass: 250.0,
            engine_mass: 160.0,
            gearbox_mass: 60.0,
            mount_strength: 0.3,
            wheel_radius: 0.32,
            stiffness_length: 0.03,
        }
    }
}

/// Nodes per ring.
pub const HR_RING: usize = 16;

/// Where the car's parts are, by ring.
#[derive(Clone, Debug, Default)]
pub struct HiresLayout {
    pub stations: usize,
    pub cabin: Vec<bool>,
    /// First and last cabin ring; the firewall is the last.
    pub cabin_first: usize,
    pub cabin_last: usize,
}

/// One node as generated (Rubble frame).
struct Node {
    id: String,
    p: Vec3,
    mass: f32,
    group: &'static str,
}

#[derive(Clone, Copy, PartialEq)]
struct BeamProps {
    spring: f32,
    damp: f32,
    deform: f32,
    strength: f32,
    kind: &'static str,
    group: &'static str,
}

struct Builder {
    nodes: Vec<Node>,
    beams: Vec<(usize, usize, BeamProps)>,
    tris: Vec<[usize; 3]>,
    /// Interior points the skin faces away from (per triangle: the reference).
    p: HiresParams,
}

impl Builder {
    fn node(&mut self, id: String, p: Vec3, mass: f32, group: &'static str) -> usize {
        self.nodes.push(Node { id, p, mass, group });
        self.nodes.len() - 1
    }

    /// A beam with stiffness scaled to its length.
    fn beam(&mut self, a: usize, b: usize, deform: f32, strength: f32, group: &'static str) {
        self.beam_k(a, b, 1.0, deform, strength, group);
    }

    fn beam_k(
        &mut self,
        a: usize,
        b: usize,
        kscale: f32,
        deform: f32,
        strength: f32,
        group: &'static str,
    ) {
        // Skip a pair already joined (perimeter skips and bulkhead braces overlap).
        if a == b
            || self.beams.iter().any(|&(p, q, pr)| {
                pr.kind == "|NORMAL" && (p.min(q), p.max(q)) == (a.min(b), a.max(b))
            })
        {
            return;
        }
        let l = (self.nodes[a].p - self.nodes[b].p).length().max(self.p.stiffness_length);
        // Two significant digits, so modifier rows repeat.
        let k = round2(self.p.spring * kscale * 0.15 / l);
        self.beams.push((
            a,
            b,
            BeamProps {
                spring: k,
                damp: self.p.damp,
                deform,
                strength,
                kind: "|NORMAL",
                group,
            },
        ));
    }

    fn support(&mut self, a: usize, b: usize) {
        let l = (self.nodes[a].p - self.nodes[b].p).length().max(self.p.stiffness_length);
        self.beams.push((
            a,
            b,
            BeamProps {
                spring: round2(self.p.spring * 0.3 * 0.15 / l),
                damp: self.p.damp,
                deform: f32::INFINITY,
                strength: f32::INFINITY,
                kind: "|SUPPORT",
                group: "",
            },
        ));
    }

    /// Triangle facing away from `inside`.
    fn tri(&mut self, a: usize, b: usize, c: usize, inside: Vec3) {
        let (pa, pb, pc) = (self.nodes[a].p, self.nodes[b].p, self.nodes[c].p);
        let n = (pb - pa).cross(pc - pa);
        let centre = (pa + pb + pc) / 3.0;
        if n.dot(centre - inside) >= 0.0 {
            self.tris.push([a, b, c]);
        } else {
            self.tris.push([a, c, b]);
        }
    }

    /// Two triangles over quad a b c d (in order around it).
    fn quad(&mut self, a: usize, b: usize, c: usize, d: usize, inside: Vec3) {
        self.tri(a, b, c, inside);
        self.tri(a, c, d, inside);
    }

    /// Every pair of `nodes` joined (a braced block).
    fn brace_all(&mut self, nodes: &[usize], kscale: f32, deform: f32, strength: f32) {
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                self.beam_k(nodes[i], nodes[j], kscale, deform, strength, "");
            }
        }
    }

    /// The `count` nodes of `pool` nearest to `p`.
    fn nearest(&self, p: Vec3, pool: &[usize], count: usize) -> Vec<usize> {
        let mut v: Vec<usize> = pool.to_vec();
        v.sort_by(|&a, &b| {
            (self.nodes[a].p - p)
                .length_squared()
                .total_cmp(&(self.nodes[b].p - p).length_squared())
        });
        v.truncate(count);
        v
    }
}

fn round2(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let e = 10f32.powi(x.abs().log10().floor() as i32 - 1);
    (x / e).round() * e
}

/// Roof (or bonnet / boot) height at t ∈ [0, 1] from rear to front.
fn top(t: f32) -> f32 {
    let pts = [
        (0.0, 0.95),
        (0.1, 1.0),
        (0.26, 1.4),
        (0.56, 1.44),
        (0.72, 1.02),
        (0.86, 0.92),
        (1.0, 0.86),
    ];
    for w in pts.windows(2) {
        let ((t0, y0), (t1, y1)) = (w[0], w[1]);
        if t <= t1 {
            let s = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let s = s * s * (3.0 - 2.0 * s);
            return y0 + (y1 - y0) * s;
        }
    }
    0.86
}

/// The car as JBeam text (one part, `rubble_saloon_hr`) and its layout.
pub fn hires_car_jbeam(p: &HiresParams) -> (String, HiresLayout) {
    let mut b = Builder {
        nodes: Vec::new(),
        beams: Vec::new(),
        tris: Vec::new(),
        p: p.clone(),
    };
    let n = p.stations.max(8);
    let hw = 0.5 * p.width;
    let floor = 0.24;
    let z_of = |k: usize| -0.5 * p.length + k as f32 / (n - 1) as f32 * p.length;
    let cabin: Vec<bool> = (0..n)
        .map(|k| top(k as f32 / (n - 1) as f32) > 1.25)
        .collect();
    let cabin_first = cabin.iter().position(|&c| c).unwrap();
    let cabin_last = cabin.iter().rposition(|&c| c).unwrap();
    let zone = |k: usize| {
        if k >= cabin_first && k <= cabin_last {
            (
                p.deform * p.cell_deform_scale,
                p.strength * p.cell_strength_scale,
            )
        } else {
            (p.deform, p.strength)
        }
    };
    // Shell rings. Half section (x ≥ 0): floor centre, floor, sill, lower side,
    // belt, upper side, roof edge, roof, roof centre; the ring runs p0, the left
    // half, p8, the right half back.
    let mut ring = vec![[0usize; HR_RING]; n];
    for k in 0..n {
        let t = k as f32 / (n - 1) as f32;
        let z = z_of(k);
        let taper = 1.0 - 0.07 * (2.0 * t - 1.0).abs().powi(4);
        let w = hw * taper;
        let tp = top(t);
        let belt = (tp - 0.15).min(0.8);
        let rw = if cabin[k] { 0.78 * w } else { 0.9 * w };
        let half = [
            (0.0, floor),
            (0.5 * w, floor),
            (0.95 * w, floor + 0.03),
            (w, 0.5 * (floor + belt)),
            (w, belt),
            (w + 0.4 * (rw - w), belt + 0.4 * (tp - belt)),
            (rw, tp - 0.02),
            (0.5 * rw, tp),
            (0.0, tp + 0.01),
        ];
        let mass = |j: usize| {
            // The cabin floor carries the interior (seats, trim, occupants' share).
            let floor_row = matches!(j, 0 | 1 | 2);
            if floor_row && cabin[k] {
                p.node_weight + p.interior_mass / (5.0 * (cabin_last - cabin_first + 1) as f32)
            } else {
                p.node_weight
            }
        };
        for (j, slot) in ring[k].iter_mut().enumerate() {
            let (hx, y, side) = match j {
                0 => (half[0].0, half[0].1, 1.0),
                8 => (half[8].0, half[8].1, 1.0),
                1..=7 => (half[j].0, half[j].1, 1.0),
                _ => (half[16 - j].0, half[16 - j].1, -1.0),
            };
            *slot = b.node(
                format!("s{k}_{j}"),
                Vec3::new(side * hx, y, z),
                mass(j),
                "hr_body",
            );
        }
    }
    let axis = |b: &Builder, k: usize| {
        let c = ring[k].iter().map(|&i| b.nodes[i].p).sum::<Vec3>() / HR_RING as f32;
        Vec3::new(0.0, c.y, c.z)
    };
    for k in 0..n {
        let (d, s) = zone(k);
        // Glass band in the cabin's upper sides: weak except at the pillars.
        let pillar = k == cabin_first || k == cabin_last || k == (cabin_first + cabin_last) / 2;
        for j in 0..HR_RING {
            let jn = (j + 1) % HR_RING;
            b.beam(ring[k][j], ring[k][jn], d, s, "");
            b.beam(ring[k][j], ring[k][(j + 2) % HR_RING], d, s, "");
            if k + 1 < n {
                let (d1, s1) = zone(k + 1);
                let (dd, ss) = (d.min(d1), s.min(s1));
                let upper = cabin[k] && matches!(j, 5 | 6 | 10 | 11);
                let (dd, ss) = if upper && !pillar {
                    (dd * 0.3, ss * 0.3)
                } else {
                    (dd, ss)
                };
                b.beam(ring[k][j], ring[k + 1][j], dd, ss, "");
                b.beam(ring[k][j], ring[k + 1][jn], dd, ss, "");
                b.beam(ring[k][jn], ring[k + 1][j], dd, ss, "");
                let inside = 0.5 * (axis(&b, k) + axis(&b, k + 1));
                b.quad(
                    ring[k][j],
                    ring[k + 1][j],
                    ring[k + 1][jn],
                    ring[k][jn],
                    inside,
                );
            }
            if k + 2 < n {
                let (d2, s2) = zone(k + 2);
                b.beam(ring[k][j], ring[k + 2][j], d.min(d2), s.min(s2), "");
            }
        }
        // Floor cross member at every ring (1–15 is already a perimeter skip beam).
        b.beam(ring[k][2], ring[k][14], d, s, "");
    }
    // Bulkheads: the ends and the firewall get a centre node, spokes and a skin.
    let bulkhead = |b: &mut Builder, k: usize, name: &str, outward: f32| {
        let c = axis(b, k);
        let ci = b.node(name.to_string(), c, p.node_weight, "hr_body");
        let (d, s) = zone(k);
        for j in 0..HR_RING {
            b.beam(ci, ring[k][j], d, s, "");
            b.beam(ring[k][j], ring[k][(j + 4) % HR_RING], d, s, "");
            let inside = c - Vec3::new(0.0, 0.0, outward * 0.3);
            b.tri(ci, ring[k][j], ring[k][(j + 1) % HR_RING], inside);
        }
        ci
    };
    bulkhead(&mut b, n - 1, "front_c", 1.0);
    bulkhead(&mut b, 0, "rear_c", -1.0);
    bulkhead(&mut b, cabin_last, "firewall_c", 1.0);
    let shell: Vec<usize> = ring.iter().flatten().copied().collect();

    // Crash rails: a box section per side from the firewall to the front ring.
    let mut rails: Vec<[usize; 4]> = Vec::new(); // per (ring, side)
    let rail_rings: Vec<usize> = (cabin_last..n).collect();
    for (ri, &k) in rail_rings.iter().enumerate() {
        let z = z_of(k);
        let grade = 1.5 - (ri as f32 / (rail_rings.len() - 1).max(1) as f32);
        for side in [1.0f32, -1.0] {
            let mut c = [0usize; 4];
            for (ci, (x, y)) in [(0.36, 0.3), (0.36, 0.46), (0.5, 0.3), (0.5, 0.46)]
                .iter()
                .enumerate()
            {
                c[ci] = b.node(
                    format!("rl{}{}_{ci}", if side > 0.0 { "l" } else { "r" }, ri),
                    Vec3::new(side * x, *y, z),
                    p.node_weight,
                    "hr_body",
                );
            }
            let rd = p.rail_deform * grade;
            let rs = p.strength * p.cell_strength_scale;
            for (a, bb) in [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3), (1, 2)] {
                b.beam_k(c[a], c[bb], 2.0, rd, rs, "");
            }
            if ri > 0 {
                let prev = rails[rails.len() - 2];
                for q in 0..4 {
                    b.beam_k(prev[q], c[q], 2.0, rd, rs, "");
                    b.beam_k(prev[q], c[(q + 1) % 4], 2.0, rd, rs, "");
                }
            }
            // Tie the rail into the shell beside it.
            for &s in &b.nearest(Vec3::new(side * 0.43, 0.38, z), &ring[k], 3) {
                for &q in &c {
                    b.beam(q, s, rd, rs, "");
                }
            }
            rails.push(c);
        }
    }
    let rail_nodes: Vec<usize> = rails.iter().flatten().copied().collect();

    // Bumpers on break groups: a beam of nodes ahead of each end.
    let bumper = |b: &mut Builder, k: usize, dz: f32, name: &'static str, prefix: &str| {
        let z = z_of(k) + dz;
        let mut nodes = Vec::new();
        for (yi, y) in [0.38f32, 0.6].iter().enumerate() {
            for (xi, x) in [0.8f32, 0.4, 0.0, -0.4, -0.8].iter().enumerate() {
                let setback = 0.08 * (x.abs() / 0.8).powi(2) * dz.signum();
                nodes.push(b.node(
                    format!("{prefix}{yi}{xi}"),
                    Vec3::new(x * hw, *y, z - setback),
                    p.node_weight * 2.0,
                    name,
                ));
            }
        }
        b.brace_all(&nodes, 1.0, p.deform * 2.0, p.strength * 2.0);
        let anchor: Vec<usize> = ring[k]
            .iter()
            .copied()
            .chain(rail_nodes.iter().copied())
            .collect();
        for &q in &nodes {
            for s in b.nearest(b.nodes[q].p, &anchor, 2) {
                b.beam(q, s, p.deform * 2.0, p.strength * p.mount_strength, name);
            }
        }
        let c = nodes.iter().map(|&i| b.nodes[i].p).sum::<Vec3>() / 10.0 - Vec3::new(0.0, 0.0, dz);
        for xi in 0..4 {
            b.quad(
                nodes[xi],
                nodes[xi + 1],
                nodes[5 + xi + 1],
                nodes[5 + xi],
                c,
            );
        }
        nodes
    };
    bumper(&mut b, n - 1, 0.14, "hr_bumper_f", "bf");
    bumper(&mut b, 0, -0.12, "hr_bumper_r", "br");

    // Engine block, gearbox and radiator: braced node boxes on mounts.
    let bay_mid = 0.5 * (z_of(cabin_last) + z_of(n - 1));
    let block = |b: &mut Builder, name: &str, c: Vec3, h: Vec3, mass: f32, group: &'static str| {
        let mut v = Vec::new();
        for (i, &(sx, sy, sz)) in [
            (1.0, -1.0, -1.0),
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        .iter()
        .enumerate()
        {
            v.push(b.node(
                format!("{name}{i}"),
                c + h * Vec3::new(sx, sy, sz),
                mass / 8.0,
                group,
            ));
        }
        b.brace_all(&v, 10.0, f32::INFINITY, f32::INFINITY);
        // Collision skin: the six faces.
        for f in [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
        ] {
            b.quad(v[f[0]], v[f[1]], v[f[2]], v[f[3]], c);
        }
        v
    };
    let engine = block(
        &mut b,
        "eng",
        Vec3::new(0.0, 0.56, bay_mid + 0.05),
        Vec3::new(0.27, 0.17, 0.22),
        p.engine_mass,
        "hr_engine",
    );
    let gearbox = block(
        &mut b,
        "gbx",
        Vec3::new(0.0, 0.46, bay_mid - 0.36),
        Vec3::new(0.16, 0.11, 0.13),
        p.gearbox_mass,
        "hr_engine",
    );
    for &e in &engine[..4] {
        for g in b.nearest(b.nodes[e].p, &gearbox, 2) {
            b.beam_k(e, g, 5.0, f32::INFINITY, f32::INFINITY, "");
        }
    }
    for &e in engine.iter().chain(&gearbox) {
        for s in b.nearest(b.nodes[e].p, &rail_nodes, 2) {
            b.beam(e, s, p.rail_deform * 0.5, p.strength * 0.4, "");
        }
    }
    let rad_z = z_of(n - 1) - 0.1;
    let mut radiator = Vec::new();
    for (i, (x, y)) in [
        (0.34f32, 0.36f32),
        (-0.34, 0.36),
        (0.34, 0.66),
        (-0.34, 0.66),
    ]
    .iter()
    .enumerate()
    {
        radiator.push(b.node(
            format!("rad{i}"),
            Vec3::new(*x, *y, rad_z),
            3.0,
            "hr_engine",
        ));
    }
    b.brace_all(&radiator, 0.5, p.deform, p.strength);
    let front_anchor: Vec<usize> = ring[n - 1].iter().chain(&ring[n - 2]).copied().collect();
    for &r in &radiator {
        for s in b.nearest(b.nodes[r].p, &front_anchor, 2) {
            b.beam(r, s, p.deform, p.strength * 0.3, "");
        }
    }

    // Bolt-on panels: (group, rings, perimeter rows), offset 4 cm outward.
    let front_bay: Vec<usize> = (cabin_last + 1..n).collect();
    let rear_bay: Vec<usize> = (0..cabin_first).collect();
    let mid = (cabin_first + cabin_last) / 2;
    let panels: [(&'static str, Vec<usize>, Vec<usize>); 6] = [
        ("hr_bonnet", front_bay.clone(), vec![5, 6, 7, 8, 9, 10, 11]),
        ("hr_boot", rear_bay.clone(), vec![5, 6, 7, 8, 9, 10, 11]),
        (
            "hr_door_fl",
            (mid + 1..cabin_last).collect(),
            vec![2, 3, 4, 5],
        ),
        (
            "hr_door_rl",
            (cabin_first + 1..mid).collect(),
            vec![2, 3, 4, 5],
        ),
        (
            "hr_door_fr",
            (mid + 1..cabin_last).collect(),
            vec![14, 13, 12, 11],
        ),
        (
            "hr_door_rr",
            (cabin_first + 1..mid).collect(),
            vec![14, 13, 12, 11],
        ),
    ];
    for (group, rings, rows) in &panels {
        if rings.len() < 2 {
            continue;
        }
        let mut grid: Vec<Vec<usize>> = Vec::new();
        for &k in rings {
            let c = axis(&b, k);
            let mut col = Vec::new();
            for &j in rows {
                let s = b.nodes[ring[k][j]].p;
                let out = Vec3::new(s.x - c.x, s.y - c.y, 0.0).normalize_or_zero();
                let q = b.node(
                    format!("{}_{k}_{j}", &group[3..]),
                    s + out * 0.04,
                    p.node_weight * 0.6,
                    group,
                );
                col.push(q);
            }
            grid.push(col);
        }
        let (r, c) = (grid.len(), rows.len());
        for i in 0..r {
            for j in 0..c {
                if j + 1 < c {
                    b.beam(grid[i][j], grid[i][j + 1], p.deform, p.strength, "");
                }
                if i + 1 < r {
                    b.beam(grid[i][j], grid[i + 1][j], p.deform, p.strength, "");
                    if j + 1 < c {
                        b.beam(grid[i][j], grid[i + 1][j + 1], p.deform, p.strength, "");
                        b.beam(grid[i][j + 1], grid[i + 1][j], p.deform, p.strength, "");
                        let k0 = rings[i];
                        let inside = 0.5 * (axis(&b, k0) + axis(&b, rings[i + 1]));
                        b.quad(
                            grid[i][j],
                            grid[i + 1][j],
                            grid[i + 1][j + 1],
                            grid[i][j + 1],
                            inside,
                        );
                    }
                }
                // Rests on the shell under it.
                b.support(grid[i][j], ring[rings[i]][rows[j]]);
            }
        }
        // Mounts on the panel's edge (hinges and latches), one break group.
        let edge_rows = [0usize, c - 1];
        for i in [0usize, r - 1] {
            for &j in &edge_rows {
                let q = grid[i][j];
                let s = ring[rings[i]][rows[j]];
                let others = b.nearest(b.nodes[s].p, &shell, 4);
                for o in others {
                    b.beam(q, o, p.deform, p.strength * p.mount_strength, group);
                }
            }
        }
    }

    // Axles: an inner and an outer node per wheel, braced into the floor.
    let wb = 0.31 * p.length;
    let mut wheels = Vec::new();
    for (wi, (z, side)) in [(wb, 1.0f32), (wb, -1.0), (-wb, 1.0), (-wb, -1.0)]
        .iter()
        .enumerate()
    {
        let inner = b.node(
            format!("ax{wi}i"),
            Vec3::new(side * (hw - 0.24), p.wheel_radius, *z),
            8.0,
            "hr_body",
        );
        let outer = b.node(
            format!("ax{wi}o"),
            Vec3::new(side * (hw + 0.02), p.wheel_radius, *z),
            8.0,
            "hr_body",
        );
        b.beam_k(inner, outer, 5.0, f32::INFINITY, f32::INFINITY, "");
        for &a in &[inner, outer] {
            for s in b.nearest(b.nodes[a].p, &shell, 4) {
                b.beam_k(
                    a,
                    s,
                    2.0,
                    p.deform * p.cell_deform_scale,
                    p.strength * 2.0,
                    "",
                );
            }
        }
        let arm = b.nearest(b.nodes[inner].p, &shell, 1)[0];
        wheels.push((inner, outer, arm, wi < 2));
    }

    // JBeam text. JBeam frame: x left, y rear, z up (Rubble: x left, y up, z forward).
    let jb = |v: Vec3| (v.x, -v.z, v.y);
    let mut s = String::new();
    let _ = writeln!(s, "{{\n\"rubble_saloon_hr\": {{");
    let _ = writeln!(
        s,
        "  \"information\": {{\"name\": \"{}\", \"authors\": \"Rubble (procedural)\"}},",
        p.name
    );
    let _ = writeln!(s, "  \"slotType\": \"main\",");
    let mid_ring = (cabin_first + cabin_last) / 2;
    let _ = writeln!(
        s,
        "  \"refNodes\": [[\"ref:\", \"back:\", \"left:\", \"up:\"], [\"s{mid_ring}_0\", \"s{}_0\", \"s{mid_ring}_2\", \"s{mid_ring}_8\"]],",
        mid_ring - 3
    );
    let _ = writeln!(s, "  \"nodes\": [[\"id\", \"posX\", \"posY\", \"posZ\"],");
    let _ = writeln!(s, "    {{\"selfCollision\": true, \"collision\": true}},");
    let (mut last_w, mut last_g) = (f32::NAN, "");
    for nd in &b.nodes {
        if nd.mass != last_w || nd.group != last_g {
            let _ = writeln!(
                s,
                "    {{\"nodeWeight\": {:.3}, \"group\": \"{}\"}},",
                nd.mass, nd.group
            );
            last_w = nd.mass;
            last_g = nd.group;
        }
        let (x, y, z) = jb(nd.p);
        let _ = writeln!(s, "    [\"{}\", {x:.4}, {y:.4}, {z:.4}],", nd.id);
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(s, "  \"beams\": [[\"id1:\", \"id2:\"],");
    let mut last: Option<BeamProps> = None;
    let fmt = |v: f32| {
        if v.is_finite() {
            format!("{v}")
        } else {
            "\"FLT_MAX\"".into()
        }
    };
    for (a, bb, pr) in &b.beams {
        if last != Some(*pr) {
            let _ = writeln!(
                s,
                "    {{\"beamType\": \"{}\", \"beamSpring\": {}, \"beamDamp\": {}, \"beamDeform\": {}, \"beamStrength\": {}, \"breakGroup\": \"{}\"}},",
                pr.kind,
                fmt(pr.spring),
                fmt(pr.damp),
                fmt(pr.deform),
                fmt(pr.strength),
                pr.group
            );
            last = Some(*pr);
        }
        let _ = writeln!(s, "    [\"{}\", \"{}\"],", b.nodes[*a].id, b.nodes[*bb].id);
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(s, "  \"triangles\": [[\"id1:\", \"id2:\", \"id3:\"],");
    for t in &b.tris {
        let _ = writeln!(
            s,
            "    [\"{}\", \"{}\", \"{}\"],",
            b.nodes[t[0]].id, b.nodes[t[1]].id, b.nodes[t[2]].id
        );
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(
        s,
        "  \"pressureWheels\": [[\"name\", \"hubGroup\", \"group\", \"node1:\", \"node2:\", \"nodeS\", \"nodeArm:\", \"wheelDir\"],"
    );
    let _ = writeln!(
        s,
        "    {{\"radius\": {}, \"tireWidth\": 0.21, \"brakeTorque\": 1500}},",
        p.wheel_radius
    );
    let names = ["FL", "FR", "RL", "RR"];
    for (wi, (inner, outer, arm, _)) in wheels.iter().enumerate() {
        let dir = if wi % 2 == 0 { 1 } else { -1 };
        let _ = writeln!(
            s,
            "    [\"{}\", \"wheel_{0}\", \"tire_{0}\", \"{}\", \"{}\", 9999, \"{}\", {dir}],",
            names[wi], b.nodes[*inner].id, b.nodes[*outer].id, b.nodes[*arm].id
        );
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(
        s,
        "  \"powertrain\": [[\"type\", \"name\", \"inputName\", \"inputIndex\"],"
    );
    let _ = writeln!(
        s,
        "    [\"combustionEngine\", \"mainEngine\", \"dummy\", 0],"
    );
    let _ = writeln!(
        s,
        "    [\"shaft\", \"axleFL\", \"mainEngine\", 1, {{\"connectedWheel\": \"FL\"}}],"
    );
    let _ = writeln!(
        s,
        "    [\"shaft\", \"axleFR\", \"mainEngine\", 1, {{\"connectedWheel\": \"FR\"}}],"
    );
    let _ = writeln!(s, "  ],");
    let _ = writeln!(s, "}}\n}}");
    let _ = wheels.iter().filter(|w| w.3).count();
    (
        s,
        HiresLayout {
            stations: n,
            cabin,
            cabin_first,
            cabin_last,
        },
    )
}

/// The high-resolution car through the JBeam importer.
pub fn hires_car_desc(p: &HiresParams) -> (SoftVehicleDesc, HiresLayout) {
    let (text, layout) = hires_car_jbeam(p);
    let d = crate::jbeam::load_jbeam(&[("rubble_saloon_hr.jbeam", &text)], None)
        .expect("generated JBeam loads");
    (d, layout)
}

/// Node indices whose names start with `prefix`.
pub fn named(desc: &SoftVehicleDesc, prefix: &str) -> Vec<u32> {
    desc.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.name.starts_with(prefix))
        .map(|(i, _)| i as u32)
        .collect()
}

/// Shell ring `k`'s nodes.
pub fn ring_nodes(desc: &SoftVehicleDesc, k: usize) -> Vec<u32> {
    (0..HR_RING)
        .filter_map(|j| {
            let id = format!("s{k}_{j}");
            desc.nodes
                .iter()
                .position(|n| n.name == id)
                .map(|i| i as u32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hires_car_is_beamng_scale_and_imports_cleanly() {
        let (d, layout) = hires_car_desc(&HiresParams::default());
        println!(
            "{} nodes, {} beams, {} triangles, {:.0} kg, cabin rings {}..={}, warnings {:?}",
            d.nodes.len(),
            d.beams.len(),
            d.triangles.len(),
            d.total_mass(),
            layout.cabin_first,
            layout.cabin_last,
            d.warnings
        );
        assert!((700..=1100).contains(&d.nodes.len()));
        assert!((4500..=7000).contains(&d.beams.len()));
        assert!(d.total_mass() > 1100.0 && d.total_mass() < 1600.0);
        assert_eq!(d.wheels.len(), 4);
        assert_eq!(d.wheels.iter().filter(|w| w.propulsion != 0).count(), 2);
        assert!(d.warnings.is_empty(), "{:?}", d.warnings);
        for g in [
            "hr_body",
            "hr_bonnet",
            "hr_boot",
            "hr_door_fl",
            "hr_door_rr",
            "hr_engine",
        ] {
            assert!(d.part(g).is_some_and(|p| !p.nodes.is_empty()), "group {g}");
        }
        // The skin faces out: every shell triangle's normal points away from the
        // car's long axis.
        let rings = (layout.stations * HR_RING) as u32;
        let mut outward = 0;
        let mut shell = 0;
        for t in &d.triangles {
            if t.nodes
                .iter()
                .all(|&i| d.nodes[i as usize].name.starts_with('s'))
            {
                let [a, b, c] = t
                    .nodes
                    .map(|i| Vec3::from_array(d.nodes[i as usize].position));
                let n = (b - a).cross(c - a);
                let m = (a + b + c) / 3.0;
                shell += 1;
                if n.dot(Vec3::new(m.x, m.y - 0.7, 0.0)) > 0.0 {
                    outward += 1;
                }
            }
        }
        let _ = rings;
        assert!(outward as f32 > 0.97 * shell as f32, "{outward} of {shell}");
    }
}

#[cfg(test)]
mod oracle_rate {
    #[test]
    fn oracle_rate_for_the_hires_car() {
        let (d, _) = super::hires_car_desc(&super::HiresParams::default());
        let mut sim = crate::explicit_ref::ExplicitSim::default();
        sim.add_vehicle(&d, glam::Vec3::ZERO, glam::Quat::IDENTITY, glam::Vec3::ZERO);
        let t = std::time::Instant::now();
        for _ in 0..200 {
            sim.substep();
        }
        println!(
            "oracle rate {:.0} Hz, {:.2} ms per substep",
            sim.hz(),
            t.elapsed().as_secs_f64() * 5.0
        );
    }
}
