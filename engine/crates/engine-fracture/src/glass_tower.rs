//! A glass office tower as ScenePack v2 nodes and bonds (`town_pack` puts one in a town).
//!
//! Built as one is (after vibe-land's town-kit office tower): a steel column grid on fixed
//! concrete footings carries reinforced-concrete floor slabs; a curtain wall hangs off the
//! slab edges and carries nothing: per storey and bay a metal spandrel over the slab edge, steel
//! mullions at the bay edges, and glass between them. Every pane is cut into Voronoi shards
//! bonded by a seam a tenth as strong as the glass (`seam_scale`), and held in its frame by
//! glazing joints, so the glass comes away and shatters long before the frame does.
//!
//! The frame: Y up, metres, the tower's footprint centred on x = z = 0, grade at y = 0.

/// A material, as a ScenePack's `defaults.solver.materials` entry (Pa, kg/m³).
#[derive(Clone, Debug, PartialEq)]
pub struct PackMaterial {
    pub name: String,
    /// Compression, tension, shear: elastic then fatal.
    pub compression: [f64; 2],
    pub tension: [f64; 2],
    pub shear: [f64; 2],
    pub elastic_modulus: f64,
    pub residual_area_fraction: f64,
    pub density: f64,
    pub color: &'static str,
    pub opacity: f64,
    pub texture_key: Option<&'static str>,
    pub roughness: f64,
    pub metalness: f64,
}

/// A node's collider, in its own frame (centred on its centroid).
#[derive(Clone, Debug, PartialEq)]
pub enum Collider {
    Cuboid { half: [f64; 3] },
    Hull { points: Vec<[f64; 3]> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub centroid: [f64; 3],
    /// 0: fixed (a support).
    pub mass: f64,
    pub volume: f64,
    /// Bounding-box size.
    pub size: [f64; 3],
    pub collider: Collider,
    pub kind: &'static str,
    /// Into [`Tower::materials`].
    pub material: usize,
    pub piece: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub centroid: [f64; 3],
    /// From `a` to `b`.
    pub normal: [f64; 3],
    pub area: f64,
    /// Into [`Tower::materials`].
    pub material: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tower {
    pub materials: Vec<PackMaterial>,
    pub nodes: Vec<Node>,
    pub bonds: Vec<Bond>,
}

#[derive(Clone, Debug)]
pub struct GlassTowerOptions {
    pub storeys: u32,
    pub storey_height: f64,
    /// Half the footprint's side (m).
    pub half: f64,
    /// Bays per face (the slab panels follow them).
    pub bays: u32,
    /// Shards per pane (1: whole panes).
    pub shards: u32,
    /// The shard seams' tension and shear, as a fraction of the glass's.
    pub seam_scale: f64,
    pub seed: u64,
}

impl Default for GlassTowerOptions {
    /// Ten 3 m storeys on a 14 m square: the town's ten-storey towers' size.
    fn default() -> Self {
        Self { storeys: 10, storey_height: 3.0, half: 7.0, bays: 3, shards: 7, seam_scale: 0.1, seed: 0x6c61_7373 }
    }
}

const FOOTING: f64 = 0.6;
const SLAB: f64 = 0.25;
const COLUMN: f64 = 0.35;
/// The spandrel band over each slab edge, above the slab's top.
const SPANDREL_UP: f64 = 0.65;
const PARAPET_UP: f64 = 1.1;
/// The curtain wall's depth outside the slab edge, and the glass's within it.
const SKIN: f64 = 0.06;
const GLASS: [f64; 2] = [0.015, 0.045];
const MULLION: f64 = 0.08;

// Material slots in `Tower::materials`.
const CONCRETE: usize = 0;
const STEEL: usize = 1;
const SPANDREL: usize = 2;
const GLASS_MAT: usize = 3;
const GLAZING_JOINT: usize = 4;
const SEAM: usize = 5;

fn materials(seam_scale: f64) -> Vec<PackMaterial> {
    // vibe-land's office tower's (town-kit office-tower.mjs), its spandrel renamed so the stress
    // import classes it as metal.
    let concrete = PackMaterial {
        name: "office-slab-concrete".into(),
        compression: [12e6, 120e6],
        tension: [1.2e6, 12e6],
        shear: [1.6e6, 16e6],
        elastic_modulus: 30e9,
        residual_area_fraction: 0.1,
        density: 2400.0,
        color: "#c9c6bf",
        opacity: 1.0,
        texture_key: Some("white-concrete"),
        roughness: 0.9,
        metalness: 0.0,
    };
    let steel = PackMaterial {
        name: "office-steel".into(),
        compression: [180e6, 2160e6],
        tension: [180e6, 2160e6],
        shear: [108e6, 1296e6],
        elastic_modulus: 200e9,
        residual_area_fraction: 0.6,
        density: 7850.0,
        color: "#3d4247",
        opacity: 1.0,
        texture_key: Some("metal"),
        roughness: 0.45,
        metalness: 0.7,
    };
    let spandrel = PackMaterial {
        name: "spandrel-metal".into(),
        density: 2700.0,
        color: "#56606a",
        roughness: 0.35,
        metalness: 0.6,
        ..steel.clone()
    };
    let glass = PackMaterial {
        name: "curtain-glass".into(),
        compression: [60e6, 63e6],
        tension: [3e6, 3.15e6],
        shear: [5.4e6, 5.67e6],
        elastic_modulus: 70e9,
        residual_area_fraction: 0.0,
        density: 2500.0,
        color: "#b8d7da",
        opacity: 0.24,
        texture_key: None,
        roughness: 0.08,
        metalness: 0.0,
    };
    let joint = PackMaterial {
        name: "glazing-joint".into(),
        compression: [2e6, 4e6],
        tension: [0.2e6, 0.4e6],
        shear: [0.3e6, 0.6e6],
        elastic_modulus: 2e9,
        color: "#ffffff",
        opacity: 0.55,
        roughness: 0.06,
        ..glass.clone()
    };
    let seam = PackMaterial {
        name: "curtain-glass-shard-seam".into(),
        tension: glass.tension.map(|x| x * seam_scale),
        shear: glass.shear.map(|x| x * seam_scale),
        ..glass.clone()
    };
    vec![concrete, steel, spandrel, glass, joint, seam]
}

/// A curtain-wall face: its outward normal and the in-plane axis `u` (y is up).
#[derive(Clone, Copy)]
struct Face {
    n: [f64; 3],
    u: [f64; 3],
    /// The face's span along `u` (front and back cover the corners).
    span: [f64; 2],
}

impl Face {
    /// A point at `u` along the face, height `y`, `d` outside the slab edge.
    fn at(&self, half: f64, u: f64, y: f64, d: f64) -> [f64; 3] {
        [0, 1, 2].map(|k| self.u[k] * u + self.n[k] * (half + d) + if k == 1 { y } else { 0.0 })
    }
}

struct Build {
    tower: Tower,
    piece: u32,
}

impl Build {
    fn node(&mut self, node: Node) -> usize {
        self.tower.nodes.push(node);
        self.tower.nodes.len() - 1
    }

    fn cuboid(&mut self, lo: [f64; 3], hi: [f64; 3], kind: &'static str, material: usize, fixed: bool) -> usize {
        let size = [0, 1, 2].map(|k| hi[k] - lo[k]);
        let volume = size[0] * size[1] * size[2];
        let density = self.tower.materials[material].density;
        self.piece += 1;
        self.node(Node {
            centroid: [0, 1, 2].map(|k| 0.5 * (lo[k] + hi[k])),
            mass: if fixed { 0.0 } else { volume * density },
            volume,
            size,
            collider: Collider::Cuboid { half: size.map(|s| 0.5 * s) },
            kind,
            material,
            piece: self.piece,
        })
    }

    fn bond(&mut self, a: usize, b: usize, centroid: [f64; 3], normal: [f64; 3], area: f64, material: usize) {
        if area > 1e-9 {
            self.tower.bonds.push(Bond { a, b, centroid, normal, area, material });
        }
    }
}

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

type P2 = [f64; 2];

/// `poly` (convex, counter-clockwise) clipped to `dot(p, n) <= c`.
fn clip(poly: &[P2], n: P2, c: f64) -> Vec<P2> {
    let side = |p: P2| p[0] * n[0] + p[1] * n[1] - c;
    let mut out = Vec::with_capacity(poly.len() + 1);
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (sp, sq) = (side(p), side(q));
        if sp <= 0.0 {
            out.push(p);
        }
        if (sp < 0.0) != (sq < 0.0) && (sp - sq).abs() > 1e-15 {
            let t = sp / (sp - sq);
            out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
        }
    }
    out
}

fn area(poly: &[P2]) -> f64 {
    0.5 * (0..poly.len())
        .map(|i| {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum::<f64>()
}

fn centroid(poly: &[P2]) -> P2 {
    let a = area(poly);
    let mut c = [0.0; 2];
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let w = p[0] * q[1] - q[0] * p[1];
        c[0] += (p[0] + q[0]) * w;
        c[1] += (p[1] + q[1]) * w;
    }
    [c[0] / (6.0 * a), c[1] / (6.0 * a)]
}

/// Voronoi cells of `n` jittered seeds in the rectangle `lo..hi` (seeds on a jittered grid, so
/// no shard is a sliver): each cell's polygon, counter-clockwise, and its seed.
fn voronoi(lo: P2, hi: P2, n: u32, rng: &mut Rng) -> Vec<(Vec<P2>, P2)> {
    let rect = vec![lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]];
    if n <= 1 {
        return vec![(rect, [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])])];
    }
    let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
    let cols = ((n as f64 * w / h).sqrt().round() as u32).clamp(1, n);
    let rows = n.div_ceil(cols);
    let mut seeds = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if seeds.len() as u32 == n {
                break;
            }
            let jitter = |rng: &mut Rng| 0.2 + 0.6 * rng.next();
            seeds.push([lo[0] + w * (c as f64 + jitter(rng)) / cols as f64, lo[1] + h * (r as f64 + jitter(rng)) / rows as f64]);
        }
    }
    seeds
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut cell = rect.clone();
            for (j, &o) in seeds.iter().enumerate() {
                if i != j {
                    // Nearer s than o: dot(p, o - s) <= (|o|² - |s|²) / 2.
                    let n = [o[0] - s[0], o[1] - s[1]];
                    let c = 0.5 * (o[0] * o[0] + o[1] * o[1] - s[0] * s[0] - s[1] * s[1]);
                    cell = clip(&cell, n, c);
                }
            }
            (cell, s)
        })
        .collect()
}

/// Build the tower.
pub fn glass_tower(o: &GlassTowerOptions) -> Tower {
    let mut b = Build {
        tower: Tower { materials: materials(o.seam_scale), nodes: Vec::new(), bonds: Vec::new() },
        piece: 0,
    };
    let half = o.half;
    let level = |k: u32| FOOTING + k as f64 * o.storey_height;
    // Bay (and slab panel) edges along each axis.
    let edges: Vec<f64> = (0..=o.bays).map(|i| -half + 2.0 * half * i as f64 / o.bays as f64).collect();
    // The column grid: 4 x 4, inset from the faces (at ±0.25 and ±0.75 of the half side).
    let grid: Vec<f64> = (0..4).map(|i| -0.75 * half + 0.5 * half * i as f64).collect();
    let panel_of = |v: f64| edges.windows(2).position(|e| v >= e[0] && v <= e[1]).unwrap_or(0);

    // Footings and columns.
    let mut footing = vec![vec![0usize; 4]; 4];
    for (i, &x) in grid.iter().enumerate() {
        for (j, &z) in grid.iter().enumerate() {
            let f = 0.3;
            footing[i][j] = b.cuboid([x - f, 0.0, z - f], [x + f, FOOTING, z + f], "foundation", CONCRETE, true);
        }
    }
    let c = 0.5 * COLUMN;
    // Slab panels per storey: slabs[k][px][pz] tops storey k.
    let mut slabs: Vec<Vec<Vec<usize>>> = Vec::new();
    for k in 0..o.storeys {
        let (bottom, top) = (level(k), level(k + 1) - SLAB);
        let mut panels = vec![vec![0usize; o.bays as usize]; o.bays as usize];
        for px in 0..o.bays as usize {
            for pz in 0..o.bays as usize {
                panels[px][pz] = b.cuboid([edges[px], top, edges[pz]], [edges[px + 1], level(k + 1), edges[pz + 1]], "slab", CONCRETE, false);
            }
        }
        let span = 2.0 * half / o.bays as f64;
        for px in 0..o.bays as usize {
            for pz in 0..o.bays as usize {
                let y = top + 0.5 * SLAB;
                if px + 1 < o.bays as usize {
                    b.bond(panels[px][pz], panels[px + 1][pz], [edges[px + 1], y, 0.5 * (edges[pz] + edges[pz + 1])], [1.0, 0.0, 0.0], span * SLAB, CONCRETE);
                }
                if pz + 1 < o.bays as usize {
                    b.bond(panels[px][pz], panels[px][pz + 1], [0.5 * (edges[px] + edges[px + 1]), y, edges[pz + 1]], [0.0, 0.0, 1.0], span * SLAB, CONCRETE);
                }
            }
        }
        for (i, &x) in grid.iter().enumerate() {
            for (j, &z) in grid.iter().enumerate() {
                let col = b.cuboid([x - c, bottom, z - c], [x + c, top, z + c], "column", STEEL, false);
                let below = if k == 0 { footing[i][j] } else { slabs[k as usize - 1][panel_of(x)][panel_of(z)] };
                b.bond(below, col, [x, bottom, z], [0.0, 1.0, 0.0], COLUMN * COLUMN, CONCRETE);
                b.bond(col, panels[panel_of(x)][panel_of(z)], [x, top, z], [0.0, 1.0, 0.0], COLUMN * COLUMN, CONCRETE);
            }
        }
        slabs.push(panels);
    }

    // The curtain wall.
    let faces = [
        Face { n: [0.0, 0.0, -1.0], u: [1.0, 0.0, 0.0], span: [-half - SKIN, half + SKIN] },
        Face { n: [0.0, 0.0, 1.0], u: [1.0, 0.0, 0.0], span: [-half - SKIN, half + SKIN] },
        Face { n: [-1.0, 0.0, 0.0], u: [0.0, 0.0, 1.0], span: [-half, half] },
        Face { n: [1.0, 0.0, 0.0], u: [0.0, 0.0, 1.0], span: [-half, half] },
    ];
    let mut rng = Rng(o.seed);
    for face in faces {
        let box_on = |b: &mut Build, u: [f64; 2], y: [f64; 2], d: [f64; 2], kind: &'static str, material: usize| {
            let (p, q) = (face.at(half, u[0], y[0], d[0]), face.at(half, u[1], y[1], d[1]));
            let lo = [0, 1, 2].map(|k| p[k].min(q[k]));
            let hi = [0, 1, 2].map(|k| p[k].max(q[k]));
            b.cuboid(lo, hi, kind, material, false)
        };
        // Bays along this face: the slab panels' edges, the outer ones out to the face's span.
        let bay = |i: usize| {
            let lo = if i == 0 { face.span[0] } else { edges[i] };
            let hi = if i + 1 == o.bays as usize { face.span[1] } else { edges[i + 1] };
            [lo, hi]
        };
        // The slab panel at a bay along this face's edge.
        let edge_panel = |k: usize, i: usize| {
            let outer = if face.n.iter().sum::<f64>() < 0.0 { 0 } else { o.bays as usize - 1 };
            if face.u[0] != 0.0 { slabs[k][i][outer] } else { slabs[k][outer][i] }
        };
        // Spandrels over each slab edge (k = 1..=storeys, the last the parapet), per bay.
        let mut spandrels: Vec<Vec<usize>> = vec![Vec::new()];
        for k in 1..=o.storeys {
            let up = if k == o.storeys { PARAPET_UP } else { SPANDREL_UP };
            let y = [level(k) - SLAB, level(k) + up];
            let row: Vec<usize> = (0..o.bays as usize)
                .map(|i| box_on(&mut b, bay(i), y, [0.0, SKIN], if k == o.storeys { "parapet" } else { "spandrel" }, SPANDREL))
                .collect();
            for (i, &s) in row.iter().enumerate() {
                let [u0, u1] = bay(i);
                let width = (u1.min(half) - u0.max(-half)).max(0.0);
                let mid = 0.5 * (u0.max(-half) + u1.min(half));
                b.bond(edge_panel(k as usize - 1, i), s, face.at(half, mid, level(k) - 0.5 * SLAB, 0.0), face.n, width * SLAB, CONCRETE);
                if i + 1 < row.len() {
                    b.bond(s, row[i + 1], face.at(half, u1, 0.5 * (y[0] + y[1]), 0.5 * SKIN), face.u, (y[1] - y[0]) * SKIN, SPANDREL);
                }
            }
            spandrels.push(row);
        }
        // Per storey: mullions at the bay edges and the glass between them.
        for k in 0..o.storeys {
            let glass_y = [if k == 0 { FOOTING } else { level(k) + SPANDREL_UP }, level(k + 1) - SLAB];
            let below = if k == 0 { None } else { Some(&spandrels[k as usize]) };
            let above = &spandrels[k as usize + 1];
            // Mullion i stands at bay edge i (0..=bays), inside the face's span.
            let mullion_u = |i: usize| -> [f64; 2] {
                let e = if i == 0 { face.span[0] } else if i == o.bays as usize { face.span[1] } else { edges[i] };
                let lo = (e - 0.5 * MULLION).max(face.span[0]);
                let hi = (e + 0.5 * MULLION).min(face.span[1]);
                [lo, hi]
            };
            let mullions: Vec<usize> = (0..=o.bays as usize)
                .map(|i| box_on(&mut b, mullion_u(i), glass_y, [0.0, SKIN], "mullion", STEEL))
                .collect();
            for (i, &m) in mullions.iter().enumerate() {
                let u = mullion_u(i);
                let mid = 0.5 * (u[0] + u[1]);
                let bay_index = i.min(o.bays as usize - 1);
                b.bond(m, above[bay_index], face.at(half, mid, glass_y[1], 0.5 * SKIN), [0.0, 1.0, 0.0], (u[1] - u[0]) * SKIN, SPANDREL);
                if let Some(below) = below {
                    b.bond(below[bay_index], m, face.at(half, mid, glass_y[0], 0.5 * SKIN), [0.0, 1.0, 0.0], (u[1] - u[0]) * SKIN, SPANDREL);
                }
            }
            // The glass: a pane per bay, in Voronoi shards.
            let t = GLASS[1] - GLASS[0];
            for i in 0..o.bays as usize {
                let (u0, u1) = (mullion_u(i)[1], mullion_u(i + 1)[0]);
                let cells = voronoi([u0, glass_y[0]], [u1, glass_y[1]], o.shards, &mut rng);
                let shards: Vec<usize> = cells
                    .iter()
                    .map(|(poly, _)| {
                        let c2 = centroid(poly);
                        let centre = face.at(half, c2[0], c2[1], 0.5 * (GLASS[0] + GLASS[1]));
                        let points: Vec<[f64; 3]> = poly
                            .iter()
                            .flat_map(|p| GLASS.map(|d| face.at(half, p[0], p[1], d)))
                            .map(|p| [0, 1, 2].map(|k| p[k] - centre[k]))
                            .collect();
                        let lo = [0, 1, 2].map(|k| points.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min));
                        let hi = [0, 1, 2].map(|k| points.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max));
                        let volume = area(poly) * t;
                        b.piece += 1;
                        let piece = b.piece;
                        b.node(Node {
                            centroid: centre,
                            mass: volume * b.tower.materials[GLASS_MAT].density,
                            volume,
                            size: [0, 1, 2].map(|k| hi[k] - lo[k]),
                            collider: Collider::Hull { points },
                            kind: "glazing",
                            material: GLASS_MAT,
                            piece,
                        })
                    })
                    .collect();
                // Seams between shards (each shared Voronoi edge), joints to the frame (each edge on
                // the pane's border).
                const ON: f64 = 1e-6;
                for (a, (poly, s)) in cells.iter().enumerate() {
                    for e in 0..poly.len() {
                        let (p, q) = (poly[e], poly[(e + 1) % poly.len()]);
                        let len = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
                        if len < 1e-6 {
                            continue;
                        }
                        let mid = [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
                        let at = face.at(half, mid[0], mid[1], 0.5 * (GLASS[0] + GLASS[1]));
                        let d2 = |x: P2, y: P2| (x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2);
                        if let Some(o2) = (0..cells.len()).find(|&j| {
                            j != a && (d2(p, *s) - d2(p, cells[j].1)).abs() < ON && (d2(q, *s) - d2(q, cells[j].1)).abs() < ON
                        }) {
                            if a < o2 {
                                let s2 = cells[o2].1;
                                let dir = [s2[0] - s[0], s2[1] - s[1]];
                                let n = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
                                let normal = [0, 1, 2].map(|k| face.u[k] * dir[0] / n + if k == 1 { dir[1] / n } else { 0.0 });
                                b.bond(shards[a], shards[o2], at, normal, len * t, SEAM);
                            }
                            continue;
                        }
                        let (frame, normal) = if (mid[1] - glass_y[0]).abs() < ON {
                            (below.map(|r| r[i]), [0.0, -1.0, 0.0])
                        } else if (mid[1] - glass_y[1]).abs() < ON {
                            (Some(above[i]), [0.0, 1.0, 0.0])
                        } else if (mid[0] - u0).abs() < ON {
                            (Some(mullions[i]), face.u.map(|x| -x))
                        } else if (mid[0] - u1).abs() < ON {
                            (Some(mullions[i + 1]), face.u)
                        } else {
                            (None, [0.0; 3])
                        };
                        if let Some(f) = frame {
                            b.bond(shards[a], f, at, normal, len * t, GLAZING_JOINT);
                        }
                    }
                }
            }
        }
    }
    b.tower
}

/// Put `tower` in a ScenePack in place of the building `group` (`nodeGroups`): that building's
/// nodes and every bond touching them go; the tower stands on the same footprint centre and grade,
/// its nodes in `name`, its materials added to the pack's (by name), its pieces numbered after
/// the pack's. Fails on a per-node array this does not know.
pub fn replace_building(pack: &mut serde_json::Value, group: &str, tower: &Tower, name: &str) -> anyhow::Result<()> {
    use anyhow::{anyhow, bail, ensure};
    use serde_json::{json, Value};
    let sc = pack["scenario"].as_object_mut().ok_or_else(|| anyhow!("pack has no scenario"))?;
    let n = sc.get("nodes").and_then(Value::as_array).map(|a| a.len()).ok_or_else(|| anyhow!("pack has no scenario.nodes"))?;
    const PER_NODE: [&str; 7] = ["nodes", "nodeSizes", "nodeColliders", "nodeTypes", "nodeMaterials", "nodePieces", "nodeGroups"];
    for (k, v) in sc.iter() {
        if let Some(a) = v.as_array() {
            if a.len() == n && !PER_NODE.contains(&k.as_str()) {
                bail!("scenario.{k} has a value per node: replace_building does not know it");
            }
        }
    }
    let groups: Vec<String> = sc["nodeGroups"].as_array().ok_or_else(|| anyhow!("no nodeGroups"))?.iter().map(|g| g.as_str().unwrap_or("").to_string()).collect();
    let gone: Vec<bool> = groups.iter().map(|g| g == group).collect();
    ensure!(gone.iter().any(|&g| g), "no building {group} in the pack");
    // Its footprint centre (x, z) and grade (its lowest point).
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let v3 = |v: &Value| [v["x"].as_f64().unwrap_or(0.0), v["y"].as_f64().unwrap_or(0.0), v["z"].as_f64().unwrap_or(0.0)];
    for i in (0..n).filter(|&i| gone[i]) {
        let (c, s) = (v3(&sc["nodes"][i]["centroid"]), v3(&sc["nodeSizes"][i]));
        for k in 0..3 {
            lo[k] = lo[k].min(c[k] - 0.5 * s[k]);
            hi[k] = hi[k].max(c[k] + 0.5 * s[k]);
        }
    }
    let at = [0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])];
    // Keep the rest, renumbered.
    let mut new_index = vec![usize::MAX; n];
    let mut kept = 0;
    for i in 0..n {
        if !gone[i] {
            new_index[i] = kept;
            kept += 1;
        }
    }
    for k in PER_NODE {
        if let Some(a) = sc.get_mut(k).and_then(Value::as_array_mut) {
            let mut i = 0;
            a.retain(|_| {
                i += 1;
                !gone[i - 1]
            });
        }
    }
    let bonds = sc.get_mut("bonds").and_then(Value::as_array_mut).ok_or_else(|| anyhow!("no bonds"))?;
    bonds.retain(|b| {
        let (a, o) = (b["node0"].as_u64().unwrap_or(0) as usize, b["node1"].as_u64().unwrap_or(0) as usize);
        a < n && o < n && !gone[a] && !gone[o]
    });
    for b in bonds.iter_mut() {
        let (a, o) = (b["node0"].as_u64().unwrap() as usize, b["node1"].as_u64().unwrap() as usize);
        b["node0"] = json!(new_index[a]);
        b["node1"] = json!(new_index[o]);
    }
    add_building(pack, tower, at, name)
}

/// The fracture lab's glass tower scene (a ScenePack v2): the town's glass tower and, 7.8 m off
/// its south face, a two-storey glass pavilion it can be felled onto, as the Unreal town's tower
/// stands by its grocery. Shots are the town's cannonballs (60 cm iron, 7 t).
pub fn felling_scene_pack() -> serde_json::Value {
    let tall = glass_tower(&GlassTowerOptions::default());
    let small = glass_tower(&GlassTowerOptions { storeys: 2, half: 6.0, seed: 0x7061_7669, ..GlassTowerOptions::default() });
    let gap = 7.8 + SKIN;
    let mut pack = serde_json::json!({
        "version": 2,
        "key": "glass-tower",
        "title": "Glass tower and pavilion",
        "defaults": {
            "solver": { "gravity": -9.81, "materials": [] },
            "projectile": { "radius": 0.6, "mass": 7.0e3 },
        },
        "scenario": {
            "nodes": [], "bonds": [], "nodeSizes": [], "nodeColliders": [], "nodeTypes": [],
            "nodeMaterials": [], "nodePieces": [], "nodeGroups": [],
        },
    });
    add_building(&mut pack, &tall, [0.0, 0.0, 0.0], "tower@glass-tower").expect("an empty pack takes a building");
    let south = -(7.0 + SKIN) - gap - (6.0 + SKIN);
    add_building(&mut pack, &small, [0.0, 0.0, south], "pavilion@glass-pavilion").expect("and another");
    pack
}

/// Add `tower` to a ScenePack, its frame's origin at `at`, its nodes in group `name`; its
/// materials join the pack's by name and its pieces are numbered after the pack's.
pub fn add_building(pack: &mut serde_json::Value, tower: &Tower, at: [f64; 3], name: &str) -> anyhow::Result<()> {
    use anyhow::anyhow;
    use serde_json::{json, Value};
    let materials = pack["defaults"]["solver"]["materials"].as_array_mut().ok_or_else(|| anyhow!("pack has no defaults.solver.materials"))?;
    let slot: Vec<usize> = tower
        .materials
        .iter()
        .map(|m| {
            if let Some(i) = materials.iter().position(|x| x["name"].as_str() == Some(m.name.as_str())) {
                return i;
            }
            materials.push(json!({
                "name": m.name,
                "compressionElastic": m.compression[0], "compressionFatal": m.compression[1],
                "tensionElastic": m.tension[0], "tensionFatal": m.tension[1],
                "shearElastic": m.shear[0], "shearFatal": m.shear[1],
                "elasticModulus": m.elastic_modulus, "residualAreaFraction": m.residual_area_fraction,
                "color": m.color, "opacity": m.opacity, "textureKey": m.texture_key,
                "roughness": m.roughness, "metalness": m.metalness, "density": m.density,
            }));
            materials.len() - 1
        })
        .collect();
    let names: Vec<String> = tower.materials.iter().map(|m| m.name.clone()).collect();
    let sc = pack["scenario"].as_object_mut().ok_or_else(|| anyhow!("pack has no scenario"))?;
    let base = sc["nodes"].as_array().map_or(0, |a| a.len());
    let piece0 = sc["nodePieces"].as_array().and_then(|a| a.iter().filter_map(Value::as_u64).max()).map_or(0, |m| m + 1);
    let xyz = |p: [f64; 3]| json!({"x": p[0], "y": p[1], "z": p[2]});
    let world = |p: [f64; 3]| [p[0] + at[0], p[1] + at[1], p[2] + at[2]];
    let mut push = |k: &str, v: Value| -> anyhow::Result<()> {
        sc.get_mut(k).and_then(Value::as_array_mut).ok_or_else(|| anyhow!("pack has no scenario.{k}"))?.push(v);
        Ok(())
    };
    for node in &tower.nodes {
        push("nodes", json!({"centroid": xyz(world(node.centroid)), "mass": node.mass, "volume": node.volume, "m": slot[node.material]}))?;
        push("nodeSizes", xyz(node.size))?;
        push("nodeColliders", match &node.collider {
            Collider::Cuboid { half } => json!({"kind": "cuboid", "halfExtents": xyz(*half)}),
            Collider::Hull { points } => json!({"kind": "convex_hull", "points": points.iter().flatten().collect::<Vec<_>>()}),
        })?;
        push("nodeTypes", json!(node.kind))?;
        push("nodeMaterials", json!(names[node.material]))?;
        push("nodePieces", json!(piece0 + node.piece as u64))?;
        push("nodeGroups", json!(name))?;
    }
    for b in &tower.bonds {
        push("bonds", json!({
            "node0": base + b.a, "node1": base + b.b,
            "centroid": xyz(world(b.centroid)), "normal": xyz(b.normal), "area": b.area, "m": slot[b.material],
        }))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_felling_scene_is_a_pack_the_lab_loads() {
        let pack = felling_scene_pack();
        let bytes = serde_json::to_vec(&pack).unwrap();
        let (scene, info) = crate::pack::build_pack_scene(&bytes).unwrap();
        let tall = glass_tower(&GlassTowerOptions::default());
        let small = glass_tower(&GlassTowerOptions { storeys: 2, half: 6.0, ..GlassTowerOptions::default() });
        assert_eq!(info.nodes, tall.nodes.len() + small.nodes.len());
        assert_eq!(info.bonds, tall.bonds.len() + small.bonds.len());
        assert!(scene.bodies.len() > info.nodes, "a body per node and the ground");
    }

    #[test]
    fn voronoi_cells_tile_the_pane() {
        let mut rng = Rng(7);
        let cells = voronoi([0.0, 0.0], [2.0, 2.5], 7, &mut rng);
        assert_eq!(cells.len(), 7);
        let total: f64 = cells.iter().map(|(p, _)| area(p)).sum();
        assert!((total - 5.0).abs() < 1e-9, "{total}");
        assert!(cells.iter().all(|(p, _)| area(p) > 0.1), "no sliver shards");
    }

    #[test]
    fn the_tower_stands_on_its_footings_and_its_glass_hangs_in_shards() {
        let t = glass_tower(&GlassTowerOptions::default());
        let count = |k: &str| t.nodes.iter().filter(|n| n.kind == k).count();
        assert_eq!(count("foundation"), 16);
        assert_eq!(count("column"), 160);
        assert_eq!(count("slab"), 90);
        // 4 faces x 10 storeys x 3 bays x 7 shards.
        assert_eq!(count("glazing"), 840);
        // Only the footings are fixed; every bond joins two nodes; areas are positive.
        assert!(t.nodes.iter().all(|n| (n.mass == 0.0) == (n.kind == "foundation")));
        assert!(t.bonds.iter().all(|b| b.a != b.b && b.a < t.nodes.len() && b.b < t.nodes.len() && b.area > 0.0));
        // Every shard is bonded: to other shards by seams and to its frame by glazing joints.
        let mut seams = vec![0; t.nodes.len()];
        let mut joints = vec![0; t.nodes.len()];
        for b in &t.bonds {
            for n in [b.a, b.b] {
                match b.material {
                    SEAM => seams[n] += 1,
                    GLAZING_JOINT => joints[n] += 1,
                    _ => {}
                }
            }
        }
        for (i, n) in t.nodes.iter().enumerate() {
            if n.kind == "glazing" {
                assert!(seams[i] > 0, "shard {i} has no seam");
            }
        }
        let framed = t.nodes.iter().enumerate().filter(|(i, n)| n.kind == "glazing" && joints[*i] > 0).count();
        assert!(framed >= 840 / 2, "{framed} of 840 shards on a pane's border");
        // The seams are a tenth of the glass.
        assert!((t.materials[SEAM].tension[1] - 0.1 * t.materials[GLASS_MAT].tension[1]).abs() < 1.0);
        // The total footprint: 14 m, 30.6 m tall to its roof slab, 31.7 m to its parapet.
        let top = t.nodes.iter().map(|n| n.centroid[1] + 0.5 * n.size[1]).fold(0.0, f64::max);
        assert!((top - (0.6 + 30.0 + 1.1)).abs() < 1e-9, "{top}");
    }
}
