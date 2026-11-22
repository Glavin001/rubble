//! Procedural buildings for the fracture lab's district scenes, as ScenePack v2 nodes and
//! bonds (`glass_tower::add_building` puts each in a pack).
//!
//! Four kinds, each built as one is:
//! - a **bearing-wall building**: brick perimeter walls and a spine wall in staggered
//!   courses carry concrete floor panels; windows are glass panes in the middle course;
//! - a **framed building**: reinforced-concrete columns on a 4 to 6 m grid carry downstand
//!   beams and floor panels (a cap over every column), with light facade panels between the
//!   perimeter columns that stop short of the beam above and carry nothing, and a walled
//!   core in buildings of three storeys or more;
//! - a **tower**: a framed building on a small footprint;
//! - a **chimney**: a hollow square masonry stack in interlocking courses.
//!
//! Every piece is an axis-aligned box. Bonds are not authored: [`face_bonds`] welds every
//! pair of boxes that touch face to face over the rectangle they share. Only footings are
//! fixed.
//!
//! The frame: Y up, metres, a building's footprint centred on x = z = 0, grade at y = 0.

use crate::glass_tower::{add_building, Bond, Collider, Node, PackMaterial, Tower};
use std::collections::HashMap;

/// Two faces closer than this touch.
const TOUCH: f64 = 1e-6;
/// The least shared rectangle that makes a bond (m²).
const MIN_BOND_AREA: f64 = 1e-4;
/// The footings' height: a building stands on them, above grade.
const FOOTING: f64 = 0.4;

// Material slots in a building's `Tower::materials`.
const RC: usize = 0;
const BRICK: usize = 1;
const GLASS: usize = 2;
const PANEL: usize = 3;
const FOOT: usize = 4;

/// The buildings' materials. The names are the stress import's calibrated asset materials
/// (stress-ref `scene_pack.rs` `map_material`).
fn materials() -> Vec<PackMaterial> {
    let rc = PackMaterial {
        name: "reinforced-concrete".into(),
        compression: [14e6, 140e6],
        tension: [1.4e6, 10e6],
        shear: [1.8e6, 9e6],
        elastic_modulus: 30e9,
        residual_area_fraction: 0.1,
        density: 2500.0,
        color: "#bdb9b0",
        opacity: 1.0,
        texture_key: Some("white-concrete"),
        roughness: 0.9,
        metalness: 0.0,
    };
    let brick = PackMaterial {
        name: "brick".into(),
        compression: [4e6, 45e6],
        tension: [0.3e6, 0.9e6],
        shear: [0.5e6, 1.4e6],
        elastic_modulus: 15e9,
        residual_area_fraction: 0.0,
        density: 1900.0,
        color: "#9c5a44",
        texture_key: None,
        roughness: 0.95,
        ..rc.clone()
    };
    let glass = PackMaterial {
        name: "glass".into(),
        compression: [60e6, 63e6],
        tension: [0.5e6, 0.6e6],
        shear: [0.6e6, 0.7e6],
        elastic_modulus: 70e9,
        residual_area_fraction: 0.0,
        density: 2500.0,
        color: "#b8d7da",
        opacity: 0.3,
        texture_key: None,
        roughness: 0.08,
        metalness: 0.0,
    };
    let panel = PackMaterial {
        name: "facade-panel".into(),
        compression: [3e6, 30e6],
        tension: [0.5e6, 0.7e6],
        shear: [0.7e6, 0.9e6],
        elastic_modulus: 10e9,
        residual_area_fraction: 0.0,
        density: 1100.0,
        color: "#d9d4c7",
        texture_key: None,
        ..rc.clone()
    };
    let footing = PackMaterial {
        name: "footing-anchor".into(),
        color: "#8f8c86",
        ..rc.clone()
    };
    vec![rc, brick, glass, panel, footing]
}

struct Build {
    tower: Tower,
    piece: u32,
}

impl Build {
    fn new() -> Self {
        Self { tower: Tower { materials: materials(), nodes: Vec::new(), bonds: Vec::new() }, piece: 0 }
    }

    /// A box from `lo` to `hi`. An empty one adds nothing.
    fn cuboid(&mut self, lo: [f64; 3], hi: [f64; 3], kind: &'static str, material: usize) {
        let size = [0, 1, 2].map(|k| hi[k] - lo[k]);
        if size.iter().any(|&s| s <= TOUCH) {
            return;
        }
        let volume = size[0] * size[1] * size[2];
        let fixed = kind == "foundation";
        self.piece += 1;
        self.tower.nodes.push(Node {
            centroid: [0, 1, 2].map(|k| 0.5 * (lo[k] + hi[k])),
            mass: if fixed { 0.0 } else { volume * self.tower.materials[material].density },
            volume,
            size,
            collider: Collider::Cuboid { half: size.map(|s| 0.5 * s) },
            kind,
            material,
            piece: self.piece,
        });
    }

    /// A box given along a wall: `run` its span along `axis` (0 or 2), `across` its span
    /// along the other horizontal axis, `y` its height span.
    fn along(&mut self, axis: usize, run: [f64; 2], across: [f64; 2], y: [f64; 2], kind: &'static str, material: usize) {
        let (mut lo, mut hi) = ([0.0, y[0], 0.0], [0.0, y[1], 0.0]);
        lo[axis] = run[0];
        hi[axis] = run[1];
        lo[2 - axis] = across[0];
        hi[2 - axis] = across[1];
        self.cuboid(lo, hi, kind, material);
    }

    fn finish(mut self) -> Tower {
        self.tower.bonds = face_bonds(&self.tower);
        self.tower
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

    /// An integer in `lo..=hi`.
    fn pick(&mut self, lo: u32, hi: u32) -> u32 {
        lo + ((self.next() * (hi - lo + 1) as f64) as u32).min(hi - lo)
    }
}

/// `a` to `b` cut into equal parts of about `target`: the cut points, both ends included.
fn cuts(a: f64, b: f64, target: f64) -> Vec<f64> {
    let n = (((b - a) / target).round() as usize).max(1);
    let mut out: Vec<f64> = (0..n).map(|i| a + (b - a) * i as f64 / n as f64).collect();
    out.push(b);
    out
}

/// [`cuts`], with every inner cut moved half a part along when `odd`: a course laid
/// across the joints of the course under it.
fn course(a: f64, b: f64, target: f64, odd: bool) -> Vec<f64> {
    let even = cuts(a, b, target);
    if !odd || even.len() < 3 {
        return even;
    }
    let step = even[1] - even[0];
    let mut out = vec![a];
    out.extend(even[..even.len() - 1].iter().map(|x| x + 0.5 * step));
    out.push(b);
    out
}

/// Bonds from geometry: a weld wherever two boxes touch face to face, over the rectangle
/// they share (at least [`MIN_BOND_AREA`]). The bond's material is the weaker side's (by
/// fatal tension). Two fixed boxes are not bonded. Sorted by (a, b).
///
/// Boxes are binned in a uniform grid, so the cost follows the number of neighbours, not
/// the square of the number of boxes.
pub fn face_bonds(tower: &Tower) -> Vec<Bond> {
    const CELL: f64 = 2.0;
    let n = &tower.nodes;
    let lo = |i: usize, k: usize| n[i].centroid[k] - 0.5 * n[i].size[k];
    let hi = |i: usize, k: usize| n[i].centroid[k] + 0.5 * n[i].size[k];
    let cell = |x: f64| (x / CELL).floor() as i32;
    let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for i in 0..n.len() {
        let (c0, c1) = ([0, 1, 2].map(|k| cell(lo(i, k) - TOUCH)), [0, 1, 2].map(|k| cell(hi(i, k) + TOUCH)));
        for x in c0[0]..=c1[0] {
            for y in c0[1]..=c1[1] {
                for z in c0[2]..=c1[2] {
                    grid.entry([x, y, z]).or_default().push(i as u32);
                }
            }
        }
    }
    let mut out = Vec::new();
    for (key, list) in &grid {
        for (s, &a) in list.iter().enumerate() {
            for &b in &list[s + 1..] {
                let (a, b) = (a as usize, b as usize);
                if n[a].mass == 0.0 && n[b].mass == 0.0 {
                    continue;
                }
                // A pair shares several cells: it is taken in the one holding the low
                // corner of what the two boxes share.
                let home = [0, 1, 2].map(|k| cell(lo(a, k).max(lo(b, k)) - TOUCH));
                if home != *key {
                    continue;
                }
                // Separation per axis (negative: overlap).
                let gap = [0, 1, 2].map(|k| lo(a, k).max(lo(b, k)) - hi(a, k).min(hi(b, k)));
                let touching: Vec<usize> = (0..3).filter(|&k| gap[k].abs() <= TOUCH).collect();
                let [axis] = touching[..] else { continue };
                let others = [(axis + 1) % 3, (axis + 2) % 3];
                if others.iter().any(|&k| gap[k] > -TOUCH) {
                    continue;
                }
                let area = gap[others[0]] * gap[others[1]];
                if area < MIN_BOND_AREA {
                    continue;
                }
                let sign = if n[b].centroid[axis] > n[a].centroid[axis] { 1.0 } else { -1.0 };
                let mut centroid = [0.0; 3];
                centroid[axis] = if sign > 0.0 { hi(a, axis) } else { lo(a, axis) };
                for k in others {
                    centroid[k] = 0.5 * (lo(a, k).max(lo(b, k)) + hi(a, k).min(hi(b, k)));
                }
                let mut normal = [0.0; 3];
                normal[axis] = sign;
                let (ma, mb) = (n[a].material, n[b].material);
                let material = if tower.materials[mb].tension[1] < tower.materials[ma].tension[1] { mb } else { ma };
                out.push(Bond { a, b, centroid, normal, area, material });
            }
        }
    }
    out.sort_by_key(|b| (b.a, b.b));
    out
}

#[derive(Clone, Debug)]
pub struct BearingOptions {
    /// The footprint (m): along x, along z.
    pub width: f64,
    pub depth: f64,
    pub storeys: u32,
}

/// A bearing-wall building: brick perimeter walls (and a spine wall where the floor would
/// span more than 8 m) carry a concrete floor of panels at every storey.
pub fn bearing_wall_building(o: &BearingOptions) -> Tower {
    const STOREY: f64 = 3.0;
    const SLAB: f64 = 0.25;
    const WALL: f64 = 0.3;
    const BLOCK: f64 = 1.2;
    const COURSES: usize = 3;
    const PANE: f64 = 0.06;
    let (hx, hz) = (0.5 * o.width, 0.5 * o.depth);
    let mut b = Build::new();
    // Each wall: the axis it runs along, its span along it, its span across, glazed or not.
    let mut walls = vec![
        (0, [-hx, hx], [-hz, -hz + WALL], true),
        (0, [-hx, hx], [hz - WALL, hz], true),
        (2, [-hz + WALL, hz - WALL], [-hx, -hx + WALL], true),
        (2, [-hz + WALL, hz - WALL], [hx - WALL, hx], true),
    ];
    if o.depth > 8.0 {
        walls.push((0, [-hx + WALL, hx - WALL], [-0.5 * WALL, 0.5 * WALL], false));
    }
    for &(axis, run, across, _) in &walls {
        for seg in course(run[0], run[1], BLOCK, false).windows(2) {
            b.along(axis, [seg[0], seg[1]], across, [0.0, FOOTING], "foundation", FOOT);
        }
    }
    for s in 0..o.storeys as usize {
        let y0 = FOOTING + s as f64 * STOREY;
        let rise = (STOREY - SLAB) / COURSES as f64;
        for &(axis, run, across, glazed) in &walls {
            for k in 0..COURSES {
                let y = [y0 + k as f64 * rise, y0 + (k + 1) as f64 * rise];
                let joints = course(run[0], run[1], BLOCK, (s + k) % 2 == 1);
                let last = joints.len() - 2;
                for (i, seg) in joints.windows(2).enumerate() {
                    if glazed && k == 1 && i % 2 == 1 && i != last {
                        let mid = 0.5 * (across[0] + across[1]);
                        b.along(axis, [seg[0], seg[1]], [mid - 0.5 * PANE, mid + 0.5 * PANE], y, "glazing", GLASS);
                    } else {
                        b.along(axis, [seg[0], seg[1]], across, y, "wall", BRICK);
                    }
                }
            }
        }
        let (xs, zs) = (cuts(-hx, hx, 2.0), cuts(-hz, hz, 2.0));
        for x in xs.windows(2) {
            for z in zs.windows(2) {
                b.cuboid([x[0], y0 + STOREY - SLAB, z[0]], [x[1], y0 + STOREY, z[1]], "slab", RC);
            }
        }
    }
    b.finish()
}

#[derive(Clone, Debug)]
pub struct FrameOptions {
    /// The footprint (m): along x, along z.
    pub width: f64,
    pub depth: f64,
    pub storeys: u32,
    /// The column grid's spacing, about (m).
    pub bay: f64,
    /// Facade panels and glazing between the perimeter columns.
    pub cladding: bool,
}

/// A span along one axis of a framed floor: over a column line, or between two.
#[derive(Clone, Copy)]
struct Strip {
    lo: f64,
    hi: f64,
    line: bool,
}

/// A floor's strips along one axis: one over each column line (the column's width), and
/// the span between two lines cut into parts of about 2.5 m.
fn strips(lines: &[f64], column: f64) -> Vec<Strip> {
    let mut out = Vec::new();
    for (i, &x) in lines.iter().enumerate() {
        out.push(Strip { lo: x - 0.5 * column, hi: x + 0.5 * column, line: true });
        if let Some(&next) = lines.get(i + 1) {
            for seg in cuts(x + 0.5 * column, next - 0.5 * column, 2.5).windows(2) {
                out.push(Strip { lo: seg[0], hi: seg[1], line: false });
            }
        }
    }
    out
}

/// A framed building: columns on a grid, downstand beams along the grid lines, floor panels
/// with a cap over every column, facade panels that carry nothing, and a walled core from
/// three storeys up.
pub fn framed_building(o: &FrameOptions) -> Tower {
    const STOREY: f64 = 3.2;
    const SLAB: f64 = 0.3;
    const BEAM: f64 = 0.5;
    const COLUMN: f64 = 0.5;
    const SKIN: f64 = 0.15;
    const PANE: f64 = 0.06;
    /// The facade panels stop this far under the beam above: they prop nothing.
    const HEAD_GAP: f64 = 0.04;
    const CORE_WALL: f64 = 0.3;
    let (hx, hz) = (0.5 * o.width, 0.5 * o.depth);
    let lines_x = cuts(-hx + 0.5 * COLUMN, hx - 0.5 * COLUMN, o.bay);
    let lines_z = cuts(-hz + 0.5 * COLUMN, hz - 0.5 * COLUMN, o.bay);
    let (sx, sz) = (strips(&lines_x, COLUMN), strips(&lines_z, COLUMN));
    // The core: inside the middle bay, clear of its beams.
    let core = (o.storeys >= 3 && lines_x.len() >= 2 && lines_z.len() >= 2).then(|| {
        let (i, j) = ((lines_x.len() - 1) / 2, (lines_z.len() - 1) / 2);
        let inset = 0.5 * COLUMN + 0.5;
        ([lines_x[i] + inset, lines_x[i + 1] - inset], [lines_z[j] + inset, lines_z[j + 1] - inset])
    });
    let mut b = Build::new();
    for s in 0..o.storeys as usize {
        let y0 = FOOTING + s as f64 * STOREY;
        let under = y0 + STOREY - SLAB;
        let ground = s == 0;
        for &x in &lines_x {
            for &z in &lines_z {
                let (lo, hi) = ([x - 0.5 * COLUMN, z - 0.5 * COLUMN], [x + 0.5 * COLUMN, z + 0.5 * COLUMN]);
                if ground {
                    b.cuboid([lo[0], 0.0, lo[1]], [hi[0], FOOTING, hi[1]], "foundation", FOOT);
                }
                b.cuboid([lo[0], y0, lo[1]], [hi[0], under, hi[1]], "column", RC);
            }
        }
        // Beams under the floor along every grid line, between the columns.
        for x in &sx {
            for z in &sz {
                if x.line != z.line {
                    b.cuboid([x.lo, under - BEAM, z.lo], [x.hi, under, z.hi], "beam", RC);
                }
                b.cuboid([x.lo, under, z.lo], [x.hi, y0 + STOREY, z.hi], "slab", RC);
            }
        }
        if o.cladding {
            let top = under - BEAM - HEAD_GAP;
            let rows = [[y0, 0.5 * (y0 + top)], [0.5 * (y0 + top), top]];
            // Each face: the axis it runs along, the grid lines along it, its outer skin.
            let faces = [
                (0, &lines_x, [-hz, -hz + SKIN]),
                (0, &lines_x, [hz - SKIN, hz]),
                (2, &lines_z, [-hx, -hx + SKIN]),
                (2, &lines_z, [hx - SKIN, hx]),
            ];
            for (axis, lines, skin) in faces {
                for bay in lines.windows(2) {
                    let joints = cuts(bay[0] + 0.5 * COLUMN, bay[1] - 0.5 * COLUMN, 1.25);
                    for (i, seg) in joints.windows(2).enumerate() {
                        if ground {
                            b.along(axis, [seg[0], seg[1]], skin, [0.0, FOOTING], "foundation", FOOT);
                        }
                        b.along(axis, [seg[0], seg[1]], skin, rows[0], "cladding", PANEL);
                        if i % 2 == 1 {
                            let mid = 0.5 * (skin[0] + skin[1]);
                            b.along(axis, [seg[0], seg[1]], [mid - 0.5 * PANE, mid + 0.5 * PANE], rows[1], "glazing", GLASS);
                        } else {
                            b.along(axis, [seg[0], seg[1]], skin, rows[1], "cladding", PANEL);
                        }
                    }
                }
            }
        }
        if let Some((cx, cz)) = core {
            let walls = [
                (0, cx, [cz[0], cz[0] + CORE_WALL]),
                (0, cx, [cz[1] - CORE_WALL, cz[1]]),
                (2, [cz[0] + CORE_WALL, cz[1] - CORE_WALL], [cx[0], cx[0] + CORE_WALL]),
                (2, [cz[0] + CORE_WALL, cz[1] - CORE_WALL], [cx[1] - CORE_WALL, cx[1]]),
            ];
            let mid = 0.5 * (y0 + under);
            for (axis, run, across) in walls {
                for (k, y) in [[y0, mid], [mid, under]].into_iter().enumerate() {
                    for seg in course(run[0], run[1], 1.5, (s + k) % 2 == 1).windows(2) {
                        b.along(axis, [seg[0], seg[1]], across, y, "core", RC);
                    }
                }
                if ground {
                    for seg in course(run[0], run[1], 1.5, false).windows(2) {
                        b.along(axis, [seg[0], seg[1]], across, [0.0, FOOTING], "foundation", FOOT);
                    }
                }
            }
        }
    }
    b.finish()
}

/// A hollow square masonry chimney, `side` across and `courses` metres tall above its
/// footing. Alternate courses run their long blocks along x, then along z, so every corner
/// interlocks.
pub fn chimney(side: f64, courses: u32) -> Tower {
    const WALL: f64 = 0.5;
    let h = 0.5 * side;
    let mut b = Build::new();
    for k in 0..=courses as usize {
        let (y, kind, material) = if k == 0 { ([0.0, FOOTING], "foundation", FOOT) } else { ([FOOTING + (k - 1) as f64, FOOTING + k as f64], "wall", BRICK) };
        let long = if k % 2 == 0 { 0 } else { 2 };
        for across in [[-h, -h + WALL], [h - WALL, h]] {
            b.along(long, [-h, h], across, y, kind, material);
            b.along(2 - long, [-h + WALL, h - WALL], across, y, kind, material);
        }
    }
    b.finish()
}

/// The fracture lab's district scenes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistrictScene {
    /// One five-storey bearing-wall building.
    Block,
    /// One five-storey framed building.
    Frame,
    /// A ten-storey tower beside a three-storey bearing-wall building it can fall on.
    Tower,
    /// Sixteen lots, 25 m apart.
    District,
    /// Thirty-six lots.
    MixedCity,
}

impl DistrictScene {
    pub const ALL: [DistrictScene; 5] = [Self::Block, Self::Frame, Self::Tower, Self::District, Self::MixedCity];

    /// The scene's key: its name in menus, URLs and the pack.
    pub fn key(self) -> &'static str {
        match self {
            Self::Block => "district-block",
            Self::Frame => "district-frame",
            Self::Tower => "district-tower",
            Self::District => "district",
            Self::MixedCity => "mixed-city",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Block => "Bearing-wall building",
            Self::Frame => "Framed building",
            Self::Tower => "Tower beside a low block",
            Self::District => "District (16 lots)",
            Self::MixedCity => "Mixed city (36 lots)",
        }
    }

    pub fn from_key(key: &str) -> Option<DistrictScene> {
        Self::ALL.into_iter().find(|s| s.key() == key)
    }
}

/// The lots' spacing (m): a tower reaches its neighbour when it goes over.
const LOT: f64 = 25.0;

/// The buildings of a scene: each with its name and where its footprint's centre stands.
pub fn district_buildings(scene: DistrictScene) -> Vec<(String, [f64; 3], Tower)> {
    let frame = |width: f64, depth: f64, storeys: u32| framed_building(&FrameOptions { width, depth, storeys, bay: 5.0, cladding: true });
    let bearing = |width: f64, depth: f64, storeys: u32| bearing_wall_building(&BearingOptions { width, depth, storeys });
    match scene {
        DistrictScene::Block => vec![("block".into(), [0.0; 3], bearing(16.0, 12.0, 5))],
        DistrictScene::Frame => vec![("frame".into(), [0.0; 3], frame(16.0, 11.0, 5))],
        DistrictScene::Tower => vec![
            ("tower".into(), [0.0; 3], frame(9.0, 9.0, 10)),
            ("block".into(), [15.5, 0.0, 0.0], bearing(14.0, 10.0, 3)),
        ],
        DistrictScene::District | DistrictScene::MixedCity => {
            let side = if scene == DistrictScene::District { 4 } else { 6 };
            let mut rng = Rng(if scene == DistrictScene::District { 0x6469_7374 } else { 0x6d69_7865 });
            let mut out = Vec::new();
            for i in 0..side {
                for j in 0..side {
                    let at = [(i as f64 - 0.5 * (side - 1) as f64) * LOT, 0.0, (j as f64 - 0.5 * (side - 1) as f64) * LOT];
                    let roll = rng.next();
                    let (name, tower) = if roll < 0.2 {
                        let w = rng.pick(8, 9) as f64;
                        ("tower", frame(w, w, rng.pick(8, 11)))
                    } else if roll < 0.45 {
                        ("midrise", frame(rng.pick(14, 18) as f64, rng.pick(10, 12) as f64, rng.pick(4, 6)))
                    } else if roll < 0.6 {
                        ("lowrise", frame(rng.pick(12, 16) as f64, rng.pick(9, 11) as f64, rng.pick(2, 3)))
                    } else if roll < 0.85 {
                        ("block", bearing(rng.pick(12, 16) as f64, rng.pick(9, 12) as f64, rng.pick(2, 5)))
                    } else if roll < 0.93 {
                        ("chimney", chimney(2.4, rng.pick(16, 22)))
                    } else {
                        continue;
                    };
                    out.push((format!("{name}-{i}-{j}"), at, tower));
                }
            }
            out
        }
    }
}

/// A district scene as a ScenePack v2 (the lab loads it as it loads the glass tower's).
/// Shots are the town's cannonballs (60 cm iron, 7 t).
pub fn district_scene_pack(scene: DistrictScene) -> serde_json::Value {
    let mut pack = serde_json::json!({
        "version": 2,
        "key": scene.key(),
        "title": scene.title(),
        "defaults": {
            "solver": { "gravity": -9.81, "materials": [] },
            "projectile": { "radius": 0.6, "mass": 7.0e3 },
        },
        "scenario": {
            "nodes": [], "bonds": [], "nodeSizes": [], "nodeColliders": [], "nodeTypes": [],
            "nodeMaterials": [], "nodePieces": [], "nodeGroups": [],
        },
    });
    for (name, at, tower) in district_buildings(scene) {
        add_building(&mut pack, &tower, at, &format!("{name}@{}", scene.key())).expect("an empty pack takes a building");
    }
    pack
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_kind() -> Vec<(&'static str, Tower)> {
        vec![
            ("bearing", bearing_wall_building(&BearingOptions { width: 16.0, depth: 12.0, storeys: 5 })),
            ("bearing, no spine", bearing_wall_building(&BearingOptions { width: 9.0, depth: 7.0, storeys: 2 })),
            ("frame", framed_building(&FrameOptions { width: 16.0, depth: 11.0, storeys: 5, bay: 5.0, cladding: true })),
            ("tower", framed_building(&FrameOptions { width: 9.0, depth: 9.0, storeys: 10, bay: 5.0, cladding: true })),
            ("bare frame", framed_building(&FrameOptions { width: 12.0, depth: 9.0, storeys: 2, bay: 4.0, cladding: false })),
            ("chimney", chimney(2.4, 20)),
        ]
    }

    /// Pairs of boxes that interpenetrate (by more than a micron on every axis), by brute force.
    fn overlaps(t: &Tower) -> usize {
        let n = &t.nodes;
        let mut count = 0;
        for a in 0..n.len() {
            for b in a + 1..n.len() {
                if (0..3).all(|k| (n[a].centroid[k] - n[b].centroid[k]).abs() < 0.5 * (n[a].size[k] + n[b].size[k]) - 1e-6) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Nodes with no bonded path to a fixed node.
    fn unsupported(t: &Tower) -> usize {
        let mut root: Vec<usize> = (0..t.nodes.len()).collect();
        fn find(root: &mut [usize], mut x: usize) -> usize {
            while root[x] != x {
                root[x] = root[root[x]];
                x = root[x];
            }
            x
        }
        for b in &t.bonds {
            let (ra, rb) = (find(&mut root, b.a), find(&mut root, b.b));
            root[ra] = rb;
        }
        let mut held = vec![false; t.nodes.len()];
        for i in 0..t.nodes.len() {
            if t.nodes[i].mass == 0.0 {
                let r = find(&mut root, i);
                held[r] = true;
            }
        }
        (0..t.nodes.len()).filter(|&i| !held[find(&mut root, i)]).count()
    }

    #[test]
    fn no_two_pieces_overlap() {
        for (name, t) in every_kind() {
            assert_eq!(overlaps(&t), 0, "{name}");
        }
    }

    #[test]
    fn every_piece_is_held_and_only_footings_are_fixed() {
        for (name, t) in every_kind() {
            assert_eq!(unsupported(&t), 0, "{name}: pieces with no bonded path to a footing");
            assert!(t.nodes.iter().all(|n| (n.mass == 0.0) == (n.kind == "foundation")), "{name}");
            for n in &t.nodes {
                let bottom = n.centroid[1] - 0.5 * n.size[1];
                if n.kind == "foundation" {
                    assert!(bottom.abs() < 1e-9 && (n.size[1] - FOOTING).abs() < 1e-9, "{name}: a footing off grade");
                } else {
                    assert!(bottom >= FOOTING - 1e-9, "{name}: a {} below the footings' top", n.kind);
                }
            }
            assert!(t.bonds.iter().all(|b| b.a < b.b && b.b < t.nodes.len() && b.area >= MIN_BOND_AREA), "{name}");
        }
    }

    /// The grid finds exactly the bonds the definition gives: every pair, tested directly.
    #[test]
    fn the_grid_finds_every_touching_pair() {
        for (name, t) in every_kind() {
            let n = &t.nodes;
            let mut brute = Vec::new();
            for a in 0..n.len() {
                for b in a + 1..n.len() {
                    if n[a].mass == 0.0 && n[b].mass == 0.0 {
                        continue;
                    }
                    let gap = [0, 1, 2].map(|k| (n[a].centroid[k] - n[b].centroid[k]).abs() - 0.5 * (n[a].size[k] + n[b].size[k]));
                    let touching: Vec<usize> = (0..3).filter(|&k| gap[k].abs() <= TOUCH).collect();
                    let [axis] = touching[..] else { continue };
                    let others = [(axis + 1) % 3, (axis + 2) % 3];
                    if others.iter().all(|&k| gap[k] <= -TOUCH) {
                        let w = others.map(|k| (-gap[k]).min(n[a].size[k]).min(n[b].size[k]));
                        if w[0] * w[1] >= MIN_BOND_AREA {
                            brute.push((a, b));
                        }
                    }
                }
            }
            let found: Vec<(usize, usize)> = t.bonds.iter().map(|b| (b.a, b.b)).collect();
            assert_eq!(found, brute, "{name}");
        }
    }

    #[test]
    fn a_frames_cladding_props_nothing() {
        let t = framed_building(&FrameOptions { width: 16.0, depth: 11.0, storeys: 5, bay: 5.0, cladding: true });
        // No facade piece is bonded to a beam: it stops short of the one above.
        for b in &t.bonds {
            let kinds = [t.nodes[b.a].kind, t.nodes[b.b].kind];
            assert!(!(kinds.contains(&"beam") && (kinds.contains(&"cladding") || kinds.contains(&"glazing"))), "{kinds:?}");
        }
        // A cap of the floor sits over every column.
        let columns: Vec<usize> = (0..t.nodes.len()).filter(|&i| t.nodes[i].kind == "column").collect();
        for &c in &columns {
            let top = t.nodes[c].centroid[1] + 0.5 * t.nodes[c].size[1];
            let capped = t.bonds.iter().any(|b| {
                let other = if b.a == c { b.b } else if b.b == c { b.a } else { return false };
                t.nodes[other].kind == "slab" && (b.centroid[1] - top).abs() < 1e-9 && (b.area - 0.25).abs() < 1e-9
            });
            assert!(capped, "column {c} has no cap");
        }
    }

    #[test]
    fn the_scenes_are_packs_the_lab_loads() {
        for scene in DistrictScene::ALL {
            let buildings = district_buildings(scene);
            let nodes: usize = buildings.iter().map(|b| b.2.nodes.len()).sum();
            let bonds: usize = buildings.iter().map(|b| b.2.bonds.len()).sum();
            let bytes = serde_json::to_vec(&district_scene_pack(scene)).unwrap();
            let (lab, info) = crate::pack::build_pack_scene(&bytes).unwrap();
            assert_eq!((info.nodes, info.bonds), (nodes, bonds), "{}", scene.key());
            assert!(lab.bodies.len() > nodes, "{}: a body per node and the ground", scene.key());
            assert_eq!(DistrictScene::from_key(scene.key()), Some(scene));
        }
    }

    #[test]
    fn the_scenes_are_the_sizes_they_say() {
        let count = |s| district_buildings(s).iter().map(|b| b.2.nodes.len()).sum::<usize>();
        let (district, city) = (count(DistrictScene::District), count(DistrictScene::MixedCity));
        eprintln!("district {district} pieces, mixed city {city}");
        assert!((6_000..40_000).contains(&district), "district: {district} pieces");
        assert!((15_000..90_000).contains(&city), "mixed city: {city} pieces");
        // The same scene every time.
        assert_eq!(district_scene_pack(DistrictScene::Tower), district_scene_pack(DistrictScene::Tower));
    }
}
