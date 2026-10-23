//! Original test content, written as Rigs of Rods `.truck` text so every
//! fixture goes through the real importer: a node–beam hatchback and a braced
//! lattice box. Licensed with Rubble (no GPL or BeamNG content needed).

use std::fmt::Write;

/// Tunables of the procedural car.
#[derive(Clone, Debug)]
pub struct CarParams {
    pub name: String,
    pub length: f32,
    pub width: f32,
    /// Stations (rings of nodes) along the length.
    pub stations: usize,
    pub dry_mass: f32,
    /// Load on the floor nodes (`l` nodes: seats, engine, fuel), lowering the centre of mass.
    pub cargo_mass: f32,
    pub minimass: f32,
    pub spring: f32,
    pub damp: f32,
    pub deform: f32,
    pub strength: f32,
    pub plastic_coef: f32,
    /// Passenger cell (stations between the crumple zones): deform threshold and
    /// strength multipliers over the crumple zones' values.
    pub cell_deform_scale: f32,
    pub cell_strength_scale: f32,
    pub wheel_radius: f32,
    pub bumper: bool,
    /// Bolt-on panels (bonnet, boot lid, four doors, rear bumper): their own nodes
    /// a few cm outside the shell, on mounts of `mount_strength` × strength, one
    /// detacher group per panel, so a hard enough hit tears a whole panel off.
    pub panels: bool,
    pub mount_strength: f32,
    /// An engine block: a dense eight-node box in the front bay on breakable
    /// mounts, carrying `engine_mass` kg (out of the dry mass). A frontal crash
    /// crushes the bay up to it and then pushes it back toward the cabin.
    pub engine_block: bool,
    pub engine_mass: f32,
    /// Pillars (beams joining the roof to the body) are brittle: elastic up to
    /// `pillar_strength` × strength, then they snap. A low bar that catches the
    /// pillars shears the roof off instead of folding it (underride).
    pub pillar_strength: f32,
}

impl Default for CarParams {
    fn default() -> Self {
        Self {
            name: "Rubble Hatch (procedural soft-body test car)".into(),
            length: 4.0,
            width: 1.7,
            stations: 11,
            dry_mass: 520.0,
            cargo_mass: 250.0,
            minimass: 5.0,
            spring: 2.0e6,
            damp: 1500.0,
            deform: 7_000.0,
            strength: 260_000.0,
            plastic_coef: 0.0,
            cell_deform_scale: 3.0,
            cell_strength_scale: 2.0,
            wheel_radius: 0.32,
            bumper: true,
            panels: true,
            mount_strength: 0.3,
            engine_block: true,
            engine_mass: 160.0,
            pillar_strength: 0.05,
        }
    }
}

/// Roof height of the body profile at fraction `t` ∈ [0, 1] from rear to front.
fn roof(t: f32) -> f32 {
    // Trunk lid, rear window rise, roof, windscreen fall, bonnet.
    let pts = [
        (0.0, 0.92),
        (0.12, 0.98),
        (0.28, 1.38),
        (0.55, 1.42),
        (0.72, 1.02),
        (0.85, 0.9),
        (1.0, 0.82),
    ];
    for w in pts.windows(2) {
        let ((t0, y0), (t1, y1)) = (w[0], w[1]);
        if t <= t1 {
            let s = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let s = s * s * (3.0 - 2.0 * s);
            return y0 + (y1 - y0) * s;
        }
    }
    0.82
}

/// Nodes per ring of the procedural car.
pub const CAR_RING: usize = 8;

/// Hatchback: rings of 8 nodes along +Z (forward): floor left/right, belt line
/// left/right, roof (or bonnet) left/right, then floor centre and roof centre.
/// Consecutive rings form a hollow, fully triangulated shell (every outer face
/// cross-braced, a skin of collision triangles), with bulkheads only at the ends
/// and through the engine bay and boot, so the cabin is open like a real body:
/// a door can intrude and a pole can fold the front. Four wheels on dedicated
/// axle nodes, an engine, and a bumper on a detacher group.
pub fn car_truck(p: &CarParams) -> String {
    car_truck_layout(p).0
}

/// The procedural car as a [`crate::SoftVehicleDesc`] with named parts: `body`
/// (the shell and axle nodes), `bonnet`, `boot`, `door_fl`, `door_rl`,
/// `door_fr`, `door_rr`, `bumper_front`, `bumper_rear` and `engine` (those the
/// parameters include).
pub fn car_desc(p: &CarParams) -> (crate::SoftVehicleDesc, CarLayout) {
    let (text, layout) = car_truck_layout(p);
    let mut d = crate::truck::load_truck(&text);
    let ring_nodes = (layout.stations * CAR_RING) as u32;
    let mut body: Vec<u32> = (0..ring_nodes).collect();
    body.extend(ring_nodes..ring_nodes + 8); // axle nodes follow the rings
    let mut parts = vec![("body", body)];
    for (name, nodes) in ["bonnet", "boot", "door_fl", "door_rl", "door_fr", "door_rr"]
        .iter()
        .zip(&layout.panels)
    {
        parts.push((name, nodes.clone()));
    }
    parts.push(("bumper_front", layout.front_bumper.clone()));
    parts.push(("bumper_rear", layout.rear_bumper.clone()));
    parts.push(("engine", layout.engine.clone()));
    parts.push(("roof", layout.roof.clone()));
    d.parts = parts
        .into_iter()
        .filter(|(_, n)| !n.is_empty())
        .map(|(name, nodes)| crate::SoftPart {
            name: name.to_string(),
            nodes,
        })
        .collect();
    (d, layout)
}

/// What the visual mesh needs to know about the procedural car's node layout.
#[derive(Clone, Debug, Default)]
pub struct CarLayout {
    pub stations: usize,
    /// Rings that belong to the passenger cell.
    pub cabin: Vec<bool>,
    pub front_bay: usize,
    pub rear_bay: usize,
    /// First ring of the front doors (the rear doors end there).
    pub mid: usize,
    pub cabin_first: usize,
    pub cabin_last: usize,
    /// Axle positions along Z (front, rear) and the axle height.
    pub axles: [f32; 2],
    pub axle_y: f32,
    /// Panel node indices: bonnet, boot lid, front-left, rear-left, front-right,
    /// rear-right door (empty without panels).
    pub panels: Vec<Vec<u32>>,
    pub front_bumper: Vec<u32>,
    pub rear_bumper: Vec<u32>,
    /// The engine block's eight nodes (empty without one).
    pub engine: Vec<u32>,
    /// Roof nodes (the passenger cell's top rows).
    pub roof: Vec<u32>,
}

/// [`car_truck`] and the layout of its nodes.
pub fn car_truck_layout(p: &CarParams) -> (String, CarLayout) {
    let mut s = String::new();
    let _ = writeln!(s, "{}", p.name);
    let _ = writeln!(
        s,
        "; Generated by engine-softbody::procedural (original Rubble content)."
    );
    let _ = writeln!(s, "fileformatversion 450");
    let _ = writeln!(
        s,
        "globals\n{:.1}, {:.1}, tracks/semi\n",
        p.dry_mass, p.cargo_mass
    );
    let _ = writeln!(s, "minimass {:.1}\n", p.minimass);
    let _ = writeln!(s, "enable_advanced_deformation");
    let _ = writeln!(
        s,
        "set_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, {:.2}\n",
        p.spring, p.damp, p.deform, p.strength, p.plastic_coef
    );
    let n = p.stations.max(4);
    let hw = 0.5 * p.width;
    let mut nodes: Vec<[f32; 3]> = Vec::new();
    let mut cabin = Vec::with_capacity(n);
    for k in 0..n {
        let t = k as f32 / (n - 1) as f32;
        let z = -0.5 * p.length + t * p.length;
        let top = roof(t);
        let is_cabin = top > 1.2;
        cabin.push(is_cabin);
        let taper = 1.0 - 0.08 * ((2.0 * t - 1.0).abs()).powi(4);
        let floor_y = 0.26;
        let belt_y = 0.78;
        let top_w = if is_cabin { hw * 0.8 } else { hw * 0.93 };
        for (x, y) in [
            (hw * taper * 0.97, floor_y),
            (hw * taper, belt_y),
            (top_w * taper, top),
        ] {
            nodes.push([x, y, z]);
            nodes.push([-x, y, z]);
        }
        nodes.push([0.0, floor_y + 0.04, z]); // transmission tunnel
        nodes.push([0.0, top + 0.04, z]);
    }
    let ring = |k: usize, j: usize| k * CAR_RING + j;
    // Perimeter of a ring, in order around the section (L = +X):
    // floorL → floorC → floorR → beltR → topR → topC → topL → beltL → floorL.
    const PERIM: [usize; 8] = [0, 6, 1, 3, 5, 7, 4, 2];
    // Axle nodes: inner (n1) and outer (n2) per wheel.
    let wb = 0.32 * p.length;
    let axle_y = p.wheel_radius;
    let mut wheel_axles = Vec::new();
    for &z in &[wb, -wb] {
        for side in [1.0f32, -1.0] {
            let i = nodes.len();
            nodes.push([side * (hw - 0.22), axle_y, z]);
            nodes.push([side * (hw + 0.02), axle_y, z]);
            wheel_axles.push((i, i + 1, z));
        }
    }
    let bumper_first = nodes.len();
    if p.bumper {
        let z = 0.5 * p.length + 0.18;
        for y in [0.38f32, 0.62] {
            for x in [hw * 0.8, 0.0, -hw * 0.8] {
                nodes.push([x, y, z - if x == 0.0 { 0.0 } else { 0.06 }]);
            }
        }
    }
    // Bolt-on panels: (rings, perimeter rows) of the shell each covers. Doors
    // cover floor→belt on each side, the bonnet and boot lid the top rows of the
    // end bays. Each gets its own nodes, offset outward from the shell nodes.
    let front_bay = (0..n).find(|&k| k > n / 2 && !cabin[k]).unwrap_or(n - 3);
    let rear_bay = (0..n).rev().find(|&k| k < n / 2 && !cabin[k]).unwrap_or(2);
    let cabin_first = rear_bay + 1;
    let cabin_last = front_bay - 1;
    let mid = (cabin_first + cabin_last + 1) / 2;
    let panel_specs: Vec<(Vec<usize>, Vec<usize>)> = if p.panels {
        vec![
            // Bonnet and boot lid: topR, topC, topL.
            ((front_bay..n).collect(), vec![5, 7, 4]),
            ((0..=rear_bay).collect(), vec![5, 7, 4]),
            // Doors: left (beltL, floorL) and right (floorR, beltR), front and rear.
            ((mid..=cabin_last + 1).collect(), vec![2, 0]),
            ((cabin_first - 1..mid).collect(), vec![2, 0]),
            ((mid..=cabin_last + 1).collect(), vec![1, 3]),
            ((cabin_first - 1..mid).collect(), vec![1, 3]),
        ]
    } else {
        Vec::new()
    };
    // Panel node index per (panel, ring position, row position).
    let mut panel_nodes: Vec<Vec<Vec<usize>>> = Vec::new();
    for (ks, rows) in &panel_specs {
        let mut grid = Vec::new();
        for &k in ks {
            let centre = {
                let c: [f32; 3] = (0..6).fold([0.0; 3], |a, j| {
                    let q = nodes[k * CAR_RING + j];
                    [a[0] + q[0] / 6.0, a[1] + q[1] / 6.0, a[2] + q[2] / 6.0]
                });
                c
            };
            let mut col = Vec::new();
            for &j in rows {
                let q = nodes[k * CAR_RING + j];
                // Doors stand off sideways (never below the floor), the lids upward.
                let (dx, dy) = if j < 4 {
                    (q[0] - centre[0], 0.0)
                } else {
                    (q[0] - centre[0], q[1] - centre[1])
                };
                let l = (dx * dx + dy * dy).sqrt().max(1e-4);
                col.push(nodes.len());
                nodes.push([
                    q[0] + 0.05 * dx / l,
                    q[1] + 0.05 * dy / l + if j < 2 { 0.03 } else { 0.0 },
                    q[2],
                ]);
            }
            grid.push(col);
        }
        panel_nodes.push(grid);
    }
    let rear_bumper_first = nodes.len();
    let mut layout = CarLayout {
        stations: n,
        cabin: cabin.clone(),
        front_bay,
        rear_bay,
        mid,
        cabin_first,
        cabin_last,
        axles: [wb, -wb],
        axle_y,
        panels: panel_nodes
            .iter()
            .map(|g| g.iter().flatten().map(|&i| i as u32).collect())
            .collect(),
        front_bumper: if p.bumper {
            (bumper_first as u32..bumper_first as u32 + 6).collect()
        } else {
            Vec::new()
        },
        rear_bumper: Vec::new(),
        engine: Vec::new(),
        roof: (0..n)
            .filter(|&k| cabin[k])
            .flat_map(|k| [4usize, 5, 7].map(|j| (k * CAR_RING + j) as u32))
            .collect(),
    };
    if p.panels {
        layout.rear_bumper = (rear_bumper_first as u32..rear_bumper_first as u32 + 6).collect();
    }
    if p.panels {
        let z = -0.5 * p.length - 0.16;
        for y in [0.38f32, 0.62] {
            for x in [hw * 0.8, 0.0, -hw * 0.8] {
                nodes.push([x, y, z + if x == 0.0 { 0.0 } else { 0.06 }]);
            }
        }
    }
    let ring_z = |k: usize| -0.5 * p.length + k as f32 / (n - 1) as f32 * p.length;
    let engine_first = nodes.len();
    let bay_rings: Vec<usize> = (front_bay..n).collect();
    let engine_z = 0.5 * (ring_z(front_bay) + ring_z(n - 1));
    if p.engine_block {
        for &y in &[0.4f32, 0.72] {
            for &z in &[engine_z - 0.22, engine_z + 0.22] {
                for &x in &[0.26f32, -0.26] {
                    nodes.push([x, y, z]);
                }
            }
        }
        layout.engine = (engine_first as u32..engine_first as u32 + 8).collect();
    }
    // Shell faces a panel covers (it is the skin there): (ring k, row a, row b).
    let covered = |k: usize, a: usize, b: usize| {
        panel_specs.iter().any(|(ks, rows)| {
            ks.contains(&k)
                && ks.contains(&(k + 1))
                && rows
                    .windows(2)
                    .any(|w| (w[0] == a && w[1] == b) || (w[0] == b && w[1] == a))
        })
    };
    let _ = writeln!(s, "nodes");
    let ring_nodes = n * CAR_RING;
    for (i, q) in nodes.iter().enumerate() {
        // Floor nodes carry the cargo (seats, engine, fuel): a low centre of mass.
        let floor = i < ring_nodes && matches!(i % CAR_RING, 0 | 1 | 6);
        if p.engine_block && (engine_first..engine_first + 8).contains(&i) {
            let _ = writeln!(
                s,
                "{i}, {:.4}, {:.4}, {:.4}, l, {:.1}",
                q[0],
                q[1],
                q[2],
                p.engine_mass / 8.0
            );
        } else if floor {
            let _ = writeln!(s, "{i}, {:.4}, {:.4}, {:.4}, l", q[0], q[1], q[2]);
        } else {
            let _ = writeln!(s, "{i}, {:.4}, {:.4}, {:.4}", q[0], q[1], q[2]);
        }
    }
    let _ = writeln!(s);
    let mut beams: Vec<(usize, usize, &str)> = Vec::new();
    for k in 0..n {
        let r = |j| ring(k, j);
        for e in 0..8 {
            beams.push((r(PERIM[e]), r(PERIM[(e + 1) % 8]), ""));
        }
        // Floor pan and roof stiffened toward the sills and rails.
        for (a, b) in [(6, 2), (6, 3), (7, 2), (7, 3)] {
            beams.push((r(a), r(b), "i"));
        }
        let end = k == 0 || k == n - 1;
        if !cabin[k] || end {
            // Bulkheads: ends, engine bay and boot (radiator support, strut braces).
            for (a, b) in [
                (2, 3),
                (0, 3),
                (1, 2),
                (6, 7),
                (2, 7),
                (3, 7),
                (0, 7),
                (1, 7),
            ] {
                beams.push((r(a), r(b), "i"));
            }
        }
        if k + 1 < n {
            for j in 0..CAR_RING {
                beams.push((ring(k, j), ring(k + 1, j), ""));
            }
            // Shear panels: both diagonals of every outer face between the rings.
            for e in 0..8 {
                let (a, b) = (PERIM[e], PERIM[(e + 1) % 8]);
                beams.push((ring(k, a), ring(k + 1, b), "i"));
                beams.push((ring(k, b), ring(k + 1, a), "i"));
            }
        }
        if k + 2 < n {
            // Bending stiffeners: sills, roof rails, tunnel.
            for j in [0, 1, 4, 5, 6] {
                beams.push((ring(k, j), ring(k + 2, j), "i"));
            }
        }
    }
    // Axles: tie each axle node into the floor and belt nodes of its ring and the
    // ring on the cabin side only. The rigid core holds the axle nodes, so a tie
    // toward the bumper would brace the crumple zone and stop it folding.
    for &(n1, n2, z) in &wheel_axles {
        let k = (((z + 0.5 * p.length) / p.length) * (n - 1) as f32).round() as usize;
        let left = nodes[n1][0] > 0.0;
        let (f, b) = if left { (0, 2) } else { (1, 3) };
        let inner = if z > 0.0 {
            k.saturating_sub(1)
        } else {
            (k + 1).min(n - 1)
        };
        for kk in [inner, k] {
            for j in [f, b, 6] {
                beams.push((n1, ring(kk, j), "i"));
                beams.push((n2, ring(kk, j), "i"));
            }
        }
        beams.push((n1, n2, "i"));
    }
    // Crumple zones (bonnet and boot stations) and the doors (belt-line nodes)
    // keep the base thresholds; the passenger cell's floor, sills, pillars and roof
    // are stronger, so a crash folds the ends first and a side hit dents the door.
    let crumple = |i: usize| {
        i < ring_nodes && {
            let k = i / CAR_RING;
            let door = cabin[k] && matches!(i % CAR_RING, 2 | 3);
            door || (!cabin[k] && (k <= 1 || k + 4 >= n))
        }
    };
    let roof_node =
        |i: usize| i < ring_nodes && cabin[i / CAR_RING] && matches!(i % CAR_RING, 4 | 5 | 7);
    let pillar = |a: usize, b: usize| roof_node(a) != roof_node(b);
    let _ = writeln!(s, "beams");
    for zone_crumple in [false, true] {
        let (deform, strength) = if zone_crumple {
            (p.deform, p.strength)
        } else {
            (
                p.deform * p.cell_deform_scale,
                p.strength * p.cell_strength_scale,
            )
        };
        let _ = writeln!(
            s,
            "set_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, {:.2}",
            p.spring, p.damp, deform, strength, p.plastic_coef
        );
        for (a, b, o) in beams
            .iter()
            .filter(|(a, b, _)| !pillar(*a, *b) && (crumple(*a) || crumple(*b)) == zone_crumple)
        {
            if o.is_empty() {
                let _ = writeln!(s, "{a}, {b}");
            } else {
                let _ = writeln!(s, "{a}, {b}, {o}");
            }
        }
    }
    // Pillars: brittle (the yield threshold above the strength), so they snap.
    let _ = writeln!(
        s,
        "set_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, 0.0",
        p.spring,
        p.damp,
        p.strength * p.pillar_strength * 2.0,
        p.strength * p.pillar_strength
    );
    for (a, b, o) in beams.iter().filter(|(a, b, _)| pillar(*a, *b)) {
        if o.is_empty() {
            let _ = writeln!(s, "{a}, {b}");
        } else {
            let _ = writeln!(s, "{a}, {b}, {o}");
        }
    }
    if p.bumper {
        // A bolted-on bumper: weaker mounts on detacher group 1 (all go together).
        let f = n - 1;
        let b0 = bumper_first;
        // Bumper bar (stays one piece): lower row b0..b0+2 (L, C, R), upper row b0+3..b0+5.
        for (a, bb) in [
            (b0, b0 + 1),
            (b0 + 1, b0 + 2),
            (b0 + 3, b0 + 4),
            (b0 + 4, b0 + 5),
            (b0, b0 + 3),
            (b0 + 1, b0 + 4),
            (b0 + 2, b0 + 5),
            (b0, b0 + 4),
            (b0 + 1, b0 + 3),
            (b0 + 1, b0 + 5),
            (b0 + 2, b0 + 4),
            (b0, b0 + 2),
            (b0 + 3, b0 + 5),
        ] {
            let _ = writeln!(s, "{a}, {bb}");
        }
        // Its mounts: weaker, on detacher group 1 (they all let go together).
        let _ = writeln!(
            s,
            "\nset_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, 0.0",
            p.spring,
            p.damp,
            p.deform,
            p.strength * 0.15
        );
        let _ = writeln!(s, "detacher_group 1");
        for (a, bb) in [
            (b0, ring(f, 0)),
            (b0, ring(f, 2)),
            (b0 + 3, ring(f, 2)),
            (b0 + 3, ring(f, 0)),
            (b0 + 2, ring(f, 1)),
            (b0 + 2, ring(f, 3)),
            (b0 + 5, ring(f, 3)),
            (b0 + 5, ring(f, 1)),
            (b0 + 1, ring(f, 6)),
            (b0 + 4, ring(f, 6)),
            (b0, ring(f - 1, 0)),
            (b0 + 2, ring(f - 1, 1)),
        ] {
            let _ = writeln!(s, "{a}, {bb}");
        }
        let _ = writeln!(s, "detacher_group end");
    }
    if p.panels {
        // Panels: a braced sheet (strong, so a panel leaves whole), then its mounts
        // to the shell on its own detacher group (groups 2…).
        let mut group = 2;
        let mut sheets: Vec<(Vec<(usize, usize)>, Vec<(usize, usize)>)> = Vec::new();
        for (pi, (ks, rows)) in panel_specs.iter().enumerate() {
            let g = &panel_nodes[pi];
            let mut sheet = Vec::new();
            let mut mounts = Vec::new();
            for (ki, &k) in ks.iter().enumerate() {
                for (ri, &j) in rows.iter().enumerate() {
                    let pn = g[ki][ri];
                    if ri + 1 < rows.len() {
                        sheet.push((pn, g[ki][ri + 1]));
                    }
                    if ki + 1 < ks.len() {
                        sheet.push((pn, g[ki + 1][ri]));
                        if ri + 1 < rows.len() {
                            sheet.push((pn, g[ki + 1][ri + 1]));
                            sheet.push((g[ki][ri + 1], g[ki + 1][ri]));
                        }
                    }
                    // Mounts: the shell nodes of the neighbouring rings (same row)
                    // and of the other rows (same ring), plus the section centre.
                    for kk in [k.wrapping_sub(1), k + 1] {
                        if kk < n {
                            mounts.push((pn, ring(kk, j)));
                        }
                    }
                    for &jj in rows {
                        if jj != j {
                            mounts.push((pn, ring(k, jj)));
                        }
                    }
                    // Out-of-plane anchor: the section's floor centre (a door must not
                    // tie the roof to the body).
                    mounts.push((pn, ring(k, 6)));
                }
            }
            sheets.push((sheet, mounts));
        }
        let rb = rear_bumper_first;
        let r0 = 0;
        let bar: Vec<(usize, usize)> = [
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (0, 3),
            (1, 4),
            (2, 5),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 4),
            (0, 2),
            (3, 5),
        ]
        .iter()
        .map(|&(a, b)| (rb + a, rb + b))
        .collect();
        let bar_mounts: Vec<(usize, usize)> = [
            (0, 0),
            (0, 2),
            (3, 2),
            (3, 0),
            (2, 1),
            (2, 3),
            (5, 3),
            (5, 1),
            (1, 6),
            (4, 6),
        ]
        .iter()
        .map(|&(a, j)| (rb + a, ring(r0, j)))
        .chain([(rb, ring(r0 + 1, 0)), (rb + 2, ring(r0 + 1, 1))])
        .collect();
        sheets.push((bar, bar_mounts));
        let _ = writeln!(
            s,
            "\nset_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, 0.0",
            p.spring,
            p.damp,
            p.deform * p.cell_deform_scale,
            p.strength * p.cell_strength_scale
        );
        for (sheet, _) in &sheets {
            for (a, b) in sheet {
                let _ = writeln!(s, "{a}, {b}");
            }
        }
        let _ = writeln!(
            s,
            "\nset_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, 0.0",
            p.spring,
            p.damp,
            p.deform,
            p.strength * p.mount_strength
        );
        for (_, mounts) in &sheets {
            let _ = writeln!(s, "detacher_group {group}");
            for (a, b) in mounts {
                let _ = writeln!(s, "{a}, {b}");
            }
            group += 1;
        }
        let _ = writeln!(s, "detacher_group end");
    }
    if p.engine_block {
        // The block itself: every pair of its corners, far stronger than the body.
        let _ = writeln!(
            s,
            "\nset_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, 0.0",
            p.spring * 2.0,
            p.damp,
            p.deform * 40.0,
            p.strength * 20.0
        );
        for a in 0..8 {
            for b in a + 1..8 {
                let _ = writeln!(s, "{}, {}", engine_first + a, engine_first + b);
            }
        }
        // Mounts: lower corners to the floor rails, upper ones to the belt line,
        // of the bay rings and the bulkhead behind it. They bend and then tear in
        // a hard frontal crash, letting the block move back.
        let _ = writeln!(
            s,
            "set_beam_defaults {:.0}, {:.0}, {:.0}, {:.0}, 0.05, tracks/beam, 0.0",
            p.spring,
            p.damp,
            p.deform * 2.0,
            p.strength * 0.6
        );
        let mut mounts = Vec::new();
        for e in 0..8 {
            let q = nodes[engine_first + e];
            let rows: &[usize] = if q[1] < 0.5 { &[0, 1] } else { &[2, 3] };
            let mut near: Vec<(f32, usize)> = bay_rings
                .iter()
                .chain(std::iter::once(&(front_bay - 1)))
                .flat_map(|&k| rows.iter().map(move |&j| ring(k, j)))
                .map(|i| {
                    let r = nodes[i];
                    (
                        ((r[0] - q[0]).powi(2) + (r[1] - q[1]).powi(2) + (r[2] - q[2]).powi(2))
                            .sqrt(),
                        i,
                    )
                })
                .collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &(_, i) in near.iter().take(4) {
                mounts.push((engine_first + e, i));
            }
        }
        for (a, b) in mounts {
            let _ = writeln!(s, "{a}, {b}");
        }
    }
    // Collision skin: every outer face of the shell, counter-clockwise from outside.
    let _ = writeln!(s, "\ncab");
    let quad = |s: &mut String, a: usize, b: usize, c: usize, d: usize| {
        let _ = writeln!(s, "{a}, {b}, {c}, c");
        let _ = writeln!(s, "{a}, {c}, {d}, c");
    };
    for k in 0..n - 1 {
        for e in 0..8 {
            let (a, b) = (PERIM[e], PERIM[(e + 1) % 8]);
            // Perimeter runs clockwise seen from the front (+Z): outward normal
            // of (k,a)→(k+1,a)→(k+1,b) points away from the section centre.
            if covered(k, a, b) {
                continue;
            }
            quad(
                &mut s,
                ring(k, a),
                ring(k + 1, a),
                ring(k + 1, b),
                ring(k, b),
            );
        }
    }
    // Panel faces, wound as the shell faces they replace.
    for (pi, (ks, rows)) in panel_specs.iter().enumerate() {
        let g = &panel_nodes[pi];
        let at = |k: usize, j: usize| {
            let ki = ks.iter().position(|&x| x == k)?;
            let ri = rows.iter().position(|&x| x == j)?;
            Some(g[ki][ri])
        };
        for k in 0..n - 1 {
            for e in 0..8 {
                let (a, b) = (PERIM[e], PERIM[(e + 1) % 8]);
                if let (Some(ka), Some(k1a), Some(k1b), Some(kb)) =
                    (at(k, a), at(k + 1, a), at(k + 1, b), at(k, b))
                {
                    quad(&mut s, ka, k1a, k1b, kb);
                }
            }
        }
    }
    // End caps: fans around the belt-line centre (floor centre → roof centre).
    for (k, front) in [(0usize, false), (n - 1, true)] {
        let c = ring(k, 6);
        for e in 2..8 {
            let (a, b) = (ring(k, PERIM[e]), ring(k, PERIM[(e + 1) % 8]));
            if front {
                let _ = writeln!(s, "{c}, {b}, {a}, c");
            } else {
                let _ = writeln!(s, "{c}, {a}, {b}, c");
            }
        }
    }
    if p.bumper {
        let b0 = bumper_first;
        quad(&mut s, b0 + 1, b0, b0 + 3, b0 + 4);
        quad(&mut s, b0 + 2, b0 + 1, b0 + 4, b0 + 5);
    }
    if p.panels {
        // Rear bumper, facing −Z.
        let b0 = rear_bumper_first;
        quad(&mut s, b0, b0 + 1, b0 + 4, b0 + 3);
        quad(&mut s, b0 + 1, b0 + 2, b0 + 5, b0 + 4);
    }
    // Reference frame: centre, back, left.
    let mid = ring(n / 2, 0);
    let _ = writeln!(s, "\ncameras\n{}, {}, {}", ring(n - 1, 6), ring(0, 6), mid);
    let _ = writeln!(s, "\nwheels");
    for &(n1, n2, z) in &wheel_axles {
        // Rear-wheel drive (as the sedan preset): weight moves onto the rear axle
        // under acceleration, so it puts more power down than front drive.
        let propulsed = if z < 0.0 { 1 } else { 0 };
        let _ = writeln!(s, "{:.3}, 0.20, 12, {n1}, {n2}, 9999, 1, {propulsed}, 9999, 15.0, 800000, 4000, tracks/wheelface, tracks/wheelband1", p.wheel_radius);
    }
    let _ = writeln!(
        s,
        // Shift down/up rpm, torque, differential, reverse, neutral, six gears: the
        // rigid sedan preset's drivetrain (engine_vehicle::presets::sedan).
        "\nengine\n2400, 5800, 420, 3.9, 3.2, 1.0, 3.4, 2.2, 1.55, 1.18, 0.95, 0.78, -1.0"
    );
    let _ = writeln!(s, "\nbrakes\n12000\n\nend");
    (s, layout)
}

/// Braced lattice box of `nx × ny × nz` nodes (spacing `step`), with a skin.
#[allow(clippy::too_many_arguments)]
pub fn box_truck(
    nx: usize,
    ny: usize,
    nz: usize,
    step: f32,
    mass: f32,
    spring: f32,
    deform: f32,
    strength: f32,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "lattice box {nx}x{ny}x{nz}\nfileformatversion 450");
    let _ = writeln!(
        s,
        "globals\n{mass:.1}, 0.0, tracks/semi\nminimass 0.5\nenable_advanced_deformation"
    );
    let _ = writeln!(
        s,
        "set_beam_defaults {spring:.0}, {:.0}, {deform:.0}, {strength:.0}, 0.05, tracks/beam, 0.0",
        spring * 0.005
    );
    let id = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
    let _ = writeln!(s, "nodes");
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let _ = writeln!(
                    s,
                    "{}, {:.4}, {:.4}, {:.4}",
                    id(x, y, z),
                    x as f32 * step,
                    y as f32 * step,
                    z as f32 * step
                );
            }
        }
    }
    let _ = writeln!(s, "beams");
    // Every pair of corners of every lattice cell (edges, face and body diagonals).
    let mut seen = std::collections::BTreeSet::new();
    for z in 0..nz.saturating_sub(1) {
        for y in 0..ny.saturating_sub(1) {
            for x in 0..nx.saturating_sub(1) {
                let corners: Vec<usize> = (0..8)
                    .map(|c| id(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1)))
                    .collect();
                for i in 0..8 {
                    for j in i + 1..8 {
                        let (a, b) = (corners[i].min(corners[j]), corners[i].max(corners[j]));
                        if seen.insert((a, b)) {
                            let _ = writeln!(s, "{a}, {b}");
                        }
                    }
                }
            }
        }
    }
    let _ = writeln!(s, "cab");
    let face = |s: &mut String, a: usize, b: usize, c: usize, d: usize| {
        let _ = writeln!(s, "{a}, {b}, {c}, c\n{a}, {c}, {d}, c");
    };
    for y in 0..ny - 1 {
        for x in 0..nx - 1 {
            face(
                &mut s,
                id(x, y, 0),
                id(x, y + 1, 0),
                id(x + 1, y + 1, 0),
                id(x + 1, y, 0),
            );
            face(
                &mut s,
                id(x, y, nz - 1),
                id(x + 1, y, nz - 1),
                id(x + 1, y + 1, nz - 1),
                id(x, y + 1, nz - 1),
            );
        }
    }
    for z in 0..nz - 1 {
        for x in 0..nx - 1 {
            face(
                &mut s,
                id(x, 0, z),
                id(x + 1, 0, z),
                id(x + 1, 0, z + 1),
                id(x, 0, z + 1),
            );
            face(
                &mut s,
                id(x, ny - 1, z),
                id(x, ny - 1, z + 1),
                id(x + 1, ny - 1, z + 1),
                id(x + 1, ny - 1, z),
            );
        }
        for y in 0..ny - 1 {
            face(
                &mut s,
                id(0, y, z),
                id(0, y, z + 1),
                id(0, y + 1, z + 1),
                id(0, y + 1, z),
            );
            face(
                &mut s,
                id(nx - 1, y, z),
                id(nx - 1, y + 1, z),
                id(nx - 1, y + 1, z + 1),
                id(nx - 1, y, z + 1),
            );
        }
    }
    let _ = writeln!(s, "end");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::truck::{load_truck, parse_truck};

    #[test]
    fn car_imports_cleanly() {
        let src = car_truck(&CarParams::default());
        let f = parse_truck(&src);
        let bad: Vec<_> = f
            .warnings
            .iter()
            .filter(|w| w.contains("unknown") || w.contains("unrecognised"))
            .collect();
        assert!(bad.is_empty(), "{bad:?}");
        let d = load_truck(&src);
        assert_eq!(d.wheels.len(), 4);
        assert!(
            d.nodes.len() > 90 && d.beams.len() > 500,
            "{} nodes {} beams",
            d.nodes.len(),
            d.beams.len()
        );
        // Every skin triangle faces outward (away from the body's centre line).
        let p = |i: u32| glam::Vec3::from_array(d.nodes[i as usize].position);
        for t in &d.triangles {
            let (a, b, c) = (p(t.nodes[0]), p(t.nodes[1]), p(t.nodes[2]));
            let n = (b - a).cross(c - a);
            let centre = glam::Vec3::new(0.0, 0.75, (a.z + b.z + c.z) / 3.0);
            let out = (a + b + c) / 3.0 - centre;
            let end = (a.z - b.z).abs() < 1e-3 && (b.z - c.z).abs() < 1e-3;
            let out = if end {
                glam::Vec3::new(0.0, 0.0, a.z.signum())
            } else {
                out
            };
            assert!(n.dot(out) > 0.0, "inward-facing triangle {:?}", t.nodes);
        }
        assert!(d.triangles.iter().all(|t| t.collision));
        assert!(d.engine.is_some());
        // The bumper's mounts share a break group; its own bracing does not.
        assert!(d.beams.iter().filter(|b| b.break_group == 1).count() >= 10);
        let bumper = d.nodes.len() as u32 - 6;
        assert!(d
            .beams
            .iter()
            .filter(|b| b.a >= bumper && b.b >= bumper)
            .all(|b| b.break_group == 0));
        let m = d.total_mass();
        assert!(m > 1000.0 && m < 1600.0, "total mass {m}");
        let fwd = crate::pack::forward_axis(&d);
        assert!(fwd.z > 0.99, "forward {fwd:?}");
    }

    #[test]
    fn box_imports() {
        let d = load_truck(&box_truck(3, 3, 3, 0.5, 100.0, 1.0e6, 1.0e5, 1.0e6));
        assert_eq!(d.nodes.len(), 27);
        assert!(d.beams.len() > 100);
    }
}
