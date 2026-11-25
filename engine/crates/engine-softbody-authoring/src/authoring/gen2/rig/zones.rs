//! The rigger's zone constants (yields, stacks, seals, hardware masses) and the helpers of its link and skin passes (split from `rig.rs`).

use super::*;

/// The yield of the crush zones' body members (rails, cans, aprons, rear
/// rails and their webs) as a multiple of the census's 1.15 %: 0.32 %, in
/// the census's body band (p10 0.2-0.3 %, p50 1.0-1.3 %). Measured on the
/// wall (surface contact on), round 4's structure: dynamic crush 581 mm at
/// 0.4, 619 at 0.3, 631 at 0.28 (620-750); t_m 66 / 71 / 72 ms (63-76);
/// b1 42.1 / 39.8 / 39.2 (<= 39.2).
/// Round 5 (push links, stacks, the hinge pillar's joints at the towers'
/// yield): 620 mm at 0.25, 625 at 0.24, 629 at 0.23; b1 39.4 / 38.6 / 38.6;
/// restitution 0.201 / 0.207 / 0.202 (each ±0.005 run to run of the yield).
pub const CRUSH_YIELD: f32 = 0.23;
/// The suspension towers' beams in the crush zone (see the zone pass).
pub const TOWER_YIELD: f32 = 0.4;
/// The footwell's (floor, sills, hinge pillars and firewall ahead of the
/// front seats, below the belt): 4.1 %, under the census's body p90
/// (4.9-9.3 %): high-strength steel there. Feet intrusion 94 to 53 mm (<= 67).
pub const FOOTWELL_YIELD: f32 = 3.0;
/// The greenhouse's (above the belt, firewall to the rear shelf): 0.52 %.
/// FMVSS 216a 5.94 to 5.02 x weight at 0.45 in round 3 (4.48-5.32); with
/// round 4's hot-stamped A-pillar and B-pillar upper, 5.57 at 0.45, 5.32 at
/// 0.38.
pub const GREENHOUSE_YIELD: f32 = 0.38;
/// The A-pillar chords' yield over the greenhouse's: a hot-stamped section
/// (boron steel, about 1,000-1,200 MPa against 300-400 MPa for the HSLA the
/// greenhouse's yield stands for).
pub const APILLAR_YIELD: f32 = 3.0;
/// The floor's cross members ahead of and behind the B-pillar (m): the
/// footwell's rear (the front seat's front mounts; the old evenly spaced
/// station, which the frontal's feet intrusion needs) and the front seat's
/// rear mounts.
pub const SEAT_CM_F: f32 = 0.53;
pub const SEAT_CM_R: f32 = 0.35;
/// The floor cross members' yield over the census's (dash, seat front, seat
/// rear, kick-up): high-strength sections a wide side load reaches through
/// the sill (`GEN3_CM_Y`).
pub const CROSS_MEMBER_YIELD: f32 = 3.0;
/// The front crush zone's yield grade, nose and firewall, on its yield
/// (1, 1: ungraded; calibrated in round 5 on the pulse's bands).
pub const CRUSH_GRADE: [f32; 2] = [1.0, 1.0];
/// The floor pan's (the floor between the cross members and its diagonals):
/// 0.7 mm sheet, which buckles in plane (`GEN3_PAN_Y`).
pub const FLOOR_PAN_YIELD: f32 = 0.3;
/// The B-pillar's upper section over the greenhouse's yield (hot-stamped)
/// and its lower section over the side's (the tailored blank's soft zone):
/// `GEN3_BP_UP`, `GEN3_BP_LO`.
pub const BPILLAR_UPPER_YIELD: f32 = 3.0;
pub const BPILLAR_LOWER_YIELD: f32 = 0.5;
/// How far the intrusion beam's outer chord sits under the door skin (m).
pub const INTRUSION_OUT: f32 = 0.02;
/// The door intrusion beams' yield over their role's.
pub const INTRUSION_YIELD: f32 = 1.0;

/// Spacing of the shut lines' edge nodes (m): a real closure's flange is
/// bolted, hinged or stopped every 15-25 cm along it.
pub const EDGE_SPACING: f32 = 0.22;
/// The front crush zone's beams across the car's axis: their yield against
/// the beams along it.
pub const CROSS_SECTION_YIELD: f32 = 1.0;
/// The bumpers' damping ratio (plastic covers on foam: the census's bumper
/// beams run at ζ 0.07-0.33, the ETK's at 0.16).
pub const BUMPER_ZETA: f32 = 0.16;
/// The rails' anchorage on the cabin floor: its yield (× the census 1.15 %).
pub const KICKDOWN_YIELD: f32 = 1.0;
/// The bumper beam's centre height (m, model): the cover's middle row.
pub const BUMPER_BEAM_Y: f32 = 0.48;
/// The front's upper band starts here (m, model) and yields at this share
/// of the crush zone's yield.
pub const UPPER_BAND_Y: f32 = 0.62;
pub const UPPER_YIELD: f32 = 0.7;
/// The bumper foam's solid thickness (m): a cover node's stop on the beam.
pub const FOAM_SOLID: f32 = 0.03;
/// The hood's yield against the census sheet's (it buckles in a frontal).
pub const HOOD_YIELD: f32 = 0.4;
/// The front rails' and crush cans' yield against the crush zone's.
pub const RAIL_YIELD: f32 = 4.0;
/// The body's shut-line edge ties ahead of the strut towers: their yield
/// against the sheet's.
pub const EDGE_TIE_YIELD: f32 = 0.35;
/// A door's window frame: its yield against the door frame's.
pub const WINDOW_FRAME_YIELD: f32 = 0.3;
/// The apron's yield from the strut tower back (× the census 1.15 %).
pub const APRON_REAR_YIELD: f32 = 1.0;
/// Seals and stops: body members a closure node bears on, their reach (m)
/// and the preload of a precompressed one (N).
pub const SEAL_K: usize = 2;
pub const SEAL_REACH: f32 = 0.30;
pub const SEAL_PRELOAD: f32 = 150.0;
/// The census door's depth: its inner panel behind its skin (m).
pub const DOOR_DEPTH: f32 = 0.10;
/// How far an edge node sits in from its panel's edge, and under its skin (m).
pub const EDGE_IN: f32 = 0.025;
pub const EDGE_UNDER: f32 = 0.01;

/// A mesh's boundary vertices (positions welded, so a split vertex counts
/// once): one vertex index per welded boundary point.
pub(super) fn boundary_points(p: &super::super::model::Part) -> Vec<usize> {
    let key = |q: Vec3| {
        (
            (q.x * 2e4).round() as i64,
            (q.y * 2e4).round() as i64,
            (q.z * 2e4).round() as i64,
        )
    };
    let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let wid: Vec<usize> = p
        .pos
        .iter()
        .map(|&q| {
            let l = weld.len();
            *weld.entry(key(q)).or_insert(l)
        })
        .collect();
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for t in &p.tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (wa, wb) = (wid[a as usize], wid[b as usize]);
            if wa != wb {
                *edges.entry((wa.min(wb), wa.max(wb))).or_insert(0) += 1;
            }
        }
    }
    let mut on = vec![false; weld.len()];
    for (&(a, b), &n) in &edges {
        if n == 1 {
            on[a] = true;
            on[b] = true;
        }
    }
    let mut seen = vec![false; weld.len()];
    (0..p.pos.len())
        .filter(|&i| {
            let w = wid[i];
            let keep = on[w] && !seen[w];
            seen[w] = true;
            keep
        })
        .collect()
}

/// Shut-line edge nodes: where two parts' render meshes meet (boundary
/// vertices of different parts within 1 cm, as the harness finds seams), a
/// structural node on each side every [`EDGE_SPACING`], [`EDGE_IN`] in from
/// its panel's edge and [`EDGE_UNDER`] under its skin, each tied into its
/// own part's frame (its nearest structural nodes, an inner one among them)
/// and to its neighbours along the edge, and a SUPPORT stop to the node
/// across the line. They are in their part's binding pool, so a border
/// vertex rides nodes within about 10 cm instead of extrapolating past the
/// part's lattice; the stops carry a closure's edge onto its neighbour as a
/// flange does. Returns the edge nodes.
/// The push links (see the call): (front part, part behind) pairs.
pub(super) fn push_links(r: &mut Rig, census: bool) {
    let pairs = [
        (Part::BumperF, Part::Body),
        (Part::BumperF, Part::FenderL),
        (Part::BumperF, Part::FenderR),
        (Part::BumperF, Part::Hood),
        (Part::FenderL, Part::Body),
        (Part::FenderR, Part::Body),
        (Part::Hood, Part::Body),
    ];
    let axle = r.axle_f;
    let reach = env_f("GEN3_PUSH_REACH", PUSH_REACH);
    let cone = (35f32).to_radians().tan();
    for (pa, pb) in pairs {
        // Census: the cover does not push the hood (an unyielding stop from
        // the cover's top row into the hood's front edge put the head-on
        // load through the hood, above the centre of mass: known-bugs S2).
        if census && pa == Part::BumperF && pb == Part::Hood {
            continue;
        }
        let fronts: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| {
                let n = &r.car.nodes[i];
                n.part == pa && !n.contact && r.g(i).z > axle
            })
            .collect();
        for a in fronts {
            let ga = r.g(a);
            let best = (0..r.car.nodes.len())
                .filter(|&j| {
                    let n = &r.car.nodes[j];
                    n.part == pb
                        && !n.contact
                        && !n.name.starts_with("hub")
                        && !n.name.starts_with("rim")
                })
                .filter_map(|j| {
                    let d = r.g(j) - ga;
                    let back = -d.z;
                    let side = (d.x * d.x + d.y * d.y).sqrt();
                    (back >= PUSH_MIN && back <= reach && side <= cone * back)
                        .then_some((d.length(), j))
                })
                .min_by(|x, y| x.0.total_cmp(&y.0));
            if let Some((_, b)) = best {
                if !r.car.has_beam(a, b) {
                    let part = if pa == Part::Body { pb } else { pa };
                    let j = r.car.beam(a, b, Role::Bearing, part);
                    r.car.beams[j].role = Role::Bearing;
                }
            }
        }
    }
}

/// Push links reach 4 cm to [`PUSH_REACH`] behind a node (shorter ones are
/// the shut line's own edge stops).
pub const PUSH_MIN: f32 = 0.04;
pub const PUSH_REACH: f32 = 0.30;

/// A latch's mass on its closure (kg): the latch or striker, its
/// reinforcement in the panel's wall and what is bolted beside it (a door's
/// handle carrier and lock; a lid's lock barrel). A door latch alone weighs
/// about 0.9 kg.
pub const DOOR_LATCH_KG: f32 = 2.5;
pub const HOOD_LATCH_KG: f32 = 1.0;
pub const TRUNK_LATCH_KG: f32 = 1.5;
/// The body's side of a latch (kg): the striker or lock and the reinforcement
/// it bolts through, mass that belongs to the member it stands on and is
/// taken from that member's nodes.
pub const STRIKE_KG: f32 = 2.5;

/// Gives a latch node and its striker node their hardware's mass. The
/// solver holds a coupled pair with a spring sized by the pair's reduced mass
/// (`explicit_beams::coupler_rates`: k = 0.16 mu / h^2), so two 0.5 kg tag
/// nodes are a 2e5 N/m latch: more than half of each door's first mode's
/// strain energy sat in it (18 Hz), a third of the lid's. The striker's mass
/// is the member's own at that place: it comes out of the nodes the striker
/// is tied to, equally, so the body's mass and its distribution stay.
pub(super) fn hardware_mass(r: &mut Rig, latch: usize, strike: usize, latch_kg: f32, ties: &[usize]) {
    r.car.nodes[latch].fixed_mass = latch_kg;
    let givers: Vec<usize> = ties
        .iter()
        .copied()
        .filter(|&j| r.car.nodes[j].fixed_mass > 2.0 * STRIKE_KG / ties.len() as f32)
        .collect();
    if givers.is_empty() {
        return;
    }
    let each = STRIKE_KG / givers.len() as f32;
    for j in givers {
        r.car.nodes[j].fixed_mass -= each;
    }
    r.car.nodes[strike].fixed_mass = STRIKE_KG;
}

/// How far a hinge's body nodes may be from its marker (m): the nearest
/// structural nodes of the shell (spacing 0.2–0.45 m) that still span three
/// directions. Farther ones made long hinge beams that, at the same spring,
/// reach their strength first (`engine/docs/measurements/hinge-breaks.md`).
pub const HINGE_REACH: f32 = 0.6;

/// The shortest hinge beam (m): the solver keeps a JBeam beam's rest length at
/// 0.1 m or more (`explicit_beams::MIN_REST`), so a shorter one cannot yield
/// in compression and breaks on its strength instead.
pub const HINGE_MIN_LEN: f32 = 0.12;

pub(super) fn hinge_long_enough(r: &Rig, a: usize, b: usize) -> bool {
    r.g(a).distance(r.g(b)) >= HINGE_MIN_LEN
}

/// A crush-zone member (front and rear rails, crush cans, bumper beam, aprons).
pub fn crush_member(m: Option<Role>) -> bool {
    matches!(
        m,
        Some(Role::Rail | Role::CrushCan | Role::BumperBeam | Role::Apron | Role::RearRail)
    )
}

/// How far a mount's beams may reach for a body node that gives them a
/// direction they lack ([`spread_pick`]).
pub const MOUNT_SPREAD_REACH: f32 = 0.75;
/// A rigidifier's depth behind its panel's skin (the study's rule 20:
/// 0.2-0.4 m), and how far its beams reach.
pub const RIGIDIFIER_DEPTH: f32 = 0.22;
pub const RIGIDIFIER_REACH: f32 = 0.95;

/// `n` body nodes for a mount at `at`: the nearest of `cands`, then each time
/// the candidate (within [`MOUNT_SPREAD_REACH`]) that most raises the
/// weakest direction of the beams so far, so the mount's beams span all
/// three axes. A hinge leaf bolted to a pillar holds in every direction;
/// the nearest nodes alone can all lie in the pillar's plane (the front
/// doors slid fore and aft on theirs at 12.6 Hz).
pub(super) fn spread_pick(r: &Rig, at: Vec3, cands: &[usize], n: usize) -> Vec<usize> {
    let mut v: Vec<(f32, usize)> = cands
        .iter()
        .map(|&i| (r.g(i).distance(at), i))
        .filter(|x| x.0 > 0.02)
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<usize> = v.iter().take(1).map(|x| x.1).collect();
    let second = |d: Vec3| glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z);
    while out.len() < n {
        let m = out.iter().fold(glam::Mat3::ZERO, |m, &j| {
            m + second((r.g(j) - at).normalize_or_zero())
        });
        let best = v
            .iter()
            .filter(|x| x.0 < MOUNT_SPREAD_REACH && !out.contains(&x.1))
            .map(|&(d, j)| {
                let mm = m + second((r.g(j) - at).normalize_or_zero());
                // Two beams leave one axis empty whatever they are: judge
                // them by their second-weakest direction.
                let lam = if out.len() == 1 {
                    let d0 = (r.g(out[0]) - at).normalize_or_zero();
                    1.0 - d0.dot((r.g(j) - at).normalize_or_zero()).abs()
                } else {
                    weakest_axis(mm).0
                };
                (lam - 0.15 * d, j)
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            Some((_, j)) => out.push(j),
            None => break,
        }
    }
    out
}

/// A panel's rigidifier ([`RIGIDIFIER_DEPTH`] under its skin at `at`, a point
/// of its inner frame): beamed to the panel's nodes within
/// [`RIGIDIFIER_REACH`], so the panel bends as a section of that depth and
/// not as its skin alone (bending stiffness goes as k d²; the study's §5.1
/// and rule 20). Not a collision node.
pub(super) fn rigidifier(r: &mut Rig, name: &str, at: Vec3, part: Part, own: &[usize]) -> usize {
    let i = r.node(name, at, part, Some(Role::DoorFrame), false);
    for &j in own {
        let d = r.g(j).distance(at);
        if d > 0.03 && d < RIGIDIFIER_REACH && !r.car.has_beam(i, j) {
            r.car.beam(i, j, Role::DoorFrame, part);
        }
    }
    i
}

/// The parts whose own solid is a collision skin: the convex hull of their
/// nodes ([`interior_skins`]).
pub const HULL_PARTS: [&str; 8] = [
    "seat_FL",
    "seat_FR",
    "seat_rear",
    "dash",
    "console",
    "engine",
    "fuel_tank",
    "radiator",
];

/// The convex hull of `p` as polygons (indices into `p`, in order around
/// each face): every plane through three points with all the others on one
/// side, its coplanar points taken together. A flat set is one polygon.
pub(super) fn hull_faces(p: &[Vec3]) -> Vec<Vec<usize>> {
    let eps = 2e-3;
    let mut seen: std::collections::BTreeSet<Vec<usize>> = Default::default();
    let mut out = Vec::new();
    for a in 0..p.len() {
        for b in a + 1..p.len() {
            for c in b + 1..p.len() {
                let n = (p[b] - p[a]).cross(p[c] - p[a]);
                if n.length() < 1e-5 {
                    continue;
                }
                let n = n.normalize();
                let d: Vec<f32> = p.iter().map(|q| n.dot(*q - p[a])).collect();
                if !(d.iter().all(|&x| x <= eps) || d.iter().all(|&x| x >= -eps)) {
                    continue;
                }
                let mut on: Vec<usize> = (0..p.len()).filter(|&i| d[i].abs() <= eps).collect();
                if !seen.insert(on.clone()) {
                    continue;
                }
                let cen = on.iter().map(|&i| p[i]).sum::<Vec3>() / on.len() as f32;
                let u = (p[on[0]] - cen).normalize_or_zero();
                let v = n.cross(u);
                on.sort_by(|&i, &j| {
                    let f = |k: usize| (p[k] - cen).dot(v).atan2((p[k] - cen).dot(u));
                    f(i).total_cmp(&f(j))
                });
                out.push(on);
            }
        }
    }
    out
}

/// The collision faces inside the defined car. Each is a triangle on its
/// part's own nodes, marked [`Tri::inner`] (no air on it); its nodes become
/// contacters with the rest of the skin's.
///
/// - **The cabin's structure** (the shell): each sill's outer face (bottom
///   to top), the floor pan, the tunnel's walls, and the firewall from the
///   cowl to the toe board.
/// - **Each door's box:** its inner panel, its bottom, its belt and its two
///   end walls, so a door is a solid on both faces.
/// - **Interior and engine-bay parts** ([`HULL_PARTS`]): the convex hull of
///   the part's nodes, the solid they are the corners of.
///
/// Before this only the outer skin had triangles: a door T-boned at 50 km/h
/// went through the sill's top edge and the A-pillar, and nothing in the
/// cabin could stop it or be pushed by it.
pub(super) fn interior_skins(r: &mut Rig) {
    let ix: HashMap<String, usize> = r
        .car
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.clone(), i))
        .collect();
    let mut have: std::collections::BTreeSet<[usize; 3]> = r
        .car
        .tris
        .iter()
        .map(|t| {
            let mut k = t.n;
            k.sort_unstable();
            k
        })
        .collect();
    let mut add: Vec<Tri> = Vec::new();
    let pos: Vec<Vec3> = (0..r.car.nodes.len()).map(|i| r.g(i)).collect();
    // The skin so far (the outer skin, then each face added), for the shell faces' choice
    // of diagonal below (the lint's skin_self_crossing). `RIG_INNER_CROSSING=1`: as before.
    let check = std::env::var("RIG_INNER_CROSSING").map_or(true, |v| v != "1");
    let mut placed: Vec<[usize; 3]> = r.car.tris.iter().map(|t| t.n).collect();
    let mut poly = |ids: &[usize], part: Part, sub: Option<&str>| {
        // A face's triangles (a fan); a shell face that would cross the skin so far tries
        // its other fans (a quad's other diagonal), and when every fan crosses it is kept
        // whole all the same (`RIG_INNER_CROSSING=1`: the first fan, unchecked): a sill's or
        // floor's face stops a door pushed in, which matters more than its crossing at rest
        // (the lint's skin_self_crossing reports what is left).
        let fan = |start: usize| -> Vec<[usize; 3]> {
            let m = ids.len();
            (1..m.saturating_sub(1)).map(|k| [ids[start], ids[(start + k) % m], ids[(start + k + 1) % m]]).collect()
        };
        let tris: Vec<[usize; 3]> = if check && sub.is_none() && ids.len() > 3 {
            (0..ids.len())
                .map(fan)
                .find(|f| !f.iter().any(|&n| tri_crosses(&pos, n, &placed)))
                .unwrap_or_else(|| fan(0))
        } else {
            fan(0)
        };
        for n in tris {
            let area = (pos[n[1]] - pos[n[0]]).cross(pos[n[2]] - pos[n[0]]).length();
            let mut key = n;
            key.sort_unstable();
            if area < 1e-5 || have.contains(&key) {
                continue;
            }
            have.insert(key);
            placed.push(n);
            add.push(Tri {
                n,
                part,
                group: None,
                sub: sub.map(str::to_string),
                inner: true,
            });
        }
    };
    let named = |names: &[String]| -> Option<Vec<usize>> {
        names.iter().map(|n| ix.get(n).copied()).collect()
    };
    // The parts that are solids of their own, first: a closed solid takes precedence over
    // the shell's faces it would cross (those duplicate the outer skin or meet the solid).
    for sub in HULL_PARTS {
        let ids: Vec<usize> = (0..r.car.nodes.len())
            .filter(|&i| r.car.nodes[i].sub.as_deref() == Some(sub))
            .collect();
        if ids.len() < 3 {
            continue;
        }
        let pts: Vec<Vec3> = ids.iter().map(|&i| pos[i]).collect();
        for face in hull_faces(&pts) {
            let f: Vec<usize> = face.iter().map(|&k| ids[k]).collect();
            poly(&f, Part::Engine, Some(sub));
        }
    }
    // The shell.
    for s in ["l", "r"] {
        for row in 1..=4 {
            let (a, b) = (row, row + 1);
            for q in [
                // the sill's outer face
                [format!("sh_c1_{a}o{s}"), format!("sh_c1_{b}o{s}"), format!("sh_c2_{b}{s}"), format!("sh_c2_{a}{s}")],
                // the floor pan
                [format!("sh_c1_{a}o{s}"), format!("sh_c1_{b}o{s}"), format!("sh_c1_{b}t{s}"), format!("sh_c1_{a}t{s}")],
                // the tunnel's wall
                [format!("sh_c1_{a}t{s}"), format!("sh_c1_{b}t{s}"), format!("sh_c3_{b}"), format!("sh_c3_{a}")],
            ] {
                named(&q).into_iter().for_each(|ids| poly(&ids, Part::Body, None));
            }
        }
        // The B-pillar, sill to waist to roof rail: without it doors crush past a pillar nothing can
        // touch, and its paint stands in the other car (side 60: 88 mm -> 0).
        for q in [["sh_c2_2", "sh_c2_3", "sh_d2_b"], ["sh_c2_3", "sh_c2_4", "sh_d2_b"], ["sh_d1_t", "sh_d2_t", "sh_d2_b"]] {
            named(&q.map(|n| format!("{n}{s}"))).into_iter().for_each(|ids| poly(&ids, Part::Body, None));
        }
    }
    let across = ["ol", "il", "ir", "or"];
    let floor1 = ["ol", "tl", "tr", "or"];
    for k in 0..3 {
        for q in [
            [format!("sh_b2_{}", across[k]), format!("sh_b2_{}", across[k + 1]), format!("sh_b1_{}", across[k + 1]), format!("sh_b1_{}", across[k])],
            [format!("sh_b1_{}", across[k]), format!("sh_b1_{}", across[k + 1]), format!("sh_c1_1{}", floor1[k + 1]), format!("sh_c1_1{}", floor1[k])],
        ] {
            named(&q).into_iter().for_each(|ids| poly(&ids, Part::Body, None));
        }
    }
    // The doors.
    for part in [Part::DoorFL, Part::DoorFR, Part::DoorRL, Part::DoorRR] {
        let dn = part.name().trim_start_matches("gen2_").to_string();
        let o = |row: usize, k: usize| format!("{dn}_o{row}{k}");
        let i = |row: usize, k: usize| format!("{dn}_i{row}{k}");
        for k in 0..2 {
            for q in [
                [i(0, k), i(0, k + 1), i(1, k + 1), i(1, k)],
                [o(0, k), o(0, k + 1), i(0, k + 1), i(0, k)],
                [o(2, k), o(2, k + 1), i(1, k + 1), i(1, k)],
            ] {
                named(&q).into_iter().for_each(|ids| poly(&ids, part, None));
            }
        }
        for k in [0, 2] {
            named(&[o(0, k), i(0, k), i(1, k), o(2, k)]).into_iter().for_each(|ids| poly(&ids, part, None));
        }
    }
    r.car.tris.extend(add);
    fill_holes(r);
    belly_cap(r);
}

/// The least share of an edge node's ties' directions along their weakest
/// axis (the smallest eigenvalue of the unit directions' second moment over
/// their number; a third is isotropic) for the node to be made.
pub const EDGE_SPREAD: f32 = 0.05;

/// The spread the ties of a body edge node at `g` could reach: its four
/// nearest structural nodes, then up to three more within 0.65 m, each the
/// one most along the weakest axis so far (as `edge_nodes` ties them).
pub(super) fn edge_tie_spread(r: &Rig, g: Vec3, edge: &[usize]) -> f32 {
    let cands: Vec<usize> = r
        .nearest(Part::Body, g, 24, |j| {
            !edge.contains(&j)
                && !r.car.nodes[j].no_group
                && !r.car.nodes[j].name.starts_with("hub")
                && !r.car.nodes[j].name.starts_with("rim")
        })
        .into_iter()
        .filter(|&j| r.g(j).distance(g) < 0.65 && r.g(j).distance(g) > 0.03)
        .collect();
    let mut tied: Vec<usize> = cands.iter().copied().take(4).collect();
    let moment = |tied: &[usize]| {
        tied.iter().fold(glam::Mat3::ZERO, |m, &j| {
            let d = (r.g(j) - g).normalize_or_zero();
            m + glam::Mat3::from_cols(d * d.x, d * d.y, d * d.z)
        })
    };
    for _ in 0..3 {
        let (lam, axis) = weakest_axis(moment(&tied));
        if lam > 0.12 * tied.len() as f32 {
            break;
        }
        let best = cands.iter().copied().filter(|j| !tied.contains(j)).max_by(|&a, &b| {
            let f = |j: usize| (r.g(j) - g).normalize_or_zero().dot(axis).abs();
            f(a).total_cmp(&f(b))
        });
        match best {
            Some(j) => tied.push(j),
            None => break,
        }
    }
    if tied.is_empty() {
        return 0.0;
    }
    weakest_axis(moment(&tied)).0 / tied.len() as f32
}
