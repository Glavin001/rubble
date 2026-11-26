//! The collision skin's lint rules (`lint::collision_skin`).

use super::*;

/// A hole in the collision skin shorter than this round (m) is a defect, not an opening:
/// past the corners over a lamp (2.2 m) the generated sedan's skin had; short of a door's
/// or a lid's outline (3.0–4.5 m) and the underside, which are openings by design.
/// **Choice** (known-bugs S4).
pub const SKIN_HOLE_MAX: f32 = 2.5;
/// A collision triangle whose height is under this share of its longest edge is a sliver
/// (it lies across its neighbours; `gen2` rig's own rule).
pub const SKIN_SLIVER: f32 = 0.1;

/// Whether segment p→q passes through triangle (a, b, c) (Möller–Trumbore, ends included).
fn seg_through(p: Vec3, q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    // Signed volumes in double precision (the orientation predicate): the segment's ends strictly
    // on opposite sides of the triangle's plane, and its line strictly inside the triangle's three
    // edges. A segment touching or lying in the plane does not cross it. (The ray test with an
    // absolute determinant cut-off read a segment nearly in the plane as crossing.)
    let o = |a: Vec3, b: Vec3, c: Vec3, d: Vec3| {
        let (a, b, c, d) = (a.as_dvec3(), b.as_dvec3(), c.as_dvec3(), d.as_dvec3());
        (b - a).cross(c - a).dot(d - a)
    };
    let (sp, sq) = (o(a, b, c, p), o(a, b, c, q));
    if !(sp > 0.0 && sq < 0.0 || sp < 0.0 && sq > 0.0) {
        return false;
    }
    let (s1, s2, s3) = (o(p, q, a, b), o(p, q, b, c), o(p, q, c, a));
    s1 > 0.0 && s2 > 0.0 && s3 > 0.0 || s1 < 0.0 && s2 < 0.0 && s3 < 0.0
}

/// The collision triangles (not anti-clip-only) with their positions, names, outer flags
/// and edges (the triangles on each, and the way each walks it).
struct SkinView {
    pos: Vec<Vec3>,
    names: Vec<String>,
    tris: Vec<[u32; 3]>,
    outer: Vec<bool>,
    by: BTreeMap<(u32, u32), Vec<(usize, bool)>>,
}

fn skin_finding(rule: &'static str, hard: bool, bad: usize, warn_only: bool, value: String, band: &str, offenders: Vec<String>) -> Finding {
    Finding {
        rule,
        hard,
        part: "car".into(),
        level: if bad == 0 {
            Level::Pass
        } else if warn_only {
            Level::Warn
        } else {
            Level::Fail
        },
        value,
        band: band.into(),
        offenders: offenders.into_iter().take(if std::env::var("LINT_ALL").is_ok() { usize::MAX } else { 8 }).collect(),
    }
}

impl SkinView {
    fn new(d: &SoftVehicleDesc) -> Self {
        let pos: Vec<Vec3> = d.nodes.iter().map(|n| Vec3::from_array(n.position)).collect();
        let names = d.nodes.iter().map(|n| n.name.clone()).collect();
        let tris: Vec<[u32; 3]> = d.triangles.iter().filter(|t| t.collision).map(|t| t.nodes).collect();
        // The outer skin: the triangles with air on them (`dragCoef` > 0); the others are
        // faces inside the car (a floor pan, a seat's solid), whose facing is their own.
        let outer: Vec<bool> = d.triangles.iter().filter(|t| t.collision).map(|t| t.aero[0] > 0.0).collect();
        let mut by: BTreeMap<(u32, u32), Vec<(usize, bool)>> = BTreeMap::new();
        for (k, t) in tris.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                by.entry((a.min(b), a.max(b))).or_default().push((k, a < b));
            }
        }
        SkinView { pos, names, tris, outer, by }
    }
    fn name(&self, i: u32) -> String {
        self.names.get(i as usize).cloned().unwrap_or(format!("#{i}"))
    }
    fn winding(&self, out: &mut Vec<Finding>) {
    for (rule, hard, want_outer) in [("skin_winding", true, true), ("skin_winding_inner", false, false)] {
        let mine = |v: &Vec<(usize, bool)>| v.iter().all(|&(k, _)| self.outer[k] == want_outer);
        let same: Vec<(u32, u32)> = self.by
            .iter()
            .filter(|(_, v)| v.len() == 2 && mine(v) && v[0].1 == v[1].1)
            .map(|(&e, _)| e)
            .collect();
        let shared = self.by.values().filter(|v| v.len() == 2 && mine(v)).count();
        out.push(skin_finding(
            rule,
            hard,
            same.len(),
            !hard,
            format!(
                "{} of {} shared {} edges walked the same way by both triangles",
                same.len(),
                shared,
                if want_outer { "outer-skin" } else { "inner-face" }
            ),
            if hard { "0" } else { "0 (warn)" },
            same.iter().map(|&(a, b)| format!("{}-{}", self.name(a), self.name(b))).collect(),
        ));
    }
    let many: Vec<(u32, u32)> = self.by.iter().filter(|(_, v)| v.len() > 2).map(|(&e, _)| e).collect();
    out.push(skin_finding(
        "skin_manifold",
        false,
        many.len(),
        true,
        format!("{} edges on more than two triangles", many.len()),
        "0 (warn)",
        many.iter().map(|&(a, b)| format!("{}-{}", self.name(a), self.name(b))).collect(),
    ));
    }
    fn facing(&self, world: &[[Vec3; 3]], out: &mut Vec<Finding>) {
    // Facing: a ray from each triangle's centre along its normal crosses the rest of the
    // skin an even number of times when it faces out (it leaves the car), odd when in (it
    // crosses the far side first). (A winding number at the face itself is shaped by the
    // face, and cannot judge its own direction.) A slight skew keeps the ray off edges.
    let mut inward: Vec<String> = Vec::new();
    for (k, t) in world.iter().enumerate() {
        if !self.outer[k] {
            continue;
        }
        // A sliver's normal is not a direction (it closes a slit; `skin_slivers` lists it).
        let l = (t[0] - t[1]).length().max((t[1] - t[2]).length()).max((t[2] - t[0]).length());
        if l <= 0.0 || (t[1] - t[0]).cross(t[2] - t[0]).length() / l < SKIN_SLIVER * l {
            continue;
        }
        let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        if n == Vec3::ZERO {
            continue;
        }
        let c = (t[0] + t[1] + t[2]) / 3.0;
        // Five rays round the normal, the majority's parity: a protrusion (a mirror) a
        // single ray clips does not turn a face over.
        let u = n.any_orthonormal_vector();
        let v = n.cross(u);
        let odd = [Vec3::ZERO, u * 0.5, -u * 0.5, v * 0.5, -v * 0.5]
            .iter()
            .filter(|&&off| {
                let dir = (n + off + Vec3::new(0.0123, 0.0071, 0.0037)).normalize();
                let (p, q) = (c + dir * 1e-4, c + dir * 50.0);
                let hits = world
                    .iter()
                    .enumerate()
                    .filter(|&(j, w)| j != k && self.outer[j] && seg_through(p, q, w[0], w[1], w[2]))
                    .count();
                hits % 2 == 1
            })
            .count();
        if odd >= 3 {
            // Turned over, or under a second layer of skin: the nearest outer triangle
            // straight along the normal within 5 cm says which.
            let near = world.iter().enumerate().any(|(j, w)| {
                j != k && self.outer[j] && seg_through(c + n * 1e-4, c + n * 0.05, w[0], w[1], w[2])
            });
            let tag = if near { "under another layer" } else { "faces in" };
            inward.push(format!("{} ({tag})", self.tris[k].map(|i| self.name(i)).join(",")));
        }
    }
    out.push(skin_finding(
        "skin_facing",
        true,
        inward.len(),
        false,
        format!(
            "{} of {} outer-skin triangles face into the car ({} of them under another layer of skin within 5 cm: a doubled layer)",
            inward.len(),
            self.outer.iter().filter(|&&o| o).count(),
            inward.iter().filter(|x| x.ends_with("(under another layer)")).count()
        ),
        "0",
        inward,
    ));
    }
    fn crossing(&self, world: &[[Vec3; 3]], out: &mut Vec<Finding>) {
    // The skin through itself: edges through triangles they share no node with.
    let mut crossing: Vec<String> = Vec::new();
    let boxes: Vec<(Vec3, Vec3)> = world.iter().map(|t| (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]))).collect();
    for &(a, b) in self.by.keys() {
        let (p, q) = (self.pos[a as usize], self.pos[b as usize]);
        let (lo, hi) = (p.min(q), p.max(q));
        for (k, t) in self.tris.iter().enumerate() {
            if t.contains(&a) || t.contains(&b) {
                continue;
            }
            let (tl, th) = boxes[k];
            if hi.cmplt(tl).any() || lo.cmpgt(th).any() {
                continue;
            }
            let [x, y, z] = world[k];
            if seg_through(p, q, x, y, z) {
                crossing.push(format!("{}-{} through {}", self.name(a), self.name(b), t.map(|i| self.name(i)).join(",")));
            }
        }
    }
    // By the parts involved (node names' prefixes): where to look.
    let prefix = |n: &str| n.split('_').next().unwrap_or(n).trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
    let mut by_parts: BTreeMap<String, usize> = BTreeMap::new();
    for c in &crossing {
        if let Some((e, t)) = c.split_once(" through ") {
            let ep: Vec<String> = e.split('-').map(prefix).collect();
            let tp: Vec<String> = t.split(',').map(prefix).collect();
            let mut ep = ep;
            ep.sort();
            ep.dedup();
            let mut tp = tp;
            tp.sort();
            tp.dedup();
            *by_parts.entry(format!("{} through {}", ep.join("+"), tp.join("+"))).or_default() += 1;
        }
    }
    let mut by_parts: Vec<(String, usize)> = by_parts.into_iter().collect();
    by_parts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out.push(skin_finding(
        "skin_self_crossing",
        true,
        crossing.len(),
        false,
        format!(
            "{} skin edges through the skin's own triangles at rest ({})",
            crossing.len(),
            by_parts.iter().map(|(k, n)| format!("{k}: {n}")).collect::<Vec<_>>().join(", ")
        ),
        "0",
        crossing,
    ));
    }
    fn holes(&self, out: &mut Vec<Finding>) {
    // Holes: loops of open edges (components whose every node has two open edges).
    let open: Vec<(u32, u32)> = self.by.iter().filter(|(_, v)| v.len() == 1).map(|(&e, _)| e).collect();
    let mut nb: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &(a, b) in &open {
        nb.entry(a).or_default().push(b);
        nb.entry(b).or_default().push(a);
    }
    let mut seen: HashSet<u32> = HashSet::new();
    let (mut holes, mut openings): (Vec<(f32, String)>, Vec<(f32, String)>) = (Vec::new(), Vec::new());
    for &s0 in nb.keys() {
        if seen.contains(&s0) {
            continue;
        }
        let mut comp = vec![s0];
        let mut stack = vec![s0];
        seen.insert(s0);
        while let Some(x) = stack.pop() {
            for &y in &nb[&x] {
                if seen.insert(y) {
                    comp.push(y);
                    stack.push(y);
                }
            }
        }
        let set: HashSet<u32> = comp.iter().copied().collect();
        let len: f32 = open
            .iter()
            .filter(|(a, b)| set.contains(a) && set.contains(b))
            .map(|&(a, b)| self.pos[a as usize].distance(self.pos[b as usize]))
            .sum();
        let c = comp.iter().map(|&i| self.pos[i as usize]).sum::<Vec3>() / comp.len() as f32;
        let mut who: Vec<String> = comp.iter().take(6).map(|&i| self.name(i)).collect();
        who.sort();
        let label = format!("{:.2} m round at ({:.2}, {:.2}, {:.2}): {}", len, c.x, c.y, c.z, who.join(","));
        // A loop whose nodes are all one door's or the lid's is that panel's own outline or
        // window (glass is visual only): an opening by design, whatever its length.
        let panel = |i: u32| {
            let n = &self.names[i as usize];
            let mut it = n.split('_');
            let (a, b) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
            match a {
                "door" => Some(format!("door_{b}")),
                "tk" => Some("tk".to_string()),
                _ => None,
            }
        };
        let own = panel(comp[0]).is_some_and(|p| comp.iter().all(|&i| panel(i).as_deref() == Some(p.as_str())));
        if len < SKIN_HOLE_MAX && !own {
            holes.push((len, label));
        } else {
            openings.push((len, label));
        }
    }
    holes.sort_by(|a, b| b.0.total_cmp(&a.0));
    openings.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.push(skin_finding(
        "skin_holes",
        true,
        holes.len(),
        false,
        format!(
            "{} holes under {SKIN_HOLE_MAX} m round ({} open edges; {} longer openings: {})",
            holes.len(),
            open.len(),
            openings.len(),
            openings.iter().take(4).map(|x| format!("{:.1} m", x.0)).collect::<Vec<_>>().join(", ")
        ),
        "0",
        holes.into_iter().map(|x| x.1).collect(),
    ));
    }
    fn slivers(&self, world: &[[Vec3; 3]], out: &mut Vec<Finding>) {
    let slivers: Vec<String> = world
        .iter()
        .zip(&self.tris)
        .filter(|(t, _)| {
            let l = (t[0] - t[1]).length().max((t[1] - t[2]).length()).max((t[2] - t[0]).length());
            let area2 = (t[1] - t[0]).cross(t[2] - t[0]).length();
            l > 0.0 && area2 / l < SKIN_SLIVER * l
        })
        .map(|(_, t)| t.map(|i| self.name(i)).join(","))
        .collect();
    out.push(skin_finding(
        "skin_slivers",
        false,
        slivers.len(),
        true,
        format!("{} slivers of {} triangles", slivers.len(), self.tris.len()),
        "0 (warn)",
        slivers,
    ));
    }
}

/// The collision skin's topology and placement at rest (known-bugs S4: holes over the
/// lamps, doubled layers at seams and a skin through itself were found by hand in a crash
/// study; these rules find them at build time). Over the collision triangles (`collision`;
/// anti-clip-only ones are not skin):
/// * `skin_winding` (hard): an edge two triangles share is walked the opposite way by each
///   (one surface on both sides of it); the same way means a face turned over or a second
///   layer laid on the first, and contact's side there is wrong;
/// * `skin_facing` (hard): every outer-skin triangle faces out of the car (rays from its
///   centre round its normal cross the rest of the outer skin an even number of times, self.by
///   majority of five); a face that points out but lies under a second layer of skin
///   within 5 cm fails too, tagged so (a doubled layer: contact's side there is ambiguous);
///   slivers are left to `skin_slivers` (their normal is not a direction);
/// * `skin_self_crossing` (hard): no skin edge passes through a triangle it shares no node
///   with (the skin through itself at rest: its two layers have no outside between them);
/// * `skin_holes` (hard): no hole (a loop of open edges) shorter than [`SKIN_HOLE_MAX`]
///   round; longer loops, and a door's or the lid's own outline or window (glass is
///   visual only), are listed as openings;
/// * `skin_manifold` (warn): no edge on more than two triangles;
/// * `skin_slivers` (warn): no triangle under [`SKIN_SLIVER`] of its longest edge high.
pub fn collision_skin(d: &SoftVehicleDesc) -> Vec<Finding> {
    let s = SkinView::new(d);
    let world: Vec<[Vec3; 3]> = s.tris.iter().map(|t| t.map(|i| s.pos[i as usize])).collect();
    let mut out = Vec::new();
    s.winding(&mut out);
    s.facing(&world, &mut out);
    s.crossing(&world, &mut out);
    s.holes(&mut out);
    s.slivers(&world, &mut out);
    out
}

/// Point to triangle distance.
pub(crate) fn tri_dist(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Ericson, Real-Time Collision Detection §5.1.5 (closest point).
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return p.distance(a);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return p.distance(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return p.distance(a + ab * v);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return p.distance(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return p.distance(a + ac * w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return p.distance(b + (c - b) * w);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    p.distance(a + ab * v + ac * w)
}
