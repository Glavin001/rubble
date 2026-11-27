//! Part outlines of a crash run, for comparing shapes with reference footage
//! (`engine/tools/crash-reference/outlines.py`, `engine/docs/crash-reference.md`).
//!
//! A vehicle's named part groups (hood, fenders, bumper, doors, roof, greenhouse, sills,
//! trunk, wheels) are its JBeam parts (`SoftVehicleDesc::parts`, the generator's panels) and,
//! for the body shell, regions of its rest geometry. Each group's outline is the boundary of
//! its collision triangles (edges used by one triangle of the group), chained into polylines
//! of node indices, so the same chain is drawn on the current state, on the "ghost" (the
//! rest shape carried rigidly with the cabin and rear: [`Ghost`]) and in the car's own side
//! frame ([`CarFrame`]). A group without collision triangles is drawn as the convex hull of
//! its nodes. The hood also has its centreline profile: per row of its skin nodes front to
//! rear, the node nearest the centre plane (or the mean of a symmetric pair), which is
//! what shows a tent fold.
//!
//! Projection is the web crash lab's (`softbody.html` `viewProj`): look-at with world up +Y,
//! perspective with a 60° vertical field of view, pixels from the top-left corner.

use super::{metrics::robust_fit, Camera, Section, Template};
use glam::{DMat3, DVec3, Vec3};
use serde::Serialize;
use std::collections::BTreeMap;

/// The page's vertical field of view (degrees).
pub const FOV_Y_DEG: f32 = 60.0;
/// The page's near plane (m): points closer than this are not drawn.
const NEAR: f32 = 0.05;

/// A look-at camera with its own up and field of view (degrees, vertical), in the
/// scenario's world frame (the named scenario cameras are relative to the point of impact
/// too: [`Camera`]).
#[derive(Clone, Debug, PartialEq, serde::Deserialize, Serialize, ts_rs::TS)]
pub struct FreeCamera {
    pub name: String,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    #[serde(default = "y_up")]
    pub up: [f32; 3],
    #[serde(default = "fov_default")]
    pub fov_y_deg: f32,
}

/// A reference image's own camera (the Reference Studio's `Frame.camera`), set by hand in
/// the scenario's world frame: as [`FreeCamera`] without a name, with the region whose crop
/// is its image.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, Serialize, ts_rs::TS)]
pub struct FrameCamera {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    #[serde(default = "y_up")]
    pub up: [f32; 3],
    #[serde(default = "fov_default")]
    pub fov_y_deg: f32,
    /// The region it applies to (its crop's aspect), or null for the whole image.
    #[serde(default)]
    pub region: Option<String>,
}

impl FrameCamera {
    pub fn free(&self, name: &str) -> FreeCamera {
        FreeCamera { name: name.into(), eye: self.eye, target: self.target, up: self.up, fov_y_deg: self.fov_y_deg }
    }
}

fn y_up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn fov_default() -> f32 {
    FOV_Y_DEG
}

impl From<&Camera> for FreeCamera {
    fn from(c: &Camera) -> Self {
        Self { name: c.name.to_string(), eye: c.eye, target: c.target, up: y_up(), fov_y_deg: FOV_Y_DEG }
    }
}

/// A camera as the page projects it, into a `width` × `height` image.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub eye: Vec3,
    /// Screen right, up and back (from the target toward the eye), world.
    right: Vec3,
    up: Vec3,
    back: Vec3,
    pub width: f32,
    pub height: f32,
    /// 1 / tan(fov / 2).
    f: f32,
}

impl View {
    pub fn new(cam: &Camera, width: f32, height: f32) -> Self {
        Self::look(&FreeCamera::from(cam), width, height)
    }

    /// Any look-at camera: its own up direction and vertical field of view (a reference
    /// image's camera, set by hand in the Reference Studio).
    pub fn look(cam: &FreeCamera, width: f32, height: f32) -> Self {
        let (eye, at) = (Vec3::from(cam.eye), Vec3::from(cam.target));
        let back = (eye - at).normalize();
        let right = Vec3::from(cam.up).cross(back).normalize();
        let up = back.cross(right);
        Self {
            eye,
            right,
            up,
            back,
            width,
            height,
            f: 1.0 / (0.5 * cam.fov_y_deg.to_radians()).tan(),
        }
    }

    /// Pixel coordinates (x right, y down) of `p`, or None behind the near plane.
    pub fn project(&self, p: Vec3) -> Option<[f32; 2]> {
        let d = p - self.eye;
        let depth = -d.dot(self.back);
        if !(depth >= NEAR) || !p.is_finite() {
            return None;
        }
        let aspect = self.width / self.height;
        let (nx, ny) = (
            self.f / aspect * d.dot(self.right) / depth,
            self.f * d.dot(self.up) / depth,
        );
        Some([
            (nx + 1.0) * 0.5 * self.width,
            (1.0 - ny) * 0.5 * self.height,
        ])
    }

    /// Distance of `p` in front of the eye along the view axis (m).
    pub fn depth(&self, p: Vec3) -> f32 {
        -(p - self.eye).dot(self.back)
    }

    /// A circle of `radius` m about `c` seen from here: its radius in pixels (the
    /// projection of a sphere's radius at that depth).
    pub fn radius_px(&self, c: Vec3, radius: f32) -> Option<f32> {
        let depth = self.depth(c);
        (depth >= NEAR).then(|| radius * self.f * 0.5 * self.height / depth)
    }
}

/// A named part group of a vehicle (node indices into its spawned nodes).
#[derive(Clone, Debug)]
pub struct Group {
    pub name: String,
    pub nodes: Vec<usize>,
    /// Its outline: chains of node indices (closed when first == last).
    pub chains: Vec<Vec<usize>>,
    /// False when the group has no collision triangles (outline = hull of its nodes).
    pub boundary: bool,
}

/// A wheel: its hub's two axle nodes and radius (m).
#[derive(Clone, Debug)]
pub struct Wheel {
    pub name: String,
    pub hub: [usize; 2],
    pub radius: f32,
}

/// What [`groups`] finds on a design, fixed for a run.
#[derive(Clone, Debug)]
pub struct Parts {
    pub groups: Vec<Group>,
    /// Hood centreline, front to rear: each point the mean of these nodes.
    pub hood_centreline: Vec<Vec<usize>>,
    pub wheels: Vec<Wheel>,
    /// Nodes the ghost's rigid fit uses (cabin and rear sections, no wheels).
    pub rigid: Vec<usize>,
}

/// The JBeam part each group is, by its name's ending (`<design>_hood`...).
const PART_GROUPS: [(&str, &str); 10] = [
    ("hood", "_hood"),
    ("fender_l", "_fender_L"),
    ("fender_r", "_fender_R"),
    ("bumper_f", "_bumper_F"),
    ("bumper_r", "_bumper_R"),
    ("door_fl", "_door_FL"),
    ("door_fr", "_door_FR"),
    ("door_rl", "_door_RL"),
    ("door_rr", "_door_RR"),
    ("trunk", "_trunk"),
];

fn section_of(t: &Template, name: &str) -> Option<Section> {
    t.zones
        .parts
        .iter()
        .filter(|(p, _)| name.starts_with(p.as_str()))
        .max_by_key(|(p, _)| p.len())
        .map(|(_, s)| *s)
}

/// Boundary chains of a triangle set: edges used by exactly one of its triangles, walked
/// into polylines (closed ones end on their first node).
pub fn boundary_chains(tris: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let mut count: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&(a, b), &c) in &count {
        if c == 1 {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
    }
    let mut used: std::collections::BTreeSet<(usize, usize)> = Default::default();
    let mut chains = Vec::new();
    // Open chains first (from nodes of odd degree), then loops.
    let mut starts: Vec<usize> = adj
        .iter()
        .filter(|(_, v)| v.len() % 2 == 1)
        .map(|(k, _)| *k)
        .collect();
    starts.extend(adj.keys().copied());
    for s in starts {
        loop {
            let Some(&n) = adj[&s]
                .iter()
                .find(|&&n| !used.contains(&(s.min(n), s.max(n))))
            else {
                break;
            };
            let mut chain = vec![s];
            let (mut prev, mut cur) = (s, n);
            used.insert((s.min(n), s.max(n)));
            loop {
                chain.push(cur);
                if cur == s {
                    break;
                }
                let Some(&next) = adj[&cur]
                    .iter()
                    .find(|&&m| m != prev && !used.contains(&(cur.min(m), cur.max(m))))
                    .or_else(|| {
                        adj[&cur]
                            .iter()
                            .find(|&&m| !used.contains(&(cur.min(m), cur.max(m))))
                    })
                else {
                    break;
                };
                used.insert((cur.min(next), cur.max(next)));
                (prev, cur) = (cur, next);
            }
            chains.push(chain);
        }
    }
    chains
}

/// 2D convex hull (closed: first point repeated at the end).
pub fn hull(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut p: Vec<[f32; 2]> = pts
        .iter()
        .copied()
        .filter(|q| q[0].is_finite() && q[1].is_finite())
        .collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut h: Vec<[f32; 2]> = Vec::new();
    for pass in 0..2 {
        let start = h.len();
        let it: Box<dyn Iterator<Item = &[f32; 2]>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in it {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h.push(h[0]);
    h
}

/// The part groups of a design. `rest`: each spawned node at rest in the body's own frame
/// (forward +Z, up +Y, left +X); `tris`: its collision triangles (not anti-clip only).
pub fn groups(t: &Template, rest: &[Vec3], tris: &[[u32; 3]]) -> Parts {
    let d = &t.desc;
    let n = rest.len();
    let part = |suffix: &str| -> Vec<usize> {
        d.parts
            .iter()
            .find(|p| p.name.ends_with(suffix))
            .map(|p| {
                p.nodes
                    .iter()
                    .map(|&i| i as usize)
                    .filter(|&i| i < n)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut skin = vec![false; n];
    for tr in tris {
        for &i in tr {
            if let Some(s) = skin.get_mut(i as usize) {
                *s = true;
            }
        }
    }
    let mut named: Vec<(String, Vec<usize>)> = PART_GROUPS
        .iter()
        .map(|(g, s)| (g.to_string(), part(s)))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    // The body shell's regions, from its rest geometry: the belt is the hood's highest
    // point (its rear edge at the cowl), the greenhouse everything above it, the roof its
    // top quarter; the sills the body's lowest skin along each side between the axles.
    let body: Vec<usize> = part("_body").into_iter().filter(|&i| skin[i]).collect();
    let hood = part("_hood");
    let y_of = |v: &[usize]| {
        v.iter()
            .map(|&i| rest[i].y)
            .fold((f32::MAX, f32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)))
    };
    let wheels = wheels_of(t, rest);
    if !body.is_empty() && !hood.is_empty() {
        let belt = y_of(&hood).1;
        let top = y_of(&body).1;
        named.push((
            "greenhouse".into(),
            body.iter()
                .copied()
                .filter(|&i| rest[i].y >= belt - 0.02)
                .collect(),
        ));
        named.push((
            "roof".into(),
            body.iter()
                .copied()
                .filter(|&i| rest[i].y >= top - 0.25 * (top - belt))
                .collect(),
        ));
        let doors: Vec<usize> = ["_door_FL", "_door_FR"]
            .iter()
            .flat_map(|s| part(s))
            .collect();
        let (zr, zf) = wheels
            .iter()
            .map(|w| 0.5 * (rest[w.hub[0]].z + rest[w.hub[1]].z))
            .fold((f32::MAX, f32::MIN), |(a, b), z| (a.min(z), b.max(z)));
        if !doors.is_empty() && zf > zr {
            let low = y_of(&doors).0 + 0.08;
            let width = body.iter().map(|&i| rest[i].x.abs()).fold(0.0, f32::max);
            for (name, side) in [("sill_l", 1.0f32), ("sill_r", -1.0)] {
                let v = body
                    .iter()
                    .copied()
                    .filter(|&i| {
                        rest[i].y <= low
                            && rest[i].x * side > 0.6 * width
                            && rest[i].z > zr
                            && rest[i].z < zf
                    })
                    .collect();
                named.push((name.into(), v));
            }
        }
    }
    let mut groups = Vec::new();
    for (name, nodes) in named {
        let inside: std::collections::BTreeSet<usize> = nodes.iter().copied().collect();
        let own: Vec<[usize; 3]> = tris
            .iter()
            .map(|t| t.map(|i| i as usize))
            .filter(|t| t.iter().all(|i| inside.contains(i)))
            .collect();
        let chains = boundary_chains(&own);
        let boundary = !chains.is_empty();
        groups.push(Group {
            name,
            nodes,
            chains,
            boundary,
        });
    }
    let hood_skin: Vec<usize> = hood.iter().copied().filter(|&i| skin[i]).collect();
    let rigid = (0..n.min(d.nodes.len()))
        .filter(|&i| {
            matches!(
                section_of(t, &d.nodes[i].name),
                Some(Section::Cabin | Section::Rear)
            )
        })
        .collect();
    Parts {
        groups,
        hood_centreline: centreline(rest, &hood_skin),
        wheels,
        rigid,
    }
}

/// Rows of `nodes` front to rear (rest z within 4 cm), each the node nearest the centre
/// plane x = 0, or the mean of the two when a pair sits either side of it within 1 cm.
fn centreline(rest: &[Vec3], nodes: &[usize]) -> Vec<Vec<usize>> {
    let mut v: Vec<usize> = nodes.to_vec();
    v.sort_by(|&a, &b| rest[b].z.total_cmp(&rest[a].z));
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in v {
        match rows.last_mut() {
            Some(r) if (rest[r[0]].z - rest[i].z).abs() <= 0.04 => r.push(i),
            _ => rows.push(vec![i]),
        }
    }
    rows.into_iter()
        .map(|mut r| {
            r.sort_by(|&a, &b| rest[a].x.abs().total_cmp(&rest[b].x.abs()));
            let best = rest[r[0]].x.abs();
            let pair = r.iter().skip(1).find(|&&j| {
                (rest[j].x.abs() - best).abs() <= 0.01 && rest[j].x * rest[r[0]].x < 0.0
            });
            match pair {
                Some(&j) => vec![r[0], j],
                None => vec![r[0]],
            }
        })
        .collect()
}

/// The design's wheels, named by position (front/rear, left/right at rest).
fn wheels_of(t: &Template, rest: &[Vec3]) -> Vec<Wheel> {
    let d = &t.desc;
    let hub = |w: &crate::desc::SoftWheel| 0.5 * (rest[w.node1 as usize] + rest[w.node2 as usize]);
    let zs: Vec<f32> = d.wheels.iter().map(|w| hub(w).z).collect();
    let mid = 0.5
        * (zs.iter().copied().fold(f32::MAX, f32::min)
            + zs.iter().copied().fold(f32::MIN, f32::max));
    d.wheels
        .iter()
        .map(|w| {
            let c = hub(w);
            let name = format!(
                "wheel_{}{}",
                if c.z > mid { "f" } else { "r" },
                if c.x > 0.0 { "l" } else { "r" }
            );
            Wheel {
                name,
                hub: [w.node1 as usize, w.node2 as usize],
                radius: w.radius,
            }
        })
        .collect()
}

/// The rest shape carried rigidly with the cabin and rear: `R p0 + t`, fitted mass-weighted
/// from `p0` to `now` over [`Parts::rigid`] ([`robust_fit`]).
#[derive(Clone, Copy, Debug)]
pub struct Ghost {
    pub r: DMat3,
    pub t: DVec3,
}

impl Ghost {
    pub fn fit(parts: &Parts, p0: &[Vec3], now: &[Vec3], mass: &[f32]) -> Self {
        let pick = |v: &[Vec3]| parts.rigid.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let w: Vec<f32> = parts.rigid.iter().map(|&i| mass[i]).collect();
        let (r, t) = robust_fit(&pick(p0), &pick(now), &w);
        Self { r, t }
    }
    pub fn apply(&self, p: Vec3) -> Vec3 {
        (self.r * p.as_dvec3() + self.t).as_vec3()
    }
}

/// The car's own side frame: x along its current heading (horizontal), y up (world),
/// metres, origin at the rear axle's centre on the ground (y = 0).
#[derive(Clone, Copy, Debug)]
pub struct CarFrame {
    pub origin: Vec3,
    pub forward: Vec3,
}

impl CarFrame {
    /// From the current state: `heading` is the body's first heading, turned by the ghost's
    /// rotation (the rear's turn), made horizontal.
    pub fn new(parts: &Parts, now: &[Vec3], ghost: &Ghost, heading: Vec3) -> Self {
        let f = (ghost.r * heading.as_dvec3()).as_vec3();
        let forward = Vec3::new(f.x, 0.0, f.z).normalize_or(heading);
        let rear: Vec<Vec3> = parts
            .wheels
            .iter()
            .filter(|w| w.name.starts_with("wheel_r"))
            .map(|w| 0.5 * (now[w.hub[0]] + now[w.hub[1]]))
            .collect();
        let c = rear.iter().copied().sum::<Vec3>() / rear.len().max(1) as f32;
        Self {
            origin: Vec3::new(c.x, 0.0, c.z),
            forward,
        }
    }
    pub fn side(&self, p: Vec3) -> [f32; 2] {
        [(p - self.origin).dot(self.forward), p.y]
    }
}

/// One group's outline in one projection: polylines (pixels or metres) and how they were
/// made (`boundary` or `hull`).
#[derive(Clone, Debug, Serialize)]
pub struct Outline {
    pub kind: &'static str,
    pub lines: Vec<Vec<[f32; 2]>>,
}

/// A group's outline under `proj` (a point dropped when it does not project).
pub fn outline(g: &Group, pos: &[Vec3], proj: &dyn Fn(Vec3) -> Option<[f32; 2]>) -> Outline {
    if g.boundary {
        let lines = g
            .chains
            .iter()
            .map(|c| c.iter().filter_map(|&i| proj(pos[i])).collect::<Vec<_>>())
            .filter(|l: &Vec<_>| l.len() >= 2)
            .collect();
        Outline {
            kind: "boundary",
            lines,
        }
    } else {
        let pts: Vec<[f32; 2]> = g.nodes.iter().filter_map(|&i| proj(pos[i])).collect();
        Outline {
            kind: "hull",
            lines: vec![hull(&pts)]
                .into_iter()
                .filter(|l| l.len() >= 2)
                .collect(),
        }
    }
}

/// The hood's centreline under `proj`, front to rear.
pub fn centreline_line(
    parts: &Parts,
    pos: &[Vec3],
    proj: &dyn Fn(Vec3) -> Option<[f32; 2]>,
) -> Vec<[f32; 2]> {
    parts
        .hood_centreline
        .iter()
        .filter_map(|r| proj(r.iter().map(|&i| pos[i]).sum::<Vec3>() / r.len() as f32))
        .collect()
}

/// A body's part groups and what its outlines are measured against, fixed for a run: its
/// groups ([`groups`] over its rest shape and its collision triangles, those that are not
/// anti-only), its spawn positions and its node masses.
pub struct BodyParts {
    pub name: String,
    pub design: String,
    pub heading: Vec3,
    pub parts: Parts,
    pub p0: Vec<Vec3>,
    pub mass: Vec<f32>,
}

impl BodyParts {
    /// For body `b` of a built run (spawned as `f`) of `t`.
    pub fn new(b: &super::Body, f: &crate::explicit_beams::ExplicitBeams, t: &Template) -> Self {
        let rest: Vec<Vec3> = (0..f.nodes.len()).map(|i| t.desc.nodes.get(i).map_or(Vec3::ZERO, |n| Vec3::from(n.position))).collect();
        let tris: Vec<[u32; 3]> = f.triangles.iter().enumerate().filter(|(k, _)| !f.tri_anti_only(*k)).map(|(_, x)| x.nodes).collect();
        Self { name: b.name.clone(), design: b.design.clone(), heading: b.heading(), parts: groups(t, &rest, &tris), p0: f.positions(), mass: f.nodes.iter().map(|n| n.mass).collect() }
    }

    /// Its outlines at one instant (node positions `now`), as `outlines.json` holds a body
    /// (`crash_outlines`) and the Reference Studio computes it from a recording: per camera
    /// view its part groups (now and ghost), hood centreline and wheels (pixels), the same in
    /// the car's own side frame (`car_frame`), and the rear's turn from the start. The one
    /// function both use, so an outline is the same for the same positions and camera.
    pub fn outlines(&self, now: &[Vec3], views: &[(String, View)]) -> serde_json::Value {
        use serde_json::json;
        let parts = &self.parts;
        let g = Ghost::fit(parts, &self.p0, now, &self.mass);
        let ghost: Vec<Vec3> = self.p0.iter().map(|&p| g.apply(p)).collect();
        let frame = CarFrame::new(parts, now, &g, self.heading);
        let groups_json = |proj: &dyn Fn(Vec3) -> Option<[f32; 2]>| {
            let mut groups = serde_json::Map::new();
            for gr in &parts.groups {
                groups.insert(gr.name.clone(), json!({ "now": outline(gr, now, proj), "ghost": outline(gr, &ghost, proj) }));
            }
            json!({
                "parts": groups,
                "hood_centreline": { "now": centreline_line(parts, now, proj), "ghost": centreline_line(parts, &ghost, proj) },
            })
        };
        let hub = |p: &[Vec3], w: &Wheel| 0.5 * (p[w.hub[0]] + p[w.hub[1]]);
        let mut cams = serde_json::Map::new();
        for (c, v) in views {
            let mut o = groups_json(&|p| v.project(p));
            o["wheels"] = json!(parts
                .wheels
                .iter()
                .map(|wh| {
                    let (c_now, c_gh) = (hub(now, wh), hub(&ghost, wh));
                    json!({ "name": wh.name, "now": { "c": v.project(c_now), "r": v.radius_px(c_now, wh.radius) }, "ghost": { "c": v.project(c_gh), "r": v.radius_px(c_gh, wh.radius) } })
                })
                .collect::<Vec<_>>());
            cams.insert(c.clone(), o);
        }
        let mut car = groups_json(&|p| Some(frame.side(p)));
        car["wheels"] = json!(parts.wheels.iter().map(|wh| json!({ "name": wh.name, "r": wh.radius, "now": frame.side(hub(now, wh)), "ghost": frame.side(hub(&ghost, wh)) })).collect::<Vec<_>>());
        // The cabin and rear's turn from the start: yaw (+ toward its left), pitch (+ nose down, rear up).
        let (yaw, pitch, _) = super::metrics::yaw_pitch_roll(g.r, self.heading);
        json!({ "name": self.name, "design": self.design, "cameras": cams, "car_frame": car, "fit_nodes": parts.rigid.len(), "rear_yaw_deg": yaw.to_degrees(), "rear_pitch_deg": pitch.to_degrees() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
    }

    #[test]
    fn projects_a_box_as_the_page_does() {
        // A unit box 5 m in front of the eye on +Z, 1280 × 720, 60° vertical field of view.
        let v = View::new(
            &Camera {
                name: "t".into(),
                eye: [0.0, 0.0, 5.0],
                target: [0.0, 0.0, 0.0],
            },
            1280.0,
            720.0,
        );
        let f = 1.0 / 30f32.to_radians().tan();
        assert!(close(v.project(Vec3::ZERO).unwrap(), [640.0, 360.0]));
        // Screen right is world +X (looking down −Z), up is +Y; depth 5 m.
        let x = 640.0 + 0.2 * f / (1280.0 / 720.0) * 640.0;
        let y = 360.0 - 0.2 * f * 360.0;
        assert!(close(v.project(Vec3::X).unwrap(), [x, 360.0]));
        assert!(close(v.project(Vec3::Y).unwrap(), [640.0, y]));
        // A face nearer the eye is larger: the box's front corner (depth 4) vs back (depth 6).
        let front = v.project(Vec3::new(0.5, 0.5, 1.0)).unwrap();
        let back = v.project(Vec3::new(0.5, 0.5, -1.0)).unwrap();
        assert!(front[0] - 640.0 > back[0] - 640.0 && 360.0 - front[1] > 360.0 - back[1]);
        assert!(close(
            front,
            [
                640.0 + 0.5 / 4.0 * f / (16.0 / 9.0) * 640.0,
                360.0 - 0.5 / 4.0 * f * 360.0
            ]
        ));
        // Behind the eye: nothing.
        assert!(v.project(Vec3::new(0.0, 0.0, 6.0)).is_none());
        // A sphere's radius at depth 5: r f / 5 of half the height.
        assert!((v.radius_px(Vec3::ZERO, 1.0).unwrap() - f / 5.0 * 360.0).abs() < 1e-3);
    }

    #[test]
    fn a_free_camera_with_its_own_up_and_field_of_view() {
        let named = Camera { name: "t".into(), eye: [0.0, 0.0, 5.0], target: [0.0, 0.0, 0.0] };
        let free = FreeCamera::from(&named);
        let (a, b) = (View::new(&named, 1280.0, 720.0), View::look(&free, 1280.0, 720.0));
        assert_eq!(a.project(Vec3::new(0.3, 0.2, 0.1)), b.project(Vec3::new(0.3, 0.2, 0.1)));
        // Half the field of view: offsets from the centre grow by tan 30° / tan 15°.
        let narrow = View::look(&FreeCamera { fov_y_deg: 30.0, ..free.clone() }, 1280.0, 720.0);
        let k = 30f32.to_radians().tan() / 15f32.to_radians().tan();
        let (p, q) = (a.project(Vec3::Y).unwrap(), narrow.project(Vec3::Y).unwrap());
        assert!(((360.0 - q[1]) - k * (360.0 - p[1])).abs() < 1e-2);
        // Up along world +X (camera rolled a quarter turn): world +X is screen up.
        let rolled = View::look(&FreeCamera { up: [1.0, 0.0, 0.0], ..free }, 1280.0, 720.0);
        let r = rolled.project(Vec3::X).unwrap();
        assert!((r[0] - 640.0).abs() < 1e-3 && r[1] < 360.0);
        // JSON with only eye and target defaults to the page's camera.
        let j: FreeCamera = serde_json::from_str("{\"name\":\"f\",\"eye\":[0,0,5],\"target\":[0,0,0]}").unwrap();
        assert_eq!(j, FreeCamera { name: "f".into(), ..FreeCamera::from(&named) });
    }

    #[test]
    fn top_view_up_is_the_eye_offset_reversed() {
        // As S02's top camera: the eye a centimetre to +x of straight above.
        let v = View::new(
            &Camera {
                name: "top".into(),
                eye: [0.01, 5.0, 0.0],
                target: [0.0, 0.0, 0.0],
            },
            1280.0,
            720.0,
        );
        let up = v.project(Vec3::new(-1.0, 0.0, 0.0)).unwrap();
        let right = v.project(Vec3::new(0.0, 0.0, -1.0)).unwrap();
        assert!(up[1] < 360.0 && (up[0] - 640.0).abs() < 1.0);
        assert!(right[0] > 640.0 && (right[1] - 360.0).abs() < 1.0);
    }

    #[test]
    fn box_faces_have_their_rims_as_boundary() {
        // The top face of a box (two triangles): one closed loop of its four corners.
        let tris = [[0, 1, 2], [0, 2, 3]];
        let c = boundary_chains(&tris);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 5);
        assert_eq!(c[0].first(), c[0].last());
        // A closed box (12 triangles) has no boundary.
        let cube = [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ];
        assert!(boundary_chains(&cube).is_empty());
        // Its projection's hull is the square's outline.
        let h = hull(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]);
        assert_eq!(h.len(), 5);
    }
}
