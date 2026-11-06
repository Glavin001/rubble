//! The generated sedan's render mesh (Collada) and materials: the same analytic
//! surface ([`super::shape`]) the physics nodes sit on, sampled at about 3 cm
//! (visual resolution decoupled from the ~0.5 m physics lattice), smooth
//! shaded, cut into its parts with real 6 mm shut-line gaps whose panel edges
//! are hemmed (a return flange turned under), the wheel arches' edges curved.
//! Each part's meshes are flexbodies bound to that part's own node group only,
//! never across a shut line; glass and lamps carry deform groups (BeamNG's
//! flexbody `deformGroup`) so they shatter or go dark when their frame bends.

use super::build::*;
use super::export::Flex;
use super::sedan::end_y;
use super::shape::*;
use glam::Vec3;
use std::collections::HashMap;
use std::fmt::Write as _;

pub struct Visual {
    pub flex: Vec<(Part, Flex)>,
    pub deform_groups: Vec<(String, Vec<usize>, f32)>,
    pub collada: String,
    pub materials: String,
}

/// Half the shut-line gap (m).
const GAP: f32 = 0.003;
/// Hem (return flange) depth (m).
const HEM: f32 = 0.007;
/// Target vertex spacing (m).
const RES: f32 = 0.03;

struct Mesh {
    name: String,
    part: Part,
    groups: Vec<String>,
    deform: Option<String>,
    pos: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    tmat: Vec<usize>,
    mats: Vec<String>,
}

fn n_of(len: f32) -> usize {
    ((len / RES).ceil() as usize).clamp(2, 400)
}

impl Mesh {
    fn new(name: &str, part: Part) -> Self {
        Mesh {
            name: name.into(),
            part,
            groups: vec![part.name().into()],
            deform: None,
            pos: Vec::new(),
            tris: Vec::new(),
            tmat: Vec::new(),
            mats: Vec::new(),
        }
    }

    fn mat(&mut self, m: &str) -> usize {
        self.mats.iter().position(|x| x == m).unwrap_or_else(|| {
            self.mats.push(m.into());
            self.mats.len() - 1
        })
    }

    /// Push a triangle facing `out` (a direction away from the surface).
    fn tri_out(&mut self, t: [u32; 3], mat: usize, out: Vec3) {
        let [a, b, c] = t.map(|i| self.pos[i as usize]);
        let n = (b - a).cross(c - a);
        if n.length_squared() < 1e-14 {
            return;
        }
        self.tris.push(if n.dot(out) >= 0.0 {
            t
        } else {
            [t[0], t[2], t[1]]
        });
        self.tmat.push(mat);
    }

    /// A grid patch: `f(u, v)` for u, v in [0, 1] (nu × nv cells), each
    /// triangle facing away from the car's long axis (or `out` if given),
    /// `mat_of(centre)` its material, hemmed along its boundary when `hem`.
    #[allow(clippy::too_many_arguments)]
    fn grid(
        &mut self,
        nu: usize,
        nv: usize,
        f: &dyn Fn(f32, f32) -> Vec3,
        mat_of: &dyn Fn(Vec3) -> &'static str,
        hem: bool,
        out: Option<Vec3>,
    ) {
        let first = self.pos.len() as u32;
        for j in 0..=nv {
            for i in 0..=nu {
                self.pos.push(f(i as f32 / nu as f32, j as f32 / nv as f32));
            }
        }
        let id = |i: usize, j: usize| first + (j * (nu + 1) + i) as u32;
        let t0 = self.tris.len();
        for j in 0..nv {
            for i in 0..nu {
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                let cen = (self.pos[a as usize] + self.pos[c as usize]) * 0.5;
                let o = out.unwrap_or_else(|| outward(cen));
                let m = self.mat(mat_of(cen));
                self.tri_out([a, b, c], m, o);
                self.tri_out([a, c, d], m, o);
            }
        }
        if hem {
            self.hem(t0);
        }
    }

    /// Return flanges on the boundary of the triangles from `t0` on.
    fn hem(&mut self, t0: usize) {
        let mut edges: HashMap<(u32, u32), (usize, u32)> = HashMap::new();
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        for (k, t) in self.tris[t0..].iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let key = (a.min(b), a.max(b));
                *count.entry(key).or_insert(0) += 1;
                edges.insert(key, (t0 + k, t[(e + 2) % 3]));
            }
        }
        // Vertex normals of the patch.
        let mut nrm: HashMap<u32, Vec3> = HashMap::new();
        for t in &self.tris[t0..] {
            let [a, b, c] = t.map(|i| self.pos[i as usize]);
            let n = (b - a).cross(c - a);
            for &i in t {
                *nrm.entry(i).or_insert(Vec3::ZERO) += n;
            }
        }
        let mut under: HashMap<u32, u32> = HashMap::new();
        let mut boundary: Vec<((u32, u32), usize, u32)> = Vec::new();
        for (key, c) in &count {
            if *c == 1 {
                let (t, opp) = edges[key];
                boundary.push((*key, t, opp));
            }
        }
        boundary.sort();
        for ((a, b), t, opp) in boundary {
            let mat = self.tmat[t];
            let mut flange = |i: u32, me: &mut Mesh| -> u32 {
                *under.entry(i).or_insert_with(|| {
                    let n = nrm[&i].normalize_or_zero();
                    me.pos.push(me.pos[i as usize] - n * HEM);
                    (me.pos.len() - 1) as u32
                })
            };
            let (a2, b2) = (flange(a, self), flange(b, self));
            // Facing away from the panel (from the opposite vertex across the edge).
            let mid = (self.pos[a as usize] + self.pos[b as usize]) * 0.5;
            let out = mid - self.pos[opp as usize];
            self.tri_out([a, b, b2], mat, out);
            self.tri_out([a, b2, a2], mat, out);
        }
    }

    /// An axis-aligned box (interior trim), faces outward.
    fn cuboid(&mut self, lo: Vec3, hi: Vec3, mat: &str) {
        let m = self.mat(mat);
        let c = (lo + hi) * 0.5;
        for axis in 0..3 {
            for side in [0.0f32, 1.0] {
                let mut q = Vec::new();
                for (u, v) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                    let mut p = [0.0f32; 3];
                    p[axis] = side;
                    p[(axis + 1) % 3] = u;
                    p[(axis + 2) % 3] = v;
                    q.push(lo + (hi - lo) * Vec3::from_array(p));
                }
                let first = self.pos.len() as u32;
                self.pos.extend(q);
                let out = Vec3::from_array({
                    let mut o = [0.0; 3];
                    o[axis] = if side > 0.5 { 1.0 } else { -1.0 };
                    o
                });
                let _ = c;
                self.tri_out([first, first + 1, first + 2], m, out);
                self.tri_out([first, first + 2, first + 3], m, out);
            }
        }
    }
}

/// Away from the car's long axis.
fn outward(p: Vec3) -> Vec3 {
    let o = p - Vec3::new(0.0, p.y, 0.70);
    if o.length() < 1e-3 {
        Vec3::Z
    } else {
        o
    }
}

fn mirror_x(p: Vec3, s: f32) -> Vec3 {
    Vec3::new(p.x * s, p.y, p.z)
}

/// The rear door's lower edge (z at y).
fn rear_door_bottom(y: f32) -> f32 {
    smooth(&[(2.36, 0.335), (2.66, 0.80), (2.84, 0.985)], y)
}

/// The fender's lower edge (around the front arch).
fn fender_bottom(y: f32) -> f32 {
    match arch_z(y) {
        Some(z) => z + GAP,
        None if y < AXLE_F => 0.42,
        None => 0.36,
    }
}

/// x of the end caps' outline at t (0 centre … 1 the seam with the side).
fn end_point(t: f32, z: f32, front: bool, s: f32) -> Vec3 {
    let seam = if front { -0.80 } else { 3.80 };
    let x0 = t * plan_x(seam);
    let y = end_y(x0.min(HALF_W * 0.9999), front);
    let p = side(y, z);
    Vec3::new(p.x * s, y, z)
}

pub fn build(car: &Car) -> Visual {
    let mut meshes: Vec<Mesh> = Vec::new();
    let paint = |_: Vec3| "gen2_paint";
    for s in [1.0f32, -1.0] {
        let sx = if s > 0.0 { "L" } else { "R" };
        // ---------------- lower side panels ----------------
        let side_patch = |m: &mut Mesh,
                          y0: f32,
                          y1: f32,
                          lo: &dyn Fn(f32) -> f32,
                          hi: &dyn Fn(f32) -> f32,
                          mat: &dyn Fn(Vec3) -> &'static str| {
            let nu = n_of(y1 - y0);
            let nv = n_of(0.6);
            m.grid(
                nu,
                nv,
                &|u, v| {
                    let y = y0 + (y1 - y0) * u;
                    let z = lo(y) + (hi(y) - lo(y)).max(0.0) * v;
                    mirror_x(side(y, z), s)
                },
                mat,
                true,
                None,
            );
        };
        let fender = if s > 0.0 {
            Part::FenderL
        } else {
            Part::FenderR
        };
        let mut m = Mesh::new(&format!("gen2_fender_{sx}"), fender);
        side_patch(
            &mut m,
            -0.80 + GAP,
            0.80 - GAP,
            &fender_bottom,
            &|y| z_belt(y) - GAP,
            &paint,
        );
        meshes.push(m);
        for front in [true, false] {
            let part = Part::door(front, s > 0.0);
            let tag = if front { "F" } else { "R" };
            let mut m = Mesh::new(&format!("gen2_door_{tag}{sx}"), part);
            if front {
                side_patch(
                    &mut m,
                    0.80 + GAP,
                    1.88 - GAP,
                    &|_| 0.335,
                    &|y| z_belt(y) - GAP,
                    &paint,
                );
            } else {
                side_patch(
                    &mut m,
                    1.92 + GAP,
                    2.86 - GAP,
                    &|y| {
                        if y < 2.36 {
                            0.335
                        } else {
                            rear_door_bottom(y) + GAP
                        }
                    },
                    &|y| z_belt(y) - GAP,
                    &paint,
                );
            }
            // Window frame (black) and the glass in it.
            let (gy0, gy1) = if front { (1.02, 1.86) } else { (1.96, 2.80) };
            let gl = |y: f32, g: f32| mirror_x(glass(y, g), s);
            let nu = n_of(gy1 - gy0);
            m.grid(
                nu,
                2,
                &|u, v| gl(gy0 + (gy1 - gy0) * u, 0.93 + 0.07 * v),
                &|_| "gen2_trim_black",
                true,
                None,
            );
            m.grid(
                nu,
                2,
                &|u, v| gl(gy0 + (gy1 - gy0) * u, 0.07 * v),
                &|_| "gen2_trim_black",
                true,
                None,
            );
            meshes.push(m);
            let mut gm = Mesh::new(&format!("gen2_glass_door_{tag}{sx}"), part);
            gm.deform = Some(format!("glass_door_{tag}{sx}"));
            let rear_lim = |g: f32| if front { gy1 } else { gy1 - 0.12 * g };
            gm.grid(
                nu,
                n_of(0.45),
                &|u, v| {
                    let g = 0.07 + 0.86 * v;
                    gl(gy0 + 0.02 + (rear_lim(g) - gy0 - 0.04) * u, g)
                },
                &|_| "gen2_glass_side",
                false,
                None,
            );
            meshes.push(gm);
            // Door inner trim (interior side), mirror on the front door.
            let mut im = Mesh::new(&format!("gen2_door_trim_{tag}{sx}"), part);
            let (y0, y1) = if front { (0.84, 1.84) } else { (1.96, 2.40) };
            im.grid(
                n_of(y1 - y0),
                n_of(0.5),
                &|u, v| {
                    let y = y0 + (y1 - y0) * u;
                    let z = 0.42 + (z_belt(y) - 0.45) * v;
                    mirror_x(side(y, z) - Vec3::X * 0.11, s)
                },
                &|_| "gen2_interior_door",
                false,
                Some(Vec3::new(-s, 0.0, 0.0)),
            );
            if front {
                let c = mirror_x(side(0.98, 1.0) + Vec3::new(0.07, 0.0, 0.02), s);
                let (lo, hi) = (
                    c - Vec3::new(0.06, 0.03, 0.05),
                    c + Vec3::new(0.06, 0.08, 0.05),
                );
                im.cuboid(lo.min(hi), lo.max(hi), "gen2_mirror_chrome");
            }
            meshes.push(im);
        }
        // Body side: sill and quarter, C-pillar sail, A-pillar side, B-pillar.
        let mut m = Mesh::new(&format!("gen2_body_side_{sx}"), Part::Body);
        side_patch(
            &mut m,
            0.42,
            2.40,
            &|y| z_bottom(y),
            &|y| if y < 0.80 { 0.36 - GAP } else { 0.335 - GAP },
            &paint,
        );
        side_patch(
            &mut m,
            2.40,
            3.80 - GAP,
            &|y| arch_z(y).map_or(z_bottom(y), |z| z + GAP),
            &|y| {
                if y < 2.86 + GAP {
                    rear_door_bottom(y.max(2.36)) - GAP
                } else {
                    z_belt(y)
                }
            },
            &paint,
        );
        let gl = |y: f32, g: f32| mirror_x(glass(y, g), s);
        m.grid(
            n_of(0.55),
            n_of(0.4),
            &|u, v| gl(2.80 + 0.53 * u, v),
            &paint,
            true,
            None,
        );
        m.grid(
            n_of(0.2),
            n_of(0.4),
            &|u, v| gl(0.84 + 0.18 * u, v),
            &paint,
            true,
            None,
        );
        m.grid(
            4,
            n_of(0.4),
            &|u, v| gl(1.88 + 0.06 * u, v),
            &|_| "gen2_trim_black",
            true,
            None,
        );
        // Wheel arch liners (black), front on the fender.
        for (ya, part) in [(AXLE_F, fender), (AXLE_R, Part::Body)] {
            let mut lm = if part == Part::Body {
                std::mem::replace(&mut m, Mesh::new("", Part::Body))
            } else {
                Mesh::new(&format!("gen2_arch_liner_{sx}"), part)
            };
            lm.grid(
                n_of(1.3),
                4,
                &|u, v| {
                    let a = std::f32::consts::PI * u;
                    let r = ARCH_R + 0.005;
                    let x = 0.62 + (side(ya, WHEEL_R + r * a.sin()).x - 0.63) * v;
                    mirror_x(Vec3::new(x, ya + r * a.cos(), WHEEL_R + r * a.sin()), s)
                },
                &|_| "gen2_liner_black",
                false,
                Some(Vec3::new(0.0, 0.0, -1.0)),
            );
            if part == Part::Body {
                m = lm;
            } else {
                meshes.push(lm);
            }
        }
        meshes.push(m);
        // Lamps.
        let mut hl = Mesh::new(&format!("gen2_headlight_{sx}"), Part::BumperF);
        hl.groups = vec![Part::Body.name().into()];
        hl.deform = Some(format!("headlight_{sx}"));
        hl.grid(
            n_of(0.5),
            n_of(0.15),
            &|u, v| {
                let t = 0.40 + 0.60 * u;
                let p0 = end_point(t, 0.60 + GAP, true, s);
                let z = 0.60 + GAP + (z_belt(p0.y) - 0.60 - 2.0 * GAP) * v;
                end_point(t, z, true, s)
            },
            &|_| "gen2_headlight_lens",
            true,
            None,
        );
        meshes.push(hl);
        let mut tl = Mesh::new(&format!("gen2_taillight_{sx}"), Part::BumperR);
        tl.groups = vec![Part::Body.name().into()];
        tl.deform = Some(format!("taillight_{sx}"));
        tl.grid(
            n_of(0.45),
            n_of(0.3),
            &|u, v| {
                let t = 0.45 + 0.55 * u;
                let p0 = end_point(t, 0.62, false, s);
                let z = 0.62 + GAP + (z_belt(p0.y) - 0.62 - 2.0 * GAP) * v;
                end_point(t, z, false, s)
            },
            &|_| "gen2_taillight_red",
            true,
            None,
        );
        meshes.push(tl);
    }
    // ---------------- fascias, grille, rear panel ----------------
    for front in [true, false] {
        let part = if front { Part::BumperF } else { Part::BumperR };
        let mut m = Mesh::new(
            if front {
                "gen2_bumper_F"
            } else {
                "gen2_bumper_R"
            },
            part,
        );
        for s in [1.0f32, -1.0] {
            m.grid(
                n_of(0.9),
                n_of(0.4),
                &|u, v| {
                    let p0 = end_point(u, 0.3, front, s);
                    let z = z_bottom(p0.y) + (0.60 - z_bottom(p0.y)) * v;
                    end_point(u, z, front, s)
                },
                &|p| {
                    if front && p.x.abs() < 0.42 && p.z > 0.40 && p.z < 0.56 {
                        "gen2_grille_black"
                    } else {
                        "gen2_paint"
                    }
                },
                true,
                None,
            );
            if front {
                // Upper grille between the headlights.
                m.grid(
                    n_of(0.4),
                    3,
                    &|u, v| {
                        let t = 0.40 * u;
                        let p0 = end_point(t, 0.6, true, s);
                        end_point(
                            t,
                            0.60 + GAP + (z_belt(p0.y) - 0.60 - 2.0 * GAP) * v,
                            true,
                            s,
                        )
                    },
                    &|_| "gen2_grille_black",
                    true,
                    None,
                );
            }
        }
        meshes.push(m);
    }
    {
        let mut m = Mesh::new("gen2_rear_panel", Part::Body);
        for s in [1.0f32, -1.0] {
            m.grid(
                n_of(0.4),
                n_of(0.3),
                &|u, v| {
                    let t = 0.45 * u;
                    let p0 = end_point(t, 0.62, false, s);
                    end_point(t, 0.62 + GAP + (z_belt(p0.y) - 0.62 - GAP) * v, false, s)
                },
                &paint,
                true,
                None,
            );
        }
        // Nose top (above the headlights, ahead of the hood) on the bumper.
        meshes.push(m);
        let mut nose = Mesh::new("gen2_nose_top", Part::BumperF);
        nose.grid(
            n_of(0.18),
            n_of(1.4),
            &|u, v| {
                top(
                    Y_FRONT + 0.02 + (-0.80 - GAP - Y_FRONT - 0.02) * u,
                    -1.0 + 2.0 * v,
                )
            },
            &paint,
            true,
            None,
        );
        meshes.push(nose);
    }
    // ---------------- top: hood, cowl, glass, roof, deck lid ----------------
    let top_patch = |m: &mut Mesh,
                     y0: f32,
                     y1: f32,
                     u0: f32,
                     u1: f32,
                     mat: &dyn Fn(Vec3) -> &'static str,
                     hem: bool| {
        m.grid(
            n_of(y1 - y0),
            n_of(1.6 * (u1 - u0) / 2.0),
            &|a, b| top(y0 + (y1 - y0) * a, u0 + (u1 - u0) * b),
            mat,
            hem,
            None,
        );
    };
    let mut hood = Mesh::new("gen2_hood", Part::Hood);
    top_patch(
        &mut hood,
        -0.80 + GAP,
        0.72 - GAP,
        -0.995,
        0.995,
        &paint,
        true,
    );
    meshes.push(hood);
    let mut body_top = Mesh::new("gen2_body_top", Part::Body);
    top_patch(
        &mut body_top,
        0.72 + GAP,
        0.84,
        -1.0,
        1.0,
        &|_| "gen2_trim_black",
        true,
    );
    top_patch(&mut body_top, 1.57, 2.80, -1.0, 1.0, &paint, true);
    top_patch(&mut body_top, 3.33, 3.36 - GAP, -1.0, 1.0, &paint, true);
    for (u0, u1) in [(-1.0, -0.93), (0.93, 1.0)] {
        top_patch(&mut body_top, 0.84, 1.57, u0, u1, &paint, true);
        top_patch(&mut body_top, 2.80, 3.33, u0, u1, &paint, true);
    }
    // Floor pan, from below (black).
    body_top.grid(
        n_of(4.5),
        n_of(1.6),
        &|a, b| {
            let y = -0.75 + 4.5 * a;
            let xs = side(y, z_bottom(y)).x - 0.02;
            Vec3::new(xs * (2.0 * b - 1.0), y, z_bottom(y) + 0.004)
        },
        &|_| "gen2_underbody_black",
        false,
        Some(Vec3::new(0.0, 0.0, -1.0)),
    );
    meshes.push(body_top);
    let mut ws = Mesh::new("gen2_windshield", Part::Body);
    ws.deform = Some("windshield".into());
    top_patch(
        &mut ws,
        0.84,
        1.57,
        -0.93,
        0.93,
        &|_| "gen2_windshield",
        false,
    );
    meshes.push(ws);
    let mut rw = Mesh::new("gen2_glass_rear", Part::Body);
    rw.deform = Some("glass_rear".into());
    top_patch(
        &mut rw,
        2.80,
        3.33,
        -0.93,
        0.93,
        &|_| "gen2_glass_rear",
        false,
    );
    meshes.push(rw);
    let mut trunk = Mesh::new("gen2_trunk", Part::Trunk);
    top_patch(&mut trunk, 3.36 + GAP, 3.92, -0.995, 0.995, &paint, true);
    meshes.push(trunk);
    // ---------------- interior ----------------
    let mut int = Mesh::new("gen2_interior", Part::Body);
    for x in [0.38f32, -0.38] {
        let c = Vec3::new(x, 1.75, 0.0);
        int.cuboid(
            c + Vec3::new(-0.25, -0.25, 0.26),
            c + Vec3::new(0.25, 0.25, 0.40),
            "gen2_interior_seat",
        );
        int.cuboid(
            c + Vec3::new(-0.24, 0.20, 0.40),
            c + Vec3::new(0.24, 0.32, 1.02),
            "gen2_interior_seat",
        );
        int.cuboid(
            c + Vec3::new(-0.12, 0.24, 1.02),
            c + Vec3::new(0.12, 0.31, 1.16),
            "gen2_interior_seat",
        );
    }
    int.cuboid(
        Vec3::new(-0.68, 2.40, 0.26),
        Vec3::new(0.68, 2.85, 0.42),
        "gen2_interior_seat",
    );
    int.cuboid(
        Vec3::new(-0.68, 2.75, 0.42),
        Vec3::new(0.68, 2.92, 0.95),
        "gen2_interior_seat",
    );
    // Instrument panel over its cross-car beam, the console, the carpet.
    int.cuboid(
        Vec3::new(-0.78, 0.86, 0.58),
        Vec3::new(0.78, 1.12, 0.88),
        "gen2_interior_dash",
    );
    int.cuboid(
        Vec3::new(-0.12, 1.10, 0.26),
        Vec3::new(0.12, 2.15, 0.52),
        "gen2_interior_dash",
    );
    int.cuboid(
        Vec3::new(-0.78, 0.80, 0.22),
        Vec3::new(0.78, 3.0, 0.25),
        "gen2_interior_carpet",
    );
    // Steering wheel: a thin ring of boxes.
    let sc = Vec3::new(0.38, 1.25, 0.90);
    for k in 0..16 {
        let a = k as f32 * std::f32::consts::TAU / 16.0;
        let p = sc + Vec3::new(0.18 * a.cos(), 0.06 * a.sin(), 0.17 * a.sin());
        int.cuboid(
            p - Vec3::splat(0.018),
            p + Vec3::splat(0.018),
            "gen2_interior_steer",
        );
    }
    meshes.push(int);
    // Headliner (from inside).
    let mut hlm = Mesh::new("gen2_headliner", Part::Body);
    hlm.grid(
        n_of(1.2),
        n_of(1.2),
        &|a, b| top(1.60 + 1.15 * a, -0.95 + 1.9 * b) - Vec3::Z * 0.035,
        &|_| "gen2_interior_headliner",
        false,
        Some(Vec3::new(0.0, 0.0, -1.0)),
    );
    meshes.push(hlm);

    // Merge meshes of the same name (the body's side patches per side).
    let mut flex = Vec::new();
    for m in &meshes {
        if m.name.is_empty() {
            continue;
        }
        flex.push((
            m.part,
            Flex {
                mesh: m.name.clone(),
                groups: m.groups.clone(),
                deform_group: m.deform.clone(),
                skin: None,
                pos: None,
            },
        ));
    }
    Visual {
        flex,
        deform_groups: deform_groups(car),
        collada: collada(&meshes),
        materials: materials(),
    }
}

/// Glass and lamps break when their frame's beams deform (BeamNG's flexbody
/// deformGroup): the beams joining these nodes carry the group.
fn deform_groups(car: &Car) -> Vec<(String, Vec<usize>, f32)> {
    let ids = |names: &[String]| -> Vec<usize> {
        names
            .iter()
            .filter_map(|n| car.nodes.iter().position(|x| &x.name == n))
            .collect()
    };
    let v = |list: &[&str]| -> Vec<String> { list.iter().map(|s| s.to_string()).collect() };
    let mut out = vec![
        (
            "windshield".to_string(),
            ids(&v(&[
                "cowlOl", "cowlOr", "cowlIl", "cowlIr", "cowlC", "rr0l", "rr0r", "rfC0", "ap1l",
                "ap1r",
            ])),
            0.02,
        ),
        (
            "glass_rear".to_string(),
            ids(&v(&["rr3l", "rr3r", "rfC3", "dkl", "dkr", "dkC"])),
            0.02,
        ),
    ];
    for (x, sx) in [("l", "L"), ("r", "R")] {
        for tag in ["F", "R"] {
            let names: Vec<String> = (0..3)
                .map(|k| format!("d{tag}2{k}{x}"))
                .chain([format!("d{tag}w0{x}"), format!("d{tag}w1{x}")])
                .collect();
            out.push((format!("glass_door_{tag}{sx}"), ids(&names), 0.02));
        }
        let fk: [usize; 3] = if x == "l" { [4, 5, 6] } else { [0, 1, 2] };
        let mut hl: Vec<String> = vec![
            format!("apr0{x}"),
            format!("bbHi0{x}"),
            format!("bbHi1{x}"),
            format!("rlHi0{x}"),
        ];
        hl.extend(fk.iter().map(|k| format!("fF2{k}")));
        out.push((format!("headlight_{sx}"), ids(&hl), 0.03));
        let mut tl: Vec<String> = vec![
            format!("qb2{x}"),
            format!("rp{x}"),
            format!("qr1{x}"),
            format!("rbHi0{x}"),
        ];
        tl.extend(fk.iter().map(|k| format!("fR2{k}")));
        out.push((format!("taillight_{sx}"), ids(&tl), 0.03));
    }
    out
}

fn collada(meshes: &[Mesh]) -> String {
    let mut all_mats: Vec<String> = Vec::new();
    for m in meshes {
        for x in &m.mats {
            if !all_mats.contains(x) {
                all_mats.push(x.clone());
            }
        }
    }
    let mut s = String::new();
    s += "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">\n";
    s += "<asset><contributor><authoring_tool>Rubble gen2 (engine_softbody::authoring::gen2)</authoring_tool></contributor><unit name=\"meter\" meter=\"1\"/><up_axis>Z_UP</up_axis></asset>\n";
    s += "<library_materials>\n";
    for m in &all_mats {
        let _ = writeln!(s, "<material id=\"{m}\" name=\"{m}\"/>");
    }
    s += "</library_materials>\n<library_geometries>\n";
    // Merge same-named meshes.
    let mut order: Vec<&str> = Vec::new();
    let mut by: HashMap<&str, Vec<&Mesh>> = HashMap::new();
    for m in meshes.iter().filter(|m| !m.name.is_empty()) {
        if !by.contains_key(m.name.as_str()) {
            order.push(&m.name);
        }
        by.entry(&m.name).or_default().push(m);
    }
    for name in &order {
        let parts = &by[name];
        let mut pos: Vec<Vec3> = Vec::new();
        let mut tris: Vec<([u32; 3], String)> = Vec::new();
        for m in parts {
            let base = pos.len() as u32;
            pos.extend(&m.pos);
            for (t, &mi) in m.tris.iter().zip(&m.tmat) {
                tris.push((t.map(|i| i + base), m.mats[mi].clone()));
            }
        }
        // Smooth normals (shared vertices within each patch).
        let mut nrm = vec![Vec3::ZERO; pos.len()];
        for (t, _) in &tris {
            let [a, b, c] = t.map(|i| pos[i as usize]);
            let n = (b - a).cross(c - a);
            for &i in t {
                nrm[i as usize] += n;
            }
        }
        let g = format!("G-{name}");
        let _ = writeln!(s, "<geometry id=\"{g}\" name=\"{name}\"><mesh>");
        let _ = write!(
            s,
            "<source id=\"{g}-pos\"><float_array id=\"{g}-pos-a\" count=\"{}\">",
            pos.len() * 3
        );
        for p in &pos {
            let _ = write!(s, "{:.5} {:.5} {:.5} ", p.x, p.y, p.z);
        }
        let _ = writeln!(s, "</float_array><technique_common><accessor source=\"#{g}-pos-a\" count=\"{}\" stride=\"3\"><param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/></accessor></technique_common></source>", pos.len());
        let _ = write!(
            s,
            "<source id=\"{g}-nrm\"><float_array id=\"{g}-nrm-a\" count=\"{}\">",
            pos.len() * 3
        );
        for n in &nrm {
            let n = n.normalize_or(Vec3::Z);
            let _ = write!(s, "{:.4} {:.4} {:.4} ", n.x, n.y, n.z);
        }
        let _ = writeln!(s, "</float_array><technique_common><accessor source=\"#{g}-nrm-a\" count=\"{}\" stride=\"3\"><param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/></accessor></technique_common></source>", pos.len());
        let _ = writeln!(s, "<vertices id=\"{g}-vtx\"><input semantic=\"POSITION\" source=\"#{g}-pos\"/></vertices>");
        let mut mats: Vec<&String> = tris.iter().map(|(_, m)| m).collect();
        mats.sort();
        mats.dedup();
        for mat in mats {
            let list: Vec<&[u32; 3]> = tris
                .iter()
                .filter(|(_, m)| m == mat)
                .map(|(t, _)| t)
                .collect();
            let _ = write!(s, "<triangles material=\"{mat}\" count=\"{}\"><input semantic=\"VERTEX\" source=\"#{g}-vtx\" offset=\"0\"/><input semantic=\"NORMAL\" source=\"#{g}-nrm\" offset=\"0\"/><p>", list.len());
            for t in list {
                let _ = write!(s, "{} {} {} ", t[0], t[1], t[2]);
            }
            s += "</p></triangles>\n";
        }
        s += "</mesh></geometry>\n";
    }
    s += "</library_geometries>\n<library_visual_scenes><visual_scene id=\"Scene\" name=\"Scene\">\n";
    for name in &order {
        let parts = &by[name];
        let mut mats: Vec<&String> = parts.iter().flat_map(|m| m.mats.iter()).collect();
        mats.sort();
        mats.dedup();
        let _ = write!(s, "<node id=\"{name}\" name=\"{name}\"><instance_geometry url=\"#G-{name}\"><bind_material><technique_common>");
        for m in mats {
            let _ = write!(s, "<instance_material symbol=\"{m}\" target=\"#{m}\"/>");
        }
        s += "</technique_common></bind_material></instance_geometry></node>\n";
    }
    s += "</visual_scene></library_visual_scenes>\n<scene><instance_visual_scene url=\"#Scene\"/></scene>\n</COLLADA>\n";
    s
}

fn materials() -> String {
    use serde_json::json;
    let mut m = serde_json::Map::new();
    let mut add = |name: &str, color: [f32; 4], extra: serde_json::Value| {
        let mut stage = serde_json::Map::new();
        stage.insert("diffuseColor".into(), json!(color));
        if let serde_json::Value::Object(o) = &extra {
            for (k, v) in o {
                if k != "translucent" {
                    stage.insert(k.clone(), v.clone());
                }
            }
        }
        let mut mat = serde_json::Map::new();
        mat.insert("name".into(), json!(name));
        mat.insert("mapTo".into(), json!(name));
        mat.insert("class".into(), json!("Material"));
        mat.insert(
            "Stages".into(),
            json!([serde_json::Value::Object(stage), {}, {}, {}]),
        );
        if extra.get("translucent").is_some() {
            mat.insert("translucent".into(), json!(true));
        }
        m.insert(name.into(), serde_json::Value::Object(mat));
    };
    add(
        "gen2_paint",
        [0.55, 0.06, 0.05, 1.0],
        json!({"instanceDiffuse": true}),
    );
    add("gen2_trim_black", [0.03, 0.03, 0.035, 1.0], json!({}));
    add("gen2_grille_black", [0.02, 0.02, 0.025, 1.0], json!({}));
    add("gen2_liner_black", [0.02, 0.02, 0.02, 1.0], json!({}));
    add("gen2_underbody_black", [0.05, 0.05, 0.05, 1.0], json!({}));
    add(
        "gen2_glass_side",
        [0.08, 0.10, 0.12, 0.5],
        json!({"translucent": true}),
    );
    add(
        "gen2_glass_rear",
        [0.08, 0.10, 0.12, 0.5],
        json!({"translucent": true}),
    );
    add(
        "gen2_windshield",
        [0.10, 0.12, 0.14, 0.4],
        json!({"translucent": true}),
    );
    add("gen2_headlight_lens", [0.85, 0.88, 0.92, 1.0], json!({}));
    add("gen2_taillight_red", [0.60, 0.02, 0.02, 1.0], json!({}));
    add("gen2_mirror_chrome", [0.75, 0.75, 0.78, 1.0], json!({}));
    add("gen2_interior_seat", [0.10, 0.10, 0.11, 1.0], json!({}));
    add("gen2_interior_dash", [0.07, 0.07, 0.08, 1.0], json!({}));
    add("gen2_interior_carpet", [0.06, 0.06, 0.06, 1.0], json!({}));
    add("gen2_interior_steer", [0.04, 0.04, 0.04, 1.0], json!({}));
    add("gen2_interior_door", [0.12, 0.12, 0.13, 1.0], json!({}));
    add(
        "gen2_interior_headliner",
        [0.55, 0.53, 0.50, 1.0],
        json!({}),
    );
    serde_json::to_string_pretty(&serde_json::Value::Object(m)).unwrap()
}
