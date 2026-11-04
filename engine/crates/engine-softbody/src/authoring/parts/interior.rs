//! The cabin's modelled parts: seats with bolsters and headrests, the
//! dashboard with its instrument binnacle and screens, the steering wheel,
//! the centre console, the parcel shelf and the floor.
//!
//! Each rides the structure group it sits on (`seat_fl`, `seat_fr`,
//! `seat_rear`, `dash`, `tunnel`, `column`): seats and the steering wheel
//! rigidly on one triangle of their frame (they never stretch, and never swing
//! out of the cabin on a far binding triangle), the dashboard and console on
//! their members' triangles. Door cards and the headliner follow the body's
//! panels and are cut with them ([`super::design`]).
//!
//! Coordinates: definition coordinates (x across, +x the car's left: the
//! driver's side; y up; z from the rear end).

use super::{Bind, Mesh, Piece};
use crate::authoring::skin::mat;
use glam::{Quat, Vec3};

/// Superellipsoid quads per face edge (scaled by the level of detail).
fn sn() -> usize {
    super::segs(5, 3)
}

/// A frame from a pitch about x (positive tips +y toward +z).
fn pitched(a: f32) -> (Vec3, Vec3, Vec3) {
    let q = Quat::from_rotation_x(a);
    (q * Vec3::X, q * Vec3::Y, q * Vec3::Z)
}

/// A front seat at `x` (cushion centre z `z`): cushion and its bolsters,
/// backrest and bolsters reclined, a headrest on two posts, a pedestal.
fn front_seat(m: &mut Mesh, x: f32, z: f32) {
    let e = sn();
    // Cushion, its front edge a little up.
    let (u, v, w) = pitched(-0.08);
    m.superellipsoid(
        Vec3::new(x, 0.50, z),
        u,
        v,
        w,
        Vec3::new(0.19, 0.055, 0.235),
        3.5,
        e,
        mat::SEAT,
        0.75,
    );
    for s in [-1.0f32, 1.0] {
        m.superellipsoid(
            Vec3::new(x + s * 0.205, 0.535, z - 0.01),
            u,
            v,
            w,
            Vec3::new(0.05, 0.065, 0.22),
            3.0,
            e,
            mat::SEAT,
            0.7,
        );
    }
    // Pedestal and rails.
    m.superellipsoid(
        Vec3::new(x, 0.38, z),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.2, 0.06, 0.2),
        8.0,
        e,
        mat::TRIM,
        0.35,
    );
    // Backrest: reclined 17° about a pivot at the cushion's rear.
    let tilt = 0.30f32;
    let dir = Vec3::new(0.0, tilt.cos(), -tilt.sin());
    let fwd = Vec3::X.cross(dir);
    let pivot = Vec3::new(x, 0.52, z - 0.23);
    let c = pivot + dir * 0.31;
    m.superellipsoid(
        c,
        Vec3::X,
        dir,
        fwd,
        Vec3::new(0.2, 0.31, 0.06),
        3.5,
        e,
        mat::SEAT,
        0.75,
    );
    for s in [-1.0f32, 1.0] {
        m.superellipsoid(
            c + Vec3::X * (s * 0.2) + fwd * 0.03 - dir * 0.04,
            Vec3::X,
            dir,
            fwd,
            Vec3::new(0.05, 0.26, 0.08),
            3.0,
            e,
            mat::SEAT,
            0.7,
        );
    }
    // Headrest on two chrome posts.
    let hc = pivot + dir * 0.74 + fwd * 0.01;
    m.superellipsoid(
        hc,
        Vec3::X,
        dir,
        fwd,
        Vec3::new(0.125, 0.085, 0.055),
        3.0,
        e,
        mat::SEAT,
        0.8,
    );
    for s in [-1.0f32, 1.0] {
        let a = pivot + dir * 0.6 + Vec3::X * (s * 0.07);
        let b = pivot + dir * 0.67 + Vec3::X * (s * 0.07);
        m.tube(&[a, b], fwd, 0.006, 0.006, 8, mat::CHROME, 0.8);
    }
}

/// The rear bench: cushion, two shaped outer places, backrest and three
/// headrests.
fn rear_bench(m: &mut Mesh, z: f32) {
    let e = sn();
    let (u, v, w) = pitched(-0.06);
    m.superellipsoid(
        Vec3::new(0.0, 0.50, z),
        u,
        v,
        w,
        Vec3::new(0.64, 0.06, 0.23),
        4.0,
        e + 2,
        mat::SEAT,
        0.7,
    );
    let tilt = 0.38f32;
    let dir = Vec3::new(0.0, tilt.cos(), -tilt.sin());
    let fwd = Vec3::X.cross(dir);
    let pivot = Vec3::new(0.0, 0.53, z - 0.22);
    let c = pivot + dir * 0.29;
    m.superellipsoid(
        c,
        Vec3::X,
        dir,
        fwd,
        Vec3::new(0.64, 0.29, 0.07),
        4.0,
        e + 2,
        mat::SEAT,
        0.7,
    );
    // The outer places: a soft contour each (seat and back).
    for s in [-1.0f32, 1.0] {
        m.superellipsoid(
            Vec3::new(s * 0.36, 0.545, z + 0.01),
            u,
            v,
            w,
            Vec3::new(0.2, 0.03, 0.2),
            3.0,
            e,
            mat::SEAT,
            0.75,
        );
        m.superellipsoid(
            c + Vec3::X * (s * 0.36) + fwd * 0.045,
            Vec3::X,
            dir,
            fwd,
            Vec3::new(0.2, 0.24, 0.035),
            3.0,
            e,
            mat::SEAT,
            0.75,
        );
    }
    for x in [-0.38f32, 0.0, 0.38] {
        m.superellipsoid(
            pivot + dir * 0.66 + Vec3::X * x,
            Vec3::X,
            dir,
            fwd,
            Vec3::new(0.11, 0.065, 0.045),
            3.0,
            e,
            mat::SEAT,
            0.8,
        );
    }
}

/// A display panel (a grid with texture coordinates, drawn by the renderer as
/// a lit screen): centre `c`, half width along `u`, half height along `v`,
/// facing `n`. `kind` picks its content (0 the instrument cluster, 1 the
/// centre screen) through the u range: kind k uses u ∈ [k, k + 1].
fn display(m: &mut Mesh, c: Vec3, u: Vec3, v: Vec3, hw: f32, hh: f32, kind: f32) {
    let (nu, nv) = (super::segs(12, 2), super::segs(5, 1));
    let n = u.cross(v).normalize();
    let mut ids = Vec::new();
    for j in 0..=nv {
        for i in 0..=nu {
            let (a, b) = (i as f32 / nu as f32, j as f32 / nv as f32);
            let p = c + u * ((a * 2.0 - 1.0) * hw) + v * ((b * 2.0 - 1.0) * hh);
            let id = m.vert(p, n, mat::DISPLAY, 1.0);
            m.uv[id as usize] = [kind + a, b];
            ids.push(id);
        }
    }
    for j in 0..nv {
        for i in 0..nu {
            let k = |i: usize, j: usize| ids[j * (nu + 1) + i];
            m.quad(k(i, j), k(i + 1, j), k(i + 1, j + 1), k(i, j + 1));
        }
    }
}

/// The dashboard: a swept section across the cabin (top pad, face, knee
/// panel), the instrument binnacle and cluster, the centre screen, vents and
/// a trim strip.
fn dashboard(m: &mut Mesh, driver_x: f32) {
    // Section in side view (z, y), round the solid.
    let sec = [
        (3.44, 0.955),
        (3.26, 0.985),
        (3.11, 0.985),
        (3.035, 0.965),
        (3.005, 0.925),
        (3.01, 0.865),
        (3.05, 0.80),
        (3.09, 0.72),
        (3.17, 0.61),
        (3.29, 0.53),
        (3.40, 0.53),
        (3.46, 0.80),
    ];
    let half = 0.74f32;
    let nx = super::segs(24, 8);
    let mut rings = Vec::new();
    for k in 0..=nx {
        let x = -half + 2.0 * half * k as f32 / nx as f32;
        // Rounded ends: the section shrinks toward its centroid at the doors.
        let end = ((x.abs() - (half - 0.05)) / 0.05).clamp(0.0, 1.0);
        let shrink = 1.0 - 0.25 * end * end;
        let (cz, cy) = (3.22, 0.76);
        rings.push(
            sec.iter()
                .map(|&(z, y)| Vec3::new(x, cy + (y - cy) * shrink, cz + (z - cz) * shrink))
                .collect::<Vec<_>>(),
        );
    }
    m.loft(&rings, |_| mat::DASH, 0.6, true);
    let e = sn();
    // Binnacle over the cluster, and the cluster.
    m.superellipsoid(
        Vec3::new(driver_x, 0.99, 3.08),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.19, 0.04, 0.085),
        4.0,
        e,
        mat::DASH,
        0.6,
    );
    let (_, cv, cw) = pitched(0.28);
    display(
        m,
        Vec3::new(driver_x, 0.925, 3.0),
        Vec3::X * -1.0,
        cv,
        0.15,
        0.048,
        0.0,
    );
    let _ = cw;
    // Centre screen: a floating tablet, its face toward the cabin.
    let (su, sv, sw) = pitched(0.2);
    let sc = Vec3::new(0.0, 1.035, 3.05);
    m.superellipsoid(
        sc,
        su,
        sv,
        sw,
        Vec3::new(0.16, 0.095, 0.012),
        8.0,
        e,
        mat::TRIM,
        0.7,
    );
    display(m, sc - sw * 0.0125, -su, sv, 0.15, 0.085, 1.0);
    // Centre stack down to the console.
    m.superellipsoid(
        Vec3::new(0.0, 0.74, 3.03),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.13, 0.17, 0.06),
        6.0,
        e,
        mat::TRIM,
        0.5,
    );
    // Vents (the honeycomb grille material) and a trim strip along the face.
    for x in [-0.6f32, -0.13, 0.13, 0.6] {
        m.superellipsoid(
            Vec3::new(x, 0.905, 3.004),
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.075, 0.022, 0.008),
            8.0,
            e,
            mat::GRILLE,
            0.5,
        );
    }
    let strip: Vec<Vec3> = (0..=16)
        .map(|k| Vec3::new(-0.7 + 1.4 * k as f32 / 16.0, 0.866, 3.008))
        .collect();
    m.tube(&strip, Vec3::Y, 0.003, 0.002, 6, mat::CHROME, 0.8);
}

/// The steering wheel: rim, three spokes, the airbag hub and the column's
/// shroud. `c` the rim's centre, `axis` toward the driver.
fn steering(m: &mut Mesh, c: Vec3, axis: Vec3) {
    let e = sn();
    m.lathe(
        c,
        axis,
        Vec3::X,
        &[
            (0.170, 0.0),
            (0.178, -0.013),
            (0.192, -0.015),
            (0.201, -0.004),
            (0.199, 0.009),
            (0.188, 0.016),
            (0.176, 0.013),
            (0.170, 0.0),
        ],
        super::segs(40, 16),
        |_, _| mat::DASH,
        |_| 0.7,
        false,
    );
    let down = (-(Vec3::Y - axis * Vec3::Y.dot(axis))).normalize();
    let side = axis.cross(down).normalize();
    for d in [side, -side, down] {
        let dd = (d * 0.9 + down * if d == down { 0.1 } else { 0.25 }).normalize();
        let pc = c + dd * 0.105 - axis * 0.01;
        m.superellipsoid(
            pc,
            dd,
            axis.cross(dd).normalize(),
            axis,
            Vec3::new(0.075, 0.024, 0.011),
            4.0,
            e,
            mat::DASH,
            0.7,
        );
    }
    m.superellipsoid(
        c + axis * 0.012,
        side,
        -down,
        axis,
        Vec3::new(0.068, 0.056, 0.028),
        3.0,
        e,
        mat::DASH,
        0.8,
    );
    // Column shroud from the dash.
    let start = c - axis * 0.06;
    let back = start - axis * 0.2;
    m.tube(&[back, start], -down, 0.045, 0.04, 12, mat::TRIM, 0.5);
}

/// The centre console between the front seats: body, armrest, gear selector.
fn console(m: &mut Mesh) {
    let e = sn();
    m.superellipsoid(
        Vec3::new(0.0, 0.46, 2.62),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.11, 0.16, 0.44),
        6.0,
        e + 1,
        mat::DASH,
        0.55,
    );
    m.superellipsoid(
        Vec3::new(0.0, 0.645, 2.32),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.1, 0.035, 0.17),
        4.0,
        e,
        mat::SEAT,
        0.7,
    );
    m.superellipsoid(
        Vec3::new(0.0, 0.625, 2.78),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.09, 0.012, 0.14),
        8.0,
        e,
        mat::TRIM,
        0.6,
    );
    m.superellipsoid(
        Vec3::new(0.0, 0.67, 2.74),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(0.022, 0.04, 0.032),
        3.0,
        e,
        mat::CHROME,
        0.8,
    );
}

/// Where the cabin's fixed points are: the dash cross beam's z at the
/// driver, and the steering column's top (the wheel's hub).
#[derive(Clone, Copy, Debug)]
pub struct Cabin {
    pub dash_z: f32,
    pub column_top: Vec3,
}

impl Cabin {
    /// From the definition's `dash` and `column` members.
    pub fn of(def: &crate::authoring::vehicle::VehicleDef) -> Self {
        let mut c = Cabin {
            dash_z: 3.24,
            column_top: Vec3::new(0.38, 0.93, 2.98),
        };
        for m in &def.members {
            if m.name == "dash" && m.path.len() > 1 {
                c.dash_z = m.path[m.path.len() / 2][2];
            }
            if m.name == "column" {
                if let Some(p) = m.path.last() {
                    c.column_top = Vec3::from_array(*p);
                }
            }
        }
        c
    }
}

/// The cabin's pieces. `group` maps a structure group's name to its index
/// (`Bind::Group`); a missing group leaves its parts out. The dashboard is
/// drawn for a beam at z 3.24 and moved with `cabin.dash_z`.
pub fn interior(group: &dyn Fn(&str) -> Option<u16>, cabin: Cabin) -> Vec<Piece> {
    let dz = cabin.dash_z - 3.24;
    let shifted = move |f: &dyn Fn(&mut Mesh), m: &mut Mesh| {
        let first = m.len();
        f(m);
        for p in m.pos[first..].iter_mut() {
            p.z += dz;
        }
    };
    let mut out = Vec::new();
    let mut add = |name: &str, g: &str, rigid: bool, build: &dyn Fn(&mut Mesh)| {
        let Some(gi) = group(g) else { return };
        let mut m = Mesh::default();
        build(&mut m);
        out.push(Piece {
            name: name.into(),
            mesh: m,
            bind: if rigid {
                Bind::GroupRigid(gi)
            } else {
                Bind::Group(gi)
            },
        });
    };
    add("seat_fl", "seat_fl", true, &|m| front_seat(m, 0.38, 2.62));
    add("seat_fr", "seat_fr", true, &|m| front_seat(m, -0.38, 2.62));
    add("seat_rear", "seat_rear", true, &|m| rear_bench(m, 1.80));
    add("dash", "dash", false, &|m| {
        shifted(&|m| dashboard(m, 0.38), m)
    });
    add("console", "tunnel", false, &console);
    let hub = cabin.column_top + Vec3::new(0.0, 0.02, -0.03);
    add("steering", "column", true, &|m| {
        steering(m, hub, Vec3::new(0.0, 0.42, -0.91).normalize())
    });
    out
}
