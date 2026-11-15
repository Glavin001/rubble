//! The engine bay under the bonnet: the engine (block, head and cam cover,
//! intake plenum), the radiator, the inner aprons with their strut towers and
//! the bulkhead, so an open or torn-off bonnet shows a bay, not a void.
//!
//! Each rides the structure group it is: the engine rigidly on the
//! `powertrain` block, the radiator rigidly on its `radiator` plate, the
//! aprons, towers and bulkhead on their members' triangles (`apron`,
//! `tower_f`, `firewall`; `_r` the right-hand ones). Their sizes come from
//! those groups' nodes, so they sit where the physics puts the parts.
//!
//! Coordinates: definition coordinates (x across, +x left; y up; z from the
//! rear end).

use super::{Bind, Mesh, Piece};
use crate::authoring::skin::mat;
use glam::Vec3;

/// A group's node positions (definition coordinates).
pub type Nodes<'a> = &'a dyn Fn(&str) -> Option<Vec<Vec3>>;

/// The bay's pieces. `group` maps a structure group to its binding index,
/// `nodes` to its nodes' positions.
pub fn engine_bay(group: &dyn Fn(&str) -> Option<u16>, nodes: Nodes) -> Vec<Piece> {
    let bounds = |n: &str| -> Option<(Vec3, Vec3)> {
        let ps = nodes(n)?;
        let lo = ps.iter().copied().fold(Vec3::splat(f32::MAX), Vec3::min);
        let hi = ps.iter().copied().fold(Vec3::splat(f32::MIN), Vec3::max);
        (!ps.is_empty()).then_some((lo, hi))
    };
    let mut out = Vec::new();
    let e = super::segs(5, 3);
    if let (Some(g), Some((lo, hi))) = (group("powertrain"), bounds("powertrain")) {
        let c = 0.5 * (lo + hi);
        let h = 0.5 * (hi - lo);
        let mut m = Mesh::default();
        // Block and head (a transverse four: long across the car).
        m.superellipsoid(
            Vec3::new(c.x, c.y - 0.03, c.z),
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(h.x * 0.92, h.y * 0.78, h.z * 0.62),
            6.0,
            e,
            mat::ENGINE,
            0.45,
        );
        // Cam cover (black, ribbed look from its rounded edges) on the head.
        m.superellipsoid(
            Vec3::new(c.x, c.y + h.y * 0.80, c.z - h.z * 0.12),
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(h.x * 0.78, h.y * 0.16, h.z * 0.36),
            5.0,
            e,
            mat::TRIM,
            0.6,
        );
        // Intake plenum and runners in front of the head.
        m.superellipsoid(
            Vec3::new(c.x, c.y + h.y * 0.45, c.z + h.z * 0.62),
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(h.x * 0.7, h.y * 0.18, h.z * 0.16),
            4.0,
            e,
            mat::ENGINE,
            0.5,
        );
        // Gearbox at the left end.
        m.superellipsoid(
            Vec3::new(c.x + h.x * 1.02, c.y - h.y * 0.25, c.z - h.z * 0.1),
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(h.x * 0.22, h.y * 0.5, h.z * 0.5),
            3.0,
            e,
            mat::ENGINE,
            0.4,
        );
        out.push(Piece {
            name: "engine".into(),
            mesh: m,
            bind: Bind::GroupRigid(g),
        });
    }
    if let (Some(g), Some((lo, hi))) = (group("radiator"), bounds("radiator")) {
        let c = 0.5 * (lo + hi);
        let h = (0.5 * (hi - lo)).max(Vec3::splat(0.02));
        let mut m = Mesh::default();
        m.block(
            c,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(h.x, h.y, 0.025),
            mat::GRILLE,
            0.5,
        );
        out.push(Piece {
            name: "radiator".into(),
            mesh: m,
            bind: Bind::GroupRigid(g),
        });
    }
    // Aprons: the bay's side walls, from 3 cm under the apron rail (its
    // nodes' height along it, so the wall stays under the bonnet) down to the
    // frame rail's height, with the strut tower's dome on them.
    for (ap, tw, rail) in [
        ("apron", "tower_f", "rail_f"),
        ("apron_r", "tower_f_r", "rail_f_r"),
    ] {
        let (Some(g), Some((lo, hi))) = (group(ap), bounds(ap)) else {
            continue;
        };
        let sx = if lo.x + hi.x > 0.0 { 1.0f32 } else { -1.0 };
        let x = if sx > 0.0 { lo.x } else { hi.x } - sx * 0.01;
        let mut along = nodes(ap).unwrap_or_default();
        along.sort_by(|a, b| a.z.total_cmp(&b.z));
        let top_at = |z: f32| -> f32 {
            let k = along
                .partition_point(|p| p.z < z)
                .clamp(1, along.len().max(2) - 1);
            let (a, b) = (along[k - 1], along[k.min(along.len() - 1)]);
            let t = if (b.z - a.z).abs() > 1e-4 {
                ((z - a.z) / (b.z - a.z)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            a.y + (b.y - a.y) * t - 0.03
        };
        let y_bot = bounds(rail).map_or(lo.y - 0.3, |(l, h)| 0.5 * (l.y + h.y));
        let (z0, z1) = (lo.z, hi.z);
        let mut m = Mesh::default();
        let (nz, ny) = (super::segs(14, 6), super::segs(5, 3));
        let mut rows = Vec::new();
        for j in 0..=ny {
            let mut row = Vec::new();
            for i in 0..=nz {
                let z = z0 + (z1 - z0) * i as f32 / nz as f32;
                let yt = top_at(z).max(y_bot + 0.02);
                let y = y_bot + (yt - y_bot) * j as f32 / ny as f32;
                row.push(m.vert(
                    Vec3::new(x, y, z),
                    Vec3::new(-sx, 0.0, 0.0),
                    mat::INNER,
                    0.4,
                ));
            }
            rows.push(row);
        }
        m.strips(&rows, false);
        let mut pieces = vec![Piece {
            name: format!("{ap}_wall"),
            mesh: m,
            // Its apron's nodes are along its top; the frame rail's along
            // its foot.
            bind: Bind::Groups([
                g,
                group(rail).unwrap_or(u16::MAX),
                group(tw).unwrap_or(u16::MAX),
            ]),
        }];
        if let (Some(gt), Some((tl, th))) = (group(tw), bounds(tw)) {
            let tc = 0.5 * (tl + th);
            let mut t = Mesh::default();
            t.superellipsoid(
                Vec3::new(tc.x, th.y - 0.04, tc.z),
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::new(0.09, 0.07, 0.09),
                2.5,
                e,
                mat::INNER,
                0.45,
            );
            pieces.push(Piece {
                name: format!("{tw}_dome"),
                mesh: t,
                bind: Bind::GroupRigid(gt),
            });
        }
        out.extend(pieces);
    }
    // The bulkhead between the bay and the cabin.
    if let (Some(g), Some((lo, hi))) = (group("firewall"), bounds("firewall")) {
        let z = hi.z + 0.005;
        let mut m = Mesh::default();
        let (nx, ny) = (super::segs(12, 5), super::segs(5, 3));
        let mut rows = Vec::new();
        for j in 0..=ny {
            let y = lo.y + (hi.y - lo.y) * j as f32 / ny as f32;
            let mut row = Vec::new();
            for i in 0..=nx {
                let x = lo.x + (hi.x - lo.x) * i as f32 / nx as f32;
                row.push(m.vert(Vec3::new(x, y, z), Vec3::Z, mat::INNER, 0.35));
            }
            rows.push(row);
        }
        m.strips(&rows, false);
        out.push(Piece {
            name: "bulkhead".into(),
            mesh: m,
            bind: Bind::Group(g),
        });
    }
    out
}
