//! Raycast wheels that see other vehicles' triangles (`DriveConfig::see_bodies`): the
//! partner scene set at a step's start and the reactions taken back after its contact pass.

use super::ExplicitBeamsWorld;
use crate::explicit_beams::{ExplicitBeams, NODE_IMMOVABLE};
use glam::Vec3;

impl ExplicitBeamsWorld {
    /// Raycast wheels on other vehicles (`DriveConfig::see_bodies`, with vehicle contact
    /// on): at a step's start, every vehicle's live collision triangles as they stand
    /// (World positions, velocities, inverse masses) for the wheels' rays of every vehicle
    /// stepped (`awake`: those; None every one). False: no vehicle's wheels look (nothing
    /// set, every step as without it).
    pub(super) fn wheels_begin(&mut self, awake: Option<&[bool]>) -> bool {
        let nv = self.vehicles.len();
        let on = |v: &ExplicitBeams| v.drive.as_ref().is_some_and(|d| d.config.see_bodies && !d.config.wheels.is_empty());
        if !self.contact || nv < 2 || !self.vehicles.iter().any(on) {
            return false;
        }
        let mut sc = crate::explicit_drive::PartnerScene::default();
        for (j, v) in self.vehicles.iter().enumerate() {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for n in &v.nodes {
                let x = v.origin + n.pos;
                lo = lo.min(x);
                hi = hi.max(x);
            }
            let t0 = sc.tris.len() as u32;
            for (t, tri) in v.triangles.iter().enumerate() {
                let dead = v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t);
                if dead {
                    continue;
                }
                let n = tri.nodes.map(|i| v.nodes[i as usize]);
                sc.tris.push(crate::explicit_drive::PartnerTri {
                    vehicle: j as u32,
                    nodes: tri.nodes,
                    x: n.map(|n| v.origin + n.pos),
                    v: n.map(|n| n.vel),
                    im: n.map(|n| if n.has(NODE_IMMOVABLE) { 0.0 } else { 1.0 / n.mass }),
                });
            }
            sc.vehicles.push((lo, hi, t0, sc.tris.len() as u32));
        }
        let sc = std::sync::Arc::new(sc);
        for (j, v) in self.vehicles.iter_mut().enumerate() {
            if awake.is_some_and(|a| !a[j]) {
                continue;
            }
            if let Some(d) = v.drive.as_mut().filter(|d| d.config.see_bodies) {
                d.partners = Some(sc.clone());
                d.owner = j as u32;
            }
        }
        true
    }

    /// After a step's contact pass: the wheels' reactions on other vehicles' triangles
    /// onto those nodes' contact sums (consumed by the next step's node pass, as vehicle
    /// contact's), and the scene dropped.
    pub(super) fn wheels_end(&mut self) {
        let mut all = Vec::new();
        for v in &mut self.vehicles {
            if let Some(d) = v.drive.as_mut() {
                d.partners = None;
                all.append(&mut d.reactions);
            }
        }
        for r in all {
            let v = &mut self.vehicles[r.vehicle as usize];
            for k in 0..3 {
                if r.w[k] > 0.0 {
                    crate::explicit_drive::apply_contact(v, r.nodes[k], r.force * r.w[k]);
                }
            }
        }
    }

}
