//! The last yield rules of the rigger (split from `rig_with`): the doors' side
//! intrusion beams, the A-pillar's chords, the floor's cross members and the
//! B-pillar.

use super::*;

/// The doors' intrusion beams, the A-pillar's chords, the floor and the
/// B-pillar's yield, by the physics block's multiples (and the `GEN3_*`
/// environment's).
pub(super) fn closing_yields(r: &mut Rig, ph: &Physics, axle_f: f32) {
    let body = Part::Body;
    let to_jbeam = |g: Vec3| Vec3::new(g.x, axle_f - g.z, g.y);
    let (apillar_yield, intrusion_yield) = (ph.apillar_yield, ph.intrusion_yield);
    let (cross_member_yield, floor_pan_yield, bpillar_upper_yield, bpillar_lower_yield) = (
        ph.cross_member_yield,
        ph.floor_pan_yield,
        ph.bpillar_upper_yield,
        ph.bpillar_lower_yield,
    );
    let env = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(d)
    };
    // The doors' side intrusion beams: `GEN3_INTRUSION_Y` times their yield.
    let iy = env("GEN3_INTRUSION_Y", intrusion_yield);
    for b in &mut r.car.beams {
        if b.role == Role::Intrusion && b.deform.is_finite() {
            b.deform *= iy;
        }
    }
    // The A-pillar's chords (hinge pillar top to the header, outer and inner
    // walls): a hot-stamped section, its yield `GEN3_APILLAR_Y` times the
    // greenhouse's.
    let ay = env("GEN3_APILLAR_Y", apillar_yield);
    let chord_ends = ["hp2", "hpi1", "rr0", "rri0"];
    for b in &mut r.car.beams {
        let na = &r.car.nodes[b.a].name;
        let nb = &r.car.nodes[b.b].name;
        let is = |n: &str| {
            chord_ends
                .iter()
                .any(|c| n.len() == c.len() + 1 && n.starts_with(c))
        };
        let low = |n: &str| n.starts_with("hp");
        if b.role == Role::PillarA && is(na) && is(nb) && low(na) != low(nb) && b.deform.is_finite()
        {
            b.deform *= ay;
        }
    }
    // The floor: cross members at the stations that carry one (sill to the
    // floor rail and the tunnel), the floor pan elsewhere. A pole at the
    // B-pillar's station meets the sill between two cross members and bends
    // it; a wide barrier pushes the sill at every station and loads them
    // end on.
    if std::env::var("GEN3_OLD_SILL").is_err() {
        let (cmy, pany) = (
            env("GEN3_CM_Y", cross_member_yield),
            env("GEN3_PAN_Y", floor_pan_yield),
        );
        // (station, kind): kind 0 sill, 1 floor rail or tunnel.
        let station = |n: &str| -> Option<(usize, u8)> {
            let t = n.trim_end_matches(['L', 'R']);
            for (pfx, kind) in [("slo", 0u8), ("slt", 0), ("slb", 0), ("flr", 1), ("tn", 1)] {
                if let Some(k) = t.strip_prefix(pfx) {
                    if let Ok(k) = k.parse::<usize>() {
                        return Some((k, kind));
                    }
                }
            }
            None
        };
        let cm = |k: usize| k != 2;
        for b in &mut r.car.beams {
            if b.part != body || !b.deform.is_finite() || b.break_group.is_some() {
                continue;
            }
            let (Some((ka, ta)), Some((kb, tb))) = (
                station(&r.car.nodes[b.a].name),
                station(&r.car.nodes[b.b].name),
            ) else {
                continue;
            };
            // The sill's own section at a station (its outer wall onto its
            // top and bottom): `GEN3_SILL_SEC_Y`.
            if ta == 0 && tb == 0 {
                if ka == kb {
                    b.deform *= env("GEN3_SILL_SEC_Y", 1.0);
                } else {
                    b.deform *= env("GEN3_LONG_Y", 1.0);
                }
                continue;
            }
            if ta == 1 && tb == 1 && ka != kb {
                b.deform *= env("GEN3_LONG_Y", 1.0);
                continue;
            }
            // Across the floor: sill to floor, or floor rail to tunnel.
            let across = ta != tb
                || (ta == 1
                    && (r.car.nodes[b.a].name.starts_with("tn")
                        != r.car.nodes[b.b].name.starts_with("tn")));
            if !across {
                continue;
            }
            // The footwell's floor (toe board to the seat's front cross
            // member) keeps its own yield: the frontal's feet intrusion.
            b.deform *= if ka == kb && cm(ka) {
                cmy
            } else if ka.max(kb) <= 1 {
                1.0
            } else {
                pany
            };
        }
        // The B-pillar: hot-stamped above the door's crease, the tailored
        // blank's soft zone below it.
        let (bu, bl) = (
            env("GEN3_BP_UP", bpillar_upper_yield),
            env("GEN3_BP_LO", bpillar_lower_yield),
        );
        let split = to_jbeam(Vec3::new(0.0, 0.60, 0.0)).z;
        for b in &mut r.car.beams {
            if b.role != Role::PillarB || !b.deform.is_finite() || b.break_group.is_some() {
                continue;
            }
            let mid = 0.5 * (r.car.nodes[b.a].p + r.car.nodes[b.b].p);
            b.deform *= if mid.z > split { bu } else { bl };
        }
    }
}
