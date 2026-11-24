//! The vehicle contact status counters' errors.

use super::VV_ROOM_MAX;

/// The contact error the status counters (outside, overflow, unconverged, too big)
/// report.
pub fn contact_error(w: &[i32]) -> Result<(), String> {
    if w[0] > 0 {
        return Err(format!(
            "vehicle {} left its frame bounds (a node moved past the travel the frame's \
             contact plan allowed, so a contact may have been missed); raise \
             ExplicitBeamsGpu::set_contact_margin",
            w[0] - 1
        ));
    }
    // (The overflow word: 1 the pair list; from 2 on the limits of the owners' state by
    // vehicle, the kernel's FL_ERR_*, the largest raised.)
    match w[1] {
        0 => {}
        2 => {
            return Err(format!(
                "the owners' candidates passed the fleet-wide run's room ({VV_ROOM_MAX} times the                  fleet's contact points: too many vehicles in contact with one another at                  once; that step's contact pass was cut short)"
            ))
        }
        3 => {
            return Err(
                "an owner's hits passed its part of the fleet-wide hit list (twice its                  partners' contact points; the hits past it were not applied)"
                    .into(),
            )
        }
        5 => {
            return Err(
                "an owner's edge contact hits passed its edge hit list (its skin edges and contact \
                 points; the hits past it were not applied)"
                    .into(),
            )
        }
        4 => {
            return Err(format!(
                "a vehicle had more than {VV_ROOM_MAX} partners with two-body contact at once                  (the two-body point counts hold {VV_ROOM_MAX} owners a node)"
            ))
        }
        _ => return Err("the overlapping-pair list overflowed".into()),
    }
    if w[2] != 0 {
        return Err("the contact grouping did not converge".into());
    }
    if w[3] != 0 {
        return Err(
            "a contact group had more than 64 vehicles (they ran without contact that frame)"
                .into(),
        );
    }
    Ok(())
}
