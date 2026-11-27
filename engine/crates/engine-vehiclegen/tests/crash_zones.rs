//! Every registered design's parts state their crash zone (`engine_vehiclegen::crash`):
//! a body-in-white member type without one fails here, so a new member never becomes
//! "other" silently. (A mechanical part cannot be made without one.)

use engine_vehiclegen::crash::Zone;
use engine_vehiclegen::{design_from_json, design_names};

#[test]
fn every_member_and_part_of_every_design_has_a_zone() {
    for n in design_names() {
        let d = design_from_json(n, "{}").unwrap_or_else(|e| panic!("{n}: {e}"));
        let st = d.structure().unwrap_or_else(|| panic!("{n}: no structure"));
        let missing: Vec<&str> = st.members.iter().filter(|m| m.zone().is_none()).map(|m| m.kind).collect();
        assert!(missing.is_empty(), "{n}: member types with no crash zone (crash::member_zone): {missing:?}");
        for node in &st.nodes {
            assert!(st.members[node.member].zone().is_some(), "{n}: shell node {} has no zone", node.name);
        }
        let parts = d.mechanical(&st);
        // the load paths the measurements read are all there
        let has = |z: Zone| st.members.iter().any(|m| m.zone() == Some(z)) || parts.iter().any(|p| p.zone == z);
        for z in [Zone::FrontLower, Zone::FrontUpper, Zone::Cabin, Zone::Rear, Zone::Wheel] {
            assert!(has(z), "{n}: nothing in {z:?}");
        }
    }
}
