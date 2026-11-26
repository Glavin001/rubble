//! The lab's options as built (split from `mod.rs`).

use super::*;

impl Default for LabOptions {
    fn default() -> Self {
        Self {
            scene: Scene::Flat,
            surface_contact: false,
            edge_contact: false,
            vehicle_contact_range: None,
            own_travel_reach: true,
            piece_labels: true,
            piece_windows: true,
            self_split: false,
            wide: true,
            wide_list: true,
            wide_fused: true,
            group_split: true,
            continuous: false,
            debris_ttl: Some(LAB_DEBRIS_TTL),
            parked: false,
            cars: 1,
            capacity: CouplingCapacity {
                bodies: 256,
                shapes: 256,
                planes: 4096,
            },
            mountain: MountainSpec::default(),
            sleep: true,
            vehicle_sleep: false,
            vehicle_sleep_cfg: VehicleSleep::default(),
            bit_exact: engine_webgpu::bit_exact_default(),
            parts: Vec::new(),
            derby: derby::DerbySpec::default(),
            others: Vec::new(),
            structures: Vec::new(),
            // Off by default (explicit-vehicles-in-world.md §13, "Open: Chrome").
            wheels_hit_bodies: false,
        }
    }
}

/// Continuous contact on the lab's vehicles: on the split group path (group split on, self
/// split and the wide passes off, which would step the cars elsewhere), or off (those back as
/// the lab's defaults).
pub(super) fn continuous_on(x: &mut engine_webgpu::ExplicitBeamsGpu, on: bool) {
    x.set_continuous(on);
    if on {
        x.set_group_split(true);
        x.set_self_split(false);
        let _ = x.set_wide(false);
    }
}
