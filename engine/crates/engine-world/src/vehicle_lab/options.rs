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
