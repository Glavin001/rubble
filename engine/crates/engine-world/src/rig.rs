//! Vehicle damage and soft-body controls: the calls the vehicle and soft-body
//! demo pages make, as World methods.

use crate::{ShapeId, SoftHandle, VehicleHandle, World};
use engine_softbody::SoftSettings;
use engine_vehicle::{FunctionalState, MountLoad};

/// A vehicle's mounted thing to sever or switch off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VehicleDamage {
    /// The wheel falls off as a free body.
    DetachWheel(u32),
    /// The wheel stays on but stops gripping, driving and steering.
    DisableWheel(u32),
    /// The part becomes a free body.
    DetachPart(u32),
    /// The part stays on but stops working (its role's function is lost).
    DisablePart(u32),
}

impl World {
    fn live_vehicle(&self, vehicle: VehicleHandle) -> anyhow::Result<()> {
        anyhow::ensure!(self.vehicles.contains(&vehicle), "no vehicle {}", vehicle.0);
        Ok(())
    }

    fn live_soft(&self, soft: SoftHandle) -> anyhow::Result<()> {
        anyhow::ensure!(
            (soft.0 as usize) < self.sim.soft_host().entries.len(),
            "no soft body {}",
            soft.0
        );
        Ok(())
    }

    /// Sever or switch off a wheel or part. Severed pieces become free bodies
    /// and the vehicle's mass, drive and steering follow.
    pub fn vehicle_damage(
        &mut self,
        vehicle: VehicleHandle,
        damage: VehicleDamage,
    ) -> anyhow::Result<()> {
        self.live_vehicle(vehicle)?;
        let before = self.sim.body_count();
        match damage {
            VehicleDamage::DetachWheel(w) => self.sim.detach_wheel(vehicle, w),
            VehicleDamage::DisableWheel(w) => self.sim.disable_wheel(vehicle, w),
            VehicleDamage::DetachPart(p) => self.sim.detach_part(vehicle, p),
            VehicleDamage::DisablePart(p) => self.sim.disable_part(vehicle, p),
        }
        self.adopt_owned(before);
        self.dirty = true;
        Ok(())
    }

    /// Functional overrides (wheel, drive, steer and brake masks; engine
    /// connected), ANDed with what the vehicle's intact parts allow.
    pub fn vehicle_set_functional_state(
        &mut self,
        vehicle: VehicleHandle,
        state: FunctionalState,
    ) -> anyhow::Result<()> {
        self.live_vehicle(vehicle)?;
        self.sim.set_functional_state(vehicle, state);
        self.dirty = true;
        Ok(())
    }

    /// Load on every breakable mount against its limits, with the peak
    /// ratio since spawn (blocking readback; for damage HUDs and tuning).
    pub fn vehicle_mount_loads(&self, vehicle: VehicleHandle) -> anyhow::Result<Vec<MountLoad>> {
        self.live_vehicle(vehicle)?;
        Ok(self.sim.vehicle_host().read_mount_loads(&self.sim, vehicle))
    }

    /// Soft-solver settings for every soft body (the world needs a soft
    /// arena). `SoftSettings::calibrated_for(hz)` is the demos' tuning.
    pub fn set_soft_settings(&mut self, settings: SoftSettings) -> anyhow::Result<()> {
        anyhow::ensure!(self.sim.has_soft(), "the world has no soft arena");
        self.sim.set_soft_settings(settings);
        self.dirty = true;
        Ok(())
    }

    /// Let soft bodies' skins collide with a rigid shape (walls, poles,
    /// ramps). Soft nodes collide with every rigid shape already; obstacles
    /// add triangle-against-shape contacts so thin shapes can't pass between
    /// nodes.
    pub fn soft_add_obstacle(&mut self, shape: ShapeId) -> anyhow::Result<()> {
        let s = self.live_shape(shape)?;
        anyhow::ensure!(self.sim.has_soft(), "the world has no soft arena");
        let collider = self.shapes[s].collider;
        self.sim
            .add_soft_obstacle(collider)
            .map_err(|e| anyhow::anyhow!("soft obstacle: {e}"))?;
        self.dirty = true;
        Ok(())
    }

    /// Is this soft body riding its rigid core (rigid mode) right now?
    /// Blocking readback.
    pub fn soft_is_rigid(&self, soft: SoftHandle) -> anyhow::Result<bool> {
        self.live_soft(soft)?;
        Ok(self.sim.read_soft_rigid(soft))
    }
}
