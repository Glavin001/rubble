//! Surface contact's frame choices on the planner (a part of `explicit_plan.rs`): the
//! frame in segments with the pass's pipelines between them, and the fast schedules with
//! the pass as pipelines after each step's hits (vehicle-perf-first-principles.md §8.53).

use super::{ExplicitLayout, FramePlanner, WIDE_MAX_VEHICLES};
#[allow(unused_imports)]
use super::{schedule, Kernel};

impl ExplicitLayout {
    /// The wide passes can run this fleet, or could but for its surface contact
    /// ([`Self::wide_ok`] without that term): with the surface pass as pipelines
    /// ([`FramePlanner::set_surface_fast`]) they do. A host builds its wide pipelines
    /// by this.
    pub fn wide_capable(&self) -> bool {
        self.wide_ok || (self.surface && self.spans.len() <= WIDE_MAX_VEHICLES && !self.aero)
    }
}

impl FramePlanner {
    /// Some vehicle has surface contact ([`ExplicitLayout::surface`]).
    pub fn has_surface(&self) -> bool {
        self.surface
    }

    /// A fleet with surface contact on the fast schedules (the wide passes with their
    /// reused pair list, self-collision split with vehicle contact), the surface pass as
    /// its own pipelines ([`Kernel::Surf`]) at the point of each step where the frame
    /// kernel runs it: after the step's self-collision hits are applied, before its
    /// slide nodes and closing barrier. Off (the default here; a host turns it on,
    /// [`schedule::surface_fast_default`]): such a fleet keeps a workgroup a vehicle or a
    /// group with the pass inside it, or the frame in segments. The same results bit
    /// for bit (vehicle-perf-first-principles.md §8.53). No effect without surface
    /// contact.
    pub fn set_surface_fast(&mut self, on: bool) {
        self.surface_fast = on;
    }

    /// The fleet has surface contact and runs it as pipelines on the fast schedules
    /// ([`Self::set_surface_fast`]).
    pub fn surface_piped(&self) -> bool {
        self.surface && self.surface_fast
    }

    /// The wide passes may run this fleet: [`ExplicitLayout::wide_ok`], and with
    /// surface contact only with the pass as pipelines ([`Self::surface_piped`]).
    pub(super) fn wide_ok(&self) -> bool {
        self.wide_ok || (self.wide_capable && self.surface_piped())
    }

    /// A host with the split modules ([`Kernel::FrameSeg`], [`Kernel::Surf`]): frames of
    /// a fleet with surface contact on the frame modules (no contact groups, not wide)
    /// run in segments, a step each, with the surface pass's pipelines between them.
    /// The same results bit for bit; more dispatches, much smaller kernels.
    pub fn set_surface_split(&mut self, on: bool) {
        self.surface_split = on;
    }

    /// This frame runs split (after [`Self::write_params`]), `wide_allowed` as
    /// [`Self::commands`] takes it.
    pub fn split_frames(&self, wide_allowed: bool) -> bool {
        self.surface_split
            && self.surface
            && wide_allowed
            && !self.contact_active()
            && !self.is_wide()
    }
}
