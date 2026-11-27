//! Which beams a torn collision triangle counts for the next contact pass, as the kernels see
//! it (`explicit_beams_triangles.slang`'s `triangleDead`: a beam's break mark set and not
//! pending). The kernel marks a beam that breaks under its own load pending (`s1.w = 1`) until
//! the next beam update, and marks the members of a break group set off this pass at their
//! update when they come after the group's master in beam order (`groupDrops`), else at the
//! next. Continuous contact's CPU surface (`explicit_contact::dat`) reads the same.

use super::ExplicitBeams;

/// The beam pass's record of how its beams broke ([`ExplicitBeams::torn`]).
#[derive(Clone, Debug, Default)]
pub struct Torn {
    /// Per beam: torn for the next contact pass (empty before the first beam pass).
    pub now: Vec<bool>,
    base: Vec<bool>,
    cascaded: Vec<u32>,
    /// The beam the pass is at (a break group's master when it sets the group off).
    pub(crate) at: u32,
}

impl Torn {
    pub(crate) fn begin(&mut self, beams: &[super::Beam]) {
        self.base.clear();
        self.base.extend(beams.iter().map(|b| b.broken));
        self.cascaded.clear();
        self.cascaded.resize(beams.len(), u32::MAX);
        self.at = u32::MAX;
    }

    /// Beam `k` broken by a break group set off at the pass's current beam.
    pub(crate) fn cascaded(&mut self, k: usize) {
        if let Some(c) = self.cascaded.get_mut(k) {
            *c = self.at;
        }
    }

    pub(crate) fn end(&mut self) {
        self.now = (0..self.base.len()).map(|k| self.base[k] || (self.cascaded[k] != u32::MAX && k as u32 > self.cascaded[k])).collect();
    }
}

impl ExplicitBeams {
    /// Whether collision triangle `t` is torn for the contact pass, as the kernels see it.
    pub fn tri_torn(&self, t: usize) -> bool {
        self.tri_dead.get(t).copied().unwrap_or(false)
            || self.tri_breaks.get(t).is_some_and(|l| l.iter().any(|&j| j != crate::explicit_jbeam_ext::NO_BEAM && self.torn.now.get(j as usize).copied().unwrap_or(false)))
    }
}
