//! Stable public handles. Storage indices may change under compaction; generation invalidates stale ids.

use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Handle {
    pub index: u32,
    pub generation: u32,
}

impl Handle {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn is_valid(self) -> bool {
        self.generation != 0 && self.index != u32::MAX
    }
}

pub type BodyId = Handle;
pub type ColliderId = Handle;
pub type JointId = Handle;
