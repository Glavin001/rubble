//! Schema constants (`learned-impact/1`).

pub const SCHEMA_VERSION: &str = "learned-impact/1";
pub const K: usize = 4;
pub const CHUNK_STATE_DIM: usize = 12;
pub const LOCAL6_DIM: usize = 6;
pub const JOINT_STATE_DIM: usize = 22;
pub const BOND_REBAR_DIM: usize = 5;

/// Fail class: 0 = none, 1..=K = time bin.
pub type FailClass = i32;
/// Fail mode label: 0 none, 1 tension, 2 shear, 3 compression, 4 buckling, 5 rebar.
pub type FailMode = i32;
