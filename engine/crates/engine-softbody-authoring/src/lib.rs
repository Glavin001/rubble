//! Vehicle authoring for [`engine_softbody`]: the generated cars (parts, rigs and
//! their visual skins, [`authoring`]), imported vehicles' visual meshes ([`flexmesh`],
//! [`flexskin`]), glTF export ([`glb`]) and the vehicle comparisons
//! ([`vehicle_compare`]).
//!
//! Kept apart from the runtime model so that the plans, GPU hosts and the World,
//! which build on `engine-softbody`, do not compile (or recompile after an edit) the
//! authoring code: about 50k lines that only the tools, tests and the showcase need.

pub mod paint_overlap;
pub mod authoring;
pub mod flexmesh;
pub mod flexskin;
pub mod glb;
pub mod vehicle_compare;
