//! A portable World state: one moment of a World with explicit vehicles that a host with
//! no scene, no vehicles and no GPU API of the host that saved it replays: the rigid
//! bodies, the coupling's carried state, the explicit vehicles' portable state
//! ([`crate::explicit_plan::portable`]) and the drive inputs of the steps that follow.
//! The scenario matrix's `--world` rows on CUDA (`engine-cuda`'s `world_replay`,
//! `CudaWorld::from_portable`); written by the wgpu World (`vehicle_lab --save-world`).
//!
//! The rigid part is the scene, not the solver's caches: bodies (current and previous
//! rows), colliders, the hull arena, joints, materials, parameters and capacity. A host
//! that loads it starts its contacts cold (no manifolds, no warm start): the first
//! replayed steps find them again, so a replay is warmed up before it is timed, and its
//! bodies follow the saved World's only as closely as a cold start allows.
//!
//! File: [`MAGIC`], the header's length (u32 LE), the header as JSON, then each blob's
//! bytes in the header's order.

use crate::explicit_plan::coupling::world as cw;
use crate::explicit_plan::portable::Portable;
use crate::explicit_plan::CouplingLayout;
use engine_schema::{AvbdParams, WorldCapacity};

pub const MAGIC: &[u8; 8] = b"RUBBLEWF";
pub const VERSION: u32 = 1;

/// The rigid scene: counts and settings; its records are blobs ([`BLOBS`]).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RigidHeader {
    pub capacity: WorldCapacity,
    pub params: AvbdParams,
    pub body_count: u32,
    pub collider_count: u32,
    /// Joint slots in use (records `0..joint_count`), and the released ones among them.
    pub joint_count: u32,
    pub free_joints: Vec<u32>,
    pub sleep_island_rounds: u32,
    /// Rigid steps the saving host had run.
    pub steps_done: u32,
    /// The saving host's step modes, for the record (a loading host runs its own).
    pub host_modes: String,
}

/// The coupling's carried state (`explicit_plan::coupling::world`); its work buffer is a
/// blob.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CouplingHeader {
    pub layout: CouplingLayout,
    pub work_layout: cw::WorkLayout,
    pub categories: u16,
    pub mask: u16,
    /// The proxy plan's reach (m), the wheels' rays' included.
    pub margin: f32,
    pub hull_geom: bool,
    /// The piece windows' label and box records ((0, 0): one window per vehicle).
    pub piece_records: (u32, u32),
    pub clock: cw::Clock,
    pub sleep_mask: Vec<u32>,
    pub any_awake: bool,
    /// The proxied list's indirect arguments.
    pub list_args: [u32; 4],
    /// The saving host planned its last window from the World broadphase's resting
    /// table (`cpl_query`) or by the scan of every collider: the same proxies.
    pub planned_by_query: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Header {
    pub version: u32,
    /// Simulated time (s) of the moment.
    pub t: f64,
    /// What it is (e.g. the matrix row and the host that saved it).
    pub note: String,
    /// The World's step (s).
    pub dt: f32,
    pub rigid: RigidHeader,
    pub coupling: CouplingHeader,
    /// Per World step after the moment, per vehicle its drive input
    /// (`explicit_plan::drive::input_words`; `None`: none set for it).
    pub inputs: Vec<Vec<Option<[u32; 4]>>>,
    /// Each blob's name and bytes, in the file's order.
    pub blobs: Vec<(String, u64)>,
}

/// The blobs a file holds: the bodies' current rows and their previous rows
/// (`BodyState`, `body_count` each), the colliders, the hull arena's records, the joint
/// records, the material table, the coupling's work buffer, and the explicit vehicles'
/// portable state (a whole `RUBBLEPF` file).
pub const BLOBS: [&str; 8] = [
    "bodies",
    "bodies_previous",
    "colliders",
    "hulls",
    "joints",
    "materials",
    "work",
    "explicit",
];

#[derive(Clone, Debug)]
pub struct PortableWorld {
    pub header: Header,
    /// [`BLOBS`], in order.
    pub blobs: Vec<Vec<u8>>,
}

impl PortableWorld {
    /// `blobs`: [`BLOBS`] in order; `header.blobs` is filled here.
    pub fn new(mut header: Header, blobs: Vec<Vec<u8>>) -> Self {
        assert_eq!(
            blobs.len(),
            BLOBS.len(),
            "a portable World holds every blob"
        );
        header.version = VERSION;
        header.blobs = BLOBS
            .iter()
            .zip(&blobs)
            .map(|(n, b)| (n.to_string(), b.len() as u64))
            .collect();
        Self { header, blobs }
    }

    pub fn blob(&self, name: &str) -> &[u8] {
        let k = BLOBS
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("no blob {name}"));
        &self.blobs[k]
    }

    /// The explicit vehicles' portable state.
    pub fn explicit(&self) -> Result<Portable, String> {
        Portable::from_bytes(self.blob("explicit")).map_err(|e| format!("explicit: {e}"))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let head = serde_json::to_vec(&self.header).expect("the header serialises");
        let mut out =
            Vec::with_capacity(12 + head.len() + self.blobs.iter().map(Vec::len).sum::<usize>());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(head.len() as u32).to_le_bytes());
        out.extend_from_slice(&head);
        for b in &self.blobs {
            out.extend_from_slice(b);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err("not a portable World state (no RUBBLEWF magic)".into());
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let head = bytes
            .get(12..12 + n)
            .ok_or("the header runs past the file")?;
        let header: Header = serde_json::from_slice(head).map_err(|e| format!("header: {e}"))?;
        if header.version != VERSION {
            return Err(format!(
                "version {} (this build reads {VERSION})",
                header.version
            ));
        }
        let names: Vec<&str> = header.blobs.iter().map(|b| b.0.as_str()).collect();
        if names != BLOBS {
            return Err(format!("blobs {names:?}, not {BLOBS:?}"));
        }
        let mut at = 12 + n;
        let mut blobs = Vec::with_capacity(BLOBS.len());
        for (name, len) in &header.blobs {
            let b = bytes
                .get(at..at + *len as usize)
                .ok_or_else(|| format!("blob {name} runs past the file"))?;
            blobs.push(b.to_vec());
            at += *len as usize;
        }
        if at != bytes.len() {
            return Err(format!("{} bytes past the last blob", bytes.len() - at));
        }
        let w = Self { header, blobs };
        w.check()?;
        Ok(w)
    }

    /// The blobs are the sizes the header's counts say.
    fn check(&self) -> Result<(), String> {
        let r = &self.header.rigid;
        let body = std::mem::size_of::<engine_schema::BodyState>();
        let collider = std::mem::size_of::<engine_schema::Collider>();
        let joint = std::mem::size_of::<engine_schema::Joint>();
        let want = [
            ("bodies", r.body_count as usize * body),
            ("bodies_previous", r.body_count as usize * body),
            ("colliders", r.collider_count as usize * collider),
            ("joints", r.joint_count as usize * joint),
            ("work", self.header.coupling.work_layout.bytes() as usize),
        ];
        for (name, bytes) in want {
            if self.blob(name).len() != bytes {
                return Err(format!(
                    "blob {name} is {} bytes, the header's counts say {bytes}",
                    self.blob(name).len()
                ));
            }
        }
        if self.blob("hulls").len() % 16 != 0 {
            return Err("blob hulls is not whole records".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_plan::portable::HostSwitches;
    use crate::explicit_plan::{CouplingCapacity, ExplicitUpload, FramePlanner, BIND_COUNT};

    /// A small World through the file and back, and a file with a blob of the wrong size
    /// refused.
    #[test]
    fn portable_world_round_trips() {
        use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
        let text = "nodes\n0, 0.0, 0.0, 0.0\n1, 0.3, 0.0, 0.0\n2, 0.0, 0.3, 0.0\nbeams\n0, 1\n1, 2\n2, 0\nend\n";
        let car = ExplicitBeams::from_truck(
            text,
            glam::Vec3::Y,
            glam::Vec3::ZERO,
            ExplicitBeamsSettings::default(),
        )
        .expect("the truck parses");
        let cap = CouplingCapacity {
            bodies: 4,
            shapes: 8,
            planes: 64,
        };
        let up = ExplicitUpload::new(std::slice::from_ref(&car), Some(cap));
        let planner = FramePlanner::new(&up.layout);
        let bindings: Vec<Vec<u8>> = (0..BIND_COUNT as u32).map(|b| up.bytes(b)).collect();
        let explicit = Portable::new(
            1.5,
            "test",
            up.layout.clone(),
            planner.settings(),
            HostSwitches::default(),
            bindings,
        );
        let capacity = WorldCapacity {
            max_bodies: 64,
            max_colliders: 64,
            max_contacts: 256,
            max_joints: 8,
            max_candidates: 256,
            max_colors: 8,
            max_commands: 8,
            max_hull_vertices: 64,
        };
        let work_layout = cw::WorkLayout::new(1, capacity.max_bodies, cap.shapes);
        let body = vec![7u8; 2 * std::mem::size_of::<engine_schema::BodyState>()];
        let header = Header {
            version: VERSION,
            t: 1.5,
            note: "test".into(),
            dt: 1.0 / 60.0,
            rigid: RigidHeader {
                capacity,
                params: AvbdParams::default(),
                body_count: 2,
                collider_count: 1,
                joint_count: 0,
                free_joints: Vec::new(),
                sleep_island_rounds: 8,
                steps_done: 90,
                host_modes: String::new(),
            },
            coupling: CouplingHeader {
                layout: up.layout.coupling.expect("coupling room"),
                work_layout,
                categories: 0xFFFF,
                mask: 0xFFFF,
                margin: 0.1,
                hull_geom: false,
                piece_records: (0, 0),
                clock: cw::Clock {
                    carry: 0.25,
                    emit_pending: true,
                    emit_wakes: true,
                    windows: 90,
                    steps: 3000,
                },
                sleep_mask: vec![0],
                any_awake: true,
                list_args: [1, 1, 1, 0],
                planned_by_query: true,
            },
            inputs: vec![vec![Some([1, 2, 3, 4])], vec![None]],
            blobs: Vec::new(),
        };
        let blobs = vec![
            body.clone(),
            body,
            vec![1u8; std::mem::size_of::<engine_schema::Collider>()],
            vec![2u8; 32],
            Vec::new(),
            vec![3u8; 16],
            vec![4u8; work_layout.bytes() as usize],
            explicit.to_bytes(),
        ];
        let w = PortableWorld::new(header, blobs.clone());
        let bytes = w.to_bytes();
        let back = PortableWorld::from_bytes(&bytes).expect("reads back");
        assert_eq!(back.blobs, blobs);
        assert_eq!(back.header.coupling.clock, w.header.coupling.clock);
        assert_eq!(back.header.inputs, w.header.inputs);
        assert_eq!(
            back.explicit().expect("the explicit state").bindings,
            explicit.bindings
        );
        let mut short = w.clone();
        short.blobs[2].clear();
        short.header.blobs[2].1 = 0;
        assert!(PortableWorld::from_bytes(&short.to_bytes())
            .unwrap_err()
            .contains("colliders"));
    }
}
