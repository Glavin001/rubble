//! A portable explicit-beams state: one moment of a fleet that any host replays with
//! no vehicles, no World and no GPU API of the host that saved it.
//!
//! A `.fx` fixture (vehicle_lab `--save-fixtures`) holds the state bindings and is
//! restored into a World built the same way (the same scene, cars and options): only the
//! wgpu host can read it. A portable fixture holds instead every binding's bytes (the
//! uploads and the host's writes into them as well: piece labels, retired nodes, sleep
//! thresholds, the coupling's header and frozen proxies, drive inputs), the layout, the
//! planner's settings and the host's frame paths ([`HostSwitches`]): a CUDA host replays
//! the scenario matrix's rows from it (`engine-cuda`'s `explicit_replay`).
//!
//! File: [`MAGIC`], the header's length (u32 LE), the header as JSON, then each binding's
//! bytes in order (their lengths in the header).

use super::{ExplicitLayout, PlannerSettings, BIND_COUNT};

pub const MAGIC: &[u8; 8] = b"RUBBLEPF";
pub const VERSION: u32 = 1;

/// The frame paths a host had on (engine_exec's frame schedule, `schedule::Switches`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HostSwitches {
    pub group_split: bool,
    pub group_split_tests: bool,
    pub self_split: bool,
    pub wide_list: bool,
    pub wide_fused: bool,
    pub vv_grid: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Header {
    pub version: u32,
    /// Simulated time (s) of the moment.
    pub t: f64,
    /// What it is (e.g. the matrix row and the host that saved it).
    pub note: String,
    pub layout: ExplicitLayout,
    pub planner: PlannerSettings,
    pub switches: HostSwitches,
    /// Bytes of each binding ([`BIND_COUNT`]).
    pub binding_bytes: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Portable {
    pub header: Header,
    /// Every binding's bytes, [`BIND_COUNT`] of them.
    pub bindings: Vec<Vec<u8>>,
}

impl Portable {
    pub fn new(t: f64, note: &str, layout: ExplicitLayout, planner: PlannerSettings, switches: HostSwitches, bindings: Vec<Vec<u8>>) -> Self {
        assert_eq!(bindings.len(), BIND_COUNT, "a portable state holds every binding");
        let header = Header {
            version: VERSION,
            t,
            note: note.to_string(),
            layout,
            planner,
            switches,
            binding_bytes: bindings.iter().map(|b| b.len() as u64).collect(),
        };
        Self { header, bindings }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let head = serde_json::to_vec(&self.header).expect("the header serialises");
        let mut out = Vec::with_capacity(12 + head.len() + self.bindings.iter().map(Vec::len).sum::<usize>());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(head.len() as u32).to_le_bytes());
        out.extend_from_slice(&head);
        for b in &self.bindings {
            out.extend_from_slice(b);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err("not a portable explicit-beams state (no RUBBLEPF magic)".into());
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let head = bytes.get(12..12 + n).ok_or("the header runs past the file")?;
        let header: Header = serde_json::from_slice(head).map_err(|e| format!("header: {e}"))?;
        if header.version != VERSION {
            return Err(format!("version {} (this build reads {VERSION})", header.version));
        }
        if header.binding_bytes.len() != BIND_COUNT {
            return Err(format!("{} bindings, not {BIND_COUNT}", header.binding_bytes.len()));
        }
        let mut at = 12 + n;
        let mut bindings = Vec::with_capacity(BIND_COUNT);
        for &len in &header.binding_bytes {
            let b = bytes.get(at..at + len as usize).ok_or("a binding runs past the file")?;
            bindings.push(b.to_vec());
            at += len as usize;
        }
        if at != bytes.len() {
            return Err(format!("{} bytes past the last binding", bytes.len() - at));
        }
        Ok(Self { header, bindings })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fleet's layout and planner through the file and back: the planner rebuilt from
    /// the layout and the settings writes the same parameters and commands.
    #[test]
    fn portable_state_round_trips() {
        use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
        let text = "nodes\n0, 0.0, 0.0, 0.0\n1, 0.3, 0.0, 0.0\n2, 0.0, 0.3, 0.0\nbeams\n0, 1\n1, 2\n2, 0\nend\n";
        let car = |dx: f32| {
            ExplicitBeams::from_truck(text, glam::Vec3::new(dx, 1.0, 0.0), glam::Vec3::ZERO, ExplicitBeamsSettings::default())
                .expect("the truck parses")
        };
        let up = super::super::ExplicitUpload::new(&[car(0.0), car(5.0)], None);
        let mut planner = super::super::FramePlanner::new(&up.layout);
        planner.set_contact_margin(0.05);
        let _ = planner.write_params(17);
        let bindings: Vec<Vec<u8>> = (0..BIND_COUNT as u32).map(|b| up.bytes(b)).collect();
        let p = Portable::new(1.5, "test", up.layout.clone(), planner.settings(), HostSwitches { group_split: true, ..Default::default() }, bindings.clone());
        let q = Portable::from_bytes(&p.to_bytes()).expect("reads back");
        assert_eq!(q.bindings, bindings);
        assert_eq!(q.header.switches, p.header.switches);
        assert_eq!(q.header.planner, planner.settings());
        let mut again = super::super::FramePlanner::new(&q.header.layout);
        again.apply(&q.header.planner);
        assert_eq!(again.write_params(17), planner.write_params(17));
        assert_eq!(again.commands(17, true), planner.commands(17, true));
    }
}
