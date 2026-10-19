//! Lossless cached bond responses, including the elastic/contact-handover state.
//! Retired bonds retain their last response: these are not continuing forces.
use crate::joint_history::{Manifest, Snapshot};
use std::{
    fs,
    path::{Path, PathBuf},
};
use stress_ref::{solver::RtBond, world::World};

pub const FORMAT: &str = "learned-bond-response/1";
pub const FIELDS: [&str; 18] = [
    "force_x",
    "force_y",
    "force_z",
    "moment_x",
    "moment_y",
    "moment_z",
    "elastic_force_x",
    "elastic_force_y",
    "elastic_force_z",
    "elastic_moment_x",
    "elastic_moment_y",
    "elastic_moment_z",
    "stored_energy",
    "tension",
    "shear",
    "normal_compression",
    "compression",
    "compressive_force",
];

pub fn pack(b: &RtBond) -> [f64; 18] {
    let q = b.force;
    let e = b.elastic;
    // Exhaustive destructuring makes a new stress-measure field a compile error.
    let stress_ref::joint::StressMeasures {
        tension,
        shear,
        normal_compression,
        compression,
        compressive_force,
    } = b.measures;
    [
        q.lin.x,
        q.lin.y,
        q.lin.z,
        q.ang.x,
        q.ang.y,
        q.ang.z,
        e.lin.x,
        e.lin.y,
        e.lin.z,
        e.ang.x,
        e.ang.y,
        e.ang.z,
        b.stored,
        tension,
        shear,
        normal_compression,
        compression,
        compressive_force,
    ]
}

pub fn read_snapshot(
    root: &Path,
    index: usize,
    bonds: usize,
) -> Result<(Vec<[f64; 18]>, Vec<[i32; 2]>), String> {
    let raw = fs::read(root.join(format!("{index:06}.f64"))).map_err(|e| e.to_string())?;
    let tags = fs::read(root.join(format!("{index:06}.i32"))).map_err(|e| e.to_string())?;
    if raw.len() != bonds * 18 * 8 || tags.len() != bonds * 2 * 4 {
        return Err("bond-response byte count mismatch".into());
    }
    let values: Vec<[f64; 18]> = raw
        .chunks_exact(18 * 8)
        .map(|r| {
            std::array::from_fn(|i| f64::from_le_bytes(r[i * 8..i * 8 + 8].try_into().unwrap()))
        })
        .collect();
    let flags: Vec<[i32; 2]> = tags
        .chunks_exact(8)
        .map(|r| {
            std::array::from_fn(|i| i32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap()))
        })
        .collect();
    if values.iter().flatten().any(|x| !x.is_finite())
        || flags
            .iter()
            .any(|f| !f.iter().all(|&x| x == 0 || x == 1) || f[1] > f[0])
    {
        return Err("invalid cached bond response".into());
    }
    Ok((values, flags))
}

pub fn validate(
    root: &Path,
    keys: &[(usize, usize)],
    frames: usize,
    dt: f64,
) -> Result<(), String> {
    let dir = root.join("bond_response");
    let m: Manifest =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if m.format != FORMAT
        || m.fields != FIELDS
        || m.flags != ["alive", "connected"]
        || m.bond_keys != keys
        || m.snapshots.len() != frames + 1
        || !dt.is_finite()
        || dt <= 0.
    {
        return Err("bond-response manifest mismatch".into());
    }
    for (i, s) in m.snapshots.iter().enumerate() {
        if !s.time.is_finite()
            || (s.time - m.snapshots[0].time - i as f64 * dt).abs() > 1e-8 + 1e-6 * i as f64 * dt
            || s.values != format!("{i:06}.f64")
            || s.flags != format!("{i:06}.i32")
        {
            return Err("bond-response snapshot alignment mismatch".into());
        }
        let (_, flags) = read_snapshot(&dir, i, keys.len())?;
        let (_, joint) =
            crate::joint_history::read_snapshot(&root.join("joint_history"), i, keys.len())?;
        if flags
            .iter()
            .zip(joint)
            .any(|(f, j)| f[0] != j[2] || f[1] != j[3])
        {
            return Err("response/joint topology mismatch".into());
        }
    }
    Ok(())
}

pub struct Recorder {
    root: PathBuf,
    manifest: Manifest,
}
impl Recorder {
    pub fn new(root: &Path, world: &World, keys: &[(usize, usize)]) -> Result<Self, String> {
        let root = root.join("bond_response");
        fs::create_dir(&root).map_err(|e| e.to_string())?;
        let mut this = Self {
            root,
            manifest: Manifest {
                format: FORMAT.into(),
                fields: FIELDS.iter().map(|s| s.to_string()).collect(),
                flags: vec!["alive".into(), "connected".into()],
                bond_keys: keys.to_vec(),
                snapshots: vec![],
            },
        };
        this.push(world)?;
        Ok(this)
    }
    pub fn push(&mut self, world: &World) -> Result<(), String> {
        let mut values = vec![];
        let mut flags = vec![];
        for &(s, b) in &self.manifest.bond_keys {
            let b = &world.solver.bonds[s][b];
            let row = pack(b);
            if row.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite cached bond response".into());
            }
            for v in row {
                values.extend_from_slice(&v.to_le_bytes());
            }
            for v in [b.alive as i32, b.connected() as i32] {
                flags.extend_from_slice(&v.to_le_bytes());
            }
        }
        let i = self.manifest.snapshots.len();
        let v = format!("{i:06}.f64");
        let f = format!("{i:06}.i32");
        fs::write(self.root.join(&v), values).map_err(|e| e.to_string())?;
        fs::write(self.root.join(&f), flags).map_err(|e| e.to_string())?;
        self.manifest.snapshots.push(Snapshot {
            time: world.solver.time,
            values: v,
            flags: f,
        });
        Ok(())
    }
    pub fn finish(self) -> Result<(), String> {
        fs::write(
            self.root.join("manifest.json"),
            serde_json::to_vec_pretty(&self.manifest).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
}
