//! Lossless joint-history sidecar. Separate from the legacy clamped f32 model ABI.
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};
use stress_ref::{
    joint::{FailureMode, JointState, Local6Ser},
    world::World,
};

pub const FORMAT: &str = "learned-joint-history/1";
pub const FIELDS: [&str; 21] = [
    "damage",
    "crush",
    "kappa",
    "kappa_c",
    "ductility",
    "ductility_c",
    "fatigue",
    "plastic_x",
    "plastic_y",
    "plastic_z",
    "plastic_rx",
    "plastic_ry",
    "plastic_rz",
    "rebar_plastic",
    "rebar_slip_x",
    "rebar_slip_y",
    "rebar_work",
    "strain_rate",
    "governing_stress",
    "dissipated",
    "utilization",
];

pub fn pack(st: &JointState) -> [f64; 21] {
    [
        st.damage,
        st.crush,
        st.kappa,
        st.kappa_c,
        st.ductility,
        st.ductility_c,
        st.fatigue,
        st.plastic.lin[0],
        st.plastic.lin[1],
        st.plastic.lin[2],
        st.plastic.ang[0],
        st.plastic.ang[1],
        st.plastic.ang[2],
        st.rebar_plastic,
        st.rebar_slip[0],
        st.rebar_slip[1],
        st.rebar_work,
        st.strain_rate,
        st.governing_stress,
        st.dissipated,
        st.utilization,
    ]
}
pub fn mode_code(mode: Option<FailureMode>) -> i32 {
    match mode {
        None => 0,
        Some(FailureMode::Tension) => 1,
        Some(FailureMode::Shear) => 2,
        Some(FailureMode::Compression) => 3,
        Some(FailureMode::Buckling) => 4,
    }
}
pub fn unpack(v: [f64; 21], rebar_broken: bool, mode: i32) -> Result<JointState, String> {
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() && !((i == 4 || i == 5) && x == f64::INFINITY) {
            return Err(format!("invalid joint-history {}", FIELDS[i]));
        }
    }
    let mode = match mode {
        0 => None,
        1 => Some(FailureMode::Tension),
        2 => Some(FailureMode::Shear),
        3 => Some(FailureMode::Compression),
        4 => Some(FailureMode::Buckling),
        _ => return Err("invalid joint-history mode".into()),
    };
    // Exhaustive initialization makes additions to JointState a compile-time contract change.
    Ok(JointState {
        damage: v[0],
        crush: v[1],
        kappa: v[2],
        kappa_c: v[3],
        ductility: v[4],
        ductility_c: v[5],
        fatigue: v[6],
        plastic: Local6Ser {
            lin: [v[7], v[8], v[9]],
            ang: [v[10], v[11], v[12]],
        },
        rebar_plastic: v[13],
        rebar_slip: [v[14], v[15]],
        rebar_work: v[16],
        rebar_broken,
        strain_rate: v[17],
        governing_stress: v[18],
        mode,
        dissipated: v[19],
        utilization: v[20],
    })
}

#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub time: f64,
    pub values: String,
    pub flags: String,
}
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub fields: Vec<String>,
    pub flags: Vec<String>,
    pub bond_keys: Vec<(usize, usize)>,
    pub snapshots: Vec<Snapshot>,
}
pub struct Recorder {
    root: PathBuf,
    manifest: Manifest,
}

pub fn read_snapshot(
    root: &Path,
    index: usize,
    bonds: usize,
) -> Result<(Vec<[f64; 21]>, Vec<[i32; 4]>), String> {
    let raw = fs::read(root.join(format!("{index:06}.f64"))).map_err(|e| e.to_string())?;
    let tags = fs::read(root.join(format!("{index:06}.i32"))).map_err(|e| e.to_string())?;
    if raw.len() != bonds * 21 * 8 || tags.len() != bonds * 4 * 4 {
        return Err("joint-history byte count mismatch".into());
    }
    let values: Vec<[f64; 21]> = raw
        .chunks_exact(21 * 8)
        .map(|row| {
            std::array::from_fn(|i| f64::from_le_bytes(row[i * 8..i * 8 + 8].try_into().unwrap()))
        })
        .collect();
    let flags: Vec<[i32; 4]> = tags
        .chunks_exact(16)
        .map(|row| {
            std::array::from_fn(|i| i32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap()))
        })
        .collect();
    for (&v, f) in values.iter().zip(&flags) {
        if ![f[0], f[2], f[3]].iter().all(|&x| x == 0 || x == 1) || f[3] > f[2] {
            return Err("invalid joint-history flags".into());
        }
        unpack(v, f[0] == 1, f[1])?;
    }
    Ok((values, flags))
}

pub fn validate(
    root: &Path,
    keys: &[(usize, usize)],
    frames: usize,
    dt: f64,
) -> Result<(), String> {
    let root = root.join("joint_history");
    let m: Manifest =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if m.format != FORMAT
        || m.fields != FIELDS
        || m.flags != ["rebar_broken", "mode", "alive", "connected"]
        || m.bond_keys != keys
        || m.snapshots.len() != frames + 1
    {
        return Err("joint-history manifest mismatch".into());
    }
    if !dt.is_finite() || dt <= 0. {
        return Err("invalid joint-history dt".into());
    }
    for (i, frame) in m.snapshots.iter().enumerate() {
        if !frame.time.is_finite()
            || (frame.time - m.snapshots[0].time - i as f64 * dt).abs()
                > 1e-8 + 1e-6 * i as f64 * dt
            || frame.values != format!("{i:06}.f64")
            || frame.flags != format!("{i:06}.i32")
        {
            return Err("joint-history snapshot alignment mismatch".into());
        }
        read_snapshot(&root, i, keys.len())?;
    }
    Ok(())
}
impl Recorder {
    pub fn new(root: &Path, world: &World, keys: &[(usize, usize)]) -> Result<Self, String> {
        let root = root.join("joint_history");
        fs::create_dir(&root).map_err(|e| e.to_string())?;
        let mut writer = Self {
            root,
            manifest: Manifest {
                format: FORMAT.into(),
                fields: FIELDS.iter().map(|s| s.to_string()).collect(),
                flags: ["rebar_broken", "mode", "alive", "connected"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                bond_keys: keys.to_vec(),
                snapshots: vec![],
            },
        };
        writer.push(world)?;
        Ok(writer)
    }
    pub fn push(&mut self, world: &World) -> Result<(), String> {
        let mut values = Vec::new();
        let mut flags = Vec::new();
        for &(s, b) in &self.manifest.bond_keys {
            let bond = &world.solver.bonds[s][b];
            let row = pack(&bond.joint);
            let mode = mode_code(bond.joint.mode);
            unpack(row, bond.joint.rebar_broken, mode)?;
            for value in row {
                values.extend_from_slice(&value.to_le_bytes());
            }
            for flag in [
                bond.joint.rebar_broken as i32,
                mode,
                bond.alive as i32,
                bond.connected() as i32,
            ] {
                flags.extend_from_slice(&flag.to_le_bytes());
            }
        }
        let index = self.manifest.snapshots.len();
        let value_path = format!("{index:06}.f64");
        let flags_path = format!("{index:06}.i32");
        fs::write(self.root.join(&value_path), values).map_err(|e| e.to_string())?;
        fs::write(self.root.join(&flags_path), flags).map_err(|e| e.to_string())?;
        self.manifest.snapshots.push(Snapshot {
            time: world.solver.time,
            values: value_path,
            flags: flags_path,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_round_trip_retains_infinity_large_ductility_signed_zero_and_mode() {
        for mode in 0..=4 {
            let mut fields = std::array::from_fn(|i| i as f64 + 0.123456789012345);
            fields[4] = f64::INFINITY;
            fields[5] = 1234567.891234567;
            fields[7] = -0.;
            fields[8] = -1.234567891234567;
            let state = unpack(fields, true, mode).unwrap();
            assert_eq!(pack(&state).map(f64::to_bits), fields.map(f64::to_bits));
            assert!(state.rebar_broken);
            assert_eq!(mode_code(state.mode), mode);
        }
        let mut fields = pack(&JointState::new());
        fields[4] = f64::NEG_INFINITY;
        assert!(unpack(fields, false, 0).is_err());
        fields[4] = 0.;
        fields[9] = f64::NAN;
        assert!(unpack(fields, false, 0).is_err());
        fields[9] = 0.;
        assert!(unpack(fields, false, 7).is_err());
    }
}
