//! Recover verified undeformed display geometry for existing recorded events.
//! The old schema has no world poses; do not invent a free-running trajectory.
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
use stress_learned::generate::{
    diversity_jobs, diversity_v2_jobs, mini_jobs, prove_jobs, smoke_jobs, wall_jobs,
};
use stress_ref::solver::ReferenceSolver;

fn floats(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() % 4 != 0 {
        return Err(format!("invalid f32 file {}", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-5 * x.abs().max(y.abs()).max(1e-6))
}
fn run(root: &Path, out: &Path) -> Result<(), String> {
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let jobs = [
        smoke_jobs(),
        mini_jobs(),
        wall_jobs(),
        prove_jobs(),
        diversity_jobs(),
        diversity_v2_jobs(),
    ]
    .concat();
    let mut candidates = Vec::new();
    for job in jobs {
        let sv = ReferenceSolver::new(&job.scene);
        let mut mapping: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut centers = Vec::new();
        let mut rotations = Vec::new();
        let mut mass = Vec::new();
        let mut half = Vec::new();
        for (s, st) in sv.structures.iter().enumerate() {
            for (c, ch) in st.chunks.iter().enumerate() {
                if !sv.chunks[s][c].active {
                    continue;
                }
                mapping.insert((s, c), centers.len());
                centers.push(sv.chunk_position(s, c).to_array());
                let cl = &sv.clusters[sv.chunks[s][c].cluster];
                rotations.push((cl.pose.rotation.to_mat3() * ch.rotation).m);
                mass.push(ch.mass as f32);
                half.extend(ch.half_extents.to_array().map(|v| v as f32));
            }
        }
        let mut edges: Vec<i32> = Vec::new();
        for (s, bonds) in sv.bonds.iter().enumerate() {
            for b in bonds {
                if b.alive {
                    if let (Some(a), Some(c)) = (
                        mapping.get(&(s, b.geometry.a)),
                        mapping.get(&(s, b.geometry.b)),
                    ) {
                        edges.extend([*a as i32, *c as i32]);
                    }
                }
            }
        }
        candidates.push((job, centers, rotations, mass, half, edges));
    }
    let mut records = serde_json::Map::new();
    for entry in manifest["events"]
        .as_array()
        .ok_or("dataset manifest needs events")?
    {
        let name = entry["path"].as_str().ok_or("event path")?;
        let event = root.join(name);
        let mass = floats(&event.join("static/chunk_mass.bin"))?;
        let half = floats(&event.join("static/chunk_half_extents.bin"))?;
        let bytes = fs::read(event.join("static/bond_endpoints.bin")).map_err(|e| e.to_string())?;
        let edges: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let matching: Vec<_> = candidates
            .iter()
            .filter(|c| {
                c.0.structure_type == entry["structure_type"].as_str().unwrap_or("")
                    && close(&mass, &c.3)
                    && close(&half, &c.4)
                    && edges == c.5
            })
            .collect();
        let chosen = matching
            .iter()
            .find(|c| c.0.event_id == entry["event_id"].as_str().unwrap_or(""))
            .copied()
            .or_else(|| matching.first().copied());
        if let Some(c) = chosen {
            // If metadata cannot identify a catalog event, require all matching
            // candidates to agree on geometry. Never guess an ambiguous arrangement.
            if c.0.event_id != entry["event_id"].as_str().unwrap_or("")
                && matching.iter().any(|v| v.1 != c.1 || v.2 != c.2)
            {
                return Err(format!("ambiguous geometry for {name}"));
            }
            records.insert(name.to_string(), json!({"centers":c.1,"rotations":c.2,"source_scene":c.0.scene.name,
                "display":"undeformed catalog geometry; mass, extents and ordered endpoints verified"}));
        } else {
            return Err(format!("no verified catalog geometry for {name}"));
        }
    }
    fs::write(
        out,
        serde_json::to_vec(&json!({"format":"learned-impact-geometry/1","events":records}))
            .unwrap(),
    )
    .map_err(|e| e.to_string())?;
    eprintln!("wrote {} geometries to {}", records.len(), out.display());
    Ok(())
}
fn main() {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 2 {
        eprintln!("usage: learned_impact_geometry DATASET OUTPUT.json");
        std::process::exit(2);
    }
    if let Err(e) = run(Path::new(&a[0]), Path::new(&a[1])) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
