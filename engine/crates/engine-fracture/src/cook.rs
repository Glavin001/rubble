//! The stress cooker: a fracture GLB as the GPU stress solver's records.
//!
//! The asset goes through the stress oracle itself: its Shardwright input
//! (`stress_export`), stress-ref's import (calibrated materials, compound hull chunks,
//! supports from world bonds), the oracle's world in its gravity-prestressed equilibrium,
//! and `stress_layout::pack` (the records `stress-gpu`'s kernels read). Nothing about the
//! bond law or the layout is restated here.
//!
//! The records are in the oracle's frame: Z up, SI, the asset's origin. Chunk `k` of the
//! output is the `k`-th fragment of `level` in id order.

use anyhow::{anyhow, Result};
use stress_layout::joint::GpuJointMaterial;
use stress_layout::{StructureTables, MAX_MATERIALS};
use stress_ref::scene::{Methods, Scene};
use stress_ref::world::World;

use crate::asset::FractureAsset;
use crate::stress_export::shardwright_stress_json;

pub struct Cooked {
    /// The oracle scene the records come from (one structure, one body).
    pub scene: Scene,
    pub tables: StructureTables,
    /// The joint material table, padded to the GPU's capacity.
    pub materials: Vec<GpuJointMaterial>,
    /// Asset fragment id of each output chunk.
    pub chunk_fragment: Vec<u32>,
    pub level: u8,
}

/// Cook `level` of the asset with the given method set (`Methods::established()` is
/// what the GPU kernels implement).
pub fn cook(asset: &FractureAsset, level: u8, methods: Methods) -> Result<Cooked> {
    let json = shardwright_stress_json(asset, level);
    let mut scene = stress_ref::shardwright::import_value(&json, &asset.physics.name).map_err(|e| anyhow!(e))?;
    scene.sim.methods = methods;
    scene.validate().map_err(|e| anyhow!(e))?;
    let scene = scene.with_derived();
    let world = World::new(&scene);
    let tables = stress_layout::pack(&world.solver).map_err(|e| anyhow!(e))?;
    let mut materials = vec![GpuJointMaterial::default(); MAX_MATERIALS];
    materials[..tables.table.materials.len()].copy_from_slice(&tables.table.materials);
    let fragments: Vec<u32> = asset.physics.fragments.iter().filter(|f| f.level == level).map(|f| f.id).collect();
    let chunk_fragment = tables.chunks.iter().map(|&(_, c)| fragments[c]).collect();
    Ok(Cooked { scene, tables, materials, chunk_fragment, level })
}

const MAGIC: &[u8; 8] = b"RSTRESS1";

/// `.stress.bin`: `RSTRESS1`, u32 section count, then per section a 16-byte name and u32
/// offset / count / stride (bytes), then the 16-byte-aligned section data.
pub fn write_stress_bin(c: &Cooked) -> Vec<u8> {
    let t = &c.tables;
    let meta = serde_json::json!({
        "format": "rubble-stress/1", "level": c.level, "frame": "oracle (Z up, SI, asset origin)",
        "chunks": t.chunks.len(), "bonds": t.bonds.len(), "islands": t.islands.len(),
        "materials": t.table.materials.len(), "methods_established": c.scene.sim.methods.is_established(),
    })
    .to_string();
    let sections: Vec<(&str, &[u8], usize)> = vec![
        ("meta", meta.as_bytes(), 1),
        ("chunk_static", bytemuck::cast_slice(&t.chunk_static), std::mem::size_of::<stress_layout::ChunkStatic>()),
        ("chunk_state", bytemuck::cast_slice(&t.state), 16),
        ("islands", bytemuck::cast_slice(&t.islands), std::mem::size_of::<stress_layout::Island>()),
        ("materials", bytemuck::cast_slice(&c.materials), std::mem::size_of::<GpuJointMaterial>()),
        ("bond_static", bytemuck::cast_slice(&t.bond_static), std::mem::size_of::<stress_layout::BondStatic>()),
        ("bond_dyn", bytemuck::cast_slice(&t.bond_dyn), std::mem::size_of::<stress_layout::BondDyn>()),
        ("csr", bytemuck::cast_slice(&t.csr), 4),
        ("chunk_fragment", bytemuck::cast_slice(&c.chunk_fragment), 4),
    ];
    let header = 12 + sections.len() * 28;
    let mut offset = header.next_multiple_of(16);
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    let mut layout = Vec::new();
    for (name, bytes, stride) in &sections {
        let mut n = [0u8; 16];
        n[..name.len()].copy_from_slice(name.as_bytes());
        out.extend_from_slice(&n);
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        out.extend_from_slice(&((bytes.len() / stride) as u32).to_le_bytes());
        out.extend_from_slice(&(*stride as u32).to_le_bytes());
        layout.push(offset);
        offset = (offset + bytes.len()).next_multiple_of(16);
    }
    for ((_, bytes, _), at) in sections.iter().zip(layout) {
        out.resize(at, 0);
        out.extend_from_slice(bytes);
    }
    out
}
