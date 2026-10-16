//! Print a digest of a built scene for parity checks against `fracture/geometry.mjs`.
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1])?;
    let asset = engine_fracture::parse_fracture_glb(&bytes)?;
    let level: u8 = args.get(2).map_or(Ok(0), |s| s.parse())?;
    let s = engine_fracture::build_scene(&asset, level)?;
    let sum = |v: &mut dyn Iterator<Item = f32>| v.map(|x| x as f64).sum::<f64>();
    println!(
        "bodies {} colliders {} hull {} verts {} lines {}",
        s.bodies.len(),
        s.colliders.len(),
        s.hull_data.len(),
        s.vertices.len(),
        s.lines.len()
    );
    println!(
        "hullsum {:.6}",
        sum(&mut s.hull_data.iter().flat_map(|v| v[..3].to_vec()))
    );
    println!(
        "bodysum {:.6}",
        sum(&mut s.bodies.iter().flat_map(|b| b
            .p
            .into_iter()
            .chain(b.q)
            .chain(b.inv_i)
            .chain([b.inv_mass])))
    );
    println!(
        "colsum {:.6}",
        sum(&mut s
            .colliders
            .iter()
            .flat_map(|c| c.params.into_iter().chain([c.density])))
    );
    println!("vertsum {:.4}", sum(&mut s.vertices.iter().copied()));
    if let Some(out) = args.get(3) {
        let bytes: Vec<u8> = s
            .hull_data
            .iter()
            .flat_map(|v| v.iter().flat_map(|x| x.to_le_bytes()))
            .collect();
        std::fs::write(out, bytes)?;
    }
    println!(
        "grid {} {} extent {} center {:?}",
        s.grid_cell_size, s.grid_levels, s.extent, s.center
    );
    Ok(())
}
