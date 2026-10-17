//! Cooks a fracture GLB's level for the GPU stress solver (`.stress.bin`).
//!
//!   cargo run --release -p engine-fracture --bin fracture_cook -- <asset.glb> <out.stress.bin> [level=0]

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (glb, out) = match (args.first(), args.get(1)) {
        (Some(g), Some(o)) => (g, o),
        _ => anyhow::bail!("usage: fracture_cook <asset.glb> <out.stress.bin> [level=0]"),
    };
    let level: u8 = args.get(2).map(|l| l.parse()).transpose()?.unwrap_or(0);
    let start = std::time::Instant::now();
    let asset = engine_fracture::parse_fracture_glb(&std::fs::read(glb)?)?;
    let cooked = engine_fracture::cook::cook(&asset, level, stress_ref::scene::Methods::established())?;
    let bytes = engine_fracture::cook::write_stress_bin(&cooked);
    std::fs::write(out, &bytes)?;
    let t = &cooked.tables;
    eprintln!(
        "{out}: level {level}, {} chunks, {} bonds, {} islands, {} joint materials, {} bytes in {:.2} s",
        t.chunks.len(),
        t.bonds.len(),
        t.islands.len(),
        t.table.materials.len(),
        bytes.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
