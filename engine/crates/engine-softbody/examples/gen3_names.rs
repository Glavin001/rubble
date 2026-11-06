//! Diagnostic: a manifest car's node names by template index, with mass and
//! position (`gen3_names <car> <index ...>`).
use engine_softbody::vehicle_compare::{default_manifest, Car, Manifest};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let name = a.first().cloned().unwrap_or_else(|| "gen2".into());
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(default_manifest()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let spec = m
        .cars
        .iter()
        .find(|c| c.name == name)
        .ok_or("no such car")?;
    let car = Car::load(spec)?;
    let s = &car.template;
    let ids: Vec<usize> = a
        .iter()
        .skip(1)
        .filter_map(|x| {
            x.parse::<usize>()
                .ok()
                .or_else(|| (0..s.nodes.len()).find(|&i| car.node_name(i) == *x))
        })
        .collect();
    for i in ids {
        let n = &s.nodes[i];
        println!("{i}: {} {:.3} kg at {:?}", car.node_name(i), n.mass, n.pos);
        for b in s
            .beams
            .iter()
            .filter(|b| (b.a as usize == i || b.b as usize == i) && b.k > 0.0)
        {
            let o = if b.a as usize == i { b.b } else { b.a } as usize;
            println!(
                "    {} k {:.3e} c {:.1} rest {:.3} ({:.3} kg)",
                car.node_name(o),
                b.k,
                b.c,
                b.rest,
                s.nodes[o].mass
            );
        }
    }
    Ok(())
}
