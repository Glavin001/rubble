//! Debug view of a `.glb` part's triangles near a point, as an SVG wireframe
//! projected on a plane (`xz` top, `zy` side, `xy` front).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example glb_wire -- PART x,y,z RADIUS xz|zy|xy out.svg [path.glb]
//! ```
use engine_softbody::glb;
use glam::Vec3;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let part = &a[0];
    let c: Vec<f32> = a[1].split(',').map(|s| s.parse().unwrap()).collect();
    let c = Vec3::new(c[0], c[1], c[2]);
    let r: f32 = a[2].parse().unwrap();
    let plane = a[3].as_str();
    let out = &a[4];
    let path = a.get(5).cloned().unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../engine-webgpu/__mods/gen2/sedan.glb"
        )
        .to_string()
    });
    let g = glb::read(&std::fs::read(path).unwrap()).unwrap();
    let pr = |p: Vec3| -> (f32, f32) {
        let d = p - c;
        match plane {
            "xz" => (d.z, d.x),
            "zy" => (d.z, -d.y),
            _ => (d.x, -d.y),
        }
    };
    let s = 400.0 / r;
    let mut svg = format!("<svg xmlns='http://www.w3.org/2000/svg' viewBox='-420 -420 840 840' width='840' height='840'><rect x='-420' y='-420' width='840' height='840' fill='white'/>");
    for m in g.meshes.iter().filter(|m| &m.name == part) {
        for p in &m.prims {
            for t in p.idx.chunks(3) {
                let v: Vec<Vec3> = t
                    .iter()
                    .map(|&i| Vec3::from_array(p.pos[i as usize]))
                    .collect();
                if v.iter().all(|q| q.distance(c) > r) {
                    continue;
                }
                let pts: Vec<String> = v
                    .iter()
                    .map(|&q| {
                        let (x, y) = pr(q);
                        format!("{:.1},{:.1}", x * s, y * s)
                    })
                    .collect();
                svg += &format!("<polygon points='{}' fill='rgba(200,60,40,0.08)' stroke='black' stroke-width='0.6'/>", pts.join(" "));
            }
        }
    }
    svg += "<circle cx='0' cy='0' r='3' fill='blue'/></svg>";
    std::fs::write(out, svg).unwrap();
}
