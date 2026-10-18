//! Cook every hull of a scene pack with the engine's cooker and report failures.
use engine_schema::convex::ConvexHull;
use glam::Vec3;
fn main() {
    let path = std::env::args().nth(1).expect("pack.json");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let (mut ok, mut bad) = (0, 0);
    for (i, c) in json["scenario"]["nodeColliders"].as_array().unwrap().iter().enumerate() {
        let lists: Vec<&serde_json::Value> = match c["kind"].as_str() {
            Some("convex_hull") => vec![c],
            Some("compound") => c["parts"].as_array().unwrap().iter().collect(),
            _ => continue,
        };
        for part in lists {
            let l = &part["points"];
            let pts: Vec<Vec3> = l.as_array().unwrap().chunks_exact(3).map(|t| Vec3::new(t[0].as_f64().unwrap() as f32, t[1].as_f64().unwrap() as f32, t[2].as_f64().unwrap() as f32)).collect();
            let cooked = match part["faces"].as_array() {
                Some(faces) => {
                    let lists: Vec<Vec<u32>> = faces.iter().map(|f| f.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()).collect();
                    let sizes: Vec<u32> = lists.iter().map(|f| f.len() as u32).collect();
                    let idx: Vec<u32> = lists.into_iter().flatten().collect();
                    ConvexHull::from_faces(&pts, &sizes, &idx)
                }
                None => ConvexHull::from_points(&pts),
            };
            match cooked {
                Ok(_) => ok += 1,
                Err(e) => {
                    bad += 1;
                    if bad <= 5 {
                        let mut dmin = f32::INFINITY;
                        for a in 0..pts.len() { for b in a + 1..pts.len() { dmin = dmin.min(pts[a].distance(pts[b])); } }
                        let span = pts.iter().fold(0f32, |m, p| m.max(p.abs().max_element()));
                        println!("node {i}: {e} | verts {} min pair dist {dmin:.3e} m | |x| max {span:.3}", pts.len());
                    }
                }
            }
        }
    }
    println!("ok {ok} failed {bad}");
}
