//! Every generated design's body paint is skinned only to nodes on the outer collision skin
//! (`gen3::binds_to_skin`, `gen3::outer_skin_nodes`): contact holds those and nothing else, so
//! paint that followed an inner node (a door's inner panel) could stand inside another car
//! whose skin it never crossed (`engine/docs/generated-vehicles-lessons.md`, "Paint binds to
//! the collision skin"). Each design rigged in memory as the site builds it, as
//! `generated_vehicle_lint.rs` does. Also prints per mesh: the weight off the skin, the
//! largest amplification (Σ|A|) and the rest error against the art.
//!
//! Glass is visual only and spans openings with no node in them: it binds to its frame
//! (`GLASS_AMP_MAX`) and is printed, not checked. Rigid pieces (mirrors, lamps), the interior
//! and the running gear are not paint the other car meets: not checked.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test paint_binds_skin -- --nocapture`
//! (`DUMP_DIR=dir`: also write each design's mod files there, for a before/after `diff`.)

use engine_softbody_authoring::authoring::gen2::gen3;
use engine_softbody_authoring::authoring::gen2::vg_input::VgInput;
use engine_softbody_authoring::authoring::rig;
use engine_softbody_authoring::flexskin;
use glam::Vec3;

/// The most weight (Σ|A|) a body-paint vertex may put on nodes off the outer collision skin.
/// 0: the binder never gives it any (the quantised weights are exact zeros).
const OFF_SKIN_MAX: f64 = 0.0;
/// The most a body-paint vertex may sit from the art at rest (m): the rest-fidelity bar.
const REST_MAX: f32 = 1e-3;

#[test]
fn every_body_paint_vertex_binds_to_the_collision_skin() {
    let mut bad = Vec::new();
    for design in engine_vehiclegen::design_titles().into_iter().map(|d| d.0) {
        let built = rig::rig_vehicle(design, None, None).expect("rig");
        let car = &built.rig.car;
        if let Ok(dir) = std::env::var("DUMP_DIR") {
            for (p, b) in rig::mod_files(&built) {
                let f = std::path::Path::new(&dir).join(design).join(&p);
                std::fs::create_dir_all(f.parent().unwrap()).unwrap();
                std::fs::write(f, b).unwrap();
            }
        }
        let on = gen3::outer_skin_nodes(car);
        let g = &built.input.glb;
        let af = built.input.rig.dims.axle_f;
        let q: Vec<Vec3> = car.nodes.iter().map(|n| n.p).collect();
        let (mut tv, mut tk, mut tw) = (0usize, 0usize, 0.0f64);
        for (m, ms) in g.meshes.iter().zip(&built.skin.meshes) {
            assert_eq!(m.name, ms.name);
            let paint = gen3::binds_to_skin(g, m, &VgInput::vg_rigid);
            if !paint && !m.name.starts_with("glass_") {
                continue;
            }
            let pos: Vec<Vec3> =
                m.prims.iter().flat_map(|p| p.pos.iter().map(|x| Vec3::new(x[0], af - x[2], x[1]))).collect();
            let (mut k, mut mx, mut sum, mut amp, mut rest) = (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f32);
            for (v, x) in ms.verts.iter().zip(&pos) {
                let off: f64 = v.w.iter().filter(|w| !on[w.0 as usize]).map(|w| w.1.abs() as f64).sum();
                if off > 0.0 {
                    k += 1;
                    mx = mx.max(off);
                    sum += off;
                }
                amp = amp.max(flexskin::amplification(v));
                rest = rest.max(flexskin::eval(&v.w, v.off, v.nrm, &q).0.distance(*x));
            }
            println!(
                "{design:14} {} {:22} verts {:6} weighted off the skin {k:6} (most {mx:.3}, mean {:.3}) | amplification {amp:.2}, rest {:.3} mm",
                if paint { "paint" } else { "glass" },
                m.name,
                ms.verts.len(),
                if k > 0 { sum / k as f64 } else { 0.0 },
                rest * 1e3
            );
            if paint {
                tv += ms.verts.len();
                tk += k;
                tw = tw.max(mx);
                if mx > OFF_SKIN_MAX {
                    bad.push(format!("{design} {}: {k} vertices weighted off the skin, most {mx:.3}", m.name));
                }
                if rest > REST_MAX {
                    bad.push(format!("{design} {}: {:.3} mm from the art at rest", m.name, rest * 1e3));
                }
            }
        }
        println!("{design}: body paint {tv} vertices, {tk} weighted off the skin (most {tw:.3})");
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
