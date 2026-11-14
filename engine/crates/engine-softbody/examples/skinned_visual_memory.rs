//! Heap high-water of building a skinned JBeam vehicle's visual (the vehicle lab's
//! `jbeamVisualSkinned` path in engine-world-wasm `showcase.rs`), stage by stage, with a
//! counting allocator: the wasm heap only grows, so its size after the load is this
//! peak (plus fragmentation). For the browser's own numbers see
//! engine/tools/vehicle-lab/lab-memory.mjs and engine/docs/measurements/lab-load-memory.md.
//!
//!   cargo run --release -p engine-softbody --example skinned_visual_memory -- <mod folder> <pc>
//!   (e.g. ~/Development/rubble-artifacts/sedan-live/mods/sedan_new sedan/sedan.pc)
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let n = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(n, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // A realloc may move: count the old and new blocks both live at its peak.
        let n = LIVE.fetch_add(new, Relaxed) + new;
        PEAK.fetch_max(n, Relaxed);
        let q = unsafe { System.realloc(p, l, new) };
        LIVE.fetch_sub(l.size(), Relaxed);
        q
    }
}
#[global_allocator]
static A: Counting = Counting;

fn mb(b: usize) -> String {
    format!("{:7.1} MB", b as f64 / (1u64 << 20) as f64)
}

fn main() -> Result<(), String> {
    use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
    use glam::Vec3;
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(args.get(1).ok_or("usage: <mod folder> <pc>")?);
    let pc_name = args.get(2).cloned().unwrap_or_else(|| "sedan/sedan.pc".into());
    let stage = |name: &str| {
        println!("{name:<52} live {}  peak {}", mb(LIVE.load(Relaxed)), mb(PEAK.load(Relaxed)));
    };
    let list: Vec<String> = serde_json::from_slice(&std::fs::read(dir.join("files.json")).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let read = |f: &str| std::fs::read(dir.join(f)).map_err(|e| format!("{f}: {e}"));
    let files: Vec<(String, String)> = list
        .iter()
        .filter(|f| f.ends_with(".jbeam"))
        .map(|f| Ok((f.clone(), String::from_utf8_lossy(&read(f)?).into_owned())))
        .collect::<Result<_, String>>()?;
    let pc = String::from_utf8_lossy(&read(&pc_name)?).into_owned();
    let glb_name = list.iter().find(|f| f.ends_with(".glb") && f.contains('/')).ok_or("no .glb")?;
    let skin_name = list.iter().find(|f| f.ends_with(".rskin")).ok_or("no .rskin")?;
    stage("jbeam text");
    // The bytes as wasm-bindgen copies them into the wasm heap.
    let glb_bytes = read(glb_name)?;
    let skin_bytes = read(skin_name)?;
    stage("glb + rskin bytes in the heap");
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, Some(&pc))?;
    let flex = engine_softbody::jbeam::assembled_rows(&refs, Some(&pc), "flexbodies")?;
    let props = engine_softbody::jbeam::assembled_rows(&refs, Some(&pc), "props")?;
    let sim = ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Raycast, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default())?.0;
    stage("desc, rows and the solver spawn");
    let g = engine_softbody::glb::read(&glb_bytes)?;
    let all: Vec<_> = g.meshes.iter().map(|m| engine_softbody::flexskin::glb_mesh(&g, m)).collect();
    stage("glb parsed (and as flexmesh meshes)");
    let skin = engine_softbody::flexskin::Skin::decode(&skin_bytes)?;
    stage("rskin decoded");
    let (vis, rep) = engine_softbody::flexmesh::jbeam_visual_skinned(&flex, &props, &desc, &sim, &all, &Default::default(), [0.6, 0.04, 0.03], Some(&skin))?;
    stage("visual built");
    // visual_js: each flattened field collected into a Vec, then copied out.
    let flat: Vec<f32> = vis.wt_w.iter().flatten().copied().collect();
    stage("visual_js's largest temporary (wtW flattened)");
    drop(flat);
    println!(
        "{} vertices, {} triangles, {} skin entries ({:.1} a vertex)",
        rep.vertices,
        rep.triangles,
        vis.wt_node.len(),
        vis.wt_node.len() as f64 / vis.positions.len().max(1) as f64
    );
    drop((vis, skin, all, g, glb_bytes, skin_bytes));
    stage("all dropped");
    Ok(())
}
