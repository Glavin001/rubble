//! Read-only f64 export of native six-DOF linear elastic mechanics.
//! This freezes the initial bond geometry. It is NOT a nonlinear World restart.
use std::{fs, path::PathBuf};
use stress_ref::{builders::*, math::Vec3, scene::*, world::World};

fn fixture(kind: &str) -> Result<Scene, String> {
    if !["free", "fixed", "pinned", "rotated"].contains(&kind) {
        return Err("fixture must be free, fixed, pinned or rotated".into());
    }
    let mut s = new_scene("elastic_graph_probe", "Frozen six-DOF linear mechanics probe");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let counts = [3, 2, 4];
    let mut chunks = grid(Vec3::ZERO, counts, Vec3::splat(0.1), "concrete");
    for i in 0..3 {
        for j in 0..2 {
            chunks[grid_index(counts, i, j, 0)].support = match kind {
                "fixed" | "rotated" => Support::Fixed,
                "pinned" => Support::Pinned,
                _ => Support::None,
            };
        }
    }
    let mut bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    if kind == "rotated" {
        // Rotate the complete asset, including its bond frames and box inertias.
        let angle: f64 = 0.53;
        let rotate = |p: [f64; 3]| [angle.cos()*p[0]-angle.sin()*p[1],
                                     angle.sin()*p[0]+angle.cos()*p[1], p[2]];
        for ch in &mut chunks {
            ch.center = rotate(ch.center);
            ch.orientation = [(angle/2.).cos(), 0., 0., (angle/2.).sin()];
            // Non-cubic inertia exercises all entries of the rotated tensor.
            ch.half_extents[0] *= 0.7;
        }
        for bond in &mut bonds {
            bond.centroid = rotate(bond.centroid);
            bond.normal = rotate(bond.normal);
            bond.tangent = rotate(bond.tangent);
        }
    }
    s.bodies.push(body("probe", chunks, bonds));
    s.gravity = [0.; 3];
    s.sim.methods = Methods::established();
    Ok(s)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut source, mut kind, mut out) = (None, None, None);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("expected --scene PATH or --fixture NAME, and --out FILE")?;
        match arg.as_str() {
            "--scene" => source = Some(PathBuf::from(value)),
            "--fixture" => kind = Some(value),
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    let scene = match (source, kind) {
        (Some(path), None) => Scene::load(&path)?,
        (None, Some(kind)) => fixture(&kind)?,
        _ => return Err("specify exactly one of --scene or --fixture".into()),
    };
    let world = World::new(&scene);
    let sv = &world.solver;
    let mut clusters = Vec::new();
    for cl in &sv.clusters {
        let st = &sv.structures[cl.structure];
        let mut mapping = vec![usize::MAX; st.chunks.len()];
        let mut chunks = Vec::new();
        for (slot, &c) in cl.chunks.iter().enumerate() {
            mapping[c] = slot;
            let ch = &st.chunks[c];
            let scale = sv.chunks[cl.structure][c].inertia_scale;
            let mut mass = [[0.; 6]; 6];
            for d in 0..3 { mass[d][d] = ch.mass*scale; }
            for i in 0..3 { for j in 0..3 { mass[i+3][j+3] = ch.inertia.m[i][j]*scale; } }
            let fixed: [bool; 6] = std::array::from_fn(|d| match ch.support {
                Support::Fixed => true, Support::Pinned => d < 3, Support::None => false,
            });
            chunks.push(serde_json::json!({"source": c, "center": ch.center.to_array(),
                "mass_matrix": mass, "fixed": fixed}));
        }
        let state: Vec<[f64; 6]> = (0..chunks.len()).map(|i| std::array::from_fn(|d| {
            let fixed = match st.chunks[cl.chunks[i]].support {
                Support::Fixed => true, Support::Pinned => d < 3, Support::None => false,
            };
            if fixed { 0. } else { (((i*37+d*19+7) as f64).sin())*1e-6 }
        })).collect();
        let mut loads = vec![[0.; 6]; chunks.len()];
        let mut bonds = Vec::new();
        for &bi in &cl.bonds {
            let b = &sv.bonds[cl.structure][bi];
            if !b.alive { continue; }
            if b.joint.damage != 0. || b.joint.crush != 0. || b.rebar.is_some() {
                return Err("export requires initial intact bonds without rebar".into());
            }
            let g = &b.geometry;
            let (a, c) = (mapping[g.a], mapping[g.b]);
            if a == usize::MAX || c == usize::MAX { return Err("cross-cluster bond".into()); }
            let at = |i: usize| (Vec3::new(state[i][0], state[i][1], state[i][2]),
                                   Vec3::new(state[i][3], state[i][4], state[i][5]));
            let ((ua, ta), (ub, tb)) = (at(a), at(c));
            let deformation = g.kinematics(ua, ta, ub, tb);
            let force = deformation.mul_elem(&b.stiffness.as_local());
            let (fa, ma, fb, mb) = g.chunk_loads(&force);
            for (i, f, m) in [(a, fa, ma), (c, fb, mb)] {
                for d in 0..3 { loads[i][d] += f[d]; loads[i][d+3] += m[d]; }
            }
            bonds.push(serde_json::json!({"source": bi, "endpoints": [a,c],
                "axes": [g.t1.to_array(),g.t2.to_array(),g.normal.to_array()],
                "ra": g.ra.to_array(), "rb": g.rb.to_array(),
                "stiffness": b.stiffness.as_local().to_array(), "damping": b.damping.to_array(),
                "probe_deformation": deformation.to_array(), "probe_force": force.to_array()}));
        }
        clusters.push(serde_json::json!({"structure": cl.structure, "chunks": chunks,
            "bonds": bonds, "probe_displacement": state, "probe_nodal_force": loads}));
    }
    let output = serde_json::json!({"schema": "native-linear-elastic-graph/1", "scene": scene,
        "scope": "Initial frozen linear elastic kinematics, mass/inertia and bond mechanics. Damping is recorded but not applied in this probe. Not a nonlinear World trajectory or restart.",
        "clusters": clusters});
    let path = out.ok_or("missing --out")?;
    let file = fs::OpenOptions::new().write(true).create_new(true).open(path).map_err(|e| e.to_string())?;
    serde_json::to_writer(file, &output).map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(1); }
}
