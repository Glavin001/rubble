//! The GPU joint law (joint.slang) against stress-ref's `JointModel::evaluate` and
//! `secant_factors` in f64, on random states and deformations of real catalogue bonds.
//! The reference evaluates the f32-rounded inputs, so differences come only from the
//! arithmetic.

use stress_gpu::gpu::{Binding, Gpu};
use stress_gpu::joint::{bond_law, from_gpu_state, mode_of, to_gpu_state, GpuJointBond, GpuJointState, MaterialTable};
use stress_ref::bond::{BondGeometry, BondStiffness, Local6};
use stress_ref::joint::{JointModel, JointState, JointStrength, RebarParams};
use stress_ref::math::Vec3;
use stress_ref::solver::ReferenceSolver;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sym(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
    fn pick(&mut self, p: f64) -> bool {
        self.next() < p
    }
}

fn r32(x: f64) -> f64 {
    x as f32 as f64
}

/// A bond's law data with every value rounded to f32.
#[derive(Clone)]
struct Case {
    geometry: BondGeometry,
    stiffness: BondStiffness,
    strength: JointStrength,
    rebar: Option<RebarParams>,
    weibull: f64,
    state: JointState,
    d: Local6,
    dt: f64,
    fracture: bool,
}

fn round_case(mut c: Case) -> Case {
    let g = &mut c.geometry;
    g.area = r32(g.area);
    g.width = [r32(g.width[0]), r32(g.width[1])];
    g.s_t1 = r32(g.s_t1);
    g.s_t2 = r32(g.s_t2);
    g.torsion_modulus = r32(g.torsion_modulus);
    g.friction_radius = r32(g.friction_radius);
    let k = &mut c.stiffness;
    (k.kn, k.ks, k.kb_t1, k.kb_t2, k.kt) = (r32(k.kn), r32(k.ks), r32(k.kb_t1), r32(k.kb_t2), r32(k.kt));
    let s = &mut c.strength;
    s.tensile = r32(s.tensile);
    s.compressive = r32(s.compressive);
    s.cohesion = r32(s.cohesion);
    s.friction = r32(s.friction);
    s.shear_cap = r32(s.shear_cap);
    s.g_tension = r32(s.g_tension);
    s.g_shear = r32(s.g_shear);
    s.g_compression = r32(s.g_compression);
    s.buckling_load = s.buckling_load.map(r32);
    s.youngs_modulus = r32(s.youngs_modulus);
    s.rate_filter_time = r32(s.rate_filter_time);
    if let Some(d) = &mut s.dif {
        (d.reference_rate, d.exponent, d.transition_rate, d.exponent_high, d.max) = (r32(d.reference_rate), r32(d.exponent), r32(d.transition_rate), r32(d.exponent_high), r32(d.max));
    }
    if let Some(f) = &mut s.static_fatigue {
        (f.exponent, f.test_time) = (r32(f.exponent), r32(f.test_time));
    }
    if let Some(r) = &mut c.rebar {
        (r.k_axial, r.k_dowel, r.yield_force, r.dowel_capacity, r.rupture_work) = (r32(r.k_axial), r32(r.k_dowel), r32(r.yield_force), r32(r.dowel_capacity), r32(r.rupture_work));
    }
    c.weibull = r32(c.weibull);
    c.state = from_gpu_state(&to_gpu_state(&c.state), &c.state);
    let v = |x: Vec3| Vec3::new(r32(x.x), r32(x.y), r32(x.z));
    c.d = Local6 { lin: v(c.d.lin), ang: v(c.d.ang) };
    c.dt = r32(c.dt);
    c
}

fn catalogue_bonds() -> Vec<(BondGeometry, BondStiffness, JointStrength, Option<RebarParams>, f64)> {
    let mut out = Vec::new();
    for scene in stress_ref::builders::catalog().into_iter().chain(stress_ref::showcases::catalog()) {
        let solver = ReferenceSolver::new(&scene);
        for bonds in &solver.bonds {
            for b in bonds.iter().step_by(7) {
                out.push((b.geometry.clone(), b.stiffness, b.strength.clone(), b.rebar, b.weibull));
            }
        }
    }
    out
}

fn random_case(rng: &mut Lcg, bonds: &[(BondGeometry, BondStiffness, JointStrength, Option<RebarParams>, f64)]) -> Case {
    let (geometry, stiffness, mut strength, mut rebar, weibull) = bonds[(rng.next() * bonds.len() as f64) as usize % bonds.len()].clone();
    strength.exact_patch = rng.pick(0.5);
    strength.patch_after_damage = rng.pick(0.3);
    strength.exact_rate_filter = rng.pick(0.5);
    if rebar.is_none() && rng.pick(0.2) {
        let steel = stress_ref::builders::elastic_steel();
        rebar = Some(RebarParams::new(0.01 * geometry.area, &steel, geometry.length, 1.0));
    }
    // Deformation scale: utilization of order one along each component.
    let ft = strength.tensile.max(1.0);
    let lin_scale = ft * geometry.area / stiffness.kn;
    let ang_scale = ft * geometry.s_t1 / stiffness.kb_t1;
    let level = 3.0 * rng.next();
    let d = Local6 {
        lin: Vec3::new(rng.sym() * lin_scale * level, rng.sym() * lin_scale * level, rng.sym() * lin_scale * level),
        ang: Vec3::new(rng.sym() * ang_scale * level, rng.sym() * ang_scale * level, rng.sym() * ang_scale * level),
    };
    let mut state = JointState::new();
    let u = rng.next();
    state.damage = if u < 0.5 { 0.0 } else if u < 0.8 { rng.next() } else { 1.0 };
    state.crush = if rng.pick(0.8) { 0.0 } else { rng.next() };
    state.kappa = if state.damage > 0.0 { 1.0 + rng.next() } else { rng.next() };
    state.kappa_c = if state.crush > 0.0 { 1.0 + rng.next() } else { rng.next() };
    if rng.pick(0.5) {
        state.ductility = 1.0 + 9.0 * rng.next();
        state.ductility_c = 1.0 + 9.0 * rng.next();
    }
    state.fatigue = if rng.pick(0.5) { 0.0 } else { 0.5 * rng.next() };
    state.plastic.lin = [rng.sym() * lin_scale, rng.sym() * lin_scale, 0.0];
    state.plastic.ang = [0.0, 0.0, rng.sym() * ang_scale];
    if rebar.is_some() {
        state.rebar_plastic = rng.sym() * lin_scale;
        state.rebar_slip = [rng.sym() * lin_scale, rng.sym() * lin_scale];
        state.rebar_work = rng.next() * rebar.unwrap().rupture_work;
        state.rebar_broken = rng.pick(0.1);
    }
    state.strain_rate = 10.0 * rng.next();
    state.governing_stress = ft * rng.next();
    let dt = if rng.pick(0.1) { 0.0 } else { 1e-7 + 1e-4 * rng.next() };
    round_case(Case { geometry, stiffness, strength, rebar, weibull, state, d, dt, fracture: rng.pick(0.8) })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TestParams {
    count: u32,
    pad: [u32; 3],
}

#[test]
fn joint_law_matches_the_reference() {
    let gpu = Gpu::new().expect("GPU");
    let bonds = catalogue_bonds();
    let mut rng = Lcg(11);
    let n = 20000;
    let cases: Vec<Case> = (0..n).map(|_| random_case(&mut rng, &bonds)).collect();
    let mut table = MaterialTable::default();
    let gpu_bonds: Vec<GpuJointBond> = cases.iter().map(|c| {
        let m = table.index(&c.strength);
        bond_law(&c.geometry, &c.stiffness, &c.strength, c.rebar.as_ref(), c.weibull, m)
    }).collect();
    let states: Vec<GpuJointState> = cases.iter().map(|c| to_gpu_state(&c.state)).collect();
    let inputs: Vec<[f32; 4]> = cases
        .iter()
        .flat_map(|c| {
            let (l, a) = (c.d.lin, c.d.ang);
            [[l.x as f32, l.y as f32, l.z as f32, c.dt as f32], [a.x as f32, a.y as f32, a.z as f32, if c.fracture { 1.0 } else { 0.0 }]]
        })
        .collect();
    let params = gpu.uniform("params", &TestParams { count: n as u32, pad: [0; 3] });
    let buffers = [
        params,
        gpu.storage("materials", &table.materials),
        gpu.storage("bonds", &gpu_bonds),
        gpu.storage("states", &states),
        gpu.storage("inputs", &inputs),
        gpu.storage("states out", &vec![GpuJointState::default(); n]),
        gpu.storage("outputs", &vec![[0f32; 4]; 5 * n]),
    ];
    let kernel = gpu.compute(
        &stress_gpu::shaders::JOINT_TEST,
        "joint_eval_test",
        &[Binding::Uniform, Binding::Storage, Binding::Storage, Binding::Storage, Binding::Storage, Binding::StorageRw, Binding::StorageRw],
    );
    let refs: Vec<&wgpu::Buffer> = buffers.iter().collect();
    let bind = gpu.bind(&kernel.group, &refs);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        kernel.dispatch(&mut pass, &bind, n);
    }
    gpu.queue.submit([encoder.finish()]);
    let out_states: Vec<GpuJointState> = gpu.read(&buffers[5]);
    // Cost of one evaluation: latency (one group, repeated) and throughput (every case).
    for (threads, reps) in [(n, 50usize), (64, 400), (64, 4000), (n, 500), (1024, 2000)] {
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let t = std::time::Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for _ in 0..reps {
                kernel.dispatch(&mut pass, &bind, threads);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let per = t.elapsed().as_secs_f64() / reps as f64;
        println!("joint law: {threads} evaluations per dispatch: {:.1} us per dispatch, {:.2} ns per evaluation", per * 1e6, per / threads as f64 * 1e9);
    }
    // One group of identical intact cases (no divergence): the latency of the path an
    // intact bond takes in the solver.
    if let Some(k) = (0..n).find(|&k| cases[k].state.damage == 0.0 && cases[k].state.crush == 0.0 && cases[k].rebar.is_none() && cases[k].dt > 0.0) {
        let same = |v: &[[f32; 4]], per: usize| -> Vec<[f32; 4]> { (0..64).flat_map(|_| v[per * k..per * (k + 1)].to_vec()).collect() };
        let one = [
            gpu.uniform("params", &TestParams { count: 64, pad: [0; 3] }),
            gpu.storage("materials", &table.materials),
            gpu.storage("bonds", &vec![gpu_bonds[k]; 64]),
            gpu.storage("states", &vec![states[k]; 64]),
            gpu.storage("inputs", &same(&inputs, 2)),
            gpu.storage("states out", &vec![GpuJointState::default(); 64]),
            gpu.storage("outputs", &vec![[0f32; 4]; 5 * 64]),
        ];
        let refs_one: Vec<&wgpu::Buffer> = one.iter().collect();
        let bind_one = gpu.bind(&kernel.group, &refs_one);
        for reps in [400usize, 4000] {
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            let t = std::time::Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                for _ in 0..reps {
                    kernel.dispatch(&mut pass, &bind_one, 64);
                }
            }
            gpu.queue.submit([encoder.finish()]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            println!("joint law, one intact case x 64: {:.1} us per dispatch", t.elapsed().as_secs_f64() / reps as f64 * 1e6);
        }
    }
    // The same dispatches with nothing to evaluate: the dispatch floor.
    let empty = gpu.uniform("params", &TestParams { count: 0, pad: [0; 3] });
    let mut refs_empty: Vec<&wgpu::Buffer> = buffers.iter().collect();
    refs_empty[0] = &empty;
    let bind_empty = gpu.bind(&kernel.group, &refs_empty);
    for reps in [400usize, 4000] {
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let t = std::time::Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for _ in 0..reps {
                kernel.dispatch(&mut pass, &bind_empty, 64);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        println!("empty dispatch: {:.1} us", t.elapsed().as_secs_f64() / reps as f64 * 1e6);
    }
    let outputs: Vec<[f32; 4]> = gpu.read(&buffers[6]);

    // Worst error per quantity, relative to that case's natural scale.
    let mut worst: std::collections::BTreeMap<&str, (f64, usize)> = Default::default();
    let mut note = |name: &'static str, err: f64, i: usize| {
        let e = worst.entry(name).or_insert((0.0, 0));
        if err > e.0 || err.is_nan() {
            *e = (err, i);
        }
    };
    let mut discrete = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let model = JointModel { geometry: &c.geometry, stiffness: &c.stiffness, strength: &c.strength, rebar: c.rebar.as_ref(), weibull: c.weibull };
        let r = model.evaluate(&c.state, &c.d, c.dt, c.fracture);
        let f = model.secant_factors(&r.state, &c.d);
        let g = &out_states[i];
        let o = &outputs[5 * i..5 * i + 5];
        let k = c.stiffness.as_local();
        let force_scale = c.d.mul_elem(&k).to_array().iter().chain(r.force.to_array().iter()).fold(1e-30f64, |m, x| m.max(x.abs()));
        let energy_scale = 0.5 * c.d.dot(&c.d.mul_elem(&k)) + r.stored.abs() + r.dissipated.abs() + 1e-30;
        let gf = [o[0][0], o[0][1], o[0][2], o[1][0], o[1][1], o[1][2]];
        for (j, x) in r.force.to_array().iter().enumerate() {
            note("force", (gf[j] as f64 - x).abs() / force_scale, i);
        }
        note("dissipated", (o[0][3] as f64 - r.dissipated).abs() / energy_scale, i);
        note("overshoot", (o[1][3] as f64 - r.overshoot).abs() / energy_scale, i);
        note("stored", (o[2][3] as f64 - r.stored).abs() / energy_scale, i);
        let gs = [o[2][0], o[2][1], o[2][2], o[3][0], o[3][1], o[3][2]];
        for (j, x) in f.to_array().iter().enumerate() {
            note("secant factors", (gs[j] as f64 - x).abs() / x.abs().max(1e-6), i);
        }
        note("damage", (g.damage as f64 - r.state.damage).abs(), i);
        note("crush", (g.crush as f64 - r.state.crush).abs(), i);
        note("kappa", (g.kappa as f64 - r.state.kappa).abs() / r.state.kappa.abs().max(1.0), i);
        note("kappa_c", (g.kappa_c as f64 - r.state.kappa_c).abs() / r.state.kappa_c.abs().max(1.0), i);
        note("utilization", (g.utilization as f64 - r.state.utilization).abs() / r.state.utilization.abs().max(1e-3), i);
        note("fatigue", (1.0 - g.life as f64 - r.state.fatigue).abs(), i);
        note("strain_rate", (g.strain_rate as f64 - r.state.strain_rate).abs() / r.state.strain_rate.abs().max(1e-3), i);
        let plastic_scale = c.d.lin.norm() + c.state.plastic.lin[0].abs() + c.state.plastic.lin[1].abs() + 1e-30;
        note("plastic", ((g.plastic_x as f64 - r.state.plastic.lin[0]).abs() + (g.plastic_y as f64 - r.state.plastic.lin[1]).abs()) / plastic_scale, i);
        let twist_scale = c.d.ang.norm() + c.state.plastic.ang[2].abs() + 1e-30;
        note("plastic twist", (g.plastic_t as f64 - r.state.plastic.ang[2]).abs() / twist_scale, i);
        note("rebar", (g.rebar_plastic as f64 - r.state.rebar_plastic).abs() / plastic_scale, i);
        if mode_of(g.mode) != r.state.mode || (o[3][3] != 0.0) != r.disconnected || (g.rebar_broken != 0.0) != r.state.rebar_broken {
            discrete.push(i);
        }
    }
    for (name, (err, i)) in &worst {
        eprintln!("{name:>15}: worst {err:.3e} (case {i})");
    }
    eprintln!("discrete mismatches (mode, disconnection, rebar rupture): {} of {n}", discrete.len());
    for &i in discrete.iter().take(5) {
        let c = &cases[i];
        let model = JointModel { geometry: &c.geometry, stiffness: &c.stiffness, strength: &c.strength, rebar: c.rebar.as_ref(), weibull: c.weibull };
        let r = model.evaluate(&c.state, &c.d, c.dt, c.fracture);
        eprintln!("  case {i}: ref mode {:?} disc {} damage {} util {}; gpu mode {:?} disc {} damage {} util {}", r.state.mode, r.disconnected, r.state.damage, r.state.utilization, mode_of(out_states[i].mode), outputs[5 * i + 3][3], out_states[i].damage, out_states[i].utilization);
    }
    // Provisional bound while the derived per-quantity bounds are worked out.
    for (name, (err, i)) in &worst {
        assert!(*err <= 1e-4, "{name}: error {err:e} at case {i}");
    }
    assert!(discrete.len() <= n / 1000, "too many discrete mismatches: {}", discrete.len());
}
