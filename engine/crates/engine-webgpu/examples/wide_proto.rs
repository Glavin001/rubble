//! Time-only prototype for wide-frame-design.md: a 2 kHz substep as two data-parallel
//! passes over every car at once, a beam pass (one thread a beam: spring-damper force
//! from its two nodes, written per beam) and a node pass (one thread a node: its
//! incident beams' forces gathered in incidence order, gravity, symplectic Euler), each
//! a dependent dispatch, 34 substeps a frame in one compute pass. The real topology
//! (beams, incidence lists) of the car given; not the solver (no yield, breaks, contact,
//! self-collision): the cost of the wide structure's chain itself. Native wgpu; prints
//! ms a frame (median of 30) and the module compile times.
//!
//!   cargo run --release -p engine-webgpu --example wide_proto -- <truck|jbeam-dir:pc> 1 32 128
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use wgpu::util::DeviceExt;

const BEAMS_WGSL: &str = r#"
struct P { beams: u32, nodes: u32, dt: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> beam: array<vec4<u32>>;     // a, b, k bits, rest bits
@group(0) @binding(2) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> force: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x;
  if (j >= p.beams) { return; }
  let b = beam[j];
  let d = pos[b.y].xyz - pos[b.x].xyz;
  let l = length(d);
  let n = select(vec3<f32>(0.0), d / l, l > 0.0);
  let k = bitcast<f32>(b.z);
  let rest = bitcast<f32>(b.w);
  let rv = dot(vel[b.y].xyz - vel[b.x].xyz, n);
  force[j] = vec4<f32>(n * (k * (l - rest) + 0.02 * k * 0.001 * rv), 0.0);
}
"#;

const NODES_WGSL: &str = r#"
struct P { beams: u32, nodes: u32, dt: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> inc_off: array<u32>;   // nodes + 1
@group(0) @binding(2) var<storage, read> inc: array<u32>;       // beam index | sign bit 31
@group(0) @binding(3) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> force: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= p.nodes) { return; }
  var f = vec3<f32>(0.0, -9.81, 0.0) * vel[i].w;
  let e1 = min(inc_off[i + 1u], inc_off[i] + 64u);
  for (var e = inc_off[i]; e < e1; e++) {
    let w = inc[e];
    let fb = force[w & 0x7fffffffu].xyz;
    f += select(fb, -fb, (w & 0x80000000u) != 0u);
  }
  let m = max(vel[i].w, 1e-3);
  var v = vel[i].xyz + f / m * p.dt;
  var x = pos[i].xyz + v * p.dt;
  if (x.y < 0.0) { x.y = 0.0; v.y = 0.0; }
  vel[i] = vec4<f32>(v, vel[i].w);
  pos[i] = vec4<f32>(x, pos[i].w);
}
"#;


const WG_WGSL: &str = r#"
struct P { beams: u32, nodes: u32, dt: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> beam: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> inc_off: array<u32>;
@group(0) @binding(3) var<storage, read> inc: array<u32>;
@group(0) @binding(4) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> force: array<vec4<f32>>;
// One workgroup a car (beams p.beams / cars each: nb = p.pad), 34 substeps in one dispatch,
// a barrier after each phase: the in-workgroup frame's structure on the same work.
@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let car = g.x;
  let nb = p.pad & 0xffffu;
  let nn = p.pad >> 16u;
  for (var s = 0u; s < 34u; s++) {
    for (var jj = l.x; jj < nb; jj += LANES) {
      let j = car * nb + jj;
      let b = beam[j];
      let d = pos[b.y].xyz - pos[b.x].xyz;
      let len = length(d);
      let n = select(vec3<f32>(0.0), d / len, len > 0.0);
      let k = bitcast<f32>(b.z);
      let rest = bitcast<f32>(b.w);
      let rv = dot(vel[b.y].xyz - vel[b.x].xyz, n);
      force[j] = vec4<f32>(n * (k * (len - rest) + 0.02 * k * 0.001 * rv), 0.0);
    }
    storageBarrier(); workgroupBarrier();
    for (var ii = l.x; ii < nn; ii += LANES) {
      let i = car * nn + ii;
      var f = vec3<f32>(0.0, -9.81, 0.0) * vel[i].w;
      let e1 = min(inc_off[i + 1u], inc_off[i] + 64u);
      for (var e = inc_off[i]; e < e1; e++) {
        let w = inc[e];
        let fb = force[w & 0x7fffffffu].xyz;
        f += select(fb, -fb, (w & 0x80000000u) != 0u);
      }
      let m = max(vel[i].w, 1e-3);
      var v = vel[i].xyz + f / m * p.dt;
      var x = pos[i].xyz + v * p.dt;
      if (x.y < 0.0) { x.y = 0.0; v.y = 0.0; }
      vel[i] = vec4<f32>(v, vel[i].w);
      pos[i] = vec4<f32>(x, pos[i].w);
    }
    storageBarrier(); workgroupBarrier();
  }
}
"#;

fn load(spec: &str) -> ExplicitBeams {
    if spec.ends_with(".truck") {
        let text = std::fs::read_to_string(spec).expect("truck");
        return ExplicitBeams::from_truck(&text, glam::Vec3::ZERO, glam::Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap();
    }
    let (dir, pc) = spec.split_once(':').unwrap_or((spec, "sedan/sedan.pc"));
    let dir = std::path::Path::new(dir);
    let list = std::fs::read_to_string(dir.join("files.json")).unwrap();
    let names: Vec<&str> = list.split('"').filter(|s| s.ends_with(".jbeam")).collect();
    let files: Vec<(String, String)> = names.iter().map(|f| (f.to_string(), std::fs::read_to_string(dir.join(f)).unwrap())).collect();
    let refs: Vec<(&str, &str)> = files.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
    let pc = std::fs::read_to_string(dir.join(pc)).ok();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap();
    let (sim, _) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        engine_softbody::explicit_beams::JbeamWheels::Pressure,
        glam::Vec3::ZERO,
        glam::Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )
    .unwrap();
    sim
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let car = load(&args[0]);
    let counts: Vec<usize> = args[1..].iter().filter_map(|a| a.parse().ok()).collect();
    // WIDE_PROTO_DUMP=<file>: the car's topology for the Chrome half
    // (tools/wide-frame/proto.html): u32 words nodes, beams, then per beam (a, b, k, rest),
    // per node (x, y, z, mass) as f32 bits, then nodes + 1 incidence offsets and the
    // incidence words (beam | sign bit 31).
    if let Ok(path) = std::env::var("WIDE_PROTO_DUMP") {
        let mut w: Vec<u32> = vec![car.nodes.len() as u32, car.beams.len() as u32];
        for b in &car.beams {
            w.extend([b.a, b.b, b.k.to_bits(), b.rest.to_bits()]);
        }
        for x in &car.nodes {
            w.extend([x.pos.x.to_bits(), (x.pos.y + 0.5).to_bits(), x.pos.z.to_bits(), x.mass.max(0.1).to_bits()]);
        }
        let mut off = vec![0u32];
        let mut inc = Vec::new();
        for i in 0..car.nodes.len() {
            for &j in &car.incidence[i] {
                inc.push(j | if car.beams[j as usize].b as usize == i { 0x8000_0000 } else { 0 });
            }
            off.push(inc.len() as u32);
        }
        w.extend(off);
        w.extend(inc);
        std::fs::write(&path, bytemuck::cast_slice(&w)).unwrap();
        println!("topology written to {path}");
        return;
    }
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    let t = std::time::Instant::now();
    let module = |src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipe = |m: &wgpu::ShaderModule| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let (bm, nm) = (module(BEAMS_WGSL), module(NODES_WGSL));
    let (bp, np) = (pipe(&bm), pipe(&nm));
    println!(
        "car: {} nodes, {} beams; compile both passes (WGSL {} + {} bytes): {:.0} ms",
        car.nodes.len(),
        car.beams.len(),
        BEAMS_WGSL.len(),
        NODES_WGSL.len(),
        t.elapsed().as_secs_f64() * 1e3
    );
    for n in if counts.is_empty() { vec![1, 32] } else { counts } {
        let (nn, nb) = (car.nodes.len() as u32, car.beams.len() as u32);
        let mut beam = Vec::new();
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let mut off = vec![0u32];
        let mut inc = Vec::new();
        for c in 0..n as u32 {
            for b in &car.beams {
                beam.push([b.a + c * nn, b.b + c * nn, b.k.to_bits(), b.rest.to_bits()]);
            }
            for (i, x) in car.nodes.iter().enumerate() {
                pos.push([x.pos.x + c as f32 * 8.0, x.pos.y + 0.5, x.pos.z, 0.0]);
                vel.push([0.0, 0.0, 0.0, x.mass.max(0.1)]);
                for &j in &car.incidence[i] {
                    let bj = &car.beams[j as usize];
                    let sign = if bj.b as usize == i { 0x8000_0000u32 } else { 0 };
                    inc.push((j + c * nb) | sign);
                }
                off.push(inc.len() as u32);
            }
        }
        let (tn, tb) = (n as u32 * nn, n as u32 * nb);
        let st = wgpu::BufferUsages::STORAGE;
        let mk = |d: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: d, usage: st });
        let beam_b = mk(bytemuck::cast_slice(&beam));
        let pos_b = mk(bytemuck::cast_slice(&pos));
        let vel_b = mk(bytemuck::cast_slice(&vel));
        let off_b = mk(bytemuck::cast_slice(&off));
        let inc_b = mk(bytemuck::cast_slice(&inc));
        let force_b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: tb as u64 * 16, usage: st, mapped_at_creation: false });
        let pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[tb, tn, 0.0005f32.to_bits(), 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = |p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect::<Vec<_>>(),
            })
        };
        let pw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[tb, tn, 0.0005f32.to_bits(), nb | (nn << 16)]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let gb = bg(&bp, &[&pu, &beam_b, &pos_b, &vel_b, &force_b]);
        let gn = bg(&np, &[&pu, &off_b, &inc_b, &pos_b, &vel_b, &force_b]);
        let frame = |substeps: u32| {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..substeps {
                    pass.set_pipeline(&bp);
                    pass.set_bind_group(0, &gb, &[]);
                    pass.dispatch_workgroups(tb.div_ceil(64), 1, 1);
                    pass.set_pipeline(&np);
                    pass.set_bind_group(0, &gn, &[]);
                    pass.dispatch_workgroups(tn.div_ceil(64), 1, 1);
                }
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        for _ in 0..10 {
            frame(34);
        }
        let mut ms = Vec::new();
        for _ in 0..30 {
            let t = std::time::Instant::now();
            frame(34);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        // The same with one substep (submit + wait cost, the frame's fixed part).
        let mut one = Vec::new();
        for _ in 0..30 {
            let t = std::time::Instant::now();
            frame(1);
            one.push(t.elapsed().as_secs_f64() * 1e3);
        }
        one.sort_by(f64::total_cmp);
        println!(
            "{n} cars ({tn} nodes, {tb} beams): WIDE 34 substeps x (beam pass + node pass) {:.3} ms a frame (median, wall); 1 substep {:.3} ms; per substep {:.1} us",
            ms[15],
            one[15],
            (ms[15] - one[15]) / 33.0 * 1e3
        );
        // The same work in one workgroup a car, 34 substeps in one dispatch, barriers.
        for lanes in [256u32, 512] {
            let t = std::time::Instant::now();
            let m = module(&WG_WGSL.replace("LANES", &format!("{lanes}u")).replace("workgroup_size({lanes}u)", &format!("workgroup_size({lanes})")));
            let wp = pipe(&m);
            let cms = t.elapsed().as_secs_f64() * 1e3;
            let gw = bg(&wp, &[&pw, &beam_b, &off_b, &inc_b, &pos_b, &vel_b, &force_b]);
            let run = || {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&wp);
                    pass.set_bind_group(0, &gw, &[]);
                    pass.dispatch_workgroups(n as u32, 1, 1);
                }
                queue.submit([enc.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            };
            for _ in 0..10 {
                run();
            }
            let mut w = Vec::new();
            for _ in 0..30 {
                let t = std::time::Instant::now();
                run();
                w.push(t.elapsed().as_secs_f64() * 1e3);
            }
            w.sort_by(f64::total_cmp);
            println!("{n} cars: WORKGROUP ({lanes} lanes a car) 34 substeps, the same two phases with barriers: {:.3} ms a frame (median, wall; compile {cms:.0} ms)", w[15]);
        }
    }
}
