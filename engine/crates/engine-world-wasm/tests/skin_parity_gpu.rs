//! Weighted skins (`engine_softbody::flexskin`): the renderer's
//! `weightedVertex()` (read from `showcase-car-render.mjs`, the shader the
//! pages run) against the CPU's `flexskin::eval` on the same deformed nodes.
//! Every vertex of a bound lattice, packed as `showcase-car-render.mjs`
//! packs them (a 64-byte Bind per vertex, its entries 4 to a 64-byte slot
//! after the vertices), run on the GPU, read back and compared.
//!
//! Skips (passes) without a GPU adapter.
use engine_softbody::flexskin;
use glam::{Quat, Vec3};

const RENDER: &str = include_str!("../../engine-webgpu/showcase-car-render.mjs");

/// The renderer's `struct Bind` line and its weighted-skin functions.
fn renderer_wgsl() -> String {
    let bind = RENDER
        .lines()
        .find(|l| l.starts_with("struct Bind {"))
        .expect("struct Bind in showcase-car-render.mjs");
    let a = RENDER.find("struct Weighted {").expect("struct Weighted");
    let f = RENDER[a..]
        .find("fn weightedVertex(")
        .expect("fn weightedVertex")
        + a;
    let end = RENDER[f..].find("\n}\n").expect("end of weightedVertex") + f + 3;
    format!("{bind}\n{}", &RENDER[a..end])
}

fn shader() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(1) var<storage, read> binds: array<Bind>;
@group(0) @binding(2) var<storage, read_write> outv: array<vec4f>;
@group(0) @binding(3) var<uniform> count: vec4u;
fn nodePos(i: u32) -> vec3f {{ return nodes[i].xyz; }}
{}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {{
  let i = gid.x;
  if (i >= count.x) {{ return; }}
  let r = weightedVertex(binds[i]);
  outv[2u * i] = vec4f(r.pos, 0.0);
  outv[2u * i + 1u] = vec4f(r.nrm, 0.0);
}}
"#,
        renderer_wgsl()
    )
}

/// A two-layer panel lattice and points over it (as `flexskin`'s tests).
fn scene() -> (Vec<u32>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
    let mut p = Vec::new();
    for i in 0..7 {
        for j in 0..5 {
            for k in 0..2 {
                p.push(Vec3::new(
                    i as f32 * 0.45,
                    j as f32 * 0.4,
                    k as f32 * 0.07 + 0.03 * (i as f32 * 0.7).sin(),
                ));
            }
        }
    }
    let n = 4000;
    let v: Vec<Vec3> = (0..n)
        .map(|k| {
            let f = k as f32;
            Vec3::new(
                0.1 + 2.5 * (f * 0.3719).fract(),
                0.1 + 1.4 * (f * 0.6127).fract(),
                0.08 + 0.02 * (f * 0.1313).fract(),
            )
        })
        .collect();
    let nr: Vec<Vec3> = (0..n)
        .map(|k| Vec3::new(0.2 * (k as f32 * 0.7).sin(), 0.1, 1.0).normalize())
        .collect();
    ((0..p.len() as u32).collect(), p, v, nr)
}

#[test]
fn weighted_vertex_matches_flexskin_eval() {
    let Ok((_, device, queue)) = engine_webgpu::default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let (ids, rest, verts, nrms) = scene();
    let (skins, _) = flexskin::bind(&ids, &rest, &verts, &nrms).unwrap();
    // A deformed state: a rigid motion, a bend and a crush.
    let q = Quat::from_rotation_y(0.4) * Quat::from_rotation_x(-0.2);
    let now: Vec<Vec3> = rest
        .iter()
        .enumerate()
        .map(|(k, &x)| {
            let a = x.x / 2.2;
            let bent = Vec3::new(
                2.2 * a.sin() - x.z * a.sin(),
                x.y,
                2.2 * (1.0 - a.cos()) + x.z * a.cos(),
            );
            let crush = Vec3::new(0.0, 0.0, 0.05 * ((k * 7919) % 13) as f32 / 13.0);
            q * (bent + crush) + Vec3::new(1.0, 0.5, -2.0)
        })
        .collect();
    // Pack as showcase-car-render.mjs: entries by |A|, descending.
    let n = verts.len();
    let mut slots = 0usize;
    let first: Vec<usize> = skins
        .iter()
        .map(|s| {
            let f = n + slots;
            slots += s.w.len().div_ceil(4);
            f
        })
        .collect();
    let mut words = vec![0u32; (n + slots) * 16];
    let mut sorted = Vec::with_capacity(n);
    for (i, s) in skins.iter().enumerate() {
        let mut w = s.w.clone();
        w.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        let o = i * 16;
        words[o] = w.len() as u32;
        words[o + 3] = first[i] as u32;
        for (k, x) in s.off.iter().enumerate() {
            words[o + 4 + k] = x.to_bits();
        }
        for (k, x) in s.nrm.iter().enumerate() {
            words[o + 8 + k] = x.to_bits();
        }
        words[o + 15] = 1 << 25;
        for (e, &(nd, a, da, db)) in w.iter().enumerate() {
            let q = first[i] * 16 + 4 * e;
            words[q] = nd;
            words[q + 1] = a.to_bits();
            words[q + 2] = da.to_bits();
            words[q + 3] = db.to_bits();
        }
        sorted.push(w);
    }
    let node_words: Vec<f32> = now.iter().flat_map(|p| [p.x, p.y, p.z, 0.0]).collect();
    use wgpu::util::DeviceExt;
    let buf = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        })
    };
    let nodes_b = buf(
        "nodes",
        bytemuck::cast_slice(&node_words),
        wgpu::BufferUsages::STORAGE,
    );
    let binds_b = buf(
        "binds",
        bytemuck::cast_slice(&words),
        wgpu::BufferUsages::STORAGE,
    );
    let out_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (n * 32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let count_b = buf(
        "count",
        bytemuck::cast_slice(&[n as u32, 0, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skin-parity"),
        source: wgpu::ShaderSource::Wgsl(shader().into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("skin-parity"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nodes_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: binds_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: count_b.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
    }
    queue.submit([enc.finish()]);
    let got =
        engine_webgpu::explicit_coupling::read_words(&device, &queue, &out_b, 0, (n * 32) as u64);
    let f = |k: usize| f32::from_bits(got[k]);
    let (mut dp, mut dn, mut bits) = (0.0f32, 0.0f32, 0usize);
    for i in 0..n {
        let (p, nn) = flexskin::eval(&sorted[i], skins[i].off, skins[i].nrm, &now);
        let gp = Vec3::new(f(8 * i), f(8 * i + 1), f(8 * i + 2));
        let gn = Vec3::new(f(8 * i + 4), f(8 * i + 5), f(8 * i + 6));
        dp = dp.max(gp.distance(p));
        // The difference of the unit normals (acos near 1 loses f32 precision).
        dn = dn.max(gn.distance(nn));
        if gp == p && gn == nn {
            bits += 1;
        }
    }
    println!("{n} vertices: position {dp:e} m, normal {dn:e} (unit-vector difference) at most; {bits} bit for bit");
    assert!(dp <= 2e-6, "position {dp}");
    assert!(dn <= 2e-5, "normal {dn}");
}
