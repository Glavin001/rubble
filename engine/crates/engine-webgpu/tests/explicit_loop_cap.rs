//! The explicit-beams modules' loop cap (`src/wgsl_loop_cap.rs`): every loop of
//! every shipped module is guarded and the modules validate; and a loop that never
//! ends stops at the cap on the GPU and raises the flag in `gAcc`'s last word.

use engine_webgpu::{
    cap_explicit_loops, default_device, explicit_beams_shipped_wgsl, explicit_loop_cap_hit,
};

/// Per kind, the loops of a capped module: (loops, with a pass counter, with a
/// clamped bound, with a literal bound and so finite as written). Slang writes
/// `loop` and its brace on separate lines.
fn census(wgsl: &str) -> (usize, usize, usize, usize) {
    let b = wgsl.as_bytes();
    let (mut loops, mut counter, mut clamped, mut literal) = (0, 0, 0, 0);
    let mut i = 0;
    while let Some(rel) = wgsl[i..].find("loop") {
        let at = i + rel;
        let after = at + 4;
        let word = (at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_'))
            && (after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_'));
        let rest = wgsl[after..].trim_start();
        if word && rest.starts_with('{') {
            loops += 1;
            let body = rest[1..].trim_start();
            let before = wgsl[..at].trim_end();
            let last_line = before.rsplit('\n').next().unwrap_or("").trim_start();
            if body.starts_with("loopGuard") {
                counter += 1;
            } else if last_line.starts_with("let ebEnd") {
                clamped += 1;
            } else if let Some(c) = body.strip_prefix("if(") {
                let cond = &c[..c.find(')').unwrap() + 1];
                if cond.contains(" < u32(") || cond.contains(" <= u32(") {
                    literal += 1;
                }
            }
        }
        i = after;
    }
    (loops, counter, clamped, literal)
}

#[test]
fn every_shipped_explicit_loop_is_capped_and_valid() {
    let mut total = 0;
    let mut kinds = (0, 0, 0);
    let modules = explicit_beams_shipped_wgsl();
    assert!(!modules.is_empty());
    for (name, _, wgsl) in modules {
        let (loops, counter, clamped, literal) = census(&wgsl);
        assert_eq!(loops, counter + clamped + literal, "{name}: uncapped loop");
        assert_eq!(
            wgsl.matches("ebLoopHit = max(ebLoopHit, ").count(),
            counter + clamped,
            "{name}: a guard without the flag"
        );
        kinds.0 += counter;
        kinds.1 += clamped;
        kinds.2 += literal;
        total += loops;
        let module = naga::front::wgsl::parse_str(&wgsl)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
    // The frozen baseline, reference, test-only and ablation builds (read at run time).
    let dir = std::path::Path::new(engine_webgpu::EXPLICIT_BEAMS_WGSL_DIR);
    let mut files = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.starts_with("explicit_") || !name.ends_with(".wgsl") {
            continue;
        }
        let wgsl = std::fs::read_to_string(&path).unwrap();
        let (loops, counter, clamped, literal) = census(&wgsl);
        if name.starts_with("explicit_ref_")
            && std::env::var("RUBBLE_EB_REF_UNCAPPED").is_ok_and(|v| v == "1")
        {
            continue;
        }
        assert_eq!(loops, counter + clamped + literal, "{name}: uncapped loop");
        files += 1;
    }
    eprintln!(
        "{total} loops in the shipped modules: {} with a pass counter, {} with a clamped bound, {} with a literal bound; {files} explicit module files checked",
        kinds.0, kinds.1, kinds.2
    );
}

#[test]
fn a_runaway_loop_stops_at_the_cap_and_raises_the_flag() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    // A loop whose exit never comes (its exit reads a word the host sets to 0),
    // an ordinary loop, and a counted loop (Slang's shape: a clamped bound) up to
    // a bound the host sets, huge for a runaway.
    let src = "@group(0) @binding(0) var<storage, read_write> gAcc_0 : array<atomic<i32>>;\n\
@group(0) @binding(1) var<storage, read_write> out_0 : array<u32>;\n\
@compute @workgroup_size(1)\n\
fn main() {\n\
    var n: u32 = 0u;\n\
    loop\n\
    {\n\
        if (out_0[1] == 1u) { break; }\n\
        n = n + 1u;\n\
    }\n\
    var i: u32 = 0u;\n\
    loop\n\
    {\n\
        if (i >= 3u) { break; }\n\
        i = i + 1u;\n\
    }\n\
    var m: u32 = out_0[3];\n\
    var c: u32 = 0u;\n\
    var j: u32 = 0u;\n\
    loop\n\
    {\n\
        if(j < m)\n\
        {\n\
        }\n\
        else\n\
        {\n\
            break;\n\
        }\n\
        c = c + 1u;\n\
        j = j + u32(1);\n\
    }\n\
    out_0[0] = n;\n\
    out_0[2] = i;\n\
    out_0[3] = c;\n\
}\n";
    let capped = cap_explicit_loops(src).unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("loop-cap"),
        source: wgpu::ShaderSource::Wgsl(capped.clone().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("loop-cap"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    assert!(
        capped.contains("let ebEnd2"),
        "the counted loop takes a clamped bound"
    );
    use wgpu::util::DeviceExt;
    let run = |init: [u32; 4]| -> (Vec<u32>, Vec<u32>) {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let acc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0i32; 4]),
            usage,
        });
        let out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&init),
            usage,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: acc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let read = |b: &wgpu::Buffer| -> Vec<u32> {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(b, 0, &staging, 0, 16);
            queue.submit([enc.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let v = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
            v
        };
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([enc.finish()]);
        (read(&out), read(&acc))
    };
    let cap = engine_webgpu::EXPLICIT_LOOP_CAP;
    // Ordinary bounds: every loop runs as written and no flag.
    let (o, a) = run([0, 1, 0, 5]);
    assert_eq!((o[0], o[2], o[3]), (0, 3, 5));
    assert_eq!(a, vec![0, 0, 0, 0]);
    // Runaways: the first loop and the counted one stop at the cap; the flag holds
    // the highest stopped loop's number (3) in gAcc's last word only.
    let (o, a) = run([0, 0, 0, u32::MAX]);
    assert_eq!(o[0], cap, "passes of the runaway loop");
    assert_eq!(o[2], 3);
    assert_eq!(o[3], cap, "passes of the runaway counted loop");
    assert_eq!(&a[..3], &[0, 0, 0]);
    assert_eq!(a[3], 3);
    // The counted loop alone.
    let (o, a) = run([0, 1, 0, u32::MAX]);
    assert_eq!((o[0], o[3]), (0, cap));
    assert_eq!(a[3], 3);
    let (_, a) = run([0, 0, 0, 7]);
    assert_eq!(a[3], 1);
    let e = explicit_loop_cap_hit(a[3] as i32).unwrap_err();
    assert!(e.contains("number 1"), "{e}");
}
