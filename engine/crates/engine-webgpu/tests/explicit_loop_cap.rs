//! The explicit-beams modules' loop caps. The shipped kernel caps its loops in its
//! Slang source (`slang/kernels/explicit_loop_cap.slang`: `EB_END`, `EB_PASS`), so
//! every target has them (WGSL here; MSL and CUDA: engine-metal's and engine-cuda's
//! tests/explicit_loop_cap.rs). Checked here: every loop of `explicit_beams.slang`
//! and its includes is constant, clamped or counted; every shipped module carries
//! the caps and validates; the frozen test modules (baseline, reference), capped by
//! the WGSL post-process (`src/wgsl_loop_cap.rs`), are capped in every loop; and on
//! the GPU a loop that never ends stops at the cap and raises the flag in `gAcc`'s
//! last word, through the Slang caps (`explicit_loop_cap_test.slang`) and through
//! the post-process.

use engine_exec::explicit_plan::{EXPLICIT_LOOP_CAP, EXPLICIT_LOOP_CAP_TAG};
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

/// The Slang source with `//` comments blanked (same length, lines kept).
fn strip_comments(s: &str) -> String {
    s.lines()
        .map(|l| match l.find("//") {
            Some(at) => format!("{}{}", &l[..at], " ".repeat(l.len() - at)),
            None => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The index past the bracket closing the one at `open` (`(` or `{`).
fn close_of(s: &[u8], open: usize) -> usize {
    let (o, c) = (s[open], if s[open] == b'(' { b')' } else { b'}' });
    let mut depth = 0i32;
    for (i, &ch) in s.iter().enumerate().skip(open) {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return i + 1;
            }
        }
    }
    panic!("unbalanced bracket at {open}")
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// A constant: a literal (`3u`, `16`) or a static const (`GRID_SUBS`).
fn constant(e: &str) -> bool {
    let e = e.trim();
    // A sum of constants (`SELF_ROUNDS + HIT_CAP + 1u`).
    if e.contains('+') {
        return e.split('+').all(constant);
    }
    let lit = e.trim_end_matches('u');
    (!lit.is_empty() && lit.bytes().all(|c| c.is_ascii_digit()))
        || (!e.is_empty()
            && e.as_bytes()[0].is_ascii_uppercase()
            && e.bytes().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_'))
}

/// A constant bound: [`constant`], or `min(expr, C)` / `min(expr, int(C))`.
fn constant_bound(e: &str) -> bool {
    let e = e.trim();
    if constant(e) {
        return true;
    }
    let Some(inner) = e.strip_prefix("min(").and_then(|r| r.strip_suffix(')')) else {
        return false;
    };
    let Some((_, c)) = inner.rsplit_once(',') else { return false };
    let c = c.trim();
    constant(c.strip_prefix("int(").and_then(|r| r.strip_suffix(')')).unwrap_or(c))
}

/// Whether `body` assigns `var` (`var =`, `var +=`, `var++`, `--var` and the like).
fn assigns(body: &str, var: &str) -> bool {
    let b = body.as_bytes();
    let mut from = 0;
    while let Some(rel) = body[from..].find(var) {
        let at = from + rel;
        from = at + var.len();
        let end = at + var.len();
        if (at > 0 && (is_ident(b[at - 1]) || b[at - 1] == b'.')) || (end < b.len() && is_ident(b[end])) {
            continue;
        }
        let before = body[..at].trim_end();
        let after = body[end..].trim_start();
        if before.ends_with("++") || before.ends_with("--") || after.starts_with("++") || after.starts_with("--") {
            return true;
        }
        let op = ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="];
        if op.iter().any(|o| after.starts_with(o)) && !after.starts_with("==") {
            return true;
        }
    }
    false
}

/// Every `for` / `while` loop of a Slang source that has none of the three forms
/// of `explicit_loop_cap.slang` (constant; `X < XStop` with `XStop = EB_END(X, ..)`;
/// `EB_PASS(..)` first in its body), as (line, header).
fn uncapped_slang_loops(text: &str) -> Vec<(usize, String)> {
    let s = strip_comments(text);
    let b = s.as_bytes();
    let mut bad = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let kw = if s[i..].starts_with("for") {
            3
        } else if s[i..].starts_with("while") {
            5
        } else {
            i += 1;
            continue;
        };
        if i > 0 && is_ident(b[i - 1]) || is_ident(b[i + kw]) {
            i += 1;
            continue;
        }
        let mut open = i + kw;
        while b[open].is_ascii_whitespace() {
            open += 1;
        }
        if b[open] != b'(' {
            i += 1;
            continue;
        }
        let close = close_of(b, open);
        let header = &s[open + 1..close - 1];
        // The body: a block, or one statement (itself possibly a loop).
        let mut at = close;
        while b[at].is_ascii_whitespace() {
            at += 1;
        }
        let body = if b[at] == b'{' {
            &s[at..close_of(b, at)]
        } else {
            let mut end = at;
            loop {
                let rest = &s[end..];
                let t = rest.trim_start();
                let skip = rest.len() - t.len();
                let head = ["for", "while", "if"].iter().find(|k| {
                    t.starts_with(**k) && t[k.len()..].trim_start().starts_with('(')
                });
                match head {
                    Some(k) => {
                        let p = end + skip + k.len() + t[k.len()..].find('(').unwrap();
                        end = close_of(b, p);
                        let r = s[end..].trim_start();
                        if r.starts_with('{') {
                            let q = end + (s[end..].len() - r.len());
                            end = close_of(b, q);
                            break;
                        }
                    }
                    None => {
                        end += s[end..].find(';').unwrap() + 1;
                        break;
                    }
                }
            }
            &s[at..end]
        };
        let line = s[..i].matches('\n').count() + 1;
        let first = body.trim_start().strip_prefix('{').unwrap_or(body).trim_start();
        let counted = first.starts_with("EB_PASS(") || first.starts_with("EB_PASS_UNIFORM(");
        let ok = counted || (kw == 3 && for_capped(header, body, &s[..i]));
        if !ok {
            bad.push((line, header.split_whitespace().collect::<Vec<_>>().join(" ")));
        }
        i = close;
    }
    bad
}

/// A `for` header with a constant bound or a clamped end (`before`: the source
/// before the loop, for an end taken just before it).
fn for_capped(header: &str, body: &str, before: &str) -> bool {
    let parts: Vec<&str> = header.split(';').map(str::trim).collect();
    if parts.len() != 3 {
        return false;
    }
    let (init, cond, step) = (parts[0], parts[1], parts[2]);
    let first = cond.split("&&").next().unwrap().trim();
    let (var, op, bound) = if let Some((v, e)) = first.split_once(" <= ") {
        (v.trim(), "<=", e.trim())
    } else if let Some((v, e)) = first.split_once(" < ") {
        (v.trim(), "<", e.trim())
    } else {
        return false;
    };
    if var.is_empty() || !var.bytes().all(is_ident) {
        return false;
    }
    let step_ok = |c: &dyn Fn(&str) -> bool| {
        step == format!("{var}++")
            || step == format!("++{var}")
            || step.strip_prefix(&format!("{var} += ")).is_some_and(|s| c(s))
    };
    // A clamped end: `X < XStop`, XStop = EB_END(X, ..) in the header or the
    // statement just before the loop.
    let stop = format!("{var}Stop");
    if op == "<" && bound == stop && step_ok(&|_| true) {
        let prev = before.trim_end();
        let prev_stmt = prev[..prev.len().saturating_sub(1)].rsplit(';').next().unwrap_or("");
        // EB_END_UNIFORM: the same clamp, for a loop that reaches a barrier (not budgeted).
        return ["EB_END", "EB_END_UNIFORM"].iter().any(|m| {
            let is_def = |text: &str| end_of(text, &stop, m).is_some_and(|args| args.get(1) == Some(&var));
            is_def(init) || (init.is_empty() && prev.ends_with(';') && is_def(prev_stmt))
        });
    }
    // A constant loop: constant bound and step, the counter assigned only by the step.
    let lit_step = step_ok(&|s| constant(s));
    constant_bound(bound) && lit_step && !assigns(body, var)
}

/// The arguments of `STOP = MACRO(ID, X, ..)` in `text` (None: no such definition).
fn end_of<'t>(text: &'t str, stop: &str, mac: &str) -> Option<Vec<&'t str>> {
    let def = format!("{stop} = {mac}(");
    let at = text.find(&def)? + def.len();
    let args = &text[at..at + close_paren(&text[at..])?];
    Some(args.split(',').map(str::trim).collect())
}

/// The index of the `)` closing an argument list that starts at `s`'s beginning.
fn close_paren(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => return Some(i),
            ')' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// The value of `static const uint NAME = 1u << N;` in a Slang source.
fn shift_const(src: &str, name: &str) -> u32 {
    let pre = format!("static const uint {name} = 1u << ");
    let line = src.lines().find(|l| l.starts_with(&pre)).unwrap_or_else(|| panic!("no {name}"));
    1 << line[pre.len()..].trim_end_matches(';').parse::<u32>().unwrap()
}

#[test]
fn every_explicit_slang_loop_is_capped() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang/kernels");
    // The file and everything it includes in this directory, followed through the
    // includes (explicit_beams_frame.slang, explicit_loop_cap.slang, ...).
    let mut files = vec![("explicit_beams.slang".to_string(), std::fs::read_to_string(dir.join("explicit_beams.slang")).unwrap())];
    let mut at = 0;
    while at < files.len() {
        let text = files[at].1.clone();
        at += 1;
        for l in text.lines() {
            let Some(inc) = l.trim().strip_prefix("#include \"").and_then(|r| r.strip_suffix('"')) else { continue };
            if inc.contains('/') || files.iter().any(|f| f.0 == inc) {
                continue;
            }
            files.push((inc.to_string(), std::fs::read_to_string(dir.join(inc)).unwrap()));
        }
    }
    assert!(files.iter().any(|f| f.0 == "explicit_loop_cap.slang"), "the caps are included");
    // The World's coupling passes carry the same caps and budget.
    let coupling = std::fs::read_to_string(dir.join("vehicle_coupling.slang")).unwrap();
    assert!(coupling.contains("#include \"explicit_loop_cap.slang\""), "the coupling passes include the caps");
    files.push(("vehicle_coupling.slang".to_string(), coupling));
    let mut loops = 0;
    for (name, text) in &files {
        let bad = uncapped_slang_loops(text);
        assert!(
            bad.is_empty(),
            "{name}: loops without a cap (see explicit_loop_cap.slang): {bad:#?}"
        );
        loops += strip_comments(text).matches("for (").count();
    }
    assert!(loops > 150, "the scan found the kernel's loops ({loops})");
    // Each entry point that flushes the flag does so at every return.
    let main: String = files.iter().map(|f| f.1.as_str()).collect::<Vec<_>>().join("\n");
    let s = strip_comments(&main);
    for (at, _) in s.match_indices("[shader(\"compute\")]") {
        let open = at + s[at..].find('{').unwrap();
        let body = &s[open..close_of(s.as_bytes(), open)];
        let name = s[at..open].split("void ").nth(1).unwrap_or("").split('(').next().unwrap_or("");
        if body.contains("ebLoopCapFlush") {
            // (A bare return right after the flush is one too.)
            for (r, _) in body.match_indices("return;") {
                assert!(body[..r].trim_end().ends_with("ebLoopCapFlush();"), "{name}: a return without ebLoopCapFlush");
            }
            // (The last statement, past any closing `#endif`: a body split by the
            // preprocessor flushes at the end of each branch.)
            let end: Vec<&str> = body.trim_end_matches('}').lines().map(str::trim).filter(|l| !l.is_empty()).collect();
            let last = end.iter().rev().find(|l| !l.starts_with('#')).copied().unwrap_or("");
            assert!(last.ends_with("ebLoopCapFlush();"), "{name}: no flush at its end");
        }
    }
    // The scan rejects what it should.
    let bad = uncapped_slang_loops(
        "for (uint i = 0u; i < n; i++) x++;\nwhile (j > 0u) { j--; }\nfor (uint k = 0u; k < 4u; k++) k = 0u;\n\
         for (uint i = 0u, iStop = EB_END(1, i, n, 1u); i < iStop; i++) {}\nwhile (a) { EB_PASS(2, g); }\n\
         for (uint k = 0u; k < KEYS; k += 2u) {}\nfor (int x = 0; x < min(s.x, int(VV_MAX_SPAN)); x++) {}\n\
         uint kStop = EB_END(3, k, n, 1u);\nfor (; k < kStop; k++) {}\n",
    );
    assert_eq!(bad.iter().map(|b| b.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    // The Slang constants are the host's.
    let caps = std::fs::read_to_string(dir.join("explicit_loop_cap.slang")).unwrap();
    assert_eq!(shift_const(&caps, "EB_LOOP_CAP"), EXPLICIT_LOOP_CAP);
    assert_eq!(shift_const(&caps, "EB_LOOP_CAP_TAG") as i32, EXPLICIT_LOOP_CAP_TAG);
    assert_eq!(engine_webgpu::EXPLICIT_LOOP_CAP, EXPLICIT_LOOP_CAP, "the WGSL post-process's cap");
}

/// Every loop cap names its loop by an id of its own (`EB_END(ID, ..)`, `EB_PASS(ID, ..)`,
/// `ebLoopCapHit(ID)`; explicit_loop_cap.slang): the flag's id finds the loop, and an
/// edit elsewhere changes no other module. Each id is a number, used once.
#[test]
fn every_loop_cap_has_an_id_of_its_own() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang/kernels");
    let mut seen: std::collections::HashMap<u32, String> = std::collections::HashMap::new();
    let mut bad = Vec::new();
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    names.sort();
    for path in names {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".slang") || name == "explicit_loop_cap.slang" {
            continue;
        }
        let text = strip_comments(&std::fs::read_to_string(&path).unwrap());
        for mac in ["EB_END_UNIFORM(", "EB_END(", "EB_PASS_UNIFORM(", "EB_PASS(", "ebLoopCapHit("] {
            for (at, _) in text.match_indices(mac) {
                // (EB_END( inside EB_END_UNIFORM( is not a match: a letter or _ before.)
                if text[..at].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                let line = text[..at].matches('\n').count() + 1;
                let rest = &text[at + mac.len()..];
                let arg = rest[..rest.find([',', ')']).unwrap_or(rest.len())].trim();
                match arg.parse::<u32>() {
                    Ok(id) if id > 0 => {
                        if let Some(first) = seen.insert(id, format!("{name}:{line}")) {
                            bad.push(format!("{name}:{line}: id {id} is also {first}'s"));
                        }
                    }
                    _ => bad.push(format!("{name}:{line}: {mac}{arg}, ..): no id (a number, the loop's own)")),
                }
            }
        }
    }
    let next = seen.keys().max().copied().unwrap_or(0) + 1;
    assert!(bad.is_empty(), "loop-cap ids (the next free one: {next}):\n{}", bad.join("\n"));
    assert!(seen.len() > 500, "the scan found the kernels' caps ({})", seen.len());
}

/// Whether a module carries the Slang caps (explicit_loop_cap.slang): its cap functions
/// (the counted loops' ends raise the flag without a call of ebLoopCapHit).
fn slang_caps(wgsl: &str) -> bool {
    ["fn ebLoopCapHit_0(", "fn ebEnd_0(", "fn ebEndMost_0("].iter().any(|f| wgsl.contains(f))
}

#[test]
fn every_shipped_explicit_loop_is_capped_and_valid() {
    let modules = explicit_beams_shipped_wgsl();
    assert!(!modules.is_empty());
    for (name, _, wgsl) in &modules {
        // The Slang caps (not the post-process: no loop guard of its own), in every
        // module with a loop that is not constant (a few wide passes have none).
        let (loops, _, _, literal) = census(wgsl);
        assert!(
            loops == literal || slang_caps(wgsl),
            "{name}: no Slang loop caps"
        );
        assert!(!wgsl.contains("loopGuard") && !wgsl.contains("ebLoopHit ="), "{name}: post-processed");
        // The flag reaches gAcc where the entry point ends (ebLoopCapFlush).
        if slang_caps(wgsl) {
            assert!(wgsl.contains("ebLoopCapFlush_0();"), "{name}: the entry point never flushes the flag");
        }
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(wgsl)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
    // The frozen baseline, reference, test-only and ablation builds (read at run
    // time): those of the current kernel carry its Slang caps; the frozen kernels'
    // are capped by the post-process, every loop.
    // (The build's modules, and the test-only ones compiled on demand.)
    let dirs = [engine_webgpu::explicit_beams_wgsl_dir(), engine_webgpu::test_kernels_dir().to_path_buf()];
    let (mut slang, mut post) = (0, 0);
    let mut kinds = (0, 0, 0);
    for entry in dirs.iter().flat_map(|d| std::fs::read_dir(d).unwrap()) {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.starts_with("explicit_") || !name.ends_with(".wgsl") {
            continue;
        }
        let wgsl = std::fs::read_to_string(&path).unwrap();
        if slang_caps(&wgsl) {
            assert!(!wgsl.contains("loopGuard"), "{name}: capped twice");
            slang += 1;
            continue;
        }
        if name.starts_with("explicit_ref_")
            && std::env::var("RUBBLE_EB_REF_UNCAPPED").is_ok_and(|v| v == "1")
        {
            continue;
        }
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
        post += 1;
    }
    assert!(slang > 0 && post > 0, "{slang} Slang-capped, {post} post-processed module files");
    eprintln!(
        "{} shipped modules checked; {slang} more module files with them; {post} frozen module files \
         post-processed ({} loops with a pass counter, {} with a clamped bound, {} with a literal bound)",
        modules.len(),
        kinds.0,
        kinds.1,
        kinds.2
    );
}

/// One dispatch of `entry` in `wgsl` (bindings 0: gAcc, 4 words; 1: out, from
/// `init`): (out, gAcc) afterwards.
fn run_module(wgsl: &str, entry: &str, init: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
    let Ok((_, device, queue)) = default_device() else {
        return None;
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("loop-cap"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("loop-cap"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    use wgpu::util::DeviceExt;
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let acc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0i32; 4]),
        usage,
    });
    let out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(init),
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
        let size = b.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, size);
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
    Some((read(&out), read(&acc)))
}

/// The id of the test kernel's loop marked `loop-cap-test: <tag>` (its
/// `EB_END(ID, ..)` / `EB_PASS(ID, ..)`).
fn id_of(tag: &str) -> u32 {
    let src = include_str!("../../../slang/kernels/explicit_loop_cap_test.slang");
    let line = src
        .lines()
        .find(|l| l.contains(&format!("loop-cap-test: {tag}")))
        .unwrap_or_else(|| panic!("no loop marked {tag}"));
    let at = line.find("EB_END(").or_else(|| line.find("EB_PASS(")).expect("a capped loop");
    let args = &line[at..];
    let open = args.find('(').unwrap() + 1;
    let id = &args[open..open + args[open..].find(',').unwrap()];
    id.trim().parse().unwrap_or_else(|_| panic!("loop {tag}: no id in {line}"))
}

#[test]
fn a_runaway_loop_stops_at_the_slang_cap_and_raises_the_flag() {
    // explicit_loop_cap_test.slang: the kernel's own caps (the same macros as
    // explicit_beams.slang), compiled by build.rs; engine-metal and engine-cuda
    // run the same kernel.
    let path = format!("{}/loop_cap_test.wgsl", engine_webgpu::explicit_beams_wgsl_dir().display());
    let wgsl = std::fs::read_to_string(path).unwrap();
    let cap = EXPLICIT_LOOP_CAP;
    let tag = EXPLICIT_LOOP_CAP_TAG as u32;
    let (pass, end, step) = (id_of("pass"), id_of("end"), id_of("step"));
    let run = |init: [u32; 8]| run_module(&wgsl, "explicit_loop_cap_test", &init);
    let Some((o, a)) = run([0, 1, 0, 5, 12, 4, 0, 0]) else {
        eprintln!("no GPU: skipped");
        return;
    };
    // Ordinary bounds: every loop runs as written and no flag.
    assert_eq!((o[0], o[2], o[3], o[6]), (0, 3, 5, 3));
    assert_eq!(a, vec![0; 4]);
    // The loop whose exit never comes: stopped after the cap's passes.
    let (o, a) = run([0, 0, 0, 5, 12, 4, 0, 0]).unwrap();
    assert_eq!(o[0], cap, "passes of the runaway loop");
    assert_eq!((o[2], o[3], o[6]), (3, 5, 3));
    assert_eq!(a, vec![0, 0, 0, tag | pass]);
    // The counted loop to a huge bound: its end clamped to the cap.
    let (o, a) = run([0, 1, 0, u32::MAX, 12, 4, 0, 0]).unwrap();
    assert_eq!((o[0], o[3]), (0, cap), "passes of the runaway counted loop");
    assert_eq!(a, vec![0, 0, 0, tag | end]);
    // A zero step: no pass, flagged.
    let (o, a) = run([0, 1, 0, 5, 12, 0, 0, 0]).unwrap();
    assert_eq!((o[3], o[6]), (5, 0));
    assert_eq!(a, vec![0, 0, 0, tag | step]);
    // All three: the flag holds the highest id (atomic max), the error names it.
    let (o, a) = run([0, 0, 0, u32::MAX, 12, 0, 0, 0]).unwrap();
    assert_eq!((o[0], o[3], o[6]), (cap, cap, 0));
    let top = pass.max(end).max(step);
    assert_eq!(a[3], tag | top);
    let e = explicit_loop_cap_hit(a[3] as i32).unwrap_err();
    assert!(e.contains(&format!("id {top}")), "{e}");
}

#[test]
fn a_runaway_loop_stops_at_the_post_process_cap_and_raises_the_flag() {
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
    assert!(
        capped.contains("let ebEnd2"),
        "the counted loop takes a clamped bound"
    );
    if run_module(&capped, "main", &[0, 1, 0, 0]).is_none() {
        eprintln!("no GPU: skipped");
        return;
    }
    let run = |init: [u32; 4]| run_module(&capped, "main", &init).unwrap();
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

/// The pass budget (explicit_loop_cap.slang `EB_LOOP_BUDGET`): a frame whose loops need
/// more passes than the budget stops at it, fast, and the status reports it; with the
/// host's budget the same frame runs as written. A budget of 8 passes stands in for a
/// runaway (a frame of a sound fleet needs far more).
#[test]
fn the_pass_budget_ends_a_frame_and_reports_it() {
    use engine_softbody::explicit_beams::ExplicitBeams;
    use engine_softbody::procedural::{car_truck_bare, CarParams};
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let car = ExplicitBeams::from_truck(&text, glam::Vec3::new(0.0, 0.5, 0.0), glam::Vec3::ZERO, Default::default())
        .unwrap();
    let fleet = vec![car];
    let mut gpu = engine_webgpu::ExplicitBeamsGpu::new(&device, &queue, &fleet);
    // The per-vehicle frame (a large car takes the wide passes, a dispatch per phase).
    gpu.set_wide(false);
    gpu.frame(34);
    gpu.wait();
    assert!(gpu.kernel_status().is_ok(), "the host's budget: no stop");
    // SAFETY: this test's own process state; the planner reads it at each frame.
    unsafe { std::env::set_var("RUBBLE_LOOP_BUDGET", "8") };
    let t = std::time::Instant::now();
    gpu.frame(34);
    gpu.wait();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    unsafe { std::env::remove_var("RUBBLE_LOOP_BUDGET") };
    let e = gpu.kernel_status().expect_err("a frame past its budget reports the stop");
    assert!(e.contains("cap"), "{e}");
    assert!(ms < 2000.0, "the stopped frame took {ms} ms");
}
