// The WGSL loop guard, one source for the crate (`lib.rs`) and its build script
// (`build.rs`, which caps the explicit-beams modules as it writes them): both
// `include!` this file.

/// Hard cap on every WGSL loop. A non-terminating kernel on Apple Silicon can stall the
/// window server and take the whole machine down, so no loop may run unbounded.
#[allow(dead_code)]
pub const WGSL_LOOP_CAP: u32 = 1 << 16;

/// The explicit-beams loop cap (`engine_exec::explicit_plan::EXPLICIT_LOOP_CAP`,
/// `EB_LOOP_CAP` in `slang/kernels/explicit_beams.slang`): higher than
/// [`WGSL_LOOP_CAP`] since a few of the kernel's loops run on one lane over a whole
/// vehicle's beams (tens of thousands for a large vehicle). The shipped kernel caps
/// its loops in its Slang source (every target); [`cap_explicit_loops`] caps the
/// frozen test modules' (`explicit_beams_baseline.slang`, `explicit_beams_ref.slang`,
/// WGSL only) to the same count: a loop that reaches it sets the sticky flag in the
/// last word of `gAcc` (`engine_exec::explicit_plan::explicit_loop_cap_hit`) and
/// stops, and the host reports the error.
#[allow(dead_code)]
pub const EXPLICIT_LOOP_CAP: u32 = 1 << 20;

/// Guard every `loop {` of `src`: after `cap` passes the loop runs `on_cap(k)` (k:
/// the loop's index in the module) and breaks.
#[allow(dead_code)]
fn cap_loops_with(src: &str, cap: u32, on_cap: &dyn Fn(usize) -> String) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len() + 8192);
    let mut last = 0usize;
    let mut i = 0usize;
    let mut k = 0usize;
    while let Some(rel) = src[i..].find("loop") {
        let at = i + rel;
        let after = at + 4;
        let word = (at == 0 || !(b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_'))
            && (after >= b.len() || !(b[after].is_ascii_alphanumeric() || b[after] == b'_'));
        let mut j = after;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if word && j < b.len() && b[j] == b'{' {
            out.push_str(&src[last..at]);
            let hit = on_cap(k);
            out.push_str(&format!(
                "var loopGuard{k}: u32 = 0u;\nloop {{\n loopGuard{k} += 1u; if (loopGuard{k} > {cap}u) {{ {hit}break; }}\n"
            ));
            k += 1;
            last = j + 1;
            i = j + 1;
        } else {
            i = after;
        }
    }
    out.push_str(&src[last..]);
    out
}

/// Every loop of `src` capped at [`WGSL_LOOP_CAP`] (a capped loop just stops).
#[allow(dead_code)]
fn cap_loop_iterations(src: &str) -> String {
    cap_loops_with(src, WGSL_LOOP_CAP, &|_| String::new())
}

/// An explicit-beams module's loops capped at [`EXPLICIT_LOOP_CAP`] passes. A loop
/// stopped by its cap records its index + 1 just past the loop (a private
/// variable), and the entry point writes the highest to `gAcc`'s last word where
/// it ends (atomicMax: sticky until the host clears it). A module with a loop
/// must bind `gAcc`. Three kinds of loop:
/// * a counted loop with a literal bound (`if(i < u32(16)) {} else { break; }`
///   first, `i = i + u32(c);` once at the loop's top level, no `continue`, the
///   bound at most the cap): finite as written, left alone;
/// * a counted loop with a loop-invariant bound E (identifiers, fields, literals
///   and arithmetic: no memory read, no call, nothing the loop assigns): E is
///   clamped once before the loop to at most the cap's passes from the counter's
///   start (and below the counter's wrap), no per-pass work;
/// * any other loop: a pass counter (`loopGuard<k>`) that stops it at the cap.
///
/// Measured on the Viper's frame (M3 Max): a counter in every loop and an atomic
/// flag write after each cost about 5 %; the clamps and one flag write per
/// invocation are what is left (see the report in the commit).
#[allow(dead_code)]
#[doc(hidden)]
pub fn cap_explicit_loops(src: &str) -> Result<String, String> {
    let b = src.as_bytes();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    // (position, removed bytes, inserted text), applied back to front.
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut i = 0usize;
    let mut k = 0usize;
    while let Some(rel) = src[i..].find("loop") {
        let at = i + rel;
        let after = at + 4;
        let word = (at == 0 || !is_ident(b[at - 1])) && (after >= b.len() || !is_ident(b[after]));
        let mut j = after;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if !(word && j < b.len() && b[j] == b'{') {
            i = after;
            continue;
        }
        let open = j;
        let close = matching_close(b, open).ok_or_else(|| format!("loop {k}: no closing brace"))?;
        let body = &src[open + 1..close];
        let flag = format!("ebLoopHit = max(ebLoopHit, {}i);", k + 1);
        match counted_loop(body) {
            Some(c)
                if matches!(c.step, Step::Literal(_))
                    && c.literal.is_some_and(|n| n <= u64::from(EXPLICIT_LOOP_CAP)) => {}
            Some(c) if c.op == "<" => {
                let x = &c.var;
                let e = &c.bound;
                let cap = EXPLICIT_LOOP_CAP;
                let max = u32::MAX;
                let mut pre = String::new();
                // ebEnd ≤ X₀ + span with span ≤ step × cap (or ≤ X₀ when X₀ + span
                // wraps: no pass), and ≤ MAX − step + 1 so X never wraps.
                let (span, wrap) = match &c.step {
                    Step::Literal(n) => {
                        let span = (*n).min(u64::from(max / cap)) * u64::from(cap);
                        let wrap = (*n > 1).then(|| format!("{}u", u64::from(max) - n + 1));
                        (format!("{span}u"), wrap)
                    }
                    Step::Var(v) => {
                        // At least 1: a zero step never ends (only a runaway has one).
                        pre.push_str(&format!("let ebStep{k}: u32 = max({v}, 1u);\n"));
                        for &(p, len) in &c.step_at {
                            edits.push((open + 1 + p, len, format!("ebStep{k}")));
                        }
                        (
                            format!("min(ebStep{k}, {}u) * {cap}u", max / cap),
                            Some(format!("{max}u - ebStep{k} + 1u")),
                        )
                    }
                };
                let end = format!("min({e}, max({x} + {span}, {x}))");
                let end = match wrap {
                    Some(w) => format!("min({end}, {w})"),
                    None => end,
                };
                pre.push_str(&format!("let ebEnd{k}: u32 = {end};\n"));
                edits.push((at, 0, pre));
                edits.push((open + 1 + c.cond_at, c.cond_len, format!("{x} < ebEnd{k}")));
                edits.push((
                    close + 1,
                    0,
                    format!("\nif (ebEnd{k} < ({e}) && {x} >= ebEnd{k}) {{ {flag} }}\n"),
                ));
            }
            _ => {
                edits.push((at, 0, format!("var loopGuard{k}: u32 = 0u;\n")));
                edits.push((
                    open + 1,
                    0,
                    format!("\n loopGuard{k} += 1u; if (loopGuard{k} > {EXPLICIT_LOOP_CAP}u) {{ break; }}\n"),
                ));
                edits.push((
                    close + 1,
                    0,
                    format!("\nif (loopGuard{k} > {EXPLICIT_LOOP_CAP}u) {{ {flag} }}\n"),
                ));
            }
        }
        k += 1;
        i = open + 1;
    }
    if k > 0 && !src.contains("gAcc_0 :") {
        return Err("a loop but no gAcc binding for the loop-cap flag".into());
    }
    // Back to front; at one position, the later-pushed edit first is wrong for
    // inserts that must keep their order, so sort stably by position only.
    if k == 0 {
        return Ok(src.to_string());
    }
    // The entry points write the hit loop's number (if any) to gAcc's last word
    // where they end: before each `return;` and at the closing brace.
    let flush =
        "if (ebLoopHit != 0i) { atomicMax(&gAcc_0[arrayLength(&gAcc_0) - 1u], ebLoopHit); }";
    let mut from = 0usize;
    while let Some(rel) = src[from..].find("@compute") {
        let at = from + rel;
        let f = at + src[at..].find("fn ").ok_or("an entry point without fn")?;
        let open = f + src[f..].find('{').ok_or("an entry point without a body")?;
        let close = matching_close(b, open).ok_or("an entry point without a closing brace")?;
        let mut r = open;
        while let Some(rr) = src[r..close].find("return;") {
            let p = r + rr;
            if p > 0 && is_ident(b[p - 1]) {
                r = p + 7;
                continue;
            }
            edits.push((p, 0, format!("{flush} ")));
            r = p + 7;
        }
        if !src[..close].trim_end().ends_with("return;") {
            edits.push((close, 0, format!("{flush}\n")));
        }
        from = close;
    }
    edits.push((0, 0, "var<private> ebLoopHit: i32 = 0i;\n".to_string()));
    edits.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = src.to_string();
    for (at, len, text) in edits {
        out.replace_range(at..at + len, &text);
    }
    Ok(out)
}

/// The `}` closing the `{` at `open`.
#[allow(dead_code)]
fn matching_close(b: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// A counted loop's shape (see [`cap_explicit_loops`]).
#[allow(dead_code)]
struct Counted {
    var: String,
    op: &'static str,
    bound: String,
    /// The bound as a literal (`u32(N)`), if it is one.
    literal: Option<u64>,
    /// The step: a literal (≥ 1) or a loop-invariant identifier.
    step: Step,
    /// The condition's place in the body.
    cond_at: usize,
    cond_len: usize,
    /// The increments' steps' places in the body (to make a variable step ≥ 1).
    step_at: Vec<(usize, usize)>,
}

#[allow(dead_code)]
#[derive(PartialEq, Debug)]
enum Step {
    Literal(u64),
    Var(String),
}

/// `body` (a loop's text between its braces) as a counted loop: first
/// `if(X < E)` or `if(X <= E)` with an empty then-block and `else { break; }`;
/// X assigned only by one `X = X + u32(C);` (C ≥ 1) at the body's top level; no
/// `continue`; E loop-invariant (identifiers, fields, `u32(N)` literals,
/// `+ - *` and parentheses, none of its identifiers assigned in the body).
#[allow(dead_code)]
fn counted_loop(body: &str) -> Option<Counted> {
    let t = body.trim_start();
    let lead = body.len() - t.len();
    let rest = t.strip_prefix("if(")?;
    // The condition up to its matching parenthesis.
    let mut depth = 1i32;
    let mut end = None;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end?;
    let cond = &rest[..end];
    let tail: String = rest[end + 1..].split_whitespace().collect();
    if !tail.starts_with("{}else{break;}") {
        return None;
    }
    let (var, op, bound) = if let Some((x, e)) = cond.split_once(" <= ") {
        (x, "<=", e)
    } else if let Some((x, e)) = cond.split_once(" < ") {
        (x, "<", e)
    } else {
        return None;
    };
    let var = var.trim();
    let bound = bound.trim();
    if var.is_empty() || !var.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
        return None;
    }
    // The bound: only identifiers, fields, u32(N), + - * and parentheses.
    if !bound
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || b"_.()+-* ".contains(&c))
    {
        return None;
    }
    let idents = idents_of(bound);
    for id in &idents {
        if id == "u32" {
            continue;
        }
        // A call (an identifier followed by '(') other than u32(N).
        if bound.contains(&format!("{id}(")) {
            return None;
        }
        if id.as_bytes()[0].is_ascii_digit() {
            continue;
        }
        if assigns(body, id) {
            return None;
        }
    }
    if bound.contains("u32(") && !u32_args_literal(bound) {
        return None;
    }
    let literal = bound
        .strip_prefix("u32(")
        .and_then(|r| r.strip_suffix(')'))
        .and_then(|n| n.parse::<u64>().ok());
    if body.contains(&format!("&{var}")) {
        return None;
    }
    // The nested loops' ranges (their continues are their own).
    let bb = body.as_bytes();
    let mut nested: Vec<(usize, usize)> = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = body[from..].find("loop") {
        let at = from + rel;
        let after = at + 4;
        let word = (at == 0 || !(bb[at - 1].is_ascii_alphanumeric() || bb[at - 1] == b'_'))
            && (after >= bb.len() || !(bb[after].is_ascii_alphanumeric() || bb[after] == b'_'));
        let mut j = after;
        while j < bb.len() && bb[j].is_ascii_whitespace() {
            j += 1;
        }
        if word && j < bb.len() && bb[j] == b'{' {
            let close = matching_close(bb, j)?;
            nested.push((at, close));
            from = close;
        } else {
            from = after;
        }
    }
    let in_nested = |i: usize| nested.iter().any(|&(a, b)| i >= a && i <= b);
    let depth_at = |p: usize| {
        let mut d = 0i32;
        for &c in &bb[..p] {
            match c {
                b'{' => d += 1,
                b'}' => d -= 1,
                _ => {}
            }
        }
        d
    };
    let ident_at = |p: usize| {
        p > 0 && (bb[p - 1].is_ascii_alphanumeric() || bb[p - 1] == b'_' || bb[p - 1] == b'.')
    };
    // The step expression `S` of `X + S` ending at a ';': a literal u32(C ≥ 1) or a
    // loop-invariant identifier; and its place in the body.
    let step_of = |at: usize| -> Option<(Step, usize, usize)> {
        let r = &body[at..];
        let semi = r.find(';')?;
        let raw = &r[..semi];
        let expr = raw.trim();
        let pos = at + (raw.len() - raw.trim_start().len());
        let st = if let Some(n) = expr.strip_prefix("u32(").and_then(|x| x.strip_suffix(')')) {
            let c: u64 = n.parse().ok()?;
            if c == 0 {
                return None;
            }
            Step::Literal(c)
        } else if !expr.is_empty() && expr.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            if expr.as_bytes()[0].is_ascii_digit() || expr == var || assigns(body, expr) {
                return None;
            }
            Step::Var(expr.to_string())
        } else {
            return None;
        };
        Some((st, pos, expr.len()))
    };
    // The increments: `X = X + S;`, or `X = T;` after `var T : u32 = X + S;`.
    // (start, end past ';', S, S's place)
    let mut incs: Vec<(usize, usize, Step, (usize, usize))> = Vec::new();
    let direct = format!("{var} = {var} + ");
    let mut from = 0usize;
    while let Some(rel) = body[from..].find(&direct) {
        let at = from + rel;
        from = at + direct.len();
        if ident_at(at) {
            continue;
        }
        if in_nested(at) {
            return None;
        }
        let (st, p, l) = step_of(at + direct.len())?;
        let end = at + body[at..].find(';')? + 1;
        incs.push((at, end, st, (p, l)));
    }
    let via = format!("{var} = ");
    let mut from = 0usize;
    while let Some(rel) = body[from..].find(&via) {
        let at = from + rel;
        from = at + via.len();
        if ident_at(at) || body[at..].starts_with(&direct) {
            continue;
        }
        let r = &body[at + via.len()..];
        let semi = r.find(';')?;
        let t_name = r[..semi].trim();
        if t_name.is_empty()
            || !t_name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            continue; // another assignment: counted below
        }
        let def = format!("var {t_name} : u32 = {var} + ");
        let d = body.find(&def)?;
        if d > at
            || ident_at(d)
            || in_nested(at)
            || in_nested(d)
            || count_assigns(body, t_name) != 0
        {
            return None;
        }
        if depth_at(d) != depth_at(at) {
            return None;
        }
        let (st, p, l) = step_of(d + def.len())?;
        incs.push((at, at + via.len() + semi + 1, st, (p, l)));
    }
    incs.sort_by_key(|x| x.0);
    // X assigned by its increments only, with one S.
    if incs.is_empty() || count_assigns(body, var) != incs.len() {
        return None;
    }
    if incs.iter().any(|x| x.2 != incs[0].2) {
        return None;
    }
    // One increment at the top level; any other is before it and is followed by
    // `continue;` (so one increment a pass).
    let top: Vec<usize> = (0..incs.len())
        .filter(|&i| depth_at(incs[i].0) == 0)
        .collect();
    if top.len() != 1 {
        return None;
    }
    let main = top[0];
    for (i, x) in incs.iter().enumerate() {
        if i != main && (i > main || !body[x.1..].trim_start().starts_with("continue;")) {
            return None;
        }
    }
    // Every continue (outside nested loops) right after an increment.
    let mut from = 0usize;
    while let Some(rel) = body[from..].find("continue;") {
        let at = from + rel;
        from = at + 9;
        if in_nested(at) {
            continue;
        }
        let before = body[..at].trim_end().len();
        if !incs.iter().any(|x| body[..x.1].trim_end().len() == before) {
            return None;
        }
    }
    let step_at: Vec<(usize, usize)> = incs.iter().map(|x| x.3).collect();
    let step = incs.swap_remove(0).2;
    let cond_at = lead + 3;
    Some(Counted {
        var: var.to_string(),
        op,
        bound: bound.to_string(),
        literal,
        step,
        cond_at,
        cond_len: cond.len(),
        step_at,
    })
}

#[allow(dead_code)]
fn idents_of(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut after_dot = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            cur.push(c);
        } else {
            if !cur.is_empty() && !after_dot {
                out.push(std::mem::take(&mut cur));
            }
            cur.clear();
            after_dot = c == '.';
        }
    }
    if !cur.is_empty() && !after_dot {
        out.push(cur);
    }
    out
}

/// Every `u32(...)` in `s` holds a decimal literal.
#[allow(dead_code)]
fn u32_args_literal(s: &str) -> bool {
    let mut rest = s;
    while let Some(p) = rest.find("u32(") {
        let r = &rest[p + 4..];
        let Some(close) = r.find(')') else {
            return false;
        };
        if !r[..close].bytes().all(|c| c.is_ascii_digit()) || close == 0 {
            return false;
        }
        rest = &r[close..];
    }
    true
}

/// Whether `body` assigns `id` (or one of its fields): `id =`, `id.f =`, `id[..] =`,
/// `id +=` and the like, or takes its address.
#[allow(dead_code)]
fn assigns(body: &str, id: &str) -> bool {
    count_assigns(body, id) > 0 || body.contains(&format!("&{id}"))
}

#[allow(dead_code)]
fn count_assigns(body: &str, id: &str) -> usize {
    let b = body.as_bytes();
    let mut n = 0;
    let mut from = 0;
    while let Some(rel) = body[from..].find(id) {
        let at = from + rel;
        from = at + id.len();
        if at > 0 && (b[at - 1].is_ascii_alphanumeric() || b[at - 1] == b'_' || b[at - 1] == b'.') {
            continue;
        }
        // Skip a field or index path after the identifier.
        let mut j = at + id.len();
        if j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
            continue;
        }
        loop {
            if j < b.len() && b[j] == b'.' {
                j += 1;
                while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                    j += 1;
                }
            } else if j < b.len() && b[j] == b'[' {
                let mut d = 0;
                while j < b.len() {
                    if b[j] == b'[' {
                        d += 1;
                    } else if b[j] == b']' {
                        d -= 1;
                        if d == 0 {
                            j += 1;
                            break;
                        }
                    }
                    j += 1;
                }
            } else {
                break;
            }
        }
        while j < b.len() && b[j] == b' ' {
            j += 1;
        }
        let r = &body[j.min(body.len())..];
        let op_assign = [
            "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=",
        ]
        .iter()
        .any(|op| r.starts_with(op) && !r.starts_with("=="));
        if op_assign || r.starts_with("++") || r.starts_with("--") {
            n += 1;
        }
    }
    n
}
