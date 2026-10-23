//! BeamNG JBeam loader (design: "BeamNG JBeam importer"). Written from the JBeam
//! documentation and BeamNG's MIT-licensed tooling as references, not from game
//! code. Pipeline:
//!
//! 1. relaxed JSON ([`parse_relaxed`]): comments, commas as whitespace, unquoted
//!    keys, `=` for `:`, `FLT_MAX` / `MINUS_FLT_MAX`;
//! 2. parts and slots ([`JbeamDb::assemble`]): start at the `main` part (or the
//!    `.pc` config's `mainPartName`), take each slot's configured or default part,
//!    recurse (50 levels), carrying slot `nodeOffset` / `nodeMove`;
//! 3. variables: `$var` from the `.pc` `vars` or the variable default (clamped),
//!    `$=expr` evaluated with Lua semantics ([`eval_expr`]); a failed expression
//!    drops the key so the default applies;
//! 4. table expansion: header row, rows with trailing option dictionaries, and
//!    modifier dictionaries applying to every following row of the section;
//! 5. lowering into [`SoftVehicleDesc`]: JBeam's frame (+X left, +Y rear, +Z up)
//!    becomes Rubble's (+X left, +Y up, +Z forward).
//!
//! Supported sections: nodes, beams (NORMAL, SUPPORT, BOUNDED, HYDRO; ANISOTROPIC
//! and PRESSURED as normal beams), triangles, quads, pressureWheels, refNodes,
//! powertrain (only to find driven wheels). LBEAM, torsionbars, rails and
//! slidenodes are counted and skipped with a warning.

use crate::desc::*;
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Relaxed JSON
// ---------------------------------------------------------------------------

struct Reader<'a> {
    s: &'a [u8],
    i: usize,
}

impl Reader<'_> {
    fn err(&self, msg: &str) -> String {
        let line = self.s[..self.i.min(self.s.len())]
            .iter()
            .filter(|&&c| c == b'\n')
            .count()
            + 1;
        format!("line {line}: {msg}")
    }

    /// Skip whitespace, commas and comments.
    fn skip(&mut self) {
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if c.is_ascii_whitespace() || c == b',' {
                self.i += 1;
            } else if c == b'/' && self.s.get(self.i + 1) == Some(&b'/') {
                while self.i < self.s.len() && self.s[self.i] != b'\n' {
                    self.i += 1;
                }
            } else if c == b'/' && self.s.get(self.i + 1) == Some(&b'*') {
                self.i += 2;
                while self.i + 1 < self.s.len()
                    && !(self.s[self.i] == b'*' && self.s[self.i + 1] == b'/')
                {
                    self.i += 1;
                }
                self.i += 2;
            } else {
                break;
            }
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        self.skip();
        let Some(&c) = self.s.get(self.i) else {
            return Err(self.err("unexpected end"));
        };
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::String(self.string()?)),
            b'-' | b'+' | b'.' | b'0'..=b'9' => self.number(),
            _ => {
                let w = self.word();
                match w.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "null" | "nil" => Ok(Value::Null),
                    "FLT_MAX" => Ok(num(f64::MAX)),
                    "MINUS_FLT_MAX" => Ok(num(-f64::MAX)),
                    "" => Err(self.err(&format!("unexpected `{}`", c as char))),
                    _ => Ok(Value::String(w)),
                }
            }
        }
    }

    fn word(&mut self) -> String {
        let start = self.i;
        while self.i < self.s.len() {
            let c = self.s[self.i];
            if c.is_ascii_alphanumeric()
                || c == b'_'
                || c == b'$'
                || c == b'.'
                || c == b'|'
                || c == b'-'
            {
                self.i += 1;
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.s[start..self.i]).into_owned()
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = Vec::new();
        while self.i < self.s.len() {
            let c = self.s[self.i];
            self.i += 1;
            match c {
                b'"' => return Ok(String::from_utf8_lossy(&out).into_owned()),
                b'\\' => {
                    let e = self.s.get(self.i).copied().unwrap_or(b'\\');
                    self.i += 1;
                    out.push(match e {
                        b'n' => b'\n',
                        b't' => b'\t',
                        b'r' => b'\r',
                        other => other,
                    });
                }
                _ => out.push(c),
            }
        }
        Err(self.err("unterminated string"))
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if matches!(self.s[self.i], b'-' | b'+') {
            self.i += 1;
        }
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_digit() || matches!(self.s[self.i], b'.' | b'e' | b'E'))
        {
            if matches!(self.s[self.i], b'e' | b'E')
                && matches!(self.s.get(self.i + 1), Some(b'-' | b'+'))
            {
                self.i += 1;
            }
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        // `-FLT_MAX` style words after a sign.
        if t == "-" || t == "+" {
            let w = self.word();
            if w == "FLT_MAX" {
                return Ok(num(if t == "-" { -f64::MAX } else { f64::MAX }));
            }
            return Err(self.err(&format!("bad number `{t}{w}`")));
        }
        t.trim_start_matches('+')
            .parse::<f64>()
            .map(num)
            .map_err(|_| self.err(&format!("bad number `{t}`")))
    }

    fn array(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            self.skip();
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated array")),
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(out));
                }
                _ => out.push(self.value()?),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.i += 1;
        let mut out = Map::new();
        loop {
            self.skip();
            let key = match self.s.get(self.i) {
                None => return Err(self.err("unterminated object")),
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(out));
                }
                Some(b'"') => self.string()?,
                _ => {
                    let w = self.word();
                    if w.is_empty() {
                        return Err(self.err("expected a key"));
                    }
                    w
                }
            };
            self.skip();
            match self.s.get(self.i) {
                Some(b':') | Some(b'=') => self.i += 1,
                _ => return Err(self.err(&format!("expected `:` after `{key}`"))),
            }
            let v = self.value()?;
            out.insert(key, v);
        }
    }
}

fn num(v: f64) -> Value {
    Number::from_f64(v)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// Parse JBeam's relaxed JSON.
pub fn parse_relaxed(src: &str) -> Result<Value, String> {
    let mut r = Reader {
        s: src.as_bytes(),
        i: 0,
    };
    let v = r.value()?;
    r.skip();
    // Stray closing brackets after the top-level value (hand-edited files) are
    // tolerated, as BeamNG's loader does; anything else is an error.
    while r.i < r.s.len() && matches!(r.s[r.i], b'}' | b']') {
        r.i += 1;
        r.skip();
    }
    if r.i < r.s.len() {
        return Err(r.err("trailing content"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// `$=` expressions (Lua semantics: only false and nil are falsy)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Lua {
    Num(f64),
    Bool(bool),
    Nil,
}

impl Lua {
    fn truthy(self) -> bool {
        !matches!(self, Lua::Bool(false) | Lua::Nil)
    }
    fn num(self) -> Option<f64> {
        match self {
            Lua::Num(v) => Some(v),
            _ => None,
        }
    }
}

struct Expr<'a> {
    t: Vec<String>,
    i: usize,
    vars: &'a HashMap<String, f64>,
}

fn tokenize(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c.is_ascii_digit()
            || (c == b'.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let st = i;
            while i < b.len()
                && (b[i].is_ascii_digit()
                    || b[i] == b'.'
                    || b[i] == b'e'
                    || b[i] == b'E'
                    || ((b[i] == b'-' || b[i] == b'+') && matches!(b[i - 1], b'e' | b'E')))
            {
                i += 1;
            }
            out.push(s[st..i].to_string());
        } else if c.is_ascii_alphabetic() || c == b'_' || c == b'$' {
            let st = i;
            while i < b.len()
                && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'$' || b[i] == b'.')
            {
                i += 1;
            }
            out.push(s[st..i].to_string());
        } else {
            let two = s.get(i..i + 2).unwrap_or("");
            if matches!(two, "<=" | ">=" | "==" | "~=" | "!=") {
                out.push(two.to_string());
                i += 2;
            } else {
                out.push((c as char).to_string());
                i += 1;
            }
        }
    }
    out
}

impl Expr<'_> {
    fn peek(&self) -> &str {
        self.t.get(self.i).map(String::as_str).unwrap_or("")
    }
    fn next(&mut self) -> String {
        let t = self.peek().to_string();
        self.i += 1;
        t
    }
    fn expect(&mut self, s: &str) -> Result<(), String> {
        if self.next() == s {
            Ok(())
        } else {
            Err(format!("expected `{s}`"))
        }
    }
    fn or(&mut self) -> Result<Lua, String> {
        let mut a = self.and()?;
        while self.peek() == "or" {
            self.i += 1;
            let b = self.and()?;
            a = if a.truthy() { a } else { b };
        }
        Ok(a)
    }
    fn and(&mut self) -> Result<Lua, String> {
        let mut a = self.cmp()?;
        while self.peek() == "and" {
            self.i += 1;
            let b = self.cmp()?;
            a = if a.truthy() { b } else { a };
        }
        Ok(a)
    }
    fn cmp(&mut self) -> Result<Lua, String> {
        let a = self.add()?;
        let op = self.peek().to_string();
        if matches!(op.as_str(), "<" | ">" | "<=" | ">=" | "==" | "~=" | "!=") {
            self.i += 1;
            let b = self.add()?;
            let (x, y) = (
                a.num().ok_or("comparison of non-numbers")?,
                b.num().ok_or("comparison of non-numbers")?,
            );
            return Ok(Lua::Bool(match op.as_str() {
                "<" => x < y,
                ">" => x > y,
                "<=" => x <= y,
                ">=" => x >= y,
                "==" => x == y,
                _ => x != y,
            }));
        }
        Ok(a)
    }
    fn add(&mut self) -> Result<Lua, String> {
        let mut a = self.mul()?;
        while matches!(self.peek(), "+" | "-") {
            let op = self.next();
            let b = self.mul()?;
            let (x, y) = (
                a.num().ok_or("arithmetic on non-number")?,
                b.num().ok_or("arithmetic on non-number")?,
            );
            a = Lua::Num(if op == "+" { x + y } else { x - y });
        }
        Ok(a)
    }
    fn mul(&mut self) -> Result<Lua, String> {
        let mut a = self.unary()?;
        while matches!(self.peek(), "*" | "/" | "%") {
            let op = self.next();
            let b = self.unary()?;
            let (x, y) = (
                a.num().ok_or("arithmetic on non-number")?,
                b.num().ok_or("arithmetic on non-number")?,
            );
            a = Lua::Num(match op.as_str() {
                "*" => x * y,
                "/" => x / y,
                _ => x - (x / y).floor() * y,
            });
        }
        Ok(a)
    }
    fn unary(&mut self) -> Result<Lua, String> {
        match self.peek() {
            "-" => {
                self.i += 1;
                Ok(Lua::Num(
                    -self.unary()?.num().ok_or("negation of non-number")?,
                ))
            }
            "not" => {
                self.i += 1;
                Ok(Lua::Bool(!self.unary()?.truthy()))
            }
            _ => self.pow(),
        }
    }
    fn pow(&mut self) -> Result<Lua, String> {
        let a = self.atom()?;
        if self.peek() == "^" {
            self.i += 1;
            let b = self.unary()?;
            return Ok(Lua::Num(
                a.num()
                    .ok_or("power of non-number")?
                    .powf(b.num().ok_or("power of non-number")?),
            ));
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<Lua, String> {
        let t = self.next();
        if t == "(" {
            let v = self.or()?;
            self.expect(")")?;
            return Ok(v);
        }
        if let Ok(v) = t.parse::<f64>() {
            return Ok(Lua::Num(v));
        }
        match t.as_str() {
            "true" => return Ok(Lua::Bool(true)),
            "false" => return Ok(Lua::Bool(false)),
            "nil" => return Ok(Lua::Nil),
            "math.pi" | "pi" => return Ok(Lua::Num(std::f64::consts::PI)),
            "math.huge" => return Ok(Lua::Num(f64::INFINITY)),
            _ => {}
        }
        if let Some(name) = t.strip_prefix('$') {
            return self
                .vars
                .get(name)
                .map(|&v| Lua::Num(v))
                .ok_or_else(|| format!("unknown variable ${name}"));
        }
        // Function call.
        if self.peek() == "(" {
            self.i += 1;
            let mut args = Vec::new();
            while self.peek() != ")" {
                args.push(self.or()?.num().ok_or("non-number argument")?);
                if self.peek() == "," {
                    self.i += 1;
                }
                if self.i > self.t.len() {
                    return Err("unterminated call".into());
                }
            }
            self.expect(")")?;
            let f = t.trim_start_matches("math.");
            let a = |k: usize| {
                args.get(k)
                    .copied()
                    .ok_or_else(|| format!("{f}: missing argument"))
            };
            return Ok(Lua::Num(match f {
                "min" => args.iter().copied().fold(f64::INFINITY, f64::min),
                "max" => args.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "abs" => a(0)?.abs(),
                "floor" => a(0)?.floor(),
                "ceil" => a(0)?.ceil(),
                "sqrt" => a(0)?.sqrt(),
                "sin" => a(0)?.sin(),
                "cos" => a(0)?.cos(),
                "tan" => a(0)?.tan(),
                "rad" => a(0)?.to_radians(),
                "deg" => a(0)?.to_degrees(),
                "exp" => a(0)?.exp(),
                "log" => a(0)?.ln(),
                "clamp" => a(0)?.clamp(a(1)?, a(2)?),
                "sign" => a(0)?.signum(),
                _ => return Err(format!("unknown function {t}")),
            }));
        }
        Err(format!("unexpected `{t}`"))
    }
}

/// Evaluate a `$=` expression with Lua semantics against variable values.
pub fn eval_expr(src: &str, vars: &HashMap<String, f64>) -> Result<f64, String> {
    let mut e = Expr {
        t: tokenize(src),
        i: 0,
        vars,
    };
    let v = e.or()?;
    if e.i < e.t.len() {
        return Err(format!("trailing `{}`", e.peek()));
    }
    match v {
        Lua::Num(x) => Ok(x),
        Lua::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        Lua::Nil => Err("nil".into()),
    }
}

// ---------------------------------------------------------------------------
// Parts, slots, variables
// ---------------------------------------------------------------------------

/// A `.pc` vehicle configuration (`{"format":2, "mainPartName", "parts", "vars"}`).
#[derive(Clone, Debug, Default)]
pub struct PcConfig {
    pub main_part: Option<String>,
    pub parts: HashMap<String, String>,
    pub vars: HashMap<String, f64>,
}

impl PcConfig {
    pub fn parse(src: &str) -> Result<Self, String> {
        let v = parse_relaxed(src)?;
        let mut c = PcConfig {
            main_part: v
                .get("mainPartName")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..Default::default()
        };
        if let Some(p) = v.get("parts").and_then(Value::as_object) {
            for (k, x) in p {
                c.parts
                    .insert(k.clone(), x.as_str().unwrap_or("").to_string());
            }
        }
        if let Some(p) = v.get("vars").and_then(Value::as_object) {
            for (k, x) in p {
                if let Some(f) = x.as_f64() {
                    c.vars.insert(k.trim_start_matches('$').to_string(), f);
                }
            }
        }
        Ok(c)
    }
}

/// Every part of a vehicle's `.jbeam` files, by name.
#[derive(Clone, Debug, Default)]
pub struct JbeamDb {
    pub parts: BTreeMap<String, Value>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
struct Placed {
    name: String,
    offset: [Value; 3],
    moved: [Value; 3],
}

impl JbeamDb {
    /// Add the parts of one `.jbeam` file.
    pub fn add_file(&mut self, name: &str, src: &str) -> Result<(), String> {
        let v = parse_relaxed(src).map_err(|e| format!("{name}: {e}"))?;
        let Value::Object(map) = v else {
            return Err(format!("{name}: top level is not an object"));
        };
        for (k, part) in map {
            if part.is_object() {
                self.parts.insert(k, part);
            }
        }
        Ok(())
    }

    fn slot_type(&self, part: &str) -> Option<&str> {
        self.parts.get(part)?.get("slotType")?.as_str()
    }

    /// Resolve the part tree from the main part through each slot's configured
    /// (or default) part, depth first. Returns the parts in assembly order with
    /// their accumulated slot node offsets.
    fn assemble(&mut self, config: &PcConfig) -> Result<Vec<Placed>, String> {
        let main = match &config.main_part {
            Some(m) => m.clone(),
            None => self
                .parts
                .iter()
                .find(|(_, p)| p.get("slotType").and_then(Value::as_str) == Some("main"))
                .map(|(k, _)| k.clone())
                .ok_or("no part with slotType \"main\"")?,
        };
        if !self.parts.contains_key(&main) {
            return Err(format!("main part `{main}` not found"));
        }
        let zero = || [num(0.0), num(0.0), num(0.0)];
        let mut out = Vec::new();
        let mut stack = vec![(
            Placed {
                name: main,
                offset: zero(),
                moved: zero(),
            },
            0u32,
        )];
        let mut seen = HashSet::new();
        while let Some((placed, depth)) = stack.pop() {
            if depth > 50 || !seen.insert(placed.name.clone()) {
                continue;
            }
            let part = self.parts[&placed.name].clone();
            let mut children = Vec::new();
            for (key, v2) in [("slots", false), ("slots2", true)] {
                let Some(rows) = part.get(key).and_then(Value::as_array) else {
                    continue;
                };
                let header: Vec<String> = rows
                    .first()
                    .and_then(Value::as_array)
                    .map(|h| {
                        h.iter()
                            .map(|x| x.as_str().unwrap_or("").to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                for row in rows.iter().skip(1) {
                    let Some(r) = row.as_array() else { continue };
                    let field = |n: &str| header.iter().position(|h| h == n).and_then(|i| r.get(i));
                    let slot_name = if v2 { field("name") } else { field("type") }
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let default = field("default")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let opts = r
                        .last()
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    let chosen = config.parts.get(&slot_name).cloned().unwrap_or(default);
                    if chosen.is_empty() {
                        continue;
                    }
                    if !self.parts.contains_key(&chosen) {
                        self.warnings
                            .push(format!("slot `{slot_name}`: part `{chosen}` not found"));
                        continue;
                    }
                    let ty = self.slot_type(&chosen).unwrap_or("");
                    let allowed = if v2 {
                        field("allowTypes")
                            .and_then(Value::as_array)
                            .is_none_or(|a| a.iter().any(|t| t.as_str() == Some(ty)))
                    } else {
                        ty == slot_name
                    };
                    if !allowed {
                        self.warnings.push(format!(
                            "slot `{slot_name}`: part `{chosen}` has slotType `{ty}`"
                        ));
                        continue;
                    }
                    let mut offset = placed.offset.clone();
                    let mut moved = placed.moved.clone();
                    for (k, target) in [("nodeOffset", &mut offset), ("nodeMove", &mut moved)] {
                        if let Some(o) = opts.get(k).and_then(Value::as_object) {
                            for (i, axis) in ["x", "y", "z"].iter().enumerate() {
                                if let Some(v) = o.get(*axis) {
                                    target[i] = v.clone();
                                }
                            }
                        }
                    }
                    children.push(Placed {
                        name: chosen,
                        offset,
                        moved,
                    });
                }
            }
            out.push(placed);
            for c in children.into_iter().rev() {
                stack.push((c, depth + 1));
            }
        }
        Ok(out)
    }
}

/// Variables: defaults from every assembled part's `variables` table, then the
/// configuration's values, clamped to their ranges.
fn collect_vars(db: &JbeamDb, parts: &[Placed], config: &PcConfig) -> HashMap<String, f64> {
    let mut vars = HashMap::new();
    let mut ranges = HashMap::new();
    for p in parts {
        let Some(rows) = db.parts[&p.name].get("variables").and_then(Value::as_array) else {
            continue;
        };
        let header: Vec<String> = rows
            .first()
            .and_then(Value::as_array)
            .map(|h| {
                h.iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();
        for row in rows.iter().skip(1) {
            let Some(r) = row.as_array() else { continue };
            let f = |n: &str| header.iter().position(|h| h == n).and_then(|i| r.get(i));
            let Some(name) = f("name").and_then(Value::as_str) else {
                continue;
            };
            let name = name.trim_start_matches('$').to_string();
            let default = f("default").and_then(Value::as_f64).unwrap_or(0.0);
            vars.entry(name.clone()).or_insert(default);
            if let (Some(lo), Some(hi)) = (
                f("min").and_then(Value::as_f64),
                f("max").and_then(Value::as_f64),
            ) {
                ranges.insert(name, (lo.min(hi), lo.max(hi)));
            }
        }
    }
    for (k, &v) in &config.vars {
        let v = ranges.get(k).map_or(v, |&(lo, hi)| v.clamp(lo, hi));
        vars.insert(k.clone(), v);
    }
    vars
}

/// Resolve `$var` and `$=expr` values; `None` when an expression fails.
fn resolve(v: &Value, vars: &HashMap<String, f64>) -> Option<Value> {
    match v {
        Value::String(s) if s.starts_with("$=") => eval_expr(&s[2..], vars).ok().map(num),
        Value::String(s) if s.starts_with('$') && s.len() > 1 && !s.contains(' ') => {
            match vars.get(&s[1..]) {
                Some(&x) => Some(num(x)),
                None => Some(v.clone()),
            }
        }
        Value::String(s) if s == "FLT_MAX" => Some(num(f64::MAX)),
        Value::String(s) if s == "MINUS_FLT_MAX" => Some(num(-f64::MAX)),
        _ => Some(v.clone()),
    }
}

fn as_f32(v: &Value) -> Option<f32> {
    v.as_f64().map(|x| {
        if x >= f32::MAX as f64 {
            f32::INFINITY
        } else if x <= -(f32::MAX as f64) {
            f32::NEG_INFINITY
        } else {
            x as f32
        }
    })
}

/// Rows of one section across the assembled parts: each row's fields by header
/// name, merged over the modifiers in effect (and part-level defaults).
fn expand(
    db: &JbeamDb,
    parts: &[Placed],
    section: &str,
    vars: &HashMap<String, f64>,
) -> Vec<(usize, Map<String, Value>)> {
    let mut out = Vec::new();
    for (pi, p) in parts.iter().enumerate() {
        let part = &db.parts[&p.name];
        let Some(rows) = part.get(section).and_then(Value::as_array) else {
            continue;
        };
        let Some(header) = rows.first().and_then(Value::as_array) else {
            continue;
        };
        let header: Vec<String> = header
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();
        let mut modifiers: Map<String, Value> = Map::new();
        for row in rows.iter().skip(1) {
            match row {
                Value::Object(m) => {
                    for (k, v) in m {
                        match resolve(v, vars) {
                            Some(r) => {
                                modifiers.insert(k.clone(), r);
                            }
                            None => {
                                modifiers.remove(k);
                            }
                        }
                    }
                }
                Value::Array(r) => {
                    let mut fields = modifiers.clone();
                    for (i, v) in r.iter().enumerate() {
                        if let Value::Object(opts) = v {
                            for (k, o) in opts {
                                if let Some(x) = resolve(o, vars) {
                                    fields.insert(k.clone(), x);
                                }
                            }
                        } else if let Some(h) = header.get(i) {
                            if let Some(x) = resolve(v, vars) {
                                fields.insert(h.trim_end_matches(':').to_string(), x);
                            }
                        }
                    }
                    out.push((pi, fields));
                }
                _ => {}
            }
        }
    }
    out
}

/// JBeam frame (+X left, +Y rear, +Z up) → Rubble (+X left, +Y up, +Z forward).
pub fn to_rubble(p: [f32; 3]) -> [f32; 3] {
    [p[0], p[2], -p[1]]
}

/// Load a vehicle from its `.jbeam` files (`(file name, source)`) and an
/// optional `.pc` configuration.
pub fn load_jbeam(files: &[(&str, &str)], config: Option<&str>) -> Result<SoftVehicleDesc, String> {
    let mut db = JbeamDb::default();
    for (name, src) in files {
        if let Err(e) = db.add_file(name, src) {
            db.warnings.push(e);
        }
    }
    let config = match config {
        Some(c) => PcConfig::parse(c)?,
        None => PcConfig::default(),
    };
    let parts = db.assemble(&config)?;
    let vars = collect_vars(&db, &parts, &config);
    let mut desc = SoftVehicleDesc {
        name: parts[0].name.clone(),
        collision_range: 0.03,
        ..Default::default()
    };
    desc.warnings.append(&mut db.warnings);
    if let Some(n) = db.parts[&parts[0].name]
        .get("information")
        .and_then(|i| i.get("name"))
        .and_then(Value::as_str)
    {
        desc.name = n.to_string();
    }

    // Nodes. Parts: each JBeam part's own nodes (named as the part), and each
    // node `group` (the sets BeamNG flexbodies bind to).
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut part_nodes: Vec<(String, Vec<u32>)> = Vec::new();
    let mut group_nodes: Vec<(String, Vec<u32>)> = Vec::new();
    for (pi, f) in expand(&db, &parts, "nodes", &vars) {
        let Some(id) = f.get("id").and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_f64().map(|x| x.to_string()))
        }) else {
            continue;
        };
        let coord = |k: &str| f.get(k).and_then(as_f32).unwrap_or(0.0);
        let place = &parts[pi];
        let off = |v: &Value| resolve(v, &vars).and_then(|x| as_f32(&x)).unwrap_or(0.0);
        let x = coord("posX");
        let pos = [
            x + x.signum() * off(&place.offset[0]) + off(&place.moved[0]),
            coord("posY") + off(&place.offset[1]) + off(&place.moved[1]),
            coord("posZ") + off(&place.offset[2]) + off(&place.moved[2]),
        ];
        let node = SoftNode {
            name: id.clone(),
            position: to_rubble(pos),
            mass: f.get("nodeWeight").and_then(as_f32).unwrap_or(25.0),
            collision: f.get("collision").and_then(Value::as_bool).unwrap_or(true),
            fixed: f.get("fixed").and_then(Value::as_bool).unwrap_or(false),
            contacter: f
                .get("selfCollision")
                .and_then(Value::as_bool)
                .unwrap_or(false)
                || f.get("collision").and_then(Value::as_bool).unwrap_or(true),
            friction: f.get("frictionCoef").and_then(as_f32).unwrap_or(1.0),
            frame: false,
        };
        let index = if let Some(&i) = ids.get(&id) {
            // A later part redefines the node (JBeam: last definition wins).
            desc.nodes[i as usize] = node;
            i
        } else {
            ids.insert(id, desc.nodes.len() as u32);
            desc.nodes.push(node);
            desc.nodes.len() as u32 - 1
        };
        let mut add = |list: &mut Vec<(String, Vec<u32>)>, name: &str| match list
            .iter_mut()
            .find(|(n, _)| n == name)
        {
            Some((_, v)) => {
                if !v.contains(&index) {
                    v.push(index);
                }
            }
            None => list.push((name.to_string(), vec![index])),
        };
        add(&mut part_nodes, &place.name);
        match f.get("group") {
            Some(Value::String(g)) if !g.is_empty() => add(&mut group_nodes, g),
            Some(Value::Array(gs)) => {
                for g in gs
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|g| !g.is_empty())
                {
                    add(&mut group_nodes, g);
                }
            }
            _ => {}
        }
    }
    for (name, nodes) in part_nodes.into_iter().chain(group_nodes) {
        if !desc.parts.iter().any(|p| p.name == name) {
            desc.parts.push(crate::desc::SoftPart { name, nodes });
        }
    }
    let node = |f: &Map<String, Value>, k: &str| {
        f.get(k)
            .and_then(Value::as_str)
            .and_then(|n| ids.get(n).copied())
    };

    // Beams.
    let mut groups: HashMap<String, u32> = HashMap::new();
    let mut skipped: BTreeMap<String, u32> = BTreeMap::new();
    let mut missing = 0u32;
    for (_, f) in expand(&db, &parts, "beams", &vars) {
        let (Some(a), Some(b)) = (node(&f, "id1"), node(&f, "id2")) else {
            if !f.get("optional").and_then(Value::as_bool).unwrap_or(false) {
                missing += 1;
            }
            continue;
        };
        let ty = f
            .get("beamType")
            .and_then(Value::as_str)
            .unwrap_or("|NORMAL")
            .trim_start_matches('|')
            .to_ascii_uppercase();
        let kind = match ty.as_str() {
            "NORMAL" | "ANISOTROPIC" | "PRESSURED" | "" => BeamKind::Normal,
            "SUPPORT" => BeamKind::Support,
            "BOUNDED" => BeamKind::Bounded,
            "HYDRO" => BeamKind::Hydro,
            "BROKEN" => continue,
            other => {
                *skipped.entry(format!("beam type {other}")).or_default() += 1;
                continue;
            }
        };
        if matches!(ty.as_str(), "ANISOTROPIC" | "PRESSURED") {
            *skipped
                .entry(format!("beam type {ty} (simulated as NORMAL)"))
                .or_default() += 1;
        }
        let g = |k: &str, d: f32| f.get(k).and_then(as_f32).unwrap_or(d);
        let group = match f.get("breakGroup") {
            Some(Value::String(s)) if !s.is_empty() => {
                let n = groups.len() as u32 + 1;
                *groups.entry(s.clone()).or_insert(n)
            }
            Some(Value::Array(a)) => a
                .first()
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map_or(0, |s| {
                    let n = groups.len() as u32 + 1;
                    *groups.entry(s.to_string()).or_insert(n)
                }),
            _ => 0,
        };
        let mut beam = SoftBeam {
            a,
            b,
            kind,
            spring: g("beamSpring", 4_300_000.0),
            damp: g("beamDamp", 580.0),
            deform: g("beamDeform", 220_000.0),
            strength: g("beamStrength", f32::INFINITY),
            plastic_coef: 0.0,
            short_bound: 0.0,
            long_bound: 0.0,
            precompression: g("beamPrecompression", 1.0).max(0.01),
            hydro_factor: 0.0,
            break_group: group,
            visible: true,
        };
        match kind {
            BeamKind::Support => {
                // Breaks past (1 + longBound)·L₀; default bound 1.
                beam.long_bound = 1.0 + g("beamLongBound", 1.0);
                beam.deform = f32::INFINITY;
            }
            BeamKind::Bounded => {
                beam.short_bound = g("beamShortBound", 1.0).min(0.99);
                beam.long_bound = g("beamLongBound", 1.0);
                // Limit spring (the slot bounded beams use for it).
                beam.deform = g("beamLimitSpring", beam.spring);
            }
            BeamKind::Hydro => {
                // Full input moves the rest length to its in/out limit.
                let factor = g("inputFactor", 1.0);
                let out_limit = g("outLimit", 1.2);
                let in_limit = g("inLimit", 0.8);
                let span = if factor >= 0.0 {
                    1.0 - in_limit
                } else {
                    out_limit - 1.0
                };
                beam.hydro_factor = span * factor.signum();
                beam.deform = f32::INFINITY;
                if f.get("inputSource").and_then(Value::as_str) != Some("steering") {
                    beam.hydro_factor = 0.0;
                }
            }
            BeamKind::Normal => {
                // deformLimit / deformLimitExpansion: plastic rest-length ratios of L₀.
                if let Some(l) = f
                    .get("deformLimit")
                    .and_then(as_f32)
                    .filter(|l| l.is_finite() && *l > 0.0 && *l < 1.0)
                {
                    beam.short_bound = l;
                }
                if let Some(l) = f
                    .get("deformLimitExpansion")
                    .and_then(as_f32)
                    .filter(|l| l.is_finite() && *l > 1.0)
                {
                    beam.long_bound = l;
                }
            }
            BeamKind::Rope => {}
        }
        desc.beams.push(beam);
    }
    if missing > 0 {
        desc.warnings.push(format!(
            "{missing} beams reference nodes not in this configuration"
        ));
    }

    // Triangles and quads.
    for (_, f) in expand(&db, &parts, "triangles", &vars) {
        let (Some(a), Some(b), Some(c)) = (node(&f, "id1"), node(&f, "id2"), node(&f, "id3"))
        else {
            continue;
        };
        let collision = f
            .get("triangleType")
            .and_then(Value::as_str)
            .is_none_or(|t| !t.contains("NONCOLLIDABLE"));
        desc.triangles.push(SoftTriangle {
            nodes: [a, b, c],
            collision,
        });
    }
    for (_, f) in expand(&db, &parts, "quads", &vars) {
        let (Some(a), Some(b), Some(c), Some(d)) = (
            node(&f, "id1"),
            node(&f, "id2"),
            node(&f, "id3"),
            node(&f, "id4"),
        ) else {
            continue;
        };
        let collision = f
            .get("triangleType")
            .and_then(Value::as_str)
            .is_none_or(|t| !t.contains("NONCOLLIDABLE"));
        desc.triangles.push(SoftTriangle {
            nodes: [a, b, c],
            collision,
        });
        desc.triangles.push(SoftTriangle {
            nodes: [a, c, d],
            collision,
        });
    }

    // Powertrain: wheels connected (through any chain) to an engine or motor are driven.
    let mut device_input: HashMap<String, (String, String)> = HashMap::new();
    let mut wheel_shafts: Vec<(String, String)> = Vec::new();
    for (_, f) in expand(&db, &parts, "powertrain", &vars) {
        let (Some(ty), Some(name)) = (
            f.get("type").and_then(Value::as_str),
            f.get("name").and_then(Value::as_str),
        ) else {
            continue;
        };
        let input = f
            .get("inputName")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        device_input.insert(name.to_string(), (ty.to_string(), input));
        if let Some(w) = f.get("connectedWheel").and_then(Value::as_str) {
            wheel_shafts.push((w.to_string(), name.to_string()));
        }
    }
    let driven_wheel = |wheel: &str| {
        wheel_shafts
            .iter()
            .filter(|(w, _)| w == wheel)
            .any(|(_, shaft)| {
                let mut cur = shaft.clone();
                for _ in 0..32 {
                    let Some((ty, input)) = device_input.get(&cur) else {
                        return false;
                    };
                    if ty.contains("Engine") || ty.contains("Motor") {
                        return true;
                    }
                    cur = input.clone();
                }
                false
            })
    };

    // Wheels.
    for (_, f) in expand(&db, &parts, "pressureWheels", &vars) {
        let (Some(n1), Some(n2)) = (node(&f, "node1"), node(&f, "node2")) else {
            continue;
        };
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let radius = f
            .get("radius")
            .and_then(as_f32)
            .or_else(|| f.get("hubRadius").and_then(as_f32))
            .unwrap_or(0.3);
        let width = f
            .get("tireWidth")
            .and_then(as_f32)
            .or_else(|| f.get("hubWidth").and_then(as_f32))
            .unwrap_or(0.2);
        let brake = f.get("brakeTorque").and_then(as_f32).unwrap_or(0.0);
        desc.wheels.push(SoftWheel {
            node1: n1,
            node2: n2,
            arm: node(&f, "nodeArm"),
            radius: radius.clamp(0.05, 3.0),
            width: width.clamp(0.02, 2.0),
            mass: 20.0,
            braked: brake > 0.0,
            propulsion: i32::from(driven_wheel(&name)),
            steerable: false,
            max_steer: 0.0,
        });
    }
    // Steering: a wheel whose arm node is driven by a steering hydro.
    for w in &mut desc.wheels {
        if let Some(arm) = w.arm {
            if desc.beams.iter().any(|b| {
                b.kind == BeamKind::Hydro && b.hydro_factor != 0.0 && (b.a == arm || b.b == arm)
            }) {
                w.steerable = true;
                w.max_steer = 35f32.to_radians();
            }
        }
    }

    // Reference frame.
    if let Some((_, f)) = expand(&db, &parts, "refNodes", &vars).into_iter().last() {
        if let (Some(center), Some(back), Some(left)) =
            (node(&f, "ref"), node(&f, "back"), node(&f, "left"))
        {
            desc.refs = Some(RefNodes { center, back, left });
        }
    }

    for section in [
        "torsionbars",
        "rails",
        "slidenodes",
        "flexbodies",
        "props",
        "thrusters",
    ] {
        let n = expand(&db, &parts, section, &vars).len();
        if n > 0 {
            skipped.insert(format!("section {section}"), n as u32);
        }
    }
    for (what, n) in skipped {
        desc.warnings.push(format!("{what}: {n} skipped"));
    }
    desc.repair(0.1);
    Ok(desc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_json_accepts_jbeam_syntax() {
        let v = parse_relaxed(
            r#"
            // comment
            { part = { "nodes": [ ["id" "posX"] /* block */ ["a", -1.5e1,] ], strength: FLT_MAX, neg: -FLT_MAX, k: [1 2 3] dup: 1 dup: 2 } }
            "#,
        )
        .unwrap();
        let p = &v["part"];
        assert_eq!(p["nodes"][1][1].as_f64(), Some(-15.0));
        assert_eq!(p["strength"].as_f64(), Some(f64::MAX));
        assert_eq!(p["neg"].as_f64(), Some(-f64::MAX));
        assert_eq!(p["k"].as_array().unwrap().len(), 3);
        assert_eq!(p["dup"].as_f64(), Some(2.0), "last duplicate wins");
        assert!(
            parse_relaxed("{ a: 1 } }").is_ok(),
            "stray closing brace tolerated"
        );
        assert!(parse_relaxed("{ a: 1 } x").is_err());
    }

    #[test]
    fn expressions_follow_lua() {
        let vars: HashMap<String, f64> =
            [("weight".to_string(), 2.0), ("k".to_string(), 0.0)].into();
        assert_eq!(eval_expr("$weight*1001000", &vars), Ok(2_002_000.0));
        assert_eq!(eval_expr("-$weight + 2^3 * (1 + 1)", &vars), Ok(14.0));
        assert_eq!(eval_expr("$k > 0 and 5 or 7", &vars), Ok(7.0));
        // Lua: 0 is truthy.
        assert_eq!(eval_expr("$k and 3 or 4", &vars), Ok(3.0));
        assert_eq!(
            eval_expr("math.max(1, $weight, 3) + min(4, 5)", &vars),
            Ok(7.0)
        );
        assert!(eval_expr("$missing + 1", &vars).is_err());
    }

    /// An original two-part vehicle (Rubble content) exercising slots, a `.pc`
    /// override, variables, expressions, modifiers, optional beams and wheels.
    const CAR: &str = r#"
    {
      "toycar": {
        "information": {"name": "Toy JBeam Car"},
        "slotType": "main",
        "slots": [["type", "default", "description"],
                  ["toycar_body", "toycar_body_soft", "Body"],
                  ["toycar_wheels", "toycar_wheels", "Wheels", {"nodeOffset": {"x": 0.1, "y": 0, "z": 0}}]],
        "variables": [["name", "type", "unit", "category", "default", "min", "max"],
                      ["$stiff", "range", "", "Body", 2, 1, 4]],
        "refNodes": [["ref:", "back:", "left:", "up:"], ["f1", "b1", "f2", "t1"]],
      },
      "toycar_body_soft": {
        "slotType": "toycar_body",
        "nodes": [["id", "posX", "posY", "posZ"],
                  {"nodeWeight": 10},
                  ["f1", -0.5, -1, 0.3], ["f2", 0.5, -1, 0.3],
                  ["b1", -0.5, 1, 0.3], ["b2", 0.5, 1, 0.3],
                  {"nodeWeight": "$=$stiff * 3"},
                  ["t1", 0, 0, 1.0, {"collision": false}]],
        "beams": [["id1:", "id2:"],
                  {"beamSpring": "$=$stiff*1000000", "beamDamp": 300, "beamDeform": 50000, "beamStrength": "FLT_MAX"},
                  ["f1", "f2"], ["b1", "b2"], ["f1", "b1"], ["f2", "b2"], ["f1", "b2"],
                  {"breakGroup": "roof"},
                  ["t1", "f1"], ["t1", "f2"], ["t1", "b1"], ["t1", "b2"],
                  {"breakGroup": ""},
                  {"beamType": "|SUPPORT", "beamLongBound": 0.5},
                  ["f2", "b1"],
                  {"beamType": "|NORMAL", "optional": true},
                  ["f1", "nosuchnode"]],
        "triangles": [["id1:", "id2:", "id3:"], ["f1", "f2", "t1"], ["b1", "t1", "b2", {"triangleType": "NONCOLLIDABLE"}]],
        "quads": [["id1:", "id2:", "id3:", "id4:"], ["f1", "b1", "b2", "f2"]],
      },
      "toycar_body_stiff": {
        "slotType": "toycar_body",
        "nodes": [["id", "posX", "posY", "posZ"], ["f1", -0.5, -1, 0.3], ["f2", 0.5, -1, 0.3], ["b1", -0.5, 1, 0.3]],
      },
      "toycar_wheels": {
        "slotType": "toycar_wheels",
        "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 5}, ["ax1", -0.6, -0.8, 0.3], ["ax2", 0.6, -0.8, 0.3]],
        "beams": [["id1:", "id2:"], ["ax1", "ax2"], ["ax1", "f1"], ["ax2", "f2"]],
        "pressureWheels": [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"],
                           {"radius": 0.31, "tireWidth": 0.2, "brakeTorque": 900},
                           ["FL", "wheel_FL", "tire_FL", "ax1", "ax2", 9999, "f1", 1]],
        "powertrain": [["type", "name", "inputName", "inputIndex"],
                       ["combustionEngine", "mainEngine", "dummy", 0],
                       ["shaft", "axleFL", "mainEngine", 1, {"connectedWheel": "FL"}]],
      }
    }
    "#;

    #[test]
    fn loads_parts_slots_variables_and_tables() {
        let pc = r#"{"format": 2, "mainPartName": "toycar", "parts": {"toycar_body": "toycar_body_soft"}, "vars": {"$stiff": 3}}"#;
        let d = load_jbeam(&[("toycar.jbeam", CAR)], Some(pc)).unwrap();
        assert_eq!(d.name, "Toy JBeam Car");
        assert_eq!(
            d.nodes.len(),
            7,
            "{:?}",
            d.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
        );
        // Frame: JBeam (x, y, z) → Rubble (x, z, −y).
        let f1 = d.nodes.iter().find(|n| n.name == "f1").unwrap();
        assert_eq!(f1.position, [-0.5, 0.3, 1.0]);
        // Slot nodeOffset: x pushed outward by sign(x)·0.1.
        let ax1 = d.nodes.iter().find(|n| n.name == "ax1").unwrap();
        assert!((ax1.position[0] + 0.7).abs() < 1e-6, "{:?}", ax1.position);
        // Variable from the .pc (3), expressions in modifiers.
        let t1 = d.nodes.iter().find(|n| n.name == "t1").unwrap();
        assert_eq!(t1.mass, 9.0);
        assert!(!t1.collision);
        assert_eq!(d.beams[0].spring, 3.0e6);
        assert!(d.beams[0].strength.is_infinite());
        // Break group, support, optional beam to a missing node dropped silently.
        assert!(d.beams.iter().filter(|b| b.break_group != 0).count() == 4);
        assert_eq!(
            d.beams
                .iter()
                .filter(|b| b.kind == BeamKind::Support)
                .count(),
            1
        );
        assert!(
            !d.warnings.iter().any(|w| w.contains("reference nodes")),
            "{:?}",
            d.warnings
        );
        // Triangles: 2 + a quad as 2; one non-collidable.
        assert_eq!(d.triangles.len(), 4);
        assert_eq!(d.triangles.iter().filter(|t| !t.collision).count(), 1);
        // Wheel: driven through the powertrain chain, braked, radius from the modifier.
        assert_eq!(d.wheels.len(), 1);
        assert_eq!(d.wheels[0].propulsion, 1);
        assert!(d.wheels[0].braked);
        assert!((d.wheels[0].radius - 0.31).abs() < 1e-6);
        assert!(d.refs.is_some());
    }

    #[test]
    fn descriptors_round_trip_through_json() {
        let d = load_jbeam(&[("toycar.jbeam", CAR)], None).unwrap();
        assert!(d.beams.iter().any(|b| b.strength.is_infinite()));
        let back: SoftVehicleDesc = serde_json::from_str(&d.to_json()).unwrap();
        assert_eq!(back.beams.len(), d.beams.len());
        assert!(back
            .beams
            .iter()
            .zip(&d.beams)
            .all(|(a, b)| a.strength >= f32::MAX && b.strength >= f32::MAX
                || a.strength == b.strength));
    }

    #[test]
    fn config_selects_another_part_and_clamps_vars() {
        let pc = r#"{"mainPartName": "toycar", "parts": {"toycar_body": "toycar_body_stiff", "toycar_wheels": ""}, "vars": {"$stiff": 99}}"#;
        let d = load_jbeam(&[("toycar.jbeam", CAR)], Some(pc)).unwrap();
        assert_eq!(d.nodes.len(), 3, "stiff body, no wheels");
        assert!(d.wheels.is_empty());
    }
}
