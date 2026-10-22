//! Decoded state `/2` (P2's full-state models): the model's frame-end state of the island's
//! bonds and chunks as `fs_*_out` columns by field path, P1's naming
//! (`full_state_calls::flatten`: `a.b` struct fields, `a[i]` list elements, `a=variant`
//! one-hot enum variants, NaN for `None`), written back into the island's entities through
//! their serialized form. No per-field code: a predicted column is the field it names.
//!
//! Composition of the island's frame-end state (`apply_full`):
//! - the `/1` decoded state (chunk u/th/v/w, the bodies' fit, impactors, breaks) applied to the
//!   frame-start world, as harness A applies a `/1` model (`World::apply_decoded_with`);
//! - every predicted bond and chunk column overwritten from the model's rows;
//! - the fields not predicted (bond force/elastic/measures/stored/damped, the per-chunk world
//!   arrays, the contact history) from `base`: the native step's island (P2's choice: those
//!   are recomputed or diagnostic) or the frame start. A native base reads the oracle's own
//!   step for those fields; the frame-start base does not (measure both).
//!
//! Rules (P2): a one-hot group takes the variant whose value is at least 0.5, else `None`;
//! `alive`, and whether the bond is connected, must agree with the `/1` decode (an error
//! otherwise, never a silent choice); a bond the decode keeps connected has its damage
//! clamped below `SURVIVOR_DAMAGE_MAX` (one already disconnected keeps its damage);
//! infinities are values (an at-rest ductility).

use std::collections::{BTreeMap, HashMap};

use rmpv::Value;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use stress_ref::world::{IslandState, SURVIVOR_DAMAGE_MAX};

/// One entity kind's predicted columns: `rows[k * columns.len() + j]` is entity `keys[k]`'s
/// value of `columns[j]`. Keys: `keys::bond_key(structure, source)` for bonds,
/// `keys::chunk_key(structure, chunk)` for chunks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FullOut {
    pub kind: String,
    pub keys: Vec<i64>,
    pub columns: Vec<String>,
    pub rows: Vec<f64>,
}

/// Where the fields a model does not predict come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    Native,
    Start,
}

pub fn to_value<T: Serialize>(x: &T) -> Result<Value, String> {
    let b = rmp_serde::to_vec_named(x).map_err(|e| e.to_string())?;
    rmpv::decode::read_value(&mut &b[..]).map_err(|e| e.to_string())
}

pub fn from_value<T: DeserializeOwned>(v: &Value) -> Result<T, String> {
    let mut b = Vec::new();
    rmpv::encode::write_value(&mut b, v).map_err(|e| e.to_string())?;
    rmp_serde::from_slice(&b).map_err(|e| e.to_string())
}

/// A path's steps: map keys and list indices (`reaction[0].x` -> reaction, 0, x).
enum Step {
    Key(String),
    Index(usize),
}

fn steps(path: &str) -> Result<Vec<Step>, String> {
    let mut out = Vec::new();
    for seg in path.split('.') {
        let (name, rest) = seg.split_once('[').map_or((seg, ""), |(n, r)| (n, r));
        if !name.is_empty() {
            out.push(Step::Key(name.to_string()));
        }
        if !rest.is_empty() {
            for ix in format!("[{rest}").split('[').filter(|x| !x.is_empty()) {
                let i = ix.trim_end_matches(']').parse::<usize>().map_err(|_| format!("{path}: index {ix}"))?;
                out.push(Step::Index(i));
            }
        }
    }
    Ok(out)
}

fn at_mut<'a>(v: &'a mut Value, path: &str) -> Result<&'a mut Value, String> {
    let mut cur = v;
    for s in steps(path)? {
        cur = match s {
            Step::Key(k) => match cur {
                Value::Map(entries) => entries.iter_mut().find(|(x, _)| x.as_str() == Some(k.as_str())).map(|(_, y)| y).ok_or_else(|| format!("{path}: no field {k}"))?,
                _ => return Err(format!("{path}: {k} of a non-struct")),
            },
            Step::Index(i) => match cur {
                Value::Array(items) => items.get_mut(i).ok_or_else(|| format!("{path}: no element {i}"))?,
                _ => return Err(format!("{path}: [{i}] of a non-list")),
            },
        };
    }
    Ok(cur)
}

/// Set one leaf to `x`, keeping its serialized type (bool, integer, float). NaN is `None`
/// (P1's flatten writes `None` as NaN; a NaN in a field that is not an option fails to
/// deserialize, loudly).
fn set_leaf(leaf: &mut Value, x: f64, path: &str) -> Result<(), String> {
    *leaf = match leaf {
        _ if x.is_nan() => Value::Nil,
        Value::Boolean(_) => Value::Boolean(x >= 0.5),
        Value::Integer(_) => {
            if !x.is_finite() {
                return Err(format!("{path}: {x} for an integer"));
            }
            Value::from(x.round() as i64)
        }
        Value::F32(_) => Value::F32(x as f32),
        _ => Value::F64(x),
    };
    Ok(())
}

/// Write `row` (named by `columns`) into the serialized entity `v`.
pub fn write_row(v: &mut Value, columns: &[String], row: &[f64]) -> Result<(), String> {
    let mut groups: BTreeMap<&str, Vec<(&str, f64)>> = BTreeMap::new();
    for (c, &x) in columns.iter().zip(row) {
        match c.split_once('=') {
            Some((p, variant)) => groups.entry(p).or_default().push((variant, x)),
            None => set_leaf(at_mut(v, c)?, x, c)?,
        }
    }
    for (p, variants) in groups {
        let best = variants.iter().copied().filter(|&(_, x)| x >= 0.5).max_by(|a, b| a.1.total_cmp(&b.1));
        *at_mut(v, p)? = match best {
            Some((name, _)) => Value::from(name),
            None => Value::Nil,
        };
    }
    Ok(())
}

fn rows_by_key(out: &FullOut) -> Result<HashMap<i64, &[f64]>, String> {
    let w = out.columns.len();
    if out.rows.len() != out.keys.len() * w {
        return Err(format!("fs_{}_out: {} values for {} rows of {w}", out.kind, out.rows.len(), out.keys.len()));
    }
    Ok(out.keys.iter().enumerate().map(|(k, &key)| (key, &out.rows[k * w..(k + 1) * w])).collect())
}

/// The island's frame-end state from the `/1`-decoded state `decoded` (the frame start with
/// the `/1` decode applied), the model's predicted columns `outs`, and `base` for the rest.
pub fn apply_full(decoded: &IslandState, base: &IslandState, outs: &[FullOut]) -> Result<IslandState, String> {
    let mut st = decoded.clone();
    let by_kind: HashMap<&str, &FullOut> = outs.iter().map(|o| (o.kind.as_str(), o)).collect();
    // Bonds: the base's bond, the predicted columns over it; alive must be the decode's.
    let base_bonds: HashMap<(usize, usize), &stress_ref::solver::RtBond> = base.solver.bonds.iter().map(|(k, b)| (*k, b)).collect();
    let bond_rows = by_kind.get("bond").map(|o| rows_by_key(o)).transpose()?;
    for ((s, bi), b) in st.solver.bonds.iter_mut() {
        let from = base_bonds.get(&(*s, *bi)).ok_or_else(|| format!("bond ({s}, {bi}) not in the base"))?;
        let mut v = to_value(*from)?;
        if let (Some(rows), Some(o)) = (&bond_rows, by_kind.get("bond")) {
            let key = stress_ref::keys::bond_key(*s, b.source);
            let row = rows.get(&key).ok_or_else(|| format!("fs_bond_out lacks bond {key}"))?;
            write_row(&mut v, &o.columns, row)?;
        }
        let mut nb: stress_ref::solver::RtBond = from_value(&v)?;
        if o_has(&by_kind, "bond", "alive") && nb.alive != b.alive {
            return Err(format!("bond ({s}, {bi}): fs alive {} against the decode's {}", nb.alive, b.alive));
        }
        nb.alive = b.alive;
        // A bond the decode keeps connected survives: its damage stays below 1 (a bond
        // already disconnected, alive at damage 1 as the oracle leaves some, is as given).
        let connected = b.connected();
        if connected {
            nb.joint.damage = nb.joint.damage.min(SURVIVOR_DAMAGE_MAX);
        }
        if nb.connected() != connected {
            return Err(format!("bond ({s}, {bi}): the model's state is {} where the decode's is {}", if nb.connected() { "connected" } else { "disconnected" }, if connected { "connected" } else { "disconnected" }));
        }
        *b = nb;
    }
    // Chunks: the decoded chunk state with the predicted columns over it.
    if let Some(o) = by_kind.get("chunk") {
        let rows = rows_by_key(o)?;
        for ((s, c), ch) in st.solver.chunks.iter_mut() {
            let key = stress_ref::keys::chunk_key(*s, *c);
            let row = rows.get(&key).ok_or_else(|| format!("fs_chunk_out lacks chunk {key}"))?;
            let mut v = to_value(&*ch)?;
            write_row(&mut v, &o.columns, row)?;
            *ch = from_value(&v)?;
        }
    }
    // Not predicted: the per-chunk world arrays, the solver scratch and the contact history.
    st.take_unpredicted(base);
    // Cluster activity and its timer, where the base has the same body.
    let acts: HashMap<u64, (stress_ref::solver::Activity, f64)> = base.solver.clusters.iter().map(|(_, c)| (c.id, (c.activity, c.active_timer))).collect();
    for (_, c) in st.solver.clusters.iter_mut() {
        if let Some(&(a, t)) = acts.get(&c.id) {
            c.activity = a;
            c.active_timer = t;
        }
    }
    Ok(st)
}

fn o_has(by_kind: &HashMap<&str, &FullOut>, kind: &str, col: &str) -> bool {
    by_kind.get(kind).is_some_and(|o| o.columns.iter().any(|c| c == col))
}
