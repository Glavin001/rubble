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
//! `alive` is the `/1` decode's (the oracle's bookkeeping), and whether the bond is connected
//! must agree with it (an error otherwise, never a silent choice); a bond the decode keeps connected has its damage
//! clamped below `SURVIVOR_DAMAGE_MAX` (one already disconnected keeps its damage); the joint
//! law's rule (failure margin >= 1 exactly when disconnected) holds: a survivor's fail_ts and
//! fail_c stay below 1, a disconnected bond's fail_ts is at least 1;
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
    /// JSON numbers, or "inf" / "-inf" / "nan" (JSON has no infinities); null is NaN.
    #[serde(deserialize_with = "lenient_f64s")]
    pub rows: Vec<f64>,
}

fn lenient_f64s<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
    let raw: Vec<serde_json::Value> = Vec::deserialize(d)?;
    raw.iter()
        .map(|v| match v {
            serde_json::Value::Number(n) => n.as_f64().ok_or_else(|| serde::de::Error::custom(format!("{n}"))),
            serde_json::Value::Null => Ok(f64::NAN),
            serde_json::Value::String(s) => match s.as_str() {
                "inf" | "Infinity" => Ok(f64::INFINITY),
                "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
                "nan" | "NaN" => Ok(f64::NAN),
                other => Err(serde::de::Error::custom(format!("row value {other:?}"))),
            },
            other => Err(serde::de::Error::custom(format!("row value {other}"))),
        })
        .collect()
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
        // `@`-prefixed columns are not the entity's fields (P1: the world's per-chunk arrays;
        // a model's informational columns, e.g. `@break_time`): not written.
        if c.starts_with('@') {
            continue;
        }
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
    apply_full_with(decoded, base, outs, ids_from_base())
}

/// `apply_full` with the bodies' bookkeeping (id, parent, list place) from the base
/// (`ids_from_base`) or the decode's own.
pub fn apply_full_with(decoded: &IslandState, base: &IslandState, outs: &[FullOut], ids_from_base: bool) -> Result<IslandState, String> {
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
            match rows.get(&key) {
                Some(row) => write_row(&mut v, &o.columns, row)?,
                // A model may leave out bonds already disconnected (inert): the base's.
                None if !b.connected() => {}
                None => return Err(format!("fs_bond_out lacks the connected bond {key}")),
            }
        }
        let mut nb: stress_ref::solver::RtBond = from_value(&v)?;
        // `alive` is the decode's (the oracle's own bookkeeping: `break_bonds` disconnects a
        // bond at damage 1 and leaves it alive unless a split cuts it); a model's `alive`
        // column is only checked through connectivity below.
        nb.alive = b.alive;
        // A bond the decode keeps connected survives: its damage stays below 1 (a bond
        // already disconnected, alive at damage 1 as the oracle leaves some, is as given).
        let connected = b.connected();
        if connected {
            nb.joint.damage = nb.joint.damage.min(SURVIVOR_DAMAGE_MAX);
            // The joint law's rule, failure margin >= 1 <=> disconnected: a survivor's
            // predicted margins stay below 1.
            nb.fail_ts = nb.fail_ts.min(SURVIVOR_DAMAGE_MAX);
            nb.fail_c = nb.fail_c.min(SURVIVOR_DAMAGE_MAX);
        } else {
            // A disconnected bond's margin is reached (as `apply_decoded` breaks one) when the
            // model's is not.
            if nb.fail_margin() < 1.0 {
                nb.fail_ts = nb.fail_ts.max(1.0);
                nb.peak_fail_margin = nb.peak_fail_margin.max(nb.fail_margin());
            }
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
    // Impactors: the decoded impactor with the predicted columns over it (`fs_impactor_out`,
    // keyed by `keys::impactor_key`).
    if let Some(o) = by_kind.get("impactor") {
        let rows = rows_by_key(o)?;
        for (i, imp) in st.impactors.iter_mut() {
            let key = stress_ref::keys::impactor_key(*i);
            let row = rows.get(&key).ok_or_else(|| format!("fs_impactor_out lacks impactor {key}"))?;
            let mut v = to_value(&*imp)?;
            write_row(&mut v, &o.columns, row)?;
            *imp = from_value(&v)?;
        }
    }
    // Not predicted: the per-chunk world arrays, the solver scratch and the contact history.
    st.take_unpredicted(base);
    // Bodies: the fields a model does not predict and the partition does not fix (P2) from
    // the base's body with the same key (its lowest chunk key); activity and its timer too
    // unless the model predicts the bodies (`fs_cluster_out`, set by `body_records`); with
    // `STRESS_FULL_IDS=base` (default) the bookkeeping (id, parent, list place) as well, with
    // `decode` the decode's own (its breaks applied in time order number the bodies as the
    // oracle's break-by-break splits do).
    let mut from_base: Vec<&str> = vec!["explicit_window", "driven", "settled_load_norm", "window_utilization"];
    if !by_kind.contains_key("cluster") {
        from_base.extend(["activity", "active_timer"]);
    }
    let key_of = |c: &stress_ref::solver::Cluster| c.chunks.iter().map(|&k| stress_ref::keys::chunk_key(c.structure, k)).min();
    let base_by_key: HashMap<i64, (Option<usize>, &stress_ref::solver::Cluster)> = base.solver.clusters.iter().filter_map(|(h, c)| key_of(c).map(|k| (k, (*h, c)))).collect();
    for (hint, c) in st.solver.clusters.iter_mut() {
        let Some(&(bh, b)) = key_of(c).and_then(|k| base_by_key.get(&k)) else { continue };
        if ids_from_base {
            (*hint, c.id, c.parent) = (bh, b.id, b.parent);
        }
        let (mut v, bv) = (to_value(&*c)?, to_value(b)?);
        for f in &from_base {
            if let (Ok(dst), Some(src)) = (at_mut(&mut v, f), bv.as_map().and_then(|m| m.iter().find(|(k, _)| k.as_str() == Some(f)).map(|(_, x)| x.clone()))) {
                *dst = src;
            }
        }
        *c = from_value(&v)?;
    }
    st.solver.next_cluster_id = st.solver.next_cluster_id.max(base.solver.next_cluster_id);
    let mut ids = std::collections::HashSet::new();
    for (_, c) in &st.solver.clusters {
        if !ids.insert(c.id) {
            return Err(format!("decoded state /2: body id {} twice", c.id));
        }
    }
    Ok(st)
}


/// An island state's bond and chunk fields as `FullOut` columns (P1's flatten), limited to
/// `predicted` (kind -> column names; None: every column): the oracle's own `/2` outcome,
/// the ceiling of a full-state model's contract.
pub fn columns_of_state(st: &IslandState, predicted: Option<&HashMap<String, Vec<String>>>) -> Result<Vec<FullOut>, String> {
    let mut outs = Vec::new();
    let kinds: [(&str, Vec<(i64, Value)>); 4] = [
        ("bond", st.solver.bonds.iter().map(|((s, _), b)| Ok((stress_ref::keys::bond_key(*s, b.source), to_value(b)?))).collect::<Result<_, String>>()?),
        ("chunk", st.solver.chunks.iter().map(|((s, c), ch)| Ok((stress_ref::keys::chunk_key(*s, *c), to_value(ch)?))).collect::<Result<_, String>>()?),
        // bodies by their lowest chunk key (`World::body_key`)
        (
            "cluster",
            st.solver.clusters.iter().filter_map(|(_, c)| c.chunks.iter().map(|&k| stress_ref::keys::chunk_key(c.structure, k)).min().map(|k| (k, c))).map(|(k, c)| Ok((k, to_value(c)?))).collect::<Result<_, String>>()?,
        ),
        ("impactor", st.impactors.iter().map(|(i, imp)| Ok((stress_ref::keys::impactor_key(*i), to_value(imp)?))).collect::<Result<_, String>>()?),
    ];
    for (kind, ents) in kinds {
        let rows: Vec<Vec<(String, f64)>> = ents
            .iter()
            .map(|(_, v)| {
                let mut r = Vec::new();
                crate::full_state_calls::flatten(v, "", &mut r);
                r
            })
            .collect();
        let columns: Vec<String> = match predicted.and_then(|p| p.get(kind)) {
            Some(cols) => cols.clone(),
            None if predicted.is_some() => continue,
            None => rows.iter().flatten().map(|(c, _)| c.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect(),
        };
        let data = rows
            .iter()
            .flat_map(|r| {
                let m: HashMap<&str, f64> = r.iter().map(|(n, x)| (n.as_str(), *x)).collect();
                columns.iter().map(move |c| m.get(c.as_str()).copied().unwrap_or(if c.contains('=') { 0.0 } else { f64::NAN })).collect::<Vec<_>>()
            })
            .collect();
        outs.push(FullOut { kind: kind.into(), keys: ents.iter().map(|(k, _)| *k).collect(), columns, rows: data });
    }
    Ok(outs)
}

/// The bodies of `fs_cluster_out` (kind "cluster", keyed by each body's lowest chunk key) as
/// `World::set_body_records` takes them: pose.position.{x,y,z}, pose.rotation.{w,x,y,z},
/// velocity.{x,y,z}, angular_velocity.{x,y,z} required; activity=Active / activity=Settled and
/// active_timer when given.
pub fn body_records(out: &FullOut) -> Result<HashMap<i64, stress_ref::world::BodyRecord>, String> {
    use stress_ref::math::{Quat, Vec3};
    let col = |n: &str| out.columns.iter().position(|c| c == n);
    let need = |n: &str| col(n).ok_or_else(|| format!("fs_cluster_out lacks {n}"));
    let v3 = |p: &str| -> Result<[usize; 3], String> { Ok([need(&format!("{p}.x"))?, need(&format!("{p}.y"))?, need(&format!("{p}.z"))?]) };
    let (pos, vel, ang) = (v3("pose.position")?, v3("velocity")?, v3("angular_velocity")?);
    let rot = [need("pose.rotation.w")?, need("pose.rotation.x")?, need("pose.rotation.y")?, need("pose.rotation.z")?];
    let (act, set, timer) = (col("activity=Active"), col("activity=Settled"), col("active_timer"));
    let rows = rows_by_key(out)?;
    let mut recs = HashMap::new();
    for (key, r) in rows {
        let v = |ix: [usize; 3]| Vec3::new(r[ix[0]], r[ix[1]], r[ix[2]]);
        let activity = match (act.map(|i| r[i]), set.map(|i| r[i])) {
            (Some(a), Some(b)) => Some(if a >= b { stress_ref::solver::Activity::Active } else { stress_ref::solver::Activity::Settled }),
            (Some(a), None) => Some(if a >= 0.5 { stress_ref::solver::Activity::Active } else { stress_ref::solver::Activity::Settled }),
            (None, Some(b)) => Some(if b >= 0.5 { stress_ref::solver::Activity::Settled } else { stress_ref::solver::Activity::Active }),
            (None, None) => None,
        };
        recs.insert(key, stress_ref::world::BodyRecord {
            position: v(pos),
            rotation: Quat { w: r[rot[0]], x: r[rot[1]], y: r[rot[2]], z: r[rot[3]] },
            velocity: v(vel),
            angular_velocity: v(ang),
            activity,
            active_timer: timer.map(|i| r[i]).filter(|t| t.is_finite()),
        });
    }
    Ok(recs)
}

/// `STRESS_FULL_IDS=decode`: the bodies' ids, parents and list places are the decode's own
/// (its breaks applied in time order, as the oracle splits); default `base`.
pub fn ids_from_base() -> bool {
    !std::env::var("STRESS_FULL_IDS").is_ok_and(|v| v == "decode")
}

/// The chunks' predicted hidden state (u, th, v, w in the body's frame) from `fs_chunk_out`
/// when it has those columns (`World::derive_bodies` derives the bodies from them).
#[allow(clippy::type_complexity)]
pub fn hidden_states(outs: &[FullOut]) -> Result<Option<HashMap<(usize, usize), (stress_ref::math::Vec3, stress_ref::math::Vec3, stress_ref::math::Vec3, stress_ref::math::Vec3)>>, String> {
    use stress_ref::math::Vec3;
    let Some(o) = outs.iter().find(|o| o.kind == "chunk") else { return Ok(None) };
    let col = |n: &str| o.columns.iter().position(|c| c == n);
    let Some(ix) = ["u", "th", "v", "w"].iter().map(|p| Some([col(&format!("{p}.x"))?, col(&format!("{p}.y"))?, col(&format!("{p}.z"))?])).collect::<Option<Vec<_>>>() else { return Ok(None) };
    let rows = rows_by_key(o)?;
    Ok(Some(
        rows.into_iter()
            .map(|(k, r)| {
                let v = |i: usize| Vec3::new(r[ix[i][0]], r[ix[i][1]], r[ix[i][2]]);
                (((k >> 32) as usize, (k & 0xffff_ffff) as usize), (v(0), v(1), v(2), v(3)))
            })
            .collect(),
    ))
}

/// Harness A's island frame from a decoded state `/2` (the shared host-side decode path):
/// on a copy of the frame-start world `start` (peak utilisations reset), the `/1` decode `d`
/// applied with its deformation kept and its breaks in time order (the partition and the
/// bodies' numbering are the oracle's own split); the bodies from the model's table
/// (`fs_cluster_out`) when given, else derived from the chunks' world and predicted hidden
/// state (`World::derive_bodies`) when given; then the predicted columns written by field
/// path over the island (`apply_full`), the rest from `base`.
///
/// `energy_bound`: Some(tol) bounds the island's mechanical energy at the frame's end by its
/// start's plus the tolerance `tol` of it (gravity's work included, through the potential):
/// when the composed state exceeds it, the chunks' hidden displacement is scaled down
/// (bisection) until the joint law's stored energy fits, and if the kinetic energy alone
/// still exceeds it, the motion relative to the centre of mass (linear momentum kept). It
/// never acts on a state that creates no energy (the oracle's own).
/// Returns the state and the scale applied (1 when the bound did not act).
#[allow(clippy::too_many_arguments)]
pub fn island_frame_bounded(start: &stress_ref::world::World, m: &stress_ref::world::IslandMembers, d: &stress_ref::world::DecodedState, full: &[FullOut], base: Base, ids_from_base: bool, energy_bound: Option<f64>) -> Result<(IslandState, f64), String> {
    let st = island_frame(start, m, d, full, base, ids_from_base)?;
    let Some(tol) = energy_bound else { return Ok((st, 1.0)) };
    let (k0, e0, p0) = start.island_energy(m);
    let budget = k0 + e0 + p0 + tol * (k0 + e0).max(1e-12);
    let total = |x: &IslandState| -> f64 {
        let (k, e, p) = stress_ref::world::World::merge_island(start, x, m).island_energy(m);
        k + e + p
    };
    if total(&st) <= budget {
        return Ok((st, 1.0));
    }
    let bisect = |f: &dyn Fn(f64) -> f64| -> f64 {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if f(mid) <= budget { lo = mid } else { hi = mid }
        }
        lo
    };
    // the stored energy first (the hidden displacement), then, if the kinetic energy alone
    // exceeds the budget, the motion relative to the centre of mass (momentum kept)
    let a = bisect(&|k| {
        let mut x = st.clone();
        x.scale_hidden_displacement(k);
        total(&x)
    });
    let mut x = st;
    x.scale_hidden_displacement(a);
    if total(&x) <= budget {
        return Ok((x, a));
    }
    let b = bisect(&|k| {
        let mut y = x.clone();
        y.scale_relative_motion(k);
        total(&y)
    });
    x.scale_relative_motion(b);
    eprintln!("[energy bound] motion relative to the centre of mass scaled by {b:.4}");
    Ok((x, a.min(b)))
}

pub fn island_frame(start: &stress_ref::world::World, m: &stress_ref::world::IslandMembers, d: &stress_ref::world::DecodedState, full: &[FullOut], base: Base, ids_from_base: bool) -> Result<IslandState, String> {
    let mut x = start.clone();
    x.solver.reset_peak_utilizations();
    // The /1 chunk states' departure from their body's rigid fit kept as hidden deformation,
    // except with `STRESS_FULL_DEFORMATION=0` (rigid bodies; diagnostic: a model's world-state
    // noise kept as deformation is strain); a body table or a predicted hidden state replaces
    // it either way.
    let keep = !std::env::var("STRESS_FULL_DEFORMATION").is_ok_and(|v| v == "0");
    x.apply_decoded_with(d, keep)?;
    // `STRESS_FULL_BODIES=derive`: bodies derived from the predicted hidden state even when a
    // body table is sent (the planning session's contract); default: the body table when sent.
    let derive = std::env::var("STRESS_FULL_BODIES").is_ok_and(|v| v == "derive");
    let hidden = hidden_states(full)?;
    let table = full.iter().find(|o| o.kind == "cluster").filter(|_| !(derive && hidden.is_some()));
    if let Some(o) = table {
        x.set_body_records(&body_records(o)?, &m.chunks)?;
        if !keep {
            // rigid bodies at the model's body records (no hidden state)
            for &(s, c) in &m.chunks {
                let cs = &mut x.solver.chunks[s][c];
                (cs.u, cs.th, cs.v, cs.w) = (stress_ref::math::Vec3::ZERO, stress_ref::math::Vec3::ZERO, stress_ref::math::Vec3::ZERO, stress_ref::math::Vec3::ZERO);
            }
        }
    } else if let Some(h) = &hidden {
        x.derive_bodies(h, &m.chunks)?;
    }
    let decoded = x.extract_island(m);
    let base = match base {
        Base::Start => {
            let mut b = start.clone();
            b.solver.reset_peak_utilizations();
            b.extract_island(m)
        }
        Base::Native => stress_ref::world::oracle_island_step(start, m)?,
    };
    // a body table derived over is not the island's
    let used: Vec<FullOut> = full.iter().filter(|o| o.kind != "cluster" || table.is_some()).cloned().collect();
    apply_full_with(&decoded, &base, &used, ids_from_base)
}

/// The oracle's own outcome of the island's frame as a decoded state `/2` (its `/1` decode
/// with the exact break times and order, and its frame-end columns limited to `predicted`):
/// the full-state contract's ceiling.
pub fn oracle_full2(start: &stress_ref::world::World, m: &stress_ref::world::IslandMembers, frame: i64, predicted: Option<&HashMap<String, Vec<String>>>) -> Result<(stress_ref::world::DecodedState, Vec<FullOut>), String> {
    let mut view = start.clone();
    let island: std::collections::HashSet<u64> = view.solver.clusters.iter().filter(|c| c.chunks.first().is_some_and(|&k| m.chunks.contains(&(c.structure, k)))).map(|c| c.id).collect();
    for cl in &mut view.solver.clusters {
        cl.driven = !island.contains(&cl.id);
    }
    for (i, imp) in view.impactors.iter_mut().enumerate() {
        imp.driven = !m.impactors.contains(&i);
    }
    let keys = view.routed_island_keys();
    let mut model = view.standalone_copy();
    model.step_frame();
    let d = view.decoded_from_oracle(&model, &keys, frame, 0);
    Ok((d, columns_of_state(&stress_ref::world::oracle_island_step(start, m)?, predicted)?))
}

impl Base {
    /// `STRESS_FULL_BASE`: `native` (default) or `start`.
    pub fn from_env() -> Result<Base, String> {
        match std::env::var("STRESS_FULL_BASE").as_deref() {
            Ok("start") => Ok(Base::Start),
            Ok("native") | Err(_) => Ok(Base::Native),
            Ok(other) => Err(format!("STRESS_FULL_BASE={other}: native|start")),
        }
    }
}
