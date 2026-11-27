//! Scenario families: one implementation of a kind of crash (a car into a block, two cars
//! head-on, a moving barrier into a side...), its parameters declared as data, and a function
//! from their values (and the designs' extents) to a run's [`Layout`].
//!
//! A reference scenario (S01...) is a named preset of a family: the family's id and values
//! ([`preset`]), plus what makes it a reference: its criteria, cameras, moments and footage.
//! A page shows a family's parameters as controls, starting from the preset's values
//! ([`CrashScenario::with_values`]); a family can also be opened with any values
//! ([`scenario`]). Changed values are no longer the preset: a scenario whose values differ
//! from its preset's is judged by the law-based criteria alone ([`desc::law_criteria`]:
//! energy within 0.1 % of the run's own starting kinetic energy, nothing into a box, momentum
//! within the outside's impulse), never by bands derived for the preset's numbers.
//!
//! Declaring a parameter is one line in the family's `params` (see [`num`], [`kmh`],
//! [`design`], [`flag`], [`choice`]) and one read of it in its layout ([`Values::f32`],
//! [`Values::text`], [`Values::flag`]).

use super::desc::law_criteria;
use super::*;
use ts_rs::TS;

/// A parameter's value: a number, a flag, or a name (a design, a choice).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, TS)]
#[serde(untagged)]
pub enum Value {
    Num(f64),
    Flag(bool),
    Text(String),
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Value::Num(x)
    }
}
impl From<bool> for Value {
    fn from(x: bool) -> Self {
        Value::Flag(x)
    }
}
impl From<&str> for Value {
    fn from(x: &str) -> Self {
        Value::Text(x.into())
    }
}

/// What a parameter is.
#[derive(Clone, Debug, PartialEq, Serialize, TS)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamKind {
    /// A number in `[min, max]`, moved in `step`s.
    Number { min: f64, max: f64, step: f64, default: f64 },
    /// One of the registered designs ([`DESIGNS`]).
    Design { default: String, choices: Vec<String> },
    Flag { default: bool },
    /// One of `choices`.
    Choice { default: String, choices: Vec<String> },
}

/// A family's parameter, as a page shows it: its key (the URL's and `--set`'s), label and
/// unit, what it means, and its kind.
#[derive(Clone, Debug, PartialEq, Serialize, TS)]
pub struct Param {
    pub key: String,
    pub label: String,
    pub unit: String,
    pub help: String,
    pub kind: ParamKind,
}

/// The designs a design parameter may name: every registered generated design
/// (`engine_vehiclegen::designs::names`) and the FMVSS 214 barrier, each with a crash template
/// (`engine-softbody-authoring` `crash_templates`). This crate cannot see the generator's
/// registry; `engine-softbody-authoring/tests/crash_designs.rs` holds this list to it.
pub const DESIGNS: [&str; 9] = ["sedan", "police_sedan", "wagon", "mach1_1969", "mach1_gasser", "suv_4x4", "truck_cab", "box_trailer", "mdb214"];

/// A number parameter (`unit` as shown).
pub fn num(key: &str, label: &str, unit: &str, default: f64, min: f64, max: f64, step: f64) -> Param {
    Param { key: key.into(), label: label.into(), unit: unit.into(), help: String::new(), kind: ParamKind::Number { min, max, step, default } }
}
/// A speed in km/h (0 to 250); a family reads it in m/s with [`Values::ms`].
pub fn kmh(key: &str, label: &str, default: f64) -> Param {
    num(key, label, "km/h", default, 0.0, 250.0, 0.5)
}
/// An angle in degrees in `[-lim, lim]`.
pub fn degrees(key: &str, label: &str, default: f64, lim: f64) -> Param {
    num(key, label, "°", default, -lim, lim, 1.0)
}
/// A design ([`DESIGNS`]).
pub fn design(key: &str, label: &str, default: &str) -> Param {
    Param { key: key.into(), label: label.into(), unit: String::new(), help: String::new(), kind: ParamKind::Design { default: default.into(), choices: DESIGNS.iter().map(|s| s.to_string()).collect() } }
}
pub fn flag(key: &str, label: &str, default: bool) -> Param {
    Param { key: key.into(), label: label.into(), unit: String::new(), help: String::new(), kind: ParamKind::Flag { default } }
}
pub fn choice(key: &str, label: &str, default: &str, choices: &[&str]) -> Param {
    Param { key: key.into(), label: label.into(), unit: String::new(), help: String::new(), kind: ParamKind::Choice { default: default.into(), choices: choices.iter().map(|s| s.to_string()).collect() } }
}

impl Param {
    /// What it means (the control's tooltip).
    pub fn help(mut self, text: &str) -> Self {
        self.help = text.into();
        self
    }
    pub fn default_value(&self) -> Value {
        match &self.kind {
            ParamKind::Number { default, .. } => Value::Num(*default),
            ParamKind::Flag { default } => Value::Flag(*default),
            ParamKind::Design { default, .. } | ParamKind::Choice { default, .. } => Value::Text(default.clone()),
        }
    }
    /// `v` if it is a value of this parameter, else why not.
    pub fn check(&self, v: &Value) -> Result<(), String> {
        let k = &self.key;
        match (&self.kind, v) {
            (ParamKind::Number { min, max, .. }, Value::Num(x)) if x.is_finite() && *x >= *min && *x <= *max => Ok(()),
            (ParamKind::Number { min, max, .. }, Value::Num(x)) => Err(format!("{k} = {x}: a number in [{min}, {max}]")),
            (ParamKind::Flag { .. }, Value::Flag(_)) => Ok(()),
            (ParamKind::Design { choices, .. } | ParamKind::Choice { choices, .. }, Value::Text(s)) if choices.contains(s) => Ok(()),
            (ParamKind::Design { choices, .. } | ParamKind::Choice { choices, .. }, Value::Text(s)) => Err(format!("{k} = {s}: one of {}", choices.join(", "))),
            (_, v) => Err(format!("{k} = {v:?}: not a value of this parameter ({:?})", self.kind)),
        }
    }
    /// `text` (a URL's or `--set`'s) as a value of this parameter.
    pub fn parse(&self, text: &str) -> Result<Value, String> {
        let v = match &self.kind {
            ParamKind::Number { .. } => Value::Num(text.trim().parse().map_err(|_| format!("{} = {text}: a number", self.key))?),
            ParamKind::Flag { .. } => Value::Flag(matches!(text.trim(), "1" | "true" | "on" | "yes")),
            _ => Value::Text(text.trim().to_string()),
        };
        self.check(&v)?;
        Ok(v)
    }
}

/// A family's values by key: every parameter's, checked ([`Family::values`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, TS)]
pub struct Values(pub BTreeMap<String, Value>);

impl Values {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }
    /// A number (0 when missing: [`Family::values`] fills every parameter).
    pub fn f64(&self, key: &str) -> f64 {
        match self.0.get(key) {
            Some(Value::Num(x)) => *x,
            _ => 0.0,
        }
    }
    pub fn f32(&self, key: &str) -> f32 {
        self.f64(key) as f32
    }
    /// A speed parameter in km/h as m/s (`kmh / 3.6`, in f32 as the scenarios state it).
    pub fn ms(&self, key: &str) -> f32 {
        self.f32(key) / 3.6
    }
    /// An angle parameter in degrees as radians.
    pub fn rad(&self, key: &str) -> f32 {
        self.f32(key).to_radians()
    }
    pub fn text(&self, key: &str) -> &str {
        match self.0.get(key) {
            Some(Value::Text(s)) => s,
            _ => "",
        }
    }
    pub fn flag(&self, key: &str) -> bool {
        matches!(self.0.get(key), Some(Value::Flag(true)))
    }
}

/// A run's layout from a family's values, each design's extent and the run's jitter.
pub type FamilyLayout = fn(&Values, &dyn Fn(&str) -> Extent, Jitter) -> Layout;

/// A kind of crash: its parameters and the one function that lays it out.
pub struct Family {
    pub id: &'static str,
    pub title: &'static str,
    pub about: &'static str,
    pub params: fn() -> Vec<Param>,
    pub layout: FamilyLayout,
    /// Whether it places static boxes (its law criteria then include the boxes').
    pub boxes: bool,
    /// Simulated seconds after first contact when it is opened directly.
    pub secs_after_impact: f32,
}

impl Family {
    /// Every parameter's value: `given`'s where it has one (checked), else the default.
    /// A key that is not a parameter is an error.
    pub fn values(&self, given: &Values) -> Result<Values, String> {
        let params = (self.params)();
        for k in given.0.keys() {
            if !params.iter().any(|p| &p.key == k) {
                return Err(format!("{}: no parameter {k} ({})", self.id, params.iter().map(|p| p.key.as_str()).collect::<Vec<_>>().join(", ")));
            }
        }
        let mut out = Values::default();
        for p in &params {
            // A number or flag given as text (a URL's) is read as one.
            let v = match (given.get(&p.key), &p.kind) {
                (Some(Value::Text(t)), ParamKind::Number { .. } | ParamKind::Flag { .. }) => p.parse(t).map_err(|e| format!("{}: {e}", self.id))?,
                (Some(v), _) => v.clone(),
                (None, _) => p.default_value(),
            };
            p.check(&v).map_err(|e| format!("{}: {e}", self.id))?;
            out.0.insert(p.key.clone(), v);
        }
        Ok(out)
    }
    /// The designs `values` name, in parameter order, each once.
    pub fn designs(&self, values: &Values) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for p in (self.params)() {
            if matches!(p.kind, ParamKind::Design { .. }) {
                let d = values.text(&p.key).to_string();
                if !out.contains(&d) {
                    out.push(d);
                }
            }
        }
        out
    }
    /// `key=value` texts (a URL's, `--set`'s) as values over `base`.
    pub fn parse_onto(&self, base: &Values, sets: &[(String, String)]) -> Result<Values, String> {
        let params = (self.params)();
        let mut out = base.clone();
        for (k, t) in sets {
            let p = params.iter().find(|p| &p.key == k).ok_or_else(|| format!("{}: no parameter {k} ({})", self.id, params.iter().map(|p| p.key.as_str()).collect::<Vec<_>>().join(", ")))?;
            out.0.insert(k.clone(), p.parse(t)?);
        }
        self.values(&out)
    }
}

/// A scenario's layout as a preset of `family` (its values given by key; the rest the
/// family's defaults). Panics on a value the family does not take: the scenarios are code,
/// and `tests::every_preset_is_valid` runs them all.
pub fn preset(family: &'static Family, values: &[(&str, Value)]) -> LayoutSource {
    preset_named(family, &[], values)
}

/// [`preset`] with its bodies named (in the family's order) as the scenario's criteria name
/// them (`["truck", "car"]` for the family's striker and struck).
pub fn preset_named(family: &'static Family, names: &[&str], values: &[(&str, Value)]) -> LayoutSource {
    let given = Values(values.iter().map(|(k, v)| (k.to_string(), v.clone())).collect());
    let values = family.values(&given).unwrap_or_else(|e| panic!("{e}"));
    LayoutSource::Family { family, values, names: names.iter().map(|s| s.to_string()).collect() }
}

/// The family `id` ([`families::all`]).
pub fn get(id: &str) -> Option<&'static Family> {
    super::families::all().into_iter().find(|f| f.id.eq_ignore_ascii_case(id))
}

/// The scenario a page or `crash_ref` asks for: the reference scenario `id` (with `values`
/// changed: [`CrashScenario::with_values`]), or the family `family` opened with `values`.
pub fn resolve(id: Option<&str>, family: Option<&str>, values: &Values) -> Result<CrashScenario, String> {
    match (id, family) {
        (Some(id), _) => {
            let s = super::get(id).ok_or_else(|| format!("no crash reference scenario {id}: {}", all().iter().map(|s| s.id.clone()).collect::<Vec<_>>().join(", ")))?;
            if values.0.is_empty() {
                Ok(s)
            } else {
                s.with_values(values)
            }
        }
        (None, Some(f)) => scenario(get(f).ok_or_else(|| format!("no scenario family {f}: {}", super::families::all().iter().map(|f| f.id).collect::<Vec<_>>().join(", ")))?, values),
        (None, None) => Err("a scenario id or a family id".into()),
    }
}

/// `family` opened directly with `values`: its layout, generic cameras and moments, the
/// law-based criteria.
pub fn scenario(family: &'static Family, values: &Values) -> Result<CrashScenario, String> {
    let values = family.values(values)?;
    Ok(CrashScenario {
        id: family.id.to_string(),
        title: family.title.to_string(),
        reference: Reference { video: "", at_s: &[], real_test: "" },
        designs: family.designs(&values),
        layout: LayoutSource::Family { family, values, names: Vec::new() },
        secs_after_impact: family.secs_after_impact,
        cameras: Vec::new(),
        moments: vec![Moment { name: "before".into(), after_impact_ms: -10.0 }, Moment { name: "peak".into(), after_impact_ms: 60.0 }],
        criteria: law_criteria(family.boxes),
        recreated: false,
        modified_from: None,
    })
}

impl CrashScenario {
    /// The family this scenario is a preset of, and its values.
    pub fn family(&self) -> Option<(&'static Family, &Values)> {
        match &self.layout {
            LayoutSource::Family { family, values, .. } => Some((family, values)),
            _ => None,
        }
    }

    /// This scenario with some of its family's values changed (`changes`: by key). Values
    /// equal to its own give it back as it is; others give the same scenario laid out from
    /// the new values, marked [`CrashScenario::modified_from`], judged by the law-based
    /// criteria alone: its own bands were derived for its own numbers.
    pub fn with_values(&self, changes: &Values) -> Result<CrashScenario, String> {
        let (family, base) = self.family().ok_or_else(|| format!("{}: not a family preset (its layout is its own code): it has no parameters", self.id))?;
        let mut merged = base.clone();
        merged.0.extend(changes.0.iter().map(|(k, v)| (k.clone(), v.clone())));
        let values = family.values(&merged)?;
        if &values == base {
            return Ok(self.clone());
        }
        let mut s = self.clone();
        s.designs = family.designs(&values);
        let names = match &self.layout {
            LayoutSource::Family { names, .. } => names.clone(),
            _ => Vec::new(),
        };
        s.layout = LayoutSource::Family { family, values, names };
        s.criteria = law_criteria(family.boxes);
        s.title = format!("{} (modified)", self.title);
        s.modified_from = Some(self.id.clone());
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ext() -> impl Fn(&str) -> Extent {
        |_d: &str| Extent { lo: Vec3::new(-0.9, 0.0, -2.4), hi: Vec3::new(0.9, 1.4, 2.4), axles: Some((1.4, -1.4)), wheel_low: Some(0.0) }
    }

    #[test]
    fn every_preset_is_valid_and_every_family_is_listed() {
        for s in all() {
            if let Some((f, v)) = s.family() {
                assert_eq!(&f.values(v).unwrap(), v, "{}", s.id);
                assert!(get(f.id).is_some(), "{}: family {} listed", s.id, f.id);
                assert_eq!(s.designs, f.designs(v), "{}: designs from its values", s.id);
            }
        }
        let ids: Vec<&str> = super::super::families::all().iter().map(|f| f.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "family ids are unique");
    }

    #[test]
    fn defaults_fill_and_bad_values_are_refused() {
        let f = get("block").unwrap();
        let v = f.values(&Values::default()).unwrap();
        assert_eq!(v.text("car"), "sedan");
        assert_eq!(v.f64("speed"), 56.3);
        assert!(f.values(&Values([("speed".to_string(), Value::Num(-1.0))].into())).is_err(), "below min");
        assert!(f.values(&Values([("car".to_string(), Value::Text("bus".into()))].into())).is_err(), "not a design");
        assert!(f.values(&Values([("nope".to_string(), Value::Num(1.0))].into())).is_err(), "not a parameter");
        assert!(f.values(&Values([("speed".to_string(), Value::Text("fast".into()))].into())).is_err(), "not a number");
        let set = f.parse_onto(&v, &[("speed".into(), "64".into()), ("car".into(), "wagon".into())]).unwrap();
        assert_eq!((set.f64("speed"), set.text("car")), (64.0, "wagon"));
        assert!(f.parse_onto(&v, &[("speed".into(), "x".into())]).is_err());
    }

    /// The block family for known values: a 40 % block 1 m past the side, a tenth of a second
    /// ahead at 36 km/h (10 m/s).
    #[test]
    fn block_for_known_values() {
        let f = get("block").unwrap();
        let v = f.values(&Values([("speed".to_string(), Value::Num(36.0)), ("overlap".to_string(), Value::Num(0.4)), ("height".to_string(), Value::Num(2.0))].into())).unwrap();
        let l = (f.layout)(&v, &ext(), Jitter::run(0));
        assert_eq!(l.bodies.len(), 1);
        assert!((l.bodies[0].speed - 10.0).abs() < 1e-5);
        let (c, h) = l.boxes[0];
        // x from 0.9 − 0.4 × 1.8 = 0.18 to 1.9; face 2.4 + 1.0 m.
        assert!((c.x - h.x - 0.18).abs() < 1e-5 && (c.x + h.x - 1.9).abs() < 1e-5, "{c} {h}");
        assert!((c.z - h.z - 3.4).abs() < 1e-5, "{c} {h}");
        assert_eq!(l.impact_s, 0.1);
    }

    /// A changed value makes a modified scenario judged by the laws alone; the preset's own
    /// values give the preset back.
    #[test]
    fn with_values_marks_a_modified_scenario() {
        let s = super::super::get("S11").unwrap();
        let same = s.with_values(&Values::default()).unwrap();
        assert!(same.modified_from.is_none());
        assert_eq!((same.criteria)().len(), (s.criteria)().len());
        let m = s.with_values(&Values([("speed".to_string(), Value::Num(64.0))].into())).unwrap();
        assert_eq!(m.modified_from.as_deref(), Some("S11"));
        let laws = (law_criteria(true))();
        assert_eq!((m.criteria)().iter().map(|c| c.metric).collect::<Vec<_>>(), laws.iter().map(|c| c.metric).collect::<Vec<_>>());
        let c = s.with_values(&Values([("car".to_string(), Value::Text("wagon".into()))].into())).unwrap();
        assert_eq!(c.designs, vec!["wagon".to_string()]);
    }
}
