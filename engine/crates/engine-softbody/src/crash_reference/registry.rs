//! The registry pages read (the crash lab, the Reference Studio): every scenario with its
//! reference footage, designs, cameras, moments and criteria, built-in ones first and then
//! the data scenarios given (recreated in the studio). Typed, so the studio's TypeScript is
//! generated from it ([`super::types`]).

use super::*;
use ts_rs::TS;

/// A criterion as a page lists it (before any run): its band, tier, what the band rests on
/// and whether failing it fails a run.
#[derive(Clone, Debug, Serialize, TS)]
pub struct CriterionInfo {
    pub name: String,
    pub metric: String,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub tier: Tier,
    pub why: String,
    pub basis: BasisInfo,
    pub asserted: bool,
}

/// [`Basis`] as JSON: `{ kind: 'law' | 'cited', source }` or `{ kind: 'estimate' }`.
#[derive(Clone, Debug, Serialize, TS)]
pub struct BasisInfo {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub source: Option<String>,
}

impl From<&Criterion> for CriterionInfo {
    fn from(c: &Criterion) -> Self {
        let basis = match c.basis {
            Basis::Law(x) => BasisInfo { kind: "law".into(), source: Some(x.into()) },
            Basis::Cited(x) => BasisInfo { kind: "cited".into(), source: Some(x.into()) },
            Basis::Estimate => BasisInfo { kind: "estimate".into(), source: None },
        };
        Self { name: c.name.into(), metric: c.metric.into(), lo: c.lo, hi: c.hi, tier: c.tier, why: c.why.into(), basis, asserted: c.asserted() }
    }
}

#[derive(Clone, Debug, Serialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ReferenceInfo {
    pub video: String,
    pub at_s: Vec<u32>,
    /// The video at each of its times.
    pub links: Vec<String>,
    pub real_test: String,
}

#[derive(Clone, Debug, Serialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct ScenarioInfo {
    pub id: String,
    pub title: String,
    pub reference: ReferenceInfo,
    pub designs: Vec<String>,
    /// The designs a page loads a mod folder for (the others are built from code).
    pub folder_designs: Vec<String>,
    pub secs_after_impact: f32,
    pub cameras: Vec<Camera>,
    pub moments: Vec<Moment>,
    pub criteria: Vec<CriterionInfo>,
    /// Recreated from data in the Reference Studio (law-based criteria only).
    pub recreated: bool,
    /// The contact setting it runs in by default.
    pub contact: Contact,
    /// A recreated scenario's description.
    #[ts(optional)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub desc: Option<ScenarioDesc>,
}

#[derive(Clone, Debug, Serialize, TS)]
#[serde(rename_all = "camelCase")]
pub struct Registry {
    pub scenarios: Vec<ScenarioInfo>,
    pub contacts: Vec<Contact>,
    pub runs: u32,
    pub sample_steps: u32,
    pub dt: f32,
}

/// One scenario's entry (`from_code`: whether a design is built from code alone, without a
/// mod folder).
pub fn info(s: &CrashScenario, desc: Option<&ScenarioDesc>, from_code: &dyn Fn(&str) -> bool) -> ScenarioInfo {
    ScenarioInfo {
        id: s.id.clone(),
        title: s.title.clone(),
        reference: ReferenceInfo { video: s.reference.video.into(), at_s: s.reference.at_s.to_vec(), links: s.reference.at_s.iter().map(|t| format!("{}&t={t}s", s.reference.video)).collect(), real_test: s.reference.real_test.into() },
        designs: s.designs.clone(),
        folder_designs: s.designs.iter().filter(|d| !from_code(d)).cloned().collect(),
        secs_after_impact: s.secs_after_impact,
        cameras: s.cameras.clone(),
        moments: s.moments.clone(),
        criteria: (s.criteria)().iter().map(CriterionInfo::from).collect(),
        recreated: s.recreated,
        contact: desc.map_or(Contact::Edge, |d| d.contact),
        desc: desc.cloned(),
    }
}

/// Every built-in scenario, then `data`'s.
pub fn registry(data: &[ScenarioDesc], from_code: &dyn Fn(&str) -> bool) -> Registry {
    let mut scenarios: Vec<ScenarioInfo> = all().iter().map(|s| info(s, None, from_code)).collect();
    scenarios.extend(data.iter().map(|d| info(&d.scenario(), Some(d), from_code)));
    Registry { scenarios, contacts: Contact::ALL.to_vec(), runs: RUNS as u32, sample_steps: SAMPLE_STEPS, dt: DT }
}
