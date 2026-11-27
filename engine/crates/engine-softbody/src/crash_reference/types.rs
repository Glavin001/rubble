//! The Reference Studio's TypeScript, generated from these Rust types (`ts-rs`, as
//! `engine-world`'s `play::api::typescript`): `apps/reference-studio/src/generated/crash-reference.ts`,
//! checked in; `engine-softbody/tests/studio_types.rs` fails when it is stale (`BLESS=1`
//! rewrites it). The studio has no hand-written copy of any of them.

/// Every declaration, in one file.
pub fn typescript() -> String {
    use ts_rs::{Config, TS};
    let cfg = Config::new().with_large_int("number");
    let mut out = String::from(
        "// Generated from engine-softbody's Rust types (crash_reference::types::typescript). Do not edit:\n\
         // `BLESS=1 cargo test -p engine-softbody --test studio_types` regenerates it.\n\n\
         export type JsonValue = number | string | boolean | Array<JsonValue> | { [key in string]?: JsonValue } | null;\n",
    );
    macro_rules! decl {
        ($($t:ty),* $(,)?) => {$(
            out.push('\n');
            out.push_str("export ");
            out.push_str(&<$t as TS>::decl(&cfg));
            out.push('\n');
        )*};
    }
    use super::*;
    decl!(
        ScenarioDesc,
        BodyDesc,
        BoxDesc,
        Camera,
        Moment,
        Contact,
        CrashReport,
        CriterionResult,
        Tier,
        registry::Registry,
        registry::ScenarioInfo,
        registry::ReferenceInfo,
        registry::CriterionInfo,
        registry::BasisInfo,
        recording::Header,
        recording::RecBody,
        outline::FreeCamera,
        outline::FrameCamera,
        shape::Vis,
        shape::ElementValue,
        shape::Instance,
        studio::SceneSource,
        studio::SceneBody,
        studio::SceneBox,
        studio::SceneInfo,
        studio::RecordingStatus,
        studio::CpuProgress,
        studio::RubbleSpec,
        studio::TracingSpec,
        studio::ShapeSpec,
        studio::CompareSource,
    );
    out
}
