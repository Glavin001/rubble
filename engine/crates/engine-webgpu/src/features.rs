//! Construction options for [`crate::WgpuSim`]: contact storage, the vehicle and
//! soft-body pools, and the engine's internal path switches.
//!
//! [`EngineFeatures::default`] is the shipped configuration. The `ENGINE_*`
//! environment variables stay as benchmark controls: [`EngineFeatures::from_env`]
//! resolves them exactly as the constructors always have, and a host that sets its
//! options explicitly never reads the environment.

use engine_softbody::SoftCapacity;
use engine_vehicle::VehicleCapacity;

/// Contact record layout (DECISIONS 8cz/8dg).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ContactStorage {
    /// 128-byte contact rows; soft bodies and `last_contacts()` read this layout.
    Canonical,
    /// 32 B pair + 64 B point records, rebuilt each step.
    Compact,
    /// Compact records kept across steps (the shipped default).
    #[default]
    Persistent,
}

impl ContactStorage {
    /// `ENGINE_CONTACTS=canonical|compact` (and the older
    /// `ENGINE_COMPACT_CONTACTS=1` / `ENGINE_PERSISTENT_CONTACTS=0`); persistent
    /// otherwise.
    pub fn from_env() -> Self {
        let legacy_compact = env_str("ENGINE_COMPACT_CONTACTS").as_deref() == Some("1");
        let legacy_off = env_str("ENGINE_PERSISTENT_CONTACTS").as_deref() == Some("0");
        match env_str("ENGINE_CONTACTS").as_deref() {
            Some("canonical") => Self::Canonical,
            Some("compact") => Self::Compact,
            _ if legacy_compact => Self::Compact,
            _ if legacy_off => Self::Canonical,
            _ => Self::Persistent,
        }
    }

    pub(crate) fn compact(self) -> bool {
        self != Self::Canonical
    }

    pub(crate) fn persistent(self) -> bool {
        self == Self::Persistent
    }
}

/// GPU timestamps around the rigid step ([`crate::WgpuSim::set_gpu_timing`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GpuTiming {
    /// No timestamp queries (the shipped default).
    #[default]
    Off,
    /// One pair around the step's compute pass: its GPU time.
    Step,
    /// A pass and a pair per phase of the plan (`broadphase`, `narrowphase`,
    /// `warm`, `graph`, ..., each iteration's `primal` and `dual`, `sleep`).
    Phases,
}

/// How manifolds persist between steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ManifoldPersistence {
    /// Per-contact persistence (shipped).
    #[default]
    Contacts,
    /// `ENGINE_MANIFOLD_PERSISTENCE=1`: per-manifold persistence pass.
    Manifolds,
    /// `ENGINE_MANIFOLD_PERSISTENCE=fused`: fused into the compact pipeline.
    Fused,
}

/// Warm-start hash records (8ed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WarmHash {
    /// 16-byte records with the pair key and previous run (shipped).
    #[default]
    Record,
    /// `ENGINE_HASH_INSERT=manifold`: one insert per previous manifold.
    Manifold,
    /// `ENGINE_HASH_INSERT=contact`: one insert per contact.
    Contact,
}

/// Broadphase grid layout (8ej).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GridLayout {
    /// Awake dynamics and sleepers in separate tables (shipped).
    #[default]
    Split,
    /// `ENGINE_GRID=scaled`: one table sized to the world.
    Scaled,
    /// `ENGINE_GRID=fixed`: the old 131072-head table.
    Fixed,
}

/// What a world with sleep does with the contacts of pairs at rest ([`EngineFeatures::resting_contacts`];
/// `World` applies it when sleep is on).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RestingContacts {
    /// The sleep archive (the shipped default): a sleeper a moving body touches forms
    /// all its resting pairs in that step, and woken pairs restart cold as the carry's do
    /// (`EngineFeatures::archive_cold`; warm from archived rows with it off). A sleeping
    /// pair costs nothing per step. Canonical and persistent contacts; compact contacts
    /// keep the carry.
    #[default]
    Archive,
    /// The resting-pair carry (opt-out, `ENGINE_REST=carry`): every pair at rest is
    /// re-emitted and kept each step (the earlier default; costs a row per resting
    /// contact every step).
    Carry,
    /// Neither (measurement, `ENGINE_REST=drop`): a sleeping pair keeps no rows and a
    /// woken one restarts cold.
    Drop,
}

/// The engine's internal path switches. Every variant computes the same physics
/// within its documented tolerance; the defaults are the measured fastest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineFeatures {
    /// First sparse primal colour on small worlds (`ENGINE_SPARSE_COLORS`, max 32).
    pub sparse_colors: u32,
    pub manifold_persistence: ManifoldPersistence,
    pub warm_hash: WarmHash,
    /// Narrowphase finishes reused copies (`ENGINE_REUSE=pass` turns it off).
    pub reuse_copy: bool,
    /// Two-pass narrowphase (`ENGINE_NP=fused` turns it off).
    pub np_split: bool,
    /// Persist work list (`ENGINE_PERSIST=scan` turns it off).
    pub persist_list: bool,
    /// Local incremental colour repair (`ENGINE_REPAIR=rounds` turns it off).
    pub local_repair: bool,
    /// Carry resting pairs across sleep (`ENGINE_REST=drop` turns it off).
    pub carry_rest: bool,
    /// Awake-indexed broadphase on hierarchical grids: a persistent resting table for
    /// static and sleeping colliders, and only moving colliders bounded, inserted and
    /// queried each step (`ENGINE_BROADPHASE=full` turns it off; same pairs).
    pub awake_broadphase: bool,
    /// With `awake_broadphase`: the uniform split grid's other table kept across steps as
    /// the resting table too, and only moving colliders inserted and queried
    /// (`ENGINE_BROADPHASE=hier` keeps the full pass on uniform grids; same pairs).
    pub awake_uniform: bool,
    /// With `awake_broadphase`: colliders that come to rest join the resting table at the
    /// end of their step instead of a rebuild of the whole table after every step with
    /// sleep events (`ENGINE_REST_TABLE=rebuild` turns it off; same pairs).
    pub rest_incremental: bool,
    /// `ccd_wake` and `ccd_unhold` as passes a thread per collider, dispatched only while
    /// a CCD body sweeps fast (`ENGINE_CCD_WAKE=body` keeps the per-body passes, whose
    /// collider search assumes colliders sorted by body; same results otherwise).
    pub ccd_by_collider: bool,
    /// `predict_bodies` and `extract_velocity_sleep` over this step's awake list where
    /// nothing else changes a body between the list and them (`ENGINE_BODY_PASSES=all`
    /// runs them over every body slot; same results).
    pub awake_body_passes: bool,
    /// The primal's headroom colours (past last step's colour count) solved with their
    /// colour dispatches' lanes in the sparse tail's single workgroup, one dispatch an
    /// iteration (`ENGINE_PRIMAL=dispatch` dispatches each headroom colour; same results).
    pub primal_tail: bool,
    /// Woken pairs start cold (λ = 0, stiffness at its floor), as the resting-pair carry
    /// restarts them (its carried rows are re-detected at rest each step), not from the
    /// sleep archive's rows (default; `ENGINE_ARCHIVE=warm` warm-starts them from the
    /// archive, the earlier default). A struck sleeping grid: warm-started, a woken box is
    /// gripped by its neighbours' full resting load from its first step and a ball barely
    /// dents the stack; cold, it collapses as with the carry. With the touch query, cold
    /// keeps no archived rows at all.
    pub archive_cold: bool,
    /// Measurement variant: rounds of `archive_spread` (sleepers resting on touched ones
    /// get their archived pairs too; `ENGINE_ARCHIVE_SPREAD=n`; default 0). With the touch
    /// query: that many more hops of its spread than the one it always runs.
    pub archive_spread_rounds: u32,
    /// Sleep archive: a sleeper a moving body touches forms all its resting pairs in the
    /// step it is touched (`archive_touch`), and its resting rows keep their archived warm
    /// start (`ENGINE_ARCHIVE_WAKE=emit` turns it off: only the pairs filed in its own
    /// archive are emitted, the earlier path).
    pub archive_touch_query: bool,
    /// Keep the resting-pair carry when a world enables the sleep archive (which turns it
    /// off by default; `ENGINE_REST=keep`): the behaviour before that change.
    pub carry_with_archive: bool,
    /// Resting contacts in worlds with sleep (`World`; [`RestingContacts::Archive`] by default).
    pub resting_contacts: RestingContacts,
    pub grid: GridLayout,
    /// Hierarchical grid override (`ENGINE_GRID_LEVELS=on|off`; `None`: convex scenes).
    pub grid_levels: Option<bool>,
    /// Primal colour headroom over last step (`ENGINE_COLOR_HEADROOM`; `None`: whole palette).
    pub color_headroom: Option<u32>,
    /// JP round headroom (`ENGINE_COLOR_ROUND_HEADROOM`; `None`: all rounds).
    pub color_round_headroom: Option<u32>,
    /// One dispatch per soft particle colour (`ENGINE_SOFT_UNFUSED`).
    pub soft_unfused: bool,
    /// Diagnostic dispatch filters (`WG_ONLY` / `WG_SKIP`).
    pub wg_only: Option<Vec<String>>,
    pub wg_skip: Option<Vec<String>>,
}

impl Default for EngineFeatures {
    fn default() -> Self {
        Self {
            sparse_colors: 32,
            manifold_persistence: ManifoldPersistence::Contacts,
            warm_hash: WarmHash::Record,
            reuse_copy: true,
            np_split: true,
            persist_list: true,
            local_repair: true,
            carry_rest: true,
            awake_broadphase: true,
            awake_uniform: true,
            rest_incremental: true,
            ccd_by_collider: true,
            awake_body_passes: true,
            primal_tail: true,
            archive_cold: true,
            archive_spread_rounds: 0,
            archive_touch_query: true,
            carry_with_archive: false,
            resting_contacts: RestingContacts::Archive,
            grid: GridLayout::Split,
            grid_levels: None,
            color_headroom: Some(4),
            color_round_headroom: Some(2),
            soft_unfused: false,
            wg_only: None,
            wg_skip: None,
        }
    }
}

impl EngineFeatures {
    /// The defaults with every `ENGINE_*` control variable applied.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            sparse_colors: env_str("ENGINE_SPARSE_COLORS")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(d.sparse_colors)
                .min(32),
            manifold_persistence: match env_str("ENGINE_MANIFOLD_PERSISTENCE").as_deref() {
                Some("1") => ManifoldPersistence::Manifolds,
                Some("fused") => ManifoldPersistence::Fused,
                _ => d.manifold_persistence,
            },
            warm_hash: match env_str("ENGINE_HASH_INSERT").as_deref() {
                Some("contact") => WarmHash::Contact,
                Some("manifold") => WarmHash::Manifold,
                _ => d.warm_hash,
            },
            reuse_copy: env_default_on("ENGINE_REUSE", &["pass"]),
            np_split: env_default_on("ENGINE_REUSE", &["pass"])
                && env_default_on("ENGINE_NP", &["fused"]),
            persist_list: env_default_on("ENGINE_REUSE", &["pass"])
                && env_default_on("ENGINE_PERSIST", &["scan"]),
            local_repair: env_default_on("ENGINE_REPAIR", &["rounds"]),
            carry_rest: env_default_on("ENGINE_REST", &["drop"]),
            awake_broadphase: env_default_on("ENGINE_BROADPHASE", &["full"]),
            awake_uniform: env_default_on("ENGINE_BROADPHASE", &["full", "hier"]),
            rest_incremental: env_default_on("ENGINE_REST_TABLE", &["rebuild"]),
            ccd_by_collider: env_default_on("ENGINE_CCD_WAKE", &["body"]),
            awake_body_passes: env_default_on("ENGINE_BODY_PASSES", &["all"]),
            primal_tail: env_default_on("ENGINE_PRIMAL", &["dispatch"]),
            archive_cold: env_str("ENGINE_ARCHIVE").as_deref() != Some("warm"),
            archive_spread_rounds: env_str("ENGINE_ARCHIVE_SPREAD")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0),
            archive_touch_query: env_str("ENGINE_ARCHIVE_WAKE").as_deref() != Some("emit"),
            carry_with_archive: env_str("ENGINE_REST").as_deref() == Some("keep"),
            resting_contacts: match env_str("ENGINE_REST").as_deref() {
                Some("carry") => RestingContacts::Carry,
                Some("drop") => RestingContacts::Drop,
                _ => d.resting_contacts,
            },
            grid: match env_str("ENGINE_GRID").as_deref() {
                Some("scaled") => GridLayout::Scaled,
                Some("fixed") => GridLayout::Fixed,
                _ => d.grid,
            },
            grid_levels: match env_str("ENGINE_GRID_LEVELS").as_deref() {
                Some("off") => Some(false),
                Some("on") => Some(true),
                _ => d.grid_levels,
            },
            color_headroom: env_headroom("ENGINE_COLOR_HEADROOM", 4),
            color_round_headroom: env_headroom("ENGINE_COLOR_ROUND_HEADROOM", 2),
            soft_unfused: env_str("ENGINE_SOFT_UNFUSED").is_some(),
            wg_only: env_list("WG_ONLY"),
            wg_skip: env_list("WG_SKIP"),
        }
    }
}

/// Everything [`crate::WgpuSim::with_device_options`] needs beyond the capacity
/// and the device.
#[derive(Clone, Debug, Default)]
pub struct SimOptions {
    pub contacts: ContactStorage,
    pub vehicles: VehicleCapacity,
    /// The character, query and projectile pool (binding 33).
    pub characters: engine_schema::character::CharacterCapacity,
    /// A soft-body arena; needs [`ContactStorage::Canonical`].
    pub soft: Option<SoftCapacity>,
    pub features: EngineFeatures,
    /// Run `pose_pack` after each step: compact render poses and the bodies that
    /// moved ([`crate::WgpuSim::pose_buffer`], `moved_buffer`). Read-only on the
    /// physics state, so the simulation is the same with it on or off.
    pub pose_output: bool,
    /// Directed debris buffers (`guide_step.slang`); `None`: guidance cannot be
    /// enabled in this world.
    pub guidance: Option<engine_schema::guidance::GuidanceCapacity>,
}

impl SimOptions {
    /// What [`crate::WgpuSim::new`] builds: environment-resolved contacts and features.
    pub fn from_env() -> Self {
        Self {
            contacts: ContactStorage::from_env(),
            features: EngineFeatures::from_env(),
            ..Self::default()
        }
    }
}

/// ENGINE_AUDIT P0: every measured win is the default on every host; an
/// `ENGINE_*` variable set to its control value restores the old path.
pub(crate) fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// `true` unless `name` is set to one of the listed control values.
fn env_default_on(name: &str, controls: &[&str]) -> bool {
    match env_str(name) {
        Some(v) => !controls.contains(&v.as_str()),
        None => true,
    }
}

fn env_list(name: &str) -> Option<Vec<String>> {
    env_str(name).map(|v| v.split(',').map(|n| n.trim().to_string()).collect())
}

/// Headroom over last step's count; `ENGINE_<X>=full` dispatches everything.
fn env_headroom(name: &str, default: u32) -> Option<u32> {
    match env_str(name).as_deref() {
        Some("full") => None,
        Some(v) => v.parse().ok().or(Some(default)),
        None => Some(default),
    }
}

/// Binding `binding` of `bytes` must fit a storage binding and a buffer on `limits`.
pub(crate) fn check_limits(
    limits: &wgpu::Limits,
    binding: usize,
    bytes: u64,
) -> anyhow::Result<()> {
    // Binding 6 is a uniform; binding 22 (the pass table) is bound 16 B at a
    // dynamic offset, so only the buffer limit applies; the rest are storage.
    let binding_max = match binding {
        6 => limits.max_uniform_buffer_binding_size,
        22 => limits.max_buffer_size,
        _ => limits.max_storage_buffer_binding_size,
    };
    anyhow::ensure!(
        bytes <= limits.max_buffer_size && bytes <= binding_max,
        "binding {binding} needs {bytes} B; the device allows {} B per buffer and {binding_max} B \
         per binding: lower the WorldCapacity or request higher limits on the device",
        limits.max_buffer_size
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_exec::rigid_plan;
    use engine_schema::WorldCapacity;

    #[test]
    fn capacity_past_a_small_device_fails_with_a_clear_error() {
        let limits = wgpu::Limits::downlevel_defaults();
        let sizes = rigid_plan::buffer_sizes(&WorldCapacity::scale(), true, true);
        let err = sizes
            .iter()
            .enumerate()
            .find_map(|(i, &b)| check_limits(&limits, i, b).err())
            .expect("the scale preset exceeds downlevel limits");
        assert!(err.to_string().contains("lower the WorldCapacity"), "{err}");
        let small = rigid_plan::buffer_sizes(&WorldCapacity::small(), false, false);
        let adapter_like = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 30,
            ..limits
        };
        for (i, &b) in small.iter().enumerate() {
            check_limits(&adapter_like, i, b).expect("small preset fits 1 GiB bindings");
        }
    }
}
