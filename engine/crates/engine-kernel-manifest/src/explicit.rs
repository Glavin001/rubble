//! The explicit-beams solver's kernel modules: every module's name, Slang entry point
//! and `-D` defines (`slang/kernels/explicit_beams.slang`), which every host's build
//! compiles. The plan (`engine_exec::explicit_plan`) re-exports them.

/// The vehicle sleep awake list's pass (`Kernel::Sleep`).
pub const SLEEP_LIST: &str = "explicit_beams_sleep_list";

/// One compiled kernel module: its name (the WGSL file / MSL / CUDA module), the Slang
/// entry point and the `-D` defines. Every host compiles `MODULES` from
/// `slang/kernels/explicit_beams.slang`.
#[derive(Clone, Copy, Debug)]
pub struct Module {
    pub name: &'static str,
    pub entry: &'static str,
    pub defines: &'static [(&'static str, &'static str)],
}

/// The frame entry point (frame, solo and contact modules).
pub const FRAME_ENTRY: &str = "explicit_beams_frame";
/// The wide passes, in dispatch-kind order (`Kernel::Wide`).
pub const WIDE_PASSES: [&str; 11] = [
    "explicit_beams_wide_begin",
    "explicit_beams_wide_rebase",
    "explicit_beams_wide_nodes",
    "explicit_beams_wide_beams",
    "explicit_beams_wide_tris",
    "explicit_beams_wide_end",
    "explicit_beams_wide_close",
    "explicit_beams_wide_hits",
    "explicit_beams_wide_centred_beams",
    "explicit_beams_wide_centred_nodes",
    "explicit_beams_wide_aux",
];
/// The vehicle-contact plan passes (`Kernel::Vv`): bounds, pairs, group.
pub const VV_PASSES: [&str; 3] = [
    "explicit_beams_vv_bounds",
    "explicit_beams_vv_pairs",
    "explicit_beams_vv_group",
];

pub const W: (&str, &str) = ("ENGINE_WGSL", "1");
pub const L256: (&str, &str) = ("EB_LANES", "256");
pub const L512: (&str, &str) = ("EB_LANES", "512");
pub const AERO: (&str, &str) = ("EB_AERO", "1");
pub const SOLO: (&str, &str) = ("EB_VV_SOLO", "1");
pub const GROUPS: (&str, &str) = ("EB_VV_GROUPS", "1");
pub const NOSURF: (&str, &str) = ("EB_NO_SURFACE", "1");
pub const SMALL: (&str, &str) = ("EB_VV_SMALL_ONLY", "1");
pub const BIG: (&str, &str) = ("EB_VV_BIG_ONLY", "1");
/// The larger contact groups' entry point (`Kernel::ContactBig`).
pub const CONTACT_ENTRY: &str = "explicit_beams_contact";
const fn big(name: &'static str, defines: &'static [(&'static str, &'static str)]) -> Module {
    Module {
        name,
        entry: CONTACT_ENTRY,
        defines,
    }
}

const fn frame(name: &'static str, defines: &'static [(&'static str, &'static str)]) -> Module {
    Module {
        name,
        entry: FRAME_ENTRY,
        defines,
    }
}
const fn pass(name: &'static str) -> Module {
    Module {
        name,
        entry: name,
        defines: &[W],
    }
}

/// Every shipped explicit-beams module (the defines are those engine-webgpu's
/// build.rs passes for the WGSL of the same name).
pub const MODULES: &[Module] = &[
    frame("explicit_beams", &[W]),
    frame("explicit_beams_wide", &[W, L512]),
    frame("explicit_beams_aero", &[W, L256, AERO]),
    frame("explicit_beams_aero_wide", &[W, L512, AERO]),
    frame("explicit_beams_solo", &[W, L256, SOLO]),
    frame("explicit_beams_solo_wide", &[W, L512, SOLO]),
    frame("explicit_beams_solo_aero", &[W, L256, AERO, SOLO]),
    frame("explicit_beams_solo_aero_wide", &[W, L512, AERO, SOLO]),
    // The frame with the contact groups of at most four members (EB_VV_SMALL_ONLY);
    // the larger groups' modules follow (Kernel::ContactBig).
    frame("explicit_beams_contact", &[W, L256, SOLO, GROUPS, SMALL]),
    frame("explicit_beams_contact_wide", &[W, L512, SOLO, GROUPS, SMALL]),
    frame(
        "explicit_beams_contact_aero",
        &[W, L256, AERO, SOLO, GROUPS, SMALL],
    ),
    frame(
        "explicit_beams_contact_aero_wide",
        &[W, L512, AERO, SOLO, GROUPS, SMALL],
    ),
    big("explicit_beams_contact_big", &[W, L256, SOLO, GROUPS, BIG]),
    big("explicit_beams_contact_big_wide", &[W, L512, SOLO, GROUPS, BIG]),
    big("explicit_beams_contact_big_aero", &[W, L256, AERO, SOLO, GROUPS, BIG]),
    big("explicit_beams_contact_big_aero_wide", &[W, L512, AERO, SOLO, GROUPS, BIG]),
    // The same twelve without surface contact (Kernel::module_for).
    frame("explicit_beams_nosurf", &[W, NOSURF]),
    frame("explicit_beams_wide_nosurf", &[W, L512, NOSURF]),
    frame("explicit_beams_aero_nosurf", &[W, L256, AERO, NOSURF]),
    frame("explicit_beams_aero_wide_nosurf", &[W, L512, AERO, NOSURF]),
    frame("explicit_beams_solo_nosurf", &[W, L256, SOLO, NOSURF]),
    frame("explicit_beams_solo_wide_nosurf", &[W, L512, SOLO, NOSURF]),
    frame("explicit_beams_solo_aero_nosurf", &[W, L256, AERO, SOLO, NOSURF]),
    frame("explicit_beams_solo_aero_wide_nosurf", &[W, L512, AERO, SOLO, NOSURF]),
    frame("explicit_beams_contact_nosurf", &[W, L256, SOLO, GROUPS, SMALL, NOSURF]),
    frame("explicit_beams_contact_wide_nosurf", &[W, L512, SOLO, GROUPS, SMALL, NOSURF]),
    frame("explicit_beams_contact_aero_nosurf", &[W, L256, AERO, SOLO, GROUPS, SMALL, NOSURF]),
    frame("explicit_beams_contact_aero_wide_nosurf", &[W, L512, AERO, SOLO, GROUPS, SMALL, NOSURF]),
    big("explicit_beams_contact_big_nosurf", &[W, L256, SOLO, GROUPS, BIG, NOSURF]),
    big("explicit_beams_contact_big_wide_nosurf", &[W, L512, SOLO, GROUPS, BIG, NOSURF]),
    big("explicit_beams_contact_big_aero_nosurf", &[W, L256, AERO, SOLO, GROUPS, BIG, NOSURF]),
    big("explicit_beams_contact_big_aero_wide_nosurf", &[W, L512, AERO, SOLO, GROUPS, BIG, NOSURF]),
    pass(VV_PASSES[0]),
    pass(VV_PASSES[1]),
    pass(VV_PASSES[2]),
    pass(WIDE_PASSES[0]),
    pass(WIDE_PASSES[1]),
    pass(WIDE_PASSES[2]),
    pass(WIDE_PASSES[3]),
    pass(WIDE_PASSES[4]),
    pass(WIDE_PASSES[5]),
    pass(WIDE_PASSES[6]),
    pass(WIDE_PASSES[7]),
    pass(WIDE_PASSES[8]),
    pass(WIDE_PASSES[9]),
    Module {
        name: WIDE_PASSES[10],
        entry: WIDE_PASSES[10],
        defines: &[W, L512],
    },
    Module {
        name: SLEEP_LIST,
        entry: SLEEP_LIST,
        defines: &[W, ("EB_SLEEP_LIST", "1")],
    },
];

/// A module of the split paths, named by its frame variant (`split_modules`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitModule {
    pub name: String,
    pub entry: &'static str,
    pub defines: Vec<(&'static str, &'static str)>,
}

/// The frame module's four variants: the tag their split modules' names end with,
/// whether they have wings (`EB_AERO`), and their lanes.
const VARIANTS: [(&str, bool, (&str, &str)); 4] =
    [("", false, L256), ("_wide", false, L512), ("_aero", true, L256), ("_aero_wide", true, L512)];

/// The tag of the frame variant with `aero` and `lanes` (`VARIANTS`).
fn variant_tag(aero: bool, lanes: u32) -> &'static str {
    match (aero, lanes) {
        (false, 512) => "_wide",
        (false, _) => "",
        (true, 512) => "_aero_wide",
        (true, _) => "_aero",
    }
}

/// The split group path's module `which` (member, group, tests, fleetfront, fleetplan,
/// fleetpost, fleetlevels, wheels) at the frame variant with `aero` and `lanes`.
pub fn split_module_name(which: &str, aero: bool, lanes: u32) -> String {
    format!("explicit_beams_split_{which}{}", variant_tag(aero, lanes))
}

/// The self-collision split's modules at the frame variant with `aero` and `lanes`,
/// with vehicle contact (`vv`) or not: (name, entry, binds the arguments at binding 10).
/// With contact, the split group path's group and tests modules follow.
pub fn self_split_module_names(aero: bool, lanes: u32, vv: bool) -> Vec<(String, &'static str, bool)> {
    let tag = variant_tag(aero, lanes);
    if vv {
        vec![
            (format!("explicit_beams_selfsplit_member_vv{tag}"), SPLIT_MEMBER_ENTRY, true),
            (format!("explicit_beams_selfsplit_walk_vv{tag}"), SELF_WALK_ENTRY, false),
            (format!("explicit_beams_selfsplit_pairs_vv{tag}"), SELF_PAIRS_ENTRY, false),
            (split_module_name("group", aero, lanes), SPLIT_GROUP_ENTRY, false),
            (split_module_name("tests", aero, lanes), SPLIT_TESTS_ENTRY, false),
        ]
    } else {
        vec![
            (format!("explicit_beams_selfsplit_step{tag}"), SELF_STEP_ENTRY, true),
            (format!("explicit_beams_selfsplit_walk{tag}"), SELF_WALK_ENTRY, false),
            (format!("explicit_beams_selfsplit_pairs{tag}"), SELF_PAIRS_ENTRY, false),
        ]
    }
}

/// The wide passes' list module for a fleet at `lanes` (wide fleets have no aero).
pub fn wide_list_module_name(lanes: u32) -> &'static str {
    if lanes == 512 {
        "explicit_beams_selfsplit_list_wide"
    } else {
        "explicit_beams_selfsplit_list"
    }
}

pub const SPLIT_MEMBER_ENTRY: &str = "explicit_beams_split_member";
pub const SPLIT_GROUP_ENTRY: &str = "explicit_beams_split_group";
/// `SPLIT_GROUP_ENTRY`'s dispatches one kind at a time, each in a module of its own
/// (`schedule::GroupEntry`): the frame's start, a step's levels under the default
/// fleet flags (`schedule::levels_once_entry`) and the levels across the GPU's end.
/// One kernel with every pass took Metal 267 s to compile cold in Chrome.
pub const SPLIT_GROUP_BEGIN_ENTRY: &str = "explicit_beams_split_group_begin";
pub const SPLIT_GROUP_LEVELS_ENTRY: &str = "explicit_beams_split_group_levels";
pub const SPLIT_GROUP_LV_ENTRY: &str = "explicit_beams_split_group_lv";
pub const SPLIT_WHEELS_ENTRY: &str = "explicit_beams_split_wheels";
/// Edge vehicle contact's pass, a dispatch a substep after the groups' contact pass.
pub const SPLIT_EDGE_ENTRY: &str = "explicit_beams_split_edge";
pub const SPLIT_TESTS_ENTRY: &str = "explicit_beams_split_tests";
pub const SELF_STEP_ENTRY: &str = "explicit_beams_self_step";
pub const SELF_WALK_ENTRY: &str = "explicit_beams_self_walk";
pub const SELF_PAIRS_ENTRY: &str = "explicit_beams_self_pairs";
pub const WIDE_LIST_ENTRY: &str = "explicit_beams_wide_list";
/// Surface contact on the self split paths: a solo vehicle's self-collision hits of a step
/// applied in a dispatch of its own, before the surface pass's pipelines (the kernel's
/// soloRest).
pub const SOLO_REST_ENTRY: &str = "explicit_beams_solo_rest";

/// The module of [`SOLO_REST_ENTRY`] at the frame variant with `aero` and `lanes`, with
/// vehicle contact (`vv`) or not: the self split's step module's defines, another entry.
pub fn solo_rest_module_name(aero: bool, lanes: u32, vv: bool) -> String {
    format!("explicit_beams_selfsplit_rest{}{}", if vv { "_vv" } else { "" }, variant_tag(aero, lanes))
}

/// The split group path's modules by `which` (`split_module_name`), each at the frame
/// module's four variants (`split_modules`).
pub const SPLIT_WHICH: [&str; 12] = [
    "member", "group", "tests", "fleetfront", "fleetplan", "fleetpost", "fleetlevels", "wheels", "groupbegin", "grouplevels", "grouplv",
    "edges",
];

/// Every module of the split paths, which every host's build compiles from the one
/// Slang source as it does `MODULES`: the split group path (EB_VV_SPLIT;
/// docs/measurements/contact-groups-design.md: the members' steps, the groups' passes,
/// their tests across workgroups and the fleet-wide front and levels), self-collision
/// split (EB_SELF_SPLIT; docs/measurements/selfcol-design.md), the wide passes' reused
/// pair list, and self split with vehicle contact. At the frame module's four variants.
pub fn split_modules() -> Vec<SplitModule> {
    let mut out = Vec::new();
    let mut push = |name: String, entry: &'static str, defines: Vec<(&'static str, &'static str)>| {
        out.push(SplitModule { name, entry, defines });
    };
    for (tag, aero, lanes) in VARIANTS {
        for which in SPLIT_WHICH {
            // (The members' module without EB_VV_GROUPS: its frame step never reaches
            // contactGroup, so it binds the workgroup variables of the steps alone:
            // Metal allows 31 a kernel.)
            let mut d = vec![W, lanes, SOLO, ("EB_VV_SPLIT", "1")];
            if aero {
                d.push(AERO);
            }
            d.push(match which {
                "member" => ("EB_SOLO_STEP", "1"),
                // Raycast wheels on each other (explicit_beams_split_wheels: the groups'
                // module's defines, a small kernel).
                "group" | "wheels" | "groupbegin" | "grouplevels" | "grouplv" | "edges" => GROUPS,
                // The front as fleet-wide dispatches (explicit_beams_fleet.slang): its
                // thread kernels, and the one that writes the next dispatches' counts.
                "fleetfront" => ("EB_FLEET_FRONT", "1"),
                "fleetplan" => ("EB_FLEET_PLAN", "1"),
                "fleetpost" => ("EB_FLEET_POST", "1"),
                // The levels across the GPU (explicit_beams_fleet_levels.slang).
                "fleetlevels" => ("EB_FLEET_LEVELS", "1"),
                // The tests across workgroups: their counters in global memory.
                _ => ("EB_VV_GLOBAL", "1"),
            });
            // The owners' state by vehicle (explicit_beams_fleet.slang), in the groups'
            // and the tests' modules.
            if which != "member" {
                d.push(("EB_VV_FLEET", "1"));
            }
            // Edge vehicle contact's pass (explicit_beams_split_edges.slang): its code in
            // this module alone, not in the contact modules (Chrome's compile limit).
            if which == "edges" {
                d.push(("EB_EDGE", "1"));
            }
            // Its node hits' residual clamp (vvResOf's lam0): the groups' module runs a
            // frame's levels whenever edge contact is on (the fleet-wide schedules are off).
            if which == "group" {
                d.push(("EB_EDGE_CLAMP", "1"));
            }
            // The levels-once module runs only with the fleet-wide schedules, which an upload
            // with contact offsets leaves off (PERF_FLEET_FAMILY): no offsets' code in it.
            if which == "grouplevels" {
                d.push(("EB_NO_OFFSETS", "1"));
            }
            let entry = match which {
                "member" => SPLIT_MEMBER_ENTRY,
                "group" => SPLIT_GROUP_ENTRY,
                "groupbegin" => SPLIT_GROUP_BEGIN_ENTRY,
                "grouplevels" => SPLIT_GROUP_LEVELS_ENTRY,
                "grouplv" => SPLIT_GROUP_LV_ENTRY,
                "wheels" => SPLIT_WHEELS_ENTRY,
                "edges" => SPLIT_EDGE_ENTRY,
                "fleetfront" => "explicit_beams_fleet_front",
                "fleetplan" => "explicit_beams_fleet_plan",
                "fleetpost" => "explicit_beams_fleet_post",
                "fleetlevels" => "explicit_beams_fleet_levels",
                _ => SPLIT_TESTS_ENTRY,
            };
            push(format!("explicit_beams_split_{which}{tag}"), entry, d);
        }
        // Self-collision's listed pairs across workgroups: a solo frame a step a
        // dispatch, cut at its tests, and the pairs dispatch. (Without surface contact:
        // these frames run only without it, and the module compiles as the frame's
        // `_nosurf` twin does.)
        for (which, entry) in [("step", SELF_STEP_ENTRY), ("walk", SELF_WALK_ENTRY), ("pairs", SELF_PAIRS_ENTRY)] {
            push(format!("explicit_beams_selfsplit_{which}{tag}"), entry, self_split_defines(aero, lanes));
        }
        // The wide passes' reused pair list (the list's part of a step alone, a small
        // kernel); wide fleets have no aero.
        if !aero {
            push(
                format!("explicit_beams_selfsplit_list{tag}"),
                WIDE_LIST_ENTRY,
                vec![W, lanes, ("EB_SOLO_STEP", "1"), ("EB_SELF_SPLIT", "1"), NOSURF],
            );
        }
        // With vehicle contact (EB_VV_SOLO): the vehicles' dispatch is the split group
        // path's members' (a solo vehicle's step cut at its tests, a member's step
        // whole), and the walk and pairs skip the members.
        let mut base = vec![W, lanes, SOLO, ("EB_SOLO_STEP", "1"), ("EB_SELF_SPLIT", "1"), NOSURF];
        if aero {
            base.push(AERO);
        }
        let mut member = base.clone();
        member.push(("EB_VV_SPLIT", "1"));
        push(format!("explicit_beams_selfsplit_member_vv{tag}"), SPLIT_MEMBER_ENTRY, member);
        push(format!("explicit_beams_selfsplit_walk_vv{tag}"), SELF_WALK_ENTRY, base.clone());
        // Surface contact on this path (the pass as pipelines): the solo vehicles' rest.
        push(format!("explicit_beams_selfsplit_rest_vv{tag}"), SOLO_REST_ENTRY, base.clone());
        push(format!("explicit_beams_selfsplit_pairs_vv{tag}"), SELF_PAIRS_ENTRY, base);
    }
    out
}

/// The self-collision split's defines (without vehicle contact) at a frame variant;
/// a host's measurement builds (EB_ABLATE) add to them.
pub fn self_split_defines(aero: bool, lanes: (&'static str, &'static str)) -> Vec<(&'static str, &'static str)> {
    let mut d = vec![W, lanes, ("EB_SOLO_STEP", "1"), ("EB_SELF_SPLIT", "1"), NOSURF];
    if aero {
        d.push(AERO);
    }
    d
}

/// `VARIANTS` for a host's own builds of split modules: (tag, aero, lanes define).
pub fn split_variants() -> [(&'static str, bool, (&'static str, &'static str)); 4] {
    VARIANTS
}

// --- Test-only modules.

/// Where a test-only module's source is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestSource {
    /// A file under `slang/kernels/`.
    File(&'static str),
    /// The shipped modules' source (`explicit_beams.slang`, or what a measurement build
    /// names: `RUBBLE_EB_SRC`, `RUBBLE_EB_REF=1`).
    Shipped,
}

/// A test-only explicit module: never shipped; compiled when a test or a measurement asks
/// for it (engine-webgpu `test_kernels_dir`), not by the build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestModule {
    pub source: TestSource,
    pub name: String,
    pub entry: String,
    pub defines: Vec<(&'static str, &'static str)>,
}

/// Lane counts of the test builds of the frame modules.
const TEST_LANES: [&str; 2] = ["256", "512"];

/// A module name's lane tag: 256 none, 512 `_wide`, else `_l<lanes>`.
fn lane_tag(lanes: &str) -> String {
    match lanes {
        "256" => String::new(),
        "512" => "_wide".into(),
        l => format!("_l{l}"),
    }
}

/// Every test-only explicit module: the self split's ablation builds (EB_ABLATE, the
/// non-aero variants); the frozen baseline (`explicit_beams_baseline.slang`) at every
/// frame variant and plan pass, with its race-probe and deterministic-hits builds; the
/// current kernel's deterministic-hits (EB_DET_HITS) and ablation (EB_ABLATE) frame
/// builds; and the frozen reference (`explicit_beams_ref.slang`).
pub fn test_modules() -> Vec<TestModule> {
    let mut out = Vec::new();
    for (tag, aero, lanes) in VARIANTS {
        if aero {
            continue;
        }
        for (which, fun) in [("step", SELF_STEP_ENTRY), ("walk", SELF_WALK_ENTRY), ("pairs", SELF_PAIRS_ENTRY)] {
            let mut d = self_split_defines(aero, lanes);
            d.push(("EB_ABLATE", "1"));
            if which != "step" {
                d.push(("EB_SELF_PAIRS_ABLATE", "1"));
            }
            out.push(TestModule {
                source: TestSource::Shipped,
                name: format!("explicit_beams_selfsplit_{which}_ablate{tag}"),
                entry: fun.to_string(),
                defines: d,
            });
        }
    }
    let base = TestSource::File("explicit_beams_baseline.slang");
    let cur = TestSource::File("explicit_beams.slang");
    let reference = TestSource::File("explicit_beams_ref.slang");
    for p in ["bounds", "pairs", "group"] {
        let n = format!("explicit_beams_vv_{p}");
        for (source, prefix) in [(base, "explicit_baseline_"), (reference, "explicit_ref_")] {
            out.push(TestModule { source, name: format!("{prefix}{n}"), entry: n.clone(), defines: vec![W] });
        }
    }
    // (source, file prefix, lane counts, extra define)
    let sets: [(TestSource, &str, &[&str], Option<&'static str>); 6] = [
        (base, "explicit_baseline_", &TEST_LANES, None),
        (base, "explicit_baseline_", &TEST_LANES, Some("EB_RACE_PROBE")),
        (base, "explicit_baseline_", &TEST_LANES, Some("EB_DET_HITS")),
        (cur, "explicit_det_", &TEST_LANES, Some("EB_DET_HITS")),
        (cur, "explicit_ablate_", &TEST_LANES, Some("EB_ABLATE")),
        // The frozen reference: the kernel before the latency work, which the current
        // kernel must match bit for bit.
        (reference, "explicit_ref_", &TEST_LANES, None),
    ];
    for (source, prefix, lanes_list, extra) in sets {
        for &lanes in lanes_list {
            let lanes: &'static str = if lanes == "256" { "256" } else { "512" };
            for aero in [false, true] {
                let tag = format!(
                    "{}{}{}",
                    if aero { "_aero" } else { "" },
                    lane_tag(lanes),
                    match extra {
                        Some("EB_RACE_PROBE") => "_probe",
                        Some("EB_DET_HITS") if prefix == "explicit_baseline_" => "_det",
                        _ => "",
                    }
                );
                let mut d = vec![W, ("EB_LANES", lanes)];
                if aero {
                    d.push(AERO);
                }
                if let Some(x) = extra {
                    d.push((x, "1"));
                }
                // The current kernel's deterministic-hits build serves the bitwise suite's
                // fleets, which have no surface contact: without the surface pass, as the
                // modules those fleets ship with (native Metal runs out of threadgroup
                // slots in a contact module with it); the ablation builds likewise (Metal
                // binds at most 31 workgroup variables).
                if prefix == "explicit_det_" || prefix == "explicit_ablate_" {
                    d.push(NOSURF);
                }
                let entry = FRAME_ENTRY.to_string();
                out.push(TestModule { source, name: format!("{prefix}frame{tag}"), entry: entry.clone(), defines: d.clone() });
                d.push(SOLO);
                out.push(TestModule { source, name: format!("{prefix}solo{tag}"), entry: entry.clone(), defines: d.clone() });
                d.push(GROUPS);
                out.push(TestModule { source, name: format!("{prefix}contact{tag}"), entry, defines: d });
            }
        }
    }
    out
}

/// The surface pass's pipelines of the fast schedules, in dispatch order: (module,
/// entry). The prep sizes the others' dispatches (`EB_SURF_ARGS`: binding 10, the
/// surface arguments), each of which runs a thread an element (`EB_SURF_SPREAD`).
pub const SURF_FAST_PASSES: [(&str, &str); 11] = [
    ("explicit_surface_prep_args", "explicit_surface_prep"),
    // The pair path's listing, a thread an element, then its decision, a workgroup a
    // vehicle (binding 10 as the prep).
    ("explicit_surface_cull_x", "explicit_surface_cull"),
    ("explicit_surface_plan", "explicit_surface_plan"),
    ("explicit_surface_find_tri_x", "explicit_surface_find"),
    ("explicit_surface_find_seg_x", "explicit_surface_find"),
    ("explicit_surface_find_point_x", "explicit_surface_find"),
    ("explicit_surface_find_pair_x", "explicit_surface_find"),
    ("explicit_surface_apply_tri_x", "explicit_surface_apply"),
    ("explicit_surface_apply_seg_x", "explicit_surface_apply"),
    ("explicit_surface_apply_point_x", "explicit_surface_apply"),
    ("explicit_surface_apply_pair_x", "explicit_surface_apply"),
];

/// [`SURF_FAST_PASSES`] with their defines, for every host's build.
pub fn surface_fast_modules() -> Vec<SplitModule> {
    let mut out = vec![
        SplitModule {
            name: SURF_FAST_PASSES[0].0.to_string(),
            entry: SURF_FAST_PASSES[0].1,
            defines: vec![W, ("EB_SURF_PIPE", "1"), ("EB_SURF_ARGS", "1")],
        },
        SplitModule {
            name: SURF_FAST_PASSES[1].0.to_string(),
            entry: SURF_FAST_PASSES[1].1,
            defines: vec![W, ("EB_SURF_PIPE", "1"), ("EB_SURF_KIND", "5"), ("EB_SURF_SPREAD", "1"), ("EB_CPL_DIRECT", "1")],
        },
        SplitModule {
            name: SURF_FAST_PASSES[2].0.to_string(),
            entry: SURF_FAST_PASSES[2].1,
            defines: vec![W, ("EB_SURF_PIPE", "1"), ("EB_SURF_KIND", "6"), ("EB_SURF_ARGS", "1")],
        },
    ];
    for (i, kind) in ["1", "2", "3", "4", "1", "2", "3", "4"].into_iter().enumerate() {
        let mut defines = vec![W, ("EB_SURF_PIPE", "1"), ("EB_SURF_KIND", kind), ("EB_SURF_SPREAD", "1"), ("EB_CPL_DIRECT", "1")];
        // The pair path's find: its hulls' records in workgroup memory
        // (`explicit_beams_surface_pool.slang`).
        if i == 3 {
            defines.push(("EB_SURF_POOL", "1"));
        }
        out.push(SplitModule {
            name: SURF_FAST_PASSES[3 + i].0.to_string(),
            entry: SURF_FAST_PASSES[3 + i].1,
            defines,
        });
    }
    out
}

/// A fleet with vehicle contact on the wide passes while the contact plan holds no
/// group (`slang/kernels/explicit_beams_vvwide.slang`, `EB_VV_WIDE`;
/// `engine_exec::explicit_plan::vv_wide`): (module, entry) of the gate that sizes the
/// frame's two schedules on the GPU, and of the node and close passes with the solo
/// vehicles' frame-bounds check.
pub const VV_GATE: (&str, &str) = ("explicit_vv_gate", "explicit_vv_gate");
pub const VV_WIDE_NODES: (&str, &str) = ("explicit_beams_wide_nodes_vv", "explicit_beams_wide_nodes_vv");
pub const VV_WIDE_CLOSE: (&str, &str) = ("explicit_beams_wide_close_vv", "explicit_beams_wide_close_vv");

/// [`VV_GATE`], [`VV_WIDE_NODES`] and [`VV_WIDE_CLOSE`] with their defines, for every
/// host's build.
pub fn vv_wide_modules() -> Vec<SplitModule> {
    [VV_GATE, VV_WIDE_NODES, VV_WIDE_CLOSE]
        .into_iter()
        .map(|(name, entry)| SplitModule { name: name.to_string(), entry, defines: vec![W, ("EB_VV_WIDE", "1")] })
        .collect()
}

/// [`surface_fast_modules`] then [`vv_wide_modules`]: the fast schedules' small modules.
pub fn fast_schedule_modules() -> Vec<SplitModule> {
    surface_fast_modules().into_iter().chain(vv_wide_modules()).collect()
}

/// The modules of a fleet with surface contact on the frame modules, split
/// (`Kernel::FrameSeg`, `Kernel::Surf`): the frame in segments at its four variants
/// (`EB_FRAME_SEG`), then the surface pass's pipelines (`EB_SURF_PIPE`: the prep, then
/// per kind its find and its apply). The same list engine-webgpu's build compiles.
pub fn surface_split_modules() -> Vec<SplitModule> {
    let mut out = Vec::new();
    for (name, defines) in [
        ("explicit_beams_seg", vec![W]),
        ("explicit_beams_wide_seg", vec![W, L512]),
        ("explicit_beams_aero_seg", vec![W, L256, AERO]),
        ("explicit_beams_aero_wide_seg", vec![W, L512, AERO]),
    ] {
        let mut defines = defines;
        defines.push(("EB_FRAME_SEG", "1"));
        out.push(SplitModule { name: name.to_string(), entry: FRAME_ENTRY, defines });
    }
    out.push(SplitModule {
        name: "explicit_surface_prep".to_string(),
        entry: "explicit_surface_prep",
        defines: vec![W, ("EB_SURF_PIPE", "1")],
    });
    for (kind, tag) in [("1", "tri"), ("2", "seg"), ("3", "point"), ("4", "pair")] {
        for (phase, entry) in [("find", "explicit_surface_find"), ("apply", "explicit_surface_apply")] {
            out.push(SplitModule {
                name: format!("explicit_surface_{phase}_{tag}"),
                entry,
                defines: vec![W, ("EB_SURF_PIPE", "1"), ("EB_SURF_KIND", kind)],
            });
        }
    }
    out
}
