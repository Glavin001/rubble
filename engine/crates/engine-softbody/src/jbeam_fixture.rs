//! An original JBeam vehicle that uses the JBeam semantics the explicit solver's
//! JBeam mode simulates (`engine/docs/jbeam-semantics-spec.md`), for tests: a box
//! body with a door on its left side hung on two hinge break groups, held shut by
//! a coupler latch, with a passive pop-open support, a bounded detent, a bounded
//! limiter with fast damping and a damping cutoff, precompressed support beams in
//! their own break group, deform limits, a damper-only beam, a torsion bar about the
//! hinge, and self-collision groups (body and door). Written for Rubble; no third-party content.

/// The fixture's JBeam text (one file, main part `jbm_box`).
pub fn latched_box_jbeam() -> String {
    // JBeam frame: +X left, +Y rear, +Z up.
    r#"{
"jbm_box": {
    "information": {"name": "JBeam-mode latched box", "authors": "Rubble"},
    "slotType": "main",
    "controller": [["fileName"], ["advancedCouplerControl", {"name": "doorCoupler"}]],
    "doorCoupler": {
        "groupType": "autoCoupling",
        "couplerNodes": [
            ["cid1", "cid2", "autoCouplingStrength", "autoCouplingRadius", "autoCouplingLockRadius", "autoCouplingSpeed", "couplingStartRadius", "breakGroup"],
            ["bl", "dl", 20000, 0.01, 0.005, 0.2, 0.1, "door_latch"]
        ]
    },
    "refNodes": [["ref:", "back:", "left:", "up:"], ["b1", "b5", "b2", "b3"]],
    "nodes": [
        ["id", "posX", "posY", "posZ"],
        {"nodeWeight": 20, "selfCollision": true, "collision": true, "group": "body"},
        ["b1", -0.6, -1.0, 0.1], ["b2", 0.6, -1.0, 0.1], ["b3", -0.6, -1.0, 0.7], ["b4", 0.6, -1.0, 0.7],
        ["b5", -0.6, 1.0, 0.1], ["b6", 0.6, 1.0, 0.1], ["b7", -0.6, 1.0, 0.7], ["b8", 0.6, 1.0, 0.7],
        {"nodeWeight": 4, "selfCollision": false, "collision": false},
        ["bl", 0.66, 0.5, 0.4],
        ["bh1", 0.66, -0.5, 0.2], ["bh2", 0.66, -0.5, 0.6],
        {"nodeWeight": 3, "selfCollision": true, "collision": true, "group": "door"},
        ["d1", 0.66, -0.45, 0.2], ["d2", 0.66, -0.45, 0.6], ["d3", 0.66, 0.5, 0.2], ["d4", 0.66, 0.5, 0.6],
        {"nodeWeight": 1, "selfCollision": false, "collision": false},
        ["dl", 0.66, 0.5, 0.4],
        ["dr", 0.5, 0.0, 0.4],
        {"group": ""}
    ],
    "beams": [
        ["id1:", "id2:"],
        {"beamPrecompression": 1, "beamType": "|NORMAL", "beamLongBound": 1, "beamShortBound": 1},
        {"beamSpring": 2001000, "beamDamp": 200, "beamDeform": 60000, "beamStrength": 400000},
        ["b1", "b2"], ["b3", "b4"], ["b5", "b6"], ["b7", "b8"], ["b1", "b3"], ["b2", "b4"],
        ["b5", "b7"], ["b6", "b8"], ["b1", "b5"], ["b2", "b6"], ["b3", "b7"], ["b4", "b8"],
        ["b1", "b4"], ["b2", "b3"], ["b5", "b8"], ["b6", "b7"], ["b1", "b7"], ["b3", "b5"],
        ["b2", "b8"], ["b4", "b6"], ["b1", "b6"], ["b2", "b5"], ["b3", "b8"], ["b4", "b7"],
        ["b1", "b8"], ["b2", "b7"], ["b3", "b6"], ["b4", "b5"],
        {"beamSpring": 1001000, "beamDamp": 100, "beamDeform": 40000, "beamStrength": "FLT_MAX"},
        ["bl", "b6"], ["bl", "b8"], ["bl", "b2"], ["bl", "b4"], ["bl", "b5"],
        ["bh1", "b2"], ["bh1", "b4"], ["bh1", "b6"], ["bh1", "b1"], ["bh1", "b3"],
        ["bh2", "b2"], ["bh2", "b4"], ["bh2", "b8"], ["bh2", "b3"], ["bh2", "bh1"],
        //door shape, with deform limits
        {"beamSpring": 801000, "beamDamp": 60, "beamDeform": 9000, "beamStrength": "FLT_MAX"},
        {"deformLimitExpansion": 0.1, "deformLimit": 0.3},
        ["d1", "d2"], ["d3", "d4"], ["d1", "d3"], ["d2", "d4"], ["d1", "d4"], ["d2", "d3"],
        ["dr", "d1"], ["dr", "d2"], ["dr", "d3"], ["dr", "d4"],
        {"deformLimitExpansion": 0.01},
        ["dl", "d3"], ["dl", "d4"], ["dl", "dr"],
        {"deformLimitExpansion": "", "deformLimit": ""},
        //hinges
        {"beamSpring": 1301000, "beamDamp": 70, "beamDeform": 9800, "beamStrength": 30000},
        {"breakGroup": "hinge_a"},
        ["d1", "bh1"], ["d1", "bh2"], ["dr", "bh1"],
        {"breakGroup": "hinge_b"},
        ["d2", "bh2"], ["d2", "bh1"], ["dr", "bh2"],
        {"breakGroup": ""},
        //pop open: passive in hinge_a
        {"beamPrecompression": 1.02, "beamSpring": 10000, "beamDamp": 600, "beamDeform": 40000, "beamStrength": 10000, "breakGroupType": 1, "breakGroup": "hinge_a", "beamType": "|SUPPORT", "beamLongBound": 25},
        ["b2", "d4"],
        //detent: no force past its bounds
        {"beamPrecompression": 1.5, "beamLongBound": 0.1, "beamShortBound": 0.1, "beamSpring": 800, "beamDamp": 200, "beamStrength": 8000, "boundZone": 0.01, "beamLimitSpring": 0, "beamLimitDamp": 0, "beamType": "|BOUNDED"},
        ["d3", "bh1"],
        //limiter with fast damping and a damping cutoff
        {"beamPrecompression": 1, "beamLongBound": 0.5, "beamShortBound": 1, "beamSpring": 0, "beamDamp": 5, "beamLimitSpring": 500000, "beamLimitDamp": 500, "beamStrength": 10000, "boundZone": "", "beamDampFast": 50, "beamDampVelocitySplit": 0.1, "dampCutoffHz": 500},
        ["d3", "bh2"],
        {"beamLimitSpring": "", "beamLimitDamp": "", "beamDampFast": "", "beamDampVelocitySplit": "", "dampCutoffHz": "", "beamType": "|NORMAL", "beamLongBound": 1, "beamShortBound": 1, "breakGroup": "", "breakGroupType": 0, "beamPrecompression": 1},
        //door supports, precompressed, in their own group
        {"beamType": "|SUPPORT", "beamLongBound": 6, "beamPrecompression": 0.98},
        {"beamSpring": 1001000, "beamDamp": 50, "beamDeform": 8000, "beamStrength": 200000},
        {"breakGroup": "door_supports"},
        ["d1", "b2"], ["d3", "b6"], ["d2", "b4"], ["d4", "b8"],
        {"breakGroup": "", "beamType": "|NORMAL", "beamPrecompression": 1, "beamLongBound": 1},
        //damper only, with a cutoff
        {"beamSpring": 0, "beamDamp": 500, "beamDeform": 27600, "beamStrength": 75000, "dampCutoffHz": 1000},
        ["b1", "b8"], ["b2", "b7"],
        {"dampCutoffHz": ""}
    ],
    "torsionbars": [
        ["id1:", "id2:", "id3:", "id4:"],
        {"spring": 20000, "damp": 5, "deform": 2000, "strength": "FLT_MAX"},
        ["d3", "bh1", "bh2", "b4"]
    ],
    "triangles": [
        ["id1:", "id2:", "id3:"],
        {"group": "body"},
        ["b2", "b6", "b8"], ["b2", "b8", "b4"],
        ["b1", "b3", "b7"], ["b1", "b7", "b5"],
        ["b3", "b4", "b8"], ["b3", "b8", "b7"],
        ["b1", "b5", "b6"], ["b1", "b6", "b2"],
        ["b1", "b2", "b4"], ["b1", "b4", "b3"],
        ["b5", "b7", "b8"], ["b5", "b8", "b6"],
        {"group": "door"},
        ["d1", "d3", "d4"], ["d1", "d4", "d2"],
        {"group": ""}
    ]
}
}"#
    .to_string()
}

// Part II fixtures (jbeam-semantics-spec.md §16–§27): original Rubble content for
// the CPU tests (tests/explicit_jbeam_ext.rs) and the GPU parity tests.

/// A ladder of nodes held up by fixed nodes (no gravity effects on the tests that
/// pull it apart by hand).
pub const PART2_GROUPS: &str = r#"
{ "t": {
  "information": {"name": "groups"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false},
            ["a0", 0, 0, 0], ["a1", 1, 0, 0], ["b0", 0, 1, 0], ["b1", 1, 1, 0], ["c0", 0, 2, 0], ["c1", 1, 2, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 100000, "beamDamp": 0, "beamDeform": "FLT_MAX", "beamStrength": 1000},
            {"breakGroup": "g1"}, ["a0", "a1"],
            {"breakGroup": ["g1", "g2"]}, ["b0", "b1"],
            {"breakGroup": "g2"}, ["c0", "c1"],
            {"breakGroup": ""}, ["a0", "b0"], ["b0", "c0"], ["a1", "b1"], ["b1", "c1"]]
}}"#;

pub const PART2_PANEL: &str = r#"
{ "t": {
  "information": {"name": "panel"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false},
            ["p0", 0, 0, 0], ["p1", 1, 0, 0], ["p2", 0, 1, 0], ["q0", 2, 0, 0], ["q1", 3, 0, 0], ["q2", 2, 1, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 100000, "beamDamp": 0, "beamDeform": 500, "beamStrength": 1000,
             "deformGroup": "glass", "deformationTriggerRatio": 0.002},
            ["p0", "p1"], ["p1", "p2"], ["p2", "p0"],
            {"deformGroup": "", "disableTriangleBreaking": true},
            ["q0", "q1"], ["q1", "q2"], ["q2", "q0"]],
  "triangles": [["id1:", "id2:", "id3:"], ["p0", "p1", "p2"], ["q0", "q1", "q2"]]
}}"#;

pub const PART2_RAMP: &str = r#"
{ "t": {
  "information": {"name": "ramp"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false, "fixed": true},
            ["a", 0, 0, 0], ["b", 1, 0, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 1000, "beamDamp": 0, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX",
             "beamPrecompression": 0.8, "beamPrecompressionTime": 0.01}, ["a", "b"]]
}}"#;

/// A box chassis on four pressure wheels (original Rubble content).
pub const PART2_CAR: &str = r#"
{ "t": {
  "information": {"name": "pw car"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 40, "frictionCoef": 1.0},
            ["fl", 0.7, -1.2, 0.45], ["fr", -0.7, -1.2, 0.45], ["rl", 0.7, 1.2, 0.45], ["rr", -0.7, 1.2, 0.45],
            ["flt", 0.5, -1.2, 0.9], ["frt", -0.5, -1.2, 0.9], ["rlt", 0.5, 1.2, 0.9], ["rrt", -0.5, 1.2, 0.9],
            {"nodeWeight": 8},
            ["fli", 0.55, -1.2, 0.45], ["fri", -0.55, -1.2, 0.45], ["rli", 0.55, 1.2, 0.45], ["rri", -0.55, 1.2, 0.45],
            ["flo", 0.85, -1.2, 0.45], ["fro", -0.85, -1.2, 0.45], ["rlo", 0.85, 1.2, 0.45], ["rro", -0.85, 1.2, 0.45]],
  "beams": [["id1:", "id2:"], {"beamSpring": 2000000, "beamDamp": 200, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"},
            ["fl", "fr"], ["rl", "rr"], ["fl", "rl"], ["fr", "rr"], ["fl", "rr"], ["fr", "rl"],
            ["flt", "frt"], ["rlt", "rrt"], ["flt", "rlt"], ["frt", "rrt"], ["flt", "rrt"],
            ["fl", "flt"], ["fr", "frt"], ["rl", "rlt"], ["rr", "rrt"], ["fl", "frt"], ["rl", "rrt"], ["fl", "rlt"], ["fr", "rrt"],
            ["fli", "flo"], ["fri", "fro"], ["rli", "rlo"], ["rri", "rro"],
            ["fli", "fl"], ["flo", "fl"], ["fli", "flt"], ["flo", "flt"], ["fli", "rl"], ["flo", "rl"],
            ["fri", "fr"], ["fro", "fr"], ["fri", "frt"], ["fro", "frt"], ["fri", "rr"], ["fro", "rr"],
            ["rli", "rl"], ["rlo", "rl"], ["rli", "rlt"], ["rlo", "rlt"], ["rli", "fl"], ["rlo", "fl"],
            ["rri", "rr"], ["rro", "rr"], ["rri", "rrt"], ["rro", "rrt"], ["rri", "fr"], ["rro", "fr"]],
  "pressureWheels": [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"],
        {"numRays": 12, "radius": 0.32, "hubRadius": 0.2, "tireWidth": 0.2, "hubWidth": 0.22,
         "hubNodeWeight": 1.0, "nodeWeight": 0.25, "pressurePSI": 30, "frictionCoef": 1.0,
         "hubBeamSpring": 800000, "hubBeamDamp": 60, "hubSideBeamSpring": 800000, "hubSideBeamDamp": 60,
         "hubTreadBeamSpring": 800000, "hubTreadBeamDamp": 60, "hubPeripheryBeamSpring": 800000, "hubPeripheryBeamDamp": 60,
         "wheelSideBeamSpring": 20000, "wheelSideBeamDamp": 150, "wheelSideBeamSpringExpansion": 200000,
         "wheelSideBeamDampExpansion": 150, "wheelSideTransitionZone": 0.05,
         "wheelTreadBeamSpring": 200000, "wheelTreadBeamDamp": 30, "wheelPeripheryBeamSpring": 200000,
         "wheelPeripheryBeamDamp": 30, "enableTireLbeams": true, "wheelLbeamSpring": 5000, "wheelLbeamDamp": 5,
         "enableTreadReinfBeams": true, "wheelTreadReinfBeamSpring": 100000, "wheelTreadReinfBeamDamp": 20},
        ["FL", "wheel_FL", "tire_FL", "fli", "flo", 9999, "flt", 1],
        ["FR", "wheel_FR", "tire_FR", "fri", "fro", 9999, "frt", -1],
        ["RL", "wheel_RL", "tire_RL", "rli", "rlo", 9999, "rlt", 1],
        ["RR", "wheel_RR", "tire_RR", "rri", "rro", 9999, "rrt", -1]]
}}"#;

pub const PART2_RAIL: &str = r#"
{ "t": {
  "information": {"name": "rail"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false},
            ["r0", -1, 0, 0, {"fixed": true}], ["r1", 1, 0, 0, {"fixed": true}], ["s", 0, 0, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 1000, "beamDamp": 0}, ["r0", "r1"]],
  "rails": {"guide": {"links:": ["r0", "r1"], "looped": false, "capped": true}},
  "slidenodes": [["id:", "railName", "attached", "fixToRail", "tolerance", "spring", "strength", "capStrength"],
                 ["s", "guide", true, true, 0, 100000, "FLT_MAX", "FLT_MAX"]]
}}"#;

/// Node couplers (`jbeam-semantics-spec.md` §7.1): a lid held on a body by BeamNG
/// node couplers (a node's `couplerTag` coupling to a node with that `tag`), written
/// for Rubble. `lc1`: an active latch (strength the lower of the two nodes', 2,500 N)
/// in group `latch_a`, which also holds the beam l2–b4; `lc2`: a passive coupler
/// (type 1) with no strength of its own, in the two hinge groups, one held by the
/// hinge beam l1–b4 (strength 100 N); `lc3`: no `couplerStartRadius` (not Active
/// at spawn: left uncoupled); `lc4`: no tag node. Both parts are tetrahedra.
pub const NODE_COUPLERS: &str = r#"{
"nc_box": {
    "information": {"name": "Node-coupler box", "authors": "Rubble"},
    "slotType": "main",
    "nodes": [
        ["id", "posX", "posY", "posZ"],
        {"nodeWeight": 5, "collision": false, "selfCollision": false},
        ["b1", 0.0, 0.0, 1.0], ["b2", 0.5, 0.0, 1.0], ["b3", 0.0, 0.5, 1.0], ["b4", 0.25, 0.25, 1.4],
        {"nodeWeight": 2},
        ["bc1", 0.4, 0.0, 1.3, {"tag": "latch", "couplerStrength": 2500}],
        ["bc2", 0.0, 0.4, 1.3, {"tag": "hinge"}],
        ["bc3", 0.2, 0.2, 1.3, {"tag": "toggle"}],
        {"nodeWeight": 3},
        ["l1", 0.0, 0.0, 1.6], ["l2", 0.5, 0.0, 1.6], ["l3", 0.0, 0.5, 1.6], ["l4", 0.25, 0.25, 2.0],
        {"nodeWeight": 2},
        ["lc1", 0.4, 0.0, 1.3, {"couplerTag": "latch", "couplerStrength": 3000, "couplerStartRadius": 0.01, "couplerLock": true, "breakGroup": "latch_a"}],
        ["lc2", 0.0, 0.4, 1.3, {"couplerTag": "hinge", "couplerStartRadius": 0.01, "couplerWeld": true, "breakGroupType": 1, "breakGroup": ["hinge_l", "hinge_r"]}],
        ["lc3", 0.2, 0.2, 1.3, {"couplerTag": "toggle", "couplerRadius": 0.5, "couplerStrength": 1000}],
        ["lc4", 0.1, 0.1, 1.3, {"couplerTag": "absent", "couplerStartRadius": 0.1}]
    ],
    "beams": [
        ["id1:", "id2:"],
        {"beamSpring": 1001000, "beamDamp": 100, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"},
        ["b1", "b2"], ["b1", "b3"], ["b1", "b4"], ["b2", "b3"], ["b2", "b4"], ["b3", "b4"],
        ["bc1", "b1"], ["bc1", "b2"], ["bc1", "b3"], ["bc1", "b4"],
        ["bc2", "b1"], ["bc2", "b2"], ["bc2", "b3"], ["bc2", "b4"],
        ["bc3", "b1"], ["bc3", "b2"], ["bc3", "b3"], ["bc3", "b4"],
        ["l1", "l2"], ["l1", "l3"], ["l1", "l4"], ["l2", "l3"], ["l2", "l4"], ["l3", "l4"],
        ["lc1", "l1"], ["lc1", "l2"], ["lc1", "l3"], ["lc1", "l4"],
        ["lc2", "l1"], ["lc2", "l2"], ["lc2", "l3"], ["lc2", "l4"],
        ["lc3", "l1"], ["lc3", "l2"], ["lc3", "l3"], ["lc3", "l4"],
        ["lc4", "l1"], ["lc4", "l2"], ["lc4", "l3"], ["lc4", "l4"],
        {"beamSpring": 101000, "beamDamp": 20},
        {"breakGroup": "latch_a"},
        ["l2", "b4"],
        {"breakGroup": "hinge_l", "beamStrength": 100},
        ["l1", "b4"],
        {"breakGroup": "", "beamStrength": "FLT_MAX"}
    ]
}
}"#;
