//! An original JBeam vehicle that uses the JBeam semantics the explicit solver's
//! JBeam mode simulates (`engine/docs/jbeam-semantics-spec.md`), for tests: a box
//! body with a door on its left side hung on two hinge break groups, held shut by
//! a coupler latch, with a passive pop-open support, a bounded detent, a bounded
//! limiter with fast damping and a damping cutoff, precompressed support beams in
//! their own break group, deform limits, a damper-only beam, and self-collision
//! groups (body and door). Written for Rubble; no third-party content.

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
