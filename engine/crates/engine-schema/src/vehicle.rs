//! Vehicle record layout in `gVehicleData` (float4 slots). Mirrors
//! `slang/physics/vehicle_layout.slang`; `layout_matches_slang` keeps them equal.

#![allow(missing_docs)]

pub const VH_COUNTS: u32 = 0; // uint4(maxVehicles, maxWheels, vehicleCount, wheelCount)
pub const VH_BASES: u32 = 1; // uint4(vehicleBase, wheelBase, curveBase, aabbTail)
pub const VH_POOLS: u32 = 2; // uint4(diffBase, antiRollBase, partBase, rayBase)
pub const VH_MISC: u32 = 3; // uint4(rayCount, maxRays, partCount, queryMode: 0 grid walk, 1 full scan)
pub const VH_SLOTS: u32 = 4;
pub const VEH_STRIDE: u32 = 40;
pub const V_CFG: u32 = 0; // uint4(chassisBody, wheelStart, wheelCount, flags)
pub const V_FWD: u32 = 1; // forwardLocal.xyz, cosMaxPitchRoll (<= -1 disables)
pub const V_UP: u32 = 2; // upLocal.xyz, maxSlopeCos (wheel contacts steeper than this are ignored)
pub const V_ENG0: u32 = 3; // maxTorque, minRPM, maxRPM, inertia
pub const V_ENG1: u32 = 4; // angularDamping, asfloat(torqueCurve), clutchStrength, interDiffLimitedSlip
pub const V_TR0: u32 = 5; // switchTime, clutchReleaseTime, switchLatency, shiftUpRPM
pub const V_TR1: u32 = 6; // shiftDownRPM, asfloat(nForward), asfloat(nReverse), pitchRollStiffness
pub const V_GEARS0: u32 = 7; // gear ratios 0..3 (forward first, then reverse as negatives)
pub const V_GEARS1: u32 = 8; // gear ratios 4..7
pub const V_AERO0: u32 = 9; // drag (½ρCdA), downforce (½ρClA), gravityScale, asfloat(steerVsSpeedCurve)
pub const V_AERO1: u32 = 10; // aeroCenterLocal.xyz, directMaxTorque (per driven wheel)
pub const V_STEER: u32 = 11; // ackermann [0,1], wheelbase, track, steerRate (input units/s)
pub const V_RESP: u32 = 12; // throttleRate, brakeRate, handbrakeRate, directTopSpeed
pub const V_POOL: u32 = 13; // uint4(diffStart, diffCount, antiRollStart, antiRollCount)
pub const V_POOL2: u32 = 14; // uint4(partStart, partCount, parkedAnchorBody, pad)
pub const V_MASS: u32 = 15; // current chassis mass, Ixx, Iyy, Izz (body axes, about COM)
pub const V_IN0: u32 = 16; // host input: forward [-1,1], steer [-1,1] (+right), brake [0,1], handbrake [0,1]
pub const V_IN1: u32 = 17; // uint4(gearRequest, clutchInput bits, wheelDetachReq, wheelDisableReq)
pub const V_IN2: u32 = 18; // uint4(wheelMaskOverride, driveMaskOverride, steerMaskOverride, brakeMaskOverride)
pub const V_IN3: u32 = 19; // uint4(partDetachReq, partDisableReq, engineOverride (bit0 = connected), wakeSeq)
pub const V_D0: u32 = 20; // engineOmega, gear, clutchFriction, switchTimeLeft
pub const V_D1: u32 = 21; // releaseTimeLeft, latencyTimeLeft, forward (smoothed), brake (smoothed)
pub const V_D2: u32 = 22; // handbrake (smoothed), steer (smoothed), asfloat(wheelMask), asfloat(driveMask)
pub const V_D3: u32 = 23; // uint4(steerMask, status, brakeMask, lastWakeSeq)
pub const V_REFP: u32 = 24; // block reference position.xyz, oldEngineRPM
pub const V_REFQ: u32 = 25; // block reference orientation
pub const V_D4: u32 = 26; // uint4(wheelsDetached, wheelsDisabled, partsDetached, partsDisabled)
pub const V_O0: u32 = 27; // forwardSpeed, engineRPM, gear, asfloat(outFlags)
pub const V_O1: u32 = 28; // lateralSpeed, yawRate, clutchFriction, asfloat(contactCount)
pub const V_O2: u32 = 29; // uint4(wheelMask, driveMask, steerMask, eventSeq)
pub const V_BIKE0: u32 = 32; // maxLean (rad, 0 = off), leanSpring (N·m/rad), leanDamping (N·m·s/rad), leanIntegration
pub const V_BIKE1: u32 = 33; // integrationDecay (1/s), smoothing [0,1), leanSteerLimit (0/1), pad
pub const V_BIKE_D: u32 = 34; // target lean.xyz (world), integrated lean angle
pub const V_TRACK: u32 = 35; // uint4(leftTrackLeader, rightTrackLeader, asfloat(turnRatio), asfloat(pivotSpeed))
pub const V_TRACK_D: u32 = 36; // leftRatio, rightRatio, effective forward, pad
pub const V_QUERY: u32 = 37; // uint4(stepsBetweenQueries (0/1 = every step), asfloat(tractionCut), left pod, right pod: firstCollider | count << POD_COUNT_SHIFT)
pub const V_ASSIST: u32 = 31; // airPitch, airYaw (rad/s² per input), uprightStiffness (1/s²), uprightDamping (1/s)
pub const V_O3: u32 = 30; // uint4(engineConnected, partsLost) , pitch/roll λ, pitch/roll k
pub const VF_ENABLED: u32 = 1;
pub const VF_DIRECT_DRIVE: u32 = 2; // per-wheel torque; no engine/gearbox
pub const VF_MANUAL_TRANS: u32 = 4; // gearRequest selects the gear
pub const VF_CAN_SLEEP: u32 = 8;
pub const VF_REVERSE_BRAKE: u32 = 16; // throttle against travel brakes first
pub const VF_SPHERE_CAST: u32 = 32; // wheels sweep a sphere of their radius (kerbs, edges)
pub const VF_TRACKED: u32 = 64; // two tracks: shared spin per track, left/right drive ratios
pub const VF_TRACTION_CONTROL: u32 = 128; // cut engine torque while a driven wheel spins past peak slip
pub const VS_ENGINE_LOST: u32 = 1;
pub const VS_ACTIVE: u32 = 2; // block built this step
pub const VS_SLIPPING: u32 = 4; // a driven wheel slipped (blocks upshift)
pub const GEAR_AUTO: u32 = 0x7FFFFFFF;
pub const WHEEL_STRIDE: u32 = 48;
pub const W_CFG: u32 = 0; // uint4(vehicle, flags, localIndex, spinGroupLeader (local; NO_INDEX = own))
pub const W_HP: u32 = 1; // hardpointLocal.xyz, radius
pub const W_DIR: u32 = 2; // suspensionDirLocal.xyz (down), width
pub const W_STEERAX: u32 = 3; // steerAxisLocal.xyz, maxSteer (rad)
pub const W_FWD: u32 = 4; // wheelForwardLocal.xyz, ackermannSide (+1 left, -1 right, 0 none)
pub const W_SUSP: u32 = 5; // minLength, maxLength, preload, asfloat(springMode)
pub const W_SPR: u32 = 6; // freqHz | k, zeta | c, bumpFreqHz, bumpZeta
pub const W_INER: u32 = 7; // inertia, angularDamping, maxBrakeTorque, maxHandbrakeTorque
pub const W_TIRE: u32 = 8; // asfloat(longCurve), asfloat(latCurve), longMuScale, latMuScale
pub const W_MISC: u32 = 9; // wheelMass, rollingResistance, bumpMaxForce, rollInfluence
pub const W_PARK: u32 = 10; // uint4(parkedBody, parkedColliderStart, parkedColliderCount, pad)
pub const W_UP: u32 = 11; // wheelUpLocal.xyz (camber), driveMultiplier
pub const W_C0: u32 = 12; // uint4(groundBody, hitCollider, dynFlags, pad)
pub const W_C1: u32 = 13; // lever r.xyz (contact − x⁻ COM), L0 (suspension length at x⁻)
pub const W_C2: u32 = 14; // normal.xyz, invCos
pub const W_C3: u32 = 15; // longitudinal t.xyz, kappa
pub const W_C4: u32 = 16; // lateral s.xyz, alpha (rad)
pub const W_C5: u32 = 17; // ground displacement eg.xyz, steerAngle
pub const W_K: u32 = 18; // K, D/dt, Kbump, Dbump/dt
pub const W_LAM: u32 = 19; // lambdaL, lambdaT, lambdaB, kL
pub const W_K2: u32 = 20; // kT, kB, c0L, c0T
pub const W_Y: u32 = 21; // y, yTilde, mY, tauBrake
pub const W_W: u32 = 22; // omega, angle, tauDrive, mShare
pub const W_F: u32 = 23; // Fn, fL, fT, fSusp (latched, last iteration)
pub const W_BS: u32 = 24; // gY, Ayy, R·kL, contact (1/0) — back-substitution cache
pub const W_HIT: u32 = 25; // hit point world.xyz, hitDistance
pub const W_GV: u32 = 26; // ground point velocity.xyz, pad
pub const W_MU: u32 = 27; // muL (sliding), muT (sliding), muL (static), muT (static)
pub const W_O_POS: u32 = 28; // visual center.xyz, suspension length
pub const W_O_ROT: u32 = 29; // visual orientation
pub const W_O_FM: u32 = 30; // mount force (chassis local).xyz, suspension force
pub const W_O_TM: u32 = 31; // mount torque (chassis local).xyz, longitudinal force
pub const W_O_CP: u32 = 32; // contact point.xyz, lateral force
pub const W_O_ST: u32 = 33; // omega, kappa, alphaDeg, asfloat(outFlags)
pub const W_O_N: u32 = 34; // contact normal.xyz, asfloat(groundBody)
pub const W_LHIT: u32 = 35; // last hit in ground-body space.xyz, valid (1/0) — query-rate extrapolation
pub const W_LN: u32 = 36; // last hit normal in ground-body space.xyz, pad
pub const W_FP: u32 = 37;
pub const W_TJ: u32 = 38; // suspension force point (chassis space).xyz, enabled (1/0)
pub const WF_STEER: u32 = 1;
pub const WF_DRIVEN: u32 = 2;
pub const WF_HANDBRAKE: u32 = 4;
pub const WF_ENABLED: u32 = 8;
pub const WF_RIGID_BUMP: u32 = 16;
pub const WF_BOX_FRICTION: u32 = 32;
pub const WD_CONTACT: u32 = 1;
pub const WD_STICK_L: u32 = 2;
pub const WD_STICK_T: u32 = 4;
pub const WD_DETACHED: u32 = 8;
pub const WD_DISABLED: u32 = 16;
pub const WD_SUSP: u32 = 32;
pub const WD_SLIDE: u32 = 64;
pub const WD_BUMP: u32 = 128;
pub const WD_PENDING_ADD: u32 = 256; // host attached this wheel slot; GPU adds its mass
pub const PART_STRIDE: u32 = 4;
pub const P_CFG: u32 = 0; // uint4(colliderStart, colliderCount, role, roleWheelMask)
pub const P_MASS: u32 = 1; // mass, Ixx, Iyy, Izz (about part COM, chassis axes)
pub const P_COM: u32 = 2; // localCom.xyz, asfloat(state)
pub const P_PARK: u32 = 3; // uint4(parkedBody, vehicle, pad, pad)
pub const PR_BODY: u32 = 0;
pub const PR_ENGINE: u32 = 1;
pub const PR_DRIVESHAFT: u32 = 2;
pub const PR_STEERING: u32 = 3;
pub const PR_BRAKE: u32 = 4;
pub const PS_ATTACHED: u32 = 0;
pub const PS_DETACHED: u32 = 1;
pub const PS_DISABLED: u32 = 2;
pub const PS_LOST: u32 = 3; // colliders moved off the chassis by someone else
pub const DIFF_STRIDE: u32 = 2; // uint4(left, right, -, -) as float bits + (ratio, split) ; (limitedSlip, engineTorqueRatio, -, -)
pub const ANTIROLL_STRIDE: u32 = 1; // asfloat(left), asfloat(right), stiffness, pad
pub const RAY_STRIDE: u32 = 4; // in: origin.xyz,maxDist | dir.xyz,asfloat(excludeBody); out: point.xyz,dist | normal.xyz,asfloat(collider)
pub const NO_INDEX: u32 = 0x7F7FFFFF;
pub const MAX_VEHICLE_WHEELS: u32 = 16;
pub const SHELL_SPHERES: u32 = 5;
/// V_QUERY.zw track pod: first collider | (collider count << POD_COUNT_SHIFT).
pub const POD_COUNT_SHIFT: u32 = 28;
pub const MAX_DRIVEN_WHEELS: u32 = 8;

/// `BodyState::flags` bit for a vehicle chassis (`initial_pad` = asfloat(block index)).
pub const BODY_VEHICLE: u32 = 32;

/// Every layout constant by name (hosts that cannot link this crate export it).
pub const LAYOUT: &[(&str, u32)] = &[
    ("VH_COUNTS", VH_COUNTS),
    ("VH_BASES", VH_BASES),
    ("VH_POOLS", VH_POOLS),
    ("VH_MISC", VH_MISC),
    ("VH_SLOTS", VH_SLOTS),
    ("VEH_STRIDE", VEH_STRIDE),
    ("V_CFG", V_CFG),
    ("V_FWD", V_FWD),
    ("V_UP", V_UP),
    ("V_ENG0", V_ENG0),
    ("V_ENG1", V_ENG1),
    ("V_TR0", V_TR0),
    ("V_TR1", V_TR1),
    ("V_GEARS0", V_GEARS0),
    ("V_GEARS1", V_GEARS1),
    ("V_AERO0", V_AERO0),
    ("V_AERO1", V_AERO1),
    ("V_STEER", V_STEER),
    ("V_RESP", V_RESP),
    ("V_POOL", V_POOL),
    ("V_POOL2", V_POOL2),
    ("V_MASS", V_MASS),
    ("V_IN0", V_IN0),
    ("V_IN1", V_IN1),
    ("V_IN2", V_IN2),
    ("V_IN3", V_IN3),
    ("V_D0", V_D0),
    ("V_D1", V_D1),
    ("V_D2", V_D2),
    ("V_D3", V_D3),
    ("V_REFP", V_REFP),
    ("V_REFQ", V_REFQ),
    ("V_D4", V_D4),
    ("V_O0", V_O0),
    ("V_O1", V_O1),
    ("V_O2", V_O2),
    ("V_BIKE0", V_BIKE0),
    ("V_BIKE1", V_BIKE1),
    ("V_BIKE_D", V_BIKE_D),
    ("V_TRACK", V_TRACK),
    ("V_TRACK_D", V_TRACK_D),
    ("V_QUERY", V_QUERY),
    ("V_ASSIST", V_ASSIST),
    ("V_O3", V_O3),
    ("VF_ENABLED", VF_ENABLED),
    ("VF_DIRECT_DRIVE", VF_DIRECT_DRIVE),
    ("VF_MANUAL_TRANS", VF_MANUAL_TRANS),
    ("VF_CAN_SLEEP", VF_CAN_SLEEP),
    ("VF_REVERSE_BRAKE", VF_REVERSE_BRAKE),
    ("VF_SPHERE_CAST", VF_SPHERE_CAST),
    ("VF_TRACKED", VF_TRACKED),
    ("VF_TRACTION_CONTROL", VF_TRACTION_CONTROL),
    ("VS_ENGINE_LOST", VS_ENGINE_LOST),
    ("VS_ACTIVE", VS_ACTIVE),
    ("VS_SLIPPING", VS_SLIPPING),
    ("GEAR_AUTO", GEAR_AUTO),
    ("WHEEL_STRIDE", WHEEL_STRIDE),
    ("W_CFG", W_CFG),
    ("W_HP", W_HP),
    ("W_DIR", W_DIR),
    ("W_STEERAX", W_STEERAX),
    ("W_FWD", W_FWD),
    ("W_SUSP", W_SUSP),
    ("W_SPR", W_SPR),
    ("W_INER", W_INER),
    ("W_TIRE", W_TIRE),
    ("W_MISC", W_MISC),
    ("W_PARK", W_PARK),
    ("W_UP", W_UP),
    ("W_C0", W_C0),
    ("W_C1", W_C1),
    ("W_C2", W_C2),
    ("W_C3", W_C3),
    ("W_C4", W_C4),
    ("W_C5", W_C5),
    ("W_K", W_K),
    ("W_LAM", W_LAM),
    ("W_K2", W_K2),
    ("W_Y", W_Y),
    ("W_W", W_W),
    ("W_F", W_F),
    ("W_BS", W_BS),
    ("W_HIT", W_HIT),
    ("W_GV", W_GV),
    ("W_MU", W_MU),
    ("W_O_POS", W_O_POS),
    ("W_O_ROT", W_O_ROT),
    ("W_O_FM", W_O_FM),
    ("W_O_TM", W_O_TM),
    ("W_O_CP", W_O_CP),
    ("W_O_ST", W_O_ST),
    ("W_O_N", W_O_N),
    ("W_LHIT", W_LHIT),
    ("W_LN", W_LN),
    ("W_FP", W_FP),
    ("W_TJ", W_TJ),
    ("WF_STEER", WF_STEER),
    ("WF_DRIVEN", WF_DRIVEN),
    ("WF_HANDBRAKE", WF_HANDBRAKE),
    ("WF_ENABLED", WF_ENABLED),
    ("WF_RIGID_BUMP", WF_RIGID_BUMP),
    ("WF_BOX_FRICTION", WF_BOX_FRICTION),
    ("WD_CONTACT", WD_CONTACT),
    ("WD_STICK_L", WD_STICK_L),
    ("WD_STICK_T", WD_STICK_T),
    ("WD_DETACHED", WD_DETACHED),
    ("WD_DISABLED", WD_DISABLED),
    ("WD_SUSP", WD_SUSP),
    ("WD_SLIDE", WD_SLIDE),
    ("WD_BUMP", WD_BUMP),
    ("WD_PENDING_ADD", WD_PENDING_ADD),
    ("PART_STRIDE", PART_STRIDE),
    ("P_CFG", P_CFG),
    ("P_MASS", P_MASS),
    ("P_COM", P_COM),
    ("P_PARK", P_PARK),
    ("PR_BODY", PR_BODY),
    ("PR_ENGINE", PR_ENGINE),
    ("PR_DRIVESHAFT", PR_DRIVESHAFT),
    ("PR_STEERING", PR_STEERING),
    ("PR_BRAKE", PR_BRAKE),
    ("PS_ATTACHED", PS_ATTACHED),
    ("PS_DETACHED", PS_DETACHED),
    ("PS_DISABLED", PS_DISABLED),
    ("PS_LOST", PS_LOST),
    ("DIFF_STRIDE", DIFF_STRIDE),
    ("ANTIROLL_STRIDE", ANTIROLL_STRIDE),
    ("RAY_STRIDE", RAY_STRIDE),
    ("NO_INDEX", NO_INDEX),
    ("MAX_VEHICLE_WHEELS", MAX_VEHICLE_WHEELS),
    ("SHELL_SPHERES", SHELL_SPHERES),
    ("POD_COUNT_SHIFT", POD_COUNT_SHIFT),
    ("MAX_DRIVEN_WHEELS", MAX_DRIVEN_WHEELS),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn slang_consts() -> Vec<(String, u32)> {
        let src = include_str!("../../../slang/physics/vehicle_layout.slang");
        let mut out = Vec::new();
        for line in src.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("static const uint ") else { continue };
            let Some((name, rhs)) = rest.split_once('=') else { continue };
            let lit = rhs.trim().split('u').next().unwrap().trim();
            let v = if let Some(h) = lit.strip_prefix("0x") {
                u32::from_str_radix(h, 16).unwrap()
            } else {
                lit.parse().unwrap()
            };
            out.push((name.trim().to_string(), v));
        }
        out
    }

    #[test]
    fn layout_matches_slang() {
        let consts = slang_consts();
        assert!(consts.len() > 100, "parsed {} constants", consts.len());
        for (name, v) in consts {
            let rust = by_name(&name).unwrap_or_else(|| panic!("{name} missing in vehicle.rs"));
            assert_eq!(rust, v, "{name}: Rust {rust} vs Slang {v}");
        }
    }

    fn by_name(name: &str) -> Option<u32> {
        LAYOUT.iter().find(|(n, _)| *n == name).map(|&(_, v)| v)
    }
}
