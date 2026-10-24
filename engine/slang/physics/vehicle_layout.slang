// Vehicle record layout in gVehicleData (RWStructuredBuffer<float4>), in float4 slots.
// Single source of truth for the GPU side; engine-schema/src/vehicle.rs mirrors these
// numbers (layout_matches_slang) and engine-webgpu/vehicle.mjs reads them from it via wasm.
#ifndef ENGINE_PHYSICS_VEHICLE_LAYOUT_SLANG
#define ENGINE_PHYSICS_VEHICLE_LAYOUT_SLANG

// ---- Header (4 slots) ----
static const uint VH_COUNTS = 0u;  // uint4(maxVehicles, maxWheels, vehicleCount, wheelCount)
static const uint VH_BASES = 1u;   // uint4(vehicleBase, wheelBase, curveBase, aabbTail)
static const uint VH_POOLS = 2u;   // uint4(diffBase, antiRollBase, partBase, rayBase)
static const uint VH_MISC = 3u;    // uint4(rayCount, maxRays, partCount, queryMode: 0 grid walk, 1 full scan)
static const uint VH_SLOTS = 4u;

// ---- Vehicle record ----
static const uint VEH_STRIDE = 44u;
static const uint V_CFG = 0u;      // uint4(chassisBody, wheelStart, wheelCount, flags)
static const uint V_FWD = 1u;      // forwardLocal.xyz, cosMaxPitchRoll (<= -1 disables)
static const uint V_UP = 2u;       // upLocal.xyz, maxSlopeCos (wheel contacts steeper than this are ignored)
static const uint V_ENG0 = 3u;     // maxTorque, minRPM, maxRPM, inertia
static const uint V_ENG1 = 4u;     // angularDamping, asfloat(torqueCurve), clutchStrength, interDiffLimitedSlip
static const uint V_TR0 = 5u;      // switchTime, clutchReleaseTime, switchLatency, shiftUpRPM
static const uint V_TR1 = 6u;      // shiftDownRPM, asfloat(nForward), asfloat(nReverse), pitchRollStiffness
static const uint V_GEARS0 = 7u;   // gear ratios 0..3 (forward first, then reverse as negatives)
static const uint V_GEARS1 = 8u;   // gear ratios 4..7
static const uint V_AERO0 = 9u;    // drag (½ρCdA), downforce (½ρClA), gravityScale, asfloat(steerVsSpeedCurve)
static const uint V_AERO1 = 10u;   // aeroCenterLocal.xyz, directMaxTorque (per driven wheel)
static const uint V_STEER = 11u;   // ackermann [0,1], wheelbase, track, steerRate (input units/s)
static const uint V_RESP = 12u;    // throttleRate, brakeRate, handbrakeRate, directTopSpeed
static const uint V_POOL = 13u;    // uint4(diffStart, diffCount, antiRollStart, antiRollCount)
static const uint V_POOL2 = 14u;   // uint4(partStart, partCount, parkedAnchorBody, pad)
static const uint V_MASS = 15u;    // current chassis mass, Ixx, Iyy, Izz (body axes, about COM)
static const uint V_IN0 = 16u;     // host input: forward [-1,1], steer [-1,1] (+right), brake [0,1], handbrake [0,1]
static const uint V_IN1 = 17u;     // uint4(gearRequest, clutchInput bits, wheelDetachReq, wheelDisableReq)
static const uint V_IN2 = 18u;     // uint4(wheelMaskOverride, driveMaskOverride, steerMaskOverride, brakeMaskOverride)
static const uint V_IN3 = 19u;     // uint4(partDetachReq, partDisableReq, engineOverride (bit0 = connected), wakeSeq)
static const uint V_D0 = 20u;      // engineOmega, gear, clutchFriction, switchTimeLeft
static const uint V_D1 = 21u;      // releaseTimeLeft, latencyTimeLeft, forward (smoothed), brake (smoothed)
static const uint V_D2 = 22u;      // handbrake (smoothed), steer (smoothed), asfloat(wheelMask), asfloat(driveMask)
static const uint V_D3 = 23u;      // uint4(steerMask, status, brakeMask, lastWakeSeq)
static const uint V_REFP = 24u;    // block reference position.xyz, oldEngineRPM
static const uint V_REFQ = 25u;    // block reference orientation
static const uint V_D4 = 26u;      // uint4(wheelsDetached, wheelsDisabled, partsDetached, partsDisabled)
static const uint V_O0 = 27u;      // forwardSpeed, engineRPM, gear, asfloat(outFlags)
static const uint V_O1 = 28u;      // lateralSpeed, yawRate, clutchFriction, asfloat(contactCount)
static const uint V_O2 = 29u;      // uint4(wheelMask, driveMask, steerMask, eventSeq)
static const uint V_BIKE0 = 32u;   // maxLean (rad, 0 = off), leanSpring (N·m/rad), leanDamping (N·m·s/rad), leanIntegration
static const uint V_BIKE1 = 33u;   // integrationDecay (1/s), smoothing [0,1), leanSteerLimit (0/1), pad
static const uint V_BIKE_D = 34u;  // target lean.xyz (world), integrated lean angle
static const uint V_TRACK = 35u;   // uint4(leftTrackLeader, rightTrackLeader, asfloat(steerStrength), pad)
static const uint V_TRACK_D = 36u; // leftRatio, rightRatio, effective forward, turn grip a·W (m²/s², static)
static const uint V_QUERY = 37u;   // uint4(stepsBetweenQueries (0/1 = every step), asfloat(tractionCut, filtered), left pod, right pod)
static const uint V_ASSIST = 31u;  // airPitch, airYaw (rad/s² per input), uprightStiffness (1/s²), uprightDamping (1/s)
static const uint V_O3 = 30u;      // uint4(engineConnected, partsLost) , pitch/roll λ, pitch/roll k
static const uint V_BRK = 38u;     // uint4(partBreakReq, wheelBreakReq, partsBroken, wheelsBroken): mount breaks (vehicle_loads)
static const uint V_V0 = 39u;      // chassis linear velocity at step start.xyz, pad (vehicle_prepare)
static const uint V_W0 = 40u;      // chassis angular velocity at step start.xyz, pad (vehicle_prepare)

// Vehicle flags (V_CFG.w)
static const uint VF_ENABLED = 1u;
static const uint VF_DIRECT_DRIVE = 2u;   // per-wheel torque; no engine/gearbox
static const uint VF_MANUAL_TRANS = 4u;   // gearRequest selects the gear
static const uint VF_CAN_SLEEP = 8u;
static const uint VF_REVERSE_BRAKE = 16u; // throttle against travel brakes first
static const uint VF_SPHERE_CAST = 32u;   // wheels sweep a sphere of their radius (kerbs, edges)
static const uint VF_TRACKED = 64u;       // two tracks: shared spin per track, left/right drive ratios
static const uint VF_TRACTION_CONTROL = 128u; // cut engine torque while a driven wheel spins past peak slip
static const uint VF_BREAKABLE = 256u;    // some part or wheel mount has a finite limit (vehicle_loads)

// Vehicle status (V_D3.y)
static const uint VS_ENGINE_LOST = 1u;
static const uint VS_ACTIVE = 2u;          // block built this step
static const uint VS_SLIPPING = 4u;        // a driven wheel slipped (blocks upshift)

// gearRequest (V_IN1.x): GEAR_AUTO or a signed gear (0 neutral, <0 reverse)
static const uint GEAR_AUTO = 0x7FFFFFFFu;

// ---- Wheel record ----
static const uint WHEEL_STRIDE = 48u;
static const uint W_CFG = 0u;      // uint4(vehicle, flags, localIndex, spinGroupLeader (local; NO_INDEX = own))
static const uint W_HP = 1u;       // hardpointLocal.xyz, radius
static const uint W_DIR = 2u;      // suspensionDirLocal.xyz (down), width
static const uint W_STEERAX = 3u;  // steerAxisLocal.xyz, maxSteer (rad)
static const uint W_FWD = 4u;      // wheelForwardLocal.xyz, ackermannSide (+1 left, -1 right, 0 none)
static const uint W_SUSP = 5u;     // minLength, maxLength, preload, asfloat(springMode)
static const uint W_SPR = 6u;      // freqHz | k, zeta | c, bumpFreqHz, bumpZeta
static const uint W_INER = 7u;     // inertia, angularDamping, maxBrakeTorque, maxHandbrakeTorque
static const uint W_TIRE = 8u;     // asfloat(longCurve), asfloat(latCurve), longMuScale, latMuScale
static const uint W_MISC = 9u;     // wheelMass, rollingResistance, bumpMaxForce, rollInfluence
static const uint W_PARK = 10u;    // uint4(parkedBody, parkedColliderStart, parkedColliderCount, pad)
static const uint W_UP = 11u;      // wheelUpLocal.xyz (camber), driveMultiplier
static const uint W_C0 = 12u;      // uint4(groundBody, hitCollider, dynFlags, pad)
static const uint W_C1 = 13u;      // lever r.xyz (contact − x⁻ COM), L0 (suspension length at x⁻)
static const uint W_C2 = 14u;      // normal.xyz, invCos
static const uint W_C3 = 15u;      // longitudinal t.xyz, kappa
static const uint W_C4 = 16u;      // lateral s.xyz, alpha (rad)
static const uint W_C5 = 17u;      // ground displacement eg.xyz, steerAngle
static const uint W_K = 18u;       // K, D/dt, Kbump, Dbump/dt
static const uint W_LAM = 19u;     // lambdaL, lambdaT, lambdaB, kL
static const uint W_K2 = 20u;      // kT, kB, c0L, c0T
static const uint W_Y = 21u;       // y, yTilde, mY, tauBrake
static const uint W_W = 22u;       // omega, angle, tauDrive, mShare
static const uint W_F = 23u;       // Fn, fL, fT, fSusp (latched, last iteration)
static const uint W_BS = 24u;      // gY, Ayy, R·kL, contact (1/0) — back-substitution cache
static const uint W_HIT = 25u;     // hit point world.xyz, hitDistance
static const uint W_GV = 26u;      // ground point velocity.xyz, pad
static const uint W_MU = 27u;      // muL (sliding), muT (sliding), muL (static), muT (static)
static const uint W_O_POS = 28u;   // visual center.xyz, suspension length
static const uint W_O_ROT = 29u;   // visual orientation
static const uint W_O_FM = 30u;    // mount force (chassis local).xyz, suspension force
static const uint W_O_TM = 31u;    // mount torque (chassis local).xyz, longitudinal force
static const uint W_O_CP = 32u;    // contact point.xyz, lateral force
static const uint W_O_ST = 33u;    // omega, kappa, alphaDeg, asfloat(outFlags)
static const uint W_O_N = 34u;     // contact normal.xyz, asfloat(groundBody)
static const uint W_LHIT = 35u;    // last hit in ground-body space.xyz, valid (1/0) — query-rate extrapolation
static const uint W_LN = 36u;      // last hit normal in ground-body space.xyz, spin servo target y* (rad/step)
static const uint W_FP = 37u;      // suspension force point (chassis space).xyz, enabled (1/0)
static const uint W_TJ = 38u;      // uint4(tyreJoint, shellCollider, twoBodyGround (NO_INDEX = static ground), pad)
static const uint W_MOUNT = 39u;   // breakForce, breakTorque, lastForce, lastTorque (vehicle_loads)
static const uint W_MOUNT2 = 40u;  // mount anchor (chassis space).xyz, peak load ratio

// Wheel config flags (W_CFG.y)
static const uint WF_STEER = 1u;
static const uint WF_DRIVEN = 2u;
static const uint WF_HANDBRAKE = 4u;
static const uint WF_ENABLED = 8u;
static const uint WF_RIGID_BUMP = 16u;
static const uint WF_BOX_FRICTION = 32u; // independent long/lat limits (Jolt) instead of the ellipse

// Wheel dynamic flags (W_C0.z)
static const uint WD_CONTACT = 1u;
static const uint WD_STICK_L = 2u;
static const uint WD_STICK_T = 4u;
static const uint WD_DETACHED = 8u;
static const uint WD_DISABLED = 16u;
static const uint WD_SUSP = 32u;
static const uint WD_SLIDE = 64u;
static const uint WD_BUMP = 128u;
static const uint WD_PENDING_ADD = 256u; // host attached this wheel slot; GPU adds its mass

// ---- Parts (chassis collider groups with a functional role) ----
static const uint PART_STRIDE = 7u;
static const uint P_CFG = 0u;      // uint4(colliderStart, colliderCount, role, roleWheelMask)
static const uint P_MASS = 1u;     // mass, Ixx, Iyy, Izz (about part COM, chassis axes)
static const uint P_COM = 2u;      // localCom.xyz, asfloat(state)
static const uint P_PARK = 3u;     // uint4(parkedBody, vehicle, pad, pad)
static const uint P_MOUNT = 4u;    // mount anchor (chassis space).xyz, breakForce
static const uint P_LOAD = 5u;     // breakTorque, lastForce, lastTorque, peak load ratio (vehicle_loads)
static const uint P_TREE = 6u;     // uint4(parent part (NO_INDEX = chassis), carrier part once carried off (NO_INDEX), pad, pad)

static const uint PR_BODY = 0u;
static const uint PR_ENGINE = 1u;
static const uint PR_DRIVESHAFT = 2u;
static const uint PR_STEERING = 3u;
static const uint PR_BRAKE = 4u;

static const uint PS_ATTACHED = 0u;
static const uint PS_DETACHED = 1u;
static const uint PS_DISABLED = 2u;
static const uint PS_LOST = 3u;     // colliders moved off the chassis by someone else

// ---- Pools ----
static const uint DIFF_STRIDE = 2u;     // uint4(left, right, -, -) as float bits + (ratio, split) ; (limitedSlip, engineTorqueRatio, -, -)
static const uint ANTIROLL_STRIDE = 1u; // asfloat(left), asfloat(right), stiffness, pad
static const uint RAY_STRIDE = 4u;      // in: origin.xyz,maxDist | dir.xyz,asfloat(excludeBody); out: point.xyz,dist | normal.xyz,asfloat(collider)
// Not 0xFFFFFFFF: indices live in float4 slots and an all-ones word is a NaN that
// WGSL constant folding rejects. 0x7F7FFFFF is FLT_MAX bits.
static const uint NO_INDEX = 0x7F7FFFFFu;
static const uint MAX_VEHICLE_WHEELS = 16u;
// Wheel shell: spheres on the chassis around each tyre (not the bottom, which the
// tyre rows own), for sidewall contact with debris, kerb faces and other cars.
static const uint SHELL_SPHERES = 5u;
static const uint POD_COUNT_SHIFT = 28u; // V_QUERY.zw: firstCollider | count << 28
// cos 35°: sphere-cast hits steeper than this belong to the wheel shell.
static const float SWEEP_CONE_COS = 0.819;
static const uint MAX_DRIVEN_WHEELS = 8u;

#endif
