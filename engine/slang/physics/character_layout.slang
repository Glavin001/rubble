// Character / scene-query / projectile pool layout in gCharData
// (RWStructuredBuffer<float4>, binding 33), in float4 slots. Single source of
// truth for the GPU side; engine-schema/src/character.rs mirrors these numbers
// (`layout_matches_slang`).
#ifndef ENGINE_PHYSICS_CHARACTER_LAYOUT_SLANG
#define ENGINE_PHYSICS_CHARACTER_LAYOUT_SLANG

// ---- Header (4 slots) ----
static const uint CH_COUNTS = 0u; // uint4(maxCharacters, characterCount, maxQueries, queryCount)
static const uint CH_BASES = 1u;  // uint4(characterBase, queryBase, projectileBase, maxProjectiles)
static const uint CH_MISC = 2u;   // uint4(projectileCount, queryMode (0 grid walk, 1 full scan), pad, pad)
static const uint CH_SLOTS = 4u;

// ---- Character record ----
static const uint CHAR_STRIDE = 16u;
static const uint C_CFG = 0u;   // uint4(innerBody, flags, queryMask, pad)
static const uint C_SHAPE = 1u; // radius, halfHeight (segment half length), skin, cosMaxSlope
static const uint C_MOVE = 2u;  // stepUp, stepDown (stick to floor), jumpSpeed, airControl (1/s)
static const uint C_MASS = 3u;  // mass, pushRatio (bodies lighter than mass·ratio do not block), maxRecovery (m/step), gravityScale
static const uint C_IN = 4u;    // host input: desired velocity.xyz (world, horizontal), asfloat(jumpSeq)
static const uint C_POS = 5u;   // capsule centre.xyz, asfloat(lastJumpSeq)
static const uint C_VEL = 6u;   // velocity.xyz, asfloat(groundState)
static const uint C_GND = 7u;   // ground normal.xyz, asfloat(groundBody) (NO_INDEX = none)
static const uint C_GPT = 8u;   // ground contact point (world).xyz, asfloat(groundCollider)
static const uint C_GVEL = 9u;  // ground point velocity.xyz, time in air (s)
static const uint C_OUT = 10u;  // uint4(eventSeq, lastEvent, planes, outFlags)
static const uint C_PUSH = 11u; // weight force on a dynamic ground (N).xyz, asfloat(candidates)
static const uint C_DBG = 12u;  // last verify-cast fraction, stepped height, planes blocking, pad

// Character flags (C_CFG.y)
static const uint CF_ENABLED = 1u;
static const uint CF_PUSH_WEIGHT = 2u; // stand on dynamic bodies with the character's weight

// Ground states (C_VEL.w)
static const uint GS_ON_GROUND = 0u;
static const uint GS_ON_STEEP = 1u;   // touching ground too steep to stand on (slides)
static const uint GS_IN_AIR = 2u;

// Out flags (C_OUT.w), this step
static const uint CO_STEPPED_UP = 1u;
static const uint CO_STUCK_TO_FLOOR = 2u;
static const uint CO_CRUSHED = 4u;
static const uint CO_JUMPED = 8u;
static const uint CO_CAST_CUT = 16u;  // the verifying cast found geometry the planes missed

// ---- Scene query record (in-step, results read back asynchronously) ----
static const uint QUERY_STRIDE = 5u;
static const uint Q_IN0 = 0u;  // origin.xyz, maxDist
static const uint Q_IN1 = 1u;  // dir.xyz, radius (0 = ray, > 0 = sphere / capsule cast)
static const uint Q_IN2 = 2u;  // uint4(excludeBody, mask, kind (bit 0 capsule along +Y, bit 8 full scan), asfloat(halfHeight))
static const uint Q_OUT0 = 3u; // hit point.xyz, distance (< 0: miss)
static const uint Q_OUT1 = 4u; // normal.xyz, asfloat(collider)

// ---- Projectile record (segment casts before the step) ----
static const uint PROJ_STRIDE = 6u;
static const uint PJ_POS = 0u;  // position.xyz, radius (0 = ray)
static const uint PJ_VEL = 1u;  // velocity.xyz, mass
static const uint PJ_CFG = 2u;  // uint4(flags, mask, excludeBody, total hits)
static const uint PJ_PEN = 3u;  // kinetic energy (J), penetration cost (J per m of collider), drag (1/s), maxHitsPerStep
static const uint PJ_OUT = 4u;  // last hit point.xyz, asfloat(collider)
static const uint PJ_OUT2 = 5u; // last hit normal.xyz, asfloat(hits this step)

static const uint PF_ALIVE = 1u;
static const uint PF_IMPULSES = 2u; // hits push dynamic bodies (impulse records)

// Event records (scratch event ring, world_step EVENT_*)
static const uint EVENT_CHAR_LANDED = 5u;
static const uint EVENT_CHAR_SUPPORT_LOST = 6u;
static const uint EVENT_CHAR_CRUSHED = 7u;
static const uint EVENT_PROJECTILE_HIT = 8u;
static const uint EVENT_PROJECTILE_STOPPED = 9u;

// Candidates and contact planes a character gathers in one step.
static const uint CHAR_MAX_CANDIDATES = 288u;
static const uint PROJ_SEGMENTS = 4u;   // arc segments per step
static const uint PROJ_MAX_HITS = 8u;   // hits per step at most (PJ_PEN.w lowers it)

#endif
