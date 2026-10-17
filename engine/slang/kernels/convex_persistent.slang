// Persistent hot/cold contact storage variants of the convex entries that emit
// contacts. SAT and argument kernels never touch contact storage and are shared.
#include "persistent_solve.slang"
#define CONVEX_SKIP_WORLD_INCLUDE 1
#define convex_classify convex_classify_persistent
#define convex_manifold convex_manifold_persistent
#define convex_sat_small convex_sat_small_persistent
#define convex_sat_large convex_sat_large_persistent
#define convex_write_args convex_write_args_persistent
#include "convex_step.slang"
