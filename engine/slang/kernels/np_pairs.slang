// Opt-in (pairs=np): the two-pass narrowphase writes each manifold's solver
// pair as it writes the manifold, so pack_solver_pairs no longer runs.
#define ENGINE_COMPACT_SOLVE 1
#define ENGINE_PERSISTENT_CONTACTS 1
#define ENGINE_REUSE_COPY 1
#define ENGINE_NP_PAIRS 1
#define narrowphase_defer narrowphase_defer_pairs_copy_persistent
#define narrowphase_fresh narrowphase_fresh_pairs_copy_persistent
#include "world_step.slang"
