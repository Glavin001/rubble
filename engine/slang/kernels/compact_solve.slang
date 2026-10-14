// Experimental solver-only contact format. Export separate entry names so the
// canonical path remains available for numerical and performance comparisons.
#define ENGINE_COMPACT_SOLVE 1
#define solve_primal_color solve_primal_color_compact
#define solve_primal_one solve_primal_one_compact
#define persist_manifolds persist_manifolds_compact
#define solve_dual solve_dual_compact
#define solve_primal_sparse solve_primal_sparse_compact
#include "world_step.slang"
