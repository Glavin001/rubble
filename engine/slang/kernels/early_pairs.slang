// Opt-in graph kernels that read collider->body indirection from SolverPairs
// packed immediately after collision (persistent storage). Separate exports keep
// the production graph entries byte-identical.
#define ENGINE_COMPACT_SOLVE 1
#define ENGINE_PERSISTENT_CONTACTS 1
#define ENGINE_EARLY_SOLVER_PAIRS 1
#define csr_degree_contacts csr_degree_contacts_pairs
#define csr_scatter_contacts csr_scatter_contacts_pairs
#define csr_sort csr_sort_pairs
#define init_colors init_colors_pairs
#define color_bodies color_bodies_pairs
#include "world_step.slang"
