// Opt-in: narrowphase finishes reused copies for moving pairs and marks them
// final; the per-contact persistence pass skips them.
#define ENGINE_COMPACT_SOLVE 1
#define ENGINE_PERSISTENT_CONTACTS 1
#define ENGINE_REUSE_COPY 1
#define narrowphase narrowphase_copy_persistent
#define narrowphase_defer narrowphase_defer_copy_persistent
#define narrowphase_fresh narrowphase_fresh_copy_persistent
#define persist_contacts persist_contacts_copy_persistent
#include "world_step.slang"
