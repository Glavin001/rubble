// vehicle_loads for the persistent-contact host (rigid_plan resolve_entry):
// the same kernel, reading contacts through the persistent hot/cold records.
// Only vehicle_loads reads contacts; the other vehicle kernels stay canonical.
#define ENGINE_COMPACT_SOLVE 1
#define ENGINE_PERSISTENT_CONTACTS 1
#define vehicle_loads vehicle_loads_persistent
#include "vehicle_step.slang"
