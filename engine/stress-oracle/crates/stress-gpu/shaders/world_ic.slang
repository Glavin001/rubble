// world.slang with implicit penalty contact (`STRESS_REF_IMPLICIT_CONTACT=1`, a
// prototype): the same kernels, `penalty_force`'s normal force at the step's end. A build of
// its own so the default one compiles exactly as before.
#define IMPLICIT_CONTACT 1
#include "world.slang"
