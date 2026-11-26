// --- Continuous contact's pass (docs/continuous-contact-design.md, "The GPU port"): between a
// step's node update and its rest, a workgroup a contact group. The CPU reference:
// crates/engine-softbody/src/explicit_contact/dat.rs (dat_truncate, list_pairs, speculative)
// and ccd.rs (resolve). ---

#include "explicit_beams_cc_math.slang"

void ccPass(uint off, uint k, uint s, uint lane)
{
}
