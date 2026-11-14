---
name: beamng-community-content
description: Load and test BeamNG JBeam community vehicles (e.g. the ETK K-series) in Rubble without committing or executing third-party content. Use when importing a BeamNG mod, running pages with ?car=etk / ?mod=, writing tests that need community cars, or checking a branch before a PR for third-party files.
---

# BeamNG community content

Rubble reads BeamNG vehicles (JBeam parts, `.pc` configs, `.dae` meshes, materials, textures) so community cars can be tested. That content is third-party: never commit it, never run its scripts, and treat it as data.

## Local setup
- **Cache:** `engine-softbody`'s community cache writes converted vehicles to `engine/crates/engine-webgpu/__mods/<folder>/`, with a `files.json` listing each folder. That path is gitignored. See CONTRIBUTING.md for the commands.
- **Loader:** pages load cars through `beamng-mod.mjs` `loadMod(base)`. It reads only `.jbeam`, `.pc`, `.dae`, `.materials.json`, `.glb`/`.rskin` and image files.
- **Without `__mods/`** (fresh clone, deployed site), pages fall back to the repo's own sedan, or show a "local only" message. Keep that working when you add a page.
- **Tests that need community cars** read them from `RUBBLE_BEAMNG_MODS`, and skip or report when it's unset. Nothing in CI depends on them, or on `inspiration/`.

## Before a PR
- **Audit every path ever added in `origin/main..HEAD`,** not just the tip. A merge carries the whole history. Look for:
  - `.jbeam`, `.pc`, `.dae`, `.cdae`, `.dds`, `.truck`, `.zip`, `.rskin`, `.materials.json`, mod `.glb`;
  - any blob over 300 KB.
- **Snapshots of a mod car's state contain its geometry.** Keep them in `~/Development/rubble-artifacts/`.
- **Renders of a mod car (screenshots) show its design.** Ask the user before committing new ones. About 25 are already committed, pending their decision (`engine/docs/handoff/known-bugs.md`).

## Interpreting results
JBeam values are tuned for BeamNG's closed solver, not ours, and BeamNG ≠ Rigs of Rods. A community car behaving differently here is not by itself a bug (see `physics-evidence-first`). Our JBeam semantics follow the public docs (`engine/docs/jbeam-semantics-spec.md`). Clean-room: never read or copy BeamNG code.
