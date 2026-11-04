# Soft-body research notes

These are reading notes on how other engines simulate deformable bodies, kept
for reference while building Rubble's deformable vehicles. The target is
BeamNG-resolution node–beam vehicles, dozens to hundreds at once, that deform
like metal rather than jelly.

- **Clean-room.** Every note describes behaviour and design in our own words.
  None contains code, and nothing was copied or translated. Source paths are
  pointers for a reader, not material to port. Rigs of Rods runs only as a
  black-box reference: [ror-ground-truth harness](../softbody-beamng-parity.md).
- **Unverified.** Subagents did the reading, reviewing the code checked out
  under the gitignored `inspiration/` folder (2026-11-02). Benchmark figures
  are the authors' or reviewers' own unless a note says otherwise. Confirm a
  claim before relying on it.

## Summary

- [softbody-solver-survey.md](../softbody-solver-survey.md): findings that
  hold across engines, and ranked recommendations for Rubble.
- [softbody-fast-tier-research.md](../softbody-fast-tier-research.md): the
  literature behind a faster implicit tier that still looks like sheet metal.

## Per engine

| Note | What it covers |
|---|---|
| [rigs-of-rods.md](rigs-of-rods.md) | The parity reference: an explicit 2 kHz node–beam solver with plastic beams, node–triangle contact, a drivetrain built from nodes, and flexbody skinning |
| [avbd.md](avbd.md) | Augmented vertex block descent (C++ demo, three-avbd on WebGPU, sandbox): hard rows, yield in the dual update, GPU colouring |
| [newton.md](newton.md) | NVIDIA Newton: VBD, XPBD, barrier self-contact, GPU launch structure |
| [physx.md](physx.md) | PhysX GPU deformables: batched launches, capped colouring, a plasticity kernel that is switched off |
| [mujoco.md](mujoco.md) | MuJoCo and MuJoCo Warp flex: implicit stiffness, Wardetzky bending, batching many worlds |
| [jolt.md](jolt.md) | Jolt: XPBD with small steps, tethers, per-body clusters on the CPU |
| [jgs2-and-benchmarks.md](jgs2-and-benchmarks.md) | JGS2 subspace coupling; an elasticity-solver benchmark comparing XPBD, PD, VBD and Newton |
