# Rubble V1 Target Architecture

This document captures the intended production architecture for Rubble as a
high-performance GPU rigid body engine targeted at both native Rust and
web/WASM.

## Goals

- One simulation architecture for native and web.
- GPU-first broadphase, narrowphase, contact solve, and warmstart.
- 2D and 3D share the same stage structure even when kernels differ.
- Rigid bodies only: static, dynamic, kinematic.
- Shapes: primitives, convex hulls, and compound shapes.
- Optimize for tens of thousands of rigid bodies in debris-style scenes.
- Stable piles, stacking, friction, and sleep-friendly settling.

## Core pipeline per step

1. integrate/predict inertial state
2. generate AABBs (motion expanded / speculative contact friendly)
3. broadphase via GPU LBVH
4. narrowphase and manifold/contact generation
5. warmstart lookup and cached dual/stiffness state
6. build body-contact graph / color schedule
7. free-motion / active-body pass
8. AVBD primal solve by color
9. AVBD dual update in parallel over contacts
10. optional velocity-level restitution pass
11. velocity extraction
12. sleep / activation bookkeeping

## Design rules

- Native and web share the same simulation logic.
- The CPU is orchestration only, not an alternate physics implementation.
- Expensive contact terms should be cached once per step and reused in the
  solve when practical.
- Scheduling / coloring must remain deterministic enough for debugging and
  stable enough for large sparse or clustered scenes.
- Profiling and instrumentation must be opt-in so benchmark numbers are not
  polluted by timestamp overhead.

## Current implementation direction

The changes in this branch move Rubble closer to that architecture by:

- keeping native and web on the same 3D GPU solve path,
- reducing coloring synchronization cost,
- simplifying the 3D primal kernel to one body per invocation,
- aligning AVBD defaults with the paper,
- and reducing per-step allocation / upload churn in shared GPU primitives.
