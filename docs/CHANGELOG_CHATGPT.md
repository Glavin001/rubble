# Change Log and Rationale

## High-level themes

- remove divergent native/web 3D solve behavior
- move 3D solver closer to the intended GPU-first AVBD architecture
- reduce per-frame CPU/GPU synchronization and buffer churn
- align solver constants with the AVBD paper
- reduce instrumentation overhead in benchmark runs

## Code changes

### 1. 3D AVBD primal kernel rewritten
`crates/rubble3d/src/gpu/avbd_solve_wgsl.rs`

- replaced the old workgroup-reduction primal kernel with a one-body-per-
  invocation kernel
- rationale: better fit for sparse rigid-body contact graphs, less shared-memory
  reduction overhead, and easier dispatch scaling

### 2. Unified native + web 3D colored solve path
`crates/rubble3d/src/gpu/mod.rs`

- removed the broken wasm-only alternate scheduling path that referenced missing
  helper functions
- both native and wasm now use the same 3D GPU colored solve flow
- rationale: reduce divergence and keep one simulation path

### 3. GPU coloring synchronization reduced
`crates/rubble3d/src/gpu/mod.rs`

- GPU coloring now runs in fixed batches before checking the unfinished counter
- rationale: avoid a CPU/GPU sync after every color assignment round

### 4. Warmstart preparation cache added
`crates/rubble3d/src/gpu/mod.rs`

- added a persistent warmstart prepare count buffer
- added a cached warmstart-prepare bind group
- rationale: reduce per-frame bind group / buffer creation churn

### 5. Event-pair extraction bind group cache added
`crates/rubble3d/src/gpu/mod.rs`

- cached the event-pair extraction bind group
- rationale: repeated contact-pair downloads for host-side consumers no longer
  rebuild identical bind groups every frame

### 6. Precise GPU timing made opt-in
`crates/rubble3d/src/gpu/mod.rs`
`crates/rubble-primitives/src/gpu_lbvh.rs`

- precise GPU timestamp profiling now requires `RUBBLE_PRECISE_GPU_TIMING=1`
- rationale: benchmark / viewer performance should not pay for timestamp-marker
  command submissions by default

### 7. AVBD defaults aligned with paper values
`crates/rubble3d/src/lib.rs`
`crates/rubble3d/src/gpu/mod.rs`
`crates/rubble2d/src/lib.rs`
`crates/rubble2d/src/gpu/mod.rs`

- 3D `beta` default changed to `10`
- warmstart decay set to `0.99`
- warmstart alpha set to `0.95`
- rationale: match the paper's recommended parameter regime more closely

### 8. Greedy coloring improved
`crates/rubble-math/src/lib.rs`

- body graph coloring now uses a largest-degree-first deterministic ordering
- rationale: usually fewer colors and more stable scheduling than plain index
  order

### 9. Prefix scan buffer churn reduced
`crates/rubble-primitives/src/prefix_scan.rs`

- removed zero-upload of block sums
- persistent parameter buffers added for scan and inclusive conversion passes
- rationale: reduce per-dispatch allocations / uploads in hot utility code

### 10. Radix sort scratch reused
`crates/rubble-primitives/src/radix_sort.rs`

- added persistent temp key/value buffers, histogram storage, and params buffer
- removed zero uploads for temp buffers and histogram buffers
- rationale: reduce per-sort overhead in hot paths such as body-color sorting and
  warmstart lookup preparation

### 11. Build dependency optimization profile added
`Cargo.toml`

- added `profile.dev.build-override` and `profile.release.build-override`
- rationale: shader/build-dependency heavy workflows should keep build tools
  optimized, especially when using the Rust GPU path

## Notes

- No Rust toolchain was available in the execution environment, so the code was
  not compile-verified here.
- The current branch is still built on the existing Rubble structure, but the
  direction is intentionally toward a GPU-first, shared-path runtime.
