# AGENTS.md

- Assume our target is 10,000+ active rigid bodies when calculating performance at scale
- Node: the repo pins its version in `.nvmrc`. Run `nvm use` (or rely on an
  nvm `cd` hook) before any `node`/`npm` command.
- Never make tests, builds or runtime code depend on `inspiration/`: it is
  gitignored and absent in CI, fresh clones and worktrees. Read it to
  understand an algorithm, then write our own tests from its definition and
  physical invariants (see `engine/crates/engine-scenes/tests/web_scenes.rs`); do not
  import or record its outputs.
- Performance work must not trade away physics quality or accuracy. Never make
  lossy changes on our own initiative: simplifying or decimating collision
  shapes, dropping contacts or solver iterations, coarsening timesteps,
  loosening tolerances, or skipping work whose result could differ. Prefer
  lossless optimizations (same results, less work). When a limit or trade-off
  is genuinely wanted:
  - make it an explicit, engine-wide configuration that fails with a clear
    error when input exceeds it, never something that silently degrades input;
  - or fix it at the source (for example, re-export assets from Shardwright with
    the desired settings) rather than altering data at load time.
  Measuring a lossy option to report its effect is fine; adopting it needs the
  user's decision.
- Destruction is forward-only. Fracture is decided before or within a step and
  never by re-simulating one: no save-and-restore of the rigid engine and no
  "internal correction" pass that undoes a step and solves it again (PhysX's
  `internalCorrectionLimit`). A strike's breaks are predicted at the end of the
  frame before it and split before the engine steps. This is the routing check
  plus a predictor such as `threshold_predict` or the GPU punch
  (`engine/stress-oracle/findings/rewind-scaling.md` design C, and the charter's
  M5 routing design).
  - The correction pass (`GpuWorld::correct_frame`) exists only as a measurement,
    behind `STRESS_CORRECTION_LIMIT=1`, off by default.
  - Do not enable it in a profile or in production code.

- Modal's GPU target is CUDA (`engine-cuda`): kernels compile to GPU code locally with
  nvcc (free), and Modal only runs the binary, seconds a container.
  `engine/tools/modal/gpu_run.sh [--fanout | --fanout-tests] -- <cargo selection>`
  runs each test binary or each test in a container of its own, all at once (built in
  local Docker). Vulkan on Modal (wgpu, the World, the matrix's `run.sh --modal`) is
  opt-in only and not for kernel edits: NVIDIA compiles each new kernel version on
  Modal's hosts, and one explicit-beams pipeline took over 28 minutes, past the cap.
  The scenario matrix runs on the Mac's GPU until it runs on CUDA. Modal timing is
  directional; bit-for-bit verdicts are exact. `engine/tools/modal/README.md`.
- Checking a kernel change: climb a ladder, cheapest rung first, and do not go up a
  rung until the one below passes. Never wait minutes for an answer a rung of seconds
  gives. The bar does not move: what must be bit for bit is checked bit for bit.
  1. **Seconds, after every edit.** `engine/tools/explicit-2khz/fast-check.sh SWITCH
     [FIXTURE ...]` replays saved moments of the lab's multi-car run (1, 2, 4, 8 and 32
     cars; falling, first collisions, pile, rest) both ways from one state, compares
     every state word and times both: about a second a fixture, 5 s for one car count.
     It builds with `--profile fast` (no link-time optimisation, incremental; the same
     optimisation level) and the kernel compiles are cached one by one, so a rebuild is
     5 s after an edit to a small kernel file and 25 s after one to
     `explicit_beams.slang` (it was 91 s). `fast-check.sh --make` saves the fixtures
     (3 minutes, once, and again after a change of the buffers' layout). A new switch
     needs its `--ab-switch` value; a new kernel without a switch is checked against
     the committed kernels (`--ab-switch kernel --ab-kernel-dir`).
  1.5. **Half a minute: the gate for every perf change.** `engine/tools/scenario-matrix/run.sh
     SWITCH` (from `engine/`) replays a dozen saved use cases, not only the mountain: one
     soft car driving, tumbling and into a wall; piles of 2, 8 and 32; head-on and side
     impacts; a car into rigid bricks and under falling rocks; a car into a node-beam
     barrier; a loaded node-beam truss alone. Each row both ways from one state, every
     state word compared, each frame timed apart (median). One table, one verdict: **PASS
     only if every row is bit for bit and no row is slower than its off half by more than
     the calibrated noise threshold** (5 % and 0.05 ms; a row past it runs twice more and
     the median of the three decides; `docs/measurements/scenario-matrix.md`).
     A change that speeds up one row and slows another does not pass. Rung 1 stays the
     per-edit loop (one fixture, seconds); run the matrix before keeping a change and
     before rung 2. `run.sh --baseline` prints each row's ms a frame against its target
     (one soft body 1 ms, eight in contact 3 ms); `run.sh --make` saves the states (once,
     and after a change of the buffers' layout); `PROFILE_SPLIT=1` adds each row's frame
     by dispatch kind. It runs on the Mac's GPU under the lock (`--modal` runs it on
     Modal through Vulkan: opt-in, never after a kernel edit, see above).
  2. **A minute.** The same with `FRAMES=20`, `EXTRA="--ab-profile"` for the frame by
     dispatch kind, `explicit_loop_cap`, and Chrome's validation of every module
     (`EXPLICIT_WGSL_MODULES_ONLY=1 node --experimental-websocket
     explicit-wgsl-check.mjs`, 5 s: Chrome rejects barriers naga accepts and has a
     workgroup storage limit Metal does not).
  3. **Before a commit that changes results or a default.** The suites that hold
     GPU == CPU reference and the paths against each other
     (`tools/explicit-2khz/perf-verify.sh default ...`), `engine-metal` parity.
     A switch that will be on by default also passes Chrome's bit-for-bit A/B on the lab
     page (minutes; from `engine/`, after the web build):
     `tools/vehicle-lab/run.sh --probe --locked --ab MASK --abUntil 40 "m8:mode=mountain&cars=8&terrain=ledges&timing=step"`
     (frame-locked, so the page's run does not depend on the GPU's speed; every simulated
     second the vehicles run 20 frames from the page's state with the switch on and off,
     `WebWorld.explicitAb`, every state word compared; the first differing state is saved
     to `out/chrome-ab/` as a fixture for `fast-check.sh perf=MASK FILE`). Two runs of
     the page with the switch on and off, compared second by second:
     `--ab 0 --abSaveAll TAG` each, then `vehicle_lab ... --from-fixture A
     --compare-fixture B`. A Chrome state that differs goes into `out/fixtures/` (rung 1).
  4. **Before a merge or a number in a doc.** The whole 40 s runs (native and the Chrome
     probe) and every suite.
  When a higher rung finds something a lower one missed, add a fixture or a test at the
  lowest rung that would have caught it, so that it is never found by a long run again.
  Timing: the fixtures' times are for ranking a change; a number for the doc comes from
  rung 4 on a GPU nothing else is using (`with-gpu-lock.py` queues GPU jobs, it does not
  keep the user's own page off the GPU).
- Generated vehicles: read `engine/docs/generated-vehicles-lessons.md` first (lessons,
  measurements, decisions; add to it rather than redoing research).
- Deformable vehicles: `engine/docs/deformable-vehicles.md` explains the model (keep it
  current when the model changes). Soft-body vehicles in the World: start with `engine/docs/handoff/soft-rigid-world.md`
  (what exists, lessons, how to run and measure) and `engine/docs/handoff/known-bugs.md`.
  Project skills live in `.cursor/skills/` (linked from `.claude/skills/`): read
  `gpu-safe-runs` before any GPU run and `physics-evidence-first` before changing physics.

## Cursor Cloud specific instructions

> All standard build/test/lint/web commands and prerequisites are in `CONTRIBUTING.md`.
> This section covers **only** Cloud-Agent-specific setup and caveats.

### Shell environment

Every shell session must export these before any Rust command:

```bash
export RUSTUP_TOOLCHAIN=stable
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

These are already appended to `~/.bashrc` by the initial setup, so new
interactive shells inherit them. For non-interactive shells (e.g. background
commands), export them explicitly.

### Running the native viewer

The virtual X11 display is at `DISPLAY=:1`. Launch the viewer with:

```bash
DISPLAY=:1 cargo run -p rubble-viewer --example demo_3d --release
```

### Chrome WebGPU in this VM

The system Chrome `.desktop` entry has been patched with the SwiftShader
WebGPU flags documented in `CONTRIBUTING.md → Chrome WebGPU without a GPU`.
If Chrome is already running without those flags, kill it and relaunch so
the web demos render properly.

### Update script scope

The VM update script installs: `mesa-vulkan-drivers`, `libxkbcommon-x11-0`,
`libxcb-xkb1`, Rust stable + `rustfmt` + `clippy`, `wasm32-unknown-unknown`
target, `cargo-nextest`, and `npm ci` for `web/`. It does **not** start any
services. See `CONTRIBUTING.md` for how to run builds, tests, and servers.
