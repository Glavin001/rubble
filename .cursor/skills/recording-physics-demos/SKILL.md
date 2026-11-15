---
name: recording-physics-demos
description: Record demo videos and stills of the WebGPU physics pages (brick-ring drive, crash town, crash orbits) that are smooth, honest about speed, and actually show the damage. Use when asked for demo videos, screenshots, before/after or orbit views of crashed vehicles, or when a video looks choppy or hides the subject.
---

# Recording physics demos

## Tools (in `engine/crates/engine-webgpu/`)
- **`brick-ring-drive-record.mjs <runs> [--inspect]`:** frame-locked headless Chrome. One World step (1/60 s) per video frame, 1280×720 at 60 fps, ffmpeg. `--inspect` (after the run names) appends an orbit around each crashed car with nearby bricks ghosted.
- **`crash-orbit.mjs`:**
  - `capture`: crash, then `window.__snapshot(i)` saves the car's node state, broken beams and renderer state;
  - `measure`: crush, length lost, largest displacement and torn-off nodes; no GPU (`crash-measure.mjs`);
  - `record`: a 12 s 360° orbit (`crash-orbit.html?snap=a.json[,b.json]`, two snapshots side by side);
  - `stills`: front three-quarter, side, rear three-quarter and top-front.
- **Page options:** `brick-ring-drive.html?inspect=auto`, or `I` in the page, ghosts the bricks around the car and orbits it. This is rendering only; the physics keeps running.

## Rules
- **Frame-locked video shows what the physics does, not how fast it runs.** Always also state the page's live steps per second, and say plainly when the live page is slower than the video.
- **Look at the stills before claiming success.** The subject has to be visible, lit and in frame. In the brick ring the bricks bury the car, so crash views need inspect mode or a snapshot orbit to show damage.
- **Run GPU recordings under `engine/scripts/with-gpu-lock.py`** with a kill timer (see `gpu-safe-runs`).
- **Save videos, frames and snapshots to `~/Development/rubble-artifacts/<topic>/`** or a worktree's gitignored `out/`, never `/tmp` or a scratchpad (restarts wipe them). Never overwrite earlier takes; suffix new ones (`-v2`, `-v3`).
- **Keep files the user receives under 30 MB** (the phone upload limit). Re-encode with `ffmpeg -crf 26` if needed.
- **Snapshots and renders of BeamNG community cars stay out of git** (see `beamng-community-content`).
- **Physics varies run to run** in Chrome, so quote ranges (e.g. 160–185 beams broken), and say which frame an "impact speed" was sampled on.
