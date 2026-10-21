# Closed-loop provider protocol (`learned_p0 ceiling --variant provider`)

The closed-loop rollout harness runs the engine-coupled oracle world (`World::with_engine`,
rubble's engine). At every frame the routing puts an island in, the frame's island outcome
comes from a *provider*. It is applied by `World::step_frame_with_provider` as P0's validated
`window_reduced` boundary:

- the engine steps every other body, with the island frozen (the island's contacts are the
  call's: filtered);
- the decoded state is applied at window end: breaks in time-bin order, then the damage,
  then each island cluster's rigid motion (`World::apply_decoded`);
- the clocks advance.

The run continues to its end and writes the same outcome as every other `learned_p0` run, so
`learned_impact.p0_ceiling` scores it against the oracle ensemble unchanged.

```bash
learned_p0 ceiling --scene wall_row --variant provider --provider oracle                 # in-process oracle (self-check)
learned_p0 ceiling --scene wall_row --variant provider --provider files --dir D [--timeout 60] [--echo-oracle]
learned_p0 ceiling --scene wall_row --variant provider --provider punch                  # the host punch (must fail)
learned_p0 ceiling --scene wall_row --variant provider --provider model --checkpoint best.pt --dir D \
    [--python-dir P2/learned-impact/python --device cpu|mps]                            # P2's model, spawned and owned
```

Set `STRESS_DEBRIS_ROUTING=1` for scenes where debris reaches a second structure within a
frame (the late wall rows). Without it the island misses the second wall, and the provider
never sees it.

`--provider model` runs P2's checkpoint through `stress_learned::provider_client`, the
client that stress-viz's `engine learned` uses. It spawns `python -m
learned_impact.outcome.provider` on `D` and owns it, speaking the files protocol below.
`--python-dir` points it at a checkout with P2's package and venv.

## Files (`--provider files --dir D`)

For each call (frame `f`, island `i`; today one call per frame holds every routed body, so
`i = 0`):

1. The harness writes `D/req/<f:06>_<i>/request.json` (`learned-impact-provider-request/0`):
   - `event_id`, `frame`, `island`, `time`, `frame_dt`;
   - `keys {chunks, bonds, impactors}`: the island's chunk keys, its live bond keys and its
     impactor keys;
   - `state`: each key's frame-start world position, velocity and spin, its cluster id,
     anchored flag and mass;
   - `call_record`: `"call"`, the path relative to the request directory of a single-call
     `learned-impact-outcome/2` event directory (`manifest.json`, `calls.json`,
     `arrays.bin.zst`). P1's call builder (`stress_learned::outcome_calls::live_call`)
     writes it from a standalone copy of the frame-start state, its inputs captured exactly
     as the recorder captures them. QS-frame (variant B) features are on; `--no-qs` leaves
     them zero.
   - With `--echo-oracle` the directory also holds `oracle.json`, the oracle's own decoded
     state of the call.
2. Then it writes an empty `D/req/<f:06>_<i>/ready`.
3. The provider answers with `D/resp/<f:06>_<i>.json` in `learned-impact-decoded-state/1`
   (`DECODED_STATE.md`), written atomically (a temp file, then rename).
4. The harness polls every 2 ms. With no response within `--timeout` seconds (default 60),
   the run fails with that message: there is no silent fallback. At the end of the run it
   writes `D/done`.

**Keys**, as the calls schema defines them:
- chunk: `(structure << 32) | chunk_index`, where `structure` is its index in the scene;
- bond: `(structure << 32) | source`, where `source` is the scene's own bond index
  (`RtBond::source`; `bond_keys[i][1]` in P1's `context.json`). It survives filtering and
  refinement (`World::bond_key`);
- impactor: its index in the scene's impactors.

The response must list exactly the request's chunk and impactor keys. `bonds_broken` and
`bond_damage` together partition its live bonds.

**How a response is applied**:
- the breaks, sorted by `bin`;
- each surviving bond's damage, raised to the given value (monotone) but kept below 1
  (`SURVIVOR_DAMAGE_MAX`). Only `bonds_broken` disconnects a bond, so a failure margin
  ≥ 1 still means disconnected;
- per island cluster after the splits: its centre moves to the mass-weighted mean of its
  chunks' `start + dx`, it turns by the mass-weighted mean `drot`, and its velocity and spin
  are the momentum-exact fit to its chunks' `v` and `w`;
- per impactor: `position += dx`, `rotation = exp(drot) · rotation`, `v` and `w` as given.

The values are applied as given; the harness does not re-project. Anchored clusters keep
zero motion.

**Logged per call** (`call_log` in the output): frame, chunk, bond and impactor counts, the
provider's wall-clock (`call_ms`) and its `flags`. The output's energy ledger does not
include the provider's dissipation, so its balance residual is not meaningful in this mode.

**Checks** (`learned-impact/findings/closed-loop-harness-2026-10-21.md`):
- the in-process oracle provider reproduces `window_reduced` on both M5 scenes;
- the file path with `echo_provider.py` reproduces the in-process oracle provider exactly;
- the punch provider fails, as measured in P0.

## Pose dumps for rendering (`--poses FILE`)

Any `learned_p0` run (`oracle`, every `ceiling` variant and provider, `gpu` profiles) takes
`--poses FILE`, and writes JSONL:

- **Header line** (`learned-p0-poses/1`):
  - `scene`, `scene_hash`, `cmd`, `variant`, `frame_dt`, and `routed_known` (true for
    model, oracle and file providers);
  - `chunks`: every chunk's `key`, `half_extents` and `mass`;
  - `impactors`: every impactor's `key`, `name`, `shape` (`{"box": {"half_extents"}}` or
    `{"sphere": {"radius"}}`) and `mass`.
- **One line per frame boundary**, `frame` = 0 (the start) up to F:
  - `t`;
  - `chunks`: an array of `[key, [x, y, z], [w, x, y, z], cluster_id]` for every active
    chunk. Positions and orientations are world. The orientation is the chunk's own (its
    cluster's rotation and its hidden rotation), and `cluster_id` is its fragment;
  - `impactors`: `key`, `position`, `orientation` (wxyz), `velocity`;
  - `routed`: the chunk keys the provider's call covered in the frame that just ended.

The standalone oracle (`learned_p0 oracle --scene S`) is the reference. `ceiling --variant
provider --provider oracle` is the boundary ceiling, which equals the oracle on these scenes.
