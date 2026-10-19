# Background datagen status — learned-impact (2026-10-19)

## Objective
Keep CPU saturated generating diverse real oracle training data. Labels unchanged
(fail class/mode, peak util, joint_state_end, force_end, chunk_state_delta).

## Completed

### Diversity v1 → `/tmp/li-diversity-v1`
- **35 events · 264 frames · 112.7 s wall** @ 16 jobs · ~115 MB
- Soft/push/ram/punch/ring walls, houses, crackling cantilevers, arch/frame/anchors
- Train holdouts written under `/tmp/li-diversity-v1-train/` (see `diversity-v1-train-2026-10-19.md`)

## In progress / next

### Diversity v2 catalog (expanded)
- New event ids (`v2_*`); `--diversity-v2 --confirm-diversity --out /tmp/li-diversity-v2`
- Adds: denser wall speeds, **angled rams** (`wall_impact_angled`), soft negatives,
  house soft/car/blast + **24–32 frame** ring-outs, more cantilever lengths/loads,
  masonry_wall + spall, longer frame/arch clips
- Keep `--diversity` / `--prove` behavior

### Generate command
```bash
cd engine/stress-oracle
cargo run --release -p stress-learned --bin learned_impact_generate -- \
  --diversity-v2 --confirm-diversity --out /tmp/li-diversity-v2 --jobs 16
```

## Log
| Time | Action | Result |
|---|---|---|
| ~02:40 | v1 generate @ 16 jobs | 35 / 264 / 112.7 s |
| ~03:00 | v2 catalog + angled wall builder | coding |
| — | v2 generate @ 16 jobs | starting |

## Next batch after v2
- Optional short MPS train smoke → `/tmp/li-diversity-v2-train/` (do not block gen)
- If v2 completes: expand v3 (more angles × speeds, more 32-frame houses) → `/tmp/li-diversity-v3`

## Constraints
No commits · no blast_parity · no inspiration/ · resume by default
