# Screenshots: the combined branch (2026-10-17)

Captured on an M3 Max for PR #32: web demos in headless Chrome WebGPU (Slang → WGSL), the
stress viewers natively on Metal.

| File | What |
|---|---|
| `demo-ring110k.jpg` | `demo.html?scene=ring110k`: 110,441 bricks, 4 iterations, about 10 ms/step GPU |
| `demo-wall-smash.jpg` | `demo.html?scene=wall`: 2k-brick wall smash, sleeping bodies blue |
| `demo-joints.jpg` | `demo.html?scene=joints`: D6 joint showcase (hinges, chains, motors, ragdolls) |
| `demo-ragdolls10k.jpg` | `demo.html?scene=ragdolls10k`: 11,002 bodies, 10,001 D6 joints |
| `fracture-lab.jpg` | `fracture/`: Shardwright veneer house after five shots |
| `vehicle-bridge.jpg` | `vehicle.html?scene=bridge`: plank bridge with breakable bolts |
| `vehicle-traffic.jpg` | `vehicle.html?scene=traffic`: 49 vehicles, 869 bodies |
| `stress-destruction-view.jpg` | `destruction-view --scene house_ram`: GPU stress solver coupled to rubble, chunks as convex hulls coloured by bond utilization |
| `stress-oracle-vs-rubble.jpg` | `stress-viz compare s_house_car`: the CPU stress reference with its own rigid bodies vs with rubble's AVBD rigid bodies |
