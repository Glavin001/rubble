# DAgger rounds (routine, after each checkpoint)

Energy-envelope admission is applied again at load (the trainer's [data] line gives the final count).

| round (checkpoint) | rollout events | states | jobs rejected | relabelled calls | admitted (energy) | disk |
|---|---:|---:|---:|---:|---:|---:|
| w64-heads-s0 | 119 | 3638 | 426 | 3212 | 1591 | 226M |

w64-heads-s0's round at load in v7: 1509 of 1591 admitted (82 leave the energy envelope).
