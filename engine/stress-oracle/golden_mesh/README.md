# Oracle mesh studies

Not goldens: OpenCourant runs of a scene at other meshes than the committed golden,
kept as evidence for the README's "Known gaps". `opencourant_kN.json` used N elements
per chunk edge (the committed `golden/<scene>/opencourant.json` uses 2).

`b5_wall_impact_v40`: push_over at 2, 3, 4 (hole at 1); impact zone converged (fully
shattered), far-field tensile cracking not converged (9 / 128 / 348 / 552 broken bonds
for 1 / 2 / 3 / 4 elements per edge); ram speed lost 4.93 / 5.97 / 6.43 / 6.36 m/s.
The oracle's chunks have no crushing or erosion (ram face pressure ~190 MPa).
