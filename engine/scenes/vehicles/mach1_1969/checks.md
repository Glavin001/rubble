| check | result | detail |
|---|---|---|
| one root named `vehicle` | pass | roots ['vehicle'] |
| flat hierarchy | pass | 96 children of `vehicle` |
| every part present | pass | 60 parts; missing [] |
| every marker present | pass | 36 markers; missing []; fender bolts 4, bumper bolts 4 |
| materials use only the spec names | pass | 10 materials; others [] |
| glass transparent, roughness <= 0.05 | pass | [('BLEND', 0.019999999552965164)] |
| dimensions (measured) | pass | length 4784 mm, width 1861 mm, height 1300 mm, wheelbase 2743 mm, track_f 1486 mm, track_r 1486 mm, front_overhang 951 mm, ground_clearance 149 mm, tyre_d 662 mm, origin_mid_axles 0 mm, origin_centre_x 0 mm |
| origin on the ground, centred, midway between axles | pass | axle midpoint z 0.0 mm, centre x 0.0 mm, min y -0.0 mm |
| tyres touch y = 0; wheel centres within 5 mm of markers | pass | tyre min y [-0.0, -0.0, -0.0, -0.0] mm; centre error [0.0, 0.0, 0.0, 0.0] mm |
| no non-manifold edges | pass | {} |
| no zero-area triangles | pass | {} |
| no inverted normals | pass | {} |
| panel edge p95 <= 30 mm | pass | p5 8.4, p50 19.8, p95 27.8 mm |
| panel triangle aspect <= 6:1 | FAIL | p99 4.62, max 18.55; over 6: {'body_shell': 41, 'door_FL': 1, 'fender_FL': 28, 'fender_FR': 28, 'hood': 4, 'trunk_lid': 4}; worst {'body_shell': (16.0, [0.662, 1.174, -0.88], [((-0.7, 1.2, -0.9), 7), ((0.8, 0.9, 0.6), 6), ((-0.8, 0.9, 0.6), 6)]), 'door_FL': (6.2, [0.79, 0.88, 0.622], [((0.8, 0.9, 0.6), 1)]), 'fender_FL': (18.1, [0.77, 0.732, 2.279], [((0.8, 0.7, 2.3), 9), ((0.7, 0.7, 2.3), 9), ((0.8, 0.6, 2.2), 5)]), 'fender_FR': (18.1, [-0.77, 0.732, 2.279], [((-0.8, 0.7, 2.3), 9), ((-0.7, 0.7, 2.3), 9), ((-0.8, 0.6, 2.2), 5)]), 'hood': (11.2, [0.771, 0.881, 0.635], [((0.8, 0.9, 0.6), 2), ((-0.8, 0.9, 0.6), 2)]), 'trunk_lid': (18.6, [-0.794, 0.946, -1.775], [((0.8, 1.0, -1.8), 1), ((-0.8, 1.0, -1.8), 1), ((-0.8, 0.9, -1.8), 1)])} |
| triangle budget | pass | body 159124 (150k-300k), wheel max 13268 (<= 15k), interior 24992 (<= 80k), lamps/glass/trim/bay 37272 (<= 60k), total 274460 (<= 500k) |
| shut-line gaps 3-6 mm | pass | 1406 border cells: min 3.6, p5 4.1, p50 4.6, p95 5.2 mm; under 3 mm: {}; over 6 mm: {'body_shell/trunk_lid': 12}; e.g. {'>6 body_shell/trunk_lid': [-0.72, 0.92, -2.42]} |
| no panel interpenetration over 1 mm | pass | deepest 0.0 mm  |
| no part pairs cross (any depth) | FAIL | 1 pairs: {'body_shell/trunk_lid': 84} |
| UV0 non-overlapping on painted panels | pass | 1011693 texels at 2048^2, 0 covered twice {} at uv {}; all in [0,1]: True |
| texture memory within 64 MB | pass | 0 images (materials are flat PBR factors) |
