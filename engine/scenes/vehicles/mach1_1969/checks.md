| check | result | detail |
|---|---|---|
| one root named `vehicle` | pass | roots ['vehicle'] |
| flat hierarchy | pass | 96 children of `vehicle` |
| every part present | pass | 60 parts; missing [] |
| every marker present | pass | 36 markers; missing []; fender bolts 4, bumper bolts 4 |
| materials use only the spec names | pass | 10 materials; others [] |
| glass transparent, roughness <= 0.05 | pass | [('BLEND', 0.019999999552965164)] |
| dimensions (measured) | pass | length 4775 mm, width 1873 mm, height 1300 mm, wheelbase 2743 mm, track_f 1486 mm, track_r 1486 mm, front_overhang 951 mm, ground_clearance 149 mm, tyre_d 662 mm, origin_mid_axles 0 mm, origin_centre_x 0 mm |
| origin on the ground, centred, midway between axles | pass | axle midpoint z 0.0 mm, centre x 0.0 mm, min y -0.0 mm |
| tyres touch y = 0; wheel centres within 5 mm of markers | pass | tyre min y [-0.0, -0.0, -0.0, -0.0] mm; centre error [0.0, 0.0, 0.0, 0.0] mm |
| no non-manifold edges | pass | {} |
| no zero-area triangles | pass | {} |
| no inverted normals | FAIL | {'body_shell': (3, [0.648, 0.918, 0.557]), 'fender_FL': (1, [0.878, 0.425, 1.755])} |
| panel edge p95 <= 30 mm | pass | p5 8.0, p50 19.9, p95 28.0 mm |
| panel triangle aspect <= 6:1 | FAIL | p99 4.58, max 100.27; over 6: {'body_shell': 51, 'fender_FL': 28, 'fender_FR': 28, 'hood': 4, 'trunk_lid': 16}; worst {'body_shell': (32.4, [0.663, 1.173, -0.88], [((-0.7, 1.2, -0.9), 7), ((0.7, 1.2, -0.9), 6), ((-0.8, 0.9, -2.4), 4)]), 'fender_FL': (18.1, [0.77, 0.732, 2.279], [((0.8, 0.7, 2.3), 9), ((0.7, 0.7, 2.3), 9), ((0.8, 0.6, 2.2), 5)]), 'fender_FR': (18.1, [-0.77, 0.732, 2.279], [((-0.8, 0.7, 2.3), 9), ((-0.7, 0.7, 2.3), 9), ((-0.8, 0.6, 2.2), 5)]), 'hood': (10.9, [0.771, 0.881, 0.635], [((0.8, 0.9, 0.6), 2), ((-0.8, 0.9, 0.6), 2)]), 'trunk_lid': (100.3, [0.78, 0.946, -1.775], [((0.8, 1.0, -1.8), 7), ((-0.8, 1.0, -1.8), 7), ((-0.8, 0.9, -1.8), 1)])} |
| triangle budget | pass | body 168666 (150k-300k), wheel max 12172 (<= 15k), interior 25310 (<= 80k), lamps/glass/trim/bay 41172 (<= 60k), total 283836 (<= 500k) |
| shut-line gaps 3-6 mm | pass | 1330 border cells: min 3.9, p5 4.1, p50 4.6, p95 5.6 mm; under 3 mm: {}; over 6 mm: {'body_shell/trunk_lid': 28, 'door_FL/fender_FL': 1}; e.g. {'>6 body_shell/trunk_lid': [-0.72, 0.92, -2.38], '>6 door_FL/fender_FL': [0.88, 0.28, 0.62]} |
| no panel interpenetration over 1 mm | pass | deepest 0.0 mm  |
| no part pairs cross (any depth) | FAIL | 3 pairs: {'body_shell/trunk_lid': 84, 'body_shell/taillight_L_lens': 4, 'body_shell/taillight_R_lens': 4} |
| UV0 non-overlapping on painted panels | pass | 814725 texels at 2048^2, 0 covered twice {} at uv {}; all in [0,1]: True |
| texture memory within 64 MB | pass | 0 images (materials are flat PBR factors) |
