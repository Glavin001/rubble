| check | result | detail |
|---|---|---|
| one root named `vehicle` | pass | roots ['vehicle'] |
| flat hierarchy | pass | 97 children of `vehicle` |
| every part present | pass | 61 parts; missing [] |
| every marker present | pass | 36 markers; missing []; fender bolts 4, bumper bolts 4 |
| materials use only the spec names | pass | 10 materials; others [] |
| glass transparent, roughness <= 0.05 | pass | [('BLEND', 0.019999999552965164)] |
| dimensions (measured) | pass | length 4711 mm, width 1870 mm, height 1381 mm, wheelbase 2743 mm, track_f 1486 mm, track_r 1486 mm, front_overhang 874 mm, ground_clearance 196 mm, tyre_d 670 mm, origin_mid_axles 0 mm, origin_centre_x 0 mm |
| origin on the ground, centred, midway between axles | pass | axle midpoint z 0.0 mm, centre x 0.0 mm, min y -0.0 mm |
| tyres touch y = 0; wheel centres within 5 mm of markers | pass | tyre min y [-0.0, -0.0, 0.0, 0.0] mm; centre error [0.0, 0.0, 0.0, 0.0] mm |
| no non-manifold edges | pass | {} |
| no zero-area triangles | pass | {} |
| no inverted normals | FAIL | {'body_shell': (2, [0.647, 1.036, 0.395]), 'fender_FL': (2, [0.787, 1.007, 0.543]), 'fender_FR': (2, [-0.789, 1.006, 0.544]), 'trunk_lid': (1, [-0.779, 0.956, -1.801])} |
| panel edge p95 <= 30 mm | pass | p5 8.6, p50 19.7, p95 27.6 mm |
| panel triangle aspect <= 6:1 | FAIL | p99 4.56, max 57.18; over 6: {'body_shell': 99, 'door_FL': 2, 'door_FR': 2, 'fender_FL': 43, 'fender_FR': 41, 'hood': 2, 'trunk_lid': 26}; worst {'body_shell': (48.3, [-0.807, 0.995, -0.9], [((0.6, 1.3, -0.8), 10), ((-0.8, 1.0, -0.9), 10), ((-0.6, 1.3, -0.8), 9)]), 'door_FL': (6.7, [0.699, 1.013, 0.512], [((0.7, 1.0, 0.5), 1), ((0.7, 1.0, 0.4), 1)]), 'door_FR': (6.7, [-0.699, 1.013, 0.512], [((-0.7, 1.0, 0.4), 1), ((-0.7, 1.0, 0.5), 1)]), 'fender_FL': (30.4, [0.821, 0.922, 2.136], [((0.9, 0.9, 2.2), 7), ((0.9, 0.7, 2.1), 6), ((0.8, 0.9, 2.1), 6)]), 'fender_FR': (20.3, [-0.654, 0.705, 2.121], [((-0.9, 0.9, 2.2), 7), ((-0.8, 0.9, 2.1), 7), ((-0.9, 0.7, 2.1), 6)]), 'hood': (7.0, [-0.773, 1.01, 0.536], [((0.8, 1.0, 0.5), 1), ((-0.8, 1.0, 0.5), 1)]), 'trunk_lid': (57.2, [0.78, 0.956, -1.802], [((0.8, 1.0, -1.8), 11), ((-0.8, 1.0, -1.8), 11), ((0.8, 1.0, -1.9), 2)])} |
| triangle budget | pass | body 168000 (150k-300k), wheel max 12172 (<= 15k), interior 24614 (<= 80k), lamps/glass/trim/bay 46220 (<= 60k), total 285482 (<= 500k) |
| shut-line gaps 3-6 mm | pass | 1285 border cells: min 3.8, p5 4.2, p50 4.5, p95 5.3 mm; under 3 mm: {}; over 6 mm: {'body_shell/trunk_lid': 16}; e.g. {'>6 body_shell/trunk_lid': [-0.7, 0.9, -2.42]} |
| no panel interpenetration over 1 mm | pass | deepest 0.0 mm  |
| no part pairs cross (any depth) | FAIL | 2 pairs: {'body_shell/trunk_lid': (84, [0.0, 0.97, -1.8]), 'body_shell/glass_quarter_R': (6, [-0.78, 1.02, -0.91])} |
| UV0 non-overlapping on painted panels | pass | 865009 texels at 2048^2, 0 covered twice {} at uv {}; all in [0,1]: True |
| texture memory within 64 MB | pass | 0 images (materials are flat PBR factors) |
