# Energy budget of a run: dissipated (bonds, contacts, damping, crushing) against the
# energy available (impactor kinetic energy + gravity potential released by the chunks'
# vertical displacement). Masses from the scene's compound parts (scipy hull volumes).
import json, sys, numpy as np
from scipy.spatial import ConvexHull
scene = json.load(open(sys.argv[1])); obs = json.load(open(sys.argv[2]))
g = -scene.get('gravity', [0, 0, -9.81])[2]
mats = scene['materials']
body = scene['bodies'][0]
mass = []
for c in body['chunks']:
    rho = mats[c['material']]['density']
    if c.get('compound'):
        mass.append(rho * sum(ConvexHull(np.array(p['points'] if isinstance(p, dict) else p)).volume for p in c['compound']))
    else:
        h = c['half_extents']; mass.append(rho * 8 * h[0] * h[1] * h[2])
chunks = obs['bodies'][body['name']]
dpe = -sum(m * g * ch.get('displacement', [0, 0, 0])[2] for m, ch in zip(mass, chunks) if not ch.get('removed'))
ke = sum(0.5 * i['mass'] * sum(v * v for v in i['velocity']) for i in scene.get('impactors', []))
v = obs['values']
diss = {k: v.get(k, 0.0) for k in ('bond_dissipation', 'contact_dissipation', 'damping_dissipation', 'crush_energy')}
total = sum(diss.values())
print(f"{scene['name']}: mass {sum(mass)/1000:.1f} t; available {ke + dpe:.4g} J (impactor KE {ke:.4g}, potential released {dpe:.4g}); dissipated {total:.4g} J {({k: f'{x:.3g}' for k, x in diss.items()})}; broken {v.get('broken_bonds')}, fragments {v.get('fragments')}  -> {'OK' if total <= 1.05 * (ke + dpe) + 1 else 'ENERGY CREATED'}")
