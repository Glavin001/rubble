"""Prints the tables of findings/rewind-scaling.md from the JSON lines next to it."""
import json,glob,os,sys
D=sys.argv[1] if len(sys.argv)>1 else os.path.dirname(os.path.abspath(__file__))
def load(name):
    out=[]
    p=os.path.join(D,name+'.jsonl')
    if not os.path.exists(p): return out
    for l in open(p):
        if l.startswith('{'):
            d=json.loads(l)
            if 'walls' in d: out.append(d)
    return out
def all_of(prefix):
    r=[]
    for p in sorted(glob.glob(os.path.join(D,prefix+'*.jsonl'))):
        r+=load(os.path.basename(p)[:-6])
    return r
def pick(rows,**k):
    for d in rows:
        if all(d.get(a)==b for a,b in k.items()): return d
def rf(d): return '—' if d is None else '%.1f / %d' % (d['ram_vy'], d['fragments'])
def imp(d): return '—' if d is None else '%.1f' % (max(d['frame_ms'])/1000)
oracle=load('oracle_40')+load('oracle_fast'); rub=load('rubble_40')+load('rubble_fast'); rr=load('rubbleredo_40')+load('rubbleredo_fast')
px=load('physxredo_40')+load('physxredo_fast'); pg=load('physxgpu'); bl=[d for d in all_of('blast') if d.get('scene','wall_row')=='wall_row']; wh=all_of('whole')
for v in [40.0,80.0,160.0]:
    print('\n#### T1 (%g m/s): ram exit speed / fragments\n' % v)
    print('| walls | oracle (truth) | rubble, no rewind | rubble with redo: redone | PhysX coupled frame (CPU ref): redone | PhysX GPU coupled | Blast-style limit 0 | limit 4 | limit 16 | limit 100 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for n in [1,2,4,8]:
        o=pick(oracle,walls=n,speed=v)
        if o is None: continue
        r=pick(rub,walls=n,speed=v); q=pick(rr,walls=n,speed=v); p=pick(px,walls=n,speed=v); g=pick(pg,walls=n,speed=v)
        cells=[]
        for lim in [0,4,16,100]:
            b=pick(bl,walls=n,speed=v,limit=lim,sub=1) or pick(bl,walls=n,speed=v,limit=lim) if True else None
            b=[d for d in bl if d['walls']==n and d['speed']==v and d['limit']==lim and d.get('sub',1)==1]
            b=b[0] if b else None
            cells.append('—' if b is None else '%.1f / %d (%d solves, %.0f s)' % (b['ram_vy'],b['fragments'],b['solves'],max(b['frame_ms'])/1000))
        red=lambda d,k='redone_frames': '—' if d is None else ('%s, %.1f / %d' % (int(d[k] or 0), d['ram_vy'], d['fragments']))
        print('| %d | %s | %s (%s s worst frame) | %s | %s | %s | %s |' % (n, rf(o), rf(r), imp(r), red(q), red(p), rf(g), ' | '.join(cells)))
print('\n#### T2. Blast-style correction limit sweep (first-front acceptance)\n')
print('| walls | speed | limit | ram (m/s) | fragments | bonds | corrections | full-frame solves | worst frame (s) | total (s) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for d in sorted([d for d in bl if d.get('sub',1)==1], key=lambda d:(d['speed'],d['walls'],d['limit'])):
    print('| %d | %g | %d | %.2f | %d | %d | %d | %d | %.1f | %.0f |' % (d['walls'],d['speed'],d['limit'],d['ram_vy'],d['fragments'],d['broken_bonds'],d['corrections'],d['solves'],max(d['frame_ms'])/1000,d['total_s']))
print('\n#### T2b. Blast-style passes accepting the whole trial fracture (40 m/s)\n')
print('| walls | sub | limit | ram (m/s) | fragments | bonds | corrections | solves |')
print('|---|---|---|---|---|---|---|---|')
for d in sorted(wh, key=lambda d:(d['walls'],d.get('sub',1),d['limit'])):
    print('| %d | %d | %d | %.2f | %d | %d | %d | %d |' % (d['walls'],d.get('sub',1),d['limit'],d['ram_vy'],d['fragments'],d['broken_bonds'],d['corrections'],d['solves']))
print('\n#### T3. PhysX steps per frame x correction limit (1 wall, 40 m/s, first-front)\n')
print('| PhysX steps per frame | limit | ram (m/s) | fragments | bonds | solves |')
print('|---|---|---|---|---|---|')
for d in sorted([d for d in bl if d['walls']==1 and d['speed']==40.0], key=lambda d:(d.get('sub',1),d['limit'])):
    print('| %d | %d | %.2f | %d | %d | %d |' % (d.get('sub',1),d['limit'],d['ram_vy'],d['fragments'],d['broken_bonds'],d['solves']))
print('\n#### T4. Striker (routing miss): ram / fragments / broken bonds\n')
print('| speed | oracle | PhysX coupled frame (CPU ref): redone | rubble with redo: redone | rubble, no rewind: unverified | Blast limit 0 | limit 4 | limit 16 | limit 100 |')
print('|---|---|---|---|---|---|---|---|---|')
os_=load('oracle_striker'); ps=load('physxredo_striker'); rs_=load('rubbleredo_striker'); rn=load('rubble_striker'); bs=load('blast_striker')
for v in [20.0,40.0]:
    o=pick(os_,speed=v); p=pick(ps,speed=v); q=pick(rs_,speed=v); r=pick(rn,speed=v)
    f=lambda d,k: '—' if d is None else '%s, %.2f / %d / %d' % (int(d[k]), d['ram_vy'], d['fragments'], d['broken_bonds'])
    bc=[]
    for lim in [0,4,16,100]:
        b=pick(bs,speed=v,limit=lim)
        bc.append('—' if b is None else '%.2f / %d / %d (%d solves)' % (b['ram_vy'],b['fragments'],b['broken_bonds'],b['solves']))
    print('| %g | %s | %s | %s | %s | %s |' % (v, '%.2f / %d / %d' % (o['ram_vy'],o['fragments'],o['broken_bonds']), f(p,'redone_frames'), f(q,'redone_frames'), f(r,'unverified_fractures'), ' | '.join(bc)))
qr=load('quiet_rubble_40'); qo=load('quiet_oracle_40'); qp=load('quiet_physxgpu_40'); px=load('physxredo_40')
if qr:
    print('\n#### T5. Cost at 40 m/s: median / worst frame (ms); whole 9-frame run (s)\n')
    print('| walls | oracle CPU | PhysX coupled, CPU reference (concurrent run) | PhysX GPU coupled (GPU stress) | rubble (GPU stress, no rewind) | Blast limit 16 (concurrent run) | Blast limit 100 (concurrent run) |')
    print('|---|---|---|---|---|---|---|')
    def c(d):
        if d is None: return '—'
        fm=sorted(d['frame_ms']); return '%.0f / %.0f; %.1f' % (fm[len(fm)//2], fm[-1], d['total_s'])
    for n in [1,2,4,8]:
        b16=[d for d in bl if d['walls']==n and d['speed']==40.0 and d['limit']==16 and d.get('sub',1)==1]
        b100=[d for d in bl if d['walls']==n and d['speed']==40.0 and d['limit']==100 and d.get('sub',1)==1]
        print('| %d | %s | %s | %s | %s | %s | %s |' % (n, c(pick(qo,walls=n)), c(pick(px,walls=n)), c(pick(qp,walls=n)), c(pick(qr,walls=n)), c(b16[0] if b16 else None), c(b100[0] if b100 else None)))
