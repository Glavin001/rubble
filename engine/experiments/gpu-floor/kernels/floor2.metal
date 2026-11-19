// The floor kernel, second layout (the host puts floor.metal's element functions before
// this file). Built from the measured cost model (microbench): a phase costs its warps'
// instructions (0.35 ns a warp-instruction), a barrier 0.07 µs, a threadgroup-memory access
// about one instruction, a device access 4 to 29 ns a warp, and more than about 48 live
// registers a lane spill. So:
//  - a beam's hot words (ends, k, c, rest length, last extension rate) stay in registers:
//    five a slot, four slots a lane; everything else of a beam is read from device memory
//    only on the step it yields, breaks or is one of the car's few special elements;
//  - a beam hands its ends one scalar (force over length) through threadgroup memory; each
//    node gathers its beams in the reference's order from a streamed list, (x_self −
//    x_other)·g, which is the reference's push bit for bit at both ends;
//  - self-collision is a lane a triangle over its own row of listed nodes, each pair
//    filtered by two plane distances (now, and a step ago) before the full rule;
//  - contact sums, side memory, break groups, triangle breaks: device memory, touched only
//    on steps flagged as having them.
// Host defines: NN NR NT NG ND LANES STEPS NC NCW NGS (scalar slots) NV (vector slots)
// AERO_LO T_LANE T_ENT T_VIDX MAXENT PROW REFNODE NCPL HIT_CAP HITW CR LIST_REACH LIST_EPS NBOX.

struct Params
{
    uint cars;
    uint frames;
    float ground;
    uint skip;
    uint rtLanes, rtNN, rtNT, rtNC, rtSteps, rtNG, rtRS, pad2;
};
#define SKIP(k) ((P.skip & (1u << (k))) != 0u)
#if RUNTIME_SIZES
#define Z_STEPS P.rtSteps
#else
#define Z_STEPS STEPS
#endif
#define RS 4
// REGMODE: which of a slot's words stay in registers between phases (the rest come from
// the slot's hot record, one streamed device load a slot a phase). 0: w0, k, c, rest, up;
// 1: w0, c, up; 2: w0, up.
#ifndef REGMODE
#define REGMODE 0
#endif
#define HOTS(s) hot[lid * 6 + (s)]
#define BAR threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#ifndef EMU
#define TGDECL(type, name, count) threadgroup type name[count]
#define TGPTR(type) threadgroup type*
#endif
#ifndef PAIR
#define PAIR 0
#endif
// XATOM: what crosses between a pair's workgroups (each on its own core) read with
// atomic loads (1), and written with atomic stores too (2). 0: plain loads and stores.
// Measured (xsync --data): a plain store is never seen by a workgroup on another core
// within the dispatch, and a plain load does not see another core's atomic add to a word
// its own group stored; atomic stores and loads are seen. So 2 wherever groups share
// words; 0 and 1 are kept to show the difference.
// 3: atomic stores, and what another group stored atomically read with a plain vector load
// (measured bit for bit on the GPU, one load in place of three): the default for pairs.
#ifndef XATOM
#define XATOM (PAIR ? 3 : 0)
#endif
#if XATOM >= 1
#define SHL(i) atomic_load_explicit((device atomic_int*)&shift[i], memory_order_relaxed)
#define SHZ(i) atomic_store_explicit((device atomic_int*)&shift[i], 0, memory_order_relaxed)
#else
#define SHL(i) shift[i]
#define SHZ(i) shift[i] = 0
#endif
#if PAIR && !defined(EMU)
// A barrier across the workgroups of a contact pair (each on its own core; Metal has
// none): after this group's own barrier its lane 0 adds one to a device counter and waits,
// a bounded number of loads, until every group has; the other lanes wait at the group's
// barrier. Hitting the bound (a group that did not run beside the others) is a status bit,
// and this group stops waiting for the rest of the dispatch (one bounded wait at most).
#define XBAR { BAR; if (lid == 0u) { xround++; atomic_fetch_add_explicit(xcnt, 1u, memory_order_relaxed); if (xlate == 0u) { const uint want_ = NGROUPS * xround; uint i_ = 0u; for (; i_ < XCAP; i_++) { if (atomic_load_explicit(xcnt, memory_order_relaxed) >= want_) break; } if (i_ >= XCAP) xlate = 1u; } } BAR; }
#define KRETURN return
#endif
// TGVEC: a node's position (velocity, a vector slot) as one three-float element of
// threadgroup memory: one address and one load in place of three.
#ifndef TGVEC
#define TGVEC 1
#endif
// ROWOVER: the pair rows' filter also tests that a point in the band is over the triangle.
#ifndef ROWOVER
#define ROWOVER 1
#endif
// ROW4: the pair rows' points four a load, each by name (no vector indexed by a variable).
#ifndef ROW4
#define ROW4 1
#endif
// HITTG: the hit passes' per-node words (force, mass, hit count, staged sums) in threadgroup
// memory, a slot a node; 0: in device memory (each hit then walks some 170 scattered words).
#ifndef HITTG
#define HITTG 1
#endif
// (A slot is 11 words of the beam scalars' array; a hit has four.)
#define H2CAP (NGS / 44u)
// TGPV: a node's position and velocity next to each other (an end of a beam is then six
// adjacent floats, where adjacent loads are nearly free).
#ifndef TGPV
#define TGPV 0
#endif
#if TGVEC
inline V3 v3p(packed_float3 a) { return v3(a.x, a.y, a.z); }
#if TGPV
#define TGP(i) v3p(tgPos[(i) * 2u])
#define TGV(i) v3p(tgPos[(i) * 2u + 1u])
#define SETP(i, val) { V3 _v = (val); tgPos[(i) * 2u] = packed_float3(_v.x, _v.y, _v.z); }
#define SETV(i, val) { V3 _v = (val); tgPos[(i) * 2u + 1u] = packed_float3(_v.x, _v.y, _v.z); }
#else
#define TGP(i) v3p(tgPos[i])
#define TGV(i) v3p(tgVel[i])
#define SETP(i, val) { V3 _v = (val); tgPos[i] = packed_float3(_v.x, _v.y, _v.z); }
#define SETV(i, val) { V3 _v = (val); tgVel[i] = packed_float3(_v.x, _v.y, _v.z); }
#endif
#define GETV(vi) v3p(tgVec[vi])
#define PUTV(vi, val) { V3 _v = (val); tgVec[vi] = packed_float3(_v.x, _v.y, _v.z); }
#else
#define TGP(i) v3(tgPos[(i) * 3], tgPos[(i) * 3 + 1], tgPos[(i) * 3 + 2])
#define TGV(i) v3(tgVel[(i) * 3], tgVel[(i) * 3 + 1], tgVel[(i) * 3 + 2])
#define GETV(vi) v3(tgVec[(vi) * 3], tgVec[(vi) * 3 + 1], tgVec[(vi) * 3 + 2])
#define SETP(i, val) { V3 _v = (val); tgPos[(i) * 3] = _v.x; tgPos[(i) * 3 + 1] = _v.y; tgPos[(i) * 3 + 2] = _v.z; }
#define SETV(i, val) { V3 _v = (val); tgVel[(i) * 3] = _v.x; tgVel[(i) * 3 + 1] = _v.y; tgVel[(i) * 3 + 2] = _v.z; }
#define PUTV(vi, val) { V3 _v = (val); tgVec[(vi) * 3] = _v.x; tgVec[(vi) * 3 + 1] = _v.y; tgVec[(vi) * 3 + 2] = _v.z; }
#endif
// Slot word of a two-node beam: a (11 bits), b (11), alive (bit 22), slope on (23),
// deform asked (24), class (30..31). Of an aero triangle: its three nodes, 10 bits each.
#define W_ALIVE (1u << 22)
#define W_SLOPE (1u << 23)
#define W_DEFORM (1u << 24)
// (The owning layout: the lane's own node is the beam's second end.)
#define W_OWNB (1u << 25)
#define TG3P(o) TGP(o)
#define TG3V(o) TGV(o)
#define ENT_ADD(acc, e) { uint o3_ = (e) & 2047u; acc = acc + (Ps - TG3P(o3_)) * tgG[(e) >> 11]; }
#define C_PLAIN 0u
#define C_SUPPORT 1u
#define C_AERO 2u
#define C_GENERIC 3u
// Flags (threadgroup): see the kernel.
#define F_SHIFT 0
#define F_FIRED 1
#define F_REBUILD 2
#define F_HITS 3
#define F_BUILDS 4
#define F_MOST 5
#define F_STATUS 6
#define F_REC 7
#define F_HELD 8
#define F_GDIRTY 9
#define F_BROKE 10
#define F_OUTER 11
#define TG_FLAGS 12

#ifndef RIGID
#define RIGID 0
#endif
#if RIGID
// ---- mechanism 3: rigid bodies as proxies (explicit_rigid.rs ProxySet) ----------------
// The reference's quaternion arithmetic is glam's NEON code: written out here in its
// operation order (a horizontal add is (x + y) + (z + w); a dot of three, (x + y) + (z + 0)).
inline float dot3n(V3 a, V3 b) { return ((a.x * b.x) + (a.y * b.y)) + ((a.z * b.z) + 0.0f); }
inline float4 qmul(float4 l, float4 r)
{
    float4 t0 = float4(l.w * r.x, l.w * r.y, l.w * r.z, l.w * r.w);
    float4 t1 = float4(l.x * r.w, -(l.x * r.z), l.x * r.y, -(l.x * r.x));
    float4 t2 = float4(l.y * r.z, l.y * r.w, -(l.y * r.x), -(l.y * r.y));
    float4 t3 = float4(-(l.z * r.y), l.z * r.x, l.z * r.w, -(l.z * r.z));
    return float4((t0.x + t1.x) + (t2.x + t3.x), (t0.y + t1.y) + (t2.y + t3.y), (t0.z + t1.z) + (t2.z + t3.z), (t0.w + t1.w) + (t2.w + t3.w));
}
inline V3 qrot(float4 q, V3 v)
{
    V3 b = v3(q.x, q.y, q.z);
    float b2 = dot3n(b, b);
    float k0 = (q.w * q.w) - b2, k1 = dot3n(v, b) * 2.0f, k2 = q.w * 2.0f;
    V3 c = cross3(b, v);
    return v3((v.x * k0 + b.x * k1) + c.x * k2, (v.y * k0 + b.y * k1) + c.y * k2, (v.z * k0 + b.z * k1) + c.z * k2);
}
inline float4 qnormalize(float4 q)
{
    float len = sqrt(((q.x * q.x) + (q.y * q.y)) + ((q.z * q.z) + (q.w * q.w)));
    float rc = 1.0f / len;
    return float4(q.x * rc, q.y * rc, q.z * rc, q.w * rc);
}
inline bool qidentity(float4 q) { return q.x == 0.0f && q.y == 0.0f && q.z == 0.0f && q.w == 1.0f; }
inline V3 invInertiaMul(float4 rot, V3 ii, V3 v)
{
    V3 l = qrot(float4(-rot.x, -rot.y, -rot.z, rot.w), v);
    return qrot(rot, v3(ii.x * l.x, ii.y * l.y, ii.z * l.z));
}
// The bit-trick exponential and power (the rigid rule's surface is not the exact one).
inline float approxExp(float x)
{
    if (x < -15.0f) return 0.0f;
    if (x > 88.0f) return 1e38f;
    return as_type<float>(int(12102203.0f * x + 1064652319.0f));
}
inline float approxPow(float x, float y)
{
    float e = float(as_type<int>(x) - 1065353216);
    return as_type<float>(int(y * e + 1065353216.0f));
}
// contact_force on the plain box surface (RoR's rule: not dissipative, bit-trick friction), depth 0.
inline V3 contactForcePlain(V3 f, V3 v, float m, V3 n, float friction)
{
    float vn = dot3(v, n);
    float fn = dot3(f, n);
    float r = -fn;
    if (vn < 0.0f) r -= (((0.8f * vn) - ((0.2f * 0.0f) / H)) * m) / H;
    if (r <= 0.0f) return v3(0, 0, 0);
    V3 sf = f - fn * n;
    V3 sl = v - vn * n;
    float sigma = len3(sl);
    if (sigma > 0.0f) sl = sl / sigma;
    float gr = (r * BOXS.strength) * friction;
    float ms = BOXS.ms * gr;
    if (sigma < BOXS.va && gr > 0.0f && dot3(sf, sf) <= ms * ms)
    {
        float e = approxExp(-sigma / BOXS.va);
        return (r * n - (ms * (1.0f - e)) * sl) - sf;
    }
    float e = approxExp(-approxPow(sigma / BOXS.vs, BOXS.alpha));
    float g = BOXS.mc + (BOXS.ms - BOXS.mc) * e;
    return r * n - ((g + rmin(BOXS.t2 * sigma, 5.0f)) * gr) * sl;
}
// Shape rows (float4): 0 body, kind (0 box, 1 sphere), radius, peak; 1 hf3; 2 reference;
// 3 impulse; 4 moment; 5 lo, identity; 6 hi; 7 centre; 8..10 axes. Body rows: 0 pos,
// inverse mass; 1 rot; 2 vel, share; 3 ang vel; 4 inverse inertia, the step's count.
#define SHW 11
#define BDW 5
// shape_face: the outward normal of the surface nearest a point strictly inside.
inline bool rigidFace(device const float4* sh, V3 x, thread V3& normal)
{
    float4 lo = sh[5], hi = sh[6];
    if (!(x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)) return false;
    V3 c = v3(sh[7]);
    if (as_type<uint>(sh[0].y) == 1u)
    {
        V3 d = x - c;
        float l = sqrt((d.x * d.x + d.y * d.y) + d.z * d.z);
        if (!(l < sh[0].z)) return false;
        normal = l > 0.0f ? v3(d.x / l, d.y / l, d.z / l) : v3(0.0f, 1.0f, 0.0f);
        return true;
    }
    V3 hf3 = v3(sh[1]);
    if (lo.w != 0.0f)
    {
        // An identity rotation: box_face on the axis-aligned box.
        V3 blo = c - hf3, bhi = c + hf3;
        if (!(x.x > blo.x && x.x < bhi.x && x.y > blo.y && x.y < bhi.y && x.z > blo.z && x.z < bhi.z)) return false;
        normal = v3(0.0f, 0.0f, -1.0f);
        float dist = x.z - blo.z;
        if (bhi.z - x.z < dist) { dist = bhi.z - x.z; normal = v3(0.0f, 0.0f, 1.0f); }
        if (x.x - blo.x < dist) { dist = x.x - blo.x; normal = v3(-1.0f, 0.0f, 0.0f); }
        if (bhi.x - x.x < dist) { dist = bhi.x - x.x; normal = v3(1.0f, 0.0f, 0.0f); }
        if (x.y - blo.y < dist) { dist = x.y - blo.y; normal = v3(0.0f, -1.0f, 0.0f); }
        if (bhi.y - x.y < dist) { dist = bhi.y - x.y; normal = v3(0.0f, 1.0f, 0.0f); }
        return true;
    }
    V3 a0 = v3(sh[8]), a1 = v3(sh[9]), a2 = v3(sh[10]);
    V3 d = x - c;
    V3 l = v3(dot3(a0, d), dot3(a1, d), dot3(a2, d));
    if (!(l.x > -hf3.x && l.x < hf3.x && l.y > -hf3.y && l.y < hf3.y && l.z > -hf3.z && l.z < hf3.z)) return false;
    normal = -a2;
    float dist = l.z + hf3.z;
    if (hf3.z - l.z < dist) { dist = hf3.z - l.z; normal = a2; }
    if (l.x + hf3.x < dist) { dist = l.x + hf3.x; normal = -a0; }
    if (hf3.x - l.x < dist) { dist = hf3.x - l.x; normal = a0; }
    if (l.y + hf3.y < dist) { dist = l.y + hf3.y; normal = -a1; }
    if (hf3.y - l.y < dist) { dist = hf3.y - l.y; normal = a1; }
    return true;
}
#endif

struct Slow { float g; float u; float rest; uint bits; };
// One two-node beam's whole rule from its device records (the step it may yield or break,
// and the special kinds every step): bits 1 alive, 2 slope on, 4 deform asked, 8 its group
// set off, 16 broke now.
inline Slow slowScalar(uint r, device const uint4* recA, device const float4* recK, device const float4* recB,
                       device const float4* recE, device float4* recState, device atomic_uint* masters,
                       V3 pa, V3 pb, V3 va, V3 vb)
{
    uint4 a = recA[r];
    uint kind = a.z & 0xFFu, ty = (a.z >> 8) & 0xFFu;
    bool ext = ((a.z >> 16) & 1u) != 0u;
    int group = as_type<int>(a.w);
    float4 kc = recK[r], bb = recB[r];
    float4 s0 = recState[r * 3], s1 = recState[r * 3 + 1];
    float4 e1 = float4(0), e2 = float4(0);
    if (ext) { e1 = recE[r * 2]; e2 = recE[r * 2 + 1]; }
    BeamOut o;
    if (kind == K_PLAIN) o = plainForce(kc, bb, s0, s1, pa, pb, va, vb);
    else o = jbeamForce(kind, ty, kc, bb, e1, e2, ext, s0, s1, pa, pb, va, vb);
    Slow out;
    out.u = dot3(va - vb, o.d) * o.inv;
    float f = o.f;
    bool brokeNow = o.broke;
    if (o.check) { brokeNow = true; f = 0.0f; }
    if (brokeNow) s1.w = 1.0f;
    s1.z = (brokeNow || kind == K_PLAIN) ? 0.0f : o.slope;
    uint bits = brokeNow ? 16u : 1u;
    if (s1.z != 0.0f) bits |= 2u;
    if (ext)
    {
        uint mask = as_type<uint>(bb.w);
        bool yields = kind == K_JPLAIN || kind == K_JSUPPORT || kind == K_JANISO;
        if ((mask & EXT_DEFORM) != 0u && (brokeNow || (yields && fabs(s0.x - e2.x) > e2.w * e2.x))) bits |= 4u;
    }
    if (brokeNow && group > 0)
    {
        atomic_fetch_min_explicit(&masters[group], r, memory_order_relaxed);
        bits |= 8u;
    }
    recState[r * 3] = s0;
    recState[r * 3 + 1] = s1;
    recState[r * 3 + 2] = float4(out.u, 0.0f, 0.0f, 0.0f);
    out.g = f * o.inv;
    out.rest = s0.x;
    out.bits = bits;
    return out;
}

#ifndef SURF
#define SURF 0
#endif
#if SURF
// ---- surface contact (explicit_surface.rs surface_pass, spec §38) ----------------------
// The car's collision triangles, its beams that are no triangle's edge and its nodes in
// the band, against the world's shapes: the static boxes, then the rigid proxies' shapes
// (one slot each, in the reference's order). The geometry is the engine's strict
// closest-feature module (slang/physics/feature_query.slang, converted as text: floor.rs
// fq_msl), so a contact is the reference's bit for bit.
//@FQ
inline float3 fv(V3 a) { return float3(a.x, a.y, a.z); }
inline float3 fv(float4 a) { return float3(a.x, a.y, a.z); }
inline V3 vf(float3 a) { return v3(a.x, a.y, a.z); }
inline float4 f4(float3 a, float w) { return float4(a.x, a.y, a.z, w); }
// surface_response: Coulomb friction on the exact, dissipative box surface.
inline V3 surfaceResponse(V3 f, V3 v, float m, V3 n, Surface s, float p, float friction, thread float& budget)
{
    budget = 0.0f;
    if (p < 0.0f) return v3(0, 0, 0);
    float vn = dot3(v, n);
    float fn = dot3(f, n);
    float r = -fn - ((vn + rmax(vn, 0.0f)) * m) / H;
    if (vn < 0.0f) budget = rmax((0.5f * m) * (vn * vn) + (0.5f * H) * (fn * vn), 0.0f);
    if (r <= 0.0f) { budget = 0.0f; return v3(0, 0, 0); }
    V3 ft = f - n * fn;
    V3 vt = v - n * vn;
    V3 stick = -(ft + vt * (m / H));
    float gr = (r * s.strength) * friction;
    float ms = s.ms * gr;
    V3 c;
    if (dot3(stick, stick) <= ms * ms) c = n * r + stick;
    else
    {
        float sl = len3(vt);
        V3 dir = sl > 0.0f ? -(vt / sl) : stick / len3(stick);
        float e = exactExp(-exactPow(sl / s.vs, s.alpha));
        float g = s.mc + (s.ms - s.mc) * e;
        c = n * r + dir * ((g + rmin(s.t2 * sl, 5.0f)) * gr);
    }
    return noPositiveWork(c, r, n, f, v, m);
}
// no_positive_work_body: the friction capped with a turning body's full response.
struct BodyK { float k, im, imt; V3 arm; float4 rot; V3 ii; };
inline V3 bodyK(BodyK b, V3 x)
{
    V3 l = vf(fqRot(float4(-b.rot.x, -b.rot.y, -b.rot.z, b.rot.w), fv(cross3(b.arm, x))));
    V3 w = vf(fqRot(b.rot, float3(b.ii.x * l.x, b.ii.y * l.y, b.ii.z * l.z)));
    return x * b.imt + (x * b.im + cross3(w, b.arm)) * b.k;
}
inline V3 noPositiveWorkBody(V3 c, V3 f, V3 v, float m, V3 n, BodyK bk)
{
    if (m <= 0.0f) return c;
    float vn = dot3(v, n);
    float fn = dot3(f, n);
    float r = -fn - ((vn + rmax(vn, 0.0f)) * m) / H;
    if (r <= 0.0f) return c;
    float hm = H / m;
    V3 rn = n * r;
    V3 t = c - rn;
    V3 u = (v + v) + f * hm;
    V3 krn = bodyK(bk, rn);
    float a0 = dot3(rn, u) + H * dot3(rn, krn);
    float a1 = dot3(t, u) + (2.0f * H) * dot3(t, krn);
    float b = H * dot3(t, bodyK(bk, t));
    if ((a0 + a1) + b <= 0.0f || b <= 0.0f) return c;
    float disc = a1 * a1 - (4.0f * b) * a0;
    float s = disc < 0.0f ? 0.0f : rclamp((sqrt(disc) - a1) / (2.0f * b), 0.0f, 1.0f);
    return rn + t * s;
}
// A slot's rows (float4): 0 centre, kind; 1..3 axes; 4 half extents, body (NONE: static);
// 5 cull box lo; 6 hi; 7 the proxy shape's index.
#define SRW 8u
#define SURF_STATIC 0xFFFFu
#define SPAIR_CAP 4096u
#define SHIT_CAP 512u
// The pass's working data is in the scalars' array (free from the self hits' passes to the
// next beam pass), not device memory: a load there is some ten instructions, a dependent
// device load 0.2 us. A slot's 24 floats (cull box, centre, axes, half extents, kind, body,
// proxy shape), then every node's force as the next node pass will take it, then the staged
// pairs (two words: the nodes, the element and slot).
#define SB(s_) ((s_) * 24u)
#define SFB (NSS * 24u)
#define SHB (SFB + NN * 3u)
// The staged box tests: the pairs (triangles' first, then beams'), each in device memory its
// least gap (ordered bits), its kept candidates' count and 16 kept candidates of three rows
// (weights and walk order, point, shape point); a task (pair, box edge; 12: the element's run
// inside) keeps its candidates' gaps (eight words).
#ifndef SURF_STAGED
#define SURF_STAGED 1
#endif
#define HCAP ((NGS - SHB) / 2u > 512u ? 512u : (NGS - SHB) / 2u)
#define HTRI (HCAP * 3u / 8u)
#define HSEG (HCAP - HTRI)
#define HPW 200u
#define KCAP 16u
// One task's candidates in the walk's order (x_candidates, seg_candidates: a box's edge
// e0-e1, or past the twelve the run inside); `a.ilo..a.ihi` takes some of them.
inline void surfCand(bool tri, FqShape fs, float3 e0, float3 e1, float3 x0, float3 x1, float3 x2, uint e, thread FqAcc& a)
{
    if (e < 12u)
    {
        if (tri) fqSegTri(fs, e0, e1, x0, x1, x2, a);
        else
        {
            float3 d = ssub3(x1, x0);
            fqEdgeBeam(fs, e0, e1, x0, x1, d, sdot(d, d), a);
        }
    }
    else if (tri) fqEdgesInside(fs, x0, x1, x2, a);
    else fqBeamInside(fs, x0, x1, a);
}
inline FqShape surfShape(device const float4* row)
{
    FqShape s;
    s.kind = as_type<uint>(row[0].w);
    s.c = fv(row[0]); s.b = float3(0.0f, 0.0f, 0.0f);
    s.ax0 = fv(row[1]); s.ax1 = fv(row[2]); s.ax2 = fv(row[3]);
    s.hext = fv(row[4]);
    s.hull = 0u; s.nv = 0u; s.nf = 0u; s.ne = 0u; s.moved = 0u; s.off = float3(0.0f, 0.0f, 0.0f);
    return s;
}
#endif

kernel void floorFrame(
    constant Params& P [[buffer(0)]],
    device uint* big [[buffer(1)]],                  // every per-car word (CAR_WORDS a car), one base
    device const float4* nodeStatic [[buffer(2)]],
    device const uint4* recA [[buffer(4)]],
    device const float4* recK [[buffer(5)]],
    device const float4* recB [[buffer(6)]],
    device const float4* recE [[buffer(7)]],
    device const uint* tab [[buffer(12)]],           // lane table, node entry lists, vector indices
    device const uint* perm [[buffer(14)]],          // RS*LANES: the record of slot s of lane l
    device const uint4* triBreaks [[buffer(17)]],
    device const uint* cands [[buffer(22)]],
    device const uint4* tris [[buffer(23)]],
    device const uint* allowed [[buffer(24)]],
    uint lid [[thread_position_in_threadgroup]],
    uint car [[threadgroup_position_in_grid]])
{
#if TGVEC
#if TGPV
    TGDECL(packed_float3, tgPos, NN * 2);
#else
    TGDECL(packed_float3, tgPos, NN);
    TGDECL(packed_float3, tgVel, NN);
#endif
    TGDECL(packed_float3, tgVec, NV);
#else
    TGDECL(float, tgPos, NN * 3);
    TGDECL(float, tgVel, NN * 3);
    TGDECL(float, tgVec, NV * 3);
#endif
    TGDECL(float, tgG, NGS);
    TGDECL(uchar, tgHeld, NN);
    TGDECL(atomic_uint, tgCount, 4); // [0] the fastest node (bits), [2] hits
    TGDECL(uint, tgFlags, TG_FLAGS);
    TGDECL(float, tgPB, 4);

    // One derived pointer a lane (a pointer a buffer was twenty live registers, past the
    // spill cliff: every operation then cost three times its price).
    device uint* const cb = big + car * CAR_WORDS;
    #define nodeState ((device float4*)(cb + O_NODE))
    #define recState ((device float4*)(cb + O_REC))
    #define groupsLast (cb + O_GROUPS)
    #define deform (cb + O_DEFORM)
    #define triDead (cb + O_TRIDEAD)
    #define acc ((device atomic_int*)(cb + O_ACC))
    #define shift ((device int*)(cb + O_SHIFT))
    #define misc (cb + O_MISC)
    #define pairs (cb + O_PAIRS)
    // The wide list the working one is filtered from (LIST2), and where its nodes (and, last,
    // the reference node) were when it was built.
    #define pairsOut (cb + O_PAIRSO)
    #define pBuildOut ((device float4*)(cb + O_PBUILDO))
    #define pBuild ((device float4*)(cb + O_PBUILD))
    #define clip ((device atomic_uint*)(cb + O_CLIP))
    // Four bits a node: what was written for it since its last node pass (1 its contact
    // sums, 2 its position step, 4 a side, 8 its staging words). Only those nodes read and
    // clear those words, and only the kinds written (every node doing all of it on every
    // step after a contact was up to 22 device atomics a node).
    #define touched ((device atomic_uint*)(cb + O_TOUCH))
    // A hit node's slot: the smallest (hit, corner) number that names it; NONE between steps.
    #define hmap ((device atomic_uint*)(cb + O_HMAP))
    #define TOUCHK(n, k) atomic_fetch_or_explicit(&touched[(n) >> 3], (uint)(k) << (((n) & 7u) * 4u), memory_order_relaxed)
    #define hitBuf ((device float4*)(cb + O_HITS))
    #define stage ((device atomic_int*)(cb + O_SCRATCH))
    #define fPub ((device float*)(cb + O_SCRATCH + NN * 6))
    #define split ((device atomic_uint*)(cb + O_SCRATCH + NN * 9))
    #define masters ((device atomic_uint*)(cb + O_SCRATCH + NN * 10))
    #define cplV ((device float*)(cb + O_SCRATCH + NN * 11))
    #define hot ((device float4*)(cb + O_HOT))
    #define candBuf (cb + O_CAND)
    #define cluBox ((device float4*)(cb + O_CLU))
    #define conBuf (cb + O_CAND + 2048)
    #define liftBuf ((device float*)(cb + O_CAND + 3072))
    #define ent4 ((device const uint4*)(tab + T_ENT))

#if SURF
    // ([0..5] the body's box now, [6..7] the slots near it, [8] pairs, [9] hits, [10] hit
    // nodes, [11] status bits, [12..13] staged pairs (triangles', beams'), [14..15] tasks.)
    TGDECL(atomic_int, tgS, 16);
    #define surfRow ((device float4*)(cb + O_SURF))
    #define surfPair (cb + O_SURF + NSS * SRW * 4u)
    #define surfHit ((device float4*)(cb + O_SURF + NSS * SRW * 4u + SPAIR_CAP))
    #define surfBH ((device atomic_uint*)(cb + O_SURF + NSS * SRW * 4u + SPAIR_CAP + SHIT_CAP * 16u))
    #define surfHP (cb + O_SURF + NSS * SRW * 4u + SPAIR_CAP + SHIT_CAP * 16u + 64u)
    #define HPGX(h_) ((device atomic_int*)(surfHP + (h_) * HPW + 1u))
    #define HPKC(h_) ((device atomic_uint*)(surfHP + (h_) * HPW + 2u))
    #define HPREC(h_) ((device float4*)(surfHP + (h_) * HPW + 4u))
    #define taskG4 ((device float4*)(surfHP + 512u * HPW))
    // A slot's shape from the scalars' array.
    #define SURF_SHAPE(fs_, s_) FqShape fs_; { const uint o_ = SB(s_); \
        fs_.kind = as_type<uint>(tgG[o_ + 21u]); fs_.b = float3(0.0f, 0.0f, 0.0f); \
        fs_.c = float3(tgG[o_ + 6u], tgG[o_ + 7u], tgG[o_ + 8u]); \
        fs_.ax0 = float3(tgG[o_ + 9u], tgG[o_ + 10u], tgG[o_ + 11u]); fs_.ax1 = float3(tgG[o_ + 12u], tgG[o_ + 13u], tgG[o_ + 14u]); fs_.ax2 = float3(tgG[o_ + 15u], tgG[o_ + 16u], tgG[o_ + 17u]); \
        fs_.hext = float3(tgG[o_ + 18u], tgG[o_ + 19u], tgG[o_ + 20u]); \
        fs_.hull = 0u; fs_.nv = 0u; fs_.nf = 0u; fs_.ne = 0u; fs_.moved = 0u; fs_.off = float3(0.0f, 0.0f, 0.0f); }
#endif
#if RIGID
    TGDECL(atomic_uint, tgAct, 2);
    #define rigBody ((device float4*)(cb + O_RIG))
    #define rigCount ((device atomic_uint*)(cb + O_RIG + NB * BDW * 4))
    #define rigShape ((device float4*)(cb + O_RIG + NB * BDW * 4 + 64))
    #define rigSum ((device atomic_int*)(cb + O_RIG + NB * BDW * 4 + 64 + NSH * SHW * 4))
#endif
#if PAIR
    // ([0..11] the two bodies' boxes, [12..13] their top speeds, [14] near, [15..18] the
    // pass's list counts, [19..20] the owners' hits.)
    TGDECL(atomic_int, tgX, 24);
    uint xround = 0u, xlate = 0u, xstep = 0u;
    // (Groups in twos, a pair's bodies; the first one's block holds the pair's barrier
    // counter.)
    device atomic_uint* const xcnt = (device atomic_uint*)(big + (car & ~1u) * CAR_WORDS + O_MISC + 14);
#elif RIGID
    TGDECL(atomic_int, tgX, 6);
#endif
    // This lane's roles.
    const uint nd = tab[T_LANE + lid * 4];
    const uint entStart = tab[T_LANE + lid * 4 + 1];
    // (Leading scalar entries | entries before the first aero one << 8 | all entries << 16.)
    const uint entCounts = tab[T_LANE + lid * 4 + 2];
    const uint entScalar = entCounts & 255u, entPreAero = (entCounts >> 8) & 255u, entCount = entCounts >> 16;
    const uint tri = tab[T_LANE + lid * 4 + 3];
    const bool isNode = nd != NONE;
    const bool isTri = tri != NONE;
    const uint ni = isNode ? nd : 0u;
    V3 F0 = v3(nodeState[ni * 3 + 2]);
    // (The lane's own node's position and velocity stay in registers too: its own reads
    // of them were four threadgroup loads a substep.)
    V3 Pown = v3(0, 0, 0), Vown = v3(0, 0, 0);
    // (The node's mass in a register: three device loads a step otherwise. Friction and the
    // flags are read where a contact or an immovable node needs them.)
    const float massN = nodeStatic[ni].x;
    if (isNode)
    {
        float4 p = nodeState[ni * 3], v = nodeState[ni * 3 + 1];
        { Pown = (v3(p)); SETP(ni, Pown); }
        { Vown = (v3(v)); SETV(ni, Vown); }
        tgHeld[ni] = (atomic_load_explicit(&clip[ni * 4], memory_order_relaxed) != CLIP_EMPTY) ? 1 : 0;
    }
    // The slots' hot words: from the image the last dispatch left (six streamed loads), or
    // built from the records (the first dispatch after an upload).
    uint w0[RS];
    float up[RS];
#if REGMODE < 2
    float cc[RS];
#endif
#if REGMODE < 1
    float kk[RS], rest[RS];
#endif
    float G = 3.0e38f, L = 3.0e38f;
#if OWN
    // The owning layout (the fast variant): a node lane's slots 0 to 2 hold three of its
    // own node's beams; their part of the node's two sums is added here from registers
    // (one end is the lane's own node, so half the loads; the node's lists are three
    // entries shorter), before the listed entries: another order than the reference's.
    V3 cOwn = v3(0, 0, 0), fOwn = v3(0, 0, 0);
#endif
    const bool hotOk = misc[11] == 1u;
    if (hotOk)
    {
        uint4 hw = as_type<uint4>(hot[lid * 6 + 4]);
        float4 hg = hot[lid * 6 + 5];
        w0[0] = hw.x; w0[1] = hw.y; w0[2] = hw.z; w0[3] = hw.w;
        G = hg.x; L = hg.y;
        for (uint s = 0; s < RS; s++)
        {
            float4 hs = HOTS(s);
            up[s] = hs.w;
#if REGMODE < 2
            cc[s] = hs.y;
#endif
#if REGMODE < 1
            kk[s] = hs.x; rest[s] = hs.z;
#endif
        }
    }
    for (uint s = 0; s < RS && !hotOk; s++)
    {
        uint r = perm[s * LANES + lid];
        w0[s] = NONE;
        float4 hs = float4(0);
        up[s] = 0.0f;
#if REGMODE < 2
        cc[s] = 0.0f;
#endif
#if REGMODE < 1
        kk[s] = 0.0f; rest[s] = 0.0f;
#endif
        HOTS(s) = hs;
        if (r == NONE) continue;
        uint4 a = recA[r];
        uint kind = a.z & 0xFFu;
        float4 st0 = recState[r * 3], st1 = recState[r * 3 + 1], ax = recState[r * 3 + 2];
        if (kind == K_JPLAIN || kind == K_JSUPPORT)
        {
            w0[s] = a.x | a.y << 11 | (st1.w == 0.0f ? W_ALIVE : 0u) | (st1.z != 0.0f ? W_SLOPE : 0u) | (kind == K_JSUPPORT ? C_SUPPORT : C_PLAIN) << 30;
#if OWN
            if (isNode && s < 3u && a.y == nd) w0[s] |= W_OWNB;
#endif
            float4 kc = recK[r];
            hs = float4(kc.x, kc.y, st0.x, ax.x);
            if (st1.w == 0.0f)
            {
                G = fmin(G, fmin(st0.w, st1.x));
                if (kind == K_JSUPPORT) L = fmin(L, recB[r].y);
            }
        }
        else if (kind == K_JATRI)
        {
            float4 e1 = recE[r * 2];
            w0[s] = a.x | a.y << 10 | as_type<uint>(e1.x) << 20 | C_AERO << 30;
            hs = float4(e1.y, e1.z, e1.w, 0.0f);
        }
        else
        {
            w0[s] = r | C_GENERIC << 30;
            if (kind == K_JCOUPLER)
            {
                // The velocities the last beam pass read (the centred correction's v0).
                uint ci = tab[T_VIDX + r] - CPL_BASE;
                float4 va = nodeState[a.x * 3 + 1], vb = nodeState[a.y * 3 + 1];
                cplV[ci * 6] = va.x; cplV[ci * 6 + 1] = va.y; cplV[ci * 6 + 2] = va.z;
                cplV[ci * 6 + 3] = vb.x; cplV[ci * 6 + 4] = vb.y; cplV[ci * 6 + 5] = vb.z;
            }
        }
        HOTS(s) = hs;
        up[s] = hs.w;
#if REGMODE < 2
        cc[s] = hs.y;
#endif
#if REGMODE < 1
        kk[s] = hs.x; rest[s] = hs.z;
#endif
    }
    if (lid < NG) atomic_store_explicit(&masters[lid], NONE, memory_order_relaxed);
    if (lid < 4) atomic_store_explicit(&tgCount[lid], 0u, memory_order_relaxed);
    if (lid == 0)
    {
        tgFlags[F_SHIFT] = misc[0]; tgFlags[F_STATUS] = misc[1]; tgFlags[F_REBUILD] = misc[3];
        tgFlags[F_HITS] = misc[4]; tgFlags[F_BUILDS] = misc[5]; tgFlags[F_MOST] = misc[6];
        tgFlags[F_REC] = misc[7]; tgFlags[F_HELD] = misc[8]; tgFlags[F_GDIRTY] = misc[9];
        tgFlags[F_OUTER] = misc[13];
        tgFlags[F_FIRED] = 0u; tgFlags[F_BROKE] = misc[10];
        float4 pb = pBuild[REFNODE];
        tgPB[0] = pb.x; tgPB[1] = pb.y; tgPB[2] = pb.z;
    }
#if RIGID
    // begin_window: each shape's impulse, moment and peak cleared, its reference its pose.
    if (lid < NSH)
    {
        device float4* sh = rigShape + lid * SHW;
        device const float4* bd = rigBody + as_type<uint>(sh[0].x) * BDW;
        V3 p = v3(bd[0]) + qrot(bd[1], v3(0, 0, 0));
        sh[2] = f4(p, 0);
        sh[3] = float4(0); sh[4] = float4(0);
        float4 h0 = sh[0]; h0.w = 0.0f; sh[0] = h0;
    }
    if (lid < 6u) atomic_store_explicit(&tgX[lid], lid < 3u ? 0x7FFFFFFF : int(0x80000000), memory_order_relaxed);
#endif
#if PAIR
    if (lid < 24u) atomic_store_explicit(&tgX[lid], lid >= 12u ? 0 : ((lid % 6u) < 3u ? 0x7FFFFFFF : int(0x80000000)), memory_order_relaxed);
#endif
#if SURF
    if (lid < 16u) atomic_store_explicit(&tgS[lid], lid < 3u ? 0x7FFFFFFF : (lid < 6u ? int(0x80000000) : 0), memory_order_relaxed);
#endif
    BAR;

    for (uint frame = 0; frame < P.frames; frame++)
    for (uint step = 0; step < Z_STEPS; step++)
    {
#if RIGID
        // ---- R0: the proxies placed (explicit_rigid.rs begin_step) ----
        // (The body's box: a warp reduces the nodes from threadgroup memory.)
        if (lid < 32u)
        {
            V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
            for (uint j = lid; j < NN; j += 32u)
            {
                V3 x = v3(0, 0, 0) + TGP(j);
                lo = v3(rmin(lo.x, x.x), rmin(lo.y, x.y), rmin(lo.z, x.z));
                hi = v3(rmax(hi.x, x.x), rmax(hi.y, x.y), rmax(hi.z, x.z));
            }
            atomic_fetch_min_explicit(&tgX[0], ordF(lo.x), memory_order_relaxed);
            atomic_fetch_min_explicit(&tgX[1], ordF(lo.y), memory_order_relaxed);
            atomic_fetch_min_explicit(&tgX[2], ordF(lo.z), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgX[3], ordF(hi.x), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgX[4], ordF(hi.y), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgX[5], ordF(hi.z), memory_order_relaxed);
        }
        if (lid < NSH)
        {
            device float4* sh = rigShape + lid * SHW;
            device const float4* bd = rigBody + as_type<uint>(sh[0].x) * BDW;
            float4 brot = bd[1];
            V3 p = v3(bd[0]) + qrot(brot, v3(0, 0, 0));
            float4 q = qmul(brot, float4(0.0f, 0.0f, 0.0f, 1.0f));
            V3 lo, hi;
            float ident = 0.0f;
            if (as_type<uint>(sh[0].y) == 1u)
            {
                float r = sh[0].z;
                lo = p - v3(r, r, r); hi = p + v3(r, r, r);
            }
            else
            {
                if (qidentity(q)) ident = 1.0f; else q = qnormalize(q);
                float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
                float xx = q.x * x2, xy = q.x * y2, xz = q.x * z2, yy = q.y * y2, yz = q.y * z2, zz = q.z * z2;
                float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
                V3 a0 = v3(1.0f - (yy + zz), xy + wz, xz - wy), a1 = v3(xy - wz, 1.0f - (xx + zz), yz + wx), a2 = v3(xz + wy, yz - wx, 1.0f - (xx + yy));
                V3 hf3 = v3(sh[1]);
                V3 ext = v3((fabs(a0.x) * hf3.x + fabs(a1.x) * hf3.y) + fabs(a2.x) * hf3.z, (fabs(a0.y) * hf3.x + fabs(a1.y) * hf3.y) + fabs(a2.y) * hf3.z, (fabs(a0.z) * hf3.x + fabs(a1.z) * hf3.y) + fabs(a2.z) * hf3.z);
                lo = p - ext; hi = p + ext;
                sh[8] = f4(a0, 0); sh[9] = f4(a1, 0); sh[10] = f4(a2, 0);
            }
            V3 mx = v3(rmax(fabs(lo.x), fabs(hi.x)), rmax(fabs(lo.y), fabs(hi.y)), rmax(fabs(lo.z), fabs(hi.z)));
            float pad = 0.01f + 1e-5f * rmax(mx.x, rmax(mx.y, mx.z));
            sh[5] = f4(lo - v3(pad, pad, pad), ident);
            sh[6] = f4(hi + v3(pad, pad, pad), 0);
            sh[7] = f4(p, 0);
#if SURF
            {
                // surface_shape: the strict pose (fq::shape_pose), a box's axes, the cull box
                // (fq::bounds). A fast body would need the casts (§38.4), which are not here:
                // the status says so.
                device float4* row = surfRow + (NBOX + lid) * SRW;
                float4 rq;
                float3 sp = fqPose(fv(bd[0]), brot, float3(0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 1.0f), rq);
                FqShape fs;
                fs.b = float3(0.0f, 0.0f, 0.0f); fs.hull = 0u; fs.nv = 0u; fs.nf = 0u; fs.ne = 0u; fs.moved = 0u; fs.off = float3(0.0f, 0.0f, 0.0f);
                fs.c = sp;
                fs.ax0 = fqRot(rq, float3(1.0f, 0.0f, 0.0f)); fs.ax1 = fqRot(rq, float3(0.0f, 1.0f, 0.0f)); fs.ax2 = fqRot(rq, float3(0.0f, 0.0f, 1.0f));
                float ext, radius;
                if (as_type<uint>(sh[0].y) == 1u)
                {
                    fs.kind = FQ_SPH; fs.hext = float3(sh[0].z, 0.0f, 0.0f);
                    ext = sh[0].z; radius = sh[0].z;
                }
                else
                {
                    V3 hf = v3(sh[1]);
                    fs.kind = FQ_BOX; fs.hext = fv(hf);
                    ext = rmin(rmin(hf.x, hf.y), hf.z); radius = len3(hf);
                }
                float3 flo, fhi;
                fqBounds(fs, flo, fhi);
                row[0] = f4(fs.c, as_type<float>(fs.kind));
                row[1] = f4(fs.ax0, 0.0f); row[2] = f4(fs.ax1, 0.0f); row[3] = f4(fs.ax2, 0.0f);
                row[4] = f4(fs.hext, sh[0].x);
                row[5] = f4(flo, 0.0f); row[6] = f4(fhi, 0.0f);
                row[7] = float4(as_type<float>(lid), 0.0f, 0.0f, 0.0f);
                float travel = len3(v3(bd[2])) * H + (len3(v3(bd[3])) * (radius + 0.01f)) * H;
                if (travel > 0.5f * ext) atomic_fetch_or_explicit(&tgS[11], 8, memory_order_relaxed);
            }
#endif
        }
        if (lid < 2u) atomic_store_explicit(&tgAct[lid], 0u, memory_order_relaxed);
        BAR;
        if (lid < NSH)
        {
            // The shapes whose bounds meet the body's box: the only ones a node tests.
            device const float4* sh = rigShape + lid * SHW;
            float4 lo = sh[5], hi = sh[6];
            V3 clo = v3(unordF(atomic_load_explicit(&tgX[0], memory_order_relaxed)), unordF(atomic_load_explicit(&tgX[1], memory_order_relaxed)), unordF(atomic_load_explicit(&tgX[2], memory_order_relaxed)));
            V3 chi = v3(unordF(atomic_load_explicit(&tgX[3], memory_order_relaxed)), unordF(atomic_load_explicit(&tgX[4], memory_order_relaxed)), unordF(atomic_load_explicit(&tgX[5], memory_order_relaxed)));
            if (!(hi.x < clo.x || hi.y < clo.y || hi.z < clo.z || lo.x > chi.x || lo.y > chi.y || lo.z > chi.z))
                atomic_fetch_or_explicit(&tgAct[lid >> 5], 1u << (lid & 31u), memory_order_relaxed);
        }
        BAR;
        // (SKIP(13), timing only: no shape is tested and no proxy integrates.)
        const uint act0 = SKIP(13) ? 0u : atomic_load_explicit(&tgAct[0], memory_order_relaxed), act1 = SKIP(13) ? 0u : atomic_load_explicit(&tgAct[1], memory_order_relaxed);
        if (lid < 6u) atomic_store_explicit(&tgX[lid], lid < 3u ? 0x7FFFFFFF : int(0x80000000), memory_order_relaxed);
#endif
        // ---- S0: the node pass ----
        const bool fShift = tgFlags[F_SHIFT] != 0u, fHits = tgFlags[F_HITS] != 0u;
        const bool fRoll = tgFlags[F_REC] != 0u || tgFlags[F_HELD] != 0u;
        // (F_GDIRTY: 1 the groups' words hold an older step's set, to clear; 2 they hold the
        // last step's already, written as the last dispatch ended.)
        const bool fFired = tgFlags[F_FIRED] != 0u, fDirty = tgFlags[F_GDIRTY] == 1u, fBroke = tgFlags[F_BROKE] != 0u;
        if (fFired)
        {
            // The groups the last beam pass set off (the torsion bars' watches read them).
            if (lid < NG)
            {
                groupsLast[lid] = atomic_load_explicit(&masters[lid], memory_order_relaxed) != NONE ? 1u : 0u;
                atomic_store_explicit(&masters[lid], NONE, memory_order_relaxed);
            }
        }
        else if (fDirty && lid < NG) groupsLast[lid] = 0u;
        if (fBroke && isTri && triDead[tri] == 0u)
        {
            // Triangle breaks (spec §18), on the steps after a beam broke.
            uint4 tb = triBreaks[tri];
            bool dead = false;
            for (uint q = 0; q < 4; q++)
            {
                uint j = tb[q];
                if (j != NONE && recState[j * 3 + 1].w != 0.0f) dead = true;
            }
            if (dead) triDead[tri] = 1u;
        }
        bool inC = false;
        uint tw = 0u;
        const bool anyTouch = fRoll || fHits || fShift;
        if (isNode && !SKIP(0))
        {
            const float mass = massN;
            if (anyTouch) tw = (atomic_load_explicit(&touched[ni >> 3], memory_order_relaxed) >> ((ni & 7u) * 4u)) & 15u;
#if HAS_IMMOVABLE
            const uint nflags = as_type<uint>(nodeStatic[ni].z);
#else
            const uint nflags = 0u;
#endif
            V3 P0 = Pown, V0 = Vown;
            // (A node neither written nor holding a side has four empty words: no roll.)
            if (fRoll && ((tw & 4u) != 0u || tgHeld[ni] != 0))
            {
                uint m2 = atomic_load_explicit(&clip[ni * 4 + 2], memory_order_relaxed);
                uint m3 = atomic_load_explicit(&clip[ni * 4 + 3], memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4], m2, memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4 + 1], m3, memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4 + 2], CLIP_EMPTY, memory_order_relaxed);
                atomic_store_explicit(&clip[ni * 4 + 3], CLIP_EMPTY, memory_order_relaxed);
                tgHeld[ni] = m2 != CLIP_EMPTY ? 1 : 0;
            }
            if (fHits && (tw & 8u) != 0u)
            {
                for (uint q = 0; q < 6; q++) atomic_store_explicit(&stage[ni * 6 + q], 0, memory_order_relaxed);
                atomic_store_explicit(&split[ni], 0u, memory_order_relaxed);
            }
            if (fHits && (tw & 1u) != 0u)
            {
                int ax_ = atomic_exchange_explicit(&acc[ni * 3], 0, memory_order_relaxed);
                int ay_ = atomic_exchange_explicit(&acc[ni * 3 + 1], 0, memory_order_relaxed);
                int az_ = atomic_exchange_explicit(&acc[ni * 3 + 2], 0, memory_order_relaxed);
                F0 = F0 + v3(float(ax_), float(ay_), float(az_)) * (1.0f / 32.0f);
            }
            else F0 = F0 + v3(0, 0, 0);
            // A node in the ground or a static box goes on the step's list: its contact
            // rule (exponentials, divisions) runs a lane a node after the barrier, and it is
            // integrated after that. Every other node is integrated here.
            V3 x = v3(0, 0, 0) + P0;
            inC = P.ground > x.y;
#if NBOX > 0
            for (uint b = 0; b < NBOX; b++)
            {
                float3 lo = BOX_LO[b], hi = BOX_HI[b];
                if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z) inC = true;
            }
#endif
#if RIGID
            if ((act0 | act1) != 0u)
            {
                // The nodes inside each body's shapes, counted (the rule's mass split).
                for (uint k = 0; k < NSH; k++)
                {
                    if (((k < 32u ? act0 : act1) & (1u << (k & 31u))) == 0u) continue;
                    V3 nrm;
                    if (rigidFace(rigShape + k * SHW, x, nrm))
                    {
                        atomic_fetch_add_explicit(&rigCount[as_type<uint>(rigShape[k * SHW].x)], 1u, memory_order_relaxed);
                        inC = true;
                    }
                }
            }
#endif
            if (inC)
            {
                uint ci = atomic_fetch_add_explicit(&tgCount[3], 1u, memory_order_relaxed);
                conBuf[ci] = ni;
                fPub[ni * 3] = F0.x; fPub[ni * 3 + 1] = F0.y; fPub[ni * 3 + 2] = F0.z;
            }
            else
            {
                if ((nflags & NODE_IMMOVABLE) == 0u)
                {
                    V0 = V0 + (F0 / mass) * H;
                    P0 = P0 + V0 * H;
                    if (fShift)
                    {
                        // (An unwritten node's step is zero: the same add of +0.)
                        if ((tw & 2u) != 0u)
                        {
                            P0 = P0 + v3(float(SHL(ni * 3)), float(SHL(ni * 3 + 1)), float(SHL(ni * 3 + 2))) * (1.0f / 16777216.0f);
                            SHZ(ni * 3); SHZ(ni * 3 + 1); SHZ(ni * 3 + 2);
                        }
                        else P0 = P0 + v3(0, 0, 0);
                    }
                }
                else if (fShift && (tw & 2u) != 0u) { SHZ(ni * 3); SHZ(ni * 3 + 1); SHZ(ni * 3 + 2); }
                { Pown = (P0); SETP(ni, Pown); }
                { Vown = (V0); SETV(ni, Vown); }
            }
        }
        BAR;
        const uint ncon = atomic_load_explicit(&tgCount[3], memory_order_relaxed);
#ifdef EMU_TRACE
        if (lid == 0u) fprintf(stderr, "T car %u step %u: ground/box/proxy contact nodes %u, flags roll %d hits %d shift %d\n", car, step, ncon, (int)fRoll, (int)fHits, (int)fShift);
#endif
        if (ncon != 0u)
        {
            for (uint kc = lid; kc < min(ncon, (uint)NN); kc += LANES)
            {
                const uint n = conBuf[kc];
                float4 ns = nodeStatic[n];
                const float mq = ns.x, frq = ns.y;
                V3 Pq = TGP(n), Vq = TGV(n);
                V3 Fq = v3(fPub[n * 3], fPub[n * 3 + 1], fPub[n * 3 + 2]);
                float lift = 0.0f;
                V3 x = v3(0, 0, 0) + Pq;
                if (P.ground > x.y)
                {
                    float budget;
                    float depth = P.ground - x.y;
                    V3 c = contactResponse(Fq, Vq, mq, v3(0, 1, 0), GROUND, depth, frq, budget);
                    if (!isZero3(c)) lift = shiftDepth(depth, budget, Fq.y);
                    Fq = Fq + c;
                }
#if NBOX > 0
                for (uint b = 0; b < NBOX; b++)
                {
                    float3 lo = BOX_LO[b], hi = BOX_HI[b];
                    if (x.x > lo.x && x.x < hi.x && x.y > lo.y && x.y < hi.y && x.z > lo.z && x.z < hi.z)
                    {
                        V3 normal = v3(0.0f, 0.0f, -1.0f);
                        float dist = x.z - lo.z;
                        float dd[5] = {hi.z - x.z, x.x - lo.x, hi.x - x.x, x.y - lo.y, hi.y - x.y};
                        V3 nn[5] = {v3(0.0f, 0.0f, 1.0f), v3(-1.0f, 0.0f, 0.0f), v3(1.0f, 0.0f, 0.0f), v3(0.0f, -1.0f, 0.0f), v3(0.0f, 1.0f, 0.0f)};
                        for (uint q = 0; q < 5; q++)
                            if (dd[q] < dist) { dist = dd[q]; normal = nn[q]; }
                        float budget;
                        V3 c = contactResponse(Fq, Vq, mq, normal, BOXS, 0.0f, frq, budget);
                        Fq = Fq + c;
                    }
                }
#endif
#if RIGID
                if ((act0 | act1) != 0u)
                {
                    // node_contact_with: every shape the node is inside, in shape order.
                    V3 total = v3(0, 0, 0);
                    for (uint k = 0; k < NSH; k++)
                    {
                        if (((k < 32u ? act0 : act1) & (1u << (k & 31u))) == 0u) continue;
                        device const float4* sh = rigShape + k * SHW;
                        V3 normal;
                        if (!rigidFace(sh, x, normal)) continue;
                        const uint bi = as_type<uint>(sh[0].x);
                        device const float4* bd = rigBody + bi * BDW;
                        V3 bpos = v3(bd[0]);
                        V3 f = Fq + total;
                        V3 r = x - bpos;
                        V3 rn = cross3(r, normal);
                        float splitK = float(max(atomic_load_explicit(&rigCount[bi], memory_order_relaxed), 1u)) * rmax(bd[2].w, 1.0f);
                        float wb = splitK * (bd[0].w + dot3(rn, invInertiaMul(bd[1], v3(bd[4]), rn)));
                        float wn = 1.0f / mq;
                        float mred = 1.0f / (wn + wb);
                        float share = wn / (wn + wb);
                        V3 vb = v3(bd[2]) + cross3(v3(bd[3]), r);
                        V3 fc = contactForcePlain(f * share, Vq - vb, mred, normal, frq);
                        total = total + fc;
                        V3 react = -fc;
                        V3 mom = cross3(x - v3(sh[2]), react);
                        atomic_fetch_add_explicit(&rigSum[k * 6], int(round(react.x * 32.0f)), memory_order_relaxed);
                        atomic_fetch_add_explicit(&rigSum[k * 6 + 1], int(round(react.y * 32.0f)), memory_order_relaxed);
                        atomic_fetch_add_explicit(&rigSum[k * 6 + 2], int(round(react.z * 32.0f)), memory_order_relaxed);
                        atomic_fetch_add_explicit(&rigSum[k * 6 + 3], int(round(mom.x * 32.0f)), memory_order_relaxed);
                        atomic_fetch_add_explicit(&rigSum[k * 6 + 4], int(round(mom.y * 32.0f)), memory_order_relaxed);
                        atomic_fetch_add_explicit(&rigSum[k * 6 + 5], int(round(mom.z * 32.0f)), memory_order_relaxed);
                    }
                    Fq = Fq + total;
                }
#endif
                fPub[n * 3] = Fq.x; fPub[n * 3 + 1] = Fq.y; fPub[n * 3 + 2] = Fq.z;
                liftBuf[n] = lift;
            }
            BAR;
            if (isNode && inC)
            {
                V3 P0 = Pown, V0 = Vown;
                F0 = v3(fPub[ni * 3], fPub[ni * 3 + 1], fPub[ni * 3 + 2]);
                const float lift = liftBuf[ni];
#if HAS_IMMOVABLE
                const uint nflags = as_type<uint>(nodeStatic[ni].z);
#else
                const uint nflags = 0u;
#endif
                if ((nflags & NODE_IMMOVABLE) == 0u)
                {
                    V0 = V0 + (F0 / massN) * H;
                    P0 = P0 + V0 * H;
                    if (fShift) P0 = P0 + ((tw & 2u) != 0u ? v3(float(SHL(ni * 3)), float(SHL(ni * 3 + 1)), float(SHL(ni * 3 + 2))) * (1.0f / 16777216.0f) : v3(0, 0, 0));
                    if (lift != 0.0f) P0.y = P0.y + lift;
                }
                if (fShift && (tw & 2u) != 0u) { SHZ(ni * 3); SHZ(ni * 3 + 1); SHZ(ni * 3 + 2); }
                { Pown = (P0); SETP(ni, Pown); }
                { Vown = (V0); SETV(ni, Vown); }
            }
            if (lid == 3) atomic_store_explicit(&tgCount[3], 0u, memory_order_relaxed);
        }
        if (lid == 0) atomic_store_explicit(&tgCount[0], 0u, memory_order_relaxed);
        if (lid == 2) atomic_store_explicit(&tgCount[2], 0u, memory_order_relaxed);
        if (lid == 1) atomic_store_explicit(&tgCount[1], 0u, memory_order_relaxed);
#if SURF
        if (lid < 16u && lid != 11u) atomic_store_explicit(&tgS[lid], lid < 3u ? 0x7FFFFFFF : (lid < 6u ? int(0x80000000) : 0), memory_order_relaxed);
#endif
        BAR;
        // (The node pass has read the written nodes' bits: cleared before this step's writers.)
        if (anyTouch && lid < ((NN + 7u) >> 3)) atomic_store_explicit(&touched[lid], 0u, memory_order_relaxed);
        if (lid == 0)
        {
            tgFlags[F_HELD] = tgFlags[F_REC]; tgFlags[F_REC] = 0u; tgFlags[F_HITS] = 0u; tgFlags[F_SHIFT] = 0u;
            tgFlags[F_GDIRTY] = (fFired || tgFlags[F_GDIRTY] == 2u) ? 1u : 0u; tgFlags[F_FIRED] = 0u; tgFlags[F_BROKE] = 0u;
        }

        // ---- S1: the centred damping's scalars (last beam pass's slopes) ----
#if SURF
        // (The body's box for this step's surface pass: the nodes are where the node pass
        // left them. A warp reduces them from threadgroup memory.)
        if (lid < 32u && !SKIP(21))
        {
            V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
            for (uint j = lid; j < NN; j += 32u)
            {
                V3 x = v3(0, 0, 0) + TGP(j);
                lo = v3(rmin(lo.x, x.x), rmin(lo.y, x.y), rmin(lo.z, x.z));
                hi = v3(rmax(hi.x, x.x), rmax(hi.y, x.y), rmax(hi.z, x.z));
            }
            atomic_fetch_min_explicit(&tgS[0], ordF(lo.x), memory_order_relaxed);
            atomic_fetch_min_explicit(&tgS[1], ordF(lo.y), memory_order_relaxed);
            atomic_fetch_min_explicit(&tgS[2], ordF(lo.z), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgS[3], ordF(hi.x), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgS[4], ordF(hi.y), memory_order_relaxed);
            atomic_fetch_max_explicit(&tgS[5], ordF(hi.z), memory_order_relaxed);
        }
#endif
#if OWN
        cOwn = v3(0, 0, 0);
#endif
        for (uint s = 0; s < RS && !SKIP(1); s++)
        {
            uint w = w0[s];
            // (The leading slots hold plain beams on every lane: no class to test there.)
            const bool sp = s < FULL_PLAIN_SLOTS;
            if (!sp && w == NONE) continue;
            const uint cls = sp ? C_PLAIN : (w >> 30);
            if (cls == C_AERO) continue;
            if (cls != C_GENERIC)
            {
                float gc = 0.0f;
                if ((w & W_SLOPE) != 0u)
                {
                    uint a = w & 2047u, b = (w >> 11) & 2047u;
#if OWN
                    V3 d, dvel, own = v3(0, 0, 0);
                    const bool owned = isNode && s < 3u;
                    if (owned)
                    {
                        const bool ob = (w & W_OWNB) != 0u;
                        const uint o = ob ? a : b;
                        V3 po = TG3P(o), vo = TG3V(o);
                        own = Pown - po;
                        d = ob ? po - Pown : own;
                        dvel = ob ? vo - Vown : Vown - vo;
                    }
                    else { d = TG3P(a) - TG3P(b); dvel = TG3V(a) - TG3V(b); }
                    float inv = exactLength(dot3(d, d)).y;
                    float du = dot3(dvel, d) * inv - up[s];
#else
                    V3 d = TG3P(a) - TG3P(b);
                    float inv = exactLength(dot3(d, d)).y;
                    float du = dot3(TG3V(a) - TG3V(b), d) * inv - up[s];
#endif
#if REGMODE < 2
                    gc = ((-0.5f * cc[s]) * du) * inv;
#else
                    gc = ((-0.5f * HOTS(s).y) * du) * inv;
#endif
#if OWN
                    if (owned) cOwn = cOwn + own * gc;
#endif
                }
                tgG[s * LANES + lid] = gc;
                continue;
            }
            uint r = w & 0xFFFFFFu;
            uint4 a = recA[r];
            uint kind = a.z & 0xFFu;
            float slope = recState[r * 3 + 1].z;
            if (kind == K_JTORSION)
            {
                uint4 a2 = recA[r + 1];
                uint vi = tab[T_VIDX + r];
                uint ids[4] = {a.x, a.y, a2.x, a2.y};
                V3 o[4] = {v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0)};
                if (slope != 0.0f)
                {
                    float4 ax = recState[r * 3 + 2];
                    Jac J = dihedralJacobian(TGP(ids[0]), TGP(ids[1]), TGP(ids[2]), TGP(ids[3]));
                    float dr = (((dot3(J.g[0], TGV(ids[0])) - ax.x) + (dot3(J.g[1], TGV(ids[1])) - ax.y)) +
                                (dot3(J.g[2], TGV(ids[2])) - ax.z)) + (dot3(J.g[3], TGV(ids[3])) - ax.w);
                    float t = (-0.5f * slope) * dr;
                    for (uint q = 0; q < 4; q++) o[q] = J.g[q] * t;
                }
                for (uint q = 0; q < 4; q++) PUTV(vi + q, o[q]);
            }
            else if (kind == K_JCOUPLER)
            {
                uint vi = tab[T_VIDX + r];
                uint ci = vi - CPL_BASE;
                V3 push = v3(0, 0, 0);
                if (slope != 0.0f)
                {
                    V3 da = TGV(a.x) - v3(cplV[ci * 6], cplV[ci * 6 + 1], cplV[ci * 6 + 2]);
                    V3 db = TGV(a.y) - v3(cplV[ci * 6 + 3], cplV[ci * 6 + 4], cplV[ci * 6 + 5]);
                    push = (da - db) * (-0.5f * slope);
                }
                PUTV(vi, push);
            }
            else
            {
                float gc = 0.0f;
                if (slope != 0.0f)
                {
                    V3 d = TGP(a.x) - TGP(a.y);
                    float inv = exactLength(dot3(d, d)).y;
                    float du = dot3(TGV(a.x) - TGV(a.y), d) * inv - recState[r * 3 + 2].x;
                    gc = ((-0.5f * slope) * du) * inv;
                }
                tgG[s * LANES + lid] = gc;
            }
        }
        BAR;

        // ---- C1: the centred correction gathered per node (kept in registers) ----
        V3 Pn = v3(0, 0, 0), Vn = v3(0, 0, 0);
        if (isNode && !SKIP(2))
        {
            V3 Ps = Pown;
#if OWN
            V3 c = cOwn;
#else
            V3 c = v3(0, 0, 0);
#endif
            for (uint k = 0; k < MAXENT; k += 4u)
            {
                // The leading scalar entries, four a load: (x_self − x_other)·g.
                if (k >= entScalar) break;
                uint4 cur = ent4[(entStart >> 2) + (k >> 2)];
                ENT_ADD(c, cur.x);
                if (k + 1u < entScalar) ENT_ADD(c, cur.y);
                if (k + 2u < entScalar) ENT_ADD(c, cur.z);
                if (k + 3u < entScalar) ENT_ADD(c, cur.w);
            }
            // (The entries from the first vector one, less the aero triangles' at the end,
            // which have no centred part.)
            uint4 cur = uint4(0u);
            for (uint k = entScalar; k < MAXENT; k++)
            {
                if (k >= entPreAero) break;
                if (k == entScalar || (k & 3u) == 0u) cur = ent4[(entStart >> 2) + (k >> 2)];
                uint e = cur[k & 3u];
                if ((e >> 31) == 0u)
                {
                    uint o3 = e & 2047u;
                    c = c + (Ps - TGP(o3)) * tgG[e >> 11];
                }
                else
                {
                    uint vi = e & 0xFFFu;
                    V3 vv = GETV(vi);
                    c = c + ((e & (1u << 30)) != 0u ? -vv : vv);
                }
            }
            Pn = Ps; Vn = Vown;
#if HAS_IMMOVABLE
            if ((as_type<uint>(nodeStatic[ni].z) & NODE_IMMOVABLE) == 0u)
#endif
            {
                V3 dv = (c / massN) * H;
                Vn = Vn + dv;
                Pn = Pn + dv * H;
            }
            atomic_fetch_max_explicit(&tgCount[0], as_type<uint>(len3(Vn)), memory_order_relaxed);
        }
        BAR;
        // ---- C2: published ----
        if (isNode && !SKIP(2))
        {
            { Pown = (Pn); SETP(ni, Pown); }
            { Vown = (Vn); SETV(ni, Vown); }
        }
        BAR;

        // ---- S3: the beam pass ----
        const float vmax = as_type<float>(atomic_load_explicit(&tgCount[0], memory_order_relaxed));
#if OWN
        fOwn = v3(0, 0, 0);
#endif
        for (uint s = 0; s < RS && !SKIP(3); s++)
        {
            uint w = w0[s];
            // (The leading slots hold plain beams on every lane: no class to test there.)
            const bool sp = s < FULL_PLAIN_SLOTS;
            if (!sp && w == NONE) continue;
            const uint cls = sp ? C_PLAIN : (w >> 30);
            if (cls == C_AERO)
            {
                uint a = w & 1023u, b = (w >> 10) & 1023u, c = (w >> 20) & 1023u;
                float4 ha = HOTS(s);
                V3 f = aeroTri(TGP(a), TGP(b), TGP(c), TGV(a), TGV(b), TGV(c), ha.x, ha.y, ha.z);
                PUTV(lid - AERO_LO, f);
                continue;
            }
            if (cls != C_GENERIC)
            {
                if ((w & W_ALIVE) == 0u)
                {
                    tgG[s * LANES + lid] = 0.0f;
                    w0[s] = w & ~W_SLOPE;
                    continue;
                }
                uint a = w & 2047u, b = (w >> 11) & 2047u;
#if OWN
                V3 pa, pb, va, vb, own = v3(0, 0, 0);
                const bool owned = isNode && s < 3u;
                if (owned)
                {
                    const bool ob = (w & W_OWNB) != 0u;
                    const uint o = ob ? a : b;
                    V3 po = TG3P(o), vo = TG3V(o);
                    own = Pown - po;
                    pa = ob ? po : Pown; pb = ob ? Pown : po;
                    va = ob ? vo : Vown; vb = ob ? Vown : vo;
                }
                else { pa = TG3P(a); pb = TG3P(b); va = TG3V(a); vb = TG3V(b); }
#else
                V3 pa = TG3P(a), pb = TG3P(b), va = TG3V(a), vb = TG3V(b);
#endif
                V3 d = pa - pb;
                float2 li = exactLength(dot3(d, d));
#if REGMODE >= 1
                float4 hs = HOTS(s);
                const float kS = hs.x, rS = hs.z;
#else
                const float kS = kk[s], rS = rest[s];
#endif
#if REGMODE >= 2
                const float cS = hs.y;
#else
                const float cS = cc[s];
#endif
                float e = li.x - rS;
                float u = dot3(va - vb, d) * li.y;
                float k = kS;
                float damp = cS * u;
                bool slow = false, slopeOn = true;
                if (cls == C_SUPPORT && e > 0.0f)
                {
                    slow = li.x > L;
                    k = 0.0f; damp = 0.0f; slopeOn = false;
                }
                float f = -k * e - damp;
                if (fabs(f) > G) slow = true;
                if (slow)
                {
                    // The step it may yield or break: the whole rule from its records.
                    Slow o = slowScalar(perm[s * LANES + lid], recA, recK, recB, recE, recState, masters, pa, pb, va, vb);
#if REGMODE < 1
                    rest[s] = o.rest;
#else
                    hs.z = o.rest;
                    HOTS(s) = hs;
#endif
                    up[s] = o.u;
                    w0[s] = (w & ~(W_ALIVE | W_SLOPE | W_DEFORM)) | ((o.bits & 1u) != 0u ? W_ALIVE : 0u) | ((o.bits & 2u) != 0u ? W_SLOPE : 0u) | ((o.bits & 4u) != 0u ? W_DEFORM : 0u);
                    if ((o.bits & 8u) != 0u) tgFlags[F_FIRED] = 1u;
                    if ((o.bits & 16u) != 0u) tgFlags[F_BROKE] = 1u;
                    tgG[s * LANES + lid] = o.g;
#if OWN
                    if (owned) fOwn = fOwn + own * o.g;
#endif
                    continue;
                }
                up[s] = u;
                w0[s] = slopeOn ? (w | W_SLOPE) : (w & ~W_SLOPE);
                tgG[s * LANES + lid] = f * li.y;
#if OWN
                if (owned) fOwn = fOwn + own * (f * li.y);
#endif
                continue;
            }
            // The car's few special elements, from their records every step.
            uint r = w & 0xFFFFFFu;
            uint4 a = recA[r];
            uint kind = a.z & 0xFFu;
            int group = as_type<int>(a.w);
            V3 pa = TGP(a.x), pb = TGP(a.y), va = TGV(a.x), vb = TGV(a.y);
            if (kind == K_JTORSION)
            {
                uint vi = tab[T_VIDX + r];
                float4 s0 = recState[r * 3], s1 = recState[r * 3 + 1];
                if (s1.w != 0.0f)
                {
                    for (uint q = 0; q < 4; q++) PUTV(vi + q, v3(0, 0, 0));
                    if (s1.z != 0.0f) { s1.z = 0.0f; recState[r * 3 + 1] = s1; }
                    continue;
                }
                float4 kc = recK[r];
                uint4 a2 = recA[r + 1];
                V3 x[4] = {pa, pb, TGP(a2.x), TGP(a2.y)};
                V3 v[4] = {va, vb, TGV(a2.x), TGV(a2.y)};
                float4 we = recE[r * 2];
                uint3 wt = uint3(as_type<uint>(we.x), as_type<uint>(we.y), as_type<uint>(we.z));
                bool watch = (wt.x != 0u && groupsLast[wt.x] != 0u) || (wt.y != 0u && groupsLast[wt.y] != 0u) ||
                             (wt.z != 0u && groupsLast[wt.z] != 0u);
                V3 o[4] = {v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0), v3(0, 0, 0)};
                bool broke = false;
                Jac J = dihedralJacobian(x[0], x[1], x[2], x[3]);
                float gv[4];
                for (uint q = 0; q < 4; q++) gv[q] = dot3(J.g[q], v[q]);
                if (watch) { broke = true; s1.y = 0.0f; }
                else
                {
                    float phi = dihedral(x[0], x[1], x[2], x[3]);
                    float rate = ((gv[0] + gv[1]) + gv[2]) + gv[3];
                    float dd = wrapAngle(phi - s0.x);
                    if (kc.x * fabs(dd) > s0.y)
                    {
                        dd = copysign(s0.y / kc.x, dd);
                        s0.x = wrapAngle(phi - dd);
                    }
                    float t = -kc.x * dd - kc.y * rate;
                    s1.y = t;
                    if (fabs(t) > s1.x) broke = true;
                    else for (uint q = 0; q < 4; q++) o[q] = J.g[q] * t;
                }
                if (broke) { s1.w = 1.0f; recState[(r + 1) * 3 + 1].w = 1.0f; tgFlags[F_BROKE] = 1u; }
                s1.z = broke ? 0.0f : kc.y;
                recState[r * 3] = s0; recState[r * 3 + 1] = s1;
                recState[r * 3 + 2] = float4(gv[0], gv[1], gv[2], gv[3]);
                for (uint q = 0; q < 4; q++) PUTV(vi + q, o[q]);
                if (broke && group > 0) { atomic_fetch_min_explicit(&masters[group], r, memory_order_relaxed); tgFlags[F_FIRED] = 1u; }
            }
            else if (kind == K_JCOUPLER)
            {
                uint vi = tab[T_VIDX + r];
                uint ci = vi - CPL_BASE;
                float4 s1 = recState[r * 3 + 1];
                cplV[ci * 6] = va.x; cplV[ci * 6 + 1] = va.y; cplV[ci * 6 + 2] = va.z;
                cplV[ci * 6 + 3] = vb.x; cplV[ci * 6 + 4] = vb.y; cplV[ci * 6 + 5] = vb.z;
                if (s1.w != 0.0f)
                {
                    PUTV(vi, v3(0, 0, 0));
                    if (s1.z != 0.0f) { s1.z = 0.0f; recState[r * 3 + 1] = s1; }
                    continue;
                }
                float4 kc = recK[r];
                V3 f = (-kc.x) * (pa - pb) - kc.y * (va - vb);
                float m = len3(f);
                s1.y = m;
                bool broke = m > s1.x;
                if (broke) { s1.w = 1.0f; tgFlags[F_BROKE] = 1u; }
                s1.z = broke ? 0.0f : kc.y;
                recState[r * 3 + 1] = s1;
                PUTV(vi, broke ? v3(0, 0, 0) : f);
                if (broke && group > 0) { atomic_fetch_min_explicit(&masters[group], r, memory_order_relaxed); tgFlags[F_FIRED] = 1u; }
            }
            else
            {
                float4 s1 = recState[r * 3 + 1];
                if (s1.w != 0.0f)
                {
                    tgG[s * LANES + lid] = 0.0f;
                    if (s1.z != 0.0f) { s1.z = 0.0f; recState[r * 3 + 1] = s1; }
                    continue;
                }
                Slow o = slowScalar(r, recA, recK, recB, recE, recState, masters, pa, pb, va, vb);
                if ((o.bits & 4u) != 0u) w0[s] = w | W_DEFORM;
                if ((o.bits & 8u) != 0u) tgFlags[F_FIRED] = 1u;
                if ((o.bits & 16u) != 0u) tgFlags[F_BROKE] = 1u;
                tgG[s * LANES + lid] = o.g;
            }
        }
        // Whether the pair list still holds every pair the box test can pass: each node's
        // move since the build, less the reference node's, at most D an axis; the list's
        // box covers the reach now and 2·D.
        if (isNode && !SKIP(6))
        {
            float4 pb = pBuild[ni];
            V3 rel = (Pown - v3(pb)) - (TGP(REFNODE) - v3(tgPB[0], tgPB[1], tgPB[2]));
            float D = fmax(fmax(fabs(rel.x), fabs(rel.y)), fabs(rel.z));
            float need = ((CR + 2.0f * vmax * H) + CLIP_DEPTH) + 2.0f * D + LIST_EPS;
            if (!(need <= LIST_REACH)) tgFlags[F_REBUILD] = 1u;
        }
        BAR;

        // ---- break groups set off in this pass (rare) ----
        if (tgFlags[F_FIRED] != 0u)
        {
            for (uint s = 0; s < RS; s++)
            {
                uint w = w0[s];
                if (w == NONE || (w >> 30) == C_AERO) continue;
                uint cls = w >> 30;
                uint r = cls == C_GENERIC ? (w & 0xFFFFFFu) : perm[s * LANES + lid];
                uint4 a = recA[r];
                int group = as_type<int>(a.w);
                if (group == 0) continue;
                uint m = atomic_load_explicit(&masters[abs(group)], memory_order_relaxed);
                if (m == NONE) continue;
                // cascade(): every member broken; one after the first to break in this pass
                // was skipped (no force, no slope, no deform of its own).
                uint kind = a.z & 0xFFu;
                bool ext = ((a.z >> 16) & 1u) != 0u;
                float4 s1 = recState[r * 3 + 1];
                if (r > m)
                {
                    s1.z = 0.0f;
                    w = w & ~(W_SLOPE | W_DEFORM);
                    if (kind == K_JTORSION) { uint vi = tab[T_VIDX + r]; for (uint q = 0; q < 4; q++) PUTV(vi + q, v3(0, 0, 0)); }
                    else if (kind == K_JCOUPLER) { PUTV(tab[T_VIDX + r], v3(0, 0, 0)); }
                    else tgG[s * LANES + lid] = 0.0f;
                }
                s1.w = 1.0f;
                recState[r * 3 + 1] = s1;
                if (kind == K_JTORSION) recState[(r + 1) * 3 + 1].w = 1.0f;
                w = w & ~W_ALIVE;
                // cascade_ext(): a member with an extension record triggers its deform groups.
                if (ext && (as_type<uint>(recB[r].w) & EXT_DEFORM) != 0u) w |= W_DEFORM;
                w0[s] = w;
            }
#if OWN
            // (A beam a group just broke adds nothing: the lane's own part again, from the
            // scalars as they now stand.)
            if (isNode)
            {
                #define OWNPART(q) ((Pown - TG3P((w0[q] & W_OWNB) != 0u ? (w0[q] & 2047u) : ((w0[q] >> 11) & 2047u))) * tgG[(q) * LANES + lid])
                fOwn = (OWNPART(0) + OWNPART(1)) + OWNPART(2);
            }
#endif
            if (lid == 0) tgFlags[F_BROKE] = 1u;
            BAR;
        }
        // ---- the pair list built (rare) ----
#ifdef EMU_TRACE
        if (lid == 0u && tgFlags[F_REBUILD] != 0u) fprintf(stderr, "T car %u step %u: pair list rebuilt\n", car, step);
#endif
        if (tgFlags[F_REBUILD] != 0u && !SKIP(14))
        {
#if LIST2
            // Two lists. The wide one (every pair within LIST_REACH_OUT, built by the
            // clusters) is rebuilt only when it can no longer hold every pair the working
            // list's box could reach: each node's move against the reference node's since
            // the wide build, at most D an axis, and the working reach and 2·D inside the
            // wide reach. The working list is then the wide list's pairs inside the working
            // box now: the same pairs, in the same order, as building it from the clusters
            // (which costs a warp every cluster's nodes for its one lane that needs them).
            if (isNode)
            {
                V3 rel = (Pown - v3(pBuildOut[ni])) - (TGP(REFNODE) - v3(pBuildOut[NN]));
                float D = fmax(fmax(fabs(rel.x), fabs(rel.y)), fabs(rel.z));
                atomic_fetch_max_explicit(&tgCount[1], as_type<uint>(D), memory_order_relaxed);
            }
            BAR;
            const float dOut = as_type<float>(atomic_load_explicit(&tgCount[1], memory_order_relaxed));
            const bool wideNow = tgFlags[F_OUTER] == 0u || !((LIST_REACH + 2.0f * dOut) + LIST_EPS <= LIST_REACH_OUT);
#ifdef EMU_TRACE
            if (lid == 0u && wideNow) fprintf(stderr, "T car %u step %u: wide list rebuilt\n", car, step);
#endif
            if (wideNow)
            {
                // Two levels (the same pairs as testing every contacter): each cluster of
                // contacters' box now, then a triangle tests only the clusters its list box
                // meets, a word of allowed bits and eight loads of node indices a cluster.
                if (lid < NCLU)
                {
                    V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
                    device const uint4* cl4 = (device const uint4*)(tab + T_CLU + lid * 32u);
                    for (uint jj = 0; jj < 8u; jj++)
                    {
                        uint4 q = cl4[jj];
                        for (uint e = 0; e < 4u; e++)
                        {
                            uint h = q[e];
                            if (h == NONE) continue;
                            V3 p = v3(0, 0, 0) + TGP(h);
                            lo = v3(rmin(lo.x, p.x), rmin(lo.y, p.y), rmin(lo.z, p.z));
                            hi = v3(rmax(hi.x, p.x), rmax(hi.y, p.y), rmax(hi.z, p.z));
                        }
                    }
                    cluBox[lid * 2u] = f4(lo, 0);
                    cluBox[lid * 2u + 1u] = f4(hi, 0);
                }
                if (isNode) pBuildOut[ni] = f4(Pown, 0);
                if (lid == 0) pBuildOut[NN] = f4(TGP(REFNODE), 0);
                BAR;
                if (isTri)
                {
                    uint4 tn = tris[tri];
                    V3 p0 = v3(0, 0, 0) + TGP(tn.x), p1 = v3(0, 0, 0) + TGP(tn.y), p2 = v3(0, 0, 0) + TGP(tn.z);
                    V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - LIST_REACH_OUT, rmin(rmin(p0.y, p1.y), p2.y) - LIST_REACH_OUT, rmin(rmin(p0.z, p1.z), p2.z) - LIST_REACH_OUT);
                    V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + LIST_REACH_OUT, rmax(rmax(p0.y, p1.y), p2.y) + LIST_REACH_OUT, rmax(rmax(p0.z, p1.z), p2.z) + LIST_REACH_OUT);
                    uint n = 0u;
                    if (triDead[tri] == 0u)
                    {
                        for (uint k = 0; k < NCLU; k++)
                        {
                            float4 clo = cluBox[k * 2u], chi = cluBox[k * 2u + 1u];
                            if (chi.x < lo.x || chi.y < lo.y || chi.z < lo.z || clo.x > hi.x || clo.y > hi.y || clo.z > hi.z) continue;
                            uint bitsw = allowed[tri * NCLU + k];
                            if (bitsw == 0u) continue;
                            device const uint4* cl4 = (device const uint4*)(tab + T_CLU + k * 32u);
                            for (uint jj = 0; jj < 8u; jj++)
                            {
                                if (((bitsw >> (jj * 4u)) & 15u) == 0u) continue;
                                uint4 q = cl4[jj];
                                for (uint e = 0; e < 4u; e++)
                                {
                                    if ((bitsw & (1u << (jj * 4u + e))) == 0u) continue;
                                    uint h = q[e];
                                    V3 p = v3(0, 0, 0) + TGP(h);
                                    if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) continue;
                                    if (n < PROW - 2u) pairsOut[tri * PROW + 2u + n] = h;
                                    n++;
                                }
                            }
                        }
                    }
                    if (n > PROW - 2u) { tgFlags[F_STATUS] = tgFlags[F_STATUS] | 1u; n = PROW - 2u; }
                    pairsOut[tri * PROW] = n;
                    pairsOut[tri * PROW + 1u] = tn.x | tn.y << 10 | tn.z << 20;
                }
                BAR;

            }
            if (isNode) pBuild[ni] = f4(Pown, 0);
            if (lid == 0) { V3 pr = TGP(REFNODE); tgPB[0] = pr.x; tgPB[1] = pr.y; tgPB[2] = pr.z; }
            if (lid == 1) atomic_store_explicit(&tgCount[1], 0u, memory_order_relaxed);
            if (isTri)
            {
                uint4 tn = tris[tri];
                V3 p0 = v3(0, 0, 0) + TGP(tn.x), p1 = v3(0, 0, 0) + TGP(tn.y), p2 = v3(0, 0, 0) + TGP(tn.z);
                V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - LIST_REACH, rmin(rmin(p0.y, p1.y), p2.y) - LIST_REACH, rmin(rmin(p0.z, p1.z), p2.z) - LIST_REACH);
                V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + LIST_REACH, rmax(rmax(p0.y, p1.y), p2.y) + LIST_REACH, rmax(rmax(p0.z, p1.z), p2.z) + LIST_REACH);
                device const uint4* out4 = (device const uint4*)(pairsOut + tri * PROW);
                uint4 cur = out4[0];
                const uint nOut = triDead[tri] == 0u ? min(cur.x, (uint)(PROW - 2u)) : 0u;
                uint n = 0u;
                #define LIST_ONE(HV) { const uint h = (HV); V3 p = v3(0, 0, 0) + TGP(h); if (!(p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z)) { pairs[tri * PROW + 2u + n] = h; n++; } }
                if (nOut > 0u) LIST_ONE(cur.z)
                if (nOut > 1u) LIST_ONE(cur.w)
                for (uint k = 2u; k < nOut; k += 4u)
                {
                    cur = out4[(k + 2u) >> 2];
                    LIST_ONE(cur.x)
                    if (k + 1u < nOut) LIST_ONE(cur.y)
                    if (k + 2u < nOut) LIST_ONE(cur.z)
                    if (k + 3u < nOut) LIST_ONE(cur.w)
                }
                pairs[tri * PROW] = n;
                pairs[tri * PROW + 1u] = tn.x | tn.y << 10 | tn.z << 20;
            }
            BAR;
            if (lid == 0) { tgFlags[F_REBUILD] = 0u; tgFlags[F_BUILDS] += 1u; if (wideNow) tgFlags[F_OUTER] = 1u; }
#else
            // Two levels (the same pairs as testing every contacter): each cluster of
            // contacters' box now, then a triangle tests only the clusters its list box
            // meets, a word of allowed bits and eight loads of node indices a cluster.
            if (lid < NCLU)
            {
                V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
                device const uint4* cl4 = (device const uint4*)(tab + T_CLU + lid * 32u);
                for (uint jj = 0; jj < 8u; jj++)
                {
                    uint4 q = cl4[jj];
                    for (uint e = 0; e < 4u; e++)
                    {
                        uint h = q[e];
                        if (h == NONE) continue;
                        V3 p = v3(0, 0, 0) + TGP(h);
                        lo = v3(rmin(lo.x, p.x), rmin(lo.y, p.y), rmin(lo.z, p.z));
                        hi = v3(rmax(hi.x, p.x), rmax(hi.y, p.y), rmax(hi.z, p.z));
                    }
                }
                cluBox[lid * 2u] = f4(lo, 0);
                cluBox[lid * 2u + 1u] = f4(hi, 0);
            }
            if (isNode) pBuild[ni] = f4(Pown, 0);
            if (lid == 0) { V3 pr = TGP(REFNODE); tgPB[0] = pr.x; tgPB[1] = pr.y; tgPB[2] = pr.z; }
            BAR;
            if (isTri)
            {
                uint4 tn = tris[tri];
                V3 p0 = v3(0, 0, 0) + TGP(tn.x), p1 = v3(0, 0, 0) + TGP(tn.y), p2 = v3(0, 0, 0) + TGP(tn.z);
                V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - LIST_REACH, rmin(rmin(p0.y, p1.y), p2.y) - LIST_REACH, rmin(rmin(p0.z, p1.z), p2.z) - LIST_REACH);
                V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + LIST_REACH, rmax(rmax(p0.y, p1.y), p2.y) + LIST_REACH, rmax(rmax(p0.z, p1.z), p2.z) + LIST_REACH);
                uint n = 0u;
                if (triDead[tri] == 0u)
                {
                    for (uint k = 0; k < NCLU; k++)
                    {
                        float4 clo = cluBox[k * 2u], chi = cluBox[k * 2u + 1u];
                        if (chi.x < lo.x || chi.y < lo.y || chi.z < lo.z || clo.x > hi.x || clo.y > hi.y || clo.z > hi.z) continue;
                        uint bitsw = allowed[tri * NCLU + k];
                        if (bitsw == 0u) continue;
                        device const uint4* cl4 = (device const uint4*)(tab + T_CLU + k * 32u);
                        for (uint jj = 0; jj < 8u; jj++)
                        {
                            if (((bitsw >> (jj * 4u)) & 15u) == 0u) continue;
                            uint4 q = cl4[jj];
                            for (uint e = 0; e < 4u; e++)
                            {
                                if ((bitsw & (1u << (jj * 4u + e))) == 0u) continue;
                                uint h = q[e];
                                V3 p = v3(0, 0, 0) + TGP(h);
                                if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) continue;
                                if (n < PROW - 2u) pairs[tri * PROW + 2u + n] = h;
                                n++;
                            }
                        }
                    }
                }
                if (n > PROW - 2u) { tgFlags[F_STATUS] = tgFlags[F_STATUS] | 1u; n = PROW - 2u; }
                pairs[tri * PROW] = n;
                pairs[tri * PROW + 1u] = tn.x | tn.y << 10 | tn.z << 20;
            }
            BAR;
            if (lid == 0) { tgFlags[F_REBUILD] = 0u; tgFlags[F_BUILDS] += 1u; }
#endif
        }

        // ---- S4: the beam pass gathered per node; the triangles' listed pairs ----
        if (isNode && !SKIP(4))
        {
            V3 Ps = Pown;
            V3 f = v3(0.0f, massN * GRAVITY, 0.0f);
#if OWN
            f = f + fOwn;
#endif
            // (A broken beam's scalar is zero: adding ±0 changes no bit of a sum that began at
            // +0 or at the weight, so it is the reference's skipped add.)
            for (uint k = 0; k < MAXENT; k += 4u)
            {
                if (k >= entScalar) break;
                uint4 cur = ent4[(entStart >> 2) + (k >> 2)];
                ENT_ADD(f, cur.x);
                if (k + 1u < entScalar) ENT_ADD(f, cur.y);
                if (k + 2u < entScalar) ENT_ADD(f, cur.z);
                if (k + 3u < entScalar) ENT_ADD(f, cur.w);
            }
            uint4 cur = uint4(0u);
            for (uint k = entScalar; k < MAXENT; k++)
            {
                if (k >= entCount) break;
                if (k == entScalar || (k & 3u) == 0u) cur = ent4[(entStart >> 2) + (k >> 2)];
                uint e = cur[k & 3u];
                if ((e >> 31) == 0u)
                {
                    uint o3 = e & 2047u;
                    f = f + (Ps - TGP(o3)) * tgG[e >> 11];
                }
                else
                {
                    uint vi = e & 0xFFFu;
                    V3 vv = GETV(vi);
                    f = f + ((e & (1u << 30)) != 0u ? -vv : vv);
                }
            }
            F0 = f;
        }
        for (uint s = 0; s < RS; s++)
        {
            // Deform groups asked for in this pass (spec §19).
            uint w = w0[s];
            if (w != NONE && (w >> 30) != C_AERO && (w & W_DEFORM) != 0u)
            {
                uint r = (w >> 30) == C_GENERIC ? (w & 0xFFFFFFu) : perm[s * LANES + lid];
                uint emask = as_type<uint>(recB[r].w);
                uint d0 = (emask >> 8) & 0xFFu, d1 = (emask >> 16) & 0xFFu;
                if (d0 != 0u) deform[d0 - 1u] = 1u;
                if (d1 != 0u) deform[d1 - 1u] = 1u;
                w0[s] = w & ~W_DEFORM;
            }
        }
#if COLLIDE
        if (isTri && !SKIP(5))
        {
            // (No node holds a side unless the last step had a contact: the held byte is read
            // only then.)
            const bool anyHeld = tgFlags[F_HELD] != 0u;
            // (The row's first load: its count, its triangle's nodes, its first two pairs.)
            device const uint4* row4 = (device const uint4*)(pairs + tri * PROW);
            uint4 cur = row4[0];
            const uint n = min(cur.x, (uint)(PROW - 2u));
#ifdef EMU_TRACE
            if (step == 5u && frame == 0u) fprintf(stderr, "row lane %u n %u\n", lid, n);
#endif
            if (n != 0u)
            {
                const uint t0 = cur.y & 1023u, t1 = (cur.y >> 10) & 1023u, t2 = (cur.y >> 20) & 1023u;
                // (Only what the filter reads stays live over the row: two points and two
                // plane normals; a pair that passes rebuilds the rest, the same values. No
                // arrays: an array indexed in a loop is stack memory.)
                V3 f0, frd, b0, brd;
#if ROWOVER
                V3 fra, frb;
#endif
                bool fok, bok;
                {
                    V3 q0 = v3(0, 0, 0) + TGP(t0), q1 = v3(0, 0, 0) + TGP(t1), q2 = v3(0, 0, 0) + TGP(t2);
                    Frame F = triFrame(q0, q1, q2);
                    Frame B = triFrame(q0 - TGV(t0) * H, q1 - TGV(t1) * H, q2 - TGV(t2) * H);
                    f0 = F.p0; frd = F.rd; fok = F.ok; b0 = B.p0; brd = B.rd; bok = B.ok;
#if ROWOVER
                    fra = F.ra; frb = F.rb;
#endif
                }
                // One listed point against the triangle. (A point that holds no side for this
                // triangle and did not cross its plane is a contact only inside the band and
                // over the triangle: the rule's own tests, the same values. Most points near a
                // panel's plane are its neighbours, beside the triangle, and stop here. One
                // that may be a contact goes on the step's list: the full rule runs a lane a
                // pair after the barrier.)
#if ROWOVER
                #define ROW_FIRST ((d0 > 0.0f && d1 < 0.0f) || (d0 < 0.0f && d1 > 0.0f))
                #define ROW_MORE \
                    if (!cand && anyHeld && tgHeld[h] != 0) \
                    { \
                        const uint c0 = atomic_load_explicit(&clip[h * 4], memory_order_relaxed), c1 = atomic_load_explicit(&clip[h * 4 + 1], memory_order_relaxed); \
                        const uint key = CLIP_SELF << 20 | tri; \
                        cand = (c0 != CLIP_EMPTY && (c0 >> 1) == key) || (c1 != CLIP_EMPTY && (c1 >> 1) == key); \
                    } \
                    if (!cand && fabs(d1) <= CR) \
                    { \
                        const float al = fok ? dot3(fra, ph - f0) : 0.0f, be = fok ? dot3(frb, ph - f0) : 0.0f; \
                        cand = al >= 0.0f && be >= 0.0f && ((1.0f - al) - be) >= 0.0f; \
                    }
#else
                #define ROW_FIRST ((anyHeld && tgHeld[h] != 0) || (d0 > 0.0f && d1 < 0.0f) || (d0 < 0.0f && d1 > 0.0f) || fabs(d1) <= CR)
                #define ROW_MORE
#endif
                #define ROW_ONE(HV) do { \
                    const uint h = (HV); \
                    V3 ph = v3(0, 0, 0) + TGP(h); \
                    V3 vh = TGV(h); \
                    float d1 = fok ? dot3(frd, ph - f0) : 0.0f; \
                    float d0 = bok ? dot3(brd, (ph - vh * H) - b0) : 0.0f; \
                    bool cand = ROW_FIRST; \
                    ROW_MORE \
                    if (!cand) break; \
                    uint ci = atomic_fetch_add_explicit(&tgCount[1], 1u, memory_order_relaxed); \
                    if (ci < CAND_CAP) candBuf[ci] = tri << 16 | h; \
                } while (false)
#if ROW4
                // (Four points a load, each by its own name: a vector indexed by a variable
                // is a store and a load.)
                ROW_ONE(cur.z);
                if (n > 1u) ROW_ONE(cur.w);
                for (uint k = 2u; k < n; k += 4u)
                {
                    cur = row4[(k + 2u) >> 2];
                    ROW_ONE(cur.x);
                    if (k + 1u < n) ROW_ONE(cur.y);
                    if (k + 2u < n) ROW_ONE(cur.z);
                    if (k + 3u < n) ROW_ONE(cur.w);
                }
#else
                for (uint k = 0; k < PROW - 2u; k++)
                {
                    if (k >= n) break;
                    uint kk1 = k + 2u;
                    if ((kk1 & 3u) == 0u) cur = row4[kk1 >> 2];
                    ROW_ONE(cur[kk1 & 3u]);
                }
#endif
            }
        }
#endif
        BAR;
#if COLLIDE
        const uint ncand = atomic_load_explicit(&tgCount[1], memory_order_relaxed);
#ifdef EMU_TRACE
        if (lid == 0u && (step == 5u || step == 20u)) fprintf(stderr, "cands frame %u step %u: %u\n", frame, step, ncand);
#endif
        if (ncand != 0u && !SKIP(15))
        {
            if (lid == 0 && ncand > CAND_CAP) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 4u;
            for (uint kc = lid; kc < min(ncand, (uint)CAND_CAP); kc += LANES)
            {
                // The full rule (explicit_beams.rs collision_pass, clip_hit).
                const uint cw = candBuf[kc];
                const uint tri = cw >> 16, h = cw & 0xFFFFu;
                const uint hdr = pairs[tri * PROW + 1u];
                const uint t0 = hdr & 1023u, t1 = (hdr >> 10) & 1023u, t2 = (hdr >> 20) & 1023u;
                V3 ph = v3(0, 0, 0) + TGP(h);
                V3 vh = TGV(h);
                    const uint4 tn = uint4(t0, t1, t2, 0u);
                    const V3 pA = v3(0, 0, 0) + TGP(t0), pB = v3(0, 0, 0) + TGP(t1), pC = v3(0, 0, 0) + TGP(t2);
                    const V3 vA = TGV(t0), vB = TGV(t1), vC = TGV(t2);
                    Frame F = triFrame(pA, pB, pC);
                    Frame B = triFrame(pA - vA * H, pB - vB * H, pC - vC * H);
                    float tv = rmax(rmax(rmax(0.0f, len3(vA)), len3(vB)), len3(vC));
                    float reach = (CR + (tv + vmax) * H) + CLIP_DEPTH;
                    V3 lo = v3(rmin(rmin(pA.x, pB.x), pC.x) - reach, rmin(rmin(pA.y, pB.y), pC.y) - reach, rmin(rmin(pA.z, pB.z), pC.z) - reach);
                    V3 hi = v3(rmax(rmax(pA.x, pB.x), pC.x) + reach, rmax(rmax(pA.y, pB.y), pC.y) + reach, rmax(rmax(pA.z, pB.z), pC.z) + reach);
                    if (ph.x < lo.x || ph.y < lo.y || ph.z < lo.z || ph.x > hi.x || ph.y > hi.y || ph.z > hi.z) continue;
                    float4 loc = frameLocal(F.p0, F.ra, F.rb, F.rd, F.ok, ph);
                    uint key = CLIP_SELF << 20 | tri;
                    float side = 0.0f;
                    uint c0 = atomic_load_explicit(&clip[h * 4], memory_order_relaxed);
                    uint c1 = atomic_load_explicit(&clip[h * 4 + 1], memory_order_relaxed);
                    if (c0 != CLIP_EMPTY && (c0 >> 1) == key) side = (c0 & 1u) != 0u ? 1.0f : -1.0f;
                    else if (c1 != CLIP_EMPTY && (c1 >> 1) == key) side = (c1 & 1u) != 0u ? 1.0f : -1.0f;
                    bool over = loc.x >= 0.0f && loc.y >= 0.0f && loc.z >= 0.0f;
                    bool haveW = false;
                    float3 wgt = float3(0);
                    if (side != 0.0f)
                    {
                        haveW = over;
                        wgt = float3(loc.x, loc.y, loc.z);
                    }
                    else
                    {
                        float4 l0 = frameLocal(B.p0, B.ra, B.rb, B.rd, B.ok, ph - vh * H);
                        float e0 = l0.w, e1 = loc.w;
                        bool crossed = false;
                        if ((e0 > 0.0f && e1 < 0.0f) || (e0 < 0.0f && e1 > 0.0f))
                        {
                            float sx = e0 / (e0 - e1);
                            float aa = l0.x + (loc.x - l0.x) * sx;
                            float bb2 = l0.y + (loc.y - l0.y) * sx;
                            float3 cw = float3(aa, bb2, (1.0f - aa) - bb2);
                            if (cw.x >= 0.0f && cw.y >= 0.0f && cw.z >= 0.0f)
                            {
                                crossed = true;
                                side = e0 > 0.0f ? 1.0f : -1.0f;
                                haveW = true;
                                wgt = cw;
                            }
                        }
                        if (!crossed)
                        {
                            if (over && fabs(loc.w) <= CR)
                            {
                                side = loc.w < 0.0f ? -1.0f : 1.0f;
                                haveW = true;
                                wgt = float3(loc.x, loc.y, loc.z);
                            }
                            else continue;
                        }
                    }
                    float dd = side * loc.w;
                    if (!(haveW && dd <= CR && dd >= -CLIP_DEPTH)) continue;
                    uint word = CLIP_SELF << 21 | tri << 1 | (side > 0.0f ? 1u : 0u);
                    uint old = atomic_fetch_min_explicit(&clip[h * 4 + 2], word, memory_order_relaxed);
                    atomic_fetch_min_explicit(&clip[h * 4 + 3], max(old, word), memory_order_relaxed);
                    tgFlags[F_REC] = 1u;
                    float pen = rmin(CR - dd, CR);
                    V3 nn = F.n * side;
                    uint i = atomic_fetch_add_explicit(&tgCount[2], 1u, memory_order_relaxed);
#if HITTG
                    TOUCHK(h, 5u); TOUCHK(tn.x, 1u); TOUCHK(tn.y, 1u); TOUCHK(tn.z, 1u);
                    if (i < H2CAP)
                    {
                        hitBuf[i * HITW] = as_type<float4>(uint4(h, tn.x, tn.y, tn.z));
                        hitBuf[i * HITW + 1] = float4(wgt.x, wgt.y, wgt.z, pen);
                        hitBuf[i * HITW + 2] = f4(nn, 0);
                        atomic_fetch_min_explicit(&hmap[h], i * 4u, memory_order_relaxed);
                        atomic_fetch_min_explicit(&hmap[tn.x], i * 4u + 1u, memory_order_relaxed);
                        atomic_fetch_min_explicit(&hmap[tn.y], i * 4u + 2u, memory_order_relaxed);
                        atomic_fetch_min_explicit(&hmap[tn.z], i * 4u + 3u, memory_order_relaxed);
                    }
#else
                    TOUCHK(h, 13u); TOUCHK(tn.x, 9u); TOUCHK(tn.y, 9u); TOUCHK(tn.z, 9u);
                    if (i < HIT_CAP)
                    {
                        hitBuf[i * HITW] = as_type<float4>(uint4(h, tn.x, tn.y, tn.z));
                        hitBuf[i * HITW + 1] = float4(wgt.x, wgt.y, wgt.z, pen);
                        hitBuf[i * HITW + 2] = f4(nn, 0);
                    }
                    atomic_fetch_add_explicit(&split[h], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&split[tn.x], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&split[tn.y], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&split[tn.z], 1u, memory_order_relaxed);
#endif
            }
            BAR;
        }
#endif
#if COLLIDE
        const uint nhits = atomic_load_explicit(&tgCount[2], memory_order_relaxed);
#ifdef EMU_TRACE
        if (lid == 0u) fprintf(stderr, "T car %u step %u: self candidates %u hits %u\n", car, step, ncand, nhits);
#endif
#if HITTG
        // ---- the step's hits: the two-body rule's passes, from threadgroup memory ----
        // Each of a hit's four nodes has a slot in the beam scalars' array (free until the
        // next beam pass): the node's force, its mass, how many hits name it, and its staged
        // sums. A node's slot is the smallest (hit, corner) number that names it (an atomic
        // min a node where the hit was recorded); the node's own lane fills it from its
        // registers, and a hit's lane keeps its record and its four slots in registers over
        // the passes.
        if (nhits != 0u && !SKIP(8))
        {
            const uint nh = min(nhits, (uint)H2CAP);
            const uint mySlot = isNode ? atomic_load_explicit(&hmap[ni], memory_order_relaxed) : NONE;
            uint4 hid = uint4(0u), hsl = uint4(0u);
            float4 hwp = float4(0.0f);
            V3 hnn = v3(0, 0, 0), hf = v3(0, 0, 0);
            float hres = 0.0f;
            if (mySlot != NONE)
            {
                const uint o = mySlot * 11u;
                tgG[o] = F0.x; tgG[o + 1u] = F0.y; tgG[o + 2u] = F0.z;
#if HAS_IMMOVABLE
                tgG[o + 3u] = (as_type<uint>(nodeStatic[ni].z) & NODE_IMMOVABLE) != 0u ? 0.0f : massN;
#else
                tgG[o + 3u] = massN;
#endif
                tgG[o + 4u] = 0.0f; tgG[o + 5u] = 0.0f; tgG[o + 6u] = 0.0f; tgG[o + 7u] = 0.0f;
                tgG[o + 8u] = 0.0f; tgG[o + 9u] = 0.0f; tgG[o + 10u] = 0.0f;
            }
            if (lid < nh)
            {
                hid = as_type<uint4>(hitBuf[lid * HITW]);
                hwp = hitBuf[lid * HITW + 1];
                hnn = v3(hitBuf[lid * HITW + 2]);
                hsl = uint4(atomic_load_explicit(&hmap[hid.x], memory_order_relaxed), atomic_load_explicit(&hmap[hid.y], memory_order_relaxed),
                            atomic_load_explicit(&hmap[hid.z], memory_order_relaxed), atomic_load_explicit(&hmap[hid.w], memory_order_relaxed));
            }
            if (lid == 0)
            {
                tgFlags[F_HITS] = 1u;
                tgFlags[F_MOST] = max(tgFlags[F_MOST], nhits);
                if (nhits > H2CAP) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 2u;
            }
            BAR;
            #define SLF(s_) v3(tgG[(s_) * 11u], tgG[(s_) * 11u + 1u], tgG[(s_) * 11u + 2u])
            #define SLM(s_) tgG[(s_) * 11u + 3u]
            #define SLKP(s_) ((TGPTR(atomic_uint))&tgG[(s_) * 11u + 4u])
            #define SLK(s_) atomic_load_explicit(SLKP(s_), memory_order_relaxed)
            #define SLS(s_, q_) ((TGPTR(atomic_int))&tgG[(s_) * 11u + 5u + (q_)])
            #define SLL(s_, q_) atomic_load_explicit(SLS(s_, q_), memory_order_relaxed)
            #define SACC(f_, s_) (SLM(s_) == 0.0f ? v3(0, 0, 0) : (f_) / SLM(s_))
            #define SIM(s_, k_) (SLM(s_) == 0.0f ? 0.0f : float(k_) / SLM(s_))
            #define SNF(s_) (SLF(s_) + v3(0, 0, 0))
            #define HV(i_) TGV(i_)
            // How many hits name each node; the map cleared for the next step.
            if (lid < nh)
            {
                atomic_fetch_add_explicit(SLKP(hsl.x), 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(SLKP(hsl.y), 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(SLKP(hsl.z), 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(SLKP(hsl.w), 1u, memory_order_relaxed);
            }
            if (mySlot != NONE) atomic_store_explicit(&hmap[ni], NONE, memory_order_relaxed);
            BAR;
            // The first pass.
            if (lid < nh)
            {
                const uint hn = hid.x, i0 = hid.y, i1 = hid.z, i2 = hid.w;
                const uint sh = hsl.x, s0 = hsl.y, s1 = hsl.z, s2 = hsl.w;
                const float alpha = hwp.x, beta = hwp.y, gamma = hwp.z, depth = hwp.w;
                const V3 n = hnn;
                V3 v = HV(hn) - ((HV(i1) * alpha + HV(i2) * beta) + HV(i0) * gamma);
                float imh = SIM(sh, SLK(sh));
                float imt = tbInvMass(SIM(s1, SLK(s1)), SIM(s2, SLK(s2)), SIM(s0, SLK(s0)), alpha, beta, gamma);
                float inv = imh + imt;
                float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                V3 at = (SACC(SNF(s1), s1) * alpha + SACC(SNF(s2), s2) * beta) + SACC(SNF(s0), s0) * gamma;
                V3 fh = (SACC(SNF(sh), sh) - at) * m;
                float budget;
                V3 f = contactResponse(fh, v, m, n, BOXS, depth, nodeStatic[hn].y, budget);
                if (!isZero3(f))
                {
                    float delta = shiftDepth(depth, budget, dot3(fh, n));
                    if (delta > 0.0f)
                    {
                        float q = delta * m;
                        V3 dh = n * (q * SIM(sh, 1u));
                        TOUCHK(hn, 2u);
                        for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[hn * 3 + a], toFixedShift(a == 0 ? dh.x : (a == 1 ? dh.y : dh.z)), memory_order_relaxed);
                        uint ti[3] = {i1, i2, i0};
                        uint tsl[3] = {s1, s2, s0};
                        float tw[3] = {alpha, beta, gamma};
                        for (uint e = 0; e < 3; e++)
                        {
                            if (tw[e] > 0.0f)
                            {
                                V3 dt = n * -(tw[e] * (q * SIM(tsl[e], 1u)));
                                TOUCHK(ti[e], 2u);
                                for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[ti[e] * 3 + a], toFixedShift(a == 0 ? dt.x : (a == 1 ? dt.y : dt.z)), memory_order_relaxed);
                            }
                        }
                        tgFlags[F_SHIFT] = 1u;
                    }
                }
                hf = f;
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ss[4] = {sh, s1, s2, s0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(SLS(ss[e], 0u), toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(SLS(ss[e], 1u), toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(SLS(ss[e], 2u), toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            BAR;
            // The residual pass, from the forces with the first pass's sums.
            #define SNOW1(s_) (SLF(s_) + v3(float(SLL(s_, 0u)), float(SLL(s_, 1u)), float(SLL(s_, 2u))) * (1.0f / 32.0f))
            if (lid < nh)
            {
                const uint hn = hid.x, i0 = hid.y, i1 = hid.z, i2 = hid.w;
                const uint sh = hsl.x, s0 = hsl.y, s1 = hsl.z, s2 = hsl.w;
                const float alpha = hwp.x, beta = hwp.y, gamma = hwp.z;
                const V3 n = hnn;
                V3 v = HV(hn) - ((HV(i1) * alpha + HV(i2) * beta) + HV(i0) * gamma);
                V3 at = (SACC(SNOW1(s1), s1) * alpha + SACC(SNOW1(s2), s2) * beta) + SACC(SNOW1(s0), s0) * gamma;
                float a = dot3(SACC(SNOW1(sh), sh) - at, n);
                float vn = dot3(v, n);
                float x = vn + rmax(vn, 0.0f);
                float r = 0.0f;
                if (vn < 0.0f && a * H + x < 0.0f)
                {
                    float inv = SIM(sh, SLK(sh)) + tbInvMass(SIM(s1, SLK(s1)), SIM(s2, SLK(s2)), SIM(s0, SLK(s0)), alpha, beta, gamma);
                    if (inv > 0.0f)
                    {
                        float m = 1.0f / inv;
                        float rr = -(a * m) - (x * m) / H;
                        if (rr > 0.0f) r = rr;
                    }
                }
                hres = r;
                if (r > 0.0f)
                {
                    V3 f = n * r;
                    V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                    uint ss[4] = {sh, s1, s2, s0};
                    for (uint e = 0; e < 4; e++)
                    {
                        atomic_fetch_add_explicit(SLS(ss[e], 3u), toFixed(fs[e].x), memory_order_relaxed);
                        atomic_fetch_add_explicit(SLS(ss[e], 4u), toFixed(fs[e].y), memory_order_relaxed);
                        atomic_fetch_add_explicit(SLS(ss[e], 5u), toFixed(fs[e].z), memory_order_relaxed);
                    }
                }
            }
            BAR;
            // The work guard, from the forces with both passes' sums; each hit's total into
            // its nodes' contact sums.
            #define SSUM2(s_, a_) (SLL(s_, a_) + SLL(s_, 3u + (a_)))
            #define SNOW2(s_) (SLF(s_) + v3(float(SSUM2(s_, 0u)), float(SSUM2(s_, 1u)), float(SSUM2(s_, 2u))) * (1.0f / 32.0f))
            if (lid < nh)
            {
                const uint hn = hid.x, i0 = hid.y, i1 = hid.z, i2 = hid.w;
                const uint sh = hsl.x, s0 = hsl.y, s1 = hsl.z, s2 = hsl.w;
                const float alpha = hwp.x, beta = hwp.y, gamma = hwp.z;
                const V3 n = hnn;
                V3 t = hf;
                if (hres > 0.0f) t = t + n * hres;
                V3 f = t;
                if (!isZero3(t))
                {
                    V3 v = HV(hn) - ((HV(i1) * alpha + HV(i2) * beta) + HV(i0) * gamma);
                    V3 a = SACC(SNOW2(sh), sh) - ((SACC(SNOW2(s1), s1) * alpha + SACC(SNOW2(s2), s2) * beta) + SACC(SNOW2(s0), s0) * gamma);
                    float inv = SIM(sh, 1u) + tbInvMass(SIM(s1, 1u), SIM(s2, 1u), SIM(s0, 1u), alpha, beta, gamma);
                    V3 aRest = a - t * inv;
                    float aa = dot3(t, (v + v) + aRest * H);
                    float bb = (H * dot3(t, t)) * inv;
                    if (aa + bb > 0.0f && bb > 0.0f)
                    {
                        float sc = aa < 0.0f ? rmin(-aa / bb, 1.0f) : 0.0f;
                        f = t + t * -(1.0f - sc);
                    }
                }
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&acc[ns[e] * 3], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            BAR;
        }
#else
        if (nhits != 0u && !SKIP(8))
        {
            // ---- the step's hits: the forces published, then the two-body rule's passes ----
            if (isNode) { fPub[ni * 3] = F0.x; fPub[ni * 3 + 1] = F0.y; fPub[ni * 3 + 2] = F0.z; }
            if (lid == 0)
            {
                tgFlags[F_HITS] = 1u;
                tgFlags[F_MOST] = max(tgFlags[F_MOST], nhits);
                if (nhits > HIT_CAP) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 2u;
            }
            BAR;
            const uint nh = min(nhits, (uint)HIT_CAP);
            #define NP(i) TGP(i)
            #define NV(i) TGV(i)
            #define NF(i) (v3(fPub[(i) * 3], fPub[(i) * 3 + 1], fPub[(i) * 3 + 2]) + v3(0, 0, 0))
            #define NM(i) nodeStatic[i].x
            #define NIMM(i) ((as_type<uint>(nodeStatic[i].z) & NODE_IMMOVABLE) != 0u)
            #define ACC(f, i) (NIMM(i) ? v3(0, 0, 0) : (f) / NM(i))
            #define IM(i, kk_) (NIMM(i) ? 0.0f : float(kk_) / NM(i))
            #define SPL(i) atomic_load_explicit(&split[i], memory_order_relaxed)
            for (uint k = lid; k < nh; k += LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z, depth = wp.w;
                V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                float imh = IM(hn, SPL(hn));
                float imt = tbInvMass(IM(i1, SPL(i1)), IM(i2, SPL(i2)), IM(i0, SPL(i0)), alpha, beta, gamma);
                float inv = imh + imt;
                float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                V3 at = (ACC(NF(i1), i1) * alpha + ACC(NF(i2), i2) * beta) + ACC(NF(i0), i0) * gamma;
                V3 fh = (ACC(NF(hn), hn) - at) * m;
                float budget;
                V3 f = contactResponse(fh, v, m, n, BOXS, depth, nodeStatic[hn].y, budget);
                if (!isZero3(f))
                {
                    float delta = shiftDepth(depth, budget, dot3(fh, n));
                    if (delta > 0.0f)
                    {
                        float q = delta * m;
                        V3 dh = n * (q * IM(hn, 1u));
                        TOUCHK(hn, 2u);
                        for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[hn * 3 + a], toFixedShift(a == 0 ? dh.x : (a == 1 ? dh.y : dh.z)), memory_order_relaxed);
                        uint ti[3] = {i1, i2, i0};
                        float tw[3] = {alpha, beta, gamma};
                        for (uint e = 0; e < 3; e++)
                        {
                            if (tw[e] > 0.0f)
                            {
                                V3 dt = n * -(tw[e] * (q * IM(ti[e], 1u)));
                                TOUCHK(ti[e], 2u);
                                for (uint a = 0; a < 3; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[ti[e] * 3 + a], toFixedShift(a == 0 ? dt.x : (a == 1 ? dt.y : dt.z)), memory_order_relaxed);
                            }
                        }
                        tgFlags[F_SHIFT] = 1u;
                    }
                }
                hitBuf[k * HITW + 3] = f4(f, 0);
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&stage[ns[e] * 6], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&stage[ns[e] * 6 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&stage[ns[e] * 6 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            BAR;
            #define NOW1(i) (v3(fPub[(i) * 3], fPub[(i) * 3 + 1], fPub[(i) * 3 + 2]) + v3(float(atomic_load_explicit(&stage[(i) * 6], memory_order_relaxed)), float(atomic_load_explicit(&stage[(i) * 6 + 1], memory_order_relaxed)), float(atomic_load_explicit(&stage[(i) * 6 + 2], memory_order_relaxed))) * (1.0f / 32.0f))
            for (uint k = lid; k < nh; k += LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z;
                V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                V3 at = (ACC(NOW1(i1), i1) * alpha + ACC(NOW1(i2), i2) * beta) + ACC(NOW1(i0), i0) * gamma;
                float a = dot3(ACC(NOW1(hn), hn) - at, n);
                float vn = dot3(v, n);
                float x = vn + rmax(vn, 0.0f);
                float r = 0.0f;
                if (vn < 0.0f && a * H + x < 0.0f)
                {
                    float inv = IM(hn, SPL(hn)) + tbInvMass(IM(i1, SPL(i1)), IM(i2, SPL(i2)), IM(i0, SPL(i0)), alpha, beta, gamma);
                    if (inv > 0.0f)
                    {
                        float m = 1.0f / inv;
                        float rr = -(a * m) - (x * m) / H;
                        if (rr > 0.0f) r = rr;
                    }
                }
                float4 h3 = hitBuf[k * HITW + 3];
                hitBuf[k * HITW + 3] = float4(h3.x, h3.y, h3.z, r);
                if (r > 0.0f)
                {
                    V3 f = n * r;
                    V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                    uint ns[4] = {hn, i1, i2, i0};
                    for (uint e = 0; e < 4; e++)
                    {
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 3], toFixed(fs[e].x), memory_order_relaxed);
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 4], toFixed(fs[e].y), memory_order_relaxed);
                        atomic_fetch_add_explicit(&stage[ns[e] * 6 + 5], toFixed(fs[e].z), memory_order_relaxed);
                    }
                }
            }
            BAR;
            #define SUM2(i, a) (atomic_load_explicit(&stage[(i) * 6 + (a)], memory_order_relaxed) + atomic_load_explicit(&stage[(i) * 6 + 3 + (a)], memory_order_relaxed))
            #define NOW2(i) (v3(fPub[(i) * 3], fPub[(i) * 3 + 1], fPub[(i) * 3 + 2]) + v3(float(SUM2(i, 0)), float(SUM2(i, 1)), float(SUM2(i, 2))) * (1.0f / 32.0f))
            for (uint k = lid; k < nh; k += LANES)
            {
                uint4 id = as_type<uint4>(hitBuf[k * HITW]);
                float4 wp = hitBuf[k * HITW + 1];
                V3 n = v3(hitBuf[k * HITW + 2]);
                float4 h3 = hitBuf[k * HITW + 3];
                uint hn = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                float alpha = wp.x, beta = wp.y, gamma = wp.z;
                V3 t = v3(h3);
                if (h3.w > 0.0f) t = t + n * h3.w;
                V3 f = t;
                if (!isZero3(t))
                {
                    V3 v = NV(hn) - ((NV(i1) * alpha + NV(i2) * beta) + NV(i0) * gamma);
                    V3 a = ACC(NOW2(hn), hn) - ((ACC(NOW2(i1), i1) * alpha + ACC(NOW2(i2), i2) * beta) + ACC(NOW2(i0), i0) * gamma);
                    float inv = IM(hn, 1u) + tbInvMass(IM(i1, 1u), IM(i2, 1u), IM(i0, 1u), alpha, beta, gamma);
                    V3 aRest = a - t * inv;
                    float aa = dot3(t, (v + v) + aRest * H);
                    float bb = (H * dot3(t, t)) * inv;
                    if (aa + bb > 0.0f && bb > 0.0f)
                    {
                        float sc = aa < 0.0f ? rmin(-aa / bb, 1.0f) : 0.0f;
                        f = t + t * -(1.0f - sc);
                    }
                }
                V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                uint ns[4] = {hn, i1, i2, i0};
                for (uint e = 0; e < 4; e++)
                {
                    atomic_fetch_add_explicit(&acc[ns[e] * 3], toFixed(fs[e].x), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 1], toFixed(fs[e].y), memory_order_relaxed);
                    atomic_fetch_add_explicit(&acc[ns[e] * 3 + 2], toFixed(fs[e].z), memory_order_relaxed);
                }
            }
            BAR;
        }
#endif
#endif
#if SURF
        // ---- the surface pass (explicit_surface.rs surface_pass): after self-collision ----
        // SKIP bits, timing only: 21 no pass; 22 the search alone; 24 no staged stage; 25 no
        // stage B; 26 no stage R; 23 nothing emitted or applied.
        uint3 sHitN = uint3(0u, 0u, 0u);
        if (!SKIP(21))
        {
            // A slot is near when the body's box (reduced in the S1 phase, after the node
            // pass), padded by the band, meets its cull box: an element's padded box is inside
            // the body's, so no other slot passes `near`. A near slot goes into the scalars'
            // array.
            if (lid < NSS)
            {
                device const float4* row = surfRow + lid * SRW;
                float4 lo = row[5], hi = row[6];
                V3 clo = v3(unordF(atomic_load_explicit(&tgS[0], memory_order_relaxed)), unordF(atomic_load_explicit(&tgS[1], memory_order_relaxed)), unordF(atomic_load_explicit(&tgS[2], memory_order_relaxed)));
                V3 chi = v3(unordF(atomic_load_explicit(&tgS[3], memory_order_relaxed)), unordF(atomic_load_explicit(&tgS[4], memory_order_relaxed)), unordF(atomic_load_explicit(&tgS[5], memory_order_relaxed)));
                if (clo.x - CR <= hi.x && clo.y - CR <= hi.y && clo.z - CR <= hi.z && chi.x + CR >= lo.x && chi.y + CR >= lo.y && chi.z + CR >= lo.z)
                {
                    const uint o = SB(lid);
                    tgG[o] = lo.x; tgG[o + 1u] = lo.y; tgG[o + 2u] = lo.z;
                    tgG[o + 3u] = hi.x; tgG[o + 4u] = hi.y; tgG[o + 5u] = hi.z;
                    for (uint k = 0u; k < 5u; k++)
                    {
                        float4 r = row[k];
                        tgG[o + 6u + k * 3u] = r.x; tgG[o + 7u + k * 3u] = r.y; tgG[o + 8u + k * 3u] = r.z;
                    }
                    tgG[o + 21u] = row[0].w; tgG[o + 22u] = row[4].w; tgG[o + 23u] = row[7].x;
                    atomic_fetch_or_explicit(&tgS[6u + (lid >> 5)], int(1u << (lid & 31u)), memory_order_relaxed);
                }
            }
            BAR;
            const uint sa0 = uint(atomic_load_explicit(&tgS[6], memory_order_relaxed)), sa1 = uint(atomic_load_explicit(&tgS[7], memory_order_relaxed));
            if ((sa0 | sa1) != 0u)
            {
                // The search: every element's padded box against the near slots' cull boxes
                // (`near`, `seg_near`); a pair that passes goes on the step's list.
                #define SURF_NEAR(e_, tlo_, thi_) for (uint hf_ = 0u; hf_ < 2u; hf_++) for (uint m_ = hf_ == 0u ? sa0 : sa1; m_ != 0u; m_ &= m_ - 1u) { \
                    const uint sl_ = hf_ * 32u + ctz(m_); const uint o_ = SB(sl_); \
                    if (tlo_.x <= tgG[o_ + 3u] && tlo_.y <= tgG[o_ + 4u] && tlo_.z <= tgG[o_ + 5u] && thi_.x >= tgG[o_] && thi_.y >= tgG[o_ + 1u] && thi_.z >= tgG[o_ + 2u]) { \
                        const uint i_ = uint(atomic_fetch_add_explicit(&tgS[8], 1, memory_order_relaxed)); \
                        if (i_ < SPAIR_CAP) surfPair[i_] = (e_) | sl_ << 16; } }
                if (isTri)
                {
                    uint4 tn = tris[tri];
                    float3 x0 = fv(v3(0, 0, 0) + TGP(tn.x)), x1 = fv(v3(0, 0, 0) + TGP(tn.y)), x2 = fv(v3(0, 0, 0) + TGP(tn.z));
                    float3 tlo = ssub3(min(min(x0, x1), x2), float3(CR, CR, CR));
                    float3 thi = sadd3(max(max(x0, x1), x2), float3(CR, CR, CR));
                    SURF_NEAR(tri, tlo, thi)
                }
                for (uint k = lid; k < NSEG; k += LANES)
                {
                    const uint ab = tab[T_SEG + k * 2u];
                    float3 x0 = fv(v3(0, 0, 0) + TGP(ab & 0xFFFFu)), x1 = fv(v3(0, 0, 0) + TGP(ab >> 16));
                    float3 tlo = ssub3(min(x0, x1), float3(CR, CR, CR));
                    float3 thi = sadd3(max(x0, x1), float3(CR, CR, CR));
                    SURF_NEAR(NT + k, tlo, thi)
                }
                if (isNode)
                {
                    float3 x0 = fv(v3(0, 0, 0) + Pown);
                    float3 tlo = ssub3(x0, float3(CR, CR, CR));
                    float3 thi = sadd3(x0, float3(CR, CR, CR));
                    SURF_NEAR(NT + NSEG + ni, tlo, thi)
                }
                BAR;
                const uint npair = uint(atomic_load_explicit(&tgS[8], memory_order_relaxed));
#ifdef EMU_TRACE
                if (lid == 0u) fprintf(stderr, "T car %u step %u: surface slots near %u, pairs %u\n", car, step, popcount(sa0) + popcount(sa1), npair);
#endif
                if (npair != 0u && !SKIP(22))
                {
                    if (lid == 0 && npair > SPAIR_CAP) atomic_fetch_or_explicit(&tgS[11], 16, memory_order_relaxed);
                    // Every node's force as the next node pass will take it (forces_now: with
                    // this step's contact sums; a node no hit has written has an empty sum),
                    // for the hits' lanes.
                    if (isNode)
                    {
                        V3 fnow = F0 + v3(float(atomic_load_explicit(&acc[ni * 3], memory_order_relaxed)), float(atomic_load_explicit(&acc[ni * 3 + 1], memory_order_relaxed)), float(atomic_load_explicit(&acc[ni * 3 + 2], memory_order_relaxed))) * (1.0f / 32.0f);
                        tgG[SFB + ni * 3u] = fnow.x; tgG[SFB + ni * 3u + 1u] = fnow.y; tgG[SFB + ni * 3u + 2u] = fnow.z;
                    }
                    // The narrow tests. A point, and any element against a sphere, is tested
                    // here, a lane a pair (cheap). A triangle or a beam against a box is the
                    // heavy query (12 box edges, each several closest-point candidates, walked
                    // up to four times: 20 to 80 us in one lane), so it is staged: a lane a
                    // (pair, box edge) finds its candidates' gaps and the pair's least (an
                    // atomic min); the lanes owning the candidates within the tie of it record
                    // them; a lane a pair finishes the contact from those.
#if RIGID
                    #define SURF_BODY_HIT(bi_) if (bi_ != SURF_STATIC) atomic_fetch_add_explicit(&surfBH[bi_], 1u, memory_order_relaxed);
#else
                    #define SURF_BODY_HIT(bi_)
#endif
                    // A hit: its record, how many hits name each node with a weight (the mass
                    // split) and each body; the next node pass reads its nodes' sums.
                    #define SURF_EMIT(n0_, n1_, n2_, slot_, w_, g_, nn_, q_, two_, w2_, q2_) { \
                        const uint bi_ = as_type<uint>(tgG[SB(slot_) + 22u]); \
                        for (uint h2_ = 0u; h2_ < ((two_) ? 2u : 1u); h2_++) { \
                            float3 wh_ = h2_ == 0u ? (w_) : (w2_), qh_ = h2_ == 0u ? (q_) : (q2_); \
                            const uint i_ = uint(atomic_fetch_add_explicit(&tgS[9], 1, memory_order_relaxed)); \
                            if (i_ < SHIT_CAP) { \
                                surfHit[i_ * 4u] = as_type<float4>(uint4(n0_, n1_, n2_, slot_)); \
                                surfHit[i_ * 4u + 1u] = f4(wh_, g_); \
                                surfHit[i_ * 4u + 2u] = f4(nn_, 0.0f); \
                                surfHit[i_ * 4u + 3u] = f4(qh_, 0.0f); } \
                            if (wh_.x > 0.0f) atomic_fetch_add_explicit(&split[n0_], 1u, memory_order_relaxed); \
                            if (wh_.y > 0.0f) atomic_fetch_add_explicit(&split[n1_], 1u, memory_order_relaxed); \
                            if (wh_.z > 0.0f) atomic_fetch_add_explicit(&split[n2_], 1u, memory_order_relaxed); \
                            SURF_BODY_HIT(bi_) } \
                        TOUCHK(n0_, 3u); \
                        if (n1_ != n0_) TOUCHK(n1_, 3u); \
                        if (n2_ != n0_) TOUCHK(n2_, 3u); }
                    for (uint kp = lid; kp < min(npair, SPAIR_CAP); kp += LANES)
                    {
                        const uint pw = surfPair[kp];
                        const uint e = pw & 0xFFFFu, slot = pw >> 16;
                        SURF_SHAPE(fs, slot)
                        uint n0, n1, n2;
                        float3 x0, x1, x2;
                        float3 w = float3(1.0f, 0.0f, 0.0f), nn, q, w2 = float3(0.0f, 0.0f, 0.0f), q2 = float3(0.0f, 0.0f, 0.0f);
                        float g = 0.0f;
                        bool two = false;
                        const bool isT = e < NT, isS = !isT && e < NT + NSEG;
                        if (isT)
                        {
                            if (triDead[e] != 0u) continue;
                            uint4 tn = tris[e];
                            n0 = tn.x; n1 = tn.y; n2 = tn.z;
                            x0 = fv(v3(0, 0, 0) + TGP(n0)); x1 = fv(v3(0, 0, 0) + TGP(n1)); x2 = fv(v3(0, 0, 0) + TGP(n2));
                        }
                        else if (isS)
                        {
                            const uint k = e - NT;
                            const uint ab = tab[T_SEG + k * 2u];
                            if (recState[tab[T_SEG + k * 2u + 1u] * 3u + 1u].w != 0.0f) continue;
                            n0 = ab & 0xFFFFu; n1 = ab >> 16; n2 = n0;
                            x0 = fv(v3(0, 0, 0) + TGP(n0)); x1 = fv(v3(0, 0, 0) + TGP(n1)); x2 = x1;
                        }
                        else
                        {
                            n0 = e - (NT + NSEG); n1 = n0; n2 = n0;
                            x0 = fv(v3(0, 0, 0) + TGP(n0)); x1 = x0; x2 = x0;
                        }
                        if (!isT && !isS)
                        {
                            if (!fqPointContact(fs, x0, CR, nn, g)) continue;
                            q = x0;
                        }
                        else
                        {
                            if (fqSeparated(fs, x0, x1, x2, CR)) continue;
                            if (fs.kind == FQ_BOX && SURF_STAGED != 0)
                            {
                                // (A degenerate element has no contact: the queries' first test.)
                                if (isT) { if (!(fqLen(scross(ssub3(x1, x0), ssub3(x2, x0))) > 1e-8f)) continue; }
                                else { float3 d = ssub3(x1, x0); if (!(sdot(d, d) > 1e-12f)) continue; }
                                const uint h = uint(atomic_fetch_add_explicit(&tgS[isT ? 12 : 13], 1, memory_order_relaxed));
                                if (h < (isT ? HTRI : HSEG))
                                {
                                    const uint hh = isT ? h : HTRI + h;
                                    tgG[SHB + hh * 2u] = as_type<float>(n0 | n1 << 10 | n2 << 20);
                                    tgG[SHB + hh * 2u + 1u] = as_type<float>(pw);
                                    atomic_store_explicit(HPGX(hh), 0x7FFFFFFF, memory_order_relaxed);
                                    atomic_store_explicit(HPKC(hh), 0u, memory_order_relaxed);
                                    continue;
                                }
                                // (No room for another staged pair: the whole query here, the
                                // same result.)
                            }
                            if (isT) { if (!fqContact(fs, x0, x1, x2, CR, w, nn, g, q)) continue; }
                            else if (!fqSegContact(fs, x0, x1, CR, w, nn, g, q, two, w2, q2)) continue;
                        }
                        if (SKIP(23)) continue;
                        SURF_EMIT(n0, n1, n2, slot, w, g, nn, q, two, w2, q2)
                    }
                    BAR;
                    const uint nTriH = min(uint(atomic_load_explicit(&tgS[12], memory_order_relaxed)), HTRI), nSegH = min(uint(atomic_load_explicit(&tgS[13], memory_order_relaxed)), HSEG);
#ifdef EMU_TRACE
                    if (lid == 0u) fprintf(stderr, "T car %u step %u: staged pairs %u + %u of %u + %u\n", car, step, uint(atomic_load_explicit(&tgS[12], memory_order_relaxed)), uint(atomic_load_explicit(&tgS[13], memory_order_relaxed)), (uint)HTRI, (uint)HSEG);
#endif
                    if ((nTriH | nSegH) != 0u && !SKIP(24))
                    {
                        // A staged pair's nodes, element and shape, from the scalars' array.
                        #define SURF_PAIR(isT_, hh_) \
                            const uint nw_ = as_type<uint>(tgG[SHB + (hh_) * 2u]); \
                            const uint n0 = nw_ & 1023u, n1 = (nw_ >> 10) & 1023u, n2 = nw_ >> 20; \
                            const uint slot_ = as_type<uint>(tgG[SHB + (hh_) * 2u + 1u]) >> 16; \
                            SURF_SHAPE(fs, slot_) \
                            float3 x0 = fv(v3(0, 0, 0) + TGP(n0)), x1 = fv(v3(0, 0, 0) + TGP(n1)), x2 = (isT_) ? fv(v3(0, 0, 0) + TGP(n2)) : x1;
                        // Stage A, a lane a task (pair, box edge), triangles' from lane 0 up and
                        // beams' from the last lane down, so a warp runs one kind: an edge
                        // farther than the band from the element's box has no candidate that
                        // counts (fqFar); else each candidate's gap, and the pair's least (the
                        // walk's first pass).
                        #define SURF_TASK_A(isT_, t_, h0_) { \
                            const uint hh_ = (h0_) + (t_) / 13u, ed_ = (t_) % 13u; \
                            SURF_PAIR(isT_, hh_) \
                            float3 e0 = float3(0.0f, 0.0f, 0.0f), e1 = e0; \
                            bool far_ = false; \
                            if (ed_ < 12u) { fqWalkEdge(fs, ed_, e0, e1); far_ = fqFar(e0, e1, min(min(x0, x1), x2), max(max(x0, x1), x2), CR); } \
                            float gs_[6] = {3.0e38f, 3.0e38f, 3.0e38f, 3.0e38f, 3.0e38f, 3.0e38f}; \
                            float gmin_ = 3.0e38f; \
                            if (!far_) { \
                                for (uint i_ = 0u; i_ < 6u; i_++) { \
                                    FqAcc a = fqAccNew(); \
                                    a.ilo = i_; a.ihi = i_ + 1u; \
                                    surfCand(isT_, fs, e0, e1, x0, x1, x2, ed_, a); \
                                    gs_[i_] = a.gx; gmin_ = fmin(gmin_, a.gx); } \
                                taskG4[(hh_ * 13u + ed_) * 2u] = float4(gs_[0], gs_[1], gs_[2], gs_[3]); \
                                atomic_fetch_min_explicit(HPGX(hh_), ordF(gmin_), memory_order_relaxed); } \
                            taskG4[(hh_ * 13u + ed_) * 2u + 1u] = float4(gs_[4], gs_[5], gmin_, 0.0f); }
                        for (uint t = lid; t < nTriH * 13u; t += LANES) SURF_TASK_A(true, t, 0u)
                        for (uint t = (LANES - 1u) - lid; t < nSegH * 13u; t += LANES) SURF_TASK_A(false, t, HTRI)
                        BAR;
                        // Stage B: a task's candidates within the tie of the pair's least gap,
                        // each walked again alone (the pair's second pass: its sum is the
                        // candidate) and recorded with its place in the walk.
                        #define SURF_TASK_B(isT_, t_, h0_) { \
                            const uint hh_ = (h0_) + (t_) / 13u, ed_ = (t_) % 13u; \
                            const float4 gB_ = taskG4[(hh_ * 13u + ed_) * 2u + 1u]; \
                            const float GX_ = unordF(atomic_load_explicit(HPGX(hh_), memory_order_relaxed)); \
                            if (GX_ <= CR && gB_.z <= nf(GX_ + FQ_TIE)) { \
                                const float4 gA_ = taskG4[(hh_ * 13u + ed_) * 2u]; \
                                SURF_PAIR(isT_, hh_) \
                                float3 e0 = float3(0.0f, 0.0f, 0.0f), e1 = e0; \
                                if (ed_ < 12u) fqWalkEdge(fs, ed_, e0, e1); \
                                for (uint i_ = 0u; i_ < 6u; i_++) { \
                                    const float gi_ = i_ < 4u ? gA_[i_] : gB_[i_ - 4u]; \
                                    if (!(gi_ <= nf(GX_ + FQ_TIE))) continue; \
                                    FqAcc a = fqAccNew(); \
                                    a.gx = GX_; a.pass = 1u; a.ilo = i_; a.ihi = i_ + 1u; \
                                    surfCand(isT_, fs, e0, e1, x0, x1, x2, ed_, a); \
                                    if (a.k != 1u) continue; \
                                    const uint idx_ = atomic_fetch_add_explicit(HPKC(hh_), 1u, memory_order_relaxed); \
                                    if (idx_ < KCAP) { \
                                        device float4* rec_ = HPREC(hh_) + idx_ * 3u; \
                                        rec_[0] = f4(a.w, as_type<float>(ed_ * 8u + i_)); \
                                        rec_[1] = f4(a.q, 0.0f); rec_[2] = f4(a.cs, 0.0f); } } } }
                        if (!SKIP(25))
                        {
                            for (uint t = lid; t < nTriH * 13u; t += LANES) SURF_TASK_B(true, t, 0u)
                            for (uint t = (LANES - 1u) - lid; t < nSegH * 13u; t += LANES) SURF_TASK_B(false, t, HTRI)
                        }
                        BAR;
                        // Stage R, a lane a pair: the query's steps between its walks
                        // (triangle_contact, segment_contact), its later passes over the kept
                        // candidates in the walk's order.
                        #define SURF_ORDER(hh_, kc_) \
                            uint keys_[KCAP]; \
                            for (uint j_ = 0u; j_ < kc_; j_++) keys_[j_] = as_type<uint>(HPREC(hh_)[j_ * 3u].w); \
                            uint ordA_ = 0u, ordB_ = 0u; \
                            for (uint j_ = 0u; j_ < kc_; j_++) { \
                                uint rank_ = 0u; \
                                for (uint i_ = 0u; i_ < kc_; i_++) rank_ += keys_[i_] < keys_[j_] ? 1u : 0u; \
                                if (rank_ < 8u) ordA_ |= j_ << (rank_ * 4u); else ordB_ |= j_ << ((rank_ - 8u) * 4u); }
                        #define SURF_REC(hh_, r_) (HPREC(hh_) + (((r_) < 8u ? ordA_ >> ((r_) * 4u) : ordB_ >> (((r_) - 8u) * 4u)) & 15u) * 3u)
                        if (lid < nTriH && !SKIP(26))
                        {
                            const uint hh = lid;
                            const float GX = unordF(atomic_load_explicit(HPGX(hh), memory_order_relaxed));
                            if (GX <= CR)
                            {
                                SURF_PAIR(true, hh)
                                bool ok = true, hit = false, two = false;
                                for (uint k = 0u; k < 3u; k++)
                                {
                                    float2 pd = fqPointDist(fs, fqX(x0, x1, x2, k));
                                    if (pd.y != 0.0f && pd.x <= nf(GX + FQ_TIE)) ok = false;
                                }
                                const uint kc = atomic_load_explicit(HPKC(hh), memory_order_relaxed);
                                float3 w = float3(0.0f, 0.0f, 0.0f), nn = w, q = w, w2 = w, q2 = w;
                                float g = GX;
                                if (ok && (kc == 0u || kc > KCAP))
                                {
#ifdef EMU_TRACE
                                    fprintf(stderr, "FALLBACK tri kc %u\n", kc);
#endif
                                    // (More candidates at the least gap than a pair keeps: the
                                    // whole query, the same result.)
                                    hit = fqContact(fs, x0, x1, x2, CR, w, nn, g, q);
                                }
                                else if (ok)
                                {
                                    SURF_ORDER(hh, kc)
                                    float3 aw = float3(0.0f, 0.0f, 0.0f), aq = aw, acs = aw;
                                    for (uint r = 0u; r < kc; r++)
                                    {
                                        device const float4* rec = SURF_REC(hh, r);
                                        aw = float3(nf(aw.x + rec[0].x), nf(aw.y + rec[0].y), nf(aw.z + rec[0].z));
                                        aq = sadd3(aq, fv(rec[1])); acs = sadd3(acs, fv(rec[2]));
                                    }
                                    float kf = float(kc);
                                    w = float3(sdiv(aw.x, kf), sdiv(aw.y, kf), sdiv(aw.z, kf));
                                    q = sdiv3(aq, kf);
                                    float3 cs = sdiv3(acs, kf);
                                    const uint nz = (w.x > 0.0f ? 1u : 0u) + (w.y > 0.0f ? 1u : 0u) + (w.z > 0.0f ? 1u : 0u);
                                    if (nz > 1u)
                                    {
                                        float3 dv = ssub3(cs, q);
                                        float dl = fqLen(dv);
                                        hit = true;
                                        if (GX > FQ_PIERCE && dl > FQ_PIERCE) nn = sdiv3(dv, dl);
                                        else
                                        {
                                            // Touching or crossing: the overlap axis, the deepest
                                            // shared candidates along it, their mean.
                                            float3 nc = scross(ssub3(x1, x0), ssub3(x2, x0));
                                            nn = fqOverlapAxis(fs, x0, x1, x2, sdiv3(nc, fqLen(nc)));
                                            bool any = false;
                                            float top = 0.0f;
                                            for (uint r = 0u; r < kc; r++)
                                            {
                                                device const float4* rec = SURF_REC(hh, r);
                                                if (!fqShared(fv(rec[0]))) continue;
                                                float d = sdot(nn, fv(rec[1]));
                                                top = any ? max(top, d) : d;
                                                any = true;
                                            }
                                            if (any)
                                            {
                                                aw = float3(0.0f, 0.0f, 0.0f); aq = aw;
                                                uint k3 = 0u;
                                                for (uint r = 0u; r < kc; r++)
                                                {
                                                    device const float4* rec = SURF_REC(hh, r);
                                                    if (!(fqShared(fv(rec[0])) && sdot(nn, fv(rec[1])) >= nf(top - FQ_TIE))) continue;
                                                    aw = float3(nf(aw.x + rec[0].x), nf(aw.y + rec[0].y), nf(aw.z + rec[0].z));
                                                    aq = sadd3(aq, fv(rec[1]));
                                                    k3++;
                                                }
                                                float kf3 = float(k3);
                                                w = float3(sdiv(aw.x, kf3), sdiv(aw.y, kf3), sdiv(aw.z, kf3));
                                                q = sdiv3(aq, kf3);
                                            }
                                        }
                                    }
                                }
                                if (hit && !SKIP(23)) SURF_EMIT(n0, n1, n2, slot_, w, g, nn, q, two, w2, q2)
                            }
                        }
                        if (lid >= LANES - nSegH && !SKIP(26))
                        {
                            const uint hh = HTRI + ((LANES - 1u) - lid);
                            const float GX = unordF(atomic_load_explicit(HPGX(hh), memory_order_relaxed));
                            if (GX <= CR)
                            {
                                SURF_PAIR(false, hh)
                                bool ok = true, hit = false, two = false;
                                for (uint k = 0u; k < 2u; k++)
                                {
                                    float2 pd = fqPointDist(fs, k == 0u ? x0 : x1);
                                    if (pd.y != 0.0f && pd.x <= nf(GX + FQ_TIE)) ok = false;
                                }
                                const uint kc = atomic_load_explicit(HPKC(hh), memory_order_relaxed);
                                float3 w = float3(0.0f, 0.0f, 0.0f), nn = w, q = w, w2 = w, q2 = w;
                                float g = GX;
                                if (ok && (kc == 0u || kc > KCAP))
                                {
#ifdef EMU_TRACE
                                    fprintf(stderr, "FALLBACK seg kc %u\n", kc);
#endif
                                    hit = fqSegContact(fs, x0, x1, CR, w, nn, g, q, two, w2, q2);
                                }
                                else if (ok)
                                {
                                    SURF_ORDER(hh, kc)
                                    float3 aw = float3(0.0f, 0.0f, 0.0f), aq = aw, acs = aw;
                                    for (uint r = 0u; r < kc; r++)
                                    {
                                        device const float4* rec = SURF_REC(hh, r);
                                        aw = float3(nf(aw.x + rec[0].x), nf(aw.y + rec[0].y), nf(aw.z + rec[0].z));
                                        aq = sadd3(aq, fv(rec[1])); acs = sadd3(acs, fv(rec[2]));
                                    }
                                    float kf = float(kc);
                                    w = float3(sdiv(aw.x, kf), sdiv(aw.y, kf), 0.0f);
                                    q = sdiv3(aq, kf);
                                    float3 cs = sdiv3(acs, kf);
                                    if (w.x > 0.0f && w.y > 0.0f)
                                    {
                                        float3 dv = ssub3(cs, q);
                                        float dl = fqLen(dv);
                                        hit = true;
                                        if (GX > FQ_PIERCE && dl > FQ_PIERCE) nn = sdiv3(dv, dl);
                                        else
                                        {
                                            // Touching or crossing: of the deepest shared candidates
                                            // along the overlap axis, the beam's two extremes.
                                            nn = fqSegOverlapAxis(fs, x0, x1);
                                            bool any = false;
                                            float top = 0.0f;
                                            for (uint r = 0u; r < kc; r++)
                                            {
                                                device const float4* rec = SURF_REC(hh, r);
                                                if (!fqShared(fv(rec[0]))) continue;
                                                float d = sdot(nn, fv(rec[1]));
                                                top = any ? max(top, d) : d;
                                                any = true;
                                            }
                                            if (any)
                                            {
                                                float3 lw = float3(0.0f, 0.0f, 0.0f), lq = lw, hw = lw, hq = lw;
                                                uint k4 = 0u;
                                                for (uint r = 0u; r < kc; r++)
                                                {
                                                    device const float4* rec = SURF_REC(hh, r);
                                                    float3 pw3 = fv(rec[0]), pq3 = fv(rec[1]);
                                                    if (!(fqShared(pw3) && sdot(nn, pq3) >= nf(top - FQ_TIE))) continue;
                                                    if (k4 == 0u || pw3.y < lw.y) { lw = pw3; lq = pq3; }
                                                    if (k4 == 0u || pw3.y > hw.y) { hw = pw3; hq = pq3; }
                                                    k4 = 1u;
                                                }
                                                w = lw; q = lq;
                                                if (hw.y > lw.y) { two = true; w2 = hw; q2 = hq; }
                                            }
                                        }
                                    }
                                }
                                if (hit && !SKIP(23)) SURF_EMIT(n0, n1, n2, slot_, w, g, nn, q, two, w2, q2)
                            }
                        }
                        BAR;
                    }
                    const uint nsHits = uint(atomic_load_explicit(&tgS[9], memory_order_relaxed));
#ifdef EMU_TRACE
                    if (lid == 0u) fprintf(stderr, "T car %u step %u: surface hits %u\n", car, step, nsHits);
#endif
                    if (nsHits != 0u)
                    {
                        const uint nsh = min(nsHits, SHIT_CAP);
                        if (lid == 0)
                        {
                            tgFlags[F_HITS] = 1u;
                            if (nsHits > SHIT_CAP) atomic_fetch_or_explicit(&tgS[11], 32, memory_order_relaxed);
                        }
                        // The hits applied, a lane a hit (apply_surface_hits): each is
                        // independent of the others (the forces and the counts are the pass's
                        // start's, the sums fixed point).
                        if (lid < nsh)
                        {
                            const uint4 id = as_type<uint4>(surfHit[lid * 4u]);
                            const float4 wg = surfHit[lid * 4u + 1u];
                            const V3 nn = v3(surfHit[lid * 4u + 2u]), q = v3(surfHit[lid * 4u + 3u]);
                            const uint slot = id.w;
                            const uint ns[3] = {id.x, id.y, id.z};
                            const float w[3] = {wg.x, wg.y, wg.z};
                            sHitN = uint3(id.x, id.y, id.z);
                            // (A node's mass 0: immovable; its count the hits that name it.)
                            float ms[3], fr[3], ims[3];
                            for (uint j = 0u; j < 3u; j++)
                            {
                                float4 nst = nodeStatic[ns[j]];
                                ms[j] = (as_type<uint>(nst.z) & NODE_IMMOVABLE) != 0u ? 0.0f : nst.x;
                                fr[j] = nst.y;
                                ims[j] = ms[j] == 0.0f ? 0.0f : float(atomic_load_explicit(&split[ns[j]], memory_order_relaxed)) / ms[j];
                            }
                            #define STERM(j_) (w[j_] > 0.0f ? (w[j_] * w[j_]) * ims[j_] : 0.0f)
                            #define SSF(j_) v3(tgG[SFB + ns[j_] * 3u], tgG[SFB + ns[j_] * 3u + 1u], tgG[SFB + ns[j_] * 3u + 2u])
                            #define SSACC(j_) (ms[j_] == 0.0f ? v3(0, 0, 0) : SSF(j_) / ms[j_])
                            const float imt = (STERM(0) + STERM(1)) + STERM(2);
                            const V3 vt = (TGV(ns[0]) * w[0] + TGV(ns[1]) * w[1]) + TGV(ns[2]) * w[2];
                            const V3 at = (SSACC(0) * w[0] + SSACC(1) * w[1]) + SSACC(2) * w[2];
                            float imsShape = 0.0f;
                            V3 vs = v3(0, 0, 0);
                            const uint bi = as_type<uint>(tgG[SB(slot) + 22u]);
#if RIGID
                            BodyK bk;
                            if (bi != SURF_STATIC)
                            {
                                device const float4* bd = rigBody + bi * BDW;
                                V3 arm = q - v3(bd[0]);
                                vs = v3(bd[2]) + cross3(v3(bd[3]), arm);
                                V3 rn = cross3(arm, nn);
                                float4 rot = bd[1];
                                V3 ii = v3(bd[4]);
                                V3 l = vf(fqRot(float4(-rot.x, -rot.y, -rot.z, rot.w), fv(rn)));
                                V3 irn = vf(fqRot(rot, float3(ii.x * l.x, ii.y * l.y, ii.z * l.z)));
                                float kb = float(max(atomic_load_explicit(&surfBH[bi], memory_order_relaxed), 1u)) * rmax(bd[2].w, 1.0f);
                                imsShape = kb * (bd[0].w + dot3(rn, irn));
                                bk.k = kb; bk.im = bd[0].w; bk.imt = imt; bk.arm = arm; bk.rot = rot; bk.ii = ii;
                            }
#endif
                            const float inv = imsShape + imt;
                            const float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                            const V3 fh = (v3(0, 0, 0) - at) * m;
                            const float friction = (fr[0] * w[0] + fr[1] * w[1]) + fr[2] * w[2];
                            const float p = rmin(CR - wg.w, CR);
                            float budget;
                            V3 fc = surfaceResponse(fh, vs - vt, m, nn, BOXS, p, friction, budget);
                            if (!isZero3(fc))
                            {
#if RIGID
                                if (bi != SURF_STATIC) fc = noPositiveWorkBody(fc, fh, vs - vt, m, nn, bk);
#endif
                                // §37: the position step (the element's share; a body takes none).
                                float delta = shiftDepth(p, budget, dot3(fh, nn));
                                if (delta > 0.0f)
                                {
                                    float qd = delta * m;
                                    for (uint j = 0u; j < 3u; j++)
                                    {
                                        if (!(w[j] > 0.0f)) continue;
                                        V3 d = nn * -(w[j] * (qd * ims[j]));
                                        for (uint a = 0u; a < 3u; a++) atomic_fetch_add_explicit((device atomic_int*)&shift[ns[j] * 3u + a], toFixedShift(a == 0u ? d.x : (a == 1u ? d.y : d.z)), memory_order_relaxed);
                                    }
                                    tgFlags[F_SHIFT] = 1u;
                                }
                                for (uint j = 0u; j < 3u; j++)
                                {
                                    V3 f = -(fc * w[j]);
                                    atomic_fetch_add_explicit(&acc[ns[j] * 3u], toFixed(f.x), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&acc[ns[j] * 3u + 1u], toFixed(f.y), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&acc[ns[j] * 3u + 2u], toFixed(f.z), memory_order_relaxed);
                                }
#if RIGID
                                if (bi != SURF_STATIC)
                                {
                                    // add_surface_reaction: the force on the shape at the contact point.
                                    const uint k = as_type<uint>(tgG[SB(slot) + 23u]);
                                    V3 mom = cross3(q - v3(rigShape[k * SHW + 2]), fc);
                                    atomic_fetch_add_explicit(&rigSum[k * 6], int(round(fc.x * 32.0f)), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&rigSum[k * 6 + 1], int(round(fc.y * 32.0f)), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&rigSum[k * 6 + 2], int(round(fc.z * 32.0f)), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&rigSum[k * 6 + 3], int(round(mom.x * 32.0f)), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&rigSum[k * 6 + 4], int(round(mom.y * 32.0f)), memory_order_relaxed);
                                    atomic_fetch_add_explicit(&rigSum[k * 6 + 5], int(round(mom.z * 32.0f)), memory_order_relaxed);
                                }
#endif
                            }
                        }
                        BAR;
                        // (The counts cleared for the next step: nothing in the phases that
                        // follow reads them.)
                        if (lid < nsh)
                        {
                            atomic_store_explicit(&split[sHitN.x], 0u, memory_order_relaxed);
                            atomic_store_explicit(&split[sHitN.y], 0u, memory_order_relaxed);
                            atomic_store_explicit(&split[sHitN.z], 0u, memory_order_relaxed);
                        }
#if RIGID
                        if (lid < NB) atomic_store_explicit(&surfBH[lid], 0u, memory_order_relaxed);
#endif
                    }
                }
            }
        }
#endif
#if RIGID
        // ---- the proxies integrate the step's reactions (end_step) ----
        if (lid < NB && !SKIP(13))
        {
            device float4* bd = rigBody + lid * BDW;
            V3 bpos = v3(bd[0]), bvel = v3(bd[2]), bang = v3(bd[3]);
            float4 brot = bd[1];
            V3 force = v3(0, 0, 0), torque = v3(0, 0, 0);
            for (uint k = 0; k < NSH; k++)
            {
                device float4* sh = rigShape + k * SHW;
                if (as_type<uint>(sh[0].x) != lid) continue;
                V3 f = v3(float(atomic_exchange_explicit(&rigSum[k * 6], 0, memory_order_relaxed)) / 32.0f, float(atomic_exchange_explicit(&rigSum[k * 6 + 1], 0, memory_order_relaxed)) / 32.0f, float(atomic_exchange_explicit(&rigSum[k * 6 + 2], 0, memory_order_relaxed)) / 32.0f);
                V3 m = v3(float(atomic_exchange_explicit(&rigSum[k * 6 + 3], 0, memory_order_relaxed)) / 32.0f, float(atomic_exchange_explicit(&rigSum[k * 6 + 4], 0, memory_order_relaxed)) / 32.0f, float(atomic_exchange_explicit(&rigSum[k * 6 + 5], 0, memory_order_relaxed)) / 32.0f);
                sh[3] = f4(v3(sh[3]) + f * H, 0);
                sh[4] = f4(v3(sh[4]) + m * H, 0);
                float4 h0 = sh[0];
                h0.w = rmax(h0.w, len3(f));
                sh[0] = h0;
                force = force + f;
                torque = torque + (m + cross3(v3(sh[2]) - bpos, f));
            }
            if (bd[0].w != 0.0f)
            {
                bvel = bvel + force * (bd[0].w * H);
                bang = bang + invInertiaMul(brot, v3(bd[4]), torque) * H;
            }
            bpos = bpos + bvel * H;
            if (!isZero3(bang))
            {
                float4 dq = qmul(float4(bang.x, bang.y, bang.z, 0.0f), brot);
                brot = qnormalize(float4(brot.x + (0.5f * H) * dq.x, brot.y + (0.5f * H) * dq.y, brot.z + (0.5f * H) * dq.z, brot.w + (0.5f * H) * dq.w));
            }
            bd[0] = f4(bpos, bd[0].w);
            bd[1] = brot;
            bd[2] = f4(bvel, bd[2].w);
            bd[3] = f4(bang, 0);
            // (The count as the reference's begin_step left it, for the host; then cleared.)
            bd[4] = f4(v3(bd[4]), as_type<float>(atomic_exchange_explicit(&rigCount[lid], 0u, memory_order_relaxed)));
        }
        BAR;
#endif
#if PAIR
//@PAIR_PASS
#endif
    }

    // ---- write back ----
    if (isNode)
    {
        nodeState[ni * 3] = f4(Pown, 0);
        nodeState[ni * 3 + 1] = f4(Vown, 0);
        nodeState[ni * 3 + 2] = f4(F0, 0);
    }
    hot[lid * 6 + 4] = as_type<float4>(uint4(w0[0], w0[1], w0[2], w0[3]));
    hot[lid * 6 + 5] = float4(G, L, 0.0f, 0.0f);
    for (uint s = 0; s < RS; s++)
    {
        float4 hs = HOTS(s);
        hs.w = up[s];
#if REGMODE < 1
        hs.x = kk[s]; hs.z = rest[s];
#endif
        HOTS(s) = hs;
        // The records' copies of the two words the hot path keeps (the last extension rate
        // and the damping slope), when the host asks to read the beams (P.pad2 bit 0).
        uint w = w0[s];
        if ((P.pad2 & 1u) == 0u || w == NONE || (w >> 30) >= C_AERO) continue;
        uint r = perm[s * LANES + lid];
        recState[r * 3 + 2] = float4(up[s], 0.0f, 0.0f, 0.0f);
        float4 s1 = recState[r * 3 + 1];
        s1.z = (w & W_SLOPE) != 0u ? hs.y : 0.0f;
        recState[r * 3 + 1] = s1;
    }
    if (lid < NG)
    {
        // (As the next dispatch's first node pass would find them.)
        if (tgFlags[F_FIRED] != 0u) groupsLast[lid] = atomic_load_explicit(&masters[lid], memory_order_relaxed) != NONE ? 1u : 0u;
        else if (tgFlags[F_GDIRTY] == 1u) groupsLast[lid] = 0u;
    }
    if (lid == 0)
    {
#if SURF
        tgFlags[F_STATUS] = tgFlags[F_STATUS] | uint(atomic_load_explicit(&tgS[11], memory_order_relaxed));
#endif
        misc[0] = tgFlags[F_SHIFT]; misc[1] = tgFlags[F_STATUS]; misc[3] = tgFlags[F_REBUILD];
        misc[13] = tgFlags[F_OUTER];
        misc[4] = tgFlags[F_HITS]; misc[5] = tgFlags[F_BUILDS]; misc[6] = tgFlags[F_MOST];
        misc[7] = tgFlags[F_REC]; misc[8] = tgFlags[F_HELD]; misc[9] = tgFlags[F_FIRED] != 0u ? 2u : (tgFlags[F_GDIRTY] == 2u ? 2u : 0u); misc[10] = tgFlags[F_BROKE];
        misc[15] = 2u;
        misc[11] = 1u;
        uint np = 0u;
        misc[2] = np;
    }
}
