// ---- absorbing boundary (RegionBoundary::Absorbing) ----------------------------
//
// The ground around an activated region is an elastic half-space of the
// region's own material, not a rigid wall. Its faces are the floor (the bed
// plane the window was lowered to) and the window's four sides up to the
// unactivated ground's surface. On each face the ground pushes on the material
// with the traction
//   t = p0·n − (k_N·u_n + ρ·c_p·v_n)·n − (k_T·u_t + ρ·c_s·v_t),
// n into the material, v the material's velocity at the face and u the face's
// displacement since activation:
// - the dashpots are Lysmer & Kuhlemeyer's (1969), exact for plane waves at
//   normal incidence;
// - the springs are the viscous-spring boundary's (Liu et al. 2006),
//   k_N = (4/3)·μ/R and k_T = (2/3)·μ/R with R the distance to the region's
//   centre: the static stiffness of the ground beyond, so a change of load
//   gives a bounded displacement instead of creep;
// - p0 is the geostatic traction at activation (ρ·g·d on the floor, K0·ρ·g·d
//   on the sides, K0 = λ/(λ + 2μ)), the stress the material starts with
//   (InitialStress::Geostatic), so material at rest stays at rest.
// The ground has the strength of granular ground: a face carries no tension
// (one whose normal traction would pull lets the material go: sand separates
// from the ground), and no more shear than the bed friction μ allows (one
// whose tangential traction would pass μ·t_n slides at μ·t_n). The dashpots
// model small-strain waves, not plastic flow along the face, which would
// otherwise meet a viscous drag ρ·c_s·v_t far above any frictional strength.
// It is discretised like any traction on the grid: node i takes ∫ w_i·t dS over
// the faces (w_i its B-spline), lumped, so its share of a face is ∫ w_i dS and
// the dashpot acts on its own velocity. That is the force the particles'
// geostatic stress transmits through the face, so the activated state is in
// equilibrium (a floor half a cell above a node layer splits between two
// layers). Each node is solved implicitly (stable for any dt).
// The displacement u belongs to the face, not to the nodes: one record per
// face point (the node lattice's columns through the face, mpmFaceRecord),
// moved once per substep, after G2P, by the material's velocity there (the
// mass-weighted B-spline mean of the node layers across the face,
// mpm_absorb_faces). Every node the face reaches springs against that one
// displacement. A record per node instead integrated each node's own Eulerian
// velocity: under the strike's flow the node layers across a face drifted
// metres apart, and their springs then pushed opposite ways across it.

// The region's absorbing faces in grid units (cells from the region's origin):
// the window's low and high corners (lo.y is the floor).
struct MpmAbsorbGeom
{
    float3 lo;
    float3 hi;
};

MpmAbsorbGeom mpmAbsorbGeometry(HeightFieldRef hf, Frame f, MpmParams p)
{
    float dx = p.origin.w;
    float3 lo = float3(hf.x0 + (float(p.window.x) - 0.5) * hf.dx, p.column.x,
        hf.z0 + (float(p.window.y) - 0.5) * hf.dz);
    float3 hi = float3(hf.x0 + (float(p.window.x + p.window.z) - 0.5) * hf.dx, p.column.x,
        hf.z0 + (float(p.window.y + p.window.w) - 0.5) * hf.dz);
    MpmAbsorbGeom a;
    a.lo = (frameToWorld(f, lo) - p.origin.xyz) / dx;
    a.hi = (frameToWorld(f, hi) - p.origin.xyz) / dx;
    return a;
}

// The five faces: floor, low x, high x, low z, high z. Each one's normal axis,
// the sign of its normal into the material, and its plane (grid units).
uint mpmFaceAxis(uint k) { return k == 0u ? 1u : (k < 3u ? 0u : 2u); }

float mpmFaceSign(uint k) { return (k == 2u || k == 4u) ? -1.0 : 1.0; }

float3 mpmAxisVector(uint axis)
{
    return float3(axis == 0u ? 1.0 : 0.0, axis == 1u ? 1.0 : 0.0, axis == 2u ? 1.0 : 0.0);
}

float mpmFacePlane(uint k, MpmAbsorbGeom a)
{
    return k == 0u ? a.lo.y : (k == 1u ? a.lo.x : (k == 2u ? a.hi.x : (k == 3u ? a.lo.z : a.hi.z)));
}

// Face records: the floor's points over the node lattice's (x, z) columns,
// then each x side's over (y, z), then each z side's over (y, x)
// (TerrainPlan::boundary_nodes counts them). The record of face k at node g's
// column.
uint mpmFaceRecord(uint k, int3 g, MpmParams p)
{
    int3 n = int3(p.dims.xyz) + 1;
    if (any(g < int3(0, 0, 0)) || any(g >= n)) return MPM_NO_SLOT;
    if (k == 0u) return uint(g.z * n.x + g.x);
    uint base = uint(n.x * n.z);
    if (k < 3u) return base + (k - 1u) * uint(n.y * n.z) + uint(g.y * n.z + g.z);
    base += 2u * uint(n.y * n.z);
    return base + (k - 3u) * uint(n.y * n.x) + uint(g.y * n.x + g.x);
}

// Face records in all (TerrainPlan::boundary_nodes).
uint mpmFaceCount(MpmParams p)
{
    uint3 n = p.dims.xyz + 1u;
    return n.x * n.z + 2u * n.y * (n.x + n.z);
}

// The face and point (grid units, on its plane) of record `slot`.
float3 mpmFacePoint(uint slot, MpmAbsorbGeom a, MpmParams p, out uint k)
{
    int3 n = int3(p.dims.xyz) + 1;
    int s = int(slot);
    k = 0u;
    if (s < n.x * n.z) return float3(float(s % n.x), a.lo.y, float(s / n.x));
    s -= n.x * n.z;
    if (s < 2 * n.y * n.z)
    {
        k = s < n.y * n.z ? 1u : 2u;
        s %= n.y * n.z;
        return float3(mpmFacePlane(k, a), float(s / n.z), float(s % n.z));
    }
    s -= 2 * n.y * n.z;
    k = s < n.y * n.x ? 3u : 4u;
    s %= n.y * n.x;
    return float3(float(s % n.x), float(s / n.x), mpmFacePlane(k, a));
}

// The quadratic B-spline (in cells) and its integral from −∞ to u.
float mpmSpline(float u)
{
    float a = abs(u);
    return a < 0.5 ? 0.75 - a * a : (a < 1.5 ? 0.5 * (1.5 - a) * (1.5 - a) : 0.0);
}

float mpmSplineCdf(float u)
{
    if (u <= -1.5) return 0.0;
    if (u >= 1.5) return 1.0;
    if (u < -0.5)
    {
        float a = u + 1.5;
        return a * a * a / 6.0;
    }
    if (u < 0.5) return 0.5 + u * (0.75 - u * u / 3.0);
    float b = 1.5 - u;
    return 1.0 - b * b * b / 6.0;
}

// ∫ w over [lo, hi] (cells) for a node at x: its share of that span.
float mpmSplineSpan(float lo, float hi, float x)
{
    return hi > lo ? mpmSplineCdf(hi - x) - mpmSplineCdf(lo - x) : 0.0;
}

// The unactivated ground's surface at sample (i, j) beside the window, in grid units.
float mpmGroundTop(HeightFieldRef hf, Frame f, MpmParams p, int i, int j)
{
    float3 xl = float3(0.0, hfHeight(hf, uint(i), uint(j)), 0.0);
    return (frameToWorld(f, xl).y - p.origin.y) / p.origin.w;
}

// The window's samples nearest grid point r (clamped into the window).
int2 mpmWindowNearest(float3 r, HeightFieldRef hf, Frame f, MpmParams p)
{
    float3 xl = frameToLocal(f, p.origin.xyz + r * p.origin.w);
    int2 lo = int2(p.window.xy), hi = lo + int2(p.window.zw) - 1;
    return clamp(int2(round((xl.xz - float2(hf.x0, hf.z0)) / float2(hf.dx, hf.dz))), lo, hi);
}

// The top of side k (1..4) at the window sample s beside it: the unactivated
// ground's surface there.
float mpmSideTop(uint k, int2 s, HeightFieldRef hf, Frame f, MpmParams p)
{
    int2 lo = int2(p.window.xy), hi = lo + int2(p.window.zw) - 1;
    int2 o = k == 1u ? int2(lo.x - 1, s.y) : (k == 2u ? int2(hi.x + 1, s.y) : (k == 3u ? int2(s.x, lo.y - 1) : int2(s.x, hi.y + 1)));
    return mpmGroundTop(hf, f, p, o.x, o.y);
}

// The material's elastic constants for the faces.
struct MpmGround
{
    float rho, cp, cs, k0, weight, mu;
};

MpmGround mpmGroundOf(MpmParams p)
{
    MpmGround m;
    m.rho = p.column.w;
    float lam = p.sand.x;
    m.mu = p.sand.y;
    m.cp = sqrt((lam + 2.0 * m.mu) / m.rho);
    m.cs = sqrt(m.mu / m.rho);
    m.k0 = lam / (lam + 2.0 * m.mu);
    m.weight = m.rho * max(-p.step.y, 0.0) * p.origin.w;
    return m;
}

// The springs per unit area at grid point r: k_N, k_T.
float2 mpmFaceSprings(float3 r, MpmGround m, MpmParams p)
{
    float R = max(length(r - 0.5 * float3(p.dims.xyz)), 1.0) * p.origin.w;
    return float2(4.0 / 3.0, 2.0 / 3.0) * (m.mu / R);
}

// Face k's geostatic traction at height y (grid units) under a column whose
// surface (at activation) is at `top`, floor at ys.
float mpmFaceP0(uint k, float top, float y, float ys, MpmGround m)
{
    return k == 0u ? m.weight * max(top - ys, 0.0) : m.k0 * m.weight * max(top - max(y, ys), 0.0);
}

// Grid velocity v (after gravity) of node g, of mass m, on the absorbing faces.
// False when no face reaches the node: it keeps the rigid bed.
bool mpmAbsorbNode(int3 g, float m, HeightFieldRef hf, Frame f, MpmParams p, inout float3 v)
{
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    float3 r = float3(g);
    int2 s = mpmWindowNearest(r, hf, f, p);
    float ys = a.lo.y;
    float ix = mpmSplineSpan(a.lo.x, a.hi.x, r.x);
    float iz = mpmSplineSpan(a.lo.z, a.hi.z, r.z);
    // The node's shares of the five faces (cells²).
    float share[5];
    share[0] = mpmSpline(ys - r.y) * ix * iz;
    share[1] = mpmSpline(a.lo.x - r.x) * iz * mpmSplineSpan(ys, mpmSideTop(1u, s, hf, f, p), r.y);
    share[2] = mpmSpline(a.hi.x - r.x) * iz * mpmSplineSpan(ys, mpmSideTop(2u, s, hf, f, p), r.y);
    share[3] = mpmSpline(a.lo.z - r.z) * ix * mpmSplineSpan(ys, mpmSideTop(3u, s, hf, f, p), r.y);
    share[4] = mpmSpline(a.hi.z - r.z) * ix * mpmSplineSpan(ys, mpmSideTop(4u, s, hf, f, p), r.y);
    if (share[0] + share[1] + share[2] + share[3] + share[4] <= 0.0) return false;
    MpmGround gr = mpmGroundOf(p);
    float2 kk = mpmFaceSprings(r, gr, p);
    float kn = kk.x, kt = kk.y;
    float dx = p.origin.w, dt = p.step.x;
    float area = dx * dx;
    float mu_g = p.step.w;
    // Each face's displacement at the node's column, and its geostatic traction.
    float3 u[5];
    float p0[5];
    for (uint k = 0u; k < 5u; k++)
    {
        u[k] = float3(0, 0, 0);
        p0[k] = 0.0;
        if (!(share[k] > 0.0)) continue;
        uint rec = mpmFaceRecord(k, g, p);
        if (rec == MPM_NO_SLOT || rec >= p.boundary.z)
        {
            share[k] = 0.0;
            continue;
        }
        float4 d = gMpmCoupled[p.boundary.y + rec];
        u[k] = d.xyz;
        float top = (frameToWorld(f, float3(0.0, d.w, 0.0)).y - p.origin.y) / dx;
        p0[k] = mpmFaceP0(k, top, r.y, ys, gr);
    }
    float3 vNew = v;
    // Solve with every face sticking; then again with any face that would
    // pull let go, and any whose shear would pass the ground's friction
    // sliding (its tangential traction μ·t_n, along the sticking one).
    uint live = 31u, slide = 0u;
    float3 slip[5];
    for (uint k = 0u; k < 5u; k++) slip[k] = float3(0, 0, 0);
    for (uint pass = 0u; pass < 3u; pass++)
    {
        float3 damp = float3(0, 0, 0), stiff = float3(0, 0, 0), push = float3(0, 0, 0);
        for (uint k = 0u; k < 5u; k++)
        {
            if ((live & (1u << k)) == 0u || !(share[k] > 0.0)) continue;
            float sk = share[k] * area;
            float3 e = mpmAxisVector(mpmFaceAxis(k));
            float3 st = kn * e;
            damp += sk * gr.rho * gr.cp * e;
            push += sk * mpmFaceSign(k) * p0[k] * e;
            if ((slide & (1u << k)) != 0u) push += sk * slip[k];
            else
            {
                damp += sk * gr.rho * gr.cs * (1.0 - e);
                st += kt * (1.0 - e);
            }
            // The spring on the face's displacement, implicit in the node's
            // own velocity (the face moves with the material there).
            stiff += sk * st;
            push -= sk * st * u[k];
        }
        vNew = (m * v + dt * push) / (m + dt * damp + dt * dt * stiff);
        if (pass == 2u) break;
        uint keep = live, slip0 = slide;
        for (uint k = 0u; k < 5u; k++)
        {
            if ((live & (1u << k)) == 0u || !(share[k] > 0.0)) continue;
            uint ax = mpmFaceAxis(k);
            float3 uNew = u[k] + dt * vNew;
            float sg = mpmFaceSign(k);
            float tn = p0[k] - kn * sg * uNew[ax] - gr.rho * gr.cp * sg * vNew[ax];
            if (tn < 0.0)
            {
                keep &= ~(1u << k);
                continue;
            }
            if ((slide & (1u << k)) != 0u) continue;
            float3 tt = -(kt * uNew + gr.rho * gr.cs * vNew) * (1.0 - mpmAxisVector(ax));
            // Its two components, without squaring tiny ones.
            float t = ax == 1u ? mpmHypot(tt.x, tt.z) : (ax == 0u ? mpmHypot(tt.y, tt.z) : mpmHypot(tt.x, tt.y));
            if (t > mu_g * tn)
            {
                slide |= 1u << k;
                slip[k] = tt * (mu_g * tn / t);
            }
        }
        if (keep == live && slide == slip0) break;
        live = keep;
    }
    v = vNew;
    return true;
}

// Once at activation, per face record: no displacement yet, and the surface
// height of the column over the face point (the window sample nearest it).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_absorb_init(uint3 tid : SV_DispatchThreadID)
{
    uint slot = mpmIndex(tid);
    MpmParams p = mpmSelArea();
    if (slot >= min(p.boundary.z, mpmFaceCount(p)) || mpmActivationRefused()) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    uint k;
    int2 s = mpmWindowNearest(mpmFacePoint(slot, a, p, k), hf, f, p);
    uint t = uint(s.y - int(p.window.y)) * p.window.z + uint(s.x - int(p.window.x));
    float dh = float(ALOAD(gMpmColumns[mpmCol(p, t)])) * p.column.y / mpmSampleArea(hf);
    float3 u = float3(0, 0, 0);
    if (gMpmParams[0].area.z != 0u) u = mpmMergedFaceDisplacement(k, mpmFacePoint(slot, a, p, k), a, p, hf, f);
    gMpmCoupled[p.boundary.y + slot] = float4(u, p.column.x + dh);
}

// A grown area's face point that one of the areas it took over had too (the
// same face, on the same plane: a side that did not move, or the floor at the
// same bed) keeps that face's displacement; elsewhere the ground beyond starts
// at rest.
float3 mpmMergedFaceDisplacement(uint k, float3 r, MpmAbsorbGeom a, MpmParams p, HeightFieldRef hf, Frame f)
{
    uint mask = gMpmParams[0].area.z;
    uint ax = mpmFaceAxis(k);
    int3 g = int3(round(r)) + mpmAreaOffset(p);
    float plane = mpmFacePlane(k, a) + float(mpmAreaOffset(p)[ax]);
    for (uint o = 0u; o < MPM_MAX_AREAS; o++)
    {
        if ((mask & (1u << o)) == 0u) continue;
        MpmParams q = mpmArea(o);
        if (q.boundary.z == 0u) continue;
        MpmAbsorbGeom b = mpmAbsorbGeometry(hf, f, q);
        int3 qo = mpmAreaOffset(q);
        if (!(abs(mpmFacePlane(k, b) + float(qo[ax]) - plane) < 1e-3)) continue;
        uint rec = mpmFaceRecord(k, g - qo, q);
        if (rec == MPM_NO_SLOT || rec >= q.boundary.z) continue;
        return gMpmCoupled[q.boundary.y + rec].xyz;
    }
    return float3(0, 0, 0);
}

// Per face record, each substep after G2P: the face point moves with the
// material's velocity there, the mass-weighted B-spline mean of the node
// layers across the face, while the material presses on it. It stays where it
// was while the face lets the material go (its normal traction would pull, as
// mpmAbsorbNode decides).
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_absorb_faces(uint3 tid : SV_DispatchThreadID)
{
    // Face records of every open area: thread s is record s of the face pool.
    uint s = mpmIndex(tid);
    MpmParams pd = mpmParams();
    uint at = 4u * asuint(pd.couple.y) + s;
    uint area = MPM_NO_AREA;
    for (uint a = 0u; a < pd.area.x; a++)
    {
        uint4 bd = gMpmParams[1u + a].boundary;
        if (gMpmParams[1u + a].dims.x != 0u && at >= bd.y && at < bd.y + bd.z) area = a;
    }
    if (area == MPM_NO_AREA || mpmAreaSkips(area)) return;
    MpmParams p = mpmArea(area);
    uint slot = at - p.boundary.y;
    int3 off = mpmAreaOffset(p);
    if (slot >= min(p.boundary.z, mpmFaceCount(p))) return;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    MpmAbsorbGeom a = mpmAbsorbGeometry(hf, f, p);
    uint k;
    float3 r = mpmFacePoint(slot, a, p, k);
    uint ax = mpmFaceAxis(k);
    // Only where a node reaches the face: within 1.5 cells of the window's
    // span, and of a side's floor and top (the ground beside it).
    if (any(r.xz < a.lo.xz - 1.5) || any(r.xz > a.hi.xz + 1.5)) return;
    if (k != 0u && (r.y < a.lo.y - 1.5 || r.y > mpmSideTop(k, mpmWindowNearest(r, hf, f, p), hf, f, p) + 1.5))
        return;
    float plane = r[ax];
    int first = int(floor(plane - 1.5)) + 1;
    float3 mv = float3(0, 0, 0);
    float mw = 0.0;
    for (int l = 0; l < 3; l++)
    {
        int3 g = int3(round(r));
        g[ax] = first + l;
        float w = mpmSpline(plane - float(g[ax]));
        uint c = mpmCellOf(g + off, p);
        if (c == MPM_NO_BLOCK || !(w > 0.0)) continue;
        float4 node = gMpmGrid[2u * c];
        if (!(node.w > p.step.z)) continue;
        mv += w * node.w * node.xyz;
        mw += w * node.w;
    }
    if (!(mw > 0.0)) return;
    float3 v = mv / mw;
    float4 d = gMpmCoupled[p.boundary.y + slot];
    MpmGround gr = mpmGroundOf(p);
    float2 kk = mpmFaceSprings(r, gr, p);
    float dx = p.origin.w, dt = p.step.x;
    float top = (frameToWorld(f, float3(0.0, d.w, 0.0)).y - p.origin.y) / dx;
    float p0 = mpmFaceP0(k, top, r.y, a.lo.y, gr);
    float sg = mpmFaceSign(k);
    float3 uNew = d.xyz + dt * v;
    float tn = p0 - kk.x * sg * uNew[ax] - gr.rho * gr.cp * sg * v[ax];
    if (tn < 0.0) return;
    gMpmCoupled[p.boundary.y + slot] = float4(uNew, d.w);
}

// Once, in the first step after a geostatic activation (it needs gravity; the
// activation made every particle of the region, and none has moved yet): each
// particle's F from its depth d below the surface of the column it came from
// (its home sample, MpmParticle c1.w and c2.w), the elastic state of a layer
// under its own weight with no lateral strain. Hencky's τ_yy = (λ + 2μ)·ln F_yy
// = −ρ·g·d and τ_xx = τ_zz = λ·ln F_yy = K0·τ_yy.
[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_geostatic(uint3 tid : SV_DispatchThreadID)
{
    uint i = mpmIndex(tid);
    MpmParams p = mpmParams();
    if (i >= mpmParticleCount()) return;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    if (mpmGeostaticFresh(q, p)) gMpmParticles[i] = q;
}

// The geostatic start of particle q when it was activated since the last
// step (MPM_PARTICLE_FRESH, which it clears): its area's (the others' material
// is moving already). True when q changed.
bool mpmGeostaticFresh(inout MpmParticle q, MpmParams p)
{
    if ((asuint(q.f2.w) & MPM_PARTICLE_FRESH) == 0u) return false;
    q.f2.w = asfloat(asuint(q.f2.w) & ~MPM_PARTICLE_FRESH);
    uint area = mpmAreaOf(q.xm.xyz, p);
    if (area != MPM_NO_AREA) mpmGeostaticStart(q, mpmArea(area));
    return true;
}

// The geostatic F of activated particle q (see mpm_geostatic); false when it
// takes none. The plan applies it in the first substep's mpm_payload, just
// before the stress (MPM_C_GEOSTATIC): the same F, one pass over the
// particles less on the strike frame.
bool mpmGeostaticStart(inout MpmParticle q, MpmParams p)
{
    if ((asuint(q.f2.w) & MPM_PARTICLE_HOME) == 0u) return false;
    float modulus = p.sand.x + 2.0 * p.sand.y;
    if (!(modulus > 0.0) || !(p.step.y < 0.0)) return false;
    Collider col = gColliders[p.dims.w];
    Frame f = colliderFrame(col, gBodies[col.body]);
    HeightFieldRef hf = heightFieldOf(col);
    int si = clamp(int(round((q.c1.w - hf.x0) / hf.dx)), int(p.window.x), int(p.window.x + p.window.z) - 1);
    int sj = clamp(int(round((q.c2.w - hf.z0) / hf.dz)), int(p.window.y), int(p.window.y + p.window.w) - 1);
    uint t = uint(sj - int(p.window.y)) * p.window.z + uint(si - int(p.window.x));
    float top = p.column.x + float(ALOAD(gMpmColumns[mpmCol(p, t)])) * p.column.y / mpmSampleArea(hf);
    float depth = max(top - frameToLocal(f, q.xm.xyz).y, 0.0);
    q.f1.y = exp(p.step.y * p.column.w * depth / modulus);
    return true;
}

// A node over the unactivated ground beside the window, at or below that
// ground's surface: a separating Coulomb contact with the surface (as the rigid
// bed's hfSurface case). Nodes inside the window, or under the ground's top
// (the walls), are the absorbing faces' and are left alone.
void mpmGroundSurfaceContact(HeightFieldRef hf, Frame f, float3 xl, MpmParams p, inout float3 v)
{
    int si = clamp(int(round((xl.x - hf.x0) / hf.dx)), 0, int(hf.nx) - 1);
    int sj = clamp(int(round((xl.z - hf.z0) / hf.dz)), 0, int(hf.nz) - 1);
    int2 lo = int2(p.window.xy), hi = int2(p.window.xy + p.window.zw) - 1;
    bool outside = si < lo.x || si > hi.x || sj < lo.y || sj > hi.y;
    if (!outside || xl.y < hfHeight(hf, uint(si), uint(sj))) return;
    float h;
    float3 nl;
    if (!(hfSurface(hf, xl.x, xl.z, h, nl) && (xl.y - h) * nl.y <= 0.0)) return;
    float3 nw = quatRotate(f.q, nl);
    float vn = dot(v, nw);
    if (vn >= 0.0) return;
    float3 vt = v - nw * vn;
    float st = length(vt);
    float mu = p.step.w;
    v = st <= -mu * vn ? float3(0, 0, 0) : vt * (1.0 + mu * vn / st);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mpm_grid(uint3 tid : SV_DispatchThreadID)
{
    uint n = mpmIndex(tid);
    MpmParams p = mpmParams();
    uint b = n / MPM_BLOCK_NODES;
    if (b >= min(ALOAD(gMpmCounters[MPM_C_BLOCKS]), p.counts.z)) return;
    // mpm_payload has set the geostatic F: later substeps do not.
    if (n == 0u) ASTORE(gMpmCounters[MPM_C_GEOSTATIC], 0u);
    float4 node = gMpmGrid[2u * n];
    if (node.w <= p.step.z) { gMpmGrid[2u * n] = float4(0, 0, 0, node.w); return; }
    float3 v = node.xyz / node.w;
    v.y += p.step.x * p.step.y;
    // The node's area, in its own frame (node g from the area's origin).
    uint area = mpmNodeArea(mpmKeyNode(ALOAD(gMpmBlocks[4u * b]), n % MPM_BLOCK_NODES), p);
    if (area == MPM_NO_AREA)
    {
        gMpmGrid[2u * n] = float4(v, node.w);
        return;
    }
    int3 gd = mpmKeyNode(ALOAD(gMpmBlocks[4u * b]), n % MPM_BLOCK_NODES);
    // The wheels' loads (their own function, ahead of the bed, which carries
    // what the material passes on).
    mpmWheelLoads(gd, area, p, v);
    int3 g = gd - mpmSlotOffset(area);
    p = mpmArea(area);
    if (p.dims.w != MPM_NO_BLOCK)
    {
        float3 xw = p.origin.xyz + float3(g) * p.origin.w;
        Collider col = gColliders[p.dims.w];
        Frame f = colliderFrame(col, gBodies[col.body]);
        HeightFieldRef hf = heightFieldOf(col);
        // A node the absorbing faces reach (its own function, ahead of the rigid bed).
        if ((p.boundary.x & MPM_BOUNDARY_ABSORBING) != 0u && mpmAbsorbNode(g, node.w, hf, f, p, v))
        {
            // The absorbing faces are the floor and the walls below the
            // ground's top; the ground's own surface beside the window (where
            // the rim spills over) stays rigid ground.
            mpmGroundSurfaceContact(hf, f, frameToLocal(f, xw), p, v);
            mpmRigidContact(n, g, node.w, p, v);
            gMpmGrid[2u * n] = float4(v, node.w);
            return;
        }
        float3 xl = frameToLocal(f, xw);
        float3 nl;
        if (mpmBedNormal(hf, xl, p, nl))
        {
            float3 nw = quatRotate(f.q, nl);
            float vn = dot(v, nw);
            if (vn < 0.0)
            {
                float3 vt = v - nw * vn;
                float st = length(vt);
                float mu = p.step.w;
                v = st <= -mu * vn ? float3(0, 0, 0) : vt * (1.0 + mu * vn / st);
            }
        }
    }
    mpmRigidContact(n, g, node.w, p, v);
    gMpmGrid[2u * n] = float4(v, node.w);
}

// Block sleep's G2P (mpm_g2p behind its early return; the update is
// mpm_g2p's). An asleep particle outside the substep, or in the outer ring
// (whose nodes miss what lies beyond), stays as it is. One in the awake blocks
// or the first ring has every node of its stencil complete: it takes its
// update as an awake particle would (so stress waves pass through it), and
// wakes once the update moves it faster than the rest speed or varies its
// velocity across a cell by more. An awake particle after `sleep_after`
// substeps at rest in a row falls asleep: it stops drawing the rings round it.
void mpmG2pSleep(uint i, MpmParams p)
{
    uint flags = asuint(gMpmParticles[i].f2.w);
    if ((flags & MPM_PARTICLE_DEAD) != 0u) return;
    bool asleep = (flags & MPM_PARTICLE_ASLEEP) != 0u;
    if (asleep)
    {
        uint c = gMpmSorted[p.counts.x + i];
        if (c == MPM_NO_BLOCK || c / MPM_BLOCK_NODES >= ALOAD(gMpmCounters[MPM_C_SLEEP_RING1])) return;
    }
    MpmParticle q = gMpmParticles[i];
    uint area = mpmAreaOf(q.xm.xyz, p);
    if (area == MPM_NO_AREA || mpmAreaSkips(area)) return;
    float dx = p.origin.w;
    float dt = p.step.x;
    float3 ao = gMpmParams[1u + area].origin.xyz;
    int3 aoff = mpmSlotOffset(area);
    int3 local = int3(floor((q.xm.xyz - ao) / dx - 0.5));
    float3 fx = (q.xm.xyz - ao) / dx - float3(local);
    int3 base = local + aoff;
    float3 wx = mpmWeights(fx.x), wy = mpmWeights(fx.y), wz = mpmWeights(fx.z);
    float3 v = float3(0, 0, 0);
    float3x3 B = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    int3 k0 = base >> 2;
    uint blk[8];
    mpmBlockCube(k0, p, blk);
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        uint c = mpmCellInCube(base + int3(off), k0, blk);
        if (c == MPM_NO_BLOCK) continue;
        float w = mpmWeightAt(wx, off.x) * mpmWeightAt(wy, off.y) * mpmWeightAt(wz, off.z);
        float3 vi = gMpmGrid[2u * c].xyz;
        float3 dpos = (float3(off) - fx) * dx;
        v += w * vi;
        B += float3x3(vi * dpos.x, vi * dpos.y, vi * dpos.z) * w;
    }
    float3x3 C = transpose(B) * (4.0 / (dx * dx));
    float speed = length(v);
    // The velocity's spread across a cell: ‖C‖·dx (Frobenius).
    float spread = sqrt(dot(C[0], C[0]) + dot(C[1], C[1]) + dot(C[2], C[2])) * dx;
    float rest = mpmSleepSpeed();
    bool still = speed <= rest && spread <= rest;
    q.xm.xyz += dt * v;
    q.vj.xyz = v;
    q.c0 = float4(C[0], q.c0.w);
    q.c1 = float4(C[1], q.c1.w);
    q.c2 = float4(C[2], q.c2.w);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    float3x3 I = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    F = mul(I + C * dt, F);
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    // Substeps at rest in a row; enough of them and it sleeps.
    uint quiet = still ? min(((flags & MPM_PARTICLE_QUIET) >> 8) + 1u, 255u) : 0u;
    flags = (flags & ~(MPM_PARTICLE_ASLEEP | MPM_PARTICLE_QUIET)) | (quiet << 8);
    if (quiet >= ALOAD(gMpmCounters[MPM_C_SLEEP_AFTER])) flags |= MPM_PARTICLE_ASLEEP;
    q.f2.w = asfloat(flags);
    AMAX2(gMpmCounters[MPM_C_MAX_SPEED], asuint(speed));
    if (p.area.x > 1u) AMAX2(gMpmCounters[MPM_C_AREA_SPEED + area], asuint(speed));
    if (!(speed <= p.limits.x)) AOR2(gMpmCounters[MPM_C_FLAGS], MPM_CFL);
    uint maxParcels = asuint(p.couple.z);
    if (!mpmInsideArea(q.xm.xyz, area) && maxParcels > 0u)
    {
        uint k;
        AADD3(gMpmCounters[MPM_C_PARCELS], 1u, k);
        if (k < maxParcels)
        {
            gMpmParcels[2u * k] = float4(q.xm.xyz, q.f1.w);
            gMpmParcels[2u * k + 1u] = float4(v, q.xm.w);
            gMpmParcels[2u * maxParcels + k] = float4(asfloat(i), 0, 0, 0);
            AADD2(gMpmCounters[MPM_C_AIRBORNE], 1u);
            mpmParcelReach(q.xm.xyz, v, p);
            AADD2(gMpmCounters[MPM_C_LIVE], 0xFFFFFFFFu);
            q.f2.w = asfloat(asuint(q.f2.w) | MPM_PARTICLE_DEAD);
        }
        else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARCELS);
    }
    gMpmParticles[i] = q;
}

// G2P's fastest speeds, reduced in its workgroup before the counters' atomics:
// each lane's speed (f32 bits, 0 for none) and its area, kept; and the tree
// over them: the speeds' maximum and the lowest and highest area that moved.
groupshared uint gsMpmG2pSpeed[64];
groupshared uint gsMpmG2pArea[64];
groupshared uint gsMpmG2pTop[64];
groupshared uint gsMpmG2pLo[64];
groupshared uint gsMpmG2pHi[64];

// G2P of particle i; its speed (f32 bits) and area for the frame's maxima, or
// area MPM_NO_AREA when it moved no material.
void mpmG2p(uint i, MpmParams p, out uint speedBits, out uint speedArea)
{
    speedBits = 0u;
    speedArea = MPM_NO_AREA;
    MpmParticle q = gMpmParticles[i];
    if (mpmParticleDead(q)) return;
    uint area = mpmAreaOf(q.xm.xyz, p);
    if (area == MPM_NO_AREA) return;
    // Area clocks: an area sitting out this pass keeps its state.
    if (mpmAreaSkips(area)) return;
    float dx = p.origin.w;
    float dt = p.step.x;
    // In the area's frame (exactly as the area alone), on the domain's lattice.
    float3 ao = gMpmParams[1u + area].origin.xyz;
    int3 aoff = mpmSlotOffset(area);
    int3 local = int3(floor((q.xm.xyz - ao) / dx - 0.5));
    float3 fx = (q.xm.xyz - ao) / dx - float3(local);
    int3 base = local + aoff;
    float3 wx = mpmWeights(fx.x), wy = mpmWeights(fx.y), wz = mpmWeights(fx.z);
    float3 v = float3(0, 0, 0);
    float3x3 B = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    // Nodes base .. base + 2 lie in the blocks from base >> 2.
    int3 k0 = base >> 2;
    uint blk[8];
    mpmBlockCube(k0, p, blk);
    for (uint o = 0u; o < 27u; o++)
    {
        uint3 off = uint3(o % 3u, (o / 3u) % 3u, o / 9u);
        uint c = mpmCellInCube(base + int3(off), k0, blk);
        if (c == MPM_NO_BLOCK) continue;
        float w = mpmWeightAt(wx, off.x) * mpmWeightAt(wy, off.y) * mpmWeightAt(wz, off.z);
        float3 vi = gMpmGrid[2u * c].xyz;
        float3 dpos = (float3(off) - fx) * dx;
        v += w * vi;
        B += float3x3(vi * dpos.x, vi * dpos.y, vi * dpos.z) * w;
    }
    // C = (4/dx²)·Σ w vᵢ dposᵀ: B above holds its columns (vᵢ·dpos_k) as rows.
    float3x3 C = transpose(B) * (4.0 / (dx * dx));
    q.xm.xyz += dt * v;
    q.vj.xyz = v;
    q.c0 = float4(C[0], q.c0.w); // c0.w: rock damage, carried
    q.c1 = float4(C[1], q.c1.w); // c1.w, c2.w: where it was activated, carried
    q.c2 = float4(C[2], q.c2.w);
    float3x3 F = float3x3(q.f0.xyz, q.f1.xyz, q.f2.xyz);
    float3x3 I = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    F = mul(I + C * dt, F);
    q.f0.xyz = F[0];
    q.f1.xyz = F[1];
    q.f2.xyz = F[2];
    float speed = length(v);
    speedBits = asuint(speed);
    speedArea = area;
    if (!(speed <= p.limits.x)) AOR2(gMpmCounters[MPM_C_FLAGS], MPM_CFL);
    // Leaving its area's box: a parcel carries the material on (with ejecta enabled).
    uint maxParcels = asuint(p.couple.z);
    if (!mpmInsideArea(q.xm.xyz, area) && maxParcels > 0u)
    {
        uint k;
        AADD3(gMpmCounters[MPM_C_PARCELS], 1u, k);
        if (k < maxParcels)
        {
            gMpmParcels[2u * k] = float4(q.xm.xyz, q.f1.w);
            gMpmParcels[2u * k + 1u] = float4(v, q.xm.w);
            gMpmParcels[2u * maxParcels + k] = float4(asfloat(i), 0, 0, 0);
            // In flight from now (mpm_parcels flies it from the next substep).
            AADD2(gMpmCounters[MPM_C_AIRBORNE], 1u);
            mpmParcelReach(q.xm.xyz, v, p);
            AADD2(gMpmCounters[MPM_C_LIVE], 0xFFFFFFFFu);
            q.f2.w = asfloat(asuint(q.f2.w) | MPM_PARTICLE_DEAD);
        }
        else AOR2(gMpmCounters[MPM_C_FLAGS], MPM_OVERFLOW_PARCELS);
    }
    gMpmParticles[i] = q;
}

