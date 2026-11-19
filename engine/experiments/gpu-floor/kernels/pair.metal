        // ===== The contact group of a pair of bodies (mechanism 2) =====
        // Bodies 0 and 1 each step on their own workgroup (their own core); this workgroup
        // runs the pair's contact pass (explicit_contact.rs contact_pass, triangle_loop)
        // between the two cross-group barriers of a substep. Both bodies' node positions
        // and velocities are copied into this group's threadgroup memory (body 0 where a
        // body keeps its own, body 1 in the scalars' array), so the search runs on
        // threadgroup memory; device memory is touched for the copy, and for hits.
        // Detection: a triangle's padded box against the partner's box, then against the
        // partner's clusters of contact points, then node by node; each stage's survivors
        // on a compact list, a lane an item. The owners' turns run one after the other
        // (owner 0's hits are applied before owner 1's residual and guard read the forces),
        // each turn order-free inside.
        #define CB(c) (big + (car - 2u + (c)) * CAR_WORDS)
        #define XNODE(c) ((device float4*)(CB(c) + O_NODE))
        #define XACC(c) ((device atomic_int*)(CB(c) + O_ACC))
        #define XSHIFT(c) ((device atomic_int*)(CB(c) + O_SHIFT))
        #define XCLIP(c) ((device atomic_uint*)(CB(c) + O_CLIP))
        #define XFLAG(c) ((device atomic_uint*)(CB(c) + O_MISC + 12))
        #define XDEAD(c) (CB(c) + O_TRIDEAD)
        #define XSTAGE(c) ((device atomic_int*)(cb + O_SCRATCH + (c) * NN * 6))
        #define XSPLIT(turn, c) ((device atomic_uint*)(cb + O_HOT + ((turn) * 2u + (c)) * NN))
        #define XCLU ((device float4*)(cb + O_HOT + 4096))
        #define XTRI (cb + O_CAND + 2048)
        #define HX 8
        #define HCAP 1024
        #define XP1(i) v3(tgG[(i) * 6u], tgG[(i) * 6u + 1u], tgG[(i) * 6u + 2u])
        #define XV1(i) v3(tgG[(i) * 6u + 3u], tgG[(i) * 6u + 4u], tgG[(i) * 6u + 5u])
        #define XP(c, i) (v3(0, 0, 0) + ((c) == 0u ? TGP(i) : XP1(i)))
        #define XV(c, i) ((c) == 0u ? TGV(i) : XV1(i))
        #define XM(i) nodeStatic[i].x
        #define XIMM(i) ((as_type<uint>(nodeStatic[i].z) & NODE_IMMOVABLE) != 0u)
        #define XACCEL(f, i) (XIMM(i) ? v3(0, 0, 0) : (f) / XM(i))
        #define XIM(i, kf) (XIMM(i) ? 0.0f : (kf) / XM(i))
        #define ILOAD(p) atomic_load_explicit(p, memory_order_relaxed)
        #define XNA(c) ((device atomic_uint*)(CB(c) + O_NODE))
        #define XLD3(c, w) v3(as_type<float>(ILOAD(&XNA(c)[(w) * 4u])), as_type<float>(ILOAD(&XNA(c)[(w) * 4u + 1u])), as_type<float>(ILOAD(&XNA(c)[(w) * 4u + 2u])))
        #if XATOM >= 1
        #define XF(c, i) XLD3(c, (i) * 3u + 2u)
        #define XISDEAD(c, t) (ILOAD((device atomic_uint*)&XDEAD(c)[t]) != 0u)
        #else
        #define XF(c, i) v3(XNODE(c)[(i) * 3 + 2])
        #define XISDEAD(c, t) (XDEAD(c)[t] != 0u)
        #endif
        for (uint frame = 0; frame < P.frames; frame++)
        for (uint step = 0; step < Z_STEPS; step++)
        {
            XBAR; // the bodies have stepped and published their nodes
            if (lid < 16u) atomic_store_explicit(&tgX[lid], lid >= 12u ? 0 : ((lid % 6u) < 3u ? 0x7FFFFFFF : int(0x80000000)), memory_order_relaxed);
            if (lid < 4u) atomic_store_explicit(&tgCount[lid], 0u, memory_order_relaxed);
            BAR;
            // X0: both bodies' nodes into threadgroup memory; their boxes and top speeds.
            for (uint k = lid; k < 2u * NN; k += LANES)
            {
                const uint c = k < NN ? 0u : 1u, i = k - c * NN;
        #if XATOM >= 1
                float4 p4 = f4(XLD3(c, i * 3u), 0), v4 = f4(XLD3(c, i * 3u + 1u), 0);
        #else
                float4 p4 = XNODE(c)[i * 3], v4 = XNODE(c)[i * 3 + 1];
        #endif
                if (c == 0u) { SETP(i, v3(p4)); SETV(i, v3(v4)); }
                else
                {
                    tgG[i * 6u] = p4.x; tgG[i * 6u + 1u] = p4.y; tgG[i * 6u + 2u] = p4.z;
                    tgG[i * 6u + 3u] = v4.x; tgG[i * 6u + 4u] = v4.y; tgG[i * 6u + 5u] = v4.z;
                }
                V3 w = v3(0, 0, 0) + v3(p4);
                atomic_fetch_min_explicit(&tgX[c * 6u], ordF(w.x), memory_order_relaxed);
                atomic_fetch_min_explicit(&tgX[c * 6u + 1u], ordF(w.y), memory_order_relaxed);
                atomic_fetch_min_explicit(&tgX[c * 6u + 2u], ordF(w.z), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[c * 6u + 3u], ordF(w.x), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[c * 6u + 4u], ordF(w.y), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[c * 6u + 5u], ordF(w.z), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[12u + c], as_type<int>(len3(v3(v4))), memory_order_relaxed);
            }
            BAR;
            V3 blo[2], bhi[2];
            float vtop[2];
            for (uint c = 0; c < 2u; c++)
            {
                blo[c] = v3(unordF(ILOAD(&tgX[c * 6u])), unordF(ILOAD(&tgX[c * 6u + 1u])), unordF(ILOAD(&tgX[c * 6u + 2u])));
                bhi[c] = v3(unordF(ILOAD(&tgX[c * 6u + 3u])), unordF(ILOAD(&tgX[c * 6u + 4u])), unordF(ILOAD(&tgX[c * 6u + 5u])));
                vtop[c] = as_type<float>(ILOAD(&tgX[12u + c]));
            }
            // (The bodies' boxes with BOX_MARGIN apart: no pass.)
            const bool apart = (bhi[0].x + 0.05f) < (blo[1].x - 0.05f) || (bhi[0].y + 0.05f) < (blo[1].y - 0.05f) || (bhi[0].z + 0.05f) < (blo[1].z - 0.05f) ||
                               (bhi[1].x + 0.05f) < (blo[0].x - 0.05f) || (bhi[1].y + 0.05f) < (blo[0].y - 0.05f) || (bhi[1].z + 0.05f) < (blo[0].z - 0.05f);
            if (!apart)
            {
                // X1: each body's clusters of contact points (box, for the cull); each
                // triangle's padded box against the partner's box.
                for (uint k = lid; k < 2u * NCLP; k += LANES)
                {
                    const uint c = k < NCLP ? 0u : 1u, q = k - c * NCLP;
                    V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
                    device const uint4* cl4 = (device const uint4*)(tab + T_PCL + q * 32u);
                    for (uint jj = 0; jj < 8u; jj++)
                    {
                        uint4 w4 = cl4[jj];
                        for (uint e = 0; e < 4u; e++)
                        {
                            uint h = w4[e];
                            if (h == NONE) continue;
                            V3 x = XP(c, h);
                            lo = v3(rmin(lo.x, x.x), rmin(lo.y, x.y), rmin(lo.z, x.z));
                            hi = v3(rmax(hi.x, x.x), rmax(hi.y, x.y), rmax(hi.z, x.z));
                        }
                    }
                    XCLU[k * 2u] = f4(lo, 0);
                    XCLU[k * 2u + 1u] = f4(hi, 0);
                }
                for (uint k = lid; k < 2u * NT; k += LANES)
                {
                    const uint a = k < NT ? 0u : 1u, t = k - a * NT, b = 1u - a;
                    if (XISDEAD(a, t)) continue;
                    uint4 tn = tris[t];
                    V3 p0 = XP(a, tn.x), p1 = XP(a, tn.y), p2 = XP(a, tn.z);
                    float travel = rmax(rmax(rmax(0.0f, len3(XV(a, tn.x))), len3(XV(a, tn.y))), len3(XV(a, tn.z))) * H;
                    // (No contact point's pad is more than the partner's fastest node's.)
                    float pad = CR + ((travel + vtop[b] * H) + CLIP_DEPTH);
                    V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - pad, rmin(rmin(p0.y, p1.y), p2.y) - pad, rmin(rmin(p0.z, p1.z), p2.z) - pad);
                    V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + pad, rmax(rmax(p0.y, p1.y), p2.y) + pad, rmax(rmax(p0.z, p1.z), p2.z) + pad);
                    if (bhi[b].x < lo.x || bhi[b].y < lo.y || bhi[b].z < lo.z || blo[b].x > hi.x || blo[b].y > hi.y || blo[b].z > hi.z) continue;
                    uint ci = atomic_fetch_add_explicit(&tgCount[1], 1u, memory_order_relaxed);
                    if (ci < 2u * NT) XTRI[ci] = a << 16 | t;
                }
                BAR;
                // X2: a lane a surviving triangle: the partner's clusters, then their nodes,
                // by the reference's padded box test.
                const uint ntri = min(ILOAD(&tgCount[1]), 2u * NT);
                for (uint k = lid; k < ntri; k += LANES)
                {
                    const uint w = XTRI[k];
                    const uint a = w >> 16, t = w & 0xFFFFu, b = 1u - a;
                    uint4 tn = tris[t];
                    V3 p0 = XP(a, tn.x), p1 = XP(a, tn.y), p2 = XP(a, tn.z);
                    float travel = rmax(rmax(rmax(0.0f, len3(XV(a, tn.x))), len3(XV(a, tn.y))), len3(XV(a, tn.z))) * H;
                    V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - CR, rmin(rmin(p0.y, p1.y), p2.y) - CR, rmin(rmin(p0.z, p1.z), p2.z) - CR);
                    V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + CR, rmax(rmax(p0.y, p1.y), p2.y) + CR, rmax(rmax(p0.z, p1.z), p2.z) + CR);
                    float padTop = (travel + vtop[b] * H) + CLIP_DEPTH;
                    for (uint q = 0; q < NCLP; q++)
                    {
                        float4 clo = XCLU[(b * NCLP + q) * 2u], chi = XCLU[(b * NCLP + q) * 2u + 1u];
                        if (chi.x < lo.x - padTop || chi.y < lo.y - padTop || chi.z < lo.z - padTop || clo.x > hi.x + padTop || clo.y > hi.y + padTop || clo.z > hi.z + padTop) continue;
                        device const uint4* cl4 = (device const uint4*)(tab + T_PCL + q * 32u);
                        for (uint jj = 0; jj < 8u; jj++)
                        {
                            uint4 w4 = cl4[jj];
                            for (uint e = 0; e < 4u; e++)
                            {
                                uint h = w4[e];
                                if (h == NONE) continue;
                                V3 x = XP(b, h);
                                float pad = (travel + len3(XV(b, h)) * H) + CLIP_DEPTH;
                                if (!(x.x >= lo.x - pad && x.y >= lo.y - pad && x.z >= lo.z - pad && x.x <= hi.x + pad && x.y <= hi.y + pad && x.z <= hi.z + pad)) continue;
                                uint ci = atomic_fetch_add_explicit(&tgCount[3], 1u, memory_order_relaxed);
                                if (ci < CAND_CAP) candBuf[ci] = a << 28 | t << 16 | h;
                            }
                        }
                    }
                }
                BAR;
                // X3: a lane a candidate: the contact rule (frame, held side, swept crossing,
                // the side vote); a contact's side recorded, its hit kept.
                const uint ncx = min(ILOAD(&tgCount[3]), (uint)CAND_CAP);
                if (lid == 0 && ILOAD(&tgCount[3]) > CAND_CAP) misc[1] = misc[1] | 4u;
                for (uint k = lid; k < ncx; k += LANES)
                {
                    const uint cw = candBuf[k];
                    const uint a = cw >> 28, t = (cw >> 16) & 0xFFFu, h = cw & 0xFFFFu, b = 1u - a;
                    uint4 tn = tris[t];
                    V3 p0 = XP(a, tn.x), p1 = XP(a, tn.y), p2 = XP(a, tn.z);
                    V3 x = XP(b, h);
                    // ContactFrame::new and local.
                    V3 u = p1 - p0, w = p2 - p0;
                    V3 c = cross3(u, w);
                    float len = sqrt(((c.x * c.x) + (c.y * c.y)) + (c.z * c.z));
                    V3 n = len > 1e-8f ? c * (1.0f / len) : c;
                    float det = dot3(u, cross3(w, n));
                    bool ok = fabs(det) > 1e-6f;
                    V3 ra = v3(0, 0, 0), rb = ra, rd = ra;
                    if (ok) { float r = 1.0f / det; ra = cross3(w, n) * r; rb = cross3(n, u) * r; rd = cross3(u, w) * r; }
                    float4 loc = frameLocal(p0, ra, rb, rd, ok, x);
                    const uint key = a << 20 | t;
                    float side = 0.0f;
                    uint c0 = ILOAD(&XCLIP(b)[h * 4]), c1 = ILOAD(&XCLIP(b)[h * 4 + 1]);
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
                        // The swept crossing: the frame a step ago, the point a step ago.
                        V3 q0 = p0 - XV(a, tn.x) * H, q1 = p1 - XV(a, tn.y) * H, q2 = p2 - XV(a, tn.z) * H;
                        V3 ub = q1 - q0, wb = q2 - q0;
                        V3 cb2 = cross3(ub, wb);
                        float lenb = sqrt(((cb2.x * cb2.x) + (cb2.y * cb2.y)) + (cb2.z * cb2.z));
                        V3 nb = lenb > 1e-8f ? cb2 * (1.0f / lenb) : cb2;
                        float detb = dot3(ub, cross3(wb, nb));
                        bool okb = fabs(detb) > 1e-6f;
                        V3 rab = v3(0, 0, 0), rbb = rab, rdb = rab;
                        if (okb) { float r = 1.0f / detb; rab = cross3(wb, nb) * r; rbb = cross3(nb, ub) * r; rdb = cross3(ub, wb) * r; }
                        float4 l0 = frameLocal(q0, rab, rbb, rdb, okb, x - XV(b, h) * H);
                        float e0 = l0.w, e1 = loc.w;
                        bool crossed = false;
                        if ((e0 > 0.0f && e1 < 0.0f) || (e0 < 0.0f && e1 > 0.0f))
                        {
                            float sx = e0 / (e0 - e1);
                            float aa = l0.x + (loc.x - l0.x) * sx;
                            float bb2 = l0.y + (loc.y - l0.y) * sx;
                            float3 cwt = float3(aa, bb2, (1.0f - aa) - bb2);
                            if (cwt.x >= 0.0f && cwt.y >= 0.0f && cwt.z >= 0.0f)
                            {
                                crossed = true;
                                side = e0 > 0.0f ? 1.0f : -1.0f;
                                haveW = true;
                                wgt = cwt;
                            }
                        }
                        if (!crossed)
                        {
                            if (!(over && fabs(loc.w) <= CR)) continue;
                            // The side vote (spec §5.3): the point's neighbours' sides.
                            int score = loc.w >= 0.0f ? 3 : -3;
                            const uint ns = tab[T_NB + h], ne = tab[T_NB + h + 1u];
                            if (ne - ns > 3u)
                            {
                                for (uint j = ns; j < ne && j < ns + 64u; j++)
                                {
                                    V3 e = XP(b, tab[T_NBL + j]) - p0;
                                    score += (((n.x * e.x) + (n.y * e.y)) + (n.z * e.z)) >= 0.0f ? 1 : -1;
                                }
                            }
                            side = score < 0 ? -1.0f : 1.0f;
                            haveW = true;
                            wgt = float3(loc.x, loc.y, loc.z);
                        }
                    }
                    float dd = side * loc.w;
                    if (!(haveW && dd <= CR && dd >= -CLIP_DEPTH)) continue;
                    uint word = a << 21 | t << 1 | (side > 0.0f ? 1u : 0u);
                    uint old = atomic_fetch_min_explicit(&XCLIP(b)[h * 4 + 2], word, memory_order_relaxed);
                    atomic_fetch_min_explicit(&XCLIP(b)[h * 4 + 3], max(old, word), memory_order_relaxed);
                    atomic_fetch_or_explicit(XFLAG(b), 4u, memory_order_relaxed);
                    float pen = rmin(CR - dd, CR);
                    uint hi_ = atomic_fetch_add_explicit(&tgCount[a == 0u ? 2u : 0u], 1u, memory_order_relaxed);
                    if (hi_ < HCAP)
                    {
                        device float4* hr = hitBuf + (a * HCAP + hi_) * HX;
                        hr[0] = as_type<float4>(uint4(h, tn.x, tn.y, tn.z));
                        hr[1] = float4(wgt.x, wgt.y, wgt.z, pen);
                        hr[2] = f4(n * side, 0);
                        // The forces at the pass's start (the contact sums before any turn).
                        uint ids[4] = {h, tn.y, tn.z, tn.x};
                        for (uint e = 0; e < 4u; e++)
                        {
                            const uint cc = e == 0u ? b : a;
                            V3 f = XF(cc, ids[e]) + v3(float(ILOAD(&XACC(cc)[ids[e] * 3])), float(ILOAD(&XACC(cc)[ids[e] * 3 + 1])), float(ILOAD(&XACC(cc)[ids[e] * 3 + 2]))) * (1.0f / 32.0f);
                            hr[4 + e] = f4(f, 0);
                        }
                    }
                    atomic_fetch_add_explicit(&XSPLIT(a, b)[h], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&XSPLIT(a, a)[tn.x], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&XSPLIT(a, a)[tn.y], 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&XSPLIT(a, a)[tn.z], 1u, memory_order_relaxed);
                }
                BAR;
                const uint nh0 = min(ILOAD(&tgCount[2]), (uint)HCAP), nh1 = min(ILOAD(&tgCount[0]), (uint)HCAP);
                if (lid == 0 && (ILOAD(&tgCount[2]) > HCAP || ILOAD(&tgCount[0]) > HCAP)) misc[1] = misc[1] | 2u;
                // The owners' turns.
                for (uint a = 0; a < 2u; a++)
                {
                    const uint b = 1u - a, nh = a == 0u ? nh0 : nh1;
                    if (nh == 0u) continue;
                    #define HREC(k) (hitBuf + (a * HCAP + (k)) * HX)
                    #define XNOW(c, i, extra) (XF(c, i) + v3(float(ILOAD(&XACC(c)[(i) * 3]) + (extra)[0]), float(ILOAD(&XACC(c)[(i) * 3 + 1]) + (extra)[1]), float(ILOAD(&XACC(c)[(i) * 3 + 2]) + (extra)[2])) * (1.0f / 32.0f))
                    // The first pass, from the forces at the pass's start.
                    for (uint k = lid; k < nh; k += LANES)
                    {
                        device float4* hr = HREC(k);
                        uint4 id = as_type<uint4>(hr[0]);
                        float4 wp = hr[1];
                        V3 n = v3(hr[2]);
                        const uint h = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                        const float alpha = wp.x, beta = wp.y, gamma = wp.z, depth = wp.w;
                        V3 v = XV(b, h) - ((XV(a, i1) * alpha + XV(a, i2) * beta) + XV(a, i0) * gamma);
                        const float imh = XIM(h, float(ILOAD(&XSPLIT(a, b)[h])));
                        const float im1 = XIM(i1, float(ILOAD(&XSPLIT(a, a)[i1]))), im2 = XIM(i2, float(ILOAD(&XSPLIT(a, a)[i2]))), im0 = XIM(i0, float(ILOAD(&XSPLIT(a, a)[i0])));
                        float inv = imh + tbInvMass(im1, im2, im0, alpha, beta, gamma);
                        float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                        V3 at = (XACCEL(v3(hr[5]), i1) * alpha + XACCEL(v3(hr[6]), i2) * beta) + XACCEL(v3(hr[7]), i0) * gamma;
                        V3 fh = (XACCEL(v3(hr[4]), h) - at) * m;
                        float budget;
                        V3 f = contactResponse(fh, v, m, n, BOXS, depth, nodeStatic[h].y, budget);
                        if (!isZero3(f))
                        {
                            float delta = shiftDepth(depth, budget, dot3(fh, n));
                            if (delta > 0.0f)
                            {
                                float q = delta * m;
                                V3 dh = n * (q * imh);
                                atomic_fetch_add_explicit(&XSHIFT(b)[h * 3], toFixedShift(dh.x), memory_order_relaxed);
                                atomic_fetch_add_explicit(&XSHIFT(b)[h * 3 + 1], toFixedShift(dh.y), memory_order_relaxed);
                                atomic_fetch_add_explicit(&XSHIFT(b)[h * 3 + 2], toFixedShift(dh.z), memory_order_relaxed);
                                atomic_fetch_or_explicit(XFLAG(b), 2u, memory_order_relaxed);
                                uint ti[3] = {i1, i2, i0};
                                float tw[3] = {alpha, beta, gamma};
                                float tm[3] = {im1, im2, im0};
                                for (uint e = 0; e < 3u; e++)
                                {
                                    if (tw[e] > 0.0f)
                                    {
                                        V3 dt = n * -(tw[e] * (q * tm[e]));
                                        atomic_fetch_add_explicit(&XSHIFT(a)[ti[e] * 3], toFixedShift(dt.x), memory_order_relaxed);
                                        atomic_fetch_add_explicit(&XSHIFT(a)[ti[e] * 3 + 1], toFixedShift(dt.y), memory_order_relaxed);
                                        atomic_fetch_add_explicit(&XSHIFT(a)[ti[e] * 3 + 2], toFixedShift(dt.z), memory_order_relaxed);
                                        atomic_fetch_or_explicit(XFLAG(a), 2u, memory_order_relaxed);
                                    }
                                }
                            }
                        }
                        hr[3] = f4(f, 0);
                        V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                        uint ns[4] = {h, i1, i2, i0};
                        for (uint e = 0; e < 4u; e++)
                        {
                            device atomic_int* st = XSTAGE(e == 0u ? b : a) + ns[e] * 6;
                            atomic_fetch_add_explicit(&st[0], toFixed(fs[e].x), memory_order_relaxed);
                            atomic_fetch_add_explicit(&st[1], toFixed(fs[e].y), memory_order_relaxed);
                            atomic_fetch_add_explicit(&st[2], toFixed(fs[e].z), memory_order_relaxed);
                        }
                    }
                    BAR;
                    // The residual pass, from the forces now (with the first pass's sums).
                    for (uint k = lid; k < nh; k += LANES)
                    {
                        device float4* hr = HREC(k);
                        uint4 id = as_type<uint4>(hr[0]);
                        float4 wp = hr[1];
                        V3 n = v3(hr[2]);
                        const uint h = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                        const float alpha = wp.x, beta = wp.y, gamma = wp.z;
                        #define S1(c, i) int3(ILOAD(&XSTAGE(c)[(i) * 6]), ILOAD(&XSTAGE(c)[(i) * 6 + 1]), ILOAD(&XSTAGE(c)[(i) * 6 + 2]))
                        V3 v = XV(b, h) - ((XV(a, i1) * alpha + XV(a, i2) * beta) + XV(a, i0) * gamma);
                        V3 at = (XACCEL(XNOW(a, i1, S1(a, i1)), i1) * alpha + XACCEL(XNOW(a, i2, S1(a, i2)), i2) * beta) + XACCEL(XNOW(a, i0, S1(a, i0)), i0) * gamma;
                        float an = dot3(XACCEL(XNOW(b, h, S1(b, h)), h) - at, n);
                        float vn = dot3(v, n);
                        float r = 0.0f;
                        if (vn < 0.0f && an * H + vn < 0.0f)
                        {
                            float inv = XIM(h, float(ILOAD(&XSPLIT(a, b)[h]))) + tbInvMass(XIM(i1, float(ILOAD(&XSPLIT(a, a)[i1]))), XIM(i2, float(ILOAD(&XSPLIT(a, a)[i2]))), XIM(i0, float(ILOAD(&XSPLIT(a, a)[i0]))), alpha, beta, gamma);
                            if (inv > 0.0f)
                            {
                                float m = 1.0f / inv;
                                float rr = -(an * m) - (vn * m) / H;
                                if (rr > 0.0f) r = rr;
                            }
                        }
                        float4 h3 = hr[3];
                        hr[3] = float4(h3.x, h3.y, h3.z, r);
                        if (r > 0.0f)
                        {
                            V3 f = n * r;
                            V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                            uint ns[4] = {h, i1, i2, i0};
                            for (uint e = 0; e < 4u; e++)
                            {
                                device atomic_int* st = XSTAGE(e == 0u ? b : a) + ns[e] * 6;
                                atomic_fetch_add_explicit(&st[3], toFixed(fs[e].x), memory_order_relaxed);
                                atomic_fetch_add_explicit(&st[4], toFixed(fs[e].y), memory_order_relaxed);
                                atomic_fetch_add_explicit(&st[5], toFixed(fs[e].z), memory_order_relaxed);
                            }
                        }
                    }
                    BAR;
                    // The work guard, from the forces now (with both passes' sums); each
                    // hit's total into the bodies' contact sums.
                    for (uint k = lid; k < nh; k += LANES)
                    {
                        device float4* hr = HREC(k);
                        uint4 id = as_type<uint4>(hr[0]);
                        float4 wp = hr[1];
                        V3 n = v3(hr[2]);
                        float4 h3 = hr[3];
                        const uint h = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                        const float alpha = wp.x, beta = wp.y, gamma = wp.z;
                        #define S2(c, i) int3(ILOAD(&XSTAGE(c)[(i) * 6]) + ILOAD(&XSTAGE(c)[(i) * 6 + 3]), ILOAD(&XSTAGE(c)[(i) * 6 + 1]) + ILOAD(&XSTAGE(c)[(i) * 6 + 4]), ILOAD(&XSTAGE(c)[(i) * 6 + 2]) + ILOAD(&XSTAGE(c)[(i) * 6 + 5]))
                        V3 t = v3(h3);
                        if (h3.w > 0.0f) t = t + n * h3.w;
                        V3 f = t;
                        if (!isZero3(t))
                        {
                            V3 v = XV(b, h) - ((XV(a, i1) * alpha + XV(a, i2) * beta) + XV(a, i0) * gamma);
                            V3 ac = XACCEL(XNOW(b, h, S2(b, h)), h) - ((XACCEL(XNOW(a, i1, S2(a, i1)), i1) * alpha + XACCEL(XNOW(a, i2, S2(a, i2)), i2) * beta) + XACCEL(XNOW(a, i0, S2(a, i0)), i0) * gamma);
                            float inv = XIM(h, 1.0f) + tbInvMass(XIM(i1, 1.0f), XIM(i2, 1.0f), XIM(i0, 1.0f), alpha, beta, gamma);
                            V3 aRest = ac - t * inv;
                            float aa = dot3(t, (v + v) + aRest * H);
                            float bb = (H * dot3(t, t)) * inv;
                            if (aa + bb > 0.0f && bb > 0.0f)
                            {
                                float sc = aa < 0.0f ? rmin(-aa / bb, 1.0f) : 0.0f;
                                f = t + t * -(1.0f - sc);
                            }
                        }
                        hr[3] = f4(f, 0);
                    }
                    BAR;
                    for (uint k = lid; k < nh; k += LANES)
                    {
                        device float4* hr = HREC(k);
                        uint4 id = as_type<uint4>(hr[0]);
                        float4 wp = hr[1];
                        V3 f = v3(hr[3]);
                        const uint h = id.x, i0 = id.y, i1 = id.z, i2 = id.w;
                        V3 fs[4] = {f, -(f * wp.x), -(f * wp.y), -(f * wp.z)};
                        uint ns[4] = {h, i1, i2, i0};
                        for (uint e = 0; e < 4u; e++)
                        {
                            const uint cc = e == 0u ? b : a;
                            atomic_fetch_add_explicit(&XACC(cc)[ns[e] * 3], toFixed(fs[e].x), memory_order_relaxed);
                            atomic_fetch_add_explicit(&XACC(cc)[ns[e] * 3 + 1], toFixed(fs[e].y), memory_order_relaxed);
                            atomic_fetch_add_explicit(&XACC(cc)[ns[e] * 3 + 2], toFixed(fs[e].z), memory_order_relaxed);
                            // (This turn's staging and counts cleared for the next.)
                            for (uint q = 0; q < 6u; q++) atomic_store_explicit(&(XSTAGE(cc) + ns[e] * 6)[q], 0, memory_order_relaxed);
                            atomic_store_explicit(&XSPLIT(a, cc)[ns[e]], 0u, memory_order_relaxed);
                        }
                        atomic_fetch_or_explicit(XFLAG(a), 1u, memory_order_relaxed);
                        atomic_fetch_or_explicit(XFLAG(b), 1u, memory_order_relaxed);
                    }
                    BAR;
                }
                if (lid == 0) { misc[5] += 1u; misc[6] = max(misc[6], nh0 + nh1); }
            }
            XBAR; // the bodies go on
        }
