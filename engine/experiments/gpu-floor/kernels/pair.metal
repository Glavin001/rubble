        // ===== Body against body (mechanism 2): the contact pass on the bodies' own cores =====
        // A pair's two bodies each step on their own workgroup (their own core) and meet
        // at cross-group barriers. After a step each body does the half of the reference's
        // contact pass that its own triangles own (explicit_contact.rs contact_pass,
        // triangle_loop), both at once: its triangles (already in its threadgroup memory)
        // against the partner's contact points (copied from the partner's block into the
        // beam scalars' array, free between two beam passes). Each stage's survivors go on
        // a compact list, a lane an item, so no lane waits on a warp's slowest. The
        // owners' turns then run one after the other, as the reference's (owner 0's hits
        // are applied before owner 1's residual and guard read the forces): the other
        // body waits at the barrier. Bodies whose boxes are apart meet once a substep.
        // The culls before the reference's own test (a point in the triangle's padded box)
        // only ever drop what that test drops: each compares against the same rounded
        // terms with a pad no smaller.
        #define ME (car & 1u)
        #define PA (1u - ME)
        #define CB(c) (big + ((car & ~1u) + (c)) * CAR_WORDS)
        // (What a body publishes for its partner, in one of two slots by the substep's
        // parity: a body can run one barrier ahead of its partner, never two, so the
        // partner still finds the substep's words when it reads them.)
        #define XPUB(c) ((device atomic_uint*)(CB(c) + O_XPUB + (xstep & 1u) * 8u))
        #define XNODE(c) ((device float4*)(CB(c) + O_NODE))
        #define XACC(c) ((device atomic_int*)(CB(c) + O_ACC))
        #define XSHIFT(c) ((device atomic_int*)(CB(c) + O_SHIFT))
        #define XCLIP(c) ((device atomic_uint*)(CB(c) + O_CLIP))
        #define XFLAG(c) ((device atomic_uint*)(CB(c) + O_MISC + 12))
        #define XTOUCHK(c, n, k) atomic_fetch_or_explicit(&((device atomic_uint*)(CB(c) + O_TOUCH))[(n) >> 3], (uint)(k) << (((n) & 7u) * 4u), memory_order_relaxed)
        #define XSTAGE(c) ((device atomic_int*)(cb + O_XSTG + (c) * NN * 6))
        #define XSPLIT(c) ((device atomic_uint*)(cb + O_XSPL + (c) * NN))
        #define XTRI (cb + O_XTRI)
        #define XTC (cb + O_XTC)
        #define XCND (cb + O_XCND)
        #define XHIT ((device float4*)(cb + O_XHIT))
        #define HX 8
        // A turn's per-node words in threadgroup memory: a slot a node, 14 words of the beam
        // scalars' array (force 3, mass, hits that name it, staged sums 6, contact sums at
        // the turn's start 3); a hit has four, so this many hits a turn.
        #define XH2CAP (NGS / 56u)
        #define XMAPA ((device atomic_uint*)(cb + O_XMAP))
        #define XMAPB ((device atomic_uint*)(cb + O_XMAP + NN))
        #define XLF(s_) v3(tgG[(s_) * 14u], tgG[(s_) * 14u + 1u], tgG[(s_) * 14u + 2u])
        #define XLM(s_) tgG[(s_) * 14u + 3u]
        #define XLKP(s_) ((TGPTR(atomic_uint))&tgG[(s_) * 14u + 4u])
        #define XLK(s_) atomic_load_explicit(XLKP(s_), memory_order_relaxed)
        #define XLS(s_, q_) ((TGPTR(atomic_int))&tgG[(s_) * 14u + 5u + (q_)])
        #define XLL(s_, q_) atomic_load_explicit(XLS(s_, q_), memory_order_relaxed)
        #define XLA(s_, q_) as_type<int>(tgG[(s_) * 14u + 11u + (q_)])
        #define XSACC(f_, s_) (XLM(s_) == 0.0f ? v3(0, 0, 0) : (f_) / XLM(s_))
        #define XSIM(s_, k_) (XLM(s_) == 0.0f ? 0.0f : (k_) / XLM(s_))
        // The partner's nodes in this group's memory: position and travel a step (|v|·h,
        // the pad's term), four floats a node; then its clusters' boxes. Its velocities
        // are read from its block where a candidate or a hit needs one.
        #define PP(i) (v3(0, 0, 0) + v3(tgG[(i) * 4u], tgG[(i) * 4u + 1u], tgG[(i) * 4u + 2u]))
        #define PS(i) tgG[(i) * 4u + 3u]
        #define PCB(q) (NN * 4u + (q) * 6u)
        #define OP(i) (v3(0, 0, 0) + TGP(i))
        #define XM(i) nodeStatic[i].x
        #define XIMM(i) ((as_type<uint>(nodeStatic[i].z) & NODE_IMMOVABLE) != 0u)
        #define XACCEL(f, i) (XIMM(i) ? v3(0, 0, 0) : (f) / XM(i))
        #define XIM(i, kf) (XIMM(i) ? 0.0f : (kf) / XM(i))
        #define ILOAD(p) atomic_load_explicit(p, memory_order_relaxed)
        #define ISTORE(p, v) atomic_store_explicit(p, v, memory_order_relaxed)
        #define XNA(c) ((device atomic_uint*)(CB(c) + O_NODE))
        #define XLD3(c, w) v3(as_type<float>(ILOAD(&XNA(c)[(w) * 4u])), as_type<float>(ILOAD(&XNA(c)[(w) * 4u + 1u])), as_type<float>(ILOAD(&XNA(c)[(w) * 4u + 2u])))
        // (XATOM 3: what the partner stored atomically read with one plain vector load.)
        #if XATOM == 1 || XATOM == 2
        #define XNV(c, w) XLD3(c, w)
        #else
        #define XNV(c, w) v3(XNODE(c)[w])
        #endif
        // (A body's own nodes' forces from its own block: any node's, a plain load. The
        // partner's only for the nodes it published, the ones that can be hit.)
        #define XF(c, i) ((c) == ME ? v3(nodeState[(i) * 3u + 2u]) : XNV(c, (i) * 3u + 2u))
        // Whether a point of body `pt` can pass any triangle's test of body `ow`: inside
        // `ow`'s box with the contact band and the largest pad any pair of them can have
        // (the same rounded terms as the reference's test, each bound no smaller). Both
        // bodies evaluate it from the same published words, so they agree.
        #define XREACH(ow, pt, p) ( \
            (p).x >= (unordF(TXI((ow) * 6u)) - CR) - XPADMAX(ow, pt) && (p).y >= (unordF(TXI((ow) * 6u + 1u)) - CR) - XPADMAX(ow, pt) && (p).z >= (unordF(TXI((ow) * 6u + 2u)) - CR) - XPADMAX(ow, pt) && \
            (p).x <= (unordF(TXI((ow) * 6u + 3u)) + CR) + XPADMAX(ow, pt) && (p).y <= (unordF(TXI((ow) * 6u + 4u)) + CR) + XPADMAX(ow, pt) && (p).z <= (unordF(TXI((ow) * 6u + 5u)) + CR) + XPADMAX(ow, pt))
        #define XPADMAX(ow, pt) ((as_type<float>(TXI(12u + (ow))) * H + as_type<float>(TXI(12u + (pt))) * H) + CLIP_DEPTH)
        #define PV(i) XNV(PA, (i) * 3u + 1u)
        #define TXI(k) ILOAD(&tgX[k])
        // A triangle's box with the contact band, and its nodes' farthest travel a step.
        #define TRIBOX(tn) \
            V3 p0 = OP(tn.x), p1 = OP(tn.y), p2 = OP(tn.z); \
            float travel = rmax(rmax(rmax(0.0f, len3(TGV(tn.x))), len3(TGV(tn.y))), len3(TGV(tn.z))) * H; \
            V3 lo = v3(rmin(rmin(p0.x, p1.x), p2.x) - CR, rmin(rmin(p0.y, p1.y), p2.y) - CR, rmin(rmin(p0.z, p1.z), p2.z) - CR); \
            V3 hi = v3(rmax(rmax(p0.x, p1.x), p2.x) + CR, rmax(rmax(p0.y, p1.y), p2.y) + CR, rmax(rmax(p0.z, p1.z), p2.z) + CR);
        {
            // A: this body's box and top speed (a warp reduces them).
            if (lid < 32u)
            {
                V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
                float vt = 0.0f;
                for (uint j = lid; j < NN; j += 32u)
                {
                    V3 w = OP(j);
                    lo = v3(rmin(lo.x, w.x), rmin(lo.y, w.y), rmin(lo.z, w.z));
                    hi = v3(rmax(hi.x, w.x), rmax(hi.y, w.y), rmax(hi.z, w.z));
                    vt = rmax(vt, len3(TGV(j)));
                }
                atomic_fetch_min_explicit(&tgX[ME * 6u], ordF(lo.x), memory_order_relaxed);
                atomic_fetch_min_explicit(&tgX[ME * 6u + 1u], ordF(lo.y), memory_order_relaxed);
                atomic_fetch_min_explicit(&tgX[ME * 6u + 2u], ordF(lo.z), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[ME * 6u + 3u], ordF(hi.x), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[ME * 6u + 4u], ordF(hi.y), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[ME * 6u + 5u], ordF(hi.z), memory_order_relaxed);
                atomic_fetch_max_explicit(&tgX[12u + ME], as_type<int>(vt), memory_order_relaxed);
            }
            BAR;
            if (lid == 0u)
            {
                for (uint k = 0; k < 6u; k++) ISTORE(&XPUB(ME)[k], (uint)TXI(ME * 6u + k));
                ISTORE(&XPUB(ME)[6], (uint)TXI(12u + ME));
            }
            XBAR; // both bodies have stepped and published
            if (lid == 0u)
            {
                for (uint k = 0; k < 6u; k++) ISTORE(&tgX[PA * 6u + k], (int)ILOAD(&XPUB(PA)[k]));
                ISTORE(&tgX[12u + PA], (int)ILOAD(&XPUB(PA)[6]));
                V3 l0 = v3(unordF(TXI(0)), unordF(TXI(1)), unordF(TXI(2))), h0 = v3(unordF(TXI(3)), unordF(TXI(4)), unordF(TXI(5)));
                V3 l1 = v3(unordF(TXI(6)), unordF(TXI(7)), unordF(TXI(8))), h1 = v3(unordF(TXI(9)), unordF(TXI(10)), unordF(TXI(11)));
                // (The bodies' boxes with BOX_MARGIN apart: no pass.)
                const bool apart = (h0.x + 0.05f) < (l1.x - 0.05f) || (h0.y + 0.05f) < (l1.y - 0.05f) || (h0.z + 0.05f) < (l1.z - 0.05f) ||
                                   (h1.x + 0.05f) < (l0.x - 0.05f) || (h1.y + 0.05f) < (l0.y - 0.05f) || (h1.z + 0.05f) < (l0.z - 0.05f);
                ISTORE(&tgX[14], (apart || SKIP(12)) ? 0 : 1);
        #ifdef EMU_TRACE
                fprintf(stderr, "car %u step %u: apart %d box0 %g..%g box1 %g..%g\n", car, step, (int)apart, l0.x, h0.x, l1.x, h1.x);
        #endif
            }
            BAR;
            if (TXI(14) != 0)
            {
                // The boxes meet: this body's nodes published for its partner (atomic
                // stores: a plain one is not seen on another core), and the bodies meet again.
                if (isNode)
                {
        #if XATOM >= 2
                    device atomic_uint* na = (device atomic_uint*)(nodeState + ni * 3);
                    // Every node's position (the partner's boxes and side votes read any
                    // of them); velocity and force only of the nodes a partner's triangle
                    // can reach (six atomic stores fewer for the others). The force also
                    // with a plain store, for this body's own lanes.
                    ISTORE(&na[0], as_type<uint>(Pown.x)); ISTORE(&na[1], as_type<uint>(Pown.y)); ISTORE(&na[2], as_type<uint>(Pown.z));
                    nodeState[ni * 3 + 2] = f4(F0, 0);
                    if (XREACH(PA, ME, Pown))
                    {
                        ISTORE(&na[4], as_type<uint>(Vown.x)); ISTORE(&na[5], as_type<uint>(Vown.y)); ISTORE(&na[6], as_type<uint>(Vown.z));
                        ISTORE(&na[8], as_type<uint>(F0.x)); ISTORE(&na[9], as_type<uint>(F0.y)); ISTORE(&na[10], as_type<uint>(F0.z));
                    }
        #else
                    nodeState[ni * 3] = f4(Pown, 0);
                    nodeState[ni * 3 + 1] = f4(Vown, 0);
                    nodeState[ni * 3 + 2] = f4(F0, 0);
        #endif
                }
                if (lid < NCLP)
                {
                    // This body's clusters of contact points' boxes, for its partner's cull
                    // (from its own memory: read from the partner's block they were 28
                    // scattered loads a lane, one after another).
                    V3 lo = v3(3.0e38f, 3.0e38f, 3.0e38f), hi = v3(-3.0e38f, -3.0e38f, -3.0e38f);
                    device const uint4* cl4 = (device const uint4*)(tab + T_PCL + lid * 32u);
                    for (uint jj = 0; jj < 8u; jj++)
                    {
                        uint4 w4 = cl4[jj];
                        for (uint e = 0; e < 4u; e++)
                        {
                            uint h = w4[e];
                            if (h == NONE) continue;
                            V3 x = OP(h);
                            lo = v3(rmin(lo.x, x.x), rmin(lo.y, x.y), rmin(lo.z, x.z));
                            hi = v3(rmax(hi.x, x.x), rmax(hi.y, x.y), rmax(hi.z, x.z));
                        }
                    }
                    device atomic_uint* cbx = (device atomic_uint*)(cb + O_XCLB + lid * 8u);
                    ISTORE(&cbx[0], as_type<uint>(lo.x)); ISTORE(&cbx[1], as_type<uint>(lo.y)); ISTORE(&cbx[2], as_type<uint>(lo.z));
                    ISTORE(&cbx[4], as_type<uint>(hi.x)); ISTORE(&cbx[5], as_type<uint>(hi.y)); ISTORE(&cbx[6], as_type<uint>(hi.z));
                }
                XBAR;
                // D: the partner's nodes into this group's memory; this body's triangles
                // against the partner's box and then its clusters' boxes.
                for (uint k = lid; k < NN; k += LANES)
                {
                    V3 p = XNV(PA, k * 3u);
                    tgG[k * 4u] = p.x; tgG[k * 4u + 1u] = p.y; tgG[k * 4u + 2u] = p.z;
                    // (A point no triangle can reach was not published: a travel that fails
                    // every box test.)
                    float sv = -3.0e38f;
        #if XATOM >= 2
                    if (XREACH(ME, PA, p))
        #endif
                        sv = len3(XNV(PA, k * 3u + 1u)) * H;
                    tgG[k * 4u + 3u] = sv;
                }
                for (uint t = lid; t < NT; t += LANES)
                {
                    if (triDead[t] != 0u) continue;
                    uint4 tn = tris[t];
                    TRIBOX(tn)
                    // (No contact point's pad is more than the partner's fastest node's.)
                    float padTop = (travel + as_type<float>(TXI(12u + PA)) * H) + CLIP_DEPTH;
                    V3 qlo = v3(unordF(TXI(PA * 6u)), unordF(TXI(PA * 6u + 1u)), unordF(TXI(PA * 6u + 2u)));
                    V3 qhi = v3(unordF(TXI(PA * 6u + 3u)), unordF(TXI(PA * 6u + 4u)), unordF(TXI(PA * 6u + 5u)));
                    if (qhi.x < lo.x - padTop || qhi.y < lo.y - padTop || qhi.z < lo.z - padTop || qlo.x > hi.x + padTop || qlo.y > hi.y + padTop || qlo.z > hi.z + padTop) continue;
                    // The partner's clusters of contact points this triangle's padded box
                    // reaches (their boxes as the partner published them), each a (triangle,
                    // cluster) item. (SKIP(17..20), timing only: the pass cut short from a
                    // stage on, on both bodies.)
                    if (SKIP(17)) continue;
                    device const float4* pcb = (device const float4*)(CB(PA) + O_XCLB);
                    for (uint q = 0; q < NCLP; q++)
                    {
                        const float4 clo = pcb[q * 2u], chi = pcb[q * 2u + 1u];
                        if (chi.x < lo.x - padTop || chi.y < lo.y - padTop || chi.z < lo.z - padTop || clo.x > hi.x + padTop || clo.y > hi.y + padTop || clo.z > hi.z + padTop) continue;
                        uint ci = (uint)atomic_fetch_add_explicit(&tgX[16], 1, memory_order_relaxed);
                        XTC[ci] = t << 8 | q;
                    }
                }
                BAR;
                // F: a lane a (triangle, cluster): the cluster's points by the reference's
                // padded box test.
                const uint ntc = SKIP(18) ? 0u : (uint)TXI(16);
                for (uint k = lid; k < ntc; k += LANES)
                {
                    const uint w = XTC[k];
                    const uint t = w >> 8, q = w & 255u;
                    uint4 tn = tris[t];
                    TRIBOX(tn)
                    device const uint4* cl4 = (device const uint4*)(tab + T_PCL + q * 32u);
                    for (uint jj = 0; jj < 8u; jj++)
                    {
                        uint4 w4 = cl4[jj];
                        for (uint e = 0; e < 4u; e++)
                        {
                            uint h = w4[e];
                            if (h == NONE) continue;
                            V3 x = PP(h);
                            float pad = (travel + PS(h)) + CLIP_DEPTH;
                            if (!(x.x >= lo.x - pad && x.y >= lo.y - pad && x.z >= lo.z - pad && x.x <= hi.x + pad && x.y <= hi.y + pad && x.z <= hi.z + pad)) continue;
                            uint ci = (uint)atomic_fetch_add_explicit(&tgX[17], 1, memory_order_relaxed);
                            if (ci < XCND_CAP) XCND[ci] = t << 16 | h;
                        }
                    }
                }
                BAR;
                // G: a lane a candidate: the contact rule (frame, held side, swept crossing,
                // the side vote); a contact's side recorded, its hit kept.
                const uint ncx = SKIP(19) ? 0u : min((uint)TXI(17), (uint)XCND_CAP);
                if (lid == 0 && (uint)TXI(17) > XCND_CAP) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 4u;
                for (uint k = lid; k < ncx; k += LANES)
                {
                    const uint cw = XCND[k];
                    const uint a = ME, b = PA, t = cw >> 16, h = cw & 0xFFFFu;
                    uint4 tn = tris[t];
                    V3 p0 = OP(tn.x), p1 = OP(tn.y), p2 = OP(tn.z);
                    V3 x = PP(h);
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
        #if XATOM == 3
                    const uint4 cq = ((device const uint4*)(CB(b) + O_CLIP))[h];
                    uint c0 = cq.x, c1 = cq.y;
        #else
                    uint c0 = ILOAD(&XCLIP(b)[h * 4]), c1 = ILOAD(&XCLIP(b)[h * 4 + 1]);
        #endif
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
                        V3 q0 = p0 - TGV(tn.x) * H, q1 = p1 - TGV(tn.y) * H, q2 = p2 - TGV(tn.z) * H;
                        V3 ub = q1 - q0, wb = q2 - q0;
                        V3 cb2 = cross3(ub, wb);
                        float lenb = sqrt(((cb2.x * cb2.x) + (cb2.y * cb2.y)) + (cb2.z * cb2.z));
                        V3 nb = lenb > 1e-8f ? cb2 * (1.0f / lenb) : cb2;
                        float detb = dot3(ub, cross3(wb, nb));
                        bool okb = fabs(detb) > 1e-6f;
                        V3 rab = v3(0, 0, 0), rbb = rab, rdb = rab;
                        if (okb) { float r = 1.0f / detb; rab = cross3(wb, nb) * r; rbb = cross3(nb, ub) * r; rdb = cross3(ub, wb) * r; }
                        float4 l0 = frameLocal(q0, rab, rbb, rdb, okb, x - PV(h) * H);
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
                                    V3 e = PP(tab[T_NBL + j]) - p0;
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
                    XTOUCHK(b, h, 5u); XTOUCHK(a, tn.x, 1u); XTOUCHK(a, tn.y, 1u); XTOUCHK(a, tn.z, 1u);
                    float pen = rmin(CR - dd, CR);
                    uint hi_ = (uint)atomic_fetch_add_explicit(&tgX[18], 1, memory_order_relaxed);
                    if (hi_ < XH2CAP)
                    {
                        device float4* hr = XHIT + hi_ * HX;
                        hr[0] = as_type<float4>(uint4(h, tn.x, tn.y, tn.z));
                        hr[1] = float4(wgt.x, wgt.y, wgt.z, pen);
                        hr[2] = f4(n * side, 0);
                        // The forces at the pass's start (the contact sums before any turn).
                        uint ids[4] = {h, tn.y, tn.z, tn.x};
                        for (uint e = 0; e < 4u; e++)
                        {
                            const uint xc = e == 0u ? b : a;
                            V3 f = XF(xc, ids[e]) + v3(float(ILOAD(&XACC(xc)[ids[e] * 3])), float(ILOAD(&XACC(xc)[ids[e] * 3 + 1])), float(ILOAD(&XACC(xc)[ids[e] * 3 + 2]))) * (1.0f / 32.0f);
                            hr[4 + e] = f4(f, 0);
                        }
                        // (Each node's slot for the turn: the smallest (hit, corner) that names it.)
                        atomic_fetch_min_explicit(&XMAPB[h], hi_ * 4u, memory_order_relaxed);
                        atomic_fetch_min_explicit(&XMAPA[tn.x], hi_ * 4u + 1u, memory_order_relaxed);
                        atomic_fetch_min_explicit(&XMAPA[tn.y], hi_ * 4u + 2u, memory_order_relaxed);
                        atomic_fetch_min_explicit(&XMAPA[tn.z], hi_ * 4u + 3u, memory_order_relaxed);
                    }
                }
                BAR;
                if (!SKIP(20))
                {
                if (lid == 0u)
                {
                    if ((uint)TXI(18) > XH2CAP) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 2u;
                    ISTORE(&XPUB(ME)[7], min((uint)TXI(18), (uint)XH2CAP));
                }
                XBAR; // both bodies' hits are known
                if (lid == 0u)
                {
                    ISTORE(&tgX[19u + ME], (int)min((uint)TXI(18), (uint)XH2CAP));
                    ISTORE(&tgX[19u + PA], (int)ILOAD(&XPUB(PA)[7]));
        #ifdef EMU_TRACE
                    fprintf(stderr, "car %u step %u: tris %d tc %d cands %d hits %d,%d\n", car, step, TXI(15), TXI(16), TXI(17), TXI(19), TXI(20));
        #endif
                }
                BAR;
                // The owners' turns: owner 0's, then owner 1's.
                for (uint a = 0; a < 2u; a++)
                {
                    const uint b = 1u - a, nh = (uint)TXI(19u + a);
                    if (nh == 0u) continue;
                    if (a == ME)
                    {
                    #define HREC(k) (XHIT + (k) * HX)
                    // T0: the turn's slots filled. This body's nodes by their own lanes (the
                    // force from the lane's register); the partner's points by the hit that
                    // names the slot. A hit's lane keeps its record and slots in registers.
                    const uint mySlot = isNode ? ILOAD(&XMAPA[ni]) : NONE;
                    uint4 hid = uint4(0u), hsl = uint4(0u);
                    float4 hwp = float4(0.0f);
                    V3 hnn = v3(0, 0, 0), hf = v3(0, 0, 0), hpv = v3(0, 0, 0);
                    float hres = 0.0f;
                    if (mySlot != NONE)
                    {
                        const uint o = mySlot * 14u;
                        tgG[o] = F0.x; tgG[o + 1u] = F0.y; tgG[o + 2u] = F0.z;
                        tgG[o + 3u] = XIMM(ni) ? 0.0f : massN;
                        tgG[o + 4u] = 0.0f; tgG[o + 5u] = 0.0f; tgG[o + 6u] = 0.0f; tgG[o + 7u] = 0.0f;
                        tgG[o + 8u] = 0.0f; tgG[o + 9u] = 0.0f; tgG[o + 10u] = 0.0f;
                        tgG[o + 11u] = as_type<float>(ILOAD(&XACC(a)[ni * 3]));
                        tgG[o + 12u] = as_type<float>(ILOAD(&XACC(a)[ni * 3 + 1]));
                        tgG[o + 13u] = as_type<float>(ILOAD(&XACC(a)[ni * 3 + 2]));
                    }
                    if (lid < nh)
                    {
                        device float4* hr = HREC(lid);
                        hid = as_type<uint4>(hr[0]);
                        hwp = hr[1];
                        hnn = v3(hr[2]);
                        hpv = PV(hid.x);
                        hsl = uint4(ILOAD(&XMAPB[hid.x]), ILOAD(&XMAPA[hid.y]), ILOAD(&XMAPA[hid.z]), ILOAD(&XMAPA[hid.w]));
                        if (hsl.x == lid * 4u)
                        {
                            const uint o = hsl.x * 14u, h = hid.x;
                            V3 fb = XF(b, h);
                            tgG[o] = fb.x; tgG[o + 1u] = fb.y; tgG[o + 2u] = fb.z;
                            tgG[o + 3u] = XIMM(h) ? 0.0f : XM(h);
                            tgG[o + 4u] = 0.0f; tgG[o + 5u] = 0.0f; tgG[o + 6u] = 0.0f; tgG[o + 7u] = 0.0f;
                            tgG[o + 8u] = 0.0f; tgG[o + 9u] = 0.0f; tgG[o + 10u] = 0.0f;
                            tgG[o + 11u] = as_type<float>(ILOAD(&XACC(b)[h * 3]));
                            tgG[o + 12u] = as_type<float>(ILOAD(&XACC(b)[h * 3 + 1]));
                            tgG[o + 13u] = as_type<float>(ILOAD(&XACC(b)[h * 3 + 2]));
                        }
                    }
                    BAR;
                    // T1: how many hits name each node; the maps cleared for the next step.
                    if (lid < nh)
                    {
                        atomic_fetch_add_explicit(XLKP(hsl.x), 1u, memory_order_relaxed);
                        atomic_fetch_add_explicit(XLKP(hsl.y), 1u, memory_order_relaxed);
                        atomic_fetch_add_explicit(XLKP(hsl.z), 1u, memory_order_relaxed);
                        atomic_fetch_add_explicit(XLKP(hsl.w), 1u, memory_order_relaxed);
                        if (hsl.x == lid * 4u) ISTORE(&XMAPB[hid.x], NONE);
                    }
                    if (mySlot != NONE) ISTORE(&XMAPA[ni], NONE);
                    BAR;
                    // The first pass, from the forces at the pass's start.
                    if (lid < nh)
                    {
                        device float4* hr = HREC(lid);
                        const uint h = hid.x, i0 = hid.y, i1 = hid.z, i2 = hid.w;
                        const uint sh = hsl.x, s0 = hsl.y, s1 = hsl.z, s2 = hsl.w;
                        const float alpha = hwp.x, beta = hwp.y, gamma = hwp.z, depth = hwp.w;
                        const V3 n = hnn;
                        V3 v = hpv - ((TGV(i1) * alpha + TGV(i2) * beta) + TGV(i0) * gamma);
                        const float imh = XSIM(sh, float(XLK(sh)));
                        const float im1 = XSIM(s1, float(XLK(s1))), im2 = XSIM(s2, float(XLK(s2))), im0 = XSIM(s0, float(XLK(s0)));
                        float inv = imh + tbInvMass(im1, im2, im0, alpha, beta, gamma);
                        float m = inv > 0.0f ? 1.0f / inv : 0.0f;
                        V3 at = (XSACC(v3(hr[5]), s1) * alpha + XSACC(v3(hr[6]), s2) * beta) + XSACC(v3(hr[7]), s0) * gamma;
                        V3 fh = (XSACC(v3(hr[4]), sh) - at) * m;
                        float budget;
                        V3 f = contactResponse(fh, v, m, n, BOXS, depth, nodeStatic[h].y, budget);
                        if (!isZero3(f))
                        {
                            float delta = shiftDepth(depth, budget, dot3(fh, n));
                            if (delta > 0.0f)
                            {
                                float q = delta * m;
                                V3 dh = n * (q * imh);
                                XTOUCHK(b, h, 2u);
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
                                        XTOUCHK(a, ti[e], 2u);
                                        atomic_fetch_add_explicit(&XSHIFT(a)[ti[e] * 3], toFixedShift(dt.x), memory_order_relaxed);
                                        atomic_fetch_add_explicit(&XSHIFT(a)[ti[e] * 3 + 1], toFixedShift(dt.y), memory_order_relaxed);
                                        atomic_fetch_add_explicit(&XSHIFT(a)[ti[e] * 3 + 2], toFixedShift(dt.z), memory_order_relaxed);
                                        atomic_fetch_or_explicit(XFLAG(a), 2u, memory_order_relaxed);
                                    }
                                }
                            }
                        }
                        hf = f;
                        V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                        uint ss[4] = {sh, s1, s2, s0};
                        for (uint e = 0; e < 4u; e++)
                        {
                            atomic_fetch_add_explicit(XLS(ss[e], 0u), toFixed(fs[e].x), memory_order_relaxed);
                            atomic_fetch_add_explicit(XLS(ss[e], 1u), toFixed(fs[e].y), memory_order_relaxed);
                            atomic_fetch_add_explicit(XLS(ss[e], 2u), toFixed(fs[e].z), memory_order_relaxed);
                        }
                    }
                    BAR;
                    // The residual pass, from the forces now (with the first pass's sums).
                    #define XNOW1(s_) (XLF(s_) + v3(float(XLA(s_, 0u) + XLL(s_, 0u)), float(XLA(s_, 1u) + XLL(s_, 1u)), float(XLA(s_, 2u) + XLL(s_, 2u))) * (1.0f / 32.0f))
                    if (lid < nh)
                    {
                        const uint i0 = hid.y, i1 = hid.z, i2 = hid.w;
                        const uint sh = hsl.x, s0 = hsl.y, s1 = hsl.z, s2 = hsl.w;
                        const float alpha = hwp.x, beta = hwp.y, gamma = hwp.z;
                        const V3 n = hnn;
                        V3 v = hpv - ((TGV(i1) * alpha + TGV(i2) * beta) + TGV(i0) * gamma);
                        V3 at = (XSACC(XNOW1(s1), s1) * alpha + XSACC(XNOW1(s2), s2) * beta) + XSACC(XNOW1(s0), s0) * gamma;
                        float an = dot3(XSACC(XNOW1(sh), sh) - at, n);
                        float vn = dot3(v, n);
                        float r = 0.0f;
                        if (vn < 0.0f && an * H + vn < 0.0f)
                        {
                            float inv = XSIM(sh, float(XLK(sh))) + tbInvMass(XSIM(s1, float(XLK(s1))), XSIM(s2, float(XLK(s2))), XSIM(s0, float(XLK(s0))), alpha, beta, gamma);
                            if (inv > 0.0f)
                            {
                                float m = 1.0f / inv;
                                float rr = -(an * m) - (vn * m) / H;
                                if (rr > 0.0f) r = rr;
                            }
                        }
                        hres = r;
                        if (r > 0.0f)
                        {
                            V3 f = n * r;
                            V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                            uint ss[4] = {sh, s1, s2, s0};
                            for (uint e = 0; e < 4u; e++)
                            {
                                atomic_fetch_add_explicit(XLS(ss[e], 3u), toFixed(fs[e].x), memory_order_relaxed);
                                atomic_fetch_add_explicit(XLS(ss[e], 4u), toFixed(fs[e].y), memory_order_relaxed);
                                atomic_fetch_add_explicit(XLS(ss[e], 5u), toFixed(fs[e].z), memory_order_relaxed);
                            }
                        }
                    }
                    BAR;
                    // The work guard, from the forces now (with both passes' sums); each
                    // hit's total into the bodies' contact sums.
                    #define XNOW2(s_) (XLF(s_) + v3(float(XLA(s_, 0u) + (XLL(s_, 0u) + XLL(s_, 3u))), float(XLA(s_, 1u) + (XLL(s_, 1u) + XLL(s_, 4u))), float(XLA(s_, 2u) + (XLL(s_, 2u) + XLL(s_, 5u)))) * (1.0f / 32.0f))
                    if (lid < nh)
                    {
                        const uint h = hid.x, i0 = hid.y, i1 = hid.z, i2 = hid.w;
                        const uint sh = hsl.x, s0 = hsl.y, s1 = hsl.z, s2 = hsl.w;
                        const float alpha = hwp.x, beta = hwp.y, gamma = hwp.z;
                        const V3 n = hnn;
                        V3 t = hf;
                        if (hres > 0.0f) t = t + n * hres;
                        V3 f = t;
                        if (!isZero3(t))
                        {
                            V3 v = hpv - ((TGV(i1) * alpha + TGV(i2) * beta) + TGV(i0) * gamma);
                            V3 ac = XSACC(XNOW2(sh), sh) - ((XSACC(XNOW2(s1), s1) * alpha + XSACC(XNOW2(s2), s2) * beta) + XSACC(XNOW2(s0), s0) * gamma);
                            float inv = XSIM(sh, 1.0f) + tbInvMass(XSIM(s1, 1.0f), XSIM(s2, 1.0f), XSIM(s0, 1.0f), alpha, beta, gamma);
                            V3 aRest = ac - t * inv;
                            float aa = dot3(t, (v + v) + aRest * H);
                            float bb = (H * dot3(t, t)) * inv;
                            if (aa + bb > 0.0f && bb > 0.0f)
                            {
                                float sc = aa < 0.0f ? rmin(-aa / bb, 1.0f) : 0.0f;
                                f = t + t * -(1.0f - sc);
                            }
                        }
                        V3 fs[4] = {f, -(f * alpha), -(f * beta), -(f * gamma)};
                        uint ns[4] = {h, i1, i2, i0};
                        for (uint e = 0; e < 4u; e++)
                        {
                            const uint xc = e == 0u ? b : a;
                            atomic_fetch_add_explicit(&XACC(xc)[ns[e] * 3], toFixed(fs[e].x), memory_order_relaxed);
                            atomic_fetch_add_explicit(&XACC(xc)[ns[e] * 3 + 1], toFixed(fs[e].y), memory_order_relaxed);
                            atomic_fetch_add_explicit(&XACC(xc)[ns[e] * 3 + 2], toFixed(fs[e].z), memory_order_relaxed);
                        }
                        atomic_fetch_or_explicit(XFLAG(a), 1u, memory_order_relaxed);
                        atomic_fetch_or_explicit(XFLAG(b), 1u, memory_order_relaxed);
                    }
                    }
                    XBAR; // the turn is applied
                }
                }
            }
            xstep++;
            // What the pass left for this body picked up; the pass's counters cleared.
            if (lid == 0u)
            {
                const uint xf = atomic_exchange_explicit((device atomic_uint*)&misc[12], 0u, memory_order_relaxed);
                if ((xf & 1u) != 0u) tgFlags[F_HITS] = 1u;
                if ((xf & 2u) != 0u) tgFlags[F_SHIFT] = 1u;
                if ((xf & 4u) != 0u) tgFlags[F_REC] = 1u;
                if (xlate != 0u) tgFlags[F_STATUS] = tgFlags[F_STATUS] | 8u;
            }
            if (lid < 24u) ISTORE(&tgX[lid], lid >= 12u ? 0 : ((lid % 6u) < 3u ? 0x7FFFFFFF : int(0x80000000)));
            BAR;
        }
