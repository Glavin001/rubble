// The converted closest-feature queries (fq.inc: floor.rs fq_msl) run on the CPU over a
// file of cases, for fqcheck: fq <cases in> <results out>. A case is 28 floats (kind:
// 0 triangle, 1 beam, 2 point; the shape's kind; the band; a pad; the shape's centre,
// axes and half extents; three vertices), a result 24 words.
#include "shim.h"
#include "fq.inc"

static void put3(uint* o, float3 v) { o[0] = as_type<uint>(v.x); o[1] = as_type<uint>(v.y); o[2] = as_type<uint>(v.z); }

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 3;
    std::vector<uint> out;
    float c[28];
    while (fread(c, 4, 28, f) == 28) {
        FqShape s = {};
        s.kind = (uint)c[1];
        const float r = c[2];
        s.c = float3(c[4], c[5], c[6]);
        s.ax0 = float3(c[7], c[8], c[9]); s.ax1 = float3(c[10], c[11], c[12]); s.ax2 = float3(c[13], c[14], c[15]);
        s.hext = float3(c[16], c[17], c[18]);
        const float3 x0(c[19], c[20], c[21]), x1(c[22], c[23], c[24]), x2(c[25], c[26], c[27]);
        uint res[24] = {};
        float3 lo, hi, w, n, q, w2, q2;
        float g = 0.0f;
        bool two = false;
        fqBounds(s, lo, hi);
        const uint kind = (uint)c[0];
        if (kind == 0u) {
            res[0] = fqNear(x0, x1, x2, r, lo, hi);
            res[1] = fqSeparated(s, x0, x1, x2, r);
            if (fqContact(s, x0, x1, x2, r, w, n, g, q)) {
                res[2] = 1u; put3(res + 4, w); put3(res + 7, n); res[10] = as_type<uint>(g); put3(res + 11, q);
            }
        } else if (kind == 1u) {
            res[0] = fqSegNear(x0, x1, r, lo, hi);
            res[1] = fqSeparated(s, x0, x1, x1, r);
            if (fqSegContact(s, x0, x1, r, w, n, g, q, two, w2, q2)) {
                res[2] = 1u; put3(res + 4, w); put3(res + 7, n); res[10] = as_type<uint>(g); put3(res + 11, q);
                if (two) { res[3] = 1u; put3(res + 14, w2); put3(res + 17, q2); }
            }
        } else {
            res[0] = fqSegNear(x0, x0, r, lo, hi);
            if (fqPointContact(s, x0, r, n, g)) { res[2] = 1u; put3(res + 7, n); res[10] = as_type<uint>(g); }
        }
        out.insert(out.end(), res, res + 24);
    }
    fclose(f);
    FILE* o = fopen(argv[2], "wb");
    fwrite(out.data(), 4, out.size(), o);
    fclose(o);
    return 0;
}
