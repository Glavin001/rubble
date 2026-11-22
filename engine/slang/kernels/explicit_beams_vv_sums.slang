// Included by explicit_beams.slang before vvResAcc.

// A staged hit's four nodes' beam sums (vvBeamSumOf; 0 immovable): h, tn.y, tn.z, tn.x.
// Computed once for the residual and the guard, which both read them (nothing writes
// the beams' forces between the two), at one call site: the compilers inline every
// call, and the eight sums of the two were eight copies of vvBeamSumOf at each site.
struct VvSums
{
    float3 h;
    float3 y;
    float3 z;
    float3 x;
};
VvSums vvSumsOf(uint va, uint vh, uint h, uint4 tn, uint parity)
{
    float3 b[4];
    for (uint k = 0u; k < 4u; k++)
    {
        uint x = k == 0u ? h : (k == 1u ? tn.y : (k == 2u ? tn.z : tn.x));
        b[k] = vvImmovable(x) ? float3(0.0, 0.0, 0.0) : vvBeamSumOf(x, k == 0u ? vh : va, parity);
    }
    VvSums S;
    S.h = b[0];
    S.y = b[1];
    S.z = b[2];
    S.x = b[3];
    return S;
}
