// Gate 0: layout + atomic round-trip. Explicit padding, no float3x3.

public struct AbiProbe
{
    float3 position;
    float pad0;
    float4 inertiaCol0;
    float4 inertiaCol1;
    float4 inertiaCol2;
    uint counter;
    uint pad1;
    uint pad2;
    uint pad3;
};

RWStructuredBuffer<AbiProbe> gProbes;
RWStructuredBuffer<uint> gAtomics;

[shader("compute")]
[numthreads(64, 1, 1)]
void abi_roundtrip(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= 1) return;
    AbiProbe p = gProbes[i];
    p.position = p.position + float3(1.0, 2.0, 3.0);
    p.inertiaCol0.x = p.inertiaCol0.x + 1.0;
    p.inertiaCol1.y = p.inertiaCol1.y + 1.0;
    p.inertiaCol2.z = p.inertiaCol2.z + 1.0;
    InterlockedAdd(gAtomics[0], 1);
    p.counter = p.counter + 1;
    gProbes[i] = p;
}
