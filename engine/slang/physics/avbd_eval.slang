#include "types.slang"
#include "math3d.slang"
#include "ldlt6.slang"
#include "shapes.slang"

public struct BodyScratch
{
    float3 pos;
    float4 rot;
    float3 inertialPos;
    float4 inertialRot;
    float3 initPos;
    float4 initRot;
    float invMass;
    float3 invInertia;
    uint flags;
    uint color;
    uint constraintStart;
    uint constraintCount;
};

public Vec6 jacobianLin(float3 n, float3 r, bool isBodyA)
{
    Vec6 j = vec6Zero();
    float s = isBodyA ? -1.0 : 1.0;
    j.v[0] = s * n.x;
    j.v[1] = s * n.y;
    j.v[2] = s * n.z;
    float3 t = s * cross(r, n);
    j.v[3] = t.x;
    j.v[4] = t.y;
    j.v[5] = t.z;
    return j;
}

public void contributeContact(
    inout Mat6 A, inout Vec6 rhs,
    BodyState body, Contact c, bool isA,
    float alpha, float3 bodyPos, float4 bodyRot)
{
    float3 r = isA
        ? quatRotate(bodyRot, c.anchorA)
        : quatRotate(bodyRot, c.anchorB);
    float3 worldAnchor = bodyPos + r;
    float3 otherHint = isA ? (c.anchorB) : (c.anchorA); // residual uses stored anchors
    // C_reg = (1-α) C0 + J Δx  ≈  penetration residual with stabilization
    float cN = c.penetration;
    float cReg = (1.0 - alpha) * c.c0N + alpha * 0.0 + cN - c.c0N;
    // Prefer paper form: C_reg = (1-α)*C0 + J·Δx; here C0 stored at match time.
    cReg = (1.0 - alpha) * c.c0N + (isA ? -1.0 : 1.0) * dot(c.normal, worldAnchor) * 0.0
         + c.penetration * alpha;
    // Stable practical form matching demo: drive toward (1-α) C0 each step.
    cReg = (1.0 - alpha) * c.c0N + (c.penetration - c.c0N);

    float3 f = float3(c.kN * cReg + c.lambdaN, 0.0, 0.0);
    // Friction rows use tangent residuals ~0 at rest
    float cT1 = (1.0 - alpha) * c.c0T1;
    float cT2 = 0.0;
    f.y = c.kT * cT1 + c.lambdaT1;
    f.z = c.kT * cT2 + c.lambdaT2;
    f = projectContactForce(f, c.friction);

    Vec6 jN = jacobianLin(c.normal, r, isA);
    Vec6 jT1 = jacobianLin(c.tangent1, r, isA);
    Vec6 jT2 = jacobianLin(c.tangent2, r, isA);
    mat6AddOuter(A, jN, c.kN);
    mat6AddOuter(A, jT1, c.kT);
    mat6AddOuter(A, jT2, c.kT);
    vec6AddScaled(rhs, jN, f.x);
    vec6AddScaled(rhs, jT1, f.y);
    vec6AddScaled(rhs, jT2, f.z);
}

public void contributeBallJoint(
    inout Mat6 A, inout Vec6 rhs,
    float3 bodyPos, float4 bodyRot, float3 localAnchor,
    float3 C, float3 C0, float k, float3 lambda, bool isBodyA, float alpha)
{
    float3 r = quatRotate(bodyRot, localAnchor);
    float3 Creg = C - alpha * C0;
    for (int axis = 0; axis < 3; axis++)
    {
        float3 n = float3(axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0);
        Vec6 j = jacobianLin(n, r, !isBodyA);
        float f = k * Creg[axis] + lambda[axis];
        mat6AddOuter(A, j, k);
        vec6AddScaled(rhs, j, f);
    }
}
