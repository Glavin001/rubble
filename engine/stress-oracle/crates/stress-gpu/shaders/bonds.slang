// The bond records every kernel reads (world.slang, split.slang): a bond's static data in
// its cluster's body frame, and its dynamic state.

import joint;

// Static data of a bond (body frame of its cluster).
public struct BondStatic
{
    public float4 t1;
    public float4 t2;
    public float4 normal;
    public float4 ra;       // centroid - centre of a
    public float4 rb;       // centroid - centre of b
    public float4 centroid; // body frame
    public float4 c_lin;    // dashpots
    public float4 c_ang;
    public JointBond law;   // ids: x material, y chunk a, z chunk b, w island
};

// Dynamic data of a bond.
public struct BondDyn
{
    public JointState js;
    public float4 force_lin;  // last force incl. dashpots; w: stored energy
    public float4 force_ang;  // w: largest utilization seen
    public float4 sums;       // dissipated, overshoot, damped (compensated with `comps`)
    public float4 comps;
    public uint4 events;      // absolute substep of: crack, creak, break (0: none); w: mode at crack
};
