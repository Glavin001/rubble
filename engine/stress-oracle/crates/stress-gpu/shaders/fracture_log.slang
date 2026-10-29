// The GPU fracture's change log (fracture.slang's frame, after its split): what a frame changed,
// built on the GPU, so a host learns of breaks and splits by reading a small log instead of
// every chunk's body, bond and frame (engine-world `threshold.rs`).
//
//   log_bonds    one thread per bond: a bond intact at the last log and broken now;
//   log_chunks   one thread per chunk: a chunk whose body changed;
//   log_frames   one thread per body slot: a slot whose frame (centre of mass, principal
//                rotation) changed.
//
// Each compares with its copy of the state at the last log (`track`) and appends an entry
// stamped with the engine step (`step`). Sized by fracture.slang's gate (`fr_gate`), they run
// only on frames that broke a bond: nothing else changes these.
//
// The log is a ring the host acknowledges: each section's cursor only grows, entry i lives at
// i mod capacity, and the host writes back how far it has read (`ack`). Entries are kept until
// then, so a host that skipped frames' readbacks still finds every change, in step order. More
// unread entries than a section holds sets `overflow`: the host then reads the state in full
// once and resumes. Within a step entries are appended in no particular order (the host sorts).

struct LogParams
{
    uint chunk_count;
    uint bond_count;
    uint body_count;
    uint step;       // the engine's next step (fracture.slang's `step`): the change's step
    uint w_alive;    // fracture.slang `work`: the bonds' alive flags
    uint cap_bonds;
    uint cap_chunks;
    uint cap_frames;
};

[[vk::binding(0, 0)]] ConstantBuffer<LogParams> params;
// fracture.slang's `work` (each chunk's body, then the bonds' alive flags) and `frames` (per
// body slot 2 float4).
[[vk::binding(1, 0)]] StructuredBuffer<uint> work;
[[vk::binding(2, 0)]] StructuredBuffer<float4> frames;
// The state at the last log: per bond alive, per chunk body, per slot its frame (8 words).
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> track;
// Header: cursors (bonds, chunks, frames), overflow; then the host's acks (bonds, chunks,
// frames); then the sections: bonds (bond, step), chunks (chunk, body, step), frames
// (slot, step, 8 words).
[[vk::binding(4, 0)]] RWStructuredBuffer<Atomic<uint>> log;

static const uint HEADER = 8;
static const uint BOND_WORDS = 2;
static const uint CHUNK_WORDS = 3;
static const uint FRAME_WORDS = 10;

uint t_body() { return params.bond_count; }
uint t_frame() { return params.bond_count + params.chunk_count; }
uint l_bonds() { return HEADER; }
uint l_chunks() { return l_bonds() + BOND_WORDS * params.cap_bonds; }
uint l_frames() { return l_chunks() + CHUNK_WORDS * params.cap_chunks; }

// The next entry of section `s` (0 bonds, 1 chunks, 2 frames) of capacity `cap`, or none
// (overflow) when the host has not read far enough.
bool claim(uint s, uint cap, out uint at)
{
    uint i = log[s].add(1);
    uint acked = log[4 + s].load();
    at = i % cap;
    if (i - acked >= cap)
    {
        log[3].store(1);
        return false;
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void log_bonds(uint3 id: SV_DispatchThreadID)
{
    uint j = id.x;
    if (j >= params.bond_count)
        return;
    uint alive = work[params.w_alive + j] != 0 ? 1u : 0u;
    if (track[j] == alive)
        return;
    track[j] = alive;
    uint at;
    if (alive == 0 && claim(0, params.cap_bonds, at))
    {
        log[l_bonds() + BOND_WORDS * at].store(j);
        log[l_bonds() + BOND_WORDS * at + 1].store(params.step);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void log_chunks(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= params.chunk_count)
        return;
    uint body = work[c];
    if (track[t_body() + c] == body)
        return;
    track[t_body() + c] = body;
    uint at;
    if (claim(1, params.cap_chunks, at))
    {
        log[l_chunks() + CHUNK_WORDS * at].store(c);
        log[l_chunks() + CHUNK_WORDS * at + 1].store(body);
        log[l_chunks() + CHUNK_WORDS * at + 2].store(params.step);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void log_frames(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= params.body_count)
        return;
    uint4 f0 = asuint(frames[2 * b]);
    uint4 f1 = asuint(frames[2 * b + 1]);
    uint t = t_frame() + 8 * b;
    bool same = all(uint4(track[t], track[t + 1], track[t + 2], track[t + 3]) == f0) && all(uint4(track[t + 4], track[t + 5], track[t + 6], track[t + 7]) == f1);
    if (same)
        return;
    for (uint w = 0; w < 4; w++)
    {
        track[t + w] = f0[w];
        track[t + 4 + w] = f1[w];
    }
    uint at;
    if (claim(2, params.cap_frames, at))
    {
        uint e = l_frames() + FRAME_WORDS * at;
        log[e].store(b);
        log[e + 1].store(params.step);
        for (uint w = 0; w < 4; w++)
        {
            log[e + 2 + w].store(f0[w]);
            log[e + 6 + w].store(f1[w]);
        }
    }
}
