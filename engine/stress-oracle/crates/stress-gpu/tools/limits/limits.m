// GPU limits probes for Apple GPUs (Metal), and per-kernel driver compile statistics.
//
// Each probe isolates one variable and sweeps it to find the knee. GPU time comes from
// command-buffer timestamps (GPUEndTime - GPUStartTime); per-operation costs are the slope
// between two loop counts, so dispatch cost cancels. The GPU is warmed before each probe
// and every point is the minimum (and median) of N repeats: the GPU is shared with other
// processes, and the minimum is the uncontended cost.
//
// Kernels are generated here, written to a work directory and compiled offline with
// `xcrun metal` (precise math, -fno-fast-math, as build.rs does; `fast` libraries use
// -ffast-math), then loaded as metallibs. Driver statistics (instruction count, temporary
// registers, spilled bytes) come from a PRIVATE Metal API: pipeline option 0x40000 makes
// MTLComputePipelineReflection answer `performanceStatistics`. When that is unavailable
// the statistics are reported as -1 and everything else still runs. Machine-code size comes
// from the offline AGX translator (tools/limits/codesize.py, also private tooling).
//
// usage:
//   limits probe [--probes a,b,...] [--repeats N] [--work DIR] [--codesize PATH]
//                [--coresidency] [--watchdog MAX_SECONDS]
//       probes: icache,regs,stack,ops,dispatch,roundtrip,bandwidth (default: all of these)
//       --coresidency  OPT-IN bounded-spin co-residency probe (blocks the GPU for a few
//                      ms per point; never run by default)
//       --watchdog S   OPT-IN command-buffer watchdog probe: single-threadgroup kernels of
//                      0.25 s, 0.5 s, ... up to S seconds until the driver kills one. It can
//                      stall the desktop for S seconds; never run by default.
//       TOML goes to stdout, progress to stderr.
//   limits stats <lib.metallib> [function ...]
//       one TSV line per kernel: name instructions temp_registers spilled_bytes
//       threadgroup_memory max_threads_per_threadgroup alu branch device_loads
//   limits device
//
// LIMITS_NO_PRIVATE_STATS=1 skips the private option (driver statistics then read -1).
//       device name and GPU architecture (e.g. applegpu_g15s)
//
// Build: tools/limits/build.sh  (clang, Metal + Foundation, no other dependencies)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PRIVATE_STATS_OPTION ((MTLPipelineOption)0x40000)

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLBuffer> buf, aux;
static int REPS = 7;
static NSString *workDir;
static NSString *codesizePath;
static NSString *archName = @"unknown";
static double tickNs;

static void progress(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void progress(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); fflush(stderr);
}

static inline uint64_t now(void) { return mach_absolute_time(); }
static inline double ns(uint64_t t) { return (double)t * tickNs; }

// ---------------------------------------------------------------- statistics helpers

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
typedef struct { double min, med, p90, max; } Stat;
static Stat stat(const double *v, int n) {
    double *c = malloc(sizeof(double) * n);
    memcpy(c, v, sizeof(double) * n);
    qsort(c, n, sizeof(double), cmpd);
    Stat s = { c[0], c[n / 2], c[(int)(n * 0.9) < n ? (int)(n * 0.9) : n - 1], c[n - 1] };
    free(c);
    return s;
}
static double median(const double *v, int n) { return stat(v, n).med; }

// First index k where y[k] and y[k+1] (when it exists) both exceed factor x baseline,
// baseline = median of the first min(4, n) points. -1 if none.
static int knee(const double *y, int n, double factor) {
    if (n < 2) return -1;
    double base = median(y, n < 4 ? n : 4);
    for (int k = 1; k < n; k++) {
        if (y[k] > factor * base && (k + 1 >= n || y[k + 1] > factor * base)) return k;
    }
    return -1;
}

// ---------------------------------------------------------------- compilation

typedef struct {
    id<MTLComputePipelineState> ps;
    int instrs, tregs, spill, alu, branch, devld, tgmem;
    NSUInteger maxThreads;
    long codeBytes;
    char name[96];
} Pipe;

static BOOL shell(NSString *cmd) {
    int rc = system(cmd.UTF8String);
    if (rc != 0) progress("command failed (%d): %s\n", rc, cmd.UTF8String);
    return rc == 0;
}

static NSString *libPath(NSString *name, BOOL fast) {
    return [workDir stringByAppendingPathComponent:[NSString stringWithFormat:@"%@%s.metallib", name, fast ? ".fast" : ""]];
}

// Offline compile exactly like build.rs: xcrun metal [-fno-fast-math|-ffast-math] -c, metallib.
static id<MTLLibrary> buildLib(NSString *name, NSString *src, BOOL fast) {
    NSString *metal = [workDir stringByAppendingPathComponent:[name stringByAppendingString:@".metal"]];
    NSString *air = [workDir stringByAppendingPathComponent:[NSString stringWithFormat:@"%@%s.air", name, fast ? ".fast" : ""]];
    NSString *lib = libPath(name, fast);
    [src writeToFile:metal atomically:YES encoding:NSUTF8StringEncoding error:nil];
    NSString *cmd = [NSString stringWithFormat:@"xcrun -sdk macosx metal %s -c '%@' -o '%@' && xcrun -sdk macosx metallib '%@' -o '%@'",
                     fast ? "-ffast-math" : "-fno-fast-math", metal, air, air, lib];
    if (!shell(cmd)) exit(2);
    NSError *err = nil;
    id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:lib] error:&err];
    if (!l) { progress("load %s: %s\n", lib.UTF8String, err.description.UTF8String); exit(2); }
    return l;
}

static int statInt(NSDictionary *st, NSString *key) {
    id v = st[key];
    return v ? [v intValue] : -1;
}

static Pipe makePipeFn(id<MTLFunction> fn) {
    Pipe p = { nil };
    strncpy(p.name, fn.name.UTF8String, sizeof p.name - 1);
    NSError *err = nil;
    MTLComputePipelineReflection *refl = nil;
    // LIMITS_NO_PRIVATE_STATS=1 simulates the private option being gone (tests the fallback).
    MTLPipelineOption opt = getenv("LIMITS_NO_PRIVATE_STATS") ? MTLPipelineOptionNone : PRIVATE_STATS_OPTION;
    p.ps = [dev newComputePipelineStateWithFunction:fn options:opt reflection:&refl error:&err];
    if (!p.ps) p.ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p.ps) { progress("pipeline %s: %s\n", p.name, err.description.UTF8String); exit(2); }
    NSDictionary *st = nil;
    @try {
        if (refl && [refl respondsToSelector:NSSelectorFromString(@"performanceStatistics")])
            st = [(id)refl valueForKey:@"performanceStatistics"];
    } @catch (NSException *e) { st = nil; }
    if (![st isKindOfClass:[NSDictionary class]]) st = nil;
    p.instrs = statInt(st, @"Instruction count");
    p.tregs = statInt(st, @"Temporary register count");
    p.spill = statInt(st, @"Spilled bytes");
    p.alu = statInt(st, @"ALU instruction count");
    p.branch = statInt(st, @"Branch instruction count");
    p.devld = statInt(st, @"Device load instruction count");
    p.tgmem = statInt(st, @"Threadgroup memory");
    p.maxThreads = p.ps.maxTotalThreadsPerThreadgroup;
    p.codeBytes = -1;
    return p;
}

static Pipe makePipe(id<MTLLibrary> lib, NSString *name) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { progress("no function %s\n", name.UTF8String); exit(2); }
    return makePipeFn(fn);
}

// Machine-code bytes (__TEXT,__text of each function's AGX binary) via codesize.py.
static void codeSizes(NSString *libName, BOOL fast, Pipe *pipes, int n) {
    if (!codesizePath || ![[NSFileManager defaultManager] fileExistsAtPath:codesizePath]) return;
    NSMutableString *cmd = [NSMutableString stringWithFormat:@"python3 '%@' --arch %@ '%@'", codesizePath, archName, libPath(libName, fast)];
    for (int i = 0; i < n; i++) [cmd appendFormat:@" %s", pipes[i].name];
    [cmd appendString:@" 2>/dev/null"];
    FILE *f = popen(cmd.UTF8String, "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char fn[256]; long bytes;
        if (sscanf(line, "%255s %ld", fn, &bytes) == 2)
            for (int i = 0; i < n; i++) if (!strcmp(fn, pipes[i].name)) pipes[i].codeBytes = bytes;
    }
    pclose(f);
}

// ---------------------------------------------------------------- running

static double runOnce(Pipe *p, uint32_t n, NSUInteger groups, NSUInteger threads) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p->ps];
    [e setBuffer:buf offset:0 atIndex:0];
    [e setBytes:&n length:4 atIndex:1];
    [e setBuffer:aux offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        progress("command buffer failed: %s\n", cb.error.description.UTF8String);
        return -1;
    }
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
}

static Pipe warmPipe;
static BOOL haveWarm;
static uint32_t warmIters = 100000;  // ~0.8 ms per warm-up run: short runs leave the GPU idle between submits and the clock stays low
static NSUInteger warmGroups = 1;

// Keep one core busy so the GPU clock is up before measuring.
static void warm(double ms) {
    if (!haveWarm) return;
    uint64_t t0 = now();
    while (ns(now() - t0) < ms * 1e6) runOnce(&warmPipe, warmIters, warmGroups, 256);
}

typedef struct { double min, med; uint32_t n1, n2; } Slope;

// Per-loop-iteration GPU time: slope between two loop counts, n2 chosen so one run takes
// about targetUs. min = (min t2 - min t1)/(n2 - n1), med likewise with medians.
static Slope slope(Pipe *p, NSUInteger groups, NSUInteger threads, double targetUs) {
    warm(30);  // the clock drops within milliseconds of idling (pipeline creation, host work)
    runOnce(p, 8, groups, threads);
    double a = runOnce(p, 16, groups, threads), b = runOnce(p, 144, groups, threads);
    for (int i = 0; i < 2; i++) { a = fmin(a, runOnce(p, 16, groups, threads)); b = fmin(b, runOnce(p, 144, groups, threads)); }
    double per = (b - a) / 128.0;
    if (per < 0.5) per = 0.5;
    double n2d = targetUs * 1e3 / per;
    if (n2d < 64) n2d = 64;
    if (n2d > 2e7) n2d = 2e7;
    uint32_t n2 = (uint32_t)n2d, n1 = n2 / 4;
    double t1[64], t2[64];
    int r = REPS > 64 ? 64 : REPS;
    for (int i = 0; i < r; i++) { t1[i] = runOnce(p, n1, groups, threads); t2[i] = runOnce(p, n2, groups, threads); }
    Stat s1 = stat(t1, r), s2 = stat(t2, r);
    Slope s = { (s2.min - s1.min) / (n2 - n1), (s2.med - s1.med) / (n2 - n1), n1, n2 };
    return s;
}

// ---------------------------------------------------------------- kernel sources

static NSString *const HEADER = @"#include <metal_stdlib>\nusing namespace metal;\n";
static NSString *const SIG =
    @"device uint* buf [[buffer(0)]], constant uint& n [[buffer(1)]], device uint* aux [[buffer(2)]], "
     "uint tid [[thread_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]";

static void aluStmt(NSMutableString *s, int j) {
    int c = j % 4, op = j % 3;
    if (op == 0) [s appendFormat:@"    x%d = fma(x%d, a, %.7ff);\n", c, c, 1.0 + (j % 97) / 128.0];
    else if (op == 1) [s appendFormat:@"    x%d = x%d * a - %.7ff;\n", c, c, (j % 89) / 64.0];
    else [s appendFormat:@"    x%d = fma(x%d, %.7ff, a);\n", c, c, 0.5 + (j % 83) / 256.0];
}

// Straight-line body of N statements in 4 independent chains (distinct constants so nothing
// folds); mixed adds a cached device load every 8th and a uniform data-dependent branch
// every 16th statement.
static void codeKernel(NSMutableString *s, NSString *name, int N, BOOL mixed) {
    [s appendFormat:@"kernel void %@(%@) {\n", name, SIG];
    [s appendString:@"  float x0 = float(buf[tid]), x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;\n"
                     "  float a = float(buf[1]) * 1e-6f + 0.999f;\n"
                     "  for (uint i = 0; i < n; i++) {\n"];
    for (int j = 0; j < N; j++) {
        int c = j % 4;
        if (mixed && j % 8 == 7) {
            [s appendFormat:@"    x%d += as_type<float>((buf[(i * 7u + %du) & 1023u] & 0x007fffffu) | 0x3f800000u);\n", c, j];
        } else if (mixed && j % 16 == 3) {
            int d = (c + 1) % 4;
            [s appendFormat:@"    if ((buf[(i + %du) & 1023u] & 1u) != 0u) { x%d = fma(x%d, a, 0.25f); x%d = x%d * a; } "
                             "else { x%d = x%d - a; x%d = x%d + 0.5f; }\n", j, c, c, d, d, c, c, d, d];
        } else {
            aluStmt(s, j);
        }
    }
    [s appendString:@"  }\n  aux[tg * 1024u + tid] = as_type<uint>(x0 + x1 + x2 + x3);\n}\n"];
}

// K floats live across the loop, every one updated each iteration.
static void regsKernel(NSMutableString *s, NSString *name, int K) {
    [s appendFormat:@"kernel void %@(%@) {\n", name, SIG];
    for (int k = 0; k < K; k++) [s appendFormat:@"  float v%d = float(buf[(tid + %du) & 4095u]);\n", k, k];
    [s appendString:@"  float a = float(buf[1]) * 1e-6f + 0.999f;\n  for (uint i = 0; i < n; i++) {\n"];
    for (int k = 0; k < K; k++) [s appendFormat:@"    v%d = fma(v%d, a, v%d);\n", k, k, (k + 1) % K];
    [s appendString:@"  }\n  float s = 0;\n"];
    for (int k = 0; k < K; k++) [s appendFormat:@"  s += v%d;\n", k];
    [s appendString:@"  aux[tg * 1024u + tid] = as_type<uint>(s);\n}\n"];
}

// A per-thread array of S words visited in a full-period LCG order (dependent loads).
static void stackKernel(NSMutableString *s, NSString *name, int S) {
    [s appendFormat:@"kernel void %@(%@) {\n", name, SIG];
    [s appendFormat:@"  uint arr[%d];\n"
                     "  for (uint i = 0; i < %du; i++) arr[i] = ((i * 13u + 1u) + buf[8192u + (i & 1023u)]) %% %du;\n"
                     "  uint p = tid %% %du;\n"
                     "  for (uint i = 0; i < n; i++) { p = arr[p]; }\n"
                     "  aux[tg * 1024u + tid] = p;\n}\n", S, S, S, S];
}

static NSString *opsSource(void) {
    NSMutableString *s = [NSMutableString stringWithString:HEADER];
    // name, init, statement, repeats per iteration
    struct { const char *name, *init, *stmt; int u; } ops[] = {
        { "op_loop_iter", "uint x = buf[tid];", "x = x * 1664525u + 1013904223u;", 1 },
        { "op_fma", "float x = float(buf[tid]); float a = float(buf[1]) * 1e-6f + 0.999f; float b = float(buf[2]) * 1e-9f;", "x = fma(x, a, b);", 16 },
        { "op_div_precise", "float x = float(buf[tid]) + 1.0f;", "x = 1.5f / (x + 0.5f);", 16 },
        { "op_div_fast", "float x = float(buf[tid]) + 1.0f;", "x = fast::divide(1.5f, x + 0.5f);", 16 },
        { "op_sqrt_precise", "float x = float(buf[tid]) + 1.0f;", "x = sqrt(x + 0.5f);", 16 },
        { "op_sqrt_fast", "float x = float(buf[tid]) + 1.0f;", "x = fast::sqrt(x + 0.5f);", 16 },
        { "op_pow_precise", "float x = float(buf[tid]) + 1.0f;", "x = precise::pow(x + 0.5f, 0.75f);", 16 },
        { "op_pow_fast", "float x = float(buf[tid]) + 1.0f;", "x = fast::pow(x + 0.5f, 0.75f);", 16 },
        { "op_exp_precise", "float x = float(buf[tid]) * 1e-3f;", "x = precise::exp(-x);", 16 },
        { "op_exp_fast", "float x = float(buf[tid]) * 1e-3f;", "x = fast::exp(-x);", 16 },
        { "op_simd_sum", "float x = float(buf[tid]);", "x = simd_sum(x) * 0.03125f + 0.25f;", 8 },
        { "op_load_device", "uint p = tid;", "p = buf[p];", 8 },
        { "op_load_stack",
          "uint arr[64]; for (uint i = 0; i < 64; i++) arr[i] = buf[4096u + ((i * 7u + tid) & 63u)] & 63u; uint p = tid & 63u;",
          "p = arr[p];", 8 },
    };
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        [s appendFormat:@"kernel void %s(%@) {\n  %s\n  for (uint i = 0; i < n; i++) {\n", ops[i].name, SIG, ops[i].init];
        for (int u = 0; u < ops[i].u; u++) [s appendFormat:@"    %s\n", ops[i].stmt];
        BOOL isUint = strstr(ops[i].init, "uint p") || strstr(ops[i].init, "uint x");
        [s appendFormat:@"  }\n  aux[tg * 1024u + tid] = %s;\n}\n",
                        isUint ? (strstr(ops[i].init, "uint p") ? "p" : "x") : "as_type<uint>(x)"];
    }
    [s appendFormat:@"kernel void op_load_threadgroup(%@) {\n"
                     "  threadgroup uint t[4096];\n"
                     "  for (uint i = tid; i < 4096u; i += 256u) t[i] = buf[i];\n"
                     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                     "  uint p = tid;\n  for (uint i = 0; i < n; i++) {\n", SIG];
    for (int u = 0; u < 8; u++) [s appendString:@"    p = t[p];\n"];
    [s appendString:@"  }\n  aux[tg * 1024u + tid] = p;\n}\n"];
    [s appendString:@"kernel void op_load_constant(constant uint* cb [[buffer(0)]], constant uint& n [[buffer(1)]], device uint* aux [[buffer(2)]],"
                     " uint tid [[thread_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {\n"
                     "  uint p = tid;\n  for (uint i = 0; i < n; i++) {\n"];
    for (int u = 0; u < 8; u++) [s appendString:@"    p = cb[p];\n"];
    [s appendString:@"  }\n  aux[tg * 1024u + tid] = p;\n}\n"];
    [s appendFormat:@"kernel void op_barrier_threadgroup(%@) {\n"
                     "  threadgroup float sh[1024];\n  float x = float(buf[tid]);\n"
                     "  for (uint i = 0; i < n; i++) {\n"
                     "    sh[tid] = x; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                     "    x = sh[(tid + 1u) & 255u] * 0.5f + 0.25f; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                     "  }\n  aux[tg * 1024u + tid] = as_type<uint>(x);\n}\n", SIG];
    [s appendFormat:@"kernel void op_barrier_device(%@) {\n"
                     "  device float* f = (device float*)(aux + 65536u);\n  float x = float(buf[tid]);\n"
                     "  for (uint i = 0; i < n; i++) {\n"
                     "    f[tid] = x; threadgroup_barrier(mem_flags::mem_device);\n"
                     "    x = f[(tid + 1u) & 255u] * 0.5f + 0.25f; threadgroup_barrier(mem_flags::mem_device);\n"
                     "  }\n  aux[tg * 1024u + tid] = as_type<uint>(x);\n}\n", SIG];
    return s;
}

static NSString *miscSource(void) {
    NSMutableString *s = [NSMutableString stringWithString:HEADER];
    [s appendFormat:@"kernel void warm_fma(%@) {\n  float x = float(buf[tid]); float a = float(buf[1]) * 1e-6f + 0.999f;\n"
                     "  for (uint i = 0; i < n; i++) { x = fma(x, a, 1e-7f); x = fma(x, a, 1e-7f); x = fma(x, a, 1e-7f); x = fma(x, a, 1e-7f); }\n"
                     "  aux[tg * 1024u + tid] = as_type<uint>(x);\n}\n", SIG];
    [s appendString:@"kernel void tiny(device uint* p [[buffer(0)]], uint t [[thread_position_in_grid]]) { if (t == 0) p[0] += 1u; }\n"];
    [s appendString:@"kernel void copy4(device const float4* src [[buffer(0)]], device float4* dst [[buffer(2)]], uint i [[thread_position_in_grid]]) { dst[i] = src[i]; }\n"];
    // Co-residency: thread 0 of each group registers, then waits for all groups for at
    // most `limit` polls (never unbounded), and records the arrivals it saw.
    [s appendString:@"kernel void coresident(device atomic_uint* ctr [[buffer(0)]], constant uint& limit [[buffer(1)]], device uint* seen [[buffer(2)]],"
                     " uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {\n"
                     "  if (tid == 0) {\n"
                     "    uint total = atomic_load_explicit(&ctr[1], memory_order_relaxed);\n"
                     "    uint c = atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed) + 1u;\n"
                     "    uint polls = 0;\n"
                     "    while (c < total && polls < limit) { c = atomic_load_explicit(&ctr[0], memory_order_relaxed); polls++; }\n"
                     "    seen[g] = c;\n"
                     "  }\n"
                     "  threadgroup_barrier(mem_flags::mem_none);\n}\n"];
    return s;
}

// ---------------------------------------------------------------- TOML output

static void tomlPoints(const char *key, const char **cols, int ncols, double *rows, int nrows) {
    printf("%s_columns = [", key);
    for (int c = 0; c < ncols; c++) printf("%s\"%s\"", c ? ", " : "", cols[c]);
    printf("]\n%s = [\n", key);
    for (int r = 0; r < nrows; r++) {
        printf("  [");
        for (int c = 0; c < ncols; c++) {
            double v = rows[r * ncols + c];
            if (v == (double)(long long)v && fabs(v) < 1e15) printf("%s%lld", c ? ", " : "", (long long)v);
            else printf("%s%.4g", c ? ", " : "", v);
        }
        printf("],\n");
    }
    printf("]\n");
}

// ---------------------------------------------------------------- probes

static void probeICache(void) {
    progress("[icache] generating and compiling\n");
    const int nAlu = (4096 - 512) / 64 + 1, nMix = (1600 - 192) / 64 + 1;  // mixed statements are ~2.5 instructions each
    NSMutableString *s = [NSMutableString stringWithString:HEADER];
    for (int i = 0; i < nAlu; i++) codeKernel(s, [NSString stringWithFormat:@"icache_alu_%05d", 512 + 64 * i], 512 + 64 * i, NO);
    for (int i = 0; i < nMix; i++) codeKernel(s, [NSString stringWithFormat:@"icache_mixed_%05d", 192 + 64 * i], 192 + 64 * i, YES);
    id<MTLLibrary> lib = buildLib(@"icache", s, NO);
    Pipe *pa = calloc(nAlu, sizeof(Pipe)), *pm = calloc(nMix, sizeof(Pipe));
    for (int i = 0; i < nAlu; i++) pa[i] = makePipe(lib, [NSString stringWithFormat:@"icache_alu_%05d", 512 + 64 * i]);
    for (int i = 0; i < nMix; i++) pm[i] = makePipe(lib, [NSString stringWithFormat:@"icache_mixed_%05d", 192 + 64 * i]);
    codeSizes(@"icache", NO, pa, nAlu);
    codeSizes(@"icache", NO, pm, nMix);
    printf("\n[instruction_cache]\n");
    printf("method = \"straight-line loop body of N statements (4 independent FMA/MUL chains; mixed: + a cached device load every 8th and a uniform branch every 16th), "
           "1 threadgroup x 256 threads, ns per driver instruction = loop-count slope / driver instruction count; knee = [last point below, first point above] 1.1x the median of the first 4 (sustained for the next point too)\"\n");
    printf("repeats = %d\n", REPS);
    for (int pass = 0; pass < 2; pass++) {
        Pipe *p = pass ? pm : pa;
        int n = pass ? nMix : nAlu, step = 64, first = pass ? 192 : 512;
        const char *tag = pass ? "mixed" : "alu";
        double *rows = calloc(n * 6, sizeof(double)), *y = calloc(n, sizeof(double));
        warm(300);
        for (int i = 0; i < n; i++) {
            Slope sl = slope(&p[i], 1, 256, 1500);
            double instr = p[i].instrs > 0 ? p[i].instrs : first + step * i;
            rows[i * 6 + 0] = first + step * i;
            rows[i * 6 + 1] = p[i].instrs;
            rows[i * 6 + 2] = p[i].codeBytes;
            rows[i * 6 + 3] = sl.min / instr;
            rows[i * 6 + 4] = sl.med / instr;
            rows[i * 6 + 5] = p[i].branch;
            y[i] = sl.min / instr;
            progress("[icache] %-20s instrs %5d bytes %6ld  %.3f ns/instr (median %.3f)\n", p[i].name, p[i].instrs, p[i].codeBytes, sl.min / instr, sl.med / instr);
        }
        int k = knee(y, n, 1.1);
        if (k > 0) {
            double below[128], above[128]; int nb = 0, na = 0;
            for (int i = 0; i < k; i++) below[nb++] = y[i];
            for (int i = k; i < n; i++) above[na++] = y[i];
            printf("%s_knee_instructions = [%.0f, %.0f]\n", tag, rows[(k - 1) * 6 + 1], rows[k * 6 + 1]);
            printf("%s_knee_code_bytes = [%.0f, %.0f]\n", tag, rows[(k - 1) * 6 + 2], rows[k * 6 + 2]);
            printf("%s_ns_per_instr_below = %.3f\n%s_ns_per_instr_above = %.3f\n", tag, median(below, nb), tag, median(above, na));
        } else {
            printf("%s_knee_instructions = \"none in sweep\"\n", tag);
        }
        const char *cols[] = { "statements", "driver_instructions", "code_bytes", "ns_per_instr_min", "ns_per_instr_median", "branches" };
        char key[64]; snprintf(key, sizeof key, "%s_points", tag);
        tomlPoints(key, cols, 6, rows, n);
        free(rows); free(y);
    }
    fflush(stdout);
    free(pa); free(pm);
}

static void probeRegs(void) {
    progress("[regs] generating and compiling\n");
    int Ks[64], n = 0;
    for (int K = 16; K <= 256; K += 8) Ks[n++] = K;
    NSMutableString *s = [NSMutableString stringWithString:HEADER];
    for (int i = 0; i < n; i++) regsKernel(s, [NSString stringWithFormat:@"regs_%03d", Ks[i]], Ks[i]);
    id<MTLLibrary> lib = buildLib(@"regs", s, NO);
    Pipe *p = calloc(n, sizeof(Pipe));
    for (int i = 0; i < n; i++) p[i] = makePipe(lib, [NSString stringWithFormat:@"regs_%03d", Ks[i]]);
    double *rows = calloc(n * 7, sizeof(double)), *y = calloc(n, sizeof(double));
    warm(300);
    int firstSpill = -1, firstOcc = -1;
    for (int i = 0; i < n; i++) {
        NSUInteger thr = p[i].maxThreads < 256 ? p[i].maxThreads : 256;
        Slope sl = slope(&p[i], 1, thr, 1500);
        rows[i * 7 + 0] = Ks[i]; rows[i * 7 + 1] = p[i].tregs; rows[i * 7 + 2] = p[i].spill;
        rows[i * 7 + 3] = p[i].maxThreads; rows[i * 7 + 4] = thr;
        rows[i * 7 + 5] = sl.min / Ks[i]; rows[i * 7 + 6] = sl.med / Ks[i];
        y[i] = sl.min / Ks[i];
        if (firstSpill < 0 && p[i].spill > 0) firstSpill = i;
        if (firstOcc < 0 && p[i].maxThreads < 1024) firstOcc = i;
        progress("[regs] K=%3d tregs %3d spill %5d maxThreads %4lu  %.3f ns/update\n", Ks[i], p[i].tregs, p[i].spill, (unsigned long)p[i].maxThreads, y[i]);
    }
    int k = knee(y, n, 1.5);
    printf("\n[live_registers]\n");
    printf("method = \"K floats live across a dynamic loop, each updated by a dependent FMA per iteration; 1 threadgroup of min(256, maxTotalThreadsPerThreadgroup); "
           "ns per update = slope / K; spill and register counts from driver statistics (private option 0x40000)\"\n");
    printf("repeats = %d\n", REPS);
    if (firstSpill >= 0) printf("first_spill_live_floats = %d\nfirst_spill_temp_registers = %d\n", Ks[firstSpill], p[firstSpill].tregs);
    else printf("first_spill_live_floats = \"none in sweep\"\n");
    if (firstOcc >= 0) printf("first_reduced_max_threads_live_floats = %d\nfirst_reduced_max_threads = %lu\nfirst_reduced_max_threads_temp_registers = %d\n",
                              Ks[firstOcc], (unsigned long)p[firstOcc].maxThreads, p[firstOcc].tregs);
    int maxRegs = 0;
    for (int i = 0; i < n; i++) if (p[i].tregs > maxRegs) maxRegs = p[i].tregs;
    printf("max_temp_registers_seen = %d\n", maxRegs);
    if (k > 0) printf("time_knee_live_floats = [%d, %d]\ntime_knee_ns_per_update = [%.3f, %.3f]\n", Ks[k - 1], Ks[k], y[k - 1], y[k]);
    else printf("time_knee_live_floats = \"none in sweep\"\n");
    const char *cols[] = { "live_floats", "temp_registers", "spilled_bytes", "max_threads_per_threadgroup", "threads_dispatched", "ns_per_update_min", "ns_per_update_median" };
    tomlPoints("points", cols, 7, rows, n);
    fflush(stdout);
    free(rows); free(y); free(p);
}

static void probeStack(void) {
    progress("[stack] generating and compiling\n");
    int Ss[] = { 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 320, 384, 512, 768, 1024, 2048 };
    int n = sizeof Ss / sizeof Ss[0];
    NSMutableString *s = [NSMutableString stringWithString:HEADER];
    for (int i = 0; i < n; i++) stackKernel(s, [NSString stringWithFormat:@"stack_%04d", Ss[i]], Ss[i]);
    id<MTLLibrary> lib = buildLib(@"stack", s, NO);
    Pipe *p = calloc(n, sizeof(Pipe));
    for (int i = 0; i < n; i++) p[i] = makePipe(lib, [NSString stringWithFormat:@"stack_%04d", Ss[i]]);
    printf("\n[stack]\n");
    printf("method = \"per-thread uint array of S words, filled at run time, then a dependent chase through all S entries in a full-period LCG order; "
           "ns per dependent stack load = loop-count slope; footprint per threadgroup = driver spilled bytes per thread (the array plus a few bytes; 4*S when not spilled) x threads, the same measure kernel_budget.sh reports; sweep 1: S at 256 threads; sweep 2: S=256 (1 KiB/thread) with threads varied; "
           "knee = first point (and the next) above 1.5x the median of the first 4\"\n");
    printf("repeats = %d\n", REPS);
    // sweep 1: array size at 256 threads
    double *rows = calloc(n * 7, sizeof(double)), *y = calloc(n, sizeof(double));
    warm(300);
    for (int i = 0; i < n; i++) {
        NSUInteger thr = p[i].maxThreads < 256 ? p[i].maxThreads : 256;
        Slope sl = slope(&p[i], 1, thr, 1500);
        rows[i * 7 + 0] = Ss[i] * 4; rows[i * 7 + 1] = thr; rows[i * 7 + 2] = (p[i].spill > 0 ? p[i].spill : Ss[i] * 4.0) * thr;
        rows[i * 7 + 3] = p[i].spill; rows[i * 7 + 4] = p[i].tregs;
        rows[i * 7 + 5] = sl.min; rows[i * 7 + 6] = sl.med;
        y[i] = sl.min;
        progress("[stack] %5d B/thread x %lu = %7.0f B/tg  spill %5d  %.2f ns/load\n", Ss[i] * 4, (unsigned long)thr, rows[i * 7 + 2], p[i].spill, sl.min);
    }
    int k = knee(y, n, 1.5);
    if (k > 0) printf("array_sweep_knee_array_bytes_per_thread = [%.0f, %.0f]\narray_sweep_knee_bytes_per_threadgroup = [%.0f, %.0f]\narray_sweep_ns_per_load = [%.2f, %.2f]\n",
                      rows[(k - 1) * 7], rows[k * 7], rows[(k - 1) * 7 + 2], rows[k * 7 + 2], y[k - 1], y[k]);
    else printf("array_sweep_knee_bytes_per_threadgroup = \"none in sweep\"\n");
    const char *cols[] = { "array_bytes_per_thread", "threads", "stack_bytes_per_threadgroup", "spilled_bytes", "temp_registers", "ns_per_load_min", "ns_per_load_median" };
    tomlPoints("array_sweep", cols, 7, rows, n);
    free(rows); free(y);
    // sweep 2: S = 256 words, threads varied
    int si = -1;
    for (int i = 0; i < n; i++) if (Ss[i] == 256) si = i;
    int Ts[] = { 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1024 };
    int nt = sizeof Ts / sizeof Ts[0];
    rows = calloc(nt * 7, sizeof(double)); y = calloc(nt, sizeof(double));
    int m = 0;
    warm(100);
    for (int i = 0; i < nt; i++) {
        if ((NSUInteger)Ts[i] > p[si].maxThreads) break;
        Slope sl = slope(&p[si], 1, Ts[i], 1500);
        rows[m * 7 + 0] = 1024; rows[m * 7 + 1] = Ts[i]; rows[m * 7 + 2] = (p[si].spill > 0 ? p[si].spill : 1024.0) * Ts[i];
        rows[m * 7 + 3] = p[si].spill; rows[m * 7 + 4] = p[si].tregs;
        rows[m * 7 + 5] = sl.min; rows[m * 7 + 6] = sl.med;
        y[m++] = sl.min;
        progress("[stack] 1024 B/thread x %4d = %7.0f B/tg  %.2f ns/load\n", Ts[i], rows[(m - 1) * 7 + 2], sl.min);
    }
    k = knee(y, m, 1.5);
    if (k > 0) printf("thread_sweep_knee_bytes_per_threadgroup = [%.0f, %.0f]\nthread_sweep_ns_per_load = [%.2f, %.2f]\n", rows[(k - 1) * 7 + 2], rows[k * 7 + 2], y[k - 1], y[k]);
    else printf("thread_sweep_knee_bytes_per_threadgroup = \"none in sweep\"\n");
    tomlPoints("thread_sweep", cols, 7, rows, m);
    fflush(stdout);
    free(rows); free(y); free(p);
}

static void probeOps(void) {
    progress("[ops] compiling\n");
    NSString *src = opsSource();
    id<MTLLibrary> lib = buildLib(@"ops", src, NO);
    struct { const char *name, *what; int perIter; } ops[] = {
        { "op_loop_iter", "dynamic loop iteration carrying one dependent integer multiply-add", 1 },
        { "op_fma", "dependent FMA", 16 },
        { "op_div_precise", "dependent add + precise divide", 16 },
        { "op_div_fast", "dependent add + fast::divide", 16 },
        { "op_sqrt_precise", "dependent add + precise sqrt", 16 },
        { "op_sqrt_fast", "dependent add + fast::sqrt", 16 },
        { "op_pow_precise", "dependent add + precise::pow", 16 },
        { "op_pow_fast", "dependent add + fast::pow", 16 },
        { "op_exp_precise", "dependent precise::exp", 16 },
        { "op_exp_fast", "dependent fast::exp", 16 },
        { "op_simd_sum", "dependent simd_sum + FMA", 8 },
        { "op_load_threadgroup", "dependent threadgroup load", 8 },
        { "op_load_device", "dependent device load (16 KiB, cached)", 8 },
        { "op_load_constant", "dependent constant-address-space load", 8 },
        { "op_load_stack", "dependent stack (private array) load", 8 },
        { "op_barrier_threadgroup", "threadgroup exchange: store, threadgroup barrier, neighbour load, barrier (per exchange)", 1 },
        { "op_barrier_device", "device exchange: store, device-memory barrier, neighbour load, barrier (per exchange)", 1 },
    };
    int n = sizeof ops / sizeof ops[0];
    printf("\n[ops]\n");
    printf("method = \"1 threadgroup x 256 threads, dynamic loop of U dependent operations; ns_min/ns_median = loop-count slope / U (loop overhead included, amortised over U); "
           "ns_excl_loop = (slope - loop_iter slope) / U, approximate; precise library compiled -fno-fast-math like build.rs\"\n");
    printf("repeats = %d\n", REPS);
    Pipe *pipes = calloc(n, sizeof(Pipe));
    for (int i = 0; i < n; i++) pipes[i] = makePipe(lib, @(ops[i].name));
    warm(300);
    double loopNs = 0;
    for (int i = 0; i < n; i++) {
        Slope sl = slope(&pipes[i], 1, 256, 1500);
        if (i == 0) loopNs = sl.min;
        double mn = sl.min / ops[i].perIter, md = sl.med / ops[i].perIter;
        double ex = i == 0 ? sl.min : (sl.min - loopNs) / ops[i].perIter;
        progress("[ops] %-24s %8.2f ns (median %8.2f, excl. loop %8.2f)  instrs %d\n", ops[i].name, mn, md, ex, pipes[i].instrs);
        printf("%s = { ns_min = %.3f, ns_median = %.3f, ns_excl_loop = %.3f, per_iteration = %d, driver_instructions = %d, what = \"%s\" }\n",
               ops[i].name + 3, mn, md, ex, ops[i].perIter, pipes[i].instrs, ops[i].what);
    }
    // The same FMA chain with one SIMD: equal to the 256-thread figure when the chain is
    // latency-bound, lower when 8 SIMDs contend for issue.
    Slope one = slope(&pipes[1], 1, 32, 1500);
    printf("fma_one_simd = { ns_min = %.3f, ns_median = %.3f, ns_excl_loop = %.3f, per_iteration = 16, what = \"dependent FMA, 1 threadgroup x 32 threads (one SIMD: no issue contention between SIMDs)\" }\n",
           one.min / 16, one.med / 16, (one.min - loopNs) / 16);
    progress("[ops] op_fma one SIMD           %8.2f ns\n", one.min / 16);
    free(pipes);
    fflush(stdout);
}

static id<MTLLibrary> miscLib;

static void probeDispatch(void) {
    progress("[dispatch]\n");
    Pipe tiny = makePipe(miscLib, @"tiny");
    int Ns[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    int n = sizeof Ns / sizeof Ns[0];
    double rows[10 * 4];
    warm(200);
    for (int i = 0; i < n; i++) {
        double t[64]; int r = REPS + 2;
        for (int k = 0; k < r; k++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:tiny.ps];
            [e setBuffer:aux offset:0 atIndex:0];
            for (int d = 0; d < Ns[i]; d++) [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            t[k] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
        }
        Stat s = stat(t + 2, r - 2);
        rows[i * 4 + 0] = Ns[i]; rows[i * 4 + 1] = s.min; rows[i * 4 + 2] = s.med; rows[i * 4 + 3] = s.min / Ns[i];
        progress("[dispatch] serial x%3d: %.1f us GPU (median %.1f)\n", Ns[i], s.min, s.med);
    }
    // least squares slope over N >= 8 (min times)
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
    for (int i = 0; i < n; i++) if (Ns[i] >= 8) { double x = Ns[i], y = rows[i * 4 + 1]; sx += x; sy += y; sxx += x * x; sxy += x * y; m++; }
    double per = (m * sxy - sx * sy) / (m * sxx - sx * sx), icpt = (sy - per * sx) / m;
    // batching comparison at N = 64
    const int N = 64;
    double serial[64], encs[64], cbsGpu[64], cbsWall[64], oneWall[64];
    int r = REPS;
    for (int k = 0; k < r; k++) {
        // one CB, one serial encoder
        uint64_t w0 = now();
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:tiny.ps]; [e setBuffer:aux offset:0 atIndex:0];
        for (int d = 0; d < N; d++) [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        oneWall[k] = ns(now() - w0) / 1e3 / N;
        serial[k] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
        // one CB, one encoder per dispatch
        cb = [queue commandBuffer];
        for (int d = 0; d < N; d++) {
            e = [cb computeCommandEncoder];
            [e setComputePipelineState:tiny.ps]; [e setBuffer:aux offset:0 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        encs[k] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / N;
        // N CBs, one dispatch each, committed back to back, wait on the last
        w0 = now();
        NSMutableArray *all = [NSMutableArray arrayWithCapacity:N];
        for (int d = 0; d < N; d++) {
            id<MTLCommandBuffer> c = [queue commandBuffer];
            e = [c computeCommandEncoder];
            [e setComputePipelineState:tiny.ps]; [e setBuffer:aux offset:0 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding]; [c commit]; [all addObject:c];
        }
        [all.lastObject waitUntilCompleted];
        cbsWall[k] = ns(now() - w0) / 1e3 / N;
        cbsGpu[k] = ([all.lastObject GPUEndTime] - [all.firstObject GPUStartTime]) * 1e6 / N;
    }
    Stat a = stat(serial, r), b = stat(encs, r), c = stat(cbsGpu, r), d = stat(cbsWall, r), o = stat(oneWall, r);
    printf("\n[dispatch]\n");
    printf("method = \"tiny dependent kernel (1 group x 32 threads, read-modify-write of one word) dispatched N times; GPU time from command-buffer timestamps; per-dispatch = least-squares slope over N >= 8 of min times\"\n");
    printf("repeats = %d\n", REPS);
    printf("serial_encoder_us_per_dispatch = %.3f\nserial_encoder_intercept_us = %.2f\n", per, icpt);
    const char *cols[] = { "dispatches", "gpu_us_min", "gpu_us_median", "gpu_us_per_dispatch_min" };
    tomlPoints("serial_encoder", cols, 4, rows, n);
    printf("batch_n = %d\n", N);
    printf("one_cb_serial_encoder_gpu_us_per_dispatch = { min = %.3f, median = %.3f }\n", a.min, a.med);
    printf("one_cb_serial_encoder_wall_us_per_dispatch = { min = %.3f, median = %.3f }\n", o.min, o.med);
    printf("one_cb_encoder_per_dispatch_gpu_us_per_dispatch = { min = %.3f, median = %.3f }\n", b.min, b.med);
    printf("cb_per_dispatch_gpu_span_us_per_dispatch = { min = %.3f, median = %.3f }\n", c.min, c.med);
    printf("cb_per_dispatch_wall_us_per_dispatch = { min = %.3f, median = %.3f }\n", d.min, d.med);
    progress("[dispatch] slope %.2f us; batched serial %.2f, encoder each %.2f, CB each GPU %.2f wall %.2f us\n", per, a.min, b.min, c.min, d.min);
    fflush(stdout);
}

static void probeRoundTrip(void) {
    progress("[roundtrip]\n");
    Pipe tiny = makePipe(miscLib, @"tiny");
    const int R = 200;
    double *v = calloc(R, sizeof(double)), *v2 = calloc(R, sizeof(double)), *v3 = calloc(R, sizeof(double));
    printf("\n[round_trip]\n");
    printf("method = \"commit -> host resumes, back to back (GPU warm), %d samples each; spin waits are bounded at 1 s\"\n", R);
    printf("samples = %d\n", R);
    warm(200);
    for (int variant = 0; variant < 3; variant++) {
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        uint64_t val = 0;
        for (int k = 0; k < R + 10; k++) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            if (variant > 0) {
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:tiny.ps]; [e setBuffer:aux offset:0 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [e endEncoding];
            }
            if (variant == 2) [cb encodeSignalEvent:ev value:++val];
            uint64_t t0 = now();
            [cb commit];
            if (variant == 2) {
                while (ev.signaledValue < val && ns(now() - t0) < 1e9) {}
            } else {
                [cb waitUntilCompleted];
            }
            uint64_t t1 = now();
            if (variant == 2) [cb waitUntilCompleted];
            if (k >= 10) {
                v[k - 10] = ns(t1 - t0) / 1e3;
                v2[k - 10] = (cb.GPUStartTime - (double)t0 * tickNs * 1e-9) * 1e6;
                v3[k - 10] = ((double)t1 * tickNs * 1e-9 - cb.GPUEndTime) * 1e6;
            }
        }
        const char *names[] = { "empty_cb_wait_until_completed", "tiny_dispatch_wait_until_completed", "tiny_dispatch_shared_event_spin" };
        Stat s = stat(v, R), s2 = stat(v2, R), s3 = stat(v3, R);
        printf("%s_us = { min = %.1f, median = %.1f, p90 = %.1f, max = %.1f, commit_to_gpu_start_median = %.1f, gpu_end_to_resume_median = %.1f }\n",
               names[variant], s.min, s.med, s.p90, s.max, s2.med, s3.med);
        progress("[roundtrip] %-36s min %.1f median %.1f p90 %.1f us\n", names[variant], s.min, s.med, s.p90);
    }
    free(v); free(v2); free(v3);
    // clock ramp: idle gap before a fixed kernel
    warm(300);
    Slope sl = slope(&warmPipe, 1, 256, 200);
    uint32_t nfix = (uint32_t)(200e3 / fmax(sl.min, 1e-3));
    double base[32];
    for (int k = 0; k < 20; k++) base[k] = runOnce(&warmPipe, nfix, 1, 256) / 1e3;
    Stat b = stat(base, 20);
    double gaps[] = { 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 33, 50 };
    int ng = sizeof gaps / sizeof gaps[0];
    double rows[10 * 7];
    for (int g = 0; g < ng; g++) {
        int r = (int)fmin(30, fmax(8, 400 / gaps[g]));
        double ex[64], st[64];
        for (int k = 0; k < r; k++) {
            usleep((useconds_t)(gaps[g] * 1000));
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            uint32_t nn = nfix;
            [e setComputePipelineState:warmPipe.ps]; [e setBuffer:buf offset:0 atIndex:0];
            [e setBytes:&nn length:4 atIndex:1]; [e setBuffer:aux offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            uint64_t t0 = now();
            [cb commit]; [cb waitUntilCompleted];
            ex[k] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
            st[k] = (cb.GPUStartTime - (double)t0 * tickNs * 1e-9) * 1e6;
        }
        Stat e = stat(ex, r), s = stat(st, r);
        rows[g * 7 + 0] = gaps[g]; rows[g * 7 + 1] = r; rows[g * 7 + 2] = e.min; rows[g * 7 + 3] = e.med; rows[g * 7 + 4] = e.max;
        rows[g * 7 + 5] = e.med / b.min; rows[g * 7 + 6] = s.med;
        progress("[clock] gap %5.2f ms: exec min %.0f median %.0f max %.0f us (x%.2f of warm %.0f)\n", gaps[g], e.min, e.med, e.max, e.med / b.min, b.min);
    }
    double worst = 0;
    for (int g = 0; g < ng; g++) worst = fmax(worst, rows[g * 7 + 4] / b.min);
    printf("\n[clock_ramp]\n");
    printf("method = \"fixed single-threadgroup FMA kernel (~200 us warm) after an idle sleep of gap ms; GPU exec from timestamps; ratio = median exec / warm min exec. Min-of-N hides the ramp, so median and max are the result here\"\n");
    printf("warm_exec_us = { min = %.1f, median = %.1f }\n", b.min, b.med);
    printf("worst_slowdown = %.2f\n", worst);
    const char *cols[] = { "gap_ms", "samples", "exec_us_min", "exec_us_median", "exec_us_max", "median_over_warm", "commit_to_gpu_start_us_median" };
    tomlPoints("points", cols, 7, rows, ng);
    fflush(stdout);
}

static void probeBandwidth(void) {
    progress("[bandwidth]\n");
    Pipe cp = makePipe(miscLib, @"copy4");
    const NSUInteger bytes = 128u << 20;
    id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> dst = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    double t[64]; int r = REPS;
    warm(100);
    for (int k = 0; k < r + 3; k++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:cp.ps]; [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dst offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(bytes / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (k >= 3) t[k - 3] = (cb.GPUEndTime - cb.GPUStartTime);
    }
    Stat s = stat(t, r);
    printf("\n[bandwidth]\n");
    printf("method = \"float4 streaming copy kernel, %lu MiB -> %lu MiB private buffers, whole GPU; GB/s = (read + write bytes) / GPU time\"\n",
           (unsigned long)(bytes >> 20), (unsigned long)(bytes >> 20));
    printf("repeats = %d\n", r);
    printf("copy_gb_per_s = { max = %.1f, median = %.1f }\n", 2.0 * bytes / s.min / 1e9, 2.0 * bytes / s.med / 1e9);
    printf("copy_ms = { min = %.3f, median = %.3f }\n", s.min * 1e3, s.med * 1e3);
    progress("[bandwidth] %.1f GB/s (median %.1f)\n", 2.0 * bytes / s.min / 1e9, 2.0 * bytes / s.med / 1e9);
    fflush(stdout);
}

static int gpuCores(void) {
    FILE *f = popen("ioreg -r -d 1 -c IOAccelerator | grep -o '\"gpu-core-count\" = [0-9]*' | head -1 | grep -o '[0-9]*$'", "r");
    int c = 0;
    if (f) { if (fscanf(f, "%d", &c) != 1) c = 0; pclose(f); }
    return c > 0 ? c : 1;
}

// OPT-IN. Bounded: every group waits at most `limit` polls (~3 ms, calibrated), then leaves.
static void probeCoresidency(void) {
    progress("[coresidency] OPT-IN bounded spin\n");
    Pipe p = makePipe(miscLib, @"coresident");
    id<MTLBuffer> ctr = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    id<MTLBuffer> seen = [dev newBufferWithLength:4 * 65536 options:MTLResourceStorageModeShared];
    double (^go)(uint32_t, NSUInteger, NSUInteger) = ^double(uint32_t limit, NSUInteger groups, NSUInteger threads) {
        uint32_t *c = ctr.contents; c[0] = 0; c[1] = (uint32_t)groups;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p.ps]; [e setBuffer:ctr offset:0 atIndex:0];
        [e setBytes:&limit length:4 atIndex:1]; [e setBuffer:seen offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    };
    // calibrate: one group that can never complete (total = 2) times out after `limit` polls
    uint32_t *c = ctr.contents;
    (void)c;
    double ms = 1e30;
    for (int k = 0; k < 3; k++) {
        uint32_t *cc = ctr.contents; cc[0] = 0; cc[1] = 2;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        uint32_t lim = 20000;
        [e setComputePipelineState:p.ps]; [e setBuffer:ctr offset:0 atIndex:0];
        [e setBytes:&lim length:4 atIndex:1]; [e setBuffer:seen offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        ms = fmin(ms, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    }
    double nsPerPoll = ms * 1e6 / 20000;
    uint32_t limit = (uint32_t)fmin(2e6, fmax(1000, 3e6 / nsPerPoll));
    int cores = gpuCores();
    printf("\n[coresidency]\n");
    printf("method = \"OPT-IN. thread 0 of every group increments an arrival counter, then polls it until all groups arrived or %u polls (~3 ms, %.0f ns/poll calibrated alone; scaled by cores/groups above one group per core, since polls of one atomic serialise) pass; "
           "a group records the arrivals it saw. All co-resident => every group sees the total; otherwise the minimum seen ~ groups resident in the first wave\"\n", limit, nsPerPoll);
    printf("gpu_cores = %d\npoll_limit = %u\n", cores, limit);
    int perCore[] = { 1, 2, 4, 8, 16, 24, 32, 48, 64 };
    NSUInteger threads[] = { 32, 64, 256, 1024 };
    double rows[4 * 9 * 6]; int m = 0;
    for (int t = 0; t < 4; t++) {
        if (threads[t] > p.maxThreads) continue;
        for (int g = 0; g < 9; g++) {
            NSUInteger groups = (NSUInteger)perCore[g] * cores;
            if (groups > 65536) break;
            // Polls of one shared atomic serialise: a poll costs ~nsPerPoll x groups / cores, so
            // the per-group limit shrinks with the group count to keep each wait ~3 ms.
            uint32_t lim = groups > (NSUInteger)cores ? (uint32_t)fmax(64, (double)limit * cores / groups) : limit;
            double gms = go(lim, groups, threads[t]);
            uint32_t *sv = seen.contents; uint32_t mn = UINT32_MAX; NSUInteger all = 0;
            for (NSUInteger i = 0; i < groups; i++) { if (sv[i] < mn) mn = sv[i]; if (sv[i] >= groups) all++; }
            rows[m * 6 + 0] = threads[t]; rows[m * 6 + 1] = perCore[g]; rows[m * 6 + 2] = groups;
            rows[m * 6 + 3] = mn; rows[m * 6 + 4] = all; rows[m * 6 + 5] = gms;
            m++;
            progress("[coresidency] %4lu threads x %5lu groups: min seen %u, %lu saw all, %.2f ms\n", (unsigned long)threads[t], (unsigned long)groups, mn, (unsigned long)all, gms);
            if (all < groups) break;  // larger counts cannot all be resident either
        }
    }
    const char *cols[] = { "threads_per_group", "groups_per_core", "groups", "min_arrivals_seen", "groups_that_saw_all", "gpu_ms" };
    tomlPoints("points", cols, 6, rows, m);
    fflush(stdout);
}

// OPT-IN. Single-threadgroup kernels of growing duration until the driver kills one.
static void probeWatchdog(double maxSeconds) {
    progress("[watchdog] OPT-IN, up to %.1f s per command buffer\n", maxSeconds);
    warm(200);
    Slope sl = slope(&warmPipe, 1, 256, 1000);
    printf("\n[watchdog]\n");
    printf("method = \"OPT-IN. one command buffer running a single-threadgroup FMA loop of the target duration (calibrated %.2f ns/iteration); durations double until the command buffer fails or max_seconds\"\n", sl.min);
    printf("max_seconds = %.2f\n", maxSeconds);
    double longest = 0, failedAt = 0; NSString *err = @"";
    for (double secs = 0.25; secs <= maxSeconds + 1e-9; secs *= 2) {
        double iters = secs * 1e9 / fmax(sl.min, 1e-3);
        uint32_t nn = (uint32_t)fmin(iters, 4e9);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:warmPipe.ps]; [e setBuffer:buf offset:0 atIndex:0];
        [e setBytes:&nn length:4 atIndex:1]; [e setBuffer:aux offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double ran = cb.GPUEndTime - cb.GPUStartTime;
        progress("[watchdog] target %.2f s: status %ld, ran %.3f s\n", secs, (long)cb.status, ran);
        if (cb.status != MTLCommandBufferStatusCompleted) { failedAt = secs; err = cb.error.localizedDescription ?: @"error"; break; }
        longest = ran;
    }
    printf("longest_completed_seconds = %.3f\n", longest);
    if (failedAt > 0) printf("failed_at_target_seconds = %.2f\nfailure = \"%s\"\n", failedAt, [[err stringByReplacingOccurrencesOfString:@"\"" withString:@"'"] UTF8String]);
    else printf("failed_at_target_seconds = \"none up to max_seconds\"\n");
    fflush(stdout);
}

// ---------------------------------------------------------------- main

static void initDevice(void) {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    tickNs = (double)tb.numer / tb.denom;
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();  // MTLCreateSystemDefaultDevice() is nil in CLI processes here
    dev = all.firstObject;
    if (!dev) { fprintf(stderr, "no Metal device\n"); exit(2); }
    if (@available(macOS 14.0, *)) archName = dev.architecture.name ?: @"unknown";
}

static int cmdStats(int argc, const char **argv) {
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&err];
    if (!lib) { fprintf(stderr, "load %s: %s\n", argv[2], err.description.UTF8String); return 2; }
    NSMutableArray *names = [NSMutableArray array];
    if (argc > 3) for (int i = 3; i < argc; i++) [names addObject:@(argv[i])];
    else {
        for (NSString *nm in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
            id<MTLFunction> f = [lib newFunctionWithName:nm];
            if (f.functionType == MTLFunctionTypeKernel) [names addObject:nm];
        }
    }
    BOOL warned = NO;
    printf("# name\tinstructions\ttemp_registers\tspilled_bytes\tthreadgroup_memory\tmax_threads_per_threadgroup\talu\tbranch\tdevice_loads\tstatic_threadgroup_memory\n");
    for (NSString *nm in names) {
        Pipe p = makePipe(lib, nm);
        if (p.instrs < 0 && !warned) { fprintf(stderr, "warning: driver statistics unavailable (private pipeline option 0x40000); reporting -1\n"); warned = YES; }
        printf("%s\t%d\t%d\t%d\t%d\t%lu\t%d\t%d\t%d\t%lu\n", p.name, p.instrs, p.tregs, p.spill, p.tgmem, (unsigned long)p.maxThreads,
               p.alu, p.branch, p.devld, (unsigned long)p.ps.staticThreadgroupMemoryLength);
    }
    return 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        initDevice();
        if (argc >= 2 && !strcmp(argv[1], "device")) {
            printf("name = \"%s\"\narchitecture = \"%s\"\n", dev.name.UTF8String, archName.UTF8String);
            return 0;
        }
        if (argc >= 3 && !strcmp(argv[1], "stats")) return cmdStats(argc, argv);
        if (argc < 2 || strcmp(argv[1], "probe")) {
            fprintf(stderr, "usage: limits probe [--probes list] [--repeats N] [--work DIR] [--codesize PATH] [--coresidency] [--watchdog S]\n"
                            "       limits stats <lib.metallib> [function ...]\n       limits device\n");
            return 1;
        }
        NSString *probes = @"icache,regs,stack,ops,dispatch,roundtrip,bandwidth";
        BOOL coresidency = NO; double watchdog = 0;
        workDir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"gpu-limits"];
        NSString *exeDir = [@(argv[0]) stringByDeletingLastPathComponent];
        codesizePath = [[exeDir stringByAppendingPathComponent:@"../codesize.py"] stringByStandardizingPath];
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--probes") && i + 1 < argc) probes = @(argv[++i]);
            else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) REPS = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--work") && i + 1 < argc) workDir = @(argv[++i]);
            else if (!strcmp(argv[i], "--codesize") && i + 1 < argc) codesizePath = @(argv[++i]);
            else if (!strcmp(argv[i], "--coresidency")) coresidency = YES;
            else if (!strcmp(argv[i], "--watchdog") && i + 1 < argc) watchdog = atof(argv[++i]);
            else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
        }
        if (getenv("WARM_ITERS")) warmIters = atoi(getenv("WARM_ITERS"));
        if (getenv("WARM_GROUPS")) warmGroups = atoi(getenv("WARM_GROUPS"));
        if (REPS < 3) REPS = 3;
        if (REPS > 64) REPS = 64;
        [[NSFileManager defaultManager] createDirectoryAtPath:workDir withIntermediateDirectories:YES attributes:nil error:nil];
        queue = [dev newCommandQueue];
        buf = [dev newBufferWithLength:4 << 20 options:MTLResourceStorageModeShared];
        aux = [dev newBufferWithLength:8 << 20 options:MTLResourceStorageModeShared];
        uint32_t *w = buf.contents;
        for (int i = 0; i < 4096; i++) w[i] = (uint32_t)((i * 1031 + 17) % 4096);  // one full permutation cycle
        for (int i = 4096; i < 8192; i++) w[i] = i & 63;
        for (int i = 8192; i < (1 << 20); i++) w[i] = 0;
        miscLib = buildLib(@"misc", miscSource(), NO);
        warmPipe = makePipe(miscLib, @"warm_fma");
        haveWarm = YES;
        BOOL statsOk = warmPipe.instrs >= 0;
        printf("[device]\nname = \"%s\"\narchitecture = \"%s\"\nmax_threads_per_threadgroup = %lu\nmax_threadgroup_memory_bytes = %lu\n"
               "recommended_max_working_set_bytes = %llu\nunified_memory = %s\ndriver_statistics_available = %s\n",
               dev.name.UTF8String, archName.UTF8String, (unsigned long)dev.maxThreadsPerThreadgroup.width,
               (unsigned long)dev.maxThreadgroupMemoryLength, (unsigned long long)dev.recommendedMaxWorkingSetSize,
               dev.hasUnifiedMemory ? "true" : "false", statsOk ? "true" : "false");
        printf("simd_width = %lu\n", (unsigned long)warmPipe.ps.threadExecutionWidth);
        if (!statsOk) progress("warning: driver statistics unavailable (private pipeline option 0x40000)\n");
        NSArray *list = [probes componentsSeparatedByString:@","];
        uint64_t t0 = now();
        if ([list containsObject:@"icache"]) probeICache();
        if ([list containsObject:@"regs"]) probeRegs();
        if ([list containsObject:@"stack"]) probeStack();
        if ([list containsObject:@"ops"]) probeOps();
        if ([list containsObject:@"dispatch"]) probeDispatch();
        if ([list containsObject:@"roundtrip"]) probeRoundTrip();
        if ([list containsObject:@"bandwidth"]) probeBandwidth();
        if (coresidency) probeCoresidency();
        if (watchdog > 0) probeWatchdog(watchdog);
        progress("done in %.1f s\n", ns(now() - t0) / 1e9);
    }
    return 0;
}
