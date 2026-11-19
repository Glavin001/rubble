// A CPU stand-in for the Metal shading language, enough to run kernels/floor.metal as
// written: each lane a coroutine, a threadgroup barrier the point where every lane has
// finished the phase. IEEE single arithmetic with no contraction (-ffp-contract=off), so
// what it checks is the kernel's logic against the CPU reference, before any GPU time.
#pragma once
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace metal {}
typedef unsigned int uint;
typedef unsigned char uchar;
struct float2 { float x, y; float2() : x(0), y(0) {} float2(float a, float b) : x(a), y(b) {} };
struct float3 { float x, y, z; float3() : x(0), y(0), z(0) {} explicit float3(float a) : x(a), y(a), z(a) {} float3(float a, float b, float c) : x(a), y(b), z(c) {} };
struct packed_float3 { float x, y, z; packed_float3() : x(0), y(0), z(0) {} packed_float3(float a, float b, float c) : x(a), y(b), z(c) {} };
struct float4 { float x, y, z, w; float4() : x(0), y(0), z(0), w(0) {} explicit float4(float a) : x(a), y(a), z(a), w(a) {} float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {} };
struct uint2 { uint x, y; };
struct int3 { int x, y, z; int3(int a, int b, int c) : x(a), y(b), z(c) {} int operator[](uint i) const { return (&x)[i]; } };
struct uint3 { uint x, y, z; uint3() : x(0), y(0), z(0) {} uint3(uint a, uint b, uint c) : x(a), y(b), z(c) {} };
struct uint4 { uint x, y, z, w; uint4() : x(0), y(0), z(0), w(0) {} explicit uint4(uint a) : x(a), y(a), z(a), w(a) {} uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
    uint operator[](uint i) const { return (&x)[i]; } };
template <class T, class U> inline T as_type(U u) { static_assert(sizeof(T) == sizeof(U), "as_type"); T t; memcpy(&t, &u, sizeof(T)); return t; }
using std::round; using std::fabs; using std::floor; using std::copysign; using std::rint; using std::sqrt; using std::fmin; using std::fmax; using std::abs;
using std::fma;
inline float clamp(float x, float lo, float hi) { return fmin(fmax(x, lo), hi); }
inline uint min(uint a, uint b) { return a < b ? a : b; }
inline uint max(uint a, uint b) { return a > b ? a : b; }
// Atomics: lanes run one at a time, so plain words.
struct atomic_int { int v; }; struct atomic_uint { uint v; }; struct atomic_float { float v; };
enum { memory_order_relaxed };
template <class A> inline auto atomic_load_explicit(const A* a, int) { return a->v; }
template <class A, class T> inline void atomic_store_explicit(A* a, T x, int) { a->v = x; }
template <class A, class T> inline auto atomic_exchange_explicit(A* a, T x, int) { auto o = a->v; a->v = x; return o; }
template <class A, class T> inline auto atomic_fetch_add_explicit(A* a, T x, int) { auto o = a->v; a->v = o + x; return o; }
template <class A, class T> inline auto atomic_fetch_min_explicit(A* a, T x, int) { auto o = a->v; if ((decltype(o))x < o) a->v = x; return o; }
template <class A, class T> inline auto atomic_fetch_or_explicit(A* a, T x, int) { auto o = a->v; a->v = o | x; return o; }
template <class A, class T> inline auto atomic_fetch_max_explicit(A* a, T x, int) { auto o = a->v; if ((decltype(o))x > o) a->v = x; return o; }
namespace mem_flags { enum { mem_threadgroup = 1, mem_device = 2 }; }
// A lane: a coroutine suspended at each barrier.
struct Lane {
    struct promise_type {
        Lane get_return_object() { return Lane{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { abort(); }
    };
    std::coroutine_handle<promise_type> h;
};
#define threadgroup_barrier(flags) co_await std::suspend_always{}
// The barrier across workgroups: the lane stops and says so; the scheduler lets the
// groups go on once every group's lanes are there.
inline bool g_at_x = false;
struct XWait { bool await_ready() { return false; } void await_suspend(std::coroutine_handle<>) { g_at_x = true; } void await_resume() {} };
#define EMU 1
#define XBAR co_await XWait{}
#define KRETURN co_return
// A workgroup's own memory: one copy a group (the kernel's `car` is its group).
#define TGPTR(type) type*
#define TGDECL(type, name, count) static type name##_all[64][count]; type* const name = name##_all[car]
#define kernel
#define device
#define constant const
#define thread
#define threadgroup static
