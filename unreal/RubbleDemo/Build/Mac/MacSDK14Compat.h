// Lets UE 5.8 modules compile against the macOS 14 SDK (Xcode 15.4, the newest Xcode on Sonoma 14.4).
//
// Engine/Source/Runtime/Core/Public/Apple/AppleRecursiveMutex.h names os_unfair_lock_flags_t and
// OS_UNFAIR_LOCK_FLAG_ADAPTIVE_SPIN, which <os/lock.h> first declares in the macOS 15 SDK. The engine
// calls the flagged lock through a pointer resolved at runtime (Core/Private/Apple/AppleRecursiveMutex.cpp):
// on macOS 14, os_unfair_lock_lock_with_flags is absent and the fallback ignores the flags and calls
// os_unfair_lock_lock. So only the declarations are missing; behaviour is the engine's own.
// Force-included by the targets in Source/*.Target.cs. Inert with the macOS 15 SDK or newer.

#pragma once

#if defined(__APPLE__)
#include <Availability.h>
#include <os/lock.h>

#if __MAC_OS_X_VERSION_MAX_ALLOWED < 150000
typedef uint32_t os_unfair_lock_flags_t;
// Value from the macOS 15 SDK's <os/lock.h>.
#define OS_UNFAIR_LOCK_FLAG_NONE ((os_unfair_lock_flags_t)0x00000000u)
#define OS_UNFAIR_LOCK_FLAG_ADAPTIVE_SPIN ((os_unfair_lock_flags_t)0x00040000u)
#endif
#endif
