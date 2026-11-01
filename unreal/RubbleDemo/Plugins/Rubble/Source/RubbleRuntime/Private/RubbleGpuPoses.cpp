#include "RubbleGpuPoses.h"

#include "Algo/AllOf.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformProcess.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RenderingThread.h"
#include "RHICommandList.h"
#include "RubbleSubsystem.h"
#include "rubble.h"

#if PLATFORM_MAC
#include "IMetalDynamicRHI.h"
#include "MetalResources.h"
#endif

namespace
{
	/** Slots in the ring: one being drawn, frames in flight behind it, and one being written. */
	constexpr int32 NumSlots = 4;

	TAutoConsoleVariable<int32> CVarGpuScene(TEXT("rubble.GpuScene"), 1,
		TEXT("Draw small dynamic bodies and chunks through GPU Scene from Rubble's GPU poses when the backend has them (read at scene load)."));
	TAutoConsoleVariable<float> CVarGpuSceneMaxExtentCm(TEXT("rubble.GpuScene.MaxExtentCm"), 150.0f,
		TEXT("Bodies and chunks larger than this (cm from their origin) stay CPU-placed, so Lumen's distance fields and surface cache see them."));
	TAutoConsoleVariable<int32> CVarGpuSceneAllMoves(TEXT("rubble.GpuScene.AllMoves"), 0,
		TEXT("Keep reporting every move of bodies drawn from GPU poses (normally only their sleep and wake changes reach the CPU)."));
}

bool RubbleGpuSceneEnabled()
{
	return CVarGpuScene.GetValueOnGameThread() != 0;
}

float RubbleGpuSceneMaxExtentCm()
{
	return CVarGpuSceneMaxExtentCm.GetValueOnGameThread();
}

bool FRubbleGpuPoses::Supported(FString& Why)
{
#if PLATFORM_MAC
	if (!IsRHIMetal())
	{
		Why = TEXT("the RHI is not Metal");
		return false;
	}
	return true;
#else
	Why = TEXT("GPU poses need the Metal bridge (Apple platforms) until the render-graph host exists");
	return false;
#endif
}

bool FRubbleGpuPoses::DeviceAndQueue(void*& OutDevice, void*& OutQueue, FString& Error)
{
#if PLATFORM_MAC
	if (!IsRHIMetal())
	{
		Error = TEXT("the RHI is not Metal");
		return false;
	}
	MTL::Device* Device = GetIMetalDynamicRHI()->RHIGetDevice();
	// A queue of Rubble's own: its steps never wait in Unreal's submission order, and can overlap
	// rendering. The ring of slots orders the two (see the header).
	// wgpu keeps up to 4096 command buffers in flight on its own queues (wgpu-hal metal
	// MAX_COMMAND_BUFFERS); Metal's default of 64 blocks a 240 Hz soft scene inside
	// [MTLCommandQueue commandBuffer].
	MTL::CommandQueue* Queue = Device ? Device->newCommandQueue(4096) : nullptr;
	if (!Queue)
	{
		Error = TEXT("no Metal command queue");
		return false;
	}
	Queue->setLabel(NS::String::string("Rubble", NS::UTF8StringEncoding));
	OutDevice = Device;
	OutQueue = Queue;
	return true;
#else
	Error = TEXT("not Metal");
	return false;
#endif
}

void FRubbleGpuPoses::ReleaseQueue(void* Queue)
{
#if PLATFORM_MAC
	if (Queue)
	{
		static_cast<MTL::CommandQueue*>(Queue)->release();
	}
#endif
}

FRubbleGpuPoses::FRubbleGpuPoses(RbWorld* InWorld, uint32 InCapacity)
	: World(InWorld)
	, NumPoses(InCapacity)
{
#if PLATFORM_MAC
	Slots.SetNum(NumSlots);
	const uint32 Count = NumPoses * 2; // two float4s per 32-byte record
	ENQUEUE_RENDER_COMMAND(RubbleGpuPosesAllocate)([this, Count](FRHICommandListImmediate& RHICmdList)
	{
		// Allocate and clear each buffer once (Metal allocates a buffer's memory on first use),
		// then read its MTLBuffer and offset with the RHI thread flushed.
		FRDGBuilder G(RHICmdList);
		for (int32 i = 0; i < Slots.Num(); ++i)
		{
			FSlot& S = Slots[i];
			const FRDGBufferDesc Desc = FRDGBufferDesc::CreateStructuredDesc(sizeof(FVector4f), Count);
			S.Current = AllocatePooledBuffer(Desc, TEXT("Rubble.Poses.Current"));
			S.Previous = AllocatePooledBuffer(Desc, TEXT("Rubble.Poses.Previous"));
			AddClearUAVPass(G, G.CreateUAV(G.RegisterExternalBuffer(S.Current)), 0u);
			AddClearUAVPass(G, G.CreateUAV(G.RegisterExternalBuffer(S.Previous)), 0u);
			S.Fence = RHICreateGPUFence(TEXT("Rubble.Poses.Read"));
		}
		G.Execute();
		RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
		auto Native = [](const TRefCountPtr<FRDGPooledBuffer>& Pooled, void*& Buffer, uint64& Offset)
		{
			const FMetalRHIBuffer* Metal = static_cast<const FMetalRHIBuffer*>(Pooled->GetRHI());
			const FMetalBufferPtr& B = Metal->GetCurrentBufferOrNull();
			Buffer = B ? B->GetMTLBuffer() : nullptr;
			Offset = B ? B->GetOffset() : 0;
		};
		for (FSlot& S : Slots)
		{
			Native(S.Current, S.CurrentBuffer, S.CurrentOffset);
			Native(S.Previous, S.PreviousBuffer, S.PreviousOffset);
		}
	});
	FlushRenderingCommands();
	bValid = World && Slots.Num() == NumSlots && Algo::AllOf(Slots, [](const FSlot& S) { return S.CurrentBuffer && S.PreviousBuffer && S.Fence; });
	if (!bValid)
	{
		UE_LOG(LogRubble, Error, TEXT("GPU poses: could not get Unreal's Metal buffers; the readback path draws instead"));
	}
#endif
}

FRubbleGpuPoses::~FRubbleGpuPoses()
{
	// Rubble's last copies must land before Unreal frees (and reuses) the memory.
	if (World && LastTicket)
	{
		const double Until = FPlatformTime::Seconds() + 2.0;
		uint64 Done = 0;
		while (rbWorld_GetCompletedPoseExport(World, &Done) == RB_OK && Done < LastTicket && FPlatformTime::Seconds() < Until)
		{
			FPlatformProcess::Sleep(0.0005f);
		}
	}
	TArray<FSlot> Released = MoveTemp(Slots);
	ENQUEUE_RENDER_COMMAND(RubbleGpuPosesRelease)([Released = MoveTemp(Released)](FRHICommandListImmediate&) mutable
	{
		Released.Reset();
	});
	Drawn = FView();
}

bool FRubbleGpuPoses::IsFree(int32 Index) const
{
	const FSlot& S = Slots[Index];
	if (Index == DrawnSlot)
	{
		return false;
	}
	if (S.Ticket != 0 && !S.bFenced)
	{
		return false; // exported, not yet drawn or retired
	}
	if (S.bFenced)
	{
		return S.FenceWritten->load() && S.Fence->Poll();
	}
	return true;
}

void FRubbleGpuPoses::Update(float Alpha)
{
	if (!bValid)
	{
		return;
	}
	// 1. Draw the newest finished export.
	uint64 Done = 0;
	if (rbWorld_GetCompletedPoseExport(World, &Done) != RB_OK)
	{
		UE_LOG(LogRubble, Error, TEXT("rbWorld_GetCompletedPoseExport: %s"), UTF8_TO_TCHAR(rbGetLastError()));
		return;
	}
	int32 Newest = INDEX_NONE;
	for (int32 i = 0; i < Slots.Num(); ++i)
	{
		const FSlot& S = Slots[i];
		if (!S.bFenced && S.Ticket != 0 && S.Ticket <= Done && (Newest == INDEX_NONE || S.Ticket > Slots[Newest].Ticket))
		{
			Newest = i;
		}
	}
	if (Newest != INDEX_NONE && Newest != DrawnSlot)
	{
		// Older finished exports were never drawn: free now.
		for (int32 i = 0; i < Slots.Num(); ++i)
		{
			FSlot& S = Slots[i];
			if (i != Newest && i != DrawnSlot && !S.bFenced && S.Ticket != 0 && S.Ticket <= Done)
			{
				S.Ticket = 0;
			}
		}
		if (DrawnSlot != INDEX_NONE)
		{
			// Retire the slot drawn until now. Every frame that drew it was handed to the render
			// thread before this command, so the fence follows their GPU work.
			FSlot& Old = Slots[DrawnSlot];
			Old.bFenced = true;
			Old.FenceWritten->store(false);
			ENQUEUE_RENDER_COMMAND(RubbleGpuPosesRetire)([Fence = Old.Fence, Written = Old.FenceWritten](FRHICommandListImmediate& RHICmdList)
			{
				Fence->Clear();
				RHICmdList.WriteGPUFence(Fence);
				Written->store(true);
			});
		}
		DrawnSlot = Newest;
		const FSlot& S = Slots[Newest];
		Drawn.Current = S.Current;
		Drawn.Previous = S.Previous;
		Drawn.Alpha = S.Alpha;
		Drawn.bValid = true;
	}
	// 2. Export the latest step into a free slot (the oldest retired one first).
	int32 Target = INDEX_NONE;
	for (int32 i = 0; i < Slots.Num(); ++i)
	{
		if (IsFree(i))
		{
			Target = i;
			break;
		}
	}
	if (Target == INDEX_NONE)
	{
		++Skipped; // the GPU is behind: draw the older poses another frame
		return;
	}
	FSlot& S = Slots[Target];
	const uint64 Bytes = uint64(NumPoses) * 32;
	uint64 Ticket = 0;
	if (rbWorld_ExportPoses(World, S.CurrentBuffer, S.CurrentOffset, S.PreviousBuffer, S.PreviousOffset, Bytes, &Ticket) != RB_OK)
	{
		UE_LOG(LogRubble, Error, TEXT("rbWorld_ExportPoses: %s"), UTF8_TO_TCHAR(rbGetLastError()));
		return;
	}
	S.Ticket = Ticket;
	S.Alpha = Alpha;
	S.bFenced = false;
	LastTicket = Ticket;
}

uint32 FRubbleGpuPoses::PoseIndex(FRubbleBodyId Body) const
{
	uint32 Index = MAX_uint32;
	if (!World || rbBody_GetPoseIndex(World, Body.Bits, &Index) != RB_OK)
	{
		return MAX_uint32;
	}
	return Index < NumPoses ? Index : MAX_uint32;
}

FTransform FRubbleGpuPoses::PoseOffset(FRubbleBodyId Body) const
{
	RbTransform T;
	if (!World || rbBody_GetPoseOffset(World, Body.Bits, &T) != RB_OK)
	{
		return FTransform::Identity;
	}
	return FTransform(FQuat(T.q.x, T.q.y, T.q.z, T.q.w), FVector(T.p.x, T.p.y, T.p.z));
}

void FRubbleGpuPoses::DrawnOnGpu(FRubbleBodyId Body) const
{
	if (World && CVarGpuSceneAllMoves.GetValueOnGameThread() == 0)
	{
		rbBody_SetMoveReports(World, Body.Bits, RB_MOVES_SLEEP_WAKE);
	}
}
