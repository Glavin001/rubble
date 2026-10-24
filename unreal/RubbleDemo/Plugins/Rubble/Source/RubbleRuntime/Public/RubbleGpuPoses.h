// Rubble's render poses on Unreal's GPU, for the GPU Scene renderers (unreal/DESIGN.md, sections
// 1 and 2): the world runs on Unreal's own Metal device, on a queue of its own, and copies each
// frame's poses into buffers Unreal owns (rbWorld_ExportPoses). Nothing per body is read back.
//
// Unreal's buffers are not hazard-tracked, so the slots form a ring:
//  - a slot is drawn only once its export is seen finished (rbWorld_GetCompletedPoseExport);
//  - it is written again only after an Unreal GPU fence shows the frames that drew it are done.
// The drawn poses are therefore one display frame behind the newest step, as with the readback
// path.
//
// Metal only for now (the Metal bridge, DESIGN.md phase 1a). Elsewhere Supported() is false and
// the readback renderers draw everything.

#pragma once

#include "CoreMinimal.h"
#include "RenderGraphResources.h"
#include "RHIResources.h"
#include "RubbleTypes.h"

struct RbWorld;

/** rubble.GpuScene: draw small dynamic bodies and chunks through GPU Scene (read at scene load). */
RUBBLERUNTIME_API bool RubbleGpuSceneEnabled();
/** rubble.GpuScene.MaxExtentCm: larger bodies and chunks stay CPU-placed, so Lumen sees them. */
RUBBLERUNTIME_API float RubbleGpuSceneMaxExtentCm();

class RUBBLERUNTIME_API FRubbleGpuPoses
{
public:
	/** Whether this RHI can share buffers with Rubble; Why says why not. */
	static bool Supported(FString& Why);
	/** Unreal's MTLDevice and a new command queue on it, for rbCreateWorldOnMetal (the queue is
	 *  returned retained: release it with ReleaseQueue once the world holds it). */
	static bool DeviceAndQueue(void*& OutDevice, void*& OutQueue, FString& Error);
	static void ReleaseQueue(void* Queue);

	/** Pose slots for Capacity bodies, allocated now on Unreal's device (flushes the render
	 *  thread). World must outlive this. */
	FRubbleGpuPoses(RbWorld* InWorld, uint32 InCapacity);
	~FRubbleGpuPoses();

	bool IsValid() const { return bValid; }
	uint32 Capacity() const { return NumPoses; }

	/** Once per display frame, after the frame's steps (game thread): draw the newest finished
	 *  export, then export the latest step's poses into a free slot. Alpha is the clock's alpha
	 *  this frame, drawn with that export. */
	void Update(float Alpha);

	/** What to draw this frame. Current and Previous are float4 pairs per pose slot (Rubble's
	 *  32-byte records, Rubble's frame); not valid until the first export finishes. */
	struct FView
	{
		TRefCountPtr<FRDGPooledBuffer> Current;
		TRefCountPtr<FRDGPooledBuffer> Previous;
		float Alpha = 1.0f;
		bool bValid = false;
	};
	const FView& Latest() const { return Drawn; }

	/** A body's slot in the pose buffers (rbBody_GetPoseIndex); MAX_uint32 if unknown. */
	uint32 PoseIndex(FRubbleBodyId Body) const;
	/** Where a body's origin is relative to its pose record (rbBody_GetPoseOffset, Unreal space):
	 *  records are solver poses (centre of mass), so a shape draws at Local x Offset x record. */
	FTransform PoseOffset(FRubbleBodyId Body) const;

	/** A body is drawn from these poses: its frames then report only its sleep and wake changes
	 *  (rbBody_SetMoveReports), so moves stay per event on the CPU (rubble.GpuScene.AllMoves 1
	 *  keeps every move). */
	void DrawnOnGpu(FRubbleBodyId Body) const;

	/** Exports skipped because every slot was busy (the GPU running behind), for the HUD. */
	uint64 SkippedExports() const { return Skipped; }

private:
	struct FSlot
	{
		TRefCountPtr<FRDGPooledBuffer> Current, Previous;
		void* CurrentBuffer = nullptr; // MTLBuffer*
		uint64 CurrentOffset = 0;
		void* PreviousBuffer = nullptr;
		uint64 PreviousOffset = 0;
		uint64 Ticket = 0; // the export in it; 0: none
		float Alpha = 1.0f;
		FGPUFenceRHIRef Fence;
		/** Drawn at some point: free again only once Fence is written and has signalled. */
		bool bFenced = false;
		TSharedRef<std::atomic<bool>, ESPMode::ThreadSafe> FenceWritten = MakeShared<std::atomic<bool>, ESPMode::ThreadSafe>(false);
	};
	bool IsFree(int32 Index) const;

	RbWorld* World = nullptr;
	uint32 NumPoses = 0;
	TArray<FSlot> Slots;
	int32 DrawnSlot = INDEX_NONE;
	uint64 LastTicket = 0;
	FView Drawn;
	uint64 Skipped = 0;
	bool bValid = false;
};
