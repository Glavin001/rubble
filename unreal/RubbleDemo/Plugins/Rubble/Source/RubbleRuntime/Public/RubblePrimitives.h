// Draws every body made of primitive shapes (boxes, spheres, capsules, hulls): one instanced mesh
// per (shape kind or hull mesh, material), each shape one instance. Two paths:
//  - GPU Scene (when the backend has poses on Unreal's GPU, FRubbleGpuPoses): small dynamic
//    bodies are GPU-only instances whose transforms a compute pass writes every frame from
//    Rubble's pose buffer. Nothing per body runs on the CPU. See unreal/DESIGN.md, section 2.
//  - Readback: static bodies and large ones, so Lumen's distance fields and surface cache see
//    them (the Lumen hybrid), and everything when there are no GPU poses. Only bodies that moved
//    are touched; sleeping ones cost nothing.
// Poses are interpolated between steps by the clock's alpha.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleTypes.h"
#include "RubblePrimitives.generated.h"

class UInstancedStaticMeshComponent;
class URubbleInstancesComponent;
class FRubbleGpuPoses;
class UStaticMesh;
struct FRubbleEvent;
struct FRubbleFrame;
struct FRubbleSceneAssets;

UCLASS()
class RUBBLERUNTIME_API ARubblePrimitives : public AActor
{
	GENERATED_BODY()

public:
	ARubblePrimitives();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

	int32 NumInstances() const;
	/** Instances drawn through GPU Scene from Rubble's GPU poses (not counting hybrids). */
	int32 NumGpuInstances() const;
	/** Hybrid bodies (the town's chunks) awake now, drawn from GPU poses. */
	int32 NumAwakeHybrids() const;
	int32 NumHybrids() const { return Hybrids.Num(); }
	int32 NumDrawnBodies() const { return BodyInstances.Num(); }
	int32 NumActiveBodies() const { return Active.Num(); }
	/** The drawn transform of a body's first shape (tests). */
	bool BodyTransform(FRubbleBodyId Body, FTransform& Out) const;
	/** A destructible chunk's drawn instance transform, as drawn this frame (scale included); false
	 *  when it has no CPU-drawn instance. Scripted cameras ride it. */
	bool ChunkTransform(int32 Chunk, FTransform& Out) const;
	double LastUpdateMs() const { return UpdateMs; }

private:
	struct FInstanceRef
	{
		int32 Batch = 0;
		int32 Instance = 0;
		FTransform Local; // the shape's local transform with its mesh scale
	};
	struct FBodyState
	{
		FTransform Previous;
		FTransform Current;
		TArray<int32> Refs; // into Refs (a split moves some to other bodies)
	};
	struct FBatch
	{
		TObjectPtr<UInstancedStaticMeshComponent> Component;
		TArray<FTransform> Transforms; // the drawn transforms, for whole-batch updates
		int32 Dirty = 0;
	};

	/** A GPU Scene batch: its instance tables are handed to the component when they change; the
	 *  transforms are written on the GPU every frame. */
	struct FGpuBatch
	{
		TObjectPtr<URubbleInstancesComponent> Component;
		TArray<uint32> PoseSlot;
		TArray<FVector4f> Local; // the component's RowsPerInstance() float4s per instance
		bool bDirty = true;
		/** A hybrid batch (below): written only while some instance is awake, and once after a
		 *  change so the instances that fell asleep are hidden. */
		bool bHybrid = false;
		int32 Awake = 0;
		bool bWritePending = true;
	};
	/**
	 * A hybrid body (bLumenVisible with GPU poses: the town's chunks): at rest it is a CPU-visible
	 * instance, so Lumen sees it; while awake a GPU Scene instance drawn from Rubble's poses, so
	 * nothing per body runs on the CPU. Its frames report only sleep and wake (rbBody_SetMoveReports):
	 * waking swaps it to the GPU instance, falling asleep puts the CPU instance at the pose the
	 * sleep reported.
	 */
	struct FHybridRef
	{
		int32 GpuBatch = 0;
		int32 GpuIndex = 0;
	};
	struct FHybrid
	{
		uint32 PoseSlot = 0;
		int32 FirstRef = 0; // into HybridRefs, one per shape (parallel to the body's Refs)
		bool bAwake = false;
	};
	TMap<uint32, FHybrid> Hybrids; // by body index
	TArray<FHybridRef> HybridRefs;
	void AddHybrid(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets);
	void SetHybridAwake(uint32 BodyIndex, FHybrid& H, bool bAwake, const FTransform* RestPose, TSet<int32>& TouchedBatches);
	/** A GPU instance's table rows: translation, rotation, scale (in its pose record's frame),
	 *  then its custom data (a tint, or the town surface's 11 floats). */
	static void GpuRows(const FRubbleShapeVisual& Shape, const FTransform& Offset, bool bTown, TArray<FVector4f>& Out);
	/** Whether a body is drawn through GPU Scene: dynamic, small, with a pose slot. */
	bool DrawsOnGpu(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets) const;
	void AddGpuBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets);
	int32 GpuBatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets, bool bTown = false, bool bHybrid = false, bool bCurtain = false, FName Surface = NAME_None);
	UStaticMesh* MeshFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets, bool bNanite);
	void WriteGpu(const FRubbleGpuPoses& Poses);
	FRubbleGpuPoses* GpuPoses() const;
	TArray<FGpuBatch> GpuBatches;
	TMap<uint64, int32> GpuBatchByKey;
	TSet<uint32> GpuBodies; // body indices drawn by GPU batches

	void Rebuild(const FRubbleSceneAssets& Assets);
	void OnFrame(const FRubbleFrame& Frame);
	int32 BatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets, bool bTown, bool bCurtain = false, FName Surface = NAME_None);
	void Apply(int32 BodyIndex, const FTransform& Pose);
	/** A Split: each child body draws its chunks' instances, from where the parent was. */
	void Rebind(const FRubbleEvent& Split);
	/** A body that appears mid-scene (a projectile): its instances are added one by one. */
	void AddLiveBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes);
	const FRubbleSceneAssets* SceneAssets = nullptr;

	TArray<FBatch> Batches;
	TMap<uint64, int32> BatchByKey;
	TMap<uint64, TObjectPtr<UStaticMesh>> BuiltMeshes;
	TArray<FInstanceRef> Refs;
	TMap<uint32, FBodyState> BodyInstances; // by body index
	TMap<int32, int32> ChunkRefs; // a Chunk shape's ref, by chunk
	TSet<uint32> Active;   // moving: drawn every tick at the interpolated pose
	TSet<uint32> Settling; // moved in an earlier frame but not the latest: one last draw at rest
	FDelegateHandle FrameHandle, SceneHandle;
	double UpdateMs = 0.0;
};
