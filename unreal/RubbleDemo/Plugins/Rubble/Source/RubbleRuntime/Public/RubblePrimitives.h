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
	/** Instances drawn through GPU Scene from Rubble's GPU poses. */
	int32 NumGpuInstances() const;
	int32 NumDrawnBodies() const { return BodyInstances.Num(); }
	int32 NumActiveBodies() const { return Active.Num(); }
	/** The drawn transform of a body's first shape (tests). */
	bool BodyTransform(FRubbleBodyId Body, FTransform& Out) const;
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
		int32 First = 0; // into Refs
		int32 Num = 0;
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
		TArray<FVector4f> Local; // four float4s per instance: translation, rotation, scale, tint
		bool bDirty = true;
	};
	/** Whether a body is drawn through GPU Scene: dynamic, small, with a pose slot. */
	bool DrawsOnGpu(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets) const;
	void AddGpuBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets);
	int32 GpuBatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets);
	UStaticMesh* MeshFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets, bool bNanite);
	void WriteGpu(const FRubbleGpuPoses& Poses);
	FRubbleGpuPoses* GpuPoses() const;
	TArray<FGpuBatch> GpuBatches;
	TMap<uint64, int32> GpuBatchByKey;
	TSet<uint32> GpuBodies; // body indices drawn by GPU batches

	void Rebuild(const FRubbleSceneAssets& Assets);
	void OnFrame(const FRubbleFrame& Frame);
	int32 BatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets, bool bTown);
	void Apply(int32 BodyIndex, const FTransform& Pose);
	/** A body that appears mid-scene (a projectile): its instances are added one by one. */
	void AddLiveBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes);
	const FRubbleSceneAssets* SceneAssets = nullptr;

	TArray<FBatch> Batches;
	TMap<uint64, int32> BatchByKey;
	TMap<uint64, TObjectPtr<UStaticMesh>> BuiltMeshes;
	TArray<FInstanceRef> Refs;
	TMap<uint32, FBodyState> BodyInstances; // by body index
	TSet<uint32> Active;   // moving: drawn every tick at the interpolated pose
	TSet<uint32> Settling; // moved in an earlier frame but not the latest: one last draw at rest
	FDelegateHandle FrameHandle, SceneHandle;
	double UpdateMs = 0.0;
};
