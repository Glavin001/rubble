// Draws every body made of primitive shapes (boxes, spheres, capsules, hulls): one instanced mesh
// per (shape kind or hull mesh, material), each shape one instance. Only bodies that moved are
// touched; sleeping ones cost nothing. Poses are interpolated between steps by the clock's alpha.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleTypes.h"
#include "RubblePrimitives.generated.h"

class UInstancedStaticMeshComponent;
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

	void Rebuild(const FRubbleSceneAssets& Assets);
	void OnFrame(const FRubbleFrame& Frame);
	int32 BatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets);
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
