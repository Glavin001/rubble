// Draws destructibles. An intact destructible is one merged mesh (its root proxy: one draw); on its
// first Split it is swapped for one mesh per chunk. A chunk is drawn at its body's pose times its
// rest transform (every body of a destructible has its origin at the destructible's frame), so a
// split never rebuilds a mesh: chunks just change body.
//
// With Rubble's poses on the GPU (FRubbleGpuPoses) and a body per chunk from the start (the Bonds
// model), small chunks are GPU Scene instances written from the pose buffer every frame, made
// hidden at load and shown when their destructible shatters: no per-chunk CPU work. Large chunks
// stay CPU-placed, so Lumen's distance fields and surface cache see them (unreal/DESIGN.md).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleTypes.h"
#include "RubbleFracture.generated.h"

class UStaticMesh;
class UStaticMeshComponent;
class URubbleInstancesComponent;
class FRubbleGpuPoses;
struct FRubbleFrame;
struct FRubbleSceneAssets;

UCLASS()
class RUBBLERUNTIME_API ARubbleFracture : public AActor
{
	GENERATED_BODY()

public:
	ARubbleFracture();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

	/** How many times each chunk is drawn right now (each should be 1): tests. */
	TArray<int32> ChunkDrawCounts() const;
	int32 NumBodies() const { return BodyChunks.Num(); }
	int32 NumGpuChunks() const;
	bool IsProxyShown(int32 Destructible) const;

private:
	struct FBodyPose
	{
		FTransform Previous;
		FTransform Current;
	};

	void Rebuild(const FRubbleSceneAssets& Assets);
	void OnFrame(const FRubbleFrame& Frame);
	void Shatter(int32 Destructible);
	void Draw(uint32 Body, const FTransform& Pose);
	FRubbleGpuPoses* GpuPoses() const;

	const FRubbleSceneAssets* Assets = nullptr;
	UPROPERTY()
	TArray<TObjectPtr<UStaticMesh>> ChunkMeshes;
	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> ChunkComponents; // null until its destructible shatters
	UPROPERTY()
	TArray<TObjectPtr<UStaticMeshComponent>> Proxies;
	/** GPU Scene chunks (null for CPU-placed ones), hidden until their destructible shatters. */
	UPROPERTY()
	TArray<TObjectPtr<URubbleInstancesComponent>> GpuChunks;
	TArray<FTransform> ChunkRest;
	TArray<uint32> ChunkBody;
	TMap<uint32, TArray<int32>> BodyChunks;
	TMap<uint32, FBodyPose> Poses;
	TSet<uint32> Active;
	TSet<uint32> Settling;
	FDelegateHandle FrameHandle, SceneHandle;
};
