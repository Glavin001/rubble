// Cosmetic effects from the frame's events: dust puffs and chips where things break, sparks where
// steel hits, glass where a soft-body part comes off. Unreal-side only (the world never sees them),
// drawn as three instanced meshes simulated here, so nothing has to be authored in Niagara.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleTypes.h"
#include "RubbleEffects.generated.h"

class UInstancedStaticMeshComponent;
struct FRubbleFrame;
struct FRubbleSceneAssets;

UCLASS()
class RUBBLERUNTIME_API ARubbleEffects : public AActor
{
	GENERATED_BODY()

public:
	ARubbleEffects();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

	int32 NumLive() const;
	bool bEnabled = true;

private:
	enum class EKind : uint8 { Chip, Dust, Spark };
	struct FParticle
	{
		FVector Pos = FVector::ZeroVector;
		FVector Vel = FVector::ZeroVector;
		FQuat Rot = FQuat::Identity;
		FVector AngVel = FVector::ZeroVector;
		FVector Scale = FVector(1.0);
		FLinearColor Color = FLinearColor::White;
		float Age = 0.0f;
		float Life = 0.0f; // 0: free
		float Size0 = 1.0f;
		float Size1 = 1.0f;
	};
	struct FPool
	{
		TObjectPtr<UInstancedStaticMeshComponent> Component;
		TArray<FParticle> Particles;
		TArray<FTransform> Transforms;
		int32 Head = 0;
		int32 Live = 0;
	};

	void OnFrame(const FRubbleFrame& Frame);
	void OnScene(const FRubbleSceneAssets& Assets);
	FParticle& Spawn(EKind Kind);
	void Chips(const FVector& At, int32 Count, ERubbleMaterial Material, float Speed, float Size);
	void DustPuffs(const FVector& At, int32 Count, float Radius);
	void Sparks(const FVector& At, const FVector& Normal, int32 Count);
	FPool& PoolOf(EKind Kind) { return Pools[int32(Kind)]; }

	FPool Pools[3];
	FRandomStream Random{ 77 };
	int32 SpawnedThisFrame = 0;
	FDelegateHandle FrameHandle, SceneHandle;
};
