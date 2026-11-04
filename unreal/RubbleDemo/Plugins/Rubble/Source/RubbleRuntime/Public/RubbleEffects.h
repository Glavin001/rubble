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
	/** Color (alpha > 0) overrides the material's chip colour. */
	void Chips(const FVector& At, int32 Count, ERubbleMaterial Material, float Speed, float Size, const FLinearColor& Color = FLinearColor::Transparent);
	/** Puffs of dust Radius cm across that grow, drift up and linger for Life seconds (a range
	 *  around it); Color tints them (alpha 0: plain grey-beige dust). */
	void DustPuffs(const FVector& At, int32 Count, float Radius, float Life = 4.0f, const FLinearColor& Color = FLinearColor::Transparent);
	/** A collapse's cloud where Bonds bonds broke: puffs scaled with the break, lingering. */
	void Collapse(const FVector& At, int32 Bonds, ERubbleMaterial Material, const FLinearColor& Color);
	void Sparks(const FVector& At, const FVector& Normal, int32 Count);
	/** A soft car's damage since the last frame, from its skin's face strain (car-render.mjs's
	 *  spawnFx): a torn panel sheds a paint chip, a pane that shatters a shard, a sudden crumple
	 *  a spark and a little dust. */
	void SoftDamage(const FRubbleSoftSkin& Skin, float Dt);
	FPool& PoolOf(EKind Kind) { return Pools[int32(Kind)]; }

	/** Per soft body: its triangles, their car surfaces and paint, and last frame's skin. */
	struct FSoftState
	{
		TArray<uint32> Indices;
		TArray<uint8> Surfaces;
		FLinearColor Paint = FLinearColor::White;
		TArray<float> PrevStrain;
		TArray<FVector3f> PrevPositions;
	};
	TArray<FSoftState> Soft;

	FPool Pools[3];
	FRandomStream Random{ 77 };
	int32 SpawnedThisFrame = 0;
	FDelegateHandle FrameHandle, SceneHandle;
};
