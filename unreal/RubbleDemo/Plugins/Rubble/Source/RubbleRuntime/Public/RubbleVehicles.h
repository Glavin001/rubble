// Draws raycast vehicles (a placeholder body per style, moved by the chassis body's pose) and every
// wheel the frame reports, for both raycast and soft-body vehicles: tyre, rim and spokes at the
// hub pose, spun and steered, interpolated between steps.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleTypes.h"
#include "RubbleVehicles.generated.h"

class USceneComponent;
class UStaticMeshComponent;
struct FRubbleFrame;
struct FRubbleSceneAssets;

UCLASS()
class RUBBLERUNTIME_API ARubbleVehicles : public AActor
{
	GENERATED_BODY()

public:
	ARubbleVehicles();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;

	/** The player's vehicle (raycast chassis or soft-body core) as last drawn, for the chase camera. */
	bool PlayerTransform(FTransform& Out) const;
	int32 NumWheels() const { return Wheels.Num(); }

private:
	struct FWheel
	{
		TObjectPtr<USceneComponent> Hub;
		TObjectPtr<USceneComponent> Spin;
		FRubbleWheelVisual Previous;
		FRubbleWheelVisual Current;
		bool bSeen = false;
	};
	struct FChassis
	{
		TObjectPtr<USceneComponent> Root;
		FTransform Previous;
		FTransform Current;
		bool bPlayer = false;
	};

	void Rebuild(const FRubbleSceneAssets& Assets);
	void OnFrame(const FRubbleFrame& Frame);
	void BuildBody(USceneComponent* Root, const FRubbleVehicleVisual& V);
	void AddWheel(uint64 Key, const FRubbleWheelDef& Def, const FTransform& Initial);
	UStaticMeshComponent* Part(USceneComponent* Parent, class UStaticMesh* Mesh, const FTransform& Local, class UMaterialInterface* Material);

	TMap<uint32, FChassis> Chassis; // by body index
	/** Chassis the backend cannot name, placed from their wheels (by vehicle index). */
	TMap<int32, FChassis> WheelPlaced;
	TArray<FRubbleVehicleVisual> VehicleDefs;
	TArray<FRubbleSoftVisual> SoftDefs;
	TMap<uint64, FWheel> Wheels;    // (soft, vehicle, wheel)
	int32 PlayerSoft = INDEX_NONE;
	FTransform PlayerSoftPose = FTransform::Identity;
	UPROPERTY()
	TArray<TObjectPtr<USceneComponent>> Owned;
	FDelegateHandle FrameHandle, SceneHandle;
};
