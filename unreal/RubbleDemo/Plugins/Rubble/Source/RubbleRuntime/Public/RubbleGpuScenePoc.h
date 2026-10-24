// Proof of concept for the GPU Scene path (no Rubble): thousands of Nanite boxes whose transforms
// a compute pass writes into GPU Scene each frame from a pose buffer in Rubble's record format
// (32 B, Rubble's frame). Verifies Nanite, virtual shadow maps and Lumen lighting before Rubble's
// own buffers feed it. Run with -RubblePoc=gpuscene.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleGpuScenePoc.generated.h"

class URubbleInstancesComponent;

UCLASS()
class RUBBLERUNTIME_API ARubbleGpuScenePoc : public AActor
{
	GENERATED_BODY()

public:
	ARubbleGpuScenePoc();
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaSeconds) override;

	UPROPERTY(VisibleAnywhere)
	TObjectPtr<URubbleInstancesComponent> Instances;

	int32 Side = 100; // Side x Side boxes
	bool UsesNaniteMesh() const { return bNaniteMesh; }

private:
	TArray<FVector4f> Current, Previous;
	TArray<uint32> Body;
	TArray<FVector4f> Local;
	float Time = 0.0f;
	bool bNaniteMesh = false;
};
