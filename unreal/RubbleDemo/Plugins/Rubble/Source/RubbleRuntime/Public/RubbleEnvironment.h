// The look: physical sky and sun, real-time sky light, height and volumetric fog, volumetric clouds,
// post-process, the ground and each stage's dressing (the yard's kerbs and poles, the crash
// track's barriers and lane lines, the lot's backdrop blocks). Rebuilt when a scene loads.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleTypes.h"
#include "RubbleEnvironment.generated.h"

class UDirectionalLightComponent;
class UExponentialHeightFogComponent;
class UPostProcessComponent;
class USkyAtmosphereComponent;
class USkyLightComponent;
class UStaticMeshComponent;
class UVolumetricCloudComponent;
class UInstancedStaticMeshComponent;

UCLASS()
class RUBBLERUNTIME_API ARubbleEnvironment : public AActor
{
	GENERATED_BODY()

public:
	ARubbleEnvironment();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;

	void ApplyStage(const FRubbleSceneAssets& Assets);

	UPROPERTY(VisibleAnywhere)
	TObjectPtr<UDirectionalLightComponent> Sun;
	UPROPERTY(VisibleAnywhere)
	TObjectPtr<USkyAtmosphereComponent> Atmosphere;
	UPROPERTY(VisibleAnywhere)
	TObjectPtr<USkyLightComponent> SkyLight;
	UPROPERTY(VisibleAnywhere)
	TObjectPtr<UExponentialHeightFogComponent> Fog;
	UPROPERTY(VisibleAnywhere)
	TObjectPtr<UVolumetricCloudComponent> Clouds;
	UPROPERTY(VisibleAnywhere)
	TObjectPtr<UPostProcessComponent> PostProcess;
	UPROPERTY(VisibleAnywhere)
	TObjectPtr<UStaticMeshComponent> Ground;

private:
	UInstancedStaticMeshComponent* Dressing(ERubbleMaterial Material, bool bCylinder = false);
	UPROPERTY()
	TArray<TObjectPtr<UInstancedStaticMeshComponent>> DressingParts;
	FDelegateHandle SceneHandle;
};
