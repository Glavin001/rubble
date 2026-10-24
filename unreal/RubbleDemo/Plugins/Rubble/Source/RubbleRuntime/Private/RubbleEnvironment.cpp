#include "RubbleEnvironment.h"

#include "Components/DirectionalLightComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "Components/PostProcessComponent.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/VolumetricCloudComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"
#include "UObject/ConstructorHelpers.h"

ARubbleEnvironment::ARubbleEnvironment()
{
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));

	Sun = CreateDefaultSubobject<UDirectionalLightComponent>(TEXT("Sun"));
	Sun->SetupAttachment(RootComponent);
	Sun->SetMobility(EComponentMobility::Movable);
	Sun->SetAtmosphereSunLight(true);
	Sun->SetIntensity(9.0f);
	Sun->SetLightSourceAngle(0.6f);
	Sun->SetDynamicShadowCascades(4);
	Sun->bCastCloudShadows = true;
	Sun->SetVolumetricScatteringIntensity(1.0f);

	Atmosphere = CreateDefaultSubobject<USkyAtmosphereComponent>(TEXT("Atmosphere"));
	Atmosphere->SetupAttachment(RootComponent);

	SkyLight = CreateDefaultSubobject<USkyLightComponent>(TEXT("SkyLight"));
	SkyLight->SetupAttachment(RootComponent);
	SkyLight->SetMobility(EComponentMobility::Movable);
	SkyLight->bRealTimeCapture = true;
	SkyLight->bLowerHemisphereIsBlack = false;
	SkyLight->SetLowerHemisphereColor(FLinearColor(0.25f, 0.24f, 0.22f));

	Fog = CreateDefaultSubobject<UExponentialHeightFogComponent>(TEXT("Fog"));
	Fog->SetupAttachment(RootComponent);
	Fog->SetFogDensity(0.012f);
	Fog->SetFogHeightFalloff(0.08f);
	Fog->SetVolumetricFog(true);
	Fog->SetVolumetricFogScatteringDistribution(0.7f);
	Fog->SetVolumetricFogExtinctionScale(0.25f);
	Fog->SetVolumetricFogDistance(15000.0f);

	Clouds = CreateDefaultSubobject<UVolumetricCloudComponent>(TEXT("Clouds"));
	Clouds->SetupAttachment(RootComponent);
	Clouds->SetLayerBottomAltitude(2.5f);
	Clouds->SetLayerHeight(5.0f);
	static ConstructorHelpers::FObjectFinderOptional<UMaterialInterface> CloudMaterial(TEXT("/Engine/EngineSky/VolumetricClouds/m_SimpleVolumetricCloud_Inst.m_SimpleVolumetricCloud_Inst"));
	if (CloudMaterial.Get())
	{
		Clouds->SetMaterial(CloudMaterial.Get());
	}

	PostProcess = CreateDefaultSubobject<UPostProcessComponent>(TEXT("PostProcess"));
	PostProcess->SetupAttachment(RootComponent);
	PostProcess->bUnbound = true;
	FPostProcessSettings& P = PostProcess->Settings;
	P.bOverride_AutoExposureBias = true;
	P.AutoExposureBias = 0.3f;
	P.bOverride_BloomIntensity = true;
	P.BloomIntensity = 0.45f;
	P.bOverride_AmbientOcclusionIntensity = true;
	P.AmbientOcclusionIntensity = 0.6f;
	P.bOverride_MotionBlurAmount = true;
	P.MotionBlurAmount = 0.35f;
	P.bOverride_VignetteIntensity = true;
	P.VignetteIntensity = 0.3f;
	P.bOverride_FilmGrainIntensity = true;
	P.FilmGrainIntensity = 0.04f;
	P.bOverride_SceneFringeIntensity = true;
	P.SceneFringeIntensity = 0.15f;

	Ground = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Ground"));
	Ground->SetupAttachment(RootComponent);
	Ground->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	static ConstructorHelpers::FObjectFinderOptional<UStaticMesh> Plane(TEXT("/Engine/BasicShapes/Plane.Plane"));
	if (Plane.Get())
	{
		Ground->SetStaticMesh(Plane.Get());
	}
}

void ARubbleEnvironment::BeginPlay()
{
	Super::BeginPlay();
	if (FParse::Param(FCommandLine::Get(), TEXT("RubbleNoVolumetrics")))
	{
		Clouds->SetVisibility(false);
		Fog->SetVolumetricFog(false);
	}
	if (URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>())
	{
		SceneHandle = Rubble->OnSceneLoaded.AddUObject(this, &ARubbleEnvironment::ApplyStage);
		if (const FRubbleSceneAssets* A = Rubble->Assets())
		{
			ApplyStage(*A);
		}
	}
}

void ARubbleEnvironment::EndPlay(const EEndPlayReason::Type Reason)
{
	if (UGameInstance* GI = GetGameInstance())
	{
		if (URubbleSubsystem* Rubble = GI->GetSubsystem<URubbleSubsystem>())
		{
			Rubble->OnSceneLoaded.Remove(SceneHandle);
		}
	}
	Super::EndPlay(Reason);
}

UInstancedStaticMeshComponent* ARubbleEnvironment::Dressing(ERubbleMaterial Material, bool bCylinder)
{
	UInstancedStaticMeshComponent* ISM = NewObject<UInstancedStaticMeshComponent>(this);
	ISM->SetStaticMesh(bCylinder ? RubbleRender::EngineCylinder() : RubbleRender::EngineCube());
	ISM->SetMaterial(0, RubbleRender::Surface(Material));
	// Movable: the environment's root is (the sun turns per stage), and static cannot attach to it.
	ISM->SetMobility(EComponentMobility::Movable);
	ISM->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	ISM->NumCustomDataFloats = 3;
	ISM->SetupAttachment(RootComponent);
	ISM->RegisterComponent();
	DressingParts.Add(ISM);
	return ISM;
}

void ARubbleEnvironment::ApplyStage(const FRubbleSceneAssets& Assets)
{
	for (UInstancedStaticMeshComponent* C : DressingParts)
	{
		if (C)
		{
			C->DestroyComponent();
		}
	}
	DressingParts.Reset();

	// Ground: the engine plane is 100 cm square.
	const float Half = Assets.GroundHalfExtentCm;
	Ground->SetVisibility(Half > 0.0f);
	Ground->SetWorldScale3D(FVector(FMath::Max(Half, 100.0f) * 2.0f / 100.0f));
	Ground->SetMaterial(0, RubbleRender::Ground(Assets.Stage));

	auto Add = [](UInstancedStaticMeshComponent* ISM, const FVector& Center, const FVector& HalfExtents, const FLinearColor& Tint, float Yaw = 0.0f) {
		const int32 i = ISM->AddInstance(FTransform(FRotator(0, Yaw, 0), Center, HalfExtents / 50.0), false);
		const float Data[3] = { Tint.R, Tint.G, Tint.B };
		ISM->SetCustomData(i, TArrayView<const float>(Data, 3), false);
	};
	FRandomStream Random(int32(Assets.Stage) * 31 + 7);
	float SunPitch = -35.0f, SunYaw = 40.0f;
	switch (Assets.Stage)
	{
	case ERubbleStage::Yard:
	{
		// Kerbs around the yard and light poles at the corners.
		UInstancedStaticMeshComponent* Kerbs = Dressing(ERubbleMaterial::Concrete);
		UInstancedStaticMeshComponent* Poles = Dressing(ERubbleMaterial::Steel, true);
		const float Edge = FMath::Max(1500.0f, Half * 0.85f);
		for (int32 Side = 0; Side < 4; ++Side)
		{
			for (int32 k = -10; k <= 10; ++k)
			{
				const float Along = k * Edge / 10.0f;
				const FVector C = Side < 2 ? FVector(Along, (Side ? 1 : -1) * Edge, 8) : FVector((Side == 3 ? 1 : -1) * Edge, Along, 8);
				Add(Kerbs, C, Side < 2 ? FVector(Edge / 10.0f * 0.48f, 15, 8) : FVector(15, Edge / 10.0f * 0.48f, 8), (k & 1) ? FLinearColor(0.8f, 0.8f, 0.78f) : FLinearColor(0.7f, 0.12f, 0.08f));
			}
			const FVector Corner((Side & 1 ? 1 : -1) * Edge, (Side & 2 ? 1 : -1) * Edge, 600);
			Add(Poles, Corner, FVector(12, 12, 600), FLinearColor(0.4f, 0.4f, 0.42f));
		}
		SunPitch = -38.0f;
		SunYaw = 35.0f;
		break;
	}
	case ERubbleStage::CrashHall:
	{
		// A crash-test track: barrier walls along both sides, lane lines, a backdrop wall.
		UInstancedStaticMeshComponent* Barriers = Dressing(ERubbleMaterial::Concrete);
		UInstancedStaticMeshComponent* Lines = Dressing(ERubbleMaterial::Plastic);
		for (int32 k = -12; k <= 12; ++k)
		{
			Add(Barriers, FVector(k * 400.0, -2400, 60), FVector(195, 30, 60), FLinearColor(0.75f, 0.74f, 0.7f));
			Add(Barriers, FVector(k * 400.0, 2400, 60), FVector(195, 30, 60), FLinearColor(0.75f, 0.74f, 0.7f));
			Add(Lines, FVector(k * 400.0, -350, 0.6), FVector(120, 6, 0.5), FLinearColor(0.9f, 0.85f, 0.2f));
			Add(Lines, FVector(k * 400.0, 350, 0.6), FVector(120, 6, 0.5), FLinearColor(0.9f, 0.85f, 0.2f));
		}
		for (int32 k = -6; k <= 6; ++k)
		{
			// Black and white calibration squares on the backdrop.
			Add(Barriers, FVector(5200, k * 400.0, 300), FVector(40, 200, 300), (k & 1) ? FLinearColor(0.85f, 0.85f, 0.85f) : FLinearColor(0.05f, 0.05f, 0.05f));
		}
		SunPitch = -22.0f; // low evening sun: long shadows across the track
		SunYaw = 110.0f;
		break;
	}
	case ERubbleStage::Lot:
	{
		// A city lot: kerbs and a backdrop of blocks.
		UInstancedStaticMeshComponent* Blocks = Dressing(ERubbleMaterial::Concrete);
		UInstancedStaticMeshComponent* Kerbs = Dressing(ERubbleMaterial::Concrete);
		for (int32 i = 0; i < 26; ++i)
		{
			const float Angle = UE_TWO_PI * i / 26.0f + Random.FRandRange(-0.08f, 0.08f);
			const float Dist = Random.FRandRange(5500.0f, 7500.0f);
			const FVector Size(Random.FRandRange(500, 1200), Random.FRandRange(500, 1200), Random.FRandRange(800, 4000));
			Add(Blocks, FVector(Dist * FMath::Cos(Angle), Dist * FMath::Sin(Angle), Size.Z), Size, FLinearColor(0.6f, 0.58f, 0.55f) * Random.FRandRange(0.7f, 1.1f), FMath::RadiansToDegrees(Angle));
		}
		for (int32 k = -15; k <= 15; ++k)
		{
			Add(Kerbs, FVector(k * 200.0, -2000, 10), FVector(98, 20, 10), FLinearColor(0.7f, 0.7f, 0.68f));
			Add(Kerbs, FVector(k * 200.0, 2000, 10), FVector(98, 20, 10), FLinearColor(0.7f, 0.7f, 0.68f));
		}
		SunPitch = -48.0f;
		SunYaw = -30.0f;
		break;
	}
	}
	// The lot is mostly seen from above, ground filling the view: auto exposure would lift it to
	// white, so it is biased down there.
	PostProcess->Settings.AutoExposureBias = Assets.Stage == ERubbleStage::Lot ? -1.3f : 0.3f;
	Sun->SetWorldRotation(FRotator(SunPitch, SunYaw, 0));
	SkyLight->RecaptureSky();
}
