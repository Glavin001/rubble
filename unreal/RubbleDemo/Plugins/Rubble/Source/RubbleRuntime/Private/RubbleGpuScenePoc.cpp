#include "RubbleGpuScenePoc.h"

#include "Engine/StaticMesh.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RubbleInstancesComponent.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

ARubbleGpuScenePoc::ARubbleGpuScenePoc()
{
	PrimaryActorTick.bCanEverTick = true;
	Instances = CreateDefaultSubobject<URubbleInstancesComponent>(TEXT("Instances"));
	RootComponent = Instances;
}

void ARubbleGpuScenePoc::BeginPlay()
{
	Super::BeginPlay();
	UStaticMesh* Mesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Game/Rubble/Meshes/SM_NaniteBox.SM_NaniteBox"), nullptr, LOAD_NoWarn | LOAD_Quiet);
	bNaniteMesh = Mesh != nullptr;
	if (!Mesh)
	{
		UE_LOG(LogRubble, Warning, TEXT("GPU Scene PoC: no Nanite box (scripts/unreal/build-content.sh); using the engine cube"));
		Mesh = RubbleRender::EngineCube();
	}
	Instances->SetStaticMesh(Mesh);
	Instances->SetMaterial(0, RubbleRender::Surface(ERubbleMaterial::Brick));
	Instances->WorldBounds = FBox(FVector(-Side * 150.0, -Side * 150.0, -100.0), FVector(Side * 150.0, Side * 150.0, 3000.0));
	const int32 N = Side * Side;
	Current.SetNumZeroed(N * 2);
	Previous.SetNumZeroed(N * 2);
	Body.SetNum(N);
	Local.SetNum(N * 4);
	for (int32 i = 0; i < N; ++i)
	{
		Body[i] = uint32(i);
		const float Hue = float(i % 97) / 97.0f;
		const FLinearColor Tint = FLinearColor::MakeFromHSV8(uint8(Hue * 255), 140, 220);
		Local[i * 4 + 0] = FVector4f(0, 0, 0, 0);        // translation (body frame)
		Local[i * 4 + 1] = FVector4f(0, 0, 0, 1);        // rotation
		Local[i * 4 + 2] = FVector4f(0.5f, 0.5f, 0.5f, 0); // scale: the cube is 100 cm
		Local[i * 4 + 3] = FVector4f(Tint.R, Tint.G, Tint.B, 0);
	}
	Instances->SetNumInstances(N);
}

void ARubbleGpuScenePoc::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	Time += DeltaSeconds;
	Previous = Current;
	// Poses in Rubble's frame (Y up, metres), as Rubble writes them: a field of spinning,
	// bobbing boxes.
	for (int32 i = 0; i < Side * Side; ++i)
	{
		const float X = (i % Side - Side * 0.5f) * 1.5f, Z = (i / Side - Side * 0.5f) * 1.5f;
		const float Phase = 0.37f * (i % 13) + 0.11f * (i % 7);
		const float Y = 1.0f + 0.8f * FMath::Abs(FMath::Sin(1.5f * Time + Phase));
		const FQuat4f Q(FVector3f(0.3f, 1.0f, 0.2f).GetSafeNormal(), 1.2f * Time + Phase);
		Current[i * 2 + 0] = FVector4f(X, Y, Z, 0);
		Current[i * 2 + 1] = FVector4f(Q.X, Q.Y, Q.Z, Q.W);
	}
	if (Previous.Num() != Current.Num() || Time <= DeltaSeconds)
	{
		Previous = Current;
	}
	Instances->WriteFromPoses(1.0f, [Cur = Current, Prev = Previous, B = Body, L = Local](FRDGBuilder& G)
	{
		FRubbleInstanceBuffers Out;
		Out.CurrentPoses = G.CreateSRV(CreateStructuredBuffer(G, TEXT("Rubble.PocCurrent"), Cur));
		Out.PreviousPoses = G.CreateSRV(CreateStructuredBuffer(G, TEXT("Rubble.PocPrevious"), Prev));
		Out.InstanceBody = G.CreateSRV(CreateStructuredBuffer(G, TEXT("Rubble.PocBody"), B));
		Out.InstanceLocal = G.CreateSRV(CreateStructuredBuffer(G, TEXT("Rubble.PocLocal"), L));
		return Out;
	});
}
