#include "RubbleGpuScenePoc.h"

#include "Engine/StaticMesh.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RubbleInstancesComponent.h"
#include "RubbleRender.h"
#include "RubbleShaders.h"
#include "RubbleSubsystem.h"

struct ARubbleGpuScenePoc::FGpu
{
	TRefCountPtr<FRDGPooledBuffer> Body, Local;
};

ARubbleGpuScenePoc::ARubbleGpuScenePoc()
{
	PrimaryActorTick.bCanEverTick = true;
	Instances = CreateDefaultSubobject<URubbleInstancesComponent>(TEXT("Instances"));
	RootComponent = Instances;
}

void ARubbleGpuScenePoc::BeginPlay()
{
	Super::BeginPlay();
	FParse::Value(FCommandLine::Get(), TEXT("RubblePocSide="), Side);
	Side = FMath::Clamp(Side, 1, 2048);
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
	Gpu = MakeShared<FGpu, ESPMode::ThreadSafe>();
	Instances->SetNumInstances(Side * Side);
	UE_LOG(LogRubble, Display, TEXT("GPU Scene PoC: %d boxes (%s)"), Side * Side, bNaniteMesh ? TEXT("Nanite mesh") : TEXT("engine cube"));
}

namespace
{
	FRDGBufferRef AnimatePoses(FRDGBuilder& G, uint32 N, uint32 Side, float Time, const TCHAR* Name)
	{
		FRDGBufferRef Poses = G.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(FVector4f), N * 2), Name);
		FRubblePocAnimateCS::FParameters* P = G.AllocParameters<FRubblePocAnimateCS::FParameters>();
		P->NumBodies = N;
		P->Side = Side;
		P->Time = Time;
		P->OutPoses = G.CreateUAV(Poses);
		TShaderMapRef<FRubblePocAnimateCS> Shader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
		FComputeShaderUtils::AddPass(G, RDG_EVENT_NAME("RubblePocAnimate(%u)", N), Shader, P,
			FComputeShaderUtils::GetGroupCountWrapped(N, FRubblePocAnimateCS::ThreadGroupSize));
		return Poses;
	}
}

void ARubbleGpuScenePoc::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	Time += DeltaSeconds;
	const uint32 N = uint32(Side * Side);
	Instances->WriteFromPoses(1.0f, [Gpu = Gpu, N, Side = uint32(Side), Time = Time, Prev = Time - DeltaSeconds](FRDGBuilder& G)
	{
		if (!Gpu->Body)
		{
			// The per-instance tables, as the renderer would build them at load: one box per body.
			TArray<uint32> Body;
			TArray<FVector4f> Local;
			Body.SetNum(N);
			Local.SetNum(N * 4);
			for (uint32 i = 0; i < N; ++i)
			{
				Body[i] = i;
				const FLinearColor Tint = FLinearColor::MakeFromHSV8(uint8(float(i % 97) / 97.0f * 255), 140, 220);
				Local[i * 4 + 0] = FVector4f(0, 0, 0, 0);          // translation (body frame)
				Local[i * 4 + 1] = FVector4f(0, 0, 0, 1);          // rotation
				Local[i * 4 + 2] = FVector4f(0.5f, 0.5f, 0.5f, 0); // scale: the cube is 100 cm
				Local[i * 4 + 3] = FVector4f(Tint.R, Tint.G, Tint.B, 0);
			}
			Gpu->Body = G.ConvertToExternalBuffer(CreateStructuredBuffer(G, TEXT("Rubble.PocBody"), Body));
			Gpu->Local = G.ConvertToExternalBuffer(CreateStructuredBuffer(G, TEXT("Rubble.PocLocal"), Local));
		}
		FRubbleInstanceBuffers Out;
		Out.CurrentPoses = G.CreateSRV(AnimatePoses(G, N, Side, Time, TEXT("Rubble.PocCurrent")));
		Out.PreviousPoses = G.CreateSRV(AnimatePoses(G, N, Side, Prev, TEXT("Rubble.PocPrevious")));
		Out.InstanceBody = G.CreateSRV(G.RegisterExternalBuffer(Gpu->Body));
		Out.InstanceLocal = G.CreateSRV(G.RegisterExternalBuffer(Gpu->Local));
		return Out;
	});
}
