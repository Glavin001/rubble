#include "RubbleInstancesComponent.h"

#include "Engine/InstancedStaticMesh.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "GPUSceneWriter.h"
#include "InstanceDataSceneProxy.h"
#include "InstancedStaticMeshSceneProxyDesc.h"
#include "NaniteSceneProxy.h"
#include "PrimitiveSceneDesc.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RubbleGpuPoses.h"
#include "RubbleShaders.h"
#include "SceneInterface.h"

namespace
{
	class FRubbleNaniteProxy : public Nanite::FSceneProxy
	{
	public:
		FRubbleNaniteProxy(Nanite::FMaterialAudit& Materials, const FInstancedStaticMeshSceneProxyDesc& Desc)
			: Nanite::FSceneProxy(Materials, Desc)
		{
			bAlwaysHasVelocity = true;
		}
	};

	class FRubbleMeshProxy : public FInstancedStaticMeshSceneProxy
	{
	public:
		FRubbleMeshProxy(const FInstancedStaticMeshSceneProxyDesc& Desc, ERHIFeatureLevel::Type FeatureLevel)
			: FInstancedStaticMeshSceneProxy(Desc, FeatureLevel)
		{
			bAlwaysHasVelocity = true;
		}
	};
}

URubbleInstancesComponent::URubbleInstancesComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	// As NiagaraNanite: these need instance data on the CPU (see DESIGN.md, the Lumen hybrid).
	bAffectDynamicIndirectLighting = false;
	bAffectDistanceFieldLighting = false;
	BodyInstance.bSimulatePhysics = false;
	SetGenerateOverlapEvents(false);
	SetCollisionEnabled(ECollisionEnabled::NoCollision);
	bNavigationRelevant = false;
	SetCanEverAffectNavigation(false);
	Mobility = EComponentMobility::Movable;
}

FPrimitiveSceneProxy* URubbleInstancesComponent::CreateSceneProxy()
{
	return NumInstances > 0 ? Super::CreateSceneProxy() : nullptr;
}

FBoxSphereBounds URubbleInstancesComponent::CalcBounds(const FTransform& LocalToWorld) const
{
	return FBoxSphereBounds(WorldBounds);
}

FPrimitiveSceneProxy* URubbleInstancesComponent::CreateStaticMeshSceneProxy(Nanite::FMaterialAudit& NaniteMaterials, bool bCreateNanite)
{
	UWorld* World = GetWorld();
	if (!World || !World->Scene || !UseGPUScene(World->Scene->GetShaderPlatform()))
	{
		return nullptr;
	}
	FInstanceSceneDataBuffers Buffers(/*bInstanceDataIsGPUOnly=*/true);
	{
		FInstanceSceneDataBuffers::FAccessTag Tag(PointerHash(this));
		FInstanceSceneDataBuffers::FWriteView View = Buffers.BeginWriteAccess(Tag);
		Buffers.SetPrimitiveLocalToWorld(GetComponentTransform().ToMatrixWithScale(), Tag);
		View.NumCustomDataFloats = CustomFloats;
		View.InstanceLocalBounds.SetNum(1);
		View.InstanceLocalBounds[0] = GetStaticMesh()->GetBounds();
		View.Flags.bHasPerInstanceCustomData = true;
		View.Flags.bHasPerInstanceDynamicData = true;
		View.NumInstancesGPUOnly = NumInstances;
		Buffers.EndWriteAccess(Tag);
		Buffers.ValidateData();
	}
	FInstancedStaticMeshSceneProxyDesc Desc;
	Desc.InitializeFromStaticMeshComponent(this);
	Desc.InstanceDataSceneProxy = MakeShared<FInstanceDataSceneProxy, ESPMode::ThreadSafe>(MoveTemp(Buffers));
	Desc.bUseGpuLodSelection = true;
	if (bCreateNanite)
	{
		return ::new FRubbleNaniteProxy(NaniteMaterials, Desc);
	}
	return ::new FRubbleMeshProxy(Desc, World->GetFeatureLevel());
}

void URubbleInstancesComponent::SetNumInstances(int32 Count)
{
	if (Count != NumInstances)
	{
		NumInstances = Count;
		RecreateRenderState_Concurrent();
	}
}

bool URubbleInstancesComponent::IsNaniteProxy() const
{
	const FPrimitiveSceneProxy* Proxy = GetSceneProxy();
	return Proxy && Proxy->IsNaniteMesh();
}

bool URubbleInstancesComponent::WriteFromPoses(float Alpha, FBufferSource Source)
{
	UWorld* World = GetWorld();
	if (!World || !World->Scene || NumInstances <= 0 || !GetSceneProxy())
	{
		return false;
	}
	FPrimitiveSceneDesc PrimitiveDesc;
	PrimitiveDesc.SceneProxy = GetSceneProxy();
	const uint32 Count = uint32(NumInstances);
	const uint32 Rows = uint32(RowsPerInstance()), Floats = uint32(CustomFloats);
	World->Scene->UpdatePrimitiveInstancesFromCompute(&PrimitiveDesc,
		FGPUSceneWriteDelegate::CreateLambda([Count, Rows, Floats, Alpha, Source = MoveTemp(Source)](FRDGBuilder& GraphBuilder, const FGPUSceneWriteDelegateParams& Params)
		{
			const FRubbleInstanceBuffers Buffers = Source(GraphBuilder);
			FRubbleGPUSceneCS::FParameters* P = GraphBuilder.AllocParameters<FRubbleGPUSceneCS::FParameters>();
			P->GPUSceneWriterParameters = Params.GPUWriteParams;
			P->NumInstances = Count;
			P->PrimitiveId = Params.PersistentPrimitiveId;
			P->Alpha = Alpha;
			P->LocalRows = Rows;
			P->NumCustomFloats = Floats;
			P->CurrentPoses = Buffers.CurrentPoses;
			P->PreviousPoses = Buffers.PreviousPoses;
			P->InstanceBody = Buffers.InstanceBody;
			P->InstanceLocal = Buffers.InstanceLocal;
			TShaderMapRef<FRubbleGPUSceneCS> Shader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("RubbleGPUScene(%u)", Count), Shader, P,
				FComputeShaderUtils::GetGroupCountWrapped(Count, FRubbleGPUSceneCS::ThreadGroupSize));
		}));
	return true;
}

void URubbleInstancesComponent::SetInstanceTables(TArray<uint32> PoseSlot, TArray<FVector4f> Local)
{
	check(Local.Num() == PoseSlot.Num() * RowsPerInstance());
	TSharedPtr<FTableData, ESPMode::ThreadSafe> Data = MakeShared<FTableData, ESPMode::ThreadSafe>();
	Data->Version = TableData ? TableData->Version + 1 : 1;
	Data->PoseSlot = MoveTemp(PoseSlot);
	Data->Local = MoveTemp(Local);
	SetNumInstances(Data->PoseSlot.Num());
	TableData = Data;
}

bool URubbleInstancesComponent::WriteFromGpuPoses(const FRubbleGpuPoses& Poses)
{
	const FRubbleGpuPoses::FView& View = Poses.Latest();
	if (!View.bValid || !TableData || TableData->PoseSlot.Num() == 0)
	{
		return false;
	}
	return WriteFromPoses(View.Alpha, [Tables = Tables, Data = TableData, Current = View.Current, Previous = View.Previous](FRDGBuilder& G)
	{
		if (Tables->Version != Data->Version)
		{
			Tables->PoseSlot = G.ConvertToExternalBuffer(CreateStructuredBuffer(G, TEXT("Rubble.InstancePoseSlot"), Data->PoseSlot));
			Tables->Local = G.ConvertToExternalBuffer(CreateStructuredBuffer(G, TEXT("Rubble.InstanceLocal"), Data->Local));
			Tables->Version = Data->Version;
		}
		FRubbleInstanceBuffers Out;
		Out.CurrentPoses = G.CreateSRV(G.RegisterExternalBuffer(Current));
		Out.PreviousPoses = G.CreateSRV(G.RegisterExternalBuffer(Previous));
		Out.InstanceBody = G.CreateSRV(G.RegisterExternalBuffer(Tables->PoseSlot));
		Out.InstanceLocal = G.CreateSRV(G.RegisterExternalBuffer(Tables->Local));
		return Out;
	});
}
