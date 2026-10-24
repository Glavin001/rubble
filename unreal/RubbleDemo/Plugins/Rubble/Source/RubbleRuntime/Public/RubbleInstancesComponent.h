// Bodies as GPU-only instances in Unreal's GPU Scene (the pattern of Unreal's NiagaraNanite): an
// instanced static mesh whose instance transforms a compute pass writes every frame from pose
// buffers on the GPU, so the bodies stay ordinary instances to Unreal (Nanite, virtual shadow
// maps, motion vectors, GPU culling). See unreal/DESIGN.md, section 2.

#pragma once

#include "CoreMinimal.h"
#include "Components/StaticMeshComponent.h"
#include "RenderGraphResources.h"
#include "RubbleInstancesComponent.generated.h"

class FRDGBuilder;

/** The buffers one frame's write reads, made on the render thread (RDG). */
struct FRubbleInstanceBuffers
{
	FRDGBufferSRVRef CurrentPoses = nullptr;  // float4 x2 per pose slot: Rubble's 32-byte records
	FRDGBufferSRVRef PreviousPoses = nullptr;
	FRDGBufferSRVRef InstanceBody = nullptr;  // uint per instance: its pose slot (0xffffffff hidden)
	FRDGBufferSRVRef InstanceLocal = nullptr; // float4 x4 per instance: translation, rotation, scale, tint
};

UCLASS()
class RUBBLERUNTIME_API URubbleInstancesComponent : public UStaticMeshComponent
{
	GENERATED_BODY()

public:
	using FBufferSource = TFunction<FRubbleInstanceBuffers(FRDGBuilder&)>;

	URubbleInstancesComponent(const FObjectInitializer& ObjectInitializer);

	virtual FPrimitiveSceneProxy* CreateSceneProxy() override;
	virtual FPrimitiveSceneProxy* CreateStaticMeshSceneProxy(Nanite::FMaterialAudit& NaniteMaterials, bool bCreateNanite) override;
	virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;
	virtual bool ShouldCreatePhysicsState() const override { return false; }

	/** Instances to allocate (recreates the render state when it grows). */
	void SetNumInstances(int32 Count);
	int32 GetNumInstances() const { return NumInstances; }
	/** Where the instances can be (culling bounds of the whole set, world space). */
	FBox WorldBounds = FBox(FVector(-100000.0), FVector(100000.0));

	/** Queue this frame's GPU write: Source makes the buffers on the render thread. */
	void WriteFromPoses(float Alpha, FBufferSource Source);

	/** Whether the proxy Unreal made is a Nanite one (the mesh has Nanite data and Nanite is on). */
	bool IsNaniteProxy() const;

private:
	int32 NumInstances = 0;
};
