#include "RubblePrimitives.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/StaticMesh.h"
#include "HAL/PlatformTime.h"
#include "HAL/IConsoleManager.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RubbleGpuPoses.h"
#include "RubbleInstancesComponent.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

namespace
{
	/** Batches past this share of moving instances are rewritten whole, in one call. */
	constexpr float WholeBatchShare = 0.25f;


}

ARubblePrimitives::ARubblePrimitives()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PostPhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ARubblePrimitives::BeginPlay()
{
	Super::BeginPlay();
	if (URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>())
	{
		FrameHandle = Rubble->OnFrame.AddUObject(this, &ARubblePrimitives::OnFrame);
		SceneHandle = Rubble->OnSceneLoaded.AddUObject(this, &ARubblePrimitives::Rebuild);
		if (const FRubbleSceneAssets* Assets = Rubble->Assets())
		{
			Rebuild(*Assets);
		}
	}
}

void ARubblePrimitives::EndPlay(const EEndPlayReason::Type Reason)
{
	if (UGameInstance* GI = GetGameInstance())
	{
		if (URubbleSubsystem* Rubble = GI->GetSubsystem<URubbleSubsystem>())
		{
			Rubble->OnFrame.Remove(FrameHandle);
			Rubble->OnSceneLoaded.Remove(SceneHandle);
		}
	}
	Super::EndPlay(Reason);
}

int32 ARubblePrimitives::BatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets)
{
	// Capsules of each size get their own mesh (a scaled capsule is not a capsule).
	const uint32 Dims = Kind == ERubbleShape::Capsule ? GetTypeHash(FVector2f(A, B)) : 0u;
	const uint64 Key = (uint64(Kind) << 56) ^ (uint64(uint8(Material)) << 48) ^ (uint64(uint32(Mesh + 1)) << 20) ^ Dims;
	if (const int32* Found = BatchByKey.Find(Key))
	{
		return *Found;
	}
	UStaticMesh* StaticMesh = MeshFor(Kind, Mesh, Material, A, B, Assets, /*bNanite=*/false);
	UInstancedStaticMeshComponent* ISM = NewObject<UInstancedStaticMeshComponent>(this);
	ISM->SetStaticMesh(StaticMesh);
	ISM->SetMaterial(0, RubbleRender::Surface(Material));
	ISM->SetMobility(EComponentMobility::Movable);
	ISM->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	ISM->NumCustomDataFloats = 3; // tint
	ISM->SetCastShadow(true);
	ISM->bAffectDistanceFieldLighting = false;
	ISM->SetupAttachment(RootComponent);
	ISM->RegisterComponent();
	FBatch Batch;
	Batch.Component = ISM;
	const int32 Index = Batches.Add(Batch);
	BatchByKey.Add(Key, Index);
	return Index;
}

UStaticMesh* ARubblePrimitives::MeshFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets, bool bNanite)
{
	switch (Kind)
	{
	case ERubbleShape::Box: return bNanite ? RubbleRender::NaniteCube() : RubbleRender::EngineCube();
	case ERubbleShape::Sphere: return bNanite ? RubbleRender::NaniteSphere() : RubbleRender::EngineSphere();
	case ERubbleShape::Capsule:
	{
		FRubbleMeshData Capsule;
		RubbleRender::CapsuleMesh(A, B, Capsule);
		Capsule.Material = Material;
		return RubbleRender::BuildMesh(this, Capsule);
	}
	case ERubbleShape::Hull:
	{
		const uint64 MeshKey = uint64(Mesh) << 8 | uint8(Material);
		if (TObjectPtr<UStaticMesh>* Built = BuiltMeshes.Find(MeshKey))
		{
			return *Built;
		}
		if (Assets.Meshes.IsValidIndex(Mesh))
		{
			FRubbleMeshData Copy = Assets.Meshes[Mesh];
			Copy.Material = Material;
			Copy.TriangleMaterials.Reset();
			UStaticMesh* Built = RubbleRender::BuildMesh(this, Copy);
			BuiltMeshes.Add(MeshKey, Built);
			return Built;
		}
		return nullptr;
	}
	}
	return nullptr;
}

void ARubblePrimitives::Rebuild(const FRubbleSceneAssets& Assets)
{
	for (FBatch& B : Batches)
	{
		if (B.Component)
		{
			B.Component->DestroyComponent();
		}
	}
	Batches.Reset();
	BatchByKey.Reset();
	for (FGpuBatch& B : GpuBatches)
	{
		if (B.Component)
		{
			B.Component->DestroyComponent();
		}
	}
	GpuBatches.Reset();
	GpuBatchByKey.Reset();
	GpuBodies.Reset();
	BuiltMeshes.Reset();
	Refs.Reset();
	BodyInstances.Reset();
	Active.Reset();
	Settling.Reset();
	SceneAssets = &Assets;

	// Instances are added per batch in one call; record each shape's slot first.
	TArray<TArray<FTransform>> NewTransforms;
	TArray<TArray<float>> NewCustom;
	for (const FRubbleBodyVisual& Body : Assets.Bodies)
	{
		if (Body.NumShapes == 0)
		{
			continue; // drawn by another renderer (vehicle chassis, destructible proxy)
		}
		if (DrawsOnGpu(Body, Assets.Shapes, Assets))
		{
			AddGpuBody(Body, Assets.Shapes, Assets);
			continue;
		}
		FBodyState State;
		State.Previous = State.Current = Body.Initial;
		State.First = Refs.Num();
		for (int32 s = 0; s < Body.NumShapes; ++s)
		{
			const FRubbleShapeVisual& Shape = Assets.Shapes[Body.FirstShape + s];
			FInstanceRef Ref;
			FVector Scale(1.0);
			switch (Shape.Kind)
			{
			case ERubbleShape::Box: Scale = FVector(Shape.HalfExtents) / 50.0; break;
			case ERubbleShape::Sphere: Scale = FVector(Shape.Radius / 50.0); break;
			default: break;
			}
			Ref.Batch = BatchFor(Shape.Kind, Shape.Mesh, Shape.Material, Shape.Radius, Shape.HalfHeight, Assets);
			Ref.Local = FTransform(Shape.Local.GetRotation(), Shape.Local.GetLocation(), Scale);
			if (NewTransforms.Num() <= Ref.Batch)
			{
				NewTransforms.SetNum(Ref.Batch + 1);
				NewCustom.SetNum(Ref.Batch + 1);
			}
			Ref.Instance = NewTransforms[Ref.Batch].Add(Ref.Local * Body.Initial);
			NewCustom[Ref.Batch].Append({ Shape.Tint.R, Shape.Tint.G, Shape.Tint.B });
			Refs.Add(Ref);
		}
		State.Num = Body.NumShapes;
		BodyInstances.Add(Body.Id.Index(), State);
	}
	for (int32 b = 0; b < NewTransforms.Num(); ++b)
	{
		FBatch& Batch = Batches[b];
		Batch.Transforms = NewTransforms[b];
		Batch.Component->AddInstances(NewTransforms[b], /*bShouldReturnIndices=*/false, /*bWorldSpace=*/false);
		for (int32 i = 0; i < NewTransforms[b].Num(); ++i)
		{
			Batch.Component->SetCustomData(i, TArrayView<const float>(&NewCustom[b][3 * i], 3), false);
		}
		Batch.Component->MarkRenderStateDirty();
	}
	UE_LOG(LogRubble, Display, TEXT("primitives: %d instances through GPU Scene (%d batches), %d read back (%d batches)"),
		NumGpuInstances(), GpuBatches.Num(), NumInstances() - NumGpuInstances(), Batches.Num());
}

void ARubblePrimitives::AddLiveBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes)
{
	if (!SceneAssets || Body.NumShapes == 0 || BodyInstances.Contains(Body.Id.Index()) || GpuBodies.Contains(Body.Id.Index()))
	{
		return;
	}
	if (DrawsOnGpu(Body, Shapes, *SceneAssets))
	{
		AddGpuBody(Body, Shapes, *SceneAssets);
		return;
	}
	FBodyState State;
	State.Previous = State.Current = Body.Initial;
	State.First = Refs.Num();
	State.Num = Body.NumShapes;
	for (int32 s = 0; s < Body.NumShapes; ++s)
	{
		const FRubbleShapeVisual& Shape = Shapes[Body.FirstShape + s];
		FVector Scale(1.0);
		if (Shape.Kind == ERubbleShape::Box)
		{
			Scale = FVector(Shape.HalfExtents) / 50.0;
		}
		else if (Shape.Kind == ERubbleShape::Sphere)
		{
			Scale = FVector(Shape.Radius / 50.0);
		}
		FInstanceRef Ref;
		Ref.Batch = BatchFor(Shape.Kind, Shape.Mesh, Shape.Material, Shape.Radius, Shape.HalfHeight, *SceneAssets);
		Ref.Local = FTransform(Shape.Local.GetRotation(), Shape.Local.GetLocation(), Scale);
		FBatch& Batch = Batches[Ref.Batch];
		Ref.Instance = Batch.Component->AddInstance(Ref.Local * Body.Initial, false);
		Batch.Transforms.SetNum(FMath::Max(Batch.Transforms.Num(), Ref.Instance + 1));
		Batch.Transforms[Ref.Instance] = Ref.Local * Body.Initial;
		const float Tint[3] = { Shape.Tint.R, Shape.Tint.G, Shape.Tint.B };
		Batch.Component->SetCustomData(Ref.Instance, TArrayView<const float>(Tint, 3), true);
		Refs.Add(Ref);
	}
	BodyInstances.Add(Body.Id.Index(), State);
}

void ARubblePrimitives::OnFrame(const FRubbleFrame& Frame)
{
	for (const FRubbleBodyVisual& B : Frame.NewBodies)
	{
		AddLiveBody(B, Frame.NewShapes);
	}
	// Bodies moving before this frame and not in it are now at rest: one last draw there.
	TSet<uint32> Moved;
	Moved.Reserve(Frame.Moves.Num());
	for (const FRubbleMove& M : Frame.Moves)
	{
		FBodyState* State = BodyInstances.Find(M.Body.Index());
		if (!State)
		{
			continue;
		}
		State->Previous = M.bHasPrevious ? M.Previous : M.Transform;
		State->Current = M.Transform;
		Moved.Add(M.Body.Index());
	}
	for (const uint32 b : Active)
	{
		if (!Moved.Contains(b))
		{
			FBodyState& State = BodyInstances[b];
			State.Previous = State.Current;
			Settling.Add(b);
		}
	}
	Active = MoveTemp(Moved);
}

void ARubblePrimitives::Apply(int32 BodyIndex, const FTransform& Pose)
{
	const FBodyState& State = BodyInstances[BodyIndex];
	for (int32 r = 0; r < State.Num; ++r)
	{
		const FInstanceRef& Ref = Refs[State.First + r];
		FBatch& Batch = Batches[Ref.Batch];
		Batch.Transforms[Ref.Instance] = Ref.Local * Pose;
		++Batch.Dirty;
	}
}

void ARubblePrimitives::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>();
	const double Start = FPlatformTime::Seconds();
	if (const FRubbleGpuPoses* Poses = GpuPoses(); Poses && GpuBatches.Num())
	{
		WriteGpu(*Poses);
	}
	if (!Rubble || (Active.Num() == 0 && Settling.Num() == 0))
	{
		UpdateMs = 1000.0 * (FPlatformTime::Seconds() - Start);
		return;
	}
	const double Alpha = Rubble->Alpha();
	for (const uint32 b : Active)
	{
		const FBodyState& State = BodyInstances[b];
		Apply(b, RubbleLerpPose(State.Previous, State.Current, Alpha));
	}
	for (const uint32 b : Settling)
	{
		Apply(b, BodyInstances[b].Current);
	}
	// Write each batch: whole when much of it moved, else instance by instance.
	TArray<bool> Touched;
	Touched.Init(false, Batches.Num());
	for (int32 i = 0; i < Batches.Num(); ++i)
	{
		FBatch& Batch = Batches[i];
		if (Batch.Dirty > 0 && Batch.Dirty >= WholeBatchShare * Batch.Transforms.Num())
		{
			Batch.Component->BatchUpdateInstancesTransforms(0, Batch.Transforms, /*bWorldSpace=*/false, /*bMarkRenderStateDirty=*/true, /*bTeleport=*/true);
			Batch.Dirty = -1; // done
		}
	}
	auto WriteBody = [this, &Touched](uint32 b) {
		const FBodyState& State = BodyInstances[b];
		for (int32 r = 0; r < State.Num; ++r)
		{
			const FInstanceRef& Ref = Refs[State.First + r];
			FBatch& Batch = Batches[Ref.Batch];
			if (Batch.Dirty > 0)
			{
				Batch.Component->UpdateInstanceTransform(Ref.Instance, Batch.Transforms[Ref.Instance], false, false, true);
				Touched[Ref.Batch] = true;
			}
		}
	};
	for (const uint32 b : Active)
	{
		WriteBody(b);
	}
	for (const uint32 b : Settling)
	{
		WriteBody(b);
	}
	for (int32 i = 0; i < Batches.Num(); ++i)
	{
		if (Touched[i])
		{
			Batches[i].Component->MarkRenderStateDirty();
		}
		Batches[i].Dirty = 0;
	}
	Settling.Reset();
	UpdateMs = 1000.0 * (FPlatformTime::Seconds() - Start);
}

int32 ARubblePrimitives::NumInstances() const
{
	int32 N = NumGpuInstances();
	for (const FBatch& B : Batches)
	{
		N += B.Component ? B.Component->GetInstanceCount() : 0;
	}
	return N;
}

int32 ARubblePrimitives::NumGpuInstances() const
{
	int32 N = 0;
	for (const FGpuBatch& B : GpuBatches)
	{
		N += B.PoseSlot.Num();
	}
	return N;
}

// ----- the GPU Scene path ----------------------------------------------------------------------

FRubbleGpuPoses* ARubblePrimitives::GpuPoses() const
{
	const URubbleSubsystem* Rubble = GetGameInstance() ? GetGameInstance()->GetSubsystem<URubbleSubsystem>() : nullptr;
	return Rubble && Rubble->Backend() ? Rubble->Backend()->GpuPoses() : nullptr;
}

bool ARubblePrimitives::DrawsOnGpu(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets) const
{
	const FRubbleGpuPoses* Poses = GpuPoses();
	if (!Poses || Body.bStatic || !RubbleGpuSceneEnabled())
	{
		return false;
	}
	// Large bodies stay on CPU-visible instances so Lumen's world-space tracing sees them.
	const float MaxCm = RubbleGpuSceneMaxExtentCm();
	for (int32 s = 0; s < Body.NumShapes; ++s)
	{
		const FRubbleShapeVisual& Shape = Shapes[Body.FirstShape + s];
		float Extent = 0.0f;
		switch (Shape.Kind)
		{
		case ERubbleShape::Box: Extent = Shape.HalfExtents.GetMax(); break;
		case ERubbleShape::Sphere: Extent = Shape.Radius; break;
		case ERubbleShape::Capsule: Extent = Shape.Radius + Shape.HalfHeight; break;
		case ERubbleShape::Hull:
			if (Assets.Meshes.IsValidIndex(Shape.Mesh))
			{
				for (const FVector3f& P : Assets.Meshes[Shape.Mesh].Positions)
				{
					Extent = FMath::Max(Extent, P.GetAbsMax());
				}
			}
			break;
		}
		if (Extent + Shape.Local.GetLocation().GetAbsMax() > MaxCm)
		{
			return false;
		}
	}
	return Poses->PoseIndex(Body.Id) != MAX_uint32;
}

int32 ARubblePrimitives::GpuBatchFor(ERubbleShape Kind, int32 Mesh, ERubbleMaterial Material, float A, float B, const FRubbleSceneAssets& Assets)
{
	const uint32 Dims = Kind == ERubbleShape::Capsule ? GetTypeHash(FVector2f(A, B)) : 0u;
	const uint64 Key = (uint64(Kind) << 56) ^ (uint64(uint8(Material)) << 48) ^ (uint64(uint32(Mesh + 1)) << 20) ^ Dims;
	if (const int32* Found = GpuBatchByKey.Find(Key))
	{
		return *Found;
	}
	URubbleInstancesComponent* C = NewObject<URubbleInstancesComponent>(this);
	C->SetStaticMesh(MeshFor(Kind, Mesh, Material, A, B, Assets, /*bNanite=*/true));
	C->SetMaterial(0, RubbleRender::Surface(Material));
	C->SetCastShadow(true);
	// Where the instances can be: the bounds GPU culling starts from.
	const double Half = FMath::Max(2.0 * Assets.GroundHalfExtentCm, 50000.0);
	C->WorldBounds = FBox(FVector(-Half, -Half, -Half), FVector(Half, Half, Half));
	C->SetupAttachment(RootComponent);
	C->RegisterComponent();
	FGpuBatch Batch;
	Batch.Component = C;
	const int32 Index = GpuBatches.Add(MoveTemp(Batch));
	GpuBatchByKey.Add(Key, Index);
	return Index;
}

void ARubblePrimitives::AddGpuBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes, const FRubbleSceneAssets& Assets)
{
	const uint32 Slot = GpuPoses()->PoseIndex(Body.Id);
	const FTransform Offset = GpuPoses()->PoseOffset(Body.Id);
	for (int32 s = 0; s < Body.NumShapes; ++s)
	{
		const FRubbleShapeVisual& Shape = Shapes[Body.FirstShape + s];
		FVector3f Scale(1.0f);
		if (Shape.Kind == ERubbleShape::Box)
		{
			Scale = Shape.HalfExtents / 50.0f;
		}
		else if (Shape.Kind == ERubbleShape::Sphere)
		{
			Scale = FVector3f(Shape.Radius / 50.0f);
		}
		FGpuBatch& Batch = GpuBatches[GpuBatchFor(Shape.Kind, Shape.Mesh, Shape.Material, Shape.Radius, Shape.HalfHeight, Assets)];
		// The instance in its pose record's frame: shape local x (origin relative to the record).
		const FTransform InRecord = FTransform(Shape.Local.GetRotation(), Shape.Local.GetLocation()) * Offset;
		const FQuat4f Q(InRecord.GetRotation());
		Batch.PoseSlot.Add(Slot);
		Batch.Local.Add(FVector4f(FVector3f(InRecord.GetLocation()), 0.0f));
		Batch.Local.Add(FVector4f(Q.X, Q.Y, Q.Z, Q.W));
		Batch.Local.Add(FVector4f(Scale, 0.0f));
		Batch.Local.Add(FVector4f(Shape.Tint.R, Shape.Tint.G, Shape.Tint.B, 0.0f));
		Batch.bDirty = true;
	}
	GpuBodies.Add(Body.Id.Index());
	GpuPoses()->DrawnOnGpu(Body.Id);
}

void ARubblePrimitives::WriteGpu(const FRubbleGpuPoses& Poses)
{
	for (FGpuBatch& Batch : GpuBatches)
	{
		if (Batch.PoseSlot.Num() == 0)
		{
			continue;
		}
		if (Batch.bDirty)
		{
			Batch.Component->SetInstanceTables(Batch.PoseSlot, Batch.Local);
			Batch.bDirty = false;
		}
		Batch.Component->WriteFromGpuPoses(Poses);
	}
}

bool ARubblePrimitives::BodyTransform(FRubbleBodyId Body, FTransform& Out) const
{
	const FBodyState* State = BodyInstances.Find(Body.Index());
	if (!State || State->Num == 0)
	{
		return false;
	}
	const FInstanceRef& Ref = Refs[State->First];
	return Batches[Ref.Batch].Component->GetInstanceTransform(Ref.Instance, Out, false);
}
