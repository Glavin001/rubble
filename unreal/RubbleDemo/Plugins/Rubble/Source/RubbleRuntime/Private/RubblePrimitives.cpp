#include "RubblePrimitives.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/StaticMesh.h"
#include "HAL/PlatformTime.h"
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
	UStaticMesh* StaticMesh = nullptr;
	switch (Kind)
	{
	case ERubbleShape::Box: StaticMesh = RubbleRender::EngineCube(); break;
	case ERubbleShape::Sphere: StaticMesh = RubbleRender::EngineSphere(); break;
	case ERubbleShape::Capsule:
	{
		FRubbleMeshData Capsule;
		RubbleRender::CapsuleMesh(A, B, Capsule);
		Capsule.Material = Material;
		StaticMesh = RubbleRender::BuildMesh(this, Capsule);
		break;
	}
	case ERubbleShape::Hull:
	{
		const uint64 MeshKey = uint64(Mesh) << 8 | uint8(Material);
		if (TObjectPtr<UStaticMesh>* Built = BuiltMeshes.Find(MeshKey))
		{
			StaticMesh = *Built;
		}
		else if (Assets.Meshes.IsValidIndex(Mesh))
		{
			FRubbleMeshData Copy = Assets.Meshes[Mesh];
			Copy.Material = Material;
			Copy.TriangleMaterials.Reset();
			StaticMesh = RubbleRender::BuildMesh(this, Copy);
			BuiltMeshes.Add(MeshKey, StaticMesh);
		}
		break;
	}
	}
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
}

void ARubblePrimitives::AddLiveBody(const FRubbleBodyVisual& Body, TArrayView<const FRubbleShapeVisual> Shapes)
{
	if (!SceneAssets || Body.NumShapes == 0 || BodyInstances.Contains(Body.Id.Index()))
	{
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
	if (!Rubble || (Active.Num() == 0 && Settling.Num() == 0))
	{
		UpdateMs = 0.0;
		return;
	}
	const double Start = FPlatformTime::Seconds();
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
	int32 N = 0;
	for (const FBatch& B : Batches)
	{
		N += B.Component ? B.Component->GetInstanceCount() : 0;
	}
	return N;
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
