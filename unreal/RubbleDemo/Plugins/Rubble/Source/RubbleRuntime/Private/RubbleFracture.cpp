#include "RubbleFracture.h"

#include "Components/StaticMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/StaticMesh.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

ARubbleFracture::ARubbleFracture()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PostPhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ARubbleFracture::BeginPlay()
{
	Super::BeginPlay();
	if (URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>())
	{
		FrameHandle = Rubble->OnFrame.AddUObject(this, &ARubbleFracture::OnFrame);
		SceneHandle = Rubble->OnSceneLoaded.AddUObject(this, &ARubbleFracture::Rebuild);
		if (const FRubbleSceneAssets* A = Rubble->Assets())
		{
			Rebuild(*A);
		}
	}
}

void ARubbleFracture::EndPlay(const EEndPlayReason::Type Reason)
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

namespace
{
	UStaticMeshComponent* NewMeshComponent(AActor* Owner, UStaticMesh* Mesh, const FTransform& World)
	{
		UStaticMeshComponent* C = NewObject<UStaticMeshComponent>(Owner);
		C->SetStaticMesh(Mesh);
		C->SetMobility(EComponentMobility::Movable);
		C->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		C->SetupAttachment(Owner->GetRootComponent());
		C->RegisterComponent();
		C->SetWorldTransform(World);
		return C;
	}
}

void ARubbleFracture::Rebuild(const FRubbleSceneAssets& InAssets)
{
	for (UStaticMeshComponent* C : ChunkComponents)
	{
		if (C)
		{
			C->DestroyComponent();
		}
	}
	for (UStaticMeshComponent* C : Proxies)
	{
		if (C)
		{
			C->DestroyComponent();
		}
	}
	ChunkMeshes.Reset();
	ChunkComponents.Reset();
	Proxies.Reset();
	ChunkRest.Reset();
	ChunkBody.Reset();
	BodyChunks.Reset();
	Poses.Reset();
	Active.Reset();
	Settling.Reset();
	Assets = &InAssets;

	const int32 NumChunks = InAssets.Chunks.Num();
	ChunkMeshes.SetNum(NumChunks);
	ChunkComponents.SetNum(NumChunks);
	ChunkRest.SetNum(NumChunks);
	ChunkBody.SetNum(NumChunks);
	for (int32 c = 0; c < NumChunks; ++c)
	{
		const FRubbleChunkVisual& V = InAssets.Chunks[c];
		ChunkRest[c] = V.Rest;
		if (InAssets.Meshes.IsValidIndex(V.Mesh))
		{
			ChunkMeshes[c] = RubbleRender::BuildMesh(this, InAssets.Meshes[V.Mesh]);
		}
	}
	for (int32 d = 0; d < InAssets.Destructibles.Num(); ++d)
	{
		const FRubbleDestructibleVisual& D = InAssets.Destructibles[d];
		// The root proxy: every chunk merged into the destructible's frame.
		FRubbleMeshData Merged;
		Merged.Material = ERubbleMaterial::Concrete;
		for (const int32 c : D.Chunks)
		{
			const FRubbleMeshData& M = InAssets.Meshes[InAssets.Chunks[c].Mesh];
			const FTransform3f Rest(ChunkRest[c]);
			const uint32 Base = Merged.Positions.Num();
			for (int32 v = 0; v < M.Positions.Num(); ++v)
			{
				Merged.Positions.Add(Rest.TransformPosition(M.Positions[v]));
				Merged.Normals.Add(Rest.TransformVectorNoScale(M.Normals.IsValidIndex(v) ? M.Normals[v] : FVector3f::UpVector));
				Merged.UVs.Add(M.UVs.IsValidIndex(v) ? M.UVs[v] : FVector2f::ZeroVector);
			}
			for (const uint32 i : M.Indices)
			{
				Merged.Indices.Add(Base + i);
			}
			for (int32 t = 0; t < M.NumTriangles(); ++t)
			{
				Merged.TriangleMaterials.Add(M.TriangleMaterials.IsValidIndex(t) ? M.TriangleMaterials[t] : M.Material);
			}
			ChunkBody[c] = InAssets.Chunks[c].Body.IsNull() ? D.RootBody.Index() : InAssets.Chunks[c].Body.Index();
		}
		UStaticMesh* ProxyMesh = RubbleRender::BuildMesh(this, Merged);
		Proxies.Add(NewMeshComponent(this, ProxyMesh, D.Placement));
		for (const int32 c : D.Chunks)
		{
			BodyChunks.FindOrAdd(ChunkBody[c]).Add(c);
			Poses.Add(ChunkBody[c], { D.Placement, D.Placement });
		}
	}
}

bool ARubbleFracture::IsProxyShown(int32 Destructible) const
{
	return Proxies.IsValidIndex(Destructible) && Proxies[Destructible] && Proxies[Destructible]->IsVisible();
}

void ARubbleFracture::Shatter(int32 Destructible)
{
	if (!IsProxyShown(Destructible))
	{
		return;
	}
	Proxies[Destructible]->SetVisibility(false);
	for (const int32 c : Assets->Destructibles[Destructible].Chunks)
	{
		const FBodyPose* Pose = Poses.Find(ChunkBody[c]);
		const FTransform World = ChunkRest[c] * (Pose ? Pose->Current : Assets->Destructibles[Destructible].Placement);
		ChunkComponents[c] = NewMeshComponent(this, ChunkMeshes[c], World);
	}
}

void ARubbleFracture::OnFrame(const FRubbleFrame& Frame)
{
	if (!Assets || Assets->Destructibles.Num() == 0)
	{
		return;
	}
	// Splits first: this frame's moves may belong to bodies a split made.
	for (const FRubbleEvent& E : Frame.Events)
	{
		if (E.Type != ERubbleEventType::Split || !Assets->Destructibles.IsValidIndex(E.Destructible))
		{
			continue;
		}
		Shatter(E.Destructible);
		const uint32 Parent = E.Parent.Body.Index();
		const FBodyPose ParentPose = Poses.Contains(Parent) ? Poses[Parent] : FBodyPose{ Assets->Destructibles[E.Destructible].Placement, Assets->Destructibles[E.Destructible].Placement };
		BodyChunks.Remove(Parent);
		for (const FRubbleActorInfo& Child : E.Children)
		{
			const uint32 Body = Child.Body.Index();
			BodyChunks.Add(Body, Child.Chunks);
			for (const int32 c : Child.Chunks)
			{
				ChunkBody[c] = Body;
			}
			// A new body starts where its parent was: they share the destructible's frame.
			if (!Poses.Contains(Body))
			{
				Poses.Add(Body, { ParentPose.Current, ParentPose.Current });
			}
		}
	}
	TSet<uint32> Moved;
	for (const FRubbleMove& M : Frame.Moves)
	{
		const uint32 Body = M.Body.Index();
		if (!BodyChunks.Contains(Body))
		{
			continue;
		}
		// Per-chunk bodies (the Bonds model): the first chunk to move breaks up the proxy.
		const TArray<int32>& Chunks = BodyChunks[Body];
		if (Chunks.Num() && Assets->Chunks.IsValidIndex(Chunks[0]) && !Assets->Chunks[Chunks[0]].Body.IsNull())
		{
			FBodyPose& Before = Poses.FindOrAdd(Body);
			Before.Current = M.bHasPrevious ? M.Previous : M.Transform;
			Shatter(Assets->Chunks[Chunks[0]].Destructible);
		}
		FBodyPose& Pose = Poses.FindOrAdd(Body);
		Pose.Previous = M.bHasPrevious ? M.Previous : M.Transform;
		Pose.Current = M.Transform;
		Moved.Add(Body);
	}
	for (const uint32 b : Active)
	{
		if (!Moved.Contains(b))
		{
			if (FBodyPose* Pose = Poses.Find(b))
			{
				Pose->Previous = Pose->Current;
			}
			Settling.Add(b);
		}
	}
	Active = MoveTemp(Moved);
}

void ARubbleFracture::Draw(uint32 Body, const FTransform& Pose)
{
	const TArray<int32>* Chunks = BodyChunks.Find(Body);
	if (!Chunks)
	{
		return;
	}
	for (const int32 c : *Chunks)
	{
		if (UStaticMeshComponent* C = ChunkComponents[c])
		{
			C->SetWorldTransform(ChunkRest[c] * Pose, false, nullptr, ETeleportType::TeleportPhysics);
		}
		else if (Proxies.IsValidIndex(Assets->Chunks[c].Destructible))
		{
			// Still intact: the proxy moves as one.
			Proxies[Assets->Chunks[c].Destructible]->SetWorldTransform(Pose, false, nullptr, ETeleportType::TeleportPhysics);
		}
	}
}

void ARubbleFracture::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>();
	if (!Rubble || !Assets)
	{
		return;
	}
	const double Alpha = Rubble->Alpha();
	for (const uint32 b : Active)
	{
		const FBodyPose& Pose = Poses[b];
		Draw(b, RubbleLerpPose(Pose.Previous, Pose.Current, Alpha));
	}
	for (const uint32 b : Settling)
	{
		Draw(b, Poses[b].Current);
	}
	Settling.Reset();
}

TArray<int32> ARubbleFracture::ChunkDrawCounts() const
{
	TArray<int32> Counts;
	if (!Assets)
	{
		return Counts;
	}
	Counts.Init(0, Assets->Chunks.Num());
	for (int32 d = 0; d < Assets->Destructibles.Num(); ++d)
	{
		if (IsProxyShown(d))
		{
			for (const int32 c : Assets->Destructibles[d].Chunks)
			{
				++Counts[c];
			}
		}
	}
	for (int32 c = 0; c < ChunkComponents.Num(); ++c)
	{
		if (ChunkComponents[c] && ChunkComponents[c]->IsVisible())
		{
			++Counts[c];
		}
	}
	return Counts;
}
