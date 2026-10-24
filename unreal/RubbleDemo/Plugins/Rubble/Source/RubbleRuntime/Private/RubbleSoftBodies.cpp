#include "RubbleSoftBodies.h"

#include "Engine/GameInstance.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "ProceduralMeshComponent.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

namespace
{
	enum ESurface : int32 { SurfacePaint, SurfaceGlass, SurfaceTrim, SurfaceCount };

	/** Which surface a car triangle is, by its rest position: glass in the greenhouse, trim low down. */
	ESurface SurfaceOf(const FVector3f& A, const FVector3f& B, const FVector3f& C, bool bCar)
	{
		if (!bCar)
		{
			return SurfacePaint;
		}
		const FVector3f M = (A + B + C) / 3.0f;
		if (M.Z > 100.0f && M.Z < 136.0f)
		{
			return SurfaceGlass;
		}
		if (M.Z < 34.0f)
		{
			return SurfaceTrim;
		}
		return SurfacePaint;
	}
}

ARubbleSoftBodies::ARubbleSoftBodies()
{
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ARubbleSoftBodies::BeginPlay()
{
	Super::BeginPlay();
	if (URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>())
	{
		FrameHandle = Rubble->OnFrame.AddUObject(this, &ARubbleSoftBodies::OnFrame);
		SceneHandle = Rubble->OnSceneLoaded.AddUObject(this, &ARubbleSoftBodies::Rebuild);
		if (const FRubbleSceneAssets* A = Rubble->Assets())
		{
			Rebuild(*A);
		}
	}
}

void ARubbleSoftBodies::EndPlay(const EEndPlayReason::Type Reason)
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

void ARubbleSoftBodies::CreateSection(FBody& Body, int32 s, const TArray<uint32>& Global)
{
	FSection& S = Body.Sections[s];
	S.Indices.Reset();
	if (S.Vertices.Num() == 0)
	{
		return; // a crate has no glass or trim
	}
	TMap<int32, int32> Local;
	for (int32 i = 0; i < S.Vertices.Num(); ++i)
	{
		Local.Add(S.Vertices[i], i);
	}
	for (const int32 t : S.Triangles)
	{
		if (Body.Torn.IsValidIndex(t) && Body.Torn[t])
		{
			continue;
		}
		for (int32 k = 0; k < 3; ++k)
		{
			S.Indices.Add(Local[Global[3 * t + k]]);
		}
	}
	Body.Mesh->CreateMeshSection_LinearColor(s, S.Positions, S.Indices, S.Normals, S.UVs, S.Colors, S.Tangents, /*bCreateCollision=*/false);
}

void ARubbleSoftBodies::Rebuild(const FRubbleSceneAssets& Assets)
{
	for (FBody& B : Bodies)
	{
		if (B.Mesh)
		{
			B.Mesh->DestroyComponent();
		}
	}
	Bodies.Reset();
	Topology.Reset();
	for (const FRubbleSoftVisual& V : Assets.SoftBodies)
	{
		FBody& Body = Bodies.AddDefaulted_GetRef();
		const FRubbleMeshData& M = Assets.Meshes[V.Mesh];
		Topology.Add(M.Indices);
		Body.NumVertices = M.Positions.Num();
		Body.Torn.Init(false, M.NumTriangles());
		Body.Mesh = NewObject<UProceduralMeshComponent>(this);
		Body.Mesh->SetMobility(EComponentMobility::Movable);
		Body.Mesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		Body.Mesh->bUseAsyncCooking = false;
		Body.Mesh->SetCastShadow(true);
		Body.Mesh->SetupAttachment(RootComponent);
		Body.Mesh->RegisterComponent();
		Body.Mesh->SetVisibility(false); // until its first skin
		const bool bCar = M.Material == ERubbleMaterial::CarPaint;
		Body.Sections.SetNum(SurfaceCount);
		TArray<TMap<int32, int32>> Seen;
		Seen.SetNum(SurfaceCount);
		for (int32 t = 0; t < M.NumTriangles(); ++t)
		{
			const uint32 a = M.Indices[3 * t], b = M.Indices[3 * t + 1], c = M.Indices[3 * t + 2];
			const ESurface Surf = SurfaceOf(M.Positions[a], M.Positions[b], M.Positions[c], bCar);
			FSection& S = Body.Sections[Surf];
			S.Triangles.Add(t);
			for (const uint32 v : { a, b, c })
			{
				if (!Seen[Surf].Contains(v))
				{
					Seen[Surf].Add(v, S.Vertices.Add(v));
					S.Positions.Add(FVector(M.Positions[v]));
					S.Normals.Add(FVector(M.Normals.IsValidIndex(v) ? M.Normals[v] : FVector3f::UpVector));
					S.UVs.Add(FVector2D(M.UVs.IsValidIndex(v) ? M.UVs[v] : FVector2f::ZeroVector));
					S.Colors.Add(FLinearColor::Black);
				}
			}
		}
		for (int32 s = 0; s < SurfaceCount; ++s)
		{
			CreateSection(Body, s, M.Indices);
		}
		Body.Mesh->SetMaterial(SurfacePaint, bCar ? static_cast<UMaterialInterface*>(RubbleRender::CarPaint(this, V.Paint)) : RubbleRender::Surface(M.Material));
		Body.Mesh->SetMaterial(SurfaceGlass, RubbleRender::Surface(ERubbleMaterial::Glass));
		Body.Mesh->SetMaterial(SurfaceTrim, RubbleRender::Surface(ERubbleMaterial::Rubber));
	}
}

void ARubbleSoftBodies::OnFrame(const FRubbleFrame& Frame)
{
	for (const FRubbleSoftSkin& Skin : Frame.Skins)
	{
		if (!Bodies.IsValidIndex(Skin.Soft))
		{
			continue;
		}
		FBody& Body = Bodies[Skin.Soft];
		if (Skin.Positions.Num() != Body.NumVertices)
		{
			UE_LOG(LogRubble, Error, TEXT("soft body %d: skin has %d vertices, topology %d"), Skin.Soft, Skin.Positions.Num(), Body.NumVertices);
			continue;
		}
		// A tear changes the index buffers: rebuild the sections it touched.
		TBitArray<> TornSections(false, SurfaceCount);
		if (Skin.TornTriangles.Num() == Body.Torn.Num())
		{
			for (int32 s = 0; s < SurfaceCount; ++s)
			{
				for (const int32 t : Body.Sections[s].Triangles)
				{
					if (Skin.TornTriangles[t] != Body.Torn[t])
					{
						TornSections[s] = true;
						break;
					}
				}
			}
			Body.Torn = Skin.TornTriangles;
		}
		for (int32 s = 0; s < SurfaceCount; ++s)
		{
			FSection& S = Body.Sections[s];
			for (int32 i = 0; i < S.Vertices.Num(); ++i)
			{
				const int32 v = S.Vertices[i];
				S.Positions[i] = FVector(Skin.Positions[v]);
				S.Normals[i] = FVector(Skin.Normals.IsValidIndex(v) ? Skin.Normals[v] : FVector3f::UpVector);
				const float Strain = Skin.Strain.IsValidIndex(v) ? Skin.Strain[v] : 0.0f;
				S.Colors[i] = FLinearColor(Strain, 0.0f, 0.0f, 1.0f);
			}
			if (S.Vertices.Num() == 0)
			{
				continue;
			}
			if (TornSections[s])
			{
				CreateSection(Body, s, Topology[Skin.Soft]);
			}
			else
			{
				Body.Mesh->UpdateMeshSection_LinearColor(s, S.Positions, S.Normals, S.UVs, S.Colors, S.Tangents);
			}
		}
		if (!Body.bShown)
		{
			Body.Mesh->SetVisibility(true);
			Body.bShown = true;
		}
	}
}

int32 ARubbleSoftBodies::NumDrawnVertices(int32 Soft) const
{
	if (!Bodies.IsValidIndex(Soft))
	{
		return 0;
	}
	TSet<int32> All;
	for (const FSection& S : Bodies[Soft].Sections)
	{
		All.Append(S.Vertices);
	}
	return All.Num();
}

int32 ARubbleSoftBodies::NumTopologyVertices(int32 Soft) const
{
	return Bodies.IsValidIndex(Soft) ? Bodies[Soft].NumVertices : 0;
}

int32 ARubbleSoftBodies::NumTornTriangles(int32 Soft) const
{
	return Bodies.IsValidIndex(Soft) ? Bodies[Soft].Torn.CountSetBits() : 0;
}
