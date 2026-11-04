#include "RubbleSoftBodies.h"

#include "Engine/GameInstance.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "ProceduralMeshComponent.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

namespace
{
	/** The mock's shells: paint, glass and trim sections. */
	enum ESurface : int32 { SurfacePaint, SurfaceGlass, SurfaceTrim, SurfaceCount };

	/** A pane that strains past this has shattered (car-render.mjs): it is no longer drawn. */
	constexpr float GlassShatter = 0.06f;

	/** Which surface a mock car triangle is, by its rest position: glass in the greenhouse, trim low down. */
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
		if ((Body.Torn.IsValidIndex(t) && Body.Torn[t]) || (Body.Shattered.IsValidIndex(t) && Body.Shattered[t]))
		{
			continue;
		}
		for (int32 k = 0; k < 3; ++k)
		{
			S.Indices.Add(Local[Global[3 * t + k]]);
		}
	}
	Body.Mesh->CreateMeshSection_LinearColor(s, S.Positions, S.Indices, S.Normals, S.UVs, S.UV1s, {}, {}, S.Colors, S.Tangents, /*bCreateCollision=*/false);
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
		// The engine's car mesh carries its surfaces; the mock's shells are split by height.
		Body.bEngineCar = M.CarSurfaces.Num() == M.NumTriangles() && M.NumTriangles() > 0;
		const int32 NumSections = Body.bEngineCar ? int32(ERubbleCarSurface::Count) : int32(SurfaceCount);
		Body.Sections.SetNum(NumSections);
		Body.Shattered.Init(false, M.NumTriangles());
		Body.bGlass.Init(false, M.NumTriangles());
		TArray<TMap<int32, int32>> Seen;
		Seen.SetNum(NumSections);
		for (int32 t = 0; t < M.NumTriangles(); ++t)
		{
			const uint32 a = M.Indices[3 * t], b = M.Indices[3 * t + 1], c = M.Indices[3 * t + 2];
			const int32 Surf = Body.bEngineCar ? int32(M.CarSurfaces[t]) : int32(SurfaceOf(M.Positions[a], M.Positions[b], M.Positions[c], bCar));
			// Glass and lamp lenses shatter (and are no longer drawn) past GlassShatter.
			Body.bGlass[t] = Body.bEngineCar
				&& (Surf == int32(ERubbleCarSurface::Glass) || Surf == int32(ERubbleCarSurface::Headlight) || Surf == int32(ERubbleCarSurface::Taillight));
			FSection& S = Body.Sections[Surf];
			S.Triangles.Add(t);
			for (const uint32 v : { a, b, c })
			{
				if (!Seen[Surf].Contains(v))
				{
					Seen[Surf].Add(v, S.Vertices.Add(v));
					S.Positions.Add(FVector(M.Positions[v]));
					S.Normals.Add(FVector(M.Normals.IsValidIndex(v) ? M.Normals[v] : FVector3f::UpVector));
					if (Body.bEngineCar)
					{
						// The rest position in metres (UV0 = x, y; UV1.x = z): the materials' damage
						// patterns stay on the panel as it bends.
						const FVector3f R = M.Positions[v] * 0.01f;
						S.UVs.Add(FVector2D(R.X, R.Y));
						S.UV1s.Add(FVector2D(R.Z, 0.0));
					}
					else
					{
						S.UVs.Add(FVector2D(M.UVs.IsValidIndex(v) ? M.UVs[v] : FVector2f::ZeroVector));
					}
					const float Seam = M.Seams.IsValidIndex(v) ? float(M.Seams[v]) : 0.0f;
					S.Colors.Add(FLinearColor(0.0f, Body.bEngineCar ? Seam : 0.0f, 0.0f, 1.0f));
				}
			}
			S.TriangleCorners.Append({ Seen[Surf][a], Seen[Surf][b], Seen[Surf][c] });
		}
		for (int32 s = 0; s < NumSections; ++s)
		{
			CreateSection(Body, s, M.Indices);
		}
		if (Body.bEngineCar)
		{
			for (int32 s = 0; s < NumSections; ++s)
			{
				Body.Mesh->SetMaterial(s, RubbleRender::SoftCarSurface(this, ERubbleCarSurface(s), V.Paint));
			}
		}
		else
		{
			Body.Mesh->SetMaterial(SurfacePaint, bCar ? static_cast<UMaterialInterface*>(RubbleRender::CarPaint(this, V.Paint)) : RubbleRender::Surface(M.Material));
			Body.Mesh->SetMaterial(SurfaceGlass, RubbleRender::Surface(ERubbleMaterial::Glass));
			Body.Mesh->SetMaterial(SurfaceTrim, RubbleRender::Surface(ERubbleMaterial::Rubber));
		}
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
		// A tear, or a pane that shattered, changes the index buffers: rebuild the sections it touched.
		const int32 NumSections = Body.Sections.Num();
		TBitArray<> Rebuilt(false, NumSections);
		const bool bFaces = Skin.FaceStrain.Num() == Body.Torn.Num();
		for (int32 s = 0; s < NumSections; ++s)
		{
			for (const int32 t : Body.Sections[s].Triangles)
			{
				const bool bTorn = Skin.TornTriangles.IsValidIndex(t) && Skin.TornTriangles[t];
				if (bTorn != Body.Torn[t])
				{
					Body.Torn[t] = bTorn;
					Rebuilt[s] = true;
				}
				if (bFaces && Body.bGlass[t] && !Body.Shattered[t] && Skin.FaceStrain[t] > GlassShatter)
				{
					Body.Shattered[t] = true; // glass does not mend
					Rebuilt[s] = true;
				}
			}
		}
		for (int32 s = 0; s < NumSections; ++s)
		{
			FSection& S = Body.Sections[s];
			if (S.Vertices.Num() == 0)
			{
				continue;
			}
			// Colour R: the strain the materials shade by, the vertex's or, sharper, the largest of
			// this surface's triangles around it (a glass crack, a crease in one panel).
			for (int32 i = 0; i < S.Vertices.Num(); ++i)
			{
				const int32 v = S.Vertices[i];
				S.Positions[i] = FVector(Skin.Positions[v]);
				S.Normals[i] = FVector(Skin.Normals.IsValidIndex(v) ? Skin.Normals[v] : FVector3f::UpVector);
				S.Colors[i].R = Skin.Strain.IsValidIndex(v) ? FMath::Clamp(Skin.Strain[v], 0.0f, 1.0f) : 0.0f;
			}
			if (bFaces)
			{
				for (int32 k = 0; k < S.Triangles.Num(); ++k)
				{
					const float F = FMath::Clamp(Skin.FaceStrain[S.Triangles[k]], 0.0f, 1.0f);
					for (int32 c = 0; c < 3; ++c)
					{
						float& R = S.Colors[S.TriangleCorners[3 * k + c]].R;
						R = FMath::Max(R, F);
					}
				}
			}
			if (Rebuilt[s])
			{
				CreateSection(Body, s, Topology[Skin.Soft]);
			}
			else
			{
				Body.Mesh->UpdateMeshSection_LinearColor(s, S.Positions, S.Normals, S.UVs, S.UV1s, {}, {}, S.Colors, S.Tangents, /*bSRGBConversion=*/false);
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

int32 ARubbleSoftBodies::NumShatteredPanes(int32 Soft) const
{
	return Bodies.IsValidIndex(Soft) ? Bodies[Soft].Shattered.CountSetBits() : 0;
}

int32 ARubbleSoftBodies::NumSections(int32 Soft) const
{
	return Bodies.IsValidIndex(Soft) ? Bodies[Soft].Sections.Num() : 0;
}
