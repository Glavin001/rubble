// Draws soft bodies from the skin each frame brings (the soft_skin kernel's output: positions,
// normals, strain, face strain, torn triangles). One procedural mesh per body, split into sections
// by surface once at load: the engine's car mesh has its own (paint, glass, chrome, lamps...), a
// mock shell is split by height. Each frame rewrites vertex positions, normals and the strain
// colour; a tear or a shattered pane rebuilds the index buffers of the sections it touched.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"
#include "RubbleTypes.h"
#include "RubbleSoftBodies.generated.h"

struct FRubbleFrame;
struct FRubbleSceneAssets;

UCLASS()
class RUBBLERUNTIME_API ARubbleSoftBodies : public AActor
{
	GENERATED_BODY()

public:
	ARubbleSoftBodies();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;

	/** The vertices drawn for soft body Soft (sum over sections), and its topology's: tests. */
	int32 NumDrawnVertices(int32 Soft) const;
	int32 NumTopologyVertices(int32 Soft) const;
	int32 NumTornTriangles(int32 Soft) const;
	/** Glass triangles that shattered (strain past 6%; no longer drawn). */
	int32 NumShatteredPanes(int32 Soft) const;
	/** Sections drawn: one per car surface for the engine's car, three for a mock shell. */
	int32 NumSections(int32 Soft) const;

private:
	struct FSection
	{
		TArray<int32> Vertices;  // global vertex of each section vertex
		TArray<int32> Triangles; // global triangle of each section triangle
		TArray<int32> Indices;   // section-local indices
		TArray<FVector> Positions;
		TArray<FVector> Normals;
		TArray<FVector2D> UVs;
		TArray<FVector2D> UV1s; // the engine car's rest z (UV1.x)
		TArray<FLinearColor> Colors;
		TArray<int32> TriangleCorners; // section vertex of each corner of each section triangle
		TArray<FProcMeshTangent> Tangents;
	};
	struct FBody
	{
		TObjectPtr<UProceduralMeshComponent> Mesh;
		TArray<FSection> Sections;
		TBitArray<> Torn;
		TBitArray<> Shattered; // glass triangles past GlassShatter
		TBitArray<> bGlass;
		bool bEngineCar = false; // the engine's car mesh: a section per car surface
		int32 NumVertices = 0;
		bool bShown = false;
	};

	void Rebuild(const FRubbleSceneAssets& Assets);
	void OnFrame(const FRubbleFrame& Frame);
	void CreateSection(FBody& Body, int32 s, const TArray<uint32>& GlobalIndices);

	TArray<FBody> Bodies;
	TArray<TArray<uint32>> Topology; // global indices per body
	FDelegateHandle FrameHandle, SceneHandle;
};
