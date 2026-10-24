// What the renderer needs from a vibe-city ScenePack (v2): each node's collider as drawn and its
// material. The physics reads the same file through rbCreateDestructibleAssetFromPack; this
// reads only the geometry and looks, from the pack's own fields:
//   scenario.nodeColliders[i]  {kind: cuboid, halfExtents} | {kind: shape, shape: k} | {kind: convex_hull, points}
//   scenario.shapeLibrary[k]   {kind: convex_hull, points}   (shared by many nodes)
//   scenario.nodeMaterials[i]  a name in defaults.solver.materials (color, roughness, metalness)
// Points and half extents are in the node's frame (centred on its centroid, the pack's axes:
// Rubble's frame, Y up, metres).

#pragma once

#include "CoreMinimal.h"
#include "RubbleTypes.h"

namespace RubbleTownPack
{
	struct FMaterial
	{
		FString Name;
		FLinearColor Color = FLinearColor(0.5f, 0.5f, 0.5f);
		float Roughness = 0.8f;
		float Metalness = 0.0f;
		ERubbleMaterial Kind = ERubbleMaterial::Concrete;
		/** vibe-land's material catalog id (shared/assets/materials/catalog.json), or empty. */
		FString TextureKey;
		/** From the catalog: the texture-array layer (-1: none) and the tile size. */
		int32 Layer = -1;
		float MetresPerTile = 1.0f;
	};

	struct FNode
	{
		/** A box (HalfExtents, Rubble's frame) or a hull: Shape indexes FPack::Hulls. */
		bool bBox = true;
		FVector3f HalfExtents = FVector3f(0.5f);
		int32 Hull = INDEX_NONE;
		int32 Material = 0;
		FVector3f Centroid = FVector3f::ZeroVector;
	};

	struct FPack
	{
		FString Title;
		TArray<FMaterial> Materials;
		/** Hull point sets, Rubble's frame: the library's, then each node's own. */
		TArray<TArray<FVector3f>> Hulls;
		TArray<FNode> Nodes;
	};

	/** Parse the pack's render fields. False and Error for a file that is not a v2 pack. */
	bool Parse(const TArray<uint8>& Json, FPack& Out, FString& Error);

	/** Look each material's textureKey up in vibe-land's catalog (its layer in the copied texture
	 *  arrays and its tile size). Materials stay tint-only when the catalog is missing. */
	void ApplyCatalog(const TArray<uint8>& CatalogJson, FPack& Pack);

	/** A hull's triangle mesh in Unreal's frame (cm, Unreal winding), flat-shaded. */
	bool HullMesh(const TArray<FVector3f>& RubblePoints, FRubbleMeshData& Out);

	/** Rubble's frame to Unreal's: X = -z, Y = x, Z = y, x100 (the C ABI's RB_COORDINATES_UNREAL). */
	inline FVector3f ToUnreal(const FVector3f& P) { return FVector3f(-P.Z, P.X, P.Y) * 100.0f; }
	/** A box's half extents (unsigned) in Unreal's frame. */
	inline FVector3f HalfToUnreal(const FVector3f& H) { return FVector3f(H.Z, H.X, H.Y) * 100.0f; }
}
