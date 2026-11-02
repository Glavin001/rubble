// What the renderer needs from a vibe-city ScenePack (v2): each node's collider as drawn and its
// material. The physics reads the same file through rbCreateDestructibleAssetFromPack; this
// reads only the geometry and looks, from the pack's own fields:
//   scenario.nodeColliders[i]  {kind: cuboid, halfExtents} | {kind: shape, shape: k} | {kind: convex_hull, points}
//   scenario.shapeLibrary[k]   {kind: convex_hull, points}   (shared by many nodes)
//   scenario.nodeMaterials[i]  a name in defaults.solver.materials (color, roughness, metalness)
//   scenario.nodeGroups[i], nodeTypes[i]  the building a node belongs to and what it is
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
		/** From the catalog: the texture-array layer (-1: none) and the tile size, and the layer its
		 *  up-facing faces take. */
		int32 Layer = -1;
		float MetresPerTile = 1.0f;
		int32 UpLayer = -1;
		float UpMetresPerTile = 1.0f;
		bool bUpUntinted = false;
	};

	struct FNode
	{
		/** A box (HalfExtents, Rubble's frame) or a hull: Shape indexes FPack::Hulls. */
		bool bBox = true;
		FVector3f HalfExtents = FVector3f(0.5f);
		int32 Hull = INDEX_NONE;
		int32 Material = 0;
		FVector3f Centroid = FVector3f::ZeroVector;
		/** A glazed facade: a tower's walls, columns and slabs (nodeGroups tower@..., all but the
		 *  foundation; its facade is fractured across all three). The pack models the towers
		 *  without openings; this client draws windows on them. */
		bool bGlazed = false;
		/** For a glazed node: the facades its bounds reach, as bits in Unreal's axes (1 +X,
		 *  2 -X, 4 +Y, 8 -Y): the sides of its building's bounds it lies on. */
		uint8 Facades = 0;
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

	/** Give each material its layers from vibe-land's catalog, as vibe-land's renderer does:
	 *   - the wall layer: the catalog entry whose id is the material's textureKey, else the entry
	 *     that lists the material's name;
	 *   - the up-facing layer: the same for metal, limestone, slate and timber, worn concrete
	 *     floor for the rest (cityTextures.ts layerCodeForTextureKey).
	 *  Then the look this client adds where vibe-land draws a flat colour (LookFor in the .cpp):
	 *  timber framing gets slate on its up faces (the roofs), painted trim and joinery painted
	 *  plaster, asphalt dark worn concrete. Materials stay tint-only when the catalog is missing. */
	void ApplyCatalog(const TArray<uint8>& CatalogJson, FPack& Pack);

	/** A hull's triangle mesh in Unreal's frame (cm, Unreal winding), flat-shaded. */
	bool HullMesh(const TArray<FVector3f>& RubblePoints, FRubbleMeshData& Out);

	/** Rubble's frame to Unreal's: X = -z, Y = x, Z = y, x100 (the C ABI's RB_COORDINATES_UNREAL). */
	inline FVector3f ToUnreal(const FVector3f& P) { return FVector3f(-P.Z, P.X, P.Y) * 100.0f; }
	/** A box's half extents (unsigned) in Unreal's frame. */
	inline FVector3f HalfToUnreal(const FVector3f& H) { return FVector3f(H.Z, H.X, H.Y) * 100.0f; }
}
