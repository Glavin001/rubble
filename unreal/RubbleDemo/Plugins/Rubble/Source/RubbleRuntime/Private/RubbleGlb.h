// Reads the render meshes of a glTF binary (.glb): every node's triangles in the scene's frame
// (node transforms applied), converted from Rubble's frame (Y up, right-handed, metres) to
// Unreal's (Z up, left-handed, centimetres) with the C ABI's own axis map (engine-ffi coords.rs:
// Unreal X = -Rubble Z, Y = Rubble X, Z = Rubble Y) and the winding flip a reflection needs.
// Rubble's C ABI names the node that renders each destructible chunk; this supplies its mesh.

#pragma once

#include "CoreMinimal.h"
#include "RubbleTypes.h"

namespace RubbleGlb
{
	struct FFile
	{
		/** Per node: its mesh in the scene frame, Unreal space (empty when the node has none). */
		TArray<FRubbleMeshData> NodeMeshes;
		/** Written by Shardwright (scenes[].extras.generator "prefracture ..."). */
		bool bShardwright = false;
	};

	/** Parse Bytes; false and Error on anything malformed or unsupported. */
	bool Read(const TArray<uint8>& Bytes, FFile& Out, FString& Error);

	/** Which surface a glTF material name reads as (brick, timber, glass..., interior faces). */
	ERubbleMaterial MaterialOf(const FString& Name);

	/** Rubble frame -> Unreal: a point in metres to centimetres. */
	inline FVector3f ToUnreal(const FVector3f& P) { return FVector3f(-P.Z, P.X, P.Y) * 100.0f; }
}
