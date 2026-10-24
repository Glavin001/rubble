// Shared render helpers: the material library (generated content when built, engine materials
// otherwise) and runtime static meshes from FRubbleMeshData.

#pragma once

#include "CoreMinimal.h"
#include "RubbleTypes.h"

class UMaterialInterface;
class UMaterialInstanceDynamic;
class UStaticMesh;
class UObject;

namespace RubbleRender
{
	/**
	 * The surface material for a Rubble material. With the generated content
	 * (scripts/unreal/build-content.sh), instance tint comes from per-instance custom data 0..2 and
	 * the soft-body damage mask from vertex colour; without it, engine materials stand in (one
	 * flat colour each, tints ignored) so the app still runs.
	 */
	RUBBLERUNTIME_API UMaterialInterface* Surface(ERubbleMaterial Material);
	/** Car paint (clear coat, damage mask from vertex colour R) as a dynamic instance of Paint. */
	RUBBLERUNTIME_API UMaterialInstanceDynamic* CarPaint(UObject* Outer, const FLinearColor& Paint);
	RUBBLERUNTIME_API UMaterialInterface* Ground(ERubbleStage Stage);
	RUBBLERUNTIME_API UMaterialInterface* Dust();
	RUBBLERUNTIME_API UMaterialInterface* Spark();
	/** True when the generated materials were found (else everything is the engine fallback). */
	RUBBLERUNTIME_API bool HasGeneratedContent();

	/** A static mesh with one section per material used (built at runtime, no collision). */
	RUBBLERUNTIME_API UStaticMesh* BuildMesh(UObject* Outer, const FRubbleMeshData& Mesh, FName Name = NAME_None);
	/** A capsule along Z (radius, half height of the cylinder part), for capsule shapes. */
	RUBBLERUNTIME_API void CapsuleMesh(float Radius, float HalfHeight, FRubbleMeshData& Out);

	RUBBLERUNTIME_API UStaticMesh* EngineCube();     // 100 cm
	RUBBLERUNTIME_API UStaticMesh* EngineSphere();   // 100 cm across
	RUBBLERUNTIME_API UStaticMesh* EngineCylinder(); // 100 cm across, 100 cm tall, along Z
}
