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
	/** M_RubbleTownChunk (build_content.py: textured town chunks, 12 custom data floats), or null
	 *  when it was not generated. */
	RUBBLERUNTIME_API UMaterialInterface* TownChunk();
	/** A town chunk's material: M_RubbleCurtainGlass (see-through, reflective) for a glass
	 *  tower's curtain wall, M_RubbleTownGlass (reflective panes over lit rooms) for other glass,
	 *  M_RubbleTownChunk otherwise; null when they were not generated. */
	RUBBLERUNTIME_API UMaterialInterface* TownSurface(ERubbleMaterial Material, bool bCurtainGlass = false);
	/** Rubble's own scanned look for a pack material (Quixel Megascans, MI_Scan_*): its generated
	 *  material instance's name, or None (no look for it, or the content was not generated). */
	RUBBLERUNTIME_API FName ScanSurface(const FString& PackMaterial);
	/** The material for a ScanSurface name; null for None. */
	RUBBLERUNTIME_API UMaterialInterface* ScanMaterial(FName Surface);
	/** Car paint (clear coat, damage mask from vertex colour R) as a dynamic instance of Paint. */
	RUBBLERUNTIME_API UMaterialInstanceDynamic* CarPaint(UObject* Outer, const FLinearColor& Paint);
	/** A soft car surface (build_content.py soft_car): paint as a dynamic instance of Paint;
	 *  the generic surfaces stand in when the content was not generated. */
	RUBBLERUNTIME_API UMaterialInterface* SoftCarSurface(UObject* Outer, ERubbleCarSurface Which, const FLinearColor& Paint);
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
	/** Nanite copies of the cube and sphere (build_content.py), or the engine's when not built. */
	RUBBLERUNTIME_API UStaticMesh* NaniteCube();
	RUBBLERUNTIME_API UStaticMesh* NaniteSphere();
	RUBBLERUNTIME_API UStaticMesh* EngineCylinder(); // 100 cm across, 100 cm tall, along Z
}
