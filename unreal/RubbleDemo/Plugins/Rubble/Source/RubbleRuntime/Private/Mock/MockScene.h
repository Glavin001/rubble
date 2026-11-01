// One mock scene per demo family. See MockShared.h: stand-in motion, not physics.

#pragma once

#include "CoreMinimal.h"
#include "Mock/MockShared.h"
#include "RubbleTypes.h"

namespace RubbleMock
{
	class FScene
	{
	public:
		virtual ~FScene() = default;
		/** Build the scene. False and Error for an id the family does not have. */
		virtual bool Build(const FRubbleSceneRequest& Request, FRubbleSceneAssets& Assets, FString& Error) = 0;
		virtual void Step(float Dt, FRubbleFrame& Out) = 0;
		virtual void SetVehicleInput(const FRubbleVehicleInput& Input) {}
		virtual void Launch(const FVector& Origin, const FVector& Direction) {}
		virtual void SideRam() {}
	};

	TArray<FRubbleDemoDesc> RigidCatalog();
	TArray<FRubbleDemoDesc> VehicleCatalog();
	TArray<FRubbleDemoDesc> SoftCatalog();
	TArray<FRubbleDemoDesc> FractureCatalog();

	TUniquePtr<FScene> MakeRigidScene();
	TUniquePtr<FScene> MakeVehicleScene();
	TUniquePtr<FScene> MakeSoftScene();
	TUniquePtr<FScene> MakeFractureScene();

	/** Adds a body drawn by one shape to both the world and the assets; returns its index. */
	struct FBodyBuilder
	{
		FWorld& World;
		FRubbleSceneAssets& Assets;

		int32 Box(const FVector& Center, const FVector& Half, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, bool bStatic, bool bAwake = false);
		int32 Sphere(const FVector& Center, float Radius, ERubbleMaterial Material, const FLinearColor& Tint, bool bAwake = false);
		int32 Capsule(const FVector& Center, float Radius, float HalfHeight, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, bool bAwake = false);
		int32 Hull(const FVector& Center, int32 Mesh, float Radius, const FQuat& Rot, ERubbleMaterial Material, const FLinearColor& Tint, bool bAwake = false);
		/** A parked projectile (hidden under the ground until fired), as demo.html's F. */
		int32 ParkedBall(float Radius);
		/** Fire a parked ball. */
		static void Fire(FWorld& World, int32 Ball, const FVector& Origin, const FVector& Direction, float Speed);

	private:
		int32 Add(FBody& Body, const FRubbleShapeVisual& Shape, bool bStatic);
	};

	/** The brick and concrete tints the scenes use. */
	FLinearColor BrickTint(FRandomStream& Random);
	FLinearColor StoneTint(FRandomStream& Random);
}
