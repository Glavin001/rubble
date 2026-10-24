#include "RubbleRender.h"

#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "MeshDescription.h"
#include "StaticMeshAttributes.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"

namespace RubbleRender
{
	namespace
	{
		const TCHAR* GeneratedRoot = TEXT("/Game/Rubble/Materials/");

		const TCHAR* SurfaceName(ERubbleMaterial M)
		{
			switch (M)
			{
			case ERubbleMaterial::Concrete: return TEXT("MI_Concrete");
			case ERubbleMaterial::Brick: return TEXT("MI_Brick");
			case ERubbleMaterial::Timber: return TEXT("MI_Timber");
			case ERubbleMaterial::Steel: return TEXT("MI_Steel");
			case ERubbleMaterial::Glass: return TEXT("MI_Glass");
			case ERubbleMaterial::Rubber: return TEXT("MI_Rubber");
			case ERubbleMaterial::CarPaint: return TEXT("MI_Plastic");
			case ERubbleMaterial::Plastic: return TEXT("MI_Plastic");
			case ERubbleMaterial::Interior: return TEXT("MI_Interior");
			default: return TEXT("MI_Concrete");
			}
		}

		FLinearColor FallbackColor(ERubbleMaterial M)
		{
			switch (M)
			{
			case ERubbleMaterial::Brick: return FLinearColor(0.45f, 0.17f, 0.11f);
			case ERubbleMaterial::Timber: return FLinearColor(0.55f, 0.38f, 0.2f);
			case ERubbleMaterial::Steel: return FLinearColor(0.55f, 0.56f, 0.58f);
			case ERubbleMaterial::Glass: return FLinearColor(0.2f, 0.3f, 0.35f);
			case ERubbleMaterial::Rubber: return FLinearColor(0.03f, 0.03f, 0.03f);
			case ERubbleMaterial::Plastic: return FLinearColor(0.6f, 0.6f, 0.6f);
			case ERubbleMaterial::Interior: return FLinearColor(0.35f, 0.33f, 0.3f);
			default: return FLinearColor(0.5f, 0.49f, 0.46f);
			}
		}

		template <typename T>
		T* LoadGenerated(const TCHAR* Name)
		{
			const FString Path = FString::Printf(TEXT("%s%s.%s"), GeneratedRoot, Name, Name);
			return LoadObject<T>(nullptr, *Path, nullptr, LOAD_NoWarn | LOAD_Quiet);
		}

		UMaterialInterface* BasicShapeMaterial()
		{
			static TStrongObjectPtr<UMaterialInterface> M(LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")));
			return M.Get();
		}

		UMaterialInterface* Fallback(const FLinearColor& Color, FName Key)
		{
			static TMap<FName, TStrongObjectPtr<UMaterialInstanceDynamic>> Cache;
			if (TStrongObjectPtr<UMaterialInstanceDynamic>* Found = Cache.Find(Key))
			{
				return Found->Get();
			}
			UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BasicShapeMaterial(), GetTransientPackage(), Key);
			MID->SetVectorParameterValue(TEXT("Color"), Color);
			Cache.Add(Key, TStrongObjectPtr<UMaterialInstanceDynamic>(MID));
			return MID;
		}
	}

	bool HasGeneratedContent()
	{
		static const bool bHas = LoadGenerated<UMaterialInterface>(TEXT("M_RubbleSurface")) != nullptr;
		return bHas;
	}

	UMaterialInterface* Surface(ERubbleMaterial Material)
	{
		if (HasGeneratedContent())
		{
			if (UMaterialInterface* M = LoadGenerated<UMaterialInterface>(SurfaceName(Material)))
			{
				return M;
			}
		}
		return Fallback(FallbackColor(Material), FName(*FString::Printf(TEXT("RubbleFallback_%d"), int32(Material))));
	}

	UMaterialInstanceDynamic* CarPaint(UObject* Outer, const FLinearColor& Paint)
	{
		UMaterialInterface* Parent = HasGeneratedContent() ? LoadGenerated<UMaterialInterface>(TEXT("M_RubbleCarPaint")) : nullptr;
		UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(Parent ? Parent : BasicShapeMaterial(), Outer);
		MID->SetVectorParameterValue(Parent ? TEXT("Paint") : TEXT("Color"), Paint);
		return MID;
	}

	UMaterialInterface* Ground(ERubbleStage Stage)
	{
		if (HasGeneratedContent())
		{
			const TCHAR* Name = Stage == ERubbleStage::CrashHall ? TEXT("MI_GroundHall") : Stage == ERubbleStage::Lot ? TEXT("MI_GroundLot") : TEXT("MI_GroundYard");
			if (UMaterialInterface* M = LoadGenerated<UMaterialInterface>(Name))
			{
				return M;
			}
		}
		return Fallback(FLinearColor(0.08f, 0.08f, 0.085f), TEXT("RubbleFallback_Ground"));
	}

	UMaterialInterface* Dust()
	{
		if (UMaterialInterface* M = HasGeneratedContent() ? LoadGenerated<UMaterialInterface>(TEXT("M_RubbleDust")) : nullptr)
		{
			return M;
		}
		return Fallback(FLinearColor(0.55f, 0.52f, 0.47f), TEXT("RubbleFallback_Dust"));
	}

	UMaterialInterface* Spark()
	{
		if (UMaterialInterface* M = HasGeneratedContent() ? LoadGenerated<UMaterialInterface>(TEXT("M_RubbleSpark")) : nullptr)
		{
			return M;
		}
		return Fallback(FLinearColor(1.0f, 0.55f, 0.1f), TEXT("RubbleFallback_Spark"));
	}

	UStaticMesh* EngineCube()
	{
		static TStrongObjectPtr<UStaticMesh> M(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube")));
		return M.Get();
	}

	UStaticMesh* EngineSphere()
	{
		static TStrongObjectPtr<UStaticMesh> M(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Sphere.Sphere")));
		return M.Get();
	}

	UStaticMesh* NaniteCube()
	{
		static TStrongObjectPtr<UStaticMesh> M(LoadObject<UStaticMesh>(nullptr, TEXT("/Game/Rubble/Meshes/SM_NaniteBox.SM_NaniteBox"), nullptr, LOAD_NoWarn | LOAD_Quiet));
		return M ? M.Get() : EngineCube();
	}

	UStaticMesh* NaniteSphere()
	{
		static TStrongObjectPtr<UStaticMesh> M(LoadObject<UStaticMesh>(nullptr, TEXT("/Game/Rubble/Meshes/SM_NaniteSphere.SM_NaniteSphere"), nullptr, LOAD_NoWarn | LOAD_Quiet));
		return M ? M.Get() : EngineSphere();
	}

	UStaticMesh* EngineCylinder()
	{
		static TStrongObjectPtr<UStaticMesh> M(LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder")));
		return M.Get();
	}

	UStaticMesh* BuildMesh(UObject* Outer, const FRubbleMeshData& Mesh, FName Name)
	{
		FMeshDescription Desc;
		FStaticMeshAttributes Attributes(Desc);
		Attributes.Register();
		TVertexAttributesRef<FVector3f> Positions = Attributes.GetVertexPositions();
		TVertexInstanceAttributesRef<FVector3f> Normals = Attributes.GetVertexInstanceNormals();
		TVertexInstanceAttributesRef<FVector2f> UVs = Attributes.GetVertexInstanceUVs();
		TPolygonGroupAttributesRef<FName> SlotNames = Attributes.GetPolygonGroupMaterialSlotNames();

		Desc.ReserveNewVertices(Mesh.Positions.Num());
		Desc.ReserveNewVertexInstances(Mesh.Indices.Num());
		Desc.ReserveNewTriangles(Mesh.NumTriangles());
		TArray<FVertexID> Vertices;
		Vertices.SetNum(Mesh.Positions.Num());
		for (int32 v = 0; v < Mesh.Positions.Num(); ++v)
		{
			Vertices[v] = Desc.CreateVertex();
			Positions[Vertices[v]] = Mesh.Positions[v];
		}
		// One polygon group (section) per material used, in material order.
		TMap<ERubbleMaterial, FPolygonGroupID> Groups;
		TArray<ERubbleMaterial> Order;
		auto GroupOf = [&](ERubbleMaterial M) {
			if (const FPolygonGroupID* G = Groups.Find(M))
			{
				return *G;
			}
			const FPolygonGroupID G = Desc.CreatePolygonGroup();
			SlotNames[G] = FName(*FString::Printf(TEXT("M%d"), int32(M)));
			Order.Add(M);
			return Groups.Add(M, G);
		};
		for (int32 t = 0; t < Mesh.NumTriangles(); ++t)
		{
			const ERubbleMaterial M = Mesh.TriangleMaterials.IsValidIndex(t) ? Mesh.TriangleMaterials[t] : Mesh.Material;
			TArray<FVertexInstanceID, TInlineAllocator<3>> Corners;
			for (int32 k = 0; k < 3; ++k)
			{
				const uint32 Index = Mesh.Indices[3 * t + k];
				const FVertexInstanceID I = Desc.CreateVertexInstance(Vertices[Index]);
				Normals[I] = Mesh.Normals.IsValidIndex(Index) ? Mesh.Normals[Index] : FVector3f::UpVector;
				UVs[I] = Mesh.UVs.IsValidIndex(Index) ? Mesh.UVs[Index] : FVector2f::ZeroVector;
				Corners.Add(I);
			}
			Desc.CreateTriangle(GroupOf(M), Corners);
		}

		UStaticMesh* StaticMesh = NewObject<UStaticMesh>(Outer, Name);
		for (const ERubbleMaterial M : Order)
		{
			StaticMesh->GetStaticMaterials().Add(FStaticMaterial(Surface(M), FName(*FString::Printf(TEXT("M%d"), int32(M)))));
		}
		UStaticMesh::FBuildMeshDescriptionsParams Params;
		Params.bBuildSimpleCollision = false;
		Params.bFastBuild = true;
		Params.bAllowCpuAccess = false;
		StaticMesh->BuildFromMeshDescriptions({ &Desc }, Params);
		return StaticMesh;
	}

	void CapsuleMesh(float Radius, float HalfHeight, FRubbleMeshData& Out)
	{
		// Rings from the bottom pole to the top pole; the hemispheres are split by the cylinder.
		const int32 Seg = 24, HalfRings = 8;
		TArray<TPair<float, float>> Rings; // (z, r)
		for (int32 i = 0; i <= HalfRings; ++i)
		{
			const float a = -UE_HALF_PI + UE_HALF_PI * float(i) / HalfRings;
			Rings.Add({ -HalfHeight + Radius * FMath::Sin(a), Radius * FMath::Cos(a) });
		}
		for (int32 i = 0; i <= HalfRings; ++i)
		{
			const float a = UE_HALF_PI * float(i) / HalfRings;
			Rings.Add({ HalfHeight + Radius * FMath::Sin(a), Radius * FMath::Cos(a) });
		}
		for (int32 r = 0; r < Rings.Num(); ++r)
		{
			const float z = Rings[r].Key, rr = Rings[r].Value;
			const float Center = z < 0.0f ? -HalfHeight : HalfHeight;
			for (int32 s = 0; s <= Seg; ++s)
			{
				const float a = UE_TWO_PI * float(s) / Seg;
				const FVector3f P(rr * FMath::Cos(a), rr * FMath::Sin(a), z);
				Out.Positions.Add(P);
				Out.Normals.Add((P - FVector3f(0, 0, Center)).GetSafeNormal());
				Out.UVs.Add(FVector2f(float(s) / Seg, float(r) / (Rings.Num() - 1)));
			}
		}
		for (int32 r = 0; r + 1 < Rings.Num(); ++r)
		{
			for (int32 s = 0; s < Seg; ++s)
			{
				const uint32 a = r * (Seg + 1) + s, b = a + 1, c = a + (Seg + 1), d = c + 1;
				// Unreal winding: (c - a) x (b - a) points out.
				Out.Indices.Append({ a, c, b, b, c, d });
			}
		}
	}
}
