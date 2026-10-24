#include "RubbleTownPack.h"

#include "CompGeom/ConvexHull3.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace RubbleTownPack
{
	namespace
	{
		FVector3f Vec(const TSharedPtr<FJsonObject>& O)
		{
			return O ? FVector3f(float(O->GetNumberField(TEXT("x"))), float(O->GetNumberField(TEXT("y"))), float(O->GetNumberField(TEXT("z")))) : FVector3f::ZeroVector;
		}

		TArray<FVector3f> Points(const TArray<TSharedPtr<FJsonValue>>& Flat)
		{
			TArray<FVector3f> Out;
			Out.Reserve(Flat.Num() / 3);
			for (int32 i = 0; i + 2 < Flat.Num(); i += 3)
			{
				Out.Add(FVector3f(float(Flat[i]->AsNumber()), float(Flat[i + 1]->AsNumber()), float(Flat[i + 2]->AsNumber())));
			}
			return Out;
		}

		/** The renderer's material family for a pack material (its colour tints it). */
		ERubbleMaterial KindOf(const FString& Name)
		{
			auto Has = [&Name](const TCHAR* S) { return Name.Contains(S); };
			if (Has(TEXT("glass")))
			{
				return ERubbleMaterial::Glass;
			}
			if (Has(TEXT("brick")))
			{
				return ERubbleMaterial::Brick;
			}
			if (Has(TEXT("steel")) || Has(TEXT("metal")))
			{
				return ERubbleMaterial::Steel;
			}
			if (Has(TEXT("wood")) || Has(TEXT("oak")) || Has(TEXT("timber")) || Has(TEXT("joinery")) || Has(TEXT("siding")))
			{
				return ERubbleMaterial::Timber;
			}
			if (Has(TEXT("trim")) || Has(TEXT("paint")))
			{
				return ERubbleMaterial::Plastic;
			}
			return ERubbleMaterial::Concrete;
		}
	}

	bool Parse(const TArray<uint8>& Json, FPack& Out, FString& Error)
	{
		TSharedPtr<FJsonObject> Root;
		const FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Json.GetData()), Json.Num());
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<TCHAR>::Create(FString(Text.Length(), Text.Get())), Root) || !Root)
		{
			Error = TEXT("not JSON");
			return false;
		}
		Out.Title = Root->GetStringField(TEXT("title"));
		const TSharedPtr<FJsonObject>* Scenario = nullptr;
		if (!Root->TryGetObjectField(TEXT("scenario"), Scenario))
		{
			Error = TEXT("no scenario");
			return false;
		}
		const TSharedPtr<FJsonObject>& S = *Scenario;
		// Materials by name (defaults.solver.materials).
		TMap<FString, int32> MaterialOf;
		const TSharedPtr<FJsonObject>* Defaults = nullptr;
		const TSharedPtr<FJsonObject>* Solver = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* Mats = nullptr;
		if (Root->TryGetObjectField(TEXT("defaults"), Defaults) && (*Defaults)->TryGetObjectField(TEXT("solver"), Solver) && (*Solver)->TryGetArrayField(TEXT("materials"), Mats))
		{
			for (const TSharedPtr<FJsonValue>& V : *Mats)
			{
				const TSharedPtr<FJsonObject> M = V->AsObject();
				FMaterial Mat;
				Mat.Name = M->GetStringField(TEXT("name"));
				FString Hex;
				if (M->TryGetStringField(TEXT("color"), Hex))
				{
					Mat.Color = FLinearColor::FromSRGBColor(FColor::FromHex(Hex));
				}
				M->TryGetNumberField(TEXT("roughness"), Mat.Roughness);
				M->TryGetNumberField(TEXT("metalness"), Mat.Metalness);
				M->TryGetStringField(TEXT("textureKey"), Mat.TextureKey);
				Mat.Kind = KindOf(Mat.Name);
				MaterialOf.Add(Mat.Name, Out.Materials.Add(Mat));
			}
		}
		if (Out.Materials.Num() == 0)
		{
			Out.Materials.AddDefaulted();
		}
		const TArray<TSharedPtr<FJsonValue>>* Library = nullptr;
		if (S->TryGetArrayField(TEXT("shapeLibrary"), Library))
		{
			for (const TSharedPtr<FJsonValue>& V : *Library)
			{
				const TArray<TSharedPtr<FJsonValue>>* P = nullptr;
				Out.Hulls.Add(V->AsObject()->TryGetArrayField(TEXT("points"), P) ? Points(*P) : TArray<FVector3f>());
			}
		}
		const TArray<TSharedPtr<FJsonValue>>* Nodes = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* Colliders = nullptr;
		if (!S->TryGetArrayField(TEXT("nodes"), Nodes) || !S->TryGetArrayField(TEXT("nodeColliders"), Colliders) || Nodes->Num() != Colliders->Num())
		{
			Error = TEXT("no nodes or nodeColliders (a v2 pack has both, one per node)");
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* NodeMaterials = nullptr;
		S->TryGetArrayField(TEXT("nodeMaterials"), NodeMaterials);
		Out.Nodes.SetNum(Nodes->Num());
		for (int32 i = 0; i < Nodes->Num(); ++i)
		{
			FNode& N = Out.Nodes[i];
			const TSharedPtr<FJsonObject>* Centroid = nullptr;
			if ((*Nodes)[i]->AsObject()->TryGetObjectField(TEXT("centroid"), Centroid))
			{
				N.Centroid = Vec(*Centroid);
			}
			const TSharedPtr<FJsonObject> C = (*Colliders)[i]->AsObject();
			const FString Kind = C->GetStringField(TEXT("kind"));
			if (Kind == TEXT("cuboid"))
			{
				const TSharedPtr<FJsonObject>* Half = nullptr;
				N.bBox = true;
				N.HalfExtents = C->TryGetObjectField(TEXT("halfExtents"), Half) ? Vec(*Half) : FVector3f(0.5f);
			}
			else if (Kind == TEXT("shape"))
			{
				N.bBox = false;
				N.Hull = int32(C->GetNumberField(TEXT("shape")));
			}
			else if (Kind == TEXT("convex_hull"))
			{
				const TArray<TSharedPtr<FJsonValue>>* P = nullptr;
				N.bBox = false;
				N.Hull = Out.Hulls.Add(C->TryGetArrayField(TEXT("points"), P) ? Points(*P) : TArray<FVector3f>());
			}
			else
			{
				Error = FString::Printf(TEXT("node %d: collider kind '%s'"), i, *Kind);
				return false;
			}
			if (!N.bBox && !Out.Hulls.IsValidIndex(N.Hull))
			{
				Error = FString::Printf(TEXT("node %d: no shape %d"), i, N.Hull);
				return false;
			}
			if (NodeMaterials && NodeMaterials->IsValidIndex(i))
			{
				if (const int32* M = MaterialOf.Find((*NodeMaterials)[i]->AsString()))
				{
					N.Material = *M;
				}
			}
		}
		return true;
	}

	void ApplyCatalog(const TArray<uint8>& CatalogJson, FPack& Pack)
	{
		TSharedPtr<FJsonObject> Root;
		const FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(CatalogJson.GetData()), CatalogJson.Num());
		const TArray<TSharedPtr<FJsonValue>>* Materials = nullptr;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<TCHAR>::Create(FString(Text.Length(), Text.Get())), Root) || !Root || !Root->TryGetArrayField(TEXT("materials"), Materials))
		{
			return;
		}
		TMap<FString, TPair<int32, float>> ById;
		for (const TSharedPtr<FJsonValue>& V : *Materials)
		{
			const TSharedPtr<FJsonObject> M = V->AsObject();
			double Layer = -1.0, Metres = 1.0;
			if (M->TryGetNumberField(TEXT("layer"), Layer))
			{
				M->TryGetNumberField(TEXT("metresPerTile"), Metres);
				ById.Add(M->GetStringField(TEXT("id")), TPair<int32, float>(int32(Layer), float(Metres)));
			}
		}
		for (FMaterial& M : Pack.Materials)
		{
			if (const TPair<int32, float>* Hit = ById.Find(M.TextureKey))
			{
				M.Layer = Hit->Key;
				M.MetresPerTile = FMath::Max(Hit->Value, 0.01f);
			}
		}
	}

	bool HullMesh(const TArray<FVector3f>& RubblePoints, FRubbleMeshData& Out)
	{
		TArray<FVector3d> P;
		P.Reserve(RubblePoints.Num());
		FVector3d Center = FVector3d::ZeroVector;
		for (const FVector3f& R : RubblePoints)
		{
			P.Add(FVector3d(ToUnreal(R)));
			Center += P.Last();
		}
		if (P.Num() < 4)
		{
			return false;
		}
		Center /= P.Num();
		UE::Geometry::FConvexHull3d Hull;
		if (!Hull.Solve(TArrayView<const FVector3d>(P)) || Hull.GetDimension() < 3)
		{
			return false;
		}
		// Flat shading: three vertices per triangle, the face normal on each; UVs box-projected
		// (metres) on the face's dominant axis.
		for (const UE::Geometry::FIndex3i& T : Hull.GetTriangles())
		{
			FVector3d A = P[T.A], B = P[T.B], C = P[T.C];
			// Unreal winding: (c - a) x (b - a) points out of the hull.
			if (FVector3d::DotProduct(FVector3d::CrossProduct(C - A, B - A), (A + B + C) / 3.0 - Center) < 0.0)
			{
				Swap(B, C);
			}
			const FVector3f Normal = FVector3f(FVector3d::CrossProduct(C - A, B - A).GetSafeNormal());
			const FVector3f AbsN = Normal.GetAbs();
			const uint32 Base = Out.Positions.Num();
			for (const FVector3d& V : { A, B, C })
			{
				const FVector3f Vf(V);
				Out.Positions.Add(Vf);
				Out.Normals.Add(Normal);
				const FVector2f UV = AbsN.Z >= AbsN.X && AbsN.Z >= AbsN.Y ? FVector2f(Vf.X, Vf.Y) : AbsN.X >= AbsN.Y ? FVector2f(Vf.Y, Vf.Z) : FVector2f(Vf.X, Vf.Z);
				Out.UVs.Add(UV / 100.0f);
			}
			Out.Indices.Append({ Base, Base + 1, Base + 2 });
		}
		return Out.Indices.Num() > 0;
	}
}
