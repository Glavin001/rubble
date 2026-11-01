#include "RubbleGlb.h"

#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace RubbleGlb
{
	namespace
	{
		constexpr uint32 Magic = 0x46546C67; // "glTF"
		constexpr uint32 JsonChunk = 0x4E4F534A;
		constexpr uint32 BinChunk = 0x004E4942;

		struct FContext
		{
			TSharedPtr<FJsonObject> Root;
			const uint8* Bin = nullptr;
			int64 BinSize = 0;
			FString Error;
		};

		const TArray<TSharedPtr<FJsonValue>>* Array(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
		{
			const TArray<TSharedPtr<FJsonValue>>* A = nullptr;
			return O->TryGetArrayField(Key, A) ? A : nullptr;
		}

		/** Read accessor Index as floats (Components each) or as indices. */
		bool ReadAccessor(FContext& C, int32 Index, int32 Components, TArray<float>* Floats, TArray<uint32>* Ints)
		{
			const TArray<TSharedPtr<FJsonValue>>* Accessors = Array(C.Root, TEXT("accessors"));
			const TArray<TSharedPtr<FJsonValue>>* Views = Array(C.Root, TEXT("bufferViews"));
			if (!Accessors || !Views || !Accessors->IsValidIndex(Index))
			{
				C.Error = FString::Printf(TEXT("accessor %d missing"), Index);
				return false;
			}
			const TSharedPtr<FJsonObject> A = (*Accessors)[Index]->AsObject();
			const int32 Count = A->GetIntegerField(TEXT("count"));
			const int32 Type = A->GetIntegerField(TEXT("componentType"));
			int32 ViewIndex = -1;
			if (!A->TryGetNumberField(TEXT("bufferView"), ViewIndex) || !Views->IsValidIndex(ViewIndex))
			{
				C.Error = TEXT("sparse or view-less accessors are not supported");
				return false;
			}
			const TSharedPtr<FJsonObject> V = (*Views)[ViewIndex]->AsObject();
			int64 Offset = 0, AccessorOffset = 0, Stride = 0;
			V->TryGetNumberField(TEXT("byteOffset"), Offset);
			A->TryGetNumberField(TEXT("byteOffset"), AccessorOffset);
			V->TryGetNumberField(TEXT("byteStride"), Stride);
			const int32 ComponentSize = Type == 5126 || Type == 5125 ? 4 : Type == 5123 || Type == 5122 ? 2 : 1;
			const int64 Elem = int64(ComponentSize) * (Floats ? Components : 1);
			Stride = Stride > 0 ? Stride : Elem;
			const int64 Base = Offset + AccessorOffset;
			if (Count > 0 && Base + Stride * (Count - 1) + Elem > C.BinSize)
			{
				C.Error = FString::Printf(TEXT("accessor %d runs past the binary chunk"), Index);
				return false;
			}
			for (int32 i = 0; i < Count; ++i)
			{
				const uint8* P = C.Bin + Base + Stride * i;
				if (Floats)
				{
					if (Type != 5126)
					{
						C.Error = TEXT("non-float vertex attributes are not supported");
						return false;
					}
					for (int32 k = 0; k < Components; ++k)
					{
						float F;
						FMemory::Memcpy(&F, P + 4 * k, 4);
						Floats->Add(F);
					}
				}
				else
				{
					uint32 U = 0;
					if (Type == 5125) FMemory::Memcpy(&U, P, 4);
					else if (Type == 5123) { uint16 S; FMemory::Memcpy(&S, P, 2); U = S; }
					else if (Type == 5121) U = *P;
					else { C.Error = TEXT("unsupported index type"); return false; }
					Ints->Add(U);
				}
			}
			return true;
		}

		FMatrix44f LocalMatrix(const TSharedPtr<FJsonObject>& Node)
		{
			const TArray<TSharedPtr<FJsonValue>>* M = Array(Node, TEXT("matrix"));
			if (M && M->Num() == 16)
			{
				// glTF is column-major: element (row r, column c) is M[c * 4 + r]. FMatrix44f is
				// row-vector (p' = p M), so its [i][j] is glTF's (j, i): exactly M[i * 4 + j].
				FMatrix44f Out;
				for (int32 i = 0; i < 4; ++i)
				{
					for (int32 j = 0; j < 4; ++j)
					{
						Out.M[i][j] = float((*M)[i * 4 + j]->AsNumber());
					}
				}
				return Out;
			}
			auto Vec = [&Node](const TCHAR* Key, int32 N, const TArray<float>& Default) {
				TArray<float> V = Default;
				if (const TArray<TSharedPtr<FJsonValue>>* A = Array(Node, Key))
				{
					for (int32 i = 0; i < N && i < A->Num(); ++i)
					{
						V[i] = float((*A)[i]->AsNumber());
					}
				}
				return V;
			};
			const TArray<float> T = Vec(TEXT("translation"), 3, { 0, 0, 0 });
			const TArray<float> R = Vec(TEXT("rotation"), 4, { 0, 0, 0, 1 });
			const TArray<float> S = Vec(TEXT("scale"), 3, { 1, 1, 1 });
			// p' = p S R T in row-vector form (glTF: T R S applied to a column vector).
			const FQuat4f Q(R[0], R[1], R[2], R[3]);
			FMatrix44f Rot = FRotationMatrix44f::Make(Q);
			FMatrix44f Out = FScaleMatrix44f(FVector3f(S[0], S[1], S[2])) * Rot;
			Out.M[3][0] = T[0];
			Out.M[3][1] = T[1];
			Out.M[3][2] = T[2];
			return Out;
		}
	}

	ERubbleMaterial MaterialOf(const FString& Name)
	{
		const FString N = Name.ToLower();
		if (N.Contains(TEXT("interior")) || N.Contains(TEXT("inside")) || N.Contains(TEXT("core")) || N.Contains(TEXT("fracture")) || N.Contains(TEXT("cut"))) return ERubbleMaterial::Interior;
		if (N.Contains(TEXT("brick"))) return ERubbleMaterial::Brick;
		if (N.Contains(TEXT("wood")) || N.Contains(TEXT("timber")) || N.Contains(TEXT("roof"))) return ERubbleMaterial::Timber;
		if (N.Contains(TEXT("glass")) || N.Contains(TEXT("window"))) return ERubbleMaterial::Glass;
		if (N.Contains(TEXT("steel")) || N.Contains(TEXT("metal")) || N.Contains(TEXT("rebar"))) return ERubbleMaterial::Steel;
		return ERubbleMaterial::Concrete;
	}

	bool Read(const TArray<uint8>& Bytes, FFile& Out, FString& Error)
	{
		auto U32 = [&Bytes](int64 At) { uint32 V = 0; FMemory::Memcpy(&V, Bytes.GetData() + At, 4); return V; };
		if (Bytes.Num() < 20 || U32(0) != Magic || U32(4) != 2)
		{
			Error = TEXT("not a glTF 2 binary");
			return false;
		}
		FContext C;
		FString Json;
		for (int64 At = 12; At + 8 <= Bytes.Num();)
		{
			const uint32 Length = U32(At), Type = U32(At + 4);
			if (At + 8 + Length > uint64(Bytes.Num()))
			{
				Error = TEXT("truncated chunk");
				return false;
			}
			if (Type == JsonChunk)
			{
				FUTF8ToTCHAR Text(reinterpret_cast<const ANSICHAR*>(Bytes.GetData() + At + 8), int32(Length));
				Json = FString(Text.Length(), Text.Get());
			}
			else if (Type == BinChunk)
			{
				C.Bin = Bytes.GetData() + At + 8;
				C.BinSize = Length;
			}
			At += 8 + ((Length + 3) & ~3u);
		}
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), C.Root) || !C.Root.IsValid())
		{
			Error = TEXT("bad JSON chunk");
			return false;
		}
		// A Shardwright fracture asset (scenes[].extras.generator "prefracture ...").
		bool bShardwright = false;
		if (const TArray<TSharedPtr<FJsonValue>>* Scenes = Array(C.Root, TEXT("scenes")))
		{
			for (const TSharedPtr<FJsonValue>& Scene : *Scenes)
			{
				const TSharedPtr<FJsonObject>* Extras = nullptr;
				FString Generator;
				if (Scene->AsObject()->TryGetObjectField(TEXT("extras"), Extras) && (*Extras)->TryGetStringField(TEXT("generator"), Generator) && Generator.StartsWith(TEXT("prefracture")))
				{
					bShardwright = true;
				}
			}
		}
		Out.bShardwright = bShardwright;
		const TArray<TSharedPtr<FJsonValue>>* Nodes = Array(C.Root, TEXT("nodes"));
		const TArray<TSharedPtr<FJsonValue>>* Meshes = Array(C.Root, TEXT("meshes"));
		const TArray<TSharedPtr<FJsonValue>>* Materials = Array(C.Root, TEXT("materials"));
		if (!Nodes)
		{
			Error = TEXT("no nodes");
			return false;
		}
		// World (scene-frame) matrices: parents first, from every node's children list.
		TArray<int32> Parent;
		Parent.Init(-1, Nodes->Num());
		for (int32 n = 0; n < Nodes->Num(); ++n)
		{
			if (const TArray<TSharedPtr<FJsonValue>>* Children = Array((*Nodes)[n]->AsObject(), TEXT("children")))
			{
				for (const TSharedPtr<FJsonValue>& Child : *Children)
				{
					const int32 c = int32(Child->AsNumber());
					if (Parent.IsValidIndex(c))
					{
						Parent[c] = n;
					}
				}
			}
		}
		TArray<FMatrix44f> World;
		TArray<bool> Done;
		World.SetNum(Nodes->Num());
		Done.Init(false, Nodes->Num());
		TFunction<const FMatrix44f&(int32)> WorldOf = [&](int32 n) -> const FMatrix44f& {
			if (!Done[n])
			{
				const FMatrix44f Local = LocalMatrix((*Nodes)[n]->AsObject());
				World[n] = Parent[n] >= 0 ? Local * WorldOf(Parent[n]) : Local;
				Done[n] = true;
			}
			return World[n];
		};

		Out.NodeMeshes.SetNum(Nodes->Num());
		for (int32 n = 0; n < Nodes->Num(); ++n)
		{
			int32 MeshIndex = -1;
			if (!(*Nodes)[n]->AsObject()->TryGetNumberField(TEXT("mesh"), MeshIndex) || !Meshes || !Meshes->IsValidIndex(MeshIndex))
			{
				continue;
			}
			const FMatrix44f& M = WorldOf(n);
			FRubbleMeshData& Mesh = Out.NodeMeshes[n];
			const TArray<TSharedPtr<FJsonValue>>* Prims = Array((*Meshes)[MeshIndex]->AsObject(), TEXT("primitives"));
			for (int32 p = 0; Prims && p < Prims->Num(); ++p)
			{
				const TSharedPtr<FJsonObject> Prim = (*Prims)[p]->AsObject();
				int32 Mode = 4;
				Prim->TryGetNumberField(TEXT("mode"), Mode);
				const TSharedPtr<FJsonObject>* Attributes = nullptr;
				if (Mode != 4 || !Prim->TryGetObjectField(TEXT("attributes"), Attributes))
				{
					continue; // triangles only
				}
				int32 PosIndex = -1, NormalIndex = -1, UvIndex = -1, IndexIndex = -1, MaterialIndex = -1;
				(*Attributes)->TryGetNumberField(TEXT("POSITION"), PosIndex);
				(*Attributes)->TryGetNumberField(TEXT("NORMAL"), NormalIndex);
				(*Attributes)->TryGetNumberField(TEXT("TEXCOORD_0"), UvIndex);
				Prim->TryGetNumberField(TEXT("indices"), IndexIndex);
				Prim->TryGetNumberField(TEXT("material"), MaterialIndex);
				TArray<float> P, N, UV;
				TArray<uint32> I;
				if (PosIndex < 0 || !ReadAccessor(C, PosIndex, 3, &P, nullptr)
					|| (NormalIndex >= 0 && !ReadAccessor(C, NormalIndex, 3, &N, nullptr))
					|| (UvIndex >= 0 && !ReadAccessor(C, UvIndex, 2, &UV, nullptr))
					|| (IndexIndex >= 0 && !ReadAccessor(C, IndexIndex, 1, nullptr, &I)))
				{
					Error = C.Error;
					return false;
				}
				const int32 NumVerts = P.Num() / 3;
				if (I.Num() == 0)
				{
					for (int32 v = 0; v < NumVerts; ++v)
					{
						I.Add(v);
					}
				}
				FString MaterialName;
				if (Materials && Materials->IsValidIndex(MaterialIndex))
				{
					(*Materials)[MaterialIndex]->AsObject()->TryGetStringField(TEXT("name"), MaterialName);
				}
				const ERubbleMaterial Surface = MaterialOf(MaterialName);
				const uint32 Base = Mesh.Positions.Num();
				for (int32 v = 0; v < NumVerts; ++v)
				{
					const FVector3f Scene = M.TransformPosition(FVector3f(P[3 * v], P[3 * v + 1], P[3 * v + 2]));
					Mesh.Positions.Add(ToUnreal(Scene));
					const FVector3f Nrm = N.Num() >= 3 * (v + 1) ? FVector3f(M.TransformVector(FVector3f(N[3 * v], N[3 * v + 1], N[3 * v + 2]))).GetSafeNormal() : FVector3f::UpVector;
					Mesh.Normals.Add(ToUnreal(Nrm).GetSafeNormal());
					Mesh.UVs.Add(UV.Num() >= 2 * (v + 1) ? FVector2f(UV[2 * v], UV[2 * v + 1]) : FVector2f::ZeroVector);
				}
				for (int32 t = 0; t + 2 < I.Num(); t += 3)
				{
					if (I[t] >= uint32(NumVerts) || I[t + 1] >= uint32(NumVerts) || I[t + 2] >= uint32(NumVerts))
					{
						Error = TEXT("index out of range");
						return false;
					}
					// glTF's front faces are counter-clockwise; the axis map is a reflection, which
					// makes them clockwise in Unreal's axes: Unreal's front face. The order is kept.
					Mesh.Indices.Append({ Base + I[t], Base + I[t + 1], Base + I[t + 2] });
					Mesh.TriangleMaterials.Add(Surface);
				}
			}
			Mesh.Material = Mesh.TriangleMaterials.Num() ? Mesh.TriangleMaterials[0] : ERubbleMaterial::Concrete;
		}
		return true;
	}
}
