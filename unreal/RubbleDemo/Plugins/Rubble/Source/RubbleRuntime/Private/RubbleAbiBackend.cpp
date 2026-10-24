#include "RubbleAbiBackend.h"

#include "HAL/PlatformFileManager.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RubbleGlb.h"
#include "RubbleTownPack.h"
#include "HAL/PlatformTime.h"
#include "RubbleSubsystem.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

THIRD_PARTY_INCLUDES_START
#include "rubble.h"
THIRD_PARTY_INCLUDES_END

namespace
{
	RbVec3 Vec(const FVector& V) { return RbVec3{ float(V.X), float(V.Y), float(V.Z) }; }
	FVector Vec(const RbVec3& V) { return FVector(V.x, V.y, V.z); }
	RbTransform Xf(const FTransform& T)
	{
		const FQuat Q = T.GetRotation();
		return RbTransform{ Vec(T.GetLocation()), RbQuat{ float(Q.X), float(Q.Y), float(Q.Z), float(Q.W) } };
	}
	FTransform Xf(const RbTransform& T) { return FTransform(FQuat(T.q.x, T.q.y, T.q.z, T.q.w), Vec(T.p)); }

	/** kg/cm³ (the ABI's density unit in Unreal coordinates). */
	float Density(ERubbleMaterial M)
	{
		switch (M)
		{
		case ERubbleMaterial::Brick: return 0.0019f;
		case ERubbleMaterial::Timber: return 0.0006f;
		case ERubbleMaterial::Steel: return 0.0078f;
		case ERubbleMaterial::Rubber: return 0.0012f;
		case ERubbleMaterial::Plastic: return 0.001f;
		default: return 0.0024f;
		}
	}

	const char* PresetOf(ERubbleVehicleStyle S, bool bDestructible)
	{
		switch (S)
		{
		case ERubbleVehicleStyle::Sports: return "sports";
		case ERubbleVehicleStyle::MonsterTruck: return "monster_truck";
		case ERubbleVehicleStyle::SixWheelTruck: return "six_wheel_truck";
		case ERubbleVehicleStyle::Kart: return "kart";
		case ERubbleVehicleStyle::Tank: return "tank";
		case ERubbleVehicleStyle::Motorcycle: return "motorcycle";
		case ERubbleVehicleStyle::DuneBuggy: return "dune_buggy";
		default: return bDestructible ? "destructible_sedan" : "sedan";
		}
	}

	/** A JSON string from an rbGet*Json call (size query, then fill). */
	template <typename F>
	bool AbiJson(F&& Call, FString& Out)
	{
		size_t Length = 0;
		if (Call(nullptr, 0, &Length) != RB_OK)
		{
			return false;
		}
		TArray<ANSICHAR> Buffer;
		Buffer.SetNumZeroed(int32(Length) + 1);
		if (Call(Buffer.GetData(), size_t(Buffer.Num()), &Length) != RB_OK)
		{
			return false;
		}
		Out = UTF8_TO_TCHAR(Buffer.GetData());
		return true;
	}

	/** Rubble's frame (Y up, metres) to Unreal's: the ABI's axis map (engine-ffi coords.rs). */
	FVector PointOf(const TArray<TSharedPtr<FJsonValue>>& A, double Scale = 100.0)
	{
		return A.Num() >= 3 ? FVector(-A[2]->AsNumber(), A[0]->AsNumber(), A[1]->AsNumber()) * Scale : FVector::ZeroVector;
	}
	FQuat RotationOf(const TArray<TSharedPtr<FJsonValue>>& Q)
	{
		// A rotation maps as (det(M) M q.xyz, w) with det(M) = -1.
		return Q.Num() >= 4 ? FQuat(Q[2]->AsNumber(), -Q[0]->AsNumber(), -Q[1]->AsNumber(), Q[3]->AsNumber()).GetNormalized() : FQuat::Identity;
	}

	/** The preset's bodywork boxes and wheels, from its descriptor (rbGetVehiclePresetJson). */
	bool PresetVisual(const char* Name, FRubbleVehicleVisual& V)
	{
		FString Json;
		TSharedPtr<FJsonObject> Root;
		if (!AbiJson([Name](char* B, size_t C, size_t* N) { return rbGetVehiclePresetJson(Name, B, C, N); }, Json)
			|| !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root.IsValid())
		{
			return false;
		}
		V.Parts.Reset();
		V.Wheels.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Parts = nullptr;
		if (Root->TryGetArrayField(TEXT("parts"), Parts))
		{
			for (const TSharedPtr<FJsonValue>& Part : *Parts)
			{
				const FString PartName = Part->AsObject()->GetStringField(TEXT("name"));
				const TArray<TSharedPtr<FJsonValue>>* Colliders = nullptr;
				if (!Part->AsObject()->TryGetArrayField(TEXT("colliders"), Colliders))
				{
					continue;
				}
				for (const TSharedPtr<FJsonValue>& C : *Colliders)
				{
					const TSharedPtr<FJsonObject> Col = C->AsObject();
					const TSharedPtr<FJsonObject>* Shape = nullptr;
					if (!Col->TryGetObjectField(TEXT("shape"), Shape))
					{
						continue;
					}
					FRubbleShapeVisual S;
					S.Local = FTransform(RotationOf(Col->GetArrayField(TEXT("rotation"))), PointOf(Col->GetArrayField(TEXT("position"))));
					S.Material = PartName.Contains(TEXT("wheel")) || PartName.Contains(TEXT("tyre")) ? ERubbleMaterial::Rubber : ERubbleMaterial::CarPaint;
					const FString Type = (*Shape)->GetStringField(TEXT("type"));
					if (Type == TEXT("box"))
					{
						const TArray<TSharedPtr<FJsonValue>>& H = (*Shape)->GetArrayField(TEXT("half"));
						S.Kind = ERubbleShape::Box;
						S.HalfExtents = FVector3f(H[2]->AsNumber(), H[0]->AsNumber(), H[1]->AsNumber()) * 100.0f;
					}
					else if (Type == TEXT("sphere"))
					{
						S.Kind = ERubbleShape::Sphere;
						S.Radius = float((*Shape)->GetNumberField(TEXT("radius")) * 100.0);
					}
					else
					{
						continue; // capsules and hulls: drawn by the placeholder's proportions
					}
					V.Parts.Add(S);
				}
			}
		}
		const TArray<TSharedPtr<FJsonValue>>* Wheels = nullptr;
		if (Root->TryGetArrayField(TEXT("wheels"), Wheels))
		{
			for (const TSharedPtr<FJsonValue>& W : *Wheels)
			{
				FRubbleWheelDef D;
				D.LocalHub = FVector3f(PointOf(W->AsObject()->GetArrayField(TEXT("position"))));
				D.Radius = float(W->AsObject()->GetNumberField(TEXT("radius")) * 100.0);
				D.Width = float(W->AsObject()->GetNumberField(TEXT("width")) * 100.0);
				V.Wheels.Add(D);
			}
		}
		V.bForwardIsMinusX = true; // presets drive along Rubble +Z, which is Unreal -X
		return V.Parts.Num() > 0;
	}

	FString AbiError(const TCHAR* What)
	{
		return FString::Printf(TEXT("%s: %s"), What, UTF8_TO_TCHAR(rbGetLastError()));
	}

	FString AssetPath(const TCHAR* File)
	{
		// The fracture lab's Shardwright assets (Git LFS), from the repo next to the project.
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../../engine/crates/engine-webgpu/fracture/assets") / File);
	}
}

FRubbleAbiBackend::FRubbleAbiBackend() : SceneAssets(MakeUnique<FRubbleSceneAssets>()) {}

FRubbleAbiBackend::~FRubbleAbiBackend()
{
	DestroyWorld();
}

bool FRubbleAbiBackend::CheckVersion(FString& Error)
{
	const RbVersion V = rbGetVersion();
	if (V.major != RB_VERSION_MAJOR || V.minor != RB_VERSION_MINOR)
	{
		Error = FString::Printf(TEXT("librubble_ffi is ABI %u.%u, rubble.h is %d.%d: run scripts/unreal/build-ffi.sh"), V.major, V.minor, RB_VERSION_MAJOR, RB_VERSION_MINOR);
		return false;
	}
	return true;
}

void FRubbleAbiBackend::DestroyWorld()
{
	Gpu.Reset(); // before the world: it waits for the world's last copies into Unreal's buffers
	if (WorldPtr)
	{
		rbDestroyWorld(WorldPtr);
		WorldPtr = nullptr;
	}
	PlayerVehicle = MAX_uint32;
	AiVehicles.Reset();
	VehicleVisual.Reset();
	NewBodies.Reset();
	NewShapes.Reset();
	ChunkAt.Reset();
	ChunkLook.Reset();
	Balls.Reset();
	NextBall = 0;
	Meteors.Reset();
	NextMeteor = 0;
	SoftCars.Reset();
	SoftCarOfSkin.Reset();
	SoftCarOfSoft.Reset();
	PlayerSoft = INDEX_NONE;
}

FString FRubbleAbiBackend::MockReason(const FRubbleSceneRequest& R)
{
	switch (R.Family)
	{
	case ERubbleFamily::Soft:
		return R.Id == TEXT("crates") ? TEXT("the lattice crates have no preset on the ABI") : FString();
	case ERubbleFamily::Rigid:
		for (const TCHAR* Scripted : { TEXT("joints"), TEXT("ragdolls"), TEXT("ragdolls10k"), TEXT("hingechains10k"), TEXT("motorfield10k") })
		{
			if (R.Id == Scripted)
			{
				return TEXT("the joint scenes' builders are not on the ABI");
			}
		}
		return FString();
	case ERubbleFamily::Fracture:
		return R.Id == TEXT("house") || R.Id == TEXT("arch") || R.Id == TEXT("town") ? FString() : TEXT("procedural destructibles (cuboids, rows, grids) are not on the ABI");
	default:
		return FString();
	}
}

bool FRubbleAbiBackend::Load(const FRubbleSceneRequest& Request, FString& Error)
{
	const FString Reason = MockReason(Request);
	if (!Reason.IsEmpty())
	{
		if (!Mock.Load(Request, Error))
		{
			return false;
		}
		DestroyWorld();
		bMock = true;
		Current = Request;
		NameText = FString::Printf(TEXT("mock: %s"), *Reason);
		return true;
	}
	if (!LoadAbi(Request, Error))
	{
		DestroyWorld();
		return false;
	}
	bMock = false;
	Current = Request;
	NameText = TEXT("rubble (C ABI)");
	return true;
}

bool FRubbleAbiBackend::CreateBodies(const FRubbleSceneAssets& Layout, FString& Error)
{
	TArray<RbBodyDef> Defs;
	TArray<int32> FromLayout;
	for (int32 b = 0; b < Layout.Bodies.Num(); ++b)
	{
		const FRubbleBodyVisual& V = Layout.Bodies[b];
		// Drawn bodies only: chassis and destructible roots are made elsewhere; the mock's
		// parked projectiles and vehicle parts are not part of the scene.
		if (V.NumShapes == 0 || V.Initial.GetLocation().Z < -50000.0 || Layout.Shapes[V.FirstShape].Material == ERubbleMaterial::CarPaint)
		{
			continue;
		}
		RbBodyDef D = rbDefaultBodyDef();
		D.type_ = V.bStatic ? RB_STATIC_BODY : RB_DYNAMIC_BODY;
		D.transform = Xf(V.Initial);
		// Settled stacks start asleep, as the page's settled scenes do; contact wakes them.
		D.is_awake = V.bAwake;
		D.linear_velocity = Vec(V.InitialVelocity);
		D.angular_velocity = Vec(V.InitialAngularVelocity);
		D.user_data = uint64(SceneAssets->Bodies.Num() + FromLayout.Num());
		Defs.Add(D);
		FromLayout.Add(b);
	}
	TArray<RbBodyId> Ids;
	Ids.SetNumZeroed(Defs.Num());
	if (Defs.Num() && rbCreateBodies(WorldPtr, Defs.GetData(), uint32(Defs.Num()), Ids.GetData()) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateBodies"));
		return false;
	}
	for (int32 k = 0; k < FromLayout.Num(); ++k)
	{
		const FRubbleBodyVisual& V = Layout.Bodies[FromLayout[k]];
		FRubbleBodyVisual Out = V;
		Out.Id = FRubbleBodyId(Ids[k]);
		Out.FirstShape = SceneAssets->Shapes.Num();
		for (int32 s = 0; s < V.NumShapes; ++s)
		{
			const FRubbleShapeVisual& Shape = Layout.Shapes[V.FirstShape + s];
			SceneAssets->Shapes.Add(Shape);
			RbShapeDef SD = rbDefaultShapeDef();
			SD.density = Density(Shape.Material);
			if (V.Mass > 0.0f && Shape.Kind == ERubbleShape::Box)
			{
				// A body of a given mass (the 1,500 kg ram, the 18 t truck, the page's 0.25 kg bricks).
				SD.density = V.Mass / (8.0f * Shape.HalfExtents.X * Shape.HalfExtents.Y * Shape.HalfExtents.Z);
			}
			else if (V.Mass > 0.0f && Shape.Kind == ERubbleShape::Sphere)
			{
				SD.density = V.Mass / (4.0f / 3.0f * UE_PI * FMath::Cube(Shape.Radius));
			}
			SD.friction = Layout.Friction > 0.0f ? Layout.Friction : 0.6f;
			RbShapeId Made = 0;
			RbResult R = RB_OK;
			switch (Shape.Kind)
			{
			case ERubbleShape::Box: R = rbCreateBoxShape(WorldPtr, Ids[k], &SD, Vec(FVector(Shape.HalfExtents)), Xf(Shape.Local), &Made); break;
			case ERubbleShape::Sphere: R = rbCreateSphereShape(WorldPtr, Ids[k], &SD, Vec(Shape.Local.GetLocation()), Shape.Radius, &Made); break;
			case ERubbleShape::Capsule: R = rbCreateCapsuleShape(WorldPtr, Ids[k], &SD, Vec(Shape.Local.GetLocation()), Shape.Radius, Shape.HalfHeight, &Made); break;
			case ERubbleShape::Hull:
			{
				TArray<RbVec3> Points;
				for (const FVector3f& P : Layout.Meshes[Shape.Mesh].Positions)
				{
					Points.Add(RbVec3{ P.X, P.Y, P.Z });
				}
				R = rbCreateHullShape(WorldPtr, Ids[k], &SD, Points.GetData(), uint32(Points.Num()), Xf(Shape.Local), &Made);
				break;
			}
			}
			if (R != RB_OK)
			{
				Error = AbiError(TEXT("create shape"));
				return false;
			}
			// Soft-body scenes: their skins collide with every rigid shape (walls, poles, rams).
			if (Current.Family == ERubbleFamily::Soft && rbSoft_AddObstacle(WorldPtr, Made) != RB_OK)
			{
				Error = AbiError(TEXT("rbSoft_AddObstacle"));
				return false;
			}
		}
		SceneAssets->Bodies.Add(Out);
	}
	return true;
}

bool FRubbleAbiBackend::CreateDestructible(const TCHAR* File, const FTransform& Placement, FString& Error)
{
	const FString Path = AssetPath(File);
	TArray<uint8> Bytes;
	if (!FFileHelper::LoadFileToArray(Bytes, *Path))
	{
		Error = FString::Printf(TEXT("%s not found (Git LFS: git lfs pull)"), *Path);
		return false;
	}
	RubbleGlb::FFile Glb;
	if (!RubbleGlb::Read(Bytes, Glb, Error))
	{
		Error = FString::Printf(TEXT("%s: %s"), File, *Error);
		return false;
	}
	RbDestructibleAssetId Asset = 0;
	// Level 0: what the fracture lab loads (samples.rs: the first level present), the arch's 23
	// stones and the house's 131 parts.
	if (rbCreateDestructibleAssetFromGlb(WorldPtr, Bytes.GetData(), size_t(Bytes.Num()), 0, &Asset) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateDestructibleAssetFromGlb"));
		return false;
	}
	// The fracture lab's setup: the solver settings the asset was tuned with (120 Hz, 8
	// iterations), and its bonds preloaded by settling one copy, so the structure starts at rest.
	float TimeStep = 0.0f;
	bool bSettled = false;
	if (rbWorld_UseDestructibleAssetSolver(WorldPtr, Asset, &TimeStep) != RB_OK || rbWorld_PreloadDestructibleAsset(WorldPtr, Asset, &bSettled) != RB_OK)
	{
		Error = AbiError(TEXT("the lab's solver and preload"));
		return false;
	}
	Dt = TimeStep;
	if (!bSettled)
	{
		UE_LOG(LogRubble, Warning, TEXT("%s: the preload did not settle in 600 steps"), File);
	}
	RbDestructibleDef D = rbDefaultDestructibleDef();
	D.asset = Asset;
	D.transform = Xf(Placement);
	RbDestructibleId Id = 0;
	uint32 Count = 0;
	if (rbCreateDestructible(WorldPtr, &D, &Id) != RB_OK || rbDestructible_GetChunkCount(WorldPtr, Id, &Count) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateDestructible"));
		return false;
	}
	FRubbleDestructibleVisual V;
	V.Placement = Placement;
	const int32 DIndex = SceneAssets->Destructibles.Num();
	TMap<int32, int32> MeshOfNode;
	TArray<RbBodyId> Bodies;
	TArray<int32> Nodes;
	Bodies.SetNumZeroed(Count);
	Nodes.Init(-1, Count);
	for (uint32 c = 0; c < Count; ++c)
	{
		if (rbDestructible_GetChunkBody(WorldPtr, Id, c, &Bodies[c]) != RB_OK || rbDestructible_GetChunkNode(WorldPtr, Id, c, &Nodes[c]) != RB_OK)
		{
			Error = AbiError(TEXT("chunk query"));
			return false;
		}
	}
	// Where the asset frame sits in Rubble's: engine-fracture build_scene centres the selected
	// level in X and Z and puts its lowest point on the ground (placement = -centre.x, -lo.y,
	// -centre.z of its collision hulls). The ABI does not report it (chunk_local is identity),
	// so it is taken the same way from the level's render meshes, which match the hulls' bounds
	// to within about half a millimetre. In Unreal axes: -centre.X, -centre.Y, -lo.Z.
	FBox Level(ForceInit);
	for (uint32 c = 0; c < Count; ++c)
	{
		if (Glb.NodeMeshes.IsValidIndex(Nodes[c]))
		{
			for (const FVector3f& P : Glb.NodeMeshes[Nodes[c]].Positions)
			{
				Level += FVector(P);
			}
		}
	}
	const FVector3f Shift = Level.IsValid ? FVector3f(-Level.GetCenter().X, -Level.GetCenter().Y, -Level.Min.Z) : FVector3f::ZeroVector;
	for (FRubbleMeshData& M : Glb.NodeMeshes)
	{
		for (FVector3f& P : M.Positions)
		{
			P += Shift;
		}
	}
	for (uint32 c = 0; c < Count; ++c)
	{
		const RbBodyId Body = Bodies[c];
		const int32 Node = Nodes[c];
		if (!Glb.NodeMeshes.IsValidIndex(Node) || Glb.NodeMeshes[Node].NumTriangles() == 0)
		{
			continue; // a chunk with nothing to draw (the ABI names no node)
		}
		FRubbleChunkVisual Chunk;
		if (const int32* Known = MeshOfNode.Find(Node))
		{
			Chunk.Mesh = *Known;
		}
		else
		{
			Chunk.Mesh = SceneAssets->Meshes.Add(Glb.NodeMeshes[Node]);
			MeshOfNode.Add(Node, Chunk.Mesh);
		}
		Chunk.Destructible = DIndex;
		Chunk.Rest = FTransform::Identity; // the Bonds model: body origin = instance frame, mesh in the asset frame
		Chunk.Body = FRubbleBodyId(Body);
		V.Chunks.Add(SceneAssets->Chunks.Add(Chunk));
		ChunkAt.Add(Body, Placement.TransformPosition(FVector(Glb.NodeMeshes[Node].Positions.Num() ? Glb.NodeMeshes[Node].Positions[0] : FVector3f::ZeroVector)));
	}
	SceneAssets->Destructibles.Add(V);
	UE_LOG(LogRubble, Display, TEXT("%s: %u chunks, %d drawn"), File, Count, V.Chunks.Num());
	return true;
}

bool FRubbleAbiBackend::CreateSoftCars(const FRubbleSceneAssets& Layout, FString& Error)
{
	// The solver settings the demos use at this step rate.
	FString Settings;
	const float Hz = float(1.0 / Dt);
	if (!AbiJson([Hz](char* B, size_t C, size_t* N) { return rbGetCalibratedSoftSettingsJson(Hz, B, C, N); }, Settings)
		|| rbWorld_SetSoftSettingsJson(WorldPtr, TCHAR_TO_UTF8(*Settings)) != RB_OK)
	{
		Error = AbiError(TEXT("soft settings"));
		return false;
	}
	FString Desc;
	TSharedPtr<FJsonObject> Root;
	if (!AbiJson([](char* B, size_t C, size_t* N) { return rbGetSoftVehiclePresetJson("procedural_car", 0.0f, B, C, N); }, Desc)
		|| !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Desc), Root) || !Root.IsValid())
	{
		Error = AbiError(TEXT("the procedural car's descriptor"));
		return false;
	}
	// Its node cage in Unreal axes (descriptor frame), and which way it faces: its steered
	// wheels are at the front.
	TArray<FVector> Nodes;
	for (const TSharedPtr<FJsonValue>& N : Root->GetArrayField(TEXT("nodes")))
	{
		Nodes.Add(PointOf(N->AsObject()->GetArrayField(TEXT("position"))));
	}
	TArray<TPair<int32, int32>> Axles;
	TArray<float> Radii;
	double FrontX = 0.0;
	for (const TSharedPtr<FJsonValue>& W : Root->GetArrayField(TEXT("wheels")))
	{
		const TSharedPtr<FJsonObject> O = W->AsObject();
		const int32 A = int32(O->GetNumberField(TEXT("node1"))), B = int32(O->GetNumberField(TEXT("node2")));
		Axles.Add({ A, B });
		Radii.Add(float(O->GetNumberField(TEXT("radius")) * 100.0));
		if (O->GetBoolField(TEXT("steerable")) && Nodes.IsValidIndex(A))
		{
			FrontX += Nodes[A].X;
		}
	}
	// The layout's cars face +X; the descriptor's front may be -X: then it is turned to match.
	const FQuat Turn = FrontX < 0.0 ? FQuat(FVector::UpVector, UE_PI) : FQuat::Identity;
	FBox Cage(ForceInit);
	for (const FVector& N : Nodes)
	{
		Cage += Turn.RotateVector(N);
	}
	const FTCHARToUTF8 DescUtf8(*Desc);
	for (const FRubbleSoftVisual& V : Layout.SoftBodies)
	{
		if (!V.bCar)
		{
			continue;
		}
		// Fit the layout's car shell (front +X) to the cage, so the skin wraps the simulated car.
		FRubbleMeshData& Mesh = SceneAssets->Meshes[V.Mesh];
		FBox Shell(ForceInit);
		for (const FVector3f& P : Mesh.Positions)
		{
			Shell += FVector(P);
		}
		const FVector Scale = Cage.GetSize() / Shell.GetSize().ComponentMax(FVector(1.0));
		for (FVector3f& P : Mesh.Positions)
		{
			P = FVector3f(Cage.Min + (FVector(P) - Shell.Min) * Scale);
		}
		// Spawned a few centimetres up: the layout's origin is at its sag below the ground.
		const FTransform Spawn(V.Spawn.GetRotation(), V.Spawn.GetLocation() + FVector(0, 0, 10));
		const FTransform AbiSpawn(Spawn.GetRotation() * Turn, Spawn.GetLocation());
		RbSoftId Soft = 0;
		if (rbCreateSoftVehicle(WorldPtr, DescUtf8.Get(), nullptr, Xf(AbiSpawn), Vec(V.Velocity), &Soft) != RB_OK
			|| rbSoft_EnableOutput(WorldPtr, Soft, true) != RB_OK)
		{
			Error = AbiError(TEXT("rbCreateSoftVehicle"));
			return false;
		}
		// In the car's own (descriptor) frame, not the world: rbSoft_BindSkin documents "world
		// coordinates at spawn", but SoftHost::bind_mesh binds against the descriptor's node
		// positions, so a world-space mesh binds to the wrong nodes (README, C ABI gaps).
		TArray<RbVec3> World;
		const FQuat Unturn = Turn.Inverse();
		for (const FVector3f& P : Mesh.Positions)
		{
			World.Add(Vec(Unturn.RotateVector(FVector(P))));
		}
		RbSkinId Skin = 0;
		if (rbSoft_BindSkin(WorldPtr, Soft, World.GetData(), uint32(World.Num()), Mesh.Indices.GetData(), uint32(Mesh.NumTriangles()), &Skin) != RB_OK)
		{
			Error = AbiError(TEXT("rbSoft_BindSkin"));
			return false;
		}
		FSoftCar Car;
		Car.Soft = Soft;
		Car.Skin = Skin;
		Car.Visual = SceneAssets->SoftBodies.Num();
		Car.bAI = V.bAI;
		Car.WheelNodes = Axles;
		Car.WheelRadius = Radii;
		Car.Spin.Init(0.0f, Axles.Num());
		Car.LastHub.Init(FVector::ZeroVector, Axles.Num());
		Car.Triangles = Mesh.Indices;
		// Rest edge lengths around each vertex: strain and tears are measured against them.
		Car.Neighbours.SetNum(Mesh.Positions.Num());
		Car.RestLength.SetNum(Mesh.Positions.Num());
		for (int32 t = 0; t + 2 < Mesh.Indices.Num(); t += 3)
		{
			for (int32 k = 0; k < 3; ++k)
			{
				const int32 a = Mesh.Indices[t + k], b = Mesh.Indices[t + (k + 1) % 3];
				if (!Car.Neighbours[a].Contains(b))
				{
					const float L = FVector3f::Dist(Mesh.Positions[a], Mesh.Positions[b]);
					Car.Neighbours[a].Add(b);
					Car.RestLength[a].Add(L);
					Car.Neighbours[b].Add(a);
					Car.RestLength[b].Add(L);
				}
			}
		}
		FRubbleSoftVisual Out = V;
		Out.Soft = Car.Visual;
		Out.Wheels.Reset();
		for (const float R : Radii)
		{
			FRubbleWheelDef D;
			D.Radius = R;
			D.Width = 20.0f;
			Out.Wheels.Add(D);
		}
		if (V.bPlayer)
		{
			PlayerSoft = SoftCars.Num();
		}
		SoftCarOfSkin.Add(Skin, SoftCars.Num());
		SoftCarOfSoft.Add(Soft, SoftCars.Num());
		SoftCars.Add(MoveTemp(Car));
		SceneAssets->SoftBodies.Add(Out);
	}
	return true;
}

bool FRubbleAbiBackend::LoadAbi(const FRubbleSceneRequest& Request, FString& Error)
{
	if (!CheckVersion(Error))
	{
		return false;
	}
	// The layout the mock would draw: the same scene, made real.
	if (!Mock.Load(Request, Error))
	{
		return false;
	}
	const FRubbleSceneAssets& Layout = Mock.Assets();
	DestroyWorld();
	Dt = 1.0 / 60.0;
	Current = Request;
	SceneAssets = MakeUnique<FRubbleSceneAssets>();
	SceneAssets->Stage = Layout.Stage;
	SceneAssets->GroundHalfExtentCm = Layout.GroundHalfExtentCm;
	SceneAssets->CameraEye = Layout.CameraEye;
	SceneAssets->CameraTarget = Layout.CameraTarget;
	SceneAssets->Meshes = Layout.Meshes; // hull meshes keep their indices

	int32 NumBodies = Layout.Bodies.Num() + 64;
	RbWorldDef W = rbDefaultWorldDef();
	W.coordinates = RB_COORDINATES_UNREAL;
	W.gravity = RbVec3{ 0.0f, 0.0f, -981.0f };
	if (Request.Family == ERubbleFamily::Soft)
	{
		// Soft scenes step at 240 Hz (softbody.html) with canonical contacts and a soft arena; a
		// procedural car is a few hundred node bodies.
		Dt = 1.0 / 240.0;
		W.enable_soft_bodies = true;
		W.contacts = RB_CONTACTS_CANONICAL;
		NumBodies += Layout.SoftBodies.Num() * 600;
		W.max_joints = FMath::Max(W.max_joints, uint32(Layout.SoftBodies.Num() * 16 + 4096));
	}
	W.time_step = float(Dt);
	// Hits from 1 m/s (the threshold is in the world's units, cm/s here).
	W.hit_event_threshold = 100.0f;
	if (Layout.SolverIterations > 0)
	{
		W.iterations = uint32(Layout.SolverIterations);
	}
	W.enable_sleep = Layout.bSleep;
	W.reuse_contacts = Layout.bReuseContacts;
	if (Request.Family == ERubbleFamily::Fracture)
	{
		NumBodies += 4096;
		W.max_joints = FMath::Max(W.max_joints, 32768u);
		W.max_hull_vertices = FMath::Max(W.max_hull_vertices, 1u << 20);
	}
	if (Request.Family == ERubbleFamily::Fracture && Request.Id == TEXT("town"))
	{
		// Vibe Town: ~52k chunks, ~133k bonds, ~25k hull chunks of up to 42 points.
		NumBodies += 64000;
		W.max_joints = FMath::Max(W.max_joints, 1u << 18);
		W.max_hull_vertices = FMath::Max(W.max_hull_vertices, 1u << 21);
	}
	W.max_bodies = FMath::Max(W.max_bodies, uint32(NumBodies * 1.25));
	W.max_colliders = FMath::Max(W.max_colliders, uint32(NumBodies * 2));
	// Bricks touch all their neighbours: demo.html budgets ~29 contacts a body (ring: 262,144 for
	// 9k); 32 here. Overflow is loud (rbWorld_Poll fails with the flag), never a silent drop.
	W.max_contacts = FMath::Max(W.max_contacts, FMath::Min(uint32(NumBodies * 32), 1u << 22));
	W.max_candidates = FMath::Max(W.max_candidates, FMath::Min(uint32(NumBodies * 32), 1u << 22));
	if (Request.Family == ERubbleFamily::Rigid)
	{
		// demo.html's sceneCap: power-of-two tiers, contacts and candidates capped at 2^21.
		const uint32 Tier = FMath::RoundUpToPowerOfTwo(uint32(FMath::Max(NumBodies, 4096)));
		W.max_bodies = FMath::Max(W.max_bodies, Tier);
		W.max_colliders = FMath::Max(W.max_colliders, FMath::RoundUpToPowerOfTwo(FMath::Max(Tier * 2, 8192u)));
		W.max_contacts = FMath::Min(FMath::RoundUpToPowerOfTwo(FMath::Max(Tier * 16, 32768u)), 1u << 21);
		W.max_candidates = FMath::Min(FMath::RoundUpToPowerOfTwo(FMath::Max(Tier * 32, 65536u)), 1u << 21);
	}
	// On Unreal's own GPU device when the platform allows (the GPU Scene renderers then read
	// the poses in place); -RubbleReadback keeps the world on its own device, read back.
	FString NoGpu;
	if (!FParse::Param(FCommandLine::Get(), TEXT("RubbleReadback")) && FRubbleGpuPoses::Supported(NoGpu))
	{
		void* Device = nullptr;
		void* Queue = nullptr;
		if (!FRubbleGpuPoses::DeviceAndQueue(Device, Queue, Error))
		{
			return false;
		}
		const RbResult R = rbCreateWorldOnMetal(&W, Device, Queue, &WorldPtr);
		FRubbleGpuPoses::ReleaseQueue(Queue); // the world holds its own reference
		if (R != RB_OK)
		{
			Error = AbiError(TEXT("rbCreateWorldOnMetal"));
			return false;
		}
		Gpu = MakeUnique<FRubbleGpuPoses>(WorldPtr, W.max_bodies);
		if (!Gpu->IsValid())
		{
			Gpu.Reset();
		}
	}
	else if (rbCreateWorld(&W, &WorldPtr) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateWorld"));
		return false;
	}

	// Ground: an infinite plane at Z = 0 (rbCreatePlaneShape: no large static in the broadphase).
	if (Layout.GroundHalfExtentCm > 0.0f)
	{
		RbBodyDef G = rbDefaultBodyDef();
		RbBodyId Ground = 0;
		RbShapeDef SD = rbDefaultShapeDef();
		SD.friction = 0.8f;
		RbShapeId Shape = 0;
		if (rbCreateBody(WorldPtr, &G, &Ground) != RB_OK || rbCreatePlaneShape(WorldPtr, Ground, &SD, RbVec3{ 0, 0, 1 }, 0.0f, &Shape) != RB_OK)
		{
			Error = AbiError(TEXT("ground"));
			return false;
		}
	}
	if (!CreateBodies(Layout, Error))
	{
		return false;
	}
	if (Request.Family == ERubbleFamily::Soft && !CreateSoftCars(Layout, Error))
	{
		return false;
	}
	switch (Request.Family)
	{
	case ERubbleFamily::Vehicles:
	{
		for (const FRubbleVehicleVisual& V : Layout.Vehicles)
		{
			const FRubbleBodyVisual* Chassis = Layout.Bodies.FindByPredicate([&V](const FRubbleBodyVisual& B) { return B.Id == V.Chassis; });
			// Spawned a little high: the preset settles onto its suspension.
			const FTransform At = (Chassis ? Chassis->Initial : FTransform::Identity) * FTransform(FVector(0, 0, 30));
			RbVehicleId Id = 0;
			if (rbCreateVehiclePreset(WorldPtr, PresetOf(V.Style, V.bPlayer && Request.Id == TEXT("destruction")), Xf(At), &Id) != RB_OK)
			{
				Error = AbiError(TEXT("rbCreateVehiclePreset"));
				return false;
			}
			FRubbleVehicleVisual Out = V;
			// The preset's own bodywork and wheels: drawn as the engine simulates them.
			PresetVisual(PresetOf(V.Style, V.bPlayer && Request.Id == TEXT("destruction")), Out);
			Out.Vehicle = SceneAssets->Vehicles.Num();
			Out.Chassis = FRubbleBodyId(); // the ABI does not name the chassis body: placed from the wheels
			VehicleVisual.Add(Id, Out.Vehicle);
			SceneAssets->Vehicles.Add(Out);
			if (V.bPlayer)
			{
				PlayerVehicle = Id;
			}
			else
			{
				AiVehicles.Add(Id);
			}
		}
		break;
	}
	case ERubbleFamily::Fracture:
		if (Request.Id == TEXT("town"))
		{
			if (!CreateTown(Error))
			{
				return false;
			}
		}
		else if (!CreateDestructible(Request.Id == TEXT("house") ? TEXT("veneer-house.glb") : TEXT("masonry-arch.glb"), FTransform::Identity, Error))
		{
			return false;
		}
		break;
	default:
		break;
	}
	// The lab's auto shot, from the camera at the structure, fired by Step.
	Steps = 0;
	// The rigid smash scenes fire one too (demo.html launches a projectile into them).
	const bool bSmash = Request.Family == ERubbleFamily::Rigid && (Request.Id == TEXT("wall") || Request.Id.StartsWith(TEXT("ring")) || Request.Id.StartsWith(TEXT("gables")));
	const float Auto = Request.Family == ERubbleFamily::Fracture ? Request.Param(TEXT("autoshot"), 1.5f) : bSmash ? 1.0f : 0.0f;
	AutoShotStep = Auto > 0.0f ? int64(Auto / Dt) : -1;
	if (!ParkBalls(12, Error))
	{
		return false;
	}
	UE_LOG(LogRubble, Display, TEXT("rubble: %s built on the C ABI (%d bodies, %d vehicles, %d chunks)"), *Request.ToString(), SceneAssets->Bodies.Num(), SceneAssets->Vehicles.Num(), SceneAssets->Chunks.Num());
	return true;
}

void FRubbleAbiBackend::Step()
{
	if (bMock)
	{
		Mock.Step();
		return;
	}
	if (!WorldPtr)
	{
		return;
	}
	// AI traffic: a steady throttle on a turn, so the loop the mock drives is driven for real.
	for (const uint32 V : AiVehicles)
	{
		rbVehicle_SetInput(WorldPtr, V, RbVehicleInput{ 0.35f, 0.2f, 0.0f, 0.0f });
	}
	for (const FSoftCar& Car : SoftCars)
	{
		if (Car.bAI)
		{
			rbSoft_SetInput(WorldPtr, Car.Soft, RbVehicleInput{ 0.4f, 0.25f, 0.0f, 0.0f });
		}
	}
	if (AutoShotStep >= 0 && int64(Steps) == AutoShotStep)
	{
		const FVector From = SceneAssets->CameraEye, Aim = SceneAssets->CameraTarget + FVector(0, 0, 50);
		const double T = (Aim - From).Size() / 3500.0;
		Launch(From, (Aim - From) + FVector(0, 0, 0.5 * 981.0 * T * T));
	}
	++Steps;
	if (rbWorld_Step(WorldPtr) != RB_OK)
	{
		UE_LOG(LogRubble, Error, TEXT("%s"), *AbiError(TEXT("rbWorld_Step")));
	}
}

void FRubbleAbiBackend::Poll(TArray<FRubbleFrame>& Out)
{
	if (bMock)
	{
		Mock.Poll(Out);
		return;
	}
	if (!WorldPtr)
	{
		return;
	}
	// One step at a time (not coalesced): the subsystem merges them, interpolating across the
	// latest step. The frame's pointers are valid until the next poll, so it is copied now.
	for (int32 Guard = 0; Guard < 64; ++Guard)
	{
		RbFrame F;
		if (rbWorld_Poll(WorldPtr, /*coalesce=*/false, /*wait=*/false, &F) != RB_OK)
		{
			UE_LOG(LogRubble, Error, TEXT("%s"), *AbiError(TEXT("rbWorld_Poll")));
			return;
		}
		if (!F.has_frame)
		{
			break;
		}
		FRubbleFrame& R = Out.AddDefaulted_GetRef();
		R.Step = F.last_step;
		R.bDropped = F.dropped;
		R.LostEvents = F.lost_events;
		R.Moves.Reserve(F.move_count);
		for (uint32 i = 0; i < F.move_count; ++i)
		{
			const RbBodyMove& M = F.moves[i];
			FRubbleMove& Move = R.Moves.AddDefaulted_GetRef();
			Move.Body = FRubbleBodyId(M.body);
			Move.Transform = Xf(M.transform);
			Move.Previous = M.has_previous ? Xf(M.previous) : Move.Transform;
			Move.bHasPrevious = M.has_previous;
			Move.bFellAsleep = M.fell_asleep;
			Move.bWoke = M.woke;
			Move.UserData = M.user_data;
			if (FVector* At = ChunkAt.Find(M.body))
			{
				*At = Move.Transform.GetLocation();
			}
		}
		for (uint32 i = 0; i < F.destruction_count; ++i)
		{
			const RbDestructionEvent& D = F.destruction[i];
			FRubbleEvent E;
			E.Magnitude = D.force;
			switch (D.type_)
			{
			case RB_JOINT_BROKE:
			case RB_CONNECTION_BROKE:
			case RB_MOUNT_BROKE:
				E.Type = ERubbleEventType::JointBroke;
				break;
			case RB_BEAM_BROKE:
				E.Type = ERubbleEventType::BeamBroke;
				E.Soft = int32(D.soft);
				break;
			case RB_PIECE_DETACHED:
				E.Type = ERubbleEventType::PieceDetached;
				E.Soft = int32(D.soft);
				break;
			case RB_BONDS_BROKEN:
			case RB_SPLIT:
			{
				// Bonds model: chunk bodies never change, so a split is an effect, not a re-binding
				// for the renderer (Destructible stays INDEX_NONE). Where: the first child's chunk.
				E.Type = ERubbleEventType::BondsBroken;
				E.Count = int32(D.bond_count);
				const RbActor* Actor = D.child_count ? &D.children[0] : &D.parent;
				if (Actor->chunk_count)
				{
					if (const FVector* At = ChunkAt.Find(Actor->bodies[0]))
					{
						E.Position = *At;
					}
					if (const TPair<ERubbleMaterial, FLinearColor>* Look = ChunkLook.Find(Actor->bodies[0]))
					{
						E.MaterialA = Look->Key;
						E.Color = Look->Value;
					}
				}
				break;
			}
			}
			// Events carry no point yet (asked of the API): breaks in a destructible take a chunk's.
			if (E.Position.IsZero() && ChunkAt.Num())
			{
				for (const FRubbleMove& M : R.Moves)
				{
					if (M.bWoke && ChunkAt.Contains(M.Body.Bits))
					{
						E.Position = M.Transform.GetLocation();
						break;
					}
				}
			}
			if (!E.Position.IsZero())
			{
				R.Events.Add(E);
			}
		}
		for (uint32 v = 0; v < F.vehicle_count; ++v)
		{
			const RbVehicleOutput& V = F.vehicles[v];
			const int32* Visual = VehicleVisual.Find(V.vehicle);
			for (uint32 w = 0; Visual && w < V.wheel_count; ++w)
			{
				const RbWheelState& S = V.wheels[w];
				FRubbleWheelVisual WV;
				WV.Vehicle = *Visual;
				WV.Wheel = int32(w);
				WV.Hub = Xf(S.transform); // spin and steer included
				WV.bOnGround = S.in_contact;
				WV.Radius = S.in_contact ? float(FVector::Dist(WV.Hub.GetLocation(), Vec(S.contact_point))) : 0.0f;
				R.Wheels.Add(WV);
			}
		}
		for (uint32 k = 0; k < F.skin_count; ++k)
		{
			const RbSkinOutput& SkinOut = F.skins[k];
			const int32* CarIndex = SoftCarOfSkin.Find(SkinOut.skin);
			if (!CarIndex)
			{
				continue;
			}
			const FSoftCar& Car = SoftCars[*CarIndex];
			FRubbleSoftSkin& Skin = R.Skins.AddDefaulted_GetRef();
			Skin.Soft = Car.Visual;
			Skin.Positions.SetNumUninitialized(SkinOut.vertex_count);
			Skin.Normals.SetNumUninitialized(SkinOut.vertex_count);
			for (uint32 v = 0; v < SkinOut.vertex_count; ++v)
			{
				Skin.Positions[v] = FVector3f(SkinOut.positions[v].x, SkinOut.positions[v].y, SkinOut.positions[v].z);
				Skin.Normals[v] = FVector3f(SkinOut.normals[v].x, SkinOut.normals[v].y, SkinOut.normals[v].z);
			}
			// Strain: the largest stretch or squash of the edges around a vertex; an edge past
			// 1.8 times its rest length tears its triangles (soft_skin's rule).
			Skin.Strain.SetNumZeroed(SkinOut.vertex_count);
			for (int32 v = 0; v < Car.Neighbours.Num() && v < int32(SkinOut.vertex_count); ++v)
			{
				float Worst = 0.0f;
				for (int32 j = 0; j < Car.Neighbours[v].Num(); ++j)
				{
					const float L = FVector3f::Dist(Skin.Positions[v], Skin.Positions[Car.Neighbours[v][j]]);
					Worst = FMath::Max(Worst, FMath::Abs(L / FMath::Max(Car.RestLength[v][j], 0.01f) - 1.0f));
				}
				// Below 5% is the skinning's own give, not damage.
				Skin.Strain[v] = FMath::Clamp((Worst - 0.05f) * 1.5f, 0.0f, 1.0f);
			}
			Skin.TornTriangles.Init(false, Car.Triangles.Num() / 3);
			for (int32 t = 0; t + 2 < Car.Triangles.Num(); t += 3)
			{
				for (int32 e = 0; e < 3; ++e)
				{
					const int32 a = Car.Triangles[t + e], b = Car.Triangles[t + (e + 1) % 3];
					const int32 j = Car.Neighbours[a].IndexOfByKey(b);
					if (j != INDEX_NONE && FVector3f::Dist(Skin.Positions[a], Skin.Positions[b]) > 1.8f * Car.RestLength[a][j] + 2.0f)
					{
						Skin.TornTriangles[t / 3] = true;
					}
				}
			}
		}
		for (uint32 k = 0; k < F.soft_count; ++k)
		{
			const RbSoftOutput& SoftOut = F.soft[k];
			const int32* CarIndex = SoftCarOfSoft.Find(SoftOut.soft);
			if (!CarIndex)
			{
				continue;
			}
			FSoftCar& Car = SoftCars[*CarIndex];
			for (int32 w = 0; w < Car.WheelNodes.Num(); ++w)
			{
				const uint32 A = uint32(Car.WheelNodes[w].Key), B = uint32(Car.WheelNodes[w].Value);
				if (A >= SoftOut.node_count || B >= SoftOut.node_count)
				{
					continue;
				}
				const FVector PA = Vec(SoftOut.nodes[A]), PB = Vec(SoftOut.nodes[B]);
				const FVector Hub = 0.5 * (PA + PB);
				const FVector Axle = (PB - PA).GetSafeNormal(1e-6, FVector::RightVector);
				const FVector Up = (FVector::UpVector - Axle * FVector::DotProduct(FVector::UpVector, Axle)).GetSafeNormal(1e-6, FVector::UpVector);
				const FVector Fwd = FVector::CrossProduct(Axle, Up);
				// Spin from how far the hub rolled along its heading this step.
				if (!Car.LastHub[w].IsZero())
				{
					Car.Spin[w] = FMath::Fmod(Car.Spin[w] + float(FVector::DotProduct(Hub - Car.LastHub[w], Fwd)) / FMath::Max(Car.WheelRadius[w], 1.0f), UE_TWO_PI);
				}
				Car.LastHub[w] = Hub;
				FRubbleWheelVisual WV;
				WV.Vehicle = Car.Visual;
				WV.bSoft = true;
				WV.Wheel = w;
				WV.Hub = FTransform(FMatrix(Fwd, Axle, Up, FVector::ZeroVector).ToQuat(), Hub);
				WV.SpinRad = Car.Spin[w];
				WV.Radius = Car.WheelRadius[w];
				R.Wheels.Add(WV);
			}
		}
		if (NewBodies.Num())
		{
			R.NewBodies = MoveTemp(NewBodies);
			R.NewShapes = MoveTemp(NewShapes);
			NewBodies.Reset();
			NewShapes.Reset();
		}
	}
}

void FRubbleAbiBackend::SetVehicleInput(const FRubbleVehicleInput& In)
{
	if (bMock)
	{
		Mock.SetVehicleInput(In);
		return;
	}
	Input = In;
	const RbVehicleInput Drive{ In.Throttle, In.Steer, In.Brake, In.bHandbrake ? 1.0f : 0.0f };
	if (WorldPtr && PlayerVehicle != MAX_uint32)
	{
		rbVehicle_SetInput(WorldPtr, PlayerVehicle, Drive);
	}
	if (WorldPtr && SoftCars.IsValidIndex(PlayerSoft) && !SoftCars[PlayerSoft].bAI)
	{
		rbSoft_SetInput(WorldPtr, SoftCars[PlayerSoft].Soft, Drive);
	}
}

bool FRubbleAbiBackend::CreateTown(FString& Error)
{
	FString Path = FPaths::ProjectContentDir() / TEXT("Rubble/Town/vibe-town.json");
	FParse::Value(FCommandLine::Get(), TEXT("RubbleTownPack="), Path);
	TArray<uint8> Bytes;
	if (!FFileHelper::LoadFileToArray(Bytes, *Path))
	{
		Error = FString::Printf(TEXT("no town pack at %s (scripts/unreal/fetch-town.sh copies it from vibe-land)"), *Path);
		return false;
	}
	const double Start = FPlatformTime::Seconds();
	RbDestructibleAssetId Asset = 0;
	if (rbCreateDestructibleAssetFromPack(WorldPtr, Bytes.GetData(), size_t(Bytes.Num()), &Asset) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateDestructibleAssetFromPack"));
		return false;
	}
	// The solver the pack is tuned with (the fracture lab's: 120 Hz, 8 iterations, its contact,
	// inertia, sleep and post-stabilization flags). Under World's defaults (60 Hz, none of those
	// flags) a struck town gains energy and throws its chunks kilometres (unreal/PLAN.md).
	float TimeStep = 0.0f;
	if (rbWorld_UseDestructibleAssetSolver(WorldPtr, Asset, &TimeStep) != RB_OK)
	{
		Error = AbiError(TEXT("the town's solver"));
		return false;
	}
	Dt = TimeStep;
	RbDestructibleDef D = rbDefaultDestructibleDef();
	D.asset = Asset;
	D.anchored = true;     // its buried footings and road subgrade are fixed
	D.start_asleep = true; // a standing town, until something hits it
	// ?strength= scales every bond (1: the pack's own limits). See unreal/PLAN.md: at 1 the town
	// does not hold its own weight once disturbed under World's bond model.
	D.strength_scale = Current.Param(TEXT("strength"), 1.0f);
	RbDestructibleId Id = 0;
	uint32 Count = 0;
	if (rbCreateDestructible(WorldPtr, &D, &Id) != RB_OK || rbDestructible_GetChunkCount(WorldPtr, Id, &Count) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateDestructible"));
		return false;
	}
	const double Physics = FPlatformTime::Seconds();
	RubbleTownPack::FPack Pack;
	if (!RubbleTownPack::Parse(Bytes, Pack, Error))
	{
		Error = FString::Printf(TEXT("%s: %s"), *Path, *Error);
		return false;
	}
	TArray<uint8> Catalog;
	if (FFileHelper::LoadFileToArray(Catalog, *(FPaths::GetPath(Path) / TEXT("catalog.json"))))
	{
		RubbleTownPack::ApplyCatalog(Catalog, Pack);
	}
	// One mesh per hull (the library's shared shapes and any node's own), built on first use.
	TArray<int32> MeshOfHull;
	MeshOfHull.Init(INDEX_NONE, Pack.Hulls.Num());
	int32 Drawn = 0;
	for (uint32 c = 0; c < Count; ++c)
	{
		RbBodyId Body = 0;
		int32 Node = -1;
		RbTransform T, Pose, Frame;
		// Every chunk body's origin is the destructible's frame. The node's collider is drawn in
		// its node frame there (rbDestructible_GetChunkNodeFrame); its pose (centre of mass,
		// principal axes) is where its breaks happen.
		if (rbDestructible_GetChunkBody(WorldPtr, Id, c, &Body) != RB_OK || rbDestructible_GetChunkNode(WorldPtr, Id, c, &Node) != RB_OK
			|| rbBody_GetTransform(WorldPtr, Body, &T) != RB_OK || rbDestructible_GetChunkPose(WorldPtr, Id, c, &Pose) != RB_OK
			|| rbDestructible_GetChunkNodeFrame(WorldPtr, Id, c, &Frame) != RB_OK)
		{
			Error = AbiError(TEXT("town chunk query"));
			return false;
		}
		if (!Pack.Nodes.IsValidIndex(Node))
		{
			continue;
		}
		const RubbleTownPack::FNode& N = Pack.Nodes[Node];
		const RubbleTownPack::FMaterial& M = Pack.Materials[N.Material];
		FRubbleShapeVisual Shape;
		Shape.Material = M.Kind;
		Shape.Tint = M.Color;
		Shape.Local = Xf(Frame);
		Shape.bTownSurface = true;
		Shape.TextureLayer = M.Layer;
		Shape.TilesPerMetre = 1.0f / M.MetresPerTile;
		Shape.UpLayer = M.UpLayer;
		Shape.UpTilesPerMetre = 1.0f / M.UpMetresPerTile;
		Shape.bUpUntinted = M.bUpUntinted;
		Shape.GlazedFacades = N.bGlazed && M.Layer >= 0 ? N.Facades : 0;
		if (N.bBox)
		{
			Shape.Kind = ERubbleShape::Box;
			Shape.HalfExtents = RubbleTownPack::HalfToUnreal(N.HalfExtents);
		}
		else
		{
			if (MeshOfHull[N.Hull] == INDEX_NONE)
			{
				FRubbleMeshData Mesh;
				Mesh.Material = M.Kind;
				MeshOfHull[N.Hull] = RubbleTownPack::HullMesh(Pack.Hulls[N.Hull], Mesh) ? SceneAssets->Meshes.Add(MoveTemp(Mesh)) : -2;
			}
			if (MeshOfHull[N.Hull] < 0)
			{
				continue; // a degenerate hull: nothing to draw (its physics is the pack's)
			}
			Shape.Kind = ERubbleShape::Hull;
			Shape.Mesh = MeshOfHull[N.Hull];
		}
		FRubbleBodyVisual V;
		V.Id = FRubbleBodyId(Body);
		V.Initial = Xf(T);
		V.FirstShape = SceneAssets->Shapes.Add(Shape);
		V.NumShapes = 1;
		V.bLumenVisible = true;
		SceneAssets->Bodies.Add(V);
		ChunkAt.Add(Body, (Xf(Pose) * V.Initial).GetLocation()); // where its breaks happen, for effects
		ChunkLook.Add(Body, TPair<ERubbleMaterial, FLinearColor>(M.Kind, M.Color));
		++Drawn;
	}
	UE_LOG(LogRubble, Display, TEXT("town: '%s', %u chunks (%d drawn, %d meshes, %d materials); physics %.0f ms, visuals %.0f ms"),
		*Pack.Title, Count, Drawn, SceneAssets->Meshes.Num(), Pack.Materials.Num(), 1000.0 * (Physics - Start), 1000.0 * (FPlatformTime::Seconds() - Physics));
	return true;
}

bool FRubbleAbiBackend::ParkBalls(int32 Count, FString& Error)
{
	// Parked high above the scene, asleep: a shot moves one into place (rbBody_SetTransform) and
	// gives it its speed, so balls are reused rather than created per shot.
	// The town's cannon: vibe-land's cannonball scale (a 60 cm iron ball).
	const float Radius = IsTown() ? 60.0f : Current.Family == ERubbleFamily::Rigid ? 50.0f : 25.0f;
	for (int32 i = 0; i < Count; ++i)
	{
		RbBodyDef B = rbDefaultBodyDef();
		B.type_ = RB_DYNAMIC_BODY;
		B.transform = Xf(FTransform(ParkedAt(i)));
		B.is_awake = false;
		// The town's thin roofs and slabs (20 cm) are less than a 45 m/s ball's step: swept (CCD).
		B.is_bullet = IsTown();
		RbBodyId Ball = 0;
		RbShapeDef SD = rbDefaultShapeDef();
		SD.density = 0.0078f;
		RbShapeId Shape = 0;
		if (rbCreateBody(WorldPtr, &B, &Ball) != RB_OK || rbCreateSphereShape(WorldPtr, Ball, &SD, RbVec3{ 0, 0, 0 }, Radius, &Shape) != RB_OK)
		{
			Error = AbiError(TEXT("parked ball"));
			return false;
		}
		Balls.Add(Ball);
		FRubbleBodyVisual V;
		V.Id = FRubbleBodyId(Ball);
		V.Initial = FTransform(ParkedAt(i));
		V.FirstShape = SceneAssets->Shapes.Num();
		V.NumShapes = 1;
		SceneAssets->Bodies.Add(V);
		FRubbleShapeVisual S;
		S.Kind = ERubbleShape::Sphere;
		S.Radius = Radius;
		S.Material = ERubbleMaterial::Steel;
		S.Tint = FLinearColor(0.55f, 0.56f, 0.58f);
		SceneAssets->Shapes.Add(S);
	}
	if (IsTown())
	{
		// Meteors: 1.2 m basalt (about 22 t), parked beside the balls.
		for (int32 i = 0; i < 6; ++i)
		{
			const FVector At = ParkedAt(Count + i) + FVector(0, 2000, 0);
			RbBodyDef B = rbDefaultBodyDef();
			B.type_ = RB_DYNAMIC_BODY;
			B.transform = Xf(FTransform(At));
			B.is_awake = false;
			B.is_bullet = true; // 50 m/s through 20 cm roofs and slabs
			RbBodyId Rock = 0;
			RbShapeDef SD = rbDefaultShapeDef();
			SD.density = 0.003f;
			RbShapeId Shape = 0;
			if (rbCreateBody(WorldPtr, &B, &Rock) != RB_OK || rbCreateSphereShape(WorldPtr, Rock, &SD, RbVec3{ 0, 0, 0 }, 120.0f, &Shape) != RB_OK)
			{
				Error = AbiError(TEXT("parked meteor"));
				return false;
			}
			Meteors.Add(Rock);
			FRubbleBodyVisual V;
			V.Id = FRubbleBodyId(Rock);
			V.Initial = FTransform(At);
			V.FirstShape = SceneAssets->Shapes.Num();
			V.NumShapes = 1;
			SceneAssets->Bodies.Add(V);
			FRubbleShapeVisual S;
			S.Kind = ERubbleShape::Sphere;
			S.Radius = 120.0f;
			S.Material = ERubbleMaterial::Concrete;
			S.Tint = FLinearColor(0.08f, 0.07f, 0.065f);
			SceneAssets->Shapes.Add(S);
		}
	}
	return true;
}

void FRubbleAbiBackend::Meteor(const FVector& Target)
{
	if (bMock || !WorldPtr || Meteors.Num() == 0)
	{
		return;
	}
	const RbBodyId Rock = Meteors[NextMeteor];
	NextMeteor = (NextMeteor + 1) % Meteors.Num();
	// Straight down from 50 m at 50 m/s: it lands on Target.
	const FVector Velocity(0.0, 0.0, -5000.0);
	const FVector From = Target - Velocity * 1.0;
	if (rbBody_SetTransform(WorldPtr, Rock, Xf(FTransform(From))) != RB_OK || rbBody_SetLinearVelocity(WorldPtr, Rock, Vec(Velocity)) != RB_OK
		|| rbBody_SetAngularVelocity(WorldPtr, Rock, RbVec3{ 0.5f, 1.0f, 0.2f }) != RB_OK || rbBody_SetAwake(WorldPtr, Rock, true) != RB_OK)
	{
		UE_LOG(LogRubble, Error, TEXT("%s"), *AbiError(TEXT("meteor")));
	}
}

FVector FRubbleAbiBackend::ParkedAt(int32 i)
{
	return FVector(i * 300.0, 0.0, 1.0e6);
}

void FRubbleAbiBackend::Launch(const FVector& Origin, const FVector& Direction)
{
	if (bMock)
	{
		Mock.Launch(Origin, Direction);
		return;
	}
	if (!WorldPtr)
	{
		return;
	}
	if (Balls.Num())
	{
		const RbBodyId Ball = Balls[NextBall];
		NextBall = (NextBall + 1) % Balls.Num();
		if (rbBody_SetTransform(WorldPtr, Ball, Xf(FTransform(Origin))) != RB_OK
			|| rbBody_SetLinearVelocity(WorldPtr, Ball, Vec(Direction.GetSafeNormal() * (IsTown() ? 4500.0 : 3500.0))) != RB_OK
			|| rbBody_SetAngularVelocity(WorldPtr, Ball, RbVec3{ 0, 0, 0 }) != RB_OK
			|| rbBody_SetAwake(WorldPtr, Ball, true) != RB_OK)
		{
			UE_LOG(LogRubble, Error, TEXT("%s"), *AbiError(TEXT("launch")));
		}
		return;
	}
	// A 25 cm steel ball at 35 m/s, as the pages' F.
	const float Radius = Current.Family == ERubbleFamily::Rigid ? 50.0f : 25.0f;
	RbBodyDef B = rbDefaultBodyDef();
	B.type_ = RB_DYNAMIC_BODY;
	B.transform = Xf(FTransform(Origin));
	B.linear_velocity = Vec(Direction.GetSafeNormal() * 3500.0);
	B.is_awake = true;
	RbBodyId Ball = 0;
	RbShapeDef SD = rbDefaultShapeDef();
	SD.density = 0.0078f;
	RbShapeId Shape = 0;
	if (rbCreateBody(WorldPtr, &B, &Ball) != RB_OK || rbCreateSphereShape(WorldPtr, Ball, &SD, RbVec3{ 0, 0, 0 }, Radius, &Shape) != RB_OK)
	{
		UE_LOG(LogRubble, Error, TEXT("%s"), *AbiError(TEXT("launch")));
		return;
	}
	FRubbleBodyVisual V;
	V.Id = FRubbleBodyId(Ball);
	V.Initial = FTransform(Origin);
	V.FirstShape = NewShapes.Num();
	V.NumShapes = 1;
	NewBodies.Add(V);
	FRubbleShapeVisual S;
	S.Kind = ERubbleShape::Sphere;
	S.Radius = Radius;
	S.Material = ERubbleMaterial::Steel;
	S.Tint = FLinearColor(0.55f, 0.56f, 0.58f);
	NewShapes.Add(S);
}

void FRubbleAbiBackend::SideRam()
{
	if (bMock)
	{
		Mock.SideRam();
	}
}

void FRubbleAbiBackend::Reset()
{
	FString Error;
	const FRubbleSceneRequest Again = Current;
	if (!Load(Again, Error))
	{
		UE_LOG(LogRubble, Error, TEXT("reset %s: %s"), *Again.ToString(), *Error);
	}
}
