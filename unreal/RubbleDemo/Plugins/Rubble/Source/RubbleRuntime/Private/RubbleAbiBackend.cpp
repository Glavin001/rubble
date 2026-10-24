#include "RubbleAbiBackend.h"

#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RubbleGlb.h"
#include "RubbleSubsystem.h"

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
}

FString FRubbleAbiBackend::MockReason(const FRubbleSceneRequest& R)
{
	switch (R.Family)
	{
	case ERubbleFamily::Soft:
		return TEXT("soft-body cars need the procedural car on the ABI (it takes .truck text only)");
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
		return R.Id == TEXT("house") || R.Id == TEXT("arch") ? FString() : TEXT("procedural destructibles (cuboids, rows, grids) are not on the ABI");
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
			SD.friction = 0.6f;
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
	W.time_step = float(Dt);
	if (Request.Family == ERubbleFamily::Fracture)
	{
		NumBodies += 4096;
		W.max_joints = FMath::Max(W.max_joints, 32768u);
		W.max_hull_vertices = FMath::Max(W.max_hull_vertices, 1u << 20);
	}
	W.max_bodies = FMath::Max(W.max_bodies, uint32(NumBodies * 1.25));
	W.max_colliders = FMath::Max(W.max_colliders, uint32(NumBodies * 2));
	// Bricks touch all their neighbours: demo.html budgets ~29 contacts a body (ring: 262,144 for
	// 9k); 32 here. Overflow is loud (rbWorld_Poll fails with the flag), never a silent drop.
	W.max_contacts = FMath::Max(W.max_contacts, uint32(NumBodies * 32));
	W.max_candidates = FMath::Max(W.max_candidates, uint32(NumBodies * 32));
	if (rbCreateWorld(&W, &WorldPtr) != RB_OK)
	{
		Error = AbiError(TEXT("rbCreateWorld"));
		return false;
	}

	// Ground: one static box with its top at Z = 0, as wide as the stage and 4 m thick. (One:
	// statics wider than half a grid cell are listed separately, at most 1024 of them, so tiling
	// the ground overflows; OVERFLOW_LARGE_STATICS.)
	if (Layout.GroundHalfExtentCm > 0.0f)
	{
		RbBodyDef G = rbDefaultBodyDef();
		G.transform = Xf(FTransform(FVector(0, 0, -200)));
		RbBodyId Ground = 0;
		RbShapeDef SD = rbDefaultShapeDef();
		SD.friction = 0.8f;
		RbShapeId Shape = 0;
		const float H = FMath::Max(2000.0f, Layout.GroundHalfExtentCm);
		if (rbCreateBody(WorldPtr, &G, &Ground) != RB_OK || rbCreateBoxShape(WorldPtr, Ground, &SD, RbVec3{ H, H, 200.0f }, Xf(FTransform::Identity), &Shape) != RB_OK)
		{
			Error = AbiError(TEXT("ground"));
			return false;
		}
	}
	if (!CreateBodies(Layout, Error))
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
		if (!CreateDestructible(Request.Id == TEXT("house") ? TEXT("veneer-house.glb") : TEXT("masonry-arch.glb"), FTransform::Identity, Error))
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
	if (WorldPtr && PlayerVehicle != MAX_uint32)
	{
		rbVehicle_SetInput(WorldPtr, PlayerVehicle, RbVehicleInput{ In.Throttle, In.Steer, In.Brake, In.bHandbrake ? 1.0f : 0.0f });
	}
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
