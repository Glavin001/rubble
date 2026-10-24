// The data a Rubble world hands the renderer, in Unreal space (Z-up, centimetres, left-handed).
//
// Each type mirrors one in Rubble's public API (engine/docs/public-api.md, `rubble::World`), so the
// backend over the rb C ABI is a field-for-field copy:
//   FRubbleBodyId        BodyId (u64 {index, generation}, zero is null)
//   FRubbleMove          BodyMoveEvent (pose, previous pose, sleep and wake)
//   FRubbleEvent         destruction events (JointBroke, BeamBroke, PieceDetached, BondsBroken, Split)
//                        and P4 hit events (Impact)
//   FRubbleWheelVisual   per-wheel vehicle output (asked of the API: raycast wheels have no body)
//   FRubbleSoftSkin      the soft_skin kernel's output (positions, normals, strain)
//   FRubbleFrame         Frame: one step's results
//   FRubbleSceneAssets   what a scene's renderers need once: shapes, hull and chunk meshes, the
//                        destructible render contract (chunk world = body x rest), car topology.

#pragma once

#include "CoreMinimal.h"

/** A body handle: {index, generation} packed in a u64, null when zero (rubble BodyId). */
struct FRubbleBodyId
{
	uint64 Bits = 0;

	FRubbleBodyId() = default;
	explicit FRubbleBodyId(uint64 InBits) : Bits(InBits) {}
	static FRubbleBodyId Make(uint32 Index, uint32 Generation) { return FRubbleBodyId((uint64(Generation) << 32) | Index); }

	bool IsNull() const { return Bits == 0; }
	uint32 Index() const { return uint32(Bits); }
	uint32 Generation() const { return uint32(Bits >> 32); }
	bool operator==(const FRubbleBodyId& O) const { return Bits == O.Bits; }
	bool operator!=(const FRubbleBodyId& O) const { return Bits != O.Bits; }
	friend uint32 GetTypeHash(const FRubbleBodyId& Id) { return ::GetTypeHash(Id.Bits); }
};

/** The demo families of the web pages. */
enum class ERubbleFamily : uint8
{
	Rigid,     // demo.html
	Vehicles,  // vehicle.html
	Soft,      // softbody.html
	Fracture,  // fracture/
};

/** One editable scene parameter (the web pages' query-string fields: ?speed=56&overlap=0.25). */
struct FRubbleParamDesc
{
	FString Key;
	FString Label;
	float Default = 0.0f;
	float Min = 0.0f;
	float Max = 1.0f;
};

/** One entry of the demo catalogue. */
struct FRubbleDemoDesc
{
	ERubbleFamily Family = ERubbleFamily::Rigid;
	FString Id;     // the web's scene id ("wall", "ring110k", "tbone", "house")
	FString Title;  // the web's menu title
	TArray<FRubbleParamDesc> Params;
	/** Fixed physics step of the family: 60 Hz, or 240 Hz for soft bodies (softbody.html). */
	double StepHz = 60.0;
};

/** A scene to load: family:id?key=value&... (the command line's -RubbleDemo=soft:wall?speed=56). */
struct FRubbleSceneRequest
{
	ERubbleFamily Family = ERubbleFamily::Rigid;
	FString Id;
	TMap<FString, float> Params;

	float Param(const FString& Key, float Default) const
	{
		const float* V = Params.Find(Key);
		return V ? *V : Default;
	}
	RUBBLERUNTIME_API static bool Parse(const FString& Text, FRubbleSceneRequest& Out, FString& Error);
	RUBBLERUNTIME_API FString ToString() const;
};

RUBBLERUNTIME_API const TCHAR* LexToString(ERubbleFamily Family);
RUBBLERUNTIME_API bool LexFromString(ERubbleFamily& Out, const FString& Text);

// ----- scene assets (served once per scene) ---------------------------------------------------

/** Surface materials the renderers know. Exterior faces use the first, fracture faces Interior. */
enum class ERubbleMaterial : uint8
{
	Concrete,
	Brick,
	Timber,
	Steel,
	Glass,
	Rubber,
	CarPaint,
	Plastic,
	Interior, // a fracture face: the inside of a broken chunk
	Count
};

/** A triangle mesh in its body's (or chunk's) local frame. Winding is Unreal's (clockwise front). */
struct FRubbleMeshData
{
	TArray<FVector3f> Positions;
	TArray<FVector3f> Normals;
	TArray<FVector2f> UVs;
	TArray<uint32> Indices;
	/** Per-triangle material; empty means Material for every triangle. */
	TArray<ERubbleMaterial> TriangleMaterials;
	ERubbleMaterial Material = ERubbleMaterial::Concrete;

	int32 NumTriangles() const { return Indices.Num() / 3; }
};

enum class ERubbleShape : uint8
{
	Box,     // HalfExtents
	Sphere,  // Radius
	Capsule, // Radius, HalfHeight (along local Z)
	Hull,    // Mesh
};

/** One collision shape as drawn (rubble shape_geometry). */
struct FRubbleShapeVisual
{
	ERubbleShape Kind = ERubbleShape::Box;
	FVector3f HalfExtents = FVector3f(50.0f);
	float Radius = 50.0f;
	float HalfHeight = 0.0f;
	int32 Mesh = INDEX_NONE; // into FRubbleSceneAssets::Meshes, for Hull
	FTransform Local = FTransform::Identity;
	ERubbleMaterial Material = ERubbleMaterial::Concrete;
	FLinearColor Tint = FLinearColor::White;
};

/** A rigid body drawn by its shapes (rigid scenes, scenery, projectiles, vehicle parts). */
struct FRubbleBodyVisual
{
	FRubbleBodyId Id;
	FTransform Initial = FTransform::Identity;
	int32 FirstShape = 0;
	int32 NumShapes = 0;
	bool bStatic = false;
	/** Starts awake (dropped grids) or asleep until something hits it (the pages' settled stacks). */
	bool bAwake = false;
	/** A moving obstacle (a ram, a shunting truck, a crusher block, an impactor ball): its start
	 *  velocity (cm/s), spin (rad/s) and mass (kg; 0: the material's density). */
	FVector InitialVelocity = FVector::ZeroVector;
	FVector InitialAngularVelocity = FVector::ZeroVector;
	float Mass = 0.0f;
};

/** A destructible chunk: chunk world = its body's transform x Rest (the D1 render contract). */
struct FRubbleChunkVisual
{
	int32 Mesh = INDEX_NONE;
	int32 Destructible = 0;
	FTransform Rest = FTransform::Identity;
	/** The chunk's own body from the start (the Bonds model: every chunk is a body); null when the
	 *  destructible's root body carries it until a Split. */
	FRubbleBodyId Body;
};

/** A placed destructible: one body (its root proxy) until it splits. */
struct FRubbleDestructibleVisual
{
	FRubbleBodyId RootBody;
	FTransform Placement = FTransform::Identity;
	TArray<int32> Chunks;
};

struct FRubbleWheelDef
{
	FVector3f LocalHub = FVector3f::ZeroVector; // rest position in the chassis frame
	float Radius = 35.0f;
	float Width = 22.0f;
};

enum class ERubbleVehicleStyle : uint8
{
	Sedan,
	Sports,
	MonsterTruck,
	SixWheelTruck,
	Kart,
	Tank,
	Motorcycle,
	DuneBuggy,
};

struct FRubbleVehicleVisual
{
	int32 Vehicle = 0;
	/** Null when the backend cannot name the chassis body: it is then placed from the wheels. */
	FRubbleBodyId Chassis;
	ERubbleVehicleStyle Style = ERubbleVehicleStyle::Sedan;
	FVector3f ChassisHalfExtents = FVector3f(220.0f, 90.0f, 45.0f);
	FLinearColor Paint = FLinearColor(0.6f, 0.05f, 0.04f);
	TArray<FRubbleWheelDef> Wheels;
	bool bPlayer = false;
	/** The vehicle's own bodywork in its chassis frame (from its descriptor); empty: a placeholder
	 *  by Style. */
	TArray<FRubbleShapeVisual> Parts;
	/** The chassis frame's +X points backwards (a descriptor driving along Rubble +Z). */
	bool bForwardIsMinusX = false;
};

/** A soft-body vehicle: its render mesh's topology; each frame's FRubbleSoftSkin moves its vertices. */
struct FRubbleSoftVisual
{
	int32 Soft = 0;
	/** Where it starts and how fast (crash tests start at speed). */
	FTransform Spawn = FTransform::Identity;
	FVector Velocity = FVector::ZeroVector;
	/** A car (wheels, the procedural car), not a crate. */
	bool bCar = true;
	bool bAI = false;
	int32 Mesh = INDEX_NONE; // topology, UVs and materials; positions are the rest pose
	FLinearColor Paint = FLinearColor(0.05f, 0.18f, 0.55f);
	TArray<FRubbleWheelDef> Wheels;
	bool bPlayer = false;
};

/** Which level look a scene wants. */
enum class ERubbleStage : uint8
{
	Yard,      // tarmac yard: rigid scenes and vehicles
	CrashHall, // crash-test hall: soft-body cars
	Lot,       // city lot: fracture
};

struct FRubbleSceneAssets
{
	ERubbleStage Stage = ERubbleStage::Yard;
	float GroundHalfExtentCm = 10000.0f;
	FVector CameraEye = FVector(-3000.0, -1500.0, 1500.0);
	FVector CameraTarget = FVector::ZeroVector;
	/** The page's solver settings for the scene: iterations (0: the world default), friction
	 *  (0: the client default) and whether resting bodies sleep. */
	int32 SolverIterations = 0;
	float Friction = 0.0f;
	bool bSleep = true;
	/** The page's reuseContacts (flags: 1): keep the contacts of pairs that have not moved. */
	bool bReuseContacts = false;

	TArray<FRubbleMeshData> Meshes;
	TArray<FRubbleShapeVisual> Shapes;
	TArray<FRubbleBodyVisual> Bodies;
	TArray<FRubbleChunkVisual> Chunks;
	TArray<FRubbleDestructibleVisual> Destructibles;
	TArray<FRubbleVehicleVisual> Vehicles;
	TArray<FRubbleSoftVisual> SoftBodies;
};

// ----- per-step output -----------------------------------------------------------------------

/** A body whose pose or sleep state changed in a step (rubble BodyMoveEvent). */
struct FRubbleMove
{
	FRubbleBodyId Body;
	FTransform Transform = FTransform::Identity;
	/** The pose reported before this step; interpolate Previous -> Transform between fixed steps. */
	FTransform Previous = FTransform::Identity;
	bool bHasPrevious = false;
	bool bFellAsleep = false;
	bool bWoke = false;
	uint64 UserData = 0;
};

/** A destructible actor described by value: the parent of a split may no longer exist when read. */
struct FRubbleActorInfo
{
	FRubbleBodyId Body;
	TArray<int32> Chunks;
	float Mass = 0.0f;
	FVector Com = FVector::ZeroVector;
	FVector LinearVelocity = FVector::ZeroVector;
	bool bAnchored = false;
};

enum class ERubbleEventType : uint8
{
	JointBroke,
	BeamBroke,
	PieceDetached, // a soft-body part (door, bumper) no longer joined to the car
	BondsBroken,   // Count bonds of Destructible broke this step
	Split,         // Parent became Children
	Impact,        // a contact hit above the hit-event threshold (P4)
};

struct FRubbleEvent
{
	ERubbleEventType Type = ERubbleEventType::Impact;
	FVector Position = FVector::ZeroVector;
	FVector Normal = FVector::UpVector;
	/** Force (N) for breaks, approach speed (cm/s) for impacts. */
	float Magnitude = 0.0f;
	int32 Count = 0;
	int32 Destructible = INDEX_NONE;
	int32 Soft = INDEX_NONE;
	FName Part;
	ERubbleMaterial MaterialA = ERubbleMaterial::Concrete;
	ERubbleMaterial MaterialB = ERubbleMaterial::Concrete;
	FRubbleActorInfo Parent;
	TArray<FRubbleActorInfo> Children;
};

/** A wheel as drawn: world hub pose, spin and steer, suspension compression (0 extended .. 1). */
struct FRubbleWheelVisual
{
	int32 Vehicle = 0;
	bool bSoft = false; // Vehicle indexes FRubbleSceneAssets::SoftBodies rather than Vehicles
	int32 Wheel = 0;
	FTransform Hub = FTransform::Identity;
	float SpinRad = 0.0f;
	float SteerRad = 0.0f;
	float Compression = 0.0f;
	bool bOnGround = true;
	/** The wheel's radius when the backend knows it (0: use the scene's wheel definition). */
	float Radius = 0.0f;
};

/** One soft body's skinned render mesh this step (soft_skin.slang's output layout). */
struct FRubbleSoftSkin
{
	int32 Soft = 0;
	TArray<FVector3f> Positions;
	TArray<FVector3f> Normals;
	/** Per vertex: the largest strain around it (0 rest .. 1 crushed); drives the damage mask. */
	TArray<float> Strain;
	/** Per triangle: torn (no longer drawn). */
	TBitArray<> TornTriangles;
};

/** One step's results (rubble Frame). */
struct FRubbleFrame
{
	uint64 Step = 0;
	bool bRan = true;
	TArray<FRubbleMove> Moves;
	TArray<FRubbleEvent> Events;
	TArray<FRubbleWheelVisual> Wheels;
	TArray<FRubbleSoftSkin> Skins;
	/** Events overwritten in the GPU ring before they were read: never silently zero. */
	uint32 LostEvents = 0;
	/** The readback was dropped under backpressure (the API's Backpressure::Drop). */
	bool bDropped = false;
	/** Bodies created since the last frame (projectiles), drawn by their shapes; FirstShape
	 *  indexes NewShapes. */
	TArray<FRubbleBodyVisual> NewBodies;
	TArray<FRubbleShapeVisual> NewShapes;
};

// ----- input --------------------------------------------------------------------------------

/** Driver input, as rubble VehicleInput (forward, steer, brake, handbrake, gear). */
struct FRubbleVehicleInput
{
	float Throttle = 0.0f; // -1 .. 1
	float Steer = 0.0f;    // -1 (left) .. 1 (right)
	float Brake = 0.0f;
	bool bHandbrake = false;
	int32 Gear = 1;
};
