// The real backend: Rubble's C ABI (engine/crates/engine-ffi/include/rubble.h), linked as
// librubble_ffi.a, in Unreal coordinates (RB_COORDINATES_UNREAL: the ABI converts exactly).
//
// Scenes are built through the public API (bodies, shapes, vehicle presets, destructibles from
// Shardwright GLBs) from the same layouts the mock draws. Scenes the ABI cannot build yet fall back
// to the mock, and Name() says so ("mock: <reason>"):
//  - soft-body cars: the ABI takes Rigs of Rods .truck text only, and the demo's procedural car
//    lives in engine_softbody (no rbCreate for it);
//  - the scripted rigid scenes (joints, ragdolls, chains, motors) and the fracture lab's
//    procedural destructibles (cuboids, rows, grids): their builders are not on the ABI.

#pragma once

#include "CoreMinimal.h"
#include "RubbleBackend.h"
#include "RubbleGpuPoses.h"
#include "RubbleMockBackend.h"

struct RbWorld;

class RUBBLERUNTIME_API FRubbleAbiBackend : public IRubbleBackend
{
public:
	FRubbleAbiBackend();
	virtual ~FRubbleAbiBackend() override;

	/** False (and why) when the linked library's ABI version is not the header's. */
	static bool CheckVersion(FString& Error);

	virtual const TCHAR* Name() const override { return *NameText; }
	virtual TArray<FRubbleDemoDesc> ListDemos() const override { return Mock.ListDemos(); }
	virtual bool Load(const FRubbleSceneRequest& Request, FString& Error) override;
	virtual const FRubbleSceneAssets& Assets() const override { return bMock ? Mock.Assets() : *SceneAssets; }
	virtual double StepSeconds() const override { return bMock ? Mock.StepSeconds() : Dt; }
	virtual void Step() override;
	virtual void Poll(TArray<FRubbleFrame>& Out) override;
	virtual void SetVehicleInput(const FRubbleVehicleInput& Input) override;
	virtual void Launch(const FVector& Origin, const FVector& Direction) override;
	virtual void SideRam() override;
	virtual void Meteor(const FVector& Target) override;
	virtual void Reset() override;

	/** Why a scene runs on the mock, or empty when the ABI builds it. */
	static FString MockReason(const FRubbleSceneRequest& Request);
	bool IsOnAbi() const { return !bMock; }
	virtual FRubbleGpuPoses* GpuPoses() const override { return bMock ? nullptr : Gpu.Get(); }
	RbWorld* World() const { return WorldPtr; }

private:
	bool LoadAbi(const FRubbleSceneRequest& Request, FString& Error);
	void DestroyWorld();
	/** The bodies of a mock layout, created for real; returns false on an ABI error. */
	bool CreateBodies(const FRubbleSceneAssets& Layout, FString& Error);
	bool CreateDestructible(const TCHAR* Glb, const FTransform& Placement, FString& Error);
	/** The town (a ScenePack v2, scripts/unreal/build-town.sh: rubble's copy of vibe-land's Vibe
	 *  Town with a glass tower): one destructible of ~52k chunks, each drawn from the pack's own
	 *  collider and material. */
	bool CreateTown(FString& Error);
	/** Balls for the shots, parked asleep; Launch reuses them through the body commands. */
	bool ParkBalls(int32 Count, FString& Error);
	static FVector ParkedAt(int32 Index);
	TArray<uint64> Balls;
	int32 NextBall = 0;
	/** The town's meteors, parked as the balls are. */
	TArray<uint64> Meteors;
	int32 NextMeteor = 0;
	/** A cannonball's speed (cm/s): the town's 45 m/s under the threshold model, the glass
	 *  tower's on bond joints the fracture lab's for it (18 m/s, its default for a scene that
	 *  size), the rest 35 m/s; ?shotspeed= (m/s). */
	double ShotSpeed() const
	{
		const double Default = IsTown() && bTownThreshold ? 45.0 : Current.Id == TEXT("glasstower") ? 18.0 : 35.0;
		return 100.0 * Current.Param(TEXT("shotspeed"), float(Default));
	}
	/** The pack scenes on the town's stage and kit: the town, and the glass tower and pavilion. */
	bool IsTown() const { return Current.Family == ERubbleFamily::Fracture && (Current.Id == TEXT("town") || Current.Id == TEXT("glasstower")); }
	/** The town and the glass tower under the threshold strength model (the default; ?bonds=1
	 *  keeps the breakable bond joints, every chunk a body). */
	bool bTownThreshold = false;
	/** The threshold town, by chunk: its body, its centre in that body's frame, its look. Splits
	 *  move chunks to new bodies. */
	TArray<uint64> TownChunkBody;
	TArray<FVector> TownChunkLocal;
	TArray<TPair<ERubbleMaterial, FLinearColor>> TownChunkLook;
	/** The threshold town's bodies' last poses (every piece's origin is the town's frame). */
	TMap<uint64, FTransform> TownBodyPose;
	FVector TownChunkAt(uint32 Chunk) const;

	/** Soft cars: the procedural car spawned per layout car, skinned with a fitted shell. */
	bool CreateSoftCars(const FRubbleSceneAssets& Layout, FString& Error);
	struct FSoftCar
	{
		uint32 Soft = 0;
		uint32 Skin = 0;
		int32 Visual = 0;
		bool bAI = false;
		TArray<TPair<int32, int32>> WheelNodes; // axle node pairs
		TArray<float> WheelRadius;
		TArray<float> Spin;
		TArray<FVector> LastHub;
		TArray<TArray<int32>> Neighbours; // per skin vertex
		TArray<TArray<float>> RestLength;
		TArray<uint32> Triangles;
	};
	TArray<FSoftCar> SoftCars;
	TMap<uint32, int32> SoftCarOfSkin;
	TMap<uint32, int32> SoftCarOfSoft;
	int32 PlayerSoft = INDEX_NONE;

	FRubbleMockBackend Mock;
	bool bMock = true;
	FString NameText = TEXT("rubble");
	FRubbleSceneRequest Current;
	RbWorld* WorldPtr = nullptr;
	/** The world's poses on Unreal's GPU: set when the world runs on Unreal's device. */
	TUniquePtr<FRubbleGpuPoses> Gpu;
	TUniquePtr<FRubbleSceneAssets> SceneAssets;
	double Dt = 1.0 / 60.0;
	uint32 PlayerVehicle = MAX_uint32;
	TArray<uint32> AiVehicles;
	TMap<uint32, int32> VehicleVisual; // RbVehicleId -> FRubbleSceneAssets::Vehicles index
	/** Bodies made since the last poll (balls), handed to the renderers with the next frame. */
	TArray<FRubbleBodyVisual> NewBodies;
	TArray<FRubbleShapeVisual> NewShapes;
	/** Last known position of chunk bodies, for where breaks happened (events carry no point). */
	TMap<uint64, FVector> ChunkAt;
	/** What a chunk body is made of, for the debris and dust of its breaks. */
	TMap<uint64, TPair<ERubbleMaterial, FLinearColor>> ChunkLook;
	FRubbleVehicleInput Input;
	uint64 Steps = 0;
	int64 AutoShotStep = -1;
};
