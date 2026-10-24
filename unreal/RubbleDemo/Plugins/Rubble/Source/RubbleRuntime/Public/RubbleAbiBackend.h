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
	virtual void Reset() override;

	/** Why a scene runs on the mock, or empty when the ABI builds it. */
	static FString MockReason(const FRubbleSceneRequest& Request);
	bool IsOnAbi() const { return !bMock; }
	RbWorld* World() const { return WorldPtr; }

private:
	bool LoadAbi(const FRubbleSceneRequest& Request, FString& Error);
	void DestroyWorld();
	/** The bodies of a mock layout, created for real; returns false on an ABI error. */
	bool CreateBodies(const FRubbleSceneAssets& Layout, FString& Error);
	bool CreateDestructible(const TCHAR* Glb, const FTransform& Placement, FString& Error);
	/** Balls for the shots, parked asleep; Launch reuses them through the body commands. */
	bool ParkBalls(int32 Count, FString& Error);
	static FVector ParkedAt(int32 Index);
	TArray<uint64> Balls;
	int32 NextBall = 0;

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
	FRubbleVehicleInput Input;
	uint64 Steps = 0;
	int64 AutoShotStep = -1;
};
