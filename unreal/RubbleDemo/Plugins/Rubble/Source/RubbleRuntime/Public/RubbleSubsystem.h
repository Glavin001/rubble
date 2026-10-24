// Owns the Rubble world for the game: the backend (mock today, the rb C ABI later), the fixed-step
// clock, and the merged frame of each display frame. ARubbleDirector advances it once a frame in
// TG_PrePhysics, after the pawn has set this frame's input; the renderers draw from it.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "RubbleBackend.h"
#include "RubbleClock.h"
#include "RubbleTypes.h"
#include "RubbleSubsystem.generated.h"

RUBBLERUNTIME_API DECLARE_LOG_CATEGORY_EXTERN(LogRubble, Log, All);

DECLARE_MULTICAST_DELEGATE_OneParam(FOnRubbleFrame, const FRubbleFrame& /*Merged*/);
DECLARE_MULTICAST_DELEGATE_OneParam(FOnRubbleSceneLoaded, const FRubbleSceneAssets& /*Assets*/);

/** What the last display frame did, for the HUD and the perf baseline. */
struct FRubbleFrameStats
{
	int32 Steps = 0;
	int32 Moves = 0;
	int32 Events = 0;
	int32 Skins = 0;
	double DroppedSeconds = 0.0;
	double BackendMs = 0.0;
	double BroadcastMs = 0.0;
	uint64 Step = 0;
};

UCLASS()
class RUBBLERUNTIME_API URubbleSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/** Replace the backend (tests use their own; -RubbleBackend= is reserved for the ABI backend). */
	void SetBackend(TUniquePtr<IRubbleBackend> InBackend);
	IRubbleBackend* Backend() const { return BackendImpl.Get(); }

	TArray<FRubbleDemoDesc> ListDemos() const;
	bool LoadScene(const FRubbleSceneRequest& Request, FString& Error);
	bool HasScene() const { return bHasScene; }
	const FRubbleSceneRequest& CurrentScene() const { return Current; }
	const FRubbleDemoDesc* CurrentDemo() const;
	const FRubbleSceneAssets* Assets() const { return bHasScene ? &BackendImpl->Assets() : nullptr; }

	/** One display frame: real time in, steps taken, finished frames merged and broadcast. */
	void Advance(double RealSeconds);

	/** Where the display is between the last step and the next (renderers interpolate by it). */
	double Alpha() const { return Clock.Alpha(); }
	const FRubbleFrameStats& Stats() const { return LastStats; }

	// Input and actions, forwarded to the backend.
	void SetVehicleInput(const FRubbleVehicleInput& Input);
	void Launch(const FVector& Origin, const FVector& Direction);
	void SideRam();
	void Reset();

	/** Paused worlds take no steps. TimeScale < 1 is slow motion (fewer steps, same step size). */
	bool bPaused = false;
	double TimeScale = 1.0;

	FOnRubbleFrame OnFrame;
	FOnRubbleSceneLoaded OnSceneLoaded;

private:
	TUniquePtr<IRubbleBackend> BackendImpl;
	FRubbleClock Clock;
	FRubbleSceneRequest Current;
	TArray<FRubbleDemoDesc> Demos;
	FRubbleFrameStats LastStats;
	bool bHasScene = false;
};
