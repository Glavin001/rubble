// The mock backend: every scene of the web pages as scripted stand-in motion (Private/Mock), behind
// the same contract as the real API: fixed steps, results one step late.

#pragma once

#include "CoreMinimal.h"
#include "RubbleBackend.h"

namespace RubbleMock { class FScene; }

class RUBBLERUNTIME_API FRubbleMockBackend : public IRubbleBackend
{
public:
	FRubbleMockBackend();
	virtual ~FRubbleMockBackend() override;

	virtual const TCHAR* Name() const override { return TEXT("mock"); }
	virtual TArray<FRubbleDemoDesc> ListDemos() const override;
	virtual bool Load(const FRubbleSceneRequest& Request, FString& Error) override;
	virtual const FRubbleSceneAssets& Assets() const override { return *SceneAssets; }
	virtual double StepSeconds() const override { return Dt; }
	virtual void Step() override;
	virtual void Poll(TArray<FRubbleFrame>& Out) override;
	virtual void SetVehicleInput(const FRubbleVehicleInput& Input) override;
	virtual void Launch(const FVector& Origin, const FVector& Direction) override;
	virtual void SideRam() override;
	virtual void Reset() override;

	/** Results are held back this many steps, as the GPU readback ring does (default 1). */
	int32 LatencySteps = 1;

private:
	TUniquePtr<RubbleMock::FScene> Scene;
	/** On the heap: scenes keep a pointer to the assets they built, so they never move. */
	TUniquePtr<FRubbleSceneAssets> SceneAssets;
	FRubbleSceneRequest Current;
	double Dt = 1.0 / 60.0;
	uint64 StepCount = 0;
	TArray<FRubbleFrame> InFlight;
};
