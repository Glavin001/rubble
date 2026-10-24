// The seam between the renderers and a Rubble world. The mock backend (RubbleMockBackend.h)
// implements it today; a backend over the rb C ABI replaces it when the public API lands, and
// nothing above this interface changes.
//
// The contract follows rubble::World: Step() advances one fixed step and never waits; Poll()
// returns the steps whose results are ready, oldest first, at least one step late (the GPU readback
// ring). Everything is in Unreal space.

#pragma once

#include "CoreMinimal.h"
#include "RubbleTypes.h"

class FRubbleGpuPoses;

class RUBBLERUNTIME_API IRubbleBackend
{
public:
	virtual ~IRubbleBackend() = default;

	/** A short name for logs ("mock", "rb"). */
	virtual const TCHAR* Name() const = 0;

	/** Every scene this backend can load: the web pages' catalogue. */
	virtual TArray<FRubbleDemoDesc> ListDemos() const = 0;

	/** Load a scene, replacing the current one. False and Error on failure (nothing is half-loaded). */
	virtual bool Load(const FRubbleSceneRequest& Request, FString& Error) = 0;

	/** The loaded scene's assets; valid until the next Load. */
	virtual const FRubbleSceneAssets& Assets() const = 0;

	/** The loaded scene's fixed step, seconds. */
	virtual double StepSeconds() const = 0;

	/** Advance one fixed step. Does not wait for results. */
	virtual void Step() = 0;

	/** Move the finished steps' frames into Out, oldest first. */
	virtual void Poll(TArray<FRubbleFrame>& Out) = 0;

	// Input and actions (each applies at the start of the next step, as rubble's queued commands).
	virtual void SetVehicleInput(const FRubbleVehicleInput& Input) = 0;
	/** Fire a projectile from Origin along Direction (the pages' F / B and the fracture lab's click). */
	virtual void Launch(const FVector& Origin, const FVector& Direction) = 0;
	/** The soft-body page's T: the parked side ram. */
	virtual void SideRam() = 0;
	/** Reload the current scene from its initial state. */
	virtual void Reset() = 0;

	/** The poses on Unreal's GPU for the GPU Scene renderers (unreal/DESIGN.md), or null when
	 *  this backend or platform has none: the readback renderers then draw everything. */
	virtual FRubbleGpuPoses* GpuPoses() const { return nullptr; }
};
