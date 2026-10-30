// Records a backend's frames and plays them back, so one simulation can be filmed from several
// cameras: the physics (GPU-stepped) is not bit-reproducible run to run, so each camera of a film
// sees the same collapse only if it sees the same frames. Recording passes everything through to
// the wrapped backend; replaying serves the recorded frames, a Poll at a time, and takes no steps.
#pragma once

#include "CoreMinimal.h"
#include "RubbleBackend.h"

class RUBBLERUNTIME_API FRubbleReplayBackend : public IRubbleBackend
{
public:
	explicit FRubbleReplayBackend(TUniquePtr<IRubbleBackend> InInner) : Inner(MoveTemp(InInner)) {}

	/** Record from here: drop any recording and capture every Poll's frames. */
	void BeginRecording();
	/** Replay from the recording's start. Load then keeps the recorded scene's assets (the frames
	 *  name its bodies) and only lets the renderers rebuild. */
	void BeginReplay();
	/** While held, a replay serves nothing and stays where it is (a film's unrecorded lead-in). */
	void SetHold(bool bInHold) { bHold = bInHold; }
	bool IsReplaying() const { return bReplaying; }
	int32 RecordedPolls() const { return Recording.Num(); }
	IRubbleBackend* InnerBackend() const { return Inner.Get(); }

	virtual const TCHAR* Name() const override { return TEXT("replay"); }
	virtual TArray<FRubbleDemoDesc> ListDemos() const override { return Inner->ListDemos(); }
	virtual bool Load(const FRubbleSceneRequest& Request, FString& Error) override;
	virtual const FRubbleSceneAssets& Assets() const override { return Inner->Assets(); }
	virtual double StepSeconds() const override { return Inner->StepSeconds(); }
	virtual void Step() override;
	virtual void Poll(TArray<FRubbleFrame>& Out) override;
	virtual void SetVehicleInput(const FRubbleVehicleInput& Input) override;
	virtual void Launch(const FVector& Origin, const FVector& Direction) override;
	virtual void SideRam() override;
	virtual void Meteor(const FVector& Target) override;
	virtual void Reset() override;
	/** None: the renderers draw from the frames' moves, recorded and replayed alike. */
	virtual FRubbleGpuPoses* GpuPoses() const override { return nullptr; }

private:
	TUniquePtr<IRubbleBackend> Inner;
	TArray<TArray<FRubbleFrame>> Recording;
	bool bRecording = false;
	bool bReplaying = false;
	bool bHold = false;
	int32 Next = 0;
};
