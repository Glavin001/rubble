// The host side of a fixed-step world: real time in, whole steps out, and the frames of one display
// frame merged into one. Pure logic (no UObjects), covered by Rubble.Clock tests.

#pragma once

#include "CoreMinimal.h"
#include "RubbleTypes.h"

/**
 * Turns real time into fixed steps, as vibe-land's poll_local does: time owed accumulates, whole
 * steps are taken, at most MaxStepsPerFrame a frame. Time owed past the cap is dropped, so a slow
 * frame slows the simulation down; it never coarsens the step.
 */
struct RUBBLERUNTIME_API FRubbleClock
{
	double StepSeconds = 1.0 / 60.0;
	int32 MaxStepsPerFrame = 4;
	/** Owed time below one step, carried to the next frame. */
	double Owed = 0.0;
	/** Seconds dropped by the cap since the scene started (reported, never hidden). */
	double Dropped = 0.0;

	void Reset(double InStepSeconds, int32 InMaxStepsPerFrame);
	/** Add a frame's real time; returns the number of steps to take now. */
	int32 Advance(double RealSeconds);
	/** How far the display is between the last step and the next, 0 .. 1. */
	double Alpha() const { return StepSeconds > 0.0 ? FMath::Clamp(Owed / StepSeconds, 0.0, 1.0) : 0.0; }
};

/**
 * Merges the frames that arrive in one display frame (soft scenes step 4 times a frame) into one:
 *  - a body's move keeps the latest Transform and the latest step's Previous, so interpolation runs
 *    across the last step (the one Alpha measures); a body that moved in an earlier step but not
 *    the latest is reported at rest (Previous = Transform);
 *  - sleep: bFellAsleep is the latest state, bWoke is set if it woke in any step;
 *  - wheels and skins: the latest per vehicle and soft body;
 *  - events: all, in step order; lost events summed; dropped if any was.
 */
struct RUBBLERUNTIME_API FRubbleFrameMerger
{
	static void Merge(TArray<FRubbleFrame>&& Frames, FRubbleFrame& Out);
};

/** Interpolated pose between two step poses (alpha 0 = Previous, 1 = Current). */
RUBBLERUNTIME_API FTransform RubbleLerpPose(const FTransform& Previous, const FTransform& Current, double Alpha);
