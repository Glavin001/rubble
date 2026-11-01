#include "RubbleClock.h"

void FRubbleClock::Reset(double InStepSeconds, int32 InMaxStepsPerFrame)
{
	StepSeconds = InStepSeconds;
	MaxStepsPerFrame = FMath::Max(1, InMaxStepsPerFrame);
	Owed = 0.0;
	Dropped = 0.0;
}

int32 FRubbleClock::Advance(double RealSeconds)
{
	if (StepSeconds <= 0.0)
	{
		return 0;
	}
	Owed += FMath::Max(0.0, RealSeconds);
	int32 Steps = FMath::FloorToInt32(Owed / StepSeconds);
	if (Steps > MaxStepsPerFrame)
	{
		// Past the cap the time is dropped, not owed: the next frame starts at rest.
		Dropped += Owed - MaxStepsPerFrame * StepSeconds;
		Steps = MaxStepsPerFrame;
		Owed = 0.0;
		return Steps;
	}
	Owed -= Steps * StepSeconds;
	return Steps;
}

void FRubbleFrameMerger::Merge(TArray<FRubbleFrame>&& Frames, FRubbleFrame& Out)
{
	Out = FRubbleFrame();
	if (Frames.Num() == 0)
	{
		Out.bRan = false;
		return;
	}
	if (Frames.Num() == 1)
	{
		Out = MoveTemp(Frames[0]);
		return;
	}
	Out.Step = Frames.Last().Step;
	Out.bRan = false;

	// Index of each body's move in Out.Moves, and the step of the frame it last moved in.
	TMap<FRubbleBodyId, int32> MoveOf;
	TMap<int32, int32> SkinOf;
	TMap<uint64, int32> WheelOf;
	const int32 Last = Frames.Num() - 1;
	for (int32 f = 0; f < Frames.Num(); ++f)
	{
		FRubbleFrame& F = Frames[f];
		Out.bRan |= F.bRan;
		Out.bDropped |= F.bDropped;
		Out.LostEvents += F.LostEvents;
		for (FRubbleMove& M : F.Moves)
		{
			if (int32* At = MoveOf.Find(M.Body))
			{
				FRubbleMove& Kept = Out.Moves[*At];
				const bool bWoke = Kept.bWoke || M.bWoke;
				Kept = M;
				Kept.bWoke = bWoke;
			}
			else
			{
				MoveOf.Add(M.Body, Out.Moves.Add(M));
			}
		}
		Out.Events.Append(MoveTemp(F.Events));
		// Spawned bodies: their shape offsets move past the shapes already merged.
		const int32 ShapeBase = Out.NewShapes.Num();
		for (FRubbleBodyVisual B : F.NewBodies)
		{
			B.FirstShape += ShapeBase;
			Out.NewBodies.Add(B);
		}
		Out.NewShapes.Append(F.NewShapes);
		for (FRubbleWheelVisual& W : F.Wheels)
		{
			const uint64 Key = (uint64(W.bSoft) << 63) | (uint64(uint32(W.Vehicle)) << 32) | uint32(W.Wheel);
			if (int32* At = WheelOf.Find(Key))
			{
				Out.Wheels[*At] = MoveTemp(W);
			}
			else
			{
				WheelOf.Add(Key, Out.Wheels.Add(MoveTemp(W)));
			}
		}
		for (FRubbleSoftSkin& S : F.Skins)
		{
			if (int32* At = SkinOf.Find(S.Soft))
			{
				Out.Skins[*At] = MoveTemp(S);
			}
			else
			{
				SkinOf.Add(S.Soft, Out.Skins.Add(MoveTemp(S)));
			}
		}
	}
	// A body that did not move in the latest step is at rest there: nothing left to interpolate.
	TSet<FRubbleBodyId> MovedLast;
	for (const FRubbleMove& M : Frames[Last].Moves)
	{
		MovedLast.Add(M.Body);
	}
	for (FRubbleMove& M : Out.Moves)
	{
		if (!MovedLast.Contains(M.Body))
		{
			M.Previous = M.Transform;
			M.bHasPrevious = true;
		}
	}
}

FTransform RubbleLerpPose(const FTransform& Previous, const FTransform& Current, double Alpha)
{
	if (Alpha >= 1.0)
	{
		return Current;
	}
	if (Alpha <= 0.0)
	{
		return Previous;
	}
	return FTransform(
		FQuat::Slerp(Previous.GetRotation(), Current.GetRotation(), Alpha),
		FMath::Lerp(Previous.GetLocation(), Current.GetLocation(), Alpha),
		FMath::Lerp(Previous.GetScale3D(), Current.GetScale3D(), Alpha));
}
