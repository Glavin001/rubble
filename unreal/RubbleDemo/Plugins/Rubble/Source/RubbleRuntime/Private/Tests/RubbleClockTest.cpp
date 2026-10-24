// Rubble.Clock: the fixed-step accumulator, the per-display-frame merge and pose interpolation.

#include "Misc/AutomationTest.h"
#include "RubbleClock.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleClockTest, "Rubble.Clock.Accumulator",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleClockTest::RunTest(const FString& Parameters)
{
	FRubbleClock Clock;
	Clock.Reset(1.0 / 60.0, 4);
	TestEqual(TEXT("half a step owes nothing yet"), Clock.Advance(0.5 / 60.0), 0);
	TestEqual(TEXT("the second half makes one step"), Clock.Advance(0.5 / 60.0 + 1e-9), 1);
	TestTrue(TEXT("alpha near 0 after a whole step"), Clock.Alpha() < 1e-3);
	TestEqual(TEXT("2.5 steps of time: 2 steps"), Clock.Advance(2.5 / 60.0), 2);
	TestTrue(TEXT("alpha 0.5"), FMath::IsNearlyEqual(Clock.Alpha(), 0.5, 1e-6));
	// A 1 s hitch: capped at 4 steps, the rest dropped (reported), nothing owed.
	const int32 Hitch = Clock.Advance(1.0);
	TestEqual(TEXT("capped"), Hitch, 4);
	TestTrue(TEXT("dropped time reported"), Clock.Dropped > 0.9);
	TestEqual(TEXT("nothing owed after the cap"), Clock.Owed, 0.0);
	// 240 Hz: four steps per 60 Hz frame.
	Clock.Reset(1.0 / 240.0, 8);
	int32 Steps = 0;
	for (int32 f = 0; f < 60; ++f)
	{
		Steps += Clock.Advance(1.0 / 60.0);
	}
	TestTrue(TEXT("240 steps in a second at 240 Hz"), FMath::Abs(Steps - 240) <= 1);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleMergeTest, "Rubble.Clock.Merge",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleMergeTest::RunTest(const FString& Parameters)
{
	auto Move = [](uint32 Body, double X0, double X1, bool bWoke = false, bool bAsleep = false) {
		FRubbleMove M;
		M.Body = FRubbleBodyId::Make(Body, 1);
		M.Previous = FTransform(FVector(X0, 0, 0));
		M.Transform = FTransform(FVector(X1, 0, 0));
		M.bHasPrevious = true;
		M.bWoke = bWoke;
		M.bFellAsleep = bAsleep;
		return M;
	};
	TArray<FRubbleFrame> Frames;
	Frames.SetNum(3);
	for (int32 i = 0; i < 3; ++i)
	{
		Frames[i].Step = i;
	}
	// Body 1 moves every step; body 2 only in the first (woke) and stops; body 3 falls asleep last.
	Frames[0].Moves = { Move(1, 0, 1), Move(2, 0, 5, true), Move(3, 0, 1) };
	Frames[1].Moves = { Move(1, 1, 2), Move(3, 1, 2) };
	Frames[2].Moves = { Move(1, 2, 3), Move(3, 2, 2.5, false, true) };
	FRubbleEvent E;
	E.Type = ERubbleEventType::BondsBroken;
	Frames[0].Events.Add(E);
	Frames[2].Events.Add(E);
	Frames[1].LostEvents = 2;
	FRubbleWheelVisual W0, W2;
	W0.SpinRad = 0.1f;
	W2.SpinRad = 0.3f;
	Frames[0].Wheels.Add(W0);
	Frames[2].Wheels.Add(W2);

	FRubbleFrame Out;
	FRubbleFrameMerger::Merge(MoveTemp(Frames), Out);
	TestEqual(TEXT("latest step"), Out.Step, uint64(2));
	TestEqual(TEXT("one move per body"), Out.Moves.Num(), 3);
	TestEqual(TEXT("events concatenated"), Out.Events.Num(), 2);
	TestEqual(TEXT("lost events summed"), Out.LostEvents, 2u);
	TestEqual(TEXT("latest wheel per wheel"), Out.Wheels.Num(), 1);
	TestEqual(TEXT("latest wheel value"), Out.Wheels[0].SpinRad, 0.3f);
	for (const FRubbleMove& M : Out.Moves)
	{
		const double P = M.Previous.GetLocation().X, C = M.Transform.GetLocation().X;
		switch (M.Body.Index())
		{
		case 1:
			TestEqual(TEXT("body 1: latest transform"), C, 3.0);
			TestEqual(TEXT("body 1: the latest step's previous (interpolate across the last step)"), P, 2.0);
			break;
		case 2:
			TestEqual(TEXT("body 2: where it stopped"), C, 5.0);
			TestEqual(TEXT("body 2: at rest in the latest step"), P, 5.0);
			TestTrue(TEXT("body 2: woke in some step"), M.bWoke);
			break;
		case 3:
			TestTrue(TEXT("body 3: asleep (latest state)"), M.bFellAsleep);
			TestEqual(TEXT("body 3: latest"), C, 2.5);
			break;
		}
	}
	// Interpolation.
	const FTransform A(FQuat::Identity, FVector(0, 0, 0)), B(FQuat(FVector::UpVector, UE_HALF_PI), FVector(10, 0, 0));
	TestTrue(TEXT("alpha 0"), RubbleLerpPose(A, B, 0.0).Equals(A));
	TestTrue(TEXT("alpha 1"), RubbleLerpPose(A, B, 1.0).Equals(B));
	const FTransform Mid = RubbleLerpPose(A, B, 0.5);
	TestTrue(TEXT("alpha 0.5 position"), Mid.GetLocation().Equals(FVector(5, 0, 0)));
	TestTrue(TEXT("alpha 0.5 rotation"), Mid.GetRotation().Equals(FQuat(FVector::UpVector, UE_HALF_PI * 0.5), 1e-6));
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleRequestTest, "Rubble.Clock.SceneRequest",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleRequestTest::RunTest(const FString& Parameters)
{
	FRubbleSceneRequest R;
	FString Error;
	TestTrue(TEXT("parses"), FRubbleSceneRequest::Parse(TEXT("soft:wall?speed=56&overlap=0.25"), R, Error));
	TestEqual(TEXT("family"), int32(R.Family), int32(ERubbleFamily::Soft));
	TestEqual(TEXT("id"), R.Id, FString(TEXT("wall")));
	TestEqual(TEXT("speed"), R.Param(TEXT("speed"), 0), 56.0f);
	TestEqual(TEXT("overlap"), R.Param(TEXT("overlap"), 0), 0.25f);
	TestFalse(TEXT("unknown family refused"), FRubbleSceneRequest::Parse(TEXT("cloth:flag"), R, Error));
	TestFalse(TEXT("non-numeric value refused"), FRubbleSceneRequest::Parse(TEXT("soft:wall?speed=fast"), R, Error));
	TestFalse(TEXT("no family refused"), FRubbleSceneRequest::Parse(TEXT("wall"), R, Error));
	return true;
}

#endif
