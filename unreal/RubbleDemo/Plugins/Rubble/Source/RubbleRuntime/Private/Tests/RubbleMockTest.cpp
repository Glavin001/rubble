// Rubble.Mock: every catalogued scene loads and runs 5 s without NaN; results are one step late;
// sleeping bodies stop reporting; a split partitions its parent's chunks exactly.

#include "Misc/AutomationTest.h"
#include "RubbleMockBackend.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace
{
	bool Finite(const FTransform& T)
	{
		return !T.ContainsNaN() && T.GetLocation().GetAbsMax() < 1e8;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleMockScenesTest, "Rubble.Mock.AllScenes",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleMockScenesTest::RunTest(const FString& Parameters)
{
	FRubbleMockBackend Backend;
	const TArray<FRubbleDemoDesc> Demos = Backend.ListDemos();
	TestTrue(TEXT("the catalogue has every family"), Demos.Num() >= 50);
	for (const FRubbleDemoDesc& D : Demos)
	{
		FRubbleSceneRequest R;
		R.Family = D.Family;
		R.Id = D.Id;
		FString Error;
		if (!TestTrue(*FString::Printf(TEXT("%s loads (%s)"), *R.ToString(), *Error), Backend.Load(R, Error)))
		{
			continue;
		}
		const FRubbleSceneAssets& A = Backend.Assets();
		for (int32 b = 0; b < A.Bodies.Num(); ++b)
		{
			if (A.Bodies[b].Id.Index() != uint32(b))
			{
				AddError(FString::Printf(TEXT("%s: body visual %d has id index %u"), *R.ToString(), b, A.Bodies[b].Id.Index()));
				break;
			}
		}
		const int32 Steps = FMath::RoundToInt32(5.0 / Backend.StepSeconds());
		int32 Bad = 0, Moves = 0;
		TArray<FRubbleFrame> Frames;
		for (int32 s = 0; s < Steps; ++s)
		{
			Backend.Step();
			Frames.Reset();
			Backend.Poll(Frames);
			for (const FRubbleFrame& F : Frames)
			{
				for (const FRubbleMove& M : F.Moves)
				{
					Bad += Finite(M.Transform) ? 0 : 1;
					++Moves;
				}
				for (const FRubbleWheelVisual& W : F.Wheels)
				{
					Bad += Finite(W.Hub) ? 0 : 1;
				}
				for (const FRubbleSoftSkin& S : F.Skins)
				{
					const FRubbleSoftVisual& V = A.SoftBodies[S.Soft];
					if (S.Positions.Num() != A.Meshes[V.Mesh].Positions.Num() || S.Normals.Num() != S.Positions.Num() || S.TornTriangles.Num() != A.Meshes[V.Mesh].NumTriangles())
					{
						++Bad;
					}
					for (const FVector3f& P : S.Positions)
					{
						Bad += P.ContainsNaN() ? 1 : 0;
					}
				}
			}
		}
		TestEqual(*FString::Printf(TEXT("%s: no NaN or malformed output over 5 s (%d moves)"), *R.ToString(), Moves), Bad, 0);
	}
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleMockLatencyTest, "Rubble.Mock.LatencyAndSleep",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleMockLatencyTest::RunTest(const FString& Parameters)
{
	FRubbleMockBackend Backend;
	FRubbleSceneRequest R;
	FString Error;
	FRubbleSceneRequest::Parse(TEXT("rigid:legacy-grid-boxes"), R, Error);
	TestTrue(TEXT("loads"), Backend.Load(R, Error));
	TArray<FRubbleFrame> Frames;
	Backend.Step();
	Backend.Poll(Frames);
	TestEqual(TEXT("nothing after the first step: results are a step late"), Frames.Num(), 0);
	Backend.Step();
	Backend.Poll(Frames);
	TestEqual(TEXT("step 0 after the second"), Frames.Num() == 1 ? int64(Frames[0].Step) : -1, int64(0));
	TestTrue(TEXT("the dropped grid moves"), Frames.Num() == 1 && Frames[0].Moves.Num() > 1000);

	// Run 8 s: every box lands and sleeps; then nothing moves, and nothing asleep is reported again.
	TSet<uint64> Asleep;
	int32 ReportedWhileAsleep = 0, LastMoves = -1;
	for (int32 s = 0; s < 480; ++s)
	{
		Backend.Step();
		Frames.Reset();
		Backend.Poll(Frames);
		for (const FRubbleFrame& F : Frames)
		{
			LastMoves = F.Moves.Num();
			for (const FRubbleMove& M : F.Moves)
			{
				if (Asleep.Contains(M.Body.Bits) && !M.bWoke)
				{
					++ReportedWhileAsleep;
				}
				if (M.bFellAsleep)
				{
					Asleep.Add(M.Body.Bits);
				}
				else
				{
					Asleep.Remove(M.Body.Bits);
				}
			}
		}
	}
	TestEqual(TEXT("a sleeping body is not reported until it wakes"), ReportedWhileAsleep, 0);
	TestEqual(TEXT("at rest, nothing moves"), LastMoves, 0);
	TestTrue(TEXT("the grid fell asleep"), Asleep.Num() >= 1000);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleMockSplitTest, "Rubble.Mock.SplitPartitions",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleMockSplitTest::RunTest(const FString& Parameters)
{
	FRubbleMockBackend Backend;
	FRubbleSceneRequest R;
	FString Error;
	FRubbleSceneRequest::Parse(TEXT("fracture:house?autoshot=0.5"), R, Error);
	TestTrue(TEXT("loads"), Backend.Load(R, Error));
	const int32 NumChunks = Backend.Assets().Chunks.Num();
	TestTrue(TEXT("the house has chunks"), NumChunks > 100);
	// Owner of each chunk, from the root bodies, updated by every split.
	TMap<int32, uint64> Owner;
	for (const FRubbleDestructibleVisual& D : Backend.Assets().Destructibles)
	{
		for (const int32 c : D.Chunks)
		{
			Owner.Add(c, D.RootBody.Bits);
		}
	}
	TestEqual(TEXT("every chunk has one destructible"), Owner.Num(), NumChunks);
	int32 Splits = 0;
	TArray<FRubbleFrame> Frames;
	for (int32 s = 0; s < 240; ++s)
	{
		Backend.Step();
		Frames.Reset();
		Backend.Poll(Frames);
		for (const FRubbleFrame& F : Frames)
		{
			for (const FRubbleEvent& E : F.Events)
			{
				if (E.Type != ERubbleEventType::Split)
				{
					continue;
				}
				++Splits;
				TSet<int32> FromChildren;
				for (const FRubbleActorInfo& C : E.Children)
				{
					for (const int32 c : C.Chunks)
					{
						TestFalse(TEXT("a chunk is in one child"), FromChildren.Contains(c));
						FromChildren.Add(c);
						Owner.Add(c, C.Body.Bits);
					}
				}
				TSet<int32> FromParent(E.Parent.Chunks);
				TestTrue(TEXT("the children partition the parent's chunks"), FromChildren.Num() == FromParent.Num() && FromChildren.Includes(FromParent));
				for (const int32 c : E.Parent.Chunks)
				{
					TestEqual(TEXT("the parent owned its chunks"), Owner.Contains(c), true);
				}
			}
		}
	}
	TestTrue(TEXT("the auto shot split the house"), Splits >= 1);
	return true;
}

#endif
