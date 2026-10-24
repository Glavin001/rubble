// Rubble.Abi: the real engine through its C ABI, in Unreal coordinates. A stack stands, every scene
// the ABI builds runs without NaN or bodies falling through the ground, and a shot breaks the house.

#include "Misc/AutomationTest.h"
#include "RubbleAbiBackend.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace
{
	/** Step Seconds of the loaded scene; every frame's moves are handed to Visit. */
	void Run(FRubbleAbiBackend& B, double Seconds, TFunctionRef<void(const FRubbleFrame&)> Visit)
	{
		const int32 Steps = FMath::RoundToInt32(Seconds / B.StepSeconds());
		TArray<FRubbleFrame> Frames;
		for (int32 s = 0; s < Steps; ++s)
		{
			B.Step();
			Frames.Reset();
			B.Poll(Frames);
			for (const FRubbleFrame& F : Frames)
			{
				Visit(F);
			}
		}
	}

	bool Load(FAutomationTestBase& T, FRubbleAbiBackend& B, const TCHAR* Scene)
	{
		FRubbleSceneRequest R;
		FString Error;
		const bool bOk = FRubbleSceneRequest::Parse(Scene, R, Error) && B.Load(R, Error);
		T.TestTrue(*FString::Printf(TEXT("%s loads (%s)"), Scene, *Error), bOk);
		T.TestTrue(*FString::Printf(TEXT("%s is on the ABI (%s)"), Scene, B.Name()), B.IsOnAbi());
		return bOk && B.IsOnAbi();
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleAbiStackTest, "Rubble.Abi.StackStands",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleAbiStackTest::RunTest(const FString& Parameters)
{
	FString Error;
	if (!TestTrue(*FString::Printf(TEXT("ABI version (%s)"), *Error), FRubbleAbiBackend::CheckVersion(Error)))
	{
		return false;
	}
	FRubbleAbiBackend B;
	if (!Load(*this, B, TEXT("rigid:legacy-stack")))
	{
		return false;
	}
	// Ten 1 m boxes, the top one at 9.5 m: after 3 s it is still there (Rubble holds stacks).
	TMap<uint64, FVector> Last;
	int32 Frames = 0;
	Run(B, 3.0, [&](const FRubbleFrame& F) {
		++Frames;
		for (const FRubbleMove& M : F.Moves)
		{
			Last.Add(M.Body.Bits, M.Transform.GetLocation());
		}
	});
	TestTrue(TEXT("frames arrived"), Frames > 100);
	double Top = 0.0;
	for (const FRubbleBodyVisual& V : B.Assets().Bodies)
	{
		const FVector* P = Last.Find(V.Id.Bits);
		const FVector At = P ? *P : V.Initial.GetLocation();
		Top = FMath::Max(Top, At.Z);
	}
	TestTrue(*FString::Printf(TEXT("the stack stands: top box at %.1f cm (built at 950)"), Top), Top > 900.0 && Top < 1000.0);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleAbiScenesTest, "Rubble.Abi.Scenes",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleAbiScenesTest::RunTest(const FString& Parameters)
{
	FRubbleAbiBackend B;
	int32 OnAbi = 0;
	for (const FRubbleDemoDesc& D : B.ListDemos())
	{
		FRubbleSceneRequest R;
		R.Family = D.Family;
		R.Id = D.Id;
		if (!FRubbleAbiBackend::MockReason(R).IsEmpty() || D.Id == TEXT("gables506k"))
		{
			continue; // the mock's (or, for 506k bodies, too slow for this test)
		}
		FString Error;
		if (!TestTrue(*FString::Printf(TEXT("%s loads (%s)"), *R.ToString(), *Error), B.Load(R, Error)))
		{
			continue;
		}
		++OnAbi;
		int32 Bad = 0, Below = 0, Moves = 0, Wheels = 0;
		Run(B, 2.0, [&](const FRubbleFrame& F) {
			for (const FRubbleMove& M : F.Moves)
			{
				++Moves;
				Bad += M.Transform.ContainsNaN() ? 1 : 0;
				// Vehicle presets park spare and detachable wheels at Rubble y = -5000 m
				// (engine-vehicle pack::parked_position): far below, not fallen.
				const double Z = M.Transform.GetLocation().Z;
				Below += Z < -200.0 && Z > -100000.0 ? 1 : 0;
			}
			Wheels += F.Wheels.Num();
		});
		TestEqual(*FString::Printf(TEXT("%s: no NaN (%d moves)"), *R.ToString(), Moves), Bad, 0);
		TestEqual(*FString::Printf(TEXT("%s: nothing falls through the ground"), *R.ToString()), Below, 0);
		if (D.Family == ERubbleFamily::Vehicles)
		{
			TestTrue(*FString::Printf(TEXT("%s: wheels reported"), *R.ToString()), Wheels > 0);
		}
	}
	TestTrue(TEXT("most scenes are on the ABI"), OnAbi >= 25);
	return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleAbiFractureTest, "Rubble.Abi.HouseBreaks",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleAbiFractureTest::RunTest(const FString& Parameters)
{
	FRubbleAbiBackend B;
	if (!Load(*this, B, TEXT("fracture:house?autoshot=0.2")))
	{
		return false;
	}
	const int32 Chunks = B.Assets().Chunks.Num();
	TestTrue(*FString::Printf(TEXT("the house has drawn chunks (%d)"), Chunks), Chunks > 50);
	int32 Breaks = 0;
	TSet<uint64> Moved;
	TSet<uint64> ChunkBodies;
	for (const FRubbleChunkVisual& C : B.Assets().Chunks)
	{
		ChunkBodies.Add(C.Body.Bits);
	}
	Run(B, 4.0, [&](const FRubbleFrame& F) {
		for (const FRubbleEvent& E : F.Events)
		{
			Breaks += E.Type == ERubbleEventType::JointBroke || E.Type == ERubbleEventType::BondsBroken ? 1 : 0;
		}
		for (const FRubbleMove& M : F.Moves)
		{
			if (ChunkBodies.Contains(M.Body.Bits))
			{
				Moved.Add(M.Body.Bits);
			}
		}
	});
	TestTrue(*FString::Printf(TEXT("the shot breaks bonds (%d events)"), Breaks), Breaks > 0);
	TestTrue(*FString::Printf(TEXT("chunks move (%d)"), Moved.Num()), Moved.Num() > 0);
	return true;
}

#endif

#if WITH_DEV_AUTOMATION_TESTS
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleAbiArchDrawnTest, "Rubble.Abi.ArchDrawn",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleAbiArchDrawnTest::RunTest(const FString& Parameters)
{
	FRubbleAbiBackend B;
	if (!Load(*this, B, TEXT("fracture:arch?autoshot=0")))
	{
		return false;
	}
	// The chunks the ABI names, drawn at rest: they must make the whole arch, not one fragment.
	const FRubbleSceneAssets& A = B.Assets();
	FBox All(ForceInit);
	TSet<int32> Meshes;
	for (const FRubbleChunkVisual& C : A.Chunks)
	{
		Meshes.Add(C.Mesh);
		for (const FVector3f& P : A.Meshes[C.Mesh].Positions)
		{
			All += FVector(P);
		}
	}
	AddInfo(FString::Printf(TEXT("%d chunks, %d distinct meshes, bounds %s"), A.Chunks.Num(), Meshes.Num(), *All.ToString()));
	TestEqual(TEXT("one mesh per chunk"), Meshes.Num(), A.Chunks.Num());
	TestTrue(*FString::Printf(TEXT("the chunks span the arch (%s)"), *All.GetSize().ToString()), All.GetSize().Y > 300.0 && All.GetSize().Z > 250.0);
	TestTrue(*FString::Printf(TEXT("it stands on the ground (lowest %.1f cm)"), All.Min.Z), FMath::Abs(All.Min.Z) < 2.0);
	TestTrue(TEXT("the lab's 23 stones"), A.Chunks.Num() == 23);
	return true;
}
#endif
