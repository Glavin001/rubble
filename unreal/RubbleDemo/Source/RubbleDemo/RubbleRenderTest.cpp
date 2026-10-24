// Rubble.Render: the renderers in the running game (scripts/unreal/test.sh runs them with -game).
//
//  * Bookkeeping: every primitive shape is one instance; after splits every chunk is drawn exactly
//    once; a soft body draws every vertex of its topology.
//  * Shots: one screenshot per family (and the scale scenes) into -RubbleShotDir= (default
//    Saved/RubbleShots), after the scene has run a while.
//  * Perf: frame time and the primitive renderer's update time on the scale scenes, logged as
//    "Rubble.Perf <scene> ..." lines and written to <ShotDir>/perf.txt.

#include "Engine/Engine.h"
#include "Engine/GameInstance.h"
#include "Engine/GameViewportClient.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "HAL/PlatformTime.h"
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "RubbleFracture.h"
#include "RubblePawn.h"
#include "RubblePrimitives.h"
#include "RubbleSoftBodies.h"
#include "RubbleSubsystem.h"
#include "Tests/AutomationCommon.h"
#include "UnrealClient.h"

#if WITH_DEV_AUTOMATION_TESTS

namespace
{
	UWorld* GameWorld()
	{
		for (const FWorldContext& Context : GEngine->GetWorldContexts())
		{
			if ((Context.WorldType == EWorldType::Game || Context.WorldType == EWorldType::PIE) && Context.World())
			{
				return Context.World();
			}
		}
		return nullptr;
	}

	template <typename T>
	T* Find()
	{
		UWorld* World = GameWorld();
		for (TActorIterator<T> It(World); World && It; ++It)
		{
			return *It;
		}
		return nullptr;
	}

	URubbleSubsystem* Rubble()
	{
		UWorld* World = GameWorld();
		return World && World->GetGameInstance() ? World->GetGameInstance()->GetSubsystem<URubbleSubsystem>() : nullptr;
	}

	bool Load(const TCHAR* Scene)
	{
		FRubbleSceneRequest R;
		FString Error;
		return FRubbleSceneRequest::Parse(Scene, R, Error) && Rubble()->LoadScene(R, Error);
	}

	FString ShotDir()
	{
		FString Dir = FPaths::ProjectSavedDir() / TEXT("RubbleShots");
		FParse::Value(FCommandLine::Get(), TEXT("RubbleShotDir="), Dir);
		return Dir;
	}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleRenderBookkeeping, "Rubble.Render.Bookkeeping",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleRenderBookkeeping::RunTest(const FString& Parameters)
{
	URubbleSubsystem* R = Rubble();
	ARubblePrimitives* Primitives = Find<ARubblePrimitives>();
	ARubbleFracture* Fracture = Find<ARubbleFracture>();
	ARubbleSoftBodies* Soft = Find<ARubbleSoftBodies>();
	if (!TestNotNull(TEXT("a game world with the Rubble actors"), R) || !Primitives || !Fracture || !Soft)
	{
		return false;
	}

	// Primitives: one instance per shape of every drawn body.
	for (const TCHAR* Scene : { TEXT("rigid:wall"), TEXT("rigid:legacy-scatter"), TEXT("vehicles:playground") })
	{
		TestTrue(*FString::Printf(TEXT("%s loads"), Scene), Load(Scene));
		int32 Shapes = 0;
		for (const FRubbleBodyVisual& B : R->Assets()->Bodies)
		{
			Shapes += B.NumShapes;
		}
		TestEqual(*FString::Printf(TEXT("%s: instances = shapes"), Scene), Primitives->NumInstances(), Shapes);
	}

	// Fracture: shoot the house, run 3 s; every chunk drawn exactly once, proxy or piece.
	TestTrue(TEXT("house loads"), Load(TEXT("fracture:house?autoshot=0.3")));
	{
		const TArray<int32> Before = Fracture->ChunkDrawCounts();
		TestTrue(TEXT("intact: every chunk drawn once (in the proxy)"), Before.Num() > 0 && !Before.ContainsByPredicate([](int32 n) { return n != 1; }));
		TestTrue(TEXT("intact: proxy shown"), Fracture->IsProxyShown(0));
		for (int32 f = 0; f < 180; ++f)
		{
			R->Advance(1.0 / 60.0);
		}
		const TArray<int32> After = Fracture->ChunkDrawCounts();
		TestFalse(TEXT("shot: the proxy is swapped for pieces"), Fracture->IsProxyShown(0));
		TestTrue(TEXT("shot: more than one body"), Fracture->NumBodies() > 1);
		const int32 Wrong = After.FilterByPredicate([](int32 n) { return n != 1; }).Num();
		TestEqual(TEXT("shot: every chunk drawn exactly once"), Wrong, 0);
	}

	// Soft bodies: the skin draws every vertex of the topology.
	TestTrue(TEXT("soft wall loads"), Load(TEXT("soft:wall")));
	for (int32 f = 0; f < 120; ++f)
	{
		R->Advance(1.0 / 60.0);
	}
	TestTrue(TEXT("a soft body"), R->Assets()->SoftBodies.Num() > 0);
	TestEqual(TEXT("soft: drawn vertices = topology vertices"), Soft->NumDrawnVertices(0), Soft->NumTopologyVertices(0));
	return true;
}

/** Load scenes one after another, let each run, and screenshot it. */
IMPLEMENT_COMPLEX_AUTOMATION_TEST(FRubbleRenderShots, "Rubble.Render.Shots",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

void FRubbleRenderShots::GetTests(TArray<FString>& OutBeautifiedNames, TArray<FString>& OutTestCommands) const
{
	static const TCHAR* Scenes[] = {
		TEXT("fracture:house?autoshot=0"), TEXT("fracture:house"), TEXT("fracture:suburb"), TEXT("fracture:arch?autoshot=0"), TEXT("fracture:arch"),
		TEXT("soft:wall"), TEXT("soft:headon"), TEXT("soft:traffic"),
		TEXT("vehicles:playground"), TEXT("vehicles:traffic"),
		TEXT("rigid:wall"), TEXT("rigid:ring"), TEXT("rigid:legacy-scatter"), TEXT("rigid:columns"),
	};
	for (const TCHAR* S : Scenes)
	{
		FString Name = S;
		Name.ReplaceInline(TEXT(":"), TEXT("_"));
		OutBeautifiedNames.Add(Name);
		OutTestCommands.Add(S);
	}
}

bool FRubbleRenderShots::RunTest(const FString& Scene)
{
	if (!TestTrue(*FString::Printf(TEXT("%s loads"), *Scene), Rubble() && Load(*Scene)))
	{
		return false;
	}
	FString File = Scene;
	File.ReplaceInline(TEXT(":"), TEXT("_"));
	File.ReplaceInline(TEXT("?"), TEXT("_"));
	File.ReplaceInline(TEXT("="), TEXT("-"));
	const FString Path = ShotDir() / (File + TEXT(".png"));
	// Crash tests and the auto shots happen within the first seconds.
	const float Wait = Scene.StartsWith(TEXT("soft:")) ? 2.2f : Scene.StartsWith(TEXT("fracture:")) ? 3.5f : 3.0f;
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(Wait));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([Path]() {
		FScreenshotRequest::RequestScreenshot(Path, /*bShowUI=*/true, /*bAddFilenameSuffix=*/false);
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.6f));
	return true;
}

/** Frame time on the scale scenes: mean and 95th percentile over 4 s, after 1 s to settle in. */
IMPLEMENT_COMPLEX_AUTOMATION_TEST(FRubbleRenderPerf, "Rubble.Render.Perf",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

void FRubbleRenderPerf::GetTests(TArray<FString>& OutBeautifiedNames, TArray<FString>& OutTestCommands) const
{
	static const TCHAR* Scenes[] = {
		TEXT("rigid:motorfield10k"),   // 10k kinematic bodies moving every step
		TEXT("rigid:legacy-grid10k"),  // 10k falling, then asleep
		TEXT("rigid:legacy-grid100k"), // 100k falling, then asleep
		TEXT("rigid:columns"),         // 100k asleep
		TEXT("soft:traffic?cars=20"),
	};
	for (const TCHAR* S : Scenes)
	{
		FString Name = S;
		Name.ReplaceInline(TEXT(":"), TEXT("_"));
		Name.ReplaceInline(TEXT("?"), TEXT("_"));
		OutBeautifiedNames.Add(Name);
		OutTestCommands.Add(S);
	}
}

bool FRubbleRenderPerf::RunTest(const FString& Scene)
{
	if (!TestTrue(*FString::Printf(TEXT("%s loads"), *Scene), Rubble() && Load(*Scene)))
	{
		return false;
	}
	struct FSample
	{
		TArray<double> Frame, Update, Backend;
		double Last = 0.0;
		int32 MaxMoves = 0;
	};
	TSharedRef<FSample> S = MakeShared<FSample>();
	const double Start = FPlatformTime::Seconds();
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([S, Start, Scene]() {
		const double Now = FPlatformTime::Seconds();
		if (S->Last > 0.0)
		{
			S->Frame.Add(1000.0 * (Now - S->Last));
			if (const ARubblePrimitives* P = Find<ARubblePrimitives>())
			{
				S->Update.Add(P->LastUpdateMs());
			}
			S->Backend.Add(Rubble()->Stats().BackendMs);
			S->MaxMoves = FMath::Max(S->MaxMoves, Rubble()->Stats().Moves);
		}
		S->Last = Now;
		if (Now - Start < 5.0)
		{
			return false;
		}
		auto Mean = [](const TArray<double>& V) { double T = 0; for (double x : V) T += x; return V.Num() ? T / V.Num() : 0.0; };
		auto P95 = [](TArray<double> V) { V.Sort(); return V.Num() ? V[FMath::Min(V.Num() - 1, int32(0.95 * V.Num()))] : 0.0; };
		const FString Line = FString::Printf(TEXT("Rubble.Perf %s frames=%d frame_ms_mean=%.2f frame_ms_p95=%.2f primitives_update_ms_mean=%.2f backend_ms_mean=%.2f max_moves=%d"),
			*Scene, S->Frame.Num(), Mean(S->Frame), P95(S->Frame), Mean(S->Update), Mean(S->Backend), S->MaxMoves);
		UE_LOG(LogRubble, Display, TEXT("%s"), *Line);
		FFileHelper::SaveStringToFile(Line + LINE_TERMINATOR, *(ShotDir() / TEXT("perf.txt")), FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
		return true;
	}));
	return true;
}

#endif

#if WITH_DEV_AUTOMATION_TESTS
#include "RubbleGpuScenePoc.h"
#include "RubbleInstancesComponent.h"
#include "RenderUtils.h"
#include "RHIGlobals.h"

/**
 * Rubble.Poc.GpuScene (run with -RubblePoc=gpuscene): 10k boxes whose transforms a compute pass
 * writes into GPU Scene are drawn by Nanite, and are lit by Lumen with virtual shadows.
 * Screenshots: lit (Lumen on), lit with Lumen off (the difference is Lumen's lighting on them),
 * and Nanite's triangle view (only Nanite geometry is coloured by cluster).
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubblePocGpuScene, "Rubble.Poc.GpuScene",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubblePocGpuScene::RunTest(const FString& Parameters)
{
	ARubbleGpuScenePoc* Poc = Find<ARubbleGpuScenePoc>();
	if (!TestNotNull(TEXT("the PoC actor (-RubblePoc=gpuscene)"), Poc))
	{
		return false;
	}
	TestTrue(TEXT("the Nanite box asset is there (build-content.sh)"), Poc->UsesNaniteMesh());
	TestEqual(TEXT("10k GPU-only instances"), Poc->Instances->GetNumInstances(), Poc->Side * Poc->Side);
	// Look at the field from above and the side.
	if (APlayerController* PC = GameWorld()->GetFirstPlayerController())
	{
		if (APawn* Pawn = PC->GetPawn())
		{
			if (ARubblePawn* RP = Cast<ARubblePawn>(Pawn))
			{
				RP->SetMode(ERubbleCamera::Fly);
			}
			Pawn->SetActorLocationAndRotation(FVector(-9000, -6000, 4500), FRotator(-22, 32, 0));
		}
	}
	const FString Dir = ShotDir() / TEXT("poc");
	auto Shot = [Dir](const TCHAR* Name) {
		return [Dir, Name]() { FScreenshotRequest::RequestScreenshot(Dir / FString(Name) + TEXT(".png"), false, false); return true; };
	};
	auto Exec = [](const TCHAR* Cmd) {
		return [Cmd]() { GEngine->Exec(GameWorld(), Cmd); return true; };
	};
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(3.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, Poc]() {
		TestTrue(TEXT("the GPU Scene instances have a scene proxy"), Poc->Instances->GetSceneProxy() != nullptr);
		if (DoesPlatformSupportNanite(GMaxRHIShaderPlatform))
		{
			TestTrue(TEXT("Unreal made a Nanite scene proxy for the GPU Scene instances"), Poc->Instances->IsNaniteProxy());
		}
		else
		{
			// Mac needs METAL_SM6 for Nanite, and UE 5.8 enables SM6 only on macOS 15+ (MetalRHI.cpp).
			AddWarning(FString::Printf(TEXT("%s does not support Nanite: only the non-Nanite GPU Scene path was checked"),
				*LexToString(GMaxRHIShaderPlatform)));
		}
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Shot(TEXT("lit_lumen_vsm"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.DynamicGlobalIlluminationMethod 0"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Shot(TEXT("lit_no_gi"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.DynamicGlobalIlluminationMethod 1"))));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.Nanite.Visualize Triangles"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Shot(TEXT("nanite_triangles"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.Nanite.Visualize Overview"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Shot(TEXT("nanite_overview"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.Nanite.Visualize Off"))));
	// Close up, low sun side: Lumen's bounce from the ground shows on the boxes' shadowed faces.
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([]() {
		if (APawn* Pawn = GameWorld()->GetFirstPlayerController()->GetPawn())
		{
			Pawn->SetActorLocationAndRotation(FVector(-2200, -2200, 350), FRotator(-12, 40, 0));
		}
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(2.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Shot(TEXT("close_lumen"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.DynamicGlobalIlluminationMethod 0"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Shot(TEXT("close_no_gi"))));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand(Exec(TEXT("r.DynamicGlobalIlluminationMethod 1"))));
	return true;
}
#endif
