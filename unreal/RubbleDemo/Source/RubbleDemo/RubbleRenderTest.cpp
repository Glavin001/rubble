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
#include "Misc/App.h"
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "DynamicRHI.h"
#include "RenderTimer.h"
#include "RubbleAbiBackend.h"
#include "RubbleFilmCapture.h"
#include "RubbleFilmRig.h"
#include "RubbleFracture.h"
#include "RubbleGpuPoses.h"
#include "RubblePawn.h"
#include "RubblePrimitives.h"
#include "RubbleReplayBackend.h"
#include "RubbleSoftBodies.h"
#include "RubbleVehicles.h"
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

/**
 * The destruction scenes at scale: frame, renderer, backend and GPU times over 6 s (the auto
 * shot lands inside it), the most moves in a frame, how much the GPU Scene path draws, and a
 * screenshot after the shot. Lines go to RubbleShots/destruction.txt.
 */
IMPLEMENT_COMPLEX_AUTOMATION_TEST(FRubbleRenderDestruction, "Rubble.Render.Destruction",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

void FRubbleRenderDestruction::GetTests(TArray<FString>& OutBeautifiedNames, TArray<FString>& OutTestCommands) const
{
	static const TCHAR* Scenes[] = {
		TEXT("rigid:ring110k"), TEXT("rigid:ring"), TEXT("rigid:gables"), TEXT("rigid:pile"), TEXT("rigid:wall"),
		TEXT("fracture:house"), TEXT("fracture:suburb"), TEXT("fracture:arch"), TEXT("fracture:town"),
		TEXT("vehicles:destruction"), TEXT("vehicles:playground"), TEXT("vehicles:traffic"),
		TEXT("soft:wall"), TEXT("soft:headon"), TEXT("soft:tbone"), TEXT("soft:traffic"),
	};
	for (const TCHAR* S : Scenes)
	{
		FString Name = S;
		Name.ReplaceInline(TEXT(":"), TEXT("_"));
		OutBeautifiedNames.Add(Name);
		OutTestCommands.Add(S);
	}
}

bool FRubbleRenderDestruction::RunTest(const FString& Scene)
{
	if (!TestTrue(*FString::Printf(TEXT("%s loads"), *Scene), Rubble() && Load(*Scene)))
	{
		return false;
	}
	GEngine->Exec(GameWorld(), TEXT("t.MaxFPS 0"));
	struct FSample
	{
		TArray<double> Frame, Update, Backend, Gpu, Game, Render;
		double Last = 0.0;
		int32 MaxMoves = 0;
		int64 Steps = 0;
		bool bShot = false;
	};
	TSharedRef<FSample> S = MakeShared<FSample>();
	const double Start = FPlatformTime::Seconds();
	FString File = Scene;
	File.ReplaceInline(TEXT(":"), TEXT("_"));
	const FString Shot = ShotDir() / TEXT("destruction") / (File + TEXT(".png"));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([S, Start, Scene, Shot]() {
		const double Now = FPlatformTime::Seconds();
		if (S->Last > 0.0)
		{
			S->Frame.Add(1000.0 * (Now - S->Last));
			if (const ARubblePrimitives* P = Find<ARubblePrimitives>())
			{
				S->Update.Add(P->LastUpdateMs());
			}
			S->Backend.Add(Rubble()->Stats().BackendMs);
			S->Gpu.Add(RHIGetGPUFrameCycles() * FPlatformTime::GetSecondsPerCycle() * 1000.0);
			S->MaxMoves = FMath::Max(S->MaxMoves, Rubble()->Stats().Moves);
			S->Steps += Rubble()->Stats().Steps;
			S->Game.Add(GGameThreadTime * FPlatformTime::GetSecondsPerCycle() * 1000.0);
			S->Render.Add(GRenderThreadTime * FPlatformTime::GetSecondsPerCycle() * 1000.0);
		}
		S->Last = Now;
		if (!S->bShot && Now - Start > 4.0)
		{
			FScreenshotRequest::RequestScreenshot(Shot, /*bShowUI=*/true, /*bAddFilenameSuffix=*/false);
			S->bShot = true;
		}
		if (Now - Start < 7.0)
		{
			return false;
		}
		auto Mean = [](const TArray<double>& V) { double T = 0; for (double x : V) T += x; return V.Num() ? T / V.Num() : 0.0; };
		auto P95 = [](TArray<double> V) { V.Sort(); return V.Num() ? V[FMath::Min(V.Num() - 1, int32(0.95 * V.Num()))] : 0.0; };
		const ARubblePrimitives* P = Find<ARubblePrimitives>();
		const ARubbleFracture* F = Find<ARubbleFracture>();
		const FString Line = FString::Printf(TEXT("Rubble.Destruction %s gpu_poses=%d bodies=%d gpu_instances=%d gpu_chunks=%d frames=%d frame_ms_mean=%.2f frame_ms_p95=%.2f game_ms_mean=%.2f render_thread_ms_mean=%.2f gpu_ms_mean=%.2f renderer_ms_mean=%.2f backend_ms_mean=%.2f steps_per_frame=%.2f backend_ms_per_step=%.2f max_moves=%d"),
			*Scene, Rubble()->Backend()->GpuPoses() != nullptr, Rubble()->Assets()->Bodies.Num(), P ? P->NumGpuInstances() : 0, F ? F->NumGpuChunks() : 0,
			S->Frame.Num(), Mean(S->Frame), P95(S->Frame), Mean(S->Game), Mean(S->Render), Mean(S->Gpu), Mean(S->Update), Mean(S->Backend),
			S->Frame.Num() ? double(S->Steps) / S->Frame.Num() : 0.0, S->Steps ? Mean(S->Backend) * S->Frame.Num() / double(S->Steps) : 0.0, S->MaxMoves);
		UE_LOG(LogRubble, Display, TEXT("%s"), *Line);
		FFileHelper::SaveStringToFile(Line + LINE_TERMINATOR, *(ShotDir() / TEXT("destruction.txt")), FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), FILEWRITE_Append);
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.3f));
	return true;
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

#if WITH_DEV_AUTOMATION_TESTS
#include "RenderTimer.h"
#include "DynamicRHI.h"

namespace
{
	/** Frame timings sampled over the frames, logged as means and 95th percentiles (ms). */
	struct FRubbleFrameSamples
	{
		TArray<double> Frame, Game, Render, Gpu;

		void Add()
		{
			const double Ms = FPlatformTime::GetSecondsPerCycle() * 1000.0;
			Frame.Add(FApp::GetDeltaTime() * 1000.0);
			Game.Add(GGameThreadTime * Ms);
			Render.Add(GRenderThreadTime * Ms);
			Gpu.Add(RHIGetGPUFrameCycles() * Ms);
		}

		static FString Describe(const TCHAR* Name, TArray<double> V)
		{
			if (V.Num() == 0)
			{
				return FString(Name) + TEXT(" -");
			}
			V.Sort();
			double Sum = 0.0;
			for (double X : V)
			{
				Sum += X;
			}
			return FString::Printf(TEXT("%s %.2f ms (p95 %.2f)"), Name, Sum / V.Num(), V[FMath::Min(V.Num() - 1, int32(V.Num() * 0.95))]);
		}

		FString Report() const
		{
			return FString::Join(TArray<FString>{Describe(TEXT("frame"), Frame), Describe(TEXT("game"), Game),
				Describe(TEXT("render"), Render), Describe(TEXT("gpu"), Gpu)}, TEXT(", "));
		}
	};
}

/**
 * Rubble.Poc.Scale (run with -RubblePoc=gpuscene -RubblePocSide=N): frame, game-thread,
 * render-thread and GPU times with N x N GPU-written boxes in view, plus a GPU profile of the
 * frame (ProfileGPU, in the log) that times the Rubble passes themselves.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubblePocScale, "Rubble.Poc.Scale",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubblePocScale::RunTest(const FString& Parameters)
{
	ARubbleGpuScenePoc* Poc = Find<ARubbleGpuScenePoc>();
	if (!TestNotNull(TEXT("the PoC actor (-RubblePoc=gpuscene)"), Poc))
	{
		return false;
	}
	GEngine->Exec(GameWorld(), TEXT("t.MaxFPS 0"));
	GEngine->Exec(GameWorld(), TEXT("r.VSync 0"));
	if (APawn* Pawn = GameWorld()->GetFirstPlayerController()->GetPawn())
	{
		if (ARubblePawn* RP = Cast<ARubblePawn>(Pawn))
		{
			RP->SetMode(ERubbleCamera::Fly);
		}
		// The whole field in view, from a corner.
		const double Half = Poc->Side * 75.0;
		Pawn->SetActorLocationAndRotation(FVector(-Half * 1.15, -Half * 1.15, Half * 0.55), FRotator(-24, 45, 0));
	}
	const int32 Count = Poc->Instances->GetNumInstances();
	TSharedRef<FRubbleFrameSamples> Samples = MakeShared<FRubbleFrameSamples>();
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(4.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([Samples]() {
		Samples->Add();
		return Samples->Frame.Num() >= 300;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, Samples, Count]() {
		const FString Line = FString::Printf(TEXT("GPU Scene PoC, %d boxes, %s: %s"), Count,
			*LexToString(GMaxRHIShaderPlatform), *Samples->Report());
		UE_LOG(LogTemp, Display, TEXT("%s"), *Line);
		AddInfo(Line);
		FScreenshotRequest::RequestScreenshot(ShotDir() / FString::Printf(TEXT("poc/scale_%d.png"), Count), false, false);
		GEngine->Exec(GameWorld(), TEXT("ProfileGPU"));
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
/**
 * Rubble.Town.Shots: Vibe Town (fracture:town, no auto shot) from fixed cameras, for the look:
 * a street in Elm Park, a house wall up close, the Market Quarter's street toward the towers, a
 * shopfront and a tower's facade, and the aerial, with the world paused.
 * Poses are Unreal cm (Unreal X = -Rubble z, Y = Rubble x). Screenshots in RubbleShots/town.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleTownShots, "Rubble.Town.Shots",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleTownShots::RunTest(const FString& Parameters)
{
	if (!TestTrue(TEXT("the town loads"), Rubble() && Load(TEXT("fracture:town?autoshot=0"))))
	{
		return false;
	}
	// The look as built: no steps, so nothing settles between the shots.
	Rubble()->bPaused = true;
	struct FPose
	{
		const TCHAR* Name;
		FVector Eye;
		FVector Target;
	};
	static const FPose Poses[] = {
		{ TEXT("elm_park_street"), FVector(-2600, -9800, 260), FVector(-1700, -7200, 300) },
		{ TEXT("house_wall"), FVector(-2500, -7300, 220), FVector(-1870, -7260, 220) },
		// The Market Quarter's street toward the towers, a shopfront (shop 4: its front at Rubble
		// z = 7.4, Unreal X = -740) and tower 1's facade (z = -8).
		{ TEXT("main_street"), FVector(0, 300, 260), FVector(-700, 9000, 900) },
		{ TEXT("shopfront"), FVector(250, 2920, 170), FVector(-740, 2920, 230) },
		{ TEXT("tower"), FVector(-600, 8000, 500), FVector(800, 8986, 1500) },
		{ TEXT("aerial"), FVector(-14000, -9000, 7000), FVector(0, 0, 400) },
	};
	const FString Dir = ShotDir() / TEXT("town");
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(4.0f));
	for (const FPose& P : Poses)
	{
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([P]() {
			if (APawn* Pawn = GameWorld()->GetFirstPlayerController()->GetPawn())
			{
				if (ARubblePawn* RP = Cast<ARubblePawn>(Pawn))
				{
					RP->LookFrom(P.Eye, P.Target);
				}
				GameWorld()->GetFirstPlayerController()->SetControlRotation((P.Target - P.Eye).Rotation());
			}
			return true;
		}));
		ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.5f));
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([Dir, P]() {
			FScreenshotRequest::RequestScreenshot(Dir / FString(P.Name) + TEXT(".png"), /*bShowUI=*/false, false);
			return true;
		}));
		ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.4f));
	}
	return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
namespace
{
	void ParkCamera(const FVector& Eye, const FVector& Target)
	{
		if (APawn* Pawn = GameWorld()->GetFirstPlayerController()->GetPawn())
		{
			if (ARubblePawn* RP = Cast<ARubblePawn>(Pawn))
			{
				RP->LookFrom(Eye, Target);
			}
			GameWorld()->GetFirstPlayerController()->SetControlRotation((Target - Eye).Rotation());
		}
	}
}

/**
 * Rubble.Town.Destroy: Vibe Town wrecked from a street in Elm Park (three cannonballs into a
 * house, a meteor onto the next) and then from the Market Quarter (a meteor onto a ten-storey
 * tower). Frames every half second or so in RubbleShots/town/destroy_*.png; frame, game, render
 * thread, backend times and moves to RubbleShots/town/destroy.txt.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleTownDestroy, "Rubble.Town.Destroy",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleTownDestroy::RunTest(const FString& Parameters)
{
	if (!TestTrue(TEXT("the town loads"), Rubble() && Load(TEXT("fracture:town?autoshot=0"))))
	{
		return false;
	}
	GEngine->Exec(GameWorld(), TEXT("t.MaxFPS 0"));
	struct FTimeline
	{
		double Start = 0.0, Last = 0.0;
		int32 Next = 0, Shot = 0;
		TArray<double> Frame, Game, Render, Backend;
		int32 MaxMoves = 0, MaxEvents = 0, Events = 0, MaxAwake = 0;
	};
	TSharedRef<FTimeline> T = MakeShared<FTimeline>();
	const FString Dir = ShotDir() / TEXT("town");
	const FVector StreetEye(-4200, -7900, 450), HouseA(-1554, -5120, 250), HouseB(-1554, -3420, 0);
	const FVector TowerEye(-15000, 3000, 3200), Tower(-6289, 8995, 0);
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(4.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([T, Dir, StreetEye, HouseA, HouseB, TowerEye, Tower]() {
		const double Now = FPlatformTime::Seconds();
		if (T->Start == 0.0)
		{
			T->Start = T->Last = Now;
			ParkCamera(StreetEye, (HouseA + HouseB) * 0.5 + FVector(0, 0, 200));
			return false;
		}
		const double At = Now - T->Start;
		T->Frame.Add(1000.0 * (Now - T->Last));
		T->Last = Now;
		const double Ms = FPlatformTime::GetSecondsPerCycle() * 1000.0;
		T->Game.Add(GGameThreadTime * Ms);
		T->Render.Add(GRenderThreadTime * Ms);
		T->Backend.Add(Rubble()->Stats().BackendMs);
		T->MaxMoves = FMath::Max(T->MaxMoves, Rubble()->Stats().Moves);
		T->MaxEvents = FMath::Max(T->MaxEvents, Rubble()->Stats().Events);
		T->Events += Rubble()->Stats().Events;
		if (const ARubblePrimitives* Prims = Find<ARubblePrimitives>())
		{
			T->MaxAwake = FMath::Max(T->MaxAwake, Prims->NumAwakeHybrids());
		}
		// The script: (time s, what).
		struct FCue { double At; int32 What; };
		static const FCue Cues[] = {
			{ 1.0, 1 }, { 1.4, 1 }, { 1.8, 1 }, { 2.6, 2 },          // cannon x3, meteor onto house B
			{ 1.2, 10 }, { 2.0, 10 }, { 3.0, 10 }, { 4.0, 10 }, { 5.5, 10 }, { 7.5, 10 },
			{ 9.0, 3 },                                                  // to the towers
			{ 10.5, 4 },                                                 // meteor onto a tower
			{ 11.0, 10 }, { 12.0, 10 }, { 13.5, 10 }, { 16.0, 10 },
		};
		for (const FCue& C : Cues)
		{
			if (At < C.At || (T->Next & (1 << int32(&C - Cues))))
			{
				continue;
			}
			T->Next |= 1 << int32(&C - Cues);
			switch (C.What)
			{
			case 1: Rubble()->Launch(StreetEye + FVector(0, 0, -150), HouseA - StreetEye); break;
			case 2: Rubble()->Meteor(HouseB); break;
			case 3: ParkCamera(TowerEye, Tower + FVector(0, 0, 1500)); break;
			case 4: Rubble()->Meteor(Tower); break;
			case 10:
				UE_LOG(LogRubble, Display, TEXT("town destroy t=%.1f s: %d awake, %d moves, %.1f ms backend, %d events this frame"), At,
					Find<ARubblePrimitives>() ? Find<ARubblePrimitives>()->NumAwakeHybrids() : 0, Rubble()->Stats().Moves, Rubble()->Stats().BackendMs, Rubble()->Stats().Events);
				FScreenshotRequest::RequestScreenshot(Dir / FString::Printf(TEXT("destroy_%02d.png"), T->Shot++), /*bShowUI=*/true, false);
				break;
			}
		}
		if (At < 17.0)
		{
			return false;
		}
		auto Mean = [](const TArray<double>& V) { double S = 0; for (double x : V) S += x; return V.Num() ? S / V.Num() : 0.0; };
		auto P95 = [](TArray<double> V) { V.Sort(); return V.Num() ? V[FMath::Min(V.Num() - 1, int32(0.95 * V.Num()))] : 0.0; };
		auto Max = [](const TArray<double>& V) { double M = 0; for (double x : V) M = FMath::Max(M, x); return M; };
		const ARubblePrimitives* P = Find<ARubblePrimitives>();
		const FString Line = FString::Printf(TEXT("Rubble.Town.Destroy frames=%d frame_ms_mean=%.2f p95=%.2f max=%.1f game_ms=%.2f render_ms=%.2f backend_ms_mean=%.2f backend_ms_max=%.1f max_moves=%d events=%d max_events_frame=%d hybrids=%d awake_at_end=%d max_awake=%d"),
			T->Frame.Num(), Mean(T->Frame), P95(T->Frame), Max(T->Frame), Mean(T->Game), Mean(T->Render), Mean(T->Backend), Max(T->Backend), T->MaxMoves, T->Events, T->MaxEvents,
			P ? P->NumHybrids() : 0, P ? P->NumAwakeHybrids() : 0, T->MaxAwake);
		UE_LOG(LogRubble, Display, TEXT("%s"), *Line);
		FFileHelper::SaveStringToFile(Line + LINE_TERMINATOR, *(Dir / TEXT("destroy.txt")));
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.3f));
	return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
/**
 * Rubble.Video.Town: a short film of Vibe Town wrecked, every frame saved to
 * RubbleShots/video/town_NNNN.png (scripts/unreal/video.sh encodes them). Engine time runs at a
 * fixed 1/30 s a frame, so the simulation plays at real speed in the film however slow a frame is
 * to make. Two shots: cannonballs and a meteor in Elm Park, then a tower's ground-storey columns
 * shot away from one side, a volley a second, until it topples onto the grocery beside it
 * (support loss).
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleVideoTown, "Rubble.Video.Town",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleVideoTown::RunTest(const FString& Parameters)
{
	if (!TestTrue(TEXT("the town loads"), Rubble() && Load(TEXT("fracture:town?autoshot=0"))))
	{
		return false;
	}
	constexpr double Fps = 30.0;
	constexpr double Length = 20.0;
	const FString Dir = ShotDir() / TEXT("video");
	IFileManager::Get().DeleteDirectory(*Dir, false, true);
	IFileManager::Get().MakeDirectory(*Dir, true);
	struct FState
	{
		int32 Frame = -1;
		int32 Fired = 0;
	};
	TSharedRef<FState> S = MakeShared<FState>();
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(4.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([S, Dir, Fps, Length]() {
		if (S->Frame < 0)
		{
			FApp::SetUseFixedTimeStep(true);
			FApp::SetFixedDeltaTime(1.0 / Fps);
			S->Frame = 0;
		}
		const double At = S->Frame / Fps;
		// Shot 1, Elm Park: a slow push toward two houses. Shot 2: the glass tower (build-town.sh,
		// in tower-165's place) felled south onto the grocery (grocery-197) 7.8 m off its face: its
		// three southern column rows shot from the west on its lowest four storeys, a volley a second.
		const FVector HouseA(-1554, -5120, 250), HouseB(-1554, -3420, 0), Street(-4600, -8300, 380);
		const bool bTowers = At >= 6.5;
		const float u = bTowers ? float((At - 6.5) / (Length - 6.5)) : float(At / 6.5);
		const float Ease = u * u * (3.0f - 2.0f * u);
		// It falls south (+X): watched side-on from the west, 65 m out and over the grocery
		// between, framing the tower and the grocery it lands on.
		const FVector Eye = bTowers ? FMath::Lerp(FVector(2600, 2300, 2100), FVector(2700, 2900, 1800), Ease)
			: FMath::Lerp(Street, FVector(-3900, -7500, 520), Ease);
		const FVector Target = bTowers ? FVector(2700, 8900, 1000) : (HouseA + HouseB) * 0.5 + FVector(0, 0, 250);
		if (ARubblePawn* Pawn = Cast<ARubblePawn>(GameWorld()->GetFirstPlayerController()->GetPawn()))
		{
			Pawn->LookFrom(Eye, Target);
		}
		struct FCue { double At; int32 What; };
		static const FCue Cues[] = { { 1.0, 1 }, { 1.4, 1 }, { 1.8, 1 }, { 2.8, 2 }, { 7.5, 3 }, { 8.5, 3 }, { 9.5, 3 }, { 10.5, 3 } };
		for (int32 c = 0; c < UE_ARRAY_COUNT(Cues); ++c)
		{
			if (At >= Cues[c].At && !(S->Fired & (1 << c)))
			{
				S->Fired |= 1 << c;
				switch (Cues[c].What)
				{
				case 1: Rubble()->Launch(Street + FVector(0, 0, -150), HouseA - Street); break;
				case 2: Rubble()->Meteor(HouseB); break;
				case 3:
					// One ball along each of the three southern column rows (pack z -20.1, -16.6,
					// -13.1 m) at each storey's height (the ground storey twice), from 7 m west of the
					// tower (pack x 77.65 m), heading east (+Y). The pack's footprint shift: x +0.03 m,
					// z -0.365 m.
					for (const double Row : { -20.1, -16.6, -13.1 })
					{
						for (const double Up : { 100.0, 260.0, 530.0, 830.0, 1130.0 })
						{
							Rubble()->Launch(FVector(-(Row - 0.365) * 100.0, (77.65 + 0.03) * 100.0, Up), FVector(0, 1, 0));
						}
					}
					break;
				}
			}
		}
		FScreenshotRequest::RequestScreenshot(Dir / FString::Printf(TEXT("town_%04d.png"), S->Frame), /*bShowUI=*/false, false);
		if (++S->Frame < int32(Length * Fps))
		{
			return false;
		}
		FApp::SetUseFixedTimeStep(false);
		UE_LOG(LogRubble, Display, TEXT("Rubble.Video.Town: %d frames in %s"), S->Frame, *Dir);
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
namespace GlassTowerFilm
{
	enum class ERide : uint8
	{
		Orbit, // the slow orbit from the west
		Chunk, // riding a chunk of the tower: where it goes, the camera goes
		Fixed, // standing still, shaken by the breaks around it
	};
	/** One camera. Eye and Target are where it stands and looks with everything at rest (cm); a
	 *  riding camera keeps that pose relative to the chunk nearest Chunk. */
	struct FView
	{
		const TCHAR* Name;
		double Length;
		ERide Ride;
		FVector Chunk;
		FVector Eye;
		FVector Target;
		float Fov;
		/** Fixed cameras: degrees of shake at the peak of the breaking. */
		float Shake;
	};
	// The pack's footprint is centred: the tower stands at X -999 cm (X -1699..-299, Y -706..706,
	// 18 storeys of 3 m), the two-storey pavilion 7.8 m off its south face at X +1099 (X 499..1699,
	// Y -600..600). The tower falls south (+X).
	static const FView Views[] = {
		{ TEXT("orbit"), 30.0, ERide::Orbit, FVector::ZeroVector, FVector::ZeroVector, FVector::ZeroVector, 70.0f, 0.0f },
		// Standing on the twelfth floor (Z 3300), at the north side of its core, facing the pavilion.
		{ TEXT("inside"), 22.0, ERide::Chunk, FVector(-1350, 0, 3290), FVector(-1350, 0, 3465), FVector(1100, 0, 2900), 95.0f, 0.0f },
		// Leaning out over the roof's south edge (past its glass), looking down and out at the pavilion.
		{ TEXT("roof"), 22.0, ERide::Chunk, FVector(-420, 0, 5390), FVector(-180, 0, 5750), FVector(1500, 0, 4200), 95.0f, 0.0f },
		// On the pavilion's upper floor, at its north window, looking up at the tower.
		{ TEXT("pavilion"), 22.0, ERide::Fixed, FVector::ZeroVector, FVector(560, 150, 465), FVector(-999, 0, 1500), 100.0f, 2.5f },
		// On the ground 26 m east of the gap between them: the tower comes down across the view.
		{ TEXT("street"), 22.0, ERide::Fixed, FVector::ZeroVector, FVector(1300, 2600, 165), FVector(-400, 0, 1800), 80.0f, 1.0f },
	};
}

/**
 * Rubble.Video.GlassTower: the fracture lab's Generated · Glass tower in Unreal under the threshold
 * model with support loss: rigid until its bonds break. The 54 m tower's lower storeys are shot from
 * either side and it falls south onto the pavilion, filmed from several cameras (GlassTowerFilm):
 * the slow orbit, riding a floor and the roof of the falling tower, inside the pavilion it lands on,
 * and from the street. The world's GPU stepping is not bit-reproducible, so the orbit's pass
 * records every frame (FRubbleReplayBackend) and the other cameras replay that one collapse.
 * Every frame (fixed 1/30 s) to RubbleShots/video/glasstower/<camera>/ (scripts/unreal/video.sh
 * glasstower encodes them).
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleVideoGlassTower, "Rubble.Video.GlassTower",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleVideoGlassTower::RunTest(const FString& Parameters)
{
	using namespace GlassTowerFilm;
	FString Error;
	if (!TestTrue(TEXT("the engine's ABI"), Rubble() && FRubbleAbiBackend::CheckVersion(Error)))
	{
		return false;
	}
	TUniquePtr<FRubbleReplayBackend> Owned = MakeUnique<FRubbleReplayBackend>(MakeUnique<FRubbleAbiBackend>());
	FRubbleReplayBackend* Replay = Owned.Get();
	Rubble()->SetBackend(MoveTemp(Owned));
	if (!TestTrue(TEXT("the glass tower loads"), Load(TEXT("fracture:glasstower?autoshot=0"))))
	{
		Rubble()->SetBackend(MakeUnique<FRubbleAbiBackend>());
		return false;
	}
	Rubble()->bPaused = true;
	constexpr double Fps = 30.0;
	// -RubbleFilmSeconds=N cuts every camera to at most N seconds (a quick look at all of them).
	double Cap = 1.0e9;
	FParse::Value(FCommandLine::Get(), TEXT("RubbleFilmSeconds="), Cap);
	const FString Root = ShotDir() / TEXT("video") / TEXT("glasstower");
	IFileManager::Get().DeleteDirectory(*Root, false, true);
	struct FState
	{
		int32 Pass = 0;
		int32 Frame = -100; // -100: the pass starts
		double At = 0.0;
		int32 Fired = 0;
		/** Per camera, the chunk it rides (found at rest, before anything moves). */
		TArray<int32> Chunks;
		bool bAttached = false;
		FTransform EyeOnChunk;
		float Breaking = 0.0f;
		TWeakObjectPtr<ARubbleFilmRig> Rig;
	};
	TSharedRef<FState> S = MakeShared<FState>();
	const FRubbleAbiBackend* Abi = static_cast<const FRubbleAbiBackend*>(Replay->InnerBackend());
	for (const FView& V : Views)
	{
		S->Chunks.Add(V.Ride == ERide::Chunk ? Abi->NearestTownChunk(V.Chunk) : INDEX_NONE);
	}
	// The cameras move after the renderers have drawn the frame (a riding camera sees its chunk
	// where it is drawn).
	ARubbleFilmRig* Rig = GameWorld()->SpawnActor<ARubbleFilmRig>();
	S->Rig = Rig;
	Rig->Late = [S]() {
		const FView& V = Views[S->Pass];
		ARubblePawn* Pawn = Cast<ARubblePawn>(GameWorld()->GetFirstPlayerController()->GetPawn());
		if (!Pawn || S->Frame == -100)
		{
			return;
		}
		switch (V.Ride)
		{
		case ERide::Orbit:
		{
			const float u = float(S->At / V.Length);
			const float Ease = u * u * (3.0f - 2.0f * u);
			const float Angle = FMath::DegreesToRadians(FMath::Lerp(-38.0f, 12.0f, Ease));
			// The 54 m tower standing, then the wreck across the pavilion.
			const float Radius = FMath::Lerp(9500.0f, 7000.0f, Ease);
			const FVector Centre(0, 0, FMath::Lerp(2300.0f, 800.0f, Ease));
			const FVector Eye = Centre + FVector(Radius * FMath::Sin(Angle), -Radius * FMath::Cos(Angle), FMath::Lerp(1200.0f, 900.0f, Ease));
			Pawn->LookFrom(Eye, Centre);
			break;
		}
		case ERide::Chunk:
		{
			ARubblePrimitives* Drawn = nullptr;
			for (TActorIterator<ARubblePrimitives> It(GameWorld()); It; ++It)
			{
				Drawn = *It;
			}
			FTransform Chunk;
			if (!Drawn || !Drawn->ChunkTransform(S->Chunks[S->Pass], Chunk))
			{
				break;
			}
			Chunk.RemoveScaling();
			if (!S->bAttached)
			{
				S->EyeOnChunk = FTransform((V.Target - V.Eye).Rotation(), V.Eye).GetRelativeTransform(Chunk);
				S->bAttached = true;
			}
			const FTransform Eye = S->EyeOnChunk * Chunk;
			Pawn->ViewFrom(Eye.GetLocation(), Eye.Rotator(), V.Fov);
			break;
		}
		case ERide::Fixed:
		{
			// Shaken as things break: a step's splits kick it, and it settles over a third of a second.
			const float Kick = FMath::Min(1.0f, float(Rubble()->Stats().Events) / 8.0f);
			S->Breaking = FMath::Max(S->Breaking * FMath::Exp(-3.0f / float(Fps)), Kick);
			const float a = V.Shake * S->Breaking;
			const float t = float(S->At);
			const FRotator Shake(a * FMath::PerlinNoise1D(13.0f * t), a * FMath::PerlinNoise1D(11.0f * t + 31.0f), 0.5f * a * FMath::PerlinNoise1D(7.0f * t + 57.0f));
			Pawn->ViewFrom(V.Eye, (V.Target - V.Eye).Rotation() + Shake, V.Fov);
			break;
		}
		}
	};
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(4.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([S, Replay, Root, Fps, Cap]() {
		const FView& V = Views[S->Pass];
		const FString Dir = Root / V.Name;
		if (S->Frame == -100)
		{
			FApp::SetUseFixedTimeStep(true);
			FApp::SetFixedDeltaTime(1.0 / Fps);
			IFileManager::Get().MakeDirectory(*Dir, true);
			if (S->Pass > 0)
			{
				// Back to the start: the renderers rebuild from the recorded scene's assets.
				Replay->BeginReplay();
				Replay->SetHold(true);
				FRubbleSceneRequest R;
				FString Error;
				FRubbleSceneRequest::Parse(TEXT("fracture:glasstower?autoshot=0"), R, Error);
				Rubble()->LoadScene(R, Error);
				Rubble()->bPaused = true;
			}
			S->Fired = 0;
			S->bAttached = false;
			S->Breaking = 0.0f;
			// Frames -40..-1 settle the camera on its first pose and Lumen on the scene, unrecorded;
			// the world runs from -10.
			S->Frame = -40;
		}
		if (S->Frame == -10)
		{
			Rubble()->bPaused = false;
			if (S->Pass == 0)
			{
				Replay->BeginRecording();
			}
			else
			{
				Replay->SetHold(false);
			}
		}
		S->At = FMath::Max(0, S->Frame) / Fps;
		// A felling notch: the pavilion side's supports go and the north row (X -1524 cm) stands as
		// the hinge. Three volleys a second apart, each one ball (45 m/s) along each of the three
		// southern column rows (X -1174, -824, -474) at three heights in each of the lowest three
		// storeys, from 7 m off the west face (heading east, +Y) or the east face, alternating shot
		// by shot so the volley's push across the tower cancels. Support loss tips the floors above
		// over the north row onto the pavilion: in twelve jittered runs of engine-world's
		// glass_tower_felling_probe (plans "notch3", "notch3b") it falls toward the pavilion every
		// time, where volleys alternating whole (plan "both") scatter it. 81 shots on 58 parked balls.
		// Replays take no shots: their frames already hold them.
		static const double Volleys[] = { 1.0, 2.0, 3.0 };
		for (int32 c = 0; c < UE_ARRAY_COUNT(Volleys) && S->Pass == 0; ++c)
		{
			if (S->At >= Volleys[c] && !(S->Fired & (1 << c)))
			{
				S->Fired |= 1 << c;
				static const double Rows[] = { -1174.0, -824.0, -474.0 };
				static const double Heights[] = { 60.0, 150.0, 240.0 };
				for (int32 i = 0; i < UE_ARRAY_COUNT(Rows); ++i)
				{
					for (int32 Storey = 0; Storey < 3; ++Storey)
					{
						for (int32 j = 0; j < UE_ARRAY_COUNT(Heights); ++j)
						{
							const double Side = (c + i + Storey + j) % 2 ? 1.0 : -1.0;
							Rubble()->Launch(FVector(Rows[i], Side * (706.0 + 700.0), 300.0 * Storey + Heights[j]), FVector(0, -Side, 0));
						}
					}
				}
			}
		}
		if (S->Frame < 0)
		{
			++S->Frame;
			return false;
		}
		FScreenshotRequest::RequestScreenshot(Dir / FString::Printf(TEXT("frame_%04d.png"), S->Frame), /*bShowUI=*/false, false);
		if (++S->Frame < int32(FMath::Min(V.Length, Cap) * Fps))
		{
			return false;
		}
		UE_LOG(LogRubble, Display, TEXT("Rubble.Video.GlassTower: %s, %d frames in %s (%d polls recorded)"), V.Name, S->Frame, *Dir, Replay->RecordedPolls());
		if (S->Pass + 1 < UE_ARRAY_COUNT(Views))
		{
			++S->Pass;
			S->Frame = -100;
			return false;
		}
		FApp::SetUseFixedTimeStep(false);
		if (S->Rig.IsValid())
		{
			S->Rig->Destroy();
		}
		Rubble()->SetBackend(MakeUnique<FRubbleAbiBackend>());
		return true;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
namespace SoftCrashFilm
{
	/** A crash test filmed: its scene and where the cars meet (cm). */
	struct FCrash
	{
		const TCHAR* Name;
		const TCHAR* Scene;
		FVector Impact;
	};
	// softbody.html's head-on (two cars at 29 km/h from X ±800) and full-width wall (at 56 km/h
	// from X -1200 into the wall's face at X 220); both meet about 0.8 s in. The wall's cameras
	// aim at the car's front half, clear of the wall's face.
	static const FCrash Crashes[] = {
		{ TEXT("headon"), TEXT("soft:headon"), FVector(0, 0, 70) },
		{ TEXT("wall"), TEXT("soft:wall?speed=56"), FVector(80, 0, 70) },
	};
	enum class ERide : uint8
	{
		Fixed,  // standing still (offsets from the impact)
		Orbit,  // circling the impact
		Chase,  // behind the player car and above it, looking past it at the impact
	};
	struct FView
	{
		const TCHAR* Name;
		ERide Ride;
		FVector Eye; // from the impact
		FVector Target;
		float Fov;
	};
	static const FView Views[] = {
		// Low at the side, at bumper height: the fronts fold toward the lens.
		{ TEXT("side"), ERide::Fixed, FVector(-60, -780, 55), FVector(0, 0, 0), 48.0f },
		{ TEXT("closeup"), ERide::Fixed, FVector(-330, -300, 60), FVector(-60, 0, 0), 40.0f },
		{ TEXT("overhead"), ERide::Fixed, FVector(-40, -120, 880), FVector(0, 0, -70), 62.0f },
		{ TEXT("chase"), ERide::Chase, FVector(-720, -280, 240), FVector(250, 0, 40), 60.0f },
		{ TEXT("orbit"), ERide::Orbit, FVector::ZeroVector, FVector::ZeroVector, 55.0f },
	};
	/** Seconds of world time: real speed, then a fifth of it from just before the impact until the
	 *  wreck has settled, then real speed to the end. */
	constexpr double SlowFrom = 0.55, SlowTo = 1.75, SlowScale = 0.2, End = 3.5;
	double ScaleAt(double WorldT) { return WorldT >= SlowFrom && WorldT < SlowTo ? SlowScale : 1.0; }
}

/**
 * Rubble.Video.SoftCrash: the soft-body crash tests in Unreal, on the engine's car mesh (the
 * softbody page's: paint, glass, chrome, lamps) skinned part by part, with the damage it measures
 * (crumpled and scraped paint, cracked and shattered glass, torn panels) and the debris it sheds.
 * Each crash runs once, slowed to a fifth around the impact, and every camera of
 * SoftCrashFilm::Views films it together (FRubbleFilmCapture: a scene capture per camera, read
 * back asynchronously into ffmpeg's hardware encoder) to RubbleShots/video/softcrash/
 * <crash>-<camera>.mp4. scripts/unreal/video.sh softcrash adds the multicam.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleVideoSoftCrash, "Rubble.Video.SoftCrash",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleVideoSoftCrash::RunTest(const FString& Parameters)
{
	using namespace SoftCrashFilm;
	FString Error;
	if (!TestTrue(TEXT("the engine's ABI"), Rubble() && FRubbleAbiBackend::CheckVersion(Error)))
	{
		return false;
	}
	Rubble()->SetBackend(MakeUnique<FRubbleAbiBackend>());
	constexpr int32 Fps = 30;
	// -RubbleFilmCrash=wall films one crash; -RubbleFilmViews=N only the first N cameras;
	// -RubbleFilmSize=WxH (1280x720).
	FString Only;
	FParse::Value(FCommandLine::Get(), TEXT("RubbleFilmCrash="), Only);
	int32 NumViews = UE_ARRAY_COUNT(Views);
	FParse::Value(FCommandLine::Get(), TEXT("RubbleFilmViews="), NumViews);
	NumViews = FMath::Clamp(NumViews, 1, int32(UE_ARRAY_COUNT(Views)));
	FIntPoint Size(1280, 720);
	FString SizeText;
	if (FParse::Value(FCommandLine::Get(), TEXT("RubbleFilmSize="), SizeText))
	{
		FString W, H;
		if (SizeText.Split(TEXT("x"), &W, &H))
		{
			Size = FIntPoint(FCString::Atoi(*W), FCString::Atoi(*H));
		}
	}
	TArray<int32> Films;
	for (int32 c = 0; c < UE_ARRAY_COUNT(Crashes); ++c)
	{
		if (Only.IsEmpty() || Only == Crashes[c].Name)
		{
			Films.Add(c);
		}
	}
	const FString Root = ShotDir() / TEXT("video") / TEXT("softcrash");
	IFileManager::Get().DeleteDirectory(*Root, false, true);
	IFileManager::Get().MakeDirectory(*Root, true);
	struct FState
	{
		int32 Film = 0;
		int32 Frame = -100; // -100: the crash starts
		double WorldT = 0.0;
		double Started = 0.0;
		TArray<FTransform> Rides; // per camera: a chase camera's smoothed frame
		TArray<bool> Riding;
		TArray<TUniquePtr<FRubbleFilmCapture>> Cameras;
		TWeakObjectPtr<ARubbleFilmRig> Rig;
		bool bDone = false;
	};
	TSharedRef<FState> S = MakeShared<FState>();
	ARubbleFilmRig* Rig = GameWorld()->SpawnActor<ARubbleFilmRig>();
	S->Rig = Rig;
	if (!Rig->GetRootComponent())
	{
		USceneComponent* RootComp = NewObject<USceneComponent>(Rig);
		Rig->SetRootComponent(RootComp);
		RootComp->RegisterComponent();
	}
	// The viewport draws nothing while the cameras film (they are the only renders).
	if (GEngine && GEngine->GameViewport)
	{
		GEngine->GameViewport->bDisableWorldRendering = true;
	}
	// Where camera v looks this frame.
	auto Pose = [S, Films](int32 v, FVector& Eye, FRotator& View, float& Fov) {
		const FCrash& C = Crashes[Films[S->Film]];
		const FView& V = Views[v];
		Fov = V.Fov;
		switch (V.Ride)
		{
		case ERide::Fixed:
			Eye = C.Impact + V.Eye;
			View = (V.Target - V.Eye).Rotation();
			return true;
		case ERide::Orbit:
		{
			// A quarter turn round the impact from the near side, closing in as it slows.
			const float u = float(FMath::Clamp(S->WorldT / End, 0.0, 1.0));
			const float Angle = FMath::DegreesToRadians(FMath::Lerp(-70.0f, 20.0f, u));
			const float Radius = FMath::Lerp(1100.0f, 750.0f, u);
			Eye = C.Impact + FVector(Radius * FMath::Sin(Angle), -Radius * FMath::Cos(Angle), 260.0f);
			View = (C.Impact - Eye).Rotation();
			return true;
		}
		case ERide::Chase:
		{
			FTransform Car;
			for (TActorIterator<ARubbleVehicles> It(GameWorld()); It; ++It)
			{
				It->PlayerTransform(Car);
			}
			if (Car.GetLocation().IsNearlyZero())
			{
				return false; // no wheels drawn yet
			}
			// Its heading taken once, from the car toward the impact (the core's own frame comes
			// from the wheel hubs, which jolt and tip in the crash); then it follows the car,
			// smoothed.
			if (!S->Riding[v])
			{
				const FVector Ahead = (C.Impact - Car.GetLocation()) * FVector(1, 1, 0);
				S->Rides[v] = FTransform(Ahead.Rotation(), Car.GetLocation());
				S->Riding[v] = true;
			}
			S->Rides[v].SetLocation(FMath::Lerp(S->Rides[v].GetLocation(), Car.GetLocation(), 0.3));
			Eye = S->Rides[v].TransformPosition(V.Eye);
			View = (S->Rides[v].TransformPosition(V.Target) - Eye).Rotation();
			return true;
		}
		}
		return false;
	};
	// The cameras aim and shoot after the renderers have drawn the frame (a chase camera sees
	// the car where it is drawn).
	Rig->Late = [S, Pose]() {
		if (S->Frame == -100 || S->bDone)
		{
			return;
		}
		for (int32 v = 0; v < S->Cameras.Num(); ++v)
		{
			FVector Eye;
			FRotator View;
			float Fov = 60.0f;
			if (Pose(v, Eye, View, Fov))
			{
				S->Cameras[v]->Aim(Eye, View, Fov);
			}
			// Settling frames render too (each camera's temporal history and Lumen), but only
			// frame 0 on goes into the video.
			if (S->Frame >= 0)
			{
				S->Cameras[v]->Shoot();
			}
			else
			{
				S->Cameras[v]->Warm();
			}
		}
	};
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(1.0f));
	ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([this, S, Root, Fps, Films, NumViews, Size]() {
		if (!Films.IsValidIndex(S->Film))
		{
			FApp::SetUseFixedTimeStep(false);
			Rubble()->TimeScale = 1.0;
			if (GEngine && GEngine->GameViewport)
			{
				GEngine->GameViewport->bDisableWorldRendering = false;
			}
			if (S->Rig.IsValid())
			{
				S->Rig->Destroy();
			}
			return true;
		}
		const FCrash& C = Crashes[Films[S->Film]];
		if (S->Frame == -100)
		{
			FApp::SetUseFixedTimeStep(true);
			FApp::SetFixedDeltaTime(1.0 / Fps);
			FString LoadError;
			FRubbleSceneRequest R;
			FRubbleSceneRequest::Parse(C.Scene, R, LoadError);
			if (!Rubble()->LoadScene(R, LoadError))
			{
				AddError(FString::Printf(TEXT("%s: %s"), C.Scene, *LoadError));
				S->Film = Films.Num();
				return false;
			}
			Rubble()->bPaused = true;
			Rubble()->TimeScale = 1.0;
			S->Cameras.Reset();
			S->Rides.Init(FTransform::Identity, NumViews);
			S->Riding.Init(false, NumViews);
			for (int32 v = 0; v < NumViews; ++v)
			{
				const FString Path = FPaths::ConvertRelativePathToFull(Root / FString::Printf(TEXT("%s-%s.mp4"), C.Name, Views[v].Name));
				S->Cameras.Add(MakeUnique<FRubbleFilmCapture>(S->Rig.Get(), Path, Size.X, Size.Y, Fps));
				if (!S->Cameras.Last()->IsOpen())
				{
					AddError(FString::Printf(TEXT("ffmpeg for %s"), *Path));
				}
			}
			S->WorldT = 0.0;
			S->Started = FPlatformTime::Seconds();
			// Frames -40..-1 settle the cameras and Lumen; the world runs from -10.
			S->Frame = -40;
			S->bDone = false;
		}
		if (S->Frame == -10)
		{
			Rubble()->bPaused = false;
		}
		if (S->Frame < 0)
		{
			++S->Frame;
			return false;
		}
		// Real speed, then a fifth of it around the impact.
		Rubble()->TimeScale = ScaleAt(S->WorldT);
		S->WorldT += Rubble()->TimeScale / Fps;
		++S->Frame;
		if (S->WorldT < End)
		{
			return false;
		}
		// This frame's shots were taken by the rig's late hook; close the videos.
		S->bDone = true;
		int32 Written = 0;
		for (TUniquePtr<FRubbleFilmCapture>& Camera : S->Cameras)
		{
			Camera->Finish();
			Written = FMath::Max(Written, Camera->FramesWritten());
		}
		UE_LOG(LogRubble, Display, TEXT("Rubble.Video.SoftCrash: %s, %d cameras x %d frames in %.1f s"),
			*FString(C.Name), S->Cameras.Num(), Written, FPlatformTime::Seconds() - S->Started);
		S->Cameras.Reset();
		S->Frame = -100;
		++S->Film;
		return false;
	}));
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.5f));
	return true;
}
#endif

#if WITH_DEV_AUTOMATION_TESTS
/**
 * Rubble.Render.CarDesign: the soft car parked in the yard, shot from eight angles for a design
 * review (RubbleShots/design/<view>.png): the three-quarter views, the side, front and rear, from
 * above, the windscreen and a wheel up close. -RubbleCarScene=<scene> picks another scene.
 */
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleRenderCarDesign, "Rubble.Render.CarDesign",
	EAutomationTestFlags::ClientContext | EAutomationTestFlags::ProductFilter)

bool FRubbleRenderCarDesign::RunTest(const FString& Parameters)
{
	FString Scene = TEXT("soft:yard");
	FParse::Value(FCommandLine::Get(), TEXT("RubbleCarScene="), Scene);
	if (!TestTrue(*FString::Printf(TEXT("%s loads"), *Scene), Rubble() && Load(*Scene)))
	{
		return false;
	}
	struct FView
	{
		const TCHAR* Name;
		FVector Eye;
		FVector Target;
		float Fov;
	};
	// The car stands at the origin facing +X; its body is about 4 m long and 1.4 m tall.
	static const FView Views[] = {
		{ TEXT("front34"), FVector(430, -340, 150), FVector(0, 0, 70), 45.0f },
		{ TEXT("side"), FVector(0, -640, 100), FVector(0, 0, 70), 45.0f },
		{ TEXT("rear34"), FVector(-430, -340, 160), FVector(0, 0, 70), 45.0f },
		{ TEXT("front"), FVector(680, 0, 110), FVector(0, 0, 70), 40.0f },
		{ TEXT("rear"), FVector(-680, 0, 110), FVector(0, 0, 70), 40.0f },
		{ TEXT("top"), FVector(20, -40, 760), FVector(0, 0, 0), 45.0f },
		{ TEXT("windscreen"), FVector(260, -160, 190), FVector(20, 0, 115), 45.0f },
		{ TEXT("wheel"), FVector(250, -250, 60), FVector(120, -80, 35), 45.0f },
	};
	const FString Dir = ShotDir() / TEXT("design");
	IFileManager::Get().MakeDirectory(*Dir, true);
	ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(3.0f)); // parked and settled
	for (const FView& V : Views)
	{
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([V]() {
			if (ARubblePawn* Pawn = Cast<ARubblePawn>(GameWorld()->GetFirstPlayerController()->GetPawn()))
			{
				Pawn->ViewFrom(V.Eye, (V.Target - V.Eye).Rotation(), V.Fov);
			}
			return true;
		}));
		ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.8f)); // Lumen and TAA settle
		const FString Path = Dir / (FString(V.Name) + TEXT(".png"));
		ADD_LATENT_AUTOMATION_COMMAND(FFunctionLatentCommand([Path]() {
			FScreenshotRequest::RequestScreenshot(Path, /*bShowUI=*/false, /*bAddFilenameSuffix=*/false);
			return true;
		}));
		ADD_LATENT_AUTOMATION_COMMAND(FEngineWaitLatentCommand(0.3f));
	}
	return true;
}
#endif
