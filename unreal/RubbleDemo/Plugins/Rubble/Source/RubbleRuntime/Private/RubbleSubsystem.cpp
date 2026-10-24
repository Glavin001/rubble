#include "RubbleSubsystem.h"

#include "HAL/PlatformTime.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "RubbleAbiBackend.h"
#include "RubbleGpuPoses.h"
#include "RubbleMockBackend.h"

DEFINE_LOG_CATEGORY(LogRubble);

void URubbleSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	// The real engine through its C ABI by default; -RubbleBackend=mock for the stand-in alone.
	FString Which = TEXT("rubble");
	FParse::Value(FCommandLine::Get(), TEXT("RubbleBackend="), Which);
	FString Error;
	if (Which.Equals(TEXT("mock"), ESearchCase::IgnoreCase))
	{
		SetBackend(MakeUnique<FRubbleMockBackend>());
	}
	else if (!FRubbleAbiBackend::CheckVersion(Error))
	{
		UE_LOG(LogRubble, Error, TEXT("%s; using the mock"), *Error);
		SetBackend(MakeUnique<FRubbleMockBackend>());
	}
	else
	{
		SetBackend(MakeUnique<FRubbleAbiBackend>());
	}
}

void URubbleSubsystem::Deinitialize()
{
	OnFrame.Clear();
	OnSceneLoaded.Clear();
	BackendImpl.Reset();
	Super::Deinitialize();
}

void URubbleSubsystem::SetBackend(TUniquePtr<IRubbleBackend> InBackend)
{
	BackendImpl = MoveTemp(InBackend);
	Demos = BackendImpl ? BackendImpl->ListDemos() : TArray<FRubbleDemoDesc>();
	bHasScene = false;
	UE_LOG(LogRubble, Display, TEXT("backend: %s, %d scenes"), BackendImpl ? BackendImpl->Name() : TEXT("none"), Demos.Num());
}

TArray<FRubbleDemoDesc> URubbleSubsystem::ListDemos() const
{
	return Demos;
}

const FRubbleDemoDesc* URubbleSubsystem::CurrentDemo() const
{
	return Demos.FindByPredicate([this](const FRubbleDemoDesc& D) { return D.Family == Current.Family && D.Id == Current.Id; });
}

bool URubbleSubsystem::LoadScene(const FRubbleSceneRequest& Request, FString& Error)
{
	if (!BackendImpl)
	{
		Error = TEXT("no backend");
		return false;
	}
	const double Start = FPlatformTime::Seconds();
	if (!BackendImpl->Load(Request, Error))
	{
		UE_LOG(LogRubble, Error, TEXT("load %s: %s"), *Request.ToString(), *Error);
		return false;
	}
	Current = Request;
	bHasScene = true;
	// Soft scenes step at 240 Hz: up to 8 steps a frame (two 60 Hz frames' worth), 4 elsewhere.
	const double StepSeconds = BackendImpl->StepSeconds();
	Clock.Reset(StepSeconds, StepSeconds < 1.0 / 120.0 ? 8 : 4);
	const FRubbleSceneAssets& A = BackendImpl->Assets();
	UE_LOG(LogRubble, Display, TEXT("loaded %s: %d bodies, %d shapes, %d meshes, %d chunks, %d vehicles, %d soft bodies (%.0f ms)"),
		*Request.ToString(), A.Bodies.Num(), A.Shapes.Num(), A.Meshes.Num(), A.Chunks.Num(), A.Vehicles.Num(), A.SoftBodies.Num(),
		1000.0 * (FPlatformTime::Seconds() - Start));
	OnSceneLoaded.Broadcast(A);
	return true;
}

void URubbleSubsystem::Advance(double RealSeconds)
{
	if (!BackendImpl || !bHasScene)
	{
		return;
	}
	FRubbleFrameStats Stats;
	const double Start = FPlatformTime::Seconds();
	const double DroppedBefore = Clock.Dropped;
	const int32 Steps = bPaused ? 0 : Clock.Advance(RealSeconds * FMath::Max(0.0, TimeScale));
	for (int32 i = 0; i < Steps; ++i)
	{
		BackendImpl->Step();
	}
	TArray<FRubbleFrame> Frames;
	BackendImpl->Poll(Frames);
	if (FRubbleGpuPoses* Gpu = BackendImpl->GpuPoses())
	{
		Gpu->Update(float(Clock.Alpha()));
	}
	Stats.BackendMs = 1000.0 * (FPlatformTime::Seconds() - Start);
	Stats.Steps = Steps;
	Stats.DroppedSeconds = Clock.Dropped - DroppedBefore;
	if (Frames.Num())
	{
		FRubbleFrame Merged;
		FRubbleFrameMerger::Merge(MoveTemp(Frames), Merged);
		Stats.Moves = Merged.Moves.Num();
		Stats.Events = Merged.Events.Num();
		Stats.Skins = Merged.Skins.Num();
		Stats.Step = Merged.Step;
		if (Merged.LostEvents)
		{
			UE_LOG(LogRubble, Warning, TEXT("step %llu: %u events lost in the ring"), Merged.Step, Merged.LostEvents);
		}
		const double BroadcastStart = FPlatformTime::Seconds();
		OnFrame.Broadcast(Merged);
		Stats.BroadcastMs = 1000.0 * (FPlatformTime::Seconds() - BroadcastStart);
	}
	else
	{
		Stats.Step = LastStats.Step;
	}
	LastStats = Stats;
}

void URubbleSubsystem::SetVehicleInput(const FRubbleVehicleInput& Input)
{
	if (BackendImpl && bHasScene)
	{
		BackendImpl->SetVehicleInput(Input);
	}
}

void URubbleSubsystem::Launch(const FVector& Origin, const FVector& Direction)
{
	if (BackendImpl && bHasScene)
	{
		BackendImpl->Launch(Origin, Direction);
	}
}

void URubbleSubsystem::Meteor(const FVector& Target)
{
	if (BackendImpl && bHasScene)
	{
		BackendImpl->Meteor(Target);
	}
}

void URubbleSubsystem::SideRam()
{
	if (BackendImpl && bHasScene)
	{
		BackendImpl->SideRam();
	}
}

void URubbleSubsystem::Reset()
{
	if (BackendImpl && bHasScene)
	{
		FString Error;
		LoadScene(Current, Error);
	}
}
