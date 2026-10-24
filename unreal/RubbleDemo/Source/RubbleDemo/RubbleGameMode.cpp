#include "RubbleGameMode.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "RubbleDirector.h"
#include "RubbleEffects.h"
#include "RubbleEnvironment.h"
#include "RubbleFracture.h"
#include "RubblePawn.h"
#include "RubblePrimitives.h"
#include "RubbleSoftBodies.h"
#include "RubbleSubsystem.h"
#include "RubbleVehicles.h"

ARubbleGameMode::ARubbleGameMode()
{
	DefaultPawnClass = ARubblePawn::StaticClass();
}

void ARubbleGameMode::StartPlay()
{
	Super::StartPlay();
	UWorld* World = GetWorld();
	World->SpawnActor<ARubbleEnvironment>();
	ARubbleDirector* Director = World->SpawnActor<ARubbleDirector>();
	// The director steps after the pawn has set this frame's input.
	for (TActorIterator<ARubblePawn> It(World); It; ++It)
	{
		Director->AddTickPrerequisiteActor(*It);
	}
	World->SpawnActor<ARubblePrimitives>();
	World->SpawnActor<ARubbleFracture>();
	World->SpawnActor<ARubbleVehicles>();
	World->SpawnActor<ARubbleSoftBodies>();
	World->SpawnActor<ARubbleEffects>();

	FString Scene = DefaultScene;
	FParse::Value(FCommandLine::Get(), TEXT("RubbleDemo="), Scene, /*bShouldStopOnSeparator=*/false);
	FRubbleSceneRequest Request;
	FString Error;
	URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>();
	if (!FRubbleSceneRequest::Parse(Scene, Request, Error) || !Rubble->LoadScene(Request, Error))
	{
		UE_LOG(LogRubble, Error, TEXT("-RubbleDemo=%s: %s; loading %s"), *Scene, *Error, DefaultScene);
		FRubbleSceneRequest::Parse(DefaultScene, Request, Error);
		Rubble->LoadScene(Request, Error);
	}
}
