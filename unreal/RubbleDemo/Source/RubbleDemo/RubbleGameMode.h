// Spawns the Rubble world's actors (environment, director, renderers, effects) into the empty
// entry map and loads the first scene: -RubbleDemo=family:scene?key=value, else the house.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "RubbleGameMode.generated.h"

UCLASS()
class RUBBLEDEMO_API ARubbleGameMode : public AGameModeBase
{
	GENERATED_BODY()

public:
	ARubbleGameMode();
	virtual void StartPlay() override;

	/** The scene shown when the command line names none. */
	static constexpr const TCHAR* DefaultScene = TEXT("fracture:house");
};
