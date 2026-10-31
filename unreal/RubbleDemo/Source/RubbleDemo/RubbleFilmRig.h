// A scripted film's late hook: it ticks after every renderer (TG_PostUpdateWork), before the
// cameras update, so a camera riding a falling chunk reads the pose drawn this frame, not last.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleFilmRig.generated.h"

UCLASS(NotPlaceable, Transient)
class RUBBLEDEMO_API ARubbleFilmRig : public AActor
{
	GENERATED_BODY()

public:
	ARubbleFilmRig()
	{
		PrimaryActorTick.bCanEverTick = true;
		PrimaryActorTick.TickGroup = TG_PostUpdateWork;
	}

	TFunction<void()> Late;

	virtual void Tick(float DeltaSeconds) override
	{
		Super::Tick(DeltaSeconds);
		if (Late)
		{
			Late();
		}
	}
};
