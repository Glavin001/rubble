// Advances the Rubble world once a frame, in TG_PrePhysics after the pawn (whose input it steps
// with), before the renderers (TG_PostPhysics) draw the result.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RubbleDirector.generated.h"

UCLASS()
class RUBBLERUNTIME_API ARubbleDirector : public AActor
{
	GENERATED_BODY()

public:
	ARubbleDirector();
	virtual void Tick(float DeltaSeconds) override;
};
