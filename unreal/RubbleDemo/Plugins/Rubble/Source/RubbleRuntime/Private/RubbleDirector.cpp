#include "RubbleDirector.h"

#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "RubbleSubsystem.h"

ARubbleDirector::ARubbleDirector()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PrePhysics;
}

void ARubbleDirector::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (UGameInstance* GI = GetGameInstance())
	{
		if (URubbleSubsystem* Rubble = GI->GetSubsystem<URubbleSubsystem>())
		{
			// Real time, not dilated: the world's own TimeScale is the slow motion.
			Rubble->Advance(GetWorld()->GetDeltaSeconds() / FMath::Max(1e-6f, GetActorTimeDilation()));
		}
	}
}
