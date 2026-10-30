#include "RubbleReplayBackend.h"

void FRubbleReplayBackend::BeginRecording()
{
	Recording.Reset();
	bRecording = true;
	bReplaying = false;
}

void FRubbleReplayBackend::BeginReplay()
{
	bRecording = false;
	bReplaying = true;
	Next = 0;
}

bool FRubbleReplayBackend::Load(const FRubbleSceneRequest& Request, FString& Error)
{
	if (bReplaying)
	{
		Next = 0;
		return true;
	}
	return Inner->Load(Request, Error);
}

void FRubbleReplayBackend::Step()
{
	if (!bReplaying)
	{
		Inner->Step();
	}
}

void FRubbleReplayBackend::Poll(TArray<FRubbleFrame>& Out)
{
	if (bReplaying)
	{
		if (bHold)
		{
			return;
		}
		// Each Poll gets what the recording's Poll at the same place got; past its end, nothing.
		if (Recording.IsValidIndex(Next))
		{
			Out.Append(Recording[Next]);
		}
		++Next;
		return;
	}
	const int32 Before = Out.Num();
	Inner->Poll(Out);
	if (bRecording)
	{
		Recording.Emplace(Out.GetData() + Before, Out.Num() - Before);
	}
}

void FRubbleReplayBackend::SetVehicleInput(const FRubbleVehicleInput& Input)
{
	if (!bReplaying)
	{
		Inner->SetVehicleInput(Input);
	}
}

void FRubbleReplayBackend::Launch(const FVector& Origin, const FVector& Direction)
{
	if (!bReplaying)
	{
		Inner->Launch(Origin, Direction);
	}
}

void FRubbleReplayBackend::SideRam()
{
	if (!bReplaying)
	{
		Inner->SideRam();
	}
}

void FRubbleReplayBackend::Meteor(const FVector& Target)
{
	if (!bReplaying)
	{
		Inner->Meteor(Target);
	}
}

void FRubbleReplayBackend::Reset()
{
	if (!bReplaying)
	{
		Inner->Reset();
	}
}
