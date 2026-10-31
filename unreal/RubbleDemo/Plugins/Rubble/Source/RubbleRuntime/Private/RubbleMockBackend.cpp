#include "RubbleMockBackend.h"

#include "Mock/MockScene.h"

FRubbleMockBackend::FRubbleMockBackend() : SceneAssets(MakeUnique<FRubbleSceneAssets>()) {}
FRubbleMockBackend::~FRubbleMockBackend() = default;

TArray<FRubbleDemoDesc> FRubbleMockBackend::ListDemos() const
{
	TArray<FRubbleDemoDesc> All = RubbleMock::RigidCatalog();
	All.Append(RubbleMock::VehicleCatalog());
	All.Append(RubbleMock::SoftCatalog());
	All.Append(RubbleMock::FractureCatalog());
	return All;
}

bool FRubbleMockBackend::Load(const FRubbleSceneRequest& Request, FString& Error)
{
	const FRubbleDemoDesc* Desc = nullptr;
	const TArray<FRubbleDemoDesc> Demos = ListDemos();
	for (const FRubbleDemoDesc& D : Demos)
	{
		if (D.Family == Request.Family && D.Id == Request.Id)
		{
			Desc = &D;
			break;
		}
	}
	if (!Desc)
	{
		Error = FString::Printf(TEXT("no scene %s"), *Request.ToString());
		return false;
	}
	TUniquePtr<RubbleMock::FScene> Next;
	switch (Request.Family)
	{
	case ERubbleFamily::Rigid: Next = RubbleMock::MakeRigidScene(); break;
	case ERubbleFamily::Vehicles: Next = RubbleMock::MakeVehicleScene(); break;
	case ERubbleFamily::Soft: Next = RubbleMock::MakeSoftScene(); break;
	case ERubbleFamily::Fracture: Next = RubbleMock::MakeFractureScene(); break;
	}
	// Parameters the scene does not declare are refused, never ignored.
	for (const TPair<FString, float>& P : Request.Params)
	{
		if (!Desc->Params.ContainsByPredicate([&P](const FRubbleParamDesc& D) { return D.Key == P.Key; }))
		{
			Error = FString::Printf(TEXT("%s has no parameter '%s'"), *Request.Id, *P.Key);
			return false;
		}
	}
	FRubbleSceneRequest WithDefaults = Request;
	for (const FRubbleParamDesc& P : Desc->Params)
	{
		if (!WithDefaults.Params.Contains(P.Key))
		{
			WithDefaults.Params.Add(P.Key, P.Default);
		}
	}
	TUniquePtr<FRubbleSceneAssets> NextAssets = MakeUnique<FRubbleSceneAssets>();
	if (!Next->Build(WithDefaults, *NextAssets, Error))
	{
		return false;
	}
	Scene = MoveTemp(Next);
	SceneAssets = MoveTemp(NextAssets);
	Current = Request;
	Dt = 1.0 / Desc->StepHz;
	StepCount = 0;
	InFlight.Reset();
	return true;
}

void FRubbleMockBackend::Step()
{
	if (!Scene)
	{
		return;
	}
	FRubbleFrame& F = InFlight.AddDefaulted_GetRef();
	F.Step = StepCount++;
	Scene->Step(float(Dt), F);
	F.bRan = F.Moves.Num() > 0 || F.Events.Num() > 0 || F.Skins.Num() > 0 || F.Wheels.Num() > 0;
}

void FRubbleMockBackend::Poll(TArray<FRubbleFrame>& Out)
{
	const int32 Ready = InFlight.Num() - LatencySteps;
	if (Ready <= 0)
	{
		return;
	}
	for (int32 i = 0; i < Ready; ++i)
	{
		Out.Add(MoveTemp(InFlight[i]));
	}
	InFlight.RemoveAt(0, Ready, EAllowShrinking::No);
}

void FRubbleMockBackend::SetVehicleInput(const FRubbleVehicleInput& Input)
{
	if (Scene)
	{
		Scene->SetVehicleInput(Input);
	}
}

void FRubbleMockBackend::Launch(const FVector& Origin, const FVector& Direction)
{
	if (Scene)
	{
		Scene->Launch(Origin, Direction);
	}
}

void FRubbleMockBackend::SideRam()
{
	if (Scene)
	{
		Scene->SideRam();
	}
}

void FRubbleMockBackend::Reset()
{
	FString Error;
	const FRubbleSceneRequest Again = Current;
	Load(Again, Error);
}
