#include "RubbleVehicles.h"

#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/GameInstance.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

namespace
{
	uint64 WheelKey(bool bSoft, int32 Vehicle, int32 Wheel)
	{
		return (uint64(bSoft) << 63) | (uint64(uint32(Vehicle)) << 32) | uint32(Wheel);
	}

	FRubbleWheelVisual LerpWheel(const FRubbleWheelVisual& A, const FRubbleWheelVisual& B, double Alpha)
	{
		FRubbleWheelVisual W = B;
		W.Hub = RubbleLerpPose(A.Hub, B.Hub, Alpha);
		W.SpinRad = A.SpinRad + float(Alpha) * FMath::FindDeltaAngleRadians(A.SpinRad, B.SpinRad);
		W.SteerRad = FMath::Lerp(A.SteerRad, B.SteerRad, float(Alpha));
		return W;
	}
}

ARubbleVehicles::ARubbleVehicles()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PostPhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ARubbleVehicles::BeginPlay()
{
	Super::BeginPlay();
	if (URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>())
	{
		FrameHandle = Rubble->OnFrame.AddUObject(this, &ARubbleVehicles::OnFrame);
		SceneHandle = Rubble->OnSceneLoaded.AddUObject(this, &ARubbleVehicles::Rebuild);
		if (const FRubbleSceneAssets* A = Rubble->Assets())
		{
			Rebuild(*A);
		}
	}
}

void ARubbleVehicles::EndPlay(const EEndPlayReason::Type Reason)
{
	if (UGameInstance* GI = GetGameInstance())
	{
		if (URubbleSubsystem* Rubble = GI->GetSubsystem<URubbleSubsystem>())
		{
			Rubble->OnFrame.Remove(FrameHandle);
			Rubble->OnSceneLoaded.Remove(SceneHandle);
		}
	}
	Super::EndPlay(Reason);
}

UStaticMeshComponent* ARubbleVehicles::Part(USceneComponent* Parent, UStaticMesh* Mesh, const FTransform& Local, UMaterialInterface* Material)
{
	UStaticMeshComponent* C = NewObject<UStaticMeshComponent>(this);
	C->SetStaticMesh(Mesh);
	C->SetMobility(EComponentMobility::Movable);
	C->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	C->SetMaterial(0, Material);
	C->SetupAttachment(Parent);
	C->SetRelativeTransform(Local);
	C->RegisterComponent();
	Owned.Add(C);
	return C;
}

void ARubbleVehicles::BuildBody(USceneComponent* Root, const FRubbleVehicleVisual& V)
{
	// Placeholder bodywork from boxes: a lower body, a cabin, bumpers; trucks, tanks and bikes differ.
	UStaticMesh* Cube = RubbleRender::EngineCube();
	UStaticMesh* Cylinder = RubbleRender::EngineCylinder();
	UMaterialInterface* Paint = RubbleRender::CarPaint(this, V.Paint);
	UMaterialInterface* Glass = RubbleRender::Surface(ERubbleMaterial::Glass);
	UMaterialInterface* Trim = RubbleRender::Surface(ERubbleMaterial::Rubber);
	UMaterialInterface* Steel = RubbleRender::Surface(ERubbleMaterial::Steel);
	if (V.Parts.Num())
	{
		// The descriptor's own bodywork: each collider box drawn in paint (rubber for tyres).
		for (const FRubbleShapeVisual& S : V.Parts)
		{
			UMaterialInterface* M = S.Material == ERubbleMaterial::CarPaint ? Paint : RubbleRender::Surface(S.Material);
			if (S.Kind == ERubbleShape::Box)
			{
				Part(Root, Cube, FTransform(S.Local.GetRotation(), S.Local.GetLocation(), FVector(S.HalfExtents) / 50.0), M);
			}
			else if (S.Kind == ERubbleShape::Sphere)
			{
				Part(Root, RubbleRender::EngineSphere(), FTransform(S.Local.GetRotation(), S.Local.GetLocation(), FVector(S.Radius / 50.0)), M);
			}
		}
		return;
	}
	const FVector H(V.ChassisHalfExtents);
	auto Box = [&](const FVector& Center, const FVector& Half, UMaterialInterface* M, const FRotator& R = FRotator::ZeroRotator) {
		Part(Root, Cube, FTransform(R, Center, Half / 50.0), M);
	};
	switch (V.Style)
	{
	case ERubbleVehicleStyle::Tank:
		Box(FVector(0, 0, 0), H, Paint);
		Box(FVector(-20, 0, H.Z + 30), FVector(H.X * 0.45, H.Y * 0.6, 30), Paint);
		Part(Root, Cylinder, FTransform(FRotator(-90, 0, 0), FVector(H.X * 0.75, 0, H.Z + 35), FVector(0.18, 0.18, 3.2)), Steel);
		Box(FVector(0, -H.Y + 10, -H.Z * 0.6), FVector(H.X * 1.02, 22, H.Z * 0.5), Trim);
		Box(FVector(0, H.Y - 10, -H.Z * 0.6), FVector(H.X * 1.02, 22, H.Z * 0.5), Trim);
		break;
	case ERubbleVehicleStyle::Motorcycle:
		Box(FVector(0, 0, 0), FVector(H.X * 0.7, H.Y, H.Z * 0.5), Paint);
		Box(FVector(-20, 0, H.Z * 0.6), FVector(35, 14, 6), Trim);
		Part(Root, Cylinder, FTransform(FRotator(0, 0, 90), FVector(H.X * 0.6, 0, H.Z * 0.9), FVector(0.05, 0.05, 0.7)), Steel);
		break;
	case ERubbleVehicleStyle::SixWheelTruck:
		Box(FVector(H.X * 0.62, 0, H.Z * 0.3), FVector(H.X * 0.36, H.Y, H.Z * 0.95), Paint);
		Box(FVector(H.X * 0.85, 0, H.Z * 0.75), FVector(4, H.Y * 0.85, H.Z * 0.35), Glass);
		Box(FVector(-H.X * 0.38, 0, H.Z * 0.1), FVector(H.X * 0.6, H.Y * 0.98, H.Z * 0.75), RubbleRender::Surface(ERubbleMaterial::Steel));
		break;
	case ERubbleVehicleStyle::Kart:
		Box(FVector(0, 0, 0), H, Paint);
		Box(FVector(-H.X * 0.3, 0, H.Z + 15), FVector(18, 22, 15), Trim);
		break;
	default:
	{
		// Cars: lower body, cabin with glass sides, bumpers.
		const bool bLow = V.Style == ERubbleVehicleStyle::Sports;
		const bool bBig = V.Style == ERubbleVehicleStyle::MonsterTruck || V.Style == ERubbleVehicleStyle::DuneBuggy;
		Box(FVector(0, 0, -H.Z * 0.25), FVector(H.X, H.Y, H.Z * 0.75), Paint);
		const float CabZ = H.Z * (bLow ? 0.75f : 0.95f);
		Box(FVector(-H.X * 0.1, 0, CabZ), FVector(H.X * (bLow ? 0.42f : 0.5f), H.Y * 0.86f, H.Z * (bLow ? 0.45f : 0.55f)), Paint);
		Box(FVector(-H.X * 0.1, 0, CabZ), FVector(H.X * (bLow ? 0.4f : 0.48f), H.Y * 0.88f, H.Z * (bLow ? 0.38f : 0.46f)), Glass);
		Box(FVector(H.X + 4, 0, -H.Z * 0.6), FVector(6, H.Y * 0.95f, H.Z * 0.22f), Trim);
		Box(FVector(-H.X - 4, 0, -H.Z * 0.6), FVector(6, H.Y * 0.95f, H.Z * 0.22f), Trim);
		if (bBig)
		{
			Box(FVector(0, 0, -H.Z * 1.1f), FVector(H.X * 0.8f, H.Y * 0.3f, 12), Steel);
		}
		break;
	}
	}
}

void ARubbleVehicles::AddWheel(uint64 Key, const FRubbleWheelDef& Def, const FTransform& Initial)
{
	USceneComponent* Hub = NewObject<USceneComponent>(this);
	Hub->SetMobility(EComponentMobility::Movable);
	Hub->SetupAttachment(RootComponent);
	Hub->RegisterComponent();
	Hub->SetWorldTransform(Initial);
	Owned.Add(Hub);
	USceneComponent* Spin = NewObject<USceneComponent>(this);
	Spin->SetMobility(EComponentMobility::Movable);
	Spin->SetupAttachment(Hub);
	Spin->RegisterComponent();
	Owned.Add(Spin);
	// The engine cylinder stands along Z: lay it along the axle (Y).
	const FRotator Axle(0, 0, 90);
	const float D = 2.0f * Def.Radius / 100.0f, W = Def.Width / 100.0f;
	Part(Spin, RubbleRender::EngineCylinder(), FTransform(Axle, FVector::ZeroVector, FVector(D, D, W)), RubbleRender::Surface(ERubbleMaterial::Rubber));
	Part(Spin, RubbleRender::EngineCylinder(), FTransform(Axle, FVector::ZeroVector, FVector(D * 0.62f, D * 0.62f, W * 1.04f)), RubbleRender::Surface(ERubbleMaterial::Steel));
	for (int32 k = 0; k < 3; ++k)
	{
		Part(Spin, RubbleRender::EngineCube(), FTransform(FRotator(60.0f * k, 0, 0), FVector::ZeroVector, FVector(Def.Radius * 1.15f / 100.0f, W * 1.08f, 0.06f)),
			RubbleRender::Surface(ERubbleMaterial::Steel));
	}
	FWheel Wheel;
	Wheel.Hub = Hub;
	Wheel.Spin = Spin;
	Wheel.Current.Hub = Wheel.Previous.Hub = Initial;
	Wheels.Add(Key, Wheel);
}

void ARubbleVehicles::Rebuild(const FRubbleSceneAssets& Assets)
{
	for (USceneComponent* C : Owned)
	{
		if (C)
		{
			C->DestroyComponent();
		}
	}
	Owned.Reset();
	Chassis.Reset();
	WheelPlaced.Reset();
	Wheels.Reset();
	PlayerSoft = INDEX_NONE;
	VehicleDefs = Assets.Vehicles;
	SoftDefs = Assets.SoftBodies;
	for (const FRubbleVehicleVisual& V : Assets.Vehicles)
	{
		const FRubbleBodyVisual* Body = Assets.Bodies.FindByPredicate([&V](const FRubbleBodyVisual& B) { return B.Id == V.Chassis; });
		const FTransform Initial = Body ? Body->Initial : FTransform::Identity;
		USceneComponent* Root = NewObject<USceneComponent>(this);
		Root->SetMobility(EComponentMobility::Movable);
		Root->SetupAttachment(RootComponent);
		Root->RegisterComponent();
		Root->SetWorldTransform(Initial);
		Owned.Add(Root);
		BuildBody(Root, V);
		FChassis C;
		C.Root = Root;
		C.Previous = C.Current = Initial;
		C.bPlayer = V.bPlayer;
		if (V.Chassis.IsNull())
		{
			WheelPlaced.Add(V.Vehicle, C);
			continue; // wheels arrive with the frames
		}
		Chassis.Add(V.Chassis.Index(), C);
		for (int32 w = 0; w < V.Wheels.Num(); ++w)
		{
			AddWheel(WheelKey(false, V.Vehicle, w), V.Wheels[w], FTransform(FVector(V.Wheels[w].LocalHub)) * Initial);
		}
	}
	for (const FRubbleSoftVisual& S : Assets.SoftBodies)
	{
		for (int32 w = 0; w < S.Wheels.Num(); ++w)
		{
			AddWheel(WheelKey(true, S.Soft, w), S.Wheels[w], FTransform(FVector(0, 0, -10000)));
		}
		if (S.bPlayer)
		{
			PlayerSoft = S.Soft;
		}
	}
}

void ARubbleVehicles::OnFrame(const FRubbleFrame& Frame)
{
	for (const FRubbleMove& M : Frame.Moves)
	{
		if (FChassis* C = Chassis.Find(M.Body.Index()))
		{
			C->Previous = M.bHasPrevious ? M.Previous : M.Transform;
			C->Current = M.Transform;
		}
	}
	for (const FRubbleWheelVisual& W : Frame.Wheels)
	{
		const uint64 Key = WheelKey(W.bSoft, W.Vehicle, W.Wheel);
		if (!Wheels.Contains(Key))
		{
			// A wheel the scene did not list: made from the frame (its radius if known).
			FRubbleWheelDef Def;
			const TArray<FRubbleWheelDef>* Defs = W.bSoft ? (SoftDefs.IsValidIndex(W.Vehicle) ? &SoftDefs[W.Vehicle].Wheels : nullptr)
														  : (VehicleDefs.IsValidIndex(W.Vehicle) ? &VehicleDefs[W.Vehicle].Wheels : nullptr);
			if (Defs && Defs->Num())
			{
				Def = (*Defs)[FMath::Min(W.Wheel, Defs->Num() - 1)];
			}
			if (W.Radius > 1.0f)
			{
				Def.Radius = W.Radius;
			}
			AddWheel(Key, Def, W.Hub);
		}
		if (FWheel* Wheel = Wheels.Find(Key))
		{
			Wheel->Previous = Wheel->bSeen ? Wheel->Current : W;
			Wheel->Current = W;
			Wheel->bSeen = true;
		}
	}
	// Chassis placed from their wheels: centre of the hubs, right along the axles, up from both.
	for (TPair<int32, FChassis>& P : WheelPlaced)
	{
		FVector Sum = FVector::ZeroVector, Right = FVector::ZeroVector;
		int32 N = 0;
		for (const FRubbleWheelVisual& W : Frame.Wheels)
		{
			if (!W.bSoft && W.Vehicle == P.Key)
			{
				Sum += W.Hub.GetLocation();
				Right += W.Hub.GetRotation().GetAxisY() * (W.SteerRad == 0.0f ? 1.0 : 0.25); // spin is about the axle
				++N;
			}
		}
		if (N == 0)
		{
			continue;
		}
		Right = Right.GetSafeNormal(1e-6, FVector::RightVector);
		const FVector Forward = FVector::CrossProduct(Right, FVector::UpVector).GetSafeNormal(1e-6, FVector::ForwardVector);
		const FVector Up = FVector::CrossProduct(Forward, Right).GetSafeNormal();
		const FRubbleVehicleVisual* Def = VehicleDefs.IsValidIndex(P.Key) ? &VehicleDefs[P.Key] : nullptr;
		const FQuat Rot = FMatrix(Forward, Right, Up, FVector::ZeroVector).ToQuat();
		// The chassis origin: the hubs' centre, less where the descriptor puts that centre.
		FVector LocalCentre(0, 0, -30.0);
		if (Def && Def->Wheels.Num())
		{
			LocalCentre = FVector::ZeroVector;
			for (const FRubbleWheelDef& W : Def->Wheels)
			{
				LocalCentre += FVector(W.LocalHub);
			}
			LocalCentre /= Def->Wheels.Num();
		}
		const FTransform Pose(Rot, Sum / N - Rot.RotateVector(LocalCentre));
		P.Value.Previous = P.Value.Root->GetComponentTransform().Equals(FTransform::Identity) ? Pose : P.Value.Current;
		P.Value.Current = Pose;
	}
	// The soft player's core: the midpoint of its wheels, facing its first wheel's heading.
	if (PlayerSoft != INDEX_NONE)
	{
		FVector Sum = FVector::ZeroVector;
		int32 N = 0;
		FQuat Heading = FQuat::Identity;
		for (const FRubbleWheelVisual& W : Frame.Wheels)
		{
			if (W.bSoft && W.Vehicle == PlayerSoft)
			{
				Sum += W.Hub.GetLocation();
				Heading = W.Wheel == 2 ? W.Hub.GetRotation() : Heading; // a rear wheel: no steer
				++N;
			}
		}
		if (N)
		{
			PlayerSoftPose = FTransform(Heading, Sum / N);
		}
	}
}

void ARubbleVehicles::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>();
	if (!Rubble)
	{
		return;
	}
	const double Alpha = Rubble->Alpha();
	for (TPair<int32, FChassis>& P : WheelPlaced)
	{
		P.Value.Root->SetWorldTransform(RubbleLerpPose(P.Value.Previous, P.Value.Current, Alpha), false, nullptr, ETeleportType::TeleportPhysics);
	}
	for (TPair<uint32, FChassis>& P : Chassis)
	{
		P.Value.Root->SetWorldTransform(RubbleLerpPose(P.Value.Previous, P.Value.Current, Alpha), false, nullptr, ETeleportType::TeleportPhysics);
	}
	for (TPair<uint64, FWheel>& P : Wheels)
	{
		if (!P.Value.bSeen)
		{
			continue;
		}
		const FRubbleWheelVisual W = LerpWheel(P.Value.Previous, P.Value.Current, Alpha);
		P.Value.Hub->SetWorldTransform(W.Hub, false, nullptr, ETeleportType::TeleportPhysics);
		P.Value.Spin->SetRelativeRotation(FQuat(FVector(0, 1, 0), W.SpinRad));
	}
}

bool ARubbleVehicles::PlayerTransform(FTransform& Out) const
{
	for (const TPair<int32, FChassis>& P : WheelPlaced)
	{
		if (P.Value.bPlayer)
		{
			Out = P.Value.Root->GetComponentTransform();
			// The chase camera wants the heading the vehicle drives along.
			if (VehicleDefs.IsValidIndex(P.Key) && VehicleDefs[P.Key].bForwardIsMinusX)
			{
				Out.SetRotation(Out.GetRotation() * FQuat(FVector::UpVector, UE_PI));
			}
			return true;
		}
	}
	for (const TPair<uint32, FChassis>& P : Chassis)
	{
		if (P.Value.bPlayer)
		{
			Out = P.Value.Root->GetComponentTransform();
			return true;
		}
	}
	if (PlayerSoft != INDEX_NONE)
	{
		Out = PlayerSoftPose;
		return true;
	}
	return false;
}
