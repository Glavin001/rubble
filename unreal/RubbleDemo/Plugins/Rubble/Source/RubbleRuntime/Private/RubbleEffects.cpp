#include "RubbleEffects.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Engine/GameInstance.h"
#include "RubbleRender.h"
#include "RubbleSubsystem.h"

namespace
{
	constexpr int32 Capacity[3] = { 4000, 1500, 2000 };
	constexpr int32 MaxSpawnPerFrame = 600;
	constexpr double GravityCm = 981.0;
	const FTransform Hidden(FQuat::Identity, FVector(0, 0, -100000.0), FVector::ZeroVector);

	FLinearColor ChipColor(ERubbleMaterial M, FRandomStream& R)
	{
		const float v = R.FRandRange(0.75f, 1.1f);
		switch (M)
		{
		case ERubbleMaterial::Brick: return FLinearColor(0.5f, 0.2f, 0.13f) * v;
		case ERubbleMaterial::Timber: return FLinearColor(0.55f, 0.38f, 0.2f) * v;
		case ERubbleMaterial::Glass: return FLinearColor(0.6f, 0.75f, 0.8f) * v;
		case ERubbleMaterial::CarPaint: return FLinearColor(0.25f, 0.25f, 0.27f) * v;
		case ERubbleMaterial::Steel: return FLinearColor(0.5f, 0.5f, 0.52f) * v;
		default: return FLinearColor(0.55f, 0.53f, 0.5f) * v;
		}
	}
}

ARubbleEffects::ARubbleEffects()
{
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.TickGroup = TG_PostPhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

void ARubbleEffects::BeginPlay()
{
	Super::BeginPlay();
	for (int32 k = 0; k < 3; ++k)
	{
		FPool& P = Pools[k];
		P.Component = NewObject<UInstancedStaticMeshComponent>(this);
		P.Component->SetStaticMesh(k == int32(EKind::Dust) ? RubbleRender::EngineSphere() : RubbleRender::EngineCube());
		P.Component->SetMaterial(0, k == int32(EKind::Chip) ? RubbleRender::Surface(ERubbleMaterial::Concrete) : k == int32(EKind::Dust) ? RubbleRender::Dust() : RubbleRender::Spark());
		P.Component->SetMobility(EComponentMobility::Movable);
		P.Component->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		P.Component->NumCustomDataFloats = 4; // rgb tint (chips), opacity or glow in 3
		P.Component->SetCastShadow(k == int32(EKind::Chip));
		P.Component->SetupAttachment(RootComponent);
		P.Component->RegisterComponent();
		P.Particles.SetNum(Capacity[k]);
		P.Transforms.Init(Hidden, Capacity[k]);
		P.Component->AddInstances(P.Transforms, false, true);
	}
	if (URubbleSubsystem* Rubble = GetGameInstance()->GetSubsystem<URubbleSubsystem>())
	{
		FrameHandle = Rubble->OnFrame.AddUObject(this, &ARubbleEffects::OnFrame);
		SceneHandle = Rubble->OnSceneLoaded.AddUObject(this, &ARubbleEffects::OnScene);
	}
}

void ARubbleEffects::EndPlay(const EEndPlayReason::Type Reason)
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

void ARubbleEffects::OnScene(const FRubbleSceneAssets& Assets)
{
	// A new scene starts clean: every particle hidden now (Tick skips empty pools).
	for (FPool& P : Pools)
	{
		for (FParticle& Q : P.Particles)
		{
			Q.Life = 0.0f;
		}
		P.Live = 0;
		for (FTransform& T : P.Transforms)
		{
			T = Hidden;
		}
		if (P.Component)
		{
			P.Component->BatchUpdateInstancesTransforms(0, P.Transforms, true, true, true);
		}
	}
}

ARubbleEffects::FParticle& ARubbleEffects::Spawn(EKind Kind)
{
	FPool& P = PoolOf(Kind);
	FParticle& Q = P.Particles[P.Head];
	P.Live += Q.Life > 0.0f ? 0 : 1;
	P.Head = (P.Head + 1) % P.Particles.Num();
	Q = FParticle();
	++SpawnedThisFrame;
	return Q;
}

void ARubbleEffects::Chips(const FVector& At, int32 Count, ERubbleMaterial Material, float Speed, float Size)
{
	for (int32 i = 0; i < Count && SpawnedThisFrame < MaxSpawnPerFrame; ++i)
	{
		FParticle& Q = Spawn(EKind::Chip);
		const float S = Size * FMath::Pow(4.0f, Random.FRand()) / 100.0f;
		Q.Scale = FVector(Random.FRandRange(0.5f, 1.2f), Random.FRandRange(0.5f, 1.2f), Random.FRandRange(0.3f, 0.9f)) * S;
		Q.Pos = At + Random.VRand() * 20.0;
		FVector Dir = Random.VRand();
		Dir.Z = FMath::Abs(Dir.Z) * 0.8 + 0.2;
		Q.Vel = Dir.GetSafeNormal() * Speed * Random.FRandRange(0.4f, 1.0f);
		Q.Rot = FQuat(Random.VRand(), Random.FRandRange(0.0f, UE_PI));
		Q.AngVel = Random.VRand() * Random.FRandRange(3.0f, 15.0f);
		Q.Color = ChipColor(Material, Random);
		Q.Life = Random.FRandRange(6.0f, 12.0f);
	}
}

void ARubbleEffects::DustPuffs(const FVector& At, int32 Count, float Radius)
{
	for (int32 i = 0; i < Count && SpawnedThisFrame < MaxSpawnPerFrame; ++i)
	{
		FParticle& Q = Spawn(EKind::Dust);
		Q.Pos = At + Random.VRand() * Radius * 0.5;
		Q.Vel = FVector(Random.FRandRange(-1.f, 1.f), Random.FRandRange(-1.f, 1.f), Random.FRandRange(0.2f, 1.0f)) * Radius * 0.8;
		Q.Size0 = Radius * Random.FRandRange(0.2f, 0.4f) / 50.0f;
		Q.Size1 = Radius * Random.FRandRange(0.9f, 1.6f) / 50.0f;
		Q.Life = Random.FRandRange(2.5f, 5.0f);
	}
}

void ARubbleEffects::Sparks(const FVector& At, const FVector& Normal, int32 Count)
{
	for (int32 i = 0; i < Count && SpawnedThisFrame < MaxSpawnPerFrame; ++i)
	{
		FParticle& Q = Spawn(EKind::Spark);
		Q.Pos = At;
		Q.Vel = (Normal + Random.VRand() * 0.8).GetSafeNormal() * Random.FRandRange(300.0f, 1200.0f);
		Q.Scale = FVector(0.03, 0.006, 0.006);
		Q.Life = Random.FRandRange(0.25f, 0.8f);
	}
}

void ARubbleEffects::OnFrame(const FRubbleFrame& Frame)
{
	if (!bEnabled)
	{
		return;
	}
	for (const FRubbleEvent& E : Frame.Events)
	{
		switch (E.Type)
		{
		case ERubbleEventType::Split:
			DustPuffs(E.Position, 10, 120.0f);
			Chips(E.Position, 30, ERubbleMaterial::Brick, 600.0f, 4.0f);
			break;
		case ERubbleEventType::BondsBroken:
			Chips(E.Position, FMath::Min(60, 2 * E.Count), ERubbleMaterial::Concrete, 500.0f, 3.0f);
			break;
		case ERubbleEventType::Impact:
			if (E.Magnitude > 300.0f)
			{
				DustPuffs(E.Position, FMath::Clamp(int32(E.Magnitude / 400.0f), 1, 6), 40.0f + 0.03f * E.Magnitude);
			}
			if ((E.MaterialA == ERubbleMaterial::Steel || E.MaterialB == ERubbleMaterial::Steel || E.MaterialA == ERubbleMaterial::CarPaint) && E.Magnitude > 700.0f)
			{
				Sparks(E.Position, E.Normal, FMath::Clamp(int32(E.Magnitude / 100.0f), 6, 30));
			}
			break;
		case ERubbleEventType::BeamBroke:
			Sparks(E.Position, E.Normal, 5);
			Chips(E.Position, 2, ERubbleMaterial::CarPaint, 300.0f, 2.0f);
			break;
		case ERubbleEventType::PieceDetached:
			Chips(E.Position + FVector(0, 0, 100), 40, ERubbleMaterial::Glass, 450.0f, 1.5f);
			DustPuffs(E.Position, 3, 60.0f);
			break;
		case ERubbleEventType::JointBroke:
			Sparks(E.Position, FVector::UpVector, 14);
			Chips(E.Position, 6, ERubbleMaterial::Steel, 400.0f, 2.0f);
			break;
		}
	}
}

void ARubbleEffects::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	SpawnedThisFrame = 0;
	const float Dt = FMath::Min(DeltaSeconds, 0.05f);
	for (int32 k = 0; k < 3; ++k)
	{
		FPool& P = Pools[k];
		if (P.Live == 0)
		{
			continue;
		}
		int32 Live = 0;
		for (int32 i = 0; i < P.Particles.Num(); ++i)
		{
			FParticle& Q = P.Particles[i];
			if (Q.Life <= 0.0f)
			{
				P.Transforms[i] = Hidden;
				continue;
			}
			Q.Age += Dt;
			if (Q.Age >= Q.Life)
			{
				Q.Life = 0.0f;
				P.Transforms[i] = Hidden;
				continue;
			}
			++Live;
			const float u = Q.Age / Q.Life;
			float Fade = 1.0f;
			switch (EKind(k))
			{
			case EKind::Chip:
			{
				Q.Vel.Z -= GravityCm * Dt;
				Q.Pos += Q.Vel * Dt;
				const double Half = Q.Scale.Z * 50.0;
				if (Q.Pos.Z < Half)
				{
					Q.Pos.Z = Half;
					Q.Vel.Z = -Q.Vel.Z * 0.3;
					Q.Vel.X *= 0.6;
					Q.Vel.Y *= 0.6;
					Q.AngVel *= 0.5;
				}
				const double W = Q.AngVel.Size();
				if (W > 1e-4)
				{
					Q.Rot = FQuat(Q.AngVel / W, W * Dt) * Q.Rot;
				}
				const float Shrink = u > 0.85f ? 1.0f - (u - 0.85f) / 0.15f : 1.0f;
				P.Transforms[i] = FTransform(Q.Rot, Q.Pos, Q.Scale * Shrink);
				Fade = 1.0f;
				break;
			}
			case EKind::Dust:
			{
				Q.Vel *= FMath::Pow(0.4f, Dt);
				Q.Vel.Z += 30.0 * Dt;
				Q.Pos += Q.Vel * Dt;
				const float Size = FMath::Lerp(Q.Size0, Q.Size1, 1.0f - FMath::Square(1.0f - u));
				P.Transforms[i] = FTransform(FQuat::Identity, Q.Pos, FVector(Size));
				Fade = FMath::Min(1.0f, 6.0f * u) * FMath::Pow(1.0f - u, 1.6f);
				break;
			}
			case EKind::Spark:
			{
				Q.Vel.Z -= GravityCm * Dt;
				Q.Pos += Q.Vel * Dt;
				// Stretched along the velocity, as a streak.
				const FQuat Along = FQuat::FindBetweenNormals(FVector(1, 0, 0), Q.Vel.GetSafeNormal());
				P.Transforms[i] = FTransform(Along, Q.Pos, FVector(0.02 + 0.00004 * Q.Vel.Size(), 0.005, 0.005));
				Fade = 1.0f - u;
				break;
			}
			}
			const float Data[4] = { Q.Color.R, Q.Color.G, Q.Color.B, Fade };
			P.Component->SetCustomData(i, TArrayView<const float>(Data, 4), false);
		}
		P.Live = Live;
		P.Component->BatchUpdateInstancesTransforms(0, P.Transforms, /*bWorldSpace=*/true, /*bMarkRenderStateDirty=*/true, /*bTeleport=*/true);
	}
}

int32 ARubbleEffects::NumLive() const
{
	return Pools[0].Live + Pools[1].Live + Pools[2].Live;
}
