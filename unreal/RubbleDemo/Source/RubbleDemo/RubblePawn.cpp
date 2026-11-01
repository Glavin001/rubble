#include "RubblePawn.h"

#include "Camera/CameraComponent.h"
#include "Engine/Engine.h"
#include "Engine/GameInstance.h"
#include "Engine/GameViewportClient.h"
#include "Engine/LocalPlayer.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "GameFramework/PlayerController.h"
#include "InputAction.h"
#include "InputCoreTypes.h"
#include "InputMappingContext.h"
#include "InputModifiers.h"
#include "RubbleEffects.h"
#include "RubbleMenu.h"
#include "RubbleSubsystem.h"
#include "RubbleVehicles.h"
#include "Widgets/SWeakWidget.h"

ARubblePawn::ARubblePawn()
{
	PrimaryActorTick.bCanEverTick = true;
	// Before the director steps (TG_PrePhysics too; ordered in BeginPlay): this frame's input.
	PrimaryActorTick.TickGroup = TG_PrePhysics;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
	Camera->SetupAttachment(RootComponent);
	Camera->SetFieldOfView(70.0f);
	Camera->PostProcessSettings.bOverride_DepthOfFieldFocalDistance = true;
	bUseControllerRotationYaw = false;
	bUseControllerRotationPitch = false;
	AutoPossessPlayer = EAutoReceiveInput::Player0;
	SetActorEnableCollision(false);
}

URubbleSubsystem* ARubblePawn::Rubble() const
{
	const UGameInstance* GI = GetGameInstance();
	return GI ? GI->GetSubsystem<URubbleSubsystem>() : nullptr;
}

void ARubblePawn::BeginPlay()
{
	Super::BeginPlay();
	if (URubbleSubsystem* R = Rubble())
	{
		SceneHandle = R->OnSceneLoaded.AddUObject(this, &ARubblePawn::OnScene);
		if (const FRubbleSceneAssets* A = R->Assets())
		{
			OnScene(*A);
		}
	}
	if (APlayerController* PC = Cast<APlayerController>(GetController()))
	{
		PC->bShowMouseCursor = true;
		PC->bEnableClickEvents = true;
		// The menu and the game both take the mouse; right-drag captures it to orbit.
		FInputModeGameAndUI Mode;
		Mode.SetHideCursorDuringCapture(false);
		Mode.SetLockMouseToViewportBehavior(EMouseLockMode::DoNotLock);
		PC->SetInputMode(Mode);
	}
	if (GEngine && GEngine->GameViewport)
	{
		Hud = RubbleMenu::MakeHud(GetGameInstance());
		GEngine->GameViewport->AddViewportWidgetContent(SNew(SWeakWidget).PossiblyNullContent(Hud), 10);
	}
}

void ARubblePawn::EndPlay(const EEndPlayReason::Type Reason)
{
	if (URubbleSubsystem* R = Rubble())
	{
		R->OnSceneLoaded.Remove(SceneHandle);
	}
	if (GEngine && GEngine->GameViewport)
	{
		if (Hud)
		{
			GEngine->GameViewport->RemoveViewportWidgetContent(Hud.ToSharedRef());
		}
		if (Menu)
		{
			GEngine->GameViewport->RemoveViewportWidgetContent(Menu.ToSharedRef());
		}
	}
	Super::EndPlay(Reason);
}

void ARubblePawn::OnScene(const FRubbleSceneAssets& Assets)
{
	Focus = Assets.CameraTarget;
	const FVector Offset = Assets.CameraEye - Assets.CameraTarget;
	Distance = FMath::Max(200.0, Offset.Size());
	const FRotator Look = (-Offset).Rotation();
	Yaw = Look.Yaw;
	Pitch = FMath::Clamp(Look.Pitch, -85.0f, 10.0f);
	ChaseYaw = 0.0f;
	if (CameraMode == ERubbleCamera::Chase && !HasPlayerVehicle())
	{
		CameraMode = ERubbleCamera::Orbit;
	}
	else if (HasPlayerVehicle() && CameraMode == ERubbleCamera::Orbit)
	{
		// Vehicle scenes start behind the wheel.
		CameraMode = ERubbleCamera::Chase;
	}
}

bool ARubblePawn::HasPlayerVehicle() const
{
	const URubbleSubsystem* R = Rubble();
	const FRubbleSceneAssets* A = R ? R->Assets() : nullptr;
	if (!A)
	{
		return false;
	}
	return A->Vehicles.ContainsByPredicate([](const FRubbleVehicleVisual& V) { return V.bPlayer; })
		|| A->SoftBodies.ContainsByPredicate([](const FRubbleSoftVisual& V) { return V.bPlayer; });
}

void ARubblePawn::LookFrom(const FVector& Eye, const FVector& Target)
{
	SetMode(ERubbleCamera::Fly);
	ViewFrom(Eye, (Target - Eye).Rotation(), 70.0f);
}

void ARubblePawn::ViewFrom(const FVector& Eye, const FRotator& View, float FieldOfView)
{
	SetMode(ERubbleCamera::Fly);
	Yaw = View.Yaw;
	Pitch = View.Pitch;
	Roll = View.Roll;
	Camera->SetFieldOfView(FieldOfView);
	SetActorLocationAndRotation(Eye, View);
}

void ARubblePawn::SetMode(ERubbleCamera InMode)
{
	CameraMode = InMode == ERubbleCamera::Chase && !HasPlayerVehicle() ? ERubbleCamera::Orbit : InMode;
}

void ARubblePawn::CreateInput()
{
	if (Mapping)
	{
		return;
	}
	Mapping = NewObject<UInputMappingContext>(this, TEXT("IMC_Rubble"));
	auto Action = [this](const TCHAR* Name, EInputActionValueType Type) {
		// Prefixed: an action named like a component of this pawn ("Camera") would replace it.
		UInputAction* A = NewObject<UInputAction>(this, *FString::Printf(TEXT("IA_%s"), Name));
		A->ValueType = Type;
		Actions.Add(Name, A);
		return A;
	};
	UInputAction* MoveA = Action(TEXT("Move"), EInputActionValueType::Axis2D);
	UInputAction* LookA = Action(TEXT("Look"), EInputActionValueType::Axis2D);
	UInputAction* UpA = Action(TEXT("Up"), EInputActionValueType::Axis1D);
	UInputAction* ZoomA = Action(TEXT("Zoom"), EInputActionValueType::Axis1D);
	UInputAction* OrbitA = Action(TEXT("Orbit"), EInputActionValueType::Boolean);
	UInputAction* HandbrakeA = Action(TEXT("Handbrake"), EInputActionValueType::Boolean);
	auto Negate = [this]() { return NewObject<UInputModifierNegate>(Mapping); };
	auto ToY = [this]() { return NewObject<UInputModifierSwizzleAxis>(Mapping); };
	// Move: X right, Y forward.
	for (const FKey& K : { EKeys::W, EKeys::Up })
	{
		Mapping->MapKey(MoveA, K).Modifiers.Add(ToY());
	}
	for (const FKey& K : { EKeys::S, EKeys::Down })
	{
		FEnhancedActionKeyMapping& M = Mapping->MapKey(MoveA, K);
		M.Modifiers.Add(ToY());
		M.Modifiers.Add(Negate());
	}
	for (const FKey& K : { EKeys::A, EKeys::Left })
	{
		Mapping->MapKey(MoveA, K).Modifiers.Add(Negate());
	}
	Mapping->MapKey(MoveA, EKeys::D);
	Mapping->MapKey(MoveA, EKeys::Right);
	Mapping->MapKey(MoveA, EKeys::Gamepad_Left2D);
	Mapping->MapKey(LookA, EKeys::Mouse2D);
	Mapping->MapKey(LookA, EKeys::Gamepad_Right2D);
	Mapping->MapKey(UpA, EKeys::E);
	Mapping->MapKey(UpA, EKeys::Q).Modifiers.Add(Negate());
	Mapping->MapKey(ZoomA, EKeys::MouseWheelAxis);
	Mapping->MapKey(OrbitA, EKeys::RightMouseButton);
	Mapping->MapKey(HandbrakeA, EKeys::SpaceBar);

	struct FPress
	{
		const TCHAR* Name;
		FKey Key;
	};
	const FPress Presses[] = {
		{ TEXT("Shoot"), EKeys::F },
		{ TEXT("Click"), EKeys::LeftMouseButton },
		{ TEXT("Ball"), EKeys::B },
		{ TEXT("Ram"), EKeys::T },
		{ TEXT("Meteor"), EKeys::M },
		{ TEXT("Reset"), EKeys::R },
		{ TEXT("Menu"), EKeys::Tab },
		{ TEXT("Camera"), EKeys::C },
		{ TEXT("Slow"), EKeys::Z },
		{ TEXT("Pause"), EKeys::P },
		{ TEXT("Hud"), EKeys::H },
		{ TEXT("Fx"), EKeys::G },
		{ TEXT("Cinematic"), EKeys::K },
	};
	for (const FPress& P : Presses)
	{
		Mapping->MapKey(Action(P.Name, EInputActionValueType::Boolean), P.Key);
	}
}

void ARubblePawn::PawnClientRestart()
{
	Super::PawnClientRestart();
	CreateInput();
	const APlayerController* PC = Cast<APlayerController>(GetController());
	const ULocalPlayer* LocalPlayer = PC ? PC->GetLocalPlayer() : nullptr;
	if (UEnhancedInputLocalPlayerSubsystem* Input = LocalPlayer ? LocalPlayer->GetSubsystem<UEnhancedInputLocalPlayerSubsystem>() : nullptr)
	{
		Input->ClearAllMappings();
		Input->AddMappingContext(Mapping, 0);
	}
}

void ARubblePawn::SetupPlayerInputComponent(UInputComponent* InputComponent)
{
	Super::SetupPlayerInputComponent(InputComponent);
	CreateInput();
	UEnhancedInputComponent* In = Cast<UEnhancedInputComponent>(InputComponent);
	if (!In)
	{
		UE_LOG(LogRubble, Error, TEXT("input component is not Enhanced Input (Config/DefaultInput.ini)"));
		return;
	}
	auto A = [this](const TCHAR* Name) { return Actions.FindChecked(Name).Get(); };
	In->BindActionValueLambda(A(TEXT("Move")), ETriggerEvent::Triggered, [this](const FInputActionValue& V) { Move = V.Get<FVector2D>(); });
	In->BindActionValueLambda(A(TEXT("Move")), ETriggerEvent::Completed, [this](const FInputActionValue&) { Move = FVector2D::ZeroVector; });
	In->BindActionValueLambda(A(TEXT("Up")), ETriggerEvent::Triggered, [this](const FInputActionValue& V) { Up = V.Get<float>(); });
	In->BindActionValueLambda(A(TEXT("Up")), ETriggerEvent::Completed, [this](const FInputActionValue&) { Up = 0.0f; });
	In->BindActionValueLambda(A(TEXT("Orbit")), ETriggerEvent::Started, [this](const FInputActionValue&) { bOrbiting = true; });
	In->BindActionValueLambda(A(TEXT("Orbit")), ETriggerEvent::Completed, [this](const FInputActionValue&) { bOrbiting = false; });
	In->BindActionValueLambda(A(TEXT("Handbrake")), ETriggerEvent::Started, [this](const FInputActionValue&) { bHandbrake = true; });
	In->BindActionValueLambda(A(TEXT("Handbrake")), ETriggerEvent::Completed, [this](const FInputActionValue&) { bHandbrake = false; });
	In->BindActionValueLambda(A(TEXT("Look")), ETriggerEvent::Triggered, [this](const FInputActionValue& V) {
		const FVector2D D = V.Get<FVector2D>();
		if (bOrbiting || CameraMode == ERubbleCamera::Fly)
		{
			Yaw += D.X * 0.25f;
			Pitch = FMath::Clamp(Pitch + D.Y * 0.25f, -88.0f, 60.0f);
			ChaseYaw += D.X * 0.25f;
			Roll = 0.0f;
		}
	});
	In->BindActionValueLambda(A(TEXT("Zoom")), ETriggerEvent::Triggered, [this](const FInputActionValue& V) {
		Distance = FMath::Clamp(Distance * FMath::Pow(0.9f, V.Get<float>()), 150.0f, 60000.0f);
	});
	In->BindActionValueLambda(A(TEXT("Shoot")), ETriggerEvent::Started, [this](const FInputActionValue&) { Shoot(false); });
	In->BindActionValueLambda(A(TEXT("Ball")), ETriggerEvent::Started, [this](const FInputActionValue&) { Shoot(false); });
	In->BindActionValueLambda(A(TEXT("Click")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		if (!bMenuOpen)
		{
			Shoot(true);
		}
	});
	In->BindActionValueLambda(A(TEXT("Meteor")), ETriggerEvent::Started, [this](const FInputActionValue&) { CallMeteor(); });
	In->BindActionValueLambda(A(TEXT("Ram")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		if (URubbleSubsystem* R = Rubble())
		{
			R->SideRam();
		}
	});
	In->BindActionValueLambda(A(TEXT("Reset")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		if (URubbleSubsystem* R = Rubble())
		{
			R->Reset();
		}
	});
	In->BindActionValueLambda(A(TEXT("Menu")), ETriggerEvent::Started, [this](const FInputActionValue&) { ToggleMenu(); });
	In->BindActionValueLambda(A(TEXT("Camera")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		SetMode(CameraMode == ERubbleCamera::Orbit ? ERubbleCamera::Chase : CameraMode == ERubbleCamera::Chase ? ERubbleCamera::Fly : ERubbleCamera::Orbit);
		if (CameraMode == ERubbleCamera::Chase && !HasPlayerVehicle())
		{
			CameraMode = ERubbleCamera::Fly;
		}
	});
	In->BindActionValueLambda(A(TEXT("Cinematic")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		SetMode(CameraMode == ERubbleCamera::Cinematic ? ERubbleCamera::Orbit : ERubbleCamera::Cinematic);
	});
	In->BindActionValueLambda(A(TEXT("Slow")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		if (URubbleSubsystem* R = Rubble())
		{
			R->TimeScale = R->TimeScale < 1.0 ? 1.0 : 0.25;
		}
	});
	In->BindActionValueLambda(A(TEXT("Pause")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		if (URubbleSubsystem* R = Rubble())
		{
			R->bPaused = !R->bPaused;
		}
	});
	In->BindActionValueLambda(A(TEXT("Hud")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		if (Hud)
		{
			Hud->SetVisibility(Hud->GetVisibility() == EVisibility::Collapsed ? EVisibility::HitTestInvisible : EVisibility::Collapsed);
		}
	});
	In->BindActionValueLambda(A(TEXT("Fx")), ETriggerEvent::Started, [this](const FInputActionValue&) {
		for (TActorIterator<ARubbleEffects> It(GetWorld()); It; ++It)
		{
			It->bEnabled = !It->bEnabled;
		}
	});
}

void ARubblePawn::ToggleMenu()
{
	if (!GEngine || !GEngine->GameViewport)
	{
		return;
	}
	bMenuOpen = !bMenuOpen;
	if (bMenuOpen)
	{
		Menu = RubbleMenu::MakeMenu(GetGameInstance(), FSimpleDelegate::CreateWeakLambda(this, [this]() {
			if (bMenuOpen)
			{
				ToggleMenu();
			}
		}));
		GEngine->GameViewport->AddViewportWidgetContent(Menu.ToSharedRef(), 20);
	}
	else if (Menu)
	{
		GEngine->GameViewport->RemoveViewportWidgetContent(Menu.ToSharedRef());
		Menu.Reset();
	}
}

void ARubblePawn::CallMeteor()
{
	URubbleSubsystem* R = Rubble();
	if (!R)
	{
		return;
	}
	// Where the view meets the ground (or 80 m ahead when looking at the sky).
	const FVector Origin = Camera->GetComponentLocation();
	const FVector Dir = Camera->GetForwardVector();
	const FVector Target = Dir.Z < -0.02 ? Origin + Dir * (-Origin.Z / Dir.Z) : Origin + Dir * 8000.0;
	R->Meteor(FVector(Target.X, Target.Y, FMath::Max(Target.Z, 0.0)));
}

void ARubblePawn::Shoot(bool bAtCursor)
{
	URubbleSubsystem* R = Rubble();
	APlayerController* PC = Cast<APlayerController>(GetController());
	if (!R || !PC)
	{
		return;
	}
	FVector Origin = Camera->GetComponentLocation();
	FVector Dir = Camera->GetForwardVector();
	if (bAtCursor)
	{
		FVector WorldOrigin, WorldDir;
		if (PC->DeprojectMousePositionToWorld(WorldOrigin, WorldDir))
		{
			Origin = WorldOrigin;
			Dir = WorldDir;
		}
	}
	// From a little ahead of the eye, so the ball is seen leaving.
	R->Launch(Origin + Dir * 150.0, Dir);
}

void ARubblePawn::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	URubbleSubsystem* R = Rubble();
	const bool bDriving = CameraMode == ERubbleCamera::Chase && HasPlayerVehicle();
	if (R)
	{
		FRubbleVehicleInput Input;
		if (bDriving && !bMenuOpen)
		{
			Input.Throttle = float(Move.Y);
			Input.Steer = float(Move.X);
			Input.bHandbrake = bHandbrake;
		}
		R->SetVehicleInput(Input);
	}

	const FRotator View(Pitch, Yaw, 0.0f);
	switch (CameraMode)
	{
	case ERubbleCamera::Orbit:
	case ERubbleCamera::Cinematic:
	{
		if (!bMenuOpen)
		{
			const FVector Forward = FRotator(0, Yaw, 0).Vector();
			const FVector Right = FRotator(0, Yaw + 90, 0).Vector();
			const float Speed = 0.8f * Distance;
			Focus += (Forward * Move.Y + Right * Move.X + FVector::UpVector * Up) * Speed * DeltaSeconds;
		}
		if (CameraMode == ERubbleCamera::Cinematic)
		{
			Yaw += 6.0f * DeltaSeconds;
		}
		const FVector Eye = Focus - View.Vector() * Distance;
		SetActorLocationAndRotation(FVector(Eye.X, Eye.Y, FMath::Max(Eye.Z, 30.0)), View);
		break;
	}
	case ERubbleCamera::Fly:
	{
		if (!bMenuOpen)
		{
			const FVector Delta = View.Vector() * Move.Y + FRotator(0, Yaw + 90, 0).Vector() * Move.X + FVector::UpVector * Up;
			SetActorLocation(GetActorLocation() + Delta * 1500.0 * DeltaSeconds);
		}
		SetActorRotation(FRotator(Pitch, Yaw, Roll));
		break;
	}
	case ERubbleCamera::Chase:
	{
		FTransform Player;
		for (TActorIterator<ARubbleVehicles> It(GetWorld()); It; ++It)
		{
			if (It->PlayerTransform(Player))
			{
				const float Heading = Player.GetRotation().Rotator().Yaw + ChaseYaw;
				const FRotator Behind(FMath::Clamp(Pitch, -40.0f, 5.0f) * 0.5f - 8.0f, Heading, 0.0f);
				const FVector Target = Player.GetLocation() + FVector(0, 0, 120);
				const FVector Want = Target - Behind.Vector() * FMath::Clamp(Distance, 500.0f, 2000.0f);
				const FVector Eye = FMath::VInterpTo(GetActorLocation(), Want, DeltaSeconds, 6.0f);
				SetActorLocationAndRotation(FVector(Eye.X, Eye.Y, FMath::Max(Eye.Z, 40.0)), (Target - Eye).Rotation());
				Focus = Player.GetLocation();
				ChaseYaw = FMath::FInterpTo(ChaseYaw, 0.0f, DeltaSeconds, bOrbiting ? 0.0f : 1.5f);
				break;
			}
		}
		break;
	}
	}
	if (APlayerController* PC = Cast<APlayerController>(GetController()))
	{
		PC->SetControlRotation(GetActorRotation());
	}
	// Depth of field focused on what the camera orbits.
	Camera->PostProcessSettings.DepthOfFieldFocalDistance = float(FVector::Dist(GetActorLocation(), Focus));
}
