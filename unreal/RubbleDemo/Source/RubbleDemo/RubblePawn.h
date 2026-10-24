// The viewer: an orbit camera (right-drag to orbit, wheel to zoom, WASD/QE to move the focus), a
// chase camera on the player's vehicle (WASD drives), a free fly camera and a slow cinematic orbit.
// The hotkeys are the web pages': F or left click shoots, B a ball, T the side ram, R resets.
// Tab opens the scene menu.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "InputActionValue.h"
#include "RubblePawn.generated.h"

class UCameraComponent;
class UInputAction;
class UInputMappingContext;
struct FRubbleSceneAssets;

UENUM()
enum class ERubbleCamera : uint8
{
	Orbit,
	Chase,
	Fly,
	Cinematic,
};

UCLASS()
class RUBBLEDEMO_API ARubblePawn : public APawn
{
	GENERATED_BODY()

public:
	ARubblePawn();
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual void Tick(float DeltaSeconds) override;
	virtual void SetupPlayerInputComponent(UInputComponent* Input) override;
	virtual void PawnClientRestart() override;

	ERubbleCamera Mode() const { return CameraMode; }
	void SetMode(ERubbleCamera Mode);
	/** The fly camera at Eye, looking at Target (scripted shots). */
	void LookFrom(const FVector& Eye, const FVector& Target);
	void ToggleMenu();
	/** Fire along the camera's view (F) or through the cursor (left click). */
	void Shoot(bool bAtCursor);
	/** M: a meteor where the view meets the ground. */
	void CallMeteor();

	UPROPERTY(VisibleAnywhere)
	TObjectPtr<UCameraComponent> Camera;

private:
	void OnScene(const FRubbleSceneAssets& Assets);
	void CreateInput();
	class URubbleSubsystem* Rubble() const;
	bool HasPlayerVehicle() const;

	ERubbleCamera CameraMode = ERubbleCamera::Orbit;
	FVector Focus = FVector::ZeroVector;
	float Yaw = 30.0f;
	float Pitch = -20.0f;
	float Distance = 2000.0f;
	FVector2D Move = FVector2D::ZeroVector;
	float Up = 0.0f;
	bool bOrbiting = false;
	bool bHandbrake = false;
	float ChaseYaw = 0.0f;
	bool bMenuOpen = false;
	TSharedPtr<class SWidget> Menu;
	TSharedPtr<class SWidget> Hud;

	UPROPERTY()
	TObjectPtr<UInputMappingContext> Mapping;
	UPROPERTY()
	TMap<FName, TObjectPtr<UInputAction>> Actions;
	FDelegateHandle SceneHandle;
};
