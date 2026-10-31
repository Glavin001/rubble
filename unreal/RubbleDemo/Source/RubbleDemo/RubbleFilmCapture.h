// A film camera that records straight to video: a scene capture with its own render target and
// rendering state (temporal AA and Lumen keep their history per camera), its frames copied off
// the GPU through a ring of asynchronous readbacks (no stall) and piped raw into ffmpeg's
// hardware encoder (VideoToolbox on Mac). Several cameras film one run of a scene together.
#pragma once

#include "CoreMinimal.h"

class AActor;
class FRubbleFilmEncoder;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

class RUBBLEDEMO_API FRubbleFilmCapture
{
public:
	/** A camera on Owner writing Width x Height at Fps to Path (.mp4). */
	FRubbleFilmCapture(AActor* Owner, const FString& Path, int32 Width, int32 Height, int32 Fps);
	~FRubbleFilmCapture();

	/** False when ffmpeg could not be started (the reason is logged). */
	bool IsOpen() const;

	/** Point the camera (cm, degrees) without recording: its history settles. */
	void Aim(const FVector& Eye, const FRotator& View, float FieldOfView);
	/** Render from the camera's pose without recording (its temporal history settles). */
	void Warm();
	/** Render this frame from the camera's pose and queue it for the video. */
	void Shoot();
	/** Wait for every queued frame, close the video. */
	void Finish();

	int32 FramesWritten() const;

private:
	USceneCaptureComponent2D* Capture = nullptr;
	UTextureRenderTarget2D* Target = nullptr;
	TSharedPtr<FRubbleFilmEncoder, ESPMode::ThreadSafe> Encoder;
	/** Render-thread state: the readback ring and the frames in flight. */
	struct FRing;
	TSharedPtr<FRing, ESPMode::ThreadSafe> Ring;
	int32 Width = 0, Height = 0;
	int32 Written = 0; // frames written once finished
};
