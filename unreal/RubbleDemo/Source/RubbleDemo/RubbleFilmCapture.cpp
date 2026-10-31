#include "RubbleFilmCapture.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "GameFramework/Actor.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "RHIGPUReadback.h"
#include "RenderingThread.h"
#include "TextureResource.h"
#include "RubbleSubsystem.h"
#include <stdio.h>

/** Raw BGRA frames into ffmpeg on a thread of its own, so a full pipe never stalls the renderer. */
class FRubbleFilmEncoder : public FRunnable
{
public:
	FRubbleFilmEncoder(const FString& Path, int32 Width, int32 Height, int32 Fps)
	{
#if PLATFORM_MAC || PLATFORM_LINUX
		// VideoToolbox HEVC on Mac (the media engine, not the CPU); libx264 elsewhere.
		const TCHAR* Codec = PLATFORM_MAC ? TEXT("-c:v hevc_videotoolbox -b:v 24M -tag:v hvc1") : TEXT("-c:v libx264 -crf 18 -preset fast");
		const FString Command = FString::Printf(
			TEXT("PATH=/opt/homebrew/bin:/usr/local/bin:$PATH ffmpeg -loglevel error -y -f rawvideo -pix_fmt bgra -s %dx%d -r %d -i - %s -pix_fmt yuv420p -movflags +faststart \"%s\""),
			Width, Height, Fps, Codec, *Path);
		Pipe = popen(TCHAR_TO_UTF8(*Command), "w");
#endif
		if (!Pipe)
		{
			UE_LOG(LogRubble, Error, TEXT("film: could not start ffmpeg for %s"), *Path);
			return;
		}
		Wake = FPlatformProcess::GetSynchEventFromPool();
		Thread = FRunnableThread::Create(this, TEXT("RubbleFilmEncoder"));
	}

	~FRubbleFilmEncoder()
	{
		Close();
	}

	bool IsOpen() const { return Pipe != nullptr; }

	void Push(TArray<uint8>&& Frame)
	{
		Queue.Enqueue(MoveTemp(Frame));
		if (Wake)
		{
			Wake->Trigger();
		}
	}

	/** Write everything queued, then close the pipe (ffmpeg finishes the file). */
	void Close()
	{
		if (Thread)
		{
			bStop = true;
			Wake->Trigger();
			Thread->WaitForCompletion();
			delete Thread;
			Thread = nullptr;
		}
		if (Wake)
		{
			FPlatformProcess::ReturnSynchEventToPool(Wake);
			Wake = nullptr;
		}
#if PLATFORM_MAC || PLATFORM_LINUX
		if (Pipe)
		{
			pclose(Pipe);
		}
#endif
		Pipe = nullptr;
	}

	virtual uint32 Run() override
	{
		for (;;)
		{
			TArray<uint8> Frame;
			while (Queue.Dequeue(Frame))
			{
				fwrite(Frame.GetData(), 1, Frame.Num(), Pipe);
				++Written;
			}
			if (bStop)
			{
				return 0;
			}
			Wake->Wait(50);
		}
	}

	TAtomic<int32> Written{ 0 };

private:
	FILE* Pipe = nullptr;
	FEvent* Wake = nullptr;
	FRunnableThread* Thread = nullptr;
	TAtomic<bool> bStop{ false };
	TQueue<TArray<uint8>, EQueueMode::Mpsc> Queue;
};

/** The readbacks in flight, oldest first; touched only on the render thread. */
struct FRubbleFilmCapture::FRing
{
	static constexpr int32 Depth = 3;
	TArray<TUniquePtr<FRHIGPUTextureReadback>> Slots;
	TArray<bool> Busy;
	int32 Next = 0; // the slot the next frame copies into
	int32 Oldest = 0;
	int32 InFlight = 0;

	FRing()
	{
		for (int32 i = 0; i < Depth; ++i)
		{
			Slots.Add(MakeUnique<FRHIGPUTextureReadback>(TEXT("RubbleFilm")));
			Busy.Add(false);
		}
	}

	/** Hand the oldest frame to the encoder; with bWait, wait for the GPU first. */
	bool Drain(FRHICommandListImmediate& RHICmdList, FRubbleFilmEncoder& Out, int32 W, int32 H, bool bWait)
	{
		if (InFlight == 0)
		{
			return false;
		}
		FRHIGPUTextureReadback& R = *Slots[Oldest];
		if (!R.IsReady())
		{
			if (!bWait)
			{
				return false;
			}
			RHICmdList.BlockUntilGPUIdle();
		}
		int32 RowPitch = 0;
		const uint8* Data = static_cast<const uint8*>(R.Lock(RowPitch));
		TArray<uint8> Frame;
		Frame.SetNumUninitialized(W * H * 4);
		for (int32 y = 0; y < H; ++y)
		{
			FMemory::Memcpy(Frame.GetData() + y * W * 4, Data + y * RowPitch * 4, W * 4);
		}
		R.Unlock();
		Out.Push(MoveTemp(Frame));
		Busy[Oldest] = false;
		Oldest = (Oldest + 1) % Depth;
		--InFlight;
		return true;
	}
};

FRubbleFilmCapture::FRubbleFilmCapture(AActor* Owner, const FString& Path, int32 InWidth, int32 InHeight, int32 Fps)
	: Width(InWidth), Height(InHeight)
{
	Target = NewObject<UTextureRenderTarget2D>(Owner);
	Target->RenderTargetFormat = RTF_RGBA8;
	Target->ClearColor = FLinearColor::Black;
	Target->InitAutoFormat(Width, Height);
	Target->UpdateResourceImmediate(true);
	Capture = NewObject<USceneCaptureComponent2D>(Owner);
	Capture->CaptureSource = SCS_FinalColorLDR;
	Capture->bCaptureEveryFrame = false;
	Capture->bCaptureOnMovement = false;
	// Temporal AA, auto exposure and Lumen's history belong to this camera's view.
	Capture->bAlwaysPersistRenderingState = true;
	Capture->TextureTarget = Target;
	Capture->SetMobility(EComponentMobility::Movable);
	Capture->SetupAttachment(Owner->GetRootComponent());
	Capture->RegisterComponent();
	Encoder = MakeShared<FRubbleFilmEncoder, ESPMode::ThreadSafe>(Path, Width, Height, Fps);
	Ring = MakeShared<FRing, ESPMode::ThreadSafe>();
}

FRubbleFilmCapture::~FRubbleFilmCapture()
{
	Finish();
	if (Capture)
	{
		Capture->DestroyComponent();
	}
}

bool FRubbleFilmCapture::IsOpen() const
{
	return Encoder.IsValid() && Encoder->IsOpen();
}

int32 FRubbleFilmCapture::FramesWritten() const
{
	return Encoder.IsValid() ? Encoder->Written.Load() : Written;
}

void FRubbleFilmCapture::Aim(const FVector& Eye, const FRotator& View, float FieldOfView)
{
	Capture->SetWorldLocationAndRotation(Eye, View);
	Capture->FOVAngle = FieldOfView;
}

void FRubbleFilmCapture::Warm()
{
	Capture->CaptureScene();
}

void FRubbleFilmCapture::Shoot()
{
	if (!IsOpen())
	{
		return;
	}
	Capture->CaptureScene();
	FTextureRenderTargetResource* Resource = Target->GameThread_GetRenderTargetResource();
	TSharedPtr<FRing, ESPMode::ThreadSafe> R = Ring;
	TSharedPtr<FRubbleFilmEncoder, ESPMode::ThreadSafe> E = Encoder;
	const int32 W = Width, H = Height;
	ENQUEUE_RENDER_COMMAND(RubbleFilmShoot)([R, E, Resource, W, H](FRHICommandListImmediate& RHICmdList) {
		// Hand finished frames on; a full ring waits for its oldest.
		while (R->Drain(RHICmdList, *E, W, H, false))
		{
		}
		if (R->InFlight == FRing::Depth)
		{
			R->Drain(RHICmdList, *E, W, H, true);
		}
		R->Slots[R->Next]->EnqueueCopy(RHICmdList, Resource->GetRenderTargetTexture(), FIntVector::ZeroValue, 0, FIntVector(W, H, 1));
		R->Busy[R->Next] = true;
		R->Next = (R->Next + 1) % FRing::Depth;
		++R->InFlight;
	});
}

void FRubbleFilmCapture::Finish()
{
	if (!Encoder.IsValid())
	{
		return;
	}
	TSharedPtr<FRing, ESPMode::ThreadSafe> R = Ring;
	TSharedPtr<FRubbleFilmEncoder, ESPMode::ThreadSafe> E = Encoder;
	const int32 W = Width, H = Height;
	ENQUEUE_RENDER_COMMAND(RubbleFilmFinish)([R, E, W, H](FRHICommandListImmediate& RHICmdList) {
		while (R->Drain(RHICmdList, *E, W, H, true))
		{
		}
	});
	FlushRenderingCommands();
	Encoder->Close();
	Written = Encoder->Written.Load();
	Encoder.Reset();
}
