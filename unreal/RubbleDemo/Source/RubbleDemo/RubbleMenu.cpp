#include "RubbleMenu.h"

#include "Engine/GameInstance.h"
#include "HAL/PlatformTime.h"
#include "RubbleSubsystem.h"
#include "Styling/CoreStyle.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SSpinBox.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/SOverlay.h"
#include "Widgets/Text/STextBlock.h"

namespace RubbleMenu
{
	namespace
	{
		const FLinearColor Panel(0.02f, 0.02f, 0.025f, 0.82f);
		const FLinearColor Accent(1.0f, 0.55f, 0.15f);

		FSlateFontInfo Font(int32 Size, bool bBold = false)
		{
			return FCoreStyle::GetDefaultFontStyle(bBold ? "Bold" : "Regular", Size);
		}

		struct FState
		{
			TWeakObjectPtr<UGameInstance> GameInstance;
			ERubbleFamily Family = ERubbleFamily::Rigid;
			FString Selected;
			TMap<FString, float> Values;
			TSharedPtr<SVerticalBox> Scenes;
			TSharedPtr<SVerticalBox> Params;
			FText Error;
			FSimpleDelegate OnClose;

			URubbleSubsystem* Rubble() const
			{
				return GameInstance.IsValid() ? GameInstance->GetSubsystem<URubbleSubsystem>() : nullptr;
			}
		};

		void FillParams(const TSharedRef<FState>& S);

		void FillScenes(const TSharedRef<FState>& S)
		{
			S->Scenes->ClearChildren();
			URubbleSubsystem* R = S->Rubble();
			if (!R)
			{
				return;
			}
			for (const FRubbleDemoDesc& D : R->ListDemos())
			{
				if (D.Family != S->Family)
				{
					continue;
				}
				const FString Id = D.Id;
				S->Scenes->AddSlot().AutoHeight().Padding(0, 1)
				[
					SNew(SButton)
					.ButtonColorAndOpacity_Lambda([S, Id]() { return S->Selected == Id ? Accent : FLinearColor(0.15f, 0.15f, 0.17f); })
					.OnClicked_Lambda([S, Id]() {
						S->Selected = Id;
						S->Values.Reset();
						FillParams(S);
						return FReply::Handled();
					})
					[
						SNew(STextBlock).Font(Font(11)).Text(FText::FromString(FString::Printf(TEXT("%s  ·  %s"), *D.Id, *D.Title)))
					]
				];
			}
		}

		void Load(const TSharedRef<FState>& S)
		{
			URubbleSubsystem* R = S->Rubble();
			if (!R || S->Selected.IsEmpty())
			{
				return;
			}
			FRubbleSceneRequest Request;
			Request.Family = S->Family;
			Request.Id = S->Selected;
			Request.Params = S->Values;
			FString Error;
			if (R->LoadScene(Request, Error))
			{
				S->Error = FText::GetEmpty();
				S->OnClose.ExecuteIfBound();
			}
			else
			{
				S->Error = FText::FromString(Error);
			}
		}

		void FillParams(const TSharedRef<FState>& S)
		{
			S->Params->ClearChildren();
			URubbleSubsystem* R = S->Rubble();
			const FRubbleDemoDesc* Desc = nullptr;
			const TArray<FRubbleDemoDesc> Demos = R ? R->ListDemos() : TArray<FRubbleDemoDesc>();
			for (const FRubbleDemoDesc& D : Demos)
			{
				if (D.Family == S->Family && D.Id == S->Selected)
				{
					Desc = &D;
				}
			}
			if (!Desc)
			{
				return;
			}
			S->Params->AddSlot().AutoHeight().Padding(0, 0, 0, 8)[SNew(STextBlock).Font(Font(13, true)).Text(FText::FromString(Desc->Title))];
			for (const FRubbleParamDesc& P : Desc->Params)
			{
				const FString Key = P.Key;
				S->Values.FindOrAdd(Key, P.Default);
				S->Params->AddSlot().AutoHeight().Padding(0, 2)
				[
					SNew(SHorizontalBox)
					+ SHorizontalBox::Slot().FillWidth(0.55f).VAlign(VAlign_Center)[SNew(STextBlock).Font(Font(10)).Text(FText::FromString(P.Label))]
					+ SHorizontalBox::Slot().FillWidth(0.45f)
					[
						SNew(SSpinBox<float>)
						.MinValue(P.Min).MaxValue(P.Max)
						.Value_Lambda([S, Key]() { return S->Values.FindRef(Key); })
						.OnValueChanged_Lambda([S, Key](float V) { S->Values.Add(Key, V); })
					]
				];
			}
			S->Params->AddSlot().AutoHeight().Padding(0, 10, 0, 0)
			[
				SNew(SButton).HAlign(HAlign_Center).ButtonColorAndOpacity(Accent).OnClicked_Lambda([S]() { Load(S); return FReply::Handled(); })
				[
					SNew(STextBlock).Font(Font(12, true)).Text(FText::FromString(TEXT("Load scene")))
				]
			];
		}
	}

	TSharedRef<SWidget> MakeMenu(UGameInstance* GameInstance, FSimpleDelegate OnClose)
	{
		TSharedRef<FState> S = MakeShared<FState>();
		S->GameInstance = GameInstance;
		S->OnClose = OnClose;
		if (const URubbleSubsystem* R = S->Rubble())
		{
			S->Family = R->CurrentScene().Family;
			S->Selected = R->CurrentScene().Id;
			S->Values = R->CurrentScene().Params;
		}
		TSharedRef<SHorizontalBox> Tabs = SNew(SHorizontalBox);
		static const TPair<ERubbleFamily, const TCHAR*> Families[] = {
			{ ERubbleFamily::Rigid, TEXT("Rigid scenes") },
			{ ERubbleFamily::Vehicles, TEXT("Vehicles") },
			{ ERubbleFamily::Soft, TEXT("Soft-body cars") },
			{ ERubbleFamily::Fracture, TEXT("Fracture lab") },
		};
		for (const auto& F : Families)
		{
			const ERubbleFamily Family = F.Key;
			Tabs->AddSlot().AutoWidth().Padding(0, 0, 4, 0)
			[
				SNew(SButton)
				.ButtonColorAndOpacity_Lambda([S, Family]() { return S->Family == Family ? Accent : FLinearColor(0.12f, 0.12f, 0.14f); })
				.OnClicked_Lambda([S, Family]() {
					S->Family = Family;
					S->Selected.Reset();
					FillScenes(S);
					FillParams(S);
					return FReply::Handled();
				})
				[SNew(STextBlock).Font(Font(12, true)).Text(FText::FromString(F.Value))]
			];
		}
		TSharedRef<SWidget> Root =
			SNew(SBox).HAlign(HAlign_Center).VAlign(VAlign_Center)
			[
				SNew(SBox).WidthOverride(980).HeightOverride(640)
				[
					SNew(SBorder).BorderImage(FCoreStyle::Get().GetBrush("WhiteBrush")).BorderBackgroundColor(Panel).Padding(18)
					[
						SNew(SVerticalBox)
						+ SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 10)
						[
							SNew(SHorizontalBox)
							+ SHorizontalBox::Slot().FillWidth(1.0f)[SNew(STextBlock).Font(Font(18, true)).Text(FText::FromString(TEXT("Rubble demos")))]
							+ SHorizontalBox::Slot().AutoWidth()
							[
								SNew(SButton).OnClicked_Lambda([S]() { S->OnClose.ExecuteIfBound(); return FReply::Handled(); })
								[SNew(STextBlock).Font(Font(11)).Text(FText::FromString(TEXT("Close (Tab)")))]
							]
						]
						+ SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 10)[Tabs]
						+ SVerticalBox::Slot().FillHeight(1.0f)
						[
							SNew(SHorizontalBox)
							+ SHorizontalBox::Slot().FillWidth(0.6f).Padding(0, 0, 14, 0)
							[
								SNew(SScrollBox) + SScrollBox::Slot()[SAssignNew(S->Scenes, SVerticalBox)]
							]
							+ SHorizontalBox::Slot().FillWidth(0.4f)[SAssignNew(S->Params, SVerticalBox)]
						]
						+ SVerticalBox::Slot().AutoHeight()
						[
							SNew(STextBlock).Font(Font(10)).ColorAndOpacity(FLinearColor(1.0f, 0.35f, 0.3f)).Text_Lambda([S]() { return S->Error; })
						]
					]
				]
			];
		FillScenes(S);
		FillParams(S);
		return Root;
	}

	TSharedRef<SWidget> MakeHud(UGameInstance* GameInstance)
	{
		TWeakObjectPtr<UGameInstance> GI = GameInstance;
		TSharedRef<double> LastTime = MakeShared<double>(FPlatformTime::Seconds());
		TSharedRef<double> SmoothedFps = MakeShared<double>(60.0);
		auto Rubble = [GI]() { return GI.IsValid() ? GI->GetSubsystem<URubbleSubsystem>() : nullptr; };
		return SNew(SOverlay).Visibility(EVisibility::HitTestInvisible)
			+ SOverlay::Slot().HAlign(HAlign_Left).VAlign(VAlign_Top).Padding(16)
			[
				SNew(SBorder).BorderImage(FCoreStyle::Get().GetBrush("WhiteBrush")).BorderBackgroundColor(FLinearColor(0, 0, 0, 0.45f)).Padding(10)
				[
					SNew(STextBlock).Font(Font(10)).ColorAndOpacity(FLinearColor(0.92f, 0.92f, 0.92f))
					.Text_Lambda([Rubble, LastTime, SmoothedFps]() {
						const double Now = FPlatformTime::Seconds();
						const double Dt = FMath::Max(1e-4, Now - *LastTime);
						*LastTime = Now;
						*SmoothedFps = FMath::Lerp(*SmoothedFps, 1.0 / Dt, 0.05);
						const URubbleSubsystem* R = Rubble();
						if (!R || !R->HasScene())
						{
							return FText::FromString(TEXT("no scene"));
						}
						const FRubbleDemoDesc* D = R->CurrentDemo();
						const FRubbleFrameStats& St = R->Stats();
						const FRubbleSceneAssets* A = R->Assets();
						FString S = FString::Printf(TEXT("%s\n%s   ·   backend: %s\n"), D ? *D->Title : TEXT(""), *R->CurrentScene().ToString(), R->Backend()->Name());
						S += FString::Printf(TEXT("%.0f fps   step %llu   %d steps/frame at %.0f Hz%s%s\n"), *SmoothedFps, St.Step, St.Steps, 1.0 / R->Backend()->StepSeconds(),
							R->bPaused ? TEXT("   PAUSED") : TEXT(""), R->TimeScale < 1.0 ? TEXT("   SLOW MOTION") : TEXT(""));
						S += FString::Printf(TEXT("bodies %d   moved %d   events %d   skins %d   chunks %d\n"), A ? A->Bodies.Num() : 0, St.Moves, St.Events, St.Skins, A ? A->Chunks.Num() : 0);
						S += FString::Printf(TEXT("backend %.2f ms   renderers %.2f ms%s"), St.BackendMs, St.BroadcastMs,
							St.DroppedSeconds > 0.0 ? *FString::Printf(TEXT("   dropped %.0f ms"), 1000.0 * St.DroppedSeconds) : TEXT(""));
						return FText::FromString(S);
					})
				]
			]
			+ SOverlay::Slot().HAlign(HAlign_Center).VAlign(VAlign_Bottom).Padding(12)
			[
				SNew(STextBlock).Font(Font(10)).ColorAndOpacity(FLinearColor(1, 1, 1, 0.7f)).ShadowOffset(FVector2D(1, 1))
				.Text(FText::FromString(TEXT("Tab scenes   F/click shoot   WASD move or drive   right-drag orbit   wheel zoom   C camera   K cinematic   T side ram   M meteor   R reset   Z slow motion   P pause   G effects   H hud")))
			];
	}
}
