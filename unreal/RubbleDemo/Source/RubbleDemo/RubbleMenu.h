// The scene menu (Tab) and the HUD, in Slate: families and scenes from the backend's catalogue (the
// web pages' menus), each scene's parameters as fields, and the frame's stats.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/SWidget.h"

class UGameInstance;

namespace RubbleMenu
{
	TSharedRef<SWidget> MakeMenu(UGameInstance* GameInstance, FSimpleDelegate OnClose);
	TSharedRef<SWidget> MakeHud(UGameInstance* GameInstance);
}
