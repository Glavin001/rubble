using UnrealBuildTool;

public class RubbleDemo : ModuleRules
{
	public RubbleDemo(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.NoPCHs;
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "RubbleRuntime" });
		PrivateDependencyModuleNames.AddRange(new string[] { "InputCore", "EnhancedInput", "Slate", "SlateCore" });
	}
}
