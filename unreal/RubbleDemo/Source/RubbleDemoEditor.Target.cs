using UnrealBuildTool;

public class RubbleDemoEditorTarget : TargetRules
{
	public RubbleDemoEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V7;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_8;
		ExtraModuleNames.Add("RubbleDemo");
		RubbleDemoMacToolchain.Apply(this, Target);
	}
}
