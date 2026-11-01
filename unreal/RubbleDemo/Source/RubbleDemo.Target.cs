using UnrealBuildTool;
using System.IO;

public class RubbleDemoTarget : TargetRules
{
	public RubbleDemoTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V7;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_8;
		ExtraModuleNames.Add("RubbleDemo");
		RubbleDemoMacToolchain.Apply(this, Target);
	}
}

// Xcode 15.4 (macOS 14 SDK) is the newest Xcode on Sonoma 14.4. UE 5.8 builds with it given
// (copied from vibe-land's VibeCity, which found these):
//  - Build/Mac/MacSDK14Compat.h force-included (declarations the engine headers need from the
//    macOS 15 SDK; the engine falls back to a plain lock at runtime on macOS 14);
//  - no PCHs (Apple clang 15 rejects a constrained TOptional destructor reached through the
//    engine's shared PCH), so every file includes what it uses;
//  - UBA off (Saved/UnrealBuildTool/BuildConfiguration.xml), which mangles -rpath link arguments.
// See unreal/README.md.
public static class RubbleDemoMacToolchain
{
	public static void Apply(TargetRules Rules, TargetInfo Target)
	{
		if (Target.Platform != UnrealTargetPlatform.Mac || Target.ProjectFile == null)
		{
			return;
		}
		Rules.bOverrideBuildEnvironment = true;
		string CompatHeader = Path.Combine(Target.ProjectFile.Directory.FullName, "Build", "Mac", "MacSDK14Compat.h");
		Rules.AdditionalCompilerArguments += " -include \"" + CompatHeader + "\"";
		Rules.bUseSharedPCHs = false;
		Rules.bUsePCHFiles = false;
	}
}
