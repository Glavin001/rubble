using UnrealBuildTool;

// Rubble's render-graph shaders (the GPU Scene writer), loaded at PostConfigInit so the global
// shader map finds them.
public class RubbleShaders : ModuleRules
{
	public RubbleShaders(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.NoPCHs;
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "RenderCore", "RHI", "Renderer" });
		PrivateDependencyModuleNames.AddRange(new string[] { "CoreUObject", "Engine", "Projects" });
	}
}
