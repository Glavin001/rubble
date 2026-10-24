using UnrealBuildTool;

public class RubbleRuntime : ModuleRules
{
	public RubbleRuntime(ReadOnlyTargetRules Target) : base(Target)
	{
		// No PCHs: Apple clang 15 (Xcode 15.4) rejects the engine's shared PCH (see unreal/README.md).
		PCHUsage = PCHUsageMode.NoPCHs;
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ProceduralMeshComponent" });
		PrivateDependencyModuleNames.AddRange(new string[] { "MeshDescription", "StaticMeshDescription", "RenderCore", "RHI", "Json", "RubbleFFI", "Renderer", "RubbleShaders" });
	}
}
