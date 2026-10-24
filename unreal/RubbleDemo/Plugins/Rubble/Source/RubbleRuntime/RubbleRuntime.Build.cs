using UnrealBuildTool;

public class RubbleRuntime : ModuleRules
{
	public RubbleRuntime(ReadOnlyTargetRules Target) : base(Target)
	{
		// No PCHs: Apple clang 15 (Xcode 15.4) rejects the engine's shared PCH (see unreal/README.md).
		PCHUsage = PCHUsageMode.NoPCHs;
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ProceduralMeshComponent" });
		PrivateDependencyModuleNames.AddRange(new string[] { "MeshDescription", "StaticMeshDescription", "RenderCore", "RHI", "Json", "RubbleFFI", "Renderer", "RubbleShaders" });
		if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			// The Metal bridge (FRubbleGpuPoses): Unreal's MTLDevice and its buffers' MTLBuffers.
			PrivateDependencyModuleNames.Add("MetalRHI");
			// MetalResources.h (FMetalRHIBuffer, public) includes MetalRHIPrivate.h. The bridge is a
			// Mac-only stepping stone (DESIGN.md, phase 1a), so the private path is acceptable here.
			PrivateIncludePaths.Add(System.IO.Path.Combine(EngineDirectory, "Source/Runtime/Apple/MetalRHI/Private"));
			AddEngineThirdPartyPrivateStaticDependencies(Target, "MetalCPP", "MetalShaderConverter");
		}
	}
}
