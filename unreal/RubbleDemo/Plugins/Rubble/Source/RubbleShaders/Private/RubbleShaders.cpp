#include "RubbleShaders.h"

#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "ShaderCore.h"

void FRubbleGPUSceneCS::ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
{
	FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
	OutEnvironment.SetDefine(TEXT("VF_SUPPORTS_PRIMITIVE_SCENE_DATA"), 1);
	OutEnvironment.SetDefine(TEXT("THREAD_GROUP_SIZE"), ThreadGroupSize);
}

IMPLEMENT_GLOBAL_SHADER(FRubbleGPUSceneCS, "/Plugin/Rubble/RubbleGPUScene.usf", "WriteInstancesCS", SF_Compute);

class FRubbleShadersModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		const FString Dir = FPaths::Combine(IPluginManager::Get().FindPlugin(TEXT("Rubble"))->GetBaseDir(), TEXT("Shaders"));
		AddShaderSourceDirectoryMapping(TEXT("/Plugin/Rubble"), Dir);
	}
};

IMPLEMENT_MODULE(FRubbleShadersModule, RubbleShaders);
