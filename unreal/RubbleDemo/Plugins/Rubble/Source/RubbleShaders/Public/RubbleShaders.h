// The compute pass that writes Rubble bodies into Unreal's GPU Scene: each instance's transform,
// previous transform (motion vectors) and tint, from Rubble's pose buffers in place. The bodies
// stay ordinary instanced static meshes (Nanite, VSM, culling); see unreal/DESIGN.md.

#pragma once

#include "GPUSceneWriter.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"

class RUBBLESHADERS_API FRubbleGPUSceneCS : public FGlobalShader
{
	DECLARE_EXPORTED_GLOBAL_SHADER(FRubbleGPUSceneCS, RUBBLESHADERS_API);
	SHADER_USE_PARAMETER_STRUCT(FRubbleGPUSceneCS, FGlobalShader);

	static constexpr uint32 ThreadGroupSize = 64;

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, RUBBLESHADERS_API)
		SHADER_PARAMETER_STRUCT_INCLUDE(FGPUSceneWriterParameters, GPUSceneWriterParameters)
		SHADER_PARAMETER(uint32, NumInstances)
		SHADER_PARAMETER(uint32, PrimitiveId)
		SHADER_PARAMETER(float, Alpha)
		SHADER_PARAMETER(uint32, LocalRows)
		SHADER_PARAMETER(uint32, NumCustomFloats)
		// Rubble's pose records (32 B: position xyz, word, rotation xyzw; Rubble's frame), as float4 pairs.
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float4>, CurrentPoses)
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float4>, PreviousPoses)
		// Per instance: the body's pose slot (0xffffffff: hidden).
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<uint>, InstanceBody)
		// Per instance, LocalRows float4s in the body frame (Unreal axes, cm): translation, rotation,
		// scale, then the custom data rows (NumCustomFloats floats).
		SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float4>, InstanceLocal)
	END_SHADER_PARAMETER_STRUCT()
};

/** The GPU Scene PoC's stand-in for a Rubble step: a field of moving boxes as pose records. */
class RUBBLESHADERS_API FRubblePocAnimateCS : public FGlobalShader
{
	DECLARE_EXPORTED_GLOBAL_SHADER(FRubblePocAnimateCS, RUBBLESHADERS_API);
	SHADER_USE_PARAMETER_STRUCT(FRubblePocAnimateCS, FGlobalShader);

	static constexpr uint32 ThreadGroupSize = 64;

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, RUBBLESHADERS_API)
		SHADER_PARAMETER(uint32, NumBodies)
		SHADER_PARAMETER(uint32, Side)
		SHADER_PARAMETER(float, Time)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float4>, OutPoses)
	END_SHADER_PARAMETER_STRUCT()
};
