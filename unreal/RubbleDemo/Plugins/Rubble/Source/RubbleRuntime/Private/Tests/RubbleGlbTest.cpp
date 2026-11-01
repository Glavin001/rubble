// Rubble.Glb: the GLB reader places every fragment by its node hierarchy (level-1 fragments are
// children of level-0 nodes), in Unreal space.

#include "Misc/AutomationTest.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RubbleGlb.h"

#if WITH_DEV_AUTOMATION_TESTS

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FRubbleGlbArchTest, "Rubble.Glb.ArchBounds",
	EAutomationTestFlags::EngineFilter | EAutomationTestFlags::ClientContext | EAutomationTestFlags::EditorContext | EAutomationTestFlags::CommandletContext)

bool FRubbleGlbArchTest::RunTest(const FString& Parameters)
{
	TArray<uint8> Bytes;
	const FString Path = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../../engine/crates/engine-webgpu/fracture/assets/masonry-arch.glb"));
	if (!TestTrue(TEXT("the arch GLB is there (Git LFS)"), FFileHelper::LoadFileToArray(Bytes, *Path)))
	{
		return false;
	}
	RubbleGlb::FFile File;
	FString Error;
	if (!TestTrue(*FString::Printf(TEXT("reads (%s)"), *Error), RubbleGlb::Read(Bytes, File, Error)))
	{
		return false;
	}
	TestTrue(TEXT("recognised as Shardwright"), File.bShardwright);
	FBox All(ForceInit);
	int32 Meshes = 0;
	for (const FRubbleMeshData& M : File.NodeMeshes)
	{
		Meshes += M.NumTriangles() > 0 ? 1 : 0;
		for (const FVector3f& P : M.Positions)
		{
			All += FVector(P);
		}
	}
	AddInfo(FString::Printf(TEXT("%d meshes, bounds %s"), Meshes, *All.ToString()));
	TestTrue(TEXT("every node has a mesh"), Meshes > 700);
	// The arch spans about 4 m across (Rubble X = Unreal Y) and stands on the ground.
	TestTrue(*FString::Printf(TEXT("spans over 3 m (%s)"), *All.GetSize().ToString()), All.GetSize().Y > 300.0);
	TestTrue(TEXT("stands on the ground (its base slab reaches 50 cm below)"), All.Min.Z > -60.0 && All.Max.Z > 150.0);
	return true;
}

#endif
