// SPDX-License-Identifier: BSD-3-Clause
#include <PxPhysicsAPI.h>
#ifndef PXD_NATIVE_ONLY
#include <NvBlastExtStressPhysXResim.h>
#endif
#include <extensions/PxDefaultCpuDispatcher.h>

int main(int argc, char**) {
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback errors;
    auto* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    if(!foundation)return 1;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,physx::PxTolerancesScale());
    if(!physics)return 2;
    physx::PxCudaContextManager* cuda=nullptr;
    if(argc>1) {
        physx::PxCudaContextManagerDesc cudaDesc;
        cuda=PxCreateCudaContextManager(*foundation,cudaDesc);
        if(!cuda || !cuda->contextIsValid())return 6;
    }
    auto* dispatcher=physx::PxDefaultCpuDispatcherCreate(1);
    physx::PxSceneDesc desc(physics->getTolerancesScale());
    desc.cpuDispatcher=dispatcher;
    desc.filterShader=physx::PxDefaultSimulationFilterShader;
    if(cuda) {
        desc.cudaContextManager=cuda;
        desc.flags|=physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
        desc.broadPhaseType=physx::PxBroadPhaseType::eGPU;
    }
    auto* scene=physics->createScene(desc);
    if(!scene)return 3;
#ifndef PXD_NATIVE_ONLY
    auto* stepper=Nv::Blast::ExtStressPhysXFrameStepper::create(*scene);
    if(!stepper)return 4;
    bool ok=stepper->stepFrame(1.0f/60.0f,desc.gravity,nullptr,0,
        Nv::Blast::ExtStressPhysXResimOptions(),nullptr,nullptr);
#else
    scene->simulate(1.0f/60.0f);
    bool ok=scene->fetchResults(true);
#endif
    auto* native=scene->getDestructionScene();
    ok &= cuda ? native!=nullptr : native==nullptr;
    if(cuda && native) {
        auto* material=physics->createMaterial(0.5f,0.5f,0);
        auto* body=physics->createRigidDynamic(physx::PxTransform(physx::PxVec3(0,5,0)));
        auto* shape=physics->createShape(physx::PxBoxGeometry(0.5f,0.5f,0.5f),*material,true);
        ok &= body && shape && body->attachShape(*shape);
        body->setRigidBodyFlag(physx::PxRigidBodyFlag::eKINEMATIC,true);
        scene->addActor(*body);shape->release();material->release();
        scene->simulate(1.0f/60.0f);ok &= scene->fetchResults(true);
        physx::PxDestructionStressChunk chunks[2]={
            {physx::PxVec3(0,-1,0),0,0,0,0xFFFFFFFFu},
            {physx::PxVec3(0),2,1,0,0xFFFFFFFFu}};
        physx::PxDestructionStressBond bond{0,1,physx::PxVec3(0,-0.5f,0),physx::PxVec3(0,1,0),1,1,1};
        physx::PxDestructionStressCluster cluster{body->getGPUIndex(),physx::PxVec3(0)};
        physx::PxDestructionStressDesc graph;graph.chunks=chunks;graph.chunkCount=2;
        graph.bonds=&bond;graph.bondCount=1;graph.clusters=&cluster;graph.clusterCount=1;
        ok &= native->configureStress(graph);
        scene->simulate(1.0f/60.0f);physx::PxU32 error=0;
        ok &= scene->fetchResults(true,&error) && !error;
        ok &= native->getLastStatus().frame==1 && native->getLastStatus().converged;
        scene->setGravity(physx::PxVec3(0,-9.81f,0));
        physx::PxDestructionMaterial stressMaterial;
        stressMaterial.compressionElasticLimit=10;stressMaterial.compressionFatalLimit=100;
        graph.materials=&stressMaterial;graph.materialCount=1;
        const physx::PxDestructionChunkMassProperties massProperties[2]={
            {{0,-1,0},0,{0,0,0,0,0,0},1},{{0,0,0},2,{1,1,1,0,0,0},0}};
        graph.chunkMassProperties=massProperties;
        ok &= native->configureStress(graph);
        scene->simulate(1.0f/60.0f);ok &= scene->fetchResults(true,&error) && !error;
        ok &= native->getLastStatus().bondCommands==1 && !native->getLastStatus().brokenBonds;
        ok &= native->getDeviceView().topologyTransaction!=nullptr;
        ok &= native->getDeviceView().acceptedTopology.chunkCount==2;
        graph.chunkMassProperties=nullptr;
        graph.bonds=nullptr;graph.bondCount=0;
        ok &= native->configureStress(graph);
        scene->simulate(1.0f/60.0f);ok &= scene->fetchResults(true,&error) && !error;
        ok &= !native->getDeviceView().bondCount && !native->getDeviceView().bondForces;
        ok &= !native->getLastStatus().iterations && native->getLastStatus().converged;
        ok &= native->clearStress();body->release();
    }
#ifndef PXD_NATIVE_ONLY
    stepper->release();
#endif
    scene->release();dispatcher->release();physics->release();
    if(cuda)cuda->release();
    foundation->release();
    return ok?0:5;
}
