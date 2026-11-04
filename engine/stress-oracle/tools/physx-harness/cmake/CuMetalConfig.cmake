# CuMetal as the PhysX install bundles it (lib/libcumetal.dylib, include/cumetal), so the
# harness links the very runtime the SDK was built and warmed with, not a newer cuda-metal
# build. PHYSX_SDK is the install prefix (CMakeLists.txt sets it).
if(NOT TARGET CuMetal::Runtime)
    add_library(CuMetal::Runtime SHARED IMPORTED)
    set_target_properties(CuMetal::Runtime PROPERTIES
        IMPORTED_LOCATION "${PHYSX_SDK}/lib/libcumetal.dylib"
        INTERFACE_INCLUDE_DIRECTORIES "${PHYSX_SDK}/include/cumetal")
endif()
set(CuMetal_FOUND TRUE)
