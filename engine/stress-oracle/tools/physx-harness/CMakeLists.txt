# Temporary measurement harnesses against a locally installed PhysX destruction SDK (the one
# vibe-land runs: PhysX/out/install/garage-multihull, commit a08abef50). Never part of rubble's
# build, tests or CI: configure by hand, with the SDK's prefix on CMAKE_PREFIX_PATH.
#
#   cmake -S engine/stress-oracle/tools/physx-harness -B <build dir> -G Ninja \
#     -DPHYSX_SDK=$HOME/Development/PhysX/out/install/garage-multihull
#   cmake --build <build dir>
cmake_minimum_required(VERSION 3.24)
# The SDK's static libraries are built for macOS 14.4.
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.4" CACHE STRING "")
project(PhysXHarness LANGUAGES CXX)
set(PHYSX_SDK "$ENV{HOME}/Development/PhysX/out/install/garage-multihull" CACHE PATH "PhysX destruction SDK install")
list(PREPEND CMAKE_PREFIX_PATH "${PHYSX_SDK}")
# The SDK's own CuMetal (cmake/CuMetalConfig.cmake).
set(CuMetal_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PhysXDestruction CONFIG REQUIRED)

# PhysX's own package-consumer test (tests/destruction/package-consumer/main.cpp at a08abef50),
# native scene only: proves the install links and its GPU stage runs here (pass any argument
# for the GPU).
add_executable(smoke smoke.cpp)
target_compile_features(smoke PRIVATE cxx_std_17)
target_compile_definitions(smoke PRIVATE PXD_NATIVE_ONLY)
target_link_libraries(smoke PRIVATE PhysXDestruction::NativeScene)
