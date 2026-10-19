cmake_minimum_required(VERSION 3.16)
project(live2d_physics_bridge LANGUAGES CXX)
set(CUBISM_SDK "" CACHE PATH "User-installed Cubism Native SDK 5-r.5")
set(CUBISM_CORE "" CACHE FILEPATH "User-installed Core shared/import library")
if(NOT EXISTS "${CUBISM_SDK}/Framework/src/Physics/CubismPhysics.cpp" OR NOT EXISTS "${CUBISM_CORE}")
  message(FATAL_ERROR "Set CUBISM_SDK and CUBISM_CORE. No SDK is downloaded or included.")
endif()
set(FW "${CUBISM_SDK}/Framework/src")
set(FRAMEWORK_SOURCES
  CubismFramework.cpp Id/CubismId.cpp Id/CubismIdManager.cpp
  Math/CubismMath.cpp Math/CubismVector2.cpp
  Model/CubismMoc.cpp Model/CubismModel.cpp Model/CubismModelMultiplyAndScreenColor.cpp
  Physics/CubismPhysics.cpp Physics/CubismPhysicsJson.cpp Effect/CubismPose.cpp
  Rendering/csmBlendMode.cpp Type/csmString.cpp Utils/CubismDebug.cpp
  Utils/CubismJson.cpp Utils/CubismString.cpp)
list(TRANSFORM FRAMEWORK_SOURCES PREPEND "${FW}/")
add_library(live2d_physics_bridge SHARED physics_bridge.cpp ${FRAMEWORK_SOURCES})
target_compile_features(live2d_physics_bridge PRIVATE cxx_std_17)
target_include_directories(live2d_physics_bridge PRIVATE "${FW}" "${CUBISM_SDK}/Core/include")
target_link_libraries(live2d_physics_bridge PRIVATE "${CUBISM_CORE}")
set_target_properties(live2d_physics_bridge PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(UNIX AND NOT APPLE)
  target_link_options(live2d_physics_bridge PRIVATE -Wl,--no-undefined)
endif()
