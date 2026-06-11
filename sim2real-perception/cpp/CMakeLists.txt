cmake_minimum_required(VERSION 3.24)
# 全工程统一动态 CRT (/MD): pybind11 模块必须与 Python 共用 CRT,
# OpenCV 静态库默认 /MT 会产生 LNK2038, 这里一并关掉
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
project(s2r_runtime CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  add_compile_options(/W4 /O2 /utf-8)
endif()

include(FetchContent)

# ---------------------------------------------------------------- onnxruntime
# 官方预编译包 (MSVC ABI), 不参与编译, 仅 IMPORTED target
set(ORT_VERSION "1.26.0" CACHE STRING "onnxruntime prebuilt version")
FetchContent_Declare(ort_prebuilt
  URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip
)
FetchContent_MakeAvailable(ort_prebuilt)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ort_prebuilt_SOURCE_DIR}/lib/onnxruntime.dll
  IMPORTED_IMPLIB   ${ort_prebuilt_SOURCE_DIR}/lib/onnxruntime.lib
  INTERFACE_INCLUDE_DIRECTORIES ${ort_prebuilt_SOURCE_DIR}/include
)
set(ORT_DLL ${ort_prebuilt_SOURCE_DIR}/lib/onnxruntime.dll)

# -------------------------------------------------------------------- opencv
# 源码最小构建: core + imgproc + imgcodecs, 静态库 (一次配置, 增量复用)
set(OPENCV_VERSION "4.10.0" CACHE STRING "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_MSMF OFF CACHE BOOL "" FORCE)
set(WITH_OBSENSOR OFF CACHE BOOL "" FORCE)
set(WITH_DSHOW OFF CACHE BOOL "" FORCE)
set(WITH_TIFF OFF CACHE BOOL "" FORCE)
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_OPENJPEG OFF CACHE BOOL "" FORCE)
set(WITH_JASPER OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
set(WITH_ADE OFF CACHE BOOL "" FORCE)
set(WITH_QUIRC OFF CACHE BOOL "" FORCE)
FetchContent_Declare(opencv
  URL https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip
)
FetchContent_MakeAvailable(opencv)

# in-tree 构建的 include 路径不会自动聚合, 手动收集
set(OPENCV_INCLUDE_DIRS
  ${opencv_SOURCE_DIR}/include
  ${opencv_SOURCE_DIR}/modules/core/include
  ${opencv_SOURCE_DIR}/modules/imgproc/include
  ${opencv_SOURCE_DIR}/modules/imgcodecs/include
  ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
)

# ------------------------------------------------------------- nlohmann json
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
)
FetchContent_MakeAvailable(json)

# ------------------------------------------------------------------ s2r_core
add_library(s2r_core STATIC
  src/manifest.cpp
  src/image_pipeline.cpp
  src/ort_engine.cpp
  src/cosine_gate.cpp
  src/runtime.cpp
)
target_include_directories(s2r_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OPENCV_INCLUDE_DIRS}
  ${json_SOURCE_DIR}/include
)
target_link_libraries(s2r_core PUBLIC
  onnxruntime opencv_core opencv_imgproc opencv_imgcodecs
)

# --------------------------------------------------------------------- apps
add_executable(deploy_replay apps/deploy_replay.cpp)
target_link_libraries(deploy_replay PRIVATE s2r_core)
add_custom_command(TARGET deploy_replay POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLL} $<TARGET_FILE_DIR:deploy_replay>
)

# -------------------------------------------------------------------- tests
enable_testing()
add_executable(s2r_tests tests/test_pipeline.cpp)
target_link_libraries(s2r_tests PRIVATE s2r_core)
add_custom_command(TARGET s2r_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLL} $<TARGET_FILE_DIR:s2r_tests>
)
add_test(NAME pipeline COMMAND s2r_tests)

# ------------------------------------------------------------- pybind11 模块
option(BUILD_PYTHON_BINDINGS "Build s2r_cpp python module" ON)
if(BUILD_PYTHON_BINDINGS)
  FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
  )
  FetchContent_MakeAvailable(pybind11)
  pybind11_add_module(s2r_cpp bindings/pymodule.cpp)
  target_link_libraries(s2r_cpp PRIVATE s2r_core)
  add_custom_command(TARGET s2r_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_DLL} $<TARGET_FILE_DIR:s2r_cpp>
  )
endif()
