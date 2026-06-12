cmake_minimum_required(VERSION 3.24)
project(vla_safety_runtime_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

add_executable(control_loop src/control_loop.cpp)
add_executable(test_seqlock src/test_seqlock.cpp)

foreach(tgt control_loop test_seqlock)
  target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(${tgt} PRIVATE /W4 /permissive- /utf-8)
endforeach()

target_link_libraries(control_loop PRIVATE winmm)
