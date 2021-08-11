cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpptorch)

add_subdirectory(fmt)

find_package(OpenCV 4.5.1 REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXECUTABLE_OUTPUT_PATH "../bin")

add_executable(main main.cpp)
add_executable(inference inference.cpp)
add_executable(inference_gpu inference_gpu.cpp)

set(LIBS ${TORCH_LIBRARIES} fmt::fmt ${OpenCV_LIBS})
target_link_libraries(main ${LIBS})
target_link_libraries(inference ${LIBS})
target_link_libraries(inference_gpu ${LIBS})

set_property(TARGET main inference inference_gpu PROPERTY CXX_STANDARD 17)
