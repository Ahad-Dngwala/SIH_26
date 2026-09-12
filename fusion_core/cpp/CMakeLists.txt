# fusion_core/cpp/ - MIP Section 5.5 step 2.
#
# Builds the `fusion_core` static library (linked by android_app/native/
# and edge_engine/src/ once those exist) plus its gtest suite. Run
# inside the `cpp-dev` Docker service (root docker-compose.yml already
# has the toolchain: build-essential, cmake, libeigen3-dev,
# libgtest-dev):
#
#   docker compose run --rm cpp-dev bash
#   cmake -S fusion_core/cpp -B fusion_core/cpp/build
#   cmake --build fusion_core/cpp/build
#   ctest --test-dir fusion_core/cpp/build --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(fusion_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(fusion_core
  src/ukf_core.cpp
  src/ukf.cpp
)
target_include_directories(fusion_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(fusion_core PUBLIC Eigen3::Eigen)
target_compile_options(fusion_core PRIVATE -Wall -Wextra)

option(FUSION_CORE_BUILD_TESTS "Build fusion_core's gtest suite" ON)
if(FUSION_CORE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)

  add_executable(fusion_core_tests tests/test_ukf.cpp)
  target_link_libraries(fusion_core_tests PRIVATE fusion_core GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(fusion_core_tests)
endif()
