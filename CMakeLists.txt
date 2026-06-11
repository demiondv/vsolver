cmake_minimum_required(VERSION 3.16)
project(vsolver C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra)

add_library(vsolver
  src/linalg/csc.c
  src/linalg/ldl.c
  src/cones/cone.c
  src/kkt/kkt.c
  src/solver/api.c
  src/solver/build.c
  src/solver/ipm.c
)
target_include_directories(vsolver PUBLIC include PRIVATE src)
target_link_libraries(vsolver PUBLIC m)

enable_testing()
foreach(t test_qp test_socp test_osqp test_cvxpy)
  add_executable(${t} tests/${t}.c)
  target_link_libraries(${t} vsolver)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

add_executable(bench tests/bench.c)
target_link_libraries(bench vsolver)

# examples (built, not registered as tests)
foreach(e lp portfolio least_squares socp)
  add_executable(example_${e} examples/${e}.c)
  target_link_libraries(example_${e} vsolver)
endforeach()
