cmake_minimum_required(VERSION 2.8)

# Define and name the project we're building.
project(npoint_mlpack C CXX Fortran)

set(DIRS
  distributed
  efficient_kernels
  infrastructure
  matchers
  resampling_classes
  results
  tests
)

# Options
option(DEBUG "Compile with debugging information" ON)
option(PROFILE "Compile with profiling information" ON)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -msse2 -msse3 -mssse3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCC)

# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if(DEBUG)
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -ggdb -O0")
else()
  add_definitions(-DARMA_NO_DEBUG)
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
endif(DEBUG)

# Profiling CFLAGS.  Turn profiling information on.
if(PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(PROFILE)

# find armadillo linear algebra library
find_package(Armadillo 2.4.0 REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIR})

# find MPI
find_package(MPI REQUIRED)
if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
endif()

# Need MLPACK
find_path(MLPACK_INCLUDE_DIR mlpack/core.hpp HINTS ${MLPACK_INCLUDEDIR})
find_library(mlpack NAMES mlpack libmlpack HINTS ${MLPACK_LIBDIR})

if (${MLPACK_INCLUDE_DIR} STREQUAL "")
  message(FATAL_ERROR
      "Don't know where to find MLPACK include files.\nTry specifying
      -D MLPACK_INCLUDEDIR=/path/to/mlpack/include/")
endif ()

include_directories(${MLPACK_INCLUDE_DIR})
include_directories("/usr/include/libxml2/")
link_directories(${MLPACK_LIBDIR})

# Cmake requires this variable, and it will need to be updated for newer 
# versions of boost
set(Boost_ADDITIONAL_VERSIONS
  "1.41" "1.41.0" "1.42" "1.42.0" "1.43" "1.43.0" "1.44" "1.44.0" "1.45.0"
  "1.46.0" "1.46.1" "1.47.0" "1.48.0" "1.49.0" "1.50" "1.50.0" "1.51" "1.51.0")
find_package(Boost
    COMPONENTS
      unit_test_framework
      mpi
      program_options
      serialization
    REQUIRED
)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIB_DIR})

# I think I need this for the unit tests
add_definitions(-DBOOST_TEST_DYN_LINK)

# need openmp
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")

# On Windows, things end up under Debug/ or Release/.
if (WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else (WIN32)
  # If not on Windows, put them under more standard UNIX-like places.  This is
  # necessary, otherwise they would all end up in
  # ${CMAKE_BINARY_DIR}/src/mlpack/methods/... or somewhere else random like
  # that.
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
endif (WIN32)

# Add each subdirectory.
foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(npoint_mlpack SHARED ${NPOINT_MLPACK_SRCS})
target_link_libraries(npoint_mlpack
  mlpack
  ${Boost_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${MPI_LIBRARIES}
)

add_executable(3_point_main
  3_point_main.cpp
)
target_link_libraries(3_point_main
    npoint_mlpack
)
add_executable(2_point_main
  2_point_main.cpp
)
target_link_libraries(2_point_main
    npoint_mlpack
)
add_executable(4_point_main
  4_point_main.cpp
)
target_link_libraries(4_point_main
    npoint_mlpack
)

add_executable(resampling_sizes
  resampling_set_sizes.cpp
)
target_link_libraries(resampling_sizes
    npoint_mlpack
)

add_executable(angle_3pt_main
  angle_3pt_main.cpp
)
target_link_libraries(angle_3pt_main
    npoint_mlpack
)

add_executable(distributed_4pt_main
    distributed_4pt_main.cpp
)
target_link_libraries(distributed_4pt_main
    npoint_mlpack
)

add_executable(distributed_3pt_main
    distributed_3pt_main.cpp
)
target_link_libraries(distributed_3pt_main
    npoint_mlpack
)

add_executable(distributed_2pt_main
    distributed_2pt_main.cpp
)
target_link_libraries(distributed_2pt_main
    npoint_mlpack
)

add_executable(distributed_angle_main
    distributed_angle_main.cpp
)
target_link_libraries(distributed_angle_main
    npoint_mlpack
)

add_executable(distributed_angle_2pt_main
    distributed_angle_2pt.cpp
)
target_link_libraries(distributed_angle_2pt_main
    npoint_mlpack
)

add_executable(multi_matcher_main
  multi_matcher_main.cpp
)
target_link_libraries(multi_matcher_main
  npoint_mlpack
)

add_executable(distributed_multi_matcher_main
  distributed_multi_matcher_main.cpp
)
target_link_libraries(distributed_multi_matcher_main
  npoint_mlpack
)





