#infrastructure
cmake_minimum_required(VERSION 2.6)

set(SOURCES
  angle_results.hpp
  angle_results.cpp
  multi_results.hpp
  multi_results.cpp
  single_results.cpp
  single_results.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(NPOINT_MLPACK_SRCS ${NPOINT_MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
