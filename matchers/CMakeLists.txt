#infrastructure
cmake_minimum_required(VERSION 2.6)

set(SOURCES
  2_point_single_matcher.hpp
  2_point_single_matcher.cpp
  3_point_single_matcher.hpp
  3_point_single_matcher.cpp
  angle_matcher.hpp
  angle_matcher.cpp
  angle_matcher_generator.hpp
  angle_matcher_generator.cpp
  efficient_angle_matcher.hpp
  efficient_angle_matcher.cpp
  efficient_cpu_matcher.hpp
  efficient_cpu_matcher.cpp
  efficient_2pt_matcher.hpp
  efficient_2pt_matcher.cpp
  efficient_4pt_matcher.hpp
  efficient_4pt_matcher.cpp
  efficient_multi_matcher.hpp
  efficient_multi_matcher.cpp
  matcher_arguments.hpp
  matcher_arguments.cpp
  multi_matcher.hpp
  multi_matcher.cpp
  multi_matcher_generator.hpp
  multi_matcher_generator.cpp
  single_matcher.cpp
  single_matcher.hpp
  unordered_multi_matcher.hpp
  unordered_multi_matcher.cpp
  unordered_efficient_multi_matcher.hpp
  unordered_efficient_multi_matcher.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(NPOINT_MLPACK_SRCS ${NPOINT_MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
