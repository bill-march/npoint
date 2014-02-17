#infrastructure
cmake_minimum_required(VERSION 2.6)

set(SOURCES
  generate_random_problem.cpp
  generate_random_problem.hpp
  generic_npt_alg_impl.hpp
  generic_npt_alg.hpp
  node_tuple.cpp
  node_tuple.hpp
  pairwise_npt_traversal.hpp
  pairwise_npt_traversal_impl.hpp
  permutations.cpp
  permutations.hpp
  resampling_helper.hpp
  resampling_helper.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(NPOINT_MLPACK_SRCS ${NPOINT_MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
