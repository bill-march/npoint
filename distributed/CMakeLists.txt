#infrastructure
cmake_minimum_required(VERSION 2.6)

set(SOURCES
  arma_serialization.hpp
  distributed_npt_driver.hpp
  distributed_npt_driver_impl.hpp
  distributed_npt_driver.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(NPOINT_MLPACK_SRCS ${NPOINT_MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)


