#infrastructure
cmake_minimum_required(VERSION 2.6)

set(SOURCES
  efficient_resampling_driver.hpp
  efficient_resampling_driver_impl.hpp
  naive_resampling_driver.hpp
  naive_resampling_driver_impl.hpp
  resampling_splitter.hpp
  resampling_splitter.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(NPOINT_MLPACK_SRCS ${NPOINT_MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
