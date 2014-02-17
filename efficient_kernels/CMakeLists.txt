#infrastructure
cmake_minimum_required(VERSION 2.6)

set(SOURCES
  common.hpp
  efficient_3pt_multi_kernel.hpp
  kernels_npt_cpu.cpp
  kernels_npt_cpu.cpp.kern0.hpp
  kernels_npt_cpu.cpp.kern1.hpp
  kernels_npt_cpu.cpp.kern2.hpp
  kernels_npt_cpu.cpp.kern4.hpp
  kernels_npt_cpu.cpp.kern5.hpp
  kernels_npt_cpu.hpp
  kernels_npt_cpu_old.hpp
  timing.hpp
  timing.cpp
  kernel_2pt_cpu.cpp
  kernel_2pt_cpu.hpp
  kernel_4pt_cpu.hpp
  kernel_4pt_cpu.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(NPOINT_MLPACK_SRCS ${NPOINT_MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
