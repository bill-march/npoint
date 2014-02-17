#infrastructure
cmake_minimum_required(VERSION 2.6)

add_executable(npoint_mlpack_test
  npoint_test_overall.cpp
  test_single_matcher.cpp
  test_resampling.cpp
  test_efficient_matcher.cpp
  test_pairwise_traversal.cpp
  test_multi_matcher.cpp
  test_angle_matcher.cpp
  test_efficient_angle_matcher.cpp
  #test_multi_thread_resampling.cpp
  #test_parallel_traversal.cpp
)
target_link_libraries(npoint_mlpack_test
  npoint_mlpack
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
)

add_executable(distributed_npoint_test
  distributed_npoint_test.cpp
)
target_link_libraries(distributed_npoint_test
  npoint_mlpack
)

add_executable(distributed_angle_test
  distributed_angle_test.cpp
)
target_link_libraries(distributed_angle_test
  npoint_mlpack
)

# Copy test data into right place.
add_custom_command(TARGET npoint_mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
      ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
