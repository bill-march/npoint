//
//  test_resampling.cpp
//  contrib_march
//
//  Created by William March on 7/9/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include <boost/test/unit_test.hpp>
#include <mlpack/core.hpp>

#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../resampling_classes/resampling_splitter.hpp"
#include "../matchers/single_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/pairwise_npt_traversal.hpp"
#include "../results/single_results.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include <omp.h>

using namespace npoint_mlpack;

BOOST_AUTO_TEST_SUITE(ResamplingClassesTests);

/*
BOOST_AUTO_TEST_CASE(naive_resampling_manual_split_test)
{
  
  BOOST_TEST_MESSAGE("Running manual split test.  Note: this uses the efficient CPU matcher.");
  
  arma::mat inputData, inputRandoms;
  
  inputData.load("test_data_1000.csv", arma::raw_ascii);
  inputRandoms.load("test_random_1000.csv", arma::raw_ascii);
  
  arma::mat resampling_data = arma::trans(inputData);
  
  ResamplingHelper helper(resampling_data);
  
  arma::mat resampling_randoms = arma::trans(inputRandoms);
  arma::mat single_randoms = arma::trans(inputRandoms);
  
  arma::colvec resampling_weights(resampling_data.n_cols);
  resampling_weights.fill(1.0);
  
  arma::colvec resampling_random_weights(resampling_randoms.n_cols);
  resampling_random_weights.fill(1.0);
  arma::colvec single_random_weights(resampling_randoms.n_cols);
  single_random_weights.fill(1.0);
  
  int tuple_size = 3;
  
  arma::mat matcher_dists(tuple_size, tuple_size);
  
  matcher_dists(0,0) = 0.0;
  matcher_dists(0,1) = 0.15;
  matcher_dists(0,2) = 0.25;
  
  matcher_dists(1,0) = 0.15;
  matcher_dists(1,1) = 0.0;
  matcher_dists(1,2) = 0.2;
  
  matcher_dists(2,0) = 0.25;
  matcher_dists(2,1) = 0.2;
  matcher_dists(2,2) = 0.0;
  
  double matcher_thick = 0.1;
  
  MatcherArguments matcher_args(matcher_dists, matcher_thick);
  
  // don't change this without changing the number of files loaded etc.
  int num_regions = 2;
  
  int leaf_size = 16;
  
  // Only testing 1 thread
  omp_set_num_threads(1);
  
  NaiveResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>,
                        NptNode, SingleResults>
  resampling_alg(resampling_data,
                 resampling_weights,
                 resampling_randoms,
                 resampling_random_weights,
                 matcher_args,
                 num_regions,
                 1, 1,
                 helper,
                 tuple_size,
                 leaf_size);
  
  resampling_alg.Compute();
  
  std::cout << "Splitter sizes:\n";
  resampling_alg.PrintRegionSizes();
  
  SingleResults resampling_results = resampling_alg.results();
  
  ///// Do the single computation
  
  // make sure it's a single thread
  omp_set_num_threads(1);
  
  arma::mat data1In, data2In;
  
  data1In.load("test_resampling_data_1.csv", arma::raw_ascii);
  data2In.load("test_resampling_data_2.csv", arma::raw_ascii);
  
  arma::mat single_data_1 = arma::trans(data1In);
  arma::mat single_data_2 = arma::trans(data2In);
  
  std::cout << "Naive sizes: " << single_data_1.n_cols << ", " << single_data_2.n_cols << "\n";
  
  ResamplingHelper helper1(single_data_1);
  ResamplingHelper helper2(single_data_2);
  
  arma::colvec single_weights_1(single_data_1.n_cols);
  single_weights_1.fill(1.0);
  arma::colvec single_weights_2(single_data_2.n_cols);
  single_weights_2.fill(1.0);
  
  NaiveResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>,
                        NptNode, SingleResults>
  single_alg_1(single_data_1,
               single_weights_1,
               single_randoms,
               single_random_weights,
               matcher_args,
               1, 1, 1,
               helper1,
               tuple_size,
               leaf_size);
  
  single_alg_1.Compute();
  
  SingleResults single_results_1 = single_alg_1.results();
  
  
  NaiveResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>,
  NptNode, SingleResults>
  single_alg_2(single_data_2,
               single_weights_2,
               single_randoms,
               single_random_weights,
               matcher_args,
               1, 1, 1,
               helper2,
               tuple_size,
               leaf_size);
  
  single_alg_2.Compute();
  
  SingleResults single_results_2 = single_alg_2.results();
  
  
  
  // compare data
  for (int i = 0; i < tuple_size; i++) {
    
    BOOST_REQUIRE_EQUAL(resampling_results.results()[0][i],
                        single_results_2.results()[0][i]);
    
    BOOST_REQUIRE_EQUAL(resampling_results.results()[1][i],
                        single_results_1.results()[0][i]);
    
    
  } // loop over data results
  
  // Check RRR results
  BOOST_REQUIRE_EQUAL(resampling_results.RRR_result(),
                      single_results_1.RRR_result());
  
  BOOST_REQUIRE_EQUAL(resampling_results.RRR_result(),
                      single_results_2.RRR_result());
  
}
 */

BOOST_AUTO_TEST_CASE(test_efficient_vs_naive)
{

  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 3;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  int num_regions_lo = 1;
  int num_regions_hi = 4;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;

  // loop over tuple sizes
  for (int tuple_size = tuple_size_lo; tuple_size <= tuple_size_hi;
       tuple_size++)
  {
    
    GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                      matcher_thick_lo, matcher_thick_hi,
                                      num_data_lo, num_data_hi);
    
    
    // Generate data set
    arma::mat data_mat(num_dimensions, num_data_points);
    problem_gen.GenerateRandomSet(data_mat);
    arma::colvec data_weights(num_data_points);
    
    ResamplingHelper helper(data_mat);
    
    // Generate a random set
    arma::mat random_mat(num_dimensions, num_random_points);
    problem_gen.GenerateRandomSet(random_mat);
    arma::colvec random_weights(num_random_points);

    
    // Generate a random matcher and matcher thickness multiplier
    arma::mat matcher_dists(tuple_size, tuple_size);
    double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
    
    MatcherArguments matcher_args(matcher_dists, matcher_thick);
    
    // copy the data 
    arma::mat naive_data(data_mat);
    arma::mat efficient_data(data_mat);
    arma::mat naive_randoms(random_mat);
    arma::mat efficient_randoms(random_mat);
    
    int num_x_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_y_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_z_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    
    int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
    
    // make sure we don't test multi-threading here
    omp_set_num_threads(1);
    
    EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                              NptNode, SingleResults>
    efficient_alg(efficient_data, 
                  data_weights,
                  efficient_randoms, 
                  random_weights,
                  matcher_args,         
                  num_x_regions,
                  num_y_regions,
                  num_z_regions,
                  helper,
                  tuple_size,
                  leaf_size);
    
    
    efficient_alg.Compute();
    
    SingleResults efficient_results = efficient_alg.results();
    
    
    // Now, do naive

    omp_set_num_threads(1);
    
    NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                          NptNode, SingleResults>
    naive_alg(naive_data, 
              data_weights,
              naive_randoms, 
              random_weights,
              matcher_args,
              num_x_regions,
              num_y_regions,
              num_z_regions,
              helper,
              tuple_size,
              leaf_size);
    
    
    naive_alg.Compute();
    
    SingleResults naive_results = naive_alg.results();
    
    BOOST_REQUIRE(naive_results == efficient_results);
    
  } // loop over tuple sizes
  
} // TestEfficient

/*
BOOST_AUTO_TEST_CASE(test_resampling_no_randoms)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 3;
  int tuple_size_hi = 3;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  int num_regions_lo = 1;
  int num_regions_hi = 4;

  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = 0;
  int num_dimensions = 3;

  for (int tuple_size = tuple_size_lo;
       tuple_size <= tuple_size_hi;
       tuple_size++)
  {

    GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                      matcher_thick_lo, matcher_thick_hi,
                                      num_data_lo, num_data_hi);
    
    
    // Generate data set
    arma::mat data_mat(num_dimensions, num_data_points);
    problem_gen.GenerateRandomSet(data_mat);
    arma::colvec data_weights(num_data_points);
    
    // Generate empty random set
    arma::mat random_mat(num_dimensions, num_random_points);
    arma::colvec random_weights(num_random_points);
    
    // Generate a random matcher and matcher thickness multiplier
    arma::mat matcher_dists(tuple_size, tuple_size);
    double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
    
    MatcherArguments matcher_args(matcher_dists, matcher_thick);
    
    // copy the data 
    arma::mat naive_data(data_mat);
    arma::mat efficient_data(data_mat);
    arma::mat naive_randoms(random_mat);
    arma::mat efficient_randoms(random_mat);
    
    int num_x_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_y_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_z_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    
    double box_length = 1.0;
    
    int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
    
    // need to do this because this function will get called many times
    omp_set_num_threads(1);
    
    //NaiveResamplingDriver<SingleMatcher, PairwiseNptTraversal<SingleMatcher>,
    EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>, 
                              NptNode, SingleResults>
    efficient_alg(efficient_data, 
                  data_weights,
                  efficient_randoms, 
                  random_weights,
                  matcher_args,         
                  num_x_regions,
                  num_y_regions,
                  num_z_regions,
                  box_length,
                  box_length,
                  box_length,
                  tuple_size,
                  leaf_size);
    
    
    efficient_alg.Compute();
    
    SingleResults efficient_results = efficient_alg.results();
    
    
    // Now, do naive
    omp_set_num_threads(1);
    
    NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                          NptNode, SingleResults>
    naive_alg(naive_data, 
              data_weights,
              naive_randoms, 
              random_weights,
              matcher_args,
              num_x_regions,
              num_y_regions,
              num_z_regions,
              box_length,
              box_length,
              box_length,
              tuple_size,
              leaf_size);
    
    
    naive_alg.Compute();
    
    SingleResults naive_results = naive_alg.results();
    
    BOOST_REQUIRE(naive_results == efficient_results);

  } // loop over tuple size
  
} // TestNoRandoms
 */

BOOST_AUTO_TEST_CASE(test_resampling_splitter)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 3;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  int num_regions_lo = 1;
  int num_regions_hi = 4;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;

  for (int tuple_size = tuple_size_lo;
       tuple_size <= tuple_size_hi;
       tuple_size++)
  {
    
    GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                      matcher_thick_lo, matcher_thick_hi,
                                      num_data_lo, num_data_hi);
    
    
    // Generate data set
    arma::mat data_mat(num_dimensions, num_data_points);
    problem_gen.GenerateRandomSet(data_mat);
    arma::colvec data_weights(num_data_points);
    
    ResamplingHelper helper(data_mat);
    
    // Generate random set
    arma::mat random_mat(num_dimensions, num_random_points);
    problem_gen.GenerateRandomSet(random_mat);
    arma::colvec random_weights(num_random_points);
    
    // Generate a random matcher and matcher thickness multiplier
    arma::mat matcher_dists(tuple_size, tuple_size);
    double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
    
    MatcherArguments matcher_args(matcher_dists, matcher_thick);
    
    // copy the data
    arma::mat naive_data(data_mat);
    arma::mat split_data(data_mat);
    arma::mat naive_randoms(random_mat);
    arma::mat split_randoms(random_mat);
    
    int num_x_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_y_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_z_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    
    int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
    
    ResamplingSplitter splitter(split_data, data_weights,
                                num_x_regions,
                                num_y_regions,
                                num_z_regions,
                                helper);
    
    
    
    // need to do this because this function will get called many times
    omp_set_num_threads(1);
    

    NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                          NptNode, SingleResults>
    naive_alg(naive_data,
              data_weights,
              naive_randoms,
              random_weights,
              matcher_args,
              num_x_regions,
              num_y_regions,
              num_z_regions,
              helper,
              tuple_size,
              leaf_size);
    
    
    naive_alg.Compute();
    
    SingleResults naive_results = naive_alg.results();
    
    // now, do the pre-split data
    NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
    NptNode, SingleResults>
    split_alg(splitter.data_mats(),
              splitter.data_weights(),
              split_randoms,
              random_weights,
              matcher_args,
              tuple_size,
              leaf_size);
    
    
    split_alg.Compute();
    
    SingleResults split_results = split_alg.results();
    
    BOOST_REQUIRE(split_results == naive_results);
    
  } // loop over tuple size
  
} // TestResampingSplitter

BOOST_AUTO_TEST_CASE(test_resampling_splitter_efficient)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 3;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  int num_regions_lo = 1;
  int num_regions_hi = 4;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;

  for (int tuple_size = tuple_size_lo;
       tuple_size <= tuple_size_hi;
       tuple_size++)
  {
    
    GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                      matcher_thick_lo, matcher_thick_hi,
                                      num_data_lo, num_data_hi);
    
    
    // Generate a random data set
    arma::mat data_mat(num_dimensions, num_data_points);
    problem_gen.GenerateRandomSet(data_mat);
    arma::colvec data_weights(num_data_points);
    
    ResamplingHelper helper(data_mat);
    
    // Generate a random random set
    arma::mat random_mat(num_dimensions, num_random_points);
    problem_gen.GenerateRandomSet(random_mat);
    arma::colvec random_weights(num_random_points);
    
    // Generate a random matcher and matcher thickness multiplier
    arma::mat matcher_dists(tuple_size, tuple_size);
    double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
    
    MatcherArguments matcher_args(matcher_dists, matcher_thick);
    
    // copy the data
    arma::mat naive_data(data_mat);
    arma::mat split_data(data_mat);
    arma::mat naive_randoms(random_mat);
    arma::mat split_randoms(random_mat);
    
    int num_x_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_y_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    int num_z_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
    
    int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
    
    ResamplingSplitter splitter(split_data, data_weights,
                                num_x_regions,
                                num_y_regions,
                                num_z_regions,
                                helper);
    
    // need to do this because this function will get called many times
    omp_set_num_threads(1);
    
    
    EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                              NptNode, SingleResults>
    naive_alg(naive_data,
              data_weights,
              naive_randoms,
              random_weights,
              matcher_args,
              num_x_regions,
              num_y_regions,
              num_z_regions,
              helper,
              tuple_size,
              leaf_size);
    
    
    naive_alg.Compute();
    
    SingleResults naive_results = naive_alg.results();
    
    
    // now use the splitter
    EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                              NptNode, SingleResults>
    split_alg(splitter.data_mats(),
              splitter.data_weights(),
              split_randoms,
              random_weights,
              matcher_args,
              tuple_size,
              leaf_size);
    
    
    split_alg.Compute();
    
    SingleResults split_results = split_alg.results();
    
    BOOST_REQUIRE(split_results == naive_results);
    
  } // loop over tuple size
  
} // TestResampingSplitterEfficient

BOOST_AUTO_TEST_CASE(test_resampling_helper)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  
  int tuple_size = 2;
  
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  int num_regions_lo = 1;
  int num_regions_hi = 4;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  
  GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                    matcher_thick_lo, matcher_thick_hi,
                                    num_data_lo, num_data_hi);
  
  
  // Generate a random data set
  arma::mat data_mat(num_dimensions, num_data_points);
  problem_gen.GenerateRandomSet(data_mat);
  arma::colvec data_weights(num_data_points);
  
  // now, put the data in a different box
  
  arma::colvec scale_col(3);
  scale_col << 0.5 << 1.5 << 1.25;
 
  arma::colvec add_col(3);
  add_col << 5.9 << 4.2 << 2.4;
  
  for (size_t i = 0; i < data_mat.n_cols; i++) {
    data_mat.col(i) = data_mat.col(i) % scale_col;
    data_mat.col(i) += add_col;
  }

  
  ResamplingHelper helper(data_mat);
  
  // Generate a random random set
  arma::mat random_mat(num_dimensions, num_random_points);
  problem_gen.GenerateRandomSet(random_mat);
  arma::colvec random_weights(num_random_points);
  
  for (size_t i = 0; i < random_mat.n_cols; i++) {
    random_mat.col(i) = random_mat.col(i) % scale_col;
    random_mat.col(i) += add_col;
  }
  
  // Generate a random matcher and matcher thickness multiplier
  arma::mat matcher_dists(tuple_size, tuple_size);
  double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
  
  MatcherArguments matcher_args(matcher_dists, matcher_thick);
  
  // copy the data
  arma::mat naive_data(data_mat);
  arma::mat split_data(data_mat);
  arma::mat naive_randoms(random_mat);
  arma::mat split_randoms(random_mat);
  
  int num_x_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
  int num_y_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
  int num_z_regions = mlpack::math::RandInt(num_regions_lo, num_regions_hi);
  
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
  ResamplingSplitter splitter(split_data, data_weights,
                              num_x_regions,
                              num_y_regions,
                              num_z_regions,
                              helper);
  
  // need to do this because this function will get called many times
  omp_set_num_threads(1);
  
  
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  naive_alg(naive_data,
            data_weights,
            naive_randoms,
            random_weights,
            matcher_args,
            num_x_regions,
            num_y_regions,
            num_z_regions,
            helper,
            tuple_size,
            leaf_size);
  
  
  naive_alg.Compute();
  
  SingleResults naive_results = naive_alg.results();
  
  
  // now use the splitter
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  split_alg(splitter.data_mats(),
            splitter.data_weights(),
            split_randoms,
            random_weights,
            matcher_args,
            tuple_size,
            leaf_size);
  
  
  split_alg.Compute();
  
  SingleResults split_results = split_alg.results();
  
  BOOST_REQUIRE(split_results == naive_results);
  

} // TestResampingSplitterEfficient


BOOST_AUTO_TEST_SUITE_END();

