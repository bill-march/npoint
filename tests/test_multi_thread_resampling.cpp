//
//  test_multi_thread_resampling.cpp
//  contrib_march
//
//  Created by William March on 6/18/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include <mlpack/core.hpp>

#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../matchers/single_matcher.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../results/single_results.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../matchers/multi_matcher.hpp"
#include "../results/multi_results.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <omp.h>

using namespace npoint_mlpack;

BOOST_AUTO_TEST_SUITE(MultiThreadResamplingTest);

BOOST_AUTO_TEST_CASE(TestMultiThreadedNaiveResampling)
{

  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 4;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;

  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  int tuple_size = mlpack::math::RandInt(tuple_size_lo, tuple_size_hi);
  
  GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                    matcher_thick_lo, matcher_thick_hi,
                                    num_data_lo, num_data_hi);

  // Generate data set
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
  arma::mat single_data(data_mat);
  arma::mat multi_data(data_mat);
  arma::mat single_randoms(random_mat);
  arma::mat multi_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it)
  int num_regions = 1;
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
  // need to do this because this function will get called many times
  omp_set_num_threads(4);
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                        NptNode, SingleResults>
  multi_alg(multi_data, 
           data_weights,
           multi_randoms, 
           random_weights,
           matcher_args,         
           num_regions,
           num_regions,
           num_regions,
           helper,
            tuple_size,
           leaf_size);
   
  multi_alg.Compute();
  
  SingleResults multi_results = multi_alg.results();
  
  
  // Now, do naive
  omp_set_num_threads(1);
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                        NptNode, SingleResults>
  single_alg(single_data, 
            data_weights,
            single_randoms, 
            random_weights,
            matcher_args,
            num_regions,
            num_regions,
            num_regions,
            helper,
             tuple_size,
            leaf_size);
  
  single_alg.Compute();
  
  SingleResults single_results = single_alg.results();
  
  BOOST_REQUIRE(single_results == multi_results);
  
} // StressTest

BOOST_AUTO_TEST_CASE(TestMultiThreadedEfficientResampling)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 4;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  int tuple_size = mlpack::math::RandInt(tuple_size_lo, tuple_size_hi);
  
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
  arma::mat efficient_data(data_mat);
  arma::mat naive_randoms(random_mat);
  arma::mat efficient_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it
  int num_regions = 1;
  
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
    
  // need to do this because this function will get called many times
  omp_set_num_threads(4);
  
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>, 
                            NptNode, SingleResults>
  efficient_alg(efficient_data, 
            data_weights,
            efficient_randoms, 
            random_weights,
            matcher_args,         
            num_regions,
            num_regions,
            num_regions,
            helper,
                tuple_size,
            leaf_size);
  
  
  efficient_alg.Compute();
  
  SingleResults efficient_results = efficient_alg.results();
  
  
  // Now, do single thread
  omp_set_num_threads(1);

  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                            NptNode, SingleResults>
  naive_alg(naive_data, 
             data_weights,
             naive_randoms, 
             random_weights,
             matcher_args,
             num_regions,
             num_regions,
             num_regions,
            helper,
            tuple_size,
             leaf_size);
  
  naive_alg.Compute();
  
  SingleResults naive_results = naive_alg.results();
  
  BOOST_REQUIRE(naive_results == efficient_results);
    
} // TestEfficient

/**
 * We can compute a multi matcher by iterating through all it's single 
 * matchers.  We test doing this in parallel here.
 */
BOOST_AUTO_TEST_CASE(threaded_iterative_multi_matcher_efficient)
{
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 5;
  
  double min_band_lo = 0.05;
  double min_band_hi = 0.1;
  double max_band_lo = 0.12;
  double max_band_hi = 0.25;
  int num_bands_lo = 1;
  int num_bands_hi = 4;
  
  int num_regions = 1;
  
  GenerateRandomProblem problem_gen(min_band_lo, max_band_hi,
                                    min_band_hi, max_band_lo,
                                    num_data_lo, num_data_hi);
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  
  arma::mat data_mat(num_dimensions, num_data_points);
  problem_gen.GenerateRandomSet(data_mat);
  arma::colvec data_weights(num_data_points);
  data_weights.fill(1.0);
  
  ResamplingHelper helper(data_mat);
  
  // Generate a random random set
  arma::mat random_mat(num_dimensions, num_random_points);
  problem_gen.GenerateRandomSet(random_mat);
  arma::colvec random_weights(num_random_points);
  random_weights.fill(1.0);
  
  int tuple_size = mlpack::math::RandInt(tuple_size_lo, tuple_size_hi);
  int tuple_size_choose_2 = (tuple_size - 1) * tuple_size / 2;
  
  arma::mat single_data(data_mat);
  arma::mat single_random(random_mat);
  
  arma::mat multi_data(data_mat);
  arma::mat multi_random(random_mat);
  
  // Generate matcher info
  arma::mat matcher_mat(tuple_size_choose_2,3);
  int total_num_matchers = 1;
  for (int i = 0; i < tuple_size_choose_2; i++) {
    
    matcher_mat.at(i,0) = mlpack::math::Random(min_band_lo, min_band_hi);
    matcher_mat.at(i,1) = mlpack::math::Random(max_band_lo, max_band_hi);
    matcher_mat.at(i,2) = mlpack::math::RandInt(num_bands_lo, num_bands_hi);
    
    total_num_matchers *= (int)matcher_mat.at(i,2);
    
  }
  
  MatcherArguments single_args(matcher_mat, tuple_size);
  single_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
  
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, MultiResults>
  single_alg(single_data,
             data_weights,
             single_random,
             random_weights,
             single_args,
             num_regions,
             num_regions,
             num_regions,
             helper,
             tuple_size,
             16);
  
  single_alg.Compute();
  
  MultiResults single_results = single_alg.results();
  
  omp_set_num_threads(4);
  
  // Now, do the multi matcher algorithm
  MatcherArguments multi_args(matcher_mat, tuple_size);
  multi_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<MultiMatcher, GenericNptAlg<MultiMatcher>,
  NptNode, MultiResults>
  multi_alg(multi_data,
            data_weights,
            multi_random,
            random_weights,
            multi_args,
            num_regions,
            num_regions,
            num_regions,
            helper,
            tuple_size,
            16);
  
  multi_alg.Compute();
  
  MultiResults multi_results = multi_alg.results();
  
  BOOST_REQUIRE(single_results == multi_results);
    
} // test iterating through a multi-matcher in parallel


BOOST_AUTO_TEST_SUITE_END();



