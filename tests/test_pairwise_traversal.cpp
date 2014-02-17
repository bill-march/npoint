//
//  test_pairwise_traversal.cpp
//  contrib_march
//
//  Created by William March on 7/4/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include <mlpack/core.hpp>

#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../matchers/single_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/pairwise_npt_traversal.hpp"
#include "../results/single_results.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <omp.h>

using namespace npoint_mlpack;

BOOST_AUTO_TEST_SUITE(PairwiseTraversalTests);

BOOST_AUTO_TEST_CASE(pairwise_traversal_vs_generic_alg)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  // can only do size 3 for pairwise traversal for now
  int tuple_size = 3;
  
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
  
  MatcherArguments multi_args(matcher_dists, matcher_thick);
  MatcherArguments pairwise_args(matcher_dists, matcher_thick);
  
  // copy the data 
  arma::mat multi_data(data_mat);
  arma::mat pairwise_data(data_mat);
  arma::mat multi_randoms(random_mat);
  arma::mat pairwise_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it)
  int num_regions = 1;
  
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
  omp_set_num_threads(1);
  
  // pairwise traversal
  NaiveResamplingDriver<SingleMatcher, PairwiseNptTraversal<SingleMatcher>, 
  NptNode, SingleResults>
  pairwise_alg(pairwise_data, 
            data_weights,
            pairwise_randoms, 
            random_weights,
            pairwise_args,
            num_regions,
            num_regions,
            num_regions,
            helper,
               tuple_size,
            leaf_size);
  
  
  pairwise_alg.Compute();
  
  SingleResults pairwise_results = pairwise_alg.results();
  
  
  // multi tree traversal
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>, 
  NptNode, SingleResults>
  multi_alg(multi_data, 
           data_weights,
           multi_randoms, 
           random_weights,
           multi_args,         
           num_regions,
           num_regions,
           num_regions,
           helper,
            tuple_size,
           leaf_size);
  
  multi_alg.Compute();
  
  SingleResults multi_results = multi_alg.results();
  
  BOOST_REQUIRE(multi_results == pairwise_results);
      
} // StressTest


// Run the test again using the pairwise alg and large NodeC optimization
BOOST_AUTO_TEST_CASE(pairwise_traversal_efficient_matcher)
{
  
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  
  // only does 3-point for now
  int tuple_size = 3;
  
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
  
  MatcherArguments single_args(matcher_dists, matcher_thick);
  MatcherArguments efficient_args(matcher_dists, matcher_thick);
  
  // copy the data
  arma::mat efficient_data(data_mat);
  arma::mat single_data(data_mat);
  arma::mat efficient_randoms(random_mat);
  arma::mat single_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it
  int num_regions = 1;
  
  // Now, do single
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                        NptNode, SingleResults>
  single_alg(single_data,
             data_weights,
             single_randoms,
             random_weights,
             single_args,
             num_regions,
             num_regions,
             num_regions,
             helper,
             tuple_size,
             leaf_size);
  
  
  single_alg.Compute();
  
  SingleResults single_results = single_alg.results();
  
  
  NaiveResamplingDriver<EfficientCpuMatcher,
                        PairwiseNptTraversal<EfficientCpuMatcher>,
                        NptNode, SingleResults>
  efficient_alg(efficient_data,
                data_weights,
                efficient_randoms,
                random_weights,
                efficient_args,
                num_regions,
                num_regions,
                num_regions,
                helper,
                tuple_size,
                leaf_size);
  
  efficient_alg.Compute();
  
  SingleResults efficient_results = efficient_alg.results();
  
  BOOST_REQUIRE(efficient_results == single_results);
  
} // test pairwise efficient cpu matcher



BOOST_AUTO_TEST_SUITE_END();

