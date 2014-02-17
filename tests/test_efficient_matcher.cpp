//
//  test_efficient_matcher.cpp
//  contrib_march
//
//  Created by William March on 7/10/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include <mlpack/core.hpp>

#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../matchers/single_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../matchers/efficient_2pt_matcher.hpp"
#include "../matchers/efficient_4pt_matcher.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/pairwise_npt_traversal.hpp"
#include "../results/single_results.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include <omp.h>

#include <boost/test/unit_test.hpp>

using namespace npoint_mlpack;

BOOST_AUTO_TEST_SUITE(EfficientCpuMatcherTests);

BOOST_AUTO_TEST_CASE(EfficientCpuMatcher_vs_SingleMatcher)
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
  
  MatcherArguments single_args(matcher_dists, matcher_thick);
  MatcherArguments efficient_args(matcher_dists, matcher_thick);
  
  // copy the data 
  arma::mat efficient_data(data_mat);
  arma::mat single_data(data_mat);
  arma::mat efficient_randoms(random_mat);
  arma::mat single_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it)
  int num_regions = 1;
  
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
  
  
  NaiveResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>, 
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
    
} // EfficientCpuMatcher vs SingleMatcher

BOOST_AUTO_TEST_CASE(Efficient2ptMatcher_vs_SingleMatcher)
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
  int tuple_size = 2;
  
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
  
  MatcherArguments single_args(matcher_dists, matcher_thick);
  MatcherArguments efficient_args(matcher_dists, matcher_thick);
  
  // copy the data
  arma::mat efficient_data(data_mat);
  arma::mat single_data(data_mat);
  arma::mat efficient_randoms(random_mat);
  arma::mat single_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it)
  int num_regions = 1;
  
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
  
  
  NaiveResamplingDriver<Efficient2ptMatcher, GenericNptAlg<Efficient2ptMatcher>,
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
  
} // EfficientCpuMatcher vs SingleMatcher

BOOST_AUTO_TEST_CASE(Efficient4ptMatcher_vs_SingleMatcher)
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
  
  int tuple_size = 4;
  
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
  
  //std::cout << "matcher dists: " << matcher_dists << "\n";
  
  MatcherArguments single_args(matcher_dists, matcher_thick);
  MatcherArguments efficient_args(matcher_dists, matcher_thick);
  
  // copy the data
  arma::mat efficient_data(data_mat);
  arma::mat single_data(data_mat);
  arma::mat efficient_randoms(random_mat);
  arma::mat single_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it)
  int num_regions = 1;
  
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
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
  
  
  EfficientResamplingDriver<Efficient4ptMatcher, GenericNptAlg<Efficient4ptMatcher>,
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
  
} // EfficientCpuMatcher vs SingleMatcher


BOOST_AUTO_TEST_SUITE_END();


