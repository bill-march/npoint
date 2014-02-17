//
//  test_multi_matcher.cpp
//  
//
//  Created by William March on 1/19/12.
//  Copyright (c) 2012 Georgia Institute of Technology. All rights reserved.
//

#include <mlpack/core.hpp>

#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../matchers/single_matcher.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/pairwise_npt_traversal.hpp"
#include "../results/multi_results.hpp"
#include "../results/single_results.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../matchers/multi_matcher.hpp"
#include "../matchers/unordered_multi_matcher.hpp"
#include "../matchers/unordered_efficient_multi_matcher.hpp"
#include "../matchers/efficient_multi_matcher.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <omp.h>

using namespace npoint_mlpack;

BOOST_AUTO_TEST_SUITE(TestMultiMatcher);

BOOST_AUTO_TEST_CASE(multi_matcher_vs_single_matcher_naive_resampling)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 4;

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
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>, 
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
  
  // Now, do the multi matcher algorithm
  MatcherArguments multi_args(matcher_mat, tuple_size);
  multi_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  NaiveResamplingDriver<MultiMatcher, GenericNptAlg<MultiMatcher>, 
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
    
} // test multi vs. single (naive resampling)

BOOST_AUTO_TEST_CASE(multi_matcher_vs_single_matcher_efficient_resampling)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 4;
  
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
  
  //mlpack::Log::Info << "\n\nSingle results: \n";
  //single_results.PrintResults();
  
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
  
  //mlpack::Log::Info << "\n\nMulti Results:\n";
  //multi_results.PrintResults();
  
  BOOST_REQUIRE(single_results == multi_results);
    
} // test multi vs. single (efficient resampling)

BOOST_AUTO_TEST_CASE(multi_matcher_vs_single_matcher_pairwise_traversal)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  
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
  
  int tuple_size = 3;
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
  
  EfficientResamplingDriver<SingleMatcher, PairwiseNptTraversal<SingleMatcher>,
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
  
  // Now, do the multi matcher algorithm
  MatcherArguments multi_args(matcher_mat, tuple_size);
  multi_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<MultiMatcher, PairwiseNptTraversal<MultiMatcher>,
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
    
} // test multi vs. single (pairwise_traversal)

BOOST_AUTO_TEST_CASE(multi_matcher_vs_efficient_multi_matcher)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  
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
  
  int tuple_size = 3;
  int tuple_size_choose_2 = (tuple_size - 1) * tuple_size / 2;
  
  arma::mat efficient_data(data_mat);
  arma::mat efficient_random(random_mat);
  
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
  
  MatcherArguments efficient_args(matcher_mat, tuple_size);
  efficient_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<EfficientMultiMatcher, GenericNptAlg<EfficientMultiMatcher>,
  NptNode, MultiResults>
  efficient_alg(efficient_data,
             data_weights,
             efficient_random,
             random_weights,
             efficient_args,
             num_regions,
             num_regions,
             num_regions,
             helper,
                tuple_size,
             16);
  
  efficient_alg.Compute();
  
  MultiResults efficient_results = efficient_alg.results();
  
  //mlpack::Log::Info << "\n\nEfficient results:\n";
  //efficient_results.PrintResults();
  
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
  
  //mlpack::Log::Info << "\n\nMulti Results\n";
  //multi_results.PrintResults();
  
  BOOST_REQUIRE(multi_results == efficient_results);

} // test multi vs. efficient multi

BOOST_AUTO_TEST_CASE(multi_matcher_vs_efficient_multi_matcher_pairwise)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  
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
  
  int tuple_size = 3;
  int tuple_size_choose_2 = (tuple_size - 1) * tuple_size / 2;
  
  arma::mat efficient_data(data_mat);
  arma::mat efficient_random(random_mat);
  
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
  
  MatcherArguments efficient_args(matcher_mat, tuple_size);
  efficient_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<EfficientMultiMatcher, PairwiseNptTraversal<EfficientMultiMatcher>,
  NptNode, MultiResults>
  efficient_alg(efficient_data,
                data_weights,
                efficient_random,
                random_weights,
                efficient_args,
                num_regions,
                num_regions,
                num_regions,
                helper,
                tuple_size,
                16);
  
  efficient_alg.Compute();
  
  MultiResults efficient_results = efficient_alg.results();
  
  // Now, do the multi matcher algorithm
  MatcherArguments multi_args(matcher_mat, tuple_size);
  multi_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<MultiMatcher, PairwiseNptTraversal<MultiMatcher>,
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
  
  BOOST_REQUIRE(efficient_results == multi_results);
  
} // test multi vs. efficient multi (pairwise)


BOOST_AUTO_TEST_CASE(multi_vs_unordered_multi)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  int tuple_size_lo = 2;
  int tuple_size_hi = 4;
  
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
  
  arma::mat unordered_data(data_mat);
  arma::mat unordered_random(random_mat);
  
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

  //////////////////////////////
  
  std::vector<arma::mat> lower_matchers;
  std::vector<arma::mat> upper_matchers;
  
  for (int i = 0; i < multi_args.total_matchers(); i++)
  {
    lower_matchers.push_back(multi_args.LowerMatcher(i));
    upper_matchers.push_back(multi_args.UpperMatcher(i));
  }
  
  
  MatcherArguments unordered_args(lower_matchers, upper_matchers);
  
  EfficientResamplingDriver<UnorderedMultiMatcher,
                        GenericNptAlg<UnorderedMultiMatcher>,
  NptNode, MultiResults>
  unordered_alg(unordered_data,
             data_weights,
             unordered_random,
             random_weights,
             unordered_args,
             num_regions,
             num_regions,
             num_regions,
             helper,
                tuple_size,
             16);
  
  unordered_alg.Compute();
  
  MultiResults unordered_results = unordered_alg.results();
  
    
  BOOST_REQUIRE(unordered_results == multi_results);

} // unordered multi (efficient)

BOOST_AUTO_TEST_CASE(unordered_multi_vs_iterate_through_single)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  
  double min_band_lo = 0.05;
  double min_band_hi = 0.1;
  double max_band_lo = 0.12;
  double max_band_hi = 0.25;
  
  double r2_mult_lo = 1.0;
  double r2_mult_hi = 2.1;
  
  double theta_min_lo = 0.175; // 10 degrees
  double theta_min_hi = 0.61; // 35 degrees
  double theta_max_lo = 2.09; // 120 degrees
  double theta_max_hi = 2.97; // 170 degrees
  
  double band_thick_min = 0.08;
  double band_thick_max = 0.3;
  
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
  
  int tuple_size = 2;
  
  arma::mat single_data(data_mat);
  arma::mat single_random(random_mat);
  
  arma::mat multi_data(data_mat);
  arma::mat multi_random(random_mat);
  
  // Generate matcher info
  double min_r1 = mlpack::math::Random(min_band_lo, min_band_hi);
  double max_r1 = mlpack::math::Random(max_band_lo, max_band_hi);
  int num_r1 = mlpack::math::RandInt(num_bands_lo, num_bands_hi);
  
  double r2_mult = mlpack::math::Random(r2_mult_lo, r2_mult_hi);
  
  double min_theta = mlpack::math::Random(theta_min_lo, theta_min_hi);
  double max_theta = mlpack::math::Random(theta_max_lo, theta_max_hi);
  int num_theta = mlpack::math::RandInt(num_bands_lo, num_bands_hi);
  
  double bin_thick = mlpack::math::Random(band_thick_min,
                                          band_thick_max);
  
  MatcherArguments angle_args(min_r1, max_r1, num_r1,
                              min_theta, max_theta, num_theta,
                              r2_mult, bin_thick);
  
  MatcherArguments multi_args = angle_args.Generate2ptMatchers();
  multi_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<UnorderedMultiMatcher, GenericNptAlg<UnorderedMultiMatcher>,
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
  
  //////////////////////////////
  
  MatcherArguments single_args = angle_args.Generate2ptMatchers();
  single_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
  
  EfficientResamplingDriver<SingleMatcher,
  GenericNptAlg<SingleMatcher>,
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
  
  
  BOOST_REQUIRE(single_results == multi_results);
  
} // unordered multi (efficient resampling)

BOOST_AUTO_TEST_CASE(unordered_multi_vs_unordered_efficient_multi)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  
  double min_band_lo = 0.05;
  double min_band_hi = 0.1;
  double max_band_lo = 0.12;
  double max_band_hi = 0.25;
  
  double r2_mult_lo = 1.0;
  double r2_mult_hi = 2.1;
  
  double theta_min_lo = 0.175; // 10 degrees
  double theta_min_hi = 0.61; // 35 degrees
  double theta_max_lo = 2.09; // 120 degrees
  double theta_max_hi = 2.97; // 170 degrees
  
  double band_thick_min = 0.08;
  double band_thick_max = 0.3;
  
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
  
  int tuple_size = 2;
  
  arma::mat efficient_data(data_mat);
  arma::mat efficient_random(random_mat);
  
  arma::mat multi_data(data_mat);
  arma::mat multi_random(random_mat);
  
  // Generate matcher info
  double min_r1 = mlpack::math::Random(min_band_lo, min_band_hi);
  double max_r1 = mlpack::math::Random(max_band_lo, max_band_hi);
  int num_r1 = mlpack::math::RandInt(num_bands_lo, num_bands_hi);
  
  double r2_mult = mlpack::math::Random(r2_mult_lo, r2_mult_hi);
  
  double min_theta = mlpack::math::Random(theta_min_lo, theta_min_hi);
  double max_theta = mlpack::math::Random(theta_max_lo, theta_max_hi);
  int num_theta = mlpack::math::RandInt(num_bands_lo, num_bands_hi);
  
  double bin_thick = mlpack::math::Random(band_thick_min,
                                          band_thick_max);
  
  MatcherArguments angle_args(min_r1, max_r1, num_r1,
                              min_theta, max_theta, num_theta,
                              r2_mult, bin_thick);
  
  MatcherArguments multi_args = angle_args.Generate2ptMatchers();
  multi_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<UnorderedMultiMatcher, GenericNptAlg<UnorderedMultiMatcher>,
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
  
  //////////////////////////////
  
  MatcherArguments efficient_args = angle_args.Generate2ptMatchers();
  efficient_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<UnorderedEfficientMultiMatcher,
  GenericNptAlg<UnorderedEfficientMultiMatcher>,
  NptNode, MultiResults>
  efficient_alg(efficient_data,
             data_weights,
             efficient_random,
             random_weights,
             efficient_args,
             num_regions,
             num_regions,
             num_regions,
             helper,
                tuple_size,
             16);
  
  efficient_alg.Compute();
  
  MultiResults efficient_results = efficient_alg.results();
  
  BOOST_REQUIRE(efficient_results == multi_results);
  
} // unordered multi (efficient resampling)

BOOST_AUTO_TEST_CASE(on_diagonal_efficient_multi_matcher)
{
  
  omp_set_num_threads(1);
  
  int num_data_lo = 50;
  int num_data_hi = 100;
  
  double min_band_lo = 0.05;
  double min_band_hi = 0.1;
  double max_band_lo = 0.12;
  double max_band_hi = 0.25;
  int num_bands_lo = 3;
  int num_bands_hi = 7;
  
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
  
  int tuple_size = 3;
  int tuple_size_choose_2 = (tuple_size - 1) * tuple_size / 2;
  
  arma::mat efficient_data(data_mat);
  arma::mat efficient_random(random_mat);
  
  arma::mat single_data(data_mat);
  arma::mat single_random(random_mat);
  
  // Generate matcher info
  arma::mat matcher_mat(tuple_size_choose_2,3);
  int total_num_matchers = 1;
  for (int i = 0; i < tuple_size_choose_2; i++) {
    
    matcher_mat.at(i,0) = mlpack::math::Random(min_band_lo, min_band_hi);
    matcher_mat.at(i,1) = mlpack::math::Random(max_band_lo, max_band_hi);
    matcher_mat.at(i,2) = mlpack::math::RandInt(num_bands_lo, num_bands_hi);
    
  }
  total_num_matchers = matcher_mat.at(0,2);
  
  //std::cout << "num matchers: " << total_num_matchers << "\n";
  
  double min_band = matcher_mat.at(0,0);
  double max_band = matcher_mat.at(0,1);
  
  MatcherArguments efficient_args(matcher_mat, tuple_size, false);
  efficient_args.set_template_type(MatcherArguments::TEMPLATE_MULTI_MATCHER);
  
  EfficientResamplingDriver<EfficientMultiMatcher, GenericNptAlg<EfficientMultiMatcher>,
  NptNode, MultiResults>
  efficient_alg(efficient_data,
                data_weights,
                efficient_random,
                random_weights,
                efficient_args,
                num_regions,
                num_regions,
                num_regions,
                helper,
                tuple_size,
                16);
  
  efficient_alg.Compute();
  
  MultiResults efficient_results = efficient_alg.results();

  //std::cout << "multi results\n";
  //efficient_results.PrintResults(std::cout);
  
  double band_step = (max_band - min_band) / (double)total_num_matchers;
  
  for (int i = 0; i < total_num_matchers; i++)
  {
    
    arma::mat lower(3,3);
    arma::mat upper(3,3);
    lower.zeros();
    upper.zeros();
    
    double lower_val = min_band + (double)i * band_step;
    double upper_val = min_band + (double)(i+1) * band_step;
    
    lower(0,1) = lower_val;
    lower(0,2) = lower_val;
    lower(1,2) = lower_val;

    lower(1,0) = lower_val;
    lower(2,0) = lower_val;
    lower(2,1) = lower_val;
    
    upper(0,1) = upper_val;
    upper(0,2) = upper_val;
    upper(1,2) = upper_val;

    upper(1,0) = upper_val;
    upper(2,0) = upper_val;
    upper(2,1) = upper_val;
    
    MatcherArguments single_args(lower, upper);
    
    EfficientResamplingDriver<EfficientCpuMatcher, GenericNptAlg<EfficientCpuMatcher>,
    NptNode, SingleResults>
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
    
    SingleResults single_results = single_alg.results();
    
    //std::cout << "single results \n";
    //single_results.PrintResults(std::cout);
    
    for (int j = 0; j < tuple_size; j++) {
      
      BOOST_REQUIRE_EQUAL(efficient_results.results()[0][j][i],
                          single_results.results()[0][j]);
      
      
    } // loop over data results
    
    // Check RRR results
    BOOST_REQUIRE_EQUAL(efficient_results.RRR_result()[i],
                        single_results.RRR_result());
    
  } // loop over matchers
  
} // test multi vs. efficient multi
BOOST_AUTO_TEST_SUITE_END();
