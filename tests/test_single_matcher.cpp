/*
 * test_single_matcher.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Basic tests for SingleMatcher, NaiveResampling (with 1 resampling region),
 * and no parallelism.
 */

#include <boost/test/unit_test.hpp>
#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../matchers/single_matcher.hpp"
#include "../results/single_results.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/pairwise_npt_traversal.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include <omp.h>

using namespace npoint_mlpack;
using namespace arma;

BOOST_AUTO_TEST_SUITE(SingleMatcherTest);

/**
 * Compare 2-point results (Single Matcher, standard tree traversal,
 * Single Matcher) to hand-computed results
 */
BOOST_AUTO_TEST_CASE(exhaustive_test_2pt)
{
  
  BOOST_TEST_MESSAGE("Running exhastive 2pt test.");
  
  omp_set_num_threads(1);
  
  // just creating simple one-dimensional data and randoms
  mat data(3, 10);
  data(span(1,2), span::all).fill(0.0);
  data(0,0) = 0.1;
  data(0,1) = 0.2;
  data(0,2) = 0.3;
  data(0,3) = 0.4;
  data(0,4) = 0.5;
  data(0,5) = 0.6;
  data(0,6) = 0.7;
  data(0,7) = 0.8;
  data(0,8) = 0.9;
  data(0,9) = 0.5523;
  
  // 10 DD pairs
  
  ResamplingHelper helper(data);
  
  colvec data_weights(10);
  data_weights.fill(1.0);
  
  mat randoms(3, 12);
  randoms(span(1,2), span::all).fill(0.0);
  randoms(0,0) = 0.15;
  randoms(0,1) = 0.25;
  randoms(0,2) = 0.35;
  randoms(0,3) = 0.45;
  randoms(0,4) = 0.55;
  randoms(0,5) = 0.65;
  randoms(0,6) = 0.75;
  randoms(0,7) = 0.85;
  randoms(0,8) = 0.95;
  randoms(0,9) = 0.231;
  randoms(0,10) = 0.001;
  randoms(0,11) = 0.999;
  
  
  // 24 DR pairs
  // 12 RR pairs
  colvec random_weights(12);
  random_weights.fill(1.0);
  
  ///////////////// Do a 2-pt case ////////////////////
  
  mat data_2pt = data;
  mat randoms_2pt = randoms;
  
  mat matcher_dists_2pt_lower(2,2);
  matcher_dists_2pt_lower(0,0) = 0.0;
  matcher_dists_2pt_lower(0,1) = 0.0;
  matcher_dists_2pt_lower(1,0) = 0.0;
  matcher_dists_2pt_lower(1,1) = 0.0;
  
  mat matcher_dists_2pt_upper(2,2);
  matcher_dists_2pt_upper(0,0) = 0.0;
  matcher_dists_2pt_upper(0,1) = 0.12;
  matcher_dists_2pt_upper(1,0) = 0.12;
  matcher_dists_2pt_upper(1,1) = 0.0;
  
  MatcherArguments args_2pt(matcher_dists_2pt_lower, matcher_dists_2pt_upper);
  
  int leaf_size = 1;
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  driver(data_2pt, data_weights,
         randoms_2pt, random_weights,
         args_2pt,
         1, 1, 1, // num resampling regions
         helper,
         2, // tuple size
         leaf_size); // leaf size
  
  driver.Compute();
  
  SingleResults results_2pt = driver.results();
  
  // just for now to make sure I'm getting it right
  //results_2pt.PrintResults(std::cout);
  
  // now, check that they're correct
  
  // RR pairs
  BOOST_REQUIRE_EQUAL(results_2pt.RRR_result(), 12);
  
  // DD pairs
  BOOST_REQUIRE_EQUAL(results_2pt.results()[0][0], 10);
  
  // DR pairs
  BOOST_REQUIRE_EQUAL(results_2pt.results()[0][1], 24);
  
} // exhaustive case (2pt)

/**
 * I ran the files test_data_200.csv and test_random_250.csv through ntropy
 * and recorded the results.  This test makes sure this code gets the same
 * answer.
 *
 * Matcher: 0.09, 0.11
 * DD: 7
 * DR: 14
 * RR: 7
 *
 * Note that the data and random sets are the same, and both only contain
 * 100 points.  I'll get around to at least fixing the names eventually.
 */
BOOST_AUTO_TEST_CASE(ntropy_tests_2pt)
{
  
  BOOST_TEST_MESSAGE("Running 2pt test Single vs. Ntropy.");
  
  mat loaded_data, loaded_randoms;
  mat input_data, input_randoms;
  
  loaded_data.load("test_data_200.csv", raw_ascii);
  loaded_randoms.load("test_randoms_250.csv", raw_ascii);
  
  input_data = loaded_data.t();
  input_randoms = loaded_randoms.t();
  
  colvec data_weights(input_data.n_cols);
  data_weights.fill(1.0);
  
  colvec random_weights(input_randoms.n_cols);
  random_weights.fill(1.0);
  
  int tuple_size = 2;
  
  mat matcher_lower(tuple_size, tuple_size);
  matcher_lower << 0.0 << 0.09 << arma::endr
  << 0.09 << 0.0 << arma::endr;
  
  mat matcher_upper(tuple_size, tuple_size);
  matcher_upper << 0.0 << 0.11 << arma::endr
  << 0.11 << 0.0 << arma::endr;
  
  MatcherArguments matcher_args(matcher_lower, matcher_upper);
  
  int num_regions = 1;
  
  int leaf_size = 2;
  
  // Only testing 1 thread
  omp_set_num_threads(1);
  
  ResamplingHelper helper(input_data);
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  alg(input_data,
      data_weights,
      input_randoms,
      random_weights,
      matcher_args,
      num_regions,
      1, 1,
      helper,
      tuple_size,
      leaf_size);
  
  alg.Compute();
  
  SingleResults& results = alg.results();
  
  //results.PrintResults();
  
  // DD
  BOOST_REQUIRE_EQUAL(results.results()[0][0], 7);
  // DR
  BOOST_REQUIRE_EQUAL(results.results()[0][1], 14);
  // RR
  BOOST_REQUIRE_EQUAL(results.RRR_result(), 7);
  
} // 2pt vs ntropy

/**
 * I ran the files test_data_200.csv and test_random_250.csv through ntropy
 * and recorded the results.  This test makes sure this code gets the same
 * answer.
 *
 * Matcher: 0.09, 0.15 0.09,0.15 0.09,0.15
 * DDD: 8
 * DDR: 24
 * DRR: 24
 * RRR: 8
 *
 * Note that the data and random sets are the same, and both only contain
 * 100 points.  I'll get around to at least fixing the names eventually.
 */

BOOST_AUTO_TEST_CASE(ntropy_tests_3pt)
{
  
  BOOST_TEST_MESSAGE("Running 3pt test Single vs. Ntropy.");
  
  mat input_data, input_randoms;
  mat loaded_data, loaded_randoms;
  
  loaded_data.load("test_data_200.csv", arma::raw_ascii);
  loaded_randoms.load("test_randoms_250.csv", arma::raw_ascii);
  
  input_data = loaded_data.t();
  input_randoms = loaded_randoms.t();
  
  colvec data_weights(input_data.n_cols);
  data_weights.fill(1.0);
  
  colvec random_weights(input_randoms.n_cols);
  random_weights.fill(1.0);
  
  int tuple_size = 3;
  
  mat matcher_lower(tuple_size, tuple_size);
  matcher_lower << 0.0 << 0.09 << 0.09 << arma::endr
  << 0.09 << 0.0 << 0.09 << arma::endr
  << 0.09 << 0.09 << 0.0 << arma::endr;
  
  mat matcher_upper(tuple_size, tuple_size);
  matcher_upper << 0.0 << 0.15 << 0.15 << arma::endr
  << 0.15 << 0.0 << 0.15 << arma::endr
  << 0.15 << 0.15 << 0.0 << arma::endr;
  
  MatcherArguments matcher_args(matcher_lower, matcher_upper);
  
  int num_regions = 1;
  
  int leaf_size = 2;
  
  // Only testing 1 thread
  omp_set_num_threads(1);
  
  ResamplingHelper helper(input_data);
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  alg(input_data,
      data_weights,
      input_randoms,
      random_weights,
      matcher_args,
      num_regions,
      1, 1,
      helper,
      tuple_size,
      leaf_size);
  
  alg.Compute();
  
  SingleResults results = alg.results();
  
  //results.PrintResults();
  
  
  BOOST_REQUIRE_EQUAL(results.results()[0][0], 8);
  BOOST_REQUIRE_EQUAL(results.results()[0][1], 24);
  BOOST_REQUIRE_EQUAL(results.results()[0][2], 24);
  BOOST_REQUIRE_EQUAL(results.RRR_result(), 8);
  
} // 3pt vs ntropy

/**
 * Run the single matcher algorithm vs. the naive algorithm on a randomly
 * generated problem.
 */
BOOST_AUTO_TEST_CASE(single_vs_naive)
{
  
  // parameters for the randomly generated problem
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
  
  for (int tuple_size = tuple_size_lo; tuple_size <= tuple_size_hi;
       tuple_size++)
  {
    
    BOOST_TEST_MESSAGE("Running " << tuple_size << "pt test Single vs. Naive.");
    
    // Generate a data set
    int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
    int num_random_points = 2 * mlpack::math::RandInt(num_data_lo, num_data_hi);
    int num_dimensions = 3;
    
    GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                      matcher_thick_lo, matcher_thick_hi,
                                      num_data_lo, num_data_hi);
    
    
    mat data_mat(num_dimensions, num_data_points);
    problem_gen.GenerateRandomSet(data_mat);
    colvec data_weights(num_data_points);
    
    ResamplingHelper helper(data_mat);
    
    // Generate a random set
    mat random_mat(num_dimensions, num_random_points);
    problem_gen.GenerateRandomSet(random_mat);
    colvec random_weights(num_random_points);
    
    // Generate a random matcher and matcher thickness multiplier
    mat matcher_dists(tuple_size, tuple_size);
    double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
    
    MatcherArguments naive_args(matcher_dists, matcher_thick);
    
    MatcherArguments tree_args(matcher_dists, matcher_thick);
    
    // copy the data
    mat tree_data(data_mat);
    mat naive_data(data_mat);
    mat tree_randoms(random_mat);
    mat naive_randoms(random_mat);
    
    // resampling info (we don't want to test this here, but the driver needs it
    int num_regions = 1;
    
    int leaf_size = std::max(num_data_points, num_random_points);
    
    // Run the naive algorithm
    NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
    NptNode, SingleResults>
    naive_alg(naive_data,
              data_weights,
              naive_randoms,
              random_weights,
              naive_args,
              num_regions,
              num_regions,
              num_regions,
              helper,
              tuple_size,
              leaf_size);
    
    
    naive_alg.Compute();
    
    SingleResults naive_results = naive_alg.results();
    //naive_results.PrintResults(std::cout);
    
    // Do the tree algorithm
    leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
    NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
    NptNode, SingleResults>
    tree_alg(tree_data,
             data_weights,
             tree_randoms,
             random_weights,
             tree_args,
             num_regions,
             num_regions,
             num_regions,
             helper,
             tuple_size,
             leaf_size);
    
    tree_alg.Compute();
    
    SingleResults tree_results = tree_alg.results();
    //tree_results.PrintResults(std::cout);
    
    BOOST_REQUIRE(tree_results == naive_results);
    
  } // loop over tuple sizes
  
} // single_vs_naive

BOOST_AUTO_TEST_CASE(single_pairwise_traversal)
{
  
  // parameters for the randomly generated problem
  double matcher_dist_lo = 0.05;
  double matcher_dist_hi = 0.15;
  double matcher_thick_lo = 0.05;
  double matcher_thick_hi = 0.12;
  int num_data_lo = 50;
  int num_data_hi = 100;
  int num_leaves_lo = 1;
  int num_leaves_hi = 25;
  
  int tuple_size = 3;
  
  // Generate a data set
  int num_data_points = mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_random_points = 2 * mlpack::math::RandInt(num_data_lo, num_data_hi);
  int num_dimensions = 3;
  
  GenerateRandomProblem problem_gen(matcher_dist_lo, matcher_dist_hi,
                                    matcher_thick_lo, matcher_thick_hi,
                                    num_data_lo, num_data_hi);
  
  
  mat data_mat(num_dimensions, num_data_points);
  problem_gen.GenerateRandomSet(data_mat);
  colvec data_weights(num_data_points);
  
  ResamplingHelper helper(data_mat);
  
  // Generate a random set
  mat random_mat(num_dimensions, num_random_points);
  problem_gen.GenerateRandomSet(random_mat);
  colvec random_weights(num_random_points);
  
  // Generate a random matcher and matcher thickness multiplier
  mat matcher_dists(tuple_size, tuple_size);
  double matcher_thick = problem_gen.GenerateRandomMatcher(matcher_dists);
  
  MatcherArguments pairwise_args(matcher_dists, matcher_thick);
  
  MatcherArguments generic_args(matcher_dists, matcher_thick);
  
  // copy the data
  mat generic_data(data_mat);
  mat pairwise_data(data_mat);
  mat generic_randoms(random_mat);
  mat pairwise_randoms(random_mat);
  
  // resampling info (we don't want to test this here, but the driver needs it
  int num_regions = 1;
  
  int leaf_size = mlpack::math::RandInt(num_leaves_lo, num_leaves_hi);
  
  // Run the naive algorithm
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
  
  NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  generic_alg(generic_data,
           data_weights,
           generic_randoms,
           random_weights,
           generic_args,
           num_regions,
           num_regions,
           num_regions,
           helper,
              tuple_size,
           leaf_size);
  
  generic_alg.Compute();
  
  SingleResults generic_results = generic_alg.results();
  
  BOOST_REQUIRE(generic_results == pairwise_results);
    
} // single_pairwise

BOOST_AUTO_TEST_SUITE_END();



