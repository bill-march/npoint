//
//  test_efficient_angle_matcher.cpp
//  contrib_march
//
//  Created by William March on 10/12/12
//
//


#include <mlpack/core.hpp>

#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/pairwise_npt_traversal.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../matchers/angle_matcher.hpp"
#include "../matchers/efficient_angle_matcher.hpp"
#include "../infrastructure/resampling_helper.hpp"

#include "../results/angle_results.hpp"

#include <boost/test/unit_test.hpp>
#include <omp.h>

using namespace npoint_mlpack;
using namespace mlpack;

BOOST_AUTO_TEST_SUITE(TestEfficientAngleMatcher);

/*
BOOST_AUTO_TEST_CASE(EfficientAngleMatcher_vs_AngleMatcher)
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
  
  int tuple_size = 3;
  
  arma::mat efficient_data(data_mat);
  arma::mat efficient_random(random_mat);
  
  arma::mat angle_data(data_mat);
  arma::mat angle_random(random_mat);
  
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
  
  EfficientResamplingDriver<AngleMatcher, GenericNptAlg<AngleMatcher>,
  NptNode, AngleResults>
  angle_alg(angle_data,
            data_weights,
            angle_random,
            random_weights,
            angle_args,
            num_regions,
            num_regions,
            num_regions,
            helper,
            tuple_size,
            5);
  
  angle_alg.Compute();
  
  AngleResults angle_results = angle_alg.results();
  
  
  MatcherArguments efficient_args(min_r1, max_r1, num_r1,
                               min_theta, max_theta, num_theta,
                               r2_mult, bin_thick);

  EfficientResamplingDriver<EfficientAngleMatcher, GenericNptAlg<EfficientAngleMatcher>,
  NptNode, AngleResults>
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
                5);
  
  efficient_alg.Compute();
  
  AngleResults efficient_results = efficient_alg.results();
  
  BOOST_REQUIRE(efficient_results == angle_results);
  
} // test full angle matcher vs iterative
*/
/*
BOOST_AUTO_TEST_CASE(EfficientAngleMatcher_vs_AngleMatcher_pairwise)
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
  
  int tuple_size = 3;
  
  arma::mat efficient_data(data_mat);
  arma::mat efficient_random(random_mat);
  
  arma::mat angle_data(data_mat);
  arma::mat angle_random(random_mat);
  
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
  
  EfficientResamplingDriver<AngleMatcher, PairwiseNptTraversal<AngleMatcher>,
  NptNode, AngleResults>
  angle_alg(angle_data,
            data_weights,
            angle_random,
            random_weights,
            angle_args,
            num_regions,
            num_regions,
            num_regions,
            helper,
            tuple_size,
            5);
  
  angle_alg.Compute();
  
  AngleResults angle_results = angle_alg.results();
  
  
  MatcherArguments efficient_args(min_r1, max_r1, num_r1,
                                  min_theta, max_theta, num_theta,
                                  r2_mult, bin_thick);
  
  EfficientResamplingDriver<EfficientAngleMatcher, PairwiseNptTraversal<EfficientAngleMatcher>,
  NptNode, AngleResults>
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
                5);
  
  efficient_alg.Compute();
  
  AngleResults efficient_results = efficient_alg.results();
  
  BOOST_REQUIRE(efficient_results == angle_results);

} // pairwise efficient angle vs angle
*/
BOOST_AUTO_TEST_SUITE_END();

