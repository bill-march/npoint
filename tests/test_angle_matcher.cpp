//
//  test_angle_matcher.cpp
//  contrib_march
//
//  Created by William March on 9/25/12.
//
//


#include <mlpack/core.hpp>

#include "../resampling_classes/naive_resampling_driver.hpp"
#include "../resampling_classes/efficient_resampling_driver.hpp"
#include "../matchers/single_matcher.hpp"
#include "../infrastructure/generate_random_problem.hpp"
#include "../infrastructure/generic_npt_alg.hpp"
#include "../infrastructure/resampling_helper.hpp"
#include "../matchers/matcher_arguments.hpp"
#include "../matchers/efficient_cpu_matcher.hpp"

#include "../results/angle_results.hpp"
#include "../results/single_results.hpp"

#include <boost/test/unit_test.hpp>
#include <omp.h>

using namespace npoint_mlpack;
using namespace mlpack;

BOOST_AUTO_TEST_SUITE(TestAngleMatcher);

/**
 * Only tests whether we can iterate through angle matchers and 
 * compute each result separately.  Doesn't actually use angle matchers yet.
 *
 * This test doesn't really test much, since a lot of the same code is used 
 * to iterate through the matchers on each end.
 */
BOOST_AUTO_TEST_CASE(angle_problem_iterative_efficient_resampling)
{
  
  BOOST_TEST_MESSAGE("Note: this test doesn't actually use the AngleMatcher.");
  
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
  
  arma::mat single_data(data_mat);
  arma::mat single_random(random_mat);
  
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
  angle_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
  
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
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
  
  // Now, compare the results
  for (int r1_ind = 0; r1_ind < num_r1; r1_ind++)
  {
    
    for (int theta_ind = 0; theta_ind < num_theta; theta_ind++)
    {
      
      MatcherArguments single_args = angle_args.GenerateMatcher(r1_ind,
                                                                theta_ind);
      
      arma::mat this_data(single_data);
      arma::mat this_random(single_random);
      
      EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
                                NptNode, SingleResults>
      single_alg(this_data,
                 data_weights,
                 this_random,
                 random_weights,
                 single_args,
                 num_regions,
                 num_regions,
                 num_regions,
                 helper,
                 tuple_size,
                 5);
      
      single_alg.Compute();
      
      SingleResults single_results = single_alg.results();
      
      for (int num_random = 0; num_random < tuple_size; num_random++)
      {
      
        // always use zero for index of resampling region since we're only
        // testing one
        BOOST_REQUIRE_EQUAL(single_results.results()[0][num_random],
                            angle_results.results()[0][num_random]
                                                   [r1_ind][theta_ind]);
        
      }

      // now check the random result
      BOOST_REQUIRE_EQUAL(single_results.RRR_result(),
                          angle_results.RRR_result()[r1_ind][theta_ind]);
      
    } // loop over thetas
    
  } // loop over r1
      
} // test multi vs. single (efficient resampling)

 
BOOST_AUTO_TEST_CASE(AngleMatcher_vs_iterate_through_single_matchers)
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
  
  arma::mat single_data(data_mat);
  arma::mat single_random(random_mat);
  
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
  
  //BOOST_TEST_MESSAGE("Doing angle matcher.");
  
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
  
  //mlpack::Log::Info << "\n\n\n\nAngle results\n";
  //angle_results.PrintResults();
  
  //BOOST_TEST_MESSAGE("Doing single matcher.");
  MatcherArguments single_args(min_r1, max_r1, num_r1,
                               min_theta, max_theta, num_theta,
                               r2_mult, bin_thick);
  single_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
  
  EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
  NptNode, AngleResults>
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
            5);
  
  single_alg.Compute();
  
  AngleResults single_results = single_alg.results();
  
  //mlpack::Log::Info << "\n\nSingle results\n";
  //single_results.PrintResults();
  
  // Now, compare the results
  BOOST_REQUIRE(angle_results == single_results);

} // test full angle matcher vs iterative

BOOST_AUTO_TEST_SUITE_END();

