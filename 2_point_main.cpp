/**
 * @file 2_point_main.cpp
 * @author Bill March (march@gatech.edu)
 * 
 * Does general 2-point computation on a single node.
 */

#include "resampling_classes/naive_resampling_driver.hpp"
#include "resampling_classes/efficient_resampling_driver.hpp"
#include "matchers/single_matcher.hpp"
#include "matchers/efficient_2pt_matcher.hpp"
#include "infrastructure/generate_random_problem.hpp"
#include "infrastructure/generic_npt_alg.hpp"
#include "results/single_results.hpp"
#include "matchers/matcher_arguments.hpp"
#include "infrastructure/resampling_helper.hpp"

#include <mlpack/core.hpp>

PROGRAM_INFO("2-point correlation estimation.",
             "Does 2pcf correlation counts for a single matcher on a single node with optional jackknife resampling.");
PARAM_STRING_REQ("data", "Point coordinates.", "d");
PARAM_STRING("random", "Optional Poisson set coordinates.", "r", "fake");
PARAM_INT("num_random", "If random isn't specified, this will generate the given number of points", "R", 0);
PARAM_STRING_REQ("matcher_lower_bounds", "The lower bound distances for the matcher, stored in a symmetric 2x2 matrix.", "l")
PARAM_STRING_REQ("matcher_upper_bounds", "The upper bound distances for the matcher, stored in a symmetric 2x2 matrix.", "u")
//PARAM_DOUBLE("box_x_length", "Length of the box containing the data in the x direction.", "a", 1.0);
//PARAM_DOUBLE("box_y_length", "Length of the box containing the data in the y direction.", "b", 1.0);
//PARAM_DOUBLE("box_z_length", "Length of the box containing the data in the z direction.", "c", 1.0);
PARAM_INT("leaf_size", "Maximum number of points in a leaf node.", "i", 16);
PARAM_INT("num_x_regions", "Number of regions to divide the input into along the x coordinate.",
          "x", 1);
PARAM_INT("num_y_regions", "Number of regions to divide the input into along the y coordinate.",
          "y", 1);
PARAM_INT("num_z_regions", "Number of regions to divide the input into along the z coordinate.",
          "z", 1);
//PARAM_INT("num_threads", "Total number of threads to use.  Leaving it as 0 will cause the system to use the default number.  This has no effect for the single thread code.", "T", -1);

PARAM_STRING("resampler", "Select the type of jackknife resampling algorithm to use.  Options are \"naive\" or \"efficient\".", "s", "efficient");
PARAM_STRING("kernel", "Select the type of base case computation.  Options are \"general\" (the old version for general n) and \"efficient\" for the hardware optimized version.", "k", "general");

using namespace mlpack;
using namespace npoint_mlpack;

int main(int argc, char* argv[])
{
  
  CLI::ParseCommandLine(argc, argv);
  
  omp_set_nested(1);
  /*
  int num_threads = CLI::GetParam<int>("num_threads");
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
   */
  omp_set_num_threads(1);
  
  std::string data_filename = CLI::GetParam<std::string>("data");
  
  arma::mat data_in, data_mat;
  data_in.load(data_filename, arma::raw_ascii);
  
  // THIS IS BAD: do it better
  if (data_in.n_rows > data_in.n_cols) {
    data_mat = arma::trans(data_in);
  }
  else {
    data_mat = data_in;
  }
  data_in.reset();
  
  arma::colvec weights;
  weights.set_size(data_mat.n_cols);
  weights.fill(1.0);

  //double box_x_length = CLI::GetParam<double>("box_x_length");
  //double box_y_length = CLI::GetParam<double>("box_y_length");
  //double box_z_length = CLI::GetParam<double>("box_z_length");
  
  ResamplingHelper helper(data_mat);
  
  arma::mat random_mat;
  arma::colvec random_weights;
  if (CLI::HasParam("random")) {
    
    std::string random_filename = CLI::GetParam<std::string>("random");
    
    arma::mat random_in;
    random_in.load(random_filename, arma::raw_ascii);
    
    // THIS IS BAD: do it better
    if (random_in.n_rows > random_in.n_cols) {
      random_mat = arma::trans(random_in);
    }
    else {
      random_mat = random_in;
    }
    random_in.reset();
    
  } // do we have two sets?
  else {
    
    int num_random = CLI::GetParam<int>("num_random");
    
    random_mat.set_size(3, num_random);
    
    if (num_random > 0) {
      GenerateRandomProblem generator(0.1, 0.5, 0.1, 0.5,
                                      num_random, num_random+1,
                                      helper);
      generator.GenerateRandomSet(random_mat);
    }
    
  }
  
  random_weights.set_size(random_mat.n_cols);
  random_weights.fill(1.0);
  
  int leaf_size = CLI::GetParam<int>("leaf_size");
  
  arma::mat matcher_lower_bounds, matcher_upper_bounds;
  
  matcher_lower_bounds.load(CLI::GetParam<std::string>("matcher_lower_bounds"));
  matcher_upper_bounds.load(CLI::GetParam<std::string>("matcher_upper_bounds"));
  
  MatcherArguments matcher_args(matcher_lower_bounds,
                                matcher_upper_bounds);
  
  unsigned int tuple_size = 2;
  
  if (tuple_size != matcher_lower_bounds.n_rows) {
    
    mlpack::Log::Fatal << "Matchers need to be 2 x 2 matrices!\n";
    return 1;
    
  }
  
  if (tuple_size != matcher_upper_bounds.n_cols ||
      tuple_size != matcher_lower_bounds.n_rows) {
    
    mlpack::Log::Fatal << "Upper and lower matchers need to both be 2 x 2!\n";
    return 1;
    
  }
  
  int num_x_regions = CLI::GetParam<int>("num_x_regions");
  int num_y_regions = CLI::GetParam<int>("num_y_regions");
  int num_z_regions = CLI::GetParam<int>("num_z_regions");
  
  ////////////// Doing algorithms ///////////////////////////////
  std::string resampler_str = CLI::GetParam<std::string>("resampler");
  std::string kernel_str = CLI::GetParam<std::string>("kernel");
  
  // doing naive resampling
  if (0 == resampler_str.compare("naive")) {
  
    // doing general matcher
    if (0 == kernel_str.compare("general")) {
      
      Log::Info << "Doing Naive Resampling, General Single Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      NaiveResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
      NptNode, SingleResults>
      driver(data_mat,
             weights,
             random_mat,
             random_weights,
             matcher_args,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             helper,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      Timer::Stop("compute_time");
      
      driver.PrintResults();
      
      
    } // end of general matcher
      // doing efficient kernel matcher (CPU)
    else if (0 == kernel_str.compare("efficient")) {
      
      Log::Info << "Doing Naive Resampling, Efficient CPU Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
      
      NaiveResamplingDriver<Efficient2ptMatcher, GenericNptAlg<Efficient2ptMatcher>,
      NptNode, SingleResults>
      driver(data_mat,
             weights,
             random_mat,
             random_weights,
             matcher_args,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             helper,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      Timer::Stop("compute_time");
      
      driver.PrintResults();
      
    } // end of efficient matcher
    else {
      Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
      return 1;
    }
    
  } // end of naive resampling
  else if (0 == resampler_str.compare("efficient")) {
    // doing efficient resampling
    
    // doing general matcher
    if (0 == kernel_str.compare("general")) {
      
      Log::Info << "Doing Efficient Resampling, General Single Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
      
      EfficientResamplingDriver<SingleMatcher, GenericNptAlg<SingleMatcher>,
      NptNode, SingleResults>
      driver(data_mat,
             weights,
             random_mat,
             random_weights,
             matcher_args,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             helper,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      Timer::Stop("compute_time");
      
      driver.PrintResults();
      
      
    } // end of general matcher
      // doing efficient kernel matcher (CPU)
    else if (0 == kernel_str.compare("efficient")) {
      
      Log::Info << "Doing Efficient Resampling, Multi-Tree Traversal, Efficient CPU Matcher.\n\n";
      
      Timer::Start("compute_time");
      
      matcher_args.set_template_type(MatcherArguments::TEMPLATE_EFFICIENT_MATCHER);
      
      EfficientResamplingDriver<Efficient2ptMatcher, GenericNptAlg<Efficient2ptMatcher>,
      NptNode, SingleResults>
      driver(data_mat,
             weights,
             random_mat,
             random_weights,
             matcher_args,
             num_x_regions,
             num_y_regions,
             num_z_regions,
             helper,
             tuple_size,
             leaf_size);
      
      driver.Compute();
      
      Timer::Stop("compute_time");
      
      driver.PrintResults();
      
    } // end of efficient matcher
    else {
      Log::Fatal << "Option " << kernel_str << " not a valid kernel type.\n";
      return 1;
    }
    
  } // end of efficient resampling
  else {
    Log::Fatal << "Option " << resampler_str << " not a valid resampling type.\n";
    return 1;
  }
  
  return 0;
  
}


