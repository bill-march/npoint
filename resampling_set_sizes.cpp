/**
 * @file resampling_set_sizes.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Computes the sizes of the resampling sets and prints them.  This 
 * functionality should go in the results classes.
 */

#include "infrastructure/generate_random_problem.hpp"
#include "resampling_classes/efficient_resampling_driver.hpp"
#include "infrastructure/generic_npt_alg.hpp"
#include "matchers/single_matcher.hpp"
#include "matchers/efficient_angle_matcher.hpp"
#include "results/single_results.hpp"
#include "infrastructure/resampling_helper.hpp"

PROGRAM_INFO("Resampling set size computation.",
             "Just computes and outputs the sizes of the resampling subsets. "
             "This is needed to normalize the counts for the npcf estimators.");
PARAM_STRING_REQ("data", "Point coordinates.", "d");
PARAM_STRING("random", "Optional Poisson set coordinates.", "r", "fake");
//PARAM_DOUBLE("box_x_length", "Length of the box containing the data in the x direction.", "a", 1.0);
//PARAM_DOUBLE("box_y_length", "Length of the box containing the data in the y direction.", "b", 1.0);
//PARAM_DOUBLE("box_z_length", "Length of the box containing the data in the z direction.", "c", 1.0);
PARAM_INT("num_x_regions", "Number of regions to divide the input into along the x coordinate.",
          "x", 1);
PARAM_INT("num_y_regions", "Number of regions to divide the input into along the y coordinate.",
          "y", 1);
PARAM_INT("num_z_regions", "Number of regions to divide the input into along the z coordinate.",
          "z", 1);

using namespace mlpack;
using namespace npoint_mlpack;

int main(int argc, char* argv[])
{
  
  CLI::ParseCommandLine(argc, argv);
  
  omp_set_num_threads(1);
  
  //double box_x_length = CLI::GetParam<double>("box_x_length");
  //double box_y_length = CLI::GetParam<double>("box_y_length");
  //double box_z_length = CLI::GetParam<double>("box_z_length");
  
  int num_x_regions = CLI::GetParam<int>("num_x_regions");
  int num_y_regions = CLI::GetParam<int>("num_y_regions");
  int num_z_regions = CLI::GetParam<int>("num_z_regions");
  
  // The total data and random inputs.  These will be split up and scattered
  // to the other processes by the root.
  arma::mat data_all_mat, random_all_mat;
  arma::colvec data_all_weights, random_all_weights;
  
  // These are the data that this process is responsible for passing to the
  // distributed driver.
  arma::mat data_mat, random_mat;
  arma::colvec data_weights, random_weights;

  std::string data_filename = CLI::GetParam<std::string>("data");
  
  arma::mat data_in;
  data_in.load(data_filename, arma::raw_ascii);
  
  // THIS IS BAD: do it better
  if (data_in.n_rows > data_in.n_cols) {
    data_mat = arma::trans(data_in);
  }
  else {
    data_mat = data_in;
  }
  data_in.reset();
  
  data_weights.set_size(data_all_mat.n_cols);
  data_weights.fill(1.0);
  
  ResamplingHelper helper(data_mat);

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
    
    random_weights.set_size(random_all_mat.n_cols);
    random_weights.fill(1.0);
  
  } // is the random set input as a file?
  else {
    
    // we need to generate our own random set
    int num_random = 0;
    
    random_all_mat.set_size(3, num_random);
    random_all_weights.set_size(num_random);
    
  }

  int tuple_size = 2;
  arma::mat lower_matcher(2,2);
  lower_matcher.fill(0.0);
  
  arma::mat upper_matcher(2,2);
  upper_matcher.fill(1.0);
  
  MatcherArguments matcher_args(lower_matcher, upper_matcher);
  
  EfficientResamplingDriver<SingleMatcher,
  GenericNptAlg<SingleMatcher>,
  NptNode, SingleResults>
  driver(data_mat,
         data_weights,
         random_mat,
         random_weights,
         matcher_args,
         num_x_regions,
         num_y_regions,
         num_z_regions,
         helper,
         tuple_size);
  
  driver.PrintRegionSizes();
  
  return 0;
  
} // main

