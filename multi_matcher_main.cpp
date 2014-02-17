/**
 * @file multi_matcher_main.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Computes raw correlation counts for a multi matcher.
 *
 * For a general value of n, a multi-matcher specifies a minimum and maximum
 * bin and a number of bins in each of the (n choose 2) dimensions.  This 
 * code computes the raw correlation counts for each choice of bins, one for 
 * each dimension. 
 */

#include "resampling_classes/naive_resampling_driver.hpp"
#include "resampling_classes/efficient_resampling_driver.hpp"

#include "matchers/single_matcher.hpp"
#include "matchers/multi_matcher.hpp"
#include "matchers/efficient_cpu_matcher.hpp"
#include "matchers/efficient_multi_matcher.hpp"

#include "infrastructure/generate_random_problem.hpp"
#include "infrastructure/generic_npt_alg.hpp"
#include "infrastructure/pairwise_npt_traversal.hpp"
#include "infrastructure/resampling_helper.hpp"

#include "results/multi_results.hpp"

#include "matchers/matcher_arguments.hpp"

#include <mlpack/core.hpp>

PROGRAM_INFO("n-point correlation estimation using multi matchers.",
             "For a general value of n, a multi-matcher specifies a minimum and maximum "
             "bin and a number of bins in each of the (n choose 2) dimensions.  This "
             "code computes the raw correlation counts for each choice of bins, one for "
             "each dimension.");
PARAM_STRING_REQ("data", "Point coordinates.", "d");
PARAM_STRING("random", "Optional Poisson set coordinates.", "r", "fake");
PARAM_INT("num_random", "If random isn't specified, this will generate the given number of points", "R", 0);
//PARAM_DOUBLE("box_x_length", "Length of the box containing the data in the x direction.", "a", 1.0);
//PARAM_DOUBLE("box_y_length", "Length of the box containing the data in the y direction.", "b", 1.0);
//PARAM_DOUBLE("box_z_length", "Length of the box containing the data in the z direction.", "c", 1.0);
PARAM_INT("leaf_size", "Maximum number of points in a leaf node.", "c", 32);
PARAM_INT("num_x_regions", "Number of regions to divide the input into along the x coordinate.",
          "x", 1);
PARAM_INT("num_y_regions", "Number of regions to divide the input into along the y coordinate.",
          "y", 1);
PARAM_INT("num_z_regions", "Number of regions to divide the input into along the z coordinate.",
          "z", 1);
PARAM_INT_REQ("tuple_size", "The order of the correlation to compute (n).", "n");
//PARAM_INT("num_threads", "Total number of threads to use.  Leaving it as 0 will cause the system to use the default number.  This has no effect for the single thread code.", "T", -1);
PARAM_STRING("resampler", "Select the type of resampling to use.  Options are \"naive\" or \"efficient\".", "s", "efficient");
PARAM_STRING("kernel", "Select the type of base case computation.  Options are \"general\" (the old version for general n)  and \"efficient\" for the hardware optimized version.", "k", "efficient");

PARAM_STRING("matcher", "Select how to handle the multiple matchers. Options are \"single\" (multiple traversals, one for each matcher), and \"multi\" (one traversal, consider all matchers simulataneously.", "M", "multi");
PARAM_STRING_REQ("matchers", "A 3 column, (n choose 2) row csv, row i is r_min, r_max, num_r for dimension i.", "m");

PARAM_FLAG("do_off_diagonal",
           "For a multi-matcher, do we care about the off diagonal entries, or do we only want to do equilateral shapes. Currently only works for efficient angle matcher.",
           "o");


using namespace mlpack;
using namespace npoint_mlpack;

int main(int argc, char* argv[])
{
  
  // Command Line Parsing and setup
  CLI::ParseCommandLine(argc, argv);
  
  omp_set_nested(1);
  /*
   int num_threads = CLI::GetParam<int>("num_threads");
   if (num_threads > 0) {
   omp_set_num_threads(num_threads);
   }
   */
  omp_set_num_threads(1);
  
  //////////////////////////////////////////////////////////////////////.
  // Resampler info
  // Tree info
  int leaf_size = CLI::GetParam<int>("leaf_size");
  
  // Resampler info
  //double box_x_length = CLI::GetParam<double>("box_x_length");
  //double box_y_length = CLI::GetParam<double>("box_y_length");
  //double box_z_length = CLI::GetParam<double>("box_z_length");
  
  int num_x_regions = CLI::GetParam<int>("num_x_regions");
  int num_y_regions = CLI::GetParam<int>("num_y_regions");
  int num_z_regions = CLI::GetParam<int>("num_z_regions");
  
  ///////////////////////////////////////////////////////////////////////
  
  // Loading the data and randoms
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
  
  ResamplingHelper helper(data_mat);
  
  // We don't do weights yet.
  arma::colvec weights;
  weights.set_size(data_mat.n_cols);
  weights.fill(1.0);
  
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
  
  /////////////////////////////////////////////////////////////////////
  
  // Reading in matcher info
  std::string matcher_filename = CLI::GetParam<std::string>("matchers");
  arma::mat matcher_mat;
  matcher_mat.load(matcher_filename, arma::raw_ascii);

  int tuple_size = CLI::GetParam<int>("tuple_size");
  
  if (tuple_size * (tuple_size - 1) / 2 != (int)matcher_mat.n_rows) {
    mlpack::Log::Fatal << "Matcher specification does not correspond to tuple size.";
  }
  
  bool do_off_diagonal = CLI::HasParam("do_off_diagonal");
  MatcherArguments matcher_args(matcher_mat, tuple_size, do_off_diagonal);
  
  ////////////////////////////////////////////////////////////////////
  // Do the algorithm
  
  MultiResults* results = new MultiResults();
  
  std::string resampler_str = CLI::GetParam<std::string>("resampler");
  std::string kernel_str = CLI::GetParam<std::string>("kernel");
  std::string matcher_str = CLI::GetParam<std::string>("matcher");
  
  if (CLI::GetParam<std::string>("resampler") == "efficient")
  {
    // efficient resampling
    if (CLI::GetParam<std::string>("kernel") == "efficient")
    {
      // efficient kernel
      if (CLI::GetParam<std::string>("matcher") == "multi") {
        
        Log::Info << "Doing Efficient Resampling, Efficient Kernel, Multi Matcher computation.\n";
        
        EfficientResamplingDriver<EfficientMultiMatcher,
        GenericNptAlg<EfficientMultiMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      }
      else {
        // single matcher
        
        Log::Info << "Doing Efficient Resampling, Efficient Kernel, Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        EfficientResamplingDriver<EfficientCpuMatcher,
        GenericNptAlg<EfficientCpuMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      } // single matcher
      
    } // efficient kernel
    else {
      // general
      if (CLI::GetParam<std::string>("matcher") == "multi") {
        
        Log::Info << "Doing Efficient Resampling, General Kernel, Multi Matcher computation.\n";
        
        EfficientResamplingDriver<MultiMatcher,
        GenericNptAlg<MultiMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      }
      else {
        // single matcher
        
        Log::Info << "Doing Efficient Resampling, General Kernel, Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        EfficientResamplingDriver<SingleMatcher,
        GenericNptAlg<SingleMatcher>,
        NptNode, MultiResults>
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
        *results = driver.results();
        
      } // single matcher
      
    } // simple kernel
    
  } // efficient resampling
  else {
    // doing naive
    
    // generic multi tree traversal
    if (CLI::GetParam<std::string>("kernel") == "efficient")
    {
      // efficient kernel
      if (CLI::GetParam<std::string>("matcher") == "multi") {
        
        Log::Info << "Doing Naive Resampling, Efficient Kernel, Multi Matcher computation.\n";
        
        NaiveResamplingDriver<EfficientMultiMatcher,
        GenericNptAlg<EfficientMultiMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      }
      else {
        // single matcher
        
        Log::Info << "Doing Naive Resampling, Efficient Kernel, Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        NaiveResamplingDriver<EfficientCpuMatcher,
        GenericNptAlg<EfficientCpuMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      } // single matcher
      
    } // efficient kernel
    else {
      // general
      if (CLI::GetParam<std::string>("matcher") == "multi") {
        
        Log::Info << "Doing Naive Resampling, General Kernel, Multi Matcher computation.\n";
        
        NaiveResamplingDriver<MultiMatcher,
        GenericNptAlg<MultiMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      }
      else {
        // single matcher
        
        Log::Info << "Doing Naive Resampling, General Kernel, Single Matcher computation.\n";
        
        matcher_args.set_template_type(MatcherArguments::TEMPLATE_SINGLE_MATCHER);
        
        NaiveResamplingDriver<SingleMatcher,
        GenericNptAlg<SingleMatcher>,
        NptNode, MultiResults>
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
        
        *results = driver.results();
        
      } // single matcher
      
    } // simple kernel
    
  } // naive resampling
  
  //////////////////////////////////////////////////////////////////
  // Process / output results
  
  results->PrintResults();
  
  
  return 0;
  
} // main


